"""Best-of-N, DPO, GRPO and RLOO. The things people reach for instead of PPO.

All four optimise the same proxy reward, so all four are exposed to the same
blind spots. The interesting question is not which scores highest on the proxy,
it is which reaches a given gold score for the least KL.

  Best-of-N   no training at all. Sample N, keep the best by proxy. Its KL from
              the reference is known in closed form, log N - (N-1)/N, which makes
              it a useful calibration point on the same axis as everything else.
  DPO         skips the reward model and fits the policy directly to preference
              pairs through the Bradley-Terry likelihood. Here it is fed the same
              pairs the reward model saw, so the comparison is fair.
  RLOO        REINFORCE with a leave-one-out baseline over k samples per prompt.
              No value network, no GAE.
  GRPO        the same idea with the group mean and standard deviation as the
              baseline, which is what makes it popular for reasoning models.
"""
from __future__ import annotations

import copy
import math

import torch
from torch import nn

from .model import TinyLM, sequence_logprob, token_logprobs


@torch.no_grad()
def best_of_n(policy: TinyLM, reward_fn, n_samples: int, n: int, length: int, seed=0):
    """Sample n*N sequences, return the best of each group of N by proxy."""
    g = torch.Generator().manual_seed(seed)
    seq = policy.generate(n * n_samples, length, generator=g)
    scores = reward_fn(seq).view(n, n_samples)
    best = scores.argmax(dim=1)
    picked = seq.view(n, n_samples, -1)[torch.arange(n), best]
    return picked, analytic_bon_kl(n_samples)


def analytic_bon_kl(n_samples: int) -> float:
    """KL of the best-of-N distribution from the reference, exact for continuous
    reward: log N - (N-1)/N."""
    return math.log(n_samples) - (n_samples - 1) / n_samples


def dpo_train(policy: TinyLM, ref: TinyLM, a, b, a_wins, steps=400, batch=64,
              lr=3e-4, beta=0.1, seed=0):
    """Direct preference optimisation on the same pairs the RM was fit to."""
    torch.manual_seed(seed)
    for p in ref.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(policy.parameters(), lr=lr)
    n = a.shape[0]
    hist = []
    for step in range(steps):
        i = torch.randint(0, n, (min(batch, n),))
        win = torch.where(a_wins[i].unsqueeze(1), a[i], b[i])
        lose = torch.where(a_wins[i].unsqueeze(1), b[i], a[i])
        with torch.no_grad():
            ref_w, ref_l = sequence_logprob(ref, win), sequence_logprob(ref, lose)
        pol_w, pol_l = sequence_logprob(policy, win), sequence_logprob(policy, lose)
        logits = beta * ((pol_w - ref_w) - (pol_l - ref_l))
        loss = -nn.functional.logsigmoid(logits).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        if step % 100 == 0 or step == steps - 1:
            hist.append({"step": step, "loss": loss.item(),
                         "acc": (logits > 0).float().mean().item()})
    return hist


def _group_policy_gradient(policy, ref, reward_fn, steps, k, batch, length, lr,
                           kl_beta, seed, mode):
    """Shared body for RLOO and GRPO. They differ only in the baseline."""
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(policy.parameters(), lr=lr)
    hist = []
    for step in range(steps):
        with torch.no_grad():
            seq = policy.generate(batch * k, length,
                                  generator=torch.Generator().manual_seed(seed * 977 + step))
            r = reward_fn(seq).view(batch, k)
            lp_p, _, _ = token_logprobs(policy, seq)
            lp_r, _, _ = token_logprobs(ref, seq)
            kl = (lp_p - lp_r).sum(dim=1).view(batch, k)
            r = r - kl_beta * kl
            if mode == "rloo":
                # leave one out: baseline for i is the mean of the other k-1
                baseline = (r.sum(dim=1, keepdim=True) - r) / (k - 1)
                adv = r - baseline
            else:  # grpo
                adv = (r - r.mean(dim=1, keepdim=True)) / (r.std(dim=1, keepdim=True) + 1e-8)
            adv = adv.reshape(-1)
        lp = sequence_logprob(policy, seq)
        loss = -(adv * lp).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        if step % 10 == 0 or step == steps - 1:
            hist.append({"step": step, "proxy": r.mean().item(), "kl": kl.mean().item()})
    return hist


def rloo_train(policy, ref, reward_fn, steps=60, k=4, batch=64, length=24,
               lr=3e-4, kl_beta=0.05, seed=0):
    return _group_policy_gradient(policy, ref, reward_fn, steps, k, batch, length,
                                  lr, kl_beta, seed, "rloo")


def grpo_train(policy, ref, reward_fn, steps=60, k=4, batch=64, length=24,
               lr=3e-4, kl_beta=0.05, seed=0):
    return _group_policy_gradient(policy, ref, reward_fn, steps, k, batch, length,
                                  lr, kl_beta, seed, "grpo")


def fresh(policy: TinyLM) -> TinyLM:
    """A trainable copy of `policy`.

    The requires_grad_(True) is load bearing. `ppo_train` freezes the reference
    policy it is handed, and every subsequent policy in a sweep is a deepcopy of
    that same reference, so without this the second run onward starts from
    frozen parameters. It does not raise where the mistake is made: the optimiser
    simply has nothing to update and the failure surfaces later as
    "element 0 of tensors does not require grad" from the backward call.
    """
    out = copy.deepcopy(policy)
    for p in out.parameters():
        p.requires_grad_(True)
    return out
