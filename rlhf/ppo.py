"""PPO for RLHF, with the implementation details that actually matter.

The ones that are in code and not in the paper, all present here:

  token level KL penalty   the reward is r_RM at the final token minus
                           beta * (logp_policy - logp_ref) at every token. Not a
                           single sequence level term, because credit needs to
                           land where the divergence happened.
  advantage whitening      normalise advantages per batch. Without it the update
                           size depends on the arbitrary scale of the reward
                           model, which is only identified up to a constant.
  ratio clipping           the PPO surrogate, on per token ratios.
  value clipping           the same trick on the value head, which stops a large
                           value error from dominating early updates.
  multiple epochs          reuse each rollout a few times, which is where the
                           importance ratio stops being 1 and clipping starts
                           doing something.
  GAE                      generalised advantage estimation over the token
                           sequence, treating the final reward as terminal.

The KL penalty is not a regulariser bolted on for stability. It is the only
thing standing between the optimiser and the reward model's blind spots, and the
whole overoptimization experiment is a sweep over how much of it you apply.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import torch

from .model import TinyLM, token_logprobs


@dataclass
class PPOConfig:
    steps: int = 60
    batch: int = 256
    length: int = 24
    lr: float = 3e-4
    epochs: int = 4
    minibatch: int = 64
    clip: float = 0.2
    vf_coef: float = 0.5
    vf_clip: float = 0.2
    ent_coef: float = 0.0
    gamma: float = 1.0
    lam: float = 0.95
    kl_beta: float = 0.05
    whiten: bool = True
    seed: int = 0
    history: list = field(default_factory=list)


def gae(rewards, values, gamma, lam):
    """Advantages and returns over a token sequence with a terminal reward."""
    T = rewards.shape[1]
    adv = torch.zeros_like(rewards)
    last = torch.zeros_like(rewards[:, 0])
    for t in reversed(range(T)):
        next_v = values[:, t + 1] if t + 1 < T else torch.zeros_like(values[:, 0])
        delta = rewards[:, t] + gamma * next_v - values[:, t]
        last = delta + gamma * lam * last
        adv[:, t] = last
    return adv, adv + values


def ppo_train(policy: TinyLM, reward_fn, cfg: PPOConfig, ref: TinyLM | None = None):
    """Optimise `policy` against `reward_fn`, kept near `ref` by a KL penalty."""
    torch.manual_seed(cfg.seed)
    # Copy before freezing: mutating the caller's reference would leak
    # requires_grad=False into every policy later deep-copied from it.
    ref = copy.deepcopy(ref) if ref is not None else copy.deepcopy(policy)
    for p in ref.parameters():
        p.requires_grad_(False)
    ref.eval()
    opt = torch.optim.AdamW(policy.parameters(), lr=cfg.lr)

    for step in range(cfg.steps):
        # ---- rollout
        with torch.no_grad():
            seq = policy.generate(cfg.batch, cfg.length,
                                  generator=torch.Generator().manual_seed(cfg.seed * 1000 + step))
            old_lp, old_v, _ = token_logprobs(policy, seq)
            ref_lp, _, _ = token_logprobs(ref, seq)
            scores = reward_fn(seq)                       # (batch,), terminal

            kl_tok = old_lp - ref_lp                      # per token
            rewards = -cfg.kl_beta * kl_tok
            rewards[:, -1] = rewards[:, -1] + scores
            adv, ret = gae(rewards, old_v, cfg.gamma, cfg.lam)
            if cfg.whiten:
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # ---- update
        n = seq.shape[0]
        for _ in range(cfg.epochs):
            perm = torch.randperm(n)
            for s in range(0, n, cfg.minibatch):
                i = perm[s:s + cfg.minibatch]
                lp, v, ent = token_logprobs(policy, seq[i])
                ratio = (lp - old_lp[i]).exp()
                a = adv[i]
                pg = -torch.min(ratio * a,
                                ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * a).mean()
                v_clipped = old_v[i] + (v - old_v[i]).clamp(-cfg.vf_clip, cfg.vf_clip)
                vf = 0.5 * torch.max((v - ret[i]) ** 2, (v_clipped - ret[i]) ** 2).mean()
                loss = pg + cfg.vf_coef * vf - cfg.ent_coef * ent.mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                opt.step()

        cfg.history.append({
            "step": step,
            "proxy_score": scores.mean().item(),
            "kl": kl_tok.sum(dim=1).mean().item(),
            "pg_loss": pg.item(),
            "vf_loss": vf.item(),
        })
    return policy, ref


@torch.no_grad()
def sequence_kl(policy: TinyLM, ref: TinyLM, n=1024, length=24, seed=0) -> float:
    """Mean per sequence KL(policy || ref), estimated on policy samples."""
    seq = policy.generate(n, length, generator=torch.Generator().manual_seed(seed))
    lp, _, _ = token_logprobs(policy, seq)
    rp, _, _ = token_logprobs(ref, seq)
    return (lp - rp).sum(dim=1).mean().item()
