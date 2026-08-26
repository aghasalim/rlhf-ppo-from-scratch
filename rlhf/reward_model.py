"""The proxy reward model: a Bradley-Terry fit to preference comparisons.

Given a pair (a, b) with a preferred, the Bradley-Terry model says

    P(a > b) = sigmoid(r(a) - r(b))

so the loss is -log sigmoid(r(a) - r(b)). Only differences are identified, which
is why the absolute scale of a reward model is meaningless and only its ordering
matters. This is also why a KL penalty rather than a reward threshold is what
keeps PPO honest.

The important design choice: **comparisons are drawn only from the reference
policy's own samples.** That is the real situation. You collect preferences on
outputs your current model produces, so your reward model is accurate on that
distribution and increasingly uninformed as the policy moves away from it. The
overoptimization experiment is entirely about what happens in that second
region.
"""
from __future__ import annotations

import torch
from torch import nn

from .gold import gold_reward
from .model import TinyLM


class RewardModel(nn.Module):
    """Same trunk shape as the policy, one scalar head on the final token."""

    def __init__(self, vocab: int, d_model: int = 64, n_layers: int = 2,
                 n_heads: int = 4, max_len: int = 32):
        super().__init__()
        self.trunk = TinyLM(vocab, d_model, n_layers, n_heads, max_len)
        self.score = nn.Linear(d_model, 1)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        h = self.trunk._hidden(seq)
        return self.score(h[:, -1]).squeeze(-1)


def build_preferences(policy: TinyLM, n_pairs: int, length: int, seed: int = 0,
                      noise: float = 0.0):
    """Sample pairs from `policy` and label them with the gold reward.

    `noise` flips a fraction of labels, standing in for human disagreement. Even
    at 0 the proxy is imperfect, because it only ever sees this distribution.
    """
    g = torch.Generator().manual_seed(seed)
    a = policy.generate(n_pairs, length, generator=g)
    b = policy.generate(n_pairs, length, generator=g)
    ra, rb = gold_reward(a), gold_reward(b)
    a_wins = ra > rb
    if noise > 0:
        flip = torch.rand(n_pairs, generator=g) < noise
        a_wins = a_wins ^ flip
    ties = ra == rb
    return a[~ties], b[~ties], a_wins[~ties]


def train_reward_model(rm: RewardModel, a, b, a_wins, steps=800, batch=128,
                       lr=1e-3, seed=0, log_every=200, quiet=True):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(rm.parameters(), lr=lr, weight_decay=0.01)
    n = a.shape[0]
    history = []
    for step in range(steps):
        i = torch.randint(0, n, (min(batch, n),))
        win = torch.where(a_wins[i].unsqueeze(1), a[i], b[i])
        lose = torch.where(a_wins[i].unsqueeze(1), b[i], a[i])
        margin = rm(win) - rm(lose)
        loss = -nn.functional.logsigmoid(margin).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(rm.parameters(), 1.0)
        opt.step()
        if step % log_every == 0 or step == steps - 1:
            acc = (margin > 0).float().mean().item()
            history.append({"step": step, "loss": loss.item(), "train_acc": acc})
            if not quiet:
                print(f"      rm step {step:4d} loss {loss.item():.4f} acc {acc:.3f}")
    return history


@torch.no_grad()
def agreement_with_gold(rm: RewardModel, seqs: torch.Tensor) -> float:
    """Fraction of random pairs the proxy orders the same way gold does.

    Evaluated on whatever distribution `seqs` came from, which is the point:
    this number is high near the reference policy and falls off it.
    """
    n = seqs.shape[0] // 2
    a, b = seqs[:n], seqs[n:2 * n]
    pg = gold_reward(a) > gold_reward(b)
    pp = rm(a) > rm(b)
    keep = gold_reward(a) != gold_reward(b)
    if keep.sum() == 0:
        return float("nan")
    return (pg[keep] == pp[keep]).float().mean().item()
