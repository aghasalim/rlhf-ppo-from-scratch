"""The gold reward: the thing we actually want, and the thing we never have.

Real RLHF has no gold reward. You have human preferences, you fit a model to
them, and you optimise that model. The gap between the fitted proxy and the
true preference is where overoptimization lives, and you cannot measure it
because the true preference is not a function you can evaluate.

The standard way to study it, from Gao, Schulman and Hilton (2022), is to invent
a gold reward, treat it as ground truth, generate preference labels from it, fit
a proxy to those labels, and then optimise the proxy while watching the gold.
That makes the invisible gap visible. Everything here follows that recipe.

The gold reward on a token sequence has three terms:

  motif       + reward for each occurrence of a target bigram. Easy to learn
              from comparisons, because it fires often near the base policy.
  repetition  - penalty for immediately repeated tokens. Also easy.
  hoarding    - penalty that only activates above a threshold count of the motif.
              Near the base policy almost nothing crosses that threshold, so the
              preference data barely contains it and the proxy cannot learn what
              it never sees.

That third term is the whole design. It is not a trick to force a result: it is
the toy version of a real and common situation, where the behaviour you want to
discourage is rare in the data you collected, so your reward model is
uninformed exactly where your optimiser is about to go.
"""
from __future__ import annotations

import torch

VOCAB = 16
MOTIF = (3, 7)          # the bigram the gold reward likes
HOARD_THRESHOLD = 3     # above this many motifs, the hidden penalty switches on
W_MOTIF = 1.0
W_REPEAT = 0.6
W_HOARD = 1.4


def count_motif(seq: torch.Tensor) -> torch.Tensor:
    """Occurrences of MOTIF in each row of (batch, length)."""
    a = seq[:, :-1] == MOTIF[0]
    b = seq[:, 1:] == MOTIF[1]
    return (a & b).sum(dim=1).float()


def count_repeats(seq: torch.Tensor) -> torch.Tensor:
    return (seq[:, 1:] == seq[:, :-1]).sum(dim=1).float()


def gold_reward(seq: torch.Tensor) -> torch.Tensor:
    """The true objective. Never shown to the policy or the reward model."""
    motif = count_motif(seq)
    repeats = count_repeats(seq)
    hoard = (motif - HOARD_THRESHOLD).clamp(min=0)
    return W_MOTIF * motif - W_REPEAT * repeats - W_HOARD * hoard


def gold_components(seq: torch.Tensor) -> dict:
    """For diagnostics: which term is driving the score."""
    motif = count_motif(seq)
    return {
        "motif": motif,
        "repeats": count_repeats(seq),
        "hoard": (motif - HOARD_THRESHOLD).clamp(min=0),
    }
