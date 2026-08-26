"""The reference policy.

A base distribution the policy starts from and is kept near. Trained to imitate
a fixed Markov chain over the vocabulary, which gives it structure without any
knowledge of the gold reward.
"""
from __future__ import annotations

import torch

from .gold import VOCAB
from .model import TinyLM


def base_sequences(n: int, length: int, seed: int = 0) -> torch.Tensor:
    """Samples from a fixed random Markov chain, sharpened so it is not uniform."""
    g = torch.Generator().manual_seed(1234)
    trans = (torch.randn(VOCAB, VOCAB, generator=g) * 1.5).softmax(-1)
    gs = torch.Generator().manual_seed(seed)
    out = torch.zeros(n, length, dtype=torch.long)
    for t in range(1, length):
        out[:, t] = torch.multinomial(trans[out[:, t - 1]], 1, generator=gs).squeeze(1)
    return out


def train_sft(model: TinyLM, steps=1200, batch=128, length=24, lr=3e-3, seed=0,
              quiet=True):
    torch.manual_seed(seed)
    data = base_sequences(20_000, length, seed=seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    history = []
    for step in range(steps):
        i = torch.randint(0, data.shape[0], (batch,))
        seq = data[i]
        logits = model.logits(seq)[:, :-1]
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), seq[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 300 == 0 or step == steps - 1:
            history.append({"step": step, "loss": loss.item()})
            if not quiet:
                print(f"      sft step {step:4d} loss {loss.item():.4f}")
    return history
