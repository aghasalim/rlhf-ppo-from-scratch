"""A small autoregressive transformer, plus a value head for PPO.

Tiny on purpose. The object of study is what happens to an optimiser pointed at
an imperfect reward, and that reproduces at any scale.
"""
from __future__ import annotations

import torch
from torch import nn


class TinyLM(nn.Module):
    def __init__(self, vocab: int, d_model: int = 64, n_layers: int = 2,
                 n_heads: int = 4, max_len: int = 32):
        super().__init__()
        self.vocab, self.max_len = vocab, max_len
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model,
                                           dropout=0.0, batch_first=True,
                                           norm_first=True, activation="gelu")
        # enable_nested_tensor=False: it is incompatible with norm_first and torch
        # only warns rather than erroring, once per construction, which is noise
        # in a sweep that builds hundreds of models.
        self.body = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.value = nn.Linear(d_model, 1)

    def _hidden(self, idx):
        s = idx.shape[1]
        h = self.tok(idx) + self.pos(torch.arange(s, device=idx.device))[None]
        mask = torch.triu(torch.ones(s, s, dtype=torch.bool, device=idx.device), 1)
        return self.norm(self.body(h, mask=mask, is_causal=True))

    def forward(self, idx):
        h = self._hidden(idx)
        return self.head(h), self.value(h).squeeze(-1)

    def logits(self, idx):
        return self.head(self._hidden(idx))

    @torch.no_grad()
    def generate(self, n: int, length: int, generator=None, temperature: float = 1.0,
                 prefix: int = 0):
        """Sample n sequences of `length` tokens, starting from token 0."""
        idx = torch.zeros(n, 1, dtype=torch.long)
        for _ in range(length - 1):
            logits = self.logits(idx[:, -self.max_len:])[:, -1] / temperature
            probs = logits.softmax(-1)
            nxt = torch.multinomial(probs, 1, generator=generator)
            idx = torch.cat([idx, nxt], dim=1)
        return idx


def sequence_logprob(model: TinyLM, seq: torch.Tensor) -> torch.Tensor:
    """Sum of log p(token_t | token_<t) over t >= 1, shape (batch,)."""
    logits = model.logits(seq)[:, :-1]
    lp = logits.log_softmax(-1)
    return lp.gather(-1, seq[:, 1:].unsqueeze(-1)).squeeze(-1).sum(dim=1)


def token_logprobs(model: TinyLM, seq: torch.Tensor):
    """Per token log probs and values, both shape (batch, length-1)."""
    logits, values = model(seq)
    lp = logits[:, :-1].log_softmax(-1)
    chosen = lp.gather(-1, seq[:, 1:].unsqueeze(-1)).squeeze(-1)
    entropy = -(lp * lp.exp()).sum(-1)
    return chosen, values[:, :-1], entropy
