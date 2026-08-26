"""Tests. Several of these exist because the corresponding bug actually happened."""
import math

import pytest
import torch

from rlhf.alternatives import analytic_bon_kl, best_of_n, fresh
from rlhf.gold import VOCAB, count_motif, count_repeats, gold_reward
from rlhf.model import TinyLM, sequence_logprob, token_logprobs
from rlhf.ppo import PPOConfig, gae, ppo_train, sequence_kl
from rlhf.reward_model import RewardModel, build_preferences, train_reward_model
from rlhf.sft import base_sequences


# --- gold reward ------------------------------------------------------------
def test_motif_counting():
    seq = torch.tensor([[3, 7, 3, 7, 0], [0, 1, 2, 3, 7], [1, 1, 1, 1, 1]])
    assert count_motif(seq).tolist() == [2.0, 1.0, 0.0]


def test_repeat_counting():
    seq = torch.tensor([[1, 1, 1, 2, 3], [0, 1, 2, 3, 4]])
    assert count_repeats(seq).tolist() == [2.0, 0.0]


def test_hoarding_penalty_activates_only_above_threshold():
    """The hidden term. Below the threshold more motifs help; above, they hurt."""
    below = torch.tensor([[3, 7, 0, 3, 7, 0, 3, 7, 0]])          # 3 motifs
    above = torch.tensor([[3, 7, 0, 3, 7, 0, 3, 7, 0, 3, 7, 0, 3, 7]])   # 5 motifs
    r_below, r_above = gold_reward(below).item(), gold_reward(above).item()
    assert count_motif(below).item() == 3 and count_motif(above).item() == 5
    # two extra motifs past the threshold are a net loss
    assert r_above < r_below + 2.0


def test_gold_prefers_motifs_over_nothing():
    good = torch.tensor([[3, 7, 0, 1, 2, 4, 5, 6]])
    bland = torch.tensor([[0, 1, 2, 4, 5, 6, 8, 9]])
    assert gold_reward(good).item() > gold_reward(bland).item()


# --- model ------------------------------------------------------------------
def test_generation_shape_and_range():
    m = TinyLM(VOCAB)
    s = m.generate(8, 12, generator=torch.Generator().manual_seed(0))
    assert s.shape == (8, 12)
    assert s.min() >= 0 and s.max() < VOCAB


def test_logprob_is_negative_and_matches_token_version():
    torch.manual_seed(0)
    m = TinyLM(VOCAB)
    s = m.generate(4, 10, generator=torch.Generator().manual_seed(0))
    total = sequence_logprob(m, s)
    per_token, _, _ = token_logprobs(m, s)
    assert (total < 0).all()
    assert torch.allclose(total, per_token.sum(dim=1), atol=1e-4)


def test_causal_masking():
    """Changing a later token must not change earlier logits."""
    torch.manual_seed(0)
    m = TinyLM(VOCAB).eval()
    a = torch.randint(0, VOCAB, (1, 12))
    b = a.clone()
    b[0, 8:] = torch.randint(0, VOCAB, (4,))
    with torch.no_grad():
        assert torch.allclose(m.logits(a)[:, :8], m.logits(b)[:, :8], atol=1e-5)


# --- GAE --------------------------------------------------------------------
def test_gae_with_terminal_reward_and_no_discount():
    """gamma=1, lam=1 and zero values makes the advantage the reward-to-go."""
    rewards = torch.zeros(1, 4)
    rewards[0, -1] = 1.0
    adv, ret = gae(rewards, torch.zeros(1, 4), gamma=1.0, lam=1.0)
    assert torch.allclose(adv, torch.ones(1, 4), atol=1e-6)
    assert torch.allclose(ret, torch.ones(1, 4), atol=1e-6)


def test_gae_zero_when_values_are_perfect():
    values = torch.tensor([[1.0, 1.0, 1.0]])
    rewards = torch.tensor([[0.0, 0.0, 1.0]])
    adv, _ = gae(rewards, values, gamma=1.0, lam=1.0)
    assert adv.abs().max() < 1e-5


# --- reward model -----------------------------------------------------------
def test_preferences_have_no_ties_and_match_gold():
    torch.manual_seed(0)
    p = TinyLM(VOCAB)
    a, b, w = build_preferences(p, 400, 16, seed=0)
    assert a.shape[0] == b.shape[0] == w.shape[0]
    assert (gold_reward(a) != gold_reward(b)).all()
    assert torch.equal(w, gold_reward(a) > gold_reward(b))


def test_reward_model_learns_the_ordering():
    torch.manual_seed(0)
    p = TinyLM(VOCAB)
    a, b, w = build_preferences(p, 1200, 16, seed=0)
    rm = RewardModel(VOCAB)
    h = train_reward_model(rm, a, b, w, steps=300)
    assert h[-1]["train_acc"] > 0.75
    assert h[-1]["loss"] < h[0]["loss"]


def test_label_noise_degrades_agreement():
    torch.manual_seed(0)
    p = TinyLM(VOCAB)
    _, _, clean = build_preferences(p, 600, 16, seed=0, noise=0.0)
    _, _, noisy = build_preferences(p, 600, 16, seed=0, noise=0.4)
    assert (clean != noisy).float().mean() > 0.2


# --- best of N --------------------------------------------------------------
@pytest.mark.parametrize("n", [2, 4, 16, 64])
def test_analytic_bon_kl_matches_formula(n):
    assert abs(analytic_bon_kl(n) - (math.log(n) - (n - 1) / n)) < 1e-12


def test_bon_kl_increases_with_n():
    vals = [analytic_bon_kl(n) for n in (2, 4, 16, 64)]
    assert vals == sorted(vals)


def test_best_of_n_picks_the_best_by_the_given_reward():
    torch.manual_seed(0)
    p = TinyLM(VOCAB)
    picked, _ = best_of_n(p, count_motif, 8, 32, 16, seed=0)
    plain = p.generate(32, 16, generator=torch.Generator().manual_seed(0))
    assert count_motif(picked).mean() >= count_motif(plain).mean()


# --- the bug that actually happened -----------------------------------------
def test_fresh_returns_a_trainable_copy():
    """ppo_train freezes the reference it is given, and later policies are
    deepcopies of that reference. Without re-enabling grad the second run in a
    sweep silently has nothing to optimise."""
    m = TinyLM(VOCAB)
    for p in m.parameters():
        p.requires_grad_(False)
    assert all(p.requires_grad for p in fresh(m).parameters())


def test_ppo_does_not_freeze_the_callers_reference():
    torch.manual_seed(0)
    pol, ref = TinyLM(VOCAB), TinyLM(VOCAB)
    before = [p.requires_grad for p in ref.parameters()]
    ppo_train(pol, lambda s: torch.zeros(s.shape[0]), PPOConfig(steps=1, batch=16, length=8), ref=ref)
    assert [p.requires_grad for p in ref.parameters()] == before


# --- ppo end to end ---------------------------------------------------------
def test_ppo_increases_the_reward_it_is_given():
    """A blunt end to end check: point PPO at motif count and it should rise."""
    torch.manual_seed(0)
    pol = TinyLM(VOCAB)
    ref = fresh(pol)
    g = torch.Generator().manual_seed(7)
    before = count_motif(pol.generate(256, 16, generator=g)).mean().item()
    cfg = PPOConfig(steps=12, batch=128, length=16, kl_beta=0.0, lr=1e-3, seed=0)
    ppo_train(pol, count_motif, cfg, ref=ref)
    after = count_motif(pol.generate(256, 16, generator=torch.Generator().manual_seed(7))).mean().item()
    assert after > before + 0.05, f"{before} -> {after}"


def test_kl_is_zero_against_self_and_positive_after_training():
    torch.manual_seed(0)
    pol = TinyLM(VOCAB)
    ref = fresh(pol)
    assert abs(sequence_kl(pol, ref, n=128, length=12)) < 1e-4
    cfg = PPOConfig(steps=8, batch=128, length=12, kl_beta=0.0, lr=1e-3, seed=0)
    ppo_train(pol, count_motif, cfg, ref=ref)
    assert sequence_kl(pol, ref, n=128, length=12) > 0.0


# --- data -------------------------------------------------------------------
def test_base_sequences_are_seeded_and_in_range():
    a = base_sequences(64, 16, seed=0)
    assert torch.equal(a, base_sequences(64, 16, seed=0))
    assert not torch.equal(a, base_sequences(64, 16, seed=1))
    assert a.min() >= 0 and a.max() < VOCAB
