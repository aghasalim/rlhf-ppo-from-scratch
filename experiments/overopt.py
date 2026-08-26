"""Goodhart's law, with a plot.

Optimise the proxy reward model and watch the gold reward. Near the reference
policy the two agree and both rise. As the policy drifts the proxy stays
confident in a region it was never trained on, and the gold turns over while the
proxy keeps climbing.

The x axis is sqrt(KL) rather than step count, following Gao, Schulman and
Hilton (2022), because KL is what actually determines how far off distribution
you are and their fitted forms are linear-ish in it. Step count would confound
learning rate with drift.

Each KL penalty coefficient traces a different distance along the same curve, so
sweeping beta and recording every step gives the whole thing rather than one
point per run.

    .venv/bin/python -m experiments.overopt
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import time
import warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")

from rlhf.alternatives import best_of_n, dpo_train, fresh, grpo_train, rloo_train
from rlhf.gold import VOCAB, gold_components, gold_reward
from rlhf.model import TinyLM
from rlhf.ppo import PPOConfig, ppo_train, sequence_kl
from rlhf.reward_model import (
    RewardModel,
    agreement_with_gold,
    build_preferences,
    train_reward_model,
)
from rlhf.sft import train_sft

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
EVAL_N, LENGTH = 1024, 24


@torch.no_grad()
def snapshot(policy, ref, rm, seed=99):
    s = policy.generate(EVAL_N, LENGTH, generator=torch.Generator().manual_seed(seed))
    comp = gold_components(s)
    return {
        "kl": sequence_kl(policy, ref, n=EVAL_N, length=LENGTH, seed=seed),
        "proxy": rm(s).mean().item(),
        "gold": gold_reward(s).mean().item(),
        "motif": comp["motif"].mean().item(),
        "repeats": comp["repeats"].mean().item(),
        "hoard": comp["hoard"].mean().item(),
        "rm_agreement": agreement_with_gold(rm, s),
    }


def setup(seed: int, n_pairs: int, rm_steps: int):
    torch.manual_seed(seed)
    sft = TinyLM(VOCAB)
    train_sft(sft, steps=1200, seed=seed)
    a, b, w = build_preferences(sft, n_pairs, LENGTH, seed=seed + 2)
    rm = RewardModel(VOCAB)
    train_reward_model(rm, a, b, w, steps=rm_steps, seed=seed)
    rm.eval()
    for p in rm.parameters():
        p.requires_grad_(False)
    return sft, rm, (a, b, w)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--betas", nargs="+", type=float, default=[0.2, 0.05, 0.01, 0.0])
    ap.add_argument("--ppo-steps", type=int, default=70)
    ap.add_argument("--pairs", type=int, default=4000)
    ap.add_argument("--rm-steps", type=int, default=600)
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    curve, table = [], []
    started = time.perf_counter()

    for seed in args.seeds:
        print(f"seed {seed}: training SFT and reward model")
        sft, rm, pairs = setup(seed, args.pairs, args.rm_steps)
        ref = copy.deepcopy(sft)

        # rm bound as a default rather than closed over: the loop rebuilds a new
        # reward model each seed, and a closure would make every earlier fn point
        # at the latest one.
        def fn(s, rm=rm):
            return rm(s)

        base = snapshot(ref, ref, rm, seed=99)
        table.append({"seed": seed, "method": "SFT (reference)", "kl": 0.0,
                      "proxy": base["proxy"], "gold": base["gold"],
                      "motif": base["motif"], "repeats": base["repeats"],
                      "hoard": base["hoard"], "wall_s": 0.0})
        print(f"  reference: proxy {base['proxy']:+.3f}  gold {base['gold']:+.3f}  "
              f"rm agreement {base['rm_agreement']:.3f}")

        # ---- the sweep: each beta traces a different distance along the curve
        for beta in args.betas:
            pol = fresh(ref)
            cfg = PPOConfig(steps=args.ppo_steps, kl_beta=beta, seed=seed)
            t0 = time.perf_counter()
            for step in range(cfg.steps):
                one = PPOConfig(steps=1, kl_beta=beta, seed=seed * 100 + step)
                ppo_train(pol, fn, one, ref=ref)
                if step % 5 == 0 or step == cfg.steps - 1:
                    snap = snapshot(pol, ref, rm)
                    curve.append({"seed": seed, "method": "PPO", "beta": beta,
                                  "step": step, **snap})
            wall = time.perf_counter() - t0
            final = snapshot(pol, ref, rm)
            table.append({"seed": seed, "method": f"PPO (beta={beta})", **{
                k: final[k] for k in ("kl", "proxy", "gold", "motif", "repeats", "hoard")},
                "wall_s": wall})
            print(f"  PPO beta={beta:<5} KL {final['kl']:7.2f}  proxy {final['proxy']:+7.3f}  "
                  f"gold {final['gold']:+7.3f}  motif {final['motif']:.2f}  "
                  f"hoard {final['hoard']:.2f}  {wall:.0f}s")

        # ---- alternatives, same reward model
        for n in (4, 16, 64):
            t0 = time.perf_counter()
            s, kl = best_of_n(ref, fn, n, EVAL_N, LENGTH, seed=seed + 3)
            comp = gold_components(s)
            table.append({"seed": seed, "method": f"Best-of-{n}", "kl": kl,
                          "proxy": rm(s).mean().item(), "gold": gold_reward(s).mean().item(),
                          "motif": comp["motif"].mean().item(),
                          "repeats": comp["repeats"].mean().item(),
                          "hoard": comp["hoard"].mean().item(),
                          "wall_s": time.perf_counter() - t0})

        a, b, w = pairs
        for name, run in (("DPO", None), ("RLOO", rloo_train), ("GRPO", grpo_train)):
            pol = fresh(ref)
            t0 = time.perf_counter()
            if name == "DPO":
                dpo_train(pol, copy.deepcopy(ref), a, b, w, steps=400, seed=seed)
            else:
                run(pol, copy.deepcopy(ref), fn, steps=40, kl_beta=0.02, seed=seed)
            snap = snapshot(pol, ref, rm)
            table.append({"seed": seed, "method": name, **{
                k: snap[k] for k in ("kl", "proxy", "gold", "motif", "repeats", "hoard")},
                "wall_s": time.perf_counter() - t0})
            print(f"  {name:9} KL {snap['kl']:7.2f}  proxy {snap['proxy']:+7.3f}  "
                  f"gold {snap['gold']:+7.3f}")

    for fname, rows in (("overopt-curve.csv", curve), ("methods.csv", table)):
        p = RESULTS / fname
        with p.open("w", newline="") as fh:
            wtr = csv.DictWriter(fh, fieldnames=sorted({k for r in rows for k in r}))
            wtr.writeheader()
            wtr.writerows(rows)
        print(f"wrote {p.relative_to(ROOT)} ({len(rows)} rows)")
    (RESULTS / "run-meta.json").write_text(json.dumps({
        **vars(args), "wall_clock_s": time.perf_counter() - started,
        "torch": torch.__version__, "device": "cpu"}, indent=1))
    print(f"total {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
