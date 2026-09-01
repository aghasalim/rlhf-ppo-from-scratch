"""Write golden reference vectors for the PPO kernels in verify/golden/.

The C and Rust checks read these files and have to reproduce the outputs from
the inputs. Everything here comes from the repository's own functions in
rlhf/ppo.py, so a bug in that file lands in the golden outputs and the other
implementations will disagree with it.

Everything is float64 so that a C double and a Rust f64 can be held to 1e-12.
The training loop itself runs in float32; what is being checked is the
arithmetic of the kernels, not the precision the sweep happened to use.

Regenerate with:
    python verify/export_golden.py

An optional argument writes the files somewhere else instead, which is how
verify/verify.sh checks that the committed vectors still match rlhf/ppo.py.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rlhf.ppo import clipped_losses, gae, shape_rewards

OUT = Path(__file__).resolve().parent / "golden"
FMT = "%.17g"


def f(x) -> str:
    return FMT % float(x)


def gae_cases(g: torch.Generator) -> list[dict]:
    """Cases chosen to exercise the corners, not just the average path.

    gamma=1 lam=1 is the reward-to-go limit, lam=0 is the one step TD limit,
    and a long sequence with a large terminal score is what the sweep actually
    produces.
    """
    specs = [
        (1.0, 0.95, 0.05, 24),
        (1.0, 1.00, 0.20, 24),
        (1.0, 0.00, 0.01, 16),
        (0.99, 0.95, 0.00, 32),
        (0.90, 0.50, 0.20, 8),
        (1.0, 0.95, 0.05, 1),
        (1.0, 0.95, 2.00, 12),
        (0.5, 0.95, 0.05, 12),
    ]
    rows = []
    for case, (gamma, lam, beta, T) in enumerate(specs):
        kl_tok = torch.randn(1, T, generator=g, dtype=torch.float64) * 0.3
        values = torch.randn(1, T, generator=g, dtype=torch.float64) * 2.0
        score = torch.randn(1, generator=g, dtype=torch.float64) * 5.0
        rewards = shape_rewards(kl_tok.clone(), score, beta)
        adv, ret = gae(rewards, values, gamma, lam)
        for t in range(T):
            rows.append({
                "case": case, "gamma": f(gamma), "lam": f(lam), "kl_beta": f(beta),
                "score": f(score[0]), "t": t, "kl_tok": f(kl_tok[0, t]),
                "reward": f(rewards[0, t]), "value": f(values[0, t]),
                "adv": f(adv[0, t]), "ret": f(ret[0, t]),
            })
    return rows


def loss_cases(g: torch.Generator) -> tuple[list[dict], list[dict]]:
    """Minibatches sized so that some ratios land outside the clip range.

    A case where nothing clips would pass against an implementation that
    forgot the clip entirely, so the log ratio spread is deliberately wide.
    """
    specs = [(0.2, 0.2, 64, 0.05), (0.2, 0.2, 64, 0.60), (0.1, 0.4, 32, 0.35),
             (0.3, 0.1, 48, 0.90), (0.2, 0.2, 8, 1.50)]
    inputs, scalars = [], []
    for case, (clip, vf_clip, n, spread) in enumerate(specs):
        old_lp = torch.randn(n, generator=g, dtype=torch.float64) * 0.5 - 2.0
        lp = old_lp + torch.randn(n, generator=g, dtype=torch.float64) * spread
        adv = torch.randn(n, generator=g, dtype=torch.float64)
        old_v = torch.randn(n, generator=g, dtype=torch.float64) * 2.0
        v = old_v + torch.randn(n, generator=g, dtype=torch.float64) * 0.5
        ret = torch.randn(n, generator=g, dtype=torch.float64) * 2.0
        pg, vf = clipped_losses(lp, old_lp, adv, v, old_v, ret, clip, vf_clip)
        for i in range(n):
            inputs.append({"case": case, "i": i, "lp": f(lp[i]), "old_lp": f(old_lp[i]),
                           "adv": f(adv[i]), "v": f(v[i]), "old_v": f(old_v[i]),
                           "ret": f(ret[i])})
        clipped = ((lp - old_lp).exp() < 1 - clip) | ((lp - old_lp).exp() > 1 + clip)
        scalars.append({"case": case, "clip": f(clip), "vf_clip": f(vf_clip), "n": n,
                        "frac_ratio_clipped": f(clipped.double().mean()),
                        "pg_loss": f(pg), "vf_loss": f(vf)})
    return inputs, scalars


def write(name: str, rows: list[dict]) -> None:
    path = OUT / name
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}  {len(rows)} rows")


def main() -> None:
    global OUT
    if len(sys.argv) > 1:
        OUT = Path(sys.argv[1])
    OUT.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(20260901)
    write("gae.csv", gae_cases(g))
    inputs, scalars = loss_cases(g)
    write("ppo_inputs.csv", inputs)
    write("ppo_loss.csv", scalars)


if __name__ == "__main__":
    main()
