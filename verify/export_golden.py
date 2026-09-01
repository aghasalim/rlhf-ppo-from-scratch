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

An optional argument writes the files somewhere else instead. With --check it
writes them to a temporary directory and compares, which is how verify.sh and
CI make sure the committed vectors still come out of rlhf/ppo.py.

The comparison is numeric to 1e-12 rather than byte for byte. torch's exp is
not guaranteed to give the same last bit on an arm64 laptop and an x86_64
runner, and one ulp is not a change to the kernel. Anything that is a change to
the kernel is orders of magnitude larger.
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


class Xorshift:
    """A deterministic generator, because torch's is not portable enough.

    The first version of this file drew its inputs with torch.randn. CI then
    disagreed with the committed vectors at the very first value: the normal
    sampler does not produce identical bits on an arm64 laptop and an x86_64
    runner. Nothing here needs a normal distribution, only fixed inputs that
    are the same everywhere, so this maps 53 integer bits onto [-1, 1), which
    every IEEE double represents exactly.
    """

    def __init__(self, seed: int):
        self.state = seed

    def next_u64(self) -> int:
        x = self.state
        x ^= (x >> 12) & 0xFFFFFFFFFFFFFFFF
        x ^= (x << 25) & 0xFFFFFFFFFFFFFFFF
        x ^= (x >> 27) & 0xFFFFFFFFFFFFFFFF
        self.state = x & 0xFFFFFFFFFFFFFFFF
        return (self.state * 0x2545F4914F6CDD1D) & 0xFFFFFFFFFFFFFFFF

    def unit(self) -> float:
        return (self.next_u64() >> 11) / float(1 << 53) * 2.0 - 1.0

    def vec(self, n: int, scale: float) -> torch.Tensor:
        return torch.tensor([self.unit() * scale for _ in range(n)],
                            dtype=torch.float64)


def f(x) -> str:
    return FMT % float(x)


def gae_cases(g: Xorshift) -> list[dict]:
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
        kl_tok = g.vec(T, 0.3).reshape(1, T)
        values = g.vec(T, 2.0).reshape(1, T)
        score = g.vec(1, 5.0)
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


def loss_cases(g: Xorshift) -> tuple[list[dict], list[dict]]:
    """Minibatches sized so that some ratios land outside the clip range.

    A case where nothing clips would pass against an implementation that
    forgot the clip entirely, so the log ratio spread is deliberately wide.
    """
    specs = [(0.2, 0.2, 64, 0.05), (0.2, 0.2, 64, 0.60), (0.1, 0.4, 32, 0.35),
             (0.3, 0.1, 48, 0.90), (0.2, 0.2, 8, 1.50)]
    inputs, scalars = [], []
    for case, (clip, vf_clip, n, spread) in enumerate(specs):
        old_lp = g.vec(n, 0.5) - 2.0
        lp = old_lp + g.vec(n, spread)
        adv = g.vec(n, 1.0)
        old_v = g.vec(n, 2.0)
        v = old_v + g.vec(n, 0.5)
        ret = g.vec(n, 2.0)
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


TOL = 1e-12
FILES = ("gae.csv", "ppo_inputs.csv", "ppo_loss.csv")


def generate() -> dict[str, list[dict]]:
    g = Xorshift(20260901)
    out = {"gae.csv": gae_cases(g)}
    inputs, scalars = loss_cases(g)
    out["ppo_inputs.csv"] = inputs
    out["ppo_loss.csv"] = scalars
    return out


def compare(fresh: dict[str, list[dict]]) -> int:
    """Require every committed value to still fall out of rlhf/ppo.py."""
    bad = 0
    for name in FILES:
        rows = list(csv.DictReader((OUT / name).open()))
        want = fresh[name]
        if len(rows) != len(want):
            print(f"  {name:16} has {len(rows)} rows, a fresh export has {len(want)}")
            bad += 1
            continue
        worst, worst_col = 0.0, ""
        for a, b in zip(rows, want):
            if a.keys() != b.keys():
                print(f"  {name:16} columns changed")
                bad += 1
                break
            for k in a:
                d = abs(float(a[k]) - float(b[k]))
                if d > worst:
                    worst, worst_col = d, k
        if worst > TOL:
            print(f"  {name:16} DIFFERS from a fresh export from rlhf/ppo.py, "
                  f"worst |d| {worst:.1e} in {worst_col}")
            bad += 1
        else:
            print(f"  {name:16} matches a fresh export from rlhf/ppo.py, "
                  f"worst |d| {worst:.1e}")
    return bad


def main() -> int:
    global OUT
    args = sys.argv[1:]
    if args and args[0] == "--check":
        return 1 if compare(generate()) else 0
    if args:
        OUT = Path(args[0])
    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows in generate().items():
        write(name, rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
