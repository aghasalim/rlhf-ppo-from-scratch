"""Figures from committed CSVs. Nothing is re-run here.

results/overopt-curve.csv and results/methods.csv are the measured output of
experiments/overopt.py. This module only reads them, so a figure can never
disagree with the numbers quoted in the README.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D

from bench.style import PALETTE, titled

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

# Red for the proxy and green for the gold, because that is how the README
# talks about them. Everything else takes a colour from the shared palette.
PROXY, GOLD = "#b2182b", "#1a9850"
MOTIF, HOARD = PALETTE[0], PALETTE[3]

# One ramp for the KL penalty, used in every figure that shows the sweep, so
# beta=0.2 is the same colour wherever a reader meets it.
BETA_RAMP = {0.2: "#f9cf94", 0.05: "#e89b3c", 0.01: "#b5570b", 0.0: "#6b2f04"}

KL_AXIS = "$\\sqrt{KL(\\pi \\,\\|\\, \\pi_{ref})}$  (nats$^{1/2}$)"
BINS = 10


def _sweep():
    """The committed sweep, with equal-count bins over sqrt(KL).

    Equal-width bins left four checkpoints in some of the high-KL bins, and the
    medians there jumped around for no reason other than the sample count. Ten
    equal-count bins put 18 checkpoints in each. The raw checkpoints are drawn
    faintly behind the medians in every panel, so the binning hides nothing.
    """
    t = pd.read_csv(RESULTS / "overopt-curve.csv")
    t = t[t["kl"] > 0].copy()
    t["sqrt_kl"] = np.sqrt(t["kl"])
    t["bin"] = pd.qcut(t["sqrt_kl"], BINS)
    return t, t.groupby("bin", observed=True)


def fig_overopt(out: Path) -> Path:
    """The plot the repo exists for: proxy up, gold over and down."""
    t, g = _sweep()
    x = g["sqrt_kl"].median().values

    fig, (a, b) = plt.subplots(1, 2, figsize=(13.5, 5.4))

    for col, colour, label in (("proxy", PROXY, "proxy reward model"),
                               ("gold", GOLD, "gold reward (true objective)")):
        a.plot(t["sqrt_kl"], t[col], "o", markersize=2.6, color=colour,
               alpha=0.22, zorder=1)
        a.fill_between(x, g[col].quantile(0.1).values, g[col].quantile(0.9).values,
                       color=colour, alpha=0.15, linewidth=0, zorder=2)
        a.plot(x, g[col].median().values, marker="o", color=colour, label=label,
               linewidth=2.2, zorder=3)

    px = x[int(np.argmax(g["gold"].median().values))]
    a.axvline(px, color="#333333", linestyle="--", linewidth=1.1, zorder=1)
    a.annotate(f"gold peaks near sqrt(KL) = {px:.1f}", xy=(px, 0.72),
               xycoords=("data", "axes fraction"), xytext=(6, 0),
               textcoords="offset points", fontsize=9, color="#333333", va="center")
    a.set_xlabel(KL_AXIS)
    a.set_ylabel("mean reward per 24-token sample\n(the two rewards share an axis, not a scale)")
    titled(a, "The proxy keeps climbing after the gold turns over",
           "line is the median of 3 seeds and 4 KL penalties, band is p10 to p90, dots are the 180 checkpoints")
    a.legend(loc="lower left")

    for col, colour, label in (("motif", MOTIF, "motifs produced"),
                               ("hoard", HOARD, "hoarding penalty, unseen by the proxy")):
        b.plot(t["sqrt_kl"], t[col], "o", markersize=2.6, color=colour, alpha=0.22, zorder=1)
        b.plot(x, g[col].median().values, marker="o", color=colour, label=label,
               linewidth=2.2, zorder=3)
    b.set_xlabel(KL_AXIS)
    b.set_ylabel("mean count per 24-token sample")
    titled(b, "Hoarding the motif is what eats the gold",
           "the policy stacks the rewarded bigram until a penalty the proxy never saw switches on")
    b.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_methods(out: Path) -> Path:
    """Gold against KL for every method. Up and to the left is better."""
    t = pd.read_csv(RESULTS / "methods.csv")
    med = t.groupby("method")[["kl", "gold"]].median()

    families = (
        ("Best-of-N", ["Best-of-4", "Best-of-16", "Best-of-64"],
         ["#a8c4dd", "#5b8db8", PALETTE[0]], "s", ["N=4", "16", "64"]),
        ("PPO, KL penalty 0.2 down to 0",
         ["PPO (beta=0.2)", "PPO (beta=0.05)", "PPO (beta=0.01)", "PPO (beta=0.0)"],
         list(BETA_RAMP.values()), "o", ["0.2", "0.05", "0.01", "0"]),
    )
    # Hand placed so a label never lands on another method's marker.
    TAG_OFFSET = {"0.2": (-10, 0, "right", "center"), "0": (0, -12, "center", "top")}
    singles = (("SFT (reference)", PALETTE[5], "*", 260),
               ("DPO", PALETTE[4], "D", 85),
               ("RLOO", PALETTE[2], "^", 100),
               ("GRPO", "#7fbf7b", "v", 100))

    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    handles = []

    for label, names, colours, marker, tags in families:
        ax.plot(med.loc[names, "kl"], med.loc[names, "gold"], color=colours[-1],
                linewidth=1.1, alpha=0.45, zorder=2)
        for name, colour, tag in zip(names, colours, tags):
            s = t[t["method"] == name]
            ax.plot(s["kl"], s["gold"], marker, markersize=4.5, color=colour,
                    alpha=0.35, zorder=3)
            ax.scatter(med.loc[name, "kl"], med.loc[name, "gold"], color=colour,
                       marker=marker, s=95, zorder=4, edgecolors="#666666", linewidths=0.6)
            dx, dy, ha, va = TAG_OFFSET.get(tag, (0, 11, "center", "bottom"))
            ax.annotate(tag, (med.loc[name, "kl"], med.loc[name, "gold"]),
                        xytext=(dx, dy), textcoords="offset points", fontsize=8.5,
                        color="#5a5a5a", ha=ha, va=va)
        handles.append(Line2D([], [], color=colours[-1], marker=marker, linewidth=1.1,
                              markersize=8, label=label))

    for name, colour, marker, size in singles:
        s = t[t["method"] == name]
        ax.plot(s["kl"], s["gold"], marker, markersize=4.5, color=colour, alpha=0.35, zorder=3)
        ax.scatter(med.loc[name, "kl"], med.loc[name, "gold"], color=colour, marker=marker,
                   s=size, zorder=4, edgecolors="#666666", linewidths=0.6)
        handles.append(Line2D([], [], color=colour, marker=marker, linestyle="none",
                              markersize=9, label=name))

    ref = med.loc["SFT (reference)", "gold"]
    ax.axhline(ref, color="#999999", linestyle=":", linewidth=1.2, zorder=1)
    ax.text(0.995, ref, "anything below this line is worse than the reference ", fontsize=9,
            color="#777777", va="bottom", ha="right", transform=ax.get_yaxis_transform())

    worst = t["gold"].idxmin()
    ax.annotate(f"one seed collapses to {t.loc[worst, 'gold']:.2f}",
                (t.loc[worst, "kl"], t.loc[worst, "gold"]), xytext=(-8, 0),
                textcoords="offset points", fontsize=8.5, color="#5a5a5a", ha="right", va="center")

    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlim(-0.08, 200)
    ax.set_xticks([0, 1, 10, 100])
    ax.set_xticklabels(["0", "1", "10", "100"])
    ax.set_xlabel("KL from the reference policy (nats, symlog axis)")
    ax.set_ylabel("mean gold reward per 24-token sample")
    titled(ax, "Best-of-N never travels far enough, PPO with no penalty travels too far",
           "all five optimise the same reward model; large marker is the median of 3 seeds, small ones are the seeds")
    ax.legend(handles=handles, loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_trajectories(out: Path) -> Path:
    """Per-beta traces, showing where each one stops."""
    t = pd.read_csv(RESULTS / "overopt-curve.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(13.5, 5.4), sharex=True)
    for beta, colour in BETA_RAMP.items():
        s = t[t["beta"] == beta]
        if s.empty:
            continue
        g = s.groupby("step")
        for ax, col in ((a, "gold"), (b, "proxy")):
            ax.plot(g[col].median().index, g[col].median().values, color=colour,
                    label=f"KL penalty {beta}", linewidth=1.9)
            ax.fill_between(g[col].median().index, g[col].min().values,
                            g[col].max().values, color=colour, alpha=0.13, linewidth=0)

    a.set_ylabel("mean gold reward per 24-token sample")
    titled(a, "Gold rises, then falls once nothing holds the policy back",
           "median over 3 seeds, band is the min to max across those seeds")
    b.set_ylabel("mean proxy score per 24-token sample")
    titled(b, "The proxy rises whatever you do, and never warns you",
           "the same runs, scored by the reward model instead of the gold")
    for ax in (a, b):
        ax.set_xlabel("PPO step")
    handles, labels = a.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    fig.savefig(out)
    plt.close(fig)
    return out


def _upto(x, y, cut):
    """The polyline (x, y) clipped at cut, with the cut segment interpolated."""
    k = int(np.searchsorted(x, cut, side="right"))
    if k == 0:
        return x[:0], y[:0]
    if k == len(x):
        return x, y
    return np.append(x[:k], cut), np.append(y[:k], np.interp(cut, x, y))


def anim_goodhart(out: Path, frames: int = 84, hold: int = 18, fps: int = 16) -> Path:
    """The same curve as fig_overopt, drawn in order of drift.

    Reads the committed sweep. Nothing is sampled or simulated here, so the GIF
    is byte identical on every run.
    """
    t, g = _sweep()
    x = g["sqrt_kl"].median().values
    med = {c: g[c].median().values for c in ("proxy", "gold")}
    peak = int(np.argmax(med["gold"]))

    far = t["sqrt_kl"].max()
    # The lines are bin medians, so they stop at the last bin centre while the
    # checkpoints run on to far. Shade the gap and say so, otherwise the end of
    # the sweep reads as a truncated line rather than as thinning data.
    last = x[-1]
    tail = int((t["sqrt_kl"] > last).sum())

    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    ax.set_xlim(0, far * 1.03)
    # headroom at the top so the running readout never sits on the proxy line
    ax.set_ylim(t[["proxy", "gold"]].min().min() - 0.8, t[["proxy", "gold"]].max().max() + 2.6)
    ax.set_xlabel(KL_AXIS)
    ax.set_ylabel("mean reward per 24-token sample")
    titled(ax, "The gold turns over, the proxy does not",
           "one sweep replayed in order of drift: 3 seeds, 4 KL penalties, 180 checkpoints")

    ax.axvspan(last, far * 1.03, color="#f0f0f0", zorder=0)
    ax.text(last + 0.06, 0.60, f"no median past here\njust {tail} checkpoints",
            transform=ax.get_xaxis_transform(), fontsize=8.0, color="#777777",
            ha="left", va="center")

    art = {}
    for col, colour, label in (("proxy", PROXY, "proxy reward model"),
                               ("gold", GOLD, "gold reward (true objective)")):
        art[col + "_raw"] = ax.plot([], [], "o", markersize=2.6, color=colour, alpha=0.22, zorder=1)[0]
        art[col] = ax.plot([], [], color=colour, linewidth=2.4, label=label, zorder=3)[0]
        art[col + "_head"] = ax.plot([], [], "o", markersize=6.5, color=colour, zorder=4)[0]
    ax.legend(loc="lower left")

    art["vline"] = ax.axvline(x[0], color="#999999", linestyle="--", linewidth=1.0, zorder=2)
    art["readout"] = ax.text(0.02, 0.97, "", transform=ax.transAxes, fontsize=9.5,
                             color="#555555", ha="left", va="top")
    art["peak"] = ax.plot([], [], "v", markersize=8, color="#333333", zorder=5)[0]
    art["peak_text"] = ax.text(x[peak], med["gold"][peak] + 1.5, "", fontsize=9, color="#333333",
                               ha="center", va="bottom")
    art["end_text"] = ax.text(far, med["proxy"][-1] + 0.7, "", fontsize=9.5, color=PROXY,
                              ha="right", va="bottom")

    cuts = np.linspace(x[0], far, frames)

    def draw(i):
        cut = cuts[min(i, frames - 1)]
        seen = t[t["sqrt_kl"] <= cut]
        for col in ("proxy", "gold"):
            art[col + "_raw"].set_data(seen["sqrt_kl"], seen[col])
            cx, cy = _upto(x, med[col], cut)
            art[col].set_data(cx, cy)
            art[col + "_head"].set_data(cx[-1:], cy[-1:])
        art["vline"].set_xdata([cut, cut])
        tag = f"\nmedians stop at {last:.1f}" if cut > last else ""
        art["readout"].set_text(f"sqrt(KL) = {cut:.1f}{tag}")
        if cut >= x[peak]:
            art["peak"].set_data([x[peak]], [med["gold"][peak] + 0.8])
            art["peak_text"].set_text(f"gold peaks here, sqrt(KL) = {x[peak]:.1f}")
        if i >= frames:
            art["end_text"].set_text("the proxy score never warned you")
        return list(art.values())

    anim = FuncAnimation(fig, draw, frames=frames + hold, interval=1000 // fps, blit=False)
    anim.save(out, writer=PillowWriter(fps=fps), dpi=100)
    plt.close(fig)
    return out


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    for p in (fig_overopt(RESULTS / "overoptimization.png"),
              fig_methods(RESULTS / "methods.png"),
              fig_trajectories(RESULTS / "trajectories.png"),
              anim_goodhart(RESULTS / "goodhart.gif")):
        print(f"-> {p.relative_to(ROOT)} ({p.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
