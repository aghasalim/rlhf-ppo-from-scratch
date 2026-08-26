"""Figures from committed CSVs. Nothing is re-run here."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def fig_overopt(out: Path) -> Path:
    """The plot the repo exists for: proxy up, gold over and down."""
    t = pd.read_csv(RESULTS / "overopt-curve.csv")
    t = t[t["kl"] > 0]
    t["sqrt_kl"] = np.sqrt(t["kl"])

    fig, (a, b) = plt.subplots(1, 2, figsize=(13.5, 5.4))
    bins = np.linspace(0, t["sqrt_kl"].max(), 13)
    t["bin"] = pd.cut(t["sqrt_kl"], bins)
    g = t.groupby("bin", observed=True)
    x = g["sqrt_kl"].median()

    for ax, col, colour, label in ((a, "proxy", "#b2182b", "proxy reward model"),
                                   (a, "gold", "#1a9850", "gold reward (true objective)")):
        med, lo, hi = g[col].median(), g[col].quantile(0.1), g[col].quantile(0.9)
        ax.plot(x, med, marker="o", color=colour, label=label, linewidth=2)
        ax.fill_between(x, lo, hi, color=colour, alpha=0.16, linewidth=0)
    peak = g["gold"].median().idxmax()
    px = x.loc[peak]
    a.axvline(px, color="#333333", linestyle="--", linewidth=1.2)
    a.text(px, a.get_ylim()[0], f"  gold peaks at sqrt(KL) = {px:.1f}", fontsize=9,
           va="bottom", color="#333333")
    a.set_xlabel("$\\sqrt{KL(\\pi \\,\\|\\, \\pi_{ref})}$")
    a.set_ylabel("mean reward")
    a.set_title("Optimising the proxy, watching the gold\n"
                "median over 3 seeds and 4 KL penalties, band is p10 to p90")
    a.grid(alpha=0.3)
    a.legend(frameon=False, fontsize=9.5)

    for col, colour, label in (("motif", "#2166ac", "motifs produced"),
                               ("hoard", "#d6604d", "hoarding penalty (unseen by the proxy)")):
        med = g[col].median()
        b.plot(x, med, marker="o", color=colour, label=label, linewidth=2)
    b.set_xlabel("$\\sqrt{KL(\\pi \\,\\|\\, \\pi_{ref})}$")
    b.set_ylabel("mean count per sequence")
    b.set_title("The mechanism\n"
                "the policy hoards the rewarded motif past the threshold\n"
                "where the gold penalty the proxy never learned switches on")
    b.grid(alpha=0.3)
    b.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_methods(out: Path) -> Path:
    """Gold against KL for every method. Up and to the left is better."""
    t = pd.read_csv(RESULTS / "methods.csv")
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    style = {
        "SFT (reference)": ("#999999", "*", 220),
        "Best-of-4": ("#762a83", "s", 90), "Best-of-16": ("#9970ab", "s", 90),
        "Best-of-64": ("#c2a5cf", "s", 90),
        "DPO": ("#1a9850", "D", 90), "RLOO": ("#2166ac", "^", 90),
        "GRPO": ("#4393c3", "v", 90),
        "PPO (beta=0.2)": ("#fdae61", "o", 90), "PPO (beta=0.05)": ("#f46d43", "o", 90),
        "PPO (beta=0.01)": ("#d73027", "o", 90), "PPO (beta=0.0)": ("#a50026", "o", 90),
    }
    for method, (colour, marker, size) in style.items():
        s = t[t["method"] == method]
        if s.empty:
            continue
        ax.scatter(s["kl"], s["gold"], color=colour, marker=marker, s=size,
                   label=method, alpha=0.85, zorder=3)
    ref = t[t["method"] == "SFT (reference)"]["gold"].median()
    ax.axhline(ref, color="#999999", linestyle=":", linewidth=1.3)
    ax.text(ax.get_xlim()[1] * 0.6, ref, " reference policy", fontsize=9, va="bottom")
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("KL from the reference policy")
    ax.set_ylabel("gold reward (the true objective)")
    ax.set_title("Every method optimises the same proxy.\n"
                 "What separates them is how much gold they buy per unit of drift.")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower left")
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_trajectories(out: Path) -> Path:
    """Per-beta traces, showing where each one stops."""
    t = pd.read_csv(RESULTS / "overopt-curve.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(13.5, 5.2), sharex=True)
    colours = {0.2: "#fdae61", 0.05: "#f46d43", 0.01: "#d73027", 0.0: "#a50026"}
    for beta, colour in colours.items():
        s = t[t["beta"] == beta]
        if s.empty:
            continue
        g = s.groupby("step")
        for ax, col in ((a, "gold"), (b, "proxy")):
            ax.plot(g[col].median().index, g[col].median().values, color=colour,
                    label=f"KL penalty {beta}", linewidth=1.9)
            ax.fill_between(g[col].median().index, g[col].min().values,
                            g[col].max().values, color=colour, alpha=0.13, linewidth=0)
    a.set_ylabel("gold reward"); a.set_title("Gold: rises, then falls without a KL penalty")
    b.set_ylabel("proxy reward"); b.set_title("Proxy: rises regardless, and never warns you")
    for ax in (a, b):
        ax.set_xlabel("PPO step"); ax.grid(alpha=0.3); ax.legend(frameon=False, fontsize=9)
    fig.suptitle("The KL penalty is the only thing stopping the optimiser, median and range over 3 seeds",
                 fontsize=12.5)
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    for p in (fig_overopt(RESULTS / "overoptimization.png"),
              fig_methods(RESULTS / "methods.png"),
              fig_trajectories(RESULTS / "trajectories.png")):
        print(f"-> {p.relative_to(ROOT)} ({p.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
