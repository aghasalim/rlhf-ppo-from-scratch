"""Fail if a number in README.md no longer matches results/methods.csv."""
from __future__ import annotations

import csv
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    rows = list(csv.DictReader((ROOT / "results" / "methods.csv").open()))
    body = (ROOT / "README.md").read_text()
    # Detail moved out of the README lives in notes/METHODS.md. A figure quoted
    # there is still a quoted figure and still has to match its source.
    _methods = ROOT / "notes" / "METHODS.md"
    if _methods.exists():
        body += "\n" + _methods.read_text()
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r["method"], []).append(r)

    # kl/proxy/gold are quoted for every method. motif and hoard only appear in
    # the overoptimization table and the DPO caveat, so checking them everywhere
    # would fail on numbers the README never claims.
    DIAGNOSTIC = {"SFT (reference)", "PPO (beta=0.2)", "PPO (beta=0.05)",
                  "PPO (beta=0.01)", "PPO (beta=0.0)", "DPO"}
    claims, failures = [], []
    for method, rs in by.items():
        cols = [("kl", 2), ("proxy", 3), ("gold", 3)]
        if method in DIAGNOSTIC:
            cols += [("motif", 2), ("hoard", 2)]
        for col, places in cols:
            v = statistics.median(float(r[col]) for r in rs)
            claims.append((f"{method} {col}", f"{abs(v):.{places}f}"))

    for label, text in claims:
        if not re.search(r"(?<![\d.])" + re.escape(text) + r"(?!\d)", body):
            failures.append(f"{label} should read {text}, not found")

    print(f"checked {len(claims)} quoted figures against results/methods.csv")
    if failures:
        print("\nDRIFT DETECTED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("no drift")
    # What this does and does not cover, so the green line is not read as more
    # than it is: each figure is recomputed from results/ and looked for in the
    # prose. It cannot catch a wrong number that happens to appear somewhere,
    # it does not check claims written in words (ratios, multiples, ranges),
    # and it does not read notes/LOGBOOK.md.
    print("this checks quoted figures against results/, not claims written in words")
    return 0


if __name__ == "__main__":
    sys.exit(main())
