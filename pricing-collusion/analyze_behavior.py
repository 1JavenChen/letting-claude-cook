"""Why do prices stay high? Two checks on a set of AI games.

1. Lock-in test (the on-path regression from Fish et al. 2024, Section 4):
       my price now = a + g * (my last price) + d * (rival's last price)
   fitted for each firm in each game. If firms follow or punish their rival,
   d is clearly positive. If firms have locked in and ignore the rival,
   d is near zero.

2. Notes analysis: the share of sentences in the agents' PLANS that talk
   about the rival's reaction (price wars, retaliation, matching) versus
   holding steady (stability, "maintain", "no change"). A simple keyword
   version of Fish et al.'s embedding-based classifier, so treat it as rough.

Example (from this folder):
    ../.venv/bin/python analyze_behavior.py "haiku_P1_a1_100r_seed*" --skip 10
"""

import argparse
import csv
import glob
import json
import os
import re

import numpy as np

REACTION = re.compile(
    r"price war|retaliat|punish|undercut|match(es|ed|ing)? (my|our|the)? ?"
    r"(price|move)|respond(s|ed)? to (my|our)|reaction|follow(s|ed)? (my|our)",
    re.I)
STABILITY = re.compile(
    r"\bmaintain|\bhold(ing)?\b|stabl|equilibrium|lock(ed)? in|no change|"
    r"don't change|do not change|keep (the )?price|stay(ing)? at|continue at",
    re.I)


def regress(rows, firm, skip):
    """Fit my price on my last price and the rival's last price."""
    me, rival = f"price{firm}", f"price{3 - firm}"
    y, X = [], []
    for prev, now in zip(rows[skip:], rows[skip + 1:]):
        y.append(now[me])
        X.append([1.0, prev[me], prev[rival]])
    X, y = np.array(X), np.array(y)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    dof = max(len(y) - 3, 1)
    cov = (resid @ resid / dof) * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    return coef[1], coef[2], se[2]


def sentences(text):
    parts = re.split(r"(?<=[.!?])\s+|\n+", text or "")
    return [p.strip(" -*#+") for p in parts if len(p.strip(" -*#+")) > 8]


def plans_text(response):
    m = re.search(r"New content for PLANS\.txt:(.*?)New content for INSIGHTS\.txt:",
                  response or "", re.S)
    return m.group(1) if m else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pattern")
    ap.add_argument("--skip", type=int, default=10,
                    help="ignore the first N rounds (initial exploration)")
    args = ap.parse_args()

    print(f"{'game':34} {'firm':>4} {'own g':>6} {'rival d':>8} {'(se)':>6} "
          f"{'reaction%':>9} {'steady%':>8}")
    ds, reacts, steadies = [], [], []
    for folder in sorted(glob.glob(os.path.join("results", "llm", args.pattern, ""))):
        folder = folder.rstrip("/")
        name = os.path.basename(folder)
        rows = [{k: float(v) for k, v in r.items()}
                for r in csv.DictReader(open(os.path.join(folder, "rounds.csv")))]
        notes = [json.loads(l) for l in open(os.path.join(folder, "transcripts.jsonl"))]
        for firm in (1, 2):
            g, d, se = regress(rows, firm, args.skip)
            sents = [s for n in notes
                     if n["firm"] == f"firm{firm}" and n.get("price_shown") is not None
                     for s in sentences(plans_text(n["response"]))]
            react = 100 * sum(bool(REACTION.search(s)) for s in sents) / max(len(sents), 1)
            steady = 100 * sum(bool(STABILITY.search(s)) for s in sents) / max(len(sents), 1)
            ds.append(d); reacts.append(react); steadies.append(steady)
            print(f"{name:34} {firm:4} {g:6.2f} {d:8.2f} {se:6.2f} "
                  f"{react:8.1f}% {steady:7.1f}%")
    print(f"{'AVERAGE':34} {'':4} {'':6} {np.mean(ds):8.2f} {'':6} "
          f"{np.mean(reacts):8.1f}% {np.mean(steadies):7.1f}%")
    print("\nrival d near 0 = rival ignored (lock-in); clearly > 0 = following "
          "or punishing the rival.")


if __name__ == "__main__":
    main()
