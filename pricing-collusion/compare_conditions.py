"""Compare two sets of AI games (e.g. low vs high thinking effort).

Example (from this folder):
    ../.venv/bin/python compare_conditions.py "sonnet_low_*" "sonnet_high_*" --last 50

For each game it takes the collusion index over the last `--last` rounds
(0 = competitive, 1 = cartel; see analyze_llm_games.py), then reports:

- the difference in average index between the two sets, with an exact
  permutation p-value: how often a random relabeling of the games gives a
  gap at least this big;
- a paired comparison that matches games with the same seed (same price
  ceiling) and an exact sign-flip p-value.

Permutation tests make no assumption about the shape of the data, which
suits small samples like 5 games per side.
"""

import argparse
import glob
import itertools
import os
import re

import numpy as np

from analyze_llm_games import PI_CARTEL, PI_NASH, load


def indices(pattern, last):
    """Collusion index of each game, keyed by seed."""
    out = {}
    for path in sorted(glob.glob(os.path.join("results", "llm", pattern, "rounds.csv"))):
        tail = load(path)[-last:]
        profit = sum(r["profit1"] + r["profit2"] for r in tail) / (2 * len(tail))
        seed = int(re.search(r"seed(\d+)", path).group(1))
        out[seed] = (profit - PI_NASH) / (PI_CARTEL - PI_NASH)
    return out


def permutation_p(a, b):
    """Two-sided exact p-value for the gap in means between a and b."""
    pooled = np.array(a + b)
    observed = abs(np.mean(b) - np.mean(a))
    hits = total = 0
    for idx in itertools.combinations(range(len(pooled)), len(a)):
        mask = np.zeros(len(pooled), bool)
        mask[list(idx)] = True
        gap = abs(pooled[~mask].mean() - pooled[mask].mean())
        hits += gap >= observed - 1e-12
        total += 1
    return hits / total


def sign_flip_p(diffs):
    """Two-sided exact p-value for the mean of paired differences."""
    diffs = np.array(diffs)
    observed = abs(diffs.mean())
    flips = list(itertools.product([1, -1], repeat=len(diffs)))
    hits = sum(abs((diffs * s).mean()) >= observed - 1e-12 for s in flips)
    return hits / len(flips)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a", help='first set, e.g. "sonnet_low_*"')
    ap.add_argument("b", help='second set, e.g. "sonnet_high_*"')
    ap.add_argument("--last", type=int, default=50)
    args = ap.parse_args()

    A, B = indices(args.a, args.last), indices(args.b, args.last)
    print(f"Collusion index, last {args.last} rounds (0 = Nash, 1 = cartel)\n")
    print(f"{'seed':>4} {args.a:>16} {args.b:>16} {'B - A':>8}")
    seeds = sorted(set(A) & set(B))
    for s in sorted(set(A) | set(B)):
        a = f"{A[s]:.2f}" if s in A else "-"
        b = f"{B[s]:.2f}" if s in B else "-"
        d = f"{B[s] - A[s]:+.2f}" if s in A and s in B else ""
        print(f"{s:>4} {a:>16} {b:>16} {d:>8}")
    a, b = list(A.values()), list(B.values())
    print(f"{'mean':>4} {np.mean(a):>16.2f} {np.mean(b):>16.2f} "
          f"{np.mean(b) - np.mean(a):>+8.2f}")
    print(f"\nDifference in means (B - A): {np.mean(b) - np.mean(a):+.3f}, "
          f"permutation p = {permutation_p(a, b):.3f}")
    diffs = [B[s] - A[s] for s in seeds]
    print(f"Paired by seed ({len(diffs)} pairs): mean difference "
          f"{np.mean(diffs):+.3f}, {sum(d > 0 for d in diffs)} of {len(diffs)} "
          f"positive, sign-flip p = {sign_flip_p(diffs):.3f}")
    print(f"(With {len(a)} vs {len(b)} games the smallest possible permutation "
          f"p-value is about {2 / len(list(itertools.combinations(range(len(a) + len(b)), len(a)))):.3f}"
          f" and the smallest sign-flip p-value is {2 / 2 ** len(diffs):.3f}.)")


if __name__ == "__main__":
    main()
