"""Summarize a set of AI games: final prices, profits, and a chart.

Example (from this folder):
    ../.venv/bin/python analyze_llm_games.py "haiku_P1_a1_100r_seed*" --last 25

For each game it averages the last `--last` rounds (the paper used the last
50 of 300). The "collusion index" is (profit - Nash profit) / (cartel profit
- Nash profit): 0 means fully competitive, 1 means full cartel. It's the
measure used by Calvano et al. (2020).
"""

import argparse
import csv
import glob
import os

import matplotlib.pyplot as plt

from bots import P_MONOPOLY, P_NASH
from market import profits

PI_NASH = profits([P_NASH, P_NASH])[0]
PI_CARTEL = profits([P_MONOPOLY, P_MONOPOLY])[0]


def load(path):
    rows = list(csv.DictReader(open(path)))
    return [{k: float(v) for k, v in r.items()} for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pattern", help='run folders under results/llm, e.g. "haiku_*"')
    ap.add_argument("--last", type=int, default=25)
    ap.add_argument("--out", default=None, help="chart file name")
    ap.add_argument("--title", default="AI pricing agents", help="chart title")
    args = ap.parse_args()

    runs = sorted(glob.glob(os.path.join("results", "llm", args.pattern, "rounds.csv")))
    games = []
    print(f"Averages over the last {args.last} rounds "
          f"(Nash price {P_NASH:.2f}, cartel {P_MONOPOLY:.2f}; "
          f"Nash profit {PI_NASH:.2f}, cartel {PI_CARTEL:.2f})\n")
    print(f"{'game':34} {'rounds':>6} {'price':>6} {'profit':>7} "
          f"{'index':>6} {'p-index':>7}")
    for path in runs:
        name = os.path.basename(os.path.dirname(path))
        log = load(path)
        tail = log[-args.last:]
        price = sum(r["price1"] + r["price2"] for r in tail) / (2 * len(tail))
        profit = sum(r["profit1"] + r["profit2"] for r in tail) / (2 * len(tail))
        index = (profit - PI_NASH) / (PI_CARTEL - PI_NASH)
        p_index = (price - P_NASH) / (P_MONOPOLY - P_NASH)
        games.append((name, log, price, profit, index, p_index))
        print(f"{name:34} {len(log):6} {price:6.2f} {profit:7.2f} "
              f"{index:6.2f} {p_index:7.2f}")
    n = len(games)
    avg = [sum(g[i] for g in games) / n for i in (2, 3, 4, 5)]
    print(f"{'AVERAGE of ' + str(n) + ' games':34} {'':6} {avg[0]:6.2f} "
          f"{avg[1]:7.2f} {avg[2]:6.2f} {avg[3]:7.2f}")
    print("(index = profit-based collusion index; p-index = price-based)")

    out = args.out or os.path.join("results", "llm",
                                   args.pattern.replace("*", "ALL") + ".png")
    plot(games, args.last, out, args.title)
    print(f"\nChart saved to {out}")


def plot(games, last, path, title):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axhline(P_MONOPOLY, color="#2e7d32", ls=":", lw=1.5,
               label=f"Cartel price ({P_MONOPOLY:.2f})")
    ax.axhline(P_NASH, color="#c62828", ls="--", lw=1.5,
               label=f"Nash price ({P_NASH:.2f})")
    colors = ["#1f4e79", "#e69f00", "#6a3d9a", "#00897b", "#d81b60", "#795548"]
    for (name, log, price, *_), c in zip(games, colors * 10):
        label = name.split("_")[-1].replace("seed", "Game ")
        ax.plot([r["round"] for r in log],
                [(r["price1"] + r["price2"]) / 2 for r in log],
                color=c, lw=1.6, label=f"{label} (avg {price:.2f})")
    rounds = max(len(g[1]) for g in games)
    ax.axvspan(rounds - last + 0.5, rounds + 0.5, color="#888", alpha=0.08)
    ax.set_ylim(1.3, 2.6)
    ax.set_xlabel("Round")
    ax.set_ylabel("Average price of the two firms")
    ax.set_title(f"{title}: {len(games)} games\n"
                 f"shaded = last {last} rounds, used for averages", fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    main()
