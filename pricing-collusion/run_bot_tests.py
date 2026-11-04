"""Step 1 check: run rule-based bots through the market, print a summary,
and chart prices over time against the Nash and cartel benchmarks.

Run from this folder with:  ../.venv/bin/python run_bot_tests.py
"""

import os

import matplotlib.pyplot as plt

import bots
from simulate import play, save_csv

ROUNDS = 100
RESULTS = "results"

SCENARIOS = [
    ("Nash vs. Nash", bots.always_nash, bots.always_nash,
     "Sanity check: competition"),
    ("Cartel vs. Cartel", bots.always_cartel, bots.always_cartel,
     "Sanity check: collusion"),
    ("Best-responder vs. Best-responder", bots.best_responder, bots.best_responder,
     "Start at cartel price, no threat of punishment"),
    ("Grim trigger vs. Cheater", bots.grim_trigger, bots.cheater,
     "Cheater undercuts in round 50; grim trigger punishes"),
    ("Grim trigger vs. Grim trigger", bots.grim_trigger, bots.grim_trigger,
     "Both threaten punishment; nobody cheats"),
]


def main():
    os.makedirs(RESULTS, exist_ok=True)
    runs = []
    print(f"Nash price {bots.P_NASH:.3f} | cartel price {bots.P_MONOPOLY:.3f}\n")
    print(f"{'Scenario':36} {'final p1':>8} {'final p2':>8} "
          f"{'total profit 1':>15} {'total profit 2':>15}")
    for name, bot1, bot2, note in SCENARIOS:
        log = play(bot1, bot2, ROUNDS)
        slug = name.lower().replace(" vs. ", "_vs_").replace(" ", "-")
        save_csv(log, os.path.join(RESULTS, f"{slug}.csv"))
        total1 = sum(row["profit1"] for row in log)
        total2 = sum(row["profit2"] for row in log)
        print(f"{name:36} {log[-1]['price1']:8.3f} {log[-1]['price2']:8.3f} "
              f"{total1:15.1f} {total2:15.1f}")
        runs.append((name, note, log))
    # Cartel vs. Cartel looks identical to Grim vs. Grim, so leave it off.
    charted = [run for run in runs if run[0] != "Cartel vs. Cartel"]
    plot(charted, os.path.join(RESULTS, "bot_tests.png"))


def plot(runs, path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True, sharey=True)
    for ax, (name, note, log) in zip(axes.flat, runs):
        rounds = [row["round"] for row in log]
        ax.axhline(bots.P_MONOPOLY, color="#2e7d32", ls=":", lw=1.5,
                   label=f"Cartel price ({bots.P_MONOPOLY:.2f})")
        ax.axhline(bots.P_NASH, color="#c62828", ls="--", lw=1.5,
                   label=f"Nash price ({bots.P_NASH:.2f})")
        ax.plot(rounds, [row["price1"] for row in log], color="#1f4e79",
                lw=2.2, label="Firm 1")
        ax.plot(rounds, [row["price2"] for row in log], color="#e69f00",
                lw=1.6, ls=(0, (4, 2)), label="Firm 2")
        ax.set_title(f"{name}\n{note}", fontsize=10)
        ax.set_ylim(1.3, 2.05)
        ax.grid(alpha=0.25)
    for ax in axes[1]:
        ax.set_xlabel("Round")
    for ax in axes[:, 0]:
        ax.set_ylabel("Price")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Rule-based bots: prices over 100 rounds", fontsize=13)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=150)
    print(f"\nChart saved to {path}")


if __name__ == "__main__":
    main()
