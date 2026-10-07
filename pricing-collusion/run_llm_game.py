"""Run one game between two AI pricing agents and save the results.

Example (from this folder):
    ../.venv/bin/python run_llm_game.py --model haiku --rounds 5 --prefix P1

Saves to results/llm/<run name>/: rounds.csv (prices and profits),
transcripts.jsonl (every AI answer, including its notes), settings.json,
and prices.png.
"""

import argparse
import json
import os
import random
import time

import matplotlib.pyplot as plt

from bots import P_MONOPOLY, P_NASH
from llm_agent import BudgetExceeded, CostTracker, LLMAgent
from simulate import play, save_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="haiku", help="haiku, sonnet, or opus")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--prefix", default="P1", choices=["P1", "P2"])
    ap.add_argument("--effort", default=None,
                    choices=["low", "medium", "high", "xhigh", "max"],
                    help="thinking effort (not supported on haiku)")
    ap.add_argument("--alpha", type=float, default=1.0, choices=[1.0, 3.2, 10.0],
                    help="currency scale shown to agents (paper: 1, 3.2, 10)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--budget", type=float, default=2.0,
                    help="stop the game once API-equivalent cost passes this ($)")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    # Like the paper: the price ceiling shown to agents is k * cartel price,
    # with k drawn at random between 1.5 and 2.5 (once per game).
    k = random.Random(args.seed).uniform(1.5, 2.5)
    ceiling = k * P_MONOPOLY

    effort = f"_{args.effort}" if args.effort else ""
    name = args.name or (f"{args.model}{effort}_{args.prefix}_a{args.alpha:g}_"
                         f"{args.rounds}r_seed{args.seed}")
    outdir = os.path.join("results", "llm", name)
    os.makedirs(outdir, exist_ok=True)
    transcripts = os.path.join(outdir, "transcripts.jsonl")
    if os.path.exists(transcripts):
        os.remove(transcripts)

    tracker = CostTracker(args.budget)
    firms = [LLMAgent(f"firm{i}", args.model, args.prefix, ceiling, args.alpha,
                      tracker, transcripts, args.effort) for i in (1, 2)]
    settings = vars(args) | {"ceiling": ceiling, "nash_price": P_NASH,
                             "cartel_price": P_MONOPOLY}
    with open(os.path.join(outdir, "settings.json"), "w") as f:
        json.dump(settings, f, indent=2)

    start = time.time()

    def progress(log):
        row = log[-1]
        print(f"Round {row['round']:3}: firm1 {row['price1']:.2f}  "
              f"firm2 {row['price2']:.2f}   "
              f"(${tracker.total:.3f} so far, {time.time() - start:.0f}s)",
              flush=True)
        save_csv(log, os.path.join(outdir, "rounds.csv"))

    print(f"{name}: Nash {P_NASH:.2f}, cartel {P_MONOPOLY:.2f}, "
          f"ceiling {ceiling:.2f} (all in base units; agents see x{args.alpha:g})")
    try:
        log = play(firms[0], firms[1], args.rounds, on_round=progress)
    except BudgetExceeded as e:
        print(f"STOPPED: {e}")
        return
    plot(log, name, os.path.join(outdir, "prices.png"))
    print(f"Done: {tracker.calls} calls, ${tracker.total:.3f} API-equivalent, "
          f"{time.time() - start:.0f}s. Saved to {outdir}/")


def plot(log, title, path):
    rounds = [r["round"] for r in log]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axhline(P_MONOPOLY, color="#2e7d32", ls=":", lw=1.5,
               label=f"Cartel price ({P_MONOPOLY:.2f})")
    ax.axhline(P_NASH, color="#c62828", ls="--", lw=1.5,
               label=f"Nash price ({P_NASH:.2f})")
    ax.plot(rounds, [r["price1"] for r in log], "o-", color="#1f4e79", label="Firm 1")
    ax.plot(rounds, [r["price2"] for r in log], "s--", color="#e69f00", label="Firm 2")
    ax.set_xlabel("Round")
    ax.set_ylabel("Price")
    ax.set_title(f"AI pricing agents: {title}")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    main()
