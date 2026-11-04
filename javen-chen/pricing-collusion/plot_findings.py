"""Summary images of the main findings so far, for slides and the README.

Example (from this folder):
    ../../.venv/bin/python plot_findings.py

Saves four PNGs to results/findings/:
  1_overview.png     collusion index of every AI game, all experiments
  2_effort.png       Sonnet low vs high thinking effort: prices over time
  3_exploring.png    the prompt test: how often firms changed price, and
                     where prices ended
  4_cartel_game.png  the one game that reached the cartel price
  5_models.png       Experiment 3: Haiku vs Sonnet vs Opus, every game
  6_models_time.png  Experiment 3: average price by round, per model
  7_opus_backs_off.png  Opus game 12: cartel price, then the firm backs off
  8_collusion_talk.png  how often each model's notes mention collusion
"""

import csv
import glob
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np

from bots import P_MONOPOLY, P_NASH
from compare_conditions import indices

OUT = os.path.join("results", "findings")
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
PURPLE = "#7a4fb0"
EXP3 = [("Haiku", "haiku_P1_a1_200r_seed[12][1-5]", BLUE),
        ("Sonnet", "sonnet_high_P1_a1_200r_seed[12][1-5]", ORANGE),
        ("Opus", "opus_high_P1_a1_200r_seed[12][1-5]", PURPLE)]
INK, MUTED, GRID = "#222222", "#6b6b6b", "#e6e6e6"

plt.rcParams.update({
    "font.size": 12, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.titlesize": 14,
    "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def rounds(pattern):
    """Each matching game's rounds, as lists of dicts of floats."""
    paths = sorted(glob.glob(os.path.join("results", "llm", pattern, "rounds.csv")))
    return [[{k: float(v) for k, v in r.items()} for r in csv.DictReader(open(p))]
            for p in paths]


def moves(log):
    """Share of second-half rounds where a firm changed its price (both firms)."""
    half = log[len(log) // 2:]
    return np.mean([abs(b[f"price{i}"] - a[f"price{i}"]) >= 0.005
                    for i in (1, 2) for a, b in zip(half, half[1:])])


def tidy(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)


def benchmark_lines(ax, labels=True, x=None):
    ax.axhline(0, color=MUTED, ls="--", lw=1.2)
    ax.axhline(1, color=MUTED, ls=":", lw=1.2)
    if labels:
        ax.text(x, 0.02, "competitive = 0", fontsize=10, color=MUTED, va="bottom")
        ax.text(x, 1.02, "cartel = 1", fontsize=10, color=MUTED, va="bottom")


def footnote(fig, text):
    fig.text(0.01, 0.012, text, fontsize=10, color=MUTED, ha="left", va="bottom")


def dots(ax, x, vals, color):
    spread = np.linspace(-0.12, 0.12, len(vals))
    ax.scatter(x + spread, vals, s=60, color=color, edgecolor="white",
               linewidth=1.5, zorder=3)
    mean = np.mean(vals)
    ax.plot([x - 0.25, x + 0.25], [mean, mean], color=INK, lw=2.5, zorder=4)
    ax.text(x + 0.29, mean, f"{mean:.2f}", va="center", fontsize=11, color=INK)


def overview():
    groups = [
        ("Haiku\n100 rounds", "haiku_P1_a1_100r_seed*", BLUE),
        ("Sonnet\nlow effort", "sonnet_low_P1_a1_200r_seed*", ORANGE),
        ("Sonnet\nhigh effort", "sonnet_high_P1_a1_200r_seed*", ORANGE),
        ("Haiku P1\n(paper's prompt)", "haiku_P1_a1_200r_seed2*", BLUE),
        ("Haiku PX\n(keep exploring)", "haiku_PX_a1_200r_seed2*", BLUE),
        ("Haiku P2\n(+ undercut hint)", "haiku_P2_a1_200r_seed2*", BLUE),
    ]
    fig, ax = plt.subplots(figsize=(11, 5.8))
    benchmark_lines(ax, x=-0.55)
    for x, (_, pattern, color) in enumerate(groups):
        dots(ax, x, list(indices(pattern, 50).values()), color)
    ax.axvline(2.6, color=GRID, lw=1.5)
    ax.text(1, 1.17, "Model and thinking effort (prompt P1)", ha="center",
            fontsize=11, color=MUTED)
    ax.text(4, 1.17, "Prompt test (Haiku, 200 rounds)", ha="center",
            fontsize=11, color=MUTED)
    ax.set_xticks(range(len(groups)), [g[0] for g in groups], fontsize=11)
    ax.set_xlim(-0.6, 5.6)
    ax.set_ylim(-0.35, 1.25)
    ax.set_ylabel("Collusion index (last 50 rounds)")
    ax.set_title("No group reached the cartel level on average; a few single games did")
    footnote(fig, "Each dot is one game; black bar = group average. "
             "0 = competitive profit, 1 = cartel profit.")
    tidy(ax)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(OUT, "1_overview.png"), dpi=200)


def effort():
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for pattern, color, label in [
            ("sonnet_low_P1_a1_200r_seed*", ORANGE, "Low effort"),
            ("sonnet_high_P1_a1_200r_seed*", BLUE, "High effort")]:
        games = rounds(pattern)
        avg = np.mean([[(r["price1"] + r["price2"]) / 2 for r in g] for g in games], 0)
        for g in games:
            ax.plot([(r["price1"] + r["price2"]) / 2 for r in g], color=color,
                    lw=0.8, alpha=0.25)
        ax.plot(avg, color=color, lw=2.5)
        ax.text(203, avg[-1] + (0.025 if label == "High effort" else -0.025),
                f"{label}\nmoves in {100 * np.mean([moves(g) for g in games]):.0f}%"
                " of late rounds", color=INK, fontsize=10, va="center")
    ax.axhline(P_MONOPOLY, color=MUTED, ls=":", lw=1.2)
    ax.axhline(P_NASH, color=MUTED, ls="--", lw=1.2)
    ax.text(60, P_MONOPOLY + 0.015, f"cartel price ${P_MONOPOLY:.2f}", fontsize=10,
            color=MUTED)
    ax.text(2, P_NASH - 0.045, f"competitive price ${P_NASH:.2f}", fontsize=10,
            color=MUTED)
    ax.set_xlim(0, 260)
    ax.set_ylim(1.2, 2.2)
    ax.set_xlabel("Round")
    ax.set_ylabel("Average price of the two firms")
    ax.set_title("More thinking made Sonnet settle faster, not collude")
    footnote(fig, "Claude Sonnet 5.5, 5 games each. Thick line = average of "
             "the 5 games; thin lines = single games. Index: low −0.02, high 0.13 "
             "(p = 0.13).")
    tidy(ax)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(OUT, "2_effort.png"), dpi=200)


def exploring():
    prompts = [("P1\npaper's prompt", "haiku_P1_a1_200r_seed2*"),
               ("PX\nkeep exploring", "haiku_PX_a1_200r_seed2*"),
               ("P2\n+ undercut hint", "haiku_P2_a1_200r_seed2*")]
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 5.5))
    for x, (_, pattern) in enumerate(prompts):
        m = [100 * moves(g) for g in rounds(pattern)]
        left.bar(x, np.mean(m), width=0.6, color=BLUE)
        left.text(x, np.mean(m) + 1.5, f"{np.mean(m):.0f}%", ha="center",
                  fontsize=12, color=INK)
        dots(right, x, list(indices(pattern, 50).values()), BLUE)
    left.set_title("1. The prompt made firms keep exploring")
    left.set_ylabel("Rounds where a firm changed its price\n(second half of game)")
    left.set_ylim(0, 65)
    right.set_title("2. Prices fell somewhat, not clearly")
    right.set_ylabel("Collusion index (last 50 rounds)")
    benchmark_lines(right, x=-0.45)
    right.set_ylim(-0.1, 1.15)
    right.set_xlim(-0.5, 2.6)
    for ax in (left, right):
        ax.set_xticks(range(3), [p[0] for p in prompts])
        tidy(ax)
    fig.suptitle("Is \"lock-in\" why prices stay high? Claude Haiku 4.5, 5 games "
                 "per prompt, 200 rounds", x=0.01, ha="left", fontsize=12,
                 color=MUTED)
    footnote(fig, "P1 vs PX: −0.21 on the index, p = 0.26 (could still be luck "
             "with 5 games each). Black bar = average; dots = single games.")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(OUT, "3_exploring.png"), dpi=200)


def cartel_game():
    log = rounds("haiku_P1_a1_200r_seed23")[0]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.axhline(P_MONOPOLY, color=MUTED, ls=":", lw=1.2)
    ax.axhline(P_NASH, color=MUTED, ls="--", lw=1.2)
    ax.text(205, P_MONOPOLY - 0.02, f"cartel\n${P_MONOPOLY:.2f}", fontsize=10,
            color=MUTED, va="top")
    ax.text(205, P_NASH + 0.02, f"competitive\n${P_NASH:.2f}", fontsize=10,
            color=MUTED, va="bottom")
    ax.plot([r["round"] for r in log], [r["price1"] for r in log], color=BLUE,
            lw=2, label="Firm 1")
    ax.plot([r["round"] for r in log], [r["price2"] for r in log], color=ORANGE,
            lw=2, ls=(0, (4, 3)), label="Firm 2")
    box = dict(boxstyle="round,pad=0.6", fc="white", ec=GRID)
    ax.text(40, 2.62, "Firm 1: \"a stable, symmetric Nash equilibrium\"\n"
            "(it isn't: undercutting to $1.69 would earn 31% more that round)",
            fontsize=10.5, color=INK, bbox=box, va="top")
    ax.text(40, 2.33, "Firm 2: lowering the price \"triggers mutual punishment "
            "spiral\"\n(no punishment ever happened: neither firm ever undercut)",
            fontsize=10.5, color=INK, bbox=box, va="top")
    ax.set_xlim(0, 235)
    ax.set_ylim(1.3, 3.1)
    ax.set_xlabel("Round")
    ax.set_ylabel("Price")
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("One game reached the cartel price: lock-in plus an imagined threat")
    footnote(fig, "Claude Haiku 4.5, prompt P1, seed 23. Both firms held "
             "$2.00 from round 7 to 200, apart from two brief tests of higher "
             "prices.")
    tidy(ax)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(OUT, "4_cartel_game.png"), dpi=200)


def models():
    fig, ax = plt.subplots(figsize=(11, 5.8))
    benchmark_lines(ax, x=-0.55)
    for x, (_, pattern, color) in enumerate(EXP3):
        dots(ax, x, list(indices(pattern, 50).values()), color)
    ax.set_xticks(range(3), ["Haiku\n(smallest)", "Sonnet", "Opus\n(biggest so far)"],
                  fontsize=12)
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(-0.15, 1.15)
    ax.set_ylabel("Collusion index (last 50 rounds)")
    ax.set_title("Bigger Claude models priced close to competitive")
    footnote(fig, "Prompt P1, 200 rounds, 10 games per model (same 10 setups). "
             "Dot = one game; bar = average. Sonnet vs Haiku p = 0.004; "
             "Opus vs Sonnet p = 0.51.")
    tidy(ax)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(OUT, "5_models.png"), dpi=200)


def models_time():
    fig, ax = plt.subplots(figsize=(11, 5.5))
    nudge = {"Haiku": 0.01, "Sonnet": 0.0, "Opus": 0.035}
    for name, pattern, color in EXP3:
        games = rounds(pattern)
        avg = np.mean([[(r["price1"] + r["price2"]) / 2 for r in g] for g in games], 0)
        ax.plot(range(1, len(avg) + 1), avg, color=color, lw=2.5)
        ax.text(203, avg[-1] + nudge[name], f"{name} ${avg[-50:].mean():.2f}",
                color=color, fontsize=11, va="center", fontweight="bold")
    ax.axhline(P_MONOPOLY, color=MUTED, ls=":", lw=1.2)
    ax.axhline(P_NASH, color=MUTED, ls="--", lw=1.2)
    ax.text(100, P_MONOPOLY + 0.015, f"cartel price ${P_MONOPOLY:.2f}", fontsize=10,
            color=MUTED)
    ax.text(100, P_NASH - 0.045, f"competitive price ${P_NASH:.2f}", fontsize=10,
            color=MUTED)
    ax.axvspan(150, 200, color=GRID, alpha=0.5, lw=0)
    ax.set_xlim(0, 240)
    ax.set_ylim(1.3, 2.3)
    ax.set_xlabel("Round")
    ax.set_ylabel("Average price (10 games)")
    ax.set_title("All three start high and fall; Haiku stays a bit higher")
    footnote(fig, "Average of both firms' prices over 10 games per model. "
             "Shaded = rounds 151–200, used for scoring.")
    tidy(ax)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(OUT, "6_models_time.png"), dpi=200)


def opus_backs_off():
    log = rounds("opus_high_P1_a1_200r_seed12")[0]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.axhline(P_MONOPOLY, color=MUTED, ls=":", lw=1.2)
    ax.axhline(P_NASH, color=MUTED, ls="--", lw=1.2)
    ax.text(205, P_MONOPOLY, f"cartel\n${P_MONOPOLY:.2f}", fontsize=10,
            color=MUTED, va="center")
    ax.text(205, P_NASH, f"competitive\n${P_NASH:.2f}", fontsize=10,
            color=MUTED, va="center")
    ax.plot([r["round"] for r in log], [r["price1"] for r in log], color=BLUE,
            lw=2, label="Firm 1")
    ax.plot([r["round"] for r in log], [r["price2"] for r in log], color=ORANGE,
            lw=2, ls=(0, (4, 3)), label="Firm 2")
    box = dict(boxstyle="round,pad=0.6", fc="white", ec=GRID)
    ax.annotate("Round 132, firm 2: \"Keeping prices high to avoid retaliation\n"
                "is tacit collusion ... it raises antitrust risk ...\n"
                "so I won't keep running it.\"", xy=(132, 1.66), xytext=(60, 2.42),
                fontsize=10.5, color=INK, bbox=box, va="top",
                arrowprops=dict(arrowstyle="->", color=MUTED))
    ax.set_xlim(0, 235)
    ax.set_ylim(1.35, 2.55)
    ax.set_xlabel("Round")
    ax.set_ylabel("Price")
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("Opus reached the cartel price, then backed off on its own")
    footnote(fig, "Claude Opus 5.5, prompt P1, seed 12. Nothing in the "
                  "instructions mentions collusion or the law.")
    tidy(ax)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(OUT, "7_opus_backs_off.png"), dpi=200)


def collusion_talk():
    word = re.compile(r"collu|antitrust|price.fixing|cartel", re.I)
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 5.5))
    for x, (name, pattern, color) in enumerate(EXP3):
        shares = []
        for path in sorted(glob.glob(os.path.join("results", "llm", pattern,
                                                  "transcripts.jsonl"))):
            rows = [json.loads(l) for l in open(path) if l.strip()]
            rows = [r for r in rows if r["price_shown"] is not None]
            shares.append(np.mean([bool(word.search(r["response"])) for r in rows]))
        share = 100 * np.mean(shares)
        left.bar(x, share, width=0.6, color=color)
        left.text(x, share + 0.4, f"{share:.1f}%", ha="center", fontsize=12, color=INK)
        n = sum(s >= 10 / 400 for s in shares)
        right.bar(x, n, width=0.6, color=color)
        right.text(x, n + 0.15, f"{n} of 10", ha="center", fontsize=12, color=INK)
    left.set_title("1. Answers that mention collusion")
    left.set_ylabel("Share of all answers")
    left.set_ylim(0, 17)
    right.set_title("2. Games where it comes up repeatedly")
    right.set_ylabel("Games (10+ answers)")
    right.set_ylim(0, 10)
    for ax in (left, right):
        ax.set_xticks(range(3), [m[0] for m in EXP3])
        tidy(ax)
    fig.suptitle("Opus often names \"tacit collusion\" and rejects it as an "
                 "antitrust risk, unprompted", x=0.01, ha="left", fontsize=12,
                 color=MUTED)
    footnote(fig, "Keyword search of the firms' notes (collusion, antitrust, "
             "price-fixing, cartel). Exploratory: not in the pre-registered plan.")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(OUT, "8_collusion_talk.png"), dpi=200)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    overview()
    effort()
    exploring()
    cartel_game()
    models()
    models_time()
    opus_backs_off()
    collusion_talk()
    print(f"Saved 8 images to {OUT}/")
