"""The key chart: collusion index of every AI game, grouped by condition.

Example (from this folder):
    ../.venv/bin/python plot_summary.py

Each dot is one game (index over its last 50 rounds: 0 = competitive,
1 = cartel). The black bar is the group average. Gray lines join Sonnet games
that share a seed (same price ceiling), so each low/high pair can be compared.
"""

import matplotlib.pyplot as plt

from compare_conditions import indices

GROUPS = [
    ("Haiku 4.5\n(100 rounds)", "haiku_P1_a1_100r_seed*"),
    ("Sonnet 5.5\nlow effort", "sonnet_low_P1_a1_200r_seed*"),
    ("Sonnet 5.5\nhigh effort", "sonnet_high_P1_a1_200r_seed*"),
]
DOT = "#2a78d6"
INK, MUTED = "#222222", "#6b6b6b"


def main(out="results/llm/summary_collusion_index.png"):
    data = [indices(pattern, 50) for _, pattern in GROUPS]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(0, color=MUTED, ls="--", lw=1.2)
    ax.axhline(1, color=MUTED, ls=":", lw=1.2)
    ax.text(-0.45, 0.015, "competitive (Nash) = 0", va="bottom", fontsize=8,
            color=MUTED)
    ax.text(-0.45, 1.015, "cartel = 1", va="bottom", fontsize=8, color=MUTED)

    # Pair lines between Sonnet low and high games with the same seed.
    low, high = data[1], data[2]
    for seed in set(low) & set(high):
        ax.plot([1, 2], [low[seed], high[seed]], color="#cccccc", lw=1, zorder=1)

    for x, d in enumerate(data):
        vals = list(d.values())
        jitter = [x + (i - (len(vals) - 1) / 2) * 0.04 for i in range(len(vals))]
        ax.scatter(jitter if x == 0 else [x] * len(vals), vals, s=55, color=DOT,
                   edgecolor="white", linewidth=1.5, zorder=3)
        mean = sum(vals) / len(vals)
        ax.plot([x - 0.18, x + 0.18], [mean, mean], color=INK, lw=2.5, zorder=4)
        ax.text(x + 0.22, mean, f"avg {mean:.2f}", va="center", fontsize=9,
                color=INK)

    ax.set_xticks(range(len(GROUPS)), [g[0] for g in GROUPS])
    ax.set_xlim(-0.5, 2.6)
    ax.set_ylim(-0.35, 1.1)
    ax.set_ylabel("Collusion index (last 50 rounds)")
    ax.set_title("AI pricing agents mostly end up near the competitive level\n"
                 "each dot = one game; prompt P1, 2 firms", fontsize=11)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
