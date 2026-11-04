"""Summary charts: collusion index of every AI game, grouped by condition.

Example (from this folder):
    ../.venv/bin/python plot_summary.py

Each dot is one game (index over its last 50 rounds: 0 = competitive,
1 = cartel). The black bar is the group average. Gray lines join games in
neighboring groups that share a seed (same price ceiling), so matched pairs
can be compared. Makes two charts: models and thinking effort (Stage 2), and
the prompt test (Experiment 2).
"""

import matplotlib.pyplot as plt

from compare_conditions import indices

STAGE2 = [
    ("Haiku 4.5\n(100 rounds)", "haiku_P1_a1_100r_seed*"),
    ("Sonnet 5.5\nlow effort", "sonnet_low_P1_a1_200r_seed*"),
    ("Sonnet 5.5\nhigh effort", "sonnet_high_P1_a1_200r_seed*"),
]
PROMPTS = [
    ("P1\npaper's prompt", "haiku_P1_a1_200r_seed2*"),
    ("PX\nexplore, no hint", "haiku_PX_a1_200r_seed2*"),
    ("P2\nexplore + undercut hint", "haiku_P2_a1_200r_seed2*"),
]
DOT = "#2a78d6"
INK, MUTED = "#222222", "#6b6b6b"


def chart(groups, out, title, paired_from=1):
    data = [indices(pattern, 50) for _, pattern in groups]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(0, color=MUTED, ls="--", lw=1.2)
    ax.axhline(1, color=MUTED, ls=":", lw=1.2)
    ax.text(-0.45, 0.015, "competitive (Nash) = 0", va="bottom", fontsize=8,
            color=MUTED)
    ax.text(-0.45, 1.015, "cartel = 1", va="bottom", fontsize=8, color=MUTED)

    # Pair lines between games in neighboring groups with the same seed.
    for x in range(paired_from, len(data) - 1):
        a, b = data[x], data[x + 1]
        for seed in set(a) & set(b):
            ax.plot([x, x + 1], [a[seed], b[seed]], color="#cccccc", lw=1,
                    zorder=1)

    for x, d in enumerate(data):
        vals = list(d.values())
        jitter = [x + (i - (len(vals) - 1) / 2) * 0.04 for i in range(len(vals))]
        paired = x >= paired_from
        ax.scatter([x] * len(vals) if paired else jitter, vals, s=55, color=DOT,
                   edgecolor="white", linewidth=1.5, zorder=3)
        mean = sum(vals) / len(vals)
        ax.plot([x - 0.18, x + 0.18], [mean, mean], color=INK, lw=2.5, zorder=4)
        ax.text(x + 0.22, mean, f"avg {mean:.2f}", va="center", fontsize=9,
                color=INK)

    ax.set_xticks(range(len(groups)), [g[0] for g in groups])
    ax.set_xlim(-0.5, 2.6)
    ax.set_ylim(-0.35, 1.1)
    ax.set_ylabel("Collusion index (last 50 rounds)")
    ax.set_title(title, fontsize=11)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


def main():
    chart(STAGE2, "results/llm/summary_collusion_index.png",
          "AI pricing agents mostly end up near the competitive level\n"
          "each dot = one game; prompt P1, 2 firms")
    chart(PROMPTS, "results/llm/exp2_prompts.png",
          "Experiment 2 (Claude Haiku 4.5, 200 rounds): does exploring lower prices?\n"
          "each dot = one game; gray lines join games with the same seed",
          paired_from=0)


if __name__ == "__main__":
    main()
