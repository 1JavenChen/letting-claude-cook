"""Ablation analyses: does the beginner-vs-expert effect survive when subsets of
the data are removed? Produces forest plots (one per metric), tier heatmaps, and
research/05b_ablations.md.

Each ablation re-estimates the same paired-within-task beginner - expert
contrast used by stats.py (task means pooled over channels/reps, Wilcoxon p,
bootstrap 95% CI), so numbers are directly comparable to the pre-registered
verdicts. Ablations are exploratory and uncorrected.

Usage: python ablations.py   (reads out/scores_skill.csv [+ judge json])
"""

import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from stats import (DEFAULT_JUDGE, DEFAULT_SCORES, DEFAULT_SEED, LEVELS,  # noqa: E402
                   bootstrap_ci, load_scores, level_task_means, mean,
                   merge_judge, paired_diffs, wilcoxon_signed_rank)

FIGDIR = HERE.parent / "research" / "figures" / "ablations"
REPORT = HERE.parent / "research" / "05b_ablations.md"
N_BOOT = 2000

METRICS = [
    ("pass_rate", "Hidden-test pass rate (H3)", (0.03, 0.06)),
    ("prose_words", "Prose words around the code (H1s)", None),
    ("advanced_idioms", "Advanced Python idioms (H2s)", None),
    ("comment_ratio", "Documentation lines per code line (H1)", None),
    ("validation_checks", "Validation/error-handling constructs (H4)", None),
]
JUDGE_METRICS = [
    ("requirements_met_prop", "Requirements met, blind judge (H5)", (None, 0.05)),
    ("condescension", "Condescension 1-5, blind judge (H8)", None),
]


def control_pass_by_task(rows):
    acc = defaultdict(list)
    for r in rows:
        if r["skill_level"] == "control" and r.get("pass_rate") is not None:
            acc[r["task_id"]].append(r["pass_rate"])
    return {t: mean(v) for t, v in acc.items()}


def build_ablations(rows):
    cp = control_pass_by_task(rows)
    band = {t for t, v in cp.items() if 0.3 <= v <= 0.9}
    not_ceiling = {t for t, v in cp.items() if v < 0.95}
    tiers = {r["task_id"]: r["difficulty"] for r in rows}
    return [
        ("All tasks (pre-registered)", lambda r: True),
        ("Tier A only (entry-level)", lambda r: tiers[r["task_id"]] == "A"),
        ("Tier B only (intermediate)", lambda r: tiers[r["task_id"]] == "B"),
        ("Tier C only (hard)", lambda r: tiers[r["task_id"]] == "C"),
        ("Exclude Tier A", lambda r: tiers[r["task_id"]] != "A"),
        (f"Solvability band, control 0.3-0.9 (n={len(band)} tasks)",
         lambda r: r["task_id"] in band),
        (f"Not at ceiling, control < 0.95 (n={len(not_ceiling)} tasks)",
         lambda r: r["task_id"] in not_ceiling),
        ("System-prompt channel only", lambda r: r["channel"] in ("system", "none")),
        ("Self-disclosure channel only", lambda r: r["channel"] in ("message", "none")),
        ("Persona-leaking responses excluded", lambda r: not r.get("leakage")),
        ("Hardening (security) tasks only", lambda r: r["hardening"]),
        ("Replicate 1 only", lambda r: r["rep"] == 1),
        ("Replicate 2 only", lambda r: r["rep"] == 2),
        ("Replicate 3 only", lambda r: r["rep"] == 3),
    ]


def contrast(rows, metric, seed=DEFAULT_SEED):
    lm = level_task_means(rows, metric)
    diffs = paired_diffs(lm, "beginner", "expert")
    if not diffs:
        return None
    point, lo, hi = bootstrap_ci(diffs, N_BOOT, seed)
    p = wilcoxon_signed_rank(diffs)["p"]
    return {"n": len(diffs), "effect": point, "lo": lo, "hi": hi, "p": p}


def forest_plot(results, label, margin, outpath):
    names = [r[0] for r in results]
    effects = [r[1]["effect"] for r in results]
    lo = [r[1]["effect"] - r[1]["lo"] for r in results]
    hi = [r[1]["hi"] - r[1]["effect"] for r in results]
    y = list(range(len(names)))[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 0.42 * len(names) + 1.6))
    if margin and margin[1]:
        ax.axvspan(-margin[1], margin[1], color="green", alpha=0.08,
                   label=f"equivalence margin +/-{margin[1]:g}")
    ax.axvline(0, color="gray", linewidth=1)
    colors = ["tab:red" if r[1]["p"] < 0.05 else "tab:blue" for r in results]
    ax.errorbar(effects, y, xerr=[lo, hi], fmt="none", ecolor="black",
                capsize=3, linewidth=1)
    ax.scatter(effects, y, c=colors, zorder=3, s=36)
    ax.set_yticks(y, [f"{n}  (n={r['n']}, p={r['p']:.2g})" for n, r in results],
                  fontsize=8)
    ax.set_xlabel("Beginner minus expert, paired within task (95% bootstrap CI)")
    ax.set_title(f"Ablations: {label}", fontsize=10)
    ax.scatter([], [], c="tab:red", label="p < .05 (uncorrected)")
    ax.scatter([], [], c="tab:blue", label="p >= .05")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def tier_heatmap(rows, metric, label, outpath):
    levels = ["control"] + LEVELS
    tiers = ["A", "B", "C"]
    acc = defaultdict(list)
    for r in rows:
        if r.get(metric) is not None:
            acc[(r["difficulty"], r["skill_level"])].append(r[metric])
    grid = [[mean(acc[(t, lv)]) if acc[(t, lv)] else float("nan") for lv in levels]
            for t in tiers]
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    im = ax.imshow(grid, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(levels)), levels)
    ax.set_yticks(range(len(tiers)), [f"Tier {t}" for t in tiers])
    for i in range(len(tiers)):
        for j in range(len(levels)):
            ax.text(j, i, f"{grid[i][j]:.3f}", ha="center", va="center",
                    color="white", fontsize=9)
    ax.set_title(f"{label} by difficulty tier x stated skill", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    return grid, levels, tiers


def tier_lines(rows, metric, label, outpath):
    acc = defaultdict(list)
    for r in rows:
        if r.get(metric) is not None:
            acc[(r["difficulty"], r["skill_level"])].append(r[metric])
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = range(len(LEVELS))
    for t, marker in zip("ABC", "osD"):
        ax.plot(list(x), [mean(acc[(t, lv)]) for lv in LEVELS], marker=marker,
                label=f"Tier {t}")
        ax.axhline(mean(acc[(t, "control")]), linestyle=":", linewidth=0.8,
                   color=ax.lines[-1].get_color())
    ax.set_xticks(list(x), LEVELS)
    ax.set_xlabel("Stated skill level (dotted = that tier's control)")
    ax.set_ylabel(label)
    ax.set_title(f"{label} vs. stated skill, by task difficulty", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def main():
    FIGDIR.mkdir(parents=True, exist_ok=True)
    rows = load_scores(DEFAULT_SCORES)
    rows = [r for r in rows if r.get("pass_rate") is not None]
    have_judge = merge_judge(rows, DEFAULT_JUDGE)
    n_judged = sum(1 for r in rows if r.get("requirements_met_prop") is not None)
    metrics = METRICS + (JUDGE_METRICS if have_judge else [])
    ablations = build_ablations(rows)

    lines = ["# Ablation analyses (exploratory, uncorrected)", "",
             f"Data: {len(rows)} scored responses; judge scores merged for "
             f"{n_judged} responses{' (PARTIAL judge pass)' if 0 < n_judged < len(rows) else ''}.",
             "Each row re-estimates the pre-registered beginner - expert contrast "
             "(paired within task, Wilcoxon p, bootstrap 95% CI) on a subset of the data. "
             "Red points in the forest plots are p < .05 uncorrected; the pre-registered "
             "verdicts in `05_analysis_automated_metrics.md` remain the confirmatory results.",
             ""]
    flips = []
    for metric, label, margin in metrics:
        results = []
        for name, flt in ablations:
            sub = [r for r in rows if flt(r)]
            res = contrast(sub, metric)
            if res and res["n"] >= 3:
                results.append((name, res))
        if not results:
            continue
        fname = f"forest_{metric}.png"
        forest_plot(results, label, margin, FIGDIR / fname)
        lines += [f"## {label}", "", f"![{label}](figures/ablations/{fname})", "",
                  "| Ablation | tasks | effect (beg - exp) | 95% CI | p |", "|---|---|---|---|---|"]
        base_sig = results[0][1]["p"] < 0.05
        for name, r in results:
            eq = ""
            if margin and margin[0] is not None:
                eq = " **EQUIV MET**" if (abs(r["effect"]) < margin[0]
                                         and -margin[1] < r["lo"] and r["hi"] < margin[1]) else " (equiv. not met)"
            elif margin and margin[1]:
                eq = " **EQUIV MET**" if (-margin[1] < r["lo"] and r["hi"] < margin[1]) else " (equiv. not met)"
            lines.append(f"| {name} | {r['n']} | {r['effect']:+.4f}{eq} | "
                         f"[{r['lo']:+.4f}, {r['hi']:+.4f}] | {r['p']:.3g} |")
            if (r["p"] < 0.05) != base_sig and name != results[0][0]:
                flips.append((label, name, r))
        lines.append("")

    for metric, label, _ in metrics[:3]:
        tier_heatmap(rows, metric, label, FIGDIR / f"heatmap_{metric}.png")
        tier_lines(rows, metric, label, FIGDIR / f"tiers_{metric}.png")
    lines += ["## Difficulty-tier breakdowns", ""]
    for metric, label, _ in metrics[:3]:
        lines += [f"![{label} heatmap](figures/ablations/heatmap_{metric}.png)",
                  f"![{label} by tier](figures/ablations/tiers_{metric}.png)", ""]

    lines += ["## Significance flips relative to the all-tasks estimate", ""]
    if flips:
        for label, name, r in flips:
            lines.append(f"- **{label}** — `{name}`: effect {r['effect']:+.4f} "
                         f"[{r['lo']:+.4f}, {r['hi']:+.4f}], p={r['p']:.3g} "
                         f"(all-tasks estimate was {'significant' if r['p'] >= 0.05 else 'non-significant'}).")
    else:
        lines.append("- None: every ablation agrees in significance with the all-tasks estimate.")
    REPORT.write_text("\n".join(lines) + "\n")
    print(f"wrote {REPORT} and {len(list(FIGDIR.glob('*.png')))} figures to {FIGDIR}")


if __name__ == "__main__":
    main()
