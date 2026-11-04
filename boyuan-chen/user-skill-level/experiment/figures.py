"""Generate report figures as PNGs into research/figures/.

Usage:
    python figures.py --pilot               # v1 education pilot (out/metrics.csv)
    python figures.py --skill               # v2 skill study (out/scores_skill.csv)

Figures are deterministic re-renders from committed/derived CSVs; rerun after any
re-scoring. Kept dependency-light: matplotlib only, no seaborn/pandas.
"""

import argparse
import csv
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
FIGDIR = HERE.parent / "research" / "figures"

V1_LEVELS = ["no_hs", "hs", "bachelors", "phd"]
V1_LABELS = ["< high school", "high school", "bachelor's", "PhD"]
V2_LEVELS = ["beginner", "hobbyist", "professional", "expert"]


def group_mean(rows, key, metric, flt=lambda r: True):
    g = defaultdict(list)
    for r in rows:
        if flt(r) and r.get(metric) not in ("", None):
            g[r[key]].append(float(r[metric]))
    return {k: st.mean(v) for k, v in g.items()}, {k: len(v) for k, v in g.items()}


def sem(vals):
    return st.stdev(vals) / len(vals) ** 0.5 if len(vals) > 1 else 0.0


def condition_series(rows, metric, levels, channels):
    """mean +/- sem per (level, channel); control returned separately."""
    out = {}
    for ch in channels:
        means, errs = [], []
        for lv in levels:
            vals = [float(r[metric]) for r in rows
                    if r["education_level" if "education_level" in r else "skill_level"] == lv
                    and r["channel"] == ch and r.get(metric) not in ("", None)]
            means.append(st.mean(vals) if vals else float("nan"))
            errs.append(sem(vals) if vals else 0.0)
        out[ch] = (means, errs)
    ctrl = [float(r[metric]) for r in rows
            if r["education_level" if "education_level" in r else "skill_level"] == "control"
            and r.get(metric) not in ("", None)]
    return out, (st.mean(ctrl) if ctrl else None, sem(ctrl) if ctrl else 0.0)


def line_figure(rows, metric, ylabel, title, levels, labels, outpath, note=None):
    series, (ctrl_mean, ctrl_sem) = condition_series(rows, metric, levels,
                                                     ["system", "message"])
    x = range(len(levels))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for ch, marker in (("system", "o"), ("message", "s")):
        means, errs = series[ch]
        ax.errorbar(x, means, yerr=errs, marker=marker, capsize=3,
                    label=f"{ch} channel")
    if ctrl_mean is not None:
        ax.axhline(ctrl_mean, color="gray", linestyle="--", linewidth=1,
                   label="control (no info)")
        ax.axhspan(ctrl_mean - ctrl_sem, ctrl_mean + ctrl_sem,
                   color="gray", alpha=0.15)
    ax.set_xticks(list(x), labels)
    ax.set_xlabel("Stated level")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    if note:
        fig.text(0.01, 0.01, note, fontsize=7, color="gray")
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print("wrote", outpath)


def pilot_figures():
    rows = list(csv.DictReader(open(HERE / "out" / "metrics.csv")))
    note = (f"v1 education pilot, n={len(rows)} responses (partial run, "
            "descriptive only), subject model gpt-oss-20b")
    line_figure(rows, "fk_grade", "Flesch-Kincaid grade level",
                "Reading level of answers vs. stated education (v1 pilot)",
                V1_LEVELS, V1_LABELS, FIGDIR / "v1_pilot_fk_grade.png", note)
    line_figure(rows, "word_count", "Words per answer",
                "Answer length vs. stated education (v1 pilot)",
                V1_LEVELS, V1_LABELS, FIGDIR / "v1_pilot_word_count.png", note)


def skill_figures():
    rows = list(csv.DictReader(open(HERE / "out" / "scores_skill.csv")))
    rows = [r for r in rows if r.get("extraction") in ("labeled", "unlabeled")]
    note = (f"v2 skill study, n={len(rows)} scored responses, "
            "subject model gpt-oss-20b")
    for metric, ylabel, fname in [
        ("pass_rate", "Hidden-test pass rate", "v2_pass_rate.png"),
        ("comment_ratio", "Documentation lines per code line", "v2_comment_ratio.png"),
        ("prose_words", "Prose words around the code", "v2_prose_words.png"),
        ("cyclomatic_mean", "Mean cyclomatic complexity", "v2_cyclomatic.png"),
    ]:
        line_figure(rows, metric, ylabel,
                    f"{ylabel} vs. stated skill (v2)",
                    V2_LEVELS, V2_LEVELS, FIGDIR / fname, note)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--skill", action="store_true")
    args = ap.parse_args()
    FIGDIR.mkdir(exist_ok=True)
    if args.pilot:
        pilot_figures()
    if args.skill:
        skill_figures()
    if not (args.pilot or args.skill):
        ap.error("pass --pilot and/or --skill")


if __name__ == "__main__":
    main()
