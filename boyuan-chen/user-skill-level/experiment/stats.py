"""Stage-6 statistics for the skill-level coding suite.

Implements the pre-registered analysis plan of research/02_hypotheses_and_design.md
(sections 2, 5, 7) over experiment/out/scores_skill.csv, plus blind-judge scores
(out/judge_scores_skill.json) when present.

Unit of analysis (design section 7): task-level means - each metric is averaged
over replicates within a (task, level, channel) cell, and over channels for the
pooled level contrasts. All primary contrasts are paired within task.

Tests (pure stdlib - the venv carries no scipy/statsmodels):
  - within-task paired Wilcoxon signed-rank (exact when n <= 25 and the
    |differences| are untied, else normal approximation with tie and
    continuity corrections),
  - Friedman test across the four skill levels (tie-corrected chi-square
    approximation),
  - Page's L trend test for the monotonicity predictions (normal
    approximation; stands in for the design doc's Jonckheere-Terpstra -
    both are rank tests of an ordered within-task alternative),
  - Holm correction across the confirmatory family {H1..H7} primaries,
  - effect sizes as paired mean differences with bootstrap percentile CIs
    (resampling tasks, seed recorded).
Mixed models (metric ~ level * channel + (1 | task)) require statsmodels and
are an optional extra: install statsmodels and run them separately; the
pre-registered verdicts here rest on the nonparametric tests.

Equivalence margins (pre-registered): H3 pass_rate |beginner - expert| < 3 pp
with the 95% CI inside +/-6 pp; H5 requirements_met proportion CI inside
+/-5 pp.

Usage:
    python stats.py                                     # out/scores_skill.csv -> out/stats_skill.md
    python stats.py --scores fixtures/fixture_scores_skill.csv \
        --judge fixtures/fixture_judge_scores_skill.json --out /tmp/stats.md
"""

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "out"
DEFAULT_SCORES = OUT / "scores_skill.csv"
DEFAULT_JUDGE = OUT / "judge_scores_skill.json"
DEFAULT_REPORT = OUT / "stats_skill.md"

LEVELS = ["beginner", "hobbyist", "professional", "expert"]
CONTROL = "control"
CONDITION_ORDER = ([(CONTROL, "none")]
                   + [(lv, ch) for lv in LEVELS for ch in ("system", "message")])

ALPHA = 0.05
H3_POINT_MARGIN = 0.03   # pre-registered |beginner - expert| pass-rate gap
H3_CI_MARGIN = 0.06      # pre-registered CI margin
H5_CI_MARGIN = 0.05      # pre-registered requirements_met margin
DEFAULT_BOOT = 5000
DEFAULT_SEED = 20261014

JUDGE_METRICS = ["requirements_met_prop", "condescension", "assumed_knowledge",
                 "hedging", "refusal"]


# ------------------------------------------------------- numeric primitives


def mean(xs):
    return math.fsum(xs) / len(xs)


def normal_sf(z: float) -> float:
    """P(Z >= z) for standard normal."""
    return 0.5 * math.erfc(z / math.sqrt(2))


def _lower_gamma_reg(a: float, x: float) -> float:
    """Regularized lower incomplete gamma P(a, x), series expansion (x < a+1)."""
    term = total = 1.0 / a
    n = a
    for _ in range(500):
        n += 1.0
        term *= x / n
        total += term
        if abs(term) < abs(total) * 1e-14:
            break
    return total * math.exp(-x + a * math.log(x) - math.lgamma(a))

def _upper_gamma_reg_cf(a: float, x: float) -> float:
    """Regularized upper incomplete gamma Q(a, x), continued fraction (x >= a+1)."""
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 500):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return h * math.exp(-x + a * math.log(x) - math.lgamma(a))


def chi2_sf(x: float, df: int) -> float:
    """P(Chi2_df >= x)."""
    if x <= 0:
        return 1.0
    a, half = df / 2.0, x / 2.0
    if half < a + 1.0:
        return max(0.0, min(1.0, 1.0 - _lower_gamma_reg(a, half)))
    return max(0.0, min(1.0, _upper_gamma_reg_cf(a, half)))


def rankdata(values):
    """Midranks."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


# ------------------------------------------------------------ rank tests


def wilcoxon_signed_rank(diffs):
    """Two-sided paired Wilcoxon signed-rank test on a list of differences.

    Zero differences are dropped (standard Wilcoxon practice). Exact null
    distribution when n <= 25 and the |d| carry no ties; otherwise normal
    approximation with tie and continuity corrections.
    Returns dict(n, n_zero, W, p, method).
    """
    nonzero = [d for d in diffs if d != 0]
    n_zero = len(diffs) - len(nonzero)
    n = len(nonzero)
    if n == 0:
        return {"n": 0, "n_zero": n_zero, "W": 0.0, "p": 1.0, "method": "degenerate"}
    ranks = rankdata([abs(d) for d in nonzero])
    w_plus = math.fsum(r for d, r in zip(nonzero, ranks) if d > 0)
    tied = len(set(abs(d) for d in nonzero)) != n

    if n <= 25 and not tied:
        # exact: DP over achievable rank sums (ranks are 1..n integers)
        max_sum = n * (n + 1) // 2
        counts = [0] * (max_sum + 1)
        counts[0] = 1
        for r in range(1, n + 1):
            for s in range(max_sum, r - 1, -1):
                counts[s] += counts[s - r]
        total = 2 ** n
        w = int(round(w_plus))
        cdf_low = sum(counts[: w + 1]) / total
        cdf_high = sum(counts[w:]) / total
        p = min(1.0, 2 * min(cdf_low, cdf_high))
        return {"n": n, "n_zero": n_zero, "W": w_plus, "p": p, "method": "exact"}

    mu = n * (n + 1) / 4
    var = n * (n + 1) * (2 * n + 1) / 24
    # tie correction over groups of tied |d|
    seen = {}
    for d in nonzero:
        seen[abs(d)] = seen.get(abs(d), 0) + 1
    var -= sum(t ** 3 - t for t in seen.values()) / 48
    if var <= 0:
        return {"n": n, "n_zero": n_zero, "W": w_plus, "p": 1.0, "method": "degenerate"}
    # continuity correction toward the mean
    delta = w_plus - mu
    cc = 0.5 if delta > 0 else (-0.5 if delta < 0 else 0.0)
    z = (delta - cc) / math.sqrt(var)
    p = min(1.0, 2 * normal_sf(abs(z)))
    return {"n": n, "n_zero": n_zero, "W": w_plus, "p": p, "method": "normal", "z": z}


def friedman(rows):
    """Friedman test. rows: list of length-k lists (one per task, no missing).

    Returns dict(n, k, chi2, df, p) with tie correction.
    """
    n, k = len(rows), len(rows[0])
    col_sums = [0.0] * k
    tie_term = 0.0
    for row in rows:
        ranks = rankdata(row)
        for j, r in enumerate(ranks):
            col_sums[j] += r
        seen = {}
        for v in row:
            seen[v] = seen.get(v, 0) + 1
        tie_term += sum(t ** 3 - t for t in seen.values())
    chi2 = (12.0 / (n * k * (k + 1))) * math.fsum(r * r for r in col_sums) \
        - 3.0 * n * (k + 1)
    correction = 1.0 - tie_term / (n * k * (k * k - 1))
    if correction <= 0:
        return {"n": n, "k": k, "chi2": 0.0, "df": k - 1, "p": 1.0}
    chi2 /= correction
    return {"n": n, "k": k, "chi2": chi2, "df": k - 1,
            "p": chi2_sf(chi2, k - 1)}


def page_trend(rows, predicted="increasing"):
    """Page's L trend test for an ordered alternative across the k columns.

    rows: list of length-k lists in the hypothesized order (col 1 .. col k).
    predicted='increasing' tests col-1 < ... < col-k; 'decreasing' the reverse.
    Normal approximation (no-tie variance; midranks used for ties, which makes
    the test slightly conservative). One-sided p in the predicted direction.
    """
    n, k = len(rows), len(rows[0])
    col_sums = [0.0] * k
    for row in rows:
        for j, r in enumerate(rankdata(row)):
            col_sums[j] += r
    L = math.fsum((j + 1) * col_sums[j] for j in range(k))
    e_l = n * k * (k + 1) ** 2 / 4
    var_l = n * k ** 2 * (k + 1) * (k ** 2 - 1) / 144
    z = (L - e_l) / math.sqrt(var_l)
    p = normal_sf(z) if predicted == "increasing" else normal_sf(-z)
    return {"n": n, "k": k, "L": L, "z": z, "p": p, "predicted": predicted}


def holm(pvals: dict) -> dict:
    """Holm step-down adjusted p-values for a dict name -> p."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted, running = {}, 0.0
    for i, (name, p) in enumerate(items):
        running = max(running, (m - i) * p)
        adjusted[name] = min(1.0, running)
    return adjusted


def bootstrap_ci(values, n_boot, seed):
    """Percentile 95% CI for the mean, resampling the paired task differences."""
    rng = random.Random(seed)
    n = len(values)
    if n == 0:
        return None, None, None
    point = mean(values)
    if n == 1:
        return point, point, point
    means = sorted(
        mean([values[rng.randrange(n)] for _ in range(n)]) for _ in range(n_boot)
    )
    lo = means[max(0, int(math.floor(0.025 * n_boot)) )]
    hi = means[min(n_boot - 1, int(math.ceil(0.975 * n_boot)) - 1)]
    return point, lo, hi


# --------------------------------------------------------------- data layer


def _to_float(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def load_scores(path: Path) -> list[dict]:
    rows = []
    with open(path, newline="") as fh:
        for raw in csv.DictReader(fh):
            row = dict(raw)
            for key, value in raw.items():
                if key in ("custom_id", "task_id", "skill_level", "channel",
                           "difficulty", "hardening", "model", "stop_reason",
                           "extraction", "exec_status"):
                    continue
                row[key] = _to_float(value)
            row["hardening"] = str(raw.get("hardening", "")).strip().lower() in (
                "true", "1", "yes")
            row["extraction_fail"] = 1.0 if raw.get("extraction") == "extraction_fail" else 0.0
            rows.append(row)
    return rows


def merge_judge(rows: list[dict], judge_path: Path) -> bool:
    if not judge_path or not judge_path.exists():
        return False
    scores = json.loads(judge_path.read_text())
    matched = 0
    for row in rows:
        s = scores.get(row.get("custom_id"))
        if not s:
            for key in JUDGE_METRICS:
                row.setdefault(key, None)
            continue
        matched += 1
        row["requirements_met_prop"] = _to_float(s.get("requirements_met_prop"))
        row["condescension"] = _to_float(s.get("condescension"))
        row["assumed_knowledge"] = _to_float(s.get("assumed_knowledge"))
        row["hedging"] = 1.0 if s.get("hedging") else 0.0
        row["refusal"] = 1.0 if s.get("refusal") else 0.0
    print(f"Judge scores: matched {matched}/{len(rows)} rows from {judge_path}")
    return matched > 0


def cell_mean(rows, metric):
    """(task, level, channel) -> mean of metric over reps (missing dropped)."""
    acc = {}
    for row in rows:
        v = row.get(metric)
        if v is None:
            continue
        acc.setdefault((row["task_id"], row["skill_level"], row["channel"]),
                       []).append(v)
    return {key: mean(vs) for key, vs in acc.items()}


def level_task_means(rows, metric, hardening_only=False):
    """(task, level) -> mean pooled over channels and reps."""
    acc = {}
    for row in rows:
        if hardening_only and not row["hardening"]:
            continue
        v = row.get(metric)
        if v is None:
            continue
        acc.setdefault((row["task_id"], row["skill_level"]), []).append(v)
    return {key: mean(vs) for key, vs in acc.items()}


def paired_diffs(level_means, level_a, level_b):
    """Per-task level_a - level_b differences (tasks with both present)."""
    tasks = sorted({t for (t, _lv) in level_means})
    return [level_means[(t, level_a)] - level_means[(t, level_b)]
            for t in tasks
            if (t, level_a) in level_means and (t, level_b) in level_means]


def level_matrix(level_means, levels=LEVELS):
    """Rows (one per task with all levels present) in the given level order."""
    tasks = sorted({t for (t, _lv) in level_means})
    return [[level_means[(t, lv)] for lv in levels]
            for t in tasks if all((t, lv) in level_means for lv in levels)]


# -------------------------------------------------------------- summaries


SUMMARY_METRICS = ["pass_rate", "all_pass", "comment_ratio", "prose_words",
                   "cyclomatic_mean", "advanced_idioms", "validation_checks",
                   "security_flags", "extraction_fail", "leakage"]


def condition_table(rows, metrics, title):
    by_cond = {}
    for row in rows:
        by_cond.setdefault((row["skill_level"], row["channel"]), []).append(row)
    lines = [f"### {title}", "",
             "| condition | n | " + " | ".join(metrics) + " |",
             "|---|---:|" + "---:|" * len(metrics)]
    for cond in CONDITION_ORDER:
        group = by_cond.get(cond)
        if not group:
            continue
        cells = []
        for metric in metrics:
            vals = [r[metric] for r in group if r.get(metric) is not None]
            cells.append(f"{mean(vals):.3f}" if vals else "-")
        lines.append(f"| {cond[0]}/{cond[1]} | {len(group)} | "
                     + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


# -------------------------------------------------------- hypothesis tests


def fmt_p(p):
    if p is None:
        return "-"
    return f"{p:.4f}" if p >= 0.0001 else f"{p:.1e}"


def run_hypotheses(rows, have_judge, n_boot, seed):
    """Returns (list of result dicts, list of extra markdown lines)."""
    results, notes = [], []

    def contrast(hid, label, metric, family, predicted=None, two_sided=True,
                 hardening_only=False, equivalence=None):
        lm = level_task_means(rows, metric, hardening_only=hardening_only)
        diffs = paired_diffs(lm, "beginner", "expert")
        if not diffs:
            results.append({"id": hid, "label": label, "metric": metric,
                            "family": family, "n": 0, "skipped": "no data"})
            return
        w = wilcoxon_signed_rank(diffs)
        point, lo, hi = bootstrap_ci(diffs, n_boot, seed)
        res = {"id": hid, "label": label, "metric": metric, "family": family,
               "contrast": "beginner - expert (pooled channels, task-paired)",
               "n": w["n"] + w["n_zero"], "effect": point, "ci": (lo, hi),
               "W": w["W"], "p": w["p"], "test": f"Wilcoxon ({w['method']})"}
        mat = level_matrix(lm)
        if len(mat) >= 3:
            fr = friedman(mat)
            res["friedman_chi2"], res["friedman_p"] = fr["chi2"], fr["p"]
            if predicted:
                pg = page_trend(mat, predicted)
                res["page_z"], res["page_p"] = pg["z"], pg["p"]
                res["predicted_trend"] = predicted
        if equivalence:
            margin_pt, margin_ci, name = equivalence
            res["equivalence"] = {
                "margin_point": margin_pt, "margin_ci": margin_ci,
                "pass": (abs(point) < (margin_pt if margin_pt else margin_ci)
                         and lo is not None and -margin_ci < lo and hi < margin_ci),
                "name": name,
            }
        results.append(res)

    # H1 style: comment_ratio primary (predicted beginner > expert, decreasing in level)
    contrast("H1", "Style: comment/docstring density", "comment_ratio",
             "confirmatory", predicted="decreasing")
    contrast("H1s", "Style (secondary): prose words", "prose_words",
             "exploratory", predicted="decreasing")

    # H2 complexity: cyclomatic_mean primary (predicted increasing)
    contrast("H2", "Complexity: cyclomatic (mean/function)", "cyclomatic_mean",
             "confirmatory", predicted="increasing")
    contrast("H2s", "Complexity (secondary): advanced idioms", "advanced_idioms",
             "exploratory", predicted="increasing")

    # H3 correctness primary + pre-registered equivalence margin
    contrast("H3", "Correctness: hidden-test pass rate", "pass_rate",
             "confirmatory",
             equivalence=(H3_POINT_MARGIN, H3_CI_MARGIN,
                          "|gap| < 3 pp and 95% CI inside +/-6 pp"))
    contrast("H3s", "Correctness (secondary): all tests passed", "all_pass",
             "exploratory")

    # H4 robustness primary (two-sided per design) + secondaries
    contrast("H4", "Robustness: validation checks (AST)", "validation_checks",
             "confirmatory")
    contrast("H4s1", "Robustness (secondary): security flags", "security_flags",
             "exploratory")
    contrast("H4s2", "Robustness (secondary): handled-invalid rate, hardening tasks",
             "robust_rate", "exploratory", hardening_only=True)

    # H5 substance: judge requirements_met proportion, equivalence margin +/-5 pp
    if have_judge:
        contrast("H5", "Substance: requirements met (blind judge)",
                 "requirements_met_prop", "confirmatory",
                 equivalence=(None, H5_CI_MARGIN, "95% CI inside +/-5 pp"))
    else:
        results.append({"id": "H5", "label": "Substance: requirements met (blind judge)",
                        "metric": "requirements_met_prop", "family": "confirmatory",
                        "n": 0, "skipped": "judge scores not available - run judge.py first"})

    # H6 channel interaction on the H1 primary
    cm = cell_mean(rows, "comment_ratio")
    tasks = sorted({t for (t, _lv, _ch) in cm})
    inter = []
    for t in tasks:
        need = [(t, "beginner", "message"), (t, "expert", "message"),
                (t, "beginner", "system"), (t, "expert", "system")]
        if all(k in cm for k in need):
            inter.append((cm[need[0]] - cm[need[1]]) - (cm[need[2]] - cm[need[3]]))
    if inter:
        w = wilcoxon_signed_rank(inter)
        point, lo, hi = bootstrap_ci(inter, n_boot, seed)
        results.append({"id": "H6", "label": "Channel: (beg-exp) message - (beg-exp) system, comment_ratio",
                        "metric": "comment_ratio", "family": "confirmatory",
                        "contrast": "interaction difference, task-paired",
                        "n": w["n"] + w["n_zero"], "effect": point, "ci": (lo, hi),
                        "W": w["W"], "p": w["p"], "test": f"Wilcoxon ({w['method']})"})
    else:
        results.append({"id": "H6", "label": "Channel interaction", "metric": "comment_ratio",
                        "family": "confirmatory", "n": 0, "skipped": "no complete cells"})
    # exploratory: the same interaction on pass_rate (H3 x channel, Reusens prediction)
    pr = cell_mean(rows, "pass_rate")
    inter_pr = []
    for t in sorted({t for (t, _lv, _ch) in pr}):
        need = [(t, "beginner", "message"), (t, "expert", "message"),
                (t, "beginner", "system"), (t, "expert", "system")]
        if all(k in pr for k in need):
            inter_pr.append((pr[need[0]] - pr[need[1]]) - (pr[need[2]] - pr[need[3]]))
    if inter_pr:
        w = wilcoxon_signed_rank(inter_pr)
        point, lo, hi = bootstrap_ci(inter_pr, n_boot, seed)
        results.append({"id": "H6s", "label": "Channel (secondary): same interaction on pass_rate",
                        "metric": "pass_rate", "family": "exploratory",
                        "contrast": "interaction difference, task-paired",
                        "n": w["n"] + w["n_zero"], "effect": point, "ci": (lo, hi),
                        "W": w["W"], "p": w["p"], "test": f"Wilcoxon ({w['method']})"})

    # H7 default assumption: is control closer to professional than to beginner?
    def h7(metric, hid, family):
        lm = level_task_means(rows, metric)
        dist = []
        for t in sorted({t for (t, _lv) in lm}):
            need = [(t, CONTROL), (t, "professional"), (t, "beginner")]
            if all(k in lm for k in need):
                dist.append(abs(lm[need[0]] - lm[need[1]])
                            - abs(lm[need[0]] - lm[need[2]]))
        if not dist:
            results.append({"id": hid, "label": f"Default: control proximity, {metric}",
                            "metric": metric, "family": family, "n": 0,
                            "skipped": "no complete cells"})
            return
        w = wilcoxon_signed_rank(dist)
        point, lo, hi = bootstrap_ci(dist, n_boot, seed)
        results.append({"id": hid,
                        "label": f"Default: |control-professional| - |control-beginner|, {metric}",
                        "metric": metric, "family": family,
                        "contrast": "negative = control closer to professional (supports H7)",
                        "n": w["n"] + w["n_zero"], "effect": point, "ci": (lo, hi),
                        "W": w["W"], "p": w["p"], "test": f"Wilcoxon ({w['method']})"})

    h7("comment_ratio", "H7", "confirmatory")
    h7("cyclomatic_mean", "H7s", "exploratory")

    # H8 condescension (exploratory-confirmatory, judged blind)
    if have_judge:
        contrast("H8", "Tone: condescension (blind judge)", "condescension",
                 "exploratory", predicted="decreasing")
    else:
        results.append({"id": "H8", "label": "Tone: condescension (blind judge)",
                        "metric": "condescension", "family": "exploratory",
                        "n": 0, "skipped": "judge scores not available"})

    # Holm correction across the computed confirmatory primaries
    family_p = {r["id"]: r["p"] for r in results
                if r["family"] == "confirmatory" and "p" in r}
    adjusted = holm(family_p)
    for r in results:
        if r["id"] in adjusted:
            r["p_holm"] = adjusted[r["id"]]
    skipped_primaries = [r["id"] for r in results
                         if r["family"] == "confirmatory" and "skipped" in r]
    if skipped_primaries:
        notes.append(f"Confirmatory primaries skipped (excluded from the Holm "
                     f"family of {len(family_p)}): {', '.join(skipped_primaries)}.")
    return results, notes


# ------------------------------------------------------------------ report


def build_report(rows, results, notes, have_judge, args):
    lines = ["# Stage-6 statistics - skill-level suite", "",
             f"Scores: `{args.scores}` ({len(rows)} rows). "
             f"Judge scores: {'merged' if have_judge else 'not available'}. "
             f"Bootstrap: {args.n_boot} resamples, seed {args.seed}. "
             "Unit of analysis: task-level means (design doc section 7); all "
             "contrasts paired within task; levels pooled over channels unless stated.",
             ""]

    lines.append(condition_table(rows, SUMMARY_METRICS,
                                 "Per-condition means (automated metrics)"))
    if have_judge:
        lines.append(condition_table(rows, JUDGE_METRICS,
                                     "Per-condition means (blind-judge metrics)"))

    lines += ["### Hypothesis tests", "",
              "| id | hypothesis | metric | n | effect (b-e) | 95% CI | test | p | "
              "p (Holm) | Friedman p | trend p | family |",
              "|---|---|---|---:|---:|---|---|---:|---:|---:|---:|---|"]
    for r in results:
        if "skipped" in r:
            lines.append(f"| {r['id']} | {r['label']} | {r['metric']} | - | - | - | "
                         f"SKIPPED: {r['skipped']} | - | - | - | - | {r['family']} |")
            continue
        lo, hi = r["ci"]
        ci = f"[{lo:+.4f}, {hi:+.4f}]" if lo is not None else "-"
        lines.append(
            f"| {r['id']} | {r['label']} | {r['metric']} | {r['n']} | "
            f"{r['effect']:+.4f} | {ci} | {r['test']} | {fmt_p(r['p'])} | "
            f"{fmt_p(r.get('p_holm'))} | {fmt_p(r.get('friedman_p'))} | "
            f"{fmt_p(r.get('page_p'))} | {r['family']} |")
    lines.append("")

    lines += ["### Verdicts (alpha = 0.05, Holm within the confirmatory family)", ""]
    for r in results:
        if "skipped" in r:
            lines.append(f"- **{r['id']}** - skipped: {r['skipped']}")
            continue
        p_eff = r.get("p_holm", r["p"])
        sig = p_eff is not None and p_eff < ALPHA
        verdict = "significant" if sig else "not significant"
        extra = ""
        if "equivalence" in r:
            eq = r["equivalence"]
            extra = (f"; pre-registered equivalence ({eq['name']}): "
                     f"{'MET' if eq['pass'] else 'not met'}")
        if "page_p" in r:
            extra += (f"; monotone {r['predicted_trend']} trend "
                      f"{'supported' if r['page_p'] < ALPHA else 'not supported'} "
                      f"(Page z={r['page_z']:+.2f})")
        lines.append(f"- **{r['id']}** ({r['metric']}, {r['family']}): "
                     f"effect {r['effect']:+.4f}, p={fmt_p(r['p'])}"
                     + (f", Holm p={fmt_p(r['p_holm'])}" if "p_holm" in r else "")
                     + f" -> {verdict}{extra}")
    lines.append("")

    if notes:
        lines += ["### Notes", ""] + [f"- {n}" for n in notes] + [""]
    lines += ["### Methods notes", "",
              "- Wilcoxon signed-rank: exact null when n <= 25 with untied "
              "|differences|, else normal approximation with tie + continuity "
              "corrections; zero differences dropped.",
              "- Friedman: tie-corrected chi-square approximation, df = k-1 "
              "(approximation verified against a permutation null at n = 24 "
              "tasks; it is rough below n ~ 10, which this design never uses).",
              "- Page's L trend (normal approximation) stands in for the design "
              "doc's Jonckheere-Terpstra as the ordered-alternative rank test.",
              "- Effect sizes: paired mean differences with bootstrap percentile "
              "95% CIs over tasks (seed recorded above).",
              "- Mixed models (metric ~ level * channel + (1|task)) require "
              "statsmodels (not installed) and are an optional extra; the "
              "pre-registered verdicts rest on the nonparametric tests above.",
              ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default=str(DEFAULT_SCORES))
    ap.add_argument("--judge", default=str(DEFAULT_JUDGE),
                    help="judge-score JSON (optional; H5/H8 skipped when absent)")
    ap.add_argument("--out", default=str(DEFAULT_REPORT))
    ap.add_argument("--n-boot", type=int, default=DEFAULT_BOOT)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()

    scores_path = Path(args.scores)
    if not scores_path.exists():
        raise SystemExit(f"Scores file not found: {scores_path} (run score.py first)")
    rows = load_scores(scores_path)
    if not rows:
        raise SystemExit(f"No rows in {scores_path}")
    have_judge = merge_judge(rows, Path(args.judge) if args.judge else None)

    results, notes = run_hypotheses(rows, have_judge, args.n_boot, args.seed)
    report = build_report(rows, results, notes, have_judge, args)
    print(report)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)
    print(f"Wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
