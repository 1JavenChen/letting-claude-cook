"""Generate SYNTHETIC (toy) stage-5/6 outputs for a full pipeline dress rehearsal.

Every output file is suffixed _TOY and lives under fixtures/dryrun/. The planted
effects are arbitrary choices to exercise the stats machinery; they are NOT data
and support no conclusions about any model.

Planted pattern ("tailoring, not rationing"):
  - pass_rate: no real level effect (null; should land inside H3 equivalence margin)
  - comment_ratio, prose_words, condescension: decrease with stated skill
  - type hints, advanced idioms, assumed_knowledge: increase with stated skill
  - control behaves like the professional condition (H7 pattern)

Usage: python fixtures/make_dryrun_data.py   (writes fixtures/dryrun/)
"""

import csv
import json
import random
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "dryrun"
rng = random.Random(424242)

LEVELS = {"beginner": 0, "hobbyist": 1, "professional": 2, "expert": 3, "control": 2.1}
CHANNELS = {"control": ["none"], "_": ["system", "message"]}

tasks = json.loads((HERE.parent / "tasks.json").read_text())

COLS = ("custom_id,task_id,skill_level,channel,rep,difficulty,hardening,model,"
        "stop_reason,output_tokens,extraction,exec_status,n_tests,n_passed,pass_rate,"
        "all_pass,n_robust,n_robust_passed,robust_rate,parse_ok,loc,comment_lines,"
        "docstring_lines,comment_ratio,has_docstring,n_functions,type_hints,try_count,"
        "raise_count,isinstance_count,validation_checks,cyclomatic_total,"
        "cyclomatic_mean,max_nesting,advanced_idioms,identifier_len_mean,"
        "security_flags,response_chars,n_code_blocks,prose_words,md_headers,"
        "md_bullets,leakage").split(",")


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def make_row(task, level, channel, rep, task_bias):
    s = LEVELS[level]  # 0..3 skill score; control ~ professional
    n_tests = len(task.get("tests", [])) or rng.randint(15, 21)
    # Null level effect on correctness; task difficulty drives it.
    p = clamp(task_bias + rng.gauss(0, 0.08), 0.05, 1.0)
    n_passed = round(p * n_tests)
    loc = max(5, round(rng.gauss(28 + 2 * s, 6)))
    comment_lines = max(0, round(rng.gauss(6 - 1.4 * s, 1.5)))
    prose = max(20, round(rng.gauss(420 - 80 * s, 60)))
    hard = task.get("hardening", False)
    n_robust = rng.randint(3, 5) if hard else 0
    row = {
        "custom_id": f"{task['id']}_{level}_{channel}_r{rep}",
        "task_id": task["id"], "skill_level": level, "channel": channel, "rep": rep,
        "difficulty": task.get("difficulty", "B"), "hardening": hard,
        "model": "TOY-SYNTHETIC", "stop_reason": "stop",
        "output_tokens": prose + loc * 8, "extraction": "ok", "exec_status": "ok",
        "n_tests": n_tests, "n_passed": n_passed,
        "pass_rate": round(n_passed / n_tests, 4), "all_pass": n_passed == n_tests,
        "n_robust": n_robust,
        "n_robust_passed": round(n_robust * clamp(rng.gauss(0.7, 0.15), 0, 1)),
        "robust_rate": 0.0, "parse_ok": True, "loc": loc,
        "comment_lines": comment_lines,
        "docstring_lines": max(0, round(rng.gauss(3 - 0.5 * s, 1))),
        "comment_ratio": round(clamp(comment_lines / loc, 0, 1), 4),
        "has_docstring": rng.random() < clamp(0.9 - 0.12 * s, 0, 1),
        "n_functions": rng.randint(1, 3),
        "type_hints": max(0, round(rng.gauss(1 + 1.3 * s, 1))),
        "try_count": rng.randint(0, 2), "raise_count": rng.randint(0, 2),
        "isinstance_count": rng.randint(0, 2),
        "validation_checks": max(0, round(rng.gauss(2.5, 1))),
        "cyclomatic_total": max(1, round(rng.gauss(8 + s, 2))),
        "cyclomatic_mean": round(clamp(rng.gauss(3 + 0.3 * s, 0.6), 1, 8), 2),
        "max_nesting": rng.randint(1, 4),
        "advanced_idioms": max(0, round(rng.gauss(0.5 + 0.9 * s, 0.8))),
        "identifier_len_mean": round(rng.gauss(9, 1.2), 2),
        "security_flags": 0, "response_chars": prose * 6 + loc * 30,
        "n_code_blocks": 1, "prose_words": prose,
        "md_headers": rng.randint(0, 3), "md_bullets": rng.randint(0, 8),
        "leakage": rng.random() < (0.05 + 0.08 * (s if level != "control" else 0)),
    }
    if n_robust:
        row["robust_rate"] = round(row["n_robust_passed"] / n_robust, 4)
    return row


def judge_entry(task, level):
    s = LEVELS[level]
    n_req = len(task.get("requirements", [])) or 4
    met = [rng.random() < clamp(0.85 + rng.gauss(0, 0.05), 0, 1) for _ in range(n_req)]
    return {
        "requirements_met": met,
        "requirements_met_prop": round(sum(met) / n_req, 4),
        "condescension": clamp(round(rng.gauss(2.6 - 0.45 * s, 0.7)), 1, 5),
        "assumed_knowledge": clamp(round(rng.gauss(1.8 + 0.7 * s, 0.7)), 1, 5),
        "hedging": rng.random() < 0.1, "refusal": False,
        "judge_model": "TOY-SYNTHETIC",
    }


def main():
    OUT.mkdir(exist_ok=True)
    rows, judge = [], {}
    for task in tasks:
        task_bias = clamp(rng.gauss(0.75, 0.15), 0.2, 0.98)
        for level in ["control", "beginner", "hobbyist", "professional", "expert"]:
            for channel in (["none"] if level == "control" else ["system", "message"]):
                for rep in (1, 2, 3):
                    row = make_row(task, level, channel, rep, task_bias)
                    rows.append(row)
                    judge[row["custom_id"]] = judge_entry(task, level)
    scores = OUT / "scores_skill_TOY.csv"
    with scores.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    (OUT / "judge_scores_TOY.json").write_text(json.dumps(judge, indent=1))
    print(f"TOY data: {len(rows)} score rows, {len(judge)} judge entries -> {OUT}")


if __name__ == "__main__":
    main()
