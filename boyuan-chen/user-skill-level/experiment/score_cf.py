"""Score v2.1 Codeforces responses (stdin/stdout programs).

Mirrors score.py's CSV schema so stats.py / ablations.py / figures.py work
unchanged, but executes each extracted program as a separate sandboxed
process per test (python -I, empty env, stdin piped, wall-clock limit) and
compares whitespace-normalized stdout tokens against the expected output.

Per-test limit = clamp(2.5 x the problem's stated limit, 2 s, 10 s). The
stated limits are set for C++; the 2.5x factor is a fixed, pre-declared
concession to Python that applies identically to every condition.

Usage:
    python score_cf.py                       # out/responses_cf.jsonl -> out/scores_cf.csv
    python score_cf.py --responses out/x.jsonl --out out/y.csv --workers 4
"""
import argparse
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from score import (ALL_FIELDS, META_FIELDS, code_metrics, extract_code,
                   prose_metrics)

HERE = Path(__file__).resolve().parent
TASKS_PATH = HERE / "tasks_cf.json"
DEFAULT_RESPONSES = HERE / "out" / "responses_cf.jsonl"
DEFAULT_SCORES = HERE / "out" / "scores_cf.csv"
EXTRA_FIELDS = ["rating", "n_wrong", "n_timeout", "n_runtime_error", "gave_up", "truncated"]
GIVEUP_RE = re.compile(r"(i.m sorry|i can.t|i cannot|i couldn.t|unfortunately|not able to)", re.I)
CF_FIELDS = ALL_FIELDS + EXTRA_FIELDS


def per_test_limit(task):
    return min(10.0, max(2.0, 2.5 * float(task.get("time_limit", 2.0))))


def normalize(s):
    return s.split()


def run_one_test(prog_path, test, limit, tmp):
    try:
        proc = subprocess.run(
            [sys.executable, "-I", str(prog_path)],
            input=test["input"], capture_output=True, text=True,
            timeout=limit, cwd=tmp, env={}, errors="replace",
        )
    except subprocess.TimeoutExpired:
        return "timeout"
    if proc.returncode != 0:
        return "runtime_error"
    return "pass" if normalize(proc.stdout) == normalize(test["output"]) else "wrong"


def run_program(code, task):
    limit = per_test_limit(task)
    with tempfile.TemporaryDirectory() as tmp:
        prog = Path(tmp) / "prog.py"
        prog.write_text(code)
        return [run_one_test(prog, t, limit, tmp) for t in task["tests"]]


def score_record(record, tasks_by_id):
    task = tasks_by_id[record["task_id"]]
    row = {field: record.get(field) for field in META_FIELDS}
    row["rating"] = task.get("rating")
    text = record.get("response_text", "")
    row.update(prose_metrics(text))
    n_tests = len(task["tests"])
    row.update({"n_tests": n_tests, "n_robust": 0, "n_passed": 0,
                "n_robust_passed": 0, "pass_rate": 0.0, "all_pass": 0,
                "robust_rate": None, "n_wrong": 0, "n_timeout": 0,
                "n_runtime_error": 0, "gave_up": 0,
                "truncated": int(record.get("stop_reason") == "length")})
    detail = {"custom_id": record.get("custom_id"), "task_id": task["id"]}

    code, how = extract_code(text, "python")
    if code is None:
        row["extraction"], row["exec_status"] = "extraction_fail", "no_code"
        # The model explicitly declined / gave up (no code, apology-style prose).
        row["gave_up"] = int(bool(GIVEUP_RE.search(text[:300])) and "```" not in text)
        return row, detail
    row["extraction"] = how
    row.update(code_metrics(code))
    try:
        compile(code, "<response>", "exec")
    except SyntaxError as exc:
        row["exec_status"] = "code_error"
        detail["sandbox"] = {"status": "code_error", "detail": f"SyntaxError: {exc}"}
        return row, detail

    outcomes = run_program(code, task)
    detail["sandbox"] = {"status": "ok", "outcomes": outcomes,
                         "per_test_seconds": per_test_limit(task)}
    row["exec_status"] = "ok"
    row["n_passed"] = outcomes.count("pass")
    row["n_wrong"] = outcomes.count("wrong")
    row["n_timeout"] = outcomes.count("timeout")
    row["n_runtime_error"] = outcomes.count("runtime_error")
    row["pass_rate"] = round(row["n_passed"] / n_tests, 4)
    row["all_pass"] = int(row["n_passed"] == n_tests)
    return row, detail


def load_tasks():
    return {t["id"]: t for t in json.loads(TASKS_PATH.read_text())}


def score_file(responses_path, out_csv, workers):
    tasks_by_id = load_tasks()
    records, skipped = [], 0
    with open(responses_path) as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("result_type") not in (None, "succeeded") or rec.get("task_id") not in tasks_by_id:
                skipped += 1
                continue
            records.append(rec)

    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, res in enumerate(pool.map(lambda r: score_record(r, tasks_by_id), records), 1):
            results.append(res)
            if i % 25 == 0:
                print(f"  scored {i}/{len(records)}", flush=True)

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(exist_ok=True)
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CF_FIELDS)
        w.writeheader()
        for row, _ in results:
            w.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in CF_FIELDS})
    with out_csv.with_suffix(".detail.jsonl").open("w") as fh:
        for _, d in results:
            fh.write(json.dumps(d) + "\n")
    print(f"Scored {len(results)} responses ({skipped} skipped) -> {out_csv}")
    return [r for r, _ in results]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", default=str(DEFAULT_RESPONSES))
    ap.add_argument("--out", default=str(DEFAULT_SCORES))
    ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) // 2))
    args = ap.parse_args()
    score_file(args.responses, args.out, args.workers)


if __name__ == "__main__":
    main()
