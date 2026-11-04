"""Score collected responses for the skill-level coding suite.

For every record in a responses JSONL file this script:
  1. extracts code with the pre-registered fence rule (first ```python block,
     else the first unlabeled fenced block that compiles as Python, else
     `extraction_fail`),
  2. executes the code against the task's hidden tests in a sandboxed
     subprocess (python -I, empty environment, stdin closed, per-test alarm,
     overall timeout, no network use assumed or required),
  3. computes static style/complexity/robustness/security metrics with the
     stdlib only (ast + tokenize),
and writes one row per response to a CSV (plus a JSONL with per-test detail).

Usage:
    python score.py                         # out/responses_skill.jsonl -> out/scores_skill.csv
    python score.py --responses fixtures/fixture_responses.jsonl --out out/scores_fixture.csv
    python score.py --verify-tasks          # run every reference solution through the sandbox
"""

import argparse
import ast
import csv
import io
import json
import re
import subprocess
import sys
import tempfile
import tokenize
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "out"
DEFAULT_RESPONSES = OUT / "responses_skill.jsonl"
DEFAULT_SCORES = OUT / "scores_skill.csv"
TASKS_PATH = HERE / "tasks.json"

PER_TEST_SECONDS = 5  # wall-clock budget per unit test inside the sandbox

# ----------------------------------------------------------- code extraction

FENCE_RE = re.compile(
    r"```[ \t]*([A-Za-z0-9_+-]*)[^\n]*\n(.*?)(?:\n?```|\Z)", re.DOTALL
)
PYTHON_LABELS = {"python", "py", "python3"}


def extract_code(text, language="python"):
    """Pre-registered extraction rule (design doc section 6).

    Returns (code, how) where how is 'labeled', 'unlabeled', or None.
    A fence left unclosed by a truncated response is still extracted.
    """
    assert language == "python", "only python tasks in this battery"
    unlabeled = []
    for match in FENCE_RE.finditer(text or ""):
        label, body = match.group(1).lower(), match.group(2)
        if label in PYTHON_LABELS:
            return body, "labeled"
        if label == "":
            unlabeled.append(body)
    for body in unlabeled:
        try:
            compile(body, "<candidate>", "exec")
            return body, "unlabeled"
        except SyntaxError:
            continue
    return None, None


# --------------------------------------------------------- sandboxed testing

HARNESS = r'''
import json, signal, sys

spec = json.load(open(sys.argv[1]))
results_path = sys.argv[2]


def finish(payload):
    with open(results_path, "w") as fh:
        json.dump(payload, fh)
    sys.exit(0)


def _timeout(signum, frame):
    raise TimeoutError("per-test time limit exceeded")


signal.signal(signal.SIGALRM, _timeout)


def handles_invalid(fn, *args, **kwargs):
    """True iff the call raises a deliberate validation error."""
    try:
        fn(*args, **kwargs)
    except (ValueError, TypeError):
        return True
    except Exception:
        return False
    return False


module_globals = {"handles_invalid": handles_invalid, "__name__": "__main__"}
signal.setitimer(signal.ITIMER_REAL, spec["per_test_seconds"] * 2)
try:
    exec(compile(spec["code"], "<response>", "exec"), module_globals)
    signal.setitimer(signal.ITIMER_REAL, 0)
except BaseException as exc:
    signal.setitimer(signal.ITIMER_REAL, 0)
    finish({"status": "code_error", "detail": f"{type(exc).__name__}: {exc}"})

if spec["entry_point"] not in module_globals:
    finish({"status": "missing_entry_point"})


def run(tests):
    outcomes = []
    for source in tests:
        scope = dict(module_globals)
        signal.setitimer(signal.ITIMER_REAL, spec["per_test_seconds"])
        try:
            exec(compile(source, "<test>", "exec"), scope)
            outcomes.append(True)
        except BaseException:
            outcomes.append(False)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
    return outcomes


finish({
    "status": "ok",
    "tests": run(spec["tests"]),
    "robustness": run(spec["robustness_tests"]),
})
'''


def run_tests_sandboxed(code, task, per_test_seconds=PER_TEST_SECONDS):
    """Execute code against a task's tests in an isolated subprocess.

    Returns a dict: status plus per-test booleans when status == 'ok'.
    """
    n_tests = len(task["tests"]) + len(task["robustness_tests"])
    overall_timeout = 20 + per_test_seconds * (n_tests + 2)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        spec_path = tmp / "spec.json"
        spec_path.write_text(json.dumps({
            "code": code,
            "entry_point": task["entry_point"],
            "tests": task["tests"],
            "robustness_tests": task["robustness_tests"],
            "per_test_seconds": per_test_seconds,
        }))
        harness_path = tmp / "harness.py"
        harness_path.write_text(HARNESS)
        results_path = tmp / "results.json"
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(harness_path), str(spec_path),
                 str(results_path)],
                cwd=tmp, env={}, stdin=subprocess.DEVNULL,
                capture_output=True, text=True, timeout=overall_timeout,
            )
        except subprocess.TimeoutExpired:
            return {"status": "sandbox_timeout"}
        if not results_path.exists():
            detail = (proc.stderr or proc.stdout or "")[-300:]
            return {"status": "harness_error", "detail": detail}
        try:
            return json.loads(results_path.read_text())
        except json.JSONDecodeError:
            return {"status": "harness_error", "detail": "unreadable results file"}


# ------------------------------------------------------------ static metrics

_DECISION_NODES = (ast.If, ast.For, ast.While, ast.AsyncFor, ast.ExceptHandler,
                   ast.Assert, ast.IfExp)
_NESTING_NODES = (ast.If, ast.For, ast.While, ast.AsyncFor, ast.With,
                  ast.AsyncWith, ast.Try, ast.FunctionDef, ast.AsyncFunctionDef,
                  ast.ClassDef)
_COMPREHENSIONS = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)


def _decision_points(node):
    count = 0
    for child in ast.walk(node):
        if isinstance(child, _DECISION_NODES):
            count += 1
        elif isinstance(child, ast.BoolOp):
            count += len(child.values) - 1
        elif isinstance(child, ast.comprehension):
            count += 1 + len(child.ifs)
        elif isinstance(child, getattr(ast, "match_case", ())):
            count += 1
    return count


def _max_nesting(node, depth=0):
    deepest = depth
    for child in ast.iter_child_nodes(node):
        next_depth = depth + 1 if isinstance(child, _NESTING_NODES) else depth
        deepest = max(deepest, _max_nesting(child, next_depth))
    return deepest


def _docstring_lines(tree):
    lines = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                expr = node.body[0].value
                lines += (expr.end_lineno or expr.lineno) - expr.lineno + 1
    return lines


_SECURITY_CALLS = {"eval", "exec", "__import__", "compile"}
_SECURITY_ATTRS = {("os", "system"), ("os", "popen"), ("pickle", "load"),
                   ("pickle", "loads"), ("marshal", "loads")}


def _security_flags(tree):
    flags = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name) and fn.id in _SECURITY_CALLS:
                flags += 1
            elif (isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name)
                  and (fn.value.id, fn.attr) in _SECURITY_ATTRS):
                flags += 1
            for kw in node.keywords:
                if (kw.arg == "shell" and isinstance(kw.value, ast.Constant)
                        and kw.value.value is True):
                    flags += 1
    return flags


def code_metrics(code):
    """Stdlib-only static metrics for one extracted code block."""
    metrics = {"parse_ok": False}
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return metrics
    metrics["parse_ok"] = True

    all_lines = code.splitlines()
    nonblank = [ln for ln in all_lines if ln.strip()]
    comment_lines = 0
    try:
        for tok in tokenize.generate_tokens(io.StringIO(code).readline):
            if tok.type == tokenize.COMMENT:
                comment_lines += 1
    except (tokenize.TokenError, IndentationError):
        comment_lines = sum(1 for ln in nonblank if ln.lstrip().startswith("#"))
    doc_lines = _docstring_lines(tree)
    code_lines = max(len(nonblank) - doc_lines
                     - sum(1 for ln in nonblank if ln.lstrip().startswith("#")), 1)

    functions = [n for n in ast.walk(tree)
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    cyclo = [1 + _decision_points(fn) for fn in functions] or [1 + _decision_points(tree)]

    type_hints = 0
    identifiers = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            identifiers.append(node.name)
            if node.returns is not None:
                type_hints += 1
            for arg in (node.args.args + node.args.posonlyargs
                        + node.args.kwonlyargs):
                identifiers.append(arg.arg)
                if arg.annotation is not None:
                    type_hints += 1
        elif isinstance(node, ast.AnnAssign):
            type_hints += 1
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            identifiers.append(node.id)

    try_count = sum(isinstance(n, ast.Try) for n in ast.walk(tree))
    raise_count = sum(isinstance(n, ast.Raise) for n in ast.walk(tree))
    isinstance_count = sum(
        1 for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        and n.func.id in ("isinstance", "issubclass")
    )
    advanced = sum(
        1 for n in ast.walk(tree)
        if isinstance(n, _COMPREHENSIONS + (ast.Lambda, ast.Yield, ast.YieldFrom,
                                            ast.NamedExpr))
    ) + sum(len(fn.decorator_list) for fn in functions) + type_hints

    metrics.update({
        "loc": len(nonblank),
        "comment_lines": comment_lines,
        "docstring_lines": doc_lines,
        "comment_ratio": round((comment_lines + doc_lines) / code_lines, 4),
        "has_docstring": int(doc_lines > 0),
        "n_functions": len(functions),
        "type_hints": type_hints,
        "try_count": try_count,
        "raise_count": raise_count,
        "isinstance_count": isinstance_count,
        "validation_checks": try_count + raise_count + isinstance_count,
        "cyclomatic_total": sum(cyclo),
        "cyclomatic_mean": round(sum(cyclo) / len(cyclo), 3),
        "max_nesting": _max_nesting(tree),
        "advanced_idioms": advanced,
        "identifier_len_mean": round(
            sum(map(len, identifiers)) / len(identifiers), 2) if identifiers else 0,
        "security_flags": _security_flags(tree),
    })
    return metrics


LEAKAGE_RE = re.compile(
    r"\b(beginner|novice|hobbyist|professional software engineer|staff-level"
    r"|expert|skill level|first programming course|fifteen[- ]plus"
    r"|years of (?:industry )?experience|new to (?:programming|coding)"
    r"|(?:just|you(?:'re| are)) (?:getting )?start(?:ing|ed) (?:to learn|out|with)"
    r")\b",
    re.IGNORECASE,
)


def prose_metrics(text):
    """Metrics on the response text outside code fences."""
    text = text or ""
    prose = FENCE_RE.sub(" ", text)
    return {
        "response_chars": len(text),
        "n_code_blocks": len(FENCE_RE.findall(text)),
        "prose_words": len(prose.split()),
        "md_headers": len(re.findall(r"^#{1,6}\s", prose, re.MULTILINE)),
        "md_bullets": len(re.findall(r"^\s*(?:[-*]|\d+\.)\s", prose, re.MULTILINE)),
        "leakage": int(bool(LEAKAGE_RE.search(prose))),
    }


# ------------------------------------------------------------------- scoring

META_FIELDS = ["custom_id", "task_id", "skill_level", "channel", "rep",
               "difficulty", "hardening", "model", "stop_reason",
               "output_tokens"]
RESULT_FIELDS = ["extraction", "exec_status", "n_tests", "n_passed",
                 "pass_rate", "all_pass", "n_robust", "n_robust_passed",
                 "robust_rate"]
STATIC_FIELDS = ["parse_ok", "loc", "comment_lines", "docstring_lines",
                 "comment_ratio", "has_docstring", "n_functions", "type_hints",
                 "try_count", "raise_count", "isinstance_count",
                 "validation_checks", "cyclomatic_total", "cyclomatic_mean",
                 "max_nesting", "advanced_idioms", "identifier_len_mean",
                 "security_flags"]
PROSE_FIELDS = ["response_chars", "n_code_blocks", "prose_words", "md_headers",
                "md_bullets", "leakage"]
ALL_FIELDS = META_FIELDS + RESULT_FIELDS + STATIC_FIELDS + PROSE_FIELDS


def score_record(record, tasks_by_id, per_test_seconds=PER_TEST_SECONDS):
    task = tasks_by_id[record["task_id"]]
    row = {field: record.get(field) for field in META_FIELDS}
    text = record.get("response_text", "")
    row.update(prose_metrics(text))

    code, how = extract_code(text, task["language"])
    n_tests, n_robust = len(task["tests"]), len(task["robustness_tests"])
    row.update({"n_tests": n_tests, "n_robust": n_robust, "n_passed": 0,
                "n_robust_passed": 0, "pass_rate": 0.0, "all_pass": 0,
                "robust_rate": 0.0 if n_robust else None})
    detail = {"custom_id": record.get("custom_id"), "task_id": task["id"]}

    if code is None:
        row["extraction"] = "extraction_fail"
        row["exec_status"] = "no_code"
        return row, detail

    row["extraction"] = how
    row.update(code_metrics(code))

    sandbox = run_tests_sandboxed(code, task, per_test_seconds)
    row["exec_status"] = sandbox["status"]
    detail["sandbox"] = sandbox
    if sandbox["status"] == "ok":
        passed = sandbox["tests"]
        row["n_passed"] = sum(passed)
        row["pass_rate"] = round(sum(passed) / n_tests, 4)
        row["all_pass"] = int(all(passed))
        if n_robust:
            robust = sandbox["robustness"]
            row["n_robust_passed"] = sum(robust)
            row["robust_rate"] = round(sum(robust) / n_robust, 4)
    return row, detail


def load_tasks():
    tasks = json.loads(TASKS_PATH.read_text())
    return {t["id"]: t for t in tasks}


def score_file(responses_path, out_csv, per_test_seconds):
    tasks_by_id = load_tasks()
    rows, details, skipped = [], [], 0
    with open(responses_path) as fh:
        for line in fh:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("result_type") not in (None, "succeeded"):
                skipped += 1
                continue
            if record.get("task_id") not in tasks_by_id:
                skipped += 1
                continue
            row, detail = score_record(record, tasks_by_id, per_test_seconds)
            rows.append(row)
            details.append(detail)
            if len(rows) % 25 == 0:
                print(f"  scored {len(rows)}")

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(exist_ok=True)
    with out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=ALL_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row.get(k))
                             for k in ALL_FIELDS})
    detail_path = out_csv.with_suffix(".detail.jsonl")
    with detail_path.open("w") as fh:
        for detail in details:
            fh.write(json.dumps(detail) + "\n")
    print(f"Scored {len(rows)} responses ({skipped} skipped) -> {out_csv} "
          f"(+ per-test detail in {detail_path.name})")
    return rows


# -------------------------------------------------------------- verify-tasks


def verify_tasks(per_test_seconds):
    """Run every task's reference solution through the real sandbox."""
    failures = 0
    for task in load_tasks().values():
        result = run_tests_sandboxed(task["reference_solution"], task,
                                     per_test_seconds)
        if result["status"] != "ok":
            print(f"FAIL {task['id']}: sandbox status {result}")
            failures += 1
            continue
        bad_main = [i for i, ok in enumerate(result["tests"]) if not ok]
        bad_robust = [i for i, ok in enumerate(result["robustness"]) if not ok]
        if bad_main or bad_robust:
            print(f"FAIL {task['id']}: tests {bad_main} robustness {bad_robust}")
            failures += 1
        else:
            print(f"ok {task['id']}: {len(result['tests'])} tests, "
                  f"{len(result['robustness'])} robustness tests")
    print("verify-tasks:", "ALL PASS" if failures == 0 else f"{failures} FAILURES")
    return failures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", default=str(DEFAULT_RESPONSES))
    ap.add_argument("--out", default=str(DEFAULT_SCORES))
    ap.add_argument("--per-test-seconds", type=int, default=PER_TEST_SECONDS)
    ap.add_argument("--verify-tasks", action="store_true",
                    help="run each reference solution through the sandbox and exit")
    args = ap.parse_args()

    if args.verify_tasks:
        sys.exit(1 if verify_tasks(args.per_test_seconds) else 0)
    score_file(args.responses, args.out, args.per_test_seconds)


if __name__ == "__main__":
    main()
