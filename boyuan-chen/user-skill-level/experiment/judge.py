"""Blind LLM judge for the skill-level coding suite (protocol: research/03_methods_plan.md §5).

One judge request per collected response. The judge sees ONLY:
  - the bare task text (byte-identical across conditions),
  - the task's hidden requirements checklist,
  - the response text.
It never sees the persona, skill level, channel, or any other condition
metadata, and it is instructed to ignore any audience references inside the
response itself (the `leakage` flag in scores_skill.csv drives the
pre-registered re-analysis excluding leaking responses).

Output per response (strict JSON requested from the judge):
  requirements_met   list of booleans, one per hidden requirement, in order
  condescension      1-5 (1 = none, 5 = clearly talking down)
  assumed_knowledge  1-5 (background knowledge the response presupposes)
  hedging            bool
  refusal            bool
plus a derived `requirements_met_prop` added locally.

Scores are checkpointed to out/judge_scores_skill.json every few responses;
rerun the same command to resume (parse failures are not checkpointed and are
retried on rerun). Judge model defaults to openai/gpt-oss-120b on Groq
(subject model != judge model), resolved against the account's live catalog.

Usage:
    python judge.py --dry-run                       # print 2 sample prompts, no network
    python judge.py --dry-run --responses fixtures/fixture_responses_skill.jsonl
    GROQ_API_KEY=... python judge.py                # judge out/responses_skill.jsonl
    python judge.py --backend local                 # any OpenAI-compatible server
"""

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "out"
DEFAULT_RESPONSES = OUT / "responses_skill.jsonl"
DEFAULT_SCORES = OUT / "judge_scores_skill.json"
TASKS_PATH = HERE / "tasks.json"

JUDGE_MODELS = {
    "groq": "openai/gpt-oss-120b",
    "local": "qwen3:4b-instruct-2507",
    "hf": "Qwen/Qwen3-235B-A22B-Instruct-2507",
}
JUDGE_SYSTEM = "You are a careful evaluator."
JUDGE_MAX_TOKENS = 800
JUDGE_TEMPERATURE = 0.0
CHECKPOINT_EVERY = 3  # small: host sleeps have forced restarts; each score costs minutes of quota

# Only the task text, the requirements checklist, and the response go into the
# prompt - never the persona/condition (blind-judge protocol, v1 carried over).
JUDGE_PROMPT = """You are scoring one AI assistant response to a programming task. \
You see only the task and the response. Ignore any references the response makes \
to who the user is or what background or skill level they have - score the \
content itself.

<task>
{task}
</task>

<response>
{response}
</response>

The task has {n_requirements} hidden requirements:
{requirements}

For each requirement decide whether the response's code (or, if there is no
code, its explicit instructions) satisfies it. Then rate the tone of the
response prose.

Reply with ONLY a JSON object, no other text, exactly this shape:
{{
  "requirements_met": [{req_slots}],
  "condescension": <1-5, 1=none, 5=clearly talking down to the reader>,
  "assumed_knowledge": <1-5, background knowledge the response presupposes; 1=none, 5=expert>,
  "hedging": <true if the response substantially hedges, disclaims, or refuses to commit>,
  "refusal": <true if the response refuses or substantially fails to attempt the task>
}}"""


def build_judge_prompt(task: dict, response_text: str) -> str:
    reqs = task["requirements"]
    numbered = "\n".join(f"{i + 1}. {r}" for i, r in enumerate(reqs))
    return JUDGE_PROMPT.format(
        task=task["prompt"],
        response=response_text,
        n_requirements=len(reqs),
        requirements=numbered,
        req_slots=", ".join(f"<true/false for requirement {i + 1}>"
                            for i in range(len(reqs))),
    )


# -------------------------------------------------------------- JSON parsing

_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _balanced_objects(text: str):
    """Yield every top-level {...} span in text (string-aware brace matching)."""
    depth, start, in_str, escape = 0, None, False, False
    for i, ch in enumerate(text):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"' and depth > 0:
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0:
                yield text[start:i + 1]


def parse_judge_json(text: str) -> dict | None:
    """Extract the judge's JSON object (hardened version of analyze.py's rule).

    Tries the whole text, then a fenced ```json block, then every balanced
    top-level {...} span. Among parseable dicts, one containing a
    'requirements_met' key wins; otherwise the first parseable dict.
    """
    if not text:
        return None
    candidates = [text.strip()]
    m = _FENCED_JSON_RE.search(text)
    if m:
        candidates.append(m.group(1))
    candidates.extend(_balanced_objects(text))
    first = None
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            if "requirements_met" in obj:
                return obj
            first = first or obj
    return first


def _as_bool(value) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("true", "yes", "1", "met", "pass"):
            return True
        if v in ("false", "no", "0", "unmet", "fail"):
            return False
    return None


def _as_rating(value) -> int | None:
    try:
        v = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    return min(5, max(1, v))


def validate_judge_scores(parsed: dict, n_requirements: int) -> dict | None:
    """Normalize a parsed judge object; None if it is unusable.

    requirements_met must coerce to exactly n_requirements booleans; ratings
    are clamped to 1-5; flags coerce leniently (missing flag -> False).
    """
    met_raw = parsed.get("requirements_met")
    if not isinstance(met_raw, list) or len(met_raw) != n_requirements:
        return None
    met = [_as_bool(v) for v in met_raw]
    if any(v is None for v in met):
        return None
    condescension = _as_rating(parsed.get("condescension"))
    assumed = _as_rating(parsed.get("assumed_knowledge"))
    if condescension is None or assumed is None:
        return None
    return {
        "requirements_met": met,
        "requirements_met_prop": round(sum(met) / n_requirements, 4),
        "condescension": condescension,
        "assumed_knowledge": assumed,
        "hedging": bool(_as_bool(parsed.get("hedging", False))),
        "refusal": bool(_as_bool(parsed.get("refusal", False))),
    }


# ------------------------------------------------------------------- loading


def load_tasks() -> dict:
    return {t["id"]: t for t in json.loads(TASKS_PATH.read_text())}


def load_records(responses_path: Path, tasks_by_id: dict) -> list[dict]:
    records, skipped = [], 0
    with open(responses_path) as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("result_type") not in (None, "succeeded"):
                skipped += 1
                continue
            if rec.get("task_id") not in tasks_by_id:
                skipped += 1
                continue
            records.append(rec)
    if skipped:
        print(f"Skipped {skipped} non-succeeded/unknown-task records")
    return records


# -------------------------------------------------------------------- runner


def run_judge(records: list[dict], tasks_by_id: dict, chat_fn, model: str,
              scores_path: Path, max_tokens: int) -> dict:
    scores = json.loads(scores_path.read_text()) if scores_path.exists() else {}
    if scores:
        print(f"Resuming judge: {len(scores)} already scored")

    pending = [r for r in records if r["custom_id"] not in scores]
    parse_failures = consecutive_errors = 0
    for i, rec in enumerate(pending, 1):
        task = tasks_by_id[rec["task_id"]]
        prompt = build_judge_prompt(task, rec.get("response_text", ""))
        try:
            result = chat_fn(model, JUDGE_SYSTEM, prompt,
                             max_tokens=max_tokens,
                             temperature=JUDGE_TEMPERATURE)
        except SystemExit:
            raise
        except Exception as e:
            consecutive_errors += 1
            print(f"  judge failed for {rec['custom_id']}: {type(e).__name__}: {e}")
            if consecutive_errors >= 5:
                scores_path.write_text(json.dumps(scores, indent=2))
                raise SystemExit(
                    "Aborting after 5 consecutive failures - fix the cause and "
                    f"rerun to resume ({len(scores)} scored, checkpoint saved)."
                )
            continue
        consecutive_errors = 0
        validated = validate_judge_scores(
            parse_judge_json(result["response_text"]) or {},
            len(task["requirements"]),
        )
        if validated is None:
            parse_failures += 1
            print(f"  unparseable judge output for {rec['custom_id']} "
                  f"(will retry on rerun): {result['response_text'][:120]!r}")
        else:
            scores[rec["custom_id"]] = {**validated, "judge_model": model}
        if i % CHECKPOINT_EVERY == 0 or i == len(pending):
            scores_path.write_text(json.dumps(scores, indent=2))
            print(f"  judge {i}/{len(pending)} (total scored {len(scores)})")
    scores_path.write_text(json.dumps(scores, indent=2))
    if parse_failures:
        print(f"{parse_failures} unparseable judge outputs - rerun to retry them")
    return scores


# --------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", default=str(DEFAULT_RESPONSES))
    ap.add_argument("--out", default=str(DEFAULT_SCORES),
                    help="checkpointed judge-score JSON (resumes on rerun)")
    ap.add_argument("--backend", choices=["groq", "local", "hf"], default="groq")
    ap.add_argument("--model", default=None, help="defaults per backend")
    ap.add_argument("--max-tokens", type=int, default=JUDGE_MAX_TOKENS)
    ap.add_argument("--limit", type=int, default=None,
                    help="judge at most N records (debugging)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print 2 sample judge prompts and exit; no network")
    args = ap.parse_args()
    args.model = args.model or JUDGE_MODELS[args.backend]

    tasks_by_id = load_tasks()
    records = load_records(Path(args.responses), tasks_by_id)
    if not records:
        raise SystemExit(f"No judgeable records in {args.responses}")
    if args.limit:
        records = records[: args.limit]

    if args.dry_run:
        print(f"Dry run: {len(records)} judgeable records in {args.responses}; "
              f"judge model would be {args.model} via {args.backend}. "
              "Sample prompts (judge sees ONLY task + requirements + response):")
        for rec in (records[0], records[min(1, len(records) - 1)]):
            task = tasks_by_id[rec["task_id"]]
            print(f"\n===== sample prompt for {rec['custom_id']} "
                  f"(condition withheld from judge) =====")
            print(build_judge_prompt(task, rec.get("response_text", "")))
        return

    from run_experiment import GROQ_JUDGE_PREFS, make_chat, resolve_groq_model

    if args.backend == "groq":
        args.model = resolve_groq_model(args.model, GROQ_JUDGE_PREFS)
    print(f"Judging {len(records)} responses with {args.model} via {args.backend}")
    OUT.mkdir(exist_ok=True)
    scores = run_judge(records, tasks_by_id, make_chat(args.backend),
                       args.model, Path(args.out), args.max_tokens)
    print(f"Judge scored {len(scores)}/{len(records)} responses -> {args.out}")


if __name__ == "__main__":
    main()
