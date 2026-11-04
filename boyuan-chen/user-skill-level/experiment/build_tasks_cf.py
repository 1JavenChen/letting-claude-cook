"""Build the v2.1 Codeforces task battery (experiment/tasks_cf.json).

Source: open-r1/codeforces (test split). Candidate pool = experiment/out/
cf_candidates.json (2025 contests, rating 1000-2000, stdio, no custom
checker, has official tests). For each candidate we join the dataset's
generated tests (shard generated_tests/test_cases_<contest>.parquet) and
keep problems that end up with at least MIN_TESTS checkable tests. We then
draw 8 problems per 200-point rating bucket with a fixed seed.

Prompt text is identical across all persona conditions (the runner wraps
it), and is neutral: statement, input/output format, up to two examples,
and the instruction to produce a complete stdin/stdout Python 3 program.
"""
import collections
import json
import random
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

HERE = Path(__file__).resolve().parent
REPO = "open-r1/codeforces"
SEED = 20261028
PER_BUCKET = 8
MIN_TESTS = 8
MAX_TESTS = 20
MAX_INPUT_CHARS = 300_000


def tier(rating):
    return "A" if rating <= 1200 else ("B" if rating <= 1600 else "C")


def clean(s):
    return (s or "").replace("$$$", "$").replace("\r\n", "\n").strip()


def as_list(v):
    return v if isinstance(v, list) else json.loads(v or "[]")


def load_rows():
    p = hf_hub_download(REPO, "data/test-00000-of-00001.parquet", repo_type="dataset")
    d = pq.read_table(p).to_pydict()
    return {d["id"][i]: {k: d[k][i] for k in d} for i in range(len(d["id"]))}


_shard_cache = {}


def generated_tests(cid):
    contest = cid.split("/")[0]
    if contest not in _shard_cache:
        try:
            p = hf_hub_download(REPO, f"generated_tests/test_cases_{int(contest):04d}.parquet",
                                repo_type="dataset")
            g = pq.read_table(p).to_pydict()
            by = collections.defaultdict(list)
            for i in range(len(g["problem_id"])):
                by[g["problem_id"][i]].append((g["test_i"][i], g["input"][i], g["output"][i]))
            _shard_cache[contest] = by
        except Exception as e:  # shard missing
            print("  no shard for contest", contest, type(e).__name__)
            _shard_cache[contest] = {}
    return [{"input": i, "output": o} for _, i, o in sorted(_shard_cache[contest].get(cid, []))]


def tests_for(row):
    tests = [{"input": t["input"], "output": t["output"]} for t in as_list(row["official_tests"])]
    seen = {(t["input"], t["output"]) for t in tests}
    for t in generated_tests(row["id"]):
        key = (t["input"], t["output"])
        if key in seen or len(t["input"]) > MAX_INPUT_CHARS:
            continue
        tests.append(t)
        seen.add(key)
        if len(tests) >= MAX_TESTS:
            break
    for t in tests:
        t["input"] = t["input"].replace("\r\n", "\n")
        t["output"] = t["output"].replace("\r\n", "\n")
    return tests


def build_prompt(r):
    ex = as_list(r["examples"])[:2]
    ex_txt = "\n\n".join(
        f"Example input:\n{e['input'].strip()}\n\nExample output:\n{e['output'].strip()}" for e in ex)
    return (f"Solve the following competitive programming problem.\n\n"
            f"Problem: {r['title']}\n\n{clean(r['description'])}\n\n"
            f"Input format:\n{clean(r['input_format'])}\n\n"
            f"Output format:\n{clean(r['output_format'])}\n\n{ex_txt}\n\n"
            f"Write a complete Python 3 program that reads from standard input and writes the "
            f"answer to standard output. Time limit: {r['time_limit']} seconds. "
            f"Put the full program in a single ```python code block.")


def main():
    rows = load_rows()
    cands = json.load(open(HERE / "out" / "cf_candidates.json"))["candidate_ids"]
    usable = {}
    for n, cid in enumerate(cands, 1):
        tests = tests_for(rows[cid])
        if len(tests) >= MIN_TESTS:
            usable[cid] = tests
        if n % 20 == 0:
            print(f"  screened {n}/{len(cands)}, usable so far {len(usable)}")
    print("usable candidates:", len(usable))

    rng = random.Random(SEED)
    by_bucket = collections.defaultdict(list)
    for cid in usable:
        by_bucket[(rows[cid]["rating"] // 200) * 200].append(cid)
    chosen = []
    for b in sorted(by_bucket):
        pool = sorted(by_bucket[b])
        rng.shuffle(pool)
        chosen += pool[:PER_BUCKET]
        print(f"  bucket {b}: {len(pool)} usable, took {min(PER_BUCKET, len(pool))}")

    tasks = []
    for cid in chosen:
        r = rows[cid]
        tasks.append({
            "id": f"cf_{cid.replace('/', '')}", "cf_id": cid, "language": "python",
            "difficulty": tier(r["rating"]), "rating": r["rating"], "hardening": False,
            "entry_point": "stdin", "time_limit": float(r["time_limit"]),
            "contest_start_year": r["contest_start_year"], "tags": as_list(r["tags"]),
            "prompt": build_prompt(r), "tests": usable[cid], "robustness_tests": [],
        })
    out = HERE / "tasks_cf.json"
    out.write_text(json.dumps(tasks, indent=1))
    print(f"wrote {out}: {len(tasks)} tasks; tests/task:", [len(t["tests"]) for t in tasks])
    print("max input chars per task:", [max(len(x["input"]) for x in t["tests"]) for t in tasks])


if __name__ == "__main__":
    main()
