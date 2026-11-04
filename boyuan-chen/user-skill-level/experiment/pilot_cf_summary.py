"""Summarize the v2.1 control-only screening pilot and pick the 24-task battery.

Reads out/scores_cf_pilot.csv (from score_cf.py), prints per-task control
pass rates, writes out/cf_pilot_summary.json and experiment/tasks_cf_selected.json
(the 24 tasks whose mean control pass rate lies in [LOW, HIGH], filling from
the nearest-to-band tasks if fewer than 24 qualify).
"""
import csv, json, statistics, collections, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
LOW, HIGH, N_SELECT = 0.30, 0.90, 24
src = sys.argv[1] if len(sys.argv) > 1 else HERE / "out" / "scores_cf_pilot.csv"
rows = list(csv.DictReader(open(src)))
tasks = {t["id"]: t for t in json.load(open(HERE / "tasks_cf.json"))}
by = collections.defaultdict(list)
for r in rows: by[r["task_id"]].append(r)
summary = []
for tid, rs in by.items():
    pr = [float(r["pass_rate"]) for r in rs]
    summary.append({"task_id": tid, "rating": int(tasks[tid]["rating"]), "tier": tasks[tid]["difficulty"],
                    "n": len(rs), "mean_pass": round(statistics.mean(pr), 3),
                    "all_pass_share": round(sum(float(r["all_pass"]) for r in rs) / len(rs), 3),
                    "refusals": sum(r["exec_status"] == "no_code" for r in rs),
                    "timeouts": sum(int(r["n_timeout"] or 0) for r in rs),
                    "mean_tokens": round(statistics.mean(float(r["output_tokens"] or 0) for r in rs))})
summary.sort(key=lambda s: (s["rating"], s["task_id"]))
allpr = [float(r["pass_rate"]) for r in rows]
print(f"{len(rows)} responses over {len(by)} tasks; mean pass {statistics.mean(allpr):.3f}; "
      f"all-pass {sum(float(r['all_pass']) for r in rows)/len(rows):.3f}; "
      f"refusals {sum(r['exec_status']=='no_code' for r in rows)}; "
      f"timeouts {sum(int(r['n_timeout'] or 0) for r in rows)} tests; "
      f"syntax errors {sum(r['exec_status']=='code_error' for r in rows)}")
tier_means = {}
for tier in "ABC":
    v = [float(r["pass_rate"]) for r in rows if tasks[r["task_id"]]["difficulty"] == tier]
    tier_means[tier] = round(statistics.mean(v), 3) if v else None
print("tier means:", tier_means)
print(f"{'task':<10}{'rating':>7}{'tier':>5}{'mean':>7}{'all':>6}{'ref':>5}{'TO':>5}{'tok':>6}")
for s in summary:
    flag = " <- in band" if LOW <= s["mean_pass"] <= HIGH else ""
    print(f"{s['task_id']:<10}{s['rating']:>7}{s['tier']:>5}{s['mean_pass']:>7.2f}{s['all_pass_share']:>6.2f}{s['refusals']:>5}{s['timeouts']:>5}{s['mean_tokens']:>6}{flag}")
in_band = [s for s in summary if LOW <= s["mean_pass"] <= HIGH]
# Fill the remainder alternately from just below the band (model usually fails:
# room to improve) and just above it (model usually succeeds: room to get worse),
# so the final set is not dominated by either ceiling or floor tasks.
below = sorted([s for s in summary if s["mean_pass"] < LOW], key=lambda s: (-s["mean_pass"], s["task_id"]))
above = sorted([s for s in summary if s["mean_pass"] > HIGH], key=lambda s: (s["mean_pass"], s["task_id"]))
selected = [s["task_id"] for s in in_band]
side = 0
while len(selected) < N_SELECT and (below or above):
    pool = below if (side == 0 and below) or not above else above
    selected.append(pool.pop(0)["task_id"]); side ^= 1
selected = sorted(selected)
n_fill = N_SELECT - len(in_band)
print(f"in band [{LOW},{HIGH}]: {len(in_band)}; selected {len(selected)} (filled {max(0, n_fill)} alternately from below/above the band)")
json.dump({"low": LOW, "high": HIGH, "n_in_band": len(in_band), "selected": selected,
           "overall_mean_pass": round(statistics.mean(allpr), 4), "tier_means": tier_means,
           "per_task": summary}, open(HERE / "out" / "cf_pilot_summary.json", "w"), indent=1)
json.dump([tasks[t] for t in selected], open(HERE / "tasks_cf_selected.json", "w"), indent=1)
sel_ratings = collections.Counter((tasks[t]["rating"]//200)*200 for t in selected)
print("selected rating buckets:", sorted(sel_ratings.items()))
