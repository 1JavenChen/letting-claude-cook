#!/bin/bash
# Stage 4: fresh out-of-sample control answers (reps 4-6) on the 24 selected tasks.
# Why: the 24 tasks were selected using control reps 1-3, so those answers are
# biased (regression to the mean) as a baseline. Reps 4-6 give an unbiased control.
cd "$(dirname "$0")"
export FORCE_IPV4=1
PY=../.venv/bin/python
until [ -f out/cf_full_run.done ]; do sleep 900; done
echo "== stage 4: fresh control reps 4-6 on tasks_cf_selected.json =="
TARGET=144
while true; do
  $PY -u run_experiment.py --suite cf --tasks-file tasks_cf_selected.json --reps 6 --control-only
  n=$($PY - <<'PY'
import json
sel = {t["id"] for t in json.load(open("tasks_cf_selected.json"))}
n = sum(1 for l in open("out/responses_cf.jsonl") if l.strip() and (r := json.loads(l)).get("result_type") == "succeeded" and r["task_id"] in sel and r["skill_level"] == "control")
print(n)
PY
)
  echo "fresh control: $n/$TARGET collected"
  [ "$n" -ge "$TARGET" ] && break
  sleep 600
done
touch out/cf_stage4.done
echo "stage 4 done"
