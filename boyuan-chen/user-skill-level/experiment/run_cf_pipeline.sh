#!/bin/bash
# v2.1 Codeforces pipeline, resumable end to end:
#   1. finish the control-only screening pilot (44 tasks x 3)
#   2. score it and select the 24-task battery (tasks_cf_selected.json)
#   3. run the full 9-condition x 3-rep matrix on the selected tasks
# Every stage is idempotent; the runner skips already-collected custom_ids,
# so the pilot's control answers are reused as the control cells of step 3.
# Re-run this script after any interruption. Needs GROQ_API_KEY.
set -u
cd "$(dirname "$0")"
PY=../.venv/bin/python
export FORCE_IPV4=1
count() { grep -c '"result_type": "succeeded"' out/responses_cf.jsonl 2>/dev/null || echo 0; }
controls() { grep -c '"custom_id": "cf_[A-Za-z0-9]*_control_none_r[123]"' out/responses_cf.jsonl 2>/dev/null || echo 0; }

echo "== stage 1: control pilot ($(controls)/132 done) =="
while [ "$(controls)" -lt 132 ]; do
  $PY -u run_experiment.py --suite cf --control-only --reps 3
  [ "$(controls)" -lt 132 ] && { echo "pilot incomplete ($(controls)/132); sleeping 10 min before resume"; sleep 600; }
done

if [ ! -f tasks_cf_selected.json ]; then
  echo "== stage 2: score pilot and select 24 tasks =="
  $PY score_cf.py --out out/scores_cf_pilot.csv --workers 4
  $PY pilot_cf_summary.py out/scores_cf_pilot.csv | tee out/cf_pilot_summary.txt
fi

echo "== stage 3: full matrix on tasks_cf_selected.json =="
TARGET=648
while true; do
  $PY -u run_experiment.py --suite cf --tasks-file tasks_cf_selected.json --reps 3
  n=$($PY - <<'PY'
import json
sel = {t["id"] for t in json.load(open("tasks_cf_selected.json"))}
n = sum(1 for l in open("out/responses_cf.jsonl") if l.strip() and (r := json.loads(l)).get("result_type") == "succeeded" and r["task_id"] in sel)
print(n)
PY
)
  echo "full matrix: $n/$TARGET collected"
  [ "$n" -ge "$TARGET" ] && break
  echo "sleeping 10 min before resume"; sleep 600
done
touch out/cf_full_run.done
echo "== done =="
