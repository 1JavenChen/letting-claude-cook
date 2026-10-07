#!/bin/bash
# Run (or resume) a list of AI games, at most N at a time.
# Usage: ./run_queue.sh N "args for game 1" "args for game 2" ...
# Each game's output is appended to results/llm/<game>/run.log.
N=$1; shift
run_one() {
  name=$(../.venv/bin/python -c "import sys; a=sys.argv[1:]; g=lambda k,d=None: a[a.index(k)+1] if k in a else d; e=g('--effort'); print(f\"{g('--model','haiku')}{'_'+e if e else ''}_{g('--prefix','P1')}_a{float(g('--alpha','1')):g}_{g('--rounds','5')}r_seed{g('--seed','1')}\")" $1)
  mkdir -p results/llm/$name
  ../.venv/bin/python -u run_llm_game.py $1 >> results/llm/$name/run.log 2>&1
  echo "finished: $name ($(tail -1 results/llm/$name/run.log))"
}
export -f run_one
printf '%s\n' "$@" | xargs -P "$N" -I{} bash -c 'run_one "{}"'
echo "QUEUE DONE"
