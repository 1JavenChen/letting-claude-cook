#!/bin/zsh
# Single-sex experiment: generate male-only and female-only datasets, train one model on each, cross-evaluate.
F=/Users/karoldabrowski/Desktop/College/FYS; PY=$F/.venv/bin/python; cd $F
log() { echo "=== $(date '+%H:%M:%S') $*"; }
gen() {  # sex outdir
  if [ -f $2/labels.csv ] && [ "$(wc -l < $2/labels.csv)" -ge 23500 ]; then log "$2 already generated"; return; fi
  log "generating $2"
  $PY src/make_mixtures.py --subset dev-clean,train-clean-100 --n_train 20000 --n_val 1500 --n_test 2000 --sex $1 --out $2 > logs/gen_$(basename $2).log 2>&1 && log "$2 done: $(tail -1 logs/gen_$(basename $2).log)" || { log "GENERATION FAILED $2"; exit 1; }
}
gen M data/mixtures_male
gen F data/mixtures_female
log "waiting for AC power before training"
while ! pmset -g batt | grep -q "AC Power"; do sleep 60; done
log "on AC power; training"
$PY src/train.py --data data/mixtures_male --epochs 30 --out models/cnn_v8_male.pt > logs/cnn_v8_male.log 2>&1 && log "cnn_v8_male: $(tail -1 logs/cnn_v8_male.log)" || log "cnn_v8_male FAILED"
$PY src/train.py --data data/mixtures_female --epochs 30 --out models/cnn_v9_female.pt > logs/cnn_v9_female.log 2>&1 && log "cnn_v9_female: $(tail -1 logs/cnn_v9_female.log)" || log "cnn_v9_female FAILED"
log "evaluating"
ev() {  # model data tag
  $PY src/evaluate.py --model models/$1.pt --data $2 --out results/$3 > /dev/null 2>&1 && log "$3: $($PY -c "import json;m=json.load(open('results/$3/metrics.json'));print('exact', round(m['n_speakers']['accuracy'],3), 'within1', round(m['n_speakers']['within_1'],3))")" || log "eval $3 FAILED"
}
ev cnn_v8_male   data/mixtures_male   cnn_v8_male_on_male
ev cnn_v4        data/mixtures_male   cnn_v4_on_male
ev cnn_v9_female data/mixtures_male   cnn_v9_female_on_male
ev cnn_v9_female data/mixtures_female cnn_v9_female_on_female
ev cnn_v4        data/mixtures_female cnn_v4_on_female
ev cnn_v8_male   data/mixtures_female cnn_v8_male_on_female
ev cnn_v8_male   data/mixtures_v3     cnn_v8_male_on_v3test
ev cnn_v9_female data/mixtures_v3     cnn_v9_female_on_v3test
log "DONE"
