#!/bin/sh
# Assemble the Space (code + model checkpoints) in a staging dir and upload it. Usage: space/deploy.sh <hf-user>/<space-name>
set -e
cd "$(dirname "$0")/.."
S=$(mktemp -d)/space; mkdir -p "$S/src" "$S/models"
cp space/Dockerfile space/requirements.txt space/serve.py space/README.md "$S/"
cp src/app.py src/common.py src/clap_zero.py "$S/src/"
for m in cnn_v2 cnn_v3_8cls cnn_v4_w48 clap_head; do mkdir -p "$S/models/$m"; cp models/$m/best.pt "$S/models/$m/"; done
cp models/clap_head/config.json "$S/models/clap_head/"
printf '*.pt filter=lfs diff=lfs merge=lfs -text\n' > "$S/.gitattributes"
du -sh "$S"
../.venv/bin/hf upload "$1" "$S" . --repo-type space
