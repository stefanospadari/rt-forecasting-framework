#!/bin/bash
# Prepares the project after a "git clone" (or after reorganize.sh): creates the folders
# and checks that data and container are there. It changes no existing files.
set -uo pipefail
cd "$(dirname "$0")"
# Usage: ./setup.sh [config]   (default configs/base.yaml)
ROOT=$PWD
ok(){ echo "  [ok]  $*"; }; ko(){ echo "  [!!]  $*"; MISSING=1; }
MISSING=0

echo "Project: $ROOT"
echo "== Folders"
for d in data trained_models results logs; do
  mkdir -p "$d"
  if [ -L "$d" ]; then ok "$d/ -> $(readlink -f "$d")  (link)"; else ok "$d/"; fi
done

CONFIG=${1:-configs/base.yaml}
echo "== Dataset (from $CONFIG: <data.path>/<room>/<metric>.csv)"
if INFO=$(python3 scripts/config_info.py "$CONFIG" 2>/dev/null); then
  DPATH=$(sed -n 1p <<< "$INFO"); ROOMS=$(sed -n 2p <<< "$INFO"); METRICS=$(sed -n 3p <<< "$INFO")
  DMISS=0
  for r in $ROOMS; do for m in $METRICS; do
    [ -s "$DPATH/$r/$m.csv" ] || { ko "missing $DPATH/$r/$m.csv"; DMISS=1; }
  done; done
  [ "$DMISS" = 0 ] && ok "rooms: $ROOMS | metrics: $METRICS"
else
  echo "  [--]  check skipped: python3 + PyYAML are not available on this host"
  echo "        (the jobs check the data anyway when they start)"
fi

echo "== Container"
if [ -f containers/tf-gpu.sif ]; then ok "containers/tf-gpu.sif ($(du -hL containers/tf-gpu.sif | cut -f1))$([ -L containers/tf-gpu.sif ] && echo "  (link)")"
else ko "containers/tf-gpu.sif missing -> ./containers/submit_build.sh (build + GPU test)"; fi
[ -s containers/requirements.lock.txt ] && ok "containers/requirements.lock.txt" || ko "containers/requirements.lock.txt missing"

echo "== Trained models"
N=$(find -L trained_models -name model.keras -o -name model.pkl 2>/dev/null | wc -l)
echo "  $N models found in trained_models/"

echo
[ "$MISSING" = 0 ] && echo "All set. Example: sbatch --export=METRIC=co2 slurm/train.slurm" \
                   || echo "Some items are missing (see [!!] above)."
