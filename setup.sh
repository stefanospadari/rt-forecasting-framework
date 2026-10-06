#!/bin/bash
# Prepares the project after a "git clone" (or after reorganize.sh): creates the folders
# and checks that data and container are there. It changes no existing files.
set -uo pipefail
cd "$(dirname "$0")"
ROOT=$PWD
ok(){ echo "  [ok]  $*"; }; ko(){ echo "  [!!]  $*"; MISSING=1; }
MISSING=0

echo "Project: $ROOT"
echo "== Folders"
for d in data trained_models results logs; do
  mkdir -p "$d"
  if [ -L "$d" ]; then ok "$d/ -> $(readlink -f "$d")  (link)"; else ok "$d/"; fi
done

echo "== Dataset (data/archive/KETI/<room>/<metric>.csv)"
ROOMS=$(python3 -c "import yaml;print(*yaml.safe_load(open('configs/base.yaml'))['data']['rooms'])" 2>/dev/null || echo "413 419 442 510 621")
for r in $ROOMS; do
  for m in co2 temperature humidity; do
    f=data/archive/KETI/$r/$m.csv
    [ -s "$f" ] || ko "missing $f"
  done
done
[ "$MISSING" = 0 ] && ok "all CSVs present for rooms: $ROOMS"

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
