#!/bin/bash
#SBATCH --job-name=smoke
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x_%j.out
#
# End-to-end check (container + paths + GPU) on a small setup, without touching the real models/results.
#   sbatch slurm/smoke.slurm                          (default partition above, metric temperature)
#   sbatch -p h100pcie --export=METRIC=co2 slurm/smoke.slurm
# Each job writes to smoke/<JOBID>/ -> several smoke jobs can run in parallel.
set -euo pipefail
COMMON="${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh"
[ -f "$COMMON" ] || { echo "ERROR: run sbatch from the project root"; exit 1; }
source "$COMMON"
METRIC=${METRIC:-temperature}
ROOM=${ROOM:-413}
OUT=smoke/${SLURM_JOB_ID:-local}
mkdir -p "$OUT"

banner "SMOKE START | metric=$METRIC room=$ROOM partition=${SLURM_JOB_PARTITION:-?}"
run_py -c "
import yaml
c = yaml.safe_load(open('configs/base.yaml'))
c['data']['rooms'] = [int('$ROOM')]
c['output']['path'] = '$OUT/trained_models'
c['output']['results_path'] = '$OUT/results'
c['models'] = [m for m in c['models'] if m['name'] in ('ARIMA_311', 'LSTM_DENSE_SMALL_W60_H1')]
yaml.safe_dump(c, open('$OUT/config.yaml', 'w'), sort_keys=False)
print('$OUT/config.yaml ok')"
run_py src/training.py --config "$OUT/config.yaml" --metric "$METRIC"
run_py src/testing.py  --config "$OUT/config.yaml" --metric "$METRIC" --n-origins 5
N=$(find "$OUT/trained_models" \( -name model.keras -o -name model.pkl \) 2>/dev/null | wc -l || true)
[ "$N" -eq 2 ] || { echo "SMOKE FAIL: expected 2 models, found $N"; exit 1; }
[ -s "$OUT/results/benchmark_results_sparse_${METRIC}_n5.csv" ] || { echo "SMOKE FAIL: results CSV missing"; exit 1; }
banner "SMOKE END - OK"
