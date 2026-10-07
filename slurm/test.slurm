#!/bin/bash
#SBATCH --job-name=test
#SBATCH --partition=l40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x_%j.out
#
# From the project root:
#   sbatch --export=METRIC=humidity slurm/test.slurm
#   sbatch --export=METRIC=co2,N_ORIGINS=10 slurm/test.slurm        (quick run, results file gets _n10)
#   sbatch --export=METRIC=co2,ROOM=413 slurm/test.slurm
#   sbatch --export=METRIC=co2,CONFIG=configs/my_experiment.yaml slurm/test.slurm
# NB: the paper's latencies are measured on L40 -> do not change the partition for the official runs.
set -euo pipefail
COMMON="${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh"
[ -f "$COMMON" ] || { echo "ERROR: run sbatch from the project root (not found: $COMMON)"; exit 1; }
source "$COMMON"
: "${METRIC:?set METRIC=<one of data.metrics in the config> with --export}"
CONFIG=${CONFIG:-configs/base.yaml}
[ -f "$CONFIG" ] || { echo "ERROR: config not found: $CONFIG"; exit 1; }

ARGS=(--config "$CONFIG" --metric "$METRIC")
[ -n "${N_ORIGINS:-}" ] && ARGS+=(--n-origins "$N_ORIGINS")
[ -n "${ROOM:-}" ]      && ARGS+=(--room "$ROOM")

banner "START testing | config=$CONFIG metric=$METRIC n_origins=${N_ORIGINS:-config} room=${ROOM:-all}"
run_py src/testing.py "${ARGS[@]}"
banner "END"
