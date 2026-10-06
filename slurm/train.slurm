#!/bin/bash
#SBATCH --job-name=train
#SBATCH --partition=l40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x_%j.out
#
# From the project root:
#   sbatch --export=METRIC=co2 slurm/train.slurm
#   sbatch --export=METRIC=temperature,MODEL_INDEX=3,ROOM=413 slurm/train.slurm
#   sbatch --export=METRIC=co2,CONFIG=configs/my_experiment.yaml slurm/train.slurm
set -euo pipefail
COMMON="${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh"
[ -f "$COMMON" ] || { echo "ERROR: run sbatch from the project root (not found: $COMMON)"; exit 1; }
source "$COMMON"
: "${METRIC:?set METRIC=co2|temperature|humidity with --export}"
CONFIG=${CONFIG:-configs/base.yaml}
[ -f "$CONFIG" ] || { echo "ERROR: config not found: $CONFIG"; exit 1; }

ARGS=(--config "$CONFIG" --metric "$METRIC")
[ -n "${ROOM:-}" ]        && ARGS+=(--room "$ROOM")
[ -n "${MODEL_INDEX:-}" ] && ARGS+=(--model-index "$MODEL_INDEX")

banner "START training | config=$CONFIG metric=$METRIC room=${ROOM:-all} model_index=${MODEL_INDEX:-all}"
run_py src/training.py "${ARGS[@]}"
banner "END"
