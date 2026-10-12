#!/bin/bash
#SBATCH --job-name=oracle
#SBATCH --partition=l40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#
# Oracle (teacher forcing) benchmark, see src/oracle_test.py. From the project root:
#   sbatch --export=ALL,METRIC=co2,ROOM=419 slurm/oracle.slurm
#   sbatch --export=ALL,METRIC=co2,ROOM=419,MODELS=LSTM_DENSE_MED_W60_H1:ARIMA_511_FFT,HORIZONS=10:30:60 slurm/oracle.slurm
# (inside --export, separate lists with ':' -- a ',' would start a new variable)
# Run it AFTER testing.py for the same config/metric: the origins are read from its predictions files.
# To match an older run kept elsewhere: REFERENCE_ROOT=<folder with <room>/<metric>/<model>/predictions/...>
# ROOM accepts a list (ROOM=413:419), TAG=<name> suffixes all outputs.
set -euo pipefail
COMMON="${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh"
[ -f "$COMMON" ] || { echo "ERROR: run sbatch from the project root (not found: $COMMON)"; exit 1; }
source "$COMMON"
: "${METRIC:?set METRIC=<one of data.metrics in the config> with --export}"
CONFIG=${CONFIG:-configs/base.yaml}

ARGS=(--config "$CONFIG" --metric "$METRIC")
[ -n "${ROOM:-}" ]     && ARGS+=(--room "${ROOM//:/,}")
[ -n "${MODELS:-}" ]   && ARGS+=(--models "${MODELS//:/,}")
[ -n "${HORIZONS:-}" ] && ARGS+=(--horizons "${HORIZONS//:/,}")
[ -n "${REFERENCE_ROOT:-}" ] && ARGS+=(--reference-root "$REFERENCE_ROOT")
[ -n "${TAG:-}" ]      && ARGS+=(--tag "$TAG")
[ "${REQUIRE_REFERENCE:-1}" = 1 ] && ARGS+=(--require-reference)   # REQUIRE_REFERENCE=0 to allow the config-rule fallback

banner "START oracle | config=$CONFIG metric=$METRIC room=${ROOM:-all} models=${MODELS:-default}"
run_py src/oracle_test.py "${ARGS[@]}"
banner "END"
