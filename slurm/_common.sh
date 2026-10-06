# Shared by train.slurm / test.slurm. Always submit from the project root:
#   cd <project> && sbatch slurm/train.slurm
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
SIF="$PROJECT_ROOT/containers/tf-gpu.sif"
if [ ! -f "$SIF" ] || [ ! -d "$PROJECT_ROOT/src" ]; then
    echo "ERROR: run sbatch from the project root (the folder with src/ and containers/tf-gpu.sif)."
    echo "       submitted from: $PROJECT_ROOT"
    exit 1
fi
export PROJECT_ROOT
cd "$PROJECT_ROOT"
# bind the project + the real targets of any links (data/, trained_models/ ...)
BINDS="$PROJECT_ROOT"
for d in data trained_models results logs; do
    [ -L "$d" ] && BINDS="$BINDS,$(readlink -f "$d")"
done
run_py(){ apptainer exec --nv --bind "$BINDS" "$SIF" python -u "$@"; }
banner(){
    echo "========================================"
    echo "Job ID:   ${SLURM_JOB_ID:-?} (${SLURM_JOB_NAME:-?})"
    echo "Node:     ${SLURMD_NODENAME:-$(hostname)} | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
    echo "Project:  $PROJECT_ROOT"
    echo "$(date)  $*"
    echo "========================================"
}
