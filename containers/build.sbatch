#!/bin/bash
#SBATCH --job-name=build-tf-sif
#SBATCH --partition=sbuild
#SBATCH --cpus-per-task=2
#SBATCH --mem=6000
#SBATCH --time=04:00:00
#SBATCH --output=logs/build_%j.log
#
# Build tf-gpu.sif. From the project root:
#   sbatch containers/build.sbatch        (or ./containers/submit_build.sh, which also chains the GPU test)
# Follow it live with:  tail -f logs/build_<JOBID>.log

set -euo pipefail
PROJECT=${SLURM_SUBMIT_DIR:-$PWD}
CDIR=$PROJECT/containers
cd "$CDIR"

ts(){ echo; echo "[$(date '+%F %T')] ===== $* ====="; }
trap 'rm -rf "/tmp/$USER-apptainer"' EXIT
trap 'echo; echo "[$(date "+%F %T")] !!!!! FAILED at line $LINENO (exit $?) !!!!!"' ERR

ts "INFO"
echo "Node: $(hostname) | Job: $SLURM_JOB_ID | CPU: $SLURM_CPUS_PER_TASK"
apptainer --version
df -h "$PROJECT" | tail -1

# --- 1) package lock ------------------------------------------------------
ts "1/3 requirements.lock.txt"
if [ ! -s requirements.lock.txt ]; then
    echo "ERROR: containers/requirements.lock.txt is missing or empty."
    echo "It holds the exact versions of the Python packages (pip freeze) and must be committed to the repo."
    exit 1
fi
echo "packages in the lock: $(wc -l < requirements.lock.txt)"
cat requirements.lock.txt

# --- 2) build ----------------------------------------------------------
ts "2/3 apptainer build"
# Apptainer cache/tmp: on the DISI cluster use scratch (home has little quota); elsewhere $HOME
SCR=/scratch.hpc/$USER; [ -d "$SCR" ] || SCR=$HOME
export APPTAINER_CACHEDIR=${APPTAINER_CACHEDIR:-$SCR/.apptainer/cache}
# Working directory: local disk of the node if it has >= 18 GB free (much faster
# than the network scratch), otherwise fall back to scratch.
LOCAL_FREE_GB=$(df -BG --output=avail /tmp | tail -1 | tr -dc 0-9)
if [ "${LOCAL_FREE_GB:-0}" -ge 18 ]; then
    export APPTAINER_TMPDIR=/tmp/$USER-apptainer
else
    export APPTAINER_TMPDIR=$SCR/.apptainer/tmp
fi
echo "APPTAINER_TMPDIR=$APPTAINER_TMPDIR  (free on local /tmp: ${LOCAL_FREE_GB}G)"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
export PYTHONUNBUFFERED=1
rm -f tf-gpu.sif.new
# --ignore-fakeroot-command: the host's libfakeroot requires GLIBC 2.38 and
# does not work inside the image (glibc 2.36). The root-mapped namespace is enough for pip.
stdbuf -oL -eL apptainer --verbose build --ignore-fakeroot-command tf-gpu.sif.new tf-gpu.def
mv -f tf-gpu.sif.new tf-gpu.sif

# --- 3) summary --------------------------------------------------------
ts "3/3 RESULT"
ls -lh tf-gpu.sif
apptainer exec tf-gpu.sif cat /opt/pip-freeze.txt > pip-freeze.txt
grep -iE '^(tensorflow|keras|numpy|nvidia-cudnn|nvidia-cuda-runtime)' pip-freeze.txt
ts "BUILD OK"
