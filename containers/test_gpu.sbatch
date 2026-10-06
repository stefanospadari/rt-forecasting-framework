#!/bin/bash
#SBATCH --job-name=test-tf-sif
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=logs/check_%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
echo "Node: $(hostname)"; nvidia-smi --query-gpu=name,driver_version --format=csv
apptainer exec --nv containers/tf-gpu.sif python -u /opt/check_gpu.py
