#!/bin/bash
# Build + GPU test in a chain. Run it from anywhere: it moves to the project root itself.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
B=$(sbatch --parsable containers/build.sbatch)
T=$(sbatch --parsable --dependency=afterok:$B containers/test_gpu.sbatch)
echo "Build: $B  ->  tail -f logs/build_$B.log"
echo "Test:  $T  ->  logs/check_$T.log   (starts only if the build succeeds)"
