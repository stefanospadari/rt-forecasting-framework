#!/bin/bash
# Submits a set of smoke jobs on different partitions/metrics/rooms to validate the deployment.
#   ./slurm/validate.sh                 -> submits the jobs
#   ./slurm/validate.sh report          -> summary of the results (after the jobs finish)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs smoke
LIST=smoke/validate_jobs.txt

if [ "${1:-}" = "report" ]; then
  [ -f "$LIST" ] || { echo "No validation submitted."; exit 1; }
  printf "%-8s %-10s %-12s %-5s %-6s %-14s %s\n" JOBID PARTITION METRIC ROOM GPU_TF NODE_GPU RESULT
  while read -r id part metric room; do
    log=logs/smoke_$id.out
    state=$(squeue -h -j "$id" -o %T 2>/dev/null || true)
    if [ -n "$state" ]; then res="($state)"
    elif grep -q "SMOKE END - OK" "$log" 2>/dev/null; then res="OK"
    else res="FAIL -> tail -30 $log"; fi
    gpu=$(grep -q "GPU detected: 1" "$log" 2>/dev/null && echo yes || echo no)
    node=$(grep -m1 -oP "GPU: \K.*" "$log" 2>/dev/null | head -c 14 || true)
    printf "%-8s %-10s %-12s %-5s %-6s %-14s %s\n" "$id" "$part" "$metric" "$room" "$gpu" "$node" "$res"
  done < "$LIST"
  exit 0
fi

# partition  metric  room   -> covers all GPU types, all 3 metrics and more than one room
COMBOS="l40 temperature 413
l40s co2 419
h100pcie humidity 442
h100sxm5 temperature 510
a100 co2 621"

: > "$LIST"
while read -r part metric room; do
  id=$(sbatch --parsable -p "$part" --export=ALL,METRIC=$metric,ROOM=$room slurm/smoke.slurm)
  echo "$id $part $metric $room" >> "$LIST"
  echo "submitted $id  ($part, $metric, room $room)"
done <<< "$COMBOS"
echo; echo "When they finish:  ./slurm/validate.sh report"
