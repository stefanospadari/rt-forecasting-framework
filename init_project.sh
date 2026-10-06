#!/bin/bash
# Initializes THIS folder (new and clean) reusing data, models and container from the
# old project, WITHOUT copying them: it creates symbolic links (instant). The old folder is not touched.
#
#   ./init_project.sh /scratch.hpc/stefano.spadari2/project
#
# Linked  (not copied): data/, trained_models/, containers/tf-gpu.sif
# Copied  (small files): containers/requirements.lock.txt, results/benchmark_results_*.csv
# To make it independent later:  see "Make the project independent" in the README.
set -euo pipefail
cd "$(dirname "$0")"
OLD=${1:?usage: ./init_project.sh <old_project_folder>}
OLD=$(readlink -f "$OLD")
[ -d "$OLD/training" ] || { echo "ERROR: $OLD/training does not exist: is this the old project?"; exit 1; }
[ "$OLD" != "$PWD" ] || { echo "ERROR: run it from the NEW folder, not from the old one."; exit 1; }

link(){  # link <target> <link_name>
  local tgt=$1 name=$2
  if [ ! -e "$tgt" ]; then echo "  [--] $tgt does not exist, skipped"; return 0; fi
  if [ -e "$name" ] || [ -L "$name" ]; then
    if [ -L "$name" ] || [ -z "$(ls -A "$name" 2>/dev/null)" ]; then rm -rf "$name"
    else echo "  [!!] $name already exists and is not empty: left as is"; return 0; fi
  fi
  ln -s "$tgt" "$name"; echo "  [ln] $name -> $tgt"
}

echo "New project: $PWD"
echo "Old project: $OLD"
mkdir -p containers results logs

echo "== Links (no copy)"
link "$OLD/data"                    data
link "$OLD/training/trained_models" trained_models
link "$OLD/containers/tf-gpu.sif"   containers/tf-gpu.sif

echo "== Copies (small files)"
for f in "$OLD/containers/requirements.lock.txt" "$OLD/containers/pip-freeze.txt"; do
  [ -f "$f" ] && [ ! -e "containers/$(basename "$f")" ] && cp "$f" containers/ && echo "  [cp] containers/$(basename "$f")"
done
for f in "$OLD"/training/benchmark_results_*.csv; do
  [ -f "$f" ] && [ ! -e "results/$(basename "$f")" ] && cp "$f" results/ && echo "  [cp] results/$(basename "$f")"
done
echo; echo "Done. Now: ./setup.sh"
