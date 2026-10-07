#!/bin/bash
# Downloads and prepares the KETI dataset ("Smart Building System", Kaggle, ranakrc)
# in data/archive/KETI/<room>/<metric>.csv, the layout expected by configs/base.yaml.
#
#   ./scripts/get_data.sh                 -> downloads from Kaggle (public dataset)
#   ./scripts/get_data.sh /path/archive.zip  -> uses a zip already downloaded by hand
#
# If Kaggle asks for authentication: create an API token (kaggle.com -> Settings -> API ->
# "Create New Token") and either put kaggle.json in ~/.kaggle/ or export
# KAGGLE_USERNAME and KAGGLE_KEY before running the script.
#
# Manual download (if the automatic one fails): https://www.kaggle.com/datasets/ranakrc/smart-building-system
# -> "Download" -> archive.zip, then pass the path to this script.
set -euo pipefail
cd "$(dirname "$0")/.."
URL="https://www.kaggle.com/api/v1/datasets/download/ranakrc/smart-building-system"
DEST=data/archive
mkdir -p "$DEST"

if [ -d "$DEST/KETI" ] && [ -n "$(ls -A "$DEST/KETI")" ]; then
  echo "data/archive/KETI already present: nothing to do."
else
  ZIP=${1:-}
  if [ -z "$ZIP" ]; then
    ZIP=data/smart-building-system.zip
    echo "Downloading from Kaggle -> $ZIP"
    AUTH=()
    if [ -n "${KAGGLE_USERNAME:-}" ] && [ -n "${KAGGLE_KEY:-}" ]; then
      AUTH=(-u "$KAGGLE_USERNAME:$KAGGLE_KEY")
    elif [ -f "$HOME/.kaggle/kaggle.json" ]; then
      U=$(python3 -c "import json;print(json.load(open('$HOME/.kaggle/kaggle.json'))['username'])")
      K=$(python3 -c "import json;print(json.load(open('$HOME/.kaggle/kaggle.json'))['key'])")
      AUTH=(-u "$U:$K")
    fi
    # stop if it does not connect within 30 s or stays below 1 KB/s for 60 s (instead of hanging)
    curl -fL --retry 2 --connect-timeout 30 --speed-limit 1024 --speed-time 60 \
         ${AUTH[@]+"${AUTH[@]}"} -o "$ZIP" "$URL" || {
      rm -f "$ZIP"
      echo "ERROR: automatic download failed (no connection, or Kaggle requires a token: see the header of this script)."
      echo "Download it by hand from https://www.kaggle.com/datasets/ranakrc/smart-building-system"
      echo "and run:  ./scripts/get_data.sh /path/archive.zip"; exit 1; }
  fi
  [ -f "$ZIP" ] || { echo "ERROR: $ZIP does not exist"; exit 1; }
  echo "Extracting $ZIP ..."
  TMP=$(mktemp -d data/.unzip.XXXX)
  unzip -q "$ZIP" -d "$TMP"
  K=$(find "$TMP" -type d -name KETI | head -1)
  [ -n "$K" ] || { echo "ERROR: no KETI folder in the zip"; rm -rf "$TMP"; exit 1; }
  mv "$K" "$DEST/KETI"; rm -rf "$TMP"
fi

# summary of what was extracted (the check against the config is done by ./setup.sh)
NR=$(find "$DEST/KETI" -mindepth 1 -maxdepth 1 -type d | wc -l)
MS=$(find "$DEST/KETI" -mindepth 2 -maxdepth 2 -name "*.csv" -printf "%f\n" | sed 's/\.csv$//' | sort -u | tr '\n' ' ')
echo "Dataset ready in $DEST/KETI: $NR rooms, available metrics: $MS"
echo "Next: ./setup.sh  (checks that the rooms/metrics of the config are there)"
