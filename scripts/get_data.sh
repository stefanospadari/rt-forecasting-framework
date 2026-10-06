#!/bin/bash
# Downloads and prepares the KETI dataset ("Smart Building System", Kaggle, ranakrc)
# in data/archive/KETI/<room>/<metric>.csv, the layout expected by configs/base.yaml.
#
#   ./scripts/get_data.sh                 -> downloads from Kaggle (public dataset)
#   ./scripts/get_data.sh /path/archive.zip  -> uses a zip already downloaded by hand
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
    curl -fL --retry 3 -o "$ZIP" "$URL" || {
      echo "ERROR: automatic download failed."
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

# check: rooms in the config x metrics used
ROOMS=$(grep -A10 '^  rooms:' configs/base.yaml | grep -oE '^\s*- [0-9]+' | grep -oE '[0-9]+' | tr '\n' ' ')
MISS=0
for r in $ROOMS; do for m in co2 temperature humidity; do
  [ -s "$DEST/KETI/$r/$m.csv" ] || { echo "  missing: $DEST/KETI/$r/$m.csv"; MISS=1; }
done; done
[ $MISS = 0 ] && echo "Dataset OK: rooms $ROOMS (co2, temperature, humidity)" || { echo "ERROR: incomplete dataset"; exit 1; }
