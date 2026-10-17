#!/usr/bin/env bash
# Usage: cell.sh NAME OVERRIDES [REPS] [SEEDS] [RATES] [DURATION_S]; run from the example directory.
set -euo pipefail
: "${LAB_CONFIG:?}" "${MEASURE_NODE:?}"
M=$(cd "$(dirname "$0")/.." && pwd)
CFG="${TMPDIR:-/tmp}/aim345-measure-$1.json"
python3 "$M/tools/make_config.py" "$LAB_CONFIG" "$CFG" \
  "{\"namespace\": \"${MEASURE_NS_PREFIX:-aim345-m-}$1\", \"placement\": \"packed\", \"nodes\": [\"$MEASURE_NODE\"]}" "$2"
bash "$M/cells/run_cell.sh" "$CFG" "results/measure/$1" "${3:-1}" "${4:-345 1001 1002}" "${5:-0.25 0.5 1 2 4}" "${6:-20}"
