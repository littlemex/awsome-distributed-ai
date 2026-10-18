#!/usr/bin/env bash
# Usage: stage_cell.sh NAME OVERRIDES "SEEDS" "RATES" [DURATION_S]; run from the example directory.
# One disaggregated cell that brackets every single-rate ramp with SGLang's per-stage counters (stages.py),
# so the instrumented per-request records and SGLang's own timer cover the same window. The engine log is read
# after each window because the container log rotates during a cell.
set -euo pipefail
: "${LAB_CONFIG:?}" "${MEASURE_NODE:?}" "${LAB_REGION:?}" "${LAB_INSTANCE_TYPE:?}"
M=$(cd "$(dirname "$0")/.." && pwd)
NAME="$1"; SEEDS="$3"; RATES="$4"; DUR="${5:-20}"
CFG="${TMPDIR:-/tmp}/aim345-measure-$NAME.json"
OUT="$PWD/results/measure/$NAME"; mkdir -p "$OUT"
python3 "$M/tools/make_config.py" "$LAB_CONFIG" "$CFG" \
  "{\"namespace\": \"${MEASURE_NS_PREFIX:-aim345-m-}$NAME\", \"placement\": \"packed\", \"nodes\": [\"$MEASURE_NODE\"]}" "$2"
NS=$(python3 -c "import json;print(json.load(open('$CFG'))['namespace'])")
CTX=$(python3 -c "import json;print(json.load(open('$CFG'))['context'])")
[ "$(kubectl config current-context)" = "$CTX" ] || { echo "refusing: kubectl current context is not $CTX"; exit 1; }
OTHER=$(kubectl get ns -o name | grep "namespace/${MEASURE_NS_PREFIX:-aim345-m-}" | grep -v "namespace/$NS$" || true)
[ -z "$OTHER" ] || { echo "refusing: another measurement stack exists: $OTHER"; exit 1; }
cleanup() {
  local code=$?
  kubectl -n "$NS" logs deploy/engines --tail=-1 > "$OUT/engine-tail.log" 2>/dev/null || true
  python3 "$M/tools/single_deploy.py" cleanup --mode disaggregated --config "$CFG" > /dev/null 2>&1 || true
  exit "$code"
}
trap cleanup EXIT
bash "$M/tools/check_exclusive.sh"
python3 "$M/tools/single_deploy.py" deploy --mode disaggregated --config "$CFG"
python3 "$M/tools/single_deploy.py" wait --mode disaggregated --config "$CFG"
python3 4.ramp.py --config "$CFG" --traffic traffic/b.json --output "$OUT/warmup" --architecture disaggregated \
  --gpus 4 --instance-type "$LAB_INSTANCE_TYPE" --region "$LAB_REGION" --evidence-scope mechanism-validation \
  --shards 2 --rates 0.5 --duration-s 10 --seed 1
for seed in $SEEDS; do
  for rate in $RATES; do
    bash "$M/tools/check_exclusive.sh"
    W="$OUT/seed$seed-rate$rate"
    python3 stages.py snapshot --config "$CFG" --output "$W.before.json"
    t0=$(date +%s); since=$(date -u +%FT%TZ)
    python3 4.ramp.py --config "$CFG" --traffic traffic/b.json --output "$W" --architecture disaggregated \
      --gpus 4 --instance-type "$LAB_INSTANCE_TYPE" --region "$LAB_REGION" --evidence-scope mechanism-validation \
      --shards 2 --rates "$rate" --duration-s "$DUR" --seed "$seed" --skip-warmup
    t1=$(date +%s)
    python3 stages.py snapshot --config "$CFG" --output "$W.after.json"
    kubectl -n "$NS" logs deploy/engines --since-time="$since" >> "$OUT/engine.log"
    python3 stages.py diff "$W.before.json" "$W.after.json" --ramp "$W" --output "$W.stages.json"
    echo "seed=$seed rate=$rate epoch_window=[$t0,$t1]" >> "$OUT/windows.txt"
  done
done
