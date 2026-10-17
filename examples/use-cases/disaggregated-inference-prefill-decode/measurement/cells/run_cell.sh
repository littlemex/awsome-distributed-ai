#!/usr/bin/env bash
# One measurement cell: deploy one packed stack on $MEASURE_NODE, run REPS rounds of ramps per seed with a random
# 0-10 s pause before each ramp (so the readiness-probe phase is not locked across windows), then capture and clean up.
# Usage: run_cell.sh CONFIG OUT REPS "SEEDS" "RATES" [DURATION_S]; PRE_HOOK runs once the stack is ready,
# POST_HOOK before cleanup, RAMP_EXTRA is appended to each 4.ramp.py call.
set -uo pipefail
CFG="$1"; OUT="$2"; REPS="$3"; SEEDS="$4"; RATES="$5"; DUR="${6:-20}"
: "${MEASURE_NODE:?}" "${LAB_REGION:?}" "${LAB_INSTANCE_TYPE:?}"
CFG=$(cd "$(dirname "$CFG")" && pwd)/$(basename "$CFG"); mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
M=$(cd "$(dirname "$0")/.." && pwd)
LAB=$(cd "$M/.." && pwd)
PY=${PY:-python3}
PREFIX="${MEASURE_NS_PREFIX:-aim345-m-}"
NS=$(python3 -c "import json;print(json.load(open('$CFG'))['namespace'])")
case "$NS" in "$PREFIX"*) ;; *) echo "refusing namespace $NS (must start with $PREFIX)"; exit 1;; esac
CTX=$(python3 -c "import json;print(json.load(open('$CFG'))['context'])")
[ "$(kubectl config current-context)" = "$CTX" ] || { echo "refusing: kubectl current context is not $CTX, the context in $CFG"; exit 1; }
OTHER=$(kubectl get ns -o name 2>/dev/null | grep "namespace/$PREFIX" | grep -v "namespace/$NS$" || true)
if [ -n "$OTHER" ]; then echo "refusing: another measurement stack exists: $OTHER"; exit 1; fi
LOG="$OUT/run.log"
cleanup() {
  local code=$?
  echo "=== $(date -u +%FT%TZ) capture + cleanup $NS ===" >> "$LOG"
  if [ -n "${POST_HOOK:-}" ]; then NS="$NS" OUT="$OUT" bash "$POST_HOOK" >> "$LOG" 2>&1 || true; fi
  kubectl -n "$NS" logs deploy/engines --tail=-1 > "$OUT/engine.log" 2>>"$LOG" || true
  kubectl -n "$NS" get pods -o wide > "$OUT/pods.txt" 2>>"$LOG" || true
  $PY "$M/tools/single_deploy.py" cleanup --mode disaggregated --config "$CFG" >> "$LOG" 2>&1 || true
  bash "$M/tools/check_exclusive.sh" >> "$LOG" 2>&1 || true
  exit "$code"
}
trap cleanup EXIT
bash "$M/tools/check_exclusive.sh" | tee -a "$LOG" || exit 1
cd "$LAB"
$PY "$M/tools/single_deploy.py" deploy --mode disaggregated --config "$CFG" >> "$LOG" 2>&1 || { echo "deploy failed" | tee -a "$LOG"; exit 1; }
$PY "$M/tools/single_deploy.py" wait --mode disaggregated --config "$CFG" >> "$LOG" 2>&1 || { echo "wait failed" | tee -a "$LOG"; exit 1; }
if [ -n "${PRE_HOOK:-}" ]; then NS="$NS" OUT="$OUT" bash "$PRE_HOOK" >> "$LOG" 2>&1 || true; fi
for r in $(seq 1 "$REPS"); do
  for seed in $SEEDS; do
    bash "$M/tools/check_exclusive.sh" | tee -a "$LOG" || exit 1
    pause=$(python3 -c "import random;print(round(random.uniform(0,10),2))")
    echo "=== $(date -u +%FT%TZ) rep=$r seed=$seed pause=$pause rates=$RATES ===" | tee -a "$LOG"
    sleep "$pause"
    $PY 4.ramp.py --config "$CFG" --traffic traffic/b.json --output "$OUT/seed${seed}r$r" --architecture disaggregated \
      --gpus 4 --instance-type "$LAB_INSTANCE_TYPE" --region "$LAB_REGION" --evidence-scope mechanism-validation \
      --shards 2 --rates $RATES --duration-s "$DUR" --seed "$seed" ${RAMP_EXTRA:-} >> "$LOG" 2>&1 || echo "ramp seed=$seed r=$r exit=$?" | tee -a "$LOG"
  done
done
echo "=== $(date -u +%FT%TZ) done ===" | tee -a "$LOG"
