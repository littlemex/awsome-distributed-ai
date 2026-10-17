#!/usr/bin/env bash
# POST_HOOK: fetch the aim345_ring dumps from the engine container before cleanup.
set -uo pipefail
POD=$(kubectl -n "$NS" get pods -l component=engine -o jsonpath='{.items[0].metadata.name}')
timeout 300 kubectl -n "$NS" exec "$POD" -c engine -- sh -c 'for f in /tmp/aim345_ring_*.jsonl; do [ -f "$f" ] && echo "=== $f ===" && cat "$f"; done' > "$OUT/ring-dumps.jsonl" 2>>"$OUT/ring-fetch.log"
wc -l "$OUT/ring-dumps.jsonl"
