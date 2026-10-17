#!/usr/bin/env bash
# Refuses to continue when a running pod outside kube-system, nvidia-device-plugin and this
# measurement's own namespaces ($MEASURE_NS_PREFIX*) is on $MEASURE_NODE, so a cell never shares GPUs.
set -euo pipefail
NODE="${MEASURE_NODE:?set MEASURE_NODE to the node hostname}"
PREFIX="${MEASURE_NS_PREFIX:-aim345-m-}"
foreign=$(kubectl get pods -A --field-selector spec.nodeName=$NODE -o json | \
  python3 -c "
import json,sys
d=json.load(sys.stdin)
allowed_ns={'kube-system','nvidia-device-plugin'}
for p in d['items']:
    ns=p['metadata']['namespace']
    phase=p['status'].get('phase')
    if ns in allowed_ns or ns.startswith('$PREFIX'):
        continue
    if phase in ('Succeeded','Failed'):
        continue
    print(f\"{ns}/{p['metadata']['name']} phase={phase}\")
")
if [ -n "$foreign" ]; then
  echo "FOREIGN POD on $NODE:"
  echo "$foreign"
  echo "EXCLUSIVITY CHECK FAILED"
  exit 1
fi
echo "exclusivity OK on $NODE: $(date -u +%FT%TZ)"
