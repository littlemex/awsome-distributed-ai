# Pre-hook: in the background, after NSYS_DELAY s, start Nsight Systems collection on the engine container's launch
# session for NSYS_WINDOW s, export to sqlite and summarise inside the container with nsys_an.py.
set -uo pipefail
H=$(cd "$(dirname "$0")" && pwd)
D=${NSYS_DELAY:-40}; W=${NSYS_WINDOW:-8}; SO=${NSYS_START_OPTS:---sample=none --cpuctxsw=none --gpuctxsw=true}
POD=$(kubectl -n "$NS" get pod -l app.kubernetes.io/part-of=aim345-lab -o name | grep engines | head -1 | cut -d/ -f2)
kubectl -n "$NS" cp "$H/nsys_an.py" "$POD":/tmp/nsys_an.py -c engine
kubectl -n "$NS" cp "$H/nsys_an2.py" "$POD":/tmp/nsys_an2.py -c engine
kubectl -n "$NS" cp "$H/nsys_an3.py" "$POD":/tmp/nsys_an3.py -c engine
kubectl -n "$NS" cp "$H/nsys_an4.py" "$POD":/tmp/nsys_an4.py -c engine; kubectl -n "$NS" cp "$H/nsys_an5.py" "$POD":/tmp/nsys_an5.py -c engine
# decode physical GPUs: CUDA_VISIBLE_DEVICES of a scheduler whose launcher serves --port 30001
DEC=$(kubectl -n "$NS" exec "$POD" -c engine -- sh -c 'for p in $(pgrep -f "sglang.launch_server"); do tr "\0" " " < /proc/$p/cmdline | grep -q "\-\-port 30001" && for c in $(pgrep -P $p -f scheduler_TP | head -1); do tr "\0" "\n" < /proc/$c/environ | grep ^CUDA_VISIBLE_DEVICES= | cut -d= -f2; done; done | head -1')
echo "decode CUDA_VISIBLE_DEVICES: $DEC" | tee -a "$OUT/nsys-devs.txt"
DP0=${DEC%%,*}; DP1=${DEC##*,}
# GPU metrics only on the GPUs this pod's engine processes use (indices from nvidia-smi).
DEVS=$(kubectl -n "$NS" exec "$POD" -c engine -- sh -c 'U=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | sort -u); nvidia-smi --query-gpu=index,uuid --format=csv,noheader | while IFS=, read i u; do echo "$U" | grep -q "$(echo $u | tr -d " ")" && echo $i; done | paste -sd, -')
echo "gpu metrics devices: $DEVS" | tee "$OUT/nsys-devs.txt"
( B=$(kubectl -n "$NS" logs "$POD" -c engine 2>/dev/null | grep -c "Decode batch"); for i in $(seq 1 300); do N=$(kubectl -n "$NS" logs "$POD" -c engine 2>/dev/null | grep -c "Decode batch"); [ "$N" -gt "$B" ] && break; sleep 2; done; echo "decode traffic seen after $((i*2)) s" >> "$OUT/nsys-devs.txt"; sleep "$D"; kubectl -n "$NS" exec "$POD" -c engine -- sh -c "mkdir -p /tmp/nsys; nsys sessions list; nsys start --session=a345 $SO --gpu-metrics-devices=$DEVS --output=/tmp/nsys/run --force-overwrite=true && sleep $W && nsys stop --session=a345; ls -la /tmp/nsys; nsys export --type sqlite --force-overwrite=true -o /tmp/nsys/run.sqlite /tmp/nsys/run.nsys-rep && python3 /tmp/nsys_an.py /tmp/nsys/run.sqlite > /tmp/nsys/summary.txt 2>&1; python3 /tmp/nsys_an2.py /tmp/nsys/run.sqlite $DP0 $DP1 > /tmp/nsys/summary2.txt 2>&1; python3 /tmp/nsys_an3.py /tmp/nsys/run.sqlite > /tmp/nsys/summary3.txt 2>&1; python3 /tmp/nsys_an4.py /tmp/nsys/run.sqlite > /tmp/nsys/summary4.txt 2>&1; python3 /tmp/nsys_an5.py /tmp/nsys/run.sqlite > /tmp/nsys/summary5.txt 2>&1; nsys stats --report cuda_gpu_mem_time_sum,cuda_gpu_kern_sum --format csv -o /tmp/nsys/stats /tmp/nsys/run.nsys-rep > /dev/null 2>&1; echo nsys-done" > "$OUT/nsys-run.txt" 2>&1 ) &
echo "nsys collection scheduled: delay $D s, window $W s, opts $SO"
