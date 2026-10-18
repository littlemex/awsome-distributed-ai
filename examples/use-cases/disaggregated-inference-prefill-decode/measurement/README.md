# Measurement tools for the same-node KV transfer investigation

This directory reproduces the measurements behind the proposal to run the AIM345 disaggregated stack with NIXL 1.4.0, `FI_EFA_ENABLE_SHM_TRANSFER=0` and KV page size 64. The `measurement/` directory is measurement-only: its Dockerfiles patch the engine image for logging, and nothing in it is meant for the workshop or for merging.

Each experiment deploys one packed stack (the prefill and decode engines in one pod, two GPUs each) on one node, sends the long-context traffic (shape B: 8,192 input tokens, 256 output tokens) and reads the result from logs, ring-buffer records written by a patch, or EFA counters. The lab scripts in the parent directory are the investigation's harness: `paired.py` and `deployment.py` accept the opt-in config keys used below (`pin_nixl_version`, `extra_env`, `page_size_tokens`, `efa_per_worker`, `patch_dispatch_logging`, `patch_room_logging`, `tdev_shim`, `nsys_launch`, `engine_privileged`, `decode_no_prog_thread`), and `4.ramp.py` sends traffic from the in-cluster runner pods.

## Layout

| Path | Contents |
|---|---|
| `patches/` | Python patches applied at image build time. `aim345_ring.py` is the shared in-process recorder; every other patch appends wrappers or one-line edits to the pinned SGLang 0.5.12.post1 |
| `images/` | One Dockerfile per patch set. Each takes `BASE_IMAGE` (the workshop engine image) and uses `patches/` as the build context |
| `shims/` | `tdev.c`: an `LD_PRELOAD` shim that logs each new thread's CUDA device and can set it (`tdev_shim`, `tdev_fix`). `tdev_twoleg.c`: the variant that adds the host-staged two-leg copy as an injected load |
| `tools/` | `single_deploy.py` (deploy, wait, clean up one cell), `check_exclusive.sh` (stop if another workload is on the node), `make_config.py`, `cell.sh` and `build.sh` (one cell; one image), `ring_fetch_hook.sh`, `nsys_pre.sh` and `nsys_an*.py` (Nsight Systems capture and in-container summaries), `nixl_bench.py` (one NIXL write between two GPUs outside SGLang), `unified_stages.py` (transfer-stage timing from engine logs; `stages.py` for the disaggregated stack is in the example directory), `diagnose-transport.py`, `descriptor_sweep.py`, `load_driver.py`, `a5_perrank.py` |
| `manifests/efa-probe.yaml` | A `hostNetwork: true` pod on the same node; the packed engine pod's network namespace does not expose the EFA `hw_counters` |
| `cells/run_cell.sh` | One cell: gate, deploy, wait, optional `PRE_HOOK`, ramps per seed with a random 0-10 s pause, optional `POST_HOOK`, capture, clean up |

## Setup

Run once per shell. The node must be dedicated to this measurement; `check_exclusive.sh` refuses to continue otherwise. The kubectl current context must be the `context` in `LAB_CONFIG`, because the hooks and the node check call kubectl without `--context`; `run_cell.sh` refuses to start otherwise. `LAB_CONFIG` is a filled-in `config.json` for packed placement (see the parent README, "Prepare the image and configuration"). Building images needs Docker and push access to `LAB_REGISTRY`.

```bash
export LAB_REPO=https://github.com/littlemex/awsome-distributed-ai.git
export LAB_BRANCH=measure/aim345-disagg-investigation
export LAB_WORKDIR="$HOME/aim345-measure"
export LAB_CONFIG="$HOME/config.packed.json"
export LAB_REGISTRY=my-registry.example.com/aim345
export BASE_IMAGE="$LAB_REGISTRY/aim345:sglang-0.5.12.post1-nixl-1.1.0"
export MEASURE_NODE=my-node-hostname
export MEASURE_NS_PREFIX=aim345-m-
export LAB_REGION=us-west-2
export LAB_INSTANCE_TYPE=g7.48xlarge
rm -rf "$LAB_WORKDIR"
git clone -b "$LAB_BRANCH" "$LAB_REPO" "$LAB_WORKDIR"
cd "$LAB_WORKDIR/examples/use-cases/disaggregated-inference-prefill-decode"
python3 -m venv .venv
source .venv/bin/activate
pip install -q -r requirements.txt
python3 -m unittest discover -p "test_*.py"
python3 3.generate-long-context.py --output traffic/b.json
```

`measurement/tools/build.sh NAME` builds and pushes `images/Dockerfile.NAME`. `measurement/tools/cell.sh NAME OVERRIDES [REPS] [SEEDS] [RATES] [DURATION_S]` writes the cell's config and runs it; results go to `results/measure/NAME/`. Every block below runs from the example directory of this shell, with the variables above exported and the virtual environment active.

## Findings and how to reproduce each

| Finding | What is measured | Image | Config keys | Recorded in |
|---|---|---|---|---|
| SAR fallback on the shm path | `cuMemGetAddressRange` failures and `fallback to using SAR` lines in the engine log, NIXL 1.1.0 vs 1.4.0 | base | `pin_nixl_version`, `extra_env.FI_LOG_LEVEL=warn` | `NIXL14-SWEEP.md` |
| EFA byte proof | `rdma_write_bytes` on the pod's EFA device before and after one ramp, against 27 layers x 576 x 2 bytes per token per rank | base | `extra_env.FI_EFA_ENABLE_SHM_TRANSFER=0`, `page_size_tokens=64` | `EFA-DEVICE-PROOF.md` |
| A. Completion wait and the shared EFA device | Per-rank completion wait with one device shared by both ranks vs one device per rank; NIXL alone with `nixl_bench.py` | `oi5d` (per-rank `AIM345_EFA_IFACES`) | `efa_per_worker=2`, `extra_env.AIM345_EFA_IFACES` | `OPEN-ITEMS-NODE5.md` A, `REVIEW-TEAM.md` |
| B. Delay before the transfer timer | Requests split by the chunk budget vs unsplit; control `AIM345_SUPPRESS_TINY_CHUNK=1` | `oi3` | `extra_env` | `OPEN-ITEMS-NODE3.md` B |
| C. Gap against SGLang's own timer | Per-request intervals vs SGLang's counter per window | `oi3` | | `OPEN-ITEMS-NODE3.md` C |
| G. Completion noticed once per scheduler loop | Transfer done vs loop boundaries; control `AIM345_EXTRA_POLL=1`; CUPTI window with `AIM345_PROF` | `oi3b`, `oi5d` | `extra_env` | `OPEN-ITEMS-NODE3.md` G, `OPEN-ITEMS-NODE5.md` G |
| D. cached_tokens and MLA/MHA per request | Attention path per forward with the prefix length and bootstrap room, joined to the client's `cached_tokens` | base | `patch_dispatch_logging`, `patch_room_logging`, `page_size_tokens` 1 or 64, `disable_prefix_cache` | `OPEN-ITEMS-NODE4.md`, `OPEN-ITEMS-NODE6.md` D |
| F. Health-check split and the remainder chain | Receive, admission and batch records; control: health check without generation | `oi9` (unmodified behaviour, records only), `oi7e` | `extra_env` | `OPEN-ITEMS-NODE7.md`, `OPEN-ITEMS-NODE9.md` |
| F. Triton extend-attention early return | Remainder-chain batch time with and without the early return | `oi7f` | | `OPEN-ITEMS-NODE7.md` Q2 |
| E. Decode copy thread on the wrong GPU (shm path) | Each new thread's CUDA device and each copy's context vs destination; control `tdev_fix` | base + `shims/tdev.c` | `tdev_shim`, `tdev_fix`, `extra_env.FI_EFA_ENABLE_SHM_TRANSFER=1` | `OPEN-ITEMS-NODE6.md` E |
| E. Two-leg copy path | Nsight Systems on decode with the mismatch vs fixed; injected two-leg copies with `tdev_twoleg.c` | `nsys` | `nsys_launch`, `engine_privileged`, `tdev_src` | `OPEN-ITEMS-NODE8.md` |
| H. Mid-generation stalls on the unified stack | Client-side gaps over 100 ms joined to same-engine prefill batches in the engine log | base | | `OPEN-ITEMS-NODE4.md` H |

The ledgers named in the last column are the investigation's records and come with the evidence bundle, not with this branch.

### SAR fallback (shm path)

```bash
bash -euo pipefail <<'X'
for v in 1.1.0 1.4.0; do
  bash measurement/tools/cell.sh "sar-${v//./}" "{\"pin_nixl_version\": \"$v\", \"page_size_tokens\": 1, \"extra_env\": {\"FI_LOG_LEVEL\": \"warn\"}}" 1 "345" "1" 20
  grep -c "cuMemGetAddressRange" "results/measure/sar-${v//./}/engine.log" || true
  grep -c "fallback to using SAR" "results/measure/sar-${v//./}/engine.log" || true
done
X
```

Expected: with 1.1.0, thousands of `cuMemGetAddressRange` lines (one per failed export on tensor-parallel rank 1) and a few `fallback to using SAR` lines, because libfabric prints that warning once per process; with 1.4.0, zero of both.

### EFA byte proof

`PRE_HOOK` starts a host-network probe pod on the same node once the stack is ready and saves the EFA counters; `POST_HOOK` saves them again after the ramps, before cleanup.

```bash
bash -euo pipefail <<'X'
cat > /tmp/efa-pre.sh <<'H'
sed -e "s|\${MEASURE_NS}|$NS|" -e "s|\${MEASURE_NODE}|$MEASURE_NODE|" -e "s|\${LAB_IMAGE}|$BASE_IMAGE|" measurement/manifests/efa-probe.yaml | kubectl apply -f -
kubectl -n "$NS" wait --for=condition=Ready pod/nixl14-efa-hostnet-probe --timeout=600s
kubectl -n "$NS" exec nixl14-efa-hostnet-probe -- sh -c 'grep . /sys/class/infiniband/*/ports/1/hw_counters/rdma_write_bytes' > "$OUT/efa-before.txt"
H
cat > /tmp/efa-post.sh <<'H'
kubectl -n "$NS" exec nixl14-efa-hostnet-probe -- sh -c 'grep . /sys/class/infiniband/*/ports/1/hw_counters/rdma_write_bytes' > "$OUT/efa-after.txt"
H
RAMP_EXTRA=--skip-warmup PRE_HOOK=/tmp/efa-pre.sh POST_HOOK=/tmp/efa-post.sh bash measurement/tools/cell.sh efa "{\"pin_nixl_version\": \"1.4.0\", \"page_size_tokens\": 64, \"extra_env\": {\"FI_EFA_ENABLE_SHM_TRANSFER\": \"0\"}}" 1 "345" "1" 15
paste results/measure/efa/efa-before.txt results/measure/efa/efa-after.txt
X
```

Expected: the `rdma_write_bytes` delta of the pod's device matches the KV volume of the measured requests (31,104 bytes per token per tensor-parallel rank for this model) within a fraction of a percent. `RAMP_EXTRA=--skip-warmup` keeps warm-up traffic out of the delta.

### A. One EFA device per rank

```bash
bash -euo pipefail <<'X'
bash measurement/tools/build.sh oi5d
bash measurement/tools/cell.sh a-shared "{\"image\": \"$LAB_REGISTRY/aim345:measure-oi5d\", \"pin_nixl_version\": \"1.4.0\", \"page_size_tokens\": 64, \"extra_env\": {\"FI_EFA_ENABLE_SHM_TRANSFER\": \"0\"}}" 1 "345" "1" 20
bash measurement/tools/cell.sh a-split "{\"image\": \"$LAB_REGISTRY/aim345:measure-oi5d\", \"pin_nixl_version\": \"1.4.0\", \"page_size_tokens\": 64, \"efa_per_worker\": 2, \"extra_env\": {\"FI_EFA_ENABLE_SHM_TRANSFER\": \"0\", \"AIM345_EFA_IFACES\": \"$EFA_IFACES\"}}" 1 "345" "1" 20
X
```

Export `EFA_IFACES` before the block as the two device names the engine container sees (`ls /sys/class/infiniband`), comma-separated, rank 0 first. NIXL alone: run `tools/nixl_bench.py --help` inside the engine container for the stagger and per-pair device options. Expected: the later rank waits about twice as long with one shared device, and about the same as the first rank with one device each; with NIXL alone, a GPU on the CPU socket opposite its device adds about 3 ms.

### B, C and G. Transfer-stage timing

```bash
bash -euo pipefail <<'X'
bash measurement/tools/build.sh oi3b
POST_HOOK="$PWD/measurement/tools/ring_fetch_hook.sh" bash measurement/tools/cell.sh g-base "{\"image\": \"$LAB_REGISTRY/aim345:measure-oi3b\", \"page_size_tokens\": 1}" 1 "345 1001 1002" "1" 20
POST_HOOK="$PWD/measurement/tools/ring_fetch_hook.sh" bash measurement/tools/cell.sh g-poll "{\"image\": \"$LAB_REGISTRY/aim345:measure-oi3b\", \"page_size_tokens\": 1, \"extra_env\": {\"AIM345_EXTRA_POLL\": \"1\"}}" 1 "345 1001 1002" "1" 20
POST_HOOK="$PWD/measurement/tools/ring_fetch_hook.sh" bash measurement/tools/cell.sh b-tiny "{\"image\": \"$LAB_REGISTRY/aim345:measure-oi3b\", \"page_size_tokens\": 1, \"extra_env\": {\"AIM345_SUPPRESS_TINY_CHUNK\": \"1\"}}" 1 "345 1001 1002" "1" 20
X
```

`results/measure/*/ring-dumps.jsonl` holds the per-request transfer events and loop boundaries. Expected: B, the pre-timer delay appears only on split requests and goes away with tiny chunks suppressed; G, completion is noticed at the end of the loop iteration after the transfer finished, and the extra poll removes the result-processing part.

### Transfer-stage breakdown against SGLang's own timer, and descriptor counts

`tools/stage_cell.sh` runs one disaggregated cell and brackets every single-rate ramp with `stages.py` snapshots, so the per-request `kvtimer` records of the `pct` image and SGLang's own `prefill_transfer_kv_cache` counter cover the same window. The five cells compare the workshop setting (shared-memory path on), the shared-memory path off with the cache on and off, the same without the patch (to see the instrumentation's own cost), and the proposed setting.

```bash
bash -euo pipefail <<'X'
bash measurement/tools/build.sh pct
P="$LAB_REGISTRY/aim345:measure-pct"
bash measurement/tools/stage_cell.sh s-ws "{\"image\": \"$P\", \"page_size_tokens\": 1}" "345 1001 1002" "0.25 1 2" 20
bash measurement/tools/stage_cell.sh s-s1 "{\"image\": \"$P\", \"page_size_tokens\": 1, \"extra_env\": {\"FI_EFA_ENABLE_SHM_TRANSFER\": \"0\"}}" "345 1001 1002" "0.25 1 2" 20
bash measurement/tools/stage_cell.sh s-s0 "{\"image\": \"$P\", \"page_size_tokens\": 1, \"disable_prefix_cache\": true, \"extra_env\": {\"FI_EFA_ENABLE_SHM_TRANSFER\": \"0\"}}" "345 1001 1002" "0.25 1 2" 20
bash measurement/tools/stage_cell.sh s-s1b "{\"page_size_tokens\": 1, \"extra_env\": {\"FI_EFA_ENABLE_SHM_TRANSFER\": \"0\"}}" "345 1001 1002" "0.25 1 2" 20
bash measurement/tools/stage_cell.sh s-p "{\"image\": \"$P\", \"pin_nixl_version\": \"1.4.0\", \"page_size_tokens\": 64, \"extra_env\": {\"FI_EFA_ENABLE_SHM_TRANSFER\": \"0\"}}" "345 1001 1002" "0.25 1 2" 20
for c in s-ws s-s1 s-s0 s-p; do python3 measurement/tools/stage_breakdown.py "results/measure/$c"; done
X
```

`stage_breakdown.py` prints, per window, the mean of each interval (dequeue, descriptor build, posting, completion wait, completion to timer stop, total), the sum of the per-request totals beside SGLang's own sum for the same window, and the rank correlation of each request's descriptor count with its total and its completion wait.

### D. MLA/MHA per request

```bash
bash -euo pipefail <<'X'
for p in 1 64; do for cache in false true; do
  bash measurement/tools/cell.sh "d-p$p-nocache-$cache" "{\"page_size_tokens\": $p, \"disable_prefix_cache\": $cache, \"patch_dispatch_logging\": true, \"patch_room_logging\": true}" 1 "345" "1" 20
done; done
X
```

The engine log carries one line per forward with the batch's prefix lengths, the chosen path and each request's bootstrap room. Expected: on each request's first prefill forward, every request with a cached prefix of 1 or more takes MLA and every request with 0 takes MHA; page 64 leaves the cached prefix at 0 for this traffic, whose shared part is 4 to 7 tokens. A request split by a health check has a second forward over its own prefix, which takes MLA. `python3 measurement/tools/join_room.py results/measure/<cell>` prints the per-request join.

### F. Health-check split

```bash
bash -euo pipefail <<'X'
bash measurement/tools/build.sh oi9
POST_HOOK="$PWD/measurement/tools/ring_fetch_hook.sh" bash measurement/tools/cell.sh f-natural "{\"image\": \"$LAB_REGISTRY/aim345:measure-oi9\", \"pin_nixl_version\": \"1.4.0\", \"page_size_tokens\": 64, \"extra_env\": {\"FI_EFA_ENABLE_SHM_TRANSFER\": \"0\"}}" 4 "345 1001 1002" "0.25 0.5 1 2 4" 20
bash measurement/tools/build.sh oi7f
POST_HOOK="$PWD/measurement/tools/ring_fetch_hook.sh" bash measurement/tools/cell.sh f-kernel "{\"image\": \"$LAB_REGISTRY/aim345:measure-oi7f\", \"pin_nixl_version\": \"1.4.0\", \"page_size_tokens\": 64, \"extra_env\": {\"FI_EFA_ENABLE_SHM_TRANSFER\": \"0\"}}" 2 "345" "2" 20
X
```

`ring-dumps.jsonl` records each receive (with the health-admission inputs), batch formation and the bootstrap queue. Expected: a split appears as a batch with the 1-token health request followed by an 8,128-token prompt chunk, the next batch carrying the 64-token remainder takes MLA, and with the early-return kernel that batch is about half as long.

### E. Decode copy thread on the wrong GPU (shm path)

```bash
bash -euo pipefail <<'X'
bash measurement/tools/cell.sh e-mismatch "{\"pin_nixl_version\": \"1.4.0\", \"page_size_tokens\": 64, \"tdev_shim\": true, \"extra_env\": {\"FI_EFA_ENABLE_SHM_TRANSFER\": \"1\"}}" 1 "345" "1" 20
bash measurement/tools/cell.sh e-fixed "{\"pin_nixl_version\": \"1.4.0\", \"page_size_tokens\": 64, \"tdev_shim\": true, \"tdev_fix\": true, \"extra_env\": {\"FI_EFA_ENABLE_SHM_TRANSFER\": \"1\"}}" 1 "345" "1" 20
X
```

The engine log carries one line per new thread with its current CUDA device. Expected: on decode TP1 the libfabric progress thread starts on device 0 with the mismatch and on its own device with `tdev_fix`, and decode TPOT drops with the fix. For the two-leg path, `build nsys` and add `"nsys_launch": "--trace=cuda --cuda-graph-trace=node", "engine_privileged": true` with `PRE_HOOK="$PWD/measurement/tools/nsys_pre.sh"`; the summaries come back as `results/measure/*/nsys-run.txt` and the trace stays in the container.

### H. Unified-stack stalls

```bash
bash -euo pipefail <<'X'
python3 measurement/tools/make_config.py "$LAB_CONFIG" /tmp/h.json "{\"namespace\": \"${MEASURE_NS_PREFIX}h\", \"placement\": \"packed\", \"nodes\": [\"$MEASURE_NODE\"], \"page_size_tokens\": 64}"
trap 'python3 measurement/tools/single_deploy.py cleanup --mode unified --config /tmp/h.json' EXIT
bash measurement/tools/check_exclusive.sh
python3 measurement/tools/single_deploy.py deploy --mode unified --config /tmp/h.json
python3 measurement/tools/single_deploy.py wait --mode unified --config /tmp/h.json
python3 4.ramp.py --config /tmp/h.json --traffic traffic/b.json --output results/measure/h --architecture unified --gpus 4 --instance-type "$LAB_INSTANCE_TYPE" --region "$LAB_REGION" --evidence-scope mechanism-validation --shards 2 --rates 1 --duration-s 20 --seed 345
kubectl -n "${MEASURE_NS_PREFIX}h" logs deploy/engines --tail=-1 > results/measure/h/engine.log
X
```

Expected: each client-side gap over 100 ms in `results/measure/h/*-measured.json` lines up, within about 30 ms, with a prefill batch of another request on the same engine in `engine.log`.

## Cleanup

Every cell removes its namespace on exit, including on failure. Delete the images pushed to `LAB_REGISTRY` with your registry's own command when done.
