#!/usr/bin/env python3
"""Control CLI for the in-cluster runner (DESIGN.md sections 4-7, 12).

Runs where the participant is. Never opens an HTTP connection to the router -- it only talks to
the cluster through short `kubectl exec` calls against each `runner-<k>` shard pod, and is the
sole writer of every lifecycle file: `manifest.json`, `run_status.json`, `state.json`, and the
merged `<rate>-<phase>.json` files `rescore.py`/`5.collect.py` already read today.

Storage deviation from DESIGN.md section 8 (recorded in README.md and DESIGN-RATIONALE.md): this
lab account has no EFS filesystem, so there is no `ReadWriteMany` volume shared by every shard and
by labctl. Each shard instead gets its own local volume (DESIGN.md section 8's own named EBS
fallback), and labctl reads or writes a shard's files with one `kubectl exec` per shard instead of
one shared directory listing. A second consequence, not named explicitly by section 8 for the EBS
fallback: `manifest.json`/`run_status.json`/`state.json` live on labctl's own local disk, not in
the cluster, so --resume only works from the machine that started the run. This is a known, scoped
limitation of the fallback, not a silent gap.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
import uuid
import atomic
from benchmark import stops_ramp, summarize
from deployment import kubectl, owned_namespace, read_config

POLL_S = 2.0
RAMP_POLL_S = 5.0  # whole-ramp mode: this gates nothing on the shard side, only how promptly labctl notices and merges a finished segment
LEASE_TTL_S = 60
LEASE_RENEW_S = 20
CANCEL_ACK_TIMEOUT_S = 30
START_SKEW_TOLERANCE_MS = 500
TERMINAL_PHASE_STATES = ("merged", "stopped_on_failure_fraction")
MANIFEST_FIELDS = ("architecture", "namespace", "shard_count", "client_path", "traffic_sha256", "rates",
                   "duration_s", "timeout_s", "ttft_slo_ms", "tpot_slo_ms", "attainment_fraction",
                   "context_length_tokens", "max_client_lag_ms", "stop_failure_fraction", "skip_warmup",
                   "seed", "max_tasks", "engine_deployment_generation", "tpot_metric", "gpus",
                   "instance_type", "region", "evidence_scope")


def harness_dir(output, run_id):
    return Path(output) / ".labharness" / run_id


def new_run_id():
    return f"run-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{uuid.uuid4().hex[:6]}"


def new_segment_id():
    return f"seg-{uuid.uuid4().hex[:6]}"


def derive_seed(seed, shard_index):
    return int(hashlib.sha256(f"{seed}:{shard_index}".encode()).hexdigest()[:8], 16)


def state_key(rate, phase):
    return f"{rate:g}:{phase}"


def plan_steps(rates, skip_warmup):
    steps = []
    for rate in rates:
        if not skip_warmup:
            steps.append((rate, "warmup"))
        steps.append((rate, "measured"))
    return steps


def phase_status(state, rate, phase):
    entry = state.get(state_key(rate, phase))
    return entry["status"] if entry else None


def resume_cursor(state, steps):
    for rate, phase in steps:
        status = phase_status(state, rate, phase)
        if status == "stopped_on_failure_fraction":
            return None  # the ramp stopped deliberately; rates after this one were never meant to run
        if status not in TERMINAL_PHASE_STATES:
            return rate, phase
    return None


# -- shard transport: one kubectl exec per shard, no shared mount (see module docstring) --------

def shard_pod(k):
    return f"runner-{k}"


SHARD_EXEC_RETRIES = 5


def shard_exec(c, k, *args, **kwargs):
    # -i only when actually piping input (shard_write): kubectl exec does not forward stdin
    # without it, so input= would silently write an empty file otherwise. But -i also attaches a
    # stdin stream the remote process never reads for a plain "cat path" (shard_read passes no
    # input=), and on this cluster leaving that stream attached and open (inherited from this
    # process's own stdin, which never closes) correlated with the connection-reset below -- -i is
    # therefore passed only when there is something to send, not unconditionally.
    cmd = ["kubectl", "--context", c["context"], "-n", c["namespace"], "exec"]
    if kwargs.get("input") is not None:
        cmd.append("-i")
    cmd += [shard_pod(k), "-c", "runner", "--", *args]
    # Observed on a real cluster (not in DESIGN.md section 13's list): kubectl exec's multiplexed
    # stream intermittently resets mid-transfer on a result file of a few tens of KB, well under
    # any size limit -- a client/API-server-proxy network issue, not an application error. Both
    # shard_write (idempotent: same tmp-then-rename) and shard_read (read-only) are safe to retry.
    for attempt in range(SHARD_EXEC_RETRIES):
        r = subprocess.run(cmd, text=True, capture_output=True, **kwargs)
        if r.returncode == 0:
            return r
        if "No such file or directory" in r.stderr:
            return r  # a normal poll miss (file not written yet), not the connection resetting; the caller's own poll loop already retries this on its own cadence
        print(f"shard {k}: exec attempt {attempt + 1}/{SHARD_EXEC_RETRIES} failed "
              f"({r.stderr.strip().splitlines()[-1] if r.stderr.strip() else 'no stderr'})", flush=True)
        if attempt + 1 < SHARD_EXEC_RETRIES:
            time.sleep(min(5.0, 0.5 * 2 ** attempt))
    return r


def shard_write(c, k, remote_path, content):
    script = f'd=$(dirname {shlex.quote(remote_path)}); mkdir -p "$d"; t="{remote_path}.tmp.$$"; cat > "$t" && mv "$t" {shlex.quote(remote_path)}'
    r = shard_exec(c, k, "sh", "-c", script, input=content)
    if r.returncode != 0:
        raise RuntimeError(f"shard {k}: write {remote_path} failed: {r.stderr.strip()}")


def shard_read(c, k, remote_path):
    r = shard_exec(c, k, "cat", remote_path)
    return r.stdout if r.returncode == 0 else None


def clock_offset_ms(c, k):
    """A one-shot, NTP-style estimate of shard k's node clock against labctl's own: remote time
    minus the midpoint of the local timestamps bracketing the exec round-trip. EKS nodes are
    assumed to run Amazon Time Sync (chrony), per DESIGN.md section 13's own standard this is
    verified, not assumed to be true -- not a substitute for a real NTP client, but enough to
    record whether node clock skew is a plausible contributor to a measured start skew, or can be
    ruled out. Returns None if the exec itself fails (reported, not hidden)."""
    before = time.time()
    r = shard_exec(c, k, "date", "+%s.%N")
    after = time.time()
    if r.returncode != 0:
        return None
    try:
        remote = float(r.stdout.strip())
    except ValueError:
        return None
    return (remote - (before + after) / 2) * 1000


def record_clock_offsets(c, hdir, shard_count):
    offsets = {k: clock_offset_ms(c, k) for k in range(shard_count)}
    if hdir is not None:
        atomic.atomic_write_json(hdir / "clock_offset_ms.json", offsets)
    for k, ms in offsets.items():
        print(f"shard {k}: node clock offset from labctl {('%.1fms' % ms) if ms is not None else 'unmeasured (exec failed)'}", flush=True)
    return offsets


def shard_ready_count(c, shard_count):
    sts = json.loads(kubectl(c, "get", "statefulset", "runner", "-o", "json", capture_output=True).stdout)
    ready = sts["status"].get("readyReplicas", 0)
    if ready < shard_count:
        raise SystemExit(f"runner StatefulSet has {ready} ready replica(s), need {shard_count}; "
                         "run creation refuses outright rather than silently measuring at reduced shard count")


def engine_generations(c):
    items = json.loads(kubectl(c, "get", "deploy", "-l", "component=engine", "-o", "json", capture_output=True).stdout)["items"]
    return {x["metadata"]["name"]: x["metadata"]["generation"] for x in sorted(items, key=lambda x: x["metadata"]["name"])}


def router_url(c):
    return f"http://router.{c['namespace']}.svc.cluster.local:8000"


# -- lease and manifest ---------------------------------------------------------------------------

def read_status(hdir):
    return atomic.read_json(hdir / "run_status.json")


def build_manifest(a, c, traffic_sha256, shard_count, generations):
    return {"architecture": a.architecture, "namespace": c["namespace"], "shard_count": shard_count,
            "client_path": "in-cluster-v1-local-lease", "traffic_sha256": traffic_sha256, "rates": list(a.rates),
            "duration_s": a.duration_s, "timeout_s": a.timeout_s, "ttft_slo_ms": a.ttft_slo_ms, "tpot_slo_ms": a.tpot_slo_ms,
            "attainment_fraction": a.attainment_fraction, "context_length_tokens": a.context_length_tokens,
            "max_client_lag_ms": a.max_client_lag_ms, "stop_failure_fraction": a.stop_failure_fraction,
            "skip_warmup": a.skip_warmup, "seed": a.seed, "max_tasks": a.max_tasks,
            "engine_deployment_generation": generations, "tpot_metric": a.tpot_metric, "gpus": a.gpus,
            "instance_type": a.instance_type, "region": a.region, "evidence_scope": a.evidence_scope}


def compare_manifest(run_id, stored, built):
    for key in MANIFEST_FIELDS:
        if stored.get(key) != built.get(key):
            raise SystemExit(f"{run_id}: stored {key} differs (stored {stored.get(key)!r}, given {built.get(key)!r}); "
                             "resume must repeat the same plan")


def acquire_lease(hdir, run_id, segment_id):
    status = read_status(hdir)
    now = time.time()
    if status:
        if status["status"] == "CREATING":
            raise SystemExit(f"{run_id}: creation did not finish; use a new run id")
        if status["status"] == "COMPLETE":
            raise SystemExit(f"{run_id}: status is COMPLETE; already measured, use a fresh run id or `labctl fetch`")
        if status["status"] == "FAILED":
            raise SystemExit(f"{run_id}: status is FAILED; a FAILED run cannot be resumed; start a fresh run")
        if status["status"] == "RUNNING" and status.get("lease_expires_at", 0) > now:
            raise SystemExit(f"run {run_id} is active (owner lease held); wait or --resume after it ends")
    new_status = {"status": "RUNNING", "owner_segment_id": segment_id, "lease_expires_at": now + LEASE_TTL_S}
    atomic.atomic_write_json(hdir / "run_status.json", new_status)
    return new_status


def renew_lease(hdir, segment_id):
    atomic.atomic_write_json(hdir / "run_status.json", {"status": "RUNNING", "owner_segment_id": segment_id,
                                                        "lease_expires_at": time.time() + LEASE_TTL_S})


class Lease:
    """Renews itself every LEASE_RENEW_S while the dispatch loop runs; this is the one source of
    truth labctl's own loop checks before issuing anything new (DESIGN.md section 4.1)."""

    def __init__(self, hdir, segment_id):
        self.hdir, self.segment_id, self.last = hdir, segment_id, time.time()

    def tick(self):
        if time.time() - self.last > LEASE_RENEW_S:
            renew_lease(self.hdir, self.segment_id)
            self.last = time.time()

    def release(self, final_status):
        atomic.atomic_write_json(self.hdir / "run_status.json",
                                 {"status": final_status, "owner_segment_id": None, "lease_expires_at": 0})


# -- dispatch, poll, merge --------------------------------------------------------------------

def write_state(hdir, state):
    atomic.atomic_write_json(hdir / "state.json", state)


START_BARRIER_BUFFER_S = 5.0


def dispatch(c, manifest, segment_id, rate, phase, traffic):
    """Two writes, not one. The job content (traffic payload and all) goes out first and can take
    an arbitrary, variable amount of time per shard (an exec round-trip, occasionally a retry
    storm -- see DESIGN-RATIONALE.md). Only once every shard provably has that content does labctl
    mint the shared start_at and send it in a second, tiny, separate write. Minting start_at
    *before* the first write (the previous design) baked today's dispatch latency into tomorrow's
    barrier: by the time a slow shard's job landed, start_at could already be in the past, so that
    shard started immediately at whatever moment its own exec call happened to land -- the exact
    thing the barrier exists to prevent. Splitting the writes removes that dependency: the second
    write's own latency is the only thing left to budget for, and it carries a few bytes."""
    shard_count = manifest["shard_count"]
    job_name = f"{segment_id}-{rate:g}-{phase}"

    def write_job(k):
        job = {"segment_id": segment_id, "rate": rate / shard_count, "phase": phase, "shard_count": shard_count,
               "gpus": manifest["gpus"], "router_url": router_url(c), "architecture": manifest["architecture"],
               "instance_type": manifest["instance_type"], "region": manifest["region"],
               "evidence_scope": manifest["evidence_scope"], "traffic": traffic,
               "args": {"seed": derive_seed(manifest["seed"], k), "duration_s": manifest["duration_s"], "timeout_s": manifest["timeout_s"],
                        "ttft_slo_ms": manifest["ttft_slo_ms"], "tpot_slo_ms": manifest["tpot_slo_ms"],
                        "attainment_fraction": manifest["attainment_fraction"], "context_length_tokens": manifest["context_length_tokens"],
                        "max_tasks": manifest["max_tasks"] // shard_count, "max_client_lag_ms": manifest["max_client_lag_ms"],
                        "stop_failure_fraction": manifest["stop_failure_fraction"], "skip_warmup": manifest["skip_warmup"],
                        "tpot_metric": manifest["tpot_metric"]}}
        shard_write(c, k, f"/local/jobs/{job_name}.job.json", json.dumps(job))

    with ThreadPoolExecutor(max_workers=shard_count) as pool:
        for future in as_completed([pool.submit(write_job, k) for k in range(shard_count)]):
            future.result()  # surface a shard_write failure immediately rather than swallowing it

    start_at = time.time() + max(START_BARRIER_BUFFER_S, 1.0 * shard_count)

    def write_start(k):
        shard_write(c, k, f"/local/jobs/{job_name}.start.json", json.dumps({"start_at": start_at}))

    with ThreadPoolExecutor(max_workers=shard_count) as pool:
        for future in as_completed([pool.submit(write_start, k) for k in range(shard_count)]):
            future.result()
    print(f"dispatched {job_name} to {shard_count} shard(s), start barrier in {start_at - time.time():.1f}s", flush=True)
    return job_name


PEER_HTTP_PORT = 7777


def peer_hostname(c, k):
    """A runner-<k> pod behind the headless `runner` Service (deployment.py's runner_resources)
    already gets this exact per-pod DNS name from the StatefulSet, no new Kubernetes object
    needed: only a container port and the shard's own tiny status server (runner_agent.py)."""
    return f"runner-{k}.runner.{c['namespace']}.svc.cluster.local"


def exec_rtt_s(c, k):
    """One-shot measured kubectl exec round-trip latency to shard k. Sizes the one remaining
    labctl-mediated barrier (segment 0 of a whole-ramp dispatch) on this run's own exec latency
    instead of a constant inherited from a different cluster or day (DESIGN-RATIONALE.md). None
    if the exec itself fails -- reported, not hidden, same convention as clock_offset_ms."""
    before = time.time()
    r = shard_exec(c, k, "true")
    after = time.time()
    return (after - before) if r.returncode == 0 else None


def measured_start_barrier_s(c, shard_count):
    rtts = [x for x in (exec_rtt_s(c, k) for k in range(shard_count)) if x is not None]
    buffer_s = max(3.0 * max(rtts), 1.0 * shard_count, 1.0) if rtts else max(START_BARRIER_BUFFER_S, 1.0 * shard_count)
    print(f"measured kubectl exec round-trip up to {max(rtts) * 1000:.0f}ms; using a {buffer_s:.1f}s start barrier" if rtts
          else f"could not measure exec round-trip; using a {buffer_s:.1f}s start barrier", flush=True)
    return buffer_s


def dispatch_ramp_whole(c, manifest, segment_id, remaining_steps, traffic, barrier_s):
    """The whole-ramp counterpart to dispatch(): one job write per shard covering every remaining
    (rate, phase) step, not one write per step. Only the first step's start_at is minted here, the
    same two-phase, barrier-after-content way dispatch() always has; every later step's start_at is
    agreed between shards over their own in-cluster peer protocol (runner_agent.execute_ramp), so
    this call -- not a per-step one -- is the last kubectl exec labctl makes until the whole ramp
    is done or cancelled (DESIGN-RATIONALE.md, "whole-ramp dispatch")."""
    shard_count = manifest["shard_count"]
    peers_all = [{"shard_index": k, "host": peer_hostname(c, k), "port": PEER_HTTP_PORT} for k in range(shard_count)]
    steps_payload = [{"rate": rate, "phase": phase} for rate, phase in remaining_steps]
    args_template = {"duration_s": manifest["duration_s"], "timeout_s": manifest["timeout_s"],
                     "ttft_slo_ms": manifest["ttft_slo_ms"], "tpot_slo_ms": manifest["tpot_slo_ms"],
                     "attainment_fraction": manifest["attainment_fraction"], "context_length_tokens": manifest["context_length_tokens"],
                     "max_tasks": manifest["max_tasks"] // shard_count, "max_client_lag_ms": manifest["max_client_lag_ms"],
                     "stop_failure_fraction": manifest["stop_failure_fraction"], "skip_warmup": manifest["skip_warmup"],
                     "tpot_metric": manifest["tpot_metric"]}

    def write_job(k):
        job = {"segment_id": segment_id, "shard_count": shard_count, "shard_index": k,
               "peers": [p for p in peers_all if p["shard_index"] != k], "http_port": PEER_HTTP_PORT,
               "steps": steps_payload, "gpus": manifest["gpus"], "router_url": router_url(c),
               "architecture": manifest["architecture"], "instance_type": manifest["instance_type"],
               "region": manifest["region"], "evidence_scope": manifest["evidence_scope"], "traffic": traffic,
               "base_seed": manifest["seed"], "args_template": args_template}
        shard_write(c, k, f"/local/jobs/{segment_id}.ramp.json", json.dumps(job))

    with ThreadPoolExecutor(max_workers=shard_count) as pool:
        for future in as_completed([pool.submit(write_job, k) for k in range(shard_count)]):
            future.result()

    start_at = time.time() + barrier_s

    def write_start(k):
        shard_write(c, k, f"/local/jobs/{segment_id}.ramp.start.json", json.dumps({"start_at": start_at}))

    with ThreadPoolExecutor(max_workers=shard_count) as pool:
        for future in as_completed([pool.submit(write_start, k) for k in range(shard_count)]):
            future.result()
    print(f"dispatched whole ramp {segment_id} ({len(remaining_steps)} segment(s)) to {shard_count} shard(s), "
          f"start barrier in {start_at - time.time():.1f}s", flush=True)


def cancel_whole_ramp(c, shard_count, segment_id):
    for k in range(shard_count):
        shard_write(c, k, f"/local/jobs/{segment_id}.ramp.cancel", "")


def cancel_job(c, shard_count, job_name, acked):
    for k in range(shard_count):
        if k not in acked:
            shard_write(c, k, f"/local/jobs/{job_name}.cancel", "")


def quiesce_shards(c, manifest, state):
    """Resuming a RUNNING run whose lease merely expired (a crash, not a clean Ctrl+C) may still
    have a shard mid-measurement on the previous segment's job. There is no shared run_status a
    shard can poll to notice the ownership change on its own (this account's EBS-per-shard
    fallback has no such shared file, see module docstring), so labctl actively cancels the old
    job by its old name before dispatching a new segment -- a bounded, best-effort drain, not a
    guarantee; the new segment's own distinct job_name means a straggler's late write never lands
    on a file the new segment reads."""
    outstanding = [v for v in state.values() if v.get("status") in ("dispatched", "shards_complete")]
    if not outstanding:
        return
    segment_ids = set()
    for entry in outstanding:
        job_name = entry["job_name"]
        for k in range(manifest["shard_count"]):
            try:
                shard_write(c, k, f"/local/jobs/{job_name}.cancel", "")
            except RuntimeError:
                pass
        # job_name is "<segment_id>-<rate:g>-<phase>"; segment_id itself is "seg-<hex>" (one
        # hyphen) and rate/phase never contain one, so this always recovers it correctly.
        segment_ids.add(job_name.rsplit("-", 2)[0])
    for segment_id in segment_ids:
        # A straggler left by a crash may be mid-execute_ramp, not mid-execute (whole-ramp
        # dispatch, DESIGN-RATIONALE.md): its own per-job .cancel file above means nothing to that
        # loop, only its ramp-level .cancel file does.
        try:
            cancel_whole_ramp(c, manifest["shard_count"], segment_id)
        except RuntimeError:
            pass
    print(f"quiescing {len(outstanding)} job(s) left by a previous, now-superseded segment...", flush=True)
    time.sleep(min(LEASE_TTL_S, 10))


def coarse_wait(duration_s, cancel, margin_s=3.0):
    """A phase cannot possibly finish before its offered window elapses, so there is nothing to
    learn by kubectl exec-ing into every shard every POLL_S during that window -- found by
    comparing a real sweep's measured wall-clock time against window+dispatch+drain+merge, which
    left a multi-minute gap this was the dominant cause of (DESIGN-RATIONALE.md). Sleeps in short,
    cancel-responsive ticks with no exec calls at all, leaving `margin_s` of the window for
    poll_acks's normal cadence to still catch an unusually fast shard."""
    deadline = time.time() + max(0.0, duration_s - margin_s)
    while time.time() < deadline:
        if cancel():
            return
        time.sleep(min(1.0, deadline - time.time()))


def poll_acks(c, shard_count, job_name, lease, cancel, overall_deadline):
    """Polls every shard's ack file; cancel() reports whether a Ctrl+C has been requested.
    overall_deadline bounds the wait even with no Ctrl+C, so a dead shard that never acks cannot
    block labctl forever. Returns (acked, interrupted)."""
    acked, cancel_sent = {}, None
    while len(acked) < shard_count:
        if cancel() and cancel_sent is None:
            cancel_job(c, shard_count, job_name, acked)
            cancel_sent = time.time()
            print("stopping safely, this can take up to 30s -- press Ctrl+C again to abandon without waiting", flush=True)
        if cancel_sent is not None and time.time() - cancel_sent > CANCEL_ACK_TIMEOUT_S:
            missing = sorted(set(range(shard_count)) - set(acked))
            print(f"shard(s) {missing} did not acknowledge within {CANCEL_ACK_TIMEOUT_S}s; marking the run interrupted anyway", flush=True)
            return acked, True
        if cancel_sent is None and time.time() > overall_deadline:
            missing = sorted(set(range(shard_count)) - set(acked))
            print(f"shard(s) {missing} did not finish within the measurement window's deadline; marking the run interrupted", flush=True)
            cancel_job(c, shard_count, job_name, acked)
            return acked, True
        for k in range(shard_count):
            if k in acked:
                continue
            text = shard_read(c, k, f"/local/acks/{job_name}.ack.json")
            if text:
                try:
                    acked[k] = json.loads(text)
                except ValueError:
                    continue
                print(f"shard {k}: ack {acked[k]['status']}", flush=True)
        lease.tick()
        if len(acked) < shard_count:
            time.sleep(POLL_S)
    return acked, cancel_sent is not None


def merge(c, manifest, segment_id, rate, phase, job_name, acked, shard_count, run_id="selfcheck"):
    failed = [k for k, ack in acked.items() if ack.get("status") == "failed"]
    if failed:
        raise RuntimeError(f"shard(s) {sorted(failed)} reported an unrecoverable error: " +
                           "; ".join(acked[k]["error"] for k in sorted(failed)))
    shard_results = []
    for k in range(shard_count):
        text = shard_read(c, k, f"/local/results/{job_name}.result.json")
        if text is None:
            raise RuntimeError(f"{job_name}: shard {k} acknowledged but its result file is missing; refusing to merge")
        shard_results.append(json.loads(text))
    requests = []
    for sr in sorted(shard_results, key=lambda r: r["summary"]["shard_index"]):
        for req in sr["requests"]:
            req["shard_index"] = sr["summary"]["shard_index"]
        requests += sr["requests"]
    requests.sort(key=lambda r: (r["shard_index"], r["task_id"]))
    starts = [sr["shard_actual_start_wall"] for sr in shard_results]
    skew_ms = (max(starts) - min(starts)) * 1000
    # start_skew_ms is passed into summarize() itself (not applied as a post-hoc override) so that
    # re-scoring the stored requests[] at the stored start_skew_ms/tolerance reproduces this same
    # client_valid/meets_joint_slo verdict byte for byte (rescore.check_reproduces requires this).
    summary = summarize(requests, manifest["duration_s"], manifest["gpus"], manifest["ttft_slo_ms"], manifest["tpot_slo_ms"],
                        manifest["attainment_fraction"], rate, manifest["max_client_lag_ms"], manifest["tpot_metric"],
                        start_skew_ms=skew_ms, start_skew_tolerance_ms=START_SKEW_TOLERANCE_MS)
    reason = None
    if skew_ms > START_SKEW_TOLERANCE_MS:
        reason = f"start skew {skew_ms:.0f}ms > {START_SKEW_TOLERANCE_MS:.0f}ms tolerance"
    elif not summary["client_valid"]:
        if summary["max_client_launch_lag_ms"] > manifest["max_client_lag_ms"]:
            reason = f"client launch lag {summary['max_client_launch_lag_ms']:.0f}ms > {manifest['max_client_lag_ms']:g}ms limit"
        elif summary["unreachable_calls"]:
            reason = f"{summary['unreachable_calls']} unreachable call(s); a wall there belongs to the client path, not the stack"
    if reason:
        summary["client_valid_reason"] = reason
        print(f"client path invalid: {reason}", flush=True)
    summary.update(offered_window_s=manifest["duration_s"], offered_task_count=len(requests))
    arguments = {"max_client_lag_ms": manifest["max_client_lag_ms"], "duration_s": manifest["duration_s"],
                "timeout_s": manifest["timeout_s"], "context_length_tokens": manifest["context_length_tokens"],
                "skip_warmup": manifest["skip_warmup"], "stop_failure_fraction": manifest["stop_failure_fraction"],
                "rates": manifest["rates"], "ramp_id": segment_id, "run_id": run_id, "tpot_metric": manifest["tpot_metric"],
                "client_path": manifest["client_path"], "shard_count": shard_count}
    base = shard_results[0]["metadata"]
    metadata = {"architecture": manifest["architecture"], "instance_type": manifest["instance_type"],
               "region": manifest["region"], "evidence_scope": manifest["evidence_scope"],
               "model": base["model"], "model_revision": base["model_revision"], "shape": base["shape"],
               "traffic_sha256": base["traffic_sha256"], "phase": phase, "seed": manifest["seed"],
               "run_nonce": segment_id, "arguments": arguments}
    return {"metadata": metadata, "summary": summary, "requests": requests, "shard_segment_ids": [sr["segment_id"] for sr in shard_results]}


def find_complete_matching(output, built):
    root = Path(output) / ".labharness"
    if not root.exists():
        return None
    for run_dir in sorted(root.iterdir()):
        manifest, status = atomic.read_json(run_dir / "manifest.json"), atomic.read_json(run_dir / "run_status.json")
        if manifest and status and status.get("status") == "COMPLETE" and all(manifest.get(k) == built.get(k) for k in MANIFEST_FIELDS):
            return run_dir.name
    return None


def write_dotfile(run_id, argv):
    atomic.atomic_write_json(Path.cwd() / ".labctl" / "current-run-id", {"run_id": run_id, "argv": argv})


# -- labctl ramp --------------------------------------------------------------------------------

def run_ramp(a):
    c = read_config(a.config)
    owned_namespace(c)
    traffic = json.loads(Path(a.traffic).read_text())
    traffic_sha256 = hashlib.sha256(json.dumps(traffic, sort_keys=True).encode()).hexdigest()
    shard_ready_count(c, a.shards)
    generations = engine_generations(c)
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=True)
    built = build_manifest(a, c, traffic_sha256, a.shards, generations)

    if a.resume:
        run_id = a.resume
        hdir = harness_dir(a.output, run_id)
        stored = atomic.read_json(hdir / "manifest.json")
        if stored is None:
            raise SystemExit(f"{run_id}: no such run id; use `labctl list` to see existing runs")
        compare_manifest(run_id, stored, built)
        manifest = stored
        segment_id = new_segment_id()
        acquire_lease(hdir, run_id, segment_id)  # refuses CREATING/COMPLETE/FAILED/live-RUNNING by name
        quiesce_shards(c, manifest, atomic.read_json(hdir / "state.json") or {})
    else:
        existing = find_complete_matching(output, built)
        if existing and not a.force_new_run:
            raise SystemExit(f"already measured as {existing}; use --resume {existing} or --force-new-run")
        run_id = new_run_id()
        hdir = harness_dir(a.output, run_id)
        atomic.atomic_write_json(hdir / "run_status.json", {"status": "CREATING", "owner_segment_id": None, "lease_expires_at": 0})
        atomic.atomic_write_json(hdir / "manifest.json", built)
        manifest = built
        # Not acquire_lease(): a brand-new run's own CREATING marker was just written by this same,
        # uninterrupted process, so there is nothing to fence against yet (DESIGN.md section 5's
        # CREATING refusal is for a *later* invocation finding a run that never finished creation).
        segment_id = new_segment_id()
        atomic.atomic_write_json(hdir / "run_status.json", {"status": "RUNNING", "owner_segment_id": segment_id, "lease_expires_at": time.time() + LEASE_TTL_S})

    lease = Lease(hdir, segment_id)
    write_dotfile(run_id, sys.argv)
    record_clock_offsets(c, hdir, manifest["shard_count"])

    cancel_requested = {"flag": False}

    def handle_sigint(signum, frame):
        if cancel_requested["flag"]:
            print("second Ctrl+C: abandoning without releasing the lease", flush=True)
            os._exit(1)
        cancel_requested["flag"] = True
    previous_handler = signal.signal(signal.SIGINT, handle_sigint)

    state = atomic.read_json(hdir / "state.json") or {}
    steps = plan_steps(manifest["rates"], manifest["skip_warmup"])

    def resume_url():
        return f"labctl ramp --resume {run_id} --config {a.config} --traffic {a.traffic} --output {a.output} --architecture {a.architecture}"

    try:
        cursor = resume_cursor(state, steps)
        remaining = steps[steps.index(cursor):] if cursor is not None else []
        if remaining and cancel_requested["flag"]:
            # A Ctrl+C that arrived between acquiring the lease and dispatching: stop before
            # issuing anything new (DESIGN.md section 7.2, step 2), identical to the pre-whole-ramp
            # behavior since nothing has been sent to a shard yet in this attempt.
            lease.release("INTERRUPTED")
            raise SystemExit(f"run {run_id} interrupted before starting the next rate; resume with: {resume_url()}")
        if remaining:
            # One dispatch for every remaining (rate, phase), not one per step (DESIGN-RATIONALE.md,
            # "whole-ramp dispatch"): the barrier between later steps is agreed in-cluster between
            # shards (runner_agent.execute_ramp), not by a fresh kubectl exec round trip from here.
            barrier_s = measured_start_barrier_s(c, manifest["shard_count"])
            dispatch_ramp_whole(c, manifest, segment_id, remaining, traffic, barrier_s)
            for rate, phase in remaining:
                state[state_key(rate, phase)] = {"status": "dispatched", "job_name": f"{segment_id}-{rate:g}-{phase}"}
            write_state(hdir, state)
        pending = list(remaining)
        per_segment_budget_s = manifest["duration_s"] + manifest["timeout_s"] + 60
        overall_deadline = time.time() + per_segment_budget_s * max(len(pending), 1)
        cancel_sent = None
        stopped = False
        while pending and not stopped:
            if cancel_requested["flag"] and cancel_sent is None:
                cancel_whole_ramp(c, manifest["shard_count"], segment_id)
                cancel_sent = time.time()
                print("stopping safely, this can take up to 30s -- press Ctrl+C again to abandon without waiting", flush=True)
            if cancel_sent is not None and time.time() - cancel_sent > CANCEL_ACK_TIMEOUT_S:
                missing = sorted(set(pending))
                print(f"segment(s) {missing} did not acknowledge within {CANCEL_ACK_TIMEOUT_S}s; marking the run interrupted anyway", flush=True)
                for rate, phase in pending:
                    state[state_key(rate, phase)] = {"status": "interrupted", "job_name": f"{segment_id}-{rate:g}-{phase}"}
                write_state(hdir, state)
                lease.release("INTERRUPTED")
                # Non-zero on purpose, not cosmetic (DESIGN-RATIONALE.md): an interrupted run must
                # not look like success to a caller chaining unified and disaggregated with &&.
                raise SystemExit(f"run {run_id} interrupted; resume with: {resume_url()}")
            if cancel_sent is None and time.time() > overall_deadline:
                print(f"segment(s) {sorted(set(pending))} did not finish within the ramp's overall deadline; marking the run interrupted", flush=True)
                cancel_whole_ramp(c, manifest["shard_count"], segment_id)
                for rate, phase in pending:
                    state[state_key(rate, phase)] = {"status": "interrupted", "job_name": f"{segment_id}-{rate:g}-{phase}"}
                write_state(hdir, state)
                lease.release("INTERRUPTED")
                raise SystemExit(f"run {run_id} interrupted; resume with: {resume_url()}")
            progressed = False
            for rate, phase in list(pending):
                job_name = f"{segment_id}-{rate:g}-{phase}"
                acked = {}
                for k in range(manifest["shard_count"]):
                    text = shard_read(c, k, f"/local/acks/{job_name}.ack.json")
                    if text:
                        try:
                            acked[k] = json.loads(text)
                        except ValueError:
                            pass
                if len(acked) < manifest["shard_count"]:
                    break  # a shard runs its own steps strictly in order, so a later one cannot ack before this one does
                progressed = True
                state[state_key(rate, phase)] = {"status": "shards_complete", "job_name": job_name}
                write_state(hdir, state)
                try:
                    merged = merge(c, manifest, segment_id, rate, phase, job_name, acked, manifest["shard_count"], run_id)
                except RuntimeError as exc:
                    lease.release("FAILED")
                    raise SystemExit(f"{exc}; a FAILED run cannot be resumed, start a fresh run") from None
                atomic.atomic_write(output / f"{rate:g}-{phase}.json", json.dumps(merged, indent=2))
                is_stop = stops_ramp(merged["summary"], manifest["stop_failure_fraction"])
                state[state_key(rate, phase)] = {"status": "stopped_on_failure_fraction" if is_stop else "merged", "job_name": job_name}
                write_state(hdir, state)
                print(json.dumps({"metadata": merged["metadata"], "summary": merged["summary"]}), flush=True)
                pending.remove((rate, phase))
                lease.tick()
                if is_stop:
                    # Detected, not prevented: every planned step was already on the shard before this
                    # one finished (whole-ramp dispatch cannot abort mid-flight, DESIGN-RATIONALE.md),
                    # so later rates still ran and are in --output; failures remain in their denominator.
                    print("Ramp reached the configured failure fraction; later rates already ran under "
                          "whole-ramp dispatch and are still written to --output.", flush=True)
                    stopped = True
                    break
            if pending and not progressed and not stopped:
                lease.tick()
                time.sleep(RAMP_POLL_S)
    finally:
        signal.signal(signal.SIGINT, previous_handler)
    lease.release("COMPLETE")
    print(f"run {run_id} COMPLETE", flush=True)


def cmd_list(a):
    root = Path(a.output) / ".labharness"
    if not root.exists():
        print(f"no runs recorded under {a.output}")
        return
    for run_dir in sorted(root.iterdir()):
        status = atomic.read_json(run_dir / "run_status.json") or {}
        manifest = atomic.read_json(run_dir / "manifest.json") or {}
        created = time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime(run_dir.stat().st_mtime))
        print(f"{run_dir.name:28} {manifest.get('architecture', '?'):<14} {status.get('status', '?'):<12} created {created}")


def cmd_status(a):
    hdir = harness_dir(a.output, a.run_id)
    status = atomic.read_json(hdir / "run_status.json")
    if status is None:
        raise SystemExit(f"{a.run_id}: no such run id; use `labctl list` to see existing runs")
    state = atomic.read_json(hdir / "state.json") or {}
    print(json.dumps({"run_id": a.run_id, "status": status, "state": state}, indent=2))


def cmd_fetch(a):
    """Fetch-only: for 'the cluster-side run finished but my local copy is missing or
    incomplete' (DESIGN.md section 12). In this account's local-lease fallback (module
    docstring), the merged files are already written straight to --output -- there is no
    separate in-cluster copy to pull from -- so this confirms completeness rather than
    transferring anything. --verify additionally confirms every expected file parses as JSON."""
    output = Path(a.output)
    hdir = harness_dir(output, a.run_id)
    status = atomic.read_json(hdir / "run_status.json")
    if status is None:
        raise SystemExit(f"{a.run_id}: no such run id; use `labctl list` to see existing runs")
    if status["status"] != "COMPLETE":
        raise SystemExit(f"{a.run_id}: status is {status['status']}, not COMPLETE; nothing to fetch")
    manifest = atomic.read_json(hdir / "manifest.json")
    problems = []
    for rate in manifest["rates"]:
        for phase in (["measured"] if manifest["skip_warmup"] else ["warmup", "measured"]):
            f = output / f"{rate:g}-{phase}.json"
            if not f.exists():
                problems.append(f"{f.name}: missing")
            elif a.verify:
                try:
                    json.loads(f.read_text())
                except ValueError:
                    problems.append(f"{f.name}: not readable as JSON")
    if problems:
        raise SystemExit(f"{a.run_id}: " + "; ".join(problems))
    print(f"ok: {a.run_id} COMPLETE, {len(manifest['rates'])} rate(s) present" + (" and verified" if a.verify else ""))


def cmd_selfcheck(a):
    """Dispatches one trivial single-call job to every shard and confirms the merge and fetch
    complete within a bounded timeout (DESIGN.md section 12) -- the thing `paired.py verify`'s
    router-loopback check was being asked to be but only ever proved router liveness."""
    c = read_config(a.config)
    owned_namespace(c)
    shard_ready_count(c, a.shards)
    record_clock_offsets(c, None, a.shards)
    traffic = json.loads(Path(a.traffic).read_text())
    traffic = dict(traffic, tasks=traffic["tasks"][:1])
    manifest = {"architecture": "selfcheck", "namespace": c["namespace"], "shard_count": a.shards,
               "client_path": "in-cluster-v1-local-lease", "duration_s": 2, "timeout_s": 30, "ttft_slo_ms": 60000,
               "tpot_slo_ms": 60000, "attainment_fraction": 0.0, "context_length_tokens": 32768,
               "max_client_lag_ms": 60000, "stop_failure_fraction": 1.0, "skip_warmup": True, "seed": 1,
               "max_tasks": a.shards * 50, "tpot_metric": "mean", "gpus": 1, "instance_type": c.get("instance_type"),
               "region": "selfcheck", "evidence_scope": "mechanism-validation", "rates": [1.0]}
    segment_id = new_segment_id()
    deadline = time.time() + 60
    try:
        job_name = dispatch(c, manifest, segment_id, 1.0, "measured", traffic)
    except RuntimeError as exc:
        raise SystemExit(f"FAIL: dispatch: {exc}")
    acked = {}
    while len(acked) < a.shards and time.time() < deadline:
        for k in range(a.shards):
            if k not in acked:
                text = shard_read(c, k, f"/local/acks/{job_name}.ack.json")
                if text:
                    acked[k] = json.loads(text)
        if len(acked) < a.shards:
            time.sleep(POLL_S)
    missing = sorted(set(range(a.shards)) - set(acked))
    if missing:
        raise SystemExit(f"FAIL: shard(s) {missing} unreachable")
    try:
        merge(c, manifest, segment_id, 1.0, "measured", job_name, acked, a.shards)
    except RuntimeError as exc:
        raise SystemExit(f"FAIL: merge/fetch: {exc}")
    print(f"ok: {a.shards}/{a.shards} shards, merge, fetch")


def add_ramp_arguments(r):
    """Shared with 4.ramp.py (DESIGN.md section 11's table keeps that filename as what a
    participant types; this module is the shared mechanism underneath both entry points)."""
    r.add_argument("--config", required=True)
    r.add_argument("--traffic", required=True)
    r.add_argument("--output", required=True)
    r.add_argument("--architecture", choices=["unified", "disaggregated", "disaggregated-scaled"], required=True)
    r.add_argument("--gpus", type=int, required=True)
    r.add_argument("--instance-type", required=True)
    r.add_argument("--region", required=True)
    r.add_argument("--evidence-scope", choices=["mechanism-validation", "production-calibration", "local-mock"], required=True)
    r.add_argument("--shards", type=int, default=2)
    r.add_argument("--rates", type=float, nargs="+", default=[0.25, 0.5, 1, 2, 4, 8])
    r.add_argument("--duration-s", type=float, default=30)
    r.add_argument("--timeout-s", type=float, default=310)
    r.add_argument("--ttft-slo-ms", type=float, default=2000)
    r.add_argument("--tpot-slo-ms", type=float, default=100)
    r.add_argument("--attainment-fraction", type=float, default=0.90)
    r.add_argument("--context-length-tokens", type=int, default=32768)
    r.add_argument("--max-tasks", type=int, default=2048)
    r.add_argument("--max-client-lag-ms", type=float, default=100)
    r.add_argument("--stop-failure-fraction", type=float, default=0.25)
    r.add_argument("--seed", type=int, default=345)
    r.add_argument("--skip-warmup", action="store_true")
    r.add_argument("--tpot-metric", choices=["mean", "itl-p99"], default="mean")
    r.add_argument("--resume", metavar="ID")
    r.add_argument("--force-new-run", action="store_true")


def validate_ramp_arguments(a):
    assert a.gpus > 0 and a.duration_s > 0 and a.timeout_s > 0 and all(x > 0 for x in a.rates) and len(set(a.rates)) == len(a.rates)
    assert 0 < a.attainment_fraction <= 1 and 0 <= a.stop_failure_fraction <= 1


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="action", required=True)

    r = sub.add_parser("ramp", help="Facilitator/debug entry point; participants run 4.ramp.py, a thin wrapper over this")
    add_ramp_arguments(r)

    lst = sub.add_parser("list")
    lst.add_argument("--output", required=True)

    st = sub.add_parser("status")
    st.add_argument("--output", required=True)
    st.add_argument("--run-id", required=True)

    ft = sub.add_parser("fetch")
    ft.add_argument("--output", required=True)
    ft.add_argument("--run-id", required=True)
    ft.add_argument("--verify", action="store_true")

    sc = sub.add_parser("selfcheck")
    sc.add_argument("--config", required=True)
    sc.add_argument("--traffic", required=True)
    sc.add_argument("--shards", type=int, default=2)

    a = p.parse_args()
    if a.action == "ramp":
        validate_ramp_arguments(a)
        run_ramp(a)
    elif a.action == "list":
        cmd_list(a)
    elif a.action == "status":
        cmd_status(a)
    elif a.action == "fetch":
        cmd_fetch(a)
    elif a.action == "selfcheck":
        cmd_selfcheck(a)


if __name__ == "__main__":
    main()
