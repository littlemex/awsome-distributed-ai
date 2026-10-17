#!/usr/bin/env python3
"""Long-lived shard entrypoint (DESIGN.md section 3, item 2).

Started once when the `runner-<k>` pod starts and never restarted for a new run. Polls its own
local volume (mounted at --local-root, one PVC per shard -- the EBS fallback from DESIGN.md
section 8, since this lab account has no EFS filesystem; see README.md "Where the in-cluster
runner's results live") for job files, executes each with benchmark.run_rate, and writes only
files named after its own job -- it never writes any file another process reads as shared state.

A job is claimed by its filename alone: labctl is this shard's only writer (section 4's "every
writer owns a disjoint set of filenames"), and because the control channel is this pod's own
private volume rather than one shared mount, there is no concurrent-claim race to defend against
with an exclusive-create idiom -- the one labctl process allowed to hold the lease (checked by
labctl itself, every dispatch) is the only process that can ever write a job file here.
"""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace
import atomic
from benchmark import run_rate
from traffic import tokenizer

POLL_S = 0.2  # DESIGN.md section 7.2: a shard's poll tick must be sub-second
PEER_POLL_S = 0.1  # in-cluster shard-to-shard polling tick; a different, much tighter network path than labctl's own kubectl exec


def paths(root, job_name):
    return (root / "jobs" / f"{job_name}.job.json", root / "jobs" / f"{job_name}.cancel",
            root / "acks" / f"{job_name}.ack.json", root / "results" / f"{job_name}.result.json")


def ramp_paths(root, segment_id):
    return (root / "jobs" / f"{segment_id}.ramp.json", root / "jobs" / f"{segment_id}.ramp.cancel",
            root / "acks" / f"{segment_id}.ramp.ack.json")


def claim_ramp(root, segment_id):
    """Same two-file claim idiom as try_claim, under a deliberately different filename suffix
    (.ramp.json/.ramp.start.json, not .job.json/.start.json) so the whole-ramp job and the
    single-segment job (still used by `labctl selfcheck`) never collide in the directory listing
    glob -- `*.job.json` would otherwise also match a hypothetical `<id>.ramp.job.json`."""
    job_path, _, _ = ramp_paths(root, segment_id)
    start_path = root / "jobs" / f"{segment_id}.ramp.start.json"
    if not start_path.exists():
        return None
    try:
        job = json.loads(job_path.read_text())
        job["start_at"] = json.loads(start_path.read_text())["start_at"]
    except ValueError:
        return None
    return job


def try_claim(root, job_name):
    """None if this job is not ready to execute yet, otherwise the job dict with start_at merged
    in. labctl's dispatch() writes the job's content and its shared start_at as two separate
    files (DESIGN-RATIONALE.md); a job is claimable only once both exist, so a shard never starts
    on a start_at that was computed before the slower of the two shards even had the job content."""
    job_path, _, _, _ = paths(root, job_name)
    start_path = root / "jobs" / f"{job_name}.start.json"
    if not start_path.exists():
        return None  # the job's content has landed but labctl's shared start_at has not yet; try again next tick, still unclaimed
    try:
        job = json.loads(job_path.read_text())
        job["start_at"] = json.loads(start_path.read_text())["start_at"]
    except ValueError:
        return None  # os.replace is atomic, so this is a transient race, not corruption; retry next tick without claiming
    return job


_TOKENIZERS = {}  # (model, revision) -> tokenizer, loaded once per process


def cached_tokenizer(model, revision):
    """Loading a tokenizer is a disk/network-bound operation of unpredictable duration (observed
    on a real cluster: large enough, inconsistently enough between shards, to blow the start-skew
    tolerance below if done fresh inside execute() every job). Every job for one run uses the same
    (model, revision), so this process-lifetime cache removes that variance after the first job."""
    key = (model, revision)
    if key not in _TOKENIZERS:
        _TOKENIZERS[key] = tokenizer(model, revision)
    return _TOKENIZERS[key]


async def execute(job, shard_index, cancel_path):
    data = job["traffic"]
    tok = cached_tokenizer(data["model"], data["revision"])
    args = SimpleNamespace(**job["args"], shard_index=shard_index, shard_count=job["shard_count"], gpus=job["gpus"], url=job["router_url"],
                           architecture=job["architecture"], instance_type=job["instance_type"], region=job["region"],
                           evidence_scope=job["evidence_scope"])
    wait = job["start_at"] - time.time()
    if wait > 0:
        await asyncio.sleep(wait)
    actual_start_wall = time.time()
    cancel_event = asyncio.Event()

    async def watch():
        while not cancel_event.is_set():
            if cancel_path.exists():
                cancel_event.set()
                return
            await asyncio.sleep(0.5)
    watcher = asyncio.ensure_future(watch())
    try:
        result = await run_rate(data, args, job["rate"], job["phase"], tok, cancel_event)
    finally:
        cancel_event.set()
        watcher.cancel()
    cancelled = cancel_path.exists()
    # Non-authoritative by construction (DESIGN.md section 3.3): a rate/shard_count-scaled summarize()
    # must never be read as a verdict by anything downstream of labctl's merge.
    summary = dict(result["summary"], authoritative=False, shard_offered_tasks_per_s=job["rate"], shard_index=shard_index)
    return {"metadata": result["metadata"], "summary": summary, "requests": result["requests"],
            "segment_id": job["segment_id"], "shard_actual_start_wall": actual_start_wall, "cancelled": cancelled}


async def _status_server(port, status_ref):
    """A tiny in-cluster peer protocol (DESIGN-RATIONALE.md, "whole-ramp dispatch"): one shard's
    current segment index and completion wall-clock time, read by its peers over the StatefulSet's
    own headless-Service DNS (runner-<k>.runner.<namespace>.svc.cluster.local), not by labctl --
    labctl never opens a connection into the cluster (module docstring), and this is a separate
    network path from its own kubectl exec round trips. One line of JSON in, one line of JSON out;
    any request body is ignored, only the fact of a connection matters."""
    async def handle(reader, writer):
        try:
            await reader.readline()
            writer.write((json.dumps(status_ref) + "\n").encode())
            await writer.drain()
        except (ConnectionError, OSError):
            pass
        finally:
            writer.close()
    return await asyncio.start_server(handle, "0.0.0.0", port)


async def _peer_status(host, port, timeout=2.0):
    """None on any failure (peer not listening yet, DNS not resolved yet, timeout): the caller
    retries on its own cadence, exactly like try_claim's "not ready yet" return."""
    try:
        reader, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout)
    except (ConnectionError, OSError, asyncio.TimeoutError):
        return None
    try:
        writer.write(b"status\n")
        await writer.drain()
        line = await asyncio.wait_for(reader.readline(), timeout)
        return json.loads(line) if line else None
    except (ConnectionError, OSError, asyncio.TimeoutError, ValueError):
        return None
    finally:
        writer.close()


async def _measure_peer_rtt_s(peers, pings=3):
    """Run once, by every shard, during the dead time before the labctl-mediated start_at for
    segment 0 (there is nothing else to do then): the basis for this shard's own barrier_s between
    later segments, so that value is measured on this run's actual network path, not a constant
    carried over from a different cluster or day."""
    if not peers:
        return 0.0
    worst = 0.0
    for peer in peers:
        for _ in range(pings):
            before = time.perf_counter()
            got = await _peer_status(peer["host"], peer["port"], timeout=2.0)
            if got is not None:
                worst = max(worst, time.perf_counter() - before)
            await asyncio.sleep(0.05)
    return worst


async def _run_one_segment(job, step, shard_index, cancel_path, start_at, tok):
    """The same work execute() does for a single-segment job, factored out so execute_ramp can
    call it once per step of a whole-ramp job without duplicating benchmark.run_rate wiring."""
    data = job["traffic"]
    at = job["args_template"]
    args = SimpleNamespace(**at, seed=derive_shard_seed(job["base_seed"], shard_index), shard_index=shard_index, shard_count=job["shard_count"],
                           gpus=job["gpus"], url=job["router_url"], architecture=job["architecture"],
                           instance_type=job["instance_type"], region=job["region"], evidence_scope=job["evidence_scope"])
    wait = start_at - time.time()
    if wait > 0:
        await asyncio.sleep(wait)
    actual_start_wall = time.time()
    cancel_event = asyncio.Event()

    async def watch():
        while not cancel_event.is_set():
            if cancel_path.exists():
                cancel_event.set()
                return
            await asyncio.sleep(0.5)
    watcher = asyncio.ensure_future(watch())
    try:
        result = await run_rate(data, args, step["rate"] / job["shard_count"], step["phase"], tok, cancel_event)
    finally:
        cancel_event.set()
        watcher.cancel()
    cancelled = cancel_path.exists()
    summary = dict(result["summary"], authoritative=False, shard_offered_tasks_per_s=step["rate"] / job["shard_count"], shard_index=shard_index)
    return {"metadata": result["metadata"], "summary": summary, "requests": result["requests"],
            "segment_id": job["segment_id"], "shard_actual_start_wall": actual_start_wall, "cancelled": cancelled}


def derive_shard_seed(base_seed, shard_index):
    # Mirrors labctl.derive_seed exactly; duplicated rather than imported to keep this module's
    # only cross-file dependency on labctl.py at zero (labctl imports from here, not the reverse).
    return int(hashlib.sha256(f"{base_seed}:{shard_index}".encode()).hexdigest()[:8], 16)


async def execute_ramp(job, shard_index, root):
    """The whole-ramp counterpart to execute(): claimed once per run instead of once per (rate,
    phase) segment. labctl sends the complete remaining step list in one job, then is out of the
    loop until the end (DESIGN-RATIONALE.md, "whole-ramp dispatch") -- every step after the first
    starts on a start_at this shard and its peers agree on over the in-cluster peer protocol
    above, not on a fresh kubectl-exec round trip from labctl. Each step still writes its own
    `{segment_id}-{rate}-{phase}.ack.json`/`.result.json` under the pre-existing filenames, so
    labctl's merge() and rescore.py need no changes at all to read them.
    Deliberately dropped, and recorded here rather than hidden: labctl's stop_failure_fraction
    early-abort of not-yet-dispatched rates cannot apply once the whole plan is already on the
    shard -- every planned step runs; labctl still detects and reports the condition when it
    merges results, it just can no longer prevent the GPU time already spent running the rest."""
    data = job["traffic"]
    tok = cached_tokenizer(data["model"], data["revision"])
    _, cancel_path, ack_path = ramp_paths(root, job["segment_id"])
    is_leader = shard_index == 0
    status = {"segment_index": -1, "finished_at": None, "published_segment_index": -1, "next_start_at": None}
    server = await _status_server(job["http_port"], status)
    start_at = job["start_at"]
    barrier_s = None
    if is_leader and job["peers"]:
        barrier_s = max(0.2, 3.0 * await _measure_peer_rtt_s(job["peers"]))
        print(f"shard {shard_index}: measured peer barrier {barrier_s * 1000:.0f}ms", flush=True)
    at = job["args_template"]
    # Generous enough that a slow-but-alive peer never trips it (matches labctl's own
    # duration_s+timeout_s+60 deadline for a single segment); a peer that is actually gone (crashed,
    # evicted) must not hang this shard forever, so this is a real bound, not decoration.
    # Overridable by job["peer_wait_budget_s"] for tests only; labctl.dispatch_ramp_whole never
    # sets that key, so production always uses the formula below.
    peer_wait_budget_s = job.get("peer_wait_budget_s") or (at["duration_s"] + at["timeout_s"] + 60)
    ran, cancelled_early, peer_lost = [], False, False
    try:
        for i, step in enumerate(job["steps"]):
            if cancel_path.exists():
                cancelled_early = True
                break
            outcome = await _run_one_segment(job, step, shard_index, cancel_path, start_at, tok)
            job_name = f"{job['segment_id']}-{step['rate']:g}-{step['phase']}"
            atomic.atomic_write_json(root / "results" / f"{job_name}.result.json", outcome)
            atomic.atomic_write_json(root / "acks" / f"{job_name}.ack.json",
                                     {"status": "cancelled" if outcome["cancelled"] else "done",
                                      "segment_id": job["segment_id"], "shard_actual_start_wall": outcome["shard_actual_start_wall"]})
            ran.append(job_name)
            print(f"shard {shard_index}: finished {job_name}", flush=True)
            if outcome["cancelled"]:
                cancelled_early = True
                break
            finished_at = time.time()
            status["segment_index"] = i
            status["finished_at"] = finished_at
            if i + 1 >= len(job["steps"]):
                break
            peer_deadline = time.time() + peer_wait_budget_s
            if is_leader:
                peer_times = []
                for peer in job["peers"]:
                    while True:
                        if cancel_path.exists():
                            cancelled_early = True
                            break
                        if time.time() > peer_deadline:
                            peer_lost = True
                            break
                        got = await _peer_status(peer["host"], peer["port"])
                        if got is not None and got.get("segment_index", -1) >= i and got.get("finished_at") is not None:
                            peer_times.append(got["finished_at"])
                            break
                        await asyncio.sleep(PEER_POLL_S)
                    if cancelled_early or peer_lost:
                        break
                if cancelled_early:
                    break
                if peer_lost:
                    raise RuntimeError(f"peer did not report segment {i} complete within {peer_wait_budget_s:.0f}s; treating this ramp as failed rather than hanging")
                start_at = max([finished_at] + peer_times) + barrier_s
                status["published_segment_index"] = i
                status["next_start_at"] = start_at
            else:
                leader = next(p for p in job["peers"] if p["shard_index"] == 0)
                while True:
                    if cancel_path.exists():
                        cancelled_early = True
                        break
                    if time.time() > peer_deadline:
                        raise RuntimeError(f"leader did not publish segment {i}'s next start within {peer_wait_budget_s:.0f}s; treating this ramp as failed rather than hanging")
                    got = await _peer_status(leader["host"], leader["port"])
                    if got is not None and got.get("published_segment_index", -1) >= i and got.get("next_start_at") is not None:
                        start_at = got["next_start_at"]
                        break
                    await asyncio.sleep(PEER_POLL_S)
                if cancelled_early:
                    break
    finally:
        server.close()
        await server.wait_closed()
    atomic.atomic_write_json(ack_path, {"status": "cancelled" if cancelled_early else "done", "ran": ran})


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shard-index", type=int, required=True)
    p.add_argument("--local-root", default="/local")
    a = p.parse_args()
    root = Path(a.local_root)
    for sub in ("jobs", "acks", "results"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    claimed, claimed_ramps = set(), set()
    print(f"runner_agent shard {a.shard_index} ready, polling {root}", flush=True)
    while True:
        for ramp_file in sorted((root / "jobs").glob("*.ramp.json")):
            segment_id = ramp_file.name[: -len(".ramp.json")]
            if segment_id in claimed_ramps:
                continue
            _, _, ramp_ack_path = ramp_paths(root, segment_id)
            if ramp_ack_path.exists():
                claimed_ramps.add(segment_id)
                continue
            job = claim_ramp(root, segment_id)
            if job is None:
                continue
            claimed_ramps.add(segment_id)
            print(f"shard {a.shard_index}: claimed whole ramp {segment_id} ({len(job['steps'])} segment(s))", flush=True)
            try:
                asyncio.run(execute_ramp(job, a.shard_index, root))
            except Exception as exc:  # execute_ramp itself writes the ack on every path it controls; this only catches what escaped that.
                atomic.atomic_write_json(ramp_ack_path, {"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            print(f"shard {a.shard_index}: finished whole ramp {segment_id}", flush=True)
        for job_file in sorted((root / "jobs").glob("*.job.json")):
            job_name = job_file.name[: -len(".job.json")]
            if job_name in claimed:
                continue
            _, cancel_path, ack_path, result_path = paths(root, job_name)
            if ack_path.exists():
                # Already executed by this shard before a pod restart cleared `claimed` from memory
                # (DESIGN.md section 5: an agent that restarts must not re-run a job it already
                # finished) -- the job file persists on this shard's own volume, the ack does not lie.
                claimed.add(job_name)
                continue
            job = try_claim(root, job_name)
            if job is None:
                continue
            claimed.add(job_name)
            print(f"shard {a.shard_index}: claimed {job_name}", flush=True)
            try:
                outcome = asyncio.run(execute(job, a.shard_index, cancel_path))
                atomic.atomic_write_json(result_path, outcome)
                atomic.atomic_write_json(ack_path, {"status": "cancelled" if outcome["cancelled"] else "done",
                                                     "segment_id": job["segment_id"], "shard_actual_start_wall": outcome["shard_actual_start_wall"]})
            except Exception as exc:  # DESIGN.md section 5: only an agent's own unhandled error sets the run FAILED.
                atomic.atomic_write_json(ack_path, {"status": "failed", "segment_id": job["segment_id"], "error": f"{type(exc).__name__}: {exc}"})
            print(f"shard {a.shard_index}: finished {job_name}", flush=True)
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
