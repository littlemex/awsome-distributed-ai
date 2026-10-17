"""Open-loop task arrivals with sequential calls within each agent task."""
import asyncio
import hashlib
import json
import math
from pathlib import Path
import random
import time
import uuid
import httpx
from traffic import initial_prompt, continuation, tokenizer


async def stream_request(client, url, input_ids, output_tokens, timeout_s):
    started = time.perf_counter()
    started_wall = time.time()
    first = last_token = None
    first_count = 0
    text = ""
    meta = {}
    done = False
    chunks = []
    result = {"ok": False, "input_tokens": len(input_ids), "output_tokens": 0}
    try:
        async with asyncio.timeout(timeout_s):
            async with client.stream("POST", url.rstrip("/") + "/generate", json={"input_ids": input_ids, "sampling_params": {"temperature": 0.0, "max_new_tokens": output_tokens, "ignore_eos": True}, "stream": True}) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        done = True
                        break
                    event = json.loads(payload)
                    if "error" in event:
                        raise RuntimeError(str(event["error"]))
                    meta = event.get("meta_info", {})
                    count = meta.get("completion_tokens", 0)
                    if count > result["output_tokens"]:
                        last_token = time.perf_counter()
                        chunks.append({"t_ms": (last_token - started) * 1000, "n": count - result["output_tokens"]})
                        if first is None:
                            first = last_token
                            first_count = count
                    result["output_tokens"] = count
                    text = event.get("text", text)
            if not done or first is None or result["output_tokens"] != output_tokens:
                raise RuntimeError("Incomplete stream or output length mismatch")
            finish_reason = meta.get("finish_reason") or {}
            if isinstance(finish_reason, dict) and finish_reason.get("type") in ("abort", "error"):
                raise RuntimeError(str(finish_reason))
            result["ok"] = True
    except (httpx.HTTPError, TimeoutError, RuntimeError, ValueError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    ended = time.perf_counter()
    # Stream chunks can contain multiple tokens. Record that limitation explicitly.
    token_intervals = result["output_tokens"] - first_count
    result.update(ttft_ms=None if first is None else (first - started) * 1000,
                  tpot_ms=None if token_intervals <= 0 else (last_token - first) * 1000 / token_intervals,
                  latency_ms=(ended - started) * 1000, first_chunk_tokens=first_count,
                  cached_tokens=meta.get("cached_tokens"), text=text,
                  start_wall=started_wall, rid=meta.get("id"),
                  # Purely additive (DESIGN.md section 14.2): every chunk that carried new tokens, including
                  # the first. ttft_ms/tpot_ms above are unchanged in meaning and do not read this field.
                  chunks=chunks if result["ok"] else [])
    return result


def percentile(values, fraction):
    values = sorted(x for x in values if x is not None)
    return None if not values else values[max(0, math.ceil(len(values) * fraction) - 1)]


# A call that never connected, for example through a closed port forward, says nothing about the stack.
CLIENT_PATH_ERRORS = ("ConnectError",)

TPOT_METRICS = ("mean", "itl-p99", "worst-itl")
# "per-call" (default): every call's own TTFT must pass, unchanged from before this field existed.
# "p95"/"p99": TTFT moves out of the per-call test into one pooled, rate-level percentile across
# every completed call -- symmetric with how itl-p99 already treats TPOT (DESIGN.md section 14.4).
TTFT_METRICS = ("per-call", "p95", "p99")


def worst_itl_ms(record):
    """The single worst raw gap (not divided by token count) between two consecutive chunks of one
    response -- the longest pause that call's own stream actually produced. Unlike pooled_itls_ms,
    this is not normalized per new token: a chunk that batches several tokens after a long stall is
    one real pause the client waited through, not several short ones, and dividing by the token
    count would understate exactly the stall this metric exists to catch (observed on real traffic:
    a unified engine's prefill forward pass blocking its own decode stream mid-response). None if the
    record has fewer than two chunks (nothing to take a gap between)."""
    chunks = record.get("chunks") or []
    if len(chunks) < 2:
        return None
    return max(b["t_ms"] - a["t_ms"] for a, b in zip(chunks, chunks[1:]))


def pooled_itls_ms(records):
    """Per-chunk gap / new-token-count, pooled across every record and repeated once per new token --
    the same construction as sglang.bench_serving --backend sglang (DESIGN.md section 14.3). None if no
    record carries two or more chunks (predates per-chunk recording, or every call failed before a
    first chunk)."""
    pooled = []
    for r in records:
        chunks = r.get("chunks") or []
        for previous, current in zip(chunks, chunks[1:]):
            if current["n"] > 0:
                pooled += [(current["t_ms"] - previous["t_ms"]) / current["n"]] * current["n"]
    return pooled or None


def summarize(records, duration_s, gpus, ttft_ms, tpot_ms, attainment, offered_tasks_s, max_client_lag_ms, tpot_metric="mean", start_skew_ms=None, start_skew_tolerance_ms=500.0, ttft_metric="per-call"):
    # start_skew_ms is None for every single-process ramp (today's behavior, unchanged); a sharded
    # merge (labctl.merge) passes its measured cross-shard start skew here instead of overriding
    # client_valid/meets_joint_slo after the fact, so that re-scoring the same stored requests[]
    # at the same recorded start_skew_ms/tolerance reproduces the same verdict byte for byte
    # (rescore.check_reproduces calls this function fresh and has no other way to see the skew).
    assert tpot_metric in TPOT_METRICS
    assert ttft_metric in TTFT_METRICS
    ok = [r for r in records if r["ok"]]
    attempts = [r for r in records if r.get("sent", False)]
    lag = max((r["client_launch_lag_ms"] for r in records), default=0)
    unreachable = sum(r.get("error", "").startswith(CLIENT_PATH_ERRORS) for r in records)
    client_valid = lag <= max_client_lag_ms and not unreachable and (start_skew_ms is None or start_skew_ms <= start_skew_tolerance_ms)
    cached = [r["cached_tokens"] for r in records if r.get("cached_tokens") is not None]
    itls = pooled_itls_ms(records)
    itl_p99 = percentile(itls, .99) if itls is not None else None
    if tpot_metric == "itl-p99" and itls is None:
        raise SystemExit("--tpot-metric itl-p99 needs per-chunk recording; this file predates it, use --tpot-metric mean")
    worsts = [worst_itl_ms(r) for r in ok]
    worsts_present = [w for w in worsts if w is not None]
    if tpot_metric == "worst-itl" and not worsts_present:
        raise SystemExit("--tpot-metric worst-itl needs per-chunk recording; this file predates it, use --tpot-metric mean")
    ttft_values = [r.get("ttft_ms") for r in ok]
    ttft_p95 = percentile(ttft_values, .95)
    ttft_p99 = percentile(ttft_values, .99)
    if ttft_metric == "per-call":
        # per-call (default): every call's own TTFT must pass, exactly as today.
        ttft_good = [r for r in records if r["ok"] and r.get("ttft_ms") is not None and r["ttft_ms"] <= ttft_ms]
        meets_ttft = True
    else:
        # p95/p99: TTFT no longer gates one call at a time -- every completed call stays eligible,
        # and the pooled percentile across the whole rate is checked once, separately.
        ttft_good = ok
        pooled_ttft = ttft_p95 if ttft_metric == "p95" else ttft_p99
        meets_ttft = pooled_ttft is not None and pooled_ttft <= ttft_ms
    # mean (default): a call is "good" only if its own averaged TPOT also passes, exactly as today.
    # itl-p99 (DESIGN.md section 14.4): TPOT moves out of the per-call test into one pooled, rate-level
    # percentile; a call counts as good once its TTFT passes, and the pooled gap is checked separately.
    # worst-itl: like mean, a per-call test, but on that call's own single worst gap instead of its
    # average -- catches a response that stuttered badly once even though its average looked fine.
    def tpot_ok(r):
        if tpot_metric == "itl-p99":
            return True
        if tpot_metric == "worst-itl":
            w = worst_itl_ms(r)
            return w is not None and w <= tpot_ms
        return r.get("tpot_ms") is not None and r["tpot_ms"] <= tpot_ms
    good = [r for r in ttft_good if tpot_ok(r)]
    hit = len(good) / len(records) if records else 0.0
    meets_tpot = tpot_metric in ("mean", "worst-itl") or (itl_p99 is not None and itl_p99 <= tpot_ms)
    meets_joint = hit >= attainment and meets_ttft and meets_tpot
    return {"offered_tasks_per_s": offered_tasks_s, "actual_sent_calls_per_s": len(attempts) / duration_s,
            "measurement_including_drain_s": duration_s, "planned_calls": len(records), "sent_calls": len(attempts),
            "completed_calls": len(ok), "failed_or_skipped_calls": len(records) - len(ok),
            "p90_ttft_ms_success_only": percentile([r.get("ttft_ms") for r in ok], .9),
            "p95_ttft_ms_success_only": percentile([r.get("ttft_ms") for r in ok], .95),
            "p99_ttft_ms_success_only": percentile([r.get("ttft_ms") for r in ok], .99),
            "p90_tpot_ms_success_only": percentile([r.get("tpot_ms") for r in ok], .9),
            "itl_ms_p50": percentile(itls, .5) if itls is not None else None,
            "itl_ms_p95": percentile(itls, .95) if itls is not None else None,
            "itl_ms_p99": itl_p99,
            "worst_itl_ms_p50": percentile(worsts, .5) if worsts_present else None,
            "worst_itl_ms_p90": percentile(worsts, .9) if worsts_present else None,
            "worst_itl_ms_p95": percentile(worsts, .95) if worsts_present else None,
            "worst_itl_ms_p99": percentile(worsts, .99) if worsts_present else None,
            "worst_itl_ms_p999": percentile(worsts, .999) if worsts_present else None,
            "slo_ttft_ms": ttft_ms, "slo_tpot_ms": tpot_ms, "slo_attainment_fraction": hit,
            "tpot_metric": tpot_metric, "ttft_metric": ttft_metric,
            "attainment_target_fraction": attainment, "meets_joint_slo": meets_joint and client_valid,
            "successful_calls_per_s": len(ok) / duration_s, "useful_calls_per_s": len(good) / duration_s,
            "useful_calls_per_s_per_gpu": len(good) / duration_s / gpus, "gpu_count": gpus,
            "max_client_launch_lag_ms": lag, "unreachable_calls": unreachable, "client_valid": client_valid,
            "start_skew_ms": start_skew_ms, "start_skew_tolerance_ms": start_skew_tolerance_ms if start_skew_ms is not None else None,
            "prefix_overlap_fraction_min": min((r["prefix_overlap_fraction"] for r in records if r.get("prefix_overlap_fraction") is not None), default=None),
            "cached_tokens_sum_reported": sum(cached) if cached else None,
            "input_tokens_sum_sent": sum(r.get("input_tokens", 0) for r in attempts)}


def stops_ramp(summary, stop_failure_fraction):
    """Shared with rescore.check_runs, so the ramp and the re-scoring read one stop rule."""
    return summary["failed_or_skipped_calls"] / max(1, summary["planned_calls"]) > stop_failure_fraction


async def run_rate(data, a, rate, phase, tok, cancel_event=None):
    # No concurrency semaphore: slower responses must not reduce the offered task arrival rate.
    rng = random.Random(a.seed)
    arrivals = []
    when = 0.0
    while when < a.duration_s:
        arrivals.append(when)
        when += rng.expovariate(rate)
    if len(arrivals) > a.max_tasks:
        raise ValueError("Raise --max-tasks deliberately or reduce rate/duration; no arrivals were sent")
    nonce = f"{a.seed}:{phase}:{rate}:{uuid.uuid4().hex}"
    start = time.perf_counter()
    records = []
    # Sharding (DESIGN.md section 3.2): shard shard_index of shard_count walks task identities
    # shard_index, shard_index+shard_count, ... (a stride, not a contiguous block), so adjacent
    # arrivals across shards do not repeatedly draw the same task in lockstep. Defaults keep every
    # existing call site, including this module's own CPU-only tests, at today's single-process behavior.
    shard_index, shard_count = getattr(a, "shard_index", 0), getattr(a, "shard_count", 1)
    async with httpx.AsyncClient(timeout=None, limits=httpx.Limits(max_connections=None, max_keepalive_connections=512), trust_env=False) as client:
        async def task(index, arrival):
            await asyncio.sleep(max(0, start + arrival - time.perf_counter()))
            # Checked between tasks inside the existing gather (DESIGN.md section 3.2): a cancelled
            # run stops issuing new HTTP calls but lets tasks already mid-flight finish their current turn.
            if cancel_event is not None and cancel_event.is_set():
                records.append({"task_id": f"shard{shard_index}-{index}", "turn_index": 0, "phase": phase, "sent": False, "ok": False, "client_launch_lag_ms": 0, "prefix_overlap_fraction": None, "error": "Cancelled before launch"})
                return
            launch_lag = max(0, time.perf_counter() - start - arrival) * 1000
            task_index = index * shard_count + shard_index
            item = data["tasks"][task_index % len(data["tasks"])]
            identity = f"{item['task_id']}-{task_index}"
            prompt = initial_prompt(tok, data["shape"], identity, nonce)
            overlap = None
            failed = False
            for turn in range(item["calls"]):
                row = {"task_id": identity, "turn_index": turn, "phase": phase, "sent": False, "ok": False, "client_launch_lag_ms": launch_lag, "prefix_overlap_fraction": overlap}
                if failed:
                    row["error"] = "Skipped after failed predecessor"
                elif cancel_event is not None and cancel_event.is_set():
                    row["error"] = "Cancelled before this turn"
                elif len(prompt) + data["shape"]["output_tokens"] > a.context_length_tokens:
                    row["error"] = "Context budget exceeded"
                    failed = True
                else:
                    row["sent"] = True
                    row["input_ids_sha256"] = hashlib.sha256(json.dumps(prompt).encode()).hexdigest()
                    response = await stream_request(client, a.url, prompt, data["shape"]["output_tokens"], a.timeout_s)
                    answer = response.pop("text")
                    row.update(response)
                    if turn == 0 and row.get("ttft_ms") is not None:
                        row["ttft_ms"] += launch_lag
                    failed = not row["ok"]
                    if not failed and turn + 1 < item["calls"]:
                        try:
                            prompt, overlap = continuation(tok, data["shape"], prompt, answer, identity, turn + 1)
                        except ValueError as exc:
                            failed = True
                            row["continuation_error"] = str(exc)
                        await asyncio.sleep(data["shape"].get("tool_think_time_s", 0))
                records.append(row)
        await asyncio.gather(*(task(i, t) for i, t in enumerate(arrivals)))
    elapsed = max(a.duration_s, time.perf_counter() - start)
    summary = summarize(records, elapsed, a.gpus, a.ttft_slo_ms, a.tpot_slo_ms, a.attainment_fraction, rate, a.max_client_lag_ms, getattr(a, "tpot_metric", "mean"))
    summary["offered_window_s"] = a.duration_s
    summary["offered_task_count"] = len(arrivals)
    traffic_sha256 = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    return {"metadata": {"architecture": a.architecture, "instance_type": a.instance_type, "region": a.region, "evidence_scope": a.evidence_scope, "model": data["model"], "model_revision": data["revision"], "shape": data["shape"], "traffic_sha256": traffic_sha256, "phase": phase, "seed": a.seed, "run_nonce": nonce, "arguments": vars(a)}, "summary": summary, "requests": records}


async def ramp(a):
    out = Path(a.output)
    if any(out.glob("*.json")):
        raise SystemExit(f"{out} already holds ramp files; give each ramp invocation its own --output directory")
    data = json.loads(Path(a.traffic).read_text())
    tok = tokenizer(data["model"], data["revision"])
    out.mkdir(parents=True, exist_ok=True)
    a.ramp_id = uuid.uuid4().hex
    for rate in a.rates:
        if not a.skip_warmup:
            warm = await run_rate(data, a, rate, "warmup", tok)
            (out / f"{rate:g}-warmup.json").write_text(json.dumps(warm, indent=2))
        measured = await run_rate(data, a, rate, "measured", tok)
        (out / f"{rate:g}-measured.json").write_text(json.dumps(measured, indent=2))
        print(json.dumps({"metadata": measured["metadata"], "summary": measured["summary"]}), flush=True)
        if stops_ramp(measured["summary"], a.stop_failure_fraction):
            print("Ramp stopped at configured failure fraction; failures remain in attainment denominator.", flush=True)
            break
