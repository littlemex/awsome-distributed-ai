"""Disposable debug patch for the PAGE-CACHE-TRANSFER investigation (not committed, not part of
any PR; applied as one throwaway Docker layer on top of the pinned engine image, the exact same
procedure DESIGN-RATIONALE.md's "Reproducing this PR's own investigation measurements" item 5
documents for the descriptor-count diagnostic). Adds additive logging only; no existing line is
removed or changed. Every insertion is a single assert-guarded str.replace so the patch fails
loudly (ImportError/AssertionError at image build time) instead of silently patching the wrong
place if the pinned source ever drifts from what this investigation read.

(1) deepseek_v2.py: wraps the one call site that resolves the final AttnForwardMethod decision
    (dispatch_attn_forward_method) and logs it, plus sum(extend_prefix_lens_cpu), for every extend
    (prefill) call -- a direct observation of the dispatch PAGE64-MHA.md otherwise only inferred
    from prefill_forward timing.
(2) nixl/conn.py: five additive log points around the KV handoff, all tagged with the request's
    own bootstrap_room so they can be joined per request after the fact:
      T0 enqueue        -- NixlKVSender.send(), when _transfer_start_time is first set
      T1 dequeue        -- transfer_worker, right after queue.get()
      T2 kv_before_post -- _send_kvcache_generic, right before the real descriptor count
                            (src_reqs.shape[0], the same variable RESULTS.md section 5 /
                            APPENDIX-E-FACTS item (a) already identified as correct) is known
      T3 kv_after_post  -- _send_kvcache_generic, right after agent.transfer() returns
      T3b aux_after_post-- send_aux, right after its own agent.transfer() returns
      T4 success        -- transfer_worker, right after update_status(room, KVPoll.Success)
(3) disaggregation/prefill.py: T5 stage_end -- right after
    req.time_stats.set_prefill_kv_transfer_finish_time(), the same call that already feeds the
    official prefill_transfer_kv_cache Prometheus stage counter -- so this patch's own T0..T4 can
    be checked against that pre-existing, unpatched counter rather than only against itself.
"""
import sglang
import pathlib

ROOT = pathlib.Path(sglang.__file__).parent / "srt"
LOG_IMPORT = "import logging as _a345_logging, time as _a345_time"


def replace_once(path, old, new, count=1):
    text = path.read_text()
    n = text.count(old)
    assert n == count, f"{path}: expected {count} occurrence(s) of block, found {n}:\n{old!r}"
    path.write_text(text.replace(old, new, count))


# ---------------------------------------------------------------------------
# (1) attention dispatch -- deepseek_v2.py
# ---------------------------------------------------------------------------
p_dsv2 = ROOT / "models" / "deepseek_v2.py"
old = """        handler = AttentionBackendRegistry.get_handler(attention_backend)
        return handler(self, forward_batch)"""
new = """        handler = AttentionBackendRegistry.get_handler(attention_backend)
        _a345_method = handler(self, forward_batch)
        if not _a345_torch.compiler.is_compiling():
            _a345_log_dispatch(forward_batch, _a345_method)
        return _a345_method"""
replace_once(p_dsv2, old, new)

# Module-level helper, called only when NOT inside Dynamo tracing. Three earlier versions of this
# patch all crashed the unified stack's engine at startup during init_piecewise_cuda_graphs's own
# warmup_compile call (a fullgraph=True trace with zero tolerance for graph breaks); each was
# found by actually deploying the unified stack, since the disaggregated prefill/decode roles in
# this investigation's own configs never run that warmup path and so never surfaced any of this:
# (1) logging inline with no guard raised torch._dynamo.exc.Unsupported on logging.getLogger's
# own name-equality check; (2) wrapping a *nested* function with `@torch.compiler.disable`
# *inside* dispatch_attn_forward_method still failed -- Dynamo traces the act of applying the
# decorator (a call to the skip-marked `disable` itself) before it ever gets to skip the decorated
# function's body; (3) calling an already-disable-wrapped, module-level function still failed --
# in fullgraph mode, Dynamo treats a disable()d function call as a graph break, and fullgraph
# forbids graph breaks of any kind, disabled-function calls included, so it still raises instead
# of silently skipping. `torch.compiler.is_compiling()` is Dynamo's own documented, specially
# recognized predicate for exactly this situation (branching the traced code's own control flow
# on whether it is currently being traced) -- guarding the call site with it, rather than
# disabling the callee, means the call is never even reached while tracing, so there is nothing
# for Dynamo to graph-break on.
old = """from sglang.srt.models.deepseek_common.attention_backend_handler import (
    AttentionBackendRegistry,
)"""
new = """from sglang.srt.models.deepseek_common.attention_backend_handler import (
    AttentionBackendRegistry,
)
import torch as _a345_torch


def _a345_log_dispatch(forward_batch, method):
    try:
        import logging as _a345_logging
        import time as _a345_time
        if forward_batch.forward_mode.is_extend_without_speculative():
            _a345_logging.getLogger("aim345").warning(
                "attn_dispatch cached=%d method=%s ts=%.9f",
                int(sum(forward_batch.extend_prefix_lens_cpu)),
                method.name,
                _a345_time.perf_counter(),
            )
    except Exception:
        pass"""
replace_once(p_dsv2, old, new)

# ---------------------------------------------------------------------------
# (2) KV transfer handoff -- nixl/conn.py
# ---------------------------------------------------------------------------
p_conn = ROOT / "disaggregation" / "nixl" / "conn.py"

# T0: NixlKVSender.send(), when _transfer_start_time is first set.
old = """        if self._transfer_start_time is None and (
            len(kv_indices) > 0 or state_indices is not None
        ):
            self._transfer_start_time = time.perf_counter()"""
new = """        if self._transfer_start_time is None and (
            len(kv_indices) > 0 or state_indices is not None
        ):
            self._transfer_start_time = time.perf_counter()
            try:
                logging.getLogger("aim345").warning(
                    "kvtimer room=%d event=T0_enqueue ts=%.9f",
                    self.bootstrap_room, self._transfer_start_time,
                )
            except Exception:
                pass"""
replace_once(p_conn, old, new)

# T1: transfer_worker, right after dequeue.
old = """            kv_chunk: TransferKVChunk = queue.get()
            room = kv_chunk.room"""
new = """            kv_chunk: TransferKVChunk = queue.get()
            room = kv_chunk.room
            try:
                logging.getLogger("aim345").warning(
                    "kvtimer room=%d event=T1_dequeue ts=%.9f", room, time.perf_counter()
                )
            except Exception:
                pass"""
replace_once(p_conn, old, new)

# T2 (descriptor count, before post) + T3 (after post): _send_kvcache_generic. Patched as one
# combined block spanning from src_reqs/dst_reqs (the real, post-grouping descriptor arrays --
# same variable RESULTS.md section 5 / APPENDIX-E-FACTS item (a) already identified as correct)
# through this function's own return. The "xfer_handle = self.agent.initialize_xfer(...) / state
# = self.agent.transfer(...) / return xfer_handle" tail is byte-identical to send_aux's own tail
# (verified: it occurs exactly twice in this file) -- disambiguated here by keeping it anchored to
# the unique, VRAM-typed src_reqs/dst_reqs preamble above it, which only this function has
# (send_aux's own descriptors are DRAM-typed and built from src_addrs/dst_addrs, patched
# separately below).
old = """        src_reqs = make_req_array(src_addrs, src_lens, self.kv_args.gpu_id)
        dst_reqs = make_req_array(dst_addrs, dst_lens, dst_gpu_id)

        logger.debug(
            f"len(src_addrs): before group: {len(prefill_data_indices)}, after group: {len(src_addrs)}"
        )
        src_descs = self.agent.get_xfer_descs(src_reqs, "VRAM")
        dst_descs = self.agent.get_xfer_descs(dst_reqs, "VRAM")
        # Transfer data
        xfer_handle = self.agent.initialize_xfer(
            "WRITE",
            src_descs,
            dst_descs,
            peer_name,
            notif.encode("ascii"),  # type: ignore
        )
        if not xfer_handle:
            raise Exception("KVSender failed to create transfer")
        state = self.agent.transfer(xfer_handle)
        if state == "ERR":
            raise Exception("KVSender failed to post transfer")
        return xfer_handle"""
new = """        src_reqs = make_req_array(src_addrs, src_lens, self.kv_args.gpu_id)
        dst_reqs = make_req_array(dst_addrs, dst_lens, dst_gpu_id)

        logger.debug(
            f"len(src_addrs): before group: {len(prefill_data_indices)}, after group: {len(src_addrs)}"
        )
        try:
            _a345_room = int(str(notif).split("_")[0])
            logging.getLogger("aim345").warning(
                "kvtimer room=%d event=T2_kv_before_post ts=%.9f descriptor_count=%d",
                _a345_room, time.perf_counter(), src_reqs.shape[0],
            )
        except Exception:
            pass
        src_descs = self.agent.get_xfer_descs(src_reqs, "VRAM")
        dst_descs = self.agent.get_xfer_descs(dst_reqs, "VRAM")
        # Transfer data
        xfer_handle = self.agent.initialize_xfer(
            "WRITE",
            src_descs,
            dst_descs,
            peer_name,
            notif.encode("ascii"),  # type: ignore
        )
        if not xfer_handle:
            raise Exception("KVSender failed to create transfer")
        state = self.agent.transfer(xfer_handle)
        if state == "ERR":
            raise Exception("KVSender failed to post transfer")
        try:
            logging.getLogger("aim345").warning(
                "kvtimer room=%d event=T3_kv_after_post ts=%.9f", _a345_room, time.perf_counter()
            )
        except Exception:
            pass
        return xfer_handle"""
replace_once(p_conn, old, new)

# T3b: send_aux, right after its own agent.transfer() returns ("Failed to create transfer" /
# "Failed to post transfer" strings reused verbatim from the same two-raise shape are what makes
# this block textually identical to T3's old-block above except for this file's own send_aux
# docstring-free signature right above it; replaced via the distinct preceding dst_descs
# assignment line, which in send_aux reads "DRAM" (aux buffers are host memory), not "VRAM" as in
# _send_kvcache_generic (KV cache lives in VRAM) -- this is the actual, source-level distinguisher
# between the two call sites, not an arbitrary occurrence count.
old = """        src_descs = self.agent.get_xfer_descs(src_addrs, "DRAM")
        dst_descs = self.agent.get_xfer_descs(dst_addrs, "DRAM")
        # Transfer data
        xfer_handle = self.agent.initialize_xfer(
            "WRITE",
            src_descs,
            dst_descs,
            peer_name,
            notif.encode("ascii"),  # type: ignore
        )
        if not xfer_handle:
            raise Exception("KVSender failed to create transfer")
        state = self.agent.transfer(xfer_handle)
        if state == "ERR":
            raise Exception("KVSender failed to post transfer")
        return xfer_handle"""
new = """        src_descs = self.agent.get_xfer_descs(src_addrs, "DRAM")
        dst_descs = self.agent.get_xfer_descs(dst_addrs, "DRAM")
        # Transfer data
        xfer_handle = self.agent.initialize_xfer(
            "WRITE",
            src_descs,
            dst_descs,
            peer_name,
            notif.encode("ascii"),  # type: ignore
        )
        if not xfer_handle:
            raise Exception("KVSender failed to create transfer")
        state = self.agent.transfer(xfer_handle)
        if state == "ERR":
            raise Exception("KVSender failed to post transfer")
        try:
            _a345_room = int(str(notif).split("_")[0])
            logging.getLogger("aim345").warning(
                "kvtimer room=%d event=T3b_aux_after_post ts=%.9f", _a345_room, time.perf_counter()
            )
        except Exception:
            pass
        return xfer_handle"""
replace_once(p_conn, old, new)

# T4: transfer_worker, right after update_status(room, KVPoll.Success) on the is_last branch.
old = """                if kv_chunk.is_last:
                    self.update_status(room, KVPoll.Success)
                    # Drop per-room state on Success (parity with mooncake
                    # transfer_worker; staging prefetch sets are NIXL-only).
                    self.transfer_infos.pop(room, None)"""
new = """                if kv_chunk.is_last:
                    self.update_status(room, KVPoll.Success)
                    try:
                        logging.getLogger("aim345").warning(
                            "kvtimer room=%d event=T4_success ts=%.9f", room, time.perf_counter()
                        )
                    except Exception:
                        pass
                    # Drop per-room state on Success (parity with mooncake
                    # transfer_worker; staging prefetch sets are NIXL-only).
                    self.transfer_infos.pop(room, None)"""
replace_once(p_conn, old, new)

# ---------------------------------------------------------------------------
# (3) stage-end -- disaggregation/prefill.py
# ---------------------------------------------------------------------------
p_prefill = ROOT / "disaggregation" / "prefill.py"
old = """                done_reqs.append(req)
                req.time_stats.set_prefill_kv_transfer_finish_time()"""
new = """                done_reqs.append(req)
                req.time_stats.set_prefill_kv_transfer_finish_time()
                try:
                    import logging as _a345_logging, time as _a345_time
                    _a345_logging.getLogger("aim345").warning(
                        "kvtimer room=%d event=T5_stage_end ts=%.9f",
                        req.bootstrap_room, _a345_time.perf_counter(),
                    )
                except Exception:
                    pass"""
replace_once(p_prefill, old, new)

# OFFICIAL_T0: logged right where the official stage timer itself starts (the exact call site a
# reviewer round asked be timestamped directly, instead of inferred from source alone, after a
# sum-arithmetic check showed this round's own T0-based total cannot be reconciled with
# stages.py's own counter as a simple population/boundary-rounding difference). Same scheduler
# thread, same statement sequence as T0 (inside NixlKVSender.send(), called from send_kv_chunk,
# which returns the line before this one runs) -- if OFFICIAL_T0 and T0 disagree materially for
# the same room, that is direct, measured evidence of where the two methods' own "start" diverges.
old = """                self.send_kv_chunk(req, last_chunk=True)
                req.time_stats.set_prefill_transfer_queue_entry_time()"""
new = """                self.send_kv_chunk(req, last_chunk=True)
                req.time_stats.set_prefill_transfer_queue_entry_time()
                try:
                    import logging as _a345_logging, time as _a345_time
                    _a345_logging.getLogger("aim345").warning(
                        "kvtimer room=%d event=OFFICIAL_T0 ts=%.9f",
                        req.bootstrap_room, _a345_time.perf_counter(),
                    )
                except Exception:
                    pass"""
replace_once(p_prefill, old, new)

print("aim345_pct_patch applied OK:", p_dsv2, p_conn, p_prefill)
