"""Disposable debug patch for OPEN-ITEMS-NODE1 (items 1, 2, 5, 9's shared instrumentation needs).
Not committed, not part of any PR; one throwaway Docker layer on top of a pinned engine image,
same procedure DESIGN-RATIONALE.md's "Reproducing this PR's own investigation measurements" item 5
and PAGE-CACHE-TRANSFER.md's "How the instrumentation was built" document. Additive only; no
existing line removed or changed; every insertion is an assert-guarded str.replace so the patch
fails loudly at image build time if the pinned source has drifted from what was read.

Unlike the earlier aim345_pct_patch.py (PAGE-CACHE-TRANSFER.md), every kvtimer/attn_dispatch/iter
event here is recorded through aim345_ring.rec() -- a lock-free-in-spirit, no-logging-module ring
buffer (see aim345_ring.py) -- instead of logging.getLogger(...).warning(...). This is item 2's
own ask: separate "the logging call itself stalls" from "GIL/lock contention unrelated to
logging" by running the *same* six-point kvtimer instrumentation through two different recording
mechanisms and comparing. Also adds (item 1): per-scheduler-iteration timing in both
event_loop_overlap_disagg_prefill (the default path when overlap scheduling is on, which this
lab's engine args never disable) and event_loop_normal_disagg_prefill (patched too, for
robustness, in case a future config disables overlap) -- run_batch time, process_batch_result
time, process_disagg_prefill_inflight_queue (poll) time, inflight-queue depth, and completions
seen that iteration, one ring-buffer record per loop iteration.
"""
import shutil
import sglang
import pathlib

ROOT = pathlib.Path(sglang.__file__).parent / "srt"


def replace_once(path, old, new, count=1):
    text = path.read_text()
    n = text.count(old)
    assert n == count, f"{path}: expected {count} occurrence(s) of block, found {n}:\n{old!r}"
    path.write_text(text.replace(old, new, count))


# ---------------------------------------------------------------------------
# (0) ring buffer module
# ---------------------------------------------------------------------------
shutil.copy("/tmp/aim345_ring.py", ROOT / "disaggregation" / "aim345_ring.py")

# ---------------------------------------------------------------------------
# (1) attention dispatch -- deepseek_v2.py (same Dynamo-safety guard as aim345_pct_patch.py)
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

old = """from sglang.srt.models.deepseek_common.attention_backend_handler import (
    AttentionBackendRegistry,
)"""
new = """from sglang.srt.models.deepseek_common.attention_backend_handler import (
    AttentionBackendRegistry,
)
import torch as _a345_torch


def _a345_log_dispatch(forward_batch, method):
    try:
        from sglang.srt.disaggregation.aim345_ring import rec as _a345_rec
        if forward_batch.forward_mode.is_extend_without_speculative():
            _a345_rec("attn_dispatch", -1, cached=int(sum(forward_batch.extend_prefix_lens_cpu)), method=method.name)
    except Exception:
        pass"""
replace_once(p_dsv2, old, new)

# ---------------------------------------------------------------------------
# (2) KV transfer handoff -- nixl/conn.py, ring-buffer variant of T0..T4/T3b
# ---------------------------------------------------------------------------
p_conn = ROOT / "disaggregation" / "nixl" / "conn.py"

old = """        if self._transfer_start_time is None and (
            len(kv_indices) > 0 or state_indices is not None
        ):
            self._transfer_start_time = time.perf_counter()"""
new = """        if self._transfer_start_time is None and (
            len(kv_indices) > 0 or state_indices is not None
        ):
            self._transfer_start_time = time.perf_counter()
            try:
                from sglang.srt.disaggregation.aim345_ring import rec as _a345_rec
                _a345_rec("T0_enqueue", self.bootstrap_room, ts=self._transfer_start_time)
            except Exception:
                pass"""
replace_once(p_conn, old, new)

old = """            kv_chunk: TransferKVChunk = queue.get()
            room = kv_chunk.room"""
new = """            kv_chunk: TransferKVChunk = queue.get()
            room = kv_chunk.room
            try:
                from sglang.srt.disaggregation.aim345_ring import rec as _a345_rec
                _a345_rec("T1_dequeue", room)
            except Exception:
                pass"""
replace_once(p_conn, old, new)

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
            from sglang.srt.disaggregation.aim345_ring import rec as _a345_rec
            _a345_room = int(str(notif).split("_")[0])
            _a345_rec("T2_kv_before_post", _a345_room, descriptor_count=int(src_reqs.shape[0]))
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
            _a345_rec("T3_kv_after_post", _a345_room)
        except Exception:
            pass
        return xfer_handle"""
replace_once(p_conn, old, new)

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
            from sglang.srt.disaggregation.aim345_ring import rec as _a345_rec
            _a345_room = int(str(notif).split("_")[0])
            _a345_rec("T3b_aux_after_post", _a345_room)
        except Exception:
            pass
        return xfer_handle"""
replace_once(p_conn, old, new)

old = """                if kv_chunk.is_last:
                    self.update_status(room, KVPoll.Success)
                    # Drop per-room state on Success (parity with mooncake
                    # transfer_worker; staging prefetch sets are NIXL-only).
                    self.transfer_infos.pop(room, None)"""
new = """                if kv_chunk.is_last:
                    self.update_status(room, KVPoll.Success)
                    try:
                        from sglang.srt.disaggregation.aim345_ring import rec as _a345_rec
                        _a345_rec("T4_success", room)
                    except Exception:
                        pass
                    # Drop per-room state on Success (parity with mooncake
                    # transfer_worker; staging prefetch sets are NIXL-only).
                    self.transfer_infos.pop(room, None)"""
replace_once(p_conn, old, new)

# ---------------------------------------------------------------------------
# (3) prefill.py: imports, T5/OFFICIAL_T0 (ring-buffer variant), per-iteration timing
# ---------------------------------------------------------------------------
p_prefill = ROOT / "disaggregation" / "prefill.py"

old = """import logging
from collections import deque"""
new = """import logging
from collections import deque
from time import perf_counter as _a345_pc
from sglang.srt.disaggregation.aim345_ring import rec as _a345_rec"""
replace_once(p_prefill, old, new)

old = """                done_reqs.append(req)
                req.time_stats.set_prefill_kv_transfer_finish_time()"""
new = """                done_reqs.append(req)
                req.time_stats.set_prefill_kv_transfer_finish_time()
                try:
                    _a345_rec("T5_stage_end", req.bootstrap_room)
                except Exception:
                    pass"""
replace_once(p_prefill, old, new)

old = """                self.send_kv_chunk(req, last_chunk=True)
                req.time_stats.set_prefill_transfer_queue_entry_time()"""
new = """                self.send_kv_chunk(req, last_chunk=True)
                req.time_stats.set_prefill_transfer_queue_entry_time()
                try:
                    _a345_rec("OFFICIAL_T0", req.bootstrap_room)
                except Exception:
                    pass"""
replace_once(p_prefill, old, new)

# Item 1: per-iteration timing, event_loop_overlap_disagg_prefill (the default path: this lab's
# engine args never pass --disable-overlap-schedule, and server_args.py's own default is
# disable_overlap_schedule=False, i.e. overlap scheduling on).
old = """        while True:
            # Receive requests
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)
            self.waiting_queue.extend(
                self.disagg_prefill_bootstrap_queue.pop_bootstrapped()
            )
            if self._engine_paused:
                continue

            # Get the next batch to run
            batch = self.get_next_disagg_prefill_batch_to_run()
            self.cur_batch = batch

            # Launch the current batch
            if batch:
                if self.enable_staging:
                    self.maybe_prefetch_staging_for_batch(batch)
                batch_result = self.run_batch(batch)
                self.result_queue.append((batch.copy(), batch_result))
            else:
                batch_result = None

            # Process the last batch
            if self.last_batch:
                tmp_batch, tmp_result = self.result_queue.popleft()
                self.process_batch_result(tmp_batch, tmp_result)
            elif batch is None:
                # When the server is idle, do self-check and re-init some states
                self.on_idle()

            self.process_disagg_prefill_inflight_queue()

            # Run sample of the current batch
            # It depends on the result of the last batch (e.g., grammar), so we run it after the last batch is processed.
            self.launch_batch_sample_if_needed(batch_result)

            # Update last_batch
            self.last_batch = batch"""
new = """        while True:
            _a345_iter_t0 = _a345_pc()
            # Receive requests
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)
            self.waiting_queue.extend(
                self.disagg_prefill_bootstrap_queue.pop_bootstrapped()
            )
            if self._engine_paused:
                continue

            # Get the next batch to run
            batch = self.get_next_disagg_prefill_batch_to_run()
            self.cur_batch = batch

            # Launch the current batch
            _a345_run_batch_ms = -1.0
            if batch:
                if self.enable_staging:
                    self.maybe_prefetch_staging_for_batch(batch)
                _a345_t = _a345_pc()
                batch_result = self.run_batch(batch)
                _a345_run_batch_ms = (_a345_pc() - _a345_t) * 1000.0
                self.result_queue.append((batch.copy(), batch_result))
            else:
                batch_result = None

            # Process the last batch
            _a345_pbr_ms = -1.0
            _a345_on_idle = False
            if self.last_batch:
                tmp_batch, tmp_result = self.result_queue.popleft()
                _a345_t = _a345_pc()
                self.process_batch_result(tmp_batch, tmp_result)
                _a345_pbr_ms = (_a345_pc() - _a345_t) * 1000.0
            elif batch is None:
                # When the server is idle, do self-check and re-init some states
                _a345_on_idle = True
                self.on_idle()

            _a345_inflight_before = len(getattr(self, "disagg_prefill_inflight_queue", None) or [])
            _a345_t = _a345_pc()
            _a345_done = self.process_disagg_prefill_inflight_queue()
            _a345_poll_ms = (_a345_pc() - _a345_t) * 1000.0
            # Record only iterations that actually did something: ran a batch, processed a last
            # batch, or polled a nonempty inflight queue (item 1's own question is about what each
            # iteration does when there is work, not about the idle busy-spin's own steady state,
            # which this overlap loop runs at well over 1kHz with nothing in flight -- recording
            # every idle iteration made a single 20s window dump beyond a gigabyte per pid and
            # never stopped growing while the deployment stayed up, discovered live this round).
            if batch or _a345_pbr_ms >= 0 or _a345_inflight_before > 0:
                try:
                    _a345_rec(
                        "iter_overlap", -1,
                        batch_present=bool(batch), on_idle=_a345_on_idle,
                        run_batch_ms=round(_a345_run_batch_ms, 4),
                        process_batch_result_ms=round(_a345_pbr_ms, 4),
                        poll_ms=round(_a345_poll_ms, 4),
                        inflight_before=_a345_inflight_before,
                        completions=len(_a345_done) if _a345_done else 0,
                        iter_total_ms=round((_a345_pc() - _a345_iter_t0) * 1000.0, 4),
                    )
                except Exception:
                    pass

            # Run sample of the current batch
            # It depends on the result of the last batch (e.g., grammar), so we run it after the last batch is processed.
            self.launch_batch_sample_if_needed(batch_result)

            # Update last_batch
            self.last_batch = batch"""
replace_once(p_prefill, old, new)

# Also instrument event_loop_normal_disagg_prefill, for robustness (not expected to run under
# this lab's own default args, but patched so a future config change does not silently lose this
# instrumentation).
old = """            # Get the next batch to run
            batch = self.get_next_disagg_prefill_batch_to_run()
            self.cur_batch = batch

            # Launch the current batch
            if batch:
                if self.enable_staging:
                    self.maybe_prefetch_staging_for_batch(batch)
                result = self.run_batch(batch)
                self.process_batch_result(batch, result)
            else:
                self.on_idle()

            self.process_disagg_prefill_inflight_queue()

            # Update last_batch
            self.last_batch = batch

    @torch.no_grad()
    def event_loop_overlap_disagg_prefill(self: Scheduler) -> None:"""
new = """            # Get the next batch to run
            batch = self.get_next_disagg_prefill_batch_to_run()
            self.cur_batch = batch

            # Launch the current batch
            _a345_run_batch_ms = -1.0
            _a345_pbr_ms = -1.0
            if batch:
                if self.enable_staging:
                    self.maybe_prefetch_staging_for_batch(batch)
                _a345_t = _a345_pc()
                result = self.run_batch(batch)
                _a345_run_batch_ms = (_a345_pc() - _a345_t) * 1000.0
                _a345_t = _a345_pc()
                self.process_batch_result(batch, result)
                _a345_pbr_ms = (_a345_pc() - _a345_t) * 1000.0
            else:
                self.on_idle()

            _a345_inflight_before = len(getattr(self, "disagg_prefill_inflight_queue", None) or [])
            _a345_t = _a345_pc()
            _a345_done = self.process_disagg_prefill_inflight_queue()
            _a345_poll_ms = (_a345_pc() - _a345_t) * 1000.0
            if batch or _a345_pbr_ms >= 0 or _a345_inflight_before > 0:
                try:
                    _a345_rec(
                        "iter_normal", -1,
                        batch_present=bool(batch),
                        run_batch_ms=round(_a345_run_batch_ms, 4),
                        process_batch_result_ms=round(_a345_pbr_ms, 4),
                        poll_ms=round(_a345_poll_ms, 4),
                        inflight_before=_a345_inflight_before,
                        completions=len(_a345_done) if _a345_done else 0,
                    )
                except Exception:
                    pass

            # Update last_batch
            self.last_batch = batch

    @torch.no_grad()
    def event_loop_overlap_disagg_prefill(self: Scheduler) -> None:"""
replace_once(p_prefill, old, new)

print("aim345_oi1_patch applied OK:", p_dsv2, p_conn, p_prefill, ROOT / "disaggregation" / "aim345_ring.py")
