"""Disposable debug patch for OPEN-ITEMS-NODE3, applied on top of aim345_oi1_patch.py.

Adds, through the same aim345_ring list recorder:
- the exact setter timestamps and rid at the two stage setters (C: identity-level reconciliation);
- the sglang:kv_transfer_latency_ms value per request, with rid and bytes (B: metric endpoints);
- one record per prefill send() with chunk_id, KV page count and is_last (B: chunk splits);
- the composition of every launched prefill batch: rid, room, extend length, prefix length,
  whether the request is the batch's new chunked request (B);
- absolute start/end timestamps of each scheduler-loop phase, on every TP rank (G);
- in the transfer worker's completion wait: number of check_xfer_state polls, the longest gap
  between two polls, and the time of the first DONE (A);
- an env-gated control, AIM345_SUPPRESS_TINY_CHUNK=N: a new request is not admitted as a
  chunk shorter than or equal to N tokens (B: removes the tiny-first-chunk path only).
Every insertion is assert-guarded so a drifted source fails the image build.
"""
import pathlib
import sglang

ROOT = pathlib.Path(sglang.__file__).parent / "srt"


def replace_once(path, old, new, count=1):
    text = path.read_text()
    n = text.count(old)
    assert n == count, f"{path}: expected {count} occurrence(s) of block, found {n}:\n{old!r}"
    path.write_text(text.replace(old, new, count))


p_prefill = ROOT / "disaggregation" / "prefill.py"
p_conn = ROOT / "disaggregation" / "nixl" / "conn.py"
p_policy = ROOT / "managers" / "schedule_policy.py"

# --- C: exact setter values and rid at both stage setters ---
replace_once(p_prefill, """                req.time_stats.set_prefill_kv_transfer_finish_time()
                try:
                    _a345_rec("T5_stage_end", req.bootstrap_room)
                except Exception:
                    pass""", """                req.time_stats.set_prefill_kv_transfer_finish_time()
                try:
                    _a345_ts = req.time_stats
                    _a345_rec("T5_stage_end", req.bootstrap_room, rid=req.rid,
                              setter_entry=_a345_ts.prefill_transfer_queue_entry_time,
                              setter_finish=_a345_ts.prefill_kv_transfer_finish_time,
                              tp=self.tp_rank)
                except Exception:
                    pass""")
replace_once(p_prefill, """                req.time_stats.set_prefill_transfer_queue_entry_time()
                try:
                    _a345_rec("OFFICIAL_T0", req.bootstrap_room)
                except Exception:
                    pass""", """                req.time_stats.set_prefill_transfer_queue_entry_time()
                try:
                    _a345_rec("OFFICIAL_T0", req.bootstrap_room, rid=req.rid, tp=self.tp_rank,
                              setter_entry=req.time_stats.prefill_transfer_queue_entry_time)
                except Exception:
                    pass""")

# --- B: kv_transfer_latency_ms value per request ---
replace_once(p_prefill, """            metrics = req.time_stats.compute_and_observe_kv_transfer_metrics(
                req.disagg_kv_sender.get_transfer_metric()
            )""", """            metrics = req.time_stats.compute_and_observe_kv_transfer_metrics(
                req.disagg_kv_sender.get_transfer_metric()
            )
            try:
                _a345_tm = req.disagg_kv_sender.get_transfer_metric()
                _a345_rec("kvmetric", req.bootstrap_room, rid=req.rid, tp=self.tp_rank,
                          latency_ms=(metrics or {}).get("latency_ms"),
                          total_mb=(metrics or {}).get("total_mb"),
                          transfer_latency_s=getattr(_a345_tm, "transfer_latency_s", None),
                          transfer_total_bytes=getattr(_a345_tm, "transfer_total_bytes", None),
                          completion_time=req.time_stats.completion_time)
            except Exception:
                pass""")

# --- B: every prefill send() ---
replace_once(p_conn, """        index_slice = slice(self.curr_idx, self.curr_idx + len(kv_indices))
        self.curr_idx += len(kv_indices)
        is_last = self.curr_idx == self.num_kv_indices
""", """        index_slice = slice(self.curr_idx, self.curr_idx + len(kv_indices))
        self.curr_idx += len(kv_indices)
        is_last = self.curr_idx == self.num_kv_indices
        try:
            from sglang.srt.disaggregation.aim345_ring import rec as _a345_rec
            _a345_rec("send", self.bootstrap_room, chunk_id=self.chunk_id, n_kv=int(len(kv_indices)),
                      is_last=bool(is_last), had_start=self._transfer_start_time is not None)
        except Exception:
            pass
""")

# --- A: completion wait polls ---
replace_once(p_conn, """                while handles:
                    states = [self.agent.check_xfer_state(h) for h in handles]
                    if any(s == "ERR" for s in states):
                        raise RuntimeError(f"NIXL transfer encountered ERR room={room}")
                    if all(s == "DONE" for s in states):
                        break
                    time.sleep(0)
""", """                _a345_wait_t0 = time.perf_counter()
                _a345_last = _a345_wait_t0
                _a345_polls = 0
                _a345_maxgap = 0.0
                _a345_n_handles = len(handles)
                while handles:
                    _a345_now = time.perf_counter()
                    _a345_maxgap = max(_a345_maxgap, _a345_now - _a345_last)
                    _a345_last = _a345_now
                    _a345_polls += 1
                    states = [self.agent.check_xfer_state(h) for h in handles]
                    if any(s == "ERR" for s in states):
                        raise RuntimeError(f"NIXL transfer encountered ERR room={room}")
                    if all(s == "DONE" for s in states):
                        break
                    time.sleep(0)
                try:
                    from sglang.srt.disaggregation.aim345_ring import rec as _a345_rec
                    _a345_rec("wait", room, t_start=_a345_wait_t0, t_done=time.perf_counter(),
                              polls=_a345_polls, max_gap_ms=_a345_maxgap * 1000.0,
                              n_handles=_a345_n_handles, is_last=bool(kv_chunk.is_last),
                              n_kv=int(len(kv_chunk.prefill_kv_indices)))
                except Exception:
                    pass
""")

# --- G + B: phase timestamps of the overlap loop, and batch composition ---
replace_once(p_prefill, """            # Get the next batch to run
            batch = self.get_next_disagg_prefill_batch_to_run()
            self.cur_batch = batch

            # Launch the current batch
            _a345_run_batch_ms = -1.0
            if batch:
                if self.enable_staging:
                    self.maybe_prefetch_staging_for_batch(batch)
                _a345_t = _a345_pc()
                batch_result = self.run_batch(batch)
                _a345_run_batch_ms = (_a345_pc() - _a345_t) * 1000.0""", """            # Get the next batch to run
            _a345_t_get = _a345_pc()
            batch = self.get_next_disagg_prefill_batch_to_run()
            self.cur_batch = batch

            # Launch the current batch
            _a345_run_batch_ms = -1.0
            _a345_t_run = None
            if batch:
                if self.enable_staging:
                    self.maybe_prefetch_staging_for_batch(batch)
                _a345_t = _a345_pc()
                _a345_t_run = _a345_t
                try:
                    _a345_rec("batch", -1, tp=self.tp_rank,
                              reqs=[[r.rid, r.bootstrap_room, int(r.extend_input_len), int(len(r.prefix_indices)),
                                     r is getattr(self, "chunked_req", None)] for r in batch.reqs])
                except Exception:
                    pass
                batch_result = self.run_batch(batch)
                _a345_run_batch_ms = (_a345_pc() - _a345_t) * 1000.0""")
replace_once(p_prefill, """            _a345_pbr_ms = -1.0
            _a345_on_idle = False
            if self.last_batch:
                tmp_batch, tmp_result = self.result_queue.popleft()
                _a345_t = _a345_pc()
                self.process_batch_result(tmp_batch, tmp_result)
                _a345_pbr_ms = (_a345_pc() - _a345_t) * 1000.0""", """            _a345_pbr_ms = -1.0
            _a345_on_idle = False
            _a345_t_pbr = None
            if self.last_batch:
                tmp_batch, tmp_result = self.result_queue.popleft()
                _a345_t = _a345_pc()
                _a345_t_pbr = _a345_t
                self.process_batch_result(tmp_batch, tmp_result)
                _a345_pbr_ms = (_a345_pc() - _a345_t) * 1000.0""")
replace_once(p_prefill, """                        completions=len(_a345_done) if _a345_done else 0,
                        iter_total_ms=round((_a345_pc() - _a345_iter_t0) * 1000.0, 4),
                    )""", """                        completions=len(_a345_done) if _a345_done else 0,
                        iter_total_ms=round((_a345_pc() - _a345_iter_t0) * 1000.0, 4),
                        tp=self.tp_rank, t_iter=_a345_iter_t0, t_get=_a345_t_get,
                        t_run=_a345_t_run, t_pbr=_a345_t_pbr, t_poll=_a345_t,
                        done_rooms=[getattr(r, "bootstrap_room", None) for r in (_a345_done or [])],
                    )""")

# --- B control: do not admit a new request as a tiny chunk ---
replace_once(p_policy, """                if trunc_len <= 0:
                    return AddReqResult.OTHER

                # Chunked prefill
                req.set_extend_input_len(trunc_len)""", """                if trunc_len <= 0:
                    return AddReqResult.OTHER
                if trunc_len <= int(__import__("os").environ.get("AIM345_SUPPRESS_TINY_CHUNK", "0")):
                    return AddReqResult.OTHER

                # Chunked prefill
                req.set_extend_input_len(trunc_len)""")

print("aim345_oi3_patch applied OK")
