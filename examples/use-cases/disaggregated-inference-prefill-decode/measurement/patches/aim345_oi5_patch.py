"""OPEN-ITEMS-NODE5 instrumentation and env-gated controls, applied after aim345_oi1/oi3/oi3b.

A: AIM345_EFA_IFACES="devA,devB": each scheduler process sets FI_EFA_IFACE to the entry for its own
   tp_rank before anything in the process can initialize libfabric.
G: per-layer CPU timestamps of the prefill model's layer loop (TP rank 0 and 1), recorded only while
   a prefill run_batch is in progress, plus two controls:
   AIM345_G_SYNC=1      before launching a batch while the previous batch's result is still pending,
                        wait for that batch's copy_done event and record how long the wait took.
   AIM345_MIDPOLL=N     during a prefill forward, check the inflight queue every N layers (both TP
                        ranks run the same layers, so the collective inside the check stays matched).
"""
import pathlib
import sglang

root = pathlib.Path(sglang.__file__).parent / "srt"


def replace_once(path, old, new):
    t = path.read_text()
    assert t.count(old) == 1, f"anchor not found exactly once in {path}: {old[:80]!r}"
    path.write_text(t.replace(old, new))


# --- A: per-rank EFA interface, set at the top of every scheduler process ---
replace_once(root / "managers" / "scheduler.py", """    pipe_writer,
):
    # Load plugins so hooks can override Scheduler and its dependencies.
    load_plugins()""", """    pipe_writer,
):
    import os as _a5os
    _a5ifs = [x for x in _a5os.environ.get("AIM345_EFA_IFACES", "").split(",") if x]
    if _a5ifs:
        _a5os.environ["FI_EFA_IFACE"] = _a5ifs[tp_rank % len(_a5ifs)]
        print(f"[aim345-oi5] pid={_a5os.getpid()} tp_rank={tp_rank} gpu_id={gpu_id} FI_EFA_IFACE={_a5os.environ['FI_EFA_IFACE']}", flush=True)
    # Load plugins so hooks can override Scheduler and its dependencies.
    load_plugins()""")

# --- G: layer hook in the DeepSeek-V2 layer loop ---
ring = root / "disaggregation" / "aim345_ring.py"
t = ring.read_text()
if "LAYER_HOOK" not in t:
    ring.write_text(t + "\n\nLAYER_HOOK = None\n")
replace_once(root / "models" / "deepseek_v2.py", """                layer = self.layers[i]
                hidden_states, residual, topk_indices = layer(
                    positions,
                    hidden_states,
                    forward_batch,
                    residual,
                    zero_allocator,
                    gemm_output_zero_allocator,
                    llama_4_scaling,
                    prev_topk_indices=topk_indices,
                )""", """                layer = self.layers[i]
                from sglang.srt.disaggregation import aim345_ring as _a5ring
                if _a5ring.LAYER_HOOK is not None:
                    _a5ring.LAYER_HOOK(i)
                hidden_states, residual, topk_indices = layer(
                    positions,
                    hidden_states,
                    forward_batch,
                    residual,
                    zero_allocator,
                    gemm_output_zero_allocator,
                    llama_4_scaling,
                    prev_topk_indices=topk_indices,
                )""")

# --- G: sync control and the hook's owner, in the overlap loop (after oi3b) ---
p_prefill = root / "disaggregation" / "prefill.py"
replace_once(p_prefill, """                _a345_t = _a345_pc()
                _a345_t_run = _a345_t
                try:
                    _a345_rec("batch", -1, tp=self.tp_rank,""", """                import os as _a5os
                from sglang.srt.disaggregation import aim345_ring as _a5ring
                if _a5os.environ.get("AIM345_G_SYNC") == "1" and self.result_queue:
                    _a5cd = self.result_queue[-1][1].copy_done
                    if _a5cd is not None:
                        _a5t = _a345_pc()
                        _a5q = _a5cd.query()
                        _a5cd.synchronize()
                        _a345_rec("gsync", -1, tp=self.tp_rank, t0=_a5t, ms=(_a345_pc() - _a5t) * 1000.0, was_done=bool(_a5q))
                _a5mid = int(_a5os.environ.get("AIM345_MIDPOLL", "0") or 0)
                _a5bid = getattr(self, "_a5_bid", 0) + 1
                self._a5_bid = _a5bid
                _a5pend = bool(self.result_queue)

                def _a5hook(i, _s=self, _bid=_a5bid, _mid=_a5mid, _pend=_a5pend):
                    _a345_rec("layer", -1, tp=_s.tp_rank, bid=_bid, i=i, pend=_pend)
                    if _mid > 0 and i % _mid == _mid - 1:
                        _d = _s.process_disagg_prefill_inflight_queue()
                        _a345_rec("midpoll", -1, tp=_s.tp_rank, bid=_bid, i=i,
                                  done_rooms=[getattr(r, "bootstrap_room", None) for r in (_d or [])])

                _a5ring.LAYER_HOOK = _a5hook
                _a345_t = _a345_pc()
                _a345_t_run = _a345_t
                try:
                    _a345_rec("batch", -1, tp=self.tp_rank, bid=_a5bid,""")
replace_once(p_prefill, """                batch_result = self.run_batch(batch)
                _a345_run_batch_ms = (_a345_pc() - _a345_t) * 1000.0
                self.result_queue.append((batch.copy(), batch_result))
                if __import__("os").environ.get("AIM345_EXTRA_POLL") == "1":""", """                batch_result = self.run_batch(batch)
                _a5ring.LAYER_HOOK = None
                _a345_run_batch_ms = (_a345_pc() - _a345_t) * 1000.0
                _a345_rec("runb", -1, tp=self.tp_rank, bid=_a5bid, t0=_a345_t, ms=_a345_run_batch_ms, pend=_a5pend)
                self.result_queue.append((batch.copy(), batch_result))
                if __import__("os").environ.get("AIM345_EXTRA_POLL") == "1":""")
print("aim345_oi5_patch applied OK")
