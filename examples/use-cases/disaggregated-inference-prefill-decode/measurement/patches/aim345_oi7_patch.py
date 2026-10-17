"""Disposable debug patch for OPEN-ITEMS-NODE7 (item F: the /health 1-token request chain). Applied on the
pinned engine image only (no earlier patch). Records, through aim345_ring.rec():
- "pbatch": every extend batch launched on TP0 (prefill server): forward_ct, wall time, and per request the rid,
  extend length, prefix length, whether it is the scheduler's chunked request, whether it is a health check;
- "bgpu": that batch's GPU time, from CUDA events recorded on the forward stream immediately before and after
  forward_batch_generation (read later with query(), never synchronizing);
- "dispatch": the attention method chosen at layer 0 for each extend forward on TP0, with the summed prefix.
And one env-gated control: AIM345_FORCE_MLA=1 makes every extend forward take the MLA path (prefix 0 included).
Every insertion is assert-guarded so a drifted source fails the image build.
"""
import pathlib
import shutil
import sglang

ROOT = pathlib.Path(sglang.__file__).parent / "srt"


def replace_once(path, old, new):
    text = path.read_text()
    n = text.count(old)
    assert n == 1, f"{path}: expected 1 occurrence, found {n}:\n{old!r}"
    path.write_text(text.replace(old, new, 1))


shutil.copy("/tmp/aim345_ring.py", ROOT / "disaggregation" / "aim345_ring.py")

p = ROOT / "models" / "deepseek_v2.py"
replace_once(p, """        handler = AttentionBackendRegistry.get_handler(attention_backend)
        return handler(self, forward_batch)""", """        handler = AttentionBackendRegistry.get_handler(attention_backend)
        _a7m = handler(self, forward_batch)
        if forward_batch.forward_mode.is_extend_without_speculative():
            if _a7_os.environ.get("AIM345_FORCE_MLA") == "1":
                _a7m = AttnForwardMethod.MLA
            if self.layer_id == 0:
                try:
                    from sglang.srt.distributed import get_tensor_model_parallel_rank as _a7tpr
                    if _a7tpr() != 0:
                        raise StopIteration
                    from sglang.srt.disaggregation.aim345_ring import rec as _a7rec
                    _a7rec("dispatch", -1, method=_a7m.name, cached=int(sum(forward_batch.extend_prefix_lens_cpu)),
                           bs=int(forward_batch.batch_size))
                except Exception:
                    pass
        return _a7m""")
replace_once(p, """from sglang.srt.models.deepseek_common.attention_backend_handler import (
    AttentionBackendRegistry,
)""", """from sglang.srt.models.deepseek_common.attention_backend_handler import (
    AttentionBackendRegistry,
)
import os as _a7_os""")

p = ROOT / "managers" / "scheduler.py"
replace_once(p, """                with self.forward_stream_ctx:
                    self.forward_stream.wait_stream(self.schedule_stream)
                    self.future_map.resolve_future(model_worker_batch)
                    batch_result = self.model_worker.forward_batch_generation(
                        model_worker_batch
                        # here pp is not compatible with overlap
                    )""", """                if getattr(self, "_a7_pending", None):
                    try:
                        from sglang.srt.disaggregation.aim345_ring import rec as _a7rec
                        _a7keep = []
                        for _a7f, _a7a, _a7b in self._a7_pending:
                            if _a7b.query():
                                _a7rec("bgpu", -1, fct=_a7f, ms=float(_a7a.elapsed_time(_a7b)))
                            else:
                                _a7keep.append((_a7f, _a7a, _a7b))
                        self._a7_pending = _a7keep
                    except Exception:
                        self._a7_pending = []
                _a7_on = batch.forward_mode.is_extend() and self.tp_rank == 0
                import os as _a7_os
                if _a7_on:
                    try:
                        from sglang.srt.disaggregation.aim345_ring import rec as _a7rec
                        _a7ck = getattr(self, "chunked_req", None)
                        _a7rec("pbatch", -1, fct=self.forward_ct, wall=time.time(),
                               reqs=[[str(r.rid)[:40], int(r.extend_input_len), int(len(r.prefix_indices)),
                                      r is _a7ck, str(r.rid).startswith("HEALTH_CHECK")] for r in batch.reqs])
                    except Exception:
                        pass
                with self.forward_stream_ctx:
                    self.forward_stream.wait_stream(self.schedule_stream)
                    self.future_map.resolve_future(model_worker_batch)
                    if _a7_on:
                        _a7s = torch.cuda.Event(enable_timing=True)
                        _a7e = torch.cuda.Event(enable_timing=True)
                        _a7s.record()
                    _a7prof = None
                    if _a7_on and _a7_os.environ.get("AIM345_PROF") == "1":
                        _a7pre = [int(len(r.prefix_indices)) for r in batch.reqs]
                        _a7ext = sum(int(r.extend_input_len) for r in batch.reqs)
                        _a7k = None
                        if _a7ext >= 4096:
                            if any(x > 0 for x in _a7pre) and any(x == 0 for x in _a7pre):
                                _a7k = "RP"
                            elif len(_a7pre) == 1 and _a7pre[0] == 0:
                                _a7k = "P"
                            elif len(_a7pre) >= 2 and all(x == 0 for x in _a7pre):
                                _a7k = "HP"
                        if not hasattr(self, "_a7_profcnt"):
                            self._a7_profcnt = {}
                        if _a7k and self._a7_profcnt.get(_a7k, 0) < 3 and self.forward_ct > 20:
                            self._a7_profcnt[_a7k] = self._a7_profcnt.get(_a7k, 0) + 1
                            torch.cuda.synchronize()
                            _a7prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA])
                            _a7prof.__enter__()
                    batch_result = self.model_worker.forward_batch_generation(
                        model_worker_batch
                        # here pp is not compatible with overlap
                    )
                    if _a7prof is not None:
                        torch.cuda.synchronize()
                        _a7prof.__exit__(None, None, None)
                        try:
                            with open(f"/tmp/aim345_prof_{self.forward_ct}_{_a7k}.txt", "w") as _a7f:
                                _a7f.write(__import__("json").dumps({"fct": self.forward_ct, "kind": _a7k,
                                                       "reqs": [[int(r.extend_input_len), int(len(r.prefix_indices))] for r in batch.reqs]}) + "\\n")
                                _a7f.write(_a7prof.key_averages().table(sort_by="cuda_time_total", row_limit=60))
                        except Exception as _a7x:
                            pass
                    if _a7_on:
                        _a7e.record()
                        if not hasattr(self, "_a7_pending"):
                            self._a7_pending = []
                        self._a7_pending.append((self.forward_ct, _a7s, _a7e))""")
replace_once(p, """        for recv_req in recv_reqs:
            # Skip health check when server is busy — ongoing requests already carry health info.
            if is_health_check_generate_req(recv_req) and not self.is_fully_idle(
                for_health_check=True
            ):
                self.return_health_check_ipcs.append(
                    getattr(recv_req, "http_worker_ipc", None)
                )
                continue
""", """        for _a7i, recv_req in enumerate(recv_reqs):
            # Skip health check when server is busy — ongoing requests already carry health info.
            if is_health_check_generate_req(recv_req) and not self.is_fully_idle(
                for_health_check=True
            ):
                if self.tp_rank == 0:
                    try:
                        from sglang.srt.disaggregation.aim345_ring import rec as _a7rec
                        _a7rec("hskip", -1, rid=str(recv_req.rid)[:40], wall=time.time(), waiting=len(self.waiting_queue),
                               chunked=self.chunked_req is not None, n_recv=len(recv_reqs), pos=_a7i)
                    except Exception:
                        pass
                self.return_health_check_ipcs.append(
                    getattr(recv_req, "http_worker_ipc", None)
                )
                continue
            if self.tp_rank == 0 and is_health_check_generate_req(recv_req):
                try:
                    from sglang.srt.disaggregation.aim345_ring import rec as _a7rec
                    _a7rec("hadmit", -1, rid=str(recv_req.rid)[:40], wall=time.time(), waiting=len(self.waiting_queue),
                           n_recv=len(recv_reqs), pos=_a7i,
                           later_gen=sum(1 for _x in recv_reqs[_a7i + 1:] if hasattr(_x, "input_ids") and not is_health_check_generate_req(_x)))
                except Exception:
                    pass
""")
print("oi7 patch applied")
