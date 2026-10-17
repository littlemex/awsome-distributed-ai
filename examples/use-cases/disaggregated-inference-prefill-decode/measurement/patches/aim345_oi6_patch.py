"""OPEN-ITEMS-NODE5 round 6 (applied after oi1, oi3, oi3b, oi5): a CUPTI window on the prefill TP0 scheduler.
   AIM345_PROF=S:N  on TP0 of the prefill scheduler, start torch.profiler (CPU + CUDA activities, i.e. CUPTI
                    runtime/driver API calls with their CPU durations and the GPU kernels) when batch S starts,
                    stop before batch S+N, export a Chrome trace to /tmp/aim345_prof_<pid>.json. While it runs,
                    each run_batch and each result processing is wrapped in a record_function span named
                    aim345_runb_<bid> / aim345_pbr_<bid> so the trace can be joined to the ring's batch ids.
"""
import pathlib
import sglang

root = pathlib.Path(sglang.__file__).parent / "srt"


def replace_once(path, old, new):
    t = path.read_text()
    assert t.count(old) == 1, f"anchor not found exactly once in {path}: {old[:80]!r}"
    path.write_text(t.replace(old, new))


ring = root / "disaggregation" / "aim345_ring.py"
t = ring.read_text()
if "def prof_step" not in t:
    ring.write_text(t + '''

PROF = None
PROF_DONE = False


def prof_step(tp_rank, bid):
    """Start or stop the CUPTI window on TP0 (AIM345_PROF=S:N)."""
    global PROF, PROF_DONE
    spec = os.environ.get("AIM345_PROF", "")
    if not spec or tp_rank != 0 or PROF_DONE:
        return
    s, n = (int(x) for x in spec.split(":"))
    if PROF is None and bid == s:
        import torch.profiler as _tp
        PROF = _tp.profile(activities=[_tp.ProfilerActivity.CPU, _tp.ProfilerActivity.CUDA])
        PROF.__enter__()
        rec("prof_start", -1, tp=tp_rank, bid=bid, epoch=time.time())
    elif PROF is not None and bid >= s + n:
        p, PROF = PROF, None
        p.__exit__(None, None, None)
        path = f"/tmp/aim345_prof_{os.getpid()}.json"
        p.export_chrome_trace(path)
        PROF_DONE = True
        rec("prof_stop", -1, tp=tp_rank, bid=bid, path=path, epoch=time.time())


def span(name):
    if PROF is None:
        import contextlib
        return contextlib.nullcontext()
    import torch.profiler as _tp
    return _tp.record_function(name)
''')

p_prefill = root / "disaggregation" / "prefill.py"
replace_once(p_prefill, """                batch_result = self.run_batch(batch)
                _a5ring.LAYER_HOOK = None""", """                _a5ring.prof_step(self.tp_rank, _a5bid)
                with _a5ring.span(f"aim345_runb_{_a5bid}"):
                    batch_result = self.run_batch(batch)
                _a5ring.LAYER_HOOK = None""")
replace_once(p_prefill, """                _a345_t_pbr = _a345_t
                self.process_batch_result(tmp_batch, tmp_result)""", """                _a345_t_pbr = _a345_t
                from sglang.srt.disaggregation import aim345_ring as _a6ring
                with _a6ring.span(f"aim345_pbr_{getattr(self, '_a5_bid', 0)}"):
                    self.process_batch_result(tmp_batch, tmp_result)""")
print("aim345_oi6_patch applied OK")
