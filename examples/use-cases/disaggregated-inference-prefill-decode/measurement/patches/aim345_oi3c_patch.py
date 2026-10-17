"""G: sample the scheduler thread's Python stack while it is inside a long run_batch.
A daemon thread samples every 5 ms; it records only once the current run_batch has lasted 100 ms.
Applied after aim345_oi1/oi3/oi3b patches."""
import pathlib
import sglang

p = pathlib.Path(sglang.__file__).parent / "srt" / "disaggregation" / "prefill.py"
t = p.read_text()
old_imp = """from time import perf_counter as _a345_pc
from sglang.srt.disaggregation.aim345_ring import rec as _a345_rec"""
new_imp = """from time import perf_counter as _a345_pc
from sglang.srt.disaggregation.aim345_ring import rec as _a345_rec
import sys as _a345_sys
import threading as _a345_threading
_a345_run = {"t": None, "tp": None, "ident": None, "started": False}


def _a345_sampler():
    import time as _t
    while True:
        _t.sleep(0.005)
        t0 = _a345_run["t"]
        if t0 is None or _a345_pc() - t0 < 0.1:
            continue
        f = _a345_sys._current_frames().get(_a345_run["ident"])
        frames = []
        while f is not None and len(frames) < 18:
            c = f.f_code
            frames.append(f"{c.co_filename.rsplit('/', 2)[-1]}:{f.f_lineno}:{c.co_name}")
            f = f.f_back
        try:
            _a345_rec("stack", -1, tp=_a345_run["tp"], t_run=t0, frames=frames)
        except Exception:
            pass


def _a345_start_sampler(tp):
    if not _a345_run["started"]:
        _a345_run.update(started=True, tp=tp, ident=_a345_threading.get_ident())
        _a345_threading.Thread(target=_a345_sampler, name="aim345-stack", daemon=True).start()"""
assert t.count(old_imp) == 1
t = t.replace(old_imp, new_imp)
old = """                batch_result = self.run_batch(batch)
                _a345_run_batch_ms = (_a345_pc() - _a345_t) * 1000.0
                self.result_queue.append((batch.copy(), batch_result))
                if __import__("os").environ.get("AIM345_EXTRA_POLL") == "1":"""
new = """                _a345_start_sampler(self.tp_rank)
                _a345_run["t"] = _a345_t
                batch_result = self.run_batch(batch)
                _a345_run["t"] = None
                _a345_run_batch_ms = (_a345_pc() - _a345_t) * 1000.0
                self.result_queue.append((batch.copy(), batch_result))
                if __import__("os").environ.get("AIM345_EXTRA_POLL") == "1":"""
assert t.count(old) == 1
t = t.replace(old, new)
p.write_text(t)
print("aim345_oi3c_patch applied OK")
