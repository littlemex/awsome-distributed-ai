"""Disposable admission-timestamp patch for OPEN-ITEMS-NODE9 (applied on the pinned workshop image only).
Appends method wrappers; no line of the original code is edited. Records through aim345_ring.rec() on TP rank 0:
prefill: "recv" (every non-empty receive and every receive after a loop iteration longer than 5 ms: wall time, that iteration's duration, the queue state the health
admission rule reads, each request's rid/room/health flag), "bsadd" (bootstrap-queue entry), "bspop" (requests
released from the bootstrap queue), "pbatch" (each prefill batch formed: rid, room, extend, prefix, chunked);
decode: "dadd" (prealloc-queue entry), "dpre" (requests whose decode KV was preallocated and handshake sent).
"""
import pathlib, shutil, sglang
ROOT = pathlib.Path(sglang.__file__).parent / "srt"
shutil.copy("/tmp/aim345_ring.py", ROOT / "disaggregation" / "aim345_ring.py")

def append(path, text):
    p = ROOT / path
    src = p.read_text()
    assert "_a9_" not in src, f"{path}: already patched"
    p.write_text(src + text)

append("disaggregation/prefill.py", '''

# ---- aim345 oi9 (investigation only) ----
import time as _a9_time
from sglang.srt.disaggregation.aim345_ring import rec as _a9_rec
def _a9_req(r):
    return [str(r.rid)[:48], int(getattr(r, "bootstrap_room", -1) or -1)]
_a9_bs_add = PrefillBootstrapQueue.add
def _a9_bs_add_w(self, req, num_kv_heads):
    out = _a9_bs_add(self, req, num_kv_heads)
    if self.tp_rank == 0:
        _a9_rec("bsadd", -1, wall=_a9_time.time(), r=_a9_req(req), qlen=len(self.queue))
    return out
PrefillBootstrapQueue.add = _a9_bs_add_w
_a9_bs_pop = PrefillBootstrapQueue.pop_bootstrapped
def _a9_bs_pop_w(self, *a, **k):
    out = _a9_bs_pop(self, *a, **k)
    if self.tp_rank == 0:
        reqs = out[0] if isinstance(out, tuple) else out
        if reqs:
            _a9_rec("bspop", -1, wall=_a9_time.time(), rs=[_a9_req(r) for r in reqs], left=len(self.queue))
    return out
PrefillBootstrapQueue.pop_bootstrapped = _a9_bs_pop_w
_a9_next = SchedulerDisaggregationPrefillMixin.get_next_disagg_prefill_batch_to_run
def _a9_next_w(self, *a, **k):
    b = _a9_next(self, *a, **k)
    if b is not None and self.tp_rank == 0:
        ck = getattr(self, "chunked_req", None)
        _a9_rec("pbatch", -1, wall=_a9_time.time(), fct=self.forward_ct,
                rs=[_a9_req(r) + [int(r.extend_input_len), int(len(r.prefix_indices)), r is ck] for r in b.reqs])
    return b
SchedulerDisaggregationPrefillMixin.get_next_disagg_prefill_batch_to_run = _a9_next_w
''')

append("disaggregation/decode.py", '''

# ---- aim345 oi9 (investigation only) ----
import time as _a9_time
from sglang.srt.disaggregation.aim345_ring import rec as _a9_rec
_a9_d_add = DecodePreallocQueue.add
def _a9_d_add_w(self, req, is_retracted=False):
    out = _a9_d_add(self, req, is_retracted=is_retracted)
    if self.tp_rank == 0:
        _a9_rec("dadd", -1, wall=_a9_time.time(), r=[str(req.rid)[:48], int(getattr(req, "bootstrap_room", -1) or -1)],
                retr=bool(is_retracted), qlen=len(self.queue))
    return out
DecodePreallocQueue.add = _a9_d_add_w
_a9_d_pre = DecodePreallocQueue.pop_preallocated
def _a9_d_pre_w(self, *a, **k):
    out = _a9_d_pre(self, *a, **k)
    if self.tp_rank == 0 and out and out[0]:
        _a9_rec("dpre", -1, wall=_a9_time.time(),
                rs=[[str(d.req.rid)[:48], int(getattr(d.req, "bootstrap_room", -1) or -1)] for d in out[0]],
                left=len(self.queue), running=len(self.scheduler.running_batch.reqs))
    return out
DecodePreallocQueue.pop_preallocated = _a9_d_pre_w
''')

append("managers/scheduler.py", '''

# ---- aim345 oi9 (investigation only) ----
import time as _a9_time
from sglang.srt.disaggregation.aim345_ring import rec as _a9_rec
_a9_recv = Scheduler.recv_requests
def _a9_recv_w(self, *a, **k):
    now = _a9_time.perf_counter()
    prev = getattr(self, "_a9_prev_recv", None)
    self._a9_prev_recv = now
    out = _a9_recv(self, *a, **k)
    if self.tp_rank == 0 and (out or (prev is not None and now - prev > 0.005)):
        try:
            lb = self.last_batch
            st = dict(run=len(self.running_batch.reqs), chunked=self.chunked_req is not None,
                      last=(lb is not None and not lb.is_empty()), rq=len(getattr(self, "result_queue", ()) or ()),
                      wait=len(self.waiting_queue))
            bq = getattr(self, "disagg_prefill_bootstrap_queue", None)
            if bq is not None:
                st["boot"] = len(bq.queue); st["infl"] = len(self.disagg_prefill_inflight_queue)
            pq = getattr(self, "disagg_decode_prealloc_queue", None)
            if pq is not None:
                st["pre"] = len(pq.queue); st["xfer"] = len(self.disagg_decode_transfer_queue.queue)
            rs = []
            for x in out:
                rid = getattr(x, "rid", None)
                if rid is None:
                    continue
                rs.append([str(rid)[:48], int(getattr(x, "bootstrap_room", -1) or -1)])
            _a9_rec("recv", -1, wall=_a9_time.time(), dt=(None if prev is None else now - prev),
                    idle_h=bool(self.is_fully_idle(for_health_check=True)), st=st, rs=rs)
        except Exception as e:
            _a9_rec("recv_err", -1, err=repr(e)[:80])
    return out
Scheduler.recv_requests = _a9_recv_w
''')
print("oi9 patch applied")
