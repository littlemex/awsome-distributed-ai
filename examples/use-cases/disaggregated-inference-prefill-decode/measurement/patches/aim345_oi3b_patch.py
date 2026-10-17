"""G control (env-gated): AIM345_EXTRA_POLL=1 checks the inflight queue once more right after the
current batch is launched, before the previous batch's result is processed (which blocks on the GPU).
Applied after aim345_oi1_patch.py and aim345_oi3_patch.py."""
import pathlib
import sglang

p = pathlib.Path(sglang.__file__).parent / "srt" / "disaggregation" / "prefill.py"
old = """                batch_result = self.run_batch(batch)
                _a345_run_batch_ms = (_a345_pc() - _a345_t) * 1000.0
                self.result_queue.append((batch.copy(), batch_result))"""
new = """                batch_result = self.run_batch(batch)
                _a345_run_batch_ms = (_a345_pc() - _a345_t) * 1000.0
                self.result_queue.append((batch.copy(), batch_result))
                if __import__("os").environ.get("AIM345_EXTRA_POLL") == "1":
                    _a345_xp = self.process_disagg_prefill_inflight_queue()
                    try:
                        _a345_rec("extra_poll", -1, tp=self.tp_rank, done_rooms=[getattr(r, "bootstrap_room", None) for r in (_a345_xp or [])])
                    except Exception:
                        pass"""
t = p.read_text()
assert t.count(old) == 1, "overlap loop block not found exactly once"
p.write_text(t.replace(old, new))
print("aim345_oi3b_patch applied OK")
