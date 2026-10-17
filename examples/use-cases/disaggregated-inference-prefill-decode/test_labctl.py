"""CPU-only checks for labctl's run-id/lease/resume contract and its shard dispatch/merge path.

No cluster is used: shard_write/shard_read are monkeypatched to an in-memory per-shard store, and
one real HTTP test server (shared with test_lab.py's pattern) stands in for the router so merge()
exercises benchmark.run_rate end to end, including cancellation and per-chunk ITL recording.
"""
import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import atomic
import labctl
import runner_agent
from benchmark import run_rate, summarize
from traffic import tokenizer


class StreamHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for count in (0, 1, 2, 4):
            message = {"text": "ready" if count else "", "meta_info": {"completion_tokens": count, "cached_tokens": 0}}
            self.wfile.write(("data: " + json.dumps(message) + "\n\n").encode())
            self.wfile.flush()
            time.sleep(.01)
        if body["input_ids"] != [999]:
            self.wfile.write(b"data: [DONE]\n\n")


class AtomicTests(unittest.TestCase):
    def test_write_then_read_round_trips_and_leaves_no_tmp_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "sub" / "run_status.json"
            atomic.atomic_write_json(path, {"status": "RUNNING"})
            self.assertEqual(atomic.read_json(path), {"status": "RUNNING"})
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_missing_file_reads_as_none(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(atomic.read_json(Path(d) / "absent.json"))

    def test_false_rename_failure_is_not_raised_once_content_matches(self):
        import errno
        import os
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "f.json"
            data = json.dumps({"ok": True})
            real_replace = os.replace

            def flaky(src, dst):
                real_replace(src, dst)  # the rename actually succeeds...
                raise OSError(errno.ENOENT, "retransmitted RENAME reported failure")  # ...but is reported as failed
            with patch("atomic.os.replace", flaky):
                atomic.atomic_write(path, data)  # must not raise: a readback shows the content already landed
            self.assertEqual(path.read_text(), data)

    def test_genuine_replace_failure_still_raises(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "f.json"

            def always_fails(src, dst):
                raise OSError(5, "permission denied")
            with patch("atomic.os.replace", always_fails):
                with self.assertRaises(OSError):
                    atomic.atomic_write(path, "{}")


class LeaseAndStateMachineTests(unittest.TestCase):
    def test_acquire_lease_on_a_fresh_run_directory_succeeds(self):
        with tempfile.TemporaryDirectory() as d:
            hdir = Path(d)
            status = labctl.acquire_lease(hdir, "run-x", "seg-1")
            self.assertEqual(status["status"], "RUNNING")
            self.assertEqual(status["owner_segment_id"], "seg-1")

    def test_acquire_lease_refuses_a_live_owner(self):
        with tempfile.TemporaryDirectory() as d:
            hdir = Path(d)
            atomic.atomic_write_json(hdir / "run_status.json", {"status": "RUNNING", "owner_segment_id": "seg-old", "lease_expires_at": time.time() + 60})
            with self.assertRaisesRegex(SystemExit, "active \\(owner lease held\\)"):
                labctl.acquire_lease(hdir, "run-x", "seg-new")

    def test_acquire_lease_allows_an_expired_lease(self):
        with tempfile.TemporaryDirectory() as d:
            hdir = Path(d)
            atomic.atomic_write_json(hdir / "run_status.json", {"status": "RUNNING", "owner_segment_id": "seg-old", "lease_expires_at": time.time() - 1})
            status = labctl.acquire_lease(hdir, "run-x", "seg-new")
            self.assertEqual(status["owner_segment_id"], "seg-new")

    def test_acquire_lease_refuses_complete_and_failed_by_name(self):
        with tempfile.TemporaryDirectory() as d:
            hdir = Path(d)
            atomic.atomic_write_json(hdir / "run_status.json", {"status": "COMPLETE"})
            with self.assertRaisesRegex(SystemExit, "COMPLETE"):
                labctl.acquire_lease(hdir, "run-x", "seg-new")
            atomic.atomic_write_json(hdir / "run_status.json", {"status": "FAILED"})
            with self.assertRaisesRegex(SystemExit, "cannot be resumed"):
                labctl.acquire_lease(hdir, "run-x", "seg-new")

    def test_acquire_lease_refuses_an_unfinished_creation(self):
        with tempfile.TemporaryDirectory() as d:
            hdir = Path(d)
            atomic.atomic_write_json(hdir / "run_status.json", {"status": "CREATING"})
            with self.assertRaisesRegex(SystemExit, "creation did not finish"):
                labctl.acquire_lease(hdir, "run-x", "seg-new")

    def test_a_fresh_run_does_not_refuse_its_own_creating_marker(self):
        """Regression: a brand-new run must reach RUNNING in the same process that wrote CREATING,
        without ever calling acquire_lease() on that marker -- acquire_lease() unconditionally
        refuses CREATING by name, so calling it here would make every fresh `labctl ramp` fail."""
        with tempfile.TemporaryDirectory() as d:
            hdir = Path(d)
            atomic.atomic_write_json(hdir / "run_status.json", {"status": "CREATING", "owner_segment_id": None, "lease_expires_at": 0})
            atomic.atomic_write_json(hdir / "manifest.json", {"ok": True})
            atomic.atomic_write_json(hdir / "run_status.json", {"status": "RUNNING", "owner_segment_id": "seg-1", "lease_expires_at": time.time() + labctl.LEASE_TTL_S})
            self.assertEqual(labctl.read_status(hdir)["status"], "RUNNING")
            # A genuinely abandoned CREATING (a later, different invocation) is still refused.
            atomic.atomic_write_json(hdir / "run_status.json", {"status": "CREATING", "owner_segment_id": None, "lease_expires_at": 0})
            with self.assertRaisesRegex(SystemExit, "creation did not finish"):
                labctl.acquire_lease(hdir, "run-x", "seg-2")

    def test_resume_cursor_skips_terminal_phases_in_plan_order(self):
        steps = labctl.plan_steps([0.5, 1, 2], skip_warmup=False)
        self.assertEqual(steps, [(0.5, "warmup"), (0.5, "measured"), (1, "warmup"), (1, "measured"), (2, "warmup"), (2, "measured")])
        state = {labctl.state_key(0.5, "warmup"): {"status": "merged", "job_name": "j1"},
                 labctl.state_key(0.5, "measured"): {"status": "merged", "job_name": "j2"},
                 labctl.state_key(1, "warmup"): {"status": "interrupted", "job_name": "j3"}}
        self.assertEqual(labctl.resume_cursor(state, steps), (1, "warmup"))

    def test_resume_cursor_is_none_once_every_phase_is_terminal(self):
        steps = labctl.plan_steps([1], skip_warmup=True)
        state = {labctl.state_key(1, "measured"): {"status": "stopped_on_failure_fraction", "job_name": "j1"}}
        self.assertIsNone(labctl.resume_cursor(state, steps))

    def test_resume_cursor_refuses_to_run_rates_planned_after_a_deliberate_stop(self):
        steps = labctl.plan_steps([1, 2], skip_warmup=True)
        state = {labctl.state_key(1, "measured"): {"status": "stopped_on_failure_fraction", "job_name": "j1"}}
        # rate 2's phase is still "pending" (absent), but it was never meant to run once rate 1 stopped.
        self.assertIsNone(labctl.resume_cursor(state, steps))

    def test_compare_manifest_refuses_a_named_field_mismatch(self):
        stored = {k: 1 for k in labctl.MANIFEST_FIELDS}
        stored["rates"] = [0.25, 0.5, 1, 2, 4]
        built = dict(stored, rates=[0.25, 0.5, 1, 2])
        with self.assertRaisesRegex(SystemExit, "stored rates differs"):
            labctl.compare_manifest("run-x", stored, built)

    def test_compare_manifest_accepts_an_identical_plan(self):
        stored = {k: 1 for k in labctl.MANIFEST_FIELDS}
        labctl.compare_manifest("run-x", stored, dict(stored))  # must not raise

    def test_find_complete_matching_only_matches_a_complete_run_with_an_equal_plan(self):
        with tempfile.TemporaryDirectory() as d:
            output = Path(d)
            built = {k: 1 for k in labctl.MANIFEST_FIELDS}
            hdir = labctl.harness_dir(output, "run-a")
            atomic.atomic_write_json(hdir / "manifest.json", built)
            atomic.atomic_write_json(hdir / "run_status.json", {"status": "INTERRUPTED"})
            self.assertIsNone(labctl.find_complete_matching(output, built))  # not COMPLETE yet
            atomic.atomic_write_json(hdir / "run_status.json", {"status": "COMPLETE"})
            self.assertEqual(labctl.find_complete_matching(output, built), "run-a")
            self.assertIsNone(labctl.find_complete_matching(output, dict(built, seed=2)))  # different plan


FAKE_MANIFEST = {"architecture": "unified", "namespace": "aim345-u", "shard_count": 2, "client_path": "in-cluster-v1-local-lease",
                 "duration_s": 1, "timeout_s": 10, "ttft_slo_ms": 5000, "tpot_slo_ms": 5000, "attainment_fraction": .5,
                 "context_length_tokens": 32768, "max_client_lag_ms": 1000, "stop_failure_fraction": .5, "skip_warmup": True,
                 "seed": 345, "max_tasks": 64, "tpot_metric": "mean", "gpus": 2, "instance_type": "g7.48xlarge",
                 "region": "us-west-2", "evidence_scope": "mechanism-validation", "rates": [4.0]}


class DispatchMergeTests(unittest.TestCase):
    """Fakes shard transport (no cluster) but runs the real benchmark.run_rate against a real
    HTTP server, so the shard side of this test is the same code runner_agent.py executes."""

    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), StreamHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.tok = tokenizer()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        self.store = {}  # (shard, path) -> content, stands in for each shard's own local volume
        patch.object(labctl, "shard_write", side_effect=lambda c, k, path, content: self.store.__setitem__((k, path), content)).start()
        patch.object(labctl, "shard_read", side_effect=lambda c, k, path: self.store.get((k, path))).start()
        self.addCleanup(patch.stopall)

    def _traffic(self):
        shape = {"name": "B-test", "input_tokens": 128, "output_tokens": 4, "document_text": "Order fulfillment incident record for the fictional warehouse service."}
        return {"model": "deepseek-ai/DeepSeek-V2-Lite-Chat", "revision": "85864749cd611b4353ce1decdb286193298f64c7",
                "shape": shape, "tasks": [{"task_id": "t0", "calls": 1}, {"task_id": "t1", "calls": 1}]}

    def _run_shard(self, c, manifest, job_name, shard_index):
        job = json.loads(self.store[(shard_index, f"/local/jobs/{job_name}.job.json")])
        from types import SimpleNamespace
        args = SimpleNamespace(**job["args"], shard_count=job["shard_count"], gpus=job["gpus"],
                               url=f"http://127.0.0.1:{self.server.server_port}", architecture=job["architecture"],
                               instance_type=job["instance_type"], region=job["region"], evidence_scope=job["evidence_scope"])
        result = asyncio.run(run_rate(job["traffic"], args, job["rate"], job["phase"], self.tok))
        summary = dict(result["summary"], authoritative=False, shard_offered_tasks_per_s=job["rate"], shard_index=shard_index)
        outcome = {"metadata": result["metadata"], "summary": summary, "requests": result["requests"],
                  "segment_id": job["segment_id"], "shard_actual_start_wall": time.time(), "cancelled": False}
        self.store[(shard_index, f"/local/results/{job_name}.result.json")] = json.dumps(outcome)
        self.store[(shard_index, f"/local/acks/{job_name}.ack.json")] = json.dumps({"status": "done", "segment_id": job["segment_id"]})

    def test_dispatch_then_merge_sums_both_shards_requests(self):
        c = {"namespace": "aim345-u", "context": "test"}
        job_name = labctl.dispatch(c, FAKE_MANIFEST, "seg-1", 4.0, "measured", self._traffic())
        for k in range(2):
            self._run_shard(c, FAKE_MANIFEST, job_name, k)
        acked = {k: json.loads(self.store[(k, f"/local/acks/{job_name}.ack.json")]) for k in range(2)}
        merged = labctl.merge(c, FAKE_MANIFEST, "seg-1", 4.0, "measured", job_name, acked, 2)
        shard_planned = sum(json.loads(self.store[(k, f"/local/results/{job_name}.result.json")])["summary"]["planned_calls"] for k in range(2))
        self.assertEqual(len(merged["requests"]), shard_planned)  # union of both shards' requests, nothing dropped
        self.assertGreater(len(merged["requests"]), 0)
        self.assertEqual(merged["summary"]["offered_tasks_per_s"], 4.0)  # the planned total, not a shard's rate/K
        self.assertEqual(merged["metadata"]["arguments"]["ramp_id"], "seg-1")
        self.assertEqual(merged["metadata"]["arguments"]["run_id"], "selfcheck")  # default when run_id is omitted
        self.assertEqual(merged["metadata"]["arguments"]["shard_count"], 2)
        self.assertTrue(merged["summary"]["client_valid"])

    def test_merge_refuses_when_a_shard_result_file_is_missing(self):
        c = {"namespace": "aim345-u", "context": "test"}
        job_name = labctl.dispatch(c, FAKE_MANIFEST, "seg-2", 4.0, "measured", self._traffic())
        self._run_shard(c, FAKE_MANIFEST, job_name, 0)  # shard 1 never writes a result
        acked = {0: json.loads(self.store[(0, f"/local/acks/{job_name}.ack.json")]), 1: {"status": "done", "segment_id": "seg-2"}}
        with self.assertRaisesRegex(RuntimeError, "result file is missing"):
            labctl.merge(c, FAKE_MANIFEST, "seg-2", 4.0, "measured", job_name, acked, 2)

    def test_merge_marks_start_skew_above_tolerance_as_client_invalid(self):
        c = {"namespace": "aim345-u", "context": "test"}
        job_name = labctl.dispatch(c, FAKE_MANIFEST, "seg-3", 4.0, "measured", self._traffic())
        for k in range(2):
            self._run_shard(c, FAKE_MANIFEST, job_name, k)
        outcome = json.loads(self.store[(1, f"/local/results/{job_name}.result.json")])
        outcome["shard_actual_start_wall"] += 1.0  # 1s skew, well above the 500ms tolerance
        self.store[(1, f"/local/results/{job_name}.result.json")] = json.dumps(outcome)
        acked = {k: json.loads(self.store[(k, f"/local/acks/{job_name}.ack.json")]) for k in range(2)}
        merged = labctl.merge(c, FAKE_MANIFEST, "seg-3", 4.0, "measured", job_name, acked, 2)
        self.assertFalse(merged["summary"]["client_valid"])
        self.assertFalse(merged["summary"]["meets_joint_slo"])

    def test_merge_start_skew_is_reproducible_by_rescore(self):
        """Regression: a skew-invalidated merge must be byte-for-byte reproducible by calling
        summarize() fresh on the stored requests[] (rescore.check_reproduces does exactly this),
        not just by labctl's own in-process override of the summary it already computed."""
        c = {"namespace": "aim345-u", "context": "test"}
        job_name = labctl.dispatch(c, FAKE_MANIFEST, "seg-5", 4.0, "measured", self._traffic())
        for k in range(2):
            self._run_shard(c, FAKE_MANIFEST, job_name, k)
        outcome = json.loads(self.store[(1, f"/local/results/{job_name}.result.json")])
        outcome["shard_actual_start_wall"] += 1.0
        self.store[(1, f"/local/results/{job_name}.result.json")] = json.dumps(outcome)
        acked = {k: json.loads(self.store[(k, f"/local/acks/{job_name}.ack.json")]) for k in range(2)}
        merged = labctl.merge(c, FAKE_MANIFEST, "seg-5", 4.0, "measured", job_name, acked, 2)
        stored = merged["summary"]
        again = summarize(merged["requests"], FAKE_MANIFEST["duration_s"], FAKE_MANIFEST["gpus"],
                          FAKE_MANIFEST["ttft_slo_ms"], FAKE_MANIFEST["tpot_slo_ms"], FAKE_MANIFEST["attainment_fraction"],
                          4.0, FAKE_MANIFEST["max_client_lag_ms"], FAKE_MANIFEST["tpot_metric"],
                          stored["start_skew_ms"], stored["start_skew_tolerance_ms"])
        self.assertEqual(again["client_valid"], stored["client_valid"])
        self.assertEqual(again["meets_joint_slo"], stored["meets_joint_slo"])
        self.assertFalse(again["client_valid"])  # the skew is still real: a fresh re-score must not forget it
        self.assertIn("start skew", merged["summary"]["client_valid_reason"])

    def test_merge_refuses_on_a_failed_shard_ack(self):
        c = {"namespace": "aim345-u", "context": "test"}
        acked = {0: {"status": "failed", "segment_id": "seg-4", "error": "RuntimeError: boom"}, 1: {"status": "done", "segment_id": "seg-4"}}
        with self.assertRaisesRegex(RuntimeError, "unrecoverable error"):
            labctl.merge(c, FAKE_MANIFEST, "seg-4", 4.0, "measured", "seg-4-4-measured", acked, 2)

    def test_cancel_job_only_writes_to_shards_without_an_ack_yet(self):
        c = {"namespace": "aim345-u", "context": "test"}
        labctl.cancel_job(c, 3, "job-x", acked={1: {"status": "done"}})
        self.assertEqual(set(k for k, path in self.store if path.endswith(".cancel")), {0, 2})

    def test_dispatch_writes_job_content_and_shared_start_at_as_two_separate_files(self):
        """Regression: the start barrier must be minted only after every shard's job content has
        landed, not before (DESIGN-RATIONALE.md) -- so it must be its own write, not a field
        embedded in the job file written before dispatch latency is known."""
        c = {"namespace": "aim345-u", "context": "test"}
        before = time.time()
        job_name = labctl.dispatch(c, FAKE_MANIFEST, "seg-6", 4.0, "measured", self._traffic())
        for k in range(2):
            job = json.loads(self.store[(k, f"/local/jobs/{job_name}.job.json")])
            self.assertNotIn("start_at", job)  # start_at is never in the job-content file
            start = json.loads(self.store[(k, f"/local/jobs/{job_name}.start.json")])
            self.assertGreater(start["start_at"], before)  # minted after dispatch(), not before it

    def test_clock_offset_ms_estimates_from_a_fake_date_command(self):
        c = {"namespace": "aim345-u", "context": "test"}
        with patch.object(labctl, "shard_exec") as fake_exec:
            fake_exec.return_value.returncode = 0
            fake_exec.return_value.stdout = "1700000000.000000\n"
            with patch("labctl.time.time", side_effect=[1700000000.0, 1700000000.0]):
                offset = labctl.clock_offset_ms(c, 0)
        self.assertAlmostEqual(offset, 0.0, places=3)

    def test_clock_offset_ms_is_none_when_exec_fails(self):
        c = {"namespace": "aim345-u", "context": "test"}
        with patch.object(labctl, "shard_exec") as fake_exec:
            fake_exec.return_value.returncode = 1
            self.assertIsNone(labctl.clock_offset_ms(c, 0))

    def test_quiesce_shards_cancels_only_outstanding_jobs_and_sleeps_briefly(self):
        c = {"namespace": "aim345-u", "context": "test"}
        state = {labctl.state_key(1, "measured"): {"status": "dispatched", "job_name": "seg-old-1-measured"},
                 labctl.state_key(2, "measured"): {"status": "merged", "job_name": "seg-old-2-measured"}}
        with patch("labctl.time.sleep") as sleep:
            labctl.quiesce_shards(c, dict(FAKE_MANIFEST, shard_count=2), state)
            sleep.assert_called_once()
        paths_written = {path for _, path in self.store}
        self.assertIn("/local/jobs/seg-old-1-measured.cancel", paths_written)
        self.assertNotIn("/local/jobs/seg-old-2-measured.cancel", paths_written)

    def test_quiesce_shards_is_a_no_op_with_nothing_outstanding(self):
        c = {"namespace": "aim345-u", "context": "test"}
        state = {labctl.state_key(1, "measured"): {"status": "merged", "job_name": "seg-old-1-measured"}}
        with patch("labctl.time.sleep") as sleep:
            labctl.quiesce_shards(c, dict(FAKE_MANIFEST, shard_count=2), state)
            sleep.assert_not_called()
        self.assertEqual(self.store, {})


class CoarseWaitTests(unittest.TestCase):
    """Regression for the dominant, previously-unmeasured source of 040's extra wall-clock time:
    polling every shard every POLL_S from the start of a phase that cannot possibly finish before
    its offered window elapses (DESIGN-RATIONALE.md)."""

    def test_coarse_wait_makes_no_shard_exec_calls(self):
        with patch.object(labctl, "shard_exec") as fake_exec:
            labctl.coarse_wait(0.3, lambda: False, margin_s=0.0)
            fake_exec.assert_not_called()

    def test_coarse_wait_returns_promptly_once_cancelled(self):
        calls = {"n": 0}

        def cancel():
            calls["n"] += 1
            return calls["n"] >= 2  # cancelled on the second check
        start = time.time()
        labctl.coarse_wait(30.0, cancel, margin_s=3.0)
        self.assertLess(time.time() - start, 3.0)  # did not wait out the full 27s window

    def test_coarse_wait_leaves_the_margin_for_normal_polling(self):
        start = time.time()
        labctl.coarse_wait(1.0, lambda: False, margin_s=0.3)
        elapsed = time.time() - start
        self.assertGreaterEqual(elapsed, 0.6)  # slept close to duration_s - margin_s (0.7s)
        self.assertLess(elapsed, 1.0)  # but left the margin for poll_acks, not the whole window


class RunnerAgentClaimTests(unittest.TestCase):
    """Real temp directories, not the fake transport -- this is runner_agent's own claim logic,
    the shard side of the two-phase dispatch fix, not labctl's."""

    def test_job_without_a_start_file_is_not_claimable(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "jobs").mkdir()
            (root / "jobs" / "j1.job.json").write_text(json.dumps({"segment_id": "seg-1"}))
            self.assertIsNone(runner_agent.try_claim(root, "j1"))

    def test_job_is_claimable_once_both_files_exist_and_start_at_is_merged_in(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "jobs").mkdir()
            (root / "jobs" / "j1.job.json").write_text(json.dumps({"segment_id": "seg-1"}))
            (root / "jobs" / "j1.start.json").write_text(json.dumps({"start_at": 12345.0}))
            job = runner_agent.try_claim(root, "j1")
            self.assertEqual(job["segment_id"], "seg-1")
            self.assertEqual(job["start_at"], 12345.0)

    def test_a_corrupt_start_file_is_not_claimable_yet_not_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "jobs").mkdir()
            (root / "jobs" / "j1.job.json").write_text(json.dumps({"segment_id": "seg-1"}))
            (root / "jobs" / "j1.start.json").write_text("not json yet")  # mid-rename race, not corruption
            self.assertIsNone(runner_agent.try_claim(root, "j1"))


class WholeRampDispatchTests(unittest.TestCase):
    """labctl's half of DESIGN-RATIONALE.md's "whole-ramp dispatch": one job write per shard for
    every remaining step, a measured (not constant) start barrier, and the ramp-level cancel path
    that quiesce_shards must now also use."""

    def setUp(self):
        self.store = {}
        patch.object(labctl, "shard_write", side_effect=lambda c, k, path, content: self.store.__setitem__((k, path), content)).start()
        patch.object(labctl, "shard_read", side_effect=lambda c, k, path: self.store.get((k, path))).start()
        self.addCleanup(patch.stopall)

    def test_dispatch_ramp_whole_sends_the_full_step_list_and_each_others_peer_info(self):
        c = {"namespace": "aim345-u", "context": "test"}
        before = time.time()
        labctl.dispatch_ramp_whole(c, FAKE_MANIFEST, "seg-r1", [(0.25, "warmup"), (0.25, "measured")], {"model": "m"}, barrier_s=0.01)
        job0 = json.loads(self.store[(0, "/local/jobs/seg-r1.ramp.json")])
        job1 = json.loads(self.store[(1, "/local/jobs/seg-r1.ramp.json")])
        self.assertEqual(job0["steps"], [{"rate": 0.25, "phase": "warmup"}, {"rate": 0.25, "phase": "measured"}])
        self.assertNotIn("start_at", job0)  # start_at is a separate file, same discipline as dispatch()
        self.assertEqual(job0["peers"], [{"shard_index": 1, "host": "runner-1.runner.aim345-u.svc.cluster.local", "port": labctl.PEER_HTTP_PORT}])
        self.assertEqual(job1["peers"], [{"shard_index": 0, "host": "runner-0.runner.aim345-u.svc.cluster.local", "port": labctl.PEER_HTTP_PORT}])
        start0 = json.loads(self.store[(0, "/local/jobs/seg-r1.ramp.start.json")])
        self.assertGreater(start0["start_at"], before)

    def test_measured_start_barrier_s_scales_with_the_slowest_measured_round_trip(self):
        with patch.object(labctl, "exec_rtt_s", side_effect=[0.05, 1.0]):
            barrier = labctl.measured_start_barrier_s({"namespace": "aim345-u", "context": "test"}, 2)
        self.assertAlmostEqual(barrier, 3.0, places=3)  # 3x the slower of the two shards' round trip, above the shard-count floor

    def test_measured_start_barrier_s_falls_back_to_the_floor_when_exec_fails_everywhere(self):
        with patch.object(labctl, "exec_rtt_s", return_value=None):
            barrier = labctl.measured_start_barrier_s({"namespace": "aim345-u", "context": "test"}, 2)
        self.assertEqual(barrier, labctl.START_BARRIER_BUFFER_S)

    def test_exec_rtt_s_is_none_when_the_exec_itself_fails(self):
        c = {"namespace": "aim345-u", "context": "test"}
        with patch.object(labctl, "shard_exec") as fake_exec:
            fake_exec.return_value.returncode = 1
            self.assertIsNone(labctl.exec_rtt_s(c, 0))

    def test_cancel_whole_ramp_writes_to_every_shard(self):
        labctl.cancel_whole_ramp({"namespace": "aim345-u", "context": "test"}, 3, "seg-x")
        self.assertEqual({k for k, path in self.store if path == "/local/jobs/seg-x.ramp.cancel"}, {0, 1, 2})

    def test_quiesce_shards_also_cancels_the_whole_ramp_the_outstanding_job_belonged_to(self):
        """Regression: a crash can leave a shard mid-execute_ramp, not mid-execute (whole-ramp
        dispatch); its per-job .cancel file means nothing to that loop, only .ramp.cancel does."""
        c = {"namespace": "aim345-u", "context": "test"}
        state = {labctl.state_key(1, "measured"): {"status": "dispatched", "job_name": "seg-old-1-measured"}}
        with patch("labctl.time.sleep"):
            labctl.quiesce_shards(c, dict(FAKE_MANIFEST, shard_count=2), state)
        paths_written = {path for _, path in self.store}
        self.assertIn("/local/jobs/seg-old.ramp.cancel", paths_written)


class ExecuteRampTests(unittest.TestCase):
    """Two real asyncio shards (leader shard 0, follower shard 1) talking over real localhost TCP
    sockets -- the actual in-cluster peer protocol, not a fake -- while benchmark.run_rate hits the
    same real HTTP server DispatchMergeTests uses, so this is the shard side of DESIGN-RATIONALE.md's
    "whole-ramp dispatch" end to end."""

    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), StreamHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.tok = tokenizer()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def _job(self, shard_index, peer_ports, steps, root, extra=None):
        args_template = {"duration_s": 0.2, "timeout_s": 5, "ttft_slo_ms": 5000, "tpot_slo_ms": 5000,
                         "attainment_fraction": 0, "context_length_tokens": 32768, "max_tasks": 64,
                         "max_client_lag_ms": 5000, "stop_failure_fraction": 1.0, "skip_warmup": True, "tpot_metric": "mean"}
        peers = [{"shard_index": k, "host": "127.0.0.1", "port": p} for k, p in peer_ports.items() if k != shard_index]
        job = {"segment_id": "seg-ramp", "shard_count": len(peer_ports) + 1, "shard_index": shard_index,
              "peers": peers, "http_port": peer_ports.get(shard_index, 0), "steps": steps,
              "gpus": 2, "router_url": f"http://127.0.0.1:{self.server.server_port}", "architecture": "unified",
              "instance_type": "g7.48xlarge", "region": "us-west-2", "evidence_scope": "mechanism-validation",
              "traffic": {"model": "deepseek-ai/DeepSeek-V2-Lite-Chat", "revision": "85864749cd611b4353ce1decdb286193298f64c7",
                         "shape": {"name": "B-test", "input_tokens": 128, "output_tokens": 4, "document_text": "Order fulfillment incident."},
                         "tasks": [{"task_id": "t0", "calls": 1}, {"task_id": "t1", "calls": 1}]},
              "base_seed": 345, "args_template": args_template, "start_at": time.time() + 0.05}
        if extra:
            job.update(extra)
        return job

    def _free_port(self):
        import socket
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        return port

    def test_two_shards_run_every_step_in_order_and_stay_well_inside_the_skew_tolerance(self):
        with tempfile.TemporaryDirectory() as d0, tempfile.TemporaryDirectory() as d1:
            root0, root1 = Path(d0), Path(d1)
            for root in (root0, root1):
                for sub in ("jobs", "acks", "results"):
                    (root / sub).mkdir()
            p0, p1 = self._free_port(), self._free_port()
            steps = [{"rate": 2.0, "phase": "warmup"}, {"rate": 2.0, "phase": "measured"}]
            job0 = self._job(0, {0: p0, 1: p1}, steps, root0)
            job1 = self._job(1, {0: p0, 1: p1}, steps, root1)

            async def run_both():
                await asyncio.gather(runner_agent.execute_ramp(job0, 0, root0), runner_agent.execute_ramp(job1, 1, root1))
            asyncio.run(run_both())

            for root in (root0, root1):
                for step in steps:
                    job_name = f"seg-ramp-{step['rate']:g}-{step['phase']}"
                    ack = json.loads((root / "acks" / f"{job_name}.ack.json").read_text())
                    self.assertEqual(ack["status"], "done")
            ramp_ack0 = json.loads((root0 / "acks" / "seg-ramp.ramp.ack.json").read_text())
            self.assertEqual(ramp_ack0["status"], "done")
            self.assertEqual(len(ramp_ack0["ran"]), 2)  # both steps ran, in order, on the leader
            measured0 = json.loads((root0 / "results" / "seg-ramp-2-measured.result.json").read_text())
            measured1 = json.loads((root1 / "results" / "seg-ramp-2-measured.result.json").read_text())
            skew_ms = abs(measured0["shard_actual_start_wall"] - measured1["shard_actual_start_wall"]) * 1000
            self.assertLess(skew_ms, 500)  # the one correctness property this whole redesign must not regress
            warm0 = json.loads((root0 / "results" / "seg-ramp-2-warmup.result.json").read_text())
            self.assertLess(measured0["shard_actual_start_wall"] - warm0["shard_actual_start_wall"], 2.0)  # the peer barrier, not labctl, gated this step
            # Each shard walks its own stride of task identities; without shard_index both shards
            # would draw tasks 0, 2, 4, ... and every identity would appear twice in the merged run.
            ids0 = {r["task_id"] for r in measured0["requests"] if r["sent"]}
            ids1 = {r["task_id"] for r in measured1["requests"] if r["sent"]}
            self.assertTrue(ids0 and ids1)
            self.assertFalse(ids0 & ids1)

    def test_cancel_between_steps_stops_the_leader_before_the_next_step_and_marks_the_ack_cancelled(self):
        with tempfile.TemporaryDirectory() as d0, tempfile.TemporaryDirectory() as d1:
            root0, root1 = Path(d0), Path(d1)
            for root in (root0, root1):
                for sub in ("jobs", "acks", "results"):
                    (root / sub).mkdir()
            p0, p1 = self._free_port(), self._free_port()
            steps = [{"rate": 2.0, "phase": "warmup"}, {"rate": 2.0, "phase": "measured"}]
            job0 = self._job(0, {0: p0, 1: p1}, steps, root0)
            job1 = self._job(1, {0: p0, 1: p1}, steps, root1)
            # Cancel as soon as the first step's ack would plausibly exist; execute_ramp checks
            # cancel_path before every step, same as the single-segment path's cancel_event.
            (root0 / "jobs" / "seg-ramp.ramp.cancel").touch()
            (root1 / "jobs" / "seg-ramp.ramp.cancel").touch()

            async def run_both():
                await asyncio.gather(runner_agent.execute_ramp(job0, 0, root0), runner_agent.execute_ramp(job1, 1, root1))
            asyncio.run(run_both())

            for root in (root0, root1):
                ramp_ack = json.loads((root / "acks" / "seg-ramp.ramp.ack.json").read_text())
                self.assertEqual(ramp_ack["status"], "cancelled")
                self.assertEqual(ramp_ack["ran"], [])  # cancelled before even the first step started
                self.assertFalse((root / "acks" / "seg-ramp-2-warmup.ack.json").exists())

    def test_a_follower_that_never_appears_fails_the_leader_within_its_peer_wait_budget_not_forever(self):
        with tempfile.TemporaryDirectory() as d0:
            root0 = Path(d0)
            for sub in ("jobs", "acks", "results"):
                (root0 / sub).mkdir()
            p0, p1 = self._free_port(), self._free_port()  # p1: nothing ever listens here
            steps = [{"rate": 2.0, "phase": "warmup"}, {"rate": 2.0, "phase": "measured"}]
            job0 = self._job(0, {0: p0, 1: p1}, steps, root0, extra={"peer_wait_budget_s": 0.5})
            start = time.time()
            # execute_ramp itself only writes the ramp ack on paths it controls (done/cancelled);
            # a lost peer raises past that write, exactly as main()'s own except: block expects to
            # catch and turn into a "failed" ack -- calling execute_ramp directly here is testing
            # that it raises promptly, not main()'s wrapper around it.
            with self.assertRaises(RuntimeError):
                asyncio.run(runner_agent.execute_ramp(job0, 0, root0))
            self.assertLess(time.time() - start, 5.0)  # bounded by peer_wait_budget_s, not hung forever
            self.assertFalse((root0 / "acks" / "seg-ramp.ramp.ack.json").exists())


if __name__ == "__main__":
    unittest.main()
