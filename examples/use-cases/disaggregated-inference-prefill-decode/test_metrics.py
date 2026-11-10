#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""metrics.py converts this lab's own result files to Prometheus text exposition, without a cluster."""
import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

import metrics


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


SUMMARY_ROW = {
    "architecture": "unified", "shape": "B-long-context", "shape_sha256": "a" * 64,
    "offered_tasks_per_s": "0.5", "slo_ttft_ms": "2000.0", "slo_tpot_ms": "100.0", "attainment_target_fraction": "0.9",
    "p90_ttft_ms_success_only": "174.6", "p90_tpot_ms_success_only": "12.3", "longest_gap_ms_p90": "80.0",
    "max_client_launch_lag_ms": "5.0", "planned_calls": "30", "failed_or_skipped_calls": "0", "unreachable_calls": "0",
    "client_valid": "True", "gpu_count": "8", "meets_joint_slo": "True", "slo_attainment_fraction": "1.0",
    "useful_calls_per_s_per_gpu": "0.0625", "itl_p99_meets_joint_slo": "True", "itl_p99_slo_attainment_fraction": "1.0",
    "itl_p99_useful_calls_per_s_per_gpu": "0.0625", "worst_itl_meets_joint_slo": "unavailable (a file of this stack has no per-chunk timing)",
    "worst_itl_slo_attainment_fraction": "unavailable (a file of this stack has no per-chunk timing)",
    "worst_itl_useful_calls_per_s_per_gpu": "unavailable (a file of this stack has no per-chunk timing)",
}


def parse(text):
    """{(metric, (sorted label items)): value} from Prometheus text exposition, for assertions."""
    out = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        head, value = line.rsplit(" ", 1)
        if "{" in head:
            name, rest = head.split("{", 1)
            labels = tuple(sorted(
                (k, v.strip('"')) for k, v in (pair.split("=", 1) for pair in rest.rstrip("}").split(","))))
        else:
            name, labels = head, ()
        out[(name, labels)] = float(value)
    return out


class SummaryTests(unittest.TestCase):
    def test_disaggregated_rate_metrics_and_rule_labels(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "summary.csv"
            write_csv(path, [SUMMARY_ROW])
            samples = parse(metrics.exposition(metrics.summary_samples(path)))
        def key(**rule):
            return tuple(sorted({"architecture": "unified", "attainment_target": "0.9", "rate": "0.5",
                                 "shape": "B-long-context", "tpot_slo_ms": "100.0", "ttft_slo_ms": "2000.0", **rule}.items()))
        base = key()
        self.assertAlmostEqual(samples[("disagg_rate_ttft_p90_seconds", base)], 0.1746)
        self.assertAlmostEqual(samples[("disagg_rate_tpot_p90_seconds", base)], 0.0123)
        self.assertEqual(samples[("disagg_rate_client_valid", base)], 1.0)
        self.assertEqual(samples[("disagg_rate_pass", key(rule="mean"))], 1.0)
        self.assertEqual(samples[("disagg_rate_pass", key(rule="itl-p99"))], 1.0)
        self.assertAlmostEqual(samples[("disagg_rate_useful_calls_per_s_per_gpu", key(rule="mean"))], 0.0625)
        # worst-itl is UNAVAILABLE on this row, so it sends no sample.
        self.assertNotIn(("disagg_rate_pass", key(rule="worst-itl")), samples)

    def test_prometheus_text_has_one_type_line_per_metric(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "summary.csv"
            write_csv(path, [SUMMARY_ROW, dict(SUMMARY_ROW, architecture="disaggregated")])
            text = metrics.exposition(metrics.summary_samples(path))
        self.assertEqual(text.count("# TYPE disagg_rate_ttft_p90_seconds gauge"), 1)
        self.assertEqual(sum(1 for line in text.splitlines() if line.startswith("disagg_rate_ttft_p90_seconds{")), 2)

    def test_a_missing_value_sends_no_sample(self):
        row = dict(SUMMARY_ROW, longest_gap_ms_p90="")
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "summary.csv"
            write_csv(path, [row])
            samples = metrics.summary_samples(path)
        self.assertFalse([s for s in samples if s[0] == "disagg_rate_longest_gap_p90_seconds"])


class TargetsTests(unittest.TestCase):
    def test_highest_passing_rate_and_goodput_per_rule(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            write_csv(out / "summary.csv", [SUMMARY_ROW])
            qualified = [{
                "group": ["unified", "g7.48xlarge", "us-west-2", "mechanism-validation", "r" * 40, "a" * 64, 8, 2000.0, 100.0, 0.9],
                "group_fields": ["architecture", "instance_type", "region", "evidence_scope", "model_revision",
                                 "shape_sha256", "gpu_count", "slo_ttft_ms", "slo_tpot_ms", "attainment_target_fraction"],
                "by_rule": {"mean": {"highest_tested_offered_tasks_per_s_meeting_slo": 0.5, "useful_calls_per_s_per_gpu_at_that_rate": 0.0625},
                            "itl-p99": None,
                            "worst-itl": "unavailable (a file of this stack has no per-chunk timing)"},
            }]
            (out / "qualified-rates.json").write_text(json.dumps(qualified))
            samples = parse(metrics.exposition(metrics.targets_samples(out)))
        labels = (("architecture", "unified"), ("attainment_target", "0.9"), ("rule", "mean"), ("shape", "B-long-context"),
                  ("tpot_slo_ms", "100.0"), ("ttft_slo_ms", "2000.0"))
        self.assertEqual(samples[("disagg_target_highest_passing_rate", labels)], 0.5)
        self.assertEqual(samples[("disagg_target_goodput", labels)], 0.0625)
        self.assertEqual(samples[("disagg_target_qualified", labels)], 1)
        # No qualifying rate: only the 0, so a push of it replaces an earlier passing value of the group.
        itl = tuple(sorted(dict(labels, rule="itl-p99").items()))
        self.assertEqual(samples[("disagg_target_qualified", itl)], 0)
        self.assertEqual([k[0] for k in samples if dict(k[1]).get("rule") == "itl-p99"], ["disagg_target_qualified"])
        # Not judged (unavailable): nothing at all.
        self.assertFalse([k for k in samples if dict(k[1]).get("rule") == "worst-itl"])

    def test_a_group_without_group_fields_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            write_csv(out / "summary.csv", [SUMMARY_ROW])
            (out / "qualified-rates.json").write_text(json.dumps([{"group": ["unified"], "by_rule": {}}]))
            with self.assertRaises(SystemExit) as e:
                metrics.targets_samples(out)
            self.assertIn("group_fields", str(e.exception))


class AnswersTests(unittest.TestCase):
    def test_longest_and_typical_gap_per_completed_call(self):
        run = {"metadata": {"architecture": "disaggregated", "shape": {"name": "B-long-context"}},
               "summary": {"offered_tasks_per_s": 0.5},
               "requests": [{"ok": True, "task_id": "t0", "turn_index": 0,
                            "chunks": [{"t_ms": 0, "n": 1}, {"t_ms": 50, "n": 1}, {"t_ms": 250, "n": 1}]},
                           {"ok": False, "task_id": "t1", "turn_index": 0, "chunks": []}]}
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "0.5-measured.json"
            f.write_text(json.dumps(run))
            samples = parse(metrics.exposition(metrics.answers_samples([str(f)])))
        labels = (("answer", "t0 call 1"), ("architecture", "disaggregated"), ("rate", "0.5"), ("shape", "B-long-context"))
        self.assertEqual(samples[("disagg_answer_longest_gap_seconds", labels)], 0.2)
        self.assertEqual(samples[("disagg_answer_typical_gap_seconds", labels)], 0.125)
        self.assertFalse([k for k in samples if dict(k[1]).get("answer") == "t1 call 1"])

    def test_missing_path_is_named(self):
        with self.assertRaises(SystemExit) as e:
            metrics.answers_samples(["/no/such/path"])
        self.assertIn("/no/such/path", str(e.exception))



class DuplicateSeriesTests(unittest.TestCase):
    def test_two_samples_with_the_same_name_and_labels_are_refused(self):
        with self.assertRaises(ValueError):
            metrics.exposition([("disagg_x", {"a": "1"}, 1.0), ("disagg_x", {"a": "1"}, 2.0)])
        self.assertIn("disagg_x", metrics.exposition([("disagg_x", {"a": "1"}, 1.0), ("disagg_x", {"a": "2"}, 2.0)]))

    def test_repeats_of_a_rate_with_another_seed_are_separate_series(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "summary.csv"
            write_csv(path, [dict(SUMMARY_ROW, seed="345"), dict(SUMMARY_ROW, seed="1001")])
            samples = metrics.summary_samples(path)
        self.assertEqual({s[1]["seed"] for s in samples}, {"345", "1001"})
        self.assertIn("disagg_rate_pass", metrics.exposition(samples))

    def test_a_refused_input_prints_nothing_so_a_push_of_its_output_sends_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "summary.csv"
            write_csv(path, [SUMMARY_ROW, SUMMARY_ROW])
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                self.assertEqual(metrics.main(["summary", str(path)]), 2)
        self.assertEqual(out.getvalue(), "")
        self.assertIn("collect them into separate 5.collect.py --output directories", err.getvalue())

    def test_an_input_with_nothing_to_convert_fails_instead_of_printing_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "summary.csv"
            path.write_text(",".join(SUMMARY_ROW) + "\n")
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                self.assertEqual(metrics.main(["summary", str(path)]), 1)
                self.assertEqual(metrics.main(["summary", str(Path(d) / "missing.csv")]), 2)
        self.assertEqual(out.getvalue(), "")
        self.assertIn("found nothing to convert", err.getvalue())
        self.assertIn("no such file", err.getvalue())


class RealFilesTests(unittest.TestCase):
    """metrics.py against the files 5.collect.py writes from measured files in benchmark.py's format (the call
    records come from test_collect's helpers), so a renamed column fails here instead of silently sending no
    sample. The Test Plan covers the field names of real measured files."""
    def test_every_column_and_field_metrics_reads_is_in_a_real_collection(self):
        from test_collect import _call, _collect, _run
        with tempfile.TemporaryDirectory() as d:
            calls = [dict(_call(20), task_id=f"t{i}", turn_index=0) for i in range(10)]
            runs = [_run(d, "unified", 1, calls), _run(d, "disaggregated", 1, calls)]
            rows, qualified, _ = _collect(Path(d) / "out", *runs)
            header = set(rows[0])
            read = (list(metrics.RATE_COLUMNS) + [c for columns in metrics.RULE_COLUMNS.values() for c in columns]
                    + [c for _, c in metrics.OBJECTIVE_LABELS] + ["architecture", "shape", "offered_tasks_per_s", "shape_sha256", "seed"])
            self.assertEqual([c for c in read if c not in header], [])
            for g in qualified:
                self.assertEqual(len(g["group_fields"]), len(g["group"]))
                for _, column in metrics.OBJECTIVE_LABELS:
                    self.assertIn(column, g["group_fields"])
            samples = (metrics.summary_samples(Path(d) / "out" / "summary.csv") + metrics.targets_samples(Path(d) / "out")
                       + metrics.answers_samples([str(r) for r in runs]))
            names = {name for name, _, _ in samples}
            self.assertLessEqual(names, metrics.EMITTED_METRICS)
            self.assertTrue({"disagg_rate_pass", "disagg_target_highest_passing_rate", "disagg_answer_longest_gap_seconds"} <= names)
            self.assertTrue(all(labels.get("seed") == "345" for name, labels, _ in samples if name.startswith("disagg_answer_")))
            metrics.exposition(samples)

    def test_the_longest_gap_is_the_value_the_worst_itl_rule_compares(self):
        from test_collect import _run
        import scoring
        chunks = [{"t_ms": 0, "n": 1}, {"t_ms": 30, "n": 4}, {"t_ms": 250, "n": 3}, {"t_ms": 300, "n": 1}]
        call = {"ok": True, "sent": True, "ttft_ms": 100, "tpot_ms": 30, "client_launch_lag_ms": 0, "chunks": chunks, "task_id": "t0", "turn_index": 0}
        with tempfile.TemporaryDirectory() as d:
            run = _run(d, "disaggregated", 1, [call] * 10)
            samples = metrics.answers_samples([str(run)])
        longest = [v for name, _, v in samples if name == "disagg_answer_longest_gap_seconds"]
        self.assertTrue(longest)
        self.assertEqual(set(longest), {scoring.worst_itl_ms(call) / 1e3})
        self.assertEqual(scoring.worst_itl_ms(call), 220)


if __name__ == "__main__":
    unittest.main()
