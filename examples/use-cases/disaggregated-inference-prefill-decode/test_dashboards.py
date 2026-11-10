#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""dashboards/gen_engines.py's generated file is current, and every panel queries a name from the list
below of SGLang engine metrics, the Prometheus meta-metric `up`, or a metric metrics.py can emit."""
import json
from pathlib import Path
import re
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "dashboards"))
import metrics
import gen_engines

# The SGLang engine metrics this lab's panels read, scraped live from the engine Services (not pushed),
# under the names SGLang exports.
SGLANG_METRICS = frozenset([
    "sglang:max_total_num_tokens", "sglang:num_prefill_bootstrap_queue_reqs", "sglang:num_queue_reqs",
    "sglang:num_prefill_inflight_queue_reqs", "sglang:num_decode_prealloc_queue_reqs",
    "sglang:num_decode_transfer_queue_reqs", "sglang:num_running_reqs", "sglang:generation_tokens_total",
    "sglang:time_to_first_token_seconds_bucket", "sglang:kv_transfer_latency_ms_bucket",
    "sglang:inter_token_latency_seconds_bucket",
])
# up is Prometheus's own per-target scrape-health meta-metric, not an SGLang or disagg_ metric.
ALLOWED_METRICS = SGLANG_METRICS | metrics.EMITTED_METRICS | {"up"}
# A metric name is sglang:..., disagg_..., or the bare word up; every other PromQL token (functions, label
# matchers, durations) is deliberately left unmatched rather than guessed at.
METRIC_TOKEN = re.compile(r"sglang:[A-Za-z0-9_]+|disagg_[A-Za-z0-9_]+|(?<![A-Za-z0-9_])up(?![A-Za-z0-9_])")


def queried_metrics(doc):
    """Every metric name named in any panel's PromQL expr."""
    names = set()
    for panel in doc["panels"]:
        for target in panel.get("targets", []):
            names.update(METRIC_TOKEN.findall(target.get("expr", "")))
    return names


class DashboardTests(unittest.TestCase):
    def test_generated_file_is_current(self):
        self.assertEqual(gen_engines.OUT.read_text(), gen_engines.render(),
                         "dashboards/engines.json is stale; run python3 dashboards/gen_engines.py")

    def test_every_panel_metric_is_producible(self):
        doc = json.loads(gen_engines.OUT.read_text())
        found = queried_metrics(doc)
        self.assertTrue(found, "no metric names were found in any panel; the extraction regex may be wrong")
        unknown = sorted(n for n in found if n not in ALLOWED_METRICS)
        self.assertEqual(unknown, [], f"panels query metrics this lab cannot produce: {unknown}")

    def test_the_extraction_sees_every_selector(self):
        # Every series selector is a metric name followed by its label matchers. A selector the extraction
        # cannot read, such as one without a name or with a name outside its patterns, fails here instead of
        # passing test_every_panel_metric_is_producible unseen.
        doc = json.loads(gen_engines.OUT.read_text())
        for panel in doc["panels"]:
            for target in panel.get("targets", []):
                expr = re.sub(r'"(?:[^"\\]|\\.)*"', '""', target.get("expr", ""))
                for match in re.finditer(r"([A-Za-z_:][A-Za-z0-9_:]*)?\{", expr):
                    self.assertIsNotNone(match.group(1), f"panel {panel['title']!r} has a selector without a metric name")
                    self.assertTrue(METRIC_TOKEN.fullmatch(match.group(1)), f"panel {panel['title']!r} queries {match.group(1)}, which the extraction does not read")

    def test_live_queries_keep_each_namespace_apart(self):
        # The two stacks' engines must never be summed together: every aggregation over a live series keeps
        # namespace, and "All" selects only the namespaces the variable lists.
        doc = json.loads(gen_engines.OUT.read_text())
        for panel in doc["panels"]:
            for target in panel.get("targets", []):
                expr = target["expr"]
                if "disagg_" in expr:
                    continue
                self.assertIn('namespace=~"$namespace"', expr, panel["title"])
                for by in re.findall(r"(?:sum|max|count) by \(([^)]*)\)", expr):
                    self.assertIn("namespace", [x.strip() for x in by.split(",")], f"{panel['title']}: {expr}")
        self.assertNotIn("allValue", doc["templating"]["list"][0])

    def test_dashboard_has_a_stable_uid_and_the_pushed_metrics_table(self):
        doc = json.loads(gen_engines.OUT.read_text())
        self.assertEqual(doc["uid"], "disagg-engines")
        tables = [p for p in doc["panels"] if p["type"] == "table"]
        self.assertTrue(any(any("disagg_rate_pass" in target["expr"] for target in t["targets"]) for t in tables))
        self.assertEqual(len({p["id"] for p in doc["panels"]}), len(doc["panels"]))

    def test_every_column_of_a_table_aggregates_by_the_same_labels(self):
        # Grafana's merge transformation joins a table's queries row by row on their common labels, so a column
        # with fewer labels would land on whichever row comes first.
        doc = json.loads(gen_engines.OUT.read_text())
        for table in (p for p in doc["panels"] if p["type"] == "table"):
            groupings = {re.match(r"max by \(([^)]*)\)", t["expr"]).group(1) for t in table["targets"]}
            self.assertEqual(len(groupings), 1, table["title"])


if __name__ == "__main__":
    unittest.main()
