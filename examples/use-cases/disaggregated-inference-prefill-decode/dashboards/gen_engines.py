#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Generate dashboards/engines.json, this lab's Grafana dashboard for an optional in-cluster Prometheus.

  python3 dashboards/gen_engines.py           # write the file
  python3 dashboards/gen_engines.py --check   # exit 1 if the file differs from this generator's output

Panels read two kinds of series: SGLang's own metrics, scraped live from the engine Services this lab
labelled with `awsome-distributed-ai/scrape` (panel titles, descriptions, column names and legends are this
lab's text, not SGLang's); and the `disagg_` series that `metrics.py` prints from this lab's own result
files, pushed to the same Pushgateway. The scrape target labels `architecture` (from the Service label
`awsome-distributed-ai/target-architecture`) and `service` (the Service name, e.g. `prefill-0`) identify a
live series; `engine_type` (`unified`, `prefill` or `decode`) is SGLang's own label on its series, and the
throughput, first-token and gap panels leave out the prefill engines, which do not decode. After a change, run
this generator again and commit the result; `--check` fails when the committed file is out of date.
"""
import argparse
import json
import sys
from pathlib import Path

OUT = Path(__file__).with_name("engines.json")
# Every live query selects the namespaces chosen in the namespace variable and keeps namespace in its grouping
# and legend, so the engines of different namespaces are never summed together. 6.scale-prefill.sh relabels
# the disaggregated stack's engine Services disaggregated-scaled.
NS = 'namespace=~"$namespace"'
LAB = 'architecture=~"unified|disaggregated(-scaled)?"'
UNIFIED, DISAGG = f'architecture="unified",{NS}', f'architecture=~"disaggregated(-scaled)?",{NS}'
YES_NO = [{"type": "value", "options": {"0": {"text": "no"}, "1": {"text": "yes"}}}]
LABELS = {"architecture": "Architecture", "shape": "Shape", "rate": "Rate (tasks/s)", "rule": "Rule", "service": "Service",
          "instance": "Comparison", "run": "Run", "namespace": "Namespace", "seed": "Seed",
          "ttft_slo_ms": "TTFT objective (ms)", "tpot_slo_ms": "TPOT objective (ms)", "attainment_target": "Attainment target"}


class Board:
    """Lays out panels left to right, wrapping at 24 grid columns, as Grafana's gridPos expects."""
    def __init__(self):
        self.panels, self.y, self.x, self.h = [], 0, 0, 0

    def row(self, title):
        self.newline()
        self.panels.append({"type": "row", "id": len(self.panels) + 1, "title": title, "gridPos": {"h": 1, "w": 24, "x": 0, "y": self.y}})
        self.y += 1

    def newline(self):
        self.y, self.x, self.h = self.y + self.h, 0, 0

    def add(self, panel, w, h):
        if self.x + w > 24:
            self.newline()
        panel.update(id=len(self.panels) + 1, gridPos={"h": h, "w": w, "x": self.x, "y": self.y})
        self.panels.append(panel)
        self.x, self.h = self.x + w, max(self.h, h)

    def table(self, title, description, by, columns, w=24, h=8):
        """columns: (expression, column name, unit or None, extra field config or None)."""
        targets, rename, overrides = [], {k: LABELS[k] for k in by}, []
        for i, (expr, name, unit, extra) in enumerate(columns):
            ref = chr(65 + i)
            query = f"max by ({', '.join(by)}) ({expr})"
            targets.append({"refId": ref, "expr": query, "instant": True, "format": "table"})
            rename[f"Value #{ref}" if len(columns) > 1 else "Value"] = name
            properties = ([{"id": "unit", "value": unit}] if unit else []) + [{"id": k, "value": v} for k, v in (extra or {}).items()]
            if properties:
                overrides.append({"matcher": {"id": "byName", "options": name}, "properties": properties})
        transformations = [{"id": "merge"}] if len(columns) > 1 else []
        transformations.append({"id": "organize", "options": {"excludeByName": {"Time": True}, "renameByName": rename}})
        self.add({"type": "table", "title": title, "description": description, "targets": targets,
                  "transformations": transformations, "fieldConfig": {"overrides": overrides}}, w, h)

    def series(self, title, description, queries, unit="short", w=12, h=8):
        """queries: (PromQL, legend)."""
        targets = [{"refId": chr(65 + i), "expr": e, "legendFormat": l} for i, (e, l) in enumerate(queries)]
        self.add({"type": "timeseries", "title": title, "description": description, "targets": targets,
                  "fieldConfig": {"defaults": {"unit": unit, "min": 0}}}, w, h)


def build():
    b = Board()
    b.row("Engines")
    b.table("Engines", "One row per scraped SGLang engine Service. Metrics read is no while an engine starts "
            "or when the optional Prometheus cannot reach it.", ("namespace", "architecture", "service"), [
                (f'up{{{LAB},{NS}}}', "Metrics read", None, {"mappings": YES_NO}),
                (f"sglang:max_total_num_tokens{{{NS}}}", "KV cache capacity (tokens)", None, None)], w=12)
    objective = ("ttft_slo_ms", "tpot_slo_ms", "attainment_target")
    rate = ("instance", "run", "architecture", "shape", "rate", "seed") + objective
    b.table("Pushed latencies", "From metrics.py's summary command: each pushed rate's p90 latencies. One row per "
            "comparison (instance), run, rate and objective, so separate pushes and objectives stay separate.", rate, [
                ("disagg_rate_ttft_p90_seconds", "p90 first token", "s", None),
                ("disagg_rate_tpot_p90_seconds", "p90 per token", "s", None)], w=12, h=10)
    b.table("Pushed verdicts", "From metrics.py's summary command: each pushed rate's verdict under each of this "
            "lab's three TPOT rules (mean, itl-p99, worst-itl); a blank cell is a rule a pushed file could not be judged "
            "under. Every column of a table carries the same labels, so its rows merge one to one.", rate + ("rule",), [
                ("disagg_rate_pass", "Pass", None, {"mappings": YES_NO}),
                ("disagg_rate_attainment_ratio", "Within both limits", "percentunit", None),
                ("disagg_rate_useful_calls_per_s_per_gpu", "Useful calls/s/GPU", None, None)], w=24, h=10)

    b.row("Live engine series")
    b.series("Disaggregated stack: requests at each stage", "Requests at each stage of the disaggregated "
             "stack's engines, read at each scrape; a peak shorter than the scrape interval can be missed.", [
                 (f'sglang:num_prefill_bootstrap_queue_reqs{{{DISAGG},engine_type="prefill"}}', "{{namespace}}/{{service}} waiting for the decode handshake"),
                 (f'sglang:num_queue_reqs{{{DISAGG},engine_type="prefill"}}', "{{namespace}}/{{service}} waiting for prefill"),
                 (f'sglang:num_prefill_inflight_queue_reqs{{{DISAGG},engine_type="prefill"}}', "{{namespace}}/{{service}} sending KV"),
                 (f'sglang:num_decode_prealloc_queue_reqs{{{DISAGG},engine_type="decode"}}', "{{namespace}}/{{service}} waiting for KV memory"),
                 (f'sglang:num_decode_transfer_queue_reqs{{{DISAGG},engine_type="decode"}}', "{{namespace}}/{{service}} receiving KV"),
                 (f'sglang:num_queue_reqs{{{DISAGG},engine_type="decode"}}', "{{namespace}}/{{service}} waiting to decode"),
                 (f'sglang:num_running_reqs{{{DISAGG},engine_type="decode"}}', "{{namespace}}/{{service}} decoding")], w=24)
    b.series("Unified stack: running and waiting requests", "Requests that each unified engine is running and "
             "holding in its queue, read at each scrape.", [
                 (f"sglang:num_running_reqs{{{UNIFIED}}}", "{{namespace}}/{{service}} running"),
                 (f"sglang:num_queue_reqs{{{UNIFIED}}}", "{{namespace}}/{{service}} waiting")])
    b.series("Output tokens per second", "Tokens that each architecture's engines generated per second, over "
             "Grafana's rate interval (at least four scrape intervals).", [(f'sum by (namespace, architecture) (rate(sglang:generation_tokens_total{{engine_type!="prefill",{NS}}}[$__rate_interval]))', "{{namespace}} {{architecture}}")], w=8)
    b.series("Engine first-token time, p90", "Measured by the engines from the moment each request reached "
             "them, so it leaves out the router and the client.", [
                 (f'histogram_quantile(0.9, sum by (le, namespace, architecture) (rate(sglang:time_to_first_token_seconds_bucket{{engine_type!="prefill",{NS}}}[$__rate_interval])))', "{{namespace}} {{architecture}}")],
             unit="s", w=8)
    b.series("KV handoff time, p90", "Time the disaggregated stack's prefill engine took to send a request's "
             "KV cache to the decode engine, from the first chunk sent until the transfer was confirmed.", [
                 (f"histogram_quantile(0.9, sum by (le, namespace, architecture) (rate(sglang:kv_transfer_latency_ms_bucket{{{DISAGG}}}[$__rate_interval])))", "{{namespace}} {{architecture}}")],
             unit="ms", w=8)
    itl = 'sglang:inter_token_latency_seconds_bucket{{le="{}",engine_type!="prefill",' + NS + '}}'
    b.series("Token gaps over 200 ms, per engine", "Token gaps longer than 200 ms that each engine recorded "
             "in each rate interval (at least four scrape intervals). An engine counts the gaps of all its requests together, so this "
             "panel cannot tell which answer a gap belongs to.", [
                 (f'sum by (namespace, architecture, service) (increase({itl.format("+Inf")}[$__rate_interval])) - '
                  f'sum by (namespace, architecture, service) (increase({itl.format("0.2")}[$__rate_interval]))', "{{namespace}}/{{service}} {{architecture}}")], w=24)

    return {"uid": "disagg-engines", "title": "Disaggregated inference: engines", "tags": ["disaggregated-inference"],
            "editable": False, "timezone": "browser", "schemaVersion": 41, "time": {"from": "now-1h", "to": "now"},
            "refresh": "10s", "panels": b.panels,
            "templating": {"list": [{"name": "namespace", "label": "Namespace", "type": "query", "refresh": 2, "sort": 1,
                                     "query": {"query": f'label_values(up{{{LAB}}}, namespace)', "refId": "namespace"},
                                     "definition": f'label_values(up{{{LAB}}}, namespace)',
                                     "multi": True, "includeAll": True,
                                     "current": {"text": "All", "value": "$__all"}}]}}


def render():
    return json.dumps(build(), indent=2, ensure_ascii=False) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--check", action="store_true")
    if p.parse_args().check:
        if not OUT.exists() or OUT.read_text() != render():
            print(f"{OUT} differs from dashboards/gen_engines.py; run it without --check", file=sys.stderr)
            return 1
        return 0
    OUT.write_text(render())
    return 0


if __name__ == "__main__":
    sys.exit(main())
