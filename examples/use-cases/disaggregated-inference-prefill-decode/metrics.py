#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Convert this lab's own result files to Prometheus text exposition, for a separate push tool to send on.

  python3 metrics.py summary results/paired-comparison/summary.csv
  python3 metrics.py targets results/paired-comparison
  python3 metrics.py answers results/paired-disaggregated-b

Each subcommand prints one block of Prometheus text exposition to stdout for one input: one `summary.csv`
(as `5.collect.py` writes it), one `5.collect.py --output` directory (reads its `qualified-rates.json` and
`summary.csv`), or one or more `4.ramp.py --output` directories or `*-measured.json` files. This tool makes
no decision about which rows matter, which shape is the default, or which objective was intended; every row
or file it can read becomes a sample, and inputs that would repeat a series in one group are refused.
`architecture` (`unified`, `disaggregated` or `disaggregated-scaled`), `shape`, `rate`, `seed` (when the input
records one) and, for the three TPOT rules, `rule` are series labels, since one input file holds every
architecture and rule this lab measured; the objective in force for a sample is also carried as labels
(`ttft_slo_ms`, `tpot_slo_ms`, `attainment_target`), not folded into the metric name. Metric names carry the
prefix `disagg_`; every value is a gauge in base units (seconds, ratios), as `scoring.py` computes them.

Pipe the output to a Pushgateway push tool that reads Prometheus text exposition on stdin, grouping each
push by `instance` (for example, the unified stack's namespace, or any name that identifies one comparison),
`kind` (`summary`, `targets` or `answers`) and `run` (the source directory name), so that one comparison's
or one run's pushes never replace another's:

  python3 metrics.py summary results/paired-comparison/summary.csv | \\
    python3 <path to the push tool>/push.py put --job disagg --group instance=<comparison> --group kind=summary \\
      --group run=paired-comparison --via kubectl --namespace <monitoring namespace>
"""
import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

from scoring import worst_itl_ms

# summary.csv column -> (disagg_rate_ metric name suffix, factor to base units). A missing, empty or
# non-numeric value sends no sample.
RATE_COLUMNS = {
    "p90_ttft_ms_success_only": ("ttft_p90_seconds", 1e-3),
    "p90_tpot_ms_success_only": ("tpot_p90_seconds", 1e-3),
    "longest_gap_ms_p90": ("longest_gap_p90_seconds", 1e-3),
    "max_client_launch_lag_ms": ("max_launch_lag_seconds", 1e-3),
    "planned_calls": ("planned_calls", 1),
    "failed_or_skipped_calls": ("failed_calls", 1),
    "unreachable_calls": ("unreachable_calls", 1),
    "client_valid": ("client_valid", 1),
    "gpu_count": ("gpu_count", 1),
}
# rule -> the summary.csv columns of (meets_joint_slo, slo_attainment_fraction, useful_calls_per_s_per_gpu)
# at that rule; "mean" is the original, unprefixed columns.
RULE_COLUMNS = {
    "mean": ("meets_joint_slo", "slo_attainment_fraction", "useful_calls_per_s_per_gpu"),
    "itl-p99": ("itl_p99_meets_joint_slo", "itl_p99_slo_attainment_fraction", "itl_p99_useful_calls_per_s_per_gpu"),
    "worst-itl": ("worst_itl_meets_joint_slo", "worst_itl_slo_attainment_fraction", "worst_itl_useful_calls_per_s_per_gpu"),
}
RULE_METRICS = ("pass", "attainment_ratio", "useful_calls_per_s_per_gpu")
OBJECTIVE_LABELS = (("ttft_slo_ms", "slo_ttft_ms"), ("tpot_slo_ms", "slo_tpot_ms"), ("attainment_target", "attainment_target_fraction"))
# Every metric name a subcommand can print, for dashboards/gen_engines.py's test to check its panels against.
EMITTED_METRICS = frozenset(
    [f"disagg_rate_{metric}" for metric, _factor in RATE_COLUMNS.values()] +
    [f"disagg_rate_{metric}" for metric in RULE_METRICS] +
    ["disagg_target_qualified", "disagg_target_highest_passing_rate", "disagg_target_goodput",
     "disagg_answer_longest_gap_seconds", "disagg_answer_typical_gap_seconds"])


def number(value):
    """A float, or None for a missing, empty or non-numeric value; "True"/"False" become 1/0."""
    try:
        return float({"True": 1, "False": 0}.get(value, value))
    except (TypeError, ValueError):
        return None


def exposition(samples):
    """Prometheus text format for samples [(metric, {label: value}, number)], one TYPE line and its samples
    per metric name, in sorted order. Two samples with the same name and labels are refused: a Pushgateway
    group cannot hold both, so the inputs of one invocation must not repeat a series."""
    seen = set()
    for metric, labels, _ in samples:
        key = (metric, tuple(sorted((k, str(v)) for k, v in labels.items())))
        if key in seen:
            raise ValueError(f"two samples of {metric} with labels {dict(key[1])}: the input mixes rows that differ only in a "
                             "field this tool does not label (instance_type, region, evidence_scope, model_revision, "
                             "gpu_count or shape_sha256); collect them into separate 5.collect.py --output directories "
                             "and push each as its own group")
        seen.add(key)
    lines = []
    for name in sorted({s[0] for s in samples}):
        lines.append(f"# TYPE {name} gauge")
        for metric, labels, value in samples:
            if metric != name:
                continue
            text = ",".join('%s="%s"' % (k, str(v).replace("\\", r"\\").replace('"', r"\"").replace("\n", r"\n"))
                            for k, v in sorted(labels.items()))
            lines.append(f"{name}{{{text}}} {float(value)!r}" if text else f"{name} {float(value)!r}")
    return ("\n".join(lines) + "\n") if lines else ""


def summary_samples(path):
    """One sample per non-empty column of every row of a summary.csv: rate (disagg_rate_*) metrics without a
    rule, and each TPOT rule's pass/attainment/goodput with `rule` added to the labels."""
    samples = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            labels = {"architecture": row["architecture"], "shape": row["shape"], "rate": f"{float(row['offered_tasks_per_s']):g}"}
            if row.get("seed"):
                labels["seed"] = row["seed"]  # repeats of one rate with another --seed are separate rows
            labels.update({label: row[column] for label, column in OBJECTIVE_LABELS})
            for column, (metric, factor) in RATE_COLUMNS.items():
                value = number(row.get(column))
                if value is not None:
                    samples.append((f"disagg_rate_{metric}", labels, value * factor))
            for rule, columns in RULE_COLUMNS.items():
                rule_labels = dict(labels, rule=rule)
                for column, metric in zip(columns, RULE_METRICS):
                    value = number(row.get(column))
                    if value is not None:
                        samples.append((f"disagg_rate_{metric}", rule_labels, value))
    return samples


def targets_samples(directory):
    """For each (architecture, shape, objective, rule) group in a 5.collect.py --output directory's
    qualified-rates.json: whether any rate met the objective (disagg_target_qualified, 1 or 0) and, when one
    did, the highest offered rate that met it and the goodput at that rate. A group that scoring.py could not
    judge under a rule (scoring.UNAVAILABLE) sends no sample for that rule. A collection where nothing
    qualified therefore still sends its zeros, which replace an earlier push of the same group."""
    directory = Path(directory)
    qualified = json.loads((directory / "qualified-rates.json").read_text())
    with open(directory / "summary.csv", newline="") as f:
        shapes = {r["shape_sha256"]: r["shape"] for r in csv.DictReader(f)}
    samples = []
    for g in qualified:
        fields = g.get("group_fields")
        if not fields or len(g["group"]) != len(fields):
            raise SystemExit(f"metrics.py: {directory}/qualified-rates.json has a group with no usable group_fields "
                             f"for its {len(g['group'])} fields; collect it again with this lab's 5.collect.py")
        key = dict(zip(fields, g["group"]))
        labels = {"architecture": key["architecture"], "shape": shapes.get(key["shape_sha256"], key["shape_sha256"])}
        labels.update({label: key[column] for label, column in OBJECTIVE_LABELS})
        for rule, best in (g.get("by_rule") or {}).items():
            if best is not None and not isinstance(best, dict):
                continue
            rule_labels = dict(labels, rule=rule)
            samples.append(("disagg_target_qualified", rule_labels, 0 if best is None else 1))
            if best is None:
                continue
            rate = best.get("highest_tested_offered_tasks_per_s_meeting_slo")
            goodput = best.get("useful_calls_per_s_per_gpu_at_that_rate")
            if rate is not None:
                samples.append(("disagg_target_highest_passing_rate", rule_labels, rate))
            if goodput is not None:
                samples.append(("disagg_target_goodput", rule_labels, goodput))
    return samples


def answers_samples(paths):
    """One pair of samples per completed call with per-chunk timing, from 4.ramp.py --output directories or
    *-measured.json files: the longest gap between two chunks (scoring.worst_itl_ms, the per-call value the
    worst-itl rule compares with the objective) and the median gap. Each call is two series, so push one run
    per group."""
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        raise SystemExit(f"metrics.py: no such file or directory: {', '.join(missing)}")
    files = [f for p in map(Path, paths) for f in (sorted(p.glob("*-measured.json")) if p.is_dir() else [p])]
    if not files:
        raise SystemExit(f"metrics.py: no *-measured.json files in {', '.join(paths)}")
    samples = []
    for f in files:
        run = json.loads(f.read_text())
        m, records = run["metadata"], run["requests"]
        base = {"architecture": m["architecture"], "shape": m["shape"]["name"],
                "rate": f"{float(run['summary']['offered_tasks_per_s']):g}"}
        if m.get("seed") is not None:
            base["seed"] = str(m["seed"])  # repeats of one rate with another --seed are separate series
        for r in records:
            chunks = (r.get("chunks") or []) if r.get("ok") else []
            gaps = [(b["t_ms"] - a["t_ms"]) / 1e3 for a, b in zip(chunks, chunks[1:])]
            if not gaps:
                continue
            labels = dict(base, answer=f"{r['task_id']} call {r['turn_index'] + 1}")
            samples.append(("disagg_answer_longest_gap_seconds", labels, worst_itl_ms(r) / 1e3))
            samples.append(("disagg_answer_typical_gap_seconds", labels, statistics.median(gaps)))
    return samples


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = p.add_subparsers(dest="command", required=True)
    s = commands.add_parser("summary", help="convert a summary.csv")
    s.add_argument("csv")
    t = commands.add_parser("targets", help="convert a 5.collect.py --output directory's qualified-rates.json")
    t.add_argument("directory")
    w = commands.add_parser("answers", help="convert 4.ramp.py measured files' per-answer gaps")
    w.add_argument("paths", nargs="+")
    a = p.parse_args(argv)
    try:
        if a.command == "summary":
            samples = summary_samples(a.csv)
        elif a.command == "targets":
            samples = targets_samples(a.directory)
        else:
            samples = answers_samples(a.paths)
    except FileNotFoundError as exc:
        print(f"metrics.py: no such file: {exc.filename}", file=sys.stderr)
        return 2
    if not samples:
        # Printing nothing would make the push tool refuse the input and leave an earlier push of this group.
        print(f"metrics.py: {a.command} found nothing to convert in its input; nothing was printed", file=sys.stderr)
        return 1
    try:
        sys.stdout.write(exposition(samples))
    except ValueError as exc:
        print(f"metrics.py: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
