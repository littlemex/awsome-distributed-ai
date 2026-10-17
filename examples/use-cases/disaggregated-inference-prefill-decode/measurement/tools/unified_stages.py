#!/usr/bin/env python3
"""Snapshot/diff sglang:per_stage_req_latency_seconds for the UNIFIED stack.

stages.py's own snapshot() refuses a config whose engines are not {'prefill','decode'}
(it is written for the disaggregated stack). The underlying per-stage latency metric
(prefill_forward etc.) is exported with engine_type="unified" too -- confirmed live via
`kubectl exec <unified engine pod> -- curl -s :30000/metrics | grep per_stage_req_latency`.
This script reuses stages.py's own parse()/from_config() (unchanged) and only replaces the
disaggregated-specific role assertion, so the same bracketed warm -> snapshot -> ramp
(--skip-warmup) -> snapshot -> diff recipe README.md documents for the disaggregated stack
applies unmodified to the unified one.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stages import parse, from_config  # noqa: E402


def snapshot(a):
    record = {source: parse(text, source) for source, text in from_config(a.config).items()}
    empty = sorted(source for source, stages in record.items() if not stages)
    if empty:
        raise SystemExit(f"{', '.join(empty)}: no rank-zero per-stage latency metrics; confirm --enable-metrics, "
                         'send one request to that engine, then retake the snapshot')
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(record, indent=2))
    print(json.dumps({'output': a.output, 'engines': len(record)}))


def diff(a):
    before, after = (json.loads(Path(p).read_text()) for p in (a.before, a.after))
    if set(before) != set(after):
        raise SystemExit('Engines changed between snapshots (a pod was replaced or restarted): '
                         f'{sorted(set(before) ^ set(after))}; retake both snapshots')
    rows = []
    for source, stages in after.items():
        for stage, now in stages.items():
            was = before[source].get(stage, {'sum_s': 0.0, 'count': 0})
            count, total_s = now['count'] - was['count'], now['sum_s'] - was['sum_s']
            if count < 0 or total_s < 0:
                raise SystemExit(f'{source} {stage}: counters went backwards; the engine restarted between snapshots')
            if count:
                rows.append({'engine': source, 'engine_type': now['engine_type'], 'stage': stage, 'observations': count,
                             'mean_ms': 1000 * total_s / count})
    if not rows:
        raise SystemExit('No stage was observed between the two snapshots')
    if a.ramp:
        files = sorted(Path(a.ramp).glob('*-measured.json')) + sorted(Path(a.ramp).glob('*-warmup.json'))
        if not files:
            raise SystemExit(f'{a.ramp}: no ramp files; pass the directory of the ramp run between the two snapshots')
        planned = sum(json.loads(f.read_text())['summary']['planned_calls'] for f in files)
        forward = sum(r['observations'] for r in rows if r['stage'] == 'prefill_forward')
        print(f'{a.ramp} planned {planned} calls; the unified engines recorded {forward} forward passes between the snapshots.')
    for row in sorted(rows, key=lambda r: (r['engine'], -r['mean_ms'])):
        print(f"{row['engine_type'] or '-':<8} {row['stage']:<28} {row['observations']:>6} observations  {row['mean_ms']:>10.1f} ms mean")
    if a.output:
        Path(a.output).parent.mkdir(parents=True, exist_ok=True)
        Path(a.output).write_text(json.dumps(rows, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='action', required=True)
    s = sub.add_parser('snapshot')
    s.add_argument('--config', required=True)
    s.add_argument('--output', required=True)
    d = sub.add_parser('diff')
    d.add_argument('before')
    d.add_argument('after')
    d.add_argument('--ramp')
    d.add_argument('--output')
    a = p.parse_args()
    snapshot(a) if a.action == 'snapshot' else diff(a)


if __name__ == '__main__':
    main()
