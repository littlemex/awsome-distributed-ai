#!/usr/bin/env python3
"""Attribute a disaggregated stack's time to its handoff stages.

SGLang exports, per engine process, a cumulative sum and count of the time requests spend in each
disaggregation stage (`sglang:per_stage_req_latency_seconds`). Two snapshots taken around a ramp
give the mean time per observation in each stage for the traffic between them. Warm the engines
first and run the bracketed ramp with --skip-warmup, so the means exclude warmup; the
README section "Attribute the split stack's time to its stages" gives the commands.

`prefill_forward` is the prefill computation. `prefill_bootstrap` is the prefill-side wait for the
decode engine to accept the request, and `prefill_transfer_kv_cache` the interval from the end of the
forward pass until the decode engine holds the cache. On the decode side, `decode_bootstrap` is the
bootstrap and `decode_transferred` the wait until the KV cache arrives, which also includes the prefill
computation and scheduling. These are overlapping intervals observed from different engines, not
additive parts. The descriptions follow the stage boundaries of the pinned engine (SGLang
v0.5.12.post1, python/sglang/srt/disaggregation/); re-check them when the engine pin changes.

The counters cover every request an engine processes, including periodic single-token generations such
as health checks, so their means do not decompose a client percentile.
"""
import argparse
import json
import math
import re
from pathlib import Path
from deployment import engine_endpoint, engine_ports, kubectl, owned_namespace, read_config

LINE = re.compile(r'sglang:per_stage_req_latency_seconds_(sum|count)\{(?P<labels>[^}]*)\} (?P<value>\S+)')


def parse(text, source):
    """Stage totals from one engine's metrics, rank 0 only (the other ranks repeat the same requests)."""
    series = {}
    for match in LINE.finditer(text):
        labels = dict(re.findall(r'(\w+)="([^"]*)"', match.group('labels')))
        if labels.get('tp_rank', '0') != '0':
            continue
        stage, kind, raw = labels.get('stage', '<no stage label>'), match.group(1), match.group('value')
        try:
            value = float(raw)
        except ValueError:
            value = math.nan
        if not math.isfinite(value) or value < 0 or (kind == 'count' and not value.is_integer()):
            raise SystemExit(f'{source} {stage}: invalid {kind} {raw} in /metrics; check the engine log')
        series.setdefault(tuple(sorted(labels.items())), {})['sum_s' if kind == 'sum' else 'count'] = value
    stages = {}
    for identity, entry in series.items():
        labels = dict(identity)
        if not {'sum_s', 'count'} <= set(entry):
            raise SystemExit(f"{source} {labels.get('stage', '<no stage label>')}: /metrics carries only one of the sum "
                             'and count series')
        total = stages.setdefault(labels.get('stage', '<no stage label>'), {'engine_type': labels.get('engine_type'), 'sum_s': 0.0, 'count': 0})
        total['sum_s'] += entry['sum_s']
        total['count'] += int(entry['count'])
    return stages


def from_config(path):
    c = read_config(path)
    owned_namespace(c)
    pods = json.loads(kubectl(c, 'get', 'pods', '-l', 'component=engine', '-o', 'json', capture_output=True).stdout)['items']
    texts = {}
    for pod in pods:
        status = next((x for x in pod.get('status', {}).get('containerStatuses', []) if x['name'] == 'engine'), None)
        if status is None or not status.get('ready'):
            raise SystemExit(f"{pod['metadata']['name']}: engine container is not ready")
        # The pod UID and restart count make a replaced or restarted engine visible to diff.
        identity = f"{pod['metadata']['name']}@{pod['metadata']['uid']}/restart-{status['restartCount']}"
        for port in engine_ports(pod['spec']):
            texts[f'{identity}:{port}'] = engine_endpoint(c, pod['metadata']['name'], port, 'metrics')
    return texts


def snapshot(a):
    record = {source: parse(text, source) for source, text in from_config(a.config).items()}
    empty = sorted(source for source, stages in record.items() if not stages)
    if empty:
        raise SystemExit(f"{', '.join(empty)}: no rank-zero per-stage latency metrics; confirm --enable-metrics, "
                         'send one request to that engine, then retake the snapshot')
    roles = {entry['engine_type'] for stages in record.values() for entry in stages.values()}
    if not {'prefill', 'decode'} <= roles:
        raise SystemExit(f'Expected per-stage metrics from both a prefill and a decode engine, found {sorted(roles)}; '
                         'is this a disaggregated stack started with --enable-metrics?')
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(record, indent=2))
    print(json.dumps({'output': a.output, 'engines': len(record)}))


def diff(a):
    before, after = (json.loads(Path(p).read_text()) for p in (a.before, a.after))
    if set(before) != set(after):
        raise SystemExit('Engines changed between snapshots (a pod was replaced or restarted): '
                         f'{sorted(set(before) ^ set(after))}; retake both snapshots')
    for source, stages in before.items():
        gone = set(stages) - set(after[source])
        if gone:
            raise SystemExit(f'{source}: stages {sorted(gone)} disappeared; the engine restarted between snapshots')
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
        print(f'{a.ramp} planned {planned} calls; the prefill engines recorded {forward} forward passes between the snapshots. '
              'Passes above the planned count are periodic single-token generations; a count below it means calls were '
              'skipped before sending, the snapshots did not bracket the ramp, or the ramp reached another stack (check its --url).')
    for row in sorted(rows, key=lambda r: (r['engine'], -r['mean_ms'])):
        print(f"{row['engine_type'] or '-':<8} {row['stage']:<28} {row['observations']:>6} observations  {row['mean_ms']:>10.1f} ms mean")
    if a.output:
        Path(a.output).parent.mkdir(parents=True, exist_ok=True)
        Path(a.output).write_text(json.dumps(rows, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='action', required=True)
    s = sub.add_parser('snapshot')
    s.add_argument('--config', required=True, help='Read every engine of this stack through kubectl exec')
    s.add_argument('--output', required=True)
    d = sub.add_parser('diff')
    d.add_argument('before')
    d.add_argument('after')
    d.add_argument('--ramp', help='Ramp directory written between the snapshots; prints its planned calls beside the forward passes')
    d.add_argument('--output')
    a = p.parse_args()
    snapshot(a) if a.action == 'snapshot' else diff(a)


if __name__ == '__main__':
    main()
