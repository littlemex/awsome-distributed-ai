#!/usr/bin/env python3
"""Re-score measured ramp runs against other latency objectives, without new traffic.

A ramp records every call's TTFT and TPOT, so the qualifying rate under any other pair of bounds can
be computed from the same files with the harness's own accounting (`benchmark.summarize`). The grid is
an exploratory screen over finite windows, not a confidence bound or a capacity estimate. The rules
for matched files, repeats and stopped ramps are in README.md, "Re-score the runs against another
objective".
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from benchmark import stops_ramp, summarize

# DESIGN.md section 4: client_path distinguishes a pre-redesign (port-forward) result from a
# post-redesign (in-cluster, sharded) one by name, so the two can never pass check_matched against
# each other even if every other field happens to agree (review-1 #10). A file with no client_path
# predates this field entirely and is labeled with the sentinel below, itself a distinct value.
LEGACY_CLIENT_PATH = 'port-forward-v0'

# Equal across every file of one comparison; check_group separately requires both architectures to share seeds and per-seed rates.
# tpot_metric and client_path default on files written before this harness revision added them
# (benchmark.TPOT_METRICS section 14.4, client_path above), so neither is required by ARGUMENTS --
# an old file never fails to load solely for lacking them, it just never matches a new one.
RAMP_MATCHED = ('instance_type', 'region', 'evidence_scope', 'model', 'model_revision', 'shape_sha256', 'gpu_count',
           'offered_window_s', 'timeout_s', 'max_client_lag_ms', 'context_length_tokens', 'skip_warmup',
           'stop_failure_fraction', 'tpot_metric', 'ttft_metric', 'client_path', 'shard_count')
ARGUMENTS = ('max_client_lag_ms', 'duration_s', 'timeout_s', 'context_length_tokens', 'skip_warmup', 'stop_failure_fraction', 'rates')
REQUIRED_METADATA = ('architecture', 'instance_type', 'region', 'evidence_scope', 'model', 'model_revision', 'shape', 'seed', 'phase')
REQUIRED_REQUEST = ('ok', 'sent', 'client_launch_lag_ms')
REQUIRED_SUMMARY = ('gpu_count', 'offered_tasks_per_s', 'measurement_including_drain_s', 'attainment_target_fraction',
                    'slo_ttft_ms', 'slo_tpot_ms')
SUMMARY_KEYS = ('actual_sent_calls_per_s', 'planned_calls', 'sent_calls', 'completed_calls', 'failed_or_skipped_calls', 'p90_ttft_ms_success_only',
                'p90_tpot_ms_success_only', 'slo_attainment_fraction', 'meets_joint_slo', 'successful_calls_per_s',
                'useful_calls_per_s', 'useful_calls_per_s_per_gpu', 'max_client_launch_lag_ms', 'client_valid')


def load_rows(directories):
    rows = []
    for directory in directories:
        files = sorted(Path(directory).glob('*-measured.json'))
        if not files:
            raise SystemExit(f'No measured files in {directory}; nothing was re-scored')
        for path in files:
            try:
                run = json.loads(path.read_text())
            except ValueError as exc:
                raise SystemExit(f'{path}: not readable as JSON ({exc}); the ramp may still be writing it') from None
            absent = [k for k in ('metadata', 'summary', 'requests') if k not in run]
            if absent:
                raise SystemExit(f"{path}: not a ramp result file, missing {', '.join(absent)}")
            m, s = run['metadata'], run['summary']
            arguments = m.get('arguments') or {}
            missing = ([f'metadata.{k}' for k in REQUIRED_METADATA if k not in m]
                       + [f'summary.{k}' for k in REQUIRED_SUMMARY + SUMMARY_KEYS if k not in s]
                       + [f'metadata.arguments.{k}' for k in ARGUMENTS if arguments.get(k) is None])
            if missing:
                raise SystemExit(f"{path}: written by an older or different harness, missing {', '.join(missing)}; "
                                 're-score only ramps written by this revision of benchmark.py')
            if m['phase'] != 'measured':
                raise SystemExit(f"{path}: metadata.phase is {m['phase']!r}; only measured windows are scored")
            bad = next((i for i, r in enumerate(run['requests']) if not all(k in r for k in REQUIRED_REQUEST)), None)
            if bad is not None:
                raise SystemExit(f"{path}: requests[{bad}] lacks one of {', '.join(REQUIRED_REQUEST)}")
            if len(set(arguments['rates'])) != len(arguments['rates']):
                raise SystemExit(f"{path}: metadata.arguments.rates repeats a rate ({arguments['rates']})")
            rows.append({
                'source': str(path), 'run': str(directory), 'ramp_id': arguments.get('ramp_id'),
                'run_id': arguments.get('run_id'), 'segment_id': arguments.get('segment_id'),
                'client_path': arguments.get('client_path', LEGACY_CLIENT_PATH),
                'shard_count': arguments.get('shard_count', 1),
                'rates': list(arguments['rates']), 'architecture': m['architecture'], 'instance_type': m['instance_type'],
                'region': m['region'], 'evidence_scope': m['evidence_scope'], 'model': m['model'],
                'model_revision': m['model_revision'], 'shape': m['shape']['name'],
                'shape_sha256': hashlib.sha256(json.dumps(m['shape'], sort_keys=True).encode()).hexdigest(),
                'traffic_sha256': m.get('traffic_sha256'), 'seed': m['seed'],
                'offered_window_s': arguments['duration_s'], 'timeout_s': arguments['timeout_s'],
                'context_length_tokens': arguments['context_length_tokens'],
                'skip_warmup': arguments['skip_warmup'],
                'stop_failure_fraction': arguments['stop_failure_fraction'],
                'gpu_count': s['gpu_count'], 'offered': s['offered_tasks_per_s'],
                'duration_s': s['measurement_including_drain_s'], 'attainment_target': s['attainment_target_fraction'],
                'max_client_lag_ms': arguments['max_client_lag_ms'], 'stored': s, 'requests': run['requests'],
                'tpot_metric': arguments.get('tpot_metric', 'mean'),
                'ttft_metric': s.get('ttft_metric', 'per-call'),
            })
    return rows


def score(row, ttft_ms, tpot_ms, attainment, tpot_metric=None, ttft_metric=None):
    """tpot_metric=None (ttft_metric=None) re-scores at the mode this row was measured under; an
    explicit override lets --tpot-metric/--ttft-metric re-judge either bound without new traffic
    (DESIGN.md section 14.4), refusing by name on a file with no per-chunk recording the same way
    an explicit itl-p99 mode already does in summarize(). start_skew_ms/start_skew_tolerance_ms are
    replayed from the stored summary (None for every non-sharded ramp) so a labctl-merged rate's
    client_valid/meets_joint_slo reproduces exactly."""
    stored = row['stored']
    return summarize(row['requests'], row['duration_s'], row['gpu_count'], ttft_ms, tpot_ms, attainment,
                     row['offered'], row['max_client_lag_ms'], tpot_metric or row['tpot_metric'],
                     stored.get('start_skew_ms'), stored.get('start_skew_tolerance_ms') or 500.0,
                     ttft_metric or row['ttft_metric'])


def check_reproduces(row):
    """Prove the re-scoring applies the ramp's own accounting before any other bound is tried."""
    stored = row['stored']
    again = score(row, stored['slo_ttft_ms'], stored['slo_tpot_ms'], stored['attainment_target_fraction'])
    # Files written before the harness counted unreachable calls, or before per-chunk ITL recording
    # (DESIGN.md section 14.2), lack those fields.
    optional = ('unreachable_calls', 'tpot_metric', 'ttft_metric', 'itl_ms_p50', 'itl_ms_p95', 'itl_ms_p99',
                'p95_ttft_ms_success_only', 'p99_ttft_ms_success_only',
                'worst_itl_ms_p50', 'worst_itl_ms_p90', 'worst_itl_ms_p95', 'worst_itl_ms_p99', 'worst_itl_ms_p999',
                'start_skew_ms', 'start_skew_tolerance_ms')
    for key in SUMMARY_KEYS + tuple(k for k in optional if k in stored):
        if again[key] != stored[key]:
            raise SystemExit(f"{row['source']}: re-scoring at the recorded objective changed {key} "
                             f"({stored[key]} -> {again[key]}); refusing to report")


def runs_per_rate(rows):
    """How many ramp invocations planned each rate; a rate's repeats for the `*` marker."""
    counts = {}
    for rates in {row['run']: row['rates'] for row in rows}.values():
        for rate in rates:
            counts[rate] = counts.get(rate, 0) + 1
    return counts


def highest_qualifying(rows, ttft_ms, tpot_ms, attainment):
    """One architecture's last rate before any repeat fails a promise, each reported metric minimized across repeats."""
    by_rate = {}
    for run in {row['run'] for row in rows}:
        ran = {row['offered']: row for row in rows if row['run'] == run}
        for rate in next(iter(ran.values()))['rates']:
            by_rate.setdefault(rate, []).append(score(ran[rate], ttft_ms, tpot_ms, attainment) if rate in ran else None)
    top = None
    for rate in sorted(by_rate):
        if not all(s is not None and s['meets_joint_slo'] for s in by_rate[rate]):
            break
        top = rate
    if top is None:
        return None
    scores = by_rate[top]
    return {'offered': top, 'useful_per_gpu': min(s['useful_calls_per_s_per_gpu'] for s in scores),
            'attainment': min(s['slo_attainment_fraction'] for s in scores),
            'planned_calls': min(s['planned_calls'] for s in scores), 'repeats': len(scores),
            'fewest_repeats': min(n for rate, n in runs_per_rate(rows).items() if rate <= top)}


def fmt_rates(xs):
    return ', '.join(f'{x:g}' for x in xs)


def multi_segment_plan(run, rows):
    """DESIGN.md sections 6, 8 and 12: labctl fetches a run's manifest.json and the final
    run_status.json into <output>/.labharness/<run_id>/, next to the merged <rate>-<phase>.json files
    4.ramp.py already writes into that same --output directory (not a separate CWD-relative
    registry, which would get lost once --output differs between two compared stacks), so
    rescore.py and 5.collect.py can check completeness without touching the cluster. A run_id's plan
    is its immutable manifest.json, not any one file's own metadata.arguments.rates, since resuming an
    interrupted run can take more than one labctl invocation (segment) to reach COMPLETE. Returns the
    planned rate order for a run_id-based run, or None when none of these rows carry a run_id (the
    legacy single-ramp_id rule below applies instead)."""
    run_ids = {row['run_id'] for row in rows}
    if run_ids == {None}:
        return None
    if len(run_ids) != 1 or None in run_ids:
        raise SystemExit(f'{run}: files mix more than one run_id ({sorted(str(x) for x in run_ids)}); give each run its own output directory')
    run_id = run_ids.pop()
    harness_dir = Path(run) / '.labharness' / run_id
    manifest_path, status_path = harness_dir / 'manifest.json', harness_dir / 'run_status.json'
    if not manifest_path.exists() or not status_path.exists():
        raise SystemExit(f'{run}: no {harness_dir}; fetch run {run_id} with labctl before re-scoring')
    manifest, status = json.loads(manifest_path.read_text()), json.loads(status_path.read_text())
    if status.get('status') != 'COMPLETE':
        raise SystemExit(f"{run}: run {run_id} is {status.get('status')!r}, not COMPLETE; resume it before re-scoring")
    order = list(manifest['rates'])
    covered = {row['offered'] for row in rows}
    if covered - set(order):
        raise SystemExit(f"{run}: run {run_id} measured {fmt_rates(sorted(covered))}, outside its own planned {fmt_rates(order)}")
    return order


def check_runs(group):
    """Refuse files that one ramp invocation (legacy) or one completed, gap-free run (DESIGN.md
    section 6) could not have written."""
    seen = {}
    for row in group:
        key = (row['architecture'], row['offered'], row['seed'])
        if key in seen:
            raise SystemExit(f"{row['source']} and {seen[key]} share architecture, rate and seed; "
                             'a repeat needs another --seed, and each run is passed once')
        seen[key] = row['source']
    for run in sorted({row['run'] for row in group}):
        rows = [row for row in group if row['run'] == run]
        order = multi_segment_plan(run, rows)
        if order is None:
            if len({(row['architecture'], row['seed'], tuple(row['rates']), row['ramp_id']) for row in rows}) != 1:
                raise SystemExit(f'{run}: files from more than one ramp invocation; give each ramp its own output directory')
            order = rows[0]['rates']
        ran = {row['offered']: row for row in rows}
        done = order[:len(ran)]
        if set(done) != set(ran):
            raise SystemExit(f'{run}: measured {fmt_rates(sorted(ran))} of {fmt_rates(order)} tasks/s run in that order; '
                             'a ramp with a skipped rate is not comparable')
        stopped = [x for x in done if stops_ramp(ran[x]['stored'], rows[0]['stop_failure_fraction'])]
        if stopped and stopped[0] != done[-1]:
            raise SystemExit(f'{run}: the ramp stopped at {stopped[0]:g} tasks/s, yet the directory also holds later rates; '
                             'give each ramp its own output directory')
        if done != order and not stopped:
            raise SystemExit(f'{run}: ended at {done[-1]:g} of {fmt_rates(order)} tasks/s without exceeding its failure fraction, '
                             'so the ramp was interrupted; rerun it')
        below = sorted(x for x in order[len(done):] if x < done[-1])
        if below:
            raise SystemExit(f'{run}: stopped at {done[-1]:g} tasks/s before the lower rates {fmt_rates(below)}; '
                             'only rates above a stop count as misses, so rerun those rates')


def matched_key(row):
    return tuple(row[k] for k in RAMP_MATCHED)


def check_matched(shape, group):
    hashed = [bool(row['traffic_sha256']) for row in group]
    keys = RAMP_MATCHED + ('traffic_sha256',)
    if not any(hashed):
        keys = RAMP_MATCHED
        print(f'{shape}: these runs predate the traffic hash; compared on the shape definition only')
    elif not all(hashed):
        raise SystemExit(f'{shape}: some runs record a traffic hash and some do not; re-score runs from one harness revision')
    for key in keys:
        values = {row['source']: row[key] for row in group}
        if len(set(values.values())) != 1:
            raise SystemExit(f'{shape}: runs differ in {key}: ' + ', '.join(f'{source} has {value}' for source, value in sorted(values.items()))
                             + '; compare one budget, one workload and one ramp protocol at a time')


def check_group(shape, group):
    check_matched(shape, group)
    archs = sorted({row['architecture'] for row in group})
    if archs != ['disaggregated', 'unified']:
        raise SystemExit(f'{shape}: a promise map compares unified with disaggregated, found {archs}')
    check_runs(group)
    protocol = {a: sorted({(row['seed'], tuple(row['rates'])) for row in group if row['architecture'] == a}) for a in archs}
    if protocol[archs[0]] != protocol[archs[1]]:
        raise SystemExit(f'{shape}: the architectures were not run with the same seeds and planned rates ({protocol}); '
                         'compare matched repeats')
    return archs, sorted({rate for row in group for rate in row['rates']})


def verdict(by_arch):
    top = max((b['offered'] for b in by_arch.values() if b), default=None)
    lead = sorted(a for a, b in by_arch.items() if b and b['offered'] == top)
    return top, (lead[0] if len(lead) == 1 else 'tie' if lead else 'neither'), lead


def write_csv(path, records):
    """One CSV for records whose keys may differ, such as summaries written before and after a field was added."""
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in records for k in r)))
        writer.writeheader()
        writer.writerows(records)


def write_rows(rows, ttft_bounds, tpot_bounds, attainment, out):
    table = []
    for row in sorted(rows, key=lambda r: (r['run'], r['offered'])):
        for ttft in ttft_bounds:
            for tpot in tpot_bounds:
                s = score(row, ttft, tpot, attainment)
                table.append({'run': row['run'], 'architecture': row['architecture'], 'shape': row['shape'],
                              'seed': row['seed'], 'offered_tasks_per_s': row['offered'], 'planned_calls': s['planned_calls'],
                              'tpot_metric': row['tpot_metric'], 'ttft_metric': row['ttft_metric'],
                              'p90_ttft_ms_success_only': s['p90_ttft_ms_success_only'],
                              'p95_ttft_ms_success_only': s['p95_ttft_ms_success_only'],
                              'p99_ttft_ms_success_only': s['p99_ttft_ms_success_only'],
                              # Request-average TPOT, the pooled worst-tail ITL, and each call's own
                              # single worst gap side by side, so a reader never has to re-run anything
                              # to see all three senses of "TPOT" at once.
                              'p90_tpot_ms_success_only': s['p90_tpot_ms_success_only'],
                              'itl_ms_p99': s['itl_ms_p99'],
                              'worst_itl_ms_p90': s['worst_itl_ms_p90'], 'worst_itl_ms_p99': s['worst_itl_ms_p99'],
                              'worst_itl_ms_p999': s['worst_itl_ms_p999'],
                              'ttft_bound_ms': ttft,
                              'tpot_bound_ms': tpot, 'slo_attainment_fraction': s['slo_attainment_fraction'],
                              'useful_calls_per_s_per_gpu': s['useful_calls_per_s_per_gpu'],
                              'meets_joint_slo': s['meets_joint_slo']})
    write_csv(out / 'rows.csv', table)
    print(f"Wrote {len(table)} scored rows to {out / 'rows.csv'}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--runs', nargs='+', required=True,
                   help='Ramp output directories, one or more per architecture, all from one deployment configuration')
    p.add_argument('--ttft-ms', type=float, nargs='+', default=[1000, 2000, 3000, 5000],
                   help='TTFT bounds in ms, the columns of the grid')
    p.add_argument('--tpot-ms', type=float, nargs='+', default=[10, 15, 20, 30, 100],
                   help='TPOT bounds in ms per output token, the rows of the grid')
    p.add_argument('--attainment-fraction', type=float,
                   help='Default: the attainment target recorded with the runs')
    p.add_argument('--rows', action='store_true',
                   help='Write every measured file scored at every promise to rows.csv instead of comparing two stacks')
    p.add_argument('--tpot-metric', choices=['mean', 'itl-p99', 'worst-itl'],
                   help='Re-judge TPOT under this sense instead of the mode each run was recorded with (DESIGN.md section 14.4); '
                        'mean is the per-call average, itl-p99 is the pooled per-chunk-gap-per-token percentile, worst-itl is '
                        'each call\'s own single worst raw gap between chunks. Refuses by name on a file with no per-chunk '
                        'recording. Default: the mode recorded with each run')
    p.add_argument('--ttft-metric', choices=['per-call', 'p95', 'p99'],
                   help='Re-judge TTFT under this sense: per-call (default) requires every call\'s own TTFT to pass; p95/p99 '
                        'instead check the pooled percentile across the whole rate once, symmetric with --tpot-metric itl-p99. '
                        'Default: the mode recorded with each run')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    for name, values in (('--ttft-ms', a.ttft_ms), ('--tpot-ms', a.tpot_ms)):
        if any(not math.isfinite(v) or v <= 0 for v in values):
            p.error(f'{name} needs finite positive bounds')

    rows = load_rows(a.runs)
    for row in rows:
        check_reproduces(row)
    if a.tpot_metric is not None:
        for row in rows:
            row['tpot_metric'] = a.tpot_metric
    if a.ttft_metric is not None:
        for row in rows:
            row['ttft_metric'] = a.ttft_metric
    targets = sorted({row['attainment_target'] for row in rows})
    if a.attainment_fraction is None and len(targets) != 1:
        raise SystemExit(f"Runs were recorded with attainment targets {', '.join(f'{t:g}' for t in targets)}; pass --attainment-fraction")
    attainment = targets[0] if a.attainment_fraction is None else a.attainment_fraction
    if not math.isfinite(attainment) or not 0 < attainment <= 1:
        p.error('the attainment fraction must be in (0, 1]; use 0.9 for 90 percent')

    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    if a.rows:
        write_rows(rows, a.ttft_ms, a.tpot_ms, attainment, out)
        return
    table = []
    for shape in sorted({row['shape'] for row in rows}):
        group = [row for row in rows if row['shape'] == shape]
        archs, rates = check_group(shape, group)
        repeats = {arch: list(runs_per_rate([r for r in group if r['architecture'] == arch]).values()) for arch in archs}
        most = max(max(counts) for counts in repeats.values())
        print(f"\n{shape}: {', '.join(archs)} on {group[0]['gpu_count']} GPUs per stack, "
              f"planned rates {', '.join(f'{r:g}' for r in rates)} tasks/s, attainment target {attainment:g}, runs per rate "
              + ', '.join(f'{arch} {min(c)}' if min(c) == max(c) else f'{arch} {min(c)} to {max(c)}' for arch, c in repeats.items()))
        lagged = sorted({(r['architecture'], r['offered']) for r in group if not r['stored']['client_valid']})
        if lagged:
            print('the client fell behind its arrival schedule or could not connect at ' + ', '.join(f'{arch} {x:g} tasks/s' for arch, x in lagged)
                  + '; a wall at those rates belongs to the client path, not to the stack')
        if min(min(c) for c in repeats.values()) < 2:
            print('a rate was run once: a cell near the attainment threshold can change on a repeat; '
                  'rerun the ramp on both stacks with another --seed')
        print('cell = architecture qualifying at the higher rate, and that rate; = tie, - neither'
              + ('; * a rate needed for the winning result was run fewer times than the most-repeated rate' if most > 1 else ''))
        print('TPOT \\ TTFT ' + ''.join(f'{t:>10g}' for t in a.ttft_ms))
        for tpot in a.tpot_ms:
            cells = []
            for ttft in a.ttft_ms:
                by_arch = {arch: highest_qualifying([r for r in group if r['architecture'] == arch], ttft, tpot, attainment)
                           for arch in archs}
                top, win, lead = verdict(by_arch)
                fewer = '*' if any(by_arch[arch]['fewest_repeats'] < most for arch in lead) else ''
                record = {'shape': shape, 'gpu_count': group[0]['gpu_count'], 'ttft_bound_ms': ttft,
                          'tpot_bound_ms': tpot, 'attainment_target_fraction': attainment, 'winner': win,
                          'markers': fewer}
                for arch in archs:
                    b = by_arch[arch]
                    record[f'{arch}_highest_qualifying_offered_tasks_per_s'] = b and b['offered']
                    record[f'{arch}_useful_calls_per_s_per_gpu'] = b and b['useful_per_gpu']
                    record[f'{arch}_planned_calls_at_that_rate'] = b and b['planned_calls']
                    record[f'{arch}_repeats_at_that_rate'] = b and b['repeats']
                table.append(record)
                if win == 'neither':
                    cells.append('-')
                elif win == 'tie':
                    cells.append(f"={top:g}{fewer}")
                else:
                    cells.append(f"{win[0].upper()}{top:g}{fewer}")
            print(f'{tpot:>11g} ' + ''.join(f'{c:>10}' for c in cells))
    write_csv(out / 'promise-map.csv', table)
    print(f"\nRecorded objective reproduced for all {len(rows)} measured rows. Wrote {out / 'promise-map.csv'}")
    print('A rate qualifies only if every ramp that planned it meets the objective there. The grid is an exploratory '
          'screen of finite windows, not a confidence bound; confirm a chosen promise with new, equally repeated runs.')


if __name__ == '__main__':
    main()
