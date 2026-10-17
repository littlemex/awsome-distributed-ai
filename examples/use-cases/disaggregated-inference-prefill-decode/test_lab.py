"""CPU checks for traffic semantics, failure accounting, and streaming parsing."""
import argparse
import asyncio
import contextlib
import csv
import io
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import re
import shlex
from unittest.mock import patch
from pathlib import Path
import threading
import time
import unittest
import httpx
from benchmark import ramp, run_rate, stream_request, summarize, worst_itl_ms
from deployment import effective_tps, engine_ports, read_config, render, total_gpus
from paired import paired_render
import rescore
import stages
import tempfile
from traffic import continuation, common_prefix, initial_prompt, task_input_tokens, tokenizer


class StreamHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.end_headers()
        for count in (0, 1, 4):
            message = {'text': 'ready' if count else '', 'meta_info': {'completion_tokens': count, 'cached_tokens': 3}}
            self.wfile.write(('data: ' + json.dumps(message) + '\n\n').encode())
            self.wfile.flush()
            time.sleep(.01)
        if body['input_ids'] != [999]:
            self.wfile.write(b'data: [DONE]\n\n')


class LabTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = tokenizer()

    def test_agentic_chains_preserve_actual_answer_and_overlap(self):
        shape = json.loads(Path('shape-a.json').read_text())
        prompt = initial_prompt(self.tok, shape, 'task-test')
        self.assertEqual(len(prompt), 1024)
        for turn in range(1, 15):
            previous = prompt
            answer = 'The warehouse is queued. Check the dispatch log. ' * 24
            prompt, overlap = continuation(self.tok, shape, previous, answer, 'task-test', turn)
            self.assertEqual(common_prefix(previous, prompt), len(previous))
            self.assertGreaterEqual(overlap, .5)
            self.assertLessEqual(overlap, .9)
            self.assertIn(answer.strip(), self.tok.decode(prompt[len(previous):]))
        self.assertLess(len(prompt) + 256, 32768)

    def test_long_context_warmup_cannot_seed_measured_prefix(self):
        shape = json.loads(Path('shape-b.json').read_text())
        a = initial_prompt(self.tok, shape, 'same-task', 'warmup')
        b = initial_prompt(self.tok, shape, 'same-task', 'measured')
        self.assertEqual(len(a), 8192)
        self.assertEqual(len(b), 8192)
        self.assertLess(common_prefix(a, b) / len(a), .05)

    def test_input_tokens_choices_draws_one_fixed_length_per_task_not_per_call(self):
        shape = json.loads(Path('shape-b.json').read_text())
        shape['input_tokens_choices'] = [1024, 16384]
        del shape['input_tokens']  # the mixed field replaces the single fixed length entirely
        lengths = {task_input_tokens(shape, f'task-{i}') for i in range(40)}
        self.assertEqual(lengths, {1024, 16384}, 'both lengths must actually appear across enough tasks')
        # Deterministic per task id, regardless of how many times it is asked (warmup vs measured
        # must offer the same length so the two phases stay comparable).
        self.assertEqual(task_input_tokens(shape, 'task-7'), task_input_tokens(shape, 'task-7'))
        a = initial_prompt(self.tok, shape, 'task-7', 'warmup')
        b = initial_prompt(self.tok, shape, 'task-7', 'measured')
        self.assertEqual(len(a), len(b))
        self.assertIn(len(a), (1024, 16384))

    def test_generate_validates_input_tokens_choices(self):
        from traffic import generate
        shape = json.loads(Path('shape-b.json').read_text())
        with tempfile.TemporaryDirectory() as temp:
            shape_path = Path(temp) / 'shape.json'
            out_path = Path(temp) / 'out.json'
            for update in (dict(input_tokens_choices=[1024]), dict(input_tokens_choices=[1024, 1024]),
                           dict(input_tokens_choices=[1024, -5]), dict(input_tokens_choices=[1024, '16384'])):
                shape_path.write_text(json.dumps(shape | update))
                with self.assertRaises(AssertionError), patch('sys.argv', ['prog', '--shape', str(shape_path), '--output', str(out_path)]):
                    generate(str(shape_path))
            a_shape = json.loads(Path('shape-a.json').read_text())
            a_shape['input_tokens_choices'] = [1024, 16384]
            shape_path.write_text(json.dumps(a_shape))
            with self.assertRaisesRegex(AssertionError, 'shape-B'), patch('sys.argv', ['prog', '--shape', str(shape_path), '--output', str(out_path)]):
                generate(str(shape_path))

    def test_joint_slo_counts_errors_and_skipped_followups(self):
        rows = [dict(r, client_launch_lag_ms=0) for r in ({'ok': True, 'sent': True, 'ttft_ms': 100, 'tpot_ms': 10},
                {'ok': True, 'sent': True, 'ttft_ms': 100, 'tpot_ms': 200},
                {'ok': False, 'sent': True}, {'ok': False, 'sent': False})]
        r = summarize(rows, 10, 2, 200, 20, .9, 1, 100)
        self.assertEqual(r['actual_sent_calls_per_s'], .3)
        self.assertEqual(r['slo_attainment_fraction'], .25)
        self.assertEqual(r['useful_calls_per_s'], .1)
        self.assertEqual(r['useful_calls_per_s_per_gpu'], .05)
        self.assertFalse(r['meets_joint_slo'])
        self.assertEqual(r['failed_or_skipped_calls'], 2)

    def test_stream_ignores_empty_initial_event_and_rejects_truncation(self):
        server = ThreadingHTTPServer(('127.0.0.1', 0), StreamHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        async def check():
            async with httpx.AsyncClient(trust_env=False) as client:
                url = f'http://127.0.0.1:{server.server_port}'
                good = await stream_request(client, url, [1], 4, 2)
                bad = await stream_request(client, url, [999], 4, 2)
                self.assertTrue(good['ok'])
                self.assertEqual(good['first_chunk_tokens'], 1)
                self.assertGreater(good['tpot_ms'], 0)
                self.assertFalse(bad['ok'])
        try:
            asyncio.run(check())
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_deployments_hold_gpu_count_and_keep_both_selectors(self):
        c = json.loads(Path('config.example.json').read_text())
        c['nodes'] = ['node-a', 'node-b', 'node-c']
        def workers(doc):
            return [x for x in doc['items'] if x['kind'] == 'Deployment' and x['metadata']['labels'].get('component') == 'engine']
        unified = workers(render(c, 'unified'))
        split = workers(render(c, 'disaggregated'))
        scaled = workers(render(c, 'disaggregated', 2))
        self.assertEqual(len(unified), 2)
        self.assertEqual(len(split), 2)
        self.assertEqual(len(scaled), 3)
        for worker in split:
            pod = worker['spec']['template']['spec']
            engine = pod['containers'][0]
            self.assertIn('--disaggregation-transfer-backend', engine['args'])
            self.assertNotIn('--disaggregation-ib-device', engine['args'])
            self.assertEqual({x['name']: x['value'] for x in engine['env']}['SGLANG_DISAGGREGATION_NIXL_BACKEND'], 'LIBFABRIC')
            self.assertEqual(pod['preemptionPolicy'], 'Never')
        self.assertEqual(split[1], scaled[1])

    def test_prefill_decode_ratio_packs_each_rolls_own_workers_onto_its_own_node(self):
        c = json.loads(Path('config.example.json').read_text())
        c.update(nodes=['node-a', 'node-b'], gpus_per_worker=2, prefill_workers=3, decode_workers=1)
        doc = render(c, 'disaggregated')
        engines = {x['metadata']['labels']['role']: x for x in doc['items']
                   if x['kind'] == 'Deployment' and x['metadata']['labels'].get('component') == 'engine'}
        self.assertEqual(set(engines), {'prefill', 'decode'})
        packed, lone = engines['prefill'], engines['decode']
        self.assertEqual(packed['metadata']['name'], 'prefills')
        pod = packed['spec']['template']['spec']
        # Real-cluster finding (DESIGN-RATIONALE.md): this must stay hostNetwork=true, unlike
        # paired.py's same-node packed pairing -- the counterpart role here is on a different
        # node, and only hostNetwork carried the real cross-node EFA path in testing.
        self.assertTrue(pod['hostNetwork'])
        self.assertEqual(pod['nodeSelector']['kubernetes.io/hostname'], 'node-a')
        engine = pod['containers'][0]
        self.assertEqual(engine['resources']['requests']['nvidia.com/gpu'], 6)
        self.assertEqual(engine['resources']['requests']['vpc.amazonaws.com/efa'], c['efa_per_worker'])
        plans = json.loads(next(x['value'] for x in engine['env'] if x['name'] == 'AIM345_ENGINE_PLANS'))
        self.assertEqual([p['env']['CUDA_VISIBLE_DEVICES'] for p in plans], ['0,1', '2,3', '4,5'])
        self.assertEqual([p['args'][p['args'].index('--port') + 1] for p in plans], ['30000', '30001', '30002'])
        self.assertTrue(all(p['role'] == 'prefill' for p in plans))
        # Each packed prefill worker must bind its own bootstrap port: real-cluster testing found
        # every one sharing the literal 8998 silently loses the second and third bind, so a
        # decode engine's KV pull against the router's advertised port only ever reaches worker 0.
        self.assertEqual([p['args'][p['args'].index('--disaggregation-bootstrap-port') + 1] for p in plans], ['8998', '8999', '9000'])
        services = {x['metadata']['name']: x for x in doc['items'] if x['kind'] == 'Service' and x['metadata']['name'] != 'router'}
        self.assertEqual({x['metadata']['name'] for x in doc['items'] if x['kind'] == 'Service'} - {'router'},
                          {'prefill-0', 'prefill-1', 'prefill-2', 'decode-0'})
        for index, name in enumerate(('prefill-0', 'prefill-1', 'prefill-2')):
            self.assertEqual(services[name]['spec']['selector']['app'], 'prefills')
            self.assertEqual(services[name]['spec']['ports'][0]['targetPort'], 30000 + index)
        # The lone decode worker on the other node is untouched by packing: same shape as a
        # one-worker role has always rendered.
        self.assertEqual(lone['metadata']['name'], 'decode-0')
        self.assertEqual(lone['spec']['template']['spec']['nodeSelector']['kubernetes.io/hostname'], 'node-b')
        self.assertTrue(lone['spec']['template']['spec']['hostNetwork'])
        self.assertEqual(lone['spec']['template']['spec']['containers'][0]['resources']['requests']['nvidia.com/gpu'], 2)
        router_args = next(x for x in doc['items'] if x['kind'] == 'Deployment' and x['metadata']['labels'].get('app') == 'router')['spec']['template']['spec']['containers'][0]['args']
        self.assertEqual(router_args.count('--prefill'), 3)
        self.assertEqual(router_args.count('--decode'), 1)
        # The router must be told each prefill worker's own bootstrap port, not the same literal
        # value three times.
        self.assertEqual([router_args[i + 2] for i, a in enumerate(router_args) if a == '--prefill'], ['8998', '8999', '9000'])
        self.assertEqual(total_gpus(c, 'disaggregated'), 8)
        self.assertEqual(effective_tps(c, 'disaggregated'), {2})

    def test_packed_decode_workers_need_no_bootstrap_port(self):
        c = json.loads(Path('config.example.json').read_text())
        c.update(nodes=['node-a', 'node-b'], gpus_per_worker=2, prefill_workers=1, decode_workers=3)
        doc = render(c, 'disaggregated')
        packed = next(x for x in doc['items'] if x['kind'] == 'Deployment' and x['metadata']['labels'].get('role') == 'decode')
        plans = json.loads(next(x['value'] for x in packed['spec']['template']['spec']['containers'][0]['env'] if x['name'] == 'AIM345_ENGINE_PLANS'))
        for p in plans:
            self.assertNotIn('--disaggregation-bootstrap-port', p['args'])
        router_args = next(x for x in doc['items'] if x['kind'] == 'Deployment' and x['metadata']['labels'].get('app') == 'router')['spec']['template']['spec']['containers'][0]['args']
        self.assertEqual(router_args.count('--decode'), 3)

    def test_unified_replicas_spreads_round_robin_and_packs_extras_per_node(self):
        c = json.loads(Path('config.example.json').read_text())
        c.update(nodes=['node-a', 'node-b'], gpus_per_worker=2, unified_replicas=4)
        doc = render(c, 'unified')
        pods = [x for x in doc['items'] if x['kind'] == 'Deployment' and x['metadata']['labels'].get('component') == 'engine']
        self.assertEqual(len(pods), 2, 'four replicas on two nodes pack two workers per pod')
        self.assertEqual(len({x['metadata']['name'] for x in pods}), 2, 'each node packed onto its own role needs a distinct pod name')
        for pod in pods:
            self.assertEqual(pod['spec']['template']['spec']['containers'][0]['resources']['requests']['nvidia.com/gpu'], 4)
        services = [x['metadata']['name'] for x in doc['items'] if x['kind'] == 'Service' and x['metadata']['name'] != 'router']
        self.assertEqual(sorted(services), ['unified-0', 'unified-1', 'unified-2', 'unified-3'])
        router_args = next(x for x in doc['items'] if x['kind'] == 'Deployment' and x['metadata']['labels'].get('app') == 'router')['spec']['template']['spec']['containers'][0]['args']
        self.assertEqual(router_args.count('--worker-urls'), 1)
        self.assertEqual(sum('unified-' in a for a in router_args), 4)
        self.assertEqual(total_gpus(c, 'unified'), 8)

    def test_ratio_defaults_are_the_exact_prior_unpacked_shape(self):
        c = json.loads(Path('config.example.json').read_text())
        c['nodes'] = ['node-a', 'node-b']
        baseline_split = render(c, 'disaggregated')
        baseline_unified = render(c, 'unified')
        for key, value in (('prefill_workers', 1), ('decode_workers', 1)):
            self.assertEqual(render({**c, key: value}, 'disaggregated'), baseline_split)
        self.assertEqual(render({**c, 'unified_replicas': 2}, 'unified'), baseline_unified)
        self.assertEqual(total_gpus(c, 'disaggregated'), 2 * c['gpus_per_worker'])
        self.assertEqual(total_gpus(c, 'unified'), 2 * c['gpus_per_worker'])

    def test_scale_prefill_cannot_combine_with_the_new_ratio_keys(self):
        c = json.loads(Path('config.example.json').read_text())
        c.update(nodes=['node-a', 'node-b', 'node-c'], prefill_workers=2)
        with self.assertRaisesRegex(AssertionError, 'cannot combine'):
            render(c, 'disaggregated', 2)

    def test_read_config_validates_the_new_role_keys(self):
        base = json.loads(Path('config.example.json').read_text())
        base.update(context='test', nodes=['a', 'b'], image='img@sha256:' + 'a' * 64,
                    router_image='router@sha256:' + 'b' * 64, instance_type='g7.48xlarge',
                    gpus_per_worker=2, efa_per_worker=1, model_cache_host_path='/mnt/x')
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'config.json'
            path.write_text(json.dumps(base))
            read_config(path)  # defaults alone must still pass
            for update in (dict(prefill_workers=0), dict(decode_workers=-1), dict(unified_replicas='3'),
                           dict(prefill_gpus_per_worker=0)):
                path.write_text(json.dumps(base | update))
                with self.assertRaises(AssertionError):
                    read_config(path)
            # DeepSeek-V2-Lite-Chat's TP legality check must also catch a per-role override.
            path.write_text(json.dumps(base | dict(decode_gpus_per_worker=3)))
            with self.assertRaises(AssertionError):
                read_config(path)

    def _run(self, directory, architecture, rate, rows, gpus=4, seed=345, traffic='0' * 64, window=10, rates=None, ramp_id=None):
        shape = {'name': 'B-long-context', 'input_tokens': 8192, 'output_tokens': 256}
        s = summarize(rows, window, gpus, 2000, 100, .9, rate, 100)
        arguments = {'max_client_lag_ms': 100, 'duration_s': window, 'timeout_s': 310, 'context_length_tokens': 32768,
                     'gpus': gpus, 'seed': seed, 'skip_warmup': False, 'stop_failure_fraction': .25,
                     'rates': rates or [rate], 'ramp_id': ramp_id}
        meta = {'architecture': architecture, 'instance_type': 'g7.48xlarge', 'region': 'test', 'evidence_scope': 'local-mock',
                'model': 'test-model', 'model_revision': 'r', 'shape': shape, 'traffic_sha256': traffic, 'seed': seed, 'phase': 'measured',
                'arguments': arguments}
        path = Path(directory) / f'{architecture}-{seed}'
        path.mkdir(exist_ok=True)
        (path / f'{rate:g}-measured.json').write_text(json.dumps({'metadata': meta, 'summary': s, 'requests': rows}))
        return path

    @staticmethod
    def _ok(ttft=100, tpot=10):
        return {'ok': True, 'sent': True, 'ttft_ms': ttft, 'tpot_ms': tpot, 'client_launch_lag_ms': 0}

    FAILED = {'ok': False, 'sent': True, 'client_launch_lag_ms': 0}

    def _winner(self, rows, ttft, tpot, attainment=.9):
        archs, _ = rescore.check_group(rows[0]['shape'], rows)
        return rescore.verdict({a: rescore.highest_qualifying([r for r in rows if r['architecture'] == a], ttft, tpot, attainment)
                                for a in archs})[1]

    def test_rescore_reproduces_the_recorded_objective_and_moves_the_winner(self):
        with tempfile.TemporaryDirectory() as d:
            u = self._run(d, 'unified', 2, [self._ok(400, 40)] * 10, rates=[1, 2])
            self._run(d, 'unified', 1, [self._ok(400, 12)] * 10, rates=[1, 2])
            s = self._run(d, 'disaggregated', 2, [self._ok(900, 15)] * 10, rates=[1, 2])
            self._run(d, 'disaggregated', 1, [self._ok(900, 15)] * 10, rates=[1, 2])
            rows = rescore.load_rows([u, s])
            self.assertEqual(rescore.check_group('B', rows), (['disaggregated', 'unified'], [1, 2]))
            for row in rows:
                rescore.check_reproduces(row)
            self.assertEqual(self._winner(rows, 2000, 100), 'tie')
            self.assertEqual(self._winner(rows, 500, 100), 'unified')
            self.assertEqual(self._winner(rows, 500, 20), 'unified')
            self.assertEqual(self._winner(rows, 2000, 20), 'disaggregated')
            self.assertEqual(self._winner(rows, 500, 10), 'neither')

    def test_rescore_keeps_failures_in_the_denominator_and_rejects_a_late_client(self):
        with tempfile.TemporaryDirectory() as d:
            row = rescore.load_rows([self._run(d, 'unified', 1, [self._ok()] * 8 + [self.FAILED] * 2)])[0]
            self.assertFalse(rescore.score(row, 10**6, 10**6, .9)['meets_joint_slo'])
            unreachable = dict(self.FAILED, error='ConnectError: All connection attempts failed')
            self.assertFalse(rescore.score(dict(row, requests=[self._ok()] * 9 + [unreachable]), 10**6, 10**6, .5)['client_valid'])
            row['requests'] = [dict(self._ok(), client_launch_lag_ms=250)] * 10
            self.assertFalse(rescore.score(row, 10**6, 10**6, .9)['meets_joint_slo'])
            self.assertFalse(rescore.score(row, 10**6, 10**6, .9)['client_valid'])

    def test_rescore_refuses_a_changed_record(self):
        with tempfile.TemporaryDirectory() as d:
            row = rescore.load_rows([self._run(d, 'unified', 1, [self._ok()] * 10)])[0]
            row['requests'][0] = self.FAILED
            with self.assertRaisesRegex(SystemExit, 'changed'):
                rescore.check_reproduces(row)
            row = rescore.load_rows([self._run(d, 'unified', 1, [self._ok()] * 10)])[0]
            row['stored']['actual_sent_calls_per_s'] *= 2
            with self.assertRaisesRegex(SystemExit, 'actual_sent_calls_per_s'):
                rescore.check_reproduces(row)
            row = rescore.load_rows([self._run(d, 'unified', 1, [self._ok()] * 10)])[0]
            row['stored']['unreachable_calls'] = 1
            with self.assertRaisesRegex(SystemExit, 'unreachable_calls'):
                rescore.check_reproduces(row)

    def test_rescore_counts_a_rate_only_when_every_repeat_qualifies(self):
        with tempfile.TemporaryDirectory() as d:
            runs = [self._run(d, 'disaggregated', 1, [self._ok()] * 10, seed=345),
                    self._run(d, 'disaggregated', 1, [self._ok()] * 5 + [self._ok(tpot=50)] * 5, seed=346),
                    self._run(d, 'unified', 1, [self._ok()] * 10, seed=345),
                    self._run(d, 'unified', 1, [self._ok()] * 10, seed=346)]
            rows = rescore.load_rows(runs)
            rescore.check_group('B', rows)
            split = [r for r in rows if r['architecture'] == 'disaggregated']
            self.assertIsNone(rescore.highest_qualifying(split, 2000, 20, .9))
            self.assertEqual(rescore.highest_qualifying(split, 2000, 100, .9)['repeats'], 2)

    def test_rescore_stops_at_the_first_failing_rate(self):
        with tempfile.TemporaryDirectory() as d:
            self._run(d, 'unified', 0.5, [self._ok()] * 10, rates=[.5, 1, 2])
            self._run(d, 'unified', 1, [self._ok(tpot=50)] * 10, rates=[.5, 1, 2])
            u = self._run(d, 'unified', 2, [self._ok()] * 10, rates=[.5, 1, 2])
            self.assertEqual(rescore.highest_qualifying(rescore.load_rows([u]), 2000, 20, .9)['offered'], 0.5)

    def test_rescore_refuses_runs_that_are_not_one_matched_comparison(self):
        ok = [self._ok()] * 10
        with tempfile.TemporaryDirectory() as d:
            u = self._run(d, 'unified', 1, ok)
            s = self._run(d, 'disaggregated', 1, ok)
            with self.assertRaisesRegex(SystemExit, 'compares unified with disaggregated'):
                rescore.check_group('B', rescore.load_rows([u]))
            other = Path(d) / 'renamed'
            other.mkdir()
            with self.assertRaisesRegex(SystemExit, 'compares unified with disaggregated'):
                rescore.check_group('B', rescore.load_rows([u, self._run(other, 'unified-b', 1, ok)]))
            with self.assertRaisesRegex(SystemExit, 'share architecture, rate and seed'):
                rescore.check_group('B', rescore.load_rows([u, s, s]))
            s346 = self._run(d, 'disaggregated', 1, ok, seed=346)
            with self.assertRaisesRegex(SystemExit, 'same seeds and planned rates'):
                rescore.check_group('B', rescore.load_rows([u, s, s346]))
            other = Path(d) / 'other'
            other.mkdir()
            s_other = self._run(other, 'disaggregated', 1, ok, traffic='1' * 64)
            with self.assertRaisesRegex(SystemExit, 'differ in traffic_sha256'):
                rescore.check_group('B', rescore.load_rows([u, s_other]))
            window = Path(d) / 'window'
            window.mkdir()
            s_window = self._run(window, 'disaggregated', 1, ok, window=30)
            with self.assertRaisesRegex(SystemExit, 'differ in offered_window_s'):
                rescore.check_group('B', rescore.load_rows([u, s_window]))

    def test_rescore_counts_rates_a_stopped_ramp_did_not_reach_as_misses(self):
        with tempfile.TemporaryDirectory() as d:
            runs = []
            for seed, first in ((345, [self.FAILED] * 4 + [self._ok()] * 6), (346, [self._ok()] * 10)):
                for arch in ('unified', 'disaggregated'):
                    runs.append(self._run(d, arch, 1, [self._ok()] * 10 if arch == 'unified' else first, seed=seed, rates=[1, 2]))
                    if arch == 'unified' or seed == 346:
                        self._run(d, arch, 2, [self._ok()] * 10, seed=seed, rates=[1, 2])
            rows = rescore.load_rows(runs)
            split = [r for r in rows if r['architecture'] == 'disaggregated']
            # Seed 345 stopped at 1 task/s with 60 percent attainment, which meets a 50 percent target;
            # its missing 2 tasks/s must stop the walk even though seed 346 qualified there.
            self.assertEqual(rescore.highest_qualifying(split, 2000, 100, .5)['offered'], 1)
            self.assertEqual(self._winner(rows, 2000, 100, .5), 'unified')

    def _matched(self, d, rates, drop=None, rows=None):
        runs = []
        for arch in ('unified', 'disaggregated'):
            for rate in rates:
                if (arch, rate) != drop:
                    run = self._run(d, arch, rate, (rows or {}).get((arch, rate), [self._ok()] * 10), rates=rates)
            runs.append(run)
        return rescore.load_rows(runs)

    def test_rescore_refuses_a_matched_field_that_differs_inside_one_directory(self):
        with tempfile.TemporaryDirectory() as d:
            rows = self._matched(d, [1, 2])
            next(r for r in rows if r['architecture'] == 'unified' and r['offered'] == 1)['traffic_sha256'] = '1' * 64
            with self.assertRaisesRegex(SystemExit, 'runs differ in traffic_sha256'):
                rescore.check_group('B', rows)

    def _collect(self, out, *runs):
        spec = importlib.util.spec_from_file_location('collect', '5.collect.py')
        collect = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(collect)
        with patch('sys.argv', ['5.collect.py', '--runs', *map(str, runs), '--output', str(out)]), contextlib.redirect_stdout(io.StringIO()):
            try:
                collect.main()
                refusal = None
            except SystemExit as exc:
                refusal = str(exc)
        return {q['group'][0]: q for q in json.loads((Path(out) / 'qualified-rates.json').read_text())}, refusal

    def test_collect_reports_one_qualifying_rate_per_architecture(self):
        with tempfile.TemporaryDirectory() as d:
            u = self._run(d, 'unified', 1, [self._ok()] * 10, rates=[1, 2])
            self._run(d, 'unified', 2, [self._ok()] * 10, rates=[1, 2])
            s = self._run(d, 'disaggregated', 1, [self._ok()] * 10, rates=[1, 2])
            self._run(d, 'disaggregated', 2, [self._ok(ttft=3000)] * 10, rates=[1, 2])
            groups, refusal = self._collect(Path(d) / 'out', u, s)
            self.assertIsNone(refusal)
            self.assertEqual({a: g['highest_qualifying_offered_tasks_per_s'] for a, g in groups.items()}, {'unified': 2, 'disaggregated': 1})

    def test_collect_keeps_the_csv_and_names_a_refused_group(self):
        with tempfile.TemporaryDirectory() as d:
            u = self._run(d, 'unified', 1, [self._ok()] * 10, rates=[1, 2])
            self._run(d, 'unified', 2, [self._ok()] * 10, rates=[1, 2])
            s = self._run(d, 'disaggregated', 1, [self._ok()] * 10, rates=[1, 2])
            groups, refusal = self._collect(Path(d) / 'out', u, s)
            self.assertIn('was interrupted', refusal)
            self.assertIn('was interrupted', groups['disaggregated']['refused'])
            self.assertEqual(groups['unified']['highest_qualifying_offered_tasks_per_s'], 2)
            self.assertEqual(len((Path(d) / 'out' / 'summary.csv').read_text().splitlines()), 4)

    def test_collect_puts_different_windows_in_separate_groups(self):
        with tempfile.TemporaryDirectory() as d:
            a = self._run(d, 'unified', 1, [self._ok()] * 10, seed=345)
            b = self._run(d, 'unified', 1, [self._ok()] * 10, seed=346, window=30)
            with self.assertRaisesRegex(SystemExit, 'runs differ in offered_window_s'):
                rescore.check_matched('B', rescore.load_rows([a, b]))
            out = Path(d) / 'o'
            _, refusal = self._collect(out, a, b)
            self.assertIsNone(refusal)
            self.assertEqual(len(json.loads((out / 'qualified-rates.json').read_text())), 2)

    def test_ramp_refuses_an_output_directory_that_already_holds_ramp_files(self):
        with tempfile.TemporaryDirectory() as d:
            out = self._run(d, 'unified', 1, [self._ok()] * 10)
            with self.assertRaisesRegex(SystemExit, 'its own --output directory'):
                asyncio.run(ramp(argparse.Namespace(output=str(out), traffic='unused')))

    def test_rescore_refuses_files_from_two_invocations_of_one_command(self):
        with tempfile.TemporaryDirectory() as d:
            self._run(d, 'unified', 1, [self._ok()] * 10, rates=[1, 2], ramp_id='a')
            u = self._run(d, 'unified', 2, [self._ok()] * 10, rates=[1, 2], ramp_id='b')
            with self.assertRaisesRegex(SystemExit, 'more than one ramp invocation'):
                rescore.check_runs(rescore.load_rows([u]))

    def test_rescore_refuses_a_file_that_is_not_a_measured_window_of_this_harness(self):
        with tempfile.TemporaryDirectory() as d:
            u = self._run(d, 'unified', 1, [self._ok()] * 10)
            f = next(Path(u).glob('*-measured.json'))
            good = json.loads(f.read_text())
            for change, reason in ((lambda r: r['metadata'].update(phase='warmup'), "phase is 'warmup'"),
                                   (lambda r: r['requests'][0].pop('client_launch_lag_ms'), r'requests\[0\] lacks'),
                                   (lambda r: r['metadata']['arguments'].update(rates=[1, 1.0]), 'repeats a rate')):
                run = json.loads(json.dumps(good))
                change(run)
                f.write_text(json.dumps(run))
                with self.assertRaisesRegex(SystemExit, reason):
                    rescore.load_rows([u])
            f.write_text('{')
            with self.assertRaisesRegex(SystemExit, 'not readable as JSON'):
                rescore.load_rows([u])

    def test_rescore_refuses_a_skipped_rate(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(SystemExit, 'skipped rate'):
                rescore.check_group('B', self._matched(d, [1, 2, 4], drop=('unified', 2)))

    def test_rescore_refuses_an_interrupted_ramp(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(SystemExit, 'was interrupted'):
                rescore.check_group('B', self._matched(d, [1, 2], drop=('disaggregated', 2)))

    def test_rescore_refuses_later_files_beside_a_stopped_rate(self):
        with tempfile.TemporaryDirectory() as d:
            stopped = {('disaggregated', 1): [self.FAILED] * 4 + [self._ok()] * 6}
            with self.assertRaisesRegex(SystemExit, 'also holds later rates'):
                rescore.check_group('B', self._matched(d, [1, 2], rows=stopped))

    def test_rescore_refuses_a_stop_with_unmeasured_lower_rates(self):
        with tempfile.TemporaryDirectory() as d:
            stopped = {('disaggregated', 2): [self.FAILED] * 4 + [self._ok()] * 6}
            with self.assertRaisesRegex(SystemExit, 'before the lower rates 1'):
                rescore.check_group('B', self._matched(d, [2, 1], drop=('disaggregated', 1), rows=stopped))

    def test_rows_scores_one_architecture_under_two_settings(self):
        with tempfile.TemporaryDirectory() as d:
            a = self._run(Path(d), 'unified', 1, [self._ok(tpot=10)] * 10)
            other = Path(d) / 'other'
            other.mkdir()
            b = self._run(other, 'unified', 1, [self._ok(tpot=30)] * 10)
            out = Path(d) / 'out'
            out.mkdir()
            rescore.write_rows(rescore.load_rows([a, b]), [2000], [20], .9, out)
            table = list(csv.DictReader((out / 'rows.csv').open()))
            self.assertEqual({r['run']: r['meets_joint_slo'] for r in table}, {str(a): 'True', str(b): 'False'})

    def test_rescore_rejects_a_result_from_another_harness_with_a_message(self):
        with tempfile.TemporaryDirectory() as d:
            u = self._run(d, 'unified', 1, [self._ok()] * 10)
            f = next(Path(u).glob('*-measured.json'))
            run = json.loads(f.read_text())
            del run['metadata']['arguments']['max_client_lag_ms']
            f.write_text(json.dumps(run))
            with self.assertRaisesRegex(SystemExit, 'max_client_lag_ms'):
                rescore.load_rows([u])

    def _config(self, **overrides):
        c = json.loads(Path('config.example.json').read_text())
        c.update(context='ctx', instance_type='g7.48xlarge', nodes=['a', 'b'], efa_per_worker=2, gpus_per_worker=4,
                 image='repo/aim345@sha256:' + '0' * 64, router_image='repo/router@sha256:' + '1' * 64)
        return dict(c, **overrides)

    def _read(self, d, c):
        path = Path(d) / 'c.json'
        path.write_text(json.dumps(c))
        return read_config(path)

    def test_mainline_model_rejects_a_tensor_parallel_size_it_cannot_capture(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(self._read(d, self._config())['gpus_per_worker'], 4)
            with self.assertRaisesRegex(AssertionError, 'serves at 1, 2 or 4 GPUs per worker'):
                self._read(d, self._config(gpus_per_worker=8))

    def test_page_size_must_be_a_positive_integer_dividing_the_prefill_chunk(self):
        with tempfile.TemporaryDirectory() as d:
            for page in (0, -1, '64', True):
                with self.assertRaisesRegex(AssertionError, 'page_size_tokens must be a positive integer'):
                    self._read(d, self._config(page_size_tokens=page))
            with self.assertRaisesRegex(AssertionError, 'must be a multiple of page_size_tokens'):
                self._read(d, self._config(page_size_tokens=48))
            self.assertEqual(self._read(d, self._config(page_size_tokens=64))['page_size_tokens'], 64)

    def test_stages_reads_rank_zero_and_reports_the_mean_of_the_difference(self):
        def text(sum_s, count):
            return '\n'.join([
                f'sglang:per_stage_req_latency_seconds_sum{{engine_type="prefill",stage="prefill_bootstrap",tp_rank="0"}} {sum_s}',
                f'sglang:per_stage_req_latency_seconds_count{{engine_type="prefill",stage="prefill_bootstrap",tp_rank="0"}} {count}',
                'sglang:per_stage_req_latency_seconds_sum{engine_type="prefill",stage="prefill_bootstrap",tp_rank="1"} 999',
                'sglang:per_stage_req_latency_seconds_count{engine_type="prefill",stage="prefill_bootstrap",tp_rank="1"} 999'])
        before, after = stages.parse(text(10.0, 4), 'e'), stages.parse(text(16.0, 7), 'e')
        self.assertEqual(before['prefill_bootstrap']['count'], 4)
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as bracketed:
            (Path(bracketed) / '1-measured.json').write_text(json.dumps({'summary': {'planned_calls': 3}}))
            diff = lambda b, a, o=None: stages.diff(argparse.Namespace(before=b, after=a, output=o, ramp=bracketed))
            b, a = Path(d) / 'b.json', Path(d) / 'a.json'
            b.write_text(json.dumps({'e': before}))
            a.write_text(json.dumps({'e': after}))
            o = Path(d) / 'new' / 'o.json'
            diff(b, a, o)
            row = json.loads(o.read_text())[0]
            self.assertEqual(row['observations'], 3)
            self.assertAlmostEqual(row['mean_ms'], 2000.0)
            with self.assertRaisesRegex(SystemExit, 'backwards'):
                diff(a, b)
            a.write_text(json.dumps({'replaced': after}))
            with self.assertRaisesRegex(SystemExit, 'Engines changed'):
                diff(b, a)
            a.write_text(json.dumps({'e': {}}))
            with self.assertRaisesRegex(SystemExit, 'disappeared'):
                diff(b, a)

    def test_stages_refuses_an_empty_ramp_directory(self):
        forward = lambda n: {'prefill_forward': {'engine_type': 'prefill', 'sum_s': float(n), 'count': n}}
        with tempfile.TemporaryDirectory() as d:
            b, a, r = Path(d) / 'b.json', Path(d) / 'a.json', Path(d) / 'ramp'
            b.write_text(json.dumps({'e': forward(4)}))
            a.write_text(json.dumps({'e': forward(7)}))
            r.mkdir()
            with self.assertRaisesRegex(SystemExit, 'no ramp files'), contextlib.redirect_stdout(io.StringIO()):
                stages.diff(argparse.Namespace(before=b, after=a, output=None, ramp=r))

    def test_engine_ports_are_read_from_the_rendered_pods(self):
        c = json.loads(Path('config.example.json').read_text())
        c['nodes'] = ['node-a', 'node-b']
        engines = lambda doc: [x['spec']['template']['spec'] for x in doc['items']
                               if x['kind'] == 'Deployment' and x['metadata']['labels'].get('component') == 'engine']
        self.assertEqual([engine_ports(x) for x in engines(render(c, 'disaggregated'))], [[30000], [30000]])
        c.update(placement='packed', nodes=['node-a'], gpus_per_worker=1, efa_per_worker=1)
        self.assertEqual([engine_ports(x) for x in engines(paired_render(c, 'disaggregated'))], [[30000, 30001]])
        self.assertEqual([engine_ports(x) for x in engines(paired_render(c, 'unified'))], [[30000, 30001]])
        pod = engines(render(dict(c, placement='separate-nodes', nodes=['node-a', 'node-b']), 'disaggregated'))[0]
        args = pod['containers'][0]['args']
        args[args.index('--port') + 1] = '31000'
        self.assertEqual(engine_ports(pod), [31000])

    def test_stages_refuses_a_stage_with_a_missing_or_invalid_series(self):
        series = lambda kind, value: f'sglang:per_stage_req_latency_seconds_{kind}{{engine_type="prefill",stage="prefill_forward",tp_rank="0"}} {value}'
        for text, reason in ((series('count', 3), 'e prefill_forward: /metrics carries only one'),
                             (series('sum', 'NaN') + '\n' + series('count', 3), 'e prefill_forward: invalid sum'),
                             (series('sum', 1.0) + '\n' + series('count', 2.5), 'e prefill_forward: invalid count')):
            with self.assertRaisesRegex(SystemExit, reason):
                stages.parse(text, 'e')
        mismatched = ('sglang:per_stage_req_latency_seconds_sum{engine_type="prefill",stage="prefill_forward",tp_rank="0",model_name="a"} 1\n'
                      'sglang:per_stage_req_latency_seconds_count{engine_type="prefill",stage="prefill_forward",tp_rank="0",model_name="b"} 1')
        with self.assertRaisesRegex(SystemExit, 'only one of the sum and count'):
            stages.parse(mismatched, 'e')

    def test_stages_refuses_an_engine_without_stage_metrics(self):
        series = lambda role: '\n'.join(f'sglang:per_stage_req_latency_seconds_{k}{{engine_type="{role}",stage="s",tp_rank="0"}} 1' for k in ('sum', 'count'))
        texts = {'prefill-0': series('prefill'), 'prefill-1': series('prefill'), 'decode-0': series('decode')}
        with tempfile.TemporaryDirectory() as d, patch('stages.from_config', return_value=texts):
            args = argparse.Namespace(config='unused', output=str(Path(d) / 'snapshot.json'))
            with contextlib.redirect_stdout(io.StringIO()):
                stages.snapshot(args)
            texts['prefill-1'] = ''
            with self.assertRaisesRegex(SystemExit, 'prefill-1: no rank-zero per-stage latency metrics'):
                stages.snapshot(args)

    def test_page_size_is_passed_to_every_engine_only_when_configured(self):
        c = json.loads(Path('config.example.json').read_text())
        c['nodes'] = ['node-a', 'node-b']
        def engine_args(doc):
            found = [x['spec']['template']['spec']['containers'][0]['args'] for x in doc['items']
                     if x['kind'] == 'Deployment' and x['metadata']['labels'].get('component') == 'engine']
            self.assertEqual(len(found), 2, 'expected two engine Deployments')
            return found
        for mode in ('unified', 'disaggregated'):
            for args in engine_args(render(c, mode)):
                self.assertNotIn('--page-size', args)
        c['page_size_tokens'] = 64
        for mode in ('unified', 'disaggregated'):
            for args in engine_args(render(c, mode)):
                self.assertEqual(args[args.index('--page-size') + 1], '64')

    def test_mixed_chunk_flag_is_opt_in(self):
        c = json.loads(Path('config.example.json').read_text())
        c['nodes'] = ['node-a', 'node-b']
        def engine_args(doc):
            return [x['spec']['template']['spec']['containers'][0]['args'] for x in doc['items']
                    if x['kind'] == 'Deployment' and x['metadata']['labels'].get('component') == 'engine']
        for mode in ('unified', 'disaggregated'):
            for args in engine_args(render(c, mode)):
                self.assertNotIn('--enable-mixed-chunk', args)
        c['enable_mixed_chunk'] = True
        for mode in ('unified', 'disaggregated'):
            for args in engine_args(render(c, mode)):
                self.assertIn('--enable-mixed-chunk', args)

    def test_schedule_policy_is_opt_in(self):
        c = json.loads(Path('config.example.json').read_text())
        c['nodes'] = ['node-a', 'node-b']
        def engine_args(doc):
            return [x['spec']['template']['spec']['containers'][0]['args'] for x in doc['items']
                    if x['kind'] == 'Deployment' and x['metadata']['labels'].get('component') == 'engine']
        for mode in ('unified', 'disaggregated'):
            for args in engine_args(render(c, mode)):
                self.assertNotIn('--schedule-policy', args)
        c['schedule_policy'] = 'lpm'
        for mode in ('unified', 'disaggregated'):
            for args in engine_args(render(c, mode)):
                self.assertEqual(args[args.index('--schedule-policy') + 1], 'lpm')

    def test_nixl_backend_is_opt_in_and_defaults_to_libfabric(self):
        c = json.loads(Path('config.example.json').read_text())
        c['nodes'] = ['node-a', 'node-b']
        def engine_env(doc):
            return [{e['name']: e.get('value') for e in x['spec']['template']['spec']['containers'][0]['env']}
                    for x in doc['items'] if x['kind'] == 'Deployment' and x['metadata']['labels'].get('component') == 'engine']
        for mode in ('unified', 'disaggregated'):
            for env in engine_env(render(c, mode)):
                self.assertEqual(env['SGLANG_DISAGGREGATION_NIXL_BACKEND'], 'LIBFABRIC')
                self.assertNotIn('SGLANG_DISAGGREGATION_QUEUE_SIZE', env)
        c['nixl_backend'] = 'UCX'
        c['nixl_queue_size'] = 8
        for mode in ('unified', 'disaggregated'):
            for env in engine_env(render(c, mode)):
                self.assertEqual(env['SGLANG_DISAGGREGATION_NIXL_BACKEND'], 'UCX')
                self.assertEqual(env['SGLANG_DISAGGREGATION_QUEUE_SIZE'], '8')

    def test_extra_env_is_opt_in_and_empty_by_default(self):
        c = json.loads(Path('config.example.json').read_text())
        c['nodes'] = ['node-a', 'node-b']
        def engine_env(doc):
            return [{e['name']: e.get('value') for e in x['spec']['template']['spec']['containers'][0]['env']}
                    for x in doc['items'] if x['kind'] == 'Deployment' and x['metadata']['labels'].get('component') == 'engine']
        # Absent by default: no new key appears, and the fixed entries this lab already sets keep
        # their usual values.
        for mode in ('unified', 'disaggregated'):
            for env in engine_env(render(c, mode)):
                self.assertNotIn('FI_EFA_ENABLE_SHM_TRANSFER', env)
                self.assertEqual(env['FI_PROVIDER'], 'efa')
        # Opt-in: an arbitrary new variable is added for every role.
        c['extra_env'] = {'FI_EFA_ENABLE_SHM_TRANSFER': 0}
        for mode in ('unified', 'disaggregated'):
            for env in engine_env(render(c, mode)):
                self.assertEqual(env['FI_EFA_ENABLE_SHM_TRANSFER'], '0')
        # Also overrides a fixed entry by name, rather than duplicating it, if the key collides.
        c['extra_env'] = {'FI_PROVIDER': 'tcp'}
        for mode in ('unified', 'disaggregated'):
            for env_list in (x['spec']['template']['spec']['containers'][0]['env']
                              for x in render(c, mode)['items']
                              if x['kind'] == 'Deployment' and x['metadata']['labels'].get('component') == 'engine'):
                names = [e['name'] for e in env_list]
                self.assertEqual(names.count('FI_PROVIDER'), 1)
                self.assertEqual({e['name']: e['value'] for e in env_list}['FI_PROVIDER'], 'tcp')

    def test_extra_env_rejects_non_flat_values(self):
        base = json.loads(Path('config.example.json').read_text())
        base.update(context='test', nodes=['a', 'b'], image='img@sha256:' + 'a' * 64,
                    router_image='router@sha256:' + 'b' * 64, instance_type='g7.48xlarge',
                    gpus_per_worker=2, efa_per_worker=1, model_cache_host_path='/mnt/x')
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'config.json'
            path.write_text(json.dumps(base | dict(extra_env={'FI_EFA_ENABLE_SHM_TRANSFER': '0'})))
            read_config(path)  # a flat string-keyed dict is accepted
            for bad in (['not', 'a', 'dict'], {'': '0'}):
                path.write_text(json.dumps(base | dict(extra_env=bad)))
                with self.assertRaises(AssertionError):
                    read_config(path)

    def test_pin_nixl_version_is_opt_in_and_empty_by_default(self):
        c = json.loads(Path('config.example.json').read_text())
        c['nodes'] = ['node-a', 'node-b']

        def engine_command(doc):
            return [x['spec']['template']['spec']['containers'][0]['command'][2]
                    for x in doc['items'] if x['kind'] == 'Deployment' and x['metadata']['labels'].get('component') == 'engine']
        # Absent by default: no pip install step is inserted, and the exec line is unchanged.
        for mode in ('unified', 'disaggregated'):
            for command in engine_command(render(c, mode)):
                self.assertNotIn('pip install', command)
                self.assertIn('&& exec python3 -m sglang.launch_server', command)
        # Opt-in: a pip install step is inserted after verify_image.py's own pinned-version
        # check and before the exec that starts the real server -- never before that check.
        c['pin_nixl_version'] = '1.4.0'
        for mode in ('unified', 'disaggregated'):
            for command in engine_command(render(c, mode)):
                self.assertIn(
                    'python3 /lab/verify_image.py && python3 -m pip install --no-deps '
                    '--force-reinstall nixl==1.4.0 nixl-cu13==1.4.0 && exec python3 -m sglang.launch_server',
                    command)

    def test_pin_nixl_version_rejects_malformed_version(self):
        base = json.loads(Path('config.example.json').read_text())
        base.update(context='test', nodes=['a', 'b'], image='img@sha256:' + 'a' * 64,
                    router_image='router@sha256:' + 'b' * 64, instance_type='g7.48xlarge',
                    gpus_per_worker=2, efa_per_worker=1, model_cache_host_path='/mnt/x')
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'config.json'
            path.write_text(json.dumps(base | dict(pin_nixl_version='1.4.0')))
            read_config(path)  # a bare X.Y.Z version is accepted
            for bad in ('1.4', 'v1.4.0', '1.4.0-rc1', 1.4):
                path.write_text(json.dumps(base | dict(pin_nixl_version=bad)))
                with self.assertRaises(AssertionError):
                    read_config(path)

    def _rate_namespace(self, url, **overrides):
        base = dict(url=url, seed=345, duration_s=0.3, max_tasks=50, timeout_s=5, context_length_tokens=32768,
                    gpus=1, ttft_slo_ms=10**6, tpot_slo_ms=10**6, attainment_fraction=.9, max_client_lag_ms=10**6,
                    architecture='unified', instance_type='test', region='test', evidence_scope='local-mock')
        return argparse.Namespace(**{**base, **overrides})

    def _mock_server(self):
        server = ThreadingHTTPServer(('127.0.0.1', 0), StreamHandler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server, f'http://127.0.0.1:{server.server_port}'

    @staticmethod
    def _shape(output_tokens=4, input_tokens=128):
        return {'name': 'B-test', 'input_tokens': input_tokens, 'output_tokens': output_tokens, 'document_text': 'filler text ' * 32}

    def test_run_rate_cancel_event_stops_issuing_new_calls(self):
        server, url = self._mock_server()
        try:
            shape = self._shape()
            data = {'model': 'm', 'revision': 'r', 'shape': shape, 'tasks': [{'task_id': f't{i}', 'calls': 1} for i in range(20)]}
            a = self._rate_namespace(url, duration_s=1.0, max_tasks=200)
            cancel = asyncio.Event()
            cancel.set()
            result = asyncio.run(run_rate(data, a, 50, 'measured', self.tok, cancel_event=cancel))
            self.assertTrue(result['requests'])
            self.assertTrue(all(not r['sent'] and not r['ok'] for r in result['requests']))
            self.assertEqual(result['summary']['sent_calls'], 0)
        finally:
            server.shutdown()
            server.server_close()

    def test_run_rate_shards_stride_task_identities_without_overlap(self):
        server, url = self._mock_server()
        try:
            shape = self._shape()
            data = {'model': 'm', 'revision': 'r', 'shape': shape, 'tasks': [{'task_id': f't{i}', 'calls': 1} for i in range(6)]}
            seen = set()
            for shard_index in (0, 1):
                a = self._rate_namespace(url, duration_s=0.4, max_tasks=10, shard_index=shard_index, shard_count=2)
                result = asyncio.run(run_rate(data, a, 10, 'measured', self.tok))
                task_ids = {r['task_id'] for r in result['requests']}
                self.assertTrue(task_ids & seen == set(), 'shards must not draw the same task identity')
                seen |= task_ids
        finally:
            server.shutdown()
            server.server_close()

    def test_chunks_are_recorded_and_pooled_into_itl_percentiles(self):
        server, url = self._mock_server()
        try:
            shape = self._shape()
            data = {'model': 'm', 'revision': 'r', 'shape': shape, 'tasks': [{'task_id': 't0', 'calls': 1}]}
            a = self._rate_namespace(url, duration_s=0.1, max_tasks=5)
            result = asyncio.run(run_rate(data, a, 10, 'measured', self.tok))
            row = result['requests'][0]
            self.assertTrue(row['ok'])
            self.assertEqual([c['n'] for c in row['chunks']], [1, 3])
            self.assertGreater(row['chunks'][-1]['t_ms'], row['chunks'][0]['t_ms'])
            self.assertIsNotNone(result['summary']['itl_ms_p99'])
            self.assertEqual(result['summary']['tpot_metric'], 'mean')
        finally:
            server.shutdown()
            server.server_close()

    def test_itl_p99_mode_is_a_pooled_rate_level_gate_and_refuses_without_chunks(self):
        # Per-chunk gaps: (5-0)/1=5ms, (25-5)/1=20ms; pooled p99 of [5, 20] is 20ms.
        good = {'ok': True, 'sent': True, 'client_launch_lag_ms': 0, 'ttft_ms': 10, 'tpot_ms': 999,
                'chunks': [{'t_ms': 0, 'n': 1}, {'t_ms': 5, 'n': 1}, {'t_ms': 25, 'n': 1}]}
        rows = [dict(good) for _ in range(5)]
        under = summarize(rows, 1, 1, 10**6, 20, .9, 1, 100, tpot_metric='itl-p99')
        self.assertEqual(under['itl_ms_p99'], 20)
        self.assertTrue(under['meets_joint_slo'])
        over = summarize(rows, 1, 1, 10**6, 19, .9, 1, 100, tpot_metric='itl-p99')
        self.assertFalse(over['meets_joint_slo'])
        self.assertEqual(over['slo_attainment_fraction'], 1.0, 'TTFT attainment side is unchanged by the tpot_metric')
        without_chunks = [{k: v for k, v in r.items() if k != 'chunks'} for r in rows]
        with self.assertRaisesRegex(SystemExit, 'needs per-chunk recording'):
            summarize(without_chunks, 1, 1, 10**6, 20, .9, 1, 100, tpot_metric='itl-p99')

    def test_worst_itl_ms_is_the_single_raw_gap_not_divided_by_token_count(self):
        # A chunk that batches several tokens after a long stall is one real pause, not a shorter
        # one per token: 20ms raw gap stays 20ms even though 3 tokens arrived in it.
        record = {'chunks': [{'t_ms': 0, 'n': 1}, {'t_ms': 5, 'n': 1}, {'t_ms': 25, 'n': 3}]}
        self.assertEqual(worst_itl_ms(record), 20)
        self.assertIsNone(worst_itl_ms({'chunks': [{'t_ms': 0, 'n': 1}]}), 'one chunk has no gap to measure')
        self.assertIsNone(worst_itl_ms({'chunks': []}))
        self.assertIsNone(worst_itl_ms({}))

    def test_worst_itl_mode_is_a_per_call_gate_on_each_calls_own_single_worst_gap(self):
        # Four calls whose own worst gap is 20ms (same chunks as the itl-p99 test above), and one
        # outlier whose own worst gap is 90ms despite a fine average tpot_ms -- worst-itl is a
        # per-call test, unlike itl-p99's pooled one, so this one call alone should fail it.
        fine = {'ok': True, 'sent': True, 'client_launch_lag_ms': 0, 'ttft_ms': 10, 'tpot_ms': 5,
                'chunks': [{'t_ms': 0, 'n': 1}, {'t_ms': 5, 'n': 1}, {'t_ms': 25, 'n': 1}]}
        stutter = {'ok': True, 'sent': True, 'client_launch_lag_ms': 0, 'ttft_ms': 10, 'tpot_ms': 5,
                   'chunks': [{'t_ms': 0, 'n': 1}, {'t_ms': 5, 'n': 1}, {'t_ms': 95, 'n': 1}]}
        rows = [dict(fine) for _ in range(4)] + [dict(stutter)]
        under = summarize(rows, 1, 1, 10**6, 20, .9, 1, 100, tpot_metric='worst-itl')
        self.assertEqual(under['worst_itl_ms_p99'], 90)
        self.assertAlmostEqual(under['slo_attainment_fraction'], 0.8, msg='the one 90ms-stutter call fails on its own, even with mean tpot_ms=5 and a fine pooled itl-p99')
        self.assertFalse(under['meets_joint_slo'], '0.8 attainment is below the 0.9 target')
        all_fine = summarize([dict(fine) for _ in range(5)], 1, 1, 10**6, 20, .9, 1, 100, tpot_metric='worst-itl')
        self.assertTrue(all_fine['meets_joint_slo'])
        without_chunks = [{k: v for k, v in r.items() if k != 'chunks'} for r in rows]
        with self.assertRaisesRegex(SystemExit, 'needs per-chunk recording'):
            summarize(without_chunks, 1, 1, 10**6, 20, .9, 1, 100, tpot_metric='worst-itl')

    def test_ttft_metric_moves_ttft_from_a_per_call_gate_to_a_pooled_rate_level_one(self):
        # Four fast-TTFT calls and one slow one (999ms); per-call default fails the slow one alone
        # (attainment 0.8, below 0.9), but a loose per-call bound of 1000ms would pass every call.
        # The pooled p95 of [10,10,10,10,999] is 999ms, so a 1000ms bound passes under p95 too, while
        # a 500ms bound fails the pooled check even though 4/5 calls individually clear 500ms.
        fast = {'ok': True, 'sent': True, 'client_launch_lag_ms': 0, 'ttft_ms': 10, 'tpot_ms': 5, 'chunks': []}
        slow = {'ok': True, 'sent': True, 'client_launch_lag_ms': 0, 'ttft_ms': 999, 'tpot_ms': 5, 'chunks': []}
        rows = [dict(fast) for _ in range(4)] + [dict(slow)]
        per_call = summarize(rows, 1, 1, 500, 10**6, .9, 1, 100, ttft_metric='per-call')
        self.assertAlmostEqual(per_call['slo_attainment_fraction'], 0.8)
        self.assertFalse(per_call['meets_joint_slo'])
        pooled_loose = summarize(rows, 1, 1, 1000, 10**6, .9, 1, 100, ttft_metric='p95')
        self.assertEqual(pooled_loose['slo_attainment_fraction'], 1.0, 'TTFT no longer gates one call at a time under a pooled metric')
        self.assertTrue(pooled_loose['meets_joint_slo'])
        pooled_tight = summarize(rows, 1, 1, 500, 10**6, .9, 1, 100, ttft_metric='p95')
        self.assertFalse(pooled_tight['meets_joint_slo'], 'the pooled p95 (999ms) itself fails a 500ms bound')
        self.assertEqual(pooled_tight['p95_ttft_ms_success_only'], 999)
        self.assertEqual(pooled_tight['p99_ttft_ms_success_only'], 999)

    def _harness(self, out, run_id, status, rates):
        harness = Path(out) / '.labharness' / run_id
        harness.mkdir(parents=True, exist_ok=True)
        (harness / 'manifest.json').write_text(json.dumps({'rates': rates}))
        (harness / 'run_status.json').write_text(json.dumps({'status': status}))

    def test_multi_segment_plan_requires_a_complete_run_covering_its_manifest_gap_free(self):
        with tempfile.TemporaryDirectory() as d:
            run = None
            for rate in (1, 2):
                run = self._run(d, 'unified', rate, [self._ok()] * 10, rates=[1, 2])
            for f in run.glob('*-measured.json'):
                body = json.loads(f.read_text())
                body['metadata']['arguments'].update(run_id='run-x', segment_id='seg-1')
                f.write_text(json.dumps(body))
            rows = rescore.load_rows([run])
            self.assertEqual({r['offered'] for r in rows}, {1, 2})
            with self.assertRaisesRegex(SystemExit, r'no .*\.labharness'):
                rescore.multi_segment_plan(str(run), rows)
            self._harness(run, 'run-x', 'INTERRUPTED', [1, 2])
            with self.assertRaisesRegex(SystemExit, "'INTERRUPTED', not COMPLETE"):
                rescore.multi_segment_plan(str(run), rows)
            self._harness(run, 'run-x', 'COMPLETE', [2, 4])
            with self.assertRaisesRegex(SystemExit, 'measured 1, 2, outside its own planned 2, 4'):
                rescore.multi_segment_plan(str(run), rows)
            self._harness(run, 'run-x', 'COMPLETE', [1, 2])
            self.assertEqual(rescore.multi_segment_plan(str(run), rows), [1, 2])
            rescore.check_runs(rows)

    def test_check_runs_falls_back_to_ramp_id_when_no_run_id_is_present(self):
        with tempfile.TemporaryDirectory() as d:
            u = self._run(d, 'unified', 1, [self._ok()] * 10, rates=[1, 2], ramp_id='a')
            self.assertIsNone(rescore.load_rows([u])[0]['run_id'])
            with self.assertRaisesRegex(SystemExit, 'was interrupted'):
                rescore.check_runs(rescore.load_rows([u]))


if __name__ == '__main__':
    unittest.main()
