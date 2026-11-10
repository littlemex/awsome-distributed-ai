"""Allocation isolation and packed process addressing, without GPU claims."""
import copy
import json
from pathlib import Path
import unittest
import subprocess
import sys
import tempfile
from unittest import mock
import argparse
import os
import paired
import deployment
from paired import check_pair, paired_render


class PairedTests(unittest.TestCase):
    def setUp(self):
        self.a = json.loads(Path('config.example.json').read_text())
        self.a.update(namespace='aim345-test-a', nodes=['node-a'], placement='packed', gpus_per_worker=1, efa_per_worker=1)
        self.b = dict(self.a, namespace='aim345-test-b', nodes=['node-b'])

    def test_pair_rejects_overlap_or_changed_budget_and_model(self):
        self.assertEqual(check_pair(self.a, self.b), 2)
        for update in [dict(nodes=['node-a']), dict(namespace=self.a['namespace']), dict(gpus_per_worker=2), dict(model_revision='a'*40)]:
            with self.assertRaises(ValueError):
                check_pair(self.a, self.b | update)

    def test_packed_processes_use_disjoint_gpus_and_distinct_service_ports(self):
        docs = [paired_render(c, mode) for c, mode in [(self.a, 'unified'), (self.b, 'disaggregated')]]
        priorities = []
        for c, doc in zip([self.a, self.b], docs):
            self.assertEqual(next(x['metadata']['name'] for x in doc['items'] if x['kind']=='Namespace'), c['namespace'])
            priorities.append(next(x['metadata']['name'] for x in doc['items'] if x['kind']=='PriorityClass'))
            deployments = {x['metadata']['name']: x for x in doc['items'] if x['kind']=='Deployment'}
            pod = deployments['engines']['spec']['template']['spec']
            container = pod['containers'][0]
            self.assertFalse(pod['hostNetwork'])
            self.assertEqual(container['resources']['requests']['nvidia.com/gpu'], 2)
            self.assertEqual(container['resources']['requests']['vpc.amazonaws.com/efa'], 1)
            plans = json.loads(next(x['value'] for x in container['env'] if x['name']=='AIM345_ENGINE_PLANS'))
            self.assertEqual([x['env']['CUDA_VISIBLE_DEVICES'] for x in plans], ['0', '1'])
            self.assertEqual([x['args'][x['args'].index('--port')+1] for x in plans], ['30000','30001'])
            services = [x for x in doc['items'] if x['kind']=='Service' and x['metadata']['name']!='router']
            self.assertEqual([x['spec']['ports'][0]['targetPort'] for x in services], [30000,30001])
            router = deployments['router']['spec']['template']['spec']
            self.assertEqual(router['nodeSelector']['kubernetes.io/hostname'],c['nodes'][0])
            self.assertTrue(any(':30001' in x for x in router['containers'][0]['args']))
            self.assertEqual(pod['preemptionPolicy'],'Never')
        self.assertNotEqual(*priorities)

    def test_preparation_enforces_reduced_node_budget(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inventory = root / 'nodes.json'
            inventory.write_text(json.dumps({'items': [
                {'metadata': {'name': name, 'labels': {'node.kubernetes.io/instance-type': 'g7.48xlarge'}},
                 'spec': {}, 'status': {'allocatable': {'nvidia.com/gpu': '8', 'vpc.amazonaws.com/efa': '2', 'cpu': '191450m', 'memory': '709364952Ki'}}}
                for name in ('node-a', 'node-b')]}))
            cmd = [sys.executable, 'facilitator/prepare-config.py', '--context', 'test',
                   '--nodes', 'node-a,node-b', '--engine-image', 'engine@sha256:'+'a'*64,
                   '--router-image', 'router@sha256:'+'b'*64, '--inventory', str(inventory),
                   '--gpus-per-stack', '4', '--efas-per-stack', '1', '--cpu-per-stack', '96',
                   '--memory-gib-per-stack', '384', '--model-cache-host-path', '/mnt/nvme/models']
            subprocess.run(cmd + ['--output', str(root/'valid')], check=True, capture_output=True)
            for mode in ('unified', 'disaggregated'):
                config = json.loads((root/'valid'/f'config.{mode}.json').read_text())
                pods = [x['spec']['template']['spec'] for x in paired_render(config, mode)['items'] if x['kind']=='Deployment']
                limits = [c['resources']['limits'] for pod in pods for c in pod['containers']]
                self.assertEqual(sum(int(c.get('nvidia.com/gpu', 0)) for c in limits), 4)
                self.assertEqual(sum(int(c.get('vpc.amazonaws.com/efa', 0)) for c in limits), 1)
                self.assertEqual(sum(int(c['cpu']) for c in limits), 96)
                self.assertEqual(sum(int(c['memory'].removesuffix('Gi')) for c in limits), 384)
            for flag, value in [('--gpus-per-stack', '10'), ('--gpus-per-stack', '3'), ('--efas-per-stack', '3'), ('--cpu-per-stack', '4'), ('--memory-gib-per-stack', '8'), ('--memory-gib-per-stack', '740'), ('--cpu-per-stack', '192')]:
                rejected = cmd.copy()
                rejected[rejected.index(flag)+1] = value
                result = subprocess.run(rejected + ['--output', str(root/'invalid')], capture_output=True)
                self.assertNotEqual(result.returncode, 0)

    def test_optional_profile_rejects_secondary_shape_and_mainline_model(self):
        from deployment import read_config, render
        c = json.loads(Path('config.v4.example.json').read_text())
        c.update(context='test', nodes=['a','b'], image='v4@sha256:'+'a'*64,
                 router_image='router@sha256:'+'b'*64)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'config.json'
            path.write_text(json.dumps(c))
            qualified = read_config(path)
            workers = [x for x in render(qualified, 'disaggregated')['items'] if x['kind']=='Deployment' and x['metadata']['labels'].get('component')=='engine']
            self.assertEqual({x['spec']['template']['spec']['nodeSelector']['kubernetes.io/hostname'] for x in workers}, {'a','b'})
            for update in [dict(instance_type='g7.24xlarge', gpus_per_worker=4, efa_per_worker=1),
                           dict(placement='packed'), dict(engine_profile='mainline'),
                           dict(model_id='deepseek-ai/DeepSeek-V2-Lite-Chat')]:
                path.write_text(json.dumps(c | update))
                with self.assertRaises(AssertionError):
                    read_config(path)

    def test_separate_node_path_preserves_two_worker_deployments(self):
        a = dict(self.a, placement='separate-nodes', nodes=['a','b'], gpus_per_worker=2)
        b = dict(a, namespace='aim345-test-b', nodes=['c','d'])
        self.assertEqual(check_pair(a,b),4)
        workers = [x for x in paired_render(b,'disaggregated')['items'] if x['kind']=='Deployment' and x['metadata']['labels'].get('component')=='engine']
        self.assertEqual(len(workers),2)
        self.assertEqual([x['spec']['template']['spec']['nodeSelector']['kubernetes.io/hostname'] for x in workers], ['c','d'])


class MonitoringLabelTests(unittest.TestCase):
    """monitoring_namespace adds the scrape label, the architecture label and the port name "metrics" to engine
    Services, and changes nothing else in any render; without it engine Services keep the port name "http"."""
    M = 'in-cluster-monitoring'

    def setUp(self):
        self.a = json.loads(Path('config.example.json').read_text())
        self.a.update(namespace='aim345-test-a', nodes=['node-a'], placement='packed', gpus_per_worker=1, efa_per_worker=1)
        self.b = dict(self.a, namespace='aim345-test-b', nodes=['node-b'])
        sep = dict(self.a, placement='separate-nodes', nodes=['node-a', 'node-b'], gpus_per_worker=2)
        self.cases = [('packed unified', lambda c: paired_render(c, 'unified'), self.a, {'unified-0', 'unified-1'}, 'unified'),
                      ('packed disaggregated', lambda c: paired_render(c, 'disaggregated'), self.b, {'prefill-0', 'decode-0'}, 'disaggregated'),
                      ('separate-nodes unified', lambda c: deployment.render(c, 'unified'), sep, {'unified-0', 'unified-1'}, 'unified'),
                      ('separate-nodes disaggregated', lambda c: deployment.render(c, 'disaggregated'), sep, {'prefill-0', 'decode-0'}, 'disaggregated'),
                      ('separate-nodes scaled', lambda c: deployment.render(c, 'disaggregated', 2), dict(sep, nodes=['node-a', 'node-b', 'node-c']),
                       {'prefill-0', 'decode-0', 'prefill-1'}, 'disaggregated-scaled')]

    def test_only_engine_services_change_and_only_by_the_two_labels_and_the_port_name(self):
        for label, render, c, engines, architecture in self.cases:
            plain, monitored = render(c), render(dict(c, monitoring_namespace=self.M))
            self.assertEqual(len(plain['items']), len(monitored['items']), label)
            changed = set()
            for before, after in zip(plain['items'], monitored['items']):
                if before == after:
                    continue
                self.assertEqual((before['kind'], before['metadata']['name']), (after['kind'], after['metadata']['name']), label)
                self.assertEqual(before['kind'], 'Service', label)
                expected = copy.deepcopy(before)
                expected['metadata']['labels'] |= {deployment.SCRAPE_LABEL: self.M, deployment.ARCH_LABEL: architecture}
                expected['spec']['ports'][0]['name'] = 'metrics'
                self.assertEqual(after, expected, label)
                changed.add(before['metadata']['name'])
            self.assertEqual(changed, engines, label)
            for x in plain['items']:
                if x['kind'] == 'Service' and x['metadata']['name'] in engines:
                    self.assertEqual(x['spec']['ports'][0]['name'], 'http', label)
                    self.assertNotIn(deployment.SCRAPE_LABEL, x['metadata']['labels'], label)

    def test_an_invalid_monitoring_namespace_is_refused_and_the_pair_must_agree(self):
        filled = dict(self.a, context='ctx', image='engine@sha256:' + 'a' * 64, router_image='router@sha256:' + 'b' * 64, instance_type='g7.48xlarge')
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'c.json'
            path.write_text(json.dumps(dict(filled, monitoring_namespace=self.M)))
            self.assertEqual(deployment.read_config(path)['monitoring_namespace'], self.M)
            for bad in ('Not_A_Namespace', '', '-x', 'a' * 64, 5):
                path.write_text(json.dumps(dict(filled, monitoring_namespace=bad)))
                with self.assertRaisesRegex(AssertionError, 'monitoring_namespace', msg=repr(bad)):
                    deployment.read_config(path)
        with self.assertRaises(ValueError):
            check_pair(dict(self.a, monitoring_namespace=self.M), self.b)
        self.assertEqual(check_pair(dict(self.a, monitoring_namespace=self.M), dict(self.b, monitoring_namespace=self.M)), 2)

    def test_preparation_writes_the_monitoring_namespace_into_both_configurations(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inventory = root / 'nodes.json'
            inventory.write_text(json.dumps({'items': [
                {'metadata': {'name': name, 'labels': {'node.kubernetes.io/instance-type': 'g7.48xlarge'}},
                 'spec': {}, 'status': {'allocatable': {'nvidia.com/gpu': '8', 'vpc.amazonaws.com/efa': '2', 'cpu': '191450m', 'memory': '709364952Ki'}}}
                for name in ('node-a', 'node-b')]}))
            cmd = [sys.executable, 'facilitator/prepare-config.py', '--context', 'test', '--nodes', 'node-a,node-b',
                   '--engine-image', 'engine@sha256:' + 'a' * 64, '--router-image', 'router@sha256:' + 'b' * 64,
                   '--inventory', str(inventory), '--model-cache-host-path', '/mnt/nvme/models']
            subprocess.run(cmd + ['--output', str(root / 'plain')], check=True, capture_output=True)
            subprocess.run(cmd + ['--output', str(root / 'monitored'), '--monitoring-namespace', self.M], check=True, capture_output=True)
            for bad in ('', 'Not_A_Namespace'):
                r = subprocess.run(cmd + ['--output', str(root / 'bad'), '--monitoring-namespace', bad], capture_output=True, text=True)
                self.assertNotEqual(r.returncode, 0, repr(bad))
                self.assertIn('--monitoring-namespace must be a valid Kubernetes namespace name', r.stderr)
                self.assertFalse((root / 'bad').exists(), repr(bad))
            for mode in ('unified', 'disaggregated'):
                plain = json.loads((root / 'plain' / f'config.{mode}.json').read_text())
                monitored = json.loads((root / 'monitored' / f'config.{mode}.json').read_text())
                self.assertNotIn('monitoring_namespace', plain)
                self.assertEqual(monitored, dict(plain, monitoring_namespace=self.M))


class WaitTests(unittest.TestCase):
    def setUp(self):
        self.a = json.loads(Path('config.example.json').read_text())
        self.a.update(namespace='aim345-test-a', nodes=['node-a'], placement='packed', gpus_per_worker=1, efa_per_worker=1)
        self.b = dict(self.a, namespace='aim345-test-b', nodes=['node-b'])
        self.items = paired_render(self.a, 'unified')['items'] + paired_render(self.b, 'disaggregated')['items']

    def ok(self, calls):
        return lambda cmd, **k: calls.append(cmd) or subprocess.CompletedProcess(cmd, 0, '', '')

    def test_wait_waits_for_every_rendered_deployment_in_its_namespace_in_render_order(self):
        calls = []
        with mock.patch.object(paired.subprocess, 'run', self.ok(calls)):
            waited = paired.wait_deployments('ctx', self.items, 60)
        self.assertEqual(waited, [(x['metadata']['namespace'], x['metadata']['name']) for x in self.items if x['kind'] == 'Deployment'])
        self.assertEqual(sorted(waited), sorted([('aim345-test-a', 'engines'), ('aim345-test-a', 'router'),
                                                 ('aim345-test-b', 'engines'), ('aim345-test-b', 'router')]))
        self.assertEqual(calls, [['kubectl', '--context', 'ctx', '-n', ns, 'wait', '--for=condition=Available', f'deployment/{name}', '--timeout=60s']
                                 for ns, name in waited])

    def test_the_separate_nodes_render_names_each_worker(self):
        sep = dict(self.a, placement='separate-nodes', nodes=['node-a', 'node-b'], gpus_per_worker=2)
        calls = []
        with mock.patch.object(paired.subprocess, 'run', self.ok(calls)):
            waited = paired.wait_deployments('ctx', deployment.render(sep, 'disaggregated')['items'], 60)
        self.assertEqual(sorted(name for _, name in waited), ['decode-0', 'prefill-0', 'router'])

    def test_the_first_deployment_that_does_not_become_available_stops_the_wait_and_is_named(self):
        calls = []
        def run(cmd, **kwargs):
            calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 1, '', 'error: timed out waiting for the condition\n')
        with mock.patch.object(paired.subprocess, 'run', run), self.assertRaises(SystemExit) as e:
            paired.wait_deployments('ctx', self.items, 60)
        first = next(x for x in self.items if x['kind'] == 'Deployment')
        self.assertIn(f"deployment {first['metadata']['namespace']}/{first['metadata']['name']} is not Available (waited up to 60 s): error: timed out",
                      str(e.exception))
        self.assertEqual(len(calls), 1)

    def test_a_deployment_that_does_not_exist_is_named_with_kubectls_message(self):
        missing = 'Error from server (NotFound): deployments.apps "engines" not found\n'
        with mock.patch.object(paired.subprocess, 'run', lambda cmd, **k: subprocess.CompletedProcess(cmd, 1, '', missing)), \
                self.assertRaises(SystemExit) as e:
            paired.wait_deployments('ctx', self.items, 60)
        self.assertIn('is not Available (waited up to 60 s): Error from server (NotFound)', str(e.exception))

    def test_the_timeout_is_a_whole_number_of_at_least_one(self):
        self.assertEqual((paired.positive_int('1'), paired.positive_int('1800')), (1, 1800))
        for bad in ('0', '-1', '01', '1.5', ' 60', '60s', '', '\uff11'):
            with self.assertRaises(argparse.ArgumentTypeError, msg=repr(bad)):
                paired.positive_int(bad)

    def test_the_wait_command_exits_with_the_message_and_no_traceback(self):
        with tempfile.TemporaryDirectory() as d:
            paths = [Path(d) / 'u.json', Path(d) / 's.json']
            filled = dict(context='ctx', image='engine@sha256:' + 'a' * 64, router_image='router@sha256:' + 'b' * 64, instance_type='g7.48xlarge')
            for path, c in zip(paths, (self.a, self.b)):
                path.write_text(json.dumps(dict(c, **filled)))
            fake = Path(d) / 'kubectl'
            fake.write_text('#!/bin/sh\necho "error: timed out waiting for the condition" >&2\nexit 1\n')
            fake.chmod(0o755)
            env = dict(os.environ, PATH=f"{d}{os.pathsep}{os.environ['PATH']}")
            r = subprocess.run([sys.executable, 'paired.py', 'wait', '--unified-config', str(paths[0]), '--disaggregated-config', str(paths[1]),
                                '--output', str(Path(d) / 'out'), '--timeout', '5'], capture_output=True, text=True, env=env)
            self.assertEqual(r.returncode, 1, r.stderr)
            self.assertIn('is not Available (waited up to 5 s): error: timed out waiting for the condition', r.stderr)
            self.assertNotIn('Traceback', r.stderr)


if __name__ == '__main__': unittest.main()
