"""Allocation isolation and packed process addressing, without GPU claims."""
import json
from pathlib import Path
import unittest
import subprocess
import sys
import tempfile
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

    def test_packed_engine_command_carries_opt_in_nixl_pin(self):
        # Absent by default: the packed engine command is unchanged.
        doc = paired_render(self.b, 'disaggregated')
        command = next(x for x in doc['items'] if x['kind'] == 'Deployment' and x['metadata']['name'] == 'engines'
                        )['spec']['template']['spec']['containers'][0]['command'][2]
        self.assertNotIn('pip install', command)
        self.assertIn('&& exec python3 /lab/packed.py', command)
        # Opt-in: inserted after verify_image.py's own check, before the exec into packed.py.
        pinned = dict(self.b, pin_nixl_version='1.4.0')
        doc = paired_render(pinned, 'disaggregated')
        command = next(x for x in doc['items'] if x['kind'] == 'Deployment' and x['metadata']['name'] == 'engines'
                        )['spec']['template']['spec']['containers'][0]['command'][2]
        self.assertIn(
            'python3 /lab/verify_image.py && python3 -m pip install --no-deps '
            '--force-reinstall nixl==1.4.0 nixl-cu13==1.4.0 && exec python3 /lab/packed.py',
            command)

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

    def test_check_pair_accepts_an_asymmetric_ratio_with_the_same_tp_and_total(self):
        # unified: TP2 x 4 (8 GPUs) vs disaggregated: prefill TP2 x3 + decode TP2 x1 (8 GPUs, 6:2).
        unified = dict(self.a, gpus_per_worker=2, unified_replicas=4)
        split = dict(self.b, gpus_per_worker=2, prefill_workers=3, decode_workers=1)
        self.assertEqual(check_pair(unified, split), 8)

    def test_check_pair_rejects_a_ratio_whose_total_or_tp_does_not_match(self):
        unified = dict(self.a, gpus_per_worker=2, unified_replicas=4)  # 8 GPUs, TP2
        mismatched_total = dict(self.b, gpus_per_worker=2, prefill_workers=2, decode_workers=1)  # 6 GPUs
        with self.assertRaisesRegex(ValueError, 'GPU totals differ'):
            check_pair(unified, mismatched_total)
        mismatched_tp = dict(self.b, gpus_per_worker=2, prefill_workers=3, decode_workers=1, decode_gpus_per_worker=2, prefill_gpus_per_worker=1)
        # prefill at TP1, decode at TP2 inside the same disaggregated stack: not one uniform TP.
        with self.assertRaisesRegex(ValueError, 'one uniform tensor-parallel size'):
            check_pair(unified, mismatched_tp)

    def test_separate_node_path_preserves_two_worker_deployments(self):
        a = dict(self.a, placement='separate-nodes', nodes=['a','b'], gpus_per_worker=2)
        b = dict(a, namespace='aim345-test-b', nodes=['c','d'])
        self.assertEqual(check_pair(a,b),4)
        workers = [x for x in paired_render(b,'disaggregated')['items'] if x['kind']=='Deployment' and x['metadata']['labels'].get('component')=='engine']
        self.assertEqual(len(workers),2)
        self.assertEqual([x['spec']['template']['spec']['nodeSelector']['kubernetes.io/hostname'] for x in workers], ['c','d'])


if __name__ == '__main__': unittest.main()
