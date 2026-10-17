"""Deploy, verify, or remove simultaneous stacks on separate equal allocations."""
import argparse
import copy
import json
from pathlib import Path
from deployment import OWNER, effective_tps, kubectl, nixl_pin_shell_prefix, owned_namespace, read_config, render, total_gpus

MATCHED = ('context', 'image', 'router_image', 'instance_type', 'efa_per_worker',
           'model_id', 'model_revision', 'context_length_tokens',
           'chunked_prefill_size_tokens', 'disable_prefix_cache', 'page_size_tokens', 'placement', 'packed_cpu', 'packed_memory')


def check_pair(unified, split):
    """`unified` and `split` must be built so the comparison between them is apples to apples:
    the same tensor-parallel size throughout each stack (deployment.effective_tps), the same TP
    between the two stacks, and the same total GPU budget (deployment.total_gpus). `gpus_per_worker`
    itself is deliberately not in MATCHED any more -- a per-role override
    (`prefill_gpus_per_worker`, etc.) can change what a worker actually runs at without changing
    that base field, so the base field alone can no longer prove the stacks match; the two checks
    below do, and name which one failed and with what numbers (DESIGN-RATIONALE.md, "prefill:decode
    GPU ratio")."""
    for key in MATCHED:
        if unified.get(key) != split.get(key):
            raise ValueError(f'Paired configuration mismatch: {key}')
    if unified['namespace'] == split['namespace']:
        raise ValueError('Paired stacks need different namespaces')
    if set(unified['nodes']) & set(split['nodes']):
        raise ValueError('Paired stacks must use disjoint node sets')
    expected = 1 if unified.get('placement') == 'packed' else 2
    if len(unified['nodes']) != expected or len(split['nodes']) != expected:
        raise ValueError(f'This paired placement requires {expected} nodes per stack')
    u_tps, s_tps = effective_tps(unified, 'unified'), effective_tps(split, 'disaggregated')
    if len(u_tps) != 1 or len(s_tps) != 1:
        raise ValueError(f'Each stack must run one uniform tensor-parallel size across its own roles: unified={sorted(u_tps)}, disaggregated={sorted(s_tps)}')
    if u_tps != s_tps:
        raise ValueError(f'Paired stacks must run the same tensor-parallel size: unified={u_tps}, disaggregated={s_tps}')
    u_gpus, s_gpus = total_gpus(unified, 'unified'), total_gpus(split, 'disaggregated')
    if u_gpus != s_gpus:
        raise ValueError(f'Paired GPU totals differ: unified={u_gpus}, disaggregated={s_gpus}')
    return u_gpus


def paired_render(c, mode):
    namespace = {'apiVersion': 'v1', 'kind': 'Namespace', 'metadata': {'name': c['namespace'], 'labels': OWNER}}
    if c.get('placement') != 'packed':
        doc = render(c, mode)
        doc['items'].insert(0, namespace)
        return doc
    # Reuse exactly the sequential engine options, then give each process its
    # own GPUs and listening port. One pod requests the EFA device once.
    expanded = copy.deepcopy(c)
    expanded['nodes'] = [c['nodes'][0], c['nodes'][0]]
    doc = render(expanded, mode)
    engines = [x for x in doc['items'] if x['kind'] == 'Deployment' and x['metadata']['labels'].get('component') == 'engine']
    packed = copy.deepcopy(engines[0])
    packed['metadata']['name'] = 'engines'
    packed['metadata']['labels'] = OWNER | {'app': 'engines', 'component': 'engine'}
    packed['spec']['selector']['matchLabels'] = {'app': 'engines'}
    template = packed['spec']['template']
    template['metadata']['labels'] = packed['metadata']['labels']
    spec = template['spec']
    spec['hostNetwork'] = False
    spec['dnsPolicy'] = 'ClusterFirst'
    engine = spec['containers'][0]
    plans = []
    for index, item in enumerate(engines):
        original = item['spec']['template']['spec']['containers'][0]
        args = original['args'][:]
        args[args.index('--port') + 1] = str(30000 + index)
        env = {x['name']: x['value'] for x in original['env']}
        env['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in range(index*c['gpus_per_worker'], (index+1)*c['gpus_per_worker']))
        plans.append({'role': item['metadata']['labels']['role'], 'args': args, 'env': env})
    descriptor_logging_prefix = ''
    if c.get('patch_descriptor_logging'):
        # Investigation-only, opt-in, uncommitted: upgrades conn.py's own existing debug line
        # (len(src_addrs) is the layer count, not the real post-grouping descriptor count) to
        # also log the real count (src_reqs.shape[0]) at INFO level, matching
        # ~/tmp/riv/issue/kvx/RESULTS.md section 5's own method exactly. No image rebuild.
        # Implemented as a base64-encoded Python patch script (not raw sed) to avoid shell/JSON
        # quoting ambiguity around the multi-line target and the f-string's own braces/quotes.
        import base64
        _old = (
            'logger.debug(\n'
            '            f"len(src_addrs): before group: {len(prefill_data_indices)}, after group: {len(src_addrs)}"\n'
            '        )'
        )
        _new = (
            'logger.info(\n'
            '            f"len(src_addrs): before group: {len(prefill_data_indices)}, after group: {len(src_addrs)}, real_descriptor_count: {src_reqs.shape[0]}"\n'
            '        )'
        )
        _patch_py = (
            'import pathlib\n'
            'p = pathlib.Path("/sgl-workspace/sglang/python/sglang/srt/disaggregation/nixl/conn.py")\n'
            's = p.read_text()\n'
            'old = ' + repr(_old) + '\n'
            'new = ' + repr(_new) + '\n'
            'assert s.count(old) == 1, f"expected exactly 1 occurrence, found {s.count(old)}"\n'
            'p.write_text(s.replace(old, new))\n'
            'print("descriptor logging patch applied")\n'
        )
        _b64 = base64.b64encode(_patch_py.encode()).decode()
        descriptor_logging_prefix = f"python3 -c \"import base64; exec(base64.b64decode('{_b64}').decode())\" && "
    dispatch_logging_prefix = ''
    if c.get('patch_dispatch_logging'):
        # Investigation-only (OPEN-ITEMS-NODE4): log each extend (prefill) forward once per batch
        # (layer 0, TP rank 0) with wall-clock time, rids, prefix and new-token lengths, and the
        # attention method chosen. Decode steps replay CUDA graphs and are not logged.
        import base64
        _old = ("        handler = AttentionBackendRegistry.get_handler(attention_backend)\n"
                "        return handler(self, forward_batch)")
        _new = ("        handler = AttentionBackendRegistry.get_handler(attention_backend)\n"
                "        _m = handler(self, forward_batch)\n"
                "        try:\n"
                "            import torch as _t, time as _tm, logging as _lg, os as _os\n"
                "            from sglang.srt.distributed import get_tensor_model_parallel_rank as _r\n"
                "            if (not _t.compiler.is_compiling()) and getattr(self, 'layer_id', 0) == 0 and _r() == 0 and forward_batch.forward_mode.is_extend_without_speculative():\n"
                "                _lg.getLogger('aim345').warning('a345_dispatch pid=%d wall=%.6f method=%s n_req=%d prefix=%s new=%s rids=%s', _os.getpid(), _tm.time(), _m.name, len(forward_batch.rids or []), ','.join(str(int(x)) for x in (forward_batch.extend_prefix_lens_cpu or [])), ','.join(str(int(x)) for x in (forward_batch.extend_seq_lens_cpu or [])), ','.join(str(x) for x in (forward_batch.rids or [])))\n"
                "        except Exception:\n"
                "            pass\n"
                "        return _m")
        _patch_py = (
            'import pathlib, sglang\n'
            'p = pathlib.Path(sglang.__file__).parent / "srt" / "models" / "deepseek_v2.py"\n'
            's = p.read_text()\n'
            'old = ' + repr(_old) + '\n'
            'new = ' + repr(_new) + '\n'
            'assert s.count(old) == 1, f"expected exactly 1 occurrence, found {s.count(old)}"\n'
            'p.write_text(s.replace(old, new))\n'
            'print("dispatch logging patch applied")\n'
        )
        _b64 = base64.b64encode(_patch_py.encode()).decode()
        dispatch_logging_prefix = f"python3 -c \"import base64; exec(base64.b64decode('{_b64}').decode())\" && "
    progthread_prefix = ''
    if c.get('decode_no_prog_thread'):
        # Investigation-only (OPEN-ITEMS-NODE4 item E control): the decode role's NIXL agent runs
        # without NIXL's own progress thread, so libfabric progress (and any receive-side copy)
        # happens in the SGLang thread that polls notifications. Prefill is unchanged.
        import base64
        _old = "        agent_config = nixl_agent_config(backends=[], num_threads=num_threads)"
        _new = "        agent_config = nixl_agent_config(backends=[], num_threads=num_threads, enable_prog_thread=(disaggregation_mode != DisaggregationMode.DECODE))"
        _patch_py = (
            'import pathlib, sglang\n'
            'p = pathlib.Path(sglang.__file__).parent / "srt" / "disaggregation" / "nixl" / "conn.py"\n'
            's = p.read_text()\n'
            'old = ' + repr(_old) + '\n'
            'new = ' + repr(_new) + '\n'
            'assert s.count(old) == 1, f"expected exactly 1 occurrence, found {s.count(old)}"\n'
            'p.write_text(s.replace(old, new))\n'
            'print("decode progress-thread patch applied")\n'
        )
        _b64 = base64.b64encode(_patch_py.encode()).decode()
        progthread_prefix = f"python3 -c \"import base64; exec(base64.b64decode('{_b64}').decode())\" && "
    room_logging_prefix = ''
    if c.get('patch_room_logging'):
        # Investigation-only (OPEN-ITEMS-NODE6 item D): log rid and bootstrap_room on both sides so a
        # client request joins to its prefill forwards: prefill writes the transfer metadata in
        # set_buf, decode commits it in _commit_transfer_to_req.
        import base64
        _edits = [
            ("srt/disaggregation/utils.py",
             "        self.cached_tokens[req.metadata_buffer_index][3] = req.cached_tokens_storage\n",
             "        self.cached_tokens[req.metadata_buffer_index][3] = req.cached_tokens_storage\n"
             "        try:\n"
             "            import logging as _lg, os as _os, time as _tm\n"
             "            _lg.getLogger('aim345').warning('a345_room side=prefill pid=%d wall=%.6f rid=%s room=%s cached=%s input_len=%d', _os.getpid(), _tm.time(), req.rid, req.bootstrap_room, req.cached_tokens, len(req.origin_input_ids))\n"
             "        except Exception:\n"
             "            pass\n"),
            ("srt/disaggregation/decode.py",
             "        decode_req.req.cached_tokens_storage = cached_tokens[3].item()\n",
             "        decode_req.req.cached_tokens_storage = cached_tokens[3].item()\n"
             "        try:\n"
             "            import logging as _lg, os as _os, time as _tm\n"
             "            _lg.getLogger('aim345').warning('a345_room side=decode pid=%d wall=%.6f rid=%s room=%s cached=%s', _os.getpid(), _tm.time(), decode_req.req.rid, decode_req.req.bootstrap_room, decode_req.req.cached_tokens)\n"
             "        except Exception:\n"
             "            pass\n"),
        ]
        _patch_py = 'import pathlib, sglang\nroot = pathlib.Path(sglang.__file__).parent\n'
        for _f, _old, _new in _edits:
            _patch_py += ('p = root / ' + repr(_f) + '\ns = p.read_text()\nold = ' + repr(_old) + '\nnew = ' + repr(_new) + '\n'
                          'assert s.count(old) == 1, f"expected exactly 1 occurrence in {p}, found {s.count(old)}"\np.write_text(s.replace(old, new))\n')
        _patch_py += 'print("room logging patch applied")\n'
        _b64 = base64.b64encode(_patch_py.encode()).decode()
        room_logging_prefix = f"python3 -c \"import base64; exec(base64.b64decode('{_b64}').decode())\" && "
    tdev_prefix = ''
    if c.get('tdev_shim'):
        # Investigation-only (OPEN-ITEMS-NODE6 item E): compile and LD_PRELOAD a pthread_create shim
        # that logs each new thread's CUDA context device; tdev_fix makes NIXL libfabric threads
        # start with their creator's device (or tdev_force_dev's) current.
        import base64, pathlib
        _src = (pathlib.Path(c['tdev_src']) if 'tdev_src' in c else pathlib.Path(__file__).parent / 'measurement' / 'shims' / 'tdev.c').read_bytes()
        _b64 = base64.b64encode(_src).decode()
        tdev_prefix = (f"echo {_b64} | base64 -d > /tmp/tdev.c && gcc -shared -fPIC -O2 -o /tmp/tdev.so /tmp/tdev.c -ldl -lpthread && "
                       "export LD_PRELOAD=/tmp/tdev.so && ")
        if c.get('tdev_fix'):
            tdev_prefix += "export AIM345_TDEV_FIX=1 && "
        if c.get('tdev_force_dev') is not None:
            tdev_prefix += f"export AIM345_TDEV_FORCE_DEV={int(c['tdev_force_dev'])} && "
        if c.get('tdev_role_port'):
            tdev_prefix += f"export AIM345_TDEV_ROLE_PORT={int(c['tdev_role_port'])} && "
        if c.get('tdev_foreign_dev') is not None:
            tdev_prefix += f"export AIM345_TDEV_FOREIGN_LOAD=1 AIM345_TDEV_FOREIGN_DEV={int(c['tdev_foreign_dev'])} AIM345_TDEV_FOREIGN_PERIOD_US={int(c.get('tdev_foreign_period_us', 10000))} && "
        for _k, _e in (('tdev_foreign_size', 'AIM345_TDEV_FOREIGN_SIZE'), ('tdev_foreign_burst', 'AIM345_TDEV_FOREIGN_BURST'), ('tdev_foreign_issue', 'AIM345_TDEV_FOREIGN_ISSUE')):
            if c.get(_k) is not None:
                tdev_prefix += f"export {_e}={c[_k]} && "
        for _k, _v in (c.get('tdev_env') or {}).items():
            tdev_prefix += f"export {_k}={_v} && "
        if c.get('mps_daemon'):
            # NODE8 control: one CUDA MPS daemon for every engine process of the pod.
            tdev_prefix += ("export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps CUDA_MPS_LOG_DIRECTORY=/tmp/mpslog && mkdir -p /tmp/mps /tmp/mpslog && "
                            "nvidia-cuda-mps-control -d && sleep 2 && echo a345_mps_started && ")
        if c.get('tdev_copylog'):
            tdev_prefix += "export AIM345_TDEV_COPYLOG=1 && "
        if c.get('tdev_foreign_mode'):
            tdev_prefix += f"export AIM345_TDEV_FOREIGN_MODE={c['tdev_foreign_mode']} AIM345_TDEV_OWN_DEV={int(c['tdev_own_dev'])} && "
    engine['command'] = ['bash', '-c', 'ulimit -l unlimited; python3 /lab/verify_image.py && ' + nixl_pin_shell_prefix(c) + descriptor_logging_prefix + dispatch_logging_prefix + progthread_prefix + room_logging_prefix + tdev_prefix + 'exec python3 /lab/packed.py']
    if c.get('nsys_launch'):
        # NODE8 investigation: run the engine processes under an Nsight Systems launch session (collection is
        # started and stopped later with `nsys start` / `nsys stop` from a hook).
        engine['command'][-1] = engine['command'][-1].replace('exec python3 /lab/packed.py', 'exec nsys launch --session-new=a345 ' + c['nsys_launch'] + ' python3 /lab/packed.py')
    if c.get('engine_privileged'):
        engine['securityContext'] = {'privileged': True}
    engine.pop('args')
    engine['env'].append({'name': 'AIM345_ENGINE_PLANS', 'value': json.dumps(plans)})
    for resources in engine['resources'].values():
        resources['nvidia.com/gpu'] = 2*c['gpus_per_worker']
        resources['cpu'] = str(c.get('packed_cpu', 8))
        resources['memory'] = c.get('packed_memory', '128Gi')
    health = "import urllib.request; [urllib.request.urlopen('http://127.0.0.1:%d/health'%p, timeout=8).read() for p in (30000,30001)]"
    for key in ('startupProbe', 'readinessProbe'):
        engine[key].pop('httpGet')
        engine[key]['exec'] = {'command': ['python3', '-c', health]}
        engine[key]['timeoutSeconds'] = 20
    result = [namespace]
    names = [x['metadata']['name'] for x in engines]
    for item in doc['items']:
        if item in engines:
            continue
        if item['kind'] == 'Service' and item['metadata']['name'] in names:
            index = names.index(item['metadata']['name'])
            item['spec']['selector'] = OWNER | {'app': 'engines'}
            # A headless service resolves directly to the pod, so both the
            # advertised and target ports must identify the right process.
            item['spec']['ports'][0].update(port=30000+index, targetPort=30000+index)
        if item['kind'] == 'Deployment' and item['metadata']['name'] == 'router':
            router_pod = item['spec']['template']['spec']
            router_pod['nodeSelector'] = spec['nodeSelector'].copy()
            router_pod['tolerations'] = c['tolerations']
            router = router_pod['containers'][0]
            router['args'] = [arg.replace(f"{names[1]}.{c['namespace']}.svc.cluster.local:30000", f"{names[1]}.{c['namespace']}.svc.cluster.local:30001") for arg in router['args']]
        result.append(item)
    result.append(packed)
    return {'apiVersion': 'v1', 'kind': 'List', 'items': result}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['render', 'deploy', 'verify', 'cleanup'])
    p.add_argument('--unified-config', default='config.unified.json')
    p.add_argument('--disaggregated-config', default='config.disaggregated.json')
    p.add_argument('--output', type=Path, default=Path('results/paired'))
    a = p.parse_args()
    configs = [read_config(a.unified_config), read_config(a.disaggregated_config)]
    gpus = check_pair(*configs)
    docs = [paired_render(c, mode) for c, mode in zip(configs, ['unified', 'disaggregated'])]
    a.output.mkdir(parents=True, exist_ok=True)
    for mode, doc in zip(['unified', 'disaggregated'], docs):
        (a.output / (mode + '.json')).write_text(json.dumps(doc, indent=2)+'\n')
    if a.action == 'render':
        print(json.dumps({'gpus_per_stack': gpus, 'namespaces': [c['namespace'] for c in configs], 'output': str(a.output)}))
        return
    if a.action == 'deploy':
        # Do not silently adopt another participant's existing namespace.
        for c in configs:
            existing = kubectl(c, 'get', 'namespace', c['namespace'], '--ignore-not-found', '-o', 'name', capture_output=True).stdout.strip()
            if existing:
                raise RuntimeError(f'Namespace already exists: {existing}; verify or explicitly clean up your own run')
        for c, doc in zip(configs, docs):
            owned_namespace(c, create=True)
            kubectl(c, 'apply', '-f', '-', input=json.dumps(doc))
        return
    if a.action == 'cleanup':
        for c in configs:
            owned_namespace(c)
            kubectl(c, 'delete', 'namespace', c['namespace'], '--wait=true')
            kubectl(c, 'delete', 'priorityclass', c['namespace']+'-nonpreempting')
        return
    # Router liveness only (DESIGN.md section 11): no --unified-url/--disaggregated-url, no port
    # forward -- a kubectl exec loopback curl against the router pod's own process. This proves the
    # router answers, not that the measurement path works end to end; `labctl selfcheck` is that gate.
    records = []
    for c in configs:
        owned_namespace(c)
        pods = json.loads(kubectl(c, 'get', 'pods', '-l', 'component=engine', '-o', 'json', capture_output=True).stdout)['items']
        count = sum(int(x['resources']['requests'].get('nvidia.com/gpu', 0)) for pod in pods for x in pod['spec']['containers'])
        if count != gpus or not pods:
            raise RuntimeError('Live GPU request does not match the paired budget')
        for pod in pods:
            if pod['spec']['nodeName'] not in c['nodes'] or not pod['status'].get('containerStatuses') or not all(x['ready'] for x in pod['status']['containerStatuses']):
                raise RuntimeError('Engine placement or readiness differs from the paired configuration')
        router_pods = json.loads(kubectl(c, 'get', 'pods', '-l', 'app=router', '-o', 'json', capture_output=True).stdout)['items']
        if not router_pods:
            raise RuntimeError('No router pod in this namespace')
        code = "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=10).status)"
        health = int(kubectl(c, 'exec', router_pods[0]['metadata']['name'], '-c', 'router', '--', 'python3', '-c', code, capture_output=True).stdout.strip())
        records.append({'namespace': c['namespace'], 'http_status': health, 'gpus': count,
                        'pod_uids': [x['metadata']['uid'] for x in pods], 'nodes': c['nodes']})
    (a.output/'endpoints.json').write_text(json.dumps(records, indent=2)+'\n')
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
