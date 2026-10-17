#!/usr/bin/env python3
"""Check installed versions, both launch selectors, and EFA RDMA byte deltas."""
import argparse
import json
from pathlib import Path
from deployment import kubectl, owned_namespace, read_config

COUNTERS = '''import pathlib,json
values={}
for p in pathlib.Path('/sys/class/infiniband').glob('*/ports/*/hw_counters/*'):
    if p.name in ('rdma_write_bytes','rdma_read_bytes','send_bytes','recv_bytes'):
        try: values[str(p)]=int(p.read_text().strip())
        except (OSError,ValueError): pass
print(json.dumps(values))'''


def snapshot(c, pods):
    return {p["metadata"]["name"]: json.loads(kubectl(c, "exec", p["metadata"]["name"], "-c", "engine", "--", "python3", "-c", COUNTERS, capture_output=True).stdout) for p in pods}


SMOKE_SCRIPT = '''
import asyncio, json, sys, uuid
sys.path.insert(0, "/lab")
import httpx
from benchmark import stream_request
from traffic import initial_prompt, tokenizer
c = json.load(sys.stdin)
tok = tokenizer(c["model_id"], c["model_revision"])
ids = initial_prompt(tok, c["shape"], "transport", uuid.uuid4().hex)
async def run():
    async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
        return await stream_request(client, c["url"], ids, 4, 310)
print(json.dumps(asyncio.run(run())))
'''


def smoke_in_cluster(c, runner_pod):
    """One probe request dispatched through a runner shard's own kubectl exec (DESIGN.md section
    12: a one-off job through the same transport as a ramp), not from this machine -- there is no
    --url/port-forward to this router. Needs the runner StatefulSet already deployed. Reuses
    benchmark.stream_request/traffic.initial_prompt directly rather than importing this file,
    which (like every numbered lab script) is not a valid Python module name. The shape dict is
    read here, not inside the pod: the ConfigMap-mounted /lab (deployment.runner_resources) has
    only runner_agent.py's own four dependencies, not shape-b.json."""
    shape = dict(json.loads(Path("shape-b.json").read_text()), input_tokens=1024)
    payload = dict(c, url=f"http://router.{c['namespace']}.svc.cluster.local:8000", shape=shape)
    r = kubectl(c, "exec", "-i", runner_pod, "-c", "runner", "--", "python3", "-c", SMOKE_SCRIPT, input=json.dumps(payload), capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"in-cluster probe via {runner_pod} failed: {r.stderr.strip()}")
    return json.loads(r.stdout)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="config.json")
    p.add_argument("--runner-pod", default="runner-0", help="Shard pod the probe request is sent from (no --url/port-forward)")
    p.add_argument("--output", required=True)
    p.add_argument("--bandwidth-gib-s", type=float, help="Measured GPU-buffer fabric bandwidth, never a datasheet peak")
    p.add_argument("--bandwidth-evidence", help="Path to the matching hardware microbenchmark transcript")
    a = p.parse_args()
    c = read_config(a.config)
    if a.bandwidth_gib_s is not None and c.get('engine_profile') == 'v4-optional':
        p.error('The optional V4 cache layout needs separate byte accounting; the V2 MLA wire-time formula does not apply')
    owned_namespace(c)
    pods = json.loads(kubectl(c, "get", "pods", "-l", "component=engine", "-o", "json", capture_output=True).stdout)["items"]
    packed = c.get('placement') == 'packed'
    assert len(pods) == 1 if packed else len(pods) >= 2
    if not packed:
        assert len({x["spec"]["nodeName"] for x in pods}) == len(pods), "Transfer must cross nodes"
    versions = {}
    launches = {}
    for pod in pods:
        engine = next(x for x in pod["spec"]["containers"] if x["name"] == "engine")
        plans = [{'args': engine.get('args', []), 'env': {x['name']: x.get('value') for x in engine['env']}}]
        if packed:
            code = "import json,pathlib; plans=json.load(open('/tmp/aim345-engine-plans.json')); print(json.dumps([dict(role=p['role'],args=pathlib.Path('/proc/%s/cmdline'%p['pid']).read_text().strip('\\0').split('\\0'),env=dict(x.split('=',1) for x in pathlib.Path('/proc/%s/environ'%p['pid']).read_text().strip('\\0').split('\\0') if x.startswith(('FI_PROVIDER=','SGLANG_DISAGGREGATION_NIXL_BACKEND=','CUDA_VISIBLE_DEVICES=')))) for p in plans]))"
            plans = json.loads(kubectl(c, 'exec', pod['metadata']['name'], '-c', 'engine', '--', 'python3', '-c', code, capture_output=True).stdout)
            assert {x['role'] for x in plans} == {'prefill', 'decode'}
        for plan in plans:
            args, env = plan['args'], plan['env']
            assert args[args.index("--disaggregation-transfer-backend") + 1] == "nixl"
            assert "--disaggregation-ib-device" not in args
            assert env["SGLANG_DISAGGREGATION_NIXL_BACKEND"] == "LIBFABRIC"
            assert env["FI_PROVIDER"] == "efa"
        launches[pod['metadata']['name']] = plans
        versions[pod["metadata"]["name"]] = json.loads(kubectl(c, "exec", pod["metadata"]["name"], "-c", "engine", "--", "python3", "-c", Path(__file__).with_name("verify_image.py").read_text(), capture_output=True).stdout)
    profiles = {v['engine_profile'] for v in versions.values()}
    expected = c.get('engine_profile', 'mainline')
    assert profiles == {expected}, f"Expected {expected}, observed {profiles}"
    before = snapshot(c, pods)
    response = smoke_in_cluster(c, a.runner_pod)
    after = snapshot(c, pods)
    deltas = {pod: {key: after[pod][key] - value for key, value in values.items()} for pod, values in before.items()}
    rdma_bytes = sum(max(0, value) for values in deltas.values() for key, value in values.items() if key.endswith(("rdma_write_bytes", "rdma_read_bytes")))
    result = {"instance_type": c["instance_type"], "evidence_scope": "mechanism-validation", "versions": versions, "engine_profile": expected, "pods": pods, "response": response, "before_bytes": before, "after_bytes": after, "delta_bytes": deltas,
              "status": "VALIDATED" if response["ok"] and (packed or rdma_bytes > 0) else "UNVALIDATED",
              "launches": launches, "placement": c.get('placement', 'separate-nodes'),
              "cross_node_efa_status": "UNVALIDATED: co-located prefill/decode" if packed else ("VALIDATED positive path" if response['ok'] and rdma_bytes > 0 else "UNVALIDATED"),
              "scope": "Positive configured path only. Shared-node counters include other traffic; inspect provider/GPU registration logs. This does not validate negative selector controls or GPUDirect absence of staging by itself."}
    if a.bandwidth_gib_s is not None:
        assert a.bandwidth_gib_s > 0 and a.bandwidth_evidence
        transcript = Path(a.bandwidth_evidence).read_text()
        # MLA logical cache, all layers. TP replication and protocol overhead are not included.
        import urllib.request
        cfg = json.load(urllib.request.urlopen(f"https://huggingface.co/{c['model_id']}/resolve/{c['model_revision']}/config.json"))
        logical_bytes = 1024 * cfg["num_hidden_layers"] * (cfg["kv_lora_rank"] + cfg["qk_rope_head_dim"]) * 2
        result["attribution"] = {"input_tokens": 1024, "logical_kv_bytes": logical_bytes, "bf16_bytes_per_element": 2, "measured_fabric_gib_per_s": a.bandwidth_gib_s, "ideal_wire_time_ms": logical_bytes / (a.bandwidth_gib_s * 1024**3) * 1000, "bandwidth_evidence": transcript, "qualification": "Lower-bound estimate; count all layers and tensor-parallel replication before attributing a residual to connection setup"}
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({"status": result["status"], "engine_profile": expected, "instance_type": c["instance_type"], "placement": result['placement'], "cross_node_efa_status": result['cross_node_efa_status'], "rdma_counter_delta_bytes": rdma_bytes, "file": str(out)}))
    if result["status"] != "VALIDATED":
        raise SystemExit(1)

if __name__ == "__main__":
    main()
