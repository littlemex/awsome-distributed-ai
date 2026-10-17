#!/usr/bin/env python3
"""Prime every fixed P/D pair before measurement, then flush prompt caches."""
import argparse
import json
from pathlib import Path
from deployment import kubectl, owned_namespace, read_config

WARM_SCRIPT = '''
import asyncio, json, sys
sys.path.insert(0, "/lab")
import httpx
from benchmark import stream_request
from traffic import tokenizer, tokens
c = json.load(sys.stdin)
tok = tokenizer(c["model_id"], c["model_revision"])
ids = tokens(tok, f"{tok.bos_token}User: Say ready.\\n\\nAssistant:")
async def run():
    results = []
    async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
        for _ in range(c["pairs"] * 2):
            r = await stream_request(client, c["url"], ids, 4, 310)
            results.append(r)
            assert r["ok"], r
    return results
print(json.dumps(asyncio.run(run())))
'''


def warm_in_cluster(c, runner_pod, pairs):
    """The connection that matters is runner-pod-to-router, which never tears down between rates
    (DESIGN.md section 11): this script's original purpose, warming a CloudShell-to-router TCP
    connection, is moot. Kept for its other purpose, priming engine caches, dispatched the same
    way as 7.verify-transport.py's probe -- a kubectl exec into a runner shard, not --url."""
    payload = dict(c, url=f"http://router.{c['namespace']}.svc.cluster.local:8000", pairs=pairs)
    r = kubectl(c, "exec", "-i", runner_pod, "-c", "runner", "--", "python3", "-c", WARM_SCRIPT, input=json.dumps(payload), capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"in-cluster warmup via {runner_pod} failed: {r.stderr.strip()}")
    return json.loads(r.stdout)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="config.json")
    p.add_argument("--runner-pod", default="runner-0", help="Shard pod the warmup requests are sent from (no --url/port-forward)")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    c = read_config(a.config)
    owned_namespace(c)
    pods = json.loads(kubectl(c, "get", "pods", "-l", "component=engine", "-o", "json", capture_output=True).stdout)["items"]
    packed = c.get('placement') == 'packed'
    prefill = pods if packed else [x for x in pods if x["metadata"]["labels"]["role"] == "prefill"]
    decode = pods if packed else [x for x in pods if x["metadata"]["labels"]["role"] == "decode"]
    assert prefill and len(decode) == 1, "Fixed round-robin P/D topology required"
    r = warm_in_cluster(c, a.runner_pod, len(prefill))
    flush = {}
    for pod in pods:
        for port in ((30000, 30001) if packed else (30000,)):
            code = f"import urllib.request; r=urllib.request.urlopen('http://127.0.0.1:{port}/flush_cache'); print(r.status, r.read().decode())"
            key = f"{pod['metadata']['name']}:{port}" if packed else pod['metadata']['name']
            flush[key] = kubectl(c, "exec", pod["metadata"]["name"], "-c", "engine", "--", "python3", "-c", code, capture_output=True).stdout
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"instance_type": c["instance_type"], "evidence_scope": "mechanism-validation", "method": "startup request priming followed by cache flush", "makeConnection_bootstrap_fix": "UNVALIDATED: requires upstream SGLang support; no local patch", "requests": r, "cache_flush": flush}, indent=2))
    print(str(out))

if __name__ == "__main__":
    main()
