#!/usr/bin/env python3
"""Deploy, wait for or clean up one packed stack and its in-cluster runner on one node (measurement only)."""
import argparse, json, pathlib, sys
LAB_DIR = str(pathlib.Path(__file__).resolve().parents[2])
sys.path.insert(0, LAB_DIR)
from deployment import owned_namespace, kubectl, read_config, runner_resources  # noqa: E402
from paired import paired_render  # noqa: E402

def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["render", "deploy", "wait", "cleanup", "podname"])
    p.add_argument("--config", required=True)
    p.add_argument("--mode", required=True, choices=["unified", "disaggregated"])
    p.add_argument("--shards", type=int, default=2)
    a = p.parse_args()
    c = read_config(a.config)
    doc = paired_render(c, a.mode)
    doc["items"] += runner_resources(c, c["nodes"][0], a.shards)
    if a.action == "render":
        print(json.dumps(doc, indent=2)); return
    if a.action == "deploy":
        if kubectl(c, "get", "namespace", c["namespace"], "--ignore-not-found", "-o", "name", capture_output=True).stdout.strip():
            raise RuntimeError("namespace already exists")
        owned_namespace(c, create=True)
        kubectl(c, "apply", "-f", "-", input=json.dumps(doc)); return
    if a.action == "wait":
        kubectl(c, "wait", "--for=condition=Ready", "pod", "-l", "component=engine", "--timeout=1800s")
        kubectl(c, "wait", "--for=condition=Ready", "pod", "-l", "app=router", "--timeout=600s")
        kubectl(c, "wait", "--for=condition=Ready", "pod", "-l", "app=runner", "--timeout=900s"); return
    if a.action == "podname":
        print(json.loads(kubectl(c, "get", "pods", "-l", "component=engine", "-o", "json", capture_output=True).stdout)["items"][0]["metadata"]["name"]); return
    if a.action == "cleanup":
        owned_namespace(c)
        kubectl(c, "delete", "namespace", c["namespace"], "--wait=true")
        kubectl(c, "delete", "priorityclass", c["namespace"] + "-nonpreempting", "--ignore-not-found"); return

if __name__ == "__main__":
    main()
