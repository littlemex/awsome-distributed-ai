#!/usr/bin/env python3
"""Collect this lab's EKS evidence and summarize measured ramp files."""
import argparse
import json
from pathlib import Path
from deployment import engine_endpoint, engine_ports, kubectl, owned_namespace, read_config
from rescore import check_matched, check_reproduces, check_runs, highest_qualifying, load_rows, matched_key, write_csv


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", required=True, help="Directories containing rate-measured.json files")
    p.add_argument("--output", required=True)
    p.add_argument("--config", help="Also collect live resources and engine logs from the owned namespace")
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    measured = load_rows(a.runs)
    for m in measured:
        check_reproduces(m)
    groups = {}
    for m in measured:
        s = m["stored"]
        groups.setdefault((m["architecture"], *matched_key(m), s["slo_ttft_ms"], s["slo_tpot_ms"], s["attainment_target_fraction"]), []).append(m)
    qualified = []
    for key, group in groups.items():
        record = {"group": list(key)}
        try:
            check_matched(group[0]["shape"], group)
            check_runs(group)
        except SystemExit as refusal:
            qualified.append({**record, "refused": str(refusal)})
            continue
        best = highest_qualifying(group, *key[-3:])
        qualified.append({**record, "highest_qualifying_offered_tasks_per_s": best and best["offered"], "max_observed_useful_calls_per_s_per_gpu": max(m["stored"]["useful_calls_per_s_per_gpu"] for m in group), "qualification": "Finite-window observation including drain; extend duration and repeat before claiming sustained capacity"})
    columns = ("source", "architecture", "instance_type", "region", "evidence_scope", "model_revision", "shape", "shape_sha256")
    write_csv(out / "summary.csv", [{**m["stored"], **{k: m[k] for k in columns}} for m in measured])
    (out / "qualified-rates.json").write_text(json.dumps(qualified, indent=2))
    # Live evidence after the result files, so a failing kubectl call cannot lose them.
    if a.config:
        c = read_config(a.config)
        owned_namespace(c)
        (out / "config.json").write_text(json.dumps(c, indent=2))
        for resource in ("pods", "deployments", "services", "events"):
            (out / f"{resource}.json").write_text(kubectl(c, "get", resource, "-o", "json", capture_output=True).stdout)
        pods = json.loads((out / "pods.json").read_text())["items"]
        for pod in pods:
            name = pod["metadata"]["name"]
            if pod["metadata"].get("labels", {}).get("app.kubernetes.io/part-of") != "aim345-lab":
                continue
            (out / f"{name}.log").write_text(kubectl(c, "logs", name, "--all-containers=true", "--tail=-1", capture_output=True).stdout)
            if pod["metadata"]["labels"].get("component") == "engine" and pod["status"]["phase"] == "Running":
                ports = engine_ports(pod["spec"])
                for port in ports:
                    for endpoint in ("metrics", "server_info"):
                        suffix = f"{port}-{endpoint}" if len(ports) > 1 else endpoint
                        (out / f"{name}-{suffix}.txt").write_text(engine_endpoint(c, name, port, endpoint))
    print(json.dumps({"summary_csv": str(out / "summary.csv"), "measured_rate_rows": len(measured), "qualified_rates": str(out / "qualified-rates.json")}))
    refused = [q["refused"] for q in qualified if "refused" in q]
    if refused:
        raise SystemExit(f"{len(refused)} group(s) refused; summary.csv lists every measured file and qualified-rates.json names each reason:\n" + "\n".join(refused))


if __name__ == "__main__":
    main()
