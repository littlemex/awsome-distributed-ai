#!/usr/bin/env python3
"""Count two libfabric/NIXL log signals across an engine pod's full log -- opt-in, read-only,
no code path this lab's default deploy/verify/ramp flow depends on or calls.

  cuMemGetAddressRange   NIXL's CUDA memory-registration path calls this CUDA driver function
                         when registering a GPU buffer for RDMA. Its count by itself is not
                         diagnostic (NIXL does not log every call it makes), but it is
                         corroborating evidence when investigating a registration failure --
                         compare it against a known-good run's own count for the same traffic,
                         never against zero.
  fallback to using SAR  libfabric's EFA provider's own log line when it could not use RDMA
                         read/write for a transfer and fell back to its slower, segmented
                         send/receive (Segmentation-And-Reassembly, SAR) protocol instead -- a
                         distinct failure mode from the one `FI_EFA_ENABLE_SHM_TRANSFER=0`
                         (DESIGN-RATIONALE.md) addresses: that setting only stops libfabric's
                         same-node shared-memory shortcut, so a transfer can still land on SAR
                         for an unrelated reason (buffer registration, message size) even with
                         it set. A positive SAR count alongside a positive RDMA byte delta from
                         `7.verify-transport.py` would mean the device counters moved for bytes
                         this lab cannot attribute to its own RDMA-write transfer; checking both
                         together is this diagnostic's own corroboration that the measured bytes
                         plausibly came from a real RDMA write, not a definitive attribution
                         either way. Zero is the bar a healthy real-cluster run should clear.

Usage: `python3 diagnose-transport.py --config config.json [--fail-on-sar]`. Reads every engine
pod's full log (`kubectl logs --all-containers --tail=-1`, the same call `5.collect.py --config`
already makes for its own evidence dump) and counts both substrings literally, case-sensitively,
once per line that contains them. This is read-only against an already-running deployment: it
starts nothing, stops nothing, and never patches an engine image or process.

This script does **not** give you the descriptor count (`src_reqs.shape[0]`) DESIGN-RATIONALE.md's
own investigation separately measured. That count is only visible after building a disposable,
local-only debug image with a one-line patch to SGLang's own NIXL connector module baked in before
the engine process ever starts -- not something this lab ships in its default path, by design (see
DESIGN-RATIONALE.md, "Reproducing this PR's own investigation measurements", item 5, for the exact
procedure, why an in-place `kubectl exec` edit against an already-running pod cannot work at all,
and why it is not automated here).
"""
import argparse
import json
import re
from deployment import kubectl, owned_namespace, read_config

SIGNALS = {
    "cuMemGetAddressRange_count": re.compile(r"cuMemGetAddressRange"),
    "sar_fallback_count": re.compile(r"fallback to using SAR"),
}


def log_signal_counts(text):
    """{signal_name: number of lines containing it} over the full text of one engine's log --
    counts lines, not raw substring occurrences, so a line that happens to repeat a pattern is not
    double-counted; irrelevant in practice for either of these two single-sentence log lines, but
    keeps this function's contract simple and tested against instead of "whatever re.findall
    happens to return"."""
    lines = text.splitlines()
    return {name: sum(1 for line in lines if pattern.search(line)) for name, pattern in SIGNALS.items()}


def engine_pods(c):
    return json.loads(kubectl(c, "get", "pods", "-l", "component=engine", "-o", "json", capture_output=True).stdout)["items"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="config.json")
    p.add_argument("--fail-on-sar", action="store_true", help="Exit nonzero if any engine pod logged a SAR fallback")
    a = p.parse_args()
    c = read_config(a.config)
    owned_namespace(c)
    pods = engine_pods(c)
    if not pods:
        # An empty selector match (a deployment mid-replacement, a selector/namespace mismatch, a
        # stack already torn down) used to fall straight through to `report = {}` -- zero signals
        # observed, printed as a clean-looking empty report, and --fail-on-sar's own
        # `sum([]) == 0` check passed on nothing rather than refusing. This refuses regardless of
        # --fail-on-sar: an empty report is not evidence of anything, with or without that flag.
        # Only the fully-empty case: a partial match (some but not all expected pods present, or
        # present but not yet Running) still produces a report and is not caught here.
        raise SystemExit(f"no engine pods matched component=engine in namespace {c['namespace']!r} "
                         f"(context {c['context']!r}); refusing to report a clean result for nothing observed")
    report = {}
    for pod in pods:
        name = pod["metadata"]["name"]
        log = kubectl(c, "logs", name, "--all-containers=true", "--tail=-1", capture_output=True).stdout
        report[name] = log_signal_counts(log)
    print(json.dumps(report, indent=2))
    total_sar = sum(counts["sar_fallback_count"] for counts in report.values())
    if a.fail_on_sar and total_sar:
        raise SystemExit(f"{total_sar} SAR fallback line(s) across {len(report)} engine pod(s); "
                          "the transfer did not stay on the EFA device path for every observed request")


if __name__ == "__main__":
    main()
