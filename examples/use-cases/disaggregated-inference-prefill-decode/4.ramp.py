#!/usr/bin/env python3
"""Dispatch one ramp through the in-cluster runner shards -- labctl.py's run_ramp(), under the
filename DESIGN.md section 11's command table keeps for what a participant types. No port
forward and no --url: the shards reach the router from inside the cluster. See labctl.py for the
mechanism (run id, lease, resume, atomic writes) and README.md's "Where the in-cluster runner's
results live" for why shard transport is one kubectl exec per shard rather than a shared mount.
"""
import argparse
import labctl

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    labctl.add_ramp_arguments(p)
    a = p.parse_args()
    labctl.validate_ramp_arguments(a)
    labctl.run_ramp(a)
