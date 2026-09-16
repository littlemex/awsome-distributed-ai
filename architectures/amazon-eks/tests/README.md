# Amazon EKS GPU architecture — Test & Validation Guide

Test procedures for the CloudFormation deploy path in [`../assets`](../assets). Each procedure
lists the exact command to run and the observation that makes it a pass — a step whose only
success criterion is "no error" is not a test.

For operational guidance (rollback behaviour, node group replacement, IAM), see
[`../docs/OPERATIONS.md`](../docs/OPERATIONS.md). For deploying a branch from your own S3
bucket, see [`../docs/DEPLOY-TESTING.md`](../docs/DEPLOY-TESTING.md).

---

## Pre-merge test matrix

Run the rows whose "When to run" applies to the change. Rows 0 and 1 are cheap and run on
every pull request; rows 2 and 3 need GPU capacity and are the ones that actually prove the
architecture works.

| # | Category | Tests | File | When to run |
|---|---|---|---|---|
| 0 | **Lint** | `bash tests/lint-templates.sh` — ten mechanical checks over the templates and the docs (mapping coverage, chart pins, generated NIC block, parameter documentation, link resolution, output names). Seconds, no AWS account needed | [`lint-templates.sh`](./lint-templates.sh) | Every PR |
| 1 | **Infrastructure** | `validate-template`; a `GpuNodeCount=0` deploy (a smoke test of the network, cluster and add-ons — **not** an end-to-end test); a nested deploy from your own bucket | [`infra-test.md`](./infra-test.md) | Every PR touching `assets/` |
| 2 | **GPU and EFA** | 2 x `g7e.12xlarge`: nodes `Ready`, `nvidia.com/gpu` at the mapped count, `vpc.amazonaws.com/efa` at the mapped count, NVMe RAID0 at `/mnt/k8s-disks/0`, cross-node traffic proven to run over the EFA provider, `/dev/infiniband` inside an EFA-requesting pod | [`gpu-efa-test.md`](./gpu-efa-test.md) | Every PR touching `eks-add-gpu-nodegroup.yaml`, `NicLayout`, `GpuCount`, the bootstrap, `KubernetesVersion` or either chart pin |
| 3 | **Disaggregated smoke** | One request served with prefill and decode pinned to different GPU nodes, using an existing `examples/inference/` sample unmodified | [`disagg-smoke-test.md`](./disagg-smoke-test.md) | Before a workshop; when the GPU node shape, the FSx module or the device-plugin install changes |
| 4 | **Cleanup** | Stack delete leaves nothing behind: GuardDuty managed endpoint and its security group, LoadBalancers, PersistentVolumes, the EKS control-plane log group | [`cleanup-test.md`](./cleanup-test.md) | Every PR that adds or removes a resource; after every hardware run |

Rows 2, 3 and 4 run against one deploy: bring up a cluster with `GpuNodeCount=2` and
`GpuInstanceType=g7e.12xlarge`, run row 2, then row 3 on the same cluster, then row 4 to
tear it down.

---

## What row 0 covers, and what a SKIP means

`lint-templates.sh` implements the checks frozen in the phase-2 interface document:

| Check | Asserts | Needs |
|---|---|---|
| 0 | The four templates, `render-nic-block.py`, `docs/PARAMETERS.md` and `README.md` exist; a `python3` that can `import yaml` is on `PATH` | — |
| 1 | `aws cloudformation validate-template` accepts all four templates | AWS credentials |
| 2 | Every `GpuInstanceType.AllowedValues` entry has a `NicLayout` and a `GpuCount` entry; every `NicLayout` entry has all four keys as strings; the two mappings cover the same types; the root and the GPU template offer the same `AllowedValues` | PyYAML |
| 3 | Every selectable GPU type appears in the pinned EFA chart's `nodeAffinity` instance-type list | `helm` |
| 4 | Every DaemonSet the pinned NVIDIA chart renders tolerates `nvidia.com/gpu:NoSchedule`, rendered with the `--set` and `--set-json` arguments the template itself passes — so a template that drops the toleration fails here | `helm` |
| 5 | `tests/render-nic-block.py` output is byte-identical to the generated block committed in `eks-add-gpu-nodegroup.yaml` | — |
| 6 | The root template's parameters and `docs/PARAMETERS.md` are the same set, in both directions | PyYAML |
| 7 | Every relative link and in-page anchor in `README.md`, `docs/*.md` and `tests/*.md` resolves | — |
| 8 | No `latest` tag, untagged image, `stable.txt` lookup or `helm install` without `--version` in the templates | — |
| 9 | Every output name the interface document freezes exists in the template that owns it | PyYAML |
| 10 | `NicLayout`'s `Cards`, `EfaInterfaces` and `PrimaryEfa` agree with `describe-instance-types` | AWS credentials |

Both chart versions, and the values check 4 renders with, are read out of the templates — through
a shell assignment, a buildspec variable, a parameter default, or a CodeBuild environment
variable whose value is a `Mappings` lookup — so the lint always tests what the templates install
rather than what the charts default to. A pinned version that does not exist in the chart
repository fails; only an unreachable repository skips. Anything it cannot resolve is a failure, never
a fallback to a default: an unreadable version, a values file it cannot reproduce, and an
unreadable `--set` argument that decides tolerations all fail the check. An unreadable `--set`
argument that touches nothing the check asserts is rendered with a placeholder and reported as
a `NOTE` line, which is neither a pass nor a skip.

```bash
bash architectures/amazon-eks/tests/lint-templates.sh
```

**Expected:** a `[OK] lint-templates.sh: 0 failures, N skipped` line and exit status 0. Any
`FAIL` line names the file, what was expected and what was found; the exit status is 1.

A `SKIP` line is **not** a pass — it means a check did not run. Skips are counted separately
for that reason. `helm` missing, no network to the chart repositories, and no AWS credentials
are the only expected causes; the CI job configures credentials *after* the lint step, so
checks 1 and 10 always skip there and are closed by a local run or by row 1 below.

`SecondaryDeviceIndex` is deliberately outside check 10: the EC2 API does not report which
`DeviceIndex` a secondary network card needs, which is why that value is carried as data and
proven by row 2 on hardware.

---

## Verified configurations

Filled in from real runs only. A row is added when the whole matrix passed on that
configuration; an unverified type belongs in the layout table in
[`../docs/COMPATIBILITY.md`](../docs/COMPATIBILITY.md), not here.

| Region (AZ ID) | Instance type | Nodes | Kubernetes | Rows run | Date (JST) |
|---|---|---|---|---|---|
| *(to be filled in by the hardware run)* | `g7e.12xlarge` | 2 | `1.36` | 0-4 | — |

---

## Cleanup

Always finish a hardware run with [`cleanup-test.md`](./cleanup-test.md) — deleting the stack
is not the same as leaving nothing behind, and two of the leftovers (the GuardDuty managed
endpoint and its security group, a participant's LoadBalancer) block the delete rather than
surviving it.
