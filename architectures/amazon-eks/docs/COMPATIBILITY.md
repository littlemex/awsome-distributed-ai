# Compatibility matrix

What is pinned, what floats, and what a deploy actually resolved to. A template that says nothing
about versions is a template that breaks on a day nobody chose: this page is what the next person
updating it reads first.

## Pinned in the templates

| Component | Version | Where |
|---|---|---|
| Kubernetes | `1.36` default, `1.35` also offered | `KubernetesVersion` parameter |
| GPU AMI | `AL2023_x86_64_NVIDIA` for the cluster version | `AmiType` in `eks-add-gpu-nodegroup.yaml` |
| System AMI | `AL2023_x86_64_STANDARD` | `AmiType` in `eks-cluster.yaml` |
| `kubectl` in the bootstrap | `1.35.8` / `1.36.4`, selected by `KubernetesVersion` | `KubectlVersion` mapping |
| `helm` in the bootstrap | `3.19.0` | `HELM_VERSION` in the buildspec |
| NVIDIA device plugin chart | `0.20.0` | `NVIDIA_DEVICE_PLUGIN_CHART_VERSION` in the buildspec |
| EFA device plugin chart | `v0.5.32` | `EFA_DEVICE_PLUGIN_CHART_VERSION` in the buildspec |
| CodeBuild image | `aws/codebuild/amazonlinux-x86_64-standard:5.0` | `Environment.Image` |
| Pre-pull holder image | `public.ecr.aws/docker/library/busybox:1.36` | the buildspec's DaemonSet |

Two of these pins are load-bearing rather than tidy:

- **The EFA device plugin chart gates itself on a list of instance types** in its DaemonSet's
  `nodeAffinity`. A type absent from that list never gets the plugin, so it never advertises
  `vpc.amazonaws.com/efa`, and the stack fails at the verification step with no obvious cause.
  `tests/lint-templates.sh` asserts that every value of `GpuInstanceType.AllowedValues` appears in the
  pinned chart's list. For reference, the manifest pinned by
  `examples/inference/vllm/dsv3-uccl-nixl/setup/install-prereqs.sh` is `v0.5.7`, which predates both
  `g7e` and `p6-b300`.
- **The NVIDIA chart's tolerations reach four DaemonSets, not one.** `gfd.enabled=true` adds GPU
  feature discovery, the MPS control daemon and a node-feature-discovery subchart, and the lint
  renders the chart **with the arguments the template passes it** for exactly that reason: rendering
  with chart defaults inspects half of them. On the cluster that was deployed, the device plugin and
  GPU feature discovery ran on both GPU nodes, the node-feature-discovery worker scheduled on the two
  system nodes only, and the GPU nodes still advertised their GPUs — so the worker's toleration is not
  what makes the labels appear. The check stays because a chart that stopped honouring the
  `tolerations` value would leave the device plugin itself unschedulable on a tainted node.

## Left at the EKS default

`vpc-cni`, `kube-proxy`, `coredns`, `eks-pod-identity-agent` and `aws-fsx-csi-driver` are created
without `AddonVersion`, so each resolves to the default for the cluster version. That is the version
EKS supports for that version, and pinning it would mean updating this file every time EKS moves the
default. Record below what a deploy resolved to, so a future failure can be compared against a
known-good set.

## Kubernetes version policy

`AllowedValues` holds the versions the templates are written against; the table at the end of this
page records which of them a deploy has actually run. Choosing a default is a support-window
decision, not a "latest" decision:

```bash
aws eks describe-cluster-versions --region us-west-2 \
  --query 'clusterVersions[].{Version:clusterVersion,Default:defaultVersion,EndOfStandardSupport:endOfStandardSupportDate}' \
  --output table
```

At the time of writing, 1.36 was the EKS default with standard support to 2027-08-02, 1.35 ran to
2027-03-27, and 1.34 to 2026-12-02. A default whose standard support ends within a few months of
publication puts every later deploy into extended support, so the default is chosen with room to
spare rather than to match whatever was newest.

When raising the default: add the version to `AllowedValues` in all three templates that carry the
parameter, add a `KubectlVersion` entry, confirm the AL2023 NVIDIA AMI exists for it
(`aws ssm get-parameter --name /aws/service/eks/optimized-ami/<version>/amazon-linux-2023/x86_64/nvidia/recommended`),
run `tests/lint-templates.sh`, and run [`../tests/gpu-efa-test.md`](../tests/gpu-efa-test.md) on it,
recording the result in the table below.

## What a deploy resolved to

| Date | Region | Kubernetes | Instance type | Nodes | AMI release | Result |
|---|---|---|---|---|---|---|
| 2026-09-16 | `us-west-2` (zone b) | 1.36 | `g4dn.8xlarge` | 2 | `1.36.3-20260911` | Both nodes `Ready`, each advertising `nvidia.com/gpu: 1` and `vpc.amazonaws.com/efa: 1`; `nvidia-smi` from a pod holding a GPU reports `Tesla T4, 580.178.04`; local NVMe assembled as `/dev/md127` (`raid0`, 838 GiB) at `/mnt/k8s-disks/0` |
| 2026-09-16 | `us-west-2` (zones a, b) | 1.36 | `g7e.12xlarge` | 2 | — | Not launched. `InsufficientInstanceCapacity` in both zones, and a capacity reservation for two instances was refused in every zone of `us-west-2`, `us-east-1` and `us-east-2` |

Add-on versions the 1.36 cluster resolved to: `vpc-cni v1.22.4-eksbuild.3`,
`kube-proxy v1.36.0-eksbuild.21`, `coredns v1.14.3-eksbuild.16`,
`eks-pod-identity-agent v1.3.10-eksbuild.3`.

Filling this table is part of the test, not an afterthought: it is the only record that ties a working
cluster to the versions that produced it, and the second row is the record that a type being
selectable says nothing about being obtainable.
