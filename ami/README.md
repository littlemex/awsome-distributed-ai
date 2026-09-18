<!-- markdownlint-disable MD013 -->

# Amazon Machine Image catalog

This directory builds 6 x86_64 AMIs with Packer and Ansible. Each `ami_*` target listed below builds 1 image. AMI builds launch billable EC2 resources, so the aggregate `ami` target exits without building anything.

## Prerequisites

Install GNU Make, [Packer](https://developer.hashicorp.com/packer/install), Ansible, and AWS credentials that can read public SSM parameters, call EC2 `DescribeImages`, and create AMIs. Initialize the plugins before the first build:

```bash
packer init packer-ami.pkr.hcl
```

Select one target explicitly. `AWS_REGION` defaults to `us-east-1`.

```bash
AWS_REGION=us-east-1 make ami_eks_ubuntu2404
```

`ami_eks_al2023` takes five optional overrides: `EKS_VERSION`, `INSTANCE_TYPE`,
`NVIDIA_DRIVER_VERSION`, `NVIDIA_KERNEL_MODULES` with the value `proprietary` or `open`, and
`PCLUSTER_AMI_REGION`. The build host has to
be an instance type whose GPU the chosen driver supports, because the playbook asserts that
`nvidia-smi` reports the pinned version — which makes the build its own test. The RTX PRO GPUs in the
`g7` family need the open kernel modules: the EKS-optimised AL2023 NVIDIA AMI ships the proprietary
ones and does not enumerate them. The same is assumed for `g7e`, which shares the GPU generation and
which no Region tried had capacity for.

```bash
AWS_REGION=eu-south-2 EKS_VERSION=1.36 INSTANCE_TYPE=g7.12xlarge \
  NVIDIA_DRIVER_VERSION=595.91.07-1.amzn2023 NVIDIA_KERNEL_MODULES=open \
  PCLUSTER_AMI_REGION=us-east-1 make ami_eks_al2023
```

`PCLUSTER_AMI_REGION` exists because Packer evaluates every data source in the file even when a
single build is selected: a Region without a ParallelCluster image otherwise fails a lookup the
build does not use.

Public SSM parameters provide 5 parent AMIs. The ParallelCluster parent is selected with EC2 `DescribeImages`. A later build from the same Git revision can therefore use a newer parent. The resulting AMI tags record the resolved parent AMI ID and lookup family; retain the Packer log as build evidence.

## Active AMIs

| Make target | Parent | Custom layer and ownership | Base AMI release notes |
| --- | --- | --- | --- |
| `ami_ec2_ubuntu2404` | Canonical Ubuntu Server 24.04 LTS | EFA 1.50.0. The workload owns CUDA, NCCL, and frameworks. | [Canonical EC2 image discovery](https://documentation.ubuntu.com/aws/aws-how-to/instances/find-ubuntu-images/) |
| `ami_ec2_ubuntu2404_dlami` | Base OSS NVIDIA Driver GPU DLAMI, Ubuntu 24.04 | Validates the parent NVIDIA driver and does not replace parent libraries. | [AWS Deep Learning Base GPU AMI release notes](https://docs.aws.amazon.com/dlami/latest/devguide/appendix-ami-release-notes.html) |
| `ami_pcluster_ubuntu2404` | AWS ParallelCluster 3.15.1, Ubuntu 24.04 | Retains the official ParallelCluster software stack and validates its image marker. | [AWS ParallelCluster 3.15.1 release notes](https://github.com/aws/aws-parallelcluster/wiki/3.15.1) |
| `ami_eks_al2023` | EKS optimized AL2023 standard, version from `EKS_VERSION` | Installs a pinned NVIDIA driver from the AL2023 NVIDIA repository with the kernel module flavour selectable, `nvidia-driver-cuda` by default or `nvidia-open`, plus the pinned NVIDIA container toolkit. Kubernetes bootstrap remains owned by `nodeadm`. | [Amazon EKS AMI releases](https://github.com/awslabs/amazon-eks-ami/releases) and [AL2023 NVIDIA advisory](https://alas.aws.amazon.com/AL2023/ALAS2023NVIDIA-2026-271.html) |
| `ami_eks_ubuntu2404` | Canonical Ubuntu 24.04 EKS 1.35 | Installs EFA 1.50.0 in minimal mode: EFA kernel module and `rdma-core` only. GPU Operator owns NVIDIA; workload images own CUDA, NCCL, Libfabric, MPI, and aws-ofi-nccl. | [Canonical EKS image discovery](https://documentation.ubuntu.com/aws/aws-how-to/instances/find-ubuntu-images/) and [EFA release notes](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa-changelog.html) |
| `ami_pcs_ubuntu2404` | Base OSS NVIDIA Driver GPU DLAMI, Ubuntu 24.04 | Adds AWS PCS Agent 1.5.1-1 and AWS PCS Slurm 25.11.7-3 with official installers. Use this AMI with the Slurm 25.11 series. `architectures/aws-pcs` installs Enroot/Pyxis at first boot when `InstallEnrootPyxis=true`. | [DLAMI release notes](https://docs.aws.amazon.com/dlami/latest/devguide/appendix-ami-release-notes.html), [AWS PCS custom AMI guide](https://docs.aws.amazon.com/pcs/latest/userguide/working-with_ami_custom.html), and [installer checksums](https://docs.aws.amazon.com/pcs/latest/userguide/working-with_ami_installers.html#working-with_ami_installers_verify) |

The Canonical links describe image discovery rather than per-build release notes. Record the resolved AMI ID and Canonical serial from EC2 image metadata for each build.

## Component ownership

Parent-provided components are not reinstalled. The ParallelCluster and Base GPU DLAMI parents retain their NVIDIA, EFA, CUDA, and communication-library stacks. The PCS custom layer adds only the PCS Agent and Slurm. It intentionally excludes Enroot and Pyxis. Keep `architectures/aws-pcs` parameter `InstallEnrootPyxis=true` unless another layer provides both components.

EFA Installer 1.50.0 full mode includes aws-ofi-nccl 1.21.1. The minimal EKS Ubuntu build excludes aws-ofi-nccl. This catalog does not build or install a separate host NCCL stack; workload images own NCCL, including the NCCL 2.31.2-1 compatibility target. NCCL benchmarks remain under `micro-benchmarks/nccl-tests`.

## Target migration

Old target names are not aliases.

| Previous target | Replacement |
| --- | --- |
| `ami_base` | `ami_ec2_ubuntu2404` |
| `ami_dlami_gpu` | `ami_ec2_ubuntu2404_dlami` |
| `ami_pcluster_gpu` | `ami_pcluster_ubuntu2404` |
| `ami_eks_gpu` | `ami_eks_al2023` |

The three retired Neuron and CPU variants have no replacement.

## Validation boundary

Packer formatting, configuration validation, Ansible syntax checks, lint checks, and dry-run Make targets are desk checks. They do not prove that an AMI builds or works on target hardware. Before publishing an AMI, build the selected target in an approved AWS account and validate its runtime contract on the intended EC2, EKS, ParallelCluster, or PCS deployment.
