# Amazon EKS GPU cluster architecture

CloudFormation templates for an Amazon EKS cluster whose GPU nodes are ready to run the
disaggregated-inference and distributed-training examples in this repository: EFA on every network
card the instance type has, both Kubernetes device plugins installed and verified, local NVMe
assembled as one volume, and optional FSx for Lustre for model weights.

One stack deploys the whole thing. The three stacks underneath it can also be deployed on their own,
so a GPU node group can be added to a cluster that already exists.

| Template | Creates |
|---|---|
| [`assets/eks-gpu-cluster-deploy-all.yaml`](./assets/eks-gpu-cluster-deploy-all.yaml) | Everything below, as nested stacks. This is the one-click and Workshop Studio artifact |
| [`assets/eks-cluster-prerequisites.yaml`](./assets/eks-cluster-prerequisites.yaml) | VPC, subnets, NAT gateway, S3 and ECR endpoints, EFA-capable node security group, optional FSx for Lustre |
| [`assets/eks-cluster.yaml`](./assets/eks-cluster.yaml) | EKS cluster, access entries, EKS add-ons, system node group |
| [`assets/eks-add-gpu-nodegroup.yaml`](./assets/eks-add-gpu-nodegroup.yaml) | GPU launch template, GPU managed node group, device plugins and their verification |
| [`eksctl/`](./eksctl/) | The eksctl manifests that preceded these templates, kept for reference. See [section 7](#7-eksctl-manifests) |

## 1. What this gives a workload, and what it does not

The line matters, because the serving examples in this repository state their cluster prerequisites
and this architecture exists to satisfy them.

| This architecture owns | An example owns |
|---|---|
| VPC, subnets, NAT, S3 and ECR endpoints | The container image and where it is built |
| EKS cluster, add-ons, system nodes | Serving framework and its operator, if it needs one |
| GPU nodes with EFA on every card | Prefill and decode pods, and how they are split |
| NVIDIA and EFA device plugins, verified per node | KV cache transport and its tuning |
| Local NVMe as one RAID0 volume at `/mnt/k8s-disks/0` | Model weights, their licences, tokens |
| Optional FSx for Lustre filesystem and CSI driver | The `PersistentVolumeClaim` that binds to it |
| The taint and labels the examples already expect | Request routing, autoscaling, benchmarking |

After `CREATE_COMPLETE` with `GpuNodeCount=N`, the cluster guarantees: N nodes labelled `role=gpu`
and tainted `nvidia.com/gpu=true:NoSchedule`, each advertising `nvidia.com/gpu` equal to the
instance type's GPU count and `vpc.amazonaws.com/efa` equal to its EFA interface count. The
bootstrap does not report success until every node advertises both, so a stack that completes has
capable nodes rather than merely installed charts.

Examples that consume this: [`examples/inference/vllm/dsv3-uccl-nixl`](../../examples/inference/vllm/dsv3-uccl-nixl),
[`examples/inference/sglang`](../../examples/inference/sglang),
[`examples/inference/nvidia-dynamo`](../../examples/inference/nvidia-dynamo). Their pods already
tolerate the taint above, and the step in their instructions that installs the two device plugins is
already done on a cluster from this architecture.

Two of them need something this stack does not provide, and the difference is worth knowing before
you follow them:

- An example written against SageMaker HyperPod EKS selects nodes by a HyperPod label
  (`sagemaker.amazonaws.com/instance-group-name`). On this cluster those selectors have to be replaced
  with `role=gpu` or with `node.kubernetes.io/instance-type`.
  [`tests/disagg-smoke-test.md`](./tests/disagg-smoke-test.md) does exactly that for the example it
  runs.
- An example that mounts model weights from a `PersistentVolumeClaim` needs the volume to exist. With
  `DeployFsxLustre=true` the filesystem and its CSI driver are here and the `PersistentVolume`
  manifest is in [`docs/OPERATIONS.md`](./docs/OPERATIONS.md#4-fsx-for-lustre-for-model-weights);
  without it, use the node's local NVMe.

Manifests elsewhere in the repository are not covered by that first sentence. The NCCL test manifest
under `micro-benchmarks/`, for instance, has no toleration for this taint and pins a different
instance type.

## 2. Quick start

[![Launch](./images/launch-stack.svg)](https://console.aws.amazon.com/cloudformation/home#/stacks/quickcreate?templateUrl=https://awsome-distributed-ai.s3.amazonaws.com/templates/amazon-eks/eks-gpu-cluster-deploy-all.yaml&stackName=eks-gpu-cluster)

Or from the CLI:

```bash
aws cloudformation create-stack \
  --stack-name eks-gpu-cluster \
  --template-url https://awsome-distributed-ai.s3.amazonaws.com/templates/amazon-eks/eks-gpu-cluster-deploy-all.yaml \
  --capabilities CAPABILITY_IAM \
  --region us-west-2 \
  --parameters \
    ParameterKey=PrimarySubnetAZ,ParameterValue=us-west-2a \
    ParameterKey=SecondarySubnetAZ,ParameterValue=us-west-2b \
    ParameterKey=GpuInstanceType,ParameterValue=g7e.12xlarge \
    ParameterKey=GpuNodeCount,ParameterValue=2
```

The stack name becomes the cluster name. `PrimarySubnetAZ` has to be the Availability Zone of the
capacity reservation when you use one: EFA traffic and the cluster placement group stay inside one
zone. Expect 20 to 25 minutes, most of it the cluster and the node groups.

The default `GpuInstanceType` is `g7e.12xlarge`, and that family needs a node AMI you build, so the
command above also needs `ParameterKey=NodeAmiId,ParameterValue=ami-…`. Section 3 says which families
that applies to and how to build one; a stack submitted without it fails parameter validation before any
resource is created, rather than after the nodes are running.

Then:

```bash
aws eks update-kubeconfig --name eks-gpu-cluster --region us-west-2
kubectl get nodes -l role=gpu -o custom-columns='NAME:.metadata.name,TYPE:.metadata.labels.node\.kubernetes\.io/instance-type,GPU:.status.allocatable.nvidia\.com/gpu,EFA:.status.allocatable.vpc\.amazonaws\.com/efa'
```

The `KubeconfigCommand` output holds the first command with this stack's cluster name and region.

**Before you deploy into a fresh account**, check the quotas below. Their defaults are what a first
attempt usually fails on, 20 minutes in. `g7e.12xlarge` needs 48 On-Demand vCPUs per node, and each
stack takes one Elastic IP for its NAT gateway.

```bash
REGION=us-west-2
# G and VT On-Demand vCPUs; use L-417A185B instead for a p4/p5/p6 instance type.
aws service-quotas get-service-quota --service-code ec2 --quota-code L-DB2E81BA --region $REGION \
  --query 'Quota.[QuotaName,Value]' --output text
# Elastic IPs.
aws service-quotas get-service-quota --service-code ec2 --quota-code L-0263D0A3 --region $REGION \
  --query 'Quota.[QuotaName,Value]' --output text
# Concurrently running CodeBuild builds on Linux/Small, which the bootstrap build uses.
aws service-quotas get-service-quota --service-code codebuild --quota-code L-9D07B6EF --region $REGION \
  --query 'Quota.[QuotaName,Value]' --output text
```

[`docs/OPERATIONS.md`](./docs/OPERATIONS.md) has what each limit does when it is reached.

## 3. GPU instance types

`GpuInstanceType` selects the type; the launch template derives the interface layout from the
`NicLayout` mapping, which records, per type, the number of network cards, how many of them carry
EFA, whether card 0 does, and the device index used on the other cards
(`describe-instance-types`, `NetworkInfo.MaximumNetworkCards` and `NetworkInfo.EfaInfo`).

| Instance type | GPUs | Network cards | EFA interfaces | Launched from this template |
|---|---|---|---|---|
| `g7.12xlarge` | 2 | 1 | 1 | yes, 2 nodes, `eu-south-2`, with `NodeAmiId` |
| `g7.24xlarge` | 4 | 1 | 1 | no; needs `NodeAmiId` |
| `g7.48xlarge` | 8 | 2 | 2 | no; needs `NodeAmiId` |
| `g7e.12xlarge` | 2 | 1 | 1 | see `docs/COMPATIBILITY.md`; needs `NodeAmiId` |
| `g7e.24xlarge` | 4 | 2 | 2 | no |
| `g7e.48xlarge` | 8 | 4 | 4 | no |
| `g6e.12xlarge` | 4 | 1 | 1 | no |
| `g6e.48xlarge` | 8 | 4 | 4 | no |
| `g5.12xlarge` | 4 | 1 | 1 | no |
| `g4dn.8xlarge` | 1 | 1 | 1 | yes, 2 nodes, `us-west-2` |
| `p4d.24xlarge` | 8 | 4 | 4 | no |
| `p4de.24xlarge` | 8 | 4 | 4 | no |
| `p5.48xlarge` | 8 | 32 | 32 | no |
| `p5en.48xlarge` | 8 | 16 | 16 | no |
| `p6-b200.48xlarge` | 8 | 8 | 8 | no |
| `p6-b300.48xlarge` | 8 | 17 | 16 (card 0 is ENA only) | no |

The last column is the honest one. Every type here has a layout taken from the EC2 API, which is what
makes it selectable; that is a weaker claim than having been launched. `describe-instance-types`
agreeing does not settle it, and neither does a template check: `validate-template` does not evaluate
conditions, and `run-instances --dry-run` accepts an EFA interface on a card that cannot carry one.
Only a deploy settles it, and
[`docs/COMPATIBILITY.md`](./docs/COMPATIBILITY.md) records which deploys happened.

### The `g7` family needs a node AMI you build

The RTX PRO GPUs in the `g7` family are not enumerated by the driver in the EKS-optimised AL2023
NVIDIA AMI: a node group without `NodeAmiId` joins, advertises its EFA interface, and never advertises
`nvidia.com/gpu`. The GPU is not the problem — the same instance shows both GPUs under driver
`595.91.07` with the open kernel modules — so what is missing is a node AMI carrying that driver.
[`ami/`](../../ami) in this repository builds one:

```bash
cd ami
packer init packer-ami.pkr.hcl
AWS_REGION=eu-south-2 EKS_VERSION=1.36 INSTANCE_TYPE=g7.12xlarge \
  NVIDIA_DRIVER_VERSION=595.91.07-1.amzn2023 NVIDIA_KERNEL_MODULES=open \
  PCLUSTER_AMI_REGION=us-east-1 make ami_eks_al2023
```

Build on the family you intend to run: the playbook asserts that `nvidia-smi` reports the pinned
driver, so the build host is what proves the driver drives that GPU. Pass the resulting AMI as
`NodeAmiId`. A `Rules` assertion rejects a `g7` instance type without one before any resource is created rather than
after the nodes are running, and `docs/COMPATIBILITY.md` records which combinations were run.

To add a type: add a `NicLayout` entry and a `GpuCount` entry, add it to `AllowedValues`, run
`tests/lint-templates.sh`, and launch it. If its card count is not already one of the thresholds in
the template's `Cards*Plus` conditions, add a condition too.

## 4. Parameters

Every parameter, with its default and what it affects, is in
[`docs/PARAMETERS.md`](./docs/PARAMETERS.md). The four that decide a deploy:

| Parameter | Default | Notes |
|---|---|---|
| `PrimarySubnetAZ` | required | Zone of the GPU nodes and of the capacity reservation |
| `NodeAmiId` | required for `g7` and `g7e` | Node AMI built with [`ami/`](../../ami). Those families' GPUs are not enumerated by the EKS-optimised AMI's driver, and the stack is rejected before any resource is created without one |
| `SecondarySubnetAZ` | required | Second zone, control plane only. Must differ from the first |
| `GpuInstanceType` | `g7e.12xlarge` | See section 3 |
| `GpuNodeCount` | `2` | Disaggregated inference needs at least 2. `0` is a smoke test, not a deploy |

## 5. Running disaggregated inference on it

The cluster gives two GPU nodes that can reach each other over EFA, which is what a prefill/decode
split needs. Pick an example from [`examples/inference`](../../examples/inference) and follow it; the
cluster-side prerequisites in its README are satisfied by this stack, and
[`tests/disagg-smoke-test.md`](./tests/disagg-smoke-test.md) walks one of them end to end as the
architecture's own acceptance test.

Two facts worth knowing before choosing where model weights live:

- `g7e.12xlarge` carries a single 3.8 TB NVMe drive, assembled as RAID0 and mounted under
  `/mnt/k8s-disks/0`. `hostPath` there is the shortest path to a model cache, and it disappears with
  the node.
- `DeployFsxLustre=true` creates an FSx for Lustre filesystem and installs the CSI driver, for
  weights that should outlive the nodes or be shared. The `PersistentVolume` manifest is in
  [`docs/OPERATIONS.md`](./docs/OPERATIONS.md); it needs the `FsxDnsName` and `FsxMountName` outputs.

## 6. Adding GPU capacity to an existing cluster

`assets/eks-add-gpu-nodegroup.yaml` deploys on its own against any EKS cluster in a VPC with an
EFA-capable security group:

```bash
CLUSTER=my-cluster
aws cloudformation create-stack \
  --stack-name my-cluster-gpu \
  --template-body file://assets/eks-add-gpu-nodegroup.yaml \
  --capabilities CAPABILITY_IAM \
  --parameters \
    ParameterKey=ClusterName,ParameterValue=$CLUSTER \
    ParameterKey=PrivateSubnetId,ParameterValue=subnet-0123456789abcdef0 \
    ParameterKey=NodeSecurityGroupId,ParameterValue=sg-0123456789abcdef0 \
    ParameterKey=ClusterSecurityGroupId,ParameterValue=$(aws eks describe-cluster --name $CLUSTER \
      --query cluster.resourcesVpcConfig.clusterSecurityGroupId --output text) \
    ParameterKey=GpuInstanceType,ParameterValue=g7e.12xlarge \
    ParameterKey=GpuNodeCount,ParameterValue=2
```

`ClusterSecurityGroupId` is not optional: as soon as a launch template specifies security groups,
EKS stops attaching the cluster security group, and nodes that do not carry it never join.

Two things the existing cluster has to be, because this template installs the device plugins from
outside it:

- **`AuthenticationMode` of `API` or `API_AND_CONFIG_MAP`.** The stack grants the bootstrap access
  with an `AWS::EKS::AccessEntry`, which a `CONFIG_MAP`-only cluster rejects. Check with
  `aws eks describe-cluster --name $CLUSTER --query cluster.accessConfig.authenticationMode`.
- **A reachable API endpoint from CodeBuild**, which runs outside your VPC. On a cluster with only
  private endpoint access, the bootstrap cannot reach the API server; give the CodeBuild project a
  `VpcConfig`, or install the two device plugins yourself with the versions in
  [`docs/COMPATIBILITY.md`](./docs/COMPATIBILITY.md).

This is also how you change the GPU instance type. A managed node group cannot change its instance
type and its launch template version in one operation — EKS returns `Version and release version
updates cannot be combined with other updates` — so deploy a second node group stack, or replace
this one.

## 7. eksctl manifests

The manifests under [`eksctl/`](./eksctl/) create a comparable two-node-group topology with
[eksctl](https://eksctl.io). They are kept for readers who manage clusters that way. They pin older
Kubernetes versions, they have not been run against a current EKS version, and they are not
maintained alongside the CloudFormation path. Replace the `PLACEHOLDER_*` values before use.

| Manifest | Nodes | Capacity |
|---|---|---|
| `eks-g4dn.yaml` | 2 x g4dn.8xlarge, new VPC | On-Demand |
| `eks-g4dn-vpc.yaml` | 2 x g4dn.8xlarge, existing VPC | On-Demand |
| `eks-p4de-odcr.yaml` | 2 x p4de.24xlarge, new VPC | ODCR |
| `eks-p4de-odcr-vpc.yaml` | 2 x p4de.24xlarge, existing VPC | ODCR |
| `eks-p5-odcr-vpc.yaml` | 1 x p5.48xlarge, existing VPC | ODCR |
| `eks-p5-capacity-block.yaml` | 1 x p5.48xlarge, existing VPC | Capacity Block |
| `eks-g5-node-autorepair.yaml` | 2 x g5.8xlarge with node auto repair | On-Demand |

```bash
eksctl create cluster -f eksctl/eks-p4de-odcr-vpc.yaml
eksctl delete cluster -f eksctl/eks-p4de-odcr-vpc.yaml
```

`efaEnabled: true` in those manifests installs the EFA device plugin. It does not install the
NVIDIA device plugin; that is a separate step in the eksctl path. The CloudFormation path installs
both and verifies them.

## 8. Cleanup

```bash
aws cloudformation delete-stack --stack-name eks-gpu-cluster
```

Nested stacks are deleted with the root. Three classes of resource are not, because the stack does
not own them, and each can leave the VPC undeletable. Check them before deleting, with the commands
in [`tests/cleanup-test.md`](./tests/cleanup-test.md):

- **Anything the cluster created in your account**: `type: LoadBalancer` services and dynamically
  provisioned persistent volumes. Delete those Kubernetes objects first.
- **GuardDuty Runtime Monitoring**, if the account has it: a managed `guardduty-data` VPC endpoint
  and a `GuardDutyManagedSecurityGroup-*` appear after the VPC does. The endpoint holds the subnets,
  and then the security group holds the VPC. Deleting both and retrying the stack deletion completes
  it.
- **Two log groups that create themselves.** `/aws/eks/<cluster>/cluster`, which EKS creates when
  cluster logging is on, and `/aws/lambda/<stack>-BootstrapTrigger-*`, which Lambda creates on its
  first invocation. Neither belongs to the stack, so neither is deleted with it:

  ```bash
  aws logs describe-log-groups --query \
    "logGroups[?contains(logGroupName,'<stack-name>')].logGroupName" --output text
  ```

## 9. Known limits

- The GPU instance type is fixed when the node group is created (section 6).
- `GpuNodeCount` sets minimum, desired and maximum to the same value, so a partly available
  reservation fails the deploy rather than delivering fewer nodes.
- Updating `GpuNodeCount` on a live stack re-runs the device-plugin verification, which can observe
  the node count from before the scaling change. Delete and recreate, or verify by hand afterwards.
- One NAT gateway and one Elastic IP per stack.

## 10. Further reading

- [`docs/PARAMETERS.md`](./docs/PARAMETERS.md) — every parameter
- [`docs/DEPLOY-TESTING.md`](./docs/DEPLOY-TESTING.md) — deploying a change that is not published yet
- [`docs/OPERATIONS.md`](./docs/OPERATIONS.md) — repairing a failed bootstrap, IAM, FSx volume, quotas
- [`docs/COMPATIBILITY.md`](./docs/COMPATIBILITY.md) — the version matrix and what a deploy resolved to
- [`docs/WORKSHOP-STUDIO.md`](./docs/WORKSHOP-STUDIO.md) — hosting the templates for an event
- [`tests/README.md`](./tests/README.md) — what to run before merging a change here
- [Amazon EKS user guide](https://docs.aws.amazon.com/eks/latest/userguide/) and
  [Elastic Fabric Adapter on EKS](https://docs.aws.amazon.com/eks/latest/userguide/node-efa.html)
