# Amazon EKS GPU cluster architecture

CloudFormation templates for an Amazon EKS cluster whose GPU nodes are ready to run the
disaggregated-inference and distributed-training examples in this repository: EFA on every network
card the instance type offers it on, both Kubernetes device plugins installed and verified, local NVMe
assembled as one volume, and optional FSx for Lustre for model weights.

One stack deploys the whole thing. Each of the four underneath it can also be deployed on its own, so
a GPU node group can be added to a cluster that already exists.

| Template | Creates |
|---|---|
| [`assets/eks-gpu-cluster-deploy-all.yaml`](./assets/eks-gpu-cluster-deploy-all.yaml) | Everything below, as nested stacks. Submit this one to deploy the whole architecture |
| [`assets/eks-cluster-prerequisites.yaml`](./assets/eks-cluster-prerequisites.yaml) | VPC, subnets, NAT gateway, S3 and ECR endpoints, EFA-capable node security group, optional FSx for Lustre |
| [`assets/eks-cluster.yaml`](./assets/eks-cluster.yaml) | EKS cluster, access entries, EKS add-ons, system node group |
| [`assets/eks-add-gpu-nodegroup.yaml`](./assets/eks-add-gpu-nodegroup.yaml) | GPU launch template, GPU managed node group, device plugins and their verification |
| [`assets/eks-gpu-node-ami.yaml`](./assets/eks-gpu-node-ami.yaml) | Builds a node AMI with EC2 Image Builder and returns its id. Created by the root only when `NodeImagePackages` or `NodeImageRecipeArn` is set |
| [`eksctl/`](./eksctl/) | Reference eksctl manifests; see [section 7](#7-eksctl-manifests) |

## 1. What this gives a workload, and what it does not

| This architecture owns | An example owns |
|---|---|
| VPC, subnets, NAT, S3 and ECR endpoints | The container image and where it is built |
| EKS cluster, add-ons, system nodes | Serving framework and its operator, if it needs one |
| GPU nodes with EFA on every card that offers it | Prefill and decode pods, and how they are split |
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

Two caveats:

- An example written against SageMaker HyperPod EKS selects nodes by a HyperPod label
  (`sagemaker.amazonaws.com/instance-group-name`). On this cluster those selectors have to be replaced
  with `role=gpu` or with `node.kubernetes.io/instance-type`. The disaggregated-inference instructions
  make this replacement where needed.
- An example that mounts model weights from a `PersistentVolumeClaim` needs the volume to exist. With
  `DeployFsxLustre=true` the filesystem and its CSI driver are created here, and
  [`pv-fsx-lustre-static.yaml`](../../examples/use-cases/openvla-oft/kubernetes/libero/pv-fsx-lustre-static.yaml)
  binds to it from the `FsxFileSystemId`, `FsxDnsName` and `FsxMountName` outputs. Without it, use
  the node's local NVMe.

Other manifests may need changes: the NCCL test under
`micro-benchmarks/` has no toleration for this taint and pins a different instance type.

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
zone. Expect 20 to 25 minutes.

The GPU nodes boot from the AMI EKS resolves for the cluster version unless you say otherwise. Pass
`NodeAmiId` to boot from an image you already have, or pass the `NodeImage` inputs to build one in the
stack: the build starts once the network exists and runs alongside the cluster, so it adds the
difference between its own time and the cluster's. Setting both is refused before any resource is
created.

The minimum a build needs is `NodeImagePackages` and `NodeImageAssertPaths`, or `NodeImageRecipeArn`
on its own. The repository and assertion inputs belong to the packages path and are
refused unless `NodeImagePackages` is set, rather than ignored.

Then:

```bash
aws eks update-kubeconfig --name eks-gpu-cluster --region us-west-2
kubectl get nodes -l role=gpu -o custom-columns='NAME:.metadata.name,TYPE:.metadata.labels.node\.kubernetes\.io/instance-type,GPU:.status.allocatable.nvidia\.com/gpu,EFA:.status.allocatable.vpc\.amazonaws\.com/efa'
```

The `KubeconfigCommand` output holds the first command with this stack's cluster name and region.

**Before you deploy into a fresh account**, check the quotas below; a default account does not
necessarily have room for them. `g7e.12xlarge` needs 48 On-Demand vCPUs per node, and each stack
takes one Elastic IP for its NAT gateway.

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


## 3. GPU instance types

`GpuInstanceType` selects the type; the launch template derives the interface layout from the
`NicLayout` mapping, which records, per type, the number of network cards, how many of them carry
EFA, whether card 0 does, and the device index used on the other cards
(`describe-instance-types`, `NetworkInfo.MaximumNetworkCards` and `NetworkInfo.EfaInfo`).

| Instance type | GPUs | Network cards | EFA interfaces | Launched from this template |
|---|---|---|---|---|
| `g7.12xlarge` | 2 | 1 | 1 | yes, `eu-south-2`, from the resolved AMI and from an image built in the stack |
| `g7.24xlarge` | 4 | 1 | 1 | no |
| `g7.48xlarge` | 8 | 2 | 2 | no |
| `g7e.12xlarge` | 2 | 1 | 1 | no |
| `g7e.24xlarge` | 4 | 2 | 2 | no |
| `g7e.48xlarge` | 8 | 4 | 4 | no |
| `g6e.12xlarge` | 4 | 1 | 1 | no |
| `g6e.48xlarge` | 8 | 4 | 4 | no |
| `g5.12xlarge` | 4 | 1 | 1 | no |
| `g4dn.8xlarge` | 1 | 1 | 1 | yes, `us-west-2`, from the resolved AMI and from an image passed in as `NodeAmiId` |
| `p4d.24xlarge` | 8 | 4 | 4 | no |
| `p4de.24xlarge` | 8 | 4 | 4 | no |
| `p5.48xlarge` | 8 | 32 | 32 | no |
| `p5en.48xlarge` | 8 | 16 | 16 | no |
| `p6-b200.48xlarge` | 8 | 8 | 8 | no |
| `p6-b300.48xlarge` | 8 | 17 | 16 (card 0 is ENA only) | no |


### Building a node image

Build an image when the AMI EKS resolves does not carry what the nodes need — a driver newer than the
one it ships, a monitoring agent, a filesystem client.

Two ways. The first names the packages. Those values are comma-separated, so they go in a parameters
file: the CLI's shorthand syntax would split each one into a list.

```json
[
  {"ParameterKey": "NodeImageRepoPackages", "ParameterValue": "nvidia-release"},
  {"ParameterKey": "NodeImagePackages",
   "ParameterValue": "nvidia-open-595.91.07-1.amzn2023,nvidia-container-toolkit-1.19.1-1"},
  {"ParameterKey": "NodeImageAssertPaths",
   "ParameterValue": "/usr/bin/nvidia-container-runtime,/usr/bin/kubelet,/usr/bin/nodeadm"}
]
```

Both packages come from the repository `nvidia-release` brings, which is why `NodeImageRepoFiles` and
`NodeImageRepoKeys` are absent here. A payload whose packages need a repository the image does not
already have takes those two.

The second takes an EC2 Image Builder recipe you already maintain, as one value:
`NodeImageRecipeArn`. The stack then creates no component and no recipe of its own, and contributes
the build environment: the subnet and security group, the instance profile, and the wait.

`NodeAmiId` takes an image built anywhere, by any tool. What it has to carry is the same either way:
`nodeadm`, so EKS can bootstrap it against the `NodeConfig` the launch template passes; a driver that
enumerates the GPUs of the instance type it will run on; and the NVIDIA container toolkit, without
which the device plugin starts but finds no NVML and the nodes advertise no GPUs.

## 4. Parameters

Every parameter, with its default and what it affects, is in
[`docs/PARAMETERS.md`](./docs/PARAMETERS.md). The ones that decide a deploy:

| Parameter | Default | Notes |
|---|---|---|
| `PrimarySubnetAZ` | required | Zone of the GPU nodes and of the capacity reservation |
| `SecondarySubnetAZ` | required | Second zone, control plane only. Must differ from the first |
| `NodeAmiId` | empty | Boot the GPU nodes from an image you already have. Leave it empty to take the AMI EKS resolves, or give the `NodeImage` inputs to build one |
| `GpuInstanceType` | `g7e.12xlarge` | See section 3 |
| `GpuNodeCount` | `2` | Disaggregated inference needs at least 2. `0` is a smoke test, not a deploy |

## 5. Where model weights live

- `g7e.12xlarge` carries a single 3.8 TB NVMe drive, assembled as RAID0 under `/mnt/k8s-disks/0`.
  A `hostPath` there is the shortest path to a model cache, and it disappears with the node.
- `DeployFsxLustre=true` creates an FSx for Lustre filesystem and its CSI driver, for weights that
  outlive the nodes or are shared. Bind to it with
  [`pv-fsx-lustre-static.yaml`](../../examples/use-cases/openvla-oft/kubernetes/libero/pv-fsx-lustre-static.yaml),
  rendering `FSX_FILESYSTEM_ID`, `FSX_DNS_NAME` and `FSX_MOUNT_NAME` from the stack outputs of the
  same names.

## 6. Adding GPU capacity to an existing cluster

`assets/eks-add-gpu-nodegroup.yaml` deploys on its own against an EKS cluster on a version this
template offers, in a VPC with an EFA-capable security group. Pass the cluster's own
`KubernetesVersion`: the bootstrap downloads the matching `kubectl`.

Deploying it more than once against the same cluster, under different `NodeGroupName` values, is how a
cluster gets GPU node groups of different instance types or from different reservations. The device
plugins are a constraint on that: one release of each serves the whole cluster, and the versions it
runs have to match the versions the stack being deployed pins. A stack that pins a different version
fails rather than moving the release under the node groups that are already using it, so raise the
version on the stacks already on the cluster before adding one that pins a newer one.

```bash
cd architectures/amazon-eks
CLUSTER=my-cluster
aws cloudformation create-stack \
  --stack-name my-cluster-gpu \
  --region "$AWS_REGION" \
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

Requirements on the existing cluster:

- **`AuthenticationMode` of `API` or `API_AND_CONFIG_MAP`.** The stack grants the bootstrap access
  with an `AWS::EKS::AccessEntry`, which a `CONFIG_MAP`-only cluster rejects. Check with
  `aws eks describe-cluster --name $CLUSTER --query cluster.accessConfig.authenticationMode`.
- **A reachable API endpoint from CodeBuild**, which runs outside your VPC. On a cluster with only
  private endpoint access, the bootstrap cannot reach the API server; give the CodeBuild project a
  `VpcConfig`, or install the two device plugins yourself with the versions the template pins.

This is also how you change the GPU instance type. A managed node group cannot change its instance
type and its launch template version in one operation — EKS returns `Version and release version
updates cannot be combined with other updates` — so deploy a second node group stack, or replace
this one.

## 7. eksctl manifests

The manifests under [`eksctl/`](./eksctl/) create a comparable two-node-group topology with
[eksctl](https://eksctl.io). They pin older
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
aws cloudformation delete-stack --stack-name eks-gpu-cluster --region "$AWS_REGION"
```

Nested stacks are deleted with the root. Some resources are not, because the stack does not own them.
The first two can leave the VPC undeletable; the log groups only linger. Check them before deleting:

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
  reservation fails the deploy rather than delivering fewer nodes. `0` is the exception: a managed
  node group rejects a maximum of 0, so the maximum becomes 1 with nothing desired.
- Updating `GpuNodeCount` on a live stack re-runs the device-plugin verification, which can observe
  the node count from before the scaling change. Delete and recreate, or verify by hand afterwards.
- One NAT gateway and one Elastic IP per root or prerequisites deployment.
