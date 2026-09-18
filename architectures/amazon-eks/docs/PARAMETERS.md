# Parameters

Every parameter of the four templates in [`../assets`](../assets). The root template
(`eks-gpu-cluster-deploy-all.yaml`) exposes the ones an operator chooses and passes the rest between
the child stacks, so a nested deploy never asks for a subnet id or a security group id.

## Root — `eks-gpu-cluster-deploy-all.yaml`

### Network

| Parameter | Type | Default | What it decides |
|---|---|---|---|
| `PrimarySubnetAZ` | AZ name | required | Zone of the public subnet, the node subnet, the NAT gateway and every GPU node. Must be the zone of the capacity reservation when one is used: EFA traffic and the cluster placement group do not cross zones |
| `SecondarySubnetAZ` | AZ name | required | Zone of the second private subnet, which exists only because EKS requires subnets in two zones. No nodes run there. Must differ from `PrimarySubnetAZ`, and a `Rules` assertion rejects the stack at submit time if it does not |
| `VpcCidr` | String | `10.0.0.0/16` | Split into three /20 subnets: public, node, control plane |

### Cluster

| Parameter | Type | Default | What it decides |
|---|---|---|---|
| `KubernetesVersion` | String | `1.36` | Control plane version, the AL2023 NVIDIA AMI release, and the `kubectl` the bootstrap downloads. `AllowedValues` are the versions this architecture has been deployed on; see [COMPATIBILITY.md](./COMPATIBILITY.md) |
| `SystemInstanceType` | String | `m5.xlarge` | Instance type of the two system nodes. They carry CoreDNS and the node-feature-discovery master, which cannot run on a tainted GPU node. The default is the type with the widest Region coverage, not the fastest: a type the Region does not offer fails the node group with `Unsupported - The requested configuration is currently not supported`, which names nothing. Check first with `aws ec2 describe-instance-type-offerings --location-type availability-zone --filters Name=instance-type,Values=<type> Name=location,Values=<az>` |
| `AdminRoleArn` | String | empty | An extra IAM principal that receives `AmazonEKSClusterAdminPolicy`. The principal that creates the stack always has it, so this is for the case where one principal provisions and another uses the cluster — which is what a pre-provisioned workshop account is |

### GPU capacity

| Parameter | Type | Default | What it decides |
|---|---|---|---|
| `GpuInstanceType` | String | `g7e.12xlarge` | Instance type of the GPU node group, and through the `NicLayout` mapping the whole interface layout. See README section 3 for which types have been launched |
| `NodeAmiId` | String | empty | Custom node AMI for the GPU nodes. Empty uses the EKS-optimised AL2023 NVIDIA AMI for `KubernetesVersion`. The `g7` family requires one and a `Rules` assertion says so at submit time: that AMI's driver does not enumerate its RTX PRO GPUs, so the nodes would join, advertise their EFA interface and never advertise `nvidia.com/gpu`. [`ami/`](../../../ami) in this repository builds a node AMI with a driver that does |
| `GpuNodeCount` | Number | `2` | Minimum, desired and maximum of the GPU node group, all the same value. A prefill/decode split needs at least 2. `0` creates the cluster and installs the device plugins with no GPU capacity: useful to test template changes cheaply, and not evidence that GPU nodes work |
| `GpuRootVolumeSize` | Number | `300` | Root EBS volume in GiB. Inference images are large, and they land on the root volume unless containerd is pointed at the NVMe volume |
| `CapacityReservationId` | String | empty | A targeted On-Demand Capacity Reservation or a Capacity Block. Empty launches On-Demand and consumes an open reservation whose attributes match |
| `CapacityReservationType` | String | `targeted-odcr` | `targeted-odcr` keeps the cluster placement group and targets the reservation. `capacity-block` sets `MarketType=capacity-block` and omits the placement group, which the Capacity Block already provides. `capacity-block` with an empty id is rejected at submit time |

### Optional

| Parameter | Type | Default | What it decides |
|---|---|---|---|
| `PrePullImage` | String | empty | An image pulled onto every GPU node by a DaemonSet, after the nodes are verified. The pull is started and not waited for: a multi-gigabyte pull must not be able to roll back a cluster. Watch it with `kubectl rollout status daemonset/prepull -n kube-system`. Rejected together with `GpuNodeCount=0`, which has nowhere to pull to |
| `DeployFsxLustre` | String | `false` | `true` creates an FSx for Lustre filesystem and installs the `aws-fsx-csi-driver` add-on. Off by default because the GPU types carry local NVMe. The `PersistentVolume` is a Kubernetes object and is not created by the stack; the manifest is in [OPERATIONS.md](./OPERATIONS.md) |
| `FsxStorageCapacity` | Number | `1200` | Filesystem size in GiB. `PERSISTENT_2` takes multiples of 1200 |

### Template location

| Parameter | Type | Default | What it decides |
|---|---|---|---|
| `S3BucketName` | String | `awsome-distributed-ai` | Bucket the three child templates are fetched from |
| `S3KeyPrefix` | String | `templates/amazon-eks/` | Key prefix of the child templates, trailing slash included |

Override both to deploy a copy that is not published yet, or to serve the templates from an event's
own bucket. See [DEPLOY-TESTING.md](./DEPLOY-TESTING.md) and [WORKSHOP-STUDIO.md](./WORKSHOP-STUDIO.md).

## `eks-cluster-prerequisites.yaml`

`PrimarySubnetAZ`, `SecondarySubnetAZ`, `VpcCidr`, `DeployFsxLustre`, `FsxStorageCapacity` — same
meaning as above.

## `eks-cluster.yaml`

| Parameter | Type | Default | What it decides |
|---|---|---|---|
| `ClusterName` | String | required | Name of the cluster. The root passes its own stack name |
| `KubernetesVersion`, `AdminRoleArn`, `SystemInstanceType` | | | As above |
| `SystemNodeCount` | Number | `2` | Number of system nodes. Not exposed by the root template |
| `PrivateSubnetId`, `ControlPlaneSubnetId`, `NodeSecurityGroupId` | ids | required | Outputs of the prerequisites stack |
| `FsxFileSystemId` | String | empty | A non-empty value installs the FSx CSI driver add-on |

Outputs added for the custom-AMI path: `ClusterEndpoint`, `ClusterCertificateAuthority` and
`ClusterServiceCidr`. A node group that names its own AMI needs all three, because EKS merges no
bootstrap user data once a launch template carries an `ImageId`.

## `eks-add-gpu-nodegroup.yaml`

| Parameter | Type | Default | What it decides |
|---|---|---|---|
| `ClusterName` | String | required | Cluster the node group joins |
| `KubernetesVersion` | String | `1.36` | Must match the cluster. Selects the AMI release and the bootstrap's `kubectl` |
| `PrivateSubnetId` | id | required | Subnet for the GPU nodes, in the zone of the reservation |
| `NodeSecurityGroupId` | id | required | A security group that allows all traffic between its own members, which is EFA's requirement |
| `ClusterSecurityGroupId` | id | required | The cluster's own security group. EKS stops attaching it once the launch template names any security group, and a node without it never joins |
| `NodeRoleArn` | String | empty | Node IAM role. Empty creates one with the four managed policies a GPU node needs |
| `NodeAmiId` | String | empty | As above. Setting it switches the node group to `AmiType: CUSTOM` and moves the whole `NodeConfig` — cluster block, labels, taint, local storage — into the launch template's user data |
| `ClusterEndpoint`, `ClusterCertificateAuthority`, `ClusterServiceCidr` | String | empty | Required with `NodeAmiId`, and rejected as a set at submit time when one is missing. Read them from `aws eks describe-cluster` |
| `GpuInstanceType`, `GpuNodeCount`, `GpuRootVolumeSize`, `CapacityReservationId`, `CapacityReservationType`, `PrePullImage` | | | As above |
