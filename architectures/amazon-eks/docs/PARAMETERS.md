# Parameters

Every parameter of the five templates in [`../assets`](../assets). The root template
(`eks-gpu-cluster-deploy-all.yaml`) exposes the ones an operator chooses and passes the rest between
the child stacks.

## Root — `eks-gpu-cluster-deploy-all.yaml`

### Network

| Parameter | Type | Default | What it decides |
|---|---|---|---|
| `PrimarySubnetAZ` | AZ name | required | Zone of the public subnet, the node subnet, the NAT gateway and every GPU node. Must be the zone of the capacity reservation when one is used: EFA traffic and the cluster placement group do not cross zones |
| `SecondarySubnetAZ` | AZ name | required | Zone of the second private subnet, which exists only because EKS requires subnets in two zones. No nodes run there. Must differ from `PrimarySubnetAZ` (asserted at submit time) |
| `VpcCidr` | String | `10.0.0.0/16` | Split into three /20 subnets: public, node, control plane |

### Cluster

| Parameter | Type | Default | What it decides |
|---|---|---|---|
| `KubernetesVersion` | String | `1.36` | Control plane version, the AL2023 NVIDIA AMI release, and the `kubectl` the bootstrap downloads. `1.35` or `1.36`: the GPU node group reads the `kubectl` version from a mapping keyed by this value |
| `SystemInstanceType` | String | `m5.xlarge` | Instance type of the two system nodes. They carry CoreDNS and the node-feature-discovery master, which cannot run on a tainted GPU node. The default is chosen for Region coverage rather than speed: a type the Region does not offer fails the node group with `Unsupported - The requested configuration is currently not supported`, which names neither the type nor the Region. Check with `aws ec2 describe-instance-type-offerings --location-type availability-zone --filters Name=instance-type,Values=<type> Name=location,Values=<az>` |
| `ServiceIpv4Cidr` | String | `10.100.0.0/16` | CIDR the cluster allocates Service addresses from. Must not overlap `VpcCidr`. Set explicitly rather than left to EKS because a node group naming its own AMI has to repeat the value in its bootstrap configuration |
| `AdminRoleArn` | String | empty | An extra IAM principal that receives `AmazonEKSClusterAdminPolicy`. The principal that creates the stack always has it, so this is for the case where one principal provisions and another uses the cluster |

### GPU capacity

| Parameter | Type | Default | What it decides |
|---|---|---|---|
| `GpuInstanceType` | String | `g7e.12xlarge` | Instance type of the GPU node group, and through the `NicLayout` mapping the whole interface layout. See README section 3 for which types have been launched |
| `AmiType` | String | `AL2023_x86_64_NVIDIA` | EKS AMI type for the GPU nodes, used when no image input is given. Not an enumeration: EKS validates the value |
| `SystemAmiType` | String | `AL2023_x86_64_STANDARD` | As above, for the system nodes |
| `NodeAmiId` | String | empty | Node AMI for the GPU nodes. Leave the `NodeImage` inputs empty when using it. Any source: `awslabs/amazon-eks-ami`, EC2 Image Builder, or your own pipeline. It has to carry `nodeadm`, a driver that enumerates the instance type's GPUs, and the NVIDIA container toolkit |
| `NodeImagePackages` | String | empty | Comma-separated packages to build into the node image, each pinned to a version. Setting it builds an image and boots the GPU nodes from it. The build does not interpret the packages |
| `NodeImageRepoPackages` | String | empty | Refused without `NodeImagePackages`. Comma-separated packages whose job is to make the others resolvable, such as a vendor's `-release` package. Installed one at a time before anything else, and before the metadata refresh. A package that enables a repository has to be in place before a name from it can be resolved, which is what keeps install order out of `NodeImagePackages` |
| `NodeImageRepoFiles` | String | empty | Refused without `NodeImagePackages`. Comma-separated URLs of repository definitions the packages need |
| `NodeImageRepoKeys` | String | empty | Refused without `NodeImagePackages`. Comma-separated URLs of signing keys to import before installing |
| `NodeImageAssertPaths` | String | empty | Comma-separated paths the build requires to exist before publishing the image. Required with `NodeImagePackages`: a build with nothing to assert publishes an image whose contents were never checked |
| `NodeImageAssertCommands` | String | empty | Refused without `NodeImagePackages`. Semicolon-separated commands the build runs on the produced image, each of which has to exit zero. `NodeImageAssertPaths` proves a file arrived; these prove it works, which is a different claim, because a kernel module can be present as a file and fail to load |
| `NodeImageRecipeArn` | String | empty | An existing EC2 Image Builder recipe to build. Use it when you already maintain one: the stack contributes the build environment and nothing about the contents |
| `NodeImageVersion` | String | `1.0.0` | Semantic version of both the component and the recipe the build composes. Image Builder resources are immutable per version, and the recipe carries the payload as parameter values, so raise this whenever the `NodeImage` inputs change; reusing a version with different contents is rejected |
| `NodeImageBuildInstanceType` | String | `m5.large` | Instance type that builds the image. The build does not need a GPU, so it is deliberately not a GPU type; the node-side check is what proves the image drives its GPUs |
| `GpuNodeCount` | Number | `2` | Minimum, desired and maximum of the GPU node group, all the same value. A prefill/decode split needs at least 2. `0` creates the cluster and installs the device plugins with no GPU capacity, for testing template changes; a managed node group rejects a maximum of 0, so that case asks for 0 out of 1 |
| `GpuRootVolumeSize` | Number | `300` | Root EBS volume in GiB. Inference images are large, and they land on the root volume unless containerd is pointed at the NVMe volume |
| `CapacityReservationId` | String | empty | A targeted On-Demand Capacity Reservation or a Capacity Block. Empty launches On-Demand and consumes an open reservation whose attributes match |
| `CapacityReservationType` | String | `targeted-odcr` | `targeted-odcr` keeps the cluster placement group and targets the reservation. `capacity-block` sets `MarketType=capacity-block` and omits the placement group, which the Capacity Block already provides. `capacity-block` with an empty id is rejected at submit time |

### Optional

| Parameter | Type | Default | What it decides |
|---|---|---|---|
| `PrePullImage` | String | empty | An image pulled onto every GPU node by a DaemonSet, after the nodes are verified. The pull is started and not waited for: a multi-gigabyte pull must not be able to roll back a cluster. Watch it with `kubectl rollout status daemonset/prepull -n kube-system`. Rejected together with `GpuNodeCount=0` |
| `DeployFsxLustre` | String | `false` | `true` creates an FSx for Lustre filesystem and installs the `aws-fsx-csi-driver` add-on. Off by default because the GPU types carry local NVMe. The `FsxFileSystemId`, `FsxDnsName` and `FsxMountName` outputs are what a static `PersistentVolume` binds to |
| `FsxStorageCapacity` | Number | `1200` | Filesystem size in GiB, as an enumeration rather than a minimum: FSx accepts only certain sizes and refuses the rest minutes into the deploy. Add a size to the template if you need one the list does not cover |

### Template location

| Parameter | Type | Default | What it decides |
|---|---|---|---|
| `S3BucketName` | String | `awsome-distributed-ai` | Bucket the child templates are fetched from |
| `S3KeyPrefix` | String | `templates/amazon-eks/` | Key prefix of the child templates, trailing slash included |

Override both to deploy a copy that is not published yet, or to serve the templates from a bucket you
control.

## `eks-cluster-prerequisites.yaml`

`PrimarySubnetAZ`, `SecondarySubnetAZ`, `VpcCidr`, `DeployFsxLustre`, `FsxStorageCapacity` — same
meaning as above.

## `eks-cluster.yaml`

| Parameter | Type | Default | What it decides |
|---|---|---|---|
| `ClusterName` | String | required | Name of the cluster. The root passes its own stack name |
| `KubernetesVersion`, `AdminRoleArn`, `SystemInstanceType`, `SystemAmiType` | | | As above |
| `SystemNodeCount` | Number | `2` | Number of system nodes. Not exposed by the root template: two nodes carry CoreDNS and the node feature discovery pods |
| `ServiceIpv4Cidr` | String | `10.100.0.0/16` | As above |
| `PrivateSubnetId`, `ControlPlaneSubnetId`, `NodeSecurityGroupId` | ids | required | Outputs of the prerequisites stack |
| `FsxFileSystemId` | String | empty | A non-empty value installs the FSx CSI driver add-on |

Custom-AMI outputs: `ClusterEndpoint`, `ClusterCertificateAuthority` and
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
| `NodeGroupName` | String | `gpu` | Name of the managed node group. Change it to add a second GPU node group to a cluster that already has one; the nodes carry the label `role=gpu` either way |
| `AmiType` | String | `AL2023_x86_64_NVIDIA` | As above. Ignored when `NodeAmiId` is set, because that switches the node group to `CUSTOM` |
| `NodeRoleArn` | String | empty | Node IAM role. Empty creates one with the four managed policies a GPU node needs |
| `NodeAmiId` | String | empty | As above. Setting it switches the node group to `AmiType: CUSTOM`, which is why the three cluster values below are then required |
| `GpuPciVendorId` | String | `0x10de` | PCI vendor id of the accelerators, read only by the diagnosis that runs when a node comes up without them |
| `ClusterEndpoint`, `ClusterCertificateAuthority`, `ClusterServiceCidr` | String | empty | Required with `NodeAmiId`, and rejected as a set at submit time when one is missing. Read them from `aws eks describe-cluster` |
| `GpuInstanceType`, `GpuNodeCount`, `GpuRootVolumeSize`, `CapacityReservationId`, `CapacityReservationType`, `PrePullImage` | | | As above |

## `eks-gpu-node-ami.yaml`

Deployed by the root when `NodeImagePackages` or `NodeImageRecipeArn` is set, or on its own. Either
way exactly one of `Packages` and `RecipeArn` is required, and `Packages` requires `AssertPaths`,
both asserted at submit time.

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `VpcId`, `PrivateSubnetId` | Id | required | Where the build instance runs. The subnet needs outbound internet access for the package repositories |
| `BuildInstanceType` | String | `m5.large` | The build needs no GPU |
| `KubernetesVersion` | String | `1.36` | Selects the EKS-optimised AL2023 **standard** parent AMI |
| `RepoPackages` | String | empty | Packages that make the others resolvable, installed first, one at a time |
| `Packages` | String | empty | Comma-separated packages, each pinned to a version, installed in one transaction |
| `RepoFiles`, `RepoKeys` | String | empty | Comma-separated URLs of repository definitions and signing keys, reachable from the build subnet |
| `AssertPaths` | String | empty | Comma-separated paths the produced image must contain |
| `AssertCommands` | String | empty | Semicolon-separated commands that must exit zero on the produced image |
| `RecipeArn` | String | empty | An existing EC2 Image Builder recipe to build instead of the one this template composes. Its parent image needs the Systems Manager agent, which Image Builder uses to reach the build instance |
| `ComponentVersion`, `RecipeVersion` | String | `1.0.0` | Image Builder resources are immutable per version. The recipe carries the inputs above as component parameter values, so changing the payload changes the recipe and needs `RecipeVersion` raised; the root passes one value to both |
