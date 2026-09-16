# Operations

## 1. When the stack fails, read the bootstrap log first

The GPU node group stack ends with a custom resource that runs a CodeBuild build: it installs the two
device plugins and then waits until every GPU node advertises the exact number of GPUs and EFA
interfaces its instance type has. Almost every failure after the node group is created is in that
build, and the stack event carries its reason.

```bash
STACK=eks-gpu-cluster
aws cloudformation describe-stack-events --stack-name $STACK \
  --query 'StackEvents[?ResourceStatus==`CREATE_FAILED`].[LogicalResourceId,ResourceStatusReason]' \
  --output table

# The log group is a stack output; the build id is in the failure reason.
aws logs tail "$(aws cloudformation describe-stacks --stack-name $STACK \
  --query 'Stacks[0].Outputs[?OutputKey==`BootstrapLogGroup`].OutputValue' --output text)" --since 1h
```

The build reports which stage failed: `tool install`, `device plugin install`, `node capability
check` or `image pre-pull`. The stages are separate so that "the charts did not install" and "a node
did not advertise what it should" are distinguishable failures.

Two properties of the failure path are worth knowing:

- The build answers CloudFormation from every path it can, including a failure while downloading its
  own tools and a shell error the script did not anticipate. A stack that is waiting on the bootstrap
  is waiting because the build is still running, not because nobody will answer.
- The custom resource declares `ServiceTimeout: 2700`, so a build that dies without answering ends
  the wait in 45 minutes rather than the default hour. GPU capacity bills for that whole wait.

### Repairing instead of rolling back

A device-plugin failure rolls the stack back and deletes a cluster that took 25 minutes to build.
When you are iterating, deploy with rollback disabled, fix the cluster by hand, and delete when done:

```bash
aws cloudformation create-stack --stack-name eks-gpu-cluster --disable-rollback ...
```

The charts can then be installed by hand with the same versions the bootstrap uses (see
[COMPATIBILITY.md](./COMPATIBILITY.md) for the pinned versions), and the node capability check is
one `kubectl get nodes` away. The bootstrap was left able to fail the stack because the alternative
— reporting success and letting the participant discover there are no device plugins — is worse in a
workshop.

## 2. When the GPU node group stays in CREATING with no instances

A node group whose Availability Zone has no capacity for the instance type does not fail. The
Auto Scaling group retries every two minutes, the node group reports `CREATING` with an empty
`health.issues`, and the stack waits. The retry loop is only visible one level down:

```bash
CLUSTER=eks-gpu-cluster
ASG=$(aws eks describe-nodegroup --cluster-name $CLUSTER --nodegroup-name gpu \
  --query 'nodegroup.resources.autoScalingGroups[0].name' --output text)
aws autoscaling describe-scaling-activities --auto-scaling-group-name "$ASG" --max-items 3 \
  --query 'Activities[].[StatusCode,StatusMessage]' --output text
```

`InsufficientInstanceCapacity` there names the zones that do have capacity. CloudFormation does
report it in the end — the node group gives up after about 35 minutes and the stack event carries
`Issue(Code=AsgInstanceLaunchFailures, Message=Could not launch On-Demand Instances.
InsufficientInstanceCapacity ...)` — but the Auto Scaling activity says the same thing within two
minutes, which is the difference between waiting and knowing.

The zone is fixed when the stack is created, and the private subnet is in it, so the fix is a new
stack in another zone rather than an update. No API reports available capacity before a launch: an
instance type being *offered* in a zone (`describe-instance-type-offerings`) says nothing about
whether it can be launched right now. A capacity reservation is the only thing that answers the
question in advance, and it answers immediately — `create-capacity-reservation` either succeeds or
returns `InsufficientInstanceCapacity` in seconds, which makes it a probe as well as a guarantee:

```bash
aws ec2 create-capacity-reservation --instance-type g7e.12xlarge --instance-platform Linux/UNIX \
  --availability-zone us-west-2b --instance-count 2 --instance-match-criteria targeted \
  --end-date-type limited --end-date "$(date -u -v+3H +%Y-%m-%dT%H:%M:%SZ)"
```

## 3. Watching the image pre-pull

`PrePullImage` starts a DaemonSet and does not wait for it, so the stack completes while the pull is
still running:

```bash
kubectl rollout status daemonset/prepull -n kube-system --timeout=40m
kubectl get pods -n kube-system -l app=prepull -o wide
```

A failing pull shows up as `ImagePullBackOff` there and nowhere else. Delete the DaemonSet when the
image is cached; it exists only to hold the layers on the node.

## 4. FSx for Lustre for model weights

`DeployFsxLustre=true` creates the filesystem and installs the CSI driver. The `PersistentVolume` is
a Kubernetes object, so the stack does not create it. Take the three values from the stack outputs
and apply:

The manifest below uses 1200Gi because that is the default `FsxStorageCapacity`. Set both `storage`
values to whatever the filesystem actually has; the CSI driver does not enforce it, and a claim that
asks for more than the filesystem holds still binds.

```bash
STACK=eks-gpu-cluster
FS_ID=$(aws cloudformation describe-stacks --stack-name $STACK \
  --query 'Stacks[0].Outputs[?OutputKey==`FsxFileSystemId`].OutputValue' --output text)
FS_DNS=$(aws cloudformation describe-stacks --stack-name $STACK \
  --query 'Stacks[0].Outputs[?OutputKey==`FsxDnsName`].OutputValue' --output text)
FS_MOUNT=$(aws cloudformation describe-stacks --stack-name $STACK \
  --query 'Stacks[0].Outputs[?OutputKey==`FsxMountName`].OutputValue' --output text)

cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: PersistentVolume
metadata:
  name: model-cache
spec:
  capacity:
    storage: 1200Gi
  volumeMode: Filesystem
  accessModes: [ReadWriteMany]
  persistentVolumeReclaimPolicy: Retain
  storageClassName: ""
  mountOptions: [flock]
  csi:
    driver: fsx.csi.aws.com
    volumeHandle: ${FS_ID}
    volumeAttributes:
      dnsname: ${FS_DNS}
      mountname: ${FS_MOUNT}
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-cache
spec:
  accessModes: [ReadWriteMany]
  storageClassName: ""
  resources:
    requests:
      storage: 1200Gi
  volumeName: model-cache
EOF
```

An example's own PVC name may differ; either rename this claim to match it or point the example at
`model-cache`. Without FSx, the alternative is the node's local NVMe at `/mnt/k8s-disks/0`, which
`examples/inference/vllm/dsv3-uccl-nixl` uses directly.

This is a **static** volume: the filesystem exists, and the manifest points at it. The CSI driver's
controller needs no AWS permissions for that, and the stack gives it none. Dynamic provisioning — a
`StorageClass` with `provisioner: fsx.csi.aws.com` that creates filesystems on demand — does need an
IAM role for the driver's service account, which this architecture does not create.

## 5. Quotas that block a first deploy

| Limit | Default | Why it bites |
|---|---|---|
| On-Demand vCPUs, G and VT family (`L-DB2E81BA`) | varies, often 0 in a new account | `g7e.12xlarge` needs 48 per node. The node group fails 20 minutes into the deploy |
| On-Demand vCPUs, P family (`L-417A185B`) | often 0 | Same, for p4/p5/p6 |
| Elastic IPs (`L-0263D0A3`) | 5 unless raised | One NAT gateway, so one Elastic IP, per stack. Check the account rather than assuming the default |
| CodeBuild concurrent builds | low in a freshly vended account | The bootstrap is a build. `StartBuild` failing is reported by the trigger function as a stack failure with the API error in the reason |

Raise them before an event, not during it.

## 6. IAM

The stack creates four roles: the cluster role, the node role, the CodeBuild bootstrap role, and the
Lambda role that starts the build. Two properties are worth stating rather than leaving to be found:

- The bootstrap role holds an EKS access entry with `AmazonEKSClusterAdminPolicy`, because installing
  a DaemonSet across the cluster requires it. **Anyone who can call `codebuild:StartBuild` on the
  bootstrap project can therefore act as cluster administrator.** In an event account that is
  acceptable; in a shared account, restrict `codebuild:StartBuild`.
- The trigger function passes CloudFormation's response URL to the build as an environment override.
  A principal who can read builds (`codebuild:BatchGetBuilds`) can read that URL while the build
  runs and answer CloudFormation in its place.

This architecture does not ship least-privilege policy stacks for cluster administrators and users;
`architectures/aws-pcs/assets/cluster-admin-iam.yaml` is the pattern if you need them.

## 7. Known limits

- The GPU instance type is fixed at node group creation. EKS rejects changing the instance type and
  the launch template version in one update: `Version and release version updates cannot be combined
  with other updates`. Deploy a second GPU node group stack, or replace the existing one.
- `GpuNodeCount` sets minimum, desired and maximum together, so a partly available reservation fails
  the deploy instead of delivering fewer nodes. This is intentional for a session that needs exactly
  two nodes, and wrong for a cluster that should degrade gracefully.
- Updating `GpuNodeCount` re-runs the verification, which can read the node count from before the
  scaling change and pass early. Verify by hand after an update, or recreate the stack.
- Recreating a stack with the same name while the previous delete is in flight collides on the
  launch template, CodeBuild project and log group names, all of which derive from the stack name.
