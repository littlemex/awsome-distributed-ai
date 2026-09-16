# Infrastructure Tests (row 1)

Validates the templates without buying GPU capacity: the lint, `validate-template`, a
`GpuNodeCount=0` deploy, the submit-time parameter rules, and the path that deploys nested
children from a bucket you control.

> **A `GpuNodeCount=0` deploy is a smoke test, not an end-to-end test.** It proves the
> network, the cluster, the system node group, the add-ons and the bootstrap's control flow.
> It proves nothing about GPUs, EFA, NVMe or the device plugins actually advertising
> resources, because there is no GPU node for them to run on. The end-to-end claim comes from
> [`gpu-efa-test.md`](./gpu-efa-test.md) and [`disagg-smoke-test.md`](./disagg-smoke-test.md).

Placeholders used throughout:

```bash
export REGION=us-east-2
export AZ_A=us-east-2b            # PrimarySubnetAZ — must offer the GPU type you will use later
export AZ_B=us-east-2a            # SecondarySubnetAZ — control plane only, any other AZ
export STACK=eks-gpu-infra
```

---

## Test 1.1 — lint

```bash
bash architectures/amazon-eks/tests/lint-templates.sh
```

**Expected:** `[OK] lint-templates.sh: 0 failures, N skipped`, exit status 0. With AWS
credentials and `helm` on `PATH`, `N` is 0 except for the `PrimaryEfa` line the EC2 API cannot
confirm. Any `FAIL` line blocks the merge; see the check table in [`README.md`](./README.md).

---

## Test 1.2 — `validate-template` on all four templates

```bash
cd architectures/amazon-eks
for t in assets/*.yaml; do
  echo "== $t"
  aws cloudformation validate-template --template-body "file://$t" \
    --region "$REGION" --query 'Parameters[].ParameterKey' --output text
done
```

**Expected:** each template prints its parameter keys and no error.
`eks-gpu-cluster-deploy-all.yaml` prints the 16 root parameters (including `S3BucketName` and
`S3KeyPrefix`); `eks-add-gpu-nodegroup.yaml` prints `ClusterName`, `PrivateSubnetId` and
`NodeSecurityGroupId` among them, which is what lets it deploy standalone against a cluster
this stack did not create.

This is the same call the publish workflow makes, so a template that fails here never reaches
the production bucket.

---

## Test 1.3 — submit-time parameter rules are rejected before any resource is created

Three deliberate mistakes. Each must be refused by the `create-stack` call itself, so nothing
is charged and nothing has to be rolled back.

```bash
# (a) the two AZs are the same — EFA and the placement group are single-AZ, and the control
#     plane needs two distinct subnets
aws cloudformation create-stack --stack-name rule-a --region "$REGION" \
  --template-body file://assets/eks-gpu-cluster-deploy-all.yaml \
  --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM \
  --parameters ParameterKey=PrimarySubnetAZ,ParameterValue=$AZ_A \
               ParameterKey=SecondarySubnetAZ,ParameterValue=$AZ_A

# (b) a Capacity Block without an id
aws cloudformation create-stack --stack-name rule-b --region "$REGION" \
  --template-body file://assets/eks-gpu-cluster-deploy-all.yaml \
  --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM \
  --parameters ParameterKey=PrimarySubnetAZ,ParameterValue=$AZ_A \
               ParameterKey=SecondarySubnetAZ,ParameterValue=$AZ_B \
               ParameterKey=CapacityReservationType,ParameterValue=capacity-block

# (c) no GPU nodes, but an image to pre-pull onto them
aws cloudformation create-stack --stack-name rule-c --region "$REGION" \
  --template-body file://assets/eks-gpu-cluster-deploy-all.yaml \
  --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM \
  --parameters ParameterKey=PrimarySubnetAZ,ParameterValue=$AZ_A \
               ParameterKey=SecondarySubnetAZ,ParameterValue=$AZ_B \
               ParameterKey=GpuNodeCount,ParameterValue=0 \
               ParameterKey=PrePullImage,ParameterValue=public.ecr.aws/docker/library/busybox:1.36
```

**Expected:** all three fail immediately with
`An error occurred (ValidationError) ... Parameter validation failed` and a message naming the
rule, and

```bash
aws cloudformation describe-stacks --stack-name rule-a --region "$REGION"
```

**Expected:** `Stack with id rule-a does not exist` — the rule ran at submit time, so no stack
and no VPC were created. Repeat for `rule-b` and `rule-c`.

A rule that is *missing* shows up here as a stack that starts creating. If any of the three
starts creating, delete it and treat it as a failure.

---

## Test 1.4 — deploy with no GPU nodes

```bash
cd architectures/amazon-eks
aws cloudformation create-stack \
  --stack-name "$STACK" --region "$REGION" \
  --template-body file://assets/eks-gpu-cluster-deploy-all.yaml \
  --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM \
  --parameters ParameterKey=PrimarySubnetAZ,ParameterValue=$AZ_A \
               ParameterKey=SecondarySubnetAZ,ParameterValue=$AZ_B \
               ParameterKey=GpuNodeCount,ParameterValue=0
```

> `--template-body` works here because the root template is deployed from your checkout while
> its children are still fetched from `S3BucketName`/`S3KeyPrefix`. To test **edited
> children**, use Test 1.5 instead.

```bash
aws cloudformation wait stack-create-complete --stack-name "$STACK" --region "$REGION"
aws cloudformation describe-stacks --stack-name "$STACK" --region "$REGION" \
  --query 'Stacks[0].[StackStatus]' --output text
```

**Expected:** `CREATE_COMPLETE` (roughly 15-20 minutes: VPC and endpoints, then the cluster,
then the system node group and the bootstrap).

### Outputs

```bash
aws cloudformation describe-stacks --stack-name "$STACK" --region "$REGION" \
  --query 'Stacks[0].Outputs[].[OutputKey,OutputValue]' --output table
```

**Expected:** twelve outputs. `ClusterName`, `ClusterArn`, `Region`, `KubeconfigCommand`,
`VpcId`, `PrivateSubnetId`, `GpuNodeGroupName`, `GpuInstanceType` and `BootstrapLogGroup` all
carry a value; `FsxFileSystemId`, `FsxDnsName` and `FsxMountName` are **empty strings**,
because `DeployFsxLustre` defaults to `false`. An empty `BootstrapLogGroup` is a failure — it
is the only way to read why the bootstrap failed.

### Cluster and add-ons

```bash
CLUSTER=$(aws cloudformation describe-stacks --stack-name "$STACK" --region "$REGION" \
  --query 'Stacks[0].Outputs[?OutputKey==`ClusterName`].OutputValue' --output text)

aws eks describe-cluster --name "$CLUSTER" --region "$REGION" \
  --query 'cluster.[status,version,platformVersion]' --output text

aws eks list-addons --cluster-name "$CLUSTER" --region "$REGION" --output text
for a in $(aws eks list-addons --cluster-name "$CLUSTER" --region "$REGION" --query 'addons[]' --output text); do
  printf '%-28s %s\n' "$a" "$(aws eks describe-addon --cluster-name "$CLUSTER" --addon-name "$a" \
    --region "$REGION" --query 'addon.[status,addonVersion]' --output text)"
done
```

**Expected:** `ACTIVE` and the `KubernetesVersion` you deployed (`1.36` by default). Every
add-on is `ACTIVE` with a version string, and **`aws-fsx-csi-driver` is not in the list** —
with `DeployFsxLustre=false` there is no filesystem, so installing its driver would be dead
weight. A `DEGRADED` add-on is a failure even though the stack reports `CREATE_COMPLETE`.

### The system node group and the two Helm releases

```bash
eval "$(aws cloudformation describe-stacks --stack-name "$STACK" --region "$REGION" \
  --query 'Stacks[0].Outputs[?OutputKey==`KubeconfigCommand`].OutputValue' --output text)"

kubectl get nodes -L role
kubectl get nodes -l role=system -o jsonpath='{range .items[*]}{.metadata.name} {.status.conditions[?(@.type=="Ready")].status}{"\n"}{end}'
helm list -A
kubectl get ds -A -o custom-columns='NS:.metadata.namespace,NAME:.metadata.name,DESIRED:.status.desiredNumberScheduled,READY:.status.numberReady'
```

**Expected:** the `KubeconfigCommand` output runs as printed and `kubectl` reaches the
cluster. Exactly `SystemNodeCount` nodes (2 by default), all labelled `role=system` and all
`True` for `Ready`; **no node labelled `role=gpu`**. `helm list -A` shows both releases
`deployed`: `nvdp` in namespace `nvidia-device-plugin` and `aws-efa-k8s-device-plugin` in
`kube-system`. Both plugin DaemonSets show `DESIRED 0` — they select GPU nodes, and there are
none. A DaemonSet with `DESIRED 2` here means it is landing on the system nodes, which is a
failure: the plugins would advertise resources on nodes that have no GPU.

### The bootstrap answered CloudFormation, and did not wait for nodes that do not exist

```bash
LOG_GROUP=$(aws cloudformation describe-stacks --stack-name "$STACK" --region "$REGION" \
  --query 'Stacks[0].Outputs[?OutputKey==`BootstrapLogGroup`].OutputValue' --output text)
echo "$LOG_GROUP"

# BootstrapProjectName is an output of the GPU child stack, not the root
GPU_STACK=$(aws cloudformation describe-stack-resources --stack-name "$STACK" --region "$REGION" \
  --query 'StackResources[?ResourceType==`AWS::CloudFormation::Stack`].PhysicalResourceId' \
  --output text | tr '\t' '\n' | grep -i gpu)
PROJECT=$(aws cloudformation describe-stacks --stack-name "$GPU_STACK" --region "$REGION" \
  --query 'Stacks[0].Outputs[?OutputKey==`BootstrapProjectName`].OutputValue' --output text)

BUILD=$(aws codebuild list-builds-for-project --project-name "$PROJECT" --region "$REGION" \
  --query 'ids[0]' --output text)
aws codebuild batch-get-builds --ids "$BUILD" --region "$REGION" \
  --query 'builds[0].[buildStatus,currentPhase,buildComplete]' --output text
```

**Expected:** a non-empty log group name, and `SUCCEEDED SUCCEEDED True` for the one build,
with the whole stack reaching `CREATE_COMPLETE` in a single pass (no `UPDATE_ROLLBACK`, no
second build).

The point of running this with zero GPU nodes is the readiness check's exit condition: it must
notice there are no GPU nodes to wait for and finish, rather than polling until the CodeBuild
timeout. A `CREATE_COMPLETE` that took longer than about 25 minutes, or a
`CREATE_FAILED ... didn't respond` on the custom resource, is that bug.

---

## Test 1.5 — nested deploy from a bucket you control

Editing a child template does nothing until the child is hosted where CloudFormation can
fetch it. This is the path for a branch or a fork.

```bash
export BUCKET=my-eks-templates          # a bucket you own; it can be private
export PREFIX=templates/amazon-eks/     # keep the trailing slash

cd architectures/amazon-eks
aws s3 sync assets/ "s3://${BUCKET}/${PREFIX}" --exclude '*' --include '*.yaml'

aws cloudformation create-stack \
  --stack-name "${STACK}-own-bucket" --region "$REGION" \
  --template-url "https://${BUCKET}.s3.amazonaws.com/${PREFIX}eks-gpu-cluster-deploy-all.yaml" \
  --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM \
  --parameters ParameterKey=PrimarySubnetAZ,ParameterValue=$AZ_A \
               ParameterKey=SecondarySubnetAZ,ParameterValue=$AZ_B \
               ParameterKey=GpuNodeCount,ParameterValue=0 \
               ParameterKey=S3BucketName,ParameterValue=$BUCKET \
               ParameterKey=S3KeyPrefix,ParameterValue=$PREFIX
aws cloudformation wait stack-create-complete --stack-name "${STACK}-own-bucket" --region "$REGION"
```

Then prove the children really came from your copy, rather than from the published bucket:

```bash
CHILD=$(aws cloudformation describe-stack-resources --stack-name "${STACK}-own-bucket" \
  --region "$REGION" \
  --query 'StackResources[?ResourceType==`AWS::CloudFormation::Stack`].PhysicalResourceId' \
  --output text | tr '\t' '\n' | grep -i prereq)

aws cloudformation get-template --stack-name "$CHILD" --region "$REGION" \
  --template-stage Original --query 'TemplateBody' --output text > /tmp/deployed-prereq.yaml
diff /tmp/deployed-prereq.yaml assets/eks-cluster-prerequisites.yaml && echo "child template is your copy"
```

**Expected:** `CREATE_COMPLETE`, and `diff` reports no differences (a trailing-newline
difference is acceptable; any other difference means the deploy used the published child, so
your edit was not tested). Leaving `S3BucketName` at its default while deploying an edited
root is the mistake this test exists to catch.

---

## Cleanup

```bash
aws cloudformation delete-stack --stack-name "$STACK" --region "$REGION"
aws cloudformation delete-stack --stack-name "${STACK}-own-bucket" --region "$REGION"
aws cloudformation wait stack-delete-complete --stack-name "$STACK" --region "$REGION"
```

A zero-GPU stack still creates a NAT gateway, an Elastic IP, interface endpoints and a
control-plane log group. Run [`cleanup-test.md`](./cleanup-test.md) after this test too — the
log group survives the delete on purpose, and it is the one that quietly keeps costing money.
