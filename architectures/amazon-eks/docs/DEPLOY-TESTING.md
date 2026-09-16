# Deploying a change that is not published yet

The one-click link and the CLI command in the [README](../README.md#2-quick-start) fetch from the
public bucket, which holds what is on `main`. A change on a branch or in a fork is not there, and the
root template fetches its three children by URL, so pointing CloudFormation at a local root file is
not enough: the children would still come from the public bucket, and you would be testing your new
root against the published children.

Host all four templates in a bucket you control and override two parameters.

## 1. Upload

Run from `architectures/amazon-eks`:

```bash
BUCKET=my-eks-templates          # a bucket you own, in any Region
PREFIX=templates/amazon-eks/     # keep the trailing slash
REGION=us-west-2

aws s3 sync assets/ "s3://${BUCKET}/${PREFIX}" --exclude "*" --include "*.yaml"
```

The bucket can be private: CloudFormation reads the children with your credentials. Put it in the
Region you deploy into — a cross-Region template URL works in most cases and is one more variable
when a nested stack fails to fetch. Re-run the sync after every change, and before updating a
deployed stack, so a newly added child exists before the root references it.

## 2. Deploy against your copy

```bash
aws cloudformation create-stack \
  --stack-name eks-gpu-test \
  --template-url "https://${BUCKET}.s3.amazonaws.com/${PREFIX}eks-gpu-cluster-deploy-all.yaml" \
  --capabilities CAPABILITY_IAM \
  --region "$REGION" \
  --parameters \
    ParameterKey=PrimarySubnetAZ,ParameterValue=${REGION}a \
    ParameterKey=SecondarySubnetAZ,ParameterValue=${REGION}b \
    ParameterKey=S3BucketName,ParameterValue=$BUCKET \
    ParameterKey=S3KeyPrefix,ParameterValue=$PREFIX \
    ParameterKey=GpuInstanceType,ParameterValue=g7e.12xlarge \
    ParameterKey=GpuNodeCount,ParameterValue=2
```

Leaving `S3BucketName` at its default is the mistake this page exists to prevent: the root would be
yours and the three stacks it creates would be the published ones.

## 3. The cheap paths, and what they do not tell you

| What you run | Cost | What it establishes | What it cannot establish |
|---|---|---|---|
| `tests/lint-templates.sh` | free | Mapping, chart allowlist, generated block, documented parameters, links | Anything about AWS behaviour |
| `aws cloudformation validate-template` | free | The template parses and its functions are well formed | Conditions are not evaluated; no resource is checked |
| `create-launch-template` from the rendered launch template | cents | The interface list is accepted as a request shape | Whether an interface can carry EFA on that card. `run-instances --dry-run` accepts EFA on a card that does not support it |
| `GpuNodeCount=0` deploy | the cluster, two system nodes, a NAT gateway and two interface endpoints | VPC, cluster, system nodes, add-ons, the bootstrap's tool install and both Helm installs | Nothing about GPU nodes. With no nodes to check, the verification step has nothing to observe and says so in its reason |
| `GpuNodeCount=2` deploy | the instance price | The whole contract in README section 1 | Only for the type you launched |

A `GpuNodeCount=0` deploy does exercise the bootstrap's first two stages, so a broken buildspec or a
chart version that does not exist is caught there. What it cannot reach is everything that needs a
GPU node: the interface layout, the AMI's driver, the allocatable resources, the RAID0 volume and
the pre-pull. [`../tests/gpu-efa-test.md`](../tests/gpu-efa-test.md) is the one that decides those.

## 4. Deploying one child on its own

Faster than the root when the change is in one layer. Each child takes the previous one's outputs:

```bash
aws cloudformation describe-stacks --stack-name eks-gpu-test \
  --query 'Stacks[0].Outputs' --output table
```

Deploy `eks-add-gpu-nodegroup.yaml` with `--template-body file://assets/eks-add-gpu-nodegroup.yaml`
and no bucket at all — it has no children. That is the fastest loop for launch-template and
bootstrap changes: delete the GPU stack, deploy it again, keep the cluster.

## 5. Cleaning up a test

```bash
aws cloudformation delete-stack --stack-name eks-gpu-test
aws cloudformation wait stack-delete-complete --stack-name eks-gpu-test
```

If the delete fails, read [`../tests/cleanup-test.md`](../tests/cleanup-test.md): in an account with
GuardDuty Runtime Monitoring, two resources the stack does not own hold the VPC.
