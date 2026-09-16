# Hosting these templates for a workshop

An event that provisions accounts ahead of the session — participants arrive at a cluster that
already exists — needs the templates served from the event's own bucket rather than from this
repository's, and needs the failure modes to be ones the operator has already seen.

The same pattern is used by `architectures/sagemaker-hyperpod-eks/cfn-templates`, whose nested root
maps each Region to a workshop asset bucket and reads its children from there.

## 1. Place all four templates under one prefix

The root fetches its children by URL, built from `S3BucketName` and `S3KeyPrefix`. Both parameters
have to point at the event's copy, and the four files have to sit under the same prefix:

```
s3://<event-bucket>/<event-prefix>/eks-gpu-cluster-deploy-all.yaml
s3://<event-bucket>/<event-prefix>/eks-cluster-prerequisites.yaml
s3://<event-bucket>/<event-prefix>/eks-cluster.yaml
s3://<event-bucket>/<event-prefix>/eks-add-gpu-nodegroup.yaml
```

Deploy the root with `S3BucketName` and `S3KeyPrefix` overridden to those values. A root deployed
from the event bucket with the parameters left at their defaults will create the **published**
children, which is the failure this paragraph exists to prevent: the stack succeeds and is not the
one you tested.

## 2. Parameters an event sets

| Parameter | Value for an event | Why |
|---|---|---|
| `PrimarySubnetAZ` | the zone the capacity is in | EFA and the placement group are single-zone. With a Capacity Block, this is the Block's zone |
| `SecondarySubnetAZ` | any other zone in the Region | Control plane only |
| `GpuInstanceType`, `GpuNodeCount` | the session's shape, at least 2 nodes for a prefill/decode split | |
| `CapacityReservationId`, `CapacityReservationType` | the event's reservation | A `targeted-odcr` is consumed from inside the stack's cluster placement group; a `capacity-block` omits the group |
| `AdminRoleArn` | the role the participant will use | The stack creator gets cluster-admin automatically, and for a pre-provisioned account that is the provisioning role, not the participant. Without this the participant can reach the cluster's API and be denied by RBAC |
| `PrePullImage` | the serving image the session uses | Saves every participant a multi-gigabyte pull during the session. The stack does not wait for it |
| `S3BucketName`, `S3KeyPrefix` | the event's copy | Section 1 |

`AdminRoleArn` is the one that is easy to miss and expensive to discover: the symptom is a
participant with valid credentials getting `error: You must be logged in to the server
(Unauthorized)` from `kubectl`.

## 3. Timing

| Step | Duration |
|---|---|
| Prerequisites stack (VPC, NAT, endpoints; plus 10 to 15 minutes when `DeployFsxLustre=true`) | 3 to 5 minutes |
| Cluster and system node group | 12 to 15 minutes |
| GPU node group | 4 to 8 minutes, longer while a reservation is being consumed |
| Device plugin install and per-node verification | 2 to 5 minutes |
| Image pre-pull, if set | not waited for; watch it separately |

Around 25 minutes for a `g7e.12xlarge` pair, and deletion takes 15 to 20. Neither number fits in the
last ten minutes of a session, which is the argument for provisioning ahead and for reclaiming
accounts centrally.

## 4. Failure modes to rehearse before the event

Each of these has bitten a first deploy somewhere. Rehearse them in one account, and the runbook for
the day writes itself.

- **Quotas.** On-Demand vCPUs for the GPU family, and Elastic IPs (one per stack). Both are checked
  in the README's quick start. A fresh account also has a low CodeBuild concurrency limit, and the
  bootstrap is a CodeBuild build.
- **Capacity.** Two distinct failures. With `GpuNodeCount` setting minimum, desired and maximum
  together, a reservation with fewer instances available than requested fails the stack rather than
  delivering fewer nodes. And without a reservation, a zone that has no capacity for the instance type
  does not fail at all: the Auto Scaling group retries every two minutes, the node group sits in
  `CREATING` with no issues reported, and the stack waits. The zone is fixed at create time, so the
  recovery is a new stack in another zone. For a scheduled event this is the argument for a capacity
  reservation rather than for a runbook — see
  [OPERATIONS.md](./OPERATIONS.md#2-when-the-gpu-node-group-stays-in-creating-with-no-instances) for
  the command that shows the retry loop.
- **GuardDuty Runtime Monitoring**, if the event's accounts have it enabled: a managed VPC endpoint
  and security group appear after the VPC and block its deletion. See
  [`../tests/cleanup-test.md`](../tests/cleanup-test.md). This matters most for an event, because
  account reclamation is automated and unattended.
- **A bootstrap failure rolls the stack back**, deleting the cluster. For provisioning runs, deploy
  with `--disable-rollback` so a failed cluster can be inspected instead of vanishing; see
  [OPERATIONS.md](./OPERATIONS.md).

## 5. What the participant is left with

Worth stating in the session's own material, because it is the boundary between this stack and the
lab content: a cluster with two GPU nodes, each advertising its GPUs and its EFA interfaces, both
device plugins installed, local NVMe at `/mnt/k8s-disks/0`, and `kubectl` access. Everything from the
serving framework upward — the images, the model, the prefill and decode manifests, the router — comes
from the lab and from [`examples/inference`](../../../examples/inference).
