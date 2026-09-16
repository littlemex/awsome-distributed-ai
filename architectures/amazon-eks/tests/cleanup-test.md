# Cleanup Test (row 4)

Deleting the stack and leaving nothing behind are two different things. Four kinds of leftover
are known: resources another service created inside this VPC (the GuardDuty managed endpoint and
its security group), resources a participant created through Kubernetes (LoadBalancers,
PersistentVolumes), and a log group the EKS service owns rather than the stack. Two of them
**block** the delete; two **survive** it and keep costing money.

Each check below is a pair: the command that finds the leftover, and the command that removes
it.

```bash
export REGION=us-east-2
export STACK=eks-gpu-efa                 # the stack you are deleting
export CLUSTER=$(aws cloudformation describe-stacks --stack-name "$STACK" --region "$REGION" \
  --query 'Stacks[0].Outputs[?OutputKey==`ClusterName`].OutputValue' --output text)
export VPC=$(aws cloudformation describe-stacks --stack-name "$STACK" --region "$REGION" \
  --query 'Stacks[0].Outputs[?OutputKey==`VpcId`].OutputValue' --output text)
echo "$CLUSTER in $VPC"
```

**Expected:** both non-empty. Record them now — after the delete, the outputs are gone and you
will have nothing to search with.

---

## Part 1 — the pre-delete check (the two that block the delete)

### 1.1 LoadBalancer services

A `Service` of type `LoadBalancer` creates an NLB or CLB and a security group that
CloudFormation does not own. The load balancer holds ENIs in the stack's subnets, so the subnet
delete fails and the stack ends in `DELETE_FAILED`.

Find:

```bash
kubectl get svc -A --field-selector spec.type=LoadBalancer
aws elbv2 describe-load-balancers --region "$REGION" \
  --query "LoadBalancers[?VpcId=='$VPC'].[LoadBalancerName,Type,State.Code]" --output table
aws elb describe-load-balancers --region "$REGION" \
  --query "LoadBalancerDescriptions[?VPCId=='$VPC'].LoadBalancerName" --output text
```

**Expected before the delete:** no rows from any of the three. The second and third commands
matter because deleting the `Service` object is what deletes the load balancer — if the
namespace was force-deleted first, the AWS resource is orphaned and `kubectl` can no longer see
it.

Remove:

```bash
kubectl delete svc <name> -n <namespace>          # the correct way: the controller deletes the LB
# only if the Service is already gone and the LB is orphaned:
aws elbv2 delete-load-balancer --load-balancer-arn <arn> --region "$REGION"
```

### 1.2 PersistentVolumes

A `PersistentVolume` with `persistentVolumeReclaimPolicy: Retain` — which is what the
FSx examples use — survives its namespace and its claim. An EBS-backed PV leaves a volume
behind; an FSx-backed PV points at a filesystem the stack does own, so the PV itself is the
only leftover.

Find:

```bash
kubectl get pv -o custom-columns=\
'NAME:.metadata.name,RECLAIM:.spec.persistentVolumeReclaimPolicy,STATUS:.status.phase,DRIVER:.spec.csi.driver,HANDLE:.spec.csi.volumeHandle'
aws ec2 describe-volumes --region "$REGION" \
  --filters "Name=tag:kubernetes.io/cluster/$CLUSTER,Values=owned" \
  --query 'Volumes[].[VolumeId,State,Size,Tags[?Key==`kubernetes.io/created-for/pvc/name`].Value|[0]]' \
  --output table
```

**Expected before the delete:** no PV rows, and no volumes. Every `Retain` PV listed here is a
leftover, and every `available` EBS volume is a charge that continues after the stack is gone.

Remove:

```bash
kubectl delete pvc --all -n <namespace>
kubectl delete pv <name>
aws ec2 delete-volume --volume-id <vol-...> --region "$REGION"   # only for volumes left 'available'
```

> Back up FSx data first if it matters — the filesystem is deleted with the stack, without a
> final backup.

---

## Part 2 — the delete

```bash
aws cloudformation delete-stack --stack-name "$STACK" --region "$REGION"
aws cloudformation wait stack-delete-complete --stack-name "$STACK" --region "$REGION"
aws cloudformation describe-stacks --stack-name "$STACK" --region "$REGION" \
  --query 'Stacks[0].StackStatus' --output text
```

**Expected:** the `wait` returns cleanly and the final `describe-stacks` fails with
`Stack with id ... does not exist`. Roughly 15-25 minutes; the GPU node group and the NAT
gateway are the slow parts.

If it ends in `DELETE_FAILED`, read which resource refused before retrying:

```bash
aws cloudformation describe-stack-events --stack-name "$STACK" --region "$REGION" \
  --query 'StackEvents[?ResourceStatus==`DELETE_FAILED`].[LogicalResourceId,ResourceStatusReason]' \
  --output text
```

**Expected diagnosis:** a message naming a security group, subnet or VPC with a *dependent
object*. That dependency is a Part 1 or Part 3 leftover — find and remove it, then re-run
`delete-stack`. Do **not** delete the stack with `--retain-resources` to make the error go away;
that converts a blocked delete into an untracked bill.

---

## Part 3 — the GuardDuty managed endpoint and its security group

If GuardDuty EKS Runtime Monitoring is enabled in the account, GuardDuty creates a VPC endpoint
and a security group **inside this VPC**, owned by GuardDuty. The stack does not know about
them, so the VPC delete fails while they exist.

Find:

```bash
aws ec2 describe-vpc-endpoints --region "$REGION" \
  --filters "Name=vpc-id,Values=$VPC" \
  --query 'VpcEndpoints[?contains(ServiceName, `guardduty`)].[VpcEndpointId,ServiceName,State]' \
  --output table

aws ec2 describe-security-groups --region "$REGION" \
  --filters "Name=vpc-id,Values=$VPC" \
  --query 'SecurityGroups[?starts_with(GroupName, `GuardDuty`)].[GroupId,GroupName]' --output table

aws guardduty list-detectors --region "$REGION"
aws guardduty get-detector --detector-id <id> --region "$REGION" \
  --query 'features[?Name==`EKS_RUNTIME_MONITORING` || Name==`RUNTIME_MONITORING`].[Name,Status]' --output text
```

**Expected:** after the EKS cluster is deleted, GuardDuty removes both by itself, so both
queries return empty. They are still listed while the cluster exists — that is normal, not a
leak.

Remove, in this order, when they outlive the cluster:

```bash
aws ec2 delete-vpc-endpoint --vpc-endpoint-id <vpce-...> --region "$REGION"
# the endpoint's ENIs take a minute to disappear; the security group cannot go before them
aws ec2 delete-security-group --group-id <sg-...> --region "$REGION"
aws cloudformation delete-stack --stack-name "$STACK" --region "$REGION"   # retry the delete
```

**Expected:** the endpoint moves to `deleting` and then vanishes, the security group delete
returns no output, and the retried stack delete completes. A `DependencyViolation` on the
security group means the endpoint's ENIs are still there — wait and retry rather than forcing
it.

The durable fix, if the account keeps blocking deletes, is to exclude workshop clusters from
Runtime Monitoring rather than to delete the endpoint each time; that is an account-level
decision, not something this stack can do.

---

## Part 4 — the log groups (the ones that survive)

```bash
aws logs describe-log-groups --region "$REGION" \
  --log-group-name-prefix "/aws/eks/$CLUSTER" \
  --query 'logGroups[].[logGroupName,retentionInDays,storedBytes]' --output table

aws logs describe-log-groups --region "$REGION" --log-group-name-prefix "/aws/codebuild/" \
  --query "logGroups[?contains(logGroupName, '$STACK')].[logGroupName,storedBytes]" --output table
```

**Expected:** `/aws/eks/$CLUSTER/cluster` is **still there** after the stack is gone — the EKS
service created it when control-plane logging was enabled, so CloudFormation never owned it and
cannot delete it. This is the documented survivor, not a bug. The bootstrap's CodeBuild log
group **is** declared by the template, so it must be gone; a CodeBuild log group still listed
here is a real leak.

Remove:

```bash
aws logs delete-log-group --log-group-name "/aws/eks/$CLUSTER/cluster" --region "$REGION"
```

A stack recreated with the same name reuses the same log group name, so leaving it behind also
means the next deploy's control-plane logs are appended to the previous run's — one more reason
to delete it.

---

## Part 5 — the sweep that catches everything else

One tag-based query, then the specific ones the tag cannot cover:

```bash
aws resourcegroupstaggingapi get-resources --region "$REGION" \
  --tag-filters "Key=aws:cloudformation:stack-name,Values=$STACK" \
  --query 'ResourceTagMappingList[].ResourceARN' --output text

aws ec2 describe-vpcs --vpc-ids "$VPC" --region "$REGION"
aws ec2 describe-addresses --region "$REGION" \
  --query "Addresses[?AssociationId==null].[PublicIp,AllocationId,Tags[?Key=='aws:cloudformation:stack-name'].Value|[0]]" \
  --output table
aws ec2 describe-placement-groups --region "$REGION" \
  --query "PlacementGroups[?contains(GroupName, '$STACK')].[GroupName,State]" --output text
aws eks list-clusters --region "$REGION" --output text
aws codebuild list-projects --region "$REGION" --query "projects[?contains(@, '$STACK')]" --output text
aws fsx describe-file-systems --region "$REGION" \
  --query "FileSystems[?Tags[?Value=='$STACK']].[FileSystemId,Lifecycle]" --output text
aws iam list-roles --query "Roles[?contains(RoleName, '$STACK')].RoleName" --output text
```

**Expected:** the tag query returns nothing; `describe-vpcs` fails with `InvalidVpcID.NotFound`;
no unassociated Elastic IP carries this stack's tag (an unassociated EIP is billed hourly); no
placement group, no cluster, no CodeBuild project, no filesystem and no IAM role named after the
stack. Anything that does come back is a leak — record which template declared it, because the
fix belongs in that template rather than in this procedure.

---

## Recording the result

A pass is: Part 1 clean before the delete, `DELETE_COMPLETE` in one attempt, Parts 3 and 5
empty afterwards, and Part 4 showing exactly one survivor — the EKS control-plane log group —
which you then delete by hand. Note the run in [`README.md`](./README.md) together with the
hardware rows; a cleanup that needed manual intervention beyond that log group is a finding,
not a pass.
