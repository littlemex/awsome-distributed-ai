# Disaggregated Inference Smoke Test (row 3)

One request served end to end with **prefill and decode on different GPU nodes**, using an
existing `examples/inference/` sample without modifying it. This is the test that shows the
architecture satisfies a real serving example's prerequisites, rather than only its own.

> **Scope.** This proves the request path and a cross-node KV transfer. It does **not** prove
> the KV cache moved over EFA: the example's worker pods do not request
> `vpc.amazonaws.com/efa`, so NIXL uses TCP over the pod network, exactly as the example's own
> README states. The EFA claim belongs to [`gpu-efa-test.md`](./gpu-efa-test.md) step 5, and
> throughput belongs to the example, not to this architecture.

---

## Which sample, and why

**Chosen: [`examples/inference/nvidia-dynamo`](../../../examples/inference/nvidia-dynamo),
scenario `gpt-oss-disagg`.**

| Criterion | How the scenario meets it on 2 x `g7e.12xlarge` (2 GPUs x 96 GB per node) |
|---|---|
| Fits the hardware | Prefill and decode are one GPU each. GPT-OSS-20B is MXFP4, about 13 GB of weights; the scenario is validated on 48 GB L40S, so 96 GB is comfortable |
| Genuinely disaggregated across nodes | Prefill and decode are separate pods with separate GPU requests, so they can be placed on different nodes. KV moves prefill to decode over NIXL |
| Public, small model | GPT-OSS-20B is public — the example says no Hugging Face token is needed, so nothing account-specific enters the test |
| Runs on this architecture as written | The scenario manifests already tolerate `nvidia.com/gpu:NoSchedule` and already select nodes by the portable `node.kubernetes.io/instance-type` label rather than a HyperPod label |
| One-request proof is cheap | The frontend is OpenAI-compatible, so a single `curl` decides the result |

Rejected, with the reason:

| Sample | Why not |
|---|---|
| [`examples/inference/vllm/dsv3-uccl-nixl`](../../../examples/inference/vllm/dsv3-uccl-nixl) | DeepSeek-V3-0324, 671B. Its smallest disaggregated config is 2 x `p5en.48xlarge` (16 x H200, 1.2 TB of HBM). Four 96 GB GPUs cannot hold the weights at any quantisation this example ships |
| [`examples/inference/sglang/dsr1-deepep-efa`](../../../examples/inference/sglang/dsr1-deepep-efa) | DeepSeek-R1 671B FP8 on 2 x `p5.48xlarge`; the same memory wall, plus DeepEP over NVSHMEM which that README states is validated on H100 and H200 only |
| [`examples/inference/sglang/kimi2.6-h200-1p1d`](../../../examples/inference/sglang/kimi2.6-h200-1p1d) | The right topology — node-level 1P1D with NIXL — but sized for 2 x 8 x H200, and its own results table is still marked *not yet measured*, so it is not a settled baseline to smoke-test against |
| [`examples/inference/sglang/dsv4flash-b300-intra-3p1d`](../../../examples/inference/sglang/dsv4flash-b300-intra-3p1d), [`examples/inference/sglang/qwen3.5-27b-b300-intra-pd`](../../../examples/inference/sglang/qwen3.5-27b-b300-intra-pd) | Prefill/decode disaggregation **inside** one 8-GPU B300 node, in a single pod, deliberately using CUDA IPC over NVLink. Wrong topology for a two-node test and needs eight GPUs on one node |
| [`examples/inference/sglang/dsv4pro-b300-single-node`](../../../examples/inference/sglang/dsv4pro-b300-single-node), [`examples/inference/sglang/glm5.2-b300-tp2-dp4`](../../../examples/inference/sglang/glm5.2-b300-tp2-dp4), [`examples/inference/vllm/cosmos-reason`](../../../examples/inference/vllm/cosmos-reason) | Not disaggregated — one engine, or replicas behind a router |

The other Dynamo scenario, `qwen3.6-disagg`, would also fit the memory budget, but its decode
worker needs `--max-running-requests` and `--mem-fraction-static` tuned for the hybrid SSM
state; `gpt-oss-disagg` has fewer moving parts for a smoke test.

---

## Step 0 — a cluster with shared storage

The example loads weights from a shared filesystem, so deploy with the optional FSx module on:

```bash
export REGION=us-east-2
export STACK=eks-gpu-disagg
export AZ_A=us-east-2b
export AZ_B=us-east-2a

cd architectures/amazon-eks
aws cloudformation create-stack \
  --stack-name "$STACK" --region "$REGION" \
  --template-body file://assets/eks-gpu-cluster-deploy-all.yaml \
  --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM \
  --parameters ParameterKey=PrimarySubnetAZ,ParameterValue=$AZ_A \
               ParameterKey=SecondarySubnetAZ,ParameterValue=$AZ_B \
               ParameterKey=GpuInstanceType,ParameterValue=g7e.12xlarge \
               ParameterKey=GpuNodeCount,ParameterValue=2 \
               ParameterKey=DeployFsxLustre,ParameterValue=true
aws cloudformation wait stack-create-complete --stack-name "$STACK" --region "$REGION"
eval "$(aws cloudformation describe-stacks --stack-name "$STACK" --region "$REGION" \
  --query 'Stacks[0].Outputs[?OutputKey==`KubeconfigCommand`].OutputValue' --output text)"
```

If the cluster from [`gpu-efa-test.md`](./gpu-efa-test.md) is still up but was deployed with
`DeployFsxLustre=false`, either update the stack or use the local-NVMe variant in the note at
the end of this file.

Create the `PersistentVolume` for the filesystem from the stack's outputs — the stack provides
the filesystem and the CSI driver, and the manifest lives in
[`../docs/OPERATIONS.md`](../docs/OPERATIONS.md):

```bash
out() { aws cloudformation describe-stacks --stack-name "$STACK" --region "$REGION" \
  --query "Stacks[0].Outputs[?OutputKey=='$1'].OutputValue" --output text; }
FS_ID=$(out FsxFileSystemId); FS_DNS=$(out FsxDnsName); FS_MOUNT=$(out FsxMountName)
echo "$FS_ID $FS_DNS $FS_MOUNT"

cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: PersistentVolume
metadata:
  name: model-cache
spec:
  capacity: { storage: 1200Gi }
  accessModes: ["ReadWriteMany"]
  persistentVolumeReclaimPolicy: Retain
  csi:
    driver: fsx.csi.aws.com
    volumeHandle: "$FS_ID"
    volumeAttributes: { dnsname: "$FS_DNS", mountname: "$FS_MOUNT" }
EOF

kubectl get sc
kubectl get pv model-cache
```

**Expected:** the three outputs are non-empty (they are empty strings when `DeployFsxLustre` is
`false`, which is the failure mode to catch here), `kubectl get sc` lists `gp2` — the Dynamo
platform's etcd and NATS ask for it — and the PV is `Available`. The example's
`02-download-model.sh` discovers this PV by its CSI driver name and creates its own PV and PVC
from it, which is why the name of this one does not matter.

---

## Step 1 — install the Dynamo platform

```bash
cd ../../examples/inference/nvidia-dynamo
cp env_vars.example env_vars
# edit env_vars: AWS_REGION, AWS_ACCOUNT_ID, EKS_CLUSTER_NAME (from the stack's ClusterName output)
source env_vars

./scripts/01-install-platform.sh
kubectl -n dynamo-system get pods -o wide
```

**Expected:** the CRDs install, the EBS CSI add-on becomes active, and the operator, etcd and
NATS pods appear. They will be **`Pending`** on this architecture, and that is expected: the
platform values target the SageMaker HyperPod label
`sagemaker.amazonaws.com/instance-group-name`, which no node here carries — the example's own
values file says so in its portability note. Place them on this architecture's system nodes and
re-run only the platform install:

```bash
cat > /tmp/dynamo-eks-nodes.yaml <<'EOF'
dynamo-operator:
  controllerManager:
    nodeSelector: { role: system }
etcd:
  nodeSelector: { role: system }
nats:
  nodeSelector: { role: system }
EOF

helm upgrade --install dynamo-platform dynamo-platform-0.7.0.tgz \
  --namespace dynamo-system \
  -f manifests/platform/values.yaml \
  -f /tmp/dynamo-eks-nodes.yaml

kubectl -n dynamo-system get pods -o wide
kubectl -n dynamo-system get pvc
```

**Expected:** operator, `dynamo-platform-etcd-0` and `dynamo-platform-nats-0` all `Running` on
nodes labelled `role=system`, and their PVCs `Bound`. A PVC stuck in `Pending` means the `gp2`
StorageClass has no working provisioner — check that the `aws-ebs-csi-driver` add-on is
`ACTIVE`.

---

## Step 2 — stage the weights on the shared filesystem

```bash
./scripts/02-download-model.sh gpt-oss
kubectl -n dynamo-system get pvc dynamo-fsx
kubectl -n dynamo-system logs job/download-gpt-oss | tail -5
```

**Expected:** the script finds the FSx PV, creates `dynamo-fsx-pv` and the `dynamo-fsx` PVC
(`Bound`), and the job ends with a `du -sh` line of roughly 13 GB under
`/fsx/models/openai-gpt-oss-20b`. The download job selects GPU nodes by instance type; if it
stays `Pending`, apply the same instance-type substitution as step 3 to
`manifests/jobs/download-model.yaml`.

---

## Step 3 — deploy the scenario with prefill and decode pinned to different nodes

The scenario as committed selects `g6e.4xlarge` and does not force the two workers apart, so
both could land on one node and the test would prove nothing. Render a copy that selects one
specific node per role, without editing the example:

```bash
export NODE_A=$(kubectl get nodes -l role=gpu -o jsonpath='{.items[0].metadata.name}')
export NODE_B=$(kubectl get nodes -l role=gpu -o jsonpath='{.items[1].metadata.name}')
echo "$NODE_A / $NODE_B"

# run from examples/inference/nvidia-dynamo; needs PyYAML
python3 - <<'PY' > /tmp/gpt-oss-disagg-2node.yaml
import os, yaml
doc = yaml.safe_load(open("manifests/scenarios/gpt-oss-disagg/gpt-oss-20b-disagg.yaml"))
where = {"Frontend": os.environ["NODE_A"], "prefill": os.environ["NODE_A"], "decode": os.environ["NODE_B"]}
for name, service in doc["spec"]["services"].items():
    terms = [{"matchExpressions": [{"key": "kubernetes.io/hostname",
                                    "operator": "In", "values": [where[name]]}]}]
    spec = service["extraPodSpec"]
    spec["affinity"]["nodeAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"]["nodeSelectorTerms"] = terms
print(yaml.safe_dump(doc, sort_keys=False))
PY

grep -A4 'kubernetes.io/hostname' /tmp/gpt-oss-disagg-2node.yaml | head -20
kubectl apply -f /tmp/gpt-oss-disagg-2node.yaml
```

**Expected:** the rendered file pins the frontend and prefill to `$NODE_A` and decode to
`$NODE_B`, using the same `affinity` field the example already relies on (so no new operator
behaviour is assumed). The `nvidia.com/gpu:NoSchedule` toleration and the one-GPU requests are
carried over untouched.

```bash
kubectl -n dynamo-system get dynamographdeployment
kubectl -n dynamo-system get pods -o wide -l nvidia.com/dynamo-graph-deployment-name=gpt-oss-disagg
```

**Expected (allow 3-5 minutes for the engines to load from FSx):** three pods `Running` — a
frontend, a prefill worker and a decode worker — and the `NODE` column shows **prefill and
decode on different nodes**. Same node for both is a failed test even if the request later
succeeds, because nothing crossed the network.

If a worker crash-loops on the attention backend, that is the one thing this scenario tunes for
its original GPU (`--attention-backend triton`, chosen for Ada L40S). Read the engine log
before changing anything:

```bash
kubectl -n dynamo-system logs -l nvidia.com/dynamo-graph-deployment-name=gpt-oss-disagg --tail=50 --all-containers
```

---

## Step 4 — one request through the disaggregated path

```bash
FRONTEND=$(kubectl -n dynamo-system get pod \
  -l nvidia.com/dynamo-graph-deployment-name=gpt-oss-disagg,nvidia.com/dynamo-component=Frontend \
  -o jsonpath='{.items[0].metadata.name}')

kubectl -n dynamo-system exec "$FRONTEND" -- curl -s http://localhost:8000/v1/models | python3 -m json.tool

kubectl -n dynamo-system exec "$FRONTEND" -- curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"openai/gpt-oss-20b","messages":[{"role":"user","content":"In one sentence, what is Amazon S3?"}],"max_tokens":128}' \
  | python3 -m json.tool
```

**Expected:**

- `/v1/models` lists `openai/gpt-oss-20b`. An empty list means the frontend never registered
  the workers through etcd and NATS.
- The completion contains a non-empty `choices[0].message.content`, a `finish_reason`, and
  **`usage.completion_tokens` greater than zero**. Zero completion tokens with HTTP 200 is the
  documented signature of a KV transfer that never arrived — the decode worker had nothing to
  continue from. That is why this test reads the token count instead of the status code.

Then confirm the transfer really crossed the two pods:

```bash
PREFILL_IP=$(kubectl -n dynamo-system get pod -l nvidia.com/dynamo-graph-deployment-name=gpt-oss-disagg \
  -o jsonpath='{range .items[*]}{.metadata.name}{" "}{.status.podIP}{"\n"}{end}' | grep prefill)
echo "$PREFILL_IP"

kubectl -n dynamo-system logs -l nvidia.com/dynamo-graph-deployment-name=gpt-oss-disagg \
  --all-containers --tail=200 | grep -Ei 'nixl|bootstrap|disagg|kv' | tail -20
```

**Expected:** the decode worker's log shows a NIXL bootstrap and a KV transfer against the
**prefill pod's IP** (the pod IP printed above), not `127.0.0.1`, and no
`Connection refused :12345`. Both together — a non-zero token count and a NIXL exchange between
two pod IPs on two nodes — are the pass condition.

---

## Step 5 — clean up

```bash
./scripts/09-cleanup-inference.sh gpt-oss-disagg
./scripts/10-uninstall-platform.sh
kubectl delete pv model-cache dynamo-fsx-pv --ignore-not-found
```

**Expected:** the scenario's pods disappear and the GPU allocatable counts return to `2` per
node. Both FSx PVs use `Retain`, so they survive a namespace delete and must be removed
explicitly — [`cleanup-test.md`](./cleanup-test.md) checks for exactly this kind of leftover
before the stack delete.

---

## Note: without the FSx module

`g7e.12xlarge` carries about 3.5 TiB of local NVMe at `/mnt/k8s-disks/0`, which the dsv3
example uses instead of a shared filesystem. To run this test with `DeployFsxLustre=false`,
replace the `fsx` volume in the rendered copy with
`hostPath: { path: /mnt/k8s-disks/0/models }` in all three services and download the weights
**once per node** (a DaemonSet, or the download job run twice with the two hostnames). The
trade-off is explicit: no shared filesystem means the weights are staged twice and the decode
worker cannot be rescheduled onto a node that has not been staged.
