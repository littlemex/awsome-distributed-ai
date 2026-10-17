"""Render or apply the lab's namespace-scoped EKS resources."""
import argparse
import copy
import json
import os
from pathlib import Path
import re
import subprocess

OWNER = {"app.kubernetes.io/part-of": "aim345-lab"}


def read_config(path):
    c = json.loads(Path(path).read_text())
    assert all("REPLACE" not in str(v) for v in c.values()), "Fill config.example.json first"
    assert re.fullmatch(r"[a-z][a-z0-9-]{0,61}[a-z0-9]", c["namespace"])
    assert c["namespace"].startswith("aim345-"), "Use a dedicated aim345-* namespace"
    assert re.fullmatch(r"[a-f0-9]{40}", c["model_revision"])
    for key in ("image", "router_image"):
        image = c[key]
        assert "@sha256:" in image or (":" in image.rsplit("/", 1)[-1] and not image.endswith(":latest"))
    assert c["gpus_per_worker"] > 0 and c["efa_per_worker"] > 0
    minimum_nodes = 1 if c.get("placement") == "packed" else 2
    assert c.get("placement", "separate-nodes") in ("packed", "separate-nodes")
    assert len(c["nodes"]) == len(set(c["nodes"])) >= minimum_nodes
    assert c["model_cache_host_path"].startswith("/mnt/"), "Use a dedicated cache under /mnt"
    if "page_size_tokens" in c:
        page = c["page_size_tokens"]
        assert type(page) is int and page >= 1, "page_size_tokens must be a positive integer"
        assert c["chunked_prefill_size_tokens"] % page == 0, "chunked_prefill_size_tokens must be a multiple of page_size_tokens"
    if c.get("extra_env") is not None:
        assert isinstance(c["extra_env"], dict) and all(isinstance(k, str) and k for k in c["extra_env"]), \
            "extra_env must be a flat string-keyed dict of engine environment variables"
    if c.get("pin_nixl_version") is not None:
        assert isinstance(c["pin_nixl_version"], str) and re.fullmatch(r"\d+\.\d+\.\d+", c["pin_nixl_version"]), \
            "pin_nixl_version must be a bare X.Y.Z version string (e.g. \"1.4.0\")"
    # Per-role worker count: how many TP groups a role runs, each its own engine process.
    # Defaulting every one of these to the single-worker-per-role shape (prefill_workers=1,
    # decode_workers=1, unified_replicas=2) keeps every config that does not set them on the
    # exact code path render() always ran before this generalization (DESIGN-RATIONALE.md,
    # "prefill:decode GPU ratio").
    for key in ("prefill_workers", "decode_workers", "unified_replicas"):
        if key in c:
            assert isinstance(c[key], int) and c[key] >= 1, f"{key} must be a positive integer"
    # Per-role TP override: lets prefill, decode or unified use a different gpus_per_worker
    # than the config's base value. Unset for every role still used today -- the base
    # gpus_per_worker alone decides TP, exactly as before.
    for key in ("prefill_gpus_per_worker", "decode_gpus_per_worker", "unified_gpus_per_worker"):
        if key in c:
            assert isinstance(c[key], int) and c[key] > 0, f"{key} must be a positive integer"
    if c['model_id'] == 'deepseek-ai/DeepSeek-V2-Lite-Chat':
        # Eight ranks abort CUDA-graph capture in the pinned engine ("hidden size must be divisible by vector size").
        # Checked for every role's effective TP, not just the base value, so a per-role override
        # cannot smuggle in an unsupported tensor-parallel size.
        for tp in {c['gpus_per_worker'], c.get('prefill_gpus_per_worker'), c.get('decode_gpus_per_worker'), c.get('unified_gpus_per_worker')} - {None}:
            assert tp in (1, 2, 4), 'DeepSeek-V2-Lite-Chat serves at 1, 2 or 4 GPUs per worker on the pinned engine'
    if c.get('engine_profile') == 'v4-optional' or c['model_id'] == 'deepseek-ai/DeepSeek-V4-Flash':
        assert c.get('engine_profile') == 'v4-optional', 'V4 requires the separately pinned optional engine'
        assert c['model_id'] == 'deepseek-ai/DeepSeek-V4-Flash', 'The optional profile is qualified only for V4'
        assert c['instance_type'] == 'g7.48xlarge' and c['gpus_per_worker'] == 8 and c['efa_per_worker'] == 2
        assert c.get('placement', 'separate-nodes') == 'separate-nodes' and len(c['nodes']) == 2
    return c


def total_gpus(c, mode):
    """Total GPU count every worker of this config's `mode` requests, generalizing the old
    `2 * gpus_per_worker` assumption (deployment.py's render() hardcoded exactly one engine per
    role) to any per-role worker count and per-role TP override. Used by paired.py's check_pair
    to compare two stacks' GPU budgets without assuming they are built the same way."""
    if mode == "unified":
        return c.get("unified_replicas", 2) * c.get("unified_gpus_per_worker", c["gpus_per_worker"])
    return (c.get("prefill_workers", 1) * c.get("prefill_gpus_per_worker", c["gpus_per_worker"])
            + c.get("decode_workers", 1) * c.get("decode_gpus_per_worker", c["gpus_per_worker"]))


def effective_tps(c, mode):
    """The set of tensor-parallel sizes this config's `mode` actually runs. A stack with a
    uniform TP across its roles (every config measured so far) is a single-element set; paired.py
    rejects a pair whose stacks don't reduce to the same single element, which is a strictly more
    accurate check than comparing the base `gpus_per_worker` field alone once a per-role override
    can change what a worker actually runs at."""
    if mode == "unified":
        return {c.get("unified_gpus_per_worker", c["gpus_per_worker"])}
    return {c.get("prefill_gpus_per_worker", c["gpus_per_worker"]), c.get("decode_gpus_per_worker", c["gpus_per_worker"])}


_cached_token = None


def token_args():
    """Fetches this kubeconfig's exec-credential plugin (e.g. `aws eks get-token`) once per
    process and returns ["--token", TOKEN] to override it on every subsequent kubectl invocation,
    instead of []. Not a performance optimization: on at least one real cluster, letting every
    `kubectl` subprocess independently re-run the token plugin correlated with `kubectl exec`
    connections being reset mid-transfer on anything beyond a trivial response once a few dozen
    calls had happened in a short window (undocumented upstream; found running labctl.py against
    the Spain EKS cluster, not reproduced in isolation). A single cached token removes the
    repeated plugin invocations and the suspected race entirely. Degrades to [] (today's
    behavior) if the kubeconfig cannot be read this way, e.g. a YAML kubeconfig with no exec
    credential, or any other shape this best-effort parse does not expect."""
    global _cached_token
    if _cached_token is not None:
        return ["--token", _cached_token]
    try:
        path = os.environ.get("KUBECONFIG", str(Path.home() / ".kube" / "config"))
        config = json.loads(Path(path).read_text())
        ctx = next(x["context"] for x in config["contexts"] if x["name"] == config["current-context"])
        user = next(x["user"] for x in config["users"] if x["name"] == ctx["user"])
        execcfg = user["exec"]
        r = subprocess.run([execcfg["command"], *execcfg["args"]], text=True, capture_output=True, check=True)
        _cached_token = json.loads(r.stdout)["status"]["token"]
        return ["--token", _cached_token]
    except Exception:
        return []


def kubectl(c, *args, **kwargs):
    return subprocess.run(["kubectl", "--context", c["context"], *token_args(), "-n", c["namespace"], *args], check=True, text=True, **kwargs)


def engine_ports(pod_spec):
    """The ports a pod's engine processes listen on, read from the spec it was started with."""
    container = next(x for x in pod_spec['containers'] if x['name'] == 'engine')
    env = {x['name']: x.get('value') for x in container['env']}
    argsets = [plan['args'] for plan in json.loads(env['AIM345_ENGINE_PLANS'])] if 'AIM345_ENGINE_PLANS' in env else [container['args']]
    return [int(args[args.index('--port') + 1]) for args in argsets]


def engine_endpoint(c, pod, port, endpoint):
    """One engine HTTP endpoint read from inside its pod, so no port forward or Service is involved."""
    code = f"import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:{port}/{endpoint}', timeout=30).read().decode())"
    return kubectl(c, 'exec', pod, '-c', 'engine', '--', 'python3', '-c', code, capture_output=True).stdout


def owned_namespace(c, create=False):
    r = subprocess.run(["kubectl", "--context", c["context"], *token_args(), "get", "namespace", c["namespace"], "--ignore-not-found", "-o", "json"], text=True, capture_output=True, check=True)
    priority = subprocess.run(["kubectl", "--context", c["context"], *token_args(), "get", "priorityclass", c["namespace"] + "-nonpreempting", "--ignore-not-found", "-o", "json"], text=True, capture_output=True, check=True)
    if priority.stdout.strip():
        assert json.loads(priority.stdout)["metadata"].get("labels", {}).get("app.kubernetes.io/part-of") == "aim345-lab", "PriorityClass is not owned by this lab"
    if r.stdout.strip():
        assert json.loads(r.stdout)["metadata"].get("labels", {}).get("app.kubernetes.io/part-of") == "aim345-lab", "Namespace is not owned by this lab"
    elif create:
        kubectl(c, "create", "-f", "-", input=json.dumps({"apiVersion": "v1", "kind": "Namespace", "metadata": {"name": c["namespace"], "labels": OWNER}}))
    else:
        raise RuntimeError("Lab namespace does not exist")


def metadata(c, name, labels=None):
    return {"name": name, "namespace": c["namespace"], "labels": OWNER | (labels or {})}


def service(c, name, port, headless=False):
    spec = {"selector": OWNER | {"app": name}, "ports": [{"name": "http", "port": port, "targetPort": port}]}
    if headless:
        spec.update(clusterIP="None", publishNotReadyAddresses=True)
    return {"apiVersion": "v1", "kind": "Service", "metadata": metadata(c, name), "spec": spec}


def nixl_pin_shell_prefix(c):
    """Opt-in, empty-string by default: a shell command fragment that live-upgrades the
    container's installed `nixl`/`nixl-cu13` wheels to `pin_nixl_version` via `pip install`,
    with no image rebuild. Placed between `verify_image.py`'s own version-pin check and the
    `exec` that starts the real server, so `verify_image.py` always asserts against the image's
    own baked-in pinned version first (today 1.1.0) -- this upgrade happens only after that
    assertion has already passed, exactly the sequence
    `~/tmp/riv/issue/kvx/RESULTS.md` section 7 and `~/tmp/riv/issue/kvx/NIXL14-SWEEP.md` describe
    testing live. `--no-deps --force-reinstall` matches the Dockerfile's own `nixl`/`nixl-cu13`
    install line so the upgrade replaces exactly those two packages and nothing else. Never
    changes anything when `pin_nixl_version` is absent (empty string, so callers can simply
    concatenate it in front of their own command without a conditional)."""
    v = c.get("pin_nixl_version")
    if not v:
        return ""
    return f"python3 -m pip install --no-deps --force-reinstall nixl=={v} nixl-cu13=={v} && "


def worker(c, name, node, role, bootstrap_port=8998):
    """`bootstrap_port` only matters for `role == "prefill"`, and only needs to differ from the
    default when more than one prefill worker shares one pod (`packed_role_pod` below): every
    prefill engine binds this port to answer a decode engine's KV pull, and packing several
    prefill processes into one pod/network-namespace means they cannot all bind the same one --
    the second bind silently loses, and the router kept telling decode to use the single literal
    port 8998 for every prefill URL regardless (DESIGN-RATIONALE.md, "prefill:decode GPU ratio",
    "a bootstrap port shared by every packed prefill worker is not a bootstrap port"). A single,
    unpacked prefill worker keeps the exact prior literal value."""
    labels = OWNER | {"app": name, "role": role, "component": "engine"}
    model_path = "/models/" + c["model_revision"]
    args = ["--model-path", model_path, "--served-model-name", "aim345", "--host", "0.0.0.0", "--port", "30000", "--tp-size", str(c["gpus_per_worker"]), "--dtype", "bfloat16", "--kv-cache-dtype", "bf16", "--context-length", str(c["context_length_tokens"]), "--chunked-prefill-size", str(c["chunked_prefill_size_tokens"]), "--mem-fraction-static", "0.75", "--enable-metrics", "--stream-interval", "1"]
    if c.get('engine_profile') == 'v4-optional':
        # Qualified V4 profile has quantized weights and needs more static memory.
        for flag in ('--dtype', '--kv-cache-dtype'):
            index = args.index(flag)
            del args[index:index + 2]
        args[args.index('--mem-fraction-static') + 1] = '0.90'
        args += ['--max-running-requests', '4', '--cuda-graph-max-bs-decode', '4', '--disable-custom-all-reduce', '--disable-flashinfer-autotune']
    if c.get("disable_prefix_cache"):
        args.append("--disable-radix-cache")
    if c.get("enable_mixed_chunk"):
        args.append("--enable-mixed-chunk")
    if c.get("schedule_policy"):
        args += ["--schedule-policy", str(c["schedule_policy"])]
    if c.get("page_size_tokens") is not None:
        args += ["--page-size", str(c["page_size_tokens"])]
    if role != "unified":
        args += ["--disaggregation-mode", role, "--disaggregation-transfer-backend", "nixl"]
        if role == "prefill":
            args += ["--disaggregation-bootstrap-port", str(bootstrap_port)]
    resources = {"nvidia.com/gpu": c["gpus_per_worker"], "vpc.amazonaws.com/efa": c["efa_per_worker"], "cpu": str(c.get("engine_cpu", "16")), "memory": c.get("engine_memory", "128Gi"), "ephemeral-storage": c["engine_ephemeral_storage"]}
    # Default kept exactly as before (LIBFABRIC/EFA): opt-in override investigates whether a
    # transport that can use CUDA IPC/shared memory for a same-node transfer (e.g. UCX) does
    # better than routing a packed stack's intra-node KV transfer over the EFA NIC.
    env = [{"name": "SGLANG_DISAGGREGATION_NIXL_BACKEND", "value": str(c.get("nixl_backend", "LIBFABRIC"))}, {"name": "FI_PROVIDER", "value": "efa"}, {"name": "FI_EFA_USE_DEVICE_RDMA", "value": "1"}, {"name": "FI_LOG_LEVEL", "value": "info"}]
    if c.get("nixl_queue_size") is not None:
        env.append({"name": "SGLANG_DISAGGREGATION_QUEUE_SIZE", "value": str(c["nixl_queue_size"])})
    # Opt-in, empty by default: a flat string-keyed escape hatch for engine environment variables
    # this lab has no dedicated config key for yet (e.g. FI_EFA_ENABLE_SHM_TRANSFER, to make
    # libfabric skip its own shm provider on a same-node transfer and use the EFA device path
    # instead). Appended last so it can override any of the fixed/derived entries above by name
    # if a future investigation needs that; never changes anything when "extra_env" is absent.
    for extra_name, extra_value in (c.get("extra_env") or {}).items():
        env = [e for e in env if e["name"] != extra_name] + [{"name": extra_name, "value": str(extra_value)}]
    mounts = [{"name": "models", "mountPath": "/models"}, {"name": "shm", "mountPath": "/dev/shm"}]
    download = "from huggingface_hub import snapshot_download; import sys; snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3])"
    spec = {
        "hostNetwork": True, "dnsPolicy": "ClusterFirstWithHostNet", "preemptionPolicy": "Never", "priorityClassName": c["namespace"] + "-nonpreempting", "automountServiceAccountToken": False,
        "nodeSelector": {"kubernetes.io/hostname": node, "node.kubernetes.io/instance-type": c["instance_type"]},
        "tolerations": c["tolerations"],
        "initContainers": [{"name": "model", "image": c["image"], "command": ["python3", "-c", download, c["model_id"], c["model_revision"], model_path], "volumeMounts": [mounts[0]], "resources": {"requests": {"cpu": "2", "memory": "4Gi"}, "limits": {"cpu": "2", "memory": "4Gi"}}}],
        "containers": [{"name": "engine", "image": c["image"], "imagePullPolicy": "IfNotPresent", "command": ["bash", "-c", 'ulimit -l unlimited; python3 /lab/verify_image.py && ' + nixl_pin_shell_prefix(c) + 'exec python3 -m sglang.launch_server "$@"', "engine"], "args": args, "env": env, "resources": {"requests": resources, "limits": resources}, "securityContext": {"capabilities": {"add": ["IPC_LOCK"]}}, "volumeMounts": mounts, "startupProbe": {"httpGet": {"path": "/health", "port": 30000}, "periodSeconds": 10, "timeoutSeconds": 10, "failureThreshold": 180}, "readinessProbe": {"httpGet": {"path": "/health", "port": 30000}, "periodSeconds": 10, "timeoutSeconds": 10}}],
        "volumes": [{"name": "models", "hostPath": {"path": c["model_cache_host_path"], "type": "DirectoryOrCreate"}}, {"name": "shm", "emptyDir": {"medium": "Memory", "sizeLimit": "32Gi"}}],
    }
    if c.get('engine_cache_host_path'):
        assert c['engine_cache_host_path'].startswith(c['model_cache_host_path'] + '/')
        spec['volumes'].append({'name': 'engine-cache', 'hostPath': {'path': c['engine_cache_host_path'], 'type': 'DirectoryOrCreate'}})
        mounts.append({'name': 'engine-cache', 'mountPath': '/root/.cache'})
    return {"apiVersion": "apps/v1", "kind": "Deployment", "metadata": metadata(c, name, labels), "spec": {"replicas": 1, "strategy": {"type": "Recreate"}, "selector": {"matchLabels": {"app": name}}, "template": {"metadata": {"labels": labels}, "spec": spec}}}


def packed_role_pod(c, node, members, role, pod_name):
    """Pack several same-role engine workers (e.g. three prefill TP2 workers for a 6:2
    prefill:decode ratio) into one exclusive pod allocation on one node, each its own process on
    its own port (30000 + index) and its own disjoint `CUDA_VISIBLE_DEVICES` slice. Generalizes
    paired.py's `paired_render` packed branch -- which pairs exactly the two roles of one stack
    onto one node -- to any worker count within a single role; see DESIGN-RATIONALE.md,
    "prefill:decode GPU ratio". A role with exactly one worker never reaches this function: render()
    keeps emitting that role's original, unpacked single-process Deployment unchanged.

    `members` is a list of `(name, gpus_per_worker)` pairs, already assigned to `node`. The GPUs
    requested are `sum(tp for _, tp in members)`; the EFA device count stays `efa_per_worker` --
    not multiplied by the worker count -- matching paired.py's existing packed convention, because
    a node offers at most 2 EFA devices total and this lab's own namespace is not the only tenant
    of it. `pod_name` is supplied by the caller rather than derived from `role` alone: a role (for
    example `unified`) can be packed onto more than one node when its replica count exceeds the
    node count, and each of those pods needs a distinct name within the namespace.

    Keeps `hostNetwork: true` (inherited from `worker()`'s own template, not overridden), unlike
    paired.py's `paired_render` packed branch, which turns it off. That branch pairs the *two
    roles of one stack* onto one node -- prefill and decode never need to leave that node, so
    there is nothing for `hostNetwork` to matter to. This function instead packs several workers
    of the *same* role whose counterpart role is on a *different* node (the entire reason this
    ratio feature exists): real-cluster testing of a 2-prefill/2-decode, both-sides-packed stack
    found every call timing out, with the prefill side stuck at a constant `#bootstrap-req` and
    the decode side never logging a single batch -- the KV handoff could not cross the EFA fabric
    between two non-host-network pods on different nodes at all. The same ratio with only one side
    packed (6:2, 2:6) had not shown this, because the *unpacked* side's `hostNetwork: true` carried
    the real path. Fixed by never turning it off here (DESIGN-RATIONALE.md, "a bootstrap port
    shared by every packed prefill worker is not a bootstrap port").

    Returns `(items, bootstrap_ports)`: `bootstrap_ports[i]` is the port `members[i]` actually
    bound, needed by `render()` to tell the router the right one per prefill URL -- see
    `worker()`'s own docstring for why packing more than one prefill worker needs this at all."""
    assert len(members) > 1, "packed_role_pod is for more than one worker; render() handles one worker directly"
    offset = 0
    plans, template_container, pod_spec, bootstrap_ports = [], None, None, []
    for name, tp in members:
        index = len(plans)
        port = 8998 + index if role == "prefill" else None
        tmpl = worker({**c, "gpus_per_worker": tp}, name, node, role, **({"bootstrap_port": port} if port is not None else {}))
        container = tmpl["spec"]["template"]["spec"]["containers"][0]
        if template_container is None:
            template_container = copy.deepcopy(container)
            pod_spec = copy.deepcopy(tmpl["spec"]["template"]["spec"])
        args = container["args"][:]
        args[args.index("--port") + 1] = str(30000 + index)
        env = {x["name"]: x["value"] for x in container["env"]}
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(offset, offset + tp))
        plans.append({"role": role, "args": args, "env": env})
        bootstrap_ports.append(port)
        offset += tp
    labels = OWNER | {"app": pod_name, "role": role, "component": "engine"}
    engine = pod_spec["containers"][0]
    engine["command"] = ["bash", "-c", "ulimit -l unlimited; python3 /lab/verify_image.py && exec python3 /lab/packed.py"]
    engine.pop("args", None)
    engine["env"] = template_container["env"] + [{"name": "AIM345_ENGINE_PLANS", "value": json.dumps(plans)}]
    for resources in engine["resources"].values():
        resources["nvidia.com/gpu"] = offset
        resources["cpu"] = str(c.get("packed_cpu", 8))
        resources["memory"] = c.get("packed_memory", "128Gi")
    ports = tuple(30000 + i for i in range(len(members)))
    health = f"import urllib.request; [urllib.request.urlopen('http://127.0.0.1:%d/health'%p, timeout=8).read() for p in {ports!r}]"
    for key in ("startupProbe", "readinessProbe"):
        engine[key].pop("httpGet", None)
        engine[key]["exec"] = {"command": ["python3", "-c", health]}
        engine[key]["timeoutSeconds"] = 20
    pod_dep = {"apiVersion": "apps/v1", "kind": "Deployment", "metadata": metadata(c, pod_name, labels),
               "spec": {"replicas": 1, "strategy": {"type": "Recreate"}, "selector": {"matchLabels": {"app": pod_name}},
                        "template": {"metadata": {"labels": labels}, "spec": pod_spec}}}
    services = []
    for index, (name, _tp) in enumerate(members):
        svc = service(c, name, 30000, headless=True)
        svc["spec"]["selector"] = OWNER | {"app": pod_name}
        svc["spec"]["ports"][0].update(port=30000 + index, targetPort=30000 + index)
        services.append(svc)
    return [pod_dep] + services, bootstrap_ports


def render(c, mode, prefill_replicas=1):
    assert prefill_replicas in (1, 2), "This lab implements one or two prefill workers"
    prefill_workers = c.get("prefill_workers", 1)
    decode_workers = c.get("decode_workers", 1)
    unified_replicas = c.get("unified_replicas", 2)
    prefill_tp = c.get("prefill_gpus_per_worker", c["gpus_per_worker"])
    decode_tp = c.get("decode_gpus_per_worker", c["gpus_per_worker"])
    unified_tp = c.get("unified_gpus_per_worker", c["gpus_per_worker"])
    custom_ratio = prefill_workers != 1 or decode_workers != 1 or prefill_tp != c["gpus_per_worker"] or decode_tp != c["gpus_per_worker"]
    custom_unified = unified_replicas != 2 or unified_tp != c["gpus_per_worker"]
    if mode == "unified":
        assert prefill_replicas == 1, "prefill_replicas is a disaggregated-only option"
        if not custom_unified:
            # Exact prior behavior and output, untouched: two unpacked replicas, one per node.
            role_groups = [(c["nodes"][0], [("unified-0", c["gpus_per_worker"])], "unified"),
                           (c["nodes"][1], [("unified-1", c["gpus_per_worker"])], "unified")]
        else:
            assert unified_replicas >= 1, "unified_replicas must be at least 1"
            grouped = {}
            for i in range(unified_replicas):
                node = c["nodes"][i % len(c["nodes"])]
                grouped.setdefault(node, []).append((f"unified-{i}", unified_tp))
            role_groups = [(node, members, "unified") for node, members in grouped.items()]
    else:
        assert len(c["nodes"]) >= prefill_replicas + 1, "Add a separately allocated node to scale prefill"
        if not custom_ratio:
            # Exact prior behavior and output, untouched: one prefill worker, one decode worker,
            # plus the optional scaled-prefill worker on a third node (`6.scale-prefill.sh`).
            role_groups = [(c["nodes"][0], [("prefill-0", c["gpus_per_worker"])], "prefill"),
                           (c["nodes"][1], [("decode-0", c["gpus_per_worker"])], "decode")]
            if prefill_replicas == 2:
                role_groups.append((c["nodes"][2], [("prefill-1", c["gpus_per_worker"])], "prefill"))
        else:
            assert prefill_replicas == 1, "prefill_replicas (scale-prefill) cannot combine with prefill_workers/decode_workers/a per-role GPU override"
            role_groups = [(c["nodes"][0], [(f"prefill-{i}", prefill_tp) for i in range(prefill_workers)], "prefill"),
                           (c["nodes"][1], [(f"decode-{i}", decode_tp) for i in range(decode_workers)], "decode")]
    items = [{"apiVersion": "scheduling.k8s.io/v1", "kind": "PriorityClass", "metadata": {"name": c["namespace"] + "-nonpreempting", "labels": OWNER}, "value": -10, "preemptionPolicy": "Never", "globalDefault": False, "description": "Nonpreempting priority for the AIM345 lab namespace"}]
    roles = []
    pod_count_by_role = {}
    for node, members, role in role_groups:
        if len(members) == 1:
            name, tp = members[0]
            worker_c = c if tp == c["gpus_per_worker"] else {**c, "gpus_per_worker": tp}
            items += [worker(worker_c, name, node, role), service(c, name, 30000, headless=True)]
            bootstrap_ports = [8998] if role == "prefill" else [None]
        else:
            # A role packed across more than one node (only possible for `unified`, whose
            # replicas spread round-robin across every node) needs one distinct pod name per
            # node it lands on.
            index = pod_count_by_role.get(role, 0)
            pod_count_by_role[role] = index + 1
            pod_name = f"{role}s" if index == 0 else f"{role}s-{index}"
            new_items, bootstrap_ports = packed_role_pod(c, node, members, role, pod_name)
            items += new_items
        roles += [(name, node, role, port) for (name, _), port in zip(members, bootstrap_ports)]
    # Use individual worker endpoints: one Service balancing replicas hides them from the router.
    url = lambda name: f"http://{name}.{c['namespace']}.svc.cluster.local:30000"
    args = ["--host", "0.0.0.0", "--port", "8000", "--policy", "round_robin", "--disable-retries", "--request-timeout-secs", "330"]
    if mode == "unified":
        args += ["--worker-urls", *[url(name) for name, _, _, _ in roles]]
    else:
        args += ["--pd-disaggregation"]
        for name, _, role, port in roles:
            args += (["--prefill", url(name), str(port)] if role == "prefill" else ["--decode", url(name)])
    labels = OWNER | {"app": "router", "component": "router"}
    pod = {"automountServiceAccountToken": False, "preemptionPolicy": "Never", "priorityClassName": c["namespace"] + "-nonpreempting", "containers": [{"name": "router", "image": c["router_image"], "command": ["python3", "-m", "sglang_router.launch_router"], "args": args, "resources": {"requests": {"cpu": "2", "memory": "4Gi", "ephemeral-storage": c["router_ephemeral_storage"]}, "limits": {"cpu": "4", "memory": "8Gi", "ephemeral-storage": c["router_ephemeral_storage"]}}, "readinessProbe": {"httpGet": {"path": "/health", "port": 8000}, "periodSeconds": 5}}]}
    items += [{"apiVersion": "apps/v1", "kind": "Deployment", "metadata": metadata(c, "router", labels), "spec": {"replicas": 1, "strategy": {"type": "Recreate"}, "selector": {"matchLabels": {"app": "router"}}, "template": {"metadata": {"labels": labels}, "spec": pod}}}, service(c, "router", 8000)]
    return {"apiVersion": "v1", "kind": "List", "items": items}


RUNNER_MODULES = ("atomic.py", "benchmark.py", "traffic.py", "runner_agent.py")


def runner_resources(c, node, shards):
    """CPU-only shard pods for the in-cluster runner (DESIGN.md section 3, item 2). Reuses the
    engine image rather than a dedicated one, so no new container image needs building or
    pushing; the pods request no GPU/EFA regardless of what the image can do. The engine image is
    pinned by digest and was built before this lab's runner_agent.py existed, so its baked /lab
    does not have it -- rather than publish a new image (this account has no in-cluster image
    builder, and finch is excluded for anything that runs on this cluster), runner_agent.py and
    its three dependencies are mounted from a ConfigMap at /lab instead, replacing that directory
    for this container only; the engine containers are untouched. One PVC per shard
    (ReadWriteOnce) is this lab account's EBS fallback for DESIGN.md section 8's shared-EFS
    design -- there is no EFS filesystem in this account (see README.md "Where results live").
    """
    labels = OWNER | {"app": "runner", "component": "runner"}
    resources = {"cpu": str(c.get("runner_cpu", "1")), "memory": c.get("runner_memory", "1Gi")}
    here = Path(__file__).parent
    code_cm = {"apiVersion": "v1", "kind": "ConfigMap", "metadata": metadata(c, "runner-code"),
              "data": {name: (here / name).read_text() for name in RUNNER_MODULES}}
    command = ["sh", "-c", 'python3 -c "import httpx, transformers" 2>/dev/null || python3 -m pip install --quiet httpx transformers; '
                           'exec python3 /lab/runner_agent.py --shard-index "${HOSTNAME##*-}"']
    spec = {
        "nodeSelector": {"kubernetes.io/hostname": node}, "tolerations": c["tolerations"],
        "automountServiceAccountToken": False, "preemptionPolicy": "Never", "priorityClassName": c["namespace"] + "-nonpreempting",
        "containers": [{"name": "runner", "image": c["image"], "imagePullPolicy": "IfNotPresent", "workingDir": "/lab",
                        "command": command, "resources": {"requests": resources, "limits": resources},
                        # Pod-to-pod only (DESIGN-RATIONALE.md, "whole-ramp dispatch"): shards exchange
                        # per-segment start/finish signals over this port via the headless `runner`
                        # Service's own per-pod DNS; labctl never connects to it, nothing is exposed
                        # outside the cluster, and this list is informational, not an admission gate.
                        "ports": [{"containerPort": 7777, "name": "peer-status"}],
                        "volumeMounts": [{"name": "local", "mountPath": "/local"}, {"name": "code", "mountPath": "/lab"}],
                        "readinessProbe": {"exec": {"command": ["test", "-d", "/local/jobs"]}, "periodSeconds": 5, "initialDelaySeconds": 5}}],
        "volumes": [{"name": "code", "configMap": {"name": "runner-code"}}],
    }
    svc = {"apiVersion": "v1", "kind": "Service", "metadata": metadata(c, "runner"), "spec": {"clusterIP": "None", "selector": OWNER | {"app": "runner"}}}
    sts = {"apiVersion": "apps/v1", "kind": "StatefulSet", "metadata": metadata(c, "runner", labels),
           "spec": {"replicas": shards, "serviceName": "runner", "selector": {"matchLabels": {"app": "runner"}},
                    "template": {"metadata": {"labels": labels}, "spec": spec},
                    "volumeClaimTemplates": [{"metadata": {"name": "local"}, "spec": {"accessModes": ["ReadWriteOnce"],
                                              "storageClassName": c.get("runner_storage_class", "gp3"),
                                              "resources": {"requests": {"storage": c.get("runner_storage", "2Gi")}}}}]}}
    return [svc, code_cm, sts]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["unified", "disaggregated", "scale-prefill", "runner", "cleanup"])
    p.add_argument("--config", default="config.json")
    p.add_argument("--node", help="Node the runner StatefulSet is pinned to (same node as the stack it measures)")
    p.add_argument("--shards", type=int, default=2)
    p.add_argument("--render", action="store_true")
    a = p.parse_args()
    c = read_config(a.config)
    if a.mode == "runner":
        owned_namespace(c, create=True)
        doc = {"apiVersion": "v1", "kind": "List", "items": runner_resources(c, a.node or c["nodes"][0], a.shards)}
        if a.render:
            print(json.dumps(doc, indent=2))
            return
        kubectl(c, "apply", "-f", "-", input=json.dumps(doc))
        print(f"Submitted runner StatefulSet with {a.shards} shard(s) on {a.node or c['nodes'][0]}. "
              "Participant cleanup never deletes this -- it is facilitator-owned, like the engines and router.")
        return
    assert c.get("placement") != "packed", "Use paired.py for packed allocations"
    if a.mode == "cleanup":
        owned_namespace(c)
        kubectl(c, "delete", "namespace", c["namespace"], "--wait=true")
        kubectl(c, "delete", "priorityclass", c["namespace"] + "-nonpreempting", "--ignore-not-found")
        return
    mode = "disaggregated" if a.mode == "scale-prefill" else a.mode
    doc = render(c, mode, 2 if a.mode == "scale-prefill" else 1)
    if a.render:
        print(json.dumps(doc, indent=2))
        return
    owned_namespace(c, create=True)
    if a.mode != "scale-prefill":
        kubectl(c, "delete", "deployment,service", "-l", "app.kubernetes.io/part-of=aim345-lab", "--ignore-not-found", "--wait=true")
        remaining = json.loads(kubectl(c, "get", "pods", "-l", "app.kubernetes.io/part-of=aim345-lab", "-o", "json", capture_output=True).stdout)["items"]
        for pod in remaining:
            kubectl(c, "wait", "--for=delete", "pod/" + pod["metadata"]["name"], "--timeout=180s")
    else:
        # Scaling adds a prefill worker and changes router membership only.
        existing = json.loads(kubectl(c, "get", "deploy", "-o", "json", capture_output=True).stdout)
        assert {"prefill-0", "decode-0", "router"} <= {x["metadata"]["name"] for x in existing["items"]}, "Deploy disaggregated first"
        # Reject configuration drift that would also restart existing engines.
        desired = {x["metadata"]["name"]: x for x in doc["items"] if x["kind"] == "Deployment"}
        for old in existing["items"]:
            if old["metadata"]["name"] not in ("prefill-0", "decode-0"):
                continue
            want = desired[old["metadata"]["name"]]["spec"]["template"]["spec"]
            have = old["spec"]["template"]["spec"]
            for key in ("nodeSelector", "tolerations"):
                assert have[key] == want[key], f"Existing engine {key} changed"
            for key in ("image", "args", "env"):
                assert have["containers"][0][key] == want["containers"][0][key], f"Existing engine {key} changed"
            normalize = lambda resources: {section: {k: str(v) for k, v in quantities.items()} for section, quantities in resources.items()}
            assert normalize(have["containers"][0]["resources"]) == normalize(want["containers"][0]["resources"]), "Existing engine resources changed"
        doc["items"] = [x for x in doc["items"] if x["metadata"]["name"] in ("prefill-1", "router")]
    kubectl(c, "apply", "-f", "-", input=json.dumps(doc))
    Path("results").mkdir(exist_ok=True)
    Path("results/deployment.json").write_text(json.dumps({"config": c, "mode": mode, "prefill_replicas": 2 if a.mode == "scale-prefill" else 1}, indent=2))
    print("Submitted. Wait for each Deployment to become Available; capture pod events if scheduling is blocked.")

if __name__ == "__main__":
    main()
