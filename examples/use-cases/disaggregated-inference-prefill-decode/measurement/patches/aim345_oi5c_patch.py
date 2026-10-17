"""OPEN-ITEMS-NODE5 A control, applied after aim345_oi5_patch.py: the per-rank interface list may come from
/dev/shm/aim345_efa_ifaces (the pod's memory volume survives an engine container restart), so one pod can be
measured with several assignments on the same GPUs and devices; the start-up line also names the role and
CUDA_VISIBLE_DEVICES."""
import pathlib
import sglang

p = pathlib.Path(sglang.__file__).parent / "srt" / "managers" / "scheduler.py"
t = p.read_text()
old = """    _a5ifs = [x for x in _a5os.environ.get("AIM345_EFA_IFACES", "").split(",") if x]"""
new = """    _a5src = _a5os.environ.get("AIM345_EFA_IFACES", "")
    if _a5os.path.exists("/dev/shm/aim345_efa_ifaces"):
        _a5src = open("/dev/shm/aim345_efa_ifaces").read().strip()
    _a5ifs = [x for x in _a5src.split(",") if x]
    print(f"[aim345-oi5c] pid={_a5os.getpid()} ppid={_a5os.getppid()} role={server_args.disaggregation_mode} tp_rank={tp_rank} gpu_id={gpu_id} CUDA_VISIBLE_DEVICES={_a5os.environ.get('CUDA_VISIBLE_DEVICES')} ifaces={_a5src!r}", flush=True)"""
assert t.count(old) == 1
p.write_text(t.replace(old, new))
print("aim345_oi5c_patch applied OK")
