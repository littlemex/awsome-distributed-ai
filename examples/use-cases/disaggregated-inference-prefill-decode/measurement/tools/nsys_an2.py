"""Build-host analysis of an nsys sqlite export (decode TP0/TP1 under one launch session).
Receive windows per receiving process = its receive copies (PtoP or the staged DtoH Device->Device leg), merged
within 2 ms. For each window set, GPU metrics per physical GPU (mean inside vs outside) and non-NCCL kernel-busy
fraction per (process, ordinal). Usage: nsys_an2.py <sqlite> <decode physical GPU of ordinal 0> <of ordinal 1>"""
import sqlite3, sys, collections, bisect
db = sqlite3.connect(sys.argv[1]); c = db.cursor()
def cols(t): return [r[1] for r in c.execute(f"pragma table_info({t})")]
print("TARGET_INFO_CUDA_DEVICE", cols("TARGET_INFO_CUDA_DEVICE"))
for r in c.execute("select * from TARGET_INFO_CUDA_DEVICE limit 40"): print("   ", r)
print("TARGET_INFO_GPU", cols("TARGET_INFO_GPU"))
for r in c.execute("select * from TARGET_INFO_GPU"): print("   ", r[:8])
print("GPU metrics types", list(c.execute("select distinct typeId, typeName from TARGET_INFO_GPU_METRICS")))
try:
    P0, P1 = int(sys.argv[2]), int(sys.argv[3])
except Exception:
    P0, P1 = 2, 3
strs = {r[0]: r[1] for r in c.execute("select id, value from StringIds")}
ck = {r[0]: r[1] for r in c.execute("select id, name from ENUM_CUDA_MEMCPY_OPER")}
mk = {r[0]: r[1] for r in c.execute("select id, name from ENUM_CUDA_MEM_KIND")}
mem = list(c.execute("select start, end, deviceId, globalPid, copyKind, srcKind, dstKind, bytes, srcDeviceId, dstDeviceId from CUPTI_ACTIVITY_KIND_MEMCPY"))
def pid(g): return g >> 24 & 0xFFFFFF
recv = collections.defaultdict(list)
for s, e, d, g, k, sk, dk, b, sd, dd in mem:
    kn = ck.get(k, ""); src = mk.get(sk, ""); dst = mk.get(dk, "")
    if kn.endswith("PTOP") and sd is not None and sd not in (0, 1) and b >= 4096:
        recv[(pid(g), "PtoP", d)].append((s, e))
    if kn.endswith("DTOH") and src.endswith("DEVICE") and dst.endswith("DEVICE") and b >= 4096:
        recv[(pid(g), "staged-leg1", d)].append((s, e))
    if kn.endswith("DTOH") and dst.endswith("PINNED") and b >= 1_000_000:
        recv[(pid(g), "d2h-pinned-large", d)].append((s, e))
def merge(iv):
    out = []
    for a, b in sorted(iv):
        if out and a <= out[-1][1] + 2_000_000: out[-1][1] = max(out[-1][1], b)
        else: out.append([a, b])
    return out
win = {k: merge(v) for k, v in recv.items()}
for k, w in win.items(): print("windows", k, "n=%d total %.1f ms" % (len(w), sum(b - a for a, b in w) / 1e6), "copies", len(recv[k]))
names = {r[0]: r[1] for r in c.execute("select metricId, metricName from TARGET_INFO_GPU_METRICS")}
want = ("GR Active [Throughput %]", "SMs Active [Throughput %]", "Async Copy Engine Active 0 [Throughput %]", "Async Copy Engine Active 1 [Throughput %]",
        "Sync Copy Engine Active [Throughput %]", "PCIe RX Throughput [Throughput %]", "PCIe TX Throughput [Throughput %]", "DRAM Read Bandwidth [Throughput %]",
        "Compute Warps in Flight [Throughput %]")
gm = collections.defaultdict(list)
for ts, ty, mid, v in c.execute("select timestamp, typeId, metricId, value from GPU_METRICS"):
    n = names.get(mid, "")
    if n in want or n.startswith("Async Copy Engine Active"): gm[(ty & 0xFF, n)].append((ts, v))
for k in gm: gm[k].sort()
def inside(ws, t):
    i = bisect.bisect_right(ws[1], t) - 1
    return i >= 0 and t < ws[0][i][1]
ker = collections.defaultdict(list)
for s, e, d, g, sn in c.execute("select start, end, deviceId, globalPid, shortName from CUPTI_ACTIVITY_KIND_KERNEL"):
    if "nccl" not in strs.get(sn, "").lower(): ker[(pid(g), d)].append((s, e))
for k in ker: ker[k] = merge_busy = sorted(ker[k])
def busy_frac(iv, ws, inside_flag):
    # fraction of time inside (or outside) windows covered by kernels, by 50 us sampling
    starts = [a for a, b in iv]; hit = n = 0
    lo, hi = iv[0][0], iv[-1][1]; t = lo
    ends = []
    mx = 0
    pref = []
    for a, b in iv: mx = max(mx, b); pref.append(mx)
    while t < hi:
        if inside(ws, t) == inside_flag:
            j = bisect.bisect_right(starts, t) - 1
            n += 1; hit += j >= 0 and pref[j] > t
        t += 50_000
    return hit / max(1, n), n
phys = {0: P0, 1: P1}
for k, w in sorted(win.items()):
    ws = (w, [a for a, b in w])
    print(f"\n== receive windows of process {k[0]} ({k[1]} on ordinal {k[2]} = physical {phys.get(k[2])})")
    for g in (P0, P1):
        for n in want:
            v = gm.get((g, n))
            if not v: continue
            ins = [x for t, x in v if inside(ws, t)]; out = [x for t, x in v if not inside(ws, t)]
            if ins and out: print(f"   GPU {g} {n}: inside {sum(ins)/len(ins):.2f}  outside {sum(out)/len(out):.2f}  (n={len(ins)}/{len(out)})")
    for (p, d), iv in sorted(ker.items()):
        if d in (0, 1) and len(iv) > 1000:
            fi, ni = busy_frac(iv, ws, True); fo, no = busy_frac(iv, ws, False)
            print(f"   non-NCCL kernel busy, process {p} ordinal {d}: inside {fi:.3f} (n={ni}) outside {fo:.3f} (n={no})")
