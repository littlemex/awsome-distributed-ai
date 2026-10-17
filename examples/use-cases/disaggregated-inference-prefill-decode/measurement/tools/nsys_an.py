"""Runs inside the engine container on an nsys sqlite export. Prints a small text summary:
tables present; memcpy kinds by device/process; for each GPU, windows of the cross-device staged copy legs and
how GPU metrics (SM active, graphics engine, PCIe) and context switches differ inside vs outside those windows;
kernel busy per device inside vs outside. Usage: python3 nsys_an.py <export.sqlite>"""
import sqlite3, sys, collections, bisect, statistics as st
db = sqlite3.connect(sys.argv[1]); c = db.cursor()
tabs = [r[0] for r in c.execute("select name from sqlite_master where type='table'")]
print("tables:", " ".join(t for t in tabs if t.isupper() or t.startswith(("CUPTI", "GPU", "TARGET", "ENUM"))))
def has(t): return t in tabs
def cols(t): return [r[1] for r in c.execute(f"pragma table_info({t})")]
def enum(t):
    return {r[0]: r[1] for r in c.execute(f"select id, name from {t}")} if has(t) else {}
ck = enum("ENUM_CUDA_MEMCPY_OPER"); mk = enum("ENUM_CUDA_MEM_KIND")
mem = []
if has("CUPTI_ACTIVITY_KIND_MEMCPY"):
    mc = cols("CUPTI_ACTIVITY_KIND_MEMCPY")
    sel = ["start", "end", "deviceId", "contextId", "streamId", "copyKind", "srcKind", "dstKind", "bytes", "globalPid"]
    extra = [x for x in ("srcDeviceId", "dstDeviceId", "copyEngineId", "channelID", "channelType") if x in mc]
    print("memcpy columns extra:", extra)
    for r in c.execute(f"select {','.join(sel + extra)} from CUPTI_ACTIVITY_KIND_MEMCPY"):
        mem.append(dict(zip(sel + extra, r)))
    g = collections.defaultdict(list)
    for m in mem:
        g[(m["globalPid"] >> 24 & 0xFFFFFF, m["deviceId"], ck.get(m["copyKind"], m["copyKind"]), mk.get(m["srcKind"], m["srcKind"]), mk.get(m["dstKind"], m["dstKind"])) + tuple(m[x] for x in extra)].append((m["end"] - m["start"], m["bytes"]))
    print("memcpy groups: (pid, device, copyKind, src, dst, extra...) count median_us total_ms median_bytes")
    for k, v in sorted(g.items(), key=lambda kv: -len(kv[1]))[:14]:
        print("  ", k, len(v), "%.1f" % (st.median(d for d, b in v) / 1e3), "%.1f" % (sum(d for d, b in v) / 1e6), st.median(b for d, b in v))
ker = collections.defaultdict(list)
strs = {r[0]: r[1] for r in c.execute("select id, value from StringIds")} if has("StringIds") else {}
if has("CUPTI_ACTIVITY_KIND_KERNEL"):
    for s_, e, d, pid, sn in c.execute("select start, end, deviceId, globalPid, shortName from CUPTI_ACTIVITY_KIND_KERNEL"):
        if "nccl" in strs.get(sn, "").lower(): continue
        ker[d].append((s_, e, pid >> 24 & 0xFFFFFF))
for d in ker: ker[d].sort()
# staged legs: memcpy whose src and dst kinds are both device and copyKind is DtoH or HtoD (as torch showed), or PtoP
def is_leg(m):
    k = str(ck.get(m["copyKind"], m["copyKind"])); s = str(mk.get(m["srcKind"], "")); t = str(mk.get(m["dstKind"], ""))
    return ("DTOH" in k.upper() or "HTOD" in k.upper()) and "DEVICE" in s.upper() and "DEVICE" in t.upper() and m["bytes"] > 1_000_000
legs = sorted((m["start"], m["end"], m["deviceId"]) for m in mem if is_leg(m))
print("large cross-device legs:", len(legs), collections.Counter(l[2] for l in legs))
def merge(iv):
    out = []
    for a, b in sorted(iv):
        if out and a <= out[-1][1] + 2_000_000: out[-1][1] = max(out[-1][1], b)   # 2 ms join: one burst
        else: out.append([a, b])
    return out
bursts = merge([(a, b) for a, b, d in legs])
print("bursts (legs joined within 2 ms):", len(bursts), "total %.1f ms" % (sum(b - a for a, b in bursts) / 1e6))
bs = [a for a, b in bursts]
def inside(t):
    i = bisect.bisect_right(bs, t) - 1
    return i >= 0 and t < bursts[i][1]
# kernel busy per device inside vs outside bursts (sampled every 100 us)
if ker and bursts:
    lo = min(k[0][0] for k in ker.values()); hi = max(k[-1][1] for k in ker.values())
    for d, ks in sorted(ker.items()):
        starts = [k[0] for k in ks]; ins = outs = bi = bo = 0
        t = lo
        while t < hi:
            j = bisect.bisect_right(starts, t) - 1; busy = j >= 0 and ks[j][1] > t
            # look back a little for overlapping long kernels
            k = j; 
            while not busy and k > 0 and j - k < 50:
                k -= 1; busy = ks[k][1] > t
            if inside(t): ins += 1; bi += busy
            else: outs += 1; bo += busy
            t += 100_000
        print(f"device {d}: non-NCCL kernel-busy fraction inside bursts {bi / max(1, ins):.3f} (n={ins}) outside {bo / max(1, outs):.3f} (n={outs})")
# GPU metrics
def gpu_metrics():
  if not has("GPU_METRICS"): return
  if True:
    names = {}
    gm = cols("GPU_METRICS"); print("GPU_METRICS columns:", gm)
    if has("TARGET_INFO_GPU_METRICS"):
        tc = cols("TARGET_INFO_GPU_METRICS"); print("TARGET_INFO_GPU_METRICS columns:", tc)
        idc = "metricId" if "metricId" in tc else tc[0]; nc = "metricName" if "metricName" in tc else tc[1]
        names = {r[0]: r[1] for r in c.execute(f"select {idc}, {nc} from TARGET_INFO_GPU_METRICS")}
    agg = collections.defaultdict(lambda: [[], []])
    q = "select timestamp, typeId, metricId, value from GPU_METRICS"
    for ts, ty, mid, v in c.execute(q):
        agg[(ty & 0xFF, names.get(mid, mid))][0 if inside(ts) else 1].append(v)
    print("GPU metrics (device-ish typeId&0xFF, metric): mean inside bursts / outside")
    for k, (a, b) in sorted(agg.items(), key=lambda kv: str(kv[0])):
        if a and b and (sum(a) + sum(b)) > 0:
            print("  ", k, "%.2f / %.2f" % (sum(a) / len(a), sum(b) / len(b)), len(a), len(b))
try:
    gpu_metrics()
except Exception as ex:
    print("gpu metrics section failed:", ex)
for t in tabs:
    if "CONTEXT_SWITCH" in t.upper() or "CTXSW" in t.upper():
        cc = cols(t); print("ctxsw table", t, cc)
        rows = list(c.execute(f"select * from {t}"))
        tcol = cc.index("timestamp") if "timestamp" in cc else cc.index("start") if "start" in cc else 0
        dcol = cc.index("deviceId") if "deviceId" in cc else None
        cnt = collections.Counter()
        for r in rows: cnt[(r[dcol] if dcol is not None else -1, inside(r[tcol]))] += 1
        dur_in = sum(b - a for a, b in bursts) / 1e9; 
        print("   events (device, inside_burst): count", dict(cnt), "burst seconds %.3f" % dur_in)

# CUDA API calls by name and thread inside vs outside bursts (sync/wait calls)
try:
    tn = {}
    if has("ThreadNames"):
        for r in c.execute("select * from ThreadNames"): tn[r[-1] if len(r) > 2 else r[1]] = strs.get(r[0], r[0])
    cnt = collections.Counter(); tot = collections.Counter()
    for s_, e, nid, gtid in c.execute("select start, end, nameId, globalTid from CUPTI_ACTIVITY_KIND_RUNTIME"):
        n = strs.get(nid, str(nid))
        if any(k in n for k in ("Synchronize", "WaitEvent", "Memcpy", "EventRecord", "Launch", "IpcOpen", "SetDevice", "GraphLaunch")):
            key = (gtid >> 24 & 0xFFFFFF, gtid & 0xFFFFFF, n, inside(s_)); cnt[key] += 1; tot[key] += e - s_
    print("CUDA API (pid, tid, name, inside_burst): count total_ms  [top by total time]")
    for k, v in sorted(tot.items(), key=lambda kv: -kv[1])[:30]:
        print("  ", k, cnt[k], "%.1f" % (v / 1e6))
except Exception as ex:
    print("api section failed:", ex)
