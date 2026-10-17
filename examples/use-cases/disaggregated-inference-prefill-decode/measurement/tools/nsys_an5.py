"""In-pod. What a decode main thread's slow cudaMemcpyAsync waits for. For each decode main thread (the two
threads with the most cudaGraphLaunch calls) and each of its cudaMemcpyAsync calls lasting >= 0.5 ms, the GPU copy
of that call (same correlationId) is found; the blocker is the GPU activity (copy or kernel, any stream of the same
process and device) whose end is the latest one at or before that copy's start. Prints, per thread, the main
copy's stream and stream type, and counts of blockers by (kind, stream, stream type, same stream?) with the median
gap between blocker end and main-copy start. Usage: nsys_an5.py <sqlite>"""
import sqlite3, sys, collections, bisect, statistics as st
db = sqlite3.connect(sys.argv[1]); c = db.cursor()
strs = {r[0]: r[1] for r in c.execute("select id, value from StringIds")}
ck = {r[0]: r[1] for r in c.execute("select id, name from ENUM_CUDA_MEMCPY_OPER")}
def pid(g): return g >> 24 & 0xFFFFFF
def tid(g): return g & 0xFFFFFF
stype = {}
try:
    se = {r[0]: r[1] for r in c.execute("select id, name from ENUM_CUPTI_STREAM_TYPE")}
    cols = [r[1] for r in c.execute("pragma table_info(TARGET_INFO_CUDA_STREAM)")]
    print("TARGET_INFO_CUDA_STREAM columns:", cols)
    for row in c.execute("select * from TARGET_INFO_CUDA_STREAM"):
        d = dict(zip(cols, row)); stype[(d.get("processId"), d.get("streamId"))] = (se.get(d.get("streamType"), d.get("streamType")), d.get("priority"), d.get("flag"))
except Exception as ex:
    print("stream info unavailable:", ex)
rt = collections.defaultdict(list); corr = {}
apis = collections.defaultdict(list)
for s, e, nid, g, cid in c.execute("select start, end, nameId, globalTid, correlationId from CUPTI_ACTIVITY_KIND_RUNTIME"):
    n = strs.get(nid, "").split("_")[0]
    if n in ("cudaGraphLaunch", "cudaMemcpyAsync"):
        rt[(pid(g), tid(g), n)].append((s, e, cid))
    apis[pid(g)].append((e, s, tid(g), n))
for k in apis: apis[k].sort()
aend = {k: [x[0] for x in v] for k, v in apis.items()}
gl = sorted(((k[0], k[1]), len(v)) for k, v in rt.items() if k[2] == "cudaGraphLaunch")
dec = [k for k, n in sorted(gl, key=lambda x: -x[1])[:2]]
print("decode main threads:", dec)
acts = collections.defaultdict(list); bycorr = {}
for s, e, d, g, k, b, sid, cid in c.execute("select start, end, deviceId, globalPid, copyKind, bytes, streamId, correlationId from CUPTI_ACTIVITY_KIND_MEMCPY"):
    kn = ck.get(k, str(k)).replace("CUDA_MEMCPY_KIND_", "")
    kind = kn + (">=1MB" if b >= 1000000 else "<1MB")
    acts[(pid(g), d)].append((e, s, kind, sid)); bycorr[(pid(g), cid)] = (s, e, d, sid, kind)
for s, e, d, g, sid in c.execute("select start, end, deviceId, globalPid, streamId from CUPTI_ACTIVITY_KIND_KERNEL"):
    acts[(pid(g), d)].append((e, s, "kernel", sid))
for k in acts: acts[k].sort()
ends = {k: [x[0] for x in v] for k, v in acts.items()}
for p, t in dec:
    slow = [(s, e, cid) for s, e, cid in rt.get((p, t, "cudaMemcpyAsync"), []) if e - s >= 500_000]
    parts = []; rel = collections.Counter(); xrel = collections.Counter(); inflight = collections.Counter()
    out = collections.Counter(); gaps = collections.defaultdict(list); mains = collections.Counter(); nomatch = 0; apims = []
    for s, e, cid in slow:
        m = bycorr.get((p, cid))
        if not m: nomatch += 1; continue
        ms, me, d, sid, kind = m; mains[(kind, sid, stype.get((p, sid)))] += 1; apims.append((e - s) / 1e6)
        parts.append(((ms - s) / 1e6, (me - ms) / 1e6, (e - me) / 1e6))
        # another thread of the same process whose API call ends within 100 us before this call returns
        A = apis[p]; j = bisect.bisect_right(aend[p], e) - 1; names = set()
        while j >= 0 and A[j][0] >= e - 100_000:
            if A[j][2] != t and A[j][1] < s: names.add(A[j][3])
            j -= 1
        for nm in names or {"(none)"}: rel[nm] += 1
        # the same for every other process (cross-process holders)
        xn = set()
        for q in apis:
            if q == p: continue
            B = apis[q]; j = bisect.bisect_right(aend[q], e) - 1
            while j >= 0 and B[j][0] >= e - 100_000:
                if B[j][1] < s and B[j][0] - B[j][1] > 200_000: xn.add((q, B[j][3]))
                j -= 1
        for nm in xn or {("(none)",)}: xrel[nm] += 1
        # same-process copies on other streams in flight while this call waited for its GPU copy to start
        L = acts[(p, d)]; fl = set()
        for x in L[bisect.bisect_left(ends[(p, d)], s):]:
            if x[0] > ms + 200_000_000: break
            if x[1] < ms and x[3] != sid: fl.add((x[2], x[3]))
        for f in fl or {("(none)",)}: inflight[f] += 1
        L = acts[(p, d)]; i = bisect.bisect_right(ends[(p, d)], ms) - 1
        while i >= 0 and L[i][1] == ms and L[i][3] == sid: i -= 1
        if i < 0: out[("none",)] += 1; continue
        be, bs, bk, bsid = L[i]
        key = (bk, bsid, stype.get((p, bsid)), "same-stream" if bsid == sid else "other-stream")
        out[key] += 1; gaps[key].append((ms - be) / 1e3)
    print(f"\nthread {p}/{t}: slow cudaMemcpyAsync (>=0.5 ms) {len(slow)}, unmatched {nomatch}, median API ms {st.median(apims) if apims else 0:.2f}")
    print("  main copy (kind, stream, stream type/priority/flag):", mains.most_common(4))
    if parts:
        print("  median ms: API start->GPU copy start %.2f, GPU copy %.3f, GPU copy end->API end %.2f" % tuple(st.median(x[i] for x in parts) for i in range(3)))
    print("  other-thread API calls (same pid) ending within 100 us before the call returns, started before it:", rel.most_common(6))
    print("  other-process API calls (>0.2 ms) ending within 100 us before the call returns, started before it:", xrel.most_common(6))
    print("  other-stream activity (same pid, device) in flight before the GPU copy started:", inflight.most_common(6))
    for k, n in out.most_common(8):
        print("  blocker", k, n, "median gap us %.1f" % (st.median(gaps[k]) if gaps[k] else -1))

# libfabric-side map/unmap calls: per (pid, tid, name) count, median, p90, total ms; and for each process, the share
# of its Open/Close time that overlaps large (>= 1 MB) copies of OTHER processes on any device.
ipc = collections.defaultdict(list)
for q, L in apis.items():
    for e, s, t, n in L:
        if n in ("cudaIpcOpenMemHandle", "cudaIpcCloseMemHandle", "cudaMemcpy"): ipc[(q, t, n)].append((s, e))
bigx = sorted((s, e, q, d) for (q, d), L in acts.items() for e, s, kind, sid in L if kind.endswith(">=1MB"))
print("\nlibfabric-side calls (pid, tid, name): n, median ms, p90 ms, total ms, share of time overlapping other processes' >=1MB copies (by device)")
for k, v in sorted(ipc.items(), key=lambda kv: -sum(e - s for s, e in kv[1]))[:10]:
    du = sorted((e - s) / 1e6 for s, e in v); tot = sum(du)
    ov = collections.Counter()
    for s, e in v:
        for bs_, be, q, d in bigx:
            if bs_ >= e: break
            if be > s and q != k[0]: ov[d] += (min(e, be) - max(s, bs_)) / 1e6
    print("  ", k, len(du), "%.3f %.3f %.1f" % (du[len(du) // 2], du[int(len(du) * 0.9)], tot), {d: round(x / tot, 3) if tot else 0 for d, x in ov.items()})

# For each long (>= 5 ms) cudaIpcCloseMemHandle: the GPU activity (any process, any device) that ended last at or
# before the call returned, and the gap; plus whether that process's own device had activity in flight at the start.
allacts = sorted((e, s, q, d, kind, sid) for (q, d), L in acts.items() for e, s, kind, sid in L)
allend = [x[0] for x in allacts]
print("\nlong cudaIpcCloseMemHandle (>= 5 ms): last GPU activity ending before return (pid, device, kind) counts, median gap us")
for k, v in ipc.items():
    if k[2] != "cudaIpcCloseMemHandle": continue
    lg = [(s, e) for s, e in v if e - s >= 5_000_000]
    if not lg: continue
    cnt = collections.Counter(); gp = collections.defaultdict(list)
    for s, e in lg:
        j = bisect.bisect_right(allend, e) - 1
        if j < 0: continue
        be, bs_, q, d, kind, sid = allacts[j]
        key = (q, d, "kernel" if kind == "kernel" else kind, "same-pid" if q == k[0] else "other-pid")
        cnt[key] += 1; gp[key].append((e - be) / 1e3)
    print("  ", k, "long calls", len(lg), "median ms %.1f" % st.median((e - s) / 1e6 for s, e in lg))
    for key, n in cnt.most_common(6): print("     ", key, n, "gap us %.1f" % st.median(gp[key]))
