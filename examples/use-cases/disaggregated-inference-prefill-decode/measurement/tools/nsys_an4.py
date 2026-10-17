"""In-pod, step level. Decode steps = intervals between consecutive cudaGraphLaunch calls of each decode
process's main thread (decode processes = the two processes with the most cudaGraphLaunch calls). Each step is
classed by which large copies (>= 1 MB, any process) overlap it on each decode GPU ordinal: staged leg
(DtoH Device->Device), PtoP, d2h-pinned, none. Per class: n, median step, median main-thread time inside
cudaMemcpyAsync / cudaEventSynchronize per step. Usage: nsys_an4.py <sqlite>"""
import sqlite3, sys, collections, bisect, statistics as st
db = sqlite3.connect(sys.argv[1]); c = db.cursor()
strs = {r[0]: r[1] for r in c.execute("select id, value from StringIds")}
ck = {r[0]: r[1] for r in c.execute("select id, name from ENUM_CUDA_MEMCPY_OPER")}
mk = {r[0]: r[1] for r in c.execute("select id, name from ENUM_CUDA_MEM_KIND")}
def pid(g): return g >> 24 & 0xFFFFFF
def tid(g): return g & 0xFFFFFF
calls = collections.defaultdict(list)
for s, e, nid, g in c.execute("select start, end, nameId, globalTid from CUPTI_ACTIVITY_KIND_RUNTIME"):
    n = strs.get(nid, "")
    if n.startswith(("cudaGraphLaunch", "cudaMemcpyAsync", "cudaEventSynchronize")):
        calls[(pid(g), tid(g), n.split("_")[0])].append((s, e))
gl = sorted(((k[0], k[1]), len(v)) for k, v in calls.items() if k[2] == "cudaGraphLaunch")
dec = [k for k, n in sorted(gl, key=lambda x: -x[1])[:2]]
print("decode main threads (pid, tid) by graph launches:", dec)
big = []
for s, e, d, g, k, sk, dk, b in c.execute("select start, end, deviceId, globalPid, copyKind, srcKind, dstKind, bytes from CUPTI_ACTIVITY_KIND_MEMCPY where bytes >= 1000000"):
    kn = ck.get(k, ""); src = mk.get(sk, ""); dst = mk.get(dk, "")
    cls = "staged" if kn.endswith("DTOH") and dst.endswith("DEVICE") else "ptop" if kn.endswith("PTOP") else "d2hpinned" if kn.endswith("DTOH") and dst.endswith("PINNED") else kn
    big.append((s, e, d, pid(g), cls))
big.sort(); bs = [x[0] for x in big]
def classes(a, b):
    i = max(0, bisect.bisect_left(bs, a) - 200); out = set()
    while i < len(big) and big[i][0] < b:
        s, e, d, p, cl = big[i]
        if e > a: out.add((cl, d, p))
        i += 1
    return tuple(sorted(out)) or ("none",)
def time_in(lst, a, b):
    t = 0
    for s, e in lst:
        if s >= b: break
        if e > a: t += min(e, b) - max(s, a)
    return t
for p, t in dec:
    st_ = sorted(s for s, e in calls[(p, t, "cudaGraphLaunch")])
    mc = sorted(calls.get((p, t, "cudaMemcpyAsync"), [])); es = sorted(calls.get((p, t, "cudaEventSynchronize"), []))
    rows = collections.defaultdict(list)
    for i in range(len(st_) - 1):
        a, b = st_[i], st_[i + 1]
        if b - a > 2e9: continue
        k = classes(a, b)
        jm = bisect.bisect_left([x[0] for x in mc], a - 50_000_000); je = bisect.bisect_left([x[0] for x in es], a - 50_000_000)
        rows[k].append(((b - a) / 1e6, time_in(mc[jm:], a, b) / 1e6, time_in(es[je:], a, b) / 1e6))
    print(f"\ndecode main thread {p}/{t}: steps by overlapping large copies (class, ordinal, pid): n, median step ms, median ms in cudaMemcpyAsync, in cudaEventSynchronize")
    for k, v in sorted(rows.items(), key=lambda kv: -len(kv[1]))[:10]:
        m = lambda j: st.median(x[j] for x in v)
        print("  ", k, len(v), "%.2f %.2f %.2f" % (m(0), m(1), m(2)))
