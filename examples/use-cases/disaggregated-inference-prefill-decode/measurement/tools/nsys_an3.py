"""In-pod: decode processes' small copies (< 64 KiB, the decode step's own memcpys) -- queueing delay from the
API call to the copy's GPU start, and copy duration -- inside vs outside the receive windows (any process's
receive copies, PtoP or staged leg 1, on the same CUDA ordinal, merged within 2 ms). Also which long copies
were running when a small copy waited. Usage: nsys_an3.py <sqlite>"""
import sqlite3, sys, collections, bisect, statistics as st
db = sqlite3.connect(sys.argv[1]); c = db.cursor()
ck = {r[0]: r[1] for r in c.execute("select id, name from ENUM_CUDA_MEMCPY_OPER")}
mk = {r[0]: r[1] for r in c.execute("select id, name from ENUM_CUDA_MEM_KIND")}
api = {}
for s, cid, g in c.execute("select start, correlationId, globalTid from CUPTI_ACTIVITY_KIND_RUNTIME"):
    api[(g >> 24 & 0xFFFFFF, cid)] = s
rows = list(c.execute("select start, end, deviceId, globalPid, copyKind, srcKind, dstKind, bytes, srcDeviceId, correlationId, streamId from CUPTI_ACTIVITY_KIND_MEMCPY"))
def pid(g): return g >> 24 & 0xFFFFFF
longc = collections.defaultdict(list)
for s, e, d, g, k, sk, dk, b, sd, cid, stid in rows:
    kn = ck.get(k, ""); src = mk.get(sk, ""); dst = mk.get(dk, "")
    if b >= 4096 and (kn.endswith("PTOP") and sd not in (0, 1) or (kn.endswith("DTOH") and src.endswith("DEVICE") and dst.endswith("DEVICE"))):
        longc[d].append((s, e, pid(g), kn))
for d in longc: longc[d].sort()
def merge(iv):
    out = []
    for a, b in sorted(iv):
        if out and a <= out[-1][1] + 2_000_000: out[-1][1] = max(out[-1][1], b)
        else: out.append([a, b])
    return out
win = {d: merge([(a, b) for a, b, p, k in v]) for d, v in longc.items()}
ws = {d: [a for a, b in w] for d, w in win.items()}
def inside(d, t):
    if d not in win: return False
    i = bisect.bisect_right(ws[d], t) - 1
    return i >= 0 and t < win[d][i][1]
ls = {d: [a for a, b, p, k in v] for d, v in longc.items()}
def running(d, t):
    if d not in longc: return None
    i = bisect.bisect_right(ls[d], t) - 1
    while i >= 0 and i > bisect.bisect_right(ls[d], t) - 40:
        a, b, p, k = longc[d][i]
        if a <= t < b: return (p, k)
        i -= 1
    return None
stat = collections.defaultdict(list)
blk = collections.Counter()
for s, e, d, g, k, sk, dk, b, sd, cid, stid in rows:
    if b >= 65536: continue
    a = api.get((pid(g), cid))
    if a is None: continue
    key = (pid(g), d, ck.get(k, "").replace("CUDA_MEMCPY_KIND_", ""), inside(d, s))
    stat[key].append(((s - a) / 1e3, (e - s) / 1e3))
    if inside(d, s) and (s - a) > 200_000:
        r = running(d, a + (s - a) / 2); blk[(pid(g), d, r)] += 1
print("small copies (pid, ordinal, kind, inside receive window): n, API->GPU-start delay p50/p90/max us, duration p50 us")
for kk, v in sorted(stat.items()):
    if len(v) < 5: continue
    dl = sorted(x for x, y in v); du = sorted(y for x, y in v)
    print("  ", kk, len(v), "%.0f / %.0f / %.0f" % (dl[len(dl) // 2], dl[int(len(dl) * .9)], dl[-1]), "%.1f" % du[len(du) // 2])
print("small copies delayed > 200 us inside windows, by the long copy running at the midpoint of their wait (pid, ordinal, (long-copy pid, kind)):")
for kk, v in blk.most_common(12): print("  ", kk, v)
print("long copies per ordinal: (ordinal) n total_ms by (pid, kind)")
for d, v in sorted(longc.items()):
    cc = collections.Counter(); tt = collections.Counter()
    for a, b, p, k in v: cc[(p, k)] += 1; tt[(p, k)] += b - a
    print("  ", d, {kk: (cc[kk], round(tt[kk] / 1e6, 1)) for kk in cc})
