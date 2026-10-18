"""Join each client request to its prefill forwards (client rid -> decode rid -> bootstrap_room -> prefill rid ->
dispatch lines) and print, per measured file, the cached prefix and attention path of each request's forwards.
Usage: join_room.py <cell dir>"""
import json, re, sys, glob
from collections import Counter, defaultdict
OUT = sys.argv[1]
disp = re.compile(r"\[[^\]]* TP(\d)\] a345_dispatch pid=(\d+) wall=([\d.]+) method=(\w+) n_req=(\d+) prefix=([\d,]*) new=([\d,]*) rids=(\S*)")
room = re.compile(r"\[[^\]]* TP(\d)\] a345_room side=(\w+) pid=(\d+) wall=([\d.]+) rid=(\S+) room=(\d+) cached=(\d+)(?: input_len=(\d+))?")
lines, bad_shape, unparsed = [], 0, Counter()
rooms = defaultdict(lambda: defaultdict(dict))  # room -> side -> tp -> (rid, cached, wall, input_len)
for line in open(f"{OUT}/engine.log", errors="replace"):
    if "a345_dispatch" in line and not disp.search(line): unparsed["dispatch"] += 1
    if "a345_room" in line and not room.search(line): unparsed["room"] += 1
    m = disp.search(line)
    if m:
        tp, pid, w, meth, n, p, nw, rids = m.groups()
        p = [int(x) for x in p.split(",") if x]; nw = [int(x) for x in nw.split(",") if x]; rids = [r for r in rids.split(",") if r]
        if not (len(p) == len(nw) == len(rids) == int(n)):
            bad_shape += 1
        lines.append(dict(tp=tp, pid=pid, wall=float(w), method=meth, prefix=p, new=nw, rids=rids))
        continue
    m = room.search(line)
    if m:
        tp, side, pid, w, rid, rm, cached, il = m.groups()
        if rm == "0":
            continue  # fake transfers (warm-up / health check), never a client request
        if tp in rooms[rm][side]:
            rooms[rm][side][tp + "dup"] = 1
        rooms[rm][side][tp] = (rid, int(cached), float(w), il)
fwd = defaultdict(list)
for i, l in enumerate(lines):
    for j, r in enumerate(l["rids"]):
        fwd[r].append(dict(line=i, wall=l["wall"], method=l["method"], prefix=l["prefix"][j], new=l["new"][j], batch_sum=sum(l["prefix"])))
dec_by_rid = {}
room_problems = Counter()
for rm, sides in rooms.items():
    for side in ("prefill", "decode"):
        s = sides.get(side, {})
        if any(k.endswith("dup") for k in s): room_problems[f"{side}_dup_line"] += 1
        if set(k for k in s if not k.endswith("dup")) != {"0", "1"}: room_problems[f"{side}_not_both_ranks"] += 1; continue
        if (s["0"][0], s["0"][1], s["0"][3]) != (s["1"][0], s["1"][1], s["1"][3]): room_problems[f"{side}_tp_value_mismatch"] += 1
        if side == "prefill" and s["0"][3] is None: room_problems["prefill_input_len_missing"] += 1
    if "0" in sides.get("decode", {}):
        r = sides["decode"]["0"][0]
        if r in dec_by_rid: room_problems["decode_rid_in_two_rooms"] += 1
        dec_by_rid[r] = rm
prefill_rooms = Counter(sides["prefill"]["0"][0] for sides in rooms.values() if "0" in sides.get("prefill", {}))
room_problems["prefill_rid_in_two_rooms"] = sum(1 for n in prefill_rooms.values() if n > 1)
print(f"{OUT}: dispatch lines={len(lines)} (bad shape {bad_shape}), unparsed marker lines={dict(unparsed)}, real rooms={len(rooms)}, room problems={dict(room_problems)}")
served = Counter(); client_prefill_rids = set()
all_client = Counter(); client_rids = Counter()
for f in sorted(glob.glob(f"{OUT}/seed*/*.json")):
    reqs = json.load(open(f))["requests"]
    c = Counter(); notes = []
    for r in reqs:
        all_client["records"] += 1; client_rids[r.get("rid")] += 1
        if not r.get("ok"):
            c["not_ok"] += 1; continue
        cc = r.get("cached_tokens")
        if type(cc) is not int or cc < 0:
            c["cached_missing_or_invalid"] += 1; continue
        rm = dec_by_rid.get(r.get("rid"))
        if rm is None:
            c["no_decode_line"] += 1; continue
        pre = rooms[rm].get("prefill", {}).get("0")
        if not pre:
            c["no_prefill_line"] += 1; continue
        prid, pcached, pwall, il = pre
        client_prefill_rids.add(prid)
        evs = sorted(fwd.get(prid, []), key=lambda e: e["wall"])
        if not evs:
            c["no_forward"] += 1; continue
        if any(e["wall"] > pwall for e in evs): c["a_forward_after_set_buf"] += 1
        if any(b["prefix"] != a["prefix"] + a["new"] for a, b in zip(evs, evs[1:])): c["chunk_progression_broken"] += 1
        if il is None: c["input_len_missing"] += 1
        elif sum(e["new"] for e in evs) + evs[0]["prefix"] != int(il): c["tokens_dont_add_up"] += 1
        else: c["tokens_add_up"] += 1
        if cc == 0 and any(x == "MLA" for x in [evs[0]["method"]]) is False and "cache_on" in OUT: pass
        dcached = rooms[rm]["decode"]["0"][1]
        c["joined"] += 1
        c["client==decode==prefill"] += (cc == dcached == pcached)
        c["client==first_prefix"] += (cc == evs[0]["prefix"])
        first = evs[0]
        c[("cache>=1" if cc >= 1 else "cache=0") + "/first:" + ("batch>0" if first["batch_sum"] > 0 else "batch=0") + "/" + first["method"]] += 1
        c[f"forwards_per_request={len(evs)}"] += 1
        if cc == 0 and OUT.rstrip("/").endswith("d-p1-on"):
            notes.append(f"zero-cache client rid={r['rid']} start_wall={r.get('start_wall'):.3f} first_forward_wall={evs[0]['wall']:.3f} batch_prefix={evs[0]['batch_sum']} first_dispatch_line_index={evs[0]['line']}")
        for e in evs[1:]:
            c["later/own>0/" + e["method"] if e["prefix"] > 0 else "later/own=0/" + e["method"]] += 1
    print(f"  {f.split(OUT)[-1]}: " + ", ".join(f"{k}={v}" for k, v in sorted(c.items())))
    for n in notes: print("    " + n)
dups = [r for r, n in client_rids.items() if n > 1]
print(f"  client records={all_client['records']}, distinct rids={len(client_rids)}, duplicated rids={len(dups)}")
for l in lines:
    kinds = set()
    for r in l["rids"]:
        kinds.add("health" if r.startswith("HEALTH_CHECK") else ("client" if r in client_prefill_rids else "other"))
    served[(l["method"], "+".join(sorted(kinds)) or "no-rid")] += 1
    if "other" in kinds: print(f"  'other' dispatch line: index {lines.index(l)} wall={l['wall']:.3f} rids={l['rids']} prefix={l['prefix']} new={l['new']}")
print("  dispatch lines by method and what they served: " + ", ".join(f"{m}/{k}={v}" for (m, k), v in sorted(served.items())))
