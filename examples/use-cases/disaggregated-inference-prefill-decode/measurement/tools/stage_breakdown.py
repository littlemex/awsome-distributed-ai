"""Per-window transfer-stage breakdown from the pct image's kvtimer lines (TP0), beside SGLang's own
prefill_transfer_kv_cache counter for the same window. Usage: stage_breakdown.py <stage_cell output dir>"""
import calendar, json, re, sys, time, statistics as st
from pathlib import Path
LINE = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) TP0\] kvtimer room=(-?\d+) event=(\S+) ts=([\d.]+)(?: descriptor_count=(\d+))?")
def rank(v):
    o = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0]*len(v); i = 0
    while i < len(v):
        j = i
        while j+1 < len(v) and v[o[j+1]] == v[o[i]]: j += 1
        for k in range(i, j+1): r[o[k]] = (i+j)/2
        i = j+1
    return r
def spearman(a, b):
    if len(a) < 3: return None
    ra, rb = rank(a), rank(b); ma, mb = st.mean(ra), st.mean(rb)
    num = sum((x-ma)*(y-mb) for x, y in zip(ra, rb)); den = (sum((x-ma)**2 for x in ra)*sum((y-mb)**2 for y in rb))**.5
    return round(num/den, 3) if den else None
def main(cell):
    d = Path(cell); ev = {}
    for line in open(d/'engine.log', errors='replace'):
        m = LINE.match(line)
        if not m: continue
        wall = calendar.timegm(time.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"))
        room = int(m.group(2)); e = m.group(3).split('_')[0]; ts = float(m.group(4))
        rec = ev.setdefault(room, {})
        rec.setdefault(e, ts); rec.setdefault('wall_'+e, wall)
        if m.group(5): rec['desc'] = rec.get('desc', 0) + int(m.group(5))
    wins = [re.match(r"seed=(\d+) rate=(\S+) epoch_window=\[(\d+),(\d+)\]", l).groups() for l in open(d/'windows.txt')]
    out = []
    for seed, rate, a, b in wins:
        rows = [r for room, r in ev.items() if room != 0 and all(k in r for k in ('T0','T1','T2','T3','T4','T5')) and int(a) <= r['wall_T0'] <= int(b)]
        f = lambda k1, k2: [(r[k2]-r[k1])*1000 for r in rows]
        tot = f('T0','T5'); wait = f('T3','T4'); tail = f('T4','T5'); desc = [r.get('desc', 0) for r in rows]
        sg = json.load(open(d/f'seed{seed}-rate{rate}.stages.json'))
        sgl = [x for x in sg if x['stage'] == 'prefill_transfer_kv_cache' and (x.get('engine_type') or '').startswith('prefill')]
        big = [i for i, x in enumerate(desc) if x >= 270]
        out.append({'seed': seed, 'rate': rate, 'n': len(rows),
            'mean_ms': {k: round(st.mean(v), 2) for k, v in (('dequeue', f('T0','T1')), ('build', f('T1','T2')), ('post', f('T2','T3')), ('wait', wait), ('tail', tail), ('total', tot))} if rows else None,
            'sglang_transfer_mean_ms': round(sgl[0]['mean_ms'], 2) if sgl else None, 'sglang_obs': sgl[0]['observations'] if sgl else None,
            'sum_ms_ours': round(sum(tot), 1), 'sum_ms_sglang': round(sgl[0]['mean_ms']*sgl[0]['observations'], 1) if sgl else None,
            'desc_median': st.median(desc) if desc else None, 'desc_max': max(desc) if desc else None, 'desc_ge270': len(big),
            'rho_desc_total': spearman(desc, tot), 'rho_desc_wait': spearman(desc, wait),
            'big_wait_ms': [round(wait[i], 1) for i in big], 'big_total_ms': [round(tot[i], 1) for i in big],
            'median_wait_ms': round(st.median(wait), 2) if wait else None, 'median_total_ms': round(st.median(tot), 2) if tot else None})
    print(json.dumps(out, indent=1))
main(sys.argv[1])
