"""A: per TP rank completion wait (last chunk), from ring dumps; rooms inside the given ramps only if
windows.txt exists (all non-warm-up rooms otherwise)."""
import json,sys,collections,statistics as st
c=sys.argv[1]
recs=collections.defaultdict(list); pid=None
for l in open(f'{c}/ring-dumps.jsonl'):
    if l.startswith('==='): pid=l.strip().split('_')[-1].split('.')[0]; continue
    try: recs[pid].append(json.loads(l))
    except Exception: pass
org={p:next(r for r in rs if r['event']=='CLOCK_ORIGIN') for p,rs in recs.items() if any(r['event']=='CLOCK_ORIGIN' for r in rs)}
wins=[]
try:
    for l in open(f'{c}/windows.txt'):
        a,b=l.split('epoch_window=[')[1].rstrip(']\n').split(','); wins.append((float(a),float(b)))
except FileNotFoundError: pass
W=collections.defaultdict(dict)
for p,rs in recs.items():
    tp=next((r.get('tp') for r in rs if r['event']=='T5_stage_end'),None)
    if tp is None or p not in org: continue
    E=lambda x,p=p: org[p]['origin_epoch']+(x-org[p]['ts'])
    for r in rs:
        if r['event']=='wait' and r.get('is_last') and r['room']!=0:
            t0=E(r['t_start'])
            if wins and not any(a<=t0<=b for a,b in wins): continue
            W[r['room']][tp]=(t0,E(r['t_done']))
for tp in (0,1):
    w=[(d[tp][1]-d[tp][0])*1000 for d in W.values() if tp in d]
    print(c.split('/')[-1],'tp',tp,'n',len(w),'wait median %.2f p10 %.2f p90 %.2f'%(st.median(w),sorted(w)[len(w)//10],sorted(w)[9*len(w)//10]))
both=[d for d in W.values() if 0 in d and 1 in d]
later=collections.Counter(1 if d[1][1]>d[0][1] else 0 for d in both)
span=[(max(d[0][1],d[1][1])-min(d[0][0],d[1][0]))*1000 for d in both]
print('  rooms with both ranks',len(both),'later-finishing rank counts',dict(later),'span median %.2f p90 %.2f'%(st.median(span),sorted(span)[9*len(span)//10]))
