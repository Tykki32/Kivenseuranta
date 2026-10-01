import csv, re, numpy as np, collections, sys
d=sys.argv[1]; logf=sys.argv[2]; endf=int(sys.argv[3])
def fl(x):
    try: return float(x)
    except: return np.nan
raw=collections.defaultdict(list)
for r in csv.DictReader(open(f'{d}/stones_raaka.csv')):
    raw[int(r['stone_id'])].append((int(r['frame']),fl(r['x_m'])*100,fl(r['y_m'])*100,int(r['tarkka']),int(r['n_runkopistetta']),fl(r['rms_px'])))
fin=set(int(r['stone_id']) for r in csv.DictReader(open(f'{d}/stones.csv')))
start={};fate={}
for l in open(logf):
    m=re.match(r"\[frame (\d+)\] Uusi kivi-ehdokas (\d+):",l)
    if m: start[int(m.group(2))]=int(m.group(1)); continue
    m=re.match(r"\[frame (\d+)\] (?:Ehdokas|Kivi|Rata) (\d+) (?:hylatty|kadotettu|pysahtynyt|yhdistetty)",l)
    if m: fate[int(m.group(2))]=int(m.group(1))
dur={k:(fate.get(k,endf)-s) for k,s in start.items()}
tot=sum(dur.values()); real_cost=sum(dur[k] for k in dur if k in fin)
print(f"ratoja {len(dur)}, ruutuja yht {tot}, oikeat heitot {sorted(fin)} {real_cost} ruutua ({100*real_cost/tot:.0f}%), roskaa {tot-real_cost}")
def state(v,n):
    v=sorted(v)[:n]
    if len(v)<n: return None
    f=np.array([q[0] for q in v]); y=np.array([q[2] for q in v]); x=np.array([q[1] for q in v])
    return dict(dy=(y[-1]-y[0])/max(1,f[-1]-f[0]), rms=np.nanmedian([q[5] for q in v]), tk=np.mean([q[3] for q in v]), xabs=np.abs(x).max())
rules={
 "A: dy>-2 cm/ruutu (ei liiku kohti)": lambda s: s['dy']>-2.0,
 "B: rms>14": lambda s: s['rms']>14.0,
 "C: rms>14 ja tarkka=0": lambda s: s['rms']>14.0 and s['tk']==0,
 "D: dy>-2 tai rms>14": lambda s: s['dy']>-2.0 or s['rms']>14.0,
 "E: dy>-2 tai (rms>12 ja tarkka=0)": lambda s: s['dy']>-2.0 or (s['rms']>12.0 and s['tk']==0),
}
for n in (8,12,20,30):
    print(f"--- paatos {n} rivin jalkeen")
    for name,fn in rules.items():
        saved=0; killed_real=[]; kills=0
        for k,v in raw.items():
            s=state(v,n)
            if s is None: continue
            if fn(s):
                if k in fin: killed_real.append(k)
                else: kills+=1; saved+=max(0,dur.get(k,0)-n)
        print(f"  {name:40s} hylkaa roskaratoja {kills:3d}, saastaa {saved:5d} ruutua ({100*saved/max(1,tot-real_cost):4.1f}% roskasta), oikeita kaadettu {killed_real}")
