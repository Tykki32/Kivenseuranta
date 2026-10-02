import csv,numpy as np
from collections import defaultdict
def load(p):
    D=defaultdict(list)
    for r in csv.DictReader(open(p)):
        D[int(r['stone_id'])].append((int(r['frame']),float(r['x_m']),float(r['y_m']),int(r['tarkka']),float(r['rms_px'] or 0)))
    return {k:np.array(sorted(v)) for k,v in D.items() if len(v)>=200}
A=load('full04_sil3/stones.csv');B=load('full05/stones.csv')
def sm(a):
    e=a[a[:,2]>26]; f=e[:,0]-e[0,0]
    px=np.polyfit(f,e[:,1],2);py=np.polyfit(f,e[:,2],2)
    return np.sqrt(((e[:,1]-np.polyval(px,f))**2).mean())*100,np.sqrt(((e[:,2]-np.polyval(py,f))**2).mean())*100,e[:,3].mean(),len(e)
ra=[];rb=[]
print('heitto A/B | alku | pituus | sileys X/Y cm A -> B | tarkka>26m | loppu y')
for ka,a in sorted(A.items(),key=lambda kv:kv[1][0,0]):
    m=[(kb,b) for kb,b in B.items() if abs(b[-1,0]-a[-1,0])<30]
    if not m: print(ka,'ei paria B:ssa'); continue
    kb,b=m[0]; sa,sb=sm(a),sm(b); ra.append(sa[:3]);rb.append(sb[:3])
    # loppupää ero: y at last frame
    ca=dict(zip(a[:,0].astype(int),a[:,2]));cb=dict(zip(b[:,0].astype(int),b[:,2]))
    com=[f for f in ca if f in cb]; d=np.abs([ca[f]-cb[f] for f in com])*100
    print(f"{ka:4d}/{kb:4d} f{int(a[0,0])}/{int(b[0,0])} n {len(a)}/{len(b)} X/Y {sa[0]:.1f}/{sa[1]:.1f} -> {sb[0]:.1f}/{sb[1]:.1f} | tarkka {sa[2]:.2f}->{sb[2]:.2f} | yend {a[-1,2]:.1f}/{b[-1,2]:.1f} | dY med {np.median(d):.1f} max {d.max():.0f} cm")
ra=np.array(ra);rb=np.array(rb);print('ka sileys A',ra.mean(0).round(2),'B',rb.mean(0).round(2))
