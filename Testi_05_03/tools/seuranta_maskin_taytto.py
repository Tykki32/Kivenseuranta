import re,csv,glob,os,sys,numpy as np,cv2,pickle
from collections import defaultdict
sys.path.insert(0,'/home/user/Kivenseuranta/Testi_03_04/tools'); sys.path.insert(0,'/home/user/Kivenseuranta/Testi_03_04')
from seuranta_kokeilu import TrackLab
import haku_silhouette as HS
pat=re.compile(r"\[SILDBG\] f=(\d+) id=(\d+) tarkka=(\w+) X,Y=\(([-\d.]+),([-\d.]+)\) -> \(([-\d.]+),([-\d.]+)\) ok=(\w+) in0=([\d.eE-]+|None)")
cnt=defaultdict(int)
for r in csv.DictReader(open('/tmp/w/full06/stones_raaka.csv')): cnt[int(r['stone_id'])]+=1
real={k for k,v in cnt.items() if v>=200}
# keep only frames that belong to a stone's final (accepted) track rows: use full06 raaka rows
acc=set((int(r['stone_id']),int(r['frame'])) for r in csv.DictReader(open('/tmp/w/full06/stones_raaka.csv')))
dumps={}
for d in ('d8m','endedump','startdump','d183'):
    for p in glob.glob(f'/tmp/w/{d}/g*.npz'): dumps.setdefault(int(os.path.basename(p)[1:7]),d)
C=[]
for l in open('/tmp/w/full06d.log'):
    m=pat.match(l)
    if m and m[9]!='None' and int(m[2]) in real and int(m[1]) in dumps and 0.3<=float(m[9])<0.56: C.append((int(m[1]),int(m[2]),float(m[4]),float(m[5]),float(m[9]),dumps[int(m[1])]))
print(len(C))
# choose diverse: by distance bins and dump
sel=[];seen=set()
for c in sorted(C,key=lambda c:c[4]):
    key=(c[1],c[0]//40)
    if key in seen: continue
    seen.add(key); sel.append(c)
import random; random.seed(1); random.shuffle(sel); sel=sorted(sel[:8],key=lambda c:c[0])
lab={d:TrackLab(f'/tmp/w/{d}','/tmp/w/mah_base/calib_profile.pkl') for d in ('d8m','endedump','startdump','d183')}
L0=lab['d8m']; prof=pickle.load(open('/tmp/w/mah_base/calib_profile.pkl','rb'))['profile']
RF=HS.SilhouetteRefiner(L0.M.k9,L0.pose,L0.R_max,L0.H_total,prof['shape_deltas'],prof['handle_r_frac'],max_shift_px=3,k94=None,use_cpp=False)
K,R,t=(np.asarray(L0.pose[k],float) for k in("K","R","t")); R=R.reshape(3,3); t=t.reshape(3)
def proj(x,y,z=0.):
    p=K@(R@np.array([x,y,z])+t); return p[0]/p[2],p[1]/p[2]

def insd(m,x,y,u,v):
    hull,notch=RF._poly(x,y); o=np.array([u-HW,v-HW]); sh=m.shape
    mh=np.zeros(sh,np.uint8);mn=np.zeros(sh,np.uint8)
    cv2.fillPoly(mh,[np.round(hull-o).astype(np.int32)],1);cv2.fillPoly(mn,[np.round(notch-o).astype(np.int32)],1)
    sil=(mh>0)&(mn==0); return (m&sil).sum()/max(1,sil.sum())
Z=10;HW=22;tiles=[]
want=[(68,3317),(79,3916),(72,3612),(183,12772),(61,3153),(79,3921)]
for f,sid,x,y,i0,d in C:
    if (sid,f) not in want: continue
    img=lab[d].frame(f); u,v=proj(x,y,L0.H_total/2);u=int(round(u));v=int(round(v))
    x0,y0=max(0,u-HW-75),max(0,v-HW-75)
    m=RF._granite_mask(np.ascontiguousarray(img[y0:v+HW+75,x0:u+HW+75]))
    mc=cv2.morphologyEx(m,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_RECT,(7,1)))
    cr=lambda a:a[v-HW-y0:v+HW-y0,u-HW-x0:u+HW-x0]>0
    m0,m1=cr(m),cr(mc)
    def draw(mm):
        vv=np.full((2*HW,2*HW,3),255,np.uint8);vv[mm]=(60,60,60);return cv2.resize(vv,None,fx=Z,fy=Z,interpolation=cv2.INTER_NEAREST)
    a=cv2.resize(img[v-HW:v+HW,u-HW:u+HW],None,fx=Z,fy=Z,interpolation=cv2.INTER_NEAREST);b=draw(m0);c=draw(m1)
    added=(m1&~m0); c[cv2.resize(added.astype(np.uint8),None,fx=Z,fy=Z,interpolation=cv2.INTER_NEAREST)>0]=(0,160,0)
    hull,notch=RF._poly(x,y);o=np.array([u-HW,v-HW])
    for im in (a,b,c):
        cv2.polylines(im,[((hull-o+0.5)*Z).astype(np.int32)],True,(0,200,255),2);cv2.polylines(im,[((notch-o+0.5)*Z).astype(np.int32)],True,(0,200,255),1)
    row=np.hstack([a,b,c]);cv2.putText(row,f"id{sid} f{f} y={y/100:.1f}m  ruutu | ENNEN inside={insd(m0,x,y,u,v):.2f} | JALKEEN (vihr.=taytetty) inside={insd(m1,x,y,u,v):.2f}",(4,14),0,0.5,(0,120,255),1);tiles.append(row)
cv2.imwrite('/tmp/w/fill.png',np.vstack(tiles))
