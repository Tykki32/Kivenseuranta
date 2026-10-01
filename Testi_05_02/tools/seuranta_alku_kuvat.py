import sys, csv, numpy as np, cv2, pickle, os
sys.path.insert(0,'/home/user/Kivenseuranta/Testi_03_04/tools')
from seuranta_kokeilu import TrackLab
lab=TrackLab('/tmp/w/startdump','/tmp/w/mah_base/calib_profile.pkl','/tmp/w/full04_sil3/stones.csv')
K,R,t=(np.asarray(lab.pose[k],float) for k in("K","R","t")); R=R.reshape(3,3); t=t.reshape(3)
def proj(x,y,z=0.):
    p=K@(R@np.array([x,y,z])+t); return p[0]/p[2],p[1]/p[2]
def hull(x,y):
    pts=np.array([proj(x+0*0,y,0)],float)
    P=[]
    for p in lab.body:
        pl=np.asarray(p,float)
        # local body pts: (x,y,z) offsets relative to stone centre in cm
        P.append(proj(x+pl[0],y+pl[1],pl[2]))
    P=np.array(P,np.float32); return cv2.convexHull(P).astype(np.int32)
def make(sid,frames,out):
    tiles=[]
    rows={r[0]:r for r in lab.rows[str(sid)]}
    for f in frames:
        if not lab.has(f) or f not in rows: continue
        img=lab.frame(f).copy()
        _,x,y,rms=rows[f]
        u,v=proj(x,y,lab.H_total/2)
        cv2.polylines(img,[hull(x,y)],True,(0,255,0) if rms<2 else (0,0,255),1)
        u=int(u);v=int(v)
        c=img[max(0,v-45):v+45,max(0,u-45):u+45]
        c=cv2.resize(c,(270,270),interpolation=cv2.INTER_CUBIC)
        cv2.putText(c,f"{f} y={y/100:.1f} rms={rms:.0f}",(3,14),0,0.45,(0,255,255),1)
        tiles.append(c)
    if tiles:
        n=len(tiles);cols=min(6,n);rowsn=(n+cols-1)//cols
        H=np.zeros((rowsn*270,cols*270,3),np.uint8)
        for i,c in enumerate(tiles): H[(i//cols)*270:(i//cols+1)*270,(i%cols)*270:(i%cols+1)*270]=c
        cv2.imwrite(out,H)

jobs={'40':range(1790,1880,6),'47':range(2240,2330,6),'70':range(3220,3330,7),'44':range(2000,2090,6)}
for sid,fr in jobs.items(): make(sid,list(fr),f'/tmp/w/start_{sid}.png')
