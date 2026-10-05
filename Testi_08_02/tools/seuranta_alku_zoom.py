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


Z=12; HW=18
def zoom(sid,frames,out):
    rows={r[0]:r for r in lab.rows[str(sid)]}
    tiles=[]
    for f in frames:
        img=lab.frame(f); _,x,y,rms=rows[f]
        u,v=proj(x,y,lab.H_total/2); u=int(round(u)); v=int(round(v))
        c=img[v-HW:v+HW,u-HW:u+HW].copy()
        big=cv2.resize(c,None,fx=Z,fy=Z,interpolation=cv2.INTER_NEAREST)
        ov=big.copy()
        h=(hull(x,y)-np.array([u-HW,v-HW]))*Z
        cv2.polylines(ov,[h.astype(np.int32)],True,(0,255,0) if rms<2 else (0,0,255),2)
        t=np.hstack([big,ov]); cv2.putText(t,f"{sid} f{f} y={y/100:.1f}m rms={rms:.0f}",(5,18),0,0.6,(0,160,255),2)
        tiles.append(t)
    cv2.imwrite(out,np.vstack(tiles))
zoom(40,[1790,1808,1832,1844],'/tmp/w/zoom_40.png')
zoom(47,[2240,2262,2300,2330],'/tmp/w/zoom_47.png')
zoom(70,[3220,3248,3290,3304],'/tmp/w/zoom_70.png')
