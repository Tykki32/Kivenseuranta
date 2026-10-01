import sys,numpy as np,cv2,csv
exec(open('/tmp/w/zoom.py').read().split('Z=12')[0])
lab=TrackLab('/tmp/w/d183','/tmp/w/mah_base/calib_profile.pkl')
def rd(p,sid): return {int(r['frame']):(float(r['x_m'])*100,float(r['y_m'])*100,int(r['tarkka'])) for r in csv.DictReader(open(p)) if int(r['stone_id'])==sid}
A=rd('/tmp/w/full04_sil3/stones_raaka.csv',181); B=rd('/tmp/w/full05/stones_raaka.csv',183)
# frames 12709-12711 from debug log (raaka may lack)
Z=8;HW=30
def tile(f):
    img=lab.frame(f); ax,ay,_=A[f]; u,v=proj(ax,ay,lab.H_total/2); u=int(round(u));v=int(round(v))
    big=cv2.resize(img[v-HW:v+HW,u-HW:u+HW],None,fx=Z,fy=Z,interpolation=cv2.INTER_NEAREST); o=np.array([u-HW,v-HW])
    cv2.polylines(big,[((hull(ax,ay)-o)*Z).astype(np.int32)],True,(0,255,0),2)
    s=f"f{f} v4.3 y={ay/100:.2f} (vihr.)"
    if f in B:
        bx,by,_=B[f]; cv2.polylines(big,[((hull(bx,by)-o)*Z).astype(np.int32)],True,(0,0,255),2); s+=f" v4.4 y={by/100:.2f} (pun.)"
    else: s+=" v4.4: ei rivia"
    cv2.putText(big,s,(4,14),0,0.45,(0,160,255),1); return big
fr=[12700,12704,12706,12708,12710,12712,12720,12735,12748,12759]
fr=[f for f in fr if lab.has(f) and f in A]
t=[tile(f) for f in fr]
rows=[np.hstack(t[i:i+5]) for i in range(0,len(t),5)]
w=max(r.shape[1] for r in rows); rows=[np.pad(r,((0,0),(0,w-r.shape[1]),(0,0))) for r in rows]
cv2.imwrite('/tmp/w/v183.png',np.vstack(rows)); print(fr,[f in B for f in fr])
