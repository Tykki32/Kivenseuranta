import sys,numpy as np,cv2,csv,pickle
sys.path.insert(0,'/home/user/Kivenseuranta/Testi_03_04/tools'); sys.path.insert(0,'/home/user/Kivenseuranta/Testi_03_04')
from seuranta_kokeilu import TrackLab
import haku_silhouette as HS
lab=TrackLab('/tmp/w/d8m','/tmp/w/mah_base/calib_profile.pkl')
prof=pickle.load(open('/tmp/w/mah_base/calib_profile.pkl','rb'))['profile']
RF=HS.SilhouetteRefiner(lab.M.k9,lab.pose,lab.R_max,lab.H_total,prof['shape_deltas'],prof['handle_r_frac'],max_shift_px=3,k94=None,use_cpp=False)
K,R,t=(np.asarray(lab.pose[k],float) for k in("K","R","t")); R=R.reshape(3,3); t=t.reshape(3)
def proj(x,y,z=0.):
    p=K@(R@np.array([x,y,z])+t); return p[0]/p[2],p[1]/p[2]
ref=lab.ref_img; print('ref',ref.shape,ref.dtype)
def rd(sid): return {int(r['frame']):(float(r['x_m'])*100,float(r['y_m'])*100) for r in csv.DictReader(open('/tmp/w/full05/stones_raaka.csv')) if int(r['stone_id'])==sid}
T={43:rd(43),61:rd(61)}
Z=10;HW=22;tiles=[]
for sid,f in ((43,2200),(43,2206),(61,3153),(61,3150)):
    x,y=T[sid][f]; u,v=proj(x,y,lab.H_total/2);u=int(round(u));v=int(round(v))
    z=np.load(f'/tmp/w/d8m/g{f:06d}.npz'); raw,sup=z['raw'],z['frame']
    sl=(slice(v-HW,v+HW),slice(u-HW,u+HW))
    a=ref[sl];b=raw[sl];c=sup[sl]
    d=np.abs(b.astype(int)-a.astype(int)).max(axis=2); dd=np.clip(d*4,0,255).astype(np.uint8); dd=cv2.cvtColor(dd,cv2.COLOR_GRAY2BGR)
    ims=[cv2.resize(i,None,fx=Z,fy=Z,interpolation=cv2.INTER_NEAREST) for i in (a,b,c,dd)]
    hull,_=RF._poly(x,y);o=np.array([u-HW,v-HW])
    for im in ims: cv2.polylines(im,[((hull-o+0.5)*Z).astype(np.int32)],True,(0,200,255),1)
    row=np.hstack(ims); cv2.putText(row,f"id{sid} f{f} y={y/100:.1f}m | MOODIKUVA (tausta) | raaka ruutu | vaimennettu | ero raaka-moodi (x4)",(4,14),0,0.5,(0,0,255),1); tiles.append(row)
cv2.imwrite('/tmp/w/mode.png',np.vstack(tiles))
