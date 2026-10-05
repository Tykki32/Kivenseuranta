import sys,types,numpy as np,pickle,cv2,csv
for n in ("tkinter","tkinter.filedialog"): sys.modules[n]=types.ModuleType(n)
sys.modules["tkinter"].filedialog=sys.modules["tkinter.filedialog"]
sys.path.insert(0,'/home/user/Kivenseuranta/Testi_03_04'); sys.path.insert(0,'/home/user/Kivenseuranta/Testi_03_04/tools')
import main as M
from seuranta_kokeilu import TrackLab
import haku_silhouette as HS
lab=TrackLab('/tmp/w/dnear','/tmp/w/mah_base/calib_profile.pkl'); ref=lab.ref_img
prof=pickle.load(open('/tmp/w/mah_base/calib_profile.pkl','rb'))['profile']
RF=HS.SilhouetteRefiner(lab.M.k9,lab.pose,lab.R_max,lab.H_total,prof['shape_deltas'],prof['handle_r_frac'],max_shift_px=3,k94=None,use_cpp=False)
K,R,t=(np.asarray(lab.pose[k],float) for k in("K","R","t")); R=R.reshape(3,3); t=t.reshape(3)
def proj(x,y,z=0.):
    p=K@(R@np.array([x,y,z])+t); return p[0]/p[2],p[1]/p[2]
P={int(r['frame']):(float(r['x_m'])*100,float(r['y_m'])*100) for r in csv.DictReader(open('/tmp/w/full10/stones_raaka.csv')) if int(r['stone_id'])==161}
Z=5;HW=36;tiles=[]
for f in (10020,10051,10088,10134,10177,10190):
    x,y=P[f]; u,v=proj(x,y,lab.H_total/2);u=int(round(u));v=int(round(v)); raw=np.load(f'/tmp/w/dnear/g{f:06d}.npz')['raw']
    outs=[raw]
    for gate,both in ((False,False),(True,False),(True,True)):
        M.COLOR_GATE=gate; M.COLOR_GATE_BOTH=False; M.COLOR_GATE_HUE_SAT=(both); outs.append(M.suppress_static_background(raw,ref,diff_threshold=M.GRANITE_DIFF_THRESHOLD))
    sl=(slice(v-HW,v+HW),slice(u-HW,u+HW)); o=np.array([u-HW,v-HW]); hull,_=RF._poly(x,y)
    ims=[cv2.resize(i[sl],None,fx=Z,fy=Z,interpolation=cv2.INTER_NEAREST) for i in outs]
    for im in ims: cv2.polylines(im,[((hull-o+0.5)*Z).astype(np.int32)],True,(0,200,255),1)
    row=np.hstack(ims); cv2.putText(row,f"f{f} y={y/100:.2f}m   raaka | ei porttia | S_ref<60 tai H5 | S_ref<60 tai (H5 ja S_ruutu>60)",(4,14),0,0.5,(0,0,255),1); tiles.append(row)
a=np.vstack(tiles[:3]);b=np.vstack(tiles[3:]); cv2.imwrite('/tmp/w/near3.png',np.hstack([a,b])); print(np.hstack([a,b]).shape)
