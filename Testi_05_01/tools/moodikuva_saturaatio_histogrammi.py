import sys,types,numpy as np,pickle,cv2,csv
for n in ("tkinter","tkinter.filedialog"): sys.modules[n]=types.ModuleType(n)
sys.modules["tkinter"].filedialog=sys.modules["tkinter.filedialog"]
sys.path.insert(0,'/home/user/Kivenseuranta/Testi_03_04'); sys.path.insert(0,'/home/user/Kivenseuranta/Testi_03_04/tools')
import main as M
from seuranta_kokeilu import TrackLab
import haku_silhouette as HS
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
lab=TrackLab('/tmp/w/dnear','/tmp/w/mah_base/calib_profile.pkl'); ref=lab.ref_img
prof=pickle.load(open('/tmp/w/mah_base/calib_profile.pkl','rb'))['profile']
RF=HS.SilhouetteRefiner(lab.M.k9,lab.pose,lab.R_max,lab.H_total,prof['shape_deltas'],prof['handle_r_frac'],max_shift_px=3,k94=None,use_cpp=False)
K,R,t=(np.asarray(lab.pose[k],float) for k in("K","R","t")); R=R.reshape(3,3); t=t.reshape(3)
def proj(x,y,z=0.):
    p=K@(R@np.array([x,y,z])+t); return p[0]/p[2],p[1]/p[2]
f=10088
x,y=[(float(r['x_m'])*100,float(r['y_m'])*100) for r in csv.DictReader(open('/tmp/w/full10/stones_raaka.csv')) if int(r['stone_id'])==161 and int(r['frame'])==f][0]
u,v=proj(x,y,lab.H_total/2);u=int(round(u));v=int(round(v))
hf=cv2.cvtColor(ref,cv2.COLOR_BGR2HSV).astype(int)
HW=45; sl=(slice(v-HW,v+HW),slice(u-HW,u+HW))
S=hf[sl][...,1]; H=hf[sl][...,0]; V=hf[sl][...,2]
raw=np.load(f'/tmp/w/dnear/g{f:06d}.npz')['raw']; hr=cv2.cvtColor(raw,cv2.COLOR_BGR2HSV).astype(int)
hull,_=RF._poly(x,y); mh=np.zeros((2*HW,2*HW),np.uint8); cv2.fillPoly(mh,[np.round(hull-[u-HW,v-HW]).astype(np.int32)],1)
# poistetut kivipikselit S_ref-portilla
diff=cv2.cvtColor(cv2.absdiff(raw,ref),cv2.COLOR_BGR2GRAY); vd=hf[...,2]-hr[...,2]
bg=(diff<10)|((vd>-3)&(vd<50))|((hr[...,1]<22)&(hr[...,2]>128))
dh=np.abs(hr[...,0]-hf[...,0]); dh=np.minimum(dh,180-dh)
gate=bg&((hf[...,1]<60)|(dh<=5))
gray=cv2.cvtColor(raw,cv2.COLOR_BGR2GRAY); stone=(mh>0)&(gray[sl]<150)
rem=stone&gate[sl]
print('kivipikseleita',int(stone.sum()),'poistettu S_ref-portilla',int(rem.sum()))
print('poistettujen S_ref: pctl 10/25/50/75/90',np.percentile(S[rem],[10,25,50,75,90]).round(0),' |dH| med',np.median(dh[sl][rem]),'dH<=5:',int((dh[sl][rem]<=5).sum()),'S_ref<60:',int((S[rem]<60).sum()))
print('ref H med poistetut',np.median(H[rem]),'raaka H med',np.median(hr[sl][...,0][rem]))
fig,ax=plt.subplots(1,3,figsize=(15,4.2))
ax[0].hist(S.ravel(),bins=np.arange(0,260,4),color='#4477aa'); ax[0].axvline(60,color='r',ls='--',label='S=60'); ax[0].set_title('Moodikuva, S: koko ruutu kiven ymparilla (90x90 px, f10088, y=3 m)'); ax[0].set_xlabel('S (0-255)'); ax[0].set_ylabel('pikselit'); ax[0].legend()
blue=(H>=85)&(H<=125)&(S>=40)
ax[1].hist(S[blue],bins=np.arange(0,260,4),color='#2255aa'); ax[1].axvline(60,color='r',ls='--'); ax[1].set_title('Moodikuva, S: vain sininen (H 85-125, S>=40) alue'); ax[1].set_xlabel('S')
ax[2].hist(S[rem],bins=np.arange(0,260,4),color='#cc6633'); ax[2].axvline(60,color='r',ls='--'); ax[2].set_title('Moodikuva S niissa kivipikseleissa jotka S_ref-portti viela poistaa (n=%d)'%rem.sum()); ax[2].set_xlabel('S')
plt.tight_layout(); plt.savefig('/tmp/w/hist.png',dpi=90)
# kuva: S_ref kartta + poistetut
Sv=cv2.applyColorMap(np.clip(S,0,255).astype(np.uint8),cv2.COLORMAP_JET); Sv=cv2.resize(Sv,None,fx=6,fy=6,interpolation=cv2.INTER_NEAREST)
rv=raw[sl].copy(); rv[rem]=(0,0,255); rv=cv2.resize(rv,None,fx=6,fy=6,interpolation=cv2.INTER_NEAREST)
rw=cv2.resize(raw[sl],None,fx=6,fy=6,interpolation=cv2.INTER_NEAREST)
o=np.array([u-HW,v-HW])
for im in (Sv,rv,rw): cv2.polylines(im,[((hull-o+0.5)*6).astype(np.int32)],True,(0,255,255),1)
cv2.imwrite('/tmp/w/hist_img.png',np.hstack([rw,Sv,rv]))
