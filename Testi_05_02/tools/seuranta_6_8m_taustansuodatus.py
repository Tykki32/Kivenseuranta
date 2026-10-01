import sys,numpy as np,cv2,csv,pickle
sys.path.insert(0,'/home/user/Kivenseuranta/Testi_03_04/tools'); sys.path.insert(0,'/home/user/Kivenseuranta/Testi_03_04')
from seuranta_kokeilu import TrackLab
import haku_silhouette as HS
lab=TrackLab('/tmp/w/d8m','/tmp/w/mah_base/calib_profile.pkl')
prof=pickle.load(open('/tmp/w/mah_base/calib_profile.pkl','rb'))['profile']; k9=lab.M.k9
RF=HS.SilhouetteRefiner(k9,lab.pose,lab.R_max,lab.H_total,prof['shape_deltas'],prof['handle_r_frac'],max_shift_px=3,k94=None,use_cpp=False)
K,R,t=(np.asarray(lab.pose[k],float) for k in("K","R","t")); R=R.reshape(3,3); t=t.reshape(3)
def proj(x,y,z=0.):
    p=K@(R@np.array([x,y,z])+t); return p[0]/p[2],p[1]/p[2]
def rd(sid): return {int(r['frame']):(float(r['x_m'])*100,float(r['y_m'])*100) for r in csv.DictReader(open('/tmp/w/full05/stones_raaka.csv')) if int(r['stone_id'])==sid}
T={43:rd(43),61:rd(61)}
Z=10;HW=22
def raw(f): return np.load(f'/tmp/w/d8m/g{f:06d}.npz')['raw']
def sil_mask(x,y,shape,ox,oy):
    hull,notch=RF._poly(x,y)
    mh=np.zeros(shape,np.uint8);mn=np.zeros(shape,np.uint8)
    cv2.fillPoly(mh,[np.round(hull-[ox,oy]).astype(np.int32)],1);cv2.fillPoly(mn,[np.round(notch-[ox,oy]).astype(np.int32)],1)
    return (mh>0)&(mn==0)
tiles=[];stats=[]
for sid,fr in ((43,range(2200,2260,6)),(61,range(3140,3190,5))):
    for f in fr:
        if f not in T[sid] or not lab.has(f): continue
        x,y=T[sid][f]; u,v=proj(x,y,lab.H_total/2);u=int(round(u));v=int(round(v))
        rw=raw(f);sp=lab.frame(f)
        cr=rw[v-HW:v+HW,u-HW:u+HW];cs=sp[v-HW:v+HW,u-HW:u+HW]
        sm=sil_mask(x,y,cr.shape[:2],u-HW,v-HW)
        gray=cv2.cvtColor(cr,cv2.COLOR_BGR2GRAY)
        white=(cs.min(axis=2)>=245)
        # stone-like in raw: sat low & darker than local ice; use gray<150
        stone_like=(gray<150)
        removed=sm&stone_like&white
        inside=sm.sum(); stats.append((sid,f,round(y/100,1),int(inside),int((sm&stone_like).sum()),int(removed.sum()),int((sm&~white).sum())))
        big=lambda a:cv2.resize(a,None,fx=Z,fy=Z,interpolation=cv2.INTER_NEAREST)
        a=big(cr);b=big(cs);c=b.copy()
        c[big(removed.astype(np.uint8))>0]=(0,0,255)
        for im in (a,b,c):
            hull,_=RF._poly(x,y); cv2.polylines(im,[((hull-[u-HW,v-HW]+0.5)*Z).astype(np.int32)],True,(0,255,0),1)
        row=np.hstack([a,b,c]); cv2.putText(row,f"{sid} f{f} y={y/100:.1f}m raaka | vaimennettu | poistunut(pun.)",(4,14),0,0.5,(0,120,255),1); tiles.append(row)
cv2.imwrite('/tmp/w/m8.png',np.vstack(tiles[:8]))
for s in stats: print(s)
print('--- sisaosuus: maski vs ei-valkoinen vs keltainen (sat>=60) siluetin sisalla')
for sid,fr in ((43,[2200,2212,2230,2248]),(61,[3145,3160])):
    for f in fr:
        x,y=T[sid][f]; u,v=proj(x,y,lab.H_total/2);u=int(round(u));v=int(round(v))
        sp=lab.frame(f); cs=sp[v-HW:v+HW,u-HW:u+HW]; sm=sil_mask(x,y,cs.shape[:2],u-HW,v-HW)
        x0,y0=max(0,u-HW-75),max(0,v-HW-75)
        m=RF._granite_mask(np.ascontiguousarray(sp[y0:v+HW+75,x0:u+HW+75]))[v-HW-y0:v+HW-y0,u-HW-x0:u+HW-x0]>0
        hsv=cv2.cvtColor(cs,cv2.COLOR_BGR2HSV); nonwhite=cs.min(axis=2)<245; yellow=nonwhite&(hsv[...,1]>=60)
        n=sm.sum(); print(sid,f,'y=%.1f'%(y/100),'maski %.2f  ei-valkoinen %.2f  keltainen(sat>=60) %.2f  ei-valk&ei-maski %.2f'%((m&sm).sum()/n,(nonwhite&sm).sum()/n,(yellow&sm).sum()/n,(nonwhite&~m&sm).sum()/n))
