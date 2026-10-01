import sys,numpy as np,cv2,pickle
exec(open('/tmp/w/zoom.py').read().split('Z=12')[0])
import haku_silhouette as HS
k9=lab.M.k9
prof=pickle.load(open('/tmp/w/mah_base/calib_profile.pkl','rb'))['profile']
RF=HS.SilhouetteRefiner(k9,lab.pose,lab.R_max,lab.H_total,prof['shape_deltas'],prof['handle_r_frac'],max_shift_px=2,k94=None,use_cpp=False)
Z=12;HW=18
def tile(sid,f):
    rows={r[0]:r for r in lab.rows[str(sid)]}; img=lab.frame(f); _,x,y,rms=rows[f]
    X1,Y1,info=RF.refine_py(img,x,y)
    u,v=proj(x,y,lab.H_total/2);u=int(round(u));v=int(round(v))
    H,W=img.shape[:2]; x0,y0=max(0,u-HW-75),max(0,v-HW-75)
    gm=k9.create_granite_mask(np.ascontiguousarray(img[y0:v+HW+75,x0:u+HW+75]))
    m=gm[v-HW-y0:v+HW-y0,u-HW-x0:u+HW-x0]>0
    crop=img[v-HW:v+HW,u-HW:u+HW]
    big=cv2.resize(crop,None,fx=Z,fy=Z,interpolation=cv2.INTER_NEAREST)
    mv=np.full((2*HW,2*HW,3),255,np.uint8); mv[m]=(60,60,60)
    mv=cv2.resize(mv,None,fx=Z,fy=Z,interpolation=cv2.INTER_NEAREST)
    o=np.array([u-HW,v-HW])
    for t in (big,mv):
        for (xx,yy,col) in ((x,y,(0,0,255)),(X1,Y1,(0,200,255))):
            hull,notch=RF._poly(xx,yy)
            cv2.polylines(t,[((hull-o+0.5)*Z).astype(np.int32)],True,col,2)
            cv2.polylines(t,[((notch-o+0.5)*Z).astype(np.int32)],True,col,1)
    out=np.hstack([big,mv]); d=(np.array(RF.proj(X1,Y1,RF.H_total/2))-np.array(RF.proj(x,y,RF.H_total/2)))
    cv2.putText(out,f"{sid} f{f} tarkka/rms={rms:.0f} shift px du={d[0]:.1f} dv={d[1]:.1f} in0={info.get('inside0',0):.2f} in1={info.get('inside1',0):.2f}",(5,16),0,0.5,(0,120,255),1)
    return out
cv2.imwrite('/tmp/w/silviz_40.png',np.vstack([tile(40,f) for f in (1808,1832,1844)]))
