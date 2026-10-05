import sys,numpy as np,cv2,pickle
exec(open('/tmp/w/zoom.py').read().split('Z=12')[0])
import haku_silhouette as HS
k9=lab.M.k9
prof=pickle.load(open('/tmp/w/mah_base/calib_profile.pkl','rb'))['profile']
orig=k9.create_granite_mask
def mk(open_k):
    def f(frame):
        hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV); g=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY).astype(np.float32)
        bg=cv2.GaussianBlur(g,(0,0),sigmaX=k9.STONE_DARKNESS_SIGMA)
        m=(((hsv[...,1]<k9.STONE_MAX_SATURATION)&((bg-g)>k9.STONE_MIN_DARKNESS))).astype(np.uint8)*255
        if open_k>1: m=cv2.morphologyEx(m,cv2.MORPH_OPEN,np.ones((open_k,)*2,np.uint8))
        return cv2.morphologyEx(m,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    return f
def refiner(shift): return HS.SilhouetteRefiner(k9,lab.pose,lab.R_max,lab.H_total,prof['shape_deltas'],prof['handle_r_frac'],max_shift_px=shift,k94=None,use_cpp=False)
res={}
for sid,f0,f1 in [(40,1790,1880),(47,2240,2330),(70,3220,3330)]:
    rows={r[0]:r for r in lab.rows[str(sid)]}
    fr=[f for f in range(f0,f1+1) if f in rows and rows[f][2]>26e2]
    base=np.array([[f,rows[f][1],rows[f][2]] for f in fr])
    def resid(a):
        px=np.polyfit(a[:,0]-f0,a[:,1],2);py=np.polyfit(a[:,0]-f0,a[:,2],2)
        return np.sqrt(((a[:,1]-np.polyval(px,a[:,0]-f0))**2).mean()),np.sqrt(((a[:,2]-np.polyval(py,a[:,0]-f0))**2).mean())
    print(sid,'CSV-rata',len(fr),'sileys X/Y cm %.2f %.2f'%resid(base))
    for name,ok in [('avaus5(nyk.)',5),('avaus3',3),('ei avausta',1)]:
        k9.create_granite_mask=mk(ok); R=refiner(3)
        out=[];ins=[]
        for f in fr:
            X1,Y1,info=R.refine(lab.frame(f),rows[f][1],rows[f][2]); out.append([f,X1,Y1]); ins.append(info.get('inside1',0))
        out=np.array(out); ins=np.array(ins)
        rx,ry=resid(out); print(f"   {name:13s} sileys X/Y cm {rx:.2f} {ry:.2f}  inside1 med {np.median(ins):.2f}  <0.4: {int((ins<0.4).sum())}/{len(ins)}  siirto med {np.median(np.hypot(out[:,1]-base[:,1],out[:,2]-base[:,2])):.1f}cm")
    k9.create_granite_mask=orig
