import sys,numpy as np,pickle,csv,cv2,importlib.util
exec(open('/tmp/w/zoom.py').read().split('Z=12')[0])
spec=importlib.util.spec_from_file_location('hs_band','/tmp/w/hs_band.py');HS=importlib.util.module_from_spec(spec);spec.loader.exec_module(HS)
k9=lab.M.k9; orig=k9.create_granite_mask
def mk(ok):
    def f(frame):
        hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV); g=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY).astype(np.float32)
        bg=cv2.GaussianBlur(g,(0,0),sigmaX=k9.STONE_DARKNESS_SIGMA)
        m=(((hsv[...,1]<k9.STONE_MAX_SATURATION)&((bg-g)>k9.STONE_MIN_DARKNESS))).astype(np.uint8)*255
        m=cv2.morphologyEx(m,cv2.MORPH_OPEN,np.ones((ok,ok),np.uint8)); return cv2.morphologyEx(m,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    return f
prof=pickle.load(open('/tmp/w/mah_base/calib_profile.pkl','rb'))['profile']
tk={}
for r in csv.DictReader(open('/tmp/w/full04_sil3/stones.csv')): tk[(int(r['stone_id']),int(r['frame']))]=int(r['tarkka'])
def resid(a,f0):
    px=np.polyfit(a[:,0]-f0,a[:,1],2);py=np.polyfit(a[:,0]-f0,a[:,2],2)
    return np.sqrt(((a[:,1]-np.polyval(px,a[:,0]-f0))**2).mean()),np.sqrt(((a[:,2]-np.polyval(py,a[:,0]-f0))**2).mean())
for sid,f0,f1 in [(40,1790,1880),(47,2240,2330),(70,3220,3330)]:
    rows={r[0]:r for r in lab.rows[str(sid)]}; fr=[f for f in range(f0,f1+1) if f in rows and rows[f][2]>2600]
    base=np.array([[f,rows[f][1],rows[f][2]] for f in fr]); rx,ry=resid(base,f0)
    print(f"{sid} CSV sileys {rx:.2f}/{ry:.2f}")
    for name,ok in [('5x5 (nyk.)',5),('3x3',3)]:
        k9.create_granite_mask=mk(ok); R_=HS.SilhouetteRefiner(k9,lab.pose,lab.R_max,lab.H_total,prof['shape_deltas'],prof['handle_r_frac'],max_shift_px=3,k94=None,use_cpp=False)
        out=[];ins=[];sh=[]
        for f in fr:
            X1,Y1,i=R_.refine_py(lab.frame(f),rows[f][1],rows[f][2]); ok_=i.get('ok') and i['inside0']>=0.4
            out.append([f,X1 if ok_ else rows[f][1],Y1 if ok_ else rows[f][2]]); ins.append(i.get('inside1',0)); sh.append(i.get('shift_px',0) if ok_ else 0)
        out=np.array(out); rx,ry=resid(out,f0); ins=np.array(ins)
        print(f"   kaista+{name:10s} sileys {rx:.2f}/{ry:.2f} inside1 med {np.median(ins):.2f} <0.4: {int((ins<0.4).sum())}/{len(ins)} siirto med {np.median(sh):.2f}px max {max(sh):.1f}")
    k9.create_granite_mask=orig
