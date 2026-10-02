import sys,numpy as np,cv2,pickle,csv
exec(open('/tmp/w/zoom.py').read().split('Z=12')[0])
import haku_silhouette as HS
k9=lab.M.k9
prof=pickle.load(open('/tmp/w/mah_base/calib_profile.pkl','rb'))['profile']
RF=HS.SilhouetteRefiner(k9,lab.pose,lab.R_max,lab.H_total,prof['shape_deltas'],prof['handle_r_frac'],max_shift_px=3,k94=None,use_cpp=False)
D={}
for r in csv.DictReader(open('/tmp/w/full05/stones.csv')):
    D.setdefault(int(r['stone_id']),[]).append((int(r['frame']),float(r['x_m'])*100,float(r['y_m'])*100,int(r['tarkka']),float(r['rms_px'] or 0)))
Z=12;HW=18
def tile(f,x,y,tk,rms):
    img=lab.frame(f); u,v=proj(x,y,lab.H_total/2);u=int(round(u));v=int(round(v))
    x0,y0=max(0,u-HW-75),max(0,v-HW-75)
    m=RF._granite_mask(np.ascontiguousarray(img[y0:v+HW+75,x0:u+HW+75]))[v-HW-y0:v+HW-y0,u-HW-x0:u+HW-x0]>0
    big=cv2.resize(img[v-HW:v+HW,u-HW:u+HW],None,fx=Z,fy=Z,interpolation=cv2.INTER_NEAREST)
    mv=np.full((2*HW,2*HW,3),255,np.uint8);mv[m]=(60,60,60);mv=cv2.resize(mv,None,fx=Z,fy=Z,interpolation=cv2.INTER_NEAREST)
    o=np.array([u-HW,v-HW]); a=RF.refine_py(img,x,y)[2]
    for t in (big,mv):
        hull,notch=RF._poly(x,y); cv2.polylines(t,[((hull-o+0.5)*Z).astype(np.int32)],True,(0,200,255),2); cv2.polylines(t,[((notch-o+0.5)*Z).astype(np.int32)],True,(0,200,255),1)
    out=np.hstack([big,mv]); cv2.putText(out,f"f{f} y={y/100:.1f}m tarkka={tk} inside={a.get('inside0',0):.2f}",(5,16),0,0.5,(0,120,255),1); return out
for sid,lo,hi in [(50,2240,2330),(72,3220,3330),(125,6110,6150),(121,5900,6000)]:
    a=np.array(sorted(D[sid])); e=a[a[:,2]>2000]; f=e[:,0]-e[0,0]
    ry=e[:,2]-np.polyval(np.polyfit(f,e[:,2],3),f); rx=e[:,1]-np.polyval(np.polyfit(f,e[:,1],3),f)
    sel=[i for i in range(len(e)) if lo<=e[i,0]<=hi and lab.has(int(e[i,0]))]
    worst=sorted(sel,key=lambda i:-(abs(rx[i])/1.0+abs(ry[i])/4.0))[:3]
    tiles=[tile(int(e[i,0]),e[i,1],e[i,2],int(e[i,3]),e[i,4]) for i in sorted(worst)]
    print(sid,[(int(e[i,0]),round(rx[i],1),round(ry[i],1),int(e[i,3])) for i in sorted(worst)])
    cv2.imwrite(f'/tmp/w/v5_{sid}.png',np.vstack(tiles))
