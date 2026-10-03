import sys,numpy as np,pickle,cv2
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
pk=pickle.load(open('/tmp/w/mah_base/calib_profile.pkl','rb')); ref=pk['calib']['frame_undistorted']; pose=pk['pose']
K,R,t=(np.asarray(pose[k],float) for k in("K","R","t")); R=R.reshape(3,3); t=t.reshape(3)
def proj(x,y,z=0.):
    p=K@(R@np.array([x,y,z])+t); return p[0]/p[2],p[1]/p[2]
# px per cm paikallisesti (y=100..500 cm)
for yy in (100,300,500):
    a=np.array(proj(0,yy)); b=np.array(proj(5,yy)); c=np.array(proj(0,yy+5))
    print('Y=%d cm: 5 cm sivusuunnassa = %.1f px, 5 cm syvyyssuunnassa = %.1f px'%(yy,np.linalg.norm(b-a),np.linalg.norm(c-a)))
# ROI: maa-alue X +-150, Y 0..600 cm
poly=np.array([proj(x,y) for x,y in ((-150,0),(150,0),(150,600),(-150,600))]).astype(np.int32)
roi=np.zeros(ref.shape[:2],np.uint8); cv2.fillPoly(roi,[poly],1)
hsv=cv2.cvtColor(ref,cv2.COLOR_BGR2HSV).astype(int); H,S,V=hsv[...,0],hsv[...,1],hsv[...,2]
blue=((H>=85)&(H<=130)&(S>=25)&(roi>0)).astype(np.uint8)
blue=cv2.morphologyEx(blue,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
n,lab,stats,_=cv2.connectedComponentsWithStats(blue,8)
keep=np.zeros_like(blue)
for i in range(1,n):
    if stats[i,cv2.CC_STAT_AREA]>=400: keep[lab==i]=1
print('sininen alue px',int(keep.sum()),'komponentteja',sum(1 for i in range(1,n) if stats[i,cv2.CC_STAT_AREA]>=400))
# 5 cm reunamarginaali: pikselimäärä paikallinen ~ 5 cm lateraalisesti; käytetään r px = 5 cm @ Y=300
r=int(round(np.linalg.norm(np.array(proj(5,300))-np.array(proj(0,300)))))
print('eroosio r =',r,'px (5 cm)')
ker=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*r+1,2*r+1))
inner=cv2.erode(keep,ker)
print('sisaalue px',int(inner.sum()))
Sb_all=S[keep>0]; Sb_in=S[inner>0]
for nm,a in (('koko sininen',Sb_all),('5 cm reunat pois',Sb_in)):
    print(nm,'n=%d'%len(a),'S pctl 0.1/1/5/25/50/75/95:',np.percentile(a,[0.1,1,5,25,50,75,95]).round(0),' <60: %.2f%%'%(100*(a<60).mean()),' <80: %.2f%%'%(100*(a<80).mean()))
ice=(S<40)&(roi>0)&(cv2.dilate(keep,ker)==0)
fig,ax=plt.subplots(1,3,figsize=(16,4.3))
b=np.arange(0,256,4)
ax[0].hist(Sb_all,bins=b,color='#8899cc'); ax[0].axvline(60,color='r',ls='--'); ax[0].set_title('Moodikuva S: koko sininen alue (pesa, Y 0-6 m)'); ax[0].set_xlabel('S')
ax[1].hist(Sb_in,bins=b,color='#2255aa'); ax[1].axvline(60,color='r',ls='--',label='raja 60'); ax[1].set_title('Moodikuva S: sininen, reunat 5 cm pois (n=%d)'%len(Sb_in)); ax[1].set_xlabel('S'); ax[1].legend()
ax[2].hist(S[ice],bins=b,color='#999999'); ax[2].axvline(60,color='r',ls='--'); ax[2].set_title('Vertailu: jaa (S<40) pesan alueella, ei sinista 5 cm lahella'); ax[2].set_xlabel('S')
for a in ax: a.set_yscale('log')
plt.tight_layout(); plt.savefig('/tmp/w/hist2.png',dpi=90)
vis=ref.copy(); vis[keep>0]=(0,160,255); vis[inner>0]=(255,80,0)
x0,x1=poly[:,0].min()-20,poly[:,0].max()+20; y0,y1=poly[:,1].min()-20,poly[:,1].max()+20
cv2.imwrite('/tmp/w/hist2_mask.png',cv2.resize(np.vstack([ref[y0:y1,x0:x1],vis[y0:y1,x0:x1]]),None,fx=2,fy=2,interpolation=cv2.INTER_NEAREST)); print(x0,x1,y0,y1)
