import sys,numpy as np,cv2,csv
exec(open('/tmp/w/mode.py').read().split("Z=10;HW=22")[0])
import importlib; M=lab.M
print('GRANITE_DIFF_THRESHOLD',M.GRANITE_DIFF_THRESHOLD,'V drop',M.SHADOW_V_DROP_MIN,M.SHADOW_V_DROP_MAX,'ice',M.ICE_S_MAX,M.ICE_V_MIN)
f=2200; x,y=T[43][f]; u,v=proj(x,y,lab.H_total/2);u=int(round(u));v=int(round(v))
z=np.load(f'/tmp/w/d8m/g{f:06d}.npz'); raw,sup=z['raw'],z['frame']
# raw is raw; "frame_u_photo" includes photometric correction; we approximate with raw
HW=22; sl=(slice(v-HW,v+HW),slice(u-HW,u+HW))
r=raw[sl].astype(np.int16); ref=lab.ref_img[sl].astype(np.int16); s=sup[sl]
hr=cv2.cvtColor(raw[sl],cv2.COLOR_BGR2HSV).astype(int); hf=cv2.cvtColor(lab.ref_img[sl],cv2.COLOR_BGR2HSV).astype(int)
diff=cv2.cvtColor(np.abs(r-ref).astype(np.uint8),cv2.COLOR_BGR2GRAY)
vd=hf[...,2]-hr[...,2]
bg=(diff<M.GRANITE_DIFF_THRESHOLD); sh=(vd>M.SHADOW_V_DROP_MIN)&(vd<M.SHADOW_V_DROP_MAX); ice=(hr[...,1]<M.ICE_S_MAX)&(hr[...,2]>M.ICE_V_MIN)
white=(s.min(axis=2)>=245)
# stone-like raw pixels removed: dark raw (gray<150) inside rough stone window x in [u-HW+... ] use silhouette
hull,notch=RF._poly(x,y);o=np.array([u-HW,v-HW]);mh=np.zeros(white.shape,np.uint8);cv2.fillPoly(mh,[np.round(hull-o).astype(np.int32)],1)
gray=cv2.cvtColor(raw[sl],cv2.COLOR_BGR2GRAY)
rem=(mh>0)&white&(gray<150)
print('poistetut kivennakoiset pikselit',int(rem.sum()))
print('kriteerit (osuus poistetuista): erotus<kynnys %.2f | varjo(V-pudotus -3..50) %.2f | jaa(S<22&V>128) %.2f'%(bg[rem].mean(),sh[rem].mean(),ice[rem].mean()))
print('vain erotus %.2f vain varjo %.2f vain jaa %.2f'%((bg&~sh&~ice)[rem].mean(),(sh&~bg&~ice)[rem].mean(),(ice&~bg&~sh)[rem].mean()))
idx=np.argwhere(rem)[::max(1,rem.sum()//8)]
print('esimerkkipikselit: raaka HSV | moodi HSV | diff_gray | V-pudotus')
for yy,xx in idx[:8]: print(hr[yy,xx],hf[yy,xx],diff[yy,xx],vd[yy,xx], 'bg' if bg[yy,xx] else '', 'varjo' if sh[yy,xx] else '', 'jaa' if ice[yy,xx] else '')
# stone pixels that stayed: same stats
keep=(mh>0)&~white&(gray<150)
print('sailyneet kivipikselit n=%d: raaka V med %.0f S med %.0f | moodi V %.0f S %.0f | diff med %.0f | V-pudotus med %.0f'%(keep.sum(),np.median(hr[...,2][keep]),np.median(hr[...,1][keep]),np.median(hf[...,2][keep]),np.median(hf[...,1][keep]),np.median(diff[keep]),np.median(vd[keep])))
print('poistetut: raaka V med %.0f S med %.0f | moodi V %.0f S %.0f | diff med %.0f | V-pudotus med %.0f'%(np.median(hr[...,2][rem]),np.median(hr[...,1][rem]),np.median(hf[...,2][rem]),np.median(hf[...,1][rem]),np.median(diff[rem]),np.median(vd[rem])))
