import sys,numpy as np,cv2,csv
exec(open('/tmp/w/mode.py').read().split("Z=10;HW=22")[0])
M=lab.M
def supp(raw,ref,new=False,ds_max=20,ref_ice_v=160):
    diff=cv2.cvtColor(cv2.absdiff(raw,ref),cv2.COLOR_BGR2GRAY)
    hr=cv2.cvtColor(raw,cv2.COLOR_BGR2HSV).astype(np.int16); hf=cv2.cvtColor(ref,cv2.COLOR_BGR2HSV).astype(np.int16)
    vd=hf[...,2]-hr[...,2]
    sh=(vd>M.SHADOW_V_DROP_MIN)&(vd<M.SHADOW_V_DROP_MAX)
    if new:
        ref_ice=(hf[...,1]<M.ICE_S_MAX)&(hf[...,2]>=ref_ice_v)          # varjotoleranssi vain kohdissa joissa tausta on jaata (ei viivaa/mainosta)
        sh&=ref_ice&(np.abs(hr[...,1]-hf[...,1])<ds_max)                  # varjo ei muuta kylläisyyttä (keltainen kahva != varjo)
    ice=(hr[...,1]<M.ICE_S_MAX)&(hr[...,2]>M.ICE_V_MIN)
    bg=(diff<M.GRANITE_DIFF_THRESHOLD)|sh|ice
    out=raw.copy();out[bg]=(255,255,255);return out
Z=10;HW=22;tiles=[];res=[]
for sid,f in ((43,2200),(43,2206),(61,3153),(61,3150),(43,2230),(43,2248)):
    x,y=T[sid][f]; u,v=proj(x,y,lab.H_total/2);u=int(round(u));v=int(round(v))
    z=np.load(f'/tmp/w/d8m/g{f:06d}.npz'); raw=z['raw']
    sl=(slice(v-HW,v+HW),slice(u-HW,u+HW))
    old=supp(raw,lab.ref_img)[sl]; new=supp(raw,lab.ref_img,True)[sl]
    hull,_=RF._poly(x,y);o=np.array([u-HW,v-HW]);mh=np.zeros(old.shape[:2],np.uint8);cv2.fillPoly(mh,[np.round(hull-o).astype(np.int32)],1)
    gray=cv2.cvtColor(raw[sl],cv2.COLOR_BGR2GRAY); stone=(mh>0)&(gray<150)
    wo=old.min(axis=2)>=245; wn=new.min(axis=2)>=245
    far=cv2.dilate(mh,np.ones((9,9),np.uint8))==0
    res.append((sid,f,int(stone.sum()),int((stone&wo).sum()),int((stone&wn).sum()),int((far&~wo).sum()),int((far&~wn).sum())))
    ims=[cv2.resize(i,None,fx=Z,fy=Z,interpolation=cv2.INTER_NEAREST) for i in (raw[sl],old,new)]
    for im in ims: cv2.polylines(im,[((hull-o+0.5)*Z).astype(np.int32)],True,(0,200,255),1)
    row=np.hstack(ims);cv2.putText(row,f"id{sid} f{f} raaka | nykyinen vaimennus | uusi (varjo vain jaalla, S ei muutu)",(4,14),0,0.5,(0,0,255),1);tiles.append(row)
cv2.imwrite('/tmp/w/crit2.png',np.vstack(tiles))
print('id f | kivipikseleita siluetissa | poistettu nyk. | poistettu uusi | roskaa siluetin ulkopuolella nyk. | uusi')
for r in res: print(r)
