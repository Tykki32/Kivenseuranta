"""Selainnakyma palvelinajoon (ei ikkunaa): seurannan kuva, ajon lokirivit, tilatiedot ja tulostiedostojen lataus.

netti_live.py --web PORTTI korvaa ikkunan (Naytto) talla: WebNaytto tarjoaa saman rajapinnan (anna, nayta_heti, paasilmukka, sulje), joten ajosilmukat eivat tieda eroa.

  /                  sivu: kuva, tila, loki, tiedostot
  /video.mjpg        kuva MJPEG-virtana (<img>-elementti toimii kaikissa selaimissa; JPEG pakataan vain, kun joku katsoo)
  /syote             aikaleimattu kuvasyote selaimen soittimelle: kuvat video-ajan mukaan harvennettuina (enintaan --web-fps kuvaa/s), rengaspuskuri SYOTE_S s; selain puskuroi ja toistaa ne
                     tasaisesti aikaleimojen mukaan, ja ± napeilla saadettava lisaviive (enintaan LISAVIIVE_MAX_S s) nayttaa kuvan vain myohemmin (analyysin viive ei muutu). ?historia=<s>: heti alkuun
                     lahetettavat aiemmat kuvat; id = sivun tunnus, lisa = sivun lisaviive. Kehys = SYOTE_OTSIKKO (pituus, video-aika, jarjestysnumero, liput) + JPEG; pituus 0 = sydankohta
  /lisa              ?id=<sivun tunnus>&s=<lisaviive s>: sivu ilmoittaa lisaviiveensa; > 0: kuvan pitaa edeta reaaliajassa (VOD-ajon kiri tauolle, ks. reaaliaika_haluttu)
  /kuva.jpg          tuorein kuva yksittaisena JPEG:na
  /tila              JSON: ajon tila, kuvien maara, katsojat
  /loki              viimeiset rivit konsolitulosteesta (teksti)
  /tiedostot/        tulostiedostot (mp4, csv); /tiedostot/<nimi> lataa (mp4 toistuu selaimessa, Range-tuki)
  /tiedostot.json    sama JSON:na
  /tilasto/tiedostot pelaajatilastoon kelpaavat heitto-CSV:t (JSON); nappi "Pelaajatilasto" sivulla
  /tilasto           pelaajakohtainen mediaani ja STD valitusta heitto-CSV:sta (JSON): ?tiedosto=<nimi>&kel=<pelaajat>&pun=<pelaajat>[&ohita=1][&hid=liuku|vapaa] (tools/pelaajatilasto.py)
  /tilasto.csv       sama CSV-tiedostona (Excel: erotin ;, desimaalipilkku; &fi=0: pilkkuerotin ja desimaalipiste)

Ei kirjautumista oletuksena: tarkoitettu kotiverkkoon. Salasana: ymparistomuuttujat NETTI_WEB_KAYTTAJA ja NETTI_WEB_SALASANA (HTTP Basic); ala avaa porttia internetiin.
Vain standardikirjasto + OpenCV (JPEG-pakkaus)."""
import base64
import collections
import hmac
import html
import http.server
import json
import mimetypes
import os
import re
import socket
import struct
import sys
import threading
import time
import urllib.parse

import cv2
import numpy as np

RUUTU_W, RUUTU_H = 1920, 1080
LOKI_RIVEJA = 400
MJPEG_RAJA = b"kehys"
RANGE_RE = re.compile(r"^bytes=(\d*)-(\d*)$")
SYOTE_S = 40.0                                                     # selaimen syotteen rengaspuskuri palvelimella (s): lisaviive (LISAVIIVE_MAX_S) + varaa
LISAVIIVE_MAX_S = 30                                               # selaimen lisaviiveen ylaraja (s) = selaimen puskuri
SYOTE_TAVUT_MAX = 400 * 1024 * 1024                                # rengaspuskurin muistiraja (tavua)
SYOTE_ODOTUS_S = 60.0                                              # kuvia koodataan viela taman ajan viimeisen katsojan poistuttua: uusi katsoja (tai uudelleenyhteys) saa historian
SYOTE_JALKEEN_S = 5.0                                              # katsoja, joka on yhteyden hitauden vuoksi yli historia + taman verran jalessa, ohittaa uusimpiin
SYOTE_SYDAN_S = 5.0                                                # tyhjan syotteen sydankohta (pituus 0) tahan tahtiin: katkennut yhteys huomataan
SYOTE_LISA_ODOTUS_S = 5.0                                          # katsojan lisaviive pysyy voimassa taman ajan yhteyden katkettua (uudelleenyhteys ei vapauta kiria hetkeksi)
SYOTE_LISA_VANHENEE_S = 20.0                                       # sivun lisaviive vanhenee, ellei sivu vahvista sita (se tekee sen 3 s valein): hiljaa kuollut yhteys ei pida kiria tauolla
SYOTE_LISA_TUNNUKSIA_MAX = 256                                     # sivutunnusten enimmaismaara (muistin suoja)
SYOTE_TUNNUS_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")             # sivun (katsojan) tunnus: selain arpoo sen sivun latauksessa
SYOTE_OTSIKKO = struct.Struct("<IdII")                             # kehyksen otsikko: JPEG:n pituus (tavua), video-aika (s), jarjestysnumero, liput (1 = historiaa: ei tuoretta tuotantoa)
TIEDOSTO_PAATTEET = (".mp4", ".csv", ".json", ".txt", ".log")        # listalle vain tulostiedostot (ei piilotiedostoja eika tilapaisia, kuten latautuvaa aanta)
SIVU = """<!doctype html>
<html lang="fi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Kivenseuranta</title>
<style>
:root{--tausta:#fff;--teksti:#1c1c1e;--himmea:#6b6b70;--reuna:#d8d8de;--laatikko:#f4f4f6;--korostus:#0a5bd3;--nappiteksti:#fff}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--tausta:#121214;--teksti:#ececf0;--himmea:#9a9aa3;--reuna:#33333a;--laatikko:#1b1b1f;--korostus:#6aa5ff;--nappiteksti:#08121f}}
body{margin:0;background:var(--tausta);color:var(--teksti);font:15px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:1920px;margin:0 auto;padding:12px 16px 40px}
h1{font-size:20px;margin:8px 0 12px}h2{font-size:15px;margin:22px 0 8px;color:var(--himmea);text-transform:uppercase;letter-spacing:.05em}
img{display:block;width:100%;height:auto;margin:0 auto;background:#000;border-radius:8px}
#tila{margin:10px 0;color:var(--himmea)}#tila b{color:var(--teksti)}
pre{background:var(--laatikko);border:1px solid var(--reuna);border-radius:8px;padding:10px 12px;margin:0;max-height:340px;overflow:auto;font:12.5px/1.4 ui-monospace,Menlo,Consolas,monospace;white-space:pre-wrap;word-break:break-word}
ul{list-style:none;margin:0;padding:0}li{padding:7px 0;border-bottom:1px solid var(--reuna);display:flex;gap:12px;justify-content:space-between;flex-wrap:wrap}
a{color:var(--korostus);text-decoration:none}a:hover{text-decoration:underline}.koko{color:var(--himmea);white-space:nowrap}
.ylapalkki{display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap}.ylapalkki h1{margin:8px 0 12px}
.nappi{display:inline-block;font:inherit;background:var(--korostus);color:var(--nappiteksti);border:1px solid var(--korostus);border-radius:8px;padding:7px 14px;cursor:pointer;text-decoration:none}
.nappi.toinen{background:var(--laatikko);color:var(--teksti);border-color:var(--reuna)}.nappi:hover{text-decoration:none;filter:brightness(1.08)}.nappi:disabled{opacity:.5;cursor:default}
.peite{position:fixed;top:0;right:0;bottom:0;left:0;z-index:10;display:flex;align-items:flex-start;justify-content:center;padding:4vh 12px;overflow:auto;background:rgba(0,0,0,.55)}.peite[hidden]{display:none}
.ikkuna{box-sizing:border-box;width:min(1240px,100%);background:var(--tausta);color:var(--teksti);border:1px solid var(--reuna);border-radius:12px;padding:16px 18px 20px;box-shadow:0 12px 40px rgba(0,0,0,.35)}
.ikkuna-ylos{display:flex;align-items:center;justify-content:space-between;gap:12px}.ikkuna h2{margin:0;font-size:18px;color:var(--teksti);text-transform:none;letter-spacing:0}
.lomake{display:flex;flex-wrap:wrap;align-items:flex-end;gap:12px 18px;margin:14px 0 8px}.lomake label{display:flex;flex-direction:column;gap:4px;font-size:13px;color:var(--himmea)}
.lomake label.valinta{flex-direction:row;align-items:center;gap:6px;color:var(--teksti);padding-bottom:8px}
.lomake select,.lomake input[type=text]{font:inherit;color:var(--teksti);background:var(--laatikko);border:1px solid var(--reuna);border-radius:8px;padding:7px 9px}
.lomake label:first-child{flex:1 1 100%}.lomake select{width:100%;min-width:0}.lomake label.hid select{width:auto;max-width:100%}@media (max-width:700px){.lomake label.hid{flex:1 1 100%}.lomake label.hid select{width:100%}}.lomake input[type=text]{width:7em}
#ts-tulos{overflow-x:auto}#ts-tulos .himmea{color:var(--himmea);margin:8px 0}
table.ts{border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums}table.ts th,table.ts td{border:1px solid var(--reuna);padding:3px 5px;white-space:nowrap}
table.ts th{background:var(--laatikko);font-weight:600;text-align:center}table.ts td.n{text-align:right}
.huom{background:var(--laatikko);border:1px solid var(--reuna);border-radius:8px;padding:8px 12px;margin:12px 0 0;color:var(--himmea);font-size:13px}.virhe{color:#c62828;margin:8px 0}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]) .virhe{color:#ff8a80}}
canvas#soitin{display:block;width:100%;height:auto;margin:0 auto;background:#000;border-radius:8px}
.viive{display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin:10px 0 0}.viive[hidden]{display:none}.viive .nappi{padding:5px 11px;min-width:3.6em}
#viive-teksti{color:var(--himmea);font-variant-numeric:tabular-nums;margin:0 6px}#viive-teksti b{color:var(--teksti)}
@media (max-width:560px){#viive-teksti{order:-1;flex:1 0 100%;margin:0}}
</style></head><body><main>
<div class="ylapalkki"><h1>Kivenseuranta</h1><button type="button" class="nappi" id="ts-avaa">Pelaajatilasto</button></div>
<img id="kuva" alt="Seurannan kuva" style="display:none">
<canvas id="soitin" style="display:none" aria-label="Seurannan kuva"></canvas>
<div class="viive" id="viiveohjain" hidden><button type="button" class="nappi toinen" data-d="-5" aria-label="Viive 5 sekuntia lyhyemmäksi">&minus;5 s</button><button type="button" class="nappi toinen" data-d="-1" aria-label="Viive sekunti lyhyemmäksi">&minus;1 s</button>
<span id="viive-teksti">Viive</span><button type="button" class="nappi toinen" data-d="1" aria-label="Viive sekunti pidemmäksi">+1 s</button><button type="button" class="nappi toinen" data-d="5" aria-label="Viive 5 sekuntia pidemmäksi">+5 s</button></div>
<div id="tila">Ladataan...</div>
<h2>Loki</h2><pre id="loki"></pre>
<h2>Tiedostot</h2><ul id="tiedostot"><li>Ladataan...</li></ul>
</main>
<div class="peite" id="tilasto" hidden>
<div class="ikkuna" role="dialog" aria-modal="true" aria-labelledby="ts-otsikko">
<div class="ikkuna-ylos"><h2 id="ts-otsikko">Pelaajakohtainen tilasto</h2><button type="button" class="nappi toinen" id="ts-sulje" aria-label="Sulje">Sulje</button></div>
<div class="lomake">
<label>Heitto-CSV (tulostiedostot)<select id="ts-tiedosto"><option>Ladataan...</option></select></label>
<label>Keltaisen joukkueen pelaajat<input type="text" id="ts-kel" value="4" placeholder="4 tai 3,3,2" autocomplete="off"></label>
<label>Punaisen joukkueen pelaajat<input type="text" id="ts-pun" value="4" placeholder="4 tai 3,3,2" autocomplete="off"></label>
<label class="hid">Hidastuvuus<select id="ts-hid"><option value="liuku">liukuvaiheen (ennen irroitusta)</option><option value="vapaa">vapaan lennon (irroituksen jälkeen, kuten kuvan paneelissa)</option></select></label>
<label class="valinta"><input type="checkbox" id="ts-ohita"> Ohita vajaat päät</label>
<button type="button" class="nappi" id="ts-laske" disabled>Laske</button>
</div>
<div id="ts-tulos"></div>
</div></div>
<script>
const $=id=>document.getElementById(id);
function koko(n){return n>=1e9?(n/1e9).toFixed(2)+" Gt":n>=1e6?(n/1e6).toFixed(1)+" Mt":Math.max(1,Math.round(n/1e3))+" kt"}
const esc=s=>String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;").replace(/'/g,"&#39;");
let edAlkoi=null;
async function paivita(){
  try{
    const t=await (await fetch("tila",{cache:"no-store"})).json();
    const kuva=$("kuva"),soitinEl=$("soitin"),ohjain=$("viiveohjain");
    if(t.joutilas){soitinPysayta();soitinEl.style.display="none";ohjain.hidden=true;kuva.style.display="none";if(kuva.getAttribute("src"))kuva.removeAttribute("src")}
    else if(SOITIN_TUKI){                                                   // aikaleimattu syote: selain puskuroi ja toistaa tasaisesti, lisaviive vain nayttoon
      kuva.style.display="none";if(kuva.getAttribute("src"))kuva.removeAttribute("src");soitinEl.style.display="";ohjain.hidden=false;
      S.yht=typeof t.viive_s==="number"?t.viive_s:null;S.nopea=t.viive_s===null;S.max=t.lisaviive_max_s||30;S.pit=t.syote_s||40;S.lisa=Math.min(S.lisa,S.max);
      if(!S.ac||t.alkoi!==S.epoch)soitinKaynnista(t.alkoi);
      else{if(t.kaynnissa&&performance.now()/1000-S.viimKehys>15)soitinYhdista(true);soitinLahetaLisa(true)}       // syote on hiljentynyt: yhdista uudelleen (puskuri sailyy); lisaviive ilmoitetaan toistuvasti (hukkuneen ilmoituksen varalta)
    }
    else{kuva.style.display="";if(t.alkoi!==edAlkoi||!kuva.getAttribute("src"))kuva.src="video.mjpg?"+Date.now()}       // varapolku: MJPEG ilman viiveensaatoa
    edAlkoi=t.alkoi;
    $("tila").innerHTML=t.joutilas?"<b>Ei analyysia käynnissä</b>"+(t.otsikko?" &middot; "+esc(t.otsikko):""):(t.kaynnissa?"<b>Ajo kaynnissa</b>":"<b>Ajo paattynyt</b>")+" &middot; kuvia "+t.kuvia+" &middot; katsojia "+t.katsojia+(t.viim_kuva_ika_s!=null?" &middot; viimeisin kuva "+t.viim_kuva_ika_s.toFixed(1)+" s sitten":"")+(t.kaynnissa&&t.viim_kuva_ika_s!=null&&t.viim_kuva_ika_s>30?" &middot; <b>kuva ei etene: katso loki (Jumivahti)</b>":"")+((t.otsikko||t.lahde)?" &middot; "+esc(t.otsikko||t.lahde):"");
    document.title=t.otsikko?"Kivenseuranta: "+t.otsikko:"Kivenseuranta";
    if(t.kuva_leveys){$("kuva").style.maxWidth=t.kuva_leveys+"px";$("soitin").style.maxWidth=t.kuva_leveys+"px"}
    const l=$("loki"),alhaalla=l.scrollTop+l.clientHeight>=l.scrollHeight-30;
    l.textContent=await (await fetch("loki",{cache:"no-store"})).text();
    if(alhaalla)l.scrollTop=l.scrollHeight;
    const f=await (await fetch("tiedostot.json",{cache:"no-store"})).json();
    $("tiedostot").innerHTML=f.length?f.map(x=>"<li><a href=\\"tiedostot/"+encodeURIComponent(x.nimi)+"\\">"+x.nimi.replace(/&/g,"&amp;").replace(/</g,"&lt;")+"</a><span class=koko>"+koko(x.koko)+"</span></li>").join(""):"<li>Ei tiedostoja viela</li>";
  }catch(e){$("tila").textContent="Yhteys palvelimeen katkesi (ajo paattynyt tai kontti sammutettu)."}
}
// ---- soitin: aikaleimattu syote, selaimen puskuri ja lisaviive --------------------------------------------------------------------------------------------------------------------
// Palvelin lahettaa kuvat video-ajan mukaan harvennettuina (kehys: 20 tavun otsikko + JPEG). Selain pitaa viimeiset S.pit sekuntia puskurissa ja toistaa ne tasaisesti aikaleimojen mukaan.
// Lisaviive (S.lisa, napit) muuttaa vain sita, kuinka kaukana tuotannon uusimmasta kuvasta toisto on; analyysin oma viive ei muutu. Lisaviive 0: nayta uusin kuva heti (kuten ennen).
// Lisaviive > 0: toisto etenee aina oikealla nopeudella (1x). Jotta se onnistuu myos VOD-ajon kirin (esim. 2x) aikana, sivu ilmoittaa lisaviiveensa palvelimelle (S.id, /lisa), ja palvelin pysayttaa kirin
// niin kauan kuin joku nakyva valilehti kayttaa lisaviivetta. Piilossa oleva valilehti ilmoittaa 0 (kukaan ei katso). Vain nopeassa ajossa (S.nopea: ei perusviivetta) toisto seuraa tuotannon vauhtia.
const SOITIN_TUKI=!!(window.fetch&&window.ReadableStream&&window.createImageBitmap&&window.AbortController);
function soitinTunnus(){let t="";try{t=crypto.randomUUID().replace(/-/g,"")}catch(e){}if(!t)t=(Math.random().toString(36).slice(2)+Math.random().toString(36).slice(2)+Date.now().toString(36)).replace(/[^A-Za-z0-9]/g,"");return t.slice(0,32)}
const S={puskuri:[],tavut:0,L:null,P:null,tila:"toisto",lisa:0,max:30,pit:40,yht:null,nopea:false,id:soitinTunnus(),lahLisa:null,nayt:null,haettu:null,naytteet:[],kiri:false,ac:null,epoch:null,viimKehys:0,edT:0,edKehysT:null,silmukka:false};
{const l=parseInt(tsMuisti("viive_lisa")||"0",10);if(isFinite(l))S.lisa=Math.max(0,Math.min(30,l))}
function soitinLisaNyt(){return document.hidden?0:S.lisa}                                         // piilossa oleva valilehti ei katso: lisaviive 0 (kiri saa jatkua)
function soitinLahetaLisa(pakko){const e=soitinLisaNyt();if(!pakko&&e===S.lahLisa)return;S.lahLisa=e;fetch("lisa?id="+S.id+"&s="+e,{cache:"no-store"}).catch(()=>{})}
function soitinNollaa(){S.puskuri=[];S.tavut=0;S.L=S.P=null;S.nayt=null;S.naytteet=[];S.kiri=false;S.tila="toisto"}
function soitinPysayta(){if(S.ac){S.ac.abort();S.ac=null}}
function soitinKaynnista(epoch){soitinPysayta();soitinNollaa();S.epoch=epoch;S.viimKehys=performance.now()/1000;soitinYhdista(false);if(!S.silmukka){S.silmukka=true;requestAnimationFrame(soitinSilmukka)}}
async function soitinYhdista(uusi){
  if(uusi)soitinPysayta();
  const ac=S.ac=new AbortController();
  while(!ac.signal.aborted){
    S.edKehysT=null;
    try{
      const h=Math.min(S.pit,Math.ceil(S.lisa+(S.L===null?2:8))),e=soitinLisaNyt();                 // historia heti alkuun: toisto voi alkaa lisaviiveen paassa
      S.lahLisa=e;
      const v=await fetch("syote?historia="+h+"&id="+S.id+"&lisa="+e,{signal:ac.signal,cache:"no-store"});
      if(!v.ok)throw new Error("HTTP "+v.status);
      S.viimKehys=performance.now()/1000;soitinLahetaLisa();                                          // lisaviive voi olla vaihtunut yhteyden avautuessa
      await soitinLue(v.body.getReader());
    }catch(e){}
    if(ac.signal.aborted)return;
    await new Promise(r=>setTimeout(r,1000));
  }
}
async function soitinLue(lukija){
  let osat=[],osia=0,ot=null;
  const ota=n=>{const u=new Uint8Array(n);let k=0;while(k<n){const o=osat[0],m=Math.min(n-k,o.length);u.set(o.subarray(0,m),k);k+=m;if(m===o.length)osat.shift();else osat[0]=o.subarray(m)}osia-=n;return u};
  for(;;){
    const r=await lukija.read();
    if(r.done)return;
    osat.push(r.value);osia+=r.value.length;
    for(;;){
      if(!ot){
        if(osia<20)break;
        const dv=new DataView(ota(20).buffer);
        ot={n:dv.getUint32(0,true),t:dv.getFloat64(4,true),liput:dv.getUint32(16,true)};
        if(ot.n>2e7)throw new Error("virheellinen kehys");
      }
      if(osia<ot.n)break;
      soitinKehys(ot,ota(ot.n));
      ot=null;
    }
  }
}
function soitinKehys(o,u){
  const nyt=performance.now()/1000;
  S.viimKehys=nyt;
  if(o.n===0)return;                                                                              // sydankohta
  if(S.edKehysT!==null&&o.t<S.edKehysT-0.5)soitinNollaa();                                        // video-aika hyppasi taaksepain (uusi aikaperusta): aloita puskuri alusta
  S.edKehysT=o.t;
  if(S.L!==null&&o.t<=S.L+1e-6)return;                                                            // jo puskurissa (uudelleenyhteyden historia)
  S.puskuri.push({t:o.t,blob:new Blob([u],{type:"image/jpeg"})});S.tavut+=o.n;S.L=o.t;
  if(!(o.liput&1))S.naytteet.push([nyt,o.t]);                                                     // tuore kuva (ei historiaa): tuotannon vauhdin arvio
  while(S.puskuri.length>1&&(S.L-S.puskuri[0].t>S.pit||S.tavut>3e8))S.tavut-=S.puskuri.shift().blob.size;
}
function soitinVauhti(nyt){                                                                       // tuotannon vauhti (video-sekuntia / s) viimeisen 20 s ajalta; 1, jos dataa on vahan
  const n=S.naytteet;while(n.length>2&&nyt-n[0][0]>20)n.shift();
  if(n.length<2)return 1;
  const a=n[0],b=n[n.length-1];
  return b[0]-a[0]>=8?(b[1]-a[1])/(b[0]-a[0]):1;
}
function soitinAsetaPaikka(){                                                                     // lisaviiveen muutos: hyppy puskurissa (ei historiaa riittavasti: odotetaan, kunnes puskuri taytyy)
  const L=S.L,vanhin=S.puskuri[0].t;
  if(S.lisa===0){S.P=L;S.tila="toisto";return}
  if(L-S.lisa>=vanhin){S.P=L-S.lisa;S.tila="toisto"}
  else{S.P=Math.max(S.P===null?vanhin:S.P,vanhin);S.tila="puskuroi"}
}
function soitinPaikka(nyt,dt){
  const L=S.L,vanhin=S.puskuri[0].t,lisa=S.lisa,v=soitinVauhti(nyt);
  if(!S.kiri&&v>1.3)S.kiri=true;else if(S.kiri&&v<1.15)S.kiri=false;                              // tuotanto etenee reaaliaikaa nopeammin (vain nopeassa ajossa, S.nopea: toisto seuraa sen vauhtia; VOD-kiri pysaytetaan palvelimella)
  if(S.P===null)soitinAsetaPaikka();
  if(lisa===0){S.P=L;S.tila="toisto";return}
  S.P=Math.max(S.P,vanhin);
  const lag=L-S.P;
  if(S.tila==="puskuroi"){if(lag<lisa-0.3)return;S.tila="toisto";if(lag>lisa+0.5){S.P=L-lisa;return}}      // puskuri taynna (esim. uudelleenlatauksen historia): suoraan lisaviiveen paahan
  if(lag<0.02){S.tila="puskuroi";return}                                                          // puskuri tyhjeni (tuotanto jai jalkeen): kerataan uudelleen lisaviiveeseen
  const ero=lag-lisa;
  if(ero>4){S.P=Math.max(vanhin,L-lisa);return}                                                   // liian kaukana tuotannosta (esim. valilehti oli taustalla): hyppaa
  S.P=Math.min(L,S.P+(S.kiri&&S.nopea?Math.min(v,4):1)*(1+Math.max(-0.1,Math.min(0.1,0.2*ero)))*dt);       // normaali nopeus (nopeassa ajossa tuotannon vauhti) +-10 % korjaus kohti lisaviivetta
}
function soitinKehysPaikalle(P){let a=0,b=S.puskuri.length-1,r=0;while(a<=b){const m=(a+b)>>1;if(S.puskuri[m].t<=P){r=m;a=m+1}else b=m-1}return S.puskuri[r]}
async function soitinPiirra(){
  const k=soitinKehysPaikalle(S.P);
  if(!k||k===S.nayt||S.haettu)return;
  S.haettu=k;
  try{
    const b=await createImageBitmap(k.blob),c=$("soitin");
    if(c.width!==b.width||c.height!==b.height){c.width=b.width;c.height=b.height}
    c.getContext("2d").drawImage(b,0,0);b.close();
  }catch(e){}
  S.nayt=k;S.haettu=null;
}
function soitinSilmukka(ms){
  requestAnimationFrame(soitinSilmukka);
  const nyt=ms/1000,dt=Math.min(0.25,Math.max(0,nyt-S.edT));S.edT=nyt;
  if(!S.puskuri.length)return;
  soitinPaikka(nyt,dt);
  soitinPiirra();
}
function soitinLisa(d){
  S.lisa=Math.max(0,Math.min(S.max,S.lisa+d));
  tsMuisti("viive_lisa",String(S.lisa));
  S.naytteet=[];S.kiri=false;                                                                       // tuotannon vauhti mitataan uudelleen (palvelin voi vaihtaa kirista 1x:aan)
  soitinLahetaLisa();
  if(S.puskuri.length)soitinAsetaPaikka();
  soitinOhjain();
}
function soitinOhjain(){
  const f=x=>String(Math.round(x*10)/10).replace(".",","),D=S.yht;
  let h="<b>Viive "+(D!==null?f(D+S.lisa)+" s</b>"+(S.lisa>0?" ("+f(D)+" + "+S.lisa+")":""):(S.lisa>0?"+"+S.lisa:"0")+" s</b>");
  if(S.tila==="puskuroi"&&S.lisa>0&&S.L!==null&&S.P!==null)h+=" &middot; puskuroidaan "+f(Math.max(0,S.L-S.P))+" / "+S.lisa+" s";
  if(S.nopea&&S.kiri&&S.lisa>0)h+=" &middot; nopea ajo: kuva etenee reaaliaikaa nopeammin, viive on videon aikaa";
  const el=$("viive-teksti");if(el.innerHTML!==h)el.innerHTML=h;
  for(const b of $("viiveohjain").querySelectorAll("button")){const d=+b.dataset.d;b.disabled=d<0?S.lisa<=0:S.lisa>=S.max}
}
$("viiveohjain").addEventListener("click",e=>{const b=e.target.closest("button[data-d]");if(b&&!b.disabled)soitinLisa(+b.dataset.d)});
document.addEventListener("visibilitychange",()=>{S.naytteet=[];S.kiri=false;soitinLahetaLisa()});          // piilossa kiri on voinut jatkua: tuotannon vauhti mitataan uudelleen
setInterval(soitinOhjain,250);
let tsTulos=null;
function tsMuisti(k,v){try{if(v===undefined)return localStorage.getItem(k);localStorage.setItem(k,v)}catch(e){return null}}
function tsKysely(){return "tiedosto="+encodeURIComponent($("ts-tiedosto").value)+"&kel="+encodeURIComponent($("ts-kel").value.trim())+"&pun="+encodeURIComponent($("ts-pun").value.trim())+($("ts-ohita").checked?"&ohita=1":"")+"&hid="+encodeURIComponent($("ts-hid").value)}
function tsNum(x,des){return x==null?"-":x.toFixed(des).replace(".",",")}
async function tsAvaa(){
  $("tilasto").hidden=false;document.body.style.overflow="hidden";
  const kel=tsMuisti("ts_kel"),pun=tsMuisti("ts_pun"),hid=tsMuisti("ts_hid");
  if(kel)$("ts-kel").value=kel;
  if(pun)$("ts-pun").value=pun;
  if(hid==="liuku"||hid==="vapaa")$("ts-hid").value=hid;
  const v=$("ts-tiedosto"),edellinen=v.value;
  try{
    const r=await fetch("tilasto/tiedostot",{cache:"no-store"}),d=await r.json();
    if(!r.ok)throw new Error(d.virhe||r.status);
    v.textContent="";                                                      // option-elementit DOM:illa: tiedostonimessa voi olla heittomerkki tai lainausmerkki (esim. YouTube-otsikko)
    for(const x of d){const o=document.createElement("option");o.value=x.nimi;o.textContent=x.nimi+" ("+new Date(x.muutettu*1000).toLocaleString("fi-FI")+", "+koko(x.koko)+")";v.appendChild(o)}
    if(!d.length){const o=document.createElement("option");o.value="";o.textContent="Ei heitto-CSV-tiedostoja";v.appendChild(o)}
    if(edellinen&&d.some(x=>x.nimi===edellinen))v.value=edellinen;
    $("ts-laske").disabled=!d.length;
    if(!d.length)$("ts-tulos").innerHTML="<p class='himmea'>Tulostiedostoissa ei ole vielä heitto-CSV:tä. Se syntyy, kun video on analysoitu (heitot_&hellip;.csv).</p>";
  }catch(e){$("ts-tulos").innerHTML="<p class='virhe'>Tiedostolistaa ei saatu: "+esc(e.message||e)+"</p>";$("ts-laske").disabled=true}
  v.focus();
}
function tsSulje(){$("tilasto").hidden=true;document.body.style.overflow=""}
async function tsLaske(){
  const nappi=$("ts-laske");nappi.disabled=true;$("ts-tulos").innerHTML="<p class='himmea'>Lasketaan...</p>";
  tsMuisti("ts_kel",$("ts-kel").value.trim());tsMuisti("ts_pun",$("ts-pun").value.trim());tsMuisti("ts_hid",$("ts-hid").value);
  try{
    const r=await fetch("tilasto?"+tsKysely(),{cache:"no-store"}),d=await r.json();
    if(!r.ok){$("ts-tulos").innerHTML="<p class='virhe'>"+esc(d.virhe||("Virhe "+r.status))+"</p>";return}
    tsNayta(d);
  }catch(e){$("ts-tulos").innerHTML="<p class='virhe'>Laskenta epäonnistui: "+esc(e.message||e)+"</p>"}
  finally{nappi.disabled=false}
}
function tsNayta(d){
  tsTulos=d;
  const des=d.desimaalit;
  let a="<th colspan='6'></th>",b="<th colspan='6'></th>",c="<th>joukkue</th><th>pelaaja</th><th>kivet</th><th>heittoja</th><th>n myötä</th><th>n vasta</th>";
  for(const g of d.ryhmat){a+="<th colspan='"+2*g.suureet.length+"'>"+esc(g.nimi)+"</th>";for(const s of g.suureet){b+="<th colspan='2'>"+esc(s)+"</th>";c+="<th>Mediaani</th><th>STD</th>"}}
  let h="<p class='himmea'>"+esc(d.tiedosto)+": "+d.heittoja+" heittoa; "+d.joukkueet.map(j=>esc(j.vari)+" "+j.pelaajia+" pelaajaa (kivet "+j.kivet.join(", ")+")").join("; ")+"; Hidastuvuus: "+(d.hidastuvuus==="vapaa"?"vapaan lennon":"liukuvaiheen")+"</p>";
  h+="<table class='ts'><thead><tr>"+a+"</tr><tr>"+b+"</tr><tr>"+c+"</tr></thead><tbody>";
  for(const r of d.rivit){
    h+="<tr><td>"+esc(r.joukkue)+"</td><td class='n'>"+r.pelaaja+"</td><td>"+esc(r.kivet)+"</td><td class='n'>"+r.heittoja+"</td><td class='n'>"+r.myota+"</td><td class='n'>"+r.vasta+"</td>";
    for(let i=0;i<r.mediaani.length;i++){const t=" title='n = "+r.n[i]+"'";h+="<td class='n'"+t+">"+tsNum(r.mediaani[i],des[i])+"</td><td class='n'"+t+">"+tsNum(r.std[i],des[i])+"</td>"}
    h+="</tr>";
  }
  h+="</tbody></table>";
  if(d.varoitukset.length)h+="<div class='huom'>"+d.varoitukset.map(v=>"<div>Huom: "+esc(v)+"</div>").join("")+"</div>";
  h+="<p><button type='button' class='nappi toinen' id='ts-kopioi'>Kopioi Exceliin</button> <a class='nappi toinen' id='ts-csv' href='#'>Lataa CSV (Excel)</a></p>";
  $("ts-tulos").innerHTML=h;
  $("ts-csv").href="tilasto.csv?"+tsKysely()+"&fi=1";                      // osoite DOM:illa (ei merkkijonoa attribuuttiin): tiedostonimen heittomerkki ei riko linkkia
  $("ts-kopioi").onclick=()=>tsKopioi($("ts-kopioi"));
}
function tsTsv(d){
  const T=String.fromCharCode(9),N=String.fromCharCode(10),des=d.desimaalit;
  const a=["","","","","",""],b=["","","","","",""],c=["joukkue","pelaaja","kivet","heittoja","n myötä","n vasta"];
  for(const g of d.ryhmat){a.push(g.nimi);for(let i=1;i<2*g.suureet.length;i++)a.push("");for(const s of g.suureet){b.push(s,"");c.push("Mediaani","STD")}}
  const rivit=[a,b,c];
  for(const r of d.rivit){const x=[r.joukkue,r.pelaaja,r.kivet,r.heittoja,r.myota,r.vasta];for(let i=0;i<r.mediaani.length;i++)x.push(r.mediaani[i]==null?"":tsNum(r.mediaani[i],des[i]+1),r.std[i]==null?"":tsNum(r.std[i],des[i]+1));rivit.push(x)}
  return rivit.map(r=>r.join(T)).join(N)+N;
}
async function tsKopioi(nappi){
  const teksti=tsTsv(tsTulos);let ok=false;
  try{if(navigator.clipboard&&window.isSecureContext){await navigator.clipboard.writeText(teksti);ok=true}}catch(e){}
  if(!ok){const k=document.createElement("textarea");k.value=teksti;k.style.position="fixed";k.style.opacity="0";document.body.appendChild(k);k.select();try{ok=document.execCommand("copy")}catch(e){}document.body.removeChild(k)}
  const vanha=nappi.textContent;nappi.textContent=ok?"Kopioitu: liitä Exceliin":"Kopiointi ei onnistunut";setTimeout(()=>{nappi.textContent=vanha},2500);
}
$("ts-avaa").onclick=tsAvaa;$("ts-sulje").onclick=tsSulje;$("ts-laske").onclick=tsLaske;
$("tilasto").addEventListener("click",e=>{if(e.target===$("tilasto"))tsSulje()});
document.addEventListener("keydown",e=>{if(e.key==="Escape"&&!$("tilasto").hidden)tsSulje()});
for(const id of ["ts-kel","ts-pun"])$(id).addEventListener("keydown",e=>{if(e.key==="Enter"&&!$("ts-laske").disabled)tsLaske()});
paivita();setInterval(paivita,3000);
</script></body></html>
"""


class LokiPuskuri:
    """Kopioi kirjoitetun tekstin riveittain pyorivaan puskuriin ja valittaa sen alkuperaiselle virralle: docker logs nayttaa kaiken kuten ennenkin, /loki saman puskurista."""

    def __init__(self, alkuperainen, rivit, lukko=None):
        self._alk = alkuperainen
        self._rivit = rivit
        self._kesken = ""
        self._lukko = lukko or threading.Lock()

    def write(self, teksti):
        n = self._alk.write(teksti)
        try:
            with self._lukko:
                self._kesken += teksti
                *valmiit, self._kesken = self._kesken.split("\n")
                for r in valmiit:
                    r = r.split("\r")[-1].rstrip()                 # edistymisrivit kirjoittavat \r:lla paalle: vain viimeinen versio
                    if r:
                        self._rivit.append(r)
                if len(self._kesken) > 20000:
                    self._kesken = self._kesken[-2000:]
        except Exception:                                          # loki ei saa kaataa tulostusta
            pass
        return n

    def flush(self):
        return self._alk.flush()

    def __getattr__(self, nimi):                                   # fileno, isatty, encoding, buffer ...
        return getattr(self._alk, nimi)


def asenna_lokipuskuri(rivit=None):
    """Korvaa sys.stdout ja sys.stderr LokiPuskurilla (idempotentti); palauttaa rivijonon."""
    if rivit is None:
        rivit = collections.deque(maxlen=LOKI_RIVEJA)
    lukko = threading.Lock()
    for nimi in ("stdout", "stderr"):
        virta = getattr(sys, nimi)
        if not isinstance(virta, LokiPuskuri):
            setattr(sys, nimi, LokiPuskuri(virta, rivit, lukko))
    return rivit


TILASTO_POIS = ("_kivet.csv", "_pelaajat.csv")                  # johdetut tiedostot: ei heitto-CSV:ta


def _tilasto_moduuli():
    """tools/pelaajatilasto.py (laiska tuonti: ohjelmakansio voi olla paivitetty vain osittain, eika puuttuva tiedosto saa kaataa muuta sivua)."""
    tools = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools")
    if tools not in sys.path:
        sys.path.append(tools)
    try:
        import pelaajatilasto
    except ImportError:
        raise RuntimeError("tools/pelaajatilasto.py puuttuu: kopioi koko ohjelmakansio (myös tools/) ja rakenna kuva uudelleen") from None
    return pelaajatilasto


def odotuskuva(teksti="Odotetaan ensimmaista kuvaa ..."):
    """Mustalle pohjalle kirjoitettu ilmoituskuva (ennen ensimmaista oikeaa kuvaa)."""
    kuva = np.zeros((RUUTU_H, RUUTU_W, 3), np.uint8)
    cv2.putText(kuva, teksti, (120, RUUTU_H // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 2, cv2.LINE_AA)
    return kuva


class WebNaytto:
    """Ikkunan korvike: sama rajapinta kuin netti_live.Naytto (anna, nayta_heti, paasilmukka, sulje, n_nayt), kuvat selaimeen HTTP:lla.
    portti 0 = vapaa portti (testaus; todellinen portti: self.portti). ulos_kansio: kansio, jonka tiedostot ovat ladattavissa. tila: kutsuttava, joka palauttaa lisatiedot /tila-vastaukseen."""

    def __init__(self, portti, ulos_kansio=None, leveys=RUUTU_W, fps=8.0, laatu=75, osoite="0.0.0.0", tila=None, loki_rivit=None):
        self.ulos_kansio = os.path.abspath(ulos_kansio) if ulos_kansio else None
        self.leveys = max(160, min(int(leveys), RUUTU_W))
        self.fps = max(0.5, float(fps))
        self.laatu = max(10, min(int(laatu), 100))
        self.tila_fn = tila
        self.loki_rivit = loki_rivit if loki_rivit is not None else collections.deque(maxlen=LOKI_RIVEJA)
        self.n_nayt = 0
        self.alkoi = time.time()
        self.kaynnissa = True
        self._ehto = threading.Condition()
        self._koodaus_lukko = threading.Lock()
        self._lopeta = threading.Event()
        self._uusi = threading.Event()
        self._kuva = odotuskuva()
        self._id = 0
        self._viim_t = None
        self._jpeg = (None, None)                                  # (kuvan id, JPEG-tavut)
        self._katsojat = 0
        self._lukko = threading.Lock()
        self._yhteydet = set()                                     # avoimet asiakasyhteydet (sulje() katkaisee ne)
        self._anna_kuva = None                                     # (kuva, video-aika)
        self._alku_mono = time.monotonic()
        # aikaleimattu syote selaimelle (soitin puskuroi ja toistaa tasaisesti; lisaviive vain nayttoon): kuvat koodataan vain, kun katsojia on (tai hetki sen jalkeen)
        self._syote_ehto = threading.Condition()
        self._syote = collections.deque()                          # (jarjestysnro, video-aika, JPEG-tavut); enintaan SYOTE_S s ja SYOTE_TAVUT_MAX tavua
        self._syote_tavut = 0
        self._syote_seq = 0
        self._syote_odottaa = None                                 # (video-aika, julkaisuhetki, kuva): seuraava koodattava (uusin voittaa)
        self._syote_viim = None                                    # (video-aika, julkaisuhetki) viimeisin syotteeseen otettu kuva (harvennus)
        self._syote_edellinen = None                               # edellinen julkaistu kuva (kuvavalin arvio)
        self._syote_dt = None                                      # lahteen kuvavali (s), arvio
        self._syote_katsojia = 0
        self._syote_lisat = {}                                     # sivun tunnus -> [lisaviive s, yhteyksia, yhteydet poissa -> voimassa asti (monotonic) | None, viimeisin vahvistus (monotonic)]
        self._syote_aktiivinen_asti = 0.0
        self._syote_saie = None
        self._kayttaja = os.environ.get("NETTI_WEB_KAYTTAJA") or ""
        self._salasana = os.environ.get("NETTI_WEB_SALASANA") or ""
        try:
            self.palvelin = http.server.ThreadingHTTPServer((osoite, int(portti)), _Kasittelija)
        except OSError as e:
            raise SystemExit(f"Web-porttia {portti} ei voitu avata ({e}): onko portti jo kaytossa?")
        self.palvelin.daemon_threads = True
        self.palvelin.naytto = self
        self.portti = self.palvelin.server_address[1]
        self._saie = threading.Thread(target=self.palvelin.serve_forever, kwargs=dict(poll_interval=0.2), name="web", daemon=True)
        self._saie.start()

    # ---- Naytto-rajapinta ------------------------------------------------------------------------------------------
    def anna(self, kuva, aika=None):
        """Seuraava nayttettava kuva (mista tahansa saikeesta); edellinen nayttamaton kuva korvautuu. aika: kuvan aika videossa (s; syotteen aikaleima: soitin toistaa tasaisesti sen mukaan);
        ilman sita seinakello."""
        with self._lukko:
            self._anna_kuva = (kuva, aika)
        self._uusi.set()

    def nayta_heti(self, kuva, aika=None):
        """Synkroninen (simulaatio): kuva julkaistaan heti; palauttaa True vain jos lopetus on pyydetty (selaimesta ei voi lopettaa)."""
        self._julkaise(kuva, aika=aika)
        return False

    def paasilmukka(self, lopeta, valmis, ilmoitus=None):
        """Paasaie: julkaisee annetut kuvat, kunnes lopeta (Event) asetetaan tai valmis() on tosi ja kuvat on julkaistu. ilmoitus(): kuva (tai None) ennen ensimmaista oikeaa kuvaa."""
        viim_ilm = 0.0
        while not lopeta.is_set():
            self._uusi.wait(0.05)
            self._uusi.clear()
            with self._lukko:
                annettu, self._anna_kuva = self._anna_kuva, None
            if annettu is not None:
                self._julkaise(annettu[0], aika=annettu[1])
            elif self.n_nayt == 0 and ilmoitus is not None and time.time() - viim_ilm > 0.25:
                ilm = ilmoitus()
                if ilm is not None:
                    self._julkaise(ilm, laske=False)
                viim_ilm = time.time()
            if valmis() and self._anna_kuva is None:
                break

    def sulje(self):
        """Lopettaa palvelun: portti vapautuu ja avoimet asiakasyhteydet katkaistaan. Pysyva (keep-alive) yhteys ei saa jaada vastaamaan vanhalla tilalla, kun sama portti on annettu seuraavalle
        palvelimelle (JoutilasSivu -> analyysi): selain pitaisi muuten 3 s:n tilakyselynsa vanhassa yhteydessa eika nakisi analyysia."""
        self.kaynnissa = False
        self._lopeta.set()
        with self._ehto:
            self._ehto.notify_all()
        with self._syote_ehto:
            self._syote_ehto.notify_all()
        try:
            self.palvelin.shutdown()
            self.palvelin.server_close()
        except Exception:
            pass
        with self._lukko:
            yhteydet = list(self._yhteydet)
        for s in yhteydet:
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def yhteys(self, s, auki):
        """Kasittelija ilmoittaa asiakasyhteyden avautumisen ja sulkeutumisen."""
        with self._lukko:
            (self._yhteydet.add if auki else self._yhteydet.discard)(s)

    # ---- kuvat -----------------------------------------------------------------------------------------------------
    def _julkaise(self, kuva, laske=True, aika=None):
        with self._ehto:
            self._kuva = kuva
            self._id += 1
            self._viim_t = time.time()
            if laske:
                self.n_nayt += 1
            self._ehto.notify_all()
        self._syotteeseen(time.monotonic() - self._alku_mono if aika is None else float(aika), kuva)

    def jpeg(self, edellinen_id=None, odotus_s=0.0):
        """(JPEG-tavut, kuvan id). edellinen_id annettu: odottaa enintaan odotus_s sekuntia kuvaa, joka eroaa siita (None, id, jos ei tullut). Pakkaus tehdaan kerran kuvaa kohden,
        kaikki katsojat jakavat sen."""
        with self._ehto:
            if edellinen_id is not None and odotus_s > 0:
                self._ehto.wait_for(lambda: self._id != edellinen_id or self._lopeta.is_set(), odotus_s)
            if edellinen_id is not None and self._id == edellinen_id:
                return None, edellinen_id
            kuva, id_ = self._kuva, self._id
        with self._koodaus_lukko:
            if self._jpeg[0] == id_:
                return self._jpeg[1], id_
            tavut = self._pakkaa(kuva)
            if tavut is None:
                return None, id_
            self._jpeg = (id_, tavut)
            return tavut, id_

    def _pakkaa(self, kuva):
        """Kuva JPEG-tavuiksi (pienennetaan leveyteen self.leveys, jos on leveampi); None, jos pakkaus epaonnistuu."""
        pieni = kuva
        if kuva.shape[1] > self.leveys:
            pieni = cv2.resize(kuva, (self.leveys, int(round(self.leveys * kuva.shape[0] / kuva.shape[1]))), interpolation=cv2.INTER_AREA)
        ok, puskuri = cv2.imencode(".jpg", pieni, [cv2.IMWRITE_JPEG_QUALITY, self.laatu])
        return puskuri.tobytes() if ok else None

    # ---- aikaleimattu syote selaimen soittimelle ------------------------------------------------------------------------
    def _syotteeseen(self, aika, kuva):
        """Julkaistu kuva syotteen koodaajalle (vain jos katsojia on tai oli aivan hetki sitten). Julkaisijan saie ei koodaa mitaan: uusin kuva jaa odottamaan, ja koodaaja harventaa video-ajan mukaan."""
        nyt = time.monotonic()
        with self._syote_ehto:
            ed = self._syote_edellinen
            if ed is not None and 0.0 < aika - ed[0] < 0.5:
                self._syote_dt = aika - ed[0] if self._syote_dt is None else 0.8 * self._syote_dt + 0.2 * (aika - ed[0])
            self._syote_edellinen = (aika, nyt)
            if self._syote_katsojia > 0 or nyt < self._syote_aktiivinen_asti:
                self._syote_odottaa = (aika, nyt, kuva)
                self._syote_ehto.notify_all()

    def syote_katsoja(self, muutos):
        """Syotteen katsojamaara (+1 / -1). Ensimmainen katsoja kaynnistaa koodaajan (ja tyhjentaa vanhentuneen historian); viimeisen poistuttua koodaus jatkuu SYOTE_ODOTUS_S s, jotta uudelleenyhteys saa historian."""
        with self._syote_ehto:
            if muutos > 0 and self._syote_katsojia == 0 and time.monotonic() >= self._syote_aktiivinen_asti:
                self._syote.clear()                                                        # koodaus on ollut tauolla: vanha historia ei jatku nykyhetkeen
                self._syote_tavut, self._syote_viim, self._syote_odottaa = 0, None, None
            self._syote_katsojia = max(0, self._syote_katsojia + muutos)
            if self._syote_katsojia == 0:
                self._syote_aktiivinen_asti = time.monotonic() + SYOTE_ODOTUS_S
            if muutos > 0 and self._syote_saie is None and not self._lopeta.is_set():
                self._syote_saie = threading.Thread(target=self._syote_koodaaja, name="web-syote", daemon=True)
                self._syote_saie.start()
            return self._syote_katsojia

    def syote_lisa_auki(self, tunnus, lisa):
        """Syoteyhteys avautui: sivu (tunnus) ilmoittaa lisaviiveensa (s). Sama sivu voi avata uuden yhteyden ennen kuin vanha sulkeutuu, joten yhteydet lasketaan."""
        with self._syote_ehto:
            self._syote_lisat_siivoa()
            if tunnus not in self._syote_lisat and len(self._syote_lisat) >= SYOTE_LISA_TUNNUKSIA_MAX:
                return
            kirje = self._syote_lisat.setdefault(tunnus, [0.0, 0, None, 0.0])
            kirje[0], kirje[1], kirje[2], kirje[3] = self._lisa_rajaa(lisa), kirje[1] + 1, None, time.monotonic()

    def syote_lisa_kiinni(self, tunnus):
        """Syoteyhteys sulkeutui: viimeisen yhteyden jalkeen lisaviive pysyy voimassa SYOTE_LISA_ODOTUS_S s (uudelleenyhteys ei vapauta kiria hetkeksi)."""
        with self._syote_ehto:
            kirje = self._syote_lisat.get(tunnus)
            if kirje is not None:
                kirje[1] = max(0, kirje[1] - 1)
                if kirje[1] == 0:
                    kirje[2] = time.monotonic() + SYOTE_LISA_ODOTUS_S

    def syote_lisa(self, tunnus, lisa):
        """Sivun uusi lisaviive (s; /lisa). Vain tunnetulle sivulle (syoteyhteys on tai oli aivan hetki sitten auki). True, jos sivu oli tunnettu."""
        with self._syote_ehto:
            self._syote_lisat_siivoa()
            kirje = self._syote_lisat.get(tunnus)
            if kirje is None:
                return False
            kirje[0], kirje[3] = self._lisa_rajaa(lisa), time.monotonic()
            return True

    @staticmethod
    def _lisa_rajaa(lisa):
        try:
            lisa = float(lisa)
        except (TypeError, ValueError):
            return 0.0
        return 0.0 if lisa != lisa else max(0.0, min(float(LISAVIIVE_MAX_S), lisa))

    def _syote_lisat_siivoa(self):
        """(_syote_ehto lukittuna) poistaa sivut, joiden yhteys on ollut poissa yli SYOTE_LISA_ODOTUS_S s tai joiden lisaviivetta ei ole vahvistettu pitkaan aikaan."""
        nyt = time.monotonic()
        for k in [k for k, v in self._syote_lisat.items() if (v[2] is not None and v[2] <= nyt) or nyt - v[3] > 3 * SYOTE_LISA_VANHENEE_S]:
            del self._syote_lisat[k]

    def reaaliaika_haluttu(self):
        """True, kun jokin syotesivu on valinnut lisaviiveen (> 0): kuvan pitaa silloin edeta oikealla nopeudella, ja netti_live.py:n kuvan aikataulu estaa VOD-ajon kirin (muuten lisaviiveen toisto
        kulkisi kirin vauhtia). Piilossa olevan valilehden sivu ilmoittaa lisaviiveeksi 0: kukaan ei katso, eika kiria silloin tarvitse estaa. Lisaviive on voimassa, kun sivun syoteyhteys on auki
        (tai sulkeutui alle SYOTE_LISA_ODOTUS_S s sitten) ja sivu on vahvistanut sen alle SYOTE_LISA_VANHENEE_S s sitten (hiljaa kuollut yhteys tai jaadytetty valilehti ei pida kiria tauolla)."""
        if not self._syote_lisat:
            return False
        nyt = time.monotonic()
        return any(k[0] > 0 and (k[2] is None or k[2] > nyt) and nyt - k[3] <= SYOTE_LISA_VANHENEE_S for k in list(self._syote_lisat.values()))

    def _syote_koodaaja(self):
        """Koodaa syotteen kuvat JPEG:ksi ja pitaa rengaspuskuria. Harvennus: kuva otetaan, kun edellisesta otetusta on kulunut vahintaan 1 / fps sekuntia seka video-ajassa etta seinakellossa (video-aika:
        tasainen liike, vaikka tuotanto on epatasaista; seinakello: kiri, jossa tuotanto etenee reaaliaikaa nopeammin, ei lisaa kuvia sekunnissa). Video-ajan hyppy taaksepain (ilmoituskuvat -> ensimmainen oikea
        kuva) aloittaa historian alusta."""
        while not self._lopeta.is_set():
            with self._syote_ehto:
                self._syote_ehto.wait_for(lambda: self._lopeta.is_set() or self._syote_odottaa is not None, 1.0)
                kohde, self._syote_odottaa = self._syote_odottaa, None
                if kohde is None:
                    continue
                aika, julkaistu, kuva = kohde
                v = self._syote_viim
                if v is not None and aika < v[0] - 0.5:
                    self._syote.clear()
                    self._syote_tavut, v = 0, None
                if v is not None:
                    vali = 1.0 / self.fps - 0.4 * (self._syote_dt or 0.0)
                    if aika - v[0] < vali or julkaistu - v[1] < vali:
                        continue
                self._syote_viim = (aika, julkaistu)
            try:
                tavut = self._pakkaa(kuva)
            except Exception as e:                                                           # koodausvirhe ei saa lopettaa syotetta
                print(f"web: syotteen kuvan pakkaus epaonnistui: {e!r}", file=sys.stderr, flush=True)
                continue
            if tavut is None:
                continue
            with self._syote_ehto:
                self._syote_seq += 1
                self._syote.append((self._syote_seq, aika, tavut))
                self._syote_tavut += len(tavut)
                while len(self._syote) > 1 and (self._syote[-1][1] - self._syote[0][1] > SYOTE_S or self._syote_tavut > SYOTE_TAVUT_MAX):
                    self._syote_tavut -= len(self._syote.popleft()[2])
                self._syote_ehto.notify_all()

    def syote_odota(self, viim, historia, odotus_s):
        """(kehykset, historiaa): syotteen kehykset (jarjestysnro, video-aika, JPEG), joita katsoja ei ole viela saanut; odottaa enintaan odotus_s s uusia. viim = viimeisin lahetetty jarjestysnumero
        (None = ensimmainen kutsu: kaikki puskurin kuvat historia sekuntia uusimmasta taaksepain, historiaa = True). Liian jaljessa (yli historia + SYOTE_JALKEEN_S s) oleva katsoja ohittaa uusimpiin."""
        with self._syote_ehto:
            self._syote_ehto.wait_for(lambda: self._lopeta.is_set() or (bool(self._syote) and (viim is None or self._syote[-1][0] > viim)), odotus_s)
            if not self._syote or self._lopeta.is_set():
                return [], False
            uusin = self._syote[-1][1]
            if viim is None:
                return [k for k in self._syote if k[1] >= uusin - historia], True
            lista = []
            for k in reversed(self._syote):
                if k[0] <= viim:
                    break
                lista.append(k)
            lista.reverse()
            if lista and uusin - lista[0][1] > historia + SYOTE_JALKEEN_S:
                lista = [k for k in lista if k[1] >= uusin - historia]
            return lista, False

    # ---- tila ------------------------------------------------------------------------------------------------------
    def katsojia(self, muutos=0):
        with self._lukko:
            self._katsojat += muutos
            return self._katsojat

    def tila(self):
        t = dict(kaynnissa=self.kaynnissa, kuvia=self.n_nayt, katsojia=self._katsojat, alkoi=self.alkoi, kuva_leveys=self.leveys,
                 viim_kuva_ika_s=None if self._viim_t is None else round(time.time() - self._viim_t, 2), syote_s=SYOTE_S, lisaviive_max_s=LISAVIIVE_MAX_S, syote_fps=self.fps, reaaliaika=self.reaaliaika_haluttu())
        if self.tila_fn is not None:
            try:
                t.update(self.tila_fn() or {})
            except Exception:
                pass
        return t

    def tiedostot(self):
        """[{nimi, koko, muutettu}] uusin ensin: vain ulos_kansion suoraan alla olevat tavalliset tulostiedostot."""
        if not self.ulos_kansio or not os.path.isdir(self.ulos_kansio):
            return []
        out = []
        try:
            for nimi in os.listdir(self.ulos_kansio):
                p = os.path.join(self.ulos_kansio, nimi)
                if nimi.startswith(".") or not nimi.lower().endswith(TIEDOSTO_PAATTEET) or os.path.islink(p) or not os.path.isfile(p):
                    continue
                st = os.stat(p)
                out.append(dict(nimi=nimi, koko=st.st_size, muutettu=st.st_mtime))
        except OSError:
            return []
        out.sort(key=lambda x: (-x["muutettu"], x["nimi"]))
        return out

    def tiedoston_polku(self, nimi):
        """Ladattavan tiedoston polku tai None. Nimen on oltava jokin tiedostot()-listan tiedosto (ei polkuerottimia, ei kansion ulkopuolelle osoittavia linkkeja)."""
        if not nimi or nimi != os.path.basename(nimi) or "\\" in nimi or "\0" in nimi or nimi.startswith(".") or not self.ulos_kansio:
            return None
        p = os.path.join(self.ulos_kansio, nimi)
        juuri = os.path.realpath(self.ulos_kansio)
        if os.path.islink(p) or not os.path.isfile(p) or not os.path.realpath(p).startswith(juuri + os.sep):
            return None
        return p if nimi.lower().endswith(TIEDOSTO_PAATTEET) else None

    def tilasto_tiedostot(self):
        """Heitto-CSV:t, joista pelaajatilasto voidaan laskea: [{nimi, koko, muutettu}] uusin ensin. Otsikkorivilla on oltava tilaston tarvitsemat sarakkeet (ei _kivet.csv, _pelaajat.csv eika muita CSV:ita).
        RuntimeError, jos tools/pelaajatilasto.py puuttuu."""
        pt = _tilasto_moduuli()
        out = []
        for t in self.tiedostot():
            nimi = t["nimi"]
            if not nimi.lower().endswith(".csv") or nimi.lower().endswith(TILASTO_POIS):
                continue
            p = self.tiedoston_polku(nimi)
            if p is None:
                continue
            try:
                with open(p, newline="", encoding="utf-8-sig") as fh:
                    otsikko = fh.readline()
            except (OSError, UnicodeDecodeError):
                continue
            kentat = {k.strip() for k in otsikko.replace(";", ",").split(",")}
            if all(k in kentat for k in pt.PAKOLLISET):
                out.append(t)
        return out

    def _tilasto_laskettu(self, nimi, keltainen, punainen, ohita, hidastuvuus):
        pt = _tilasto_moduuli()
        polku = self.tiedoston_polku(nimi) if nimi and nimi.lower().endswith(".csv") else None
        if polku is None:
            raise ValueError("tiedostoa ei löydy tulostiedostoista")
        return pt, pt.laske_tiedosto(polku, {"keltainen": keltainen, "punainen": punainen}, ohita_puutteelliset=ohita, hidastuvuus=hidastuvuus or "liuku")

    def tilasto(self, nimi, keltainen=None, punainen=None, ohita=False, hidastuvuus="liuku"):
        """Pelaajatilasto valitusta heitto-CSV:sta (JSON-kelpoinen dict): ryhmat (Myötä / Vasta / Hidastuvuus suureineen, kuten Excelissa), desimaalit suureittain ja pelaajarivit (Mediaani, STD ja heittomaara n
        suureittain). keltainen / punainen: pelaajien maara tai kivijako (3,3,2); puuttuvalle 4. hidastuvuus: 'liuku' (liukuvaiheen) tai 'vapaa' (vapaan lennon, kuten paneelissa).
        ValueError, jos tiedosto, maarat tai hidastuvuus eivat kelpaa."""
        pt, t = self._tilasto_laskettu(nimi, keltainen, punainen, ohita, hidastuvuus)
        return dict(tiedosto=t["tiedosto"], heittoja=t["heittoja"], hidastuvuus=hidastuvuus or "liuku", ryhmat=[dict(nimi=r, suureet=list(ss)) for r, ss in pt.RYHMAT], desimaalit=list(pt.DES_TULOSTE),
                    rivit=t["tulokset"], varoitukset=t["varoitukset"],
                    joukkueet=[dict(vari=v, pelaajia=len(lohkot), kivet=[pt.kivet_tekstina(sum(lohkot[:i]) + 1, n) for i, n in enumerate(lohkot)]) for v, lohkot in t["jaot"].items()])

    def tilasto_csv(self, nimi, keltainen=None, punainen=None, ohita=False, fi=True, hidastuvuus="liuku"):
        """(ladattavan tiedoston nimi, CSV-tavut); fi: Excelille erotin ; ja desimaalipilkku (UTF-8 BOM)."""
        pt, t = self._tilasto_laskettu(nimi, keltainen, punainen, ohita, hidastuvuus)
        return os.path.splitext(nimi)[0] + "_pelaajat.csv", pt.csv_tavuina(t["tulokset"], fi)

    def tunnistettu(self, otsake):
        """Perusautentikointi (jos salasana on asetettu)."""
        if not (self._kayttaja or self._salasana):
            return True
        if not otsake or not otsake.startswith("Basic "):
            return False
        try:
            tunnus = base64.b64decode(otsake[6:].strip()).decode("utf-8")
        except Exception:
            return False
        odotettu = f"{self._kayttaja}:{self._salasana}"
        return hmac.compare_digest(tunnus.encode("utf-8"), odotettu.encode("utf-8"))


class _Kasittelija(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "Kivenseuranta"
    sys_version = ""

    def log_message(self, *args):                                  # ei rivia jokaisesta pyynnosta lokiin
        pass

    def setup(self):
        super().setup()
        self.server.naytto.yhteys(self.connection, True)

    def finish(self):
        try:
            super().finish()
        finally:
            self.server.naytto.yhteys(self.connection, False)

    # ---- apurit ----------------------------------------------------------------------------------------------------
    def _vastaa(self, koodi, sisalto, tyyppi, lisaotsakkeet=(), head=False):
        self.send_response(koodi)
        self.send_header("Content-Type", tyyppi)
        self.send_header("Content-Length", str(len(sisalto)))
        self.send_header("Cache-Control", "no-store")
        for k, v in lisaotsakkeet:
            self.send_header(k, v)
        self.end_headers()
        if not head:
            self.wfile.write(sisalto)

    def _teksti(self, koodi, teksti, head=False):
        self._vastaa(koodi, teksti.encode("utf-8"), "text/plain; charset=utf-8", head=head)

    def _json(self, data, head=False):
        self._vastaa(200, json.dumps(data, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8", head=head)

    # ---- reitit ----------------------------------------------------------------------------------------------------
    def do_HEAD(self):
        self._kasittele(head=True)

    def do_GET(self):
        self._kasittele(head=False)

    def _kasittele(self, head):
        n = self.server.naytto
        if n._lopeta.is_set():                                         # suljettu palvelin ei vastaa (ehti vastaanottaa pyynnon juuri ennen sulkua): yhteys poikki, selain avaa uuden
            self.close_connection = True
            return
        if not n.tunnistettu(self.headers.get("Authorization")):
            self._vastaa(401, b"Kirjaudu", "text/plain; charset=utf-8", [("WWW-Authenticate", 'Basic realm="Kivenseuranta", charset="UTF-8"')], head)
            return
        polku = urllib.parse.urlsplit(self.path).path
        try:
            if polku in ("/", "/index.html"):
                self._vastaa(200, SIVU.encode("utf-8"), "text/html; charset=utf-8", head=head)
            elif polku == "/video.mjpg":
                if head:
                    self._vastaa(200, b"", "multipart/x-mixed-replace; boundary=" + MJPEG_RAJA.decode(), head=True)
                else:
                    self._mjpeg(n)
            elif polku == "/syote":
                self._syote(n, urllib.parse.urlsplit(self.path).query, head)
            elif polku == "/lisa":
                self._lisa(n, urllib.parse.urlsplit(self.path).query, head)
            elif polku == "/kuva.jpg":
                jpeg, _ = n.jpeg()
                if jpeg is None:
                    self._teksti(503, "Ei kuvaa vielä", head)
                else:
                    self._vastaa(200, jpeg, "image/jpeg", head=head)
            elif polku == "/tila":
                self._json(n.tila(), head)
            elif polku == "/loki":
                self._teksti(200, "\n".join(list(n.loki_rivit)[-LOKI_RIVEJA:]), head)
            elif polku == "/tiedostot.json":
                self._json(n.tiedostot(), head)
            elif polku == "/tilasto/tiedostot":
                self._tilasto_tiedostot(n, head)
            elif polku in ("/tilasto", "/tilasto.csv"):
                self._tilasto(n, polku, urllib.parse.urlsplit(self.path).query, head)
            elif polku in ("/tiedostot", "/tiedostot/"):
                self._tiedostolista(n, head)
            elif polku.startswith("/tiedostot/"):
                self._tiedosto(n, urllib.parse.unquote(polku[len("/tiedostot/"):]), head)
            else:
                self._teksti(404, "Ei löytynyt", head)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, TimeoutError):
            self.close_connection = True
        except Exception as e:                                         # kasittelijan virhe ei saa kaataa muuta ajoa eika tulostaa jaljitysta lokiin joka pyynnolla
            self.close_connection = True
            print(f"web: pyynto {polku} epaonnistui: {e!r}", file=sys.stderr, flush=True)
            try:
                self._teksti(500, "Palvelinvirhe", head)
            except Exception:
                pass

    def _mjpeg(self, n):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=" + MJPEG_RAJA.decode())
        self.send_header("Cache-Control", "no-store, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        n.katsojia(+1)
        edellinen, viim = None, 0.0
        try:
            while not n._lopeta.is_set():
                tauko = 1.0 / n.fps - (time.monotonic() - viim)       # kuvataajuuden ylaraja per katsoja (selain/verkko ei tarvitse 30 kuvaa/s)
                if tauko > 0:
                    time.sleep(tauko)
                jpeg, edellinen_uusi = n.jpeg(edellinen, odotus_s=2.0)
                if jpeg is None:
                    continue
                edellinen = edellinen_uusi
                self.wfile.write(b"--" + MJPEG_RAJA + b"\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n")
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
                viim = time.monotonic()
        finally:
            n.katsojia(-1)

    def _syote(self, n, kysely, head):
        """/syote?historia=<s>: aikaleimattu kuvasyote selaimen soittimelle (puskuri ja lisaviive selaimessa). Binaarikehykset: SYOTE_OTSIKKO (JPEG:n pituus, video-aika s, jarjestysnumero, liput) + JPEG;
        pituus 0 = sydankohta (ei kuvaa). historia: montako sekuntia aiempia kuvia lahetetaan heti alkuun (liput = 1); sen jalkeen kuvat tulevat sita mukaa kun ne valmistuvat."""
        q = {k: v[0] for k, v in urllib.parse.parse_qs(kysely, keep_blank_values=True).items()}
        try:
            historia = max(0.0, min(SYOTE_S, float(q.get("historia") or 0.0)))
            lisa = float(q.get("lisa") or 0.0)
        except ValueError:
            self._virhe_json(400, "historia ja lisa: sekuntien maara (historia 0 ... %d, lisa 0 ... %d)" % (SYOTE_S, LISAVIIVE_MAX_S), head)
            return
        if historia != historia:                                       # NaN
            historia = 0.0
        tunnus = q.get("id") if SYOTE_TUNNUS_RE.match(q.get("id") or "") else None        # sivun tunnus: lisaviive (kiri tauolle) seuraa sivua; ilman tunnusta sivu ei ilmoita lisaviivettaan
        if head:
            self._vastaa(200, b"", "application/octet-stream", head=True)
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Cache-Control", "no-store, private")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        n.katsojia(+1)                                                 # sivun tilarivin katsojamaara
        n.syote_katsoja(+1)
        if tunnus:
            n.syote_lisa_auki(tunnus, lisa)
        try:
            viim, sydan = None, time.monotonic()
            while not n._lopeta.is_set():
                kehykset, historiaa = n.syote_odota(viim, historia, min(1.0, SYOTE_SYDAN_S))
                if kehykset:
                    for seq, aika, tavut in kehykset:
                        self.wfile.write(SYOTE_OTSIKKO.pack(len(tavut), aika, seq & 0xFFFFFFFF, 1 if historiaa else 0))
                        self.wfile.write(tavut)
                        viim = seq
                    self.wfile.flush()
                    sydan = time.monotonic()
                elif time.monotonic() - sydan >= SYOTE_SYDAN_S:        # ei kuvia: sydankohta, jotta katkennut yhteys huomataan (molemmin puolin)
                    self.wfile.write(SYOTE_OTSIKKO.pack(0, 0.0, 0, 0))
                    self.wfile.flush()
                    sydan = time.monotonic()
        finally:
            if tunnus:
                n.syote_lisa_kiinni(tunnus)
            n.syote_katsoja(-1)
            n.katsojia(-1)

    def _lisa(self, n, kysely, head):
        """/lisa?id=<sivun tunnus>&s=<lisaviive s>: sivu ilmoittaa lisaviiveensa (napit +-1 s / +-5 s; piilossa oleva valilehti ilmoittaa 0). Lisaviive > 0: kuvan pitaa edeta reaaliajassa, joten VOD-ajon kiri
        pysaytetaan (reaaliaika_haluttu). Vastaus: {"reaaliaika": bool}. Vain tunnetulle sivulle (syoteyhteys on auki); HEAD ei muuta mitaan."""
        q = {k: v[0] for k, v in urllib.parse.parse_qs(kysely, keep_blank_values=True).items()}
        tunnus = q.get("id") or ""
        try:
            lisa = float(q.get("s"))
        except (TypeError, ValueError):
            lisa = None
        if not SYOTE_TUNNUS_RE.match(tunnus) or lisa is None or lisa != lisa:
            self._virhe_json(400, "id: sivun tunnus (kirjaimia, numeroita, - ja _, enintaan 40 merkkia), s: lisaviive sekunteina", head)
            return
        if not head:
            n.syote_lisa(tunnus, lisa)
        self._json(dict(reaaliaika=n.reaaliaika_haluttu()), head)

    def _virhe_json(self, koodi, viesti, head=False):
        self._vastaa(koodi, json.dumps(dict(virhe=viesti), ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8", head=head)

    def _tilasto_tiedostot(self, n, head):
        try:
            self._json(n.tilasto_tiedostot(), head)
        except RuntimeError as e:
            self._virhe_json(503, str(e), head)

    def _tilasto(self, n, polku, kysely, head):
        """/tilasto?tiedosto=<nimi>&kel=<pelaajat>&pun=<pelaajat>[&ohita=1][&hid=liuku|vapaa]: pelaajatilasto JSON:na; /tilasto.csv: sama CSV-tiedostona (&fi=0: pilkkuerotin ja desimaalipiste). Laskenta ei kirjoita mitaan levylle."""
        q = {k: v[0] for k, v in urllib.parse.parse_qs(kysely, keep_blank_values=True).items()}
        try:
            args = dict(nimi=q.get("tiedosto", ""), keltainen=q.get("kel") or None, punainen=q.get("pun") or None, ohita=q.get("ohita") == "1", hidastuvuus=q.get("hid") or "liuku")
            if polku == "/tilasto.csv":
                nimi, data = n.tilasto_csv(fi=q.get("fi", "1") != "0", **args)
                self._vastaa(200, data, "text/csv; charset=utf-8", [("Content-Disposition", f"attachment; filename*=UTF-8''{urllib.parse.quote(nimi)}")], head)
            else:
                self._json(n.tilasto(**args), head)
        except ValueError as e:
            self._virhe_json(400, str(e), head)
        except RuntimeError as e:
            self._virhe_json(503, str(e), head)

    def _tiedostolista(self, n, head):
        rivit = "".join(f'<li><a href="{urllib.parse.quote(t["nimi"])}">{html.escape(t["nimi"])}</a> <span>{t["koko"] / 1e6:.1f} Mt</span></li>' for t in n.tiedostot())
        sivu = f'<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Tiedostot</title><h1>Tiedostot</h1><ul>{rivit or "<li>Ei tiedostoja</li>"}</ul><p><a href="/">Takaisin</a></p>'
        self._vastaa(200, sivu.encode("utf-8"), "text/html; charset=utf-8", head=head)

    def _tiedosto(self, n, nimi, head):
        p = n.tiedoston_polku(nimi)
        if p is None:
            self._teksti(404, "Ei löytynyt", head)
            return
        koko = os.path.getsize(p)
        alku, loppu, koodi = 0, koko - 1, 200
        pyynto = self.headers.get("Range")
        if pyynto:
            m = RANGE_RE.match(pyynto.strip())
            if not m or (m.group(1) == "" and m.group(2) == ""):
                self._vastaa(416, b"", "text/plain", [("Content-Range", f"bytes */{koko}")], head=True)
                return
            if m.group(1) == "":                                       # bytes=-N: viimeiset N tavua
                alku, loppu = max(0, koko - int(m.group(2))), koko - 1
            else:
                alku = int(m.group(1))
                loppu = min(int(m.group(2)), koko - 1) if m.group(2) else koko - 1
            if alku >= koko or alku > loppu:
                self._vastaa(416, b"", "text/plain", [("Content-Range", f"bytes */{koko}")], head=True)
                return
            koodi = 206
        tyyppi = mimetypes.guess_type(nimi)[0] or "application/octet-stream"
        if tyyppi.startswith("text/"):
            tyyppi += "; charset=utf-8"
        sijoitus = "inline" if tyyppi.startswith("video/") else "attachment"
        pituus = loppu - alku + 1
        self.send_response(koodi)
        self.send_header("Content-Type", tyyppi)
        self.send_header("Content-Length", str(pituus))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Disposition", f"{sijoitus}; filename*=UTF-8''{urllib.parse.quote(nimi)}")
        if koodi == 206:
            self.send_header("Content-Range", f"bytes {alku}-{loppu}/{koko}")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if head:
            return
        with open(p, "rb") as f:
            f.seek(alku)
            jaljella = pituus
            while jaljella > 0:
                osa = f.read(min(1 << 20, jaljella))
                if not osa:
                    break
                self.wfile.write(osa)
                jaljella -= len(osa)


class JoutilasSivu:
    """Selainnakyma ajojen valissa: kun analyysia ei ole kaynnissa (valvonta odottaa uusia videoita) sama sivu, tiedostolista ja tyokalut (pelaajatilasto) ovat silti selaimessa.
    Analyysiajo avaa saman portin itse (netti_live.py --web), joten palvelin sulkee taman sivun ennen jokaista ajoa (pysayta) ja avaa sen uudelleen ajon jalkeen (kaynnista)."""

    def __init__(self, portti, ulos_kansio, loki_rivit=None, osoite="0.0.0.0", viesti="valvonta odottaa uusia videoita", loki=print):
        self.portti, self.ulos_kansio, self.loki_rivit, self.osoite, self.viesti, self._loki = portti, ulos_kansio, loki_rivit, osoite, viesti, loki
        self._naytto = None
        self._varoitettu = False

    def kaynnista(self):
        """Avaa sivun (jo auki: ei tee mitaan); False, jos porttia ei saatu auki (yritetaan seuraavalla kerralla)."""
        if self._naytto is not None:
            return True
        try:
            n = WebNaytto(self.portti, self.ulos_kansio, osoite=self.osoite, tila=lambda: dict(joutilas=True, otsikko=self.viesti), loki_rivit=self.loki_rivit)
        except SystemExit as e:                                    # portti varattu (esim. edellinen ajo sulkeutuu viela): ajoa ei saa kaataa
            if not self._varoitettu:
                self._loki(f"selainnakyma ajojen valissa ei kaynnistynyt: {e}")
                self._varoitettu = True
            return False
        n.kaynnissa = False
        n._julkaise(odotuskuva("Ei analyysia kaynnissa"), laske=False)
        self._naytto = n
        self._varoitettu = False
        return True

    def pysayta(self):
        """Vapauttaa portin (idempotentti)."""
        n, self._naytto = self._naytto, None
        if n is not None:
            n.sulje()

    @property
    def avoinna(self):
        return self._naytto is not None

    @property
    def avattu_portti(self):
        return None if self._naytto is None else self._naytto.portti
