"""Selainnakyma palvelinajoon (ei ikkunaa): seurannan kuva, ajon lokirivit, tilatiedot ja tulostiedostojen lataus.

netti_live.py --web PORTTI korvaa ikkunan (Naytto) talla: WebNaytto tarjoaa saman rajapinnan (anna, nayta_heti, paasilmukka, sulje), joten ajosilmukat eivat tieda eroa.

  /                  sivu: kuva, tila, loki, tiedostot
  /video.mjpg        kuva MJPEG-virtana (<img>-elementti toimii kaikissa selaimissa; JPEG pakataan vain, kun joku katsoo)
  /kuva.jpg          tuorein kuva yksittaisena JPEG:na
  /tila              JSON: ajon tila, kuvien maara, katsojat
  /loki              viimeiset rivit konsolitulosteesta (teksti)
  /tiedostot/        tulostiedostot (mp4, csv); /tiedostot/<nimi> lataa (mp4 toistuu selaimessa, Range-tuki)
  /tiedostot.json    sama JSON:na
  /tilasto/tiedostot pelaajatilastoon kelpaavat heitto-CSV:t (JSON); nappi "Pelaajatilasto" sivulla
  /tilasto           pelaajakohtainen mediaani ja STD valitusta heitto-CSV:sta (JSON): ?tiedosto=<nimi>&kel=<pelaajat>&pun=<pelaajat>[&ohita=1] (tools/pelaajatilasto.py)
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
.lomake label:first-child{flex:1 1 100%}.lomake select{width:100%;min-width:0}.lomake input[type=text]{width:7em}
#ts-tulos{overflow-x:auto}#ts-tulos .himmea{color:var(--himmea);margin:8px 0}
table.ts{border-collapse:collapse;font-size:14px;font-variant-numeric:tabular-nums}table.ts th,table.ts td{border:1px solid var(--reuna);padding:4px 8px;white-space:nowrap}
table.ts th{background:var(--laatikko);font-weight:600}table.ts td.n{text-align:right}
.huom{background:var(--laatikko);border:1px solid var(--reuna);border-radius:8px;padding:8px 12px;margin:12px 0 0;color:var(--himmea);font-size:13px}.virhe{color:#c62828;margin:8px 0}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]) .virhe{color:#ff8a80}}
</style></head><body><main>
<div class="ylapalkki"><h1>Kivenseuranta</h1><button type="button" class="nappi" id="ts-avaa">Pelaajatilasto</button></div>
<img id="kuva" alt="Seurannan kuva" style="display:none">
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
    const kuva=$("kuva");
    if(t.joutilas){kuva.style.display="none";if(kuva.getAttribute("src"))kuva.removeAttribute("src")}
    else{kuva.style.display="";if(t.alkoi!==edAlkoi||!kuva.getAttribute("src"))kuva.src="video.mjpg?"+Date.now()}
    edAlkoi=t.alkoi;
    $("tila").innerHTML=t.joutilas?"<b>Ei analyysia käynnissä</b>"+(t.otsikko?" &middot; "+esc(t.otsikko):""):(t.kaynnissa?"<b>Ajo kaynnissa</b>":"<b>Ajo paattynyt</b>")+" &middot; kuvia "+t.kuvia+" &middot; katsojia "+t.katsojia+(t.viim_kuva_ika_s!=null?" &middot; viimeisin kuva "+t.viim_kuva_ika_s.toFixed(1)+" s sitten":"")+(t.kaynnissa&&t.viim_kuva_ika_s!=null&&t.viim_kuva_ika_s>30?" &middot; <b>kuva ei etene: katso loki (Jumivahti)</b>":"")+((t.otsikko||t.lahde)?" &middot; "+esc(t.otsikko||t.lahde):"");
    document.title=t.otsikko?"Kivenseuranta: "+t.otsikko:"Kivenseuranta";
    if(t.kuva_leveys)$("kuva").style.maxWidth=t.kuva_leveys+"px";
    const l=$("loki"),alhaalla=l.scrollTop+l.clientHeight>=l.scrollHeight-30;
    l.textContent=await (await fetch("loki",{cache:"no-store"})).text();
    if(alhaalla)l.scrollTop=l.scrollHeight;
    const f=await (await fetch("tiedostot.json",{cache:"no-store"})).json();
    $("tiedostot").innerHTML=f.length?f.map(x=>"<li><a href=\\"tiedostot/"+encodeURIComponent(x.nimi)+"\\">"+x.nimi.replace(/&/g,"&amp;").replace(/</g,"&lt;")+"</a><span class=koko>"+koko(x.koko)+"</span></li>").join(""):"<li>Ei tiedostoja viela</li>";
  }catch(e){$("tila").textContent="Yhteys palvelimeen katkesi (ajo paattynyt tai kontti sammutettu)."}
}
let tsTulos=null;
function tsMuisti(k,v){try{if(v===undefined)return localStorage.getItem(k);localStorage.setItem(k,v)}catch(e){return null}}
function tsKysely(){return "tiedosto="+encodeURIComponent($("ts-tiedosto").value)+"&kel="+encodeURIComponent($("ts-kel").value.trim())+"&pun="+encodeURIComponent($("ts-pun").value.trim())+($("ts-ohita").checked?"&ohita=1":"")}
function tsNum(x,des){return x==null?"-":x.toFixed(des).replace(".",",")}
async function tsAvaa(){
  $("tilasto").hidden=false;document.body.style.overflow="hidden";
  const kel=tsMuisti("ts_kel"),pun=tsMuisti("ts_pun");
  if(kel)$("ts-kel").value=kel;
  if(pun)$("ts-pun").value=pun;
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
  tsMuisti("ts_kel",$("ts-kel").value.trim());tsMuisti("ts_pun",$("ts-pun").value.trim());
  try{
    const r=await fetch("tilasto?"+tsKysely(),{cache:"no-store"}),d=await r.json();
    if(!r.ok){$("ts-tulos").innerHTML="<p class='virhe'>"+esc(d.virhe||("Virhe "+r.status))+"</p>";return}
    tsNayta(d);
  }catch(e){$("ts-tulos").innerHTML="<p class='virhe'>Laskenta epäonnistui: "+esc(e.message||e)+"</p>"}
  finally{nappi.disabled=false}
}
function tsNayta(d){
  tsTulos=d;
  const o=d.otsikot,des=d.desimaalit,sar=x=>"<th>"+esc(x)+"</th>";
  let h="<p class='himmea'>"+esc(d.tiedosto)+": "+d.heittoja+" heittoa; "+d.joukkueet.map(j=>esc(j.vari)+" "+j.pelaajia+" pelaajaa (kivet "+j.kivet.join(", ")+")").join("; ")+"</p>";
  h+="<table class='ts'><thead><tr><th colspan='4'></th><th colspan='3'>Mediaani</th><th colspan='3'>STD</th></tr><tr><th>joukkue</th><th>pelaaja</th><th>kivet</th><th>heittoja</th>"+o.map(sar).join("")+o.map(sar).join("")+"</tr></thead><tbody>";
  for(const r of d.rivit)h+="<tr><td>"+esc(r.joukkue)+"</td><td class='n'>"+r.pelaaja+"</td><td>"+esc(r.kivet)+"</td><td class='n'>"+r.heittoja+"</td>"+r.mediaani.map((x,i)=>"<td class='n'>"+tsNum(x,des[i])+"</td>").join("")+r.std.map((x,i)=>"<td class='n'>"+tsNum(x,des[i])+"</td>").join("")+"</tr>";
  h+="</tbody></table>";
  if(d.varoitukset.length)h+="<div class='huom'>"+d.varoitukset.map(v=>"<div>Huom: "+esc(v)+"</div>").join("")+"</div>";
  h+="<p><button type='button' class='nappi toinen' id='ts-kopioi'>Kopioi Exceliin</button> <a class='nappi toinen' id='ts-csv' href='#'>Lataa CSV (Excel)</a></p>";
  $("ts-tulos").innerHTML=h;
  $("ts-csv").href="tilasto.csv?"+tsKysely()+"&fi=1";                      // osoite DOM:illa (ei merkkijonoa attribuuttiin): tiedostonimen heittomerkki ei riko linkkia
  $("ts-kopioi").onclick=()=>tsKopioi($("ts-kopioi"));
}
function tsTsv(d){
  const T=String.fromCharCode(9),N=String.fromCharCode(10),o=d.otsikot,des=d.desimaalit;
  const rivit=[["","","","","Mediaani","","","STD","",""],["joukkue","pelaaja","kivet","heittoja"].concat(o,o)];
  for(const r of d.rivit)rivit.push([r.joukkue,r.pelaaja,r.kivet,r.heittoja].concat(r.mediaani.map((x,i)=>x==null?"":tsNum(x,des[i]+1)),r.std.map((x,i)=>x==null?"":tsNum(x,des[i]+1))));
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
        self._anna_kuva = None
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
    def anna(self, kuva):
        """Seuraava nayttettava kuva (mista tahansa saikeesta); edellinen nayttamaton kuva korvautuu."""
        with self._lukko:
            self._anna_kuva = kuva
        self._uusi.set()

    def nayta_heti(self, kuva):
        """Synkroninen (simulaatio): kuva julkaistaan heti; palauttaa True vain jos lopetus on pyydetty (selaimesta ei voi lopettaa)."""
        self._julkaise(kuva)
        return False

    def paasilmukka(self, lopeta, valmis, ilmoitus=None):
        """Paasaie: julkaisee annetut kuvat, kunnes lopeta (Event) asetetaan tai valmis() on tosi ja kuvat on julkaistu. ilmoitus(): kuva (tai None) ennen ensimmaista oikeaa kuvaa."""
        viim_ilm = 0.0
        while not lopeta.is_set():
            self._uusi.wait(0.05)
            self._uusi.clear()
            with self._lukko:
                kuva, self._anna_kuva = self._anna_kuva, None
            if kuva is not None:
                self._julkaise(kuva)
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
    def _julkaise(self, kuva, laske=True):
        with self._ehto:
            self._kuva = kuva
            self._id += 1
            self._viim_t = time.time()
            if laske:
                self.n_nayt += 1
            self._ehto.notify_all()

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
            pieni = kuva
            if kuva.shape[1] > self.leveys:
                pieni = cv2.resize(kuva, (self.leveys, int(round(self.leveys * kuva.shape[0] / kuva.shape[1]))), interpolation=cv2.INTER_AREA)
            ok, puskuri = cv2.imencode(".jpg", pieni, [cv2.IMWRITE_JPEG_QUALITY, self.laatu])
            if not ok:
                return None, id_
            self._jpeg = (id_, puskuri.tobytes())
            return self._jpeg[1], id_

    # ---- tila ------------------------------------------------------------------------------------------------------
    def katsojia(self, muutos=0):
        with self._lukko:
            self._katsojat += muutos
            return self._katsojat

    def tila(self):
        t = dict(kaynnissa=self.kaynnissa, kuvia=self.n_nayt, katsojia=self._katsojat, alkoi=self.alkoi, kuva_leveys=self.leveys,
                 viim_kuva_ika_s=None if self._viim_t is None else round(time.time() - self._viim_t, 2))
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

    def _tilasto_laskettu(self, nimi, keltainen, punainen, ohita):
        pt = _tilasto_moduuli()
        polku = self.tiedoston_polku(nimi) if nimi and nimi.lower().endswith(".csv") else None
        if polku is None:
            raise ValueError("tiedostoa ei löydy tulostiedostoista")
        return pt, pt.laske_tiedosto(polku, {"keltainen": keltainen, "punainen": punainen}, ohita_puutteelliset=ohita)

    def tilasto(self, nimi, keltainen=None, punainen=None, ohita=False):
        """Pelaajatilasto valitusta heitto-CSV:sta (JSON-kelpoinen dict). keltainen / punainen: pelaajien maara tai kivijako (3,3,2); puuttuvalle 4. ValueError, jos tiedosto tai maarat eivat kelpaa."""
        pt, t = self._tilasto_laskettu(nimi, keltainen, punainen, ohita)
        return dict(tiedosto=t["tiedosto"], heittoja=t["heittoja"], otsikot=list(pt.MITAT), desimaalit=list(pt.DES_TULOSTE), rivit=t["tulokset"], varoitukset=t["varoitukset"],
                    joukkueet=[dict(vari=v, pelaajia=len(lohkot), kivet=[pt.kivet_tekstina(sum(lohkot[:i]) + 1, n) for i, n in enumerate(lohkot)]) for v, lohkot in t["jaot"].items()])

    def tilasto_csv(self, nimi, keltainen=None, punainen=None, ohita=False, fi=True):
        """(ladattavan tiedoston nimi, CSV-tavut); fi: Excelille erotin ; ja desimaalipilkku (UTF-8 BOM)."""
        pt, t = self._tilasto_laskettu(nimi, keltainen, punainen, ohita)
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

    def _virhe_json(self, koodi, viesti, head=False):
        self._vastaa(koodi, json.dumps(dict(virhe=viesti), ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8", head=head)

    def _tilasto_tiedostot(self, n, head):
        try:
            self._json(n.tilasto_tiedostot(), head)
        except RuntimeError as e:
            self._virhe_json(503, str(e), head)

    def _tilasto(self, n, polku, kysely, head):
        """/tilasto?tiedosto=<nimi>&kel=<pelaajat>&pun=<pelaajat>[&ohita=1]: pelaajatilasto JSON:na; /tilasto.csv: sama CSV-tiedostona (&fi=0: pilkkuerotin ja desimaalipiste). Laskenta ei kirjoita mitaan levylle."""
        q = {k: v[0] for k, v in urllib.parse.parse_qs(kysely, keep_blank_values=True).items()}
        try:
            args = dict(nimi=q.get("tiedosto", ""), keltainen=q.get("kel") or None, punainen=q.get("pun") or None, ohita=q.get("ohita") == "1")
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
