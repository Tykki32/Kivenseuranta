"""Testi_07_01: seurannan nakyma puhelimen selaimella samassa verkossa (wifi).

v7.5 (kayttajan pyynto, kone mahdollisimman vahan kuormitettuna):
  * EI videota eika tallennusta: debug-nakyma (sama kuva kuin debug-videossa: kuva + kivien aariviivat + tulospaneelit)
    piirretaan omassa taustasaikeessaan KERRAN SEKUNNISSA (seinakello) ja pakataan JPEG:ksi.
  * Pieni HTTP-palvelin (Pythonin http.server, oma saie) jakaa sivun ja viimeisimman kuvan; sivu hakee uuden kuvan
    kerran sekunnissa. Jos piirto on viela kesken kun seuraava kuva pyydetaan, pyynto ohitetaan (paasaie ei koskaan odota).

Kaytto: main.py --live --katselu [PORTTI]  (oletus 8080). Osoite tulostetaan kaynnistyksessa, esim. http://192.168.1.23:8080/
Windows kysyy ensimmaisella kerralla palomuurin luvan Pythonille: salli "Yksityiset verkot".
"""

import os
import json
import time
import socket
import threading
import base64
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

import cv2

KUVA_VALI_S = float(os.environ.get("KATSELU_VALI_S", "1.0"))        # kuinka usein nakyma piirretaan (s)
KUVA_LEVEYS = int(os.environ.get("KATSELU_LEVEYS", "1100"))          # JPEG-kuvan leveys (px); debug-kuva on 1468 px
KUVA_LAATU = int(os.environ.get("KATSELU_LAATU", "75"))              # JPEG-laatu

_server = None

# v8.5: naytto paalle -varakeino. Wake Lock -rajapinta toimii vain https:lla (ja localhostilla); kotiverkon http-osoitteessa
# selain ei anna sita. Varakeinona sivulla toistetaan silmukassa pienta (16x16, 2 s, 2,5 kt) mykistettya videota: kun video
# toistuu, puhelin/tabletti ei sammuta nayttoa (sama tekniikka kuin NoSleep.js-kirjastossa). Kaynnistyy ensimmaisesta napautuksesta.
_HEREILLA_MP4 = base64.b64decode(
    "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"
    "IC0gQ29weWxlZnQgMjAwMy0yMDIzIC0gaHR0cDovL3d3dy52aWRlb2xhbi5vcmcveDI2NC5odG1sIC0gb3B0aW9uczogY2FiYWM9"
    "MCByZWY9MyBkZWJsb2NrPTE6MDowIGFuYWx5c2U9MHgxOjB4MTExIG1lPWhleCBzdWJtZT03IHBzeT0xIHBzeV9yZD0xLjAwOjAu"
    "MDAgbWl4ZWRfcmVmPTEgbWVfcmFuZ2U9MTYgY2hyb21hX21lPTEgdHJlbGxpcz0xIDh4OGRjdD0wIGNxbT0wIGRlYWR6b25lPTIx"
    "LDExIGZhc3RfcHNraXA9MSBjaHJvbWFfcXBfb2Zmc2V0PS0yIHRocmVhZHM9MSBsb29rYWhlYWRfdGhyZWFkcz0xIHNsaWNlZF90"
    "aHJlYWRzPTAgbnI9MCBkZWNpbWF0ZT0xIGludGVybGFjZWQ9MCBibHVyYXlfY29tcGF0PTAgY29uc3RyYWluZWRfaW50cmE9MCBi"
    "ZnJhbWVzPTAgd2VpZ2h0cD0wIGtleWludD0yNTAga2V5aW50X21pbj01IHNjZW5lY3V0PTQwIGludHJhX3JlZnJlc2g9MCByY19s"
    "b29rYWhlYWQ9NDAgcmM9Y3JmIG1idHJlZT0xIGNyZj0yMy4wIHFjb21wPTAuNjAgcXBtaW49MCBxcG1heD02OSBxcHN0ZXA9NCBp"
    "cF9yYXRpbz0xLjQwIGFxPTE6MS4wMACAAAAAC2WIhAR8mKAANiOAARggBwEYIAcAAAAGQZo4CPqAARggBwEYIAcAAAAGQZpUAj6g"
    "ARggBwAAAAVBmmAR9QEYIAcBGCAHAAAABUGagBH1ARggBwAAAAVBmqAR9QEYIAcBGCAHAAAABUGawBH1ARggBwAAAAVBmuAR9QEY"
    "IAcBGCAHAAAABUGbABD1ARggBwEYIAcAAAAFQZsgP9QBGCAH"
)

_HEREILLA_WEBM = base64.b64decode(        # sama WebM-muodossa (VP8) selaimille joilla ei ole H.264:aa
    "GkXfo59ChoEBQveBAULygQRC84EIQoKEd2VibUKHgQRChYECGFOAZwEAAAAAAAg5EU2bdLpNu4tTq4QVSalmU6yBoU27i1OrhBZU"
    "rmtTrIHYTbuMU6uEElTDZ1OsggGETbuMU6uEHFO7a1Osgggj7AEAAAAAAABZAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAVSalmsirXsYMPQkBNgI1M"
    "YXZmNjAuMTYuMTAwV0GNTGF2ZjYwLjE2LjEwMESJiECfYAAAAAAAFlSua0CmrgEAAAAAAAA414EBc8WIw2v2ZWJTzkicgQAitZyD"
    "dW5kiIEAhoVWX1ZQOIOBASPjg4QL68IA4ImwgRC6gRCagQKuAQAAAAAAAFzXgQJzxYjAgZVj49Up2ZyBACK1nIN1bmSIgQCGhkFf"
    "T1BVU1aqg2MuoFa7hATEtACDgQLhkZ+BAbWIQOdwAAAAAABiZIEQY6KTT3B1c0hlYWQBATgBgLsAAAAAABJUw2dA1nNzoGPAgGfI"
    "mkWjh0VOQ09ERVJEh41MYXZmNjAuMTYuMTAwc3PWY8CLY8WIw2v2ZWJTzkhnyKFFo4dFTkNPREVSRIeUTGF2YzYwLjMxLjEwMiBs"
    "aWJ2cHhnyKFFo4hEVVJBVElPTkSHkzAwOjAwOjAyLjAwMDAwMDAwMABzc9djwItjxYjAgZVj49Up2WfIokWjh0VOQ09ERVJEh5VM"
    "YXZjNjAuMzEuMTAyIGxpYm9wdXNnyKFFo4hEVVJBVElPTkSHkzAwOjAwOjAyLjAwODAwMDAwMAAfQ7Z1Rb3ngQCji4IAAIAIC+S5"
    "oLyEo6OBAACAEAIAnQEqEAAQAABHCIWFiIWEiAICAAwNYAD+/6tQgKOKggAVgAgHxrMOxqOKggApgAgHxrMOxqOKggA9gAgHxrMO"
    "xqOKggBRgAgHxrMOxqOKggBlgAgHxrMOxqOKggB5gAgHxrMOxqOKggCNgAgHxrMOxqOKggChgAgHxrMOxqOKggC1gAgHxrMOxqOK"
    "ggDJgAgHxrMOxqOVgQDIALEBAAEQEAAYABhYL/QACAAAo4qCAN2ACAfGsw7Go4qCAPGACAfGsw7Go4qCAQWACAfGsw7Go4qCARmA"
    "CAfGsw7Go4qCAS2ACAfGsw7Go4qCAUGACAfGsw7Go4qCAVWACAfGsw7Go4qCAWmACAfGsw7Go4qCAX2ACAfGsw7Go4qCAZGACAfG"
    "sw7Go5WBAZAAsQEAARAQABgAGFgv9AAIAACjioIBpYAIB8azDsajioIBuYAIB8azDsajioIBzYAIB8azDsajioIB4YAIB8azDsaj"
    "ioIB9YAIB8azDsajioICCYAIB8azDsajioICHYAIB8azDsajioICMYAIB8azDsajioICRYAIB8azDsajioICWYAIB8azDsajlYEC"
    "WACxAQABEBAAGAAYWC/0AAgAAKOKggJtgAgHxrMOxqOKggKBgAgHxrMOxqOKggKVgAgHxrMOxqOKggKpgAgHxrMOxqOKggK9gAgH"
    "xrMOxqOKggLRgAgHxrMOxqOKggLlgAgHxrMOxqOKggL5gAgHxrMOxqOKggMNgAgHxrMOxqOKggMhgAgHxrMOxqOVgQMgALEBAAEQ"
    "EAAYABhYL/QACAAAo4qCAzWACAfGsw7Go4qCA0mACAfGsw7Go4qCA12ACAfGsw7Go4qCA3GACAfGsw7Go4qCA4WACAfGsw7Go4qC"
    "A5mACAfGsw7Go4qCA62ACAfGsw7Go4qCA8GACAfGsw7Go4qCA9WACAfGsw7Go4qCA+mACAfGsw7Go5WBA+gAsQEAARAQABgAGFgv"
    "9AAIAACjioID/YAIB8azDsajioIEEYAIB8azDsajioIEJYAIB8azDsajioIEOYAIB8azDsajioIETYAIB8azDsajioIEYYAIB8az"
    "DsajioIEdYAIB8azDsajioIEiYAIB8azDsajioIEnYAIB8azDsajioIEsYAIB8azDsajlYEEsACxAQABEBAAGAAYWC/0AAgAAKOK"
    "ggTFgAgHxrMOxqOKggTZgAgHxrMOxqOKggTtgAgHxrMOxqOKggUBgAgHxrMOxqOKggUVgAgHxrMOxqOKggUpgAgHxrMOxqOKggU9"
    "gAgHxrMOxqOKggVRgAgHxrMOxqOKggVlgAgHxrMOxqOKggV5gAgHxrMOxqOVgQV4ALEBAAEQEBRgAGFgv9AAIAAAo4qCBY2ACAfG"
    "sw7Go4qCBaGACAfGsw7Go4qCBbWACAfGsw7Go4qCBcmACAfGsw7Go4qCBd2ACAfGsw7Go4qCBfGACAfGsw7Go4qCBgWACAfGsw7G"
    "o4qCBhmACAfGsw7Go4qCBi2ACAfGsw7Go4qCBkGACAfGsw7Go5WBBkAAsQEAARAQABgAGFgv9AAIAACjioIGVYAIB8azDsajioIG"
    "aYAIB8azDsajioIGfYAIB8azDsajioIGkYAIB8azDsajioIGpYAIB8azDsajioIGuYAIB8azDsajioIGzYAIB8azDsajioIG4YAI"
    "B8azDsajioIG9YAIB8azDsajioIHCYAIB8azDsajlYEHCACxAQABEBAAGAAYWC/0AAgAAKOKggcdgAgHxrMOxqOKggcxgAgHxrMO"
    "xqOKggdFgAgHxrMOxqOKggdZgAgHxrMOxqOKggdtgAgHxrMOxqOKggeBgAgHxrMOxqOKggeVgAgHxrMOxqOKggepgAgHxrMOxqOK"
    "gge9gAgHxrMOxqCToYqCB9EACAfGsw7GdaKEAM3+YBxTu2uRu4+zgQC3iveBAfGCAmDwgRA="
)


def active():
    return _server


def lan_ip():
    """Koneen osoite lahiverkossa (ei laheta mitaan: UDP-'yhteys' vain valitsee reitin)."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


_PAGE = """<!doctype html>
<html lang="fi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Kivenseuranta</title>
<style>
 html,body{margin:0;background:#111;color:#ddd;font:14px system-ui,sans-serif}
 html,body{height:100%;overflow:hidden}
 body{display:flex;flex-direction:column}
 #k{display:block;width:100vw;flex:1 1 auto;min-height:0;object-fit:contain;background:#000}
 #t{flex:0 0 auto;padding:6px 10px;background:rgba(0,0,0,.7)}
 #t label{white-space:nowrap;margin-right:8px}
 #t input[type=checkbox]{width:auto;margin:0 3px 0 0;vertical-align:middle}
 #h{position:fixed;right:0;bottom:0;width:2px;height:2px;opacity:0.01;pointer-events:none}
 #o{position:fixed;left:0;top:0;pointer-events:none}
 #t input{width:3.2em;font:inherit;background:#222;color:#fff;border:1px solid #666;border-radius:3px;padding:1px 3px;margin:0 6px 0 2px}
</style></head><body>
<img id="k" alt="">
<canvas id="o"></canvas>
<div id="t">Alku:<input id="a" type="number" inputmode="numeric" min="0">Loppu:<input id="b" type="number" inputmode="numeric" min="0"><label><input id="cv" type="checkbox" checked>Vasenkätinen</label><label><input id="co" type="checkbox" checked>Oikeakätinen</label><span id="s">Yhdistetaan...</span></div>
<video id="h" muted loop playsinline preload="auto"><source src="hereilla.webm" type="video/webm"><source src="hereilla.mp4" type="video/mp4"></video>
<script>
const k=document.getElementById('k'),t=document.getElementById('s'),o=document.getElementById('o');
const ia=document.getElementById('a'),ib=document.getElementById('b');
// Testi_08_02 t10: korostus - paneelilaatikot joiden ika (s kaukohogin ylityksesta) on Alku..Loppu
ia.value='25';ib.value='40';
try{const a0=localStorage.getItem('alku'),b0=localStorage.getItem('loppu');if(a0!==null)ia.value=a0;if(b0!==null)ib.value=b0}catch(e){}
for(const el of [ia,ib]){el.addEventListener('click',e=>e.stopPropagation());
  el.addEventListener('input',()=>{try{localStorage.setItem('alku',ia.value);localStorage.setItem('loppu',ib.value)}catch(e){}piirra();paivita()})}
// t12: vasenkatinen = vasemman hakin viiva ja luku (punainen), oikeakatinen = oikean (vihrea); valinta myos koneelle paneelin lukuja varten
const cv=document.getElementById('cv'),co=document.getElementById('co');
try{const v0=localStorage.getItem('vasen'),o0=localStorage.getItem('oikea');if(v0!==null)cv.checked=v0==='1';if(o0!==null)co.checked=o0==='1'}catch(e){}
for(const el of [cv,co]){el.parentNode.addEventListener('click',e=>e.stopPropagation());
  el.addEventListener('change',()=>{try{localStorage.setItem('vasen',cv.checked?'1':'0');localStorage.setItem('oikea',co.checked?'1':'0')}catch(e){}
    piirra();paivita()})}
let laatikot=[];
function piirra(){
  const W=window.innerWidth,H=window.innerHeight;o.width=W;o.height=H;const c=o.getContext('2d');c.clearRect(0,0,W,H);
  const a=parseFloat(ia.value),b=parseFloat(ib.value);if(isNaN(a)||isNaN(b)||!k.naturalWidth)return;
  const r=k.getBoundingClientRect(),s=Math.min(r.width/k.naturalWidth,r.height/k.naturalHeight);
  const ox=r.left+(r.width-k.naturalWidth*s)/2,oy=r.top+(r.height-k.naturalHeight*s)/2;
  c.lineWidth=5;c.strokeStyle='#ff30ff';c.fillStyle='rgba(255,48,255,0.18)';
  const P=q=>[ox+q[0]*s,oy+q[1]*s];
  for(const q of laatikot){if(q.ika>=Math.min(a,b)&&q.ika<=Math.max(a,b)){
    const x=ox+q.x*s,y=oy+q.y*s,w=q.w*s,h=q.h*s;c.fillRect(x,y,w,h);c.strokeRect(x-2.5,y-2.5,w+5,h+5);   // reunus laatikon ulkopuolelle
    }}}

window.addEventListener('resize',piirra);k.addEventListener('load',piirra);   // kuvan vaihdon jalkeen koko tiedossa vasta load-tapahtumassa
let last=-1,busy=false;
// v7.6: napautus = koko naytto (selaimen osoitepalkki piiloon), uusi napautus palauttaa
function kokoNaytto(){const d=document,e=d.documentElement;
  if(d.fullscreenElement||d.webkitFullscreenElement){(d.exitFullscreen||d.webkitExitFullscreen).call(d)}
  else{const f=e.requestFullscreen||e.webkitRequestFullscreen;if(f)f.call(e,{navigationUI:'hide'})}}
// v8.5: naytto paalla - Wake Lock (vain https) tai varakeinona silmukkavideo; kaynnistyy ensimmaisesta napautuksesta
let lukko=null,hereilla='';
async function pidaHereilla(){
  if('wakeLock' in navigator){try{lukko=await navigator.wakeLock.request('screen');hereilla='n\u00e4ytt\u00f6 p\u00e4\u00e4ll\u00e4';return}catch(e){}}
  const h=document.getElementById('h');h.muted=true;
  try{await h.play();hereilla='n\u00e4ytt\u00f6 p\u00e4\u00e4ll\u00e4 (video)'}catch(e){hereilla='n\u00e4ytt\u00f6 voi sammua'}}
document.addEventListener('visibilitychange',()=>{if(document.visibilityState==='visible'&&hereilla)pidaHereilla()});
document.body.addEventListener('click',()=>{kokoNaytto();pidaHereilla()});
async function paivita(){
  if(busy)return;busy=true;
  try{
    const r=await fetch('tila.json?v='+(cv.checked?1:0)+'&o='+(co.checked?1:0)+'&a='+encodeURIComponent(ia.value)+'&b='+encodeURIComponent(ib.value),{cache:'no-store'});const j=await r.json();
    let txt=j.tila;
    if(j.kuva<0){txt+=' (kuva alkaa kun seuranta alkaa)'}
    else if(j.kuva!==last){
      await new Promise(res=>{const im=new Image();im.onload=()=>{k.src=im.src;res()};im.onerror=res;
        im.src='kuva.jpg?n='+j.kuva});
      last=j.kuva}
    laatikot=j.laatikot||[];piirra();
    if(j.ika>3)txt+=' - kuva '+Math.round(j.ika)+' s vanha';
    if(!(document.fullscreenElement||document.webkitFullscreenElement))txt+=' \u00b7 napauta = koko n\u00e4ytt\u00f6';
    if(hereilla)txt+=' \u00b7 '+hereilla;
    t.textContent=txt;
  }catch(e){t.textContent='Ei yhteytta koneeseen'}
  busy=false}
setInterval(paivita,1000);paivita();
</script></body></html>
"""


class _Handler(BaseHTTPRequestHandler):
    server_version = "Kivenseuranta"

    def log_message(self, *a):          # ei lokia konsoliin
        pass

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        srv = self.server.katselu
        path = self.path.split("?", 1)[0]
        try:
            if path in ("/", "/index.html"):
                self._send(200, "text/html; charset=utf-8", _PAGE.encode("utf-8"))
            elif path == "/tila.json":
                q = dict(x.split("=", 1) for x in self.path.split("?", 1)[1].split("&") if "=" in x) if "?" in self.path else {}
                if "v" in q and "o" in q:          # t12: puhelimen vasen-/oikeakatinen-valinta paneelin liukulukuihin
                    srv.nayta = (q["v"] == "1", q["o"] == "1")
                try:                                # t13: Alku/Loppu -> korostettujen liukuviivat piirretaan koneella
                    _a, _b = float(q["a"]), float(q["b"])
                    srv.korostus = (min(_a, _b), max(_a, _b))
                except (KeyError, ValueError):
                    if "a" in q:
                        srv.korostus = None
                self._send(200, "application/json", json.dumps(srv.status()).encode("utf-8"))
            elif path == "/hereilla.webm":
                self._send(200, "video/webm", _HEREILLA_WEBM)
            elif path == "/hereilla.mp4":
                self._send(200, "video/mp4", _HEREILLA_MP4)
            elif path == "/kuva.jpg":
                jpg, _n, _t = srv.latest()
                if jpg is None:
                    self._send(404, "text/plain", b"ei viela")
                else:
                    self._send(200, "image/jpeg", jpg)
            else:
                self._send(404, "text/plain", b"ei loydy")
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass


class KatseluServer:
    def __init__(self, port):
        self.port = int(port)
        self._lock = threading.Lock()
        self._state = "Kaynnistyy"
        self._jpg = None
        self._boxes = []          # Testi_08_02 t10: paneelilaatikot (x, y, w, h JPEG-pikseleina + ika) korostusta varten
        self._boxes_t = 0.0
        self.nayta = (True, True)  # t12: (vasenkatinen, oikeakatinen) puhelimen valintaruuduista
        self.korostus = None       # t13: (alku, loppu) s puhelimesta; liukuviivat korostetuille laatikoille
        self._n = -1
        self._t = 0.0
        self.httpd = ThreadingHTTPServer(("0.0.0.0", self.port), _Handler)
        self.httpd.daemon_threads = True
        self.httpd.katselu = self
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True, name="katselu-http")
        self._thread.start()
        # piirtosaie: yksi tyopaikka (uusin korvaa), paasaie ei koskaan odota
        self._job = None
        self._job_evt = threading.Event()
        self._busy = False
        self._next_t = 0.0
        self.n_rendered = 0
        self.t_render = 0.0
        self._rthread = threading.Thread(target=self._render_loop, daemon=True, name="katselu-piirto")
        self._rthread.start()

    # --- tila ---
    def set_state(self, text):
        with self._lock:
            self._state = str(text)

    def status(self):
        with self._lock:
            now = time.time()
            lt = [dict(x=b["x"], y=b["y"], w=b["w"], h=b["h"], viivat=b.get("viivat", []), risti=b.get("risti"),
                       ika=int(now - b["wall"]) if b.get("wall") is not None else int(b["ika"] + (now - self._boxes_t)))
                  for b in self._boxes]
            return {"tila": self._state, "kuva": self._n, "ika": (now - self._t) if self._n >= 0 else None, "laatikot": lt}

    def latest(self):
        with self._lock:
            return self._jpg, self._n, self._t

    # --- piirto ---
    def due(self):
        """True kun on aika piirtaa uusi nakyma (kerran KUVA_VALI_S:ssa) eika edellinen ole kesken."""
        return (not self._busy) and time.time() >= self._next_t

    def submit(self, fn, *args):
        """Piirtotyo fn(*args) -> BGR-kuva taustasaikeeseen. Kutsu vain kun due() on True."""
        self._next_t = time.time() + KUVA_VALI_S
        self._busy = True
        self._job = (fn, args)
        self._job_evt.set()

    def _render_loop(self):
        try:
            import hog_analyysi as _ha
            _ha.lower_thread_priority()
        except Exception:
            pass
        while True:
            self._job_evt.wait()
            self._job_evt.clear()
            job, self._job = self._job, None
            if job is None:
                continue
            t0 = time.perf_counter()
            try:
                fn, args = job
                for a in args:                      # t12: valinta paneelin liukulukuihin ennen piirtoa
                    if hasattr(a, "_geom"):
                        a.nayta = self.nayta
                        a.korostus = self.korostus
                        a.viive = self.korostus[0] if self.korostus is not None else 0.0   # t14: keskikuva Alku s myohassa
                img = fn(*args)
                h, w = img.shape[:2]
                sk = 1.0
                if w > KUVA_LEVEYS:
                    sk = KUVA_LEVEYS / float(w)
                    img = cv2.resize(img, (KUVA_LEVEYS, int(round(h * KUVA_LEVEYS / w))), interpolation=cv2.INTER_AREA)
                comp = next((a for a in args if hasattr(a, "boxes")), None)
                def _sc(b):
                    d = dict(b, x=b["x"] * sk, y=b["y"] * sk, w=b["w"] * sk, h=b["h"] * sk)
                    if "viivat" in b:
                        d["viivat"] = [dict(v, p=[[q[0] * sk, q[1] * sk] for q in v["p"]]) for v in b["viivat"]]
                    if "risti" in b:
                        d["risti"] = [b["risti"][0] * sk, b["risti"][1] * sk]
                    return d
                boxes = [_sc(b) for b in getattr(comp, "boxes", [])]
                boxes_t = getattr(comp, "boxes_t", time.time())
                ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, KUVA_LAATU])
                if ok:
                    with self._lock:
                        self._jpg = buf.tobytes()
                        self._boxes, self._boxes_t = boxes, boxes_t
                        self._n += 1
                        self._t = time.time()
                    self.n_rendered += 1
                    self.t_render += time.perf_counter() - t0
            except Exception as e:          # nakyma ei saa kaataa seurantaa
                print(f"Katselu: kuvan piirto epaonnistui ({e!r})")
            finally:
                self._busy = False

    def stop(self):
        try:
            self.httpd.shutdown()
            self.httpd.server_close()
        except Exception:
            pass
        if self.n_rendered:
            print(f"Katselu: {self.n_rendered} kuvaa piirretty, {1000.0 * self.t_render / self.n_rendered:.1f} ms/kuva "
                  f"(kerran {KUVA_VALI_S:g} s:ssa)")


def start(port=8080):
    """Kaynnistaa palvelimen (kerran). Palauttaa KatseluServer-olion tai None jos portti on varattu."""
    global _server
    if _server is not None:
        return _server
    try:
        _server = KatseluServer(port)
    except OSError as e:
        print(f"Katselu: palvelinta ei voitu kaynnistaa portissa {port} ({e}). Kokeile eri porttia: --katselu 8081")
        return None
    url = f"http://{lan_ip()}:{_server.port}/"
    print(f"Katselu: avaa puhelimen selaimella {url}  (sama wifi; Windows-palomuuri: salli Python yksityisissa verkoissa). "
          f"Nakyma paivittyy kerran {KUVA_VALI_S:g} s:ssa, videota ei tallenneta.")
    return _server


def set_state(text):
    if _server is not None:
        _server.set_state(text)


def stop():
    global _server
    if _server is not None:
        _server.stop()
        _server = None
