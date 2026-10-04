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
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

import cv2

KUVA_VALI_S = float(os.environ.get("KATSELU_VALI_S", "1.0"))        # kuinka usein nakyma piirretaan (s)
KUVA_LEVEYS = int(os.environ.get("KATSELU_LEVEYS", "1100"))          # JPEG-kuvan leveys (px); debug-kuva on 1468 px
KUVA_LAATU = int(os.environ.get("KATSELU_LAATU", "75"))              # JPEG-laatu

_server = None


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
 #k{display:block;width:100%;max-height:calc(100vh - 32px);object-fit:contain;background:#000}
 #t{position:fixed;left:0;right:0;bottom:0;padding:6px 10px;background:rgba(0,0,0,.7)}
</style></head><body>
<img id="k" alt="">
<div id="t">Yhdistetaan...</div>
<script>
const k=document.getElementById('k'),t=document.getElementById('t');
let last=-1,busy=false;
async function paivita(){
  if(busy)return;busy=true;
  try{
    const r=await fetch('tila.json',{cache:'no-store'});const j=await r.json();
    let txt=j.tila;
    if(j.kuva<0){txt+=' (kuva alkaa kun seuranta alkaa)'}
    else if(j.kuva!==last){
      await new Promise(res=>{const im=new Image();im.onload=()=>{k.src=im.src;res()};im.onerror=res;
        im.src='kuva.jpg?n='+j.kuva});
      last=j.kuva}
    if(j.ika>3)txt+=' - kuva '+Math.round(j.ika)+' s vanha';
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
                self._send(200, "application/json", json.dumps(srv.status()).encode("utf-8"))
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
            return {"tila": self._state, "kuva": self._n, "ika": (time.time() - self._t) if self._n >= 0 else None}

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
                img = fn(*args)
                h, w = img.shape[:2]
                if w > KUVA_LEVEYS:
                    img = cv2.resize(img, (KUVA_LEVEYS, int(round(h * KUVA_LEVEYS / w))), interpolation=cv2.INTER_AREA)
                ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, KUVA_LAATU])
                if ok:
                    with self._lock:
                        self._jpg = buf.tobytes()
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
