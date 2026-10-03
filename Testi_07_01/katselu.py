"""Testi_07_01: debug-videon katselu puhelimen selaimella samassa verkossa (wifi).

Periaate (koneen kuormitus mahdollisimman pieni):
  * HLS: debug-videon ffmpeg-koodaus (QSV / x264) kirjoittaa SAMASTA koodatusta virrasta myos 1 s HLS-palat
    (ffmpeg tee-muxer) -> ei toista koodausta, vain palojen kirjoitus levylle. Viive n. 2-5 s.
  * Pieni HTTP-palvelin (Pythonin http.server, oma saie) jakaa sivun ja palat. Puhelimen selain toistaa HLS:n
    natiivisti (iPhone Safari, Android Chrome); muuten sivu lataa hls.js:n verkosta.
  * Varakeino /mjpeg: JPEG-kuvat (pienennetty, KATSELU_MJPEG_FPS/s) VAIN kun joku katsoo sita -> ei kuormaa muuten.

Kaytto: main.py --katselu [PORTTI]  (oletus 8080). Osoite tulostetaan kaynnistyksessa, esim. http://192.168.1.23:8080/
Windows kysyy ensimmaisella kerralla palomuurin luvan Pythonille: salli "Yksityiset verkot".
"""

import os
import json
import time
import shutil
import socket
import tempfile
import threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

import cv2

HLS_SEGMENT_S = float(os.environ.get("KATSELU_HLS_PALA_S", "1"))
HLS_LIST_SIZE = int(os.environ.get("KATSELU_HLS_PALOJA", "6"))
MJPEG_FPS = float(os.environ.get("KATSELU_MJPEG_FPS", "5"))
MJPEG_WIDTH = int(os.environ.get("KATSELU_MJPEG_LEVEYS", "734"))
MJPEG_QUALITY = int(os.environ.get("KATSELU_MJPEG_LAATU", "70"))

_server = None


def active():
    return _server


def lan_ip():
    """Koneen osoite lahiverkossa (ei lahetä mitaan: UDP-'yhteys' vain valitsee reitin)."""
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
 #v,#m{display:block;width:100%;max-height:100vh;object-fit:contain;background:#000}
 #m{display:none}
 #t{position:fixed;left:0;right:0;bottom:0;padding:6px 10px;background:rgba(0,0,0,.6)}
 #t a{color:#9cf}
</style></head><body>
<video id="v" muted autoplay playsinline controls></video>
<img id="m" alt="">
<div id="t"><span id="s">Yhdistetaan...</span> &middot; <a href="#" id="sw">vaihda kuvatilaan</a></div>
<script>
const v=document.getElementById('v'),m=document.getElementById('m'),s=document.getElementById('s');
const SRC='hls/stream.m3u8';
let mode='hls',hls=null;
function mjpeg(){mode='mjpeg';if(hls){hls.destroy();hls=null}v.pause();v.style.display='none';
  m.style.display='block';m.src='mjpeg?'+Date.now();document.getElementById('sw').textContent='vaihda videotilaan'}
function startHls(){
  if(v.canPlayType('application/vnd.apple.mpegurl')){v.src=SRC;v.play().catch(()=>{});return}
  const sc=document.createElement('script');
  sc.src='https://cdn.jsdelivr.net/npm/hls.js@1/dist/hls.min.js';
  sc.onload=()=>{if(!window.Hls||!Hls.isSupported()){mjpeg();return}
    hls=new Hls({liveSyncDurationCount:2,liveMaxLatencyDurationCount:5,manifestLoadingMaxRetry:1e9,
                 manifestLoadingRetryDelay:2000});
    hls.loadSource(SRC);hls.attachMedia(v);
    hls.on(Hls.Events.ERROR,(e,d)=>{if(d.fatal){setTimeout(()=>{if(mode==='hls'){hls.loadSource(SRC)}},2000)}});
    v.play().catch(()=>{})};
  sc.onerror=mjpeg;document.head.appendChild(sc)}
document.getElementById('sw').onclick=e=>{e.preventDefault();if(mode==='hls'){mjpeg()}else{location.reload()}};
// natiivi-HLS: jos lista puuttuu viela (seuranta ei ole alkanut), yritetaan uudelleen
v.addEventListener('error',()=>{if(mode==='hls'&&!hls)setTimeout(()=>{v.src=SRC+'?'+Date.now();v.play().catch(()=>{})},2000)});
// pysytaan lahella reaaliaikaa: jos jaljessa > 6 s, hypataan loppuun
setInterval(()=>{if(mode==='hls'&&v.seekable.length){const end=v.seekable.end(v.seekable.length-1);
  if(end-v.currentTime>6)v.currentTime=end-1.5}},3000);
async function tila(){try{const r=await fetch('tila.json',{cache:'no-store'});const j=await r.json();
  s.textContent=j.tila+(j.hls||mode==='mjpeg'?'':' (video alkaa kun seuranta alkaa)');
  if(j.seuranta&&!j.hls_kaytossa&&mode==='hls')mjpeg()}catch(e){s.textContent='Ei yhteytta koneeseen'}}
setInterval(tila,2000);tila();startHls();
</script></body></html>
"""


class _Handler(BaseHTTPRequestHandler):
    server_version = "Kivenseuranta"

    def log_message(self, *a):          # ei lokia konsoliin
        pass

    def _send(self, code, ctype, body, extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
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
            elif path.startswith("/hls/"):
                name = os.path.basename(path)
                fp = os.path.join(srv.hls_dir, name)
                if not (name.endswith(".m3u8") or name.endswith(".ts")) or not os.path.isfile(fp):
                    self._send(404, "text/plain", b"ei viela")
                    return
                with open(fp, "rb") as f:
                    body = f.read()
                ctype = "application/vnd.apple.mpegurl" if name.endswith(".m3u8") else "video/mp2t"
                self._send(200, ctype, body)
            elif path == "/mjpeg":
                self._mjpeg(srv)
            else:
                self._send(404, "text/plain", b"ei loydy")
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _mjpeg(self, srv):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=kehys")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        srv.mjpeg_clients_add(1)
        try:
            last_id = -1
            period = 1.0 / max(0.5, MJPEG_FPS)
            while not srv.stopped:
                t0 = time.monotonic()
                jpg, fid = srv.latest_jpeg(last_id)
                if jpg is not None:
                    last_id = fid
                    self.wfile.write(b"--kehys\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                     + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                time.sleep(max(0.02, period - (time.monotonic() - t0)))
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        finally:
            srv.mjpeg_clients_add(-1)


class KatseluServer:
    def __init__(self, port):
        self.port = int(port)
        self.hls_dir = tempfile.mkdtemp(prefix="kivenseuranta_katselu_")
        self.stopped = False
        self._lock = threading.Lock()
        self._state = "Kaynnistyy"
        self.hls_enabled = False    # ffmpeg-kirjoittaja tekee HLS-palat (False -> cv2-kirjoittaja, vain MJPEG)
        self._mjpeg_clients = 0
        self._frame = None          # viimeisin debug-kuva (viittaus, ei kopiota) - vain kun MJPEG-katsojia on
        self._frame_id = 0
        self._jpeg = (None, -1)
        self.httpd = ThreadingHTTPServer(("0.0.0.0", self.port), _Handler)
        self.httpd.daemon_threads = True
        self.httpd.katselu = self
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self._thread.start()

    # --- tila ---
    def set_state(self, text):
        with self._lock:
            self._state = str(text)

    def status(self):
        with self._lock:
            st = self._state
        return {"tila": st, "hls": os.path.isfile(os.path.join(self.hls_dir, "stream.m3u8")),
                "hls_kaytossa": self.hls_enabled, "seuranta": st.startswith("Seuranta")}

    # --- HLS (ffmpeg kirjoittaa) ---
    def hls_output(self):
        """ffmpeg tee-muxerin HLS-haara. Tiedostonimet SUHTEELLISINA: ffmpeg ajetaan hls_dir-kansiossa (cwd), koska
        Windows-polun kaksoispiste (C:) sotkee tee-muxerin sisakkaiset asetukset."""
        return (f"[f=hls:hls_time={HLS_SEGMENT_S:g}:hls_list_size={HLS_LIST_SIZE}"
                f":hls_flags=delete_segments+independent_segments+omit_endlist"
                f":hls_segment_filename=pala%05d.ts]stream.m3u8")

    # --- MJPEG-varakeino ---
    def mjpeg_clients_add(self, d):
        with self._lock:
            self._mjpeg_clients += d

    def wants_frames(self):
        return self._mjpeg_clients > 0

    def offer_frame(self, img):
        """Debug-kuvan kirjoitussaie: talletetaan vain viittaus (halpa). Koodaus JPEG:ksi vasta katsojan saikeessa."""
        if self._mjpeg_clients > 0:
            with self._lock:
                self._frame = img
                self._frame_id += 1

    def latest_jpeg(self, last_id):
        with self._lock:
            img, fid = self._frame, self._frame_id
            cached = self._jpeg
        if img is None or fid == last_id:
            return None, last_id
        if cached[1] == fid:
            return cached[0], fid
        h, w = img.shape[:2]
        if w > MJPEG_WIDTH:
            img = cv2.resize(img, (MJPEG_WIDTH, int(round(h * MJPEG_WIDTH / w))), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, MJPEG_QUALITY])
        if not ok:
            return None, last_id
        jpg = buf.tobytes()
        with self._lock:
            self._jpeg = (jpg, fid)
        return jpg, fid

    def stop(self):
        self.stopped = True
        try:
            self.httpd.shutdown()
            self.httpd.server_close()
        except Exception:
            pass
        shutil.rmtree(self.hls_dir, ignore_errors=True)


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
    print(f"Katselu: avaa puhelimen selaimella {url}  (sama wifi; Windows-palomuuri: salli Python yksityisissa verkoissa)")
    return _server


def set_state(text):
    if _server is not None:
        _server.set_state(text)


def stop():
    global _server
    if _server is not None:
        _server.stop()
        _server = None
