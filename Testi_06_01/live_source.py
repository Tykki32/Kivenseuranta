"""Testi_06_01: elava kamerasyote (Elgato Cam Link / UVC) videotiedoston sijaan.

Rakenne
-------
* ``FrameStore``: ruutupuskuri RAM:ssa. Kamerasaie kirjoittaa ruudut juoksevalla indeksilla (0, 1, 2, ...).
  - Perakkaislukija (paasilmukka, ``LiveEngine.read``) lukee ruudut jarjestyksessa ja odottaa tarvittaessa seuraavaa.
  - Hyppivat lukijat (kalibroinnin profiilin opettelu, alfa-havainnot, varireferenssi) saavat
    ``BufferedCapture``-olion, joka kayttaytyy kuten ``cv2.VideoCapture`` (read/grab/set(POS_FRAMES)/get):
    taaksepain puskurin sisalla ja eteenpain (odottaa kunnes kamera on tuottanut ruudun).
  - Vanhat ruudut poistetaan kun ne ovat yli ``keep_back`` ruutua perakkaislukijan takana (kalibrointi tarvitsee
    ~30 s taaksepain, katso STONE_TRACK_WINDOW_SECONDS). Elavan seurannan alkaessa ``keep_back`` pienennetaan.
* ``skip_to_latest``: elavan seurannan alussa perakkaislukija hyppaa uusimpaan ruutuun (kalibroinnin aikana
  kertynyt viive pois). Seuranta alkaa vasta tasta, joten hyppy ei katkaise yhtaan seurattua kivea.
* Lahteet:
  - ``CameraSource``: cv2.VideoCapture(laite) (Windows: DirectShow), oletuksena pienennys 1280x720 (INTER_AREA),
    jos kamera antaa esim. 50 fps ja tavoite on 25 fps, joka toinen ruutu (LIVE_FPS).
  - ``FileSimSource``: videotiedosto "kamerana" testaukseen ilman kameraa (``--live-sim``). Ilman tahdistusta
    (``--live-sim-tahti 0``) puskuri ei koskaan pudota ruutuja -> tulosten pitaa olla identtiset tiedostoajon kanssa.
* Kirjanpito: ``<csv-pohja>_live_aikaleimat.csv`` (kasitelty ruutu -> kameran ruutu -> seinakelloaika).
"""
import collections
import csv
import os
import sys
import threading
import time

import cv2
import numpy as np


class LiveBufferError(RuntimeError):
    pass


def _backend():
    if sys.platform.startswith("win"):
        return cv2.CAP_DSHOW, "DirectShow"
    if sys.platform.startswith("linux"):
        return cv2.CAP_V4L2, "V4L2"
    return cv2.CAP_ANY, "ANY"


def _fourcc_text(v):
    v = int(v)
    t = "".join(chr((v >> (8 * i)) & 0xFF) for i in range(4))
    return t if t.isprintable() and v else "?"


class FrameStore:
    """Indeksoitu ruutupuskuri. Kirjoittaja: put(). Lukijat: read_next() (perakkainen) ja get(idx) (hyppiva)."""

    def __init__(self, width, height, fps, capacity_frames, keep_back_frames, blocking_producer=False, min_keep_back_frames=None):
        self.width, self.height, self.fps = int(width), int(height), float(fps)
        self.capacity = max(16, int(capacity_frames))
        self.keep_back = max(0, int(keep_back_frames))
        # pakollinen minimihistoria: kun puskuri on taynna, vanhinta historiaa poistetaan tahan asti ENNEN kuin uusia ruutuja pudotetaan
        self.min_keep_back = self.keep_back if min_keep_back_frames is None else min(self.keep_back, max(0, int(min_keep_back_frames)))
        self.blocking_producer = blocking_producer
        self._frames = {}                 # indeksi -> ruutu (uint8 HxWx3)
        self._wall = {}                   # indeksi -> seinakello (time.time()) kun ruutu saapui
        self._cam_idx = {}                # indeksi -> kameran oma ruutunumero (ennen harvennusta)
        self._head = 0                    # seuraavan kirjoitettavan ruudun indeksi
        self._tail = 0                    # vanhin puskurissa oleva indeksi
        self._seq_next = 0                # perakkaislukijan seuraava indeksi
        self._stopped = False
        self._stop_reason = ""
        self._cond = threading.Condition()
        self.dropped = 0                  # pudotetut (puskuri taynna) ruudut
        self.skipped = 0                  # skip_to_latest-hypyissa ohitetut ruudut
        self.max_lag_frames = 0
        self.log = []                     # (kasitelty indeksi, kameran ruutu, seinakello) perakkaislukijan lukemille

    # ---------------- kirjoittaja ----------------
    def put(self, frame, cam_index=None):
        with self._cond:
            if self.blocking_producer:
                while not self._stopped and (self._head - self._tail) >= self.capacity:
                    self._evict_locked()
                    if (self._head - self._tail) >= self.capacity:
                        self._cond.wait(0.05)
            else:
                self._evict_locked()
                if (self._head - self._tail) >= self.capacity:
                    self._evict_locked(self.min_keep_back)      # ensin vanhaa historiaa pois (minimiin asti)
                if (self._head - self._tail) >= self.capacity:
                    self.dropped += 1         # puskuri taynna lukemattomia -> uusin ruutu pudotetaan
                    if self.dropped in (1, 10, 100) or self.dropped % 1000 == 0:
                        print(f"\nVAROITUS (live): puskuri taynna, pudotettu {self.dropped} ruutua - kasittely ei pysy kameran tahdissa "
                              f"(suurenna --live-puskuri-s tai kevenna asetuksia)")
                    return False
            if self._stopped:
                return False
            i = self._head
            self._frames[i] = frame
            self._wall[i] = time.time()
            self._cam_idx[i] = i if cam_index is None else int(cam_index)
            self._head += 1
            self._cond.notify_all()
            return True

    def stop(self, reason=""):
        with self._cond:
            if not self._stopped:
                self._stopped = True
                self._stop_reason = reason
            self._cond.notify_all()

    @property
    def stopped(self):
        return self._stopped

    # ---------------- sisaiset ----------------
    def _evict_locked(self, keep_back=None):
        floor = self._seq_next - (self.keep_back if keep_back is None else keep_back)
        while self._tail < floor and self._tail < self._head:
            self._frames.pop(self._tail, None)
            self._wall.pop(self._tail, None)
            self._cam_idx.pop(self._tail, None)
            self._tail += 1

    def _wait_for_locked(self, idx, timeout):
        t_end = None if timeout is None else time.time() + timeout
        while idx >= self._head and not self._stopped:
            rem = None if t_end is None else t_end - time.time()
            if rem is not None and rem <= 0:
                return False
            self._cond.wait(0.2 if rem is None else min(0.2, rem))
        return idx < self._head

    # ---------------- lukijat ----------------
    def read_next(self, timeout=None):
        """Perakkaislukija: palauttaa (indeksi, ruutu) tai (None, None) kun syote loppui."""
        with self._cond:
            i = self._seq_next
            if not self._wait_for_locked(i, timeout):
                return None, None
            if i < self._tail:
                raise LiveBufferError(f"ruutu {i} on jo poistettu puskurista")
            f = self._frames[i]
            self._seq_next = i + 1
            lag = self._head - self._seq_next
            self.max_lag_frames = max(self.max_lag_frames, lag)
            self.log.append((i, self._cam_idx[i], self._wall[i], time.time(), lag))
            self._evict_locked()
            self._cond.notify_all()
        return i, f

    def get(self, idx, timeout=None):
        """Hyppiva lukija: ruutu indeksilla (odottaa jos ei viela tullut). None jos syote loppui ennen ruutua."""
        with self._cond:
            if not self._wait_for_locked(idx, timeout):
                return None
            if idx < self._tail:
                raise LiveBufferError(
                    f"ruutu {idx} on jo poistettu puskurista (vanhin {self._tail}) - suurenna LIVE_TAAKSE_S"
                )
            return self._frames[idx]

    def skip_to_latest(self, keep_frames=0, new_keep_back=None):
        """Perakkaislukija hyppaa uusimpaan ruutuun (jattaa keep_frames ruutua lukematta taakse)."""
        with self._cond:
            target = max(self._seq_next, self._head - max(0, int(keep_frames)))
            self.skipped += target - self._seq_next
            self._seq_next = target
            if new_keep_back is not None:
                self.keep_back = max(0, int(new_keep_back))
                self.min_keep_back = min(self.min_keep_back, self.keep_back)
            self._evict_locked()
            self._cond.notify_all()
            return target

    def lag_frames(self):
        with self._cond:
            return self._head - self._seq_next

    def first_frame(self, timeout=30.0):
        return self.get(0, timeout=timeout)

    def write_log(self, path):
        try:
            with open(path, "w", newline="") as fh:
                w = csv.writer(fh)
                # v6.2: kasittelyhetki ja viive (puskurissa odottavat ruudut lukuhetkella) -> viivekayra suoraan tiedostosta
                w.writerow(["kasitelty_ruutu", "puskurin_indeksi", "kameran_ruutu", "seinakello_unix", "seinakello",
                            "kasittely_unix", "viive_s", "odottavia_ruutuja"])
                for n, (i, ci, t, tp, lag) in enumerate(self.log):
                    w.writerow([n, i, ci, f"{t:.3f}", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t)) + f".{int((t % 1) * 1000):03d}",
                                f"{tp:.3f}", f"{tp - t:.2f}", lag])
            return True
        except Exception as e:
            print(f"Live-aikaleimojen tallennus epaonnistui: {e!r}")
            return False


class _SourceBase:
    def __init__(self):
        self.store = None
        self._thread = None
        self._stop_evt = threading.Event()
        self.info = {}

    def start(self):
        """Kaynnistaa lukusaikeen (kutsu kun mahdollinen tallentaja on kytketty -> ensimmainenkin ruutu tallentuu)."""
        if self._thread is not None and not self._thread.is_alive():
            self._thread.start()

    def stop(self, reason="pysaytetty"):
        self._stop_evt.set()
        if self.store is not None:
            self.store.stop(reason)

    def join(self, timeout=5.0):
        if self._thread is not None:
            self._thread.join(timeout)


class CameraSource(_SourceBase):
    """Kamera (UVC, esim. Cam Link) taustasaikeessa -> FrameStore."""

    def __init__(self, device=None, cap_w=1920, cap_h=1080, out_w=1280, out_h=720, target_fps=0.0,
                 buffer_s=75.0, keep_back_s=40.0, recorder=None, fourcc=None, backend_name=None, min_keep_back_s=None):
        super().__init__()
        self.recorder = recorder
        backend, bname = _backend()
        if backend_name:
            bname = backend_name.upper()
            backend = {"DSHOW": cv2.CAP_DSHOW, "MSMF": cv2.CAP_MSMF, "V4L2": cv2.CAP_V4L2, "ANY": cv2.CAP_ANY}[bname]
        devices = [device] if device is not None else list(range(8))
        cap = None
        for d in devices:
            c = cv2.VideoCapture(d, backend)
            if not c.isOpened():
                continue
            if fourcc:
                c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
            c.set(cv2.CAP_PROP_FRAME_WIDTH, cap_w)
            c.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_h)
            ok, fr = c.read()
            if ok and fr is not None and (device is not None or (fr.shape[1] == cap_w and fr.shape[0] == cap_h)):
                cap, self.device, first = c, d, fr
                break
            c.release()
        if cap is None:
            raise RuntimeError(f"Kameraa ({cap_w}x{cap_h}) ei loytynyt (laitteet {devices}). Tarkista Cam Link / HDMI / kameran tila.")
        self.cap = cap
        cam_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        measured = self._measure_fps(cap, 50)
        std = min((24, 25, 30, 50, 60), key=lambda f: abs(f - measured))
        cam_fps_eff = std if abs(std - measured) / std < 0.08 else (cam_fps or measured)
        self.cam_fps = float(cam_fps_eff)
        self.decim = 1
        if target_fps and target_fps > 0 and self.cam_fps > target_fps * 1.5:
            self.decim = max(1, int(round(self.cam_fps / target_fps)))
        self.fps = self.cam_fps / self.decim
        h0, w0 = first.shape[:2]
        self.out_w, self.out_h = (out_w, out_h) if (out_w and out_h) else (w0, h0)
        self.info = dict(laite=self.device, taustajarjestelma=bname, kameran_koko=f"{w0}x{h0}",
                         fourcc=_fourcc_text(cap.get(cv2.CAP_PROP_FOURCC)), fps_ilmoitettu=round(cam_fps, 2),
                         fps_mitattu=round(measured, 2), fps_kaytetty=self.fps, harvennus=self.decim,
                         ulos=f"{self.out_w}x{self.out_h}")
        self.store = FrameStore(self.out_w, self.out_h, self.fps, int(buffer_s * self.fps), int(keep_back_s * self.fps),
                                blocking_producer=False,
                                min_keep_back_frames=None if min_keep_back_s is None else int(min_keep_back_s * self.fps))
        self._thread = threading.Thread(target=self._run, daemon=True, name="live-kamera")   # kaynnistetaan start():lla

    @staticmethod
    def _measure_fps(cap, n):
        t0 = time.time()
        k = 0
        for _ in range(n):
            ok, _f = cap.read()
            if not ok:
                break
            k += 1
        return k / max(time.time() - t0, 1e-6)

    def _convert(self, fr):
        if fr.shape[1] != self.out_w or fr.shape[0] != self.out_h:
            fr = cv2.resize(fr, (self.out_w, self.out_h), interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(fr)

    def _run(self):
        n = 0
        fails = 0
        try:
            while not self._stop_evt.is_set():
                ok, fr = self.cap.read()
                if not ok or fr is None:
                    fails += 1
                    if fails > 50:
                        print("\nLive: kameran luku epaonnistui toistuvasti - syote paattyy.")
                        break
                    time.sleep(0.02)
                    continue
                fails = 0
                if n % self.decim == 0:
                    out = self._convert(fr)
                    if self.store.put(out, cam_index=n) and self.recorder is not None:
                        self.recorder.write(out)
                n += 1
        finally:
            self.cap.release()
            self.store.stop("kamera suljettu")


class FileSimSource(_SourceBase):
    """Videotiedosto 'kamerana' (testaus ilman kameraa). realtime=True: tahdistus tiedoston fps:aan (kuten kamera, pudottaa
    ruutuja jos puskuri taynna); realtime=False: niin nopeasti kuin kasittely ehtii, EI pudota (tulos = tiedostoajo)."""

    def __init__(self, path, realtime=True, buffer_s=75.0, keep_back_s=40.0, out_w=0, out_h=0, recorder=None, min_keep_back_s=None):
        super().__init__()
        self.recorder = recorder
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Simulaatiovideota ei voitu avata: {path}")
        w0 = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h0 = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.out_w, self.out_h = (out_w, out_h) if (out_w and out_h) else (w0, h0)
        self.realtime = realtime
        self.info = dict(simulaatio=os.path.basename(path), kameran_koko=f"{w0}x{h0}", fps_kaytetty=self.fps,
                         tahdistus="reaaliaika" if realtime else "ei (niin nopeasti kuin ehtii)", ulos=f"{self.out_w}x{self.out_h}")
        self.store = FrameStore(self.out_w, self.out_h, self.fps, int(buffer_s * self.fps), int(keep_back_s * self.fps),
                                blocking_producer=not realtime,
                                min_keep_back_frames=None if min_keep_back_s is None else int(min_keep_back_s * self.fps))
        self._thread = threading.Thread(target=self._run, daemon=True, name="live-simulaatio")   # kaynnistetaan start():lla

    def _run(self):
        t0 = time.time()
        n = 0
        try:
            while not self._stop_evt.is_set():
                ok, fr = self.cap.read()
                if not ok:
                    break
                if fr.shape[1] != self.out_w or fr.shape[0] != self.out_h:
                    fr = cv2.resize(fr, (self.out_w, self.out_h), interpolation=cv2.INTER_AREA)
                if self.realtime:
                    dt = t0 + n / self.fps - time.time()
                    if dt > 0:
                        time.sleep(dt)
                if self.store.put(np.ascontiguousarray(fr), cam_index=n) and self.recorder is not None:
                    self.recorder.write(fr)
                n += 1
        finally:
            self.cap.release()
            self.store.stop("simulaatiovideo loppui")


class BufferedCapture:
    """cv2.VideoCapture-yhteensopiva lukija FrameStoresta (read, grab, retrieve, set/get POS_FRAMES, isOpened, release)."""

    def __init__(self, store, start=0):
        self.store = store
        self.pos = int(start)
        self._last = None

    def isOpened(self):
        return True

    def set(self, prop, value):
        if prop == cv2.CAP_PROP_POS_FRAMES:
            self.pos = max(0, int(value))
            return True
        return False

    def get(self, prop):
        s = self.store
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(s.width)
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(s.height)
        if prop == cv2.CAP_PROP_FPS:
            return float(s.fps)
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return float(LIVE_FRAME_COUNT)
        if prop == cv2.CAP_PROP_POS_FRAMES:
            return float(self.pos)
        return 0.0

    def grab(self):
        f = self.store.get(self.pos)
        if f is None:
            return False
        self._last = f
        self.pos += 1
        return True

    def retrieve(self):
        if self._last is None:
            return False, None
        return True, self._last.copy()

    def read(self):
        if not self.grab():
            return False, None
        return True, self._last.copy()

    def release(self):
        self._last = None


# Live-tilassa videon pituus ei ole tiedossa: kalibroinnin ikkunalaskelmat kayttavat taman ylarajana.
LIVE_FRAME_COUNT = 10 ** 9


class LiveEngine:
    """mode_engine.ModeEngine-kaare live-tilaan: read() hakee seuraavan ruudun puskurista ja antaa sen C++:lle (set_frame).
    Muut metodit (set_transform, add_mode_frame, start_mode_background, ...) suoraan ModeEnginelle."""

    def __init__(self, mode_engine_module, source, tile_size):
        self.source = source
        self.store = source.store
        self._eng = mode_engine_module.ModeEngine(self.store.width, self.store.height, float(self.store.fps), int(tile_size))
        self._empty = np.empty((0,), np.uint8)
        self.last_index = -1

    def read(self):
        i, f = self.store.read_next()
        if f is None:
            return self._empty
        self.last_index = i
        self._eng.set_frame(f)
        return f.copy()          # kopio: kutsuja saa muokata ruutua, puskurin ruutu pysyy koskemattomana hyppiville lukijoille

    def total_frames(self):
        return 0

    def __getattr__(self, name):
        return getattr(self._eng, name)


class FfmpegRecorder:
    """Valinnainen tallennus (--live-tallenna): ruudut taustasaikeessa ffmpegille (oletus h264_qsv, muuten libx264)."""

    def __init__(self, path, w, h, fps, encoder="auto"):
        import queue
        import subprocess
        self._q = queue.Queue(maxsize=int(fps * 4))
        # v6.2: syote ffmpegille valmiina YUV 4:2:0:na (cv2.cvtColor tassa saikeessa, SIMD) -> ffmpegin hidas BGR-muunnos
        # (swscale) jaa pois ja putkeen puolet vahemman dataa (kuten debug-videon DEBUG_YUV). Vaatii parilliset mitat.
        self._yuv = (int(w) % 2 == 0 and int(h) % 2 == 0)
        pix = "yuv420p" if self._yuv else "bgr24"
        frame_bytes = int(w) * int(h) * 3 // 2 if self._yuv else int(w) * int(h) * 3
        enc_args = {
            "qsv": ["-vf", "format=nv12", "-c:v", "h264_qsv", "-global_quality", "20", "-look_ahead", "0"],
            "x264": ["-c:v", "libx264", "-preset", "veryfast", "-crf", "16"],
        }
        order = ["qsv", "x264"] if encoder == "auto" else [encoder]
        self.encoder = None
        self.dropped = 0
        for enc in order:
            cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", pix,
                   "-s", f"{w}x{h}", "-r", f"{fps:.4f}", "-i", "-", "-an"] + enc_args[enc] + [path]
            try:
                probe = subprocess.run(cmd[:-1] + ["-frames:v", "1", "-f", "null", "-"], input=bytes(frame_bytes),
                                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=30)
                if probe.returncode != 0:
                    continue
                self._p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                self.encoder = enc
                break
            except Exception:
                continue
        if self.encoder is None:
            raise RuntimeError("Live-tallennus: ffmpeg/qsv tai ffmpeg/libx264 ei kaytettavissa")
        self.path = path
        self._t = threading.Thread(target=self._run, daemon=True, name="live-tallennus")
        self._t.start()

    def write(self, frame):
        try:
            self._q.put_nowait(frame)
        except Exception:
            self.dropped += 1

    def _run(self):
        while True:
            f = self._q.get()
            if f is None:
                break
            try:
                if self._yuv:
                    f = cv2.cvtColor(f, cv2.COLOR_BGR2YUV_I420)
                self._p.stdin.write(np.ascontiguousarray(f).data)
            except Exception:
                break

    def close(self):
        self._q.put(None)
        self._t.join(120)        # jonossa voi olla ~4 s ruutuja koodaamatta -> odotetaan ne loppuun (ei katkaista tallennetta)
        try:
            self._p.stdin.close()
            self._p.wait(30)
        except Exception:
            pass


# ------------------------------------------------------------------
# main.py:n kayttama globaali tila (yksi live-lahde kerrallaan)
# ------------------------------------------------------------------
_active = None


def activate(source):
    global _active
    _active = source


def active():
    return _active


def open_capture(path):
    """cv2.VideoCapture(path) korvaaja: live-tilassa BufferedCapture puskurista, muuten tavallinen cv2.VideoCapture."""
    if _active is not None:
        return BufferedCapture(_active.store)
    return cv2.VideoCapture(path)
