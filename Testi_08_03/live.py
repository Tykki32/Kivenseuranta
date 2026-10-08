"""Elava kamerasyote (Elgato Cam Link / UVC) videotiedoston sijaan (--live) tai videotiedosto kamerana (--live-sim).

* ``FrameStore``: ruutupuskuri RAM:ssa. Kamerasaie kirjoittaa ruudut juoksevalla indeksilla.
  - Perakkaislukija (paasilmukka, ``LiveEngine.read``) lukee ruudut jarjestyksessa ja odottaa tarvittaessa seuraavaa.
  - Hyppivat lukijat (kiviprofiilin opettelu, alfa-havainnot, varireferenssi) saavat ``BufferedCapture``-olion, joka
    kayttaytyy kuten ``cv2.VideoCapture``.
  - Vanhat ruudut poistetaan kun ne ovat yli ``keep_back`` ruutua perakkaislukijan takana. Kiinnitetyt (pin) ruudut sailyvat.
  - Kalibroinnin naytekohdat ja seurannan rengas (``hires_seuranta``) myos kameran taydella resoluutiolla.
* ``skip_to_latest``: seurannan alussa perakkaislukija hyppaa uusimpaan ruutuun (kalibroinnin aikana kertynyt viive pois).
* Lahteet: ``CameraSource`` (cv2.VideoCapture, Windows: DirectShow/MSMF) ja ``FileSimSource`` (videotiedosto; ilman
  tahdistusta puskuri ei pudota ruutuja -> tulos sama kuin tiedostoajossa).
* Kirjanpito: ``<csv-pohja>_live_aikaleimat.csv`` (kasitelty ruutu -> kameran ruutu -> seinakelloaika).
"""
import csv
import os
import sys
import threading
import time

import cv2
import numpy as np

import asetukset as A
import yleiset


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


_RAW_CODES = {"YUY2": cv2.COLOR_YUV2BGR_YUY2, "YVYU": cv2.COLOR_YUV2BGR_YVYU, "UYVY": cv2.COLOR_YUV2BGR_UYVY}


def raw_yuy2_to_bgr(raw, w, h, out_w, out_h, gpu=False, order="YUY2"):
    """kameran raakaruutu (YUY2, CAP_PROP_CONVERT_RGB=0) -> BGR + pienennys out_w x out_h.
    gpu=True: muunnos ja pienennys OpenCL:lla (cv2.UMat, esim. Intel UHD) -> lahes ei CPU-kuormaa.
    Palauttaa None jos raakaruudun koko ei vastaa YUY2:ta (w*h*2 tavua)."""
    a = np.asarray(raw)
    if a.size != w * h * 2:
        return None
    a = a.reshape(h, w, 2)
    if gpu:
        u = cv2.cvtColor(cv2.UMat(a), _RAW_CODES[order])
        if out_w and out_h and (out_w != w or out_h != h):
            u = cv2.resize(u, (out_w, out_h), interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(u.get())
    b = cv2.cvtColor(a, _RAW_CODES[order])
    if out_w and out_h and (out_w != w or out_h != h):
        b = cv2.resize(b, (out_w, out_h), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(b)


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
        self._hires = {}                  # indeksi -> taysresoluutioinen ruutu (kalibroinnin naytteet)
        self.hires_every = 0              # > 0: joka hires_every:s ruutu myos taydella resoluutiolla (kalibroinnin aikana)
        # seurannan taysresoluutioiset ruudut (rengas, hires_seuranta viimeisinta ruutua; 0 = pois). Arvo: BGR-kuva
        # tai ('yuy2', HxWx2-raakakuva, tavujarjestys) -> muunnetaan BGR:ksi vain pyydetylta alueelta (get_hires_roi).
        self.hires_seuranta = 0
        self._hires_trk = {}
        self.hires_done = -1              # seuranta (paasaie) on kasitellyt ruudut <= tama -> poistetaan
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
        # kiinnitetyt ruudut (pin): sailyvat vaikka historia karsitaan (kiven profiili-/varihavaintojen ruudut,
        # joita luetaan myohemmin, jopa kymmenia sekunteja taaksepain). unpin_all() vapauttaa.
        self._pins = set()
        self._pinned = {}

    # ---------------- kirjoittaja ----------------
    def get_hires(self, idx):
        """kalibroinnin naytekohdan taysresoluutioinen ruutu (tai None)."""
        with self._cond:
            return self._hires.get(int(idx))

    def mark_hires_done(self, idx):
        """paasaie on seurannut ruudun idx -> sen ja vanhemmat taysresoluutioiset ruudut pois."""
        with self._cond:
            self.hires_done = max(self.hires_done, int(idx))
            for k_old in [k for k in self._hires_trk if k <= self.hires_done]:
                del self._hires_trk[k_old]
            self._cond.notify_all()

    def get_hires_roi(self, idx, x0, y0, x1, y1):
        """taysresoluutioisen ruudun alue [y0:y1, x0:x1] BGR:na -> (kuva, todellinen x0) tai (None, 0)."""
        with self._cond:
            e = self._hires_trk.get(int(idx))
            if e is None:
                e = self._hires.get(int(idx))
        if e is None:
            return None, 0
        if isinstance(e, tuple):
            _, a, order = e
            x0e = max(0, int(x0) & ~1)
            x1e = min(a.shape[1], (int(x1) + 1) & ~1)
            roi = np.ascontiguousarray(a[int(y0):int(y1), x0e:x1e])
            return cv2.cvtColor(roi, _RAW_CODES[order]), x0e
        return e[int(y0):int(y1), int(x0):int(x1)], int(x0)

    def put(self, frame, cam_index=None, hires=None, hires_trk=None):
        with self._cond:
            if self.blocking_producer:
                # tiedostosimulaatio (ei tahdistusta) ei saa ehtia taysresoluutioisen renkaan verran seurannan edelle
                while not self._stopped and self.hires_seuranta > 0 and (self._head - self.hires_done) >= self.hires_seuranta - 2:
                    self._cond.wait(0.05)
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
            if hires is not None:          # kalibroinnin taysresoluutioinen naytekuva (vain naytekohdissa)
                self._hires[i] = hires
                for k_old in [k for k in self._hires if k < i - 40 * max(1, self.hires_every)]:
                    del self._hires[k_old]
            if hires_trk is not None and self.hires_seuranta > 0:
                self._hires_trk[i] = hires_trk
                for k_old in [k for k in self._hires_trk if k <= i - self.hires_seuranta or k <= self.hires_done]:
                    del self._hires_trk[k_old]
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
            f = self._frames.pop(self._tail, None)
            if self._tail in self._pins and f is not None:
                self._pinned[self._tail] = f
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
                f = self._pinned.get(idx)
                if f is not None:
                    return f
                raise LiveBufferError(
                    f"ruutu {idx} on jo poistettu puskurista (vanhin {self._tail}) - suurenna LIVE_TAAKSE_S"
                )
            return self._frames[idx]

    def available(self, idx, timeout=None):
        """True kun ruutu idx on saapunut (vaikka se olisi jo poistettu); False jos syote loppui ennen sita."""
        with self._cond:
            return self._wait_for_locked(int(idx), timeout)

    def pin(self, indices):
        """kiinnittaa ruudut (ei poisteta historiaa karsittaessa). Jo poistettua ruutua ei voi palauttaa."""
        with self._cond:
            for i in indices:
                self._pins.add(int(i))        # jo poistettua ruutua ei palauteta: get() antaa selvan virheen kuten ennenkin

    def unpin_all(self):
        with self._cond:
            self._pins.clear()
            self._pinned.clear()

    def pinned_count(self):
        with self._cond:
            return len(self._pinned)

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

    def skip_to(self, idx, timeout=None):
        """perakkaislukija hyppaa ruutuun idx (odottaa kunnes se on saapunut). Palauttaa idx:n tai None jos syote loppui."""
        with self._cond:
            idx = max(int(idx), self._seq_next)
            if not self._wait_for_locked(idx, timeout):
                return None
            self.skipped += idx - self._seq_next
            self._seq_next = idx
            self._evict_locked()
            self._cond.notify_all()
            return idx

    def latest_index(self):
        with self._cond:
            return self._head - 1

    def capture_time(self, idx):
        """ruudun saapumisaika (time.time()) tai None jos ruutu on jo poistettu."""
        with self._cond:
            return self._wall.get(int(idx))

    def lag_frames(self):
        with self._cond:
            return self._head - self._seq_next

    def first_frame(self, timeout=30.0):
        return self.get(0, timeout=timeout)

    def write_log(self, path):
        try:
            with open(path, "w", newline="") as fh:
                w = csv.writer(fh)
                # kasittelyhetki ja viive (puskurissa odottavat ruudut lukuhetkella) -> viivekayra suoraan tiedostosta
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
                 buffer_s=75.0, keep_back_s=40.0, recorder=None, fourcc=None, backend_name=None, min_keep_back_s=None,
                 conversion="ajuri", raw_order=None):
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
        # kuvan muunnos: "ajuri" = ajuri/OpenCV-taustajarjestelma antaa BGR:n (oletus), "raw" = raaka YUY2 + cv2.cvtColor
        # (SIMD, monisaikeinen), "gpu" = raaka YUY2 + muunnos ja pienennys OpenCL:lla. raw/gpu palaa ajuriin jos raakakuva ei ole YUY2.
        self.conversion = "ajuri"
        self.cam_w, self.cam_h = w0, h0
        self.raw_order = "YUY2"
        self.color_check = None
        if conversion in ("raw", "gpu"):
            # raakatila (CAP_PROP_CONVERT_RGB=0) pitaa asettaa ENNEN ensimmaista lukua - jo kaynnistetty MSMF-virta alustuu
            # uudelleen ja antaa RGB32:ta (havaittu: muoto (1, 8294400) = 1920x1080x4). Siksi kamera avataan uudelleen raakatilassa.
            # VARITARKISTUS: vertailukuva = tunnistuksessa ajurin muuntama ruutu (first, ~1-2 s aiemmin; kamera paikallaan).
            # Tavujarjestys (YUY2/YVYU/UYVY) valitaan pienimman keskieron mukaan ja hyvaksytaan jos ero < 5 TAI selvasti pienempi
            # kuin seuraavaksi parhaan (liike ruutujen valilla nostaa kaikkia eroja saman verran; sandbox: oikea 1,0, U/V vaihtunut 8,4).
            ref = cv2.resize(first, (self.out_w, self.out_h), interpolation=cv2.INTER_AREA).astype(np.int16)
            cap.release()                                   # sama laite ei aukea kahdesti samanaikaisesti
            raw_cap = cv2.VideoCapture(self.device, backend)
            raw = None
            if raw_cap.isOpened():
                if fourcc:
                    raw_cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
                raw_cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_w)
                raw_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_h)
                raw_cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
                for _ in range(3):
                    ok_r, raw = raw_cap.read()
            ds = []
            chan = {}
            if raw is not None and np.asarray(raw).size == w0 * h0 * 2:
                for order in _RAW_CODES:
                    if raw_order and order != raw_order:
                        continue
                    t = raw_yuy2_to_bgr(raw, w0, h0, self.out_w, self.out_h, gpu=False, order=order)
                    diff = t.astype(np.int16) - ref
                    # MEDIAANI (liikkuvat alueet eivat vaaristä) + kanavittainen etumerkillinen keskiero (systemaattinen savyero)
                    ds.append((float(np.median(np.abs(diff).mean(axis=2))), order))
                    chan[order] = [round(float(diff[:, :, c].mean()), 1) for c in range(3)]
                ds.sort()
            best = ds[0] if ds else None
            if raw_order:
                ok_conv = best is not None and best[0] < 20.0      # pakotettu jarjestys: hylataan vain selvasti vaara kuva
            else:
                ok_conv = best is not None and best[0] < 20.0 and (best[0] < 4.0 or (len(ds) > 1 and best[0] < 0.7 * ds[1][0]))
            if best is not None:
                print("Live: varitarkistus (mediaaniero ajurin kuvaan; pienin = oikea tavujarjestys): "
                      + ", ".join(f"{o} {d:.1f}" for d, o in ds) + f" | {best[1]} kanavittain B/G/R {chan.get(best[1])}")
            if ok_conv and conversion == "gpu":
                try:
                    raw_yuy2_to_bgr(raw, w0, h0, self.out_w, self.out_h, gpu=True, order=best[1])
                except Exception as e:
                    print(f"Live: GPU-muunnos (OpenCL) ei toimi ({e!r}) -> raaka + OpenCV (CPU)")
                    conversion = "raw"
            if ok_conv:
                cap = raw_cap
                self.cap = cap
                self.conversion = conversion
                self.raw_order = best[1]
                self.color_check = round(best[0], 2)
            else:
                raw_cap.release()
                cap = cv2.VideoCapture(self.device, backend)        # takaisin ajurin muunnokseen (BGR)
                if fourcc:
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_w)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_h)
                cap.read()
                self.cap = cap
                why = ("ei YUY2-kokoinen (muoto " + str(None if raw is None else np.asarray(raw).shape) + ")") if best is None else \
                    "varit eroavat ajurin kuvasta (" + ", ".join(f"{o} {d:.1f}" for d, o in ds) + ")"
                print(f"Live: raakakuva {why} -> muunnos ajurilla")
        self._avaus = (backend, fourcc, cap_w, cap_h)     # uudelleenavaukseen (_avaa_uudelleen)
        self.uudelleenavaukset = []                       # [(katkon alku, kesto s)]
        self.cpu_read = 0.0
        self.cpu_conv = 0.0
        self.n_conv = 0
        self.cpu_grab = 0.0          # harvennuksessa ohitettujen ruutujen grab()-aika
        self.n_grab = 0
        self.backend_name = bname
        self.info = dict(laite=self.device, taustajarjestelma=bname, kameran_koko=f"{w0}x{h0}", muunnos=self.conversion,
                         raakajarjestys=(self.raw_order if self.conversion != "ajuri" else "-"), varitarkistus_keskiero=self.color_check,
                         fourcc=_fourcc_text(cap.get(cv2.CAP_PROP_FOURCC)), fps_ilmoitettu=round(cam_fps, 2),
                         fps_mitattu=round(measured, 2), fps_kaytetty=self.fps, harvennus=self.decim,
                         ulos=f"{self.out_w}x{self.out_h}")
        self.store = FrameStore(self.out_w, self.out_h, self.fps, int(buffer_s * self.fps), int(keep_back_s * self.fps),
                                blocking_producer=False,
                                min_keep_back_frames=None if min_keep_back_s is None else int(min_keep_back_s * self.fps))
        self._thread = threading.Thread(target=self._run, daemon=True, name="live-kamera")   # kaynnistetaan start():lla

    def close(self):
        """Vapauttaa kameran (kaytetaan kun automaattivalinta kokeilee toista taustajarjestelmaa)."""
        try:
            self.cap.release()
        except Exception:
            pass

    def grab_still(self, n_flush=10):
        """yksi kuva kamerasta ENNEN kaynnistysta (start) paneelien valintaan - ei puskuriin eika tallenteeseen.
        n_flush vanhaa ruutua luetaan ohi (ajurin jonossa voi olla vanhoja kuvia)."""
        fr = None
        for _ in range(max(1, n_flush)):
            ok, f = self.cap.read()
            if ok and f is not None:
                fr = f
        if fr is None:
            raise RuntimeError("Kamerasta ei saatu kuvaa paneelien valintaan.")
        return self._convert(fr).copy()

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
        if self.conversion != "ajuri":
            out = raw_yuy2_to_bgr(fr, self.cam_w, self.cam_h, self.out_w, self.out_h, gpu=(self.conversion == "gpu"),
                                  order=self.raw_order)
            if out is not None:
                return out
            raise RuntimeError("raakaruudun koko muuttui")
        if fr.shape[1] != self.out_w or fr.shape[0] != self.out_h:
            fr = cv2.resize(fr, (self.out_w, self.out_h), interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(fr)

    def _convert_full(self, fr):
        """sama muunnos kuin _convert, mutta kameran taydella resoluutiolla (vain kalibroinnin naytekohdissa)."""
        try:
            if self.conversion != "ajuri":
                out = raw_yuy2_to_bgr(fr, self.cam_w, self.cam_h, self.cam_w, self.cam_h, gpu=(self.conversion == "gpu"),
                                      order=self.raw_order)
                return None if out is None else np.ascontiguousarray(out).copy()
            return np.ascontiguousarray(fr).copy()
        except Exception:
            return None

    def _avaa_uudelleen(self):
        """Avaa saman laitteen uudelleen samoilla asetuksilla (raakatila jos muunnos ei ole ajurin). None jos laite ei
        viela anna oikean kokoista kuvaa."""
        backend, fourcc, cap_w, cap_h = self._avaus
        c = cv2.VideoCapture(self.device, backend)
        if not c.isOpened():
            c.release()
            return None
        if fourcc:
            c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        c.set(cv2.CAP_PROP_FRAME_WIDTH, cap_w)
        c.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_h)
        raaka = self.conversion != "ajuri"
        if raaka:
            c.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        for _ in range(5):
            ok, fr = c.read()
            if ok and fr is not None:
                if (np.asarray(fr).size == self.cam_w * self.cam_h * 2) if raaka else fr.shape[:2] == (self.cam_h, self.cam_w):
                    return c
                break
        c.release()
        return None

    def _odota_kameraa(self):
        """Kamera lakkasi antamasta kuvaa (laite katosi): yritetaan avata uudelleen A.LIVE_KAMERA_UUDELLEEN_VALI_S valein
        enintaan A.LIVE_KAMERA_UUDELLEEN_MAX_S. Puskurin lukijat odottavat sillä aikaa. True = kamera palasi."""
        if A.LIVE_KAMERA_UUDELLEEN_MAX_S <= 0:
            return False
        try:
            self.cap.release()
        except Exception:
            pass
        t0 = time.time()
        print(f"\nLive: kamera ei anna kuvaa (laite {self.device}) - yritetaan avata uudelleen "
              f"(enintaan {A.LIVE_KAMERA_UUDELLEEN_MAX_S:.0f} s; tarkista Cam Link / USB / HDMI).")
        t_ilmoitus = t0
        while not self._stop_evt.is_set() and time.time() - t0 < A.LIVE_KAMERA_UUDELLEEN_MAX_S:
            self._stop_evt.wait(A.LIVE_KAMERA_UUDELLEEN_VALI_S)
            if time.time() - t_ilmoitus >= 15.0:
                t_ilmoitus = time.time()
                print(f"Live: odotetaan kameraa {time.time() - t0:.0f} s (irrota ja kytke Cam Link; tarkista kameran HDMI-kuva "
                      f"ja ettei kamera ole sammunut). Ctrl+C lopettaa.")
            try:
                c = self._avaa_uudelleen()
            except Exception:
                c = None
            if c is not None:
                self.cap = c
                kesto = time.time() - t0
                self.uudelleenavaukset.append((time.strftime("%H:%M:%S", time.localtime(t0)), round(kesto, 1)))
                print(f"Live: kamera palasi {kesto:.1f} s katkon jalkeen - seuranta jatkuu (katkon ajalta ei ole ruutuja).")
                return True
        return False

    def _run(self):
        n = 0
        fails = 0
        t_ok = time.time()                                # viimeisin onnistunut luku (katkon tunnistus ajasta)
        try:
            while not self._stop_evt.is_set():
                c0 = time.thread_time()
                if self.decim > 1 and n % self.decim != 0:
                    # harvennuksessa pois jaava ruutu: grab() ilman kuvan kopiointia/muunnosta
                    ok = self.cap.grab()
                    self.cpu_grab += time.thread_time() - c0
                    if ok:
                        self.n_grab += 1
                        fails = 0
                        t_ok = time.time()
                        n += 1
                        continue
                    fr = None
                else:
                    ok, fr = self.cap.read()
                c1 = time.thread_time()
                self.cpu_read += c1 - c0
                if not ok or fr is None:
                    fails += 1
                    # katko: 50 perakkaista virhetta TAI ei onnistunutta lukua A.LIVE_KAMERA_KATKO_S:iin (MSMF:n lukukutsu voi
                    # odottaa ~10 s ennen virhetta E_PENDING, jolloin 50 virhetta kestaisi ~8 min)
                    if fails > 50 or time.time() - t_ok > A.LIVE_KAMERA_KATKO_S:
                        if self._odota_kameraa():
                            fails = 0
                            t_ok = time.time()
                            continue
                        if not self._stop_evt.is_set():
                            print("\nLive: kameran luku epaonnistui toistuvasti - syote paattyy.")
                        break
                    time.sleep(0.02)
                    continue
                fails = 0
                t_ok = time.time()
                if n % self.decim == 0:
                    out = self._convert(fr)
                    hi = None
                    he = self.store.hires_every
                    if he > 0 and self.store._head % he == 0 and (self.cam_w > self.out_w or self.cam_h > self.out_h):
                        hi = self._convert_full(fr)      # kalibroinnin naytekohta taydella resoluutiolla
                    self.cpu_conv += time.thread_time() - c1
                    self.n_conv += 1
                    trk = None
                    if self.store.hires_seuranta > 0 and (self.cam_w > self.out_w or self.cam_h > self.out_h):
                        # seurannan taysresoluutioinen ruutu: raakatilassa YUY2 sellaisenaan (muunnos vain kiven alueelta)
                        if self.conversion in ("raw", "gpu"):
                            trk = ("yuy2", np.asarray(fr).reshape(self.cam_h, self.cam_w, 2), self.raw_order)
                        else:
                            trk = fr
                    self.store.put(out, cam_index=n, hires=hi, hires_trk=trk)
                    # tallenteeseen KAIKKI ruudut, myos ne jotka puskuri pudotti (taynna kalibroinnin aikana)
                    if self.recorder is not None and not self.store.stopped:
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

    def grab_still(self, n_flush=0):
        """simulaation ensimmainen ruutu paneelien valintaan; video kelataan takaisin alkuun."""
        ok, fr = self.cap.read()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if not ok:
            raise RuntimeError("Simulaatiovideosta ei saatu kuvaa paneelien valintaan.")
        if fr.shape[1] != self.out_w or fr.shape[0] != self.out_h:
            fr = cv2.resize(fr, (self.out_w, self.out_h), interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(fr)

    def close(self):
        try:
            self.cap.release()
        except Exception:
            pass

    def _run(self):
        t0 = time.time()
        n = 0
        try:
            while not self._stop_evt.is_set():
                ok, fr = self.cap.read()
                if not ok:
                    break
                hi = None
                he = self.store.hires_every
                if he > 0 and self.store._head % he == 0 and (fr.shape[1] > self.out_w or fr.shape[0] > self.out_h):
                    hi = fr.copy()                       # kalibroinnin naytekohta taydella resoluutiolla
                trk = fr if (self.store.hires_seuranta > 0 and (fr.shape[1] > self.out_w or fr.shape[0] > self.out_h)) else None
                if fr.shape[1] != self.out_w or fr.shape[0] != self.out_h:
                    fr = cv2.resize(fr, (self.out_w, self.out_h), interpolation=cv2.INTER_AREA)
                if self.realtime:
                    dt = t0 + n / self.fps - time.time()
                    if dt > 0:
                        time.sleep(dt)
                self.store.put(np.ascontiguousarray(fr), cam_index=n, hires=hi, hires_trk=trk)
                if self.recorder is not None and not self.store.stopped:     # myos pudotetut ruudut tallenteeseen
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
        # grab ei hae ruutua (ohitettavat ruudut eteenpainluvussa saavat olla jo poistettu puskurista);
        # ruutu haetaan vasta retrieve()/read()-kutsussa.
        if not self.store.available(self.pos):
            return False
        self._last_idx = self.pos
        self._last = None
        self.pos += 1
        return True

    def retrieve(self):
        if self._last is None:
            if getattr(self, "_last_idx", None) is None:
                return False, None
            f = self.store.get(self._last_idx)
            if f is None:
                return False, None
            self._last = f
        return True, self._last.copy()

    def read(self):
        f = self.store.get(self.pos)
        if f is None:
            return False, None
        self._last, self._last_idx = f, self.pos
        self.pos += 1
        return True, f.copy()

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
        # syote ffmpegille valmiina YUV 4:2:0:na (cv2.cvtColor tassa saikeessa, SIMD) -> ffmpegin hidas BGR-muunnos
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
                self._p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                           **yleiset.bg_popen_kwargs())
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
        self.cpu = 0.0
        self.n = 0
        yleiset.alempi_prioriteetti()          # jonossa ~4 s varaa -> saa odottaa seurantaa
        while True:
            f = self._q.get()
            if f is None:
                break
            c0 = time.thread_time()
            self.n += 1
            try:
                if self._yuv:
                    f = cv2.cvtColor(f, cv2.COLOR_BGR2YUV_I420)
                self._p.stdin.write(np.ascontiguousarray(f).data)
            except Exception:
                break
            finally:
                self.cpu += time.thread_time() - c0

    def close(self):
        self._q.put(None)
        self._t.join(120)        # jonossa voi olla ~4 s ruutuja koodaamatta -> odotetaan ne loppuun (ei katkaista tallennetta)
        try:
            self._p.stdin.close()
            self._p.wait(30)
        except Exception:
            pass


# ------------------------------------------------------------------
# aktiivinen live-lahde (yksi live-lahde kerrallaan)
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
