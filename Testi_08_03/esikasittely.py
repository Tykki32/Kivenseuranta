"""Kuvan esikasittely seurantaa varten: valotasapaino, stabilointi (vaihekorrelaatio moodikuvaan), linssikorjaus,
varjonsietoinen taustanvaimennus ja niiden liukuhihna.

Seurannan ruutu kulkee kolmessa vaiheessa rinnan (LivePipeline):
  A) luku + harmaasavy + stabilointi: jokainen ruutu verrataan SUORAAN kalibroinnin moodikuvaan vaihekorrelaatiolla
     (ei ketjutusta edelliseen ruutuun -> virhe ei kerry). Vaihekorrelaatiot ajetaan useassa saikeessa rinnan.
  B) warpAffine + linssikorjaus (remap) + valotasapaino + varjonsietoinen taustanvaimennus (C++ / GPU)
     -> frame_u (korjattu kuva) ja frame_u_for_tracking (tausta valkoiseksi; HAKU ja SEURANTA kayttavat tata).
     HAKU kaynnistetaan heti kun ruutu on valmis (tulos on valmiina kun paasaie ehtii ruutuun).
  C) paasaie: HAKU/SEURANTA, tulokset, CSV (seuranta.py).
Ruutujen jarjestys ja sisalto ovat samat kuin ilman hihnaa.

Valotasapaino: kanavakohtainen gain/bias moodikuvaan (robusti PNS), estimoidaan taustasaikeessa kerran sekunnissa ja
otetaan kayttoon tasan A.VALO_KAYTTOON_VIIVE_RUUTUA ruudun paasta (toistettava ajo).
"""
import collections
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

import stone_tracker

import asetukset as A
import yleiset

# Liukuhihnan tilastot pullonkaula-analyysiin (yleiset.pullonkaula_raportti)
PIPE_STATS = {"wall0": None, "cpu0": None, "qC_sum": 0, "qC_n": 0, "qA_sum": 0, "qA_n": 0, "cpu_a": None, "cpu_b": None,
              "depth": 3}


# =====================================================================================================================
# VALOTASAPAINO
# =====================================================================================================================

def _robust_gain_bias_single_channel(frame_values, reference_values, n_iter=3):
    """(gain, bias) joka minimoi (gain * frame + bias - reference)^2 robustisti: jokaisen kierroksen jalkeen suurimman
    jaannoksen pikselit (yli 80. persentiilin: etuala, pelaajat, kivet) pois seuraavasta kierroksesta."""
    mask = np.ones(frame_values.shape, dtype=bool)
    gain, bias = 1.0, 0.0
    for _ in range(n_iter):
        x = frame_values[mask]
        y = reference_values[mask]
        mean_x = x.mean()
        mean_y = y.mean()
        centered_x = x - mean_x
        denom = np.dot(centered_x, centered_x)
        if denom > 1e-6:
            gain = float(np.dot(centered_x, y - mean_y) / denom)
        else:
            gain = 1.0
        bias = float(mean_y - gain * mean_x)
        residual = np.abs(reference_values - (gain * frame_values + bias))
        threshold = np.percentile(residual, 80)
        mask = residual < threshold
    return gain, bias


def estimate_photometric_correction(frame_bgr, reference_bgr):
    """Kanavakohtainen (gains, biases) moodikuvaan: kattaa seka kirkkauden etta valkotasapainon ajautumisen."""
    gains, biases = [], []
    st_ = A.VALO_HARVENNUS
    frame_bgr = frame_bgr[::st_, ::st_]
    reference_bgr = reference_bgr[::st_, ::st_]
    for channel in range(3):
        gain, bias = _robust_gain_bias_single_channel(
            frame_bgr[:, :, channel].astype(np.float64).ravel(), reference_bgr[:, :, channel].astype(np.float64).ravel()
        )
        gains.append(gain)
        biases.append(bias)
    return gains, biases


_PHOTO_LUT_CACHE = {"key": None, "lut": None}


def apply_photometric_correction(frame_bgr, gains, biases):
    key = (tuple(float(g) for g in gains), tuple(float(b) for b in biases))
    if _PHOTO_LUT_CACHE["key"] != key:
        x = np.arange(256, dtype=np.float32)
        lut = np.stack([np.clip(x * gains[c] + biases[c], 0, 255).astype(np.uint8) for c in range(3)], axis=1).reshape(
            256, 1, 3
        )
        _PHOTO_LUT_CACHE["key"], _PHOTO_LUT_CACHE["lut"] = key, lut
    return cv2.LUT(frame_bgr, _PHOTO_LUT_CACHE["lut"])


def suppress_shadow_background(frame_u, ref_u, gain, bias):
    """Varjonsietoinen taustanvaimennus (C++): taustaksi (valkoinen) pikselit jotka ovat lahella moodikuvaa, varjoa
    (saturaatio sama, V pudonnut A.VARJO_V_PUDOTUS_MIN..MAX) tai jaata (S < A.JAA_S_MAX, V > A.JAA_V_MIN)."""
    return stone_tracker.suppress_shadow_background(
        frame_u, ref_u, np.asarray(gain, dtype=np.float64), np.asarray(bias, dtype=np.float64),
        float(A.GRANIITTI_EROKYNNYS), float(A.VARJO_V_PUDOTUS_MIN), float(A.VARJO_V_PUDOTUS_MAX), int(A.JAA_S_MAX),
        int(A.JAA_V_MIN),
    )


# =====================================================================================================================
# STABILOINTI: VAIHEKORRELAATIO MOODIKUVAAN
# =====================================================================================================================
_hann_window_cache = {}
_phase_ref_cache = {}
_stab_small_ref = {}


def _phase_correlate_full_frame(gray_a, gray_b):
    """Vaihekorrelaatio OpenCV:lla (varapolku kun kuvan koko ei sovi nopealle versiolle). Palauttaa siirron (dx, dy),
    jolla gray_b linjautuu gray_a:n kanssa, rajattuna +-A.STAB_MAX_SIIRTO_PX."""
    h, w = gray_a.shape[:2]
    hann = _hann_window_cache.get((w, h))
    if hann is None:
        hann = cv2.createHanningWindow((w, h), cv2.CV_32F)
        _hann_window_cache[(w, h)] = hann
    (dx, dy), _response = cv2.phaseCorrelate(gray_a.astype(np.float32), gray_b.astype(np.float32), hann)
    dx = float(np.clip(dx, -A.STAB_MAX_SIIRTO_PX, A.STAB_MAX_SIIRTO_PX))
    dy = float(np.clip(dy, -A.STAB_MAX_SIIRTO_PX, A.STAB_MAX_SIIRTO_PX))
    return dx, dy


def _phase_correlate_small(ref_gray, gray):
    """Nopea vaihekorrelaatio: referenssin ikkunoitu FFT valimuistissa, elementtikohtaiset vaiheet C++:ssa.
    Sama tulos kuin _phase_correlate_full_frame (virhe < 1e-6 px)."""
    h, w = ref_gray.shape[:2]
    if (h & 1) or (w & 1) or cv2.getOptimalDFTSize(h) != h or cv2.getOptimalDFTSize(w) != w or gray.shape[:2] != (h, w):
        return _phase_correlate_full_frame(ref_gray, gray)
    key = (id(ref_gray), h, w)
    ent = _phase_ref_cache.get(key)
    if ent is None:
        hann = cv2.createHanningWindow((w, h), cv2.CV_32F)
        ref_f = np.ascontiguousarray(ref_gray, dtype=np.uint8)
        ref_ccs = cv2.dft(stone_tracker.phase_window(ref_f, hann))
        _phase_ref_cache.clear()
        ent = (hann, ref_ccs, ref_gray)   # ref_gray talteen jotta id() pysyy voimassa
        _phase_ref_cache[key] = ent
    hann, ref_ccs, _ = ent
    g8 = np.ascontiguousarray(gray, dtype=np.uint8)
    f_ccs = cv2.dft(stone_tracker.phase_window(g8, hann))
    p = stone_tracker.phase_mulnorm(ref_ccs, f_ccs)
    c = cv2.dft(p, flags=cv2.DFT_INVERSE | cv2.DFT_SCALE)
    dx, dy = stone_tracker.phase_peak(c)
    dx = float(np.clip(dx, -A.STAB_MAX_SIIRTO_PX, A.STAB_MAX_SIIRTO_PX))
    dy = float(np.clip(dy, -A.STAB_MAX_SIIRTO_PX, A.STAB_MAX_SIIRTO_PX))
    return dx, dy


def phase_correlate(ref_gray, gray):
    """Siirto (dx, dy) moodikuvaan. Leveampi kuin A.STAB_LEVEYS -> lasketaan pienennetysta kuvasta ja skaalataan."""
    h0, w0 = ref_gray.shape[:2]
    if A.STAB_LEVEYS > 0 and w0 > A.STAB_LEVEYS and gray.shape[:2] == (h0, w0):
        sw = A.STAB_LEVEYS
        sh = int(round(h0 * sw / w0 / 2.0)) * 2
        sh = cv2.getOptimalDFTSize(sh) if cv2.getOptimalDFTSize(sh) % 2 == 0 else sh
        key = (id(ref_gray), sw, sh)
        ent = _stab_small_ref.get(key)
        if ent is None:
            _stab_small_ref.clear()
            ent = (cv2.resize(ref_gray, (sw, sh), interpolation=cv2.INTER_AREA), ref_gray)
            _stab_small_ref[key] = ent
        g_small = cv2.resize(gray, (sw, sh), interpolation=cv2.INTER_AREA)
        dx, dy = _phase_correlate_small(ent[0], g_small)
        return dx * (w0 / sw), dy * (h0 / sh)
    return _phase_correlate_small(ref_gray, gray)


def stabilization_matrix(dx, dy):
    """Ruutu siirretaan referenssia kohti: korjaus on -dx, -dy (phase_correlate antaa referenssin siirron ruutuun)."""
    return np.array([[1.0, 0.0, -dx], [0.0, 1.0, -dy]], dtype=np.float64)


# =====================================================================================================================
# RUUDUN VALMISTELU SEURANNALLE (vaihe B)
# =====================================================================================================================

def _estimate_photo_timed(frame_bgr, ref_bgr):
    t0 = time.perf_counter()
    res = estimate_photometric_correction(frame_bgr, ref_bgr)
    yleiset.prof_add("bg: valotasapaino taustasaikeessa (rinnan, ei lisaa)", time.perf_counter() - t0)
    return res


class LivePrep:
    """Koko ruudun valmistelu: warpAffine + remap, valotasapaino ja varjonsietoinen taustanvaimennus
    -> (frame_u, frame_u_for_tracking, diff_threshold). Ajetaan paasaikeessa (ensimmainen seurantaruutu) tai
    liukuhihnan vaiheessa B. GPU-polku (OpenCL) otetaan kayttoon jos kaytettavissa; virheessa paluu CPU:lle.
    prefix: aikamittausavaimen etuliite."""

    def __init__(self, photo_executor, fps, width, height, prefix="py:"):
        self.photo_executor = photo_executor
        self.fps = fps
        self.width, self.height = width, height
        self.photo_gain = None
        self.photo_bias = None
        self.photo_future = None
        self.next_photo_update_frame = 0
        self.photo_apply_at = 0
        self.prefix = prefix
        self._gpu_b = None

    def _gpu_init(self, frame, live_state, ref_u):
        self._gpu_b = False
        if not A.GPU_VAIHE_B:
            return
        try:
            if ref_u is not None and ref_u.shape == frame.shape:
                info = stone_tracker.gpu_b_init(
                    live_state["map1"], live_state["map2"], ref_u, float(A.GRANIITTI_EROKYNNYS),
                    float(A.VARJO_V_PUDOTUS_MIN), float(A.VARJO_V_PUDOTUS_MAX), int(A.JAA_S_MAX), int(A.JAA_V_MIN),
                )
                self._gpu_b = not info.startswith("EI KAYTETTAVISSA")
                print(f"GPU-vaihe B: {info}")
            else:
                print("GPU-vaihe B: EI KAYTETTAVISSA (kokoero)")
        except Exception as e:
            print(f"GPU-vaihe B: EI KAYTETTAVISSA ({e!r})")

    def process(self, frame, stab_matrix, frame_index, live_state, calib_result):
        pf = self.prefix
        ref_u = calib_result["calib"]["frame_undistorted"]
        t_warp0 = time.perf_counter()
        if self._gpu_b is None:
            self._gpu_init(frame, live_state, ref_u)

        frame_u = None
        if self._gpu_b:
            frame_u = stone_tracker.gpu_b_warp(frame, np.ascontiguousarray(stab_matrix, dtype=np.float64))
            if frame_u is None:
                print("GPU-vaihe B: virhe -> palataan CPU-polkuun")
                self._gpu_b = False
        if frame_u is None:
            stabilized = cv2.warpAffine(frame, stab_matrix, (self.width, self.height))
            frame_u = cv2.remap(stabilized, live_state["map1"], live_state["map2"], interpolation=cv2.INTER_LINEAR)
        yleiset.prof_add(pf + " warpAffine+remap (koko frame)", time.perf_counter() - t_warp0)

        # valotasapaino: uusi arvo kayttoon tasan A.VALO_KAYTTOON_VIIVE_RUUTUA ruudun paasta (toistettava ajo)
        if self.photo_future is not None and frame_index >= self.photo_apply_at:
            self.photo_gain, self.photo_bias = self.photo_future.result()
            self.photo_future = None
        if self.photo_gain is None:
            t_photo0 = time.perf_counter()
            self.photo_gain, self.photo_bias = estimate_photometric_correction(frame_u, ref_u)
            yleiset.prof_add(pf + " valotasapaino (alkuestimaatti, synkroninen)", time.perf_counter() - t_photo0)
            self.next_photo_update_frame = frame_index + max(1, int(round(self.fps)))
        elif self.photo_future is None and frame_index >= self.next_photo_update_frame:
            self.photo_future = self.photo_executor.submit(_estimate_photo_timed, frame_u.copy(), ref_u)
            self.photo_apply_at = frame_index + A.VALO_KAYTTOON_VIIVE_RUUTUA
            self.next_photo_update_frame = frame_index + max(1, int(round(self.fps)))

        t_shadow0 = time.perf_counter()
        frame_u_for_tracking = None
        if self._gpu_b:
            frame_u_for_tracking = stone_tracker.gpu_b_suppress(
                np.asarray(self.photo_gain, dtype=np.float64), np.asarray(self.photo_bias, dtype=np.float64),
                int(frame_u.shape[0]), int(frame_u.shape[1]),
            )
            if frame_u_for_tracking is None:
                print("GPU-vaihe B: virhe -> palataan CPU-polkuun")
                self._gpu_b = False
        if frame_u_for_tracking is None:
            frame_u_for_tracking = suppress_shadow_background(frame_u, ref_u, self.photo_gain, self.photo_bias)
        yleiset.prof_add(pf + " varjonsietoinen taustanvaimennus (koko frame)", time.perf_counter() - t_shadow0)
        # frame_u_for_tracking on jo taustanvaimennettu -> C++:n oma vaimennus ohitetaan (kynnys 0.0)
        return frame_u, frame_u_for_tracking, 0.0


# =====================================================================================================================
# LIUKUHIHNA (vaiheet A ja B omissa saikeissaan)
# =====================================================================================================================

class LivePipeline:
    """Vaiheet A ja B taustasaikeissa, paasaie ottaa valmiit ruudut get():lla. Kaytetaan kun kalibrointi ja
    kiviprofiili ovat valmiit (seurantavaihe). haku_ahead(idx, frame_u_for_tracking, thr) -> HAKU-future tai None."""

    def __init__(self, source_read, engine, prep, ref_gray, live_state, calib_result, first_index, depth, haku_ahead=None,
                 stab_harvennus=1):
        self._haku_ahead = haku_ahead
        self._stab_n = max(1, int(stab_harvennus))
        self._qa = queue.Queue(maxsize=depth)   # A -> B
        self._q = queue.Queue(maxsize=depth)    # B -> paasaie
        self._stop = threading.Event()
        self._source_read = source_read
        self._engine = engine
        self._prep = prep
        self._ref_gray = ref_gray
        self._live_state = live_state
        self._calib_result = calib_result
        self._index = first_index
        PIPE_STATS.update(wall0=time.perf_counter(), cpu0=time.process_time(), qC_sum=0, qC_n=0, qA_sum=0, qA_n=0,
                          cpu_a=None, cpu_b=None, depth=depth, stab_laskettu=0, stab_interp=0)
        self._ta = threading.Thread(target=self._run_a, daemon=True)
        self._tb = threading.Thread(target=self._run_b, daemon=True)
        self._ta.start()
        self._tb.start()

    def _put(self, q, item, key):
        t0 = time.perf_counter()
        while not self._stop.is_set():
            try:
                q.put(item, timeout=0.2)
                break
            except queue.Full:
                continue
        yleiset.prof_add(key, time.perf_counter() - t0)

    def _run_a(self):
        if self._stab_n > 1:
            return self._run_a_harvennettu()
        return self._run_a_joka_ruutu()

    def _run_a_harvennettu(self):
        """Vaihe A, mukautuva stabilointi (live): vaihekorrelaatio vain joka N. ruudusta (avainruutu). Jos kahden
        avainruudun siirrot eroavat alle A.STAB_HARVENNUS_KYNNYS_PX, valiruutujen siirto interpoloidaan lineaarisesti;
        muuten (tarina) valiruudut lasketaan kaikki rinnakkain. Tarinan jalkeen lasketaan joka ruutu, kunnes kokonainen
        jakso on rauhallinen (ruudusta ruutuun < kynnys). Viive enintaan N ruutua."""
        N, thr = self._stab_n, float(A.STAB_HARVENNUS_KYNNYS_PX)
        ex = ThreadPoolExecutor(max_workers=A.STAB_SAIKEET)
        put_key = "pipe A: odottaa vaihetta B (jono taynna)"

        def _job(gray):
            t = time.perf_counter()
            dxy = phase_correlate(self._ref_gray, gray)
            return np.asarray(dxy, dtype=np.float64), time.perf_counter() - t

        def _emit(idx, frame, dxy, dt_work, lahde):
            if dt_work > 0:
                yleiset.prof_add("bg: stabilointi tyoaika (rinnakkaiset tyontekijat, ei lisaa)", dt_work)
                PIPE_STATS["stab_laskettu"] += 1
            else:
                PIPE_STATS["stab_interp"] += 1
            stab = stabilization_matrix(float(dxy[0]), float(dxy[1]))
            t0 = time.perf_counter()
            self._engine.set_transform(stab)
            yleiset.prof_add("pipe A: set_transform", time.perf_counter() - t0)
            self._put(self._qa, (idx, frame, stab, dt_work, lahde), put_key)

        def _laske_kaikki(buf):
            t0 = time.perf_counter()
            futs = [ex.submit(_job, g) for _, _, g in buf]
            res = [f.result() for f in futs]
            yleiset.prof_add("pipe A: stabilointi (vaihekorrelaatio)", time.perf_counter() - t0)
            return res

        try:
            key = None             # viimeisimman avainruudun siirto
            buf = []               # (idx, frame, gray) avainruudun jalkeen
            tarina = False
            while not self._stop.is_set():
                t0 = time.perf_counter()
                frame = self._source_read()
                yleiset.prof_add("pipe A: read(video)", time.perf_counter() - t0)
                if frame is None or frame.size == 0:
                    for (i, f, _), (d, dt) in zip(buf, _laske_kaikki(buf)):
                        _emit(i, f, d, dt, "vaihekorrelaatio (hihna)")
                    self._put(self._qa, None, put_key)
                    return
                t0 = time.perf_counter()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                yleiset.prof_add("pipe A: gray cvtColor", time.perf_counter() - t0)
                idx = self._index
                self._index += 1
                if key is None:      # ensimmainen ruutu: avainruutu (alustaa myos referenssi-FFT:n)
                    d, dt = _job(gray)
                    _emit(idx, frame, d, dt, "vaihekorrelaatio (hihna)")
                    key = d
                    continue
                buf.append((idx, frame, gray))
                if len(buf) < N:
                    continue
                t0 = time.perf_counter()
                d_j, dt_j = _job(buf[-1][2])                     # uusi avainruutu
                yleiset.prof_add("pipe A: stabilointi (vaihekorrelaatio)", time.perf_counter() - t0)
                if tarina or float(np.hypot(*(d_j - key))) > thr:
                    res = _laske_kaikki(buf[:-1])
                    sarja = np.array([key] + [r[0] for r in res] + [d_j])
                    tarina = bool(np.hypot(*np.diff(sarja, axis=0).T).max() > thr)
                    for (i, f, _), (d, dt) in zip(buf[:-1], res):
                        _emit(i, f, d, dt, "vaihekorrelaatio (hihna)")
                else:
                    for m, (i, f, _) in enumerate(buf[:-1], 1):
                        _emit(i, f, key + (m / N) * (d_j - key), 0.0, "interpoloitu (hihna)")
                _emit(buf[-1][0], buf[-1][1], d_j, dt_j, "vaihekorrelaatio (hihna)")
                key, buf = d_j, []
        except BaseException as e:      # valitetaan eteenpain
            self._put(self._qa, e, put_key)
        finally:
            ex.shutdown(wait=False)
            PIPE_STATS["cpu_a"] = time.thread_time()

    def _run_a_joka_ruutu(self):
        """Vaihe A: luku + gray jarjestyksessa tassa saikeessa, vaihekorrelaatiot (ruudut toisistaan riippumattomia)
        A.STAB_SAIKEET-tyontekijassa; tulokset jonoon alkuperaisessa jarjestyksessa -> tulos identtinen."""
        ex = ThreadPoolExecutor(max_workers=A.STAB_SAIKEET)
        pending = collections.deque()
        put_key = "pipe A: odottaa vaihetta B (jono taynna)"

        def _job(gray):
            t = time.perf_counter()
            dxy = phase_correlate(self._ref_gray, gray)
            return dxy, time.perf_counter() - t

        def _emit_oldest():
            idx, frame, fut = pending.popleft()
            t0 = time.perf_counter()
            (dx, dy), dt_work = fut.result()
            yleiset.prof_add("pipe A: stabilointi (vaihekorrelaatio)", time.perf_counter() - t0)
            yleiset.prof_add("bg: stabilointi tyoaika (rinnakkaiset tyontekijat, ei lisaa)", dt_work)
            stab = stabilization_matrix(dx, dy)
            t0 = time.perf_counter()
            self._engine.set_transform(stab)
            yleiset.prof_add("pipe A: set_transform", time.perf_counter() - t0)
            PIPE_STATS["stab_laskettu"] += 1
            self._put(self._qa, (idx, frame, stab, dt_work, "vaihekorrelaatio (hihna)"), put_key)

        try:
            first = True
            while not self._stop.is_set():
                t0 = time.perf_counter()
                frame = self._source_read()
                yleiset.prof_add("pipe A: read(video)", time.perf_counter() - t0)
                if frame is None or frame.size == 0:
                    while pending and not self._stop.is_set():
                        _emit_oldest()
                    self._put(self._qa, None, put_key)
                    return
                t0 = time.perf_counter()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                yleiset.prof_add("pipe A: gray cvtColor", time.perf_counter() - t0)
                if first:
                    # ensimmainen ruutu synkronisesti: alustaa referenssi-FFT-valimuistin ennen rinnakkaisajoa
                    first = False
                    fut0 = ex.submit(_job, gray)
                    fut0.result()
                    pending.append((self._index, frame, fut0))
                else:
                    pending.append((self._index, frame, ex.submit(_job, gray)))
                self._index += 1
                while len(pending) >= A.STAB_SAIKEET:
                    _emit_oldest()
        except BaseException as e:      # valitetaan eteenpain
            self._put(self._qa, e, put_key)
        finally:
            ex.shutdown(wait=False)
            PIPE_STATS["cpu_a"] = time.thread_time()

    def _run_b(self):
        put_key = "pipe B: odottaa paasaiketta (jono taynna)"
        try:
            while not self._stop.is_set():
                t0 = time.perf_counter()
                PIPE_STATS["qA_sum"] += self._qa.qsize()
                PIPE_STATS["qA_n"] += 1
                try:
                    item = self._qa.get(timeout=0.2)
                except queue.Empty:
                    continue
                yleiset.prof_add("pipe B: odottaa vaihetta A", time.perf_counter() - t0)
                if item is None or isinstance(item, BaseException):
                    self._put(self._q, item, put_key)
                    return
                idx, frame, stab, stab_s, stab_lahde = item
                frame_u, frame_u_for_tracking, thr = self._prep.process(frame, stab, idx, self._live_state,
                                                                        self._calib_result)
                haku_fut = self._haku_ahead(idx, frame_u_for_tracking, thr) if self._haku_ahead is not None else None
                self._put(self._q, {"frame": frame, "stab": stab, "stab_s": stab_s, "stab_lahde": stab_lahde, "frame_u": frame_u,
                                    "frame_u_for_tracking": frame_u_for_tracking, "thr": thr, "haku_future": haku_fut},
                          put_key)
        except BaseException as e:
            self._put(self._q, e, put_key)
        finally:
            PIPE_STATS["cpu_b"] = time.thread_time()

    def get(self):
        t0 = time.perf_counter()
        PIPE_STATS["qC_sum"] += self._q.qsize()
        PIPE_STATS["qC_n"] += 1
        item = self._q.get()
        yleiset.prof_add("pipe C: paasaie odottaa hihnaa (sisaltyy py: read(video)-riviin)", time.perf_counter() - t0)
        if isinstance(item, BaseException):
            raise item
        return item

    def close(self):
        self._stop.set()
        for q in (self._q, self._qa):
            try:
                while True:
                    q.get_nowait()
            except queue.Empty:
                pass
        self._ta.join(timeout=10.0)
        self._tb.join(timeout=10.0)


class FramePrefetcher:
    """Videon luku + dekoodaus omassa saikeessaan (engine.read() vapauttaa GIL:n). Kaytetaan vain seurantavaiheessa,
    jolloin engine.add_mode_frame ei ole enaa kaytossa. Ruutujarjestys ja -sisalto identtiset."""

    def __init__(self, engine, depth=4):
        self._engine = engine
        self._q = queue.Queue(maxsize=depth)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            try:
                f = self._engine.read()
            except Exception as e:      # valitetaan paasaikeelle
                f = e
            while not self._stop.is_set():
                try:
                    self._q.put(f, timeout=0.2)
                    break
                except queue.Full:
                    continue
            if isinstance(f, Exception) or f is None or f.size == 0:
                return

    def read(self):
        f = self._q.get()
        if isinstance(f, Exception):
            raise f
        return f

    def close(self):
        self._stop.set()
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass
        self._thread.join(timeout=5.0)
