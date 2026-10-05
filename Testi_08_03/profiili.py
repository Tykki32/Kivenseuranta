"""Kiven 3D-profiilin opettelu videosta (kalibroinnin jalkeen, ennen seurantaa).

1) Tihea skannaus (TiheaSkannaus): pelialueelta (A.PROFIILI_ALUE_*) etsitaan kivikandidaatit C++:lla
   (scan_stone_candidates) A.PROFIILI_SKANNAUS_VALI_S valein; kandidaatit ketjutetaan lyhyiksi radoiksi. Kun jokin rata
   liikkuu radan suuntaan (> A.PROFIILI_LIIKE_CM 2 s:ssa), siita tulee siemen.
2) Siemenesta seurataan kivea ajassa molempiin suuntiin (track_stone_window): ensin lyhyt esitarkistus
   (+-A.PROFIILI_ESITARKISTUS_S, R_max ja RMS -raja), sitten koko ikkuna (+-A.PROFIILI_IKKUNA_S).
3) Havainnoista lasketaan alfa-aariviivat (kiven reuna alipikselin tarkkuudella peittavyydesta, ei binaarimaskista:
   alfa = (taustataso - k) / (taustataso - graniitti), k = V / V_moodikuva) ja hylataan huonot (tumma alue kiven
   ymparilla, pinta-ala poikkeaa lahiruuduista). Yhteinen alfa-taso = havaintojen graniitti/jaa-suhteen mediaani.
4) Yksittaisen kiven havainnoista sovitetaan profiili (kelpaako kiveksi: RMS <= A.PROFIILI_YKSI_MAX_RMS_PX);
   hyvaksytyt kerataan ja koko kokoelmaan sovitetaan profiili (C++: fit_stone_profile_cpp). Profiili on valmis kun
   RMS ja R_max ovat rajoissa ja kivia on vahintaan A.PROFIILI_MIN_KIVIA.
"""
import math
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

import stone_tracker

import asetukset as A
import esikasittely
import kivimalli
import live
import rata

# kandidaattitunnistuksen rajat (scan_stone_candidates)
SCAN_MIN_AREA = 80
SCAN_MAX_AREA = 200000
SCAN_MIN_FILL_RATIO = 0.4
SCAN_MIN_ASPECT_RATIO = 0.15
TRACK_MAX_JUMP_CM = 45.0      # suurin siirtyma perakkaisten ruutujen valilla (siemenen seuranta)
TRACK_MAX_MISSES = 10


# =====================================================================================================================
# PELIALUE
# =====================================================================================================================

def play_area_bounds():
    s = A.PROFIILI_ALUE_PARALLAKSIVARA_CM
    return (A.PROFIILI_ALUE_X_MIN_CM - s, A.PROFIILI_ALUE_X_MAX_CM + s,
            A.PROFIILI_ALUE_Y_MIN_CM - s, A.PROFIILI_ALUE_Y_MAX_CM + s)


_play_roi_cache = {}


def play_area_roi(pose, frame_w, frame_h):
    """Pelialueen (X, Y -suorakulmio, Z = 0 ... kiven korkeus) projektion rajaava suorakulmio (x0, y0, x1, y1) px."""
    key = (id(pose), frame_w, frame_h)
    roi = _play_roi_cache.get(key)
    if roi is not None:
        return roi
    x0, x1, y0, y1 = play_area_bounds()
    pts = np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (0.0, 16.0)], dtype=np.float64)
    u, v = kivimalli.project_3d(pose["K"], pose["R"], pose["t"], pts)
    ok = np.isfinite(u) & np.isfinite(v)
    if not np.any(ok):
        roi = (0, 0, frame_w, frame_h)
    else:
        m = 12
        roi = (int(max(0, math.floor(u[ok].min()) - m)), int(max(0, math.floor(v[ok].min()) - m)),
               int(min(frame_w, math.ceil(u[ok].max()) + m)), int(min(frame_h, math.ceil(v[ok].max()) + m)))
    _play_roi_cache.clear()
    _play_roi_cache[key] = roi
    return roi


def _scan_args(calib, pose, frame_w, frame_h):
    """scan_stone_candidates-kutsun argumentit kuvan jalkeen."""
    x_min, x_max, y_min, y_max = play_area_bounds()
    return (calib["frame_undistorted"], calib["H_final"], pose["K"], pose["R"], pose["t"], SCAN_MIN_AREA, SCAN_MAX_AREA,
            SCAN_MIN_FILL_RATIO, SCAN_MIN_ASPECT_RATIO, x_min, x_max, y_min, y_max, rata.PIXELS_PER_CM,
            rata.OUTPUT_X_MIN_CM, rata.OUTPUT_Y_MAX_CM, 30.0, *play_area_roi(pose, frame_w, frame_h))


def _undistort_maps(calib, frame_w, frame_h):
    dist_coeffs = np.array([calib["best_k1"], 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return kivimalli.undistort_maps(calib["camera_matrix"], dist_coeffs, (frame_w, frame_h))


# =====================================================================================================================
# 1) TIHEA SKANNAUS -> SIEMEN
# =====================================================================================================================

class TiheaSkannaus:
    """Pitaa lyhyet kandidaattiradat ja antaa siemenen (paikka, ruutu) kun jokin rata liikkuu. Skannaustulokset
    jaetaan siemenen seurannalle (cache)."""

    KEEP_S = 40.0

    def __init__(self, calib, pose, frame_w, frame_h, fps):
        self.fps = fps
        self.step_frames = max(1, int(round(fps * A.PROFIILI_SKANNAUS_VALI_S)))
        self.max_gap = max(self.step_frames, int(round(fps * 1.0)))
        self.map1, self.map2 = _undistort_maps(calib, frame_w, frame_h)
        self.args = _scan_args(calib, pose, frame_w, frame_h)
        self.cache = {}
        self.tracklets = []       # {"obs": [(ruutu, (X, Y))], "tried": bool}
        self.tried_tracks = []    # tarkistettujen ratojen {ruutu: (X, Y)}
        self.next_frame = 0
        self.n_scans = 0
        self.t_scan = 0.0

    def due(self, frame_idx):
        return frame_idx >= self.next_frame

    def _scan(self, frame):
        frame_u = cv2.remap(frame, self.map1, self.map2, interpolation=cv2.INTER_LINEAR)
        return stone_tracker.scan_stone_candidates(frame_u, *self.args)

    def mark_tried_track(self, track):
        if track:
            self.tried_tracks.append({t["frame_idx"]: t["pos_cm"] for t in track})
            del self.tried_tracks[:-20]

    def _overlaps_tried(self, f, pos):
        for tt in self.tried_tracks:
            for df in (0, -1, 1, -2, 2):
                q = tt.get(f + df)
                if q is not None and math.hypot(q[0] - pos[0], q[1] - pos[1]) < TRACK_MAX_JUMP_CM:
                    return True
        return False

    def step(self, frame_idx, frame):
        """Skannaa ruudun ja palauttaa siemenpaikan (X, Y) tai None."""
        self.next_frame = frame_idx + self.step_frames
        cands = self.cache.get(frame_idx)
        if cands is None:
            t0 = time.perf_counter()
            cands = self._scan(frame)
            self.t_scan += time.perf_counter() - t0
            self.n_scans += 1
            self.cache[frame_idx] = cands
        if len(self.cache) > 4 * self.KEEP_S * self.fps / self.step_frames:
            lim = frame_idx - int(self.KEEP_S * self.fps)
            for k in [k for k in self.cache if k < lim]:
                del self.cache[k]

        # ketjutus: lahin pari ensin (kivi liikkuu enintaan ~12 cm/ruutu)
        live_tl = [tl for tl in self.tracklets if 0 < frame_idx - tl["obs"][-1][0] <= self.max_gap]
        pairs = []
        for ti, tl in enumerate(live_tl):
            f0, p0 = tl["obs"][-1]
            lim = 30.0 + 12.0 * (frame_idx - f0)
            for ci, c in enumerate(cands):
                d = math.hypot(c["pos_cm"][0] - p0[0], c["pos_cm"][1] - p0[1])
                if d <= lim:
                    pairs.append((d, ti, ci))
        pairs.sort()
        used_t, used_c = set(), set()
        for d, ti, ci in pairs:
            if ti in used_t or ci in used_c:
                continue
            used_t.add(ti)
            used_c.add(ci)
            live_tl[ti]["obs"].append((frame_idx, tuple(cands[ci]["pos_cm"][:2])))
            del live_tl[ti]["obs"][:-50]
        for ci, c in enumerate(cands):
            if ci not in used_c:
                live_tl.append({"obs": [(frame_idx, tuple(c["pos_cm"][:2]))], "tried": False})
        self.tracklets = live_tl

        # liikkuva rata (radan suuntaan) -> siemen, suurin siirtyma ensin
        best = None
        for tl in live_tl:
            if tl["tried"] or len(tl["obs"]) < A.PROFIILI_SIEMEN_MIN_HAVAINNOT or tl["obs"][-1][0] != frame_idx:
                continue
            f1, p1 = tl["obs"][-1]
            f0, p0 = next(o for o in tl["obs"] if f1 - o[0] <= int(round(2.0 * self.fps)))
            dx, dy = p1[0] - p0[0], p1[1] - p0[1]
            disp = math.hypot(dx, dy)
            if disp < A.PROFIILI_LIIKE_CM or abs(dy) < abs(dx):
                continue
            if self._overlaps_tried(f1, p1):
                tl["tried"] = True
                continue
            if best is None or disp > best[0]:
                best = (disp, tl)
        if best is None:
            return None
        best[1]["tried"] = True
        return best[1]["obs"][-1][1]


# =====================================================================================================================
# 2) SIEMENEN SEURANTA AJASSA
# =====================================================================================================================

def track_stone_window(video_path, calib, pose, seed_frame_idx, seed_pos_cm, shared_cache):
    """Kivi seurataan siemenruudusta molempiin suuntiin (lahin kandidaatti, hyppy <= TRACK_MAX_JUMP_CM). Ruudut
    skannataan vasta kun rata niita tarvitsee (1 s palasina): rata loppuu kun kivi katoaa. shared_cache: tihean
    skannauksen tulokset (sama C++-skannaus samoille ruuduille -> sama tulos); taman seurannan skannaukset lisataan
    siihen kun seuranta on valmis. Palauttaa havaintolistan ({"ellipse", "contour", "pos_cm", "frame_idx"}) tai []
    jos esitarkistus hylkaa."""
    cache = dict(shared_cache)
    background_ref = calib["frame_undistorted"]
    cap = live.open_capture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    map1, map2 = _undistort_maps(calib, frame_w, frame_h)
    args = _scan_args(calib, pose, frame_w, frame_h)[1:]

    def _scan_one(frame):
        frame_u = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)
        return stone_tracker.scan_stone_candidates(frame_u, background_ref, *args)

    read_pos = [None]   # ruutu jonka seuraava cap.read() antaa (None = tuntematon -> haku)

    def scan_indices(indices):
        """Lukee ruudut jarjestyksessa; remap + skannaus (GIL vapaana) A.PROFIILI_SKANNAUS_SAIKEET-saikeessa rinnan."""
        indices = [i for i in indices if i not in cache]
        if not indices:
            return
        wanted = set(indices)
        if read_pos[0] != indices[0]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, indices[0])
        idx = indices[0]
        last = indices[-1]
        pending = {}
        workers = A.PROFIILI_SKANNAUS_SAIKEET
        ex = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
        try:
            while idx <= last:
                ok, frame = cap.read()
                if not ok:
                    read_pos[0] = None
                    break
                read_pos[0] = idx + 1
                if idx in wanted:
                    if ex is None:
                        cache[idx] = _scan_one(frame)
                    else:
                        pending[idx] = ex.submit(_scan_one, frame)
                        if len(pending) >= 2 * workers:      # rajoitetaan muistissa odottavia ruutuja
                            oldest = min(pending)
                            cache[oldest] = pending.pop(oldest).result()
                idx += 1
            for i in sorted(pending):
                cache[i] = pending[i].result()
        finally:
            if ex is not None:
                ex.shutdown(wait=True)
        for i in indices:      # lukematta jaaneet (videon loppu) -> ei kandidaatteja
            cache.setdefault(i, [])

    lazy_chunk = max(1, int(round(fps * 1.0)))

    def build_track(half_window_frames):
        start_frame = max(0, seed_frame_idx - half_window_frames)
        end_frame = min(n_frames - 1, seed_frame_idx + half_window_frames)

        def candidates_at(i, step=1):
            if i not in cache:
                if step > 0:
                    idxs = list(range(i, min(end_frame, i + (lazy_chunk - 1)) + 1))
                else:
                    idxs = list(range(i, max(start_frame, i - (lazy_chunk - 1)) - 1, -1))[::-1]
                scan_indices(idxs)
            return cache.get(i, [])

        seed_cands = candidates_at(seed_frame_idx)
        if not seed_cands:
            return []
        seed = min(seed_cands, key=lambda c: math.hypot(c["pos_cm"][0] - seed_pos_cm[0], c["pos_cm"][1] - seed_pos_cm[1]))
        seed["frame_idx"] = seed_frame_idx

        def track_direction(step):
            track = []
            last_pos = seed["pos_cm"]
            misses = 0
            idx = seed_frame_idx + step
            while start_frame <= idx <= end_frame and misses < TRACK_MAX_MISSES:
                cands = candidates_at(idx, step)
                if cands:
                    best = min(cands, key=lambda c: math.hypot(c["pos_cm"][0] - last_pos[0], c["pos_cm"][1] - last_pos[1]))
                    d = math.hypot(best["pos_cm"][0] - last_pos[0], best["pos_cm"][1] - last_pos[1])
                    if d <= TRACK_MAX_JUMP_CM:
                        best["frame_idx"] = idx
                        track.append(best)
                        last_pos = best["pos_cm"]
                        misses = 0
                    else:
                        misses += 1
                else:
                    misses += 1
                idx += step
            return track

        backward = track_direction(-1)
        forward = track_direction(+1)
        full_track = list(reversed(backward)) + [seed] + forward
        full_track.sort(key=lambda t: t["frame_idx"])
        return full_track

    # esitarkistus lyhyella ikkunalla: selvasti ei-kivi hylataan ennen koko ikkunan skannausta
    precheck_track = build_track(int(round(A.PROFIILI_ESITARKISTUS_S * fps)))
    n_min = A.PROFIILI_ESITARKISTUS_MIN_HAVAINNOT
    if len(precheck_track) >= n_min:
        idxs = sorted(set(np.linspace(0, len(precheck_track) - 1, min(n_min, len(precheck_track))).astype(int).tolist()))
        obs = [{"ellipse": precheck_track[i]["ellipse"], "contour": precheck_track[i]["contour"]} for i in idxs]
        _, precheck_ok = try_fit_profile(
            pose, obs, max_rms_px=A.PROFIILI_ESITARKISTUS_MAX_RMS_PX, min_samples=n_min,
            label="  esitarkistus (%.0fs)" % A.PROFIILI_ESITARKISTUS_S,
            r_min_cm=A.PROFIILI_ESITARKISTUS_R_MIN_CM, r_max_cm=A.PROFIILI_ESITARKISTUS_R_MAX_CM,
        )
        if not precheck_ok:
            cap.release()
            return []

    full_track = build_track(int(round(A.PROFIILI_IKKUNA_S * fps)))
    cap.release()
    shared_cache.update(cache)
    return full_track


# =====================================================================================================================
# 3) ALFA-AARIVIIVA
# =====================================================================================================================

def alpha_observation(frame_u, ref_u, cx, cy):
    """Alfa-kartta + laatumitat yhdelle havainnolle (C++). Palauttaa dict tai None (ei graniittia)."""
    gain, bias = esikasittely.estimate_photometric_correction(frame_u, ref_u)
    return stone_tracker.alpha_observation_cpp(
        frame_u, ref_u, np.asarray(gain, dtype=np.float64), np.asarray(bias, dtype=np.float64), int(cx), int(cy),
        float(A.GRANIITTI_EROKYNNYS), float(A.VARJO_V_PUDOTUS_MIN), float(A.VARJO_V_PUDOTUS_MAX), int(A.JAA_S_MAX),
        int(A.JAA_V_MIN), int(A.ALFA_RAJAUS_PUOLIKAS_PX), int(A.ALFA_LAAJENNUS_PX), int(A.ALFA_TUMMA_V_MAX),
    )


def alpha_contour(ao, level):
    """Kuminauha-aariviiva (liukuluku, kuvan pikselit) + pinta-ala (px) tasolta level, tai (None, 0)."""
    c, area = stone_tracker.alpha_contour_cpp(
        ao["alpha"], ao["adil"], float(ao["center"][0]), float(ao["center"][1]), int(ao["origin"][0]),
        int(ao["origin"][1]), float(level), int(A.ALFA_YLINAYTTEISTYS),
    )
    return c, float(area)


class ForwardFrameReader:
    """Ruutu indeksilla. Jos haluttu ruutu on lahella edessapain (<= A.ETEENPAINLUKU_MAX_RUUTUA), luetaan eteenpain
    (grab) - halvempi kuin haku (cap.set POS_FRAMES). Sama ruutu kummallakin tavalla."""

    def __init__(self, cap):
        self.cap = cap
        self.next_idx = None

    def read(self, idx):
        idx = int(idx)
        if self.next_idx is None or idx < self.next_idx or idx - self.next_idx > A.ETEENPAINLUKU_MAX_RUUTUA:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        else:
            while self.next_idx < idx:
                if not self.cap.grab():
                    self.next_idx = None
                    return False, None
                self.next_idx += 1
        ok, frame = self.cap.read()
        self.next_idx = idx + 1 if ok else None
        return ok, frame


def attach_alpha_observations(video_path, calib, observations):
    """Liittaa havaintoihin alfa-kartan ('alpha_obs'). Hylkaa: ei graniittia, graniitti/jaa-suhde rajojen ulkopuolella,
    tumma alue kiven ymparilla. Palauttaa (havainnot, tilasto)."""
    cap = live.open_capture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    map1, map2 = _undistort_maps(calib, w, h)
    ref_u = calib["frame_undistorted"]
    out = []
    st_ = {"in": len(observations), "ei_kuvaa": 0, "ei_graniittia": 0, "suhde": 0, "tumma": 0, "ala": 0}
    reader = ForwardFrameReader(cap)
    for o in observations:
        ok, frame = reader.read(o["frame_idx"])
        if not ok:
            st_["ei_kuvaa"] += 1
            continue
        frame_u = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)
        (cx, cy), _, _ = o["ellipse"]
        ao = alpha_observation(frame_u, ref_u, int(round(cx)), int(round(cy)))
        if ao is None:
            st_["ei_graniittia"] += 1
            continue
        if not (A.ALFA_SUHDE_MIN <= ao["ratio"] <= A.ALFA_SUHDE_MAX):
            st_["suhde"] += 1
            continue
        if ao["dark_px"] > A.ALFA_TUMMA_MAX_PX:
            st_["tumma"] += 1
            continue
        o2 = dict(o)
        o2["alpha_obs"] = ao
        out.append(o2)
    cap.release()
    return out, st_


def alpha_common_level(observations):
    """Yhteinen alfa-taso = havaintojen graniitin V / jaan V -suhteen mediaani (oletus jos ei havaintoja)."""
    r = [o["alpha_obs"]["ratio"] for o in observations if "alpha_obs" in o]
    return float(np.median(r)) if r else A.ALFA_TASO_OLETUS


def apply_area_rule(observations, level):
    """Hylkaa havainnot joiden alfa-aariviivan pinta-ala poikkeaa > A.ALFA_ALA_TOLERANSSI lahiruutujen liukuvasta
    mediaanista. Palauttaa (havainnot, hylattyjen maara)."""
    obs = sorted(observations, key=lambda o: o["frame_idx"])
    areas = np.array([alpha_contour(o["alpha_obs"], level)[1] for o in obs], dtype=np.float64)
    keep = []
    for i, o in enumerate(obs):
        lo, hi = max(0, i - A.ALFA_ALA_IKKUNA), min(len(obs), i + A.ALFA_ALA_IKKUNA + 1)
        nb = areas[lo:hi]
        nb = nb[nb > 0]
        if areas[i] <= 0 or len(nb) == 0:
            continue
        med = float(np.median(nb))
        if abs(areas[i] - med) <= A.ALFA_ALA_TOLERANSSI * med:
            keep.append(o)
    return keep, len(obs) - len(keep)


def alpha_contour_observations(observations, level):
    """Kopiot havainnoista joiden 'contour' on alfa-aariviiva tasolta level."""
    res = []
    for o in observations:
        c, _ = alpha_contour(o["alpha_obs"], level)
        if c is None or len(c) < 8:
            continue
        o2 = dict(o)
        o2["contour"] = c
        res.append(o2)
    return res


# =====================================================================================================================
# 4) PROFIILIN SOVITUS
# =====================================================================================================================

def fit_stone_profile(pose, stones):
    """Kiven profiili havaintojen aariviivoihin (C++, Levenberg-Marquardt). stones: [{"ellipse", "contour"}]."""
    positions0 = []
    for stone in stones:
        (cx, cy), _, _ = stone["ellipse"]
        positions0.append(kivimalli.stone_ground_position_z0(pose, cx, cy))
    return stone_tracker.fit_stone_profile_cpp(
        np.asarray(pose["K"], dtype=np.float64), np.asarray(pose["R"], dtype=np.float64),
        np.asarray(pose["t"], dtype=np.float64),
        [np.ascontiguousarray(s_["contour"], dtype=np.float64) for s_ in stones],
        np.asarray(positions0, dtype=np.float64),
        40, float(kivimalli.STONE_NOMINAL_RADIUS_CM), float(kivimalli.STONE_HEIGHT_CM),
        float(kivimalli.STONE_HEIGHT_MAX_CM), float(kivimalli.HANDLE_NOTCH_R_FRAC_MIN),
        float(kivimalli.HANDLE_NOTCH_R_FRAC_MAX), float(kivimalli.HANDLE_NOTCH_R_FRAC),
        float(kivimalli.STONE_SHAPE_REG_WEIGHT), 100,
    )


def try_fit_profile(pose, stones, max_rms_px=None, min_samples=None, label="profiilikoe", r_min_cm=None, r_max_cm=None):
    """Sovittaa profiilin -> (profiili, riittava). Riittava: vahintaan min_samples havaintoa, jaannosvirhe
    <= max_rms_px ja R_max rajoissa."""
    max_rms_px = A.PROFIILI_MAX_RMS_PX if max_rms_px is None else max_rms_px
    min_samples = A.PROFIILI_MIN_HAVAINNOT if min_samples is None else min_samples
    if len(stones) < min_samples:
        return None, False
    profile = fit_stone_profile(pose, stones)
    rms_ok = profile["residual_rms_px"] <= max_rms_px
    r_lo = A.PROFIILI_R_MIN_CM if r_min_cm is None else r_min_cm
    r_hi = A.PROFIILI_R_MAX_CM if r_max_cm is None else r_max_cm
    r_max_ok = r_lo <= profile["R_max_cm"] <= r_hi
    riittava = rms_ok and r_max_ok
    print(f"  {label}: {len(stones)} havaintoa, RMS={profile['residual_rms_px']:.2f}px (kynnys {max_rms_px}px), "
          f"R_max={profile['R_max_cm']:.2f}cm (sallittu {r_lo}-{r_hi}cm) -> {'RIITTAVA' if riittava else 'ei riittava'}")
    return profile, riittava


# =====================================================================================================================
# OPETTELUVAIHE (pääsilmukka kutsuu joka ruudulla kunnes profiili on valmis)
# =====================================================================================================================

class ProfiiliOppija:
    def __init__(self, video_file, calib_result, width, height, fps):
        self.video_file = video_file
        self.calib = calib_result["calib"]
        self.pose = calib_result["pose"]
        self.skannaus = TiheaSkannaus(self.calib, self.pose, width, height, fps)
        print(f"Tihea profiiliskannaus: pelialue C++:lla {self.skannaus.step_frames} ruudun valein.")
        self.havainnot = []              # hyvaksyttyjen kivien havainnot
        self.n_kivia = 0
        self.paras_riittava = None       # viimeisin riittava profiili (kaytetaan jos kivia ei tule tarpeeksi)
        self.profiili = None

    def ruutu(self, frame_index, frame):
        """Kasittelee ruudun. Palauttaa valmiin profiilin (dict) tai None."""
        if not self.skannaus.due(frame_index):
            return None
        seed_pos = self.skannaus.step(frame_index, frame)
        if seed_pos is None:
            return None
        print()
        print(f"[frame {frame_index}] Liikkuva kandidaatti loytyi kohdasta ({seed_pos[0]:.1f}, {seed_pos[1]:.1f}) cm "
              f"- seurataan koko liu'un ajan...")
        track = track_stone_window(self.video_file, self.calib, self.pose, frame_index, seed_pos, self.skannaus.cache)
        self.skannaus.mark_tried_track(track)
        print(f"  seuranta valmis: {len(track)} havaintoa.")
        if len(track) < 2:
            return None
        idxs = sorted(set(np.linspace(0, len(track) - 1, A.PROFIILI_HAVAINTOJA_KIVESTA).astype(int).tolist()))
        obs = [{"ellipse": track[i]["ellipse"], "contour": track[i]["contour"], "frame_idx": track[i]["frame_idx"],
                "pos_cm": track[i]["pos_cm"]} for i in idxs]
        if live.active() is not None:
            # havaintojen ruudut luetaan uudelleen (alfa) -> kiinnitetaan puskuriin, ettei historian karsinta poista niita
            live.active().store.pin(o["frame_idx"] for o in obs)

        # kelpaako yksittainen kivi (pelaaja tms. ei saa saastuttaa kokoelmaa)
        obs, a_st = attach_alpha_observations(self.video_file, self.calib, obs)
        level_solo = alpha_common_level(obs)
        obs, n_area_rej = apply_area_rule(obs, level_solo)
        print(f"  alfa-havainnot: {a_st['in']} -> {len(obs)} (ei graniittia {a_st['ei_graniittia']}, suhde {a_st['suhde']}, "
              f"tumma alue {a_st['tumma']}, koko {n_area_rej}); alfa-taso (tama kivi) {level_solo:.3f}")
        _, solo_ok = try_fit_profile(self.pose, alpha_contour_observations(obs, level_solo),
                                     max_rms_px=A.PROFIILI_YKSI_MAX_RMS_PX, label="  yksittaisen kandidaatin tarkistus")
        if not solo_ok:
            print("  hylatty - ei nayta kivelta (esim. pelaaja/muu liikkuva kohde), ei lisata kokoelmaan.")
            return None

        self.havainnot.extend(obs)
        self.n_kivia += 1
        alpha_level = alpha_common_level(self.havainnot)
        print(f"  yhteinen alfa-taso {alpha_level:.3f} ({len(self.havainnot)} havainnosta)")
        profile, riittava = try_fit_profile(self.pose, alpha_contour_observations(self.havainnot, alpha_level))
        if profile is not None:
            profile["alpha_level"] = alpha_level
        if not riittava:
            return None
        self.paras_riittava = profile
        if self.n_kivia < A.PROFIILI_MIN_KIVIA:
            print(f"  profiili jo riittava, mutta jatketaan viela lisaa kivia varten ({self.n_kivia}/{A.PROFIILI_MIN_KIVIA}).")
            return None
        print(f"3D-kiviprofiili riittava ({self.n_kivia} hyvaksyttya kiveä) - lopetetaan koko radan skannaus.")
        if self.skannaus.n_scans:
            print(f"  tihea skannaus: {self.skannaus.n_scans} skannausta, "
                  f"{1000.0 * self.skannaus.t_scan / self.skannaus.n_scans:.1f} ms/kpl")
        self.profiili = profile
        return profile
