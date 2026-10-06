"""Paaputki: kalibrointi -> kiviprofiilin opettelu -> elava moni-kiven seuranta + CSV.

Kaikki vaiheet jakavat saman videon (tai kameran puskurin) perakkaisen luvun:
  1) KALIBROINTI: ensimmaisen A.KALIB_MOODI_S sekunnin ajalta kerataan A.KALIB_NAYTEVALI_S valein naytteita
     moodikuvaan (C++ ModeEngine), kamera stabiloidaan paneelien avulla (paneelit.py). Moodikuvasta kalibroidaan kamera
     (kalibrointi.py); live-tilassa kameran taydella resoluutiolla.
  2) PROFIILI: kiven 3D-profiili opetellaan liikkuvista kivista (profiili.py).
  3) SEURANTA: joka ruudulla (stabilointi moodikuvaan, esikasittely.py:n liukuhihna)
     - HAKU (C++ search_new_stones, oma saie, joka A.HAKU_VALI_RUUTUA:s ruutu) etsii uusia kivia hakualueelta
       kaukohogin ymparilta; ehdokas tarkennetaan siluetilla ja suodatetaan (jo seurattu, liikkuvan kiven takana).
     - SEURANTA (C++ track_stones_batch, kivet rinnan) paivittaa jokaisen aktiivisen kiven paikan hakualueelta, joka
       kasvaa fysikaalisen nopeusrajan mukaan; tulos tarkistetaan (siluettituki, hakualue, ei taaksepain, keskiviiva).
     - Radan elinkaari: uusi rata on ehdokas kunnes se on liikkunut A.VAHVISTUS_SIIRTYMA_CM (rivit puskuroidaan),
       vahvistetun radan rivit raaka-CSV:hen; rata loppuu kun kivi pysahtyy, katoaa, ohittaa lahihogin, liikkuu
       taaksepain tai sen tarkka-osuus putoaa. Duplikaattiradat yhdistetaan.
     - Hog-hog-analyysi, kahva ja kierteet (heitot.py, kierre.py), debug-video ja puhelinnakyma (nakyma.py).
  Lopuksi heittoportti (heitot.py) kirjoittaa varsinaisen CSV:n.
"""
import csv
import math
import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

import mode_engine
import stone_tracker

import asetukset as A
import esikasittely
import heitot
import kalibrointi
import katselu
import kierre
import kivimalli
import live
import nakyma
import paikallinen
import paneelit
import profiili
import rata
import siluetti
import yleiset

# SEURANNAN paikannus C++:ssa: "ensemble" = ristikkohaku + kaksi mean-shiftia + liike-ennuste (C++ locate_mode 5)
ENNUSTE_TAAKSE_RUUTUA = 12      # liike-ennuste viimeisista havainnoista (enintaan nain monta ruutua taaksepain)
ENNUSTE_MIN_VALI_RUUTUA = 4


def render_frame(frame_u, gain, bias, items, pose, header, results, plus_right, composer, local_pts_body, now_video_s=None,
                 cap_wall=None):
    """Debug-videon / puhelinnakyman yksi kuva (taustasaikeessa): valokorjattu kuva + kivien aariviivat/ID:t + paneelit."""
    img = esikasittely.apply_photometric_correction(frame_u, gain, bias) if gain is not None else frame_u.copy()
    img = np.ascontiguousarray(img)
    labels = []
    for bx, by, s_id, color, label in items:
        hull = kivimalli.predicted_stone_hull(local_pts_body, pose, bx, by)
        if hull is None:
            continue
        hull_i = hull.astype(np.int32)
        if composer.puhelin:
            # puhelin: ei aariviivaa; kiven ID kiven oikealla puolella (kaannetyssa kuvassa oikea = suurempi alkuperainen y)
            labels.append((float(hull_i[:, 0, 0].mean()), int(hull_i[:, 0, 1].max()), f"{s_id}", color))
            continue
        cv2.polylines(img, [hull_i], True, color, 2)
        labels.append((int(hull_i[:, 0, 0].min()), int(hull_i[:, 0, 1].min()) - 8, label, color))

    def _proj(X, Y):          # jaan piste (Z = 0) -> korjattu kuva (liukusuorat ja irroitusristi)
        u, v = kivimalli.project_3d(pose["K"], pose["R"], pose["t"], np.array([[X, Y, 0.0]]))
        return float(u[0]), float(v[0])
    composer.project = _proj
    return composer.compose(img, labels, header, results, plus_right, now_video_s, cap_wall)


def _forward_motion_cm(st):
    """Radan liike radan suuntaan (Y pienenee) historiaikkunan aikana, cm (5 ensimmaisen ja 5 viimeisen mediaanit)."""
    h = st.get("position_history") or []
    if len(h) < 10 or h[-1][0] - h[0][0] < 20:
        return 0.0
    return float(np.median([q[2] for q in h[:5]]) - np.median([q[2] for q in h[-5:]]))


def edessa_hahmo_osuus(img, pose, bx, by, r_cm):
    """Osuus kiven edessa (pienempi Y, kameran puolella) olevan jaakaistan pisteista, jotka ovat etualaa (img =
    taustanvaimennettu seurantakuva: tausta, jaa ja varjot valkoisia) - pelaajan paan edessa on vartalo, aidon kiven
    edessa tyhjaa jaata."""
    xs = bx + np.arange(-A.EDESSA_PUOLILEVEYS_CM, A.EDESSA_PUOLILEVEYS_CM + 0.1, 4.0)
    ys = by - r_cm - np.arange(A.EDESSA_VALI_CM, A.EDESSA_PITUUS_CM + 0.1, 4.0)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])
    u, v = kivimalli.project_3d(pose["K"], pose["R"], pose["t"], pts)
    h, w = img.shape[:2]
    ok = np.isfinite(u) & np.isfinite(v)
    u, v = np.round(u[ok]).astype(int), np.round(v[ok]).astype(int)
    ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    if not np.any(ok):
        return 0.0
    p = img[v[ok], u[ok]]
    etuala = (p < 255).any(axis=1) if p.ndim > 1 else p < 255
    return float(np.mean(etuala))


def is_protected_mover(st):
    """Rata liikkuu radan suuntaan (heitetty kivi): suojattu duplikaattiyhdistamisessa ja paikanvarauksessa."""
    return A.SUOJAA_ETEENPAIN_CM > 0 and _forward_motion_cm(st) >= A.SUOJAA_ETEENPAIN_CM


def pick_eviction_victim(active_stones):
    """Poistettava rata (huonoin), kun paikkoja ei ole uudelle ehdokkaalle; None jos kaikki ovat hyvia."""
    def med_rms(st):
        h = list(st.get("rms_hist", []))
        return float(np.median(h)) if h else 1e9
    movers = [st for st in active_stones if is_protected_mover(st)]
    if movers and len(movers) < len(active_stones):
        active_stones = [st for st in active_stones if not is_protected_mover(st)]
    unconfirmed = [st for st in active_stones if not st["confirmed"]]
    if unconfirmed:
        # huonoin sovitus (tai ei sovitusta) ensin; tasatilanteessa vanhin (pienin id)
        return max(unconfirmed, key=lambda st: (med_rms(st), -st["stone_id"]))
    bad = [st for st in active_stones if med_rms(st) >= A.POISTA_MIN_RMS]
    if bad:
        return max(bad, key=lambda st: med_rms(st))
    return None


def stone_prediction(stones, frame_index, half_x, half_y):
    """Kivikohtainen liike-ennuste (cm) edellisesta paikasta: vakionopeus viimeisista havainnoista, skaalattuna
    puuttuneilla ruuduilla (misses + 1) ja rajattuna hakualueen sisaan."""
    pdx, pdy = [], []
    for i, s in enumerate(stones):
        hist = [h for h in s["position_history"] if frame_index - ENNUSTE_TAAKSE_RUUTUA <= h[0] <= frame_index - 1 - s["misses"]]
        dx = dy = 0.0
        if len(hist) >= 2:
            (f0, x0, y0), (f1, x1, y1) = hist[0], hist[-1]
            if f1 - f0 >= ENNUSTE_MIN_VALI_RUUTUA:
                n = s["misses"] + 1
                dx = (x1 - x0) / (f1 - f0) * n
                dy = (y1 - y0) / (f1 - f0) * n
                dx = max(-float(half_x[i]), min(float(half_x[i]), dx))
                dy = max(-float(half_y[i]), min(float(half_y[i]), dy))
        pdx.append(dx)
        pdy.append(dy)
    return np.array(pdx, dtype=np.float64), np.array(pdy, dtype=np.float64)


class Seuranta:
    def __init__(self, video_file, panel_data, calib_diag_output, csv_output, debug_video_output=None):
        self.video_file = video_file
        self.csv_output = csv_output
        self.calib_diag_output = calib_diag_output
        self.debug_video_output = debug_video_output
        if live.active() is not None:      # ruudut kameran puskurista (sama ModeEngine, ruudut annetaan Pythonista)
            self.engine = live.LiveEngine(mode_engine, live.active(), A.MOODI_TIILI)
        else:
            self.engine = mode_engine.ModeEngine(video_file, A.MOODI_TIILI, A.VIDEO_LAITTEISTODEKOODAUS)
        e = self.engine
        self.width, self.height, self.fps, self.total_frames = e.width(), e.height(), e.fps(), e.total_frames()
        fps = self.fps
        print()
        print(f"Resoluutio: {self.width} x {self.height}")
        print(f"Ruutuja: {self.total_frames}")
        print(f"FPS: {fps:.3f}")

        # 1) kalibrointi
        self.calib_mode_max_frames = int(round(fps * A.KALIB_MOODI_S))
        self.calib_sample_every = max(1, int(round(fps * A.KALIB_NAYTEVALI_S)))
        print(f"Kalibroinnin moodikuvaan kerataan naytteet videon ensimmaisen {A.KALIB_MOODI_S:.0f} sekunnin ajalta, "
              f"yksi naytepiste {A.KALIB_NAYTEVALI_S:.0f} sekunnin valein (enintaan "
              f"{self.calib_mode_max_frames // self.calib_sample_every + 1} naytetta).")
        self.panel_data = panel_data
        self.reference_centers = [np.asarray(p["center"], dtype=np.float32).copy() for p in panel_data]
        self.histories = [[c.copy()] for c in self.reference_centers]
        self.prev_stab = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        self.stab_history = deque(maxlen=A.KALIB_STAB_MEDIAANI_RUUTUA)
        self.next_calib_sample_frame = 0
        self.hires_samples = []
        self.calib_result = None
        self.ref_gray = None                # moodikuvan harmaasavy (stabiloinnin referenssi kalibroinnin jalkeen)

        # 2) profiili
        self.oppija = None
        self.profile = None

        # 3) seuranta
        self.track_lost_max_misses = max(5, int(round(fps * A.KADONNUT_S)))
        self.stop_tracking_frames = max(1, int(round(fps * A.PYSAHTYNYT_S)))
        self.live_state = None
        self.active_stones = []
        self.stone_registry = {}            # vahvistetut kivet (stone_id -> tila)
        self.next_stone_id = 0
        self.n_confirmed_stones = 0
        self.hog_results = []
        self._stab_csv = None
        self._stab_writer = None
        self.csv_file = None
        self.csv_writer = None
        self.debug_video_writer = None
        self.debug_composer = None
        self.katselu_composer = None
        self.haku_refiner = None
        self.track_refiner = None
        self.paik = None
        self.n_window_rejects = 0
        self.n_haku_takana = 0
        self.n_haku_edessa = 0

        # saikeet ja liukuhihna
        self.panel_executor = ThreadPoolExecutor(max_workers=A.PANEELI_SAIKEET)
        self.kahva_executor = ThreadPoolExecutor(max_workers=1)      # kahvan vari + kierrepiirteet (jarjestyksessa)
        self._kahva_taysi_kesken = deque()     # (ruutu, future): taysresoluutioinen ruutu pidetaan renkaassa kunnes valmis
        self.haku_executor = ThreadPoolExecutor(max_workers=1)
        self.photo_executor = ThreadPoolExecutor(max_workers=1)
        self.live_prep = esikasittely.LivePrep(self.photo_executor, fps, self.width, self.height)
        self.frame_pipeline = None
        self.frame_prefetcher = None
        stone_tracker.set_spawn_filter(A.HAKU_UUSI_MAX_RMS, A.HAKU_UUSI_MIN_RUNKO, A.HAKU_UUSI_MAX_RUNKO,
                                       A.HAKU_UUSI_MAX_ABS_X_CM, A.HAKU_UUSI_MAX_PISTEET)
        print(f"HAKU spawn-suodatin PAALLA: rms <= {A.HAKU_UUSI_MAX_RMS}, n_body {A.HAKU_UUSI_MIN_RUNKO}..{A.HAKU_UUSI_MAX_RUNKO}, "
              f"|X| <= {A.HAKU_UUSI_MAX_ABS_X_CM} cm, pistemaara <= {A.HAKU_UUSI_MAX_PISTEET}")

        # aikamittaus
        self.start_time = time.time()
        self.n_haku_calls = 0
        self.t_haku = 0.0
        self.n_seuranta_calls = 0
        self.t_seuranta = 0.0
        self.n_seuranta_updates = 0

    # =================================================================================================================
    # PAASILMUKKA
    # =================================================================================================================
    def aja(self):
        frame_index = 0
        try:
            while True:
                t_frame0 = time.perf_counter()
                pipe_item = None
                t_read0 = time.perf_counter()
                if self.frame_pipeline is not None:
                    pipe_item = self.frame_pipeline.get()
                    frame = None if pipe_item is None else pipe_item["frame"]
                elif self.calib_result is not None and live.active() is None:
                    if self.frame_prefetcher is None:
                        self.frame_prefetcher = esikasittely.FramePrefetcher(self.engine)
                    frame = self.frame_prefetcher.read()
                else:
                    frame = self.engine.read()
                yleiset.prof_add("py: read(video)", time.perf_counter() - t_read0)
                if frame is None or frame.size == 0:
                    break
                t0 = time.perf_counter()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if pipe_item is None else None
                yleiset.prof_add("py: gray cvtColor", time.perf_counter() - t0)

                t_stab0 = time.perf_counter()
                if self.calib_result is None:
                    stab = self._paneelistabilointi(gray)
                    stab_lahde, stab_s = "paneelit", time.perf_counter() - t_stab0
                elif pipe_item is None:
                    stab = esikasittely.stabilization_matrix(*esikasittely.phase_correlate(self.ref_gray, gray))
                    stab_lahde, stab_s = "vaihekorrelaatio", time.perf_counter() - t_stab0
                else:
                    stab = pipe_item["stab"]
                    stab_lahde, stab_s = pipe_item.get("stab_lahde", "vaihekorrelaatio (hihna)"), pipe_item.get("stab_s")
                yleiset.prof_add("py: stabilointi (vaihekorrelaatio+paneelit)", time.perf_counter() - t_stab0)
                self._kirjaa_stabilointi(frame_index, stab, stab_lahde, stab_s)
                t0 = time.perf_counter()
                if pipe_item is None:
                    self.engine.set_transform(stab)
                yleiset.prof_add("py: set_transform", time.perf_counter() - t0)

                if self.calib_result is None:
                    self._kalibrointiruutu(frame_index, stab)
                elif self.profile is None:
                    if self.oppija is None:
                        self.oppija = profiili.ProfiiliOppija(self.video_file, self.calib_result, self.width, self.height,
                                                              self.fps)
                    self.profile = self.oppija.ruutu(frame_index, frame)
                else:
                    self._seurantaruutu(frame_index, frame, stab, pipe_item)

                yleiset.prof_add("FRAME_KOKO", time.perf_counter() - t_frame0)
                frame_index += 1
                if A.MAX_RUUTU and frame_index >= A.MAX_RUUTU:
                    break
                if frame_index % A.RAPORTTI_VALI_RUUTUA == 0 or frame_index == self.total_frames:
                    self._edistyminen(frame_index)
        finally:
            self._lopeta()
        return self._yhteenveto(frame_index)

    # =================================================================================================================
    # 1) KALIBROINTI
    # =================================================================================================================
    def _kirjaa_stabilointi(self, frame_index, stab, lahde, laskenta_s):
        """Joka ruudun stabilointi -> <pohja>_stabilointi.csv: siirto (px, 720p-kuvassa; ruutua siirretaan taman verran),
        kierto, skaala ja laskenta-aika (vaihekorrelaation tyoaika tai paneelien seuranta)."""
        if not A.STAB_TALLENNA or not self.csv_output:
            return
        if self._stab_csv is None:
            path = os.path.splitext(self.csv_output)[0] + "_stabilointi.csv"
            self._stab_csv = open(path, "w", newline="")
            self._stab_writer = csv.writer(self._stab_csv)
            self._stab_writer.writerow(["frame", "lahde", "dx_px", "dy_px", "kierto_deg", "skaala", "laskenta_ms"])
        M = np.asarray(stab, dtype=np.float64)
        self._stab_writer.writerow([frame_index, lahde, f"{M[0, 2]:.3f}", f"{M[1, 2]:.3f}",
                                    f"{math.degrees(math.atan2(M[1, 0], M[0, 0])):.4f}", f"{math.hypot(M[0, 0], M[1, 0]):.5f}",
                                    "" if laskenta_s is None else f"{laskenta_s * 1000:.2f}"])

    def _paneelistabilointi(self, gray):
        """Paneelien seuranta ja niista stabilointimatriisi (RANSAC, mediaanisuodatus) kalibroinnin naytteenoton ajan."""
        futures = []
        for i in range(len(self.panel_data)):
            prev = self.histories[i][-1] if self.histories[i] else self.reference_centers[i]
            futures.append(self.panel_executor.submit(paneelit.track_panel, gray, prev))
        current = []
        for i, fut in enumerate(futures):
            result = fut.result()
            if result is None:
                current.append(None)
                continue
            center = np.asarray(result["center"], dtype=np.float32)
            prev = self.histories[i][-1] if self.histories[i] else self.reference_centers[i]
            if np.linalg.norm(center - prev) <= A.KALIB_PANEELI_MAX_HYPPY_PX:
                self.histories[i].append(center.copy())
                if len(self.histories[i]) > A.KALIB_PANEELI_HISTORIA:
                    self.histories[i].pop(0)
                current.append(center)
            else:
                current.append(None)
        valid_ref = [self.reference_centers[i] for i in range(len(current)) if current[i] is not None]
        valid_cur = [c for c in current if c is not None]
        M_stab = self.prev_stab.copy()
        if len(valid_ref) >= 2:
            M, _inliers = cv2.estimateAffinePartial2D(
                np.asarray(valid_ref, dtype=np.float32), np.asarray(valid_cur, dtype=np.float32), method=cv2.RANSAC,
                ransacReprojThreshold=3.0, maxIters=2000, confidence=0.99,
            )
            if M is not None:
                M_stab = cv2.invertAffineTransform(M)
        # mediaanisuodatus perakkaisten ruutujen yli: poistaa varinan, seuraa hidasta ajautumista
        self.stab_history.append(M_stab.copy())
        M_stab = np.asarray(np.median(np.stack(self.stab_history, axis=0), axis=0), dtype=np.float64)
        self.prev_stab = M_stab.copy()
        return M_stab

    def _kalibrointiruutu(self, frame_index, stab):
        if self.calib_mode_max_frames > frame_index >= self.next_calib_sample_frame:
            self.engine.add_mode_frame()
            lv = live.active()
            if lv is not None:
                hi = lv.store.get_hires(frame_index)
                if hi is not None:
                    # sama stabilointi taydella resoluutiolla (siirto skaalattuna)
                    M = np.asarray(stab, dtype=np.float64)[:2].copy()
                    M[:, 2] *= hi.shape[1] / float(self.width)
                    self.hires_samples.append(cv2.warpAffine(hi, M, (hi.shape[1], hi.shape[0])))
            self.next_calib_sample_frame += self.calib_sample_every
        if self.next_calib_sample_frame >= self.calib_mode_max_frames:
            self._kalibroi()

    def _kalibroi(self):
        """Moodikuva naytteista (C++, taustasaie) ja kalibrointi siita (kertaluonteinen tauko)."""
        print()
        print(f"Kalibroinnin moodinaytteet ovat kasassa ({self.engine.mode_frame_count()} kpl). Lasketaan moodikuva...")
        calib_mode_output = os.path.splitext(self.video_file)[0] + "_kalibrointi_moodikuva.png"
        self.engine.start_mode_background(calib_mode_output)
        self.engine.wait_for_mode()
        print(f"Moodikuva valmis: {calib_mode_output}")
        print("Kalibroidaan moodikuvasta (calibrate_camera_from_image_with_seed + build_pose_from_calibration)...")
        calib = None
        lv = live.active()
        if lv is not None:
            lv.store.hires_every = 0        # taysresoluutioisia naytteita ei enaa tarvita
        if len(self.hires_samples) >= 3:
            try:
                hi_path = os.path.splitext(calib_mode_output)[0] + "_taysi.png"
                cv2.imwrite(hi_path, kalibrointi.hires_median(self.hires_samples))
                print(f"Kalibroidaan TAYDELLA resoluutiolla ({self.hires_samples[0].shape[1]}x{self.hires_samples[0].shape[0]}, "
                      f"{len(self.hires_samples)} naytetta): {hi_path}")
                calib_hi = kalibrointi.calibrate_camera_from_image_with_seed(hi_path)
                pose_hi = kalibrointi.build_pose_from_calibration(calib_hi)
                calib, pose = kalibrointi.scale_calibration(calib_hi, pose_hi, cv2.imread(calib_mode_output))
                print(f"  taysresoluutioinen kalibrointi skaalattu seurannan resoluutiolle ({self.width}x{self.height})")
            except Exception as e:
                print(f"  taysresoluutioinen kalibrointi epaonnistui ({e}) -> kalibroidaan {self.width}x{self.height}-moodikuvasta")
                calib = None
            self.hires_samples = []
        if calib is None:
            calib = kalibrointi.calibrate_camera_from_image_with_seed(calib_mode_output)
            pose = kalibrointi.build_pose_from_calibration(calib)
        print(f"  fokaalivali f = {pose['K'][0, 0]:.1f} px, kameran sijainti (cm): {np.round(pose['camera_position_cm'], 1)}")
        print(f"  lahemman pesan jaannosvirhe (px): keskiarvo={pose['near_reproj_err_px']['mean']:.2f}, "
              f"max={pose['near_reproj_err_px']['max']:.2f}")
        topdown = cv2.warpPerspective(calib["frame_undistorted"], calib["H_final"], (calib["output_w"], calib["output_h"]))
        cv2.imwrite(self.calib_diag_output, topdown)
        print(f"Kalibroinnin topdown-tarkistuskuva: {self.calib_diag_output}")
        self.calib_result = {"calib": calib, "pose": pose}
        katselu.set_state("Kalibrointi valmis - etsitaan liikkuvaa kivea profiilia varten")
        # tasta eteenpain jokainen ruutu stabiloidaan suoraan moodikuvan raakaversioon (ei ketjutusta)
        self.ref_gray = cv2.cvtColor(calib["frame"], cv2.COLOR_BGR2GRAY)

    # =================================================================================================================
    # 3) SEURANTA
    # =================================================================================================================
    def _aloita_seuranta(self, frame_index):
        """Seurantavaiheen alustus (kerran): kivimalli, siluettitarkennus, raaka-CSV, nakymat, live-hyppy."""
        profile = self.profile
        calib, pose = self.calib_result["calib"], self.calib_result["pose"]
        R_max, H_total, shape_deltas = profile["R_max_cm"], profile["H_total_cm"], profile["shape_deltas"]
        handle_r_frac = profile["handle_r_frac"]
        dist_coeffs = np.array([calib["best_k1"], 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        map1, map2 = kivimalli.undistort_maps(calib["camera_matrix"], dist_coeffs, (self.width, self.height))
        local_pts_body = kivimalli.build_local_stone_rings(R_max, H_total, shape_deltas, n_theta=28, n_per_segment=5)
        local_pts_search = kivimalli.build_local_stone_rings(R_max, H_total, shape_deltas, n_theta=14, n_per_segment=2)

        self.haku_refiner = siluetti.SilhouetteRefiner(pose, R_max, H_total, shape_deltas, handle_r_frac)
        print("HAKU siluettitarkennus PAALLA (C++)")
        self.track_refiner = siluetti.SilhouetteRefiner(pose, R_max, H_total, shape_deltas, handle_r_frac,
                                                        max_shift_px=A.SEURANTA_SIL_SIIRTO_PX)
        print(f"SEURANTA maskituki PAALLA (inside >= {A.SEURANTA_MIN_SISALLA}, siluettitarkennus +-{A.SEURANTA_SIL_SIIRTO_PX} px, "
              f"kaikille ruuduille kun Y > {A.SEURANTA_SIL_KAIKKI_Y_CM:.0f} cm)")
        sil_cfg = self.track_refiner.batch_config()
        stone_tracker.set_seuranta_silhouette(*sil_cfg)       # siluettitarkennus SEURANNAN C++-kivisaikeissa

        lv = live.active()
        if lv is not None:
            lv.store.unpin_all()        # profiilin havaintoruudut vapaaksi

        raw_csv_path = os.path.splitext(self.csv_output)[0] + "_raaka.csv"
        self.csv_file = open(raw_csv_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(heitot.CSV_HEADER)

        if self.debug_video_output is not None:
            vw, vh, _sc, tw = nakyma.debug_layout(self.width, self.height)
            self.debug_video_writer = nakyma.AsyncVideoWriter(nakyma.open_debug_writer(self.debug_video_output, self.fps, (tw, vh)))
            self.debug_composer = nakyma.DebugComposer(self.width, self.height)
            print(f"Debug-video: stabiloitu+korjattu, kaannetty 90 astetta vastapaivaan, {tw}x{vh} (video {vw}x{vh} + "
                  f"paneelit {nakyma.PANEL_W} px/puoli)")
        if katselu.active() is not None:
            self.katselu_composer = nakyma.DebugComposer(self.width, self.height, puhelin=True)
            self.katselu_composer.tausta = calib.get("frame_undistorted")
        katselu.set_state("Seuranta kaynnissa")
        print()
        print(f"Elava moni-kiven seuranta alkaa (frame {frame_index}, kulunut {time.time() - self.start_time:.1f} s) - "
              f"CSV: {self.csv_output}")

        paik_offset = 0
        if lv is not None:
            # kalibroinnin aikana kertynyt viive pois - seuranta alkaa uusimmasta ruudusta; hyppiville lukijoille ei enaa
            # tarvita pitkaa historiaa -> puskuri pienenee A.LIVE_TAAKSE_SEURANNASSA_S:iin
            lag0 = lv.store.lag_frames()
            to = lv.store.skip_to_latest(keep_frames=0, new_keep_back=int(A.LIVE_TAAKSE_SEURANNASSA_S * self.fps))
            paik_offset = int(to) - (frame_index + 1)      # paasilmukan ruutunumero -> puskurin indeksi hypyn jalkeen
            print(f"Live: hypataan uusimpaan ruutuun (ohitettiin {lag0} ruutua = {lag0 / self.fps:.1f} s kalibroinnin aikana "
                  f"kertynytta viivetta; puskurin indeksi {to})")
        stone_tracker.set_cv_single_thread_persistent(1)     # ei saiepoolin uudelleenluontia joka kutsulla
        if A.PAIKALLINEN_TAYSI and lv is not None:
            self.paik = paikallinen.PaikallinenTaysi.alusta(self.calib_result, lv.store, self.fps, paik_offset, frame_index,
                                                            profile, sil_cfg)
        self.live_state = {
            "map1": map1, "map2": map2, "local_pts_body": local_pts_body, "local_pts_search": local_pts_search,
            "R_max": R_max, "H_total": H_total, "handle_r_frac": handle_r_frac,
        }

    def _haku_args(self, frame_u_for_tracking, thr):
        """search_new_stones-kutsun argumentit (hakualue kaukohogin ymparilla)."""
        ls = self.live_state
        pose = self.calib_result["pose"]
        # hakualue nimellisen kaukohogin ymparilla (ei mitatun: ero +-20 cm ei vaikuta hakuun)
        hog = rata.NOMINAL_FAR_HOGLINE_Y_CM
        y_min, y_max = hog - A.HAKU_ALUE_ENNEN_HOGIA_CM, hog + A.HAKU_ALUE_HOGIN_TAKANA_CM
        return (frame_u_for_tracking, self.calib_result["calib"]["frame_undistorted"], ls["local_pts_body"],
                ls["local_pts_search"], pose["K"], pose["R"], pose["t"], 0.0, A.HAKU_ALUE_X_PUOLIKAS_CM,
                (y_min + y_max) / 2.0, (y_max - y_min) / 2.0, A.HAKU_KARKEA_ASKEL_CM, A.HAKU_HIENO_ASKEL_CM,
                A.HAKU_PISTEKYNNYS, ls["R_max"], ls["H_total"], ls["handle_r_frac"], ls["handle_r_frac"], thr)

    def _run_haku(self, *args):
        t0 = time.time()
        result = stone_tracker.search_new_stones(*args, max_results=A.HAKU_MAX_TULOKSET, max_attempts=A.HAKU_MAX_YRITYKSET)
        return result, time.time() - t0

    def _kaynnista_liukuhihna(self, frame_index):
        """Liukuhihna kaynnistetaan ensimmaisen (paasaikeessa valmistellun) seurantaruudun jalkeen."""
        if self.frame_prefetcher is None:
            self.frame_prefetcher = esikasittely.FramePrefetcher(self.engine)
        self.live_prep.prefix = "pipe B:"

        def haku_ahead(idx, fut_frame, thr):
            if idx % A.HAKU_VALI_RUUTUA != 0:
                return None
            return self.haku_executor.submit(self._run_haku, *self._haku_args(fut_frame, thr))

        self.frame_pipeline = esikasittely.LivePipeline(
            self.frame_prefetcher.read, self.engine, self.live_prep, self.ref_gray, self.live_state, self.calib_result,
            frame_index + 1, depth=A.JONON_SYVYYS, haku_ahead=haku_ahead,
            stab_harvennus=A.STAB_HARVENNUS_LIVE if live.active() is not None else 1, live=live.active() is not None,
        )
        print(f"Liukuhihna: stabilointi {A.STAB_SAIKEET} ruudulle rinnan, jonojen syvyys {A.JONON_SYVYYS}")

    def _seurantaruutu(self, frame_index, frame, stab, pipe_item):
        if self.live_state is None:
            self._aloita_seuranta(frame_index)
        if pipe_item is None:
            frame_u, frame_u_for_tracking, thr = self.live_prep.process(frame, stab, frame_index, self.live_state,
                                                                        self.calib_result)
            if self.frame_pipeline is None:
                self._kaynnista_liukuhihna(frame_index)
        else:
            frame_u, frame_u_for_tracking, thr = pipe_item["frame_u"], pipe_item["frame_u_for_tracking"], pipe_item["thr"]

        timestamp = frame_index / self.fps
        if self.paik is not None:
            kq = self._kahva_taysi_kesken
            while kq and kq[0][1].done():
                kq.popleft()
            vapauta = min(frame_index - 1, kq[0][0] - 1) if kq else frame_index - 1
            if not self.paik.ruutu_alussa(frame_index, frame, vapauta):
                self.paik = None
        pose = self.calib_result["pose"]

        haku_future = None
        if frame_index % A.HAKU_VALI_RUUTUA == 0:
            t_hs = time.perf_counter()
            if pipe_item is not None and pipe_item.get("haku_future") is not None:
                haku_future = pipe_item["haku_future"]      # kaynnistetty jo liukuhihnan vaiheessa B
            else:
                haku_future = self.haku_executor.submit(self._run_haku, *self._haku_args(frame_u_for_tracking, thr))
            yleiset.prof_add("py: HAKU submit (Python, saikeen kaynnistys)", time.perf_counter() - t_hs)

        draw_items = []
        if self.active_stones:
            self._seuranta(frame_index, timestamp, frame_u, frame_u_for_tracking, thr, stab, pose, draw_items)
        if haku_future is not None:
            self._haku_tulos(frame_index, timestamp, haku_future, frame_u_for_tracking, draw_items)

        # debug-video (taustasaikeessa) ja puhelinnakyma (kerran sekunnissa, omassa saikeessaan)
        header = f"frame {frame_index}  t={timestamp:.2f}s  kivia={len(self.active_stones)}"
        plus_right = nakyma.plus_x_is_right(pose["K"], pose["R"], pose["t"])
        lp = self.live_prep
        if self.debug_video_writer is not None:
            t0 = time.perf_counter()
            self.debug_video_writer.submit(render_frame, frame_u, lp.photo_gain, lp.photo_bias, list(draw_items), pose, header,
                                           list(self.hog_results), plus_right, self.debug_composer,
                                           self.live_state["local_pts_body"], timestamp)
            yleiset.prof_add("py: debug-video (piirto + kirjoitus)", time.perf_counter() - t0)
        kv = katselu.active()
        if kv is not None and self.katselu_composer is not None and kv.due():
            cap_w = live.active().store.capture_time(frame_index) if live.active() is not None else None
            kv.submit(render_frame, frame_u, lp.photo_gain, lp.photo_bias, list(draw_items), pose, header,
                      list(self.hog_results), plus_right, self.katselu_composer, self.live_state["local_pts_body"],
                      timestamp, cap_w)

    # ---- SEURANTA: aktiivisten kivien paivitys ----
    def _seuranta(self, frame_index, timestamp, frame_u, frame_u_for_tracking, thr, stab, pose, draw_items):
        stones = self.active_stones
        ls = self.live_state
        ref_u = self.calib_result["calib"]["frame_undistorted"]
        X0 = np.array([s["last_xy"][0] for s in stones], dtype=np.float64)
        Y0 = np.array([s["last_xy"][1] for s in stones], dtype=np.float64)
        # hakualue fysikaalisesta nopeusrajasta ja viimeisesta vahvistetusta paikasta kuluneesta ajasta
        elapsed_s = np.array([s["misses"] + 1 for s in stones], dtype=np.float64) / self.fps
        hy = np.minimum(A.SEURANTA_MAX_NOPEUS_Y_CM_S * elapsed_s, A.SEURANTA_MAX_HAKUALUE_CM)
        hx = np.minimum(A.SEURANTA_MAX_NOPEUS_Y_CM_S * A.SEURANTA_SIVUNOPEUS_OSUUS * elapsed_s, A.SEURANTA_MAX_HAKUALUE_CM)
        pred = stone_prediction(stones, frame_index, hx, hy)

        t_seur0 = time.time()
        trk_img, trk_ref, trk_K = frame_u_for_tracking, ref_u, pose["K"]
        if self.paik is not None:
            t_p0 = time.perf_counter()
            try:
                hi = self.paik.seurantakuva(frame_index, stab, X0, Y0, hx, hy, ls["R_max"], ls["H_total"], pose,
                                            self.live_prep.photo_gain, self.live_prep.photo_bias)
            except Exception as e:
                print(f"Paikallinen taysi resoluutio: virhe ({e!r}) -> pois kaytosta")
                self.paik, hi = None, None
            yleiset.prof_add("py: paikallinen taysi resoluutio (kivien alueet)", time.perf_counter() - t_p0)
            if hi is not None:
                trk_img, trk_ref, trk_K = hi
        results = stone_tracker.track_stones_batch(
            trk_img, trk_ref, X0, Y0, hx, hy, ls["local_pts_body"], ls["local_pts_search"], trk_K, pose["R"], pose["t"],
            A.SEURANTA_KARKEA_ASKEL_CM, A.SEURANTA_HIENO_ASKEL_CM, A.SEURANTA_PISTEKYNNYS, ls["R_max"], ls["H_total"],
            ls["handle_r_frac"], ls["handle_r_frac"], A.SEURANTA_MAX_TAAKSE_CM, thr,
            ms_gain=A.SEURANTA_MS_KERROIN, ms_max_iter=A.SEURANTA_MS_MAX_ITER, ms_tol_px=A.SEURANTA_MS_TOL_PX,
            ms_inner_weight=A.SEURANTA_MS_SISAPAINO, ms_margin_scale=A.SEURANTA_MS_MARGINAALI, ms_tau=A.SEURANTA_MS_TAU,
            ms_polish_step_cm=A.SEURANTA_MS_VIIMEISTELY_CM, pred_dx=pred[0], pred_dy=pred[1],
            ens_back_tol_cm=A.SEURANTA_VALINTA_TAAKSE_TOL_CM, ens_back_pen=A.SEURANTA_VALINTA_TAAKSE_SAKKO,
            ens_pred_pen=A.SEURANTA_VALINTA_ENNUSTE_SAKKO,
        )
        if self.paik is not None and trk_img is not frame_u_for_tracking:
            self.paik.skaalaa_rms(results)
        self.t_seuranta += time.time() - t_seur0
        yleiset.prof_add("py: SEURANTA dispatch+C++ seinakello", time.time() - t_seur0)
        t_post0 = time.perf_counter()
        self.n_seuranta_calls += 1
        self.n_seuranta_updates += len(stones)

        # siluettitarkennus on laskettu C++:n kivisaikeissa ("sil", set_seuranta_silhouette)
        sil = {i: tuple(r["sil"]) for i, r in enumerate(results) if r["found"]}

        still_active = []
        for si, (s, refined) in enumerate(zip(stones, results)):
            refined = self._tarkista_havainto(s, refined, sil.get(si), X0[si], Y0[si], pred[0][si], pred[1][si], hx[si], hy[si])
            if refined["found"]:
                if self._kasittele_osuma(s, refined, frame_index, timestamp, frame_u, stab, pose, draw_items):
                    still_active.append(s)
            else:
                s["misses"] += 1
                draw_items.append((s["last_xy"][0], s["last_xy"][1], s["stone_id"], (0, 0, 255), f"{s['stone_id']} MISS"))
                if s["misses"] < self.track_lost_max_misses:
                    still_active.append(s)
                elif s["confirmed"]:
                    print(f"[frame {frame_index}] Kivi {s['stone_id']} kadotettu."
                          + (" Pysahtymista ei vahvistettu (ei pysahtymispaikkaa)." if s.get("pysahdys_ehdokas") is not None else ""))
                else:
                    s["pending_rows"] = []
                    print(f"[frame {frame_index}] Ehdokas {s['stone_id']} hylatty (kadotettu ennen kuin liikkui riittavasti - "
                          "todennakoisesti jo paikallaan ollut kohde, ei aito heitto).")
        self.active_stones = still_active
        self._yhdista_duplikaatit(frame_index)
        yleiset.prof_add("py: SEURANTA jalkeen: tulossilmukka yht. (sis. varidiagn., CSV, portti)", time.perf_counter() - t_post0)

    def _tarkista_havainto(self, s, refined, sil, x0, y0, pdx, pdy, hx, hy):
        """SEURANNAN tuloksen tarkistukset: siluettituki graniittimaskista, hakualueen rajat, ei taaksepain, keskiviiva."""
        if not refined["found"]:
            return refined
        # siluettituki: kaukana (heittopaa) tarkka ruutu ei hylkaydy ja siluetti tasoittaa paikan; lahella ei-tarkka
        # ruutu hylataan ilman maskitukea (esim. pelaajan paa) ja muuten siluetti siirtaa paikan
        far = refined["Y_cm"] > A.SEURANTA_SIL_KAIKKI_Y_CM
        tx, ty, tinfo = sil
        refined = dict(refined)
        min_in = A.SEURANTA_MIN_SISALLA if (far or not refined.get("tarkka")) else A.SEURANTA_MIN_SISALLA_LAHELLA
        sil_ok = tinfo.get("ok") and tinfo.get("inside0", 0.0) >= min_in
        if far:
            if sil_ok:
                refined["X_cm"], refined["Y_cm"] = tx, ty
            elif not refined.get("tarkka"):
                refined["found"] = False
        else:
            if not sil_ok:
                refined["found"] = False
            elif not refined.get("tarkka"):
                refined["X_cm"], refined["Y_cm"] = tx, ty
        # hakualueen rajat koskevat myos lopullista (tarkennettua) paikkaa: enintaan A.SEURANTA_HAKUALUE_VARA_CM yli
        # (alue viimeisesta paikasta tai ennusteesta); tarkennus voi tarttua viereiseen kohteeseen (harja)
        if refined["found"]:
            ex = min(abs(refined["X_cm"] - x0), abs(refined["X_cm"] - (x0 + pdx))) - float(hx)
            ey = min(abs(refined["Y_cm"] - y0), abs(refined["Y_cm"] - (y0 + pdy))) - float(hy)
            if ex > A.SEURANTA_HAKUALUE_VARA_CM or ey > A.SEURANTA_HAKUALUE_VARA_CM:
                refined["found"] = False
                self.n_window_rejects += 1
        # kumulatiivinen "ei taaksepain": koko elinkaaren pienimpaan Y:hyn verrattuna (pienet askeleet eivat kerry)
        if refined["found"] and refined["Y_cm"] - s["min_y_seen"] > A.SEURANTA_MAX_TAAKSE_CM:
            refined["found"] = False
        # liian kaukana keskiviivasta -> todennakoisesti sivuttain liikkuva kohde
        if refined["found"] and abs(refined["X_cm"]) > A.SEURANTA_MAX_ABS_X_CM:
            refined["found"] = False
        return refined

    def _kasittele_osuma(self, s, refined, frame_index, timestamp, frame_u, stab, pose, draw_items):
        """Loydetty kivi: kahva/kierre, paikka, vahvistus ja CSV, hog-analyysi, lopetussaannot. True = jatkaa."""
        if frame_u is not None:
            # kahvan vari ja kierrepiirteet eivat vaikuta seurantaan -> taustasaikeessa (yksi saie, jarjestys sailyy).
            # Keraataanko piirteet, paatetaan tassa (hog-tulos asetetaan paasaikeessa).
            Y = refined["Y_cm"]
            piirre = Y <= A.KIERRE_Y_MAX_CM and "hog_result" not in s
            piirre_taysi = self.paik is not None and Y <= A.KIERRE_TAYSI_Y_MAX_CM and "hog_result" not in s
            fut = self.kahva_executor.submit(self._kahva, s, refined["X_cm"], Y, frame_index, frame_u, stab, pose,
                                             piirre, piirre_taysi)
            q = s.setdefault("kahva_kesken", deque())
            while q and q[0].done():
                q.popleft().result()
            q.append(fut)
            if piirre_taysi:
                self._kahva_taysi_kesken.append((frame_index, fut))
        s["last_xy"] = (refined["X_cm"], refined["Y_cm"])
        s.setdefault("rms_hist", deque(maxlen=10)).append(refined.get("rms_px") if refined.get("rms_px") is not None else 1e9)
        s["min_y_seen"] = min(s["min_y_seen"], refined["Y_cm"])
        s["misses"] = 0
        draw_items.append((refined["X_cm"], refined["Y_cm"], s["stone_id"], (0, 255, 0) if refined["tarkka"] else (0, 165, 255),
                           str(s["stone_id"])))

        reject_low_tarkka = False
        if not s["confirmed"]:
            reject_low_tarkka = self._vahvistus(s, refined, frame_index, timestamp)
        else:
            # jatkuva tarkka-osuustarkistus liukuvassa ikkunassa: pysyvasti matala -> SEURANTA ajautunut pelaajaan
            s["confirmed_tarkka_window"].append(bool(refined.get("tarkka")))
            window = s["confirmed_tarkka_window"]
            if len(window) >= A.VAHVISTETTU_TARKKA_IKKUNA and (sum(window) / len(window)) < A.VAHVISTETTU_MIN_TARKKA:
                reject_low_tarkka = True
                print(f"[frame {frame_index}] Kivi {s['stone_id']} lopetetaan (tarkka-osuus (liukuva ikkuna) {sum(window)}/"
                      f"{len(window)} ({100 * sum(window) / len(window):.0f}%) pudonnut pysyvasti liian matalaksi - "
                      "todennakoisesti SEURANTA ajautunut pelaajaan/lakaisijaan aidon kiven vierella).")
            else:
                s.setdefault("all_rows", []).append((frame_index, timestamp, dict(refined)))
                heitot.write_stone_csv_row(self.csv_writer, frame_index, timestamp, s["stone_id"], refined)

        history = s["position_history"]
        history.append((frame_index, refined["X_cm"], refined["Y_cm"]))
        while history[-1][0] - history[0][0] > self.stop_tracking_frames:
            history.pop(0)
        if s.get("confirmed"):
            heitot.hog_check(s, frame_index, self.fps, self.hog_results, frame_u, self.csv_output, pose,
                             odota=lambda: self._odota_kahva(s))
        stopped = self._lopetussaannot(s, history, refined, frame_index)
        return not stopped and not reject_low_tarkka

    def _odota_kahva(self, s):
        """Odottaa radan taustalla lasketut kahva- ja kierrepiirteet (ennen kuin ne luetaan)."""
        q = s.get("kahva_kesken")
        while q:
            q.popleft().result()

    def _kahva(self, s, X, Y, frame_index, frame_u, stab, pose, piirre, piirre_taysi):
        """Kahvan savyt ja kierrepiirre (720p ja paikallinen taysi resoluutio) radalle (taustasaikeessa)."""
        ls = self.live_state
        paik = self.paik
        try:
            kh = kierre.kahva_hist(frame_u, pose, X, Y, ls["H_total"] + A.KAHVA_Z_LISA_CM,
                                   ls["handle_r_frac"] * ls["R_max"], piirre=piirre)
            if kh is not None and piirre_taysi and paik is not None:
                pvt = paik.kierre_piirre(frame_index, stab, *kh[5])
                if pvt is not None:
                    klt = s.setdefault("kierre_taysi", [])
                    klt.append((frame_index, pvt))
                    if len(klt) > 1200:
                        del klt[0]
            if kh is not None:
                s.setdefault("kahva", []).append((frame_index,) + kh[:4])
                if kh[4] is not None:
                    kl = s.setdefault("kierre", [])
                    kl.append((frame_index, kh[4]))
                    if len(kl) > 900:
                        del kl[0]
        except Exception:
            pass

    def _vahvistus(self, s, refined, frame_index, timestamp):
        """Ehdokas vahvistetaan kun se on siirtynyt A.VAHVISTUS_SIIRTYMA_CM ensimmaisesta havainnosta; puskuroidut rivit
        kirjoitetaan silloin kerralla. Hylataan jos tarkka-osuus ja sovitus ovat huonoja (pelaaja). True = hylatty."""
        s["pending_rows"].append((frame_index, timestamp, dict(refined)))
        if abs(refined["Y_cm"] - s["y0_first_cm"]) < A.VAHVISTUS_SIIRTYMA_CM:
            return False
        n_pending = len(s["pending_rows"])
        n_tarkka = sum(1 for _, _, prow in s["pending_rows"] if prow.get("tarkka"))
        tarkka_frac = n_tarkka / n_pending if n_pending > 0 else 1.0
        # kaukaiset aidot heitot saavat huonon sovituksen (rms 8-10 px, tarkka=0) -> hylataan vain jos myos rms on korkea
        pre_rms = [prow.get("rms_px") for _, _, prow in s["pending_rows"] if prow.get("rms_px") is not None]
        pre_rms_med = float(np.median(pre_rms)) if pre_rms else 1e9
        if (n_pending >= A.VAHVISTUS_MIN_RUUTUJA and tarkka_frac < A.VAHVISTUS_MIN_TARKKA
                and pre_rms_med >= A.VAHVISTUS_PELASTUS_MAX_RMS):
            s["pending_rows"] = []
            print(f"[frame {frame_index}] Ehdokas {s['stone_id']} hylatty (tarkka-osuus {n_tarkka}/{n_pending} ({tarkka_frac:.0%}) "
                  "liian matala ennen liikevahvistusta - todennakoisesti pelaaja/lakaisija, ei aito kivi).")
            return True
        s["confirmed"] = True
        self.n_confirmed_stones += 1
        self.stone_registry[s["stone_id"]] = s
        for pf, pt, prow in s["pending_rows"]:
            s.setdefault("all_rows", []).append((pf, pt, dict(prow)))
            heitot.write_stone_csv_row(self.csv_writer, pf, pt, s["stone_id"], prow)
            s["confirmed_tarkka_window"].append(bool(prow.get("tarkka")))
        s["pending_rows"] = []
        return False

    def _lopetussaannot(self, s, history, refined, frame_index):
        """Radan lopetus: lahihogin ohitus, taaksepain liikkuminen, pysahtyminen. True = lopetetaan."""
        # lahihogin jalkeen ei enaa seurata (hog-analyysi kayttaa vain pisteita lahihog + 50 cm ... kaukohog - 100 cm)
        if (A.LOPETA_LAHIHOGIN_JALKEEN_CM > 0 and not A.SEURAA_PYSAHTYMISEEN and s.get("confirmed")
                and refined["Y_cm"] < rata.NEAR_HOGLINE_Y_CM - A.LOPETA_LAHIHOGIN_JALKEEN_CM):
            print(f"[frame {frame_index}] Kivi {s['stone_id']} ohitti lahi-hoglinen - lopetetaan seuranta.")
            return True
        # --full: pysahtynyt kivi vahvistetaan vasta kun se on pysynyt paikallaan A.PYSAHDYS_VAHVISTUS_S
        if s.get("pysahdys_ehdokas") is not None:
            return self._vahvista_pysahdys(s, refined, frame_index)
        # taaksepain liikkuva rata ei ole heitetty kivi (pelaaja / takaisin vietava kivi)
        if (A.LOPETA_TAAKSEPAIN_CM > 0 and len(history) >= 10
                and history[-1][0] - history[0][0] >= self.stop_tracking_frames - 1):
            y_now = float(np.median([h[2] for h in history[-5:]]))
            y_then = float(np.median([h[2] for h in history[:5]]))
            if y_now - y_then > A.LOPETA_TAAKSEPAIN_CM:
                if not s.get("confirmed"):
                    s["pending_rows"] = []
                print(f"[frame {frame_index}] Rata {s['stone_id']} liikkuu taaksepain ({y_now - y_then:.0f} cm / "
                      f"{A.PYSAHTYNYT_S:.0f} s) - ei heitetty kivi, lopetetaan seuranta.")
                return True
        # pysahtyminen: liikkunut < A.PYSAHTYNYT_CM viimeisen A.PYSAHTYNYT_S aikana
        if history[-1][0] - history[0][0] >= self.stop_tracking_frames:
            _, old_x, old_y = history[0]
            displacement = math.hypot(s["last_xy"][0] - old_x, s["last_xy"][1] - old_y)
            if displacement < A.PYSAHTYNYT_CM:
                if s["confirmed"] and A.SEURAA_PYSAHTYMISEEN:
                    return self._pysahtyi(s, history, frame_index)
                if s["confirmed"]:
                    print(f"[frame {frame_index}] Kivi {s['stone_id']} pysahtynyt (liikkunut {displacement:.1f}cm viimeisen "
                          f"{A.PYSAHTYNYT_S:.0f}s aikana) - lopetetaan seuranta.")
                else:
                    s["pending_rows"] = []
                    print(f"[frame {frame_index}] Ehdokas {s['stone_id']} hylatty (ei liikkunut riittavasti - todennakoisesti "
                          "jo paikallaan ollut kohde, ei aito heitto).")
                return True
        return False

    def _pysahtyi(self, s, history, frame_index):
        """--full: vahvistettu kivi on hidastunut (< A.PYSAHTYNYT_CM / A.PYSAHTYNYT_S). Kun se on todella pysahtynyt (5
        ensimmaisen ja 5 viimeisen paikan mediaanit < A.PYSAHTYNYT_TARKKA_CM toisistaan, tai hidas 3 x A.PYSAHTYNYT_S),
        siita tulee pysahdysehdokas (10 viimeisen paikan mediaani). Rataa seurataan edelleen (_vahvista_pysahdys).
        Palauttaa aina False (rata jatkuu)."""
        alku = s.setdefault("hidas_alku", frame_index)
        xy = np.array([(h[1], h[2]) for h in history], dtype=np.float64)
        siirto = float(np.hypot(*(np.median(xy[-5:], 0) - np.median(xy[:5], 0))))
        if siirto >= A.PYSAHTYNYT_TARKKA_CM and frame_index - alku < 3 * self.stop_tracking_frames:
            return False
        s["pysahdys_ehdokas"] = (frame_index, tuple(float(v) for v in np.median(xy[-10:], 0)))
        s["pysahdys_paikat"] = [tuple(p) for p in xy[-10:]]
        return False

    def _vahvista_pysahdys(self, s, refined, frame_index):
        """Pysahdysehdokas hyvaksytaan vain, jos kivi pysyy nakyvissa ja paikallaan (< A.PYSAHDYS_MAX_SIIRTO_CM
        ehdokaspaikasta) A.PYSAHDYS_VAHVISTUS_S. Jos kivi liikkuu (pelaaja pysaytti / vei kiven, toinen kivi osui),
        ehdokas hylataan (uusi ehdokas voi syntya, jos kivi pysahtyy uudelleen); jos rata katoaa, pysahtymispaikkaa ei
        anneta. True = rata lopetetaan (pysahtyminen vahvistettu)."""
        f0, (X0, Y0) = s["pysahdys_ehdokas"]
        s["pysahdys_paikat"].append((refined["X_cm"], refined["Y_cm"]))
        # yksittainen huono sovitus ei ole liiketta: verrataan 5 viimeisen paikan mediaania
        Xm, Ym = np.median(np.array(s["pysahdys_paikat"][-5:]), 0)
        if math.hypot(Xm - X0, Ym - Y0) > A.PYSAHDYS_MAX_SIIRTO_CM:
            print(f"[frame {frame_index}] Kivi {s['stone_id']} liikkui pysahtymisen jalkeen - ehdokas hylataan.")
            s["pysahdys_ehdokas"] = None
            s.pop("hidas_alku", None)
            return False
        if frame_index - f0 < A.PYSAHDYS_VAHVISTUS_S * self.fps:
            return False
        X, Y = (float(v) for v in np.median(np.array(s["pysahdys_paikat"][10:]), 0))   # vahvistusjakson paikat
        s["pysahtyi_xy"] = (X, Y)
        res = s.get("hog_result")
        if res is not None:
            res["pysahtyi_X_cm"], res["pysahtyi_Y_cm"] = X, Y
            res["pysahtyi_x_cm"] = X - A.PYSAHDYS_NOLLA_X_CM
            res["pysahtyi_y_cm"] = Y - (rata.NEAR_HOUSE_Y_CM + A.PYSAHDYS_NOLLA_Y_CM)
        print(f"[frame {frame_index}] Kivi {s['stone_id']} pysahtyi: X {X:+.1f} cm, Y {Y:.1f} cm "
              f"(T-viivasta {Y - rata.NEAR_HOUSE_Y_CM:+.1f} cm)")
        return True

    def _yhdista_duplikaatit(self, frame_index):
        """Kaksi rataa saman kiven paalla (< A.DUPLIKAATTI_CM, A.DUPLIKAATTI_RUUDUT perakkaista ruutua) -> huonompi pois:
        ensin pysahtynyt kuin radan suuntaan liikkuva, sitten vahvistamaton, muuten uudempi."""
        stones = self.active_stones
        if A.DUPLIKAATTI_CM <= 0 or len(stones) < 2:
            return
        losers = set()
        for ia in range(len(stones)):
            a = stones[ia]
            for ib in range(ia + 1, len(stones)):
                b = stones[ib]
                close = (a["misses"] == 0 and b["misses"] == 0
                         and math.hypot(a["last_xy"][0] - b["last_xy"][0], a["last_xy"][1] - b["last_xy"][1]) < A.DUPLIKAATTI_CM)
                cnt = a.setdefault("dup_counts", {})
                if close:
                    cnt[b["stone_id"]] = cnt.get(b["stone_id"], 0) + 1
                else:
                    cnt.pop(b["stone_id"], None)
                if close and cnt.get(b["stone_id"], 0) >= A.DUPLIKAATTI_RUUDUT:
                    ma, mb = is_protected_mover(a), is_protected_mover(b)
                    if ma != mb:
                        loser = b if ma else a
                    elif a["confirmed"] != b["confirmed"]:
                        loser = a if not a["confirmed"] else b
                    else:
                        loser = b if b["stone_id"] > a["stone_id"] else a
                    losers.add(loser["stone_id"])
        if losers:
            for s_lose in stones:
                if s_lose["stone_id"] in losers:
                    s_lose["pending_rows"] = []
                    print(f"[frame {frame_index}] Rata {s_lose['stone_id']} yhdistetty toiseen (duplikaatti, < {A.DUPLIKAATTI_CM:.0f} cm).")
            self.active_stones = [x for x in stones if x["stone_id"] not in losers]

    # ---- HAKU: uudet kivet ----
    def _haku_tulos(self, frame_index, timestamp, haku_future, frame_u_for_tracking, draw_items):
        """HAKU-tulos (taustasaie ehti laskea rinnan SEURANNAN kanssa): uudet kivet aktiivisiin."""
        with yleiset.prof("py: HAKU odotus (result) - taakka SEURANNAN jalkeen"):
            results, dt = haku_future.result()
        yleiset.prof_add("py: HAKU taustasaikeen oma kesto (ei lisaa, ajaa rinnan)", dt)
        self.t_haku += dt
        self.n_haku_calls += 1
        # siluettitarkennus: ehdokas jonka siluetissa ei ole maskitukea (esim. pelaajan paa) ei ole kivi
        for hr in results:
            if not hr.get("found"):
                continue
            x1, y1, info = self.haku_refiner.refine(frame_u_for_tracking, hr["X_cm"], hr["Y_cm"])
            if info.get("ok") and info.get("inside1", 1.0) < A.SEURANTA_MIN_SISALLA:
                hr["found"] = False
                continue
            if info.get("ok"):
                hr["X_haku"], hr["Y_haku"] = hr["X_cm"], hr["Y_cm"]
                hr["X_cm"], hr["Y_cm"] = x1, y1
        new_this_scan = []
        for hr in results:
            if not hr["found"]:
                continue
            bx, by = hr["X_cm"], hr["Y_cm"]
            stones = self.active_stones
            already = (any(math.hypot(bx - s["last_xy"][0], by - s["last_xy"][1]) < A.HAKU_JO_SEURATTU_CM
                           for s in stones if s["confirmed"])
                       or any(math.hypot(bx - nx, by - ny) < A.HAKU_SAMA_HAKU_CM for nx, ny in new_this_scan))
            edessa = None
            if not already:
                edessa = edessa_hahmo_osuus(frame_u_for_tracking, self.calib_result["pose"], bx, by,
                                            self.live_state["R_max"])
                if edessa > A.EDESSA_MAX_OSUUS:
                    already = True
                    self.n_haku_edessa += 1
                    print(f"[frame {frame_index}] HAKU-ehdokas ({bx:.1f}, {by:.1f}) hylatty: hahmo edessa "
                          f"({100 * edessa:.0f} %) - todennakoisesti pelaajan paa")
            # ehdokas liikkuvan heittokiven takana (heittaja liukuu kiven perassa, harjaajat) -> ei uutta rataa
            if not already and A.HAKU_TAKANA_MAX_CM > 0:
                for hs in stones:
                    if not (hs.get("confirmed") and is_protected_mover(hs)):
                        continue
                    dy = by - hs["last_xy"][1]
                    if A.HAKU_TAKANA_MIN_CM < dy < A.HAKU_TAKANA_MAX_CM and abs(bx - hs["last_xy"][0]) < A.HAKU_TAKANA_SIVU_CM:
                        already = True
                        self.n_haku_takana += 1
                        print(f"[frame {frame_index}] HAKU-ehdokas ({bx:.1f}, {by:.1f}) hylatty: {dy:.0f} cm liikkuvan "
                              f"heittokiven {hs['stone_id']} takana (edessa hahmoa {100 * edessa:.0f} %)")
                        break
            # paikat taynna -> huonoin rata pois uuden ehdokkaan tielta
            if not already and len(stones) >= A.MAX_KIVIA:
                victim = pick_eviction_victim(stones)
                if victim is not None:
                    victim["pending_rows"] = []
                    self.active_stones = stones = [x for x in stones if x["stone_id"] != victim["stone_id"]]
                    print(f"[frame {frame_index}] Rata {victim['stone_id']} poistettu (paikanvaraus uudelle ehdokkaalle, "
                          "huonoin sovitus).")
            if already or len(stones) >= A.MAX_KIVIA:
                continue
            stone_id = self.next_stone_id
            self.next_stone_id += 1
            new_this_scan.append((bx, by))
            stones.append({
                "stone_id": stone_id, "last_xy": (bx, by), "min_y_seen": by, "misses": 0,
                "position_history": [(frame_index, bx, by)], "y0_first_cm": by, "confirmed": False,
                "pending_rows": [(frame_index, timestamp, dict(hr))],
                "confirmed_tarkka_window": deque(maxlen=A.VAHVISTETTU_TARKKA_IKKUNA),
            })
            draw_items.append((bx, by, stone_id, (0, 255, 255), f"{stone_id} UUSI"))
            print(f"[frame {frame_index}] Uusi kivi-ehdokas {stone_id}: ({bx:.1f}, {by:.1f}) cm (odottaa liikevahvistusta "
                  f"ennen CSV-kirjausta) [ehdokasominaisuudet score={hr.get('score')} rms={hr.get('rms_px')} "
                  f"n_body={hr.get('n_body')} n_ring={hr.get('n_ring')} tarkka={hr.get('tarkka')} edessa={edessa:.2f}]")

    # =================================================================================================================
    # RAPORTIT JA LOPETUS
    # =================================================================================================================
    def _edistyminen(self, frame_index):
        elapsed = time.time() - self.start_time
        fps_now = frame_index / elapsed if frame_index > 0 else 0.0
        eta = (self.total_frames - frame_index) / fps_now if fps_now > 0 else 0.0
        live_txt = ""
        if live.active() is not None:
            st = live.active().store
            live_txt = f" | live-viive {st.lag_frames() / self.fps:.1f} s, pudotettu {st.dropped}"
            eta = 0.0
        print(f"Ruutu {frame_index} / {self.total_frames if self.total_frames > 0 else 'live'} | C++:lle tallennettuja kuvia: "
              f"{self.engine.mode_frame_count()} | nopeus: {fps_now:.1f} r/s | kulunut: {int(elapsed // 60):02d}:"
              f"{int(elapsed % 60):02d} | ETA: {int(eta // 60):02d}:{int(eta % 60):02d}{live_txt}")
        tila = ("Kalibroidaan" if self.calib_result is None else
                "Etsitaan kivea profiilia varten" if self.profile is None else "Seuranta kaynnissa")
        katselu.set_state(tila + f" - ruutu {frame_index}{live_txt.replace(' | ', ', ')}")
        n = max(1, frame_index)
        print(f"  kivenseuranta (C++): HAKU {self.n_haku_calls} kutsua, ka {self.t_haku / max(1, self.n_haku_calls) * 1000:.1f} "
              f"ms/kutsu | SEURANTA {self.n_seuranta_calls} kutsua, ka "
              f"{self.t_seuranta / max(1, self.n_seuranta_calls) * 1000:.1f} ms/kutsu "
              f"({self.n_seuranta_updates / max(1, self.n_seuranta_calls):.1f} kivea/kutsu), "
              f"{self.t_seuranta / n * 1000:.2f} ms/ruutu ka")
        bn = yleiset.pullonkaula_tiivis(esikasittely.PIPE_STATS)
        if bn:
            print(bn)

    def _lopeta(self):
        self.kahva_executor.shutdown(wait=True)
        self.panel_executor.shutdown(wait=True)
        self.haku_executor.shutdown(wait=True)
        if self.frame_pipeline is not None:
            self.frame_pipeline.close()
        self.photo_executor.shutdown(wait=True)
        if self.frame_prefetcher is not None:
            self.frame_prefetcher.close()
        if self.csv_file is not None:
            self.csv_file.close()
        if self._stab_csv is not None:
            self._stab_csv.close()
            print(f"Stabilointi ruuduittain: {os.path.splitext(self.csv_output)[0]}_stabilointi.csv")
        ps = esikasittely.PIPE_STATS
        if ps.get("stab_interp"):
            n_ = ps["stab_laskettu"] + ps["stab_interp"]
            print(f"Stabilointi (mukautuva, joka {A.STAB_HARVENNUS_LIVE}. ruutu, kynnys {A.STAB_HARVENNUS_KYNNYS_PX} px): "
                  f"laskettu {ps['stab_laskettu']} / {n_} ruutua ({100.0 * ps['stab_laskettu'] / max(1, n_):.0f} %), "
                  f"interpoloitu {ps['stab_interp']}")
        heitot.odota_kierteet()
        if self.stone_registry:
            heitot.kirjoita_heitot(self.stone_registry, self.hog_results, self.csv_output)
            try:
                kierre.kahva_kirjoita(self.stone_registry, self.hog_results, self.csv_output)
            except Exception as e:
                print(f"Kahvan varin kirjoitus epaonnistui: {e!r}")
        heitot.hog_kirjoita_csv(self.hog_results, self.csv_output)
        if self.debug_video_writer is not None:
            self.debug_video_writer.release()

    def _yhteenveto(self, frame_index):
        print()
        print(f"Videon lapikaynti valmis ({frame_index}/{self.total_frames} ruutua).")
        elapsed = time.time() - self.start_time
        n = max(1, frame_index)
        print()
        print("=== NOPEUSSEURANTA (HAKU + SEURANTA) ===")
        print(f"Koko ajo: {elapsed:.1f}s / {frame_index} ruutua ({n / elapsed:.2f} r/s keskimaarin)")
        print(f"HAKU: {self.n_haku_calls} kutsua, yhteensa {self.t_haku:.2f}s, "
              f"ka {self.t_haku / max(1, self.n_haku_calls) * 1000:.2f} ms/kutsu")
        print(f"SEURANTA: {self.n_seuranta_calls} kutsua, yhteensa {self.t_seuranta:.2f}s, "
              f"ka {self.t_seuranta / max(1, self.n_seuranta_calls) * 1000:.2f} ms/kutsu "
              f"({self.n_seuranta_updates / max(1, self.n_seuranta_calls):.2f} kivea/kutsu)")
        print(f"SEURANTA: hakualueen ulkopuolelle tarkentuneita havaintoja hylatty {self.n_window_rejects}")
        print(f"HAKU: liikkuvan heittokiven takana hylattyja ehdokkaita {self.n_haku_takana} "
              f"({A.HAKU_TAKANA_MIN_CM:.0f}-{A.HAKU_TAKANA_MAX_CM:.0f} cm takana, sivussa < {A.HAKU_TAKANA_SIVU_CM:.0f} cm)")
        print(f"HAKU: hahmo edessa (pelaajan paa) hylattyja ehdokkaita {self.n_haku_edessa} "
              f"(osuus > {A.EDESSA_MAX_OSUUS:.2f})")
        if self.paik is not None:
            self.paik.raportti()
        yleiset.aikamittausraportti(n, self.n_seuranta_updates, esikasittely.PIPE_STATS)

        if self.calib_result is None:
            raise RuntimeError("Kalibrointi ei onnistunut - video loppui kesken moodinaytteiden keruun (video liian lyhyt?).")
        profile = self.profile
        if profile is None and self.oppija is not None and self.oppija.paras_riittava is not None:
            print(f"VAROITUS: alle {A.PROFIILI_MIN_KIVIA} kivea loytyi ({self.oppija.n_kivia} kpl) ennen videon loppua - "
                  "kaytetaan viimeisinta riittavaa profiilia silti.")
            profile = self.oppija.paras_riittava
        n_obs = len(self.oppija.havainnot) if self.oppija is not None else 0
        if profile is None:
            print(f"VAROITUS: 3D-kiviprofiili ei tullut riittavaksi ennen videon loppua ({n_obs} havaintoa kerattyna).")
        return {
            "width": self.width, "height": self.height, "profile": profile, "n_profile_observations": n_obs,
            "csv_output": self.csv_output if self.live_state is not None else None,
            "n_stones_seen": self.n_confirmed_stones,
            "debug_video_output": self.debug_video_output if self.debug_video_writer is not None else None,
        }
