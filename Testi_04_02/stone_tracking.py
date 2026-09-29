"""Liikkuvan kiven etsinta ja seuranta videosta seka varireferenssi."""

import math
import cv2
import numpy as np
import stone_tracker
from config import (
    COLOR_MATCH_MIN_VALID_POINTS,
    COLOR_REF_MIN_OBSERVATIONS,
    COLOR_REF_MIN_RING_SPACING_PX,
    OUTPUT_X_MAX_CM,
    OUTPUT_X_MIN_CM,
    OUTPUT_Y_MAX_CM,
    OUTPUT_Y_MIN_CM,
    PIXELS_PER_CM,
    PROFILE_MAX_RMS_PX,
    PROFILE_MIN_SAMPLES,
    PROFILE_R_MAX_MAX_CM,
    PROFILE_R_MAX_MIN_CM,
    STONE_MOTION_THRESHOLD_CM,
    STONE_SHEET_MARGIN_CM,
    STONE_TRACK_MAX_JUMP_CM,
    STONE_TRACK_MAX_MISSES,
    STONE_TRACK_MIN_AREA,
    STONE_TRACK_MIN_ASPECT_RATIO,
    STONE_TRACK_MIN_FILL_RATIO,
    STONE_TRACK_PRECHECK_MAX_RMS_PX,
    STONE_TRACK_PRECHECK_MIN_SAMPLES,
    STONE_TRACK_PRECHECK_WINDOW_SECONDS,
    STONE_TRACK_SAMPLE_STRIDE,
    STONE_TRACK_WINDOW_SECONDS,
)
from geometry import _project_3d
from stone_model import (
    _TEMPLATE_EQUATOR_IDX,
    _TEMPLATE_Z_FRAC,
    _bilinear_sample_vec,
    _build_undistort_maps,
    _candidates_in_frame,
    fit_stone_profile,
)
from stabilization import suppress_static_background


def _scan_stone_candidates(frame_bgr, calib, pose, background_reference=None):

    frame_bgr = suppress_static_background(
        frame_bgr, background_reference
    )

    return _candidates_in_frame(
        frame_bgr, calib, pose, calib["H_final"],
        STONE_TRACK_MIN_AREA, STONE_TRACK_MIN_FILL_RATIO,
        STONE_TRACK_MIN_ASPECT_RATIO
    )


def find_moving_candidate(candidates_prev, candidates_curr,
                           motion_threshold_cm=STONE_MOTION_THRESHOLD_CM):

    if not candidates_prev or not candidates_curr:
        return None

    for curr in candidates_curr:

        nearest = min(
            candidates_prev,
            key=lambda p: math.hypot(
                p["pos_cm"][0] - curr["pos_cm"][0],
                p["pos_cm"][1] - curr["pos_cm"][1]
            )
        )

        d = math.hypot(
            nearest["pos_cm"][0] - curr["pos_cm"][0],
            nearest["pos_cm"][1] - curr["pos_cm"][1]
        )

        if d >= motion_threshold_cm:
            return curr["pos_cm"]

    return None


def track_stone_in_video_windowed(video_path, calib, pose, seed_frame_idx,
                                   seed_pos_cm, window_seconds=STONE_TRACK_WINDOW_SECONDS,
                                   background_reference_undistorted=None):

    max_jump_cm = STONE_TRACK_MAX_JUMP_CM * STONE_TRACK_SAMPLE_STRIDE
    max_misses = STONE_TRACK_MAX_MISSES
    min_area = STONE_TRACK_MIN_AREA
    max_area = 200000
    min_fill_ratio = STONE_TRACK_MIN_FILL_RATIO
    min_aspect_ratio = STONE_TRACK_MIN_ASPECT_RATIO

    H_final = calib["H_final"]
    camera_matrix = calib["camera_matrix"]
    dist_coeffs = np.array(
        [calib["best_k1"], 0.0, 0.0, 0.0, 0.0], dtype=np.float64
    )

    # Kandidaattien skannaus (suppress_static_background + find_stone_
    # candidates + ray_plane_intersection) tehdaan C++:ssa (stone_tracker.
    # scan_stone_candidates) - katso alempana scan_indices. Rajat tasan
    # samat kuin kamera9_01.py:n find_stone_candidates:in oletusarvot.
    K = pose["K"]
    R = pose["R"]
    t = pose["t"]
    x_min = OUTPUT_X_MIN_CM - STONE_SHEET_MARGIN_CM
    x_max = OUTPUT_X_MAX_CM + STONE_SHEET_MARGIN_CM
    y_min = OUTPUT_Y_MIN_CM - STONE_SHEET_MARGIN_CM
    y_max = OUTPUT_Y_MAX_CM + STONE_SHEET_MARGIN_CM
    pixels_per_cm = PIXELS_PER_CM
    output_x_min_cm = OUTPUT_X_MIN_CM
    output_y_max_cm = OUTPUT_Y_MAX_CM

    cap = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    map1, map2 = _build_undistort_maps(
        camera_matrix, dist_coeffs, (frame_w, frame_h)
    )

    # Naytteistetyt indeksit +-half_window_frames -alueella, AINA
    # tasmalleen seed_frame_idx:sta lahtien molempiin suuntiin
    # STONE_TRACK_SAMPLE_STRIDE:n valein - talla tavalla esitarkistuksen
    # (lyhyt ikkuna) indeksit ovat AINA tasan taman saman hilan
    # osajoukko kuin taysi ikkuna, joten esitarkistuksen tulokset
    # voidaan uudelleenkayttaa suoraan taydessa skannauksessa.
    def build_indices(half_window_frames):
        idxs = list(range(
            seed_frame_idx,
            max(0, seed_frame_idx - half_window_frames) - 1,
            -STONE_TRACK_SAMPLE_STRIDE
        ))
        idxs += list(range(
            seed_frame_idx + STONE_TRACK_SAMPLE_STRIDE,
            min(n_frames - 1, seed_frame_idx + half_window_frames) + 1,
            STONE_TRACK_SAMPLE_STRIDE
        ))
        return sorted(set(i for i in idxs if 0 <= i < n_frames))

    # YKSI haku ensimmaisen halutun indeksin kohdalle (ei per-frame haku
    # - katso kamera9_04.py:n kommentti seek:in hitaudesta/epatarkkuudesta),
    # sitten sekvenssiluku - kalliit vaiheet (vaanto+taustavaimennus+
    # kandidaattitunnistus) tehdaan VAIN halutuille (naytteistetyille)
    # indekseille, muut framet vain dekoodataan (halpa) ohi.
    def scan_indices(indices):
        if not indices:
            return {}
        wanted = set(indices)
        cache = {}
        cap.set(cv2.CAP_PROP_POS_FRAMES, indices[0])
        idx = indices[0]
        last = indices[-1]
        while idx <= last:
            ok, frame = cap.read()
            if not ok:
                break
            if idx in wanted:
                frame_u = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)
                cache[idx] = stone_tracker.scan_stone_candidates(
                    frame_u, background_reference_undistorted,
                    H_final, K, R, t,
                    min_area, max_area, min_fill_ratio, min_aspect_ratio,
                    x_min, x_max, y_min, y_max,
                    pixels_per_cm, output_x_min_cm, output_y_max_cm
                )
            idx += 1
        return cache

    def build_track_from_cache(cache, half_window_frames):

        start_frame = max(0, seed_frame_idx - half_window_frames)
        end_frame = min(n_frames - 1, seed_frame_idx + half_window_frames)

        def candidates_at(i):
            return cache.get(i, [])

        seed_cands = candidates_at(seed_frame_idx)

        if not seed_cands:
            return []

        seed = min(
            seed_cands,
            key=lambda c: math.hypot(
                c["pos_cm"][0] - seed_pos_cm[0], c["pos_cm"][1] - seed_pos_cm[1]
            )
        )
        seed["frame_idx"] = seed_frame_idx

        def track_direction(step):

            track = []
            last_pos = seed["pos_cm"]
            misses = 0
            idx = seed_frame_idx + step * STONE_TRACK_SAMPLE_STRIDE

            while start_frame <= idx <= end_frame and misses < max_misses:

                cands = candidates_at(idx)

                if cands:

                    best = min(
                        cands,
                        key=lambda c: math.hypot(
                            c["pos_cm"][0] - last_pos[0], c["pos_cm"][1] - last_pos[1]
                        )
                    )

                    d = math.hypot(
                        best["pos_cm"][0] - last_pos[0], best["pos_cm"][1] - last_pos[1]
                    )

                    if d <= max_jump_cm:
                        best["frame_idx"] = idx
                        track.append(best)
                        last_pos = best["pos_cm"]
                        misses = 0
                    else:
                        misses += 1
                else:
                    misses += 1

                idx += step * STONE_TRACK_SAMPLE_STRIDE

            return track

        backward = track_direction(-1)
        forward = track_direction(+1)

        full_track = list(reversed(backward)) + [seed] + forward
        full_track.sort(key=lambda t: t["frame_idx"])

        return full_track

    # ------------------------------------------------
    # VAIHE 1: HALPA ESITARKISTUS (katso taman funktion
    # ylapuolella oleva kommentti)
    # ------------------------------------------------

    precheck_half_frames = int(round(STONE_TRACK_PRECHECK_WINDOW_SECONDS * fps))
    precheck_indices = build_indices(precheck_half_frames)
    precheck_cache = scan_indices(precheck_indices)
    precheck_track = build_track_from_cache(precheck_cache, precheck_half_frames)

    if len(precheck_track) >= STONE_TRACK_PRECHECK_MIN_SAMPLES:

        idxs = sorted(set(
            np.linspace(
                0, len(precheck_track) - 1,
                min(STONE_TRACK_PRECHECK_MIN_SAMPLES, len(precheck_track))
            ).astype(int).tolist()
        ))

        precheck_observations = [
            {"ellipse": precheck_track[i]["ellipse"], "contour": precheck_track[i]["contour"]}
            for i in idxs
        ]

        _, precheck_ok = try_fit_profile(
            pose, precheck_observations,
            max_rms_px=STONE_TRACK_PRECHECK_MAX_RMS_PX,
            min_samples=STONE_TRACK_PRECHECK_MIN_SAMPLES,
            label="  esitarkistus (%.0fs)" % STONE_TRACK_PRECHECK_WINDOW_SECONDS
        )

        if not precheck_ok:
            cap.release()
            return []

    # ------------------------------------------------
    # VAIHE 2: TAYSI IKKUNA (uudelleenkayttaa esitarkistuksen
    # jo skannaamat framet - katso build_indices:in kommentti)
    # ------------------------------------------------

    window_frames = int(round(window_seconds * fps))
    full_indices = build_indices(window_frames)
    remaining_indices = [i for i in full_indices if i not in precheck_cache]

    full_cache = dict(precheck_cache)
    full_cache.update(scan_indices(remaining_indices))

    cap.release()

    return build_track_from_cache(full_cache, window_frames)


def try_fit_profile(pose, stones, max_rms_px=PROFILE_MAX_RMS_PX,
                     min_samples=PROFILE_MIN_SAMPLES, label="profiilikoe"):
    """Yrittaa sovittaa 3D-profiilin annettuihin havaintoihin - palauttaa
    (profile, riittava_bool). Riittavyys: min_samples verran havaintoja
    JA jaannosvirhe max_rms_px:n sisalla. Kaytetaan SEKA yksittaisen
    seuratun kiven OMAN kelvollisuuden tarkistukseen (katso taman
    tiedoston alkupaan kommentti SOLO_TRACK_MAX_RMS_PX:sta) etta koko
    kerätyn kokoelman lopulliseen riittavyystarkistukseen."""

    if len(stones) < min_samples:
        return None, False

    profile = fit_stone_profile(pose, stones)

    rms_ok = profile["residual_rms_px"] <= max_rms_px

    r_max_ok = (
        PROFILE_R_MAX_MIN_CM <= profile["R_max_cm"] <= PROFILE_R_MAX_MAX_CM
    )

    riittava = rms_ok and r_max_ok

    print(
        f"  {label}: {len(stones)} havaintoa, "
        f"RMS={profile['residual_rms_px']:.2f}px "
        f"(kynnys {max_rms_px}px), "
        f"R_max={profile['R_max_cm']:.2f}cm "
        f"(sallittu {PROFILE_R_MAX_MIN_CM}-{PROFILE_R_MAX_MAX_CM}cm) -> "
        f"{'RIITTAVA' if riittava else 'ei riittava'}"
    )

    return profile, riittava


def _write_stone_csv_row(writer, frame_index, timestamp, stone_id, refined):

    writer.writerow([
        frame_index, f"{timestamp:.3f}", stone_id,
        f"{refined['X_cm'] / 100.0:.5f}", f"{refined['Y_cm'] / 100.0:.5f}",
        int(refined["tarkka"]), refined["n_body"], refined["n_ring"],
        f"{refined['rms_px']:.3f}" if refined["rms_px"] is not None else "",
        f"{refined['ring_radius_cm']:.3f}" if refined["ring_radius_cm"] is not None else "",
    ])


def _equator_row_index(n_theta, n_dense):
    """local_pts_body (build_local_stone_rings) on litistetty (n_dense,
    n_theta) -ristikko, z (korkeus) hitaammin vaihtuvana - palauttaa
    RIVIN INDEKSIN joka vastaa "paivaa" (leveimmalla kohdalla,
    _TEMPLATE_EQUATOR_IDX) - sama rengas jolla graniitin nauha nakyy."""

    z_equator_frac = float(_TEMPLATE_Z_FRAC[_TEMPLATE_EQUATOR_IDX])
    z_frac_dense = np.linspace(0.0, 1.0, n_dense)
    return int(np.argmin(np.abs(z_frac_dense - z_equator_frac)))


def _ring_spacing_px(u, v, n_theta, equator_row):
    """"Paiva"-renkaan vierekkaisten theta-pisteiden mediaanietaisyys
    kuvassa (px) - kertoo onko kivi kuvassa riittavan iso hienon
    pintakuvion (esim. nauhan) luotettavaan erottamiseen. Katso
    kommentti COLOR_REF_*-vakioiden kohdalla."""

    n_dense = len(u) // n_theta
    u_grid = u.reshape(n_dense, n_theta)
    v_grid = v.reshape(n_dense, n_theta)
    ring_u = u_grid[equator_row]
    ring_v = v_grid[equator_row]
    du = np.diff(np.concatenate([ring_u, ring_u[:1]]))
    dv = np.diff(np.concatenate([ring_v, ring_v[:1]]))
    return float(np.median(np.hypot(du, dv)))


def build_stone_color_reference(video_file, calib, pose, local_pts_body,
                                 n_theta, observations):
    """Rakentaa kiven OMAN pintavarireferenssin PER PISTE local_pts_
    body:sta (sama pistejoukko jota elava SEURANTA kayttaa) hyvaksytyn
    3D-profiilin havainnoista - katso kommentti COLOR_REF_*-vakioiden
    kohdalla taman tiedoston alkupaassa. Lukee havainnoista uudelleen
    todelliset videoruudut (frame_idx/pos_cm) naytteistaakseen oikean
    pikselivarin jokaisessa pisteessa - EI ainoastaan ellipse/contour
    joita profiilisovitus kaytti.

    Palauttaa None jos yhtaan pistetta ei saatu riittavan monesta
    (COLOR_REF_MIN_OBSERVATIONS) resoluutioltaan kelvollisesta
    havainnosta - ominaisuus jaa tallöin hiljaisesti pois kaytosta.
    """

    n_pts = local_pts_body.shape[0]
    n_dense = n_pts // n_theta
    equator_row = _equator_row_index(n_theta, n_dense)

    camera_matrix = calib["camera_matrix"]
    dist_coeffs = np.array(
        [calib["best_k1"], 0.0, 0.0, 0.0, 0.0], dtype=np.float64
    )
    frame_h, frame_w = calib["frame_undistorted"].shape[:2]
    map1, map2 = _build_undistort_maps(
        camera_matrix, dist_coeffs, (frame_w, frame_h)
    )

    color_sum = np.zeros((n_pts, 3), dtype=np.float64)
    counts = np.zeros(n_pts, dtype=np.int64)
    n_observations_used = 0
    n_observations_skipped_resolution = 0

    cap = cv2.VideoCapture(video_file)

    for obs in observations:

        if "frame_idx" not in obs or "pos_cm" not in obs:
            continue

        X0, Y0 = obs["pos_cm"]
        pts = local_pts_body + np.array([X0, Y0, 0.0])
        u, v = _project_3d(pose["K"], pose["R"], pose["t"], pts)

        if not (np.all(np.isfinite(u)) and np.all(np.isfinite(v))):
            continue

        # --------------------------------
        # RESOLUUTIOTARKISTUS: katso kommentti COLOR_REF_*-vakioiden
        # kohdalla - "paiva"-renkaan (leveimmalla kohdalla, sama rengas
        # jolla graniitin nauha nakyy) vierekkaisten theta-pisteiden
        # mediaanietaisyys kuvassa kertoo onko kivi kuvassa riittavan
        # iso hienon pintakuvion luotettavaan erottamiseen - jos ei,
        # koko havainto hylataan (ei vain nauhaosalta - muutenkin
        # epaluotettava).
        # --------------------------------

        ring_spacing_px = _ring_spacing_px(u, v, n_theta, equator_row)

        if ring_spacing_px < COLOR_REF_MIN_RING_SPACING_PX:
            n_observations_skipped_resolution += 1
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(obs["frame_idx"]))
        ok, frame = cap.read()

        if not ok:
            continue

        frame_u = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)

        valid = (
            (u >= 0) & (u < frame_w - 1) &
            (v >= 0) & (v < frame_h - 1)
        )

        b = _bilinear_sample_vec(frame_u[:, :, 0].astype(np.float64), u, v)
        g = _bilinear_sample_vec(frame_u[:, :, 1].astype(np.float64), u, v)
        r = _bilinear_sample_vec(frame_u[:, :, 2].astype(np.float64), u, v)

        pt_valid = valid & np.isfinite(b) & np.isfinite(g) & np.isfinite(r)

        if not np.any(pt_valid):
            continue

        # kirkkausnormalisointi: jakaa taman havainnon naytteet OMALLA
        # keskikirkkaudellaan, jotta valaistuksen/altistuksen vaihtelu
        # eri havaintojen valilla ei vaikuta - vain suhteelliset varit
        # sailyvat.
        brightness = float(np.mean([b[pt_valid], g[pt_valid], r[pt_valid]]))

        if brightness < 1e-6:
            continue

        colors = np.stack([b, g, r], axis=-1) / brightness

        color_sum[pt_valid] += colors[pt_valid]
        counts[pt_valid] += 1
        n_observations_used += 1

    cap.release()

    print(
        f"  varireferenssi: {n_observations_used} havaintoa kaytetty, "
        f"{n_observations_skipped_resolution} hylatty (kivi liian "
        "pieni kuvassa)."
    )

    valid_mask = counts >= COLOR_REF_MIN_OBSERVATIONS

    if n_observations_used == 0 or not np.any(valid_mask):

        print(
            "  varireferenssia ei saatu (ei riittavasti lahelta "
            "otettuja havaintoja) - ominaisuus pois kaytosta talla "
            "ajolla."
        )

        return None

    reference_color = np.zeros((n_pts, 3), dtype=np.float64)
    reference_color[valid_mask] = (
        color_sum[valid_mask] / counts[valid_mask, None]
    )

    print(
        f"  varireferenssi valmis: {int(np.count_nonzero(valid_mask))}/"
        f"{n_pts} pistetta kelvollista."
    )

    return {
        "reference_color": reference_color,
        "valid_mask": valid_mask,
        "n_theta": n_theta,
        "equator_row": equator_row,
    }


def color_match_median_diff(frame_u_f64, local_pts_body, pose, X0, Y0,
                             color_ref, frame_w, frame_h):
    """Vertaa SEURANTA-osuman (X0,Y0) todellista pintavaria kiven
    varireferenssiin (build_stone_color_reference) - katso kommentti
    COLOR_REF_*/COLOR_MATCH_*-vakioiden kohdalla. Palauttaa None jos
    liian vahan vertailukelpoisia pisteita (esim. kivi liian kaukana/
    pieni) - tallöin tarkistusta EI voida tehda luotettavasti (osuma
    hyvaksytaan kutsukohdassa oletuksena)."""

    pts = local_pts_body + np.array([X0, Y0, 0.0])
    u, v = _project_3d(pose["K"], pose["R"], pose["t"], pts)

    if not (np.all(np.isfinite(u)) and np.all(np.isfinite(v))):
        return None

    # --------------------------------
    # RESOLUUTIOTARKISTUS (osuman OMA, EI referenssin rakennusaikainen):
    # katso kommentti COLOR_REF_*-vakioiden kohdalla - jos TAMA osuma on
    # kuvassa liian pieni (esim. kivi viela kaukana), vertailua ei voida
    # tehda luotettavasti vaikka referenssi itse olisikin kunnossa -
    # tarkistus jaa siis pois kaytosta talla osumalla (hyvaksytaan
    # oletuksena), ei vain kun referenssia rakennettiin.
    # --------------------------------

    ring_spacing_px = _ring_spacing_px(
        u, v, color_ref["n_theta"], color_ref["equator_row"]
    )

    if ring_spacing_px < COLOR_REF_MIN_RING_SPACING_PX:
        return None

    valid = (
        color_ref["valid_mask"] &
        (u >= 0) & (u < frame_w - 1) &
        (v >= 0) & (v < frame_h - 1)
    )

    if np.count_nonzero(valid) < COLOR_MATCH_MIN_VALID_POINTS:
        return None

    b = _bilinear_sample_vec(frame_u_f64[:, :, 0], u, v)
    g = _bilinear_sample_vec(frame_u_f64[:, :, 1], u, v)
    r = _bilinear_sample_vec(frame_u_f64[:, :, 2], u, v)

    pt_valid = valid & np.isfinite(b) & np.isfinite(g) & np.isfinite(r)

    if np.count_nonzero(pt_valid) < COLOR_MATCH_MIN_VALID_POINTS:
        return None

    brightness = float(np.mean([b[pt_valid], g[pt_valid], r[pt_valid]]))

    if brightness < 1e-6:
        return None

    colors = np.stack([b, g, r], axis=-1) / brightness
    diff = np.abs(colors[pt_valid] - color_ref["reference_color"][pt_valid])
    per_point_diff = np.mean(diff, axis=1)

    # MEDIANI (EI keskiarvo) yli pisteiden: kayttajan aiemman
    # vaatimuksen mukaisesti (heittaja/lakaisija saa nakya kiven
    # PAALLA/vieressa) osa pisteista voi olla oikeasti graniittia
    # peittavan ihmisen varinen ilman etta osuma on vaara - mediaani
    # sietaa vahemmiston (esim. jalka ohittaa kiven) vaikuttamatta
    # kokonaistulokseen, kun taas KESKIARVO vaaristyisi jo muutamasta
    # peittyneesta pisteesta.
    return float(np.median(per_point_diff))
