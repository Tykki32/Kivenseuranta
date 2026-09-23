import os
import sys
import math
import csv
import time
import argparse
import importlib.util
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from concurrent.futures import ThreadPoolExecutor
import mode_engine
import stone_tracker
from collections import deque
import subprocess


# ============================================================
# MUUT PROJEKTIN TIEDOSTOT (kamera8_01.py/kamera9_01.py/.../kamera9_04.py)
# - kayttajan pyynnosta (2026-09) siirretty TAHAN kansioon OMIKSI
# kopioikseen, jotta Testi_01_01 ei enaa riipu Kivenseuranta-
# juurikansion versioista (jotka jaavat koskemattomiksi arkisto-
# referensseiksi muuta kehitysta varten). Ladataan silti dynaamisesti
# (ei tavallisella import-lauseella) - sama periaate kuin kamera9_02.py
# -> kamera9_04.py -ketjussa NAIDEN OMIEN kopioiden sisalla (kukin
# lataa seuraavan SAMASTA kansiosta kuin itse on - katso niiden omat
# _load_kamera*-funktiot): uudelleenkaytetaan jo validoitua koodia
# (automaattinen kalibrointi, 3D-kiviprofiilin sovitus, nopeutettu
# yhteissovitus) sen sijaan etta kirjoitettaisiin se uudelleen.
# ============================================================

def _load_project_module(module_name, filename):
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


k94 = _load_project_module("k94_ref", "kamera9_04.py")
k93 = k94.k93
k92 = k94.k92
k9 = k94.k9
k8 = k94.k8


# ============================================================
# KAUKAISEN PESAN LOYTAMISEN VARMISTAMINEN (kayttajan pyynnosta) -
# koskematon kamera9_01.py:n calibrate_camera_from_image tekee sokean
# ristikkohaun (COARSE-vaihe) AINA saman kiinteän oletuksen (X=0, Y=0
# eli tasan FAR_HOUSE_Y_CM:n paassa, keskilinjalla) ymparilta. Uudella
# kamerakulmalla (kayttajan lahettama leikattu_kalibrointi_moodikuva.
# png) tama oletus osui liian kauas oikeasta sijainnista, ja COARSE-
# haku lukkiutui vaarn kohteeseen (havaittu: skaala pieneni jokaisella
# hakuvaiheella 0.7->0.5->0.42, mika ei tapahdu OIKEAAN kaukaiseen
# pesaan lukkiutuessa - katso alla oleva calibrate_camera_from_image_
# with_seed, joka TOISTAA kamera9_01.py:n calibrate_camera_from_image:
# in TASMALLEEN, MUUTTAEN VAIN COARSE-vaiheen alkuarvausta).
#
# Alkuarvaus haetaan etsimalla KUVASTA KAIKKI pienet sinireunaiset
# rengaskandidaatit (sama k8.find_house_ellipses jota lahemman pesan
# tunnistus jo kayttaa, vain paljon pienemmalla min_area:lla ja
# lahempi pesa itse poissuljettuna) ja kaantamalla jokaisen keskipiste
# jaatasolle (Z=0) TASMALLEEN k8.project_point:in oman, koskemattoman
# kameramallin avulla - katso unproject_to_ice_plane. Fysikaalisesti
# uskottavin (Y lahinna FAR_HOUSE_Y_CM:aa) kandidaatti kaytetaan
# alkuarvauksena COARSE-haulle (SAMALLA hakualueella/-tarkkuudella
# kuin ennen - vain KESKIPISTE muuttuu). Jos mitaan uskottavaa
# kandidaattia ei loydy, palataan vanhaan (0,0)-oletukseen - EI
# regressiota jos kuvassa ei ole naita pieniä kandidaatteja lainkaan.
#
# Validoitu: testattu SEKA uudella kamerakulmalla (leikattu_
# kalibrointi_moodikuva.png - korjasi virheellisen kalibroinnin,
# RMS 7.6-22cm -> 5.4-8.5cm, skaala vakautui 0.47:aan eika enaa
# laskenut jokaisella kierroksella, topdown-kuvassa nyt yksi selkea
# kaukainen rengaskohde aiemman kaksoiskuvan sijaan) ETTA vanhalla,
# jo toimivalla kamerakulmalla (00011 - Trim.mp4:n oma ruutu - tuotti
# lahes identtisen topdown-kuvan ja laadun kuin koskematon, seedaamaton
# versio: RMS 5.39cm->5.39cm, worst_ratio 0.966->0.982).
# ============================================================

def unproject_to_ice_plane(px, py, camera, v_t, v_cl):
    """Kaantaa k8.project_point:in (koskematon, kamera9_01.py:n
    calibrate_camera_from_image:in sisainen kameramalli) - pikselista
    (px,py) arvioitu (X,Y) jaatasolla (Z=0). v_t/v_cl ovat ortonormaalit
    (katso k8.build_image_directions), joten projektio niille on pelkkä
    pistetulo - suljetun muodon kaannos on siis suoraviivainen."""

    diff = np.array([px, py], dtype=np.float64) - camera["center"]
    local_u = float(np.dot(diff, v_t))
    local_v = float(np.dot(diff, v_cl))

    f = camera["f"]
    theta = camera["theta"]
    camera_height = camera["camera_height"]
    near_depth = k8.NEAR_HOUSE_Y_CM * math.sin(theta) + camera_height

    denom = f * math.cos(theta) - local_v * math.sin(theta)

    if abs(denom) < 1e-8:
        return None

    relative_Y = local_v * near_depth / denom
    Y = relative_Y + k8.NEAR_HOUSE_Y_CM

    Zc = near_depth + relative_Y * math.sin(theta)

    if Zc <= 1e-8:
        return None

    X = local_u * Zc / f

    return X, Y


def find_far_house_coarse_seed(blue_mask, near_blue_outer, camera, v_t, v_cl,
                                min_area=20, min_ratio=0.5,
                                exclude_radius_px=None,
                                max_y_error_cm=1000.0, max_x_abs_cm=400.0):
    """Etsii PIENIA sinisia rengaskandidaatteja (paitsi jo tunnettu
    lahempi pesa) koko kuvasta kaukaisen pesan COARSE-haun alkuarvaukseksi
    - katso kommentti taman tiedoston alkupaassa. Palauttaa None jos
    mitaan fysikaalisesti uskottavaa (Y lahella FAR_HOUSE_Y_CM:aa, X
    jarkevan lahella keskilinjaa) ei loytynyt - haku kayttaa tallöin
    vanhaa (0,0)-oletusta (ei regressiota)."""

    if exclude_radius_px is None:
        exclude_radius_px = max(near_blue_outer[1]) * 0.6

    near_center = np.array(near_blue_outer[0], dtype=np.float64)

    candidates = k8.find_house_ellipses(blue_mask, min_area=min_area, min_ratio=min_ratio)

    best = None
    best_y_error = None

    for c in candidates:

        if np.linalg.norm(c["center"] - near_center) < exclude_radius_px:
            continue

        r = unproject_to_ice_plane(c["center"][0], c["center"][1], camera, v_t, v_cl)

        if r is None:
            continue

        X, Y = r

        if abs(X) > max_x_abs_cm:
            continue

        y_error = abs(Y - k8.FAR_HOUSE_Y_CM)

        if y_error > max_y_error_cm:
            continue

        if best is None or y_error < best_y_error:
            best = (X, Y - k8.FAR_HOUSE_Y_CM)
            best_y_error = y_error

    if best is None:
        return None

    return {"x_offset_cm": best[0], "y_offset_cm": best[1], "y_error_cm": best_y_error}


# Kaukaisen pesan FINE-haun pisteytyspinta on paikoin lahes tasapelissa
# usean eri kiertokulman/skaalan valilla (kaksi lahes identtista
# moodikuvaa samasta paikallaan pysyvasta kamerasta antoivat FINE-
# vaiheessa TAYSIN eri parhaan kulman, -12 vs -36 astetta - katso
# diagnoosi taman ominaisuuden kehityshistoriassa) - yhden globaalin
# parhaan sijasta pidetaan hengissa FAR_HOUSE_FINE_TOPK eri (x,y)-
# ruudun parasta tulosta, viedaan JOKAINEN niista ULTRA-vaiheen lapi,
# ja PAATETAAN vasta lopuksi homografian RMS:lla (katso alempana) kumpi
# niista on oikeasti oikea - sama periaate jota koodi jo kayttaa
# teoreettisen/varipohjaisen kaukaisen pesan kandidaatin valinnassa.
FAR_HOUSE_FINE_TOPK = 5

# Sama ongelma toistuu jo COARSE-vaiheessa: kaksi lahes identtista
# moodikuvaa antoivat COARSE:sta TASMALLEEN saman (vaaran) -24 asteen
# kulman, ja koska FINE-vaiheen hakuikkuna (keskitetty COARSE:n yhteen
# tulokseen) on vain +-80cm leveä, se ei koskaan tavoita sita (x,y)-
# aluetta missa oikea (hoglinen mukaan lukien oikeasti vaakasuora)
# vastaus olisi. Pidetaan siis hengissa myos COARSE_TOPK eri (x,y)-
# ruudun tulosta, ja jokaisesta ajetaan oma FINE_TOPK_PER_COARSE:n
# FINE-haku - kokonaismaara pysyy suunnilleen ennallaan (COARSE_TOPK *
# FINE_TOPK_PER_COARSE ULTRA+geometrinen-korjaus -ajoa).
COARSE_TOPK = 6
FINE_TOPK_PER_COARSE = 6


def _search_far_house_topk(
    template, frame, camera, v_t, v_cl,
    center_x, center_y, center_range, center_step,
    angle_center, angle_range, angle_step,
    scale_center, scale_range, scale_step,
    description, top_k
):
    """Kuten k8.search_far_house, mutta palauttaa YHDEN parhaan tuloksen
    sijasta listan TOP-K parasta ERI (x,y)-ruutua (kullekin ruudulle
    silti vain sen oma paras kulma+skaala, sama valinta kuin k8.search_
    far_house tekisi sille ruudulle) - katso FAR_HOUSE_FINE_TOPK:n
    kommentti."""

    x_values = k8.make_range(center_x, center_range, center_step)
    y_values = k8.make_range(center_y, center_range, center_step)
    angle_values = k8.make_range(angle_center, angle_range, angle_step)
    scale_values = k8.make_range(scale_center, scale_range, scale_step)

    image_height, image_width = frame.shape[:2]
    hsv = template["_hsv"]
    hsv_model = template["hsv_model"]

    print()
    print("=" * 65)
    print(f"KAUKAISEN PESAN HAKU: {description}")
    print("=" * 65)
    print(f"Kandidaatteja yhteensa: {len(x_values) * len(y_values) * len(angle_values) * len(scale_values)}")

    per_cell_best = []

    for x_offset in x_values:
        for y_offset in y_values:

            projected = k8._project_template_for_offset(
                template, x_offset, y_offset, camera, v_t, v_cl
            )

            if projected["far_center"] is None:
                continue

            scores = k8.score_far_candidates_batch(
                projected, angle_values, scale_values, hsv, hsv_model,
                image_width, image_height
            )

            local_best_idx = int(np.argmax(scores))
            local_best_angle = float(angle_values[local_best_idx // len(scale_values)])
            local_best_scale = float(scale_values[local_best_idx % len(scale_values)])

            per_cell_best.append({
                "x_offset_cm": x_offset, "y_offset_cm": y_offset,
                "angle_deg": local_best_angle, "scale": local_best_scale,
                "score": float(scores[local_best_idx]),
            })

    per_cell_best.sort(key=lambda r: r["score"], reverse=True)
    top = per_cell_best[:top_k]

    for rank, r in enumerate(top):
        print(
            f"  #{rank}: X={r['x_offset_cm']:.3f} Y={r['y_offset_cm']:.3f} "
            f"kulma={r['angle_deg']:.3f} skaala={r['scale']:.4f} "
            f"pisteytys={r['score']:.6f}"
        )

    return top


def calibrate_camera_from_image_with_seed(filename):
    """kamera9_01.py:n calibrate_camera_from_image, laajennettuna kahdella
    tavalla (kayttajan pyynnosta, katso kommentit alempana koodissa):

    1) Kaukaisen pesan COARSE-haun alkuarvaus (center_x/center_y) haetaan
       datasta (find_far_house_coarse_seed) sokean (0,0)-oletuksen sijaan.
    2) Kaukaisen pesan haku (COARSE/FINE) pitaa hengissa TOP-K parasta eri
       (x,y)-kandidaattia yhden globaalin parhaan sijasta (katso COARSE_
       TOPK/FAR_HOUSE_FINE_TOPK:n kommentit), jokainen viedaan ULTRA-
       vaiheen + taman TASMALLEEN k8.refine_geometric_homography:n lapi,
       ja PAras lopullinen kandidaatti valitaan sen OIKEASTI lopullisesta
       pyoreydesta/koosta/hoglinen kulmasta - ei pelkasta pistesovituksen
       RMS:sta, joka osoittautui riittamattomaksi (katso kommentit).

    Itse k8/k9-funktioihin (haku, pisteytys, homografia, vaaristyman
    korjaus) EI ole koskettu - kaikki uusi logiikka on tata orkestrointia,
    joka kutsuu niita moneen kertaan ja vertailee tuloksia."""

    frame = cv2.imread(filename)

    if frame is None:
        raise RuntimeError(f"Kuvaa ei voitu avata: {filename}")

    image_height, image_width = frame.shape[:2]

    near_blue_mask = k8.create_blue_mask(frame)
    near_red_mask = k8.create_red_mask(frame)

    blue_outer, blue_inner = k8.find_house_pair(
        near_blue_mask, k8.NEAR_BLUE_MIN_AREA, k8.NEAR_BLUE_MIN_RATIO, k8.NEAR_BLUE_MIN_SIZE_RATIO
    )
    red_outer, red_inner = k8.find_house_pair(
        near_red_mask, k8.NEAR_RED_MIN_AREA, k8.NEAR_RED_MIN_RATIO, k8.NEAR_RED_MIN_SIZE_RATIO
    )

    if blue_outer is None or blue_inner is None or red_outer is None or red_inner is None:
        raise RuntimeError("Lahemman pesan renkaita ei loytynyt kokonaan.")

    segments = k8.detect_line_segments(frame, blue_outer)
    (t_pair, centerline_pair, t_line, centerline, house_center) = k8.select_t_and_centerline(
        segments, blue_outer
    )
    v_t, v_cl = k8.build_image_directions(t_pair, centerline_pair)

    observations = k8.build_calibration_observations(blue_outer, blue_inner, red_outer, red_inner)

    if len(observations) < 2:
        raise RuntimeError("Kalibrointiin tarvitaan vahintaan 2 ympyraa.")

    camera = k8.build_camera_model(observations, image_width, image_height)

    template = k8.build_near_house_template(frame, blue_outer, camera, v_t, v_cl)
    template["_hsv"] = template["hsv_model"]["hsv"]

    seed = find_far_house_coarse_seed(near_blue_mask, blue_outer, camera, v_t, v_cl)
    seed_x = seed["x_offset_cm"] if seed else 0.0
    seed_y = seed["y_offset_cm"] if seed else 0.0

    print(
        "  kaukaisen pesan alkuarvaus: "
        f"X={seed_x:.1f} Y-siirto={seed_y:.1f} "
        f"({'loytyi' if seed else 'ei loytynyt, kaytetaan oletusta 0,0'})"
    )

    coarse_candidates = _search_far_house_topk(
        template, frame, camera, v_t, v_cl,
        center_x=seed_x, center_y=seed_y,
        center_range=k8.SEARCH_CENTER_RANGE_CM, center_step=k8.COARSE_CENTER_STEP_CM,
        angle_center=0.0, angle_range=k8.SEARCH_CENTER_RANGE_DEG, angle_step=k8.COARSE_ANGLE_STEP_DEG,
        scale_center=1.2, scale_range=k8.SEARCH_CENTER_RANGE_SCALE, scale_step=k8.COARSE_SCALE_STEP,
        description="COARSE", top_k=COARSE_TOPK
    )

    fine_candidates = []
    for coarse_idx, coarse in enumerate(coarse_candidates):
        fine_candidates.extend(_search_far_house_topk(
            template, frame, camera, v_t, v_cl,
            center_x=coarse["x_offset_cm"], center_y=coarse["y_offset_cm"],
            center_range=k8.FINE_CENTER_RANGE_CM, center_step=k8.FINE_CENTER_STEP_CM,
            angle_center=coarse["angle_deg"], angle_range=k8.FINE_ANGLE_RANGE_DEG,
            angle_step=k8.FINE_ANGLE_STEP_DEG,
            scale_center=coarse["scale"], scale_range=k8.FINE_SCALE_RANGE, scale_step=k8.FINE_SCALE_STEP,
            description=f"FINE (coarse {coarse_idx})", top_k=FINE_TOPK_PER_COARSE
        ))

    # Jokainen FINE-vaiheen top-K-kandidaatti viedaan ULTRA-vaiheen lapi
    # (tasmalleen sama k8.search_far_house kuin ennen, ei muutoksia
    # itse hakuun), ja PARAS lopullinen kandidaatti valitaan vasta
    # homografian RMS:lla (near+far pisteet yhdessa) - katso FAR_HOUSE_
    # FINE_TOPK:n kommentti. Sama RMS-vertailu joka jo aiemmin valitsi
    # teoreettisen/varipohjaisen kaukaisen pesan valilla, laajennettu
    # nyt vertailemaan myos naita eri FINE-kandidaatteja keskenaan.
    camera_matrix = k8.build_camera_matrix(image_width, image_height)
    output_w = int(round((k8.OUTPUT_X_MAX_CM - k8.OUTPUT_X_MIN_CM) * k8.PIXELS_PER_CM))
    output_h = int(round((k8.OUTPUT_Y_MAX_CM - k8.OUTPUT_Y_MIN_CM) * k8.PIXELS_PER_CM))

    best_overall_key = None
    best_overall_result = None
    best_overall_rms = None

    for fine_idx, fine in enumerate(fine_candidates):

        optimized = k8.search_far_house(
            template, frame, camera, v_t, v_cl,
            center_x=fine["x_offset_cm"], center_y=fine["y_offset_cm"],
            center_range=k8.ULTRA_CENTER_RANGE_CM, center_step=k8.ULTRA_CENTER_STEP_CM,
            angle_center=fine["angle_deg"], angle_range=k8.ULTRA_ANGLE_RANGE_DEG,
            angle_step=k8.ULTRA_ANGLE_STEP_DEG,
            scale_center=fine["scale"], scale_range=k8.ULTRA_SCALE_RANGE, scale_step=k8.ULTRA_SCALE_STEP,
            description=f"ULTRA FINE (kandidaatti {fine_idx})"
        )

        x_offset, y_offset = optimized["x_offset_cm"], optimized["y_offset_cm"]
        angle_deg, scale = optimized["angle_deg"], optimized["scale"]

        far_center_raw = k8.project_point(x_offset, k8.FAR_HOUSE_Y_CM + y_offset, camera, v_t, v_cl)
        far_left_raw = k8.project_point(x_offset - k8.HOUSE_RADIUS_CM, k8.FAR_HOUSE_Y_CM + y_offset, camera, v_t, v_cl)
        far_right_raw = k8.project_point(x_offset + k8.HOUSE_RADIUS_CM, k8.FAR_HOUSE_Y_CM + y_offset, camera, v_t, v_cl)

        if far_center_raw is None or far_left_raw is None or far_right_raw is None:
            continue

        far_left_x, far_left_y = k8.transform_projected_points(
            far_left_raw[0], far_left_raw[1], far_center_raw[0], far_center_raw[1], angle_deg, 1
        )
        far_right_x, far_right_y = k8.transform_projected_points(
            far_right_raw[0], far_right_raw[1], far_center_raw[0], far_center_raw[1], angle_deg, 1
        )

        far_left = np.array([float(far_left_x), float(far_left_y)])
        far_right = np.array([float(far_right_x), float(far_right_y)])
        far_center_est = np.asarray(far_center_raw, dtype=np.float64)

        dir_forward = far_center_est - np.asarray(house_center, dtype=np.float64)
        dir_forward /= np.linalg.norm(dir_forward)
        dir_lateral = far_right - far_left
        dir_lateral /= np.linalg.norm(dir_lateral)

        near_img_pts, near_phys_pts, near_labels = k8.near_house_correspondences(
            t_line, centerline, house_center, blue_outer, blue_inner, red_outer, red_inner,
            dir_lateral, dir_forward
        )

        theoretical_img_pts, theoretical_phys_pts, _ = k8.theoretical_far_house_correspondences(
            x_offset, y_offset, angle_deg, scale, camera, v_t, v_cl, far_center_est
        )

        hue_result = k8.create_far_house_hue_masks(frame, x_offset, y_offset, camera, v_t, v_cl, angle_deg, scale)
        far_ellipses = k8.find_far_house_ellipses_from_hue(hue_result)
        hue_img_pts, hue_phys_pts, _, _ = k8.far_house_correspondences_from_ellipses(
            far_ellipses, dir_lateral, dir_forward, far_center_est
        )

        candidate_sources = []

        if len(theoretical_img_pts) >= 3:
            candidate_sources.append((theoretical_img_pts, theoretical_phys_pts))

        if len(hue_img_pts) >= 3:
            candidate_sources.append((hue_img_pts, hue_phys_pts))

        if not candidate_sources:
            continue

        far_img_pts, far_phys_pts = min(
            candidate_sources,
            key=lambda c: k8._homography_rms(
                np.array(near_img_pts + c[0], dtype=np.float64),
                k8.physical_to_output_px(near_phys_pts + c[1])
            )
        )

        rms = k8._homography_rms(
            np.array(near_img_pts + far_img_pts, dtype=np.float64),
            k8.physical_to_output_px(near_phys_pts + far_phys_pts)
        )

        # HUOM (kayttajan pyynnosta, korjaa aiemman version puutteen):
        # pelkka near+far-pisteiden RMS EI riittavasti erottele hyvaa/
        # huonoa kandidaattia toisistaan - homografia voi sovittaa nama
        # ~20 pistetta kohtalaisen hyvin (matala RMS) vaikka kaukainen
        # pesa nayttaisi silti vinolta/vaaran kokoiselta lopullisessa
        # topdown-kuvassa (todettu: RMS=48.7cm kandidaatti tuotti silti
        # vinon topdown-kuvan). Myos "halpa" pelkkaan cv2.findHomography-
        # arvioon (ilman objektiivin vaaristyman korjausta) perustuva
        # pyoreysmittaus todettiin EPALUOTETTAVAKSI (nayttaa hyvalta
        # ilman undistort:ia, mutta lopullinen - oikea - tulos huonompi)
        # - siksi jokaiselle kandidaatille ajetaan TASMALLEEN sama
        # tayspitka putki kuin voittajalle aiemmin ajettiin (best_k1 +
        # undistort + k8.refine_geometric_homography, joka tunnistaa
        # SEKA lahemman etta kaukaisen pesan renkaat JA molemmat
        # hoglinet SUORAAN lopullisesta warpatusta kuvasta), ja valinta
        # tehdaan sen OIKEASTI lopullisesta k8.measure_house_quality-
        # tuloksesta (pyoreys ensisijaisena, koko-virhe ja RMS
        # tasapelin ratkaisijoina).
        all_img_pts = near_img_pts + far_img_pts
        all_phys_pts = near_phys_pts + far_phys_pts

        cand_best_k1, cand_best_k1_rms, cand_baseline_rms = k8.estimate_radial_distortion_k1(
            all_img_pts, all_phys_pts, camera_matrix
        )
        cand_k1_improvement = (
            (cand_baseline_rms - cand_best_k1_rms) / cand_baseline_rms
            if cand_baseline_rms > 1e-9 and math.isfinite(cand_baseline_rms) else 0.0
        )

        if cand_k1_improvement > k8.K1_MIN_RELATIVE_IMPROVEMENT and abs(cand_best_k1) > k8.K1_MIN_MAGNITUDE:
            cand_dist_coeffs = np.array([cand_best_k1, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        else:
            cand_best_k1 = 0.0
            cand_dist_coeffs = np.zeros(5, dtype=np.float64)

        cand_frame_undistorted = cv2.undistort(frame, camera_matrix, cand_dist_coeffs)

        cand_near_pts_frame = k8.undistort_points_px(
            np.array(near_img_pts, dtype=np.float64), camera_matrix, cand_best_k1
        )
        cand_far_pts_frame_init = k8.undistort_points_px(
            np.array(far_img_pts, dtype=np.float64), camera_matrix, cand_best_k1
        )

        cand_src_pts = np.vstack([cand_near_pts_frame, cand_far_pts_frame_init]).astype(np.float32)
        cand_dst_pts = k8.physical_to_output_px(all_phys_pts).astype(np.float32)

        cand_H_init, _ = cv2.findHomography(cand_src_pts, cand_dst_pts, method=0)

        if cand_H_init is None:
            continue

        cand_near_phys_arr = np.array(near_phys_pts, dtype=np.float64)

        cand_refined = k8.refine_geometric_homography(
            cand_frame_undistorted, cand_H_init, cand_near_pts_frame, cand_near_phys_arr,
            output_w, output_h
        )
        cand_quality = cand_refined["quality"]

        # HUOM (kayttajan pyynnosta, toinen puute edellisessa versiossa):
        # pyoreys+koko EIVAT reagoi lainkaan koko topdown-kuvan YHTEISEEN
        # KIERTOON - ympyra pysyy ympyrana vaikka koko kuva olisi
        # vaarassa kulmassa. Todettiin etta usea kandidaatti (my "fixed"
        # tulos mukaan lukien) lapaisi taydella pyoreydella/koolla mutta
        # hogline oli silti ~15-18 astetta vinossa. Mitataan siis myos
        # lahemman hoglinen kulma SUORAAN kandidaatin omasta lopullisesta
        # topdown-kuvasta (k8.detect_hogline_points + k8.robust_line_
        # angle_from_points, sama funktio jota kaytetaan muualla) ja
        # painotetaan tata ENSISIJAISESTI - pyoreys/koko toimivat vasta
        # tasapelin ratkaisijoina saman kulmaluokan sisalla.
        #
        # HUOM (bugikorjaus): cand_refined["topdown_raw"] on renderoity
        # H_current:lla SITA ITERAATIOTA EDELTAVALLA homografialla, ei
        # lopullisella H_final:lla (topdown_raw tallennetaan iteraation
        # ALUSSA, H_final=H_new lasketaan VASTA sen jalkeen samalla
        # kierroksella) - hogline pitaa siis mitata TUOREELTA, H_final:
        # lla itse warpatulta kuvalta, muuten mittaus on yhden iteraation
        # jaljessa todellisesta lopputuloksesta.
        cand_topdown_final = cv2.warpPerspective(
            cand_frame_undistorted, cand_refined["H_final"], (output_w, output_h)
        )
        near_hog_pts_topdown = k8.detect_hogline_points(cand_topdown_final, k8.NEAR_HOGLINE_Y_CM)
        near_hog_angle, near_hog_conf = k8.robust_line_angle_from_points(near_hog_pts_topdown)
        far_hog_pts_topdown = k8.detect_hogline_points(cand_topdown_final, k8.FAR_HOGLINE_Y_CM)
        far_hog_angle, far_hog_conf = k8.robust_line_angle_from_points(far_hog_pts_topdown)

        # HUOM: kaukainen hogline-alue taman videon kuvakulmalla on
        # osoitettu (suora visuaalinen + pistediagnostiikka) osittain
        # mainostaulujen/hyllyjen peitossa - matalan luottamuksen
        # (raakojen pisteiden rivi vaihtelee epajohdonmukaisesti,
        # ei muodosta oikeaa suoraa) kaukainen lukema on siis KOHINAA,
        # ei todiste vaarasta kierrosta - sama ilmio jo dokumentoitu
        # kamera7_06.py:n refine_geometric_homography:ssa 00008.png/
        # Kivilla.png:lle (katso sen kommentti). Luotetaan kaukaiseen
        # vain jos luottamus on samaa suuruusluokkaa kuin tyypillinen
        # aito lahempi havainto (>=800) - muuten sita EI kayteta
        # rangaistuksena, koska se palkitsisi vain sattumanvaraisesti
        # "sopivan nakoisia" kohinapisteita.
        HOGLINE_MIN_TRUST_CONFIDENCE = 800.0

        hog_angle_penalty = abs(near_hog_angle) if near_hog_conf > 0.0 else 999.0
        if far_hog_conf >= HOGLINE_MIN_TRUST_CONFIDENCE:
            hog_angle_penalty = max(hog_angle_penalty, abs(far_hog_angle))

        print(
            f"  ULTRA-kandidaatti {fine_idx}: RMS={rms:.3f} cm, "
            f"{k8.house_quality_str(cand_quality)}, "
            f"hogline lahi={near_hog_angle:.2f}deg(luott={near_hog_conf:.0f}) "
            f"kauko={far_hog_angle:.2f}deg(luott={far_hog_conf:.0f})"
        )

        # Portti: kelpuutetaan vain kandidaatit joiden pyoreys tayttaa
        # kayttajan vaatimuksen (>0.95) - muuten hoglinen sattumanvarainen
        # hyva kulma huonommalla pyoreydella voisi voittaa. Sen sisalla
        # valinta ensisijaisesti pienimman hogline-poikkeaman mukaan.
        roundness_gate = cand_quality["worst_ratio"] > 0.95
        candidate_key = (
            roundness_gate,
            -hog_angle_penalty,
            cand_quality["worst_ratio"],
            -cand_quality["worst_size_err"],
            -rms,
        )

        if best_overall_key is None or candidate_key > best_overall_key:
            best_overall_key = candidate_key
            best_overall_rms = rms
            best_overall_result = {
                "frame": frame,
                "frame_undistorted": cand_frame_undistorted,
                "camera_matrix": camera_matrix,
                "best_k1": cand_best_k1,
                "H_final": cand_refined["H_final"],
                "quality": cand_quality,
                "image_width": image_width,
                "image_height": image_height,
                "output_w": output_w,
                "output_h": output_h,
                "near_pts_frame": cand_near_pts_frame,
                "near_phys": cand_near_phys_arr,
            }

    if best_overall_result is None:
        raise RuntimeError("Kaukaiselle pesalle ei saatu yhtaan alkukorrespondenssia.")

    print(
        f"  Valittu ULTRA-kandidaatti: RMS = {best_overall_rms:.3f} cm, "
        f"{k8.house_quality_str(best_overall_result['quality'])}"
    )

    return best_overall_result


# ============================================================
# ASETUKSET
# ============================================================

ENABLE_MODE_FILTER = False

# Kayttajan pyynnosta: debug-ominaisuus joka tallentaa elavan moni-
# kiven seurannan ajalta UUDEN videotiedoston (<video>_debug_seuranta.
# mp4), jossa jokaisen tunnistetun/seuratun kiven ENNUSTETTU ääriviiva
# (k94.predicted_stone_hull_fast - sama profiilimalli jota itse
# yhteissovituskin kayttaa) on piirretty framen paalle, jotta nakee
# SUORAAN missa/miksi HAKU tai SEURANTA tunnistaa jotain vaarin
# (esim. pelaajan kivena). Vihrea=tarkka, oranssi=ei-tarkka, keltainen
# =juuri HAKU:n loytama uusi kivi, punainen=SEURANTA hukkasi taman
# framen (piirretaan viimeisimpaan tunnettuun sijaintiin). HIDASTAA
# ajoa (VideoWriter-enkoodaus joka framella) - pida False normaali-
# ajoissa, aseta True vain debugatessa. Voidaan myos kytkea paalle
# dynaamisesti komentorivilta TATA VAKIOTA muokkaamatta: aja
# "python main.py --debug" (tai "-d") - katso if __name__=="__main__".
DEBUG_SAVE_TRACKING_VIDEO = False

WALL_OFFSET = 40
GRAY_RADIUS = 2
HISTORY_LENGTH = 10
MORPH_SIZE = 3
MIN_COMPONENT_AREA = 100
MAX_COMPONENT_AREA = 250000
SUBPIX_WINDOW = (5, 5)

DISPLAY_WIDTH = 1400
DISPLAY_HEIGHT = 800

MAX_POSITION_ERROR = 20.0
REPORT_EVERY = 1000
MAX_WORKERS = 8

ROI_HALF_WIDTH = 100
ROI_HALF_HEIGHT = 120

MODE_DURATION_SECONDS = 120.0
MODE_FRAME_INTERVAL = 4

FILTER_DISTANCE_THRESHOLD = 20.0
STABILIZATION_MEDIAN_FRAMES = 20

# Moodikuvan laskenta tehdään paloissa
TILE_SIZE = 128

# ============================================================
# PANEELIEN AUTOMAATTITUNNISTUKSEN TOLERANSSI
#
# Kayttajan pyynnosta: paneelit ovat "samantyylisesti samalla
# seudulla", mutta kamera sijoitetaan/zoomataan aina hieman eri
# tavalla kasin, joten niiden sijainti VAIHTELEE hieman referenssiin
# nahden. detect_panel (alla) hakee jo omasta ROI:staan (ROI_HALF_
# WIDTH/HEIGHT = 100x120px) lahimman tummasuorakulmion annetun
# pisteen ymparilta - PANEL_REFERENCE_SEARCH_SCALE suurentaa taman
# haun VAIN referenssipohjaiselle automaattitunnistukselle (ei
# vaikuta paneiliSEURANTAAN framejen valilla, joka kayttaa detect_
# panel:ia normaalilla ROI:lla edellisen sijainnin ymparilta - kuvan
# drifti framejen valilla on paljon pienempi kuin kameran koko
# uudelleensijoittelun tuoma ero referenssiin).
# ============================================================

PANEL_REFERENCE_SEARCH_SCALE = 10

# ============================================================
# AUTOMAATTINEN KALIBROINTI - MOODIKUVA-POHJAINEN
#
# Kayttajan pyynnosta: sen sijaan etta kalibroidaan yhdesta
# raakaframesta (jossa voi olla liikkuvia kivia/pelaajia hairitsemassa
# lahemman pesan renkaiden tunnistusta), kerataan STABILOITUJA (mutta
# EI perspektiivikorjattuja - katso bootstrap_calibration:in kommentti
# MIKSI) frameja CALIB_MODE_SAMPLE_INTERVAL_SECONDS valein videon
# ensimmaisen CALIB_MODE_DURATION_SECONDS ajalta, ja lasketaan niista
# TILE_SIZE-kokoisittain moodikuva (olemassa oleva C++-mekanismi,
# sama add_mode_frame/start_mode_background/wait_for_mode kuin
# ENABLE_MODE_FILTER-ominaisuudessa) - liikkuvat kohteet haviavat,
# jaljelle jaa puhdas jaakuva jota vasten calibrate_camera_from_image
# (kamera9_01.py) toimii huomattavasti luotettavammin.
# ============================================================

CALIB_MODE_DURATION_SECONDS = 60.0
CALIB_MODE_SAMPLE_INTERVAL_SECONDS = 5.0

# ============================================================
# KOKO RADAN SKANNAUS LIIKKUVAN KIVEN LOYTAMISEKSI (3D-PROFIILIA
# VARTEN)
#
# Kayttajan pyynnosta: videon alussa kivi voi liikkua kumpaankin
# suuntaan eika sen sijaintia tiedeta etukateen - skannataan siis
# KOKO fyysinen rata (ei kamera9_02.py:n kiinteaa paata-rajattua
# HAKU-vyohyketta) STONE_SCAN_INTERVAL_SECONDS valein kunnes jotain
# loytyy - "ripea silloin kun vain odottaa kiveä" (kayttajan sanoin):
# harva aikavali pitaa taman odotusvaiheen halpana.
# ============================================================

STONE_SCAN_INTERVAL_SECONDS = 2.0

# HUOM: k9.find_stone_candidates suodattaa jo OLETUKSENA koko radan
# fyysisiin rajoihin (k8.OUTPUT_X/Y_MIN/MAX_CM + sheet_margin_cm) -
# ei tarvita omaa erillista rajausta, "koko rata" tulee ilmaiseksi.

# Kahden peräkkäisen skannauksen (STONE_SCAN_INTERVAL_SECONDS valein)
# valinen siirtyma, jonka ylittava sama (lahin) kandidaatti tulkitaan
# AIDOSTI LIIKKUVAKSI kiveksi (ei paikallaan olevaksi) - hidaskin
# liu'un loppuvaihe siirtyy enemman kuin tama 2 sekunnissa.
STONE_MOTION_THRESHOLD_CM = 15.0

# ============================================================
# 3D-PROFIILIN RIITTAVYYDEN VALIDOINTI
#
# Kayttajan pyynnosta: yksittainen loydetty/seurattu kivi ei aina
# riita hyvaan profiiliin (esim. kivi nakyy vain lyhyesti alussa) -
# sovitusta (k9.fit_stone_profile) YRITETAAN jokaisen uuden loydetyn
# kiven jalkeen KAIKKIEN TAHAN ASTI kerattyjen havaintojen paalla, ja
# hyvaksytaan vasta kun se on RIITTAVAN hyva nailla kynnysarvoilla -
# muuten jatketaan koko radan skannausta ja kerataan lisaa (mahdollisesti
# eri kivesta/hetkesta), YHDISTAEN havainnot samaan sovitukseen.
# ============================================================

# HUOM (kayttajan huomio: osa heitoista osittain harjan peittamia):
# aidon kiven yksinainen jaannosvirhe voi olla 6-11px kun osa reunasta
# on harjan takana - kun useampi taman tasoinen havainto YHDISTETAAN
# (kayttajan pyynto: vahintaan PROFILE_MIN_ACCEPTED_STONES kivea),
# pooled-sovituksen RMS luonnollisesti kasvaa (yhteinen malli selittaa
# useampaa, kohinaisempaa havaintoa yhtaikaa) - 5.0px oli viritetty
# ajalta jolloin PARI puhdasta havaintoa riitti. 10.0px sallii tuon
# realistisen kohinatason turvautumatta R_max:in fysikaaliseen
# jarkevyystarkistukseen (katso alla) ainoana suojana vaarilta
# kandidaateilta.
PROFILE_MAX_RMS_PX = 10.0
PROFILE_MIN_SAMPLES = 15
PROFILE_SAMPLES_PER_STONE = 25

# Turvaverkko RMS-kynnyksen LISAKSI (ei sen sijaan): kamera9_01.py:n
# fit_stone_profile:in oma dokumentaatio sanoo R_max:in olevan
# fyysisesti n. 14.0-14.6cm (oikean kiven halkaisija ~28cm). Aiemmin
# testatessa loydettiin tapaus jossa vaara kandidaatti (mainosteksti)
# lapaisi RMS-kynnyksen mutta antoi R_max~66cm - n. 4.5x liian ison.
# Reilut rajat (paljon RMS-kynnysta tiukemmat vaatimukset olisivat
# turhia, mutta karsivat selvasti fysiikan vastaiset tulokset).
PROFILE_R_MAX_MIN_CM = 10.0
PROFILE_R_MAX_MAX_CM = 20.0

# Kayttajan huomio (tarkea arkkitehtuurikorjaus): koko radan HIDASTA
# segmentointipohjaista skannausta EI kannata jatkaa keraamaan monta
# kivea - sen ainoa tehtava on saada AIKAISEKSI jokin RIITTAVA
# profiili mahdollisimman NOPEASTI. Sen jalkeen ELAVA SEURANTA (katso
# alempana "ELAVA MONI-KIVEN SEURANTA") jo hakee UUDET kivet paljon
# tehokkaammin JA luotettavammin: mallipohjaisella ristikkohaulla
# VAIN kaukaisen paan kiinnealta vyohykkeelta (k92.SEARCH_Y_MIN/MAX_CM
# = kaukainen hogline...pesan ulkoreuna, k92.SEARCH_X_HALF_WIDTH_CM=
# +-50cm keskiviivasta) - ei enaa tarvitse luottaa hitaaseen, harjasta
# helposti hairiintyvaan vapaamuotoiseen liikkeentunnistukseen koko
# radalla. Siksi PROFILE_MIN_ACCEPTED_STONES=1: skannaus pysahtyy heti
# ensimmaisen riittavan profiilin loydyttya, ja loput heitot jaavat
# ELAVAN SEURANNAN (nopeamman, tarkemman) vastuulle.
PROFILE_MIN_ACCEPTED_STONES = 1

# Testatessa oikealla videolla loytyi KAKSI ongelmaa jotka nama
# kynnysarvot/mekanismit korjaavat:
#
#   1) find_stone_candidates hyvaksyy MYOS ei-kivia (kayttajan
#      dokumentoima aiempi havainto: "pelaajan vaatteet" jne. - katso
#      k9.find_stone_candidates:in kommentti) - LIIKKUVA ei-kivi
#      (esim. pelaaja/harja) voi lapaista liikkeentunnistuksen. Siksi
#      JOKAINEN seurattu kandidaatti tarkistetaan YKSINAAN (SOLO_TRACK_
#      MAX_RMS_PX) ENNEN kuin sen havainnot lisataan pysyvaan kokoelmaan
#      - jos yksinaankin jaannosvirhe on aivan liian suuri (ei nayta
#      kivelta), havainnot HYLATAAN eika saastuteta koko kokoelmaa
#      pysyvasti.
#   2) SAMA fyysinen kohde (oli se sitten kivi tai hylatty ei-kivi)
#      loydettiin uudelleen JOKA skannauksella niin kauan kuin se pysyi
#      liikkeessa (esim. kavelevaa/juokseva pelaaja useiden 2s-ikkunoiden
#      ajan) - HUOM: sijaintipohjainen jaahdytys EI RIITA, koska nopeasti
#      liikkuva kohde (kayttajan testivideolla havaittu, luultavasti
#      pelaaja, n. 200cm/2s) karkaa sijaintikynnyksen ulkopuolelle ennen
#      jaahdytysajan paattymista. Sen sijaan STONE_SCAN_COOLDOWN_FRAMES
#      estaa uuden kalliin taydellisen seurannan koko taman monta framea
#      kattavalla IKKUNALLA viimeisimman yrityksen (kivi tai hylatty)
#      SIEMENFRAMESTA riippumatta kohteen nopeudesta - karkea mutta
#      luotettava nopeusrajoitin kalliille yrityksille.
# HUOM (kayttajan huomio: osa heitoista vaikeampia tunnistaa koko
# matkaa harjan takia): testatessa nakyi selva kahtiajako yksinaisen
# kandidaatin RMS-jakaumassa - selvasti ei-kivet (esim. pelaajat)
# antoivat RMS n. 12-40px, kun taas todennakoisesti AIDOT mutta
# osittain harjan peittamat kivet jaivat n. 6-11px valille (vrt.
# background-suodatuksen jalkeen hyvaksytyt puhtaat havainnot, jotka
# olivat 2-6px). SOLO_TRACK_MAX_RMS_PX:n tehtava on vain karsia
# SELVASTI ei-kivet pois pysyvasta kokoelmasta - lopullinen laatu-
# portti on POOLED-sovituksen oma, tiukempi PROFILE_MAX_RMS_PX
# (katso try_fit_profile) - joten kynnysta voi nostaa tanne asti
# ilman etta koko sovituksen lopullinen tarkkuus karsii.
SOLO_TRACK_MAX_RMS_PX = 12.0

# HUOM (kayttajan huomio: n. 9 heittoa videolla, mutta vain 2
# hyvaksyttiin): 750 framea (30s) osoittautui liian pitkaksi -
# lokista nakyi etta UUSI yritys kaynnistyi lahes AINA TASAN
# jaahdytyksen paatyttya (siis jotain "liikkuvaa" loytyy joka
# ikinen skannaus), joten jaahdytys itse rajoitti karkeasti
# videon_pituus/30s attempts-maaran - jos kaksi oikeaa heittoa
# tapahtuu alle 30s valein, jalkimmainen jai kokonaan loytamatta.
# track_stone_in_video_windowed on jo AIKAIKKUNAAN rajattu (ei
# koko videota) joten yrityksen hinta on paljon pienempi kuin
# alkuperaisessa "kallis" -perustelussa - lyhyempi jaahdytys on
# nyt varaa. Sama oikea kivi loytyneena kahdesti (esim. jos
# jaahdytys paattyy kesken sen oman liu'un) ei ole haitallista
# (background-suodatuksen jalkeen hyvaksytyt havainnot ovat
# aidosti kivia, joten kaksinkertainen havainto vain vahvistaa
# sovitusta, ei saastuta sita - toisin kuin aiempi ongelma
# vaarilla ei-kivi-kandidaateilla).
STONE_SCAN_COOLDOWN_FRAMES = 150   # 6s 25fps:lla

# ============================================================
# ELAVA MONI-KIVEN SEURANTA + CSV
#
# Kayttajan pyynnosta: max 4 kiveä samanaikaisesti, yksinkertainen
# lahin-ehdokas-per-kivi -logiikka riittaa (kivet lahekkain vasta
# pysahtymisen jalkeen, jolloin ID:lla ei ole enaa merkitysta).
# Uusien kivien HAKU kaytta kamera9_02.py:n kiinteaa paata-rajattua
# vyohyketta (SEARCH_X/Y_*, k92-moduulista) - TOISIN kuin 3D-profiilin
# koko-radan-skannaus (Task 3): kivet HEITETAAN aina samaan suuntaan/
# paahan, joten kiinteä HAKU-vyohyke on jarkeva/tehokas tassa.
# ============================================================

MAX_CONCURRENT_STONES = 4

# HAKU-valin PAIKALLINEN ylikirjoitus (kayttajan pyynnosta) - EI
# muuteta kamera9_02.py:n omaa SEARCH_EVERY_N_FRAMES:ia (se tiedosto
# on koskematon referenssi, katso taman tiedoston alkupaan kommentti).
# kamera9_02.py:n oma arvo (10 framea=0.4s 25fps:lla) vaihdettu
# harvempaan, sekuntipohjaiseen valiin - vahemman HAKU-kutsuja
# (jokainen n. 150-220ms taydella kuormalla) maksaa vahemman CPU-
# aikaa, hintana etta uuden kiven havaitsemisessa voi kestaa taman
# verran pidempaan (radalle tulevan kiven ensimmaiset havainnot
# puuttuvat CSV:sta talta ajalta).
HAKU_SEARCH_INTERVAL_SECONDS = 1.0

# Jos uusi HAKU-loytö on tata lahempana jotain jo AKTIIVISTA kiveä,
# tulkitaan samaksi kiveksi (ei uutta ID:ta) - estaa saman kiven
# kaksoiskirjautumisen. HUOM: HAKU:n karkea koko-framen ristikkohaku
# (SEARCH_COARSE/FINE_STEP_CM) ja SEURANTA:n oma hienompi haku
# (TRACK_COARSE/FINE_STEP_CM) eivat aina osu tarkalleen samaan
# pisteeseen samalla framella varsinkin nopeasti liikkuvalla kivella -
# testivideolla havaittu poikkeama n. 66 cm yhden ja saman kiven
# kahden eri hakumekanismin valilla, joten kynnys pidetaan reilusti
# sen ylapuolella.
NEW_STONE_DEDUP_CM = 100.0

# Kayttajan pyynnosta: kun kivi on ollut lahes paikallaan (liikkunut
# alle STOP_TRACKING_DISPLACEMENT_CM) STOP_TRACKING_SECONDS ajan,
# lopetetaan sen aktiivinen SEURANTA - viimeinen sijainti jaa CSV:hen.
# Kaksi hyotya: (1) sailyttaa aktiivisen paikan MAX_CONCURRENT_STONES:sta
# uusille kiville nopeammin, (2) estaa jo pysahtyneen kiven haun
# ajautumisen taustan muuhun sisaltoon pitkalla aikavalilla (havaittu
# testatessa: yksi pitkaan paikallaan seurattu kivi ajautui lopulta
# fyysisesti mahdottomaan sijaintiin, Y=-102cm).
STOP_TRACKING_SECONDS = 1.0
STOP_TRACKING_DISPLACEMENT_CM = 20.0

# VARIREFERENSSI (kayttajan pyynnosta): SEURANTA voi harvoin "hypata"
# pitkaan seuratusta oikeasta kivesta lahella olevaan vieraaseen
# kohteeseen (esim. pelaaja pesan lahella) - naytonmuoto-tarkistus
# (tarkka) menee tassa tapauksessa LAPI, koska vieras kohde sattuu
# olemaan riittavan kiven-muotoinen. Rakennetaan siis KERRAN, heti kun
# 3D-profiili on valmis, kiven OMA pintavarireferenssi hyvaksytyn
# profiilin havainnoista (build_stone_color_reference) - PER PISTE
# local_pts_body:sta (EI keskiarvoistettuna yhdeksi lukemaksi koko
# pinnalta!), jotta graniitin n. keskikorkeudella nakyva vaaleampi
# nauha ("paiva"/leveimmillaan-kohta, _TEMPLATE_EQUATOR_IDX) sailyy
# omana piirteenaan eika sekoitu muun pinnan keskiarvoon. Kayttajan
# huomio: nauha EI nay kaukana olevissa kivissa (resoluutio) - siksi
# seka referenssin rakennus etta elavan osuman vertailu HYLKAAVAT
# (skip, ei estoa) havainnot/osumat joissa kivi on kuvassa liian pieni
# nauhan (tai minkaan hienon piirteen) luotettavaan erottamiseen -
# tarkistus vaikuttaa siis kaytannossa vain lahelta (esim. pesan
# lahella) otettuihin osumiin, mika sopii yhteen sen kanssa etta juuri
# se on havaittu ongelma-alue.
#
# SUUNNITTELUPAATOS (useiden testikierrosten jalkeen - kayttajan
# prioriteetti "heittaja/lakaisija ei saa aiheuttaa kiven katoamista"
# ohjaa tata): varitarkistus EI vaikuta MITENKAAN normaaliin SEURANTAan
# (haku-ankkuri, s["misses"], "kadotettu") - kokeiltiin ensin useita
# hystereesi-/streak-pohjaisia versioita jotka KASVATTIVAT/vahensivat
# "huonoa" laskuria ja kayttivat sita joko hyvaksymiseen TAI kiven
# katoamiseen, mutta havaittiin etta MIKA TAHANSA vuotava/palautuva
# laskuri lopulta "antaa anteeksi" pysyvasti vaaran kohteen (esim.
# pelaajan jalka), koska aidosti kohiseva data tuottaa ENNEMMIN TAI
# MYOHEMMIN riittavan pitkan "hyvan" jakson kumotakseen kertyneen
# laskurin - TAMA PATEE YHTA LAILLA todelliseen kiveen jota peittaa
# pitkaan (havaittu: 30+ ruutua/1.2s+) esim. mittaava pelaaja, jonka
# aikana muototunnistus (refined["found"]) pysyy silti onnistuneena.
# Koska aito peitto ja aito identiteettihyppy nayttavat siis VARILTAAN
# samankaltaisilta pitkalla aikavalilla, MIKAAN kynnys+streak-yhdistelma
# ei erottele niita luotettavasti - vain KESTO eroaa (peitto loppuu,
# hyppy ei koskaan "korjaannu").
#
# Sen sijaan varitarkistus vaikuttaa VAIN "pysahtynyt"-ilmoitukseen
# (alempana SEURANTA-silmukassa): liukuvan ikkunan (color_diff_history,
# sama aikaikkuna kuin position_history) KESKIARVO on oltava aidosti
# graniittimainen ennen kuin pysahtyminen hyvaksytaan LOPULLISENA - jos
# ei, ilmoitus vain LYKKAANTYY (kivi pysyy normaalisti aktiivisena/
# seurattuna, ei koskaan "kadotettu" varin takia). Tama kohdistuu
# suoraan alkuperaiseen ongelmaan (vaaran kohteen virheellinen
# kirjaaminen "pysahtyneeksi kiveksi") ilman etta se voi koskaan
# aiheuttaa oikean, vain hetkellisesti/pitkaan peitetyn kiven katoamista.
#
# Kynnysarvot kalibroitu oikealla videolla (stone0:n omat, itse
# vahvistetut lahelta-pesaa -havainnot dbg_full_v3.csv:sta) - katso
# kommentti color_match_median_diff:in MEDIAANI-aggregoinnin kohdalla
# (miksi ei keskiarvo per piste). Saman kiven oma mitattu keskimaarainen
# poikkeama itsestaan: min=0.085, max=0.204, ka=0.141 (30 nayteruutua) -
# COLOR_STOP_MAX_AVG_DIFF pidetty reilusti taman ylapuolella.
COLOR_REF_MIN_RING_SPACING_PX = 2.5
COLOR_REF_MIN_OBSERVATIONS = 3
COLOR_MATCH_MIN_VALID_POINTS = 20
COLOR_STOP_MAX_AVG_DIFF = 0.22

CSV_HEADER = [
    "frame", "timestamp_s", "stone_id", "x_m", "y_m", "tarkka",
    "n_runkopistetta", "n_reunapistetta", "rms_px", "rengas_r_cm",
]

# ============================================================
# APUFUNKTIOT
# ============================================================

def get_gray_value(gray, x, y, radius=GRAY_RADIUS):
    h, w = gray.shape

    x = int(round(x))
    y = int(round(y))

    x1 = max(0, x - radius)
    x2 = min(w, x + radius + 1)
    y1 = max(0, y - radius)
    y2 = min(h, y + radius + 1)

    roi = gray[y1:y2, x1:x2]

    if roi.size == 0:
        return 0

    return float(np.median(roi))


def order_points(pts):
    pts = np.asarray(pts, dtype=np.float32)

    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()

    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]

    return rect

# ============================================================
# PANEELIN TUNNISTUS
# ============================================================

def detect_panel(gray, x, y, roi_half_width=None, roi_half_height=None):

    if roi_half_width is None:
        roi_half_width = ROI_HALF_WIDTH

    if roi_half_height is None:
        roi_half_height = ROI_HALF_HEIGHT

    h, w = gray.shape

    x = int(round(x))
    y = int(round(y))

    if x < 0 or x >= w or y < 0 or y >= h:
        return None

    center_gray = get_gray_value(
        gray,
        x,
        y
    )

    threshold = center_gray + WALL_OFFSET
    threshold = min(
        255,
        threshold
    )

    # --------------------------------------------------------
    # ROI
    # --------------------------------------------------------

    x1 = max(
        0,
        x - roi_half_width
    )

    x2 = min(
        w,
        x + roi_half_width + 1
    )

    y1 = max(
        0,
        y - roi_half_height
    )

    y2 = min(
        h,
        y + roi_half_height + 1
    )

    roi_gray = gray[
        y1:y2,
        x1:x2
    ]

    # --------------------------------------------------------
    # THRESHOLD
    # --------------------------------------------------------

    roi_binary = (
        roi_gray < threshold
    ).astype(
        np.uint8
    ) * 255

    # --------------------------------------------------------
    # MORFOLOGIA
    # --------------------------------------------------------

    kernel = np.ones(
        (
            MORPH_SIZE,
            MORPH_SIZE
        ),
        dtype=np.uint8
    )

    roi_binary = cv2.morphologyEx(
        roi_binary,
        cv2.MORPH_OPEN,
        kernel
    )

    roi_binary = cv2.morphologyEx(
        roi_binary,
        cv2.MORPH_CLOSE,
        kernel
    )

    # --------------------------------------------------------
    # CONNECTED COMPONENTS
    #
    # TÄRKEÄÄ:
    # tehdään vain ROI:lle, ei koko 1920x1080-kuvalle.
    # --------------------------------------------------------

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            roi_binary,
            connectivity=8
        )
    )

    if num_labels <= 1:
        return None

    # Keskipiste ROI:n koordinaateissa
    local_x = x - x1
    local_y = y - y1

    # --------------------------------------------------------
    # VALITSE KOMPONENTTI
    # --------------------------------------------------------

    if (
        0 <= local_y < labels.shape[0]
        and
        0 <= local_x < labels.shape[1]
    ):
        center_label = labels[
            local_y,
            local_x
        ]
    else:
        center_label = 0

    if center_label != 0:

        label = int(
            center_label
        )

    else:

        best_label = None
        best_distance = float("inf")

        for i in range(
            1,
            num_labels
        ):

            area = stats[
                i,
                cv2.CC_STAT_AREA
            ]

            if area < MIN_COMPONENT_AREA:
                continue

            if area > MAX_COMPONENT_AREA:
                continue

            cx, cy = centroids[i]

            # Muutetaan ROI-koordinaateiksi
            distance = (
                (cx - local_x) ** 2 +
                (cy - local_y) ** 2
            )

            if distance < best_distance:

                best_distance = distance
                best_label = i

        if best_label is None:
            return None

        label = best_label

    # --------------------------------------------------------
    # TARKISTA KOMPONENTIN KOKO
    # --------------------------------------------------------

    area = stats[
        label,
        cv2.CC_STAT_AREA
    ]

    if area < MIN_COMPONENT_AREA:
        return None

    if area > MAX_COMPONENT_AREA:
        return None

    # --------------------------------------------------------
    # KOMPONENTIN MASKI
    # --------------------------------------------------------

    component = np.uint8(
        labels == label
    ) * 255

    contours, _ = cv2.findContours(
        component,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    contour = max(
        contours,
        key=cv2.contourArea
    )

    if (
        cv2.contourArea(contour)
        < MIN_COMPONENT_AREA
    ):
        return None

    # --------------------------------------------------------
    # NELIÖ / MIN AREA RECT
    # --------------------------------------------------------

    epsilon = (
        0.02 *
        cv2.arcLength(
            contour,
            True
        )
    )

    approx = cv2.approxPolyDP(
        contour,
        epsilon,
        True
    )

    if len(approx) == 4:

        corners = (
            approx.reshape(
                4,
                2
            ).astype(
                np.float32
            )
        )

    else:

        rect = cv2.minAreaRect(
            contour
        )

        corners = (
            cv2.boxPoints(
                rect
            ).astype(
                np.float32
            )
        )

    # --------------------------------------------------------
    # SIIRRETÄÄN ROI-KOORDINAATEISTA
    # KOKO KUVAN KOORDINAATEIKSI
    # --------------------------------------------------------

    corners[:, 0] += x1
    corners[:, 1] += y1

    corners = order_points(
        corners
    )

    # --------------------------------------------------------
    # SUBPIKSELI
    # --------------------------------------------------------

    gray_float = np.ascontiguousarray(
        gray
    )

    try:

        refined = cv2.cornerSubPix(
            gray_float,
            corners.reshape(
                -1,
                1,
                2
            ),
            SUBPIX_WINDOW,
            (-1, -1),
            (
                cv2.TERM_CRITERIA_EPS +
                cv2.TERM_CRITERIA_MAX_ITER,
                30,
                0.01
            )
        )

        if refined is not None:

            corners = refined.reshape(
                4,
                2
            )

    except cv2.error:
        pass

    # --------------------------------------------------------
    # KESKIPISTE
    # --------------------------------------------------------

    center = np.mean(
        corners,
        axis=0
    )

    return {
        "corners": corners,
        "center": center
    }


# ============================================================
# PANEELIN SEURANTA
# ============================================================

def track_panel(gray, previous_center):
    return detect_panel(
        gray,
        previous_center[0],
        previous_center[1]
    )


# ============================================================
# PANEELITIEDOSTON TALLENNUS
# ============================================================

def save_panel_data(
    panel_data,
    video_file
):

    base = os.path.splitext(
        video_file
    )[0]

    filename = (
        base +
        "_panel_corners.txt"
    )

    with open(
        filename,
        "w",
        encoding="utf-8"
    ) as f:

        for i, panel in enumerate(
            panel_data
        ):

            f.write(
                f"PANEL {i + 1}\n"
            )

            for p in panel["corners"]:

                f.write(
                    f"{p[0]:.6f} "
                    f"{p[1]:.6f}\n"
                )

            f.write(
                f"CENTER "
                f"{panel['center'][0]:.6f} "
                f"{panel['center'][1]:.6f}\n\n"
            )

    return filename


# ============================================================
# PANEELITIEDOSTON LATAUS
#
# _parse_panel_corners_file on jaettu jasennyslogiikka - load_panel_
# data lataa TAMAN videon OMAN (aiemmin taman ohjelman tallentaman)
# tiedoston, load_panel_reference lataa TOISEN (referenssi-) videon
# tiedoston jota kaytetaan automaattitunnistuksen pohjana.
# ============================================================

def _parse_panel_corners_file(filename):

    if not os.path.exists(filename):
        return None

    with open(
        filename,
        "r",
        encoding="utf-8"
    ) as f:

        lines = [
            line.strip()
            for line in f
            if line.strip()
        ]

    panels = []
    i = 0

    while i < len(lines):

        if not lines[i].startswith(
            "PANEL"
        ):
            i += 1
            continue

        i += 1

        corners = []

        for _ in range(4):

            if i >= len(lines):
                raise ValueError(
                    "Kulmapisteitä puuttuu."
                )

            parts = lines[i].split()

            if len(parts) != 2:
                raise ValueError(
                    "Virheellinen kulmapiste."
                )

            corners.append(
                [
                    float(parts[0]),
                    float(parts[1])
                ]
            )

            i += 1

        if i >= len(lines):
            raise ValueError(
                "CENTER puuttuu."
            )

        parts = lines[i].split()

        if (
            len(parts) != 3
            or parts[0] != "CENTER"
        ):
            raise ValueError(
                "Virheellinen CENTER-rivi."
            )

        center = [
            float(parts[1]),
            float(parts[2])
        ]

        i += 1

        panels.append(
            {
                "corners": np.asarray(
                    corners,
                    dtype=np.float32
                ),
                "center": np.asarray(
                    center,
                    dtype=np.float32
                )
            }
        )

    return panels


def load_panel_data(video_file):

    base = os.path.splitext(
        video_file
    )[0]

    filename = (
        base +
        "_panel_corners.txt"
    )

    try:

        panels = _parse_panel_corners_file(filename)

        if panels is None or len(panels) < 2:
            return None

        print(
            f"Ladattu paneelitiedosto: "
            f"{filename}"
        )

        print(
            f"Paneeleita: {len(panels)}"
        )

        return panels

    except Exception as e:

        print(
            f"Paneelitiedoston lataus epäonnistui: "
            f"{e}"
        )

        return None


def load_panel_reference(filename):
    """Lataa REFERENSSITIEDOSTON (kayttajan aiemmasta, samantyyppisesta
    kamera-asettelusta antama *_panel_corners.txt) automaattitunnistuksen
    pohjaksi - sama tiedostomuoto kuin load_panel_data/save_panel_data,
    mutta EKSPLISIITTINEN polku (ei johdeta nykyisen videon nimesta)."""

    panels = _parse_panel_corners_file(filename)

    if panels is None:
        raise RuntimeError(
            f"Referenssitiedostoa ei loytynyt: {filename}"
        )

    if len(panels) < 2:
        raise RuntimeError(
            f"Referenssitiedostossa liian vahan paneeleita: {filename}"
        )

    print(f"Ladattu referenssipaneelitiedosto: {filename}")
    print(f"Referenssipaneeleita: {len(panels)}")

    return panels


# ============================================================
# PANEELIEN AUTOMAATTITUNNISTUS REFERENSSIN LAHELTA
#
# Kayttajan pyynnosta (korvaa kasin-klikkauksen select_panels): jokaista
# referenssipaneelia kohti etsitaan TODELLINEN paneeli sen keskipisteen
# LAHELTA - detect_panel:in oma ROI (ROI_HALF_WIDTH/HEIGHT) tarjoaa jo
# perustoleranssin; suurennamme sita viela PANEL_REFERENCE_SEARCH_
# SCALE:lla, koska kamera sijoitetaan/zoomataan kasin joka kerta hieman
# eri tavalla (kayttajan sanoin: "samantyylisesti samalla seudulla...
# hieman hajontaa"). Referenssin OMAT kulmat/koko eivat ole muuten
# merkityksellisia - vain keskipiste kaytetaan hakupisteena, koska
# detect_panel loytaa TODELLISEN muodon/koon uudelleen jokaiselle
# videolle erikseen (kameran zoomi voi muuttaa paneelin PIKSELIKOKOA).
# ============================================================

def detect_panels_from_reference(gray, reference_panels,
                                  search_scale=PANEL_REFERENCE_SEARCH_SCALE):

    search_w = int(round(ROI_HALF_WIDTH * search_scale))
    search_h = int(round(ROI_HALF_HEIGHT * search_scale))

    panel_data = []
    missing = 0

    for idx, ref in enumerate(reference_panels):

        ref_center = ref["center"]

        result = detect_panel(
            gray,
            float(ref_center[0]),
            float(ref_center[1]),
            roi_half_width=search_w,
            roi_half_height=search_h
        )

        if result is None:

            missing += 1

            print(
                f"VAROITUS: paneelia {idx + 1} ei loytynyt "
                f"referenssikohdasta ({ref_center[0]:.1f}, "
                f"{ref_center[1]:.1f}) - ohitetaan."
            )

            continue

        panel_data.append(result)

    print(
        f"Paneelien automaattitunnistus: {len(panel_data)}/"
        f"{len(reference_panels)} loytyi ({missing} puuttuu)."
    )

    if len(panel_data) < 2:
        raise RuntimeError(
            "Liian vahan paneeleita loytyi automaattisesti "
            "(vahintaan 2 tarvitaan stabilointiin)."
        )

    return panel_data


# ============================================================
# KOKO RADAN KIVIEHDOKKAIDEN HAKU + LIIKKEEN TUNNISTUS
#
# _scan_stone_candidates: sama segmentointipohjainen tunnistus kuin
# k9.track_stone_in_video/track_stone_in_video_fast kayttavat SISAISESTI
# (k9._candidates_in_frame) - EI mallipohjaista ristikkohakua, koska
# tassa vaiheessa 3D-profiilia (jota malli tarvitsisi) EI VIELA OLE -
# se on juuri se mita etsitaan.
#
# find_moving_candidate: vertaa KAHDEN PERAKKAISEN skannauksen (n.
# STONE_SCAN_INTERVAL_SECONDS valein) kandidaattilistoja - sama fyysinen
# kivi (lahin osuma) jonka sijainti on muuttunut enemman kuin STONE_
# MOTION_THRESHOLD_CM tulkitaan AIDOSTI LIIKKUVAKSI (ei jo-paikallaan-
# olevaksi) - tama siirtymä-havainto ANTAA SIEMENEN (frame_idx, X, Y)
# k94.track_stone_in_video_fast:lle, joka sitten seuraa koko liu'un.
#
# suppress_static_background: k9.create_granite_mask:in kommentti
# (kamera9_01.py) sanoo 5x5-avauksen poistavan OHUET staattiset
# rakenteet (sponsoritekstin kirjaimet, maalatut viivat) - mutta
# testissa havaittiin etta TAMA EI RIITTANYT: staattinen mainosteksti
# tuli silti virheellisesti tunnistetuksi "kiveksi" (n. 4.5x liian
# suuri R_max 3D-profiilin sovituksessa, katso git-historia/keskustelu).
# Koska meilla ON jo kalibroinnin moodikuva (puhdas, kivi-/pelaaja-
# vapaa staattinen tausta - juuri se mita moodisuodatus on suunniteltu
# tuottamaan), kayttajan ehdotuksesta: verrataan nykyista framea
# TAHAN referenssiin ja PEITETAAN (korvataan valkoisella) alueet jotka
# vastaavat sita - jaljelle jaa vain AIDOSTI poikkeava/vaihtuva sisalto
# (kivet, pelaajat), joten mikaan staattinen painettu sisalto ei voi
# enaa tulla virhetunnistetuksi kiveksi tassa vaiheessa.
# ============================================================

def suppress_static_background(frame_bgr, reference_bgr, diff_threshold=30):

    if reference_bgr is None or frame_bgr.shape != reference_bgr.shape:
        return frame_bgr

    diff = cv2.absdiff(frame_bgr, reference_bgr)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    background_mask = diff_gray < diff_threshold

    out = frame_bgr.copy()
    out[background_mask] = (255, 255, 255)

    return out

def _scan_stone_candidates(frame_bgr, calib, pose, background_reference=None):

    frame_bgr = suppress_static_background(
        frame_bgr, background_reference
    )

    return k9._candidates_in_frame(
        frame_bgr, calib, pose, calib["H_final"],
        k9.STONE_TRACK_MIN_AREA, k9.STONE_TRACK_MIN_FILL_RATIO,
        k9.STONE_TRACK_MIN_ASPECT_RATIO
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


# ============================================================
# AIKAIKKUNAAN RAJATTU KIVEN SEURANTA (EI koko videota)
#
# k94.track_stone_in_video_fast (kamera9_04.py, EI kosketa) skannaa
# TARKOITUKSELLA koko videon jokaisen framen - jarkevaa SEN omassa
# kayttotarkoituksessaan (kertaluontoinen kalibrointi lyhyella
# videolla). Tassa TIEDOSSA jo ON karkea siemen (koko radan skannaus
# loysi sen), joten koko videon uudelleenskannaus JOKAISELLA
# yrityksella (havaittu testatessa: n. 700s per yritys 7500 framen
# videolla) olisi kohtuuttoman hidas, varsinkin jos useampi kandidaatti
# (esim. pelaaja) hylataan ennen aidon kiven loytymista. Tama funktio
# rajaa saman algoritmin (sama jatkuvuuslogiikka, samat kynnysarvot)
# STONE_TRACK_WINDOW_SECONDS-ikkunaan siemenen ymparilta - riittava
# kattamaan koko liu'un (kivi ei ole jaalla montaa kymmenta sekuntia
# kerrallaan) mutta paljon halvempi kuin koko video.
# ============================================================

STONE_TRACK_WINDOW_SECONDS = 30.0

# KAYTTAJAN MITTAAMA LOYDOS (oikea video, todellinen skannaus alusta
# asti): track_stone_in_video_windowed oli 73.7% koko kalibrointi+
# skannausvaiheen ajasta, JA 5/6 kutsusta (83%) kohdistui HYLATTYYN
# ehdokkaaseen (pelaaja/kohina) - jokainen hylkays maksoi silti taysin
# saman KOKO 60s ikkunan (1501 framea) hinnan kuin hyvaksytty kivikin,
# koska solo_ok-kelvollisuustarkistus tehtiin vasta KOKO ikkunan
# skannauksen JALKEEN. Kaksi toisiaan taydentavaa optimointia (EI
# Python-porttaus/kamera9_04.py-muutos - vain tama tiedosto):
#
# 1) HALPA ESITARKISTUS ENNEN taytta ikkunaa: skannataan ensin vain
#    +-STONE_TRACK_PRECHECK_WINDOW_SECONDS (paljon lyhyempi), ja jos
#    tama nayttaa jo selvasti EI-kivelta, hylataan HETI ilman koko
#    60s ikkunan skannausta. Kayttaa LOYHEMPAA RMS-kynnysta kuin
#    lopullinen solo_ok-tarkistus (STONE_TRACK_PRECHECK_MAX_RMS_PX),
#    koska lyhyt ikkuna antaa vahemman kulmavaihtelua (katso taman
#    tiedoston toisen kommentin selitys profiilisovituksen kulma-
#    vaihtelutarpeesta) - tama tekisi RMS:sta systemaattisesti
#    huonomman myos AIDOLLE kivelle, joten liian tiukka kynnys
#    hylkaisi vaarin. R_max-jarkevyystarkistus (try_fit_profile:in
#    SISALLA, PROFILE_R_MAX_MIN/MAX_CM) pysyy silti samana - se on
#    fyysinen koko-tarkistus joka ei riipu ikkunan pituudesta, ja
#    havaitussa datassa (R_max=21-247cm hylatyilla vs. 12.28cm
#    hyvaksytylla) juuri TAMA erotti selvimmin.
#    Jos esitarkistuksen data on liian niukka paatokseen (alle
#    STONE_TRACK_PRECHECK_MIN_SAMPLES havaintoa), EI hylata datan
#    puutteen takia - jatketaan varovaisuudesta taydelliseen ikkunaan.
#
# 2) NAYTTEISTYS: taydessa (ja esitarkistus-) ikkunassa kasitellaan
#    VAIN joka STONE_TRACK_SAMPLE_STRIDE:s frame taysin (vaanto+tausta
#    vaimennus+kandidaattitunnistus - kalliit vaiheet), ei jokaista -
#    kandidaattitunnistukseen kaytetyt havainnot (PROFILE_SAMPLES_
#    PER_STONE=25) ovat joka tapauksessa paljon harvempia kuin taysi
#    1501 framen tiheys tuottaisi, joten tiheampi kuin naytteistetty
#    seuranta ei lisaa profiilin laatua. max_jump_cm skaalataan
#    STRIDE:lla (perustason kynnys on mitoitettu PERAKKAISILLE
#    framille) - muuten nopeasti liikkuva kivi voisi karata seurannasta
#    naytteiden valilla.
STONE_TRACK_PRECHECK_WINDOW_SECONDS = 3.0
STONE_TRACK_PRECHECK_MIN_SAMPLES = 5
STONE_TRACK_PRECHECK_MAX_RMS_PX = 24.0
# HUOM (kayttajan mittaama loydos + paatos): stride=5 JA stride=3
# heikensivat molemmat havaittavasti profiilisovituksen laatua (RMS
# n. 5.9px -> 10.9px stride=5:lla samalle oikealle kivelle taydessa
# 30s ikkunassa - riittavan lahella PROFILE_MAX_RMS_PX=10.0px
# -kynnysta etta yksi muuten kelvollinen kivi ei enaa riittanyt
# sellaisenaan). Kayttaja paatti sen sijaan pitaa TAYDEN
# framekohtaisen tarkkuuden (stride=1, degeneroituu tayteen
# tiheyteen build_indices:issa) ja nojata nopeuteen sen sijaan
# halpaan esitarkistukseen + C++-porttaukseen (katso alempana
# scan_stone_candidates/stone_tracker.cpp).
STONE_TRACK_SAMPLE_STRIDE = 1


def track_stone_in_video_windowed(video_path, calib, pose, seed_frame_idx,
                                   seed_pos_cm, window_seconds=STONE_TRACK_WINDOW_SECONDS,
                                   background_reference_undistorted=None):

    max_jump_cm = k9.STONE_TRACK_MAX_JUMP_CM * STONE_TRACK_SAMPLE_STRIDE
    max_misses = k9.STONE_TRACK_MAX_MISSES
    min_area = k9.STONE_TRACK_MIN_AREA
    max_area = 200000
    min_fill_ratio = k9.STONE_TRACK_MIN_FILL_RATIO
    min_aspect_ratio = k9.STONE_TRACK_MIN_ASPECT_RATIO

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
    x_min = k8.OUTPUT_X_MIN_CM - k9.STONE_SHEET_MARGIN_CM
    x_max = k8.OUTPUT_X_MAX_CM + k9.STONE_SHEET_MARGIN_CM
    y_min = k8.OUTPUT_Y_MIN_CM - k9.STONE_SHEET_MARGIN_CM
    y_max = k8.OUTPUT_Y_MAX_CM + k9.STONE_SHEET_MARGIN_CM
    pixels_per_cm = k8.PIXELS_PER_CM
    output_x_min_cm = k8.OUTPUT_X_MIN_CM
    output_y_max_cm = k8.OUTPUT_Y_MAX_CM

    cap = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    map1, map2 = k94._build_undistort_maps(
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

    profile = k9.fit_stone_profile(pose, stones)

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

    z_equator_frac = float(k9._TEMPLATE_Z_FRAC[k9._TEMPLATE_EQUATOR_IDX])
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
    map1, map2 = k94._build_undistort_maps(
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
        u, v = k9._project_3d(pose["K"], pose["R"], pose["t"], pts)

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

        b = k94._bilinear_sample_vec(frame_u[:, :, 0].astype(np.float64), u, v)
        g = k94._bilinear_sample_vec(frame_u[:, :, 1].astype(np.float64), u, v)
        r = k94._bilinear_sample_vec(frame_u[:, :, 2].astype(np.float64), u, v)

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
    u, v = k9._project_3d(pose["K"], pose["R"], pose["t"], pts)

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

    b = k94._bilinear_sample_vec(frame_u_f64[:, :, 0], u, v)
    g = k94._bilinear_sample_vec(frame_u_f64[:, :, 1], u, v)
    r = k94._bilinear_sample_vec(frame_u_f64[:, :, 2], u, v)

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


# ============================================================
# YKSI PAAPUTKI - kaikki vaiheet (kalibrointi -> kiviprofiilin haku ->
# elava moni-kiven seuranta) jakavat SAMAN videon peräkkäisen luvun ja
# SAMAN joka-frame paneiliseurannan/stabiloinnin (kayttajan pyynnosta
# tama on hidas, tasainen driftinseuranta - katso stabilointimatriisin
# mediaanisuodatuksen kommentti alla, EI kosketa sita). Vaiheet
# etenevat sisainen tila (calib_result, profile_result, jne.)
# perusteella - kukin vaihe kaynnistyy vasta edellisen valmistuttua.
# ============================================================

def run_pipeline(
    video_file,
    panel_data,
    calib_diag_output,
    csv_output,
    precomputed_calib_result=None,
    precomputed_profile_result=None,
    debug_video_output=None
):

    engine = mode_engine.ModeEngine(
        video_file,
        TILE_SIZE
    )

    width = engine.width()
    height = engine.height()
    fps = engine.fps()
    total_frames = engine.total_frames()

    # katso HAKU_SEARCH_INTERVAL_SECONDS:in kommentti - korvaa
    # kamera9_02.py:n SEARCH_EVERY_N_FRAMES:in elavan seurannan
    # HAKU-ajastuksessa (koskematon kamera9_02.py itse ennallaan).
    haku_interval_frames = max(1, int(round(HAKU_SEARCH_INTERVAL_SECONDS * fps)))

    print()
    print(
        f"Resoluutio: "
        f"{width} x {height}"
    )

    print(
        f"Ruutuja: {total_frames}"
    )

    print(
        f"FPS: {fps:.3f}"
    )

    # ========================================================
    # AUTOMAATTISEN KALIBROINNIN MOODINAYTTEENOTON AJASTUS
    #
    # HUOM: engine.set_perspective_matrix EI kutsuta - perspective_
    # enabled_ pysyy C++:ssa false:na, joten add_mode_frame tuottaa
    # VAIN stabiloidun (EI ylhaaltapain-warpatun) framen. calibrate_
    # camera_from_image TARVITSEE juuri taman - se ratkaisee itse
    # vaantokertoimen+homografian RAAasta/vaantyneesta kuvasta, joten
    # syotteen ei pida olla jo perspektiivikorjattu.
    # ========================================================

    calib_mode_max_frames = int(
        round(fps * CALIB_MODE_DURATION_SECONDS)
    )

    calib_sample_every = max(
        1, int(round(fps * CALIB_MODE_SAMPLE_INTERVAL_SECONDS))
    )

    print(
        f"Kalibroinnin moodikuvaan kerataan naytteet videon "
        f"ensimmaisen {CALIB_MODE_DURATION_SECONDS:.0f} sekunnin "
        f"ajalta, yksi naytepiste {CALIB_MODE_SAMPLE_INTERVAL_SECONDS:.0f} "
        f"sekunnin valein (enintaan "
        f"{calib_mode_max_frames // calib_sample_every + 1} naytetta)."
    )

    reference_centers = [
        np.asarray(
            panel["center"],
            dtype=np.float32
        ).copy()
        for panel in panel_data
    ]

    histories = [
        [center.copy()]
        for center in reference_centers
    ]

    frame_index = 0
    next_calib_sample_frame = 0

    calib_result = precomputed_calib_result

    stone_scan_interval_frames = max(
        1, int(round(fps * STONE_SCAN_INTERVAL_SECONDS))
    )
    stop_tracking_frames = max(
        1, int(round(fps * STOP_TRACKING_SECONDS))
    )
    next_stone_scan_frame = 0
    prev_scan_candidates = None
    accumulated_stones = []
    n_accepted_stones = 0
    best_sufficient_profile = None
    profile_result = precomputed_profile_result
    next_allowed_scan_track_frame = 0

    if precomputed_calib_result is not None:
        print(
            "Kalibrointi annettu valmiiksi laskettuna "
            "(ohitetaan moodikuva-bootstrap)."
        )
    if precomputed_profile_result is not None:
        print(
            "3D-kiviprofiili annettu valmiiksi laskettuna "
            "(ohitetaan koko radan skannaus)."
        )

    active_stones = []
    next_stone_id = 0
    live_state = None
    csv_writer = None
    csv_file = None
    debug_video_writer = None

    previous_stabilization_matrix = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ],
        dtype=np.float64
    )
    stabilization_history = deque(
        maxlen=STABILIZATION_MEDIAN_FRAMES
    )

    start_time = time.time()

    total_gray_time = 0.0
    total_tracking_time = 0.0
    total_transform_time = 0.0
    total_sample_time = 0.0

    # --------------------------------------------------------
    # NOPEUSSEURANTA (kayttajan pyynnosta): HAKU (stone_tracker.
    # search_new_stone) ja SEURANTA (stone_tracker.track_stones_
    # batch) ovat nyt molemmat C++:aa - tallennetaan niiden
    # kutsumaarat+kokonaisajat tanne jotta REPORT_EVERY-valein
    # tulostettava yhteenveto (ja lopuksi koko ajon yhteenveto)
    # kertoo TARKALLEEN missa aika kuluu MYOS kayttajan omalla
    # koneella - katso raportin tulostus alempana.
    # --------------------------------------------------------

    total_haku_time = 0.0
    n_haku_calls = 0

    total_seuranta_time = 0.0
    n_seuranta_calls = 0
    n_seuranta_stone_updates = 0

    # Lisamittarit "mihin loput ajasta menee" -selvitykseen (kayttajan
    # pyynnosta): engine.read() (videon luku+dekoodaus), stabilointi-
    # matriisin RANSAC-laskenta, ja warpAffine+remap (KOKO framelle,
    # jokaisella elavan seurannan framella - nama eivat olleet aiemmin
    # ollenkaan ajastettuja).
    total_read_time = 0.0
    total_stabilize_compute_time = 0.0
    total_warp_remap_time = 0.0

    executor = ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    )

    # HAKU (stone_tracker.search_new_stone) ja SEURANTA (stone_tracker.
    # track_stones_batch) ovat riippumattomia (molemmat lukevat vain
    # jo valmiin frame_u:n, eivat toistensa tulosta) - kayttajan
    # pyynnosta HAKU kaynnistetaan omalle taustasaikeelle JA SEURANTA
    # ajetaan SAMAAN AIKAAN paasaikeessa, katso alempana "HAKU:
    # kaynnistetaan..." ja "HAKU:n tuloksen keraaminen...". Uuden
    # kiven rekisterointi pysyy SILTI samalla framella kuin ennen -
    # HAKU:n tulos noudetaan (.result()) SEURANTAN VALMISTUTTUA,
    # ennen seuraavaan frameen siirtymista, EI viivastu. Vain 1
    # HAKU-kutsu voi olla kerrallaan kesken (SEARCH_EVERY_N_FRAMES
    # varmistaa etta edellinen on aina jo koottu ennen seuraavaa).
    haku_executor = ThreadPoolExecutor(max_workers=1)

    def _run_haku_timed(*args):
        t0 = time.time()
        result = stone_tracker.search_new_stone(*args)
        return result, time.time() - t0

    try:

        while True:

            t_read0 = time.perf_counter()
            frame = engine.read()
            total_read_time += time.perf_counter() - t_read0

            if frame is None or frame.size == 0:
                break
            t0 = time.perf_counter()
            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )
            t1 = time.perf_counter()
            total_gray_time += t1 - t0

            # ------------------------------------------------
            # PANEELIEN SEURANTA
            # ------------------------------------------------

            t2 = time.perf_counter()
            futures = []

            for panel_index in range(
                len(panel_data)
            ):

                if histories[panel_index]:

                    previous_center = (
                        histories[
                            panel_index
                        ][-1]
                    )

                else:

                    previous_center = (
                        reference_centers[
                            panel_index
                        ]
                    )

                futures.append(
                    executor.submit(
                        track_panel,
                        gray,
                        previous_center
                    )
                )

            current_centers = []

            for panel_index, future in enumerate(
                futures
            ):

                result = future.result()

                if result is None:

                    current_centers.append(
                        None
                    )

                    continue

                center = np.asarray(
                    result["center"],
                    dtype=np.float32
                )

                previous_center = (
                    histories[
                        panel_index
                    ][-1]
                    if histories[
                        panel_index
                    ]
                    else reference_centers[
                        panel_index
                    ]
                )

                error = np.linalg.norm(
                    center -
                    previous_center
                )

                if error <= MAX_POSITION_ERROR:

                    histories[
                        panel_index
                    ].append(
                        center.copy()
                    )

                    if len(
                        histories[
                            panel_index
                        ]
                    ) > HISTORY_LENGTH:

                        histories[
                            panel_index
                        ].pop(0)

                    current_centers.append(
                        center
                    )

                else:

                    current_centers.append(
                        None
                    )

            t3 = time.perf_counter()
            total_tracking_time += t3 - t2
            # ------------------------------------------------
            # STABILOINTIMATRIISI
            # ------------------------------------------------

            t_stab0 = time.perf_counter()

            valid_reference = []
            valid_current = []

            for i in range(
                len(reference_centers)
            ):

                if current_centers[i] is not None:

                    valid_reference.append(
                        reference_centers[i]
                    )

                    valid_current.append(
                        current_centers[i]
                    )

            stabilization_matrix = (
                previous_stabilization_matrix.copy()
            )

            if len(valid_reference) >= 2:

                reference = np.asarray(
                    valid_reference,
                    dtype=np.float32
                )

                current = np.asarray(
                    valid_current,
                    dtype=np.float32
                )

                M, inliers = (
                    cv2.estimateAffinePartial2D(
                        reference,
                        current,
                        method=cv2.RANSAC,
                        ransacReprojThreshold=3.0,
                        maxIters=2000,
                        confidence=0.99
                    )
                )

                if M is not None:

                    stabilization_matrix = (
                        cv2.invertAffineTransform(
                            M
                        )
                    )

            # ------------------------------------------------
            # STABILOINNIN MEDIAANISUODATUS
            #
            # Käytetään peräkkäisten ruutujen mediaania.
            # Tämä poistaa pientä värinää, mutta seuraa
            # edelleen hidasta driftia.
            # ------------------------------------------------

            stabilization_history.append(
                stabilization_matrix.copy()
            )

            stabilization_matrix = np.median(
                np.stack(
                    stabilization_history,
                    axis=0
                ),
                axis=0
            )

            stabilization_matrix = np.asarray(
                stabilization_matrix,
                dtype=np.float64
            )


            previous_stabilization_matrix = (
                stabilization_matrix.copy()
            )

            total_stabilize_compute_time += time.perf_counter() - t_stab0

            # ------------------------------------------------
            # LÄHETÄ STABILOINNIN MATRIX C++:LLE
            # ------------------------------------------------
            t4 = time.perf_counter()
            engine.set_transform(
                stabilization_matrix
            )
            t5 = time.perf_counter()
            total_transform_time += t5 - t4

            # ------------------------------------------------
            # AUTOMAATTISEN KALIBROINNIN MOODINAYTTEET
            #
            # (katso taman funktion alkupaan kommentti MIKSI
            # set_perspective_matrix:ia ei kutsuta - add_mode_frame
            # tuottaa siis stabiloidun mutta MUUTEN raa'an framen).
            # ------------------------------------------------

            if calib_result is None:

                if (
                    frame_index < calib_mode_max_frames
                    and frame_index >= next_calib_sample_frame
                ):

                    t6 = time.perf_counter()

                    engine.add_mode_frame()

                    t7 = time.perf_counter()

                    total_sample_time += (
                        t7 - t6
                    )

                    next_calib_sample_frame += calib_sample_every

                # --------------------------------------------
                # Kun kalibroinnin moodinaytteet on kerätty,
                # lasketaan moodikuva JA kalibroidaan siita -
                # tama on KERTALUONTEINEN tauko (moodikuvan
                # lasku + calibrate_camera_from_image/build_
                # pose_from_calibration kestavat yhteensa
                # kymmenia sekunteja) - videon lukua EI jatketa
                # taman valmistumisen aikana, mika on hyvaksyttavaa
                # koska taman vaiheen ei tarvitse olla reaaliaikainen
                # (vain elava kiven seuranta myohemmin vaatii sen).
                # --------------------------------------------

                if (
                    next_calib_sample_frame >= calib_mode_max_frames
                ):

                    print()
                    print(
                        "Kalibroinnin moodinaytteet ovat kasassa "
                        f"({engine.mode_frame_count()} kpl). "
                        "Lasketaan moodikuva..."
                    )

                    calib_mode_output = (
                        os.path.splitext(video_file)[0] +
                        "_kalibrointi_moodikuva.png"
                    )

                    engine.start_mode_background(
                        calib_mode_output
                    )

                    engine.wait_for_mode()

                    print(
                        f"Moodikuva valmis: {calib_mode_output}"
                    )

                    print(
                        "Kalibroidaan moodikuvasta "
                        "(calibrate_camera_from_image_with_seed + "
                        "build_pose_from_calibration)..."
                    )

                    calib = calibrate_camera_from_image_with_seed(
                        calib_mode_output
                    )

                    pose = k9.build_pose_from_calibration(calib)

                    print(
                        f"  fokaalivali f = {pose['K'][0, 0]:.1f} px, "
                        f"kameran sijainti (cm): "
                        f"{np.round(pose['camera_position_cm'], 1)}"
                    )

                    print(
                        f"  lahemman pesan jaannosvirhe (px): "
                        f"keskiarvo="
                        f"{pose['near_reproj_err_px']['mean']:.2f}, "
                        f"max={pose['near_reproj_err_px']['max']:.2f}"
                    )

                    topdown = cv2.warpPerspective(
                        calib["frame_undistorted"],
                        calib["H_final"],
                        (calib["output_w"], calib["output_h"])
                    )

                    cv2.imwrite(calib_diag_output, topdown)

                    print(
                        f"Kalibroinnin topdown-tarkistuskuva: "
                        f"{calib_diag_output}"
                    )

                    calib_result = {"calib": calib, "pose": pose}

            # ------------------------------------------------
            # KOKO RADAN SKANNAUS LIIKKUVAN KIVEN LOYTAMISEKSI
            # (3D-PROFIILIA VARTEN) - vasta kun kalibrointi on
            # valmis (tarvitaan calib/pose fyysisten sijaintien
            # laskentaan) JA profiili ei viela ole riittava.
            # ------------------------------------------------

            elif profile_result is None:

                if frame_index >= next_stone_scan_frame:

                    # Kandidaattitunnistuksen syote stabiloidaan SAMALLA
                    # matriisilla kuin moodikuva-referenssi aikanaan
                    # laskettiin (engine.add_mode_frame) - muuten
                    # taustavertailu (suppress_static_background) ei
                    # osu kohdalleen, ja pikselikoordinaatit eivat
                    # vastaa poseen kalibrointireferenssia.
                    stabilized_scan_frame = cv2.warpAffine(
                        frame, stabilization_matrix, (width, height)
                    )

                    curr_candidates = _scan_stone_candidates(
                        stabilized_scan_frame, calib_result["calib"],
                        calib_result["pose"],
                        background_reference=calib_result["calib"]["frame"]
                    )

                    seed_pos = find_moving_candidate(
                        prev_scan_candidates, curr_candidates
                    )

                    # ------------------------------------------
                    # JAAHDYTYS: katso taman tiedoston alkupaan
                    # kommentti MIKSI taman on oltava frame-ikkuna-
                    # pohjainen (ei sijaintipohjainen) - nopeasti
                    # liikkuva ei-kivi karkaisi sijaintikynnyksesta.
                    # ------------------------------------------

                    if frame_index < next_allowed_scan_track_frame:
                        seed_pos = None

                    if seed_pos is not None:

                        print()
                        print(
                            f"[frame {frame_index}] Liikkuva kandidaatti "
                            f"loytyi kohdasta ({seed_pos[0]:.1f}, "
                            f"{seed_pos[1]:.1f}) cm - seurataan koko "
                            f"liu'un ajan..."
                        )

                        track = track_stone_in_video_windowed(
                            video_file, calib_result["calib"],
                            calib_result["pose"],
                            seed_frame_idx=frame_index,
                            seed_pos_cm=seed_pos,
                            background_reference_undistorted=(
                                calib_result["calib"]["frame_undistorted"]
                            )
                        )

                        print(
                            f"  seuranta valmis: {len(track)} havaintoa."
                        )

                        next_allowed_scan_track_frame = (
                            frame_index + STONE_SCAN_COOLDOWN_FRAMES
                        )

                        if len(track) >= 2:

                            idxs = sorted(set(
                                np.linspace(
                                    0, len(track) - 1,
                                    PROFILE_SAMPLES_PER_STONE
                                ).astype(int).tolist()
                            ))

                            candidate_observations = [
                                {
                                    "ellipse": track[i]["ellipse"],
                                    "contour": track[i]["contour"],
                                    # frame_idx/pos_cm: EI kayteta
                                    # profiilisovitukseen (try_fit_
                                    # profile), vain myohempaan
                                    # varireferenssin rakennukseen
                                    # (build_stone_color_reference) -
                                    # katso kommentti COLOR_REF_*-
                                    # vakioiden kohdalla.
                                    "frame_idx": track[i]["frame_idx"],
                                    "pos_cm": track[i]["pos_cm"],
                                }
                                for i in idxs
                            ]

                            # --------------------------------
                            # YKSINAINEN KELVOLLISUUSTARKISTUS
                            # ENNEN pysyvaan kokoelmaan lisaamista
                            # (katso taman tiedoston alkupaan
                            # kommentti - estaa ei-kivien, esim.
                            # pelaajien, saastuttamasta koko
                            # kokoelmaa pysyvasti).
                            # --------------------------------

                            _, solo_ok = try_fit_profile(
                                calib_result["pose"],
                                candidate_observations,
                                max_rms_px=SOLO_TRACK_MAX_RMS_PX,
                                label="  yksittaisen kandidaatin tarkistus"
                            )

                            if not solo_ok:

                                print(
                                    "  hylatty - ei nayta kivelta "
                                    "(esim. pelaaja/muu liikkuva "
                                    "kohde), ei lisata kokoelmaan."
                                )

                            else:

                                accumulated_stones.extend(
                                    candidate_observations
                                )
                                n_accepted_stones += 1

                                profile, riittava = try_fit_profile(
                                    calib_result["pose"],
                                    accumulated_stones
                                )

                                if riittava:

                                    best_sufficient_profile = profile

                                    if (
                                        n_accepted_stones >=
                                        PROFILE_MIN_ACCEPTED_STONES
                                    ):

                                        print(
                                            "3D-kiviprofiili riittava "
                                            f"({n_accepted_stones} "
                                            "hyvaksyttya kiveä) - "
                                            "lopetetaan koko radan "
                                            "skannaus."
                                        )

                                        profile_result = profile

                                    else:

                                        print(
                                            f"  profiili jo riittava, "
                                            "mutta jatketaan viela "
                                            f"lisaa kivia varten "
                                            f"({n_accepted_stones}/"
                                            f"{PROFILE_MIN_ACCEPTED_STONES})."
                                        )

                    prev_scan_candidates = curr_candidates
                    next_stone_scan_frame = frame_index + stone_scan_interval_frames

            # ------------------------------------------------
            # ELAVA MONI-KIVEN SEURANTA + CSV - vasta kun SEKA
            # kalibrointi ETTA 3D-kiviprofiili ovat valmiit.
            # ------------------------------------------------

            else:

                if live_state is None:

                    profile = profile_result
                    R_max = profile["R_max_cm"]
                    H_total = profile["H_total_cm"]
                    shape_deltas = profile["shape_deltas"]

                    camera_matrix = calib_result["calib"]["camera_matrix"]
                    dist_coeffs = np.array(
                        [calib_result["calib"]["best_k1"], 0.0, 0.0, 0.0, 0.0],
                        dtype=np.float64
                    )

                    map1, map2 = k94._build_undistort_maps(
                        camera_matrix, dist_coeffs, (width, height)
                    )

                    local_pts_body = k94.build_local_stone_rings(
                        R_max, H_total, shape_deltas, n_theta=28, n_per_segment=5
                    )
                    local_pts_search = k94.build_local_stone_rings(
                        R_max, H_total, shape_deltas,
                        n_theta=k92.SEARCH_HULL_N_THETA,
                        n_per_segment=k92.SEARCH_HULL_N_PER_SEGMENT
                    )

                    # stone_tracker.cpp:n refine_position_joint-portin
                    # ring_r_frac_guess - katso kamera9_04.py:n refine_
                    # position_joint_fast:in oma laskenta, riippuu vain
                    # profiilin muodosta (shape_deltas), ei framesta,
                    # joten lasketaan kerran tanne kuten muukin live_
                    # state.
                    ring_r_frac_guess = float(
                        (k9._TEMPLATE_R_FRAC + shape_deltas)[-1]
                    )

                    print(
                        "Rakennetaan kiven pintavarireferenssia "
                        f"({len(accumulated_stones)} havainnosta)..."
                    )

                    stone_color_reference = build_stone_color_reference(
                        video_file, calib_result["calib"],
                        calib_result["pose"], local_pts_body, 28,
                        accumulated_stones
                    )

                    csv_file = open(csv_output, "w", newline="")
                    csv_writer = csv.writer(csv_file)
                    csv_writer.writerow(CSV_HEADER)

                    if debug_video_output is not None:

                        debug_video_writer = cv2.VideoWriter(
                            debug_video_output,
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            fps, (width, height)
                        )

                    print()
                    print(
                        f"Elava moni-kiven seuranta alkaa (frame "
                        f"{frame_index}) - CSV: {csv_output}"
                    )

                    live_state = {
                        "map1": map1, "map2": map2,
                        "local_pts_body": local_pts_body,
                        "local_pts_search": local_pts_search,
                        "R_max": R_max, "H_total": H_total,
                        "ring_r_frac_guess": ring_r_frac_guess,
                        "color_reference": stone_color_reference,
                    }

                # HUOM: kokeiltiin taalla yhdistaa warpAffine+remap
                # yhdeksi remap-kutsuksi (cv2.transform map1/map2:lle +
                # yksi remap) - eristetty mikrobenchmark nayttii ensin
                # parannusta, mutta TOISTETTUNA (ja koko putken sisalla)
                # tulos vaihteli suunnasta toiseen ajojen valilla - ero
                # oli taman hiekkalaatikon oman kohinan sisalla, EI
                # luotettavasti mitattavissa (map1/map2 ovat float32
                # 2-kanavaisia, 8 tavua/pikseli - enemman dataa per
                # pikseli kuin itse BGR-kuva, joten "halpa" koordinaatti-
                # muunnos ei ollutkaan niin halpa). Palautettu alku-
                # peraiseen kahteen erilliseen vaiheeseen - EI otettu
                # kayttoon todentamatonta optimointia.
                t_warp0 = time.perf_counter()
                stabilized = cv2.warpAffine(
                    frame, stabilization_matrix, (width, height)
                )
                frame_u = cv2.remap(
                    stabilized, live_state["map1"], live_state["map2"],
                    interpolation=cv2.INTER_LINEAR
                )
                total_warp_remap_time += time.perf_counter() - t_warp0

                timestamp = frame_index / fps
                pose = calib_result["pose"]
                local_pts_body = live_state["local_pts_body"]
                local_pts_search = live_state["local_pts_search"]

                # --------------------------------------------
                # HAKU: uusia kiviä kiinteältä paata-rajatulta
                # vyohykkeelta (kamera9_02.py:n SEARCH_*), vain
                # jos tilaa (< MAX_CONCURRENT_STONES) ja tama on
                # hakutarkistusframe. C++-porttaus (stone_tracker.
                # cpp:n search_new_stone, Task 6) - ristikkohaku
                # JA yhteissovitus yhdessa kutsussa, samaan tapaan
                # kuin SEURANTA (Task 5).
                #
                # KAYNNISTETAAN OMALLE TAUSTASAIKEELLE (kayttajan
                # pyynnosta): HAKU ja SEURANTA ovat riippumattomia
                # (molemmat lukevat vain jo valmiin frame_u:n,
                # eivat toistensa tulosta talta framelta), joten
                # ne ajetaan SAMANAIKAISESTI - HAKU taustasaikeessa,
                # SEURANTA paasaikeessa alempana. TULOS KERATAAN
                # VASTA SEURANTAN JALKEEN (katso "HAKU:n tuloksen
                # keraaminen" alempana) - uuden kiven rekisterointi
                # pysyy SILTI TASMALLEEN samalla framella kuin ennen,
                # EI viivastu, koska odotamme HAKU:n tuloksen ennen
                # seuraavaan frameen siirtymista. Katso stone_tracker.
                # cpp:n search_new_stone/track_stones_batch -kommentit
                # (ScopedSingleThreadedOpenCV) OpenCV:n oman sisaisen
                # rinnakkaistuksen turvallisesta poiskytkennasta taman
                # samanaikaisuuden ajaksi.
                # --------------------------------------------

                haku_future = None

                if (
                    len(active_stones) < MAX_CONCURRENT_STONES
                    and frame_index % haku_interval_frames == 0
                ):

                    x_center = 0.0
                    y_center = (
                        k92.SEARCH_Y_MIN_CM + k92.SEARCH_Y_MAX_CM
                    ) / 2.0
                    y_half = (
                        k92.SEARCH_Y_MAX_CM - k92.SEARCH_Y_MIN_CM
                    ) / 2.0

                    haku_future = haku_executor.submit(
                        _run_haku_timed,
                        frame_u, calib_result["calib"]["frame_undistorted"],
                        local_pts_body, local_pts_search,
                        pose["K"], pose["R"], pose["t"],
                        x_center, k92.SEARCH_X_HALF_WIDTH_CM, y_center, y_half,
                        k92.SEARCH_COARSE_STEP_CM, k92.SEARCH_FINE_STEP_CM,
                        k92.SEARCH_SCORE_THRESHOLD,
                        live_state["R_max"], live_state["H_total"],
                        live_state["ring_r_frac_guess"]
                    )

                # --------------------------------------------
                # SEURANTA: paivitetaan JOKAINEN aktiivinen kivi
                # JOKA frame - C++-porttaus (stone_tracker.cpp,
                # Task 5) laskee KAIKKIEN taman framen SEURANTA-
                # kivien ROI-rajatun maskin/saturaation, ristikko-
                # haun ja LM-yhteissovituksen YHDESSA std::thread-
                # rinnakkaistetussa kutsussa (yksi worker-saie per
                # kivi, atominen tyonvarastus - katso mode_engine.
                # cpp:n save_mode-kommentti samasta periaatteesta).
                # TASMALLEEN sama matematiikka/algoritmi kuin
                # k94.locate_by_grid_search_fast + k94.refine_
                # position_joint_fast - katso stone_tracker.cpp:n
                # oma kommentti numeerisesta validoinnista (oikealla
                # videolla, Python-tulosta vasten) seka tiedoston
                # alun kommentti sen tunnetusta, algoritmin OMASTA
                # (ei porttausvirheen) numeerisesta herkkyydesta
                # refine_position_joint:in MAD-poikkeavien-hylkays-
                # kynnyksella. Kayttajan pyynnosta (katso git-historia):
                # SAA nyt taustanvaimennuksen (calib_result:in staattinen
                # referenssikuva) SAMOIN kuin HAKU jo aiemmin - ilman
                # tata kivi menetti tarkkuutensa (rms_px jopa 130-146px
                # normaalin ~2px sijaan) aina kun se ylitti staattisen
                # jaamerkinnan (pesan renkaat, hogline, mainokset).
                # --------------------------------------------

                still_active = []

                # DEBUG_SAVE_TRACKING_VIDEO:in kerays - katso taman
                # tiedoston alkupaan kommentti. Lista (X_cm, Y_cm,
                # stone_id, vari_bgr, teksti) - piirretaan framelle
                # HAKU:n tuloksen keraamisen jalkeen alempana.
                debug_draw_items = []

                # HAKU:n (jos kaynnissa) mahdollisesti loytama uusi
                # kivi EI ole viela active_stones:issa tassa vaiheessa
                # (sen tulos kerataan vasta alempana) - ei siis
                # tarvetta erikseen suodattaa sita pois taalta.
                seuranta_stones = active_stones

                if seuranta_stones:

                    X0_arr = np.array(
                        [s["last_xy"][0] for s in seuranta_stones],
                        dtype=np.float64
                    )
                    Y0_arr = np.array(
                        [s["last_xy"][1] for s in seuranta_stones],
                        dtype=np.float64
                    )

                    t_seuranta0 = time.time()
                    batch_results = stone_tracker.track_stones_batch(
                        frame_u, calib_result["calib"]["frame_undistorted"],
                        X0_arr, Y0_arr,
                        local_pts_body, local_pts_search,
                        pose["K"], pose["R"], pose["t"],
                        k92.TRACK_HALF_RANGE_CM,
                        k92.TRACK_COARSE_STEP_CM, k92.TRACK_FINE_STEP_CM,
                        k92.TRACK_SCORE_THRESHOLD,
                        live_state["R_max"], live_state["H_total"],
                        live_state["ring_r_frac_guess"]
                    )
                    total_seuranta_time += time.time() - t_seuranta0
                    n_seuranta_calls += 1
                    n_seuranta_stone_updates += len(seuranta_stones)

                    color_ref = live_state["color_reference"]
                    frame_u_f64 = None

                    for s, refined in zip(seuranta_stones, batch_results):

                        # --------------------------------
                        # VARITARKISTUS: katso kommentti COLOR_REF_*/
                        # COLOR_STOP_MAX_AVG_DIFF:in kohdalla taman
                        # tiedoston alkupaassa. HUOM (kayttajan pyynnosta
                        # tehty uudelleensuunnittelu): tama EI vaikuta
                        # MITENKAAN normaaliin SEURANTAan (haku-ankkuri,
                        # s["misses"], "kadotettu") - position-/miss-
                        # logiikka alla on identtinen kuin ennen vari-
                        # tarkistuksen lisaamista, joten heittaja/
                        # lakaisija kiven paalla EI VOI aiheuttaa kiven
                        # katoamista sen enempaa kuin ennenkaan. Vari
                        # keraytyy vain LIUKUVAAN IKKUNAAN (color_diff_
                        # history, sama aikaikkuna kuin position_history)
                        # ja sen KESKIARVOA kaytetaan LISAEHTONA vasta
                        # "pysahtynyt"-ilmoituksen kohdalla (alempana) -
                        # jos vari ei vahvista, ilmoitus vain LYKKAANTYY
                        # (kivi pysyy normaalisti seurattuna), ei koskaan
                        # katoa/vaaristu taman takia.
                        # --------------------------------

                        if refined["found"] and color_ref is not None:

                            if frame_u_f64 is None:
                                frame_u_f64 = frame_u.astype(np.float64)

                            median_diff = color_match_median_diff(
                                frame_u_f64, local_pts_body, pose,
                                refined["X_cm"], refined["Y_cm"],
                                color_ref, width, height
                            )

                            if median_diff is not None:

                                diff_history = s["color_diff_history"]
                                diff_history.append((frame_index, median_diff))

                                while (
                                    diff_history
                                    and frame_index - diff_history[0][0]
                                    > stop_tracking_frames
                                ):
                                    diff_history.pop(0)

                                if os.environ.get("COLOR_DEBUG"):
                                    print(
                                        f"[COLOR_DEBUG] frame={frame_index} "
                                        f"stone={s['stone_id']} "
                                        f"diff={median_diff:.3f}"
                                    )

                        if refined["found"]:

                            s["last_xy"] = (
                                refined["X_cm"], refined["Y_cm"]
                            )
                            s["misses"] = 0

                            debug_draw_items.append((
                                refined["X_cm"], refined["Y_cm"],
                                s["stone_id"],
                                (0, 255, 0) if refined["tarkka"] else (0, 165, 255),
                                str(s["stone_id"])
                            ))

                            _write_stone_csv_row(
                                csv_writer, frame_index, timestamp,
                                s["stone_id"], refined
                            )

                            # --------------------------------
                            # PYSAHTYMISTARKISTUS: katso taman
                            # tiedoston alkupaan kommentti STOP_
                            # TRACKING_SECONDS/DISPLACEMENT_CM:sta.
                            # Liukuva ikkuna framen INDEKSIN, ei
                            # listan pituuden, mukaan - kestaa
                            # satunnaiset valiin jaavat missit.
                            # --------------------------------

                            history = s["position_history"]
                            history.append((
                                frame_index,
                                refined["X_cm"], refined["Y_cm"]
                            ))

                            while (
                                history[-1][0] - history[0][0]
                                > stop_tracking_frames
                            ):
                                history.pop(0)

                            stopped = False

                            if (
                                history[-1][0] - history[0][0]
                                >= stop_tracking_frames
                            ):
                                _, old_x, old_y = history[0]
                                displacement = math.hypot(
                                    s["last_xy"][0] - old_x,
                                    s["last_xy"][1] - old_y
                                )

                                if displacement < STOP_TRACKING_DISPLACEMENT_CM:

                                    # --------------------------------
                                    # VARIVAHVISTUS: katso kommentti
                                    # COLOR_STOP_MAX_AVG_DIFF:in kohdalla.
                                    # Viimeisen sekunnin KESKIMAARAINEN
                                    # vari (EI yksittainen ruutu - kestaa
                                    # siis kohinan/hetkelliset poikkeamat)
                                    # on oltava aidosti graniittimainen
                                    # ennen kuin "pysahtynyt" hyvaksytaan
                                    # lopullisena - estaa vaaran kohteen
                                    # (esim. pelaaja) virheellisen
                                    # kirjaamisen kiveksi.
                                    # --------------------------------

                                    color_confirms = True

                                    if color_ref is not None and s["color_diff_history"]:
                                        recent_diffs = [
                                            d for _, d in s["color_diff_history"]
                                        ]
                                        avg_diff = (
                                            sum(recent_diffs) / len(recent_diffs)
                                        )
                                        color_confirms = (
                                            avg_diff <= COLOR_STOP_MAX_AVG_DIFF
                                        )

                                    if color_confirms:
                                        stopped = True
                                        print(
                                            f"[frame {frame_index}] Kivi "
                                            f"{s['stone_id']} pysahtynyt "
                                            f"(liikkunut {displacement:.1f}cm "
                                            f"viimeisen {STOP_TRACKING_SECONDS:.0f}s "
                                            "aikana) - lopetetaan seuranta."
                                        )

                            if not stopped:
                                still_active.append(s)

                        else:

                            s["misses"] += 1

                            debug_draw_items.append((
                                s["last_xy"][0], s["last_xy"][1],
                                s["stone_id"], (0, 0, 255),
                                f"{s['stone_id']} MISS"
                            ))

                            if s["misses"] < k92.TRACK_LOST_MAX_MISSES:
                                still_active.append(s)
                            else:
                                print(
                                    f"[frame {frame_index}] Kivi "
                                    f"{s['stone_id']} kadotettu."
                                )

                active_stones = still_active

                # --------------------------------------------
                # HAKU:n tuloksen keraaminen - SEURANTA (ylla) ehti
                # jo laskea RINNAN HAKU:n kanssa, joten odotus tassa
                # (.result(), jos HAKU on viela kesken) on vain sen
                # verran kuin HAKU oli SEURANTAa hitaampi (yleensa
                # HAKU on selvasti hitaampi -> odotus n. HAKU_aika -
                # SEURANTA_aika, joskus jopa 0 jos SEURANTA oli
                # hitaampi). Uusi kivi lisataan TASSA active_stones:
                # iin - TASMALLEEN samalla framella/timestampilla
                # kuin ennen, EI viivastynyt kayttaytyminen.
                # --------------------------------------------

                if haku_future is not None:

                    haku_result, haku_dt = haku_future.result()
                    total_haku_time += haku_dt
                    n_haku_calls += 1

                    if haku_result["found"]:

                        refined = haku_result
                        bx, by = refined["X_cm"], refined["Y_cm"]

                        already_tracked = any(
                            math.hypot(
                                bx - s["last_xy"][0], by - s["last_xy"][1]
                            ) < NEW_STONE_DEDUP_CM
                            for s in active_stones
                        )

                        if not already_tracked:

                            stone_id = next_stone_id
                            next_stone_id += 1

                            active_stones.append({
                                "stone_id": stone_id,
                                "last_xy": (
                                    refined["X_cm"], refined["Y_cm"]
                                ),
                                "misses": 0,
                                "color_diff_history": [],
                                "position_history": [(
                                    frame_index,
                                    refined["X_cm"], refined["Y_cm"]
                                )],
                            })

                            debug_draw_items.append((
                                refined["X_cm"], refined["Y_cm"],
                                stone_id, (0, 255, 255),
                                f"{stone_id} UUSI"
                            ))

                            print(
                                f"[frame {frame_index}] Uusi kivi "
                                f"{stone_id}: "
                                f"({refined['X_cm']:.1f}, "
                                f"{refined['Y_cm']:.1f}) cm"
                            )

                            _write_stone_csv_row(
                                csv_writer, frame_index, timestamp,
                                stone_id, refined
                            )

                # --------------------------------------------
                # DEBUG_SAVE_TRACKING_VIDEO: piirretaan taman framen
                # kaikkien HAKU/SEURANTA-havaintojen (debug_draw_items,
                # katso yllapuoliset kohdat) ENNUSTETUT ääriviivat
                # (k94.predicted_stone_hull_fast - sama profiilimalli
                # jota itse yhteissovituskin kayttaa) frame_u:n paalle
                # ja kirjoitetaan debug-videoon. Katso taman tiedoston
                # alkupaan DEBUG_SAVE_TRACKING_VIDEO-kommentti varien
                # merkityksesta.
                # --------------------------------------------

                if debug_video_writer is not None:

                    debug_frame = frame_u.copy()

                    for bx, by, s_id, color, label in debug_draw_items:

                        hull = k94.predicted_stone_hull_fast(
                            local_pts_body, pose, bx, by
                        )

                        if hull is None:
                            continue

                        hull_i = hull.astype(np.int32)

                        cv2.polylines(
                            debug_frame, [hull_i], True, color, 2
                        )

                        cv2.putText(
                            debug_frame, label,
                            (
                                int(hull_i[:, 0, 0].min()),
                                int(hull_i[:, 0, 1].min()) - 8
                            ),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
                        )

                    cv2.putText(
                        debug_frame,
                        f"frame {frame_index}  t={timestamp:.2f}s  "
                        f"kivia={len(active_stones)}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 255, 255), 2
                    )

                    debug_video_writer.write(debug_frame)

            frame_index += 1

            # ------------------------------------------------
            # ETA
            # ------------------------------------------------

            if (
                frame_index % REPORT_EVERY == 0
                or frame_index == total_frames
            ):

                elapsed = (
                    time.time() -
                    start_time
                )

                if frame_index > 0:

                    frames_per_second = (
                        frame_index /
                        elapsed
                    )

                    remaining_frames = (
                        total_frames -
                        frame_index
                    )

                    eta_seconds = (
                        remaining_frames /
                        frames_per_second
                    )

                else:

                    frames_per_second = 0
                    eta_seconds = 0

                elapsed_minutes = int(
                    elapsed // 60
                )

                elapsed_secs = int(
                    elapsed % 60
                )

                eta_minutes = int(
                    eta_seconds // 60
                )

                eta_secs = int(
                    eta_seconds % 60
                )

                print(
                    f"Ruutu "
                    f"{frame_index} / "
                    f"{total_frames} | "
                    f"C++:lle tallennettuja kuvia: "
                    f"{engine.mode_frame_count()} | "
                    f"nopeus: "
                    f"{frames_per_second:.1f} r/s | "
                    f"kulunut: "
                    f"{elapsed_minutes:02d}:"
                    f"{elapsed_secs:02d} | "
                    f"ETA: "
                    f"{eta_minutes:02d}:"
                    f"{eta_secs:02d}"
                )
                processed = frame_index
                print(
                    f"  keskimäärin: "
                    f"gray {(total_gray_time / processed) * 1000:.1f} ms | "
                    f"tracking {(total_tracking_time / processed) * 1000:.1f} ms | "
                    f"transform {(total_transform_time / processed) * 1000:.3f} ms | "
                    f"sample {(total_sample_time / max(1, engine.mode_frame_count())) * 1000:.1f} ms"
                )
                print(
                    f"  kivenseuranta (C++): "
                    f"HAKU {n_haku_calls} kutsua, "
                    f"ka {(total_haku_time / max(1, n_haku_calls)) * 1000:.1f} ms/kutsu, "
                    f"{(total_haku_time / processed) * 1000:.2f} ms/ruutu ka | "
                    f"SEURANTA {n_seuranta_calls} kutsua, "
                    f"ka {(total_seuranta_time / max(1, n_seuranta_calls)) * 1000:.1f} ms/kutsu "
                    f"({n_seuranta_stone_updates / max(1, n_seuranta_calls):.1f} kivea/kutsu ka), "
                    f"{(total_seuranta_time / processed) * 1000:.2f} ms/ruutu ka"
                )
                print(
                    f"  muu (ei viela optimoitu): "
                    f"read(video) {(total_read_time / processed) * 1000:.2f} ms/ruutu | "
                    f"stabilointi-RANSAC {(total_stabilize_compute_time / processed) * 1000:.2f} ms/ruutu | "
                    f"warpAffine+remap(koko frame) {(total_warp_remap_time / processed) * 1000:.2f} ms/ruutu"
                )

    finally:

        executor.shutdown(
            wait=True
        )

        haku_executor.shutdown(
            wait=True
        )

        if csv_file is not None:
            csv_file.close()

        if debug_video_writer is not None:
            debug_video_writer.release()

    print()

    print(
        f"Videon lapikaynti valmis "
        f"({frame_index}/{total_frames} ruutua)."
    )

    # --------------------------------------------------------
    # NOPEUSSEURANNAN LOPPUYHTEENVETO - kopioi/liita tama takaisin
    # jos haluat kertoa miten kivenseuranta (HAKU+SEURANTA, molemmat
    # C++:aa) kayttaytyy omalla koneellasi oikealla datalla.
    # --------------------------------------------------------

    total_elapsed = time.time() - start_time
    processed_frames = max(1, frame_index)

    print()
    print("=== NOPEUSSEURANTA (kivenseuranta C++: HAKU+SEURANTA) ===")
    print(f"Koko ajo: {total_elapsed:.1f}s / {frame_index} ruutua "
          f"({processed_frames / total_elapsed:.2f} r/s keskimaarin)")
    print(f"HAKU: {n_haku_calls} kutsua, yhteensa {total_haku_time:.2f}s, "
          f"ka {(total_haku_time / max(1, n_haku_calls)) * 1000:.2f} ms/kutsu, "
          f"{(total_haku_time / processed_frames) * 1000:.2f} ms/ruutu "
          "(koko videon yli keskiarvoistettuna)")
    print(f"SEURANTA: {n_seuranta_calls} kutsua, yhteensa "
          f"{total_seuranta_time:.2f}s, "
          f"ka {(total_seuranta_time / max(1, n_seuranta_calls)) * 1000:.2f} ms/kutsu "
          f"({n_seuranta_stone_updates / max(1, n_seuranta_calls):.2f} kivea/kutsu "
          "keskimaarin), "
          f"{(total_seuranta_time / processed_frames) * 1000:.2f} ms/ruutu "
          "(koko videon yli keskiarvoistettuna)")
    print(f"read(video): {total_read_time:.2f}s yhteensa, "
          f"{(total_read_time / processed_frames) * 1000:.2f} ms/ruutu")
    print(f"stabilointi-RANSAC (cv2.estimateAffinePartial2D+mediaani): "
          f"{total_stabilize_compute_time:.2f}s yhteensa, "
          f"{(total_stabilize_compute_time / processed_frames) * 1000:.2f} ms/ruutu")
    print(f"warpAffine+remap (KOKO frame, joka elavan seurannan ruutu): "
          f"{total_warp_remap_time:.2f}s yhteensa, "
          f"{(total_warp_remap_time / processed_frames) * 1000:.2f} ms/ruutu")
    muu_yhteensa = (
        total_read_time + total_stabilize_compute_time
        + total_warp_remap_time + total_gray_time
        + total_tracking_time + total_transform_time
    )
    print(f"Yhteensa HAKU+SEURANTA+muu mitattu: "
          f"{((total_haku_time + total_seuranta_time + muu_yhteensa) / processed_frames) * 1000:.2f} "
          "ms/ruutu keskimaarin (25fps-reaaliaikatavoite = 40.0 ms/ruutu, "
          "tiukempi tavoite hyvalla marginaalilla = 20.0 ms/ruutu)")
    print("===========================================================")

    if calib_result is None:
        raise RuntimeError(
            "Kalibrointi ei onnistunut - video loppui kesken "
            "moodinaytteiden keruun (video liian lyhyt?)."
        )

    if profile_result is None and best_sufficient_profile is not None:
        print(
            f"VAROITUS: alle {PROFILE_MIN_ACCEPTED_STONES} kivea "
            f"loytyi ({n_accepted_stones} kpl) ennen videon loppua - "
            "kaytetaan viimeisinta riittavaa profiilia silti."
        )
        profile_result = best_sufficient_profile

    if profile_result is None:
        print(
            "VAROITUS: 3D-kiviprofiili ei tullut riittavaksi ennen "
            f"videon loppua ({len(accumulated_stones)} havaintoa "
            "kerattyna)."
        )

    return {
        "engine": engine,
        "calib": calib_result["calib"],
        "pose": calib_result["pose"],
        "profile": profile_result,
        "n_profile_observations": len(accumulated_stones),
        "csv_output": csv_output if live_state is not None else None,
        "n_stones_seen": next_stone_id,
        "debug_video_output": (
            debug_video_output if debug_video_writer is not None else None
        ),
    }


# ============================================================
# MAIN
# ============================================================

def _time_str_to_seconds(time_str):
    """Muuntaa "[HH:]MM:SS"-muotoisen ajan (tai pelkan sekuntiluvun)
    sekunneiksi."""

    seconds = 0.0

    for part in time_str.split(":"):
        seconds = seconds * 60 + float(part)

    return seconds


def main(debug=None, start_time=None, end_time=None):

    # debug=None (oletus): kayta DEBUG_SAVE_TRACKING_VIDEO-vakion
    # arvoa (katso sen kommentti). debug=True/False komentoriviltä
    # (-d/--debug, katso alempana if __name__=="__main__") ohittaa
    # vakion - kayttajan pyynnosta dynaaminen paalle/pois-kytkenta
    # ilman lahdekoodin muokkausta.
    effective_debug = (
        DEBUG_SAVE_TRACKING_VIDEO if debug is None else debug
    )

    root = tk.Tk()
    root.withdraw()
    
    input_file = filedialog.askopenfilename(
        title="Valitse video",
        filetypes=[
            (
                "Videot",
                "*.mts *.MTS *.mp4 *.MP4 "
                "*.mov *.MOV *.avi *.AVI"
            ),
            (
                "Kaikki tiedostot",
                "*.*"
            )
        ]
    )

    root.destroy()

    if not input_file:

        print(
            "Videota ei valittu."
        )

        return

    input_stem, _input_ext = os.path.splitext(os.path.basename(input_file))

    video_file = os.path.join(
        os.path.dirname(input_file),
        f"{input_stem}_leikattu.mp4"
    )

    print()
    print(
        f"Video: {input_file}"
    )

    command = ["ffmpeg"]

    if start_time is not None:
        command += ["-ss", start_time]

    command += ["-i", input_file]

    if end_time is not None:
        # HUOM: "-to" input-option "-ss":n jalkeen EI viittaa alkuperaisen
        # videon aikajanaan vaan "-ss":n siirtamaan (nollasta alkavaan)
        # ulostulon aikajanaan - "--end 00:21:00" leikkaisi siis 21 min
        # PITUISEN palan alkaen "--start"-kohdasta, ei alkuperaisen videon
        # kohtaan 21 min asti. Lasketaan siksi KESTO ("-t") itse.
        start_seconds = (
            _time_str_to_seconds(start_time) if start_time is not None else 0.0
        )
        end_seconds = _time_str_to_seconds(end_time)
        duration_seconds = end_seconds - start_seconds

        if duration_seconds <= 0:
            raise ValueError(
                f"--end ({end_time}) on ennen tai samassa kohdassa kuin "
                f"--start ({start_time})."
            )

        command += ["-t", str(duration_seconds)]

    command += ["-c", "copy", video_file]
    
    subprocess.run(command, check=True)
    
    # --------------------------------------------------------
    # PANEELIT - automaattitunnistus referenssitiedoston lahelta
    # (katso detect_panels_from_reference:in kommentti) - EI enaa
    # kasin klikkausta.
    # --------------------------------------------------------

    panel_data = load_panel_data(
        video_file
    )

    if panel_data is None:

        print()

        root = tk.Tk()
        root.withdraw()

        reference_file = filedialog.askopenfilename(
            title="Valitse referenssi-paneelitiedosto "
                  "(*_panel_corners.txt samantyyppisesta "
                  "kamera-asettelusta)",
            filetypes=[
                ("Paneelitiedostot", "*_panel_corners.txt"),
                ("Kaikki tiedostot", "*.*"),
            ]
        )

        root.destroy()

        if not reference_file:
            print("Referenssitiedostoa ei valittu.")
            return

        reference_panels = load_panel_reference(reference_file)

        cap = cv2.VideoCapture(video_file)

        if not cap.isOpened():
            raise RuntimeError("Videotiedostoa ei voitu avata.")

        ret, first_frame = cap.read()
        cap.release()

        if not ret:
            raise RuntimeError("Videon ensimmaista kuvaa ei voitu lukea.")

        first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

        panel_data = detect_panels_from_reference(
            first_gray, reference_panels
        )

        filename = save_panel_data(
            panel_data,
            video_file
        )

        print()

        print(
            f"Paneelien koordinaatit "
            f"tallennettu: {filename}"
        )

    # --------------------------------------------------------
    # PAAPUTKI: kalibrointi -> 3D-kiviprofiilin haku -> elava
    # moni-kiven seuranta + CSV
    # --------------------------------------------------------

    calib_diag_output = (
        os.path.splitext(video_file)[0] +
        "_kalibrointi_topdown.png"
    )

    csv_output = (
        os.path.splitext(video_file)[0] +
        "_kivien_sijainnit.csv"
    )

    debug_video_output = None

    if effective_debug:

        debug_video_output = (
            os.path.splitext(video_file)[0] +
            "_debug_seuranta.mp4"
        )

        print(
            f"Debug-seurantavideo: {debug_video_output}"
        )

    result = run_pipeline(
        video_file,
        panel_data,
        calib_diag_output,
        csv_output,
        debug_video_output=debug_video_output
    )

    print()
    print("=" * 60)
    print("KALIBROINTI + 3D-KIVIPROFIILI VALMIS")
    print("=" * 60)

    print(
        f"Resoluutio: "
        f"{result['engine'].width()} x "
        f"{result['engine'].height()}"
    )

    print(
        f"Topdown-tarkistuskuva: {calib_diag_output}"
    )

    if result["profile"] is not None:

        profile = result["profile"]

        print(
            f"3D-kiviprofiili: R_max={profile['R_max_cm']:.2f} cm, "
            f"H_total={profile['H_total_cm']:.2f} cm, "
            f"RMS={profile['residual_rms_px']:.2f} px "
            f"({result['n_profile_observations']} havaintoa)"
        )

    else:

        print(
            "3D-kiviprofiili EI valmistunut riittavaksi "
            f"({result['n_profile_observations']} havaintoa kerattyna)."
        )

    if result["csv_output"] is not None:

        print(
            f"Kivien sijainti-CSV: {result['csv_output']} "
            f"({result['n_stones_seen']} eri kivea havaittu)"
        )

    if result["debug_video_output"] is not None:

        print(
            f"Debug-seurantavideo: {result['debug_video_output']}"
        )


if __name__ == "__main__":

    # Kayttajan pyynnosta: debug-seurantavideon (DEBUG_SAVE_TRACKING_
    # VIDEO, katso sen kommentti) voi kytkea paalle komentorivilta
    # ilman lahdekoodin muokkausta, esim: python main.py --debug
    _arg_parser = argparse.ArgumentParser()
    _arg_parser.add_argument(
        "--debug", "-d", action="store_true", default=None,
        help=(
            "Tallenna debug-seurantavideo (<video>_debug_seuranta.mp4) "
            "jossa HAKU/SEURANTA-tunnistusten ennustetut ääriviivat on "
            "piirretty framejen paalle. Ohittaa DEBUG_SAVE_TRACKING_"
            "VIDEO-vakion. Ilman tata lippua kaytetaan vakion arvoa."
        )
    )

    _arg_parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Videon alkuaika, esim. 00:10:00"
    )

    _arg_parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="Videon loppuaika, esim. 00:21:00"
    )
    
    _args = _arg_parser.parse_args()

    main(debug=_args.debug,
        start_time=_args.start,
        end_time=_args.end)
