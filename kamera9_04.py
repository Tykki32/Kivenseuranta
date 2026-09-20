# ============================================================
# kamera9_04.py - SAMA YHTEISSOVITUS KUIN kamera9_03.py, MUTTA
# NOPEUTETTU LIVE-KAYTTOA VARTEN (tavoite: n. 0.6s/frame -> paljon
# alle, jotta videon voi lukea suunnilleen reaaliajassa)
#
# EI MUUTETA MATEMATIIKKAA/ALGORITMIA - kamera9_03.py:n tulos on
# kayttajan mukaan jo riittavan tarkka ("todennakoisesti tuolla
# lahderesoluutiolla ei saa parempaa"), joten tama tiedosto EI YRITA
# parantaa tarkkuutta eika loysaa mitaan kynnysarvoa/toleranssia -
# se ratkaisee TASMALLEEN saman [X,Y,rengas_R]-LM-sovituksen samoilla
# pisteilla ja samoilla iteraatioilla, vain NOPEAMMIN LASKETTUNA.
# Profiloinnilla (cProfile, 30 oikeaa framea SEURANTA-tilassa)
# loydetyt pullonkaulat JA niiden korjaus tassa tiedostossa:
#
#   1) _predicted_stone_hull (kamera9_01.py) laskee Catmull-Rom-
#      splinin ja rakentaa kaikki rengaspisteet UUDELLEEN JOKAISELLA
#      KUTSULLA, vaikka kiven MUOTO (profile: R_max/H_total/
#      shape_deltas) on VAKIO koko videon ajan - vain (X0,Y0) muuttuu
#      kutsujen valilla. Tama oli ~80% koko framen ajasta (LM-sovitus
#      kutsuu sen satoja kertoja/frame, karkea ristikkohaku ~200
#      kertaa/frame lisaksi). KORJAUS: lasketaan kiven paikallinen
#      (origon-suhteinen) rengasgeometria KERRAN (_build_local_rings),
#      ja jokainen kutsu vain SIIRTAA (+X0,+Y0) ja projisoi sen -
#      TASMALLEEN sama lopputulos, ei mitaan approksimaatiota.
#   2) refine_position_joint (kamera9_03.py) laski graniittimaskin
#      (create_granite_mask, mm. GaussianBlur+morfologia KOKO framelle)
#      UUDELLEEN, vaikka search_and_track_stones_precise oli JUURI
#      laskenut saman maskin ristikkohakua varten. KORJAUS: maski
#      lasketaan kerran/frame ja annetaan parametrina.
#   3) Saturaatiokanava (cv2.cvtColor+HSV, koko frame) laskettiin
#      UUDELLEEN seka _filter_ice_boundary_points:ssa etta JOKAISELLA
#      _detect_boundary_points-iteraatiolla (jopa 5x/frame). KORJAUS:
#      lasketaan kerran/frame ja annetaan parametrina.
#   4) Reunapisteiden sateenmuotoinen skannaus (_find_boundary_point_
#      along_ray) kutsui bilineaarista naytintoa (_bilinear_sample)
#      YKSITELLEN PYTHON-SILMUKASSA (kymmenia tuhansia funktiokutsuja/
#      frame). KORJAUS: vektoroitu numpylla (kaikki sateet+askeleet
#      yhdella kertaa) - sama lineaarinen interpolointi, sama tulos.
#   5) cv2.undistort() laskee EI-VAKIO-parametrien takia sisaisesti
#      UUDET vaantokartat JOKA KUTSULLA, vaikka kalibrointi on vakio
#      koko videolle. KORJAUS: initUndistortRectifyMap kerran + remap
#      per frame (sama INTER_LINEAR-interpolointi, sama tulos).
#
# EI KOSKETA: kamera9_01.py, kamera9_02.py, kamera9_03.py pysyvat
# koskemattomina (kayttajan aiempi pyynto) - tama tiedosto TUO NE
# SISAAN ja korvaa vain nama viisi pullonkaulaa uusilla, nopeammilla
# mutta MATEMAATTISESTI IDENTTISILLA funktioilla.
# ============================================================

import os
import csv
import math
import time
import importlib.util

import numpy as np
import cv2


def _load_kamera9_03():
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        "kamera9_03", os.path.join(here, "kamera9_03.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


k93 = _load_kamera9_03()
k92 = k93.k92
k9 = k93.k9
k8 = k93.k8


# ============================================================
# 1) KIVEN PAIKALLINEN RENGASGEOMETRIA KERTAALLEEN - _predicted_
#    stone_hull:in kallis Catmull-Rom+rengasrakennus tehdaan VAIN
#    KERRAN per (profiili, resoluutio) -yhdistelma, ja jokainen haku-
#    /sovituspiste vain siirtaa+projisoi valmiin pistejoukon.
# ============================================================

def build_local_stone_rings(R_max, H_total, shape_deltas, n_theta, n_per_segment):
    """
    Palauttaa (N,3)-taulukon KIVEN OMAAN PYSTYAKSELIIN (X0=0,Y0=0)
    NAHDEN paikallisia (x,y,z)-pisteita - TASMALLEEN samat pisteet
    jotka kamera9_01.py:n _predicted_stone_hull rakentaisi, ennen
    (X0,Y0)-siirtoa ja projisointia. Riippuu VAIN profiilin muodosta
    (ei X0/Y0:sta) - siksi tama voidaan laskea kerran ja uudelleen-
    kayttaa jokaisessa haku-/LM-kutsussa.
    """

    R_max = abs(R_max)
    H_total = max(abs(H_total), 1e-6)

    r_fracs = k9._TEMPLATE_R_FRAC + shape_deltas
    r_fracs[k9._TEMPLATE_EQUATOR_IDX] = 1.0
    r_fracs = np.clip(r_fracs, 0.05, 1.0)

    n_dense = max(len(k9._TEMPLATE_Z_FRAC) * n_per_segment, 2)
    z_frac_dense = np.linspace(0.0, 1.0, n_dense)
    r_frac_dense = k9._catmull_rom_r_frac(z_frac_dense, r_fracs=r_fracs)

    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)

    z = z_frac_dense * H_total          # (n_dense,)
    r = r_frac_dense * R_max            # (n_dense,)

    xs = r[:, None] * np.cos(theta)[None, :]     # (n_dense, n_theta)
    ys = r[:, None] * np.sin(theta)[None, :]
    zs = np.broadcast_to(z[:, None], xs.shape)

    return np.stack([xs, ys, zs], axis=-1).reshape(-1, 3)


def predicted_stone_hull_fast(local_pts, pose, X0, Y0):
    """Sama tulos kuin kamera9_01.py:n _predicted_stone_hull(pose, X0,
    Y0, ...) valmiiksi lasketulla local_pts:lla (build_local_stone_
    rings) - vain siirto+projisointi+kupera peite, ei splinia."""

    pts = local_pts + np.array([X0, Y0, 0.0])
    u, v = k9._project_3d(pose["K"], pose["R"], pose["t"], pts)
    points_2d = np.column_stack([u, v]).astype(np.float32)

    if not np.all(np.isfinite(points_2d)):
        return None

    hull = cv2.convexHull(points_2d)

    if len(hull) < 3:
        return None

    return hull


def profile_residuals_fast(local_pts, pose, X0, Y0, contour, n_sample):
    hull = predicted_stone_hull_fast(local_pts, pose, X0, Y0)
    sampled = k9._sample_contour_points(contour, n_sample)

    if hull is None:
        return np.full(len(sampled), 1000.0)

    return np.array([
        cv2.pointPolygonTest(hull, (float(p[0]), float(p[1])), True)
        for p in sampled
    ])


# ============================================================
# 2) KARKEA RISTIKKOHAKU (kamera9_02.py:n vastine) NOPEALLA HYLLYLLA
# ============================================================

def _score_candidate_fast(local_pts_search, mask, pose, X, Y):
    hull = predicted_stone_hull_fast(local_pts_search, pose, X, Y)
    return k92._hull_overlap_score(mask, hull)


def _grid_search_best_fast(local_pts_search, mask, pose, x_vals, y_vals):
    best_score = -1.0
    best_xy = (float(x_vals[0]), float(y_vals[0]))

    for X in x_vals:
        for Y in y_vals:
            score = _score_candidate_fast(local_pts_search, mask, pose, float(X), float(Y))
            if score > best_score:
                best_score = score
                best_xy = (float(X), float(Y))

    return best_xy, best_score


def locate_by_grid_search_fast(local_pts_search, mask, x_center, x_half_range, y_center, y_half_range,
                                coarse_step, fine_step, pose):

    x_vals = np.arange(x_center - x_half_range, x_center + x_half_range + 1e-6, coarse_step)
    y_vals = np.arange(y_center - y_half_range, y_center + y_half_range + 1e-6, coarse_step)
    (bx, by), score = _grid_search_best_fast(local_pts_search, mask, pose, x_vals, y_vals)

    x_vals = np.arange(bx - coarse_step, bx + coarse_step + 1e-6, fine_step)
    y_vals = np.arange(by - coarse_step, by + coarse_step + 1e-6, fine_step)
    (bx, by), score = _grid_search_best_fast(local_pts_search, mask, pose, x_vals, y_vals)

    return (bx, by), score


# ============================================================
# 3)+4) GRANIITTI/MUOVI-RAJAN ALIPIKSELIHAVAINNOT - VEKTOROITU
#    (sama lineaarinen kynnysylitys-interpolointi kuin kamera9_03.py,
#    mutta numpy-taulukkolaskentana Python-silmukan sijaan)
# ============================================================

def _bilinear_sample_vec(channel, xs, ys):
    """Vektoroitu versio kamera9_03.py:n _bilinear_sample:sta -
    palauttaa NaN:in siella missa piste on kuvan ulkopuolella."""

    h, w = channel.shape

    x0 = np.floor(xs).astype(np.int64)
    y0 = np.floor(ys).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1

    valid = (x0 >= 0) & (y0 >= 0) & (x1 < w) & (y1 < h)

    x0c = np.clip(x0, 0, w - 1)
    x1c = np.clip(x1, 0, w - 1)
    y0c = np.clip(y0, 0, h - 1)
    y1c = np.clip(y1, 0, h - 1)

    fx = xs - x0
    fy = ys - y0

    v00 = channel[y0c, x0c]
    v10 = channel[y0c, x1c]
    v01 = channel[y1c, x0c]
    v11 = channel[y1c, x1c]

    val = (v00 * (1 - fx) * (1 - fy) + v10 * fx * (1 - fy) +
           v01 * (1 - fx) * fy + v11 * fx * fy)

    return np.where(valid, val, np.nan)


def detect_boundary_points_fast(sat, cx_px, cy_px, r_px_approx,
                                 n_angles=k93.BOUNDARY_N_ANGLES,
                                 threshold=k93.BOUNDARY_SATURATION_THRESHOLD,
                                 r_step=k93.BOUNDARY_RADIAL_STEP_PX,
                                 r_factor_min=k93.BOUNDARY_RADIUS_SEARCH_FACTOR_MIN,
                                 r_factor_max=k93.BOUNDARY_RADIUS_SEARCH_FACTOR_MAX):
    """Sama tulos kuin kamera9_03.py:n _detect_boundary_points, mutta
    kaikki sateet+askeleet naytetaan YHDELLA vektoroidulla bilineaari-
    kutsulla (Python-silmukka jaljella vain 72 kulman ylitys-
    interpoloinnille, ei enaa per-askel-naytoille)."""

    r_min = r_px_approx * r_factor_min
    r_max = r_px_approx * r_factor_max

    n_steps = int(math.floor((r_max - r_min) / r_step)) + 1
    if n_steps < 2:
        return np.zeros((0, 2))

    r_vals = r_min + np.arange(n_steps) * r_step
    theta = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    xs = cx_px + r_vals[None, :] * cos_t[:, None]   # (n_angles, n_steps)
    ys = cy_px + r_vals[None, :] * sin_t[:, None]

    vals = _bilinear_sample_vec(sat, xs, ys)

    valid = ~np.isnan(vals)
    above = valid & (vals >= threshold)
    below = valid & (vals < threshold)

    # ENSIMMAINEN ylitys korkeasta (kahva) matalaan (graniitti) -
    # sama ehto kuin _find_boundary_point_along_ray: prev_v>=thr>v.
    cross = above[:, :-1] & below[:, 1:]

    points = []

    for i in range(n_angles):
        row = cross[i]
        if not row.any():
            continue
        idx = int(np.argmax(row))
        v0, v1 = vals[i, idx], vals[i, idx + 1]
        r0, r1 = r_vals[idx], r_vals[idx + 1]
        span = v1 - v0
        t = (threshold - v0) / span if abs(span) > 1e-9 else 0.0
        r_cross = r0 + t * (r1 - r0)
        points.append((cx_px + r_cross * cos_t[i], cy_px + r_cross * sin_t[i]))

    return np.array(points, dtype=np.float64) if points else np.zeros((0, 2))


def filter_ice_boundary_points_fast(sat, contour, check_dist_px=k93.BODY_HANDLE_CHECK_DIST_PX,
                                     threshold=k93.BOUNDARY_SATURATION_THRESHOLD):
    """Vektoroitu versio kamera9_03.py:n _filter_ice_boundary_points:sta
    - sama "katso hieman ulospain, hylkaa jos kirkas (kahva)" -periaate."""

    pts = contour.reshape(-1, 2).astype(np.float64)

    if len(pts) == 0:
        return np.zeros((0, 2))

    cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())

    dx = pts[:, 0] - cx
    dy = pts[:, 1] - cy
    norm = np.hypot(dx, dy)
    valid_norm = norm > 1e-6
    safe_norm = np.where(valid_norm, norm, 1.0)

    ox = pts[:, 0] + (dx / safe_norm) * check_dist_px
    oy = pts[:, 1] + (dy / safe_norm) * check_dist_px

    sat_val = _bilinear_sample_vec(sat, ox, oy)
    keep = valid_norm & ~np.isnan(sat_val) & (sat_val < threshold)

    return pts[keep]


# ============================================================
# YHTEISSOVITUS (kamera9_03.py:n refine_position_joint - sama
# algoritmi, mask+saturaatio annetaan valmiiksi laskettuna parametrina
# eika lasketa uudelleen, ja hitaat funktiokutsut korvattu nopeilla)
# ============================================================

def refine_position_joint_fast(mask, sat, local_pts_body, pose, profile, X0_approx, Y0_approx):

    R_max, H_total, shape_deltas = profile["R_max_cm"], profile["H_total_cm"], profile["shape_deltas"]
    ring_height_cm = H_total

    approx_pt3d = np.array([[X0_approx, Y0_approx, H_total / 2.0]])
    u0, v0 = k9._project_3d(pose["K"], pose["R"], pose["t"], approx_pt3d)
    approx_px = (float(u0[0]), float(v0[0]))

    raw_contour = k93._find_contour_near(mask, approx_px)
    body_pts = (
        filter_ice_boundary_points_fast(sat, raw_contour)
        if raw_contour is not None else np.zeros((0, 2))
    )
    n_body = len(body_pts)

    ring_r_frac_guess = float((k9._TEMPLATE_R_FRAC + shape_deltas)[-1])
    X_cur, Y_cur, R_cur = X0_approx, Y0_approx, ring_r_frac_guess * R_max
    ring_pts = np.zeros((0, 2))
    rms_px = None

    def residuals(pts_body, pts_ring):
        def f(params):
            X, Y, R = params
            parts = []
            if len(pts_body) > 0:
                parts.append(profile_residuals_fast(
                    local_pts_body, pose, X, Y, pts_body, len(pts_body)
                ))
            if len(pts_ring) > 0:
                parts.append(k93._ring_point_residuals(pose, X, Y, R, ring_height_cm, pts_ring))
            return np.concatenate(parts) if parts else np.zeros(0)
        return f

    for _ in range(k93.BOUNDARY_MAX_ITERATIONS):

        center_pts_3d = np.array([
            [X_cur, Y_cur, ring_height_cm],
            [X_cur + R_cur, Y_cur, ring_height_cm],
        ])
        u, v = k9._project_3d(pose["K"], pose["R"], pose["t"], center_pts_3d)

        cx_px, cy_px = float(u[0]), float(v[0])
        r_px_approx = math.hypot(float(u[1]) - cx_px, float(v[1]) - cy_px)

        ring_pts = np.zeros((0, 2))

        if r_px_approx >= k93.BOUNDARY_MIN_PREDICTED_RADIUS_PX:
            candidate_ring_pts = detect_boundary_points_fast(sat, cx_px, cy_px, r_px_approx)
            if (len(candidate_ring_pts) >= k93.BOUNDARY_MIN_VALID_POINTS and
                    k93._angular_spread_deg(candidate_ring_pts, cx_px, cy_px) >= k93.BOUNDARY_MIN_ANGULAR_SPREAD_DEG):
                ring_pts = candidate_ring_pts

        n_ring = len(ring_pts)

        if n_body < k93.BODY_MIN_VALID_POINTS and n_ring < k93.BOUNDARY_MIN_VALID_POINTS:
            return {
                "X_cm": X0_approx, "Y_cm": Y0_approx, "tarkka": False,
                "n_body": n_body, "n_ring": n_ring, "rms_px": None, "ring_radius_cm": None,
            }

        params0 = np.array([X_cur, Y_cur, R_cur], dtype=np.float64)
        res_fn = residuals(body_pts, ring_pts)
        params_final = k8._levenberg_marquardt(res_fn, params0, max_iterations=30)

        clean_body = body_pts
        clean_ring = ring_pts

        if n_body > 0:
            body_resid = profile_residuals_fast(
                local_pts_body, pose, params_final[0], params_final[1], body_pts, len(body_pts)
            )
            mad_b = float(np.median(np.abs(body_resid - np.median(body_resid)))) + 1e-6
            body_inliers = np.abs(body_resid - np.median(body_resid)) < 3.0 * 1.4826 * mad_b
            if np.count_nonzero(body_inliers) >= k93.BODY_MIN_VALID_POINTS:
                clean_body = body_pts[body_inliers]

        if n_ring > 0:
            ring_resid = k93._ring_point_residuals(
                pose, params_final[0], params_final[1], params_final[2], ring_height_cm, ring_pts
            )
            mad_r = float(np.median(np.abs(ring_resid - np.median(ring_resid)))) + 1e-6
            ring_inliers = np.abs(ring_resid - np.median(ring_resid)) < 3.0 * 1.4826 * mad_r
            if np.count_nonzero(ring_inliers) >= k93.BOUNDARY_MIN_VALID_POINTS:
                clean_ring = ring_pts[ring_inliers]

        if len(clean_body) != n_body or len(clean_ring) != n_ring:
            params_final = k8._levenberg_marquardt(
                residuals(clean_body, clean_ring), params_final, max_iterations=30
            )

        resid_final = residuals(clean_body, clean_ring)(params_final)
        rms_px = float(np.sqrt(np.mean(resid_final ** 2))) if len(resid_final) else None

        X_new, Y_new, R_new = float(params_final[0]), float(params_final[1]), float(abs(params_final[2]))
        moved = math.hypot(X_new - X_cur, Y_new - Y_cur)
        X_cur, Y_cur, R_cur = X_new, Y_new, R_new

        if moved < k93.BOUNDARY_CONVERGENCE_CM:
            break

    total_shift = math.hypot(X_cur - X0_approx, Y_cur - Y0_approx)

    if total_shift > k93.BOUNDARY_MAX_SHIFT_FROM_APPROX_CM:
        return {
            "X_cm": X0_approx, "Y_cm": Y0_approx, "tarkka": False,
            "n_body": n_body, "n_ring": len(ring_pts), "rms_px": rms_px, "ring_radius_cm": None,
        }

    return {
        "X_cm": X_cur, "Y_cm": Y_cur, "tarkka": True,
        "n_body": n_body, "n_ring": len(ring_pts), "rms_px": rms_px, "ring_radius_cm": R_cur,
    }


# ============================================================
# 5) ENNAKKOLASKETTU UNDISTORT-KARTTA - cv2.undistort() laskisi
#    saman kartan sisaisesti JOKA KUTSULLA, vaikka kalibrointi ei
#    muutu framejen valilla.
# ============================================================

def _build_undistort_maps(camera_matrix, dist_coeffs, frame_size):
    map1, map2 = cv2.initUndistortRectifyMap(
        camera_matrix, dist_coeffs, None, camera_matrix, frame_size, cv2.CV_32FC1
    )
    return map1, map2


# ============================================================
# 6) RAJATTU KUVA-ALUE (ROI) MASKILLE+SATURAATIOLLE SEURANTA-TILASSA
#
# create_granite_mask (kamera9_01.py) on jaljelle jaanyt suurin
# yksittainen kustannus (profiloinnissa n. 40% framen ajasta,
# GaussianBlur+morfologia KOKO 1920x1080-framelle), vaikka SEURANTA-
# tilassa kivi voi olla VAIN pienella, jatkuvuuden rajaamalla alueella
# (last_xy +- TRACK_HALF_RANGE_CM +- kiven oma sade). Lasketaan siis
# graniittimaski+saturaatio VAIN taman alueen ymparille rajatusta
# kuvasta - loppuosa framesta jataan nollaksi (kaikki koodi joka
# indeksoi maskia/saturaatiota kayttaa aina ABSOLUTTISIA framepikseli-
# koordinaatteja, joten mikaan muu ei muutu - vain rajatun alueen
# ULKOPUOLELLA olevat pisteet nayttaisivat vaarin "tyhjalta jaalta",
# joten MARGINAALIN ON OLTAVA RIITTAVAN SUURI ettei mikaan koskaan
# oikeasti naytelty piste (ristikkohaun kandidaatit, renkaan alipikseli-
# haku, kahva-tarkistus) osu tuon rajan ulkopuolelle).
#
# MARGINAALIN MITOITUS: GaussianBlur(sigma=25) tarvitsee n. 3-4*sigma
# (75-100px) puskurin reunoilta ettei kuvan reunan/rajauksen keinotekoi-
# nen "toistoreuna" (cv2:n oletus-bordertyyppi) vuoda kiven omaan
# alueeseen - MASK_ROI_MARGIN_PX=150 on reilusti tama plus kaikki muut
# pienemmat marginaalit (BODY_CONTOUR_MAX_SEARCH_DIST_PX=60,
# renkaan hakusade <~35px) yhteenlaskettuna.
# ============================================================

MASK_ROI_MARGIN_PX = 150


def _track_roi_bounds(pose, X_center, Y_center, half_range_cm, R_max_cm, H_total_cm,
                       frame_w, frame_h, margin_px=MASK_ROI_MARGIN_PX):
    """Palauttaa (x0,y0,x1,y1) - pikselirajatun alueen joka varmasti
    sisaltaa KAIKKI mahdolliset ehdokassijainnit (last_xy +-
    half_range_cm) JA kiven oman geometrisen ulottuvuuden (R_max_cm,
    H_total_cm) niilla, plus reilun turvamarginaalin."""

    reach = half_range_cm + R_max_cm
    corners_3d = []
    for zx in (X_center - reach, X_center + reach):
        for zy in (Y_center - reach, Y_center + reach):
            for zz in (0.0, H_total_cm):
                corners_3d.append([zx, zy, zz])

    u, v = k9._project_3d(pose["K"], pose["R"], pose["t"], np.array(corners_3d))

    x0 = int(math.floor(np.min(u))) - margin_px
    x1 = int(math.ceil(np.max(u))) + margin_px
    y0 = int(math.floor(np.min(v))) - margin_px
    y1 = int(math.ceil(np.max(v))) + margin_px

    x0 = max(x0, 0)
    y0 = max(y0, 0)
    x1 = min(x1, frame_w)
    y1 = min(y1, frame_h)

    return x0, y0, x1, y1


def create_granite_mask_roi(frame_u, roi):
    """Sama tulos kuin kamera9_01.py:n create_granite_mask KOKO framelle
    ROI-alueen SISALLA (rajatun kuvan reunoilta marginaalin verran -
    katso ylla) - ROI:n ulkopuolella nollia. Kutsuu MUUTTAMATONTA
    create_granite_mask-funktiota pelkastaan pienemmalla syotteella,
    ei mitaan omaa (uutta) kuvankasittelylogiikkaa."""

    x0, y0, x1, y1 = roi
    full_mask = np.zeros(frame_u.shape[:2], dtype=np.uint8)

    if x1 <= x0 or y1 <= y0:
        return full_mask

    crop = frame_u[y0:y1, x0:x1]
    full_mask[y0:y1, x0:x1] = k9.create_granite_mask(crop)

    return full_mask


def compute_sat_roi(frame_u, roi):
    """Sama kuin cv2.cvtColor(frame_u,...)[:,:,1] KOKO framelle ROI:n
    SISALLA, nollilla taytettyna ulkopuolella - katso create_granite_
    mask_roi:n kommentti samasta periaatteesta."""

    x0, y0, x1, y1 = roi
    full_sat = np.zeros(frame_u.shape[:2], dtype=np.float32)

    if x1 <= x0 or y1 <= y0:
        return full_sat

    crop = frame_u[y0:y1, x0:x1]
    full_sat[y0:y1, x0:x1] = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)[:, :, 1].astype(np.float32)

    return full_sat


# ============================================================
# HAKU + SEURANTA KOKO VIDEOLLE, CSV-TULOSTUS - SAMA TILAKONE JA
# SAMA CSV-SKEEMA KUIN kamera9_03.py:n search_and_track_stones_
# precise, vain nopeutetuilla funktioilla.
# ============================================================

def search_and_track_stones_precise_fast(video_path, calib, pose, profile, csv_path,
                                           search_x_half_width=k92.SEARCH_X_HALF_WIDTH_CM,
                                           search_y_min=k92.SEARCH_Y_MIN_CM,
                                           search_y_max=k92.SEARCH_Y_MAX_CM,
                                           search_every_n=k92.SEARCH_EVERY_N_FRAMES,
                                           progress=True):

    cap = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        fps = 25.0

    camera_matrix = calib["camera_matrix"]
    dist_coeffs = np.array([calib["best_k1"], 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    map1, map2 = _build_undistort_maps(camera_matrix, dist_coeffs, (frame_w, frame_h))

    R_max, H_total, shape_deltas = profile["R_max_cm"], profile["H_total_cm"], profile["shape_deltas"]
    local_pts_body = build_local_stone_rings(R_max, H_total, shape_deltas, n_theta=28, n_per_segment=5)
    local_pts_search = build_local_stone_rings(
        R_max, H_total, shape_deltas,
        n_theta=k92.SEARCH_HULL_N_THETA, n_per_segment=k92.SEARCH_HULL_N_PER_SEGMENT
    )

    state = "HAKU"
    stone_id = 0
    last_xy = None
    misses = 0
    n_written = 0
    n_precise = 0

    t_start = time.time()

    with open(csv_path, "w", newline="") as f:

        writer = csv.writer(f)
        writer.writerow(["frame", "timestamp_s", "stone_id", "x_m", "y_m", "tarkka",
                          "n_runkopistetta", "n_reunapistetta", "rms_px", "rengas_r_cm"])

        frame_idx = 0

        while frame_idx < n_frames:

            ok, frame = cap.read()

            if not ok:
                break

            do_this_frame = (state == "SEURANTA") or (frame_idx % search_every_n == 0)

            if not do_this_frame:
                frame_idx += 1
                continue

            frame_u = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)

            if state == "SEURANTA" and last_xy is not None:
                # SEURANTA: kivi on jatkuvuuden rajaamalla pienella
                # alueella - maski+saturaatio riittaa laskea vain sen
                # ymparilta (katso _track_roi_bounds:in kommentti).
                # HUOM: ROI:n on katettava PAHIN MAHDOLLINEN kokonais-
                # siirtyma last_xy:sta - paitsi ristikkohaun oman
                # etsinta-alueen (TRACK_HALF_RANGE_CM), MYOS refine_
                # position_joint_fast:in OMA sisainen iteraatio voi
                # viela siirtaa sijaintia jopa BOUNDARY_MAX_SHIFT_FROM_
                # APPROX_CM verran ristikkohaun tuloksesta - muuten ROI
                # voisi (aarimmaisen, joskin epatodennakoisen, framen)
                # tapauksessa leikata todellisen sijainnin pois.
                roi_half_range = k92.TRACK_HALF_RANGE_CM + k93.BOUNDARY_MAX_SHIFT_FROM_APPROX_CM
                roi = _track_roi_bounds(
                    pose, last_xy[0], last_xy[1], roi_half_range, R_max, H_total,
                    frame_w, frame_h
                )
                mask = create_granite_mask_roi(frame_u, roi)
                sat = compute_sat_roi(frame_u, roi)
            else:
                # HAKU: koko rajattu hakualue voi kattaa ison osan
                # framesta eika ajeta joka framella - ei kannata
                # rajata, lasketaan koko framelle kuten ennenkin.
                mask = k9.create_granite_mask(frame_u)
                sat = cv2.cvtColor(frame_u, cv2.COLOR_BGR2HSV)[:, :, 1].astype(np.float32)

            if state == "HAKU":

                x_center = 0.0
                y_center = (search_y_min + search_y_max) / 2.0
                y_half = (search_y_max - search_y_min) / 2.0

                (bx, by), score = locate_by_grid_search_fast(
                    local_pts_search, mask, x_center, search_x_half_width, y_center, y_half,
                    k92.SEARCH_COARSE_STEP_CM, k92.SEARCH_FINE_STEP_CM, pose
                )

                if score >= k92.SEARCH_SCORE_THRESHOLD:
                    stone_id += 1
                    misses = 0
                    state = "SEURANTA"

                    refined = refine_position_joint_fast(mask, sat, local_pts_body, pose, profile, bx, by)
                    last_xy = (refined["X_cm"], refined["Y_cm"])

                    if progress:
                        tarkka_str = f"KYLLA (rms={refined['rms_px']:.2f}px)" if refined["tarkka"] else "EI (karkea)"
                        print(f"[frame {frame_idx}] LOYTYI kivi {stone_id}: "
                              f"({last_xy[0]:.1f}, {last_xy[1]:.1f}) cm, peitto={score:.2f}, "
                              f"alipikselitarkkuus={tarkka_str} (runko={refined['n_body']}, "
                              f"rengas={refined['n_ring']})")

                    if refined["tarkka"]:
                        n_precise += 1

                    timestamp = frame_idx / fps
                    writer.writerow([
                        frame_idx, f"{timestamp:.3f}", stone_id,
                        f"{last_xy[0] / 100.0:.5f}", f"{last_xy[1] / 100.0:.5f}",
                        int(refined["tarkka"]), refined["n_body"], refined["n_ring"],
                        f"{refined['rms_px']:.3f}" if refined["rms_px"] is not None else "",
                        f"{refined['ring_radius_cm']:.3f}" if refined["ring_radius_cm"] is not None else "",
                    ])
                    n_written += 1

            else:  # SEURANTA

                (bx, by), score = locate_by_grid_search_fast(
                    local_pts_search, mask, last_xy[0], k92.TRACK_HALF_RANGE_CM,
                    last_xy[1], k92.TRACK_HALF_RANGE_CM,
                    k92.TRACK_COARSE_STEP_CM, k92.TRACK_FINE_STEP_CM, pose
                )

                if score >= k92.TRACK_SCORE_THRESHOLD:
                    misses = 0

                    refined = refine_position_joint_fast(mask, sat, local_pts_body, pose, profile, bx, by)
                    last_xy = (refined["X_cm"], refined["Y_cm"])

                    if refined["tarkka"]:
                        n_precise += 1

                    timestamp = frame_idx / fps
                    writer.writerow([
                        frame_idx, f"{timestamp:.3f}", stone_id,
                        f"{last_xy[0] / 100.0:.5f}", f"{last_xy[1] / 100.0:.5f}",
                        int(refined["tarkka"]), refined["n_body"], refined["n_ring"],
                        f"{refined['rms_px']:.3f}" if refined["rms_px"] is not None else "",
                        f"{refined['ring_radius_cm']:.3f}" if refined["ring_radius_cm"] is not None else "",
                    ])
                    n_written += 1
                else:
                    misses += 1

                    if misses >= k92.TRACK_LOST_MAX_MISSES:
                        if progress:
                            print(f"[frame {frame_idx}] Kivi {stone_id} kadotettu "
                                  f"({misses} huonoa framea), palataan hakuun.")
                        state = "HAKU"
                        last_xy = None

            if progress and frame_idx % 100 == 0:
                elapsed = time.time() - t_start
                print(f"  ... frame {frame_idx}/{n_frames} (tila={state}, "
                      f"{elapsed:.0f}s kulunut, {n_written} rivia, "
                      f"{n_precise} alipikselitarkkaa)")

            frame_idx += 1

    cap.release()

    if progress:
        print(f"Valmis: {n_written} rivia ({n_precise} alipikselitarkkaa) "
              f"tiedostoon {csv_path} ({time.time() - t_start:.0f}s)")

    return {"n_written": n_written, "n_precise": n_precise, "n_stones": stone_id}


# ============================================================
# 7) VIDEON KIVIPROFIILIN SOVITUS (build_stone_profile_from_video) -
#    ALKUUN KERRAN AJETTAVA KALIBROINTIVAIHE, EI PER-FRAME
#
# KAYTTAJAN HAVAINTO: tama vaihe (kamera9_01.py:n track_stone_in_video,
# jota build_stone_profile_from_video kutsuu) kesti n. 3 minuuttia.
# SYY LOYTYI PROFILOIMALLA: track_stone_in_video hakee jokaisen framen
# ERIKSEEN video_capture.set(CAP_PROP_POS_FRAMES, idx):lla ENNEN
# lukua - VAIKKA idx KASVAA/VAHENEE AINA TASAN YHDELLA per kutsu (siis
# JARJESTYKSESSA seuraava frame - taysin turha hakea/"seek"). Mitattu
# TASSA VIDEOSSA: 200 framen sekvenssiluku 0.5s, mutta 200 framen
# seek+luku 16.2s - SEEK ON SIIS N. 32x HITAAMPI kuin sekvenssiluku
# (video on H.264/H.265-tyyppisesti pakattu - "seek" joutuu aina
# dekoodaamaan lahimmasta avainkuvasta eteenpain kohdeframeen asti).
#
# KORJAUS: LUETAAN VIDEO LAPI TASAN KERRAN JARJESTYKSESSA (ei koskaan
# cap.set:ia), lasketaan+valimuistetaan jokaisen framen kandidaatit
# (_candidates_in_frame_fast, sama funktio kuin kamera9_01.py:n
# _candidates_in_frame mutta ottaa jo-vaantokorjatun framen valmiina
# eika vaanna sita itse uudelleen joka kutsulla - katso undistort-
# kommentti ylempaa samasta periaatteesta). SITTEN ajetaan TASMALLEEN
# SAMA jatkuvuushaku (track_direction, sama koodi kuin kamera9_01.py:ssa
# rivi riviltä) valimuistetun listan paalla eteen- ja taaksepain -
# EI mitaan I/O:ta enaa talla kierroksella. Lopputulos on TASMALLEEN
# sama full_track kuin alkuperaisella (sama kandidaattilogiikka, sama
# jatkuvuusehto, sama jarjestys) - vain paljon nopeammin saatu.
# ============================================================

def _candidates_in_frame_fast(frame_u, pose, H_final, min_area, min_fill_ratio, min_aspect_ratio):
    """Sama kuin kamera9_01.py:n _candidates_in_frame, mutta ottaa
    valmiiksi vaantokorjatun framen (ei vaanna sita itse uudelleen)."""

    stones = k9.find_stone_candidates(
        frame_u, H_final, min_area=min_area,
        min_fill_ratio=min_fill_ratio, min_aspect_ratio=min_aspect_ratio
    )

    out = []

    for stone in stones:
        (cx, cy), _, _ = stone["ellipse"]
        X0, Y0 = k9._stone_ground_position_z0(pose, cx, cy)
        out.append({"ellipse": stone["ellipse"], "contour": stone["contour"], "pos_cm": (X0, Y0)})

    return out


def track_stone_in_video_fast(video_path, calib, pose, seed_frame_idx, seed_pos_cm,
                               max_jump_cm=None, max_misses=None, min_area=None,
                               min_fill_ratio=None, min_aspect_ratio=None):
    """Sama tulos kuin kamera9_01.py:n track_stone_in_video - katso
    taman osion alkupaan kommentti mika on eri (vain I/O-jarjestys,
    ei algoritmi/kynnysarvot)."""

    max_jump_cm = k9.STONE_TRACK_MAX_JUMP_CM if max_jump_cm is None else max_jump_cm
    max_misses = k9.STONE_TRACK_MAX_MISSES if max_misses is None else max_misses
    min_area = k9.STONE_TRACK_MIN_AREA if min_area is None else min_area
    min_fill_ratio = k9.STONE_TRACK_MIN_FILL_RATIO if min_fill_ratio is None else min_fill_ratio
    min_aspect_ratio = k9.STONE_TRACK_MIN_ASPECT_RATIO if min_aspect_ratio is None else min_aspect_ratio

    H_final = calib["H_final"]
    camera_matrix = calib["camera_matrix"]
    dist_coeffs = np.array([calib["best_k1"], 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    cap = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    map1, map2 = _build_undistort_maps(camera_matrix, dist_coeffs, (frame_w, frame_h))

    candidates_cache = [None] * n_frames
    idx = 0

    while idx < n_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frame_u = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)
        candidates_cache[idx] = _candidates_in_frame_fast(
            frame_u, pose, H_final, min_area, min_fill_ratio, min_aspect_ratio
        )
        idx += 1

    cap.release()

    def candidates_at(i):
        if 0 <= i < n_frames and candidates_cache[i] is not None:
            return candidates_cache[i]
        return []

    seed_cands = candidates_at(seed_frame_idx)
    if not seed_cands:
        raise RuntimeError(f"Ei kandidaatteja siemen-framessa {seed_frame_idx}.")

    seed = min(seed_cands, key=lambda c: math.hypot(
        c["pos_cm"][0] - seed_pos_cm[0], c["pos_cm"][1] - seed_pos_cm[1]
    ))
    seed["frame_idx"] = seed_frame_idx

    def track_direction(step):
        track = []
        last_pos = seed["pos_cm"]
        misses = 0
        idx = seed_frame_idx + step

        while 0 <= idx < n_frames and misses < max_misses:

            cands = candidates_at(idx)

            if cands:
                best = min(cands, key=lambda c: math.hypot(
                    c["pos_cm"][0] - last_pos[0], c["pos_cm"][1] - last_pos[1]
                ))
                d = math.hypot(best["pos_cm"][0] - last_pos[0], best["pos_cm"][1] - last_pos[1])

                if d <= max_jump_cm:
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


def build_stone_profile_from_video_fast(calib, pose, video_path, seed_frame_idx, seed_pos_cm,
                                         extra_stones=None, n_video_samples=25,
                                         front_view_path=None):
    """Sama tulos kuin kamera9_01.py:n build_stone_profile_from_video,
    kayttaen track_stone_in_video_fast:ia (katso talla osion alkupaan
    kommentti nopeutuksesta)."""

    track = track_stone_in_video_fast(video_path, calib, pose, seed_frame_idx, seed_pos_cm)

    if len(track) < 2:
        raise RuntimeError(f"Video-seuranta loysi vain {len(track)} havaintoa.")

    idxs = sorted(set(np.linspace(0, len(track) - 1, n_video_samples).astype(int).tolist()))
    video_stones = [{"ellipse": track[i]["ellipse"], "contour": track[i]["contour"]} for i in idxs]

    all_stones = (extra_stones or []) + video_stones
    profile = k9.fit_stone_profile(pose, all_stones)

    if front_view_path is not None:
        k9.render_profile_front_view(profile, front_view_path)

    return {"profile": profile, "track": track}


# ============================================================
# PAAOHJELMA (katso kamera9_02.py:n kommentti siemen-arvoista)
# ============================================================

def main():

    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()

    calib_filename = filedialog.askopenfilename(
        title="Valitse kalibrointikuva (jolla H_final/poosi ratkaistaan)",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
            ("All files", "*.*"),
        ]
    )

    if not calib_filename:
        print("Kalibrointikuvaa ei valittu.")
        root.destroy()
        return

    video_filename = filedialog.askopenfilename(
        title="Valitse video (kiven paikannukseen/seurantaan)",
        filetypes=[
            ("Video files", "*.mp4 *.avi *.mov *.mkv"),
            ("All files", "*.*"),
        ]
    )

    root.destroy()

    if not video_filename:
        print("Videota ei valittu.")
        return

    print("Kalibroidaan kamera (kamera8_01.py:n putki)...")
    calib = k9.calibrate_camera_from_image(calib_filename)
    pose = k9.build_pose_from_calibration(calib)
    print(f"  fokaalivali f = {pose['K'][0, 0]:.1f} px, "
          f"kameran sijainti (cm): {np.round(pose['camera_position_cm'], 1)}")

    kivilla_stones = k9.find_stone_candidates(calib["frame_undistorted"], calib["H_final"])
    print(f"  loydettiin {len(kivilla_stones)} paikallaan olevaa kivea kalibrointikuvasta")

    print(f"Sovitetaan 3D-profiili (kalibrointikuvan kivet + video, "
          f"siemen frame={k92.DEFAULT_SEED_FRAME_IDX})...")
    result = build_stone_profile_from_video_fast(
        calib, pose, video_filename,
        seed_frame_idx=k92.DEFAULT_SEED_FRAME_IDX, seed_pos_cm=k92.DEFAULT_SEED_POS_CM,
        extra_stones=kivilla_stones, n_video_samples=25,
    )
    profile = result["profile"]
    print(f"  R_max={profile['R_max_cm']:.2f} cm, H_total={profile['H_total_cm']:.2f} cm, "
          f"RMS={profile['residual_rms_px']:.2f} px")

    print("Haetaan ja seurataan kivea/kivia videolta (nopeutettu, alipikselitarkka Y)...")

    csv_path = "kivien_sijainnit_tarkka_nopea.csv"
    search_and_track_stones_precise_fast(video_filename, calib, pose, profile, csv_path)


if __name__ == "__main__":
    main()
