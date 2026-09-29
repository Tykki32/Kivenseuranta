"""Kameran kalibrointi: kameramalli, homografian geometrinen sovitus, koko 3D-pose ja kalibrointiputki moodikuvasta."""

import math
import cv2
import numpy as np
from config import (
    BLUE_INNER_RADIUS_CM,
    BLUE_OUTER_RADIUS_CM,
    FAR_HOGLINE_Y_CM,
    FAR_HOUSE_Y_CM,
    HOMOGRAPHY_REFINE_MAX_ITERATIONS,
    HOMOGRAPHY_REFINE_MIN_RELATIVE_IMPROVEMENT,
    HOUSE_CROP_HALF_HEIGHT_CM,
    K1_MIN_MAGNITUDE,
    K1_MIN_RELATIVE_IMPROVEMENT,
    K1_SEARCH_RANGE,
    K1_SEARCH_REFINE_ROUNDS,
    K1_SEARCH_REFINE_SHRINK,
    K1_SEARCH_STEPS,
    NEAR_BLUE_MIN_AREA,
    NEAR_BLUE_MIN_RATIO,
    NEAR_BLUE_MIN_SIZE_RATIO,
    NEAR_HOGLINE_Y_CM,
    NEAR_HOUSE_Y_CM,
    NEAR_RED_MIN_AREA,
    NEAR_RED_MIN_RATIO,
    NEAR_RED_MIN_SIZE_RATIO,
    NEAR_TRUST_FLOOR_CM,
    PIXELS_PER_CM,
    RED_INNER_RADIUS_CM,
    RED_OUTER_RADIUS_CM,
)
from geometry import (
    _apply_h,
    _decompose_planar_homography,
    _ensure_camera_above_ice,
    _h_from_params,
    _homography_rms,
    _levenberg_marquardt,
    _params_from_h,
    _physical_to_frame_homography,
    _project_3d,
    _robust_scale_cm,
    build_camera_matrix,
    frame_points_from_topdown,
    line_ellipse_intersections,
    output_px_to_physical,
    physical_to_output_px,
    project_point,
    ray_plane_intersection,
    robust_circle_fit_with_inliers,
    undistort_points_px,
)
from house_detection import (
    build_far_house_center_seed,
    candidate_is_better,
    compute_crop_row_range,
    create_blue_mask,
    create_red_mask,
    create_topdown_score_maps,
    crop_house_view,
    detect_hogline_points,
    detect_line_segments,
    expected_house_center_in_crop,
    far_ring_points_color_based,
    find_house_pair,
    find_ring_edge_points,
    hogline_points_gray_threshold,
    house_quality_str,
    measure_house_quality,
    robust_line_angle_from_points,
    select_t_and_centerline,
)


def build_image_directions(t_pair, centerline_pair):

    v_cl = centerline_pair[1]["mid"] - centerline_pair[0]["mid"]

    n = np.linalg.norm(v_cl)

    if n < 1e-8:
        raise RuntimeError("Virheellinen keskilinjan suunta.")

    v_cl /= n

    image_right = np.array([1.0, 0.0], dtype=np.float64)

    if np.dot(v_cl, image_right) < 0:
        v_cl = -v_cl

    v_t = np.array([-v_cl[1], v_cl[0]])
    v_t /= np.linalg.norm(v_t)

    original_t = t_pair[1]["mid"] - t_pair[0]["mid"]

    if np.dot(v_t, original_t) < 0:
        v_t = -v_t

    return v_t, v_cl


def build_calibration_observations(blue_outer, blue_inner, red_outer, red_inner):

    observations = []

    if blue_outer is not None:
        observations.append({
            "name": "BLUE OUTER", "ellipse": blue_outer,
            "radius_cm": BLUE_OUTER_RADIUS_CM
        })

    if blue_inner is not None:
        observations.append({
            "name": "BLUE INNER", "ellipse": blue_inner,
            "radius_cm": BLUE_INNER_RADIUS_CM
        })

    if red_outer is not None:
        observations.append({
            "name": "RED OUTER", "ellipse": red_outer,
            "radius_cm": RED_OUTER_RADIUS_CM
        })

    if red_inner is not None:
        observations.append({
            "name": "RED INNER", "ellipse": red_inner,
            "radius_cm": RED_INNER_RADIUS_CM
        })

    return observations


def calculate_ellipse_calibration(ellipse, physical_radius_cm, focal_length_px):

    (cx, cy), (w, h), angle = ellipse

    major_px = max(w, h)
    minor_px = min(w, h)

    projected_radius_px = major_px / 2.0

    axis_ratio = np.clip(minor_px / major_px, 1e-6, 1.0)

    theta = math.acos(axis_ratio)

    distance_to_plane = (
        focal_length_px * physical_radius_cm / projected_radius_px
    )

    return {
        "center": np.array([cx, cy], dtype=np.float64),
        "theta": theta,
        "distance_to_plane": distance_to_plane,
        "name": None,
    }


def build_camera_model(observations, image_width, image_height):

    if not observations:
        raise RuntimeError("Kalibrointiympyroita ei ole saatavilla.")

    f = float(max(image_width, image_height))

    calibration_data = []

    for observation in observations:

        data = calculate_ellipse_calibration(
            observation["ellipse"], observation["radius_cm"], f
        )

        data["name"] = observation["name"]

        calibration_data.append(data)

    theta = float(np.median([d["theta"] for d in calibration_data]))

    distance_to_plane = float(
        np.median([d["distance_to_plane"] for d in calibration_data])
    )

    camera_height = distance_to_plane * math.cos(theta)

    centers = np.array([d["center"] for d in calibration_data])
    common_center = np.median(centers, axis=0)

    return {
        "f": f,
        "theta": theta,
        "theta_deg": math.degrees(theta),
        "distance_to_plane": distance_to_plane,
        "camera_height": camera_height,
        "center": common_center,
        "calibration_data": calibration_data,
        "observation_count": len(calibration_data),
    }


def near_house_correspondences(
    t_line, centerline, house_center,
    blue_outer, blue_inner, red_outer, red_inner,
    dir_lateral, dir_forward
):

    image_pts = []
    physical_pts = []
    labels = []

    image_pts.append(np.asarray(house_center, dtype=np.float64))
    physical_pts.append((0.0, NEAR_HOUSE_Y_CM))
    labels.append("NEAR_CENTER")

    circle_specs = [
        ("BLUE_OUTER", blue_outer, BLUE_OUTER_RADIUS_CM),
        ("BLUE_INNER", blue_inner, BLUE_INNER_RADIUS_CM),
        ("RED_OUTER", red_outer, RED_OUTER_RADIUS_CM),
        ("RED_INNER", red_inner, RED_INNER_RADIUS_CM),
    ]

    for name, ellipse, radius in circle_specs:

        if ellipse is None:
            raise RuntimeError(
                f"Ympyraa {name} ei loytynyt - 17 pisteen homografia "
                f"vaatii kaikki 4 ympyraa."
            )

        t_pts = line_ellipse_intersections(t_line, ellipse)

        if len(t_pts) != 2:
            raise RuntimeError(
                f"T-linja ei leikannut ellipsia {name} kahdessa pisteessa."
            )

        for p in t_pts:

            d = float(np.dot(
                np.asarray(p) - np.asarray(house_center), dir_lateral
            ))

            x_phys = radius if d > 0 else -radius

            image_pts.append(np.asarray(p, dtype=np.float64))
            physical_pts.append((x_phys, NEAR_HOUSE_Y_CM))
            labels.append(f"{name}_T_{'POS' if d > 0 else 'NEG'}")

        c_pts = line_ellipse_intersections(centerline, ellipse)

        if len(c_pts) != 2:
            raise RuntimeError(
                f"Keskilinja ei leikannut ellipsia {name} kahdessa pisteessa."
            )

        for p in c_pts:

            d = float(np.dot(
                np.asarray(p) - np.asarray(house_center), dir_forward
            ))

            y_phys = (
                NEAR_HOUSE_Y_CM + radius if d > 0 else NEAR_HOUSE_Y_CM - radius
            )

            image_pts.append(np.asarray(p, dtype=np.float64))
            physical_pts.append((0.0, y_phys))
            labels.append(f"{name}_CL_{'FAR' if d > 0 else 'NEAR'}")

    return image_pts, physical_pts, labels


def estimate_radial_distortion_k1(img_pts, phys_pts, camera_matrix):
    """
    Hakee yhden parametrin (k1) sateittaisen vaaristyman coarse ->
    fine -> ultra fine -periaatteella (sama tyyli kuin kaukaisen
    pesan haussa muualla koodissa): kokeillaan eri k1-arvoja, ja
    jokaiselle lasketaan RMS-uudelleenprojisointivirhe kun pisteet
    ensin oikaistaan (undistort) ja niihin sovitetaan homografia.
    Paras (pienin RMS) k1 voittaa; hakualuetta kavennetaan jokaisen
    kierroksen jalkeen loydetyn parhaan arvon ymparille.

    Palauttaa (paras_k1, paras_rms, rms_ilman_korjausta).
    """

    dst_pts = physical_to_output_px(phys_pts).astype(np.float64)
    src_pts = np.asarray(img_pts, dtype=np.float64)

    def rms_for_k1(k1):
        undistorted = undistort_points_px(src_pts, camera_matrix, k1)
        return _homography_rms(undistorted, dst_pts)

    baseline_rms = rms_for_k1(0.0)

    best_k1 = 0.0
    best_rms = baseline_rms
    center = 0.0
    half_range = K1_SEARCH_RANGE

    for _ in range(K1_SEARCH_REFINE_ROUNDS):

        candidates = np.linspace(
            center - half_range, center + half_range, K1_SEARCH_STEPS
        )

        for k1 in candidates:

            rms = rms_for_k1(float(k1))

            if rms < best_rms:
                best_rms = rms
                best_k1 = float(k1)

        center = best_k1
        half_range /= K1_SEARCH_REFINE_SHRINK

    return best_k1, best_rms, baseline_rms


def _far_ring_distance(H, far_pts, far_radius):
    if not len(far_pts):
        return np.zeros(0)
    proj_cm = output_px_to_physical(_apply_h(H, far_pts))
    return np.hypot(proj_cm[:, 0], proj_cm[:, 1] - FAR_HOUSE_Y_CM) - far_radius


def _hogline_distance(H, hog_pts, hog_y):
    if not len(hog_pts):
        return np.zeros(0)
    proj_cm = output_px_to_physical(_apply_h(H, hog_pts))
    return proj_cm[:, 1] - hog_y


def _geometric_residuals(params, near_pts, near_phys,
                          far_pts, far_radius, far_weight,
                          hog_pts, hog_y, hog_weight):

    H = _h_from_params(params)
    parts = []

    if len(near_pts):
        proj_cm = output_px_to_physical(_apply_h(H, near_pts))
        parts.append((proj_cm - near_phys).ravel())

    if len(far_pts):
        parts.append(_far_ring_distance(H, far_pts, far_radius) * far_weight)

    if len(hog_pts):
        parts.append(_hogline_distance(H, hog_pts, hog_y) * hog_weight)

    return np.concatenate(parts) if parts else np.zeros(0)


def solve_homography_geometric(H_init, near_pts, near_phys,
                                far_pts, far_radius, far_weight,
                                hog_pts, hog_y, hog_weight):
    """
    Ratkaisee koko homografian (8 vapausastetta) YHDELLA epalineaari-
    sella sovituksella: lahemman pesan 17 pistetta ovat tavallisia
    pistekorrespondensseja, kaukaisen pesan rengaspisteet YMPYRA-
    RAJOITTEITA (ei vaadi T-linjan suuntaa) ja hogline-pisteet VIIVA-
    RAJOITTEITA (ei vaadi X-kohdetta). Katso taman tiedoston alkupaan
    kommentti periaatteesta.

    far_weight/hog_weight painottavat kunkin ryhman residuaalit NIIDEN
    OMAN, JUURI ENNEN TATA KUTSUA arvioidun kohinatason mukaan (katso
    _robust_scale_cm ja refine_geometric_homography) - ei kasin
    viritettya vakiota, vaan tavanomainen painotetun pienimman nelion
    (inverse-variance) periaate.
    """

    residual_fn = lambda p: _geometric_residuals(
        p, near_pts, near_phys, far_pts, far_radius, far_weight,
        hog_pts, hog_y, hog_weight
    )

    params = _levenberg_marquardt(residual_fn, _params_from_h(H_init))

    return _h_from_params(params)


def far_house_ring_points_in_frame(topdown_raw, H_current, max_points_per_ring=30):
    """
    Tunnistaa kaukaisen pesan sinisen ja punaisen ULKOKEHAN reuna-
    pisteet nykyisesta top-down-kuvasta (sama sateittainen reunanhaku +
    robusti ympyransovitus kuin detect_far_house_concentric_circles
    kayttaa), ja siirtaa ne (frame_points_from_topdown) takaisin
    oikaistun kuvan pikselikoordinaatistoon.

    Pisteet EIVAT ole korrespondensseja mihinkaan yksittaiseen
    kohdepisteeseen - niiden fyysinen rajoite on "etaisyys 0 tunnettuun
    ympyraan" (katso solve_homography_geometric).

    max_points_per_ring rajoittaa pistemaaran samaan suuruusluokkaan
    kuin lahemman pesan 17 pistetta - muuten satojen kulmanaytteiden
    (find_ring_edge_points, num_angles=720) maara hallitsisi koko
    sovitusta pelkalla lukumaarallaan, vaikka ne kuvaavat vain KAHTA
    ympyraa (5 vapausastetta kumpikin).

    Palauttaa (points, radii) - Nx2- ja N-pituiset taulukot.
    """

    far_row_top, _ = compute_crop_row_range(
        topdown_raw.shape[0], FAR_HOUSE_Y_CM, HOUSE_CROP_HALF_HEIGHT_CM
    )
    far_view = crop_house_view(topdown_raw, FAR_HOUSE_Y_CM, HOUSE_CROP_HALF_HEIGHT_CM)
    far_expected_center = expected_house_center_in_crop(
        topdown_raw.shape[0], FAR_HOUSE_Y_CM, HOUSE_CROP_HALF_HEIGHT_CM
    )

    blue_score, red_score = create_topdown_score_maps(far_view)

    blue_points = find_ring_edge_points(
        blue_score, far_expected_center, BLUE_OUTER_RADIUS_CM * PIXELS_PER_CM
    )
    red_points = find_ring_edge_points(
        red_score, far_expected_center, RED_OUTER_RADIUS_CM * PIXELS_PER_CM
    )

    topdown_pts = []
    radii = []

    for edge_points, radius_cm in (
        (blue_points, BLUE_OUTER_RADIUS_CM), (red_points, RED_OUTER_RADIUS_CM)
    ):

        fit = robust_circle_fit_with_inliers(edge_points)

        if fit is None:
            continue

        _, _, _, inliers = fit

        if len(inliers) > max_points_per_ring:
            idx = np.linspace(0, len(inliers) - 1, max_points_per_ring).astype(int)
            inliers = inliers[idx]

        for x, y in inliers:
            topdown_pts.append((x, y + far_row_top))
            radii.append(radius_cm)

    if not topdown_pts:
        return np.zeros((0, 2)), np.zeros(0)

    points = frame_points_from_topdown(np.array(topdown_pts, dtype=np.float64), H_current)

    return points, np.array(radii, dtype=np.float64)


def hogline_points_in_frame(topdown_raw, hogline_y_cm, H_current):
    """
    detect_hogline_points (validoitu majority-vote-kontaminaatiokorjaus,
    katso funktion kommentti) + siirto oikaistun kuvan koordinaatistoon.

    Palauttaa (points, weights) - weights on kunkin pisteen havaittu
    "strength" NORMALISOITUNA (keskiarvo 1.0) - suhteellinen luottamus
    hogline-pisteiden VALILLA sailyy, mutta absoluuttinen taso ei paase
    vaikuttamaan piste-/ympyrarajoitteiden painotukseen nahden (katso
    taman tiedoston alkupaan kommentti fyysisesta painotuksesta).
    """

    raw_points = detect_hogline_points(topdown_raw, hogline_y_cm)

    if not raw_points:
        return np.zeros((0, 2)), np.zeros(0)

    topdown_pts = np.array([(p[0], p[1]) for p in raw_points], dtype=np.float64)
    strengths = np.array([p[2] for p in raw_points], dtype=np.float64)
    weights = strengths / max(float(np.mean(strengths)), 1e-6)

    points = frame_points_from_topdown(topdown_pts, H_current)

    return points, weights


def _search_focal_length(obj_pts, img_pts, camera_matrix0, rvec0, t0):
    """
    Hakee fokaalivalin (f) COARSE -> FINE -periaatteella (sama tyyli
    kuin kamera8_01.py:n kaukaisen pesan haku), kayttaen VAIN lahemman
    pesan 17 (tarkasti tunnetun) pistetta - nama ovat tiiviisti
    ryhmittyneet ja siksi VAKAA pohja f:n haulle (toisin kuin koko
    kaukaisenkin datan sisaltava yhteissovitus, joka testattu olevan
    altis f:n/etaisyyden keskinaiselle epamaaraytyneisyydelle jos f
    saa muuttua VAPAASTI samassa sovituksessa poosin kanssa).

    HUOM (kriittinen cv2-sudenkuoppa): cv2.solvePnP KIRJOITTAA
    rvec/tvec-parametrit PAIKALLEEN kun useExtrinsicGuess=True JA
    annettu taulukko on jo olemassa - .copy() on siis PAKOLLINEN
    jokaisessa kutsussa, tai alkuarvaus turmeltuu hiljaisesti jo
    ensimmaisen hakukierroksen jalkeen (loydetty testatessa: ilman
    .copy():a koko haku hajosi mielettomiin f-arvoihin).
    """

    def error_for_f(f_try):

        K_try = np.array([
            [f_try, 0.0, camera_matrix0[0, 2]],
            [0.0, f_try, camera_matrix0[1, 2]],
            [0.0, 0.0, 1.0],
        ])

        ok, rvec, tvec = cv2.solvePnP(
            obj_pts, img_pts, K_try, np.zeros(5),
            rvec0.copy(), t0.reshape(3, 1).copy(),
            useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE
        )

        proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K_try, np.zeros(5))
        err = float(np.linalg.norm(proj.reshape(-1, 2) - img_pts, axis=1).mean())

        return err, rvec, tvec

    center_f = float(camera_matrix0[0, 0])
    half_range = center_f

    for _ in range(4):

        candidates = np.linspace(max(center_f - half_range, 100.0), center_f + half_range, 25)
        best_err, best_f = min((error_for_f(f)[0], f) for f in candidates)
        center_f = best_f
        half_range /= 5.0

    err, rvec, tvec = error_for_f(center_f)

    return center_f, rvec, tvec, err


def fit_full_camera_pose(calib, near_pts_frame, near_phys, far_pts, far_radius, far_weight,
                          hog_pts, hog_line_idx, hog_weight):
    """
    Ratkaisee TAYDEN 3D-kameramallin (K, R, t) kamera8_01.py:n jo
    konvergoituneesta H_final:sta: (1) taso-homografian purku
    alkuarvaukseksi, (2) fokaalivalin haku lahemman pesan datalla,
    (3) poosin (R,t) hienosaato PAINOTETULLA pienimman nelion
    sovituksella SAMOILLA rajoitteilla (piste/ympyra/viiva) kuin
    kamera8_01.py:n geometrinen H-sovitus kaytti - katso taman
    tiedoston alkupaan kommentti.

    far_pts/far_radius/far_weight/hog_pts/hog_line_idx/hog_weight:
    katso kamera8_01.py:n far_house_ring_points_in_frame/
    hogline_points_in_frame - sama data, uudelleenkaytettyna.

    Palauttaa dictin: K, R, t, camera_position_cm, near_reproj_err_px.
    """

    camera_matrix0 = calib["camera_matrix"]
    H_plane_to_frame = _physical_to_frame_homography(calib)

    R0, t0 = _decompose_planar_homography(H_plane_to_frame, camera_matrix0)
    R0, t0 = _ensure_camera_above_ice(R0, t0)
    rvec0, _ = cv2.Rodrigues(R0)

    obj_pts = np.column_stack([near_phys, np.zeros(len(near_phys))])

    f, rvec1, tvec1, near_err_f = _search_focal_length(obj_pts, near_pts_frame, camera_matrix0, rvec0, t0)

    K = np.array([[f, 0.0, camera_matrix0[0, 2]], [0.0, f, camera_matrix0[1, 2]], [0.0, 0.0, 1.0]])

    def residuals(params):

        R, _ = cv2.Rodrigues(params[0:3])
        t = params[3:6]

        parts = []

        u, v = _project_3d(K, R, t, obj_pts)
        parts.append((np.column_stack([u, v]) - near_pts_frame).ravel())

        if len(far_pts):
            X, Y = ray_plane_intersection(K, R, t, far_pts[:, 0], far_pts[:, 1], 0.0)
            dist = np.hypot(X, Y - FAR_HOUSE_Y_CM)
            parts.append((dist - far_radius) * far_weight)

        if len(hog_pts):
            X, Y = ray_plane_intersection(K, R, t, hog_pts[:, 0], hog_pts[:, 1], 0.0)
            target = np.where(hog_line_idx == 0, NEAR_HOGLINE_Y_CM, FAR_HOGLINE_Y_CM)
            parts.append((Y - target) * hog_weight)

        return np.concatenate(parts)

    params0 = np.concatenate([rvec1.flatten(), tvec1.flatten()])
    params_final = _levenberg_marquardt(residuals, params0, max_iterations=100)

    R, _ = cv2.Rodrigues(params_final[0:3])
    t = params_final[3:6]
    R, t = _ensure_camera_above_ice(R, t)

    u, v = _project_3d(K, R, t, obj_pts)
    near_err = np.linalg.norm(np.column_stack([u, v]) - near_pts_frame, axis=1)

    return {
        "K": K,
        "R": R,
        "t": t,
        "camera_position_cm": -R.T @ t,
        "near_reproj_err_px": {"mean": float(near_err.mean()), "max": float(near_err.max())},
    }


def build_pose_from_calibration(calib):
    """
    Mukavuuskutsu: poimii kaukaisen renkaan/hoglinen rajoitteet
    kamera8_01.py:n JO konvergoituneesta H_final:sta (samat funktiot,
    sama painotusperiaate kuin kamera8_01.py:n oma geometrinen
    silmukka - katso sen kommentti) ja kutsuu fit_full_camera_pose:a.
    """

    frame_undistorted = calib["frame_undistorted"]
    H_final = calib["H_final"]
    topdown_raw = cv2.warpPerspective(frame_undistorted, H_final, (calib["output_w"], calib["output_h"]))

    far_pts, far_radius = far_house_ring_points_in_frame(topdown_raw, H_final)
    near_hog, near_hog_w = hogline_points_in_frame(topdown_raw, NEAR_HOGLINE_Y_CM, H_final)
    far_hog, far_hog_w = hogline_points_in_frame(topdown_raw, FAR_HOGLINE_Y_CM, H_final)

    hog_pts = np.concatenate([near_hog, far_hog]) if (len(near_hog) or len(far_hog)) else np.zeros((0, 2))
    hog_line_idx = np.concatenate([
        np.zeros(len(near_hog), dtype=np.int64), np.ones(len(far_hog), dtype=np.int64)
    ])
    hog_strength = np.concatenate([near_hog_w, far_hog_w])

    far_resid_now = _far_ring_distance(H_final, far_pts, far_radius)
    far_weight = 1.0 / _robust_scale_cm(far_resid_now, NEAR_TRUST_FLOOR_CM)

    # Hoglinen kiintea, taysi paino - katso kamera8_01.py:n kommentti
    # (l' = H^-T l pitaa maarata kiertoa, ei saa vaientua).
    hog_weight = hog_strength / NEAR_TRUST_FLOOR_CM

    return fit_full_camera_pose(
        calib, calib["near_pts_frame"], calib["near_phys"],
        far_pts, far_radius, far_weight, hog_pts, hog_line_idx, hog_weight
    )


def refine_geometric_homography_color(frame_undistorted, H_init, near_pts, near_phys,
                                       output_w, output_h,
                                       max_iterations=None, min_relative_improvement=None):
    """Sama iteratiivinen konvergenssiperiaate kuin refine_geometric_
    homography (katso sen kommentti), mutta kaukaisen renkaan ja
    hoglinien tunnistus jokaisella kierroksella tehdaan tama tiedoston
    varipohjaisilla/harmaasavypohjaisilla funktioilla (far_ring_points_
    color_based, hogline_points_gray_threshold) k8:n omien sijaan.
    Palauttaa dictin: H_final, topdown_raw, rms, quality, iterations."""

    if max_iterations is None:
        max_iterations = HOMOGRAPHY_REFINE_MAX_ITERATIONS
    if min_relative_improvement is None:
        min_relative_improvement = HOMOGRAPHY_REFINE_MIN_RELATIVE_IMPROVEMENT

    STALL_PATIENCE = 2
    H_current = H_init
    best = None
    best_rms = float("inf")
    stall_count = 0

    for iteration in range(1, max_iterations + 1):

        topdown_raw = cv2.warpPerspective(frame_undistorted, H_current, (output_w, output_h))

        far_pts, far_radius = far_ring_points_color_based(topdown_raw, H_current)
        near_hog_pts, near_hog_w = hogline_points_gray_threshold(topdown_raw, NEAR_HOGLINE_Y_CM, H_current)
        far_hog_pts, far_hog_w = hogline_points_gray_threshold(topdown_raw, FAR_HOGLINE_Y_CM, H_current)

        near_angle_now, near_conf_now = robust_line_angle_from_points(
            [(p[0], p[1], w) for p, w in zip(near_hog_pts, near_hog_w)]
        ) if len(near_hog_pts) else (0.0, 0.0)
        far_angle_now, far_conf_now = robust_line_angle_from_points(
            [(p[0], p[1], w) for p, w in zip(far_hog_pts, far_hog_w)]
        ) if len(far_hog_pts) else (0.0, 0.0)

        if near_conf_now > 0.0 and far_conf_now > 0.0 and abs(far_angle_now - near_angle_now) > 5.0:
            far_hog_pts, far_hog_w = np.zeros((0, 2)), np.zeros(0)

        hog_pts = np.concatenate([near_hog_pts, far_hog_pts]) if (
            len(near_hog_pts) or len(far_hog_pts)
        ) else np.zeros((0, 2))
        hog_y = np.concatenate([
            np.full(len(near_hog_pts), NEAR_HOGLINE_Y_CM),
            np.full(len(far_hog_pts), FAR_HOGLINE_Y_CM),
        ])
        hog_strength = np.concatenate([near_hog_w, far_hog_w])

        far_resid_now = _far_ring_distance(H_current, far_pts, far_radius)
        far_scale = _robust_scale_cm(far_resid_now, NEAR_TRUST_FLOOR_CM)
        far_weight = 1.0 / far_scale
        hog_weight = hog_strength / NEAR_TRUST_FLOOR_CM

        H_new = solve_homography_geometric(
            H_current, near_pts, near_phys, far_pts, far_radius, far_weight,
            hog_pts, hog_y, hog_weight
        )

        near_proj = output_px_to_physical(_apply_h(H_new, near_pts))
        rms = math.sqrt(float(np.mean(np.sum((near_proj - near_phys) ** 2, axis=1))))
        quality = measure_house_quality(frame_undistorted, H_new)

        print(
            f"[Geometrinen korjaus (varipohjainen) {iteration}/{max_iterations}] "
            f"lahempi RMS: {rms:.4f} cm, {house_quality_str(quality)} "
            f"(kaukaisen renkaan pisteita {len(far_pts)}, paino {far_weight:.3f}; "
            f"hogline-pisteita {len(hog_pts)}, lahi/kauko-kulma "
            f"{near_angle_now:.2f}/{far_angle_now:.2f})"
        )

        if best is not None:
            accept, _, _ = candidate_is_better(
                frame_undistorted, H_new, rms, frame_undistorted, best["H_final"], best["rms"]
            )
        else:
            accept = True

        if not accept or rms >= best_rms:
            stall_count += 1
            if stall_count >= STALL_PATIENCE:
                print("  Ei enaa hyvaksyttavaa parannusta, pysaytetaan.")
                break
            H_current = H_new
            continue

        stall_count = 0
        relative_improvement = (best_rms - rms) / best_rms if math.isfinite(best_rms) else None
        best = {
            "H_final": H_new, "topdown_raw": topdown_raw, "rms": rms,
            "quality": quality, "iterations": iteration,
        }
        best_rms = rms
        H_current = H_new

        if relative_improvement is not None and relative_improvement < min_relative_improvement:
            break

    if best is None:
        raise RuntimeError("Geometrista homografian korjausta (varipohjainen) ei saatu laskettua.")

    return best


def calibrate_camera_from_image_with_seed(filename):
    """Testi_01_02:n oma korvaava kalibrointi (katso taman tiedoston
    alkupaan kommentti periaatteesta) - lahemman pesan tunnistus on
    TASMALLEEN sama kuin Testi_01_01:ssa/kamera9_01.py:ssa (koskematon),
    mutta kaukaisen pesan loytaminen ja koko homografian ratkaisu on
    korvattu uudella, kayttajan kanssa askel askeleelta validoidulla
    menetelmalla (rivi-skannaus + varipohjainen rengastunnistus +
    harmaasavypohjainen hoglinetunnistus + geometrinen tarkennus)."""

    frame = cv2.imread(filename)

    if frame is None:
        raise RuntimeError(f"Kuvaa ei voitu avata: {filename}")

    image_height, image_width = frame.shape[:2]

    near_blue_mask = create_blue_mask(frame)
    near_red_mask = create_red_mask(frame)

    blue_outer, blue_inner = find_house_pair(
        near_blue_mask, NEAR_BLUE_MIN_AREA, NEAR_BLUE_MIN_RATIO, NEAR_BLUE_MIN_SIZE_RATIO
    )
    red_outer, red_inner = find_house_pair(
        near_red_mask, NEAR_RED_MIN_AREA, NEAR_RED_MIN_RATIO, NEAR_RED_MIN_SIZE_RATIO
    )

    if blue_outer is None or blue_inner is None or red_outer is None or red_inner is None:
        raise RuntimeError("Lahemman pesan renkaita ei loytynyt kokonaan.")

    segments = detect_line_segments(frame, blue_outer)
    (t_pair, centerline_pair, t_line, centerline, house_center) = select_t_and_centerline(
        segments, blue_outer
    )
    v_t, v_cl = build_image_directions(t_pair, centerline_pair)

    observations = build_calibration_observations(blue_outer, blue_inner, red_outer, red_inner)

    if len(observations) < 2:
        raise RuntimeError("Kalibrointiin tarvitaan vahintaan 2 ympyraa.")

    camera = build_camera_model(observations, image_width, image_height)

    far_ref = project_point(0.0, 1000.0, camera, v_t, v_cl)
    dir_forward = np.asarray(far_ref, dtype=np.float64) - np.asarray(house_center, dtype=np.float64)
    dir_forward /= np.linalg.norm(dir_forward)
    dir_lateral = np.array(v_t, dtype=np.float64)
    dir_lateral = dir_lateral / np.linalg.norm(dir_lateral)

    near_img_pts, near_phys_pts, near_labels = near_house_correspondences(
        t_line, centerline, house_center, blue_outer, blue_inner, red_outer, red_inner,
        dir_lateral, dir_forward
    )

    camera_matrix = build_camera_matrix(image_width, image_height)
    best_k1, best_k1_rms, baseline_rms = estimate_radial_distortion_k1(
        near_img_pts, near_phys_pts, camera_matrix
    )
    k1_improvement = (
        (baseline_rms - best_k1_rms) / baseline_rms
        if baseline_rms > 1e-9 and math.isfinite(baseline_rms) else 0.0
    )
    if k1_improvement > K1_MIN_RELATIVE_IMPROVEMENT and abs(best_k1) > K1_MIN_MAGNITUDE:
        dist_coeffs = np.array([best_k1, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    else:
        best_k1 = 0.0
        dist_coeffs = np.zeros(5, dtype=np.float64)

    frame_undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs)
    near_pts_frame = undistort_points_px(
        np.array(near_img_pts, dtype=np.float64), camera_matrix, best_k1
    )

    print("  Etsitaan kaukaisen pesan karkea sijainti (rivi-skannaus + lahi-pesa-homografia)...")
    H_v1, output_w, output_h = build_far_house_center_seed(
        frame_undistorted, near_pts_frame, near_phys_pts
    )

    print("  Geometrinen tarkennus (varipohjainen kaukainen rengas + harmaasavy-hoglinet)...")
    best = refine_geometric_homography_color(
        frame_undistorted, H_v1, near_pts_frame, near_phys_pts, output_w, output_h
    )

    print(f"  Valittu H: RMS = {best['rms']:.3f} cm, {house_quality_str(best['quality'])}")

    return {
        "frame": frame,
        "frame_undistorted": frame_undistorted,
        "camera_matrix": camera_matrix,
        "best_k1": best_k1,
        "H_final": best["H_final"],
        "quality": best["quality"],
        "image_width": image_width,
        "image_height": image_height,
        "output_w": output_w,
        "output_h": output_h,
        "near_pts_frame": near_pts_frame,
        "near_phys": near_phys_pts,
    }
