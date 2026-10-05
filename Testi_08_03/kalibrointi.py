"""Kameran kalibrointi moodikuvasta: homografia (jaataso -> kuva), objektiivin vaaristyma k1, hoglinjojen paikka ja
kameran 3D-asento (K, R, t).

Menetelma:
 1) Lahemman pesan renkaat ja viivat tunnistetaan kuvasta (kalibrointi_perus.py) -> 17 pistevastaavuutta ja k1.
 2) Homografia pelkasta lahipesasta laajennettuna koko radalle; laajennettu top-down-kuva skannataan rivi kerrallaan:
    rivi jolla on sinista keskiviivan molemmin puolin ja punaista niiden valissa = kaukainen pesa (karkea Y ja X).
 3) Alustava homografia: lahipesan 17 pistetta + kaukaisen pesan keskipiste (0, FAR_HOUSE_Y_CM).
 4) Geometrinen tarkennus: kaukaisen pesan renkaat (varipohjainen tunnistus, ympyrarajoitteet), hoglinjat
    (harmaasavykynnys, viivarajoitteet Y = vakio) ja keskiviiva koko radalta (X = 0) samaan pienimman neliosumman
    sovitukseen lahipesan pisteiden kanssa (kalibrointi_perus.solve_homography_geometric).
 5) Hoglinjojen paikka sovitetaan +-A.KALIB_HOGLINJA_TOLERANSSI_CM sisalla (yhteinen siirtyma molemmille, koska
    hoglinjat ovat samalla etaisyydella omasta T-viivastaan); vapaampi sovitus valitaan vain jos pesat eivat huonone.
 6) Kameran asento (K, R, t) puretaan homografiasta ja tarkennetaan samoilla rajoitteilla (fit_full_camera_pose).

Live-tilassa kalibrointi tehdaan kameran taydella resoluutiolla (1920x1080) ja skaalataan seurannan resoluutiolle
(scale_calibration); seurannan paikallinen taysi resoluutio kayttaa myos taysresoluutioista tulosta.
"""

import math

import cv2
import numpy as np

import asetukset as A
import kalibrointi_perus as kp
import kivimalli
import rata

# kaukaisen pesan rivi-skannaus (laajennettu top-down-kuva)
FAR_HOUSE_ROW_SCAN_MAX_GAP_PX = 15
FAR_HOUSE_ROW_SCAN_CENTER_MARGIN_PX = 5
# kaukaisen renkaan varikynnys saadetaan niin, etta ROI:sta tama osuus on sinista / punaista
FAR_HOUSE_ROI_TARGET_BLUE_FRACTION = 0.31
FAR_HOUSE_ROI_TARGET_RED_FRACTION = 0.11
# hoglinjan haku: kaista +-50 cm, harmaasavykynnys niin etta 98 % kaistasta on "jaata"
HOGLINE_GRAY_HALF_BAND_CM = 50.0
HOGLINE_GRAY_TARGET_COVERAGE = 0.98


def _extended_canvas_size(extended_y_max_cm):
    output_w = int(round((rata.OUTPUT_X_MAX_CM - rata.OUTPUT_X_MIN_CM) * rata.PIXELS_PER_CM))
    output_h = int(round(extended_y_max_cm * rata.PIXELS_PER_CM))
    return output_w, output_h


def _physical_to_output_px_extended(physical_pts, extended_y_max_cm):
    pts = np.asarray(physical_pts, dtype=np.float64)
    ox = (pts[:, 0] - rata.OUTPUT_X_MIN_CM) * rata.PIXELS_PER_CM
    oy = (extended_y_max_cm - pts[:, 1]) * rata.PIXELS_PER_CM
    return np.column_stack([ox, oy])


def _to_output_px_extended(x_cm, y_cm, extended_y_max_cm):
    px = (x_cm - rata.OUTPUT_X_MIN_CM) * rata.PIXELS_PER_CM
    py = (extended_y_max_cm - y_cm) * rata.PIXELS_PER_CM
    return px, py


def _compute_crop_row_range_extended(full_height_px, center_y_cm, half_height_cm, extended_y_max_cm):
    """kp.compute_crop_row_range laajennetulle kanvaasille (Y-max = extended_y_max_cm): kalibrointi_perus kayttaa
    kiinteaa rata.OUTPUT_Y_MAX_CM:aa, joka antaisi vaaran rivialueen."""
    _, row_a = _to_output_px_extended(0.0, center_y_cm + half_height_cm, extended_y_max_cm)
    _, row_b = _to_output_px_extended(0.0, center_y_cm - half_height_cm, extended_y_max_cm)
    row_top = int(max(0, math.floor(min(row_a, row_b))))
    row_bottom = int(min(full_height_px - 1, math.ceil(max(row_a, row_b))))
    return row_top, row_bottom


def _crop_house_view_extended(topdown, center_y_cm, half_height_cm, extended_y_max_cm):
    row_top, row_bottom = _compute_crop_row_range_extended(
        topdown.shape[0], center_y_cm, half_height_cm, extended_y_max_cm
    )
    return topdown[row_top : row_bottom + 1, :].copy()


def _expected_house_center_in_crop_extended(full_height_px, center_y_cm, half_height_cm, extended_y_max_cm):
    row_top, _ = _compute_crop_row_range_extended(full_height_px, center_y_cm, half_height_cm, extended_y_max_cm)
    cx, cy_full = _to_output_px_extended(0.0, center_y_cm, extended_y_max_cm)
    return (cx, cy_full - row_top)


def _find_blue_red_blue_pattern(
    blue_mask,
    red_mask,
    center_col,
    row_range=None,
    margin_px=FAR_HOUSE_ROW_SCAN_CENTER_MARGIN_PX,
    max_gap_px=FAR_HOUSE_ROW_SCAN_MAX_GAP_PX,
):
    """Kaukaisen pesan tunnistus: rivit, joilla on sinista center_col:in molemmin puolin ja punaista niiden valissa
    (kuvio ei osu mainospaneeleihin, toisin kuin pelkka suurin sininen alue). Palauttaa (rivi, sarake) suurimman
    rivijoukon keskelta tai None."""

    h = blue_mask.shape[0]
    y_lo, y_hi = (0, h) if row_range is None else row_range

    qualifying = []
    for y in range(y_lo, y_hi):
        blue_cols = np.where(blue_mask[y] > 0)[0]
        if len(blue_cols) == 0:
            continue
        left_cols = blue_cols[blue_cols < center_col - margin_px]
        right_cols = blue_cols[blue_cols > center_col + margin_px]
        if len(left_cols) == 0 or len(right_cols) == 0:
            continue
        left_edge = left_cols.max()
        right_edge = right_cols.min()
        red_cols = np.where(red_mask[y] > 0)[0]
        red_between = red_cols[(red_cols > left_edge) & (red_cols < right_edge)]
        if len(red_between) == 0:
            continue
        qualifying.append((y, (left_edge + right_edge) / 2.0))

    if not qualifying:
        return None

    rows = np.array([q[0] for q in qualifying])
    order = np.argsort(rows)
    rows_sorted = rows[order]
    cols_sorted = np.array([q[1] for q in qualifying])[order]

    clusters = []
    start_idx = 0
    for i in range(1, len(rows_sorted)):
        if rows_sorted[i] - rows_sorted[i - 1] > max_gap_px:
            clusters.append((start_idx, i))
            start_idx = i
    clusters.append((start_idx, len(rows_sorted)))

    best = max(clusters, key=lambda c: rows_sorted[c[1] - 1] - rows_sorted[c[0]])
    lo, hi = best
    found_row = float(np.mean(rows_sorted[lo:hi]))
    found_col = float(np.mean(cols_sorted[lo:hi]))
    return found_row, found_col


def find_far_house_row_scan_estimate(topdown_extended, extended_y_max_cm):
    """Skannaa laajennetun (lahi-pesa-only-homografialla tehdyn) topdown-
    kuvan sininen-punainen-sininen-kuviolla (_find_blue_red_blue_pattern)
    KOKO kuvan leveydelta, keskiviivan ymparilta. Lahemman pesan oma
    alue (aivan kuvan alareunassa, jossa ehto tayttyy itsestaan
    selvasti) suljetaan pois haista. Palauttaa (found_row, found_col)
    tai None jos mitaan ei loydy."""

    blue = kp.create_blue_mask(topdown_extended)
    red = kp.create_red_mask(topdown_extended)
    h, w = blue.shape
    center_col = int(round((0.0 - rata.OUTPUT_X_MIN_CM) * rata.PIXELS_PER_CM))
    near_house_exclude_row = int(h - (rata.HOUSE_RADIUS_CM + 100.0) * rata.PIXELS_PER_CM)

    return _find_blue_red_blue_pattern(blue, red, center_col, row_range=(0, near_house_exclude_row))


def _fit_circle_algebraic(points):
    """Yksinkertainen algebrallinen ympyrasovitus (Kasa) pistejoukkoon.
    Palauttaa (cx, cy, r) tai None jos pisteita on liian vahan."""
    if len(points) < 5:
        return None
    x, y = points[:, 0], points[:, 1]
    A = np.column_stack([x, y, np.ones_like(x)])
    b = x**2 + y**2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = sol[0] / 2.0, sol[1] / 2.0
    r = math.sqrt(max(sol[2] + cx**2 + cy**2, 0.0))
    return cx, cy, r


def _three_ellipse_fit_center_in_crop(crop, center_col_hint, search_radius_px=None):
    """Kaukaisen pesan keskipiste cropissa: vahvistaa sininen-punainen-sininen -kuvion center_col_hint:in ymparilta,
    saataa sinisen/punaisen HSV-kynnyksen pinta-alaosuuteen (FAR_HOUSE_ROI_TARGET_*_FRACTION) ja sovittaa ympyrat
    kolmelle renkaalle sateittaisella reunanhaulla (molemmat kaaret symmetrisesti). Palauttaa keskipisteiden
    keskiarvon crop-koordinaateissa tai None."""

    blue_raw = kp.create_blue_mask(crop)
    red_raw = kp.create_red_mask(crop)
    pattern = _find_blue_red_blue_pattern(blue_raw, red_raw, int(round(center_col_hint)))
    if pattern is None:
        return None
    pattern_row, pattern_col = pattern

    ys, xs = np.where(blue_raw > 0)
    if len(xs) < 5:
        return None
    pts_all = np.column_stack([xs, ys]).astype(np.float32)
    if search_radius_px is not None:
        dists = np.hypot(pts_all[:, 0] - pattern_col, pts_all[:, 1] - pattern_row)
        near_pattern = pts_all[dists <= search_radius_px]
        if len(near_pattern) >= 5:
            pts_all = near_pattern
    (ecx, ecy), erad = cv2.minEnclosingCircle(pts_all)

    roi_mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    cv2.circle(roi_mask, (int(ecx), int(ecy)), int(erad), 255, -1)
    roi_area = np.count_nonzero(roi_mask)
    if roi_area == 0:
        return None

    s_blue = _search_s_low_for_area_fraction(
        _hsv_blue_mask_s_low, crop, roi_mask, roi_area, FAR_HOUSE_ROI_TARGET_BLUE_FRACTION
    )
    s_red = _search_s_low_for_area_fraction(
        _hsv_red_mask_s_low, crop, roi_mask, roi_area, FAR_HOUSE_ROI_TARGET_RED_FRACTION
    )
    blue_final = cv2.bitwise_and(_hsv_blue_mask_s_low(crop, s_blue), roi_mask)
    red_final = cv2.bitwise_and(_hsv_red_mask_s_low(crop, s_red), roi_mask)

    center_local = (ecx, ecy)
    blue_inner_pts, blue_outer_pts = _radial_ring_edges(blue_final, center_local, erad + 40)
    _, red_outer_pts = _radial_ring_edges(red_final, center_local, erad + 40)

    centers = []
    for pts in (blue_inner_pts, blue_outer_pts, red_outer_pts):
        fit = _fit_circle_algebraic(pts)
        if fit is not None:
            centers.append((fit[0], fit[1]))

    if not centers:
        return None

    return tuple(np.mean(np.array(centers), axis=0))


def build_far_house_center_seed(frame_undistorted, near_pts_frame, near_phys_pts):
    """Toteuttaa taman tiedoston alkupaan kommentin vaiheet 1-4: rakentaa
    lahi-pesa-only-homografian, laajentaa 10m takarajan yli, loytaa
    karkean arvion kaukaisen pesan sijainnista rivi-skannauksella,
    "skaalaa" (pystysuora affiini) loydetyn rivin FAR_HOUSE_Y_CM:n
    kohdalle, ja palauttaa (H_v1, output_w, output_h) - alustavan
    homografian koko radalle (lahempi pesa + kaukaisen pesan keski-
    piste), valmiina geometriseen tarkennukseen (refine_with_color_far).
    Nostaa RuntimeError:in jos karkeaa arviota ei loydy."""

    back_line_y_cm = rata.FAR_HOUSE_Y_CM + rata.HOUSE_RADIUS_CM
    extended_y_max_cm = back_line_y_cm + 1000.0
    ext_w, ext_h = _extended_canvas_size(extended_y_max_cm)

    dst_ext = _physical_to_output_px_extended(near_phys_pts, extended_y_max_cm).astype(np.float32)
    src = np.asarray(near_pts_frame, dtype=np.float32)
    H_near_only, _ = cv2.findHomography(src, dst_ext, method=0)

    topdown_extended = cv2.warpPerspective(frame_undistorted, H_near_only, (ext_w, ext_h))

    estimate = find_far_house_row_scan_estimate(topdown_extended, extended_y_max_cm)
    if estimate is None:
        raise RuntimeError(
            "Kaukaista pesaa ei loytynyt rivi-skannauksella "
            "(ei sininen-punainen-sininen -kuviota laajennetusta topdown-kuvasta)."
        )
    found_row, found_col = estimate

    # Koko laajennettu topdown-kuva venytetaan pystysuunnassa (ankkuri lahempi pesa) niin, etta loydetty rivi osuu
    # FAR_HOUSE_Y_CM:iin. Kolmen ellipsin sovitus tehdaan topdown-avaruudessa, jossa rengas on jo lahes ympyra.
    target_row = (extended_y_max_cm - rata.FAR_HOUSE_Y_CM) * rata.PIXELS_PER_CM
    anchor_row = float(ext_h)
    scale_factor = (anchor_row - target_row) / (anchor_row - found_row)
    a = scale_factor
    b = anchor_row * (1.0 - scale_factor)
    rescale_M = np.array([[1.0, 0.0, 0.0], [0.0, a, b]], dtype=np.float64)
    topdown_rescaled_ext = cv2.warpAffine(topdown_extended, rescale_M, (ext_w, ext_h))

    # Rivi <-> Y on nyt sama kaava kuin standardikanvaasilla; rajaukseen laajennetut versiot (oikea Y-max).
    far_row_top_ext, _ = _compute_crop_row_range_extended(
        ext_h, rata.FAR_HOUSE_Y_CM, kp.HOUSE_CROP_HALF_HEIGHT_CM, extended_y_max_cm
    )
    far_crop_rescaled = _crop_house_view_extended(
        topdown_rescaled_ext, rata.FAR_HOUSE_Y_CM, kp.HOUSE_CROP_HALF_HEIGHT_CM, extended_y_max_cm
    )
    expected_center_local = _expected_house_center_in_crop_extended(
        ext_h, rata.FAR_HOUSE_Y_CM, kp.HOUSE_CROP_HALF_HEIGHT_CM, extended_y_max_cm
    )

    fit_local = _three_ellipse_fit_center_in_crop(far_crop_rescaled, expected_center_local[0])

    if fit_local is None:
        # Ei loytynyt kolmen ellipsin sovitusta rescaloidusta topdownista
        # (esim. liikesumennuksen pirstoma rengas) - kaytetaan silti
        # rivi-skannauksen omaa karkeaa (row,col)-arviota takaisin-
        # projisoituna, parempi kuin ei mitaan.
        fit_row_ext = far_row_top_ext + expected_center_local[1]
        fit_col_ext = found_col
    else:
        fit_col_ext, fit_local_row = fit_local
        fit_row_ext = far_row_top_ext + fit_local_row

    # Takaisin: rescaloitu-extended -> extended (kaanteinen affiini) ->
    # kehys (kaanteinen H_near_only).
    fit_row_ext_unrescaled = (fit_row_ext - b) / a
    H_inv = np.linalg.inv(H_near_only)
    far_center_frame = cv2.perspectiveTransform(
        np.array([[[fit_col_ext, fit_row_ext_unrescaled]]], dtype=np.float64), H_inv
    )[0, 0]

    output_w = int(round((rata.OUTPUT_X_MAX_CM - rata.OUTPUT_X_MIN_CM) * rata.PIXELS_PER_CM))
    output_h = int(round(rata.OUTPUT_Y_MAX_CM * rata.PIXELS_PER_CM))

    all_img_pts = list(near_pts_frame) + [far_center_frame]
    all_phys_pts = list(near_phys_pts) + [(0.0, rata.FAR_HOUSE_Y_CM)]
    src_v1 = np.array(all_img_pts, dtype=np.float32)
    dst_v1 = rata.physical_to_output_px(np.array(all_phys_pts, dtype=np.float64)).astype(np.float32)
    H_v1, _ = cv2.findHomography(src_v1, dst_v1, method=0)

    return H_v1, output_w, output_h


def _hsv_blue_mask_s_low(frame, s_low, v_low=40):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([90, s_low, v_low], dtype=np.uint8), np.array([140, 255, 255], dtype=np.uint8))
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel)


def _hsv_red_mask_s_low(frame, s_low, v_low=40):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, np.array([0, s_low, v_low], dtype=np.uint8), np.array([10, 255, 255], dtype=np.uint8))
    m2 = cv2.inRange(hsv, np.array([170, s_low, v_low], dtype=np.uint8), np.array([180, 255, 255], dtype=np.uint8))
    mask = cv2.bitwise_or(m1, m2)
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel)


def _search_s_low_for_area_fraction(mask_fn, view, roi_mask, roi_area, target_fraction):
    best_s, best_diff = 40, float("inf")
    for s_low in range(5, 90, 1):
        mask = cv2.bitwise_and(mask_fn(view, s_low), roi_mask)
        frac = np.count_nonzero(mask) / roi_area
        diff = abs(frac - target_fraction)
        if diff < best_diff:
            best_diff, best_s = diff, s_low
    return best_s


def _radial_ring_edges(mask, center, max_r, min_r=20, num_angles=360):
    """Kavelee jokaisen kulman suunnassa keskipisteesta ulospain ja
    palauttaa (sisareuna, ulkoreuna) -pisteet niissa kulmissa joissa
    maski osuu johonkin sateeseen - puuttuvat kulmat (esim. kiven/
    tangon peittama kohta) jaavat yksinkertaisesti pois, koska rengas
    on talla kamerakulmalla usein osittain pirstoutunut/peittynyt."""
    cx, cy = center
    inner_pts, outer_pts = [], []
    for i in range(num_angles):
        theta = 2 * math.pi * i / num_angles
        dx, dy = math.cos(theta), math.sin(theta)
        prev = 0
        r_in, r_out = None, None
        for r in range(min_r, int(max_r) + 20):
            x, y = int(round(cx + dx * r)), int(round(cy + dy * r))
            if not (0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]):
                break
            val = 1 if mask[y, x] > 0 else 0
            if prev == 0 and val == 1 and r_in is None:
                r_in = r
            if prev == 1 and val == 0:
                r_out = r
            prev = val
        if r_in is not None:
            inner_pts.append((cx + dx * r_in, cy + dy * r_in))
        if r_out is not None:
            outer_pts.append((cx + dx * r_out, cy + dy * r_out))
    return np.array(inner_pts), np.array(outer_pts)


def far_ring_points_color_based(topdown_raw, H_current, max_points_per_ring=30):
    """Sama sopimus kuin kp.far_house_ring_points_in_frame (palauttaa
    (points, radii) oikaistun kuvan koordinaatistossa geometrista
    tarkennusta - kp.solve_homography_geometric - varten), mutta
    tunnistus tehdaan VARIPOHJAISELLA (HSV) menetelmalla kalibrointi_perus.py:n omaan
    radiaalihaku+pakotettu-yhteismuoto-sovitukseen sijaan: pinta-
    alaosuuteen (ROI = sinisen maskin pienin ymparoiva ympyra) saadettu
    Saturation-kynnys + sateittainen reunanhaku molemmille renkaille
    (sininen ulko/sisa, punainen ulko). Kayttajan kanssa todettu
    tuottavan visuaalisesti tarkemman (oikean kokoisen) sovituksen
    kuin kalibrointi_perus.py:n oma menetelma talla kamerakulmalla."""

    far_row_top, _ = kp.compute_crop_row_range(topdown_raw.shape[0], rata.FAR_HOUSE_Y_CM, kp.HOUSE_CROP_HALF_HEIGHT_CM)
    view = kp.crop_house_view(topdown_raw, rata.FAR_HOUSE_Y_CM, kp.HOUSE_CROP_HALF_HEIGHT_CM)

    # Sama vahvistus kuin _three_ellipse_fit_center_in_crop:issa: sininen-punainen-sininen -kuvion pitaa loytya
    # paikallisesti ennen kuin renkaan pisteita palautetaan (H_current voi viela olla niin vino, etta crop osuu
    # mainospaneeliin). Jos kuviota ei loydy, palautetaan tyhja: ratkaisija nojaa silloin lahempaan pesaan ja
    # hoglineihin.
    blue_raw = kp.create_blue_mask(view)
    red_raw = kp.create_red_mask(view)
    center_col = int(round((0.0 - rata.OUTPUT_X_MIN_CM) * rata.PIXELS_PER_CM))
    pattern = _find_blue_red_blue_pattern(blue_raw, red_raw, center_col)
    if pattern is None:
        return np.zeros((0, 2)), np.zeros(0)
    pattern_row, pattern_col = pattern

    ys, xs = np.where(blue_raw > 0)
    if len(xs) < 5:
        return np.zeros((0, 2)), np.zeros(0)
    pts_all = np.column_stack([xs, ys]).astype(np.float32)
    dists = np.hypot(pts_all[:, 0] - pattern_col, pts_all[:, 1] - pattern_row)
    near_pattern = pts_all[dists <= kp.HOUSE_CROP_HALF_HEIGHT_CM * rata.PIXELS_PER_CM * 0.6]
    if len(near_pattern) < 5:
        near_pattern = pts_all
    (ecx, ecy), erad = cv2.minEnclosingCircle(near_pattern)

    roi_mask = np.zeros(view.shape[:2], dtype=np.uint8)
    cv2.circle(roi_mask, (int(ecx), int(ecy)), int(erad), 255, -1)
    roi_area = np.count_nonzero(roi_mask)
    if roi_area == 0:
        return np.zeros((0, 2)), np.zeros(0)

    s_blue = _search_s_low_for_area_fraction(
        _hsv_blue_mask_s_low, view, roi_mask, roi_area, FAR_HOUSE_ROI_TARGET_BLUE_FRACTION
    )
    s_red = _search_s_low_for_area_fraction(
        _hsv_red_mask_s_low, view, roi_mask, roi_area, FAR_HOUSE_ROI_TARGET_RED_FRACTION
    )
    blue_final = cv2.bitwise_and(_hsv_blue_mask_s_low(view, s_blue), roi_mask)
    red_final = cv2.bitwise_and(_hsv_red_mask_s_low(view, s_red), roi_mask)

    center_local = (ecx, ecy)
    blue_inner_pts, blue_outer_pts = _radial_ring_edges(blue_final, center_local, erad + 40)
    _, red_outer_pts = _radial_ring_edges(red_final, center_local, erad + 40)

    topdown_pts, radii = [], []
    for pts, radius_cm in (
        (blue_inner_pts, rata.BLUE_INNER_RADIUS_CM),
        (blue_outer_pts, rata.BLUE_OUTER_RADIUS_CM),
        (red_outer_pts, rata.RED_OUTER_RADIUS_CM),
    ):
        if len(pts) == 0:
            continue
        if len(pts) > max_points_per_ring:
            idx = np.linspace(0, len(pts) - 1, max_points_per_ring).astype(int)
            pts = pts[idx]
        for x, y in pts:
            topdown_pts.append((x, y + far_row_top))
            radii.append(radius_cm)

    if not topdown_pts:
        return np.zeros((0, 2)), np.zeros(0)

    points = kp.frame_points_from_topdown(np.array(topdown_pts, dtype=np.float64), H_current)
    return points, np.array(radii, dtype=np.float64)


def hogline_points_gray_threshold(topdown_raw, hogline_y_cm, H_current, half_band_cm=HOGLINE_GRAY_HALF_BAND_CM):
    """Sama sopimus kuin kp.hogline_points_in_frame (palauttaa (points,
    weights) oikaistun kuvan koordinaatistossa), mutta harmaasavy-
    kynnys-menetelmalla: haetaan nimellisen hogline-rivin ymparilta
    (half_band_cm) kynnysarvo (harmaasavy < t) joka antaa YHTENAISEN
    (lahes 100% sarakepeittavyyden) tumman kaistan koko leveydelta -
    kayttajan kanssa todettu paljon luotettavammaksi kuin kalibrointi_perus.py:n oma
    segmenttipohjainen detect_hogline_points talla kamerakulmalla
    (joka loysi usein vain muutaman hajanaisen pisteen tai tarttui
    sponsoritekstin kontaminaatioon)."""

    _, nominal_row = rata.to_output_px(0.0, hogline_y_cm)
    half_px = int(half_band_cm * rata.PIXELS_PER_CM)
    y0 = max(0, int(nominal_row) - half_px)
    y1 = min(topdown_raw.shape[0], int(nominal_row) + half_px)
    if y1 - y0 < 10:
        return np.zeros((0, 2)), np.zeros(0)

    band = topdown_raw[y0:y1, :]
    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    margin = int(gray.shape[1] * 0.05)
    n_cols = gray.shape[1] - 2 * margin
    if n_cols <= 0:
        return np.zeros((0, 2)), np.zeros(0)

    best_mask, best_cov = None, 0.0
    for t in range(0, 256, 2):
        mask = (gray < t).astype(np.uint8) * 255
        cols_with_dark = np.count_nonzero(np.any(mask[:, margin : gray.shape[1] - margin] > 0, axis=0))
        cov = cols_with_dark / n_cols
        if cov >= HOGLINE_GRAY_TARGET_COVERAGE:
            best_mask, best_cov = mask, cov
            break
        if cov > best_cov:
            best_mask, best_cov = mask, cov

    if best_mask is None:
        return np.zeros((0, 2)), np.zeros(0)

    darkness = 255.0 - gray.astype(np.float32)
    topdown_pts, strengths = [], []
    for x in range(margin, gray.shape[1] - margin):
        rows = np.where(best_mask[:, x] > 0)[0]
        if len(rows) == 0:
            continue
        weights = darkness[rows, x]
        if weights.sum() <= 0:
            weights = np.ones_like(weights)
        y_mean = float(np.average(rows, weights=weights))
        topdown_pts.append((float(x), y_mean + y0))
        strengths.append(float(np.max(weights)))

    if not topdown_pts:
        return np.zeros((0, 2)), np.zeros(0)

    topdown_pts = np.array(topdown_pts, dtype=np.float64)
    strengths = np.array(strengths, dtype=np.float64)
    weights_norm = strengths / max(float(np.mean(strengths)), 1e-6)

    points = kp.frame_points_from_topdown(topdown_pts, H_current)
    return points, weights_norm


def _hogline_targets(H, near_pts, far_pts, tol_cm=None):
    """Molempien hoglinjojen tavoite-Y. Hoglinjat ovat samalla etaisyydella omasta T-viivastaan (640 + d cm) -> yksi
    yhteinen siirtyma d molempien viivojen mittauksista painotettuna (kaukaisen paino A.KALIB_KAUKOHOG_PAINO; lahihog
    mitataan tarkemmin). d rajataan +-tol_cm."""
    if tol_cm is None:
        tol_cm = A.KALIB_HOGLINJA_TOLERANSSI_CM
    if tol_cm <= 0.0 or (len(near_pts) == 0 and len(far_pts) == 0):
        return float(rata.NOMINAL_NEAR_HOGLINE_Y_CM), float(rata.NOMINAL_FAR_HOGLINE_Y_CM)
    parts, weights = [], []
    if len(near_pts):
        y = rata.output_px_to_physical(rata.apply_h(H, np.asarray(near_pts, dtype=np.float64)))[:, 1]
        parts.append(float(np.median(y)) - rata.NOMINAL_NEAR_HOGLINE_Y_CM)  # + = kauempana lahi-T:sta
        weights.append(1.0)
    if len(far_pts):
        y = rata.output_px_to_physical(rata.apply_h(H, np.asarray(far_pts, dtype=np.float64)))[:, 1]
        parts.append(rata.NOMINAL_FAR_HOGLINE_Y_CM - float(np.median(y)))  # + = kauempana kauko-T:sta
        weights.append(A.KALIB_KAUKOHOG_PAINO if len(near_pts) else 1.0)
    d = float(np.clip(np.average(parts, weights=weights), -tol_cm, tol_cm))
    return float(rata.NOMINAL_NEAR_HOGLINE_Y_CM + d), float(rata.NOMINAL_FAR_HOGLINE_Y_CM - d)


# Keskiviiva koko radalta: pisteet top-down-kuvasta CL_STEP_CM valein (tumma kapea viiva X ~ 0), viivarajoitteina X = 0.
# 1280x720-kuvassa kaukopaan pikseli on n. 2 cm, joten pelkka pesien keskipisteiden kautta maaraytyva X = 0 oli 1-3 cm sivussa.
CL_STEP_CM = 20.0  # pistevali radan suunnassa
CL_HALF_BAND_CM = 8.0  # yhden pisteen rivikaista (+-)
CL_SEARCH_CM = 15.0  # haku nykyisen X = 0:n ymparilta (+-)
CL_MIN_CONTRAST = 4.0  # viivan tummuus taustaan nahden (harmaasavy)
CL_MAX_WIDTH_CM = 16.0  # viivan leveys enintaan (puolen tummuuden kohdalta); kaukopaassa sumea
CL_Y_RANGE_CM = (150.0, 3900.0)


def centerline_points_topdown(topdown_raw, H_current):
    """Keskiviivan pisteet (oikaistun kuvan px) + painot. Jokaiselle Y:lle CL_STEP_CM valein: rivikaistan keskiarvoprofiili
    X:n suhteen, tummin kohta +-CL_SEARCH_CM, vaaditaan kapea (< CL_MAX_WIDTH_CM) ja riittavan tumma laakso; alipikseli-
    tarkennus paraabelilla. Vaakaviivat (hog-, T- ja takaviivat) tummentavat koko kaistan -> ei laaksoa -> ohitetaan."""
    gray = cv2.cvtColor(topdown_raw, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ppc = rata.PIXELS_PER_CM
    pts, strengths = [], []
    for y_cm in np.arange(CL_Y_RANGE_CM[0], CL_Y_RANGE_CM[1], CL_STEP_CM):
        _, row = rata.to_output_px(0.0, y_cm)
        r0, r1 = int(row - CL_HALF_BAND_CM * ppc), int(row + CL_HALF_BAND_CM * ppc)
        xa, _ = rata.to_output_px(-CL_SEARCH_CM - 15.0, y_cm)
        xb, _ = rata.to_output_px(CL_SEARCH_CM + 15.0, y_cm)
        xa, xb = int(xa), int(xb)
        if r0 < 0 or r1 > gray.shape[0] or xa < 0 or xb > gray.shape[1]:
            continue
        prof = gray[r0:r1, xa:xb].mean(axis=0)
        prof = np.convolve(prof, np.ones(3) / 3.0, mode="same")
        m = int(15.0 * ppc)  # haku vain keskiosasta (+-CL_SEARCH_CM)
        i = m + int(np.argmin(prof[m : len(prof) - m]))
        base = float(np.median(prof))
        depth = base - float(prof[i])
        if depth < CL_MIN_CONTRAST:
            continue
        half = base - depth / 2.0
        lo = i
        while lo > 0 and prof[lo] < half:
            lo -= 1
        hi = i
        while hi < len(prof) - 1 and prof[hi] < half:
            hi += 1
        if (hi - lo) / ppc > CL_MAX_WIDTH_CM:
            continue
        # keskikohta: painotettu keskiarvo puolen tummuuden alittavalta alueelta (sumea, leveakin viiva -> vakaa)
        idx = np.arange(lo + 1, hi)
        w = half - prof[lo + 1 : hi]
        if len(idx) == 0 or w.sum() <= 0:
            continue
        pts.append((xa + float(np.sum(idx * w) / np.sum(w)), row))
        strengths.append(depth)
    if not pts:
        return np.zeros((0, 2)), np.zeros(0)
    strengths = np.array(strengths, dtype=np.float64)
    return kp.frame_points_from_topdown(np.array(pts, dtype=np.float64), H_current), strengths / max(
        float(np.mean(strengths)), 1e-6
    )


def _solve_homography_with_centerline(
    H_init, near_pts, near_phys, far_pts, far_radius, far_weight, hog_pts, hog_y, hog_weight, cl_pts, cl_weight
):
    """kp.solve_homography_geometric + keskiviivarajoite (X = 0 jokaiselle keskiviivapisteelle)."""

    def resid(p):
        base = kp._geometric_residuals(
            p, near_pts, near_phys, far_pts, far_radius, far_weight, hog_pts, hog_y, hog_weight
        )
        if not len(cl_pts):
            return base
        Hh = kp._h_from_params(p)
        xcm = rata.output_px_to_physical(rata.apply_h(Hh, cl_pts))[:, 0]
        return np.concatenate([base, xcm * cl_weight])

    return kp._h_from_params(kp._levenberg_marquardt(resid, kp._params_from_h(H_init)))


def refine_geometric_homography_color(
    frame_undistorted,
    H_init,
    near_pts,
    near_phys,
    output_w,
    output_h,
    max_iterations=None,
    min_relative_improvement=None,
    hog_tol_cm=0.0,
    label="varipohjainen",
):
    """Sama iteratiivinen konvergenssiperiaate kuin kp.refine_geometric_
    homography (katso sen kommentti), mutta kaukaisen renkaan ja
    hoglinien tunnistus jokaisella kierroksella tehdaan tama tiedoston
    varipohjaisilla/harmaasavypohjaisilla funktioilla (far_ring_points_
    color_based, hogline_points_gray_threshold) kalibrointi_perus.py:n omien sijaan.
    Palauttaa dictin: H_final, topdown_raw, rms, quality, iterations."""

    if max_iterations is None:
        max_iterations = kp.HOMOGRAPHY_REFINE_MAX_ITERATIONS
    if min_relative_improvement is None:
        min_relative_improvement = kp.HOMOGRAPHY_REFINE_MIN_RELATIVE_IMPROVEMENT

    STALL_PATIENCE = 2
    H_current = H_init
    best = None
    best_rms = float("inf")
    stall_count = 0

    for iteration in range(1, max_iterations + 1):

        topdown_raw = cv2.warpPerspective(frame_undistorted, H_current, (output_w, output_h))

        far_pts, far_radius = far_ring_points_color_based(topdown_raw, H_current)
        near_hog_pts, near_hog_w = hogline_points_gray_threshold(topdown_raw, rata.NEAR_HOGLINE_Y_CM, H_current)
        far_hog_pts, far_hog_w = hogline_points_gray_threshold(topdown_raw, rata.FAR_HOGLINE_Y_CM, H_current)

        near_angle_now, near_conf_now = (
            kp.robust_line_angle_from_points([(p[0], p[1], w) for p, w in zip(near_hog_pts, near_hog_w)])
            if len(near_hog_pts)
            else (0.0, 0.0)
        )
        far_angle_now, far_conf_now = (
            kp.robust_line_angle_from_points([(p[0], p[1], w) for p, w in zip(far_hog_pts, far_hog_w)])
            if len(far_hog_pts)
            else (0.0, 0.0)
        )

        if near_conf_now > 0.0 and far_conf_now > 0.0 and abs(far_angle_now - near_angle_now) > 5.0:
            far_hog_pts, far_hog_w = np.zeros((0, 2)), np.zeros(0)

        hog_pts = (
            np.concatenate([near_hog_pts, far_hog_pts]) if (len(near_hog_pts) or len(far_hog_pts)) else np.zeros((0, 2))
        )
        near_hog_y, far_hog_y = _hogline_targets(H_current, near_hog_pts, far_hog_pts, hog_tol_cm)
        hog_y = np.concatenate([np.full(len(near_hog_pts), near_hog_y), np.full(len(far_hog_pts), far_hog_y)])
        hog_strength = np.concatenate([near_hog_w, far_hog_w])

        far_resid_now = kp._far_ring_distance(H_current, far_pts, far_radius)
        far_scale = kp._robust_scale_cm(far_resid_now, kp.NEAR_TRUST_FLOOR_CM)
        far_weight = 1.0 / far_scale
        hog_weight = hog_strength / kp.NEAR_TRUST_FLOOR_CM

        cl_info = ""
        cl_pts, cl_w = centerline_points_topdown(topdown_raw, H_current)
        if len(cl_pts):
            # poikkeavat pois (esim. sponsoriteksti tai rengasreuna keskiviivan kohdalla): |X| > max(3 MAD, 3 cm)
            cl_x = rata.output_px_to_physical(rata.apply_h(H_current, cl_pts))[:, 0]
            med = float(np.median(cl_x))
            mad = 1.4826 * float(np.median(np.abs(cl_x - med)))
            keep = np.abs(cl_x - med) <= max(3.0 * mad, 3.0)
            cl_pts, cl_w = cl_pts[keep], cl_w[keep]
            cl_info = f"; keskiviivapisteita {len(cl_pts)}, X ka {float(np.mean(cl_x[keep])):+.2f} cm"
        H_new = _solve_homography_with_centerline(
            H_current,
            near_pts,
            near_phys,
            far_pts,
            far_radius,
            far_weight,
            hog_pts,
            hog_y,
            hog_weight,
            cl_pts,
            cl_w / kp.NEAR_TRUST_FLOOR_CM,
        )
        near_proj = rata.output_px_to_physical(rata.apply_h(H_new, near_pts))
        rms = math.sqrt(float(np.mean(np.sum((near_proj - near_phys) ** 2, axis=1))))
        quality = kp.measure_house_quality(frame_undistorted, H_new)

        print(
            f"[Geometrinen korjaus ({label}) {iteration}/{max_iterations}] "
            f"lahempi RMS: {rms:.4f} cm, {kp.house_quality_str(quality)} "
            f"(kaukaisen renkaan pisteita {len(far_pts)}, paino {far_weight:.3f}; "
            f"hogline-pisteita {len(hog_pts)}, lahi/kauko-kulma "
            f"{near_angle_now:.2f}/{far_angle_now:.2f}{cl_info})"
        )

        if best is not None:
            accept, _, _ = kp.candidate_is_better(
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
            "H_final": H_new,
            "topdown_raw": topdown_raw,
            "rms": rms,
            "quality": quality,
            "iterations": iteration,
            "near_hog_pts": near_hog_pts,
            "far_hog_pts": far_hog_pts,
        }
        best_rms = rms
        H_current = H_new

        if relative_improvement is not None and relative_improvement < min_relative_improvement:
            break

    if best is None:
        raise RuntimeError("Geometrista homografian korjausta (varipohjainen) ei saatu laskettua.")

    return best


def hires_median(samples, rows=60):
    """Mediaanikuva paloittain (muisti: 25 x 1920x1080 float64 kerralla olisi ~1,2 Gt)."""
    h = samples[0].shape[0]
    out = np.empty_like(samples[0])
    for r0 in range(0, h, rows):
        out[r0 : r0 + rows] = np.median(np.stack([s_[r0 : r0 + rows] for s_ in samples]), axis=0).astype(np.uint8)
    return out


def scale_calibration(calib_hi, pose_hi, frame_lo):
    """Taysresoluutioinen kalibrointi + asento -> seurannan resoluutio (frame_lo = 1280x720-moodikuva)."""
    h_lo, w_lo = frame_lo.shape[:2]
    s = w_lo / float(calib_hi["image_width"])
    S_inv = np.diag([1.0 / s, 1.0 / s, 1.0])
    K_lo = kp.build_camera_matrix(w_lo, h_lo)
    k1 = calib_hi["best_k1"]
    calib = dict(calib_hi)
    calib.update(
        frame=frame_lo,
        frame_undistorted=cv2.undistort(frame_lo, K_lo, np.array([k1, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)),
        camera_matrix=K_lo,
        H_final=calib_hi["H_final"] @ S_inv,
        image_width=w_lo,
        image_height=h_lo,
        near_pts_frame=np.asarray(calib_hi["near_pts_frame"], dtype=np.float64) * s,
        taysi_resoluutio=(calib_hi["image_width"], calib_hi["image_height"]),
        # seurannan paikallista taytta resoluutiota varten
        taysi_frame_undistorted=calib_hi["frame_undistorted"],
        taysi_camera_matrix=np.asarray(calib_hi["camera_matrix"], dtype=np.float64),
    )
    pose = dict(pose_hi)
    pose["K"] = np.diag([s, s, 1.0]) @ np.asarray(pose_hi["K"], dtype=np.float64)
    pose["K_taysi"] = np.asarray(pose_hi["K"], dtype=np.float64)
    return calib, pose


def calibrate_camera_from_image_with_seed(filename):
    """Kalibrointi moodikuvasta: lahemman pesan tunnistus (kalibrointi_perus), kaukaisen pesan loytaminen
    rivi-skannauksella ja varipohjaisella rengastunnistuksella, hoglinet harmaasavysta ja geometrinen tarkennus
    (homografia + objektiivin k1)."""

    frame = cv2.imread(filename)

    if frame is None:
        raise RuntimeError(f"Kuvaa ei voitu avata: {filename}")

    rata.set_hogline_positions(rata.NOMINAL_NEAR_HOGLINE_Y_CM, rata.NOMINAL_FAR_HOGLINE_Y_CM)

    image_height, image_width = frame.shape[:2]

    near_blue_mask = kp.create_blue_mask(frame)
    near_red_mask = kp.create_red_mask(frame)

    blue_outer, blue_inner = kp.find_house_pair(
        near_blue_mask, kp.NEAR_BLUE_MIN_AREA, kp.NEAR_BLUE_MIN_RATIO, kp.NEAR_BLUE_MIN_SIZE_RATIO
    )
    red_outer, red_inner = kp.find_house_pair(
        near_red_mask, kp.NEAR_RED_MIN_AREA, kp.NEAR_RED_MIN_RATIO, kp.NEAR_RED_MIN_SIZE_RATIO
    )

    if blue_outer is None or blue_inner is None or red_outer is None or red_inner is None:
        raise RuntimeError("Lahemman pesan renkaita ei loytynyt kokonaan.")

    segments = kp.detect_line_segments(frame, blue_outer)
    t_pair, centerline_pair, t_line, centerline, house_center = kp.select_t_and_centerline(segments, blue_outer)
    v_t, v_cl = kp.build_image_directions(t_pair, centerline_pair)

    observations = kp.build_calibration_observations(blue_outer, blue_inner, red_outer, red_inner)

    if len(observations) < 2:
        raise RuntimeError("Kalibrointiin tarvitaan vahintaan 2 ympyraa.")

    camera = kp.build_camera_model(observations, image_width, image_height)

    far_ref = kp.project_point(0.0, 1000.0, camera, v_t, v_cl)
    dir_forward = np.asarray(far_ref, dtype=np.float64) - np.asarray(house_center, dtype=np.float64)
    dir_forward /= np.linalg.norm(dir_forward)
    dir_lateral = np.array(v_t, dtype=np.float64)
    dir_lateral = dir_lateral / np.linalg.norm(dir_lateral)
    # kp.build_image_directions ottaa v_t:n merkin T-viivan janaparin JARJESTYKSESTA (sattumanvarainen) ->
    # joskus sivusuunta kaantyi ja homografiasta tuli PEILIKUVA (fysikaaliset x:t +-vaihtuivat, kaukaista pesaa ei
    # loytynyt). Ylhaalta katsova kamera ei koskaan peilaa: kuvassa (y alas) eteen x sivulle > 0 kuten
    # topdown-kuvassa (eteen = ylos, +x = oikealle). Kun merkki oli jo oikein, tulos on taysin ennallaan.
    if dir_forward[0] * dir_lateral[1] - dir_forward[1] * dir_lateral[0] < 0:
        dir_lateral = -dir_lateral

    near_img_pts, near_phys_pts, near_labels = kp.near_house_correspondences(
        t_line, centerline, house_center, blue_outer, blue_inner, red_outer, red_inner, dir_lateral, dir_forward
    )

    camera_matrix = kp.build_camera_matrix(image_width, image_height)
    best_k1, best_k1_rms, baseline_rms = kp.estimate_radial_distortion_k1(near_img_pts, near_phys_pts, camera_matrix)
    k1_improvement = (
        (baseline_rms - best_k1_rms) / baseline_rms if baseline_rms > 1e-9 and math.isfinite(baseline_rms) else 0.0
    )
    if k1_improvement > kp.K1_MIN_RELATIVE_IMPROVEMENT and abs(best_k1) > kp.K1_MIN_MAGNITUDE:
        dist_coeffs = np.array([best_k1, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    else:
        best_k1 = 0.0
        dist_coeffs = np.zeros(5, dtype=np.float64)

    print("  Etsitaan kaukaisen pesan karkea sijainti (rivi-skannaus + lahi-pesa-homografia)...")
    # k1 hyvaksytaan kun se parantaa lahipesan RMS:aa > 8 %. Rajatapauksessa (esim. 8.2 %) huonosti maaraytynyt
    # k1 vie 28 m paahan ekstrapoloidun kaukaisen pesan hakualueen ulkopuolelle -> kalibrointi kaatui. Jos kaukaista
    # pesaa ei loydy k1:lla, yritetaan ilman vaaristymakorjausta (k1 = 0). Kun k1:lla loytyy, tulos on ennallaan.
    k1_candidates = [best_k1] + ([0.0] if best_k1 != 0.0 else [])
    for k1_try in k1_candidates:
        dist_coeffs = np.array([k1_try, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        frame_undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs)
        near_pts_frame = kp.undistort_points_px(np.array(near_img_pts, dtype=np.float64), camera_matrix, k1_try)
        try:
            H_v1, output_w, output_h = build_far_house_center_seed(frame_undistorted, near_pts_frame, near_phys_pts)
        except RuntimeError as e:
            if k1_try == k1_candidates[-1]:
                raise
            print(f"  {e} -> uusi yritys ilman vaaristymakorjausta (k1 {k1_try:.3f} -> 0)")
            continue
        best_k1 = k1_try
        break

    print("  Geometrinen tarkennus (varipohjainen kaukainen rengas + harmaasavy-hoglinet)...")
    best = refine_geometric_homography_color(frame_undistorted, H_v1, near_pts_frame, near_phys_pts, output_w, output_h)
    print(f"  Hoglinet nimellisessa paikassa: RMS = {best['rms']:.3f} cm, {kp.house_quality_str(best['quality'])}")

    # hoglinjojen paikka voi poiketa nimellisesta +-A.KALIB_HOGLINJA_TOLERANSSI_CM. Sovitetaan myos versio, jossa
    # hogline saa olla missa tahansa toleranssin sisalla; se valitaan, jos pesien pyoreys/koko ei huonone
    # (kp.candidate_is_better, sama saanto kuin kalibroinnissa muutenkin). Kaukainen hogline on 30 m paassa ja
    # sen paikka maaraytyy kuvasta heikosti -> jos vapaampi sovitus huonontaa pesia, pidetaan nimellinen.
    hog_y = [rata.NOMINAL_NEAR_HOGLINE_Y_CM, rata.NOMINAL_FAR_HOGLINE_Y_CM]
    tol = A.KALIB_HOGLINJA_TOLERANSSI_CM
    if tol > 0.0:
        free = refine_geometric_homography_color(
            frame_undistorted,
            H_v1,
            near_pts_frame,
            near_phys_pts,
            output_w,
            output_h,
            hog_tol_cm=tol,
            label=f"hogline +-{tol:.0f} cm",
        )
        free_y = list(_hogline_targets(free["H_final"], free["near_hog_pts"], free["far_hog_pts"]))
        accept, _, _ = kp.candidate_is_better(
            frame_undistorted, free["H_final"], free["rms"], frame_undistorted, best["H_final"], best["rms"]
        )
        print(
            f"  Hoglinet +-{tol:.0f} cm: RMS = {free['rms']:.3f} cm, "
            f"{kp.house_quality_str(free['quality'])}, lahi {free_y[0] - hog_y[0]:+.1f} cm, "
            f"kauko {free_y[1] - hog_y[1]:+.1f} cm -> {'VALITAAN' if accept else 'hylataan (pesat eivat parane)'}"
        )
        if accept:
            best, hog_y = free, free_y

    print(
        f"  Valittu H: RMS = {best['rms']:.3f} cm, {kp.house_quality_str(best['quality'])} | hoglinet: "
        f"lahi {hog_y[0]:.1f} cm ({hog_y[0] - rata.NOMINAL_NEAR_HOGLINE_Y_CM:+.1f}), "
        f"kauko {hog_y[1]:.1f} cm ({hog_y[1] - rata.NOMINAL_FAR_HOGLINE_Y_CM:+.1f})"
    )
    rata.set_hogline_positions(*hog_y)

    return {
        "hogline_y_cm": tuple(hog_y),
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


# ============================================================
# TAYSI 3D-KAMERAMALLI: H_final -> (K, R, t)
#
# H_final (kamera8_01.py) kuvaa oikaistun raakakuvan pikselit output-
# px:ksi, joka itse on VAKIO affiini kuvaus fyysisesta (X,Y) cm-tasosta
# (katso to_output_px). Yhdistamalla nama saadaan TASO-HOMOGRAFIA
# H_plane_to_frame joka kuvaa fyysisen (X,Y,0)-tason SUORAAN oikaistun
# kuvan pikseleiksi - tasmalleen sellainen homografia jota standardi
# kamerakalibroinnissa puretaan poosiksi (Zhang-tyylinen dekompositio:
# H = K [r1 r2 t] skaalaan asti).
#
# Tama YKSIN ei kuitenkaan riita tarkaksi 3D-malliksi (validoitu
# testatessa: karkealla kameramatriisilla dekompositio jattaa
# 15-30 px jaannosvirheen) - kameramatriisin K fokaalivali (f) on vain
# karkea arvaus (katso kamera8_01.py:n build_camera_matrix). Siksi f
# haetaan TASTA UUDELLEEN (haku, ei arvaus) ja koko poosi (R, t)
# hienosaadetaan PAINOTETULLA epalineaarisella sovituksella SAMOILLA
# havainnoilla (lahempi pesa 17 pistetta, kaukaisen pesan rengasrajoite,
# hogline-rajoite) joita kamera8_01.py jo kaytti H_final:in sovitukseen
# - katso kamera8_01.py:n alkupaan kommentti periaatteesta.
# ============================================================


def _physical_to_frame_homography(calib):
    """
    H_plane_to_frame: fyysinen (X,Y,1) cm-taso -> oikaistun kuvan
    pikselit (frame_undistorted). Katso kommentti ylla.
    """

    A = np.array(
        [
            [rata.PIXELS_PER_CM, 0.0, -rata.OUTPUT_X_MIN_CM * rata.PIXELS_PER_CM],
            [0.0, -rata.PIXELS_PER_CM, rata.OUTPUT_Y_MAX_CM * rata.PIXELS_PER_CM],
            [0.0, 0.0, 1.0],
        ]
    )

    return np.linalg.inv(calib["H_final"]) @ A


def _decompose_planar_homography(H_plane_to_frame, camera_matrix):
    """
    Zhang-tyylinen taso-homografian purku poosiksi: H = K [r1 r2 t]
    skaalaan asti. r1,r2 normalisoidaan yksikkopituisiksi (keskiarvo,
    koska kohinan takia ne eivat ole tasan yhta pitkia), ja lahin
    AIDOSTI ortonormaali (r1,r2,r3) haetaan SVD:lla (standarditemppu -
    ilman tata pieni kohina rikkoisi rotaatiomatriisin ortogonaalisuuden).

    Palauttaa (R, t) - EI VIELA merkkivarmistettu (katso
    _ensure_camera_above_ice), koska taso-homografialla on aina
    kaksitahoinen "kumpi puoli tasoa" -epamaarayys jota pelkat Z=0-
    havainnot eivat voi ratkaista.
    """

    M = np.linalg.inv(camera_matrix) @ H_plane_to_frame

    norm1 = np.linalg.norm(M[:, 0])
    norm2 = np.linalg.norm(M[:, 1])
    scale = 2.0 / (norm1 + norm2)

    r1 = M[:, 0] * scale
    r2 = M[:, 1] * scale
    t = M[:, 2] * scale

    U, _, Vt = np.linalg.svd(np.column_stack([r1, r2, np.cross(r1, r2)]))
    R = U @ Vt

    return R, t


def _ensure_camera_above_ice(R, t):
    """
    Taso-homografialla on aina kaksi yhta hyvin Z=0-havaintoihin
    sopivaa ratkaisua: (R,t) ja (R',t') = ([-r1,-r2,r3], -t) - nama
    antavat TASMALLEEN samat projisoinnit Z=0-pisteille (todistettavissa:
    p_cam' = -p_cam, ja projisointi on muuttumaton koko vektorin
    etumerkin vaihdolle), mutta vain toinen vastaa fyysista todellisuutta
    (kamera on jaan YLAPUOLELLA, Z>0). Valitaan se kumpi antaa
    kameran maailmansijainnin (C = -R^T t) Z-koordinaatiksi positiivisen.
    """

    C = -R.T @ t

    if C[2] < 0:
        R = R.copy()
        R[:, 0] = -R[:, 0]
        R[:, 1] = -R[:, 1]
        t = -t

    return R, t


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

        K_try = np.array([[f_try, 0.0, camera_matrix0[0, 2]], [0.0, f_try, camera_matrix0[1, 2]], [0.0, 0.0, 1.0]])

        ok, rvec, tvec = cv2.solvePnP(
            obj_pts,
            img_pts,
            K_try,
            np.zeros(5),
            rvec0.copy(),
            t0.reshape(3, 1).copy(),
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
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


def fit_full_camera_pose(
    calib, near_pts_frame, near_phys, far_pts, far_radius, far_weight, hog_pts, hog_line_idx, hog_weight
):
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

        u, v = kivimalli.project_3d(K, R, t, obj_pts)
        parts.append((np.column_stack([u, v]) - near_pts_frame).ravel())

        if len(far_pts):
            X, Y = kivimalli.ray_plane_intersection(K, R, t, far_pts[:, 0], far_pts[:, 1], 0.0)
            dist = np.hypot(X, Y - rata.FAR_HOUSE_Y_CM)
            parts.append((dist - far_radius) * far_weight)

        if len(hog_pts):
            X, Y = kivimalli.ray_plane_intersection(K, R, t, hog_pts[:, 0], hog_pts[:, 1], 0.0)
            target = np.where(hog_line_idx == 0, rata.NEAR_HOGLINE_Y_CM, rata.FAR_HOGLINE_Y_CM)
            parts.append((Y - target) * hog_weight)

        return np.concatenate(parts)

    params0 = np.concatenate([rvec1.flatten(), tvec1.flatten()])
    params_final = kp._levenberg_marquardt(residuals, params0, max_iterations=100)

    R, _ = cv2.Rodrigues(params_final[0:3])
    t = params_final[3:6]
    R, t = _ensure_camera_above_ice(R, t)

    u, v = kivimalli.project_3d(K, R, t, obj_pts)
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

    far_pts, far_radius = kp.far_house_ring_points_in_frame(topdown_raw, H_final)
    near_hog, near_hog_w = kp.hogline_points_in_frame(topdown_raw, rata.NEAR_HOGLINE_Y_CM, H_final)
    far_hog, far_hog_w = kp.hogline_points_in_frame(topdown_raw, rata.FAR_HOGLINE_Y_CM, H_final)

    hog_pts = np.concatenate([near_hog, far_hog]) if (len(near_hog) or len(far_hog)) else np.zeros((0, 2))
    hog_line_idx = np.concatenate([np.zeros(len(near_hog), dtype=np.int64), np.ones(len(far_hog), dtype=np.int64)])
    hog_strength = np.concatenate([near_hog_w, far_hog_w])

    far_resid_now = kp._far_ring_distance(H_final, far_pts, far_radius)
    far_weight = 1.0 / kp._robust_scale_cm(far_resid_now, kp.NEAR_TRUST_FLOOR_CM)

    # Hoglinen kiintea, taysi paino - katso kamera8_01.py:n kommentti
    # (l' = H^-T l pitaa maarata kiertoa, ei saa vaientua).
    hog_weight = hog_strength / kp.NEAR_TRUST_FLOOR_CM

    return fit_full_camera_pose(
        calib,
        calib["near_pts_frame"],
        calib["near_phys"],
        far_pts,
        far_radius,
        far_weight,
        hog_pts,
        hog_line_idx,
        hog_weight,
    )
