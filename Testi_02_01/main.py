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
# KAUKAISEN PESAN LOYTAMINEN + KOKO HOMOGRAFIAN RATKAISU
# (Testi_01_02, kayttajan pyynnosta - korvaa Testi_01_01:n mallipohjaisen
# COARSE/FINE/ULTRA-haun, joka osoittautui liian herkaksi skaala-syvyys-
# degeneraatiolle ja mainospaneelien vaareille osumille tallä kamera-
# kulmalla)
#
# Uusi menetelma (kehitetty/validoitu askel askeleelta yhdessa kayttajan
# kanssa Testi_01_01:n pohjalta, katso sen keskusteluhistoria):
#
# 1) Rakennetaan homografia (H_near_only) PELKASTAAN lahemman pesan 17
#    pisteesta (near_house_correspondences), laajennettuna kattamaan
#    koko rata + 10m takarajan yli. Tama ekstrapolointi on tiedetysti
#    epatarkka kaukana (Y-suunnassa), mutta X-suunnassa (sivuttain)
#    riittavan luotettava karkeaan paikannukseen.
# 2) Skannataan tama laajennettu topdown rivi kerrallaan: etsitaan rivi
#    jolla on sinista maskia keskiviivan MOLEMMIN puolin ja punaista
#    niiden VALISSA - tama kuvio loytyy vain kaukaisen pesan (ja
#    lahemman pesan, joka suljetaan pois) kohdalta, ei muualta radalta.
#    Loydettyjen rivien suurin yhtenainen klusteri = karkea arvio
#    kaukaisen pesan Y-sijainnista (ja sen X-keskikohdasta).
# 3) "Skaalataan" (affiini pystysuora venytys/puristus, ankkuroituna
#    lahempaan pesaan Y=0:ssa) laajennettu topdown niin etta loydetty
#    rivi osuu TARKALLEEN sille riville jolla FAR_HOUSE_Y_CM pitaisi
#    olla - karkea arvio nayttaa taten OIKEASSA fyysisessä Y-kohdassa,
#    vaikka nain saatu asteikko onkin viela epatarkka.
# 4) Muodostetaan alustava homografia (H_v1) lahemman pesan 17 pisteesta
#    + tasta loydetyn kaukaisen pesan KESKIPISTEESTA (fyysinen oletus:
#    (0, FAR_HOUSE_Y_CM), koska curling-saannot pakottavat pesan
#    keskiviivalle tasan tunnetulle etaisyydelle).
# 5) H_v1:sta alkaen ajetaan sama geometrinen tarkennussilmukka jota
#    kamera8_01.py:n refine_geometric_homography jo kayttaa (ympyra-
#    rajoitteet kaukaiselle renkaalle + viivarajoitteet molemmille
#    hoglineille, ratkaistuna solve_homography_geometric:illa) - MUTTA
#    kaukaisen renkaan tunnistus KORVATTU varipohjaisella (HSV-pinta-
#    alaosuuteen saadetty kynnys + sateittainen reunanhaku, katso
#    far_ring_points_color_based) ja hoglinen tunnistus KORVATTU
#    harmaasavy-kynnys-menetelmalla (katso hogline_points_gray_
#    threshold) - molemmat kayttajan kanssa erikseen todettu
#    luotettavammiksi talla kamerakulmalla kuin kamera8_01.py:n omat
#    (radiaalihaku+pakotettu-yhteismuoto, segmenttipohjainen hogline-
#    haku).
#
# Validoitu (2026-09) molemmilla testikuvilla (leikattu_kalibrointi_
# moodikuva.png, 00011_kalibrointi_moodikuva.png): molempien hoglinien
# kulmaero < 0.02 astetta (aiemmin jopa 47 astetta vaarin), kaukaisen
# pesan pyoreys 0.94-1.00, koko-virhe alle 3%.
# ============================================================

FAR_HOUSE_ROW_SCAN_MAX_GAP_PX = 15
FAR_HOUSE_ROW_SCAN_CENTER_MARGIN_PX = 5
FAR_HOUSE_ROI_TARGET_BLUE_FRACTION = 0.31
FAR_HOUSE_ROI_TARGET_RED_FRACTION = 0.11
HOGLINE_GRAY_HALF_BAND_CM = 50.0
HOGLINE_GRAY_TARGET_COVERAGE = 0.98


def _extended_canvas_size(extended_y_max_cm):
    output_w = int(round((k8.OUTPUT_X_MAX_CM - k8.OUTPUT_X_MIN_CM) * k8.PIXELS_PER_CM))
    output_h = int(round(extended_y_max_cm * k8.PIXELS_PER_CM))
    return output_w, output_h


def _physical_to_output_px_extended(physical_pts, extended_y_max_cm):
    pts = np.asarray(physical_pts, dtype=np.float64)
    ox = (pts[:, 0] - k8.OUTPUT_X_MIN_CM) * k8.PIXELS_PER_CM
    oy = (extended_y_max_cm - pts[:, 1]) * k8.PIXELS_PER_CM
    return np.column_stack([ox, oy])


def _to_output_px_extended(x_cm, y_cm, extended_y_max_cm):
    px = (x_cm - k8.OUTPUT_X_MIN_CM) * k8.PIXELS_PER_CM
    py = (extended_y_max_cm - y_cm) * k8.PIXELS_PER_CM
    return px, py


def _compute_crop_row_range_extended(full_height_px, center_y_cm, half_height_cm, extended_y_max_cm):
    """KRIITTINEN BUGIKORJAUS (kayttajan pyynnosta - katso keskustelu-
    historia): k8.compute_crop_row_range/crop_house_view/expected_house_
    center_in_crop kayttavat SISAISESTI k8:n omaa to_output_px:aa, joka
    on kiinteasti sidottu STANDARDIIN k8.OUTPUT_Y_MAX_CM:aan (4000cm) -
    EI kelpaa taman tiedoston LAAJENNETULLE (10m takarajan yli
    ulottuvalle, eri korkuiselle) kanvaasille. Naiden suora kayttö
    laajennetulla kuvalla antoi VAARAN rivialueen (havaittu: koko
    kolmen ellipsin sovitus epaonnistui systemaattisesti koska crop
    osui aivan vaarille riveille, nakyi mustana kuvana). Tama on sama
    laskukaava mutta parametrisoituna oikealla Y-max:lla."""
    _, row_a = _to_output_px_extended(0.0, center_y_cm + half_height_cm, extended_y_max_cm)
    _, row_b = _to_output_px_extended(0.0, center_y_cm - half_height_cm, extended_y_max_cm)
    row_top = int(max(0, math.floor(min(row_a, row_b))))
    row_bottom = int(min(full_height_px - 1, math.ceil(max(row_a, row_b))))
    return row_top, row_bottom


def _crop_house_view_extended(topdown, center_y_cm, half_height_cm, extended_y_max_cm):
    row_top, row_bottom = _compute_crop_row_range_extended(
        topdown.shape[0], center_y_cm, half_height_cm, extended_y_max_cm
    )
    return topdown[row_top:row_bottom + 1, :].copy()


def _expected_house_center_in_crop_extended(full_height_px, center_y_cm, half_height_cm, extended_y_max_cm):
    row_top, _ = _compute_crop_row_range_extended(full_height_px, center_y_cm, half_height_cm, extended_y_max_cm)
    cx, cy_full = _to_output_px_extended(0.0, center_y_cm, extended_y_max_cm)
    return (cx, cy_full - row_top)


def _find_blue_red_blue_pattern(blue_mask, red_mask, center_col, row_range=None,
                                 margin_px=FAR_HOUSE_ROW_SCAN_CENTER_MARGIN_PX,
                                 max_gap_px=FAR_HOUSE_ROW_SCAN_MAX_GAP_PX):
    """YDINTARKISTUS koko taman tiedoston kaukaisen pesan loytamiselle
    (kayttajan pyynnosta - katso keskusteluhistoria): etsii rivit joilla
    on sinista maskia center_col:in MOLEMMIN puolin JA punaista niiden
    VALISSA - tama kuvio on riittavan erikoislaatuinen etta se EI osu
    esim. mainospaneeleihin tai muihin sinisiin/punaisiin kohteisiin
    jotka eivat ole oikeasti rengasmaisia (todettu ja korjattu kehitys-
    vaiheessa: pelkka "suurin sininen kontuuri" -haku tarttui toistuvasti
    vaariin kohteisiin). Kaytetaan seka koko-kuvan karkeaan hakuun etta
    paikalliseen tarkennukseen/vahvistukseen (samalla funktiolla - sama
    tarkistus, vain eri hakualue). Palauttaa (found_row, found_col)
    suurimman loydetyn rivi-klusterin keskikohtana, tai None jos mitaan
    riittavan pitkaa yhtenaista kuviota ei loydy."""

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

    blue = k8.create_blue_mask(topdown_extended)
    red = k8.create_red_mask(topdown_extended)
    h, w = blue.shape
    center_col = int(round((0.0 - k8.OUTPUT_X_MIN_CM) * k8.PIXELS_PER_CM))
    near_house_exclude_row = int(h - (k8.HOUSE_RADIUS_CM + 100.0) * k8.PIXELS_PER_CM)

    return _find_blue_red_blue_pattern(blue, red, center_col, row_range=(0, near_house_exclude_row))


def _fit_circle_algebraic(points):
    """Yksinkertainen algebrallinen ympyrasovitus (Kasa) pistejoukkoon.
    Palauttaa (cx, cy, r) tai None jos pisteita on liian vahan."""
    if len(points) < 5:
        return None
    x, y = points[:, 0], points[:, 1]
    A = np.column_stack([x, y, np.ones_like(x)])
    b = x ** 2 + y ** 2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = sol[0] / 2.0, sol[1] / 2.0
    r = math.sqrt(max(sol[2] + cx ** 2 + cy ** 2, 0.0))
    return cx, cy, r


def _three_ellipse_fit_center_in_crop(crop, center_col_hint, search_radius_px=None):
    """YDINSOVITUS (kayttajan pyynnosta - katso keskusteluhistoria):
    vahvistaa etta cropista loytyy sininen-punainen-sininen -kuvio
    (_find_blue_red_blue_pattern) annetun center_col_hint:in ymparilta,
    rakentaa sen ymparille ROI:n (pienin ymparoiva ympyra), saataa
    sinisen/punaisen HSV-kynnyksen pinta-alaosuuteen (FAR_HOUSE_ROI_
    TARGET_*_FRACTION), ja sovittaa ympyran KOLMELLE renkaalle (sininen
    ulko/sisa, punainen ulko) SATEITTAISELLA reunanhaulla (_radial_
    ring_edges - kerää pisteita molemmista nakyvista kaarista
    symmetrisesti, toisin kuin yksittainen suurin-kontuuri-haku joka
    voi tarttua vain YHTEEN pirstoutuneeseen renkaan palaan ja antaa
    vinon keskipisteen). Palauttaa kolmen sovituksen keskipisteiden
    KESKIARVON crop-paikallisissa koordinaateissa, tai None jos kuviota
    ei loydy tai yhtaan ympyraa ei saada sovitettua."""

    blue_raw = k8.create_blue_mask(crop)
    red_raw = k8.create_red_mask(crop)
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

    back_line_y_cm = k8.FAR_HOUSE_Y_CM + k8.HOUSE_RADIUS_CM
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

    # "Skaalataan" (pystysuora affiini venytys/puristus, ankkuroituna
    # lahempaan pesaan Y=0:ssa/ext_h:ssa) KOKO laajennettu topdown-KUVA
    # niin etta loydetty rivi osuu tarkalleen FAR_HOUSE_Y_CM:n kohdalle -
    # TARKALLEEN sama jarjestys kuin kasin tehdyssa prosessissa (katso
    # keskusteluhistoria): tama tehdaan KUVALLE, EI vain yhdelle
    # pisteelle, koska kolmen ellipsin sovitus PITAA tehda jo suunnilleen
    # oikein skaalatussa TOPDOWN-avaruudessa (missa rengas nayttaa jo
    # suunnilleen ympyralta) - EI takaisinprojisoituna vaaristyneeseen
    # KEHYSAVARUUTEEN (missa kaukainen pieni rengas nakyy hyvin ohuena/
    # venyneena ellipsina ja sateittainen reunanhaku/pattern-tarkistus
    # toimii epaluotettavammin, todettu kehitysvaiheessa: leikattu-kuvan
    # tarkistus epaonnistui systemaattisesti kehysavaruudessa).
    target_row = (extended_y_max_cm - k8.FAR_HOUSE_Y_CM) * k8.PIXELS_PER_CM
    anchor_row = float(ext_h)
    scale_factor = (anchor_row - target_row) / (anchor_row - found_row)
    a = scale_factor
    b = anchor_row * (1.0 - scale_factor)
    rescale_M = np.array([[1.0, 0.0, 0.0], [0.0, a, b]], dtype=np.float64)
    topdown_rescaled_ext = cv2.warpAffine(topdown_extended, rescale_M, (ext_w, ext_h))

    # Rivi<->Y-suhde on nyt (affiinin rakentamistavan ansiosta) sama
    # VAKIOKAAVA koko kuvan matkalta kuin standardikanvaasilla - katso
    # keskusteluhistorian perustelu. HUOM (loydetty ja korjattu kayttajan
    # pyynnosta nayttaessa jokaisen vaiheen kuvat): k8.compute_crop_row_
    # range/crop_house_view/expected_house_center_in_crop kayttavat
    # SISAISESTI k8:n OMAA to_output_px:aa, joka on kiinteasti sidottu
    # STANDARDIIN k8.OUTPUT_Y_MAX_CM:aan (4000cm) - EIVAT kelpaa tälle
    # LAAJENNETULLE (eri korkuiselle) kanvaasille sellaisenaan. Kaytetaan
    # siis omia extended-versioita (_compute_crop_row_range_extended jne,
    # parametrisoitu oikealla extended_y_max_cm:lla).
    far_row_top_ext, _ = _compute_crop_row_range_extended(
        ext_h, k8.FAR_HOUSE_Y_CM, k8.HOUSE_CROP_HALF_HEIGHT_CM, extended_y_max_cm
    )
    far_crop_rescaled = _crop_house_view_extended(
        topdown_rescaled_ext, k8.FAR_HOUSE_Y_CM, k8.HOUSE_CROP_HALF_HEIGHT_CM, extended_y_max_cm
    )
    expected_center_local = _expected_house_center_in_crop_extended(
        ext_h, k8.FAR_HOUSE_Y_CM, k8.HOUSE_CROP_HALF_HEIGHT_CM, extended_y_max_cm
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

    output_w = int(round((k8.OUTPUT_X_MAX_CM - k8.OUTPUT_X_MIN_CM) * k8.PIXELS_PER_CM))
    output_h = int(round(k8.OUTPUT_Y_MAX_CM * k8.PIXELS_PER_CM))

    all_img_pts = list(near_pts_frame) + [far_center_frame]
    all_phys_pts = list(near_phys_pts) + [(0.0, k8.FAR_HOUSE_Y_CM)]
    src_v1 = np.array(all_img_pts, dtype=np.float32)
    dst_v1 = k8.physical_to_output_px(np.array(all_phys_pts, dtype=np.float64)).astype(np.float32)
    H_v1, _ = cv2.findHomography(src_v1, dst_v1, method=0)

    return H_v1, output_w, output_h


def _hsv_blue_mask_s_low(frame, s_low, v_low=40):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv, np.array([90, s_low, v_low], dtype=np.uint8), np.array([140, 255, 255], dtype=np.uint8)
    )
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
    """Sama sopimus kuin k8.far_house_ring_points_in_frame (palauttaa
    (points, radii) oikaistun kuvan koordinaatistossa geometrista
    tarkennusta - k8.solve_homography_geometric - varten), mutta
    tunnistus tehdaan VARIPOHJAISELLA (HSV) menetelmalla k8:n omaan
    radiaalihaku+pakotettu-yhteismuoto-sovitukseen sijaan: pinta-
    alaosuuteen (ROI = sinisen maskin pienin ymparoiva ympyra) saadettu
    Saturation-kynnys + sateittainen reunanhaku molemmille renkaille
    (sininen ulko/sisa, punainen ulko). Kayttajan kanssa todettu
    tuottavan visuaalisesti tarkemman (oikean kokoisen) sovituksen
    kuin k8:n oma menetelma talla kamerakulmalla."""

    far_row_top, _ = k8.compute_crop_row_range(
        topdown_raw.shape[0], k8.FAR_HOUSE_Y_CM, k8.HOUSE_CROP_HALF_HEIGHT_CM
    )
    view = k8.crop_house_view(topdown_raw, k8.FAR_HOUSE_Y_CM, k8.HOUSE_CROP_HALF_HEIGHT_CM)

    # Sama vahvistus kuin _three_ellipse_fit_center_in_crop:issa
    # (kayttajan pyynnosta): ei luoteta pelkkaan "suurin sininen alue"
    # -oletukseen, koska nykyinen H_current voi silla hetkella olla
    # viela riittavan vino etta crop osuu vaaraan kohteeseen (esim.
    # mainospaneeliin) - vahvistetaan ETTA sininen-punainen-sininen
    # -kuvio loytyy paikallisesti ENNEN kuin renkaan pisteita palautetaan
    # geometriselle ratkaisijalle. Jos kuviota ei loydy, palautetaan
    # TYHJA (turvallisempi kuin vaara rengas - silloin ratkaisija
    # nojaa vain lahempaan pesaan + hoglineihin talla kierroksella).
    blue_raw = k8.create_blue_mask(view)
    red_raw = k8.create_red_mask(view)
    center_col = int(round((0.0 - k8.OUTPUT_X_MIN_CM) * k8.PIXELS_PER_CM))
    pattern = _find_blue_red_blue_pattern(blue_raw, red_raw, center_col)
    if pattern is None:
        return np.zeros((0, 2)), np.zeros(0)
    pattern_row, pattern_col = pattern

    ys, xs = np.where(blue_raw > 0)
    if len(xs) < 5:
        return np.zeros((0, 2)), np.zeros(0)
    pts_all = np.column_stack([xs, ys]).astype(np.float32)
    dists = np.hypot(pts_all[:, 0] - pattern_col, pts_all[:, 1] - pattern_row)
    near_pattern = pts_all[dists <= k8.HOUSE_CROP_HALF_HEIGHT_CM * k8.PIXELS_PER_CM * 0.6]
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
        (blue_inner_pts, k8.BLUE_INNER_RADIUS_CM),
        (blue_outer_pts, k8.BLUE_OUTER_RADIUS_CM),
        (red_outer_pts, k8.RED_OUTER_RADIUS_CM),
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

    points = k8.frame_points_from_topdown(np.array(topdown_pts, dtype=np.float64), H_current)
    return points, np.array(radii, dtype=np.float64)


def hogline_points_gray_threshold(topdown_raw, hogline_y_cm, H_current,
                                   half_band_cm=HOGLINE_GRAY_HALF_BAND_CM):
    """Sama sopimus kuin k8.hogline_points_in_frame (palauttaa (points,
    weights) oikaistun kuvan koordinaatistossa), mutta harmaasavy-
    kynnys-menetelmalla: haetaan nimellisen hogline-rivin ymparilta
    (half_band_cm) kynnysarvo (harmaasavy < t) joka antaa YHTENAISEN
    (lahes 100% sarakepeittavyyden) tumman kaistan koko leveydelta -
    kayttajan kanssa todettu paljon luotettavammaksi kuin k8:n oma
    segmenttipohjainen detect_hogline_points talla kamerakulmalla
    (joka loysi usein vain muutaman hajanaisen pisteen tai tarttui
    sponsoritekstin kontaminaatioon)."""

    _, nominal_row = k8.to_output_px(0.0, hogline_y_cm)
    half_px = int(half_band_cm * k8.PIXELS_PER_CM)
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
        cols_with_dark = np.count_nonzero(np.any(mask[:, margin:gray.shape[1] - margin] > 0, axis=0))
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

    points = k8.frame_points_from_topdown(topdown_pts, H_current)
    return points, weights_norm


def refine_geometric_homography_color(frame_undistorted, H_init, near_pts, near_phys,
                                       output_w, output_h,
                                       max_iterations=None, min_relative_improvement=None):
    """Sama iteratiivinen konvergenssiperiaate kuin k8.refine_geometric_
    homography (katso sen kommentti), mutta kaukaisen renkaan ja
    hoglinien tunnistus jokaisella kierroksella tehdaan tama tiedoston
    varipohjaisilla/harmaasavypohjaisilla funktioilla (far_ring_points_
    color_based, hogline_points_gray_threshold) k8:n omien sijaan.
    Palauttaa dictin: H_final, topdown_raw, rms, quality, iterations."""

    if max_iterations is None:
        max_iterations = k8.HOMOGRAPHY_REFINE_MAX_ITERATIONS
    if min_relative_improvement is None:
        min_relative_improvement = k8.HOMOGRAPHY_REFINE_MIN_RELATIVE_IMPROVEMENT

    STALL_PATIENCE = 2
    H_current = H_init
    best = None
    best_rms = float("inf")
    stall_count = 0

    for iteration in range(1, max_iterations + 1):

        topdown_raw = cv2.warpPerspective(frame_undistorted, H_current, (output_w, output_h))

        far_pts, far_radius = far_ring_points_color_based(topdown_raw, H_current)
        near_hog_pts, near_hog_w = hogline_points_gray_threshold(topdown_raw, k8.NEAR_HOGLINE_Y_CM, H_current)
        far_hog_pts, far_hog_w = hogline_points_gray_threshold(topdown_raw, k8.FAR_HOGLINE_Y_CM, H_current)

        near_angle_now, near_conf_now = k8.robust_line_angle_from_points(
            [(p[0], p[1], w) for p, w in zip(near_hog_pts, near_hog_w)]
        ) if len(near_hog_pts) else (0.0, 0.0)
        far_angle_now, far_conf_now = k8.robust_line_angle_from_points(
            [(p[0], p[1], w) for p, w in zip(far_hog_pts, far_hog_w)]
        ) if len(far_hog_pts) else (0.0, 0.0)

        if near_conf_now > 0.0 and far_conf_now > 0.0 and abs(far_angle_now - near_angle_now) > 5.0:
            far_hog_pts, far_hog_w = np.zeros((0, 2)), np.zeros(0)

        hog_pts = np.concatenate([near_hog_pts, far_hog_pts]) if (
            len(near_hog_pts) or len(far_hog_pts)
        ) else np.zeros((0, 2))
        hog_y = np.concatenate([
            np.full(len(near_hog_pts), k8.NEAR_HOGLINE_Y_CM),
            np.full(len(far_hog_pts), k8.FAR_HOGLINE_Y_CM),
        ])
        hog_strength = np.concatenate([near_hog_w, far_hog_w])

        far_resid_now = k8._far_ring_distance(H_current, far_pts, far_radius)
        far_scale = k8._robust_scale_cm(far_resid_now, k8.NEAR_TRUST_FLOOR_CM)
        far_weight = 1.0 / far_scale
        hog_weight = hog_strength / k8.NEAR_TRUST_FLOOR_CM

        H_new = k8.solve_homography_geometric(
            H_current, near_pts, near_phys, far_pts, far_radius, far_weight,
            hog_pts, hog_y, hog_weight
        )

        near_proj = k8.output_px_to_physical(k8._apply_h(H_new, near_pts))
        rms = math.sqrt(float(np.mean(np.sum((near_proj - near_phys) ** 2, axis=1))))
        quality = k8.measure_house_quality(frame_undistorted, H_new)

        print(
            f"[Geometrinen korjaus (varipohjainen) {iteration}/{max_iterations}] "
            f"lahempi RMS: {rms:.4f} cm, {k8.house_quality_str(quality)} "
            f"(kaukaisen renkaan pisteita {len(far_pts)}, paino {far_weight:.3f}; "
            f"hogline-pisteita {len(hog_pts)}, lahi/kauko-kulma "
            f"{near_angle_now:.2f}/{far_angle_now:.2f})"
        )

        if best is not None:
            accept, _, _ = k8.candidate_is_better(
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

    far_ref = k8.project_point(0.0, 1000.0, camera, v_t, v_cl)
    dir_forward = np.asarray(far_ref, dtype=np.float64) - np.asarray(house_center, dtype=np.float64)
    dir_forward /= np.linalg.norm(dir_forward)
    dir_lateral = np.array(v_t, dtype=np.float64)
    dir_lateral = dir_lateral / np.linalg.norm(dir_lateral)

    near_img_pts, near_phys_pts, near_labels = k8.near_house_correspondences(
        t_line, centerline, house_center, blue_outer, blue_inner, red_outer, red_inner,
        dir_lateral, dir_forward
    )

    camera_matrix = k8.build_camera_matrix(image_width, image_height)
    best_k1, best_k1_rms, baseline_rms = k8.estimate_radial_distortion_k1(
        near_img_pts, near_phys_pts, camera_matrix
    )
    k1_improvement = (
        (baseline_rms - best_k1_rms) / baseline_rms
        if baseline_rms > 1e-9 and math.isfinite(baseline_rms) else 0.0
    )
    if k1_improvement > k8.K1_MIN_RELATIVE_IMPROVEMENT and abs(best_k1) > k8.K1_MIN_MAGNITUDE:
        dist_coeffs = np.array([best_k1, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    else:
        best_k1 = 0.0
        dist_coeffs = np.zeros(5, dtype=np.float64)

    frame_undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs)
    near_pts_frame = k8.undistort_points_px(
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

    print(f"  Valittu H: RMS = {best['rms']:.3f} cm, {k8.house_quality_str(best['quality'])}")

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

# ============================================================
# GRANIITTIMASKIN TAUSTANVAIMENNUKSEN KYNNYSARVO (Testi_02_01, havaittu
# oikean Testivideo-julkaisun analyysissa - katso keskusteluhistoria):
# stone_tracker.cpp:n search_new_stone/track_stones_batch ottavat
# diff_threshold-parametrin, mutta main.py EI KOSKAAN antanut sita
# eksplisiittisesti ennen tata - kaytossa oli siis hiljaa C++:n oma
# OLETUSARVO 30.0 (katso PYBIND11-RAJAPINTA). Tama havaittiin LIIAN
# LOYHAKSI juuri lahemman pesan rengaskuvion (vaalea/tumma sinisten
# kolmioiden mosaiikki) kohdalla: diagnoosissa (kaksi oikeaa, koko
# matkan onnistuneesti seurattua kivea, molemmat menettivat SEURANNAN
# tasan pesan reunalla) graniittimaski oli arvolla 30 JOKO pirstoutunut
# ohuiksi viivoiksi TAI TAYSIN TYHJA (0 pikselia!) juuri niissa
# kohdissa missa kivi lepasi tumman kolmion paalla - koska diff_
# threshold=30 valkaisi (tulkitsi taustaksi) MYOS osan itse kivesta,
# ei vain oikeaa taustaa. Tiukempi (PIENEMPI) kynnys saattaa vain
# LAHELLA moodikuva-arvoa olevat pikselit taustaksi - kayttajan
# ehdottama korjaussuunta, vahvistettu vertailulla (+-3/+-6/+-10/+-20/
# +-30): arvolla 10 maski pysyi yhtenaisena molemmissa aidon kiven
# pysahdyskohdissa (725->1852px ja 0->809px), ilman etta+-3/+-6:n
# ylimaarainen kohina (94 vs 22 yhtenaista aluetta koko framessa) tulisi
# mukaan. HUOM: tama vaikuttaa SEKA HAKUun etta SEURANTAAN (molemmat
# saavat saman arvon alla) - HAKU on tasta herkempi (skannaa koko
# vyohykkeen joka sekunti, ei vain pientä paikallishakua kuten SEURANTA),
# joten jos vaarat HAKU-loydot lisaantyvat jatkossa, tata voi eriyttaa
# HAKU:lle omaksi (loyhemmaksi) arvokseen - MIN_CONFIRMED_THROW_
# DISPLACEMENT_CM (myohemmin tassa tiedostossa) antaa jo suojan naita
# vastaan.
# ============================================================
GRANITE_DIFF_THRESHOLD = 10.0

# ============================================================
# VARJONSIETOINEN TAUSTANVAIMENNUS + JOKA-FRAME SUB-PIKSELI-
# KOHDISTUS (kayttajan pyynnosta, katso keskusteluhistoria):
# empiirisesti havaittu etta paneiliseurantaan perustuva stabilointi
# EI ole taysin sub-pikseli-tarkka, ja etta myos VARJOSTUNEET (ei
# vain suoraan moodikuvaa vastaavat) pikselit kannattaa tulkita
# taustaksi - yhdessa nama auttoivat loytamaan JA pitamaan kiinni
# testivideon (0001.mp4, "Testivideo"-julkaisu) ENSIMMAISESTA
# heitosta, joka aluksi nakyy vain muutamana irrallisena pikselina
# pelaajan vierella (suurin osa kivesta pelaajan peitossa). Ilman
# tata HAKU ei loytanyt kyseista heittoa LAINKAAN; taman kanssa
# SEURANTA piti kiinni siita KOKO matkan (n. 22.7s) levolle asti -
# lopuksi jopa kiven kahva erottui selvasti, vahvistaen etta koko
# ajan seurattiin oikeaa kivea.
#
# HELPPO POISTAA KAYTOSTA: aseta ENABLE_SHADOW_TOLERANT_STABILIZATION
# = False - talloin varjonsietoinen taustanvaimennus palaa TASMALLEEN
# aiempaan kayttaytymiseen (raaka diff_threshold, ei varjosaantoa) -
# katso kayttokohta suppress_static_background():ssa.
#
# REHELLINEN VARAUS: tama on toistaiseksi validoitu vain YHDEN erittain
# hankalan (voimakkaasti okkludoidun) heiton osalta tarkalla, kasin
# ohjatulla diagnoosilla - EI VIELA koko videon lapikaynnilla A/B-
# vertailuna (kuten esim. GRANITE_DIFF_THRESHOLD/BODY_CONTOUR-poisto
# tassa samassa tiedostossa). Suositus: aja koko video lapi seka
# paalla etta pois paalta (tools/review_tracks.py) ennen kuin luotat
# tahan oikeassa ottelussa.
# ============================================================
ENABLE_SHADOW_TOLERANT_STABILIZATION = True

SHADOW_V_DROP_MAX = 30.0  # kuinka paljon V (HSV) saa pudota ja silti tulkita taustaksi/varjoksi

# ============================================================
# JOKA-FRAME SUB-PIKSELI-KOHDISTUS - OMA, ERILLINEN lippunsa (irrotettu
# ENABLE_SHADOW_TOLERANT_STABILIZATIONista, kayttajan pyynnosta: katso
# keskusteluhistoria). ENSIMMAINEN versio mittasi korjauksen VAIN
# yhdesta kiintesta pienesta ankkuripisteesta (kaukaisen pesan takana)
# ja sovelsi sen koko kuvaan jaykkana muunnoksena - koko videon
# lapikaynti paljasti etta tama HUONONSI seurannan vakautta muualla
# kuvassa (erityisesti lahella pesaa, kaukana ankkurista): pieni
# paikallinen korjaus ei ollut edustava koko kuvalle, ja kiertovirhe
# (kuvan keskipisteen ympari) vahvistui etaisyyden mukana.
#
# KORJATTU (kayttajan ehdotuksesta): kohdistus haetaan nyt KOKO NAYTON
# pikseleita vasten moodikuvaa vasten, taysresoluutioisena (kayttajan
# huomio: kuvan pienentaminen tuhoaisi juuri sub-pikseli-tarkkuuden
# jota haetaan). cv2.phaseCorrelate (FFT-pohjainen vaihekorrelaatio)
# laskee koko framen sub-pikseli-tarkan TRANSLAATION yhdella kutsulla -
# ei enaa ristikkohakua eika kiertoa (katso estimate_subpixel_alignment).
# ============================================================
ENABLE_SUBPIXEL_ALIGNMENT = False

SUBPIXEL_ALIGN_RANGE_PX = 1.0  # turvaraja - katso estimate_subpixel_alignment

# Kuinka suuri, kuvan keskelle keskitetty osuus (leveys JA korkeus)
# kaytetaan vaihekorrelaatioon - EI pienennys, vain RAJAUS (resoluutio
# sailyy taysimittaisena) - katso estimate_subpixel_alignment.
SUBPIXEL_ALIGN_CROP_FRACTION = 0.67

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
# = kaukainen hogline...hogline+3m, k92.SEARCH_X_HALF_WIDTH_CM=
# +-70cm keskiviivasta) - ei enaa tarvitse luottaa hitaaseen, harjasta
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

# ============================================================
# SEURANNAN SIETOKYKY HETKELLISELLE TAYDELLE PEITOLLE (Testi_02_01,
# kayttajan pyynnosta): kamera9_02.py:n oma TRACK_LOST_MAX_MISSES=5
# (0.2s 25fps:lla) on LIIAN LYHYT lahemman pesan luona - taman
# tiedoston OMA, jo aiemmin dokumentoitu havainto (katso COLOR_STOP_
# MAX_AVG_DIFF:in ylapuolinen kommentti) mainitsee mittaavan pelaajan
# peittaneen kiven "30+ ruutua/1.2s+" - SELVASTI pidempaan kuin 5
# ruutua. Koska HAKU (kamera9_02.py:n SEARCH_Y_MIN/MAX_CM) toimii VAIN
# kaukaisen paan vyohykkeella (katso taman tiedoston oma ASETUKSET-
# kommentti ylla: "kivet HEITETAAN aina samaan suuntaan/paahan"), kivi
# joka menettaa SEURANTANSA lahella pesaa EI VOI koskaan loytya
# uudelleen - se katoaa CSV:sta pysyvasti lopun liu'un/pysahtymisen
# ajaksi, vaikka peittava pelaaja/harja siirtyisi pois sekunnin
# sisalla. TRACK_LOST_MAX_MISSES=5 ei siis erota "hetkellinen taysi
# peitto" (pitaisi selvita, kivi on siina missa ennenkin - haku-ankkuri
# s["last_xy"] EI paivity misseilla, katso alempana SEURANTA-silmukka)
# tapauksesta "kivi todella poissa/vaara kandidaatti".
#
# PAIKALLINEN ylikirjoitus (EI muuteta kamera9_02.py:n omaa arvoa -
# sama periaate kuin HAKU_SEARCH_INTERVAL_FRAMES alla): nostetaan
# sallittu peraikkainen peitto TRACK_LOST_GRACE_SECONDS:iin asti,
# reilusti yli tuon dokumentoidun 1.2s+ havainnon ylapuolelle, jotta
# tavanomainen "pelaaja/harja kokonaan kiven edessa hetken" -tilanne ei
# enaa katkaise seurantaa. Kayttajan oma rajaus (aidosti PITKAKESTOINEN
# peitto, esim. koko lopun liu'un ajan joku seisoo kiven paalla, jaa
# vaistamatta havaitsematta) sailyy silti - kivi hylataan (kuten
# ennenkin) jos peitto kestaa PIDEMPAAN kuin tama reilu ikkuna.
# ============================================================

TRACK_LOST_GRACE_SECONDS = 3.0

# ============================================================
# NOPEUSRAJOITETTU SEURANTA-HAKUALUE + "EI TAAKSEPAIN" (kayttajan
# pyynnosta, katso keskusteluhistoria): SEURANTA:n ristikkohaku etsi
# aiemmin AINA kiintealta, isotrooppiselta alueelta (kamera9_02.py:n
# TRACK_HALF_RANGE_CM=35cm joka suuntaan) riippumatta siita kuinka
# kauan kivesta on todellisuudessa aikaa edellisesta havainnosta.
# Koska oikea curling-kivi EI VOI liikkua nopeammin kuin fysikaalinen
# yla­raja (arvioitu turvamarginaalilla: Y-suunnassa, radan pituus-
# suunnassa, TRACK_MAX_SPEED_Y_CM_S=4.5m/s - reilusti yli tyypillisen
# heittonopeuden ~2.5-3m/s; X-suunnassa, sivuttaisliike/curl, paljon
# hitaampaa, TRACK_MAX_SPEED_X_CM_S=0.6m/s), hakualue voidaan laskea
# SUORAAN nopeusrajasta JOKA framella JOKAISELLE kivelle erikseen:
#   half_range = max_speed_cm_s * (misses+1) / fps
# ("misses+1" framea sitten oli viimeisin VAHVISTETTU sijainti, koska
# s["last_xy"] EI paivity misseilla - katso TRACK_LOST_GRACE_SECONDS:in
# kommentti ylla) - haku ei siis KOSKAAN tarvitse etsia kauempaa, ja
# pitkan peiton (misses>0) jalkeen hakualue kasvaa automaattisesti
# oikeassa suhteessa. Katso kaytto SEURANTA-kutsun valmistelussa
# (build_track_half_ranges alempana) ja stone_tracker.cpp:n
# trackStoneUpdateOne (track_half_range_x_cm/track_half_range_y_cm,
# EI enaa yhta isotrooppista kamera9_02.py:n TRACK_HALF_RANGE_CM:ia).
#
# MAX_BACKWARD_CM: kivi liikkuu tassa videossa AINA Y:n pienetessa
# (katso Y-suunnan kommentti main.py:n alkupaassa) - aito liikkuva
# kivi hidastuu mutta EI KOSKAAN peruuta. Jos yhden SEURANTA-paivityksen
# tulos siirtaisi kiven yli metrin "taaksepain" (Y kasvaisi), kyseessa
# on lahes aina ristikkohaun ajautuminen vaaraan kohteeseen (esim.
# viereinen kivi/pelaaja) - hylataan (kuten oversized_reject, katso
# stone_tracker.cpp) sen sijaan etta hyvaksytaan virheellinen hyppy.
# ============================================================

TRACK_MAX_SPEED_Y_CM_S = 450.0
TRACK_MAX_SPEED_X_CM_S = 60.0
TRACK_MAX_BACKWARD_CM = 100.0

# YLARAJA nopeuspohjaiselle hakualueelle (havaittu VALTTAMATTOMAKSI
# koko videon lapikaynnilla, katso keskusteluhistoria): half_range =
# max_speed_cm_s * elapsed_s KASVAA RAJATTA pitkien miss-sarjojen
# aikana (TRACK_LOST_GRACE_SECONDS=3.0s asti) - esim. 3s peitolla
# half_range_y = 450*3 = 1350cm, mika teki ristikkohausta (ja sen ROI-
# leikkeesta) valtavan hitaan (mitattu: SEURANTA-kutsu keskimaarin yli
# 1s/kutsu koko videon lapikaynnissa, n. 15-20x hitaampi kuin ennen
# tata ominaisuutta - videon lapikaynti olisi kestanyt yli 2h 5min
# videolle). Katakkaa hakualue tahan, jotta pahin tapaus pysyy
# hallittavana - jos kivi ei loydy edes tallä alueella pitkan peiton
# jalkeen, se joka tapauksessa kadotetaan (TRACK_LOST_MAX_MISSES/
# TRACK_LOST_GRACE_SECONDS) ja loytyy tarvittaessa uudelleen HAKU:n
# kautta, joten rajaus ei heikenna luotettavuutta merkittavasti.
TRACK_HALF_RANGE_MAX_CM = 100.0

# ============================================================
# "UUSI KIVI" -REKISTEROINNIN VAHVISTUS LIIKKEELLA (Testi_02_01,
# havaittu oikean 5min Testivideo-julkaisun analyysissa): kivien
# Y-koordinaatti PIENENEE ajassa taman videon suunnistuksessa (heitto-
# paa/HAKU-vyohyke = suuri Y, lahempi pesa = pieni Y - VAHVISTETTU
# oikealla datalla: aidot, koko matkan onnistuneesti seuratut heitot
# nayttavat sileaa, HIDASTUVAA Y:n pienenemista n. 1.8-1.9 m/s:sta
# muutamaan kymmeneen cm/s:iin - fysikaalisesti oikea kitkahidastuvuus).
#
# HAVAITTU ONGELMA (samalla oikealla datalla, 13 rekisteroitya stone_
# id:ta ~5-6 aidon heiton sijaan): HAKU (kiintea kaukaisen paan
# vyohyke) loytaa saannollisesti UUDELLEEN saman jo LEVOSSA olevan
# (aiemmin jo raportoidun/pysahtyneen TAI peiton takia kadotetun)
# kiven tai muun paikallaan olevan kohteen, koska se poistuu active_
# stones:ista ("pysahtynyt" TAI "kadotettu" jalkeen) ja seuraava HAKU-
# kierros (1s valein) loytaa saman paikallaan olevan kohteen taas -
# rekisteroiden sen VIRHEELLISESTI UUTENA kivena. Esimerkkeja oikealta
# datalta: monta stone_id:ta joiden koko elinika oli alle 5s ja netto-
# siirtyma vain 1-250cm (verrattuna aitojen heittojen n. 2900-3100cm:iin
# koko HAKU-loytohetkesta lahemman pesan levahdyspaikkaan).
#
# Korjaus: koska KAIKKI aidot kivet ovat jo LIIKKEESSA kun HAKU loytaa
# ne ja jatkavat matkaansa satoja senttteja ENNEN pysahtymista, mutta
# paikallaan-jo-olevat kohteet eivat LIIKU merkittavasti: vaaditaan etta
# uusi HAKU-loytö siirtyy vahintaan taman verran (itseisarvo, ensim-
# maisesta havainnosta) ENNEN kuin sen havainnot kirjataan CSV:hen
# aitona heittona - katso pending_rows/confirmed-kasittely alempana.
# Kunnes vahvistus tapahtuu, havainnot puskuroidaan (ei kirjoiteta) -
# jos kivi kadotetaan/pysahtyy KOSKAAN vahvistumatta, koko puskuroitu
# havaintosarja hylataan hiljaisesti (ei CSV-rivia, ei lasketa mukaan
# "n_stones_seen":iin). Aito heitetty kivi ylittaa tamän kynnyksen jo
# ensimmaisen sekunnin sisalla (katso yllapuoliset esimerkit); pai-
# kallaan lepäävä kohde ei koskaan.
MIN_CONFIRMED_THROW_DISPLACEMENT_CM = 25.0

# HAKU-valin PAIKALLINEN ylikirjoitus (kayttajan pyynnosta) - EI
# muuteta kamera9_02.py:n omaa SEARCH_EVERY_N_FRAMES:ia (se tiedosto
# on koskematon referenssi, katso taman tiedoston alkupaan kommentti).
#
# HUOM (kayttajan uusin pyynto, katso keskusteluhistoria): palautettu
# takaisin kamera9_02.py:n alkuperaiseen, framepohjaiseen tahtiin (10
# framea=0.4s 25fps:lla) - aiempi, tata harvempi sekuntipohjainen vali
# (1.0s, CPU-saastosta) korvattu, koska kavennettu HAKU-vyohyke
# (kamera9_02.py:n SEARCH_Y_MIN/MAX_CM, nyt vain hogline...hogline+3m
# eika enaa asti pesan takareunalle) tekee jokaisesta HAKU-kutsusta jo
# halvemman, joten tihempi tarkistus ei enaa maksa yhta paljon - ja
# uuden kiven ensimmaiset havainnot loytyvat aiempaa nopeammin.
HAKU_SEARCH_INTERVAL_FRAMES = 10

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
# PANEELIEN KASIN-MERKINTA (VARAMENETTELY)
#
# Kayttajan pyynnosta: jos automaattitunnistus (referenssin lahelta)
# ei loyda vahintaan MIN_AUTO_PANELS_BEFORE_MANUAL (6) paneelia, se ei
# riita luotettavaan stabilointiin koko videon ajaksi - avataan siis
# skaalattavan kokoinen (cv2.WINDOW_NORMAL, kayttaja voi venyttaa
# ikkunan haluamaansa kokoon) ikkuna jossa kayttaja klikkaa PUUTTUVAT
# paneelit hiirella. Jo automaattisesti loydetyt paneelit nakyvat
# valmiiksi merkittyina (vihrea nelikulmio + jarjestysnumero), joten
# kayttajan tarvitsee klikata vain loput. Jokainen klikkaus ajaa
# saman detect_panel-tunnistuksen (tarkka kulma-/subpikselisovitus)
# klikkauskohdan ymparilta - sama menetelma kuin automaattitunnistus,
# vain hakupisteen ALKUPERA on kasin annettu. Enter lopettaa (vaatii
# vahintaan 2 paneelia yhteensa), Esc peruuttaa kokonaan.
# ============================================================

MIN_AUTO_PANELS_BEFORE_MANUAL = 6


def select_panels_manually(gray, frame_bgr=None, existing_panel_data=None,
                            window_name="Klikkaa puuttuvat paneelit (Enter=valmis, Esc=peruuta)"):

    display_base = frame_bgr if frame_bgr is not None else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    display_base = display_base.copy()

    panel_data = list(existing_panel_data) if existing_panel_data else []
    n_preexisting = len(panel_data)

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        result = detect_panel(gray, float(x), float(y))
        if result is None:
            print(f"  Ei loytynyt paneelia klikkauskohdasta ({x},{y}) - kokeile uudelleen (klikkaa lahempana paneelin keskikohtaa).")
            return
        panel_data.append(result)
        print(f"  Paneeli {len(panel_data)} merkitty kohtaan ({x},{y}).")

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)

    print(
        f"  Avataan kasin-merkinta-ikkuna: {n_preexisting} paneelia jo "
        f"automaattisesti loydetty, klikkaa loput hiirella. Enter lopettaa "
        f"(vahintaan 2 paneelia yhteensa), Esc peruuttaa."
    )

    try:
        while True:
            display = display_base.copy()
            for i, result in enumerate(panel_data):
                corners = result["corners"].astype(int)
                color = (0, 200, 255) if i < n_preexisting else (0, 255, 0)
                cv2.polylines(display, [corners], True, color, 2)
                center = tuple(result["center"].astype(int))
                cv2.putText(display, str(i + 1), center, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(
                display, f"Paneeleita merkitty: {len(panel_data)} - Enter=valmis, Esc=peruuta",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2
            )
            cv2.imshow(window_name, display)
            key = cv2.waitKey(20) & 0xFF

            if key in (13, 10):
                if len(panel_data) >= 2:
                    break
                print("  Vahintaan 2 paneelia tarvitaan ennen kuin voi jatkaa.")
            elif key == 27:
                raise RuntimeError("Kayttaja peruutti paneelien kasin-merkinnan.")
    finally:
        cv2.destroyWindow(window_name)

    print(f"  Paneelien kasin-merkinta valmis: {len(panel_data)} paneelia yhteensa.")
    return panel_data


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

def detect_panels_from_reference(gray, reference_panels, frame_bgr=None,
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

    if len(panel_data) < MIN_AUTO_PANELS_BEFORE_MANUAL:
        print(
            f"Automaattitunnistus loysi vain {len(panel_data)}/"
            f"{MIN_AUTO_PANELS_BEFORE_MANUAL} vaadittua paneelia - "
            f"avataan kasin-merkinta puuttuvien loytamiseksi."
        )
        panel_data = select_panels_manually(gray, frame_bgr, existing_panel_data=panel_data)

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

def _shadow_tolerant_background_mask(frame_bgr, reference_bgr, diff_threshold,
                                      shadow_v_drop_max):
    """ENABLE_SHADOW_TOLERANT_STABILIZATION:in ydinsaanto (katso sen
    kommentti): tausta = (tavallinen pieni erotus) TAI (saturaatio
    lahes sama mutta V pudonnut korkeintaan shadow_v_drop_max - eli
    varjo, ei aito objekti)."""

    diff = cv2.absdiff(frame_bgr, reference_bgr)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    background_mask = diff_gray < diff_threshold

    frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV).astype(np.int16)
    ref_hsv = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2HSV).astype(np.int16)
    v_drop = ref_hsv[..., 2] - frame_hsv[..., 2]
    shadow_mask = (v_drop > 0) & (v_drop <= shadow_v_drop_max)

    return background_mask | shadow_mask


def suppress_static_background(frame_bgr, reference_bgr, diff_threshold=30):

    if reference_bgr is None or frame_bgr.shape != reference_bgr.shape:
        return frame_bgr

    if ENABLE_SHADOW_TOLERANT_STABILIZATION:
        background_mask = _shadow_tolerant_background_mask(
            frame_bgr, reference_bgr, diff_threshold, SHADOW_V_DROP_MAX
        )
    else:
        diff = cv2.absdiff(frame_bgr, reference_bgr)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        background_mask = diff_gray < diff_threshold

    out = frame_bgr.copy()
    out[background_mask] = (255, 255, 255)

    return out


_hann_window_cache = {}


_ref_gray_cache = {}


def estimate_subpixel_alignment(frame_u, reference_u):
    """ENABLE_SUBPIXEL_ALIGNMENT:in kohdistushaku - katso sen kommentti
    taman tiedoston alkupaassa. TOINEN VERSIO (kayttajan pyynnosta,
    katso keskusteluhistoria): ENSIMMAINEN versio hyodynsi vain pienta
    paikallista aluetta, mika osoittautui EPAEDUSTAVAKSI koko kuvalle.
    Kayttaja ehdotti koko naytön pikseleiden vertaamista moodikuvaan -
    mutta ristikkohaku (satoja warpAffine-kutsuja) koko taysresoluutio-
    kuvalle olisi liian hidas, ja kuvan PIENENTAMINEN tuhoaisi juuri
    sen sub-pikseli-tarkkuuden jota haetaan (kayttajan huomio).
    Ratkaisu: cv2.phaseCorrelate (FFT-pohjainen vaihekorrelaatio) laskee
    TAYSRESOLUUTIOISEN kuvan sub-pikseli-tarkan KAANNON yhdella
    kutsulla - ei tarvitse testata erikseen satoja ehdokkaita eika
    pienentaa kuvaa. Rajattu SUBPIXEL_ALIGN_CROP_FRACTION-kokoiseen
    keskitettyyn alueeseen (EI pienennetty, vain rajattu - resoluutio
    sailyy) laskenta-ajan hillitsemiseksi (mitattu: taysi 1920x1080
    ~130ms/frame olisi liikaa) - reunat (katsomo/mainostaulut, eivat
    osa jaata) eivat muutenkaan auta kohdistuksessa, joten rajaus ei
    heikenna tarkkuutta. EI enaa kiertoa (angle) - alkuperainen kierto-
    korjaus oli juuri se osa joka VAHVISTI virheen etaisyyden mukana
    (katso ENABLE_SUBPIXEL_ALIGNMENT:in kommentti); paneiliseurannan
    RANSAC-affiinimuunnos jo korjaa suurimman osan kierrosta, joten
    jaljella oleva sub-pikseli-virhe on kaytannossa lahes pelkkaa
    translaatiota."""

    h, w = frame_u.shape[:2]
    cw = int(round(w * SUBPIXEL_ALIGN_CROP_FRACTION))
    ch = int(round(h * SUBPIXEL_ALIGN_CROP_FRACTION))
    x0 = (w - cw) // 2
    y0 = (h - ch) // 2

    key = (x0, y0, cw, ch)
    hann = _hann_window_cache.get(key)
    if hann is None:
        hann = cv2.createHanningWindow((cw, ch), cv2.CV_32F)
        _hann_window_cache[key] = hann

    ref_gray = _ref_gray_cache.get(key)
    if ref_gray is None:
        ref_crop = reference_u[y0:y0 + ch, x0:x0 + cw]
        ref_gray = cv2.cvtColor(ref_crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
        _ref_gray_cache[key] = ref_gray

    frame_crop = frame_u[y0:y0 + ch, x0:x0 + cw]
    frame_gray = cv2.cvtColor(frame_crop, cv2.COLOR_BGR2GRAY).astype(np.float32)

    (dx, dy), _response = cv2.phaseCorrelate(ref_gray, frame_gray, hann)

    # Turvaraja (kayttajan alkuperaisen SUBPIXEL_ALIGN_RANGE_PX:n
    # hengessa): tama on tarkoitettu VAIN pieneksi jaljella olevaksi
    # sub-pikseli-korjaukseksi, ei yleiseksi liikkeentunnistukseksi -
    # jos koko kuvan vaihekorrelaatio jostain syysta antaisi ison
    # arvon (esim. pelaaja peittaa suuren osan framesta), rajataan
    # se pois sen sijaan etta sovelletaan virheellisen suurta kaantoa.
    dx = float(np.clip(dx, -SUBPIXEL_ALIGN_RANGE_PX, SUBPIXEL_ALIGN_RANGE_PX))
    dy = float(np.clip(dy, -SUBPIXEL_ALIGN_RANGE_PX, SUBPIXEL_ALIGN_RANGE_PX))

    return (dx, dy, 0.0)

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

    # katso HAKU_SEARCH_INTERVAL_FRAMES:in kommentti - korvaa
    # kamera9_02.py:n SEARCH_EVERY_N_FRAMES:in elavan seurannan
    # HAKU-ajastuksessa (koskematon kamera9_02.py itse ennallaan).
    haku_interval_frames = max(1, HAKU_SEARCH_INTERVAL_FRAMES)

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
    # katso ASETUKSET-kommentti TRACK_LOST_GRACE_SECONDS:in kohdalla -
    # PAIKALLINEN ylikirjoitus kamera9_02.py:n TRACK_LOST_MAX_MISSES:lle
    # (max(), koska ylikirjoitus EI saa koskaan olla tiukempi kuin
    # alkuperainen, vain sallivampi).
    track_lost_max_misses = max(
        k92.TRACK_LOST_MAX_MISSES, int(round(fps * TRACK_LOST_GRACE_SECONDS))
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
    # Lasketaan VAIN liikevahvistetut (katso MIN_CONFIRMED_THROW_
    # DISPLACEMENT_CM) kivet - EI raakoja HAKU-ehdokkaita, joista suurin
    # osa (katso ASETUKSET-kommentti) on paikallaan-jo-olevien kohteiden
    # virheellisia uudelleenlöytöjä, ei aitoja heittoja.
    n_confirmed_stones = 0
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

    # ENABLE_SUBPIXEL_ALIGNMENT / ENABLE_SHADOW_TOLERANT_STABILIZATION
    # (kayttajan huomio: nama EIVAT olleet mukana "Yhteensa mitattu"
    # -summassa aiemmin, vaikka molemmat ajetaan joka elavan seurannan
    # framella - raportti siis ALIARVIOI kokonaisajan kun jompikumpi on
    # paalla).
    total_subpixel_align_time = 0.0
    total_shadow_suppress_time = 0.0

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
                # ENABLE_SUBPIXEL_ALIGNMENT / ENABLE_SHADOW_TOLERANT_
                # STABILIZATION (katso niiden kommentit taman tiedoston
                # alkupaassa) - kaksi ERILLISTA lippua, HELPPO POISTAA
                # KUMPIKIN itsenaisesti. HUOM: frame_u ITSE (varitarkistus,
                # debug-video) EI koskaan taustanvaimenneta tassa - vain
                # erillinen frame_u_for_tracking-kopio, jota kaytetaan
                # VAIN HAKU/SEURANTA-kutsuissa alla.
                # --------------------------------------------
                ref_undist_live = calib_result["calib"]["frame_undistorted"]

                if ENABLE_SUBPIXEL_ALIGNMENT:

                    t_align0 = time.perf_counter()

                    align_dx, align_dy, _align_angle = estimate_subpixel_alignment(
                        frame_u, ref_undist_live
                    )

                    M_align = np.array(
                        [[1.0, 0.0, align_dx], [0.0, 1.0, align_dy]],
                        dtype=np.float64
                    )
                    frame_u = cv2.warpAffine(
                        frame_u, M_align, (width, height),
                        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
                    )

                    total_subpixel_align_time += time.perf_counter() - t_align0

                if ENABLE_SHADOW_TOLERANT_STABILIZATION:
                    t_shadow0 = time.perf_counter()
                    frame_u_for_tracking = suppress_static_background(
                        frame_u, ref_undist_live, diff_threshold=GRANITE_DIFF_THRESHOLD
                    )
                    total_shadow_suppress_time += time.perf_counter() - t_shadow0
                    # frame_u_for_tracking on jo taustanvaimennettu (myos
                    # varjonsietoisesti) - C++:n OMA sisainen vaimennus
                    # HAKU/SEURANTA-kutsuissa ohitetaan antamalla sille 0.0,
                    # jotta frame_u_for_tracking:ia ei vaimenneta uudelleen.
                    haku_seuranta_diff_threshold = 0.0
                else:
                    frame_u_for_tracking = frame_u
                    haku_seuranta_diff_threshold = GRANITE_DIFF_THRESHOLD

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
                        frame_u_for_tracking, ref_undist_live,
                        local_pts_body, local_pts_search,
                        pose["K"], pose["R"], pose["t"],
                        x_center, k92.SEARCH_X_HALF_WIDTH_CM, y_center, y_half,
                        k92.SEARCH_COARSE_STEP_CM, k92.SEARCH_FINE_STEP_CM,
                        k92.SEARCH_SCORE_THRESHOLD,
                        live_state["R_max"], live_state["H_total"],
                        live_state["ring_r_frac_guess"],
                        haku_seuranta_diff_threshold
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

                    # TRACK_MAX_SPEED_Y/X_CM_S:in kommentti (taman
                    # tiedoston alkupaassa): hakualue lasketaan JOKA
                    # framella JOKAISELLE kivelle erikseen suoraan
                    # fysikaalisesta nopeusrajasta ja siita kuinka monta
                    # frameä on todellisuudessa kulunut viimeisimmasta
                    # VAHVISTETUSTA sijainnista (misses+1, koska last_xy
                    # ei paivity misseilla) - EI enaa kamera9_02.py:n
                    # kiinteaa, isotrooppista TRACK_HALF_RANGE_CM:ia.
                    elapsed_frames_arr = np.array(
                        [s["misses"] + 1 for s in seuranta_stones],
                        dtype=np.float64
                    )
                    elapsed_seconds_arr = elapsed_frames_arr / fps
                    half_range_y_arr = np.minimum(
                        TRACK_MAX_SPEED_Y_CM_S * elapsed_seconds_arr, TRACK_HALF_RANGE_MAX_CM
                    )
                    half_range_x_arr = np.minimum(
                        TRACK_MAX_SPEED_X_CM_S * elapsed_seconds_arr, TRACK_HALF_RANGE_MAX_CM
                    )

                    t_seuranta0 = time.time()
                    batch_results = stone_tracker.track_stones_batch(
                        frame_u_for_tracking, ref_undist_live,
                        X0_arr, Y0_arr,
                        half_range_x_arr, half_range_y_arr,
                        local_pts_body, local_pts_search,
                        pose["K"], pose["R"], pose["t"],
                        k92.TRACK_COARSE_STEP_CM, k92.TRACK_FINE_STEP_CM,
                        k92.TRACK_SCORE_THRESHOLD,
                        live_state["R_max"], live_state["H_total"],
                        live_state["ring_r_frac_guess"],
                        TRACK_MAX_BACKWARD_CM,
                        haku_seuranta_diff_threshold
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

                        # --------------------------------
                        # KUMULATIIVINEN "EI TAAKSEPAIN" -TARKISTUS
                        # (kayttajan pyynnosta, katso keskusteluhistoria):
                        # stone_tracker.cpp:n trackStoneUpdateOne hylkaa jo
                        # YHDEN paivityksen joka siirtaisi kiven >100cm
                        # taaksepain EDELLISESTA framesta - mutta tama EI
                        # estä montaa PIENTA (<100cm) taaksepain-askelta
                        # kasautumasta suureksi ajautumaksi usean sekunnin/
                        # minuutin aikana (havaittu koko videon lapikaynnissa:
                        # levossa ollut kivi "liukui" pikkuhiljaa kohti
                        # heittopaata, todennakoisesti SEURANNAN tarttuessa
                        # kiven vierella kavelevaan pelaajaan/lakaisijaan -
                        # tama tayttaa lopulta KAIKKI MAX_CONCURRENT_STONES-
                        # paikat haamuilla, estaen uusien aitojen heittojen
                        # rekisteroinnin). Verrataan siis KOKO elinkaaren
                        # PIENIMPAAN havaittuun Y-arvoon (s["min_y_seen"],
                        # ei vain edelliseen frameen) - kumulatiivinen
                        # ajautuma yli TRACK_MAX_BACKWARD_CM:n hylataan
                        # samoin kuin yksittainen liian iso hyppy.
                        # --------------------------------

                        if refined["found"] and (
                            refined["Y_cm"] - s["min_y_seen"] > TRACK_MAX_BACKWARD_CM
                        ):
                            refined = dict(refined)
                            refined["found"] = False

                        if refined["found"]:

                            s["last_xy"] = (
                                refined["X_cm"], refined["Y_cm"]
                            )
                            s["min_y_seen"] = min(
                                s["min_y_seen"], refined["Y_cm"]
                            )
                            s["misses"] = 0

                            debug_draw_items.append((
                                refined["X_cm"], refined["Y_cm"],
                                s["stone_id"],
                                (0, 255, 0) if refined["tarkka"] else (0, 165, 255),
                                str(s["stone_id"])
                            ))

                            # --------------------------------
                            # LIIKEVAHVISTUS: katso ASETUKSET-kommentti
                            # MIN_CONFIRMED_THROW_DISPLACEMENT_CM:in
                            # kohdalla - ei kirjoiteta CSV:hen ENNEN
                            # kuin kivi on siirtynyt riittavasti ensim-
                            # maisesta havainnostaan (torjuu paikallaan
                            # jo olevien kohteiden toistuvan virheelli-
                            # sen uudelleenrekisteroinnin). Puskuroidut
                            # rivit kirjoitetaan KAIKKI KERRALLA heti
                            # kun vahvistus tapahtuu - ei menetetä aikai-
                            # sia, aitoja havaintoja.
                            # --------------------------------

                            if not s["confirmed"]:

                                s["pending_rows"].append(
                                    (frame_index, timestamp, dict(refined))
                                )

                                if (
                                    abs(refined["Y_cm"] - s["y0_first_cm"])
                                    >= MIN_CONFIRMED_THROW_DISPLACEMENT_CM
                                ):

                                    s["confirmed"] = True
                                    n_confirmed_stones += 1

                                    for pf, pt, prow in s["pending_rows"]:
                                        _write_stone_csv_row(
                                            csv_writer, pf, pt,
                                            s["stone_id"], prow
                                        )

                                    s["pending_rows"] = []

                            else:

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
                                        if s["confirmed"]:
                                            print(
                                                f"[frame {frame_index}] Kivi "
                                                f"{s['stone_id']} pysahtynyt "
                                                f"(liikkunut {displacement:.1f}cm "
                                                f"viimeisen {STOP_TRACKING_SECONDS:.0f}s "
                                                "aikana) - lopetetaan seuranta."
                                            )
                                        else:
                                            # Ei koskaan liikkunut riittavasti
                                            # (katso MIN_CONFIRMED_THROW_
                                            # DISPLACEMENT_CM) - todennakoisesti
                                            # jo paikallaan ollut kohde, ei aito
                                            # heitto. Puskuroidut havainnot
                                            # hylataan hiljaisesti (EI CSV-riviä).
                                            s["pending_rows"] = []
                                            print(
                                                f"[frame {frame_index}] Ehdokas "
                                                f"{s['stone_id']} hylatty "
                                                "(ei liikkunut riittavasti - "
                                                "todennakoisesti jo paikallaan "
                                                "ollut kohde, ei aito heitto)."
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

                            if s["misses"] < track_lost_max_misses:
                                still_active.append(s)
                            elif s["confirmed"]:
                                print(
                                    f"[frame {frame_index}] Kivi "
                                    f"{s['stone_id']} kadotettu."
                                )
                            else:
                                s["pending_rows"] = []
                                print(
                                    f"[frame {frame_index}] Ehdokas "
                                    f"{s['stone_id']} hylatty (kadotettu "
                                    "ennen kuin liikkui riittavasti - "
                                    "todennakoisesti jo paikallaan ollut "
                                    "kohde, ei aito heitto)."
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
                                "min_y_seen": refined["Y_cm"],
                                "misses": 0,
                                "color_diff_history": [],
                                "position_history": [(
                                    frame_index,
                                    refined["X_cm"], refined["Y_cm"]
                                )],
                                # katso ASETUKSET-kommentti MIN_CONFIRMED_
                                # THROW_DISPLACEMENT_CM:in kohdalla - ei
                                # kirjoiteta CSV:hen ennen kuin liike on
                                # vahvistettu (torjuu paikallaan-jo-olevien
                                # kohteiden toistuvan uudelleenrekisterointi-
                                # ongelman).
                                "y0_first_cm": refined["Y_cm"],
                                "confirmed": False,
                                "pending_rows": [
                                    (frame_index, timestamp, dict(refined))
                                ],
                            })

                            debug_draw_items.append((
                                refined["X_cm"], refined["Y_cm"],
                                stone_id, (0, 255, 255),
                                f"{stone_id} UUSI"
                            ))

                            print(
                                f"[frame {frame_index}] Uusi kivi-ehdokas "
                                f"{stone_id}: "
                                f"({refined['X_cm']:.1f}, "
                                f"{refined['Y_cm']:.1f}) cm "
                                "(odottaa liikevahvistusta ennen CSV-kirjausta)"
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
                    f"warpAffine+remap(koko frame) {(total_warp_remap_time / processed) * 1000:.2f} ms/ruutu | "
                    f"sub-pikseli-kohdistus {(total_subpixel_align_time / processed) * 1000:.2f} ms/ruutu | "
                    f"varjosuodatus {(total_shadow_suppress_time / processed) * 1000:.2f} ms/ruutu"
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
    print(f"sub-pikseli-kohdistus (ENABLE_SUBPIXEL_ALIGNMENT): "
          f"{total_subpixel_align_time:.2f}s yhteensa, "
          f"{(total_subpixel_align_time / processed_frames) * 1000:.2f} ms/ruutu")
    print(f"varjonsietoinen taustanvaimennus (ENABLE_SHADOW_TOLERANT_STABILIZATION): "
          f"{total_shadow_suppress_time:.2f}s yhteensa, "
          f"{(total_shadow_suppress_time / processed_frames) * 1000:.2f} ms/ruutu")
    muu_yhteensa = (
        total_read_time + total_stabilize_compute_time
        + total_warp_remap_time + total_gray_time
        + total_tracking_time + total_transform_time
        + total_subpixel_align_time + total_shadow_suppress_time
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
        "n_stones_seen": n_confirmed_stones,
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
            first_gray, reference_panels, frame_bgr=first_frame
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
