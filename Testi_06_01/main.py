import os
# v5.7: NumPyn BLAS (OpenBLAS/MKL) yhdelle saikeelle ENNEN numpy/cv2-importia. Koodin matriisit ovat pienia (3x3, polyfit, lstsq),
# joten monisaikeinen BLAS ei nopeuta mitaan, mutta OpenBLASin tyosaikeet jaavat jokaisen kutsun jalkeen pyorimaan (busy-wait,
# sched_yield / SwitchToThread) ja veivat profiloinnissa ~20 % CPU-ajasta SEURANNAN C++-saikeilta ja vaiheilta A/B.
# BLAS_THREADS=n palauttaa monisaikeisuuden kokeiluun.
for _blas_var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_blas_var, os.environ.get("BLAS_THREADS", "1"))
import sys
import math
import csv
import time
import threading
import argparse
import importlib.util
import cv2

# Testi_05_03: OpenCV:n oma saiemaara (cv2.setNumThreads) koeajoihin: CV_THREADS=2 python main.py --max-frame 3000 --no-debug (oletus: OpenCV:n oma)
if os.environ.get("CV_THREADS"):
    cv2.setNumThreads(int(os.environ["CV_THREADS"]))
import numpy as np
import tkinter as tk
from tkinter import filedialog
from concurrent.futures import ThreadPoolExecutor
import mode_engine
import stone_tracker
import haku_silhouette   # Testi_03_04: HAKU:n siluettitarkennus (valinnainen)
import live_source      # Testi_06_01: elava kamerasyote (Cam Link) + ruutupuskuri
import hog_analyysi      # Testi_05_01: hog-hog -analyysi (toisen asteen sovitus Y(t), nopeus/hidastuvuus/hog-hog-aika)
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
# v6.7: hoglinien paikkaa ei ole mitattu tarkasti -> todellinen paikka voi poiketa nimellisesta (640 cm teesta)
# +-HOGLINE_POSITION_TOLERANCE_CM. Kalibrointi ei pakota hoglinea nimelliseen paikkaan, vaan mittaa sen paikan
# toleranssin sisalla (hogline rajoittaa vain suoruutta/kiertoa) ja kayttaa mitattua paikkaa kaikkialla.
HOGLINE_POSITION_TOLERANCE_CM = 20.0
# v6.8: hoglinet samalla etaisyydella omasta T-viivastaan (yksi yhteinen siirtyma); kaukaisen hoglinen mittauksen
# paino yhteisessa siirtymassa (lahi = 1; kauko on 30 m paassa ja maaraytyy kuvasta heikommin).
HOGLINE_SYMMETRIC = int(os.environ.get("HOGLINE_SYMMETRIC", "1"))
HOGLINE_SYMMETRIC_FAR_WEIGHT = float(os.environ.get("HOGLINE_SYMMETRIC_FAR_WEIGHT", "0.0"))
NOMINAL_NEAR_HOGLINE_Y_CM = k8.NEAR_HOGLINE_Y_CM
NOMINAL_FAR_HOGLINE_Y_CM = k8.FAR_HOGLINE_Y_CM


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


def _hogline_target_y(H, hog_pts, nominal_y_cm, tol_cm=None):
    """v6.7: hoglinen tavoite-Y homografian sovitukseen: hoglinepisteiden mediaani-Y nykyisella H:lla, rajattuna
    nimellinen +-tol_cm. Toleranssin sisalla hogline ei siis veda radan pituussuuntaista mittakaavaa (paikka
    maaraytyy pesista), vain viivan suoruus ja kierto rajoittavat; toleranssin ulkopuolelle se ei paase."""
    if tol_cm is None:
        tol_cm = HOGLINE_POSITION_TOLERANCE_CM
    if len(hog_pts) == 0 or tol_cm <= 0.0:
        return float(nominal_y_cm)
    y = k8.output_px_to_physical(k8._apply_h(H, np.asarray(hog_pts, dtype=np.float64)))[:, 1]
    return float(nominal_y_cm + np.clip(np.median(y) - nominal_y_cm, -tol_cm, tol_cm))


def _hogline_targets(H, near_pts, far_pts, tol_cm=None):
    """v6.8: molempien hoglinien tavoite-Y. HOGLINE_SYMMETRIC=1 (oletus): hoglinet ovat SAMALLA etaisyydella
    omasta T-viivastaan (640 + d cm) -> yksi yhteinen siirtyma d, joka lasketaan molempien viivojen mittauksista
    painotettuna (HOGLINE_SYMMETRIC_FAR_WEIGHT; lahi-hogline mitataan tarkemmin). d rajataan +-tol_cm."""
    if tol_cm is None:
        tol_cm = HOGLINE_POSITION_TOLERANCE_CM
    if not HOGLINE_SYMMETRIC:
        return (_hogline_target_y(H, near_pts, NOMINAL_NEAR_HOGLINE_Y_CM, tol_cm),
                _hogline_target_y(H, far_pts, NOMINAL_FAR_HOGLINE_Y_CM, tol_cm))
    if tol_cm <= 0.0 or (len(near_pts) == 0 and len(far_pts) == 0):
        return float(NOMINAL_NEAR_HOGLINE_Y_CM), float(NOMINAL_FAR_HOGLINE_Y_CM)
    parts, weights = [], []
    if len(near_pts):
        y = k8.output_px_to_physical(k8._apply_h(H, np.asarray(near_pts, dtype=np.float64)))[:, 1]
        parts.append(float(np.median(y)) - NOMINAL_NEAR_HOGLINE_Y_CM)      # + = kauempana lahi-T:sta
        weights.append(1.0)
    if len(far_pts):
        y = k8.output_px_to_physical(k8._apply_h(H, np.asarray(far_pts, dtype=np.float64)))[:, 1]
        parts.append(NOMINAL_FAR_HOGLINE_Y_CM - float(np.median(y)))       # + = kauempana kauko-T:sta
        weights.append(HOGLINE_SYMMETRIC_FAR_WEIGHT if len(near_pts) else 1.0)
    d = float(np.clip(np.average(parts, weights=weights), -tol_cm, tol_cm))
    return float(NOMINAL_NEAR_HOGLINE_Y_CM + d), float(NOMINAL_FAR_HOGLINE_Y_CM - d)


def set_hogline_positions(near_y_cm, far_y_cm):
    """v6.7: asettaa hoglinien paikan (k8-moduulin vakiot) -> kayttoon kaikkialla: kameran asento (k9),
    heittoportti, hog-ylitys ja hog-hog-analyysi, piirrot."""
    k8.NEAR_HOGLINE_Y_CM = float(near_y_cm)
    k8.FAR_HOGLINE_Y_CM = float(far_y_cm)


def refine_geometric_homography_color(frame_undistorted, H_init, near_pts, near_phys,
                                       output_w, output_h,
                                       max_iterations=None, min_relative_improvement=None,
                                       hog_tol_cm=0.0, label="varipohjainen"):
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
        near_hog_y, far_hog_y = _hogline_targets(H_current, near_hog_pts, far_hog_pts, hog_tol_cm)
        hog_y = np.concatenate([
            np.full(len(near_hog_pts), near_hog_y),
            np.full(len(far_hog_pts), far_hog_y),
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
            f"[Geometrinen korjaus ({label}) {iteration}/{max_iterations}] "
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
            "near_hog_pts": near_hog_pts, "far_hog_pts": far_hog_pts,
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

    set_hogline_positions(NOMINAL_NEAR_HOGLINE_Y_CM, NOMINAL_FAR_HOGLINE_Y_CM)

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
    # v6.7: k8.build_image_directions ottaa v_t:n merkin T-viivan janaparin JARJESTYKSESTA (sattumanvarainen) ->
    # joskus sivusuunta kaantyi ja homografiasta tuli PEILIKUVA (fysikaaliset x:t +-vaihtuivat, kaukaista pesaa ei
    # loytynyt). Ylhaalta katsova kamera ei koskaan peilaa: kuvassa (y alas) eteen x sivulle > 0 kuten
    # topdown-kuvassa (eteen = ylos, +x = oikealle). Kun merkki oli jo oikein, tulos on taysin ennallaan.
    if dir_forward[0] * dir_lateral[1] - dir_forward[1] * dir_lateral[0] < 0:
        dir_lateral = -dir_lateral

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

    print("  Etsitaan kaukaisen pesan karkea sijainti (rivi-skannaus + lahi-pesa-homografia)...")
    # v6.7: k1 hyvaksytaan kun se parantaa lahipesan RMS:aa > 8 %. Rajatapauksessa (esim. 8.2 %) huonosti maaraytynyt
    # k1 vie 28 m paahan ekstrapoloidun kaukaisen pesan hakualueen ulkopuolelle -> kalibrointi kaatui. Jos kaukaista
    # pesaa ei loydy k1:lla, yritetaan ilman vaaristymakorjausta (k1 = 0). Kun k1:lla loytyy, tulos on ennallaan.
    k1_candidates = [best_k1] + ([0.0] if best_k1 != 0.0 else [])
    for k1_try in k1_candidates:
        dist_coeffs = np.array([k1_try, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        frame_undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs)
        near_pts_frame = k8.undistort_points_px(
            np.array(near_img_pts, dtype=np.float64), camera_matrix, k1_try
        )
        try:
            H_v1, output_w, output_h = build_far_house_center_seed(
                frame_undistorted, near_pts_frame, near_phys_pts
            )
        except RuntimeError as e:
            if k1_try == k1_candidates[-1]:
                raise
            print(f"  {e} -> uusi yritys ilman vaaristymakorjausta (k1 {k1_try:.3f} -> 0)")
            continue
        best_k1 = k1_try
        break

    print("  Geometrinen tarkennus (varipohjainen kaukainen rengas + harmaasavy-hoglinet)...")
    best = refine_geometric_homography_color(
        frame_undistorted, H_v1, near_pts_frame, near_phys_pts, output_w, output_h
    )
    print(f"  Hoglinet nimellisessa paikassa: RMS = {best['rms']:.3f} cm, {k8.house_quality_str(best['quality'])}")

    # v6.7: hoglinien paikka voi poiketa nimellisesta +-HOGLINE_POSITION_TOLERANCE_CM. Sovitetaan myos versio, jossa
    # hogline saa olla missa tahansa toleranssin sisalla; se valitaan, jos pesien pyoreys/koko ei huonone
    # (k8.candidate_is_better, sama saanto kuin kalibroinnissa muutenkin). Kaukainen hogline on 30 m paassa ja
    # sen paikka maaraytyy kuvasta heikosti -> jos vapaampi sovitus huonontaa pesia, pidetaan nimellinen.
    hog_y = [NOMINAL_NEAR_HOGLINE_Y_CM, NOMINAL_FAR_HOGLINE_Y_CM]
    if HOGLINE_POSITION_TOLERANCE_CM > 0.0:
        free = refine_geometric_homography_color(
            frame_undistorted, H_v1, near_pts_frame, near_phys_pts, output_w, output_h,
            hog_tol_cm=HOGLINE_POSITION_TOLERANCE_CM, label=f"hogline +-{HOGLINE_POSITION_TOLERANCE_CM:.0f} cm"
        )
        free_y = list(_hogline_targets(free["H_final"], free["near_hog_pts"], free["far_hog_pts"]))
        accept, _, _ = k8.candidate_is_better(
            frame_undistorted, free["H_final"], free["rms"], frame_undistorted, best["H_final"], best["rms"]
        )
        print(f"  Hoglinet +-{HOGLINE_POSITION_TOLERANCE_CM:.0f} cm: RMS = {free['rms']:.3f} cm, "
              f"{k8.house_quality_str(free['quality'])}, lahi {free_y[0] - hog_y[0]:+.1f} cm, "
              f"kauko {free_y[1] - hog_y[1]:+.1f} cm -> {'VALITAAN' if accept else 'hylataan (pesat eivat parane)'}")
        if accept:
            best, hog_y = free, free_y

    print(f"  Valittu H: RMS = {best['rms']:.3f} cm, {k8.house_quality_str(best['quality'])} | hoglinet: "
          f"lahi {hog_y[0]:.1f} cm ({hog_y[0] - NOMINAL_NEAR_HOGLINE_Y_CM:+.1f}), "
          f"kauko {hog_y[1]:.1f} cm ({hog_y[1] - NOMINAL_FAR_HOGLINE_Y_CM:+.1f})")
    set_hogline_positions(*hog_y)

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
#
# Kayttajan pyynnosta: debug-videoon piirretaan moodikuvalla suodatettu
# frame_u_for_tracking (se mita HAKU/SEURANTA oikeasti kayttaa
# tunnistukseen - katso ENABLE_SHADOW_TOLERANT_STABILIZATION), EI
# alkuperaista frame_u:ta - nain nakee suoraan mita tunnistus itse
# asiassa "nakee".
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

# Kayttajan kanssa kasin lasin 4:15/4:57-frameilla (MAH00014) testatut
# raja-arvot - katso keskusteluhistoria: -3<v_drop<50 antoi hieman
# paremman (pienemman fg-vuodon) tuloksen kuin alkuperainen 0<v_drop<=30
# SAMALLA color_diff<10-arvolla molemmissa testatuissa frameissa.
SHADOW_V_DROP_MIN = -3.0  # kuinka paljon V saa NOUSTA (negatiivinen pudotus) ja silti tulkita taustaksi
SHADOW_V_DROP_MAX = 50.0  # kuinka paljon V (HSV) saa pudota ja silti tulkita taustaksi/varjoksi

# ============================================================
# ABSOLUUTTINEN JAA-SUODATUS (S/V) - kayttajan pyynnosta, katso
# keskusteluhistoria: moodikuvasta (VAIN rata-alueen pikselit, rajattu
# sadetasoleikkauksella fyysisiin rajoihin k8.OUTPUT_X/Y_MIN/MAX_CM:aan,
# EI koko kameranakymaa - laidat/katto/pesamainokset olisivat vaaris-
# taneet jakaumaa) laskettu S/V-2D-histogrammi paljasti etta selva
# enemmisto (~68%) radan omista jaapikseleista tayttaa S<22 & V>128:n.
#
# TARKISTETTU (kayttajan pyynnosta) etta tama EI syo kivea merkittavasti:
# alkuperainen epailys (kiven kirkkaat pikselit osuisivat samaan
# alueeseen) osoittautui SUURELTA OSIN oman rajausvirheen (liian valjan
# kivimaskin, joka sisalsi kiven ymparilla olevaa sumeaa jaa/kiiltohaloa)
# aiheuttamaksi - Canny-reunantunnistus vahvisti ettei kiven ja jaan
# valilla ollut edes oikeaa gradienttia siina kohtaa, ja pikseliarvot
# (~V170-186) vastasivat suoraan jaata (~V163), eivat kiven omaa tummaa
# ydinta (~V50). Tiukalla, todelliseen reunaan sovitetulla kivimaskilla
# haviota oli VAIN n. 0.1% kiven pikseleista (molemmat testikivet,
# stone_check_frame.png) - tama on positio-RIIPPUMATON suodatus (ei
# vertaa samaan pikseliin moodikuvassa kuten ylla olevat kaksi), joten
# se auttaa NIMENOMAAN tapauksissa joissa valotasapaino/kirkkaus on jo
# korjattu mutta yksittainen pikseli silti eroaa referenssista (esim.
# heijastus/kiilto joka liikkuu framen mukana) - katso myos ylla oleva
# valotasapainokorjaus, joka jo hoitaa suurimman osan hitaasta ajautu-
# masta ETUKATEEN, joten tama on lisasuoja sen PAALLE, ei korvaa sita.
# ============================================================
ICE_S_MAX = 22
ICE_V_MIN = 128
# Testi_03_04 v4.5-koe: varitarkistus taustanvaimennuksessa: (ero tai varjo tai jaa) JA (S_ref = moodikuvan kylläisyys < COLOR_GATE_S_MAX TAI |H_ruutu - H_ref| <= COLOR_GATE_H_TOL).
# Testi_05_03: kivea seurataan vain joka toisella ruudulla kun Y < tama (cm); 0 = pois (oletus v5.4:sta alkaen). Ymparistomuuttuja SEURANTA_HALF_RATE_Y_CM (esim. 1500 = 15 m).
SEURANTA_HALF_RATE_Y_CM = float(os.environ.get("SEURANTA_HALF_RATE_Y_CM", "0"))   # v5.4: OLETUKSENA POIS (ei nopeuttanut mitaan mutta heikensi laatua: 24 -> 22 heittoa)

COLOR_GATE = os.environ.get("COLOR_GATE", "1") == "1"      # v4.5: OLETUKSENA PAALLA (COLOR_GATE=0 = vanha kolmen kriteerin vaimennus)
COLOR_GATE_S_MAX = int(os.environ.get("COLOR_GATE_S_MAX", "60"))
COLOR_GATE_H_TOL = int(os.environ.get("COLOR_GATE_H_TOL", "10"))
COLOR_GATE_HUE_SAT = os.environ.get("COLOR_GATE_HUE_SAT", "1") == "1"      # 1: bg &= (S_ref < S_MAX) | (|dH| <= H_TOL & S_ruutu > S_MAX)
COLOR_GATE_BOTH = os.environ.get("COLOR_GATE_BOTH", "0") == "1"      # 1: matala kylläisyys vaaditaan seka moodikuvalta etta ruudulta (kahva sailyy jaalla, tumma kivi mainoksen paalla sailyy)
if hasattr(stone_tracker, "set_color_gate"):
    stone_tracker.set_color_gate(COLOR_GATE_S_MAX if COLOR_GATE else 256, COLOR_GATE_H_TOL, int(COLOR_GATE_BOTH), int(COLOR_GATE_HUE_SAT))

# Testi_05_03: GPU-ristikkohaku (OpenCL, esim. Intel UHD) SEURANNAN hienolle ristikolle. GPU_GRID=1 paalle (oletus pois), GPU_GRID_VERIFY=1 vertaa CPU:hun ja kerryttaa tilastoa,
# GPU_GRID_DEVICE=gpu|cpu|any (oletus gpu).

# Testi_05_03: GPU_B=1 siirtaa vaiheen B (warpAffine+remap + varjotoleranssi-taustanvaimennus + valotasapaino) OpenCL:lle (esim. Intel UHD); tulos bitti-identtinen CPU:n (OpenCV 4.x) kanssa.
# v5.9: OLETUKSENA PAALLA (nopein asetus Windows-vertailuajossa; jos OpenCL-laitetta ei loydy, palataan CPU-polkuun). GPU_B=0 pois.
GPU_B = os.environ.get("GPU_B", "1") == "1"

# Testi_05_03 v5.5: liukuhihnan asetukset (kokeiluun, katso tools/vertailuajo.py).
#   STAB_WORKERS=n : stabilointi (vaihekorrelaatio, vaihe A) n ruudulle rinnan (oletus 1 = ennallaan); ruudut ovat toisistaan riippumattomia (kukin vs. moodikuva), jarjestys sailyy -> tulos identtinen.
#   PIPE_DEPTH=n   : vaiheiden valisten jonojen koko (oletus 3); isompi tasoittaa vaiheiden aikavaihtelua.
#   v5.9: oletukset 2 ja 6 (nopein Windows-vertailuajossa; aiemmin 1 ja 3).
STAB_WORKERS = max(1, int(os.environ.get("STAB_WORKERS", "2")))
PIPE_DEPTH = max(1, int(os.environ.get("PIPE_DEPTH", "6")))
# Testi_05_03 v5.6: INTRA_PARALLEL=1 ajaa kiven kolme hakua (ristikko + 2 mean-shiftia) rinnan eri saikeissa (oletus pois; tulos identtinen).
#   v5.9: OLETUKSENA PAALLA (INTRA_PARALLEL=0 pois).
INTRA_PARALLEL = os.environ.get("INTRA_PARALLEL", "1") == "1"
# v5.10: SEURANNAN ristikkohaun hieno vaihe GRID_THREADS saikeelle (oletus 2; 1 = ennallaan) ja kiven valmistelun etualamaski rinnan
# (PREP_PARALLEL, oletus 1). Molemmat bitti-identtisia perakkaisen version kanssa.
GRID_THREADS = max(1, int(os.environ.get("GRID_THREADS", "2")))
PREP_PARALLEL = os.environ.get("PREP_PARALLEL", "1") == "1"
SAT_PARALLEL = os.environ.get("SAT_PARALLEL", "0") == "1"   # v5.13: saturaatio rinnan graniittimaskin kanssa (oletus pois: hidasti Windowsissa)
# v5.7: Pythonin GIL-vaihtovali (sys.setswitchinterval, oletus 5 ms). Liukuhihnassa on useita Python-saikeita (A, B, debug-piirto, HAKU,
# valotasapaino); kun paasaie palaa C++/cv2-kutsusta (GIL vapautettu), se joutuu odottamaan GIL:ia enimmillaan koko vaihtovalin jos toinen
# saie ajaa Python-koodia. Lyhyempi vali -> paasaie (pullonkaula) saa GIL:n nopeammin takaisin. Ei vaikuta tuloksiin. 0 = Pythonin oletus.
PY_SWITCH_INTERVAL_MS = float(os.environ.get("PY_SWITCH_INTERVAL_MS", "0"))
if PY_SWITCH_INTERVAL_MS > 0:
    sys.setswitchinterval(PY_SWITCH_INTERVAL_MS / 1000.0)
def _print_cpp_opencv_info():
    """v5.11: C++-moduulin (stone_tracker) OpenCV:n kaannosasetukset: puuttuva AVX2-optimointi tai IPP selittaisi hitaan
    graniittimaskin (cvtColor/GaussianBlur). Tulostetaan vain olennaiset rivit."""
    if not hasattr(stone_tracker, "opencv_build_info"):
        return
    try:
        keys = ("General configuration for OpenCV", "Baseline:", "Dispatched code generation:", "requested:", "Parallel framework:",
                "Intel IPP:", "at:", "C++ flags (Release):", "Configuration:", "Built as dynamic libs?:")
        lines = [l.strip() for l in stone_tracker.opencv_build_info().splitlines() if any(k in l for k in keys)]
        print("C++-moduulin OpenCV: " + " | ".join(lines[:14]))
    except Exception as e:
        print(f"C++-moduulin OpenCV: tietoja ei saatu ({e!r})")


_print_cpp_opencv_info()
if hasattr(stone_tracker, "set_grid_threads"):
    stone_tracker.set_grid_threads(GRID_THREADS)
    stone_tracker.set_prep_parallel(int(PREP_PARALLEL))
    if hasattr(stone_tracker, "set_sat_parallel"):
        stone_tracker.set_sat_parallel(int(SAT_PARALLEL))
if hasattr(stone_tracker, "set_intra_parallel"):
    stone_tracker.set_intra_parallel(int(INTRA_PARALLEL))
    if INTRA_PARALLEL:
        print("SEURANTA: kiven sisainen rinnakkaisuus paalla (ristikkohaku + 2 mean-shiftia rinnan)")
GPU_GRID = os.environ.get("GPU_GRID", "0") == "1"
GPU_GRID_VERIFY = os.environ.get("GPU_GRID_VERIFY", "0") == "1"
if hasattr(stone_tracker, "set_gpu_grid"):
    _gpu_info = stone_tracker.set_gpu_grid(int(GPU_GRID), int(GPU_GRID_VERIFY))
    if GPU_GRID:
        print(f"GPU-ristikkohaku: {_gpu_info}" + (" (verify: vertaa CPU-tuloksiin)" if GPU_GRID_VERIFY else ""))

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
# ei enaa ristikkohakua eika kiertoa (katso _phase_correlate_full_frame).
#
# PAALLA (kayttajan pyynnosta): validoitu koko videon lapikaynnilla
# (SUBPIXEL_ALIGN_CROP_FRACTION=0.67:lla) - ei enaa karkaavia haamuja,
# kaikki viisi tunnistettavissa olevaa oikeaa heittoa loytyivat. HUOM:
# lisaa merkittavasti laskenta-aikaa - katso NOPEUSSEURANTA-raportin
# oma rivi. Kayttajan pyynnosta SUBPIXEL_ALIGN_CROP_FRACTION nostettu
# takaisin 1.0:aan (koko rata) taman validoinnin jalkeen - EI VIELA
# uudelleenvalidoitu koko videon lapikaynnilla taysikokoisena.
#
# ENABLE_SUBPIXEL_ALIGNMENT-LIPPU POISTETTU (kayttajan havainto + oma
# empiirinen vahvistus, katso keskusteluhistoria "loppuvideon" PIKSELI-
# TARKKA SUORA STABILOINTI -kommentin kohdalla): talla lipulla ohjattu
# jalkikorjaus (estimate_subpixel_alignment alla) mittasi ALKUPERAISESTI
# pienen residuaalin epatarkan paneilipohjaisen stabiloinnin PAALLE -
# nyt kun paastabilointi TEKEE JO TASMALLEEN saman suoran moodikuva-
# vertailun, jalkikorjaus vain toisti saman mittauksen ja kasvatti
# virhetta (havaittu: taustanvaimennuksen reunavuoto +54% jalkikorjauksen
# kanssa vs. ilman). estimate_subpixel_alignment/_phase_correlate_full_
# frame jataan silti talteen - JALKIMMAINEN on edelleen KAYTOSSA "loppu-
# videon" paastabiloinnin OMANA ytimena (katso sen kaytto alempana).
# ============================================================

# HAKU:n SPAWN-SUODATIN (Testi_03_02): roskaehdokkaat hylataan jo rekisteroinnissa (katso stone_tracker.cpp
# set_spawn_filter). SPAWN_FILTER=0 kytkee pois. Kynnykset johdettu MAH-videon 673 ehdokkaasta (26 oikeaa).
# PAIKANVARAUS (Testi_03_02): kun MAX_CONCURRENT_STONES on taynna, HAKU ajetaan silti ja uusi ehdokas saa
# paikan poistamalla huonoimman aktiivisen radan (ensin vahvistamattomat, sitten vahvistetut joiden sovitus on
# huono, rms >= EVICT_MIN_RMS). Oikea heitto ei saa jaada rekisteroimatta siksi etta roskaratoja on taysi maara.
EVICT_AT_CAP = os.environ.get("EVICT_AT_CAP", "1") == "1"
EVICT_MIN_RMS = float(os.environ.get("EVICT_MIN_RMS", "8.0"))
SPAWN_FILTER = os.environ.get("SPAWN_FILTER", "1") == "1"
SPAWN_RMS_MAX = float(os.environ.get("SPAWN_RMS_MAX", "3.5"))
SPAWN_NB_MIN = int(os.environ.get("SPAWN_NB_MIN", "14"))
SPAWN_NB_MAX = int(os.environ.get("SPAWN_NB_MAX", "45"))
SPAWN_ABS_X_MAX = float(os.environ.get("SPAWN_ABS_X_MAX", "65"))
SPAWN_SCORE_MAX = float(os.environ.get("SPAWN_SCORE_MAX", "0.9"))
SEEK_MAX_SKIP = int(os.environ.get("SEEK_MAX_SKIP", "25"))   # v5.7: havaintoruutujen luku: eteenpain luku jos <= nain monta ruutua (0 = aina haku, kuten ennen)
SCAN_WORKERS = max(1, int(os.environ.get("SCAN_WORKERS", "3")))   # v5.7: profiilin opettelun kandidaattiskannaus rinnan (1 = perakkain, kuten ennen)
LIVE_PIPELINE = os.environ.get("LIVE_PIPELINE", "1") == "1"
HAKU_AHEAD = os.environ.get("HAKU_AHEAD", "1") == "1"
# Testi_06_01: live-tila (katso live_source.py). LIVE_HYPPY=1: elavan seurannan alussa hypataan uusimpaan kameran ruutuun
# (kalibroinnin aikana kertynyt viive pois; 0 = jatketaan perakkain, simulaation tarkistusajoon). LIVE_TAAKSE_SEURANTA_S:
# kuinka paljon historiaa puskuri pitaa elavan seurannan aikana (kalibroinnin aikana --live-taakse-s, oletus 40 s).
LIVE_HYPPY = os.environ.get("LIVE_HYPPY", "1") == "1"
LIVE_TAAKSE_SEURANTA_S = float(os.environ.get("LIVE_TAAKSE_SEURANTA_S", "2"))
CV_SINGLE_PERSIST = os.environ.get("CV_SINGLE_PERSIST", "1") == "1"   # v5.12: C++-OpenCV pysyvasti 1 saikeelle elavassa vaiheessa (0 = vaihto joka kutsulla kuten ennen)
SIL_IN_BATCH = os.environ.get("SIL_IN_BATCH", "1") == "1"   # v5.11: SEURANNAN siluettitarkennus C++-kivisaikeissa (0 = erillinen vaihe)   # v5.11: HAKU kaynnistetaan jo liukuhihnan vaiheessa B (0 = paasaikeessa kuten ennen)  # liukuhihna: ruudun valmistelu omassa saikeessa
VIDEO_PREFETCH = os.environ.get("VIDEO_PREFETCH", "1") == "1"  # videon luku omassa saikeessa elavassa vaiheessa
PHOTO_APPLY_DELAY_FRAMES = int(os.environ.get("PHOTO_APPLY_DELAY_FRAMES", "10"))  # valotasapainon uusi arvo kayttoon tasan N ruudun paasta (toistettava ajo)
PHOTO_SUBSAMPLE = int(os.environ.get("PHOTO_SUBSAMPLE", "4"))  # valotasapainon estimoinnin pikseliharvennus (1 = kaikki pikselit)
SUBPIXEL_ALIGN_RANGE_PX = 10.0  # turvaraja - katso _phase_correlate_full_frame (kayttajan pyynnosta 10x, oli 1.0)

# Kuinka suuri, kuvan keskelle keskitetty osuus (leveys JA korkeus)
# kaytetaan vaihekorrelaatioon - EI pienennys, vain RAJAUS (resoluutio
# sailyy taysimittaisena) - katso _phase_correlate_full_frame. Kayttajan
# pyynnosta 1.0 (koko rata/koko kuva, EI vain keskustaa) - nopeuden
# hillitsemiseksi aiemmin kokeiltu 0.67 (~38ms/ruutu) hylattiin, koska
# stabiloinnin pitaa kattaa koko radan, ei vain kuvan keskiosaa. Koko
# kuvan kaytto maksaa enemman (mitattu taysikokoisena ~130-140ms/ruutu
# tassa Linux-diagnoosiymparistossa).
SUBPIXEL_ALIGN_CROP_FRACTION = 1.0

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
PROFILE_MAX_RMS_PX = 1.5   # Testi_03_03: oli 10.0 (rms < 1 px = oikean kokoinen kivi, < 5 px = jo vaaran kokoinen)
PROFILE_MIN_SAMPLES = 15
PROFILE_SAMPLES_PER_STONE = 40   # Testi_03_04: oli 25 (alfa-saantojen hylkaykset pienentavat havaintomaaraa; min 15)

# Turvaverkko RMS-kynnyksen LISAKSI (ei sen sijaan): kamera9_01.py:n
# fit_stone_profile:in oma dokumentaatio sanoo R_max:in olevan
# fyysisesti n. 14.0-14.6cm (oikean kiven halkaisija ~28cm). Aiemmin
# testatessa loydettiin tapaus jossa vaara kandidaatti (mainosteksti)
# lapaisi RMS-kynnyksen mutta antoi R_max~66cm - n. 4.5x liian ison.
# Reilut rajat (paljon RMS-kynnysta tiukemmat vaatimukset olisivat
# turhia, mutta karsivat selvasti fysiikan vastaiset tulokset).
PROFILE_R_MAX_MIN_CM = 12.5   # Testi_03_03: oli 10.0
PROFILE_R_MAX_MAX_CM = 15.0   # Testi_03_03: oli 20.0 (saannot: ymparysmitta <= 91.44 cm -> R <= 14.55 cm)

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
# radalla.
#
# PROFILE_MIN_ACCEPTED_STONES=2 (kayttajan pyynnosta, nostettu 1:sta):
# profiili sovitetaan YHTEISESTI kaikkien hyvaksyttyjen kivien havain-
# noista (katso fit_stone_profile) - kahdella kivella yhdella sijasta
# profiili ei ylisovitu yhden kiven omiin (esim. kulman/valaistuksen
# aiheuttamiin) satunnaisvirheisiin, joten se yleistyy paremmin MUIHIN
# kiviin joita ELAVA SEURANTA sen jalkeen kayttaa.
PROFILE_MIN_ACCEPTED_STONES = 2

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
SOLO_TRACK_MAX_RMS_PX = 3.0   # Testi_03_03: oli 12.0

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
# Kayttajan pyynnosta (katso keskusteluhistoria - alunperin 4, nostettu
# 8:aan): riittavasti tilaa samanaikaisesti aktiivisille kiville, myos
# silloin kun jokin (esim. paikallaan pysyva/hoglinen tuntumassa oleva)
# kohde tukkii yhden paikan pitkaksi aikaa - ei enaa estä uusien aitojen
# heittojen rekisteroitymista yhtä helposti. Yksinkertainen lahin-
# ehdokas-per-kivi -logiikka riittaa (kivet lahekkain vasta pysahtymisen
# jalkeen, jolloin ID:lla ei ole enaa merkitysta). Uusien kivien HAKU
# kaytta kamera9_02.py:n kiinteaa paata-rajattua vyohyketta (SEARCH_X/
# Y_*, k92-moduulista) - TOISIN kuin 3D-profiilin koko-radan-skannaus
# (Task 3): kivet HEITETAAN aina samaan suuntaan/paahan, joten kiinteä
# HAKU-vyohyke on jarkeva/tehokas tassa.
# ============================================================

MAX_CONCURRENT_STONES = 8

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
# suunnassa, TRACK_MAX_SPEED_Y_CM_S - kayttajan helposti muutettavissa
# alla; X-suunnassa, sivuttaisliike/curl, paljon hitaampaa - JOHDETTU
# suoraan Y-nopeudesta, TRACK_MAX_SPEED_X_FRACTION_OF_Y=10%:na siita,
# EI oma erillinen vakionsa), hakualue voidaan laskea
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

TRACK_MAX_SPEED_Y_CM_S = 300.0  # kayttajan helposti muutettava arvo (3 m/s)

# X-suunnan nopeusraja JOHDETAAN Y-nopeudesta (kayttajan pyynnosta) - EI
# oma erillinen vakionsa - aina TRACK_MAX_SPEED_X_FRACTION_OF_Y verran
# TRACK_MAX_SPEED_Y_CM_S:sta, joten X paivittyy automaattisesti kun
# Y-arvoa muutetaan yllä.
TRACK_MAX_SPEED_X_FRACTION_OF_Y = 0.10
TRACK_MAX_SPEED_X_CM_S = TRACK_MAX_SPEED_Y_CM_S * TRACK_MAX_SPEED_X_FRACTION_OF_Y
TRACK_MAX_BACKWARD_CM = 100.0

# ============================================================
# ABSOLUUTTINEN KESKIVIIVAETAISYYSRAJA (kayttajan pyynnosta, katso
# keskusteluhistoria): kayttajan omaan kokemukseen perustuva havainto
# tasta radasta/pelityylista - kivi joka on koskaan yli 1.2m
# keskiviivasta (X=0) on hyvin todennakoisesti EI kivi (SEURANTA on
# ajautunut pelaajaan/lakaisijaan tms.), koska aidot heitot pysyvat
# tallä radalla kaytannossa aina tata lahempana keskiviivaa - HAKU:n
# oma hakuvyohyke (kamera9_02.py:n SEARCH_X_HALF_WIDTH_CM=70cm) on jo
# tatakin tiukempi UUDEN kiven loytohetkella, mutta SEURANNAN paikal-
# linen haku (TRACK_HALF_RANGE_CM=35cm/frame) voi ajan mittaan ajautua
# hakuvyohykkeen ulkopuolelle jos se tarttuu jatkuvasti sivuttain
# liikkuvaan kohteeseen (esim. pyyhkija) - sama periaate kuin TRACK_
# MAX_BACKWARD_CM:ssa ylla, mutta X-suunnassa ja molempiin suuntiin.
# ============================================================
MAX_ABS_X_FROM_CENTERLINE_CM = 120.0

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

# HAVAITTU ONGELMA (kayttajan raportoima, katso keskusteluhistoria):
# pelaaja/lakaisija HAKU-alueella (katso kamera9_02.py:n SEARCH_Y_MIN/
# MAX_CM) voi silloin talloin TAYTTAA HAKU:n omankin tiukan hyvaksynta-
# kriteerin (search_new_stone vaatii refined.tarkka:n - OIKEAN pyorean
# reunan/rengasrakenteen LM-sovituksen, EI pelkkaa peitto-osuutta, katso
# stone_tracker.cpp:n search_new_stone-kommentti) YHDELLA framella
# esim. vaatteen/harjan reunan sattuessa hetkeksi ympyramaiseksi. Kun
# tama tapahtuu, SEURANTA jatkaa sen jalkeen kandidaattia LOYSEMMALLA
# per-frame-kynnyksella (TRACK_SCORE_THRESHOLD, katso kamera9_02.py -
# EI vaadi tarkka:a joka framella, koska "jatkuvuus on jo vahva prior"
# - jarkeva oletus AIDOLLE, jo vahvistetulle kivelle, joka voi hetkeksi
# olla osittain peitossa). Tama sallii kandidaatin, joka EI OIKEASTI ole
# kivi, "ajautua" pelaajan mukana useita sekunteja ja YLITTAA yllaolevan
# siirtymakynnyksen, tullen vahvistetuksi VIRHEELLISESTI.
#
# Todellisella datalla (katso keskusteluhistoria, fullrun_positions.csv,
# frame 4530 -> 4716, ihmiseksi silmamaarin vahvistettu): tama kandi-
# daatti saavutti tarkka=1:n vain 3/8 (38%) ensimmaisesta 8 havain-
# nostaan (ennen 25cm-vahvistuskynnysta), kun taas kaikki pitkaan
# seuratut, selvasti aidot kivet (>=500 rivia koko elinajalta) olivat
# 100% tarkka=1 samassa ikkunassa. Siksi: vaaditaan MYOS etta riittavan
# pitkan (>=MIN_PRECONFIRM_TARKKA_OBSERVATIONS havaintoa - lyhyemmalla
# otoksella tarkka-osuus on liian kohinainen yksittaisten aitojenkin
# heittojen alkumetreilla hylattavaksi, katso keskusteluhistorian
# tilastot) puskuroidun ikkunan tarkka-osuus ylittaa MIN_PRECONFIRM_
# TARKKA_FRACTION:in ENNEN vahvistusta - muuten koko puskuroitu
# havaintosarja hylataan (kuten "ei liikkunut riittavasti" -tapauksessa)
# eika kandidaattia enaa seurata.
MIN_PRECONFIRM_TARKKA_OBSERVATIONS = 8
MIN_PRECONFIRM_TARKKA_FRACTION = 0.4
PRECONFIRM_RESCUE_MAX_RMS = float(os.environ.get("PRECONFIRM_RESCUE_MAX_RMS", "12.0"))

# JATKUVA TARKKA-OSUUSTARKISTUS VAHVISTETUILLE KIVILLE (Testi_02_02,
# kayttajan pyynnosta) - katso kayttokohdan kommentti. Eri (loyhempi)
# kynnys kuin MIN_PRECONFIRM_*: vahvistettu kivi on jo läpaissyt
# tiukemman esitarkistuksen kerran, joten tama on lisasuoja PITKAAN
# kestavaa ajautumaa vastaan, ei alkuperainen suodatin.
#
# 50 -> 150 (kayttajan pyynnosta tehdyn SEURANNAN maskiyhdistelma-
# korjauksen - stone_tracker.cpp:n createForegroundFromWhitened -
# jalkeen havaittu tarve, katso myos confirmed_tarkka_window:in oma
# kommentti kayttokohdassa): korjaus tekee findContourNear:ista
# herkemman todelliselle, mutta LYHYTAIKAISELLE kosketukselle naapuri-
# kohteeseen (esim. heittaja/lakaisija VIELA kiven vieressa muutaman
# sekunnin ajan HETI heiton jalkeen, ennen kuin kivi jaa yksin) - tama
# nakyy matalana tarkka-osuutena vain ENSIMMAISTEN havaintojen ajan.
# Ikkuna (deque) sailyttaa vain VIIMEISIMMAT N havaintoa, mutta koska
# ensimmainen tarkistus tapahtuu heti kun ikkuna on TAYTTYNYT ENSI
# KERRAN (havainto N), lyhyt N (50 = 2s) tarkistaa VIELA TASMALLEEN
# saman (kontaminoituneen) jakson kuin kumulatiivinen laskuri olisi -
# ikkuna ei ehdi "unohtaa" mitaan ennen ensimmaista tarkistusta.
# Mitattu oikealla datalla: aito kivi hylattiin virheellisesti N=50:n
# kohdalla (19/50=38%), vaikka sen havainnot valilta t=622.68-623.12
# (juuri se jakso jonka kayttaja alunperin liputti X-hyppimisongelmana)
# olivat jo taydellisia (n_body 26-29, rms 0.5-0.9px) - lyhyt n. 1.5s
# kontaminaatio heti heiton jalkeen ei ollut viela ehtinyt "laimentua"
# lyhyessa ikkunassa, ja N=50 katkaisi seurannan JUURI TAMAN puhtaan
# jakson keskella. N=150 (6s) antaa tallaiselle lyhyelle alkukonta-
# minaatiolle tarpeeksi tilaa laimentua ENNEN ensimmaista tarkistusta -
# uudella koko radan ajolla sama kivi sailyi nyt seurattuna KOKO sen
# kayttajan alunperin liputtaman puhtaan jakson ajan (t=622.68-623.12,
# rms edelleen 0.5-0.9px) ja viela n. 6 sekuntia sen ohi, ennen kuin
# SEURANTA ajautui my­ohemmin (t~624.2+) toiseen, jo ERIKSEEN tunnet-
# tuun ja kayttajan tietoisesti rajaamaan ei-kriittiseen harjaan-
# ajautumis-ongelmaan (katso keskusteluhistoria - tata TOISTA ongelmaa
# EI korjata tassa, kayttajan nimenomaisesta pyynnosta). Aidot ongel-
# malliset kandidaatit (havaittu samasta ajosta: n_body 260-280, rms
# 25-33px koko elinkaarensa ajan yhden ainoan alkuhavainnon jalkeen,
# ei yhtaan pitkaa puhdasta jaksoa) jaavat silti selvasti alle 0.5:n
# kynnyksen N=150:n kohdallakin - vain n. 100 ylimaaraista (vaaraa)
# CSV-riviä ehtii kertya ennen hylkaysta lyhyemman N:n sijaan.
#
# 150 -> 650 (kayttajan huomio: "yli 20 heittoa pitaisi pystya
# seuraamaan", mutta HAKU-korjauksen jalkeenkin vain n. 16 heittoa
# sai "luotettavan" (disp>15m, tarkka>75%) leiman - juurisyy loytyi
# TASTA kynnyksesta): HAKU-korjauksen (createForegroundFromWhitened)
# jalkeen SEURANTA yhdistaa rakeisuus- ja etualamaskit - tama tekee
# findContourNear:ista herkemman havaitsemaan MYOS PITKAKESTOISEN,
# ihan tavallisen curling-tilanteen: lakaisija joka kavelee/lakaisee
# JATKUVASTI aidon, juuri heitetyn kiven VIERELLA/PAALLA KOKO heiton
# ajan (ei vain hetken alussa, kuten N=150:n oma tapaus ylla) - tama
# EI ole sama tapaus kuin "SEURANTA ajautui kokonaan pois oikealta
# kivelta johonkin toiseen kohteeseen" (jota tama tarkistus alunperin
# yritti torjua), koska X/Y-sijainti itsessaan pysyy koko ajan
# sileana ja fysikaalisesti jarkevana (Y vahenee tasaisesti/hidastuen
# kohti pesaa, X pysyy lahella keskiviivaa) - VAIN tarkka/n_body/rms
# nayttavat huonolta, koska loydetty yhtenainen kontuuri sisaltaa
# seka kiven etta lakaisijan. Mitattu oikealla datalla (00014_0202_
# fp5_positions.csv): AINAKIN 14 todennakoisesti aitoa heittoa (esim.
# id=134: X=0.1-0.7m koko ajan, Y vahenee sileasti 33.4m->19.1m,
# n_body vaihtelee 12:sta yli 500:aan mutta sijainti ei koskaan
# hyppaa) katkaistiin TASAN N=150:n (tai N:n lahella) kohdalla, vaikka
# muut, kokonaan puhtaat heitot samalla videolla kestivat luonnol-
# lisesti 472-613 havaintoa (19-24s) ennen omaa "pysahtynyt"-
# paattymistaan - N=150 (6s) ei siis riittanyt EDES YHDEN normaalin
# heiton luonnolliseen kestoon, joten se katkaisi tehokkaasti LAHES
# JOKAISEN lakaisijan saattaman heiton keskelta. N=650 (26s) on
# reilusti pidempi kuin pisin havaittu aito heitto (613 havaintoa) -
# antaa siis KAIKILLE aidoille heitoille (myos
# koko ajan lakaistuille) tilaa kestaa luonnolliseen loppuunsa
# ("pysahtynyt"-tarkistus tai videon loppu) asti, samalla kun aidosti
# ongelmalliset kandidaatit (nolla-lahella tarkka-osuus koko elin-
# kaarensa, ei koskaan puhdasta/liikkuvaa jaksoa) jaavat silti
# selvasti alle 0.5:n kynnyksen - vain hieman my­ohemmin (26s vs 6s)
# kuin ennen.
MIN_CONFIRMED_TARKKA_OBSERVATIONS = 650
MIN_CONFIRMED_TARKKA_FRACTION = 0.5

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
#
# HUOM (kayttajan loytama bugi MAH-videolla, katso keskusteluhistoria):
# tama vertailu koski AIEMMIN KAIKKIA active_stones-listan kiviä, myos
# viela VAHVISTAMATTOMIA (s["confirmed"]==False) ehdokkaita. Jos HAKU
# tunnisti pelaajan/lakaisijan jalan viereltä kiveksi (aito ongelma -
# pelkkaan muotoon/kokoon perustuva sovitus ei tunne varia), tama
# vahvistamaton "kivi" esti TAYSIN HILJAA (ei lokiviestia) sen VIERESSA
# olevan OIKEAN kiven rekisteroitymisen niin kauan kuin vaara ehdokas
# pysyi aktiivisena (jopa useita sekunteja) - koska etaisyys mitattiin
# myos siihen. Korjaus: dedup-vertailu tehdaan nyt vain jo VAHVISTETTUIHIN
# kiviin (s["confirmed"]) - kaksi AITOA, jo vahvistettua kiveä eivat
# edelleenkaan voi saada duplikaatti-ID:ta, mutta viela vahvistamaton
# (mahdollisesti vaara) ehdokas ei enaa voi tukkia vieressa olevan
# oikean kiven havaitsemista.
# Testi_02_03: HAKU palauttaa KAIKKI kelvolliset ehdokkaat (HAKU_MAX_RESULTS) eika vain ensimmaista -
# muuten pelaaja/lakaisija voittaa kilpailun ja vieressa oleva oikea kivi jaa rekisteroimatta.
# HAKU_MULTI=0 = alkuperainen (yksi ehdokas / kutsu).
# HEITTOPORTTI (Testi_02_03): videon lopussa varsinaiseen CSV:hen kirjoitetaan vain radat jotka ovat
# koko elinkaarensa perusteella HEITTOJA (ei pelaajia/lakaisijoita): matka eteenpain (Y pienenee)
# >= GATE_TRAVEL_CM, >= GATE_MIN_ROWS riviä, rms-mediaani <= GATE_MAX_RMS, tarkka-osuus >= GATE_MIN_TARKKA;
# duplikaattiradat yhdistetaan. RAAKA (portittamaton) CSV kirjoitetaan viereen tiedostoon *_raaka.csv.
# THROW_GATE=0 = alkuperainen (vain yksi raaka CSV).
THROW_GATE = os.environ.get("THROW_GATE", "1") == "1"
GATE_TRAVEL_CM = float(os.environ.get("GATE_TRAVEL_CM", "1500"))
GATE_MIN_ROWS = int(os.environ.get("GATE_MIN_ROWS", "300"))
GATE_MAX_END_Y_CM = float(os.environ.get("GATE_MAX_END_Y_CM", "1100"))   # lahihogline 823 cm + marginaali
GATE_MAX_SPEED_RATIO = float(os.environ.get("GATE_MAX_SPEED_RATIO", "0.6"))
GATE_RESCUE_MAX_RMS = float(os.environ.get("GATE_RESCUE_MAX_RMS", "12.0"))
GATE_RESCUE_MIN_TARKKA = float(os.environ.get("GATE_RESCUE_MIN_TARKKA", "0.1"))
GATE_CROSS_DEDUP_FRAMES = float(os.environ.get("GATE_CROSS_DEDUP_FRAMES", "40"))
GATE_HEAD_GAP_FRAMES = int(os.environ.get("GATE_HEAD_GAP_FRAMES", "30"))
GATE_HEAD_MAX_ROWS = int(os.environ.get("GATE_HEAD_MAX_ROWS", "40"))
GATE_CROSS_EXTRAPOLATE_MAX = float(os.environ.get("GATE_CROSS_EXTRAPOLATE_MAX", "60"))
GATE_MAX_RMS = float(os.environ.get("GATE_MAX_RMS", "3.0"))
GATE_MIN_TARKKA = float(os.environ.get("GATE_MIN_TARKKA", "0.3"))


def _track_kinematics(rows):
    """(matka, loppu-Y, hidastuvuussuhde, hogline-ylitysframe) radan riveista. Hidastuvuus =
    loppuvaiheen (viimeiset 20 % rivista) nopeus / alkuvaiheen (ensimmaiset 20 %) nopeus:
    aito kivi hidastuu (0.2-0.4), pelaajan/lakaisijan paa liikkuu tasaisella nopeudella (~1)."""
    ys = np.array([r["Y_cm"] for _, _, r in rows])
    fr = np.array([f for f, _, _ in rows])
    n = len(rows)
    a = max(2, n // 5)
    va = (ys[0] - ys[a]) / max(1, fr[a] - fr[0])
    vb = (ys[-a - 1] - ys[-1]) / max(1, fr[-1] - fr[-a - 1])
    ratio = vb / va if va > 0.5 else 9.9
    cross = None
    hog = k8.FAR_HOGLINE_Y_CM
    for i in range(n - 1):
        if ys[i] > hog >= ys[i + 1]:
            cross = float(fr[i])
            break
    if cross is None and ys[0] <= hog:
        cross = float(fr[0])
    return float(ys[0] - ys.min()), float(ys[-1]), float(ratio), cross


def _trim_track_head(rows):
    """Pudottaa radan alun, jos sen jalkeen on GATE_HEAD_GAP_FRAMES:ia pidempi rivitön aukko ja alku on lyhyt
    (< GATE_HEAD_MAX_ROWS riviä): rata on alussa lukkiutunut vääraan kohteeseen (esim. pelaajan jalka)
    ja löytänyt oikean kiven vasta aukon jalkeen. Havaittu: heitto #4 (rata 158)."""
    for i in range(min(len(rows) - 1, GATE_HEAD_MAX_ROWS)):
        if rows[i + 1][0] - rows[i][0] > GATE_HEAD_GAP_FRAMES:
            return rows[i + 1:]
    return rows


def _estimate_crossing(rows):
    """Kaukaisen hoglinen ylitysframe. Jos rata alkaa jo hoglinen ALAPUOLELTA (rekisteroity myohassa),
    ylitys ekstrapoloidaan taaksepain radan alun nopeudesta (enintaan GATE_CROSS_EXTRAPOLATE_MAX ruutua)."""
    cross = _track_kinematics(rows)[3]
    ys = np.array([r["Y_cm"] for _, _, r in rows])
    fr = np.array([f for f, _, _ in rows], dtype=float)
    hog = k8.FAR_HOGLINE_Y_CM
    if ys[0] <= hog and len(rows) >= 10:
        m = min(len(rows), 30)
        v = (ys[0] - ys[m - 1]) / max(1.0, fr[m - 1] - fr[0])   # cm/frame, Y pienenee -> v > 0
        if v > 1.0:
            back = min((hog - ys[0]) / v, GATE_CROSS_EXTRAPOLATE_MAX)
            return float(fr[0] - back)
    return cross


def _track_throw_class(rows):
    """0 = ei heitto, 2 = heitto hyvalla sovituksella, 1 = heitto heikolla sovituksella (esim. lakaisija
    peittaa osan kivesta): kaikilla tarvitaan heittomainen liike (matka >= GATE_TRAVEL_CM, loppu-Y <=
    GATE_MAX_END_Y_CM, hidastuvuus <= GATE_MAX_SPEED_RATIO, >= GATE_MIN_ROWS riviä)."""
    if len(rows) < GATE_MIN_ROWS:
        return 0
    travel, yend, ratio, _ = _track_kinematics(rows)
    if travel < GATE_TRAVEL_CM or yend > GATE_MAX_END_Y_CM or ratio > GATE_MAX_SPEED_RATIO:
        return 0
    rms = [r["rms_px"] for _, _, r in rows if r.get("rms_px") is not None]
    if not rms:
        return 0
    med = float(np.median(rms))
    tk = sum(1 for _, _, r in rows if r.get("tarkka")) / len(rows)
    if med <= GATE_MAX_RMS and tk >= GATE_MIN_TARKKA:
        return 2
    if med <= GATE_RESCUE_MAX_RMS and tk >= GATE_RESCUE_MIN_TARKKA:
        return 1
    return 0


def _select_throws(tracks):
    """tracks: [(stone_id, rows)] -> vain heittoportin lapaisseet, yksi rata / hogline-ylitys:
    kahden radan ylitysajat < GATE_CROSS_DEDUP_FRAMES toisistaan = sama heitto (kayttajan tieto: joka
    kerta vain YKSI kivi ylittaa hoglinen) -> sailyy parempi (hyva sovitus ensin, sitten pienin rms)."""
    cands = []
    for sid, rows in tracks:
        rows = _trim_track_head(rows)
        cls = _track_throw_class(rows)
        if cls == 0:
            continue
        rms = [r["rms_px"] for _, _, r in rows if r.get("rms_px") is not None]
        cross = _estimate_crossing(rows)
        cands.append((-cls, float(np.median(rms)), -len(rows), sid, rows, cross))
    cands.sort(key=lambda c: c[:3])
    kept = []
    for c in cands:
        cross = c[5]
        if cross is not None and any(
            k[5] is not None and abs(k[5] - cross) < GATE_CROSS_DEDUP_FRAMES for k in kept
        ):
            continue
        kept.append(c)
    return sorted(((k[3], k[4]) for k in kept), key=lambda t: t[1][0][0])



# ============================================================
# HOG-HOG -ANALYYSI (Testi_05_01): kun kivi on kulkenut riittavasti (Y <= lahihog + 50 cm), otetaan radan pisteet valilta [lahihog + 50 cm, kaukohog - 100 cm], sovitetaan
# Y(t) = a t^2 + b t + c, pudotetaan 10 huonoiten sopivaa pistetta ja sovitetaan uudelleen. Jos R > 0.99: nopeus kaukohoglinella, keskihidastuvuus alueella ja hog-hog-aika
# (yhtalon mukaan) tulostetaan terminaaliin ja kirjataan debug-videon kuvaan lahemman hoglinen luona radan sivuun HOG_OVERLAY_SECONDS ajaksi (+ still-kuva ja <csv>_hog.csv).
# HOG_ANALYSIS=0 kytkee pois. Katso hog_analyysi.py.
# ============================================================
HOG_ANALYSIS = os.environ.get("HOG_ANALYSIS", "1") == "1"
HOG_OVERLAY_SECONDS = float(os.environ.get("HOG_OVERLAY_SECONDS", "5"))
HOG_SAVE_SNAPSHOT = os.environ.get("HOG_SAVE_SNAPSHOT", "1") == "1"


def _render_debug_frame(frame_u, gain, bias, items, pose, header, results, plus_right, composer, local_pts_body):
    """Debug-videon yksi ruutu (ajetaan taustasaikeessa): valokorjattu kuva + ennustetut ääriviivat + paneelit."""
    if gain is not None:
        img = apply_photometric_correction(frame_u, gain, bias)
    else:
        img = frame_u.copy()
    img = np.ascontiguousarray(img)
    labels = []
    for bx, by, s_id, color, label in items:
        hull = k94.predicted_stone_hull_fast(local_pts_body, pose, bx, by)
        if hull is None:
            continue
        hull_i = hull.astype(np.int32)
        cv2.polylines(img, [hull_i], True, color, 2)
        labels.append((int(hull_i[:, 0, 0].min()), int(hull_i[:, 0, 1].min()) - 8, label, color))
    return composer.compose(img, labels, header, results, plus_right)


def _hog_check(s, frame_index, fps, near_hog, far_hog, overlays, results, frame_img, csv_output, pose):
    rows = s.get("all_rows")
    if not rows:
        return
    y = rows[-1][2].get("Y_cm")
    if y is None:
        return
    if "hog_result" not in s and y <= near_hog + hog_analyysi.NEAR_MARGIN_CM:
        res = hog_analyysi.analyze_hog(rows, near_hog, far_hog, tee_cm=k8.NEAR_HOUSE_Y_CM)
        res["stone_id"] = s["stone_id"]; res["frame"] = frame_index
        s["hog_result"] = res
        if res.get("ok"):                 # TULOSTETAAN VAIN jos R_y > 0.99, R_x > 0.99 ja R_y * R_x > 0.99 (muuten ei mitaan)
            results.append(res)
            print(f"[frame {frame_index}] " + " | ".join(hog_analyysi.format_lines(res, s["stone_id"])))
    r = s.get("hog_result")
    if r and r.get("ok") and not s.get("hog_overlay_started") and y <= near_hog:
        s["hog_overlay_started"] = True
        lines = hog_analyysi.format_lines(r, s["stone_id"])
        overlays.append(dict(start=frame_index, end=frame_index + int(round(HOG_OVERLAY_SECONDS * fps)), lines=lines, stone_id=s["stone_id"]))
        if HOG_SAVE_SNAPSHOT and frame_img is not None and csv_output:
            try:
                snap = hog_analyysi.draw_overlay(frame_img.copy(), lines, pose["K"], pose["R"], pose["t"], near_hog)
                snap_path = os.path.splitext(csv_output)[0] + f"_hog_kivi{s['stone_id']}.png"
                cv2.imwrite(snap_path, snap)
            except Exception as e_:       # still-kuva ei saa kaataa seurantaa
                print(f"[hog] still-kuvan tallennus epaonnistui: {e_}")


def _hog_write_csv(results, csv_output):
    if not results:
        return
    path = os.path.splitext(csv_output)[0] + "_hog.csv"
    cols = ["stone_id", "frame", "R", "R_ennen_suodatusta", "n_kaytetty", "n_pudotettu", "v_far_hog_ms", "decel_ms2", "hog_hog_s", "t_far_hog_s", "t_near_hog_s", "v_near_hog_ms", "a", "b", "c",
            "R_x", "R_x_ennen_suodatusta", "R_tulo", "x_far_hog_cm", "dir_far_hog_deg", "slope_dxdy", "x_straight_at_tee_cm", "px", "qx", "rx",
            "liuku_x_tee_cm", "liuku_dir_deg", "liuku_n", "liuku_rms_cm"]
    with open(path, "w", newline="") as hf:
        w = csv.writer(hf); w.writerow(cols)
        for r in results:
            w.writerow([r.get(c_) for c_ in cols])
    print(f"Hog-hog -analyysi: {len(results)} heittoa -> {path}")

HAKU_MULTI = os.environ.get("HAKU_MULTI", "1") == "1"
HAKU_MAX_RESULTS = int(os.environ.get("HAKU_MAX_RESULTS", "4"))
HAKU_MAX_ATTEMPTS = int(os.environ.get("HAKU_MAX_ATTEMPTS", "8"))
NEW_STONE_SAME_SCAN_CM = float(os.environ.get("NEW_STONE_SAME_SCAN_CM", "30.0"))
NEW_STONE_DEDUP_CM = float(os.environ.get("NEW_STONE_DEDUP_CM", "20.0"))

# Testi_02_03: kaksi rataa jotka ovat DUP_MERGE_CM:n sisalla toisistaan SAMALLA
# framella (molemmat loysivat kohteensa) tulkitaan samaksi kiveksi ja uudempi
# (tai vahvistamaton) poistetaan. Tekee mahdolliseksi pienentaa NEW_STONE_DEDUP_CM:aa
# (vahvistettu ROSKARATA ei enaa estä oikean kiven rekisteroitymista), koska
# mahdolliset duplikaatit siivotaan tassa. 0 = pois. HUOM: Testi_02_03:n OLETUS on 30 (+ NEW_STONE_DEDUP_CM=20);
# alkuperainen Testi_02_02-kayttaytyminen: NEW_STONE_DEDUP_CM=100 DUP_MERGE_CM=0.
DUP_MERGE_CM = float(os.environ.get("DUP_MERGE_CM", "30"))
DUP_MERGE_FRAMES = int(os.environ.get("DUP_MERGE_FRAMES", "3"))

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
# Varitarkistus vaikutti aiemmin (Testi_02_01, alkuperainen versio)
# "pysahtynyt"-ilmoitukseen: liukuvan ikkunan (color_diff_history)
# KESKIARVON piti olla aidosti graniittimainen (alle COLOR_STOP_MAX_
# AVG_DIFF) ennen kuin pysahtyminen hyvaksyttiin. TARKOITUS oli etta
# tama EI voisi koskaan aiheuttaa oikean kiven katoamista (ilmoitus
# vain "lykkaantyisi") - mutta koko videon lapikaynti (kayttajan
# pyynnosta, katso keskusteluhistoria) osoitti etta oletus oli VAARIN:
# aidosti paikallaan olevia kivia EI koskaan tulostettu pysahtyneeksi
# (avg_diff pysyi kynnyksen ylapuolella), jolloin ne jaivat pysyvasti
# aktiivisiksi ja tukkivat MAX_CONCURRENT_STONES-paikat - estaen uusien
# aitojen heittojen rekisteroinnin (havaittu konkreettisesti: viimeinen
# heitto jai kokonaan puuttumaan CSV:sta). Varitarkistus on siis
# POISTETTU "pysahtynyt"-paatoksesta - katso SEURANTA-silmukan oma
# kommentti (STOP_TRACKING_DISPLACEMENT_CM/SECONDS RIITTAA yksinaan).
# color_match_median_diff/color_diff_history sailyvat silti (COLOR_DEBUG-
# diagnostiikkaa varten), vain paatoksentekoon ei enaa vaikuta.
_COLOR_DEBUG = bool(os.environ.get("COLOR_DEBUG"))   # v5.7: SEURANNAN varidiagnostiikka lasketaan vain tassa tilassa
COLOR_REF_MIN_RING_SPACING_PX = 2.5
COLOR_REF_MIN_OBSERVATIONS = 3
COLOR_MATCH_MIN_VALID_POINTS = 20

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
                                      shadow_v_drop_max, shadow_v_drop_min=0.0):
    """ENABLE_SHADOW_TOLERANT_STABILIZATION:in ydinsaanto (katso sen
    kommentti): tausta = (tavallinen pieni erotus) TAI (saturaatio
    lahes sama mutta V-pudotus valilla (shadow_v_drop_min,
    shadow_v_drop_max) - eli varjo, ei aito objekti) TAI (absoluuttinen
    jaa-suodatus, katso ICE_S_MAX/ICE_V_MIN:in kommentti - positio-
    riippumaton, toimii vaikka pikseli poikkeaisi referenssista).
    shadow_v_drop_min voi olla negatiivinen (sallii pienen V:n NOUSUN
    silti taustaksi - kayttajan kanssa kasin testattu, katso
    SHADOW_V_DROP_MIN:in kommentti)."""

    diff = cv2.absdiff(frame_bgr, reference_bgr)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    background_mask = diff_gray < diff_threshold

    frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV).astype(np.int16)
    ref_hsv = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2HSV).astype(np.int16)
    v_drop = ref_hsv[..., 2] - frame_hsv[..., 2]
    shadow_mask = (v_drop > shadow_v_drop_min) & (v_drop < shadow_v_drop_max)

    ice_mask = (
        (frame_hsv[..., 1] < ICE_S_MAX) & (frame_hsv[..., 2] > ICE_V_MIN)
    )

    bg = background_mask | shadow_mask | ice_mask
    if COLOR_GATE:
        # Testi_03_04 v4.5-koe: varitarkistus - suodata vain jos S_ref (moodikuvan kylläisyys) < COLOR_GATE_S_MAX TAI sama savy kuin referenssissa (|dH| <= COLOR_GATE_H_TOL, H 0..179 ymparoi)
        dh = np.abs(frame_hsv[..., 0] - ref_hsv[..., 0])
        dh = np.minimum(dh, 180 - dh)
        low_s = ref_hsv[..., 1] < COLOR_GATE_S_MAX                                   # S = MOODIKUVAN kylläisyys
        if COLOR_GATE_BOTH:
            low_s &= frame_hsv[..., 1] < COLOR_GATE_S_MAX
        hue_ok = dh <= COLOR_GATE_H_TOL
        if COLOR_GATE_HUE_SAT:
            hue_ok &= frame_hsv[..., 1] > COLOR_GATE_S_MAX
        bg &= low_s | hue_ok
    return bg


def suppress_static_background(frame_bgr, reference_bgr, diff_threshold=30):

    if reference_bgr is None or frame_bgr.shape != reference_bgr.shape:
        return frame_bgr

    if ENABLE_SHADOW_TOLERANT_STABILIZATION:
        background_mask = _shadow_tolerant_background_mask(
            frame_bgr, reference_bgr, diff_threshold, SHADOW_V_DROP_MAX,
            SHADOW_V_DROP_MIN
        )
    else:
        diff = cv2.absdiff(frame_bgr, reference_bgr)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        background_mask = diff_gray < diff_threshold

    out = frame_bgr.copy()
    out[background_mask] = (255, 255, 255)

    return out


# ============================================================
# VALOTASAPAINO/KIRKKAUS-KORJAUS "LOPPUVIDEOLLE" (kayttajan pyynnosta,
# katso keskusteluhistoria): pikselitarkka geometrinen stabilointi (katso
# PIKSELITARKKA SUORA STABILOINTI) EI korjaa valon/kameran auto-
# valotuksen/valkotasapainon HIDASTA ajautumista pitkan (>10 min) videon
# aikana - kayttaja huomasi etta debug-videossa jaa/katto/seinat eivat
# olleet TAYSIN valkoisia myohemmin videolla, vaikka geometrinen
# kohdistus oli kunnossa (vahvistettu diff-heatmapilla: HAJA, ei terava,
# ero - viittaa valotasapainoon/kirkkauteen, ei sijaintivirheeseen).
#
# ENSIMMAINEN yritys (histogrammin persentiilikohdistus koko framelle)
# HUONONSI tulosta kaikilla testatuilla frameilla (+7% - +107% vuotoa) -
# jaan lahes saturoitunut valkoisuus + etualan (pelaajien) poikkeavat
# varit vaaristivat globaalin persentiilipohjaisen arvion.
#
# TOIMIVA ratkaisu (kayttajan ehdotuksesta): kayttaa SAMAA periaatetta
# kuin stabiloinnin siirtymahaku - EI kiintea/analyyttinen kaava koko
# kuvalle, vaan ROBUSTI ITEROITU pienimman neliosumman sovitus (per
# BGR-kanava): sovitetaan lineaarinen gain+bias frame->referenssi,
# poistetaan sovituksen JALKEEN suurimman jaannoksen pikselit (etuala:
# pelaajat, kivet - niiden varit eivat mitenkaan liity valotasapainoon)
# ja toistetaan - lahentyy nopeasti (3 kierrosta riittaa) taustan
# TODELLISEEN foto­metriseen suhteeseen. Validoitu: n. sama parannus
# kuin raaka ristikkohaku fg-pikselien maaralla mitattuna (esim. 194114
# -> 92657 vs. haun 90738), mutta ~15x nopeampi (~250-750ms/kutsu vs.
# 5-7s/kutsu) - silti liian hidas JOKA framelle (25fps-budjetti 40ms),
# joten kaytetaan vain KERRAN SEKUNNISSA (katso kaytto run_pipeline:ssa)
# - valotasapainon ajautuminen on hidasta, ei tarvitse paivittaa joka
# framella kuten geometrinen siirtyma.
# ============================================================

def _robust_gain_bias_single_channel(frame_values, reference_values, n_iter=3):
    """Palauttaa (gain,bias) joka minimoi (gain*frame_values+bias -
    reference_values)**2:n, ROBUSTISTI - jokaisen kierroksen jalkeen
    suurimman jaannoksen (80. persentiili ylittavat, tyypillisesti
    etualan/pelaajien/kivien pikselit) pikselit poistetaan seuraavasta
    kierroksesta."""

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

        residual = np.abs(
            reference_values - (gain * frame_values + bias)
        )
        threshold = np.percentile(residual, 80)
        mask = residual < threshold

    return gain, bias


def estimate_photometric_correction(frame_bgr, reference_bgr):
    """Ajaa _robust_gain_bias_single_channel:in erikseen jokaiselle BGR-
    kanavalle - palauttaa (gains,biases), molemmat 3-alkioisia listoja.
    Kanavakohtaisuus kattaa seka yleisen KIRKKAUDEN (kaikki kanavat
    samansuuntaisesti) etta VALKOTASAPAINON/varisavyn ajautumisen (kanavat
    eri suuntiin) yhdella samalla mekanismilla."""

    gains = []
    biases = []

    # Testi_03_01: estimoidaan harvennetusta pikseliruudukosta (joka PHOTO_SUBSAMPLE:s
    # pikseli molemmissa suunnissa) - gain/bias on koko kuvan hidas globaali suure,
    # 1/16 pikseleista riittaa (ero koko kuvan estimaattiin ~1e-4), ~100 -> ~7 ms.
    st_ = PHOTO_SUBSAMPLE
    frame_bgr = frame_bgr[::st_, ::st_]
    reference_bgr = reference_bgr[::st_, ::st_]

    for channel in range(3):

        gain, bias = _robust_gain_bias_single_channel(
            frame_bgr[:, :, channel].astype(np.float64).ravel(),
            reference_bgr[:, :, channel].astype(np.float64).ravel()
        )

        gains.append(gain)
        biases.append(bias)

    return gains, biases


_PHOTO_LUT_CACHE = {"key": None, "lut": None}


def apply_photometric_correction(frame_bgr, gains, biases):
    # Testi_05_02: kanavakohtainen LUT (cv2.LUT) float32-ruutu-numpyn sijaan (9 ms -> 0.8 ms, tulos TASMALLEEN sama: clip(v*gain + bias, 0, 255) float32:na -> uint8).
    key = (tuple(float(g) for g in gains), tuple(float(b) for b in biases))
    if _PHOTO_LUT_CACHE["key"] != key:
        x = np.arange(256, dtype=np.float32)
        lut = np.stack([np.clip(x * gains[c] + biases[c], 0, 255).astype(np.uint8) for c in range(3)], axis=1).reshape(256, 1, 3)
        _PHOTO_LUT_CACHE["key"], _PHOTO_LUT_CACHE["lut"] = key, lut
    return cv2.LUT(frame_bgr, _PHOTO_LUT_CACHE["lut"])


_hann_window_cache = {}


def _phase_correlate_full_frame(gray_a_full, gray_b_full):
    """Yhteinen vaihekorrelaatio-ydin (kayttaa SUBPIXEL_ALIGN_CROP_
    FRACTION-rajausta + valimuistitettua Hanning-ikkunaa) - kaytetaan
    "loppuvideon" (profiiliskannaus + elava seuranta) PIKSELITARKASSA
    SUORASSA STABILOINNISSA (katso sen kommentti run_pipeline:ssa):
    jokainen frame verrataan suoraan moodikuvareferenssiin taman
    funktion kautta.
    gray_a_full/gray_b_full: TAYSRESOLUUTIOISET harmaasavykuvat (uint8
    tai float32). Palauttaa (dx,dy): siirto joka pitaa lisata jotta
    gray_b linjautuisi gray_a:n kanssa, SUBPIXEL_ALIGN_RANGE_PX:aan
    rajattuna."""

    h, w = gray_a_full.shape[:2]
    cw = int(round(w * SUBPIXEL_ALIGN_CROP_FRACTION))
    ch = int(round(h * SUBPIXEL_ALIGN_CROP_FRACTION))
    x0 = (w - cw) // 2
    y0 = (h - ch) // 2

    key = (x0, y0, cw, ch)
    hann = _hann_window_cache.get(key)
    if hann is None:
        hann = cv2.createHanningWindow((cw, ch), cv2.CV_32F)
        _hann_window_cache[key] = hann

    a_crop = gray_a_full[y0:y0 + ch, x0:x0 + cw].astype(np.float32)
    b_crop = gray_b_full[y0:y0 + ch, x0:x0 + cw].astype(np.float32)

    (dx, dy), _response = cv2.phaseCorrelate(a_crop, b_crop, hann)

    dx = float(np.clip(dx, -SUBPIXEL_ALIGN_RANGE_PX, SUBPIXEL_ALIGN_RANGE_PX))
    dy = float(np.clip(dy, -SUBPIXEL_ALIGN_RANGE_PX, SUBPIXEL_ALIGN_RANGE_PX))
    return dx, dy


_phase_ref_cache = {}


def _phase_correlate_cached(ref_gray, gray):
    """Nopea vaihekorrelaatio moodikuvaa (ref_gray) vasten (Testi_03_01): sama tulos kuin
    _phase_correlate_full_frame (virhe < 1e-6 px), mutta referenssin ikkunoitu FFT
    valimuistitetaan ja elementtikohtaiset vaiheet ajetaan C++:ssa (stone_tracker.phase_*).
    Palaa tarvittaessa hitaaseen versioon (rajaus != koko frame tai puuttuva C++-tuki)."""
    h, w = ref_gray.shape[:2]
    if (
        not hasattr(stone_tracker, "phase_mulnorm")
        or SUBPIXEL_ALIGN_CROP_FRACTION != 1.0
        or (h & 1) or (w & 1)
        or cv2.getOptimalDFTSize(h) != h or cv2.getOptimalDFTSize(w) != w
        or gray.shape[:2] != (h, w)
    ):
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
    dx = float(np.clip(dx, -SUBPIXEL_ALIGN_RANGE_PX, SUBPIXEL_ALIGN_RANGE_PX))
    dy = float(np.clip(dy, -SUBPIXEL_ALIGN_RANGE_PX, SUBPIXEL_ALIGN_RANGE_PX))
    return dx, dy


# ============================================================
# PELIALUE PROFIILIN OPETTELUVAIHEESSA (Testi_03_03): kivia etsitaan VAIN alueelta X = -2...+2 m, Y = 4...20 m
# (fyysiset koordinaatit). Kuvasta kasitellaan (taustanvaimennus, graniittimaski, ääriviivat) vain tata aluetta
# vastaava suorakulmio (PLAY_ROI): alueen 8 kulmaa (Z = 0 ja Z = kiven korkeus) projisoidaan kuvaan.
# PLAY_POS_SLACK_CM: parallaksivara ehdokkaan sijaintisuodatukseen (kiven keskipiste on ~7 cm jaan ylapuolella).
# ============================================================
PLAY_X_MIN_CM = -200.0
PLAY_X_MAX_CM = 200.0
PLAY_Y_MIN_CM = 400.0
PLAY_Y_MAX_CM = 2000.0
PLAY_POS_SLACK_CM = 25.0
PLAY_ROI_MARGIN_PX = 12
PLAY_STONE_HEIGHT_CM = 16.0


def _play_area_bounds():
    return (
        PLAY_X_MIN_CM - PLAY_POS_SLACK_CM, PLAY_X_MAX_CM + PLAY_POS_SLACK_CM,
        PLAY_Y_MIN_CM - PLAY_POS_SLACK_CM, PLAY_Y_MAX_CM + PLAY_POS_SLACK_CM,
    )


_play_roi_cache = {}


def _play_area_roi(pose, frame_w, frame_h):
    """Pelialueen (X,Y-suorakulmio, Z = 0...kiven korkeus) projektion rajaava suorakulmio (x0, y0, x1, y1) px."""
    key = (id(pose), frame_w, frame_h)
    roi = _play_roi_cache.get(key)
    if roi is not None:
        return roi
    xs = (PLAY_X_MIN_CM - PLAY_POS_SLACK_CM, PLAY_X_MAX_CM + PLAY_POS_SLACK_CM)
    ys = (PLAY_Y_MIN_CM - PLAY_POS_SLACK_CM, PLAY_Y_MAX_CM + PLAY_POS_SLACK_CM)
    zs = (0.0, PLAY_STONE_HEIGHT_CM)
    pts = np.array([[x, y, z] for x in xs for y in ys for z in zs], dtype=np.float64)
    u, v = k9._project_3d(pose["K"], pose["R"], pose["t"], pts)
    ok = np.isfinite(u) & np.isfinite(v)
    if not np.any(ok):
        roi = (0, 0, frame_w, frame_h)
    else:
        m = PLAY_ROI_MARGIN_PX
        roi = (
            int(max(0, math.floor(u[ok].min()) - m)), int(max(0, math.floor(v[ok].min()) - m)),
            int(min(frame_w, math.ceil(u[ok].max()) + m)), int(min(frame_h, math.ceil(v[ok].max()) + m)),
        )
    _play_roi_cache.clear()
    _play_roi_cache[key] = roi
    return roi


def _scan_stone_candidates(frame_bgr, calib, pose, background_reference=None):
    h_img, w_img = frame_bgr.shape[:2]
    roi = _play_area_roi(pose, w_img, h_img)
    x0, y0, x1, y1 = roi
    # Testi_03_03: taustanvaimennus vain pelialueen kuva-alueella; ulkopuoli valkoista (= ei kivea)
    if background_reference is not None and frame_bgr.shape == background_reference.shape:
        suppressed = np.full_like(frame_bgr, 255)
        suppressed[y0:y1, x0:x1] = suppress_static_background(
            frame_bgr[y0:y1, x0:x1], background_reference[y0:y1, x0:x1]
        )
    else:
        suppressed = frame_bgr
    return k9._candidates_in_frame(
        suppressed, calib, pose, calib["H_final"],
        k9.STONE_TRACK_MIN_AREA, k9.STONE_TRACK_MIN_FILL_RATIO,
        k9.STONE_TRACK_MIN_ASPECT_RATIO,
        bounds=_play_area_bounds(), roi=roi
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
STONE_TRACK_PRECHECK_MAX_RMS_PX = 8.0   # Testi_03_03: oli 24.0
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
                                   background_reference_undistorted=None,
                                   skip_precheck=False):

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
    x_min, x_max, y_min, y_max = _play_area_bounds()      # Testi_03_03: pelialue (ei koko kentta + marginaali)
    pixels_per_cm = k8.PIXELS_PER_CM
    output_x_min_cm = k8.OUTPUT_X_MIN_CM
    output_y_max_cm = k8.OUTPUT_Y_MAX_CM

    cap = live_source.open_capture(video_path)      # Testi_06_01: live-tilassa puskurista
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    map1, map2 = k94._build_undistort_maps(
        camera_matrix, dist_coeffs, (frame_w, frame_h)
    )
    play_roi = _play_area_roi(pose, frame_w, frame_h)

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
    def _scan_one(frame):
        frame_u = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)
        return stone_tracker.scan_stone_candidates(
            frame_u, background_reference_undistorted,
            H_final, K, R, t,
            min_area, max_area, min_fill_ratio, min_aspect_ratio,
            x_min, x_max, y_min, y_max,
            pixels_per_cm, output_x_min_cm, output_y_max_cm,
            30.0, *play_roi
        )

    def scan_indices(indices):
        if not indices:
            return {}
        wanted = set(indices)
        cache = {}
        cap.set(cv2.CAP_PROP_POS_FRAMES, indices[0])
        idx = indices[0]
        last = indices[-1]
        # v5.7: ruudut luetaan jarjestyksessa tassa saikeessa, mutta remap + kandidaattiskannaus (GIL vapaana) ajetaan
        # SCAN_WORKERS-saikeessa rinnan. Ruudut ovat toisistaan riippumattomia ja tulos tallennetaan ruudun indeksilla -> sama tulos.
        pending = {}
        ex = ThreadPoolExecutor(max_workers=SCAN_WORKERS) if SCAN_WORKERS > 1 else None
        try:
            while idx <= last:
                ok, frame = cap.read()
                if not ok:
                    break
                if idx in wanted:
                    if ex is None:
                        cache[idx] = _scan_one(frame)
                    else:
                        pending[idx] = ex.submit(_scan_one, frame)
                        if len(pending) >= 2 * SCAN_WORKERS:      # rajoitetaan muistissa odottavia ruutuja
                            oldest = min(pending)
                            cache[oldest] = pending.pop(oldest).result()
                idx += 1
            for i in sorted(pending):
                cache[i] = pending[i].result()
        finally:
            if ex is not None:
                ex.shutdown(wait=True)
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

    if (not skip_precheck) and len(precheck_track) >= STONE_TRACK_PRECHECK_MIN_SAMPLES:

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


PROFILE_FIT_CPP = os.environ.get("PROFILE_FIT_CPP", "1") == "1"   # C++-sovitus (fit_stone_profile_cpp), 0 = Python-versio


def _fit_stone_profile(pose, stones):
    """k9.fit_stone_profile C++:lla (Testi_03_03): sama malli/residuaali/LM, ~100x nopeampi. Palaa Python-versioon
    jos C++-funktiota ei ole tai kayttaja kytki sen pois (PROFILE_FIT_CPP=0)."""
    if not (PROFILE_FIT_CPP and hasattr(stone_tracker, "fit_stone_profile_cpp")) or len(stones) < 2:
        return k9.fit_stone_profile(pose, stones)
    positions0 = []
    for stone in stones:
        (cx, cy), _, _ = stone["ellipse"]
        X0, Y0 = k9._stone_ground_position_z0(pose, cx, cy)
        positions0.append((X0, Y0))
    return stone_tracker.fit_stone_profile_cpp(
        np.asarray(pose["K"], dtype=np.float64), np.asarray(pose["R"], dtype=np.float64),
        np.asarray(pose["t"], dtype=np.float64),
        [np.ascontiguousarray(s_["contour"], dtype=np.float64) for s_ in stones],   # Testi_03_04: liukuluvut (alipikselireuna)
        np.asarray(positions0, dtype=np.float64),
        40, float(k9.STONE_NOMINAL_RADIUS_CM), float(k9.STONE_HEIGHT_CM), float(k9.STONE_HEIGHT_MAX_CM),
        float(k9.HANDLE_NOTCH_R_FRAC_MIN), float(k9.HANDLE_NOTCH_R_FRAC_MAX), float(k9.HANDLE_NOTCH_R_FRAC),
        float(k9.STONE_SHAPE_REG_WEIGHT), 100
    )


# ============================================================
# ALFA-AARIVIIVA (Testi_03_04): kiven reuna alipikselin tarkkuudella PEITTAVYYDESTA, ei binaarimaskista.
#
# MIKSI: kiven reunapikselit ovat graniitin ja jaan sekoituksia. Varjosuodatin + binaarimaski (+ findContours, joka kulkee
# reunapikselien KESKIPISTEITA pitkin) sisentaa reunaa ~1-2 px => R ~12 cm (oikea ~14). Seurannan oma aariviiva taas vuotaa
# varjoon ja viivoihin (R ~14 mutta hajonta suuri, kivi 244: R=35). Alfa-kartta korjaa molemmat:
#
#   k     = V / V_ref            (V = kirkkain varikanava valotasapainon jalkeen, V_ref = kalibrointikuvan (moodikuva) V)
#   kg    = graniitin k           (maskin A ytimen mediaani; A = varjosuodatettu graniittimaski)
#   ks    = paikallinen taustataso (k:n normalisoitu konvoluutio kiven ULKOPUOLELTA -> seuraa jaata ja varjoa)
#   alfa  = (ks - k) / (ks - kg)  rajattuna 0..1 = kuinka suuri osa pikselista on graniittia
#
# Reuna = alfa-kartan tasa-arvokayra tasolla ALPHA_LEVEL (4x ylinaytteistys, bicubic), kuminauha (kupera peite, ei koloja),
# rajattuna graniittimaskiin A (laajennettuna ALPHA_A_DILATE_PX): yla- ja kahvan puoli pysyvat graniittimaskista (kahva = lovi).
#
# YHTEINEN ALFA-TASO: graniitin V / jaan V (nyt ~0.434, vaihtelu kivien valilla vain +-0.015). Maaritetaan profiilin opettelussa
# KAIKKIEN hyvaksyttyjen havaintojen mediaanina ja kaytetaan kaikille kiville (ei kivikohtaista saatoa - ei parantanut mitaan).
# Yksittaisen havainnon suhde hylataan jos se on valilla ALPHA_RATIO_MIN..MAX ulkopuolella.
#
# HYLKAYSSAANNOT (kivi 222: lakaisijan harja/kenka sotki ruudut):
#   1) tumma alue kiven ymparilla: > ALPHA_DARK_RING_MAX_PX pikselia joilla V < ALPHA_DARK_V_MAX renkaassa (31 px - 9 px laajennus A:sta)
#   2) aariviivan pinta-ala poikkeaa > ALPHA_AREA_TOL lahiruutujen (+-ALPHA_AREA_WINDOW) liukuvasta mediaanista (radan sisalla, koska
#      koko muuttuu etaisyyden mukana) - poimii esim. maalatun keskiviivan joka liittyy kiveen.
#   Sovituksen oma poikkeamakarsinta sailyy.
#
# TODO (seuranta, ei viela tehty): alfa-kartta + yhteinen alfa-taso parantaisi todennakoisesti LUOTTAMUSTA myos live-seurannassa
# (kuvassa nakyva kiven reuna ilman varjoa). Tarkastellaan kun kaydaan seurantaa lapi. Alfa-laskenta on C++:ssa
# (stone_tracker.alpha_observation_cpp / alpha_contour_cpp, ~7 ms/havainto, tulos identtinen numpy/cv2-versioon (ALPHA_CPP=0)); tarkempi optimointi myohemmin.
# ============================================================
# HAKU:n SILUETTITARKENNUS (oletus PAALLA, HAKU_SILHOUETTE=0 kytkee pois): HAKUn loytaman ehdokkaan paikka tarkennetaan kiven 3D-mallin siluetilla graniittimaskista
# (katso haku_silhouette.py). HAKU_SIL_LOG=polku.csv kirjaa jokaisen tarkennuksen.
HAKU_SILHOUETTE = os.environ.get("HAKU_SILHOUETTE", "1") == "1"   # Testi_03_04: oletuksena PAALLA (0 = pois)
HAKU_SIL_LOG = os.environ.get("HAKU_SIL_LOG")
# SEURANNAN MASKITUKI + SILUETTITARKENNUS (Testi_03_04, oletus PAALLA, SEURANTA_SILHOUETTE=0 kytkee pois): kun LM-tarkennus epaonnistuu (tarkka=0, tyypillisesti radan alkupaa,
# rms ~9 px), paikka tarkistetaan graniittimaskista: kiven 3D-mallin siluetin SISALLA pitaa olla vahintaan SEURANTA_MIN_INSIDE (0.4) maskipikselien osuus, muuten ruutu
# hylataan (miss) - esim. pelaajan paa (kiven kokoinen, pyorea, mutta ei graniittia) ei voi enaa vieda seurantaa. Hyvaksytty paikka tarkennetaan siluetilla
# (max SEURANTA_SIL_SHIFT_PX). Tarkat (tarkka=1) LM-paikat jatetaan ennalleen. Katso haku_silhouette.py ja tulokset/seuranta_*.png (kivi 51: seuranta seurasi paata 80 ruutua, virhe ~285 cm ->
# maskituella 17 cm; kivi 3: virhe 32 cm -> 4 cm).
SEURANTA_SILHOUETTE = os.environ.get("SEURANTA_SILHOUETTE", "1") == "1"
SEURANTA_MIN_INSIDE = float(os.environ.get("SEURANTA_MIN_INSIDE", "0.4"))
SEURANTA_SIL_SHIFT_PX = int(os.environ.get("SEURANTA_SIL_SHIFT_PX", "3"))
# Testi_03_04 v4.4: kaukana (Y > SEURANTA_SIL_ALL_Y_CM, heittopaa) siluettitarkennus ajetaan KAIKILLE loydetyille ruuduille (myos tarkka=1): kiekko on siella vain 5-6 px
# leveä ja heittajan tumma vartalo on kiinni kivessa -> LM:n rms/tarkka-lippu ei kerro paikan laadusta, siluetti (oma 3x3-avaus-maski + kaistarajattu ylitysrangaistus)
# tasoittaa rataa (sileys 1.4-1.7 -> 0.5-0.8 cm sivusuunnassa). Tarkassa ruudussa maskiton tulos (inside0 < SEURANTA_MIN_INSIDE) EI hylkaa ruutua vaan jaa LM-paikka.
SEURANTA_SIL_ALL_Y_CM = float(os.environ.get("SEURANTA_SIL_ALL_Y_CM", "2000"))
# Lahella (Y <= SEURANTA_SIL_ALL_Y_CM) siluettiporttia sovelletaan MYOS tarkkoihin ruutuihin (LM:n tarkka-lippu ei suojaa: heitto 181 hyppasi ruuduissa 12705-12708 lakaisijan kateen
# tarkka=1-ruuduilla) ja raja on korkeampi (SEURANTA_MIN_INSIDE_NEAR 0.65: oikean kiven inside0 ~0.8-0.9, lakaisijan 0.54-0.60). Tarkassa ruudussa siluetti EI siirra paikkaa lahella.
# Testi_03_04 v4.5-koe: raja 0.65 lahella TOIMI heitolle 181 (rata pysyi kiinni, 594 riviä), mutta hylkasi liikaa oikeita kivia (y ~ 6-8 m inside0 ~ 0.55-0.60 myos oikealle kivelle;
# rivit 14640 -> 14028, useita heittoja loppuu 1-2 m aiemmin, heitto 127 katoaa y=10.3 m) -> OLETUS 0 = ei porttia tarkoille ruuduille lahella (kuten v4.4). Ks. SEURANTA_KOKEILU.md.
SEURANTA_MIN_INSIDE_NEAR = float(os.environ.get("SEURANTA_MIN_INSIDE_NEAR", "0"))
PROFILE_ALPHA = os.environ.get("PROFILE_ALPHA", "1") == "1"   # 0 = vanha (seurannan oma aariviiva)
ALPHA_LEVEL_DEFAULT = 0.434
ALPHA_RATIO_MIN = 0.2
ALPHA_RATIO_MAX = 0.8
ALPHA_UP = 4
ALPHA_CROP_HALF_PX = 60
ALPHA_A_DILATE_PX = 5
ALPHA_DARK_V_MAX = 60
ALPHA_DARK_RING_MAX_PX = 100
ALPHA_AREA_TOL = 0.30
ALPHA_AREA_WINDOW = 4


def _alpha_pick_component(mask, cx, cy, min_px=60):
    n, lab, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    best = None
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_px:
            continue
        ys, xs = np.where(lab == i)
        d = float(np.hypot(xs.mean() - cx, ys.mean() - cy))
        if best is None or d < best[0]:
            best = (d, i)
    return (lab == best[1]).astype(np.uint8) if best is not None else None


def _alpha_observation_py(frame_u, ref_u, cx, cy):
    """Alfa-kartta + laatumitat yhdelle havainnolle (frame_u = korjattu/vaantamaton ruutu, ref_u = moodikuva).
    Palauttaa dict (alpha, adil, origin, center, ratio, dark_px, a_area) tai None."""
    gain, bias = estimate_photometric_correction(frame_u, ref_u)
    gain = np.asarray(gain, dtype=np.float64)
    bias = np.asarray(bias, dtype=np.float64)
    sup = stone_tracker.suppress_shadow_background(
        frame_u, ref_u, gain, bias, float(GRANITE_DIFF_THRESHOLD), float(SHADOW_V_DROP_MIN),
        float(SHADOW_V_DROP_MAX), int(ICE_S_MAX), int(ICE_V_MIN))
    A = _alpha_pick_component(k9.create_granite_mask(sup), cx, cy)
    if A is None:
        return None
    f = np.clip(frame_u.astype(np.float32) * gain.astype(np.float32)[None, None, :] + bias.astype(np.float32)[None, None, :],
                0, 255).astype(np.uint8)
    H, W = A.shape
    hw = ALPHA_CROP_HALF_PX
    x0, y0 = max(0, cx - hw), max(0, cy - hw)
    x1, y1 = min(W, cx + hw), min(H, cy + hw)
    vf = f.max(axis=2).astype(np.float32)[y0:y1, x0:x1]
    vr = ref_u.max(axis=2).astype(np.float32)[y0:y1, x0:x1]
    Ac = A[y0:y1, x0:x1]
    k = vf / np.maximum(vr, 1.0)
    core = cv2.erode(Ac, np.ones((3, 3), np.uint8))
    if int(core.sum()) < 5:
        core = Ac
    kg = float(np.median(k[core > 0]))
    excl = cv2.dilate(Ac, np.ones((15, 15), np.uint8))
    near = cv2.dilate(Ac, np.ones((45, 45), np.uint8))
    bg_ok = (excl == 0) & (near > 0) & (k > 0.6)
    if int(bg_ok.sum()) < 50:
        return None
    # paikallinen taustataso (jaa tai varjo): normalisoitu konvoluutio kiven ulkopuolelta
    w = ((excl == 0) & (near > 0) & (k > kg + 0.25)).astype(np.float32)
    num = cv2.GaussianBlur(k * w, (0, 0), 5)
    den = cv2.GaussianBlur(w, (0, 0), 5)
    ks = np.where(den > 1e-3, num / np.maximum(den, 1e-3), 1.0).astype(np.float32)
    ks = np.clip(ks, kg + 0.25, 1.05)
    alpha = np.clip((ks - k) / np.maximum(ks - kg, 0.15), 0.0, 1.0).astype(np.float32)
    # laatumitat
    ratio = float(np.median(vf[core > 0]) / max(float(np.median(vf[bg_ok])), 1.0))        # graniitin V / jaan V
    ring = (cv2.dilate(Ac, np.ones((31, 31), np.uint8)) > 0) & (cv2.dilate(Ac, np.ones((9, 9), np.uint8)) == 0)
    dark_px = int(((vf < ALPHA_DARK_V_MAX) & ring).sum())
    adil = cv2.dilate(Ac, np.ones((ALPHA_A_DILATE_PX, ALPHA_A_DILATE_PX), np.uint8))
    return {"alpha": alpha, "adil": adil, "origin": (x0, y0), "center": (cx - x0, cy - y0),
            "ratio": ratio, "dark_px": dark_px, "a_area": int(Ac.sum())}


def _alpha_contour_py(ao, level):
    """Kuminauha-aariviiva (liukuluku, natiivit pikselikoordinaatit) + pinta-ala (px) tasolta level, tai (None, 0)."""
    up = ALPHA_UP
    alpha = ao["alpha"]
    ch, cw = alpha.shape
    au = cv2.resize(alpha, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)
    adu = cv2.resize(ao["adil"], None, fx=up, fy=up, interpolation=cv2.INTER_NEAREST)
    lcx, lcy = ao["center"][0] * up, ao["center"][1] * up
    yy, xx = np.mgrid[0:ch * up, 0:cw * up]
    win = (np.abs(xx - lcx) < 45 * up) & (np.abs(yy - lcy) < 50 * up)
    reg = ((au > level) & win).astype(np.uint8)
    n, lab, stats, cen = cv2.connectedComponentsWithStats(reg, connectivity=8)
    keep = [i for i in range(1, n)
            if stats[i, cv2.CC_STAT_AREA] >= 6 * up * up and float(np.hypot(cen[i][0] - lcx, cen[i][1] - lcy)) < 28 * up]
    if not keep:
        return None, 0.0
    comp = np.isin(lab, keep).astype(np.uint8)
    hm = np.zeros_like(comp)
    cv2.fillConvexPoly(hm, cv2.convexHull(cv2.findNonZero(comp)), 1)
    fin = ((hm > 0) & (adu > 0)).astype(np.uint8)
    cs, _ = cv2.findContours(fin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return None, 0.0
    c = max(cs, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    c = (c + 0.5) / up - 0.5 + np.array(ao["origin"], dtype=np.float64)        # 4x-hila -> natiivit pikselikoordinaatit
    return c, float(fin.sum()) / float(up * up)


ALPHA_CPP = os.environ.get("ALPHA_CPP", "1") == "1" and hasattr(stone_tracker, "alpha_observation_cpp")   # 0 = numpy/cv2-versio


def alpha_observation(frame_u, ref_u, cx, cy):
    """Alfa-kartta + laatumitat yhdelle havainnolle (C++: stone_tracker.alpha_observation_cpp; varapolku numpy/cv2). Palauttaa dict tai None."""
    if not ALPHA_CPP:
        return _alpha_observation_py(frame_u, ref_u, cx, cy)
    gain, bias = estimate_photometric_correction(frame_u, ref_u)
    return stone_tracker.alpha_observation_cpp(
        frame_u, ref_u, np.asarray(gain, dtype=np.float64), np.asarray(bias, dtype=np.float64), int(cx), int(cy),
        float(GRANITE_DIFF_THRESHOLD), float(SHADOW_V_DROP_MIN), float(SHADOW_V_DROP_MAX), int(ICE_S_MAX), int(ICE_V_MIN),
        int(ALPHA_CROP_HALF_PX), int(ALPHA_A_DILATE_PX), int(ALPHA_DARK_V_MAX))


def alpha_contour(ao, level):
    """Kuminauha-aariviiva (liukuluku, natiivit pikselikoordinaatit) + pinta-ala (px) tasolta level, tai (None, 0)."""
    if not ALPHA_CPP:
        return _alpha_contour_py(ao, level)
    c, area = stone_tracker.alpha_contour_cpp(
        ao["alpha"], ao["adil"], float(ao["center"][0]), float(ao["center"][1]),
        int(ao["origin"][0]), int(ao["origin"][1]), float(level), int(ALPHA_UP))
    return c, float(area)


class _ForwardFrameReader:
    """v5.7: ruutu indeksilla. Jos haluttu ruutu on lahella edessapain (<= SEEK_MAX_SKIP ruutua), luetaan eteenpain (grab) - halvempi
    kuin cap.set(CAP_PROP_POS_FRAMES) (~45 ms: haku avainruutuun + dekoodaus eteenpain). Muuten haku kuten ennen. Sama ruutu
    (H.264-dekoodaus on deterministinen) -> sama tulos."""

    def __init__(self, cap):
        self.cap = cap
        self.next_idx = None        # seuraavan read()-kutsun ruutu (None = tuntematon -> haku)

    def read(self, idx):
        idx = int(idx)
        if self.next_idx is None or idx < self.next_idx or idx - self.next_idx > SEEK_MAX_SKIP:
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
    """Lukee havaintojen ruudut (frame_idx) ja liittaa jokaiseen alfa-kartan ('alpha_obs'). Hylkaa: ei graniittia /
    tausta ei mitattavissa, suhde ALPHA_RATIO_MIN..MAX:n ulkopuolella, sääntö 1 (tumma alue kiven ymparilla).
    Palauttaa (havainnot, tilasto-dict)."""
    cap = live_source.open_capture(video_path)      # Testi_06_01: live-tilassa puskurista
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dist_coeffs = np.array([calib["best_k1"], 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    map1, map2 = k94._build_undistort_maps(calib["camera_matrix"], dist_coeffs, (w, h))
    ref_u = calib["frame_undistorted"]
    out = []
    st_ = {"in": len(observations), "ei_kuvaa": 0, "ei_graniittia": 0, "suhde": 0, "tumma": 0, "ala": 0}
    reader = _ForwardFrameReader(cap)
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
        if not (ALPHA_RATIO_MIN <= ao["ratio"] <= ALPHA_RATIO_MAX):
            st_["suhde"] += 1
            continue
        if ao["dark_px"] > ALPHA_DARK_RING_MAX_PX:
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
    return float(np.median(r)) if r else ALPHA_LEVEL_DEFAULT


def apply_area_rule(observations, level):
    """Saanto 2: hylkaa havainnot joiden alfa-aariviivan pinta-ala poikkeaa > ALPHA_AREA_TOL lahiruutujen liukuvasta mediaanista.
    Palauttaa (havainnot, hylattyjen maara)."""
    obs = sorted(observations, key=lambda o: o["frame_idx"])
    areas = np.array([alpha_contour(o["alpha_obs"], level)[1] for o in obs], dtype=np.float64)
    keep = []
    for i, o in enumerate(obs):
        lo, hi = max(0, i - ALPHA_AREA_WINDOW), min(len(obs), i + ALPHA_AREA_WINDOW + 1)
        nb = areas[lo:hi]
        nb = nb[nb > 0]
        if areas[i] <= 0 or len(nb) == 0:
            continue
        med = float(np.median(nb))
        if abs(areas[i] - med) <= ALPHA_AREA_TOL * med:
            keep.append(o)
    return keep, len(obs) - len(keep)


def alpha_contour_observations(observations, level):
    """Kopiot havainnoista joiden 'contour' on alfa-aariviiva tasolta level (alkuperainen seurannan aariviiva ei muutu)."""
    res = []
    for o in observations:
        c, _ = alpha_contour(o["alpha_obs"], level)
        if c is None or len(c) < 8:
            continue
        o2 = dict(o)
        o2["contour"] = c
        res.append(o2)
    return res


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

    profile = _fit_stone_profile(pose, stones)

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

    cap = live_source.open_capture(video_file)      # Testi_06_01: live-tilassa puskurista
    reader = _ForwardFrameReader(cap)

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

        ok, frame = reader.read(obs["frame_idx"])

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


# ============================================================
# SEURANNAN HAKUVAIHEEN VALINTA (Testi_02_03)
#
# Ympäristömuuttujilla ohjattava. OLETUS TRACKER_MODE=meanshift (hybridi);
# TRACKER_MODE=grid palauttaa TÄSMÄLLEEN Testi_02_02:n alkuperäisen käytöksen:
#   TRACKER_MODE=grid            alkuperäinen laajeneva ristikkohaku
#   TRACKER_MODE=meanshift       mean-shift + ristikkohaku varana (hybridi)
#   TRACKER_MODE=meanshift_pure  pelkkä mean-shift (ei varahakua)
#   TRACKER_MODE=ensemble        yhdistelmähaku: ristikkohaku (ennustetusta keskipisteestä) + mean-shift
#                                (viimeisestä ja ennustetusta paikasta); valinta = pistemäärä - rangaistus
#                                taaksepäin liikkumisesta (ENS_BACK_PEN) ja ennusteesta poikkeamisesta
#                                (ENS_PRED_PEN). Käyttää kivikohtaista liike-ennustetta (PRED_LOOKBACK).
#   TRACKER_MODE=compare         ajaa SEKÄ grid- ETTÄ mean-shift-haun samoilla
#                                syötteillä joka framella ja kirjaa vertailun
#                                CSV:hen (TRACKER_COMPARE_LOG). Tilaa ohjaa
#                                COMPARE_DRIVER (grid|ms, oletus grid).
#   TRACKER_GOLD=1               compare-tilassa lisäksi TYHJENTÄVÄ ristikkohaku
#                                (kulta-standardi, hullOverlapScore-optimi).
#   MS_VARIANT=meanshift|meanshift_pure   (compare-tilassa vertailtava variantti)
#   MS_GAIN, MS_MAX_ITER, MS_TOL_PX, MS_INNER_WEIGHT, MS_MARGIN, MS_TAU, MS_POLISH
#                          virityskertoimet (oletukset = offline-viritetyt: 0.8/6/0.3/2.0/1.10/0.5/4.0)
#   MS_GAINS=0.8,1.0,1.2   compare-tilassa ajetaan lisäksi nämä gain-arvot
#                          (sarakkeet ms<gain>_*) - painokertoimen optimointiin.
# ============================================================

TRACKER_MODE = os.environ.get("TRACKER_MODE", "ensemble")
TRACKER_COMPARE_LOG = os.environ.get("TRACKER_COMPARE_LOG")
TRACKER_GOLD = os.environ.get("TRACKER_GOLD", "0") == "1"
COMPARE_DRIVER = os.environ.get("COMPARE_DRIVER", "grid")
MS_VARIANT = os.environ.get("MS_VARIANT", "meanshift")
MS_GAIN = float(os.environ.get("MS_GAIN", "0.8"))
MS_MAX_ITER = int(os.environ.get("MS_MAX_ITER", "6"))
MS_TOL_PX = float(os.environ.get("MS_TOL_PX", "0.30"))
MS_TAU = float(os.environ.get("MS_TAU", "0.5"))
MS_POLISH_CM = float(os.environ.get("MS_POLISH", "4.0"))
MS_INNER_WEIGHT = float(os.environ.get("MS_INNER_WEIGHT", "2.0"))
MS_MARGIN = float(os.environ.get("MS_MARGIN", "1.10"))
ENS_BACK_TOL_CM = float(os.environ.get("ENS_BACK_TOL_CM", "2.0"))
ENS_BACK_PEN = float(os.environ.get("ENS_BACK_PEN", "0.15"))
ENS_PRED_PEN = float(os.environ.get("ENS_PRED_PEN", "0.02"))
PRED_LOOKBACK = int(os.environ.get("PRED_LOOKBACK", "12"))
PRED_MIN_SPAN = int(os.environ.get("PRED_MIN_SPAN", "4"))
GOLD_COARSE_STEP_CM = 1.0
GOLD_FINE_STEP_CM = 0.25
GOLD_MAX_HALF_RANGE_CM = 30.0

_TRACKER_MODE_ID = {"grid": 0, "meanshift": 1, "meanshift_pure": 2, "gold": 3, "ensemble": 5}
_compare_state = {"file": None, "writer": None, "rows": 0, "parity": 0}
_COMPARE_FIELDS = ["frame", "stone_id", "X0", "Y0", "hx", "hy"]
MS_GAINS = [float(x) for x in os.environ.get("MS_GAINS", "").split(",") if x.strip()]
_COMPARE_METHODS = ["grid", "ms", "gold"] + [f"ms{g:g}" for g in MS_GAINS]
_COMPARE_PER = ["wall_ms", "found", "X", "Y", "rms_px", "tarkka", "score",
                "loc_X", "loc_Y", "prep_ms", "locate_ms", "refine_ms", "iters",
                "converged", "fallback"]
for _m in _COMPARE_METHODS:
    _COMPARE_FIELDS += [f"{_m}_{k}" for k in _COMPARE_PER]


def _compare_log_row(frame_index, stone_id, X0, Y0, hx, hy, per_method):
    if TRACKER_COMPARE_LOG is None:
        return
    if _compare_state["file"] is None:
        _compare_state["file"] = open(TRACKER_COMPARE_LOG, "w", newline="")
        _compare_state["writer"] = csv.DictWriter(
            _compare_state["file"], fieldnames=_COMPARE_FIELDS
        )
        _compare_state["writer"].writeheader()
    row = {"frame": frame_index, "stone_id": stone_id,
           "X0": X0, "Y0": Y0, "hx": hx, "hy": hy}
    for name, (r, wall_ms) in per_method.items():
        row[f"{name}_wall_ms"] = wall_ms
        row[f"{name}_found"] = int(bool(r.get("found")))
        row[f"{name}_X"] = r.get("X_cm", "")
        row[f"{name}_Y"] = r.get("Y_cm", "")
        row[f"{name}_rms_px"] = r.get("rms_px", "")
        row[f"{name}_tarkka"] = int(bool(r.get("tarkka", False)))
        row[f"{name}_score"] = r.get("score", "")
        row[f"{name}_loc_X"] = r.get("loc_X", "")
        row[f"{name}_loc_Y"] = r.get("loc_Y", "")
        row[f"{name}_prep_ms"] = r.get("prep_ms", "")
        row[f"{name}_locate_ms"] = r.get("locate_ms", "")
        row[f"{name}_refine_ms"] = r.get("refine_ms", "")
        row[f"{name}_iters"] = r.get("iters", "")
        row[f"{name}_converged"] = int(bool(r.get("converged", False)))
        row[f"{name}_fallback"] = int(bool(r.get("used_fallback", False)))
    _compare_state["writer"].writerow(row)
    _compare_state["rows"] += 1
    if _compare_state["rows"] % 200 == 0:
        _compare_state["file"].flush()


def _seuranta_dispatch(frame_index, stone_ids, base_args, X0_arr, Y0_arr,
                       half_x, half_y, pred=None):
    """Kutsuu stone_tracker.track_stones_batch:ia valitulla hakutavalla
    (TRACKER_MODE). base_args = kaikki positioargumentit diff_threshold:iin
    asti (X0/Y0/half_range-taulukot mukana indekseissä 2-5)."""

    def call(mode_name, gold=False, gain=None, pred=None):
        t0 = time.perf_counter()
        kwargs = dict(
            locate_mode=_TRACKER_MODE_ID[mode_name],
            ms_gain=MS_GAIN if gain is None else gain, ms_max_iter=MS_MAX_ITER, ms_tol_px=MS_TOL_PX,
            ms_inner_weight=MS_INNER_WEIGHT, ms_margin_scale=MS_MARGIN,
            ms_tau=MS_TAU, ms_polish_step_cm=MS_POLISH_CM,
            ens_back_tol_cm=ENS_BACK_TOL_CM, ens_back_pen=ENS_BACK_PEN, ens_pred_pen=ENS_PRED_PEN,
        )
        if pred is not None and mode_name in ("ensemble",):
            kwargs["pred_dx"], kwargs["pred_dy"] = pred
        args = list(base_args)
        if gold:
            # coarse/fine-askeleet kulta-standardille (indeksit 11, 12)
            args[11] = GOLD_COARSE_STEP_CM
            args[12] = GOLD_FINE_STEP_CM
        res = stone_tracker.track_stones_batch(*args, **kwargs)
        return res, (time.perf_counter() - t0) * 1000.0

    # Virityskehyksen tallennus (TRACKER_DUMP_DIR + TRACKER_DUMP_RANGE=alku:loppu):
    # tallentaa seurantaframen + syotteet .npz:ksi offline-kokeiluja varten.
    if os.environ.get("TRACKER_DUMP_DIR"):
        lo, hi = [int(x) for x in os.environ.get("TRACKER_DUMP_RANGE", "0:0").split(":")]
        if lo <= frame_index <= hi:
            np.savez_compressed(
                os.path.join(os.environ["TRACKER_DUMP_DIR"], f"f{frame_index:06d}.npz"),
                frame=base_args[0], X0=np.asarray(X0_arr), Y0=np.asarray(Y0_arr),
                hx=np.asarray(half_x), hy=np.asarray(half_y),
                stone_ids=np.asarray(stone_ids), diff_threshold=base_args[19]
            )
        elif frame_index > hi:
            if _compare_state["file"] is not None:
                _compare_state["file"].flush()
            os._exit(0)

    if TRACKER_MODE in ("grid", "meanshift", "meanshift_pure", "ensemble"):
        res, _ = call(TRACKER_MODE, pred=pred)
        # grid-tulokset talteen dumpin viereen (vertailukohta offline-kokeille)
        if os.environ.get("TRACKER_DUMP_DIR"):
            lo, hi = [int(x) for x in os.environ.get("TRACKER_DUMP_RANGE", "0:0").split(":")]
            if lo <= frame_index <= hi:
                import pickle as _pk
                _pk.dump(res, open(os.path.join(os.environ["TRACKER_DUMP_DIR"], f"r{frame_index:06d}.pkl"), "wb"))
        return res

    # compare
    order = ["grid", MS_VARIANT]
    if _compare_state["parity"] % 2:
        order.reverse()
    _compare_state["parity"] += 1
    out = {}
    for name in order:
        out[name] = call(name)
    res_grid, wall_grid = out["grid"]
    res_ms, wall_ms = out[MS_VARIANT]

    res_gold = None
    if TRACKER_GOLD and max(float(np.max(half_x)), float(np.max(half_y))) <= GOLD_MAX_HALF_RANGE_CM:
        res_gold = call("gold", gold=True)

    res_gain = {g: call(MS_VARIANT, gain=g) for g in MS_GAINS}

    for i, sid in enumerate(stone_ids):
        per = {"grid": (res_grid[i], wall_grid), "ms": (res_ms[i], wall_ms)}
        if res_gold is not None:
            per["gold"] = (res_gold[0][i], res_gold[1])
        for g, (rg, wg) in res_gain.items():
            per[f"ms{g:g}"] = (rg[i], wg)
        _compare_log_row(frame_index, sid, float(X0_arr[i]), float(Y0_arr[i]),
                         float(half_x[i]), float(half_y[i]), per)

    return res_ms if COMPARE_DRIVER == "ms" else res_grid



# HAKU-diagnostiikka (Testi_02_03): HAKU_LOG=polku.csv kirjaa jokaisen HAKU-kutsun
# tuloksen ja sen, esikoko dedup/enimmaismaara tehtavaa uuden kiven rekisteroinnin.
HAKU_LOG = os.environ.get("HAKU_LOG")
_haku_log_state = {"file": None, "writer": None}


def _haku_log(frame_index, n_active, called, found, bx, by, score, registered,
              blocker_id, blocker_dist, blocker_confirmed):
    if not HAKU_LOG:
        return
    if _haku_log_state["file"] is None:
        _haku_log_state["file"] = open(HAKU_LOG, "w", newline="")
        _haku_log_state["writer"] = csv.writer(_haku_log_state["file"])
        _haku_log_state["writer"].writerow(
            ["frame", "n_active", "called", "found", "X", "Y", "score",
             "registered", "blocker_id", "blocker_dist_cm", "blocker_confirmed"])
    _haku_log_state["writer"].writerow(
        [frame_index, n_active, int(called), int(found),
         "" if bx is None else f"{bx:.1f}", "" if by is None else f"{by:.1f}",
         "" if score is None else f"{score:.3f}", int(registered),
         "" if blocker_id is None else blocker_id,
         "" if blocker_dist is None else f"{blocker_dist:.1f}",
         "" if blocker_confirmed is None else int(blocker_confirmed)])
    if frame_index % 100 == 0:
        _haku_log_state["file"].flush()



# Koko framejen tallennus simulaattoria varten (Testi_02_03): TRACKER_FRAME_DUMP_DIR +
# TRACKER_FRAME_DUMP_RANGES="a:b,c:d" tallentaa JOKAISEN framen (frame_u_for_tracking)
# valeilta g<frame>.npz ja lopettaa ajon viimeisen valin jalkeen.
_FRAME_DUMP_DIR = os.environ.get("TRACKER_FRAME_DUMP_DIR")
_FRAME_DUMP_RANGES = [
    tuple(int(v) for v in part.split(":"))
    for part in os.environ.get("TRACKER_FRAME_DUMP_RANGES", "").split(",") if part.strip()
]


def _frame_dump_hook(frame_index, frame_for_tracking, frame_raw=None):
    if not _FRAME_DUMP_DIR or not _FRAME_DUMP_RANGES:
        return
    if any(lo <= frame_index <= hi for lo, hi in _FRAME_DUMP_RANGES):
        extra = {}
        # Testi_03_04: TRACKER_FRAME_DUMP_RAW=1 tallentaa myos vaimentamattoman (stabiloidun+oikaistun) ruudun taustanvaimennuskokeisiin
        if frame_raw is not None and os.environ.get("TRACKER_FRAME_DUMP_RAW") == "1":
            extra["raw"] = frame_raw
        np.savez_compressed(
            os.path.join(_FRAME_DUMP_DIR, f"g{frame_index:06d}.npz"), frame=frame_for_tracking, **extra
        )
    elif frame_index > max(hi for _, hi in _FRAME_DUMP_RANGES):
        os._exit(0)



def _stone_prediction(seuranta_stones, frame_index, half_x, half_y):
    """Kivikohtainen liike-ennuste (cm) edellisesta paikasta: vakionopeus viimeisista havainnoista
    (position_history, PRED_LOOKBACK framea, vahintaan PRED_MIN_SPAN framea), skaalattuna
    puuttuneilla frameilla (misses+1) ja rajattuna hakualueen sisaan."""
    pdx, pdy = [], []
    for i, s in enumerate(seuranta_stones):
        hist = [h for h in s["position_history"] if frame_index - PRED_LOOKBACK <= h[0] <= frame_index - 1 - s["misses"] - s.get("gap_extra", 0)]
        dx = dy = 0.0
        if len(hist) >= 2:
            (f0, x0, y0), (f1, x1, y1) = hist[0], hist[-1]
            if f1 - f0 >= PRED_MIN_SPAN:
                n = s["misses"] + 1 + s.get("gap_extra", 0)
                dx = (x1 - x0) / (f1 - f0) * n
                dy = (y1 - y0) / (f1 - f0) * n
                dx = max(-float(half_x[i]), min(float(half_x[i]), dx))
                dy = max(-float(half_y[i]), min(float(half_y[i]), dy))
        pdx.append(dx)
        pdy.append(dy)
    return np.array(pdx, dtype=np.float64), np.array(pdy, dtype=np.float64)


# ------------------------------------------------------------------
# AIKAMITTAUS (Testi_03_01): _PROF["nimi"] = [sekunnit, kutsut]. Kaytetaan
# with-lohkona: `with _prof("x"): ...`. MAX_FRAME (ymparistomuuttuja)
# lopettaa ajon N ruudun jalkeen, jotta mittaus kattaa vain muutaman kiven.
# ------------------------------------------------------------------
import contextlib as _ctxlib
_PROF = {}
MAX_FRAME = int(os.environ.get("MAX_FRAME", "0") or 0)


@_ctxlib.contextmanager
def _prof(name):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        e = _PROF.setdefault(name, [0.0, 0])
        e[0] += time.perf_counter() - t0
        e[1] += 1



# ============================================================
# PULLONKAULA-ANALYYSI (Testi_05_02): kuka oikeasti rajoittaa nopeutta? Liukuhihnan vaiheet A (luku+stabilointi), B (warp+varjosuodatus) ja C (paasaie: HAKU/SEURANTA/CSV/debug)
# ajavat rinnan; nopeuden maaraa HITAIN vaihe (suurin "palveluaika" ms/ruutu = pienin kapasiteetti r/s). Muut odottavat joko syotetta (tyhja jono) tai tulostetta (jono taynna).
# CPU < 100 % kun pullonkaulavaihe on yksi saie. Raportti tulostuu nopeusraportin alkuun ja tiiviina edistymisrivin yhteydessa. PULLONKAULA_RAPORTTI=0 = pois.
# ============================================================
PULLONKAULA_RAPORTTI = os.environ.get("PULLONKAULA_RAPORTTI", "1") == "1"
_PIPE_STATS = {"wall0": None, "cpu0": None, "qC_sum": 0, "qC_n": 0, "qA_sum": 0, "qA_n": 0, "cpu_a": None, "cpu_b": None, "depth": 3}


def _stage_numbers():
    """Palauttaa dict: vaiheiden palveluaika / odotukset (ms per ruutu) _PROF:n pohjalta, tai None jos hihnaa ei ole ajettu."""
    P = lambda k: _PROF.get(k, [0.0, 0])
    a_n = P("pipe A: stabilointi (vaihekorrelaatio)")[1]
    b_n = P("pipe B: odottaa vaihetta A")[1]
    c_n = P("pipe C: paasaie odottaa hihnaa (sisaltyy py: read(video)-riviin)")[1]
    if a_n < 20 or b_n < 20 or c_n < 20:
        return None
    a_busy = sum(v[0] for k, v in _PROF.items() if k.startswith("pipe A:") and "odottaa" not in k) / a_n * 1000
    a_wout = P("pipe A: odottaa vaihetta B (jono taynna)")[0] / a_n * 1000
    b_busy = sum(v[0] for k, v in _PROF.items() if k.startswith("pipe B:") and "odottaa" not in k) / b_n * 1000
    b_win = P("pipe B: odottaa vaihetta A")[0] / b_n * 1000
    b_wout = P("pipe B: odottaa paasaiketta (jono taynna)")[0] / b_n * 1000
    c_win = P("pipe C: paasaie odottaa hihnaa (sisaltyy py: read(video)-riviin)")[0] / c_n * 1000
    c_tot = P("FRAME_KOKO")[0] / max(P("FRAME_KOKO")[1], 1) * 1000
    c_busy = max(c_tot - c_win, 1e-6)
    return dict(a=(a_busy, 0.0, a_wout), b=(b_busy, b_win, b_wout), c=(c_busy, c_win, 0.0), c_tot=c_tot)


def _bottleneck_compact():
    s = _stage_numbers()
    if s is None:
        return None
    names = {"a": "A luku+stabilointi", "b": "B warp+varjosuodatus", "c": "C paasaie"}
    worst = max(("a", "b", "c"), key=lambda k: s[k][0])
    q = ""
    if _PIPE_STATS["qC_n"]:
        q = f" | jonot A->B {_PIPE_STATS['qA_sum'] / max(_PIPE_STATS['qA_n'], 1):.1f}/{_PIPE_STATS['depth']} B->C {_PIPE_STATS['qC_sum'] / _PIPE_STATS['qC_n']:.1f}/{_PIPE_STATS['depth']}"
    return (f"  pullonkaula: {names[worst]} ({s[worst][0]:.0f} ms/ruutu = max {1000 / s[worst][0]:.1f} r/s) | "
            f"A {s['a'][0]:.0f} ms, B {s['b'][0]:.0f} ms, C {s['c'][0]:.0f} ms{q}")


def _bottleneck_report_lines(n_frames):
    s = _stage_numbers()
    if s is None:
        return []
    L = ["=== PULLONKAULA-ANALYYSI (kuka oikeasti rajoittaa nopeutta) ==="]
    names = {"a": "A  luku + gray + stabilointi (vaihekorrelaatio)", "b": "B  warpAffine+remap + valotasapaino + varjosuodatus", "c": "C  paasaie: HAKU/SEURANTA/CSV/debug-video"}
    L.append(f"{'vaihe':52s} {'palvelu':>8s} {'kapas.':>7s} {'odottaa':>8s} {'odottaa':>8s} {'kuorm.':>7s}")
    L.append(f"{'':52s} {'ms/rt':>8s} {'r/s':>7s} {'syotetta':>8s} {'tulosta':>8s} {'%':>7s}")
    for k in ("a", "b", "c"):
        busy, win, wout = s[k]
        L.append(f"{names[k]:52s} {busy:8.1f} {1000 / busy:7.1f} {win:8.1f} {wout:8.1f} {100 * busy / (busy + win + wout):7.0f}")
    worst = max(("a", "b", "c"), key=lambda k: s[k][0])
    L.append(f"=> PULLONKAULA: {names[worst].split('  ', 1)[1].strip()} - kapasiteetti {1000 / s[worst][0]:.1f} r/s (mitattu kokonaisnopeus {1000 / s['c_tot']:.1f} r/s). "
             "Muut vaiheet odottavat tahan (jono taynna) eivatka ole rajoittavia.")
    if _PIPE_STATS["qC_n"]:
        L.append(f"Jonojen keskitaytto: A->B {_PIPE_STATS['qA_sum'] / max(_PIPE_STATS['qA_n'], 1):.2f}/{_PIPE_STATS['depth']}, B->C {_PIPE_STATS['qC_sum'] / _PIPE_STATS['qC_n']:.2f}/{_PIPE_STATS['depth']} "
                 "(B->C taynna = paasaie ei ehdi ottaa; tyhja = ylavirran vaihe on hitain)")
    # CPU
    try:
        wall = time.perf_counter() - _PIPE_STATS["wall0"]; cpu = time.process_time() - _PIPE_STATS["cpu0"]
        th = []
        if _PIPE_STATS["cpu_a"] is not None:
            th.append(f"A {100 * _PIPE_STATS['cpu_a'] / wall:.0f}%")
        if _PIPE_STATS["cpu_b"] is not None:
            th.append(f"B {100 * _PIPE_STATS['cpu_b'] / wall:.0f}%")
        c_cpu = time.thread_time()
        th.append(f"paasaie {100 * c_cpu / max(wall, 1e-9):.0f}%")
        L.append(f"CPU: prosessi {cpu / wall:.2f} ydinta keskimaarin ({os.cpu_count()} loogista ydinta); saikeiden CPU/seinakello: " + ", ".join(th) +
                 " (HUOM: paasaie odottaa C++-kutsun (SEURANTA/HAKU) aikana, jolloin sen oma CPU% on matala vaikka se on pullonkaula - C++ tyo kuluu C++:n tyosaikeissa; luotettavin mittari on yo. palveluaika + jonojen taytto)")
    except Exception:
        pass
    # paasaikeen sisainen jako
    items = [(k, v[0] / max(_PROF.get("FRAME_KOKO", [0, 1])[1], 1) * 1000) for k, v in _PROF.items()
             if k.startswith("py:") and "taustasaikeen oma kesto" not in k and "MUU / JAANNOS" not in k and "tulossilmukka" not in k]
    items.sort(key=lambda kv: -kv[1])
    if worst == "c" and items:
        L.append("Paasaikeen suurimmat osat (ms/ruutu, % palveluajasta):")
        for k, ms in items[:6]:
            L.append(f"   {k:62s} {ms:7.2f} {100 * ms / s['c'][0]:5.1f}%")
        top = items[0][0]
        if "debug-video" in top:
            L.append("Vihje: debug-video on suurin osa - aja ilman --debug (tai nopeuta piirtoa).")
        elif "SEURANTA dispatch" in top:
            L.append("Vihje: SEURANTA (C++ track_stones_batch) on peräkkäinen kutsu joka ruudulle; sisainen rinnakkaisuus rajoittuu kivien maaraan/kutsu. "
                     "Nopeutus: vahenna kivi-ehdokkaita, pienenna hakualueita, tai porrasta HAKU/SEURANTA useammalle saikeelle.")
        elif "HAKU odotus" in top:
            L.append("Vihje: HAKU-taustasaie ei ehdi valmistua ruudun aikana - harvenna HAKU-vali tai nopeuta HAKU:a.")
    elif worst == "a":
        L.append("Vihje: stabilointi (vaihekorrelaatio) on hitain - laske pienemmalla resoluutiolla/harvemmin tai jaa kahteen saikeeseen.")
    elif worst == "b":
        L.append("Vihje: warp+remap / varjosuodatus on hitain - yhdista yhteen lapikaynti tai laske pienemmalla alueella.")
    L.append("================================================")
    return L


def _print_prof_report(n_frames, n_seuranta_updates):
    print()
    if PULLONKAULA_RAPORTTI:
        for _l in _bottleneck_report_lines(n_frames):
            print(_l)
        print()
    print("=== VAIHEKOHTAINEN AIKAMITTAUS (Testi_03_01) ===")
    print(f"ruutuja: {n_frames}, kivipaivityksia: {n_seuranta_updates}")
    _serial = sum(
        sec for k, (sec, n) in _PROF.items()
        if k.startswith("py:") and "taustasaikeen oma kesto" not in k and "varidiagnostiikka" not in k
    )
    _tot = _PROF.get("FRAME_KOKO", [0.0, 1])[0]
    _PROF["py: MUU / JAANNOS (ei mitattu: FRAME_KOKO - mitatut sarjavaiheet)"] = [max(0.0, _tot - _serial), n_frames]
    print("--- Python-puoli (ms/ruutu, kutsuja) ---")
    tot = _PROF.get("FRAME_KOKO", [0.0, 1])[0]
    for k, (sec, n) in sorted(_PROF.items(), key=lambda kv: -kv[1][0]):
        print(f"{k:58s} {sec / n_frames * 1000:8.2f} ms/ruutu {100 * sec / max(tot, 1e-9):5.1f}%  n={n}")
    try:
        import stone_tracker as _st
        snap = _st.prof_snapshot()
    except Exception:
        snap = []
    try:
        if GPU_GRID and hasattr(_st, "gpu_grid_stats"):
            _g = _st.gpu_grid_stats()
            print(f"GPU-ristikkohaku: kutsuja {_g['calls']}, GPU-polku kaytossa {_g['used']}, CPU-varapolku {_g['fallbacks']}, ehdokkaita tarkalla CPU:lla (leikkautuva/ei-kupera) {_g['cand_exact_cpu']}"
                  + (f" | verify: {_g['verified']} ristikkoa ({_g['cand_compared']} ehdokasta), eri pistemaara {_g['score_diffs']}, eri voittaja {_g['mismatches']}, suurin pistemaaraero {_g['err_max']:.2e}" if GPU_GRID_VERIFY else ""))
    except Exception as _e:
        print("GPU-tilasto ei saatavilla:", _e)
    if snap:
        print("--- C++-puoli (summattu CPU-aika saikeiden yli) ---")
        print(f"{'vaihe':58s} {'ms/ruutu':>9s} {'ms/kutsu':>9s} {'kutsuja':>8s}")
        for name, ms, n in snap:
            if n == 0:
                continue
            if "ulkoiteraatioita" in name:
                print(f"{name:58s} {'':>9s} {'':>9s} {n:8d}   (ulkoiteraatioita / LM-tarkennus)")
                continue
            print(f"{name:58s} {ms / n_frames:9.2f} {ms / n:9.3f} {n:8d}")
    print("Versio: " + _version_string())
    print("================================================")


# ------------------------------------------------------------------
# VERSIO (Testi_03_01): nakyy nopeusraporttien lopussa. Nosta SOFTWARE_VERSION jokaisen julkaistavan
# muutoksen yhteydessa; git-tiivisteen (jos kansio on git-repo) ja C++-moduulien kaannosajan avulla
# nakee myos onko .so kaannetty uudelleen (vanha .so + uusi main.py on tyypillinen sekaannus).
# ------------------------------------------------------------------
SOFTWARE_VERSION = "Testi_06_01 v6.8 (live-kamera + puskuri; kalibrointi: hoglinet +-20 cm symmetrisesti T-viivoista, peili- ja k1-varmistus; pohja Testi_05_03 v5.13: oletukset: GPU_B=1, STAB_WORKERS=2, PIPE_DEPTH=6, INTRA_PARALLEL=1, DEBUG_YUV=1, GRID_THREADS=2, PREP_PARALLEL=1, HAKU_AHEAD=1, SIL_IN_BATCH=1, CV_SINGLE_PERSIST=1, SAT_PARALLEL=0; seuranta identtinen v5.6:n kanssa) (2026-10-02)"


def _version_string():
    parts = [SOFTWARE_VERSION]
    try:
        import subprocess
        here = os.path.dirname(os.path.abspath(__file__))
        h = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=here,
            capture_output=True, text=True, timeout=3
        )
        if h.returncode == 0 and h.stdout.strip():
            d = subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no", "."], cwd=here,
                capture_output=True, text=True, timeout=3
            )
            parts.append("git " + h.stdout.strip() + ("+muokattu" if d.stdout.strip() else ""))
    except Exception:
        pass
    for mod in (stone_tracker, mode_engine):
        try:
            parts.append(mod.build_info())
        except Exception:
            parts.append(getattr(mod, "__name__", "?") + " (ei build_info: vanha .so - kaanna uudelleen)")
    return " | ".join(parts)


def _estimate_photo_timed(frame_bgr, ref_bgr):
    t0 = time.perf_counter()
    res = estimate_photometric_correction(frame_bgr, ref_bgr)
    _e = _PROF.setdefault("bg: valotasapaino taustasaikeessa (rinnan, ei lisaa)", [0.0, 0])
    _e[0] += time.perf_counter() - t0; _e[1] += 1
    return res


def _pick_eviction_victim(active_stones):
    """Palauttaa poistettavan radan (huonoin) tai None. Suojaa hyvin sovittuja vahvistettuja ratoja."""
    def med_rms(st):
        h = list(st.get("rms_hist", []))
        return float(np.median(h)) if h else 1e9
    unconfirmed = [st for st in active_stones if not st["confirmed"]]
    if unconfirmed:
        # huonoin sovitus (tai ei sovitusta) ensin; tasatilanteessa vanhin (pienin id)
        return max(unconfirmed, key=lambda st: (med_rms(st), -st["stone_id"]))
    bad = [st for st in active_stones if med_rms(st) >= EVICT_MIN_RMS]
    if bad:
        return max(bad, key=lambda st: med_rms(st))
    return None


class _LivePrep:
    """Elavan vaiheen KOKO RUUDUN valmistelu (Testi_03_01): warpAffine+remap, valotasapaino ja
    varjonsietoinen taustanvaimennus -> (frame_u, frame_u_for_tracking, haku_seuranta_diff_threshold).
    Sama logiikka kuin aiemmin suoraan pääsilmukassa; nyt omana luokkanaan jotta sama koodi voidaan
    ajaa joko pääsäikeessä (ensimmäinen elävä ruutu) tai liukuhihnan tuottajasäikeessä.
    prefix: aikamittausavaimen etuliite ("py:" = pääsäie, "pipe:" = tuottajasäie)."""

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

    def process(self, frame, stabilization_matrix, frame_index, live_state, calib_result):
        pf = self.prefix
        ref_undist_live = calib_result["calib"]["frame_undistorted"]
        t_warp0 = time.perf_counter()

        # GPU-vaihe B (valinnainen): alustus kerran, virheessa pysyva paluu CPU-polkuun
        if GPU_B and getattr(self, "_gpu_b", None) is None:
            self._gpu_b = False
            try:
                if (ENABLE_SHADOW_TOLERANT_STABILIZATION and hasattr(stone_tracker, "gpu_b_init") and ref_undist_live is not None
                        and ref_undist_live.shape == frame.shape):
                    _info = stone_tracker.gpu_b_init(live_state["map1"], live_state["map2"], ref_undist_live, float(GRANITE_DIFF_THRESHOLD),
                                                     float(SHADOW_V_DROP_MIN), float(SHADOW_V_DROP_MAX), int(ICE_S_MAX), int(ICE_V_MIN))
                    self._gpu_b = not _info.startswith("EI KAYTETTAVISSA")
                    print(f"GPU-vaihe B: {_info}")
                else:
                    print("GPU-vaihe B: EI KAYTETTAVISSA (varjotoleranssi pois / vanha moduuli / kokoero)")
            except Exception as _e:
                print(f"GPU-vaihe B: EI KAYTETTAVISSA ({_e!r})")

        frame_u = None
        if getattr(self, "_gpu_b", False):
            frame_u = stone_tracker.gpu_b_warp(frame, np.ascontiguousarray(stabilization_matrix, dtype=np.float64))
            if frame_u is None:
                print("GPU-vaihe B: virhe -> palataan CPU-polkuun")
                self._gpu_b = False
        if frame_u is None:
            stabilized = cv2.warpAffine(
                frame, stabilization_matrix, (self.width, self.height)
            )
            frame_u = cv2.remap(
                stabilized, live_state["map1"], live_state["map2"],
                interpolation=cv2.INTER_LINEAR
            )
        _e = _PROF.setdefault(pf + " warpAffine+remap (koko frame)", [0.0, 0]); _e[0] += time.perf_counter() - t_warp0; _e[1] += 1

        # VALOTASAPAINO: estimointi taustasaikeessa kerran sekunnissa. Uusi gain/bias otetaan kayttoon
        # AINA tasan PHOTO_APPLY_DELAY_FRAMES ruutua laheteyksen jalkeen (tarvittaessa odotetaan tulosta),
        # jolloin ajo on TOISTETTAVA - ei riipu siita kuinka nopeasti taustasaie ehti valmistua.
        if self.photo_future is not None and frame_index >= self.photo_apply_at:
            self.photo_gain, self.photo_bias = self.photo_future.result()
            self.photo_future = None

        if self.photo_gain is None:
            t_photo0 = time.perf_counter()
            self.photo_gain, self.photo_bias = estimate_photometric_correction(
                frame_u, ref_undist_live
            )
            _e = _PROF.setdefault(pf + " valotasapaino (alkuestimaatti, synkroninen)", [0.0, 0]); _e[0] += time.perf_counter() - t_photo0; _e[1] += 1
            self.next_photo_update_frame = frame_index + max(1, int(round(self.fps)))
        elif self.photo_future is None and frame_index >= self.next_photo_update_frame:
            self.photo_future = self.photo_executor.submit(
                _estimate_photo_timed, frame_u.copy(), ref_undist_live
            )
            self.photo_apply_at = frame_index + PHOTO_APPLY_DELAY_FRAMES
            self.next_photo_update_frame = frame_index + max(1, int(round(self.fps)))

        _fast_shadow = (
            ENABLE_SHADOW_TOLERANT_STABILIZATION
            and hasattr(stone_tracker, "suppress_shadow_background")
            and ref_undist_live is not None
            and frame_u.shape == ref_undist_live.shape
        )

        frame_u_photo = None
        if not _fast_shadow:
            frame_u_photo = apply_photometric_correction(
                frame_u, self.photo_gain, self.photo_bias
            )

        if ENABLE_SHADOW_TOLERANT_STABILIZATION:
            t_shadow0 = time.perf_counter()
            if _fast_shadow:
                frame_u_for_tracking = None
                if getattr(self, "_gpu_b", False):
                    frame_u_for_tracking = stone_tracker.gpu_b_suppress(
                        np.asarray(self.photo_gain, dtype=np.float64), np.asarray(self.photo_bias, dtype=np.float64),
                        int(frame_u.shape[0]), int(frame_u.shape[1])
                    )
                    if frame_u_for_tracking is None:
                        print("GPU-vaihe B: virhe -> palataan CPU-polkuun")
                        self._gpu_b = False
                if frame_u_for_tracking is None:
                    frame_u_for_tracking = stone_tracker.suppress_shadow_background(
                        frame_u, ref_undist_live,
                        np.asarray(self.photo_gain, dtype=np.float64),
                        np.asarray(self.photo_bias, dtype=np.float64),
                        float(GRANITE_DIFF_THRESHOLD), float(SHADOW_V_DROP_MIN),
                        float(SHADOW_V_DROP_MAX), int(ICE_S_MAX), int(ICE_V_MIN)
                    )
            else:
                frame_u_for_tracking = suppress_static_background(
                    frame_u_photo, ref_undist_live, diff_threshold=GRANITE_DIFF_THRESHOLD
                )
            _e = _PROF.setdefault(pf + " varjonsietoinen taustanvaimennus (koko frame)", [0.0, 0]); _e[0] += time.perf_counter() - t_shadow0; _e[1] += 1
            # frame_u_for_tracking on jo taustanvaimennettu - C++:n oma vaimennus ohitetaan (0.0).
            haku_seuranta_diff_threshold = 0.0
        else:
            frame_u_for_tracking = frame_u_photo
            haku_seuranta_diff_threshold = GRANITE_DIFF_THRESHOLD

        return frame_u, frame_u_for_tracking, haku_seuranta_diff_threshold


class _LivePipeline:
    """LIUKUHIHNA (Testi_03_01), 3 vaihetta rinnan:
      A) tuottajasaie 1: luku + gray + stabilointi (vaihekorrelaatio) + set_transform
      B) tuottajasaie 2: warp+remap + valotasapaino + varjonsuodatus
      C) paasaie: HAKU/SEURANTA, tulokset, CSV
    Ruutujen jarjestys ja sisalto ovat samat kuin ilman hihnaa. Kaytetaan vasta kun calib_result on
    valmis ja live_state luotu."""

    def __init__(self, source_read, engine, prep, ref_gray, live_state, calib_result, first_index, depth=3, haku_ahead=None):
        import queue as _queue
        self._haku_ahead = haku_ahead      # v5.11: kutsu(idx, frame_u_for_tracking, thr) -> HAKU-future tai None
        self._qa = _queue.Queue(maxsize=depth)   # A -> B
        self._q = _queue.Queue(maxsize=depth)    # B -> paasaie
        self._stop = threading.Event()
        self._source_read = source_read
        self._engine = engine
        self._prep = prep
        self._ref_gray = ref_gray
        self._live_state = live_state
        self._calib_result = calib_result
        self._index = first_index
        _PIPE_STATS.update(wall0=time.perf_counter(), cpu0=time.process_time(), qC_sum=0, qC_n=0, qA_sum=0, qA_n=0, cpu_a=None, cpu_b=None, depth=depth)
        self._ta = threading.Thread(target=self._run_a, daemon=True)
        self._tb = threading.Thread(target=self._run_b, daemon=True)
        self._ta.start()
        self._tb.start()

    def _put(self, q, item, key):
        import queue as _queue
        t0 = time.perf_counter()
        while not self._stop.is_set():
            try:
                q.put(item, timeout=0.2)
                break
            except _queue.Full:
                continue
        _e = _PROF.setdefault(key, [0.0, 0])
        _e[0] += time.perf_counter() - t0; _e[1] += 1

    def _run_a(self):
        hog_analyysi.lower_thread_priority()      # v5.7: taustasaie (A) -> SEURANTA saa ytimet ensin
        if STAB_WORKERS > 1:
            return self._run_a_parallel()
        try:
            while not self._stop.is_set():
                t0 = time.perf_counter()
                frame = self._source_read()
                _e = _PROF.setdefault("pipe A: read(video)", [0.0, 0]); _e[0] += time.perf_counter() - t0; _e[1] += 1
                if frame is None or frame.size == 0:
                    self._put(self._qa, None, "pipe A: odottaa vaihetta B (jono taynna)")
                    return

                t0 = time.perf_counter()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                _e = _PROF.setdefault("pipe A: gray cvtColor", [0.0, 0]); _e[0] += time.perf_counter() - t0; _e[1] += 1

                t0 = time.perf_counter()
                dx, dy = _phase_correlate_cached(self._ref_gray, gray)
                stab = np.array([[1.0, 0.0, -dx], [0.0, 1.0, -dy]], dtype=np.float64)
                _e = _PROF.setdefault("pipe A: stabilointi (vaihekorrelaatio)", [0.0, 0]); _e[0] += time.perf_counter() - t0; _e[1] += 1

                t0 = time.perf_counter()
                self._engine.set_transform(stab)
                _e = _PROF.setdefault("pipe A: set_transform", [0.0, 0]); _e[0] += time.perf_counter() - t0; _e[1] += 1

                self._put(self._qa, (self._index, frame, stab), "pipe A: odottaa vaihetta B (jono taynna)")
                self._index += 1
        except BaseException as e:      # valitetaan eteenpain
            self._put(self._qa, e, "pipe A: odottaa vaihetta B (jono taynna)")
        finally:
            _PIPE_STATS["cpu_a"] = time.thread_time()

    def _run_a_parallel(self):
        """Vaihe A usealle ruudulle rinnan (STAB_WORKERS): luku + gray jarjestyksessa tassa saikeessa, vaihekorrelaatiot (ruudut ovat
        toisistaan riippumattomia: kukin verrataan moodikuvaan) tyontekijasaikeissa; tulokset viedaan jonoon ALKUPERAISESSA JARJESTYKSESSA,
        joten tulos on identtinen. A:n palveluaika = taman saikeen oma kierto (sis. odotuksen tyontekijoita), tyontekijoiden yhteenlaskettu aika raportoidaan 'bg:'-rivilla."""
        import collections
        from concurrent.futures import ThreadPoolExecutor
        ex = ThreadPoolExecutor(max_workers=STAB_WORKERS, initializer=hog_analyysi.lower_thread_priority)
        pending = collections.deque()

        def _job(gray):
            t = time.perf_counter()
            dxy = _phase_correlate_cached(self._ref_gray, gray)
            return dxy, time.perf_counter() - t

        def _emit_oldest():
            idx, frame, fut = pending.popleft()
            t0 = time.perf_counter()
            (dx, dy), dt_work = fut.result()
            _e = _PROF.setdefault("pipe A: stabilointi (vaihekorrelaatio)", [0.0, 0]); _e[0] += time.perf_counter() - t0; _e[1] += 1
            _e = _PROF.setdefault("bg: stabilointi tyoaika (rinnakkaiset tyontekijat, ei lisaa)", [0.0, 0]); _e[0] += dt_work; _e[1] += 1
            stab = np.array([[1.0, 0.0, -dx], [0.0, 1.0, -dy]], dtype=np.float64)
            t0 = time.perf_counter()
            self._engine.set_transform(stab)
            _e = _PROF.setdefault("pipe A: set_transform", [0.0, 0]); _e[0] += time.perf_counter() - t0; _e[1] += 1
            self._put(self._qa, (idx, frame, stab), "pipe A: odottaa vaihetta B (jono taynna)")

        try:
            first = True
            while not self._stop.is_set():
                t0 = time.perf_counter()
                frame = self._source_read()
                _e = _PROF.setdefault("pipe A: read(video)", [0.0, 0]); _e[0] += time.perf_counter() - t0; _e[1] += 1
                if frame is None or frame.size == 0:
                    while pending and not self._stop.is_set():
                        _emit_oldest()
                    self._put(self._qa, None, "pipe A: odottaa vaihetta B (jono taynna)")
                    return

                t0 = time.perf_counter()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                _e = _PROF.setdefault("pipe A: gray cvtColor", [0.0, 0]); _e[0] += time.perf_counter() - t0; _e[1] += 1

                if first:
                    # ensimmainen ruutu synkronisesti: alustaa referenssi-FFT-valimuistin ennen rinnakkaisajoa
                    first = False
                    fut0 = ex.submit(_job, gray)
                    fut0.result()
                    pending.append((self._index, frame, fut0))
                else:
                    pending.append((self._index, frame, ex.submit(_job, gray)))
                self._index += 1
                while len(pending) >= STAB_WORKERS:
                    _emit_oldest()
        except BaseException as e:      # valitetaan eteenpain
            self._put(self._qa, e, "pipe A: odottaa vaihetta B (jono taynna)")
        finally:
            ex.shutdown(wait=False)
            _PIPE_STATS["cpu_a"] = time.thread_time()

    def _run_b(self):
        import queue as _queue
        hog_analyysi.lower_thread_priority()      # v5.7: taustasaie (B)
        try:
            while not self._stop.is_set():
                t0 = time.perf_counter()
                _PIPE_STATS["qA_sum"] += self._qa.qsize(); _PIPE_STATS["qA_n"] += 1
                try:
                    item = self._qa.get(timeout=0.2)
                except _queue.Empty:
                    continue
                _e = _PROF.setdefault("pipe B: odottaa vaihetta A", [0.0, 0]); _e[0] += time.perf_counter() - t0; _e[1] += 1
                if item is None or isinstance(item, BaseException):
                    self._put(self._q, item, "pipe B: odottaa paasaiketta (jono taynna)")
                    return
                idx, frame, stab = item
                frame_u, frame_u_for_tracking, thr = self._prep.process(
                    frame, stab, idx, self._live_state, self._calib_result
                )
                # v5.11: HAKU kaynnistetaan heti kun ruutu on valmis (syote riippuu vain ruudusta ja kalibroinnista) -> tulos on
                # valmiina kun paasaie ehtii tahan ruutuun (jono PIPE_DEPTH ruutua). Sama syote ja sama ruutu kuin ennen -> sama tulos.
                haku_fut = self._haku_ahead(idx, frame_u_for_tracking, thr) if self._haku_ahead is not None else None
                self._put(self._q, {
                    "frame": frame, "stab": stab, "frame_u": frame_u,
                    "frame_u_for_tracking": frame_u_for_tracking, "thr": thr, "haku_future": haku_fut,
                }, "pipe B: odottaa paasaiketta (jono taynna)")
        except BaseException as e:
            self._put(self._q, e, "pipe B: odottaa paasaiketta (jono taynna)")
        finally:
            _PIPE_STATS["cpu_b"] = time.thread_time()

    def get(self):
        t0 = time.perf_counter()
        _PIPE_STATS["qC_sum"] += self._q.qsize(); _PIPE_STATS["qC_n"] += 1
        item = self._q.get()
        _e = _PROF.setdefault("pipe C: paasaie odottaa hihnaa (sisaltyy py: read(video)-riviin)", [0.0, 0])
        _e[0] += time.perf_counter() - t0; _e[1] += 1
        if isinstance(item, BaseException):
            raise item
        return item

    def close(self):
        import queue as _queue
        self._stop.set()
        for q in (self._q, self._qa):
            try:
                while True:
                    q.get_nowait()
            except _queue.Empty:
                pass
        self._ta.join(timeout=10.0)
        self._tb.join(timeout=10.0)


class _FramePrefetcher:
    """Videon luku+dekoodaus omassa saikeessaan (Testi_03_01): engine.read() vapauttaa GIL:n, joten
    dekoodaus ja paasaikeen tyo ajavat rinnan. Kaytetaan vain ELAVASSA vaiheessa (calib_result != None),
    jolloin engine.add_mode_frame (joka lukee engine.current_frame_:ia) ei ole enaa kaytossa.
    Framejarjestys ja -sisalto ovat identtiset suoran engine.read()-kutsun kanssa."""

    def __init__(self, engine, depth=4):
        import queue as _queue
        self._engine = engine
        self._q = _queue.Queue(maxsize=depth)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        import queue as _queue
        hog_analyysi.lower_thread_priority()      # v5.7: taustasaie (videon luku)
        while not self._stop.is_set():
            try:
                f = self._engine.read()
            except Exception as e:      # valitetaan paasaikeelle
                f = e
            while not self._stop.is_set():
                try:
                    self._q.put(f, timeout=0.2)
                    break
                except _queue.Full:
                    continue
            if isinstance(f, Exception) or f is None or f.size == 0:
                return

    def read(self):
        f = self._q.get()
        if isinstance(f, Exception):
            raise f
        return f

    def close(self):
        import queue as _queue
        self._stop.set()
        try:
            while True:
                self._q.get_nowait()
        except _queue.Empty:
            pass
        self._thread.join(timeout=5.0)


def run_pipeline(
    video_file,
    panel_data,
    calib_diag_output,
    csv_output,
    precomputed_calib_result=None,
    precomputed_profile_result=None,
    debug_video_output=None
):

    if live_source.active() is not None:
        # Testi_06_01: ruudut kameran puskurista (sama ModeEngine, ruudut annetaan Pythonista)
        engine = live_source.LiveEngine(mode_engine, live_source.active(), TILE_SIZE)
    else:
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
    haku_refiner = None
    track_refiner = None
    haku_sil_log_file = None
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
    stone_registry = {}   # vahvistetut kivet (stone_id -> tila), portitus videon lopussa
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
    debug_composer = None
    hog_overlays = []      # Testi_05_01: hog-hog -tekstit (debug-video), katso HOG-HOG -ANALYYSI
    hog_results = []

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

    # PIKSELITARKKA SUORA STABILOINTI (kayttajan pyynnosta, katso
    # kommentti alempana kaytonkohdalla) - moodikuvan RAAKA (ei viela
    # undistorted) harmaasavyreferenssi, jota vasten "loppuvideon" joka
    # frame vaihekorrelaatiolla verrataan. Asetetaan heti kun calib_
    # result tulee valmiiksi (katso alempana) - TAI HETI TASSA jos
    # calib_result annettiin jo valmiiksi laskettuna (precomputed_calib_
    # result, esim. testeissa): korjattu bugi, jossa loppuvideo_ref_gray
    # jai koskaan asettamatta talla polulla ja _phase_correlate_full_
    # frame kaatui "NoneType has no attribute shape" heti loppuvideo-
    # vaiheen ensimmaisella framella.
    loppuvideo_ref_gray = None

    if calib_result is not None:
        loppuvideo_ref_gray = cv2.cvtColor(
            calib_result["calib"]["frame"], cv2.COLOR_BGR2GRAY
        )

    # VALOTASAPAINO/KIRKKAUS-KORJAUS (kayttajan pyynnosta, katso
    # estimate_photometric_correction:in kommentti): xy-siirtyma haetaan
    # JOKA framella, mutta valotasapaino/kirkkaus vain KERRAN SEKUNNISSA
    # (hidas ajautuminen, liian hidas laskea joka framelle). photo_gain/
    # photo_bias pysyvat viimeisimpana laskettuina arvoina niiden
    # valissa olevilla frameilla.
    photo_gain = None
    photo_bias = None
    next_photo_update_frame = 0
    total_photometric_time = 0.0

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

    # ENABLE_SHADOW_TOLERANT_STABILIZATION (kayttajan huomio: tama EI
    # ollut mukana "Yhteensa mitattu" -summassa aiemmin, vaikka se
    # ajetaan joka elavan seurannan framella - raportti siis ALIARVIOI
    # kokonaisajan kun se on paalla).
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
    photo_executor = ThreadPoolExecutor(max_workers=1, initializer=hog_analyysi.lower_thread_priority)
    frame_prefetcher = None
    photo_future = None
    live_prep = _LivePrep(photo_executor, fps, width, height)
    if hasattr(stone_tracker, "set_spawn_filter"):
        if SPAWN_FILTER:
            stone_tracker.set_spawn_filter(
                SPAWN_RMS_MAX, SPAWN_NB_MIN, SPAWN_NB_MAX, SPAWN_ABS_X_MAX, SPAWN_SCORE_MAX
            )
            print(
                f"HAKU spawn-suodatin PAALLA: rms <= {SPAWN_RMS_MAX}, n_body {SPAWN_NB_MIN}..{SPAWN_NB_MAX}, "
                f"|X| <= {SPAWN_ABS_X_MAX} cm, pistemaara <= {SPAWN_SCORE_MAX}"
            )
        else:
            stone_tracker.set_spawn_filter(0.0, 0, 1000000, 1e9, 1e9)
    frame_pipeline = None

    def _run_haku_timed(*args):
        t0 = time.time()
        if HAKU_MULTI:
            result = stone_tracker.search_new_stones(
                *args, max_results=HAKU_MAX_RESULTS, max_attempts=HAKU_MAX_ATTEMPTS
            )
        else:
            result = stone_tracker.search_new_stone(*args)
        return result, time.time() - t0

    try:

        while True:

            t_frame_wall0 = time.perf_counter()
            _t_post0 = None
            pipe_item = None
            t_read0 = time.perf_counter()
            if frame_pipeline is not None:
                pipe_item = frame_pipeline.get()
                frame = None if pipe_item is None else pipe_item["frame"]
            elif calib_result is not None and VIDEO_PREFETCH:
                if frame_prefetcher is None:
                    frame_prefetcher = _FramePrefetcher(engine)
                frame = frame_prefetcher.read()
            else:
                frame = engine.read()
            total_read_time += time.perf_counter() - t_read0
            _PROF.setdefault("py: read(video)", [0.0, 0])[0] += time.perf_counter() - t_read0
            _PROF["py: read(video)"][1] += 1

            if frame is None or frame.size == 0:
                break
            t0 = time.perf_counter()
            gray = (
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if pipe_item is None else None
            )
            t1 = time.perf_counter()
            total_gray_time += t1 - t0
            _e = _PROF.setdefault("py: gray cvtColor", [0.0, 0]); _e[0] += t1 - t0; _e[1] += 1

            # ------------------------------------------------
            # PANEELIEN SEURANTA + STABILOINTIMATRIISI - AJETAAN
            # ENAA VAIN MOODIKUVAN RAKENTAMISEN AIKANA (kayttajan
            # pyynnosta empiirisesti korjattu, katso PIKSELITARKKA
            # SUORA STABILOINTI -kommentti alempana): panel_
            # stabilization_matrix EI ENAA OLE KAYTOSSA "loppuvideon"
            # stabilointiin kalibroinnin jalkeen (suora moodikuva-
            # vertailu korvasi senkin), joten koko paneiliseuranta+
            # RANSAC-laskenta (mitattu 37 ms/ruutu - lahes koko 25fps-
            # reaaliaikabudjetti) ohitetaan silloin kokonaan sen sijaan
            # etta laskettaisiin turhaan joka framella.
            #
            # HUOM (korjattu bugi): t_stab0 on asetettava TASSA, ENNEN
            # if calib_result is None -haaraa, koska total_stabilize_
            # compute_time += time.perf_counter() - t_stab0 lasketaan
            # AINA (myos "loppuvideon" suoran stabiloinnin ajaksi) -
            # jos t_stab0 asetettaisiin vain haaran SISALLA, se jaisi
            # jaatyneeksi vanhaan arvoon heti kun calib_result valmistuu,
            # ja mittari kasvaisi rajattomasti (havaittu: 50000 ms/ruutu).
            # ------------------------------------------------

            t_stab0 = time.perf_counter()

            if calib_result is None:

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

                panel_stabilization_matrix = (
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

                        panel_stabilization_matrix = (
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
                    panel_stabilization_matrix.copy()
                )

                panel_stabilization_matrix = np.median(
                    np.stack(
                        stabilization_history,
                        axis=0
                    ),
                    axis=0
                )

                panel_stabilization_matrix = np.asarray(
                    panel_stabilization_matrix,
                    dtype=np.float64
                )


                previous_stabilization_matrix = (
                    panel_stabilization_matrix.copy()
                )

            # ------------------------------------------------
            # PIKSELITARKKA SUORA STABILOINTI "LOPPUVIDEOLLE" MOODIKUVAAN
            # VERRATEN (kayttajan pyynnosta empiirisesti korjattu, katso
            # keskusteluhistoria): AIEMPI versio KETJUTTI joka framen
            # pienen deltan edellisen framen jo-vahvistettuun sijaintiin,
            # olettaen etta koko framen kaytto (ei vain muutama paneelin
            # kulmapiste) estaa karkaamisen. Kayttajan oma lyhyt testi
            # osoitti etta kivi/tracking EI silti alkanut samasta nakymasta
            # kuin moodikuva - oma diagnostiikka (Linux-ymparistossa, 65s
            # klippi) VAHVISTI tarkan syyn: KETJUTUS ON KUMULATIIVINEN
            # SUMMA jokaisen framen pienesta mittausvirheesta (klassinen
            # "dead reckoning" -ongelma), ja vaikka yksittainen delta on
            # pieni, se kertyy - samalla klipilla ajautui 60 sekunnissa yli
            # 70 pikselia sivuun (fg-pikselien maara kasvoi 13186:sta
            # 752325:een). SUORA vertailu joka framella KIINTEAAN
            # moodikuvareferenssiin sen sijaan EI kerry - jokainen mittaus
            # on itsenainen (ei riipu edellisesta framesta), joten
            # yksittaisen framen mittausvirhe (esim. liikkuva pelaaja
            # hairitsee vaihekorrelaatiota) EI jaa pysyvasti - seuraava
            # frame korjaa itsensa taas suoraan referenssiin, ei edellisen
            # (jo vaaran) framen pohjalta. SAMALLA klipilla mitattu siirtyma
            # pysyi koko 60s ajan alle 1 pikselin (kamera on kaytannossa
            # paikallaan) - katso diag_direct_ref_fix.py-tulokset.
            #
            # Vertailu tehdaan RAA'ASSA (ei viela vaannon/undistortin
            # jalkeisessa) pikseliavaruudessa suoraan calib["frame"]:a
            # (moodikuvan RAAKA, paneilistabiloitu mutta EI viela
            # undistorted versio) vasten - loppuvideo_ref_gray on
            # valimuistitettu heti kun calib_result tulee valmiiksi (katso
            # sen laskenta ylempana). Paneiliseurantaa/panel_stabilization_
            # matrix:ia kaytetaan siis enaa VAIN moodikuvan rakentamiseen
            # (calib_result is None) - sen jalkeen (profiiliskannaus +
            # elava seuranta) KOKO frame vaihekorrelaatiolla suoraan
            # referenssiin, EI ketjutettuna.
            # ------------------------------------------------

            if calib_result is None:

                stabilization_matrix = panel_stabilization_matrix

            else:

                # HUOM (korjattu merkkivirhe, kayttajan "miksi viivat eivat
                # haviaisi/miksi huononee" -kysymysten paljastama, katso
                # keskusteluhistoria): _phase_correlate_full_frame(A,B)
                # palauttaa siirtyman JOLLA A (referenssi) piti siirtaa
                # jotta saadaan B (nykyinen frame) - EI siirtyma jolla B
                # pitaa siirtaa jotta saadaan A. Nama ovat VASTAKKAISET
                # (kaanteiset toisiinsa nahden) puhtaalle translaatiolle.
                # Koska frame pitaa siirtaa PAIN referenssia (ei referenssi
                # pain framea), oikea korjaus on -dx,-dy, EI dx,dy suoraan.
                # Vahvistettu oikealla videodatalla (MAH00014, t=659s):
                # vanha (suora dx,dy) merkki antoi taustanvaimennuksessa
                # 266403 vuotanutta pikselia - HUONOMMIN kuin EI MITAAN
                # korjausta (234413) - kun taas -dx,-dy antoi 194114
                # (selvasti paras). Tama selittaa miksi reunaviivat eivat
                # havinneet kokonaan eivatka pysyneet tasaisina ajan
                # mukana: vaara suunta kasvatti virhetta sita enemman mita
                # suurempi todellinen siirtyma oli.
                if pipe_item is None:
                    dx, dy = _phase_correlate_cached(
                        loppuvideo_ref_gray, gray
                    )

                    stabilization_matrix = np.array(
                        [[1.0, 0.0, -dx], [0.0, 1.0, -dy]],
                        dtype=np.float64
                    )
                else:
                    stabilization_matrix = pipe_item["stab"]

            total_stabilize_compute_time += time.perf_counter() - t_stab0
            _e = _PROF.setdefault("py: stabilointi (vaihekorrelaatio+paneelit)", [0.0, 0]); _e[0] += time.perf_counter() - t_stab0; _e[1] += 1

            # ------------------------------------------------
            # LÄHETÄ STABILOINNIN MATRIX C++:LLE
            # ------------------------------------------------
            t4 = time.perf_counter()
            if pipe_item is None:
                engine.set_transform(
                    stabilization_matrix
                )
            t5 = time.perf_counter()
            total_transform_time += t5 - t4
            _e = _PROF.setdefault("py: set_transform", [0.0, 0]); _e[0] += t5 - t4; _e[1] += 1

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
                    # LOPPUVIDEON SUORAN STABILOINNIN REFERENSSI
                    # (kayttajan pyynnosta empiirisesti korjattu, katso
                    # PIKSELITARKKA SUORA STABILOINTI -kommentti alempana
                    # kaytonkohdalla): valimuistitetaan moodikuvan RAAKA
                    # (ei viela undistorted) harmaasavykuva heti kun
                    # calib_result on valmis - jokainen "loppuvideon"
                    # frame verrataan TASTA LAHTIEN suoraan tahan, EI
                    # ketjutettuna edellisen framen sijaintiin (katso
                    # miksi ketjutus hylattiin).
                    # ------------------------------------------------

                    loppuvideo_ref_gray = cv2.cvtColor(
                        calib["frame"], cv2.COLOR_BGR2GRAY
                    )

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

                            # Testi_03_04: alfa-aariviiva + hylkayssaannot (katso ALFA-AARIVIIVA-kommentti)
                            solo_observations = candidate_observations
                            if PROFILE_ALPHA:
                                candidate_observations, a_st = attach_alpha_observations(
                                    video_file, calib_result["calib"], candidate_observations
                                )
                                level_solo = alpha_common_level(candidate_observations)
                                candidate_observations, n_area_rej = apply_area_rule(
                                    candidate_observations, level_solo
                                )
                                print(
                                    f"  alfa-havainnot: {a_st['in']} -> {len(candidate_observations)} "
                                    f"(ei graniittia {a_st['ei_graniittia']}, suhde {a_st['suhde']}, "
                                    f"tumma alue {a_st['tumma']}, koko {n_area_rej}); "
                                    f"alfa-taso (tama kivi) {level_solo:.3f}"
                                )
                                solo_observations = alpha_contour_observations(
                                    candidate_observations, level_solo
                                )

                            _, solo_ok = try_fit_profile(
                                calib_result["pose"],
                                solo_observations,
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

                                fit_observations = accumulated_stones
                                if PROFILE_ALPHA:
                                    # yhteinen alfa-taso kaikkien hyvaksyttyjen havaintojen mediaanina
                                    alpha_level = alpha_common_level(accumulated_stones)
                                    fit_observations = alpha_contour_observations(
                                        accumulated_stones, alpha_level
                                    )
                                    print(f"  yhteinen alfa-taso {alpha_level:.3f} "
                                          f"({len(accumulated_stones)} havainnosta)")

                                profile, riittava = try_fit_profile(
                                    calib_result["pose"],
                                    fit_observations
                                )
                                if PROFILE_ALPHA and profile is not None:
                                    profile["alpha_level"] = alpha_level

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

            # MUISTA TARKISTAA (Testi_03_04, sovittu kayttajan kanssa): profiilin opettelussa kaytetty ALFA-kartta
            # (peittavyys, paikallinen varjotaso, yhteinen alfa-taso) parantaisi todennakoisesti merkittavasti myos
            # SEURANNAN luottamusta: reuna alipikselin tarkkuudella ilman varjoa/sisennysta, ja sama yhteinen taso
            # kaikille kiville. Ei viela kaytossa seurannassa - tarkastellaan kun kaydaan seuranta lapi
            # (katso ALFA-AARIVIIVA-kommentti profiilin sovituksen lahella).

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
                    # ring_r_frac_guess - alkuarvaus ERILLISELLE saturaatio-
                    # pohjaiselle "rengashaulle" (kahvan kiinnityslevyn reunan
                    # etsintaan, katso stone_tracker.cpp:n oma kommentti) -
                    # kaytetaan sovitettua handle_r_frac:ia myos taman haun
                    # alkuarvauksena, koska se on parempi (datasta sovitettu)
                    # lahtokohta kuin vanha kiintea HANDLE_NOTCH_R_FRAC-vakio.
                    #
                    # handle_r_frac - kahvan aiheuttaman kolon SADE, joka
                    # VAHENNETAAN graniittirungon konveksista peitteesta
                    # (predictedNotchHull/profileResiduals, katso kamera9_01.
                    # py:n HANDLE_NOTCH_R_FRAC-kommentti). Kayttajan pyynnosta
                    # fit_stone_profile SOVITTAA taman datasta, ja stone_
                    # tracker.cpp:n elava seuranta kayttaa NYT samaa sovitettua
                    # arvoa ajonaikaisena parametrina (EI enaa kiintea C++-vakio).
                    ring_r_frac_guess = profile["handle_r_frac"]
                    handle_r_frac = profile["handle_r_frac"]

                    if HAKU_SILHOUETTE:
                        haku_refiner = haku_silhouette.SilhouetteRefiner(
                            k9, calib_result["pose"], R_max, H_total, shape_deltas, handle_r_frac, k94=k94
                        )
                        print("HAKU siluettitarkennus PAALLA (haku_silhouette.py, C++: %s)" % (haku_refiner.body_pts is not None))
                    if SEURANTA_SILHOUETTE:
                        track_refiner = haku_silhouette.SilhouetteRefiner(
                            k9, calib_result["pose"], R_max, H_total, shape_deltas, handle_r_frac,
                            max_shift_px=SEURANTA_SIL_SHIFT_PX, k94=k94
                        )
                        print(f"SEURANTA maskituki PAALLA (inside >= {SEURANTA_MIN_INSIDE}, siluettitarkennus +-{SEURANTA_SIL_SHIFT_PX} px, kaikille ruuduille kun Y > {SEURANTA_SIL_ALL_Y_CM:.0f} cm)")
                        # v5.11: siluettitarkennus suoraan SEURANNAN C++-kivisaikeissa (SIL_IN_BATCH=0 = erillinen vaihe kuten ennen)
                        _sil_cfg = track_refiner.batch_config()
                        if SIL_IN_BATCH and _sil_cfg is not None and hasattr(stone_tracker, "set_seuranta_silhouette"):
                            stone_tracker.set_seuranta_silhouette(*_sil_cfg)

                    print(
                        "Rakennetaan kiven pintavarireferenssia "
                        f"({len(accumulated_stones)} havainnosta)..."
                    )

                    stone_color_reference = build_stone_color_reference(
                        video_file, calib_result["calib"],
                        calib_result["pose"], local_pts_body, 28,
                        accumulated_stones
                    )

                    raw_csv_path = (
                        os.path.splitext(csv_output)[0] + "_raaka.csv"
                        if THROW_GATE else csv_output
                    )
                    csv_file = open(raw_csv_path, "w", newline="")
                    csv_writer = csv.writer(csv_file)
                    csv_writer.writerow(CSV_HEADER)

                    if debug_video_output is not None:

                        _dbg_vw, _dbg_vh, _dbg_scale, _dbg_total_w = hog_analyysi.debug_layout(width, height)
                        debug_video_writer = hog_analyysi.AsyncVideoWriter(hog_analyysi.open_debug_writer(
                            debug_video_output, fps, (_dbg_total_w, _dbg_vh)
                        ))
                        debug_composer = hog_analyysi.DebugComposer(width, height)
                        print(f"Debug-video: stabiloitu+korjattu, ei maskeja, kaannetty 90 astetta vastapaivaan, {_dbg_total_w}x{_dbg_vh} (video {_dbg_vw}x{_dbg_vh} + paneelit {hog_analyysi.PANEL_W} px/puoli)")

                    print()
                    print(
                        f"Elava moni-kiven seuranta alkaa (frame "
                        f"{frame_index}, kulunut {time.time() - start_time:.1f} s) - CSV: {csv_output}"
                    )
                    _live = live_source.active()
                    if _live is not None and LIVE_HYPPY:
                        # Testi_06_01: kalibroinnin aikana kertynyt viive pois - seuranta alkaa uusimmasta ruudusta.
                        # Hyppiville lukijoille ei enaa tarvita pitkaa historiaa -> puskuri pienenee LIVE_TAAKSE_SEURANTA_S:iin.
                        _lag0 = _live.store.lag_frames()
                        _to = _live.store.skip_to_latest(keep_frames=0, new_keep_back=int(LIVE_TAAKSE_SEURANTA_S * fps))
                        print(f"Live: hypataan uusimpaan ruutuun (ohitettiin {_lag0} ruutua = {_lag0 / fps:.1f} s kalibroinnin aikana kertynytta viivetta; "
                              f"puskurin indeksi {_to})")
                    if CV_SINGLE_PERSIST and hasattr(stone_tracker, "set_cv_single_thread_persistent"):
                        stone_tracker.set_cv_single_thread_persistent(1)     # v5.12: ei saiepoolin uudelleenluontia joka kutsulla

                    live_state = {
                        "map1": map1, "map2": map2,
                        "local_pts_body": local_pts_body,
                        "local_pts_search": local_pts_search,
                        "R_max": R_max, "H_total": H_total,
                        "ring_r_frac_guess": ring_r_frac_guess,
                        "handle_r_frac": handle_r_frac,
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
                if pipe_item is None:
                    frame_u, frame_u_for_tracking, haku_seuranta_diff_threshold = live_prep.process(
                        frame, stabilization_matrix, frame_index, live_state, calib_result
                    )
                else:
                    frame_u = pipe_item["frame_u"]
                    frame_u_for_tracking = pipe_item["frame_u_for_tracking"]
                    haku_seuranta_diff_threshold = pipe_item["thr"]

                # LIUKUHIHNA kaynnistetaan ensimmaisen (paasaikeessa valmistellun) elavan ruudun JALKEEN:
                # tuottajasaie ottaa seuraavat ruudut (frame_index+1...) ja valmistelee ne rinnan seurannan kanssa.
                if pipe_item is None and frame_pipeline is None and LIVE_PIPELINE:
                    if VIDEO_PREFETCH:
                        if frame_prefetcher is None:
                            frame_prefetcher = _FramePrefetcher(engine)
                        _pipe_source = frame_prefetcher.read
                    else:
                        _pipe_source = engine.read
                    live_prep.prefix = "pipe B:"

                    _haku_ahead = None
                    if HAKU_AHEAD and EVICT_AT_CAP:
                        # v5.11: sama HAKU-kutsu kuin alempana (EVICT_AT_CAP: ehto riippuu vain ruudun indeksista)
                        _ha_pose = calib_result["pose"]
                        _ha_ref = calib_result["calib"]["frame_undistorted"]
                        _ha_body = live_state["local_pts_body"]
                        _ha_search = live_state["local_pts_search"]

                        def _haku_ahead(idx, fut_frame, thr):
                            if idx % haku_interval_frames != 0:
                                return None
                            y_c = (k92.SEARCH_Y_MIN_CM + k92.SEARCH_Y_MAX_CM) / 2.0
                            y_h = (k92.SEARCH_Y_MAX_CM - k92.SEARCH_Y_MIN_CM) / 2.0
                            return haku_executor.submit(
                                _run_haku_timed,
                                fut_frame, _ha_ref,
                                _ha_body, _ha_search,
                                _ha_pose["K"], _ha_pose["R"], _ha_pose["t"],
                                0.0, k92.SEARCH_X_HALF_WIDTH_CM, y_c, y_h,
                                k92.SEARCH_COARSE_STEP_CM, k92.SEARCH_FINE_STEP_CM,
                                k92.SEARCH_SCORE_THRESHOLD,
                                live_state["R_max"], live_state["H_total"],
                                live_state["ring_r_frac_guess"], live_state["handle_r_frac"],
                                thr
                            )

                    frame_pipeline = _LivePipeline(
                        _pipe_source, engine, live_prep, loppuvideo_ref_gray,
                        live_state, calib_result, frame_index + 1, depth=PIPE_DEPTH,
                        haku_ahead=_haku_ahead
                    )
                    if STAB_WORKERS > 1 or PIPE_DEPTH != 3:
                        print(f"Liukuhihna: stabilointi {STAB_WORKERS} ruudulle rinnan, jonojen syvyys {PIPE_DEPTH}")

                timestamp = frame_index / fps
                pose = calib_result["pose"]
                local_pts_body = live_state["local_pts_body"]
                local_pts_search = live_state["local_pts_search"]

                # --------------------------------------------
                # ENABLE_SHADOW_TOLERANT_STABILIZATION (katso sen kommentti
                # taman tiedoston alkupaassa). HUOM: frame_u ITSE (vari-
                # tarkistus, debug-video) EI koskaan taustanvaimenneta tassa
                # - vain erillinen frame_u_for_tracking-kopio, jota kaytetaan
                # VAIN HAKU/SEURANTA-kutsuissa alla.
                #
                # ENABLE_SUBPIXEL_ALIGNMENT:in erillinen jalkikorjaus (tama
                # kommenttiblokki oli aiemmin tassa) POISTETTU (kayttajan
                # havainto + oma empiirinen vahvistus, katso keskustelu-
                # historia): kayttaja huomasi etta jaa/katto/seinat eivat
                # olleet kaytannossa taysin valkoisia debug-videossa - oma
                # testi paljasti etta tama jalkikorjaus (alunperin suunni-
                # teltu PIENEKSI hienosaadoksi epatarkan paneilipohjaisen
                # stabiloinnin paalle) on NYT TARPEETON JA HAITALLINEN, koska
                # PAASTABILOINTI (stabilization_matrix, katso PIKSELITARKKA
                # SUORA STABILOINTI ylempana) tekee jo TASMALLEEN saman
                # suoran moodikuvavertailun - jalkikorjaus vain mittasi
                # lahes saman asian UUDELLEEN ja lisasi sen PAALLE (havaittu:
                # jalkikorjaus -1.7px samaan suuntaan kuin paastabiloinnin jo
                # tekema -0.89px korjaus), mika KASVATTI reunapikselien
                # vuotoa taustanvaimennuksessa 115599:sta 178113:een (+54%)
                # samalla testiframella, ei pienentanyt sita.
                # --------------------------------------------
                ref_undist_live = calib_result["calib"]["frame_undistorted"]

                _frame_dump_hook(frame_index, frame_u_for_tracking, frame_u)

                haku_future = None

                if (
                    len(active_stones) >= MAX_CONCURRENT_STONES
                    and not EVICT_AT_CAP
                    and frame_index % haku_interval_frames == 0
                ):
                    _haku_log(frame_index, len(active_stones), False, False,
                              None, None, None, False, None, None, None)

                if (
                    (len(active_stones) < MAX_CONCURRENT_STONES or EVICT_AT_CAP)
                    and frame_index % haku_interval_frames == 0
                ):

                    x_center = 0.0
                    y_center = (
                        k92.SEARCH_Y_MIN_CM + k92.SEARCH_Y_MAX_CM
                    ) / 2.0
                    y_half = (
                        k92.SEARCH_Y_MAX_CM - k92.SEARCH_Y_MIN_CM
                    ) / 2.0

                    _t_hs = time.perf_counter()
                    if pipe_item is not None and pipe_item.get("haku_future") is not None:
                        haku_future = pipe_item["haku_future"]      # v5.11: kaynnistetty jo vaiheessa B
                    else:
                        haku_future = haku_executor.submit(
                            _run_haku_timed,
                            frame_u_for_tracking, ref_undist_live,
                            local_pts_body, local_pts_search,
                            pose["K"], pose["R"], pose["t"],
                            x_center, k92.SEARCH_X_HALF_WIDTH_CM, y_center, y_half,
                            k92.SEARCH_COARSE_STEP_CM, k92.SEARCH_FINE_STEP_CM,
                            k92.SEARCH_SCORE_THRESHOLD,
                            live_state["R_max"], live_state["H_total"],
                            live_state["ring_r_frac_guess"], live_state["handle_r_frac"],
                            haku_seuranta_diff_threshold
                        )
                    _PROF.setdefault("py: HAKU submit (Python, saikeen kaynnistys)", [0.0, 0])
                    _PROF["py: HAKU submit (Python, saikeen kaynnistys)"][0] += time.perf_counter() - _t_hs
                    _PROF["py: HAKU submit (Python, saikeen kaynnistys)"][1] += 1

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
                # Testi_05_03: kun kivi on lahempana kuin SEURANTA_HALF_RATE_Y_CM, sita seurataan vain joka toisella ruudulla
                # (kaukana mittaus on epatarkempaa -> kaikki data; lahella tarkka -> harvempi riittaa). Ohitettu kivi sailyy
                # aktiivisena ja sen ohittamat ruudut kirjataan s["gap_extra"]:ksi (hakualue/ennuste kasvavat vastaavasti).
                if SEURANTA_HALF_RATE_Y_CM > 0 and frame_index % 2 == 1:
                    seuranta_stones = []
                    for _s in active_stones:
                        if _s["last_xy"][1] < SEURANTA_HALF_RATE_Y_CM:
                            _s["gap_extra"] = _s.get("gap_extra", 0) + 1
                            still_active.append(_s)
                            debug_draw_items.append((
                                _s["last_xy"][0], _s["last_xy"][1], _s["stone_id"], (0, 200, 0), f"{_s['stone_id']}"
                            ))
                        else:
                            seuranta_stones.append(_s)
                else:
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
                        [s["misses"] + 1 + s.get("gap_extra", 0) for s in seuranta_stones],
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
                    batch_results = _seuranta_dispatch(
                        frame_index, [s["stone_id"] for s in seuranta_stones],
                        (
                            frame_u_for_tracking, ref_undist_live,
                            X0_arr, Y0_arr,
                            half_range_x_arr, half_range_y_arr,
                            local_pts_body, local_pts_search,
                            pose["K"], pose["R"], pose["t"],
                            k92.TRACK_COARSE_STEP_CM, k92.TRACK_FINE_STEP_CM,
                            k92.TRACK_SCORE_THRESHOLD,
                            live_state["R_max"], live_state["H_total"],
                            live_state["ring_r_frac_guess"], live_state["handle_r_frac"],
                            TRACK_MAX_BACKWARD_CM,
                            haku_seuranta_diff_threshold
                        ),
                        X0_arr, Y0_arr, half_range_x_arr, half_range_y_arr,
                        pred=(
                            _stone_prediction(
                                seuranta_stones, frame_index,
                                half_range_x_arr, half_range_y_arr
                            ) if TRACKER_MODE == "ensemble" else None
                        )
                    )
                    total_seuranta_time += time.time() - t_seuranta0
                    _e = _PROF.setdefault("py: SEURANTA dispatch+C++ seinakello", [0.0, 0])
                    _e[0] += time.time() - t_seuranta0; _e[1] += 1
                    _t_post0 = time.perf_counter()
                    n_seuranta_calls += 1
                    n_seuranta_stone_updates += len(seuranta_stones)

                    color_ref = live_state["color_reference"]
                    frame_u_f64 = None

                    # v5.7: siluettitarkennus kaikille loydetyille kiville kerralla (C++, kivet rinnan). Syote = C++-tulos (X_cm, Y_cm)
                    # ennen alla olevia tarkistuksia, kuten ennenkin (kivisilmukka ei muuta niita ennen refine-kutsua) -> tulos sama.
                    _sil_results = {}
                    if track_refiner is not None:
                        _sil_idx = [i for i, r in enumerate(batch_results) if r["found"]]
                        if _sil_idx and all("sil" in batch_results[i] for i in _sil_idx):
                            _sil_results = {i: tuple(batch_results[i]["sil"]) for i in _sil_idx}     # v5.11: laskettu jo C++:ssa
                        elif _sil_idx:
                            with _prof("py: SEURANTA maskituki + siluettitarkennus"):
                                _sil_out = track_refiner.refine_many(
                                    frame_u_for_tracking, [(batch_results[i]["X_cm"], batch_results[i]["Y_cm"]) for i in _sil_idx]
                                )
                            _sil_results = dict(zip(_sil_idx, _sil_out))

                    for _si, (s, refined) in enumerate(zip(seuranta_stones, batch_results)):

                        # --------------------------------
                        # VARIHISTORIAN KERAYS (COLOR_DEBUG-diagnostiikkaa
                        # varten - EI enaa vaikuta "pysahtynyt"-paatokseen,
                        # katso kommentti COLOR_REF_*-vakioiden kohdalla
                        # taman tiedoston alkupaassa). HUOM (kayttajan pyynnosta
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

                        # v5.7: varidiagnostiikka (color_diff_history) ei vaikuta mihinkaan paatokseen eika tulosteeseen (vain COLOR_DEBUG-
                        # tulostus, katso COLOR_REF_*-kommentti) -> lasketaan vain kun COLOR_DEBUG on asetettu (saastaa ~1 ms/ruutu paasaikeessa).
                        if refined["found"] and color_ref is not None and _COLOR_DEBUG:

                            with _prof("py: SEURANTA jalkeen: varidiagnostiikka (astype+median)"):
                                if frame_u_f64 is None:
                                    # Testi_03_01: bilineaarinen naytteistys toimii suoraan uint8-kuvalla
                                    # (numpy nostaa painokertoimien kanssa float64:ksi) - koko framen
                                    # astype(float64) (~22 MB) maksoi ~5 ms/ruutu.
                                    frame_u_f64 = frame_u

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

                        if track_refiner is not None and refined["found"]:
                            _far = refined["Y_cm"] > SEURANTA_SIL_ALL_Y_CM
                            _tx, _ty, _tinfo = _sil_results[_si]
                            if os.environ.get("SEURANTA_SIL_DEBUG") and int(os.environ.get("SEURANTA_SIL_DEBUG").split(":")[0]) <= frame_index <= int(os.environ.get("SEURANTA_SIL_DEBUG").split(":")[1]):
                                print(f"[SILDBG] f={frame_index} id={s['stone_id']} tarkka={refined.get('tarkka')} X,Y=({refined['X_cm']:.1f},{refined['Y_cm']:.1f}) -> ({_tx:.1f},{_ty:.1f}) "
                                      f"ok={_tinfo.get('ok')} in0={_tinfo.get('inside0')} in1={_tinfo.get('inside1')} sh={_tinfo.get('shift_px')} min_y={s['min_y_seen']:.1f}")
                            refined = dict(refined)
                            _min_in = SEURANTA_MIN_INSIDE if (_far or not refined.get("tarkka")) else SEURANTA_MIN_INSIDE_NEAR
                            _sil_ok = _tinfo.get("ok") and _tinfo.get("inside0", 0.0) >= _min_in
                            if _far:
                                # kaukana: tarkka ruutu ei hylkaydy, ei-tarkka hylataan jos ei maskitukea
                                if _sil_ok:
                                    refined["X_cm"], refined["Y_cm"] = _tx, _ty
                                elif not refined.get("tarkka"):
                                    refined["found"] = False
                            else:
                                if not _sil_ok:
                                    refined["found"] = False          # lahella portti koskee myos tarkkoja ruutuja
                                elif not refined.get("tarkka"):
                                    refined["X_cm"], refined["Y_cm"] = _tx, _ty

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

                        # --------------------------------
                        # KESKIVIIVAETAISYYSTARKISTUS: katso ASETUKSET-
                        # kommentti MAX_ABS_X_FROM_CENTERLINE_CM:in
                        # kohdalla - hylataan (kuten TAAKSEPAIN-tarkistus
                        # ylla) havainto joka veisi kiven liian kauas
                        # keskiviivasta, todennakoisesti SEURANNAN
                        # ajauduttua sivuttain liikkuvaan kohteeseen.
                        # --------------------------------

                        if refined["found"] and (
                            abs(refined["X_cm"]) > MAX_ABS_X_FROM_CENTERLINE_CM
                        ):
                            refined = dict(refined)
                            refined["found"] = False

                        if refined["found"]:

                            s["last_xy"] = (
                                refined["X_cm"], refined["Y_cm"]
                            )
                            s.setdefault("rms_hist", deque(maxlen=10)).append(
                                refined.get("rms_px") if refined.get("rms_px") is not None else 1e9
                            )
                            s["min_y_seen"] = min(
                                s["min_y_seen"], refined["Y_cm"]
                            )
                            s["misses"] = 0
                            s["gap_extra"] = 0

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

                            reject_low_tarkka = False

                            if not s["confirmed"]:

                                s["pending_rows"].append(
                                    (frame_index, timestamp, dict(refined))
                                )

                                if (
                                    abs(refined["Y_cm"] - s["y0_first_cm"])
                                    >= MIN_CONFIRMED_THROW_DISPLACEMENT_CM
                                ):

                                    # --------------------------------
                                    # TARKKA-OSUUSTARKISTUS: katso
                                    # ASETUKSET-kommentti MIN_PRECONFIRM_
                                    # TARKKA_OBSERVATIONS/FRACTION:in
                                    # kohdalla - torjuu pelaajan/
                                    # lakaisijan SEURANNAN loysemman
                                    # per-frame-kynnyksen kautta
                                    # "ajautumisen" virheellisesti
                                    # vahvistetuksi kiveksi.
                                    # --------------------------------

                                    n_pending = len(s["pending_rows"])
                                    n_tarkka = sum(
                                        1 for _, _, prow in s["pending_rows"]
                                        if prow.get("tarkka")
                                    )
                                    tarkka_frac = (
                                        n_tarkka / n_pending
                                        if n_pending > 0 else 1.0
                                    )

                                    # Testi_03_01: kaukaiset (30+ m) AIDOT heitot saavat huonon sovituksen
                                    # (rms 8-10 px, tarkka=0) mutta ovat nopeita ja vahvistuvat pian; hauras
                                    # "1 ruutu myohemmin -> hylatty" -kisa hylkasi aitoja heittoja. Hylataan
                                    # siksi vain jos myos rms-mediaani on korkea (roskaradat 15-25 px).
                                    _pre_rms = [
                                        prow.get("rms_px") for _, _, prow in s["pending_rows"]
                                        if prow.get("rms_px") is not None
                                    ]
                                    _pre_rms_med = float(np.median(_pre_rms)) if _pre_rms else 1e9

                                    if (
                                        n_pending >= MIN_PRECONFIRM_TARKKA_OBSERVATIONS
                                        and tarkka_frac < MIN_PRECONFIRM_TARKKA_FRACTION
                                        and _pre_rms_med >= PRECONFIRM_RESCUE_MAX_RMS
                                    ):

                                        reject_low_tarkka = True
                                        s["pending_rows"] = []
                                        print(
                                            f"[frame {frame_index}] Ehdokas "
                                            f"{s['stone_id']} hylatty (tarkka-"
                                            f"osuus {n_tarkka}/{n_pending} "
                                            f"({tarkka_frac:.0%}) liian "
                                            "matala ennen liikevahvistusta "
                                            "- todennakoisesti pelaaja/"
                                            "lakaisija, ei aito kivi)."
                                        )

                                    else:

                                        s["confirmed"] = True
                                        n_confirmed_stones += 1

                                        stone_registry[s["stone_id"]] = s
                                        for pf, pt, prow in s["pending_rows"]:
                                            s.setdefault("all_rows", []).append((pf, pt, dict(prow)))
                                            _write_stone_csv_row(
                                                csv_writer, pf, pt,
                                                s["stone_id"], prow
                                            )
                                            s["confirmed_tarkka_window"].append(
                                                bool(prow.get("tarkka"))
                                            )

                                        s["pending_rows"] = []

                            else:

                                # --------------------------------
                                # JATKUVA TARKKA-OSUUSTARKISTUS (Testi_
                                # 02_02, kayttajan pyynnosta - katso
                                # keskusteluhistoria): MIN_PRECONFIRM_
                                # TARKKA_* tarkistaa tarkka-osuuden VAIN
                                # kerran, liikevahvistushetkella. Havait-
                                # tiin oikealla datalla (0001/MAH00014,
                                # jaasuodatuksen jalkeen) etta muutama
                                # kandidaatti (esim. pyyhkija joka kavelee
                                # pitkan matkan aidon kiven vieressa)
                                # lapaisee talla hetkella riittavan
                                # tarkka-osuuden mutta putoaa sen jalkeen
                                # pysyvasti matalaksi (esim. 9-28% n. 400
                                # havainnon ajan) - vahentaa 400+ vaarin
                                # CSV-riviä 50:aan. Kynnys (0.5, n>=50)
                                # validoitu koko videoiden oikealla
                                # datalla: kaikki selvasti aidot pitkaan
                                # seuratut kivet pysyivat aina >=51.5%:ssa
                                # (marginaali), kaikki selvasti ongelmal-
                                # liset kandidaatit alittivat 0.5:n heti
                                # ensimmaisen 50 havainnon jalkeen (13-48%).
                                #
                                # LIUKUVA IKKUNA (ei enaa kumulatiivinen
                                # koko elinkaarelta, kayttajan pyynnosta
                                # tehdyn maskiyhdistelma-korjauksen
                                # (stone_tracker.cpp, createForeground-
                                # FromWhitened) jalkeen havaittu tarve):
                                # aito kivi voi olla hetken (esim. juuri
                                # heiton jalkeen, heittaja/lakaisija viela
                                # vierella) kosketuksissa pelaajaan JA
                                # SEN JALKEEN seurata puhtaasti pitkan
                                # matkaa yksin - kumulatiivinen koko-
                                # elinkaaren osuus jaisi talloin PYSYVASTI
                                # alle kynnyksen tuon alkuhetken takia,
                                # vaikka loppuosa olisi taydellinen (havait-
                                # tiin: n_body 26-29/rms 0.5-0.9px koko
                                # lopun ajan, mutta silti hylattiin 49
                                # havainnon kohdalla). Ikkuna (deque,
                                # maxlen=MIN_CONFIRMED_TARKKA_OBSERVATIONS)
                                # "unohtaa" vanhan kontaminaation automaat-
                                # tisesti kun tarpeeksi uusia havaintoja on
                                # kertynyt, mutta havaitsee silti PYSYVAN
                                # ajautumisen (jolloin koko ikkuna on jat-
                                # kuvasti matala) - sama kynnys (0.5,
                                # n>=50) sailyy, koska se on jo validoitu.
                                # --------------------------------

                                s["confirmed_tarkka_window"].append(
                                    bool(refined.get("tarkka"))
                                )
                                window = s["confirmed_tarkka_window"]

                                if (
                                    len(window) >= MIN_CONFIRMED_TARKKA_OBSERVATIONS
                                    and (sum(window) / len(window))
                                    < MIN_CONFIRMED_TARKKA_FRACTION
                                ):

                                    reject_low_tarkka = True
                                    print(
                                        f"[frame {frame_index}] Kivi "
                                        f"{s['stone_id']} lopetetaan "
                                        f"(tarkka-osuus (liukuva ikkuna) "
                                        f"{sum(window)}/{len(window)} "
                                        f"({100*sum(window)/len(window):.0f}%) "
                                        "pudonnut pysyvasti liian matalaksi "
                                        "- todennakoisesti SEURANTA ajautunut "
                                        "pelaajaan/lakaisijaan aidon kiven "
                                        "vierella)."
                                    )

                                else:

                                    s.setdefault("all_rows", []).append(
                                        (frame_index, timestamp, dict(refined))
                                    )
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

                            if HOG_ANALYSIS and s.get("confirmed"):
                                _hog_check(
                                    s, frame_index, fps, k8.NEAR_HOGLINE_Y_CM, k8.FAR_HOGLINE_Y_CM,
                                    hog_overlays, hog_results, frame_u, csv_output, pose
                                )

                            stopped = False

                            # Testi_05_03: puolitetulla ruututaajuudella (vain parilliset ruudut) historian ikkuna on parillinen ->
                            # ikkunan pituus ei koskaan tayta tasan stop_tracking_frames; sallitaan 1 ruudun vajaus.
                            if (
                                history[-1][0] - history[0][0]
                                >= stop_tracking_frames - (1 if SEURANTA_HALF_RATE_Y_CM > 0 else 0)
                            ):
                                _, old_x, old_y = history[0]
                                displacement = math.hypot(
                                    s["last_xy"][0] - old_x,
                                    s["last_xy"][1] - old_y
                                )

                                if displacement < STOP_TRACKING_DISPLACEMENT_CM:

                                    # --------------------------------
                                    # PYSAHTYMINEN pelkasta GEOMETRIASTA
                                    # (kayttajan pyynnosta, katso keskus-
                                    # teluhistoria): aiemmin vaadittiin
                                    # TAMAN LISAKSI varivahvistus (viimeisen
                                    # sekunnin ka. vari COLOR_STOP_MAX_AVG_
                                    # DIFF:in alle) - havaittiin koko videon
                                    # lapikaynnissa etta tama esti "pysah-
                                    # tynyt"-ilmoituksen aidoilla, selvasti
                                    # paikallaan olevilla kivilla (esim.
                                    # kivi joka ei liikkunut yli minuuttiin
                                    # ei koskaan tulostanut "pysahtynyt" -
                                    # jaljelle jaaneet MAX_CONCURRENT_STONES-
                                    # paikat tukkeutuivat, estaen uusien
                                    # aitojen heittojen rekisteroinnin).
                                    # Nyt: 3D-mallin/maskin sijainti (EI
                                    # liikkunut yli STOP_TRACKING_DISPLACE-
                                    # MENT_CM:aa STOP_TRACKING_SECONDS:in
                                    # aikana) RIITTAA yksinaan.
                                    # --------------------------------

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

                            if not stopped and not reject_low_tarkka:
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

                if DUP_MERGE_CM > 0 and len(active_stones) > 1:
                    # Duplikaattien yhdistaminen (katso DUP_MERGE_CM-kommentti).
                    # Laskuri s["dup_frames"]: montako perakkaista framea rata
                    # on ollut toisen radan DUP_MERGE_CM:n sisalla (molemmat found).
                    losers = set()
                    for ia in range(len(active_stones)):
                        a = active_stones[ia]
                        for ib in range(ia + 1, len(active_stones)):
                            b = active_stones[ib]
                            key = (b["stone_id"],)
                            close = (
                                a["misses"] == 0 and b["misses"] == 0
                                and math.hypot(
                                    a["last_xy"][0] - b["last_xy"][0],
                                    a["last_xy"][1] - b["last_xy"][1]
                                ) < DUP_MERGE_CM
                            )
                            cnt = a.setdefault("dup_counts", {})
                            if close:
                                cnt[b["stone_id"]] = cnt.get(b["stone_id"], 0) + 1
                            else:
                                cnt.pop(b["stone_id"], None)
                            if close and cnt.get(b["stone_id"], 0) >= DUP_MERGE_FRAMES:
                                # havioaja: vahvistamaton ensin, muuten uudempi id
                                if a["confirmed"] != b["confirmed"]:
                                    loser = a if not a["confirmed"] else b
                                else:
                                    loser = b if b["stone_id"] > a["stone_id"] else a
                                losers.add(loser["stone_id"])
                    if losers:
                        for s_lose in active_stones:
                            if s_lose["stone_id"] in losers:
                                s_lose["pending_rows"] = []
                                print(
                                    f"[frame {frame_index}] Rata {s_lose['stone_id']} "
                                    "yhdistetty toiseen (duplikaatti, "
                                    f"< {DUP_MERGE_CM:.0f} cm)."
                                )
                        active_stones = [
                            x for x in active_stones if x["stone_id"] not in losers
                        ]

                # --------------------------------------------
                if _t_post0 is not None:
                    _e = _PROF.setdefault("py: SEURANTA jalkeen: tulossilmukka yht. (sis. varidiagn., CSV, portti)", [0.0, 0])
                    _e[0] += time.perf_counter() - _t_post0; _e[1] += 1

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

                    with _prof("py: HAKU odotus (result) - taakka SEURANNAN jalkeen"):
                        haku_result_raw, haku_dt = haku_future.result()
                    _e = _PROF.setdefault("py: HAKU taustasaikeen oma kesto (ei lisaa, ajaa rinnan)", [0.0, 0])
                    _e[0] += haku_dt; _e[1] += 1
                    total_haku_time += haku_dt
                    n_haku_calls += 1

                    haku_result_list = (
                        haku_result_raw if isinstance(haku_result_raw, list)
                        else [haku_result_raw]
                    )
                    new_this_scan = []

                    if haku_refiner is not None:
                        for _hr in haku_result_list:
                            if not _hr.get("found"):
                                continue
                            _x1, _y1, _info = haku_refiner.refine(
                                frame_u_for_tracking, _hr["X_cm"], _hr["Y_cm"]
                            )
                            if _info.get("ok") and _info.get("inside1", 1.0) < SEURANTA_MIN_INSIDE:
                                _hr["found"] = False          # ei maskitukea (esim. pelaajan paa) -> ei kivi
                                continue
                            if _info.get("ok"):
                                if HAKU_SIL_LOG:
                                    if haku_sil_log_file is None:
                                        haku_sil_log_file = open(HAKU_SIL_LOG, "w", newline="")
                                        haku_sil_log_file.write("frame,X_haku,Y_haku,X_new,Y_new,shift_cm,shift_px,score0,score1\n")
                                    haku_sil_log_file.write(
                                        f"{frame_index},{_hr['X_cm']:.2f},{_hr['Y_cm']:.2f},{_x1:.2f},{_y1:.2f},"
                                        f"{_info['shift_cm']:.2f},{_info['shift_px']:.2f},{_info['score0']:.3f},{_info['score1']:.3f}\n"
                                    )
                                    haku_sil_log_file.flush()
                                _hr["X_haku"], _hr["Y_haku"] = _hr["X_cm"], _hr["Y_cm"]
                                _hr["X_cm"], _hr["Y_cm"] = _x1, _y1

                    if not haku_result_list:
                        _haku_log(frame_index, len(active_stones), True, False,
                                  None, None, None, False, None, None, None)

                    for haku_result in haku_result_list:

                        if not haku_result["found"]:
                            _haku_log(frame_index, len(active_stones), True, False,
                                      None, None, haku_result.get("score"), False,
                                      None, None, None)

                        if haku_result["found"]:

                            refined = haku_result
                            bx, by = refined["X_cm"], refined["Y_cm"]

                            already_tracked = any(
                                math.hypot(
                                    bx - s["last_xy"][0], by - s["last_xy"][1]
                                ) < NEW_STONE_DEDUP_CM
                                for s in active_stones
                                if s["confirmed"]
                            ) or any(
                                math.hypot(bx - nx, by - ny) < NEW_STONE_SAME_SCAN_CM
                                for nx, ny in new_this_scan
                            )
                            if (
                                not already_tracked
                                and EVICT_AT_CAP
                                and len(active_stones) >= MAX_CONCURRENT_STONES
                            ):
                                _victim = _pick_eviction_victim(active_stones)
                                if _victim is not None:
                                    _victim["pending_rows"] = []
                                    active_stones = [
                                        x for x in active_stones
                                        if x["stone_id"] != _victim["stone_id"]
                                    ]
                                    print(
                                        f"[frame {frame_index}] Rata {_victim['stone_id']} "
                                        "poistettu (paikanvaraus uudelle ehdokkaalle, huonoin sovitus)."
                                    )
                            already_tracked = (
                                already_tracked
                                or len(active_stones) >= MAX_CONCURRENT_STONES
                            )

                            if HAKU_LOG:
                                near = min(
                                    ((math.hypot(bx - s["last_xy"][0], by - s["last_xy"][1]), s)
                                     for s in active_stones),
                                    key=lambda x: x[0], default=(None, None)
                                )
                                _haku_log(frame_index, len(active_stones), True, True,
                                          bx, by, haku_result.get("score"),
                                          not already_tracked,
                                          near[1]["stone_id"] if near[1] else None,
                                          near[0],
                                          near[1]["confirmed"] if near[1] else None)

                            # --------------------------------------------
                            # DIAGNOSTIIKKA (kayttajan raportoima bugi, katso
                            # keskusteluhistoria): HAKU tunnisti PELAAJAN/
                            # LAKAISIJAN kiveksi (pelkkaan muotoon/kokoon
                            # perustuva C++-yhteissovitus ei tunne varia).
                            # Tulostetaan TASSA vain diagnostiikkana (ei viela
                            # hylkaa mitaan) uuden ehdokkaan varipoikkeama
                            # kivivarireferenssiin - kaytetaan naiden lukujen
                            # keraamiseen sopivan hylkayskynnyksen maarittamiseksi
                            # (katso HAKU_COLOR_MAX_DIFF alempana taman
                            # validoinnin jalkeen).
                            # --------------------------------------------

                            if os.environ.get("HAKU_COLOR_DEBUG") and live_state.get("color_reference") is not None:
                                haku_frame_u_f64 = frame_u.astype(np.float64)
                                haku_median_diff = color_match_median_diff(
                                    haku_frame_u_f64, local_pts_body, pose,
                                    bx, by, live_state["color_reference"],
                                    width, height
                                )
                                print(
                                    f"[HAKU_COLOR_DEBUG] frame={frame_index} "
                                    f"ehdokas ({bx:.1f},{by:.1f}) "
                                    f"varidiff={haku_median_diff}"
                                )

                            if not already_tracked:

                                stone_id = next_stone_id
                                next_stone_id += 1
                                new_this_scan.append((bx, by))

                                active_stones.append({
                                    "stone_id": stone_id,
                                    "last_xy": (
                                        refined["X_cm"], refined["Y_cm"]
                                    ),
                                    "min_y_seen": refined["Y_cm"],
                                    "misses": 0,
                                    "gap_extra": 0,
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
                                    "confirmed_tarkka_window": deque(
                                        maxlen=MIN_CONFIRMED_TARKKA_OBSERVATIONS
                                    ),
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
                                    f" [ehdokasominaisuudet score={refined.get('score')} "
                                    f"rms={refined.get('rms_px')} n_body={refined.get('n_body')} "
                                    f"n_ring={refined.get('n_ring')} tarkka={refined.get('tarkka')}]"
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

                _t_dbg0 = time.perf_counter()
                if debug_video_writer is not None:

                    # Testi_05_02 v5.2: koko debug-videon piirto (valokorjaus, ääriviivat, kokoonpano, kirjoitus) omassa saikeessa;
                    # paasaie vain luovuttaa tilannekuvan (frame_u ei muutu enaa taman ruudun jalkeen, tulokset kopioidaan).
                    debug_video_writer.submit(
                        _render_debug_frame, frame_u, live_prep.photo_gain, live_prep.photo_bias, list(debug_draw_items),
                        pose, f"frame {frame_index}  t={timestamp:.2f}s  kivia={len(active_stones)}",
                        list(hog_results), hog_analyysi.plus_x_is_right(pose["K"], pose["R"], pose["t"]),
                        debug_composer, local_pts_body)
                    _e = _PROF.setdefault("py: debug-video (piirto + kirjoitus)", [0.0, 0]); _e[0] += time.perf_counter() - _t_dbg0; _e[1] += 1

            _e = _PROF.setdefault("FRAME_KOKO", [0.0, 0])
            _e[0] += time.perf_counter() - t_frame_wall0; _e[1] += 1
            frame_index += 1
            if MAX_FRAME and frame_index >= MAX_FRAME:
                break

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

                _live_txt = ""
                if live_source.active() is not None:
                    _lst = live_source.active().store
                    _live_txt = f" | live-viive {_lst.lag_frames() / fps:.1f} s, pudotettu {_lst.dropped}"
                    eta_minutes = eta_secs = 0
                print(
                    f"Ruutu "
                    f"{frame_index} / "
                    f"{total_frames if total_frames > 0 else 'live'} | "
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
                    f"{_live_txt}"
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
                    f"stabilointimatriisi {(total_stabilize_compute_time / processed) * 1000:.2f} ms/ruutu | "
                    f"warpAffine+remap(koko frame) {(total_warp_remap_time / processed) * 1000:.2f} ms/ruutu | "
                    f"valotasapaino {(total_photometric_time / processed) * 1000:.2f} ms/ruutu | "
                    f"varjosuodatus {(total_shadow_suppress_time / processed) * 1000:.2f} ms/ruutu"
                )
                if PULLONKAULA_RAPORTTI:
                    _bn = _bottleneck_compact()
                    if _bn:
                        print(_bn)

    finally:

        executor.shutdown(
            wait=True
        )

        haku_executor.shutdown(
            wait=True
        )
        if frame_pipeline is not None:
            frame_pipeline.close()
        photo_executor.shutdown(wait=True)
        if frame_prefetcher is not None:
            frame_prefetcher.close()

        if csv_file is not None:
            csv_file.close()

        if THROW_GATE and stone_registry:
            all_tr = [
                (sid, st_["all_rows"]) for sid, st_ in stone_registry.items()
                if st_.get("all_rows")
            ]
            n_before = sum(1 for _, rw in all_tr if _track_throw_class(rw) > 0)
            gated = _select_throws(all_tr)
            with open(csv_output, "w", newline="") as gf:
                gw = csv.writer(gf)
                gw.writerow(CSV_HEADER)
                for sid, rows in gated:
                    for rf, rt, rr in rows:
                        _write_stone_csv_row(gw, rf, rt, sid, rr)
            print(
                f"Heittoportti: {len(stone_registry)} vahvistettua rataa -> "
                f"{n_before} heittomaista rataa -> {len(gated)} heittoa (yksi / hogline-ylitys) "
                f"({csv_output})"
            )

        if HOG_ANALYSIS:
            _hog_write_csv(hog_results, csv_output)

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

    # Liukuhihnan tuottajasaikeen mittaukset mukaan yhteenvetolaskureihin (ne ajavat RINNAN paasaikeen kanssa).
    def _ps(k):
        return _PROF.get(k, [0.0, 0])[0]
    total_read_time += _ps("pipe A: read(video)")
    total_gray_time += _ps("pipe A: gray cvtColor")
    total_stabilize_compute_time += _ps("pipe A: stabilointi (vaihekorrelaatio)")
    total_transform_time += _ps("pipe A: set_transform")
    total_warp_remap_time += _ps("pipe B: warpAffine+remap (koko frame)")
    total_shadow_suppress_time += _ps("pipe B: varjonsietoinen taustanvaimennus (koko frame)")

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
    print(f"stabilointimatriisin laskenta (paneili-RANSAC moodikuvavaiheessa, "
          f"suora vaihekorrelaatio moodikuvaan loppuvideolla): "
          f"{total_stabilize_compute_time:.2f}s yhteensa, "
          f"{(total_stabilize_compute_time / processed_frames) * 1000:.2f} ms/ruutu")
    print(f"warpAffine+remap (KOKO frame, joka elavan seurannan ruutu): "
          f"{total_warp_remap_time:.2f}s yhteensa, "
          f"{(total_warp_remap_time / processed_frames) * 1000:.2f} ms/ruutu")
    print(f"valotasapaino/kirkkaus-korjaus (kerran sekunnissa): "
          f"{total_photometric_time:.2f}s yhteensa, "
          f"{(total_photometric_time / processed_frames) * 1000:.2f} ms/ruutu")
    print(f"varjonsietoinen taustanvaimennus (ENABLE_SHADOW_TOLERANT_STABILIZATION): "
          f"{total_shadow_suppress_time:.2f}s yhteensa, "
          f"{(total_shadow_suppress_time / processed_frames) * 1000:.2f} ms/ruutu")
    muu_yhteensa = (
        total_read_time + total_stabilize_compute_time
        + total_warp_remap_time + total_gray_time
        + total_tracking_time + total_transform_time
        + total_photometric_time + total_shadow_suppress_time
    )
    print(f"Yhteensa HAKU+SEURANTA+muu mitattu: "
          f"{((total_haku_time + total_seuranta_time + muu_yhteensa) / processed_frames) * 1000:.2f} "
          "ms/ruutu keskimaarin (25fps-reaaliaikatavoite = 40.0 ms/ruutu, "
          "tiukempi tavoite hyvalla marginaalilla = 20.0 ms/ruutu)")
    print("Versio: " + _version_string())
    print("===========================================================")
    _print_prof_report(processed_frames, n_seuranta_stone_updates)

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


def main(debug=None, start_time=None, end_time=None, video=None, live=None, panel_ref=None):

    # debug=None (oletus): kayta DEBUG_SAVE_TRACKING_VIDEO-vakion
    # arvoa (katso sen kommentti). debug=True/False komentoriviltä
    # (-d/--debug, katso alempana if __name__=="__main__") ohittaa
    # vakion - kayttajan pyynnosta dynaaminen paalle/pois-kytkenta
    # ilman lahdekoodin muokkausta.
    effective_debug = (
        DEBUG_SAVE_TRACKING_VIDEO if debug is None else debug
    )

    if live is not None:
        # Testi_06_01: elava kamerasyote - ei tiedostovalintaa eika ffmpeg-leikkausta. video_file = tulostiedostojen
        # nimipohja (<kansio>/live_<aika>.mp4; tiedostoa ei ole, ruudut tulevat puskurista).
        video_file = live["video_file"]
        print()
        print(f"Live-syote: {live['kuvaus']}")
        print(f"Tulokset: {os.path.splitext(video_file)[0]}_*")
    else:
        if video:
            input_file = video          # --video: ei tiedostovalitsinta
        else:
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

        # -y: leikattu video (<nimi>_leikattu.mp4) ylikirjoitetaan AINA ilman kysymysta (ilman tata ffmpeg kysyy "File exists. Overwrite? [y/N]")
        command = ["ffmpeg", "-y"]

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

        if panel_ref:
            reference_file = panel_ref          # Testi_06_01: --paneelit (ei valitsinta)
            print(f"Referenssi-paneelitiedosto: {reference_file}")
        else:
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

        cap = live_source.open_capture(video_file)      # Testi_06_01: live-tilassa ensimmainen ruutu puskurista

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
            f"kahvan_r={profile['handle_r_frac']:.3f} "
            f"({profile['handle_r_frac']*profile['R_max_cm']:.2f} cm), "
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


def _parse_size(txt):
    if not txt or str(txt).strip() in ("0", "kamera"):
        return 0, 0
    w, h = str(txt).lower().split("x")
    return int(w), int(h)


def _start_live(args):
    """Testi_06_01: kaynnistaa kameran (tai simulaation) taustasaikeeseen ja aktivoi puskurin main.py:n lukijoille."""
    import signal
    out_w, out_h = _parse_size(args.live_koko)
    cam_w, cam_h = _parse_size(args.live_kamerakoko)
    # puskurin pitaa mahtua: historia taaksepain + profiilin opettelun ikkuna eteenpain (STONE_TRACK_WINDOW_SECONDS) + varaa
    buffer_s = max(args.live_puskuri_s, args.live_taakse_s + STONE_TRACK_WINDOW_SECONDS + 5.0)
    # pakollinen historia (profiilin opettelu lukee STONE_TRACK_WINDOW_SECONDS taaksepain siemenruudusta): taman alle ei poisteta
    min_back_s = min(args.live_taakse_s, STONE_TRACK_WINDOW_SECONDS + 1.0)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    outdir = args.live_kansio or os.getcwd()
    os.makedirs(outdir, exist_ok=True)
    base = os.path.join(outdir, args.live_nimi or f"live_{stamp}")
    recorder = None
    if args.live_sim:
        src = live_source.FileSimSource(args.live_sim, realtime=bool(args.live_sim_tahti), buffer_s=buffer_s,
                                        keep_back_s=args.live_taakse_s, out_w=out_w, out_h=out_h, min_keep_back_s=min_back_s)
        kuvaus = f"SIMULAATIO {args.live_sim} ({'reaaliaika' if args.live_sim_tahti else 'ei tahdistusta'})"
    else:
        dev = None if args.live in (None, "auto") else int(args.live)

        def _open(backend, conv):
            return live_source.CameraSource(device=dev, cap_w=cam_w or 1920, cap_h=cam_h or 1080, out_w=out_w, out_h=out_h,
                                            target_fps=args.live_fps, buffer_s=buffer_s, keep_back_s=args.live_taakse_s,
                                            fourcc=args.live_fourcc, backend_name=backend, min_keep_back_s=min_back_s,
                                            conversion=conv, raw_order=args.live_raakajarjestys)

        src = None
        if args.live_muunnos == "auto" and sys.platform.startswith("win") and not args.live_taustaj:
            # v6.4: Windows-mittaus (kamera_testi): DSHOW + ajuri 30 ms CPU/ruutu, MSMF + raaka YUY2 + GPU 11 ms -> kokeillaan ensin MSMF + gpu.
            # Laitenumero: MSMF numeroi laitteet eri jarjestyksessa kuin DSHOW -> annettu numero koskee vain DSHOW:ta (MSMF etsii 1920x1080-laitteen).
            try:
                cand = live_source.CameraSource(device=None, cap_w=cam_w or 1920, cap_h=cam_h or 1080, out_w=out_w, out_h=out_h,
                                                 target_fps=args.live_fps, buffer_s=buffer_s, keep_back_s=args.live_taakse_s,
                                                 fourcc=args.live_fourcc, backend_name="MSMF", min_keep_back_s=min_back_s,
                                                 conversion="gpu", raw_order=args.live_raakajarjestys)
                if cand.conversion == "gpu":
                    src = cand
                else:
                    print(f"Live: MSMF + gpu ei kaytettavissa (muunnos {cand.conversion}) -> DirectShow + ajuri")
                    cand.close()
            except Exception as e:
                print(f"Live: MSMF-kameraa ei saatu auki ({e}) -> DirectShow + ajuri")
            if src is None:
                src = _open("DSHOW", "ajuri")
        else:
            src = _open(args.live_taustaj, "ajuri" if args.live_muunnos == "auto" else args.live_muunnos)
        kuvaus = f"KAMERA laite {src.device}"
    if args.live_tallenna:
        try:
            recorder = live_source.FfmpegRecorder(base + "_live.mp4", src.store.width, src.store.height, src.store.fps)
            src.recorder = recorder
            print(f"Live-tallennus: {recorder.path} (ffmpeg/{recorder.encoder})")
        except Exception as e:
            print(f"Live-tallennus ei kaytettavissa: {e}")
    print("Live-lahde: " + ", ".join(f"{k}={v}" for k, v in src.info.items()))
    print(f"Live-puskuri: enintaan {buffer_s:.0f} s (~{buffer_s * src.store.fps * src.store.width * src.store.height * 3 / 1e9:.1f} GB), "
          f"historia kalibroinnissa {args.live_taakse_s:.0f} s, seurannassa {LIVE_TAAKSE_SEURANTA_S:.0f} s. Lopetus: Ctrl+C"
          + (f" tai {args.live_kesto:.0f} s" if args.live_kesto > 0 else ""))
    live_source.activate(src)
    src.start()          # v6.2: vasta tallentajan kytkemisen jalkeen

    def _sigint(_sig, _frm):
        print("\nLive: lopetus pyydetty (Ctrl+C) - kasitellaan puskuri loppuun ja kirjoitetaan tulokset. "
              "Paina Ctrl+C uudelleen keskeyttaaksesi heti.")
        src.stop("Ctrl+C")
        signal.signal(signal.SIGINT, signal.default_int_handler)

    signal.signal(signal.SIGINT, _sigint)
    if args.live_kesto > 0:
        threading.Timer(args.live_kesto, lambda: src.stop(f"--live-kesto {args.live_kesto:.0f} s")).start()
    return src, dict(video_file=base + ".mp4", kuvaus=kuvaus, base=base, recorder=recorder)


def _finish_live(src, cfg):
    src.stop("ajo valmis")
    src.join(5.0)
    st = src.store
    fps = st.fps
    print()
    print("=== LIVE-SYOTE ===")
    print(f"Kasiteltyja ruutuja {len(st.log)}, puskurin ruutuja yhteensa {st._head}, ohitettu hypyissa {st.skipped} "
          f"({st.skipped / fps:.1f} s), pudotettu (puskuri taynna) {st.dropped}, suurin viive {st.max_lag_frames} ruutua "
          f"({st.max_lag_frames / fps:.1f} s). Lopetuksen syy: {st._stop_reason or '-'}")
    if getattr(src, "n_conv", 0):
        print(f"Kamerasaie (CPU/ruutu): luku {src.cpu_read / max(1, src.n_conv) * 1000:.1f} ms + muunnos/pienennys "
              f"{src.cpu_conv / src.n_conv * 1000:.1f} ms (muunnos: {src.conversion})")
    if cfg.get("recorder") is not None and getattr(cfg["recorder"], "n", 0):
        print(f"Tallennussaie (CPU/ruutu): {cfg['recorder'].cpu / cfg['recorder'].n * 1000:.1f} ms")
    path = cfg["base"] + "_kivien_sijainnit_live_aikaleimat.csv"
    if st.write_log(path):
        print(f"Live-aikaleimat (kasitelty ruutu -> kameran ruutu -> seinakello): {path}")
    if cfg.get("recorder") is not None:
        cfg["recorder"].close()
        print(f"Live-tallennus: {cfg['recorder'].path} (pudotettu {cfg['recorder'].dropped})")


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
    
    _arg_parser.add_argument(
        "--video", type=str, default=None,
        help="Videotiedoston polku (ohittaa tiedostovalitsimen), esim. --video D:\\Tikku\\Suorita\\MAH00014.MP4"
    )

    _arg_parser.add_argument(
        "--max-frame", type=int, default=0,
        help=(
            "Pysayta ajo (ja tulosta nopeus- ja pullonkaularaportti) kun tama ruutu on kasitelty - nopea testi omalla koneella, "
            "esim. --max-frame 3000. Ohittaa MAX_FRAME-ymparistomuuttujan jos annettu."
        )
    )

    _arg_parser.add_argument(
        "--no-debug", action="store_true", default=False,
        help="Pakota debug-video POIS (vertailuajo: nopeus ilman debug-videota). Ohittaa --debug ja DEBUG_SAVE_TRACKING_VIDEO."
    )

    # Testi_06_01: elava kamerasyote
    _arg_parser.add_argument("--live", nargs="?", const="auto", default=None,
                             help="Lue elava kuva kamerasta (Cam Link). Valinnainen laitenumero, esim. --live 1 (oletus: etsi 1920x1080-laite).")
    _arg_parser.add_argument("--live-sim", default=None,
                             help="Testaus ilman kameraa: videotiedosto 'kamerana' (puskuri + hyppy kuten live-tilassa).")
    _arg_parser.add_argument("--live-sim-tahti", type=int, default=1,
                             help="--live-sim: 1 = reaaliaikainen tahti (kuten kamera, oletus), 0 = niin nopeasti kuin kasittely ehtii (ei pudotuksia).")
    _arg_parser.add_argument("--live-koko", default="1280x720",
                             help="Kasittelykoko (kameran kuva pienennetaan), oletus 1280x720 (testattu putki). 0 = kameran oma koko.")
    _arg_parser.add_argument("--live-kamerakoko", default="1920x1080", help="Kameralta pyydetty koko (oletus 1920x1080).")
    _arg_parser.add_argument("--live-fourcc", default=None, help="Kameralta pyydetty pakkausmuoto, esim. MJPG, YUY2, NV12 (oletus: kameran oma).")
    _arg_parser.add_argument("--live-taustaj", default=None,
                             help="Kameran taustajarjestelma: DSHOW tai MSMF (oletus: automaattinen, katso --live-muunnos auto).")
    _arg_parser.add_argument("--live-muunnos", default="auto", choices=["auto", "ajuri", "raw", "gpu"],
                             help="Kameran kuvan muunnos BGR:ksi: auto (oletus; Windowsissa ensin MSMF + gpu, jos ei onnistu DirectShow + ajuri), "
                                  "ajuri, raw (raaka YUY2 + OpenCV), gpu (raaka YUY2 + OpenCL).")
    _arg_parser.add_argument("--live-raakajarjestys", default=None, choices=["YUY2", "YVYU", "UYVY"],
                             help="Pakota raakakuvan tavujarjestys (ohittaa automaattisen valinnan; vain selvasti vaara kuva hylataan).")
    _arg_parser.add_argument("--live-fps", type=float, default=25.0,
                             help="Kasittelyn tavoite-fps: jos kamera antaa selvasti enemman (esim. 50), kaytetaan joka n:s ruutu. 0 = kaikki.")
    _arg_parser.add_argument("--live-puskuri-s", type=float, default=90.0, help="Puskurin enimmaiskoko sekunteina (RAM: 1280x720 ~ 69 MB/s).")
    _arg_parser.add_argument("--live-taakse-s", type=float, default=40.0,
                             help="Kalibroinnin aikana sailytettava historia (s); profiilin opettelu lukee 30 s taaksepain.")
    _arg_parser.add_argument("--live-kesto", type=float, default=0.0, help="Lopeta automaattisesti N sekunnin jalkeen (0 = Ctrl+C lopettaa).")
    _arg_parser.add_argument("--live-kansio", default=None, help="Tulosten kansio (oletus: nykyinen kansio).")
    _arg_parser.add_argument("--live-nimi", default=None, help="Tulostiedostojen nimipohja (oletus live_<paiva>_<aika>).")
    _arg_parser.add_argument("--live-tallenna", action="store_true", help="Tallenna kasiteltava kuva myos videoksi (<pohja>_live.mp4, QSV/x264).")
    _arg_parser.add_argument("--paneelit", default=None,
                             help="Referenssi-paneelitiedosto (*_panel_corners.txt) ilman valitsinta (live-tilassa tarpeen).")

    _args = _arg_parser.parse_args()

    if _args.max_frame:
        MAX_FRAME = _args.max_frame
    if _args.no_debug:
        _args.debug = False

    _live_cfg = None
    _live_src = None
    if _args.live is not None or _args.live_sim:
        _live_src, _live_cfg = _start_live(_args)

    try:
        main(debug=_args.debug,
            start_time=_args.start,
            end_time=_args.end,
            video=_args.video,
            live=_live_cfg,
            panel_ref=_args.paneelit)
    finally:
        if _live_src is not None:
            _finish_live(_live_src, _live_cfg)
