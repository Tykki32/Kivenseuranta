import os
import math
import csv
import time
import argparse
import subprocess
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import mode_engine
import stone_tracker


# ============================================================
# PESIEN MITAT
# ============================================================
BLUE_OUTER_RADIUS_M = 1.829


BLUE_INNER_RADIUS_M = 1.219


RED_OUTER_RADIUS_M = 0.610


RED_INNER_RADIUS_M = 0.152


BLUE_OUTER_RADIUS_CM = BLUE_OUTER_RADIUS_M * 100.0


BLUE_INNER_RADIUS_CM = BLUE_INNER_RADIUS_M * 100.0


RED_OUTER_RADIUS_CM = RED_OUTER_RADIUS_M * 100.0


RED_INNER_RADIUS_CM = RED_INNER_RADIUS_M * 100.0


HOUSE_RADIUS_CM = BLUE_OUTER_RADIUS_CM


NEAR_HOUSE_Y_CM = HOUSE_RADIUS_CM


HOUSE_DISTANCE_CM = 3474.7


FAR_HOUSE_Y_CM = NEAR_HOUSE_Y_CM + HOUSE_DISTANCE_CM


# ============================================================
# LOPULLISEN (TOP-DOWN) KUVAN ASETUKSET
#
# HUOM: Y-akseli on KAANNETTY: fyysisesti pieni Y (lahempi pesa)
# piirretaan kuvan ALAOSAAN ja suuri Y (kaukainen pesa) kuvan
# YLAOSAAN.
# ============================================================
OUTPUT_X_MIN_CM = -200.0


OUTPUT_X_MAX_CM = 200.0


OUTPUT_Y_MIN_CM = 0.0


OUTPUT_Y_MAX_CM = 4000.0


PIXELS_PER_CM = 2.0


HOUSE_CROP_HALF_HEIGHT_CM = 300.0


# Hogline (etulinja) sijaitsee curlingissa 6.4 m paassa T-linjalta
# (pesan keskipisteesta), molemmin puolin kenttaa.
HOG_LINE_DISTANCE_FROM_TEE_CM = 640.0


# ============================================================
# LAHEMMAN PESAN ELLIPSIEN HAKUASETUKSET
# ============================================================
NEAR_BLUE_MIN_AREA = 1000


NEAR_BLUE_MIN_RATIO = 0.20


NEAR_BLUE_MIN_SIZE_RATIO = 0.30


NEAR_RED_MIN_AREA = 50


NEAR_RED_MIN_RATIO = 0.15


NEAR_RED_MIN_SIZE_RATIO = 0.15


# ============================================================
# OBJEKTIIVIN VAARISTYMAN ITSEKALIBROINTI JA HOMOGRAFIAN ROBUSTIUS
#
# Yksittainen 3x3-homografia on LINEAARINEN (projektiivinen) kuvaus,
# eika se pysty korjaamaan objektiivin EPALINEAARISTA sateittaista
# vaaristymaa (barrel/pincushion). Tama nakyy jaljella olevana pesien
# soikeutena ja hogline-viivojen vinoutena, vaikka korrespondenssi-
# pisteet olisi mitattu tarkasti - homografialla ei yksinkertaisesti
# ole vapausasteita korjata epalineaarista vaaristymaa.
#
# Koska erillisia kalibrointikuvia (esim. shakkilautaa) ei ole
# saatavilla ja objektiivin zoomaus voi vaihdella kuvien valilla,
# vaaristymaa EI kalibroida etukateen kiintein kertoimin. Sen sijaan
# yhden parametrin sateittainen vaaristyma (k1) ESTIMOIDAAN JOKA
# KUVASTA ERIKSEEN suoraan jo tunnetuista pesa-/hogline-korrespons-
# sipisteista (katso estimate_radial_distortion_k1) - sama periaate
# kuin "plumb-line"-itsekalibroinnissa: tunnetun geometrian (pesat
# ovat ympyroita, hoglinet suoria ja kohtisuorassa keskilinjaa
# vastaan) pitaisi toteutua kuvassa, ja k1 valitaan niin etta se
# toteutuu mahdollisimman hyvin.
# ============================================================

# HUOM: Homografiat (talla sivulla ja compute_corrected_homography/
# main:ssa) sovitetaan method=0:lla (tavallinen pienimman nelion
# sovitus), EI RANSAC:lla. Testattu oikealla datalla: nailla ~22
# korrespondenssipisteella (17 tiiviisti ryhmittynytta lahempaa +
# 5 harvaa kaukaista) RANSAC:n satunnaisotanta osuu usein kokonaan
# lahemman pesan tiiviiseen ryhmaan, mika tuottaa lahes degeneroi-
# tuneen homografian joka ekstrapoloituu hallitsemattomasti (tuhansien
# pikselien virhe) kaukaiselle pesalle. Plain LSQ oli aina vakaa ja
# jopa tarkempi.
K1_SEARCH_RANGE = 0.6


K1_SEARCH_STEPS = 25


K1_SEARCH_REFINE_ROUNDS = 3


K1_SEARCH_REFINE_SHRINK = 6.0


# Vaaristymaa ei oteta kayttoon, jos se ei parantaisi korrespondenssi-
# pisteiden RMS-uudelleenprojisointivirhetta riittavasti - talla
# valtetaan sovittamasta "vaaristymaa" pelkkaan mittauskohinaan.
K1_MIN_RELATIVE_IMPROVEMENT = 0.08


K1_MIN_MAGNITUDE = 0.01


HOMOGRAPHY_REFINE_MAX_ITERATIONS = 15


HOMOGRAPHY_REFINE_MIN_RELATIVE_IMPROVEMENT = 0.03


# Montako PERAKKAISTA ei-parantunutta kierrosta refine_homography_
# corrections sietaa ennen pysahtymista (katso sen sisainen kommentti).
STALL_PATIENCE = 3


# ============================================================
# GEOMETRIA-APURIT
# ============================================================


# Hogline (etulinja) sijaitsee curlingissa 6.4 m paassa T-linjalta
# (pesan keskipisteesta), molemmin puolin kenttaa.
NEAR_HOGLINE_Y_CM = NEAR_HOUSE_Y_CM + HOG_LINE_DISTANCE_FROM_TEE_CM


FAR_HOGLINE_Y_CM = FAR_HOUSE_Y_CM - HOG_LINE_DISTANCE_FROM_TEE_CM


def distance(p1, p2):

    return float(np.linalg.norm(np.asarray(p1) - np.asarray(p2)))


def midpoint(p1, p2):

    return np.array(
        [(p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0],
        dtype=np.float64
    )


def line_from_points(p1, p2):

    x1, y1 = p1
    x2, y2 = p2

    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - x2 * y1

    n = math.hypot(a, b)

    if n < 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)

    return np.array([a / n, b / n, c / n], dtype=np.float64)


def intersect_lines(l1, l2):

    p = np.cross(l1, l2)

    if abs(p[2]) < 1e-12:
        return None

    return np.array([p[0] / p[2], p[1] / p[2]], dtype=np.float64)


def line_angle_deg(p1, p2):

    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    return math.degrees(math.atan2(dy, dx)) % 180.0


def angle_distance_to_horizontal(angle):

    d = abs(angle)
    return min(d, 180.0 - d)


def angle_distance_to_vertical(angle):

    d = abs(angle - 90.0)
    return min(d, 180.0 - d)


def point_line_distance(point, line):

    x, y = point
    a, b, c = line

    return abs(a * x + b * y + c)


def line_ellipse_intersections(line, ellipse):

    (cx, cy), (w, h), angle = ellipse

    a = w / 2.0
    b = h / 2.0

    theta = math.radians(angle)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    A, B, C = line

    direction = np.array([-B, A], dtype=np.float64)
    dn = np.linalg.norm(direction)

    if dn < 1e-12:
        return []

    direction /= dn

    p0 = -C * np.array([A, B], dtype=np.float64)

    q = p0 - np.array([cx, cy], dtype=np.float64)

    qx = q[0] * cos_t + q[1] * sin_t
    qy = -q[0] * sin_t + q[1] * cos_t

    dx = direction[0] * cos_t + direction[1] * sin_t
    dy = -direction[0] * sin_t + direction[1] * cos_t

    aa = (dx / a) ** 2 + (dy / b) ** 2
    bb = 2.0 * (qx * dx / (a * a) + qy * dy / (b * b))
    cc = (qx / a) ** 2 + (qy / b) ** 2 - 1.0

    disc = bb * bb - 4.0 * aa * cc

    if disc < 0:
        return []

    sqrt_disc = math.sqrt(max(0.0, disc))

    if abs(disc) < 1e-10:
        t = -bb / (2.0 * aa)
        return [p0 + direction * t]

    t1 = (-bb - sqrt_disc) / (2.0 * aa)
    t2 = (-bb + sqrt_disc) / (2.0 * aa)

    return [p0 + direction * t1, p0 + direction * t2]


def create_blue_mask(frame):

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([90, 60, 40], dtype=np.uint8)
    upper_blue = np.array([140, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def create_red_mask(frame):

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red_1 = np.array([0, 60, 40], dtype=np.uint8)
    upper_red_1 = np.array([10, 255, 255], dtype=np.uint8)

    lower_red_2 = np.array([170, 60, 40], dtype=np.uint8)
    upper_red_2 = np.array([180, 255, 255], dtype=np.uint8)

    mask_1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
    mask_2 = cv2.inRange(hsv, lower_red_2, upper_red_2)

    mask = cv2.bitwise_or(mask_1, mask_2)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def create_topdown_score_maps(view):
    """
    Sinisyys-/punaisuuspisteytyskartat top-down-nakymille (liukuluku,
    ei typistetty 8-bittiseksi - reunanetsinta tarvitsee jatkuvan
    arvoalueen osapikselitarkkuutta varten).

    Nama EIVAT korvaa eivatka muuta create_blue_mask / create_red_mask
    -funktioita, joita kaytetaan alkuperaisessa (perspektiivikuvan)
    tunnistuksessa - ne pysyvat ennallaan.

        blueness = B - 0.5*(R+G)
        redness  = R - 0.5*(B+G)
    """

    smoothed = cv2.GaussianBlur(view, (9, 9), 0)

    b = smoothed[:, :, 0].astype(np.float32)
    g = smoothed[:, :, 1].astype(np.float32)
    r = smoothed[:, :, 2].astype(np.float32)

    blue_score = b - 0.5 * (r + g)
    red_score = r - 0.5 * (b + g)

    return blue_score, red_score


def _bilinear_sample(score, xs, ys):
    """
    Naytteistaa harmaasavykartan (float) annetuista (xs, ys) -pisteista
    bilineaarisella interpoloinnilla (osapikselitarkkuus).
    """

    h, w = score.shape[:2]

    xs = np.clip(xs, 0.0, w - 1.001)
    ys = np.clip(ys, 0.0, h - 1.001)

    x0 = np.floor(xs).astype(int)
    x1 = x0 + 1
    y0 = np.floor(ys).astype(int)
    y1 = y0 + 1

    fx = xs - x0
    fy = ys - y0

    v00 = score[y0, x0]
    v01 = score[y0, x1]
    v10 = score[y1, x0]
    v11 = score[y1, x1]

    return (
        v00 * (1 - fx) * (1 - fy) + v01 * fx * (1 - fy) +
        v10 * (1 - fx) * fy + v11 * fx * fy
    )


def _smooth_reflect(values, kernel_size=5):
    """
    Liukuva keskiarvo peiliheijastus-reunaehdolla (EI nollapehmennysta).
    Tavallinen np.convolve(..., mode="same") pehmentaa taulukon paihin
    implisiittisesti nollilla, mika synnyttaa keinotekoisen (virheellisen)
    voimakkaan "reunan" aivan sateen alku-/loppupaahan - taman takia
    reunapisteet suodatetaan pois erikseen margin-parametrilla lisaksi.
    """

    pad = kernel_size // 2
    padded = np.pad(values, (pad, pad), mode="reflect")
    kernel = np.ones(kernel_size) / kernel_size

    return np.convolve(padded, kernel, mode="valid")


def _uniform_gradient(y, h):
    """
    Sama kuin np.gradient(y, x) kun x:n valit ovat TASAVALISIA (askel
    h) - keskeisdifferenssi sisapisteille, yksipuolinen differenssi
    paissa. np.gradient tukee myos epatasavalisia valeja, mika tekee
    siita hitaamman (yleisempi tarkistus-/haarautumislogiikka) - tassa
    kaytetty tapaus (find_ring_edge_points:in radii_v) on AINA
    tasavalinen (np.arange(r_lo, r_hi, r_step)), joten tama antaa
    BITTITARKASTI saman tuloksen mutta nopeammin.
    """

    grad = np.empty_like(y)

    if len(y) < 2:
        grad[:] = 0.0
        return grad

    grad[1:-1] = (y[2:] - y[:-2]) / (2.0 * h)
    grad[0] = (y[1] - y[0]) / h
    grad[-1] = (y[-1] - y[-2]) / h

    return grad


def find_ring_edge_points(
    score, expected_center, expected_radius_px,
    radius_tol=0.25, num_angles=720, r_step=0.5, edge_margin_frac=0.08
):
    """
    Etsii pesan renkaan reunan OSAPIKSELITARKKUUDELLA sailtamalla
    sateittain (num_angles suuntaa expected_center:sta) sinisyys-/
    punaisuuspisteytyksen [expected_radius*(1-tol), expected_radius*(1+tol)]
    valilta, ja poimimalla kultakin sateelta radius, jossa pisteytyksen
    gradientti on voimakkain (= varin reuna).

    Taman jalkeen reunapisteista sovitetaan koko ympyra (katso
    robust_circle_fit), joten yksittaisten sateiden kohina (esim.
    kolmiokuvion sahalaita) keskiarvoistuu pois - tulos on paljon
    tarkempi kuin suora kontuuri- tai Hough-tunnistus.
    """

    cx0, cy0 = expected_center

    angles = np.linspace(0.0, 2.0 * math.pi, num_angles, endpoint=False)

    r_lo = expected_radius_px * (1.0 - radius_tol)
    r_hi = expected_radius_px * (1.0 + radius_tol)
    radii = np.arange(r_lo, r_hi, r_step)

    h, w = score.shape[:2]

    # NOPEUSOPTIMOINTI (kamera7_02.py): alkuperainen versio kutsui
    # _bilinear_sample:a ERIKSEEN JOKAISELLE sateelle (720 kertaa) -
    # tama oli valtaosa funktion ajasta (720 pientä numpy-kutsujoukkoa
    # per find_ring_edge_points-kutsu). Lasketaan sen sijaan KAIKKIEN
    # sateiden (x,y)-koordinaatit ja niiden pisteytysarvot YHDELLA
    # vektoroidulla kutsulla (num_angles x num_radii -ruudukko), ja
    # tehdaan vain sateen sisainen pehmennys/gradientti/huippukohta
    # -laskenta sateittain (halpa - ei ena sisalla kallista naytteis-
    # tysta). "valid"-alue on jokaisella sateella YHTENAINEN VALI joka
    # alkaa r_lo:sta (koska sateen suunta on kiintea, kuvan reunaehdot
    # ovat monotonisia sateen sailtaman r:n suhteen), joten se voidaan
    # koota rivikohtaisen validien maaran (n_valid) avulla TAYSIN
    # samalla logiikalla kuin alkuperainen versio - validoitu antavan
    # BITTITARKASTI saman tuloksen molemmilla testikuvilla.
    dx_all = np.cos(angles)
    dy_all = np.sin(angles)

    xs_grid = cx0 + np.outer(dx_all, radii)
    ys_grid = cy0 + np.outer(dy_all, radii)

    valid_grid = (
        (xs_grid >= 0) & (xs_grid < w - 1) &
        (ys_grid >= 0) & (ys_grid < h - 1)
    )
    n_valid = valid_grid.sum(axis=1)

    vals_grid = _bilinear_sample(score, xs_grid, ys_grid)

    points = []

    for i in range(num_angles):

        n = int(n_valid[i])

        if n < 20:
            continue

        dx, dy = dx_all[i], dy_all[i]
        radii_v = radii[:n]

        vals = vals_grid[i, :n]
        vals_smooth = _smooth_reflect(vals, kernel_size=5)

        # NOPEUSOPTIMOINTI: radii_v:n valit ovat AINA tasavalisia
        # (r_step, koska radii = np.arange(r_lo, r_hi, r_step)), joten
        # _uniform_gradient antaa BITTITARKASTI saman tuloksen kuin
        # np.gradient(vals_smooth, radii_v) mutta ilman sen yleisemman
        # (epatasavalisen) tapauksen tarkistus-/haarautumisoverheadia.
        grad = _uniform_gradient(vals_smooth, r_step)

        margin = max(3, int(n * edge_margin_frac))

        if n - 2 * margin < 5:
            continue

        interior_grad = grad[margin:n - margin]
        local_idx = int(np.argmax(np.abs(interior_grad)))
        idx = margin + local_idx

        r_edge = radii_v[idx]
        strength = abs(grad[idx])

        points.append((cx0 + r_edge * dx, cy0 + r_edge * dy, strength))

    return points


def _fit_circle_kasa(points):
    """
    Algebrallinen (Kasa) ympyransovitus pistejoukkoon: minimoi
    sum (x^2+y^2 + D*x + E*y + F)^2 - suljetun muodon lineaarinen
    pienimman nelion sovitus, nopea ja vakaa isolle pistemaaralle.
    """

    pts = np.asarray(points, dtype=np.float64)

    x = pts[:, 0]
    y = pts[:, 1]

    A = np.column_stack([x, y, np.ones_like(x)])
    b = -(x ** 2 + y ** 2)

    (D, E, F), *_ = np.linalg.lstsq(A, b, rcond=None)

    cx = -D / 2.0
    cy = -E / 2.0
    r = math.sqrt(max(cx * cx + cy * cy - F, 1e-6))

    return cx, cy, r


def robust_circle_fit_with_inliers(points, keep_frac=0.7, iterations=4, min_points=20):
    """
    Sama kuin robust_circle_fit, mutta palauttaa MYOS lopulliset
    (poikkeavien hylkayksen jalkeiset) sisapisteet - kayttokelpoinen
    kun niita halutaan kayttaa edelleen esim. YHTEISEN keskipisteen
    sovitukseen useamman renkaan kesken (katso
    fit_concentric_circles_shared_center).

    Palauttaa (cx, cy, r, inlier_points) tai None.
    """

    if len(points) < min_points:
        return None

    pts = [(p[0], p[1]) for p in points]
    strengths = np.array([p[2] for p in points])

    order = np.argsort(strengths)[::-1]
    n_keep = max(min_points, int(len(pts) * keep_frac))

    selected = [pts[i] for i in order[:n_keep]]

    cx, cy, r = _fit_circle_kasa(selected)

    for _ in range(iterations):

        dists = np.array([
            abs(math.hypot(px - cx, py - cy) - r) for px, py in selected
        ])

        median = np.median(dists)
        mad = np.median(np.abs(dists - median)) + 1e-6

        keep = dists < median + 3.0 * mad

        if keep.sum() < min_points:
            break

        selected = [selected[i] for i in range(len(selected)) if keep[i]]
        cx, cy, r = _fit_circle_kasa(selected)

    return cx, cy, r, np.array(selected, dtype=np.float64)


def _select_top_strength_points(points, keep_frac=0.7, min_points=20):
    """
    Palauttaa find_ring_edge_points:in (x,y,edge_strength) -pisteista
    Nx2-taulukon, jossa on sailytetty vain voimakkaimmat (keep_frac)
    reunapisteet (heikoimmat ovat todennakoisimmin kohinaa/vaaraa
    reunaa). Palauttaa None jos pisteita on liian vahan.
    """

    if len(points) < min_points:
        return None

    pts = np.array([(p[0], p[1]) for p in points], dtype=np.float64)
    strengths = np.array([p[2] for p in points])

    order = np.argsort(strengths)[::-1]
    n_keep = max(min_points, int(len(pts) * keep_frac))

    return pts[order[:n_keep]]


def _solve_shared_ellipse_shape(point_groups, theta, k):
    """
    ANNETULLA (kiinnitetylla) kiertokulmalla theta ja akselisuhteella
    k (= sivuakseli/paaakseli, 0 < k <= 1) ratkaisee LINEAARISESTI
    (algebrallinen Kasa-tyylinen konikkisovitus) KAIKKIEN ryhmien
    (esim. sinisen ja punaisen ulkokehan) YHTEISEN keskipisteen
    (cx, cy) seka kunkin ryhman OMAN paaakselin pituuden a_i.

    PERUSTELU (katso fit_shared_ellipse_shape): pesan renkaat ovat
    fyysisesti samankeskisia JA saman (paikallisen affiinin/projek-
    tiivisen kuvauksen aiheuttaman) vaaristyman alaisia, joten niilla
    on sama todellinen keskipiste, sama ellipsin kiertokulma JA sama
    akselisuhde - vain koko (sade) vaihtelee renkaittain. Kun theta ja
    k kiinnitetaan, jaljella oleva ongelma (cx, cy, a_i) on ellipsin
    yhtalossa AIDOSTI LINEAARINEN (kuten Kasa-ympyransovitus), joten
    se voidaan ratkaista suljetussa muodossa ilman paikallisminimien
    riskia - grid_search_shared_ellipse_shape hakee parhaan (theta,k)
    -parin kokeilemalla useita ja vertaamalla GEOMETRISTA jaannosta.

    Ellipsin yhtalo pisteelle (x,y), keskipisteessa (cx,cy) kierretyssa
    (theta) koordinaatistossa (u,v):
        u^2 + (v/k)^2 = a^2
    Kirjoitettuna alkuperaisiin koordinaatteihin (dx=x-cx, dy=y-cy):
        P*dx^2 + Q*dy^2 + 2*R*dx*dy = a^2 * k^2
    missa P = k^2*cos^2(theta) + sin^2(theta), Q = k^2*sin^2(theta) +
    cos^2(theta), R = cos(theta)*sin(theta)*(k^2 - 1). Talla on cx:n,
    cy:n ja (a_i*k)^2:n suhteen lineaarinen muoto (vrt. Kasa).

    Palauttaa (cx, cy, [a_i per ryhma]).
    """

    ct, st = math.cos(theta), math.sin(theta)
    p_coef = k * k * ct * ct + st * st
    q_coef = k * k * st * st + ct * ct
    r_coef = ct * st * (k * k - 1.0)

    n_groups = len(point_groups)
    rows = []
    rhs = []

    for gi, pts in enumerate(point_groups):

        x = pts[:, 0]
        y = pts[:, 1]

        lhs = p_coef * x ** 2 + q_coef * y ** 2 + 2.0 * r_coef * x * y
        col_cx = -2.0 * (p_coef * x + r_coef * y)
        col_cy = -2.0 * (q_coef * y + r_coef * x)

        block = np.zeros((len(x), 2 + n_groups))
        block[:, 0] = col_cx
        block[:, 1] = col_cy
        block[:, 2 + gi] = 1.0

        rows.append(block)
        rhs.append(-lhs)

    A = np.vstack(rows)
    b = np.concatenate(rhs)

    sol, *_ = np.linalg.lstsq(A, b, rcond=None)

    cx, cy = float(sol[0]), float(sol[1])
    consts = sol[2:]

    a_list = []

    for gi in range(n_groups):
        t_sq = p_coef * cx * cx + q_coef * cy * cy + 2.0 * r_coef * cx * cy - consts[gi]
        a_list.append(math.sqrt(max(t_sq, 1e-6)) / k)

    return cx, cy, a_list


def _shared_ellipse_geometric_mse(point_groups, cx, cy, theta, k, a_list):
    """
    Laskee TODELLISEN (geometrisen - etaisyys pisteesta ellipsin
    reunaan sateen suunnassa) jaannosneliokeskiarvon annetulla
    (cx,cy,theta,k,a_i) -parametrisoinnilla. Kayttokelpoinen ERI
    (theta,k) -ehdokkaiden VERTAILUUN keskenaan grid-haussa - pelkka
    _solve_shared_ellipse_shape:in palauttama algebrallinen jaannos
    EI ole suoraan vertailukelpoinen eri k:n arvojen valilla (yhtalo
    on skaalattu eri tavalla).
    """

    ct, st = math.cos(theta), math.sin(theta)
    total = 0.0
    n = 0

    for pts, a in zip(point_groups, a_list):

        dx = pts[:, 0] - cx
        dy = pts[:, 1] - cy
        u = dx * ct + dy * st
        v = -dx * st + dy * ct
        rho = np.sqrt(u * u + (v / k) ** 2)

        total += float(np.sum((rho - a) ** 2))
        n += len(pts)

    return total / max(n, 1)


def _grid_search_shared_ellipse_shape(
    point_groups, theta_center_deg, theta_half_range_deg, theta_step_deg,
    k_center, k_half_range, k_step
):
    """
    Hakee parhaan (theta, k) -parin (pienin geometrinen jaannos) kaikkien
    ryhmien pisteille annetulta grid-alueelta - sama coarse/fine-
    periaate kuin esim. search_far_house:ssa tai k1-itsekalibroinnissa
    (katso estimate_radial_distortion_k1), koska (theta,k) -ongelma EI
    ole lineaarinen (toisin kuin cx,cy,a_i kiinnitetylla theta,k:lla).

    Palauttaa (mse, theta_deg, k, cx, cy, a_list).
    """

    thetas = make_range(theta_center_deg, theta_half_range_deg, theta_step_deg)
    ks = make_range(k_center, k_half_range, k_step)
    ks = ks[(ks > 0.05) & (ks <= 1.0)]

    best = None

    for theta_deg in thetas:

        theta = math.radians(float(theta_deg))

        for k in ks:

            k = float(k)
            cx, cy, a_list = _solve_shared_ellipse_shape(point_groups, theta, k)
            mse = _shared_ellipse_geometric_mse(point_groups, cx, cy, theta, k, a_list)

            if best is None or mse < best[0]:
                best = (mse, float(theta_deg), k, cx, cy, a_list)

    return best


def fit_shared_ellipse_shape(point_groups, iterations=3, mad_multiplier=3.0, min_points=20):
    """
    Sovittaa USEALLE pisteryhmalle (esim. kaukaisen pesan sinisen ja
    punaisen ULKOKEHAN reunapisteet) YHTEISEN ellipsin MUODON: sama
    keskipiste (cx,cy), sama kiertokulma (theta) JA sama akselisuhde
    (k = sivuakseli/paaakseli) - vain kunkin renkaan oma sade (a_i)
    saa vaihdella.

    PERUSTELU: pesan renkaat ovat FYYSISESTI samankeskisia, ja koska
    pesan halkaisija (3.66 m) on hyvin pieni verrattuna kameran
    etaisyyteen, paikallinen projektiivinen kuvaus pesan alueella on
    lahes AFFIINI - affiini kuvaus vie KAIKKI samankeskiset ympyrat
    (sateesta riippumatta) samankeskisiksi, SAMAN SUUNTAISIKSI ja
    SAMAN MUOTOISIKSI (sama akselisuhde) ellipseiksi. Pakottamalla
    seka keskipiste ETTA muoto (kiertokulma+akselisuhde) jaettavaksi
    KAIKKIEN renkaiden ~700+700 reunapisteen kesken (vain 6 vapaus-
    astetta: cx,cy,theta,k,a_blue,a_red - itsenaisilla ellipsisovi-
    tuksilla olisi 2*5=10) kohina keskiarvoistuu POIS huomattavasti
    tehokkaammin kuin itsenaisilla ellipsisovituksilla (jotka ovat
    lisaksi tunnetusti alttiita "eksentrisyysharhalle" kohinaisella
    kohteella, katso robust_ellipse_fit:in kommentti).

    Havaittu testatessa: kaukaisen pesan itsenainen (vapaa) ellipsi-
    sovitus antoi pyoreyden ~0.80, mutta tama yhteismuotoinen sovitus
    ~0.96-0.99 SAMOISTA reunapisteista - molemmilla testikuvilla.

    Ratkaistaan grid-haulla (coarse -> fine, katso
    _grid_search_shared_ellipse_shape) + iteratiivinen poikkeavien
    pisteiden hylkays (mediaani + 3*MAD geometrisesta jaannoksesta).

    Palauttaa (cx, cy, theta_deg, k, [a_i per ryhma]) tai None jos
    yhdellakaan ryhmalla ei ole tarpeeksi pisteita.
    """

    groups = [g for g in point_groups if g is not None and len(g) >= min_points]

    if not groups:
        return None

    mse, theta_deg, k, cx, cy, a_list = _grid_search_shared_ellipse_shape(
        groups, 90.0, 90.0, 2.0, 0.75, 0.25, 0.02
    )
    mse, theta_deg, k, cx, cy, a_list = _grid_search_shared_ellipse_shape(
        groups, theta_deg, 3.0, 0.1, k, 0.03, 0.002
    )

    for _ in range(iterations):

        theta = math.radians(theta_deg)
        ct, st = math.cos(theta), math.sin(theta)
        trimmed = []

        for pts, a in zip(groups, a_list):

            dx = pts[:, 0] - cx
            dy = pts[:, 1] - cy
            u = dx * ct + dy * st
            v = -dx * st + dy * ct
            rho = np.sqrt(u * u + (v / k) ** 2)
            resid = np.abs(rho - a)

            median = np.median(resid)
            mad = np.median(np.abs(resid - median)) + 1e-6
            keep = resid < median + mad_multiplier * mad

            trimmed.append(pts[keep] if keep.sum() >= min_points else pts)

        groups = trimmed

        mse, theta_deg, k, cx, cy, a_list = _grid_search_shared_ellipse_shape(
            groups, theta_deg, 2.0, 0.1, k, 0.02, 0.002
        )

    return cx, cy, theta_deg, k, a_list


def detect_far_house_shared_ellipses(view, expected_center):
    """
    Tunnistaa kaukaisen pesan sinisen ja punaisen ULKOKEHAN YHTEISELLA
    (pakotetulla) ellipsin MUODOLLA - katso fit_shared_ellipse_shape:in
    perustelu. Toisin kuin aiempi pakotettu YMPYRA (joka olettaa jo
    korjatun kuvan olevan taydellisen pyorea), tama sallii jaljella
    olevan lievan soikeuden nayttaytya OIKEIN molemmissa renkaissa
    samalla tavalla - mikä on tarkempi silloin kun homografia ei viela
    ole talydellinen.

    Palauttaa dictin jossa "blue_outer"/"red_outer" (ellipsi-muodossa)
    tai None jos ei loytynyt.
    """

    blue_score, red_score = create_topdown_score_maps(view)

    result = {"blue_outer": None, "blue_inner": None, "red_outer": None, "red_inner": None}

    blue_points = find_ring_edge_points(
        blue_score, expected_center, BLUE_OUTER_RADIUS_CM * PIXELS_PER_CM
    )
    red_points = find_ring_edge_points(
        red_score, expected_center, RED_OUTER_RADIUS_CM * PIXELS_PER_CM
    )

    blue_sel = _select_top_strength_points(blue_points)
    red_sel = _select_top_strength_points(red_points)

    ring_names = [
        name for name, sel in (("blue_outer", blue_sel), ("red_outer", red_sel))
        if sel is not None
    ]
    groups = [sel for sel in (blue_sel, red_sel) if sel is not None]

    fit = fit_shared_ellipse_shape(groups)

    if fit is None:
        return result

    cx, cy, theta_deg, k, a_list = fit

    for name, a in zip(ring_names, a_list):
        width = 2.0 * a
        height = 2.0 * a * k
        result[name] = ((cx, cy), (width, height), theta_deg)

    return result


def find_house_ellipses(mask, min_area=1000, min_ratio=0.20, min_contour_len=20):

    contours, _ = cv2.findContours(
        mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
    )

    candidates = []

    for contour in contours:

        if len(contour) < min_contour_len:
            continue

        area = cv2.contourArea(contour)

        if area < min_area:
            continue

        ellipse = cv2.fitEllipse(contour)
        (cx, cy), (w, h), angle = ellipse

        if w <= 0 or h <= 0:
            continue

        ratio = min(w, h) / max(w, h)

        if ratio < min_ratio:
            continue

        candidates.append({
            "ellipse": ellipse,
            "area": area,
            "center": np.array([cx, cy], dtype=np.float64),
        })

    candidates.sort(key=lambda x: x["area"], reverse=True)

    return candidates


def find_nested_ellipse(ellipses, outer_ellipse, min_size_ratio=0.30):

    outer_center = np.array(outer_ellipse[0], dtype=np.float64)
    outer_w = outer_ellipse[1][0]
    outer_h = outer_ellipse[1][1]

    max_center_distance = max(100.0, min(outer_w, outer_h) * 0.25)

    candidates = []

    for candidate in ellipses:

        ellipse = candidate["ellipse"]

        if ellipse is outer_ellipse:
            continue

        center = np.array(ellipse[0], dtype=np.float64)
        center_distance = np.linalg.norm(center - outer_center)

        if center_distance > max_center_distance:
            continue

        w = ellipse[1][0]
        h = ellipse[1][1]

        if w >= outer_w or h >= outer_h:
            continue

        if w / outer_w < min_size_ratio:
            continue

        if h / outer_h < min_size_ratio:
            continue

        theta = math.radians(outer_ellipse[2])

        dx = center[0] - outer_center[0]
        dy = center[1] - outer_center[1]

        local_x = dx * math.cos(theta) + dy * math.sin(theta)
        local_y = -dx * math.sin(theta) + dy * math.cos(theta)

        normalized_distance = (
            (local_x / (outer_w / 2.0)) ** 2 +
            (local_y / (outer_h / 2.0)) ** 2
        )

        if normalized_distance > 1.0:
            continue

        candidates.append({"ellipse": ellipse, "area": candidate["area"]})

    if not candidates:
        return None

    candidates.sort(key=lambda x: x["area"], reverse=True)

    return candidates[0]["ellipse"]


def find_house_pair(mask, min_area, min_ratio, min_size_ratio, min_contour_len=20):
    """
    Yleinen apufunktio: etsii maskista suurimman (ulko-) ellipsin
    ja siihen sisakkain olevan (sisa-) ellipsin. Kaytetaan seka
    sinisen etta punaisen, seka lahemman etta kaukaisen pesan
    (hue-maskien) kanssa.
    """

    ellipses = find_house_ellipses(
        mask, min_area=min_area, min_ratio=min_ratio,
        min_contour_len=min_contour_len
    )

    if not ellipses:
        return None, None

    outer = ellipses[0]["ellipse"]
    inner = find_nested_ellipse(ellipses, outer, min_size_ratio=min_size_ratio)

    return outer, inner


# ============================================================
# JAA-SUODATUS-POHJAINEN LISAREUNALAHDE (Testi_02_02, kayttajan
# pyynnosta - katso keskusteluhistoria): joissakin videoissa (esim.
# MAH00014, jossa lahempi pesa on aivan kuvan reunassa) hogline/
# keskiviiva on niin haalea (matala kontrasti jaata vasten) etta
# tavallinen harmaasavy-Canny (alla) ei loyda sita luotettavasti -
# se hukkuu kirkkaan jaan omaan kohinaan. Rata-alueen S/V-histogrammin
# (Testi_02_01:ssa kehitetty jaa-suodatus, main.py:n ICE_S_MAX/ICE_
# V_MIN) HAVAITTIIN paljastavan tallaisen haalean viivan selvasti
# PAREMMIN kuin Canny: jaa on tasaisen kirkasta JA matalasaturoitunutta
# (S<LINE_ICE_S_MAX, V>LINE_ICE_V_MIN), joten kaikki mika EI tayta tata
# (mukaan lukien haalea harmaa viiva, JOKA ON hieman jaata tummempi)
# erottuu binaarimaskina huomattavasti kontrastikkaammin kuin raa'an
# harmaasavykuvan gradientti. Tama LISATAAN (ei korvata) alkuperaisen
# Cannyn rinnalle - molempien lahteiden segmentit yhdistetaan, joten
# jo ennestaan toimiva videon (0001) tulos ei voi huonontua (samat
# vanhat segmentit ovat yha mukana, uudet vain lisaavat kandidaatteja).
# ============================================================
LINE_ICE_S_MAX = 22


LINE_ICE_V_MIN = 165


def _detect_line_segments_from_edges(edges, roi_mask):

    edges = cv2.bitwise_and(edges, roi_mask)

    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180.0,
        threshold=50, minLineLength=80, maxLineGap=25
    )

    segments = []

    if lines is None:
        return segments

    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):

        p1 = np.array([x1, y1], dtype=np.float64)
        p2 = np.array([x2, y2], dtype=np.float64)

        length = distance(p1, p2)

        if length < 60:
            continue

        angle = line_angle_deg(p1, p2)

        segments.append({
            "p1": p1, "p2": p2, "length": length,
            "angle": angle, "mid": midpoint(p1, p2)
        })

    return segments


def detect_line_segments(frame, ellipse):

    h, w = frame.shape[:2]

    (cx, cy), (ew, eh), _ = ellipse

    margin_x = int(max(250, ew * 1.0))
    margin_y = int(max(300, eh * 0.8))

    x1 = max(0, int(cx - margin_x))
    x2 = min(w - 1, int(cx + margin_x))
    y1 = max(0, int(cy - margin_y))
    y2 = min(h - 1, int(cy + margin_y))

    roi_mask = np.zeros((h, w), dtype=np.uint8)
    roi_mask[y1:y2 + 1, x1:x2 + 1] = 255

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_edges = cv2.Canny(gray, 50, 150)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.int16)
    non_ice = (
        (hsv[..., 1] >= LINE_ICE_S_MAX) | (hsv[..., 2] <= LINE_ICE_V_MIN)
    ).astype(np.uint8) * 255
    ice_edges = cv2.Canny(non_ice, 50, 150)

    segments = _detect_line_segments_from_edges(gray_edges, roi_mask)
    segments += _detect_line_segments_from_edges(ice_edges, roi_mask)

    return segments


def pair_line_segments(segments, ellipse, orientation, center):

    if orientation == "vertical":
        candidates = [
            s for s in segments
            if angle_distance_to_vertical(s["angle"]) <= 35.0
        ]
    else:
        candidates = [
            s for s in segments
            if angle_distance_to_horizontal(s["angle"]) <= 35.0
        ]

    best_pair = None
    best_score = float("inf")

    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):

            s1 = candidates[i]
            s2 = candidates[j]

            angle_diff = abs(s1["angle"] - s2["angle"])
            angle_diff = min(angle_diff, 180.0 - angle_diff)

            if angle_diff > 12.0:
                continue

            line1 = line_from_points(s1["p1"], s1["p2"])
            line_distance = point_line_distance(s2["mid"], line1)

            if line_distance > 80:
                continue

            v1 = s1["mid"] - center
            v2 = s2["mid"] - center

            if np.dot(v1, v2) >= 0:
                continue

            combined = line_from_points(s1["p1"], s1["p2"])
            center_distance = point_line_distance(center, combined)

            if center_distance > 80:
                continue

            intersections = line_ellipse_intersections(combined, ellipse)

            if len(intersections) != 2:
                continue

            score = (
                line_distance * 5.0 +
                center_distance * 3.0 +
                angle_diff * 2.0 -
                min(s1["length"], s2["length"]) * 0.01
            )

            if score < best_score:
                best_score = score
                best_pair = (s1, s2, combined)

    return best_pair


def _single_sided_line(segments, ellipse, orientation, center):
    """VARAKEINO (Testi_02_02, kayttajan pyynnosta - katso keskustelu-
    historia): pair_line_segments vaatii AINA kaksi segmenttia, yhden
    keskipisteen KUMMALTAKIN puolelta - tama toimii vain jos lahempi
    pesa on kuvassa niin etta rataa nakyy molemmin puolin sita. Jos
    pesa on aivan kuvan reunassa (esim. MAH00014, kamera kuvaa rataa
    sivulta), toinen puoli ei koskaan nay eika paria loydy vaikka
    varsinainen viiva olisi selvasti nakyvissa toisella puolella.
    Kaytetaan silloin PARASTA YKSITTAISTA riittavan pitkaa segmenttia
    joka kulkee lahella keskipistetta - se maarittaa itsessaan koko
    suoran (combined lasketaan AINA vain yhden segmentin p1/p2:sta,
    katso pair_line_segments), joten toista puolta ei oikeasti
    tarvita geometrista suoraa varten - VAIN build_image_directions:in
    suuntavektorille (keskipisteen KAUTTA peilattu synteettinen
    "toinen puoli", jotta sen "mid" antaa oikean suunnan)."""

    if orientation == "vertical":
        candidates = [
            s for s in segments
            if angle_distance_to_vertical(s["angle"]) <= 35.0
        ]
    else:
        candidates = [
            s for s in segments
            if angle_distance_to_horizontal(s["angle"]) <= 35.0
        ]

    best = None
    best_score = float("inf")

    for s in candidates:

        combined = line_from_points(s["p1"], s["p2"])
        center_distance = point_line_distance(center, combined)

        if center_distance > 80:
            continue

        intersections = line_ellipse_intersections(combined, ellipse)

        if len(intersections) != 2:
            continue

        score = center_distance * 3.0 - s["length"] * 0.01

        if score < best_score:
            best_score = score
            best = (s, combined)

    if best is None:
        return None

    s1, combined = best
    mirrored_mid = 2.0 * center - s1["mid"]
    s2 = {
        "p1": s1["p1"], "p2": s1["p2"],
        "length": s1["length"], "angle": s1["angle"],
        "mid": mirrored_mid,
    }

    return (s1, s2, combined)


def select_t_and_centerline(segments, ellipse):

    center = np.array(ellipse[0], dtype=np.float64)

    t_pair = pair_line_segments(segments, ellipse, "vertical", center)
    centerline_pair = pair_line_segments(segments, ellipse, "horizontal", center)

    if t_pair is None:
        t_pair = _single_sided_line(segments, ellipse, "vertical", center)

    if centerline_pair is None:
        centerline_pair = _single_sided_line(segments, ellipse, "horizontal", center)

    if t_pair is None or centerline_pair is None:
        raise RuntimeError("T-linjaa tai keskilinjaa ei loytynyt.")

    t_line = t_pair[2]
    centerline = centerline_pair[2]

    house_center = intersect_lines(t_line, centerline)

    if house_center is None:
        raise RuntimeError("T-linja ja keskilinja ovat yhdensuuntaiset.")

    return t_pair, centerline_pair, t_line, centerline, house_center


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


def project_point(X, Y, camera, v_t, v_cl):

    f = camera["f"]
    theta = camera["theta"]
    camera_height = camera["camera_height"]
    center = camera["center"]

    relative_Y = Y - NEAR_HOUSE_Y_CM

    near_depth = NEAR_HOUSE_Y_CM * math.sin(theta) + camera_height
    Zc = near_depth + relative_Y * math.sin(theta)

    if Zc <= 1e-8:
        return None

    local_u = f * X / Zc
    local_v = f * relative_Y * math.cos(theta) / Zc

    return center + local_u * v_t + local_v * v_cl


def make_range(center, half_range, step):

    start = center - half_range
    stop = center + half_range

    count = int(round((stop - start) / step))

    values = start + np.arange(count + 1, dtype=np.float64) * step

    if values[-1] < stop - 1e-8:
        values = np.append(values, stop)

    return values


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


def to_output_px(x_cm, y_cm):

    px = (x_cm - OUTPUT_X_MIN_CM) * PIXELS_PER_CM
    py = (OUTPUT_Y_MAX_CM - y_cm) * PIXELS_PER_CM

    return px, py


def physical_to_output_px(physical_pts):

    pts = np.asarray(physical_pts, dtype=np.float64)

    ox = (pts[:, 0] - OUTPUT_X_MIN_CM) * PIXELS_PER_CM
    oy = (OUTPUT_Y_MAX_CM - pts[:, 1]) * PIXELS_PER_CM

    return np.column_stack([ox, oy])


def output_px_to_physical(output_pts):
    """
    physical_to_output_px:n kaanteisfunktio - palauttaa output-px-
    koordinaatiston pisteet takaisin fyysisina (cm) koordinaatteina.
    Kayttokelpoinen kun halutaan kayttaa top-down-kuvasta (output-px)
    jo tunnettuja pisteita uudelleen jonkin muun laskennan (esim.
    objektiivin vaaristyman uudelleenarviointi) syotteena, joka
    odottaa fyysisia (cm) kohdekoordinaatteja - katso
    refine_lens_distortion_k1.
    """

    pts = np.asarray(output_pts, dtype=np.float64)

    x_cm = pts[:, 0] / PIXELS_PER_CM + OUTPUT_X_MIN_CM
    y_cm = OUTPUT_Y_MAX_CM - pts[:, 1] / PIXELS_PER_CM

    return np.column_stack([x_cm, y_cm])


def build_camera_matrix(image_width, image_height):
    """
    Karkea kameramatriisi (sama polttovali-arvaus kuin muualla
    koodissa: f = max(leveys, korkeus), paapiste kuvan keskella).
    Tarkkaa polttovalia ei tarvita - matriisi toimii tassa vain
    skaalaustekijana k1:n normalisoinnille ja cv2:n undistort-
    funktioiden vaatimana muotona.
    """

    f = float(max(image_width, image_height))
    cx = image_width / 2.0
    cy = image_height / 2.0

    return np.array(
        [[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]], dtype=np.float64
    )


def undistort_points_px(points, camera_matrix, k1, k2=0.0):
    """
    Poistaa sateittaisen vaaristyman pistejoukosta ja palauttaa
    pisteet SAMASSA pikselikoordinaatistossa (P=camera_matrix).
    """

    pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    dist_coeffs = np.array([k1, k2, 0.0, 0.0, 0.0], dtype=np.float64)

    undistorted = cv2.undistortPoints(
        pts, camera_matrix, dist_coeffs, P=camera_matrix
    )

    return undistorted.reshape(-1, 2)


def _homography_rms(src_pts, dst_pts):

    # HUOM: method=0 (tavallinen pienimman nelion sovitus), EI RANSAC.
    # Naiden 22 pisteen jakauma on hyvin epatasainen (17 tiiviisti
    # ryhmittynytta lahempaa pistetta + 5 harvaa kaukaista pistetta) -
    # RANSAC:n satunnaisotannalla valittu 4 pisteen minimijoukko osuu
    # ~1/3 kerroista kokonaan lahemman pesan tiiviiseen ryhmaan, mika
    # tuottaa lahes degeneroituneen (numeerisesti epavakaan) homografia-
    # arvion joka sopii noihin 4 pisteeseen mutta ekstrapoloituu
    # hallitsemattomasti kaukaiselle pesalle. Testattu oikealla datalla:
    # RANSAC (millä tahansa kynnysarvolla 3-30 px) antoi toistuvasti
    # tuhansien pikselien virheen, plain LSQ oli aina vakaa ja tarkempi.
    H, _ = cv2.findHomography(
        src_pts.astype(np.float32), dst_pts.astype(np.float32), method=0
    )

    if H is None:
        return float("inf")

    projected = cv2.perspectiveTransform(
        src_pts.reshape(-1, 1, 2).astype(np.float32), H
    ).reshape(-1, 2)

    errors = np.linalg.norm(projected - dst_pts, axis=1)

    return math.sqrt(float(np.mean(errors ** 2)))


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


def compute_crop_row_range(full_height_px, center_y_cm, half_height_cm):
    """
    Laskee crop_house_view:n kayttaman rivialueen (row_top, row_bottom)
    - taman avulla voidaan laskea myos TARKKA odotettu keskipiste
    leikatussa kuvassa, myos silloin kun leikkaus on jouduttu
    rajaamaan topdown-kuvan reunaan (esim. lahempi pesa, jonka
    puolikkaan korkeuden verran ei mahdu topdown-kuvan ala-/ylareunan
    yli).
    """

    _, row_a = to_output_px(0.0, center_y_cm + half_height_cm)
    _, row_b = to_output_px(0.0, center_y_cm - half_height_cm)

    row_top = int(max(0, math.floor(min(row_a, row_b))))
    row_bottom = int(min(full_height_px - 1, math.ceil(max(row_a, row_b))))

    if row_top > row_bottom:
        row_top, row_bottom = row_bottom, row_top

    return row_top, row_bottom


def expected_house_center_in_crop(full_height_px, center_y_cm, half_height_cm):
    """
    Pesan TARKKA odotettu keskipiste leikatun top-down-nakyman
    pikselikoordinaateissa (kayttaa samaa rivilaskentaa kuin
    crop_house_view, jotta reunaan rajautuminen ei siirra oletettua
    keskipistetta vaarin).
    """

    row_top, _ = compute_crop_row_range(full_height_px, center_y_cm, half_height_cm)

    cx, cy_full = to_output_px(0.0, center_y_cm)

    return (cx, cy_full - row_top)


def crop_house_view(topdown, center_y_cm, half_height_cm):

    row_top, row_bottom = compute_crop_row_range(
        topdown.shape[0], center_y_cm, half_height_cm
    )

    return topdown[row_top:row_bottom + 1, :].copy()


def _split_two_line_clusters(points, iterations=10):
    """
    Jakaa pisteet KAHTEEN ryhmaan y-koordinaatin mukaan (yksinkertainen
    1D k-means, k=2) - kayttokelpoinen kun hakukaistassa on KAKSI
    erillista tummaa piirretta (esim. oikea hogline + sponsoriteksti/
    tarra, katso detect_hogline_points:in kommentti), jolloin pisteet
    jakautuvat kahteen selvaan y-tasoon.

    Palauttaa (group_a, group_b) - group_b on tyhja lista jos jako ei
    onnistu jarkevasti (esim. liian vahan pisteita, tai kaikki pisteet
    lankeavat samaan ryhmaan - ei aitoa bimodaalisuutta).
    """

    if len(points) < 4:
        return points, []

    ys = np.array([p[1] for p in points], dtype=np.float64)

    c1, c2 = float(np.percentile(ys, 25)), float(np.percentile(ys, 75))

    if abs(c2 - c1) < 1e-6:
        return points, []

    assign_a = None

    for _ in range(iterations):

        d1 = np.abs(ys - c1)
        d2 = np.abs(ys - c2)
        new_assign_a = d1 <= d2

        if not new_assign_a.any() or new_assign_a.all():
            return points, []

        new_c1 = float(ys[new_assign_a].mean())
        new_c2 = float(ys[~new_assign_a].mean())

        converged = (
            assign_a is not None and np.array_equal(new_assign_a, assign_a)
        )

        assign_a = new_assign_a
        c1, c2 = new_c1, new_c2

        if converged:
            break

    group_a = [points[i] for i in range(len(points)) if assign_a[i]]
    group_b = [points[i] for i in range(len(points)) if not assign_a[i]]

    return group_a, group_b


def _line_value_at_x(points, x_eval):
    """
    Sovittaa painotetun suoran y=a*x+b klusterin KAIKKIIN pisteisiin
    (sama periaate kuin robust_line_angle_from_points) ja palauttaa sen
    arvon EKSTRAPOLOITUNA kohtaan x_eval (esim. keskiviiva, fyysinen
    X=0). Kayttaa koko klusterin dataa, ei vain suppeaa aluetta
    x_eval:in ymparilta - katso _select_best_hogline_cluster:in
    kommentti siita miksi tama on tarkeaa. Palauttaa None jos pisteita
    on liian vahan suoran sovittamiseen.
    """

    if len(points) < 2:
        return None

    xs = np.array([p[0] for p in points], dtype=np.float64)
    ys = np.array([p[1] for p in points], dtype=np.float64)
    ws = np.array([p[2] for p in points], dtype=np.float64)

    A = np.column_stack([xs, np.ones_like(xs)])
    W = np.diag(ws)

    try:
        coeffs, *_ = np.linalg.lstsq(W @ A, W @ ys, rcond=None)
    except np.linalg.LinAlgError:
        return None

    a, b = coeffs

    return float(a * x_eval + b)


def _select_best_hogline_cluster(points, center_x_px, expected_row_px,
                                  max_center_deviation_px=400.0):
    """
    _split_two_line_clusters loytaa hakukaistasta usein KAKSI erillista
    viivaa (todellinen hogline + esim. sponsoriteksti/kylttinauha, katso
    detect_hogline_points:in kommentti) - molemmat voivat olla aidosti
    suoria ja ulottua keskiviivan (fyysinen X=0) yli, joten pelkka
    "onko klusterilla piste/tukea keskiviivan laheisyydessa" EI aina
    erottele niita (todennettu: 00008.png:lla kontaminoiva piirre
    ulottuu lahes koko leveydelta, myos +-30 cm keskiviivaikkunan yli).

    KAYTTAJAN EHDOTTAMA RATKAISU: molemmille klustereille sovitetaan
    OMA suora (_line_value_at_x, KAIKKI kunkin klusterin pisteet
    mukana - ei vain suppea keskiviiva-alue) ja EKSTRAPOLOIDAAN se
    keskiviivalle - se klusteri jonka keskiviivan-ylitys on LAHIMPANA
    tunnettua/nimellista odotettua riviä (expected_row_px, esim.
    FAR_HOGLINE_Y_CM:n mukainen rivi) valitaan todelliseksi hoglineksi.
    Tama kayttaa KOKO klusterin dataa (vakaampi kuin suppea ikkuna) ja
    vastaa suoraan fyysista faktaa: hogline ON tunnetulla etaisyydella
    T-linjasta, joten sen keskiviivan-ylitys on lahella nimellista
    riviä, kun taas kontaminoiva piirre (yleensa eri fyysinen kohde) ei
    yleensa ole.

    HUOM: molemmat klusterit (myos PIENEMPI, esim. vain 4 pistetta)
    OVAT AINA mukana vertailussa - todellinen hogline voi olla
    HARVEMMIN havaittu (himmeampi) kuin kontaminoiva piirre (todennettu:
    Kivilla.png:lla oikea hogline sai vain 4/14 segmentista osuman,
    kontaminaatio 10/14 - pistemaaran enemmisto olisi tallöin valinnut
    vaarin, kuten aiemmin tapahtuikin). Ainoa vaatimus klusterille on
    riittava pistemaara SUORAN sovittamiseksi ylipaataan
    (MIN_CLUSTER_FIT_POINTS), ei suhteellinen enemmisto.

    Jos molemmilta klustereilta loytyy suora, valitaan lahempi; jos
    vain toiselta, kaytetaan sita (jarkevyystarkistettuna); jos EI
    kummaltakaan tai paraskaan ei ole riittavan lahella nimellista
    riviä (max_center_deviation_px), palautetaan tyhja lista - ei
    arvata vaarin.
    """

    MIN_CLUSTER_FIT_POINTS = 3

    group_a, group_b = _split_two_line_clusters(points)
    candidates = [points] if not group_b else [group_a, group_b]

    scored = []

    for grp in candidates:

        if len(grp) < MIN_CLUSTER_FIT_POINTS:
            continue

        y_at_center = _line_value_at_x(grp, center_x_px)

        if y_at_center is None:
            continue

        scored.append((abs(y_at_center - expected_row_px), grp))

    if not scored:
        return []

    best_deviation, best_group = min(scored, key=lambda t: t[0])

    if best_deviation > max_center_deviation_px:
        return []

    return best_group


def detect_hogline_points(
    topdown_raw, expected_y_cm, half_height_cm=400.0,
    edge_margin_frac=0.1, bg_sigma=40.0,
    num_segments=14, min_peak_strength=2.0,
    outlier_iterations=3, outlier_mad_multiplier=3.0, min_points=6,
    max_peaks_per_segment=1, min_peak_separation_px=20
):
    """
    NOPEUSOPTIMOINTI (kamera7_02.py) -saakka hogline-viivalle sovitettiin
    VAIN YKSI kulma koko kaistan leveydelta kerralla (detect_hogline_angle,
    kiertokulman haku joka maksimoi rivien tummuusprofiilin varianssin
    KOKO leveydella yhdella kertaa). Tama osoittautui HERKAKSI paikalli-
    selle hairiolle (esim. sponsoriteksti/logo lahella hoglinea) - yksi
    voimakas paikallinen tummuus voi vetaa koko leveyden kattavan kierto-
    kulmahaun harhaan, vaikka itse hogline olisi lahes vaakasuora
    (havaittu: 00008.png:n kaukainen hogline mittautui +4.2 astetta
    vaikka visuaalinen tarkastelu nayttaa viivan olevan lahes suora -
    kulmahaku oli osunut viereisen tekstin aiheuttamaan tummuuteen).

    Tama funktio (kamera7_04.py) etsii hoglinen PYSTYSIJAINNIN (rivin)
    ERIKSEEN USEASSA x-segmentissa koko leveydelta, ja HYLKAA poikkeavat
    segmentit (mediaani + MAD, katso muualla koodissa kaytetty periaate,
    esim. robust_circle_fit) - yksittainen paikallinen hairio vaikuttaa
    talloin VAIN sen kattamiin 1-2 segmenttiin, ei koko mittaukseen.

    KRIITTINEN BUGIKORJAUS (kamera7_06.py, loytyi diagnosoitaessa miksi
    Kivilla.png:n pesan koko/pyoreys huononi joka kerta kun jotain
    hoglinen kaukaista mittausta hyodyntavaa muutosta kokeiltiin):
    seka Kivilla.png:n etta 00008.png:n kaukaisen hoglinen hakukaistassa
    on TOINEN, erillinen tumma piirre (nayttaa sponsoritekstilta/
    tarralta), ja se on PAIKOIN TUMMEMPI (suurempi "strength") kuin
    itse hogline. Alkuperainen mediaani+MAD-hylkays (koko kaistan
    painotettuun SUORAAN SOVITUKSEEN perustuva) ei erottanut tata
    luotettavasti - todennettu visuaalisesti (kaksi selvasti erillista
    tummaa vaakakaistaa kaukaisen hoglinen hakualueella).

    KOKEILTU JA HYLATTY 1 (pienenna half_height_cm 400->150 + esisuodata
    RAAKAPISTEET joiden etaisyys OLETETUSTA rivista on suuri, ENNEN
    klusterointia): VAARIN, 00008.png:lla TODELLINEN hogline oli itse
    asiassa ~150 cm PAASSA oletetusta (viela hieman epatarkasta)
    rivista, kun taas HAIRIO oli VAIN ~12 cm paassa - siis LAHEMPANA
    oletettua keskustaa kuin itse hogline. Pisteiden ESISUODATUS
    etaisyyden mukaan ei siis ole luotettava tunnusmerkki.

    KOKEILTU JA HYLATTY 2 (poimi useampi huippu/segmentti + valitse
    klusteri jonka suora-sovitus on TIUKEMPI): MYOS VAARIN - sponsori-
    tekstin/tarran tummuushuippu osoittautui usein TIUKEMMIN paikannet-
    tavaksi kuin itse (kauempana, himmeampi, siis rivikohtaisesti
    kohinaisempi) hogline, molemmilla testikuvilla. "Tiukempi sovitus"
    EI ole todellisen hoglinen tunnusmerkki tassa aineistossa.

    NYKYINEN ratkaisu: LEVEA hakukaista (±400 cm, ennallaan) + YKSI
    huippu/segmentti (max_peaks_per_segment=1, kuten alunperin) +
    KAHDEN KLUSTERIN jako y-koordinaatin mukaan (_split_two_line_
    clusters, 1D k-means) + KUMMANKIN KLUSTERIN OMA SUORASOVITUS
    EKSTRAPOLOITUNA KESKIVIIVALLE, VALITEN LAHIMPANA NIMELLISTA
    ODOTETTUA RIVIA OLEVA (kayttajan ehdottama, katso _select_best_
    hogline_cluster:in kommentti - HUOM: tama EI ole sama kuin edella
    hylatty "esisuodata etaisyyden mukaan": tassa KOKO klusterin data
    kaytetaan suoran sovitukseen, ja vasta sen JALKEEN, keskiviivalla
    EKSTRAPOLOITUNA, verrataan etaisyytta - paljon vakaampi kuin
    yksittaisten raakapisteiden esisuodatus) + tavallinen mediaani+MAD-
    hylkays SEN JALKEEN valitun klusterin SISALLA (siivoaa viela
    jaljelle jaavan pienemman kohinan, esim. yksittaisen kaukaisen
    poikkeavan pisteen).

    Palauttaa listan (x_px, y_px, strength) top-down-kuvan TAYSISSA
    pikselikoordinaateissa (ei kaistan sisaisissa) - vain sailyneet
    (ei-poikkeavat) pisteet poikkeavien hylkayksen jalkeen, tai tyhjan
    listan jos kumpikaan klusteri ei ekstrapoloidu riittavan lahelle
    nimellista odotettua riviä keskiviivalla.
    """

    cx_full, row_f = to_output_px(0.0, expected_y_cm)
    row = int(round(row_f))

    half_px = int(half_height_cm * PIXELS_PER_CM)

    h = topdown_raw.shape[0]
    y1 = max(0, row - half_px)
    y2 = min(h, row + half_px)

    if y2 - y1 < 20:
        return []

    crop = topdown_raw[y1:y2, :]

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)

    lo = float(np.percentile(gray, 2))
    hi = float(np.percentile(gray, 98))

    if hi - lo < 1e-3:
        hi = lo + 1.0

    norm = np.clip((gray - lo) / (hi - lo), 0.0, 1.0) * 255.0
    darkness = 255.0 - norm

    background = cv2.GaussianBlur(darkness, (0, 0), sigmaX=bg_sigma)
    signal = darkness - background

    ch, cw = signal.shape
    margin = int(cw * edge_margin_frac)
    x_lo, x_hi = margin, cw - margin

    if x_hi - x_lo < num_segments * 5:
        return []

    seg_width = (x_hi - x_lo) / num_segments

    raw_points = []

    for i in range(num_segments):

        sx0 = int(round(x_lo + i * seg_width))
        sx1 = int(round(x_lo + (i + 1) * seg_width))

        if sx1 - sx0 < 5:
            continue

        segment = signal[:, sx0:sx1]
        row_profile = segment.mean(axis=1)
        baseline = float(np.median(row_profile))

        # Poimitaan (kamera7_06.py) jopa max_peaks_per_segment ERILLISTA
        # paikallista huippua tasta segmentista - ei vain yhta globaalia
        # argmax:ia - katso funktion kommentti siita MIKSI (jotta seka
        # todellinen hogline etta mahdollinen hairio paasevat molemmat
        # mukaan silloin kun kumpikin on nakyvissa taman segmentin
        # kohdalla).
        working_profile = row_profile.copy()

        for _ in range(max_peaks_per_segment):

            peak_row = int(np.argmax(working_profile))
            peak_val = float(working_profile[peak_row])
            strength = peak_val - baseline

            if strength < min_peak_strength:
                break

            # Osapikselitarkka huippu paraabelisovituksella naapuri-
            # pisteista - AINA alkuperaisesta row_profile:sta (ei
            # vaimennetusta working_profile:sta).
            offset = 0.0
            if 0 < peak_row < len(row_profile) - 1:
                y0v = row_profile[peak_row - 1]
                y1v = row_profile[peak_row]
                y2v = row_profile[peak_row + 1]
                denom = y0v - 2.0 * y1v + y2v
                if abs(denom) > 1e-6:
                    offset = max(-1.0, min(1.0, 0.5 * (y0v - y2v) / denom))

            x_center_full = (sx0 + sx1) / 2.0
            y_center_full = float(y1 + peak_row + offset)

            raw_points.append((x_center_full, y_center_full, strength))

            # Vaimennetaan tama huippu (ja sen valiton ymparisto)
            # working_profile:sta, jotta seuraava argmax loytaa AIDOSTI
            # eri (riittavan kaukana olevan) huipun, ei vain naapuri-
            # pikselin samasta piikista.
            lo = max(0, peak_row - min_peak_separation_px)
            hi = min(len(working_profile), peak_row + min_peak_separation_px + 1)
            working_profile[lo:hi] = -1e9

    if len(raw_points) < min_points:
        return raw_points

    # KAHDEN KLUSTERIN EROTUS + KESKIVIIVAEKSTRAPOLOINTI (katso
    # _select_best_hogline_cluster:in kommentti): jos hakukaistassa on
    # kaksi erillista tummaa piirretta (todellinen hogline + esim.
    # sponsoriteksti/kylttinauha), pistejoukko on usein BIMODAALINEN
    # y-koordinaatin suhteen. Kummallekin klusterille sovitetaan oma
    # suora ja ekstrapoloidaan se keskiviivalle - se jonka ylitys on
    # LAHIMPANA nimellista odotettua riviä (row_f) valitaan.
    raw_points = _select_best_hogline_cluster(raw_points, cx_full, row_f)

    # Poikkeavien hylkays: sovitetaan painotettu suora y = a*x + b
    # jaljella oleviin pisteisiin, hylataan ne joiden jaannos poikkeaa
    # liikaa (mediaani + MAD), toistetaan muutama kierros - sama
    # periaate kuin robust_circle_fit:ssa.
    selected = list(raw_points)

    for _ in range(outlier_iterations):

        if len(selected) < min_points:
            break

        xs = np.array([p[0] for p in selected])
        ys = np.array([p[1] for p in selected])
        ws = np.array([p[2] for p in selected])

        A = np.column_stack([xs, np.ones_like(xs)])
        W = np.diag(ws)
        try:
            coeffs, *_ = np.linalg.lstsq(W @ A, W @ ys, rcond=None)
        except np.linalg.LinAlgError:
            break

        a, b = coeffs
        residuals = np.abs(ys - (a * xs + b))

        median = np.median(residuals)
        mad = np.median(np.abs(residuals - median)) + 1e-6
        keep = residuals < median + outlier_mad_multiplier * mad

        if keep.sum() < min_points or keep.sum() == len(selected):
            selected = [selected[i] for i in range(len(selected)) if keep[i]]
            break

        selected = [selected[i] for i in range(len(selected)) if keep[i]]

    return selected


def robust_line_angle_from_points(points):
    """
    Sovittaa painotetun suoran (y = a*x + b, sama periaate kuin
    detect_hogline_points:in sisainen poikkeavien hylkays) annettuun
    pisteryhmaan (esim. detect_hogline_points:in palauttamat, jo
    dekontaminoidut pisteet) ja palauttaa sen KULMAN asteina (sama
    etumerkkikonventio kuin muualla koodissa: positiivinen kulma <->
    positiivinen kulmakerroin output-px-koordinaatistossa).

    Palauttaa (angle_deg, confidence) - confidence on pisteiden
    yhteenlaskettu "strength" (0.0 jos alle 2 pistetta), kaytettavissa
    esim. lahemman/kaukaisen hoglinen keskinaiseen painotukseen.
    """

    if len(points) < 2:
        return 0.0, 0.0

    xs = np.array([p[0] for p in points], dtype=np.float64)
    ys = np.array([p[1] for p in points], dtype=np.float64)
    ws = np.array([p[2] for p in points], dtype=np.float64)

    A = np.column_stack([xs, np.ones_like(xs)])
    W = np.diag(ws)

    try:
        coeffs, *_ = np.linalg.lstsq(W @ A, W @ ys, rcond=None)
    except np.linalg.LinAlgError:
        return 0.0, 0.0

    a, _b = coeffs

    return math.degrees(math.atan(a)), float(np.sum(ws))


def measure_house_quality(frame_undistorted, H_final):
    """
    Mittaa SEKA lahemman etta kaukaisen pesan MUODON (pyoreys) JA
    ABSOLUUTTISEN KOON yhdessa. Palauttaa dictin:
      "worst_ratio"   - huonoin (pienin) pyoreys (min(w,h)/max(w,h))
                        molemmista pesista.
      "worst_size_err"- suurin suhteellinen koon poikkeama (ulomman
                        renkaan mitatun ELLIPSIN keskimaarainen
                        halkaisija (w+h)/2 verrattuna tunnettuun
                        fyysiseen kokoon) molemmista pesista.

    KRIITTINEN BUGIKORJAUS (loytyi kamera7_04.py:sta kayttajan
    ilmoituksesta "hogline-tasoitus rikkoi pesien koon"): aiempi
    mittari, measure_far_house_shape_ratio, tarkasti VAIN kaukaisen
    pesan PYOREYDEN - ei koskaan lahempaa pesaa, eika kummankaan
    pesan ABSOLUUTTISTA kokoa. Homografialla on riittavasti vapaus-
    asteita tuottaa TAYDELLISEN PYOREA mutta VAARAN KOKOINEN ellipsi
    (skaalattu/vinoutunut kokonaisuudessaan) - tallainen kandidaatti
    lapaisi vanhan tarkistuksen huomaamatta.

    Empiirisesti varmistettu (before_after_size-diagnostiikka,
    molemmat testikuvat): kamera7_04.py:n hogline-tasoitus kutisti
    LAHEMMAN pesan sinisen ulkokehan keskimaarin ~4-7 % (ja pyoreys
    huononi 0.98:sta ~0.94:aan) samalla kun KAUKAISEN pesan pyoreys
    pysyi 1.000:ssa (ja jopa hieman KASVOI yli oikean koon, ~3 %) -
    koska vain kaukaista mitattiin. Talla mittarilla molemmat pesat
    JA molemmat ominaisuudet (muoto+koko) huomioidaan, joten tallainen
    kandidaatti hylataan (katso candidate_is_better/
    house_quality_is_acceptable).
    """

    output_w = int(round((OUTPUT_X_MAX_CM - OUTPUT_X_MIN_CM) * PIXELS_PER_CM))
    output_h = int(round((OUTPUT_Y_MAX_CM - OUTPUT_Y_MIN_CM) * PIXELS_PER_CM))

    topdown_final = cv2.warpPerspective(frame_undistorted, H_final, (output_w, output_h))

    return measure_house_quality_from_topdown(topdown_final)


def measure_house_quality_from_topdown(topdown_final):
    """
    Sama kuin measure_house_quality, mutta ottaa VALMIIN top-down-kuvan
    suoraan (ei frame_undistorted+H_final -paria). Tarpeellinen (kamera7_
    06.py) apply_hogline_warp_correction:in tuottamalle lopputulokselle,
    joka EI enaa ole yhden homografian warpPerspective-tulos (vaan
    cv2.remap-pohjainen rivikohtainen korjaus) - H_final:ia ei siis ole
    enaa mielekkaasti olemassa taman kuvan tuottamiseen.
    """

    expected_diam = {
        "blue_outer": 2.0 * BLUE_OUTER_RADIUS_CM * PIXELS_PER_CM,
        "red_outer": 2.0 * RED_OUTER_RADIUS_CM * PIXELS_PER_CM,
    }

    worst_ratio = 1.0
    worst_size_err = 0.0

    for house_y_cm in (NEAR_HOUSE_Y_CM, FAR_HOUSE_Y_CM):

        view = crop_house_view(topdown_final, house_y_cm, HOUSE_CROP_HALF_HEIGHT_CM)
        expected_center = expected_house_center_in_crop(
            topdown_final.shape[0], house_y_cm, HOUSE_CROP_HALF_HEIGHT_CM
        )
        verify = detect_far_house_shared_ellipses(view, expected_center)

        ring = "blue_outer" if verify.get("blue_outer") is not None else "red_outer"
        ellipse = verify.get(ring)

        if ellipse is None:
            worst_ratio = 0.0
            worst_size_err = 1.0
            continue

        (_, _), (w, h), _ = ellipse

        if max(w, h) <= 0:
            worst_ratio = 0.0
            worst_size_err = 1.0
            continue

        ratio = min(w, h) / max(w, h)
        avg_diam = (w + h) / 2.0
        size_err = abs(avg_diam - expected_diam[ring]) / expected_diam[ring]

        worst_ratio = min(worst_ratio, ratio)
        worst_size_err = max(worst_size_err, size_err)

    return {"worst_ratio": worst_ratio, "worst_size_err": worst_size_err}


def house_quality_str(q):
    return (f"pyoreys(huonoin)={q['worst_ratio']:.3f}, "
            f"koko-virhe(suurin)={q['worst_size_err'] * 100:.1f}%")


def house_quality_is_acceptable(candidate_q, baseline_q, ratio_tol=0.02, size_tol=0.015):
    """
    True jos candidate_q ei ole MERKITTAVASTI huonompi kuin baseline_q
    - kumpikaan mitta (huonoin pyoreys TAI suurin koko-virhe,
    kummankin MOLEMMISTA pesista) ei saa huonontua sallittua
    toleranssia enempaa. ratio_tol=0.02 vastaa aiemmin kaytettya
    "0.02 notkahdusta"; size_tol=0.015 (1.5 %) on tiukempi koska
    koon virhe ei aiemmin ollut lainkaan sallittu kasvaa - havaittu
    virhe (4-7 %) on moninkertainen tahan toleranssiin nahden, joten
    se hylataan luotettavasti.
    """
    ratio_ok = candidate_q["worst_ratio"] > baseline_q["worst_ratio"] - ratio_tol
    size_ok = candidate_q["worst_size_err"] < baseline_q["worst_size_err"] + size_tol
    return ratio_ok and size_ok


def candidate_is_better(candidate_frame_undistorted, candidate_H, candidate_rms,
                         baseline_frame_undistorted, baseline_H, baseline_rms,
                         ratio_margin=0.01, size_improve_margin=0.005):
    """
    Yleinen hyvaksymissaanto: kaytetaan JOKAISESSA kohdassa (kamera7_
    05.py) jossa kandidaattihomografiaa/k1:aa verrataan nykyiseen
    parhaaseen - korvaa kamera7_04.py:n paikoittaiset (kaukaisen
    pesan pyoreyteen rajoittuneet) tarkistukset.

    Hylkaa kandidaatin JOS se huonontaa jommankumman mitan (pyoreys
    TAI koko, MOLEMMAT pesat) sallittua toleranssia enempaa (katso
    house_quality_is_acceptable). Muuten hyvaksyy jos jokin mittari
    OIKEASTI paranee: pyoreys selvasti, TAI koko selvasti, TAI (jos
    kumpikaan ei muutu paljon) RMS paranee.

    Palauttaa (accept: bool, candidate_quality, baseline_quality).
    """

    cand_q = measure_house_quality(candidate_frame_undistorted, candidate_H)
    base_q = measure_house_quality(baseline_frame_undistorted, baseline_H)

    if not house_quality_is_acceptable(cand_q, base_q):
        return False, cand_q, base_q

    if cand_q["worst_ratio"] > base_q["worst_ratio"] + ratio_margin:
        return True, cand_q, base_q

    if cand_q["worst_size_err"] < base_q["worst_size_err"] - size_improve_margin:
        return True, cand_q, base_q

    if (abs(cand_q["worst_ratio"] - base_q["worst_ratio"]) <= ratio_margin
            and candidate_rms < baseline_rms):
        return True, cand_q, base_q

    return False, cand_q, base_q


# ============================================================
# GEOMETRINEN LOPPUVAIHE (kamera8_01.py)
#
# KAYTTAJAN VAATIMUS: lopullinen kuvamuunnos saa koostua VAIN
# objektiivin sateittaisen vaaristyman korjauksesta (k1, jo tehty
# ennen tata - koko frame_undistorted on oikaistu KERRAN) ja YHDESTA
# homografiasta - ei rivikohtaista korjausta, ei erillista
# jalkikasittelyvaihetta (kamera7_06.py:n apply_hogline_warp_correction
# hylattiin talla perusteella).
#
# RATKAISU: pesien renkaat ja hoglinet KAYTETAAN SUORAAN GEOMETRISINA
# RAJOITTEINA sen homografian sovituksessa, sen sijaan etta niista
# rakennettaisiin keinotekoisia pistevastaavuuksia jotka vaativat
# T-linjan SUUNNAN arvaamista (kamera7_xx.py:n dir_lateral/angle-
# painotus, hauras - katso kamera7_06.py:n compute_corrected_homography).
#
#   YMPYRARAJOITE (vastaa duaalikonia Q' = H^-T Q H^-1):
#     kaukaisen pesan renkaan reunapisteet TIEDETAAN olevan tunnetun-
#     sateisella ympyralla fyysisessa tasossa, mutta EI TIEDETA mika
#     reunapiste vastaa mitakin ympyran kohtaa (rengas on rotaatio-
#     symmetrinen) - pistekohtaista vastaavuutta ei siis voi eika
#     tarvitse muodostaa. Toteutettu pistejoukon RAJA-ARVONA: jokaisen
#     havaitun reunapisteen jaannos on sen ETAISYYS tunnettuun
#     ympyraan (ei etaisyys mihinkaan yksittaiseen kohdepisteeseen).
#     Tama on rajalla (paljon pisteita) sama rajoite kuin Q':n
#     algebrallinen yhtaloryhma, mutta ei vaadi konimatriisien
#     kasittelya.
#
#   VIIVARAJOITE (vastaa l' = H^-T l): hoglinen fyysinen Y-koordinaatti
#     tunnetaan tasan, mutta X ei (poikittainen viiva koko radan
#     leveydelta). Jokaisen havaitun hogline-pisteen jaannos on sen
#     etaisyys (Y-suunnassa) tunnettuun fyysiseen viivaan - ei vaadita
#     yksittaisen pisteen X-kohdetta.
#
# Molemmat rajoitteet YHDISTETAAN lahemman pesan 17 tavallisen piste-
# korrespondenssin kanssa YHTEEN epalineaariseen pienimman nelion
# sovitukseen (koko H, 8 vapausastetta, kerralla) - kaikki jaannokset
# FYYSISESSA YKSIKOSSA (cm), joten piste-/ympyra-/viivarajoitteiden
# VALINEN painotus tulee luonnostaan oikeaksi ilman keinotekoisia
# painokertoimia (piste antaa 2 residuaalia koska koko 2D-sijainti
# tunnetaan, ympyra-/viivapiste antaa 1 koska vain etaisyys tunnetaan -
# vastaa kunkin havainnon todellista informaatiosisaltoa; vrt. aiempi
# HOUSE_POINT_WEIGHT=6-viritys kamera7_05/06.py:ssa, joka ei ollut
# taman kaltaiseen fyysiseen perusteluun sidottu).
#
# Koska kaikki pisteet pidetaan TASSA silmukassa yhden kiinteän k1:n
# jo oikaisemassa kuvassa (frame_undistorted, katso main()), tata
# vaihetta EI tarvitse ajaa objektiivin vaaristyman uudelleenarvioinnin
# vuoksi (kamera7_xx.py:n "yhteisoptimointi") - kayttajan oman
# arvion mukaan objektiivikorjaus ei ole enaa kriittinen kun H
# lasketaan oikein, joten k1 estimoidaan vain KERRAN, ennen tata
# silmukkaa.
# ============================================================

# Ylaraja luottamukselle jota kaukaisen pesan rengas-/hogline-pisteet
# voivat saada painotuksessa (katso _robust_scale_cm) - lahemman pesan
# 17 pistetta (T-linjan/keskilinjan suoralla leikkauksella saatuja)
# ovat kaytannossa kohinattomia, ja niiden havaittu jaljelle jaava
# sovitusvirhe (plain DLT: ~1-2 cm, katso testaus) antaa tasta
# luonnollisen mittakaavan: mikaan muu havaintoryhma ei voi olla
# NAITA luotettavampi, joten sen paino ei voi ylittaa 1/NEAR_TRUST_FLOOR_CM.
NEAR_TRUST_FLOOR_CM = 1.0


def _h_from_params(params):
    return np.array([
        [params[0], params[1], params[2]],
        [params[3], params[4], params[5]],
        [params[6], params[7], 1.0],
    ], dtype=np.float64)


def _params_from_h(H):
    H = H / H[2, 2]
    return np.array([
        H[0, 0], H[0, 1], H[0, 2],
        H[1, 0], H[1, 1], H[1, 2],
        H[2, 0], H[2, 1],
    ], dtype=np.float64)


def _apply_h(H, points):
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    ones = np.ones((len(pts), 1), dtype=np.float64)
    hom = np.hstack([pts, ones]) @ H.T
    return hom[:, :2] / hom[:, 2:3]


def _levenberg_marquardt(residual_fn, params0, max_iterations=50,
                          lambda_init=1e-3, rel_tol=1e-10):
    """
    Pieni, riippuvuudeton Levenberg-Marquardt -sovitin (numeerinen
    Jacobian aarellisilla differensseilla - H:lla on vain 8 vapaata
    parametria, joten taman hinta on mitaton). Ei kayteta scipy:ta,
    jotta tiedosto pysyy itsenaisena (kuten kamera7_xx.py, jonka
    ainoat riippuvuudet olivat cv2/numpy/tkinter).
    """

    params = np.array(params0, dtype=np.float64)
    residuals = residual_fn(params)
    cost = float(np.sum(residuals ** 2))

    lam = lambda_init
    eps = 1e-6

    for _ in range(max_iterations):

        n = len(params)
        J = np.zeros((len(residuals), n), dtype=np.float64)

        for j in range(n):
            step = eps * max(1.0, abs(params[j]))
            p_plus = params.copy()
            p_plus[j] += step
            J[:, j] = (residual_fn(p_plus) - residuals) / step

        JTJ = J.T @ J
        JTr = J.T @ residuals
        diag = np.diag(JTJ) + 1e-12

        step_taken = False
        rel_improvement = 0.0

        for _ in range(12):

            A = JTJ + lam * np.diag(diag)

            try:
                delta = np.linalg.solve(A, -JTr)
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue

            trial_params = params + delta
            trial_residuals = residual_fn(trial_params)
            trial_cost = float(np.sum(trial_residuals ** 2))

            if trial_cost < cost:
                rel_improvement = (cost - trial_cost) / max(cost, 1e-12)
                params, residuals, cost = trial_params, trial_residuals, trial_cost
                lam = max(lam / 5.0, 1e-12)
                step_taken = True
                break

            lam *= 5.0

        if not step_taken or rel_improvement < rel_tol:
            break

    return params


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


def _robust_scale_cm(residuals, floor_cm):
    """
    MAD-pohjainen (mediaani+MAD, sama periaate kuin muualla koodissa
    kaytetty poikkeavien hylkays) arvio residuaaliryhman OMASTA
    kohinatasosta (cm) - 1.4826 on vakiokerroin joka muuntaa MAD:in
    keskihajonnaksi normaalijakaumalle. floor_cm estaa painon
    kasvamisen rajattomaksi kun ryhma on jo lahes konvergoitunut, ja
    asettaa YLARAJAN luottamukselle: kaukaisen pesan renkaan/hoglinen
    pistetta EI KOSKAAN luoteta enempaa kuin lahemman pesan 17 pistetta
    (jotka ovat T-linjan/keskilinjan suoralla leikkauksella saatuja,
    kaytannossa kohinattomia).
    """

    if len(residuals) == 0:
        return floor_cm

    median = np.median(residuals)
    mad = np.median(np.abs(residuals - median))

    return max(1.4826 * mad, floor_cm)


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


def frame_points_from_topdown(topdown_pts, H_current):
    """
    Muuntaa top-down-pisteet TAKAISIN sen (kertaalleen oikaistun)
    kuvan pikselikoordinaatistoon jota H_current warppaa (H_current^-1)
    - ei tarvitse palata alkuperaiseen vaaristyneeseen raakakuvaan asti,
    koska k1 on jo kiinnitetty yhdella itsekalibroinnilla ennen
    refine_geometric_homography-silmukkaa (katso main()).
    """

    topdown_pts = np.asarray(topdown_pts, dtype=np.float64)

    if len(topdown_pts) == 0:
        return np.zeros((0, 2))

    H_inv = np.linalg.inv(H_current)

    return cv2.perspectiveTransform(
        topdown_pts.reshape(-1, 1, 2), H_inv
    ).reshape(-1, 2)


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


# ============================================================
# KIVEN NIMELLISMITAT (WCF) - kaytetaan VAIN fit_stone_profile:in
# alkuarvauksena (lahtokohta hienosaadolle) ja jarkevyystarkistukseen.
# Seka sade etta korkeus RATKAISTAAN oikeasti havainnoista - katso
# fit_stone_profile ja sen ylla oleva kommentti kovakoodatusta
# karkeasta 3D-mallista. STONE_HEIGHT_CM/STONE_HEIGHT_MAX_CM ovat
# korkeuden ALA-/YLARAJA (WCF-vahimmaiskorkeus ... kayttajan antama
# 15cm katto) - itse H_total SOVITETAAN naiden valiin, ei kiinniteta.
# ============================================================
STONE_NOMINAL_RADIUS_CM = 91.44 / math.pi / 2.0


STONE_HEIGHT_CM = 11.43


STONE_HEIGHT_MAX_CM = 15.0


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

    A = np.array([
        [PIXELS_PER_CM, 0.0, -OUTPUT_X_MIN_CM * PIXELS_PER_CM],
        [0.0, -PIXELS_PER_CM, OUTPUT_Y_MAX_CM * PIXELS_PER_CM],
        [0.0, 0.0, 1.0],
    ])

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


def _project_3d(K, R, t, points_3d):
    points_3d = np.asarray(points_3d, dtype=np.float64).reshape(-1, 3)
    p_cam = (R @ points_3d.T).T + t
    p_img = (K @ p_cam.T).T
    return p_img[:, 0] / p_img[:, 2], p_img[:, 1] / p_img[:, 2]


def ray_plane_intersection(K, R, t, u, v, Z_target):
    """
    YLEISTETTY sadetasoleikkaus: kuvapisteesta (u,v) lahteva sade
    leikataan VAAKATASON Z=Z_target kanssa (Z_target=0 vastaa
    kamera8_01.py:n koko homografiaa - tama funktio yleistaa sen
    mille tahansa korkeudelle, esim. kiven korkeudelle).

    Palauttaa (X, Y) fyysisissa cm:issa.
    """

    u = np.atleast_1d(np.asarray(u, dtype=np.float64))
    v = np.atleast_1d(np.asarray(v, dtype=np.float64))

    K_inv = np.linalg.inv(K)
    uv1 = np.column_stack([u, v, np.ones_like(u)])
    ray_cam = (K_inv @ uv1.T).T

    R_t = R.T
    C = -R_t @ t
    d = (R_t @ ray_cam.T).T

    s = (Z_target - C[2]) / d[:, 2]
    X = C[0] + s * d[:, 0]
    Y = C[1] + s * d[:, 1]

    return X, Y


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


# ============================================================
# KIVEN (GRANIITTIOSAN) TUNNISTUS RAAKAKUVASTA
#
# Graniitti on tumma ja VAHASATURAATIOINEN (harmaa/musta) - sama
# vahasaturaatio kuin jaalla, joten erottelu ei voi perustua pelkkaan
# saturaatioon (S). Sen sijaan graniitti on PAIKALLISESTI TUMMEMPI
# kuin ymparoiva jaa - sama "paikallinen tummuus taustaan nahden"
# -periaate kuin kamera8_01.py:n detect_hogline_points kayttaa
# (GaussianBlur-taustavahennys), tassa 2D-versiona koko kuvalle.
# Varikas kahva (korkea saturaatio) suljetaan pois eksplisiittisesti.
# ============================================================
STONE_MAX_SATURATION = 60


STONE_DARKNESS_SIGMA = 25


STONE_MIN_DARKNESS = 15.0


# Suodatinrajat find_stone_candidates:lle - katso sen docstring: nama
# eivat ole mielivaltaisia, ne on haettu Kivilla.png:n 3 tunnetun kiven
# perusteella (katso kehityshistoria: ilman jaatasoporttia+muotosuodatinta
# tunnistus loysi 101+ virhekandidaattia - tekstia, viivoja, lakimiehen/
# pyyhkijan vaatteita; jaataso+muoto -yhdistelmalla juuri oikeat 3 jaa).
STONE_MIN_FILL_RATIO = 0.65


STONE_MIN_ASPECT_RATIO = 0.35


STONE_SHEET_MARGIN_CM = 250.0


def create_granite_mask(frame):

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=STONE_DARKNESS_SIGMA)
    darkness = background - gray

    low_saturation = hsv[:, :, 1] < STONE_MAX_SATURATION
    dark_enough = darkness > STONE_MIN_DARKNESS

    mask = (low_saturation & dark_enough).astype(np.uint8) * 255

    # 5x5-avaus on TAHALLAAN isompi kuin tavanomainen 3x3: se poistaa
    # ohuet (muutaman pikselin) rakenteet - sponsoritekstin kirjaimet,
    # keskiviivan/hoglinen maalatun viivan - mutta sailyttaa kiven
    # graniittiosan (paikallisesti kymmenia pikseleita leveana tayttyva
    # alue). 3x3 paasti nama ohuet rakenteet lapi (havaittu testatessa).
    kernel_open = np.ones((5, 5), np.uint8)
    kernel_close = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    return mask


def find_stone_candidates(
    frame, H_final,
    min_area=250, max_area=200000,
    min_fill_ratio=STONE_MIN_FILL_RATIO,
    min_aspect_ratio=STONE_MIN_ASPECT_RATIO,
    sheet_margin_cm=STONE_SHEET_MARGIN_CM,
):
    """
    Etsii kiven NAKYVAN GRANIITTIOSAN ellipsit raakakuvasta (ei
    ylhaaltakuvatusta - kiven oma silhuetti tulkitaan suoraan
    alkuperaisessa perspektiivissa, 'frame' PITAA olla sama
    oikaistu kuva - frame_undistorted - jota H_final/kameramalli
    kayttavat). Palauttaa listan ellipseja (cv2.fitEllipse-muodossa)
    suuruusjarjestyksessa (isoin=lahin ensin) - EI vaadita tasan 3:a,
    kutsuja paattaa mita niista kayttaa.

    Pelkka koko+pyoreys -suodatus EI RIITA (testattu: Kivilla.png:ssa
    101 virhekandidaattia jaljella pelkalla silla) - suurin osa
    virheista (sponsoritekstit, mainostaulut, pelaajan vaatteet) ovat
    kuitenkin fyysisesti KAUKANA itse jaasta. Siksi jokainen kandidaatti
    projisoidaan H_final:lla (Z=0-taso-oletus) fyysiseksi (X,Y)-
    sijainniksi ja hylataan jos se on selvasti radan ULKOPUOLELLA
    (OUTPUT_X/Y-rajat + marginaali) - jaataso-homografia antaa
    JARJETTOMAN kaukaisia (X,Y)-arvoja pisteille jotka eivat ole
    lahella jaatasoa (esim. taustan mainostaulut), joten tama on
    tehokas karkeasuodatin VAIKKA kivella itsellaan onkin korkeutta
    (parallaksin aiheuttama virhe on senttien, ei metrien, luokkaa).

    Palauttaa listan DICTEJA {"ellipse":..., "contour":...} - contour
    (raaka cv2.findContours-ulostulo, muoto (N,1,2)) sailytetaan MYOS,
    koska pelkka 5-parametrinen ellipsi ei riita fit_stone_profile:in
    pyorahdyskappale-muotosovitukseen (katso sen kommentti).
    """

    mask = create_granite_mask(frame)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    x_min = OUTPUT_X_MIN_CM - sheet_margin_cm
    x_max = OUTPUT_X_MAX_CM + sheet_margin_cm
    y_min = OUTPUT_Y_MIN_CM - sheet_margin_cm
    y_max = OUTPUT_Y_MAX_CM + sheet_margin_cm

    candidates = []

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < min_area or area > max_area or len(contour) < 5:
            continue

        ellipse = cv2.fitEllipse(contour)
        (cx, cy), (w, h), angle = ellipse

        ellipse_area = math.pi * (w / 2.0) * (h / 2.0)

        if ellipse_area < 1e-6:
            continue

        fill_ratio = area / ellipse_area

        if fill_ratio < min_fill_ratio:
            continue

        aspect_ratio = min(w, h) / max(w, h)

        if aspect_ratio < min_aspect_ratio:
            continue

        center_px = np.array([[[cx, cy]]], dtype=np.float64)
        output_px = cv2.perspectiveTransform(center_px, H_final).reshape(1, 2)
        phys = output_px_to_physical(output_px)[0]

        if not (x_min <= phys[0] <= x_max and y_min <= phys[1] <= y_max):
            continue

        candidates.append((ellipse, area, contour))

    candidates.sort(key=lambda c: c[1], reverse=True)

    return [{"ellipse": c[0], "contour": c[2]} for c in candidates]


# ============================================================
# KIVEN SEURANTA VIDEOSTA - USEITA KYMMENIA/SATOJA HAVAINTOJA YHDESTA
# KIVESTA LAAJALTA KULMA-ALUEELTA
#
# Kayttajan havainto: yksittainen liikkuva kivi lipuu heittopaasta
# (lahes vaakatasosta, pieni korkeuskulma - kamera kaukana, matala
# projektio) pesaa kohti (jyrkempi korkeuskulma, n. 15-20 astetta),
# eli SAMA fyysinen kivi antaa VALTAVASTI enemman kulmavaihtelua kuin
# 3 paikallaan olevaa kivea samassa kuvassa (jotka olivat kaikki
# suunnilleen samalla etaisyydella, n. 10-25 astetta). Tama on
# TARKEA, koska fit_stone_profile:in H_total-skannaus (katso sen
# kommentti) osoitti RMS:n olevan LITTEA H:n suhteen juuri PUUTTUVAN
# kulmavaihtelun takia - videosta saatu laajempi kulma-alue voi siis
# aidosti parantaa korkeuden erottelukykya.
#
# TUNNISTUSSTRATEGIA: sama create_granite_mask/find_stone_candidates
# -pohjainen tunnistus kuin still-kuville, mutta VAPAAMMILLA muoto-
# rajoilla (kivi on video kompressoinnin+liike-epaterävyyden takia
# usein huonommin rajautunut kuin still-kuvassa, ja PIENI kaukainen
# kivi jaa muuten helposti tiukkojen rajojen alle) - erottelu vaarista
# kandidaateista (pyyhkija, kiinteat jo-paikallaan-olevat kivet,
# kiinteat virhelahteet kuten logotekstit) EI perustu muotoon vaan
# JATKUVUUTEEN: valitaan aina se kandidaatti joka on LAHIMPANA
# edellisen (kasitellyn) framen sijaintia, hylataan jos hyppy on liian
# suuri (max_jump_cm) - staattiset objektit (mukaan lukien kiinteat
# virhekandidaatit) EIVAT liiku framesta toiseen, joten ne eivat
# yleensa hairitse jatkuvuuspohjaista seurantaa kunhan max_jump_cm on
# jarkevasti mitoitettu kiven todelliseen nopeuteen nahden.
#
# Kaytannossa toimivaksi havaittu (00011 - Trim.mp4): seurataan
# MOLEMPIIN suuntiin (framet kasvaen JA vahentyen) yhdesta hyvasta
# "siemen"-havainnosta - katso track_stone_in_video:in kutsuesimerkki
# taman tiedoston main()in tai kommenttien yhteydessa.
# ============================================================
STONE_TRACK_MIN_AREA = 80


STONE_TRACK_MIN_FILL_RATIO = 0.4


STONE_TRACK_MIN_ASPECT_RATIO = 0.15


STONE_TRACK_MAX_JUMP_CM = 45.0


STONE_TRACK_MAX_MISSES = 10


def _candidates_in_frame(frame_bgr, calib, pose, H_final,
                          min_area, min_fill_ratio, min_aspect_ratio):

    frame_u = cv2.undistort(frame_bgr, calib["camera_matrix"],
                             np.array([calib["best_k1"], 0.0, 0.0, 0.0, 0.0]))
    stones = find_stone_candidates(
        frame_u, H_final, min_area=min_area,
        min_fill_ratio=min_fill_ratio, min_aspect_ratio=min_aspect_ratio
    )

    out = []

    for stone in stones:
        (cx, cy), _, _ = stone["ellipse"]
        X0, Y0 = _stone_ground_position_z0(pose, cx, cy)
        out.append({"ellipse": stone["ellipse"], "contour": stone["contour"], "pos_cm": (X0, Y0)})

    return out


# ============================================================
# KOVAKOODATTU KARKEA 3D-MALLI KIVEN GRANIITTIOSASTA
#
# Kayttajan pyynnosta: sen sijaan etta sovitettaisiin vapaamuotoinen tai
# yksinkertainen (lieriö/puoliellipsi) muoto suoraan kolmesta havainnosta
# (edelliset yritykset, katso git-historia - lieriomalli antoi selvasti
# liian pienen sateen ~9.8cm koska se ei huomioinut etta KUVUN alla
# oleva PIILOSSA OLEVA (itsensa varjostama) osa kivesta on todellisuudessa
# LEVEAMPI), tassa KOVAKOODATAAN karkea, kasin arvioitu pyorahdysprofiili
# joka jo suunnilleen VASTAA oikean curling-kiven muotoa (kapea ylhaalta
# missa kahva pultataan kiinni, levenee alaspain kohti juoksurengasta),
# ja Python-koodi HIENOSAATAA talle muutaman muotoparametrin + globaalin
# skaalan (R_max, H_total) KAIKKIEN havaittujen kivien aariviivaa vasten
# YHTEISESTI. Tama on paljon paremmin rajoitettu (well-posed) ongelma
# kuin vapaan muodon sovitus 3 havainnosta, koska lahtokohta on jo
# LAHELLA oikeaa muotoa - optimointi vain KORJAA sen, ei keksi sita
# tyhjasta. Lopputulos on tarkka, JUURI NAIDEN kivien mukainen malli,
# jota voi kayttaa myos KAUKAISTEN/pienten havaintojen tunnistukseen
# (locate_stone_from_profile) koska malli itse on jo fysikaalisesti
# jarkeva eika ole ylisovitettu yksittaisen (kohinaisen) aariviivan
# yksityiskohtiin.
#
# Profiili (normalisoitu, z_frac ja r_frac molemmat valilla [0,1]):
# MITATTU OIKEASTA KIVESTA (kayttaja lisasi Kivi.jpg-referenssikuvan -
# kivi kuvattuna tasan sivulta poydalla). Mitattu kuvankasittelylla:
# graniitin aariviiva segmentoitiin (rajattu keltaisesta kahvasta ja
# taustasta), leveys mitattiin joka rivilla, normalisoitu leveimman
# kohdan (max leveys) suhteen. Vain ALAPUOLISKO (pohjasta puoliväliin)
# on mitattua dataa - YLAPUOLISKO PAKOTETAAN taman PEILIKUVAKSI
# (kayttajan pyynnosta): oikea curling-kivi on fyysisesti symmetrinen
# puolivalikorkeuden suhteen (kivi kaannetaan ymparí kun toinen
# juoksurengas kuluu, joten graniittirunko on valmistettu symmetriseksi
# - alkuperaisessa kuvassa nakyva ylareunan jyrkka kavennys EI ole
# graniitin oma muoto vaan kahvan kiinnityslevyn AIHEUTTAMA rajaus,
# katso HANDLE_PLATE_R_FRAC_GUESS alempana talle eri, ei-symmetriselle
# ominaisuudelle). Pohjan (z=0) tarkka arvo on arvio (poydan/varjon
# reunalla vaikea mitata tarkasti kuvasta - katso git-historia), muu on
# suoraan mitattua.
_HALF_PROFILE_TEMPLATE_NORM = [
    (0.00, 0.75),   # pohja (arvioitu - kovera alusta, ei tarkkaan mitattavissa kuvasta)
    (0.18, 0.94),   # levenee nopeasti
    (0.30, 0.98),
    (0.45, 1.00),
    (0.50, 1.00),   # puolivali - "paiva", leveimmillaan (symmetria-akseli)
]


def _mirror_half_profile(half_profile):
    """
    Rakentaa taydet (z_frac, r_frac) -taulukot puoliskosta peilaamalla
    puolivalin (viimeinen piste, r_frac=1.0) ylapuolelle - katso
    _HALF_PROFILE_TEMPLATE_NORM:in kommentti symmetriaoletuksesta.
    """

    z_half = np.array([p[0] for p in half_profile], dtype=np.float64)
    r_half = np.array([p[1] for p in half_profile], dtype=np.float64)

    z_top = 1.0 - z_half[-2::-1]
    r_top = r_half[-2::-1]

    return np.concatenate([z_half, z_top]), np.concatenate([r_half, r_top])


_TEMPLATE_Z_FRAC, _TEMPLATE_R_FRAC = _mirror_half_profile(_HALF_PROFILE_TEMPLATE_NORM)


_TEMPLATE_HALF_LEN = len(_HALF_PROFILE_TEMPLATE_NORM)


_TEMPLATE_EQUATOR_IDX = _TEMPLATE_HALF_LEN - 1


# Kahvan (muovi+kiinnitys) aiheuttaman kolon sade suhteessa R_max:iin.
# Kolon SIJAINTI on kayttajan pyynnosta MAARITELTY (ei sovitettu):
# KESKITETTY kiven pystyakselille (X0,Y0), korkeudella z=H_total - ei
# siirretty sivuun, koska kahvan tarkka atsimuuttikulma vaihtelee
# heitosta toiseen (kivi pyorii liu'un aikana) eika ole taten
# luotettavasti paateltavissa. SADE SEN SIJAAN ON (kayttajan pyynnosta,
# katso keskusteluhistoria - yhden kiven koko liu'un kattava seuranta
# antaa riittavasti kulmavaihtelua/dataa taman luotettavaan
# ratkaisuun) fit_stone_profile:in vapaa, sovitettu parametri - tama
# HANDLE_NOTCH_R_FRAC-vakio toimii enaa VAIN (1) sovituksen
# alkuarvauksena ja (2) stone_tracker.cpp:n elavan seurannan KIINTEANA
# arvona (C++-puolen notch-vahennysta EI ole viela tehty ajonaikaisesti
# parametroitavaksi - katso stone_tracker.cpp:n oma HANDLE_NOTCH_R_FRAC-
# kommentti). Sovitettu arvo on kaytannossa havaittu hyvin lahella
# tata (~0.68-0.70), joten ero on pieni.
HANDLE_NOTCH_R_FRAC = 0.70


HANDLE_NOTCH_R_FRAC_MIN = 0.30


HANDLE_NOTCH_R_FRAC_MAX = 0.95


# Kuinka voimakkaasti muotoa (kontrollipisteiden sateet) rangaistaan
# poikkeamasta kovakoodattuun mallinnukseen nahden (yksikko: "pikselia
# per r_frac-yksikko" - katso fit_stone_profile). Pitaa hienosaadon
# LAHELLA fysikaalisesti jarkevaa lahtokohtaa, estaa ylisovituksen
# kohinaisen aariviivan yli.
#
# HUOM (paivitetty - katso git-historia): kun profiilin KAIKKI
# kontrollipisteet vapautettiin (ei enaa vain "paivan ylapuoliset"),
# vapaita muotoparametreja oli 5 - reilusti enemman kuin ennen (2-3).
# Alkuperainen 25 (viritetty vanhalle, suppeammalle mallille) antoi
# 3 kiven aineistolla epatasaisia, lievasti epafyysisia tuloksia
# (pieni "olkapaa"-kohouma profiilissa) - nostettu 60:een, joka pitaa
# muodon sileampana/uskottavampana samalla kun sallii aidon korjauksen.
# (Symmetriapakon jalkeen vapaita muotoparametreja on enaa PUOLET
# taman verran - vain alapuoliskon kontrollipisteet, paiva pois lukien
# - ylapuolisko peilataan, ei sovita erikseen. Painoa ei ole viritetty
# uudelleen taman muutoksen jalkeen.)
STONE_SHAPE_REG_WEIGHT = 60.0


def _catmull_rom_r_frac(z_query, z_fracs=_TEMPLATE_Z_FRAC, r_fracs=None):
    """
    SILEA (C1-jatkuva) kayra harvojen kontrollipisteiden (6 kpl) lapi -
    EI scipy:ta (projektin kaytanto, katso kamera8_01.py:n kommentti),
    pelkka Catmull-Rom-splini numpylla. Ilman tata paloittain-
    LINEAARINEN interpolointi (np.interp) tekee siluetista kulmikkaan/
    "monikulmiomaisen" - oikea kivi on kuitenkin sileapintainen, joten
    kulmikkuus oli suora syy siihen etta sivukuva ei nayttanyt oikealta
    curling-kivelta (kayttajan havainto).

    Reunat kasitellaan TOISTAMALLA ensimmainen/viimeinen kontrollipiste
    (vakiintunut Catmull-Rom-reunakasittely) - antaa jarkevan, ei-
    ylitse-ampuvan tangentin reunoilla ilman erillista reunaehtoa.
    """

    if r_fracs is None:
        r_fracs = _TEMPLATE_R_FRAC

    n = len(z_fracs)
    z_ext = np.concatenate([[z_fracs[0]], z_fracs, [z_fracs[-1]]])
    r_ext = np.concatenate([[r_fracs[0]], r_fracs, [r_fracs[-1]]])

    r_query = np.empty_like(z_query, dtype=np.float64)

    for i in range(n - 1):
        z0, z1 = z_fracs[i], z_fracs[i + 1]
        mask = (z_query >= z0) & (z_query <= z1)
        if not np.any(mask):
            continue
        span = z1 - z0
        t = (z_query[mask] - z0) / span if span > 1e-12 else np.zeros(np.sum(mask))
        p0, p1, p2, p3 = r_ext[i], r_ext[i + 1], r_ext[i + 2], r_ext[i + 3]
        m1 = (p2 - p0) / 2.0
        m2 = (p3 - p1) / 2.0
        t2 = t * t
        t3 = t2 * t
        h00 = 2 * t3 - 3 * t2 + 1
        h10 = t3 - 2 * t2 + t
        h01 = -2 * t3 + 3 * t2
        h11 = t3 - t2
        r_query[mask] = h00 * p1 + h10 * m1 + h01 * p2 + h11 * m2

    return r_query


def _stone_ground_position_z0(pose, cx, cy):
    """
    KARKEA (parallaksin sisaltava) maa-asema: Z=0-sadetasoleikkaus
    ellipsin KESKIPISTEEN lapi. EI viela Z-korjattu - katso
    compute_stone_ground_position taman alla oikealle korjaukselle.
    """

    X, Y = ray_plane_intersection(pose["K"], pose["R"], pose["t"], cx, cy, 0.0)

    return float(X[0]), float(Y[0])


def _predicted_stone_hull(pose, X0, Y0, R_max, H_total, shape_deltas, n_theta=28, n_per_segment=5):
    """
    Ennustaa kiven kuvassa nakyvan siluetin KUPERAN PEITTEEN annetulla
    profiililla (kovakoodattu STONE_PROFILE_TEMPLATE_NORM + hienosaato-
    deltat r_frac:iin). Palauttaa cv2.convexHull-muotoisen polygonin
    (float32, muoto (N,1,2)) tai None.

    HUOM (korjattu - katso git-historia): TAMA NAYTTEISTAA KOKO
    PROFIILIN (z=0 pohjasta huippuun), EI VAIN "paivan" (levein kohta)
    ylapuolista osaa. Aiempi versio rajasi VAIN paivan ylapuolisen osan
    olettaen etta pohja on AINA itsensa varjossa/piilossa - tama pitaa
    paikkansa JYRKASTA (lahes ylhaaltapain) kuvakulmasta, mutta EI
    matalasta/lahes vaakatasoisesta kuvakulmasta (kayttajan havainto:
    videosta seuratun kiven matalimmat kuvakulmat, n. 6 astetta,
    nayttavat aidosti ENEMMAN kiven kyljesta kuin paivan ylapuolisen
    osan malli pystyi selittamaan - tama "vuosi" virheellisesti
    sovitettuihin muotoparametreihin, jotka nakyivat vinoina/
    epafyysisina sivukuvassa). KUPERA PEITE koko profiilista hoitaa
    itse-varjostuksen OIKEIN AUTOMAATTISESTI: pohjan lahella olevien
    rengaspisteiden projektiot jaavat leveamman paivan/kuvun kattaman
    alueen SISALLE (eivat vaikuta kuperaan peitteeseen) JYRKASTA
    kulmasta, mutta tulevat NAKYVIIN (peitteen reunalle) matalasta
    kulmasta - juuri niin kuin todellisuudessakin.
    """

    R_max = abs(R_max)
    H_total = max(abs(H_total), 1e-6)

    # "Paiva" (_TEMPLATE_EQUATOR_IDX) MAARITTELEE R_max:in (leveimman
    # kohdan sade ON R_max, per maaritelma) - sen oma delta EI SAA
    # olla vapaa (muuten sama fyysinen suure - "kuinka levea kivi on
    # leveimmillaan" - olisi ilmaistu KAHDESTI redundantisti, R_max:in
    # JA paivan oman deltan kautta, mika teki optimoinnista rappeutuneen:
    # jokin MUU kontrollipiste saattoi "livahtaa" paivaa leveammaksi,
    # tuottaen epafyysisen kaksoiskumpu-muodon - katso git-historia).
    # MIKAAN piste ei myoskaan saa olla paivaa LEVEAMPI (paiva ON
    # maaritelmallisesti levein kohta) - siksi ylaraja on tasan 1.0.
    r_fracs = _TEMPLATE_R_FRAC + shape_deltas
    r_fracs[_TEMPLATE_EQUATOR_IDX] = 1.0
    r_fracs = np.clip(r_fracs, 0.05, 1.0)

    n_dense = max(len(_TEMPLATE_Z_FRAC) * n_per_segment, 2)
    z_frac_dense = np.linspace(0.0, 1.0, n_dense)
    r_frac_dense = _catmull_rom_r_frac(z_frac_dense, r_fracs=r_fracs)

    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)

    rings = []

    for zf, rf in zip(z_frac_dense, r_frac_dense):
        z = zf * H_total
        r = rf * R_max
        xs = X0 + r * np.cos(theta)
        ys = Y0 + r * np.sin(theta)
        zs = np.full(n_theta, z)
        rings.append(np.column_stack([xs, ys, zs]))

    points_3d = np.vstack(rings)
    u, v = _project_3d(pose["K"], pose["R"], pose["t"], points_3d)
    points_2d = np.column_stack([u, v]).astype(np.float32)

    if not np.all(np.isfinite(points_2d)):
        return None

    hull = cv2.convexHull(points_2d)

    if len(hull) < 3:
        return None

    return hull


def _handle_notch_hull(pose, X0, Y0, R_max, H_total, r_frac=HANDLE_NOTCH_R_FRAC, n_theta=28):
    """
    Kahvan aiheuttaman kolon projisoitu 2D-ääriviiva: MAARITELTY
    litteä kiekko kiven pystyakselin KESKELLA (X0,Y0), korkeudella
    z=H_total, sateella r_frac*R_max - katso HANDLE_NOTCH_R_FRAC:in
    kommentti. Palauttaa cv2.convexHull-muotoisen polygonin (float32)
    tai None jos projektio epaonnistuu.
    """

    r_cm = abs(r_frac) * abs(R_max)
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    xs = X0 + r_cm * np.cos(theta)
    ys = Y0 + r_cm * np.sin(theta)
    zs = np.full(n_theta, H_total)
    points_3d = np.column_stack([xs, ys, zs])

    u, v = _project_3d(pose["K"], pose["R"], pose["t"], points_3d)
    points_2d = np.column_stack([u, v]).astype(np.float32)

    if not np.all(np.isfinite(points_2d)):
        return None

    return cv2.convexHull(points_2d)


def _signed_dist_with_notch(hull, notch_hull, point):
    """
    Etumerkillinen etaisyys graniittirungon (hull) MIINUS kahvan kolon
    (notch_hull) reunaan - positiivinen SISALLA todellisessa (kolollisessa)
    muodossa, negatiivinen ULKOPUOLELLA (joko kokonaan hullin ulkopuolella
    TAI kolon SISALLA, koska kolo on POIS LEIKATTU alue).
    """

    d_outer = cv2.pointPolygonTest(hull, point, True)

    if notch_hull is None or d_outer <= 0:
        return d_outer

    d_notch = cv2.pointPolygonTest(notch_hull, point, True)

    if d_notch > 0:
        return -d_notch

    return min(d_outer, -d_notch)


def _sample_contour_points(contour, n_sample):
    """Tasavalisesti alinaytetty kontuuri - koko kontuuria (satoja
    pisteita) ei tarvita, muutama kymmenen riittaa sovitukseen ja
    pitaa jokaisen residuals()-kutsun nopeana."""

    points = contour.reshape(-1, 2).astype(np.float64)

    if len(points) <= n_sample:
        return points

    idx = np.linspace(0, len(points) - 1, n_sample).astype(int)

    return points[idx]


def _profile_residuals_for_stone(pose, X0, Y0, R_max, H_total, shape_deltas, contour, n_sample,
                                  handle_r_frac=HANDLE_NOTCH_R_FRAC):

    hull = _predicted_stone_hull(pose, X0, Y0, R_max, H_total, shape_deltas)
    sampled = _sample_contour_points(contour, n_sample)

    if hull is None:
        return np.full(len(sampled), 1000.0)

    notch_hull = _handle_notch_hull(pose, X0, Y0, R_max, H_total, r_frac=handle_r_frac)

    return np.array([
        _signed_dist_with_notch(hull, notch_hull, (float(p[0]), float(p[1])))
        for p in sampled
    ])


def _sigmoid_bounded(x, lo, hi):
    """Kuvaa rajoittamattoman x:n valille (lo, hi) - kayttaa LM-sovitin
    (_levenberg_marquardt) EI tue rajoitettuja parametreja suoraan,
    joten raja pakotetaan tallä logistisella uudelleenparametroinnilla
    (LM nakee vain rajoittamattoman x:n, ei voi koskaan tuottaa lo/hi:n
    ULKOPUOLELLA olevaa H_total:ia riippumatta askeleen koosta)."""

    return lo + (hi - lo) / (1.0 + np.exp(-x))


def _inverse_sigmoid_bounded(v, lo, hi):
    p = np.clip((v - lo) / (hi - lo), 1e-6, 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def _expand_symmetric_shape_deltas(half_deltas):
    """
    half_deltas: korjaukset _HALF_PROFILE_TEMPLATE_NORM:in pisteisiin
    PAIVAA (puolivalia) lukuunottamatta, siis pituus _TEMPLATE_HALF_LEN-1.
    Palauttaa TAYDEN (molempien puoliskojen) delta-taulukon, jossa
    ylapuolisko on PAKOTETUSTI sama kuin alapuolisko (peilattuna) -
    katso _HALF_PROFILE_TEMPLATE_NORM:in kommentti symmetriaoletuksesta.
    Jarjestys vastaa _mirror_half_profile:n rakentamaa tayden profiilin
    jarjestysta.
    """

    full = np.zeros(len(_TEMPLATE_R_FRAC))
    full[:len(half_deltas)] = half_deltas
    full[_TEMPLATE_HALF_LEN:] = half_deltas[::-1]
    return full


def fit_stone_profile(pose, stones, n_sample_per_stone=40,
                       initial_radius_cm=STONE_NOMINAL_RADIUS_CM,
                       height_min_cm=STONE_HEIGHT_CM,
                       height_max_cm=STONE_HEIGHT_MAX_CM,
                       shape_reg_weight=STONE_SHAPE_REG_WEIGHT,
                       handle_r_frac_min=HANDLE_NOTCH_R_FRAC_MIN,
                       handle_r_frac_max=HANDLE_NOTCH_R_FRAC_MAX):
    """
    HIENOSAATAA kovakoodatun karkean mallin (STONE_PROFILE_TEMPLATE_NORM)
    KAIKKIEN havaittujen kivien KOKO AARIVIIVAA vasten YHTEISESTI - katso
    taman osion alkupaan kommentti periaatteesta. Tuntemattomat: R_max
    (paivan/juoksurenkaan sade) + H_total (korkeus, katso alla) + pieni
    korjaus (delta) JOKAISEN ALAPUOLISKON kontrollipisteen r_frac:iin
    (paivaa lukuunottamatta - katso _predicted_stone_hull:in kommentti
    MIKSI matalasta kuvakulmasta nakyy aidosti myos paivan ALApuolista
    kylkea, joten sekin voi tulla oikeasti sovitetuksi, ei vain
    oletukseksi) + jokaisen kiven oma maa-asema (X0,Y0). YLAPUOLISKON
    deltat EIVAT OLE vapaita - ne PAKOTETAAN samoiksi kuin alapuoliskon
    (peilattuna, katso _expand_symmetric_shape_deltas), koska kivi on
    kayttajan pyynnosta oletettu fyysisesti symmetriseksi puolivali-
    korkeuden suhteen (oikea curling-kivi kaannetaan ymparí kun toinen
    juoksurengas kuluu, joten graniittirunko ON valmistettu symmetriseksi
    - vain kahvan kiinnityslevy, joka EI kuulu tahan profiiliin, rikkoo
    symmetrian oikeasti). Muotokorjaukset ovat REGULOITUJA (shape_reg_
    weight, SKAALATTUNA havaintojen maaran mukaan - katso alla) nollaa
    (=kovakoodattu malli) kohti, jotta havainnot eivat ylisovita muotoa -
    vain skaala ja KARKEA muototrendi (esim. onko malli hieman liian/
    liian vahan kupera) voi todella muuttua.

    HUOM regularisoinnin SKAALAUKSESTA (havaittu testatessa videosta
    seurattua 25+ pisteen aineistoa - katso git-historia): datan
    jaannostermien maara kasvaa LINEAARISESTI havaintojen lukumaaran
    (N) mukaan, mutta regularisointitermien maara EI (aina n_shape
    kappaletta) - siis SAMALLA shape_reg_weight:lla regularisointi
    "laimenee" pois suhteessa N:aan, ja isolla N:lla (esim. 28 kiven
    video+still-yhdistelmadata) malli alkoi taipua EPAFYYSISEEN,
    ei-monotoniseen muotoon (kohina/liike-epaterävyys imeytyi muotoon
    "aitona" rakenteena). Korjattu kertomalla shape_reg_weight
    suhteella len(stones)/3 (3 = alkuperainen virityspiste, jolla
    shape_reg_weight=25 antoi jo hyvan tuloksen) - pitaa regularisoinnin
    SUHTEELLISEN vaikutuksen samana havaintomaarasta riippumatta.

    HUOM H_total (kayttajan pyynnosta, katso git-historia AIEMMASTA
    kiinteasta versiosta): nyt VAPAA parametri, mutta RAJOITETTU
    valille [height_min_cm, height_max_cm] (oletus: WCF-vahimmaiskorkeus
    11.43cm ... kayttajan antama 15cm katto) _sigmoid_bounded:in kautta,
    koska LM-sovitin itse ei tue rajoituksia. TAMA EI POISTA aiemmin
    havaittua degeneraatiota (3 kivea + n. 10-25 asteen korkeuskulma-
    alue ei riita erottamaan "hieman pienempi R + suurempi H" ja "hieman
    suurempi R + pienempi H" -ratkaisuja toisistaan, koska nailla on
    lahes SAMA siluetti - RMS oli litea valilla n. 6-12cm) - rajat vain
    ESTAVAT sovitusta ajautumasta fysikaalisesti mahdottomaan arvoon
    (esim. alle WCF-minimin) sen sijaan etta korjaisivat itse
    tunnistettavuusongelman. Jos havaintoaineistossa ei ole aidosti
    matalia (~alle 10 asteen) kuvakulmia, H_total voi silti asettua
    lahes mielivaltaisesti rajojen sisalle - kayttajan kannattaa
    tarkistaa residual_rms_px:n herkkyys H:lle tapauskohtaisesti.

    HUOM handle_r_frac (kayttajan pyynnosta lisatty): kahvan aiheuttaman
    kolon SADE (r_frac * R_max) on nyt MYOS vapaa parametri, rajoitettuna
    valille [handle_r_frac_min, handle_r_frac_max] samalla _sigmoid_
    bounded-periaatteella kuin H_total. Kolon SIJAINTI (keskitetty
    X0,Y0-akselille, z=H_total:ssa) pysyy kuitenkin MAARITELTYNA, ei
    vapaana - katso HANDLE_NOTCH_R_FRAC:in kommentti MIKSI atsimuutti-
    kulmaa ei voi luotettavasti sovittaa. Sateen sovitus VAATII riittavan
    laajan kulmavaihtelun toimiakseen luotettavasti - kayttajan mittaus
    (yhden kiven KOKO liu'un kattava seuranta, ~25 tasavalisesti
    naytteistettya havaintoa radan molemmista paista) antoi vakaan,
    toistettavan tuloksen (~0.68-0.70) - paljon lyhyemmalla/suppeammalla
    havaintojoukolla (esim. vain muutama lahekkainen frame) tulos voi
    olla epaluotettava samasta syysta kuin H_total:in degeneraatio-huomio
    ylla.

    stones: find_stone_candidates:in palauttamat dictit (tarvitaan seka
    "ellipse" etta "contour").

    Palauttaa dictin: R_max_cm (sovitettu), H_total_cm (sovitettu, katso
    yllaoleva HUOM), handle_r_frac (sovitettu, katso yllaoleva HUOM),
    shape_deltas (hienosaadetut poikkeamat kovakoodattuun malliin,
    molemmat puoliskot, ylapuolisko peilattu alapuoliskosta),
    positions_cm, residuals_px (VAIN aariviiva-jaannokset, ilman
    regularisointitermeja), residual_rms_px.
    """

    if len(stones) < 2:
        raise RuntimeError(
            "Profiilin sovitukseen tarvitaan vahintaan 2 kiven havaintoa "
            f"(saatiin {len(stones)})."
        )

    # Vain ALAPUOLISKON kontrollipisteet (paivaa lukuunottamatta) ovat
    # vapaita - katso taman funktion docstring symmetriapakosta.
    # _TEMPLATE_EQUATOR_IDX:in (=paivan) delta EI OLE vapaa parametri
    # muutenkaan - katso _predicted_stone_hull:in kommentti: se piste
    # MAARITTELEE R_max:in (leveimman kohdan sade on R_max per
    # maaritelma), joten oma vapaa delta sille olisi redundantti (ja
    # aiheutti rappeutuneen, ei-monotonisen sovituksen - katso git-
    # historia).
    n_shape = _TEMPLATE_HALF_LEN - 1
    effective_reg_weight = shape_reg_weight * (len(stones) / 3.0)

    initial_height_cm = 0.5 * (height_min_cm + height_max_cm)
    h_free0 = _inverse_sigmoid_bounded(initial_height_cm, height_min_cm, height_max_cm)

    handle_free0 = _inverse_sigmoid_bounded(HANDLE_NOTCH_R_FRAC, handle_r_frac_min, handle_r_frac_max)

    positions0 = []

    for stone in stones:
        (cx, cy), _, _ = stone["ellipse"]
        X0, Y0 = _stone_ground_position_z0(pose, cx, cy)
        positions0.append((X0, Y0))

    def unpack(params):
        R_max = params[0]
        H_total = _sigmoid_bounded(params[1], height_min_cm, height_max_cm)
        handle_r_frac = _sigmoid_bounded(params[2], handle_r_frac_min, handle_r_frac_max)
        half_deltas = params[3:3 + n_shape]
        shape_deltas = _expand_symmetric_shape_deltas(half_deltas)
        positions = params[3 + n_shape:].reshape(-1, 2)
        return R_max, H_total, handle_r_frac, shape_deltas, positions

    def residuals(params, include_reg=True):

        R_max, H_total, handle_r_frac, shape_deltas, positions = unpack(params)
        parts = []

        for (X0, Y0), stone in zip(positions, stones):
            parts.append(_profile_residuals_for_stone(
                pose, X0, Y0, R_max, H_total, shape_deltas,
                stone["contour"], n_sample_per_stone,
                handle_r_frac=handle_r_frac
            ))

        if include_reg:
            half_deltas = params[3:3 + n_shape]
            parts.append(half_deltas * effective_reg_weight)

        return np.concatenate(parts)

    params0 = np.concatenate([
        [initial_radius_cm, h_free0, handle_free0],
        np.zeros(n_shape),
        np.array(positions0, dtype=np.float64).ravel(),
    ])

    params_final = _levenberg_marquardt(residuals, params0, max_iterations=100)
    R_max, H_total, handle_r_frac, shape_deltas, positions = unpack(params_final)
    resid_contour_only = residuals(params_final, include_reg=False)

    return {
        "R_max_cm": float(abs(R_max)),
        "H_total_cm": float(H_total),
        "handle_r_frac": float(handle_r_frac),
        "shape_deltas": shape_deltas,
        "positions_cm": [(float(x), float(y)) for x, y in positions],
        "residuals_px": resid_contour_only,
        "residual_rms_px": float(np.sqrt(np.mean(resid_contour_only ** 2))),
    }


# ============================================================
# HAKUALUE ("kauempi paa, keskiviivasta +-70cm, hogline-1m...hogline+3m")
#
# Kayttajan pyynnosta kavennettu/kohdennettu (katso keskusteluhistoria):
# AIEMMIN hogline...pesan takareuna (tangentti 12-jalkaisen renkaan
# taakse, HOUSE_RADIUS_CM paan pesan keskipisteesta poispain hoglinesta),
# SITTEN hogline...hogline+3m - koska kivi ylittaa hoglinen aina heiton
# alkupaassa, 3m riittaa kattamaan sen EIKA alue enaa ulotu asti pesan
# kohdalle (jossa pelaajat/pyyhkijat useimmin seisovat/kavelevat, katso
# ENABLE_SHADOW_TOLERANT_STABILIZATION-tyon empiirinen havainto). X-
# leveys kasvatettu 50->70cm HAKU:n omaksi turvamarginaaliksi. Kaukainen
# paa (FAR_*) koska talla videolla kivi heitetaan lahempaa hoglinea
# kohti sita - katso kamera9_01.py:n track_stone_in_video-kommentti
# samasta videosta.
#
# NYT (kayttajan pyynnosta) kasvatettu hieman TAAKSEPAIN: hogline-1m...
# hogline+3m - alaraja ulottuu 100cm hoglinen TAAKSE (kohti kaukaista
# pesaa) kattaakseen myos kiven joka on juuri ja juuri ylittanyt
# hoglinen jo ennen ensimmaista hakutarkistusframea.
# ============================================================
SEARCH_X_HALF_WIDTH_CM = 70.0


SEARCH_Y_MIN_CM = FAR_HOGLINE_Y_CM - 100.0


SEARCH_Y_MAX_CM = FAR_HOGLINE_Y_CM + 300.0


# Karkea->hieno ristikkohaku (sama periaate kuin kamera8_01.py:n
# search_far_house) - kaksi tasoa riittaa, koska kolmas (ultra-hieno)
# taso ei enaa muuta lopputulosta merkittavasti mutta maksaa yhta
# paljon kuin ensimmainen.
SEARCH_COARSE_STEP_CM = 10.0


SEARCH_FINE_STEP_CM = 2.0


# Karkean ristikkohaun ESIKARSINTA (kayttajan pyynnosta laskettu, katso
# keskusteluhistoria): tama on VAIN nopea karsinta ENNEN yhteissovitusta
# (refinePositionJoint) - lopullinen hyvaksynta tehdaan stone_tracker.
# cpp:n HAKU_ACCEPT_SCORE_THRESHOLD:lla VASTA sovituksen jalkeen (samalla
# peitto-/ulkopuoli-pisteytyksella, katso sen oma kommentti). Pidetty
# TAHALLAAN MATALAMPANA kuin lopullinen kynnys, jotta kaukana/pienena
# nakyvat AIDOT kivet (epatarkka karkea ristikko-osuma ennen hienosaatoa)
# eivat karsiudu pois jo tassa vaiheessa - havaittu oikealla MAH-videolla
# etta vanha 0.55 esti useiden aitojen, 20m+ liu'un tehneiden kivien
# rekisteroinnin taalla kaukovyohykkeella.
SEARCH_SCORE_THRESHOLD = 0.15  # vain esikarsinta, katso yllaoleva kommentti


TRACK_COARSE_STEP_CM = 7.0


TRACK_FINE_STEP_CM = 1.5


TRACK_SCORE_THRESHOLD = 0.35  # matalampi - jatkuvuus on jo vahva prior


TRACK_LOST_MAX_MISSES = 5     # montako peräkkäistä huonoa framea ennen kuin palataan hakuun


# Pienempi resoluutio hylylle HAUSSA/SEURANNASSA (nopeampi, riittava
# karkean sijainnin loytamiseen - ei tarvita samaa tarkkuutta kuin
# esim. profiilin sovituksessa).
SEARCH_HULL_N_THETA = 14


SEARCH_HULL_N_PER_SEGMENT = 2


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

    r_fracs = _TEMPLATE_R_FRAC + shape_deltas
    r_fracs[_TEMPLATE_EQUATOR_IDX] = 1.0
    r_fracs = np.clip(r_fracs, 0.05, 1.0)

    n_dense = max(len(_TEMPLATE_Z_FRAC) * n_per_segment, 2)
    z_frac_dense = np.linspace(0.0, 1.0, n_dense)
    r_frac_dense = _catmull_rom_r_frac(z_frac_dense, r_fracs=r_fracs)

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
    u, v = _project_3d(pose["K"], pose["R"], pose["t"], pts)
    points_2d = np.column_stack([u, v]).astype(np.float32)

    if not np.all(np.isfinite(points_2d)):
        return None

    hull = cv2.convexHull(points_2d)

    if len(hull) < 3:
        return None

    return hull


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
    output_w = int(round((OUTPUT_X_MAX_CM - OUTPUT_X_MIN_CM) * PIXELS_PER_CM))
    output_h = int(round(extended_y_max_cm * PIXELS_PER_CM))
    return output_w, output_h


def _physical_to_output_px_extended(physical_pts, extended_y_max_cm):
    pts = np.asarray(physical_pts, dtype=np.float64)
    ox = (pts[:, 0] - OUTPUT_X_MIN_CM) * PIXELS_PER_CM
    oy = (extended_y_max_cm - pts[:, 1]) * PIXELS_PER_CM
    return np.column_stack([ox, oy])


def _to_output_px_extended(x_cm, y_cm, extended_y_max_cm):
    px = (x_cm - OUTPUT_X_MIN_CM) * PIXELS_PER_CM
    py = (extended_y_max_cm - y_cm) * PIXELS_PER_CM
    return px, py


def _compute_crop_row_range_extended(full_height_px, center_y_cm, half_height_cm, extended_y_max_cm):
    """KRIITTINEN BUGIKORJAUS (kayttajan pyynnosta - katso keskustelu-
    historia): compute_crop_row_range/crop_house_view/expected_house_
    center_in_crop kayttavat SISAISESTI k8:n omaa to_output_px:aa, joka
    on kiinteasti sidottu STANDARDIIN OUTPUT_Y_MAX_CM:aan (4000cm) -
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

    blue = create_blue_mask(topdown_extended)
    red = create_red_mask(topdown_extended)
    h, w = blue.shape
    center_col = int(round((0.0 - OUTPUT_X_MIN_CM) * PIXELS_PER_CM))
    near_house_exclude_row = int(h - (HOUSE_RADIUS_CM + 100.0) * PIXELS_PER_CM)

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

    blue_raw = create_blue_mask(crop)
    red_raw = create_red_mask(crop)
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

    back_line_y_cm = FAR_HOUSE_Y_CM + HOUSE_RADIUS_CM
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
    target_row = (extended_y_max_cm - FAR_HOUSE_Y_CM) * PIXELS_PER_CM
    anchor_row = float(ext_h)
    scale_factor = (anchor_row - target_row) / (anchor_row - found_row)
    a = scale_factor
    b = anchor_row * (1.0 - scale_factor)
    rescale_M = np.array([[1.0, 0.0, 0.0], [0.0, a, b]], dtype=np.float64)
    topdown_rescaled_ext = cv2.warpAffine(topdown_extended, rescale_M, (ext_w, ext_h))

    # Rivi<->Y-suhde on nyt (affiinin rakentamistavan ansiosta) sama
    # VAKIOKAAVA koko kuvan matkalta kuin standardikanvaasilla - katso
    # keskusteluhistorian perustelu. HUOM (loydetty ja korjattu kayttajan
    # pyynnosta nayttaessa jokaisen vaiheen kuvat): compute_crop_row_
    # range/crop_house_view/expected_house_center_in_crop kayttavat
    # SISAISESTI k8:n OMAA to_output_px:aa, joka on kiinteasti sidottu
    # STANDARDIIN OUTPUT_Y_MAX_CM:aan (4000cm) - EIVAT kelpaa tälle
    # LAAJENNETULLE (eri korkuiselle) kanvaasille sellaisenaan. Kaytetaan
    # siis omia extended-versioita (_compute_crop_row_range_extended jne,
    # parametrisoitu oikealla extended_y_max_cm:lla).
    far_row_top_ext, _ = _compute_crop_row_range_extended(
        ext_h, FAR_HOUSE_Y_CM, HOUSE_CROP_HALF_HEIGHT_CM, extended_y_max_cm
    )
    far_crop_rescaled = _crop_house_view_extended(
        topdown_rescaled_ext, FAR_HOUSE_Y_CM, HOUSE_CROP_HALF_HEIGHT_CM, extended_y_max_cm
    )
    expected_center_local = _expected_house_center_in_crop_extended(
        ext_h, FAR_HOUSE_Y_CM, HOUSE_CROP_HALF_HEIGHT_CM, extended_y_max_cm
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

    output_w = int(round((OUTPUT_X_MAX_CM - OUTPUT_X_MIN_CM) * PIXELS_PER_CM))
    output_h = int(round(OUTPUT_Y_MAX_CM * PIXELS_PER_CM))

    all_img_pts = list(near_pts_frame) + [far_center_frame]
    all_phys_pts = list(near_phys_pts) + [(0.0, FAR_HOUSE_Y_CM)]
    src_v1 = np.array(all_img_pts, dtype=np.float32)
    dst_v1 = physical_to_output_px(np.array(all_phys_pts, dtype=np.float64)).astype(np.float32)
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
    """Sama sopimus kuin far_house_ring_points_in_frame (palauttaa
    (points, radii) oikaistun kuvan koordinaatistossa geometrista
    tarkennusta - solve_homography_geometric - varten), mutta
    tunnistus tehdaan VARIPOHJAISELLA (HSV) menetelmalla k8:n omaan
    radiaalihaku+pakotettu-yhteismuoto-sovitukseen sijaan: pinta-
    alaosuuteen (ROI = sinisen maskin pienin ymparoiva ympyra) saadettu
    Saturation-kynnys + sateittainen reunanhaku molemmille renkaille
    (sininen ulko/sisa, punainen ulko). Kayttajan kanssa todettu
    tuottavan visuaalisesti tarkemman (oikean kokoisen) sovituksen
    kuin k8:n oma menetelma talla kamerakulmalla."""

    far_row_top, _ = compute_crop_row_range(
        topdown_raw.shape[0], FAR_HOUSE_Y_CM, HOUSE_CROP_HALF_HEIGHT_CM
    )
    view = crop_house_view(topdown_raw, FAR_HOUSE_Y_CM, HOUSE_CROP_HALF_HEIGHT_CM)

    # Sama vahvistus kuin _three_ellipse_fit_center_in_crop:issa
    # (kayttajan pyynnosta): ei luoteta pelkkaan "suurin sininen alue"
    # -oletukseen, koska nykyinen H_current voi silla hetkella olla
    # viela riittavan vino etta crop osuu vaaraan kohteeseen (esim.
    # mainospaneeliin) - vahvistetaan ETTA sininen-punainen-sininen
    # -kuvio loytyy paikallisesti ENNEN kuin renkaan pisteita palautetaan
    # geometriselle ratkaisijalle. Jos kuviota ei loydy, palautetaan
    # TYHJA (turvallisempi kuin vaara rengas - silloin ratkaisija
    # nojaa vain lahempaan pesaan + hoglineihin talla kierroksella).
    blue_raw = create_blue_mask(view)
    red_raw = create_red_mask(view)
    center_col = int(round((0.0 - OUTPUT_X_MIN_CM) * PIXELS_PER_CM))
    pattern = _find_blue_red_blue_pattern(blue_raw, red_raw, center_col)
    if pattern is None:
        return np.zeros((0, 2)), np.zeros(0)
    pattern_row, pattern_col = pattern

    ys, xs = np.where(blue_raw > 0)
    if len(xs) < 5:
        return np.zeros((0, 2)), np.zeros(0)
    pts_all = np.column_stack([xs, ys]).astype(np.float32)
    dists = np.hypot(pts_all[:, 0] - pattern_col, pts_all[:, 1] - pattern_row)
    near_pattern = pts_all[dists <= HOUSE_CROP_HALF_HEIGHT_CM * PIXELS_PER_CM * 0.6]
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
        (blue_inner_pts, BLUE_INNER_RADIUS_CM),
        (blue_outer_pts, BLUE_OUTER_RADIUS_CM),
        (red_outer_pts, RED_OUTER_RADIUS_CM),
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

    points = frame_points_from_topdown(np.array(topdown_pts, dtype=np.float64), H_current)
    return points, np.array(radii, dtype=np.float64)


def hogline_points_gray_threshold(topdown_raw, hogline_y_cm, H_current,
                                   half_band_cm=HOGLINE_GRAY_HALF_BAND_CM):
    """Sama sopimus kuin hogline_points_in_frame (palauttaa (points,
    weights) oikaistun kuvan koordinaatistossa), mutta harmaasavy-
    kynnys-menetelmalla: haetaan nimellisen hogline-rivin ymparilta
    (half_band_cm) kynnysarvo (harmaasavy < t) joka antaa YHTENAISEN
    (lahes 100% sarakepeittavyyden) tumman kaistan koko leveydelta -
    kayttajan kanssa todettu paljon luotettavammaksi kuin k8:n oma
    segmenttipohjainen detect_hogline_points talla kamerakulmalla
    (joka loysi usein vain muutaman hajanaisen pisteen tai tarttui
    sponsoritekstin kontaminaatioon)."""

    _, nominal_row = to_output_px(0.0, hogline_y_cm)
    half_px = int(half_band_cm * PIXELS_PER_CM)
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

    points = frame_points_from_topdown(topdown_pts, H_current)
    return points, weights_norm


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


# Kayttajan pyynnosta: debug-ominaisuus joka tallentaa elavan moni-
# kiven seurannan ajalta UUDEN videotiedoston (<video>_debug_seuranta.
# mp4), jossa jokaisen tunnistetun/seuratun kiven ENNUSTETTU ääriviiva
# (predicted_stone_hull_fast - sama profiilimalli jota itse
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


MAX_POSITION_ERROR = 20.0


REPORT_EVERY = 1000


MAX_WORKERS = 8


ROI_HALF_WIDTH = 100


ROI_HALF_HEIGHT = 120


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
# sadetasoleikkauksella fyysisiin rajoihin OUTPUT_X/Y_MIN/MAX_CM:aan,
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


# HUOM: find_stone_candidates suodattaa jo OLETUKSENA koko radan
# fyysisiin rajoihin (OUTPUT_X/Y_MIN/MAX_CM + sheet_margin_cm) -
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
# sovitusta (fit_stone_profile) YRITETAAN jokaisen uuden loydetyn
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
# VAIN kaukaisen paan kiinnealta vyohykkeelta (SEARCH_Y_MIN/MAX_CM
# = kaukainen hogline...hogline+3m, SEARCH_X_HALF_WIDTH_CM=
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
#      find_stone_candidates:in kommentti) - LIIKKUVA ei-kivi
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
# track_stone_in_video/track_stone_in_video_fast kayttavat SISAISESTI
# (_candidates_in_frame) - EI mallipohjaista ristikkohakua, koska
# tassa vaiheessa 3D-profiilia (jota malli tarvitsisi) EI VIELA OLE -
# se on juuri se mita etsitaan.
#
# find_moving_candidate: vertaa KAHDEN PERAKKAISEN skannauksen (n.
# STONE_SCAN_INTERVAL_SECONDS valein) kandidaattilistoja - sama fyysinen
# kivi (lahin osuma) jonka sijainti on muuttunut enemman kuin STONE_
# MOTION_THRESHOLD_CM tulkitaan AIDOSTI LIIKKUVAKSI (ei jo-paikallaan-
# olevaksi) - tama siirtymä-havainto ANTAA SIEMENEN (frame_idx, X, Y)
# track_stone_in_video_fast:lle, joka sitten seuraa koko liu'un.
#
# suppress_static_background: create_granite_mask:in kommentti
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

    return background_mask | shadow_mask | ice_mask


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

    for channel in range(3):

        gain, bias = _robust_gain_bias_single_channel(
            frame_bgr[:, :, channel].astype(np.float64).ravel(),
            reference_bgr[:, :, channel].astype(np.float64).ravel()
        )

        gains.append(gain)
        biases.append(bias)

    return gains, biases


def apply_photometric_correction(frame_bgr, gains, biases):

    out = frame_bgr.astype(np.float32).copy()

    for channel in range(3):
        out[:, :, channel] = (
            out[:, :, channel] * gains[channel] + biases[channel]
        )

    return np.clip(out, 0, 255).astype(np.uint8)


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


# ============================================================
# AIKAIKKUNAAN RAJATTU KIVEN SEURANTA (EI koko videota)
#
# track_stone_in_video_fast (kamera9_04.py, EI kosketa) skannaa
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
        TRACK_LOST_MAX_MISSES, int(round(fps * TRACK_LOST_GRACE_SECONDS))
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
                dx, dy = _phase_correlate_full_frame(
                    loppuvideo_ref_gray, gray
                )

                stabilization_matrix = np.array(
                    [[1.0, 0.0, -dx], [0.0, 1.0, -dy]],
                    dtype=np.float64
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

                    pose = build_pose_from_calibration(calib)

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

                    map1, map2 = _build_undistort_maps(
                        camera_matrix, dist_coeffs, (width, height)
                    )

                    local_pts_body = build_local_stone_rings(
                        R_max, H_total, shape_deltas, n_theta=28, n_per_segment=5
                    )
                    local_pts_search = build_local_stone_rings(
                        R_max, H_total, shape_deltas,
                        n_theta=SEARCH_HULL_N_THETA,
                        n_per_segment=SEARCH_HULL_N_PER_SEGMENT
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

                # VALOTASAPAINO/KIRKKAUS - vain kerran sekunnissa (katso
                # estimate_photometric_correction:in kommentti) - ei
                # kosketa frame_u:ta itsea (varitarkistus/debug-video),
                # vain erillinen frame_u_photo-kopio taustanvaimennukselle.
                if (
                    photo_gain is None
                    or frame_index >= next_photo_update_frame
                ):
                    t_photo0 = time.perf_counter()
                    photo_gain, photo_bias = estimate_photometric_correction(
                        frame_u, ref_undist_live
                    )
                    total_photometric_time += time.perf_counter() - t_photo0
                    next_photo_update_frame = frame_index + max(
                        1, int(round(fps))
                    )

                frame_u_photo = apply_photometric_correction(
                    frame_u, photo_gain, photo_bias
                )

                if ENABLE_SHADOW_TOLERANT_STABILIZATION:
                    t_shadow0 = time.perf_counter()
                    frame_u_for_tracking = suppress_static_background(
                        frame_u_photo, ref_undist_live, diff_threshold=GRANITE_DIFF_THRESHOLD
                    )
                    total_shadow_suppress_time += time.perf_counter() - t_shadow0
                    # frame_u_for_tracking on jo taustanvaimennettu (myos
                    # varjonsietoisesti) - C++:n OMA sisainen vaimennus
                    # HAKU/SEURANTA-kutsuissa ohitetaan antamalla sille 0.0,
                    # jotta frame_u_for_tracking:ia ei vaimenneta uudelleen.
                    haku_seuranta_diff_threshold = 0.0
                else:
                    frame_u_for_tracking = frame_u_photo
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
                        SEARCH_Y_MIN_CM + SEARCH_Y_MAX_CM
                    ) / 2.0
                    y_half = (
                        SEARCH_Y_MAX_CM - SEARCH_Y_MIN_CM
                    ) / 2.0

                    haku_future = haku_executor.submit(
                        _run_haku_timed,
                        frame_u_for_tracking, ref_undist_live,
                        local_pts_body, local_pts_search,
                        pose["K"], pose["R"], pose["t"],
                        x_center, SEARCH_X_HALF_WIDTH_CM, y_center, y_half,
                        SEARCH_COARSE_STEP_CM, SEARCH_FINE_STEP_CM,
                        SEARCH_SCORE_THRESHOLD,
                        live_state["R_max"], live_state["H_total"],
                        live_state["ring_r_frac_guess"], live_state["handle_r_frac"],
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
                # locate_by_grid_search_fast + refine_
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
                        TRACK_COARSE_STEP_CM, TRACK_FINE_STEP_CM,
                        TRACK_SCORE_THRESHOLD,
                        live_state["R_max"], live_state["H_total"],
                        live_state["ring_r_frac_guess"], live_state["handle_r_frac"],
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

                                    if (
                                        n_pending >= MIN_PRECONFIRM_TARKKA_OBSERVATIONS
                                        and tarkka_frac < MIN_PRECONFIRM_TARKKA_FRACTION
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

                                        for pf, pt, prow in s["pending_rows"]:
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
                            if s["confirmed"]
                        )

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
                            )

                # --------------------------------------------
                # DEBUG_SAVE_TRACKING_VIDEO: piirretaan taman framen
                # kaikkien HAKU/SEURANTA-havaintojen (debug_draw_items,
                # katso yllapuoliset kohdat) ENNUSTETUT ääriviivat
                # (predicted_stone_hull_fast - sama profiilimalli
                # jota itse yhteissovituskin kayttaa) frame_u:n paalle
                # ja kirjoitetaan debug-videoon. Katso taman tiedoston
                # alkupaan DEBUG_SAVE_TRACKING_VIDEO-kommentti varien
                # merkityksesta.
                # --------------------------------------------

                if debug_video_writer is not None:

                    # Kayttajan pyynnosta: debug-videoon tallennetaan
                    # moodikuvalla suodatettu frame_u_for_tracking (se
                    # mita HAKU/SEURANTA oikeasti NAKEE), EI alkuperaista
                    # frame_u:ta - nain debug-videosta voi suoraan
                    # tarkistaa mita tunnistus itse asiassa kaytti.
                    debug_frame = frame_u_for_tracking.copy()

                    for bx, by, s_id, color, label in debug_draw_items:

                        hull = predicted_stone_hull_fast(
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
                    f"stabilointimatriisi {(total_stabilize_compute_time / processed) * 1000:.2f} ms/ruutu | "
                    f"warpAffine+remap(koko frame) {(total_warp_remap_time / processed) * 1000:.2f} ms/ruutu | "
                    f"valotasapaino {(total_photometric_time / processed) * 1000:.2f} ms/ruutu | "
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
