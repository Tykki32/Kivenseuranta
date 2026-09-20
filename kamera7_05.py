import cv2
import numpy as np
import math
import io
import contextlib
import tkinter as tk
from tkinter import filedialog


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

TOPDOWN_OUTPUT = "curling_topdown.png"
NEAR_HOUSE_VIEW_OUTPUT = "near_house_topdown.png"
FAR_HOUSE_VIEW_OUTPUT = "far_house_topdown.png"
MASKS_DEBUG_OUTPUT = "masks_debug.png"
DEBUG_FAR_SEARCH_OUTPUT = "far_house_search_debug.png"
DEBUG_POINTS_OUTPUT = "homography_points_debug.png"
FAR_HOUSE_VERIFY_OUTPUT = "far_house_topdown_verify.png"
NEAR_HOUSE_VERIFY_OUTPUT = "near_house_topdown_verify.png"
CORRECTED_TOPDOWN_OUTPUT = "curling_topdown_corrected.png"

FONT = cv2.FONT_HERSHEY_SIMPLEX

HOUSE_CROP_HALF_HEIGHT_CM = 300.0

# Hogline (etulinja) sijaitsee curlingissa 6.4 m paassa T-linjalta
# (pesan keskipisteesta), molemmin puolin kenttaa.
HOG_LINE_DISTANCE_FROM_TEE_CM = 640.0
NEAR_HOG_LINE_Y_CM = NEAR_HOUSE_Y_CM + HOG_LINE_DISTANCE_FROM_TEE_CM
FAR_HOG_LINE_Y_CM = FAR_HOUSE_Y_CM - HOG_LINE_DISTANCE_FROM_TEE_CM

FAR_HUE_TOLERANCE = 30

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
# KAUKAISEN PESAN ELLIPSIEN HAKUASETUKSET (HUE-MASKEISTA)
# ============================================================

FAR_BLUE_MIN_AREA = 20
FAR_BLUE_MIN_RATIO = 0.15
FAR_BLUE_MIN_SIZE_RATIO = 0.15
FAR_BLUE_MIN_CONTOUR_LEN = 6

FAR_RED_MIN_AREA = 8
FAR_RED_MIN_RATIO = 0.12
FAR_RED_MIN_SIZE_RATIO = 0.12
FAR_RED_MIN_CONTOUR_LEN = 6

# ============================================================
# HAKUASETUKSET - KAUKAISEN PESAN LOYTAMINEN
# (coarse -> fine -> ultra fine, alkuperaisen menetelman mukaan)
# ============================================================

SEARCH_CENTER_RANGE_CM = 200.0
SEARCH_CENTER_RANGE_DEG = 30
SEARCH_CENTER_RANGE_SCALE = 0.5

Muutos = 5.0

COARSE_CENTER_STEP_CM = SEARCH_CENTER_RANGE_CM / Muutos
COARSE_ANGLE_STEP_DEG = SEARCH_CENTER_RANGE_DEG / Muutos
COARSE_SCALE_STEP = SEARCH_CENTER_RANGE_SCALE / Muutos

FINE_CENTER_RANGE_CM = COARSE_CENTER_STEP_CM * 2
FINE_CENTER_STEP_CM = FINE_CENTER_RANGE_CM / Muutos
FINE_ANGLE_RANGE_DEG = COARSE_ANGLE_STEP_DEG * 2
FINE_ANGLE_STEP_DEG = FINE_ANGLE_RANGE_DEG / Muutos
FINE_SCALE_RANGE = COARSE_SCALE_STEP * 2
FINE_SCALE_STEP = FINE_SCALE_RANGE / Muutos

ULTRA_CENTER_RANGE_CM = FINE_CENTER_STEP_CM * 2
ULTRA_CENTER_STEP_CM = ULTRA_CENTER_RANGE_CM / Muutos
ULTRA_ANGLE_RANGE_DEG = FINE_ANGLE_STEP_DEG * 2
ULTRA_ANGLE_STEP_DEG = ULTRA_ANGLE_RANGE_DEG / Muutos
ULTRA_SCALE_RANGE = FINE_SCALE_STEP * 2
ULTRA_SCALE_STEP = ULTRA_SCALE_RANGE / Muutos

MAX_BLUE_TEMPLATE_POINTS = 1200
MAX_RED_TEMPLATE_POINTS = 1000
MAX_BACKGROUND_TEMPLATE_POINTS = 1000


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

# NOPEUSOPTIMOINTI (kamera7_02.py): trial_final_rms:in (kaukaisen pesan
# korrespondenssikandidaatin ALUSTAVA vertailu, katso sen kommentti)
# kayttama pienempi iteraatiokatto - varsinainen ("oikea") ajo
# voittaneelle kandidaatille kayttaa edelleen taytta
# HOMOGRAPHY_REFINE_MAX_ITERATIONS-arvoa.
TRIAL_MAX_ITERATIONS = 6


# ============================================================
# GEOMETRIA-APURIT
# ============================================================

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


def ellipse_axis_points(ellipse):
    """
    Palauttaa ellipsin keskipisteen seka leveys- ja korkeusakselin
    paatepisteet (kuvatason pikselikoordinaateissa).

    OpenCV:n RotatedRect-konventio: "width"-akselin suunta on
    (cos(angle), sin(angle)) ja "height"-akselin suunta
    (-sin(angle), cos(angle)).
    """

    (cx, cy), (w, h), angle_deg = ellipse

    theta = math.radians(angle_deg)

    width_dir = np.array([math.cos(theta), math.sin(theta)], dtype=np.float64)
    height_dir = np.array([-math.sin(theta), math.cos(theta)], dtype=np.float64)

    center = np.array([cx, cy], dtype=np.float64)

    half_w = w / 2.0
    half_h = h / 2.0

    width_pts = (center + width_dir * half_w, center - width_dir * half_w)
    height_pts = (center + height_dir * half_h, center - height_dir * half_h)

    return center, width_dir, width_pts, height_dir, height_pts


def ellipse_to_correspondences(ellipse, radius_cm, dir_lateral, dir_forward,
                                base_y_cm, name, only_lateral=False):
    """
    Muuttaa yhden ellipsin fyysisiksi korrespondensseiksi laskemalla,
    missa KIINTEASUUNTAINEN (dir_lateral / dir_forward) suora ellipsin
    keskipisteen kautta leikkaa ellipsin reunan - EI ellipsin omia
    (mahdollisesti kiertyneita) leveys-/korkeusakseleita.

    Tama on tarkeaa: jos ellipsi on kiertynyt (esim. pieni jaljella
    oleva kalibrointivirhe), sen oma leveys-/korkeusakseli EI enaa
    osu yhteen T-linjan/keskilinjan suunnan kanssa, ja akselin
    paatepisteen kayttaminen sellaisenaan antaisi systemaattisesti
    vaaran (sivuun karahtaneen) pisteen. Suora-ellipsi-leikkaus
    antaa oikean pisteen kaikissa tapauksissa (sama menetelma kuin
    lahemman pesan alkuperaisessa T-linja/keskilinja-tunnistuksessa).

    only_lateral=True: palauttaa VAIN lateraaliset (vasen/oikea,
    T-linjan suuntaiset) pisteet - ei etaisyyssuunnan (lahi/kauka)
    pisteita.
    """

    center = np.array(ellipse[0], dtype=np.float64)

    lateral_line = line_from_points(center, center + dir_lateral)
    lateral_pts = line_ellipse_intersections(lateral_line, ellipse)

    image_pts = []
    physical_pts = []
    labels = []

    for p in lateral_pts:

        p = np.asarray(p, dtype=np.float64)
        d = float(np.dot(p - center, dir_lateral))
        x_phys = radius_cm if d > 0 else -radius_cm

        image_pts.append(p)
        physical_pts.append((x_phys, base_y_cm))
        labels.append(f"{name}_LAT_{'POS' if d > 0 else 'NEG'}")

    if only_lateral:
        return image_pts, physical_pts, labels

    forward_line = line_from_points(center, center + dir_forward)
    forward_pts = line_ellipse_intersections(forward_line, ellipse)

    for p in forward_pts:

        p = np.asarray(p, dtype=np.float64)
        d = float(np.dot(p - center, dir_forward))
        y_phys = base_y_cm + radius_cm if d > 0 else base_y_cm - radius_cm

        image_pts.append(p)
        physical_pts.append((0.0, y_phys))
        labels.append(f"{name}_FWD_{'FAR' if d > 0 else 'NEAR'}")

    return image_pts, physical_pts, labels


def ellipse_lateral_correspondences(ellipse, radius_cm, dir_lateral, base_y_cm, name):
    """
    Sama kuin ellipse_to_correspondences(only_lateral=True): palauttaa
    VAIN T-linjan suuntaiset (lateraaliset, vasen/oikea) pisteet -
    laskettuna kiintean dir_lateral-suoran ja ellipsin leikkauksena
    (ei ellipsin omasta, mahdollisesti kiertyneesta akselista).

    Kaytetaan kaukaiselle pesalle, jolta halutaan vain T-linjalla
    olevat pisteet (ulkokehan vasen ja oikea reuna).
    """

    return ellipse_to_correspondences(
        ellipse, radius_cm, dir_lateral, dir_lateral, base_y_cm, name,
        only_lateral=True
    )


# ============================================================
# VARIMASKIT
# ============================================================

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


def robust_circle_fit(points, keep_frac=0.7, iterations=4, min_points=20):
    """
    Robusti ympyransovitus find_ring_edge_points:in tuottamiin
    (x, y, edge_strength) -pisteisiin:

      1. Sailytetaan aluksi vain voimakkaimmat (keep_frac) reunapisteet
         (heikoimmat ovat todennakoisimmin kohinaa/sahalaitaa).
      2. Sovitetaan ympyra (Kasa), lasketaan jokaisen pisteen etaisyys
         sovitetusta ympyranreunasta, ja hylataan poikkeavat pisteet
         (mediaani + 3*MAD), toistetaan muutama kierros.

    Palauttaa (cx, cy, r) tai None, jos pisteita on liian vahan.
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


def fit_concentric_circles_shared_center(
    point_groups, init_center, iterations=25, step=0.5
):
    """
    point_groups: lista Nx2-pistetaulukoita (esim. sinisen ja punaisen
    ULKOKEHAN reunapisteet erikseen) - KAIKKI oletetaan samankeskisiksi
    (sama TODELLINEN keskipiste), mutta kullakin on oma (tuntematon)
    sateensa.

    PERUSTELU: pesan eri renkaat (sininen/punainen ulkokeha) ovat
    FYYSISESTI aina samankeskiset. Jos ne sovitetaan itsenaisesti
    (oma keskipiste kummallekin), pienikin kohina (osapikselitasolla,
    etenkin kaukaisella/pienella/sumealla pesalla) voi siirtaa niiden
    sovitettuja keskipisteita toisistaan poikkeaviin suuntiin -
    havaittu testatessa: jopa >10 px ero blue_outer:in ja red_outer:in
    itsenaisesti sovitettujen keskipisteiden valilla. YHTEISEN
    keskipisteen pakottaminen kayttaa KAIKKIEN renkaiden pisteita
    saman 2 vapausasteen (cx,cy) rajoittamiseen - huomattavasti
    kohinankestavampi kuin kaksi itsenaista 3 vapausasteen sovitusta.

    Ratkaistaan vuorottelevalla minimoinnilla (kiinnitetaan sateet ->
    gradienttiaskel keskipisteeseen -> toista), koska ongelma ei ole
    lineaarinen (Kasa-tyylinen suljetun muodon ratkaisu ei suoraan
    toimi kun sateet eivat ole samat).

    Palauttaa (cx, cy, [r_per_group]).
    """

    cx, cy = float(init_center[0]), float(init_center[1])

    for _ in range(iterations):

        radii = []

        for pts in point_groups:
            d = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
            radii.append(float(np.mean(d)))

        gx = 0.0
        gy = 0.0
        n_total = 0

        for pts, r in zip(point_groups, radii):

            dx = cx - pts[:, 0]
            dy = cy - pts[:, 1]
            d = np.maximum(np.hypot(dx, dy), 1e-6)
            coeff = (d - r) / d

            gx += float(np.sum(coeff * dx))
            gy += float(np.sum(coeff * dy))
            n_total += len(pts)

        if n_total == 0:
            break

        cx -= step * gx / n_total
        cy -= step * gy / n_total

    radii = []

    for pts in point_groups:
        d = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
        radii.append(float(np.mean(d)))

    return cx, cy, radii


def detect_far_house_concentric_circles(view, expected_center):
    """
    Tunnistaa kaukaisen pesan sinisen ja punaisen ULKOKEHAN YHTEISELLA
    (pakotetulla samankeskisyydella) - katso
    fit_concentric_circles_shared_center:in perustelu. Kaytetaan
    ITERATIIVISEN korjauksen VALIVAIHEIDEN korrespondenssipisteiden
    lahteena (katso refine_homography_corrections) - katso siella
    oleva kommentti siita miksi rajoitettu YMPYRA (ei yhteismuotoinen
    ellipsi) sopii tahan paremmin (rotaatioinvarianssi).

    Palauttaa dictin jossa "blue_outer"/"red_outer" (ellipsi-muodossa,
    ymparysta kayttavat pyoreat sateet) tai None jos ei loytynyt.
    """

    blue_score, red_score = create_topdown_score_maps(view)

    result = {"blue_outer": None, "blue_inner": None, "red_outer": None, "red_inner": None}

    blue_points = find_ring_edge_points(
        blue_score, expected_center, BLUE_OUTER_RADIUS_CM * PIXELS_PER_CM
    )
    red_points = find_ring_edge_points(
        red_score, expected_center, RED_OUTER_RADIUS_CM * PIXELS_PER_CM
    )

    blue_fit = robust_circle_fit_with_inliers(blue_points)
    red_fit = robust_circle_fit_with_inliers(red_points)

    groups = []
    center_estimates = []
    ring_names = []

    if blue_fit is not None:
        groups.append(blue_fit[3])
        center_estimates.append(np.array([blue_fit[0], blue_fit[1]]))
        ring_names.append("blue_outer")

    if red_fit is not None:
        groups.append(red_fit[3])
        center_estimates.append(np.array([red_fit[0], red_fit[1]]))
        ring_names.append("red_outer")

    if not groups:
        return result

    # Painotetaan alkuarvaus pistemaaran mukaan (isompi/luotettavampi
    # rengas - yleensa sininen ulkokeha - saa enemman painoarvoa).
    weights = np.array([len(g) for g in groups], dtype=np.float64)
    init_center = np.average(np.array(center_estimates), axis=0, weights=weights)

    cx, cy, radii = fit_concentric_circles_shared_center(groups, init_center)

    for name, r in zip(ring_names, radii):
        result[name] = ((cx, cy), (2.0 * r, 2.0 * r), 0.0)

    return result


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


def detect_precise_circle(
    score, expected_center, expected_radius_px,
    radius_tol=0.25, num_angles=720, r_step=0.5
):
    """
    Yhdistaa find_ring_edge_points + robust_circle_fit: etsii YHDEN
    ympyran (pesan renkaan reunan) osapikselitarkkuudella annetusta
    pisteytyskartasta, kayttaen fyysisesta mittakaavasta tiedettya
    odotettua sadetta ja keskipistetta priorina.

    Palauttaa (cx, cy, r) koko kuvan koordinaateissa, tai None jos
    reunaa ei loytynyt luotettavasti.
    """

    points = find_ring_edge_points(
        score, expected_center, expected_radius_px,
        radius_tol=radius_tol, num_angles=num_angles, r_step=r_step
    )

    return robust_circle_fit(points)


def circle_to_ellipse(circle):

    if circle is None:
        return None

    cx, cy, r = circle

    return ((cx, cy), (2.0 * r, 2.0 * r), 0.0)


def _fit_ellipse_from_points(points):
    """
    Sovittaa yleisen (ei-ympyra-rajoitetun) ellipsin pistejoukkoon
    cv2.fitEllipse:lla. Vaatii vahintaan 5 pistetta.
    """

    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)

    return cv2.fitEllipse(pts)


def _ellipse_point_residual(point, ellipse):
    """
    Likimaarainen etaisyys (pikseleissa) pisteesta sovitetun ellipsin
    reunalle: muunnetaan piste ellipsin paikalliseen (keskitettyyn,
    kierrettyyn) koordinaatistoon, normalisoidaan puoliakseleilla, ja
    skaalataan normalisoitu poikkeama takaisin pikseliyksikoihin
    keskimaaraisella sateella. Riittavan tarkka poikkeavien pisteiden
    (outlierien) hylkaamiseen iteratiivisessa sovituksessa.
    """

    (cx, cy), (w, h), angle_deg = ellipse

    a = max(w / 2.0, 1e-6)
    b = max(h / 2.0, 1e-6)

    theta = math.radians(angle_deg)

    dx = point[0] - cx
    dy = point[1] - cy

    local_x = dx * math.cos(theta) + dy * math.sin(theta)
    local_y = -dx * math.sin(theta) + dy * math.cos(theta)

    norm = math.sqrt((local_x / a) ** 2 + (local_y / b) ** 2)

    return abs(norm - 1.0) * ((a + b) / 2.0)


def robust_ellipse_fit(points, keep_frac=0.7, iterations=4, min_points=20):
    """
    Sama periaate kuin robust_circle_fit, mutta sovittaa YLEISEN
    ellipsin (ei rajoitettu ympyraksi). Kayttokelpoinen esim.
    lahemmalle pesalle, jos top-down-kuvassa on jaljella pieni
    perspektiivi-/kalibrointivirhe, joka nakyy lievana soikeutena -
    talloin ellipsisovitus osuu tarkemmin kuin ympyrasovitus.

    Palauttaa ((cx,cy),(w,h),angle) tai None.
    """

    if len(points) < max(min_points, 5):
        return None

    pts = [(p[0], p[1]) for p in points]
    strengths = np.array([p[2] for p in points])

    order = np.argsort(strengths)[::-1]
    n_keep = max(min_points, int(len(pts) * keep_frac))

    selected = [pts[i] for i in order[:n_keep]]

    ellipse = _fit_ellipse_from_points(selected)

    for _ in range(iterations):

        dists = np.array([
            _ellipse_point_residual(p, ellipse) for p in selected
        ])

        median = np.median(dists)
        mad = np.median(np.abs(dists - median)) + 1e-6

        keep = dists < median + 3.0 * mad

        if keep.sum() < max(min_points, 5):
            break

        selected = [selected[i] for i in range(len(selected)) if keep[i]]
        ellipse = _fit_ellipse_from_points(selected)

    return ellipse


def detect_precise_ellipse(
    score, expected_center, expected_radius_px,
    radius_tol=0.25, num_angles=720, r_step=0.5
):
    """
    Sama kuin detect_precise_circle, mutta sovittaa ympyran sijaan
    yleisen ellipsin (find_ring_edge_points + robust_ellipse_fit).

    Palauttaa ((cx,cy),(w,h),angle) tai None.
    """

    points = find_ring_edge_points(
        score, expected_center, expected_radius_px,
        radius_tol=radius_tol, num_angles=num_angles, r_step=r_step
    )

    return robust_ellipse_fit(points)


def _fit_ellipse_cv(points):

    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)

    return cv2.fitEllipse(pts)


def _ellipse_point_residual(ellipse, x, y):
    """
    Likimaarainen etaisyys (pikseleina) pisteesta (x,y) annetun
    ellipsin reunaan: normalisoitu sade ellipsin omassa (kierretyssa)
    koordinaatistossa, skaalattuna takaisin pikseleiksi keskisateella.
    """

    (cx, cy0), (w, h), angle_deg = ellipse

    theta = math.radians(angle_deg)

    dx = x - cx
    dy = y - cy0

    local_x = dx * math.cos(theta) + dy * math.sin(theta)
    local_y = -dx * math.sin(theta) + dy * math.cos(theta)

    a = w / 2.0
    b = h / 2.0

    normalized_radius = math.sqrt((local_x / a) ** 2 + (local_y / b) ** 2)

    return abs(normalized_radius - 1.0) * ((a + b) / 2.0)


def robust_ellipse_fit(points, keep_frac=0.7, iterations=4, min_points=20):
    """
    Sama periaate kuin robust_circle_fit, mutta sovittaa YLEISEN
    ellipsin (cv2.fitEllipse) ympyran sijaan. Kayttokelpoinen kun
    kohde ei ole taydellinen ympyra edes top-down-nakymassa - esim.
    pieni jaljella oleva perspektiivi-/kalibrointivirhe litistaa tai
    kiertaa lahempaa pesaa hieman.
    """

    if len(points) < min_points:
        return None

    pts = [(p[0], p[1]) for p in points]
    strengths = np.array([p[2] for p in points])

    order = np.argsort(strengths)[::-1]
    n_keep = max(min_points, int(len(pts) * keep_frac))

    selected = [pts[i] for i in order[:n_keep]]

    ellipse = _fit_ellipse_cv(selected)

    for _ in range(iterations):

        residuals = np.array([
            _ellipse_point_residual(ellipse, x, y) for x, y in selected
        ])

        median = np.median(residuals)
        mad = np.median(np.abs(residuals - median)) + 1e-6

        keep = residuals < median + 3.0 * mad

        if keep.sum() < min_points:
            break

        selected = [selected[i] for i in range(len(selected)) if keep[i]]
        ellipse = _fit_ellipse_cv(selected)

    return ellipse


def detect_precise_ellipse(
    score, expected_center, expected_radius_px,
    radius_tol=0.25, num_angles=720, r_step=0.5
):
    """
    Sama kuin detect_precise_circle, mutta palauttaa yleisen
    ellipsin (cx,cy),(leveys,korkeus),kulma - suoraan kayttovalmiina
    esim. ellipse_to_correspondences -funktiolle.
    """

    points = find_ring_edge_points(
        score, expected_center, expected_radius_px,
        radius_tol=radius_tol, num_angles=num_angles, r_step=r_step
    )

    return robust_ellipse_fit(points)


def shift_ellipse(ellipse, dx=0.0, dy=0.0):
    """
    Siirtaa ellipsin keskipistetta (dx,dy) verran - kayttokelpoinen
    kun ellipsi on tunnistettu leikatusta (crop) kuvasta ja se pitaa
    muuntaa takaisin koko top-down-kuvan koordinaatteihin.
    """

    if ellipse is None:
        return None

    (cx, cy), (w, h), angle = ellipse

    return ((cx + dx, cy + dy), (w, h), angle)



# ============================================================
# HUE-APURIT (kaytetaan seka lahemman pesan templatessa etta
# kaukaisen pesan hue-mediaanimaskeissa)
# ============================================================

def circular_hue_distance(hue, center):
    """
    Etaisyys HSV Hue-avaruudessa. OpenCV:n Hue-alue on 0..179.
    """

    hue = np.asarray(hue, dtype=np.float32)

    return np.abs((hue - float(center) + 90.0) % 180.0 - 90.0)


def circular_mean_hue(values):

    values = np.asarray(values, dtype=np.float64)

    if len(values) == 0:
        return 0.0

    angles = values / 180.0 * 2.0 * math.pi

    s = np.mean(np.sin(angles))
    c = np.mean(np.cos(angles))

    angle = math.atan2(s, c)

    if angle < 0:
        angle += 2.0 * math.pi

    return angle / (2.0 * math.pi) * 180.0


def calculate_hue_median_from_mask(hsv, mask):
    """
    Laskee Hue-mediaanin annettujen maskipikselien kohdalta.
    S- ja V-arvoja ei kayteta.
    """

    hue = hsv[:, :, 0]

    values = hue[mask > 0]

    if len(values) == 0:
        return None

    return float(np.median(values))


def create_hue_tolerance_mask(hsv, hue_median, tolerance=FAR_HUE_TOLERANCE):
    """
    Luo maskin pelkan Hue-arvon perusteella.
    Hyvaksyy Hue-arvot: mediaani +- tolerance.
    S- ja V-arvoja ei huomioida.
    """

    if hue_median is None:
        return np.zeros(hsv.shape[:2], dtype=np.uint8)

    hue = hsv[:, :, 0].astype(np.float32)

    d = circular_hue_distance(hue, hue_median)

    mask = (d <= tolerance).astype(np.uint8) * 255

    return mask


# ============================================================
# ELLIPSIEN ETSINTA
# ============================================================

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
# T-LINJA / KESKILINJA (HOUGH) - LAHEMPAA PESAA VARTEN
# ============================================================

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
    edges = cv2.Canny(gray, 50, 150)
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


def select_t_and_centerline(segments, ellipse):

    center = np.array(ellipse[0], dtype=np.float64)

    t_pair = pair_line_segments(segments, ellipse, "vertical", center)
    centerline_pair = pair_line_segments(segments, ellipse, "horizontal", center)

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


# ============================================================
# KAMERAKALIBROINTI (LAHEMMAN PESAN YMPYROISTA)
# ============================================================

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


def print_camera_calibration(camera):

    print()
    print("=" * 65)
    print("LAHEMMAN PESAN KAMERAKALIBROINTI")
    print("=" * 65)
    print(f"Kalibrointiympyroita kaytetty: {camera['observation_count']}")
    print(f"Nakokulma (theta)   : {camera['theta_deg']:.3f} deg")
    print(f"Tasoetaisyys        : {camera['distance_to_plane']:.2f} cm")
    print(f"Kameran korkeus     : {camera['camera_height']:.2f} cm")


# ============================================================
# KAMERAPROJEKTIO (teoreettinen malli, kalibroinnin pohjalta)
# ============================================================

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


def project_points_vectorized(X, Y, camera, v_t, v_cl):

    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    f = camera["f"]
    theta = camera["theta"]
    camera_height = camera["camera_height"]
    center = camera["center"]

    relative_Y = Y - NEAR_HOUSE_Y_CM

    near_depth = NEAR_HOUSE_Y_CM * math.sin(theta) + camera_height
    Zc = near_depth + relative_Y * math.sin(theta)

    valid = Zc > 1e-8

    u = np.zeros_like(X)
    v = np.zeros_like(Y)

    u[valid] = f * X[valid] / Zc[valid]
    v[valid] = f * relative_Y[valid] * math.cos(theta) / Zc[valid]

    px = center[0] + u * v_t[0] + v * v_cl[0]
    py = center[1] + u * v_t[1] + v * v_cl[1]

    return px, py, valid


def image_points_to_physical(px, py, camera, v_t, v_cl):

    px = np.asarray(px, dtype=np.float64)
    py = np.asarray(py, dtype=np.float64)

    f = camera["f"]
    theta = camera["theta"]
    camera_height = camera["camera_height"]
    center = camera["center"]

    dx = px - center[0]
    dy = py - center[1]

    u = dx * v_t[0] + dy * v_t[1]
    v = dx * v_cl[0] + dy * v_cl[1]

    s = math.sin(theta)
    c = math.cos(theta)

    near_depth = NEAR_HOUSE_Y_CM * s + camera_height

    denominator = f * c - v * s
    valid = np.abs(denominator) > 1e-8

    relative_Y = np.zeros_like(v)
    relative_Y[valid] = v[valid] * near_depth / denominator[valid]

    Y = NEAR_HOUSE_Y_CM + relative_Y

    Zc = near_depth + relative_Y * s

    X = np.zeros_like(u)
    valid &= Zc > 1e-8
    X[valid] = u[valid] * Zc[valid] / f

    return X, Y, valid


# ============================================================
# HSV TEMPLATE (LAHEMMASTA PESASTA) - kaytetaan haussa
# ============================================================

def derive_near_house_hsv_model(frame, outer_ellipse):

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    h = hsv[:, :, 0].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32)

    roi = np.zeros(frame.shape[:2], dtype=np.uint8)

    cv2.ellipse(
        roi,
        tuple(np.round(outer_ellipse[0]).astype(int)),
        tuple(np.round(np.asarray(outer_ellipse[1]) / 2.0).astype(int)),
        float(outer_ellipse[2]),
        0, 360, 255, -1
    )

    roi_bool = roi > 0

    blue_candidate = roi_bool & (s >= 40) & (h >= 85) & (h <= 145)
    blue_hues = h[blue_candidate]

    if len(blue_hues) > 20:

        blue_center = circular_mean_hue(blue_hues)
        blue_distances = circular_hue_distance(blue_hues, blue_center)
        blue_width = np.clip(
            float(np.percentile(blue_distances, 95)) + 5.0, 10.0, 30.0
        )

    else:

        blue_center = 110.0
        blue_width = 20.0

    red_candidate = roi_bool & (s >= 40) & ((h <= 25) | (h >= 155))
    red_hues = h[red_candidate]

    if len(red_hues) > 20:

        red_center = circular_mean_hue(red_hues)
        red_distances = circular_hue_distance(red_hues, red_center)
        red_width = np.clip(
            float(np.percentile(red_distances, 95)) + 5.0, 8.0, 25.0
        )

    else:

        red_center = 0.0
        red_width = 15.0

    color_candidate = roi_bool & (s >= 40)

    if np.any(color_candidate):

        saturation_threshold = np.clip(
            float(np.percentile(s[color_candidate], 10)), 30.0, 100.0
        )

    else:

        saturation_threshold = 50.0

    return {
        "blue_center": float(blue_center),
        "blue_width": float(blue_width),
        "red_center": float(red_center),
        "red_width": float(red_width),
        "saturation_threshold": float(saturation_threshold),
        "hsv": hsv,
        "roi": roi,
    }


def create_near_house_template_masks(hsv_model):

    hsv = hsv_model["hsv"]

    h = hsv[:, :, 0].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32)
    v = hsv[:, :, 2].astype(np.float32)

    roi = hsv_model["roi"] > 0
    s_threshold = hsv_model["saturation_threshold"]

    blue_distance = circular_hue_distance(h, hsv_model["blue_center"])
    red_distance = circular_hue_distance(h, hsv_model["red_center"])

    blue_mask = (
        roi & (s >= s_threshold) &
        (blue_distance <= hsv_model["blue_width"]) & (v >= 15)
    )

    red_mask = (
        roi & (s >= s_threshold) &
        (red_distance <= hsv_model["red_width"]) & (v >= 15)
    )

    blue_mask = (blue_mask.astype(np.uint8) * 255)
    red_mask = (red_mask.astype(np.uint8) * 255)

    kernel = np.ones((3, 3), np.uint8)

    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)

    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

    return blue_mask, red_mask


def sample_mask_points(mask, max_points, rng):

    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        return np.array([]), np.array([])

    if len(xs) > max_points:

        indices = rng.choice(len(xs), size=max_points, replace=False)
        xs = xs[indices]
        ys = ys[indices]

    return xs.astype(np.float64), ys.astype(np.float64)


def build_near_house_template(frame, outer_ellipse, camera, v_t, v_cl):

    print()
    print("=" * 65)
    print("LAHEMMAN PESAN HSV-TEMPLATEN RAKENNUS")
    print("=" * 65)

    hsv_model = derive_near_house_hsv_model(frame, outer_ellipse)

    blue_mask, red_mask = create_near_house_template_masks(hsv_model)

    rng = np.random.default_rng(12345)

    blue_x, blue_y = sample_mask_points(blue_mask, MAX_BLUE_TEMPLATE_POINTS, rng)
    red_x, red_y = sample_mask_points(red_mask, MAX_RED_TEMPLATE_POINTS, rng)

    combined = cv2.bitwise_or(blue_mask, red_mask)
    roi = hsv_model["roi"]

    background_mask = cv2.bitwise_and(roi, cv2.bitwise_not(combined))

    bg_x, bg_y = sample_mask_points(
        background_mask, MAX_BACKGROUND_TEMPLATE_POINTS, rng
    )

    blue_X, blue_Y, blue_valid = image_points_to_physical(
        blue_x, blue_y, camera, v_t, v_cl
    )

    red_X, red_Y, red_valid = image_points_to_physical(
        red_x, red_y, camera, v_t, v_cl
    )

    bg_X, bg_Y, bg_valid = image_points_to_physical(
        bg_x, bg_y, camera, v_t, v_cl
    )

    blue_X, blue_Y = blue_X[blue_valid], blue_Y[blue_valid]
    red_X, red_Y = red_X[red_valid], red_Y[red_valid]
    bg_X, bg_Y = bg_X[bg_valid], bg_Y[bg_valid]

    print(f"Sininen template-pisteita : {len(blue_X)}")
    print(f"Punainen template-pisteita: {len(red_X)}")
    print(f"Tausta-pisteita           : {len(bg_X)}")

    return {
        "blue_X": blue_X, "blue_Y": blue_Y,
        "red_X": red_X, "red_Y": red_Y,
        "background_X": bg_X, "background_Y": bg_Y,
        "hsv_model": hsv_model,
    }


# ============================================================
# TEMPLATE -> KAUKAINEN PESA (fyysinen siirto + kuvatason kierto/skaala)
# ============================================================

def transform_projected_points(px, py, center_x, center_y, angle_deg, scale):

    px = np.asarray(px, dtype=np.float64)
    py = np.asarray(py, dtype=np.float64)

    angle = math.radians(angle_deg)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    dx = (px - center_x) * scale
    dy = (py - center_y) * scale

    rotated_x = cos_a * dx - sin_a * dy
    rotated_y = sin_a * dx + cos_a * dy

    return center_x + rotated_x, center_y + rotated_y


def transform_template_to_far(
    template_X, template_Y, x_offset_cm, y_offset_cm,
    angle_deg, scale, camera, v_t, v_cl
):

    local_X = template_X
    local_Y = template_Y - NEAR_HOUSE_Y_CM

    far_X = x_offset_cm + local_X
    far_Y = FAR_HOUSE_Y_CM + y_offset_cm + local_Y

    px, py, valid = project_points_vectorized(far_X, far_Y, camera, v_t, v_cl)

    far_center = project_point(
        x_offset_cm, FAR_HOUSE_Y_CM + y_offset_cm, camera, v_t, v_cl
    )

    if far_center is None:
        return px, py, np.zeros_like(valid, dtype=bool)

    rotated_px, rotated_py = transform_projected_points(
        px, py, far_center[0], far_center[1], angle_deg, scale
    )

    return rotated_px, rotated_py, valid


# ============================================================
# HSV-VASTAAVUUSPISTEYTYS (haun pisteytysfunktio)
# ============================================================

def hsv_color_match_score(
    px, py, hsv, color_center, color_width,
    saturation_threshold, image_width, image_height
):

    px_int = np.round(px).astype(np.int32)
    py_int = np.round(py).astype(np.int32)

    valid = (
        (px_int >= 0) & (px_int < image_width) &
        (py_int >= 0) & (py_int < image_height)
    )

    if not np.any(valid):
        return 0.0, 0.0

    hsv_values = hsv[py_int[valid], px_int[valid]]

    h = hsv_values[:, 0].astype(np.float32)
    s = hsv_values[:, 1].astype(np.float32)
    v = hsv_values[:, 2].astype(np.float32)

    hue_distance = circular_hue_distance(h, color_center)

    hue_score = np.exp(-0.5 * (hue_distance / max(1.0, color_width)) ** 2)

    saturation_score = np.clip(
        (s - saturation_threshold) / (255.0 - saturation_threshold + 1e-6),
        0.0, 1.0
    )

    visibility_score = np.clip(v / 50.0, 0.0, 1.0)

    score = 0.70 * hue_score + 0.25 * saturation_score + 0.05 * visibility_score

    return float(np.mean(score)), float(np.mean(valid))


def background_match_score(px, py, hsv, hsv_model, image_width, image_height):

    px_int = np.round(px).astype(np.int32)
    py_int = np.round(py).astype(np.int32)

    valid = (
        (px_int >= 0) & (px_int < image_width) &
        (py_int >= 0) & (py_int < image_height)
    )

    if not np.any(valid):
        return 0.0, 0.0

    hsv_values = hsv[py_int[valid], px_int[valid]]

    h = hsv_values[:, 0].astype(np.float32)
    s = hsv_values[:, 1].astype(np.float32)

    blue_distance = circular_hue_distance(h, hsv_model["blue_center"])
    red_distance = circular_hue_distance(h, hsv_model["red_center"])

    blue_color = np.exp(
        -0.5 * (blue_distance / max(1.0, hsv_model["blue_width"])) ** 2
    )

    red_color = np.exp(
        -0.5 * (red_distance / max(1.0, hsv_model["red_width"])) ** 2
    )

    saturation_score = np.clip(
        (s - hsv_model["saturation_threshold"]) /
        (255.0 - hsv_model["saturation_threshold"] + 1e-6),
        0.0, 1.0
    )

    coloredness = np.maximum(blue_color, red_color) * saturation_score
    score = 1.0 - coloredness

    return float(np.mean(score)), float(np.mean(valid))


def score_far_candidate(
    template, x_offset_cm, y_offset_cm, angle_deg, scale,
    frame, camera, v_t, v_cl
):

    image_height, image_width = frame.shape[:2]

    hsv = template["_hsv"]
    hsv_model = template["hsv_model"]

    blue_px_x, blue_px_y, blue_valid = transform_template_to_far(
        template["blue_X"], template["blue_Y"], x_offset_cm, y_offset_cm,
        angle_deg, scale, camera, v_t, v_cl
    )

    blue_score, blue_valid_fraction = hsv_color_match_score(
        blue_px_x, blue_px_y, hsv, hsv_model["blue_center"],
        hsv_model["blue_width"], hsv_model["saturation_threshold"],
        image_width, image_height
    )

    red_px_x, red_px_y, red_valid = transform_template_to_far(
        template["red_X"], template["red_Y"], x_offset_cm, y_offset_cm,
        angle_deg, scale, camera, v_t, v_cl
    )

    red_score, red_valid_fraction = hsv_color_match_score(
        red_px_x, red_px_y, hsv, hsv_model["red_center"],
        hsv_model["red_width"], hsv_model["saturation_threshold"],
        image_width, image_height
    )

    bg_px_x, bg_px_y, bg_valid = transform_template_to_far(
        template["background_X"], template["background_Y"],
        x_offset_cm, y_offset_cm, angle_deg, scale, camera, v_t, v_cl
    )

    bg_score, bg_valid_fraction = background_match_score(
        bg_px_x, bg_px_y, hsv, hsv_model, image_width, image_height
    )

    valid_fraction = (
        0.40 * blue_valid_fraction +
        0.40 * red_valid_fraction +
        0.20 * bg_valid_fraction
    )

    color_score = 0.50 * blue_score + 0.35 * red_score + 0.15 * bg_score

    return float(color_score * (0.50 + 0.50 * valid_fraction))


def _project_template_for_offset(template, x_offset_cm, y_offset_cm, camera, v_t, v_cl):
    """
    Laskee sen OSAN transform_template_to_far:in tyosta, joka riippuu
    VAIN (x_offset_cm, y_offset_cm):sta - eli fyysisen sijainnin
    projisoinnin kameran kuvatasolle (project_points_vectorized) -
    EI viela kiertoa/skaalausta (angle_deg, scale), koska ne
    sovelletaan vasta transform_projected_points:lla NAIHIN valmiiksi
    projisoituihin pisteisiin (katso transform_template_to_far).

    NOPEUSOPTIMOINTI (kamera7_02.py): search_far_house:in alkuperainen
    silmukka kutsui transform_template_to_far:aa (joka sisaltaa taman
    projisoinnin) JOKAISELLE (x,y,angle,scale)-yhdistelmalle, vaikka
    projisointi ei riipu angle/scale:sta lainkaan - saman projisoinnin
    laskeminen 11x11=121 kertaa uudelleen jokaiselle (x,y)-parille oli
    puhdasta hukkatyota. Kayttamalla tata valimuistia search_far_house
    laskee projisoinnin vain KERRAN per (x,y), ja silmukoi angle/scale:n
    yli vain halvalla kierto+skaalaus+pisteytys-vaiheella - tulos on
    matemaattisesti TAYSIN sama kuin ennen, vain nopeampi.
    """

    blue_far_X = x_offset_cm + template["blue_X"]
    blue_far_Y = FAR_HOUSE_Y_CM + y_offset_cm + (template["blue_Y"] - NEAR_HOUSE_Y_CM)
    blue_px, blue_py, blue_valid = project_points_vectorized(
        blue_far_X, blue_far_Y, camera, v_t, v_cl
    )

    red_far_X = x_offset_cm + template["red_X"]
    red_far_Y = FAR_HOUSE_Y_CM + y_offset_cm + (template["red_Y"] - NEAR_HOUSE_Y_CM)
    red_px, red_py, red_valid = project_points_vectorized(
        red_far_X, red_far_Y, camera, v_t, v_cl
    )

    bg_far_X = x_offset_cm + template["background_X"]
    bg_far_Y = FAR_HOUSE_Y_CM + y_offset_cm + (template["background_Y"] - NEAR_HOUSE_Y_CM)
    bg_px, bg_py, bg_valid = project_points_vectorized(
        bg_far_X, bg_far_Y, camera, v_t, v_cl
    )

    far_center = project_point(
        x_offset_cm, FAR_HOUSE_Y_CM + y_offset_cm, camera, v_t, v_cl
    )

    return {
        "blue_px": blue_px, "blue_py": blue_py,
        "red_px": red_px, "red_py": red_py,
        "bg_px": bg_px, "bg_py": bg_py,
        "far_center": far_center,
    }


def score_far_candidate_from_projected(
    projected, angle_deg, scale, hsv, hsv_model, image_width, image_height
):
    """
    Sama pisteytys kuin score_far_candidate, mutta kayttaa VALMIIKSI
    projisoituja pisteita (_project_template_for_offset) - soveltaa
    vain kierron/skaalauksen (angle_deg, scale) ja pisteyttaa. Katso
    _project_template_for_offset:in perustelu.
    """

    far_center = projected["far_center"]

    blue_rx, blue_ry = transform_projected_points(
        projected["blue_px"], projected["blue_py"],
        far_center[0], far_center[1], angle_deg, scale
    )
    blue_score, blue_valid_fraction = hsv_color_match_score(
        blue_rx, blue_ry, hsv, hsv_model["blue_center"],
        hsv_model["blue_width"], hsv_model["saturation_threshold"],
        image_width, image_height
    )

    red_rx, red_ry = transform_projected_points(
        projected["red_px"], projected["red_py"],
        far_center[0], far_center[1], angle_deg, scale
    )
    red_score, red_valid_fraction = hsv_color_match_score(
        red_rx, red_ry, hsv, hsv_model["red_center"],
        hsv_model["red_width"], hsv_model["saturation_threshold"],
        image_width, image_height
    )

    bg_rx, bg_ry = transform_projected_points(
        projected["bg_px"], projected["bg_py"],
        far_center[0], far_center[1], angle_deg, scale
    )
    bg_score, bg_valid_fraction = background_match_score(
        bg_rx, bg_ry, hsv, hsv_model, image_width, image_height
    )

    valid_fraction = (
        0.40 * blue_valid_fraction +
        0.40 * red_valid_fraction +
        0.20 * bg_valid_fraction
    )

    color_score = 0.50 * blue_score + 0.35 * red_score + 0.15 * bg_score

    return float(color_score * (0.50 + 0.50 * valid_fraction))


def _transform_projected_points_batch(px, py, center_x, center_y, angle_rad, scale):
    """
    Sama kuin transform_projected_points, mutta angle_rad ja scale ovat
    M-pituisia taulukoita (yksi per (kulma,skaala)-ehdokas) ja px,py
    N-pituisia (yksi per templaten piste) - palauttaa (M,N)-taulukot
    (broadcastattu). Kayttaa VALMIIKSI radiaaneiksi muunnettua kulmaa,
    jotta math.radians ei ole tarpeen kutsua M kertaa.
    """

    cos_a = np.cos(angle_rad)[:, None]
    sin_a = np.sin(angle_rad)[:, None]
    scale = scale[:, None]

    dx = (px[None, :] - center_x) * scale
    dy = (py[None, :] - center_y) * scale

    rotated_x = cos_a * dx - sin_a * dy
    rotated_y = sin_a * dx + cos_a * dy

    return center_x + rotated_x, center_y + rotated_y


def _hsv_color_match_score_batch(
    px, py, hsv, color_center, color_width,
    saturation_threshold, image_width, image_height
):
    """
    Vektoroitu (M,N) -versio hsv_color_match_score:sta - laskee
    pisteytyksen KAIKILLE M ehdokkaalle yhdella kutsulla (px,py ovat
    (M,N)-taulukoita). Palauttaa (M,) pisteytys- ja validi-osuus-
    taulukot. Matemaattisesti identtinen rivi riviltä kutsuttuun
    hsv_color_match_score:iin - katso search_far_house:in kommentti.
    """

    px_int = np.round(px).astype(np.int32)
    py_int = np.round(py).astype(np.int32)

    valid = (
        (px_int >= 0) & (px_int < image_width) &
        (py_int >= 0) & (py_int < image_height)
    )

    px_c = np.clip(px_int, 0, image_width - 1)
    py_c = np.clip(py_int, 0, image_height - 1)

    hsv_values = hsv[py_c, px_c]

    h = hsv_values[..., 0].astype(np.float32)
    s = hsv_values[..., 1].astype(np.float32)
    v = hsv_values[..., 2].astype(np.float32)

    hue_distance = circular_hue_distance(h, color_center)
    hue_score = np.exp(-0.5 * (hue_distance / max(1.0, color_width)) ** 2)

    saturation_score = np.clip(
        (s - saturation_threshold) / (255.0 - saturation_threshold + 1e-6),
        0.0, 1.0
    )

    visibility_score = np.clip(v / 50.0, 0.0, 1.0)

    score = 0.70 * hue_score + 0.25 * saturation_score + 0.05 * visibility_score

    n_valid = valid.sum(axis=1)
    score_sum = np.where(valid, score, 0.0).sum(axis=1)

    mean_score = np.where(n_valid > 0, score_sum / np.maximum(n_valid, 1), 0.0)
    valid_fraction = n_valid / px.shape[1]

    return mean_score, valid_fraction


def _background_match_score_batch(px, py, hsv, hsv_model, image_width, image_height):
    """
    Vektoroitu (M,N) -versio background_match_score:sta - katso
    _hsv_color_match_score_batch:in kommentti.
    """

    px_int = np.round(px).astype(np.int32)
    py_int = np.round(py).astype(np.int32)

    valid = (
        (px_int >= 0) & (px_int < image_width) &
        (py_int >= 0) & (py_int < image_height)
    )

    px_c = np.clip(px_int, 0, image_width - 1)
    py_c = np.clip(py_int, 0, image_height - 1)

    hsv_values = hsv[py_c, px_c]

    h = hsv_values[..., 0].astype(np.float32)
    s = hsv_values[..., 1].astype(np.float32)

    blue_distance = circular_hue_distance(h, hsv_model["blue_center"])
    red_distance = circular_hue_distance(h, hsv_model["red_center"])

    blue_color = np.exp(-0.5 * (blue_distance / max(1.0, hsv_model["blue_width"])) ** 2)
    red_color = np.exp(-0.5 * (red_distance / max(1.0, hsv_model["red_width"])) ** 2)

    saturation_score = np.clip(
        (s - hsv_model["saturation_threshold"]) /
        (255.0 - hsv_model["saturation_threshold"] + 1e-6),
        0.0, 1.0
    )

    coloredness = np.maximum(blue_color, red_color) * saturation_score
    score = 1.0 - coloredness

    n_valid = valid.sum(axis=1)
    score_sum = np.where(valid, score, 0.0).sum(axis=1)

    mean_score = np.where(n_valid > 0, score_sum / np.maximum(n_valid, 1), 0.0)
    valid_fraction = n_valid / px.shape[1]

    return mean_score, valid_fraction


def score_far_candidates_batch(
    projected, angle_values, scale_values, hsv, hsv_model,
    image_width, image_height
):
    """
    NOPEUSOPTIMOINTI (kamera7_02.py): laskee pisteytyksen KAIKILLE
    (angle,scale)-yhdistelmille YHDELLA vektoroidulla kutsulla, sen
    sijaan etta score_far_candidate_from_projected kutsuttaisiin
    erikseen jokaiselle (alkuperainen: 11x11=121 erillista Python-
    tason kutsua per (x,y) - kukin niista useita pieniä numpy-
    kutsuja). Profiloinnissa havaittiin etta itse pisteytys (ei
    projisointi) on search_far_house:in painavin osa - tama poistaa
    sen toistuvan Python-/numpy-kutsuoverheadin kokonaan.

    Palauttaa (len(angle_values)*len(scale_values),) -pisteytys-
    taulukon, jarjestyksessa [angle0,scale0], [angle0,scale1], ...
    (rivi = angle, sarake = scale, litistettyna) - sama jarjestys
    kuin search_far_house:in silmukka kavisi lapi.
    """

    far_center = projected["far_center"]

    angle_grid, scale_grid = np.meshgrid(angle_values, scale_values, indexing="ij")
    angle_rad = np.radians(angle_grid.ravel())
    scale_flat = scale_grid.ravel()

    blue_rx, blue_ry = _transform_projected_points_batch(
        projected["blue_px"], projected["blue_py"],
        far_center[0], far_center[1], angle_rad, scale_flat
    )
    blue_score, blue_valid_fraction = _hsv_color_match_score_batch(
        blue_rx, blue_ry, hsv, hsv_model["blue_center"],
        hsv_model["blue_width"], hsv_model["saturation_threshold"],
        image_width, image_height
    )

    red_rx, red_ry = _transform_projected_points_batch(
        projected["red_px"], projected["red_py"],
        far_center[0], far_center[1], angle_rad, scale_flat
    )
    red_score, red_valid_fraction = _hsv_color_match_score_batch(
        red_rx, red_ry, hsv, hsv_model["red_center"],
        hsv_model["red_width"], hsv_model["saturation_threshold"],
        image_width, image_height
    )

    bg_rx, bg_ry = _transform_projected_points_batch(
        projected["bg_px"], projected["bg_py"],
        far_center[0], far_center[1], angle_rad, scale_flat
    )
    bg_score, bg_valid_fraction = _background_match_score_batch(
        bg_rx, bg_ry, hsv, hsv_model, image_width, image_height
    )

    valid_fraction = (
        0.40 * blue_valid_fraction +
        0.40 * red_valid_fraction +
        0.20 * bg_valid_fraction
    )

    color_score = 0.50 * blue_score + 0.35 * red_score + 0.15 * bg_score

    return color_score * (0.50 + 0.50 * valid_fraction)


# ============================================================
# HAKURUUDUKKO / OPTIMOINTI (coarse -> fine -> ultra fine)
# ============================================================

def make_range(center, half_range, step):

    start = center - half_range
    stop = center + half_range

    count = int(round((stop - start) / step))

    values = start + np.arange(count + 1, dtype=np.float64) * step

    if values[-1] < stop - 1e-8:
        values = np.append(values, stop)

    return values


def search_far_house(
    template, frame, camera, v_t, v_cl,
    center_x, center_y, center_range, center_step,
    angle_center, angle_range, angle_step,
    scale_center, scale_range, scale_step,
    description
):

    print()
    print("=" * 65)
    print(f"KAUKAISEN PESAN HAKU: {description}")
    print("=" * 65)

    x_values = make_range(center_x, center_range, center_step)
    y_values = make_range(center_y, center_range, center_step)
    angle_values = make_range(angle_center, angle_range, angle_step)
    scale_values = make_range(scale_center, scale_range, scale_step)

    total = len(x_values) * len(y_values) * len(angle_values) * len(scale_values)

    print(f"Kandidaatteja yhteensa: {total}")

    image_height, image_width = frame.shape[:2]
    hsv = template["_hsv"]
    hsv_model = template["hsv_model"]

    best_score = -float("inf")
    best_x, best_y = center_x, center_y
    best_angle, best_scale = angle_center, scale_center

    for x_offset in x_values:
        for y_offset in y_values:

            # Projisointi lasketaan vain KERRAN per (x,y) - katso
            # _project_template_for_offset:in perustelu.
            projected = _project_template_for_offset(
                template, x_offset, y_offset, camera, v_t, v_cl
            )

            if projected["far_center"] is None:
                continue

            # Kaikki (angle,scale)-yhdistelmat pisteytetaan yhdella
            # vektoroidulla kutsulla - katso score_far_candidates_batch:in
            # perustelu.
            scores = score_far_candidates_batch(
                projected, angle_values, scale_values, hsv, hsv_model,
                image_width, image_height
            )

            local_best_idx = int(np.argmax(scores))

            if scores[local_best_idx] > best_score:
                best_score = float(scores[local_best_idx])
                best_x, best_y = x_offset, y_offset
                best_angle = float(angle_values[local_best_idx // len(scale_values)])
                best_scale = float(scale_values[local_best_idx % len(scale_values)])

    print(f"Paras X-siirto  : {best_x:.3f} cm")
    print(f"Paras Y-siirto  : {best_y:.3f} cm")
    print(f"Paras kulma     : {best_angle:.3f} deg")
    print(f"Paras skaala    : {best_scale:.4f}")
    print(f"Paras pisteytys : {best_score:.6f}")

    return {
        "x_offset_cm": best_x, "y_offset_cm": best_y,
        "angle_deg": best_angle, "scale": best_scale, "score": best_score
    }


# ============================================================
# KAUKAISEN PESAN FYYSINEN ROI (haun tuloksen perusteella)
# ============================================================

def create_far_house_physical_roi(
    frame,
    x_offset_cm,
    y_offset_cm,
    camera,
    v_t,
    v_cl,
    angle_deg,
    scale=1.0,
    half_width_cm=300.0,
    half_height_cm=300.0
):
    """
    Luo kaukaisen pesan optimoidun keskipisteen ympärille
    fyysisesti suorakaiteen, joka projisoidaan kameramallilla ja
    johon sovelletaan haussa loydettya kulmakorjausta ja
    skaalausta.
    """

    h, w = frame.shape[:2]

    far_x_cm = x_offset_cm
    far_y_cm = FAR_HOUSE_Y_CM + y_offset_cm

    corners_physical = np.array([
        [far_x_cm - half_width_cm, far_y_cm - half_height_cm],
        [far_x_cm + half_width_cm, far_y_cm - half_height_cm],
        [far_x_cm + half_width_cm, far_y_cm + half_height_cm],
        [far_x_cm - half_width_cm, far_y_cm + half_height_cm],
    ], dtype=np.float64)

    px, py, valid = project_points_vectorized(
        corners_physical[:, 0],
        corners_physical[:, 1],
        camera,
        v_t,
        v_cl
    )

    if not np.all(valid):
        raise RuntimeError(
            "Kaukaisen pesan ROI:n projisointi epaonnistui."
        )

    far_center_raw = project_point(
        far_x_cm,
        far_y_cm,
        camera,
        v_t,
        v_cl
    )

    transformed_x = []
    transformed_y = []

    for x, y in zip(px, py):
        tx, ty = transform_projected_points(
            x, y, far_center_raw[0], far_center_raw[1], angle_deg, scale
        )

        transformed_x.append(tx)
        transformed_y.append(ty)

    polygon = np.column_stack([
        np.round(transformed_x).astype(np.int32),
        np.round(transformed_y).astype(np.int32)
    ])

    roi = np.zeros((h, w), dtype=np.uint8)

    cv2.fillConvexPoly(roi, polygon, 255)

    return roi, polygon


def create_far_house_hue_masks(
    frame,
    x_offset_cm,
    y_offset_cm,
    camera,
    v_t,
    v_cl,
    angle_deg,
    scale
):
    """
    Laskee kaukaisen pesan sinisen ja punaisen Hue-mediaanin
    (haun loytaman ROI:n sisalta) ja muodostaa lopulliset maskit
    PELKASTAAN Hue-arvon perusteella. S- ja V-arvoja ei kayteta
    lopullisissa maskeissa.
    """

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    house_roi, roi_polygon = create_far_house_physical_roi(
        frame,
        x_offset_cm,
        y_offset_cm,
        camera,
        v_t,
        v_cl,
        angle_deg=angle_deg,
        scale=scale,
        half_width_cm=220.0,
        half_height_cm=300.0
    )

    blue_hsv_mask = create_blue_mask(frame)
    red_hsv_mask = create_red_mask(frame)

    blue_sampling_mask = cv2.bitwise_and(blue_hsv_mask, house_roi)
    red_sampling_mask = cv2.bitwise_and(red_hsv_mask, house_roi)

    blue_hue_median = calculate_hue_median_from_mask(hsv, blue_sampling_mask)
    red_hue_median = calculate_hue_median_from_mask(hsv, red_sampling_mask)

    blue_hue_mask = create_hue_tolerance_mask(hsv, blue_hue_median, FAR_HUE_TOLERANCE)
    red_hue_mask = create_hue_tolerance_mask(hsv, red_hue_median, FAR_HUE_TOLERANCE)

    blue_hue_mask = cv2.bitwise_and(blue_hue_mask, house_roi)
    red_hue_mask = cv2.bitwise_and(red_hue_mask, house_roi)

    kernel = np.ones((3, 3), np.uint8)

    blue_hue_mask = cv2.morphologyEx(blue_hue_mask, cv2.MORPH_OPEN, kernel)
    blue_hue_mask = cv2.morphologyEx(blue_hue_mask, cv2.MORPH_CLOSE, kernel)

    red_hue_mask = cv2.morphologyEx(red_hue_mask, cv2.MORPH_OPEN, kernel)
    red_hue_mask = cv2.morphologyEx(red_hue_mask, cv2.MORPH_CLOSE, kernel)

    return {
        "house_roi": house_roi,
        "roi_polygon": roi_polygon,
        "blue_hue_median": blue_hue_median,
        "red_hue_median": red_hue_median,
        "blue_hue_mask": blue_hue_mask,
        "red_hue_mask": red_hue_mask,
    }


# ============================================================
# KAUKAISEN PESAN ELLIPSIT HUE-MASKEISTA + KORRESPONDENSSIT
# ============================================================

def find_far_house_ellipses_from_hue(hue_result):
    """
    Sovittaa ulko- ja sisaellipsit kaukaisen pesan Hue-maskeihin
    (samalla tavalla kuin lahemmalle pesalle).
    """

    blue_outer, blue_inner = find_house_pair(
        hue_result["blue_hue_mask"], FAR_BLUE_MIN_AREA, FAR_BLUE_MIN_RATIO,
        FAR_BLUE_MIN_SIZE_RATIO, min_contour_len=FAR_BLUE_MIN_CONTOUR_LEN
    )

    red_outer, red_inner = find_house_pair(
        hue_result["red_hue_mask"], FAR_RED_MIN_AREA, FAR_RED_MIN_RATIO,
        FAR_RED_MIN_SIZE_RATIO, min_contour_len=FAR_RED_MIN_CONTOUR_LEN
    )

    return {
        "blue_mask": hue_result["blue_hue_mask"],
        "red_mask": hue_result["red_hue_mask"],
        "blue_outer": blue_outer,
        "blue_inner": blue_inner,
        "red_outer": red_outer,
        "red_inner": red_inner,
    }


def far_house_correspondences_from_ellipses(
    far_result, dir_lateral, dir_forward, fallback_center_img
):
    """
    Muodostaa kaukaisen pesan korrespondenssipisteet sen OMISTA
    hue-maskeista sovitetuista ellipseista.

    Sisaellipsit (blue_inner, red_inner) on suodatettu pois -
    niita ei kayteta homografiaan. Kaytossa on vain:
      - FAR_CENTER (keskipiste)
      - FAR_BLUE_OUTER: vasen + oikea (T-linjan suuntaiset pisteet)
      - FAR_RED_OUTER : vasen + oikea (T-linjan suuntaiset pisteet)

    Keskilinjan (lahi/kauka) suuntaisia pisteita ei kayteta
    kaukaiselle pesalle - vain T-linjalla olevat.
    """

    ref_ellipse = far_result["blue_outer"] or far_result["red_outer"]

    if ref_ellipse is not None:
        far_center_img = np.array(ref_ellipse[0], dtype=np.float64)
    else:
        far_center_img = np.asarray(fallback_center_img, dtype=np.float64)

    image_pts = [far_center_img]
    physical_pts = [(0.0, FAR_HOUSE_Y_CM)]
    labels = ["FAR_CENTER"]

    specs = [
        ("FAR_BLUE_OUTER", far_result["blue_outer"], BLUE_OUTER_RADIUS_CM),
        ("FAR_RED_OUTER", far_result["red_outer"], RED_OUTER_RADIUS_CM),
    ]

    for name, ellipse, radius in specs:

        if ellipse is None:
            continue

        pts, phys, labs = ellipse_lateral_correspondences(
            ellipse, radius, dir_lateral, FAR_HOUSE_Y_CM, name
        )

        image_pts.extend(pts)
        physical_pts.extend(phys)
        labels.extend(labs)

    return image_pts, physical_pts, labels, far_center_img


# ============================================================
# KAUKAISEN PESAN KORRESPONDENSSIT SUORAAN HAUN OPTIMOIDUSTA
# MUUNNOKSESTA (EI ERILLISTA ELLIPSIN SOVITUSTA RAAKAKUVASTA)
#
# search_far_house (coarse -> fine -> ultra) LOYTAA (x_offset,
# y_offset, angle_deg, scale) MAKSIMOIMALLA lahemman pesan HSV-
# templaatin (tuhansia pisteita) sopivuuden todelliseen kuvaan
# kaukaisen pesan kohdalla. Namä parametrit siis JO sisaltavat
# todellisesta kuvasta opitun tiedon kaukaisen pesan sijainnista,
# koosta JA muodosta (angle_deg+scale ovat nimenomaan empiirinen
# korjaus, jonka haku loysi selittaakseen kameramallin ja todellisen
# kuvan valisen eron).
#
# far_house_correspondences_from_ellipses (yllaoleva funktio) heittaa
# taman tiedon pois ja yrittaa sovittaa UUDEN, ERILLISEN ellipsin
# kaukaisen pesan hue-maskin kontuuriin - mika on hauras, koska
# kontuuri on aarimmilleen perspektiivin venyttama (esim. muotosuhde
# ~0.13). Talla funktiolla projisoidaan sen sijaan pesan TUNNETUT
# fyysiset reunapisteet (sade ± BLUE/RED_OUTER_RADIUS_CM T-linjalla)
# SUORAAN haun jo optimoiman muunnoksen lapi (transform_template_to_far,
# sama mekanismi jolla search_far_house pisteyttaa ehdokkaita) - ei
# vaadita mitaan erillista kontuuri-/ellipsihakua kaukaisesta pesasta.
# ============================================================

def theoretical_far_house_correspondences(
    x_offset, y_offset, angle_deg, scale, camera, v_t, v_cl, far_center_img
):
    """
    Palauttaa (image_pts, phys_pts, labels) - keskipiste + sinisen ja
    punaisen ULKOKEHAN vasen/oikea T-linjalla (5 pistetta), samassa
    muodossa kuin far_house_correspondences_from_ellipses, mutta
    projisoituna suoraan kameramallin + haun optimoiman muunnoksen
    kautta.
    """

    image_pts = [np.asarray(far_center_img, dtype=np.float64)]
    physical_pts = [(0.0, FAR_HOUSE_Y_CM)]
    labels = ["FAR_CENTER_TEOR"]

    specs = [
        ("FAR_BLUE_OUTER_TEOR", BLUE_OUTER_RADIUS_CM),
        ("FAR_RED_OUTER_TEOR", RED_OUTER_RADIUS_CM),
    ]

    for name, radius_cm in specs:

        for sign, side in ((-1.0, "NEG"), (1.0, "POS")):

            px, py, valid = transform_template_to_far(
                np.array([sign * radius_cm]), np.array([NEAR_HOUSE_Y_CM]),
                x_offset, y_offset, angle_deg, scale, camera, v_t, v_cl
            )

            if not bool(valid[0]):
                continue

            image_pts.append(np.array([float(px[0]), float(py[0])]))
            physical_pts.append((sign * radius_cm, FAR_HOUSE_Y_CM))
            labels.append(f"{name}_{side}")

    return image_pts, physical_pts, labels


# ============================================================
# 17 PISTEEN KORRESPONDENSSIT - LAHEMPI PESA
# ============================================================

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


# ============================================================
# FYYSINEN (CM) -> LOPPUKUVAN PIKSELIT (Y KAANNETTY)
# ============================================================

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


# ============================================================
# OBJEKTIIVIN SATEITTAISEN VAARISTYMAN ITSEKALIBROINTI
#
# Ei vaadi erillisia kalibrointikuvia: k1 haetaan JOKA KUVASTA
# ERIKSEEN niin, etta jo tunnetut pesa-/hogline-korrespondenssi-
# pisteet selittyvat mahdollisimman hyvin YHDELLA homografialla.
# Jos objektiivissa ei ole merkittavaa vaaristymaa, paras loydetty
# k1 on lahella nollaa eika sita oteta kayttoon (katso
# K1_MIN_RELATIVE_IMPROVEMENT / K1_MIN_MAGNITUDE).
# ============================================================

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


def distort_points_px(points, camera_matrix, k1, k2=0.0):
    """
    undistort_points_px:n KAANTEISFUNKTIO: ottaa oikaistut (vaaristy-
    mattomat) pikselikoordinaatit ja palauttaa vastaavat VAARISTYNEET
    (alkuperaisen, kayttamattoman raakakuvan) pikselikoordinaatit
    SAMALLA k1:lla.

    cv2.undistortPoints ratkaisee vaaristyman poiston ITERATIIVISESTI
    (Newton-tyylisella menetelmalla), koska vaaristymamalli ei ole
    suljetussa muodossa kaannettavissa - mutta VAARISTYMISEN suunta
    (oikaistu -> vaaristynyt) ON suljetussa muodossa (sama kaava jota
    cv2:n oma malli kayttaa): normalisoidut koordinaatit kerrotaan
    tekijalla (1 + k1*r^2 + k2*r^4), missa r on etaisyys paapisteesta.
    Validoitu numeerisesti: distort_points_px(undistort_points_px(p))
    == p ja painvastoin, virhe < 1e-4 px.

    Kayttokelpoinen kun halutaan siirtaa top-down-kuvasta (jo kerran
    undistort_points_px:lla oikaistun raakakuvan pohjalta laskettu)
    piste TAKAISIN alkuperaisen, KAYTTAMATTOMAN raakakuvan koordi-
    naatistoon - katso refine_lens_distortion_k1.
    """

    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)

    f = camera_matrix[0, 0]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    x = (pts[:, 0] - cx) / f
    y = (pts[:, 1] - cy) / f

    r2 = x * x + y * y
    factor = 1.0 + k1 * r2 + k2 * r2 * r2

    x_distorted = x * factor
    y_distorted = y * factor

    px = x_distorted * f + cx
    py = y_distorted * f + cy

    return np.column_stack([px, py])


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


def raw_points_from_converged_result(refined, camera_matrix, current_k1):
    """
    Muuntaa refined:in near_verify/far_verify -ellipsisovituksista
    (pesien 17+5 pistetta) TUOREESTI lasketut top-down-pisteet
    TAKAISIN alkuperaisen, kayttamattoman RAAKAKUVAN pikselikoordi-
    naatistoon seka fyysisiksi (cm) kohdekoordinaateiksi - katso
    refine_lens_distortion_k1:in perustelu siita MIKSI tama on
    hyodyllista.

    HUOM (kamera7_05.py): aiemmin (kamera7_03/04.py) tama luki
    valmiiksi TALLENNETUT topdown_src_pts/topdown_dst_pts -kentat,
    joita VAIN refine_homography_corrections:in palauttama dict
    sisaltaa. Yhteisoptimoinnissa (refine_round) refined voi kuitenkin
    olla MYOS suoran uudelleensovituksen tai hogline-tasoituksen
    KORVAAMA dict, jolla noita kenttia ei ole - aiempi versio kaatui
    talloin KeyError:iin. Nyt pisteet LASKETAAN AINA TUOREENA suoraan
    refined["near_verify"]/refined["far_verify"] -ellipseista (samalla
    tavalla kuin compute_corrected_homography tekee homografian
    laskiessaan), kayttaen refined["H_before_correction"]:aa (se
    homografia jolla refined["topdown_raw"] - josta near_verify/
    far_verify on tunnistettu - aikoinaan tehtiin; refine_round pitaa
    huolen etta tama kentta on aina ajan tasalla myos korvatuissa
    dicteissa). Tama on toiminnallisesti sama laskenta kuin ennen -
    vain lahde (uudelleenlaskettu vs. valiin tallennettu) on eri -
    ja itseasiassa hieman TARKEMPI, koska fyysiset kohteet saadaan
    suoraan (ei enaa pyoristettya output_px_to_physical-edestakaisin-
    muunnosta float32-valimuodon kautta).

    Palauttaa (raw_img_pts, phys_pts) - molemmat Nx2 numpy-taulukoita.
    """

    topdown_raw = refined["topdown_raw"]
    H_before = refined["H_before_correction"]

    near_row_top, _ = compute_crop_row_range(
        topdown_raw.shape[0], NEAR_HOUSE_Y_CM, HOUSE_CROP_HALF_HEIGHT_CM
    )
    far_row_top, _ = compute_crop_row_range(
        topdown_raw.shape[0], FAR_HOUSE_Y_CM, HOUSE_CROP_HALF_HEIGHT_CM
    )

    near_ellipses_full = {
        key: shift_ellipse(refined["near_verify"].get(key), dy=near_row_top)
        for key in ("blue_outer", "blue_inner", "red_outer", "red_inner")
    }
    far_ellipses_full = {
        key: shift_ellipse(refined["far_verify"].get(key), dy=far_row_top)
        for key in ("blue_outer", "blue_inner", "red_outer", "red_inner")
    }

    near_hog_angle, near_hog_score = detect_hogline_angle(topdown_raw, NEAR_HOGLINE_Y_CM)
    far_hog_angle_raw, far_hog_score = detect_hogline_angle(topdown_raw, FAR_HOGLINE_Y_CM)

    total_hog_score = near_hog_score + far_hog_score
    far_weight = far_hog_score / total_hog_score if total_hog_score > 1e-6 else 0.5
    far_hog_angle = far_weight * far_hog_angle_raw + (1.0 - far_weight) * near_hog_angle

    near_dir_lateral, near_dir_forward = direction_from_angle(near_hog_angle)
    far_dir_lateral, _ = direction_from_angle(far_hog_angle)

    near_img_pts, near_phys_pts, _ = build_topdown_near_correspondences(
        near_ellipses_full, dir_lateral=near_dir_lateral, dir_forward=near_dir_forward
    )
    far_img_pts, far_phys_pts, _ = build_topdown_far_correspondences(
        far_ellipses_full, dir_lateral=far_dir_lateral
    )

    topdown_pts = np.array(near_img_pts + far_img_pts, dtype=np.float64)
    phys_pts = np.array(near_phys_pts + far_phys_pts, dtype=np.float64)

    H_inv = np.linalg.inv(H_before)

    frame_undistorted_pts = cv2.perspectiveTransform(
        topdown_pts.reshape(-1, 1, 2), H_inv
    ).reshape(-1, 2)

    raw_img_pts = distort_points_px(frame_undistorted_pts, camera_matrix, current_k1)

    return raw_img_pts, phys_pts


def refine_lens_distortion_k1(refined, camera_matrix, current_k1):
    """
    NOPEUSOPTIMOINTI (kamera7_02.py) -pipeline arvioi k1:n VAIN KERRAN,
    ennen mitaan homografiaa, 22:sta alkuperaisesta (raakakuvan
    ellipsisovituksesta suoraan saadusta, kohinaisesta - etenkin
    kaukainen pesa vain ~30-40 px leveana natiivikuvassa) korres-
    pondenssipisteesta. Nama pisteet ovat lisaksi PAINOTTUNEET
    voimakkaasti lahemman pesan tiiviiseen 17 pisteen ryhmaan (vain
    5 harvaa kaukaista pistetta) - huono, kapea otanta sateittaisen
    vaaristyman arviointiin, joka fyysisesti nakyy selvimmin VASTA
    kuvan reuna-alueilla (siis nimenomaan kaukaisella pesalla).

    Tama funktio (kamera7_03.py) arvioi k1:n UUDELLEEN sen jalkeen
    kun iteratiivinen homografian korjaus on jo KONVERGOITUNUT -
    tuolloin near_verify/far_verify -pisteet ovat tarkkoja (top-down-
    kuvasta osapikselitarkalla sateittaisella reunanetsinnalla + robus-
    tilla sovituksella tunnistettuja, ei enaa raakakuvan alkuperaista
    karkeaa ellipsisovitusta), ja ne kattavat SAMAN 17+5 pisteen
    joukon - mutta LAADUKKAAMPANA. Nain saadaan huomattavasti
    luotettavampi arvio siita, JAAKO jaljelle aitoa, homografialla
    (paikallisesti lineaarinen/projektiivinen malli) korjaamatonta
    sateittaista vaaristymaa.

    Palauttaa (uusi_k1, uusi_rms, rms_ilman_korjausta) - samat
    palautusarvot kuin estimate_radial_distortion_k1:lla.
    """

    raw_img_pts, phys_pts = raw_points_from_converged_result(
        refined, camera_matrix, current_k1
    )

    return estimate_radial_distortion_k1(raw_img_pts, phys_pts, camera_matrix)


def draw_reference_overlay(topdown):

    def to_px(x_cm, y_cm):
        px, py = to_output_px(x_cm, y_cm)
        return int(round(px)), int(round(py))

    p1 = to_px(0.0, OUTPUT_Y_MIN_CM)
    p2 = to_px(0.0, OUTPUT_Y_MAX_CM)
    cv2.line(topdown, p1, p2, (0, 255, 255), 1, cv2.LINE_AA)

    y = OUTPUT_Y_MIN_CM
    while y <= OUTPUT_Y_MAX_CM + 1e-6:

        p1 = to_px(OUTPUT_X_MIN_CM, y)
        p2 = to_px(OUTPUT_X_MAX_CM, y)

        cv2.line(topdown, p1, p2, (60, 60, 60), 1, cv2.LINE_AA)

        cv2.putText(
            topdown, f"{y:.0f} cm", (p1[0] + 5, p1[1] + 15),
            FONT, 0.5, (0, 255, 255), 1, cv2.LINE_AA
        )

        y += 500.0

    for house_y in (NEAR_HOUSE_Y_CM, FAR_HOUSE_Y_CM):

        for radius, color in (
            (BLUE_OUTER_RADIUS_CM, (255, 0, 0)),
            (BLUE_INNER_RADIUS_CM, (0, 255, 0)),
            (RED_OUTER_RADIUS_CM, (0, 0, 255)),
            (RED_INNER_RADIUS_CM, (0, 165, 255)),
        ):

            center_px = to_px(0.0, house_y)
            radius_px = int(round(radius * PIXELS_PER_CM))

            cv2.circle(topdown, center_px, radius_px, color, 1, cv2.LINE_AA)

    cv2.putText(
        topdown, "LAHEMPI PESA", (20, topdown.shape[0] - 20),
        FONT, 0.8, (0, 255, 255), 2, cv2.LINE_AA
    )

    cv2.putText(
        topdown, "KAUKAINEN PESA", (20, 30),
        FONT, 0.8, (0, 255, 255), 2, cv2.LINE_AA
    )

    return topdown


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


def detect_transversal_line(
    topdown_raw, expected_y_cm, band_half_height_cm=40.0,
    min_slope_deg=-30.0, max_slope_deg=30.0,
    contrast_percentile=40.0, ransac_iterations=3000,
    ransac_threshold_px=4.0, min_inlier_fraction=0.4
):
    """
    Etsii POIKITTAISEN (lahes vaakasuoran) viivan - kuten hogline -
    top-down-kuvasta odotetun fyysisen Y-koordinaatin ymparilta.

    Menetelma: jokaiselle kuvan sarakkeelle etsitaan hakukaistaleen
    (band_half_height_cm) sisalta tummin pikseli (hogline on tummempi
    kuin jaa), minka jalkeen naihin ehdokaspisteisiin sovitetaan suora
    RANSAC:lla. RANSAC sietaa paljon poikkeavia pisteita, mika on
    valttamatonta koska top-down-kuvassa (etenkin kaukana) on paljon
    pystysuuntaista raidoituskohinaa, joka muuten dominoisi suoraa
    pienimman nelion sovitusta.

    Palauttaa dictin {"angle_deg", "dir_lateral", "point", "inlier_fraction"}
    tai None, jos luotettavaa viivaa ei loytynyt (esim. inlier-osuus
    liian pieni).

    "point" on piste suoralla top-down-kuvan pystysuoran keskilinjan
    (fyysinen X=0) kohdalla, koko topdown_raw:n pikselikoordinaateissa -
    kayttokelpoinen lisakorrespondenssipisteeksi (fyysinen Y tunnetaan).
    """

    h, w = topdown_raw.shape[:2]

    _, row_center = to_output_px(0.0, expected_y_cm)

    band_half_px = int(band_half_height_cm * PIXELS_PER_CM)
    row_top = max(0, int(row_center - band_half_px))
    row_bottom = min(h, int(row_center + band_half_px))

    if row_bottom - row_top < 10:
        return None

    crop = topdown_raw[row_top:row_bottom, :]

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # Pehmenna enemman pystysuunnassa (kohinanpoisto) kuin vaakasuunnassa
    # (sailytetaan vaakasuuntainen tarkkuus viivan muodon loytamiseksi).
    gray_smooth = cv2.GaussianBlur(gray, (3, 15), 0)

    xs = np.arange(w, dtype=np.float64)
    rows = np.argmin(gray_smooth, axis=0).astype(np.float64)
    row_idx = rows.astype(int)
    vals = gray_smooth[row_idx, np.arange(w)]
    bg = np.percentile(gray_smooth, 85, axis=0)
    contrast = bg - vals

    keep = contrast > np.percentile(contrast, contrast_percentile)

    xs_k = xs[keep]
    ys_k = rows[keep]
    w_k = np.clip(contrast[keep], 1e-3, None)

    if len(xs_k) < 20:
        return None

    min_slope = math.tan(math.radians(min_slope_deg))
    max_slope = math.tan(math.radians(max_slope_deg))

    rng = np.random.default_rng(0)
    n = len(xs_k)

    best_inliers = None
    best_count = -1

    for _ in range(ransac_iterations):

        i, j = rng.choice(n, size=2, replace=False)

        if xs_k[i] == xs_k[j]:
            continue

        a = (ys_k[j] - ys_k[i]) / (xs_k[j] - xs_k[i])

        if a < min_slope or a > max_slope:
            continue

        b = ys_k[i] - a * xs_k[i]

        resid = np.abs(ys_k - (a * xs_k + b))
        inliers = resid < ransac_threshold_px
        count = int(inliers.sum())

        if count > best_count:
            best_count = count
            best_inliers = inliers

    if best_inliers is None:
        return None

    inlier_fraction = float(best_inliers.sum()) / n

    if inlier_fraction < min_inlier_fraction:
        return None

    xin = xs_k[best_inliers]
    yin = ys_k[best_inliers]
    win = w_k[best_inliers]

    A = np.column_stack([xin, np.ones_like(xin)])
    Aw = A * win[:, None]

    (a, b), *_ = np.linalg.lstsq(Aw.T @ A, Aw.T @ yin, rcond=None)

    theta = math.atan(a)
    dir_lateral = np.array([math.cos(theta), math.sin(theta)], dtype=np.float64)

    x_center, _ = to_output_px(0.0, expected_y_cm)
    y_at_center = a * x_center + b

    point_full = np.array(
        [x_center, y_at_center + row_top], dtype=np.float64
    )

    return {
        "angle_deg": math.degrees(theta),
        "dir_lateral": dir_lateral,
        "point": point_full,
        "inlier_fraction": inlier_fraction,
    }


def verify_house_from_topdown(
    view, expected_center, use_ellipse=False, include_red_inner=False
):
    """
    Tunnistaa pesan renkaat SUORAAN top-down-nakymasta osapikseli-
    tarkalla sateittaisella reunanetsinnalla + robustilla sovituksella
    (detect_precise_circle / detect_precise_ellipse), kayttaen
    fyysisesta mittakaavasta (PIXELS_PER_CM) tiedettyja odotettuja
    sateita ja annettua odotettua keskipistetta priorina.

    use_ellipse=False (oletus, sopii kaukaiselle pesalle): sovitetaan
    TAYDELLINEN YMPYRA - kaukainen pesa nakyy top-down-nakymassa jo
    lahes taydellisena ympyrana.

    use_ellipse=True (lahempi pesa): sovitetaan YLEINEN ELLIPSI.
    Lahempi pesa voi nakya top-down-nakymassa hieman soikeana/
    kiertyneena pienen jaljella olevan kalibrointi-/perspektiivi-
    virheen takia - ellipsisovitus antaa talloin huomattavasti
    paremman osuvuuden kuin pakotettu ympyra.

    include_red_inner=True tunnistaa myos red_inner-renkaan (esim.
    kun tarvitaan taydet 4 rengasta / 17 pistetta korjatun
    homografian laskentaan).

    Tama on VISUAALINEN TARKISTUS - tuloksia ei kayteta homografiassa
    eika missaan muussa laskennassa. Alkuperaisen (perspektiivikuvan)
    tunnistuksen maskit/kynnykset eivat muutu.
    """

    blue_score, red_score = create_topdown_score_maps(view)

    detect = detect_precise_ellipse if use_ellipse else detect_precise_circle
    to_ellipse = (lambda e: e) if use_ellipse else circle_to_ellipse

    blue_outer = to_ellipse(detect(
        blue_score, expected_center, BLUE_OUTER_RADIUS_CM * PIXELS_PER_CM
    ))
    blue_inner = to_ellipse(detect(
        blue_score, expected_center, BLUE_INNER_RADIUS_CM * PIXELS_PER_CM
    ))
    red_outer = to_ellipse(detect(
        red_score, expected_center, RED_OUTER_RADIUS_CM * PIXELS_PER_CM
    ))

    red_inner = None

    if include_red_inner:
        red_inner = to_ellipse(detect(
            red_score, expected_center, RED_INNER_RADIUS_CM * PIXELS_PER_CM
        ))

    return {
        "blue_score": blue_score,
        "red_score": red_score,
        "blue_outer": blue_outer,
        "blue_inner": blue_inner,
        "red_outer": red_outer,
        "red_inner": red_inner,
    }


def crop_house_view(topdown, center_y_cm, half_height_cm):

    row_top, row_bottom = compute_crop_row_range(
        topdown.shape[0], center_y_cm, half_height_cm
    )

    return topdown[row_top:row_bottom + 1, :].copy()


# ============================================================
# KORJATTU HOMOGRAFIA TOP-DOWN-TUNNISTUKSISTA
#
# Top-down-nakymassa fyysinen X-akseli on suoraan kuvan X-akseli ja
# fyysinen Y-akseli kasvaa YLOSPAIN kuvassa (Y on kaannetty, katso
# to_output_px) - joten lateraali-/etaisyyssuunnat ovat VAKIOT
# (ei tarvitse paatella T-linjasta/keskilinjasta kuten alkuperaisessa
# kuvassa).
# ============================================================

TOPDOWN_DIR_LATERAL = np.array([1.0, 0.0], dtype=np.float64)
TOPDOWN_DIR_FORWARD = np.array([0.0, -1.0], dtype=np.float64)

# Hog line: poikittainen viiva 6.4 m paassa T-linjasta (WCF-saanto).
# Nama viivat ovat OIKEASTI radalle maalattuja, joten niiden mitattu
# suunta top-down-kuvassa kertoo suoraan, missa suunnassa T-linja/
# keskilinja todella kulkevat - toisin kuin pesan oma ympyra-/
# ellipsisovitus, joka ei yksinaan kerro suuntaa.
HOG_LINE_DISTANCE_CM = 640.0
NEAR_HOGLINE_Y_CM = NEAR_HOUSE_Y_CM + HOG_LINE_DISTANCE_CM
FAR_HOGLINE_Y_CM = FAR_HOUSE_Y_CM - HOG_LINE_DISTANCE_CM


def _hogline_angle_scan(signal, angle_lo_deg, angle_hi_deg, angle_step_deg):
    """
    Kayy lapi annetun kulma-alueen (rotaatio + rivien tummuusprofiilin
    varianssi) ANNETULLE signal-kuvalle - erotettu omaksi funktiokseen
    jotta detect_hogline_angle voi kutsua sita KAHDESTI eri resoluutiolla
    (katso siella oleva perustelu nopeusoptimoinnille).

    Palauttaa (angle_deg, confidence_score).
    """

    best_angle = 0.0
    best_score = -1.0

    for angle_deg in np.arange(angle_lo_deg, angle_hi_deg + 1e-9, angle_step_deg):

        M = cv2.getRotationMatrix2D(
            (signal.shape[1] / 2.0, signal.shape[0] / 2.0), float(angle_deg), 1.0
        )
        rotated = cv2.warpAffine(
            signal, M, (signal.shape[1], signal.shape[0]),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )

        row_profile = rotated.mean(axis=1)
        score = float(row_profile.var())

        if score > best_score:
            best_score = score
            best_angle = float(angle_deg)

    return best_angle, best_score


def detect_hogline_angle(
    topdown_raw, expected_y_cm, half_height_cm=400.0,
    angle_range_deg=35.0, angle_step_deg=0.1,
    edge_margin_frac=0.1, bg_sigma=40.0,
    coarse_max_width_px=180, coarse_step_deg=1.0, fine_range_deg=2.5
):
    """
    Tunnistaa hogline-viivan kulman top-down-kuvasta PROJEKTIOPROFIILI-
    menetelmalla: kuva leikataan kapeaksi vaakakaistaksi oletetun
    hogline-sijainnin ymparilta, kontrasti normalisoidaan (koska
    kaukainen hogline voi olla hyvin himmea), ja kokeillaan eri
    kiertokulmia - oikea kulma tekee rivien tummuusprofiilista
    terävimman (suurimman varianssin), koska silloin viiva keskittyy
    harvimpiin riveihin.

    Talla loydetaan hogline luotettavasti myos silloin, kun se on
    liian himmea/sumea tavalliselle Canny+Hough-viivantunnistukselle.

    HUOM (angle_range_deg=35.0): havaittu testatessa "worst case"
    -kuvalla (kivia/pelaaja radalla, viela osittain korjaamaton H),
    etta kaukaisen hoglinen todellinen kulma voi ennen konvergenssia
    poiketa reilusti (havaittu jopa -25 astetta) - aiempi 20 asteen
    hakuvali TYPISTI tuloksen hakurajaan (piste, jossa pisteytys ei
    ollut viela maksimissaan), mika antoi vaaran kulman HYVALLA
    luottamuspisteytyksella (siis harhaanjohtavan, ei vain epavarman).
    Laajempi hakuvali loytaa oikean, selvasti terävämmän huipun.

    NOPEUSOPTIMOINTI (kamera7_02.py, katso myos SPEED_NOTES.md): tama
    funktio oli profiloinnissa YLIVOIMAISESTI suurin yksittainen
    ajankayttaja koko putkessa (~80/127 s, eli n. 62 % kokonaisajasta) -
    alkuperainen versio kavi 700 kiertokulmaa (35 astetta * 2 / 0.1)
    lapi TAYDELLA resoluutiolla (n. 800x1600 px kaista), joka teki
    700 raskasta cv2.warpAffine-kutsua JOKAISELLE hogline-kutsulle.
    Kaytetaan nyt KARKEA -> TARKKA -hakua (sama periaate kuin
    search_far_house:ssa ja k1-itsekalibroinnissa): ensin karkea haku
    KOKO kulma-alueelta PIENENNETYSTA kuvasta (nopea, karkea resoluutio
    riittaa - etsitaan vain OIKEA NAAPURUSTO, ei tarkkaa kulmaa), sitten
    tarkka haku alkuperaisella tarkkuudella (angle_step_deg) mutta vain
    KAPEALTA (+-fine_range_deg) alueelta karkean tuloksen ymparilta.
    Lopputulos on matemaattisesti kaytannossa sama kulma/pisteytys kuin
    alkuperaisella tayden resoluution/koko-alueen haulla (validoitu
    molemmilla testikuvilla, katso SPEED_NOTES.md), mutta ~15-20x
    nopeampi.

    Palauttaa (angle_deg, confidence_score).
    """

    _, row_f = to_output_px(0.0, expected_y_cm)
    row = int(round(row_f))

    half_px = int(half_height_cm * PIXELS_PER_CM)

    h = topdown_raw.shape[0]
    y1 = max(0, row - half_px)
    y2 = min(h, row + half_px)

    if y2 - y1 < 20:
        return 0.0, 0.0

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
    signal = signal[:, margin:cw - margin]

    # Karkea haku: pienennetty kuva, koko kulma-alue, isolla askeleella.
    ch2, cw2 = signal.shape
    downscale = min(1.0, coarse_max_width_px / max(cw2, 1))

    if downscale < 1.0:
        small_signal = cv2.resize(
            signal, None, fx=downscale, fy=downscale,
            interpolation=cv2.INTER_AREA
        )
    else:
        small_signal = signal

    coarse_angle, _ = _hogline_angle_scan(
        small_signal, -angle_range_deg, angle_range_deg, coarse_step_deg
    )

    # Tarkka haku: talla resoluutiolla, mutta vain kapealta alueelta
    # karkean tuloksen ymparilta - reuna rajataan alkuperaiseen
    # hakuvaliin, jos karkea osuma sattuu olemaan aivan sen reunalla.
    fine_lo = max(-angle_range_deg, coarse_angle - fine_range_deg)
    fine_hi = min(angle_range_deg, coarse_angle + fine_range_deg)

    best_angle, best_score = _hogline_angle_scan(
        signal, fine_lo, fine_hi, angle_step_deg
    )

    return best_angle, best_score


def detect_hogline_points(
    topdown_raw, expected_y_cm, half_height_cm=400.0,
    edge_margin_frac=0.1, bg_sigma=40.0,
    num_segments=14, min_peak_strength=2.0,
    outlier_iterations=3, outlier_mad_multiplier=3.0, min_points=6
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

    Palauttaa listan (x_px, y_px, strength) top-down-kuvan TAYSISSA
    pikselikoordinaateissa (ei kaistan sisaisissa) - vain sailyneet
    (ei-poikkeavat) pisteet poikkeavien hylkayksen jalkeen.
    """

    _, row_f = to_output_px(0.0, expected_y_cm)
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

        peak_row = int(np.argmax(row_profile))
        peak_val = float(row_profile[peak_row])
        baseline = float(np.median(row_profile))
        strength = peak_val - baseline

        if strength < min_peak_strength:
            continue

        # Osapikselitarkka huippu paraabelisovituksella naapuripisteista.
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

    if len(raw_points) < min_points:
        return raw_points

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


def build_hogline_correspondences(
    points, hogline_y_cm, name_prefix, H_current, camera_matrix, k1,
    min_points=6
):
    """
    Muuttaa detect_hogline_points:in loytamat (x_px, y_px, strength)
    -pisteet KORRESPONDENSSIPISTEIKSI homografian sovitukseen: kunkin
    pisteen fyysinen KOHDE on (x_phys, hogline_y_cm) missa x_phys on
    pisteen NYKYINEN (annetun topdown-kuvan oman parhaan arvion
    mukainen) fyysinen x-sijainti (output_px_to_physical) - x:aa EI
    pakoteta mihinkaan tiettyyn arvoon (sita ei tunneta tarkasti ilman
    lisatietoa), mutta y PAKOTETAAN tasan hoglinen oikeaan fyysiseen
    y-arvoon.

    HUOM (TARKEA KOORDINAATISTOMUUNNOS): points on annettu TOPDOWN/
    OUTPUT-PX-koordinaatistossa (koska detect_hogline_points tunnistaa
    ne warpatusta topdown-kuvasta), mutta cv2.findHomography:in muissa
    pisteissa (house_img_pts) kaytetty lahdekoordinaatisto on
    ALKUPERAISEN RAAKAKUVAN (vaaristyneen) pikselikoordinaatisto - naita
    EI SAA sekoittaa keskenaan. Siksi jokainen piste siirretaan tassa
    TAKAISIN raakakuvan koordinaatistoon: topdown-px -> (H_current:in
    KAANTEISHOMOGRAFIA) -> oikaistun raakakuvan px -> (distort_points_px
    samalla k1:lla) -> ALKUPERAISEN raakakuvan px. Tama on sama ketju
    kuin raw_points_from_converged_result:ssa pesan pisteille - ilman
    tata muunnosta (aiemmin puuttui, havaittu ja korjattu testatessa:
    hogline-tasoitus antoi mieletonta -35..+35 asteen heittelya) tulos
    on taysin merkityksetonta, koska undistort_points_px tulkitsisi
    topdown-pikselit virheellisesti raakakuvan pikkeleina.

    Talla on suora vaikutus: kun nama pisteet lisataan SAMAAN
    pienimman nelion homografiasovitukseen pesan rengaspisteiden
    kanssa, sovitus joutuu MYOS suoraan selittamaan "hogline on aidosti
    suora ja vaakasuora" - ei enaa vain valillisesti (pesan L/R-pisteiden
    sijoittelun kautta paateltyna, katso build_topdown_near/far_
    correspondences), vaan SUORAAN koko leveydelta poimituilla
    pisteilla.

    Palauttaa (image_pts, physical_pts, labels) - tyhjat listat jos
    liian vahan pisteita.
    """

    if len(points) < min_points:
        return [], [], []

    topdown_pts = np.array([[p[0], p[1]] for p in points], dtype=np.float64)

    H_inv = np.linalg.inv(H_current)
    frame_undistorted_pts = cv2.perspectiveTransform(
        topdown_pts.reshape(-1, 1, 2), H_inv
    ).reshape(-1, 2)
    raw_img_pts = distort_points_px(frame_undistorted_pts, camera_matrix, k1)

    x_phys_all = output_px_to_physical(topdown_pts)[:, 0]

    image_pts = []
    physical_pts = []
    labels = []

    for i in range(len(points)):
        image_pts.append(raw_img_pts[i])
        physical_pts.append((float(x_phys_all[i]), hogline_y_cm))
        labels.append(f"{name_prefix}_{i}")

    return image_pts, physical_pts, labels


def refine_with_hogline_constraints(
    frame_undistorted, H_start, output_w, output_h, camera_matrix, k1,
    house_img_pts, house_phys_pts, max_iterations=6
):
    """
    Ajaa "lisaa hogline-pisteet mukaan sovitukseen -> tunnista hoglinet
    uudesta kuvasta uudelleen -> toista" -kierroksen (samaan tapaan kuin
    refine_homography_corrections pesille) - koska hogline-pisteiden omat
    fyysiset x-kohteet (build_hogline_correspondences) riippuvat NYKYISESTA
    parhaasta homografiasta, tulos tarkentuu kierros kierrokselta kun
    hoglinet tunnistetaan uudelleen jo hieman korjatusta kuvasta.

    Kayttaa house_img_pts/house_phys_pts:aa (pesien 22 pistetta,
    RAAKAKUVAN - ei topdown:in - koordinaatistossa, distortoimattomina)
    JOKA KIERROKSELLA muuttumattomina - vain hogline-pisteet paivittyvat.

    Palauttaa parhaan loydetyn tuloksen dictina:
      H_final, topdown_corrected_raw, near_angle, far_angle, ratio,
      iterations
    valittu ENSISIJAISESTI pienimman jaljella olevan hogline-kulman
    (lahempi JA kaukainen, suurempi niista) ja TOISSIJAISESTI parhaan
    kaukaisen pesan pyoreyden mukaan (pyoristettyna 0.1 asteen tarkkuu-
    teen, jotta lahes samanarvoiset kulmat eivat syrjayta parempaa
    pyoreytta) - vastaa kayttajan pyyntoa: hoglinet EHDOTTOMASTI
    vaakaan, pesat SEN JALKEEN mahdollisimman pyoreiksi.
    """

    H_current = H_start
    best = None
    best_score = None

    for iteration in range(1, max_iterations + 1):

        topdown = cv2.warpPerspective(
            frame_undistorted, H_current, (output_w, output_h)
        )

        near_hog_pts = detect_hogline_points(topdown, NEAR_HOGLINE_Y_CM)
        far_hog_pts = detect_hogline_points(topdown, FAR_HOGLINE_Y_CM)

        near_img, near_phys, near_labels = build_hogline_correspondences(
            near_hog_pts, NEAR_HOGLINE_Y_CM, "NEAR_HOGPT",
            H_current, camera_matrix, k1
        )
        far_img, far_phys, far_labels = build_hogline_correspondences(
            far_hog_pts, FAR_HOGLINE_Y_CM, "FAR_HOGPT",
            H_current, camera_matrix, k1
        )

        if not near_img and not far_img:
            print(f"  [Hogline-tasoitus {iteration}/{max_iterations}] "
                  f"ei loytynyt riittavasti hogline-pisteita, pysaytetaan.")
            break

        # PAINOTUS (kamera7_04.py): pesan 22 pistetta ovat jo tunnetusti
        # tarkkoja (osapikselitarkka rengassovitus), mutta hogline-
        # pisteita voi kertya YHTA PALJON TAI ENEMMAN (jopa ~10-14 per
        # hogline) - jos molemmat saavat tasan saman painon tavallisessa
        # (painottamattomassa) pienimman nelion sovituksessa, hogline-
        # pisteet (jotka ovat yksittain kohinaisempia, katso detect_
        # hogline_points:in kommentti) voivat HALLITA sovitusta ja
        # tehda siita epavakaan kierros kierrokselta - havaittu
        # testatessa (kaukaisen hoglinen kulma heitteli -34..+2 astetta
        # kierrosten valilla). cv2.findHomography ei tue painoja
        # suoraan, joten paino toteutetaan MONISTAMALLA pesan pisteet
        # HOUSE_POINT_WEIGHT kertaa - tama vastaa TAYSIN painotettua
        # pienimman nelion sovitusta (kokonaislukupainolla) tavallisessa
        # sovituksessa. Arvo 6 loydetty kokeellisesti (skannattu 3-15):
        # pienemmilla (3-5) tulos on epavakaa kierrosten valilla (kaukainen
        # hogline saattaa hypata jopa +-10 astetta), suuremmilla (8-15)
        # pesan pisteet alkavat hallita liikaa eika hogline-korjaus enaa
        # riita (kaukainen hogline jaa 3-4 asteen paahan vaakasuorasta) -
        # 6 antoi molemmilla testikuvilla parhaan YHDISTELMAN vakautta
        # JA korjauksen voimaa.
        HOUSE_POINT_WEIGHT = 6

        all_img = house_img_pts * HOUSE_POINT_WEIGHT + near_img + far_img
        all_phys = house_phys_pts * HOUSE_POINT_WEIGHT + near_phys + far_phys

        src_pts = undistort_points_px(
            np.array(all_img, dtype=np.float64), camera_matrix, k1
        ).astype(np.float32)
        dst_pts = physical_to_output_px(all_phys).astype(np.float32)

        H_new, _ = cv2.findHomography(src_pts, dst_pts, method=0)

        if H_new is None:
            print(f"  [Hogline-tasoitus {iteration}/{max_iterations}] "
                  f"homografian laskenta epaonnistui, pysaytetaan.")
            break

        topdown_new = cv2.warpPerspective(
            frame_undistorted, H_new, (output_w, output_h)
        )

        near_angle, _ = detect_hogline_angle(topdown_new, NEAR_HOGLINE_Y_CM)
        far_angle, _ = detect_hogline_angle(topdown_new, FAR_HOGLINE_Y_CM)
        quality = measure_house_quality(frame_undistorted, H_new)

        max_angle = max(abs(near_angle), abs(far_angle))

        print(f"  [Hogline-tasoitus {iteration}/{max_iterations}] "
              f"lahempi={near_angle:+.3f} deg, kaukainen={far_angle:+.3f} deg "
              f"(pisteita {len(near_img)}+{len(far_img)}), {house_quality_str(quality)}")

        # HUOM (kamera7_05.py): pisteytys kaytti aiemmin VAIN kaukaisen
        # pesan pyoreytta (-ratio) - talloin sisainen kierrosvalinta
        # saattoi valita kierroksen joka rikkoi LAHEMMAN pesan koon/
        # muodon, koska sita ei mitattu lainkaan (katso
        # measure_house_quality:in kommentti). Nyt pisteytys huomioi
        # MOLEMMAT pesat: ensisijaisesti jaljella oleva hogline-kulma,
        # toissijaisesti suurin koko-virhe (pienempi parempi),
        # kolmantena huonoin pyoreys.
        score = (round(max_angle, 1), quality["worst_size_err"], -quality["worst_ratio"])

        if best_score is None or score < best_score:
            best_score = score
            best = {
                "H_final": H_new,
                "topdown_corrected_raw": topdown_new,
                "near_angle": near_angle,
                "far_angle": far_angle,
                "ratio": quality["worst_ratio"],
                "quality": quality,
                "iterations": iteration,
            }

        H_current = H_new

    return best


def direction_from_angle(angle_deg):
    """
    Palauttaa (dir_lateral, dir_forward) yksikkövektorit annetulle
    kulmalle (asteina vaakatasosta). dir_forward on dir_lateral:sta
    -90 astetta kierretty (sama konventio kuin TOPDOWN_DIR_LATERAL/
    TOPDOWN_DIR_FORWARD kulmalla 0).
    """

    theta = math.radians(angle_deg)

    dir_lateral = np.array([math.cos(theta), math.sin(theta)], dtype=np.float64)
    dir_forward = np.array([dir_lateral[1], -dir_lateral[0]], dtype=np.float64)

    return dir_lateral, dir_forward


def build_topdown_near_correspondences(near_ellipses, dir_lateral=None, dir_forward=None):
    """
    Muodostaa lahemman pesan 17 korrespondenssipistetta (keskipiste +
    4 rengasta x 4 pistetta) top-down-nakymasta TARKASTI tunnistetuista
    ellipseista (blue_outer, blue_inner, red_outer, red_inner).

    dir_lateral/dir_forward: T-linjan/keskilinjan oletettu suunta.
    Oletuksena TOPDOWN_DIR_LATERAL/TOPDOWN_DIR_FORWARD (vaaka/pysty),
    mutta tarkempi tulos saadaan antamalla lahimman hogline-viivan
    MITATTU suunta (katso detect_hogline_angle) - talloin pieni
    jaljella oleva kierto lahemmankin pesan kohdalla tulee huomioitua.
    """

    if dir_lateral is None:
        dir_lateral = TOPDOWN_DIR_LATERAL
    if dir_forward is None:
        dir_forward = TOPDOWN_DIR_FORWARD

    ref = (
        near_ellipses.get("blue_outer") or near_ellipses.get("blue_inner")
        or near_ellipses.get("red_outer") or near_ellipses.get("red_inner")
    )

    if ref is None:
        raise RuntimeError(
            "Lahemman pesan top-down-tunnistuksesta ei loytynyt yhtaan "
            "ellipsia - korjattua homografiaa ei voida laskea."
        )

    center_img = np.array(ref[0], dtype=np.float64)

    image_pts = [center_img]
    physical_pts = [(0.0, NEAR_HOUSE_Y_CM)]
    labels = ["NEAR_CENTER_TD"]

    specs = [
        ("NEAR_BLUE_OUTER_TD", near_ellipses.get("blue_outer"), BLUE_OUTER_RADIUS_CM),
        ("NEAR_BLUE_INNER_TD", near_ellipses.get("blue_inner"), BLUE_INNER_RADIUS_CM),
        ("NEAR_RED_OUTER_TD", near_ellipses.get("red_outer"), RED_OUTER_RADIUS_CM),
        ("NEAR_RED_INNER_TD", near_ellipses.get("red_inner"), RED_INNER_RADIUS_CM),
    ]

    for name, ellipse, radius in specs:

        if ellipse is None:
            continue

        pts, phys, labs = ellipse_to_correspondences(
            ellipse, radius, dir_lateral, dir_forward,
            NEAR_HOUSE_Y_CM, name
        )

        image_pts.extend(pts)
        physical_pts.extend(phys)
        labels.extend(labs)

    return image_pts, physical_pts, labels


def build_topdown_far_correspondences(far_ellipses, dir_lateral=None):
    """
    Muodostaa kaukaisen pesan 5 korrespondenssipistetta (keskipiste +
    sinisen ja punaisen ULKOKEHAN vasen/oikea - eli vain T-linjalla
    olevat pisteet) top-down-nakymasta tunnistetuista ympyroista.
    Sisaympyroita EI kayteta, kuten alkuperaisessakaan kuvassa.

    dir_lateral: T-linjan oletettu suunta (yksikkovektori). Oletuksena
    TOPDOWN_DIR_LATERAL (vaakasuora) - mutta koska ympyra on itsessaan
    rotaatiosymmetrinen, taman suunnan TODELLINEN arvo pitaa paatella
    muualta (katso estimate_far_house_t_line_angle).
    """

    if dir_lateral is None:
        dir_lateral = TOPDOWN_DIR_LATERAL

    ref = far_ellipses.get("blue_outer") or far_ellipses.get("red_outer")

    if ref is None:
        raise RuntimeError(
            "Kaukaisen pesan top-down-tunnistuksesta ei loytynyt yhtaan "
            "ellipsia - korjattua homografiaa ei voida laskea."
        )

    center_img = np.array(ref[0], dtype=np.float64)

    image_pts = [center_img]
    physical_pts = [(0.0, FAR_HOUSE_Y_CM)]
    labels = ["FAR_CENTER_TD"]

    specs = [
        ("FAR_BLUE_OUTER_TD", far_ellipses.get("blue_outer"), BLUE_OUTER_RADIUS_CM),
        ("FAR_RED_OUTER_TD", far_ellipses.get("red_outer"), RED_OUTER_RADIUS_CM),
    ]

    for name, ellipse, radius in specs:

        if ellipse is None:
            continue

        pts, phys, labs = ellipse_lateral_correspondences(
            ellipse, radius, dir_lateral, FAR_HOUSE_Y_CM, name
        )

        image_pts.extend(pts)
        physical_pts.extend(phys)
        labels.extend(labs)

    return image_pts, physical_pts, labels


def estimate_far_house_t_line_angle(
    near_img_pts, near_phys_pts, far_ellipses,
    angle_range_deg=30.0, angle_step_deg=0.05
):
    """
    Kaukaisen pesan renkaat sovitetaan YMPYROINA top-down-nakymassa,
    joten ne eivat itsessaan kerro, missa suunnassa T-linja (ja sita
    vastaan kohtisuora keskilinja) siella todellisuudessa kulkee -
    ympyra on rotaatiosymmetrinen.

    Tama funktio paattelee T-linjan suunnan EPASUORASTI: koska
    korjaava homografia on ylimaaratty (22 pistetta / 8 vapausastetta),
    kokeillaan T-linjan suunnaksi eri kulmia (-angle_range..+angle_range,
    askelin angle_step) ja valitaan kulma, joka minimoi KOKO korjatun
    homografian RMS-uudelleenprojisointivirheen (lahempi pesa mukana
    kiintealla, oikealla suunnalla). Oikea kulma tuottaa parhaiten
    globaalisti yhdenmukaisen (yhdella homografialla selittyvan)
    pistejoukon, joten RMS:n minimi paljastaa todellisen T-linjan
    (ja sen normaalina keskilinjan) suunnan kaukaisella pesalla.

    Palauttaa (best_angle_deg, best_rms, best_far_correspondences).
    """

    angles = np.arange(
        -angle_range_deg, angle_range_deg + 1e-9, angle_step_deg
    )

    best_angle = 0.0
    best_rms = float("inf")
    best_far = None

    for angle_deg in angles:

        theta = math.radians(float(angle_deg))
        dir_lateral_candidate = np.array(
            [math.cos(theta), math.sin(theta)], dtype=np.float64
        )

        far_img_pts, far_phys_pts, far_labels = build_topdown_far_correspondences(
            far_ellipses, dir_lateral=dir_lateral_candidate
        )

        all_img_pts = near_img_pts + far_img_pts
        all_phys_pts = near_phys_pts + far_phys_pts

        src = np.array(all_img_pts, dtype=np.float32)
        dst = physical_to_output_px(all_phys_pts).astype(np.float32)

        H_candidate, _ = cv2.findHomography(src, dst, method=0)

        if H_candidate is None:
            continue

        projected = cv2.perspectiveTransform(
            src.reshape(-1, 1, 2), H_candidate
        ).reshape(-1, 2)

        errors = np.linalg.norm(projected - dst, axis=1)
        rms = math.sqrt(float(np.mean(errors ** 2)))

        if rms < best_rms:
            best_rms = rms
            best_angle = float(angle_deg)
            best_far = (far_img_pts, far_phys_pts, far_labels)

    return best_angle, best_rms, best_far


def compute_corrected_homography(frame, H, topdown_raw, near_verify, far_verify):
    """
    Laskee KORJATUN homografian kayttaen top-down-nakymasta tarkasti
    (osapikselitasolla) tunnistettuja pesien renkaita:

      - lahempi pesa: 17 pistetta (keskipiste + 4 rengasta x 4 pistetta,
        ellipsisovitus)
      - kaukainen pesa: 5 pistetta (keskipiste + sinisen ja punaisen
        ULKOKEHAN vasen/oikea, ellipsisovitus)

    Seka lahemman etta kaukaisen pesan T-linjan/keskilinjan TODELLINEN
    suunta top-down-kuvassa PAATELLAAN mittaamalla oikeat, radalle
    maalatut hogline-viivat (6.4 m paassa kummankin pesan T-linjasta,
    katso detect_hogline_angle) - pelkka pesan ympyra-/ellipsisovitus
    EI kerro suuntaa luotettavasti (ympyra on rotaatiosymmetrinen, ja
    ellipsin oma kiertyma voi erota T-linjan todellisesta suunnasta).

    Nama top-down-nakymassa tunnistetut pisteet edustavat sita, MISSA
    pesat OIKEASTI top-down-kuvassa nakyvat (pienine jaljella olevine
    virheineen), ja niita verrataan siihen missa niiden PITAISI olla
    (tunnetut fyysiset sateet). Talla korjataan jaljella oleva
    perspektiivi-/kalibrointivirhe.

    Palauttaa (H_final, topdown_corrected_raw, src_pts, dst_pts, labels)
    - H_final on koko ketjun (alkuperainen kuva -> korjattu top-down)
    yhdistetty homografia.
    """

    near_row_top, _ = compute_crop_row_range(
        topdown_raw.shape[0], NEAR_HOUSE_Y_CM, HOUSE_CROP_HALF_HEIGHT_CM
    )
    far_row_top, _ = compute_crop_row_range(
        topdown_raw.shape[0], FAR_HOUSE_Y_CM, HOUSE_CROP_HALF_HEIGHT_CM
    )

    near_ellipses_full = {
        key: shift_ellipse(near_verify.get(key), dy=near_row_top)
        for key in ("blue_outer", "blue_inner", "red_outer", "red_inner")
    }

    far_ellipses_full = {
        key: shift_ellipse(far_verify.get(key), dy=far_row_top)
        for key in ("blue_outer", "blue_inner", "red_outer", "red_inner")
    }

    print()
    print("=" * 65)
    print("HOGLINE-VIIVOJEN SUUNNAN TUNNISTUS")
    print("(radalle maalatut poikittaiset viivat, 6.4 m T-linjasta -")
    print(" niiden mitattu suunta kertoo T-linjan/keskilinjan")
    print(" TODELLISEN suunnan top-down-kuvassa, myos kaukaisella")
    print(" pesalla jonka oma ympyra/ellipsi ei sita kerro)")
    print("=" * 65)

    near_hog_angle, near_hog_score = detect_hogline_angle(
        topdown_raw, NEAR_HOGLINE_Y_CM
    )
    far_hog_angle_raw, far_hog_score = detect_hogline_angle(
        topdown_raw, FAR_HOGLINE_Y_CM
    )

    # Kaukaisen hoglinen luottamus (far_hog_score) on tyypillisesti
    # PALJON heikompi kuin lahemman (~90-460 vs ~580-670 havaittu
    # testikuvilla) - se on pieni, himmea ja kaukana. Painotetaan
    # kaukaisen pesan KAYTTAMA suunta luottamuspisteiden mukaan: kun
    # kaukaisen oma mittaus on suhteessa heikko, nojataan enemman
    # lahemman (luotettavamman) mittaukseen. Tama on FYYSISESTI
    # PERUSTELTUA (ei mielivaltaista silottelua): molemmat hoglinet
    # ovat radalla AINA tarkalleen yhdensuuntaiset (molemmat kohti-
    # suorassa keskilinjaa vastaan, WCF-saanto), joten lahemman
    # tarkempi mittaus on validi arvio myos kaukaiselle silloin kun
    # kaukaisen oma mittaus on epavarma. Ilman tata painotusta
    # kaukaisen pesan (viela osittain venyneen ellipsin) "vasen/oikea"
    # -pisteet siirtyivat havaittavasti eri kohtiin joka kierroksella
    # pelkan hogline-kohinan takia, mika esti konvergenssin (testattu:
    # RMS heilui 13->16->21 px sen sijaan etta laskisi tasaisesti).
    total_hog_score = near_hog_score + far_hog_score
    far_weight = far_hog_score / total_hog_score if total_hog_score > 1e-6 else 0.5
    far_hog_angle = far_weight * far_hog_angle_raw + (1.0 - far_weight) * near_hog_angle

    print(f"Lahempi hogline  (y={NEAR_HOGLINE_Y_CM:.1f} cm): "
          f"kulma {near_hog_angle:+.2f} deg (luottamus {near_hog_score:.1f})")
    print(f"Kaukainen hogline (y={FAR_HOGLINE_Y_CM:.1f} cm): "
          f"mitattu {far_hog_angle_raw:+.2f} deg, kaytetty (luottamuspainotettu) "
          f"{far_hog_angle:+.2f} deg (far_weight={far_weight:.2f}, "
          f"luottamus {far_hog_score:.1f})")

    near_dir_lateral, near_dir_forward = direction_from_angle(near_hog_angle)
    far_dir_lateral, _ = direction_from_angle(far_hog_angle)

    near_img_pts, near_phys_pts, near_labels = build_topdown_near_correspondences(
        near_ellipses_full, dir_lateral=near_dir_lateral, dir_forward=near_dir_forward
    )

    far_img_pts, far_phys_pts, far_labels = build_topdown_far_correspondences(
        far_ellipses_full, dir_lateral=far_dir_lateral
    )

    all_img_pts = near_img_pts + far_img_pts
    all_phys_pts = near_phys_pts + far_phys_pts
    all_labels = near_labels + far_labels

    src_pts = np.array(all_img_pts, dtype=np.float32)
    dst_pts = physical_to_output_px(all_phys_pts).astype(np.float32)

    print()
    print("=" * 65)
    print("KORJATTU HOMOGRAFIA TOP-DOWN-TUNNISTUKSISTA")
    print("=" * 65)
    print(f"Lahemman pesan pisteita: {len(near_img_pts)} (odotettu 17)")
    print(f"Kaukaisen pesan pisteita: {len(far_img_pts)} (odotettu 5)")
    print(f"Pisteita yhteensa: {len(all_img_pts)}")

    # method=0 (LSQ), ei RANSAC - katso perustelu _homography_rms:n
    # docstringista/kommentista: RANSAC on epavakaa tallä epatasaisesti
    # jakautuneella (17 tiivista + 5 harvaa) pistejoukolla.
    H_correction, _ = cv2.findHomography(src_pts, dst_pts, method=0)

    if H_correction is None:
        raise RuntimeError("Korjaavan homografian laskenta epaonnistui.")

    projected = cv2.perspectiveTransform(
        src_pts.reshape(-1, 1, 2), H_correction
    ).reshape(-1, 2)

    errors = np.linalg.norm(projected - dst_pts, axis=1)

    for label, err in zip(all_labels, errors):
        print(f"  {label:<20}: {err:6.2f} px")

    rms = math.sqrt(float(np.mean(errors ** 2)))

    print(f"RMS virhe: {rms:.2f} px")
    print(f"Max virhe: {float(np.max(errors)):.2f} px")

    H_final = H_correction @ H

    output_h, output_w = topdown_raw.shape[:2]

    topdown_corrected_raw = cv2.warpPerspective(frame, H_final, (output_w, output_h))

    return H_final, topdown_corrected_raw, src_pts, dst_pts, all_labels, rms


def refine_homography_corrections(
    frame_undistorted, H_initial, output_w, output_h,
    max_iterations=None, min_relative_improvement=None
):
    """
    Ajaa "top-down-tunnistus + korjaava homografia" -kierroksen
    (compute_corrected_homography) toistuvasti: jokainen kierros
    tunnistaa pesat/hoglinet UUDESTAAN edellisen kierroksen jo
    osittain korjatusta top-down-kuvasta, joten tulos tarkentuu
    kierros kierrokselta. Pysahtyy kun RMS-uudelleenprojisointivirhe
    ei enaa parane merkittavasti (min_relative_improvement) tai kun
    max_iterations tayttyy.

    Palauttaa dictin:
      H_final, H_before_correction, topdown_src_pts, topdown_dst_pts,
      topdown_raw, topdown_corrected_raw,
      near_verify, far_verify, rms, iterations
    - topdown_raw on SE raaka top-down-kuva josta near_verify/
      far_verify tunnistettiin (eli edeltaa H_final:ia - kayttokel-
      poinen esim. tarkistusylikuvien piirtamiseen johdonmukaisesti).
    - H_before_correction, topdown_src_pts, topdown_dst_pts: se
      homografia (ennen taman kierroksen korjausta) jolla topdown_raw
      tehtiin, seka sen korrespondenssipisteet top-down-pikseleina
      (topdown_src_pts) ja niiden fyysiset kohteet output-px-
      koordinaatistossa (topdown_dst_pts) - kayttokelpoiset esim.
      objektiivin vaaristyman UUDELLEENARVIOINTIIN konvergoituneilla,
      tarkoilla pisteilla (katso refine_lens_distortion_k1).
    """

    if max_iterations is None:
        max_iterations = HOMOGRAPHY_REFINE_MAX_ITERATIONS
    if min_relative_improvement is None:
        min_relative_improvement = HOMOGRAPHY_REFINE_MIN_RELATIVE_IMPROVEMENT

    H_current = H_initial
    best_result = None
    best_rms = float("inf")
    stall_count = 0

    for iteration in range(1, max_iterations + 1):

        topdown_raw = cv2.warpPerspective(
            frame_undistorted, H_current, (output_w, output_h)
        )

        near_view_clean = crop_house_view(
            topdown_raw, NEAR_HOUSE_Y_CM, HOUSE_CROP_HALF_HEIGHT_CM
        )
        far_view_clean = crop_house_view(
            topdown_raw, FAR_HOUSE_Y_CM, HOUSE_CROP_HALF_HEIGHT_CM
        )

        near_expected_center = expected_house_center_in_crop(
            topdown_raw.shape[0], NEAR_HOUSE_Y_CM, HOUSE_CROP_HALF_HEIGHT_CM
        )
        far_expected_center = expected_house_center_in_crop(
            topdown_raw.shape[0], FAR_HOUSE_Y_CM, HOUSE_CROP_HALF_HEIGHT_CM
        )

        near_verify = verify_house_from_topdown(
            near_view_clean, near_expected_center,
            use_ellipse=True, include_red_inner=True
        )
        # HUOM: kaukaiselle pesalle kaytetaan RAJOITETTUA YMPYRAA (ei
        # vapaata/yhteismuotoista ellipsia) KORRESPONDENSSIPISTEIDEN
        # rakentamiseen - vaikka fit_shared_ellipse_shape (katso alla)
        # ONKIN tarkempi kuvaus pesan TODELLISESTA muodosta, sen oma
        # kiertokulma (theta) tekee build_topdown_far_correspondences:in
        # kayttamasta line_ellipse_intersections-menetelmasta HERKAN
        # theta:n tarkkuudelle: L/R-pisteet lasketaan dir_lateral-suoran
        # ja SOVITETUN (mahdollisesti viela vaarin suunnatun, etenkin
        # ennen konvergenssia) ellipsin leikkauksena, joten virhe theta:ssa
        # siirtaa L/R-pisteita systemaattisesti - havaittu testatessa etta
        # tama TEKI iteratiivisesta korjauksesta epavakaamman (RMS/pyoreys
        # huononi molemmilla testikuvilla). YMPYRA ON ROTAATIOINVARIANTTI
        # (L/R-pisteet ovat aina tasan sateen paassa keskipisteesta suun-
        # nasta riippumatta), joten se on vakaampi VALIVAIHEEN pisteiden
        # lahteeksi. Pesan TODELLINEN pyoreys mitataan tarkasti ja
        # rehellisesti (yhteismuotoisella ellipsisovituksella) vasta
        # lopuksi, katso measure_far_house_shape_ratio.
        far_verify = detect_far_house_concentric_circles(
            far_view_clean, far_expected_center
        )

        try:
            H_final, topdown_corrected_raw, src_pts, dst_pts, _, rms = compute_corrected_homography(
                frame_undistorted, H_current, topdown_raw, near_verify, far_verify
            )
        except RuntimeError as exc:
            print(f"[Iteratiivinen korjaus {iteration}/{max_iterations}] "
                  f"epaonnistui ({exc}) - pysaytetaan ja kaytetaan viimeisin "
                  f"onnistunut kierros.")
            break

        print(f"[Iteratiivinen korjaus {iteration}/{max_iterations}] "
              f"RMS: {rms:.3f} px")

        # TARKEA VARMISTUS: pesien/hoglinejen tunnistus top-down-kuvasta
        # on jaljella (etenkin kaukainen pesa on pieni ja matalakontras-
        # tinen), joten yksittainen kierros voi satunnaisesti osua hieman
        # huonompaan paikalliseen minimiin vaikka prosessi kokonaisuutena
        # on viela parantumassa. HYVAKSYTAAN talloin enintaan
        # STALL_PATIENCE perakkaista ei-parantunutta kierrosta ja
        # JATKETAAN niista (jotta prosessi voi "paasta yli" tilapaisesta
        # kohinasta), mutta H_final PALAUTETAAN aina siita kierroksesta
        # jolla RMS oli PARAS NAHTY - ei koskaan huonommasta.
        if rms >= best_rms:

            stall_count += 1

            print(f"  RMS ei parantunut edellisesta kierroksesta "
                  f"({stall_count}/{STALL_PATIENCE}).")

            if stall_count >= STALL_PATIENCE:
                print("  Pysaytetaan ja kaytetaan paras loydetty tulos.")
                break

            # Jatketaan silti TASTA (mahdollisesti huonommasta) H:sta,
            # jotta seuraava kierros voi loytaa paremman lahtokohdan -
            # best_result/best_rms EIVAT paivity, joten paras tulos
            # sailyy tallessa vaikka tama polku ei parantaisikaan.
            H_current = H_final
            continue

        stall_count = 0

        relative_improvement = (
            (best_rms - rms) / best_rms if math.isfinite(best_rms) else None
        )

        best_result = {
            "H_final": H_final,
            "H_before_correction": H_current,
            "topdown_src_pts": src_pts,
            "topdown_dst_pts": dst_pts,
            "topdown_raw": topdown_raw,
            "topdown_corrected_raw": topdown_corrected_raw,
            "near_verify": near_verify,
            "far_verify": far_verify,
            "rms": rms,
            "iterations": iteration,
        }
        best_rms = rms
        H_current = H_final

        if relative_improvement is not None and relative_improvement < min_relative_improvement:
            break

    if best_result is None:
        raise RuntimeError(
            "Korjaavaa homografiaa ei saatu laskettua yhdellakaan "
            "iteraatiokierroksella."
        )

    return best_result


def far_house_shape_ratio(far_verify):
    """
    Palauttaa kaukaisen pesan sovitetun ellipsin PYOREYDEN
    (min(w,h)/max(w,h), 1.0 = taydellinen ympyra) blue_outer:sta,
    tai red_outer:sta jos blue_outer puuttuu. Palauttaa 0.0 jos
    kumpaakaan ei loytynyt.

    Kayttokelpoinen kandidaattien VERTAILUUN: testattaessa havaittiin
    etta pelkka pistekorrespondenssien RMS-uudelleenprojisointivirhe
    EI luotettavasti ennusta nayttaako kaukainen pesa lopulta
    pyorealta - 22 pisteen (17+5) RMS voi olla numeerisesti hyva
    vaikka pesa on visuaalisesti selvasti soikea. Pesan OMA mitattu
    muoto on suora, valitetty mittari sille mita oikeasti halutaan.
    """

    ellipse = far_verify.get("blue_outer") or far_verify.get("red_outer")

    if ellipse is None:
        return 0.0

    (_, _), (w, h), _ = ellipse

    if max(w, h) <= 0:
        return 0.0

    return min(w, h) / max(w, h)


def measure_far_house_shape_ratio(frame_undistorted, H_final):
    """
    Mittaa kaukaisen pesan LOPULLISEN (H_final:lla warpatun) top-down-
    kuvan pyoreyden TUOREELLA tunnistuksella.

    HUOM: refine_homography_corrections:in palauttama "far_verify" EI
    kelpaa tahan - se mittaa muodon SIITA topdown-kuvasta joka oli
    olemassa ENNEN sen kierroksen korjausta (H_current, ei H_final).
    Eli se kuvaa edellisen kierroksen H:n laatua, ei lopullisen H:n
    laatua - havaittu ja korjattu testatessa (pyoreysvertailu antoi
    systemaattisesti vaaria tuloksia ilman tata korjausta).

    HUOM 2: mittaus kayttaa fit_shared_ellipse_shape:aa (sinisen JA
    punaisen ulkokehan YHTEINEN muoto), EI itsenaista yhden renkaan
    vapaata ellipsisovitusta. Havaittu testatessa: itsenainen sovitus
    (pelkasta sinisesta ulkokehasta, ~700 pistetta) yliarvioi
    epakeskisyytta systemaattisesti kohinaisella/pienella kohteella
    (tunnettu ellipsisovituksen harha, katso robust_ellipse_fit:in
    kommentti) - sama kuva antoi pyoreydeksi ~0.80 itsenaisella
    sovituksella, mutta ~0.95-0.99 kun sininen JA punainen ulkokeha
    sovitetaan YHDESSA samaa muotoa jakaen (fyysisesti perusteltua,
    katso fit_shared_ellipse_shape). Tama on siis TARKEMPI, ei
    lievempi, mittari - molemmat renkaat nakyvat jo tassa vaiheessa
    (H_final:n jalkeen), joten mittaus on edelleen riippumaton siita
    miten H_final on laskettu.
    """

    output_w = int(round((OUTPUT_X_MAX_CM - OUTPUT_X_MIN_CM) * PIXELS_PER_CM))
    output_h = int(round((OUTPUT_Y_MAX_CM - OUTPUT_Y_MIN_CM) * PIXELS_PER_CM))

    topdown_final = cv2.warpPerspective(frame_undistorted, H_final, (output_w, output_h))

    far_view = crop_house_view(topdown_final, FAR_HOUSE_Y_CM, HOUSE_CROP_HALF_HEIGHT_CM)
    far_expected_center = expected_house_center_in_crop(
        topdown_final.shape[0], FAR_HOUSE_Y_CM, HOUSE_CROP_HALF_HEIGHT_CM
    )

    far_verify_final = detect_far_house_shared_ellipses(far_view, far_expected_center)

    return far_house_shape_ratio(far_verify_final)


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


def refine_round(frame, camera_matrix, best_k1, frame_undistorted, refined,
                  src_pts_distorted, dst_pts, output_w, output_h, round_num):
    """
    Yksi YHTEISOPTIMOINTIKIERROS (kamera7_05.py, kayttajan pyynnosta:
    "lasketaan H matriisi ja kameran objektiivin virhe peräkkäin/
    samaan aikaan useammin, ja jos arvot paranee, kaytetaan niita
    iteroinnissa").

    Yrittaa perakkain kolme vaihetta, aina nykyisella parhaalla
    (frame_undistorted, refined, best_k1):
      1) objektiivin k1:n uudelleenarviointi konvergoituneista
         pisteista (refine_lens_distortion_k1) + koko putken koeajo
         uudella k1:lla,
      2) suora uudelleensovitus (raw_points_from_converged_result +
         yksi suora findHomography),
      3) hogline-tasoitus (refine_with_hogline_constraints).

    JOKAINEN vaihe hyvaksytaan VAIN candidate_is_better:in kautta -
    talloin MOLEMPIEN pesien pyoreys JA absoluuttinen koko tarkis-
    tetaan aina (katso measure_house_quality), ei enaa pelkka
    kaukaisen pesan pyoreys kuten kamera7_04.py:ssa.

    Palauttaa (changed, best_k1, frame_undistorted, refined) - jos
    changed=False, mikaan kolmesta vaiheesta ei parantanut tulosta
    talla kierroksella, ja kutsuja voi lopettaa yhteisoptimoinnin
    (konvergoitunut).
    """

    changed = False

    # ---- (1) objektiivin k1:n uudelleenarviointi ----

    refined_k1, refined_k1_rms, refined_k1_baseline_rms = refine_lens_distortion_k1(
        refined, camera_matrix, best_k1
    )

    if refined_k1_baseline_rms > 1e-9 and math.isfinite(refined_k1_baseline_rms):
        refined_k1_improvement = (
            (refined_k1_baseline_rms - refined_k1_rms) / refined_k1_baseline_rms
        )
    else:
        refined_k1_improvement = 0.0

    print()
    print(f"--- Kierros {round_num}: objektiivin k1 uudelleenarviointi ---")
    print(f"Nykyinen k1: {best_k1:+.4f}  ehdokas k1: {refined_k1:+.4f} "
          f"(RMS {refined_k1_baseline_rms:.2f} -> {refined_k1_rms:.2f} px, "
          f"parannus {refined_k1_improvement * 100:.1f} %)")

    k1_changed_enough = abs(refined_k1 - best_k1) > K1_MIN_MAGNITUDE
    k1_significant = refined_k1_improvement > 0.0

    if k1_significant and k1_changed_enough:

        trial_dist_coeffs = np.array(
            [refined_k1, 0.0, 0.0, 0.0, 0.0], dtype=np.float64
        )
        trial_frame_undistorted = cv2.undistort(frame, camera_matrix, trial_dist_coeffs)

        trial_src_pts = undistort_points_px(
            src_pts_distorted, camera_matrix, refined_k1
        ).astype(np.float32)

        trial_H, _ = cv2.findHomography(trial_src_pts, dst_pts, method=0)

        if trial_H is None:
            print("   -> Uuden k1:n homografia epaonnistui, sailytetaan alkuperainen.")
        else:
            try:
                trial_refined = refine_homography_corrections(
                    trial_frame_undistorted, trial_H, output_w, output_h
                )
                accept, cand_q, base_q = candidate_is_better(
                    trial_frame_undistorted, trial_refined["H_final"], trial_refined["rms"],
                    frame_undistorted, refined["H_final"], refined["rms"],
                )

                print(f"   Vanha (k1={best_k1:+.4f}): RMS {refined['rms']:.3f} px, "
                      f"{house_quality_str(base_q)}")
                print(f"   Uusi  (k1={refined_k1:+.4f}): RMS {trial_refined['rms']:.3f} px, "
                      f"{house_quality_str(cand_q)}")

                if accept:
                    print("   -> Uusi k1 parantaa tulosta, otetaan kayttoon.")
                    best_k1 = refined_k1
                    frame_undistorted = trial_frame_undistorted
                    refined = trial_refined
                    changed = True
                else:
                    print("   -> Uusi k1 ei parantanut tulosta riittavasti, "
                          "sailytetaan alkuperainen.")

            except RuntimeError as exc:
                print(f"   -> Uuden k1:n kokeilu epaonnistui ({exc}), "
                      f"sailytetaan alkuperainen.")
    else:
        print("   -> Ei parannusta, k1 sailyy ennallaan.")

    # ---- (2) suora uudelleensovitus ----

    raw_img_pts_final, phys_pts_final = raw_points_from_converged_result(
        refined, camera_matrix, best_k1
    )

    direct_src_pts = undistort_points_px(
        raw_img_pts_final, camera_matrix, best_k1
    ).astype(np.float32)
    direct_dst_pts = physical_to_output_px(phys_pts_final).astype(np.float32)

    H_direct, _ = cv2.findHomography(direct_src_pts, direct_dst_pts, method=0)

    print(f"--- Kierros {round_num}: suora uudelleensovitus ---")

    if H_direct is not None:

        projected_direct = cv2.perspectiveTransform(
            direct_src_pts.reshape(-1, 1, 2), H_direct
        ).reshape(-1, 2)
        direct_errors = np.linalg.norm(projected_direct - direct_dst_pts, axis=1)
        direct_rms = math.sqrt(float(np.mean(direct_errors ** 2)))

        accept, cand_q, base_q = candidate_is_better(
            frame_undistorted, H_direct, direct_rms,
            frame_undistorted, refined["H_final"], refined["rms"],
        )

        print(f"   Ketju : RMS {refined['rms']:.3f} px, {house_quality_str(base_q)}")
        print(f"   Suora : RMS {direct_rms:.3f} px, {house_quality_str(cand_q)}")

        if accept:

            print("   -> Suora uudelleensovitus parempi, otetaan kayttoon.")

            topdown_direct_raw = cv2.warpPerspective(
                frame_undistorted, H_direct, (output_w, output_h)
            )

            near_view_direct = crop_house_view(
                topdown_direct_raw, NEAR_HOUSE_Y_CM, HOUSE_CROP_HALF_HEIGHT_CM
            )
            far_view_direct = crop_house_view(
                topdown_direct_raw, FAR_HOUSE_Y_CM, HOUSE_CROP_HALF_HEIGHT_CM
            )
            near_center_direct = expected_house_center_in_crop(
                topdown_direct_raw.shape[0], NEAR_HOUSE_Y_CM, HOUSE_CROP_HALF_HEIGHT_CM
            )
            far_center_direct = expected_house_center_in_crop(
                topdown_direct_raw.shape[0], FAR_HOUSE_Y_CM, HOUSE_CROP_HALF_HEIGHT_CM
            )

            refined = {
                "H_final": H_direct,
                "H_before_correction": H_direct,
                "topdown_raw": topdown_direct_raw,
                "topdown_corrected_raw": topdown_direct_raw,
                "near_verify": verify_house_from_topdown(
                    near_view_direct, near_center_direct,
                    use_ellipse=True, include_red_inner=True
                ),
                "far_verify": detect_far_house_concentric_circles(
                    far_view_direct, far_center_direct
                ),
                "rms": direct_rms,
                "iterations": refined["iterations"],
            }
            changed = True
        else:
            print("   -> Iteratiivinen ketju pysyy parempana, sailytetaan se.")

    # ---- (3) hogline-tasoitus ----

    current_topdown_for_check = cv2.warpPerspective(
        frame_undistorted, refined["H_final"], (output_w, output_h)
    )
    current_near_angle, _ = detect_hogline_angle(
        current_topdown_for_check, NEAR_HOGLINE_Y_CM
    )
    current_far_angle, _ = detect_hogline_angle(
        current_topdown_for_check, FAR_HOGLINE_Y_CM
    )
    current_max_angle = max(abs(current_near_angle), abs(current_far_angle))
    current_quality_hog = measure_house_quality(frame_undistorted, refined["H_final"])

    print(f"--- Kierros {round_num}: hogline-tasoitus ---")
    print(f"Nykytila: lahempi={current_near_angle:+.3f} deg, "
          f"kaukainen={current_far_angle:+.3f} deg, "
          f"{house_quality_str(current_quality_hog)}")

    # Lasketaan pisteet TUOREENA nykyisesta refined:ista (ei vaiheen
    # (2) valimuuttujista) - raw_points_from_converged_result toimii
    # nyt riippumatta siita mika refine_round:in kolmesta vaiheesta
    # refined:in viimeksi tuotti, koska jokainen niista pitaa huolen
    # etta "topdown_raw"/"H_before_correction"/"near_verify"/
    # "far_verify" ovat ajan tasalla (katso raw_points_from_converged_
    # result:in kommentti).
    raw_img_pts_hog, phys_pts_hog = raw_points_from_converged_result(
        refined, camera_matrix, best_k1
    )

    hogline_result = refine_with_hogline_constraints(
        frame_undistorted, refined["H_final"], output_w, output_h,
        camera_matrix, best_k1,
        list(raw_img_pts_hog), list(phys_pts_hog)
    )

    if hogline_result is not None:

        new_max_angle = max(
            abs(hogline_result["near_angle"]), abs(hogline_result["far_angle"])
        )
        cand_quality_hog = hogline_result["quality"]

        print(f"Paras hogline-tasoitettu tulos: lahempi="
              f"{hogline_result['near_angle']:+.3f} deg, kaukainen="
              f"{hogline_result['far_angle']:+.3f} deg, "
              f"{house_quality_str(cand_quality_hog)} "
              f"({hogline_result['iterations']} kierrosta)")

        # Hyvaksytaan jos hogline-kulma parani (pienempi suurin
        # jaljella oleva kulma) EIKA pesien LAATU (pyoreys TAI koko,
        # MOLEMMAT pesat - katso house_quality_is_acceptable)
        # huonontunut merkittavasti.
        angle_improved = new_max_angle < current_max_angle - 0.02
        quality_ok = house_quality_is_acceptable(cand_quality_hog, current_quality_hog)

        if angle_improved and quality_ok:

            print("   -> Hogline-tasoitus parantaa tulosta, otetaan kayttoon.")

            near_view_hog = crop_house_view(
                hogline_result["topdown_corrected_raw"],
                NEAR_HOUSE_Y_CM, HOUSE_CROP_HALF_HEIGHT_CM
            )
            far_view_hog = crop_house_view(
                hogline_result["topdown_corrected_raw"],
                FAR_HOUSE_Y_CM, HOUSE_CROP_HALF_HEIGHT_CM
            )
            near_center_hog = expected_house_center_in_crop(
                hogline_result["topdown_corrected_raw"].shape[0],
                NEAR_HOUSE_Y_CM, HOUSE_CROP_HALF_HEIGHT_CM
            )
            far_center_hog = expected_house_center_in_crop(
                hogline_result["topdown_corrected_raw"].shape[0],
                FAR_HOUSE_Y_CM, HOUSE_CROP_HALF_HEIGHT_CM
            )

            refined = {
                "H_final": hogline_result["H_final"],
                "H_before_correction": hogline_result["H_final"],
                "topdown_raw": hogline_result["topdown_corrected_raw"],
                "topdown_corrected_raw": hogline_result["topdown_corrected_raw"],
                "near_verify": verify_house_from_topdown(
                    near_view_hog, near_center_hog,
                    use_ellipse=True, include_red_inner=True
                ),
                "far_verify": detect_far_house_concentric_circles(
                    far_view_hog, far_center_hog
                ),
                "rms": refined["rms"],
                "iterations": refined["iterations"],
            }
            changed = True
        else:
            print("   -> Hogline-tasoitus ei parantanut tulosta riittavasti, "
                  "sailytetaan alkuperainen.")
    else:
        print("   -> Hogline-pisteita ei loytynyt riittavasti, "
              "sailytetaan alkuperainen.")

    return changed, best_k1, frame_undistorted, refined


def trial_final_rms(frame, image_width, image_height, near_img_pts, near_phys_pts,
                     far_img_pts, far_phys_pts, max_iterations=TRIAL_MAX_ITERATIONS):
    """
    Kevyt "koeajo" kokonaiselle putkelle (k1-itsekalibrointi ->
    oikaisu -> ensimmainen H -> iteratiivinen korjaus) annetulle
    korrespondenssipistejoukolle - palauttaa lopullisen konvergoi-
    tuneen RMS-virheen SEKA kaukaisen pesan lopullisen pyoreyden
    (far_house_shape_ratio). Tulostukset vaimennetaan (tama on
    tarkoitettu KAHDEN kaukaisen pesan korrespondenssikandidaatin
    VERTAILUUN etukateen; putken "oikea" ajo tehdaan vasta
    paremmalle kandidaatille).

    NOPEUSOPTIMOINTI (kamera7_02.py): max_iterations rajoitettu
    oletuksena (TRIAL_MAX_ITERATIONS) alle iteratiivisen korjauksen
    normaalin oletuksen (HOMOGRAPHY_REFINE_MAX_ITERATIONS) - tama
    funktio ajetaan KAHDESTI candidate-vertailua varten ENNEN
    varsinaista (paremmalle kandidaatille tehtavaa) taydella
    iteraatiomaaralla ajettavaa "oikeaa" ajoa, joten sen ei tarvitse
    konvergoida aivan loppuun asti - riittaa etta kumman kandidaatin
    RMS/pyoreys on parempi SAMASSA (rajoitetussa) iteraatiomaarassa,
    koska molempia kandidaatteja verrataan samoin ehdoin. Validoitu:
    kandidaatin valinta pysyy samana molemmilla testikuvilla kuin
    taydella iteraatiomaaralla.

    Palauttaa (rms, shape_ratio) - (float("inf"), 0.0) jos putki
    epaonnistuu talla pistejoukolla.
    """

    all_img_pts = near_img_pts + far_img_pts
    all_phys_pts = near_phys_pts + far_phys_pts

    silence = io.StringIO()

    try:
        with contextlib.redirect_stdout(silence):

            camera_matrix = build_camera_matrix(image_width, image_height)

            best_k1, best_k1_rms, baseline_rms = estimate_radial_distortion_k1(
                all_img_pts, all_phys_pts, camera_matrix
            )

            if baseline_rms > 1e-9 and math.isfinite(baseline_rms):
                k1_improvement = (baseline_rms - best_k1_rms) / baseline_rms
            else:
                k1_improvement = 0.0

            if (
                k1_improvement > K1_MIN_RELATIVE_IMPROVEMENT
                and abs(best_k1) > K1_MIN_MAGNITUDE
            ):
                dist_coeffs = np.array(
                    [best_k1, 0.0, 0.0, 0.0, 0.0], dtype=np.float64
                )
            else:
                best_k1 = 0.0
                dist_coeffs = np.zeros(5, dtype=np.float64)

            frame_undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs)

            src_pts = undistort_points_px(
                np.array(all_img_pts, dtype=np.float64), camera_matrix, best_k1
            ).astype(np.float32)
            dst_pts = physical_to_output_px(all_phys_pts).astype(np.float32)

            H, _ = cv2.findHomography(src_pts, dst_pts, method=0)

            if H is None:
                return float("inf"), 0.0

            output_w = int(round(
                (OUTPUT_X_MAX_CM - OUTPUT_X_MIN_CM) * PIXELS_PER_CM
            ))
            output_h = int(round(
                (OUTPUT_Y_MAX_CM - OUTPUT_Y_MIN_CM) * PIXELS_PER_CM
            ))

            refined = refine_homography_corrections(
                frame_undistorted, H, output_w, output_h,
                max_iterations=max_iterations
            )

        final_ratio = measure_far_house_shape_ratio(
            frame_undistorted, refined["H_final"]
        )

        return refined["rms"], final_ratio

    except RuntimeError:
        return float("inf"), 0.0


# ============================================================
# DEBUG-VISUALISOINNIT
# ============================================================

def build_masks_debug(frame, near_blue_mask, near_red_mask, far_result,
                       roi_polygon, blue_outer, blue_inner,
                       red_outer, red_inner):

    h, w = frame.shape[:2]

    near_color = np.zeros((h, w, 3), dtype=np.uint8)
    near_color[near_blue_mask > 0] = (255, 0, 0)
    near_color[near_red_mask > 0] = (0, 0, 255)

    for ellipse, color in (
        (blue_outer, (0, 255, 0)), (blue_inner, (0, 255, 255)),
        (red_outer, (0, 255, 0)), (red_inner, (0, 255, 255)),
    ):
        if ellipse is not None:
            cv2.ellipse(near_color, ellipse, color, 2, cv2.LINE_AA)

    cv2.putText(
        near_color, "LAHEMPI PESA - MASKI", (20, 40),
        FONT, 0.9, (255, 255, 255), 2, cv2.LINE_AA
    )

    far_color = np.zeros((h, w, 3), dtype=np.uint8)
    far_color[far_result["blue_mask"] > 0] = (255, 0, 0)
    far_color[far_result["red_mask"] > 0] = (0, 0, 255)

    for ellipse, color in (
        (far_result["blue_outer"], (0, 255, 0)),
        (far_result["blue_inner"], (0, 255, 255)),
        (far_result["red_outer"], (0, 255, 0)),
        (far_result["red_inner"], (0, 255, 255)),
    ):
        if ellipse is not None:
            cv2.ellipse(far_color, ellipse, color, 2, cv2.LINE_AA)

    cv2.polylines(
        far_color, [roi_polygon.reshape(-1, 1, 2)],
        True, (0, 255, 255), 2, cv2.LINE_AA
    )

    cv2.putText(
        far_color, "KAUKAINEN PESA - HUE-MASKI (ROI)", (20, 40),
        FONT, 0.8, (255, 255, 255), 2, cv2.LINE_AA
    )

    near_overlay = frame.copy()

    for ellipse, color in (
        (blue_outer, (255, 0, 0)), (blue_inner, (0, 255, 0)),
        (red_outer, (0, 0, 255)), (red_inner, (0, 165, 255)),
    ):
        if ellipse is not None:
            cv2.ellipse(near_overlay, ellipse, color, 2, cv2.LINE_AA)

    cv2.polylines(
        near_overlay, [roi_polygon.reshape(-1, 1, 2)],
        True, (0, 255, 255), 2, cv2.LINE_AA
    )

    cv2.putText(
        near_overlay, "ALKUPERAINEN + KAUKAISEN PESAN ROI", (20, 40),
        FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA
    )

    far_overlay = frame.copy()

    for ellipse, color in (
        (far_result["blue_outer"], (255, 0, 0)),
        (far_result["blue_inner"], (0, 255, 0)),
        (far_result["red_outer"], (0, 0, 255)),
        (far_result["red_inner"], (0, 165, 255)),
    ):
        if ellipse is not None:
            cv2.ellipse(far_overlay, ellipse, color, 2, cv2.LINE_AA)

    cv2.putText(
        far_overlay, "KAUKAISEN PESAN ELLIPSIT", (20, 40),
        FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA
    )

    top = np.hstack([near_color, far_color])
    bottom = np.hstack([near_overlay, far_overlay])

    return np.vstack([top, bottom])


# ============================================================
# PAAOHJELMA
# ============================================================

def main():

    root = tk.Tk()
    root.withdraw()

    filename = filedialog.askopenfilename(
        title="Valitse kuva",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
            ("All files", "*.*"),
        ]
    )

    root.destroy()

    if not filename:
        print("Kuvaa ei valittu.")
        return

    frame = cv2.imread(filename)

    if frame is None:
        raise RuntimeError("Kuvaa ei voitu avata.")

    original = frame.copy()

    image_height, image_width = frame.shape[:2]

    print(f"Kuvan koko: {image_width} x {image_height}")

    # --------------------------------------------------------
    # LAHEMPI PESA - SININEN + PUNAINEN
    # --------------------------------------------------------

    near_blue_mask = create_blue_mask(frame)
    near_red_mask = create_red_mask(frame)

    blue_outer, blue_inner = find_house_pair(
        near_blue_mask, NEAR_BLUE_MIN_AREA, NEAR_BLUE_MIN_RATIO,
        NEAR_BLUE_MIN_SIZE_RATIO
    )

    if blue_outer is None or blue_inner is None:
        raise RuntimeError("Lahemman pesan sinista ei loytynyt kokonaan.")

    red_outer, red_inner = find_house_pair(
        near_red_mask, NEAR_RED_MIN_AREA, NEAR_RED_MIN_RATIO,
        NEAR_RED_MIN_SIZE_RATIO
    )

    if red_outer is None or red_inner is None:
        raise RuntimeError("Lahemman pesan punaista ei loytynyt kokonaan.")

    # --------------------------------------------------------
    # T-LINJA / KESKILINJA
    # --------------------------------------------------------

    segments = detect_line_segments(frame, blue_outer)

    (
        t_pair, centerline_pair, t_line, centerline, house_center
    ) = select_t_and_centerline(segments, blue_outer)

    v_t, v_cl = build_image_directions(t_pair, centerline_pair)

    # --------------------------------------------------------
    # KAMERAKALIBROINTI
    # --------------------------------------------------------

    observations = build_calibration_observations(
        blue_outer, blue_inner, red_outer, red_inner
    )

    if len(observations) < 2:
        raise RuntimeError("Kalibrointiin tarvitaan vahintaan 2 ympyraa.")

    camera = build_camera_model(observations, image_width, image_height)

    print_camera_calibration(camera)

    # --------------------------------------------------------
    # LAHEMMAN PESAN HSV-TEMPLATE
    # --------------------------------------------------------

    template = build_near_house_template(frame, blue_outer, camera, v_t, v_cl)
    template["_hsv"] = template["hsv_model"]["hsv"]

    # --------------------------------------------------------
    # KAUKAISEN PESAN AUTOMAATTINEN HAKU
    # (coarse -> fine -> ultra fine)
    # --------------------------------------------------------

    coarse = search_far_house(
        template, frame, camera, v_t, v_cl,
        center_x=0.0, center_y=0.0,
        center_range=SEARCH_CENTER_RANGE_CM, center_step=COARSE_CENTER_STEP_CM,
        angle_center=0.0, angle_range=SEARCH_CENTER_RANGE_DEG, angle_step=COARSE_ANGLE_STEP_DEG,
        scale_center=1.2, scale_range=SEARCH_CENTER_RANGE_SCALE, scale_step=COARSE_SCALE_STEP,
        description="COARSE +-200 cm / KUVA +-90 deg"
    )

    fine = search_far_house(
        template, frame, camera, v_t, v_cl,
        center_x=coarse["x_offset_cm"], center_y=coarse["y_offset_cm"],
        center_range=FINE_CENTER_RANGE_CM, center_step=FINE_CENTER_STEP_CM,
        angle_center=coarse["angle_deg"], angle_range=FINE_ANGLE_RANGE_DEG,
        angle_step=FINE_ANGLE_STEP_DEG,
        scale_center=coarse["scale"], scale_range=FINE_SCALE_RANGE,
        scale_step=FINE_SCALE_STEP,
        description="FINE"
    )

    optimized = search_far_house(
        template, frame, camera, v_t, v_cl,
        center_x=fine["x_offset_cm"], center_y=fine["y_offset_cm"],
        center_range=ULTRA_CENTER_RANGE_CM, center_step=ULTRA_CENTER_STEP_CM,
        angle_center=fine["angle_deg"], angle_range=ULTRA_ANGLE_RANGE_DEG,
        angle_step=ULTRA_ANGLE_STEP_DEG,
        scale_center=fine["scale"], scale_range=ULTRA_SCALE_RANGE,
        scale_step=ULTRA_SCALE_STEP,
        description="ULTRA FINE"
    )

    x_offset = optimized["x_offset_cm"]
    y_offset = optimized["y_offset_cm"]
    angle_deg = optimized["angle_deg"]
    scale = optimized["scale"]

    print()
    print("=" * 65)
    print("KAUKAISEN PESAN HAUN LOPULLINEN TULOS")
    print("=" * 65)
    print(f"X-siirto : {x_offset:.3f} cm")
    print(f"Y-siirto : {y_offset:.3f} cm")
    print(f"Kulma    : {angle_deg:.3f} deg")
    print(f"Skaala   : {scale:.4f}")
    print(f"Pisteytys: {optimized['score']:.6f}")

    # --------------------------------------------------------
    # SUUNTAVEKTORIT (dir_forward, dir_lateral) HAUN TULOKSESTA
    # Kaytetaan kaukaisen pesan ellipsipisteiden ja lahemman
    # pesan T-/keskilinjapisteiden merkkien maarittamiseen.
    # --------------------------------------------------------

    far_center_raw = project_point(
        x_offset, FAR_HOUSE_Y_CM + y_offset, camera, v_t, v_cl
    )

    far_left_raw = project_point(
        x_offset - HOUSE_RADIUS_CM, FAR_HOUSE_Y_CM + y_offset, camera, v_t, v_cl
    )

    far_right_raw = project_point(
        x_offset + HOUSE_RADIUS_CM, FAR_HOUSE_Y_CM + y_offset, camera, v_t, v_cl
    )

    if far_center_raw is None or far_left_raw is None or far_right_raw is None:
        raise RuntimeError("Kaukaisen pesan projisointi epaonnistui.")

    far_left_x, far_left_y = transform_projected_points(
        far_left_raw[0], far_left_raw[1],
        far_center_raw[0], far_center_raw[1], angle_deg, 1
    )

    far_right_x, far_right_y = transform_projected_points(
        far_right_raw[0], far_right_raw[1],
        far_center_raw[0], far_center_raw[1], angle_deg, 1
    )

    far_left = np.array([float(far_left_x), float(far_left_y)])
    far_right = np.array([float(far_right_x), float(far_right_y)])
    far_center_est = np.asarray(far_center_raw, dtype=np.float64)

    dir_forward = far_center_est - np.asarray(house_center, dtype=np.float64)
    dir_forward /= np.linalg.norm(dir_forward)

    dir_lateral = far_right - far_left
    dir_lateral /= np.linalg.norm(dir_lateral)

    # --------------------------------------------------------
    # 17 PISTEEN KORRESPONDENSSIT - LAHEMPI PESA
    # (siirretty ennen kaukaisen pesan korrespondensseja - ei
    # riippuvuutta, mutta looginen jarjestys: lahempi -> kaukainen)
    # --------------------------------------------------------

    near_img_pts, near_phys_pts, near_labels = near_house_correspondences(
        t_line, centerline, house_center,
        blue_outer, blue_inner, red_outer, red_inner,
        dir_lateral, dir_forward
    )

    # --------------------------------------------------------
    # KAUKAISEN PESAN KORRESPONDENSSIT - KAKSI MENETELMAA,
    # PARAS (LOPULLISEN KONVERGOITUNEEN RMS:N MUKAAN) VOITTAA
    #
    # MENETELMA A ("teoreettinen"): projisoidaan pesan tunnetut
    # reunapisteet SUORAAN search_far_house:n jo optimoiman
    # muunnoksen (x_offset, y_offset, angle_deg, scale) lapi - katso
    # theoretical_far_house_correspondences:in yllapuolinen kommentti.
    # Ei vaadi erillista ellipsin sovitusta kaukaisesta pesasta.
    #
    # MENETELMA B ("raakakuvan hue-maskit", alkuperainen tapa):
    # sovitetaan ellipsi suoraan kaukaisen pesan hue-maskin kontuuriin.
    # Voi epaonnistua tai olla epatarkka aarimmaisen perspektiivi-
    # venytyksen takia, mutta joskus silti tarkempi.
    #
    # Kumpikaan ei ole aina parempi - testataan molemmat ja valitaan
    # se joka antaa paremman LOPULLISEN (koko iteratiivisen korjauksen
    # jalkeisen) RMS-virheen. Kumpikaan menetelma EI riipu mistaan
    # "laajasta" haku-/rektifiointivaiheesta, joten kummallakaan ei
    # ole riskia osua vahingossa toisen radan pesaan - molemmat ovat
    # ankkuroituja search_far_house:n jo validoituun, oikeaan ROI:hin.
    # --------------------------------------------------------

    theoretical_img_pts, theoretical_phys_pts, theoretical_labels = (
        theoretical_far_house_correspondences(
            x_offset, y_offset, angle_deg, scale, camera, v_t, v_cl, far_center_est
        )
    )

    hue_result = create_far_house_hue_masks(
        frame, x_offset, y_offset, camera, v_t, v_cl, angle_deg, scale
    )
    far_result = find_far_house_ellipses_from_hue(hue_result)

    raw_img_pts, raw_phys_pts, raw_labels, raw_center_img = (
        far_house_correspondences_from_ellipses(
            far_result, dir_lateral, dir_forward, far_center_est
        )
    )

    print()
    print("=" * 65)
    print("KAUKAISEN PESAN KORRESPONDENSSIT - KAKSI MENETELMAA")
    print("=" * 65)
    for name in ("blue_outer", "blue_inner", "red_outer", "red_inner"):
        found = far_result[name] is not None
        print(f"  hue-maski {name:<12}: {'loytyi' if found else 'EI loytynyt'}")

    candidates = [
        ("teoreettinen (haun muunnos)", theoretical_img_pts, theoretical_phys_pts,
         theoretical_labels, theoretical_img_pts[0]),
    ]

    if len(raw_img_pts) > 1:
        candidates.append((
            "raakakuvan hue-maskit", raw_img_pts, raw_phys_pts, raw_labels,
            raw_center_img
        ))

    # HUOM (havaittu testatessa): pelkka pistekorrespondenssien RMS-
    # uudelleenprojisointivirhe EI luotettavasti ennusta nayttaako
    # kaukainen pesa lopulta pyorealta top-down-kuvassa - se voi olla
    # numeerisesti hyva vaikka pesa on visuaalisesti selvasti soikea.
    # Valitaan siis ENSISIJAISESTI se kandidaatti, jonka kaukainen
    # pesa on LOPUKSI PYOREIN (far_house_shape_ratio, 1.0 = taydellinen
    # ympyra) - RMS kaytetaan vain tasapelin ratkaisijana kun pyoreys
    # on lahes sama (< 0.03 ero).
    best_choice = None
    best_choice_rms = float("inf")
    best_choice_ratio = -1.0

    for name, f_img, f_phys, f_labels, f_center in candidates:

        rms, ratio = trial_final_rms(
            frame, image_width, image_height,
            near_img_pts, near_phys_pts, f_img, f_phys
        )

        print(f"  {name:<28}: {len(f_img)} pistetta, "
              f"lopullinen RMS {rms:.2f} px, kaukaisen pesan pyoreys {ratio:.3f}")

        is_better = (
            ratio > best_choice_ratio + 0.03
            or (abs(ratio - best_choice_ratio) <= 0.03 and rms < best_choice_rms)
        )

        if is_better:
            best_choice_rms = rms
            best_choice_ratio = ratio
            best_choice = (f_img, f_phys, f_labels, f_center)

    far_img_pts, far_phys_pts, far_labels, far_center_img = best_choice

    print(f"-> Valittiin parempi menetelma (pyoreys {best_choice_ratio:.3f}, "
          f"RMS {best_choice_rms:.2f} px, {len(far_img_pts)} pistetta).")

    all_img_pts = near_img_pts + far_img_pts
    all_phys_pts = near_phys_pts + far_phys_pts
    all_labels = near_labels + far_labels

    print()
    print(f"Lahemman pesan pisteita : {len(near_img_pts)} (odotettu 17)")
    print(f"Kaukaisen pesan pisteita: {len(far_img_pts)}")
    print(f"Pisteita yhteensa       : {len(all_img_pts)}")

    # --------------------------------------------------------
    # OBJEKTIIVIN VAARISTYMAN ITSEKALIBROINTI (k1)
    #
    # Haetaan JOKA KUVASTA ERIKSEEN se sateittainen vaaristyma, joka
    # selittaa jo mitatut pesa-/hogline-korrespondenssipisteet
    # parhaiten YHDELLA homografialla. Ei vaadi kalibrointikuvia, ja
    # toimii vaikka objektiivin zoomaus vaihtelisi kuvien valilla.
    # --------------------------------------------------------

    camera_matrix = build_camera_matrix(image_width, image_height)

    best_k1, best_k1_rms, baseline_rms = estimate_radial_distortion_k1(
        all_img_pts, all_phys_pts, camera_matrix
    )

    if baseline_rms > 1e-9 and math.isfinite(baseline_rms):
        k1_improvement = (baseline_rms - best_k1_rms) / baseline_rms
    else:
        k1_improvement = 0.0

    print()
    print("=" * 65)
    print("OBJEKTIIVIN SATEITTAISEN VAARISTYMAN ITSEKALIBROINTI")
    print("=" * 65)
    print(f"RMS ilman korjausta (k1=0) : {baseline_rms:.2f} px")
    print(f"Paras loydetty k1          : {best_k1:+.4f}")
    print(f"RMS korjauksen jalkeen     : {best_k1_rms:.2f} px")
    print(f"Parannus                   : {k1_improvement * 100:.1f} %")

    if (
        k1_improvement > K1_MIN_RELATIVE_IMPROVEMENT
        and abs(best_k1) > K1_MIN_MAGNITUDE
    ):
        dist_coeffs = np.array([best_k1, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        print("-> Merkittava vaaristyma havaittu, otetaan kayttoon.")
    else:
        best_k1 = 0.0
        dist_coeffs = np.zeros(5, dtype=np.float64)
        print("-> Ei merkittavaa vaaristymaa, jatketaan ilman korjausta.")

    frame_undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs)

    # --------------------------------------------------------
    # HOMOGRAFIA
    # --------------------------------------------------------

    src_pts_distorted = np.array(all_img_pts, dtype=np.float64)
    src_pts = undistort_points_px(
        src_pts_distorted, camera_matrix, best_k1
    ).astype(np.float32)
    dst_pts = physical_to_output_px(all_phys_pts).astype(np.float32)

    # method=0 (LSQ), ei RANSAC - katso perustelu _homography_rms:n
    # kommentista: RANSAC on epavakaa tallä epatasaisesti jakautuneella
    # (17 tiivista lahempaa + 5 harvaa kaukaista) pistejoukolla.
    H, _ = cv2.findHomography(src_pts, dst_pts, method=0)

    if H is None:
        raise RuntimeError("Homografian laskenta epaonnistui.")

    projected = cv2.perspectiveTransform(
        src_pts.reshape(-1, 1, 2), H
    ).reshape(-1, 2)

    errors = np.linalg.norm(projected - dst_pts, axis=1)

    print()
    print(f"Uudelleenprojisointivirhe ({PIXELS_PER_CM} px/cm):")

    for label, err in zip(all_labels, errors):
        print(f"  {label:<20}: {err:6.2f} px")

    print(f"RMS virhe: {math.sqrt(float(np.mean(errors ** 2))):.2f} px")
    print(f"Max virhe: {float(np.max(errors)):.2f} px")

    # --------------------------------------------------------
    # TOP-DOWN-KUVA (lahempi ALHAALLA, kaukainen YLHAALLA)
    # --------------------------------------------------------

    output_w = int(round((OUTPUT_X_MAX_CM - OUTPUT_X_MIN_CM) * PIXELS_PER_CM))
    output_h = int(round((OUTPUT_Y_MAX_CM - OUTPUT_Y_MIN_CM) * PIXELS_PER_CM))

    topdown_raw = cv2.warpPerspective(frame_undistorted, H, (output_w, output_h))

    topdown = topdown_raw.copy()
    draw_reference_overlay(topdown)

    cv2.imwrite(TOPDOWN_OUTPUT, topdown)

    print()
    print(f"Top-down -kuva tallennettu: {TOPDOWN_OUTPUT}")
    print(f"Kuvan koko: {output_w} x {output_h} px")

    # --------------------------------------------------------
    # LAHEMMAN JA KAUKAISEN PESAN OMAT YLHAALTA-NAKYMAT
    # --------------------------------------------------------

    near_view = crop_house_view(topdown, NEAR_HOUSE_Y_CM, HOUSE_CROP_HALF_HEIGHT_CM)
    far_view = crop_house_view(topdown, FAR_HOUSE_Y_CM, HOUSE_CROP_HALF_HEIGHT_CM)

    cv2.imwrite(NEAR_HOUSE_VIEW_OUTPUT, near_view)
    cv2.imwrite(FAR_HOUSE_VIEW_OUTPUT, far_view)

    print(f"Lahemman pesan ylhaalta-nakyma : {NEAR_HOUSE_VIEW_OUTPUT}")
    print(f"Kaukaisen pesan ylhaalta-nakyma: {FAR_HOUSE_VIEW_OUTPUT}")

    # --------------------------------------------------------
    # VARMISTUS: TUNNISTA SEKA LAHEMMAN ETTA KAUKAISEN PESAN
    # ELLIPSIT SUORAAN TOP-DOWN-NAKYMASTA.
    #
    # Kaytetaan PUHTAITA (ilman keltaisia/harmaita/varillisia
    # referenssiviivoja) leikkauksia topdown_raw:sta, jotta
    # referenssiympyrat eivat sekoita tunnistusta. Tama on
    # PELKKA VISUAALINEN TARKISTUS - tuloksia ei kayteta
    # mihinkaan laskentaan, eika alkuperainen (perspektiivikuvan)
    # tunnistus muutu.
    # --------------------------------------------------------

    near_view_clean = crop_house_view(topdown_raw, NEAR_HOUSE_Y_CM, HOUSE_CROP_HALF_HEIGHT_CM)
    far_view_clean = crop_house_view(topdown_raw, FAR_HOUSE_Y_CM, HOUSE_CROP_HALF_HEIGHT_CM)

    near_expected_center = expected_house_center_in_crop(
        topdown_raw.shape[0], NEAR_HOUSE_Y_CM, HOUSE_CROP_HALF_HEIGHT_CM
    )
    far_expected_center = expected_house_center_in_crop(
        topdown_raw.shape[0], FAR_HOUSE_Y_CM, HOUSE_CROP_HALF_HEIGHT_CM
    )

    near_verify = verify_house_from_topdown(
        near_view_clean, near_expected_center,
        use_ellipse=True, include_red_inner=True
    )
    far_verify = verify_house_from_topdown(
        far_view_clean, far_expected_center,
        use_ellipse=True, include_red_inner=False
    )

    print()
    print("=" * 65)
    print("PESIEN TARKISTUS TOP-DOWN-NAKYMISTA")
    print("(vain visuaalinen tarkistus, ei kayteta homografiaan sellaisenaan -")
    print(" naista lasketaan kuitenkin KORJATTU homografia alempana)")
    print("=" * 65)

    for label, verify_result, ring_names in (
        ("LAHEMPI", near_verify, ("blue_outer", "blue_inner", "red_outer", "red_inner")),
        ("KAUKAINEN", far_verify, ("blue_outer", "blue_inner", "red_outer")),
    ):
        print(f"{label} PESA:")
        for name in ring_names:
            found = verify_result[name] is not None
            print(f"    {name:<12}: {'loytyi' if found else 'EI loytynyt'}")

    def draw_verify_overlay(view_clean, verify_result, title, draw_red_inner=False):

        result_img = view_clean.copy()

        rings = [
            (verify_result["blue_outer"], (255, 0, 0)),
            (verify_result["blue_inner"], (0, 255, 0)),
            (verify_result["red_outer"], (0, 0, 255)),
        ]

        if draw_red_inner:
            rings.append((verify_result["red_inner"], (0, 165, 255)))

        for ellipse, color in rings:
            if ellipse is not None:
                cv2.ellipse(result_img, ellipse, color, 2, cv2.LINE_AA)

        cv2.putText(
            result_img, title, (15, 30),
            FONT, 0.7, (0, 255, 255), 2, cv2.LINE_AA
        )

        return result_img

    # --------------------------------------------------------
    # KORJATTU HOMOGRAFIA top-down-tunnistuksista (17 + 5 pistetta),
    # AJETTUNA ITERATIIVISESTI kunnes RMS-virhe ei enaa parane
    # merkittavasti (katso refine_homography_corrections).
    # --------------------------------------------------------

    refined = refine_homography_corrections(
        frame_undistorted, H, output_w, output_h
    )

    print()
    print(f"Iteratiivinen korjaus pysahtyi {refined['iterations']} "
          f"kierroksen jalkeen (RMS {refined['rms']:.3f} px).")

    # --------------------------------------------------------
    # YHTEISOPTIMOINTI (kamera7_05.py): H-matriisi + objektiivin k1
    # vuorotellen, USEAMPI KIERROS.
    #
    # Kayttajan pyynnosta (kamera7_04.py:n hogline-tasoitus rikkoi
    # lahemman pesan koon, koska hyvaksynta tarkisti vain kaukaisen
    # pesan pyoreyden - katso measure_house_quality:in kommentti):
    # objektiivin k1 ja homografia (suora uudelleensovitus + hogline-
    # tasoitus) lasketaan nyt PERAKKAIN, USEAMMAN KIERROKSEN ajan
    # (refine_round), ja jokainen vaihe hyvaksytaan VAIN jos se
    # OIKEASTI parantaa tulosta huonontamatta kummankaan pesan
    # pyoreytta TAI absoluuttista kokoa (candidate_is_better /
    # house_quality_is_acceptable - tarkistaa MOLEMMAT pesat, ei enaa
    # vain kaukaista). Kierroksia jatketaan kunnes yksikaan kolmesta
    # vaiheesta ei enaa tuo parannusta (konvergoitunut) tai
    # MAX_JOINT_ROUNDS tayttyy.
    # --------------------------------------------------------

    MAX_JOINT_ROUNDS = 4

    print()
    print("=" * 65)
    print("YHTEISOPTIMOINTI: H-matriisi + objektiivin k1 vuorotellen")
    print("=" * 65)

    for round_num in range(1, MAX_JOINT_ROUNDS + 1):

        print()
        print(f"### Yhteisoptimointikierros {round_num}/{MAX_JOINT_ROUNDS} ###")

        changed, best_k1, frame_undistorted, refined = refine_round(
            frame, camera_matrix, best_k1, frame_undistorted, refined,
            src_pts_distorted, dst_pts, output_w, output_h, round_num
        )

        if not changed:
            print(f"-> Kierros {round_num}: ei enaa parannusta, "
                  f"yhteisoptimointi konvergoitunut.")
            break
    else:
        print(f"-> Saavutettiin maksimikierrosmaara "
              f"({MAX_JOINT_ROUNDS}), pysaytetaan yhteisoptimointi.")

    H_final = refined["H_final"]
    topdown_corrected_raw = refined["topdown_corrected_raw"]

    # Paivitetaan tarkistuskuvat vastaamaan VIIMEISTA kierrosta (eika
    # vain ensimmaista, esikorjaamatonta homografiaa) - muuten
    # tallennettu tarkistuskuva ja siihen piirretyt ellipsit eivat
    # vastaisi toisiaan.
    near_view_clean = crop_house_view(
        refined["topdown_raw"], NEAR_HOUSE_Y_CM, HOUSE_CROP_HALF_HEIGHT_CM
    )
    far_view_clean = crop_house_view(
        refined["topdown_raw"], FAR_HOUSE_Y_CM, HOUSE_CROP_HALF_HEIGHT_CM
    )
    near_verify = refined["near_verify"]
    far_verify = refined["far_verify"]

    near_view_verify = draw_verify_overlay(
        near_view_clean, near_verify, "TARKISTUS (kayt. korjattuun H:hon)", draw_red_inner=True
    )
    far_view_verify = draw_verify_overlay(
        far_view_clean, far_verify, "TARKISTUS (kayt. korjattuun H:hon)"
    )

    cv2.imwrite(NEAR_HOUSE_VERIFY_OUTPUT, near_view_verify)
    cv2.imwrite(FAR_HOUSE_VERIFY_OUTPUT, far_view_verify)

    print(f"Lahemman pesan tarkistuskuva : {NEAR_HOUSE_VERIFY_OUTPUT}")
    print(f"Kaukaisen pesan tarkistuskuva: {FAR_HOUSE_VERIFY_OUTPUT}")

    topdown_corrected = topdown_corrected_raw.copy()
    draw_reference_overlay(topdown_corrected)

    cv2.imwrite(CORRECTED_TOPDOWN_OUTPUT, topdown_corrected)

    print(f"Korjattu top-down -kuva: {CORRECTED_TOPDOWN_OUTPUT}")

    # --------------------------------------------------------
    # DEBUG: MASKIT
    # --------------------------------------------------------

    masks_debug = build_masks_debug(
        frame, near_blue_mask, near_red_mask, far_result,
        hue_result["roi_polygon"], blue_outer, blue_inner, red_outer, red_inner
    )

    cv2.imwrite(MASKS_DEBUG_OUTPUT, masks_debug)

    print(f"Maskien debugkuva: {MASKS_DEBUG_OUTPUT}")

    # --------------------------------------------------------
    # DEBUG: KAUKAISEN PESAN HAKU
    # --------------------------------------------------------

    far_search_debug = original.copy()

    cv2.ellipse(far_search_debug, blue_outer, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.ellipse(far_search_debug, blue_inner, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.ellipse(far_search_debug, red_outer, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.ellipse(far_search_debug, red_inner, (0, 165, 255), 2, cv2.LINE_AA)

    for ellipse, color in (
        (far_result["blue_outer"], (255, 0, 0)),
        (far_result["blue_inner"], (0, 255, 0)),
        (far_result["red_outer"], (0, 0, 255)),
        (far_result["red_inner"], (0, 165, 255)),
    ):
        if ellipse is not None:
            cv2.ellipse(far_search_debug, ellipse, color, 2, cv2.LINE_AA)

    for p, color, label in (
        (far_left, (0, 255, 255), "FAR_LEFT"),
        (far_right, (0, 255, 255), "FAR_RIGHT"),
        (far_center_img, (0, 0, 255), "FAR_CENTER"),
    ):
        q = (int(round(p[0])), int(round(p[1])))
        cv2.circle(far_search_debug, q, 6, color, -1)
        cv2.putText(
            far_search_debug, label, (q[0] + 8, q[1] - 8),
            FONT, 0.5, color, 1, cv2.LINE_AA
        )

    cv2.polylines(
        far_search_debug, [hue_result["roi_polygon"].reshape(-1, 1, 2)],
        True, (0, 200, 200), 2, cv2.LINE_AA
    )

    cv2.imwrite(DEBUG_FAR_SEARCH_OUTPUT, far_search_debug)

    print(f"Kaukaisen pesan haku -debugkuva: {DEBUG_FAR_SEARCH_OUTPUT}")

    # --------------------------------------------------------
    # DEBUG: KAIKKI KORRESPONDENSSIPISTEET
    # --------------------------------------------------------

    debug = original.copy()

    for ellipse, color in (
        (blue_outer, (255, 0, 0)), (blue_inner, (0, 255, 0)),
        (red_outer, (0, 0, 255)), (red_inner, (0, 165, 255)),
    ):
        if ellipse is not None:
            cv2.ellipse(debug, ellipse, color, 2, cv2.LINE_AA)

    for ellipse, color in (
        (far_result["blue_outer"], (255, 0, 0)),
        (far_result["blue_inner"], (0, 255, 0)),
        (far_result["red_outer"], (0, 0, 255)),
        (far_result["red_inner"], (0, 165, 255)),
    ):
        if ellipse is not None:
            cv2.ellipse(debug, ellipse, color, 2, cv2.LINE_AA)

    for p in near_img_pts:
        q = tuple(np.round(p).astype(int))
        cv2.circle(debug, q, 4, (255, 255, 0), -1)

    for p, label in zip(far_img_pts, far_labels):
        q = tuple(np.round(p).astype(int))
        cv2.circle(debug, q, 5, (0, 255, 255), -1)
        cv2.putText(
            debug, label, (q[0] + 6, q[1] - 6),
            FONT, 0.35, (0, 255, 255), 1, cv2.LINE_AA
        )

    cv2.imwrite(DEBUG_POINTS_OUTPUT, debug)

    print(f"Kaikkien pisteiden debugkuva: {DEBUG_POINTS_OUTPUT}")

    # --------------------------------------------------------
    # NAYTA
    # --------------------------------------------------------

    cv2.imshow("Lahemman pesan tarkistus (ellipsisovitus)", near_view_verify)
    cv2.imshow("Kaukaisen pesan tarkistus (ympyransovitus)", far_view_verify)
    cv2.imshow("Koko rata ylhaalta - KORJATTU H (lahempi alhaalla)", topdown_corrected)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()