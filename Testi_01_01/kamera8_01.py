import cv2
import numpy as np
import math
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


def refine_geometric_homography(frame_undistorted, H_init, near_pts, near_phys,
                                 output_w, output_h,
                                 max_iterations=None, min_relative_improvement=None):
    """
    Toistaa "tunnista kaukaisen pesan renkaat + hoglinet nykyisesta
    top-down-kuvasta -> ratkaise H uudelleen geometrisilla rajoitteilla
    (solve_homography_geometric) -> toista" kunnes muutos ei ole enaa
    merkittava - sama konvergenssiperiaate kuin kamera7_06.py:n
    refine_homography_corrections:lla, mutta rajoitteiden LAATU on nyt
    oikea (ei suunta-arvauksia kaukaiselle pesalle eika hoglinelle).

    near_pts/near_phys (lahemman pesan 17 pistetta, oikaistun kuvan
    koordinaatistossa) ovat KIINTEAT koko ajan - ne on tunnistettu
    KERTAALLEEN raakakuvasta T-linjan/keskilinjan suoralla leikkauksella
    (near_house_correspondences, main()), eivat vaadi uudelleen-
    tunnistusta eivatka suunta-arvausta missaan vaiheessa.

    HUOM (rehellisesti raportoitu rajoitus): koska kaukaisen pesan
    ympyrarajoite on TAHALLAAN rotaatioinvariantti (ei kerro T-linjan
    suuntaa), kaukaisen kentan KIERTO jaa kokonaan hoglinien varaan.
    Jos hoglinen mittaus on jollain kuvalla epaluotettava (esim. hyvin
    kaukana, himmea), kaukaisen paan kierto voi jaada osittain vaaraksi
    vaikka pesien muoto/koko olisivatkin jo hyvia - tama on todettu
    (Kivilla.png) eika sita peitella: katso main()in lopullinen,
    suodattamaton raportointi.

    Palauttaa dictin: H_final, topdown_raw, rms, quality, iterations.
    """

    if max_iterations is None:
        max_iterations = HOMOGRAPHY_REFINE_MAX_ITERATIONS
    if min_relative_improvement is None:
        min_relative_improvement = HOMOGRAPHY_REFINE_MIN_RELATIVE_IMPROVEMENT

    H_current = H_init
    best = None
    best_rms = float("inf")
    stall_count = 0

    for iteration in range(1, max_iterations + 1):

        topdown_raw = cv2.warpPerspective(frame_undistorted, H_current, (output_w, output_h))

        far_pts, far_radius = far_house_ring_points_in_frame(topdown_raw, H_current)
        near_hog_pts, near_hog_w = hogline_points_in_frame(topdown_raw, NEAR_HOGLINE_Y_CM, H_current)
        far_hog_pts, far_hog_w = hogline_points_in_frame(topdown_raw, FAR_HOGLINE_Y_CM, H_current)

        # JOHDONMUKAISUUSTARKISTUS: molemmat hoglinet ovat radalla AINA
        # tarkalleen yhdensuuntaiset (WCF-saanto). Jos lahemman ja
        # kaukaisen kulmien ero on suuri (tallakin kierroksella viela
        # osittain vaaralla H:lla mitattuna), kaukaisen mittaus on liian
        # epavarma ohjaamaan tata kierrosta - jatetaan se pois taman
        # kierroksen sovituksesta ja luotetaan lahempaan (havaittu
        # vakauttavan iteraatiota - Kivilla.png ilman tata pysahtyy
        # aikaisemmin, huonompaan tulokseen).
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

        # Kaukaisen renkaan paino: NIIDEN OMASTA residuaalihajonnasta
        # NYKYISELLA H_current:lla arvioitu (inverse-variance) kohina-
        # taso, ylhaalta rajattu lahemman pesan luottamustasolla
        # (NEAR_TRUST_FLOOR_CM) - katso _robust_scale_cm:in kommentti.
        far_resid_now = _far_ring_distance(H_current, far_pts, far_radius)
        far_scale = _robust_scale_cm(far_resid_now, NEAR_TRUST_FLOOR_CM)
        far_weight = 1.0 / far_scale

        # Hoglinen paino EI ole enaa itsekalibroitu jaljella olevasta
        # residuaalihajonnasta (kayttajan huomio: kaukainen hogline
        # PITAA maarata H:ta - l' = H^-T l - eika sen anneta vaientua
        # vain siksi etta nykyinen H viela vaarasti pyorittaa sen).
        # Nyt kun detect_hogline_points valitsee klusterin OIKEIN
        # (keskiviivaekstrapolointi nimellista riviä vasten, ei enaa
        # pistemaaran enemmisto - katso funktion kommentti), pisteet
        # OVAT todistetusti oikea fyysinen hogline eivatka kontaminaatio
        # - jaljella oleva iso residuaali on siis AITO, KORJATTAVA
        # kiertovirhe H:ssa, ei kohinaa, eika sita pida sekoittaa
        # kaukaisen renkaan kanssa samaan itsekalibrointiin (jonka
        # residuaali ON aitoa kohinaa). Paino on siis KIINTEA, samalla
        # luottamustasolla kuin lahempi pesa (NEAR_TRUST_FLOOR_CM).
        hog_weight = hog_strength / NEAR_TRUST_FLOOR_CM

        H_new = solve_homography_geometric(
            H_current, near_pts, near_phys, far_pts, far_radius, far_weight,
            hog_pts, hog_y, hog_weight
        )

        # Lahemman pesan 17 (tarkasti tunnetun, muuttumattoman) pisteen
        # RMS-uudelleenprojisointivirhe - vertailukelpoinen mittari
        # konvergenssin seurantaan kierrosten valilla.
        near_proj = output_px_to_physical(_apply_h(H_new, near_pts))
        rms = math.sqrt(float(np.mean(np.sum((near_proj - near_phys) ** 2, axis=1))))

        quality = measure_house_quality(frame_undistorted, H_new)

        print(f"[Geometrinen korjaus {iteration}/{max_iterations}] "
              f"lahempi RMS: {rms:.4f} cm, {house_quality_str(quality)} "
              f"(kaukaisen renkaan pisteita {len(far_pts)}, paino {far_weight:.3f}; "
              f"hogline-pisteita {len(hog_pts)}, keskipaino {1.0 / NEAR_TRUST_FLOOR_CM:.3f})")

        # Hyvaksytaan kierroksen tulos VAIN jos se ei huononna pesien
        # muotoa/kokoa merkittavasti (candidate_is_better, sama portti
        # kuin kamera7_06.py:ssa) - pelkka lahemman pesan RMS EI riita,
        # koska se ei nae kaukaista pesaa lainkaan.
        if best is not None:
            accept, _, _ = candidate_is_better(
                frame_undistorted, H_new, rms,
                frame_undistorted, best["H_final"], best["rms"]
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

        relative_improvement = (
            (best_rms - rms) / best_rms if math.isfinite(best_rms) else None
        )

        best = {
            "H_final": H_new,
            "topdown_raw": topdown_raw,
            "rms": rms,
            "quality": quality,
            "iterations": iteration,
        }
        best_rms = rms
        H_current = H_new

        if relative_improvement is not None and relative_improvement < min_relative_improvement:
            break

    if best is None:
        raise RuntimeError("Geometrista homografian korjausta ei saatu laskettua.")

    return best


def draw_verify_overlay(view_clean, verify_result, title, draw_red_inner=False):

    result_img = view_clean.copy()

    rings = [
        (verify_result.get("blue_outer"), (255, 0, 0)),
        (verify_result.get("blue_inner"), (0, 255, 0)),
        (verify_result.get("red_outer"), (0, 0, 255)),
    ]

    if draw_red_inner:
        rings.append((verify_result.get("red_inner"), (0, 165, 255)))

    for ellipse, color in rings:
        if ellipse is not None:
            cv2.ellipse(result_img, ellipse, color, 2, cv2.LINE_AA)

    cv2.putText(
        result_img, title, (15, 30), FONT, 0.7, (0, 255, 255), 2, cv2.LINE_AA
    )

    return result_img


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

    image_height, image_width = frame.shape[:2]

    print(f"Kuvan koko: {image_width} x {image_height}")

    # --------------------------------------------------------
    # LAHEMPI PESA - SININEN + PUNAINEN
    # --------------------------------------------------------

    near_blue_mask = create_blue_mask(frame)
    near_red_mask = create_red_mask(frame)

    blue_outer, blue_inner = find_house_pair(
        near_blue_mask, NEAR_BLUE_MIN_AREA, NEAR_BLUE_MIN_RATIO, NEAR_BLUE_MIN_SIZE_RATIO
    )

    if blue_outer is None or blue_inner is None:
        raise RuntimeError("Lahemman pesan sinista ei loytynyt kokonaan.")

    red_outer, red_inner = find_house_pair(
        near_red_mask, NEAR_RED_MIN_AREA, NEAR_RED_MIN_RATIO, NEAR_RED_MIN_SIZE_RATIO
    )

    if red_outer is None or red_inner is None:
        raise RuntimeError("Lahemman pesan punaista ei loytynyt kokonaan.")

    # --------------------------------------------------------
    # T-LINJA / KESKILINJA (genuinesti Hough-tunnistettu, ei arvattu)
    # --------------------------------------------------------

    segments = detect_line_segments(frame, blue_outer)

    (t_pair, centerline_pair, t_line, centerline, house_center) = select_t_and_centerline(
        segments, blue_outer
    )

    v_t, v_cl = build_image_directions(t_pair, centerline_pair)

    # --------------------------------------------------------
    # KAMERAKALIBROINTI
    # --------------------------------------------------------

    observations = build_calibration_observations(blue_outer, blue_inner, red_outer, red_inner)

    if len(observations) < 2:
        raise RuntimeError("Kalibrointiin tarvitaan vahintaan 2 ympyraa.")

    camera = build_camera_model(observations, image_width, image_height)

    print_camera_calibration(camera)

    # --------------------------------------------------------
    # LAHEMMAN PESAN HSV-TEMPLATE + KAUKAISEN PESAN HAKU
    # (coarse -> fine -> ultra fine)
    # --------------------------------------------------------

    template = build_near_house_template(frame, blue_outer, camera, v_t, v_cl)
    template["_hsv"] = template["hsv_model"]["hsv"]

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
        scale_center=coarse["scale"], scale_range=FINE_SCALE_RANGE, scale_step=FINE_SCALE_STEP,
        description="FINE"
    )

    optimized = search_far_house(
        template, frame, camera, v_t, v_cl,
        center_x=fine["x_offset_cm"], center_y=fine["y_offset_cm"],
        center_range=ULTRA_CENTER_RANGE_CM, center_step=ULTRA_CENTER_STEP_CM,
        angle_center=fine["angle_deg"], angle_range=ULTRA_ANGLE_RANGE_DEG,
        angle_step=ULTRA_ANGLE_STEP_DEG,
        scale_center=fine["scale"], scale_range=ULTRA_SCALE_RANGE, scale_step=ULTRA_SCALE_STEP,
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

    far_center_raw = project_point(x_offset, FAR_HOUSE_Y_CM + y_offset, camera, v_t, v_cl)
    far_left_raw = project_point(x_offset - HOUSE_RADIUS_CM, FAR_HOUSE_Y_CM + y_offset, camera, v_t, v_cl)
    far_right_raw = project_point(x_offset + HOUSE_RADIUS_CM, FAR_HOUSE_Y_CM + y_offset, camera, v_t, v_cl)

    if far_center_raw is None or far_left_raw is None or far_right_raw is None:
        raise RuntimeError("Kaukaisen pesan projisointi epaonnistui.")

    far_left_x, far_left_y = transform_projected_points(
        far_left_raw[0], far_left_raw[1], far_center_raw[0], far_center_raw[1], angle_deg, 1
    )
    far_right_x, far_right_y = transform_projected_points(
        far_right_raw[0], far_right_raw[1], far_center_raw[0], far_center_raw[1], angle_deg, 1
    )

    far_left = np.array([float(far_left_x), float(far_left_y)])
    far_right = np.array([float(far_right_x), float(far_right_y)])
    far_center_est = np.asarray(far_center_raw, dtype=np.float64)

    dir_forward = far_center_est - np.asarray(house_center, dtype=np.float64)
    dir_forward /= np.linalg.norm(dir_forward)

    dir_lateral = far_right - far_left
    dir_lateral /= np.linalg.norm(dir_lateral)

    # --------------------------------------------------------
    # 17 PISTEEN KORRESPONDENSSIT - LAHEMPI PESA (T-linjan/keskilinjan
    # suoralla leikkauksella, genuinesti tunnettu suunta - katso
    # taman tiedoston alkupaan kommentti: nama pisyvat KIINTEINA koko
    # geometrisen jalostuksen ajan).
    # --------------------------------------------------------

    near_img_pts, near_phys_pts, near_labels = near_house_correspondences(
        t_line, centerline, house_center, blue_outer, blue_inner, red_outer, red_inner,
        dir_lateral, dir_forward
    )

    # --------------------------------------------------------
    # KAUKAISEN PESAN ALKUKORRESPONDENSSIT (VAIN alustavan
    # homografian ja k1:n itsekalibroinnin pohjaksi - geometrinen
    # jalostus alempana ei enaa tarvitse naita, koska rengasrajoite
    # ei vaadi T-linjan suuntaa). Kaksi menetelmaa, parempi (pienempi
    # yhdistetty RMS) valitaan.
    # --------------------------------------------------------

    theoretical_img_pts, theoretical_phys_pts, _ = theoretical_far_house_correspondences(
        x_offset, y_offset, angle_deg, scale, camera, v_t, v_cl, far_center_est
    )

    hue_result = create_far_house_hue_masks(frame, x_offset, y_offset, camera, v_t, v_cl, angle_deg, scale)
    far_ellipses = find_far_house_ellipses_from_hue(hue_result)
    hue_img_pts, hue_phys_pts, _, _ = far_house_correspondences_from_ellipses(
        far_ellipses, dir_lateral, dir_forward, far_center_est
    )

    candidates = []
    if len(theoretical_img_pts) >= 3:
        candidates.append(("teoreettinen", theoretical_img_pts, theoretical_phys_pts))
    if len(hue_img_pts) >= 3:
        candidates.append(("hue-maskit", hue_img_pts, hue_phys_pts))

    if not candidates:
        raise RuntimeError("Kaukaiselle pesalle ei saatu yhtaan alkukorrespondenssia.")

    best_far_name, far_img_pts, far_phys_pts = min(
        candidates,
        key=lambda c: _homography_rms(
            np.array(near_img_pts + c[1], dtype=np.float64),
            physical_to_output_px(near_phys_pts + c[2])
        )
    )

    print(f"Kaukaisen pesan alkukorrespondenssit: {best_far_name} "
          f"({len(far_img_pts)} pistetta)")

    all_img_pts = near_img_pts + far_img_pts
    all_phys_pts = near_phys_pts + far_phys_pts

    # --------------------------------------------------------
    # OBJEKTIIVIN SATEITTAISEN VAARISTYMAN ITSEKALIBROINTI (k1) -
    # KERTAALLEEN, ennen homografian geometrista jalostusta (kayttajan
    # oman arvion mukaan objektiivikorjaus ei ole enaa kriittinen kun
    # H lasketaan oikein - katso taman tiedoston alkupaan kommentti).
    # --------------------------------------------------------

    camera_matrix = build_camera_matrix(image_width, image_height)

    best_k1, best_k1_rms, baseline_rms = estimate_radial_distortion_k1(
        all_img_pts, all_phys_pts, camera_matrix
    )

    k1_improvement = (
        (baseline_rms - best_k1_rms) / baseline_rms
        if baseline_rms > 1e-9 and math.isfinite(baseline_rms) else 0.0
    )

    print()
    print("=" * 65)
    print("OBJEKTIIVIN SATEITTAISEN VAARISTYMAN ITSEKALIBROINTI")
    print("=" * 65)
    print(f"RMS ilman korjausta (k1=0) : {baseline_rms:.2f} px")
    print(f"Paras loydetty k1          : {best_k1:+.4f}")
    print(f"RMS korjauksen jalkeen     : {best_k1_rms:.2f} px")
    print(f"Parannus                   : {k1_improvement * 100:.1f} %")

    if k1_improvement > K1_MIN_RELATIVE_IMPROVEMENT and abs(best_k1) > K1_MIN_MAGNITUDE:
        dist_coeffs = np.array([best_k1, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        print("-> Merkittava vaaristyma havaittu, otetaan kayttoon.")
    else:
        best_k1 = 0.0
        dist_coeffs = np.zeros(5, dtype=np.float64)
        print("-> Ei merkittavaa vaaristymaa, jatketaan ilman korjausta.")

    frame_undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs)

    # --------------------------------------------------------
    # ALUSTAVA HOMOGRAFIA (tavallinen DLT, 22 pistetta)
    # --------------------------------------------------------

    near_pts_frame = undistort_points_px(
        np.array(near_img_pts, dtype=np.float64), camera_matrix, best_k1
    )
    far_pts_frame_init = undistort_points_px(
        np.array(far_img_pts, dtype=np.float64), camera_matrix, best_k1
    )

    src_pts = np.vstack([near_pts_frame, far_pts_frame_init]).astype(np.float32)
    dst_pts = physical_to_output_px(all_phys_pts).astype(np.float32)

    H_init, _ = cv2.findHomography(src_pts, dst_pts, method=0)

    if H_init is None:
        raise RuntimeError("Alustavan homografian laskenta epaonnistui.")

    output_w = int(round((OUTPUT_X_MAX_CM - OUTPUT_X_MIN_CM) * PIXELS_PER_CM))
    output_h = int(round((OUTPUT_Y_MAX_CM - OUTPUT_Y_MIN_CM) * PIXELS_PER_CM))

    # --------------------------------------------------------
    # GEOMETRINEN JALOSTUS: pesien renkaat (ympyrarajoite) + hoglinet
    # (viivarajoite) + lahemman pesan 17 pistetta, YHDESSA epalineaari-
    # sessa sovituksessa - katso taman tiedoston alkupaan kommentti.
    # --------------------------------------------------------

    print()
    print("=" * 65)
    print("GEOMETRINEN HOMOGRAFIAN JALOSTUS")
    print("(pesien ympyrat + hoglinet suorina rajoitteina, ei suunta-arvauksia)")
    print("=" * 65)

    near_phys_arr = np.array(near_phys_pts, dtype=np.float64)

    refined = refine_geometric_homography(
        frame_undistorted, H_init, near_pts_frame, near_phys_arr, output_w, output_h
    )

    H_final = refined["H_final"]

    # refined["topdown_raw"] edeltaa H_final:ia yhdella kierroksella (se on
    # warpattu SILLA H:lla joka oli voimassa ENNEN taman kierroksen
    # paivitysta - sama rakenne kuin kamera7_06.py:n refine_homography_
    # corrections:ssa). Lasketaan lopullinen topdown-kuva UUDELLEEN
    # suoraan H_final:lla, jotta tallennetut kuvat ja mitatut arvot
    # vastaavat AIDOSTI lopullista homografiaa.
    topdown_final_raw = cv2.warpPerspective(frame_undistorted, H_final, (output_w, output_h))

    print()
    print(f"Geometrinen jalostus pysahtyi {refined['iterations']} kierroksen "
          f"jalkeen (lahempi RMS {refined['rms']:.4f} cm).")

    # --------------------------------------------------------
    # LOPULLINEN RAPORTOINTI - rehellisesti kaikki neljä
    # kayttajan alkuperaista kriteeria, vaikka ne eivat olisikaan
    # kaikki tayttyneet (katso taman tiedoston alkupaan kommentti:
    # tama on tarkoituksella VAIN homografia+objektiivikorjaus, ei
    # mitaan erillista pakotusmekanismia joka takaisi kriteerit).
    # --------------------------------------------------------

    topdown_final = topdown_final_raw.copy()
    draw_reference_overlay(topdown_final)
    cv2.imwrite(TOPDOWN_OUTPUT, topdown_final)

    near_view_clean = crop_house_view(topdown_final_raw, NEAR_HOUSE_Y_CM, HOUSE_CROP_HALF_HEIGHT_CM)
    far_view_clean = crop_house_view(topdown_final_raw, FAR_HOUSE_Y_CM, HOUSE_CROP_HALF_HEIGHT_CM)

    near_expected_center = expected_house_center_in_crop(
        topdown_final_raw.shape[0], NEAR_HOUSE_Y_CM, HOUSE_CROP_HALF_HEIGHT_CM
    )
    far_expected_center = expected_house_center_in_crop(
        topdown_final_raw.shape[0], FAR_HOUSE_Y_CM, HOUSE_CROP_HALF_HEIGHT_CM
    )

    near_verify = verify_house_from_topdown(
        near_view_clean, near_expected_center, use_ellipse=True, include_red_inner=True
    )
    far_verify = verify_house_from_topdown(
        far_view_clean, far_expected_center, use_ellipse=True, include_red_inner=False
    )

    cv2.imwrite(
        NEAR_HOUSE_VERIFY_OUTPUT,
        draw_verify_overlay(near_view_clean, near_verify, "LAHEMPI (mitattu)", draw_red_inner=True)
    )
    cv2.imwrite(
        FAR_HOUSE_VERIFY_OUTPUT,
        draw_verify_overlay(far_view_clean, far_verify, "KAUKAINEN (mitattu)")
    )

    near_hog_final = detect_hogline_points(topdown_final_raw, NEAR_HOGLINE_Y_CM)
    far_hog_final = detect_hogline_points(topdown_final_raw, FAR_HOGLINE_Y_CM)
    near_hog_angle, _ = robust_line_angle_from_points(near_hog_final)
    far_hog_angle, _ = robust_line_angle_from_points(far_hog_final)

    def center_x_deviation_cm(ellipse):
        if ellipse is None:
            return None
        (cx_px, _), _, _ = ellipse
        x_cm, _ = output_px_to_physical(np.array([[cx_px, 0.0]]))[0]
        return x_cm

    near_dev = center_x_deviation_cm(near_verify.get("blue_outer"))
    far_dev = center_x_deviation_cm(far_verify.get("blue_outer") or far_verify.get("red_outer"))

    quality = refined["quality"]
    max_hog_angle = max(abs(near_hog_angle), abs(far_hog_angle))

    print()
    print("=" * 65)
    print("LOPPUTULOS (kaikki neljä alkuperaista kriteeria, rehellisesti)")
    print("=" * 65)
    print(f"Pyoreys (huonoin, tavoite >= 0.95)   : {quality['worst_ratio']:.4f} "
          f"{'OK' if quality['worst_ratio'] >= 0.95 else 'EI TAYTY'}")
    print(f"Koko-virhe (suurin, tavoite <= 2%)   : {quality['worst_size_err'] * 100:.2f}% "
          f"{'OK' if quality['worst_size_err'] <= 0.02 else 'EI TAYTY'}")
    print(f"Hogline (suurin poikkeama, tavoite <= 1 deg): "
          f"lahempi={near_hog_angle:+.3f} deg, kaukainen={far_hog_angle:+.3f} deg "
          f"{'OK' if max_hog_angle <= 1.0 else 'EI TAYTY'}")
    if near_dev is not None:
        print(f"Lahemman pesan keskipisteen poikkeama keskiviivalta: {near_dev:+.3f} cm")
    if far_dev is not None:
        print(f"Kaukaisen pesan keskipisteen poikkeama keskiviivalta: {far_dev:+.3f} cm")
    print()
    print(f"Kuvat tallennettu: {TOPDOWN_OUTPUT}, {NEAR_HOUSE_VERIFY_OUTPUT}, "
          f"{FAR_HOUSE_VERIFY_OUTPUT}")


if __name__ == "__main__":
    main()
