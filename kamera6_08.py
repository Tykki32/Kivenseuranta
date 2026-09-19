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

HOMOGRAPHY_RANSAC_THRESHOLD_PX = 3.0

K1_SEARCH_RANGE = 0.6
K1_SEARCH_STEPS = 25
K1_SEARCH_REFINE_ROUNDS = 3
K1_SEARCH_REFINE_SHRINK = 6.0

# Vaaristymaa ei oteta kayttoon, jos se ei parantaisi korrespondenssi-
# pisteiden RMS-uudelleenprojisointivirhetta riittavasti - talla
# valtetaan sovittamasta "vaaristymaa" pelkkaan mittauskohinaan.
K1_MIN_RELATIVE_IMPROVEMENT = 0.08
K1_MIN_MAGNITUDE = 0.01

HOMOGRAPHY_REFINE_MAX_ITERATIONS = 3
HOMOGRAPHY_REFINE_MIN_RELATIVE_IMPROVEMENT = 0.03


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

    points = []

    for theta in angles:

        dx, dy = math.cos(theta), math.sin(theta)

        xs = cx0 + radii * dx
        ys = cy0 + radii * dy

        valid = (xs >= 0) & (xs < w - 1) & (ys >= 0) & (ys < h - 1)

        if valid.sum() < 20:
            continue

        xs_v = xs[valid]
        ys_v = ys[valid]
        radii_v = radii[valid]

        vals = _bilinear_sample(score, xs_v, ys_v)
        vals_smooth = _smooth_reflect(vals, kernel_size=5)

        grad = np.gradient(vals_smooth, radii_v)

        n = len(grad)
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

    for item in lines:

        x1, y1, x2, y2 = item[0]

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

    best_score = -float("inf")
    best_x, best_y = center_x, center_y
    best_angle, best_scale = angle_center, scale_center

    for x_offset in x_values:
        for y_offset in y_values:
            for angle in angle_values:
                for scale in scale_values:

                    score = score_far_candidate(
                        template, x_offset, y_offset, angle, scale,
                        frame, camera, v_t, v_cl
                    )

                    if score > best_score:
                        best_score = score
                        best_x, best_y = x_offset, y_offset
                        best_angle, best_scale = angle, scale

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


def _homography_rms(src_pts, dst_pts):

    H, _ = cv2.findHomography(
        src_pts.astype(np.float32), dst_pts.astype(np.float32),
        method=cv2.RANSAC, ransacReprojThreshold=HOMOGRAPHY_RANSAC_THRESHOLD_PX
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


def detect_hogline_angle(
    topdown_raw, expected_y_cm, half_height_cm=400.0,
    angle_range_deg=20.0, angle_step_deg=0.1,
    edge_margin_frac=0.1, bg_sigma=40.0
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

    best_angle = 0.0
    best_score = -1.0

    for angle_deg in np.arange(-angle_range_deg, angle_range_deg + 1e-9, angle_step_deg):

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
    far_hog_angle, far_hog_score = detect_hogline_angle(
        topdown_raw, FAR_HOGLINE_Y_CM
    )

    print(f"Lahempi hogline  (y={NEAR_HOGLINE_Y_CM:.1f} cm): "
          f"kulma {near_hog_angle:+.2f} deg (luottamus {near_hog_score:.1f})")
    print(f"Kaukainen hogline (y={FAR_HOGLINE_Y_CM:.1f} cm): "
          f"kulma {far_hog_angle:+.2f} deg (luottamus {far_hog_score:.1f})")

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

    H_correction, _ = cv2.findHomography(
        src_pts, dst_pts,
        method=cv2.RANSAC, ransacReprojThreshold=HOMOGRAPHY_RANSAC_THRESHOLD_PX
    )

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
      H_final, topdown_raw, topdown_corrected_raw,
      near_verify, far_verify, rms, iterations
    - topdown_raw on SE raaka top-down-kuva josta near_verify/
      far_verify tunnistettiin (eli edeltaa H_final:ia - kayttokel-
      poinen esim. tarkistusylikuvien piirtamiseen johdonmukaisesti).
    """

    if max_iterations is None:
        max_iterations = HOMOGRAPHY_REFINE_MAX_ITERATIONS
    if min_relative_improvement is None:
        min_relative_improvement = HOMOGRAPHY_REFINE_MIN_RELATIVE_IMPROVEMENT

    H_current = H_initial
    prev_rms = None
    result = None

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
        far_verify = verify_house_from_topdown(
            far_view_clean, far_expected_center,
            use_ellipse=True, include_red_inner=False
        )

        H_final, topdown_corrected_raw, _, _, _, rms = compute_corrected_homography(
            frame_undistorted, H_current, topdown_raw, near_verify, far_verify
        )

        print(f"[Iteratiivinen korjaus {iteration}/{max_iterations}] "
              f"RMS: {rms:.3f} px")

        result = {
            "H_final": H_final,
            "topdown_raw": topdown_raw,
            "topdown_corrected_raw": topdown_corrected_raw,
            "near_verify": near_verify,
            "far_verify": far_verify,
            "rms": rms,
            "iterations": iteration,
        }

        stop = False

        if prev_rms is not None:

            if prev_rms <= 1e-9:
                stop = True
            else:
                relative_improvement = (prev_rms - rms) / prev_rms
                stop = relative_improvement < min_relative_improvement

        prev_rms = rms
        H_current = H_final

        if stop:
            break

    return result


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
    # KAUKAISEN PESAN HUE-MEDIAANIMASKIT (haun loytaman ROI:n
    # sisalta) JA NIIHIN SOVITETUT ELLIPSIT
    # --------------------------------------------------------

    hue_result = create_far_house_hue_masks(
        frame, x_offset, y_offset, camera, v_t, v_cl, angle_deg, scale
    )

    print()
    print("=" * 65)
    print("KAUKAISEN PESAN HUE-MEDIAANIT")
    print("=" * 65)

    if hue_result["blue_hue_median"] is not None:
        print(f"Sininen Hue-mediaani : {hue_result['blue_hue_median']:.2f}")
    else:
        print("Sinisen alueen Hue-mediaania ei voitu laskea.")

    if hue_result["red_hue_median"] is not None:
        print(f"Punainen Hue-mediaani: {hue_result['red_hue_median']:.2f}")
    else:
        print("Punaisen alueen Hue-mediaania ei voitu laskea.")

    print(f"Hue-toleranssi       : +/-{FAR_HUE_TOLERANCE}")

    far_result = find_far_house_ellipses_from_hue(hue_result)

    if all(
        far_result[k] is None
        for k in ("blue_outer", "blue_inner", "red_outer", "red_inner")
    ):
        raise RuntimeError(
            "Kaukaisesta pesasta ei loytynyt yhtaan ellipsia hue-maskeista."
        )

    print()
    print("=" * 65)
    print("KAUKAISEN PESAN ELLIPSIT (HUE-MASKEISTA)")
    print("=" * 65)
    for name in ("blue_outer", "blue_inner", "red_outer", "red_inner"):
        found = far_result[name] is not None
        print(f"  {name:<12}: {'loytyi' if found else 'EI loytynyt'}")

    (
        far_img_pts, far_phys_pts, far_labels, far_center_img
    ) = far_house_correspondences_from_ellipses(
        far_result, dir_lateral, dir_forward, far_center_est
    )

    print(f"Kaukaisen pesan keskipiste: "
          f"({far_center_img[0]:.1f}, {far_center_img[1]:.1f}) px")
    print(f"Kaukaisen pesan pisteita  : {len(far_img_pts)}")

    # --------------------------------------------------------
    # 17 PISTEEN KORRESPONDENSSIT - LAHEMPI PESA
    # --------------------------------------------------------

    near_img_pts, near_phys_pts, near_labels = near_house_correspondences(
        t_line, centerline, house_center,
        blue_outer, blue_inner, red_outer, red_inner,
        dir_lateral, dir_forward
    )

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

    H, _ = cv2.findHomography(
        src_pts, dst_pts,
        method=cv2.RANSAC, ransacReprojThreshold=HOMOGRAPHY_RANSAC_THRESHOLD_PX
    )

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