"""Geometrian perusrakennuspalikat: viivat, ellipsit, homografia, LM-sovitus, kameran projektio ja koordinaattimuunnokset."""

import math
import cv2
import numpy as np
from config import (
    NEAR_HOUSE_Y_CM,
    OUTPUT_X_MIN_CM,
    OUTPUT_Y_MAX_CM,
    PIXELS_PER_CM,
)


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
