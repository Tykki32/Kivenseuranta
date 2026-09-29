"""Pesan (ja hogline-viivan) tunnistus kuvasta: vari-/reunamaskit, rengaspisteet, ellipsisovitukset, viivasegmentit, hogline sekä pesan laadun mittaus."""

import math
import cv2
import numpy as np
from config import (
    BLUE_INNER_RADIUS_CM,
    BLUE_OUTER_RADIUS_CM,
    FAR_HOUSE_ROI_TARGET_BLUE_FRACTION,
    FAR_HOUSE_ROI_TARGET_RED_FRACTION,
    FAR_HOUSE_ROW_SCAN_CENTER_MARGIN_PX,
    FAR_HOUSE_ROW_SCAN_MAX_GAP_PX,
    FAR_HOUSE_Y_CM,
    HOGLINE_GRAY_HALF_BAND_CM,
    HOGLINE_GRAY_TARGET_COVERAGE,
    HOUSE_CROP_HALF_HEIGHT_CM,
    HOUSE_RADIUS_CM,
    LINE_ICE_S_MAX,
    LINE_ICE_V_MIN,
    NEAR_HOUSE_Y_CM,
    OUTPUT_X_MAX_CM,
    OUTPUT_X_MIN_CM,
    OUTPUT_Y_MAX_CM,
    OUTPUT_Y_MIN_CM,
    PIXELS_PER_CM,
    RED_OUTER_RADIUS_CM,
)
from geometry import (
    _fit_circle_algebraic,
    angle_distance_to_horizontal,
    angle_distance_to_vertical,
    distance,
    frame_points_from_topdown,
    intersect_lines,
    line_angle_deg,
    line_ellipse_intersections,
    line_from_points,
    make_range,
    midpoint,
    physical_to_output_px,
    point_line_distance,
    to_output_px,
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
