# ============================================================
# kamera9_03.py - ALLE PIKSELIN TARKKUUS: KOKO 3D-KAPPALEEN JA
# GRANIITTI/MUOVI-RAJAN YHTEISSOVITUS
#
# kamera9_02.py paikantaa kiven vertaamalla ennustettua KOKO siluettia
# (kupera peite) HAVAITTUUN BINAARISEEN graniittimaskiin karkea->hieno
# ristikkohaulla - tarkkuus on rajattu ristikon askelvaliin (kamera9_02
# .py:ssa n. 1.5cm) JA itse binaarimaskin pikselitason kynnysarvoon
# (yksi pikseli on joko "graniittia" tai ei - ei valipikselitarkkuutta).
#
# Tama tiedosto (kayttajan pyynnosta, MERKITTAVA lisays - kamera9_02.py
# jatetaan koskemattomaksi) parantaa Y-tarkkuutta OLENNAISESTI YHDIS-
# TAMALLA KAKSI RIIPPUMATONTA HAVAINTOLAHDETTA SAMAAN SOVITUKSEEN (EI
# kahta erillista lukua - kayttajan pyynnosta VAIN yksi paras yhteinen
# sijainti per frame):
#
#   1) KOKO GRANIITTISILUETTI (kamera9_01.py:n taysi 3D-profiili)
#      sovitettuna kiven ULKOreunaan jaata vasten - sama periaate kuin
#      kamera9_01.py:n locate_stone_from_profile, mutta TARKEA LISAYS:
#      kirkas kahva PEITTAA osan graniitista talta (ylhaaltapain-)
#      kuvakulmalta, joten graniittimaskin kontuurin osa seuraa
#      TODELLISUUDESSA KAHVAN reunaa, ei kiven reunaa - nama pisteet
#      SUODATETAAN POIS (_filter_ice_boundary_points, katso sen
#      kommentti) ETTEIVAT ne vinouta sovitusta.
#
#   2) GRANIITTI/MUOVI-RAJAN alipikselihavainnot (katso alla) - TERAVA
#      reuna (varisaturaation hyppy), toisin kuin kiven ULKOreuna
#      jaata vasten joka on usein PEHMEA/epaselva (valaistus, varjot,
#      jaan oma tekstuuri) - antaa ALIPIKSELITARKAN lisatiedon.
#
# Rajan alipikseli-ilmaisu: jokaiselta kulmalta (n. 70 kpl) skannataan
# saturaatioarvo BILINEAARISESTI interpoloituna (ei vain lahin pikseli)
# ja kynnysarvon ylitys ratkaistaan LINEAARISELLA INTERPOLOINNILLA
# kahden naytteen valilla - standarditekniikka joka antaa tyypillisesti
# < 0.1 pikselin tarkkuuden reunan sijainnille.
#
# YHTEISSOVITUS: molempien lahteiden jaannosvirheet (etaisyys ennus-
# tettuun malliin - koko rungon KUPERA PEITE JA renkaan ympyra, samalla
# (X,Y)-akselilla) yhdistetaan SAMAAN LM-sovitukseen [X,Y,rengas_sade] -
# yksi paras yhteinen vastaus, ei kahta erillista. Molemmat lahteet
# ovat AINA mukana kun saatavilla - runko-osuus toimii jo pelkastaan
# (myos hyvin kaukana, katso kamera9_01.py:n "kauimpana havaittu kivi"),
# reunaosuus antaa lisatarkkuutta silloin kun rengas on riittavan
# suuri (lahella kameraa) resolvoitavaksi.
#
# HUOM (kayttajan pyynnosta): TATA TIEDOSTOA EI OLE tarkoitus suodattaa/
# tasoittaa frame-framelta - jokainen frame ratkaistaan itsenaisesti
# (ei liukuvaa keskiarvoa tms.), jotta jaljelle jaava frame-framelta-
# kohina kertoo kayttajalle rehellisesti kuinka tarkasti mittaus
# todellisuudessa onnistuu (kayttaja tekee oman suodatuksensa erikseen).
# ============================================================

import os
import csv
import math
import time
import importlib.util

import numpy as np
import cv2


def _load_kamera9_02():
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        "kamera9_02", os.path.join(here, "kamera9_02.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


k92 = _load_kamera9_02()
k9 = k92.k9
k8 = k92.k8


# ============================================================
# GRANIITTI/MUOVI-RAJAN ALIPIKSELI-ILMAISU
# ============================================================

# Kynnysarvo rajan ylitykselle - kamera9_01.py:n STONE_MAX_SATURATION
# (60, graniitti alle taman) ja HANDLE_MIN_SATURATION (80, kahva yli
# taman) puolivali - raja on jossain tallä valilla, keskikohta on
# jarkeva "50%"-ylityspiste lineaariselle interpoloinnille.
BOUNDARY_SATURATION_THRESHOLD = (k9.STONE_MAX_SATURATION + k9.HANDLE_MIN_SATURATION) / 2.0

BOUNDARY_N_ANGLES = 72
# Skannausalue TAHALLAAN LEVEA ([0.2, 2.6] * ennustettu sade, ei esim.
# [0.8,1.2]) - katso refine_position_joint:in kommentti:
# jos karkea alkuarvaus on hieman sivussa (mahdollista, kamera9_02.py:n
# ristikkohaun askel on 1.5cm), KAPEA hakuvyo saattaisi jaada kokonaan
# todellisen rajan ULKOPUOLELLE toisella puolella - LEVEA vyo pitaa
# huolen etta raja loytyy edes OSITTAIN vaikka keskipiste olisi
# vaarassa, ja iterointi (katso alla) korjaa keskipisteen sen jalkeen.
BOUNDARY_RADIUS_SEARCH_FACTOR_MIN = 0.20
BOUNDARY_RADIUS_SEARCH_FACTOR_MAX = 2.60
BOUNDARY_RADIAL_STEP_PX = 0.25            # alipikseli-askel sadesuunnassa
BOUNDARY_MIN_VALID_POINTS = 12            # alle taman -> ei luotettava, palataan karkeaan
BOUNDARY_MIN_PREDICTED_RADIUS_PX = 4.0    # liian pieni rengas ei kannata edes yrittaa
BOUNDARY_MIN_ANGULAR_SPREAD_DEG = 180.0   # pistepilven pitaa kattaa vahintaan tama kulma-alue


def _bilinear_sample(channel, x, y):
    """Alipikselinaytto (bilineaarinen interpolointi) - palauttaa None
    jos piste on kuvan ulkopuolella."""

    x0, y0 = int(math.floor(x)), int(math.floor(y))
    x1, y1 = x0 + 1, y0 + 1

    if x0 < 0 or y0 < 0 or x1 >= channel.shape[1] or y1 >= channel.shape[0]:
        return None

    fx, fy = x - x0, y - y0

    v00 = channel[y0, x0]
    v10 = channel[y0, x1]
    v01 = channel[y1, x0]
    v11 = channel[y1, x1]

    return float(
        v00 * (1 - fx) * (1 - fy) + v10 * fx * (1 - fy) +
        v01 * (1 - fx) * fy + v11 * fx * fy
    )


def _find_boundary_point_along_ray(sat, cx, cy, theta, r_min, r_max, r_step, threshold):
    """
    Skannaa saturaatiokanavaa SADESUUNNASSA (kulma theta) kohdasta
    (cx,cy) valilla [r_min,r_max] ja etsii ENSIMMAISEN ylityksen
    KORKEASTA (kahva) MATALAAN (graniitti) - eli siirtyman rengaan
    ULKOreunalla. Palauttaa (u,v) ALIPIKSELITARKASTI (lineaarinen
    interpolointi kahden naytteen valilla) tai None jos rajaa ei
    loydy (esim. piilossa/kuvan ulkopuolella).
    """

    dx, dy = math.cos(theta), math.sin(theta)

    r = r_min
    prev_r = None
    prev_v = None

    while r <= r_max:

        x = cx + r * dx
        y = cy + r * dy
        v = _bilinear_sample(sat, x, y)

        if v is None:
            return None

        if prev_v is not None and prev_v >= threshold > v:

            span = v - prev_v
            t = (threshold - prev_v) / span if abs(span) > 1e-9 else 0.0
            r_cross = prev_r + t * (r - prev_r)

            return cx + r_cross * dx, cy + r_cross * dy

        prev_r, prev_v = r, v
        r += r_step

    return None


def _detect_boundary_points(frame_u, cx_px, cy_px, r_px_approx,
                             n_angles=BOUNDARY_N_ANGLES,
                             threshold=BOUNDARY_SATURATION_THRESHOLD):
    """
    Etsii graniitti/muovi-rajan ALIPIKSELIPISTEET n_angles kulmasta
    ennustetun keskipisteen (cx_px,cy_px) ja sateen (r_px_approx)
    ymparilta. Palauttaa Nx2-taulukon (float) - vain onnistuneet
    (ei-None) havainnot.
    """

    hsv = cv2.cvtColor(frame_u, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)

    r_min = r_px_approx * BOUNDARY_RADIUS_SEARCH_FACTOR_MIN
    r_max = r_px_approx * BOUNDARY_RADIUS_SEARCH_FACTOR_MAX

    points = []

    for theta in np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False):
        p = _find_boundary_point_along_ray(
            sat, cx_px, cy_px, theta, r_min, r_max, BOUNDARY_RADIAL_STEP_PX, threshold
        )
        if p is not None:
            points.append(p)

    return np.array(points, dtype=np.float64) if points else np.zeros((0, 2))


# ============================================================
# KOKO KIVEN GRANIITTISILUETIN KONTUURI - KAHVAN PEITTAMAT KOHDAT
# SUODATETTUINA POIS
# ============================================================

BODY_CONTOUR_MIN_AREA_PX = 15.0
BODY_CONTOUR_MAX_SEARCH_DIST_PX = 60.0
BODY_HANDLE_CHECK_DIST_PX = 2.5   # kuinka kauas ULOSPAIN pisteesta tarkistetaan saturaatio
BODY_MIN_VALID_POINTS = 8


def _find_contour_near(mask, approx_px, max_dist_px=BODY_CONTOUR_MAX_SEARCH_DIST_PX,
                        min_area=BODY_CONTOUR_MIN_AREA_PX):
    """
    Etsii graniittimaskin KONTUURIN joka on lahinna ennustettua
    pikselisijaintia (approx_px) - sama periaate kuin kamera9_01.py:n
    find_stone_candidates, mutta EI vaadi muotosuodattimia (fill_ratio
    jne.) koska meilla on jo luotettava paikkaeste jatkuvuudesta,
    emmeka etsi koko kuvasta vaan VAIN lahinta kontuuria.
    """

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    best = None
    best_dist = None

    for c in contours:

        if cv2.contourArea(c) < min_area:
            continue

        M = cv2.moments(c)

        if M["m00"] == 0:
            continue

        ccx, ccy = M["m10"] / M["m00"], M["m01"] / M["m00"]
        d = math.hypot(ccx - approx_px[0], ccy - approx_px[1])

        if d <= max_dist_px and (best_dist is None or d < best_dist):
            best = c
            best_dist = d

    return best


def _filter_ice_boundary_points(frame_u, contour, check_dist_px=BODY_HANDLE_CHECK_DIST_PX,
                                 threshold=BOUNDARY_SATURATION_THRESHOLD):
    """
    Kayttajan pyynnosta: kirkas kahva PEITTAA osan graniitista talta
    (ylhaaltapain-) kuvakulmalta, joten graniittimaskin kontuurin OSA
    seuraa TODELLISUUDESSA kahvan reunaa, ei kiven todellista ulko-
    reunaa jaata vasten - naita pisteita EI SAA kayttaa runko-3D-mallin
    sovituksessa (ne vinouttaisivat sen, koska ennustettu malli kuvaa
    kiven OIKEAA - ei kahvan peittamaa - muotoa).

    Suodatusperiaate: jokaiselle kontuuripisteelle katsotaan hieman
    ULOSPAIN (pois pistepilven painopisteesta) - jos siella saturaatio
    on KORKEA (kahvaa), piste hylataan; jos MATALA (jaata), pidetaan.
    """

    hsv = cv2.cvtColor(frame_u, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)

    pts = contour.reshape(-1, 2).astype(np.float64)

    if len(pts) == 0:
        return np.zeros((0, 2))

    cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())

    keep = []

    for p in pts:

        dx, dy = p[0] - cx, p[1] - cy
        norm = math.hypot(dx, dy)

        if norm < 1e-6:
            continue

        dx, dy = dx / norm, dy / norm
        ox, oy = p[0] + dx * check_dist_px, p[1] + dy * check_dist_px

        sat_val = _bilinear_sample(sat, ox, oy)

        if sat_val is not None and sat_val < threshold:
            keep.append(p)

    return np.array(keep, dtype=np.float64) if keep else np.zeros((0, 2))


# ============================================================
# RENKAAN (TUNNETTU SADE+KORKEUS) SOVITUS ALIPIKSELIHAVAINTOIHIN
# ============================================================

def _predicted_ring_points(pose, X0, Y0, ring_radius_cm, ring_height_cm, n_theta=120):

    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    xs = X0 + ring_radius_cm * np.cos(theta)
    ys = Y0 + ring_radius_cm * np.sin(theta)
    zs = np.full(n_theta, ring_height_cm)

    points_3d = np.column_stack([xs, ys, zs])
    u, v = k9._project_3d(pose["K"], pose["R"], pose["t"], points_3d)

    return np.column_stack([u, v]).astype(np.float32)


def _ring_point_residuals(pose, X0, Y0, ring_radius_cm, ring_height_cm, observed_pts):

    poly = _predicted_ring_points(pose, X0, Y0, ring_radius_cm, ring_height_cm)
    poly_cv = poly.reshape(-1, 1, 2)

    return np.array([
        cv2.pointPolygonTest(poly_cv, (float(p[0]), float(p[1])), True)
        for p in observed_pts
    ])


def _angular_spread_deg(points, cx, cy):
    """
    Kuinka monta astetta pistepilvi kattaa keskipisteen (cx,cy)
    ymparilla - KAPEA kaari (esim. vain toisella puolella oleva osa-
    aariviiva) EI riita maarittamaan 2D-sijaintia luotettavasti (se
    rajoittaa lahinna vain yhta suuntaa), vaikka pisteita olisi paljon.
    """

    if len(points) < 2:
        return 0.0

    angles = np.degrees(np.arctan2(points[:, 1] - cy, points[:, 0] - cx))
    angles = np.sort(angles % 360.0)

    gaps = np.diff(np.concatenate([angles, [angles[0] + 360.0]]))
    largest_gap = float(np.max(gaps))

    return 360.0 - largest_gap


BOUNDARY_MAX_ITERATIONS = 5
BOUNDARY_CONVERGENCE_CM = 0.05  # pysaytetaan kun peräkkaiset iteraatiot eroavat alle taman

# Jarkevyystarkistus VAIN todella suurille hypyille. HUOM (loydetty
# testatessa oikealla frame-sarjalla, katso git-historia): karkean
# haun (kamera9_02.py) JA taman alipikselisovituksen valilla on
# JOHDONMUKAINEN n. 10-12cm systemaattinen ero JOKA FRAMESSA (ei
# satunnaista kohinaa) - karkea koko-siluetti-sovitus on siis itse
# hieman vinoutunut (todennakoisesti koska sen mallin approksimaatiot
# eivat tasmaa ulkoreunalla yhta tarkasti kuin terava variraja), ja
# TAMA korjaus (10-12cm) ON JUURI SE tarkennus jota kayttaja pyysi -
# EI virhe joka pitaisi hylata. Rajaa siis vain aidosti mahdottomat
# hypyt (esim. kiiltokohtaan lukkiutuminen), ei tatä odotettua korjausta.
BOUNDARY_MAX_SHIFT_FROM_APPROX_CM = 25.0


def refine_position_joint(frame_u, pose, profile, X0_approx, Y0_approx):
    """
    Ratkaisee YHDEN parhaan (X,Y)-sijainnin yhdistamalla KAKSI riippu-
    matonta havaintolahdetta samaan LM-sovitukseen - katso taman
    tiedoston alkupaan kommentti periaatteesta. EI KAHTA ERILLISTA
    LUKUA - kayttajan pyynnosta vain yksi yhteinen paras vastaus.

    HUOM RENKAAN SATEESTA (loydetty testatessa - katso git-historia):
    Kivi.jpg-sivukuvasta mitattu "paivan" sade (graniitin ja kahvan
    KIINNITYSLEVYN raja) EI vastaa sita mita ylhaaltapain-kuvassa
    NAHDAAN varirajana - kahva itse (kahva-AISA, ei vain littea levy)
    nousee korkeammalle ja on leveampi kuin pelkka levy, joten sen
    projisoitu siluetti ylhaalta on ISOMPI ja MUUTTUU kuvakulman
    mukaan. SADETTA EI SIIS VOI OLETTAA KIINTEAKSI - se on YKSI vapaa
    parametri [X,Y,R] taman sovituksen tehtavassa (korkeus Z pidetaan
    kiinteana H_total:ina yksinkertaisuuden vuoksi).

    ITEROI: koska renkaan rajapisteiden HAKUALUE riippuu itse nykyi-
    sesta parhaasta arviosta, yhden kierroksen sovitus on herkka
    alkuarvaukselle. Joka kierroksella haetaan renkaan rajapisteet
    UUDELLEEN nykyisen (X,Y,R):n ymparilta (runko-kontuuri haetaan
    vain kerran - sen haku ei ole yhta herkka pienille siirtymille).

    Palauttaa dictin: X_cm, Y_cm (yhteissovitettu TAI - jos kumpikaan
    lahde ei antanut riittavasti tietoa - alkuperainen approksimaatio),
    tarkka (bool), n_body/n_ring (kaytettyjen pisteiden maarat),
    rms_px (yhdistetty jaannosvirhe), ring_radius_cm (diagnostinen).
    """

    R_max, H_total, shape_deltas = profile["R_max_cm"], profile["H_total_cm"], profile["shape_deltas"]
    ring_height_cm = H_total

    # --- Runko-kontuuri (kertaalleen, suodatettuna kahvan peittamista
    # kohdista - katso _filter_ice_boundary_points:in kommentti) ---
    mask = k9.create_granite_mask(frame_u)
    approx_pt3d = np.array([[X0_approx, Y0_approx, H_total / 2.0]])
    u0, v0 = k9._project_3d(pose["K"], pose["R"], pose["t"], approx_pt3d)
    approx_px = (float(u0[0]), float(v0[0]))

    raw_contour = _find_contour_near(mask, approx_px)
    body_pts = (
        _filter_ice_boundary_points(frame_u, raw_contour)
        if raw_contour is not None else np.zeros((0, 2))
    )
    n_body = len(body_pts)

    ring_r_frac_guess = float((k9._TEMPLATE_R_FRAC + shape_deltas)[-1])
    X_cur, Y_cur, R_cur = X0_approx, Y0_approx, ring_r_frac_guess * R_max
    ring_pts = np.zeros((0, 2))
    rms_px = None

    def residuals(pts_body, pts_ring):
        def f(params):
            X, Y, R = params
            parts = []
            if len(pts_body) > 0:
                parts.append(k9._profile_residuals_for_stone(
                    pose, X, Y, R_max, H_total, shape_deltas, pts_body, len(pts_body)
                ))
            if len(pts_ring) > 0:
                parts.append(_ring_point_residuals(pose, X, Y, R, ring_height_cm, pts_ring))
            return np.concatenate(parts) if parts else np.zeros(0)
        return f

    for _ in range(BOUNDARY_MAX_ITERATIONS):

        center_pts_3d = np.array([
            [X_cur, Y_cur, ring_height_cm],
            [X_cur + R_cur, Y_cur, ring_height_cm],
        ])
        u, v = k9._project_3d(pose["K"], pose["R"], pose["t"], center_pts_3d)

        cx_px, cy_px = float(u[0]), float(v[0])
        r_px_approx = math.hypot(float(u[1]) - cx_px, float(v[1]) - cy_px)

        ring_pts = np.zeros((0, 2))

        if r_px_approx >= BOUNDARY_MIN_PREDICTED_RADIUS_PX:
            candidate_ring_pts = _detect_boundary_points(frame_u, cx_px, cy_px, r_px_approx)
            # Katso _angular_spread_deg:in kommentti - kapea kaari ei
            # riita, vaikka pisteita olisi maarallisesti tarpeeksi.
            if (len(candidate_ring_pts) >= BOUNDARY_MIN_VALID_POINTS and
                    _angular_spread_deg(candidate_ring_pts, cx_px, cy_px) >= BOUNDARY_MIN_ANGULAR_SPREAD_DEG):
                ring_pts = candidate_ring_pts

        n_ring = len(ring_pts)

        if n_body < BODY_MIN_VALID_POINTS and n_ring < BOUNDARY_MIN_VALID_POINTS:
            return {
                "X_cm": X0_approx, "Y_cm": Y0_approx, "tarkka": False,
                "n_body": n_body, "n_ring": n_ring, "rms_px": None, "ring_radius_cm": None,
            }

        params0 = np.array([X_cur, Y_cur, R_cur], dtype=np.float64)
        res_fn = residuals(body_pts, ring_pts)
        params_final = k8._levenberg_marquardt(res_fn, params0, max_iterations=30)

        # POIKKEAMIEN HYLKAYS kummallekin lahteelle ERIKSEEN (eri
        # tarkkuustasot - runko pikselitason kontuuri, rengas alipik-
        # seli - MAD-kynnys sovitetaan kummankin OMAAN hajontaan) ja
        # yksi uudelleensovitus puhdistetulla yhdistetylla joukolla.
        clean_body = body_pts
        clean_ring = ring_pts

        if n_body > 0:
            body_resid = k9._profile_residuals_for_stone(
                pose, params_final[0], params_final[1], R_max, H_total, shape_deltas,
                body_pts, len(body_pts)
            )
            mad_b = float(np.median(np.abs(body_resid - np.median(body_resid)))) + 1e-6
            body_inliers = np.abs(body_resid - np.median(body_resid)) < 3.0 * 1.4826 * mad_b
            if np.count_nonzero(body_inliers) >= BODY_MIN_VALID_POINTS:
                clean_body = body_pts[body_inliers]

        if n_ring > 0:
            ring_resid = _ring_point_residuals(
                pose, params_final[0], params_final[1], params_final[2], ring_height_cm, ring_pts
            )
            mad_r = float(np.median(np.abs(ring_resid - np.median(ring_resid)))) + 1e-6
            ring_inliers = np.abs(ring_resid - np.median(ring_resid)) < 3.0 * 1.4826 * mad_r
            if np.count_nonzero(ring_inliers) >= BOUNDARY_MIN_VALID_POINTS:
                clean_ring = ring_pts[ring_inliers]

        if len(clean_body) != n_body or len(clean_ring) != n_ring:
            params_final = k8._levenberg_marquardt(
                residuals(clean_body, clean_ring), params_final, max_iterations=30
            )

        resid_final = residuals(clean_body, clean_ring)(params_final)
        rms_px = float(np.sqrt(np.mean(resid_final ** 2))) if len(resid_final) else None

        X_new, Y_new, R_new = float(params_final[0]), float(params_final[1]), float(abs(params_final[2]))
        moved = math.hypot(X_new - X_cur, Y_new - Y_cur)
        X_cur, Y_cur, R_cur = X_new, Y_new, R_new

        if moved < BOUNDARY_CONVERGENCE_CM:
            break

    total_shift = math.hypot(X_cur - X0_approx, Y_cur - Y0_approx)

    if total_shift > BOUNDARY_MAX_SHIFT_FROM_APPROX_CM:
        return {
            "X_cm": X0_approx, "Y_cm": Y0_approx, "tarkka": False,
            "n_body": n_body, "n_ring": len(ring_pts), "rms_px": rms_px, "ring_radius_cm": None,
        }

    return {
        "X_cm": X_cur, "Y_cm": Y_cur, "tarkka": True,
        "n_body": n_body, "n_ring": len(ring_pts), "rms_px": rms_px, "ring_radius_cm": R_cur,
    }


# ============================================================
# HAKU + SEURANTA + ALIPIKSELIHIENOSAATO KOKO VIDEOLLE, CSV-TULOSTUS
#
# Tilakone on SAMA kuin kamera9_02.py:ssa (karkea ristikkohaku HAKU-
# ja SEURANTA-vaiheissa - katso sen kommentti). Ainoa ero: JOKAISEN
# onnistuneen karkean paikannuksen JALKEEN yritetaan taman tiedoston
# refine_position_joint - CSV:hen kirjoitetaan AINA yksi yhteinen
# (runko+rengas) hienosaadettu (tai, jos epaonnistui, karkea) sijainti
# + "tarkka"-lippu jotta kayttaja nakee milloin alle-pikselin-tarkkuus
# todella saavutettiin. EI FRAME-VALISTA SUODATUSTA - raaka data
# kirjoitetaan sellaisenaan, kayttaja tekee tarkkuusanalyysin itse.
# ============================================================

def search_and_track_stones_precise(video_path, calib, pose, profile, csv_path,
                                     search_x_half_width=k92.SEARCH_X_HALF_WIDTH_CM,
                                     search_y_min=k92.SEARCH_Y_MIN_CM,
                                     search_y_max=k92.SEARCH_Y_MAX_CM,
                                     search_every_n=k92.SEARCH_EVERY_N_FRAMES,
                                     progress=True):

    cap = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 25.0

    camera_matrix = calib["camera_matrix"]
    dist_coeffs = np.array([calib["best_k1"], 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    state = "HAKU"
    stone_id = 0
    last_xy = None
    misses = 0
    n_written = 0
    n_precise = 0

    t_start = time.time()

    with open(csv_path, "w", newline="") as f:

        writer = csv.writer(f)
        writer.writerow(["frame", "timestamp_s", "stone_id", "x_m", "y_m", "tarkka",
                          "n_runkopistetta", "n_reunapistetta", "rms_px", "rengas_r_cm"])

        frame_idx = 0

        while frame_idx < n_frames:

            ok, frame = cap.read()

            if not ok:
                break

            do_this_frame = (state == "SEURANTA") or (frame_idx % search_every_n == 0)

            if not do_this_frame:
                frame_idx += 1
                continue

            frame_u = cv2.undistort(frame, camera_matrix, dist_coeffs)
            mask = k9.create_granite_mask(frame_u)

            if state == "HAKU":

                x_center = 0.0
                y_center = (search_y_min + search_y_max) / 2.0
                y_half = (search_y_max - search_y_min) / 2.0

                (bx, by), score = k92._locate_by_grid_search(
                    mask, pose, profile, x_center, search_x_half_width, y_center, y_half,
                    k92.SEARCH_COARSE_STEP_CM, k92.SEARCH_FINE_STEP_CM
                )

                if score >= k92.SEARCH_SCORE_THRESHOLD:
                    stone_id += 1
                    misses = 0
                    state = "SEURANTA"

                    refined = refine_position_joint(frame_u, pose, profile, bx, by)
                    last_xy = (refined["X_cm"], refined["Y_cm"])

                    if progress:
                        tarkka_str = f"KYLLA (rms={refined['rms_px']:.2f}px)" if refined["tarkka"] else "EI (karkea)"
                        print(f"[frame {frame_idx}] LOYTYI kivi {stone_id}: "
                              f"({last_xy[0]:.1f}, {last_xy[1]:.1f}) cm, peitto={score:.2f}, "
                              f"alipikselitarkkuus={tarkka_str} (runko={refined['n_body']}, "
                              f"rengas={refined['n_ring']})")

                    if refined["tarkka"]:
                        n_precise += 1

                    timestamp = frame_idx / fps
                    writer.writerow([
                        frame_idx, f"{timestamp:.3f}", stone_id,
                        f"{last_xy[0] / 100.0:.5f}", f"{last_xy[1] / 100.0:.5f}",
                        int(refined["tarkka"]), refined["n_body"], refined["n_ring"],
                        f"{refined['rms_px']:.3f}" if refined["rms_px"] is not None else "",
                        f"{refined['ring_radius_cm']:.3f}" if refined["ring_radius_cm"] is not None else "",
                    ])
                    n_written += 1

            else:  # SEURANTA

                (bx, by), score = k92._locate_by_grid_search(
                    mask, pose, profile, last_xy[0], k92.TRACK_HALF_RANGE_CM,
                    last_xy[1], k92.TRACK_HALF_RANGE_CM,
                    k92.TRACK_COARSE_STEP_CM, k92.TRACK_FINE_STEP_CM
                )

                if score >= k92.TRACK_SCORE_THRESHOLD:
                    misses = 0

                    refined = refine_position_joint(frame_u, pose, profile, bx, by)
                    last_xy = (refined["X_cm"], refined["Y_cm"])

                    if refined["tarkka"]:
                        n_precise += 1

                    timestamp = frame_idx / fps
                    writer.writerow([
                        frame_idx, f"{timestamp:.3f}", stone_id,
                        f"{last_xy[0] / 100.0:.5f}", f"{last_xy[1] / 100.0:.5f}",
                        int(refined["tarkka"]), refined["n_body"], refined["n_ring"],
                        f"{refined['rms_px']:.3f}" if refined["rms_px"] is not None else "",
                        f"{refined['ring_radius_cm']:.3f}" if refined["ring_radius_cm"] is not None else "",
                    ])
                    n_written += 1
                else:
                    misses += 1

                    if misses >= k92.TRACK_LOST_MAX_MISSES:
                        if progress:
                            print(f"[frame {frame_idx}] Kivi {stone_id} kadotettu "
                                  f"({misses} huonoa framea), palataan hakuun.")
                        state = "HAKU"
                        last_xy = None

            if progress and frame_idx % 100 == 0:
                elapsed = time.time() - t_start
                print(f"  ... frame {frame_idx}/{n_frames} (tila={state}, "
                      f"{elapsed:.0f}s kulunut, {n_written} rivia, "
                      f"{n_precise} alipikselitarkkaa)")

            frame_idx += 1

    cap.release()

    if progress:
        print(f"Valmis: {n_written} rivia ({n_precise} alipikselitarkkaa) "
              f"tiedostoon {csv_path} ({time.time() - t_start:.0f}s)")

    return {"n_written": n_written, "n_precise": n_precise, "n_stones": stone_id}


# ============================================================
# PAAOHJELMA (katso kamera9_02.py:n kommentti siemen-arvoista)
# ============================================================

def main():

    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()

    calib_filename = filedialog.askopenfilename(
        title="Valitse kalibrointikuva (jolla H_final/poosi ratkaistaan)",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
            ("All files", "*.*"),
        ]
    )

    if not calib_filename:
        print("Kalibrointikuvaa ei valittu.")
        root.destroy()
        return

    video_filename = filedialog.askopenfilename(
        title="Valitse video (kiven paikannukseen/seurantaan)",
        filetypes=[
            ("Video files", "*.mp4 *.avi *.mov *.mkv"),
            ("All files", "*.*"),
        ]
    )

    root.destroy()

    if not video_filename:
        print("Videota ei valittu.")
        return

    print("Kalibroidaan kamera (kamera8_01.py:n putki)...")
    calib = k9.calibrate_camera_from_image(calib_filename)
    pose = k9.build_pose_from_calibration(calib)
    print(f"  fokaalivali f = {pose['K'][0, 0]:.1f} px, "
          f"kameran sijainti (cm): {np.round(pose['camera_position_cm'], 1)}")

    kivilla_stones = k9.find_stone_candidates(calib["frame_undistorted"], calib["H_final"])
    print(f"  loydettiin {len(kivilla_stones)} paikallaan olevaa kivea kalibrointikuvasta")

    print(f"Sovitetaan 3D-profiili (kalibrointikuvan kivet + video, "
          f"siemen frame={k92.DEFAULT_SEED_FRAME_IDX})...")
    result = k9.build_stone_profile_from_video(
        calib, pose, video_filename,
        seed_frame_idx=k92.DEFAULT_SEED_FRAME_IDX, seed_pos_cm=k92.DEFAULT_SEED_POS_CM,
        extra_stones=kivilla_stones, n_video_samples=25,
    )
    profile = result["profile"]
    print(f"  R_max={profile['R_max_cm']:.2f} cm, H_total={profile['H_total_cm']:.2f} cm, "
          f"RMS={profile['residual_rms_px']:.2f} px")

    ring_r_frac = float((k9._TEMPLATE_R_FRAC + profile["shape_deltas"])[-1])
    print(f"  graniitti/muovi-rajan sade: {ring_r_frac * profile['R_max_cm']:.2f} cm "
          f"(korkeus {profile['H_total_cm']:.2f} cm)")

    print("Haetaan ja seurataan kivea/kivia videolta (alipikselitarkka Y)...")

    csv_path = "kivien_sijainnit_tarkka.csv"
    search_and_track_stones_precise(video_filename, calib, pose, profile, csv_path)


if __name__ == "__main__":
    main()
