"""Kiven 3D-profiili: kivikandidaatit, ennustettu aariviiva ja profiilin sovitus."""

import math
import cv2
import numpy as np
from config import (
    HANDLE_NOTCH_R_FRAC,
    HANDLE_NOTCH_R_FRAC_MAX,
    HANDLE_NOTCH_R_FRAC_MIN,
    OUTPUT_X_MAX_CM,
    OUTPUT_X_MIN_CM,
    OUTPUT_Y_MAX_CM,
    OUTPUT_Y_MIN_CM,
    STONE_DARKNESS_SIGMA,
    STONE_HEIGHT_CM,
    STONE_HEIGHT_MAX_CM,
    STONE_MAX_SATURATION,
    STONE_MIN_ASPECT_RATIO,
    STONE_MIN_DARKNESS,
    STONE_MIN_FILL_RATIO,
    STONE_NOMINAL_RADIUS_CM,
    STONE_SHAPE_REG_WEIGHT,
    STONE_SHEET_MARGIN_CM,
)
from geometry import (
    _levenberg_marquardt,
    _project_3d,
    output_px_to_physical,
    ray_plane_intersection,
)


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
