# ============================================================
# kamera9_01.py - KIVEN 3D-SIJAINTI (Z-KORJATTU) JO KALIBROIDULLA
# KAMERAMALLILLA
#
# kamera8_01.py laskee vain fyysisen JAATASON (Z=0) YLHAALTA-
# KUVAUKSEN - se on TAHALLAAN vain YKSI homografia (H_final) + yksi
# objektiivikorjaus (k1), eika tallainen kuvaus voi koskaan olla
# oikein pisteille jotka EIVAT ole jaatasolla. Kivi on lieriomainen
# esine, jonka korkeus (WCF: vahintaan 11.43 cm) nostaa sen suurimman
# osan jaatasosta - homografian kautta laskettu "sijainti" olisi
# systemaattisesti siirtynyt kameran nadiirista poispain (klassinen
# fotogrammetrinen parallaksi-/kallistumisilmio).
#
# Tama tiedosto (kayttajan pyynnosta, MERKITTAVA lisays, siksi oma
# tiedostonsa - kamera8_01.py jatetaan koskemattomaksi) PURKAA
# kamera8_01.py:n jo validoidun taso-homografian (H_final) TAYDEKSI
# 3D-kameramalliksi (fokaalivali + rotaatio + translaatio suhteessa
# jaatasoon), ja kayttaa sita SADETASOLEIKKAUKSEEN mielivaltaisella
# korkeudella - eli Z-korjaukseen. Lisaksi: koska radalla nakyvat
# kivet (graniittiosa, ei varikasta kahvaa) ovat KAIKKI SAMANKOKOISIA,
# niiden havaittu koko usealla eri etaisyydella antaa YLIMAARATYN
# yhtaloryhman kiven todelliselle sateelle - validoi/tarkentaa samalla
# koko kalibroinnin.
#
# TARKEA RAJOITUS: kamera8_01.py:n H_final/k1 EIVAT riita sellaisenaan
# - taydelliseen 3D-malliin tarvitaan MYOS fokaalivali (f), jota
# kamera8_01.py:n "karkea kameramatriisi" vain ARVASI (f=max(leveys,
# korkeus)) koska se riitti tason homografialle. Tassa f haetaan
# UUDELLEEN, tarkemmin, koska se vaikuttaa suoraan siihen miten
# korkeus (Z) tulkitaan.
# ============================================================

import os
import math
import importlib.util

import numpy as np
import cv2


def _load_kamera8_01():
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        "kamera8_01", os.path.join(here, "kamera8_01.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


k8 = _load_kamera8_01()


# ============================================================
# KIVEN NIMELLISMITAT (WCF) - kaytetaan VAIN fit_stone_profile:in
# alkuarvauksena (lahtokohta hienosaadolle) ja jarkevyystarkistukseen.
# Seka sade etta korkeus RATKAISTAAN oikeasti havainnoista - katso
# fit_stone_profile ja sen ylla oleva kommentti kovakoodatusta
# karkeasta 3D-mallista.
# ============================================================

STONE_NOMINAL_RADIUS_CM = 91.44 / math.pi / 2.0
STONE_HEIGHT_CM = 11.43


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
        [k8.PIXELS_PER_CM, 0.0, -k8.OUTPUT_X_MIN_CM * k8.PIXELS_PER_CM],
        [0.0, -k8.PIXELS_PER_CM, k8.OUTPUT_Y_MAX_CM * k8.PIXELS_PER_CM],
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
            dist = np.hypot(X, Y - k8.FAR_HOUSE_Y_CM)
            parts.append((dist - far_radius) * far_weight)

        if len(hog_pts):
            X, Y = ray_plane_intersection(K, R, t, hog_pts[:, 0], hog_pts[:, 1], 0.0)
            target = np.where(hog_line_idx == 0, k8.NEAR_HOGLINE_Y_CM, k8.FAR_HOGLINE_Y_CM)
            parts.append((Y - target) * hog_weight)

        return np.concatenate(parts)

    params0 = np.concatenate([rvec1.flatten(), tvec1.flatten()])
    params_final = k8._levenberg_marquardt(residuals, params0, max_iterations=100)

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

    far_pts, far_radius = k8.far_house_ring_points_in_frame(topdown_raw, H_final)
    near_hog, near_hog_w = k8.hogline_points_in_frame(topdown_raw, k8.NEAR_HOGLINE_Y_CM, H_final)
    far_hog, far_hog_w = k8.hogline_points_in_frame(topdown_raw, k8.FAR_HOGLINE_Y_CM, H_final)

    hog_pts = np.concatenate([near_hog, far_hog]) if (len(near_hog) or len(far_hog)) else np.zeros((0, 2))
    hog_line_idx = np.concatenate([
        np.zeros(len(near_hog), dtype=np.int64), np.ones(len(far_hog), dtype=np.int64)
    ])
    hog_strength = np.concatenate([near_hog_w, far_hog_w])

    far_resid_now = k8._far_ring_distance(H_final, far_pts, far_radius)
    far_weight = 1.0 / k8._robust_scale_cm(far_resid_now, k8.NEAR_TRUST_FLOOR_CM)

    # Hoglinen kiintea, taysi paino - katso kamera8_01.py:n kommentti
    # (l' = H^-T l pitaa maarata kiertoa, ei saa vaientua).
    hog_weight = hog_strength / k8.NEAR_TRUST_FLOOR_CM

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

    x_min = k8.OUTPUT_X_MIN_CM - sheet_margin_cm
    x_max = k8.OUTPUT_X_MAX_CM + sheet_margin_cm
    y_min = k8.OUTPUT_Y_MIN_CM - sheet_margin_cm
    y_max = k8.OUTPUT_Y_MAX_CM + sheet_margin_cm

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
        phys = k8.output_px_to_physical(output_px)[0]

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
# kohdan (max leveys) suhteen. Tama KORVAA aiemman kasin-arvatun
# mallin (joka oletti leveimman kohdan lahella pohjaa - VAARIN, katso
# git-historia) - todellisuudessa leveimmillaan LAHELLA KESKIKORKEUTTA
# (n. 45-60% korkeudesta), ja profiili on suht. SYMMETRINEN ylhaalta
# alas (poikkeuksena aivan ylin reuna, jossa kahvan kiinnityslevy
# peittaa/rajaa nakyvan graniitin jyrkasti - sama efekti nakyy myos
# ylhaaltapain-kuvissa, koska sielläkin kahva rajataan pois
# saturaatiolla, joten tama on OIKEA vertailukohta). Pohjan (z=0)
# tarkka arvo on arvio (poydan/varjon reunalla vaikea mitata tarkasti
# kuvasta - katso git-historia), muu on suoraan mitattua.
STONE_PROFILE_TEMPLATE_NORM = [
    (0.00, 0.75),   # pohja (arvioitu - kovera alusta, ei tarkkaan mitattavissa kuvasta)
    (0.18, 0.94),   # levenee nopeasti
    (0.30, 0.98),
    (0.45, 1.00),   # "paiva" - leveimmillaan
    (0.60, 1.00),   # pysyy leveimmillaan (loiva huippu, ei terava)
    (0.75, 0.97),
    (0.88, 0.90),
    (0.96, 0.81),
    (1.00, 0.37),   # kahvan kiinnityslevyn reuna - jyrkka rajaus
]

_TEMPLATE_Z_FRAC = np.array([p[0] for p in STONE_PROFILE_TEMPLATE_NORM])
_TEMPLATE_R_FRAC = np.array([p[1] for p in STONE_PROFILE_TEMPLATE_NORM])
_TEMPLATE_EQUATOR_IDX = int(np.argmax(_TEMPLATE_R_FRAC))

# Kuinka voimakkaasti muotoa (kontrollipisteiden sateet) rangaistaan
# poikkeamasta kovakoodattuun mallinnukseen nahden (yksikko: "pikselia
# per r_frac-yksikko" - katso fit_stone_profile). Pitaa hienosaadon
# LAHELLA fysikaalisesti jarkevaa lahtokohtaa, estaa ylisovituksen
# kohinaisen aariviivan yli.
#
# HUOM (paivitetty - katso git-historia): kun profiilin KAIKKI
# kontrollipisteet vapautettiin (ei enaa vain "paivan ylapuoliset"),
# vapaita muotoparametreja on 5 - reilusti enemman kuin ennen (2-3).
# Alkuperainen 25 (viritetty vanhalle, suppeammalle mallille) antoi
# 3 kiven aineistolla epatasaisia, lievasti epafyysisia tuloksia
# (pieni "olkapaa"-kohouma profiilissa) - nostettu 60:een, joka pitaa
# muodon sileampana/uskottavampana samalla kun sallii aidon korjauksen.
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


def _sample_contour_points(contour, n_sample):
    """Tasavalisesti alinaytetty kontuuri - koko kontuuria (satoja
    pisteita) ei tarvita, muutama kymmenen riittaa sovitukseen ja
    pitaa jokaisen residuals()-kutsun nopeana."""

    points = contour.reshape(-1, 2).astype(np.float64)

    if len(points) <= n_sample:
        return points

    idx = np.linspace(0, len(points) - 1, n_sample).astype(int)

    return points[idx]


def _profile_residuals_for_stone(pose, X0, Y0, R_max, H_total, shape_deltas, contour, n_sample):

    hull = _predicted_stone_hull(pose, X0, Y0, R_max, H_total, shape_deltas)
    sampled = _sample_contour_points(contour, n_sample)

    if hull is None:
        return np.full(len(sampled), 1000.0)

    return np.array([
        cv2.pointPolygonTest(hull, (float(p[0]), float(p[1])), True)
        for p in sampled
    ])


def fit_stone_profile(pose, stones, n_sample_per_stone=40,
                       initial_radius_cm=STONE_NOMINAL_RADIUS_CM,
                       height_cm=STONE_HEIGHT_CM,
                       shape_reg_weight=STONE_SHAPE_REG_WEIGHT):
    """
    HIENOSAATAA kovakoodatun karkean mallin (STONE_PROFILE_TEMPLATE_NORM)
    KAIKKIEN havaittujen kivien KOKO AARIVIIVAA vasten YHTEISESTI - katso
    taman osion alkupaan kommentti periaatteesta. Tuntemattomat: R_max
    (paivan/juoksurenkaan sade) + pieni korjaus (delta) JOKAISEN
    kontrollipisteen r_frac:iin (KOKO profiili, ei vain "paivan"
    ylapuolinen osa - katso _predicted_stone_hull:in kommentti MIKSI:
    matalasta kuvakulmasta nakyy aidosti myos paivan ALApuolista
    kylkea, joten sekin voi tulla oikeasti sovitetuksi, ei vain
    oletukseksi) + jokaisen kiven oma maa-asema (X0,Y0). Muotokorjaukset
    ovat REGULOITUJA (shape_reg_weight, SKAALATTUNA havaintojen maaran
    mukaan - katso alla) nollaa (=kovakoodattu malli) kohti, jotta
    havainnot eivat ylisovita muotoa - vain skaala ja KARKEA muototrendi
    (esim. onko malli hieman liian/liian vahan kupera) voi todella
    muuttua.

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

    HUOM height_cm EI OLE VAPAA PARAMETRI (kokeiltiin - katso git-
    historia): RMS-jaannosvirhe on kaytannossa LITTEA H_total:in
    suhteen valilla n. 6-12cm (kaikki n. 3.5px, ero vain kohinaa),
    koska 3 kivea + n. 10-25 asteen korkeuskulma-alue ei riita
    erottamaan "hieman pienempi R + suurempi H" ja "hieman suurempi R +
    pienempi H" -ratkaisuja toisistaan (nailla on lahes SAMA siluetti).
    Vapaana parametrina optimoija valitsi taman litean alueen SISALTA
    mielivaltaisesti (esim. H~8cm), mika EI ole mittaus vaan kohinaa -
    testattu antavan fysikaalisesti mahdottoman lyhyen kiven (WCF:n
    minimikorkeus on 11.43cm). R_max SEN SIJAAN ON hyvin rajoitettu
    (sama optimointi antaa R_max~14.0-14.6cm riippumatta kiinnitetysta
    H:sta, tasmaa mitattua ~28cm halkaisijaa vasten) - siksi height_cm
    KIINNITETAAN tunnettuun fysikaaliseen arvoon (WCF-vahimmaismitta
    oletuksena) sen sijaan etta yritettaisiin "ratkaista" jotain mita
    tama data ei yksinkertaisesti sisalla.

    stones: find_stone_candidates:in palauttamat dictit (tarvitaan seka
    "ellipse" etta "contour").

    Palauttaa dictin: R_max_cm (sovitettu), H_total_cm (KIINTEA, =
    height_cm), shape_deltas (hienosaadetut poikkeamat kovakoodattuun
    malliin), positions_cm, residuals_px (VAIN aariviiva-jaannokset,
    ilman regularisointitermeja), residual_rms_px.
    """

    if len(stones) < 2:
        raise RuntimeError(
            "Profiilin sovitukseen tarvitaan vahintaan 2 kiven havaintoa "
            f"(saatiin {len(stones)})."
        )

    # _TEMPLATE_EQUATOR_IDX:in delta EI OLE vapaa parametri - katso
    # _predicted_stone_hull:in kommentti: se piste MAARITTELEE R_max:in
    # (leveimman kohdan sade on R_max per maaritelma), joten oma vapaa
    # delta sille olisi redundantti (ja aiheutti rappeutuneen, ei-
    # monotonisen sovituksen - katso git-historia).
    free_idx = [i for i in range(len(STONE_PROFILE_TEMPLATE_NORM)) if i != _TEMPLATE_EQUATOR_IDX]
    n_shape = len(free_idx)
    effective_reg_weight = shape_reg_weight * (len(stones) / 3.0)

    positions0 = []

    for stone in stones:
        (cx, cy), _, _ = stone["ellipse"]
        X0, Y0 = _stone_ground_position_z0(pose, cx, cy)
        positions0.append((X0, Y0))

    def unpack(params):
        R_max = params[0]
        shape_deltas = np.zeros(len(STONE_PROFILE_TEMPLATE_NORM))
        shape_deltas[free_idx] = params[1:1 + n_shape]
        positions = params[1 + n_shape:].reshape(-1, 2)
        return R_max, shape_deltas, positions

    def residuals(params, include_reg=True):

        R_max, shape_deltas, positions = unpack(params)
        parts = []

        for (X0, Y0), stone in zip(positions, stones):
            parts.append(_profile_residuals_for_stone(
                pose, X0, Y0, R_max, height_cm, shape_deltas,
                stone["contour"], n_sample_per_stone
            ))

        if include_reg:
            parts.append(shape_deltas[free_idx] * effective_reg_weight)

        return np.concatenate(parts)

    params0 = np.concatenate([
        [initial_radius_cm],
        np.zeros(n_shape),
        np.array(positions0, dtype=np.float64).ravel(),
    ])

    params_final = k8._levenberg_marquardt(residuals, params0, max_iterations=100)
    R_max, shape_deltas, positions = unpack(params_final)
    resid_contour_only = residuals(params_final, include_reg=False)

    return {
        "R_max_cm": float(abs(R_max)),
        "H_total_cm": float(height_cm),
        "shape_deltas": shape_deltas,
        "positions_cm": [(float(x), float(y)) for x, y in positions],
        "residuals_px": resid_contour_only,
        "residual_rms_px": float(np.sqrt(np.mean(resid_contour_only ** 2))),
    }
