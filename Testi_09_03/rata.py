"""Radan mitat ja koordinaatistot.

Fyysinen koordinaatisto (cm): X = 0 keskiviivalla, Y kasvaa kohti kaukaista paata. Lahemman pesan keskipiste (T) on
Y = HOUSE_RADIUS_CM (pesan takareuna Y = 0), kaukaisen pesan keskipiste FAR_HOUSE_Y_CM.

Top-down-kuva (kalibroinnin tarkistuskuva ja hakualueet): PIXELS_PER_CM px/cm, X valilla OUTPUT_X_MIN..MAX, Y valilla
OUTPUT_Y_MIN..MAX. Y-akseli on KAANNETTY: lahempi pesa kuvan alaosassa, kaukainen ylaosassa.

HOGLINJAT: nimellinen paikka 6,40 m T-viivasta. Kalibrointi mittaa todellisen paikan (+-20 cm) ja asettaa sen
set_hogline_positions():lla -> lue AINA rata.NEAR_HOGLINE_Y_CM / rata.FAR_HOGLINE_Y_CM kutsuhetkella (ei kopiota).
"""
import numpy as np

# ---- pesien mitat ----
BLUE_OUTER_RADIUS_CM = 182.9
BLUE_INNER_RADIUS_CM = 121.9
RED_OUTER_RADIUS_CM = 61.0
RED_INNER_RADIUS_CM = 15.2
HOUSE_RADIUS_CM = BLUE_OUTER_RADIUS_CM
NEAR_HOUSE_Y_CM = HOUSE_RADIUS_CM               # lahemman pesan keskipiste (T-viiva)
HOUSE_DISTANCE_CM = 3474.7                      # T-viivojen etaisyys
FAR_HOUSE_Y_CM = NEAR_HOUSE_Y_CM + HOUSE_DISTANCE_CM

# ---- hoglinjat ----
HOG_LINE_DISTANCE_FROM_TEE_CM = 640.0
NOMINAL_NEAR_HOGLINE_Y_CM = NEAR_HOUSE_Y_CM + HOG_LINE_DISTANCE_FROM_TEE_CM
NOMINAL_FAR_HOGLINE_Y_CM = FAR_HOUSE_Y_CM - HOG_LINE_DISTANCE_FROM_TEE_CM
NEAR_HOGLINE_Y_CM = NOMINAL_NEAR_HOGLINE_Y_CM   # kalibroinnin mittaama paikka (set_hogline_positions)
FAR_HOGLINE_Y_CM = NOMINAL_FAR_HOGLINE_Y_CM


def set_hogline_positions(near_y_cm, far_y_cm):
    """Asettaa hoglinjojen mitatun paikan -> kayttoon kaikkialla (kameran asento, heittoportti, hog-analyysi, piirrot)."""
    global NEAR_HOGLINE_Y_CM, FAR_HOGLINE_Y_CM
    NEAR_HOGLINE_Y_CM = float(near_y_cm)
    FAR_HOGLINE_Y_CM = float(far_y_cm)


# ---- top-down-kuva ----
OUTPUT_X_MIN_CM = -200.0
OUTPUT_X_MAX_CM = 200.0
OUTPUT_Y_MIN_CM = 0.0
OUTPUT_Y_MAX_CM = 4000.0
PIXELS_PER_CM = 2.0


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
    """physical_to_output_px:n kaanteisfunktio: top-down-kuvan pisteet fyysisiksi (cm)."""
    pts = np.asarray(output_pts, dtype=np.float64)
    x_cm = pts[:, 0] / PIXELS_PER_CM + OUTPUT_X_MIN_CM
    y_cm = OUTPUT_Y_MAX_CM - pts[:, 1] / PIXELS_PER_CM
    return np.column_stack([x_cm, y_cm])


def apply_h(H, points):
    """Homografia H pisteisiin (N x 2)."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    ones = np.ones((len(pts), 1), dtype=np.float64)
    hom = np.hstack([pts, ones]) @ H.T
    return hom[:, :2] / hom[:, 2:3]
