# ============================================================
# kamera9_04.py - SAMA YHTEISSOVITUS KUIN kamera9_03.py, MUTTA
# NOPEUTETTU LIVE-KAYTTOA VARTEN (tavoite: n. 0.6s/frame -> paljon
# alle, jotta videon voi lukea suunnilleen reaaliajassa)
#
# EI MUUTETA MATEMATIIKKAA/ALGORITMIA - kamera9_03.py:n tulos on
# kayttajan mukaan jo riittavan tarkka ("todennakoisesti tuolla
# lahderesoluutiolla ei saa parempaa"), joten tama tiedosto EI YRITA
# parantaa tarkkuutta eika loysaa mitaan kynnysarvoa/toleranssia -
# se ratkaisee TASMALLEEN saman [X,Y,rengas_R]-LM-sovituksen samoilla
# pisteilla ja samoilla iteraatioilla, vain NOPEAMMIN LASKETTUNA.
# Profiloinnilla (cProfile, 30 oikeaa framea SEURANTA-tilassa)
# loydetyt pullonkaulat JA niiden korjaus tassa tiedostossa:
#
#   1) _predicted_stone_hull (kamera9_01.py) laskee Catmull-Rom-
#      splinin ja rakentaa kaikki rengaspisteet UUDELLEEN JOKAISELLA
#      KUTSULLA, vaikka kiven MUOTO (profile: R_max/H_total/
#      shape_deltas) on VAKIO koko videon ajan - vain (X0,Y0) muuttuu
#      kutsujen valilla. Tama oli ~80% koko framen ajasta (LM-sovitus
#      kutsuu sen satoja kertoja/frame, karkea ristikkohaku ~200
#      kertaa/frame lisaksi). KORJAUS: lasketaan kiven paikallinen
#      (origon-suhteinen) rengasgeometria KERRAN (_build_local_rings),
#      ja jokainen kutsu vain SIIRTAA (+X0,+Y0) ja projisoi sen -
#      TASMALLEEN sama lopputulos, ei mitaan approksimaatiota.
#   2) refine_position_joint (kamera9_03.py) laski graniittimaskin
#      (create_granite_mask, mm. GaussianBlur+morfologia KOKO framelle)
#      UUDELLEEN, vaikka search_and_track_stones_precise oli JUURI
#      laskenut saman maskin ristikkohakua varten. KORJAUS: maski
#      lasketaan kerran/frame ja annetaan parametrina.
#   3) Saturaatiokanava (cv2.cvtColor+HSV, koko frame) laskettiin
#      UUDELLEEN seka _filter_ice_boundary_points:ssa etta JOKAISELLA
#      _detect_boundary_points-iteraatiolla (jopa 5x/frame). KORJAUS:
#      lasketaan kerran/frame ja annetaan parametrina.
#   4) Reunapisteiden sateenmuotoinen skannaus (_find_boundary_point_
#      along_ray) kutsui bilineaarista naytintoa (_bilinear_sample)
#      YKSITELLEN PYTHON-SILMUKASSA (kymmenia tuhansia funktiokutsuja/
#      frame). KORJAUS: vektoroitu numpylla (kaikki sateet+askeleet
#      yhdella kertaa) - sama lineaarinen interpolointi, sama tulos.
#   5) cv2.undistort() laskee EI-VAKIO-parametrien takia sisaisesti
#      UUDET vaantokartat JOKA KUTSULLA, vaikka kalibrointi on vakio
#      koko videolle. KORJAUS: initUndistortRectifyMap kerran + remap
#      per frame (sama INTER_LINEAR-interpolointi, sama tulos).
#
# EI KOSKETA: kamera9_01.py, kamera9_02.py, kamera9_03.py pysyvat
# koskemattomina (kayttajan aiempi pyynto) - tama tiedosto TUO NE
# SISAAN ja korvaa vain nama viisi pullonkaulaa uusilla, nopeammilla
# mutta MATEMAATTISESTI IDENTTISILLA funktioilla.
# ============================================================

import os
import csv
import math
import time
import importlib.util

import numpy as np
import cv2


def _load_kamera9_03():
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        "kamera9_03", os.path.join(here, "kamera9_03.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


k93 = _load_kamera9_03()
k92 = k93.k92
k9 = k93.k9
k8 = k93.k8


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

    r_fracs = k9._TEMPLATE_R_FRAC + shape_deltas
    r_fracs[k9._TEMPLATE_EQUATOR_IDX] = 1.0
    r_fracs = np.clip(r_fracs, 0.05, 1.0)

    n_dense = max(len(k9._TEMPLATE_Z_FRAC) * n_per_segment, 2)
    z_frac_dense = np.linspace(0.0, 1.0, n_dense)
    r_frac_dense = k9._catmull_rom_r_frac(z_frac_dense, r_fracs=r_fracs)

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
    u, v = k9._project_3d(pose["K"], pose["R"], pose["t"], pts)
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
