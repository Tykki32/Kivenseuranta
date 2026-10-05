"""Kiven 3D-malli ja projektiot.

Kiven graniittiosa mallinnetaan pyorahdyskappaleena: normalisoitu profiili r_frac(z_frac) (mitattu oikeasta kivesta
sivukuvasta, alapuolisko; ylapuolisko on sen peilikuva, koska graniittirunko on symmetrinen puolivalin suhteen).
Profiilin opettelu (profiili.py, C++: fit_stone_profile_cpp) sovittaa havaintoihin kiven sateen R_max, korkeuden
H_total, muotopoikkeamat (shape_deltas) ja kahvan lovien sateen (handle_r_frac). Seuranta kayttaa mallin pisteita
kiven paikallisessa koordinaatistossa (build_local_stone_rings), jotka vain siirretaan ja projisoidaan.

Kameramalli: pose = {"K", "R", "t"} (kalibrointi.py); projektio p = K (R X + t).
"""
import math

import cv2
import numpy as np

# Kiven nimellismitat (WCF): vain profiilisovituksen alkuarvaus ja korkeuden rajat (H_total sovitetaan valille).
STONE_NOMINAL_RADIUS_CM = 91.44 / math.pi / 2.0
STONE_HEIGHT_CM = 11.43
STONE_HEIGHT_MAX_CM = 15.0

# Profiilin alapuolisko (z_frac, r_frac), mitattu kivesta (pohjan arvo arvioitu).
_HALF_PROFILE_TEMPLATE_NORM = [
    (0.00, 0.75),
    (0.18, 0.94),
    (0.30, 0.98),
    (0.45, 1.00),
    (0.50, 1.00),   # puolivali, leveimmillaan (symmetria-akseli)
]


def _mirror_half_profile(half_profile):
    """Taydet (z_frac, r_frac) -taulukot puoliskosta peilaamalla puolivalin ylapuolelle."""
    z_half = np.array([p[0] for p in half_profile], dtype=np.float64)
    r_half = np.array([p[1] for p in half_profile], dtype=np.float64)
    z_top = 1.0 - z_half[-2::-1]
    r_top = r_half[-2::-1]
    return np.concatenate([z_half, z_top]), np.concatenate([r_half, r_top])


TEMPLATE_Z_FRAC, TEMPLATE_R_FRAC = _mirror_half_profile(_HALF_PROFILE_TEMPLATE_NORM)
TEMPLATE_EQUATOR_IDX = len(_HALF_PROFILE_TEMPLATE_NORM) - 1

# Kahvan (muovi + kiinnitys) lovi kiven ylapinnassa, keskitettyna pystyakselille: sade suhteessa R_max:iin.
# Sovitetaan profiilin opettelussa valille MIN..MAX (alkuarvaus HANDLE_NOTCH_R_FRAC).
HANDLE_NOTCH_R_FRAC = 0.70
HANDLE_NOTCH_R_FRAC_MIN = 0.30
HANDLE_NOTCH_R_FRAC_MAX = 0.95
# Muotopoikkeamien rangaistus profiilisovituksessa (pitaa muodon lahella mitattua profiilia).
STONE_SHAPE_REG_WEIGHT = 60.0


def project_3d(K, R, t, points_3d):
    """3D-pisteet (N x 3, cm) -> kuvakoordinaatit (u, v)."""
    points_3d = np.asarray(points_3d, dtype=np.float64).reshape(-1, 3)
    p_cam = (R @ points_3d.T).T + t
    p_img = (K @ p_cam.T).T
    return p_img[:, 0] / p_img[:, 2], p_img[:, 1] / p_img[:, 2]


def ray_plane_intersection(K, R, t, u, v, Z_target):
    """Kuvapisteesta (u, v) lahtevan sateen leikkaus vaakatason Z = Z_target kanssa -> (X, Y) cm."""
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


def stone_ground_position_z0(pose, cx, cy):
    """Karkea maa-asema: Z = 0 -sadetasoleikkaus kuvapisteen (cx, cy) lapi (sisaltaa parallaksin)."""
    X, Y = ray_plane_intersection(pose["K"], pose["R"], pose["t"], cx, cy, 0.0)
    return float(X[0]), float(Y[0])


def catmull_rom_r_frac(z_query, z_fracs=TEMPLATE_Z_FRAC, r_fracs=None):
    """Silea (C1) Catmull-Rom-kayra profiilin kontrollipisteiden lapi (reunoilla paatepiste toistetaan)."""
    if r_fracs is None:
        r_fracs = TEMPLATE_R_FRAC
    n = len(z_fracs)
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


def build_local_stone_rings(R_max, H_total, shape_deltas, n_theta, n_per_segment):
    """(N, 3)-pisteet kiven omaan pystyakseliin (X0 = Y0 = 0) nahden: profiilin renkaat n_theta pisteella.
    Riippuu vain profiilista -> lasketaan kerran ja jokainen haku vain siirtaa ja projisoi pisteet."""
    R_max = abs(R_max)
    H_total = max(abs(H_total), 1e-6)
    r_fracs = TEMPLATE_R_FRAC + shape_deltas
    r_fracs[TEMPLATE_EQUATOR_IDX] = 1.0
    r_fracs = np.clip(r_fracs, 0.05, 1.0)
    n_dense = max(len(TEMPLATE_Z_FRAC) * n_per_segment, 2)
    z_frac_dense = np.linspace(0.0, 1.0, n_dense)
    r_frac_dense = catmull_rom_r_frac(z_frac_dense, r_fracs=r_fracs)
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    z = z_frac_dense * H_total
    r = r_frac_dense * R_max
    xs = r[:, None] * np.cos(theta)[None, :]
    ys = r[:, None] * np.sin(theta)[None, :]
    zs = np.broadcast_to(z[:, None], xs.shape)
    return np.stack([xs, ys, zs], axis=-1).reshape(-1, 3)


def predicted_stone_hull(local_pts, pose, X0, Y0):
    """Kiven siluetin kupera peite kuvassa paikassa (X0, Y0) (local_pts = build_local_stone_rings) tai None."""
    pts = local_pts + np.array([X0, Y0, 0.0])
    u, v = project_3d(pose["K"], pose["R"], pose["t"], pts)
    points_2d = np.column_stack([u, v]).astype(np.float32)
    if not np.all(np.isfinite(points_2d)):
        return None
    hull = cv2.convexHull(points_2d)
    if len(hull) < 3:
        return None
    return hull


def undistort_maps(camera_matrix, dist_coeffs, frame_size):
    """Linssikorjauksen kartat (cv2.remap) - lasketaan kerran, kalibrointi ei muutu ruutujen valilla."""
    return cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, camera_matrix, frame_size, cv2.CV_32FC1)
