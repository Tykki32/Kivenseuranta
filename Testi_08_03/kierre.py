"""Kahvan vari ja kiven pyoriminen (kierteet hog-hog-valilla).

KAHVA: jokaisessa seurantaruudussa kiven pyorimisakselin ylapaa (X, Y, H_total) projisoidaan kuvaan; sen ympari kuvaan
ympyra, sateena A.KAHVA_SADE_KERROIN x kahvan sade (profiilista) kuvaan projisoituna. Ympyran sisalta kerataan
varikkaiden pikseleiden savyt (S >= A.KAHVA_MIN_S, V >= A.KAHVA_MIN_V; jaa ja graniitti ovat harmaita). Radan
paattyessa kahvan savy = savy, jonka +-A.KAHVA_SAVY_TOL valilla on eniten pikseleita -> <csv>_kahva.csv.

KIERTEET: kahvan ympyra skaalataan joka ruudussa 32x32-palaksi, josta lasketaan harmaasavyn gradientti (Sobel).
Kahvan muoto toistuu pyoriessa (vahvin toisto PUOLIKIERROS: vastakkaisiin suuntiin osoittava kahva nayttaa lahes
samalta). Pyoriminen hidastuu: omega(t) = omega_loppu + A.KIERRE_HIDASTUVUUS (t_loppu - t) (0,02 rad/s^2, MAH00014:n
12 selvan kiven yhteinen arvo). Kierrosaika sovitetaan ruutuparien samankaltaisuuteen; kierteet = integraali
omega dt / 2 pi kaukohogista lahihogiin. Kahva erottuu vasta n. 23 m:sta -> alkuosa ekstrapoloidaan.
"""
import csv
import os

import cv2
import numpy as np

import asetukset as A
import kivimalli

_KAHVA_ANG = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
_KIERRE_W = np.exp(np.linspace(np.log(2 * np.pi / 14.0), np.log(2 * np.pi / 2.0), 240))   # koko kierros 2-14 s


def kierre_piirre(frame_u, cx, cy, rp):
    """32x32 gradienttipala kahvan ympyrasta (normalisoitu, float16) tai None."""
    k = 16.0 / rp
    M = np.float32([[k, 0, 16 - k * cx], [0, k, 16 - k * cy]])
    p = cv2.warpAffine(frame_u, M, (32, 32), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    g = cv2.GaussianBlur(cv2.cvtColor(p, cv2.COLOR_BGR2GRAY).astype(np.float32), (3, 3), 0.7)
    v = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)).ravel()
    v -= v.mean()
    n = float(np.linalg.norm(v))
    return (v / n).astype(np.float16) if n > 1e-6 else None


def kierre_arvio(obs, fps, t_far, t_near, alpha=None, max_lag_s=7.0, n_pairs=8000):
    """obs: [(ruutu, piirre)]. Palauttaa dict(kierrokset, kierrosaika_far_s, kierrosaika_near_s, r2, n) tai None."""
    if alpha is None:
        alpha = A.KIERRE_HIDASTUVUUS
    if len(obs) < A.KIERRE_MIN_RUUDUT:
        return None
    T = np.array([o[0] for o in obs], np.float64) / fps
    F = np.array([o[1] for o in obs], np.float32)
    S = F @ F.T
    n = len(T)
    i, k = np.triu_indices(n, 1)
    L = T[k] - T[i]
    m = (L > 0.15) & (L < max_lag_s)
    i, k, L = i[m], k[m], L[m]
    s = S[i, k].astype(np.float64)
    for _ in range(3):      # ruudun oma taso (harjan peitto, etaisyys) pois
        cnt = np.maximum(np.bincount(i, None, n) + np.bincount(k, None, n), 1)
        rd = (np.bincount(i, s, n) + np.bincount(k, s, n)) / cnt
        rd -= rd.mean()
        s = s - rd[i] - rd[k]
    if len(s) > n_pairs:
        q = np.random.default_rng(0).choice(len(s), n_pairs, replace=False)
        i, k, L, s = i[q], k[q], L[q], s[q]
    if len(s) < 500:
        return None
    Q, _ = np.linalg.qr(np.column_stack([np.ones_like(L), L, L ** 2, L ** 3]))   # hidas viiveriippuvuus pois
    s = s - Q @ (Q.T @ s)
    ss = float(s @ s)
    if ss <= 0:
        return None
    t_end = T[-1]
    u = t_end - T
    base = alpha * (u[i] ** 2 - u[k] ** 2) / 2.0
    X = np.cos(2.0 * (np.outer(_KIERRE_W, L) + base[None, :]))                 # puolikierroksen toisto
    X -= (X @ Q) @ Q.T
    c = np.maximum((X @ s) / np.maximum(np.einsum("ij,ij->i", X, X), 1e-12), 0.0)
    r2 = c * (X @ s) / ss
    j = int(np.argmax(r2))
    if r2[j] < A.KIERRE_MIN_R2:
        return None
    w_end = float(_KIERRE_W[j])
    om = lambda t: w_end + alpha * (t_end - t)
    if om(t_near) <= 0:
        return None
    rad = w_end * (t_near - t_far) + alpha * ((t_end - t_far) ** 2 - (t_end - t_near) ** 2) / 2.0
    return dict(kierrokset=rad / (2 * np.pi), kierrosaika_far_s=2 * np.pi / om(t_far),
                kierrosaika_near_s=2 * np.pi / om(t_near), r2=float(r2[j]), n=n)


def kahva_hist(frame_u, pose, X, Y, Z, r_cm, piirre=False):
    """Kahvan ympyran savyhistogrammi kuvasta. Palauttaa (savylokerot, maarat, ympyran pinta-ala px, levyn ellipsin
    pinta-ala px, kierrepiirre tai None, (cx, cy, rp)) tai None. Ympyra on suoraan kuvassa (ei projisoitu ellipsi)."""
    pts = np.array([[X, Y, Z], [X + r_cm, Y, Z], [X - r_cm, Y, Z]], dtype=np.float64)
    pts = np.vstack([pts, np.column_stack([X + r_cm * np.cos(_KAHVA_ANG), Y + r_cm * np.sin(_KAHVA_ANG),
                                           np.full(_KAHVA_ANG.size, Z)])])
    u, v = kivimalli.project_3d(pose["K"], pose["R"], pose["t"], pts)
    if not (np.all(np.isfinite(u)) and np.all(np.isfinite(v))):
        return None
    # kahvan alla olevan levyn (vaakasuora r_cm-ympyra) projisoidun ellipsin pinta-ala: nakyy aina pyorimisesta riippumatta
    levy = float(cv2.contourArea(np.column_stack([u[3:], v[3:]]).astype(np.float32)))
    cx, cy = float(u[0]), float(v[0])
    rp = max(1.0, 0.5 * A.KAHVA_SADE_KERROIN * float(np.hypot(u[1] - u[2], v[1] - v[2])))
    h_img, w_img = frame_u.shape[:2]
    x0, x1 = int(max(0, np.floor(cx - rp))), int(min(w_img, np.ceil(cx + rp) + 1))
    y0, y1 = int(max(0, np.floor(cy - rp))), int(min(h_img, np.ceil(cy + rp) + 1))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    yy, xx = np.mgrid[y0:y1, x0:x1]
    circ = (xx - cx) ** 2 + (yy - cy) ** 2 <= rp * rp
    hsv = cv2.cvtColor(frame_u[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    sel = circ & (hsv[:, :, 1] >= A.KAHVA_MIN_S) & (hsv[:, :, 2] >= A.KAHVA_MIN_V)
    hist = np.bincount(hsv[:, :, 0][sel].ravel(), minlength=180)
    nz = np.nonzero(hist)[0]
    pv = kierre_piirre(frame_u, cx, cy, rp) if piirre else None
    return nz.astype(np.uint8), hist[nz].astype(np.uint16), int(np.count_nonzero(circ)), int(round(levy)), pv, (cx, cy, rp)


def kahva_vari_nimi(h):
    if h is None:
        return ""
    return ("punainen" if h <= 8 or h >= 170 else "oranssi" if h <= 20 else "keltainen" if h <= 34 else
            "vihrea" if h <= 85 else "sininen" if h <= 130 else "violetti" if h <= 155 else "pinkki")


def _kahva_window(hist, h, tol):
    idx = (np.arange(h - tol, h + tol + 1) % 180)
    return int(hist[idx].sum())


def kahva_kirjoita(stone_registry, hog_results, csv_output):
    """Radan kahvan savy + ruutukohtaiset pikselimaarat -> <csv>_kahva.csv; hog-tuloksiin kahva_h ja kahva_vari."""
    path = os.path.splitext(csv_output)[0] + "_kahva.csv"
    hues = {}
    n_rows = 0
    tol = A.KAHVA_SAVY_TOL
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "stone_id", "kahva_h", "kahva_vari", "kahva_px", "ympyra_px", "levy_px", "kahva_netto_px"])
        for sid, st in sorted(stone_registry.items()):
            obs = st.get("kahva")
            if not obs:
                continue
            tot = np.zeros(180, np.int64)
            for _, nz, cnt, _a, _l in obs:
                tot[nz] += cnt
            if tot.sum() == 0:
                continue
            win = np.array([_kahva_window(tot, h_, tol) for h_ in range(180)])
            h_best = int(np.argmax(win))
            hues[sid] = h_best
            idx = set(int(i) for i in (np.arange(h_best - tol, h_best + tol + 1) % 180))
            name = kahva_vari_nimi(h_best)
            for fr, nz, cnt, area, levy in obs:
                px = int(sum(int(c) for b, c in zip(nz, cnt) if int(b) in idx))
                w.writerow([fr, sid, h_best, name, px, area, levy, px - levy])
                n_rows += 1
    for r in hog_results:
        h_ = hues.get(r.get("stone_id"))
        r["kahva_h"] = h_
        r["kahva_vari"] = kahva_vari_nimi(h_)
    print(f"Kahvan vari: {len(hues)} rataa, {n_rows} ruutua -> {path}")
