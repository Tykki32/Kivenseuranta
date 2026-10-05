"""Kahvan vari ja kiven pyoriminen (kierteet hog-hog-valilla).

KAHVA: jokaisessa seurantaruudussa kiven pyorimisakselin ylapaa (X, Y, H_total) projisoidaan kuvaan; sen ympari kuvaan
ympyra, sateena A.KAHVA_SADE_KERROIN x kahvan sade (profiilista) kuvaan projisoituna. Ympyran sisalta kerataan
varikkaiden pikseleiden savyt (S >= A.KAHVA_MIN_S, V >= A.KAHVA_MIN_V; jaa ja graniitti ovat harmaita). Radan
paattyessa kahvan savy = savy, jonka +-A.KAHVA_SAVY_TOL valilla on eniten pikseleita -> <csv>_kahva.csv.

KIERTEET: joka ruudusta kiven akselin suuntainen 32x32-pala kahvasta (ylimmat rivit = kahvan profiili), harmaasavyn
gradientti (Sobel). Peittyneet ruudut (harja, kenka) hylataan ajallisen jatkuvuuden perusteella. Kahvan muoto toistuu
PUOLIKIERROKSEN valein. Pyoriminen hidastuu: omega(t) = omega_loppu + A.KIERRE_HIDASTUVUUS (t_loppu - t). Loppukierrosaika
haetaan laajalta valilta (0,6-40 s) mallilla c1 cos(2 dphi) + c2 cos(4 dphi) kaikkien ruutuparien samankaltaisuuteen;
hyvaksynta R^2:n ja erottuvuuden (huippu vs. muut kuin kerrannaiset) mukaan. Kierteet = integraali omega dt / 2 pi
kaukohogista lahihogiin. Kahva erottuu vasta n. 23 m:sta -> alkuosa ekstrapoloidaan.
"""
import csv
import os

import cv2
import numpy as np

import asetukset as A
import kivimalli

_KAHVA_ANG = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
# kierrosajan (loppu) hakuvali: laaja, ettei hakuvali rajaa tulosta (MAH00014: 1-10 kierrosta hog-hog)
_KIERRE_P = np.exp(np.linspace(np.log(A.KIERRE_P_MAX_S), np.log(A.KIERRE_P_MIN_S), A.KIERRE_P_N))
_KIERRE_W = 2.0 * np.pi / _KIERRE_P


def kierre_piirre(img, cx, cy, gx, gy, rp, size=32):
    """Kiven akselin suuntainen gradienttipala kahvasta (normalisoitu, float16) tai None.
    Palan "ylos" = kiven akseli kuvassa (maan keskipisteesta (gx, gy) ylapinnan keskipisteeseen (cx, cy)), joten kahvan
    profiili on palassa aina samassa asennossa kameran kallistuksesta ja kuvan kierrosta riippumatta. Pala kattaa
    +-rp (kahvan ympyran sade) ylapinnan keskipisteen ympari; mukaan otetaan vain ylimmat A.KIERRE_PALA_RIVIT
    (kahvan profiili graniitin ylapuolella, ei kiven reunaa ja jaata)."""
    a = np.array([cx - gx, cy - gy], np.float64)
    na = float(np.hypot(a[0], a[1]))
    if na < 1e-6:
        return None
    a /= na
    l = np.array([-a[1], a[0]])
    k = rp / (size / 2.0)                          # px kuvassa / px palassa
    h = size / 2.0 - 0.5
    M = np.zeros((2, 3), np.float64)
    M[:, 0] = k * l
    M[:, 1] = -k * a
    M[:, 2] = np.array([cx, cy]) - h * k * l + h * k * a
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    p = cv2.warpAffine(g.astype(np.float32), cv2.invertAffineTransform(M), (size, size), flags=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REPLICATE)
    p = cv2.GaussianBlur(p, (0, 0), 0.7)
    v = cv2.magnitude(cv2.Sobel(p, cv2.CV_32F, 1, 0), cv2.Sobel(p, cv2.CV_32F, 0, 1))
    v = v[:int(size * A.KIERRE_PALA_RIVIT), :].ravel()
    v -= v.mean()
    n = float(np.linalg.norm(v))
    return (v / n).astype(np.float16) if n > 1e-6 else None


def _jatkuvat(F):
    """Ruutujen ajallinen jatkuvuus: piirteen mediaanisamankaltaisuus naapureihin t+-1, t+-2. Ruutu hylataan, jos se
    on poikkeava (< mediaani - 4 x 1,4826 x MAD; harja, kenka tai heittaja kahvan edessa). Palauttaa maskin."""
    n = len(F)
    if n < 6:
        return np.ones(n, bool)
    q = np.zeros(n)
    for k in range(n):
        q[k] = np.median([float(F[k] @ F[j]) for j in (k - 2, k - 1, k + 1, k + 2) if 0 <= j < n])
    med = np.median(q)
    mad = np.median(np.abs(q - med)) + 1e-6
    return q > med - 4.0 * 1.4826 * mad


def kierre_arvio(obs, fps, t_far, t_near, alpha=None, n_pairs=12000, chunk=100):
    """obs: [(ruutu, piirre)]. Palauttaa dict(kierrokset, kierrosaika_far_s, kierrosaika_near_s, r2, erottuvuus, n)
    tai None."""
    if alpha is None:
        alpha = A.KIERRE_HIDASTUVUUS
    if not obs:
        return None
    T = np.array([o[0] for o in obs], np.float64) / fps
    F = np.array([o[1] for o in obs], np.float32)
    m = _jatkuvat(F)
    T, F = T[m], F[m]
    n = len(T)
    if n < A.KIERRE_MIN_RUUDUT:
        return None
    S = F @ F.T
    i, k = np.triu_indices(n, 1)
    L = T[k] - T[i]
    m = L > 0.08                                   # kaikki viiveet (pitkat viiveet erottavat lahekkaiset kierrosajat)
    i, k, L = i[m], k[m], L[m]
    s = S[i, k].astype(np.float64)
    for _ in range(3):      # ruudun oma taso (peitto, etaisyys, valaistus) pois
        cnt = np.maximum(np.bincount(i, None, n) + np.bincount(k, None, n), 1)
        rd = (np.bincount(i, s, n) + np.bincount(k, s, n)) / cnt
        rd -= rd.mean()
        s = s - rd[i] - rd[k]
    if len(s) > n_pairs:
        q = np.random.default_rng(0).choice(len(s), n_pairs, replace=False)
        i, k, L, s = i[q], k[q], L[q], s[q]
    if len(s) < 500:
        return None
    Ln = L / L.max()
    Q, _ = np.linalg.qr(np.column_stack([np.ones_like(Ln), Ln, Ln ** 2, Ln ** 3]))   # hidas viiveriippuvuus pois
    s = s - Q @ (Q.T @ s)
    ss = float(s @ s)
    if ss <= 0:
        return None
    t_end = T[-1]
    u = t_end - T
    base = alpha * (u[i] ** 2 - u[k] ** 2) / 2.0
    # malli: s = c1 cos(2 dphi) + c2 cos(4 dphi) (puolikierroksen toisto + sen yliaalto), dphi = kiertokulma ruutujen
    # valilla. c1 > 0 ja c2 <= c1 vaaditaan: muuten kaksinkertainen kierrosaika sopisi yliaallon kautta.
    nw = len(_KIERRE_W)
    r2 = np.zeros(nw)
    for a in range(0, nw, chunk):
        ph = np.outer(_KIERRE_W[a:a + chunk], L) + base[None, :]
        X1 = np.cos(2.0 * ph)
        X1 -= (X1 @ Q) @ Q.T
        X2 = np.cos(4.0 * ph)
        X2 -= (X2 @ Q) @ Q.T
        a11 = np.einsum("ij,ij->i", X1, X1)
        a22 = np.einsum("ij,ij->i", X2, X2)
        a12 = np.einsum("ij,ij->i", X1, X2)
        b1, b2 = X1 @ s, X2 @ s
        det = a11 * a22 - a12 ** 2 + 1e-12
        c1 = (a22 * b1 - a12 * b2) / det
        c2 = (a11 * b2 - a12 * b1) / det
        r2[a:a + chunk] = np.where((c1 > 0) & (c2 <= c1), (c1 * b1 + c2 * b2) / ss, 0.0)
    j = int(np.argmax(r2))
    # erottuvuus: paras huippu / paras huippu kun +-30 % ja kerrannaiset (x2, /2) ovat pois
    P = _KIERRE_P
    muu = np.ones(nw, bool)
    for f_ in (1.0, 2.0, 0.5):
        muu &= ~((P > P[j] * f_ / 1.3) & (P < P[j] * f_ * 1.3))
    erott = float(r2[j] / max(r2[muu].max() if muu.any() else 0.0, 1e-5))
    if r2[j] < A.KIERRE_MIN_R2 or erott < A.KIERRE_MIN_EROTTUVUUS:
        return None
    w_end = float(_KIERRE_W[j])
    om = lambda t: w_end + alpha * (t_end - t)
    if om(t_near) <= 0:
        return None
    rad = w_end * (t_near - t_far) + alpha * ((t_end - t_far) ** 2 - (t_end - t_near) ** 2) / 2.0
    return dict(kierrokset=rad / (2 * np.pi), kierrosaika_far_s=2 * np.pi / om(t_far),
                kierrosaika_near_s=2 * np.pi / om(t_near), r2=float(r2[j]), erottuvuus=erott, n=n)


def kahva_hist(frame_u, pose, X, Y, Z, r_cm, piirre=False):
    """Kahvan ympyran savyhistogrammi kuvasta. Palauttaa (savylokerot, maarat, ympyran pinta-ala px, levyn ellipsin
    pinta-ala px, kierrepiirre tai None, (cx, cy, rp, gx, gy)); (gx, gy) = kiven maan keskipiste kuvassa tai None. Ympyra on suoraan kuvassa (ei projisoitu ellipsi)."""
    pts = np.array([[X, Y, Z], [X + r_cm, Y, Z], [X - r_cm, Y, Z]], dtype=np.float64)
    pts = np.vstack([pts, [[X, Y, 0.0]], np.column_stack([X + r_cm * np.cos(_KAHVA_ANG), Y + r_cm * np.sin(_KAHVA_ANG),
                                           np.full(_KAHVA_ANG.size, Z)])])
    u, v = kivimalli.project_3d(pose["K"], pose["R"], pose["t"], pts)
    if not (np.all(np.isfinite(u)) and np.all(np.isfinite(v))):
        return None
    # kahvan alla olevan levyn (vaakasuora r_cm-ympyra) projisoidun ellipsin pinta-ala: nakyy aina pyorimisesta riippumatta
    levy = float(cv2.contourArea(np.column_stack([u[4:], v[4:]]).astype(np.float32)))
    cx, cy = float(u[0]), float(v[0])
    gx, gy = float(u[3]), float(v[3])
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
    pv = kierre_piirre(frame_u, cx, cy, gx, gy, rp) if piirre else None
    return nz.astype(np.uint8), hist[nz].astype(np.uint16), int(np.count_nonzero(circ)), int(round(levy)), pv, (cx, cy, rp, gx, gy)


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
