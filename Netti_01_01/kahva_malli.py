"""Kahvan muovin kiintean kokoinen malli (oppimisvaihe): kaikilla kahvoilla oletetaan sama muovin koko.

Muovin juuri on vaakasuora soikio (puoliakselit A sivusuunnassa (X), B radan suunnassa (Y)), jonka keskipiste on kiven akselilla
korkeudella Z0_CM. Soikio projisoidaan kameralla kuvaan; per ruutu sovitetaan vain akselin paikka (X, Y). Koko (A, B) on kaikille
ruuduille yhteinen ja opitaan puhtaista ruuduista (`opi_koko`); lopullisessa koodissa kokoa ei voi olettaa, sillä kahvat voivat erota.

Havainto ja malli (alipikseli, kuvan sumeus huomioiden):
* havainto p (0..1): muovin vari (keltainen: b*, punainen: a*) taustasta (moodikuva) muovin tasanteeseen
* malli: soikion peitto sumennettuna kromakanavan pistelevityksella SIGMA_PX (a*, b* ovat videossa n. 2,1 px:n sumeita, L vain 0,8 px;
  mitattu renkaiden reunoista) -> varikynnyksellä saatu silhuetti ja erityisesti sen karjet jaavat sisaanpain, malli korjaa tamän
* vahvistus ja siirto ratkaistaan ruuduittain lineaarisesti (kahvan varjostus ei vaikuta muotoon)
* kasi, hanska ja iho ovat "ei valia" -aluetta (paino 0): kasi ellipsin sisalla ei haittaa, vain jaa ei saa nakya
* vain etupuoli (kameran puoli) sovitetaan: takaosan muovi (kupu, kahvapalkki) ja kasi eivat vaikuta; tipat tulevat koosta."""
import numpy as np
import cv2
from scipy.optimize import least_squares

import kivi_netti as kv

Z0_CM = 17.0              # soikion korkeus (aiempi arvio: 16,9-17,1 cm, +-3 cm)
A_CM = 11.0               # puoliakseli sivusuunnassa (opittu puhtaista keltaisista ruuduista, Y = 330-520 cm)
B_CM = 8.3                # puoliakseli radan suunnassa (opittu; vaihtelee 8,0-8,6 heittojen valilla)
SIGMA_PX = 2.1            # kromakanavan (a*, b*) pistelevitys px (renkaiden reunoista; L: 0,8 px)
H = 56                    # leikkeen puolikas (px)
N = 2 * H
SS = 4                    # peiton ylinaytteistus
TH = np.linspace(0.0, 2.0 * np.pi, 120, endpoint=False)
CT, ST = np.cos(TH), np.sin(TH)


class Kevyt:
    """Vain kameran asento (K, R, t): kevyt, jotta monisaikeisuus ei kopioi linssikorjauskarttoja."""

    def __init__(self, cam):
        self.K, self.R, self.t = cam.K, cam.R, cam.t


def soikio_kuvaan(cam, X, Y, z0=Z0_CM, A=A_CM, B=B_CM):
    """Soikion reuna kuvassa (120 pistetta, kuvapikselit)."""
    pts = np.column_stack([X + A * CT, Y + B * ST, np.full_like(TH, z0)])
    u, v = kv.kivimalli.project_3d(cam.K, cam.R, cam.t, pts)
    return np.column_stack([u, v])


def peitto(poly, x0, y0, sigma=SIGMA_PX):
    """Soikion peitto leikkeessa (alipikseli, SS-kertainen ylinaytteistus) sumennettuna."""
    q = (poly - np.array([x0, y0]) + 0.5) * SS - 0.5
    m = np.zeros((N * SS, N * SS), np.uint8)
    cv2.fillConvexPoly(m, np.round(q * 16).astype(np.int32), 255, cv2.LINE_8, 4)
    c = cv2.resize(m, (N, N), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    return cv2.GaussianBlur(c, (0, 0), sigmaX=sigma, sigmaY=sigma)


def leike(cam, frame, uv0):
    """Leike (kehys, moodikuva) kuvapisteen uv0 ymparilta, reunat replikoiden. Palauttaa (crop, mode, x0, y0)."""
    h, w = frame.shape[:2]
    x0, y0 = int(round(uv0[0])) - H, int(round(uv0[1])) - H
    xs = np.clip(np.arange(x0, x0 + N), 0, w - 1)
    ys = np.clip(np.arange(y0, y0 + N), 0, h - 1)
    return frame[np.ix_(ys, xs)].copy(), cam.mode[np.ix_(ys, xs)].copy(), x0, y0


def havainto(crop, mode, vari):
    """Muovin peitto p (0..1), huomioitavat pikselit w0 (0/1) ja kontrasti (tasanne - tausta) tai None jos muovia ei loydy."""
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    lm = cv2.cvtColor(mode, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[..., 0] * (100.0 / 255.0)
    a, b = lab[..., 1] - 128.0, lab[..., 2] - 128.0
    am, bm = lm[..., 1] - 128.0, lm[..., 2] - 128.0
    if vari == "keltainen":
        f, fm = b, bm
        ei_valia = ((a > 10) & (b > 4)) | (L < 30)                       # iho, hanska
        karkea = ((b - bm) >= 14) & (a <= 10)
    else:
        f, fm = a, am
        ei_valia = (L < 25) | ((L > 52) & (b > 8) & (a > 8))             # hanska, iho
        karkea = ((a - am) >= 18) & (L <= 65)
    karkea = cv2.morphologyEx(karkea.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, lab_c, st, cen = cv2.connectedComponentsWithStats(karkea)
    ehdokkaat = [i for i in range(1, n) if st[i, 4] >= 30]
    if not ehdokkaat:
        return None
    k = min(ehdokkaat, key=lambda i: np.hypot(cen[i][0] - H, cen[i][1] - H))
    kontrasti = float(np.percentile((f - fm)[lab_c == k], 75))
    if kontrasti < 12.0:
        return None
    p = np.clip((f - fm) / kontrasti, 0.0, 1.0).astype(np.float32)
    return p, (~ei_valia).astype(np.float32), kontrasti


def _lin(cv, po, wv):
    """Vahvistus ja siirto (pienimman neliosumman) -> jaannos."""
    M = np.column_stack([cv, np.ones_like(cv)]) * wv[:, None]
    s = np.linalg.lstsq(M, po * wv, rcond=None)[0]
    g = float(np.clip(s[0], 0.6, 1.6))
    o = float(np.clip(s[1], -0.25, 0.25))
    return g, o, (g * cv + o - po) * wv


def sovita(cam, crop, mode, x0, y0, vari, uv0, A=A_CM, B=B_CM, z0=Z0_CM, sigma=SIGMA_PX):
    """Akselin (X, Y) sovitus ruutuun kiintean kokoisella soikiolla. uv0 = karkea kuvapiste (kahvan vari-seuranta).
    Vaiheet: (1) ruudukkohaku etupuolen peittovirheella, (2) tarkennus pienimmalla neliosumma etureunan kaistalla (soft-L1).
    Palauttaa dict (X, Y, poly, rms, n, vahvistus, siirto, kontrasti) tai None."""
    h = havainto(crop, mode, vari)
    if h is None:
        return None
    p, w0, kontrasti = h
    X0, Y0 = kv.maan_paikka(cam, uv0[0], uv0[1], z=z0)
    cols = np.arange(N, dtype=np.float32)[None, :] * np.ones((N, 1), np.float32)
    rows = np.arange(N, dtype=np.float32)[:, None] * np.ones((1, N), np.float32)

    def kuva(X, Y):
        poly = soikio_kuvaan(cam, X, Y, z0, A, B)
        return poly, peitto(poly, x0, y0, sigma)

    # (1) ruudukko: etupuolen (u >= keskipiste - 2) pikselit soikion korkeusvalilla, vahvistus ja siirto vapaat
    best = None
    for dX in np.arange(-5.0, 5.01, 1.0):
        for dY in np.arange(-10.0, 10.01, 2.0):
            poly, c = kuva(X0 + dX, Y0 + dY)
            uc, vmin, vmax = poly[:, 0].mean() - x0, poly[:, 1].min() - y0, poly[:, 1].max() - y0
            sel = (cols >= uc - 2.0) & (rows >= vmin - 6.0) & (rows <= vmax + 6.0) & (w0 > 0) & (c < 0.99)
            if sel.sum() < 40:
                continue
            g, o, r = _lin(c[sel], p[sel], np.ones(int(sel.sum()), np.float32))
            e = float(np.mean(r ** 2))
            if best is None or e < best[0]:
                best = (e, X0 + dX, Y0 + dY)
    if best is None:
        return None
    _, Xg, Yg = best
    # (2) tarkennus: etureunan kaista (siirtymavyohyke +- 3 px), kiinteat pikselit
    poly0, c0 = kuva(Xg, Yg)
    uc0 = poly0[:, 0].mean() - x0
    trans = ((c0 > 0.01) & (c0 < 0.99)).astype(np.uint8)
    kaista = cv2.dilate(trans, np.ones((7, 7), np.uint8)) > 0
    sel = (cols >= uc0 - 2.0) & kaista & (w0 > 0)
    idx = np.where(sel)
    if len(idx[0]) < 40:
        return None
    wv, po = w0[idx], p[idx]

    def res(q):
        poly = soikio_kuvaan(cam, q[0], q[1], z0, A, B)
        c = peitto(poly, x0, y0, sigma)
        return _lin(c[idx], po, wv)[2]

    sol = least_squares(res, [Xg, Yg], loss="soft_l1", f_scale=0.25, diff_step=1e-3, max_nfev=40,
                        bounds=([Xg - 2.0, Yg - 4.0], [Xg + 2.0, Yg + 4.0]))
    r = res(sol.x)
    poly = soikio_kuvaan(cam, sol.x[0], sol.x[1], z0, A, B)
    cfin = peitto(poly, x0, y0, sigma)
    g, o, _ = _lin(cfin[idx], po, wv)
    return dict(X=float(sol.x[0]), Y=float(sol.x[1]), poly=poly, rms=float(np.sqrt(np.mean(r ** 2))), n=int(len(idx[0])),
                cost=float(np.sum(r ** 2)), vahvistus=g, siirto=o, kontrasti=kontrasti)


def _opi_tyo(args):
    cam, crop, mode, x0, y0, vari, uv0, A, B = args
    r = sovita(cam, crop, mode, x0, y0, vari, uv0, A, B)
    return None if r is None else (r["cost"], r["n"])


def opi_koko(cam, leikkeet, A0=A_CM, B0=B_CM, pool=None):
    """Yhteinen koko (A, B) puhtaista ruuduista: minimoi soikiopeiton neliovirheen summan (Nelder-Mead).
    leikkeet = [(crop, mode, x0, y0, vari, uv0)]. Palauttaa (A, B, J) (J = painotettu keskineliovirhe)."""
    from scipy.optimize import minimize

    def J(q):
        args = [(cam, c, m, x0, y0, v, uv, q[0], q[1]) for (c, m, x0, y0, v, uv) in leikkeet]
        res = pool.map(_opi_tyo, args) if pool is not None else [_opi_tyo(a) for a in args]
        res = [r for r in res if r is not None]
        return sum(r[0] for r in res) / max(sum(r[1] for r in res), 1)

    x0 = np.array([A0, B0])
    sol = minimize(J, x0, method="Nelder-Mead", options=dict(initial_simplex=[x0, x0 + [0.5, 0.0], x0 + [0.0, 0.5]], xatol=0.02, fatol=1e-7, maxiter=60))
    return float(sol.x[0]), float(sol.x[1]), float(sol.fun)
