"""Kahvan vari ja kiven pyoriminen (kierteet hog-hog-valilla).

KAHVA: jokaisessa seurantaruudussa kiven pyorimisakselin ylapaa (X, Y, H_total) projisoidaan kuvaan; sen ympari kuvaan
ympyra, sateena A.KAHVA_SADE_KERROIN x kahvan sade (profiilista) kuvaan projisoituna. Ympyran sisalta kerataan
varikkaiden pikseleiden savyt (S >= A.KAHVA_MIN_S, V >= A.KAHVA_MIN_V; jaa ja graniitti ovat harmaita). Radan
paattyessa kahvan savy = savy, jonka +-A.KAHVA_SAVY_TOL valilla on eniten pikseleita -> <csv>_kahva.csv.

KIERTEET: joka ruudusta kiven akselin suuntainen harmaasavypala kahvasta. Palat kohdistetaan kiven runkoon
(graniitti), piirteena kahvan ikkunan gradientti (Sobel) seka sen peilikuva. Peittyneet ruudut hylataan ajallisen
jatkuvuuden perusteella. Kahvan muoto toistuu PUOLIKIERROKSEN valein ja kahva kulmassa phi nayttaa kulman -phi
peilikuvalta: kaksi riippumatonta mittausta. Pyoriminen hidastuu: omega(t) = omega_loppu + A.KIERRE_HIDASTUVUUS
(t_loppu - t). Loppukierrosaika haetaan laajalta valilta (0,6-40 s); hyvaksynta R^2:n, erottuvuuden ja toiston ja peilin
yhtapitavyyden mukaan. Kierteet = integraali omega dt / 2 pi kaukohogista lahihogiin. Kahva erottuu vasta n. 23 m:sta
-> alkuosa ekstrapoloidaan.
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
# kierrepala: ISO pala 48x48 (+-1,5 rp), josta kohdistuksen jalkeen leikataan 32x32 (+-rp) siirrolla (dy, dx)
_ISO = 48
_REUNA = 8                                  # 32x32-palan vasen ylakulma isossa palassa ilman siirtoa
_HAKU = 4                                   # kohdistuksen siirto +-4 px (isossa palassa)
_RUNKO_R = (16, 30)                         # rungon rivit 32x32-palassa (kiven ylapinnasta alaspain)
_RUNKO_C = (-4, 36)                         # rungon sarakkeet 32x32-palassa
_IKKUNA = (4, 19, 7, 25)                    # kahvan ikkuna 32x32-palassa (rivit, sarakkeet; symmetrinen akselin suhteen)


def kierre_piirre(img, cx, cy, gx, gy, rp):
    """Kiven akselin suuntainen harmaasavypala kahvasta (48x48 uint8) tai None.
    Palan "ylos" = kiven akseli kuvassa (maan keskipisteesta (gx, gy) ylapinnan keskipisteeseen (cx, cy)), joten kahvan
    profiili on palassa aina samassa asennossa kameran kallistuksesta ja kuvan kierrosta riippumatta. Pala kattaa
    +-1,5 rp (rp = kahvan ympyran sade) ylapinnan keskipisteen ympari; kierre_arvio kohdistaa palat kiven runkoon ja
    leikkaa niista +-rp-palat."""
    a = np.array([cx - gx, cy - gy], np.float64)
    na = float(np.hypot(a[0], a[1]))
    if na < 1e-6:
        return None
    a /= na
    l = np.array([-a[1], a[0]])
    k = 1.5 * rp / (_ISO / 2.0)                    # px kuvassa / px palassa
    h = _ISO / 2.0 - 0.5
    M = np.zeros((2, 3), np.float64)
    M[:, 0] = k * l
    M[:, 1] = -k * a
    M[:, 2] = np.array([cx, cy]) - h * k * l + h * k * a
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return cv2.warpAffine(g, cv2.invertAffineTransform(M), (_ISO, _ISO), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def _grad(p):
    g = cv2.GaussianBlur(p.astype(np.float32), (0, 0), 0.7)
    return cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))


def _norm(v):
    v = v.astype(np.float32).ravel()
    v = v - v.mean()
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-6 else None


def _kohdista(G):
    """Palat kohdistetaan kiven runkoon (graniitti; pyorimisesta riippumaton ja harvoin peitossa): rungon
    gradienttimalli = mediaani, siirto = paras normalisoitu korrelaatio +-_HAKU px. Kaksi kierrosta."""
    n = len(G)
    sh = np.zeros((n, 2), int)
    r0, r1 = _REUNA + _RUNKO_R[0], _REUNA + _RUNKO_R[1]
    c0, c1 = _REUNA + _RUNKO_C[0], _REUNA + _RUNKO_C[1]
    gr = np.array([_grad(g) for g in G], np.float32)
    alue = gr[:, r0 - _HAKU:r1 + _HAKU, c0 - _HAKU:c1 + _HAKU]
    ik = np.lib.stride_tricks.sliding_window_view(alue, (r1 - r0, c1 - c0), axis=(1, 2))   # n x 9 x 9 x h x w
    ik = ik - ik.mean(axis=(3, 4), keepdims=True)
    ik_n = np.sqrt(np.einsum("nabhw,nabhw->nab", ik, ik)) + 1e-6
    for _ in range(2):
        tm = np.median(ik[np.arange(n), sh[:, 0] + _HAKU, sh[:, 1] + _HAKU], 0)
        tm = tm - tm.mean()
        res = np.einsum("nabhw,hw->nab", ik, tm) / ik_n
        best = res.reshape(n, -1).argmax(1)
        sh = np.column_stack(np.unravel_index(best, res.shape[1:])) - _HAKU
    return sh


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


def _parit(S, i, k, n):
    """Ruutuparien samankaltaisuus, ruudun oma taso (peitto, etaisyys, valaistus) pois."""
    s = S[i, k].astype(np.float64)
    for _ in range(3):
        cnt = np.maximum(np.bincount(i, None, n) + np.bincount(k, None, n), 1)
        rd = (np.bincount(i, s, n) + np.bincount(k, s, n)) / cnt
        rd -= rd.mean()
        s = s - rd[i] - rd[k]
    return s


def _erottuvuus(sc, j):
    """Paras huippu / paras huippu, kun +-30 % ja kerrannaiset (x2, /2) ovat pois."""
    P = _KIERRE_P
    muu = np.ones(len(P), bool)
    for f_ in (1.0, 2.0, 0.5):
        muu &= ~((P > P[j] * f_ / 1.3) & (P < P[j] * f_ * 1.3))
    return float(sc[j] / max(sc[muu].max() if muu.any() else 0.0, 1e-5))


def kierre_arvio(obs, fps, t_far, t_near, alpha=None, n_pairs=12000, chunk=50):
    """obs: [(ruutu, kierre_piirre-pala)]. Palauttaa dict(kierrokset, kierrosaika_far_s, kierrosaika_near_s, r2,
    erottuvuus, peili_r2, peili_kierrosaika_s, n) tai None."""
    if alpha is None:
        alpha = A.KIERRE_HIDASTUVUUS
    if len(obs) < A.KIERRE_MIN_RUUDUT:
        return None
    T = np.array([o[0] for o in obs], np.float64) / fps
    G = [o[1] for o in obs]
    sh = _kohdista(G)
    r0, r1, c0, c1 = _IKKUNA
    F, Fm = [], []
    for g, (dy, dx) in zip(G, sh):
        p = g[_REUNA + dy:_REUNA + dy + 32, _REUNA + dx:_REUNA + dx + 32]
        F.append(_norm(_grad(p)[r0:r1, c0:c1]))
        Fm.append(_norm(_grad(p[:, ::-1])[r0:r1, 32 - c1:32 - c0]))     # peilikuva kiven akselin suhteen
    ok = np.array([f is not None and m is not None for f, m in zip(F, Fm)])
    T = T[ok]
    F = np.array([f for f, k_ in zip(F, ok) if k_], np.float32)
    Fm = np.array([f for f, k_ in zip(Fm, ok) if k_], np.float32)
    if len(F) < A.KIERRE_MIN_RUUDUT:
        return None
    m = _jatkuvat(F)
    T, F, Fm = T[m], F[m], Fm[m]
    n = len(T)
    if n < A.KIERRE_MIN_RUUDUT:
        return None
    i, k = np.triu_indices(n, 1)
    L = T[k] - T[i]
    m = L > 0.08                                   # kaikki viiveet (pitkat viiveet erottavat lahekkaiset kierrosajat)
    i, k, L = i[m], k[m], L[m]
    s = _parit(F @ F.T, i, k, n)
    Sm = F @ Fm.T
    sp = _parit((Sm + Sm.T) / 2.0, i, k, n)
    if len(s) > n_pairs:
        q = np.random.default_rng(0).choice(len(s), n_pairs, replace=False)
        i, k, L, s, sp = i[q], k[q], L[q], s[q], sp[q]
    if len(s) < 500:
        return None
    Ln = L / L.max()
    Q, _ = np.linalg.qr(np.column_stack([Ln ** p for p in range(4)]))     # hidas viiveriippuvuus pois
    Q = Q.astype(np.float32)
    s = s - Q @ (Q.T @ s)
    Tc = (T[i] + T[k]) / 2.0
    Tc = (Tc - Tc.min()) / (np.ptp(Tc) + 1e-9)
    Qm, _ = np.linalg.qr(np.column_stack([Ln ** p * Tc ** q_ for p in range(4) for q_ in range(4) if p + q_ <= 4]))
    sp = sp - Qm @ (Qm.T @ sp)                     # peilille myos hidas ajallinen muutos pois
    Qm = Qm.astype(np.float32)
    ss, ssp = float(s @ s), float(sp @ sp)
    s32, sp32 = s.astype(np.float32), sp.astype(np.float32)
    if ss <= 0 or ssp <= 0:
        return None
    t_end = T[-1]
    u = t_end - T
    # kiertokulma phi(t) = -w u - alpha u^2 / 2 (omega(t) = w + alpha u, u = t_end - t)
    # identtinen: s = c1 cos(2 dphi) + c2 cos(4 dphi), dphi = phi_i - phi_j (puolikierroksen toisto + yliaalto).
    #   c1 > 0 ja c2 <= c1 vaaditaan: muuten kaksinkertainen kierrosaika sopisi yliaallon kautta.
    # peili: kahva kulmassa phi nayttaa kulman -phi peilikuvalta -> peilisamankaltaisuus = a cos(Sphi) + b sin(Sphi)
    #   + a2 cos(2 Sphi) + b2 sin(2 Sphi), Sphi = phi_i + phi_j. Riippumaton toinen mittaus (lakaisun rytmi ei tuota tata).
    nw = len(_KIERRE_W)
    r2 = np.zeros(nw)
    r2p = np.zeros(nw)
    for a in range(0, nw, chunk):
        w = _KIERRE_W[a:a + chunk]
        ph = (-np.outer(w, u) - alpha * u ** 2 / 2.0).astype(np.float32)          # w x ruudut
        d = ph[:, i] - ph[:, k]
        X1 = np.cos(2.0 * d)
        X2 = 2.0 * X1 * X1 - 1.0                   # cos(4 dphi)
        X1 -= (X1 @ Q) @ Q.T
        X2 -= (X2 @ Q) @ Q.T
        a11 = np.einsum("ij,ij->i", X1, X1).astype(np.float64)
        a22 = np.einsum("ij,ij->i", X2, X2).astype(np.float64)
        a12 = np.einsum("ij,ij->i", X1, X2).astype(np.float64)
        b1, b2 = (X1 @ s32).astype(np.float64), (X2 @ s32).astype(np.float64)
        det = a11 * a22 - a12 ** 2 + 1e-12
        c1 = (a22 * b1 - a12 * b2) / det
        c2 = (a11 * b2 - a12 * b1) / det
        r2[a:a + chunk] = np.where((c1 > 0) & (c2 <= c1), (c1 * b1 + c2 * b2) / ss, 0.0)
        Sg = ph[:, i] + ph[:, k]
        cs, sn = np.cos(Sg), np.sin(Sg)
        Y = np.stack([cs, sn, 2.0 * cs * cs - 1.0, 2.0 * sn * cs], axis=1)          # w x 4 x parit
        Y -= (Y @ Qm) @ Qm.T
        YtY = Y @ Y.transpose(0, 2, 1) + 1e-6 * np.eye(4, dtype=np.float32)[None]
        Ytm = Y @ sp32
        e = np.linalg.solve(YtY.astype(np.float64), Ytm[:, :, None].astype(np.float64))[:, :, 0]
        r2p[a:a + chunk] = np.einsum("wc,wc->w", e, Ytm) / ssp
    j = int(np.argmax(r2))
    erott = _erottuvuus(r2, j)
    jp = int(np.argmax(r2p))
    peili_ok = r2p[jp] >= A.KIERRE_PEILI_MIN_R2
    if peili_ok:
        # peilisignaali on riittava -> sen on osoitettava samaan kierrosaikaan; lopullinen huippu yhteisesta spektrista
        if abs(np.log(_KIERRE_P[jp] / _KIERRE_P[j])) > np.log(1.0 + A.KIERRE_PEILI_TOL):
            return None
        yht = r2 + r2p
        lahella = np.abs(np.log(_KIERRE_P / _KIERRE_P[j])) <= np.log(1.0 + A.KIERRE_PEILI_TOL)
        j = int(np.argmax(np.where(lahella, yht, -1.0)))
    if r2[j] < A.KIERRE_MIN_R2 or (not peili_ok and erott < A.KIERRE_MIN_EROTTUVUUS):
        return None
    w_end = float(_KIERRE_W[j])
    om = lambda t: w_end + alpha * (t_end - t)
    if om(t_near) <= 0:
        return None
    rad = w_end * (t_near - t_far) + alpha * ((t_end - t_far) ** 2 - (t_end - t_near) ** 2) / 2.0
    return dict(kierrokset=rad / (2 * np.pi), kierrosaika_far_s=2 * np.pi / om(t_far),
                kierrosaika_near_s=2 * np.pi / om(t_near), r2=float(r2[j]), erottuvuus=erott,
                peili_r2=float(r2p[jp]), peili_kierrosaika_s=float(_KIERRE_P[jp]), n=n)


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
