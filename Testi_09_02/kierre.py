"""Kahvan vari ja kiven pyoriminen (kierteet hog-hog-valilla).

KAHVA: jokaisessa seurantaruudussa kiven pyorimisakselin ylapaa (X, Y, H_total) projisoidaan kuvaan; sen ympari kuvaan
ympyra, sateena A.KAHVA_SADE_KERROIN x kahvan sade (profiilista) kuvaan projisoituna. Ympyran sisalta kerataan
varikkaiden pikseleiden savyt (S >= A.KAHVA_MIN_S, V >= A.KAHVA_MIN_V; jaa ja graniitti ovat harmaita). Radan
paattyessa kahvan savy = savy, jonka +-A.KAHVA_SAVY_TOL valilla on eniten pikseleita -> <csv>_kahva.csv.

KIERTEET: joka ruudusta kiven akselin suuntainen harmaasavypala kahvasta. Palat kohdistetaan kiven runkoon
(graniitti), piirteena kahvan ikkunan gradientti (Sobel) seka sen peilikuva. Peittyneet ruudut hylataan ajallisen
jatkuvuuden perusteella. Kahvan muoto toistuu PUOLIKIERROKSEN valein ja kahva kulmassa phi nayttaa kulman -phi
peilikuvalta: kaksi riippumatonta mittausta. Pyoriminen hidastuu: omega(t) = omega_loppu + alpha (t_loppu - t); hidastuvuus
alpha haetaan kahvasignaalista (A.KIERRE_ALPHA_HAKU; karkea haku + tarkennus, suurin selitysaste) tai on kiintea
A.KIERRE_HIDASTUVUUS. Loppukierrosaika haetaan laajalta valilta (0,6-40 s); hyvaksynta R^2:n, erottuvuuden ja toiston ja peilin
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


def _kierre_valmistele(obs, fps, n_pairs):
    """Piirteet, ruutuparit ja hitaat trendit (yhteiset kaikille hidastuvuuksille). Palauttaa dict tai None."""
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
    if ss <= 0 or ssp <= 0:
        return None
    return dict(n=n, i=i, k=k, Q=Q, Qm=Qm, s32=s.astype(np.float32), sp32=sp.astype(np.float32), ss=ss, ssp=ssp,
                t_end=T[-1], u=T[-1] - T)


# NUMBA (A.KIERRE_NUMBA): hidastuvuuden haun toistomallin selitysaste kaannettyna, rinnakkain loppukierrosaikojen yli,
# yhdella lapikaynnilla parien yli (projektiot Q:lle kertyvat summina, ei w x parit -matriiseja). Lopullinen spektri
# (myos peili) lasketaan kerran numpylla. Puuttuu tai kaannos epaonnistuu -> numpy-versio.
_nb_r2 = None
if A.KIERRE_NUMBA:
    try:
        import numba as _nb

        @_nb.njit(cache=True, nogil=True, fastmath=True, parallel=True)
        def _nb_r2(W, alpha, du, du2, s, Q, ss):
            """vain toistomallin selitysaste (hidastuvuuden haku); sama kuin _kierre_spektri:n r2."""
            nw, npar, nq = len(W), len(du), Q.shape[1]
            r2 = np.zeros(nw)
            for a in _nb.prange(nw):
                w = W[a]
                a11 = a22 = a12 = b1 = b2 = 0.0
                q1 = np.zeros(nq); q2 = np.zeros(nq)
                for p in range(npar):
                    x1 = np.cos(2.0 * (-w * du[p] - alpha * du2[p]))
                    x2 = 2.0 * x1 * x1 - 1.0
                    a11 += x1 * x1; a22 += x2 * x2; a12 += x1 * x2
                    b1 += x1 * s[p]; b2 += x2 * s[p]
                    for c in range(nq):
                        q1[c] += x1 * Q[p, c]; q2[c] += x2 * Q[p, c]
                for c in range(nq):
                    a11 -= q1[c] * q1[c]; a22 -= q2[c] * q2[c]; a12 -= q1[c] * q2[c]
                det = a11 * a22 - a12 * a12 + 1e-12
                c1 = (a22 * b1 - a12 * b2) / det
                c2 = (a11 * b2 - a12 * b1) / det
                r2[a] = (c1 * b1 + c2 * b2) / ss if (c1 > 0 and c2 <= c1) else 0.0
            return r2
    except Exception as _e:
        _nb_r2 = None
        print(f"[kierre] numba ei kaytossa ({_e!r}), kaytetaan numpya")


def _nb_valmistele(P):
    """numba-ytimen taulukot (kerran): parikohtainen vaihe-ero phi_i - phi_j = -w du - alpha du2."""
    if "du" in P:
        return
    u, i, k = P["u"], P["i"], P["k"]
    P["du"] = (u[i] - u[k]).astype(np.float64); P["du2"] = ((u[i] ** 2 - u[k] ** 2) / 2.0).astype(np.float64)
    P["Q64"] = np.ascontiguousarray(P["Q"], np.float64); P["s64"] = P["s32"].astype(np.float64)
    P["W64"] = _KIERRE_W.astype(np.float64)


def _kierre_r2(P, alpha, chunk=50):
    """Vain toistomallin selitysaste (hidastuvuuden haku): numba-ydin tai koko spektrin r2."""
    if _nb_r2 is not None:
        _nb_valmistele(P)
        try:
            return _nb_r2(P["W64"], float(alpha), P["du"], P["du2"], P["s64"], P["Q64"], P["ss"])
        except Exception as e_:
            print(f"[kierre] numba-r2 epaonnistui ({e_!r}), kaytetaan numpya")
    return _kierre_spektri(P, alpha, chunk)[0]


def _kierre_spektri(P, alpha, chunk=50):
    """Selitysasteet kaikille loppukierrosajoille annetulla hidastuvuudella: (r2 identtinen, r2 peili)."""
    i, k, Q, Qm, s32, sp32, u = P["i"], P["k"], P["Q"], P["Qm"], P["s32"], P["sp32"], P["u"]
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
        r2[a:a + chunk] = np.where((c1 > 0) & (c2 <= c1), (c1 * b1 + c2 * b2) / P["ss"], 0.0)
        Sg = ph[:, i] + ph[:, k]
        cs, sn = np.cos(Sg), np.sin(Sg)
        Y = np.stack([cs, sn, 2.0 * cs * cs - 1.0, 2.0 * sn * cs], axis=1)          # w x 4 x parit
        Y -= (Y @ Qm) @ Qm.T
        YtY = Y @ Y.transpose(0, 2, 1) + 1e-6 * np.eye(4, dtype=np.float32)[None]
        Ytm = Y @ sp32
        e = np.linalg.solve(YtY.astype(np.float64), Ytm[:, :, None].astype(np.float64))[:, :, 0]
        r2p[a:a + chunk] = np.einsum("wc,wc->w", e, Ytm) / P["ssp"]
    return r2, r2p


def _kierre_hidastuvuus(P, chunk):
    """Hidastuvuus kahvasignaalista: karkea haku (A.KIERRE_ALPHA_KARKEA) ja tarkennus parhaan ymparilla
    (A.KIERRE_ALPHA_ASKEL). Kriteeri: suurin toistomallin selitysaste (hyvaksynta vasta lopulliselle; peilimalli
    lasketaan vain lopulliselle hidastuvuudelle)."""
    a0, a1 = A.KIERRE_ALPHA_MIN, A.KIERRE_ALPHA_MAX
    tulos = {}
    def arvo(al):
        al = round(float(al), 6)
        if al not in tulos:
            tulos[al] = float(_kierre_r2(P, al, chunk).max())
        return tulos[al]
    karkea = np.arange(a0, a1 + 1e-9, A.KIERRE_ALPHA_KARKEA)
    paras = max(karkea, key=arvo)
    hieno = np.arange(max(a0, paras - A.KIERRE_ALPHA_KARKEA), min(a1, paras + A.KIERRE_ALPHA_KARKEA) + 1e-9,
                      A.KIERRE_ALPHA_ASKEL)
    paras = round(float(max(hieno, key=arvo)), 6)
    return paras, _kierre_spektri(P, paras, chunk)


def kierre_arvio(obs, fps, t_far, t_near, alpha=None, n_pairs=12000, chunk=50):
    """obs: [(ruutu, kierre_piirre-pala)]. Palauttaa dict(kierrokset, kierrosaika_far_s, kierrosaika_near_s, r2,
    erottuvuus, peili_r2, peili_kierrosaika_s, alpha, n) tai None. alpha (hidastuvuus rad/s^2): annettu -> kaytetaan;
    muuten haetaan kahvasignaalista (A.KIERRE_ALPHA_HAKU) tai kiintea A.KIERRE_HIDASTUVUUS."""
    P = _kierre_valmistele(obs, fps, n_pairs)
    if P is None:
        return None
    if alpha is None and A.KIERRE_ALPHA_HAKU:
        alpha, (r2, r2p) = _kierre_hidastuvuus(P, chunk)
    else:
        if alpha is None:
            alpha = A.KIERRE_HIDASTUVUUS
        r2, r2p = _kierre_spektri(P, alpha, chunk)
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
    t_end = P["t_end"]
    w_end = float(_KIERRE_W[j])
    om = lambda t: w_end + alpha * (t_end - t)
    if om(t_near) <= 0:
        return None
    rad = w_end * (t_near - t_far) + alpha * ((t_end - t_far) ** 2 - (t_end - t_near) ** 2) / 2.0
    return dict(kierrokset=rad / (2 * np.pi), kierrosaika_far_s=2 * np.pi / om(t_far),
                kierrosaika_near_s=2 * np.pi / om(t_near), r2=float(r2[j]), erottuvuus=erott,
                peili_r2=float(r2p[jp]), peili_kierrosaika_s=float(_KIERRE_P[jp]), alpha=float(alpha), n=P["n"])


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


# ------------------------------------------------------------------
# TOINEN ARVIO: KAHVAN SILUETIN LEVEYS (riippumaton jaksohausta)
# Kahvan siluetti kiven kuvun ylapuolella on leveimmillaan kun kahva on sivuttain ja kapeimmillaan kun se osoittaa
# kameraa kohti / poispain -> leveys vaihtelee kerran PUOLIKIERROKSESSA. Leveys w(t) jokaisesta kierre_piirre-palasta
# (tausta rivi kerrallaan kahvan sivuilta: lakaisija kiven takana), hidas trendi pois robustilla 3. asteen polynomilla.
# Malli: w ~ a cos 2 theta + b sin 2 theta, omega lineaarinen kaukohogista lahihogiin, omega_lahi / omega_kauko in
# [0,45, 1,05] (kierre hidastuu, ei kiihdy). Haku (omega_kauko, suhde) Huber-painotetuilla pienimmilla neliöilla.
# Erottuvuus = paras selitysaste / paras kun keskiomega +-25 % ja x2, /2 pois. Testi (22 kivea): yhtapitava nykyisen
# arvion kanssa 0-6 % kaikilla kivilla joille nykyinen arvio saatiin.
# ------------------------------------------------------------------
_LEV_WF = np.geomspace(0.2, 4.5, 260)
_LEV_RHO = np.linspace(0.45, 1.05, 13)


def _leveys(g, r0=6, r1=22, kynnys=10.0):
    """kahvan siluetin sivuttaisleveys (px) 48x48-palassa kiven kuvun ylapuolella; tausta rivi kerrallaan sivuilta."""
    p = g.astype(np.float32)
    bg = np.median(np.c_[p[:, 2:9], p[:, 39:46]], axis=1)[:, None]
    d = np.abs(cv2.GaussianBlur(p, (0, 0), 0.6) - bg)[r0:r1, 6:42]
    w = np.clip((d - kynnys) / 20.0, 0, 1)
    n = float(w.sum())
    if n < 2:
        return 0.0
    xs = np.arange(w.shape[1], dtype=np.float32)[None, :]
    mx = float((w * xs).sum() / n)
    return float(np.sqrt((w * (xs - mx) ** 2).sum() / n))


def kierre_leveys(obs, fps, t_far, t_near):
    """obs: [(ruutu, kierre_piirre-pala)]. Palauttaa dict(kierrokset, omega_far, omega_near, r2, erottuvuus, n) tai None."""
    if t_near is None or len(obs) < A.KIERRE_MIN_RUUDUT:
        return None
    T = np.array([o[0] for o in obs], np.float64) / fps
    w = np.array([_leveys(o[1]) for o in obs])
    m = np.isfinite(w) & (T >= t_far - 0.3) & (T <= t_near + 1.5)
    T, w = T[m], w[m]
    if len(T) < A.KIERRE_MIN_RUUDUT:
        return None
    tt = (T - T.mean()) / max(T.std(), 1e-6)
    h = np.ones(len(T), bool)
    for _ in range(5):
        co = np.polyfit(tt[h], w[h], 3)
        r = w - np.polyval(co, tt)
        h = np.abs(r) < 3.5 * (1.4826 * np.median(np.abs(r[h])) + 1e-3)
    x = w - np.polyval(co, tt)
    x = x / (1.4826 * np.median(np.abs(x)) + 1e-6)
    D = t_near - t_far
    tau = T - t_far
    S = np.zeros((len(_LEV_RHO), len(_LEV_WF)))
    for i, rho in enumerate(_LEV_RHO):
        th = 2.0 * np.outer(tau + (rho - 1.0) * tau ** 2 / (2.0 * D), _LEV_WF)          # (n, nWF)
        c, s = np.cos(th), np.sin(th)
        W = np.ones_like(th)
        for _ in range(3):                                                             # IRLS (Huber, c = 1,5)
            scc, sss, scs = (W * c * c).sum(0), (W * s * s).sum(0), (W * c * s).sum(0)
            bc, bs = (W * c * x[:, None]).sum(0), (W * s * x[:, None]).sum(0)
            det = np.maximum(scc * sss - scs ** 2, 1e-9)
            a_ = (bc * sss - bs * scs) / det
            b_ = (bs * scc - bc * scs) / det
            r = x[:, None] - a_ * c - b_ * s
            W = 1.0 / np.maximum(1.0, np.abs(r) / 1.5)
        S[i] = 1.0 - (W * r ** 2).sum(0) / (W * x[:, None] ** 2).sum(0)
    i, j = np.unravel_index(np.argmax(S), S.shape)
    wmean = _LEV_WF[None, :] * (1.0 + _LEV_RHO[:, None]) / 2.0
    wb = wmean[i, j]
    muu = np.ones_like(S, bool)
    for f_ in (1.0, 2.0, 0.5):
        muu &= ~((wmean > wb * f_ / 1.25) & (wmean < wb * f_ * 1.25))
    erott = float(S[i, j] / max(S[muu].max(), 1e-3))
    wf, rho = float(_LEV_WF[j]), float(_LEV_RHO[i])
    return dict(kierrokset=wf * (1.0 + rho) / 2.0 * D / (2.0 * np.pi), omega_far=wf, omega_near=wf * rho,
                r2=float(S[i, j]), erottuvuus=erott, n=int(len(T)))


def yhdista_kierteet(k_nyk, lev):
    """Nykyinen (jaksohaku) + siluetin leveys -> (kierteet, varmuus). varma: yhtapitavat (A.KIERRE_YHTAPITAVYYS) ja leveyden
    erottuvuus >= A.KIERRE_LEVEYS_MIN_EROTT; melko varma: yhtapitavat; epavarma: eri mielta (nykyinen kaytetaan) tai vain toinen
    saatiin (leveys yksin vain jos erottuvuus >= A.KIERRE_LEVEYS_YKSIN_EROTT; koko matkan lakaistuilla kivilla se ei toimi)."""
    k_lev = None if lev is None else lev["kierrokset"]
    if k_nyk is not None and k_lev is not None and abs(np.log(k_nyk / k_lev)) < np.log(1.0 + A.KIERRE_YHTAPITAVYYS):
        luokka = "varma" if lev["erottuvuus"] >= A.KIERRE_LEVEYS_MIN_EROTT else "melko varma"
        return 0.5 * (k_nyk + k_lev), luokka
    if k_nyk is not None:
        return k_nyk, "epavarma"
    if k_lev is not None and lev["erottuvuus"] >= A.KIERRE_LEVEYS_YKSIN_EROTT:
        return k_lev, "epavarma"
    return None, None
