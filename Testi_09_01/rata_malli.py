"""Kiven kulkulinjan ennuste kaukohogista pysahdykseen: kytketty naarmu- ja kitkamalli.

Tila: Y, nopeus v, kierre w (rad/s), kulkusuunnan kaantyma phi ja sivusiirtyma H. Kivi liikkuu kohti pienempaa Y:ta.
  pitkittainen  dv/dt = -g <mu(U) u_y / U>  + naarmuseinan voima (ohjausvoima + seinakitka mu_w)
  kierre        dw/dt = -(g r / k^2) <mu(U) (v sin th + w r) / U>  + naarmuvoimien momentti
  sivuttainen   a_sivu = naarmuseinan ohjausvoima;  dphi/dt = a_sivu / v,  dH/dt = v phi
  <.> = keskiarvo running bandin kehalla, U = pisteen nopeus jaan suhteen, mu(U) = A + B ln U (Coulombin kitka
  jokaisessa pisteessa omalla nopeudellaan).
Naarmuseina: kiven etureuna tekee naarmuja, takareunan nyppylat ylittavat ne kulmassa alpha (sin alpha ~ 2 beta |cos th|,
beta = w r / v) ja painuvat uran seinaan nopeudella v_n = v sin alpha. Voima f = x^p exp(-x^(m p)), x = v_n/vc (p, m: asetukset.py
osio 13): x^p = seinan vastus ennen murtumaa (p = 1: viskoosi kontakti), exp(-x^(m p)) = Weibullin todennakoisyys
sille, etta seina ei murru (murtunut seina ei ohjaa, voima haviaa). Ohjausvoima on kohtisuoraan
takapisteen nopeutta (ei tee tyota), seinakitka mu_w * f takapisteen liiketta vastaan. Naarmutiheys w(th) =
min(1, c0 D(r sin th)), D = running bandin (leveys ENNUSTE_KAISTA_MM) naarmujen tiheys sivusuunnassa.
Naarmuvoimien keskiarvot (Fx, Fy, momentti) lasketaan kerran taulukoksi (beta, v/vc) -> naarmu_taulukko.npz.

Heittokohtainen sovitus hog-valin datasta (Levenberg-Marquardt, numeerinen Jacobiaani rinnakkain numpy-erana):
y0, v0, ln w0, A, ln kappa; X(t) = c0 + c1 S(t) + s H(t) (c0, c1 suoraan pienimmilla neliöilla, s = kaarron suunta);
kierteet hog-hog (kahvasta) pakotteena. Yhteiset B, mu_w, vc: asetukset.py osio 13.
"""
import os
import threading

import numpy as np

import asetukset as A

G = 9.81
_SK = 0.01                                           # kappa-skaala: naarmukiihtyvyys = kappa * g * 0.01 * taulukko
_Z = np.linspace(0.0, 0.8, 81)                       # z = beta / (1 + beta)  ->  beta 0..4
_LU = np.log(np.geomspace(0.02, 200.0, 61))          # ln(v / vc)
_TAULUKKO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "naarmu_taulukko.npz")
_lukko = threading.Lock()
_T = None


# ------------------------------------------------------------------
# NAARMUTAULUKKO
# ------------------------------------------------------------------
def _avain():
    return np.array([A.ENNUSTE_C0, A.ENNUSTE_KAISTA_MM, A.ENNUSTE_R_M, len(_Z), len(_LU), A.ENNUSTE_LAKI_P, A.ENNUSTE_LAKI_M], float)


def _laske_taulukko():
    """T[k, i, j], k = 0..2: ohjausvoima (Fx, Fy, M/r), 3..5: seinakitka per mu_w = 1; per (kappa m g), keskiarvo
    koko kehalla (takapuolisko th = 90..270 astetta, etupuoliskolla ei naarmuvoimaa)."""
    th = np.linspace(np.pi / 2, 3 * np.pi / 2, 4001)[1:-1]
    sn, cs = np.sin(th), np.cos(th)
    r = A.ENNUSTE_R_M * 100.0; b = A.ENNUSTE_KAISTA_MM / 10.0
    xs = np.linspace(-r - b / 2, r + b / 2, 4001); dens = np.zeros_like(xs)
    for ri in np.linspace(r - b / 2, r + b / 2, 41):
        m = np.abs(xs) < ri
        dens[m] += 1.0 / np.sqrt(ri ** 2 - xs[m] ** 2)
    dens /= dens[len(xs) // 2]
    w = np.minimum(1.0, A.ENNUSTE_C0 * np.interp(r * sn, xs, dens))
    kn = len(th) / np.pi * 2 * np.pi                 # summa -> keskiarvo koko kehalla
    T = np.zeros((6, len(_Z), len(_LU)))
    u = np.exp(_LU)[:, None]
    for i, z in enumerate(_Z):
        beta = max(z / (1 - z), 1e-5)
        def e(c):
            ux, uy = -beta * c, 1 + beta * sn; n = np.hypot(ux, uy); return ux / n, uy / n
        ebx, eby = e(cs); efx, efy = e(-cs)          # takapisteen liikesuunta, naarmun suunta (etupisteen jalki)
        al = np.arcsin(np.clip(np.abs(efx * eby - efy * ebx), 0, 1))
        d = efx * ebx + efy * eby; px, py = efx - d * ebx, efy - d * eby
        pn = np.maximum(np.hypot(px, py), 1e-15); px, py = px / pn, py / pn
        x = u * np.sin(al)[None, :]
        p = A.ENNUSTE_LAKI_P
        F = x ** p * np.exp(-x ** (A.ENNUSTE_LAKI_M * p)) * w[None, :]       # (u, th)
        for k0, (fx, fy) in ((0, (px, py)), (3, (-ebx, -eby))):
            T[k0, i] = (F * fx).sum(1) / kn
            T[k0 + 1, i] = (F * fy).sum(1) / kn
            T[k0 + 2, i] = (F * (sn * fy - cs * fx)).sum(1) / kn
    return T


def taulukko():
    global _T
    with _lukko:
        if _T is None:
            if os.path.exists(_TAULUKKO):
                z = np.load(_TAULUKKO)
                if z["avain"].shape == _avain().shape and np.allclose(z["avain"], _avain()):
                    _T = z["T"]
            if _T is None:
                _T = _laske_taulukko()
                try:
                    np.savez(_TAULUKKO, T=_T, avain=_avain())
                except OSError:
                    pass
    return _T


# ------------------------------------------------------------------
# SIMULOINTI (numpy-era: n parametrivektoria rinnakkain)
# ------------------------------------------------------------------
def _derivaatat(v, w, Aa, kap, T, rengas):
    B, R, K2 = A.ENNUSTE_MU_B, A.ENNUSTE_R_M, A.ENNUSTE_K2_M2
    sn, cs = rengas
    ux = -w[:, None] * R * cs[None, :]
    uy = v[:, None] + w[:, None] * R * sn[None, :]
    U = np.maximum(np.hypot(ux, uy), 1e-4)
    mu = np.maximum(Aa[:, None] + B * np.log(U), 0.0)
    ay = -G * np.mean(mu * uy / U, axis=1)
    aw = -G * R / K2 * np.mean(mu * (v[:, None] * sn[None, :] + w[:, None] * R) / U, axis=1)
    vv = np.maximum(v, 1e-3)
    be = w * R / vv
    fz = be / (1 + be) / (_Z[1] - _Z[0])
    i = np.clip(fz.astype(int), 0, len(_Z) - 2); fz = np.clip(fz - i, 0.0, 1.0)
    fu = (np.log(vv / A.ENNUSTE_VC_MS) - _LU[0]) / (_LU[1] - _LU[0])
    j = np.clip(np.floor(fu).astype(int), 0, len(_LU) - 2); fu = np.clip(fu - j, 0.0, 1.0)
    t = (T[:, i, j] * (1 - fz) * (1 - fu) + T[:, i + 1, j] * fz * (1 - fu)
         + T[:, i, j + 1] * (1 - fz) * fu + T[:, i + 1, j + 1] * fz * fu)                       # (6, n)
    s = np.where(w > 0, kap * G * _SK, 0.0)
    mw = A.ENNUSTE_MU_W
    a_sivu = -s * (t[0] + mw * t[3])
    ay = ay + s * (t[1] + mw * t[4])
    aw = aw + s * R / K2 * (t[2] + mw * t[5])
    return ay, aw, a_sivu


# ------------------------------------------------------------------
# NUMBA (jos asennettu, A.ENNUSTE_NUMBA): sama simulointi kaannettyna, ~50x nopeampi. Muuten numpy-era alla.
# ------------------------------------------------------------------
_nb_sim = None
if A.ENNUSTE_NUMBA:
    try:
        import numba as _nb

        @_nb.njit(cache=True, nogil=True)
        def _nb_derivaatat(v, w, a_, kap, T, Z, LU, sn, cs, B, R, K2, vc, mw):
            ay = 0.0; aw = 0.0; nr = len(sn)
            for k in range(nr):
                ux = -w * R * cs[k]; uy = v + w * R * sn[k]
                U = max(np.sqrt(ux * ux + uy * uy), 1e-4)
                mu = max(a_ + B * np.log(U), 0.0)
                ay -= mu * uy / U; aw -= mu * (v * sn[k] + w * R) / U
            ay *= G / nr; aw *= G * R / K2 / nr
            asv = 0.0
            if w > 0:
                vv = max(v, 1e-3); be = w * R / vv
                fz = be / (1 + be) / (Z[1] - Z[0]); i = min(max(int(fz), 0), len(Z) - 2); fz = min(max(fz - i, 0.0), 1.0)
                fu = (np.log(vv / vc) - LU[0]) / (LU[1] - LU[0]); j = min(max(int(np.floor(fu)), 0), len(LU) - 2)
                fu = min(max(fu - j, 0.0), 1.0)
                t = np.empty(6)
                for q in range(6):
                    t[q] = (T[q, i, j] * (1 - fz) * (1 - fu) + T[q, i + 1, j] * fz * (1 - fu)
                            + T[q, i, j + 1] * (1 - fz) * fu + T[q, i + 1, j + 1] * fz * fu)
                s = kap * G * 0.01
                asv = -s * (t[0] + mw * t[3]); ay += s * (t[1] + mw * t[4]); aw += s * R / K2 * (t[2] + mw * t[5])
            return ay, aw, asv

        @_nb.njit(cache=True, nogil=True)
        def _nb_sim(P, kmax, pysahdykseen, dt, T, Z, LU, sn, cs, B, R, K2, vc, mw):
            n = P.shape[0]
            out = np.empty((kmax + 1, 5, n)); kmax_k = 0
            for m in range(n):
                y = P[m, 0]; v = max(P[m, 1], 1e-3); w = max(P[m, 2], 0.0); a_ = P[m, 3]; kap = P[m, 4]
                ph = 0.0; S = 0.0; H = 0.0; seis = False
                out[0, 0, m] = y; out[0, 1, m] = v; out[0, 2, m] = w; out[0, 3, m] = 0.0; out[0, 4, m] = 0.0
                k = 0
                while k < kmax:
                    if seis:
                        if pysahdykseen:
                            break
                        k += 1
                        out[k, 0, m] = y; out[k, 1, m] = 0.0; out[k, 2, m] = 0.0; out[k, 3, m] = S; out[k, 4, m] = H
                        continue
                    a1, b1, c1 = _nb_derivaatat(v, w, a_, kap, T, Z, LU, sn, cs, B, R, K2, vc, mw)
                    v2 = max(v + dt * a1, 1e-3); w2 = max(w + dt * b1, 0.0)
                    a2, b2, c2 = _nb_derivaatat(v2, w2, a_, kap, T, Z, LU, sn, cs, B, R, K2, vc, mw)
                    vn = v + dt * (a1 + a2) / 2; wn = max(w + dt * (b1 + b2) / 2, 0.0)
                    if vn <= 1e-3:
                        seis = True; vn = 0.0; wn = 0.0
                    phn = ph + dt * (c1 / max(v, 1e-3) + c2 / max(v2, 1e-3)) / 2
                    ds = 100 * dt * (v + vn) / 2
                    y -= ds; S += ds; H += 100 * dt * (v * ph + vn * phn) / 2
                    v = vn; w = wn; ph = phn; k += 1
                    out[k, 0, m] = y; out[k, 1, m] = v; out[k, 2, m] = w; out[k, 3, m] = S; out[k, 4, m] = H
                kmax_k = max(kmax_k, k)
                for kk in range(k + 1, kmax + 1):          # pysahtynyt aiemmin kuin muut: paikallaan loppuun
                    for c in range(5):
                        out[kk, c, m] = out[k, c, m]
            return out[:kmax_k + 1]
    except Exception as _e:                                # numba puuttuu tai kaannos epaonnistui -> numpy
        print(f"[ennuste] numba ei kaytossa ({_e!r}), kaytetaan numpya")
        _nb_sim = None


def simuloi(P, t_alku, t_loppu, pysahdykseen=False):
    """P: (n, 5) [y0_cm, v0_ms, w0_rads, A, kappa] hetkella t_alku. Palauttaa (t, Y) jossa Y (k, 5, n):
    [Y_cm, v_ms, w_rads, S_cm (kuljettu matka), H_cm (sivusiirtyma kaartosuuntaan)]. pysahdykseen: kunnes kaikki v = 0."""
    global _nb_sim
    T = taulukko()
    P = np.atleast_2d(np.asarray(P, float))
    n = len(P); dt = A.ENNUSTE_DT_S
    th = (np.arange(A.ENNUSTE_RENGAS_PISTEET) + 0.5) * 2 * np.pi / A.ENNUSTE_RENGAS_PISTEET
    rengas = (np.sin(th), np.cos(th))
    kmax = int(np.ceil(((120.0 if pysahdykseen else t_loppu) - t_alku) / dt)) + 1
    if _nb_sim is not None:
        try:
            Y = _nb_sim(np.ascontiguousarray(P), kmax, bool(pysahdykseen), float(dt), np.ascontiguousarray(T), _Z, _LU,
                        rengas[0], rengas[1], float(A.ENNUSTE_MU_B), float(A.ENNUSTE_R_M), float(A.ENNUSTE_K2_M2),
                        float(A.ENNUSTE_VC_MS), float(A.ENNUSTE_MU_W))
            return t_alku + dt * np.arange(len(Y)), Y
        except Exception as e_:
            print(f"[ennuste] numba-simulointi epaonnistui ({e_!r}), kaytetaan numpya")
            _nb_sim = None
    y, v, w = P[:, 0].copy(), np.maximum(P[:, 1], 1e-3), np.maximum(P[:, 2], 0.0)
    Aa, kap = P[:, 3], P[:, 4]
    ph = np.zeros(n); S = np.zeros(n); H = np.zeros(n); seis = np.zeros(n, bool)
    out = [np.stack([y, v, w, S, H])]
    for _ in range(kmax):
        a1, b1, c1 = _derivaatat(v, w, Aa, kap, T, rengas)
        v2 = np.maximum(v + dt * a1, 1e-3); w2 = np.maximum(w + dt * b1, 0.0)
        a2, b2, c2 = _derivaatat(v2, w2, Aa, kap, T, rengas)
        vn = v + dt * (a1 + a2) / 2
        wn = np.maximum(w + dt * (b1 + b2) / 2, 0.0)
        uusi_seis = seis | (vn <= 1e-3)
        vn = np.where(uusi_seis, 0.0, vn); wn = np.where(uusi_seis, 0.0, wn)
        phn = np.where(seis, ph, ph + dt * (c1 / np.maximum(v, 1e-3) + c2 / np.maximum(v2, 1e-3)) / 2)
        ds = 100 * dt * (v + vn) / 2 * ~seis
        y = y - ds; S = S + ds; H = H + 100 * dt * (v * ph + vn * phn) / 2 * ~seis
        v, w, ph, seis = vn, wn, phn, uusi_seis
        out.append(np.stack([y, v, w, S, H]))
        if pysahdykseen and seis.all():
            break
    Y = np.array(out)
    return t_alku + dt * np.arange(len(Y)), Y


# ------------------------------------------------------------------
# SOVITUS
# ------------------------------------------------------------------
def _rob(r, f=2.0):
    z = r / f
    return f * np.sign(z) * np.sqrt(2 * (np.sqrt(1 + z * z) - 1))


_ALA = np.array([-1e9, 0.3, np.log(0.05), 0.0005, np.log(0.01)])
_YLA = np.array([1e9, 4.5, np.log(30.0), 0.03, np.log(100.0)])


def _p(q):
    q = np.atleast_2d(q)
    return np.column_stack([q[:, 0], q[:, 1], np.exp(q[:, 2]), q[:, 3], np.exp(q[:, 4])])


def _jaannokset(D, Q, palauta=False):
    """Q: (n, 5) sovitusparametrit -> jaannokset (n, m) (Y, X robustit + kierre)."""
    t, Y = simuloi(_p(Q), D["ts"], D["te"])
    dt = A.ENNUSTE_DT_S
    k = np.clip((D["t"] - D["ts"]) / dt, 0, len(t) - 1.001); i0 = k.astype(int); f = (k - i0)[:, None]
    at = lambda c: Y[i0, c] * (1 - f) + Y[i0 + 1, c] * f                           # (m, n)
    ry = (at(0) - D["y"][:, None]) / 1.5
    S, Hh = at(3), at(4)
    wx = D["wx"][:, None]
    bx = (D["x"][:, None] - D["s"] * Hh)
    # painotettu suora c0 + c1 S jokaiselle erajasenelle
    sw, sS, sSS = wx.sum(0), (wx * S).sum(0), (wx * S * S).sum(0)
    sb, sSb = (wx * bx).sum(0), (wx * S * bx).sum(0)
    det = sw * sSS - sS ** 2
    c1 = (sw * sSb - sS * sb) / det; c0 = (sb - c1 * sS) / sw
    rx = np.sqrt(wx) * (c0 + c1 * S - bx) / 1.0
    rot = np.empty(len(Q))
    cum = np.concatenate([np.zeros((1, len(Q))), np.cumsum((Y[1:, 2] + Y[:-1, 2]) / 2 * dt, axis=0)])
    for n in range(len(Q)):
        yy = -Y[:, 0, n]
        tf = np.interp(-D["yf"], yy, t); tn = np.interp(-D["yn"], yy, t)
        rot[n] = (np.interp(tn, t, cum[:, n]) - np.interp(tf, t, cum[:, n])) / (2 * np.pi)
    rr = (rot - D["kierteet"]) / D.get("sigma", A.ENNUSTE_KIERRE_SIGMA)
    R = np.vstack([_rob(ry), _rob(rx), rr[None, :]]).T
    if palauta:
        return R, dict(c0=c0, c1=c1, ry=ry * 1.5, rx=(c0 + c1 * S - bx), rot=rot)
    return R


def _lm(D, q, iters=60):
    """Levenberg-Marquardt: Jacobiaani (5 haulikkoa) ja viisi vaimennusta kokeillaan kumpikin yhtena numpy-erana."""
    h = np.array([0.2, 0.001, 0.005, 1e-5, 0.005])
    lam = 1e-3
    r = _jaannokset(D, q[None])[0]; c = r @ r
    for _ in range(iters):
        J = ((_jaannokset(D, q + np.diag(h)) - r) / h[:, None]).T
        JJ = J.T @ J; g = J.T @ r; dg = np.diag(np.diag(JJ) + 1e-12)
        ls = [lam * f for f in (0.01, 0.1, 1.0, 10.0, 100.0)]
        kok = []
        for l in ls:
            try:
                kok.append(np.clip(q - np.linalg.solve(JJ + l * dg, g), _ALA, _YLA))
            except np.linalg.LinAlgError:
                kok.append(q.copy())
        Rk = _jaannokset(D, np.array(kok))
        ck = (Rk ** 2).sum(1); j = int(np.argmin(ck))
        if ck[j] >= c:
            lam *= 1000
            if lam > 1e6:
                break
            continue
        muutos = c - ck[j]
        lam, q, r, c = ls[j], kok[j], Rk[j], ck[j]
        if muutos < 1e-5 * c:
            break
    return q, c


def ennusta(res, rows):
    """Kulkulinja hog-analyysin tuloksesta (res: analyze_hog + kierteet) ja radan riveista (frame, t, dict).
    Palauttaa dict(ok=True, polku [(X, Y) cm] kaukohogista pysahdykseen, pysahtyy_X_cm, pysahtyy_Y_cm, ...),
    dict(ok=False, syy) jos sovitus on huono, tai None jos lahtotiedot eivat riita (ei kierteita / hog-analyysia)."""
    kt = res.get("kierteet_yhd")
    if kt is None:                                       # ei yhdistelmaa (vanha tulos): jaksohaun arvo
        kt = res.get("kierteet_taysi", res.get("kierteet"))
    sigma = A.ENNUSTE_KIERRE_SIGMA_EPAVARMA if res.get("kierre_varmuus") == "epavarma" else A.ENNUSTE_KIERRE_SIGMA
    if not res.get("ok") or kt is None or res.get("t_near_hog_s") is None or not np.isfinite(kt) or kt <= 0:
        return None
    t0 = float(res["t0"]); ylo, yhi = res["y_lo_cm"], res["y_hi_cm"]
    pts = [(float(t) - t0, float(r["Y_cm"]), float(r["X_cm"]), r.get("rms_px")) for _, t, r in rows
           if r.get("Y_cm") is not None and r.get("X_cm") is not None and ylo <= float(r["Y_cm"]) <= yhi]
    if len(pts) < A.HOG_MIN_PISTEET:
        return None
    t = np.array([p[0] for p in pts]); y = np.array([p[1] for p in pts]); x = np.array([p[2] for p in pts])
    rms = np.array([np.nan if p[3] is None else float(p[3]) for p in pts])
    wx = 1.0 / (1.0 + (np.nan_to_num(rms, nan=A.HOG_X_PAINO_RMS_PX) / A.HOG_X_PAINO_RMS_PX) ** 2)
    t_far, t_near = float(res["t_far_hog_s"]) - t0, float(res["t_near_hog_s"]) - t0
    yf, yn = float(res["far_hog_cm"]), float(res["near_hog_cm"])
    ts = min(t_far, float(t.min())) - 0.1
    D = dict(t=t, y=y, x=x, wx=wx, s=np.sign(res.get("kurvi_k_ms2") or 1.0) or 1.0, yf=yf, yn=yn, ts=ts,
             te=float(t.max()) + 0.1, kierteet=float(kt), sigma=sigma)
    v0 = float(res["v_far_hog_ms"])
    y0 = yf + 100.0 * v0 * (t_far - ts)
    a0 = float(res["mu_a"]) + (A.HOG_MU_B - A.ENNUSTE_MU_B) * np.log(1.5)
    w0 = 2 * np.pi * float(kt) / max(t_near - t_far, 1.0)
    # alkuarvo: kappa-haku yhtena erana, sitten LM parhaasta
    kg = np.array([0.3, 0.6, 1.0, 1.8, 3.0, 5.0, 8.0])
    Q0 = np.array([[y0, v0, np.log(w0), max(a0 - 0.0005, 0.001), np.log(k)] for k in kg])
    c0 = (_jaannokset(D, Q0) ** 2).sum(1)
    q, c = _lm(D, Q0[int(np.argmin(c0))])
    R, info = _jaannokset(D, q[None], palauta=True)
    p = _p(q)[0]
    y_rms = float(np.sqrt(np.mean(info["ry"][:, 0] ** 2))); x_rms = float(np.sqrt(np.mean(info["rx"][:, 0] ** 2)))
    if y_rms > A.ENNUSTE_MAX_Y_RMS_CM or x_rms > A.ENNUSTE_MAX_X_RMS_CM:
        return dict(ok=False, syy=f"sovitus ei riittavan hyva (Y rms {y_rms:.1f} cm, X rms {x_rms:.1f} cm)",
                    y_rms_cm=y_rms, x_rms_cm=x_rms)
    tp, Yp = simuloi(p[None], ts, 0.0, pysahdykseen=True)
    Yp = Yp[:, :, 0]
    X = info["c0"][0] + info["c1"][0] * Yp[:, 3] + D["s"] * Yp[:, 4]
    alku = int(np.searchsorted(-Yp[:, 0], -yf))                 # kaukohogista eteenpain
    S = Yp[alku:, 3]; idx = [alku]
    for k in range(alku + 1, len(Yp)):
        if Yp[k, 3] - Yp[idx[-1], 3] >= A.ENNUSTE_PISTE_VALI_CM:
            idx.append(k)
    if idx[-1] != len(Yp) - 1:
        idx.append(len(Yp) - 1)
    return dict(ok=True, polku=[(float(X[k]), float(Yp[k, 0])) for k in idx],
                pysahtyy_X_cm=float(X[-1]), pysahtyy_Y_cm=float(Yp[-1, 0]), liukuaika_s=float(tp[-1] - t_far),
                A=float(p[3]), kappa=float(p[4]), w_far_rads=float(np.interp(t_far, tp, Yp[:, 2])),
                y_rms_cm=y_rms, x_rms_cm=x_rms,
                kierre_ero=float(info["rot"][0] - kt))
