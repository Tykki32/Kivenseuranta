"""Kiven vakiot (kitkataso A ja kaarto kappa) kaukopaan radasta: rata_malli.py:n kytketty naarmu- ja kitkamalli sovitetaan kaukopaassa seurattuun rataan (hog-viivalta pysahtymiseen).

Malli (rata_malli.py, vakiot asetukset.py osio 13): kiven rata kaukopaassa riippuu alkutilasta (paikka Y', nopeus v0, kierre w0) seka kiven omista vakioista
  A      = kitkataso (mu(U) = (A + B ln Ub) (U / Ub)^NL): maaraa pysahtymismatkan (hidastuvuuden),
  kappa  = naarmuvoiman voimakkuus: maaraa kaarron (kiven kulkulinjan sivusiirtyman).
Kaukopaan seuranta antaa radan (aika, X', Y') pysahtymiseen asti, joten molemmat vakiot ovat mitattavissa: A pysahtymisprofiilista (Y'(t)), kappa sivusiirtymasta (X'(t)).
Sovitus: y0, v0, ln w0, A, ln kappa; X'(t) = c0 + c1 S(t) + s H(t) (c0, c1 pienimmilla neliöilla; S = kuljettu matka, H = sivusiirtyma kaarron suuntaan, s = kaarron suunta X':ssa),
kierteen mittaus kaukopaasta (omega) pakotteena. Huom: harjaajat pienentavat kitkaa ja kaarroa (A ja kappa ovat tehollisia arvoja harjauksen kanssa); puhtaat kiven vakiot
saadaan vain harjaamattomista heitoista (ei tunnisteta).

sovita(...) palauttaa dict(ok=True, A, A_sd, kappa, kappa_sd, kappa_ok, y0, v0, w0, rms_y_cm, rms_x_cm, n, ennuste=dict(pysahtyy_X_cm, pysahtyy_Y_cm, ...)) tai dict(ok=False, syy=...).
ennusta(A, kappa, y0, v0, w0, kaarto_merkki) laskee kulkulinjan ja pysahtymispaikan kiven vakioilla (kuinka vakioita kaytetaan, kun lasketaan minne kivi kulkee).
"""
import numpy as np

import asetukset as A
import rata_malli as rm

ALA = np.array([-1e9, 0.3, np.log(0.05), 0.0005, np.log(0.01)])
YLA = np.array([1e9, 4.5, np.log(30.0), 0.03, np.log(100.0)])
SIGMA_Y_CM = 1.5                 # Y-jaannoksen mittakaava (robusti)
SIGMA_OMEGA_OLETUS = 0.15        # rad/s: kierron mittauksen oletusepavarmuus (~9 astetta/s)
MIN_PISTEITA = 40                # vahintaan nain monta paikkaa sovitukseen
MIN_MATKA_CM = 150.0             # ... ja rata kattaa vahintaan taman matkan (Y'-vaihtelu)
MIN_KESTO_S = 3.0                # ... ja vahintaan taman ajan (hidastuvuus ei ole mitattavissa nopeasta, lyhyesta radasta: A karkaa ylarajalle)
MAX_RMS_Y_CM = 6.0               # sovituksen laatu: yli -> ok = False
MAX_RMS_X_CM = 3.0
V0_MAX_MS = 1.6                  # mallin kelvollisuus: hog-viivan nopeus enintaan tama (nopeampi poistoheitto osuu muihin kiviin eika kitkamalli (0,5-1,3 m/s) ole voimassa)
OMEGA_OLETUS = (0.95, 0.45)      # kierre (rad/s, sigma) kaukopaan radan alussa, kun mittausta ei ole (tyypillinen mitattu 50-60 astetta/s): vain A:n sovitusta varten; kappa ei silloin ole luotettava
MAX_A_SUHT_VIRHE = 0.2           # A:n keskivirhe suhteessa A:han enintaan tama (muuten A ei maaraydy radasta)
MAX_LNK_VIRHE = 0.4              # ln kappan keskivirhe enintaan tama (kappa +-40 %)
A_RAJA = (0.0015, 0.025)         # uskottava kitkataso: rajojen ulkopuolella (tai aivan parametrirajalla) sovitus ei ole luotettava
KAPPA_ALKU = (0.3, 0.6, 1.0, 1.8, 3.0, 5.0, 8.0)


def _rob(r, f=2.0):
    return rm._rob(r, f)


def _p(q):
    return rm._p(q)


def _jaannokset(D, Q, palauta=False):
    """Q: (n, 5) sovitusparametrit [y0, v0, ln w0, A, ln kappa] -> jaannokset (n, m): Y- ja X-robustit sekä kierrepakote."""
    t, Y = rm.simuloi(_p(Q), D["ts"], D["te"])
    dt = A.ENNUSTE_DT_S
    k = np.clip((D["t"] - D["ts"]) / dt, 0, len(t) - 1.001)
    i0 = k.astype(int)
    f = (k - i0)[:, None]
    at = lambda c: Y[i0, c] * (1 - f) + Y[i0 + 1, c] * f                       # (m, n)
    ry = (at(0) - D["y"][:, None]) / SIGMA_Y_CM
    S, Hh = at(3), at(4)
    wx = D["wx"][:, None]
    bx = D["x"][:, None] - D["s"] * Hh
    sw, sS, sSS = wx.sum(0), (wx * S).sum(0), (wx * S * S).sum(0)
    sb, sSb = (wx * bx).sum(0), (wx * S * bx).sum(0)
    det = np.maximum(sw * sSS - sS ** 2, 1e-9)
    c1 = (sw * sSb - sS * sb) / det
    c0 = (sb - c1 * sS) / sw
    rx = np.sqrt(wx) * (c0 + c1 * S - bx)
    cols = [_rob(ry), _rob(rx)]
    for tw, wm, sg in D["omega"]:                                              # kierre: mallin omega hetkella tw ~ mitattu
        kw = np.clip((tw - D["ts"]) / dt, 0, len(t) - 1.001)
        j0 = int(kw)
        fw = kw - j0
        wmod = Y[j0, 2] * (1 - fw) + Y[j0 + 1, 2] * fw                          # (n,)
        cols.append(((wmod - wm) / sg)[None, :])
    R = np.vstack(cols).T
    if palauta:
        return R, dict(c0=c0, c1=c1, ry=ry * SIGMA_Y_CM, rx=(c0 + c1 * S - bx), S=S, H=Hh)
    return R


def _lm(D, q, iters=60):
    """Levenberg-Marquardt (kuten rata_malli._lm): numeerinen Jacobiaani ja viisi vaimennusta kokeillaan kukin yhtena numpy-erana."""
    h = np.array([0.2, 0.001, 0.005, 1e-5, 0.005])
    lam = 1e-3
    r = _jaannokset(D, q[None])[0]
    c = r @ r
    JJ = None
    for _ in range(iters):
        J = ((_jaannokset(D, q + np.diag(h)) - r) / h[:, None]).T
        JJ = J.T @ J
        g = J.T @ r
        dg = np.diag(np.diag(JJ) + 1e-12)
        ls = [lam * f for f in (0.01, 0.1, 1.0, 10.0, 100.0)]
        kok = []
        for l in ls:
            try:
                kok.append(np.clip(q - np.linalg.solve(JJ + l * dg, g), ALA, YLA))
            except np.linalg.LinAlgError:
                kok.append(q.copy())
        Rk = _jaannokset(D, np.array(kok))
        ck = (Rk ** 2).sum(1)
        j = int(np.argmin(ck))
        if ck[j] >= c:
            lam *= 1000
            if lam > 1e6:
                break
            continue
        muutos = c - ck[j]
        lam, q, r, c = ls[j], kok[j], Rk[j], ck[j]
        if muutos < 1e-5 * c:
            break
    J = ((_jaannokset(D, q + np.diag(h)) - r) / h[:, None]).T                       # Jacobiaani ratkaisussa: parametrien epavarmuus
    return q, c, J.T @ J


def kaarron_suunta(t, y, x):
    """Kaarron suunta X':ssa (+1 / -1) radasta: kivi kulkee kohti pienempaa Y':ta, sivusiirtyma H(S) on kupera (H'' > 0), joten X'(Y') = ... + c Y'^2 + ... ja c:n merkki on kaarron
    suunta (s = sign c). Vastapaivaan pyorivan kiven kaarto on kiven vasemmalle puolelle = +X' kaukopaassa (kulkusuunta -Y')."""
    if len(y) < 8:
        return 0
    c = np.polyfit(y, x, 2)[0]
    return 0 if abs(c) < 1e-8 else (1 if c > 0 else -1)


def sovita(t, X, Y, vahva=None, omega=None, kaarto_merkki=None, t_loppu=None, pysahtynyt=True):
    """Kaukopaan rata (t s, X' cm, Y' cm; vahva = B-paikka) -> kiven vakiot. omega = (t_keski, omega_rad_s (> 0), sigma) kierteen mittaus (valinnainen); kaarto_merkki
    = +1 / -1 (kaarto kohti +X' / -X'; muuten haetaan radasta). t_loppu: sovita vain tahan asti (ennustekoe). Palauttaa dict."""
    t = np.asarray(t, float)
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    vahva = np.ones(len(t), bool) if vahva is None else np.asarray(vahva, bool)
    m = np.ones(len(t), bool) if t_loppu is None else (t <= t_loppu)
    t, X, Y, vahva = t[m], X[m], Y[m], vahva[m]
    if len(t) < MIN_PISTEITA or float(Y.max() - Y.min()) < MIN_MATKA_CM or float(t[-1] - t[0]) < MIN_KESTO_S:
        return dict(ok=False, syy=f"rata liian lyhyt ({len(t)} paikkaa, {float(Y.max() - Y.min()) if len(t) else 0:.0f} cm, {float(t[-1] - t[0]) if len(t) else 0:.1f} s)")
    s = kaarto_merkki if kaarto_merkki else kaarron_suunta(t, Y, X)
    if not s:
        s = 1
    w = np.where(vahva, 1.0, 0.25)
    t0 = float(t[0])
    om_sov = omega if omega is not None else (float(t[len(t) // 4]), OMEGA_OLETUS[0], OMEGA_OLETUS[1])      # ilman mittausta heikko oletus (kierre ei saa karata mielivaltaiseksi)
    D = dict(t=t - t0, y=Y, x=X, wx=w, s=float(s), ts=-0.1, te=float(t[-1] - t0) + 0.1, omega=[(float(om_sov[0] - t0), float(om_sov[1]), float(om_sov[2]))])
    # alkuarvot: nopeus radan alusta, kierre mittauksesta (tai 1 rad/s), kappa-haku yhtena erana
    n0 = min(len(t), 12)
    v0 = float(np.clip(-np.polyfit(t[:n0], Y[:n0], 1)[0] / 100.0, 0.3, 4.0))
    w0 = float(om_sov[1]) * 1.05
    y0 = float(Y[0]) + 100.0 * v0 * 0.1
    Q0 = np.array([[y0, v0, np.log(max(w0, 0.1)), 0.005, np.log(k)] for k in KAPPA_ALKU])
    c0 = (_jaannokset(D, Q0) ** 2).sum(1)
    q, c, JJ = _lm(D, Q0[int(np.argmin(c0))])
    R, info = _jaannokset(D, q[None], palauta=True)
    p = _p(q)[0]
    rms_y = float(np.sqrt(np.mean(info["ry"][:, 0] ** 2)))
    rms_x = float(np.sqrt(np.mean(info["rx"][:, 0] ** 2)))
    tp, Yp = rm.simuloi(p[None], -0.1, 0.0, pysahdykseen=True)
    Yp = Yp[:, :, 0]
    Xp = info["c0"][0] + info["c1"][0] * Yp[:, 3] + s * Yp[:, 4]
    ennuste = dict(pysahtyy_X_cm=float(Xp[-1]), pysahtyy_Y_cm=float(Yp[-1, 0]), pysahtyy_t=float(tp[-1] + t0), matka_cm=float(Yp[-1, 3]))
    try:                                                                            # parametrien keskivirheet (kovarianssi ~ (J^T J)^-1; jaannokset ovat suunnilleen yksikkovarianssisia): A:n suhteellinen virhe, ln kappan virhe
        cov = np.linalg.inv(JJ + 1e-9 * np.eye(5))
        A_sd = float(np.sqrt(max(cov[3, 3], 0.0)))
        lnk_sd = float(np.sqrt(max(cov[4, 4], 0.0)))
    except np.linalg.LinAlgError:
        A_sd = lnk_sd = float("inf")
    a_ok = bool(A_RAJA[0] <= p[3] <= A_RAJA[1] and A_sd <= MAX_A_SUHT_VIRHE * p[3])
    v_ok = bool(float(p[1]) <= V0_MAX_MS)
    ok = bool(rms_y <= MAX_RMS_Y_CM and rms_x <= MAX_RMS_X_CM and a_ok and v_ok)
    # kappa on tunnistettavissa vain, kun kierre on rajattu mittauksella (muuten kappa ja w0 korvaavat toisensa)
    kappa_ok = bool(ok and omega is not None and np.exp(YLA[4]) * 0.9 > p[4] > np.exp(ALA[4]) * 1.1 and lnk_sd <= MAX_LNK_VIRHE)
    syy = None if ok else (f"A ei maaraydy (A {p[3]:.4f} +- {A_sd:.4f})" if not a_ok else (f"nopeus {p[1]:.2f} m/s yli {V0_MAX_MS} m/s (poisto / tormays)" if not v_ok else f"sovitus huono (Y rms {rms_y:.1f} cm, X rms {rms_x:.1f} cm)"))
    return dict(ok=ok, syy=syy, A=float(p[3]), A_sd=A_sd, kappa=float(p[4]), kappa_sd=float(p[4] * lnk_sd), kappa_ok=kappa_ok, y0=float(p[0]), v0=float(p[1]), w0=float(p[2]), kaarto_merkki=int(s),
                rms_y_cm=rms_y, rms_x_cm=rms_x, n=int(len(t)), kesto_s=float(t[-1] - t[0]), kustannus=float(c), ennuste=ennuste, t_alku=t0)


def ennusta(A, kappa, y0_cm, v0_ms, w0_rads, kaarto_merkki, x0_cm=0.0, vali_cm=25.0):
    """Kulkulinja kiven vakioilla: kivi lahtee kaukopaan kuvassa paikasta (x0, y0') nopeudella v0 (m/s, kohti pesaa) ja kierteella w0 (rad/s, > 0); A ja kappa ovat kiven vakiot (sovita;
    kivikohtainen mediaani Analyysi.kivien_vakiot / <csv>_kivet.csv), kaarto_merkki = +1 / -1 (kaarto kohti +X' / -X': vastapaivaan pyorivalla +1). Palauttaa dict(polku=[(X', Y')] vali_cm:n
    valein, pysahtyy_X_cm, pysahtyy_Y_cm, kesto_s, matka_cm). X'(t) = x0 + s H(t) (sivusiirtyma kaarron suuntaan; pieni suuntakulma jatetaan huomiotta)."""
    P = np.array([[float(y0_cm), float(v0_ms), float(max(w0_rads, 0.0)), float(A), float(kappa)]])
    t, Y = rm.simuloi(P, 0.0, 0.0, pysahdykseen=True)
    Y = Y[:, :, 0]
    X = float(x0_cm) + float(kaarto_merkki) * Y[:, 4]
    idx = [0]
    for k in range(1, len(Y)):
        if Y[k, 3] - Y[idx[-1], 3] >= vali_cm:
            idx.append(k)
    if idx[-1] != len(Y) - 1:
        idx.append(len(Y) - 1)
    return dict(polku=[(float(X[k]), float(Y[k, 0])) for k in idx], pysahtyy_X_cm=float(X[-1]), pysahtyy_Y_cm=float(Y[-1, 0]), kesto_s=float(t[-1]), matka_cm=float(Y[-1, 3]))
