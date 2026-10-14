"""Kiven liu'un ja irroituksen tunnistus radasta: python liuku_irroitus.py VIDEO tulokset vasen [--ulos tulokset/liuku_irroitus_vasen] [--ei-ruutuja]

Tunnistus perustuu pelkkaan kiven rataan (tulokset/radat_<puoli>.csv: ruutu, todellinen aika t_kor_s, X, Y) -- ei kahvaan eika kateen:
  * LIUKU (alku): kivi lahtee hackista (lepo -> liike), tyonto kiihdyttaa sen liukunopeuteen (90 % tasanteesta).
  * IRROITUS: irroituksen jalkeen kivi on vapaa kappale, eli rata on tasainen 2. asteen kayra ilman teravia muutoksia: X(Y) paraabeli (kaarto,
    "curl") JA Y(t) 2. asteen polynomi (vakiohidastuvuus). Ennen irroitusta heittaja tyontaa ja ohjaa kivea: kiihtyvyys (Y(t)) ja kaarevuus (X(Y))
    ovat toiset. Irroitus = kohta, jossa molemmat vaihtuvat SILEASTI (paikka JA nopeus/suunta jatkuvat, vain 2. derivaatta hyppaa): kahden
    paraabelin C1-sovitus (robusti, Huber), vaihtokohta ruudukkohaulla, X- ja Y-kanavan yhteinen vaihtokohta (khi^2-summa), 2-sigma-vali profiilista,
    tilastollinen merkitsevyys yhta paraabelia vastaan; sovitus toistetaan eri alkukohdilla (vakaus).
  * IRROITUSTILA: sijainti, suunta (kulma radan akseliin, + = oikealle heittajasta katsoen), nopeus, hidastuvuus ja kaarevuus vapaan lennon sovituksesta.
Aikaperusta: ks. aikaperusta.py (lahdevirran ruutuindeksi; pelkka k / 25 ei kelpaa, koska lahdevirrasta puuttuu ruutuja).

Kirjoittaa <ulos>.json, <ulos>.png ja (jos VIDEO) <ulos>_ruudut.jpg (kontaktiarkki irroituksen ymparilta, tunnistettu ruutu merkitty)."""
import argparse
import csv
import json
import os

import numpy as np

# ---------------------------------------------------------------------------------------------------------------------
# PARAMETRIT
# ---------------------------------------------------------------------------------------------------------------------
V_LEPO_CM_S = 15.0           # lepo: nopeus alle tama
LIUKU_OSUUS = 0.9            # tyonto paattyy kun nopeus >= 0,9 * tasanne
MARGINAALIT_CM = (50, 100, 150, 200, 250, 300)     # kierros 0: ohjatun vaiheen alkukohta = tyonnon loppu + marginaali
TARKENNUS_L_CM = (150, 200, 250, 300)              # tarkennuskierrokset: ohjatun vaiheen ikkuna alkaa L cm ennen edellista vaihtokohtaa
TARKENNUS_MAX = 4            # tarkennuskierrosten enimmaismaara (lopetus kun vaihtokohta muuttuu < 10 cm)
MIN_ENNEN_CM = 100.0         # ohjatussa vaiheessa vahintaan tama matka vaihtokohdan edella
MIN_JALKEEN_CM = 150.0       # vapaata lentoa vahintaan tama (irroitus ei voi olla aivan radan lopussa)
HUBER_C = 1.5                # Huberin kynnys (sigmaa)
VIIVE_S = 0.12               # irroitustilan sovitus alkaa tama kauemmin vaihtokohdan jalkeen
KOHINA_HAL_CM = 300.0        # kohinan arvio: radan viimeiset 300 cm (vapaa lento)
RAJA_DCHI2 = 40.0            # merkitseva vaihtokohta: khi^2-parannus vahintaan tama (nollahypoteesin suurin arvo simuloinnissa 35, vapaan lennon ikkunoissa 33)


# ---------------------------------------------------------------------------------------------------------------------
# AINEISTO
# ---------------------------------------------------------------------------------------------------------------------
def lataa_radat(kansio, puoli):
    """{rata: dict(f, t, X, Y)}: ruutunumero, todellinen aika [s] (aikaperusta.py), X, Y [cm] kiven raaka paikka."""
    out = {}
    p = os.path.join(kansio, f"radat_{puoli}.csv")
    rows = list(csv.DictReader(open(p)))
    if "t_kor_s" not in rows[0]:
        raise SystemExit(f"{p}: sarake t_kor_s puuttuu -- aja ensin 'python aja_kivi.py VIDEO --puoli {puoli} --ulos {kansio} --ketjuta-vain'")
    for r in rows:
        d = out.setdefault(int(r["rata"]), {"f": [], "t": [], "X": [], "Y": []})
        d["f"].append(int(r["ruutu"]))
        d["t"].append(float(r["t_kor_s"]))
        d["X"].append(float(r["X_cm"]))
        d["Y"].append(float(r["Y_cm"]))
    return {k: {n: np.array(a) for n, a in d.items()} for k, d in out.items()}


def liukuva(t, y, h=4, aste=2):
    """Liukuva polynomisovitus +-h naytetta (reunoilla katkaistu): palauttaa taulukon (y, dy/dt, d2y/dt2)."""
    n = len(t)
    out = np.zeros((n, 3))
    for i in range(n):
        a, b = max(0, i - h), min(n, i + h + 1)
        c = np.polyfit(t[a:b] - t[i], y[a:b], min(aste, b - a - 1))
        c = np.concatenate([np.zeros(3 - len(c)), c])
        out[i] = (c[-1], c[-2], 2.0 * c[-3])
    return out


# ---------------------------------------------------------------------------------------------------------------------
# LIUKU
# ---------------------------------------------------------------------------------------------------------------------
def liuku_alku(t, Y):
    """Lahtohetki ja tyonnon loppu: t_90 = ensimmainen hetki kun v >= 0,9 * tasanne (tasanne = v:n mediaani Y = -50...300), t_lahto = viimeinen hetki
    ennen t_90:aa jolloin v < 0,15 m/s (lepo). Jos rata alkaa jo liikkeesta (kivi nakyy vasta myohemmin), lahto ARVIOIDAAN taaksepain
    ekstrapoloimalla (vakiokiihtyvyys ensimmaisista naytteista)."""
    v = liukuva(t, Y, h=3, aste=1)[:, 1]
    pl = float(np.median(v[(Y > -50) & (Y < 300)]))
    i90 = next(i for i in range(len(t)) if v[i] >= LIUKU_OSUUS * pl and Y[i] > -260)
    i0 = i90
    while i0 > 0 and v[i0] >= V_LEPO_CM_S:
        i0 -= 1
    havaittu = bool(v[i0] < V_LEPO_CM_S)
    t0, Y0 = float(t[i0]), float(Y[i0])
    if not havaittu:
        n = min(10, i90 + 1)
        c = np.polyfit(t[:n] - t[0], Y[:n], 2)                 # Y = c0 tau^2 + c1 tau + c2
        a = 2.0 * c[0]
        if a > 0 and c[1] > 0:
            tau0 = -c[1] / a                                   # v = 0
            t0, Y0 = float(t[0] + tau0), float(np.polyval(c, tau0))
    return dict(i_lahto=int(i0), t_lahto=t0, Y_lahto=Y0, lahto_havaittu=havaittu, i_90=int(i90), t_90=float(t[i90]), Y_90=float(Y[i90]),
                v_tasanne_m_s=pl / 100.0)


# ---------------------------------------------------------------------------------------------------------------------
# IRROITUS: ROBUSTI KAHDEN PARAABELIN C1-SOVITUS
# ---------------------------------------------------------------------------------------------------------------------
def _kustannus(r, s):
    """Huber-kustannus khi^2-yksikoissa (pienilla jaannoksilla sum (r/s)^2)."""
    a = np.abs(r) / s
    return float(np.sum(2.0 * np.where(a <= HUBER_C, 0.5 * a * a, HUBER_C * a - 0.5 * HUBER_C ** 2)))


def _irls(A, y, s, kierroksia=6):
    w = np.ones(len(y))
    for _ in range(kierroksia):
        sw = np.sqrt(w)
        p = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)[0]
        a = np.abs(y - A @ p) / (HUBER_C * s)
        w = np.where(a <= 1.0, 1.0, 1.0 / np.maximum(a, 1e-9))
    return p, _kustannus(y - A @ p, s)


def _c0(u, y, s):
    d = u - u.mean()
    return _irls(np.column_stack([np.ones_like(d), d, 0.5 * d ** 2]), y, s)


def _c1(u, y, ur, s):
    """y = y_r + v_r (u - ur) + 0,5 c1 (u - ur)^2 (u < ur) / c2 (u >= ur): arvo ja 1. derivaatta jatkuvia, 2. derivaatta hyppaa."""
    d = u - ur
    return _irls(np.column_stack([np.ones_like(d), d, 0.5 * d ** 2 * (d < 0), 0.5 * d ** 2 * (d >= 0)]), y, s)


def kohinat(t, X, Y):
    """Mittauskohina (robusti MAD) radan viimeisilta KOHINA_HAL_CM sentilta (vapaa lento): (sigma_Y [cm], sigma_X [cm])."""
    m = Y >= Y.max() - KOHINA_HAL_CM
    tt, yy, xx = t[m], Y[m], X[m]
    ry = yy - np.polyval(np.polyfit(tt - tt.mean(), yy, 2), tt - tt.mean())
    rx = xx - np.polyval(np.polyfit(yy - yy.mean(), xx, 2), yy - yy.mean())
    mad = lambda r: 1.4826 * float(np.median(np.abs(r - np.median(r))))
    return max(mad(ry), 0.3), max(mad(rx), 0.06)


def irroitus_ikkuna(t, X, Y, Y_a, sY, sX, askel_s=0.02):
    """Vaihtokohdan haku alueella Y >= Y_a. Palauttaa dict tai None. Vaihtokohta t_r on yhteinen: Y(t) C1-sovitus antaa paikan Y_r(t_r), johon X(Y):n
    C1-sovituksen vaihtokohta asetetaan; kustannus = Y- ja X-kanavan khi^2:n summa."""
    m = Y >= Y_a
    tt, yy, xx = t[m], Y[m], X[m]
    if len(tt) < 25:
        return None
    k0y = _c0(tt, yy, sY)[1]
    k0x = _c0(yy, xx, sX)[1]
    prof = []
    for tr in np.arange(tt[0], tt[-1], askel_s):
        Yr = float(np.interp(tr, tt, yy))
        if Yr < Y_a + MIN_ENNEN_CM or Yr > yy[-1] - MIN_JALKEEN_CM or (tt < tr).sum() < 8 or (tt >= tr).sum() < 10:
            continue
        py, cy = _c1(tt, yy, tr, sY)
        px, cx = _c1(yy, xx, py[0], sX)
        prof.append((tr, float(py[0]), cy, cx, py, px))
    if not prof:
        return None
    P = np.array([[a, b, c, d] for a, b, c, d, _, _ in prof])
    out = dict(Y_a=float(Y_a), n=int(len(tt)), k0y=k0y, k0x=k0x)
    for nimi, kol, k0 in (("yht", None, k0y + k0x), ("y", 2, k0y), ("x", 3, k0x)):
        s = P[:, 2] + P[:, 3] if kol is None else P[:, kol]
        i = int(np.argmin(s))
        ok = P[s <= s[i] + 4.0, 1]
        out[nimi] = dict(t_r=float(P[i, 0]), Y_r=float(P[i, 1]), Y_lo=float(ok.min()), Y_hi=float(ok.max()), dchi2=float(k0 - s[i]))
    i = int(np.argmin(P[:, 2] + P[:, 3]))
    py, px = prof[i][4], prof[i][5]
    out["a1_m_s2"], out["a2_m_s2"] = float(py[2] / 100.0), float(py[3] / 100.0)
    out["c1_1_cm"], out["c2_1_cm"] = float(px[2]), float(px[3])
    out["profiili"] = P[:, [1, 2, 3]].tolist()           # (Y_r, khi2_Y, khi2_X)
    return out


def _jasenet(t, X, Y, alut, sY, sX):
    out = []
    for Y_a in alut:
        r = irroitus_ikkuna(t, X, Y, Y_a, sY, sX)
        if r is not None:
            out.append(r)
    return out


def irroitus_varmistettu(t, X, Y, Y_90, sY, sX):
    """Vaihtokohta usealla ohjatun vaiheen ikkunalla. Kierros 0: ikkuna alkaa tyonnon lopusta + marginaali (laaja, karkea). Tarkennuskierrokset: ikkuna
    alkaa L cm ENNEN edellista vaihtokohtaa (ohjattu vaihe ei ole 2. asteen kayra koko matkaa; paikallinen ikkuna on lahempana mallia). Palauttaa dict:
    mediaani, vaihteluvali, jasenet, merkitsevyys (enemmistolla jasenista khi^2-parannus >= RAJA_DCHI2)."""
    jasenet = _jasenet(t, X, Y, [Y_90 + m for m in MARGINAALIT_CM], sY, sX)
    if not jasenet:
        return None
    Y_r = float(np.median([r["yht"]["Y_r"] for r in jasenet]))
    historia = [Y_r]
    for _ in range(TARKENNUS_MAX):
        uudet = _jasenet(t, X, Y, [max(Y_90 + 50.0, Y_r - L) for L in TARKENNUS_L_CM], sY, sX)
        if not uudet:
            break
        jasenet = uudet
        uusi = float(np.median([r["yht"]["Y_r"] for r in jasenet]))
        historia.append(uusi)
        muutos = abs(uusi - Y_r)
        Y_r = uusi
        if muutos < 10.0:
            break
    mer = [r for r in jasenet if r["yht"]["dchi2"] >= RAJA_DCHI2]
    ytm = lambda ks: float(np.median([r[ks]["Y_r"] for r in jasenet]))
    return dict(jasenet=[{k: v for k, v in r.items() if k != "profiili"} for r in jasenet], profiilit=[r["profiili"] for r in jasenet],
                historia=historia, Y_r=Y_r, Y_min=float(min(r["yht"]["Y_r"] for r in jasenet)), Y_max=float(max(r["yht"]["Y_r"] for r in jasenet)),
                Y_r_Y_kanava=ytm("y"), Y_r_X_kanava=ytm("x"), dchi2_med=float(np.median([r["yht"]["dchi2"] for r in jasenet])),
                merkitsevia=len(mer), jasenia=len(jasenet), merkitseva=bool(len(mer) * 2 > len(jasenet)))


# ---------------------------------------------------------------------------------------------------------------------
# IRROITUSTILA
# ---------------------------------------------------------------------------------------------------------------------
def _paraabeli_arvo(u, y, u0):
    """Paraabelisovitus (Huber) -> (arvo, 1. derivaatta, 2. derivaatta, virhe arvo, virhe derivaatta) kohdassa u0; virheet LS-kovarianssista."""
    d = u - u0
    A = np.column_stack([np.ones_like(d), d, 0.5 * d ** 2])
    r0 = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
    s = max(1.4826 * float(np.median(np.abs(r0 - np.median(r0)))), 1e-3)
    p, _ = _irls(A, y, s)
    r = y - A @ p
    cov = np.linalg.inv(A.T @ A) * max(float(np.sum(r ** 2)) / max(len(y) - 3, 1), 1e-12)
    return float(p[0]), float(p[1]), float(p[2]), float(np.sqrt(cov[0, 0])), float(np.sqrt(cov[1, 1]))


def irroitustila(t, X, Y, t_r, Y_vali=None):
    """Vapaan lennon sovitus (t >= t_r + VIIVE_S): X(Y) ja Y(t) paraabelit. Tila hetkella t_r: paikka, suunta, nopeus, hidastuvuus, kaarevuus.
    Suunnan virhe = LS-virhe, vaihtokohdan epavarmuudesta (kaarevuus * puoli Y-valia) ja radan paikallisen suunnan (ikkuna 120 cm) hajonta paraabelin
    ympari (taustaviivojen ja logojen aiheuttama sivuttaishairio) neliosumma."""
    m = t >= t_r + VIIVE_S
    tt, x, y = t[m], X[m], Y[m]
    Yr, vY, aY, _, _ = _paraabeli_arvo(tt, y, t_r)
    Xr, dXdY, cX, _, edXdY = _paraabeli_arvo(y, x, Yr)
    nopeus = vY * np.sqrt(1.0 + dXdY ** 2)
    suunta = float(np.degrees(np.arctan(dXdY)))
    e_stat = float(np.degrees(edXdY / (1.0 + dXdY ** 2)))
    e_vali = float(np.degrees(abs(cX) * 0.5 * (Y_vali[1] - Y_vali[0]) / (1.0 + dXdY ** 2))) if Y_vali else 0.0
    yc, th = suunta_paikallinen(y, x)
    e_loc = float(np.sqrt(np.mean((th - np.degrees(np.arctan(dXdY + cX * (yc - Yr)))) ** 2))) if len(yc) >= 3 else 0.0
    return dict(t_r=float(t_r), X_cm=Xr, Y_cm=Yr, suunta_deg=suunta, suunta_virhe_deg=float(np.sqrt(e_stat ** 2 + e_vali ** 2 + e_loc ** 2)),
                suunta_virhe_osat_deg=dict(sovitus=e_stat, vaihtokohta=e_vali, paikallinen_hajonta=e_loc), nopeus_m_s=float(nopeus / 100.0),
                hidastuvuus_m_s2=float(aY / 100.0), kaarevuus_1_cm=float(cX),
                kaarto=("suora (|kaarevuus| < 0,1e-4 /cm)" if abs(cX) < 1e-5 else ("vasemmalle" if cX < 0 else "oikealle")), ruutuja=int(m.sum()))


def suunta_paikallinen(Y, X, ikkuna_cm=120.0, askel_cm=20.0):
    """Suuntakulma (deg) Y:n funktiona: lineaarinen X(Y) liukuvassa ikkunassa. Palauttaa (Y_keski, kulma)."""
    ys, th = [], []
    for yc in np.arange(Y.min() + ikkuna_cm / 2, Y.max() - ikkuna_cm / 2, askel_cm):
        m = (Y >= yc - ikkuna_cm / 2) & (Y <= yc + ikkuna_cm / 2)
        if m.sum() >= 6:
            ys.append(yc)
            th.append(np.degrees(np.arctan(np.polyfit(Y[m], X[m], 1)[0])))
    return np.array(ys), np.array(th)


def analysoi(rata):
    """Yhden heiton koko tunnistus: dict."""
    t, X, Y = rata["t"], rata["X"], rata["Y"]
    a = liuku_alku(t, Y)
    sY, sX = kohinat(t, X, Y)
    ir = irroitus_varmistettu(t, X, Y, a["Y_90"], sY, sX)
    out = dict(liuku=a, kohina=dict(sigma_Y_cm=sY, sigma_X_cm=sX), irroitus=ir)
    if ir is not None:
        Y_r = ir["Y_r"]
        t_r = float(np.interp(Y_r, Y, t))
        ir["t_r"] = t_r
        ir["t_vali"] = [float(np.interp(ir["Y_min"], Y, t)), float(np.interp(ir["Y_max"], Y, t))]
        out["tila"] = irroitustila(t, X, Y, t_r, (ir["Y_min"], ir["Y_max"]))
        out["liuku"]["tyonto_kesto_s"] = a["t_90"] - a["t_lahto"]
        out["liuku"]["tyonto_matka_cm"] = a["Y_90"] - a["Y_lahto"]
        out["liuku"]["liuku_kesto_s"] = t_r - a["t_lahto"]
        out["liuku"]["liuku_matka_cm"] = out["tila"]["Y_cm"] - a["Y_lahto"]
    return out


# ---------------------------------------------------------------------------------------------------------------------
# KUVA JA KONTAKTIARKKI
# ---------------------------------------------------------------------------------------------------------------------
def _ero_viitteeseen(Y_r, Y_vali):
    if Y_vali[0] <= Y_r <= Y_vali[1]:
        return "valissa"
    return f"ennen: {Y_r - Y_vali[0]:+.0f} cm" if Y_r < Y_vali[0] else f"jalkeen: {Y_r - Y_vali[1]:+.0f} cm"


def lue_viite(kansio, puoli, radat):
    """Silmin todettu irroitus (irroitus_visuaalinen_<puoli>.json): {rata: dict(ruutu_ote, ruutu_irti, Y_vali, t_vali)}. ruutu_ote = viimeinen ruutu, jossa
    sormet vielä pitavat kahvasta, ruutu_irti = ensimmainen ruutu, jossa sormien ja kahvan valilla on selva rako; Y ja t luetaan radalta ruudun kohdalta."""
    p = os.path.join(kansio, f"irroitus_visuaalinen_{puoli}.json")
    if not os.path.exists(p):
        return None
    out = {}
    for k, r in json.load(open(p)).items():
        if k.startswith("_"):
            continue
        d = radat[int(k)]
        out[int(k)] = dict(ruutu_ote=r["ruutu_ote"], ruutu_irti=r["ruutu_irti"],
                           Y_vali=[float(np.interp(r[n], d["f"], d["Y"])) for n in ("ruutu_ote", "ruutu_irti")],
                           t_vali=[float(np.interp(r[n], d["f"], d["t"])) for n in ("ruutu_ote", "ruutu_irti")])
    return out


def piirra(radat, tulokset, ref, polku):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    vari = {0: "#d4a017", 1: "#c0392b", 2: "#2c7fb8"}
    n = len(radat)
    fig, ax = plt.subplots(4, n, figsize=(5.2 * n, 12.5))

    def merkit(a, k, ir, kuvaus=False):
        if ir is not None:
            a.axvspan(ir["Y_min"], ir["Y_max"], color="k", alpha=0.12, label="irroitus (vaihtelu jasenten valilla)" if kuvaus else None)
            a.axvline(ir["Y_r"], color="k", lw=1.5)
        if ref and k in ref:
            a.axvspan(ref[k]["Y_vali"][0], ref[k]["Y_vali"][1], color="g", alpha=0.25, label="silmin todettu irtoaminen (kontaktiarkki)" if kuvaus else None)

    for j, k in enumerate(sorted(radat)):
        d, res = radat[k], tulokset[k]
        t, X, Y = d["t"], d["X"], d["Y"]
        m = (Y > -50)
        ir, til = res["irroitus"], res.get("tila")
        a = ax[0, j]
        a.plot(Y[m], X[m], ".", color=vari[k], ms=4)
        if til is not None:
            yy = np.linspace(Y[m].min(), Y.max(), 200)
            mm = t >= til["t_r"] + VIIVE_S
            cx = np.polyfit(Y[mm] - til["Y_cm"], X[mm], 2)
            a.plot(yy, np.polyval(cx, yy - til["Y_cm"]), "k--", lw=1, label="vapaa paraabeli (irroituksen jalkeen)")
        merkit(a, k, ir, True)
        a.set(xlabel="Y (cm)", ylabel="X (cm)", title=f"heitto {k + 1}: rata X(Y)")
        a.legend(fontsize=7, loc="best")
        a = ax[1, j]
        ys, th = suunta_paikallinen(Y[m], X[m])
        a.plot(ys, th, "-o", ms=3, color=vari[k])
        merkit(a, k, ir)
        a.set(xlabel="Y (cm)", ylabel="suunta (deg, + oikealle)", title="suunta (ikkuna 120 cm)")
        a = ax[2, j]
        sm = liukuva(t, Y, h=8, aste=2)
        a.plot(Y, sm[:, 1] / 100.0, "-", color=vari[k])
        a2 = a.twinx()
        ma = Y > -150
        a2.plot(Y[ma], sm[ma, 2] / 100.0, "-", color="0.6", lw=1)
        a2.set_ylim(-0.6, 0.6)
        a2.set_ylabel("kiihtyvyys (m/s2, harmaa)")
        la = res["liuku"]
        a.axvline(la["Y_lahto"], color="b", lw=1, ls="--")
        a.axvline(la["Y_90"], color="b", lw=1, ls="-.")
        merkit(a, k, ir)
        a.set(xlabel="Y (cm)", ylabel="nopeus (m/s)", title="nopeus ja kiihtyvyys; sin. = lahto / tyonnon loppu")
        a = ax[3, j]
        if ir is not None:
            for jas, P in zip(ir["jasenet"], ir["profiilit"]):
                P = np.array(P)
                a.plot(P[:, 0], P[:, 1] + P[:, 2], lw=1, label=f"ikkuna Y >= {jas['Y_a']:.0f}")
            a.set_yscale("log")
        merkit(a, k, ir)
        a.set(xlabel="vaihtokohta Y_r (cm)", ylabel="khi^2 (Y(t) + X(Y))", title="vaihtokohdan profiili, viimeinen kierros (pienempi = parempi)")
        a.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(polku, dpi=100)


def kontaktiarkki(video, kansio, puoli, radat, tulokset, ref, polku_pohja, sade_s=0.45, sarakkeita=6, S=3):
    """Yksi kuva per heitto (<polku_pohja>_h<N>.jpg): kaikki yksilolliset ruudut t_r +- sade_s, kahvan ymparille rajattu, 3x suurennus. Punainen kehys =
    kinemaattinen irroitus, vihrea teksti = silmin todettu irtoamisvali (ruutu_ote ... ruutu_irti)."""
    import cv2

    import kivi_netti as kv

    cam = kv.Kamera(puoli, os.path.join(kansio, f"kalibrointi_{puoli}.json"), os.path.join(kansio, f"moodikuva_{puoli}.png"))
    cap = cv2.VideoCapture(video)
    kirjoitetut = []
    for k in sorted(radat):
        d, res = radat[k], tulokset[k]
        if res.get("tila") is None:
            continue
        t_r = res["irroitus"]["t_r"]
        i_r = int(np.argmin(np.abs(d["t"] - t_r)))
        ruudut = []
        for i in range(len(d["t"])):
            if abs(d["t"][i] - t_r) > sade_s:
                continue
            fr = cam.ruutu(cap, int(d["f"][i]))
            u, v = kv.kivimalli.project_3d(cam.K, cam.R, cam.t, np.array([[d["X"][i], d["Y"][i], 15.0]]))
            cx, cy = int(u[0]), int(v[0])
            x0, y0 = max(0, cx - 80), max(0, cy - 60)
            big = cv2.resize(fr[y0:cy + 60, x0:cx + 40], None, fx=S, fy=S, interpolation=cv2.INTER_CUBIC)
            big = cv2.copyMakeBorder(big, 0, max(0, 120 * S - big.shape[0]), 0, max(0, 120 * S - big.shape[1]), cv2.BORDER_CONSTANT, value=(255, 255, 255))
            vihrea = bool(ref and k in ref and ref[k]["ruutu_ote"] <= d["f"][i] <= ref[k]["ruutu_irti"])
            cv2.putText(big, f"f={int(d['f'][i])} Y={d['Y'][i]:.0f} ({d['t'][i] - t_r:+.2f} s)", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 160, 0) if vihrea else (0, 0, 255), 2)
            if i == i_r:
                cv2.rectangle(big, (1, 1), (big.shape[1] - 2, big.shape[0] - 2), (0, 0, 255), 4)
            ruudut.append(big)
        if not ruudut:
            continue
        kork, lev = ruudut[0].shape[:2]
        while len(ruudut) % sarakkeita:
            ruudut.append(np.full((kork, lev, 3), 255, np.uint8))
        img = np.vstack([np.hstack(ruudut[i:i + sarakkeita]) for i in range(0, len(ruudut), sarakkeita)])
        polku = f"{polku_pohja}_h{k + 1}.jpg"
        cv2.imwrite(polku, img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        kirjoitetut.append(polku)
    return kirjoitetut


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?", default=None)
    ap.add_argument("kansio")
    ap.add_argument("puoli")
    ap.add_argument("--ulos", default=None)
    ap.add_argument("--ei-ruutuja", action="store_true")
    a = ap.parse_args()
    ulos = a.ulos or os.path.join(a.kansio, f"liuku_irroitus_{a.puoli}")
    radat = lataa_radat(a.kansio, a.puoli)
    tulokset = {k: analysoi(d) for k, d in radat.items()}
    ref = lue_viite(a.kansio, a.puoli, radat)
    for k in sorted(tulokset):
        r = tulokset[k]
        la, ir, ti = r["liuku"], r["irroitus"], r.get("tila")
        print(f"heitto {k + 1}: lahto t = {la['t_lahto']:.2f} s (Y = {la['Y_lahto']:.0f} cm{'' if la['lahto_havaittu'] else ', ARVIO'}), tyonto loppuu t = {la['t_90']:.2f} s "
              f"(Y = {la['Y_90']:.0f} cm), tasanne {la['v_tasanne_m_s']:.2f} m/s; kohina Y {r['kohina']['sigma_Y_cm']:.2f} cm, X {r['kohina']['sigma_X_cm']:.2f} cm")
        if ir is None:
            print("   irroitusta ei voitu maarittaa")
            continue
        print(f"   irroitus: Y = {ir['Y_r']:.0f} cm (jasenten vaihtelu {ir['Y_min']:.0f}...{ir['Y_max']:.0f}; kierrokset {[round(h) for h in ir['historia']]}), "
              f"t = {ir['t_r']:.2f} s; kanavat: Y(t) {ir['Y_r_Y_kanava']:.0f}, X(Y) {ir['Y_r_X_kanava']:.0f}; dchi2 mediaani {ir['dchi2_med']:.0f}, "
              f"merkitsevia {ir['merkitsevia']}/{ir['jasenia']} -> {'TUNNISTETTU' if ir['merkitseva'] else 'EI merkitseva'}")
        print(f"   tila: X = {ti['X_cm']:.1f} cm, suunta {ti['suunta_deg']:+.2f} +- {ti['suunta_virhe_deg']:.2f} deg, nopeus {ti['nopeus_m_s']:.2f} m/s, "
              f"hidastuvuus {ti['hidastuvuus_m_s2']:+.3f} m/s2, kaarevuus {ti['kaarevuus_1_cm'] * 1e4:+.2f}e-4 /cm ({ti['kaarto']})")
        print(f"   liuku: kesto {la['liuku_kesto_s']:.2f} s, matka {la['liuku_matka_cm']:.0f} cm (tyonto {la['tyonto_kesto_s']:.2f} s / {la['tyonto_matka_cm']:.0f} cm)")
        if ref and k in ref:
            rf = ref[k]
            print(f"   silmin todettu irtoaminen: Y = {rf['Y_vali'][0]:.0f}...{rf['Y_vali'][1]:.0f} cm (ruudut {rf['ruutu_ote']}...{rf['ruutu_irti']}) -> kinemaattinen "
                  f"{_ero_viitteeseen(ir['Y_r'], rf['Y_vali'])}")
    ilman_profiileja = {str(k): {**v, "irroitus": {n: a for n, a in v["irroitus"].items() if n != "profiilit"}} if v.get("irroitus") else v for k, v in tulokset.items()}
    json.dump(ilman_profiileja, open(ulos + ".json", "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    piirra(radat, tulokset, ref, ulos + ".png")
    print("->", ulos + ".json", ulos + ".png")
    if a.video and not a.ei_ruutuja:
        for p in kontaktiarkki(a.video, a.kansio, a.puoli, radat, tulokset, ref, ulos + "_ruudut"):
            print("->", p)
