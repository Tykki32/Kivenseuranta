"""Kiven liu'un ja irroituksen tunnistus radasta: python liuku_irroitus.py VIDEO tulokset vasen [--ulos tulokset/liuku_irroitus_vasen] [--ei-ruutuja]

Tunnistus perustuu pelkkaan kiven rataan (tulokset/radat_<puoli>.csv: ruutu, todellinen aika t_kor_s, X, Y) -- ei kahvaan eika kateen:
  * LIUKU (alku): kivi lahtee hackista (lepo -> liike), tyonto kiihdyttaa sen liukunopeuteen (90 % tasanteesta).
  * IRROITUS: irroituksen jalkeen kivi on vapaa kappale, eli rata on tasainen 2. asteen kayra ilman teravia muutoksia: X(Y) paraabeli (kaarto,
    "curl") JA Y(t) 2. asteen polynomi (vakiohidastuvuus). Ennen irroitusta heittaja tyontaa ja ohjaa kivea: kiihtyvyys (Y(t)) ja kaarevuus (X(Y))
    ovat toiset. Irroitus = viimeinen merkitseva nopeuden tai suunnan muutos: kohta, jossa kiihtyvyys tai kaarevuus viimeksi muuttuu merkitsevasti
    (MENETELMA "muutos", irroitus_muutos: kinkkitilaston moniskaalainen profiili, viimeisen yhtenaisen merkitsevan alueen voimakkain kohta). Vanhat menetelmat
    vertailua varten: "hanta" (vapaan lennon ikkuna kasvaa takaperin: loysa, irroitus liian aikaisin) ja "kinkki" (kahden paraabelin C1-sovitus, irroitus_varmistettu).
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
LIUKU_ALKU_VIIVE_S = 0.6     # liu'un hidastuvuus: ikkuna alkaa tyonnon lopusta + tama (tyonnon jalkeinen tasoittuminen pois)
LIUKU_LOPPU_VIIVE_S = 0.15   # ... ja paattyy tama ennen irroitusta
NOPEUS_IKKUNA_S = 0.4        # "juuri ennen irroitusta" -nopeus: viimeiset 0,4 s
RAJA_DCHI2 = 40.0            # merkitseva vaihtokohta: khi^2-parannus vahintaan tama (nollahypoteesin suurin arvo simuloinnissa 35, vapaan lennon ikkunoissa 33)
# Vapaan lennon fysiikka pehmeana esitietona (MAP): irroituksen jalkeen kivi on vapaa kappale, joka vain hidastuu (puhtailla heitoilla 0,06-0,07 m/s2) ja kaartaa vahan. Ilman esitietoa
# vapaan lennon paraabeli saa kiihtya (ohjatun vaiheen tyonto mukaan) tai kaartua mielivaltaisesti: irroitus tunnistuttiin liian aikaisin (nopeus irroituksen jalkeen 0,2-0,3 m/s liian pieni,
# suunta jopa 1 aste pielessa; kts. README "Irroituksen tunnistuksen fysiikka"). Esitieto vaikuttaa vain, kun data ei rajaa parametria (lyhyt vapaa lento) tai rata ei ole vapaan lennon mukainen.
A_VAPAA_M_S2 = 0.065         # vapaan lennon hidastuvuus (m/s2)
SIGMA_A_VAPAA_M_S2 = 0.03    # ... ja sen hajonta
SIGMA_KAAREVUUS_1_CM = 5e-5  # vapaan lennon X(Y)-kaarevuus: 0 +- tama (mitattu hajonta 4-5e-5 1/cm, sis. mittakohinan): kinkkihaun ja hannan ikkunan ennustetesti
SIGMA_KAAREVUUS_TILA_1_CM = 5e-5   # ... ja irroitustilan (suunta irroituksessa) sovituksessa: tiukempi esitieto vakauttaa suunnan (irroituksen suunta = hog-viivan suunta - kaarto matkalla)
Y_IRROITUS_MAX_CM = None     # irroitus ei voi olla taman jalkeen (saanto: ennen hog-viivaa; reaaliaikaisessa asetuksessa ESTIMAATTI.Y_r_max)
MENETELMA = "muutos"           # irroituksen tunnistus: "muutos" (viimeinen merkitseva kiihtyvyyden / kaarevuuden muutos; kts. irroitus_muutos), "hanta" (vapaan lennon hanta, takaperin kasvava ikkuna; kts. irroitus_hanta) tai "kinkki" (kahden paraabelin C1-sovitus, irroitus_varmistettu)


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


def _irls(A, y, s, kierroksia=6, esi=None):
    """Huber-IRLS. esi = (rivi, tavoite, sigma): pehmea esitieto yhdelle parametrille (rivi @ p ~ N(tavoite, sigma^2); ei Huber-painotusta). Palauttaa (p, kustannus_yht, kustannus_data):
    yht = data + esitiedon sakko ((rivi @ p - tavoite) / sigma)^2 (vaihtokohdan valinta), data = pelkka Huber-kustannus (merkitsevyys)."""
    w = np.ones(len(y))
    for _ in range(kierroksia):
        sw = np.sqrt(w)
        A2, y2 = A * sw[:, None], y * sw
        if esi is not None:
            k = s / esi[2]
            A2, y2 = np.vstack([A2, np.asarray(esi[0], float) * k]), np.append(y2, esi[1] * k)
        p = np.linalg.lstsq(A2, y2, rcond=None)[0]
        a = np.abs(y - A @ p) / (HUBER_C * s)
        w = np.where(a <= 1.0, 1.0, 1.0 / np.maximum(a, 1e-9))
    c = _kustannus(y - A @ p, s)
    return p, c + (0.0 if esi is None else float(((np.asarray(esi[0], float) @ p - esi[1]) / esi[2]) ** 2)), c


def _c0(u, y, s):
    d = u - u.mean()
    return _irls(np.column_stack([np.ones_like(d), d, 0.5 * d ** 2]), y, s)


def _c1(u, y, ur, s, esi=None):
    """y = y_r + v_r (u - ur) + 0,5 c1 (u - ur)^2 (u < ur) / c2 (u >= ur): arvo ja 1. derivaatta jatkuvia, 2. derivaatta hyppaa. esi: esitieto vapaan lennon 2. derivaatalle c2."""
    d = u - ur
    return _irls(np.column_stack([np.ones_like(d), d, 0.5 * d ** 2 * (d < 0), 0.5 * d ** 2 * (d >= 0)]), y, s, esi=esi)


ESI_Y = ((0.0, 0.0, 0.0, 1.0), -100.0 * A_VAPAA_M_S2, 100.0 * SIGMA_A_VAPAA_M_S2)     # Y(t): vapaan lennon 2. derivaatta (cm/s2) ~ hidastuvuus
ESI_X = ((0.0, 0.0, 0.0, 1.0), 0.0, SIGMA_KAAREVUUS_1_CM)                              # X(Y): vapaan lennon kaarevuus (1/cm)


def kohinat(t, X, Y):
    """Mittauskohina (robusti MAD) radan viimeisilta KOHINA_HAL_CM sentilta (vapaa lento): (sigma_Y [cm], sigma_X [cm])."""
    m = Y >= Y.max() - KOHINA_HAL_CM
    tt, yy, xx = t[m], Y[m], X[m]
    ry = yy - np.polyval(np.polyfit(tt - tt.mean(), yy, 2), tt - tt.mean())
    rx = xx - np.polyval(np.polyfit(yy - yy.mean(), xx, 2), yy - yy.mean())
    mad = lambda r: 1.4826 * float(np.median(np.abs(r - np.median(r))))
    return max(mad(ry), 0.3), max(mad(rx), 0.06)


def irroitus_ikkuna(t, X, Y, Y_a, sY, sX, askel_s=0.02, min_ennen=None, min_jalkeen=None, Y_r_min=None, Y_r_max=None, esitieto=False):
    """Vaihtokohdan haku alueella Y >= Y_a. Palauttaa dict tai None. Vaihtokohta t_r on yhteinen: Y(t) C1-sovitus antaa paikan Y_r(t_r), johon X(Y):n
    C1-sovituksen vaihtokohta asetetaan; kustannus = Y- ja X-kanavan khi^2:n summa (esitieto: + vapaan lennon fysiikan sakko, ks. ESI_Y / ESI_X; vaihtokohta valitaan taman
    kokonaiskustannuksen minimista, merkitsevyys lasketaan pelkasta datakustannuksesta).
    Reaaliaikaisessa (osittaisessa) aineistossa: min_jalkeen = vapaan lennon vahimmaismatka (oletus MIN_JALKEEN_CM), Y_r_min = irroituksen aikaisin paikka
    (esitieto: irroitus ei tapahdu aivan hackin vieressa), Y_r_max = myohaisin paikka (saanto: irroitus ennen hog-viivaa; oletus Y_IRROITUS_MAX_CM)."""
    min_ennen = MIN_ENNEN_CM if min_ennen is None else min_ennen
    min_jalkeen = MIN_JALKEEN_CM if min_jalkeen is None else min_jalkeen
    Y_r_max = Y_IRROITUS_MAX_CM if Y_r_max is None else Y_r_max
    Y_alaraja = Y_a + min_ennen if Y_r_min is None else max(Y_a + min_ennen, Y_r_min)
    m = Y >= Y_a
    tt, yy, xx = t[m], Y[m], X[m]
    if len(tt) < 25:
        return None
    k0y = _c0(tt, yy, sY)[1]
    k0x = _c0(yy, xx, sX)[1]
    esy, esx = (ESI_Y, ESI_X) if esitieto else (None, None)
    prof = []
    for tr in np.arange(tt[0], tt[-1], askel_s):
        Yr = float(np.interp(tr, tt, yy))
        if Yr < Y_alaraja or Yr > yy[-1] - min_jalkeen or (Y_r_max is not None and Yr > Y_r_max) or (tt < tr).sum() < 8 or (tt >= tr).sum() < 10:
            continue
        py, cy, dy = _c1(tt, yy, tr, sY, esy)
        px, cx, dx = _c1(yy, xx, py[0], sX, esx)
        prof.append((tr, float(py[0]), cy, cx, py, px, dy, dx))
    if not prof:
        return None
    P = np.array([[a, b, c, d, e, f] for a, b, c, d, _, _, e, f in prof])        # t_r, Y_r, khi2_Y, khi2_X (kokonais), datakustannukset Y, X
    out = dict(Y_a=float(Y_a), n=int(len(tt)), k0y=k0y, k0x=k0x)
    for nimi, kol, kold, k0 in (("yht", None, None, k0y + k0x), ("y", 2, 4, k0y), ("x", 3, 5, k0x)):
        s = P[:, 2] + P[:, 3] if kol is None else P[:, kol]
        sd = P[:, 4] + P[:, 5] if kol is None else P[:, kold]
        i = int(np.argmin(s))
        ok = P[s <= s[i] + 4.0, 1]
        out[nimi] = dict(t_r=float(P[i, 0]), Y_r=float(P[i, 1]), Y_lo=float(ok.min()), Y_hi=float(ok.max()), dchi2=float(k0 - sd[i]))
    i = int(np.argmin(P[:, 2] + P[:, 3]))
    py, px = prof[i][4], prof[i][5]
    out["a1_m_s2"], out["a2_m_s2"] = float(py[2] / 100.0), float(py[3] / 100.0)
    out["c1_1_cm"], out["c2_1_cm"] = float(px[2]), float(px[3])
    out["profiili"] = P[:, [1, 2, 3]].tolist()           # (Y_r, khi2_Y, khi2_X)
    return out


def _jasenet(t, X, Y, alut, sY, sX, **asetus):
    out = []
    for Y_a in alut:
        r = irroitus_ikkuna(t, X, Y, Y_a, sY, sX, **asetus)
        if r is not None:
            out.append(r)
    return out


def irroitus_varmistettu(t, X, Y, Y_90, sY, sX, **asetus):
    """Vaihtokohta usealla ohjatun vaiheen ikkunalla. Kierros 0: ikkuna alkaa tyonnon lopusta + marginaali (laaja, karkea). Tarkennuskierrokset: ikkuna
    alkaa L cm ENNEN edellista vaihtokohtaa (ohjattu vaihe ei ole 2. asteen kayra koko matkaa; paikallinen ikkuna on lahempana mallia). Palauttaa dict:
    mediaani, vaihteluvali, jasenet, merkitsevyys (enemmistolla jasenista khi^2-parannus >= RAJA_DCHI2)."""
    jasenet = _jasenet(t, X, Y, [Y_90 + m for m in MARGINAALIT_CM], sY, sX, **asetus)
    if not jasenet:
        return None
    Y_r = float(np.median([r["yht"]["Y_r"] for r in jasenet]))
    historia = [Y_r]
    for _ in range(TARKENNUS_MAX):
        uudet = _jasenet(t, X, Y, [max(Y_90 + 50.0, Y_r - L) for L in TARKENNUS_L_CM], sY, sX, **asetus)
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
# IRROITUS: VAPAAN LENNON HANTA (takaperin kasvava ikkuna)
# ---------------------------------------------------------------------------------------------------------------------
HANTA_Y_ANKKURI_CM = 631.8   # ankkuri: hog-viivan jalkeinen rata on varmasti vapaata lentoa (irroitus ennen viivaa: kiven keskipiste Y <= 631,8 cm); reaaliaikaisessa asetuksessa ESTIMAATTI
HANTA_MIN_N = 12             # ankkuri-ikkunassa vahintaan nain monta paikkaa
HANTA_M = 6                  # ennustejaannoksia tarkistetaan tama monta ikkunaa edeltavaa paikkaa kerrallaan
HANTA_Z = 3.0                # ... ja irroitus on siina, missa niiden |z|-mediaani ylittaa taman
HANTA_SIGMA_Y = (0.3, 0.8)   # mittakohinan (cm) rajat, kun kohinaa ei anneta: robusti arvio ankkuri-ikkunasta (lyhyt ikkuna aliarvioi: kayta analysoi-funktion kohinat-arviota)
HANTA_SIGMA_X = (0.07, 0.15)
HANTA_SIGMA_Y_MIN = 0.5      # analysoi: kohinat()-arvion alaraja (cm); alle tama ennustejaannokset eivat ole luotettavia (korreloitu virhe: B-sovituksen harha hog-viivan kohdalla, mallin vaje)
HANTA_SIGMA_X_MIN = 0.10
HANTA_PERSISTENSSI = 3       # irroitus on vasta, kun |z|-mediaani ylittaa HANTA_Z:n taman monessa perakkaisessa ikkunassa (yksittainen ylitys on kohinaa)


def _paraabeli_posteriori(u, y, s, esi=None, kierroksia=5):
    """Paraabeli y = p0 + p1 d + p2 d^2 / 2 (d = u - u[0]); Huber-IRLS, esitieto p2 ~ N(esi[0], esi[1]). Palauttaa (p, Q, u0), Q = (A^T W A + esitieto)^-1 (kovarianssi = s^2 Q)."""
    u0 = float(u[0])
    d = u - u0
    A = np.column_stack([np.ones_like(d), d, 0.5 * d * d])
    w = np.ones(len(y))
    for _ in range(kierroksia):
        sw = np.sqrt(w)
        A2, y2 = A * sw[:, None], y * sw
        if esi is not None:
            k = s / esi[1]
            A2, y2 = np.vstack([A2, [0.0, 0.0, k]]), np.append(y2, esi[0] * k)
        p = np.linalg.lstsq(A2, y2, rcond=None)[0]
        a = np.abs(y - A @ p) / (HUBER_C * s)
        w = np.where(a <= 1.0, 1.0, 1.0 / np.maximum(a, 1e-9))
    info = A.T @ (A * w[:, None])
    if esi is not None:
        info = info + np.diag([0.0, 0.0, (s / esi[1]) ** 2])
    return p, np.linalg.inv(info), u0, A @ p - y


def _ennuste_z(p, Q, u0, s, u, y):
    """Ikkunan sovituksen ennuste ikkunan ulkopuolisille naytteille: standardoidut jaannokset (ennusteen epavarmuus mukana)."""
    d = u - u0
    Hh = np.column_stack([np.ones_like(d), d, 0.5 * d * d])
    return (y - Hh @ p) / (s * np.sqrt(1.0 + np.einsum("ij,jk,ik->i", Hh, Q, Hh)))


def irroitus_hanta(t, X, Y, sY=None, sX=None, Y_ankkuri=None, Y_r_min=None, m=None, z_raja=None, esitieto=True, **_):
    """Irroitus = varhaisin kohta, josta alkaen rata on vapaan lennon mukainen: ikkuna alkaa ankkurista (hog-viivan jalkeen, varmasti vapaata lentoa) ja kasvaa takaperin; kutakin
    ikkunaa vasten sovitetaan vapaan lennon malli (Y(t): vakiohidastuvuus, X(Y): vahan kaartuva suora, fysikaaliset esitiedot) ja sita edeltavien m naytteen standardoitujen
    ennustejaannosten |z|-mediaani (Y- ja X-kanavan suurempi) tarkistetaan; irroitus on ensimmaisessa ikkunassa, jonka edella jaannokset ylittavat HANTA_Z:n (tai aikaisimmassa
    sallitussa kohdassa Y_r_min, jolloin merkitseva = False). Ohjatun vaiheen (tyonto, kasin johdettu liuku) mallia ei tarvita, eika tulos riipu monesta paikallisesta vaihtokohdasta:
    myohainen tyonto (kasi kiihdyttaa viela irroituksen alla) tai sivuttaisliike katkaisevat ikkunan. Palauttaa dict (kuten irroitus_varmistettu) tai None (ankkuri puuttuu)."""
    Y_ankkuri = HANTA_Y_ANKKURI_CM if Y_ankkuri is None else Y_ankkuri
    m = HANTA_M if m is None else m
    z_raja = HANTA_Z if z_raja is None else z_raja
    n = len(t)
    if not (Y >= Y_ankkuri).any():
        return None
    i_a = min(int(np.argmax(Y >= Y_ankkuri)), n - HANTA_MIN_N)
    if i_a < m + 1:
        return None
    esy = (-100.0 * A_VAPAA_M_S2, 100.0 * SIGMA_A_VAPAA_M_S2) if esitieto else None
    esx = (0.0, SIGMA_KAAREVUUS_1_CM) if esitieto else None
    if sY is None or sX is None:                                     # kohina ankkuri-ikkunasta (robusti MAD)
        py, _, _, ry = _paraabeli_posteriori(t[i_a:], Y[i_a:], 0.5, esy)
        px, _, _, rx = _paraabeli_posteriori(Y[i_a:], X[i_a:], 0.1, esx)
        mad = lambda r: 1.4826 * float(np.median(np.abs(r - np.median(r))))
        sY = float(np.clip(mad(ry), *HANTA_SIGMA_Y)) if sY is None else sY
        sX = float(np.clip(mad(rx), *HANTA_SIGMA_X)) if sX is None else sX
    Y_raja = -np.inf if Y_r_min is None else Y_r_min
    i0, z_rikki, rajattu, ylit = i_a, None, False, 0
    while True:
        if i0 - m < 0 or Y[i0 - 1] < Y_raja:
            rajattu = True
            break
        py, Qy, ty, _ = _paraabeli_posteriori(t[i0:], Y[i0:], sY, esy)
        px, Qx, uy, _ = _paraabeli_posteriori(Y[i0:], X[i0:], sX, esx)
        zy = _ennuste_z(py, Qy, ty, sY, t[i0 - m:i0], Y[i0 - m:i0])
        zx = _ennuste_z(px, Qx, uy, sX, Y[i0 - m:i0], X[i0 - m:i0])
        z = max(float(np.median(np.abs(zy))), float(np.median(np.abs(zx))))
        ylit = ylit + 1 if z > z_raja else 0
        if ylit >= HANTA_PERSISTENSSI:
            z_rikki = z
            i0 += ylit - 1                                            # irroitus = viimeinen johdonmukainen ikkuna (ennen ensimmaista ylitysta)
            break
        i0 -= 1
    Y_r, t_r = float(Y[i0]), float(t[i0])
    Y_lo, Y_hi = float(Y[max(i0 - m, 0)]), Y_r
    return dict(jasenet=[], profiilit=[], historia=[Y_r], Y_r=Y_r, Y_min=Y_lo, Y_max=Y_hi, Y_r_Y_kanava=Y_r, Y_r_X_kanava=Y_r, dchi2_med=float((z_rikki or 0.0) ** 2 * m), merkitsevia=int(not rajattu),
                jasenia=1, merkitseva=bool(not rajattu), t_r=t_r, menetelma="hanta", sigma_Y_cm=float(sY), sigma_X_cm=float(sX), ikkuna_n=int(n - i0))


# ---------------------------------------------------------------------------------------------------------------------
# IRROITUS: VIIMEINEN MUUTOS (moniskaalainen kinkkitilasto)
# ---------------------------------------------------------------------------------------------------------------------
# Heittaja tyontaa ja ohjaa kivea irroitukseen asti: rata poikkeaa vapaasta lennosta (kiihtyvyys Y(t), kaarevuus X(Y)); irroituksen jalkeen vain hidastuvuus ja kaarto. Irroitus on siis viimeinen
# merkitseva nopeuden tai suunnan muutos. Silmin todettu (22 heiton kontaktiarkit, keltaisella kahvalla irtoaminen nakyy selvasti): tyonnon loppu = nopeuden huippu, kasi irtoaa 0-60 cm myohemmin.
# Vapaan lennon ikkunan kasvattaminen takaperin (irroitus_hanta) pysahtyy vasta kun poikkeama on 3 sigmaa, koska loiva tyonto sopii vapaan lennon paraabeliin (vapaat parametrit): irroitus tuli
# 0,7-1,2 m liian aikaisin, ja tyonnon aikana nopeus ennen ja jalkeen irroituksen erosivat 0,1-0,25 m/s.
MUUTOS_SKAALAT_CM = (100.0, 150.0, 200.0, 300.0)    # kinkkitilaston ikkunan puolikas L (cm): lyhyt paikantaa terävan muutoksen, pitka havaitsee loivan tyonnon
MUUTOS_ASKEL_CM = 5.0                               # ehdokaskohtien vali
MUUTOS_Y_ALKU_CM = 250.0                            # aikaisin ehdokas (hackista lahtevan tyonnon jalkeen)
MUUTOS_SIGMA_Y_CM = 0.7                             # nimellinen mittakohina (valkoinen) kinkkitilaston nimittajassa; vakio, koska radan lopusta arvioitu kohina sisaltaa myohaisen tyonnon (keltaisilla 0,3-1,6 cm)
MUUTOS_SIGMA_X_CM = 0.12
MUUTOS_KYNNYS_Y = 5.0                               # z-kynnys kiihtyvyyden pudotukselle (tyonto loppuu): vapaan lennon heitoilla (24) pienin z alueella Y = 520-632 cm oli -4,9 (skaalat 100-200 cm)
MUUTOS_KYNNYS_X = 9.0                               # ... ja suurin |z| kaarevuuden muutokselle 8,5 (X-kanava herkempi: sivusuuntainen kohina 0,12 cm, kasi heiluttaa kiveä ohjatussa vaiheessa)
MUUTOS_HEIKKO = 0.6                                 # heikoin kelvollinen tapahtuma (voimakkuus S; S >= 1 = merkitseva)
MUUTOS_LAMBDA_CM = 300.0                            # lopun ennakko: tapahtuman S painotetaan exp((Y - Y_ylaraja) / lambda): myohempi muutos voittaa aikaisemman, ellei se ole selvasti heikompi (heittajat irroittavat 0-3 m
                                                    # ennen hog-viivaa, viimeinen muutos); pehmea valinta on vakaa kynnysten suhteen (kova 'viimeinen merkitseva alue' hyppasi 12 %:lla heitoista > 60 cm kynnysten +-10 %:n muutoksella)
MUUTOS_PAIKANNUS_OSUUS = 0.8                        # paikannus: hienoin skaala, jolla S >= taman osuus valitun kohdan S:sta +-MUUTOS_PAIKANNUS_CM:n sisalla (karkea skaala vain havaitsee loivan tyonnon)
MUUTOS_PAIKANNUS_CM = 100.0
MUUTOS_VALI_OSUUS = 0.75                            # epavarmuusvali: yhtenainen alue valitun kohdan ymparilla, jossa S >= taman osuus huipusta
MUUTOS_MIN_N = 6                                    # kinkin molemmin puolin vahintaan taman verran pisteita
MUUTOS_PUHDISTUS = 4.0                              # yksittaiset mittausvirheet (kasi peittaa kiven, B-sovituksen harha): jaannos naapureiden paraabelista > taman * robusti hajonta -> pois
MUUTOS_PUHDISTUS_S, MUUTOS_PUHDISTUS_CM = 0.3, 40.0   # naapuruston koko: Y(t) +-0,3 s, X(Y) +-40 cm
MUUTOS_TYONTO_L_CM = 150.0                          # tyonto: ikkunan [Y_h - L, Y_h] kiihtyvyys on selvasti yli vapaan lennon hidastuvuuden (kasi tyontaa kivea): irroitus ei voi olla ennen viimeista tyontoa
MUUTOS_TYONTO_Z = 4.0
MUUTOS_TYONTO_MIN_M_S2 = 0.10


def _kinkki_z(u, y, uh, skaala, s):
    """Kinkkitilasto kohdassa uh: ikkunan pisteet (u, y) sovitetaan kahdella paraabelilla, joiden arvo ja 1. derivaatta jatkuvat uh:ssa (y = y_h + v d + 0,5 c_jalkeen d^2 (d >= 0) /
    0,5 c_ennen d^2 (d < 0)); z = (c_jalkeen - c_ennen) / sigma, sigma = s * sqrt(Var-kerroin) (valkoinen kohina s). skaala vain kuntoon asettamiseen. nan, jos pisteita liian vahan."""
    d = (u - uh) / skaala
    ja = d >= 0.0
    if int(ja.sum()) < MUUTOS_MIN_N or int((~ja).sum()) < MUUTOS_MIN_N:
        return np.nan
    d2 = 0.5 * d * d
    A = np.column_stack([np.ones_like(d), d, d2 * ja, d2 * (~ja)])
    try:
        Ni = np.linalg.inv(A.T @ A)
    except np.linalg.LinAlgError:
        return np.nan
    p = Ni @ (A.T @ y)
    v = Ni[2, 2] + Ni[3, 3] - 2.0 * Ni[2, 3]
    if not v > 0.0:
        return np.nan
    return float((p[2] - p[3]) / (s * np.sqrt(v)))


def _poikkeamat(t, X, Y):
    """Totuusarvotaulukko: False = yksittainen mittausvirhe. Kunkin pisteen jaannos naapuriensa (itse pois) paraabelista; hylataan, jos jaannos > MUUTOS_PUHDISTUS * max(robusti hajonta, nimellinen).
    Y(t) ja X(Y) erikseen. Aidot muutokset (kinkki, sileä poikkeama) eivat ole yksittaisia: naapuriston paraabeli seuraa niita."""
    ok = np.ones(len(t), bool)
    for u, v, s0, ikk in ((t, Y, MUUTOS_SIGMA_Y_CM, MUUTOS_PUHDISTUS_S), (Y, X, MUUTOS_SIGMA_X_CM, MUUTOS_PUHDISTUS_CM)):
        r = np.full(len(t), np.nan)
        for k in range(len(t)):
            m = np.abs(u - u[k]) <= ikk
            m[k] = False
            if int(m.sum()) >= 6:
                r[k] = v[k] - np.polyfit(u[m] - u[k], v[m], 2)[2]
        if np.isfinite(r).sum() >= 10:
            s = max(1.4826 * float(np.nanmedian(np.abs(r - np.nanmedian(r)))), s0)
            ok &= ~(np.abs(np.nan_to_num(r)) > MUUTOS_PUHDISTUS * s)
    return ok


def _muutos_profiilit(t, X, Y, ys):
    """Kinkkitilastot z_Y (kiihtyvyyden hyppy, Y(t)) ja z_X (kaarevuuden hyppy, X(Y)) skaaloittain (rivit) ja ehdokaskohdittain ys (sarakkeet); ikkuna Y in [Y_h - L, Y_h + L]."""
    ZY = np.full((len(MUUTOS_SKAALAT_CM), len(ys)), np.nan)
    ZX = np.full_like(ZY, np.nan)
    for a, L in enumerate(MUUTOS_SKAALAT_CM):
        for k, Yh in enumerate(ys):
            m = (Y >= Yh - L) & (Y <= Yh + L)
            if int(m.sum()) < 2 * MUUTOS_MIN_N:
                continue
            ZY[a, k] = _kinkki_z(t[m], Y[m], float(np.interp(Yh, Y, t)), 1.0, MUUTOS_SIGMA_Y_CM)
            ZX[a, k] = _kinkki_z(Y[m], X[m], Yh, L, MUUTOS_SIGMA_X_CM)
    return ZY, ZX


def _tyonto_loppu(t, Y, Y_alku):
    """Viimeinen ikkunan paate Y_h (askel MUUTOS_ASKEL_CM, Y_h >= Y_alku), jonka ikkunassa [Y_h - L, Y_h] (L = MUUTOS_TYONTO_L_CM) kivea tyonnetaan: paraabelin kiihtyvyys on vahintaan
    MUUTOS_TYONTO_MIN_M_S2 ja z >= MUUTOS_TYONTO_Z yli vapaan lennon hidastuvuuden (A_VAPAA_M_S2). Kasi vaikuttaa silloin ikkunassa, joten irroitus ei ole sita aikaisemmin. None, jos tyontoa ei ole."""
    viimeinen = None
    for Yh in np.arange(Y_alku, float(Y.max()) + 1e-9, MUUTOS_ASKEL_CM):
        m = (Y >= Yh - MUUTOS_TYONTO_L_CM) & (Y <= Yh)
        if int(m.sum()) < 10:
            continue
        d = t[m] - float(t[m].mean())
        A = np.column_stack([np.ones_like(d), d, 0.5 * d * d])
        try:
            Ni = np.linalg.inv(A.T @ A)
        except np.linalg.LinAlgError:
            continue
        a = float((Ni @ (A.T @ Y[m]))[2])                                        # cm/s2
        if a / 100.0 >= MUUTOS_TYONTO_MIN_M_S2 and (a + 100.0 * A_VAPAA_M_S2) / (MUUTOS_SIGMA_Y_CM * float(np.sqrt(Ni[2, 2]))) >= MUUTOS_TYONTO_Z:
            viimeinen = float(Yh)
    return viimeinen


def irroitus_muutos(t, X, Y, Y_r_min=None, Y_r_max=None, min_jalkeen=None, **_):
    """Irroitus = viimeinen merkitseva nopeuden tai suunnan muutos. Ehdokaskohdat Y_h (askel MUUTOS_ASKEL_CM, alue MUUTOS_Y_ALKU_CM / Y_r_min ... min(Y_r_max, radan loppu - min_jalkeen)); kullekin
    kinkkitilastot z_Y (Y(t): kiihtyvyyden pudotus = tyonto loppuu) ja z_X (X(Y): kaarevuuden muutos) skaaloilla MUUTOS_SKAALAT_CM; voimakkuus S_L = max(-z_Y / KYNNYS_Y, |z_X| / KYNNYS_X), S >= 1 = merkitseva.
    Valinta: verhokayran max_L S_L suurin arvo painotettuna exp((Y - ylaraja) / MUUTOS_LAMBDA_CM): myohempi muutos voittaa aikaisemman, ellei se ole selvasti heikompi (pehmea 'viimeinen muutos'; kova kynnys
    hyppasi tuhansia senttimetreja kynnysten pienilla muutoksilla). Paikannus hienoimmalla skaalalla, jolla muutos on edelleen selva (MUUTOS_PAIKANNUS_OSUUS): lyhyt skaala paikantaa terävan muutoksen, pitka vain
    havaitsee loivan tyonnon. Irroitus ei voi olla ennen viimeista merkitsevaa tyontoa (_tyonto_loppu: ikkunan kiihtyvyys selvasti yli vapaan lennon), ja jos kivea tyonnetaan viela radan lopussa, irroitus
    ei ole viela tapahtunut -> ehdokasalueen ylareuna (merkitseva = False). merkitseva = valitun muutoksen S >= 1; heikoin hyvaksytty S on MUUTOS_HEIKKO.
    Palauttaa dict (kuten irroitus_hanta) tai None (rata liian lyhyt / ei ehdokasta; kutsuja palaa kinkkihakuun). Y on korjattava tunnetuista paikkaan sidotuista mittausharhoista ennen (netti_seuranta.korjaa_hog_harha)."""
    min_jalkeen = MIN_JALKEEN_CM if min_jalkeen is None else min_jalkeen
    if len(t) < 30:
        return None
    ok = _poikkeamat(t, X, Y)
    t, X, Y = t[ok], X[ok], Y[ok]
    if len(t) < 25:
        return None
    Y_a = max(MUUTOS_Y_ALKU_CM, 0.0 if Y_r_min is None else Y_r_min)
    Y_b = float(Y.max()) - min_jalkeen
    if Y_r_max is not None:
        Y_b = min(Y_b, Y_r_max)
    if Y_b < Y_a + 20.0:
        return None
    ys = np.arange(Y_a, Y_b + 1e-9, MUUTOS_ASKEL_CM)
    ZY, ZX = _muutos_profiilit(t, X, Y, ys)
    SLY = -np.where(np.isfinite(ZY), ZY, 0.0) / MUUTOS_KYNNYS_Y                   # voimakkuus skaaloittain (rivit) ja kanavittain: kiihtyvyyden pudotus (Y) ...
    SLX = np.where(np.isfinite(ZX), np.abs(ZX), 0.0) / MUUTOS_KYNNYS_X            # ... ja kaarevuuden muutos (X)
    Y_t = _tyonto_loppu(t, Y, Y_a)                                                # kasi vaikuttaa viimeksi ikkunassa [Y_t - L, Y_t]: irroitus ei ole sita aikaisemmin
    jatkuu = Y_t is not None and Y_t >= float(Y.max()) - 2.0 * MUUTOS_ASKEL_CM
    if Y_t is not None and not jatkuu:
        sallittu = ys >= Y_t - MUUTOS_TYONTO_L_CM
        SLY, SLX = np.where(sallittu, SLY, 0.0), np.where(sallittu, SLX, 0.0)
    SL = np.maximum(SLY, SLX)
    env = SL.max(axis=0)                                                          # verhokayra yli skaalojen: ilmaisu
    if jatkuu:                                                                    # kivea tyonnetaan viela radan lopussa: irroitus ei ole viela tapahtunut
        k = len(ys) - 1
        sk = int(np.argmax(SL[:, k]))
        merkitseva = False
    else:
        paino = np.exp((ys - ys[-1]) / MUUTOS_LAMBDA_CM)                          # myohaisempi muutos voittaa, ellei se ole selvasti heikompi
        k = int(np.argmax(env * paino))
        merkitseva = bool(env[k] >= 1.0)
        if env[k] >= MUUTOS_HEIKKO:
            near = np.abs(ys - ys[k]) <= MUUTOS_PAIKANNUS_CM                      # paikannus hienoimmalla skaalalla, jolla muutos on edelleen selva
            sk = next((r for r in range(SL.shape[0]) if SL[r, near].max() >= MUUTOS_PAIKANNUS_OSUUS * env[k]), int(np.argmax(SL[:, k])))
            k = int(np.nonzero(near)[0][np.argmax(SL[sk, near])])
        elif Y_t is None:
            return None
        else:                                                                     # tyonto loppui tahan mennessa, mutta kinkki ei erotu: tyontoikkunan keskipiste
            k = int(np.argmin(np.abs(ys - min(Y_b, Y_t - 0.5 * MUUTOS_TYONTO_L_CM))))
            sk = int(np.argmax(SL[:, k]))
    raja = max(MUUTOS_VALI_OSUUS * float(SL[sk, k]), 0.3)
    a = b = k
    while a > 0 and SL[sk, a - 1] >= raja:
        a -= 1
    while b < len(ys) - 1 and SL[sk, b + 1] >= raja:
        b += 1
    Y_r = float(ys[k])
    kanava = "y" if SLY[sk, k] >= SLX[sk, k] else "x"
    z = MUUTOS_KYNNYS_Y * SLY[sk, k] if kanava == "y" else MUUTOS_KYNNYS_X * SLX[sk, k]
    return dict(jasenet=[], profiilit=[], historia=[Y_r], Y_r=Y_r, Y_min=float(ys[a]), Y_max=float(ys[b]), Y_r_Y_kanava=float(ys[int(np.argmax(SLY.max(axis=0) * np.exp((ys - ys[-1]) / MUUTOS_LAMBDA_CM)))]),
                Y_r_X_kanava=float(ys[int(np.argmax(SLX.max(axis=0) * np.exp((ys - ys[-1]) / MUUTOS_LAMBDA_CM)))]), dchi2_med=float(z * z), merkitsevia=int(merkitseva), jasenia=1, merkitseva=bool(merkitseva),
                menetelma="muutos", voimakkuus=float(SL[sk, k]), kanava=kanava, skaala_cm=float(MUUTOS_SKAALAT_CM[sk]), tyonto_jatkuu=bool(jatkuu), tyonto_loppu=Y_t, sigma_Y_cm=MUUTOS_SIGMA_Y_CM,
                sigma_X_cm=MUUTOS_SIGMA_X_CM, ikkuna_n=int((Y >= Y_r).sum()))


# ---------------------------------------------------------------------------------------------------------------------
# IRROITUSTILA
# ---------------------------------------------------------------------------------------------------------------------
def _paraabeli_arvo(u, y, u0, esi=None):
    """Paraabelisovitus (Huber) -> (arvo, 1. derivaatta, 2. derivaatta, virhe arvo, virhe derivaatta) kohdassa u0; virheet LS-kovarianssista (esitiedon kanssa a posteriori -kovarianssi).
    esi = (tavoite, sigma): pehmea esitieto 2. derivaatalle."""
    d = u - u0
    A = np.column_stack([np.ones_like(d), d, 0.5 * d ** 2])
    r0 = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
    s = max(1.4826 * float(np.median(np.abs(r0 - np.median(r0)))), 1e-3)
    p, _, _ = _irls(A, y, s, esi=None if esi is None else ((0.0, 0.0, 1.0), esi[0], esi[1]))
    r = y - A @ p
    s2 = max(float(np.sum(r ** 2)) / max(len(y) - 3, 1), 1e-12)
    info = A.T @ A
    if esi is not None:
        info = info + np.diag([0.0, 0.0, s2 / esi[1] ** 2])
    cov = np.linalg.inv(info) * s2
    return float(p[0]), float(p[1]), float(p[2]), float(np.sqrt(cov[0, 0])), float(np.sqrt(cov[1, 1]))


def irroitustila(t, X, Y, t_r, Y_vali=None, esitieto=False):
    """Vapaan lennon sovitus (t >= t_r + VIIVE_S): X(Y) ja Y(t) paraabelit (esitieto: hidastuvuus ja kaarevuus fysikaalisissa rajoissa). Tila hetkella t_r: paikka, suunta, nopeus,
    hidastuvuus, kaarevuus. Suunnan virhe = LS-virhe, vaihtokohdan epavarmuudesta (kaarevuus * puoli Y-valia) ja radan paikallisen suunnan (ikkuna 120 cm) hajonta paraabelin
    ympari (taustaviivojen ja logojen aiheuttama sivuttaishairio) neliosumma."""
    m = t >= t_r + VIIVE_S
    tt, x, y = t[m], X[m], Y[m]
    Yr, vY, aY, _, evY = _paraabeli_arvo(tt, y, t_r, (-100.0 * A_VAPAA_M_S2, 100.0 * SIGMA_A_VAPAA_M_S2) if esitieto else None)
    Xr, dXdY, cX, _, edXdY = _paraabeli_arvo(y, x, Yr, (0.0, SIGMA_KAAREVUUS_TILA_1_CM) if esitieto else None)
    nopeus = vY * np.sqrt(1.0 + dXdY ** 2)
    suunta = float(np.degrees(np.arctan(dXdY)))
    e_stat = float(np.degrees(edXdY / (1.0 + dXdY ** 2)))
    e_vali = float(np.degrees(abs(cX) * 0.5 * (Y_vali[1] - Y_vali[0]) / (1.0 + dXdY ** 2))) if Y_vali else 0.0
    yc, th = suunta_paikallinen(y, x)
    e_loc = float(np.sqrt(np.mean((th - np.degrees(np.arctan(dXdY + cX * (yc - Yr)))) ** 2))) if len(yc) >= 3 else 0.0
    return dict(t_r=float(t_r), X_cm=Xr, Y_cm=Yr, nopeus_virhe_m_s=float(evY * np.sqrt(1.0 + dXdY ** 2) / 100.0), suunta_deg=suunta, suunta_virhe_deg=float(np.sqrt(e_stat ** 2 + e_vali ** 2 + e_loc ** 2)),
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


def theil_sen(x, y):
    """Robusti suoran kulmakerroin (parittaisten kulmakertoimien mediaani)."""
    s = [(y[j] - y[i]) / (x[j] - x[i]) for i in range(len(x)) for j in range(i + 1, len(x)) if x[j] - x[i] > 0.2]
    return float(np.median(s))


def liuku_suureet(t, X, Y, la, ir, til):
    """Liun ja irroituksen suureet kayttajan kysymyksiin (kaikki heittopaan radasta):
      * liun suunta = X(Y):n pienimman neliosumman suora hackista (lahto) irroituspisteeseen (kulma Y-akseliin, + = oikealle heittajasta katsoen);
      * irroituksen suunta = vapaan lennon suunta irroitushetkella (irroitustila);
      * nopeus juuri ennen irroitusta = nopeus viimeisen NOPEUS_IKKUNA_S ajalta ennen irroitusta (Y(t):n suora sovitus);
      * nopeus irroituksen jalkeen = vapaan lennon sovituksen nopeus irroitushetkella;
      * liun hidastuvuus = -(kiihtyvyys) liu'un aikana tyonnon jalkeen: paikallisten nopeuksien Theil-Sen-suora (robusti), 90 % vali bootstrapista;
        negatiivinen hidastuvuus = kivi kiihtyy liu'un aikana (heittaja tyontaa).
    Irroitus on kinemaattinen: hetki, jonka jalkeen kasi ei enaa vaikuta rataan (ei sormien irtoamishetki)."""
    t_r, Y_r = ir["t_r"], til["Y_cm"]
    m = (Y >= la["Y_lahto"]) & (Y <= Y_r)
    c = np.polyfit(Y[m], X[m], 1)
    liun_suunta = float(np.degrees(np.arctan(c[0])))
    # nopeus juuri ennen irroitusta
    w = (t >= t_r - NOPEUS_IKKUNA_S) & (t <= t_r)
    cy, covy = np.polyfit(t[w] - t_r, Y[w], 1, cov=True)
    cx = np.polyfit(Y[w], X[w], 1)
    kerroin = np.sqrt(1.0 + cx[0] ** 2)
    v_ennen = float(cy[0] * kerroin / 100.0)
    e_ennen = float(np.sqrt(covy[0, 0]) * kerroin / 100.0)
    # liun hidastuvuus
    v = liukuva(t, Y, h=4, aste=1)[:, 1] / 100.0
    g = (t >= la["t_90"] + LIUKU_ALKU_VIIVE_S) & (t <= t_r - LIUKU_LOPPU_VIIVE_S)
    idx = np.where(g)[0]
    if len(idx) >= 4:
        a_liuku = theil_sen(t[g], v[g])
    else:                                                              # irroitus heti tyonnon jalkeen: liukuvaihetta ei ole mitattavaksi
        a_liuku = float("nan")
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(200 if len(idx) >= 8 else 0):
        ii = np.unique(rng.choice(idx, len(idx)))
        if len(ii) >= 8:
            bs.append(theil_sen(t[ii], v[ii]))
    vali = (-float(np.percentile(bs, 95)), -float(np.percentile(bs, 5))) if bs else (float("nan"), float("nan"))
    return dict(liun_suunta_deg=liun_suunta, liun_suora_kulmakerroin=float(c[0]), liun_suora_X_Y0_cm=float(c[1]), liun_suunta_alku_cm=float(la["Y_lahto"]), liun_suunta_loppu_cm=float(Y_r),
                irroituksen_suunta_deg=til["suunta_deg"], irroituksen_suunta_virhe_deg=til["suunta_virhe_deg"],
                nopeus_ennen_m_s=v_ennen, nopeus_ennen_virhe_m_s=e_ennen, nopeus_jalkeen_m_s=til["nopeus_m_s"],
                nopeus_jalkeen_virhe_m_s=til["nopeus_virhe_m_s"],
                liun_hidastuvuus_m_s2=-a_liuku, liun_hidastuvuus_vali_m_s2=list(vali), liun_ikkuna_s=[float(la["t_90"] + LIUKU_ALKU_VIIVE_S), float(t_r - LIUKU_LOPPU_VIIVE_S)],
                vapaan_lennon_hidastuvuus_m_s2=-til["hidastuvuus_m_s2"])


def analysoi(rata, kohinakatto=None, **asetus):
    """Yhden heiton koko tunnistus: dict. Reaaliaikaisessa (osittaisessa) aineistossa: kohinakatto = (sigma_Y, sigma_X) [cm] rajaa kohina-arvion ylhaalta
    (jos radan loppuosa ei ole vielakaan vapaata lentoa, MAD-arvio paisuu ja vaihtokohdan haku sokeutuu); asetus: irroitus_ikkuna (min_jalkeen, Y_r_min)."""
    t, X, Y = rata["t"], rata["X"], rata["Y"]
    a = liuku_alku(t, Y)
    sY, sX = kohinat(t, X, Y)
    if kohinakatto is not None:
        sY, sX = min(sY, kohinakatto[0]), min(sX, kohinakatto[1])
    ir = None
    menetelma = asetus.get("menetelma", MENETELMA)
    if menetelma == "muutos":                                                    # viimeinen merkitseva nopeuden / suunnan muutos; ei ehdokasta (lyhyt rata): kinkkihaku
        ir = irroitus_muutos(t, X, Y, Y_r_min=asetus.get("Y_r_min"), Y_r_max=asetus.get("Y_r_max", Y_IRROITUS_MAX_CM), min_jalkeen=asetus.get("min_jalkeen"))
    elif menetelma == "hanta":                                                   # vapaan lennon hanta, kun rata yltaa hog-viivan taakse (muuten kinkkihaku: osittainen aineisto)
        ya = asetus.get("Y_r_max")
        ir = (irroitus_hanta(t, X, Y, sY=max(sY, HANTA_SIGMA_Y_MIN), sX=max(sX, HANTA_SIGMA_X_MIN), Y_ankkuri=ya, Y_r_min=asetus.get("Y_r_min"), esitieto=asetus.get("esitieto", False))
              if ya is not None or Y_IRROITUS_MAX_CM is not None else None)
    if ir is None:
        ir = irroitus_varmistettu(t, X, Y, a["Y_90"], sY, sX, **{k: v for k, v in asetus.items() if k != "menetelma"})
    out = dict(liuku=a, kohina=dict(sigma_Y_cm=sY, sigma_X_cm=sX), irroitus=ir)
    if ir is not None:
        Y_r = ir["Y_r"]
        t_r = float(np.interp(Y_r, Y, t))
        ir["t_r"] = t_r
        ir["t_vali"] = [float(np.interp(ir["Y_min"], Y, t)), float(np.interp(ir["Y_max"], Y, t))]
        out["tila"] = irroitustila(t, X, Y, t_r, (ir["Y_min"], ir["Y_max"]), esitieto=asetus.get("esitieto", False))
        out["liuku"]["tyonto_kesto_s"] = a["t_90"] - a["t_lahto"]
        out["liuku"]["tyonto_matka_cm"] = a["Y_90"] - a["Y_lahto"]
        out["liuku"]["liuku_kesto_s"] = t_r - a["t_lahto"]
        out["liuku"]["liuku_matka_cm"] = out["tila"]["Y_cm"] - a["Y_lahto"]
        out["yhteenveto"] = liuku_suureet(t, X, Y, a, ir, out["tila"])
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
        a.set(xlabel="Y (cm)", ylabel="nopeus (m/s)", title="nopeus ja kiihtyvyys (kiihtyvyyden kohina ~+-0,2 m/s2); sin. = lahto / tyonto")
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
        ys = r["yhteenveto"]
        print(f"   liun suunta {ys['liun_suunta_deg']:+.2f} deg (hackista irroitukseen), irroituksen suunta {ys['irroituksen_suunta_deg']:+.2f} +- {ys['irroituksen_suunta_virhe_deg']:.2f} deg; "
              f"nopeus juuri ennen irroitusta {ys['nopeus_ennen_m_s']:.2f} +- {ys['nopeus_ennen_virhe_m_s']:.2f} m/s, irroituksen jalkeen {ys['nopeus_jalkeen_m_s']:.2f} +- {ys['nopeus_jalkeen_virhe_m_s']:.2f} m/s; "
              f"liun hidastuvuus {ys['liun_hidastuvuus_m_s2']:+.3f} m/s2 (90 % vali {ys['liun_hidastuvuus_vali_m_s2'][0]:+.3f}...{ys['liun_hidastuvuus_vali_m_s2'][1]:+.3f}), vapaan lennon hidastuvuus {ys['vapaan_lennon_hidastuvuus_m_s2']:+.3f} m/s2")
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
