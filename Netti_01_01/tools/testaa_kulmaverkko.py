"""Kahvan kulman neuroverkon (kulmaverkko.py) yksikkotestit: kuvan kierto / peilaus -konventio, NumPy-verkon tulos sama kuin opetuksessa kaytetyn JAX-verkon (kultatiedostot), kiertoratasovitus (lineaarisesti
hidastuva kierto) synteettisella radalla, ihmisen viivat epatayllisina (robusti sovitus, jalostus; ei videota, ei JAX:ia).

    python tools/testaa_kulmaverkko.py

Kultatiedostot (kierre_validointi/kulmaverkko_testi_*.npz): pienet satunnaisverkot ('pieni' ja 'resf'), syoteleikkeet ja JAX-verkon (tools/kulmaverkko_opeta.py) tulos, myos TTA:lla. Jos verkon rakenne tai
kierto-/peilauskonventio muuttuu, nama testit epaonnistuvat ennen kuin opetettu paino lakkaa toimimasta. Poistumiskoodi 1, jos jokin testi epaonnistuu."""
import os
import sys

import cv2
import numpy as np

KANSIO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, KANSIO)

import kulmaverkko as kv  # noqa: E402

VIRHEITA = []


def tarkista(nimi, ehto, teksti=""):
    print(("  OK   " if ehto else "  VIRHE ") + nimi + (f": {teksti}" if teksti else ""))
    if not ehto:
        VIRHEITA.append(nimi)


def _palkin_kulma(kuva):
    """Kirkkaiden pikselien paaakseli (aste, jakso 180, 0 = vaakasuora, + = vastapaivaan nayton mukaan)."""
    g = kuva.astype(np.float64).sum(axis=2)
    ys, xs = np.where(g > g.max() * 0.5)
    c = (kuva.shape[0] - 1) / 2.0
    x, y = xs - c, -(ys - c)
    return (0.5 * np.degrees(np.arctan2(2 * (x * y).mean(), (x * x).mean() - (y * y).mean()))) % 180.0


def _palkki(psi, n=kv.KOKO):
    kuva = np.zeros((n, n, 3), np.uint8)
    a = np.radians(psi)
    c = (n - 1) / 2.0
    cv2.line(kuva, (int(round(c + 18 * np.cos(a))), int(round(c - 18 * np.sin(a)))), (int(round(c - 18 * np.cos(a))), int(round(c + 18 * np.sin(a)))), (255, 255, 255), 3)
    return kuva


def testaa_kierto_ja_peilaus():
    """kierra(kuva, a): palkin kulma kasvaa a:n verran; peilaa: theta -> -theta."""
    pahin = 0.0
    for psi in (0, 20, 45, 75, 110, 160):
        k0 = _palkki(psi)
        for a in (-60, 25, 80):
            pahin = max(pahin, abs(kv.ero180(_palkin_kulma(kv.kierra(k0, a)), _palkin_kulma(k0) + a)))
        pahin = max(pahin, abs(kv.ero180(_palkin_kulma(kv.peilaa(k0)), -_palkin_kulma(k0))))
    tarkista("kierto ja peilaus siirtavat palkin kulmaa odotetusti", pahin < 1.0, f"suurin virhe {pahin:.2f} astetta")


def testaa_verkko_kultatiedostot():
    """NumPy-verkko antaa saman jakauman kuin opetuksessa kaytetty JAX-verkko (pieni ja jaannosverkko), ilman TTA:ta ja TTA:lla."""
    for nimi in ("pieni", "resf"):
        polku = os.path.join(KANSIO, "kierre_validointi", f"kulmaverkko_testi_{nimi}.npz")
        d = np.load(polku, allow_pickle=False)
        malli = kv.lataa(polku)
        x = d["tiles"].astype(np.float32) / 255.0 - 0.5
        e_raaka = float(np.abs(kv.etenemista(malli, x) - d["p_raw"]).max())
        e_tta = float(np.abs(kv.ennusta(malli, d["tiles"], tta=True) - d["p_tta"]).max())
        tarkista(f"verkko '{nimi}': NumPy = JAX ({d['p_raw'].shape[0]} leikettä)", e_raaka < 1e-4 and e_tta < 1e-4 and d["p_raw"].max() > 0.5, f"ero {e_raaka:.1e} (raaka), {e_tta:.1e} (TTA)")


def testaa_tallenna_lataa_ja_yhdistelma():
    """tallenna / lataa toistaa verkon; kahden verkon jakauma on geometrinen keskiarvo; yksi verkko listassa = sama kuin sellaisenaan."""
    import tempfile
    polku = os.path.join(KANSIO, "kierre_validointi", "kulmaverkko_testi_pieni.npz")
    d = np.load(polku, allow_pickle=False)
    m1 = kv.lataa(polku)
    with tempfile.TemporaryDirectory() as hak:
        p = os.path.join(hak, "m.npz")
        kv.tallenna(p, m1["arch"], m1["P"], m1["blocks"])
        m1b = kv.lataa(p)
        tarkista("tallenna / lataa toistaa verkon", m1b["arch"] == m1["arch"] and all(np.array_equal(m1b["P"][k], m1["P"][k]) for k in m1["P"]))
    m2 = dict(m1, P=dict(m1["P"], **{"d2/w": m1["P"]["d2/w"] * 0.8}))                      # samankaltainen toinen verkko (kaksi satunnaisverkkoa olisi eri mieltä kaikesta)
    a = kv.ennusta(m1, d["tiles"], tta=False)
    b = kv.ennusta([m1], d["tiles"], tta=False)
    ab = kv.ennusta([m1, m2], d["tiles"], tta=False)
    g = np.sqrt(kv.ennusta(m1, d["tiles"], tta=False) * kv.ennusta(m2, d["tiles"], tta=False))
    g /= g.sum(axis=1, keepdims=True)
    tarkista("verkkojen yhdistelma = geometrinen keskiarvo, summa 1", np.allclose(a, b) and np.abs(ab - g).max() < 1e-5 and np.allclose(ab.sum(axis=1), 1.0, atol=1e-5), f"ero {np.abs(ab - g).max():.1e}")


def testaa_mallikohtainen_tta():
    """Verkon oma oletus-TTA (`kierrot` tiedostossa): kierrotta opetettu verkko (kierrot = (0,)) ennustaa oletuksena vain peilauksella eikä kierretyillä näkymillä; tallenna / lataa toistaa asetuksen; yhdistelmässä
    jokainen verkko käyttää omaa oletustaan; kierrot-argumentti ohittaa oletuksen."""
    import tempfile
    polku = os.path.join(KANSIO, "kierre_validointi", "kulmaverkko_testi_pieni.npz")
    d = np.load(polku, allow_pickle=False)
    X = d["tiles"][:6]
    m1 = kv.lataa(polku)
    tarkista("ilman kierrot-kenttää verkon oletus on TTA_KIERROT", "kierrot" not in m1)
    with tempfile.TemporaryDirectory() as hak:
        p = os.path.join(hak, "m.npz")
        kv.tallenna(p, m1["arch"], m1["P"], m1["blocks"], kierrot=(0.0,))
        m0 = kv.lataa(p)
    tarkista("tallenna / lataa toistaa verkon oletus-TTA:n", m0.get("kierrot") == (0.0,) and all(np.array_equal(m0["P"][k], m1["P"][k]) for k in m1["P"]), f"kierrot {m0.get('kierrot')}")
    oletus = kv.ennusta(m0, X)
    vain_peilaus = kv.ennusta(m1, X, kierrot=(0.0,))
    kaikki = kv.ennusta(m1, X)
    tarkista("kierrotta opetettu verkko: oletus = vain peilaus (ei kierrettyjä näkymiä)", np.allclose(oletus, vain_peilaus, atol=1e-6) and np.abs(oletus - kaikki).max() > 1e-4,
             f"ero vain peilaukseen {np.abs(oletus - vain_peilaus).max():.1e}, ero täyteen TTA:han {np.abs(oletus - kaikki).max():.1e}")
    tarkista("kierrot-argumentti ohittaa verkon oletuksen", np.allclose(kv.ennusta(m0, X, kierrot=kv.TTA_KIERROT), kaikki, atol=1e-6))
    yhd = kv.ennusta([m0, m1], X)
    g = np.sqrt(vain_peilaus * kaikki)
    g /= g.sum(axis=1, keepdims=True)
    tarkista("yhdistelmässä kukin verkko käyttää omaa TTA:taan", np.abs(yhd - g).max() < 1e-5, f"ero {np.abs(yhd - g).max():.1e}")


def _jakaumat(kulmat, kohina, rng, kappa=20.0):
    havaittu = (kulmat + rng.normal(0, kohina, len(kulmat))) % 180.0
    p = np.exp(np.cos(np.deg2rad(2.0 * (kv.KUL[None, :] - havaittu[:, None]))) * kappa)
    return p / p.sum(axis=1, keepdims=True)


def testaa_johdonmukaisuus():
    """Johdonmukaisuus ilman merkintöjä (`kulmaverkko.johdonmukaisuus`): tarkalla radalla ja 4 asteen ruutukohinalla ruutujen ero sovitettuun rataan on noin 3 astetta; jos 10 % ruuduista on aivan väärin, ne näkyvät yli 10 asteen osuudessa
    mutta mediaani pysyy."""
    rng = np.random.default_rng(5)
    jak, ajat = [], []
    for _ in range(8):
        t = 3000.0 + np.arange(30) / 30.0
        kulmat = (rng.uniform(0, 180) + rng.choice([-1, 1]) * rng.uniform(50, 150) * kv.rata_aika(t - t[0])) % 180.0
        jak.append(_jakaumat(kulmat, 4.0, rng))
        ajat.append(t)
    r = kv.johdonmukaisuus(jak, ajat)
    tarkista("johdonmukaisuus: tarkka rata + 4 asteen ruutukohina -> ero rataan noin 3 astetta, suunta varma", 2.2 < r["ero"].mean() < 4.0 and r["ero"].shape == (240,) and (r["suunta_p"] > 0.99).all() and r["heitto"].max() == 7,
             f"ero keski {r['ero'].mean():.2f}, suunta-p vähintään {r['suunta_p'].min():.3f}")
    jak2 = []
    for p in jak:
        p = p.copy()
        idx = rng.choice(30, 3, replace=False)
        p[idx] = _jakaumat(rng.uniform(0, 180, 3), 1.0, rng)
        jak2.append(p)
    r2 = kv.johdonmukaisuus(jak2, ajat)
    tarkista("johdonmukaisuus: 10 % ruuduista väärin -> näkyy yli 10 asteen osuudessa, mediaani pysyy", 0.05 < np.mean(r2["ero"] > 10) < 0.16 and np.median(r2["ero"]) < 3.8,
             f"yli 10 astetta {np.mean(r2['ero'] > 10) * 100:.1f} %, mediaani {np.median(r2['ero']):.2f}")


def testaa_vyohyke():
    """Hog-viivan vyöhyke (kiven y 682,8-702,6 cm, ruudut joissa viiva on kahvan takana): `vyohykkeessa` rajat; `sovita_heitto(ohita=...)` jättää ruutujen jakaumat pois sovituksesta mutta antaa niille radan kulman, ja kun vyöhykkeen
    ruutujen arvio on vedetty kohti vaakasuuntaa, rata palautuu ohituksella; `johdonmukaisuus` näyttää ohitetut ruudut; `pseudo_aineisto` pitää ohitetut ruudut opetuksessa radan kulmalla."""
    v = kv.vyohykkeessa([682.7, 682.8, 690.0, 702.6, 702.7])
    tarkista("vyohykkeessa: rajat 682,8 ja 702,6 cm mukana", v.tolist() == [False, True, True, True, False], str(v.tolist()))
    rng = np.random.default_rng(11)
    n = 10
    t = 3000.0 + np.arange(n) / 25.0
    Y = 684.0 + np.arange(n) * 6.0                      # 684, 690, 696, 702 vyöhykkeessä, loput ulkopuolella
    z = kv.vyohykkeessa(Y)
    tosi = (40.0 - 70.0 * kv.rata_aika(t - t[0])) % 180.0
    havaittu = tosi.copy()
    havaittu[z] -= 16.0                                 # vyöhykkeen ruudut vetäytyvät kohti vaakasuuntaa
    p = _jakaumat(havaittu, 1.5, rng)
    r0 = kv.sovita_heitto(p, t)
    r1 = kv.sovita_heitto(p, t, ohita=z)
    e0, e1 = np.abs(kv.ero180(r0["kulma"], tosi)).mean(), np.abs(kv.ero180(r1["kulma"], tosi)).mean()
    tarkista("ohita: vyöhykkeen harhaiset ruudut pois -> rata palautuu (virhe koko heitossa, myös ohitetuissa ruuduissa)", e1 < 2.0 and e0 > e1 + 1.0 and r1["ohitettu"].tolist() == z.tolist() and np.isfinite(r1["kulma"]).all(), f"virhe ilman ohitusta {e0:.2f}, ohituksella {e1:.2f}")
    tarkista("ohita = ei yhtään ruutua: sama tulos kuin ilman", np.allclose(kv.sovita_heitto(p, t, ohita=np.zeros(n, bool))["kulma"], r0["kulma"]), "")
    r = kv.johdonmukaisuus([p, p], [t, t], ohita=[z, z])
    tarkista("johdonmukaisuus: ohitetut ruudut näkyvät ja niiden ero rataan on suuri", r["ohitettu"].sum() == 2 * z.sum() and r["ero"][r["ohitettu"]].mean() > 3 * r["ero"][~r["ohitettu"]].mean(),
             f"ohitetut {r['ero'][r['ohitettu']].mean():.1f}, muut {r['ero'][~r['ohitettu']].mean():.1f} astetta")
    import kulmaverkko_opeta as ko
    kulma = np.linspace(40, 20, n)
    resid = np.where(z, 40.0, 2.0)
    L = {("T", 1): dict(f=np.arange(n), tiles=np.zeros((n, 56, 56, 3), np.uint8))}
    PS = {("T", 1): dict(kulma=kulma, R=np.full(n, 0.99), residuaali=resid, suunta_p=1.0, ohitettu=z)}
    X, TH, SIG, W, heittoja = ko.pseudo_aineisto(L, PS, min_ruutuja=8)
    PS0 = {("T", 1): dict(kulma=kulma, R=np.full(n, 0.99), residuaali=resid, suunta_p=1.0)}
    try:
        X0 = ko.pseudo_aineisto(L, PS0, min_ruutuja=8)[0]
        ilman = len(X0)
    except ValueError:                                  # vanha pickle (ei ohitettu-kenttää): vyöhykkeen ruudut pudotetaan residuaalin vuoksi, heitto jää liian lyhyeksi
        ilman = 0
    tarkista("pseudo_aineisto: ohitetut ruudut säilyvät radan kulmalla (vanhassa pickle-muodossa ne putoaisivat)", len(X) == n and heittoja == 1 and np.allclose(TH, kulma) and ilman < n, f"mukana {len(X)} / {n} ruutua, ilman ohitettu-kenttää {ilman}")


def testaa_vakiokierto():
    """Kiertoratasovitus (kulma on aikafunktio, kierto hidastuu lineaarisesti): kulmat, omega ja suunta palautuvat; peilattu rata vaihtaa suunnan; yksittainen 90 asteen virheruutu ei vie sovitusta, mutta nakyy residuaalissa."""
    rng = np.random.default_rng(1)
    t = 3000.0 + np.arange(40) * 0.04
    G = (t - t[0]) - 0.5 * kv.K_KIERRE * (t - t[0]) ** 2                   # lineaarisesti hidastuva kierto (omega = omega_0 (1 - K s)), kirjoitettu auki: ei kv.rata_aika:a
    for om_tosi in (110.0, -70.0, 25.0, 300.0):
        th = (20.0 + om_tosi * G) % 180.0
        r = kv.sovita_heitto(_jakaumat(th, 6.0, rng), t)
        e = np.abs(kv.ero180(r["kulma"], th))
        tarkista(f"vakiokierto omega {om_tosi:+.0f} aste/s", e.mean() < 2.5 and abs(r["omega"] - om_tosi) < max(6.0, 0.06 * abs(om_tosi)) and r["suunta_p"] > 0.99 and (r["omega"] > 0) == (om_tosi > 0),
                 f"kulmavirhe keski {e.mean():.2f}, omega {r['omega']:+.1f} +- {r['omega_sd']:.1f}, suunta-p {r['suunta_p']:.3f}")
    th = (20.0 + 110.0 * G) % 180.0
    p = _jakaumat(th, 6.0, rng)
    paha = p.copy()
    paha[17] = _jakaumat(np.array([(th[17] + 90.0) % 180.0]), 0.0, rng, kappa=40.0)[0]      # yksi ruutu 90 astetta vaarassa kulmassa, varmana
    r = kv.sovita_heitto(paha, t)
    tarkista("yksittainen virheruutu ei vie sovitusta", abs(kv.ero180(r["kulma"][17], th[17])) < 6.0 and r["residuaali"][17] > 60.0 and np.abs(kv.ero180(r["kulma"], th)).mean() < 3.0,
             f"virheruudun sovitusvirhe {abs(kv.ero180(r['kulma'][17], th[17])):.1f}, residuaali {r['residuaali'][17]:.0f}")


def testaa_rata_aika():
    """rata_aika(s) = s - K s^2 / 2: alussa nopeus 1, hidastuvuus K (omega(s) = omega_0 (1 - K s)), pysähtyy hetkellä 1 / K, ei kasva sen jälkeen."""
    K, h = kv.K_KIERRE, 1e-4
    g = kv.rata_aika
    nopeus = (g(h) - g(-h)) / (2 * h)
    kiihtyvyys = (g(1.0 + h) - 2 * g(1.0) + g(1.0 - h)) / h ** 2
    tarkista("kiertorata: nopeus 1 alussa, hidastuvuus K, lineaarinen hidastuminen", abs(nopeus - 1.0) < 1e-6 and abs(kiihtyvyys + K) < 1e-4 and abs(g(1.0 / K) - 0.5 / K) < 1e-9 and g(3.0 / K) == g(1.0 / K),
             f"g'(0) = {nopeus:.6f}, g'' = {kiihtyvyys:.5f} (K = {K}), g(1/K) = {g(1.0 / K):.3f}")
    a = kv.rata_aika(np.array([0.0, 0.5, 1.0]))
    tarkista("kiertorata: arvot", np.allclose(a, [0.0, 0.5 - 0.125 * K, 1.0 - 0.5 * K]), str(a))


def _verkon_jakaumat(th, kohina, rng, kappa=12.0, harha=None):
    hav = th + rng.normal(0, kohina, len(th))
    if harha is not None:
        hav = hav + harha
    p = np.exp(np.cos(np.deg2rad(2.0 * (kv.KUL[None, :] - hav[:, None]))) * kappa)
    return p / p.sum(axis=1, keepdims=True)


def testaa_viivat_epatayllisina():
    """Ihmisen viivat epätäydellisinä: (1) yksi aivan väärä viiva ei vie rataa (kulma väärän viivan ruudussa pysyy lähellä oikeaa), (2) viivat parantavat verkon harhaista todistetta, (3) pelkistä viivoista (ei verkkoa) rata palautuu,
    (4) viivarangaistus on rajoitettu ylhäältä, (5) posteriorin huipun massa kertoo ruudun luotettavuuden."""
    rng = np.random.default_rng(5)
    t = 3000.0 + np.arange(30) * 0.04
    th = (30.0 - 120.0 * kv.rata_aika(t - t[0])) % 180.0
    idx = np.array([3, 9, 15, 21, 27])
    a = th[idx] + np.array([1.5, -2.0, 2.5, -1.0, 40.0])         # viimeinen viiva 40 astetta pielessa
    harha = np.zeros(len(t))
    harha[-8:] = 25.0                                                   # verkko on 8 loppuruudussa 25 astetta pielessa
    p = _verkon_jakaumat(th, 8.0, rng, harha=harha)
    r0 = kv.sovita_heitto(p, t)
    r1 = kv.sovita_heitto(p, t, viivat=dict(idx=idx, kulma=a))
    e0, e1 = np.abs(kv.ero180(r0["kulma"], th)), np.abs(kv.ero180(r1["kulma"], th))
    tarkista("viivat parantavat verkon harhaista todistetta", e1.mean() < e0.mean() - 1.5 and e1[-8:].mean() < e0[-8:].mean() - 3.0, f"keskivirhe {e0.mean():.2f} -> {e1.mean():.2f}, loppuruudut {e0[-8:].mean():.2f} -> {e1[-8:].mean():.2f}")
    tarkista("aivan väärä viiva ei vie rataa", abs(kv.ero180(r1["kulma"][27], th[27])) < 20.0 and abs(kv.ero180(r1["kulma"][27], a[4])) > 15.0, f"väärän viivan ruudussa virhe radasta {abs(kv.ero180(r1['kulma'][27], th[27])):.1f}, erosta viivaan {abs(kv.ero180(r1['kulma'][27], a[4])):.1f}")
    r2 = kv.sovita_heitto(None, t, viivat=dict(idx=idx[:4], kulma=a[:4]))
    e2 = np.abs(kv.ero180(r2["kulma"], th))
    tarkista("pelkistä viivoista rata palautuu", e2.mean() < 3.0 and abs(r2["omega"] + 120.0) < 35.0, f"keskivirhe {e2.mean():.2f}, omega {r2['omega']:+.0f} +- {r2['omega_sd']:.0f}")
    ero = np.array([0.0, 4.0, 8.0, 12.0, 25.0, 40.0, 80.0])
    pen = kv.viiva_rangaistus(ero)
    tarkista("viivarangaistus: nolla nollassa, kasvaa, rajoitettu ylhäältä", abs(pen[0]) < 1e-9 and np.all(np.diff(pen[:4]) > 0) and abs(pen[-1] - pen[-2]) < 1e-9 and pen[-1] < 8.0, f"{np.round(pen, 2)}")
    tarkista("posteriorin huipun massa: korkea hyvin määrätyillä ruuduilla", np.median(r1["massa_moodi"]) > 0.9 and np.all(r1["sigma_moodi"] >= 1.0) and np.all(r1["massa_moodi"] <= 1.0 + 1e-9),
             f"massa mediaani {np.median(r1['massa_moodi']):.2f}, huipun hajonta mediaani {np.median(r1['sigma_moodi']):.1f}")


def testaa_jalosta():
    """tools/kulmaverkko_opeta.jalosta: ihmisen viivat + rata -> jalostetut kulmat heiton ruuduille; väärä viiva ilmoitetaan, jalostettu aineisto kokoaa oikeat ruudut (synteettinen heitto, ei videota, ei JAX:ia)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import kulmaverkko_opeta as ko
    rng = np.random.default_rng(2)
    t = 500.0 + np.arange(24) * 0.04
    th = (80.0 + 90.0 * kv.rata_aika(t - t[0])) % 180.0
    L = {("Testi", 1): dict(t=t, f=np.arange(24), tiles=(rng.random((24, kv.KOKO, kv.KOKO, 3)) * 255).astype(np.uint8))}
    idx = [1, 8, 15, 21]
    viivat = [(j, float((th[j] + d) % 180.0), 1.0, f"Testi_h1_f{j}") for j, d in zip(idx, (1.0, -2.0, 1.5, 35.0))]     # viimeinen viiva 35 astetta pielessa
    V = {("Testi", 1): dict(viivat=viivat, ankkuri=None, rata=None)}
    JAL = ko.jalosta(L, V)
    j = JAL[("Testi", 1)]
    epailyttavat = [id_ for id_, ihm, ero in j["viivat"] if abs(ero) > 2.5 * kv.SIGMA_VIIVA]
    e = np.abs(kv.ero180(j["kulma"], th[j["idx"]]))
    tarkista("jalosta: kulmat radalla, ruutuja riittävästi", len(j["idx"]) >= 15 and e.mean() < 4.0, f"{len(j['idx'])} / 24 ruutua, keskivirhe {e.mean():.2f}")
    tarkista("jalosta: väärä viiva ilmoitetaan, muut eivät", epailyttavat == ["Testi_h1_f21"], f"epäilyttävät: {epailyttavat}")
    X, TH, SIG, W = ko.jalostettu_aineisto(L, JAL)
    tarkista("jalostettu aineisto: ruudut, kulmat, hajonnat, painot", len(X) == len(TH) == len(SIG) == len(W) == len(j["idx"]) and np.array_equal(X, L[("Testi", 1)]["tiles"][j["idx"]]) and np.all(SIG >= ko.SIGMA_JAL0), f"{len(X)} ruutua, hajonta {SIG.min():.1f}-{SIG.max():.1f}")


def testaa_opetuskoodi_jax():
    """Opetuskoodin (tools/kulmaverkko_opeta.py, JAX) verkko laskee samaa kuin ennustuskoodi (NumPy), ja opetus (muutama askel pienellä verkolla) ajaa ja vie verkon NumPy-muotoon. Ohitetaan, jos JAX puuttuu."""
    try:
        import jax  # noqa: F401
        import optax  # noqa: F401
    except ImportError:
        print("  (ohitettu) opetuskoodi: JAX / optax puuttuu")
        return
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import kulmaverkko_opeta as ko
    import jax.numpy as jnp
    rng = np.random.default_rng(8)
    tiles = (rng.random((4, kv.KOKO, kv.KOKO, 3)) * 255).astype(np.uint8)
    for verkko, kw in (("pieni", dict(ch=(4, 8, 16), hid=16)), ("resf", dict(ch=(8, 12, 16), blocks=1, hid=16))):
        P = ko.alusta(verkko, 3, **kw)
        for k in list(P):
            if k.endswith("_c2"):
                P[k] = dict(w=rng.normal(0, 0.1, P[k]["w"].shape).astype(np.float32), b=P[k]["b"])      # nollaalustus pois: jäännöshaara mukaan
        P["d2"] = dict(w=np.asarray(P["d2"]["w"]) * 15.0, b=np.asarray(P["d2"]["b"]))
        d = dict(E=P, verkko=verkko, kw=kw)
        arch, Pn, blocks = ko.npz_muotoon(d)
        malli = dict(arch=arch, blocks=blocks, P=Pn)
        x = tiles.astype(np.float32) / 255.0 - 0.5
        p_jax = np.asarray(jax.nn.softmax(ko.eteenpain(verkko, P, jnp.asarray(x), blocks=blocks)))
        e1 = float(np.abs(kv.etenemista(malli, x) - p_jax).max())
        e2 = float(np.abs(ko.jax_ennustaja(d)(tiles, tta=True) - kv.ennusta(malli, tiles, tta=True)).max())
        tarkista(f"opetuskoodin verkko '{verkko}' = ennustuskoodi", e1 < 1e-4 and e2 < 1e-4, f"ero {e1:.1e} (raaka), {e2:.1e} (TTA)")
    # opetus ajaa: 4 askelta, painot palautuvat numpy-taulukkoina ja ennustus on jakauma
    TH = rng.uniform(0, 180, 16)
    X = (rng.random((16, kv.KOKO, kv.KOKO, 3)) * 255).astype(np.uint8)
    E = ko.opeta(X, TH, np.full(16, 4.5), np.ones(16), "pieni", 4, 8, 1e-3, 0, dict(ch=(4, 8, 16), hid=16), loki=None)
    p = kv.ennusta(dict(arch="pieni", blocks=2, P=ko.npz_muotoon(dict(E=E, verkko="pieni", kw={}))[1]), X[:3], tta=False)
    tarkista("opetus ajaa ja verkko vie NumPy-muotoon", np.allclose(p.sum(axis=1), 1.0, atol=1e-5) and p.shape == (3, kv.NB), f"jakauman muoto {p.shape}")


def testaa_toimitetut_painot():
    """Repon mukana toimitetut painot (kulmaverkko_paino_*.npz) latautuvat, antavat jakauman ja sopivat ihmisen merkitsemiin ruutuihin (kierre_validointi/kulmaverkko_ihmisen_ruudut.npz, 39 viivaa). Lopullisten verkkojen opetuksessa ovat
    mukana nämä viivat (jalostettuina: rata hylkää kolme viivaa, joten kaikkia ei toisteta), joten tämä on toimivuustarkistus (paino ei ole rikki, kulmakonventio oikein) eikä riippumaton testi; riippumaton arvio on README:n protokolla
    (`kulmaverkko_opeta.py arvioi --pois ...`)."""
    import glob
    polut = sorted(glob.glob(os.path.join(KANSIO, "kulmaverkko_paino_*.npz")))
    d = np.load(os.path.join(KANSIO, "kierre_validointi", "kulmaverkko_ihmisen_ruudut.npz"))
    tarkista("toimitettuja painoja löytyy", len(polut) > 0, ", ".join(os.path.basename(p) for p in polut))
    mallit = [kv.lataa(p) for p in polut]
    for polku, malli in list(zip(polut, mallit)) + ([("kaikki yhdessä", mallit)] if len(mallit) > 1 else []):
        p = kv.ennusta(malli, d["tiles"])
        e = np.abs(kv.ero180(kv.kulma_jakaumasta(p)[0], d["kulma"]))
        tarkista(f"painot {os.path.basename(polku)}: ihmisen viivat keskimäärin alle 6 astetta, enintään 3 viivaa yli 12 asteen, ei yli 30", p.shape == (len(d["tiles"]), kv.NB) and np.allclose(p.sum(axis=1), 1.0, atol=1e-4)
                 and e.mean() < 6.0 and np.median(e) < 4.5 and (e > 12.0).sum() <= 3 and e.max() < 30.0, f"keskivirhe {e.mean():.1f}, mediaani {np.median(e):.1f}, suurin {e.max():.1f} astetta, yli 12 asteen {(e > 12.0).sum()} ({len(e)} viivaa)")


def main():
    for t in (testaa_opetuskoodi_jax, testaa_kierto_ja_peilaus, testaa_verkko_kultatiedostot, testaa_tallenna_lataa_ja_yhdistelma, testaa_mallikohtainen_tta, testaa_johdonmukaisuus, testaa_vyohyke, testaa_rata_aika, testaa_vakiokierto, testaa_viivat_epatayllisina, testaa_jalosta, testaa_toimitetut_painot):
        t()
    print()
    if VIRHEITA:
        print(f"{len(VIRHEITA)} testia epaonnistui: {', '.join(VIRHEITA)}")
        return 1
    print("kaikki testit OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
