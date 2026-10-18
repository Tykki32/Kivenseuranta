"""Yksikkotestit kaukopaan seurannan jatkokasittelylle: kiven vakioiden sovitus (netti_vakiot), kierteen yhdistelma (netti_kierre.yhdista), pysahtymishetki ja kohteen vaihtuminen
(netti_seuranta), joukkueiden kello (Analyysi.kello / kello_nyt).

    python tools/testaa_vakiot.py

Vakioiden sovitus testataan simuloidulla radalla (rata_malli.simuloi): tunnetut A, kappa, v0, omega0 -> rata 25 r/s, kohina 2 cm (B) -> sovitus palauttaa A:n ja kappan.
Poistumiskoodi 1, jos jokin testi epaonnistuu."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import netti_kierre as nk  # noqa: E402
import netti_seuranta as ns  # noqa: E402
import liuku_irroitus as li  # noqa: E402
import netti_vakiot as nv  # noqa: E402
import rata_malli as rm  # noqa: E402

VIRHEITA = []


def tarkista(nimi, ehto, teksti=""):
    print(("  OK   " if ehto else "  VIRHE ") + nimi + (f": {teksti}" if teksti else ""))
    if not ehto:
        VIRHEITA.append(nimi)


def simuloitu_rata(A, kappa, v0, w0, s, kohina=2.0, fps=25.0, siemen=1):
    """Kaukopaan rata (t, X', Y', vahva) hog-viivalta (Y' = 646) pysahtymiseen; tunnetut A, kappa, v0 (m/s), w0 (rad/s), kaarron suunta s (+-1 X':ssa)."""
    P = np.array([[646.0, v0, w0, A, kappa]])
    t, Y = rm.simuloi(P, 0.0, 0.0, pysahdykseen=True)
    Y = Y[:, :, 0]
    tt = np.arange(0.0, t[-1], 1.0 / fps)
    y = np.interp(tt, t, Y[:, 0])
    S = np.interp(tt, t, Y[:, 3])
    H = np.interp(tt, t, Y[:, 4])
    x = 60.0 - 0.01 * S + s * H
    rng = np.random.default_rng(siemen)
    return tt, x + rng.normal(0, kohina, len(tt)), y + rng.normal(0, kohina, len(tt)), np.ones(len(tt), bool), float(Y[-1, 0])


def testaa_vakiot():
    print("kiven vakiot (netti_vakiot.sovita), simuloitu rata")
    for A, kappa, v0, w0, s in ((0.0062, 2.6, 0.85, 1.0, -1), (0.0075, 3.5, 1.0, 1.3, 1), (0.0055, 2.0, 0.75, 0.8, -1)):
        t, X, Y, vahva, y_pys = simuloitu_rata(A, kappa, v0, w0, s)
        om_mit = (float(t[len(t) // 2]), w0 * 0.97, 0.1)                    # kierron mittaus (3 % pieleen)
        r = nv.sovita(t, X, Y, vahva=vahva, omega=om_mit)
        tarkista(f"A {A} kappa {kappa}: sovitus onnistuu", bool(r.get("ok")), r.get("syy") or "")
        if r.get("ok"):
            tarkista("  A osuu (+-8 %)", abs(r["A"] - A) <= 0.08 * A, f"{r['A']:.4f} vs {A}")
            tarkista("  kappa osuu (+-20 %)", abs(r["kappa"] - kappa) <= 0.2 * kappa, f"{r['kappa']:.2f} vs {kappa}")
            tarkista("  kaarron suunta", r["kaarto_merkki"] == s)
            tarkista("  pysahtymispaikka (Y +-10 cm)", abs(r["ennuste"]["pysahtyy_Y_cm"] - y_pys) <= 10.0, f"{r['ennuste']['pysahtyy_Y_cm']:.0f} vs {y_pys:.0f}")
            tarkista("  kappa_ok (kierre rajattu)", bool(r["kappa_ok"]))
    t, X, Y, vahva, _ = simuloitu_rata(0.0062, 2.6, 0.85, 1.0, -1)
    r = nv.sovita(t, X, Y, vahva=vahva, omega=None)
    tarkista("ilman kierron mittausta A silti sovitetaan, kappa ei ole luotettava", bool(r.get("ok")) and not r["kappa_ok"])
    r = nv.sovita(t[:50], X[:50], Y[:50], vahva=vahva[:50])
    tarkista("liian lyhyt rata hylataan", not r["ok"], r.get("syy", ""))
    e = nv.ennusta(0.0062, 2.6, 646.0, 0.85, 1.0, -1, x0_cm=60.0)
    t2, X2, Y2, v2, y_pys2 = simuloitu_rata(0.0062, 2.6, 0.85, 1.0, -1, kohina=0.0)
    tarkista("ennusta: pysahtymispaikka (Y)", abs(e["pysahtyy_Y_cm"] - y_pys2) <= 1.0, f"{e['pysahtyy_Y_cm']:.0f} vs {y_pys2:.0f}")
    tarkista("ennusta: kaarto kohti -X'", e["pysahtyy_X_cm"] < 60.0 - 5.0, f"{e['pysahtyy_X_cm']:.0f}")
    tarkista("ennusta: polku alkaa lahtopaikasta", abs(e["polku"][0][1] - 646.0) < 1e-6 and len(e["polku"]) > 5)


def testaa_yhdista():
    print("kierteen yhdistelma (netti_kierre.yhdista)")
    om1, k = np.radians(80.0), 0.021
    t1, t2, tn, tf = 10.0, 28.0, 8.0, 31.0
    mn = (tn, om1, np.radians(6.0), True)
    mf = (tf, om1 * np.exp(-k * (tf - tn)), np.radians(6.0), True)
    tosi = om1 / (2 * np.pi * k) * (np.exp(-k * (t1 - tn)) - np.exp(-k * (t2 - tn)))
    r = nk.yhdista(mn, mf, t1, t2)
    tarkista("kaksi mittausta: N osuu (+-0,05)", abs(r["N"] - tosi) <= 0.05, f"{r['N']:.2f} vs {tosi:.2f}")
    tarkista("kaksi mittausta: vaimeneminen osuu (+-0,004 / s)", abs(r["k"] - k) <= 0.004, f"{r['k']:.4f}")
    tarkista("kaksi mittausta: epavarmuus pieni", r["N_sigma"] < 0.3, f"{r['N_sigma']:.2f}")
    r1 = nk.yhdista(mn, None, t1, t2)
    tarkista("vain heittopaa: N osuu vakiovaimenemisella", abs(r1["N"] - tosi) <= 0.05 and r1["N_sigma"] > r["N_sigma"], f"{r1['N']:.2f} +- {r1['N_sigma']:.2f}")
    mf_v = (tf, -mf[1], mf[2], True)
    rv = nk.yhdista(mn, mf_v, t1, t2)
    tarkista("etumerkit ristiriidassa: ristiriita, heittopaan etumerkki", rv["ristiriita"] and rv["N"] > 0)
    mf_k = (tf, om1 * 1.6, np.radians(6.0), True)
    rk = nk.yhdista(mn, mf_k, t1, t2)
    tarkista("omega kasvaa: ristiriita", rk["ristiriita"])
    rk2 = nk.yhdista(mn, (tf, om1 * 1.4, np.radians(5.0), True), t1, t2)
    tarkista("omega kasvaa 40 % (pienet mittausvirheet; ennakon leveys ei peita): ristiriita, N lahella heittopaan arviota", rk2["ristiriita"] and abs(rk2["N"] - nk.yhdista(mn, None, t1, t2)["N"]) <= 2.0 * rk2["N_sigma"], f"N {rk2['N']:.2f} +- {rk2['N_sigma']:.2f}")
    tarkista("omega kasvaa 10 % (mittausvirheiden sisalla): ei ristiriitaa", not nk.yhdista(mn, (tf, om1 * 1.1, np.radians(5.0), True), t1, t2)["ristiriita"])
    tarkista("nopea vaimeneminen (k = 0,040, ~1,3 hajontaa): ei ristiriitaa", not nk.yhdista(mn, (tf, om1 * np.exp(-0.040 * (tf - tn)), np.radians(4.0), True), t1, t2)["ristiriita"])
    tarkista("hidas vaimeneminen (k = 0,003): ei ristiriitaa", not nk.yhdista(mn, (tf, om1 * np.exp(-0.003 * (tf - tn)), np.radians(4.0), True), t1, t2)["ristiriita"])


def testaa_pysahtymishetki():
    print("pysahtymishetki ja kohteen vaihtuminen (netti_seuranta)")
    fps = 30.0
    rng = np.random.default_rng(3)
    t = np.arange(0.0, 20.0, 1.0 / 25.0)
    v0, a = 0.8, 0.1
    t_s = v0 / a                                                             # pysahtyy 8 s kohdalla
    y = 640.0 - 100.0 * np.where(t < t_s, v0 * t - 0.5 * a * t * t, 0.5 * v0 * t_s)
    P = np.column_stack([60.0 + rng.normal(0, 2.5, len(t)), y + rng.normal(0, 2.5, len(t))])       # kohina levossakin ~2,5 cm
    F = np.round(t * fps)
    f_pys, X, Y = ns.pysahtymishetki(F, P, fps)
    tarkista("pysahtymishetki +-1,5 s", abs(f_pys / fps - t_s) <= 1.5, f"{f_pys / fps:.1f} s vs {t_s:.1f} s")
    tarkista("lepopaikka +-3 cm", abs(Y - y[-1]) <= 3.0 and abs(X - 60.0) <= 3.0)
    tarkista("ei kiihtymiskohtaa vapaassa liu'ussa", ns.kiihtymiskohta(F, P, fps) is None)
    P2 = P.copy()
    i0 = int(np.searchsorted(t, 6.5))                                        # hidas kivi (~25 cm/s) -> 1 m/s: toinen kohde
    P2[i0:, 0] += 100.0 * (t[i0:] - t[i0])
    ik = ns.kiihtymiskohta(F, P2, fps)
    tarkista("kohteen vaihtuminen havaitaan", ik is not None and abs(t[ik] - 6.5) <= 1.5, "" if ik is None else f"{t[ik]:.1f} s")
    # pelista poistuminen: kivi liukuu taka-viivan yli (Y' = -183 - 14,55 = -197,6 cm) nopeudella 100 cm/s
    tt = np.arange(0.0, 8.0, 1.0 / 25.0)
    Pu = np.column_stack([np.full(len(tt), 20.0), 300.0 - 100.0 * tt])
    Fu = np.round(tt * fps)
    fu = ns.poistumisruutu(Fu, Pu, fps)
    t_tosi = (300.0 - ns.PELI_TAKA_Y_CM) / 100.0
    tarkista("poistuminen taka-viivan yli (+-0,1 s)", fu is not None and abs(fu / fps - t_tosi) <= 0.1, "" if fu is None else f"{fu / fps:.2f} s vs {t_tosi:.2f} s")
    tarkista("pelissa pysyva kivi: ei poistumista", ns.poistumisruutu(Fu[:80], Pu[:80], fps) is None)
    # T-viivan ylitys: kivi kiihtyy hackista (Y = -290 cm) tasaisesti 1,0 m/s2 (nopeus T-viivalla ~ 2,2 m/s), kohina 2 cm
    rng = np.random.default_rng(5)
    ta = np.arange(0.0, 4.0, 1.0 / 25.0)
    Ya = -290.0 + 0.5 * 100.0 * 1.0 * np.maximum(ta - 0.3, 0.0) ** 2
    Fa = np.round(ta * fps)
    pts = [(f, y + rng.normal(0, 2.0), True) for f, y in zip(Fa, Ya)]
    t_tosi = 0.3 + np.sqrt(2.0 * (ns.KELLO_TVIIVA_Y_CM + 290.0) / 100.0)
    ft = ns.tviiva_ruutu(pts)
    tarkista("T-viivan ylitys (+-0,05 s)", ft is not None and abs(ft / fps - t_tosi) <= 0.05, "" if ft is None else f"{ft / fps:.3f} s vs {t_tosi:.3f} s")
    pts_a = [(f, y + rng.normal(0, 2.0) + (40.0 if i % 9 == 0 else 0.0), False) for i, (f, y) in enumerate(zip(Fa, Ya))]       # vain A-paikkoja, osa karkeita
    fa = ns.tviiva_ruutu(pts_a)
    tarkista("T-viivan ylitys A-paikoista karkeista huolimatta (+-0,1 s)", fa is not None and abs(fa / fps - t_tosi) <= 0.1, "" if fa is None else f"{fa / fps:.3f} s")
    tarkista("T-viivan ylitys: ei ylitysta -> None", ns.tviiva_ruutu([p for p in pts if p[1] < -150.0]) is None and ns.tviiva_ruutu(pts[:3]) is None)
    Ps = np.column_stack([-100.0 + 80.0 * tt, 100.0 - 5.0 * tt])               # sivuviivan yli (|X'| > 252 cm)
    fs = ns.poistumisruutu(Fu, Ps, fps)
    tarkista("poistuminen sivuviivan yli", fs is not None and abs(fs / fps - (ns.PELI_SIVU_X_CM + 100.0) / 80.0) <= 0.2)


class _H:
    """Heitto-korvike kellotestiin. tviiva = ruutu, jolla kiven etureuna osuu heittopaan T-viivaan (oletus: lahtoruutu); None = ei viela selvilla (heitto kesken)."""
    def __init__(self, tunnus, vari, puoli, f_alku, f_pys, tormays=None, tila="pysahtyi", f_ulos=None, tviiva=-1):
        self.tunnus, self.vari, self.puoli, self.f_alku, self.kivi_nro = tunnus, vari, puoli, f_alku, 1
        self._tviiva = f_alku if tviiva == -1 else tviiva
        self.tila, self.lopullinen = "valmis", True
        self.kauko = dict(tila=tila, f_pys=f_pys, tormays=tormays, f_ulos=f_ulos) if f_pys is not None else None
        self.viivat = None
        self.hog1_f = None
        self.hog2_f = None
        self.kierre = None

    def tviiva_f(self):
        return self._tviiva


class _N:
    """Kaukopaan seurannan naytteen korvike: ruutu, B-paikka (kivi() = True)."""
    def __init__(self, f, xy):
        self.f, self._xy = f, xy

    def kivi(self):
        return True

    def paikka(self):
        return self._xy


class _KS:
    """Kaukopaan seurannan korvike (Analyysi.kaukoseurannat): heitto h ja sen naytteet."""
    def __init__(self, h, naytteet):
        self.h = h
        self.heitto = type("KH", (), {})()
        self.heitto.naytteet = naytteet


def testaa_kello():
    print("joukkueiden kello (Analyysi.kello / kello_nyt)")
    fps = 30.0

    class An(ns.Analyysi):
        def __init__(self):                                                  # ei kameroita eika prosesseja
            self.fps = fps
            self.heitot = []
    an = An()
    # paa 1 (vasen): K1 lahtee 10 s, pysahtyy 30 s; P1 lahtee 55 s (miettimisaika 25 s), pysahtyy 75 s (tormays 72 s -> 76 s); K2 lahtee 80 s (4 s); paa 2 (oikea): P1 lahtee 200 s
    an.heitot = [_H(1, "keltainen", "vasen", 10 * fps, 30 * fps), _H(2, "punainen", "vasen", 55 * fps, 75 * fps, tormays=dict(f=72 * fps)),
                 _H(3, "keltainen", "vasen", 80 * fps, 100 * fps), _H(4, "punainen", "oikea", 200 * fps, 215 * fps)]
    k = an.kello()
    tarkista("paan ensimmaisella heitolla ei aikaa", k[1]["aika_s"] is None)
    tarkista("miettimisaika 25 s", abs(k[2]["aika_s"] - 25.0) < 1e-6, str(k[2]["aika_s"]))
    tarkista("tormays: pysahtyminen viimeistaan 4 s tormayksen jalkeen -> 80 - 76 = 4 s", abs(k[3]["aika_s"] - 4.0) < 1e-6, str(k[3]["aika_s"]))
    tarkista("uusi paa: ei aikaa (kello ei kay paiden valilla)", k[4]["aika_s"] is None and k[4]["paa"] == 2)
    tarkista("joukkueen aika yhteensa", abs(k[3]["joukkue_yht_s"] - 4.0) < 1e-6 and abs(k[4]["joukkue_yht_s"] - 25.0) < 1e-6, f"{k[3]['joukkue_yht_s']} / {k[4]['joukkue_yht_s']}")
    n = an.kello_nyt(60 * fps)                                               # P1 on lahtenyt, kello pysahtynyt (P1 liikkuu)
    tarkista("kello_nyt: heitto kaynnissa -> kello seis", n["kay"] is None and abs(n["yht"]["punainen"] - 25.0) < 1e-6)
    n = an.kello_nyt(78 * fps)                                               # P1:n pysahtyminen 76 s, K2 ei ole lahtenyt: keltaisen kello kay 2 s
    tarkista("kello_nyt: edellinen pysahtynyt -> vastavari kay", n["kay"] == "keltainen" and abs(n["juokseva_s"] - 2.0) < 1e-6, str(n["juokseva_s"]))
    n = an.kello_nyt(150 * fps)                                              # K2 pysahtyi 100 s, paa ei viela loppunut (3 heittoa): punainen kay
    tarkista("kello_nyt: kuluva aika kasvaa", n["kay"] == "punainen" and abs(n["juokseva_s"] - 50.0) < 1e-6)
    n = an.kello_nyt(210 * fps)
    tarkista("kello_nyt: uuden paan heitto kaynnissa: kello seis", n["kay"] is None)
    an.heitot = [_H(1, "keltainen", "vasen", 10 * fps, 40 * fps, f_ulos=30 * fps), _H(2, "punainen", "vasen", 60 * fps, 80 * fps)]
    k = an.kello()
    tarkista("pelista poistunut kivi: kello kay poistumishetkesta (60 - 30 = 30 s, ei 60 - 40 = 20 s)", abs(k[2]["aika_s"] - 30.0) < 1e-6, str(k[2]["aika_s"]))
    # kello pysahtyy vasta T-viivalla: K1 pysahtyy 30 s; P1 lahtee hackista 55 s, kivi osuu T-viivaan 58 s -> miettimisaika 28 s (ei 25 s)
    an.heitot = [_H(1, "keltainen", "vasen", 10 * fps, 30 * fps, tviiva=13 * fps), _H(2, "punainen", "vasen", 55 * fps, 75 * fps, tviiva=58 * fps),
                 _H(3, "keltainen", "vasen", 80 * fps, 100 * fps, tviiva=None)]
    an.heitot[2].tila, an.heitot[2].lopullinen = "liikkuu", False
    k = an.kello()
    tarkista("kello pysahtyy T-viivalla: 58 - 30 = 28 s", abs(k[2]["aika_s"] - 28.0) < 1e-6 and abs(k[2]["f_stop"] / fps - 58.0) < 1e-6, str(k[2]["aika_s"]))
    tarkista("kesken oleva heitto (T-viivaa ei vielae ylitetty): ei aikaa viela", k[3]["aika_s"] is None and k[3]["f_stop"] is None)
    n = an.kello_nyt(56.5 * fps)                                             # P1 on lahtenyt hackista (55 s), mutta ei ole viela T-viivalla: kello kay viela
    tarkista("kello_nyt: kivi lahtenyt hackista, T-viivalle matkaa: kello kay viela", n["kay"] == "punainen" and abs(n["juokseva_s"] - 26.5) < 1e-6, str(n["juokseva_s"]))
    n = an.kello_nyt(58.5 * fps)
    tarkista("kello_nyt: T-viiva ylitetty: kello seis, heiton aika 28 s", n["kay"] is None and abs(n["viim_aika_s"] - 28.0) < 1e-6 and abs(n["yht"]["punainen"] - 28.0) < 1e-6, str(n["viim_aika_s"]))
    an.heitot[1].tila = an.heitot[1].tila
    h_ei = _H(9, "punainen", "oikea", 300 * fps, 330 * fps, tviiva=None)
    an.heitot = [h_ei]
    an.heitot[0].tila, an.heitot[0].lopullinen = "valmis", True
    tarkista("T-viivaa ei mitattu, heitto valmis: lahto + oletusviive", abs(an.kello_ruutu(h_ei) / fps - (300 + ns.KELLO_TVIIVA_VIIVE_S)) < 1e-6)
    # mallipohjainen pysahtymisarvio (ei kaukopaan seurannan tulosta): P1 hog-viivan nopeus 1,9 m/s, ylittaa lahipaan hog-viivan 20 s
    p1 = _H(1, "punainen", "vasen", 10 * fps, None, tviiva=13 * fps)
    p1.viivat, p1.hog1_f = dict(nopeus_hog1=1.9, nopeus_jalkeen=2.0, irr_Y=300.0, vapaa_hidastuvuus=0.065, vapaa_matka_cm=300.0), 20 * fps
    k2 = _H(2, "keltainen", "vasen", 80 * fps, 100 * fps, tviiva=83 * fps)
    an.heitot = [p1, k2]
    v1, a1 = 1.9, ns.HIDASTUVUUS_TYYPILLINEN
    v2 = (v1 ** 2 - 2 * a1 * (ns.ETAISYYS_CM - 2 * ns.HOG_KESKI_CM) / 100.0) ** 0.5
    f_malli = 20 * fps + fps * (v1 - v2) / a1 + fps * min(v2 / ns.KAUKO_PYS_HIDASTUVUUS, ns.KELLO_ARVIO_MAX_S)
    tarkista("malli: pysahtymisruutu hog-nopeudesta (loppulaskenta, ei viim_f)", abs(an.pysahtyi_f(p1) - f_malli) < 1e-6, f"{an.pysahtyi_f(p1) / fps:.1f} s vs {f_malli / fps:.1f} s")
    tarkista("malli: miettimisaika = T-viiva - mallin pysahtyminen", abs(an.kello()[2]["aika_s"] - (83 * fps - f_malli) / fps) < 1e-6)
    an.viim_f = f_malli + 4.0 * fps
    tarkista("malli: alle 5 s mallin pysahtymisesta -> kello ei kay viela", an.pysahtyi_f(p1) is None and an.kello_nyt(f_malli + 1)["kay"] is None)
    an.viim_f = f_malli + 5.5 * fps
    tarkista("malli: yli 5 s mallin pysahtymisesta eika kivea nahty -> pysahtynyt mallin hetkella", abs(an.pysahtyi_f(p1) - f_malli) < 1e-6)
    n = an.kello_nyt(f_malli + 2 * fps)
    tarkista("kello_nyt: kello lahtee kayntiin mallin pysahtymishetkesta", n["kay"] == "keltainen" and abs(n["juokseva_s"] - 2.0) < 1e-6, str(n["juokseva_s"]))
    an.kaukoseurannat = [_KS(p1, [_N(an.viim_f - 40 + 4 * i, (20.0, 300.0 - 6.0 * i)) for i in range(10)])]            # kivi liikkuu (6 cm / 4 ruutua), viimeinen B-paikka alle 2 s sitten
    tarkista("malli: kivi nahty liikkeessa -> ei oleteta pysahtyneeksi", an.pysahtyi_f(p1) is None)
    an.kaukoseurannat = [_KS(p1, [_N(an.viim_f - 40 + 4 * i, (20.0 + (i % 2), 300.0 + (i % 3))) for i in range(10)])]    # kivi levossa (kohina)
    tarkista("malli: kivi levossa (B-kohina) -> pysahtynyt mallin hetkella", abs(an.pysahtyi_f(p1) - f_malli) < 1e-6)
    an.kaukoseurannat = []
    an.viim_f = 120 * fps
    p1.hog2_f = 40 * fps                                                                                                  # mitattu kaukopaan hog-ylitys antaa pysahtymisen ankkurin
    tarkista("malli: mitattu hog-ylitys kayttoon", abs(an.pysahtyi_f(p1) - (40 * fps + fps * min(v2 / ns.KAUKO_PYS_HIDASTUVUUS, ns.KELLO_ARVIO_MAX_S))) < 1e-6)
    p1.hog2_f = None
    p1.viivat["nopeus_hog1"] = 0.9                                                                                        # hidas kivi ei yllä kaukopaan hog-viivalle: pysahtyy matkalla (v / a)
    tarkista("malli: kivi ei yllä kaukopaan hog-viivalle -> pysahtyy v / a hog-ylityksesta", abs(an.pysahtyi_f(p1) - (20 * fps + fps * 0.9 / a1)) < 1e-6)
    p1.viivat = None
    tarkista("malli: ei viivaestimaattia -> heiton lahto + KELLO_ALKU_PYS_S", abs(an.pysahtyi_f(p1) - (p1.f_alku + fps * ns.KELLO_ALKU_PYS_S)) < 1e-6)
    an.viim_f = p1.f_alku + fps * (ns.KELLO_ALKU_PYS_S + 2.0)
    tarkista("malli: varaarvio ei ala ennen odotusaikaa", an.pysahtyi_f(p1) is None)
    an.viim_f = 120 * fps
    del an.viim_f


def _synteettinen_heitto(y_irr_alku, v_irr, rng, push=True):
    """Syntettinen heittopaan rata (25 r/s): tyonto levosta hackista, liuku (hidastuvuus 0,11 m/s2), myohainen tyonto (kasi kiihdyttaa kiven nopeuteen v_irr 0,5 s:ssa) ja vapaa lento (0,065 m/s2).
    X(Y): suora (suunta -3,4 astetta) + pieni kaarto. Palauttaa (t, X, Y, tosi irroituksen Y, tosi nopeus irroituksessa m/s)."""
    dt = 1.0 / 25.0
    t, y, v, ph = [0.0], [-330.0], [0.0], "tyonto"
    t_vaihe = 0.0
    while y[-1] < 750.0:
        tt, vv = t[-1], v[-1]
        if ph == "tyonto":
            a = 2.2
            if vv >= 1.75:
                ph = "liuku"
        elif ph == "liuku":
            a = -0.11
            if y[-1] >= y_irr_alku:
                ph, t_vaihe = ("tyonto2" if push else "vapaa"), tt
                v_a = vv
        if ph == "tyonto2":
            a = (v_irr - v_a) / 0.5
            if tt - t_vaihe >= 0.5:
                ph = "vapaa"
                y_irr, v_irr_tosi = y[-1], vv
        if ph == "vapaa":
            a = -0.065
        vv = max(vv + a * dt, 0.0)
        v.append(vv)
        y.append(y[-1] + 100.0 * 0.5 * (v[-1] + v[-2]) * dt)
        t.append(tt + dt)
    if not push:
        y_irr, v_irr_tosi = y_irr_alku, float(np.interp(y_irr_alku, y, v))
    t, y = np.array(t), np.array(y)
    X = -60.0 + np.tan(np.radians(-3.4)) * (y + 330.0) + 0.5 * 1e-5 * y ** 2
    return t, X + rng.normal(0, 0.12, len(t)), y + rng.normal(0, 0.5, len(t)), y_irr, v_irr_tosi


def testaa_irroitus():
    print("irroituksen tunnistus (liuku_irroitus.analysoi; vapaan lennon hanta vs kinkkihaku)")
    rng = np.random.default_rng(11)
    asetus = dict(ns.ESTIMAATTI)
    kk = asetus.pop("kohinakatto")
    t, X, Y, y_tosi, v_tosi = _synteettinen_heitto(440.0, 2.0, rng, push=True)
    r_h = li.analysoi(dict(t=t, X=X, Y=Y), kohinakatto=kk, menetelma="hanta", **asetus)
    r_k = li.analysoi(dict(t=t, X=X, Y=Y), kohinakatto=kk, menetelma="kinkki", esitieto=False, **{a: b for a, b in asetus.items() if a not in ("Y_r_max", "esitieto")})
    ti_h, ti_k = r_h["tila"], r_k["tila"]
    print(f"      (myohainen tyonto: tosi irroitus Y = {y_tosi:.0f} cm, nopeus {v_tosi:.2f} m/s; hanta: {ti_h['Y_cm']:.0f} cm, {ti_h['nopeus_m_s']:.2f} m/s, hid. {-ti_h['hidastuvuus_m_s2']:+.3f}; kinkki (ei esitietoa): {ti_k['Y_cm']:.0f} cm, {ti_k['nopeus_m_s']:.2f} m/s, hid. {-ti_k['hidastuvuus_m_s2']:+.3f})")
    tarkista("myohainen tyonto: irroitus tunnistetaan tyonnon loppuun (+-80 cm)", abs(ti_h["Y_cm"] - y_tosi) <= 80.0, f"{ti_h['Y_cm']:.0f} vs {y_tosi:.0f}")
    tarkista("myohainen tyonto: nopeus irroituksen jalkeen oikein (+-0,07 m/s)", abs(ti_h["nopeus_m_s"] - v_tosi) <= 0.07, f"{ti_h['nopeus_m_s']:.2f} vs {v_tosi:.2f}")
    tarkista("myohainen tyonto: vapaan lennon hidastuvuus ei ole kiihtyvyytta", -ti_h["hidastuvuus_m_s2"] >= 0.0, f"{-ti_h['hidastuvuus_m_s2']:+.3f}")
    tarkista("(vertailu) kinkkihaku ilman esitietoa tunnistaa irroituksen liian aikaisin ja aliarvioi nopeuden", ti_k["Y_cm"] < y_tosi - 80.0 and ti_k["nopeus_m_s"] < v_tosi - 0.07, f"{ti_k['Y_cm']:.0f} cm, {ti_k['nopeus_m_s']:.2f} m/s")
    t, X, Y, y_tosi, v_tosi = _synteettinen_heitto(430.0, 2.0, rng, push=False)               # sileä: irroitus ei nay radassa; nopeus silti oikein
    r = li.analysoi(dict(t=t, X=X, Y=Y), kohinakatto=kk, menetelma="hanta", **asetus)
    tarkista("sileä rata (ei tyontoa irroituksen alla): nopeus oikein (+-0,07 m/s)", abs(r["tila"]["nopeus_m_s"] - float(np.interp(r["tila"]["Y_cm"], Y, np.gradient(Y, t) / 100.0))) <= 0.07, f"{r['tila']['nopeus_m_s']:.2f}")
    tarkista("sileä rata: suunta oikein (+-0,4 astetta, tosi -3,4 + kaarto)", abs(r["tila"]["suunta_deg"] - (-3.4 + np.degrees(1e-5 * r["tila"]["Y_cm"]))) <= 0.4, f"{r['tila']['suunta_deg']:+.2f}")


def main():
    for t in (testaa_vakiot, testaa_yhdista, testaa_pysahtymishetki, testaa_kello, testaa_irroitus):
        t()
    print()
    if VIRHEITA:
        print(f"{len(VIRHEITA)} testia epaonnistui: {', '.join(VIRHEITA)}")
        return 1
    print("kaikki testit OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
