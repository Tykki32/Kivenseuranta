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
import lepo_haku as lh  # noqa: E402
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
    # harva rata levossa: hidas vaihe ja lepo naytteistetaan harvemmin (B ei tule joka ruudulle), joten 45 viimeista paikkaa kattaisi liikkeen: lepopaikka ajan mukaan
    Ff = np.arange(0.0, 100.0)                                              # liike: tihea (joka ruutu), 330 -> 160 cm
    yf = 160.0 + 170.0 * (1.0 - Ff / 99.0) ** 2
    Fc = np.arange(100.0, 160.0, 5.0)                                       # hiipiminen: joka 5. ruutu (2 s), 160 -> 100 cm
    yc = 160.0 - 60.0 * (Fc - 100.0) / 55.0
    Fl = np.arange(160.0, 280.0, 6.0)                                       # lepo: joka 6. ruutu (4 s)
    Fh = np.concatenate([Ff, Fc, Fl])
    yh = np.concatenate([yf, np.minimum(yc, 160.0), np.full(len(Fl), 100.0)])
    yh = np.maximum(yh, 100.0)
    Ph = np.column_stack([60.0 + rng.normal(0, 1.0, len(Fh)), yh + rng.normal(0, 1.0, len(Fh))])
    f_h, Xh, Yh = ns.pysahtymishetki(Fh, Ph, fps)
    tarkista("harva lepo: lepopaikka +-3 cm (45 paikan ikkuna olisi sisaltanyt liikkeen)", abs(Yh - 100.0) <= 3.0 and abs(Xh - 60.0) <= 3.0, f"Y {Yh:.1f} X {Xh:.1f}")
    tarkista("harva lepo: pysahtymishetki lepoon saapumisen (5 s) ja sen +3 s valilla", 5.0 <= f_h / fps <= 8.0, f"{f_h / fps:.1f} s")
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


def _heitto_tyonnolla(y_tyonto, kesto_s, a_tyonto, rng, ohjaus=0.0, kohouma=0.0, v_liuku=1.8, y_loppu=750.0):
    """Synteettinen heittopaan rata (25 r/s) tunnetulla irroituksella: tyonto levosta, tasainen liuku (nopeus v_liuku, hidastuvuus 0,10 m/s2), loppuun tyonto (kiihtyvyys a_tyonto m/s2, kesto kesto_s,
    alkaa kun Y = y_tyonto; a_tyonto = 0: ei tyontoa) ja irroituksen jalkeen vapaa lento (0,07 m/s2). Sivusuunta: suora (-3,4 astetta) + kasi ohjaa irroitukseen asti (kaarevuus `ohjaus` 1/cm),
    vapaassa lennossa kaarevuus 0,5e-5 1/cm. kohouma: hog-viivan harha kuten B-sovituksessa (netti_seuranta.HOG_HARHA_CM kerrottuna taman luvulla; Y:ssa nakyva). Palauttaa (t, X, Y, tosi irroituksen Y; None, jos kasi
    ei ole irronnut radan loppuun mennessa)."""
    dt = 1.0 / 25.0
    t, y, v, ph, t_ph = [0.0], [-330.0], [0.0], "tyonto0", 0.0
    y_irr = None
    while y[-1] < y_loppu:
        tt, vv = t[-1], v[-1]
        if ph == "tyonto0":
            a = 2.2
            if vv >= v_liuku:
                ph = "liuku"
        elif ph == "liuku":
            a = -0.10
            if y[-1] >= y_tyonto:
                ph, t_ph = ("tyonto", tt) if a_tyonto != 0.0 else ("vapaa", tt)
                if ph == "vapaa":
                    y_irr = y[-1]
        if ph == "tyonto":
            a = a_tyonto
            if tt - t_ph >= kesto_s:
                ph, y_irr = "vapaa", y[-1]
        if ph == "vapaa":
            a = -0.07
        vv = max(vv + a * dt, 0.0)
        v.append(vv)
        y.append(y[-1] + 100.0 * 0.5 * (v[-1] + v[-2]) * dt)
        t.append(tt + dt)
    t, y = np.array(t), np.array(y)
    d = y + 330.0
    c = np.where(y < (np.inf if y_irr is None else y_irr), ohjaus, 0.5e-5)                      # kaarevuus (1/cm); suunta jatkuva irroituksessa
    th = np.tan(np.radians(-3.4)) + np.concatenate([[0.0], np.cumsum(0.5 * (c[1:] + c[:-1]) * np.diff(d))])
    X = -60.0 + np.concatenate([[0.0], np.cumsum(0.5 * (th[1:] + th[:-1]) * np.diff(d))])
    Yh = y + (kohouma * np.interp(y, *np.array(ns.HOG_HARHA_CM, float).T, left=0.0, right=0.0) if kohouma else 0.0)
    return t, X + rng.normal(0, 0.12, len(t)), Yh + rng.normal(0, 0.5, len(t)), y_irr


def testaa_irroitus_muutos():
    print("irroituksen tunnistus: viimeinen merkitseva muutos (liuku_irroitus.irroitus_muutos)")
    rng = np.random.default_rng(5)
    for nimi, y_tyonto, kesto, a, ohjaus, raja in (("tyonnon loppu (+0,4 m/s2, 0,7 s)", 380.0, 0.7, 0.4, 0.0, 70.0), ("myohainen voimakas tyonto (+0,6 m/s2, 0,5 s)", 540.0, 0.5, 0.6, 0.0, 70.0),
                                                   ("loiva pitka tyonto (+0,25 m/s2, 1,2 s)", 330.0, 1.2, 0.25, 0.0, 90.0), ("sivusuunnan ohjaus loppuu (ei tyontoa)", 400.0, 0.0, 0.0, 6e-5, 70.0)):
        t, X, Y, y_tosi = _heitto_tyonnolla(y_tyonto, kesto, a, rng, ohjaus)
        if a == 0.0 and ohjaus != 0.0:
            y_tosi = y_tyonto                                                                    # ohjaus kestaa tyonnon alun paikkaan asti (a = 0: vapaa lento alkaa heti)
        ir = li.irroitus_muutos(t, X, Y, Y_r_max=ns.ESTIMAATTI["Y_r_max"], min_jalkeen=70.0)
        tarkista(f"{nimi}: irroitus oikeassa kohdassa (+-{raja:.0f} cm) ja merkitseva", ir is not None and abs(ir["Y_r"] - y_tosi) <= raja and ir["merkitseva"], f"{ir and round(ir['Y_r'])} vs {y_tosi:.0f}")
    t, X, Y, y_tosi = _heitto_tyonnolla(380.0, 0.0, 0.0, rng)                                    # ei tyontoa eika ohjausta: rata sileä, irroitus ei nay
    ir = li.irroitus_muutos(t, X, Y, Y_r_max=ns.ESTIMAATTI["Y_r_max"], min_jalkeen=70.0)
    tarkista("sileä rata: ei merkitsevaa irroitusta", ir is None or not ir["merkitseva"], f"{ir and (round(ir['Y_r']), round(ir['voimakkuus'], 2))}")
    t, X, Y, y_tosi = _heitto_tyonnolla(380.0, 0.7, 0.4, rng)
    ir = li.irroitus_muutos(t, X, Y, Y_r_max=ns.ESTIMAATTI["Y_r_max"], min_jalkeen=70.0)
    mp = ir["muutos_profiili"]
    tarkista("voimakkuusprofiili (kuvaa varten): skaaloittain, huippu irroituksen kohdalla", len(mp["S"]) == len(li.MUUTOS_SKAALAT_CM) and len(mp["S"][0]) == len(mp["Y"]) and abs(mp["Y"][int(np.argmax(np.max(mp["S"], axis=0)))] - ir["Y_r"]) <= 150.0,
             f"{len(mp['S'])} skaalaa x {len(mp['Y'])} kohtaa")
    # hog-viivan harha: B-sovituksen +1 cm kohouma (Y ~ 640) korjataan ennen tunnistusta: irroitus oikein, ja kohouman jalkeen nopeus ei heilahda
    t, X, Y, y_tosi = _heitto_tyonnolla(380.0, 0.7, 0.4, rng, kohouma=1.0)
    Yk = ns.korjaa_hog_harha(Y)
    ir_kork = li.irroitus_muutos(t, X, Yk, Y_r_max=ns.ESTIMAATTI["Y_r_max"], min_jalkeen=70.0)
    tarkista("hog-viivan harha korjattuna: irroitus oikein (+-70 cm)", ir_kork is not None and abs(ir_kork["Y_r"] - y_tosi) <= 70.0, f"{ir_kork and round(ir_kork['Y_r'])} vs {y_tosi:.0f}")
    def hog_poikkeama(y):                                                                      # nopeus hog-viivalla (suora mittaus) miinus irroituksen tilan ennuste samalle hetkelle
        ti = li.analysoi(dict(t=t, X=X, Y=y), kohinakatto=(0.8, 0.15), **{k: v for k, v in ns.ESTIMAATTI.items() if k != "kohinakatto"})["tila"]
        return ns.hog_ikkuna(t, X, y)["v_m_s"] - (ti["nopeus_m_s"] + ti["hidastuvuus_m_s2"] * (float(np.interp(646.0, y, t)) - ti["t_r"]))
    p_kork = hog_poikkeama(Yk)
    tarkista("hog-viivan harha korjattuna: hog-viivan nopeus sopii irroituksen jalkeiseen vapaaseen lentoon (< 0,02 m/s)", abs(p_kork) < 0.02, f"{p_kork:+.3f}")
    # osittainen rata: tyonto jatkuu viela radan lopussa -> irroitus ei ole viela tapahtunut (ehdokasalueen ylareuna)
    t, X, Y, y_tosi = _heitto_tyonnolla(380.0, 3.0, 0.45, rng)
    m = Y < 560.0
    ir = li.irroitus_muutos(t[m], X[m], Y[m], Y_r_max=ns.ESTIMAATTI["Y_r_max"], min_jalkeen=70.0)
    tarkista("tyonto jatkuu radan lopussa: irroitus = ehdokasalueen ylareuna, ei merkitseva", ir is not None and ir["tyonto_jatkuu"] and not ir["merkitseva"] and ir["Y_r"] >= Y[m].max() - 80.0, f"{ir and round(ir['Y_r'])} (rata {Y[m].max():.0f})")
    tarkista("harhakorjaus: nolla taulukon ulkopuolella, kohouma enintaan 1,5 cm", float(ns.korjaa_hog_harha(np.array([300.0, 450.0, 750.0])).tolist() == [300.0, 450.0, 750.0]) and 0.5 <= float(np.max(np.array([590.0, 640.0, 700.0]) - ns.korjaa_hog_harha(np.array([590.0, 640.0, 700.0])))) <= 1.5)


def testaa_vakaa():
    print("estimaatin 'vakaa'-lippu (Analyysi._aseta_viivat): rata kattaa koko irroituksen hakualueen")
    import types
    kattava_y = ns.ESTIMAATTI["Y_r_max"] + ns.ESTIMAATTI["min_jalkeen"]

    def estimaatit(y_max, irr=(300.0, 305.0, 310.0), merkitseva=True, lopullinen=False):
        h = types.SimpleNamespace(viivat_hist=[], viivat_loki=[])
        out = []
        for y in irr:
            r = dict(irr_Y=y, merkitseva=merkitseva, Y_max=y_max, vapaa_matka_cm=y_max - y, irr_virhe_deg=0.1, irr_suunta_deg=1.0, irr_X=0.0, liuku_a=0.0, liuku_b=0.0)
            ns.Analyysi._aseta_viivat(h, r, 100, lopullinen)
            out.append(r["vakaa"])
        return out
    tarkista("lyhyt rata (ei kata hakualuetta): ei vakaa, vaikka estimaatit pysyvat paikallaan ja vapaata lentoa on > 250 cm", estimaatit(kattava_y - 20.0) == [False, False, False], str(estimaatit(kattava_y - 20.0)))
    tarkista("kattava rata: vakaa, kun kaksi peräkkäistä estimaattia eroaa < 30 cm", estimaatit(kattava_y + 20.0) == [False, True, True], str(estimaatit(kattava_y + 20.0)))
    tarkista("kattava rata, irroitus hyppaa > 30 cm: ei vakaa", estimaatit(kattava_y + 20.0, irr=(300.0, 400.0)) == [False, False], str(estimaatit(kattava_y + 20.0, irr=(300.0, 400.0))))
    tarkista("ei merkitseva: ei vakaa (kattavallakaan radalla)", estimaatit(kattava_y + 20.0, merkitseva=False) == [False, False, False])
    tarkista("lopullinen estimaatti: vakaa, kun merkitseva (rata keskeytyi ennen hakualueen kattamista)", estimaatit(kattava_y - 60.0, lopullinen=True) == [True, True, True])


def _aikaperusta_rata(R, rng, kesto=5.0, v0=2.2, a=-0.07, aukot=((12, 15), (45, 52))):
    """Synteettinen heittopaan rata (Y cm, ruutunumero 30 r/s -videossa) tunnetulla liikkeella Y(tau) = -330 + 100 (v0 tau + a tau^2 / 2) (tau = todellinen aika, s), kun kuvan lahderuudut tulevat
    R = 25 r/s (kopioruutu joka kuudes: ruudun sisalto = lahderuudun viimeisin) tai R = 30 r/s (kivi liikkuu 30 askelta / s, joka kuudes ruutu vanha). B-aukkoja (kivea ei loydy) kohdissa aukot."""
    f, Y, tau = [], [], []
    for m in range(int(kesto * 30)):
        if m % 6 == 5:                                   # kopioruutu: sisalto = edellinen, tracker ohittaa
            continue
        t_lahde = (m * 25 // 30) / 25.0 if R == 25 else m / 30.0
        f.append(5000 + m); tau.append(t_lahde); Y.append(-330 + 100 * (v0 * t_lahde + 0.5 * a * t_lahde ** 2) + rng.normal(0, 0.4))
    f, Y, tau = np.array(f, float), np.array(Y), np.array(tau)
    keep = Y > 0
    f, Y, tau = f[keep], Y[keep], tau[keep]
    ok = np.ones(len(f), bool)
    for a_, b_ in aukot:
        ok[a_:b_] = False
    return f[ok], Y[ok], tau[ok]


def testaa_aikaperusta():
    print("aikaperusta: lahderuutujen taajuus R radan omasta liikkeesta (25 r/s ja 30 r/s -kadenssi), aukot radassa")
    import aikaperusta as ap
    rng = np.random.default_rng(11)
    for R in (25, 30):
        f, Y, tau = _aikaperusta_rata(R, rng)
        aj = ap.Aikajana(f, Y)
        t = aj.aika(f)
        c = np.polyfit(t - t.mean(), Y, 2)
        v = (c[1] - (-0.07 * 100) * 0) / 100.0                       # nopeus keskella (m/s)
        v_tosi = (2.2 - 0.07 * (t.mean() - t[0] + (tau[0] if False else 0))) if False else None
        c_tosi = np.polyfit(tau - tau.mean(), Y, 2)
        tarkista(f"R = {R}: taajuus tunnistetaan", aj.R == R, f"arvio {aj.R}")
        tarkista(f"R = {R}: nopeus (Y(t):n sovituksesta) +-2 % oikeasta", abs(c[1] / c_tosi[1] - 1.0) < 0.02, f"{c[1] / 100:.3f} vs {c_tosi[1] / 100:.3f} m/s")
        tarkista(f"R = {R}: hidastuvuus +-0,01 m/s2", abs(2 * c[2] - 2 * c_tosi[2]) / 100.0 < 0.01, f"{-2 * c[2] / 100:.3f} vs {-2 * c_tosi[2] / 100:.3f} m/s2")
        d = aj.j - aj.R * aj.f / 30.0
        tarkista(f"R = {R}: lahdeindeksin ajautuma seinakelloon nahden rajoissa (vaihtelu <= 4)", float(np.ptp(d - np.median(d))) <= 4.0, f"{np.ptp(d - np.median(d)):.1f}")


def testaa_kalibroinnin_tarkistus():
    print("tarkista_kalibrointi: kuvan siirtyma kansion moodikuvaan nahden (synteettinen video, tunnetut siirrot)")
    import shutil
    import tempfile
    import cv2
    import tarkista_kalibrointi as tk
    rng = np.random.default_rng(5)
    kuva = cv2.GaussianBlur(rng.normal(128, 60, (1080, 1920)).astype(np.float32), (0, 0), 3.0)         # staattinen tekstuuri: kaikissa ikkunoissa gradienttia
    kuva = np.clip(kuva + 40 * np.sin(np.arange(1920) / 37.0)[None, :], 0, 255).astype(np.uint8)
    tmp = tempfile.mkdtemp()
    try:
        for puoli in ("vasen", "oikea"):
            x, y, w, h = tk.kn.PUOLET[puoli]
            cv2.imwrite(os.path.join(tmp, f"moodikuva_{puoli}.png"), cv2.rotate(cv2.cvtColor(kuva[y:y + h, x:x + w], cv2.COLOR_GRAY2BGR), cv2.ROTATE_90_COUNTERCLOCKWISE))
        for nimi, (sx, sy) in (("sama", (0.0, 0.0)), ("siirretty", (6.0, -4.0))):
            siirretty = cv2.warpAffine(kuva, np.float32([[1, 0, sx], [0, 1, sy]]), (1920, 1080), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            polku = os.path.join(tmp, f"{nimi}.avi")
            kirj = cv2.VideoWriter(polku, cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (1920, 1080))
            for _ in range(30):
                kirj.write(cv2.cvtColor(siirretty, cv2.COLOR_GRAY2BGR))
            kirj.release()
            t = tk.siirtymat(polku, tmp, n=8, alku_s=0.0)
            for puoli in ("vasen", "oikea"):
                sopii, _ = tk.arvioi(t[puoli])
                if nimi == "sama":
                    tarkista(f"{puoli}: sama kuva sopii", sopii is True and abs(t[puoli]["dx"]) < 0.3 and abs(t[puoli]["dy"]) < 0.3, f"dx {t[puoli].get('dx')}")
                else:
                    tarkista(f"{puoli}: siirto (6, -4) px mitataan ja kalibrointi hylataan", sopii is False and abs(t[puoli]["dx"] - sx) < 0.5 and abs(t[puoli]["dy"] - sy) < 0.5, f"dx {t[puoli].get('dx'):.2f} dy {t[puoli].get('dy'):.2f}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _heitto_hog_kohoumalla(rng, kohouma_cm=3.5, tyonto_loppu=None, v0=2.1, ruutuja_pois=0.17):
    """Synteettinen heittopaan rata (t, X, Y): nopeus v0 - 0,065 t (m/s), vapaa lento; hog-viivalla (Y = 635, puolileveys 40 cm) mitattuun Y:hyn lisataan kolmionmuotoinen kohouma kohouma_cm
    (B-sovituksen Y-harha: nopeus nousee ja laskee takaisin, mika nayttaa tyonnon loppuna; paikka palaa ennalleen). tyonto_loppu: kivea tyonnetaan (+0,6 m/s2) tahan Y:hyn asti, ei kohoumaa
    (aito myohainen irroitus: nopeus kasvaa pysyvasti)."""
    dt = 1.0 / 30.0
    t, Y, v = [0.0], [200.0], 100.0 * v0
    while Y[-1] < 740.0:
        y, tt = Y[-1], t[-1]
        v -= 100.0 * 0.065 * dt
        if tyonto_loppu is not None and tyonto_loppu - 100.0 <= y < tyonto_loppu:
            v += 100.0 * 0.6 * dt                                                         # tyonto: nopeus kasvaa pysyvasti
        Y.append(y + v * dt)
        t.append(tt + dt)
    t, Y = np.array(t), np.array(Y)
    X = 0.05 * (Y - 200.0) + rng.normal(0.0, 0.12, len(Y))
    if tyonto_loppu is None:
        Y = Y + kohouma_cm * np.maximum(0.0, 1.0 - np.abs(Y - 635.0) / 40.0)
    keep = rng.random(len(t)) > ruutuja_pois
    return t[keep], X[keep], Y[keep] + rng.normal(0.0, 0.4, int(keep.sum()))


def testaa_hog_alue():
    print("hog-viivan hairioalueen tasoitus (netti_seuranta.tasoita_hog_alue) ja irroitus: nopeuden kohouma hog-viivalla ei ole tyontoa")
    rng = np.random.default_rng(7)
    for kohouma in (2.0, 3.5, 5.0):
        paikalla, ennen = [], []
        for _ in range(6):
            t, X, Y = _heitto_hog_kohoumalla(rng, kohouma)
            Xt, Yt, ok = ns.tasoita_hog_alue(t, X, Y)
            r_ennen = li.analysoi(dict(t=t, X=X, Y=Y), **ns.ESTIMAATTI)
            r = li.analysoi(dict(t=t, X=Xt, Y=Yt), **ns.ESTIMAATTI)
            ennen.append(r_ennen["irroitus"]["Y_max"] if r_ennen.get("irroitus") else None)
            paikalla.append(ok and (r.get("irroitus") is None or not r["irroitus"]["merkitseva"] or r["tila"]["Y_cm"] < ns.HOG_ALUE_CM[0]))
        tarkista(f"kohouma {kohouma} cm: tasoitettu, ja irroitus ei lukitu hog-viivalle (Y < {ns.HOG_ALUE_CM[0]:.0f} cm tai ei merkitseva)", all(paikalla), f"{sum(paikalla)} / 6")
    t, X, Y = _heitto_hog_kohoumalla(rng, 3.5, ruutuja_pois=0.0)
    Xt, Yt, ok = ns.tasoita_hog_alue(t, X, Y)
    a = (Y >= ns.HOG_ALUE_CM[0]) & (Y <= ns.HOG_ALUE_CM[1])
    d = np.concatenate([[np.nan], np.diff(Yt) / np.diff(t) / 100.0])
    tarkista("tasoitettu rata: nopeus alueella tasainen (vaihtelu < 0,05 m/s)", ok and float(np.ptp(d[a][2:-2])) < 0.05, f"{np.ptp(d[a][2:-2]):.3f} m/s")
    tarkista("alueen ulkopuoliset naytteet ennallaan", bool(np.array_equal(Yt[~a], Y[~a])))
    # aito myohainen tyonto hog-viivalle asti: reunat eivat sovi yhteen -> ennallaan, irroitus loytyy viivan lahelta
    ennallaan, lahella = [], []
    for _ in range(6):
        t, X, Y = _heitto_hog_kohoumalla(rng, tyonto_loppu=615.0)
        Xt, Yt, ok = ns.tasoita_hog_alue(t, X, Y)
        r = li.analysoi(dict(t=t, X=Xt, Y=Yt), **ns.ESTIMAATTI)
        ennallaan.append((not ok) and bool(np.array_equal(Yt, Y)))
        lahella.append(r.get("irroitus") is not None and r["tila"]["Y_cm"] >= 570.0)
    tarkista("aito tyonto alueelle asti: rataa ei tasoiteta eika irroitus siirry alueen ylapuolelle (>= 570 cm)", all(ennallaan) and sum(lahella) >= 5, f"ennallaan {sum(ennallaan)} / 6, irroitus >= 570: {sum(lahella)} / 6")
    t, X, Y = _heitto_hog_kohoumalla(rng, 3.5)
    m = Y < 640.0                                                                                 # rata ei yllä alueen yli: ei reunaa jalkeen -> ennallaan
    Xt, Yt, ok = ns.tasoita_hog_alue(t[m], X[m], Y[m])
    tarkista("rata paattyy alueen sisaan (ei jalkeista reunaa): ennallaan", (not ok) and bool(np.array_equal(Yt, Y[m])))


def testaa_harva_rata():
    print("liuku_irroitus.analysoi: harva tai liikkeesta alkava rata ei kaada estimaattia (toisen nettivideon heitoilla LinAlgError / TypeError -> ei estimaattia)")
    rng = np.random.default_rng(11)
    t, X, Y = _heitto_hog_kohoumalla(rng, tyonto_loppu=520.0)                                      # rata alkaa jo nopeasta liikkeesta (Y = 200 cm)
    try:
        r = li.analysoi(dict(t=t, X=X, Y=Y), **ns.ESTIMAATTI)
        tarkista("rata alkaa liikkeesta: estimaatti saadaan", r.get("irroitus") is not None and np.isfinite(r["tila"]["Y_cm"]), f"{r.get('tila') and round(r['tila']['Y_cm'])}")
    except Exception as e:                                                                         # noqa: BLE001
        tarkista("rata alkaa liikkeesta: estimaatti saadaan", False, repr(e))
    for nimi, (a, b) in (("aukko irroituksen kohdalla (Y 470-560)", (470.0, 560.0)), ("aukko heti irroituksen jalkeen (Y 520-600)", (520.0, 600.0))):
        m = ~((Y >= a) & (Y <= b))
        try:
            r = li.analysoi(dict(t=t[m], X=X[m], Y=Y[m]), **ns.ESTIMAATTI)
            ok = r.get("irroitus") is None or all(np.isfinite(r["tila"]["Y_cm"]) for _ in (0,))
            tarkista(f"{nimi}: ei poikkeusta", ok)
        except Exception as e:                                                                     # noqa: BLE001
            tarkista(f"{nimi}: ei poikkeusta", False, repr(e))


def testaa_kalibroinnin_valinta():
    print("kalibrointi_valinta: videoon sopiva kalibrointikansio valitaan automaattisesti (synteettinen video ja kansiot)")
    import shutil
    import tempfile
    import cv2
    import kalibrointi_valinta as kv
    import tarkista_kalibrointi as tk
    rng = np.random.default_rng(9)
    pohja = cv2.GaussianBlur(rng.normal(128, 60, (1080, 1920)).astype(np.float32), (0, 0), 3.0)
    pohja = np.clip(pohja + 40 * np.sin(np.arange(1920) / 37.0)[None, :], 0, 255).astype(np.uint8)

    def kansio(polku, kuva):
        os.makedirs(polku, exist_ok=True)
        for puoli in ("vasen", "oikea"):
            x, y, w, h = tk.kn.PUOLET[puoli]
            cv2.imwrite(os.path.join(polku, f"moodikuva_{puoli}.png"), cv2.rotate(cv2.cvtColor(kuva[y:y + h, x:x + w], cv2.COLOR_GRAY2BGR), cv2.ROTATE_90_COUNTERCLOCKWISE))
            open(os.path.join(polku, f"kalibrointi_{puoli}.json"), "w").write("{}")

    def video(polku, kuva):
        kirj = cv2.VideoWriter(polku, cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (1920, 1080))
        for _ in range(1860):                                                                      # referenssi() ohittaa 30 s alusta ja lopusta
            kirj.write(cv2.cvtColor(kuva, cv2.COLOR_GRAY2BGR))
        kirj.release()

    tmp = tempfile.mkdtemp()
    try:
        siirretty = cv2.warpAffine(pohja, np.float32([[1, 0, 7.0], [0, 1, -5.0]]), (1920, 1080), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        kansio(os.path.join(tmp, "tulokset"), pohja)
        kansio(os.path.join(tmp, "tulokset_toinen"), siirretty)
        video(os.path.join(tmp, "a.avi"), pohja)
        video(os.path.join(tmp, "b.avi"), siirretty)
        k, sel = kv.valitse(os.path.join(tmp, "a.avi"), projekti=tmp, luo=False)
        tarkista("video a: kansio tulokset", os.path.basename(k) == "tulokset", sel[:80])
        k, sel = kv.valitse(os.path.join(tmp, "b.avi"), projekti=tmp, luo=False)
        tarkista("video b (kamera siirtynyt 7, -5 px): kansio tulokset_toinen", os.path.basename(k) == "tulokset_toinen", sel[:80])
        shutil.rmtree(os.path.join(tmp, "tulokset_toinen"))
        k, sel = kv.valitse(os.path.join(tmp, "b.avi"), projekti=tmp, luo=False)
        tarkista("ei sopivaa kansiota (luo = False): tulokset ja selitys", os.path.basename(k) == "tulokset" and "ei sovi" in sel, sel[:80])
        tarkista("osat jakavat automaattisen kansion", kv.automaattinen_nimi("x/Matarmaa_osa_05.mp4") == kv.automaattinen_nimi("Matarmaa_osa_12.mp4") == "tulokset_auto_Matarmaa", kv.automaattinen_nimi("x/Matarmaa_osa_05.mp4"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def testaa_tarkista_kivet():
    print("tarkista_kivet: puuttuva heitto havaitaan (ei totuusaineistoa)")
    import tarkista_kivet as tkv

    def rivit(jono, paa=1, alku=100.0, valit=None):
        out = []
        for i, c in enumerate(jono):
            out.append(dict(paa=str(paa), vari="keltainen" if c == "k" else "punainen", kivi_nro="1", alku_s=str(alku + (valit[i] if valit else 50.0 * i))))
        return out
    hyva = rivit("kp" * 8) + rivit("pk" * 8, paa=2, alku=2000.0)
    ong, _ = tkv.tarkista(hyva)
    tarkista("32 heittoa, vuorottelu kunnossa: ei ongelmia", ong == [], str(ong))
    ong, _ = tkv.tarkista(rivit("kp" * 8) + rivit("pk" * 8, paa=2, alku=2000.0) + rivit("kpkpkpkpkpkpkp", paa=3, alku=4000.0))
    tarkista("viimeinen pää kesken (video loppuu): ei virhe", ong == [], str(ong))
    ong, _ = tkv.tarkista(rivit("kp" * 8) + rivit("pkpkpkpkpkpkpk", paa=2, alku=2000.0) + rivit("kp" * 8, paa=3, alku=4000.0))
    tarkista("keskimmäisen pään viimeinen heitto puuttuu: havaitaan", any("pää 2: 14 heittoa" in o for o in ong), str(ong))
    ong, _ = tkv.tarkista(rivit("kpkpkkpkpkpkpkpk"))
    tarkista("heitto puuttuu keskeltä (kaksi samaa väriä peräkkäin): havaitaan", any("eivät vuorottele" in o for o in ong), str(ong))


def testaa_kahva_ala_mittakaava():
    print("kahvan alaraja hack-alueella on maatason cm2 (px / skaala^2): videon 3 kamerassa (0,60 px/cm) punainen kahva on vain ~46-57 px, kiintea 50 px virittaa kiven vain sattumalta")
    raja = {}
    for kansio in ("tulokset", "tulokset_nettivideo2", "tulokset_nettivideo3"):
        an = ns.Analyysi(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", kansio), fps=30.0, saikeet=0, tarkka=False, synkroninen=True, fps_lahde=None, kierre=False)
        for puoli, hp in an.paat.items():
            raja[(kansio, puoli)] = ns.KAHVA_ALA_CM2[0] * hp.hack_skaala2
    for (kansio, puoli), px in raja.items():
        if kansio == "tulokset_nettivideo3":
            tarkista(f"{kansio} {puoli}: videon 3 punaisen kahvan 46 px kelpaa, 25 px (kasi peittaa) ei", px <= 46.0 and px > 25.0, f"alaraja {px:.1f} px")
        else:
            tarkista(f"{kansio} {puoli}: alaraja ennallaan (~50 px)", abs(px - 50.0) < 1.5, f"alaraja {px:.1f} px")


def testaa_tausta_maskit():
    print("taustan luotettavuus (kivi_tarkka.tausta_maskit) ja graniitin vari (kivi_netti.graniitti_vari): kameran jaan mukaan, jatkuva virhemalli kiintean reunalaajennuksen sijaan")
    import types
    import cv2
    import kivi_netti as kv
    import kivi_tarkka as kt

    def kamera(jaa_bgr, punainen_bgr):
        h, w = 160, 160
        mode = np.zeros((h, w, 3), np.uint8)
        mode[:] = jaa_bgr
        cv2.circle(mode, (80, 80), 40, punainen_bgr, -1)                              # terava, korkeakontrastinen reuna
        cv2.line(mode, (10, 20), (150, 20), tuple(int(c) - 22 for c in jaa_bgr), 1)  # ohut, matalakontrastinen viiva (esim. tekstin pehmea reuna)
        mode = cv2.GaussianBlur(mode, (0, 0), 0.8)
        return types.SimpleNamespace(mode=mode, lane=np.ones((h, w), np.uint8))

    k1 = kamera((236, 229, 227), (117, 93, 180))                                       # video 1: punainen on RGB-etaisyydella 56 graniitista (aiempi raja 60 hylkasi)
    k3 = kamera((213, 209, 207), (81, 81, 222))
    g1, g3 = kv.graniitti_vari(k1), kv.graniitti_vari(k3)
    tarkista("graniitin vari seuraa kameran jaata (video 1 ~ 128, video 3 ~ 115 kanavalla B)", abs(g1[0] - 128.0) < 3.0 and abs(g3[0] - 115.7) < 3.0 and g3[0] < g1[0] - 8.0, f"{g1.round(0)} / {g3.round(0)}")
    for nimi, k in (("video 1", k1), ("video 3", k3)):
        jaa, vari = kt.tausta_maskit(k)
        tarkista(f"{nimi}: punaisen ympyran sisus on luotettavaa varillista taustaa", float(vari[70:90, 70:90].mean()) > 0.95, f"{vari[70:90, 70:90].mean():.2f}")
        tarkista(f"{nimi}: jaa luotettavaa jaata (kaukana reunoista)", float(jaa[100:150, 5:40].mean()) > 0.95, f"{jaa[100:150, 5:40].mean():.2f}")
        tarkista(f"{nimi}: terava punainen reuna on epaluotettava (alfan virhe liian suuri)", float((jaa | vari)[78:83, 39:42].mean()) < 0.5, f"{(jaa | vari)[78:83, 39:42].mean():.2f}")
        tarkista(f"{nimi}: ohut matalakontrastinen viiva ei hylkaa ymparistoaan (aiempi 7 x 7 -laajennus hylkasi)", float(jaa[17:24, 20:140].mean()) > 0.9, f"{jaa[17:24, 20:140].mean():.2f}")


def testaa_vaihto_todennettu():
    print("levossa olevan kiven viritys: tuore todiste ja kiven keskipiste ratkaisevat, mika kahva on kiven kahva (viereinen kenkapohja ei saa jaada virityksen kohteeksi)")
    kansio = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tulokset")
    an = ns.Analyysi(kansio, fps=30.0, saikeet=0, tarkka=False, synkroninen=True, fps_lahde=None, kierre=False)
    hp = an.paat["vasen"]

    def nayte(f, kahva, B=None, Bk=None, poik=None):
        n = ns.Naytteet(f, kahva, 200.0)
        n.lepo = True
        n.B, n.Bk = B, Bk
        if B is not None or Bk is not None:
            n.peitto, n.rms, n.kelv, n.poik = 0.8, 0.15, 0.5, poik
        return n

    def asetelma(kahva, naytteet, ehdokas, validoitu):
        h = ns.Heitto(1, "vasen", "punainen", 0, kahva)
        h.lepo = np.array(kahva, float)
        h.naytteet = naytteet
        h.validoitu = validoitu
        hp.heitto, hp.vari = h, "punainen"
        vaihdot = []
        hp._vakaa_kandidaatti = lambda f, ruutu, pois=None: ("keltainen", [], ehdokas)
        hp._virita_kandidaatista = lambda f, ruutu, vari, ikk, XY: vaihdot.append((vari, XY))
        return h, vaihdot

    # video 1, 3105 s: punainen kenkapohja (0, -343) virittyi, kun kivea kannettiin sen ohi (B <= 12 cm kahvasta); kivi asettui (-13, -325):n tienoille, keltainen kahva (-15, -321)
    vanhat = [nayte(100, (0.0, -343.0), B=(-6.0, -338.0), poik=7.0), nayte(108, (0.0, -343.0), B=(-5.0, -336.0), poik=8.0)]
    uudet = [nayte(200 + 8 * i, (0.0, -340.0), Bk=bk, poik=18.0) for i, bk in enumerate([(-10.0, -327.0), (-12.0, -326.0), (-13.0, -325.0)])]
    h, v = asetelma((0.0, -340.0), vanhat + uudet, (-15.0, -321.0), True)
    tarkista("vanha todiste ei pidata kenkapohjaa todennettuna, kun tuoreet sovitukset osoittavat kiven muualle", not hp._validoitu(h, 222))
    tarkista("kiven keskipiste tuoreista sovituksista", np.allclose(hp._kiven_keskipiste(h, 222), (-12.0, -326.0)))
    tarkista("kenkapohja virittyneena, keltainen kahva kiven keskella: vaihto", hp._vaihda_kandidaatti(222, None) and len(v) == 1)

    uudet = [nayte(200 + 8 * i, (-10.0, -325.0), B=b, poik=3.0) for i, b in enumerate([(-12.0, -327.0), (-12.0, -328.0), (-11.0, -327.0)])]
    h, v = asetelma((-10.0, -325.0), uudet, (-15.0, -321.0), False)
    tarkista("kahvan kohdalla olevat sovitukset todentavat kahvan", hp._validoitu(h, 222))
    tarkista("oikea kahva kiven keskella: ei vaihtoa", not hp._vaihda_kandidaatti(222, None) and not v)
    h, v = asetelma((0.0, -340.0), vanhat + [nayte(200 + 8 * i, (0.0, -340.0), Bk=bk, poik=18.0) for i, bk in enumerate([(-10.0, -327.0), (-12.0, -326.0), (-13.0, -325.0)])], (-50.0, -281.0), True)
    tarkista("toinen kandidaatti kaukana kivesta: ei vaihtoa", not hp._vaihda_kandidaatti(222, None) and not v)

    liikkuva = [nayte(200 + 8 * i, (0.0, -340.0), Bk=bk, poik=18.0) for i, bk in enumerate([(-10.0, -327.0), (-25.0, -344.0), (-4.0, -354.0)])]
    h, v = asetelma((0.0, -340.0), vanhat[:1] + liikkuva, (-15.0, -321.0), True)
    tarkista("kivi ei lepaa (sovitukset hajallaan): ei keskipistetta", hp._kiven_keskipiste(h, 222) is None)
    tarkista("todettu kivi, joka ei nay levossa: viritys pysyy", not hp._vaihda_kandidaatti(222, None) and not v)

    h, v = asetelma((0.0, -340.0), [nayte(200, (0.0, -340.0))], (-15.0, -321.0), False)
    tarkista("vaihtoa ei tehda ennen VAIHTO_S:aa viritysta", not hp._vaihda_kandidaatti(222, None) and not v)
    tarkista("kahvan luona ei kivea eika todisteita: vaihto vakaaseen kandidaattiin", hp._vaihda_kandidaatti(240, None) and len(v) == 1)


def testaa_kandidaatti_kivi():
    print("kahvakandidaatti viriteaan vain, jos sen kohdalla on kivi (harjan paa ei saa estaa viereisen kiven viritysta)")
    kansio = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tulokset_nettivideo3")
    an = ns.Analyysi(kansio, fps=30.0, saikeet=0, tarkka=False, synkroninen=True, fps_lahde=None, kierre=False)
    an.tarkka = True                                                                  # B-sovitus kaytossa (sovitus korvataan alla)
    hp = an.paat["oikea"]

    def blobi(X, Y, ala=70.0):
        u, v = hp.m.maa_oikaistuun(X, Y)
        return (float(u[0]), float(v[0]), ala, (int(u[0]) - 5, int(v[0]) - 5, 10, 11))

    def aja(kahvat, kivet):
        """kahvat: vari -> (X, Y); kivet: kohdat (X, Y), joissa on kivi (sovitus loytaa kiven siita, enintaan 3 cm kahvasta); muualla ei kivea (peitto 0,2)."""
        hp.kandidaatit, hp._kivi_muisti = {}, {}
        hp.m.valmistele = lambda ruutu, X0, Y0: ("roi", 0, 0)
        def sovita(roi, x0, y0, X0, Y0):
            for (X, Y) in kivet:
                if np.hypot(X - X0, Y - Y0) <= 20.0:
                    return dict(X=X + 2.0, Y=Y, rms=0.1, peitto=0.8, kelvolliset=0.5)
            return dict(X=X0 + 20.0, Y=Y0 - 20.0, rms=0.13, peitto=0.2, kelvolliset=0.5)
        hp.m.sovita_leike = sovita
        hp.m.kahva = lambda ruutu, alue, vari, ennuste, min_ala=0, taustaero=False: [blobi(*kahvat[vari])] if vari in kahvat else []
        r = None
        for f in range(0, 60, 3):
            r = hp._vakaa_kandidaatti(f, None) or r
            if r:
                break
        return r

    r = aja({"punainen": (-11.0, -316.0)}, [(-11.0, -316.0)])
    tarkista("kahva kiven paalla: kandidaatti", r is not None and r[0] == "punainen")
    r = aja({"keltainen": (-43.0, -286.0)}, [])
    tarkista("harjan paa (ei kivea): ei kandidaattia", r is None)
    r = aja({"keltainen": (-43.0, -286.0), "punainen": (-11.0, -316.0)}, [(-11.0, -316.0)])
    tarkista("keltainen haamu ja punainen kivi: punainen virittyy (keltainen ei ole ensin vain varin takia)", r is not None and r[0] == "punainen")


def testaa_graniittihaku():
    print("graniittihaku (kivi_haku): kiven runko loytyy ilman kahvan varia; ihmisen vaatteet ja kahvan muovi eivat ole graniittia")
    import kivi_haku as kh
    import kivi_netti as kvn
    import kivi_tarkka as ktt
    kansio = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tulokset_nettivideo3")
    m = ns.Mittari(kansio, "oikea")
    h = kh.GraniittiHaku(m, deci=3)
    d, cam = h.deci, m.cam
    g = kvn.graniitti_vari(cam)

    def kuva(X, Y, kivi=True, vaate=None):
        """Harvennettu ROI: tausta + graniittirunko (nakyva osa) kohdassa (X, Y) + valinnainen vaatekappale (BGR) kiven vierella."""
        roi = h.tausta.astype(np.float32).copy()
        win = (h.x0, h.y0, roi.shape[1] * d, roi.shape[0] * d)
        if kivi:
            c, wt = ktt.malli_kuva(cam, m.runko, win, X, Y)
            a = (c * wt)[::d, ::d][:roi.shape[0], :roi.shape[1]]
            roi = roi * (1 - a[..., None]) + g[None, None, :] * a[..., None]
        if vaate is not None:
            u, v = (np.asarray(q)[0] for q in m.maa_oikaistuun(X + 70.0, Y, 0.0))
            i, j = int((v - h.y0) / d), int((u - h.x0) / d)
            roi[max(i - 8, 0):i + 8, max(j - 6, 0):j + 6] = vaate
        return np.clip(roi, 0, 255).astype(np.uint8)

    X0, Y0 = 20.0, 560.0
    c = [x for x in h.blobit(kuva(X0, Y0)) if kh.ALA_RAJAT[0] <= x["suhde"] <= kh.ALA_RAJAT[1]]
    paras = min(c, key=lambda x: np.hypot(x["X"] - X0, x["Y"] - Y0)) if c else None
    tarkista("kivi loytyy rungosta (keskipiste <= 8 cm akselista)", paras is not None and np.hypot(paras["X"] - X0, paras["Y"] - Y0) <= 8.0)
    tarkista("alan suhde odotettuun jarkeva (0,35-1,9)", paras is not None and kh.ALA_RAJAT[0] <= paras["suhde"] <= kh.ALA_RAJAT[1])
    for nimi, vari in (("sininen takki", (170, 60, 20)), ("musta housu", (15, 15, 15)), ("punainen harja", (40, 40, 200)), ("keltainen harja", (40, 190, 230))):
        c = [x for x in h.blobit(kuva(X0, Y0, kivi=False, vaate=vari)) if kh.ALA_RAJAT[0] <= x["suhde"] <= kh.ALA_RAJAT[1]]
        tarkista(f"{nimi} ei ole graniittia", not c)


def testaa_saapuminen_graniitti():
    print("kaukopaan saapuminen graniittikanavalla: kulkeva, B:lla todettu rata, jonka keskella on kiven varinen kahva, hyvaksytaan; paikallaan oleva kivi, B-tonta ja kahvaton (sweeperin paita) rata eivat")
    kansio = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tulokset_nettivideo3")

    def aja(ehdokkaat, b_ok, kahva=True):
        an = ns.Analyysi(kansio, fps=30.0, saikeet=0, tarkka=False, synkroninen=True, fps_lahde=None, kierre=False)
        an.tarkka = True
        h = ns.Heitto(7, "vasen", "punainen", 0, (0.0, -330.0))
        h.f_loppu, h.ala_med = 0, 30.0
        sa = ns.Saapuminen(an, h)
        sa._blobit = lambda r: []
        sa._kartoita = lambda f, r: None
        sa._kahva_lahella = lambda ruutu, X, Y: kahva
        sa.f_alku = 0
        sa.haku = type("H", (), {"ehdokkaat": lambda self, r: ehdokkaat(self.f)})()
        pyynnot = []

        def b(hp, heitto, naytteet, ruutu, X0, Y0, ennuste=None):
            pyynnot.append(naytteet.f)
            if b_ok:
                naytteet.B = (X0, Y0)
        an.pyyda_b = b
        for f in range(0, 90):
            sa.haku.f = f
            if not sa.kasittele(f, None):
                break
        return h.saapuminen, len(an.kaukoseurannat), pyynnot

    liikkuva = lambda f: [dict(X=40.0, Y=720.0 - 5.0 * f, ala=900, suhde=1.0)]                    # 5 cm / ruutu = 150 cm/s kohti pesaa
    paikallaan = lambda f: [dict(X=110.0, Y=600.0, ala=900, suhde=1.0)]
    tila, n, _ = aja(liikkuva, True)
    tarkista("kulkeva graniittirata B-todistein hyvaksytaan", tila == "loytyi" and n == 1)
    tila, n, _ = aja(liikkuva, False)
    tarkista("kulkeva rata ilman B-todisteita ei kelpaa (ihmisen varuste)", tila != "loytyi" and n == 0)
    tila, n, pyynnot = aja(liikkuva, True, kahva=False)
    tarkista("kulkeva B:lla todettu rata ilman kiven varista kahvaa ei kelpaa (sweeperin kuviollinen paita on graniitin nakoinen)", tila != "loytyi" and n == 0 and pyynnot)
    tila, n, pyynnot = aja(paikallaan, True)
    tarkista("paikallaan oleva kivi ei ole saapuva kivi eika sille pyydeta B:ta", tila != "loytyi" and n == 0 and not pyynnot)


def testaa_viritys_rungosta():
    print("viritys kiven rungosta: vakaa graniittiblobi + B todentaa kiven + kahva kiven keskella maaraa varin; ilman kahvaa ei viritysta")
    kansio = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tulokset")
    an = ns.Analyysi(kansio, fps=30.0, saikeet=0, tarkka=False, synkroninen=True, fps_lahde=None, kierre=False)
    an.tarkka = True
    hp = an.paat["vasen"]
    ala_kahva = 1.2 * ns.KAHVA_ALA_CM2[0] * hp.hack_skaala2

    def blobi(X, Y, ala):
        u, v = hp.m.maa_oikaistuun(X, Y)
        return (float(u[0]), float(v[0]), ala, (int(u[0]) - 5, int(v[0]) - 5, 10, 11))

    def aja(runko, kahvat, b_ok=True, liike=0.0):
        """runko: (X, Y) graniittiblobin keskipiste; kahvat: vari -> [(X, Y, ala)]; liike: blobin siirtyma cm / kutsu."""
        hp.kandidaatit_g, hp._runko_muisti = [], {}
        haku = type("H", (), {"ehdokkaat": lambda self, r: [dict(X=runko[0] + liike * self.n, Y=runko[1], ala=900, suhde=1.0)], "n": 0})()
        an.graniittihaku_hack = lambda puoli: haku
        hp._runko_b = lambda f, ruutu, X, Y: ((X + 1.0, Y - 1.0) if b_ok else None)
        hp.m.kahva = lambda ruutu, alue, vari, ennuste, min_ala=0, taustaero=False: [blobi(*q) for q in kahvat.get(vari, [])]
        r = None
        for i, f in enumerate(range(0, 60, 3)):
            haku.n = i
            r = hp._vakaa_runko(f, None)
            if r:
                break
        return r

    r = aja((-5.0, -325.0), {"keltainen": [(-4.0, -327.0, ala_kahva)]})
    tarkista("vakaa runko + kahva kiven keskella: viritys keltaisena", r is not None and r[0] == "keltainen" and abs(r[2][0] + 4.0) < 1.5)
    r = aja((-5.0, -325.0), {"keltainen": [(-4.0, -327.0, ala_kahva)], "punainen": [(30.0, -330.0, ala_kahva)]})
    tarkista("toinen vari kauempana (kenkapohja 35 cm): vari kiven keskelta", r is not None and r[0] == "keltainen")
    r = aja((-5.0, -325.0), {"keltainen": [(-4.0, -327.0, ala_kahva)], "punainen": [(-2.0, -338.0, ala_kahva)]})
    tarkista("molemmat varit kiven lahella: lahin keskipisteeseen (keltainen 2 cm, punainen 13 cm -> keltainen)", r is not None and r[0] == "keltainen")
    r = aja((-5.0, -325.0), {"punainen": [(40.0, -300.0, ala_kahva)]})
    tarkista("kahva yli 12 cm keskipisteesta (kenka, harjan paa): ei viritysta", r is None)
    r = aja((-5.0, -325.0), {"keltainen": [(-4.0, -327.0, ala_kahva)]}, b_ok=False)
    tarkista("B ei toteaa kivea: ei viritysta", r is None)
    r = aja((-5.0, -325.0), {"keltainen": [(-4.0, -327.0, ala_kahva)]}, liike=3.0)
    tarkista("blobi liikkuu (kantaminen): ei viritysta", r is None)
    r = aja((-5.0, -325.0), {"keltainen": [(-4.0, -327.0, 0.6 * ns.KAHVA_ALA_CM2[0] * hp.hack_skaala2)]})
    tarkista("osittain peitetty kahva (0,6 x alaraja) riittaa varin maaraamiseen", r is not None)
    r = aja((-5.0, -325.0), {"keltainen": [(-4.0, -327.0, 0.3 * ns.KAHVA_ALA_CM2[0] * hp.hack_skaala2)]})
    tarkista("liian pieni kahvan sirpale (0,3 x alaraja): ei viritysta", r is None)


def testaa_ulottuma_vahvistettu():
    print("B hyvaksytaan ulottuman perusteella myos vahvistetulla heitolla (siemen jaa nopean kiven jalkeen), mutta ei pitkan aukon jalkeen eika kaukopaassa")
    kansio = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tulokset")
    an = ns.Analyysi(kansio, fps=30.0, saikeet=0, tarkka=False, synkroninen=True, fps_lahde=None, kierre=False)

    def heitto(luokka=ns.Heitto, tila="vahvistettu", f_b=100):
        h = luokka(1, "vasen", "punainen", 0, (0.0, -330.0)) if luokka is ns.Heitto else luokka(1, "vasen", "punainen")
        h.tila = tila
        n = ns.Naytteet(f_b)
        n.B = (10.0, 100.0)
        h.naytteet.append(n)
        return h

    def tulos(h, f, Y, seed_Y, peitto=0.7):
        n = ns.Naytteet(f)
        h.naytteet.append(n)
        an._liita_b(h, n, dict(X=10.0, Y=Y, rms=0.1, kelvolliset=0.5, peitto=peitto), (10.0, seed_Y))
        return n.B is not None

    tarkista("siemen 24 cm jaljessa, sovitus 9 cm edellisesta B:sta (1 ruutu): hyvaksytaan", tulos(heitto(), 101, 109.0, 85.0))
    tarkista("sama liikkuvalla (ei vahvistetulla) heitolla", tulos(heitto(tila="liikkuu"), 101, 109.0, 85.0))
    tarkista("heikko peitto (0,3): ei ulottumahyvaksyntaa", not tulos(heitto(), 101, 109.0, 85.0, peitto=0.3))
    tarkista("liian kaukana edellisesta B:sta (15 + 8 cm / ruutu: 40 cm 1 ruudussa): ei hyvaksyta", not tulos(heitto(), 101, 140.0, 85.0))
    tarkista("vahvistetulla heitolla yli 12 ruudun aukko: ei ulottumahyvaksyntaa", not tulos(heitto(f_b=80), 101, 109.0, 85.0))
    tarkista("kaukopaan seuranta (KaukoHeitto) ei kayta ulottumaa", not tulos(heitto(ns.KaukoHeitto), 101, 109.0, 85.0))


def testaa_lepo_haku():
    """Lepopaikan palautus (lepo_haku.LepoHaku): paikallaanolo ajan mukaan, vanhat kivet, kivi ei peruuta, valinta, odotus ennen valintaa, tyhjentyneet vanhat kivet."""
    class Kartta:
        def __init__(self, blobit):
            self.blobit = blobit

        def kahvat(self, ruutu, vari, ala_med):
            return list(self.blobit)

    class Haku(lh.LepoHaku):
        def _b(self, ruutu, X, Y):
            return (X, Y)                                            # B-todennus korvattu: kaikki paikallaan pysyvat ovat kivia

    def uusi(blobit, ennen=(), viim=(0, 0.0, 100.0, 0.0, 0.0)):
        return Haku(None, Kartta(blobit), "punainen", None, ennen, viim, 0, 30.0)

    def aja(h, kesto_s, vali=lh.LEPO_VALI, f0=100):
        f = f0
        while f - f0 <= kesto_s * 30.0 and h.kasittele(f, None):
            f += vali
        return f

    h = uusi([(50.0, 60.0, 400)])
    f = aja(h, 12.0)
    tarkista("paikallaan pysyva kahva (7 katselua / 3 s) loytyy odotuksen jalkeen", h.tulos is not None and "X" in h.tulos and abs(h.tulos["X"] - 50.0) < 1e-6, str(h.tulos))
    tarkista("valinta odottaa LEPO_ODOTUS_S ensimmaisen vahvistuksen jalkeen", h.tulos is not None and h.tulos["f_vahv"] - h.f_vahv1 >= lh.LEPO_ODOTUS_S * 30.0, str(h.tulos))
    h = uusi([(50.0, 60.0, 400)])
    aja(h, 1.5)
    tarkista("1,5 s ei riita paikallaanoloksi", h.tulos is None and all(h._paikallaan(c) is None for c in h.ehdokkaat))
    h = uusi([])
    for k in range(14):
        h.kartta.blobit = [(50.0 + 2.0 * k, 60.0, 400)]                          # hiipiva kivi / kuljetettava esine: 2 cm per katselu (liittyy samaan ehdokkaaseen, mutta ei pysy paikallaan)
        h.kasittele(100 + k * lh.LEPO_VALI, None)
    tarkista("hiipiva blobi (2 cm / 0,5 s) ei ole paikallaanoloa", h.tulos is None and len(h.ehdokkaat) == 1 and all(h._paikallaan(c) is None for c in h.ehdokkaat))
    h = uusi([(50.0, 60.0, 400)], ennen=[(51.0, 61.0)])
    aja(h, 12.0)
    tarkista("ennen heittoa paikallaan ollut kivi (kartta) ei ole ehdokas", not h.ehdokkaat)
    h = uusi([(50.0, 160.0, 400)], viim=(0, 0.0, 100.0, 0.0, 0.0))
    aja(h, 12.0)
    tarkista("kivi ei peruuta: ehdokas yli 30 cm viimeisen paikan takana hylataan", not h.ehdokkaat)
    h = uusi([(50.0, 60.0, 400), (-120.0, 20.0, 400)], viim=(0, 40.0, 110.0, 0.0, -0.5))
    aja(h, 14.0)
    tarkista("kaksi ehdokasta, toinen selvasti lahempana kuvitellulle pysahtymispisteelle: valitaan lahempi", h.tulos is not None and "X" in h.tulos and abs(h.tulos["X"] - 50.0) < 1e-6 and h.tulos["ehdokkaita"] == 2, str(h.tulos))
    h = uusi([(40.0, 60.0, 400), (-40.0, 60.0, 400)], viim=(0, 0.0, 110.0, 0.0, -0.5))
    aja(h, 14.0)
    tarkista("kaksi yhta kaukana olevaa ehdokasta: epaselva, ei paikkaa", h.tulos is not None and h.tulos.get("epaselva") and "X" not in h.tulos, str(h.tulos))
    h = uusi([(50.0, 60.0, 400)], ennen=[(-100.0, 0.0), (100.0, 0.0)])
    aja(h, 12.0)
    tarkista("molemmat vanhat kivet nakymattomia 3 katselun ajan: tyhjentyneita 2 tuloksessa", h.tulos is not None and h.tulos["tyhjentyneita"] == 2, str(h.tulos))
    h = uusi([(50.0, 60.0, 400)])
    aja(h, lh.LEPO_AIKARAJA_S + 5.0, f0=100)
    tarkista("aikaraja: tulos on aina (loytyi tai ei_loytynyt)", h.tulos is not None)
    import types
    nk_ = types.SimpleNamespace(karttat=[(0, [(10.0, 10.0), (50.0, 60.0)]), (60, [(10.5, 10.2)]), (120, [(10.2, 9.9), (-80.0, 20.0)]), (400, [(10.0, 10.0), (99.0, 99.0)])])
    kk_ = ns.Saapuminen.kartta_ennen(nk_, 200)
    tarkista("kartta ennen: ennen rataa otettujen karttojen yhdiste (kiven pudottanut kartta ei jata sita pois), duplikaatit 3 cm:n sisalla yhdistetty, rataa myohemmin otettu kartta pois",
             len(kk_) == 3 and any(abs(q[0] - 50.0) < 1e-6 for q in kk_) and any(abs(q[0] + 80.0) < 1e-6 for q in kk_) and not any(abs(q[0] - 99.0) < 1e-6 for q in kk_), str(kk_))
    h = uusi([(50.0, 60.0, 400)])
    aja(h, 4.0)
    h.paata_kesken(10 ** 6)
    tarkista("videon loppu kesken odotuksen: jo vahvistettu ehdokas ratkaistaan", h.tulos is not None and "X" in h.tulos, str(h.tulos))


def main():
    for t in (testaa_vakiot, testaa_yhdista, testaa_pysahtymishetki, testaa_kello, testaa_irroitus, testaa_irroitus_muutos, testaa_vakaa, testaa_aikaperusta, testaa_kalibroinnin_tarkistus, testaa_hog_alue, testaa_harva_rata, testaa_kalibroinnin_valinta, testaa_tarkista_kivet, testaa_kahva_ala_mittakaava, testaa_tausta_maskit, testaa_vaihto_todennettu, testaa_kandidaatti_kivi, testaa_graniittihaku, testaa_saapuminen_graniitti, testaa_viritys_rungosta, testaa_ulottuma_vahvistettu, testaa_lepo_haku):
        t()
    print()
    if VIRHEITA:
        print(f"{len(VIRHEITA)} testia epaonnistui: {', '.join(VIRHEITA)}")
        return 1
    print("kaikki testit OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
