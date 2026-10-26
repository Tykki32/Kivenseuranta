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


def testaa_esilataus_viive():
    """netti_live.Esilataus (pitka puskuri pakattuna) ja Piirtaja: 'Viive:' kellon ylapuolella."""
    import time
    import cv2
    import netti_live as nl
    import netti_piirto as npi
    rng = np.random.default_rng(5)

    def ruudut(n, viive_s=0.0):
        for f in range(n):
            if viive_s:
                time.sleep(viive_s)
            fr = np.full((64, 96, 3), 120, np.uint8)
            fr[10:30, 20:60] = (f % 200, 60, 200 - f % 200)                         # tasainen vari: YUV 4:2:0 -pakkaus ei vaikuta lahes lainkaan
            yield f, fr

    ulos = list(nl.Esilataus(ruudut(20), 4, 2.0, 10.0))
    tarkista("esilataus: jarjestys ja maara sailyvat (pakattu puskuri)", [f for f, _ in ulos] == list(range(20)), str([f for f, _ in ulos]))
    f0, fr0 = ulos[3]
    tarkista("esilataus: pakkauksen kautta kulkenut ruutu on lahes sama (tasainen kuva, virhe <= 2)", fr0.shape == (64, 96, 3) and int(np.abs(fr0[10:30, 20:60].astype(int) - np.array([f0 % 200, 60, 200 - f0 % 200])).max()) <= 2 and int(np.abs(fr0[40:60, 70:90].astype(int) - 120).max()) <= 2)
    ulos = list(nl.Esilataus(ruudut(5), 4, 0.0, 10.0))
    tarkista("esilataus: ilman pitkaa puskuria ruutu on raaka (tarkalleen sama)", np.array_equal(ulos[1][1][40:60, 70:90], np.full((20, 20, 3), 120, np.uint8)))
    ulos = list(nl.Esilataus(ruudut(5), 0, 0.0, 10.0))
    tarkista("esilataus: n = 0 -> ei erillista saiettta, kaikki ruudut", len(ulos) == 5)
    e = nl.Esilataus(ruudut(40), 2, 3.0, 10.0)                                        # kapasiteetti 30 ruutua
    time.sleep(0.5)
    tarkista("esilataus: puskuri tayttyy lukijan edelle (kapasiteetti 3 s = 30 ruutua)", 2.5 <= e.pituus_s <= 3.0 + 1e-9, f"{e.pituus_s:.1f} s")
    f, _ = next(e)
    tarkista("esilataus: ruutu 0 ensin, lukuhetki kirjattu", f == 0 and e.t_luettu <= time.time())
    e.sulje()

    def rikkoutuva():
        yield from ruudut(3)
        raise RuntimeError("verkko katkesi")
    sain = []
    try:
        for f, _ in nl.Esilataus(rikkoutuva(), 2, 1.0, 10.0):
            sain.append(f)
        virhe = False
    except RuntimeError:
        virhe = True
    tarkista("esilataus: luvun poikkeus annetaan kuluttajalle ruutujen jalkeen", sain == [0, 1, 2] and virhe, str(sain))

    class An:
        heitot = []
    for v, vari_pitaisi in ((0.3, "valkoinen"), (2.0, "keltainen"), (5.0, "punainen")):
        p = npi.Piirtaja(None, fps=30.0)
        p.viive_nyt, p.puskuri_s = v, 28.0
        kuva = np.zeros((1080, 1920, 3), np.uint8)
        p.piirra(kuva, 0, An())
        y0 = npi.KELLO_Y - 4 - 33
        alue = kuva[max(0, y0 - 10):npi.KELLO_Y, :]
        tekstia = alue[:, (1920 - 270) // 2: (1920 + 270) // 2].reshape(-1, 3)
        kirkkaat = tekstia[tekstia.max(axis=1) > 150]
        tarkista(f"Piirtaja: 'Viive' piirtyy kellon ylapuolelle ({v:.1f} s: {vari_pitaisi})", len(kirkkaat) > 50, f"{len(kirkkaat)} kirkasta pikselia")
        if len(kirkkaat) > 50:
            kesk = kirkkaat.mean(axis=0)                       # BGR
            ok = {"valkoinen": kesk.min() > 150, "keltainen": kesk[2] > 150 and kesk[0] < 100, "punainen": kesk[2] > 150 and kesk[1] < 120 and kesk[0] < 120}[vari_pitaisi]
            tarkista(f"Piirtaja: 'Viive'-rivin vari ({vari_pitaisi})", bool(ok), f"BGR {kesk.astype(int).tolist()}")
    p = npi.Piirtaja(None, fps=30.0)
    kuva = np.zeros((1080, 1920, 3), np.uint8)
    p.piirra(kuva, 0, An())
    tarkista("Piirtaja: ilman viive_nyt (toistotilat) ei piirreta Viive-riviä", int(kuva.max()) == 0)


def testaa_merkki():
    print("maalimerkki (merkki.py): harjanpaa (oikaistussa kuvassa vaakasuuntainen) loytyy, kiven kahva (pystysuuntainen), puulattia ja tyhja kuva eivat ole maali; harjanpaa + varsi, varjossa oleva harjanpaa, usea harja")
    import types
    import cv2
    import merkki as mk
    kansio = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tulokset_nettivideo3")
    m = ns.Mittari(kansio, "oikea")
    kuvain = m.kuvain
    mitta = types.SimpleNamespace(cam=m.cam, oikaistu_maahan=m.oikaistu_maahan, maa_oikaistuun=m.maa_oikaistuun, skaala=m.skaala,                      # oikaistu kuva suoraan (ei uudelleenkartoitusta lahdekuvasta)
                                  kuvain=types.SimpleNamespace(leike=lambda r, x0, y0, x1, y1: (r[max(y0, 0):y1, max(x0, 0):x1], max(x0, 0), max(y0, 0)), puolisko=kuvain.puolisko, mx=kuvain.mx, my=kuvain.my))

    def bgr(h, s_, v):
        return tuple(int(c) for c in cv2.cvtColor(np.array([[[h, s_, v]]], np.uint8), cv2.COLOR_HSV2BGR)[0, 0])

    def suorakulmio(kuva, X, Y, z, leveys_px, korkeus_px, vari, siirto_u=0.0):
        """Oikaistun kuvan suorakulmio maatason pisteen (X, Y, z) kohdalla (kuva on oikaistussa avaruudessa): leveys vaakasuunnassa (kentan suunta, jyrkasti lyhentynyt), korkeus pystysuunnassa."""
        u, v = (float(q[0]) for q in m.maa_oikaistuun([X], [Y], z))
        u += siirto_u
        cv2.rectangle(kuva, (int(round(u - leveys_px / 2)), int(round(v - korkeus_px / 2))), (int(round(u + leveys_px / 2)), int(round(v + korkeus_px / 2))), vari, -1)

    def tausta():
        return np.full((kuvain.h, kuvain.w, 3), 232, np.uint8)

    def etsi(kuva):
        return mk.etsi(mitta, cv2.GaussianBlur(kuva, (0, 0), 0.8))

    oranssi = bgr(14, 165, 190)
    X0, Y0 = -150.0, -30.0
    kuva = tausta()
    suorakulmio(kuva, X0, Y0, 3.0, 26, 12, oranssi)
    maali, ehd = etsi(kuva)
    tarkista("harjanpaa loytyy (<= 6 cm akselista)", maali is not None and np.hypot(maali["X"] - X0, maali["Y"] - Y0) <= 6.0,
             "ei maalia" if maali is None else f"({maali['X']:.0f}, {maali['Y']:.0f})")
    tarkista("harjanpaan pitka akseli on vaakasuuntainen (kulma <= 35 astetta)", maali is not None and maali["kulma"] <= mk.MERKKI_KULMA_MAX, "" if maali is None else f"{maali['kulma']:.0f} deg")
    u0, v0 = (float(q[0]) for q in m.maa_oikaistuun([X0], [Y0], 3.0))
    suunta = 1.0 if float(m.maa_oikaistuun([X0], [Y0 + 10.0], 3.0)[0][0]) > u0 else -1.0                     # Y':n kasvusuunta (heittajaa kohti) oikaistussa kuvassa
    ua, va = (float(q[0]) for q in m.maa_oikaistuun([maali["alin_X"]], [maali["alin_Y"]], mk.Z_ALIN_CM))
    tarkista("alin kohta: heittajaa lahin paa (suorakulmion alareuna +- 2,5 px, leveyden keskella) ja Y' suurempi kuin keskipisteella",
             maali is not None and abs(ua - (u0 + suunta * 13.0)) <= 2.5 and abs(va - v0) <= 4.0 and maali["alin_Y"] > maali["Y"], "" if maali is None else f"du {ua - u0:+.1f}, dv {va - v0:+.1f}, dY {maali['alin_Y'] - maali['Y']:+.1f}")
    r = mk._reuna(mitta)
    tarkista("kuvan reunavyohyke: kuvan reunat mukana, keskusta ei", bool(r[0, :].all() and r[:, -1].all() and r[r.shape[0] // 2, r.shape[1] // 2] == 0))
    tarkista("harjanpaa keskella kuvaa ei ole kuvan reunalla", maali is not None and maali["reunalla"] is False)
    mitta._merkki_reuna = np.ones((kuvain.h, kuvain.w), np.uint8)                                              # koko kuva reunavyohykkeena: harjanpaa koskettaa reunaa
    maali_r, _ = etsi(kuva)
    del mitta._merkki_reuna
    tarkista("reunavyohykkeeseen osuva harjanpaa on merkitty reunalla-lipulla", maali_r is not None and maali_r["reunalla"] is True)
    kuva = tausta()
    suorakulmio(kuva, 20.0, 30.0, 12.0, 12, 26, oranssi)                                                       # kiven kahva: sama savy, pystysuuntainen
    maali, ehd = etsi(kuva)
    tarkista("kiven kahva (pystysuuntainen, sama savy) ei ole maali", maali is None and any(e["syy"] == "suunta" for e in ehd), str([(round(e["X"]), round(e["Y"]), e["syy"]) for e in ehd]))
    kuva = tausta()
    u, v = (float(q[0]) for q in m.maa_oikaistuun([0.0], [0.0], 3.0))
    v_lattia = float(m.maa_oikaistuun([262.0], [0.0], 3.0)[1][0])
    kuva[int(v_lattia):, :] = oranssi                                                                           # puulattia jaan reunan takana (|X'| > 245)
    maali, ehd = etsi(kuva)
    tarkista("puulattia (jaan ulkopuolella) ei ole maali", maali is None, "" if maali is None else f"({maali['X']:.0f}, {maali['Y']:.0f})")
    maali, ehd = etsi(tausta())
    tarkista("tyhja kuva: ei maalia, ei ehdokkaita", maali is None and not ehd)
    kuva = tausta()
    suorakulmio(kuva, X0, Y0, 3.0, 26, 12, oranssi)
    suorakulmio(kuva, X0, Y0, 3.0, 70, 5, oranssi, siirto_u=-48.0)                                              # puuvarsi harjanpaan vieressa (sama savy, kapea)
    maali, ehd = etsi(kuva)
    tarkista("harjanpaa + varsi: varsi irrotetaan, maali harjanpaan kohdalla (<= 10 cm)", maali is not None and np.hypot(maali["X"] - X0, maali["Y"] - Y0) <= 10.0,
             "ei maalia" if maali is None else f"({maali['X']:.0f}, {maali['Y']:.0f}) ala {maali['ala_cm2']:.0f}")
    kuva = tausta()
    suorakulmio(kuva, X0, Y0, 3.0, 26, 12, bgr(10, 130, 135))                                                   # varjossa oleva ruskea harjanpaa
    maali, ehd = etsi(kuva)
    tarkista("varjossa oleva, ruskeampi harjanpaa loytyy", maali is not None and np.hypot(maali["X"] - X0, maali["Y"] - Y0) <= 6.0)
    kuva = tausta()
    suorakulmio(kuva, 60.0, -30.0, 3.0, 22, 10, oranssi)                                                        # pesassa (pienempi) ja pesan takana lojuva harja (suurempi)
    suorakulmio(kuva, -40.0, -260.0, 3.0, 26, 12, oranssi)
    maali, ehd = etsi(kuva)
    tarkista("pesan takana (Y' < TAKA_Y_MIN_CM) oleva harja ei ole maali, vaikka on suurempi", maali is not None and abs(maali["X"] - 60.0) <= 6.0 and maali["Y"] > mk.TAKA_Y_MIN_CM,
             "ei maalia" if maali is None else f"({maali['X']:.0f}, {maali['Y']:.0f})")
    kuva = tausta()
    suorakulmio(kuva, -40.0, 330.0, 3.0, 26, 12, oranssi)
    maali, ehd = etsi(kuva)
    tarkista("pesan edessa (vartijakivi-maali): kelpaa", maali is not None and abs(maali["Y"] - 330.0) <= 8.0, "ei maalia" if maali is None else f"({maali['X']:.0f}, {maali['Y']:.0f})")


def testaa_merkki_konsensus():
    print("maalimerkki, moniruutuhaku (merkki.konsensus, MerkkiSeuranta, Analyysi.merkki_alku / merkki_valmiit): epavakaa, myohemmin tuotu, lojuva ja pieni (kahva) harja")
    import merkki as mk

    def harja(X, Y, ala=330.0, pit=33.0, kulma=0.0, reunalla=False):
        return dict(X=X, Y=Y, alin_X=X, alin_Y=Y + 20.0, ala_cm2=ala, pituus_cm=pit, kulma=kulma, reunalla=reunalla, kelvollinen=True)         # alin kohta 20 cm keskipisteen takana (heittajaa kohti)

    ajat = [-10.0 + 0.5 * i for i in range(45)]                                    # -10 ... +12 s
    def sarja(f):
        return [(t, f(t)) for t in ajat]

    k = mk.konsensus(sarja(lambda t: [harja(100.0, -30.0)]))
    tarkista("vakaa harja: loytyy, esiintyma 1, yksiselitteinen", k is not None and abs(k["X"] - 100.0) < 1 and k["esiintyma"] == 1.0 and k["ehdokkaita"] == 1, str(k and (k["X"], k["esiintyma"], k["ehdokkaita"])))
    k = mk.konsensus(sarja(lambda t: [harja(100.0 + 0.2 * (hash(t) % 3), -30.0)] if int(t * 2) % 5 in (0, 1) else []))                 # nakyy vain 40 % ruuduista (epavakaa havainto)
    tarkista("epavakaa havainto (40 % ruuduista) loytyy", k is not None and abs(k["X"] - 100.0) < 1 and 0.3 <= k["esiintyma"] <= 0.6, str(k and k["esiintyma"]))
    k = mk.konsensus(sarja(lambda t: [harja(60.0, -20.0)] if t >= 2.0 else []))
    tarkista("heiton alkua myohemmin (+2 s) paikalle tuotu harja loytyy (yhden ruudun haku heiton alussa ei loytaisi)", k is not None and abs(k["X"] - 60.0) < 1 and k["esiintyma"] < 0.6, str(k and k["esiintyma"]))
    k = mk.konsensus(sarja(lambda t: [harja(100.0, -30.0)] if t <= -3.0 else []))
    tarkista("harja poistunut ennen tavoiteikkunaa: ei maalia", k is None)
    k = mk.konsensus(sarja(lambda t: []))
    tarkista("ei havaintoja: ei maalia", k is None)
    k = mk.konsensus(sarja(lambda t: [harja(100.0, -30.0)] if 0.0 <= t <= 1.0 else []))
    tarkista("liian lyhyt esiintyma (alle MIN_ESIINTYMA): ei maalia", k is None)
    k = mk.konsensus(sarja(lambda t: [harja(40.0, -260.0, ala=420.0)] + ([harja(120.0, -25.0, ala=300.0)] if t <= 4.5 else [])))
    tarkista("pesan takana lojuva (suurempi, pysyva) harja ei voita pesan harjaa", k is not None and abs(k["X"] - 120.0) < 1 and k["Y"] > mk.TAKA_Y_MIN_CM, str(k and (k["X"], k["Y"])))
    k = mk.konsensus(sarja(lambda t: [harja(-170.0, -30.0, ala=330.0), harja(20.0, 10.0, ala=90.0, pit=13.0)]))
    tarkista("pysyva pieni (kiven kahvan kokoinen) ehdokas ei voita harjanpaata", k is not None and abs(k["X"] + 170.0) < 1, str(k and k["X"]))
    k = mk.konsensus(sarja(lambda t: [harja(-170.0, -30.0), harja(180.0, -30.0)]))
    tarkista("kaksi yhta hyvaa harjaa: ehdokkaita 2", k is not None and k["ehdokkaita"] == 2, str(k and k["ehdokkaita"]))
    k = mk.konsensus(sarja(lambda t: [harja(-50.0, 10.0, ala=400.0, pit=39.0), harja(-46.0, 138.0, ala=330.0, pit=38.0)]))
    tarkista("harjanpaa + samalla linjalla yhtaikaa nakyva isompi varsi/kahvaosa (Video2 492): harjanpaa on heittajan puoleisin, yksiselitteinen",
             k is not None and abs(k["keski_Y"] - 138.0) < 1 and abs(k["X"] + 46.0) < 1 and k["ehdokkaita"] == 1, str(k and (k["X"], k["keski_Y"], k["ehdokkaita"])))
    k = mk.konsensus(sarja(lambda t: [harja(138.0, -30.0)] if t < 2.5 else [harja(136.0, 11.0)]))
    tarkista("sama harjanpaa siirtyy toiseen paikkaan (ei yhtaikaa nakyvia): ei vaihdeta heittajan puoleisempaan, yksiselitteinen",
             k is not None and abs(k["keski_Y"] + 30.0) < 1 and k["ehdokkaita"] == 1, str(k and (k["keski_Y"], k["ehdokkaita"])))
    k = mk.konsensus(sarja(lambda t: [harja(-50.0, 10.0, ala=400.0), harja(-50.0, 300.0, ala=330.0)]))
    tarkista("saman linjan osat yli LINJA_DY_CM:n paassa toisistaan ovat eri harjoja (ehdokkaita 2, isompi voittaa)", k is not None and k["ehdokkaita"] == 2 and abs(k["keski_Y"] - 10.0) < 1, str(k and (k["keski_Y"], k["ehdokkaita"])))
    k = mk.konsensus(sarja(lambda t: [harja(100.0, -30.0)] if t < 0 else [harja(100.0, -30.0)]))
    tarkista("tulos on aikaikkunan mediaani (ei ensimmaista havaintoa)", k is not None and abs(k["keski_Y"] + 30.0) < 1e-6)
    tarkista("maali on harjanpaan alin kohta (heittajaa lahin paa), keskipiste erikseen", k is not None and abs(k["Y"] + 10.0) < 1e-6 and abs(k["X"] - 100.0) < 1e-6 and abs(k["keski_Y"] + 30.0) < 1e-6,
             str(k and (k["X"], k["Y"], k["keski_Y"])))
    # kuvan reunan katkaisema harja: nakyy vain hetken (esim. kippari seisoo laidalla, harjanpaa on suurimman osan ajasta kuvan ulkopuolella)
    k = mk.konsensus(sarja(lambda t: [harja(-239.0, 259.0, ala=94.0, reunalla=True)] if t == 5.5 else []))
    tarkista("kuvan reunan katkaisema harja (yksi havainto, tavoiteikkunan ulkopuolella): loytyy, reunalla, esiintyma koko ikkunasta",
             k is not None and k["reunalla"] and abs(k["X"] + 239.0) < 1 and 0.0 < k["esiintyma"] < 0.1, str(k and (k["X"], k["esiintyma"], k["reunalla"])))
    k = mk.konsensus(sarja(lambda t: [harja(-239.0, 259.0, ala=94.0)] if t == 5.5 else []))
    tarkista("sama hetkellinen havainto ilman reunalla-lippua: ei maalia", k is None)
    k = mk.konsensus(sarja(lambda t: ([harja(-239.0, 259.0, ala=94.0, reunalla=True)] if t == 5.5 else []) + [harja(100.0, -30.0)]))
    tarkista("tavallinen harja voittaa reunan katkaiseman (reunaharja vain, jos tavallista ei ole)", k is not None and abs(k["X"] - 100.0) < 1 and not k["reunalla"])
    k = mk.konsensus(sarja(lambda t: [harja(-239.0, -250.0, ala=94.0, reunalla=True)] if t == 5.5 else []))
    tarkista("reunalla oleva mutta pesan takana oleva havainto: ei maalia", k is None)
    k = mk.konsensus(sarja(lambda t: [harja(-239.0, 259.0, ala=94.0, reunalla=True)] if t >= -2.0 else []))
    tarkista("reunalla oleva harja, joka nakyy koko ajan: tavallinen maali, reunalla-lippu paalla", k is not None and k["esiintyma"] == 1.0 and k["reunalla"], str(k and (k["esiintyma"], k["reunalla"])))

    # T-viivapaikka: suora viiva hackista maalin kautta kaukopaan T-viivalle
    L = 3474.7
    tarkista("t_viivalle: maali T-viivalla keskella -> 0", abs(mk.t_viivalle(0.0, 0.0, 0.0, -366.0, L)) < 1e-9)
    tarkista("t_viivalle: maali T-viivan kohdalla (Y' = 0) on samassa X:ssa riippumatta hackista (X' on peilikuva)", abs(mk.t_viivalle(-50.0, 0.0, 14.0, -366.0, L) - 50.0) < 1e-9)
    X_T = mk.t_viivalle(-50.0, 100.0, 0.0, -366.0, L)
    tarkista("t_viivalle: maali 100 cm ennen T-viivaa: viiva jatkuu T-viivalle ja X kasvaa suhteessa etaisyyteen", abs(X_T - 50.0 * (L + 366.0) / (L - 100.0 + 366.0)) < 1e-9 and X_T > 50.0, f"{X_T:.3f}")
    tarkista("t_viivalle: maali 100 cm T-viivan takana: X pienenee", mk.t_viivalle(-50.0, -100.0, 0.0, -366.0, L) < 50.0)
    hx, hy, Xm, Ym = 14.0, -366.0, 80.0, L - 40.0
    X_T = mk.t_viivalle(-80.0, 40.0, hx, hy, L)
    tarkista("t_viivalle: hack, maali ja T-viivan piste ovat samalla suoralla", abs((X_T - hx) / (L - hy) - (Xm - hx) / (Ym - hy)) < 1e-12)

    # skannaus: molempien puoliskojen sarjat, vali, puskurin rajaus
    kutsut = []

    def feikki(m, ruutu):
        kutsut.append(m)
        f = ruutu
        return [dict(harja(100.0, -30.0), kelvollinen=True)] if m == "A" else ([dict(harja(-50.0, 10.0), kelvollinen=True)] if f >= 60 else [])
    vanha = mk.ehdokkaat
    mk.ehdokkaat = feikki
    try:
        ms = mk.MerkkiSeuranta({"vasen": "A", "oikea": "B"}, 30.0)
        n = sum(ms.syota(f, f) for f in range(0, 1200))
        tarkista("skannaus 0,5 s valein (30 r/s: 15 ruudun valein)", n == 80, str(n))
        tarkista("molemmat puoliskot skannataan joka kerta", len(kutsut) == 160 and kutsut.count("A") == 80)
        tarkista("puskuri rajautuu (ei kasva rajatta)", len(ms.sarjat["vasen"]) < 80 and ms.sarjat["vasen"][0][0] >= 1185 - ms.puskuri - 15)
        k = ms.konsensus("vasen", 1000)
        tarkista("MerkkiSeuranta.konsensus: heiton ymparilta (lahto ruudussa 1000)", k is not None and abs(k["X"] - 100.0) < 1)
        k = ms.konsensus("oikea", 1000)
        tarkista("MerkkiSeuranta.konsensus: toinen puolisko omilla havainnoillaan", k is not None and abs(k["X"] + 50.0) < 1)
        tarkista("f_asti rajaa havainnot (ennen puskurin alkua ei mitaan)", ms.konsensus("vasen", 1000, f_asti=700) is None)
    finally:
        mk.ehdokkaat = vanha

    # Analyysi: alustava merkki heiton lahdossa, lopullinen konsensusikkunan lopussa
    kansio = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tulokset_nettivideo3")
    an = ns.Analyysi(kansio, fps=30.0, saikeet=0, tarkka=False, synkroninen=True, fps_lahde=None, kierre=False)
    mk.ehdokkaat = lambda m, ruutu: ([harja(80.0, -30.0)] if ruutu >= 100 else [])
    try:
        h = ns.Heitto(1, "vasen", "punainen", 0, (0.0, 0.0))
        h.f_alku = 300
        for f in range(0, 361, 3):                                             # (analyysin syotto: skannaus vain 15 ruudun valein)
            an.merkit.syota(f, f)
        an.merkki_alku(h, 360)
        tarkista("heiton lahdossa alustava merkki skannauksista tahan asti", h.merkki is not None and "X" in h.merkki and abs(h.merkki["X"] - 80.0) < 1, str(h.merkki))
        n_odottaa = len(an._merkki_odottavat)
        an.merkki_valmiit(400)
        tarkista("lopullista ei viela ole ennen konsensusikkunan loppua", len(an._merkki_odottavat) == n_odottaa == 1)
        for f in range(363, 700, 3):
            an.merkit.syota(f, f)
        an.merkki_valmiit(300 + int(mk.KONSENSUS_LOPPU_S * 30) + 1)
        tarkista("konsensusikkunan lopussa merkki viimeistellaan (kaikki ruudut mukana)", len(an._merkki_odottavat) == 0 and h.merkki["esiintyma"] == 1.0, str(h.merkki))
        Lr = ns.ETAISYYS_CM
        tarkista("merkin T-viivapaikka (h.lepo = hack (0, 0)): alin kohta (X' 80, Y' -10) -> -80 * L / (L + 10)", abs(h.merkki["T_cm"] + 80.0 * Lr / (Lr + 10.0)) < 1e-6, str(h.merkki.get("T_cm")))
        h4 = ns.Heitto(4, "vasen", "punainen", 0, (12.0, -366.0)); h4.f_alku = 300
        an.merkki_alku(h4, 700)
        tarkista("merkin T-viivapaikka kayttaa heiton hackia (h.lepo = (12, -366))", abs(h4.merkki["T_cm"] - mk.t_viivalle(80.0, -10.0, 12.0, -366.0, Lr)) < 1e-9 and abs(h4.merkki["T_cm"] - h.merkki["T_cm"]) > 1e-3, str(h4.merkki.get("T_cm")))
        h2 = ns.Heitto(2, "oikea", "keltainen", 0, (0.0, 0.0))
        h2.f_alku = 500
        an.merkki_alku(h2, 505)                                                # kaukopaa = vasen: ruudut >= 100 nakevat harjan; oikean heiton kaukopaa on vasen
        an.merkki_valmiit(None, kaikki=True)
        tarkista("videon loppu viimeistelee kesken olevat", len(an._merkki_odottavat) == 0 and h2.merkki is not None)
        an2 = ns.Analyysi(kansio, fps=30.0, saikeet=0, tarkka=False, synkroninen=True, fps_lahde=None, kierre=False)
        h3 = ns.Heitto(3, "vasen", "punainen", 0, (0.0, 0.0)); h3.f_alku = 100
        an2.merkki_alku(h3, 110)
        tarkista("ei skannauksia: merkki on dict(ehdokkaita=0), ei None eika poikkeus", h3.merkki == dict(ehdokkaita=0), str(h3.merkki))
    finally:
        mk.ehdokkaat = vanha


def testaa_paneeli_merkista():
    print("paneeli: Merkki-rivi (T-viivapaikka); Liuku ja Heitto merkista (+ = kierteesta ulospain; myotapaivaan -> vasemmalle), ilman merkkia keskiviivasta, ? kunnes kierron suunta tiedetaan; Irroitus = Heitto - Liuku")
    import types
    import netti_piirto as npi

    def heitto(merkki, omega):
        viivat = dict(liuku_a=10.0, liuku_b=0.0, irr_X=0.0, irr_Y=400.0, irr_suunta_deg=0.0, irr_virhe_deg=0.1, nopeus_ennen=2.0, nopeus_jalkeen=2.0, vapaa_hidastuvuus=0.05,
                      vapaa_matka_cm=300.0, vakaa=True)                                    # liuku osuu T-viivalle kohtaan +10 cm, irroitussuora kohtaan 0 cm
        kierto = None if omega is None else dict(omega_deg_s=omega, varma=True)
        return types.SimpleNamespace(viivat=viivat, merkki=merkki, kierto=kierto, kauko_kierto=None, kierre=None, kauko=None, vakiot=None, vari="punainen", kivi_nro=3, tunnus=1,
                                     kierto_tila="valmis", saapuminen=None, kauko_kierto_tila=None)

    def rivit(h):
        s = ns.heiton_suureet(h)
        d = {}
        for t, _, lisa in npi.paneelin_rivit(h, s):
            d.setdefault(t.split(":")[0], (t, lisa))                                   # (kaksi 'Irroitus'-riviä: ensimmäinen on heitto - liuku)
        return s, d

    s, r = rivit(heitto(dict(X=0.0, Y=0.0, ehdokkaita=1, T_cm=-15.0), +50.0))                 # vastapaivaan (kaartaa vasemmalle), merkki T-viivalla -15 cm: suorat ovat merkin oikealla puolella = ulkopuolella
    tarkista("vastapaivaan, merkki -15 cm: liuku +25 cm, heitto +15 cm merkista (+ = ulos, tassa oikealle)", abs(s["liuku_merkista_cm"] - 25.0) < 1e-9 and abs(s["heitto_merkista_cm"] - 15.0) < 1e-9, str((s["liuku_merkista_cm"], s["heitto_merkista_cm"])))
    tarkista("paneeli: 'Liuku: +25cm' ja viite 'merkistä, + ulos', 'Heitto: +15cm'", r["Liuku"] == ("Liuku: +25cm", "merkistä, + ulos") and r["Heitto"][0] == "Heitto: +15cm", str(r["Liuku"] + r["Heitto"]))
    tarkista("Irroitus = Heitto - Liuku samalla etumerkilla (-10 cm)", r["Irroitus"][0] == "Irroitus: -10cm" and abs(s["irroitus_cm"] - (s["heitto_merkista_cm"] - s["liuku_merkista_cm"])) < 1e-9, r["Irroitus"][0])
    s, r = rivit(heitto(dict(X=0.0, Y=0.0, ehdokkaita=1, T_cm=35.0), -50.0))                  # myotapaivaan (kaartaa oikealle), merkki +35 cm: suorat ovat merkin vasemmalla puolella = ulkopuolella
    tarkista("myotapaivaan, merkki +35 cm: suorat merkin vasemmalla (ulkopuolella) -> liuku +25 cm, heitto +35 cm", abs(s["liuku_merkista_cm"] - 25.0) < 1e-9 and abs(s["heitto_merkista_cm"] - 35.0) < 1e-9, str((s["liuku_merkista_cm"], s["heitto_merkista_cm"])))
    s, r = rivit(heitto(dict(X=0.0, Y=0.0, ehdokkaita=1, T_cm=-15.0), -50.0))                 # myotapaivaan (kaartaa oikealle), suorat merkin oikealla = kaarron suuntaan = sisapuolella
    tarkista("myotapaivaan, suorat merkin oikealla (kaarron suuntaan): negatiivinen (-25, -15)", abs(s["liuku_merkista_cm"] + 25.0) < 1e-9 and abs(s["heitto_merkista_cm"] + 15.0) < 1e-9, str((s["liuku_merkista_cm"], s["heitto_merkista_cm"])))
    s, r = rivit(heitto(dict(X=0.0, Y=0.0, ehdokkaita=1, T_cm=-15.0), None))
    tarkista("merkki mutta kierron suunta tuntematon: ? (ei arvattua etumerkkia)", s["liuku_merkista_cm"] is None and r["Liuku"][0] == "Liuku: ?" and r["Heitto"][0] == "Heitto: ?" and s["merkki_T_cm"] == -15.0, str(r["Liuku"] + r["Heitto"]))
    h = heitto(dict(X=0.0, Y=0.0, ehdokkaita=1, T_cm=-15.0), +50.0)
    s, r = rivit(h)
    nimet = [t.split(":")[0] for t, _, _ in npi.paneelin_rivit(h, s)]
    tarkista("paneeli: Merkki-rivi (T-viivapaikka keskiviivasta) heti Kivi-rivin jalkeen, ennen Liukua", r["Merkki"] == ("Merkki: -15cm", "keskiviivasta") and nimet[:3] == ["Kivi", "Merkki", "Liuku"], str(r["Merkki"]) + str(nimet[:3]))
    p = npi.Piirtaja(None, fps=30.0)
    p._x = None
    kuva, maski = p._paneeli_kuva([(t, v, l) for t, v, l in npi.paneelin_rivit(h, s)], vlk="_x")
    tarkista("paneeli: yhdeksan riviä mahtuu laatikkoon (viimeinen rivi ei leikkaudu)", kuva.shape[0] >= 12 + npi.RIVI_PX * 9 and bool(maski[-24:].any()) and kuva.shape[0] <= npi.PANEELI_KORKEUS_MAX, f"korkeus {kuva.shape[0]}")
    s, r = rivit(heitto(dict(X=0.0, Y=0.0, ehdokkaita=1, T_cm=257.6, reunalla=True), +50.0))
    tarkista("Merkki-rivi: kuvan reunan katkaisema harja -> 'reunalla'", r["Merkki"] == ("Merkki: +258cm", "reunalla"), str(r["Merkki"]))
    s, r = rivit(heitto(dict(X=0.0, Y=0.0, ehdokkaita=2, T_cm=-15.0), +50.0))
    tarkista("Merkki-rivi: kaksi kilpailevaa harjaa -> '2 harjaa'", r["Merkki"] == ("Merkki: -15cm", "2 harjaa"), str(r["Merkki"]))
    s, r = rivit(heitto(dict(ehdokkaita=0), +50.0))
    tarkista("Merkki-rivi: merkkia ei loytynyt -> 'Merkki: ei', Liuku keskiviivasta", r["Merkki"][0] == "Merkki: ei" and r["Liuku"][1] == "keskiviivasta", str(r["Merkki"] + r["Liuku"]))
    s, r = rivit(heitto(None, +50.0))
    tarkista("ei Merkki-riviä, kun merkin hakua ei ole (h.merkki None): paneeli kuten ennen", "Merkki" not in r and r["Liuku"][1] == "keskiviivasta", str(list(r)))
    for merkki in (None, dict(ehdokkaita=0), dict(X=0.0, Y=0.0, ehdokkaita=1, T_cm=None)):
        s, r = rivit(heitto(merkki, +50.0))
        tarkista(f"ei merkkia ({merkki}): luvut keskiviivasta (+10, +0 cm, + = heittajan oikealle), viite 'keskiviivasta'",
                 s["merkki_T_cm"] is None and r["Liuku"] == ("Liuku: +10cm", "keskiviivasta") and r["Heitto"][0] == "Heitto: +0cm" and abs(s["liuku_cm"] - 10.0) < 1e-9, str(r["Liuku"] + r["Heitto"]))


def testaa_viivat_ilman_tekstia():
    print("piirto: liuku- ja irroituslinjoilla ei ole tekstia (luvut ovat tietopaneelissa), T-viivan leikkauspisteet ja viivat piirretaan")
    import types
    import netti_piirto as npi

    p = npi.Piirtaja({"vasen": object(), "oikea": object()}, fps=30.0)
    p._viivat_data = lambda h, kauko: dict(pts=np.array([[1100.0, 300.0], [1120.0, 700.0]]), pts2=np.array([[1160.0, 300.0], [1180.0, 700.0]]), XT_liuku=10.0, XT_irr=-5.0, alustava=False,
                                           uv_liuku=(1140.0, 420.0), uv_irr=(1300.0, 450.0))
    h = types.SimpleNamespace(viivat=dict(x=1), puoli="vasen")                          # kaukopaa = oikea puolisko
    kutsut = []
    vanha = npi._teksti
    npi._teksti = lambda *a, **k: kutsut.append(a) or (0, 0)
    try:
        img = np.zeros((1080, 1920, 3), np.uint8)
        p._viivat(img, h)
    finally:
        npi._teksti = vanha
    tarkista("viivoilla ei tekstia (ei LIUKU- eika IRROITUS-nimikkeita)", not kutsut, str(kutsut))
    tarkista("T-viivan leikkauspisteet piirretaan (liuku oranssi, irroitus magenta)", tuple(img[420, 1140]) == npi.VARI_LIUKU and tuple(img[450, 1300]) == npi.VARI_IRROITUS, f"{img[420, 1140].tolist()} {img[450, 1300].tolist()}")
    tarkista("viivat piirretaan", int((img[500, 1090:1150] != 0).any(axis=1).sum()) > 0 and int((img[500, 1150:1210] != 0).any(axis=1).sum()) > 0)


def testaa_piirto_liuku():
    print("piirto: koko mitattu pistejono (liuku + irroituksen jalkeinen lento) kaikki kerralla heti kun heiton alku on tulosteessa, irroituksen kohdalla musta ympyra (kun viivaestimaatti on valmis)")
    import types
    import cv2
    import netti_piirto as npi

    class Geom:
        @staticmethod
        def maa_kuvaan(X, Y, z):
            return np.column_stack([np.asarray(X, float) + 100.0, np.asarray(Y, float) * 0.5 + 50.0])
    h = types.SimpleNamespace(tunnus=1, puoli="vasen", vari="punainen", kivi_nro=1, f_alku=100, f_loppu=400, naytteet=[], b_laskuri=0, viivat=None,
                              pisteet=lambda: [(100 + i, 0.0, 10.0 * i) for i in range(40)])             # piste i: kuvassa (100, 50 + 5 i)
    p = npi.Piirtaja({"vasen": Geom(), "oikea": Geom()}, fps=30.0)

    def kuva(f_out):
        img = np.full((300, 400, 3), 255, np.uint8)
        p._pisteet(img, h, f_out)
        return img

    def vihrea(img, i):
        return bool(np.abs(img[50 + 5 * i - 1: 50 + 5 * i + 2, 98:103].astype(int).reshape(-1, 3) - np.array(npi.VARI_PISTE)).sum(axis=1).min() <= 40)       # (reunapisteet on mediaanisilotettu: kayta sisapisteita)

    def keltainen_rengas(img, i):
        alue = img[50 + 5 * i - 1: 50 + 5 * i + 2, 100 + 5: 100 + 10].astype(int).reshape(-1, 3)
        return bool(np.abs(alue - np.array(npi.VARI_NYKYINEN)).sum(axis=1).min() <= 90)

    sisapisteet = (2, 3, 10, 15, 19, 20, 25, 30, 36)
    img = kuva(105)                                                            # kivi on pisteessa 5
    tarkista("ilman estimaattia: koko pistejono kerralla (myos kivea edella olevat pisteet)", all(vihrea(img, i) for i in sisapisteet))
    tarkista("ilman estimaattia: ei mustaa ympyraa", int((img.sum(axis=2) <= 30).sum()) == 0)
    tarkista("kivi (piste 5) on rengas nykyisessa paikassa, ei jonon lopussa", keltainen_rengas(img, 5) and not keltainen_rengas(img, 36))
    h.viivat = dict(irr_X=0.0, irr_Y=200.0)                                    # irroitus pisteen 20 kohdalla (v = 150)
    img = kuva(105)
    tarkista("estimaatti valmis: koko pistejono kerralla, myos irroituksen jalkeiset pisteet (25, 30, 36) eivat lopu irroitukseen", all(vihrea(img, i) for i in sisapisteet))
    rengas = img[150, 100 + npi.IRROITUSPISTE_R - 1: 100 + npi.IRROITUSPISTE_R + 2]
    tarkista("irroituksen kohdalla musta ympyra (sade IRROITUSPISTE_R)", int(rengas.sum(axis=1).min()) <= 60, str(rengas.tolist()))
    tarkista("ympyran ulkopuolella (sade + 5 px) jaa valkoista", bool((img[150, 100 + npi.IRROITUSPISTE_R + 5] == 255).all()))
    tarkista("kivi (piste 5) on edelleen rengas nykyisessa paikassa, ei irroituksessa", keltainen_rengas(img, 5) and not keltainen_rengas(img, 20))
    img = kuva(135)                                                            # kivi on pisteessa 35, jonon loppuvaiheessa
    tarkista("kivi seuraa jonoa: rengas pisteessa 35, aiemmat ja myohemmat pisteet edelleen nakyvissa", keltainen_rengas(img, 35) and all(vihrea(img, i) for i in (2, 10, 20, 30)))
    img = kuva(99)
    tarkista("ennen lahtoa (f_out < f_alku) ei piirreta mitaan", bool((img == 255).all()))
    h.viivat = dict(virhe="x")
    img = kuva(105)
    tarkista("viivaestimaatin virhetulos (ei irr_X) ei riko piirtoa: pistejono ilman ympyraa", all(vihrea(img, i) for i in sisapisteet) and int((img.sum(axis=2) <= 30).sum()) == 0)
    h.viivat = dict(irr_X=0.0, irr_Y=200.0)
    h.pisteet = lambda: [(100 + i, 0.0, 10.0 * i) for i in range(25)]          # analyysi on tulosteen perassa: vain 25 pistetta tiedossa
    h.b_laskuri = 1                                                            # (pistejonon valimuistin avain muuttuu, kun aineisto muuttuu)
    img = kuva(105)
    tarkista("analyysi jaljessa: tiedossa olevat pisteet kerralla, tuntemattomia (>= 25) ei piirreta", all(vihrea(img, i) for i in (2, 10, 20)) and not vihrea(img, 30))


def testaa_kopioruutu():
    print("kopioruututesti (Analyysi._kopio): kaistoittainen varhainen lopetus antaa saman paatoksen kuin koko kuvan harmaasavyvertailu")
    import cv2
    import kivi_netti as kv
    kansio = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tulokset_nettivideo3")
    an = ns.Analyysi(kansio, fps=30.0, saikeet=0, tarkka=False, synkroninen=True, fps_lahde=None, kierre=False)
    y0, h = kv.kn.PUOLET["vasen"][1], kv.kn.PUOLET["vasen"][3]

    class Vanha:
        edellinen = None

        def paatos(self, ruutu):
            g = cv2.cvtColor(ruutu[y0:y0 + h], cv2.COLOR_BGR2GRAY)
            kopio = False
            if self.edellinen is not None:
                d = cv2.absdiff(g, self.edellinen)
                puoli = d.shape[1] // 2
                raja = 0.001 * d.shape[0] * puoli
                kopio = bool(np.count_nonzero(d[:, :puoli] > ns.KOPIO_KYNNYS) < raja and np.count_nonzero(d[:, puoli:] > ns.KOPIO_KYNNYS) < raja)
            self.edellinen = g
            return kopio
    v = Vanha()
    rng = np.random.default_rng(11)
    kuva = rng.integers(0, 256, (1080, 1920, 3), dtype=np.uint8)
    erot, kopioita, n = [], 0, 0

    def vertaa(nimi, ruutu):
        nonlocal kopioita, n
        a, b = an._kopio(ruutu), v.paatos(ruutu)
        n += 1
        kopioita += b
        if a != b:
            erot.append(nimi)
        return a
    tarkista("ensimmainen ruutu: ei kopio", vertaa("eka", kuva) is False)
    tarkista("tarkka kopio: kopio", vertaa("kopio", kuva.copy()) is True)
    kohina = np.clip(kuva.astype(np.int16) + rng.integers(-3, 4, kuva.shape), 0, 255).astype(np.uint8)
    tarkista("pieni kohina (< KOPIO_KYNNYS): kopio", vertaa("kohina", kohina) is True)
    ed = kohina.copy()
    ed[900:1000, 100:400] = 255 - ed[900:1000, 100:400]                                         # 30 000 px vasemmassa puoliskossa, alhaalla
    tarkista("suuri muutos (alhaalla vasemmalla): ei kopio", vertaa("alas", ed) is False)
    tarkista("sama ruutu uudelleen: kopio", vertaa("alas-kopio", ed.copy()) is True)
    ed2 = ed.copy()
    ed2[300:380, 1200:1400] = 255 - ed2[300:380, 1200:1400]                                   # ylhaalla oikealla (eri kaista kuin edellinen ero)
    tarkista("muutos toisessa kaistassa ja puoliskossa: ei kopio", vertaa("ylos", ed2) is False)
    # useita satunnaisia muutoksia: pienia (alle raja), isoja, kaistojen rajoilla, kummassakin puoliskossa
    nykyinen = ed2
    for i in range(60):
        uusi = nykyinen.copy()
        for _ in range(int(rng.integers(0, 3))):
            ph, pw = int(rng.integers(1, 40)) * (1 if rng.random() < 0.5 else 8), int(rng.integers(1, 60)) * (1 if rng.random() < 0.5 else 8)
            ry = int(rng.integers(y0 - 20, 1080 - ph)) if rng.random() < 0.3 else int(rng.integers(y0, y0 + h - ph))
            rx = int(rng.integers(0, 1920 - pw))
            uusi[max(ry, 0):ry + ph, rx:rx + pw] = rng.integers(0, 256, (uusi[max(ry, 0):ry + ph, rx:rx + pw].shape), dtype=np.uint8)
        vertaa(f"satunnainen {i}", uusi)
        nykyinen = uusi if rng.random() < 0.7 else nykyinen
    tarkista(f"satunnaiset muutokset: sama paatos kuin vanhalla vertailulla ({n} ruutua, {kopioita} kopiota)", not erot and 5 < kopioita < n - 5, str(erot[:5]))
    an.ohita()
    tarkista("ohita(): seuraavaa ruutua ei verrata vanhaan (ei kopio, vaikka identtinen)", an._kopio(nykyinen) is False)
    tarkista("ohituksen jalkeen vertailu jatkuu normaalisti", an._kopio(nykyinen.copy()) is True)


def testaa_kahva_maski():
    print("kahvan varimaski (kivi_netti.kahva_maski): inRange-toteutus antaa saman maskin kuin alkuperainen numpy-lauseke")
    import cv2
    import kivi_netti as kv

    def vanha(frame, vari):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        if vari == "keltainen":
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.int16)
            m = ((lab[..., 2] - 128) >= 16) & ((lab[..., 1] - 128) <= 10)
        else:
            m = ((hsv[..., 0] <= kv.PUNAINEN_H[0]) | (hsv[..., 0] >= kv.PUNAINEN_H[1])) & (hsv[..., 1] >= kv.PUNAINEN_SMIN) & (hsv[..., 2] >= kv.PUNAINEN_VMIN) & (hsv[..., 2] <= kv.PUNAINEN_VMAX)
        return m.astype(np.uint8)
    rng = np.random.default_rng(2)
    erot = 0
    for _ in range(120):
        kuva = rng.integers(0, 256, (int(rng.integers(8, 70)), int(rng.integers(8, 90)), 3), dtype=np.uint8)
        for vari in ("keltainen", "punainen"):
            a, b = kv.kahva_maski(kuva, vari), vanha(kuva, vari)
            erot += int(a.dtype != b.dtype or a.shape != b.shape or not np.array_equal(a, b))
    tarkista("satunnaiskuvat, molemmat varit: maskit identtiset (uint8 0/1)", erot == 0, f"{erot} erilaista")
    kuva = np.zeros((20, 20, 3), np.uint8)
    kuva[5:15, 5:15] = (0, 220, 235)                                                         # keltainen (BGR)
    tarkista("keltainen alue: maskissa", int(kv.kahva_maski(kuva, "keltainen").sum()) > 50 and int(kv.kahva_maski(kuva, "punainen").sum()) == 0)


def testaa_jaettu_leike():
    print("A-tason haku: kahvan molemmat varit samasta leikkeesta (Mittari.jaettu_leike): yksi remap, sama tulos")
    kansio = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tulokset_nettivideo3")
    m = ns.Mittari(kansio, "vasen", runko=False)
    kutsut = []
    alkuperainen = m.kuvain.leike
    m.kuvain.leike = lambda ruutu, *alue: kutsut.append(alue) or alkuperainen(ruutu, *alue)
    ruutu = np.zeros((1080, 1920, 3), np.uint8)
    ruutu[600:640, 300:340] = (0, 220, 235)
    alue = (200, 450, 500, 700)
    a = [m.kahva(ruutu, alue, v, None, min_ala=4) for v in ("keltainen", "punainen")]
    tarkista("ilman jaettua leikett: remap kummallekin varille", len(kutsut) == 2, str(len(kutsut)))
    kutsut.clear()
    with m.jaettu_leike(ruutu, alue):
        b = [m.kahva(ruutu, alue, v, None, min_ala=4) for v in ("keltainen", "punainen")]
        m.kahva(ruutu, (200, 450, 480, 700), "keltainen", None, min_ala=4)                       # toinen alue: oma remap
    tarkista("jaetulla leikkeella yksi remap kahdelle varille (+ yksi toiselle alueelle)", len(kutsut) == 2, str(len(kutsut)))
    tarkista("tulos sama kuin ilman jakoa", a == b and len(a[0]) == 1, f"{a} {b}")
    kutsut.clear()
    m.kahva(ruutu, alue, "keltainen", None, min_ala=4)
    tarkista("with-lohkon jalkeen muisti on tyhja (uusi remap)", len(kutsut) == 1 and m._jaettu is None)
    ruutu2 = ruutu.copy()
    with m.jaettu_leike(ruutu, alue):
        m.kahva(ruutu, alue, "keltainen", None, min_ala=4)
        kutsut.clear()
        m.kahva(ruutu2, alue, "keltainen", None, min_ala=4)                                       # eri ruutu: ei kayteta muistia
    tarkista("eri ruutu samassa lohkossa: oma remap", len(kutsut) == 1)


def testaa_merkki_taustaskannaus():
    print("maalimerkin taustaskannaus (MerkkiSeuranta pool + tyo): tulos sama kuin suoraan skannattaessa, vaikka tulokset valmistuvat eri jarjestyksessa; konsensus odottaa; epaonnistunut tyo tutkitaan paikan paalla")
    import concurrent.futures
    import threading
    import time
    import merkki as mk

    def harja(X, Y, ala=330.0):
        return dict(X=X, Y=Y, alin_X=X, alin_Y=Y + 20.0, ala_cm2=ala, pituus_cm=33.0, kulma=0.0, reunalla=False, kelvollinen=True)

    def ehd(m, f):
        return [harja(100.0, -30.0)] if m == "A" else ([harja(-50.0, 10.0)] if f >= 60 else []) + ([dict(harja(0.0, 0.0), kelvollinen=False)] if f % 7 == 0 else [])
    vanha = (mk.ehdokkaat, mk.leike, mk.ehdokkaat_leikkeesta)
    mk.ehdokkaat = lambda m, ruutu: ehd(m, ruutu)
    mk.leike = lambda m, ruutu: (ruutu, 0, 0)
    mk.ehdokkaat_leikkeesta = lambda m, roi, x0, y0: ehd(m, roi)
    mittarit = {"vasen": "A", "oikea": "B"}

    def tyo(args):
        puoli, roi, x0, y0 = args
        return [e for e in mk.ehdokkaat_leikkeesta(mittarit[puoli], roi, x0, y0) if e["kelvollinen"]]

    class Pooli:
        """submit palauttaa Future-olion, jonka tuloksen testi asettaa itse (mielivaltaisessa jarjestyksessa)."""
        def __init__(self):
            self.tyot = []

        def submit(self, fn, arg):
            fut = concurrent.futures.Future()
            self.tyot.append((fut, fn, arg))
            return fut

        def valmista(self, jarjestys, virheet=()):
            for i in jarjestys:
                fut, fn, arg = self.tyot[i]
                if i in virheet:
                    fut.set_exception(RuntimeError("tyoprosessi kaatui"))
                else:
                    fut.set_result(fn(arg))
    try:
        suora = mk.MerkkiSeuranta(mittarit, 30.0)
        for f in range(0, 1200):
            suora.syota(f, f)
        pooli = Pooli()
        ms = mk.MerkkiSeuranta(mittarit, 30.0, pool=pooli, tyo=tyo)
        ms.ODOTTAVIA_MAX = 10 ** 6                                                      # (tama testi pitaa kaikki skannaukset kesken)
        for f in range(0, 1200):
            ms.syota(f, f)
        tarkista("taustalla: skannaukset jonossa, sarjoihin ei viela mitaan (tyot kesken)", len(ms._odottavat) == 160 and not ms.sarjat["vasen"], f"{len(ms._odottavat)} jonossa")
        jarj = list(range(len(pooli.tyot)))
        np.random.default_rng(4).shuffle(jarj)
        sailiot = threading.Timer(0.3, lambda: pooli.valmista(jarj))                    # tulokset valmistuvat myohemmin ja sekajarjestyksessa
        sailiot.start()
        t0 = time.time()
        k = ms.konsensus("vasen", 1000)
        odotus = time.time() - t0
        k0 = suora.konsensus("vasen", 1000)
        tarkista("konsensus odottaa kesken olevat skannaukset ja antaa saman tuloksen kuin suora skannaus", odotus >= 0.2 and k is not None and k == k0, f"odotus {odotus:.2f} s")
        tarkista("oikea puolisko ja f_asti samoin", ms.konsensus("oikea", 1000) == suora.konsensus("oikea", 1000) and ms.konsensus("vasen", 1000, f_asti=700) == suora.konsensus("vasen", 1000, f_asti=700))
        tarkista("sarjat sisaltavat saman kuin suorassa skannauksessa (ruudut ja ehdokkaat, puskurin rajaus)", all(list(ms.sarjat[p]) == list(suora.sarjat[p]) for p in mittarit) or all(list(ms.sarjat[p]) == list(suora.sarjat[p])[-len(ms.sarjat[p]):] for p in mittarit))
        # epaonnistunut tyo: tulos tutkitaan paikan paalla (ei havioa)
        pooli2 = Pooli()
        ms2 = mk.MerkkiSeuranta(mittarit, 30.0, pool=pooli2, tyo=tyo)
        ms2.ODOTTAVIA_MAX = 10 ** 6
        for f in range(0, 1200):
            ms2.syota(f, f)
        pooli2.valmista(range(len(pooli2.tyot)), virheet=set(range(0, len(pooli2.tyot), 3)))
        tarkista("tyoprosessin virhe: leike tutkitaan paikan paalla, tulos sama", ms2.konsensus("vasen", 1000) == k0 and ms2.konsensus("oikea", 1000) == suora.konsensus("oikea", 1000))
        # ei pooleja (pool = None) ja ruutu ilman leiketta
        mk.leike = lambda m, ruutu: (None, 0, 0)
        pooli3 = Pooli()
        ms3 = mk.MerkkiSeuranta(mittarit, 30.0, pool=pooli3, tyo=tyo)
        for f in range(0, 90):
            ms3.syota(f, f)
        ms3.konsensus("vasen", 30)
        tarkista("ruutu ilman leiketta (kuvan ulkopuolella): ei tyota, tyhja sarja", not pooli3.tyot and all(e == [] for _, e in ms3.sarjat["vasen"]))
        # odottavien enimmaismaara: liian jaljessa oleva tyoprosessi ei kasvata muistia rajatta (syotto odottaa vanhimman, kun jono on taynna)
        mk.leike = lambda m, ruutu: (ruutu, 0, 0)
        pooli4 = Pooli()
        ms4 = mk.MerkkiSeuranta(mittarit, 30.0, pool=pooli4, tyo=tyo)
        ms4.ODOTTAVIA_MAX = 6
        seis = threading.Event()

        def valmista_jatkuvasti():                                                      # hidas tyoprosessi: yksi tulos 10 ms:n valein
            i = 0
            while not seis.is_set():
                if i < len(pooli4.tyot):
                    pooli4.valmista([i])
                    i += 1
                    time.sleep(0.01)
                else:
                    time.sleep(0.002)
        ajastin = threading.Thread(target=valmista_jatkuvasti, daemon=True)
        ajastin.start()
        suurin = 0
        for f in range(0, 15 * 12):
            ms4.syota(f, f)
            suurin = max(suurin, len(ms4._odottavat))
        seis.set()
        ajastin.join()
        tarkista("odottavien maara on rajattu (ODOTTAVIA_MAX), eika syotto jumiudu", suurin <= ms4.ODOTTAVIA_MAX and len(pooli4.tyot) == 24, f"suurin {suurin}, tyota {len(pooli4.tyot)}")
    finally:
        mk.ehdokkaat, mk.leike, mk.ehdokkaat_leikkeesta = vanha


def testaa_netti_fit_nopeutukset():
    print("netti_fit (C++): nopeutetut laskennat antavat saman tuloksen kuin yleinen reitti (kupera peite = cv::convexHull, reunakaista = dilate - erode)")
    if ns._netti_fit is None:
        print("  (netti_fit ei ole kaytossa: ohitetaan)")
        return
    rng = np.random.default_rng(3)
    juuri = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    n_kupera = n_kaista = n_nopea = 0
    erot = []
    for kansio in ("tulokset", "tulokset_nettivideo2", "tulokset_nettivideo3"):
        for puoli in ("vasen", "oikea"):
            if not os.path.exists(os.path.join(juuri, kansio, f"kalibrointi_{puoli}.json")):
                continue
            m = ns.Mittari(os.path.join(juuri, kansio), puoli)
            b = m._cpp
            if b is None or not hasattr(b, "tarkista_kupera"):
                print("  (vanha netti_fit ilman testikoukkuja: ohitetaan)")
                return
            for X, Y in zip(rng.uniform(-260, 260, 250), rng.uniform(-420, 780, 250)):
                n_kupera += 1
                if not b.tarkista_kupera(float(X), float(Y)):
                    erot.append(f"kupera {kansio} {puoli} ({X:.0f}, {Y:.0f})")
                for kaista in (5, 6, 8):
                    kaytossa, sama, _ = b.tarkista_kaista(float(X), float(Y), kaista)
                    n_kaista += 1
                    n_nopea += bool(kaytossa)
                    if not sama:
                        erot.append(f"kaista {kansio} {puoli} ({X:.0f}, {Y:.0f}) {kaista}")
    tarkista(f"kupera peite: pisteiden karsinta antaa samat kulmat kuin cv::convexHull ({n_kupera} paikkaa, kaikki tasot ja kulmatasot)", n_kupera > 0 and not [e for e in erot if e.startswith("kupera")], str(erot[:3]))
    tarkista(f"reunakaista: rivivalien laskenta = dilate - erode ({n_kaista} paikkaa, nopea reitti {n_nopea} kertaa)", n_kaista > 0 and not [e for e in erot if e.startswith("kaista")] and n_nopea > 0.9 * n_kaista, str(erot[:3]))


def testaa_esilataus_purkusaie():
    print("esilataus: pakattu puskuri puretaan omassa saikeessa (analyysisaie ei pura)")
    import threading
    import time
    import netti_live as nl

    def ruudut(n):
        for f in range(n):
            fr = np.full((64, 96, 3), 90, np.uint8)
            fr[10:30, 20:60] = (f % 200, 60, 200 - f % 200)
            yield f, fr

    e = nl.Esilataus(ruudut(60), 4, 3.0, 10.0)
    nimet = [t.name for t in threading.enumerate()]
    tarkista("pakatussa tilassa purkusaie kaynnissa", "esilataus-purku" in nimet and "esilataus" in nimet, str(nimet))
    time.sleep(0.3)
    tarkista("purettuja ruutuja valmiina etukateen (lyhyt jono tayttyy)", e._purettu.qsize() == e._purettu.maxsize, f"{e._purettu.qsize()} / {e._purettu.maxsize}")
    f, fr = next(e)
    tarkista("annettu ruutu on BGR (3 kanavaa) ja oikea", f == 0 and fr.shape == (64, 96, 3) and int(np.abs(fr[10:30, 20:60].astype(int) - np.array([0, 60, 200])).max()) <= 2)
    sain = [f] + [x for x, _ in e]
    tarkista("jarjestys ja maara sailyvat hitaallekin kuluttajalle", sain == list(range(60)), str(sain[:8]))
    ennen = set(threading.enumerate())
    e2 = nl.Esilataus(ruudut(40), 4, 3.0, 10.0)
    omat = [t for t in threading.enumerate() if t not in ennen]
    next(e2)
    e2.sulje()
    time.sleep(0.8)
    tarkista("sulje(): lukija- ja purkusaie loppuvat", len(omat) == 2 and not any(t.is_alive() for t in omat), str([(t.name, t.is_alive()) for t in omat]))
    e3 = nl.Esilataus(ruudut(5), 4, 0.0, 10.0)
    tarkista("raakatilassa (ei pitkaa puskuria) ei purkusaietta", e3._purettu is None)
    list(e3)


def testaa_b_prioriteetti():
    print("B-tyoprosessit: normaali prioriteetti toistettavassa ajossa; live-virrassa pudotetaan (ns.pudota_prioriteetti, Analyysi.matalaksi_b_tyoprosessit)")
    import subprocess
    import types
    if os.name == "nt":
        print("  (Windows: prioriteettiluokka asetetaan ctypesilla, ei testata tassa)")
        return
    lapsi = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        ennen = os.getpriority(os.PRIO_PROCESS, lapsi.pid)
        ns.pudota_prioriteetti(lapsi.pid)
        jalkeen = os.getpriority(os.PRIO_PROCESS, lapsi.pid)
        tarkista("pudota_prioriteetti: lapsiprosessin nice nousee (5)", jalkeen >= ennen + 5 or jalkeen >= 5, f"{ennen} -> {jalkeen}")
        lapsi2 = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            an = ns.Analyysi.__new__(ns.Analyysi)
            an.pool = types.SimpleNamespace(_processes={0: types.SimpleNamespace(pid=lapsi2.pid)})
            an.matalaksi_b_tyoprosessit()
            tarkista("Analyysi.matalaksi_b_tyoprosessit: poolin tyoprosessit alemmalle prioriteetille", os.getpriority(os.PRIO_PROCESS, lapsi2.pid) >= 5)
            an.pool = None
            an.matalaksi_b_tyoprosessit()
            tarkista("ilman poolia ei tee mitaan (ei poikkeusta)", True)
        finally:
            lapsi2.kill()
    finally:
        lapsi.kill()
    import inspect
    tarkista("alusta_tyo ei laske B-tyoprosessin prioriteettia (prioriteettikaannos: analyysi odottaa B-tulokset)", "matala_prioriteetti()" not in inspect.getsource(ns.alusta_tyo).split('"""')[-1])


def testaa_puskuri_ja_viivevaroitus():
    print("viivepuskurin muodot (yuv / jpeg / bgr), ylivuotosuoja ja viivevaroituksen syy (analyysi vai piirto)")
    import types
    import netti_live as nl
    kuva = np.full((64, 96, 3), 100, np.uint8)
    kuva[10:30, 20:60] = (30, 140, 200)
    laji, data = nl.pakkaa_ruutu(kuva, "bgr")
    tarkista("bgr: pakkaus ei kopioi eika muunna (ruutu sellaisenaan)", laji == "bgr" and data is kuva)
    purettu = nl.pura_ruutu(laji, data)
    alkuperainen = kuva.copy()
    purettu[:] = 0                                                                # (piirto muokkaa purettua ruutua paikallaan)
    tarkista("bgr: purettu ruutu on kopio (piirto ei muuta ruutua, jonka analyysi voi viela lukea)", purettu is not kuva and not np.shares_memory(purettu, kuva) and np.array_equal(kuva, alkuperainen))
    tarkista("bgr: purku antaa saman sisallon", np.array_equal(nl.pura_ruutu(laji, data), kuva))
    for muoto in ("yuv", "jpeg"):
        laji, data = nl.pakkaa_ruutu(kuva, muoto)
        takaisin = nl.pura_ruutu(laji, data)
        sisa = (slice(14, 26), slice(28, 52))                                     # (terava vareja reuna sumenee JPEG:n ja YUV 4:2:0:n varikanavissa: verrataan tasaisia alueita)
        ero = max(int(np.abs(takaisin[sisa].astype(int) - kuva[sisa]).max()), int(np.abs(takaisin[40:60, 70:90].astype(int) - kuva[40:60, 70:90]).max()))
        tarkista(f"{muoto}: pakkaus ja purku antavat lahes saman ruudun (tasaiset alueet)", laji == muoto and takaisin.shape == kuva.shape and ero <= (3 if muoto == "yuv" else 14), f"max ero {ero}")
    tarkista("jpeg: laatu-argumentti vaikuttaa kokoon", len(nl.pakkaa_ruutu(kuva, "jpeg", 20)[1]) < len(nl.pakkaa_ruutu(kuva, "jpeg", 98)[1]))
    tarkista("muodon valinta: yuv oletuksena, jpeg pyydettaessa", nl.puskurin_muoto("yuv", 10, 360, 420) == "yuv" and nl.puskurin_muoto("jpeg", 10, 360, 420) == "jpeg")
    tarkista("--puskuri auto: bgr kun muistia on vahintaan 14 Gt (nimellisesti 16 Gt:n kone ilmoittaa 14,9-15,9), muuten yuv (myos kun muisti ei ole tiedossa); nimetty muoto sellaisenaan",
             nl.valitse_puskuri("auto", 32.0) == "bgr" and nl.valitse_puskuri("auto", 15.6) == "bgr" and nl.valitse_puskuri("auto", 14.0) == "bgr" and nl.valitse_puskuri("auto", 11.9) == "yuv"
             and nl.valitse_puskuri("auto", 7.8) == "yuv" and nl.valitse_puskuri("auto", None) == "yuv"
             and nl.valitse_puskuri("yuv", 64.0) == "yuv" and nl.valitse_puskuri("jpeg", 64.0) == "jpeg" and nl.valitse_puskuri("bgr", 4.0) == "bgr")
    gt = nl.fyysinen_muisti_gt()
    tarkista("fyysinen muisti: luku (Gt) tai None", gt is None or 0.5 < gt < 100000, str(gt))
    tarkista("bgr: raskas muoto vain kun jono on lyhyt, sen jalkeen yuv, sen jalkeen jpeg (muistin suoja)",
             nl.puskurin_muoto("bgr", 100, 360, 420) == "bgr" and nl.puskurin_muoto("bgr", 361, 360, 420) == "yuv" and nl.puskurin_muoto("bgr", 421, 360, 420) == "jpeg" and nl.puskurin_muoto("yuv", 421, 360, 420) == "jpeg")

    class Sovitin:
        def __init__(self, nimi):
            self.sovitin = nimi
    an = types.SimpleNamespace(M={"vasen": Sovitin("C++ (netti_fit)")}, saikeet=3)
    t = nl.viive_varoitus(5.0, dict(analyysia_odotettu_s=4.0, syota_ms=[44.0] * 20), an, 30.0)
    tarkista("analyysia odotettu: varoitus nimeaa analyysin ja ruudun ajan", "analyysi ei ehdi" in t and "44 ms" in t and "alle 33 ms" in t and "Pythonilla" not in t, t[:120])
    an.M["vasen"] = Sovitin("Python (kivi_tarkka)")
    t = nl.viive_varoitus(5.0, dict(analyysia_odotettu_s=4.0, syota_ms=[44.0] * 20), an, 30.0)
    tarkista("Python-sovitus: varoitus neuvoo kaantamaan C++-moduulin", "Pythonilla" in t and "netti_fit" in t)
    t = nl.viive_varoitus(5.0, dict(analyysia_odotettu_s=0.2, syota_ms=[5.0] * 20), an, 30.0)
    tarkista("piirto / tallennus jaljessa: vanha varoitus", "piirto / tallennus" in t and "analyysi ei ehdi" not in t)


def testaa_piirto_nopeutukset():
    print("piirto: peittavan laatikon cv2.copyTo = np.copyto, kuvaan rajatut viivasegmentit valimuistissa (sama kuva kuin ennen, koon vaihtuessa lasketaan uudelleen)")
    import types
    import cv2
    import netti_piirto as npi
    rng = np.random.default_rng(5)
    pohja = rng.integers(0, 256, (400, 600, 3), dtype=np.uint8)
    kuva = rng.integers(0, 256, (40, 100, 3), dtype=np.uint8)
    maski = rng.random((40, 100)) > 0.7
    a = pohja.copy()
    npi.Piirtaja._laatikko(a, kuva, maski, (30, 20))
    b = pohja.copy()
    roi = b[20:60, 30:130]
    cv2.addWeighted(kuva, 0.72, roi, 0.28, 0, roi)
    np.copyto(roi, kuva, where=maski[:, :, None])
    tarkista("laatikko: sama kuva kuin np.copyto-toteutuksella", np.array_equal(a, b))
    c = pohja.copy()
    npi.Piirtaja._laatikko(c, kuva, maski, (560, 20))
    tarkista("laatikko kuvan ulkopuolella: kuva ei muutu", np.array_equal(c, pohja))
    pts = np.array([[-50.0, 10.0], [20.0, 30.0], [700.0, 40.0], [300.0, 500.0], [310.0, 520.0]])
    segs = npi._leikatut_segmentit(pts, 400, 600)
    tarkista("segmentit rajataan kuvaan (kaikki paat kuvassa), kokonaan ulkona oleva pois", len(segs) == 3 and all((0 <= q[:, 0]).all() and (q[:, 0] <= 599).all() and (0 <= q[:, 1]).all() and (q[:, 1] <= 399).all() for q in segs), str(len(segs)))
    tarkista("katkoviiva: joka toinen segmentti", len(npi._leikatut_segmentit(pts, 400, 600, katkoviiva=True)) == 2)
    x = pohja.copy()
    npi._viiva_kuvaan(x, pts, npi.VARI_LIUKU, 3)
    y = pohja.copy()
    npi._piirra_segmentit(y, segs, npi.VARI_LIUKU, 3)
    tarkista("_viiva_kuvaan = rajaus + piirto", np.array_equal(x, y) and not np.array_equal(x, pohja))

    p = npi.Piirtaja({"vasen": object(), "oikea": object()}, fps=30.0)
    laskuri = []
    data = dict(pts=np.array([[1100.0, 300.0], [1120.0, 700.0]]), pts2=np.array([[1160.0, 300.0], [1180.0, 700.0]]), XT_liuku=10.0, XT_irr=-5.0, alustava=True, uv_liuku=(1140.0, 420.0), uv_irr=(1300.0, 450.0))
    p._viivat_data = lambda h, kauko: data
    vanha = npi._leikatut_segmentit
    npi._leikatut_segmentit = lambda *a, **k: laskuri.append(a[1:3]) or vanha(*a, **k)
    try:
        h = types.SimpleNamespace(viivat=dict(x=1), puoli="vasen")
        kuvat = []
        for _ in range(3):
            img = np.zeros((1080, 1920, 3), np.uint8)
            p._viivat(img, h)
            kuvat.append(img)
        tarkista("rajatut segmentit lasketaan kerran (kaksi viivaa), ei joka ruudulle", len(laskuri) == 2, str(laskuri))
        tarkista("valimuistista piirretty kuva on sama joka ruudulla", np.array_equal(kuvat[0], kuvat[1]) and np.array_equal(kuvat[1], kuvat[2]) and int(kuvat[0].max()) > 0)
        img = np.zeros((900, 1600, 3), np.uint8)
        p._viivat(img, h)
        tarkista("kuvan koon vaihtuessa segmentit lasketaan uudelleen", len(laskuri) == 4 and laskuri[-1] == (900, 1600))
        data["alustava"] = False
        data["segmentit_koko"] = None
        img = np.zeros((1080, 1920, 3), np.uint8)
        p._viivat(img, h)
        tarkista("katkoviiva (alustava) ja yhtenainen viiva eroavat", int((img[500, 1150:1210] != 0).any(axis=1).sum()) >= int((kuvat[0][500, 1150:1210] != 0).any(axis=1).sum()))
    finally:
        npi._leikatut_segmentit = vanha


def testaa_mittaa_nopeus_lahde():
    print("tools/mittaa_nopeus.py: virran alkuun siirtyminen (haku tai ruutujen ohitus) ja lahteen valinta")
    import contextlib
    import io
    import tempfile
    import cv2
    import mittaa_nopeus as mn
    fps, n = 10.0, 100
    polku = os.path.join(tempfile.mkdtemp(), "koodattu.avi")
    kirj = cv2.VideoWriter(polku, cv2.VideoWriter_fourcc(*"MJPG"), fps, (64, 48))
    for i in range(n):
        kirj.write(np.full((48, 64, 3), 20 + 2 * i, np.uint8))                  # ruudun jarjestysnumero harmaasavynä
    kirj.release()
    cap = cv2.VideoCapture(polku)
    viestit = []
    kohta = mn.siirry(cap, 5.0, fps, tulosta=viestit.append)
    ok, kuva = cap.read()
    numero = (float(kuva.mean()) - 20.0) / 2.0 if ok else None
    tarkista("siirry: virta, jossa haku toimii -> luku jatkuu tavoitekohdasta, ei ohiteta lukemalla", abs(kohta - 50) <= fps and numero is not None and abs(numero - 50) <= 3 and not viestit, f"kohta {kohta}, ruutu {numero}")
    cap.release()

    class EiHakua:
        """Virta, joka ei salli hakua: set ei tee mitaan, grab lukee eteenpain."""
        def __init__(self, ruutuja):
            self.kohta, self.ruutuja = 0, ruutuja

        def set(self, *_):
            return False

        def get(self, *_):
            return float(self.kohta)

        def grab(self):
            if self.kohta >= self.ruutuja:
                return False
            self.kohta += 1
            return True

    class Ylihyppy(EiHakua):
        def set(self, tunnus, arvo):
            self.kohta = int(arvo) + 40
            return True

    c, viestit = EiHakua(100), []
    kohta = mn.siirry(c, 5.0, fps, tulosta=viestit.append)
    tarkista("siirry: virta ilman hakua -> ruudut luetaan ohi (50 ruutua) ja siita kerrotaan", kohta == 50 and c.kohta == 50 and any("ohitetaan" in v for v in viestit), f"kohta {kohta}")
    c = EiHakua(30)
    tarkista("siirry: virta loppuu ennen tavoitetta -> palautetaan ohitettujen maara", mn.siirry(c, 5.0, fps, tulosta=lambda *_: None) == 30 and c.kohta == 30 and mn.ohita_ruudut(EiHakua(7), 20) == 7)
    c, viestit = Ylihyppy(200), []
    kohta = mn.siirry(c, 5.0, fps, tulosta=viestit.append)
    tarkista("siirry: haku meni yli -> ei lueta taaksepain, ilmoitetaan", kohta == 90 and c.kohta == 90 and any("yli" in v for v in viestit), f"kohta {kohta}")
    c = EiHakua(10)
    tarkista("siirry: alku 0 -> ei hakua eika ohitusta", mn.siirry(c, 0.0, fps) == 0 and c.kohta == 0)

    vanha = sys.argv
    try:
        for argv in ([], ["a.mp4", "--url", "http://x"], ["--youtube", "abc", "--url", "http://x"], ["a.mp4", "--alku", "10", "--loppu", "5"]):
            sys.argv = ["mittaa_nopeus.py"] + argv
            try:
                with contextlib.redirect_stderr(io.StringIO()):
                    mn.main()
                koodi = None
            except SystemExit as e:
                koodi = e.code
            tarkista(f"lahteen ja jakson tarkistus: {' '.join(argv) or '(ei lahdetta)'} -> kayttovirhe", koodi == 2, f"poistumiskoodi {koodi}")
    finally:
        sys.argv = vanha


def main():
    for t in (testaa_vakiot, testaa_yhdista, testaa_pysahtymishetki, testaa_kello, testaa_irroitus, testaa_irroitus_muutos, testaa_vakaa, testaa_aikaperusta, testaa_kalibroinnin_tarkistus, testaa_hog_alue, testaa_harva_rata, testaa_kalibroinnin_valinta, testaa_tarkista_kivet, testaa_kahva_ala_mittakaava, testaa_tausta_maskit, testaa_vaihto_todennettu, testaa_kandidaatti_kivi, testaa_graniittihaku, testaa_saapuminen_graniitti, testaa_viritys_rungosta, testaa_ulottuma_vahvistettu, testaa_lepo_haku, testaa_esilataus_viive, testaa_esilataus_purkusaie, testaa_puskuri_ja_viivevaroitus, testaa_mittaa_nopeus_lahde, testaa_kopioruutu, testaa_kahva_maski, testaa_jaettu_leike, testaa_b_prioriteetti, testaa_netti_fit_nopeutukset, testaa_merkki, testaa_merkki_konsensus, testaa_merkki_taustaskannaus, testaa_paneeli_merkista, testaa_viivat_ilman_tekstia, testaa_piirto_liuku, testaa_piirto_nopeutukset):
        t()
    print()
    if VIRHEITA:
        print(f"{len(VIRHEITA)} testia epaonnistui: {', '.join(VIRHEITA)}")
        return 1
    print("kaikki testit OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
