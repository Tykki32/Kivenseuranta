"""Kahvan kulman neuroverkon (kulmaverkko.py) yksikkotestit: kuvan kierto / peilaus -konventio, NumPy-verkon tulos sama kuin opetuksessa kaytetyn JAX-verkon (kultatiedostot), vakiokiertosovitus synteettisella
radalla (ei videota, ei JAX:ia).

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


def _jakaumat(kulmat, kohina, rng, kappa=20.0):
    havaittu = (kulmat + rng.normal(0, kohina, len(kulmat))) % 180.0
    p = np.exp(np.cos(np.deg2rad(2.0 * (kv.KUL[None, :] - havaittu[:, None]))) * kappa)
    return p / p.sum(axis=1, keepdims=True)


def testaa_vakiokierto():
    """Vakiokiertosovitus (kulmat suoralla kierteen vaimeneminen mukana): kulmat, omega ja suunta palautuvat; peilattu rata vaihtaa suunnan; yksittainen 90 asteen virheruutu ei vie sovitusta, mutta nakyy residuaalissa."""
    rng = np.random.default_rng(1)
    t = 3000.0 + np.arange(40) * 0.04
    G = (1.0 - np.exp(-kv.K_KIERRE * (t - t[0]))) / kv.K_KIERRE
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
    mukana kaikki nämä viivat, joten tämä on toimivuustarkistus (paino ei ole rikki, kulmakonventio oikein) eikä riippumaton testi; riippumaton arvio on README:n protokolla (`kulmaverkko_opeta.py arvioi --pois ...`)."""
    import glob
    polut = sorted(glob.glob(os.path.join(KANSIO, "kulmaverkko_paino_*.npz")))
    d = np.load(os.path.join(KANSIO, "kierre_validointi", "kulmaverkko_ihmisen_ruudut.npz"))
    tarkista("toimitettuja painoja löytyy", len(polut) > 0, ", ".join(os.path.basename(p) for p in polut))
    mallit = [kv.lataa(p) for p in polut]
    for polku, malli in list(zip(polut, mallit)) + ([("kaikki yhdessä", mallit)] if len(mallit) > 1 else []):
        p = kv.ennusta(malli, d["tiles"])
        e = np.abs(kv.ero180(kv.kulma_jakaumasta(p)[0], d["kulma"]))
        tarkista(f"painot {os.path.basename(polku)}: ihmisen viivat keskimäärin alle 6 astetta, ei yli 20", p.shape == (len(d["tiles"]), kv.NB) and np.allclose(p.sum(axis=1), 1.0, atol=1e-4) and e.mean() < 6.0 and e.max() < 20.0,
                 f"keskivirhe {e.mean():.1f}, mediaani {np.median(e):.1f}, suurin {e.max():.1f} astetta ({len(e)} viivaa)")


def main():
    for t in (testaa_opetuskoodi_jax, testaa_kierto_ja_peilaus, testaa_verkko_kultatiedostot, testaa_tallenna_lataa_ja_yhdistelma, testaa_vakiokierto, testaa_toimitetut_painot):
        t()
    print()
    if VIRHEITA:
        print(f"{len(VIRHEITA)} testia epaonnistui: {', '.join(VIRHEITA)}")
        return 1
    print("kaikki testit OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
