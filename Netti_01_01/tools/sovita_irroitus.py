"""Irroituksen tunnistuksen uudelleenajo tallennetuista radoista ilman koko videon analyysia, ja tulosten uskottavuuden tarkistus.

    python netti_live.py --lahde netti.mp4 --nopea --ei-ikkunaa --ei-kierretta --csv heitot.csv --radat radat.pkl      # kerran (--ei-kierretta riittaa: heittopaan radat); radat.pkl sisaltaa laske_viivat:n tarkan syotteen
    python tools/sovita_irroitus.py radat.pkl                           # irroitus nykyisella koodilla kaikille heitoille + vertailu tallennettuun estimaattiin + uskottavuusmittarit
    python tools/sovita_irroitus.py radat.pkl --menetelma hanta         # vanha vapaan lennon hanta vertailua varten (--menetelma kinkki: kahden paraabelin C1-sovitus)
    python tools/sovita_irroitus.py radat.pkl --csv irroitus.csv        # heittokohtainen taulukko

Mittarit (kaikki suoraan heittopaan radasta, ilman kaukopaan seurantaa, ellei toisin mainita):
  * v_jalk - v_hog1: irroituksen jalkeinen nopeus miinus nopeus hog-viivalla (suora mitaus viivan ymparilta). Vapaa kappale vain hidastuu: ero on a_vapaa x aika irroituksesta hog-viivalle
    (a_vapaa ~0,07 m/s2: myohainen irroitus 1 m ennen viivaa -> ~0,03 m/s, aikainen 3 m -> ~0,1 m/s); 'jaannos' = ero miinus tama ennuste (mediaani ~0, hajonta ~0,02 m/s). Negatiivinen ero tai suuri
    jaannos tarkoittaa, etta 'vapaa lento' sisalsi viela tyontoa (irroitus tunnistettu liian aikaisin ja nopeus aliarvioitu) tai irroitus on liian myohainen.
  * suunta - suunta_hog1: irroituksen suunta miinus kiven suunta hog-viivalla (kaarto muuttaa suuntaa matkalla vain ~0,1-0,4 astetta).
  * hid_vapaa: vapaan lennon hidastuvuus (m/s2; < 0 = kiihtyy).
  * kaukopaan seurantaan perustuva (--kauko): irroituksen nopeudesta ja kiven vakioista lasketun pysahtymismatkan ero havaittuun (heittajan kiven pysahtyessa ilman tormaysta):
    A_matka = kitkataso, jolla malli osuu havaittuun pysahtymiseen; jos irroituksen nopeus on oikea, A_matka on lahes vakio (harjaus muuttaa matkaa enintaan ~2 m = ~7 %)."""
import argparse
import csv
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import liuku_irroitus as li  # noqa: E402
import netti_seuranta as ns  # noqa: E402
import netti_vakiot as nv  # noqa: E402


class _H:
    """Heitto-korvike: viivat, kauko, kierre (netti_seuranta.Analyysi._irr_tarkistus_argumentit)."""
    def __init__(self, viivat, d):
        self.viivat, self.kauko, self.kierre, self.kierto, self.kauko_kierto = viivat, d.get("kauko"), d.get("kierre"), d.get("kierto"), d.get("kauko_kierto")


def kauko_tarkistus(v, d):
    """netti_vakiot.tarkista_irroitus viivaestimaatille v ja heiton tallennetulle kaukopaan datalle d (None, jos kivi ei pysahtynyt / tormasi / ei kaukopaan seurantaa)."""
    h = _H(v, d)
    irr = ns.Analyysi._irr_tarkistus_argumentit(ns.Analyysi.__new__(ns.Analyysi), h)
    if irr is None:
        return None
    va = d.get("vakiot")
    return nv.tarkista_irroitus(**irr, kappa=(va["kappa"] if va and va.get("kappa_ok") else None))


def mittarit(v):
    """Uskottavuusmittarit viivaestimaatista v (laske_viivat:n dict): dict."""
    out = dict(irr_Y=v["irr_Y"], v_jalk=v["nopeus_jalkeen"], suunta=v["irr_suunta_deg"], hid_vapaa=v["vapaa_hidastuvuus"], merkitseva=bool(v["merkitseva"]), vapaa_matka=v["vapaa_matka_cm"])
    vh, sh = v.get("nopeus_hog1"), v.get("suunta_hog1_deg")
    out["v_hog1"], out["suunta_hog1"] = vh, sh
    out["dv"] = None if vh is None else v["nopeus_jalkeen"] - vh
    out["dsuunta"] = None if sh is None else v["irr_suunta_deg"] - sh
    yk = v.get("hog1_Y_keski")
    out["hog_matka_m"] = None if yk is None else max(float(yk) - float(v["irr_Y"]), 0.0) / 100.0       # irroituksesta hog-ikkunan keskelle
    return out


def _f(x, n=2, leveys=6):
    return " " * (leveys - 1) + "-" if x is None or not np.isfinite(x) else f"{x:{leveys}.{n}f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("radat", help="netti_live.py --radat -tiedosto (pickle)")
    ap.add_argument("--menetelma", choices=("muutos", "hanta", "kinkki"), default=None, help="irroituksen tunnistus (oletus: liuku_irroitus.MENETELMA)")
    ap.add_argument("--ei-esitietoa", action="store_true", help="ilman vapaan lennon fysiikan esitietoa (hidastuvuus, kaarevuus)")
    ap.add_argument("--sigma-a", type=float, default=None, help="vapaan lennon hidastuvuuden esitiedon hajonta (m/s2)")
    ap.add_argument("--sigma-c", type=float, default=None, help="vapaan lennon kaarevuuden esitiedon hajonta (1/cm)")
    ap.add_argument("--csv", help="heittokohtainen taulukko CSV:hen")
    ap.add_argument("--kauko", action="store_true", help="laske myos kaukopaan seurantaan perustuva pysahtymismatkan tarkistus (hidas: kiven rata_malli-simulointeja)")
    ap.add_argument("--fps-lahde", type=float, default=25.0)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--heitot", help="vain nama heitot (pilkuilla eroteltu, 1-alkuinen)")
    a = ap.parse_args()
    if a.menetelma:
        li.MENETELMA = a.menetelma
    if a.sigma_a:
        li.SIGMA_A_VAPAA_M_S2 = a.sigma_a
        li.ESI_Y = ((0.0, 0.0, 0.0, 1.0), -100.0 * li.A_VAPAA_M_S2, 100.0 * a.sigma_a)
    if a.sigma_c:
        li.SIGMA_KAAREVUUS_1_CM = a.sigma_c
        li.ESI_X = ((0.0, 0.0, 0.0, 1.0), 0.0, a.sigma_c)
    if a.ei_esitietoa:
        ns.ESTIMAATTI["esitieto"] = False
    heitot = pickle.load(open(a.radat, "rb"))
    valitut = {int(x) for x in a.heitot.split(",")} if a.heitot else None
    rivit, ero = [], []
    print(f"menetelma {li.MENETELMA}; esitieto {'ei' if a.ei_esitietoa else f'kylla (hidastuvuus {li.A_VAPAA_M_S2} +- {li.SIGMA_A_VAPAA_M_S2} m/s2, kaarevuus 0 +- {li.SIGMA_KAAREVUUS_1_CM:.0e} 1/cm)'}")
    print("   i vari  n  pu | uusi: irrY   v_jalk v_hog1  dv    suunta suunta_h dsuunta hid_vap merk | tallennettu: irrY  v_jalk  suunta  hid_vap")
    for i, h in enumerate(heitot, 1):
        if valitut and i not in valitut:
            continue
        syote = h.get("viivat_syote")
        if not syote:
            continue
        f, X, Y = (np.array(c, float) for c in zip(*syote))
        v = ns.laske_viivat(f, X, Y, a.fps_lahde, a.fps)
        if v is None or "virhe" in v:
            print(f"{i:4d} {h['vari'][:3]} {h['kivi_nro']!s:>2} {h['puoli'][:2]} | ei estimaattia ({v})")
            continue
        m = mittarit(v)
        vo = h.get("viivat")
        mo = mittarit(vo) if vo else None
        print(f"{i:4d} {h['vari'][:3]} {h['kivi_nro']!s:>2} {h['puoli'][:2]} | {m['irr_Y']:9.0f} {_f(m['v_jalk'])} {_f(m['v_hog1'])} {_f(m['dv'], 2, 5)} {_f(m['suunta'])} {_f(m['suunta_hog1'], 2, 7)} {_f(m['dsuunta'], 2, 7)} {_f(m['hid_vapaa'], 3, 7)}  {int(m['merkitseva'])}   | "
              + (f"{mo['irr_Y']:9.0f} {_f(mo['v_jalk'])} {_f(mo['suunta'])} {_f(mo['hid_vapaa'], 3, 7)}" if mo else "-"))
        rivit.append(dict(heitto=i, vari=h["vari"], kivi_nro=h["kivi_nro"], puoli=h["puoli"], **m, **({f"vanha_{k}": mo[k] for k in ("irr_Y", "v_jalk", "suunta", "hid_vapaa")} if mo else {})))
        if a.kauko:
            tk = kauko_tarkistus(v, h)
            tko = kauko_tarkistus(vo, h) if vo else None
            rivit[-1].update(matka_ero_m=None if tk is None else tk["matka_ero_m"], A_matka=None if tk is None else tk["A_matka"], sivu_ero_cm=None if tk is None else tk["sivu_ero_cm"],
                             vanha_matka_ero_m=None if tko is None else tko["matka_ero_m"], vanha_A_matka=None if tko is None else tko["A_matka"])
        if mo:
            ero.append((m["irr_Y"] - mo["irr_Y"], m["v_jalk"] - mo["v_jalk"], m["suunta"] - mo["suunta"]))
    if not rivit:
        print("ei heittoja, joilla on viivat_syote (aja netti_live.py uudella koodilla --radat)")
        return 1
    dv = np.array([r["dv"] for r in rivit if r["dv"] is not None])
    ds = np.array([r["dsuunta"] for r in rivit if r["dsuunta"] is not None])
    hv = np.array([r["hid_vapaa"] for r in rivit])
    print(f"\n{len(rivit)} heittoa")
    print(f"v_jalk - v_hog1: mediaani {np.median(dv):+.3f} m/s, 1,48 MAD {1.4826 * np.median(np.abs(dv - np.median(dv))):.3f}; < -0,05 m/s (irroitus liian aikaisin): {int((dv < -0.05).sum())} / {len(dv)}")
    jn = np.array([r["dv"] - float(np.median(hv)) * r["hog_matka_m"] / r["v_jalk"] for r in rivit if r["dv"] is not None and r.get("hog_matka_m") is not None and r["v_jalk"] > 0.3])
    if len(jn):
        print(f"  jaannos (v_jalk - v_hog1 miinus vapaan lennon ennuste {float(np.median(hv)):.3f} m/s2 x aika irroituksesta hog-viivalle): mediaani {np.median(jn):+.3f} m/s, 1,48 MAD {1.4826 * np.median(np.abs(jn - np.median(jn))):.3f}; |jaannos| > 0,05 m/s: {int((np.abs(jn) > 0.05).sum())} / {len(jn)}")
    print(f"suunta - suunta_hog1: mediaani {np.median(ds):+.2f} astetta, 1,48 MAD {1.4826 * np.median(np.abs(ds - np.median(ds))):.2f}; |ero| > 0,5 astetta: {int((np.abs(ds) > 0.5).sum())} / {len(ds)}; > 1 aste: {int((np.abs(ds) > 1.0).sum())}")
    print(f"vapaan lennon hidastuvuus < 0 (kiihtyy): {int((hv < 0).sum())} / {len(hv)}; mediaani {np.median(hv):+.3f} m/s2")
    if ero:
        e = np.array(ero)
        print(f"muutos tallennettuun: irr_Y mediaani {np.median(e[:, 0]):+.0f} cm (|muutos| > 50 cm: {int((np.abs(e[:, 0]) > 50).sum())}); v_jalk mediaani {np.median(e[:, 1]):+.3f} m/s (|muutos| > 0,05: {int((np.abs(e[:, 1]) > 0.05).sum())});"
              f" suunta mediaani {np.median(e[:, 2]):+.2f} astetta (|muutos| > 0,3: {int((np.abs(e[:, 2]) > 0.3).sum())})")
    if a.kauko:
        for nimi, avain in (("uusi", "matka_ero_m"), ("vanha (tallennettu)", "vanha_matka_ero_m")):
            e = np.array([r[avain] for r in rivit if r.get(avain) is not None])
            Ak = np.array([r["A_matka" if avain == "matka_ero_m" else "vanha_A_matka"] for r in rivit if r.get(avain) is not None])
            if len(e):
                print(f"kaukopaan pysahtymismatka vs rata_malli ({nimi}): {len(e)} heittoa; ero (havaittu - ennuste) mediaani {np.median(e):+.2f} m, 1,48 MAD {1.4826 * np.median(np.abs(e - np.median(e))):.2f} m, "
                      f"|ero| > {nv.MATKA_RAJA_M:.0f} m: {int((np.abs(e) > nv.MATKA_RAJA_M).sum())} / {len(e)}; kitkataso A_matka mediaani {np.median(Ak):.4f}, hajonta {100 * 1.4826 * np.median(np.abs(Ak - np.median(Ak))) / np.median(Ak):.1f} %")
    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rivit[0].keys()))
            w.writeheader()
            w.writerows(rivit)
        print("kirjoitettu", a.csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
