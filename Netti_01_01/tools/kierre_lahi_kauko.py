"""Heittopään ja kaukopään kierteen mittausten yhdenmukaisuus (riippumaton tarkistus: ei ihmisen merkintöjä eikä videota, vain ajon radat.pkl):

    python tools/kierre_lahi_kauko.py r_Matarmaa/radat.pkl r_Video1/radat.pkl ...

Kaukopää mittaa samaa kiveä 10-25 s myöhemmin toisella kameralla, ja kierto vaimenee tasaisesti (omega_lahi = omega_kauko exp(K_KIERRE dt)). Siksi
  * SUUNTA: heittopään varman (suunnan posteriori >= SUUNTA_VARMA) mittauksen etumerkin pitää olla sama kuin kaukopään; kaukopää luokitellaan varmaksi, "suunta varma" (p >= 0,99, mutta suuruus
    epävarma) ja epävarmaksi (p < 0,99: ei todiste, vain arpa), ja vastaavuus lasketaan luokittain;
  * SUURUUS: k = ln(|omega_lahi| / |omega_kauko|) / dt on lähellä K_KIERRE:tä (mediaani ja robusti hajonta), ja |ln r| = |ln(|omega_lahi| / (|omega_kauko| exp(K_KIERRE dt)))| kertoo suuruuden eron.
Tulostaa heittopään varmojen heittojen määrän ja nämä tilastot; ajamalla sen ennen ja jälkeen muutoksen (esim. netti_kierre.HANTA_PAINO_LAHI) näkee, kuinka mittauksen varmuus ja yhdenmukaisuus muuttuivat.
Kaukopään mittaus ei ole totuus (varma kaukopää voi olla väärässä, ja sen ikkuna on usein lyhyt), mutta se on riippumaton lähipään kuvista: ristiriita on merkki virheestä jommassakummassa."""
import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import netti_kierre as nk  # noqa: E402

KAUKO_SUUNTA_VARMA = 0.99      # kaukopään "suunta varma" -luokka (p_suunta), vaikka suuruus ei olisi varma


def mitta_ok(k):
    return k is not None and "virhe" not in k and "omega_deg_s" in k


def rivit_radoista(radat, nimi=""):
    """Heitot, joilla on heittopään mittaus: dict(nimi, heitto, vari, lahi, kauko) (kauko = None, ellei kaukopään mittausta ole). radat = radat.pkl:n heitto-lista."""
    out = []
    for i, h in enumerate(radat):
        lahi, kauko = h.get("kierto"), h.get("kauko_kierto")
        if mitta_ok(lahi):
            out.append(dict(nimi=nimi, heitto=i + 1, vari=h.get("vari"), lahi=lahi, kauko=kauko if mitta_ok(kauko) else None))
    return out


def kauko_luokka(k):
    """Kaukopään mittauksen luokka suuntatodisteena: 'varma' | 'suunta' (p >= KAUKO_SUUNTA_VARMA, suuruus epävarma) | 'epavarma' (p < KAUKO_SUUNTA_VARMA)."""
    if k.get("varma"):
        return "varma"
    return "suunta" if k.get("p_suunta", 0.0) >= KAUKO_SUUNTA_VARMA else "epavarma"


def tilastot(rivit):
    """Yhteenveto heittopään mittauksista rivit (rivit_radoista): heittoja, heittopään varmoja, suunnan vastaavuus kaukopään luokittain (varmoista lähipään mittauksista), k ja |ln r| varmoille pareille."""
    varmat = [r for r in rivit if r["lahi"].get("p_suunta", 0.0) >= nk.SUUNTA_VARMA]
    y = dict(heittoja=len(rivit), lahi_varma=len(varmat), kauko_mitattu=sum(1 for r in rivit if r["kauko"] is not None),
             suunta={k: [0, 0] for k in ("varma", "suunta", "epavarma")}, ristiriidat=[], k=[], lnr=[])
    for r in varmat:
        if r["kauko"] is None:
            continue
        sama = bool(np.sign(r["lahi"]["omega_deg_s"]) == np.sign(r["kauko"]["omega_deg_s"]))
        luokka = kauko_luokka(r["kauko"])
        y["suunta"][luokka][0] += sama
        y["suunta"][luokka][1] += 1
        if not sama and luokka != "epavarma":
            y["ristiriidat"].append((r["nimi"], r["heitto"], float(r["lahi"]["omega_deg_s"]), float(r["kauko"]["omega_deg_s"])))
        if sama and luokka == "varma":
            dt = float(r["kauko"]["t_keski"] - r["lahi"]["t_keski"])
            if dt > 0.0:
                a, b = abs(float(r["lahi"]["omega_deg_s"])), abs(float(r["kauko"]["omega_deg_s"]))
                y["k"].append(float(np.log(a / b) / dt))
                y["lnr"].append(float(np.log(a / (b * np.exp(nk.K_KIERRE * dt)))))
    return y


def tulosta(y, ulos=print):
    ulos(f"Heittoja heittopään mittauksella {y['heittoja']}, joista suunta varma (p >= {nk.SUUNTA_VARMA}) {y['lahi_varma']}; kaukopään mittaus {y['kauko_mitattu']} heitolla.")
    ulos("Heittopään varman suunnan etumerkki vs kaukopää (oikein / n):")
    for luokka, nimi in (("varma", "kaukopää varma"), ("suunta", f"kaukopää suunta p >= {KAUKO_SUUNTA_VARMA}, suuruus epävarma"), ("epavarma", f"kaukopää epävarma (p < {KAUKO_SUUNTA_VARMA}; ei todiste)")):
        o, n = y["suunta"][luokka]
        ulos(f"  {nimi:58s} {o:4d} / {n:<4d}")
    for nimi, heitto, a, b in y["ristiriidat"]:
        ulos(f"  RISTIRIITA {nimi} heitto {heitto}: heittopää {a:+.1f}, kaukopää {b:+.1f} aste/s")
    if y["k"]:
        k, lr = np.array(y["k"]), np.array(y["lnr"])
        ulos(f"Suuruus (varmat parit, sama etumerkki, n = {len(k)}): k mediaani {np.median(k):.4f} (K_KIERRE {nk.K_KIERRE}), robusti hajonta {1.4826 * np.median(np.abs(k - np.median(k))):.4f}; "
             f"|ln r| mediaani {np.median(np.abs(lr)):.3f}, keski {np.mean(np.abs(lr)):.3f}, yli 0,5: {int((np.abs(lr) > 0.5).sum())}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("radat", nargs="+", help="radat.pkl-tiedostot (netti_live.py --radat); nimi = tiedoston kansion nimi")
    a = ap.parse_args()
    rivit = []
    for p in a.radat:
        with open(p, "rb") as fh:
            rivit += rivit_radoista(pickle.load(fh), os.path.basename(os.path.dirname(os.path.abspath(p))))
    tulosta(tilastot(rivit))


if __name__ == "__main__":
    main()
