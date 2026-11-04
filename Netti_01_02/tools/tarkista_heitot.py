"""Tarkistaa netti_live.py:n tunnistamien heittojen rakenteen: puuttuuko heittoja (hukattuja kiviä)?

    python tools/tarkista_heitot.py heitot.csv                  # netti_live.py --csv heitot.csv -ajon tulos
    python tools/tarkista_heitot.py ajo.log                     # netti_live.py -ajon konsolituloste (kaikki vahvistetut heitot, myos ilman viivaestimaattia)
    python tools/tarkista_heitot.py ajo1.log ajo2.log --per-paa 16

Pelissa heitetaan vuorotellen keltainen / punainen, 8 kiveä per väri per pää (16 heittoa), ja päät vaihtuvat (vasen, oikea, vasen, ...). Tarkistus ryhmittelee heitot päihin
(heittopuolen vaihtuessa alkaa uusi pää) ja ilmoittaa
  * päät, joissa on alle --per-paa heittoa,
  * kohdat, joissa sama väri heitetään kahdesti peräkkäin (välistä puuttuu toisen värin heitto: väli aikoina),
  * kivinumeroiden epäjatkuvuudet (väreittäin 1, 2, 3 ...; numerointi siirtyy, jos heitto jää tunnistamatta).
CSV:ssä ovat kaikki valmistuneet heitot (myös ne, joille ei saatu viivaestimaattia: viivasarakkeet tyhjiä; sarakkeet ks. tools/yhteenveto.py ja README_netti.md); konsolitulosteessa (lokissa) kaikki
vahvistetut heitot. Poistumiskoodi 1, jos jotain puuttuu."""
import argparse
import csv
import re
import sys

LOKI_RIVI = re.compile(r"heitto (\d+): (\w+) (\d+) \((\w+)\) vahvistettu, lahto ([\d.]+) s")


def lue(polku):
    """[(lahto_s, vari, numero, puoli)] aikajarjestyksessa."""
    rivit = []
    if polku.lower().endswith(".csv"):
        with open(polku, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                rivit.append((float(r["alku_s"]), r["vari"].lower(), int(r["kivi_nro"] or 0), r["puoli"]))
    else:
        with open(polku, encoding="utf-8", errors="replace") as fh:
            for rivi in fh:
                m = LOKI_RIVI.search(rivi)
                if m:
                    rivit.append((float(m[5]), m[2].lower(), int(m[3]), m[4]))
    return sorted(rivit)


def paat(rivit):
    """Ryhmittely paihin: uusi paa, kun heittopuoli vaihtuu."""
    ryhmat = []
    for r in rivit:
        if not ryhmat or ryhmat[-1][0][3] != r[3]:
            ryhmat.append([])
        ryhmat[-1].append(r)
    return ryhmat


def tarkista(rivit, per_paa):
    ongelmia = 0
    for i, paa in enumerate(paat(rivit), 1):
        vareja = {v: sum(1 for r in paa if r[1] == v) for v in sorted({r[1] for r in paa})}
        yht = ", ".join(f"{n} {v}" for v, n in vareja.items())
        huom = []
        for a, b in zip(paa, paa[1:]):
            if a[1] == b[1]:
                muu = "punainen" if a[1] == "keltainen" else "keltainen"
                huom.append(f"{a[0]:.0f} - {b[0]:.0f} s: {a[1]} kahdesti peräkkäin, väliltä puuttuu {muu} heitto")
        for v in vareja:
            nrot = [r[2] for r in paa if r[1] == v]
            if nrot != list(range(1, len(nrot) + 1)):
                huom.append(f"{v}: kivinumerot {nrot} (pitäisi olla 1..{len(nrot)}): numerointi on siirtynyt")
        if len(paa) < per_paa:
            huom.append(f"vain {len(paa)} heittoa ({per_paa} odotettu): puuttuu {per_paa - len(paa)}")
        elif len(paa) > per_paa:
            huom.append(f"{len(paa)} heittoa ({per_paa} odotettu): ylimääräisiä {len(paa) - per_paa} (väärä tunnistus tai pää ryhmittyi väärin)")
        ongelmia += len(huom)
        print(f"pää {i} ({paa[0][3]}, {paa[0][0]:.0f} - {paa[-1][0]:.0f} s): {len(paa)} heittoa ({yht})" + ("" if huom else "  OK"))
        for h in huom:
            print(f"    HUOM {h}")
    return ongelmia


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tiedostot", nargs="+", help="netti_live.py:n CSV (--csv) tai konsolitulosteen loki")
    ap.add_argument("--per-paa", type=int, default=16, help="heittoja per pää (oletus 16 = 8 kiveä x 2 väriä)")
    a = ap.parse_args()
    rivit = []
    for p in a.tiedostot:
        rivit += lue(p)
    rivit.sort()
    if not rivit:
        print("Ei yhtään heittoa tiedostoissa (odotettu: CSV tai loki, jossa rivit 'heitto N: Vari K (puoli) vahvistettu, lahto T s').")
        return 2
    n = tarkista(rivit, a.per_paa)
    print(f"yhteensä {len(rivit)} heittoa, {n} huomautusta")
    return 1 if n else 0


if __name__ == "__main__":
    sys.exit(main())
