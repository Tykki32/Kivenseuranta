"""B-sovituksen toistuvan, paikkaan sidotun Y-harhan (netti_seuranta.HOG_HARHA_CM) estimointi tallennetuista radoista.

    python netti_live.py --lahde netti.mp4 --nopea --ei-ikkunaa --ei-kierretta --radat radat.pkl     # kerran (radat.pkl sisaltaa viivaestimaatin tarkan syotteen: viivat_syote)
    python tools/kalibroi_paikkavirhe.py radat.pkl                                                    # tulostaa taulukon, jonka voi liittaa netti_seuranta.HOG_HARHA_CM:ksi

Kiven etureunan ylittaessa hog-viivan (Y ~ 632 cm) B-sovituksen Y nayttaa ~1 cm liian suurelta (viivan vari kiven siluetin reunalla). Harha on sama kaikilla heitoilla ja molemmilla vareilla, mutta sen paikallinen
2. derivaatta (~0,4 m/s2) nayttaa tyonnon alulta ja loppumiselta ja sekoittaa irroituksen tunnistuksen (irroitus_muutos). Menetelma: heitot, joiden irroitus on ennen Y = --raja (vapaata lentoa sen jalkeen),
sovitetaan paraabelilla (Y(t), alue 480...740 cm) ja jaannosten mediaani heittojen yli antaa harhan Y:n funktiona; pienin ja suurin paikallinen osa vahvistetaan vakioksi (paikallinen kohouma: arvo 0 alueen 590...700
ulkopuolella, jana kohouman reunojen valilla vahennetaan). Kierrokset: irroituksen tunnistus ajetaan kunkin kierroksen taulukolla korjatulle radalle (vapaiden heittojen joukko tarkentuu), kunnes taulukko ei muutu."""
import argparse
import os
import pickle
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import liuku_irroitus as li  # noqa: E402
import netti_seuranta as ns  # noqa: E402

ALKU, LOPPU = 480.0, 740.0          # paraabelisovituksen alue (cm)
TUKI = (590.0, 700.0)               # kohouman tuki: harha = 0 taman ulkopuolella
ASKEL = 10.0


def radat(heitot, fps_lahde, fps_ulos):
    """[(t, X, Y raaka)] kaikille heitoille, joilla on viivat_syote."""
    out = []
    for h in heitot:
        s = h.get("viivat_syote")
        if not s:
            continue
        f, X, Y = (np.array(c, float) for c in zip(*s))
        t, X, Y, _ = ns.valmistele_rata(f, X, Y, fps_lahde, fps_ulos, None, korjaa=False)
        out.append((t, X, Y))
    return out


def korjaa(Y, taulukko):
    if not len(taulukko):
        return Y
    ys, e = np.array(taulukko, float).T
    return Y - np.interp(Y, ys, e, left=0.0, right=0.0)


def vapaat(rs, taulukko, raja):
    """Indeksit heitoille, joiden irroitus (korjatulla radalla) on merkitseva ja ennen rajaa."""
    out = []
    ESTI = ns.ESTIMAATTI
    for i, (t, X, Y) in enumerate(rs):
        ir = li.irroitus_muutos(t, X, korjaa(Y, taulukko), Y_r_min=ESTI.get("Y_r_min"), Y_r_max=ESTI.get("Y_r_max"), min_jalkeen=ESTI.get("min_jalkeen"))
        if ir is not None and ir["merkitseva"] and not ir["tyonto_jatkuu"] and ir["Y_r"] <= raja:
            out.append(i)
    return out


def taulukko_vapaista(rs, idx):
    """Mediaanijaannos paraabelista (raaka Y) Y:n funktiona, kohouma vakiotukeen. Palauttaa ((Y, harha cm), ...) ja heittokohtaisen hajonnan."""
    yy = np.arange(ALKU, LOPPU + 1e-9, ASKEL)
    R = []
    for i in idx:
        t, X, Y = rs[i]
        m = (Y >= ALKU - 20.0) & (Y <= LOPPU + 20.0)
        if int(m.sum()) < 15:
            continue
        r = Y[m] - np.polyval(np.polyfit(t[m], Y[m], 2), t[m])
        R.append(np.interp(yy, Y[m], r, left=np.nan, right=np.nan))
    R = np.array(R)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)             # alueen reunalla ei kaikilla heitoilla havaintoja
        med = np.nanmedian(R, axis=0)
        sd = 1.4826 * np.nanmedian(np.abs(R - med), axis=0)
    med = np.convolve(np.pad(med, 1, mode="edge"), np.ones(3) / 3, mode="valid")           # 3 pisteen liukuva keskiarvo
    i0, i1 = int(round((TUKI[0] - ALKU) / ASKEL)), int(round((TUKI[1] - ALKU) / ASKEL))
    kohouma = med[i0:i1 + 1] - np.linspace(med[i0], med[i1], i1 - i0 + 1)                  # jana tuen reunojen valilla pois
    return tuple((float(y), round(float(e), 2)) for y, e in zip(yy[i0:i1 + 1], kohouma)), len(R), float(np.nanmedian(sd[i0:i1 + 1]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("radat", help="netti_live.py --radat -tiedosto (pickle)")
    ap.add_argument("--raja", type=float, default=470.0, help="vapaa lento: irroitus ennen tata Y:ta (cm)")
    ap.add_argument("--kierroksia", type=int, default=3)
    ap.add_argument("--fps-lahde", type=float, default=25.0)
    ap.add_argument("--fps", type=float, default=30.0)
    a = ap.parse_args()
    rs = radat(pickle.load(open(a.radat, "rb")), a.fps_lahde, a.fps)
    taulukko = ()
    for k in range(a.kierroksia):
        idx = vapaat(rs, taulukko, a.raja)
        uusi, n, sd = taulukko_vapaista(rs, idx)
        muutos = max((abs(u[1] - v[1]) for u, v in zip(uusi, taulukko)), default=float("inf")) if len(taulukko) == len(uusi) else float("inf")
        print(f"kierros {k + 1}: {len(rs)} heittoa, vapaita (irroitus <= {a.raja:.0f} cm) {n}; heittokohtainen hajonta kohouman alueella {sd:.2f} cm; taulukon muutos {muutos:.2f} cm")
        taulukko = uusi
        if muutos < 0.05:
            break
    print("\nHOG_HARHA_CM = (" + ", ".join(f"({y:.0f}, {e:+.2f})" for y, e in taulukko) + ")")
    print(f"suurin harha {max(taulukko, key=lambda x: abs(x[1]))[1]:+.2f} cm kohdassa Y = {max(taulukko, key=lambda x: abs(x[1]))[0]:.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
