"""Kiven koon ja korkeuden pysyvyys: python testaa_kivi_koko.py VIDEO tulokset vasen [--ulos tulokset/kivi_koko_vasen.png] [--ikkunat 8-22,56-66,104-114]

Jokaiselle ruudulle VAPAA sovitus (X, Y, sade R (keskikorkeudella), korkeus H) kiinteaksi opitulla profiilin muodolla (tulokset/kivimalli_<puoli>.json). Jos
kamera ja malli ovat oikein, R ja H pysyvat vakiona radan alusta loppuun ja kolmella kivella: tama on mallin koe (kayttaa vain ruutuja, joissa
kivi nakyy hyvin; huonosti sopivat ja kasin peittamat ruudut karsitaan). Kirjoittaa taulukon (Y-lokerot, heitot), CSV:n ja kuvan."""
import argparse
import csv
import os
from multiprocessing import Pool

import numpy as np

import kalibroi_kivi as kk
import kivi_netti as kv
import kivi_tarkka as kt

_G = {}


def _alusta(kansio, puoli):
    _G["cam"] = kv.Kamera(puoli, os.path.join(kansio, f"kalibrointi_{puoli}.json"), os.path.join(kansio, f"moodikuva_{puoli}.png"))
    _G["runko"] = kt.Runko.lataa(kansio, puoli)


def _vapaa_tyo(d):
    s = kt.sovita(_G["cam"], _G["runko"], d["alfa"], d["ok"], d["X0"], d["Y0"], vapaa=True, ox=d["ox"], oy=d["oy"])
    s["frame"], s["rata"], s["t"] = d["frame"], d["rata"], d["t"]
    return s


def tilasto(rows, runko, lokerot):
    """Taulukko: Y-lokero x heitto: R-mediaani, H-mediaani."""
    out = []
    for lo, hi in lokerot:
        r = {"lo": lo, "hi": hi}
        for rata in (0, 1, 2):
            q = [x for x in rows if x["rata"] == rata and lo <= x["Y"] < hi]
            if len(q) >= 3:
                R = np.array([x["sR"] * runko.R_keski for x in q])
                Hh = np.array([x["H"] for x in q])
                r[rata] = (len(q), float(np.median(R)), float(np.median(Hh)))
        q = [x for x in rows if lo <= x["Y"] < hi]
        if len(q) >= 3:
            R = np.array([x["sR"] * runko.R_keski for x in q])
            Hh = np.array([x["H"] for x in q])
            r["kaikki"] = (len(q), float(np.median(R)), float(R.std()), float(np.median(Hh)), float(Hh.std()))
        out.append(r)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("kansio")
    ap.add_argument("puoli")
    ap.add_argument("--ikkunat", default="8-22,56-66,104-114")
    ap.add_argument("--ulos", default=None)
    ap.add_argument("--rms-max", type=float, default=0.12)
    ap.add_argument("--kelv-min", type=float, default=0.35)
    ap.add_argument("--ymax", type=float, default=700.0)
    a = ap.parse_args()
    ikk = [tuple(float(x) for x in w.split("-")) for w in a.ikkunat.split(",")]
    H = kk.kerää_havainnot(a.video, a.kansio, a.puoli, ikk)
    runko = kt.Runko.lataa(a.kansio, a.puoli)
    q = [d for d in H if d["rms"] < a.rms_max and d["kelvolliset"] > a.kelv_min and d["Y0"] < a.ymax]
    # alkuvaiheen levossa oleva kivi (Y < -150) pelkistetaan: enintaan joka 4. ruutu
    q = [d for i, d in enumerate(q) if d["Y0"] > -150 or i % 4 == 0]
    with Pool(4, initializer=_alusta, initargs=(a.kansio, a.puoli)) as pool:
        res = pool.map(_vapaa_tyo, q, chunksize=4)
    ulos = a.ulos or os.path.join(a.kansio, f"kivi_koko_{a.puoli}.png")
    csv_p = os.path.splitext(ulos)[0] + ".csv"
    with open(csv_p, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["heitto", "frame", "t_s", "X_cm", "Y_cm", "R_keski_cm", "H_cm", "rms", "kelvolliset"])
        for s in res:
            wr.writerow([s["rata"] + 1, s["frame"], f"{s['t']:.3f}", f"{s['X']:.2f}", f"{s['Y']:.2f}", f"{s['sR'] * runko.R_keski:.3f}", f"{s['H']:.3f}",
                         f"{s['rms']:.4f}", f"{s['kelvolliset']:.3f}"])
    print(f"profiilin R(H/2) = {runko.R_keski:.2f} cm (suurin {runko.R_max:.2f}), H = {runko.H:.2f} cm; vapaa sovitus {len(res)} ruudulle -> {csv_p}")
    lokerot = [(-300, -150), (-150, 0), (0, 150), (150, 300), (300, 450), (450, 600), (600, 750)]
    print("Y-lokero (cm)    n   R(H/2) med (sd) cm   H med (sd) cm   | heitto 1 R/H | heitto 2 R/H | heitto 3 R/H")
    for r in tilasto(res, runko, lokerot):
        if "kaikki" not in r:
            continue
        n, Rm, Rs, Hm, Hs = r["kaikki"]
        s = f"{r['lo']:5d}..{r['hi']:4d}  {n:4d}   {Rm:6.2f} ({Rs:4.2f})     {Hm:6.2f} ({Hs:4.2f})   |"
        for rata in (0, 1, 2):
            s += f" {r[rata][1]:5.2f}/{r[rata][2]:5.2f}  |" if rata in r else "      -        |"
        print(s)
    R = np.array([s["sR"] * runko.R_keski for s in res])
    Hh = np.array([s["H"] for s in res])
    Y = np.array([s["Y"] for s in res])
    rata = np.array([s["rata"] for s in res])
    m = (Y > 100) & (Y < 650)
    print(f"Y 100..650 cm: R(H/2) = {np.median(R[m]):.2f} cm (hajonta {R[m].std():.2f}), H = {np.median(Hh[m]):.2f} cm (hajonta {Hh[m].std():.2f})")
    for k in range(3):
        mm = m & (rata == k)
        if mm.sum():
            print(f"  heitto {k + 1}: R(H/2) {np.median(R[mm]):.2f} cm, H {np.median(Hh[mm]):.2f} cm ({mm.sum()} ruutua)")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    vari = ["#d4a017", "#c0392b", "#2c7fb8"]
    for k, nimi in enumerate(["heitto 1 (keltainen)", "heitto 2 (punainen)", "heitto 3 (keltainen)"]):
        for j, (arr, yl) in enumerate(((R, "R keskikorkeudella (cm)"), (Hh, "H (cm)"))):
            mk = rata == k
            ax[j].plot(Y[mk], arr[mk], ".", ms=4, color=vari[k], alpha=0.35)
            ys, ms = [], []
            for lo in range(-300, 750, 75):
                z = mk & (Y >= lo) & (Y < lo + 75)
                if z.sum() >= 3:
                    ys.append(lo + 37.5)
                    ms.append(np.median(arr[z]))
            ax[j].plot(ys, ms, "-o", color=vari[k], lw=2, label=nimi if j == 0 else None)
            ax[j].set_xlabel("Y (cm, talon keskipisteesta)")
            ax[j].set_ylabel(yl)
    ax[0].axhline(runko.R_keski, color="k", lw=0.8, ls=":")
    ax[1].axhline(runko.H, color="k", lw=0.8, ls=":")
    ax[0].set_ylim(runko.R_keski - 0.8, runko.R_keski + 0.8)
    ax[1].set_ylim(runko.H - 3, runko.H + 3)
    ax[0].set_title("kiven sade keskikorkeudella, ruutukohtainen vapaa sovitus (viiva = lokeroiden mediaani)")
    ax[1].set_title("kiven korkeus kalibrointitasosta (ylarengas)")
    ax[0].legend(loc="lower left")
    for x in ax:
        x.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ulos, dpi=110)
    print("kuva ->", ulos)
