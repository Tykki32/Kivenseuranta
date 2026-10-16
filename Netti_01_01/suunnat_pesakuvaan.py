"""Liun ja irroituksen suunnat toisen paan pesakuvaan T-viivalle: python suunnat_pesakuvaan.py tulokset vasen [--alkuperainen] [--ulos tulokset/suunnat_pesakuva.png]

Heittopaan (vasen kamera) radasta lasketut suunnat jatketaan suorina viivoina radan toiseen paahan (T-viivojen etaisyys 34,747 m, rata.HOUSE_DISTANCE_CM):
  * liun suunta = X(Y):n suora hackista irroituspisteeseen (mihin kivi oli keskimaarin menossa ennen viimeista kaden vaikutusta),
  * irroituksen suunta = vapaan lennon suunta irroitushetkella (mihin kivi lahti viimeisen kaden vaikutuksen jalkeen).
Viivat piirretaan toisen paan (oikea kamera) pesan top-down-kuvaan (tulokset/topdown_<toinen>.png; kalibrointi_netti.py), jossa pesan keskipiste on T-viivan ja
keskiviivan leikkauksessa. Koordinaatit: toisen paan fyysinen koordinaatisto on heittopaan koordinaatiston 180 asteen kierto pystyakselin ymparilla
(X' = -X, Y' = 3474,7 - Y), joten heittajan nakokulmasta X on suoraan sivusuunta (+ = heittajan oikealle) ja kuva kierretaan 180 astetta (kauko-paa ylhaalla).
Suorat viivat ovat suuntia, eivat ennuste: kierre (curl) kaartaa kivea matkalla, joten viivat osoittavat tahtayspisteen (aim), johon kierre vaikuttaa.
Jos tulokset/kierto_<puoli>.json on olemassa, kierre (kiertosuunta ja nopeus) merkitaan heittojen selitteeseen."""
import argparse
import json
import os

import cv2
import numpy as np

import rata

VARI = {0: "#d4a017", 1: "#c0392b", 2: "#2c7fb8"}
NIMI = {0: "heitto 1 (keltainen)", 1: "heitto 2 (punainen)", 2: "heitto 3 (keltainen)"}
PPC = 2.0
TD_X = (-260.0, 260.0)
TD_Y = (-450.0, 900.0)


def viivat(heitto):
    """Liun ja irroituksen suorat heittopaan koordinaatistossa: X(Y) = a + b Y. Palauttaa dict (liuku, irroitus) = (a, b, kulma_deg)."""
    til, ys = heitto["tila"], heitto["yhteenveto"]
    b_i = np.tan(np.radians(til["suunta_deg"]))
    a_i = til["X_cm"] - b_i * til["Y_cm"]
    return dict(liuku=(ys["liun_suora_X_Y0_cm"], ys["liun_suora_kulmakerroin"], ys["liun_suunta_deg"]), irroitus=(a_i, b_i, til["suunta_deg"]))


def piirra(kansio, puoli, ulos, alkuperainen=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    toinen = "oikea" if puoli == "vasen" else "vasen"
    liuku = json.load(open(os.path.join(kansio, f"liuku_irroitus_{puoli}.json")))
    kp = os.path.join(kansio, f"kierto_{puoli}.json")
    kierto = json.load(open(kp)) if os.path.exists(kp) else {}
    kuva = cv2.cvtColor(cv2.imread(os.path.join(kansio, f"topdown_{toinen}.png")), cv2.COLOR_BGR2RGB)
    L = rata.HOUSE_DISTANCE_CM
    if alkuperainen:                      # toisen paan oma top-down: hog ylhaalla, kivi tulee ylhaalta alas; X' = -X
        sx, ext, ylim, img = -1.0, [TD_X[0], TD_X[1], TD_Y[0], TD_Y[1]], (-320.0, 760.0), kuva
    else:                                 # heittajan nakokulma: 180 asteen kierto, kauko-paa ylhaalla, kivi tulee alhaalta ylos
        sx, ext, ylim, img = 1.0, [TD_X[0], TD_X[1], TD_Y[1], TD_Y[0]], (760.0, -320.0), np.rot90(kuva, 2)
    fig, ax = plt.subplots(figsize=(7.2, 11.5))
    ax.set_facecolor("#282828")
    ax.imshow(img, extent=ext, aspect="equal", zorder=1)
    # radan reunat, keski- ja T-viiva
    for x in (-237.5, 237.5):
        ax.axvline(x, color="#ffa000", lw=0.8, ls="--", zorder=2)
    ax.axvline(0, color="k", lw=0.6, ls=":", zorder=2)
    ax.axhline(0, color="k", lw=1.0, zorder=2)
    ax.text(ext[0] + 6, 0 + (14 if not alkuperainen else 14), "T-viiva", fontsize=8, color="k", va="bottom" if alkuperainen else "top")
    ypts = np.linspace(ylim[0] if not alkuperainen else ylim[0], ylim[1], 200)
    risteykset = {}
    for k in sorted(liuku, key=int):
        kk = int(k)
        v = viivat(liuku[k])
        for nimi, (a, b, ang), ls, lw in (("liuku", v["liuku"], "--", 1.6), ("irroitus", v["irroitus"], "-", 2.2)):
            Yp = ypts                                   # etaisyys kaukopaan T-viivasta heittajaa kohti
            Xp = a + b * (L - Yp)                       # heittopaan X (heittajan oikealle +)
            ed = Yp >= 0                                # T-viivaan asti (heittajan puolella)
            ax.plot(sx * Xp[ed], Yp[ed], ls=ls, lw=lw, color=VARI[kk], zorder=5, solid_capstyle="round")
            ax.plot(sx * Xp[~ed], Yp[~ed], ls=":", lw=1.0, color=VARI[kk], alpha=0.7, zorder=4)
            XT = a + b * L
            risteykset[(kk, nimi)] = (XT, ang)
            ax.plot([sx * XT], [0], "o", ms=7, mfc=VARI[kk], mec="k", mew=0.8, zorder=6)
        til = liuku[k]["tila"]
        sT = (L - til["Y_cm"]) / np.cos(np.radians(til["suunta_deg"])) ** 2 * np.radians(til["suunta_virhe_deg"])
        XT = risteykset[(kk, "irroitus")][0]
        ax.plot([sx * (XT - sT), sx * (XT + sT)], [0, 0], "-", color=VARI[kk], lw=4, alpha=0.55, zorder=4, solid_capstyle="butt")
    # selite
    rivit = []
    for kk in (0, 1, 2):
        if (kk, "irroitus") not in risteykset:
            continue
        XT_i, ang_i = risteykset[(kk, "irroitus")]
        XT_l, ang_l = risteykset[(kk, "liuku")]
        rivi = f"{NIMI[kk]}\n  irroitus {ang_i:+.2f} astetta -> T-viiva {XT_i:+.0f} cm\n  liuku {ang_l:+.2f} astetta -> T-viiva {XT_l:+.0f} cm"
        if str(kk) in kierto:
            om = kierto[str(kk)]["omega_deg_s"]
            rivi += f"\n  kierre {'vastapaivaan' if om > 0 else 'myotapaivaan'} {abs(om):.0f} astetta/s"
        rivit.append(rivi)
    x_laatikko, ha = (0.015, "left") if not alkuperainen else (0.985, "right")
    for kk, rivi in zip((0, 1, 2), rivit):
        ax.text(x_laatikko, 0.985 - 0.095 * kk, rivi, transform=ax.transAxes, fontsize=7.5, va="top", ha=ha, color=VARI[kk],
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=VARI[kk], alpha=0.92), zorder=10)
    ax.plot([], [], "-", color="k", lw=2.2, label="irroituksen suunta (yhtenainen viiva): kuhun kivi lahti viimeisen kaden vaikutuksen jalkeen")
    ax.plot([], [], "--", color="k", lw=1.6, label="liun suunta (katkoviiva): kuhun kivi oli keskimaarin menossa ennen sita (hackista irroitukseen)")
    ax.plot([], [], "--", color="#ffa000", lw=0.8, label="radan reunat (+-237,5 cm)")
    ax.legend(loc="lower right", fontsize=6.5, framealpha=0.92)
    # kulkusuunta
    ax.annotate("", xy=(ext[0] + 22, 560.0), xytext=(ext[0] + 22, 700.0), arrowprops=dict(arrowstyle="-|>", color="w", lw=2), zorder=9)
    ax.text(ext[0] + 32, 630.0, "kiven kulkusuunta\n(hog-linjalta pesaan)", color="w", fontsize=7, va="center", zorder=9)
    ax.set_xlim(-330, 330)
    ax.set_ylim(*ylim)
    sivu = "sama kuin heittajan oikea/vasen" if not alkuperainen else "kuvassa oikea = heittajan vasen"
    ax.set_xlabel(f"sivusuunta keskiviivasta (cm); {sivu}")
    ax.set_ylabel("etaisyys T-viivasta kaukopaan pesasta heittajaa kohti (cm)")
    ax.set_title(f"Suunnat toisen paan pesakuvassa ({'heittajan nakokulma' if not alkuperainen else 'toisen paan top-down'})", fontsize=10)
    fig.tight_layout()
    fig.savefig(ulos, dpi=110)
    return risteykset


def yhteenveto(kansio, puoli, risteykset=None):
    """Kayttajan kysymykset heitoittain (CSV + markdown): kierre (nopeus, suunta), irroituksen suunta, liun suunta, nopeus irroituksen jalkeen ja juuri ennen, liun
    hidastuvuus; lisaksi viivojen osumakohdat toisen paan T-viivalla."""
    liuku = json.load(open(os.path.join(kansio, f"liuku_irroitus_{puoli}.json")))
    kp = os.path.join(kansio, f"kierto_{puoli}.json")
    kierto = json.load(open(kp)) if os.path.exists(kp) else {}
    L = rata.HOUSE_DISTANCE_CM
    rivit = []
    for k in sorted(liuku, key=int):
        kk = int(k)
        y, til = liuku[k]["yhteenveto"], liuku[k]["tila"]
        v = viivat(liuku[k])
        d = dict(heitto=kk + 1,
                 kierre_deg_s=kierto[k]["omega_deg_s"] if k in kierto else None,
                 kierre_vali_deg_s=kierto[k]["omega_vali_deg_s"] if k in kierto else None,
                 kierre_suunta=kierto[k]["suunta"] if k in kierto else None,
                 irroituksen_suunta_deg=y["irroituksen_suunta_deg"], irroituksen_suunta_virhe_deg=y["irroituksen_suunta_virhe_deg"],
                 liun_suunta_deg=y["liun_suunta_deg"],
                 nopeus_jalkeen_m_s=y["nopeus_jalkeen_m_s"], nopeus_ennen_m_s=y["nopeus_ennen_m_s"],
                 liun_hidastuvuus_m_s2=y["liun_hidastuvuus_m_s2"], liun_hidastuvuus_vali_m_s2=y["liun_hidastuvuus_vali_m_s2"],
                 vapaan_lennon_hidastuvuus_m_s2=y["vapaan_lennon_hidastuvuus_m_s2"],
                 T_viiva_irroitus_cm=v["irroitus"][0] + v["irroitus"][1] * L, T_viiva_liuku_cm=v["liuku"][0] + v["liuku"][1] * L)
        rivit.append(d)
    polku = os.path.join(kansio, f"irroitus_yhteenveto_{puoli}")
    import csv
    with open(polku + ".csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["heitto", "kierre_deg_s (+ = vastapaivaan)", "kierre_vali_deg_s", "irroituksen_suunta_deg", "irroituksen_suunta_virhe_deg", "liun_suunta_deg",
                     "nopeus_irroituksen_jalkeen_m_s", "nopeus_juuri_ennen_irroitusta_m_s", "liun_hidastuvuus_m_s2", "liun_hidastuvuus_vali_m_s2", "vapaan_lennon_hidastuvuus_m_s2",
                     "T_viiva_irroitus_cm", "T_viiva_liuku_cm"])
        for d in rivit:
            wr.writerow([d["heitto"], f"{d['kierre_deg_s']:.0f}" if d["kierre_deg_s"] is not None else "", f"{d['kierre_vali_deg_s'][0]:.0f}...{d['kierre_vali_deg_s'][1]:.0f}" if d["kierre_vali_deg_s"] else "",
                         f"{d['irroituksen_suunta_deg']:+.2f}", f"{d['irroituksen_suunta_virhe_deg']:.2f}", f"{d['liun_suunta_deg']:+.2f}", f"{d['nopeus_jalkeen_m_s']:.2f}", f"{d['nopeus_ennen_m_s']:.2f}",
                         f"{d['liun_hidastuvuus_m_s2']:+.3f}", f"{d['liun_hidastuvuus_vali_m_s2'][0]:+.3f}...{d['liun_hidastuvuus_vali_m_s2'][1]:+.3f}", f"{d['vapaan_lennon_hidastuvuus_m_s2']:+.3f}",
                         f"{d['T_viiva_irroitus_cm']:+.0f}", f"{d['T_viiva_liuku_cm']:+.0f}"])
    return rivit, polku


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("kansio")
    ap.add_argument("puoli")
    ap.add_argument("--alkuperainen", action="store_true", help="toisen paan oma top-down-suunta (hog ylhaalla) eika heittajan nakokulma")
    ap.add_argument("--ulos", default=None)
    a = ap.parse_args()
    ulos = a.ulos or os.path.join(a.kansio, "suunnat_pesakuva" + ("_alkuperainen" if a.alkuperainen else "") + ".png")
    r = piirra(a.kansio, a.puoli, ulos, a.alkuperainen)
    yv, ypolku = yhteenveto(a.kansio, a.puoli)
    for d in yv:
        print(f"heitto {d['heitto']}: kierre {d['kierre_deg_s']} astetta/s {d['kierre_suunta']}, irroituksen suunta {d['irroituksen_suunta_deg']:+.2f}, liun suunta {d['liun_suunta_deg']:+.2f}, "
              f"nopeus jalkeen {d['nopeus_jalkeen_m_s']:.2f} / ennen {d['nopeus_ennen_m_s']:.2f} m/s, liun hidastuvuus {d['liun_hidastuvuus_m_s2']:+.3f} m/s2")
    print("->", ypolku + ".csv")
    for (kk, n), (XT, ang) in sorted(r.items()):
        print(f"heitto {kk + 1} {n}: suunta {ang:+.2f} astetta -> T-viivalla {XT:+.0f} cm keskiviivasta (+ = heittajan oikealle)")
    print("->", ulos)
