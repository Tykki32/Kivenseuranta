"""Heittopaan kiven seurannan onnistuminen: python tools/seurannan_laatu.py radat.pkl [radat2.pkl ...] [--heitot 10] [--vyohykkeet]

DIAGNOSTIIKKAA, ei suodatusta: mittarit kertovat, kuinka hyvin kiven tunnistus ja seuranta onnistui (radat.pkl = netti_live.py --radat), jotta virheiden juurisyy voidaan loytaa. Mikaan
mittari ei muuta rataa eika karsi pisteita. Mittarit vyohykkeittain (kiven Y heittopaan koordinaateissa: hack -330, pesan keskipiste 0, hog-viiva ~640 cm):

  * hypyt: B-pisteiden jaannos paikallisesta sileasta radasta (ruutuindeksin funktiona, ei aikaperustan arviosta: leave-one-out, robusti toisen asteen sovitus +-12 ruutua), osuus
    pisteista, joilla jaannos > 8 cm ja > 15 cm. Kiven todellinen rata on sile; lahdevirran pudonneet ruudut tekevat yksittaisen askeleen (valkoisella jaalla Y > 300 cm jaannos > 8 cm
    on ~2 %: tama on mittarin pohja), joten vaikeus nakyy vain, kun osuus on tata selvasti suurempi (pesan renkailla ja heittajan kaden kohdalla ennen korjausta ~20 %).
  * pesan reunan epajatkuvuus: pesan sisapuolen pisteiden (Y 120...183 cm) Y-poikkeama suorasta, joka on sovitettu valkoisen puolen pisteisiin (Y 183...265 cm). Valkoisella jaalla kiven
    paikka on varma, joten jos B jaa pesan renkailla kiven taakse (kasia kohti), seuranta "hyppaa" kohdalleen juuri renkaiden reunalla. Mediaani heitoittain; |poikkeama| > 5 / > 10 cm.
  * takaisinliike: peräkkaiset B-pisteet, joiden Y pienenee > 2 cm, vaikka kivi etenee (Y -150...450 cm): kivi ei peruuta.
  * A-B-sopu: kahvan blobin (A, riippumaton mittaus) ja siluettisovituksen (B) Y-eron robusti hajonta ja osuudet > 6 / > 12 cm, kun heiton oma mediaaniero on poistettu (A:n
    systemaattinen poikkeama kiven akselista). Jos B seuraa kaytta kiven sijaan, ero kasvaa.
  * kattavuus: B-pisteiden osuus ruuduista lahdon ja Y = 450 cm valilla ja pisin B-aukko (ruutua).
  * irroituksen jalkeinen vapaan lennon hidastuvuus (viivaestimaatti): kivi hidastuu, ei kiihdy (hidastuvuus > 0).
"""
import argparse
import pickle
import sys

import numpy as np

VYOHYKKEET = [(-150, 0), (0, 100), (100, 190), (190, 300), (300, 450)]   # pesan reuna (12 jalan rengas) on Y = 183 cm
PESAN_REUNA_CM = 183.0
IKKUNA = 12            # ruutua
HYPPY_CM = (8.0, 15.0)


def jaannokset(f, P, w=IKKUNA, min_n=6):
    """Jokaiselle pisteelle jaannos (cm) paikallisesta robustista toisen asteen sovituksesta (piste itse pois): [n]; NaN, jos naapureita liian vahan."""
    n = len(f)
    d = np.full(n, np.nan)
    for i in range(n):
        m = (np.abs(f - f[i]) <= w) & (np.arange(n) != i)
        if m.sum() < min_n:
            continue
        tt, Q = f[m] - f[i], P[m]
        wt = np.ones(len(tt))
        c = None
        for _ in range(4):
            B = np.column_stack([np.ones(len(tt)), tt, tt * tt])
            c = np.linalg.lstsq(B * wt[:, None], Q * wt[:, None], rcond=None)[0]
            r = Q - B @ c
            dd = np.hypot(r[:, 0], r[:, 1])
            s = max(1.4826 * float(np.median(dd)), 0.5)
            wt = np.clip(1.0 - (dd / (4.685 * s)) ** 2, 0.0, 1.0)
            if (wt > 0).sum() < min_n:
                break
        d[i] = float(np.hypot(*(P[i] - c[0])))
    return d


def reunan_epajatkuvuus(f, P):
    """Pesan reunan epajatkuvuus (cm, Y-suunnassa; negatiivinen = pesan puolella kiven taakse) tai None, jos pisteita liian vahan."""
    Y = P[:, 1]
    v = (Y >= PESAN_REUNA_CM) & (Y < 265.0)
    s = (Y >= 120.0) & (Y < PESAN_REUNA_CM)
    if v.sum() < 5 or s.sum() < 3:
        return None
    c = np.polyfit(f[v] - f[v].mean(), Y[v], 1)
    return float(np.median(Y[s] - np.polyval(c, f[s] - f[v].mean())))


def heiton_mitat(h):
    """Yhden heiton mittarit (dict) tai None, jos B-pisteita on alle 12."""
    pts = [p for p in h["near"] if p[3]]
    if len(pts) < 12:
        return None
    f = np.array([p[0] for p in pts], float)
    P = np.array([[p[1], p[2]] for p in pts], float)
    Y = P[:, 1]
    d = jaannokset(f, P)
    out = dict(vyoh={}, back=0, ab={}, kattavuus=None, max_aukko=None, hidastuvuus=None, raja=reunan_epajatkuvuus(f, P))
    for z in VYOHYKKEET:
        m = np.isfinite(d) & (Y >= z[0]) & (Y < z[1])
        out["vyoh"][z] = (int(m.sum()), int((d[m] > HYPPY_CM[0]).sum()), int((d[m] > HYPPY_CM[1]).sum()))
    m = (Y > -150) & (Y < 450)
    out["back"] = int((np.diff(Y[m]) < -2.0).sum())
    # A-B-sopu
    A = {int(a[0]): (a[2], a[3]) for a in h.get("kahva_ala", [])}
    e = np.array([(p[2], A[int(p[0])][1] - p[2]) for p in h["near"] if p[3] and int(p[0]) in A]).reshape(-1, 2)
    if len(e) >= 10:
        med = float(np.median(e[:, 1]))
        for z in VYOHYKKEET:
            mm = (e[:, 0] >= z[0]) & (e[:, 0] < z[1])
            out["ab"][z] = (e[mm, 1] - med).tolist()
    # kattavuus: B-ruudut / kaikki ruudut lahdosta Y = 450:een
    f_alku = h.get("f_alku") or f[0]
    loppu = f[np.argmax(Y > 450)] if (Y > 450).any() else f[-1]
    valit = np.diff(f[(f >= f_alku) & (f <= loppu)])
    n_kaikki = int(loppu - f_alku) + 1
    out["kattavuus"] = float(((f >= f_alku) & (f <= loppu)).sum() / max(n_kaikki, 1))
    out["max_aukko"] = int(valit.max() - 1) if len(valit) else 0
    v = h.get("viivat") or {}
    out["hidastuvuus"] = v.get("vapaa_hidastuvuus")
    return out


def raportti(nimi, radat, n_pahimmat=0, vyohykkeet=True):
    tot = {z: np.zeros(3) for z in VYOHYKKEET}
    ab = {z: [] for z in VYOHYKKEET}
    back = kat = 0
    n = 0
    hidas = []
    aukot = []
    rajat = []
    rivit = []
    for h in radat:
        m = heiton_mitat(h)
        if m is None:
            continue
        n += 1
        for z in VYOHYKKEET:
            tot[z] += m["vyoh"][z]
            ab[z] += m["ab"].get(z, [])
        back += m["back"]
        kat += m["kattavuus"]
        aukot.append(m["max_aukko"])
        if m["raja"] is not None:
            rajat.append(m["raja"])
        if m["hidastuvuus"] is not None:
            hidas.append(m["hidastuvuus"])
        yht = sum(m["vyoh"][z][1] for z in VYOHYKKEET)
        rivit.append((yht, h["tunnus"], h.get("f_alku"), m))
    print(f"{nimi}: {n} heittoa")
    if vyohykkeet:
        print("  vyohyke Y (cm)    pisteita   hyppy > 8 cm   > 15 cm   | A-B-sopu: robusti sd   > 6 cm   > 12 cm")
        for z in VYOHYKKEET:
            a = np.array(ab[z])
            ab_s = (f"{1.4826 * np.median(np.abs(a - np.median(a))):7.2f} cm {np.mean(np.abs(a) > 6):8.1%} {np.mean(np.abs(a) > 12):8.1%}" if len(a) else "      -")
            print(f"  {z[0]:5d}..{z[1]:4d}      {int(tot[z][0]):7d}   {tot[z][1] / max(tot[z][0], 1):10.1%} {tot[z][2] / max(tot[z][0], 1):9.1%}   |        {ab_s}")
    print(f"  takaisinliikkeita (dY < -2 cm): {back} ({back / max(n, 1):.2f} / heitto); B-kattavuus lahdosta Y 450 cm:iin keskimaarin {kat / max(n, 1):.1%}; pisin B-aukko mediaani {np.median(aukot):.0f} ruutua, max {max(aukot) if aukot else 0}")
    if rajat:
        r = np.abs(np.array(rajat))
        print(f"  pesan reunan epajatkuvuus (Y): {len(r)} heittoa, |poikkeama| mediaani {np.median(r):.1f} cm, > 5 cm: {np.mean(r > 5):.0%}, > 10 cm: {np.mean(r > 10):.0%}; mediaani etumerkillisena {np.median(rajat):+.1f} cm")
    if hidas:
        h = np.array(hidas, float)
        print(f"  irroituksen jalkeinen hidastuvuus: mediaani {np.median(h):.3f} m/s2, <= 0: {int((h <= 0).sum())} / {len(h)}")
    if n_pahimmat:
        print(f"  {n_pahimmat} eniten > 8 cm hyppyja:")
        for yht, tun, fa, m in sorted(rivit, key=lambda r: -r[0])[:n_pahimmat]:
            print(f"    heitto {tun} (alku {fa / 30.0 if fa else 0:.1f} s): hyppyja yhteensa {yht}, takaisinliikettä {m['back']}, pisin B-aukko {m['max_aukko']} ruutua")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("radat", nargs="+", help="radat.pkl (netti_live.py --radat)")
    ap.add_argument("--heitot", type=int, default=0, help="listaa nain monta eniten hyppivaa heittoa")
    a = ap.parse_args()
    for p in a.radat:
        with open(p, "rb") as fh:
            raportti(p, pickle.load(fh), a.heitot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
