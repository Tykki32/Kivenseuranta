"""Heittopaan kierteen estimaattorin (netti_kierre.posteriori_omega) yksikkotesti synteettisilla kulmakustannuskayrilla (ei videota).

    python tools/testaa_kierre.py

Tarkistetaan: (1) omega palautuu kaikilla nopeuksilla +-26 ... +-480 deg/s (ei kovia rajoja), (2) suunta oikein ja varmuus (P) sopii virheeseen, (3) epavarmuus on kalibroitu (virhe / sd), (4) peilattu
kasivaihe (kulma theta nakyy kulmana 2 a - theta, V-muoto; "lataus") ei kaanna suuntaa, kun puhtaita ruutuja on vahan, (5) tulos on toistettava."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import netti_kierre as nk

KAPPA = 0.0083              # kustannuksen nousu / aste^2 puhtaan ruudun kayrassa (6 asteen paassa ~0,3)
SIG_PUHDAS = 4.5            # puhtaan ruudun kulmakohina (aste)
AXIS = 90.0                 # peilauksen akseli (aste)


def ero(a, b):
    return (a - b + 90.0) % 180.0 - 90.0


def kayra(th_hav, cmin, toinen=None, toinen_lisa=0.15):
    th = nk.KULMAT
    c = cmin + KAPPA * ero(th, th_hav) ** 2
    if toinen is not None:
        c = np.minimum(c, cmin + toinen_lisa + KAPPA * ero(th, toinen) ** 2)
    return c


def synteettinen(omega, rng, th0=40.0, n_puhdas=14, n_kasi=10, fps=25.0, peilaus=False):
    """Kasivaihe (n_kasi ruutua: kasi peittaa, kohina ~15 aste, paras kulma joko oikea tai peilattu, toinen minimi toisena) ja sen jalkeen puhtaat ruudut (sigma 4,5 aste)."""
    n = n_kasi + n_puhdas
    t = np.arange(n) / fps
    th_tosi = (th0 + omega * (t - t.mean())) % 180.0
    Cz, nkz, nvz = [], [], []
    for i in range(n):
        if i < n_kasi:
            th_hav = (th_tosi[i] + rng.normal(0, 15.0)) % 180.0
            peili = (2 * AXIS - th_tosi[i]) % 180.0
            if peilaus:
                Cz.append(kayra(peili, 14.0, toinen=th_hav))             # paras kulma on peilikuva, oikea kulma toisena minimina
            else:
                Cz.append(kayra(th_hav, 14.0, toinen=peili))
            nkz.append(80)
        else:
            th_hav = (th_tosi[i] + rng.normal(0, SIG_PUHDAS)) % 180.0
            Cz.append(kayra(th_hav, 4.0))
            nkz.append(0)
        nvz.append(500)
    return dict(t=t, f=t * fps, Y=np.zeros(n), Cz=np.array(Cz), nkz=np.array(nkz), nvz=np.array(nvz))


def synteettinen_sarana(omega, rng, t_c=0.5, th_p=120.0, n_kasi=22, n_puhdas=10, fps=25.0):
    """Saranamalli: kahvan kulma on vakio th_p hetkeen t_c (s irroituksesta) asti ja kiertyy sen jalkeen vakionopeudella; ruudut t_irr + nk.SARANA_T_ALKU alkaen, kasi peittaa ensimmaiset n_kasi ruutua
    (kohina ~15 aste), loput puhtaita. t_irr = 0. Palauttaa (k, t_irr). t_c annetaan ikkunan alun -0,6 s mukaan ja siirretaan ikkunan alun mukana (nk.SARANA_T_ALKU + 0,6): rakenne (kasi, puhtaat, kierron alku ikkunassa) ei riipu ikkunan sijainnista irroitukseen nahden."""
    n = n_kasi + n_puhdas
    t_c = t_c + (nk.SARANA_T_ALKU + 0.6)
    t = nk.SARANA_T_ALKU + np.arange(n) / fps
    th_tosi = np.where(t < t_c, th_p, th_p + omega * (t - t_c)) % 180.0
    Cz, nkz, nvz = [], [], []
    for i in range(n):
        if i < n_kasi:
            Cz.append(kayra((th_tosi[i] + rng.normal(0, 15.0)) % 180.0, 14.0, toinen=(2 * AXIS - th_tosi[i]) % 180.0)); nkz.append(80)
        else:
            Cz.append(kayra((th_tosi[i] + rng.normal(0, SIG_PUHDAS)) % 180.0, 4.0)); nkz.append(0)
        nvz.append(500)
    return dict(t=t, f=t * fps, Y=np.zeros(n), Cz=np.array(Cz), nkz=np.array(nkz), nvz=np.array(nvz)), 0.0


def arvioi(k):
    P = nk.posteriori_omega(k)
    s, mu, sd, om_ml, p = nk.tiivista_posteriori(P)
    return s, mu, sd, p


def main():
    rng = np.random.default_rng(1)
    virheita = 0
    print("omega (deg/s) | suunta oikein | |virhe| < 3 sd | P(suunta) mediaani | sd mediaani   [n_kasi = 10, n_puhdas = 14, 25 ruutua/s]")
    for om in (26.0, 60.0, 110.0, 214.0, 480.0, -26.0, -60.0, -110.0, -214.0, -480.0):
        ok_s = ok_z = 0
        ps, sds = [], []
        n_try = 12
        for _ in range(n_try):
            s, mu, sd, p = arvioi(synteettinen(om, rng, th0=float(rng.uniform(0, 180))))
            ok_s += s == np.sign(om)
            ok_z += abs(mu - om) < 3.0 * max(sd, 1.0)
            ps.append(p); sds.append(sd)
        print(f"{om:+8.0f}      | {ok_s:2d} / {n_try}       | {ok_z:2d} / {n_try}        | {np.median(ps):.3f}              | {np.median(sds):5.1f}")
        if abs(om) >= 60 and ok_s < n_try:
            virheita += 1; print("  VIRHE: suunta vaarin")
        if ok_z < n_try - 2:
            virheita += 1; print("  VIRHE: virhe yli 3 sd liian usein (epavarmuus ei kalibroitu)")
    # peilattu kasivaihe: puhtaita vahan; suunta ei saa kaantya varmasti vaaraan
    print("peilattu kasivaihe (V-muoto), puhtaita ruutuja 8:")
    vaara_varma = 0
    for om in (60.0, 90.0, -60.0, -90.0):
        ok = 0
        n_try = 12
        for _ in range(n_try):
            s, mu, sd, p = arvioi(synteettinen(om, rng, th0=float(rng.uniform(0, 180)), n_puhdas=8, n_kasi=14, peilaus=True))
            ok += s == np.sign(om)
            if s != np.sign(om) and p >= nk.SUUNTA_VARMA:
                vaara_varma += 1
        print(f"  omega {om:+4.0f}: suunta oikein {ok} / {n_try}")
    if vaara_varma:
        virheita += 1; print(f"  VIRHE: {vaara_varma} kertaa vaara suunta varmana (P >= {nk.SUUNTA_VARMA})")
    # saranamalli: plateau (kasi pitaa kahvaa, kulma vakio) ja vasta sitten kierto; lineaarinen malli aliarvioisi omegan, saranamalli ei
    print("saranamalli A (kierto alkaa t_c = 0 ... 0,3 s, kasi 22 ruutua, puhtaita 14):  omega | suunta oikein, |virhe| < 3 sd, sd | lineaarisen |omega| / tosi")
    for om in (60.0, 110.0, 214.0, 480.0, -60.0, -110.0, -480.0):
        ok_s = ok_z = 0
        sds, lin = [], []
        n_try = 8
        for _ in range(n_try):
            k, t_irr = synteettinen_sarana(om, rng, t_c=float(rng.uniform(0.0, 0.3)), th_p=float(rng.uniform(0, 180)), n_puhdas=14)
            P = nk.posteriori_omega_sarana(k, t_irr)
            s_, mu, sd, _, p = nk.tiivista_posteriori(P)
            ok_s += s_ == np.sign(om); ok_z += abs(mu - om) < 3.0 * max(sd, 1.0); sds.append(sd)
            lin.append(abs(arvioi(k)[1]) / abs(om))
        print(f"  {om:+5.0f}  | {ok_s:2d} / {n_try}, {ok_z:2d} / {n_try}, sd {np.median(sds):5.1f} | {np.median(lin):.2f}")
        if ok_s < n_try or ok_z < n_try - 1:
            virheita += 1; print("  VIRHE: saranamalli ei palauta omegaa")
    print("saranamalli B (myohainen kierto t_c = 0,5 ... 0,9 s, puhtaita vain 6): ei saa olla varmasti vaarin; 3 sd -kattavuus")
    vaara_varma = 0
    ok_z = n_all = 0
    for om in (60.0, 110.0, -60.0, -110.0):
        for _ in range(10):
            k, t_irr = synteettinen_sarana(om, rng, t_c=float(rng.uniform(0.5, 0.9)), th_p=float(rng.uniform(0, 180)), n_puhdas=6)
            s_, mu, sd, _, p = nk.tiivista_posteriori(nk.posteriori_omega_sarana(k, t_irr))
            vaara_varma += (s_ != np.sign(om)) and p >= nk.SUUNTA_VARMA
            ok_z += abs(mu - om) < 3.0 * max(sd, 1.0) or s_ != np.sign(om) and False
            n_all += 1
    print(f"  varmasti vaara suunta {vaara_varma} / {n_all}; omega 3 sd:n sisalla {ok_z} / {n_all}")
    if vaara_varma > 0:
        virheita += 1; print("  VIRHE: varmasti vaara suunta myohaisella kierrolla")
    # toistettavuus
    a = synteettinen(110.0, np.random.default_rng(5)); b = synteettinen(110.0, np.random.default_rng(5))
    if arvioi(a) != arvioi(b):
        virheita += 1; print("VIRHE: tulos ei ole toistettava")
    print("OK" if virheita == 0 else f"{virheita} VIRHETTA")
    raise SystemExit(0 if virheita == 0 else 1)


if __name__ == "__main__":
    main()
