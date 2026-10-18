"""Yhteenveto netti_live.py:n heitto-CSV:sta: kaukopaan seuranta, kierteen varmuus, kiven vakiot kivikohtaisesti ja joukkueiden kello.

    python tools/yhteenveto.py heitot.csv
    python tools/yhteenveto.py heitot.csv --kivet            # lisaksi kunkin kiven (vari, numero) vakiot ja hajonnan jako kivien valille / kiven sisalle

Tulostaa
  * kaukopaan seuranta: kuinka monelle heitolle kivi loydettiin, seurattiin pysahtymiseen asti, tormays, hog-viivan ylitys,
  * kierre: heittojen osuus, joiden kierrosmaara on 'varma' (kahden paan mittaus + vaimeneminen) ja mittausten lahde,
  * kiven vakiot (rata_malli.py: kitkataso A, kaarto kappa): sovitusten onnistuminen ja jakauma; --kivet: kivikohtaiset mediaanit ja varianssianalyysi
    (jos kivi on yksilollinen (vari, numero), kiven sisainen hajonta on pienempi kuin kivien valinen),
  * joukkueiden kello: miettimisaika heittokohtaisesti (mediaani, ylin) ja yhteensa joukkueittain."""
import argparse
import csv
import sys

import numpy as np


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def lue(polku):
    with open(polku, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _osuus(n, kaikki):
    return f"{n} / {kaikki} ({100.0 * n / kaikki:.0f} %)" if kaikki else "-"


def kauko(rivit):
    print("KAUKOPAAN SEURANTA")
    n = len(rivit)
    tilat = {}
    for r in rivit:
        tilat[r["kauko_tila"] or "ei seurantaa"] = tilat.get(r["kauko_tila"] or "ei seurantaa", 0) + 1
    print("  heittoja", n, "; tila:", ", ".join(f"{k} {v}" for k, v in sorted(tilat.items())))
    print("  hog-viivan ylitys kaukopaassa:", _osuus(sum(1 for r in rivit if r["hog2_s"]), n), "; mitattu B-radasta:", _osuus(sum(1 for r in rivit if r["hog2_lahde"] == "B-rata"), n))
    print("  tormayksia (nopeus putoaa jyrkasti):", _osuus(sum(1 for r in rivit if r["tormays_s"]), n))
    pys = [r for r in rivit if r["kauko_tila"] == "pysahtyi"]
    if pys:
        X = np.array([_f(r["lepo_X_cm"]) for r in pys])
        Y = np.array([_f(r["lepo_Y_cm"]) for r in pys])
        D = np.array([_f(r["lepo_napista_cm"]) for r in pys])
        print(f"  pysahtyneiden lepopaikka (X + = heittajan oikealle, Y = T-viivasta heittajaa kohti): X {np.median(X):.0f} cm ({X.min():.0f} ... {X.max():.0f}), Y {np.median(Y):.0f} cm ({Y.min():.0f} ... {Y.max():.0f});"
              f" etaisyys napista mediaani {np.median(D):.0f} cm, pesassa (<= 183 cm) {int((D <= 183).sum())} / {len(D)}")


def kierre(rivit):
    print("KIERRE")
    n = len(rivit)
    mitt = [r for r in rivit if r["kierre"]]
    print("  kierrosmaara saatu:", _osuus(len(mitt), n), "; varma:", _osuus(sum(1 for r in mitt if r["kierre_varma"] == "1"), n))
    lahteet = {}
    for r in mitt:
        lahteet[r["kierre_lahde"]] = lahteet.get(r["kierre_lahde"], 0) + 1
    print("  mittausten lahde:", ", ".join(f"{k} {v}" for k, v in sorted(lahteet.items())))
    sg = np.array([_f(r["kierre_sigma"]) for r in mitt if r["kierre_sigma"]])
    if len(sg):
        print(f"  kierrosmaaran epavarmuus (1 sigma): mediaani {np.median(sg):.2f}, 90 % alle {np.percentile(sg, 90):.2f} kierrosta")
    kk = np.array([_f(r["kierre_vaimeneminen_1_s"]) for r in mitt if r["kierre_lahde"] == "kaksi mittausta" and r["kierre_vaimeneminen_1_s"]])
    if len(kk):
        print(f"  mitattu vaimeneminen (kun molemmat paat): mediaani {100 * np.median(kk):.1f} % / s, hajonta {100 * kk.std():.1f} % / s (n = {len(kk)})")


def _anova(ryhmat):
    """Yksisuuntainen varianssianalyysi: (F, p, kiven sisainen sd, kivien valinen sd (ylimaarainen)) tai None."""
    ryhmat = [np.asarray(g, float) for g in ryhmat if len(g) >= 2]
    if len(ryhmat) < 2:
        return None
    N = sum(len(g) for g in ryhmat)
    k = len(ryhmat)
    mu = np.mean(np.concatenate(ryhmat))
    ssb = sum(len(g) * (g.mean() - mu) ** 2 for g in ryhmat)
    ssw = sum(((g - g.mean()) ** 2).sum() for g in ryhmat)
    if N - k <= 0 or ssw <= 0:
        return None
    F = (ssb / (k - 1)) / (ssw / (N - k))
    try:
        from scipy.stats import f as fd
        p = float(fd.sf(F, k - 1, N - k))
    except Exception:
        p = float("nan")
    msw = ssw / (N - k)
    n0 = (N - sum(len(g) ** 2 for g in ryhmat) / N) / (k - 1)
    var_valilla = max((ssb / (k - 1) - msw) / n0, 0.0)
    return float(F), p, float(np.sqrt(msw)), float(np.sqrt(var_valilla))


def vakiot(rivit, kivet):
    print("KIVEN VAKIOT (rata_malli.py; tehollisia: harjaus pienentaa kitkaa ja kaarroa)")
    n = len(rivit)
    ok = [r for r in rivit if r["vakio_A"]]
    kap = [r for r in ok if r["vakio_kappa_ok"] == "1" and r["vakio_kappa"]]
    print("  sovitus onnistui (A):", _osuus(len(ok), n), "; kappa (kierre rajattu mittauksella):", _osuus(len(kap), n))
    if ok:
        A = np.array([_f(r["vakio_A"]) for r in ok])
        print(f"  kitkataso A: mediaani {np.median(A):.4f}, hajonta (1,48 MAD) {1.4826 * np.median(np.abs(A - np.median(A))):.4f}, valille {A.min():.4f} ... {A.max():.4f}")
    if kap:
        K = np.array([_f(r["vakio_kappa"]) for r in kap])
        print(f"  kaarto kappa: mediaani {np.median(K):.2f}, hajonta (1,48 MAD) {1.4826 * np.median(np.abs(K - np.median(K))):.2f}, valille {K.min():.2f} ... {K.max():.2f}")
    if not kivet:
        return
    print("  kivikohtaisesti (vari numero: heittoja, A mediaani +- hajonta; kappa mediaani +- hajonta):")
    ryh, ryhk = {}, {}
    for r in ok:
        ryh.setdefault((r["vari"], int(r["kivi_nro"] or 0)), []).append(_f(r["vakio_A"]))
    for r in kap:
        ryhk.setdefault((r["vari"], int(r["kivi_nro"] or 0)), []).append(_f(r["vakio_kappa"]))
    for avain in sorted(ryh):
        a = np.array(ryh[avain])
        k = np.array(ryhk.get(avain, []))
        ka = f"{np.median(k):.2f} +- {k.std():.2f} (n = {len(k)})" if len(k) > 1 else (f"{k[0]:.2f}" if len(k) == 1 else "-")
        print(f"    {avain[0]:10s} {avain[1]}: {len(a)} heittoa, A {np.median(a):.4f} +- {a.std():.4f}; kappa {ka}")
    for nimi, ryhmat in (("A", ryh), ("kappa", ryhk)):
        r = _anova(list(ryhmat.values()))
        if r is None:
            continue
        F, p, sd_sis, sd_val = r
        print(f"  {nimi}: varianssianalyysi (kivi = vari + numero): F = {F:.2f}, p = {p:.3f}; kiven sisainen hajonta {sd_sis:.4g}, kivien valinen {sd_val:.4g}"
              + ("  -> kiven vakiot eroavat toisistaan (kivi on yksilollinen)" if p < 0.05 else "  -> kivien valista eroa ei voi osoittaa tasta aineistosta"))


def irroitus(rivit):
    print("IROITUKSEN USKOTTAVUUS (nopeus ja suunta irroituksessa vs. hog-viivalla suoraan radasta mitattu; kaukopaan pysahtymismatka vs. rata_malli)")
    n = len(rivit)
    dv = np.array([_f(r["nopeus_jalkeen_m_s"]) - _f(r["nopeus_hog1_m_s"]) for r in rivit if _f(r.get("nopeus_jalkeen_m_s")) is not None and _f(r.get("nopeus_hog1_m_s")) is not None])
    ds = np.array([_f(r["irroituksen_suunta_deg"]) - _f(r["suunta_hog1_deg"]) for r in rivit if _f(r.get("irroituksen_suunta_deg")) is not None and _f(r.get("suunta_hog1_deg")) is not None])
    hid = np.array([_f(r["hidastuvuus_m_s2"]) for r in rivit if _f(r.get("hidastuvuus_m_s2")) is not None])
    if len(dv):
        print(f"  nopeus irroituksen jalkeen - nopeus hog-viivalla: mediaani {np.median(dv):+.3f} m/s (odotus +0,05 ... +0,15: kivi vain hidastuu), 1,48 MAD {1.4826 * np.median(np.abs(dv - np.median(dv))):.3f};"
              f" alle -0,05 m/s (irroitus tunnistettu liian aikaisin): {_osuus(int((dv < -0.05).sum()), len(dv))}")
    if len(ds):
        print(f"  suunta irroituksessa - suunta hog-viivalla: mediaani {np.median(ds):+.2f} astetta, 1,48 MAD {1.4826 * np.median(np.abs(ds - np.median(ds))):.2f}; yli 0,5 asteen ero: {_osuus(int((np.abs(ds) > 0.5).sum()), len(ds))}")
    if len(hid):
        print(f"  vapaan lennon hidastuvuus < 0 (kivi kiihtyisi irroituksen jalkeen, mahdotonta): {_osuus(int((hid < 0).sum()), len(hid))}")
    tk = [r for r in rivit if r.get("irr_tarkistus")]
    if tk:
        ep = [r for r in tk if r["irr_tarkistus"] == "epailyttava"]
        e = np.array([_f(r["irr_matka_ero_m"]) for r in tk])
        A = np.array([_f(r["irr_A_matka"]) for r in tk])
        print(f"  pysahtymismatka vs. malli (kaukopaan seuranta pysahtyi, ei tormaysta): {len(tk)} heittoa; ero (havaittu - ennuste) mediaani {np.median(e):+.1f} m, 1,48 MAD {1.4826 * np.median(np.abs(e - np.median(e))):.1f} m;"
              f" kitkataso koko matkalle mediaani {np.median(A):.4f}, hajonta {100 * 1.4826 * np.median(np.abs(A - np.median(A))) / np.median(A):.0f} %; epailyttavia (ero yli 2 m): {_osuus(len(ep), len(tk))}")
        for r in ep:
            print(f"    heitto {r['heitto']:>3} {r['vari']:9s} {r['kivi_nro']}: ero {_f(r['irr_matka_ero_m']):+.1f} m, nopeus irroituksessa {_f(r['nopeus_jalkeen_m_s']):.2f} m/s (hog-viivalla {_f(r['nopeus_hog1_m_s']) if _f(r['nopeus_hog1_m_s']) is not None else float('nan'):.2f}), irroitus Y {_f(r['irroitus_Y_cm']):.0f} cm")


def kello(rivit):
    print("JOUKKUEIDEN KELLO (miettimisaika: edellisen kiven pysahtymisesta siihen, kun oma kivi osuu heittopaan T-viivaan; ei kay paiden valilla)")
    aj = {}
    for r in rivit:
        a = _f(r["miettimisaika_s"])
        if a is not None:
            aj.setdefault(r["vari"], []).append(a)
    for vari, a in sorted(aj.items()):
        a = np.array(a)
        print(f"  {vari:10s}: {len(a)} heittoa, yhteensa {a.sum():.0f} s ({int(a.sum() // 60)}:{a.sum() % 60:04.1f}), mediaani {np.median(a):.1f} s, ylin {a.max():.1f} s, alle 3 s: {int((a < 3).sum())}")
    paat = {}
    for r in rivit:
        a = _f(r["miettimisaika_s"])
        if a is not None:
            paat.setdefault(r["paa"], {}).setdefault(r["vari"], 0.0)
            paat[r["paa"]][r["vari"]] += a
    if paat:
        print("  paittain (s):", "; ".join(f"paa {p}: " + ", ".join(f"{v} {s:.0f}" for v, s in sorted(d.items())) for p, d in sorted(paat.items(), key=lambda x: int(x[0]))))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--kivet", action="store_true", help="kivikohtaiset vakiot ja varianssianalyysi")
    a = ap.parse_args()
    rivit = lue(a.csv)
    if not rivit:
        print("CSV on tyhja")
        return 2
    print(f"{len(rivit)} heittoa, {len({r['paa'] for r in rivit})} paata\n")
    for f_ in (kauko, kierre):
        f_(rivit)
        print()
    vakiot(rivit, a.kivet)
    print()
    irroitus(rivit)
    print()
    kello(rivit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
