"""Pelaajakohtainen tilasto heitto-CSV:sta (netti_live.py --csv): mediaani ja STD sarakkeille liun_hidastuvuus_m_s2, irroitus_delta_cm ja irroitus vauhti.

    python tools/pelaajatilasto.py heitot.csv                            # kysyy pelaajamaarat
    python tools/pelaajatilasto.py heitot.csv --pelaajia 4               # kummassakin joukkueessa 4 pelaajaa
    python tools/pelaajatilasto.py heitot.csv --keltainen 4 --punainen 3
    python tools/pelaajatilasto.py heitot.csv --punainen 3,3,2           # 3 pelaajaa, kivet 1-3, 4-6 ja 7-8 (sama kuin oletus kolmelle pelaajalle)
    python tools/pelaajatilasto.py heitot.csv --fi                       # Excelille: erotin ;, desimaalipilkku, UTF-8 BOM

Joukkue = kiven vari (CSV:n sarake vari). Pelaaja lasketaan kiven numerosta (sarake kivi_nro 1-8, vareittain per paa, kuten kuvassa nakyva "Keltainen 3"): pelaajat heittavat kivensa
perakkain numerojarjestyksessa. Oletusjako: 4 pelaajaa = kivet 1-2, 3-4, 5-6, 7-8 (pelaaja 1 = ensimmainen heittaja, pelaaja 4 = viimeinen); 3 pelaajaa = 1-3, 4-6, 7-8; 2 pelaajaa = 1-4, 5-8.
Muu jako annetaan kivimaarina pilkulla erotettuna (esim. --punainen 2,3,3: ensimmainen heittaa kaksi kivea, toinen ja kolmas kolme).

Taulukko pelaajaa kohti, kuten Excelissa: Mediaani | STD sarakkeille
    liun_hidastuvuus_m_s2   liun hidastuvuus (m/s2, sarake liun_hidastuvuus_m_s2)
    irroitus_delta_cm       irroituksen delta (cm, sarake irroitus_delta_cm)
    irroitus vauhti         nopeus irroituksen jalkeen - nopeus ennen irroitusta (m/s, nopeus_jalkeen_m_s - nopeus_ennen_m_s)
STD on otoskeskihajonta (n - 1, kuten Excelin STDEV.S; --std-populaatio: jakaja n). Kunkin luvun laskentaan kaytetaan heittoja, joilta kyseinen mittaus on olemassa (n_ sarakkeet); yksittaisesta
arvosta ei saa hajontaa (tyhja). Tulos tulostetaan ja tallennetaan CSV:ksi (oletus <syote>_pelaajat.csv; --ei-tallenna: vain tuloste).

Huom: kivinumero tulee seurannasta. Jos paassa on vahemman kuin joukkueen kivet (video alkoi tai loppui kesken paan, tai seuranta ohitti heiton), varoitus kertoo paan; --ohita-puutteelliset
jattaa tallaiset paat pois. Jos analyysi aloitettiin kesken paan, ensimmaisen paan kivinumerot ovat aluksi vaarin (korjaa kivi_nro CSV:ssa tai ohita se)."""
import argparse
import csv
import os
import statistics
import sys

KIVIA_PAASSA = 8                       # joukkueen kivet / paa (kuten netti_seuranta.KIVIA_PAASSA: kivi_nro 1-8)
OLETUS_PELAAJIA = 4
MITAT = ("liun_hidastuvuus_m_s2", "irroitus_delta_cm", "irroitus vauhti")          # sarakeotsikot kuten Excelissa
DES_TULOSTE = (3, 1, 3)                # desimaalit tulosteessa
DES_CSV = (4, 2, 4)                    # desimaalit tallennettavassa CSV:ssa
PAKOLLISET = ("vari", "kivi_nro", "liun_hidastuvuus_m_s2", "irroitus_delta_cm", "nopeus_ennen_m_s", "nopeus_jalkeen_m_s")


def luku(s):
    """Merkkijono -> float (desimaalipilkku kelpaa) tai None (tyhja, virheellinen, NaN)."""
    if s is None:
        return None
    s = str(s).strip().replace(",", ".")
    if not s:
        return None
    try:
        x = float(s)
    except ValueError:
        return None
    return x if x == x else None


def lue(polku):
    """CSV-rivit (dict). Erotin (, tai ;) tunnistetaan otsikkorivista."""
    with open(polku, newline="", encoding="utf-8-sig") as fh:
        ensimmainen = fh.readline()
        fh.seek(0)
        erotin = ";" if ensimmainen.count(";") > ensimmainen.count(",") else ","
        lukija = csv.DictReader(fh, delimiter=erotin)
        return list(lukija), list(lukija.fieldnames or [])


def jako(maaare, kivia=KIVIA_PAASSA):
    """Pelaajien kivimaarat heittojarjestyksessa. '4' (pelaajien maara) -> [2, 2, 2, 2]; '3' -> [3, 3, 2]; '2,3,3' (kivia pelaajaa kohti) -> [2, 3, 3]. ValueError, jos virheellinen."""
    osat = [x for x in str(maaare).replace(" ", "").split(",") if x != ""]
    try:
        luvut = [int(x) for x in osat]
    except ValueError:
        raise ValueError(f"ei kokonaislukuja: {maaare!r}") from None
    if not luvut or any(n < 1 for n in luvut):
        raise ValueError(f"pelaajia tai kivia pitaa olla vahintaan 1: {maaare!r}")
    if len(luvut) == 1:
        n = luvut[0]
        if n > kivia:
            raise ValueError(f"pelaajia enintaan {kivia} (kivia {kivia} / paa): {maaare!r}")
        pohja, yli = divmod(kivia, n)
        return [pohja + (1 if i < yli else 0) for i in range(n)]
    return luvut


def kivet_tekstina(alku, n):
    return str(alku) if n == 1 else f"{alku}-{alku + n - 1}"


def paa_numerot(rivit, kentat):
    """Kunkin rivin paa: sarake paa, tai (ellei sita ole) paa vaihtuu kun heittosuunta (sarake puoli) vaihtuu."""
    if "paa" in kentat and all(luku(r.get("paa")) is not None for r in rivit):
        return [int(luku(r["paa"])) for r in rivit]
    paa, edell, out = 0, None, []
    for r in rivit:
        if r.get("puoli") != edell:
            paa += 1
            edell = r.get("puoli")
        out.append(paa)
    return out


def mediaani(x):
    return statistics.median(x) if x else None


def hajonta(x, populaatio=False):
    """Otoskeskihajonta (n - 1; populaatio: n); None, jos arvoja on liian vahan."""
    if populaatio:
        return statistics.pstdev(x) if x else None
    return statistics.stdev(x) if len(x) >= 2 else None


def mitat(r):
    """(liun hidastuvuus, irroituksen delta, irroitusvauhti = nopeus jalkeen - nopeus ennen) yhdelta heitolta; puuttuva mittaus on None."""
    ennen, jalkeen = luku(r.get("nopeus_ennen_m_s")), luku(r.get("nopeus_jalkeen_m_s"))
    return luku(r.get("liun_hidastuvuus_m_s2")), luku(r.get("irroitus_delta_cm")), (None if ennen is None or jalkeen is None else jalkeen - ennen)


def laske(rivit, kentat, jaot, ohita_puutteelliset=False, populaatio=False):
    """(tulokset, varoitukset). jaot: {vari: [pelaajan kivimaara, ...]}. tulokset: [dict(joukkue, pelaaja, kivet, heittoja, n, mediaani, std)] joukkueittain ja pelaajajarjestyksessa."""
    varoitukset = []
    paat = paa_numerot(rivit, kentat)
    lasku = {}                                                       # (paa, vari) -> heittoja
    for r, p in zip(rivit, paat):
        lasku[(p, r["vari"])] = lasku.get((p, r["vari"]), 0) + 1
    huonot = set()
    for p in sorted({p for p, _ in lasku}):
        osat = []
        for v, lohkot in jaot.items():
            odotettu, n = sum(lohkot), lasku.get((p, v), 0)
            if n != odotettu:
                osat.append(f"{v} {n} / {odotettu}")
        if osat:
            huonot.add(p)
            varoitukset.append(f"paa {p}: heittoja {', '.join(osat)} (ei taydellinen paa: kivinumerot ja pelaajajako voivat olla tassa paassa virheellisia)")
    keratty = {}                                                     # (vari, pelaaja) -> [mitat]
    pelaaja_kivelle = {v: [p for p, n in enumerate(lohkot, 1) for _ in range(n)] for v, lohkot in jaot.items()}
    for r, p in zip(rivit, paat):
        if ohita_puutteelliset and p in huonot:
            continue
        v, nro = r["vari"], luku(r.get("kivi_nro"))
        if v not in jaot:
            continue
        if nro is None or not (1 <= int(nro) <= len(pelaaja_kivelle[v])):
            varoitukset.append(f"heitto {r.get('heitto', '?')} ({v}, kivi_nro {r.get('kivi_nro') or 'puuttuu'}): ei kuulu kenellekaan pelaajalle, ohitettu")
            continue
        keratty.setdefault((v, pelaaja_kivelle[v][int(nro) - 1]), []).append(mitat(r))
    tulokset = []
    for v, lohkot in jaot.items():
        alku = 1
        for p, n in enumerate(lohkot, 1):
            heitot = keratty.get((v, p), [])
            sarakkeet = [[m[i] for m in heitot if m[i] is not None] for i in range(3)]
            tulokset.append(dict(joukkue=v, pelaaja=p, kivet=kivet_tekstina(alku, n), heittoja=len(heitot), n=tuple(len(s) for s in sarakkeet),
                                 mediaani=tuple(mediaani(s) for s in sarakkeet), std=tuple(hajonta(s, populaatio) for s in sarakkeet)))
            alku += n
    return tulokset, varoitukset


def _muoto(x, des, pilkku=False):
    s = "" if x is None else f"{x:.{des}f}"
    return s.replace(".", ",") if pilkku else s


def taulukko(tulokset):
    """Tulosteen rivit (Mediaani / STD -ryhmaotsikko, sarakeotsikot, pelaajarivit), sarakkeet tasattu."""
    otsikot = ["joukkue", "pelaaja", "kivet", "heittoja"] + list(MITAT) + list(MITAT)
    rivit = [[t["joukkue"], str(t["pelaaja"]), t["kivet"], str(t["heittoja"])] + [(_muoto(x, d) or "-") for x, d in zip(t["mediaani"], DES_TULOSTE)] +
             [(_muoto(x, d) or "-") for x, d in zip(t["std"], DES_TULOSTE)] for t in tulokset]
    leveys = [max(len(otsikot[i]), *(len(r[i]) for r in rivit)) if rivit else len(otsikot[i]) for i in range(len(otsikot))]

    def rivi(solut):
        return "  ".join(s.ljust(leveys[i]) if i < 3 else s.rjust(leveys[i]) for i, s in enumerate(solut))
    alku_med = sum(leveys[:4]) + 2 * 4
    alku_std = alku_med + sum(leveys[4:7]) + 2 * 3
    ryhma = " " * alku_med + "Mediaani".ljust(alku_std - alku_med) + "STD"
    return [ryhma.rstrip(), rivi(otsikot)] + [rivi(r) for r in rivit]


def kirjoita_csv(polku, tulokset, fi=False):
    """Tallentaa taulukon: ensimmainen rivi ryhmat (Mediaani, STD), toinen sarakeotsikot, sitten pelaajat; viimeiset sarakkeet n_* kertovat, montako heittoa kunkin luvun laskentaan kaytettiin."""
    with open(polku, "w", newline="", encoding="utf-8-sig" if fi else "utf-8") as fh:
        w = csv.writer(fh, delimiter=";" if fi else ",")
        w.writerow(["", "", "", "", "Mediaani", "", "", "STD", "", "", "n", "", ""])
        w.writerow(["joukkue", "pelaaja", "kivet", "heittoja"] + list(MITAT) + list(MITAT) + ["liun_hidastuvuus_m_s2", "irroitus_delta_cm", "irroitus vauhti"])
        for t in tulokset:
            w.writerow([t["joukkue"], t["pelaaja"], t["kivet"], t["heittoja"]] + [_muoto(x, d, fi) for x, d in zip(t["mediaani"], DES_CSV)] +
                       [_muoto(x, d, fi) for x, d in zip(t["std"], DES_CSV)] + list(t["n"]))


def jarjestys(rivit):
    """Joukkueet (varit) ensiesiintymisjarjestyksessa."""
    out = []
    for r in rivit:
        if r["vari"] and r["vari"] not in out:
            out.append(r["vari"])
    return out


def kysy(vari, syotto=input):
    """Kysyy pelaajamaaran (tai kivijaon); Enter = oletus. Toistaa, kunnes syote kelpaa."""
    while True:
        vastaus = syotto(f"Montako pelaajaa joukkueessa '{vari}'? (Enter = {OLETUS_PELAAJIA}; tai kivet pilkuilla, esim. 3,3,2): ").strip() or str(OLETUS_PELAAJIA)
        try:
            jako(vastaus)
            return vastaus
        except ValueError as e:
            print(f"  virhe: {e}", file=sys.stderr)


def main(argv=None, syotto=input, interaktiivinen=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="netti_live.py:n heitto-CSV (heitot_*.csv)")
    ap.add_argument("--pelaajia", metavar="N", help="pelaajia kummassakin joukkueessa (tai kivijako 3,3,2); ohitetaan joukkuekohtaisella valitsimella")
    ap.add_argument("--keltainen", metavar="N", help="keltaisen joukkueen pelaajamaara (tai kivijako, esim. 3,3,2)")
    ap.add_argument("--punainen", metavar="N", help="punaisen joukkueen pelaajamaara (tai kivijako, esim. 3,3,2)")
    ap.add_argument("--ulos", metavar="CSV", help="tulostiedosto (oletus: <syote>_pelaajat.csv)")
    ap.add_argument("--ei-tallenna", action="store_true", help="vain tuloste, ei CSV-tiedostoa")
    ap.add_argument("--fi", action="store_true", help="suomalainen Excel: erotin ;, desimaalipilkku ja UTF-8 BOM tulostiedostoon")
    ap.add_argument("--ohita-puutteelliset", action="store_true", help="jata pois paat, joissa jonkin joukkueen heittojen maara ei ole sama kuin sen kivet (video alkoi / loppui kesken paan tai heitto jai tunnistamatta)")
    ap.add_argument("--std-populaatio", action="store_true", help="STD jakajalla n (oletus n - 1, kuten Excelin STDEV.S)")
    a = ap.parse_args(argv)
    if not os.path.isfile(a.csv):
        print(f"tiedostoa ei loydy: {a.csv}", file=sys.stderr)
        return 2
    rivit, kentat = lue(a.csv)
    puuttuu = [k for k in PAKOLLISET if k not in kentat]
    if puuttuu:
        print(f"CSV:sta puuttuvat sarakkeet: {', '.join(puuttuu)} (tiedoston pitaa olla netti_live.py:n heitto-CSV)", file=sys.stderr)
        return 2
    rivit = [r for r in rivit if (r.get("vari") or "").strip()]
    if not rivit:
        print("CSV:ssa ei ole heittoja", file=sys.stderr)
        return 2
    varit = jarjestys(rivit)
    if interaktiivinen is None:
        interaktiivinen = sys.stdin is not None and sys.stdin.isatty()
    jaot = {}
    for v in varit:
        maaare = getattr(a, v, None) if v in ("keltainen", "punainen") else None
        if maaare is None:
            maaare = a.pelaajia
        if maaare is None:
            if interaktiivinen:
                maaare = kysy(v, syotto)
            else:
                maaare = str(OLETUS_PELAAJIA)
                print(f"pelaajamaaraa ei annettu joukkueelle '{v}': oletus {OLETUS_PELAAJIA} (--pelaajia N tai --{v} N)", file=sys.stderr)
        try:
            jaot[v] = jako(maaare)
        except ValueError as e:
            ap.error(f"joukkue '{v}': {e}")
    for v, lohkot in jaot.items():
        if sum(lohkot) != KIVIA_PAASSA:
            print(f"Huom: joukkueen '{v}' kivijako {','.join(str(n) for n in lohkot)} on yhteensa {sum(lohkot)} kivea (paassa oletetaan {KIVIA_PAASSA})", file=sys.stderr)
    tulokset, varoitukset = laske(rivit, kentat, jaot, a.ohita_puutteelliset, a.std_populaatio)
    print(f"{os.path.basename(a.csv)}: {len(rivit)} heittoa; " + "; ".join(f"{v}: {len(l)} pelaajaa (kivet {', '.join(kivet_tekstina(sum(l[:i]) + 1, n) for i, n in enumerate(l))})" for v, l in jaot.items()))
    print("\n".join(taulukko(tulokset)))
    for v in varoitukset:
        print("Huom:", v)
    if not a.ei_tallenna:
        ulos = a.ulos or (os.path.splitext(a.csv)[0] + "_pelaajat.csv")
        kirjoita_csv(ulos, tulokset, a.fi)
        print(f"tallennettu: {ulos}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
