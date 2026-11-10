"""Pelaajakohtainen tilasto heitto-CSV:sta (netti_live.py --csv): Mediaani ja STD sarakkeille Liuku, Heitto ja Tönäisy (kierron suunnan mukaan: Myötä / Vasta) sekä Hidastuvuus.

    python tools/pelaajatilasto.py heitot.csv                            # kysyy pelaajamaarat
    python tools/pelaajatilasto.py heitot.csv --pelaajia 4               # kummassakin joukkueessa 4 pelaajaa
    python tools/pelaajatilasto.py heitot.csv --keltainen 4 --punainen 3
    python tools/pelaajatilasto.py heitot.csv --punainen 3,3,2           # 3 pelaajaa, kivet 1-3, 4-6 ja 7-8 (sama kuin oletus kolmelle pelaajalle)
    python tools/pelaajatilasto.py heitot.csv --fi                       # Excelille: erotin ;, desimaalipilkku, UTF-8 BOM
    python tools/pelaajatilasto.py heitot.csv --hidastuvuus vapaa        # Hidastuvuus = vapaan lennon hidastuvuus (kuten kuvan paneelissa)

Joukkue = kiven vari (CSV:n sarake vari). Pelaaja lasketaan kiven numerosta (sarake kivi_nro 1-8, vareittain per paa, kuten kuvassa nakyva "Keltainen 3"): pelaajat heittavat kivensa
perakkain numerojarjestyksessa. Oletusjako: 4 pelaajaa = kivet 1-2, 3-4, 5-6, 7-8 (pelaaja 1 = ensimmainen heittaja, pelaaja 4 = viimeinen); 3 pelaajaa = 1-3, 4-6, 7-8; 2 pelaajaa = 1-4, 5-8.
Muu jako annetaan kivimaarina pilkulla erotettuna (esim. --punainen 2,3,3: ensimmainen heittaa kaksi kivea, toinen ja kolmas kolme).

Taulukko pelaajaa kohti, kuten Excelissa; kustakin suureesta Mediaani ja STD:
    Myötä   Liuku   Heitto   Tönäisy        kierroltaan myotapaivaan pyorivat kivet
    Vasta   Liuku   Heitto   Tönäisy        kierroltaan vastapaivaan pyorivat kivet
    Hidastuvuus                              kaikki heitot (ei jaeta suunnan mukaan)
Kierron suunta on CSV:n sarakkeen kierre etumerkki, kuten kuvan paneelin Kierre: + myotapaivaan (Myötä), - vastapaivaan (Vasta). Jos kierre puuttuu, suunta paatellaan sarakkeista
irroitus_delta_cm = (heitto_T_cm - liuku_T_cm) * suunta; jos suuntaa ei saada selville, heitto ei ole Myötä- eika Vasta-luvuissa (huomautus kertoo maaran).
    Liuku, Heitto   kuvan paneelin arvot (cm): liukulinjan ja irroituslinjan osuma kaukopaan T-viivalla MERKISTA (maalimerkki = kipparin harjanpaa), + = kierteesta ulospain (sarakkeet
                    liuku_merkista_cm ja heitto_merkista_cm). Vanhassa CSV:ssa niita ei ole: silloin keskiviivasta (liuku_T_cm, heitto_T_cm; + = heittajan oikealle), ja huomautus kertoo sen
    Tönäisy         nopeus irroituksen jalkeen - nopeus ennen irroitusta (m/s, nopeus_jalkeen_m_s - nopeus_ennen_m_s)
    Hidastuvuus     oletus (--hidastuvuus liuku): liukuvaiheen hidastuvuus ennen irroitusta (m/s2, liun_hidastuvuus_m_s2); --hidastuvuus vapaa: vapaan lennon hidastuvuus irroituksen jalkeen
                    (hidastuvuus_m_s2) kuten paneelissa, eli vain vakaat heitot (vakaa = 1) ja arvot 0 ... 0,25
STD on otoskeskihajonta (n - 1, kuten Excelin STDEV.S; --std-populaatio: jakaja n). Kunkin luvun laskentaan kaytetaan heittoja, joilta kyseinen mittaus on olemassa (CSV:n lopussa n-sarakkeet);
yksittaisesta arvosta ei saa hajontaa (tyhja). Tulos tulostetaan ja tallennetaan CSV:ksi (oletus <syote>_pelaajat.csv; --ei-tallenna: vain tuloste).

Huom: kivinumero tulee seurannasta. Jos paassa on vahemman kuin joukkueen kivet (video alkoi tai loppui kesken paan, tai seuranta ohitti heiton), varoitus kertoo paan; --ohita-puutteelliset
jattaa tallaiset paat pois. Jos analyysi aloitettiin kesken paan, ensimmaisen paan kivinumerot ovat aluksi vaarin (korjaa kivi_nro CSV:ssa tai ohita se).

Funktiot laske_tiedosto() ja csv_tavuina() ovat myos palvelinversion selainnakyman (Netti_01_02, netti_web.py: nappi "Pelaajatilasto") kaytossa."""
import argparse
import csv
import io
import os
import statistics
import sys

KIVIA_PAASSA = 8                       # joukkueen kivet / paa (kuten netti_seuranta.KIVIA_PAASSA: kivi_nro 1-8)
OLETUS_PELAAJIA = 4
RYHMAT = (("Myötä", ("Liuku", "Heitto", "Tönäisy")), ("Vasta", ("Liuku", "Heitto", "Tönäisy")), ("", ("Hidastuvuus",)))       # otsikot kuten Excelissa: ryhma (kierron suunta) ja sen suureet
SARAKKEET = tuple((ryhma, suure) for ryhma, suureet in RYHMAT for suure in suureet)       # 7 suuretta: 0-2 Myötä, 3-5 Vasta, 6 Hidastuvuus; kustakin Mediaani ja STD
DES_TULOSTE = (1, 1, 3, 1, 1, 3, 3)    # desimaalit tulosteessa (cm, cm, m/s, cm, cm, m/s, m/s2)
DES_CSV = (2, 2, 4, 2, 2, 4, 4)        # desimaalit tallennettavassa CSV:ssa
TUNNISTEET = ("joukkue", "pelaaja", "kivet", "heittoja", "n myötä", "n vasta")           # pelaajarivin tunnistetiedot ennen lukuja (n myötä / n vasta = heittoja kierron suunnan mukaan)
HIDASTUVUUS_SARAKE = {"liuku": "liun_hidastuvuus_m_s2", "vapaa": "hidastuvuus_m_s2"}          # Hidastuvuus-sarakkeen lahde
VAPAA_MAX = 0.25                       # vapaan lennon hidastuvuus: suurempi arvo on kohinaa (kuten paneelissa)
PAKOLLISET = ("vari", "kivi_nro", "liuku_T_cm", "heitto_T_cm", "nopeus_ennen_m_s", "nopeus_jalkeen_m_s")


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
    """CSV-rivit (dict). Erotin (, tai ;) tunnistetaan otsikkorivista. ValueError (selkea viesti), jos tiedostoa ei voi lukea tekstina."""
    try:
        with open(polku, newline="", encoding="utf-8-sig") as fh:
            ensimmainen = fh.readline()
            fh.seek(0)
            erotin = ";" if ensimmainen.count(";") > ensimmainen.count(",") else ","
            lukija = csv.DictReader(fh, delimiter=erotin)
            return list(lukija), list(lukija.fieldnames or [])
    except UnicodeDecodeError:
        raise ValueError("tiedostoa ei voi lukea: ei UTF-8-tekstia (tiedoston pitaa olla netti_live.py:n heitto-CSV)") from None
    except (csv.Error, OSError) as e:
        raise ValueError(f"tiedostoa ei voi lukea: {e}") from None


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


def kierron_suunta(r):
    """Heiton kierron suunta: +1 vastapaivaan (Vasta), -1 myotapaivaan (Myötä), 0 tuntematon. Kuten kuvan paneelin Kierre: sarakkeen kierre etumerkki (+ myotapaivaan, - vastapaivaan; tyhja, ellei
    suuntaa tai kierrosmaaraa tunneta). Jos kierre puuttuu, mutta irroitus_delta_cm on, suunta paatellaan siita: irroitus_delta_cm = (heitto_T_cm - liuku_T_cm) * suunta (netti_seuranta.heiton_suureet)."""
    k = luku(r.get("kierre"))
    if k is not None and k != 0:
        return -1 if k > 0 else 1
    d, h, l = luku(r.get("irroitus_delta_cm")), luku(r.get("heitto_T_cm")), luku(r.get("liuku_T_cm"))
    if d is not None and h is not None and l is not None and abs(d) >= 0.2 and abs(h - l) >= 0.2:            # alle 0,2 cm: pyoristys (CSV:ssa 0,1 cm:n tarkkuus) tekee etumerkista epavarman
        return (1 if d > 0 else -1) * (1 if h > l else -1)
    return 0


def mitta_asetukset(kentat, hidastuvuus="liuku"):
    """(asetukset, huomautukset): mista CSV:n sarakkeista Liuku, Heitto ja Hidastuvuus luetaan. Liuku ja Heitto ovat kuvan paneelin arvot: suorien osuma kaukopaan T-viivalla maalimerkista, + = kierteesta
    ulospain (sarakkeet liuku_merkista_cm ja heitto_merkista_cm); vanhassa CSV:ssa niita ei ole, ja silloin kaytetaan keskiviivasta mitattuja arvoja (liuku_T_cm, heitto_T_cm; + = heittajan oikealle).
    hidastuvuus: 'liuku' (liukuvaiheen hidastuvuus, liun_hidastuvuus_m_s2) tai 'vapaa' (vapaan lennon hidastuvuus, hidastuvuus_m_s2, kuten paneelissa). ValueError, jos muu."""
    if hidastuvuus not in HIDASTUVUUS_SARAKE:
        raise ValueError(f"hidastuvuus: {' tai '.join(HIDASTUVUUS_SARAKE)} (annettu {hidastuvuus!r})")
    huom = []
    if "liuku_merkista_cm" in kentat and "heitto_merkista_cm" in kentat:
        liuku, heitto = "liuku_merkista_cm", "heitto_merkista_cm"
    else:
        liuku, heitto = "liuku_T_cm", "heitto_T_cm"
        huom.append("CSV:ssa ei ole merkista laskettuja sarakkeita (liuku_merkista_cm, heitto_merkista_cm; vanha CSV): Liuku ja Heitto ovat keskiviivasta, + = heittajan oikealle")
    hid = HIDASTUVUUS_SARAKE[hidastuvuus]
    if hid not in kentat:
        huom.append(f"CSV:ssa ei ole saraketta {hid}: Hidastuvuus jaa tyhjaksi")
    return dict(liuku=liuku, heitto=heitto, hid=hid, vapaa=(hidastuvuus == "vapaa"), vakaa="vakaa" in kentat), huom


def mitat(r, a):
    """(suunta, [liuku, heitto, tonaisy], hidastuvuus) yhdelta heitolta (a = mitta_asetukset); puuttuva mittaus on None. suunta: +1 Vasta (vastapaivaan), -1 Myötä (myotapaivaan), 0 tuntematon.
    Tönäisy = nopeus irroituksen jalkeen - nopeus ennen. Vapaan lennon hidastuvuus kuten paneelissa: vain vakaasta heitosta ja arvoilla 0 ... 0,25 (muuten ? eli ei lukua)."""
    ennen, jalkeen = luku(r.get("nopeus_ennen_m_s")), luku(r.get("nopeus_jalkeen_m_s"))
    hid = luku(r.get(a["hid"]))
    if a["vapaa"] and hid is not None and not (0.0 <= hid <= VAPAA_MAX and (not a["vakaa"] or luku(r.get("vakaa")) == 1)):
        hid = None
    return kierron_suunta(r), [luku(r.get(a["liuku"])), luku(r.get(a["heitto"])), None if ennen is None or jalkeen is None else jalkeen - ennen], hid


def laske(rivit, kentat, jaot, ohita_puutteelliset=False, populaatio=False, hidastuvuus="liuku"):
    """(tulokset, varoitukset). jaot: {vari: [pelaajan kivimaara, ...]}. tulokset: [dict(joukkue, pelaaja, kivet, heittoja, myota, vasta, n, mediaani, std)] joukkueittain ja pelaajajarjestyksessa;
    heittoja = pelaajan heitot, myota / vasta = niista kierroltaan myotapaivaan / vastapaivaan pyorivat; n, mediaani ja std = tuplat SARAKKEET-jarjestyksessa (Myötä: Liuku, Heitto, Tönäisy; Vasta: sama;
    Hidastuvuus kaikista heitoista). hidastuvuus: 'liuku' tai 'vapaa' (ks. mitta_asetukset)."""
    asetus, varoitukset = mitta_asetukset(kentat, hidastuvuus)
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
        keratty.setdefault((v, pelaaja_kivelle[v][int(nro) - 1]), []).append(mitat(r, asetus))
    tuntematon = sum(1 for lista in keratty.values() for m in lista if m[0] == 0)
    if tuntematon:
        varoitukset.append(f"{tuntematon} heitolta puuttuu kierron suunta (sarake kierre tyhja): ne eivat ole Myötä- eika Vasta-luvuissa, mutta Hidastuvuuteen ne otetaan")
    tulokset = []
    for v, lohkot in jaot.items():
        alku = 1
        for p, n in enumerate(lohkot, 1):
            heitot = keratty.get((v, p), [])
            ryhmat = ([m for m in heitot if m[0] < 0], [m for m in heitot if m[0] > 0])                 # Myötä, Vasta
            sarakkeet = [[m[1][i] for m in ryhma if m[1][i] is not None] for ryhma in ryhmat for i in range(3)] + [[m[2] for m in heitot if m[2] is not None]]
            tulokset.append(dict(joukkue=v, pelaaja=p, kivet=kivet_tekstina(alku, n), heittoja=len(heitot), myota=len(ryhmat[0]), vasta=len(ryhmat[1]), n=tuple(len(x) for x in sarakkeet),
                                 mediaani=tuple(mediaani(x) for x in sarakkeet), std=tuple(hajonta(x, populaatio) for x in sarakkeet)))
            alku += n
    return tulokset, varoitukset


def _muoto(x, des, pilkku=False):
    s = "" if x is None else f"{x:.{des}f}"
    return s.replace(".", ",") if pilkku else s


def otsikkorivit(n_lohko=False):
    """Kolme otsikkoriviä kuten Excelissa: ryhmat (Myötä, Vasta), suureet ja Mediaani / STD (tunnisteiden nimet alimmalla rivilla); ryhma- ja suurenimi on sarakeparin ensimmaisessa sarakkeessa, toinen tyhja.
    n_lohko: lisaa loppuun heittomaarasarakkeet n (montako heittoa kunkin luvun laskentaan kaytettiin)."""
    r1, r2, r3 = [""] * len(TUNNISTEET), [""] * len(TUNNISTEET), list(TUNNISTEET)
    for ryhma, suureet in RYHMAT:
        r1 += [ryhma] + [""] * (2 * len(suureet) - 1)
        for suure in suureet:
            r2 += [suure, ""]
            r3 += ["Mediaani", "STD"]
    if n_lohko:
        r1 += ["n"] + [""] * (len(SARAKKEET) - 1)
        r2 += [""] * len(SARAKKEET)
        r3 += [(ryhma + " " + suure).strip() for ryhma, suure in SARAKKEET]
    return r1, r2, r3


def pelaajarivi(t, des, pilkku=False, n_lohko=False):
    """Pelaajan rivi otsikkorivien mukaan: tunnisteet, Mediaani ja STD kustakin suureesta (des = desimaalit suureittain), halutessa heittomaarat n."""
    solut = [t["joukkue"], t["pelaaja"], t["kivet"], t["heittoja"], t["myota"], t["vasta"]]
    for i in range(len(SARAKKEET)):
        solut += [_muoto(t["mediaani"][i], des[i], pilkku), _muoto(t["std"][i], des[i], pilkku)]
    return solut + (list(t["n"]) if n_lohko else [])


def taulukko(tulokset):
    """Tulosteen rivit: kolme otsikkoriviä (Myötä / Vasta, suureet, Mediaani / STD) ja pelaajarivit; sarakkeet tasattu, ryhma- ja suurenimet keskitetty sarakkeidensa ylle."""
    _, _, r3 = otsikkorivit()
    rivit = [[str(x) if i < len(TUNNISTEET) else (str(x) or "-") for i, x in enumerate(pelaajarivi(t, DES_TULOSTE))] for t in tulokset]
    n = len(r3)
    leveys = [max(len(r3[i]), *(len(r[i]) for r in rivit)) if rivit else len(r3[i]) for i in range(n)]
    alut = [sum(leveys[:i]) + 2 * i for i in range(n)]               # sarakkeen alkukohta rivilla
    kokonaisleveys = alut[-1] + leveys[-1]

    def rivi(solut):
        return "  ".join(s.ljust(leveys[i]) if i < 3 else s.rjust(leveys[i]) for i, s in enumerate(solut))

    def otsikko(kohdat):                                             # [(ensimmainen sarake, sarakkeita, teksti)]: teksti keskitetaan sarakkeidensa ylle
        ulos = [" "] * kokonaisleveys
        for i, lkm, teksti in kohdat:
            alku, loppu = alut[i], alut[i + lkm - 1] + leveys[i + lkm - 1]
            x = alku + max(0, (loppu - alku - len(teksti)) // 2)
            ulos[x:x + len(teksti)] = list(teksti)
        return "".join(ulos).rstrip()
    ryhmat, suureet, i = [], [], len(TUNNISTEET)
    for ryhma, suurenimet in RYHMAT:
        if ryhma:
            ryhmat.append((i, 2 * len(suurenimet), ryhma))
        for suure in suurenimet:
            suureet.append((i, 2, suure))
            i += 2
    return [otsikko(ryhmat), otsikko(suureet), rivi(r3)] + [rivi(r) for r in rivit]


def kirjoita_csv_virtaan(fh, tulokset, fi=False):
    """Kirjoittaa taulukon tiedostomaiseen virtaan (newline=""): kolme otsikkoriviä (Myötä / Vasta, suureet, Mediaani / STD), sitten pelaajat; lopussa n-sarakkeet kertovat, montako heittoa kunkin luvun
    laskentaan kaytettiin. fi: erotin ; ja desimaalipilkku."""
    w = csv.writer(fh, delimiter=";" if fi else ",")
    for rivi in otsikkorivit(n_lohko=True):
        w.writerow(rivi)
    for t in tulokset:
        w.writerow(pelaajarivi(t, DES_CSV, fi, n_lohko=True))


def kirjoita_csv(polku, tulokset, fi=False):
    """Tallentaa taulukon tiedostoon (fi: UTF-8 BOM, jotta suomalainen Excel avaa sen oikein)."""
    with open(polku, "w", newline="", encoding="utf-8-sig" if fi else "utf-8") as fh:
        kirjoita_csv_virtaan(fh, tulokset, fi)


def csv_tavuina(tulokset, fi=False):
    """Taulukko CSV-tavuina (selaimen lataus); fi: UTF-8 BOM + erotin ; + desimaalipilkku."""
    puskuri = io.StringIO(newline="")
    kirjoita_csv_virtaan(puskuri, tulokset, fi)
    return (("\ufeff" if fi else "") + puskuri.getvalue()).encode("utf-8")


def jakohuomautukset(jaot):
    """Huomautukset kivijaoista, joiden kivet eivat ole yhteensa KIVIA_PAASSA (esim. 2,2 tai 3,3,3): jako ei vastaa paata."""
    return [f"joukkueen '{v}' kivijako {','.join(str(n) for n in lohkot)} on yhteensa {sum(lohkot)} kivea (paassa oletetaan {KIVIA_PAASSA})" for v, lohkot in jaot.items() if sum(lohkot) != KIVIA_PAASSA]


def laske_tiedosto(polku, maaraykset=None, oletus=OLETUS_PELAAJIA, ohita_puutteelliset=False, populaatio=False, hidastuvuus="liuku"):
    """Koko tilasto tiedostosta (selainnakyma kayttaa tata; komentorivi tekee saman vaiheittain, koska se kysyy maarat): dict(tiedosto, heittoja, jaot, tulokset, varoitukset).
    maaraykset: {vari: pelaajien maara tai kivijako}; puuttuvalle joukkueelle oletus. hidastuvuus: 'liuku' tai 'vapaa' (ks. mitta_asetukset).
    ValueError (selkea viesti), jos tiedosto, maaraykset tai hidastuvuus eivat kelpaa."""
    rivit, kentat = lue(polku)
    puuttuu = [k for k in PAKOLLISET if k not in kentat]
    if puuttuu:
        raise ValueError(f"CSV:sta puuttuvat sarakkeet: {', '.join(puuttuu)} (tiedoston pitaa olla netti_live.py:n heitto-CSV)")
    rivit = [r for r in rivit if (r.get("vari") or "").strip()]
    if not rivit:
        raise ValueError("CSV:ssa ei ole heittoja")
    jaot = {}
    for v in jarjestys(rivit):
        maaare = (maaraykset or {}).get(v)
        if maaare is None or str(maaare).strip() == "":
            maaare = oletus
        try:
            jaot[v] = jako(maaare)
        except ValueError as e:
            raise ValueError(f"joukkue '{v}': {e}") from None
    tulokset, varoitukset = laske(rivit, kentat, jaot, ohita_puutteelliset, populaatio, hidastuvuus)
    return dict(tiedosto=os.path.basename(polku), heittoja=len(rivit), jaot=jaot, tulokset=tulokset, varoitukset=jakohuomautukset(jaot) + varoitukset)


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
    ap.add_argument("--hidastuvuus", choices=list(HIDASTUVUUS_SARAKE), default="liuku",
                    help="Hidastuvuus-sarakkeen lahde: liuku = liukuvaiheen hidastuvuus ennen irroitusta (oletus), vapaa = vapaan lennon hidastuvuus irroituksen jalkeen (kuten kuvan paneelissa)")
    a = ap.parse_args(argv)
    for virta in (sys.stdout, sys.stderr):
        try:
            virta.reconfigure(errors="replace")                      # Myötä / Tönäisy: konsoli, joka ei osaa ääkkösiä, ei kaada tulostusta
        except (AttributeError, ValueError, OSError):
            pass
    if not os.path.isfile(a.csv):
        print(f"tiedostoa ei loydy: {a.csv}", file=sys.stderr)
        return 2
    try:
        rivit, kentat = lue(a.csv)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 2
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
    for huomautus in jakohuomautukset(jaot):
        print("Huom:", huomautus, file=sys.stderr)
    tulokset, varoitukset = laske(rivit, kentat, jaot, a.ohita_puutteelliset, a.std_populaatio, a.hidastuvuus)
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
