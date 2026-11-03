"""Kahvan kulman merkintasivu (Artifact-HTML): kierre_leikkeet.py:n kuvista satunnainen, enintaan --kuvia kuvan joukko sekoitettuna satunnaiseen jarjestykseen.

    python tools/luo_kahvamerkitsin.py --leikkeet leikkeet --ulos sivu [--heittoja 12] [--kuvia 50] [--siemen 1]

Lukee <leikkeet>/naytteet.json ja kuvakoosteet <leikkeet>/s/*.jpg (python tools/kierre_leikkeet.py ...), poimii satunnaiset heitot (vuorotellen eri videoista, kunnes heittoja on --heittoja tai kuvia
olisi yli --kuvia) ja sekoittaa niiden ruudut kiinteällä siemenellä satunnaiseen jarjestykseen (kahta saman heiton ruutua ei tule peräkkäin, jos se on mahdollista), jolloin merkitsija ei nae heittojen
jarjestysta eika aikaa. Kirjoittaa <ulos>/kahvamerkitsin.html (kuvakoosteet upotettuina data-osoitteina; sivun nayteluettelossa ei ole koneen arvioita), <ulos>/s/ (valitut kuvakoosteet) ja <ulos>/valinta.json (valitut naytteet kaikkine
tietoineen: heitto, aika ja koneen arvio merkintojen vertailuun). Sivun kokoonpano ja tallennus: tools/kahvamerkitsin_pohja.html (Artifact; merkinnat sivun tietokantaan, kokoelma viivat).

Merkinta (tietokannan asiakirja viivat/<id>): tila ok / epavarma / ei_nay / kasi; ok ja epavarma: x1, y1, x2, y2 topdown-leikkeen pikseleina (keskipiste kokonaisluvussa, y alaspain), kulma_deg
(0 = vaakasuora, kasvaa vastapaivaan kuvassa, 0...180) ja pituus_cm."""
import argparse
import base64
import json
import os
import random
import shutil

import kierre_leikkeet as kl

VAKIOT = dict(td_n=kl.TD_N, raw_n=kl.RAW_N, px_cm=kl.TD_PX_CM)
SIVUN_KENTAT = ("id", "nimi", "heitto", "vari", "puoli", "kivi_nro", "f", "t_s", "k", "sprite", "H", "A")
POHJA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kahvamerkitsin_pohja.html")


def heitoittain(naytteet):
    """{(nimi, heitto): [naytteet ajan mukaan]}."""
    r = {}
    for n in naytteet:
        r.setdefault((n["nimi"], n["heitto"]), []).append(n)
    for v in r.values():
        v.sort(key=lambda n: n["k"])
    return r


def poimi_heitot(naytteet, heittoja, kuvia, siemen=1):
    """Satunnaiset heitot: videot vuorotellen (videoiden jarjestys ja heitot videossa arvotaan), kunnes heittoja on `heittoja`; heitto, jonka ruudut ylittaisivat `kuvia`, ohitetaan.
    Palauttaa valittujen heittojen naytteet (heitoittain ajan mukaan)."""
    rnd = random.Random(siemen)
    ryhmat = heitoittain(naytteet)
    videot = {}
    for avain in sorted(ryhmat):
        videot.setdefault(avain[0], []).append(avain)
    for v in videot.values():
        rnd.shuffle(v)
    nimet = sorted(videot)
    rnd.shuffle(nimet)
    valitut, yht = [], 0
    while len(valitut) < heittoja and any(videot.values()):
        for nimi in nimet:
            if len(valitut) >= heittoja or not videot[nimi]:
                continue
            avain = videot[nimi].pop()
            if yht + len(ryhmat[avain]) > kuvia:
                continue
            valitut.append(avain)
            yht += len(ryhmat[avain])
    return [n for avain in valitut for n in ryhmat[avain]]


def sekoita(naytteet, siemen=1, yrityksia=500):
    """Satunnainen jarjestys (kiinteä siemen), jossa kaksi saman heiton ruutua ei ole peräkkäin, jos sellainen jarjestys loytyy yrityksia-kokeilulla."""
    rnd = random.Random(siemen + 1)
    lst = list(naytteet)
    for _ in range(yrityksia):
        rnd.shuffle(lst)
        if all((a["nimi"], a["heitto"]) != (b["nimi"], b["heitto"]) for a, b in zip(lst, lst[1:])):
            break
    return lst


def sivun_luettelo(naytteet):
    return [{k: n[k] for k in SIVUN_KENTAT} for n in naytteet]


def kirjoita_sivu(leikkeet, ulos, heittoja=12, kuvia=50, siemen=1):
    """Kirjoittaa sivun, kuvakoosteet ja valinta.json; palauttaa (jarjestetyt naytteet, sivun polku)."""
    naytteet = json.load(open(os.path.join(leikkeet, "naytteet.json"), encoding="utf-8"))
    valitut = poimi_heitot(naytteet, heittoja, kuvia, siemen)
    if not valitut:
        raise ValueError("ei yhtaan sopivaa heittoa: aja ensin kierre_leikkeet.py")
    jarjestys = sekoita(valitut, siemen)
    os.makedirs(os.path.join(ulos, "s"), exist_ok=True)
    for sp in sorted({n["sprite"] for n in jarjestys}):
        shutil.copyfile(os.path.join(leikkeet, sp), os.path.join(ulos, sp))
    pohja = open(POHJA, encoding="utf-8").read()
    for avain in ("/*__VAKIOT__*/{}", "/*__NAYTTEET__*/[]", "/*__SPRITET__*/{}"):
        if pohja.count(avain) != 1:
            raise ValueError("pohjasta puuttuu paikkamerkki " + avain)
    spritet = {sp: "data:image/jpeg;base64," + base64.b64encode(open(os.path.join(leikkeet, sp), "rb").read()).decode() for sp in sorted({n["sprite"] for n in jarjestys})}
    html = (pohja.replace("/*__VAKIOT__*/{}", json.dumps(VAKIOT)).replace("/*__NAYTTEET__*/[]", json.dumps(sivun_luettelo(jarjestys), ensure_ascii=True, separators=(",", ":")))
            .replace("/*__SPRITET__*/{}", json.dumps(spritet, separators=(",", ":"))))
    polku = os.path.join(ulos, "kahvamerkitsin.html")
    with open(polku, "w", encoding="utf-8") as fh:
        fh.write(html)
    with open(os.path.join(ulos, "valinta.json"), "w", encoding="utf-8") as fh:
        json.dump(jarjestys, fh, ensure_ascii=False, indent=0)
    return jarjestys, polku


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--leikkeet", required=True, help="kierre_leikkeet.py:n --ulos-kansio")
    ap.add_argument("--ulos", required=True)
    ap.add_argument("--heittoja", type=int, default=12)
    ap.add_argument("--kuvia", type=int, default=50, help="kuvien enimmaismaara")
    ap.add_argument("--siemen", type=int, default=1)
    a = ap.parse_args()
    jarjestys, polku = kirjoita_sivu(a.leikkeet, a.ulos, a.heittoja, a.kuvia, a.siemen)
    heitot = {(n["nimi"], n["heitto"]) for n in jarjestys}
    print(f"{len(jarjestys)} kuvaa, {len(heitot)} heittoa ({', '.join(sorted({n['nimi'] for n in jarjestys}))}); sivu {polku}, {os.path.getsize(polku) / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
