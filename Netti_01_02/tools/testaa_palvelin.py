"""Palvelinversion (Netti_01_02) testit: kontin resurssirajat (netti_ymparisto), asetukset ja ajosilmukka (netti_palvelin), selainnakyma (netti_web), sulkusignaali ja tilatiedosto
(netti_live) sekä kayttajan kalibrointikansio (kalibrointi_valinta).

    python tools/testaa_palvelin.py        (kontissa: docker run --rm IMAGE python /app/tools/testaa_palvelin.py)

Ei tarvitse videota, YouTubea eika verkkoa. Poistumiskoodi 1, jos jokin testi epaonnistuu."""
import base64
import collections
import io
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import netti_palvelin as np_  # noqa: E402
import netti_web as nw  # noqa: E402
import netti_ymparisto as ny  # noqa: E402

VIRHEITA = []
KOTI = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def tarkista(nimi, ehto, teksti=""):
    print(("  OK   " if ehto else "  VIRHE ") + nimi + (f": {teksti}" if teksti else ""))
    if not ehto:
        VIRHEITA.append(nimi)


def kirjoita(polku, sisalto="x", mtime=None):
    os.makedirs(os.path.dirname(polku), exist_ok=True)
    mode = "wb" if isinstance(sisalto, bytes) else "w"
    with open(polku, mode) as f:
        f.write(sisalto)
    if mtime is not None:
        os.utime(polku, (mtime, mtime))
    return polku


# ---------------------------------------------------------------------------------------------------------------------
def testaa_ymparisto():
    print("kontin resurssirajat (netti_ymparisto): cgroup v1 / v2")
    with tempfile.TemporaryDirectory() as t:
        v2 = lambda s: kirjoita(os.path.join(t, "cpu.max"), s)                   # noqa: E731
        tarkista("cgroup v2 'max 100000': ei rajaa", ny.cgroup_ytimet(v2("max 100000"), "/ei/ole", "/ei/ole") is None)
        tarkista("cgroup v2 '200000 100000': 2 ydinta", ny.cgroup_ytimet(v2("200000 100000"), "/ei/ole", "/ei/ole") == 2.0)
        tarkista("cgroup v2 '150000 100000': 1,5 ydinta", ny.cgroup_ytimet(v2("150000 100000"), "/ei/ole", "/ei/ole") == 1.5)
        tarkista("cgroup v2 roska: ei kaada", ny.cgroup_ytimet(v2("roskaa"), "/ei/ole", "/ei/ole") is None)
        kv, kj = kirjoita(os.path.join(t, "q"), "300000"), kirjoita(os.path.join(t, "p"), "100000")
        tarkista("cgroup v1 kvootti 300000 / jakso 100000: 3 ydinta", ny.cgroup_ytimet("/ei/ole", kv, kj) == 3.0)
        kirjoita(kv, "-1")
        tarkista("cgroup v1 kvootti -1: ei rajaa", ny.cgroup_ytimet("/ei/ole", kv, kj) is None)
        tarkista("ei cgroup-tiedostoja: ei rajaa", ny.cgroup_ytimet("/ei/ole", "/ei/ole", "/ei/ole") is None)
        m = lambda s: kirjoita(os.path.join(t, "mem"), s)                         # noqa: E731
        tarkista("muistiraja 4 Gt", abs(ny.cgroup_muisti_gt((m(str(4 * 2 ** 30)),)) - 4.0) < 1e-9)
        tarkista("muistiraja 'max': ei rajaa", ny.cgroup_muisti_gt((m("max"),)) is None)
        tarkista("cgroup v1 'ei rajaa' (9223372036854771712): ei rajaa", ny.cgroup_muisti_gt((m("9223372036854771712"),)) is None)
    n = ny.ytimet()
    tarkista("ytimet() >= 1 ja enintaan koneen ydinmaara", 1 <= n <= (os.cpu_count() or n), str(n))
    mg = ny.muisti_gt()
    tarkista("muisti_gt() on luku tai None", mg is None or mg > 0.1, str(mg))
    tarkista("kuvaus() on yksi rivi", "\n" not in ny.kuvaus(), ny.kuvaus())


def testaa_asetukset():
    print("asetukset (netti_palvelin.Asetukset): ymparistomuuttujat -> valitsimet")
    with tempfile.TemporaryDirectory() as t:
        video = os.path.join(t, "video")
        ulos = os.path.join(t, "ulos")
        env = dict(NETTI_DATA=t)
        a = np_.Asetukset(env)
        tarkista("ei lahdetta: video-kansion valvonta", a.laji == "tiedostot" and a.arvo["kokoelma"] and a.jatkuva, str(a.arvo))
        tarkista("oletuskansiot NETTI_DATA:n alla", a.video_kansio == video and a.ulos_kansio == ulos)
        tarkista("oletukset: video paalla, csv paalla, libx264, viive 10, web 8080", a.tallenna_video and a.csv and a.koodekki == "cpu" and a.viive == 10.0 and a.web_portti == 8080)
        a = np_.Asetukset(dict(env, NETTI_LAHDE="hnTTLqHr5x8"))
        tarkista("YouTube-tunnus", (a.laji, a.arvo) == ("youtube", "hnTTLqHr5x8") and not a.jatkuva)
        a = np_.Asetukset(dict(env, NETTI_LAHDE="https://www.youtube.com/watch?v=hnTTLqHr5x8"))
        tarkista("YouTube-osoite", a.laji == "youtube")
        a = np_.Asetukset(dict(env, NETTI_LAHDE="https://youtu.be/hnTTLqHr5x8"))
        tarkista("youtu.be-osoite", a.laji == "youtube")
        a = np_.Asetukset(dict(env, NETTI_LAHDE="https://example.com/live/stream.m3u8"))
        tarkista("muu verkko-osoite: --url", a.laji == "url")
        a = np_.Asetukset(dict(env, NETTI_LAHDE="rtsp://kamera.lan:554/virta"))
        tarkista("rtsp-osoite: --url", a.laji == "url")
        kirjoita(os.path.join(video, "peli.mp4"))
        a = np_.Asetukset(dict(env, NETTI_LAHDE="peli.mp4"))
        tarkista("tiedosto video-kansiossa (suhteellinen nimi)", a.laji == "tiedostot" and not a.arvo["kokoelma"] and a.arvo["kuvio"] == os.path.join(video, "peli.mp4") and not a.jatkuva)
        a = np_.Asetukset(dict(env, NETTI_LAHDE=os.path.join(video, "peli.mp4")))
        tarkista("tiedosto (absoluuttinen polku)", a.laji == "tiedostot" and not a.arvo["kokoelma"])
        a = np_.Asetukset(dict(env, NETTI_LAHDE=video))
        tarkista("kansio: kokoelma ja valvonta paalla", a.arvo["kokoelma"] and a.jatkuva)
        a = np_.Asetukset(dict(env, NETTI_LAHDE=video, NETTI_JATKUVA="0"))
        tarkista("kansio, NETTI_JATKUVA=0: kertaajo", not a.jatkuva)
        a = np_.Asetukset(dict(env, NETTI_LAHDE=os.path.join(video, "*.mp4")))
        tarkista("kuvio *.mp4: kokoelma", a.arvo["kokoelma"])
        kirjoita(os.path.join(video, "11merkkia_x"), "x")                           # 11 merkkia pitka tiedostonimi ei ole YouTube-tunnus
        a = np_.Asetukset(dict(env, NETTI_LAHDE="11merkkia_x"))
        tarkista("11-merkkinen nimi, joka on olemassa oleva tiedosto, on tiedosto", a.laji == "tiedostot")
        for huono in ("ei_ole_olemassa.mp4", "lyhyt"):
            try:
                np_.Asetukset(dict(env, NETTI_LAHDE=huono))
                tarkista(f"tuntematon lahde {huono!r} -> AsetusVirhe", False)
            except np_.AsetusVirhe as e:
                tarkista(f"tuntematon lahde {huono!r} -> AsetusVirhe", "NETTI_LAHDE" in str(e))
        a = np_.Asetukset(dict(env, NETTI_VIIVE="7,5", NETTI_VIDEO="ei", NETTI_CSV="0", NETTI_WEB_PORTTI="0", NETTI_SAIKEET="2", NETTI_NOPEA="false", NETTI_ALKU="12.5"))
        tarkista("arvot: desimaalipilkku, ei/0/false, kokonaisluvut", a.viive == 7.5 and not a.tallenna_video and not a.csv and a.web_portti == 0 and a.saikeet == 2 and not a.nopea and a.alku == 12.5)
        for nimi, arvo in (("NETTI_VIIVE", "paljon"), ("NETTI_VIDEO", "ehka"), ("NETTI_WEB_PORTTI", "70000"), ("NETTI_SAIKEET", "-1"), ("NETTI_LISAVALITSIMET", "--x 'auki")):
            try:
                np_.Asetukset(dict(env, **{nimi: arvo}))
                tarkista(f"{nimi}={arvo!r} -> AsetusVirhe", False)
            except np_.AsetusVirhe as e:
                tarkista(f"{nimi}={arvo!r} -> AsetusVirhe", nimi in str(e), str(e)[:70])
        tyhjat = {n: "" for n in ("NETTI_LAHDE", "NETTI_KANSIO", "NETTI_VIDEO", "NETTI_CSV", "NETTI_JATKUVA", "NETTI_NOPEA", "NETTI_VIIVE", "NETTI_SAIKEET", "NETTI_ALKU", "NETTI_LATAUS", "NETTI_WEB_PORTTI",
                                  "NETTI_LISAVALITSIMET", "NETTI_AANI", "NETTI_KOODEKKI", "NETTI_EI_KIERRETTA", "NETTI_HILJAA", "NETTI_ODOTUS_S")}
        a = np_.Asetukset(dict(env, **tyhjat))
        tarkista("tyhjat muuttujat (Unraid valittaa ne) = ei asetettu: oletukset", a.laji == "tiedostot" and a.jatkuva and a.tallenna_video and a.csv and a.nopea and a.viive == 10.0 and a.saikeet is None
                 and a.web_portti == 8080 and a.lisa == [] and a.aani == "auto" and a.koodekki == "cpu" and not a.ei_kierretta and a.odotus_s == 60.0)
        a = np_.Asetukset(dict(env, NETTI_LISAVALITSIMET="--kiri 0 --yt-formaatti 'best[height<=1080]'"))
        tarkista("NETTI_LISAVALITSIMET jaetaan kuten kuori", a.lisa == ["--kiri", "0", "--yt-formaatti", "best[height<=1080]"], str(a.lisa))


def testaa_komento():
    print("komentorivi (netti_palvelin.rakenna_komento)")
    with tempfile.TemporaryDirectory() as t:
        kirjoita(os.path.join(t, "video", "Matarmaa osa 1 (2).mp4"))
        env = dict(NETTI_DATA=t)
        a = np_.Asetukset(dict(env, NETTI_LAHDE="Matarmaa osa 1 (2).mp4"))
        c = np_.rakenna_komento(a, a.laji, a.arvo["kuvio"], "20260101_120000", "/tmp/tila.json")
        j = " ".join(c)
        tarkista("netti_live.py ajetaan -u:lla ja ilman ikkunaa", c[1] == "-u" and c[2].endswith("netti_live.py") and "--ei-ikkunaa" in c)
        tarkista("tiedosto: --lahde ja --nopea", c[c.index("--lahde") + 1].endswith("Matarmaa osa 1 (2).mp4") and "--nopea" in c)
        tarkista("tulostiedostojen nimet: tunniste + aikaleima", os.path.join(t, "ulos", "kivenseuranta_Matarmaa_osa_1_2_20260101_120000.mp4") in c and os.path.join(t, "ulos", "heitot_Matarmaa_osa_1_2_20260101_120000.csv") in c)
        tarkista("web-valitsimet ja tilatiedosto", "--web 8080" in j and "--web-leveys 1280" in j and "--tila-tiedosto /tmp/tila.json" in j)
        tarkista("tiedostolle ei --lataus eika --kiri", "--lataus" not in c and "--kiri" not in c)
        a = np_.Asetukset(dict(env, NETTI_LAHDE="Matarmaa osa 1 (2).mp4", NETTI_NOPEA="0", NETTI_ALKU="30", NETTI_LOPPU="90", NETTI_VIDEO="0", NETTI_CSV="0", NETTI_WEB_PORTTI="0"))
        c = np_.rakenna_komento(a, a.laji, a.arvo["kuvio"], "x")
        tarkista("NETTI_NOPEA=0: ei --nopea; --alku / --loppu", "--nopea" not in c and c[c.index("--alku") + 1] == "30" and c[c.index("--loppu") + 1] == "90")
        tarkista("ei videota: ei --ulos; ei csv: --ei-csv; ei webia: ei --web", "--ulos" not in c and "--ei-csv" in c and "--csv" not in c and "--web" not in c)
        a = np_.Asetukset(dict(env, NETTI_LAHDE="hnTTLqHr5x8", NETTI_KIRI="0", NETTI_LISAVALITSIMET="--etumatka 5"))
        c = np_.rakenna_komento(a, a.laji, a.arvo, "x", None, 32.0, ["--debug-viive"])
        tarkista("YouTube: --youtube TUNNUS, ei --nopea, --kiri 0", c[c.index("--youtube") + 1] == "hnTTLqHr5x8" and "--nopea" not in c and c[c.index("--kiri") + 1] == "0")
        tarkista("runsaasti muistia: netti_live:n oma --lataus-oletus", "--lataus" not in c)
        tarkista("ylimaaraiset valitsimet perassa (ensin ymparistomuuttujasta, sitten komentorivilta)", c[-3:] == ["--etumatka", "5", "--debug-viive"], str(c[-3:]))
        tarkista("yt-tunniste", np_.tunniste("youtube", "hnTTLqHr5x8") == "yt_hnTTLqHr5x8" and np_.tunniste("youtube", "https://www.youtube.com/watch?v=hnTTLqHr5x8&t=5") == "yt_hnTTLqHr5x8")
        tarkista("yt-tunniste kanavan live-osoitteesta puuttuu: 'yt'", np_.tunniste("youtube", "https://www.youtube.com/@kanava/live") == "yt")
        tarkista("virran tunniste", np_.tunniste("url", "rtsp://x") == "virta")
        for muisti, odotettu in ((None, None), (32.0, None), (10.0, None), (8.0, 15.0), (6.0, 15.0), (4.0, 8.0)):
            tarkista(f"esilataus muistilla {muisti} Gt", np_.valitse_lataus(muisti) == odotettu, str(np_.valitse_lataus(muisti)))
        c = np_.rakenna_komento(a, a.laji, a.arvo, "x", None, 4.0)
        tarkista("vahan muistia: --lataus 8 virralle", c[c.index("--lataus") + 1] == "8")
        a = np_.Asetukset(dict(env, NETTI_LAHDE="hnTTLqHr5x8", NETTI_LATAUS="20"))
        c = np_.rakenna_komento(a, a.laji, a.arvo, "x", None, 4.0)
        tarkista("NETTI_LATAUS ohittaa muistiperusteen", c[c.index("--lataus") + 1] == "20")


def testaa_tiedostot_ja_muisti():
    print("valvottavat tiedostot ja valvonnan muisti (kasitellyt.json)")
    with tempfile.TemporaryDirectory() as t:
        nyt = time.time()
        v = os.path.join(t, "video")
        kirjoita(os.path.join(v, "a.mp4"), "aaa", nyt - 100)
        kirjoita(os.path.join(v, "B.MKV"), "bbb", nyt - 100)
        kirjoita(os.path.join(v, "tuore.mp4"), "ccc", nyt - 5)
        kirjoita(os.path.join(v, "tyhja.mp4"), "", nyt - 100)
        kirjoita(os.path.join(v, ".piilo.mp4"), "ddd", nyt - 100)
        kirjoita(os.path.join(v, "muistio.txt"), "eee", nyt - 100)
        kirjoita(os.path.join(v, "ali", "c.mp4"), "fff", nyt - 100)
        arvo = dict(kuvio=os.path.join(v, "*"), kokoelma=True, kansio=True)
        nimet = [os.path.basename(p) for p in np_.tiedostot(arvo, nyt, 30.0)]
        tarkista("kokoelma: videopaatteiset, vakaat, ei tyhjia / piilo / alikansioita", nimet == ["B.MKV", "a.mp4"], str(nimet))
        nimet = [os.path.basename(p) for p in np_.tiedostot(arvo, nyt, 0.0)]
        tarkista("vakaus 0 s: tuore tiedosto mukaan", "tuore.mp4" in nimet)
        kirjoita(os.path.join(v, "x[1].mp4"), "g", nyt - 100)
        yks = dict(kuvio=os.path.join(v, "x[1].mp4"), kokoelma=False, kansio=False)
        tarkista("yksittainen tiedosto, jonka nimessa on kuvion merkkeja", np_.tiedostot(yks) == [os.path.join(v, "x[1].mp4")])
        k = np_.Kasitellyt(os.path.join(t, "ulos", "kasitellyt.json"))
        p = os.path.join(v, "a.mp4")
        tarkista("alussa ei mitaan valmista", not k.valmis("tiedostot", p))
        k.kirjaa("tiedostot", p, True)
        tarkista("onnistunut ajo kirjataan", k.valmis("tiedostot", p))
        tarkista("muisti sailyy tiedostossa (uusi olio)", np_.Kasitellyt(k.polku).valmis("tiedostot", p))
        kirjoita(p, "muuttunut sisalto", nyt - 50)
        tarkista("tiedosto vaihtui (koko / aika): ajetaan uudelleen", not np_.Kasitellyt(k.polku).valmis("tiedostot", p))
        q = os.path.join(v, "B.MKV")
        for i in range(1, np_.YRITYKSIA_MAX + 1):
            k.kirjaa("tiedostot", q, False)
            tarkista(f"epaonnistunut yritys {i}/{np_.YRITYKSIA_MAX}: " + ("ohitetaan" if i >= np_.YRITYKSIA_MAX else "yritetaan uudelleen"), k.valmis("tiedostot", q) == (i >= np_.YRITYKSIA_MAX))
        k.kirjaa("youtube", "abcdefghijk", True)
        tarkista("YouTube-lahde kirjataan tunnuksella", np_.Kasitellyt(k.polku).valmis("youtube", "abcdefghijk") and not k.valmis("youtube", "toinen_id_11"))
        kirjoita(k.polku, "{ rikki")
        tarkista("rikkinainen muistitiedosto: aloitetaan tyhjasta", not np_.Kasitellyt(k.polku).data)


class EsiPysaytys:
    """Testi-Pysaytys: nuku() ei odota oikeasti; pysaytys pyydetaan n:nnella nukkumisella."""

    def __init__(self, pysayta_nukkumisella=None):
        self.tapahtuma = threading.Event()
        self.signaali = None
        self.nukkumiset = []
        self.raja = pysayta_nukkumisella

    def pyydetty(self):
        return self.tapahtuma.is_set()

    def nuku(self, s):
        self.nukkumiset.append(s)
        if self.raja is not None and len(self.nukkumiset) >= self.raja:
            self.tapahtuma.set()
        return self.tapahtuma.is_set()


def testaa_silmukka():
    print("ajosilmukka (netti_palvelin.palvele): kertaajo ja valvonta, lapsiprosessi korvattu")
    with tempfile.TemporaryDirectory() as t:
        nyt = time.time()
        v = os.path.join(t, "video")
        for n in ("a.mp4", "b.mp4"):
            kirjoita(os.path.join(v, n), n, nyt - 100)
        ajot = []

        def ajuri_ok(cmd, env=None, cwd=None):
            ajot.append(cmd)
            return 0

        a = np_.Asetukset(dict(NETTI_DATA=t, NETTI_LAHDE=v, NETTI_JATKUVA="0"))
        rc = np_.palvele(a, EsiPysaytys(), ajuri_ok, 32.0, lambda: nyt)
        tarkista("kertaajo kansiolle: molemmat videot, paluukoodi 0", rc == 0 and [c[c.index("--lahde") + 1].rsplit("/", 1)[1] for c in ajot] == ["a.mp4", "b.mp4"], str([c[c.index("--lahde") + 1][-8:] for c in ajot]))
        tarkista("ulos- ja tyokansiot luotu", os.path.isdir(a.ulos_kansio) and os.path.isdir(a.tyo_kansio))
        ajot.clear()

        def ajuri_vika(cmd, env=None, cwd=None):
            ajot.append(cmd)
            return 7 if cmd[cmd.index("--lahde") + 1].endswith("b.mp4") else 0

        rc = np_.palvele(a, EsiPysaytys(), ajuri_vika, 32.0, lambda: nyt)
        tarkista("kertaajo: yhden videon virhe ei estä toista, paluukoodi = virhekoodi", rc == 7 and len(ajot) == 2)
        ajot.clear()
        a = np_.Asetukset(dict(NETTI_DATA=t, NETTI_LAHDE=os.path.join(t, "tyhja", "*.mp4"), NETTI_JATKUVA="0"))
        tarkista("kertaajo ilman videoita: paluukoodi 3", np_.palvele(a, EsiPysaytys(), ajuri_ok, 32.0, lambda: nyt) == 3)

        # valvonta: kaksi kierrosta, sitten pysaytys
        a = np_.Asetukset(dict(NETTI_DATA=t, NETTI_LAHDE=v, NETTI_ODOTUS_S="5", NETTI_VAKAUS_S="30"))
        tarkista("kansio-lahteella valvonta on oletuksena paalla", a.jatkuva)
        ps = EsiPysaytys(pysayta_nukkumisella=2)
        rc = np_.palvele(a, ps, ajuri_vika, 32.0, lambda: nyt)
        tarkista("valvonta: pysaytys kesken odotuksen -> paluukoodi 0", rc == 0)
        a_ajot = [c[c.index("--lahde") + 1].rsplit("/", 1)[1] for c in ajot]
        tarkista("valvonta: a.mp4 valmis kerran, b.mp4 (epaonnistuu) yritetaan uudelleen", a_ajot.count("a.mp4") == 1 and a_ajot.count("b.mp4") >= 2, str(a_ajot))
        tarkista("valvonta: odotusaika asetuksen mukainen", all(s == 5.0 for s in ps.nukkumiset), str(ps.nukkumiset))
        ajot.clear()
        ps = EsiPysaytys(pysayta_nukkumisella=6)
        np_.palvele(a, ps, ajuri_vika, 32.0, lambda: nyt)
        k = np_.Kasitellyt(a.kasitellyt_polku)
        b_ajot = [c[c.index("--lahde") + 1].endswith("b.mp4") for c in ajot].count(True)
        tarkista("valvonta (uusi kaynnistys): b.mp4 saa kolmannen ja viimeisen yrityksen, sen jalkeen se jaa pois", b_ajot == 1 and k.valmis("tiedostot", os.path.join(v, "b.mp4")), str(b_ajot))
        tarkista("valvonta: a.mp4 ei aja uudelleen uudessa kaynnistyksessa", [c[c.index("--lahde") + 1].endswith("a.mp4") for c in ajot].count(True) == 0)
        ajot.clear()
        np_.palvele(a, EsiPysaytys(pysayta_nukkumisella=3), ajuri_vika, 32.0, lambda: nyt)
        tarkista("valvonta: yritysrajan ylittanytta b.mp4:aa ei enaa ajeta", ajot == [])
        kirjoita(os.path.join(v, "c.mp4"), "c", nyt - 100)
        ajot.clear()
        np_.palvele(a, EsiPysaytys(pysayta_nukkumisella=1), ajuri_ok, 32.0, lambda: nyt)
        tarkista("valvonta: uusi video c.mp4 havaitaan ja ajetaan", [c[c.index("--lahde") + 1].rsplit("/", 1)[1] for c in ajot] == ["c.mp4"])

        # YouTube-lahteen valvonta: VOD kirjataan valmiiksi, live ajetaan uudelleen
        ulos = os.path.join(t, "ulos2")
        a = np_.Asetukset(dict(NETTI_DATA=t, NETTI_ULOS_KANSIO=ulos, NETTI_LAHDE="hnTTLqHr5x8", NETTI_JATKUVA="1", NETTI_ODOTUS_S="60"))
        tila_polku = os.path.join(a.tyo_kansio, "ajon_tila.json")
        ajot.clear()

        def ajuri_vod(cmd, env=None, cwd=None):
            ajot.append(cmd)
            kirjoita(cmd[cmd.index("--tila-tiedosto") + 1], json.dumps(dict(live=False)))
            return 0

        ps = EsiPysaytys(pysayta_nukkumisella=2)
        np_.palvele(a, ps, ajuri_vod, 32.0, lambda: nyt)
        tarkista("YouTube VOD valmistui: ei ajeta uudelleen valvonnassa", len(ajot) == 1 and np_.Kasitellyt(a.kasitellyt_polku).valmis("youtube", "hnTTLqHr5x8"), str(len(ajot)))
        ajot.clear()
        os.remove(a.kasitellyt_polku)

        def ajuri_live(cmd, env=None, cwd=None):
            ajot.append(cmd)
            kirjoita(cmd[cmd.index("--tila-tiedosto") + 1], json.dumps(dict(live=True)))
            return 0

        np_.palvele(a, EsiPysaytys(pysayta_nukkumisella=3), ajuri_live, 32.0, lambda: nyt)
        tarkista("live-lahetyksen loputtua yritetaan uudelleen (seuraava lahetys)", len(ajot) == 3, str(len(ajot)))
        ajot.clear()
        ps = EsiPysaytys(pysayta_nukkumisella=4)

        def ajuri_ei_saatavilla(cmd, env=None, cwd=None):
            ajot.append(cmd)
            return 1                                                              # nopea epaonnistuminen: lahetys ei ole alkanut

        np_.palvele(a, ps, ajuri_ei_saatavilla, 32.0, lambda: nyt)
        tarkista("nopeat epaonnistumiset pidentavat odotusta (60, 120, 240, 480 s)", ps.nukkumiset == [60.0, 120.0, 240.0, 480.0], str(ps.nukkumiset))
        ps = EsiPysaytys(pysayta_nukkumisella=8)
        np_.palvele(a, ps, ajuri_ei_saatavilla, 32.0, lambda: nyt)
        tarkista("odotus ei kasva yli 15 min", max(ps.nukkumiset) <= np_.ODOTUS_MAX_S, str(ps.nukkumiset))

        # kansioiden oikeudet: selkea virhe heti
        tiedosto_kansion_tilalla = kirjoita(os.path.join(t, "tiedosto"), "x")
        a_huono = np_.Asetukset(dict(NETTI_DATA=t, NETTI_ULOS_KANSIO=os.path.join(tiedosto_kansion_tilalla, "ulos"), NETTI_LAHDE=v, NETTI_JATKUVA="0"))
        ok, syy = np_.valmistele_kansiot(a_huono)
        tarkista("kansiota ei voi luoda (polun varrella on tiedosto): virheilmoitus", not ok and "ulos" in syy and "oikeudet" in syy, syy[:90])
        ajot.clear()
        tarkista("palvele: kansiovirhe -> paluukoodi 2, ajoa ei kaynnisteta", np_.palvele(a_huono, EsiPysaytys(), ajuri_ok, 32.0, lambda: nyt) == 2 and ajot == [])
        if os.getuid() != 0:                                                      # rootilla kirjoitusoikeus ei koskaan puutu
            vain_luku = os.path.join(t, "vain_luku")
            os.makedirs(vain_luku)
            os.chmod(vain_luku, 0o555)
            a_huono = np_.Asetukset(dict(NETTI_DATA=t, NETTI_ULOS_KANSIO=vain_luku, NETTI_LAHDE=v, NETTI_JATKUVA="0"))
            ok, syy = np_.valmistele_kansiot(a_huono)
            tarkista("kansioon ei voi kirjoittaa: virheilmoitus", not ok and "kirjoittaa" in syy, syy[:90])
            os.chmod(vain_luku, 0o755)

        # signaali kesken ajon: pysaytetty ajo ei kirjaudu valmiiksi
        a = np_.Asetukset(dict(NETTI_DATA=t, NETTI_LAHDE=v, NETTI_JATKUVA="1"))
        os.path.exists(a.kasitellyt_polku) and os.remove(a.kasitellyt_polku)
        ps = EsiPysaytys()

        def ajuri_pysaytetty(cmd, env=None, cwd=None):
            ps.tapahtuma.set()
            return 0

        rc = np_.palvele(a, ps, ajuri_pysaytetty, 32.0, lambda: nyt)
        tarkista("pysaytys kesken ajon: paluukoodi 0, ajoa ei kirjata valmiiksi", rc == 0 and not np_.Kasitellyt(a.kasitellyt_polku).valmis("tiedostot", os.path.join(v, "a.mp4")))


def testaa_lapsiprosessi():
    print("lapsiprosessi ja signaalit (netti_palvelin.aja_lapsi)")
    with tempfile.TemporaryDirectory() as t:
        merkki = os.path.join(t, "viimeistelty.txt")
        koodi = ("import signal, sys, time\n"
                 f"def k(s, f):\n    open({merkki!r}, 'w').write('viimeistelty'); sys.exit(0)\n"
                 "signal.signal(signal.SIGTERM, k)\nsignal.signal(signal.SIGINT, k)\nprint('valmis', flush=True)\n"
                 "time.sleep(60)\n")
        ps = np_.Pysaytys()
        ps.tapahtuma.clear()
        t0 = time.monotonic()
        threading.Timer(1.0, lambda: (setattr(ps, "signaali", signal.SIGTERM), ps.tapahtuma.set())).start()
        rc = np_.aja_lapsi([sys.executable, "-c", koodi], ps, sulku_s=20.0)
        tarkista("lopetuspyynto valitetaan lapselle ja se viimeistelee", rc == 0 and os.path.exists(merkki) and time.monotonic() - t0 < 10.0, f"rc {rc}, {time.monotonic() - t0:.1f} s")
        koodi2 = "import signal, time\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\nprint('ei lopu', flush=True)\ntime.sleep(60)\n"
        ps2 = np_.Pysaytys()
        threading.Timer(0.5, lambda: (setattr(ps2, "signaali", signal.SIGTERM), ps2.tapahtuma.set())).start()
        t0 = time.monotonic()
        rc = np_.aja_lapsi([sys.executable, "-c", koodi2], ps2, sulku_s=1.5)
        tarkista("lapsi, joka ei lopu: tapetaan sulkuajan jalkeen", rc != 0 and 1.5 <= time.monotonic() - t0 < 12.0, f"rc {rc}, {time.monotonic() - t0:.1f} s")
        rc = np_.aja_lapsi([sys.executable, "-c", "import sys; sys.exit(5)"], np_.Pysaytys())
        tarkista("lapsen paluukoodi valitetaan", rc == 5)
        tarkista("lue_tila: puuttuva / rikkinainen tiedosto -> None", np_.lue_tila(os.path.join(t, "ei_ole")) is None and np_.lue_tila(kirjoita(os.path.join(t, "r.json"), "{")) is None)

        # koko kaynnistaja: SIGTERM palvelimelle -> lapsi viimeistelee -> paluukoodi 0
        v = os.path.join(t, "data", "video")
        kirjoita(os.path.join(v, "jono.mp4"), "x", time.time() - 100)
        env = dict(os.environ, NETTI_DATA=os.path.join(t, "data"), NETTI_LAHDE=v, NETTI_JATKUVA="1", NETTI_ODOTUS_S="30", NETTI_WEB_PORTTI="0")
        p = subprocess.Popen([sys.executable, os.path.join(KOTI, "netti_palvelin.py"), "--nayta"], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        ulos, _ = p.communicate(timeout=60)
        tarkista("--nayta tulostaa asetukset ja komennon eika aja mitaan", p.returncode == 0 and "komento:" in ulos and "netti_live.py" in ulos and "--ei-ikkunaa" in ulos, ulos.strip().splitlines()[-1][:100] if ulos.strip() else "")
        env["NETTI_VIIVE"] = "paljon"
        p = subprocess.run([sys.executable, os.path.join(KOTI, "netti_palvelin.py")], env=env, capture_output=True, text=True, timeout=60)
        tarkista("asetusvirhe: paluukoodi 2 ja selkea viesti", p.returncode == 2 and "NETTI_VIIVE" in p.stderr, p.stderr.strip()[-100:])


def pyyda(portti, polku, otsakkeet=None, metodi="GET"):
    r = urllib.request.Request(f"http://127.0.0.1:{portti}{polku}", headers=otsakkeet or {}, method=metodi)
    try:
        with urllib.request.urlopen(r, timeout=10) as x:
            return x.status, dict(x.headers), x.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def testaa_web():
    print("selainnakyma (netti_web.WebNaytto)")
    with tempfile.TemporaryDirectory() as t:
        kirjoita(os.path.join(t, "heitot_1.csv"), "a,b\n1,2\n")
        kirjoita(os.path.join(t, "k.mp4"), bytes(range(256)) * 4000, time.time() - 5)
        kirjoita(os.path.join(t, "salainen.key"), "x")
        kirjoita(os.path.join(t, ".piilo.csv"), "x")
        os.makedirs(os.path.join(t, "ali"))
        kirjoita(os.path.join(t, "ali", "x.csv"), "x")
        ulkona = kirjoita(os.path.join(os.path.dirname(t), "ulkona_" + os.path.basename(t) + ".csv"), "salaisuus")
        try:
            os.symlink(ulkona, os.path.join(t, "linkki.csv"))
        except OSError:
            pass
        rivit = nw.asenna_lokipuskuri()
        print("rivi lokiin", flush=True)
        w = nw.WebNaytto(0, t, leveys=640, fps=25, laatu=70, tila=lambda: dict(lahde="testi"), loki_rivit=rivit)
        p = w.portti
        try:
            s, h, b = pyyda(p, "/")
            tarkista("sivu", s == 200 and b"<img" in b and "text/html" in h["Content-Type"])
            s, h, b = pyyda(p, "/kuva.jpg")
            tarkista("kuva.jpg heti (odotuskuva)", s == 200 and h["Content-Type"] == "image/jpeg" and b[:2] == b"\xff\xd8")
            kuva = np.zeros((1080, 1920, 3), np.uint8)
            kuva[:, :, 1] = 200
            lopeta, valmis = threading.Event(), threading.Event()
            silmukka = threading.Thread(target=lambda: w.paasilmukka(lopeta, valmis.is_set), daemon=True)
            silmukka.start()
            w.anna(kuva)
            for _ in range(100):
                if w.n_nayt >= 1:
                    break
                time.sleep(0.02)
            s, h, b = pyyda(p, "/kuva.jpg")
            tarkista("annettu kuva tulee selaimeen, pienennettyna asetettuun leveyteen", s == 200 and w.n_nayt == 1)
            import cv2
            dekoodattu = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
            tarkista("kuvan leveys 640 ja sisalto (vihrea)", dekoodattu.shape[1] == 640 and abs(int(dekoodattu[10, 10, 1]) - 200) < 10, str(dekoodattu.shape))
            j1, id1 = w.jpeg()
            j2, id2 = w.jpeg()
            tarkista("JPEG pakataan kerran kuvaa kohden (katsojat jakavat)", j1 is j2 and id1 == id2)
            sock = socket.create_connection(("127.0.0.1", p), timeout=10)
            sock.sendall(b"GET /video.mjpg HTTP/1.1\r\nHost: x\r\n\r\n")
            data, t0 = b"", time.time()

            def tuota():
                for _ in range(30):
                    w.anna(kuva.copy())
                    time.sleep(0.05)
            tuottaja = threading.Thread(target=tuota)
            tuottaja.start()
            while data.count(b"--kehys") < 3 and time.time() - t0 < 10:
                data += sock.recv(65536)
            tarkista("MJPEG-virta: multipart-otsake ja vahintaan 3 kehysta", b"multipart/x-mixed-replace; boundary=kehys" in data and data.count(b"--kehys") >= 3 and b"Content-Type: image/jpeg" in data)
            tarkista("katsoja lasketaan", w.tila()["katsojia"] == 1)
            sock.close()
            tuottaja.join()
            for _ in range(100):
                if w.tila()["katsojia"] == 0:
                    break
                time.sleep(0.05)
            tarkista("katsoja poistuu laskurista yhteyden katketessa", w.tila()["katsojia"] == 0)
            valmis.set()
            silmukka.join(3)
            tarkista("paasilmukka paattyy, kun valmis() on tosi", not silmukka.is_alive())
            tila = json.loads(pyyda(p, "/tila")[2])
            tarkista("/tila: kuvia, kaynnissa ja lahteen lisatieto", tila["kuvia"] >= 4 and tila["kaynnissa"] and tila["lahde"] == "testi" and tila["viim_kuva_ika_s"] is not None, str(tila))
            tarkista("/loki: konsolirivit mukana", "rivi lokiin" in pyyda(p, "/loki")[2].decode())
            tiedostot = json.loads(pyyda(p, "/tiedostot.json")[2])
            tarkista("/tiedostot.json: vain tulostiedostot (mp4, csv), ei salaisia, piilo-, alikansio- eika linkkitiedostoja", sorted(x["nimi"] for x in tiedostot) == ["heitot_1.csv", "k.mp4"], str([x["nimi"] for x in tiedostot]))
            s, h, b = pyyda(p, "/tiedostot/")
            tarkista("/tiedostot/ HTML-lista", s == 200 and b"heitot_1.csv" in b and b"salainen" not in b)
            s, h, b = pyyda(p, "/tiedostot/heitot_1.csv")
            tarkista("CSV latautuu liitteena", s == 200 and b == b"a,b\n1,2\n" and "attachment" in h["Content-Disposition"] and h["Content-Type"].startswith("text/csv"))
            s, h, b = pyyda(p, "/tiedostot/k.mp4")
            tarkista("mp4 toistuu selaimessa (inline, video/mp4, Accept-Ranges)", s == 200 and h["Content-Type"] == "video/mp4" and h["Content-Disposition"].startswith("inline") and h["Accept-Ranges"] == "bytes" and len(b) == 1024000)
            s, h, b = pyyda(p, "/tiedostot/k.mp4", {"Range": "bytes=100-199"})
            tarkista("Range: 206 ja oikeat tavut", s == 206 and h["Content-Range"] == "bytes 100-199/1024000" and b == (bytes(range(256)) * 4000)[100:200])
            s, h, b = pyyda(p, "/tiedostot/k.mp4", {"Range": "bytes=-10"})
            tarkista("Range: viimeiset 10 tavua", s == 206 and len(b) == 10 and h["Content-Range"] == "bytes 1023990-1023999/1024000")
            s, h, b = pyyda(p, "/tiedostot/k.mp4", {"Range": "bytes=1024000-"})
            tarkista("Range: tiedoston ohi -> 416", s == 416)
            s, h, b = pyyda(p, "/tiedostot/k.mp4", {"Range": "bytes=a-b"})
            tarkista("Range: roska -> 416", s == 416)
            s, h, b = pyyda(p, "/tiedostot/k.mp4", metodi="HEAD")
            tarkista("HEAD: otsakkeet ilman sisaltoa", s == 200 and b == b"" and h["Content-Length"] == "1024000")
            for polku in ("/tiedostot/..%2f" + os.path.basename(ulkona), "/tiedostot/%2e%2e%2f%2e%2e%2fetc%2fpasswd", "/tiedostot/..\\x.csv", "/tiedostot/.piilo.csv", "/tiedostot/salainen.key",
                          "/tiedostot/ali/x.csv", "/tiedostot/linkki.csv", "/tiedostot/k.mp4%00", "/tiedostot/ei_ole.csv", "/etc/passwd", "/../etc/passwd"):
                tarkista(f"suojaus: {polku} -> 404", pyyda(p, polku)[0] == 404)
            tarkista("POST ei tuettu", pyyda(p, "/tila", metodi="POST")[0] in (501, 405))
        finally:
            w.sulje()
        tarkista("sulkemisen jalkeen portti ei vastaa", _portti_kiinni(p))
        os.remove(ulkona)

        # salasana
        os.environ["NETTI_WEB_KAYTTAJA"], os.environ["NETTI_WEB_SALASANA"] = "kaytt", "sala sana:1"
        try:
            w2 = nw.WebNaytto(0, t)
        finally:
            del os.environ["NETTI_WEB_KAYTTAJA"], os.environ["NETTI_WEB_SALASANA"]
        try:
            tunn = lambda k, s: {"Authorization": "Basic " + base64.b64encode(f"{k}:{s}".encode()).decode()}   # noqa: E731
            s, h, _ = pyyda(w2.portti, "/tila")
            tarkista("salasana asetettu: ilman tunnuksia 401 ja WWW-Authenticate", s == 401 and "Basic" in h.get("WWW-Authenticate", ""))
            tarkista("vaara salasana 401", pyyda(w2.portti, "/tila", tunn("kaytt", "vaara"))[0] == 401)
            tarkista("oikea salasana (valilyonti ja kaksoispiste mukana) 200", pyyda(w2.portti, "/tila", tunn("kaytt", "sala sana:1"))[0] == 200)
            tarkista("tiedostot suojattu samoin", pyyda(w2.portti, "/tiedostot.json")[0] == 401 and pyyda(w2.portti, "/video.mjpg")[0] == 401)
        finally:
            w2.sulje()

        # portti varattu
        raja = socket.socket()
        raja.bind(("0.0.0.0", 0))
        raja.listen(1)
        try:
            nw.WebNaytto(raja.getsockname()[1], t)
            tarkista("varattu portti -> SystemExit selkealla viestilla", False)
        except SystemExit as e:
            tarkista("varattu portti -> SystemExit selkealla viestilla", "Web-porttia" in str(e), str(e)[:60])
        finally:
            raja.close()
        # ilmoituskuva ennen ensimmaista kuvaa
        w3 = nw.WebNaytto(0, None)
        try:
            lopeta = threading.Event()
            ilm = nw.odotuskuva("Puskuroidaan 3.0 s")
            kerrat = []
            silmukka = threading.Thread(target=lambda: w3.paasilmukka(lopeta, lambda: len(kerrat) >= 3, lambda: (kerrat.append(1), ilm)[1]), daemon=True)
            silmukka.start()
            silmukka.join(5)
            tarkista("puskurointi-ilmoitus julkaistaan, mutta ei lasketa nayttokuvaksi", len(kerrat) >= 3 and w3.n_nayt == 0 and w3.tila()["viim_kuva_ika_s"] is not None)
            tarkista("ei ulos-kansiota: tiedostolista tyhja", w3.tiedostot() == [] and pyyda(w3.portti, "/tiedostot.json")[2] == b"[]")
        finally:
            lopeta.set()
            w3.sulje()

    print("lokipuskuri (netti_web.LokiPuskuri)")
    alku = io.StringIO()
    rivit = collections.deque(maxlen=5)
    lp = nw.LokiPuskuri(alku, rivit)
    lp.write("yksi\nkaksi\nkol")
    lp.write("mas\nedistys 10 %\redistys 90 %\n\n")
    tarkista("rivit puskuriin, keskenerainen rivi odottaa, \\r:n jalkeen vain viimeinen versio, tyhjat pois", list(rivit) == ["yksi", "kaksi", "kolmas", "edistys 90 %"], str(list(rivit)))
    tarkista("alkuperainen virta saa kaiken sellaisenaan", alku.getvalue() == "yksi\nkaksi\nkolmas\nedistys 10 %\redistys 90 %\n\n")
    for i in range(10):
        lp.write(f"r{i}\n")
    tarkista("puskuri pyorii (enintaan maxlen)", len(rivit) == 5 and rivit[-1] == "r9")
    tarkista("muut attribuutit valitetaan alkuperaiselle (fileno-tyyppiset kutsut toimivat)", lp.getvalue().startswith("yksi") and lp.closed is False)


def _portti_kiinni(portti):
    try:
        socket.create_connection(("127.0.0.1", portti), timeout=1).close()
        return False
    except OSError:
        return True


def testaa_netti_live_palvelinosat():
    print("netti_live: sulkusignaali, tilatiedosto, valitsimet, kalibrointikansio, keksit")
    import netti_aani as na
    import kalibrointi_valinta as kv
    import netti_live as nl
    vanha = signal.getsignal(signal.SIGTERM)
    try:
        nl.asenna_sulkusignaali()
        try:
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(1.0)
            tarkista("SIGTERM muuttuu KeyboardInterruptiksi (ajo viimeistellaan)", False)
        except KeyboardInterrupt:
            tarkista("SIGTERM muuttuu KeyboardInterruptiksi (ajo viimeistellaan)", True)
    finally:
        signal.signal(signal.SIGTERM, vanha)
    with tempfile.TemporaryDirectory() as t:
        p = os.path.join(t, "alikansio_ei_ole", "tila.json")
        nl.kirjoita_tila(os.path.join(t, "tila.json"), lahde="x ä", live=True, fps=30.0)
        d = json.load(open(os.path.join(t, "tila.json"), encoding="utf-8"))
        tarkista("kirjoita_tila: JSON (myos skandit), ei tilapaistiedostoa jaljelle", d == dict(lahde="x ä", live=True, fps=30.0) and not os.path.exists(os.path.join(t, "tila.json.tmp")))
        nl.kirjoita_tila(None, a=1)
        nl.kirjoita_tila(p, a=1)
        tarkista("kirjoita_tila: puuttuva polku / kansio ei kaada", True)

        # kalibrointikansiot
        lisa = os.path.join(t, "kalibroinnit")
        for nimi in ("oma1", "oma2", "puutteellinen"):
            for tied in kv.TARVITTAVAT if nimi != "puutteellinen" else kv.TARVITTAVAT[:2]:
                kirjoita(os.path.join(lisa, nimi, tied), "x")
        kirjoita(os.path.join(lisa, "tiedosto_ei_kansio"), "x")
        os.environ[kv.LISAKANSIO_ENV] = lisa
        try:
            e = kv.ehdokkaat()
            oma = [os.path.basename(k) for k in e if os.path.dirname(k) == os.path.abspath(lisa)]
            tarkista("ehdokkaat: oman kansion alikansiot, joissa kaikki tarvittavat tiedostot", oma == ["oma1", "oma2"], str(oma))
            tarkista("ehdokkaat: sovelluksen omat kansiot ensin", [os.path.basename(k) for k in e][:3] == ["tulokset", "tulokset_nettivideo2", "tulokset_nettivideo3"], str([os.path.basename(k) for k in e]))
            tarkista("lisakansio() lukee ymparistomuuttujan", kv.lisakansio() == os.path.abspath(lisa))
        finally:
            del os.environ[kv.LISAKANSIO_ENV]
        tarkista("ilman NETTI_KALIBROINNIT: vain omat", kv.lisakansio() is None and all(os.path.dirname(k) == os.path.abspath(kv.KOHDE) for k in kv.ehdokkaat()))

        # keksit
        os.environ.pop("YTDLP_KEKSIT", None)
        tarkista("keksitiedosto: ei asetettu -> None", na.keksitiedosto() is None)
        alku = kirjoita(os.path.join(t, "cookies.txt"), "# Netscape HTTP Cookie File\n.youtube.com\tTRUE\t/\tTRUE\t0\tA\tb\n")
        os.environ["YTDLP_KEKSIT"] = alku
        try:
            k = na.keksitiedosto()
            tarkista("keksitiedosto: kopio tilapaiseen paikkaan (alkuperaista ei kosketa)", k and k != alku and open(k).read() == open(alku).read())
            os.remove(k)
            os.environ["YTDLP_KEKSIT"] = os.path.join(t, "ei_ole.txt")
            tarkista("keksitiedosto: puuttuva tiedosto -> None", na.keksitiedosto() is None)
        finally:
            del os.environ["YTDLP_KEKSIT"]

    s = subprocess.run([sys.executable, os.path.join(KOTI, "netti_live.py"), "--help"], capture_output=True, text=True, timeout=120).stdout
    tarkista("netti_live --help: uudet valitsimet (--web, --web-leveys, --web-fps, --web-laatu, --tila-tiedosto)", all(x in s for x in ("--web ", "--web-leveys", "--web-fps", "--web-laatu", "--tila-tiedosto")))


def main():
    for t in (testaa_ymparisto, testaa_asetukset, testaa_komento, testaa_tiedostot_ja_muisti, testaa_silmukka, testaa_lapsiprosessi, testaa_web, testaa_netti_live_palvelinosat):
        t()
    print()
    if VIRHEITA:
        print(f"{len(VIRHEITA)} testia epaonnistui: {', '.join(VIRHEITA)}")
        return 1
    print("kaikki testit OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
