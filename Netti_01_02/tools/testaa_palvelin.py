"""Palvelinversion (Netti_01_02) testit: kontin resurssirajat (netti_ymparisto), asetukset ja ajosilmukka (netti_palvelin), selainnakyma (netti_web), sulkusignaali ja tilatiedosto
(netti_live) sekä kayttajan kalibrointikansio (kalibrointi_valinta).

    python tools/testaa_palvelin.py        (kontissa: docker run --rm IMAGE python /app/tools/testaa_palvelin.py)

Ei tarvitse videota, YouTubea eika verkkoa. Poistumiskoodi 1, jos jokin testi epaonnistuu."""
import base64
import collections
import csv
import io
import json
import os
import re
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
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
        tarkista("oletukset: video paalla, csv paalla, libx264, viive 10, web 63884", a.tallenna_video and a.csv and a.koodekki == "cpu" and a.viive == 10.0 and a.web_portti == 63884)
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
                                  "NETTI_LISAVALITSIMET", "NETTI_AANI", "NETTI_KOODEKKI", "NETTI_EI_KIERRETTA", "NETTI_HILJAA", "NETTI_ODOTUS_S", "NETTI_KALIBROINTI")}
        a = np_.Asetukset(dict(env, **tyhjat))
        tarkista("tyhjat muuttujat (Unraid valittaa ne) = ei asetettu: oletukset", a.laji == "tiedostot" and a.jatkuva and a.tallenna_video and a.csv and a.nopea and a.viive == 10.0 and a.saikeet is None
                 and a.web_portti == 63884 and a.lisa == [] and a.aani == "auto" and a.koodekki == "cpu" and not a.ei_kierretta and a.odotus_s == 60.0 and a.kalibrointi is None)
        tarkista("NETTI_KALIBROINTI: oletus ei asetettu (netti_live:n oletus uusi), uusi / valitse (kirjainkoolla ei valia)", np_.Asetukset(env).kalibrointi is None and np_.Asetukset(dict(env, NETTI_KALIBROINTI="uusi")).kalibrointi == "uusi"
                 and np_.Asetukset(dict(env, NETTI_KALIBROINTI=" Valitse ")).kalibrointi == "valitse")
        try:
            np_.Asetukset(dict(env, NETTI_KALIBROINTI="aina"))
            tarkista("NETTI_KALIBROINTI='aina' -> AsetusVirhe", False)
        except np_.AsetusVirhe as e:
            tarkista("NETTI_KALIBROINTI='aina' -> AsetusVirhe", "NETTI_KALIBROINTI" in str(e) and "uusi" in str(e) and "valitse" in str(e), str(e)[:90])
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
        tarkista("tiedosto: --lahde ja --nopea", lahde_komennosta(c)[1].endswith("Matarmaa osa 1 (2).mp4") and "--nopea" in c)
        tarkista("tulostiedostojen nimet: tunniste + aikaleima", os.path.join(t, "ulos", "kivenseuranta_Matarmaa_osa_1_2_20260101_120000.mp4") in c and os.path.join(t, "ulos", "heitot_Matarmaa_osa_1_2_20260101_120000.csv") in c)
        tarkista("web-valitsimet ja tilatiedosto", "--web 63884" in j and "--web-leveys 1920" in j and "--tila-tiedosto /tmp/tila.json" in j)
        tarkista("tiedostolle ei --lataus eika --kiri", "--lataus" not in c and "--kiri" not in c)
        tarkista("kalibrointi: oletuksena ei --kalibrointi eika --kansio (netti_live kalibroi videon aina uudelleen)", "--kalibrointi" not in c and "--kansio" not in c)
        c = np_.rakenna_komento(np_.Asetukset(dict(env, NETTI_LAHDE="Matarmaa osa 1 (2).mp4", NETTI_KALIBROINTI="valitse")), "tiedostot", a.arvo["kuvio"], "x")
        tarkista("NETTI_KALIBROINTI=valitse: --kalibrointi valitse", c[c.index("--kalibrointi") + 1] == "valitse" and "--kansio" not in c)
        c = np_.rakenna_komento(np_.Asetukset(dict(env, NETTI_LAHDE="Matarmaa osa 1 (2).mp4", NETTI_KALIBROINTI="valitse", NETTI_KANSIO="tulokset_nettivideo3")), "tiedostot", a.arvo["kuvio"], "x")
        tarkista("NETTI_KANSIO ohittaa kalibroinnin: vain --kansio", c[c.index("--kansio") + 1] == "tulokset_nettivideo3" and "--kalibrointi" not in c)
        a = np_.Asetukset(dict(env, NETTI_LAHDE="Matarmaa osa 1 (2).mp4", NETTI_NOPEA="0", NETTI_ALKU="30", NETTI_LOPPU="90", NETTI_VIDEO="0", NETTI_CSV="0", NETTI_WEB_PORTTI="0"))
        c = np_.rakenna_komento(a, a.laji, a.arvo["kuvio"], "x")
        tarkista("NETTI_NOPEA=0: ei --nopea; --alku / --loppu", "--nopea" not in c and c[c.index("--alku") + 1] == "30" and c[c.index("--loppu") + 1] == "90")
        tarkista("ei videota: ei --ulos; ei csv: --ei-csv; ei webia: ei --web", "--ulos" not in c and "--ei-csv" in c and "--csv" not in c and "--web" not in c)
        a = np_.Asetukset(dict(env, NETTI_LAHDE="hnTTLqHr5x8", NETTI_KIRI="0", NETTI_LISAVALITSIMET="--etumatka 5"))
        c = np_.rakenna_komento(a, a.laji, a.arvo, "x", None, 32.0, ["--debug-viive"])
        tarkista("YouTube: --youtube TUNNUS, ei --nopea, --kiri 0", lahde_komennosta(c) == ("--youtube", "hnTTLqHr5x8") and "--nopea" not in c and c[c.index("--kiri") + 1] == "0")
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
        c = np_.rakenna_komento(a, "youtube", "-Z4JPPqtlmU", "x", None, 32.0)
        tarkista("viivalla alkava YouTube-tunnus (noin 1,6 % tunnuksista): lahde yhtena argumenttina --youtube=-Z4JPPqtlmU", "--youtube=-Z4JPPqtlmU" in c and "--youtube" not in c and lahde_komennosta(c) == ("--youtube", "-Z4JPPqtlmU"))
        tarkista("lahteen arvo ei jaa omaksi argumentikseen", "-Z4JPPqtlmU" not in c)
        ymp = dict(os.environ, HTTPS_PROXY="http://127.0.0.1:9", https_proxy="http://127.0.0.1:9", HTTP_PROXY="http://127.0.0.1:9", http_proxy="http://127.0.0.1:9", NO_PROXY="", no_proxy="")
        pr = subprocess.run([sys.executable, os.path.join(KOTI, "netti_live.py"), "--youtube=-Z4JPPqtlmU", "--ei-ikkunaa", "--hiljaa"], capture_output=True, text=True, timeout=120, env=ymp)
        ulos = pr.stdout + pr.stderr
        tarkista("netti_live hyvaksyy --youtube=-Z4JPPqtlmU (tunnus ei ole valitsin; ajo yrittaa hakea virran, ja haku epaonnistuu, koska verkko on poissa)",
                 "unrecognized arguments" not in ulos and ("watch?v=-Z4JPPqtlmU" in ulos or "yt_dlp puuttuu" in ulos), ulos.strip().splitlines()[-1][:110] if ulos.strip() else "")
        pr = subprocess.run([sys.executable, os.path.join(KOTI, "netti_live.py"), "--youtube", "-Z4JPPqtlmU", "--ei-ikkunaa", "--hiljaa"], capture_output=True, text=True, timeout=120, env=ymp)
        tarkista("vertailu: erillisena argumenttina viivalla alkava tunnus hylataan (siksi yhtena argumenttina)", "unrecognized arguments" in (pr.stdout + pr.stderr))


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


def lahde_komennosta(cmd):
    """(valitsin, arvo) netti_live-komennosta: lahde annetaan yhtena argumenttina (--youtube=ARVO / --url=ARVO / --lahde=ARVO), jotta viivalla alkava arvo ei tulkittu valitsimeksi."""
    for t in cmd:
        for v in ("--youtube", "--url", "--lahde"):
            if t.startswith(v + "="):
                return v, t[len(v) + 1:]
    return None, None


class EsiPysaytys:
    """Testi-Pysaytys: nuku() ei odota oikeasti; pysaytys pyydetaan n:nnella nukkumisella."""

    def __init__(self, pysayta_nukkumisella=None, kello=None):
        self.tapahtuma = threading.Event()
        self.signaali = None
        self.nukkumiset = []
        self.raja = pysayta_nukkumisella
        self.kello = kello                                      # EsiKello: etenee nukutun ajan verran

    def pyydetty(self):
        return self.tapahtuma.is_set()

    def nuku(self, s):
        self.nukkumiset.append(s)
        if self.kello is not None:
            self.kello.t += s
        if self.raja is not None and len(self.nukkumiset) >= self.raja:
            self.tapahtuma.set()
        return self.tapahtuma.is_set()


class EsiKello:
    """Testikello: kello() palauttaa t:n; EsiPysaytys.nuku() kasvattaa sita."""

    def __init__(self, t=1_800_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


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
        tarkista("kertaajo kansiolle: molemmat videot, paluukoodi 0", rc == 0 and [lahde_komennosta(c)[1].rsplit("/", 1)[1] for c in ajot] == ["a.mp4", "b.mp4"], str([lahde_komennosta(c)[1][-8:] for c in ajot]))
        tarkista("ulos- ja tyokansiot luotu", os.path.isdir(a.ulos_kansio) and os.path.isdir(a.tyo_kansio))
        ajot.clear()

        def ajuri_vika(cmd, env=None, cwd=None):
            ajot.append(cmd)
            return 7 if lahde_komennosta(cmd)[1].endswith("b.mp4") else 0

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
        a_ajot = [lahde_komennosta(c)[1].rsplit("/", 1)[1] for c in ajot]
        tarkista("valvonta: a.mp4 valmis kerran, b.mp4 (epaonnistuu) yritetaan uudelleen", a_ajot.count("a.mp4") == 1 and a_ajot.count("b.mp4") >= 2, str(a_ajot))
        tarkista("valvonta: odotusaika asetuksen mukainen", all(s == 5.0 for s in ps.nukkumiset), str(ps.nukkumiset))
        ajot.clear()
        ps = EsiPysaytys(pysayta_nukkumisella=6)
        np_.palvele(a, ps, ajuri_vika, 32.0, lambda: nyt)
        k = np_.Kasitellyt(a.kasitellyt_polku)
        b_ajot = [lahde_komennosta(c)[1].endswith("b.mp4") for c in ajot].count(True)
        tarkista("valvonta (uusi kaynnistys): b.mp4 saa kolmannen ja viimeisen yrityksen, sen jalkeen se jaa pois", b_ajot == 1 and k.valmis("tiedostot", os.path.join(v, "b.mp4")), str(b_ajot))
        tarkista("valvonta: a.mp4 ei aja uudelleen uudessa kaynnistyksessa", [lahde_komennosta(c)[1].endswith("a.mp4") for c in ajot].count(True) == 0)
        ajot.clear()
        np_.palvele(a, EsiPysaytys(pysayta_nukkumisella=3), ajuri_vika, 32.0, lambda: nyt)
        tarkista("valvonta: yritysrajan ylittanytta b.mp4:aa ei enaa ajeta", ajot == [])
        kirjoita(os.path.join(v, "c.mp4"), "c", nyt - 100)
        ajot.clear()
        np_.palvele(a, EsiPysaytys(pysayta_nukkumisella=1), ajuri_ok, 32.0, lambda: nyt)
        tarkista("valvonta: uusi video c.mp4 havaitaan ja ajetaan", [lahde_komennosta(c)[1].rsplit("/", 1)[1] for c in ajot] == ["c.mp4"])

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


def testaa_linkkilista():
    print("linkkilista (youtube.txt): YouTube-osoitteet ilman kontin uudelleenkaynnistysta")
    with tempfile.TemporaryDirectory() as t:
        p = os.path.join(t, "youtube.txt")
        kirjoita(p, "\ufeff# Matarmaa-peli\r\nhnTTLqHr5x8\r\n\r\nhttps://www.youtube.com/watch?v=abcdefghijk&t=30s   # toinen peli\r\nhttps://youtu.be/hnTTLqHr5x8\r\n"
                    "Matarmaa osa 1\r\nrtsp://kamera.lan/virta\r\nhttps://example.com/live/a.m3u8 jotain tekstia\r\nHyvinkaa_pe ja tekstia\r\nhttps://www.youtube.com/@kanava/live\r\n")
        huom = set()
        rivit = np_.lue_linkit(p, huom)
        tarkista("lue_linkit: BOM, CRLF, kommentit, tyhjat rivit, #t= sailyy, sama video eri muodoissa vain kerran",
                 rivit == [("youtube", "hnTTLqHr5x8"), ("youtube", "https://www.youtube.com/watch?v=abcdefghijk&t=30s"), ("url", "rtsp://kamera.lan/virta"), ("url", "https://example.com/live/a.m3u8"),
                           ("youtube", "https://www.youtube.com/@kanava/live")], str(rivit))
        tarkista("lue_linkit: tuntematon teksti ja 11 merkin sana tekstin kanssa ohitetaan ja ilmoitetaan kerran", huom == {"Matarmaa osa 1", "Hyvinkaa_pe ja tekstia"}, str(sorted(huom)))
        tarkista("lue_linkit: sama virheellinen rivi ei ilmoiteta uudelleen", (np_.lue_linkit(p, huom), len(huom))[1] == 2)
        tarkista("lue_linkit: puuttuva tiedosto ja None ovat tyhja lista", np_.lue_linkit(os.path.join(t, "ei_ole.txt")) == [] and np_.lue_linkit(None) == [])
        kirjoita(p, "https://youtu.be/hnTTLqHr5x8\n")
        tarkista("lue_linkit: pelkka osoite ilman rivinvaihtoa / kommenttia", np_.lue_linkit(p) == [("youtube", "https://youtu.be/hnTTLqHr5x8")])
        tarkista("linkin_avain: tunnus, watch?v= ja youtu.be ovat sama video", np_.linkin_avain("youtube", "hnTTLqHr5x8") == np_.linkin_avain("youtube", "https://www.youtube.com/watch?v=hnTTLqHr5x8&t=5")
                 == np_.linkin_avain("youtube", "https://youtu.be/hnTTLqHr5x8") == "yt_hnTTLqHr5x8")
        tarkista("linkin_avain: kanavan osoite ja muu verkko-osoite sellaisenaan", np_.linkin_avain("youtube", "https://www.youtube.com/@k/live") == "https://www.youtube.com/@k/live" and np_.linkin_avain("url", "rtsp://x") == "rtsp://x")

        # valvonnan muisti: rivikohtainen odotus ja yritysraja
        k = np_.Kasitellyt(os.path.join(t, "ulos", "kasitellyt.json"))
        nyt = 1_800_000_000.0
        tarkista("avain: tunnus ja osoite jakavat avaimen", k.avain("youtube_lista", "hnTTLqHr5x8") == k.avain("youtube_lista", "https://youtu.be/hnTTLqHr5x8") == "lista:yt_hnTTLqHr5x8")
        odotukset = []
        for i in range(1, np_.LINKKI_YRITYKSIA_MAX + 1):
            r = k.kirjaa("youtube_lista", "hnTTLqHr5x8", False, nyt, 60.0)
            odotukset.append(r["seuraava"] - nyt)
            tarkista(f"epaonnistuminen {i}: " + ("rivi jaa pois" if i >= np_.LINKKI_YRITYKSIA_MAX else "yritetaan uudelleen"), k.valmis("youtube_lista", "hnTTLqHr5x8") == (i >= np_.LINKKI_YRITYKSIA_MAX))
        tarkista("odotus pitenee 60, 120, 240, 480, 900 s ja pysyy 900 s:ssa", odotukset[:7] == [60.0, 120.0, 240.0, 480.0, 900.0, 900.0, 900.0] and max(odotukset) == np_.ODOTUS_MAX_S, str(odotukset))
        k.kirjaa("youtube_lista", "abcdefghijk", False, nyt, 60.0)
        tarkista("odottaa(): ennen seuraavaa yritysta tosi, sen jalkeen epatosi", k.odottaa("youtube_lista", "abcdefghijk", nyt + 59) and not k.odottaa("youtube_lista", "abcdefghijk", nyt + 60))
        r = k.kirjaa("youtube_lista", "abcdefghijk", True, nyt + 61, 60.0)
        tarkista("onnistuminen: valmis, ei odotusta, yrityslaskuri nollattu, linkki talteen", k.valmis("youtube_lista", "abcdefghijk") and "seuraava" not in r and r["yrityksia"] == 0 and r["linkki"] == "abcdefghijk")
        tarkista("tiedostojen kirjaus ei muutu (3 yritysta, ei odotusta)", "seuraava" not in k.kirjaa("tiedostot", os.path.join(t, "x.mp4"), False))

    with tempfile.TemporaryDirectory() as t:
        nyt = time.time()
        kirjoita(os.path.join(t, "video", "a.mp4"), "a", nyt - 100)
        lista = os.path.join(t, "youtube.txt")
        kirjoita(lista, "hnTTLqHr5x8\nhttps://www.youtube.com/watch?v=abcdefghijk\n")
        a = np_.Asetukset(dict(NETTI_DATA=t))
        tarkista("NETTI_LINKIT: oletuspolku on <data>/youtube.txt", a.linkit_polku == lista)
        tarkista("lahteet: videot ensin, sitten linkit tiedoston jarjestyksessa", np_.lahteet(a, None, nyt) == [("tiedostot", os.path.join(t, "video", "a.mp4")), ("youtube_lista", "hnTTLqHr5x8"), ("youtube_lista", "https://www.youtube.com/watch?v=abcdefghijk")],
                 str(np_.lahteet(a, None, nyt)))
        for env, selitys in ((dict(NETTI_LINKIT="ei"), "NETTI_LINKIT=ei"), (dict(NETTI_JATKUVA="0"), "kertaajo (ei valvontaa)"), (dict(NETTI_LAHDE=os.path.join(t, "video", "a.mp4")), "yksittainen tiedosto"), (dict(NETTI_LAHDE="hnTTLqHr5x8"), "yksittainen YouTube-lahde")):
            a2 = np_.Asetukset(dict(dict(NETTI_DATA=t), **env))
            tarkista(f"lahteet: ei linkkilistaa, kun {selitys}", all(not l.endswith("_lista") for l, _ in np_.lahteet(a2, None, nyt)), str(np_.lahteet(a2, None, nyt)))
        a3 = np_.Asetukset(dict(NETTI_DATA=t, NETTI_LINKIT=os.path.join(t, "muu.txt")))
        kirjoita(a3.linkit_polku, "rtsp://kamera.lan/x\n")
        tarkista("NETTI_LINKIT: oma polku", np_.lahteet(a3, None, nyt)[-1] == ("url_lista", "rtsp://kamera.lan/x"))
        a4 = np_.Asetukset(dict(NETTI_DATA=t, NETTI_LINKIT=""))
        tarkista("NETTI_LINKIT tyhja = oletus (Unraid valittaa tyhjat)", a4.linkit_polku == lista)
        k = np_.Kasitellyt(a.kasitellyt_polku)
        k.kirjaa("youtube_lista", "https://youtu.be/hnTTLqHr5x8", True)
        k.kirjaa("youtube_lista", "abcdefghijk", False, nyt, 60.0)
        tarkista("lahteet: valmis (eri osoitemuodossa kirjattu) ja odottava rivi jaavat pois", np_.lahteet(a, k, nyt) == [("tiedostot", os.path.join(t, "video", "a.mp4"))], str(np_.lahteet(a, k, nyt)))
        tarkista("lahteet: odotuksen jalkeen rivi palaa mukaan", ("youtube_lista", "https://www.youtube.com/watch?v=abcdefghijk") in np_.lahteet(a, k, nyt + 61))
        # komento
        c = np_.rakenna_komento(a, "youtube_lista", "hnTTLqHr5x8", "20260101_120000", None, 32.0)
        tarkista("komento: youtube_lista -> --youtube TUNNUS, ei --nopea, tulostiedostojen tunniste yt_<tunnus>", lahde_komennosta(c) == ("--youtube", "hnTTLqHr5x8") and "--nopea" not in c and "--lahde" not in c
                 and os.path.join(a.ulos_kansio, "kivenseuranta_{otsikko}_yt_hnTTLqHr5x8_20260101_120000.mp4") in c, " ".join(c[3:6]))
        c = np_.rakenna_komento(a, "url_lista", "rtsp://kamera.lan/x", "x", None, 4.0)
        tarkista("komento: url_lista -> --url OSOITE, vahan muistia: --lataus 8", lahde_komennosta(c) == ("--url", "rtsp://kamera.lan/x") and c[c.index("--lataus") + 1] == "8")

    # ajosilmukka: linkit ajetaan kerran, lisatty rivi huomataan, epaonnistuminen pidentaa odotusta
    with tempfile.TemporaryDirectory() as t:
        ek = EsiKello()
        lista = os.path.join(t, "youtube.txt")
        kirjoita(lista, "hnTTLqHr5x8\n")
        a = np_.Asetukset(dict(NETTI_DATA=t, NETTI_ODOTUS_S="60"))
        ajot = []

        def ajuri(cmd, env=None, cwd=None):
            ref = lahde_komennosta(cmd)[1]
            ajot.append((ek.t, ref))
            if len(ajot) == 1:
                with open(lista, "a") as f:                                       # kesken ensimmaisen ajon kayttaja lisaa uuden linkin tiedostoon (kontin kaynnistysta ei tarvita)
                    f.write("https://youtu.be/abcdefghijk\n")
            return 0

        ps = EsiPysaytys(pysayta_nukkumisella=5, kello=ek)
        rc = np_.palvele(a, ps, ajuri, 32.0, ek)
        tarkista("silmukka: molemmat linkit ajetaan, kumpikin kerran, lisatty rivi huomataan ilman uudelleenkaynnistysta", rc == 0 and [r for _, r in ajot] == ["hnTTLqHr5x8", "https://youtu.be/abcdefghijk"], str(ajot))
        tarkista("silmukka: valmiit kirjattu muistiin (kasitellyt.json)", np_.Kasitellyt(a.kasitellyt_polku).valmis("youtube_lista", "hnTTLqHr5x8") and np_.Kasitellyt(a.kasitellyt_polku).valmis("youtube_lista", "abcdefghijk"))
        ajot.clear()
        np_.palvele(a, EsiPysaytys(pysayta_nukkumisella=3, kello=ek), ajuri, 32.0, ek)
        tarkista("silmukka: uudelleenkaynnistys ei aja valmiita linkkeja uudelleen", ajot == [])

    with tempfile.TemporaryDirectory() as t:
        ek = EsiKello()
        kirjoita(os.path.join(t, "youtube.txt"), "hnTTLqHr5x8\n")
        a = np_.Asetukset(dict(NETTI_DATA=t, NETTI_ODOTUS_S="60"))
        ajat = []

        def ajuri_vika(cmd, env=None, cwd=None):
            ajat.append(ek.t - 1_800_000_000.0)
            return 1                                                              # esim. lahetys ei ole viela alkanut

        np_.palvele(a, EsiPysaytys(pysayta_nukkumisella=120, kello=ek), ajuri_vika, 32.0, ek)
        tarkista(f"epaonnistuva linkki: {np_.LINKKI_YRITYKSIA_MAX} yritysta pidentyvin valein (0, 60, 180, 420, 900, 1800 ... s), sitten rivi jaa pois", ajat == [0, 60, 180, 420, 900, 1800, 2700, 3600, 4500, 5400], str(ajat))
        tarkista("epaonnistunut linkki ei kuormita: muisti sailyy ja rivi on 'valmis' (ohitettu)", np_.Kasitellyt(a.kasitellyt_polku).valmis("youtube_lista", "hnTTLqHr5x8"))

    with tempfile.TemporaryDirectory() as t:
        ek = EsiKello()
        kirjoita(os.path.join(t, "youtube.txt"), "hnTTLqHr5x8\nabcdefghijk\n")
        a = np_.Asetukset(dict(NETTI_DATA=t))
        ps = EsiPysaytys(kello=ek)

        def ajuri_pysaytetty(cmd, env=None, cwd=None):
            ps.tapahtuma.set()                                                    # docker stop kesken linkin ajon
            return 0

        rc = np_.palvele(a, ps, ajuri_pysaytetty, 32.0, ek)
        k = np_.Kasitellyt(a.kasitellyt_polku)
        tarkista("pysaytys kesken linkin ajon: paluukoodi 0, linkkia ei kirjata valmiiksi (ajetaan uudestaan kaynnistyksen jalkeen)", rc == 0 and not k.valmis("youtube_lista", "hnTTLqHr5x8") and not k.data)


def testaa_odotuksen_ylivuoto():
    print("valvonta: odotuksen kasvu ei ylivuoda (pitka jakso nopeita epaonnistumisia)")
    import contextlib
    with tempfile.TemporaryDirectory() as t:
        ek = EsiKello()
        a = np_.Asetukset(dict(NETTI_DATA=t, NETTI_LAHDE="hnTTLqHr5x8", NETTI_JATKUVA="1", NETTI_ODOTUS_S="60"))
        ajot = []

        def ajuri_ei_lahetysta(cmd, env=None, cwd=None):
            ajot.append(lahde_komennosta(cmd))
            return 1                                                              # nopea epaonnistuminen: lahetys ei ole alkanut

        ps = EsiPysaytys(pysayta_nukkumisella=1300, kello=ek)                     # 1300 perakkaista nopeaa epaonnistumista (~15 vrk): ennen korjausta OverflowError kohdassa ~1025
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                rc = np_.palvele(a, ps, ajuri_ei_lahetysta, 32.0, ek)
            tarkista("odotus pysyy rajoissa 1300 perakkaisen nopean epaonnistumisen jalkeen: ei ylivuotoa, odotus enintaan 15 min", rc == 0 and len(ps.nukkumiset) == 1300 and max(ps.nukkumiset) == np_.ODOTUS_MAX_S,
                     f"{rc} {len(ps.nukkumiset)} {max(ps.nukkumiset)}")
        except OverflowError as e:
            tarkista("odotus pysyy rajoissa 1300 perakkaisen nopean epaonnistumisen jalkeen: ei ylivuotoa", False, str(e))


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
            tarkista("tiedostot suojattu samoin (myos kuvasyote)", pyyda(w2.portti, "/tiedostot.json")[0] == 401 and pyyda(w2.portti, "/video.mjpg")[0] == 401 and pyyda(w2.portti, "/syote")[0] == 401 and pyyda(w2.portti, "/lisa?id=a&s=5")[0] == 401)
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
    import inspect
    import netti_live as nl
    tarkista("naytto-rajapinta: anna / nayta_heti hyvaksyvat kuvan ajan (selaimen syotteen aikaleima), ikkunanaytto ja WebNaytto",
             all("aika" in inspect.signature(f).parameters for f in (nl.Naytto.anna, nl.Naytto.nayta_heti, nw.WebNaytto.anna, nw.WebNaytto.nayta_heti)))
    tarkista("netti_live antaa kuville video-ajan (reaaliaika: f / fps, simulaatio: f_out / fps), selaimen syote ei jaa seinakellon varaan",
             "naytto.anna(npi_kuva, f / fps)" in inspect.getsource(nl.aja_reaaliaika) and "naytto.nayta_heti(kuva, f_out / lahde.fps)" in inspect.getsource(nl.aja_sim))
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
            tarkista("keksitiedosto: kopio on vain omistajan luettavissa (0600)", oct(os.stat(k).st_mode & 0o777) == "0o600", oct(os.stat(k).st_mode & 0o777))
            os.remove(k)
            pr = subprocess.run([sys.executable, "-c", "import netti_aani as na; print(na.keksitiedosto())"], capture_output=True, text=True, timeout=60, cwd=KOTI, env=dict(os.environ, YTDLP_KEKSIT=alku))
            kopio = pr.stdout.strip()
            tarkista("keksitiedosto: kopio poistuu, kun prosessi paattyy (ei jaa /tmp-kansioon)", pr.returncode == 0 and kopio.endswith(".txt") and not os.path.exists(kopio), kopio)
            os.environ["YTDLP_KEKSIT"] = os.path.join(t, "ei_ole.txt")
            tarkista("keksitiedosto: puuttuva tiedosto -> None", na.keksitiedosto() is None)
        finally:
            del os.environ["YTDLP_KEKSIT"]

    s = subprocess.run([sys.executable, os.path.join(KOTI, "netti_live.py"), "--help"], capture_output=True, text=True, timeout=120).stdout
    tarkista("netti_live --help: uudet valitsimet (--web, --web-leveys, --web-fps, --web-laatu, --tila-tiedosto)", all(x in s for x in ("--web ", "--web-leveys", "--web-fps", "--web-laatu", "--tila-tiedosto")))


def testaa_videon_nimi_palvelin():
    print("YouTube-videon nimi tulostiedostojen nimiin (netti_palvelin: komento ja loki)")
    import contextlib
    import netti_live as nl
    tarkista("OTSIKKO_PAIKKA on sama netti_palvelinissa ja netti_livessa", np_.OTSIKKO_PAIKKA == nl.OTSIKKO_PAIKKA == "{otsikko}")
    with tempfile.TemporaryDirectory() as t:
        a = np_.Asetukset(dict(NETTI_DATA=t, NETTI_LAHDE="hnTTLqHr5x8"))
        c = np_.rakenna_komento(a, "youtube", "hnTTLqHr5x8", "20260101_120000", None, 32.0)
        ulos, csv_ = c[c.index("--ulos") + 1], c[c.index("--csv") + 1]
        tarkista("YouTube: --ulos ja --csv nimessa {otsikko}-paikka tunnisteen edessa (netti_live taydentaa nimen, kun yt_dlp on kertonut sen)",
                 ulos == os.path.join(a.ulos_kansio, "kivenseuranta_{otsikko}_yt_hnTTLqHr5x8_20260101_120000.mp4") and csv_ == os.path.join(a.ulos_kansio, "heitot_{otsikko}_yt_hnTTLqHr5x8_20260101_120000.csv"), f"{ulos} {csv_}")
        nimella = [nl.otsikko_polkuun(x, "Matarmaa SM-finaali: Hyvinkää", ascii_vain=False) for x in (ulos, csv_)]
        tarkista("netti_live taydentaa: nimi tunnisteen edessa (mp4 ja csv; _kivet.csv tulee csv:n viereen)", nimella == [os.path.join(a.ulos_kansio, "kivenseuranta_Matarmaa SM-finaali Hyvinkää_yt_hnTTLqHr5x8_20260101_120000.mp4"),
                                                                                                           os.path.join(a.ulos_kansio, "heitot_Matarmaa SM-finaali Hyvinkää_yt_hnTTLqHr5x8_20260101_120000.csv")], str(nimella))
        tarkista("ilman nimea (YouTube ei kertonut) nimet ovat kuten ennen", [nl.otsikko_polkuun(x, None) for x in (ulos, csv_)] == [os.path.join(a.ulos_kansio, "kivenseuranta_yt_hnTTLqHr5x8_20260101_120000.mp4"),
                                                                                                                                       os.path.join(a.ulos_kansio, "heitot_yt_hnTTLqHr5x8_20260101_120000.csv")])
        c = np_.rakenna_komento(a, "url", "rtsp://kamera.lan/x", "x", None, 32.0)
        tarkista("virta (--url): ei otsikkopaikkaa (nimea ei ole)", all("{otsikko}" not in x for x in c))
        c = np_.rakenna_komento(a, "youtube_lista", "https://youtu.be/hnTTLqHr5x8", "x", None, 32.0)
        tarkista("linkkilistan YouTube-rivi: otsikkopaikka mukana", any("{otsikko}_yt_hnTTLqHr5x8" in x for x in c))

        # ajosilmukan loki kertoo videon nimen (lapsi kirjoittaa sen tilatiedostoon)
        nyt = time.time()

        def ajuri(cmd, env=None, cwd=None):
            polku = cmd[cmd.index("--tila-tiedosto") + 1]
            with open(polku, "w", encoding="utf-8") as f:
                json.dump(dict(lahde="hnTTLqHr5x8", otsikko="Matarmaa SM-finaali – Hyvinkää", live=False), f, ensure_ascii=False)
            return 0
        a = np_.Asetukset(dict(NETTI_DATA=t, NETTI_LAHDE="hnTTLqHr5x8", NETTI_JATKUVA="0"))
        ulos_loki = io.StringIO()
        with contextlib.redirect_stdout(ulos_loki):
            rc = np_.palvele(a, EsiPysaytys(), ajuri, 32.0, lambda: nyt)
        tarkista("ajosilmukan loki: 'ajo paattyi (<videon nimi>): koodi 0'", rc == 0 and "ajo paattyi (Matarmaa SM-finaali – Hyvinkää): koodi 0" in ulos_loki.getvalue(), ulos_loki.getvalue()[-200:])

        def ajuri_ilman(cmd, env=None, cwd=None):
            return 0
        ulos_loki = io.StringIO()
        with contextlib.redirect_stdout(ulos_loki):
            np_.palvele(a, EsiPysaytys(), ajuri_ilman, 32.0, lambda: nyt)
        tarkista("ajosilmukan loki: ilman nimea kuten ennen", "ajo paattyi: koodi 0" in ulos_loki.getvalue(), ulos_loki.getvalue()[-200:])


def testaa_web_resoluutio():
    print("selainnakyma: kuva oletuksena taydella resoluutiolla (1920 px), ei pienennetty 1280 px:aan")
    import cv2
    a = np_.Asetukset({})
    tarkista("NETTI_WEB_LEVEYS oletus 1920 (analysoitu kuva sellaisenaan)", a.web_leveys == 1920 and np_.Asetukset(dict(NETTI_WEB_LEVEYS="1280")).web_leveys == 1280, str(a.web_leveys))
    with tempfile.TemporaryDirectory() as t:
        w = nw.WebNaytto(0, t, fps=25, tila=lambda: dict(lahde="testi"))                         # leveys oletuksella
        p = w.portti
        try:
            kuva = np.zeros((1080, 1920, 3), np.uint8)
            kuva[:, :960] = (0, 200, 0)
            kuva[::2, 1900:] = 255                                                                      # tarkka kuvio koko leveydelle: pienennys sotkisi sen
            w._julkaise(kuva)
            s, h, b = pyyda(p, "/kuva.jpg")
            d = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
            tarkista("oletusleveydella kuva menee selaimeen sellaisenaan: 1920 x 1080", s == 200 and d.shape[:2] == (1080, 1920), str(d.shape))
            tarkista("oikean reunan hienoin kuvio sailyy (pienennetyssa kuvassa se olisi tasoittunut)", d[::2, 1905:].mean() > 200 and d[1::2, 1905:].mean() < 60, f"{d[::2, 1905:].mean():.0f} / {d[1::2, 1905:].mean():.0f}")
            tila = json.loads(pyyda(p, "/tila")[2])
            tarkista("/tila kertoo kuvan leveyden (sivu rajaa kuvan natiivikokoonsa)", tila["kuva_leveys"] == 1920, str(tila.get("kuva_leveys")))
            sivu = pyyda(p, "/")[2].decode("utf-8")
            tarkista("sivu: sisalto saa olla 1920 px leveä ja kuva rajataan tilan kuva_leveys-arvoon", "main{max-width:1920px" in sivu and "kuva_leveys" in sivu and "max-width:1280px" not in sivu)
        finally:
            w.sulje()
        w2 = nw.WebNaytto(0, t, leveys=1280, fps=25, tila=lambda: dict(lahde="testi"))
        try:
            w2._julkaise(kuva)
            d2 = cv2.imdecode(np.frombuffer(pyyda(w2.portti, "/kuva.jpg")[2], np.uint8), cv2.IMREAD_COLOR)
            tarkista("NETTI_WEB_LEVEYS=1280: kuva pienennetaan (kevyempi verkolle) ja tila kertoo sen", d2.shape[:2] == (720, 1280) and json.loads(pyyda(w2.portti, "/tila")[2])["kuva_leveys"] == 1280, str(d2.shape))
        finally:
            w2.sulje()


def testaa_web_videon_nimi():
    print("selainnakyma: videon nimi tilatiedossa ja aakkosia sisaltavat tiedostonimet")
    import html
    import urllib.parse
    with tempfile.TemporaryDirectory() as t:
        mp4 = "kivenseuranta_Matarmaa SM-finaali – Hyvinkää 2026_yt_hnTTLqHr5x8_20260101_120000.mp4"
        csv_ = "heitot_Matarmaa SM-finaali – Hyvinkää 2026_yt_hnTTLqHr5x8_20260101_120000_kivet.csv"
        kirjoita(os.path.join(t, mp4), bytes(range(256)) * 10)
        kirjoita(os.path.join(t, csv_), "a,b\n1,2\n")
        w = nw.WebNaytto(0, t, leveys=320, fps=5, laatu=70, tila=lambda: dict(lahde="hnTTLqHr5x8", otsikko="Matarmaa SM-finaali <b>&</b>"))
        p = w.portti
        try:
            tila = json.loads(pyyda(p, "/tila")[2])
            tarkista("/tila: videon nimi mukana lahteen vieressa", tila["otsikko"] == "Matarmaa SM-finaali <b>&</b>" and tila["lahde"] == "hnTTLqHr5x8", str(tila))
            sivu = pyyda(p, "/")[2].decode("utf-8")
            tarkista("sivu: nimi naytetaan escapoituna (esc) ja valilehden otsikkona (document.title)", "esc(t.otsikko||t.lahde)" in sivu and "document.title" in sivu and "t.lahde.replace" not in sivu)
            tiedostot = json.loads(pyyda(p, "/tiedostot.json")[2])
            tarkista("tiedostolista: aakkoset, valilyonnit ja ajatusviiva sailyvat nimessa", sorted(x["nimi"] for x in tiedostot) == sorted([mp4, csv_]), str([x["nimi"] for x in tiedostot]))
            s, h, b = pyyda(p, "/tiedostot/" + urllib.parse.quote(mp4))
            tarkista("mp4 latautuu aakkosnimella (koodattu osoite), Content-Disposition filename*=UTF-8", s == 200 and len(b) == 2560 and "filename*=UTF-8''" in h["Content-Disposition"] and urllib.parse.quote(mp4) in h["Content-Disposition"], str(h.get("Content-Disposition")))
            s, h, b = pyyda(p, "/tiedostot/" + urllib.parse.quote(csv_))
            tarkista("csv latautuu aakkosnimella", s == 200 and b == b"a,b\n1,2\n")
            s, h, b = pyyda(p, "/tiedostot/")
            tarkista("HTML-lista: nimi escapoituna ja linkki koodattuna", html.escape(mp4).encode("utf-8") in b and urllib.parse.quote(mp4).encode() in b)
        finally:
            w.sulje()


def odota(ehto, s=5.0):
    """Odottaa, kunnes ehto() on tosi (enintaan s sekuntia); palauttaa viimeisen arvon."""
    loppu = time.monotonic() + s
    while time.monotonic() < loppu:
        if ehto():
            return True
        time.sleep(0.02)
    return bool(ehto())


def lue_syote(portti, historia=None, kehyksia=1, odotus_s=10.0, lisa=None):
    """Lukee /syote-virtaa raakana, kuten selaimen soitin: (tilakoodi, otsakkeet, kehykset). Kehys = (JPEG:n pituus, video-aika, jarjestysnumero, liput, JPEG-tavut); pituus 0 = sydankohta.
    Lopettaa, kun kuvakehyksia on `kehyksia` kpl tai odotus_s sekuntia on kulunut. lisa: (sivun tunnus, lisaviive s) tai pelkka lisaviive-merkkijono (ilman tunnusta)."""
    sock = socket.create_connection(("127.0.0.1", portti), timeout=5)
    kehykset, otsakkeet, koodi, puskuri, koko = [], None, None, b"", nw.SYOTE_OTSIKKO.size
    try:
        param = ([] if historia is None else [f"historia={historia}"]) + ([] if lisa is None else [f"id={lisa[0]}&lisa={lisa[1]}"] if isinstance(lisa, tuple) else [f"lisa={lisa}"])
        sock.sendall(("GET /syote" + ("?" + "&".join(param) if param else "") + " HTTP/1.1\r\nHost: x\r\n\r\n").encode())
        sock.settimeout(0.2)
        loppu = time.monotonic() + odotus_s
        while time.monotonic() < loppu and sum(1 for k in kehykset if k[0] > 0) < kehyksia:
            try:
                osa = sock.recv(65536)
            except socket.timeout:
                continue
            if not osa:
                break
            puskuri += osa
            if otsakkeet is None:
                i = puskuri.find(b"\r\n\r\n")
                if i < 0:
                    continue
                rivit = puskuri[:i].decode("latin-1").split("\r\n")
                koodi = int(rivit[0].split()[1])
                otsakkeet = dict((r.split(":", 1)[0].strip().lower(), r.split(":", 1)[1].strip()) for r in rivit[1:] if ":" in r)
                puskuri = puskuri[i + 4:]
            while len(puskuri) >= koko:
                n, aika, seq, liput = nw.SYOTE_OTSIKKO.unpack(puskuri[:koko])
                if len(puskuri) < koko + n:
                    break
                kehykset.append((n, aika, seq, liput, puskuri[koko:koko + n]))
                puskuri = puskuri[koko + n:]
    finally:
        sock.close()
    return koodi, otsakkeet, kehykset


def tuota_kuvia(w, t_alku, kesto_s, fps=30.0):
    """Julkaisee kuvia reaaliajassa (video-aika t_alku + i / fps). Kuvan vari koodaa video-ajan: R = sekunnit mod 50, G = sekunnin osa (1/50 s); 5:n kerrannaisina, jotta JPEG ei sotke."""
    pohja = np.zeros((90, 160, 3), np.uint8)
    mono0, i = time.monotonic(), 0
    while i / fps < kesto_s:
        viive = mono0 + i / fps - time.monotonic()
        if viive > 0:
            time.sleep(viive)
        t = t_alku + i / fps
        k = pohja.copy()
        k[:, :, 2] = (int(t) % 50) * 5
        k[:, :, 1] = int((t % 1) * 50) * 5
        w.nayta_heti(k, t)
        i += 1


def syote_aika(w, i):
    """Syotepuskurin i. kuvan video-aika (0.0, jos puskuri on tyhja: virheellinen toteutus ei saa kaataa testia)."""
    return w._syote[i][1] if w._syote else 0.0


def kuvan_aika(tavut, lahella):
    """Syotteen JPEG-kuvaan koodattu video-aika (tuota_kuvia): 50 s:n jaksosta se, joka on lahinna arvoa `lahella`; (aika, kuvan koko)."""
    import cv2
    d = cv2.imdecode(np.frombuffer(tavut, np.uint8), cv2.IMREAD_COLOR)
    perus = round(int(d[5, 5, 2]) / 5) + round(int(d[5, 5, 1]) / 5) / 50.0
    return perus + 50 * round((lahella - perus) / 50.0), d.shape


def testaa_web_syote():
    print("selainnakyman aikaleimattu syote (/syote: soittimen puskuri ja lisaviive selaimessa)")
    vanhat = (nw.SYOTE_S, nw.SYOTE_TAVUT_MAX, nw.SYOTE_ODOTUS_S, nw.SYOTE_SYDAN_S)
    w = w2 = w3 = w4 = None
    try:
        w = nw.WebNaytto(0, None, leveys=160, fps=10, laatu=95, tila=lambda: dict(lahde="testi", viive_s=10.0))
        p = w.portti
        tila = json.loads(pyyda(p, "/tila")[2])
        tarkista("/tila: syotteen pituus, lisaviiveen ylaraja (30 s), kuvataajuus ja perusviive", tila["syote_s"] == nw.SYOTE_S and tila["lisaviive_max_s"] == nw.LISAVIIVE_MAX_S == 30 and tila["syote_fps"] == 10 and tila["viive_s"] == 10.0, str(tila))
        tarkista("syotteen puskuri kattaa lisaviiveen ylarajan ja uudelleenyhteyden varan (soitin pyytaa historiaa lisaviive + 8 s)", nw.SYOTE_S >= nw.LISAVIIVE_MAX_S + 8)
        sivu = pyyda(p, "/")[2].decode("utf-8")
        tarkista("sivu: soitin (canvas), viivenapit -5 / -1 / +1 / +5, syoteyhteys ja MJPEG-varapolku", all(x in sivu for x in ('<canvas id="soitin"', 'id="viiveohjain"', 'data-d="-5"', 'data-d="-1"', 'data-d="1"', 'data-d="5"', '"syote?historia="', "video.mjpg", "createImageBitmap")))
        s, h, b = pyyda(p, "/syote?historia=3", metodi="HEAD")
        tarkista("HEAD /syote: otsakkeet ilman sisaltoa (virtaa ei avata)", s == 200 and b == b"" and h["Content-Type"] == "application/octet-stream" and w._syote_katsojia == 0)
        s, h, b = pyyda(p, "/syote?historia=abc")
        tarkista("historia roskaa -> 400 ja JSON-virhe", s == 400 and "historia" in json.loads(b)["virhe"], str(b))

        tuota_kuvia(w, 90.0, 0.5)
        tarkista("ilman katsojaa kuvia ei koodata eika koodaajasaiettakaan kaynnisteta (ei kuormaa, kun kukaan ei katso)", len(w._syote) == 0 and w._syote_saie is None)

        # virta: 30 kuvaa/s -> kuvataajuus 10 (harvennus video-ajan ja seinakellon mukaan), kuvat oikeassa jarjestyksessa ja oikealla aikaleimalla
        tulos1 = []
        lukija = threading.Thread(target=lambda: tulos1.append(lue_syote(p, 0, kehyksia=10 ** 6, odotus_s=3.6)), daemon=True)
        lukija.start()
        tarkista("katsoja lasketaan (syotteen katsoja ja tilarivin katsojamaara)", odota(lambda: w._syote_katsojia == 1) and w.tila()["katsojia"] == 1)
        tuottaja = threading.Thread(target=tuota_kuvia, args=(w, 100.0, 9.0), daemon=True)
        tuottaja.start()
        lukija.join(10)
        koodi, o, k = tulos1[0]
        kuvat = [x for x in k if x[0] > 0]
        tarkista("virta: 200, binaarivirta, ei valimuistia", koodi == 200 and o["content-type"] == "application/octet-stream" and "no-store" in o["cache-control"], str(o))
        tarkista("virta: kuvia 6-10 / s (harvennus 10 kuvaan sekunnissa, 30 kuvan lahteesta)", 21 <= len(kuvat) <= 40, str(len(kuvat)))
        vali = [b[1] - a[1] for a, b in zip(kuvat, kuvat[1:])]
        tarkista("virta: video-ajat kasvavat, kuvien vali ei alle 1 / fps - 0,4 * lahteen kuvavali", bool(vali) and min(vali) >= 0.1 - 0.4 / 30 - 1e-6 and max(vali) < 0.35, f"{min(vali):.3f} ... {max(vali):.3f}")
        tarkista("virta: jarjestysnumerot perakkaisia (ei hukattuja kuvia nopealle katsojalle)", bool(kuvat) and [x[2] for x in kuvat] == list(range(kuvat[0][2], kuvat[0][2] + len(kuvat))))
        tarkista("virta: ensimmainen kuva on historiaa (liput 1), tuoreet 0", bool(kuvat) and kuvat[0][3] == 1 and all(x[3] == 0 for x in kuvat[1:]))
        tarkista("virta: JPEG-kuvat ehjia ja oikean kokoisia, ja kuvaan koodattu aika = otsakkeen video-aika", all(x[4][:2] == b"\xff\xd8" for x in kuvat) and all(abs(x[1] - kuvan_aika(x[4], x[1])[0]) < 0.045 and kuvan_aika(x[4], x[1])[1] == (90, 160, 3) for x in kuvat[::4]))

        # historia: heti alkuun aiempia kuvia
        odota(lambda: syote_aika(w, -1) >= 105.0, 10)
        uusin = syote_aika(w, -1)
        _, _, k2 = lue_syote(p, 2, kehyksia=10 ** 6, odotus_s=0.6)
        hist = [x for x in k2 if x[0] > 0 and x[3] & 1]
        elava = [x for x in k2 if x[0] > 0 and not x[3] & 1]
        tarkista("historia=2: heti alkuun noin 2 s aiempia kuvia (liput 1), jarjestyksessa", 17 <= len(hist) <= 25 and hist[0][1] >= uusin - 2.0 - 1e-6 and hist[-1][1] <= uusin + 0.5 and all(b[1] > a[1] for a, b in zip(hist, hist[1:])), f"{len(hist)} kuvaa")
        tarkista("historian jalkeen tulee tuoreita kuvia (liput 0) sen jatkoksi", len(elava) >= 3 and bool(hist) and elava[0][1] > hist[-1][1])
        uusin = syote_aika(w, -1)
        _, _, k3 = lue_syote(p, 0, kehyksia=3, odotus_s=3.0)
        tarkista("historia=0: vain uusin kuva historiana, sen jalkeen tuoreet", len(k3) == 3 and k3[0][3] & 1 and not k3[1][3] & 1 and k3[0][1] >= uusin and k3[1][1] > k3[0][1])
        _, _, k3 = lue_syote(p, -5, kehyksia=2, odotus_s=3.0)
        tarkista("historia negatiivinen: kuten 0", len(k3) == 2 and k3[0][3] & 1 and not k3[1][3] & 1)
        _, _, k3 = lue_syote(p, None, kehyksia=2, odotus_s=3.0)
        tarkista("historia puuttuu: kuten 0", len(k3) == 2 and k3[0][3] & 1 and not k3[1][3] & 1)
        _, _, k4 = lue_syote(p, 999, kehyksia=10 ** 6, odotus_s=0.5)
        hist4 = [x for x in k4 if x[3] & 1]
        tarkista("historia yli puskurin pituuden: rajataan puskuriin (koko puskuri alusta)", len(hist4) >= 40 and hist4[0][1] <= 100.3, f"{len(hist4)} kuvaa, alku {hist4[0][1] if hist4 else None}")

        tuottaja.join(10)
        tarkista("katsojat poistuvat laskureista yhteyden katketessa", odota(lambda: w._syote_katsojia == 0 and w.tila()["katsojia"] == 0))

        # video-ajan hyppy taaksepain (uusi aikaperusta): vanha historia ei jatku uuteen
        tulos6 = []
        lukija = threading.Thread(target=lambda: tulos6.append(lue_syote(p, 0, kehyksia=10 ** 6, odotus_s=2.0)), daemon=True)
        lukija.start()
        odota(lambda: w._syote_katsojia == 1)
        tuota_kuvia(w, 50.0, 1.0)
        lukija.join(5)
        k6 = [x for x in tulos6[0][2] if x[0] > 0]
        tarkista("aikaperustan vaihto: katsoja saa myos uuden jakson kuvat (video-aika hyppaa taaksepain)", any(x[1] > 100 for x in k6) and any(x[1] < 60 for x in k6), str([round(x[1], 1) for x in k6[:3]]))
        _, _, k7 = lue_syote(p, 40, kehyksia=10 ** 6, odotus_s=0.5)
        tarkista("aikaperustan vaihto: vanha historia tyhjennetaan (uusi katsoja saa vain uuden jakson)", len([x for x in k7 if x[0] > 0]) >= 5 and all(x[1] < 60 for x in k7 if x[0] > 0))
        w.sulje()
        w = None

        # rengaspuskurin rajat: enintaan SYOTE_S s ja SYOTE_TAVUT_MAX tavua
        nw.SYOTE_S = 2.0
        w2 = nw.WebNaytto(0, None, leveys=160, fps=20, laatu=95)
        lukija = threading.Thread(target=lambda: lue_syote(w2.portti, 0, kehyksia=10 ** 6, odotus_s=4.0), daemon=True)
        lukija.start()
        odota(lambda: w2._syote_katsojia == 1)
        tuota_kuvia(w2, 10.0, 4.0)
        time.sleep(0.3)
        vali_s = syote_aika(w2, -1) - syote_aika(w2, 0)
        tarkista("rengaspuskuri: enintaan SYOTE_S s video-aikaa (muuten vanhin pois)", 1.5 <= vali_s <= nw.SYOTE_S and len(w2._syote) >= 20, f"{vali_s:.2f} s, {len(w2._syote)} kuvaa")
        nw.SYOTE_S = 40.0
        nw.SYOTE_TAVUT_MAX = 6 * (len(w2._syote[-1][2]) if w2._syote else 1000)
        tuota_kuvia(w2, 20.0, 1.5)                                  # katsoja on jo poistunut: koodaus jatkuu odotusajan (SYOTE_ODOTUS_S)
        time.sleep(0.3)
        tarkista("rengaspuskuri: enintaan SYOTE_TAVUT_MAX tavua (muistiraja), ainakin kaksi kuvaa jaa", 2 <= len(w2._syote) <= 8 and w2._syote_tavut <= nw.SYOTE_TAVUT_MAX, f"{len(w2._syote)} kuvaa, {w2._syote_tavut} / {nw.SYOTE_TAVUT_MAX} tavua")
        tarkista("rengaspuskurin tavukirjanpito tasmaa", w2._syote_tavut == sum(len(x[2]) for x in w2._syote))
        w2.sulje()
        w2 = None

        # koodaus vain katsojan ollessa paikalla (ja hetki sen jalkeen)
        nw.SYOTE_S, nw.SYOTE_TAVUT_MAX, nw.SYOTE_ODOTUS_S = vanhat[0], vanhat[1], 1.5
        w3 = nw.WebNaytto(0, None, leveys=160, fps=20, laatu=95)
        tulos8 = []
        lukija = threading.Thread(target=lambda: tulos8.append(lue_syote(w3.portti, 0, kehyksia=3, odotus_s=5)), daemon=True)
        lukija.start()
        odota(lambda: w3._syote_katsojia == 1)
        tuota_kuvia(w3, 10.0, 0.5)
        lukija.join(5)
        odota(lambda: w3._syote_katsojia == 0)
        n_ennen = len(w3._syote)
        tuota_kuvia(w3, 11.0, 0.4)
        time.sleep(0.2)
        tarkista("viimeisen katsojan poistuttua koodaus jatkuu hetken (uudelleenyhteys saa historian)", len(w3._syote) > n_ennen, f"{n_ennen} -> {len(w3._syote)}")
        while time.monotonic() < w3._syote_aktiivinen_asti + 0.2:
            time.sleep(0.05)
        n_ennen = len(w3._syote)
        tuota_kuvia(w3, 12.0, 0.4)
        time.sleep(0.2)
        tarkista("odotusajan jalkeen ilman katsojaa kuvia ei enaa koodata", len(w3._syote) == n_ennen, f"{n_ennen} -> {len(w3._syote)}")
        tulos9 = []
        lukija = threading.Thread(target=lambda: tulos9.append(lue_syote(w3.portti, 40, kehyksia=2, odotus_s=5)), daemon=True)
        lukija.start()
        odota(lambda: w3._syote_katsojia == 1)
        tuota_kuvia(w3, 20.0, 0.5)
        lukija.join(5)
        k9 = [x for x in tulos9[0][2] if x[0] > 0]
        tarkista("uusi katsoja odotusajan jalkeen: vanhentunutta historiaa ei lahteta (puskuri tyhjennetty)", len(k9) == 2 and all(x[1] >= 20.0 for x in k9), str([round(x[1], 1) for x in k9]))
        w3.sulje()
        w3 = None

        # sydankohta: tyhja syote ei ole hiljaa (katkennut yhteys huomataan)
        nw.SYOTE_SYDAN_S = 0.2
        w4 = nw.WebNaytto(0, None)
        _, _, k10 = lue_syote(w4.portti, 0, kehyksia=1, odotus_s=2.5)
        tarkista("tyhja syote: sydankohta (pituus 0) lahtee, ettei yhteys nayta kuolleelta", len(k10) >= 2 and all(x[:4] == (0, 0.0, 0, 0) for x in k10), str([x[:4] for x in k10]))
        w4.sulje()
        w4 = None
    finally:
        for x in (w, w2, w3, w4):
            if x is not None:
                x.sulje()
        nw.SYOTE_S, nw.SYOTE_TAVUT_MAX, nw.SYOTE_ODOTUS_S, nw.SYOTE_SYDAN_S = vanhat

    solmu = shutil.which("node")
    if solmu:
        skripti = re.findall(r"<script>(.*?)</script>", nw.SIVU, re.S)
        with tempfile.TemporaryDirectory() as t:
            tied = os.path.join(t, "sivu.js")
            with open(tied, "w", encoding="utf-8") as f:
                f.write("\n".join(skripti))
            r = subprocess.run([solmu, "--check", tied], capture_output=True, text=True, timeout=60)
        tarkista("sivun JavaScript on syntaktisesti kelvollista (node --check)", bool(skripti) and r.returncode == 0, r.stderr.strip()[-200:])
    else:
        print("  (node puuttuu: sivun JavaScriptin syntaksitarkistus ohitetaan)")


def testaa_web_lisaviive():
    print("selaimen lisaviive palvelimelle (/lisa, /syote id + lisa): VOD-ajon kiri tauolle, kun nakyva sivu on valinnut lisaviiveen")
    vanhat = (nw.SYOTE_LISA_ODOTUS_S, nw.SYOTE_LISA_TUNNUKSIA_MAX, nw.SYOTE_LISA_VANHENEE_S, nw.SYOTE_SYDAN_S)
    w = nw.WebNaytto(0, None, leveys=160, fps=10, laatu=70, tila=lambda: dict(lahde="testi", viive_s=10.0))
    p = w.portti

    def tila_reaaliaika():
        return json.loads(pyyda(p, "/tila")[2])["reaaliaika"]

    def lisaa(tunnus, s_):
        s_koodi, _, b = pyyda(p, f"/lisa?id={tunnus}&s={s_}")
        return s_koodi, json.loads(b)

    def yhteys(tunnus, lisa, kesto_s):
        """Syoteyhteys taustasaikeessa kesto_s sekuntia (ei kuvia: pelkka yhteys; sydankohdat huomaavat katkenneen yhteyden)."""
        t_ = threading.Thread(target=lambda: lue_syote(p, 0, kehyksia=10 ** 6, odotus_s=kesto_s, lisa=(tunnus, lisa) if tunnus else lisa), daemon=True)
        t_.start()
        return t_

    def tyhjaa():
        """Odottaa, etta kaikki sivut on unohdettu (rekisteri siivotaan laiskasti uuden yhteyden / /lisa-pyynnon yhteydessa, joten pyyntoja tehdaan odotuksen aikana)."""
        return odota(lambda: (lisaa("siivous", 0), not w._syote_lisat and w._syote_katsojia == 0)[1], 6.0)
    try:
        nw.SYOTE_LISA_ODOTUS_S, nw.SYOTE_SYDAN_S = 0.6, 0.2
        tarkista("ilman katsojia reaaliaikaa ei haluta, /tila kertoo sen", not w.reaaliaika_haluttu() and tila_reaaliaika() is False)
        koodi, vast = lisaa("sivu1", 5)
        tarkista("/lisa tuntemattomalle sivulle (ei syoteyhteytta): 200, ei rekisteroida, reaaliaikaa ei haluta", koodi == 200 and vast == {"reaaliaika": False} and not w._syote_lisat)
        for kysely in ("", "?id=abc", "?s=5", "?id=abc&s=x", "?id=abc&s=nan", "?id=a%20b&s=5", "?id=" + "a" * 41 + "&s=5", "?id=a/b&s=5", "?id=a.b&s=5"):
            tarkista(f"/lisa{kysely or ' (ei parametreja)'}: virheellinen pyynto -> 400", pyyda(p, "/lisa" + kysely)[0] == 400)
        s, _, b = pyyda(p, "/syote?lisa=abc")
        tarkista("/syote: lisa roskaa -> 400 ja JSON-virhe", s == 400 and "lisa" in json.loads(b)["virhe"], str(b))

        # sivu avaa syoteyhteyden id:lla ja lisaviiveella 5; sivu ilmoittaa muutokset /lisa-pyynnoilla
        t1 = yhteys("sivu1", 5, 5.0)
        tarkista("syoteyhteys id:lla ja lisaviiveella 5 -> reaaliaikaa halutaan (kiri tauolle), /tila kertoo", odota(lambda: w.reaaliaika_haluttu()) and tila_reaaliaika() is True)
        koodi, vast = lisaa("sivu1", 0)
        tarkista("/lisa s=0 (esim. valilehti piiloutui) -> reaaliaikaa ei enaa haluta, kiri saa jatkua", koodi == 200 and vast == {"reaaliaika": False} and not w.reaaliaika_haluttu() and tila_reaaliaika() is False)
        koodi, vast = lisaa("sivu1", 7)
        tarkista("/lisa s=7 -> reaaliaikaa halutaan taas, lisaviive tallentuu", vast == {"reaaliaika": True} and w._syote_lisat["sivu1"][0] == 7.0)
        lisaa("sivu1", 999)
        tarkista("lisaviive rajataan 0 ... LISAVIIVE_MAX_S (30 s)", w._syote_lisat["sivu1"][0] == float(nw.LISAVIIVE_MAX_S))
        koodi, vast = lisaa("sivu1", -3)
        tarkista("negatiivinen lisaviive -> 0: reaaliaikaa ei haluta", vast == {"reaaliaika": False} and w._syote_lisat["sivu1"][0] == 0.0)
        lisaa("sivu1", 4)
        s, h, b = pyyda(p, "/lisa?id=sivu1&s=0", metodi="HEAD")
        tarkista("HEAD /lisa ei muuta mitaan", s == 200 and b == b"" and w.reaaliaika_haluttu() and w._syote_lisat["sivu1"][0] == 4.0)

        # useampi sivu: kiri on tauolla, kun yksikin sivu kayttaa lisaviivetta
        t2 = yhteys("sivu2", 0, 5.0)
        odota(lambda: "sivu2" in w._syote_lisat)
        lisaa("sivu1", 0)
        tarkista("sivu 2 (lisaviive 0) ja sivu 1 (0): reaaliaikaa ei haluta", not w.reaaliaika_haluttu())
        lisaa("sivu2", 3)
        tarkista("vain sivulla 2 on lisaviive -> reaaliaikaa halutaan", w.reaaliaika_haluttu())
        lisaa("sivu1", 6)
        lisaa("sivu2", 0)
        tarkista("sivu 1 jatkaa lisaviiveella, vaikka sivu 2 vapautuu -> reaaliaikaa halutaan edelleen", w.reaaliaika_haluttu())
        lisaa("sivu1", 0)
        tarkista("molemmat 0 -> ei reaaliaikaa", not w.reaaliaika_haluttu())
        t1.join(10)
        t2.join(10)
        tarkista("sivut poistuivat (yhteydet sulkeutuivat ja odotusaika kului)", tyhjaa())

        # yhteyden katketessa lisaviive pysyy hetken voimassa (uudelleenyhteys ei vapauta kiria), sitten sivu unohdetaan
        t3 = yhteys("sivu3", 5, 1.5)
        odota(lambda: w.reaaliaika_haluttu())
        t3.join(5)
        tarkista("syoteyhteys sulkeutui: lisaviive pysyy voimassa SYOTE_LISA_ODOTUS_S s (uudelleenyhteys ei vapauta kiria)", odota(lambda: w._syote_lisat.get("sivu3", [0, 1])[1] == 0) and w.reaaliaika_haluttu())
        tarkista("odotusajan jalkeen reaaliaikaa ei haluta", odota(lambda: not w.reaaliaika_haluttu(), 3.0))
        koodi, vast = lisaa("sivu3", 5)
        tarkista("katkennut sivu unohdetaan (/lisa ei herata sita henkiin, rekisteri siivotaan)", vast == {"reaaliaika": False} and "sivu3" not in w._syote_lisat)

        # sama sivu kahdella yhteydella (uudelleenyhteys ehtii ennen vanhan sulkeutumista): vanhan sulkeutuminen ei pura uuden lisaviivetta
        ta = yhteys("sivu4", 5, 1.5)
        odota(lambda: w.reaaliaika_haluttu())
        tb = yhteys("sivu4", 5, 6.0)
        odota(lambda: w._syote_lisat.get("sivu4", [0, 0])[1] == 2)
        ta.join(5)
        odota(lambda: w._syote_lisat.get("sivu4", [0, 0])[1] == 1)
        time.sleep(nw.SYOTE_LISA_ODOTUS_S + 0.4)
        lisaa("sivu4", 5)
        tarkista("sama sivu kahdella yhteydella: vanhan sulkeutuminen ei vapauta kiria, kun uusi yhteys on auki", w.reaaliaika_haluttu() and w._syote_lisat["sivu4"][1] == 1 and w._syote_lisat["sivu4"][2] is None)
        tb.join(10)
        tyhjaa()

        # lisaviive vanhenee, ellei sivu vahvista sitä (hiljaa kuollut yhteys, jaadytetty valilehti): kiri ei jaa tauolle
        nw.SYOTE_LISA_VANHENEE_S = 0.9
        t5 = yhteys("sivu5", 5, 5.0)
        tarkista("vahvistamaton lisaviive on voimassa heti", odota(lambda: w.reaaliaika_haluttu()))
        for _ in range(4):
            time.sleep(0.4)
            lisaa("sivu5", 5)
        tarkista("sivu vahvistaa lisaviiveensa (/lisa) -> voimassa pitkaan, vaikka yhteys on auki yli vanenemisajan", w.reaaliaika_haluttu())
        tarkista("vahvistus loppuu -> lisaviive vanhenee SYOTE_LISA_VANHENEE_S s:ssa (yhteys vielä auki), kiri vapautuu", odota(lambda: not w.reaaliaika_haluttu(), 3.0) and w._syote_lisat["sivu5"][1] == 1)
        tarkista("vanhentunut lisaviive tulee voimaan vasta uudella vahvistuksella", (lisaa("sivu5", 5)[1] == {"reaaliaika": True}) and w.reaaliaika_haluttu())
        t5.join(10)
        tyhjaa()
        nw.SYOTE_LISA_VANHENEE_S = vanhat[2]

        # ilman tunnusta sivu ei ilmoita lisaviivettaan; tunnusten enimmaismaara
        tyhjaa()
        tc = yhteys(None, 5, 1.2)
        time.sleep(0.5)
        tarkista("syoteyhteys ilman id:ta: lisaviivetta ei rekisteroida (reaaliaikaa ei haluta)", not w.reaaliaika_haluttu() and not w._syote_lisat, str(w._syote_lisat))
        tc.join(4)
        nw.SYOTE_LISA_TUNNUKSIA_MAX = 2
        saikeet = [yhteys(f"tunnus{i}", 5, 2.0) for i in range(3)]
        time.sleep(0.8)
        tarkista("tunnusten enimmaismaara (muistin suoja): kolmatta sivua ei rekisteroida", len(w._syote_lisat) == 2, str(sorted(w._syote_lisat)))
        for t_ in saikeet:
            t_.join(6)
    finally:
        nw.SYOTE_LISA_ODOTUS_S, nw.SYOTE_LISA_TUNNUKSIA_MAX, nw.SYOTE_LISA_VANHENEE_S, nw.SYOTE_SYDAN_S = vanhat
        w.sulje()
    tarkista("katsojat poistuivat laskureista", odota(lambda: w._syote_katsojia == 0, 3.0))
    sivu = nw.SIVU
    tarkista("sivu: lisaviive ilmoitetaan palvelimelle (lisa?id=, syoteyhteyden id ja lisa), valilehden piiloutuminen = 0",
             all(x in sivu for x in ('"lisa?id="', '"&id="', '"&lisa="', "visibilitychange", "document.hidden", "soitinLahetaLisa")))
    tarkista("sivu: lisaviiveella toisto etenee 1x (tuotannon vauhtia seurataan vain nopeassa ajossa, S.nopea)", "S.kiri&&S.nopea" in sivu and "S.nopea=t.viive_s===null" in sivu)


HEITTO_KENTAT = ["heitto", "vari", "kivi_nro", "puoli", "paa", "kierre", "irroitus_delta_cm", "liuku_T_cm", "heitto_T_cm", "merkki_T_cm", "liuku_merkista_cm", "heitto_merkista_cm",
                 "nopeus_ennen_m_s", "nopeus_jalkeen_m_s", "liun_hidastuvuus_m_s2", "hidastuvuus_m_s2", "vakaa"]


def heitto_arvot(e, vari, k):
    """Synteettisen heiton arvot: riippuvat paasta e, pelaajasta p (4 + 4 pelaajaa: kivet 1-2, 3-4, 5-6, 7-8; punaisella p + 4), kivesta k ja kierron suunnasta s (kivi pariton = myotapaivaan,
    s = -1 (Myötä); parillinen = vastapaivaan, s = +1 (Vasta)), joten tilaston oikeellisuus voidaan laskea tasta erikseen."""
    p = (k + 1) // 2 + (4 if vari == "punainen" else 0)
    s = -1 if k % 2 else 1
    liuku_m = 2.0 * p + 0.5 * e + 0.1 * k + (3.0 if s < 0 else -3.0)
    heitto_m = liuku_m + 1.5 + 0.2 * e
    T = 5.0 + k
    return dict(p=p, s=s, liuku_m=liuku_m, heitto_m=heitto_m, T=T, liuku_T=T + liuku_m * s, heitto_T=T + heitto_m * s, delta=heitto_m - liuku_m, kierre=-s * (2.0 + 0.1 * k), ennen=2.0 + 0.01 * p,
                tonaisy=0.002 * p + 0.001 * e + (0.004 if s < 0 else 0.0), liu_hid=0.01 * p + 0.001 * e + 0.0001 * k, vap_hid=0.05 + 0.002 * p + 0.0005 * e)


def heitto_rivit(paita=3, pois=()):
    """Synteettisen pelin heitto-CSV:n rivit (heitto_arvot): paita paata, molemmilla joukkueilla 8 kivea / paa. pois: {(paa, vari, kivi)} jaavat pois (puutteellinen paa)."""
    rivit, n = [], 0
    for e in range(1, paita + 1):
        for k in range(1, 9):
            for vari in ("keltainen", "punainen"):
                n += 1
                if (e, vari, k) in pois:
                    continue
                a = heitto_arvot(e, vari, k)
                rivit.append([n, vari, k, "vasen" if e % 2 else "oikea", e, f"{a['kierre']:.1f}", f"{a['delta']:.1f}", f"{a['liuku_T']:.1f}", f"{a['heitto_T']:.1f}", f"{a['T']:.1f}", f"{a['liuku_m']:.1f}", f"{a['heitto_m']:.1f}",
                              f"{a['ennen']:.2f}", f"{a['ennen'] + a['tonaisy']:.3f}", f"{a['liu_hid']:.4f}", f"{a['vap_hid']:.4f}", "1"])
    return rivit


def kirjoita_heitto_csv(polku, rivit, fi=False, mtime=None):
    """Heitto-CSV (fi: erotin ;, desimaalipilkku ja UTF-8 BOM, kuten suomalaisen Excelin tallentama)."""
    os.makedirs(os.path.dirname(polku), exist_ok=True)
    with open(polku, "w", newline="", encoding="utf-8-sig" if fi else "utf-8") as fh:
        w = csv.writer(fh, delimiter=";" if fi else ",")
        w.writerow(HEITTO_KENTAT + ["muu"])
        for r in rivit:
            w.writerow([x.replace(".", ",") if fi and isinstance(x, str) and i >= 5 else x for i, x in enumerate(r)] + ["x"])
    if mtime is not None:
        os.utime(polku, (mtime, mtime))
    return polku


def vapaa_portti():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    portti = s.getsockname()[1]
    s.close()
    return portti


def virheena(fn, *a, **k):
    """ValueError-viesti, jos fn nostaa sen; None, jos ei virhetta; muu poikkeus nayttaa nimensa."""
    try:
        fn(*a, **k)
    except ValueError as e:
        return str(e)
    except Exception as e:
        return "ODOTTAMATON POIKKEUS " + repr(e)
    return None


def testaa_web_tilasto():
    print("selainnakyma: pelaajatilasto (/tilasto/tiedostot, /tilasto, /tilasto.csv; tools/pelaajatilasto.py)")
    pt = nw._tilasto_moduuli()
    nyt = time.time()
    nimi_aak = "heitot_Matarmaa's SM – Hyvinkää_yt_hnTTLqHr5x8_20260101_120000.csv"             # heittomerkki, ajatusviiva, ääkköset ja välilyönnit (YouTube-otsikko)
    with tempfile.TemporaryDirectory() as t:
        kirjoita_heitto_csv(os.path.join(t, "heitot_peli.csv"), heitto_rivit(3), mtime=nyt - 300)
        kirjoita_heitto_csv(os.path.join(t, "heitot_puutteellinen.csv"), heitto_rivit(3, pois={(2, "punainen", 5)}), mtime=nyt - 200)
        kirjoita_heitto_csv(os.path.join(t, nimi_aak), heitto_rivit(2), fi=True, mtime=nyt - 100)
        kirjoita(os.path.join(t, "tyhja_heitot.csv"), ",".join(HEITTO_KENTAT) + "\n", nyt - 10)                  # pelkka otsikko (ajo juuri alkanut): listalla, laskenta antaa selkean virheen
        # ei listalle:
        kirjoita_heitto_csv(os.path.join(t, "heitot_peli_kivet.csv"), heitto_rivit(1))                           # johdetut tiedostot
        kirjoita_heitto_csv(os.path.join(t, "heitot_peli_pelaajat.csv"), heitto_rivit(1))
        kirjoita(os.path.join(t, "muu.csv"), "a,b\n1,2\n")
        kirjoita(os.path.join(t, "k.mp4"), "x")
        kirjoita_heitto_csv(os.path.join(t, ".piilo.csv"), heitto_rivit(1))
        kirjoita(os.path.join(t, "rikki.csv"), b"\xff\xfe\x00\x81\x82\x83\n1,2\n")
        kirjoita_heitto_csv(os.path.join(t, "ali", "heitot_ali.csv"), heitto_rivit(1))
        ulkona = kirjoita_heitto_csv(os.path.join(os.path.dirname(t), "ulkona_" + os.path.basename(t) + ".csv"), heitto_rivit(1))
        linkki = False
        try:
            os.symlink(ulkona, os.path.join(t, "linkki.csv"))
            linkki = True
        except OSError:
            pass
        sisalto_ennen = sorted(os.listdir(t))
        w = nw.WebNaytto(0, t)
        p = w.portti
        try:
            # ---- moduulitaso ------------------------------------------------------------------------------------------------
            lista = w.tilasto_tiedostot()
            tarkista("tilasto_tiedostot: vain heitto-CSV:t (otsikossa tilaston sarakkeet), uusin ensin; ei _kivet / _pelaajat, muuta CSV:ta, mp4:aa, piilo-, alikansio-, linkki- eika rikkinaista tiedostoa",
                     [x["nimi"] for x in lista] == ["tyhja_heitot.csv", nimi_aak, "heitot_puutteellinen.csv", "heitot_peli.csv"], str([x["nimi"] for x in lista]))
            tarkista("tilasto_tiedostot: koko ja muokkausaika mukana", all(x["koko"] > 0 and x["muutettu"] > 0 for x in lista))
            d = w.tilasto("heitot_peli.csv")
            tarkista("tilasto: oletus 4 + 4 pelaajaa: 48 heittoa, 8 riviä, ei varoituksia; ryhmat Myötä (Liuku, Heitto, Tönäisy), Vasta (sama) ja Hidastuvuus kuten Excelissa, desimaalit suureittain",
                     d["tiedosto"] == "heitot_peli.csv" and d["heittoja"] == 48 and len(d["rivit"]) == 8 and d["varoitukset"] == [] and d["hidastuvuus"] == "liuku" and d["desimaalit"] == [1, 1, 3, 1, 1, 3, 3]
                     and d["ryhmat"] == [dict(nimi="Myötä", suureet=["Liuku", "Heitto", "Tönäisy"]), dict(nimi="Vasta", suureet=["Liuku", "Heitto", "Tönäisy"]), dict(nimi="", suureet=["Hidastuvuus"])], str({k: d[k] for k in ("heittoja", "varoitukset", "ryhmat", "desimaalit")}))
            tarkista("tilasto: joukkueet, pelaajamaarat ja kivijaot (kivet 1-2, 3-4, 5-6, 7-8)", d["joukkueet"] == [dict(vari=v, pelaajia=4, kivet=["1-2", "3-4", "5-6", "7-8"]) for v in ("keltainen", "punainen")], str(d["joukkueet"]))
            tarkista("tilasto: vastaus on JSON-kelpoinen (tuplat listoiksi)", json.loads(json.dumps(d))["rivit"][0]["joukkue"] == "keltainen")
            r2 = d["rivit"][1]                                                                      # keltainen pelaaja 2: kivet 3 (pariton: Myötä) ja 4 (parillinen: Vasta), kolmessa paassa
            odota = lambda k, avain, suure: [heitto_arvot(e, "keltainen", k)[avain] for e in (1, 2, 3)]       # noqa: E731
            tarkista("tilasto: keltainen pelaaja 2 (kivet 3-4, 6 heittoa, 3 Myötä + 3 Vasta): Myötä-Liuku, Vasta-Heitto, Myötä-Tönäisy ja Hidastuvuus (liukuvaiheen) Mediaani ja STD (n - 1) samat kuin statistics-moduulilla erikseen laskettuna",
                     (r2["joukkue"], r2["pelaaja"], r2["kivet"], r2["heittoja"], r2["myota"], r2["vasta"]) == ("keltainen", 2, "3-4", 6, 3, 3)
                     and all(abs(r2["mediaani"][i] - statistics.median(v)) < 1e-9 and abs(r2["std"][i] - statistics.stdev(v)) < 1e-9 for i, v in ((0, odota(3, "liuku_m", 0)), (4, odota(4, "heitto_m", 0)), (2, odota(3, "tonaisy", 0)),
                                                                                                                                             (6, odota(3, "liu_hid", 0) + odota(4, "liu_hid", 0)))) and list(r2["n"]) == [3, 3, 3, 3, 3, 3, 6], str(r2))
            tarkista("tilasto: Myötä ja Vasta eri luvut (kierron suunta erottaa heitot)", r2["mediaani"][0] != r2["mediaani"][3] and r2["mediaani"][2] != r2["mediaani"][5])
            dv = w.tilasto("heitot_peli.csv", hidastuvuus="vapaa")
            tarkista("tilasto: hidastuvuus='vapaa' -> Hidastuvuus = hidastuvuus_m_s2 (kuten paneelissa); muut luvut samat", dv["hidastuvuus"] == "vapaa" and abs(dv["rivit"][1]["mediaani"][6] - statistics.median(odota(3, "vap_hid", 0) + odota(4, "vap_hid", 0))) < 1e-9
                     and dv["rivit"][1]["mediaani"][:6] == r2["mediaani"][:6] and dv["rivit"][1]["mediaani"][6] != r2["mediaani"][6])
            tarkista("tilasto: virheellinen hidastuvuus -> ValueError", "hidastuvuus" in (virheena(w.tilasto, "heitot_peli.csv", hidastuvuus="x") or ""))
            d2 = w.tilasto("heitot_peli.csv", keltainen="3", punainen="2,3,3")
            tarkista("tilasto: keltainen 3 pelaajaa (kivet 1-3, 4-6, 7-8) ja punaisella oma kivijako 2,3,3 (1-2, 3-5, 6-8)", [(j["vari"], j["pelaajia"], j["kivet"]) for j in d2["joukkueet"]]
                     == [("keltainen", 3, ["1-3", "4-6", "7-8"]), ("punainen", 3, ["1-2", "3-5", "6-8"])] and len(d2["rivit"]) == 6, str(d2["joukkueet"]))
            d3, d4 = w.tilasto("heitot_puutteellinen.csv"), w.tilasto("heitot_puutteellinen.csv", ohita=True)
            tarkista("tilasto: puutteellinen paa -> varoitus nimeaa paan ja joukkueen (puuttuvan kiven pelaajalla 5 heittoa, muilla 6); ohita=True jattaa paan pois kaikilta (4 heittoa pelaajaa kohti), varoitus silti",
                     len(d3["varoitukset"]) == 1 and "paa 2" in d3["varoitukset"][0] and "punainen 7 / 8" in d3["varoitukset"][0] and sorted({r["heittoja"] for r in d3["rivit"]}) == [5, 6]
                     and all(r["heittoja"] == 4 for r in d4["rivit"]) and len(d4["varoitukset"]) == 1, str(d3["varoitukset"]))
            d5 = w.tilasto("heitot_peli.csv", keltainen="2,2")
            tarkista("tilasto: kivijako, jonka kivet eivat ole yhteensa 8 (2,2) -> huomautus ensimmaisena", d5["varoitukset"] and "kivijako 2,2 on yhteensa 4 kivea" in d5["varoitukset"][0], str(d5["varoitukset"][:2]))
            d6 = w.tilasto(nimi_aak)
            odotettu_fi = [heitto_arvot(e, "keltainen", 1)["liuku_m"] for e in (1, 2)]               # keltainen pelaaja 1, kivi 1 (Myötä), kaksi paata
            tarkista("tilasto: heittomerkki, ajatusviiva ja ääkköset nimessa; puolipisteerotin, desimaalipilkut ja BOM luetaan oikein (2 paata = 32 heittoa)", d6["heittoja"] == 32 and d6["tiedosto"] == nimi_aak
                     and abs(d6["rivit"][0]["mediaani"][0] - statistics.median(odotettu_fi)) < 1e-9 and list(d6["rivit"][0]["n"]) == [2, 2, 2, 2, 2, 2, 4], str(d6["rivit"][0]))
            nimi, data = w.tilasto_csv("heitot_peli.csv", keltainen="4", punainen="3")
            rivit_fi = list(csv.reader(io.StringIO(data.decode("utf-8-sig")), delimiter=";"))
            tarkista("tilasto_csv (fi): tiedostonimi <syote>_pelaajat.csv, UTF-8 BOM, erotin ;, desimaalipilkku; kolme otsikkoriviä (Myötä / Vasta, Liuku Heitto Tönäisy ... Hidastuvuus, Mediaani STD) + 7 pelaajaa", nimi == "heitot_peli_pelaajat.csv" and data.startswith(b"\xef\xbb\xbf")
                     and rivit_fi[0][6] == "Myötä" and rivit_fi[0][12] == "Vasta" and rivit_fi[1][6:20] == ["Liuku", "", "Heitto", "", "Tönäisy", "", "Liuku", "", "Heitto", "", "Tönäisy", "", "Hidastuvuus", ""] and rivit_fi[2][6:20] == ["Mediaani", "STD"] * 7
                     and len(rivit_fi) == 10 and "," in rivit_fi[3][6] and "." not in rivit_fi[3][6], str(rivit_fi[:4]))
            d7 = w.tilasto("heitot_peli.csv", keltainen="4", punainen="3")
            tarkista("tilasto_csv: arvot samat kuin ruudulla (kahden desimaalin tarkkuudella)", abs(float(rivit_fi[3][6].replace(",", ".")) - round(d7["rivit"][0]["mediaani"][0], 2)) < 1e-9)
            nimi3, data3 = w.tilasto_csv("heitot_peli.csv", hidastuvuus="vapaa")
            tarkista("tilasto_csv: hidastuvuus='vapaa' vaihtaa vain Hidastuvuus-sarakkeet", data3 != w.tilasto_csv("heitot_peli.csv")[1] and nimi3 == "heitot_peli_pelaajat.csv")
            nimi2, data2 = w.tilasto_csv(nimi_aak, fi=False)
            rivit_en = list(csv.reader(io.StringIO(data2.decode("utf-8")), delimiter=","))
            tarkista("tilasto_csv (fi=False): ei BOM:ia, pilkkuerotin ja desimaalipiste; nimi sailyy (heittomerkki, ääkköset)", nimi2 == nimi_aak[:-4] + "_pelaajat.csv" and not data2.startswith(b"\xef\xbb\xbf") and "." in rivit_en[3][6] and len(rivit_en) == 11, str(rivit_en[3]))
            # virheelliset syotteet
            huonot = ["", None, "ei_ole.csv", "../" + os.path.basename(ulkona), "..\\x.csv", "/etc/passwd", "/etc/passwd.csv", os.path.join(t, "heitot_peli.csv"), "ali/heitot_ali.csv", ".piilo.csv", "k.mp4", "heitot_peli.csv\0"]
            if linkki:
                huonot.append("linkki.csv")
            viestit = {n: virheena(w.tilasto, n) for n in huonot}
            tarkista("tilasto: polkuerottimet, ylahakemisto, absoluuttinen polku, alikansio, piilotiedosto, linkki, ei-CSV, puuttuva ja tyhja nimi -> ValueError, eika tiedostoa luettu (ei vuoda ulkopuolisen sisaltoa)",
                     all(v and "löydy" in v for v in viestit.values()), str({k: v for k, v in viestit.items() if not (v and "löydy" in v)}))
            tarkista("tilasto: tiedosto ilman tilaston sarakkeita -> selkea virhe sarakkeiden nimilla", "puuttuvat sarakkeet" in (virheena(w.tilasto, "muu.csv") or "") and "liuku_T_cm" in (virheena(w.tilasto, "muu.csv") or ""))
            tarkista("tilasto: pelkka otsikko (ei heittoja) -> selkea virhe", "ei ole heittoja" in (virheena(w.tilasto, "tyhja_heitot.csv") or ""))
            v_rikki = virheena(w.tilasto, "rikki.csv") or ""
            tarkista("tilasto: ei-UTF-8-tiedosto -> selkea virhe (ei teknista UnicodeDecodeErroria)", "ei voi lukea" in v_rikki and "codec" not in v_rikki, v_rikki)
            for maara in ("0", "9", "a", "3,0,5", "-1"):
                v = virheena(w.tilasto, "heitot_peli.csv", keltainen=maara)
                tarkista(f"tilasto: virheellinen pelaajamaara {maara!r} -> ValueError, viestissa joukkue", v is not None and "joukkue 'keltainen'" in v, str(v))
            w0 = nw.WebNaytto(0, None)
            try:
                tarkista("tilasto: ilman ulos-kansiota ei listaa eika laske", w0.tilasto_tiedostot() == [] and virheena(w0.tilasto, "heitot_peli.csv") is not None)
            finally:
                w0.sulje()
            tarkista("tilasto ei kirjoita mitaan levylle (kansion sisalto muuttumaton laskujen jalkeen)", sorted(os.listdir(t)) == sisalto_ennen, str(sorted(os.listdir(t))))

            # ---- HTTP ---------------------------------------------------------------------------------------------------
            s, h, b = pyyda(p, "/tilasto/tiedostot")
            tarkista("/tilasto/tiedostot: 200, JSON ja sama lista kuin moduulissa (aakkosnimi sailyy)", s == 200 and h["Content-Type"].startswith("application/json") and [x["nimi"] for x in json.loads(b)] == [x["nimi"] for x in lista], str(b[:200]))
            s, h, b = pyyda(p, "/tilasto?tiedosto=heitot_peli.csv&kel=4&pun=3")
            hd = json.loads(b)
            tarkista("/tilasto: 200, 7 pelaajaa (4 + 3), joukkuetiedot, ei varoituksia", s == 200 and len(hd["rivit"]) == 7 and [(j["vari"], j["pelaajia"]) for j in hd["joukkueet"]] == [("keltainen", 4), ("punainen", 3)] and hd["varoitukset"] == [] and hd["heittoja"] == 48, str(hd)[:200])
            tarkista("/tilasto: sama tulos kuin moduulifunktiolla", hd == json.loads(json.dumps(w.tilasto("heitot_peli.csv", keltainen="4", punainen="3"))))
            s, h, b = pyyda(p, "/tilasto?tiedosto=" + urllib.parse.quote(nimi_aak))
            tarkista("/tilasto: aakkos- ja heittomerkkinimi koodattuna osoitteessa; tyhjat kel / pun -> oletus 4 + 4", s == 200 and json.loads(b)["heittoja"] == 32 and len(json.loads(b)["rivit"]) == 8 and pyyda(p, "/tilasto?tiedosto=heitot_peli.csv&kel=&pun=")[0] == 200)
            hp = json.loads(pyyda(p, "/tilasto?tiedosto=heitot_puutteellinen.csv")[2])
            ho = json.loads(pyyda(p, "/tilasto?tiedosto=heitot_puutteellinen.csv&ohita=1")[2])
            tarkista("/tilasto: puutteellinen paa -> varoitus; &ohita=1 -> paa pois (4 heittoa pelaajaa kohti)", len(hp["varoitukset"]) == 1 and all(r["heittoja"] == 4 for r in ho["rivit"]), str(hp["varoitukset"]))
            hv = json.loads(pyyda(p, "/tilasto?tiedosto=heitot_peli.csv&hid=vapaa")[2])
            hl = json.loads(pyyda(p, "/tilasto?tiedosto=heitot_peli.csv&hid=liuku")[2])
            tarkista("/tilasto: &hid=vapaa -> Hidastuvuus = vapaan lennon hidastuvuus (muut luvut samat); &hid=liuku ja ilman parametria = liukuvaiheen", hv["hidastuvuus"] == "vapaa" and hv["rivit"][1]["mediaani"][6] != hl["rivit"][1]["mediaani"][6] and hv["rivit"][1]["mediaani"][:6] == hl["rivit"][1]["mediaani"][:6]
                     and hl == json.loads(pyyda(p, "/tilasto?tiedosto=heitot_peli.csv")[2]) and hv == json.loads(json.dumps(w.tilasto("heitot_peli.csv", hidastuvuus="vapaa"))))
            sh, hh, bh = pyyda(p, "/tilasto?tiedosto=heitot_peli.csv&hid=x")
            sh2, _, bh2 = pyyda(p, "/tilasto.csv?tiedosto=heitot_peli.csv&hid=x")
            tarkista("/tilasto ja /tilasto.csv: virheellinen &hid -> 400 + JSON {virhe}", sh == 400 and sh2 == 400 and "hidastuvuus" in json.loads(bh)["virhe"] and hh["Content-Type"].startswith("application/json"), str(bh))
            s, h, b = pyyda(p, "/tilasto.csv?tiedosto=heitot_peli.csv&kel=4&pun=3&fi=1")
            tarkista("/tilasto.csv?fi=1: 200, text/csv, liitteena (filename*), BOM + ; + desimaalipilkku", s == 200 and h["Content-Type"] == "text/csv; charset=utf-8" and h["Content-Disposition"] == "attachment; filename*=UTF-8''heitot_peli_pelaajat.csv" and b == data, str(h))
            s, h, b = pyyda(p, "/tilasto.csv?tiedosto=" + urllib.parse.quote(nimi_aak) + "&fi=0")
            tarkista("/tilasto.csv?hid=vapaa: CSV-lataus noudattaa valintaa", pyyda(p, "/tilasto.csv?tiedosto=heitot_peli.csv&kel=4&pun=3&fi=1&hid=vapaa")[2] != data and pyyda(p, "/tilasto.csv?tiedosto=heitot_peli.csv&kel=4&pun=3&fi=1&hid=liuku")[2] == data)
            tarkista("/tilasto.csv?fi=0: pilkkuerotin, desimaalipiste, ei BOM:ia; aakkosnimi Content-Dispositionissa koodattuna", s == 200 and not b.startswith(b"\xef\xbb\xbf") and b == data2 and urllib.parse.quote(nimi_aak[:-4] + "_pelaajat.csv") in h["Content-Disposition"], str(h.get("Content-Disposition")))
            kielletyt = ["/tilasto", "/tilasto?tiedosto=", "/tilasto?tiedosto=ei_ole.csv", "/tilasto?tiedosto=..%2f" + urllib.parse.quote(os.path.basename(ulkona)), "/tilasto?tiedosto=%2e%2e%2f%2e%2e%2fetc%2fpasswd", "/tilasto?tiedosto=%2fetc%2fpasswd",
                         "/tilasto?tiedosto=" + urllib.parse.quote(os.path.join(t, "heitot_peli.csv")), "/tilasto?tiedosto=ali%2fheitot_ali.csv", "/tilasto?tiedosto=.piilo.csv", "/tilasto?tiedosto=k.mp4", "/tilasto?tiedosto=heitot_peli.csv%00",
                         "/tilasto?tiedosto=heitot_peli.csv&kel=9", "/tilasto?tiedosto=heitot_peli.csv&pun=a", "/tilasto?tiedosto=muu.csv", "/tilasto?tiedosto=rikki.csv"]
            if linkki:
                kielletyt.append("/tilasto?tiedosto=linkki.csv")
            ongelmat = []
            for polku in kielletyt:
                for tarkenne in ("", ".csv"):
                    s, h, b = pyyda(p, polku.replace("/tilasto", "/tilasto" + tarkenne, 1))
                    if not (s == 400 and h["Content-Type"].startswith("application/json") and "virhe" in json.loads(b)):
                        ongelmat.append((polku, tarkenne, s))
            tarkista("/tilasto ja /tilasto.csv: virheelliset pyynnot (puuttuva / ulkopuolinen / piilo- / linkki- / ei-CSV -tiedosto, huono pelaajamaara) -> 400 + JSON {virhe} (ei CSV-latausta, ei ulkopuolisen tiedoston sisaltoa)", not ongelmat, str(ongelmat[:5]))
            s, h, b = pyyda(p, "/tilasto/tiedostot", metodi="HEAD")
            s2, h2, b2 = pyyda(p, "/tilasto?tiedosto=heitot_peli.csv", metodi="HEAD")
            tarkista("HEAD: otsakkeet ilman sisaltoa", s == 200 and b == b"" and int(h["Content-Length"]) > 2 and s2 == 200 and b2 == b"" and int(h2["Content-Length"]) > 100)
            tarkista("POST ei tuettu (ei laskentaa eika kirjoitusta)", pyyda(p, "/tilasto?tiedosto=heitot_peli.csv", metodi="POST")[0] in (501, 405))
            tarkista("HTTP-kutsut eivat kirjoita levylle", sorted(os.listdir(t)) == sisalto_ennen, str(sorted(os.listdir(t))))
            # sivu: nappi, valitsin ja ikkuna; tiedostonimi ei saa paasta HTML:aan merkkijonona attribuuttiin (heittomerkki rikkoisi value-attribuutin)
            sivu = pyyda(p, "/")[2].decode("utf-8")
            tarkista("sivu: Pelaajatilasto-nappi, ikkuna, tiedostovalitsin, joukkueiden pelaajakentat, vajaiden paiden valinta, Laske ja Sulje", all(f'id="{x}"' in sivu for x in ("ts-avaa", "tilasto", "ts-tiedosto", "ts-kel", "ts-pun", "ts-hid", "ts-ohita", "ts-laske", "ts-tulos", "ts-sulje")) and ">Pelaajatilasto<" in sivu)
            tarkista("sivu: Hidastuvuus-valitsin (liukuvaiheen / vapaan lennon), valinta kulkee kyselyssa (&hid=) ja muistetaan selaimessa", 'value="liuku"' in sivu and 'value="vapaa"' in sivu and '"&hid="' in sivu and 'tsMuisti("ts_hid"' in sivu)
            tarkista("sivu: taulukko rakennetaan palvelimen ryhmista (Myötä / Vasta / Hidastuvuus), Mediaani ja STD kustakin suureesta, heittomaarat n kunkin luvun title-vihjeena; leikepoyta samoilla kolmella otsikkorivilla",
                     "d.ryhmat" in sivu and "<th>Mediaani</th><th>STD</th>" in sivu and "n myötä" in sivu and "n vasta" in sivu and "title='n = " in sivu and 'c.push("Mediaani","STD")' in sivu and "r.mediaani[i]" in sivu and "d.otsikot" not in sivu)
            tarkista("sivu: kutsuu /tilasto/tiedostot ja /tilasto, kopiointi Exceliin ja CSV-lataus (fi=1)", "tilasto/tiedostot" in sivu and '"tilasto?"' in sivu and "tsKopioi" in sivu and "&fi=1" in sivu and "tilasto.csv?" in sivu)
            tarkista("sivu: esc() muuttaa myos lainausmerkit; valitsimen option-elementit ja CSV-linkki rakennetaan DOM:illa, ei merkkijonoattribuutilla (tiedostonimen heittomerkki ei riko value / href)",
                     '.replace(/"/g,"&quot;").replace(/\'/g,"&#39;")' in sivu and 'createElement("option")' in sivu and "<option value='" not in sivu and '$("ts-csv").href=' in sivu and "href='tilasto.csv" not in sivu)
            tarkista("sivu: joutilastila (ei analyysia): kuva piiloon ja virta pois, status kertoo asian", 't.joutilas' in sivu and "Ei analyysia käynnissä" in sivu)
            # puuttuva tools/pelaajatilasto.py (osittain paivitetty ohjelmakansio) ei kaada muuta sivua
            vanha = sys.modules.get("pelaajatilasto", None)
            sys.modules["pelaajatilasto"] = None                                                  # import nostaa ImportErrorin
            try:
                s1, _, b1 = pyyda(p, "/tilasto/tiedostot")
                s2, _, b2 = pyyda(p, "/tilasto?tiedosto=heitot_peli.csv")
                s3, _, _ = pyyda(p, "/tilasto.csv?tiedosto=heitot_peli.csv")
                ok_muu = pyyda(p, "/")[0] == 200 and pyyda(p, "/tila")[0] == 200 and pyyda(p, "/tiedostot.json")[0] == 200
            finally:
                if vanha is None:
                    del sys.modules["pelaajatilasto"]
                else:
                    sys.modules["pelaajatilasto"] = vanha
            tarkista("tools/pelaajatilasto.py puuttuu: tilasto-osoitteet 503 + selkea ohje (JSON {virhe}), muu sivu toimii", (s1, s2, s3) == (503, 503, 503) and "pelaajatilasto.py puuttuu" in json.loads(b1)["virhe"] and "virhe" in json.loads(b2) and ok_muu, str((s1, s2, s3, b1[:120])))
        finally:
            w.sulje()
        os.remove(ulkona)

        # salasana suojaa myos tilastoa
        os.environ["NETTI_WEB_KAYTTAJA"], os.environ["NETTI_WEB_SALASANA"] = "kaytt", "sala sana:1"
        try:
            w2 = nw.WebNaytto(0, t)
        finally:
            del os.environ["NETTI_WEB_KAYTTAJA"], os.environ["NETTI_WEB_SALASANA"]
        try:
            tunn = {"Authorization": "Basic " + base64.b64encode("kaytt:sala sana:1".encode()).decode()}
            osoitteet = ["/tilasto/tiedostot", "/tilasto?tiedosto=heitot_peli.csv", "/tilasto.csv?tiedosto=heitot_peli.csv"]
            tarkista("salasana asetettu: tilasto-osoitteet 401 ilman tunnuksia, 200 oikeilla", all(pyyda(w2.portti, o)[0] == 401 for o in osoitteet) and all(pyyda(w2.portti, o, tunn)[0] == 200 for o in osoitteet))
        finally:
            w2.sulje()


def testaa_joutilas():
    print("selainnakyma ajojen valissa (netti_web.JoutilasSivu) ja sen kaytto ajosilmukassa (netti_palvelin.palvele)")
    import contextlib
    import http.client
    with tempfile.TemporaryDirectory() as t:
        kirjoita_heitto_csv(os.path.join(t, "heitot_peli.csv"), heitto_rivit(2))
        kirjoita(os.path.join(t, "k.mp4"), "x")
        portti = vapaa_portti()
        viestit = []
        js = nw.JoutilasSivu(portti, t, loki=viestit.append)
        js.pysayta()
        tarkista("ennen kaynnistysta: ei avoinna eika porttia; pysayta ei tee mitaan", not js.avoinna and js.avattu_portti is None and _portti_kiinni(portti))
        tarkista("kaynnista: True, sivu avoinna annetulla portilla", js.kaynnista() is True and js.avoinna and js.avattu_portti == portti)
        try:
            s, h, b = pyyda(portti, "/")
            tarkista("sivu toimii ajojen valissa (Pelaajatilasto-nappi mukana)", s == 200 and b"ts-avaa" in b)
            tila = json.loads(pyyda(portti, "/tila")[2])
            tarkista("/tila: joutilas = True, ei kaynnissa, ei kuvia, viesti otsikkona", tila["joutilas"] is True and tila["kaynnissa"] is False and tila["kuvia"] == 0 and "valvonta" in tila["otsikko"], str(tila))
            s, h, b = pyyda(portti, "/kuva.jpg")
            tarkista("/kuva.jpg: ilmoituskuva (JPEG)", s == 200 and b[:2] == b"\xff\xd8")
            tarkista("tiedostolista ja pelaajatilasto kaytettavissa ajojen valissa", sorted(x["nimi"] for x in json.loads(pyyda(portti, "/tiedostot.json")[2])) == ["heitot_peli.csv", "k.mp4"]
                     and json.loads(pyyda(portti, "/tilasto?tiedosto=heitot_peli.csv")[2])["heittoja"] == 32)
            tarkista("kaynnista kahdesti: ei uutta palvelinta (sama portti)", js.kaynnista() is True and js.avattu_portti == portti)
            # pysyva yhteys: selain pitaa yhteyden auki ja kysyy tilaa 3 s valein; kun analyysi ottaa portin, vanha yhteys ei saa jaada vastaamaan joutilastilalla
            yhteys = http.client.HTTPConnection("127.0.0.1", portti, timeout=10)
            yhteys.request("GET", "/tila")
            r = yhteys.getresponse()
            tarkista("pysyva yhteys: ensimmainen kysely vastaa joutilastilalla", json.loads(r.read())["joutilas"] is True)
            js.pysayta()
            tarkista("pysayta: portti vapautuu ja avoimet asiakasyhteydet katkaistaan", not js.avoinna and js.avattu_portti is None and _portti_kiinni(portti))
            analyysi = nw.WebNaytto(portti, t, tila=lambda: dict(lahde="analyysi"))                       # analyysilapsi ottaa saman portin
            try:
                try:
                    yhteys.request("GET", "/tila")
                    vanha = json.loads(yhteys.getresponse().read())
                except (http.client.HTTPException, OSError):
                    vanha = None                                                                           # yhteys katkaistu: selain avaa uuden
                tarkista("vanha pysyva yhteys ei vastaa joutilastilalla sen jalkeen, kun analyysi on ottanut portin", vanha is None or vanha.get("lahde") == "analyysi", str(vanha))
                tarkista("uusi yhteys saa analyysin tilan", json.loads(pyyda(portti, "/tila")[2]).get("lahde") == "analyysi")
            finally:
                analyysi.sulje()
                yhteys.close()
        finally:
            js.pysayta()
        tarkista("pysayta uudelleen ei tee mitaan", not js.avoinna and _portti_kiinni(portti))
        tarkista("avaus uudelleen samaan porttiin heti sulkemisen jalkeen (analyysin jalkeen)", js.kaynnista() is True and pyyda(portti, "/tila")[0] == 200)
        js.pysayta()
        # portti varattu: ajoa ei saa kaataa
        raja = socket.socket()
        raja.bind(("0.0.0.0", 0))
        raja.listen(1)
        viestit.clear()
        js2 = nw.JoutilasSivu(raja.getsockname()[1], t, loki=viestit.append)
        try:
            r1, r2 = js2.kaynnista(), js2.kaynnista()
            tarkista("varattu portti: kaynnista palauttaa False eika kaada (SystemExit ei karkaa); varoitus lokiin vain kerran", r1 is False and r2 is False and not js2.avoinna and len(viestit) == 1 and "ei kaynnistynyt" in viestit[0], str(viestit))
        finally:
            raja.close()
        tarkista("portin vapauduttua seuraava kaynnistys onnistuu", js2.kaynnista() is True and js2.avoinna)
        js2.pysayta()

    class EsiJoutilas:
        """Kirjaa avaukset ja sulkemiset (vain muutokset) ajojarjestyksessa."""

        def __init__(self):
            self.auki = False
            self.tapahtumat = []

        def kaynnista(self):
            if not self.auki:
                self.auki = True
                self.tapahtumat.append("auki")
            return True

        def pysayta(self):
            if self.auki:
                self.auki = False
                self.tapahtumat.append("kiinni")

    class JoutilasPysaytys(EsiPysaytys):
        """EsiPysaytys, joka kirjaa joutilassivun tilan jokaisen odotuksen alussa."""

        def __init__(self, joutilas, **k):
            super().__init__(**k)
            self.joutilas = joutilas
            self.odotuksessa = []

        def nuku(self, s):
            self.odotuksessa.append(self.joutilas.auki)
            return super().nuku(s)

    with tempfile.TemporaryDirectory() as t:
        nyt = time.time()
        v = os.path.join(t, "video")
        for n in ("a.mp4", "b.mp4"):
            kirjoita(os.path.join(v, n), n, nyt - 100)
        ej = EsiJoutilas()
        ajossa = []

        def ajuri(cmd, env=None, cwd=None):
            ajossa.append(ej.auki)
            return 7 if lahde_komennosta(cmd)[1].endswith("b.mp4") else 0

        a = np_.Asetukset(dict(NETTI_DATA=t, NETTI_LAHDE=v, NETTI_ODOTUS_S="5", NETTI_VAKAUS_S="30"))
        ps = JoutilasPysaytys(ej, pysayta_nukkumisella=2)
        rc = np_.palvele(a, ps, ajuri, 32.0, lambda: nyt, joutilas=ej)
        tarkista("ajosilmukka (valvonta): joutilassivu EI ole auki yhdenkaan ajon aikana (analyysi avaa portin itse)", len(ajossa) >= 3 and not any(ajossa), str(ajossa))
        tarkista("ajosilmukka: sivu on jo auki, kun odotus alkaa (ajojen valissa)", len(ps.odotuksessa) == 2 and all(ps.odotuksessa), str(ps.odotuksessa))
        tarkista("ajosilmukka: sivu avataan odotuksen ajaksi ja suljetaan ennen seuraavaa ajoa (auki, kiinni, auki, kiinni)", ej.tapahtumat == ["auki", "kiinni", "auki", "kiinni"] and rc == 0, str(ej.tapahtumat))
        tarkista("ajosilmukka: lopetus sulkee sivun (portti vapautuu)", not ej.auki)

        # odotus nostaa poikkeuksen, kun sivu on auki: poikkeus nousee ylos ja sivu suljetaan silti (portti ei jaa varatuksi)
        ej2 = EsiJoutilas()

        class PoikkeusPysaytys(EsiPysaytys):
            def nuku(self, s):
                raise RuntimeError("testivirhe")
        a = np_.Asetukset(dict(NETTI_DATA=t, NETTI_LAHDE=v, NETTI_ODOTUS_S="5", NETTI_VAKAUS_S="30"))
        try:
            np_.palvele(a, PoikkeusPysaytys(), lambda cmd, env=None, cwd=None: 0, 32.0, lambda: nyt, joutilas=ej2)
            nostettu = False
        except RuntimeError:
            nostettu = True
        tarkista("ajosilmukka: poikkeus odotuksessa nousee ylos ja sivu suljetaan silti", nostettu and ej2.tapahtumat == ["auki", "kiinni"], str(ej2.tapahtumat))

        # kansiot eivat kelpaa (paluukoodi 2): sivua ei avata
        ej3 = EsiJoutilas()
        kirjoita(os.path.join(t, "tiedosto"), "x")
        a = np_.Asetukset(dict(NETTI_DATA=os.path.join(t, "tiedosto", "ali"), NETTI_LAHDE=v))
        with contextlib.redirect_stderr(io.StringIO()):
            rc3 = np_.palvele(a, EsiPysaytys(), ajuri, 32.0, lambda: nyt, joutilas=ej3)
        tarkista("ajosilmukka: kansiovirhe (paluukoodi 2) -> sivua ei avata", rc3 == 2 and ej3.tapahtumat == [], str((rc3, ej3.tapahtumat)))

        # kertaajo: sivua ei tarvita (kontti poistuu ajon jalkeen)
        ej4 = EsiJoutilas()
        a = np_.Asetukset(dict(NETTI_DATA=t, NETTI_LAHDE=v, NETTI_JATKUVA="0"))
        np_.palvele(a, EsiPysaytys(), lambda cmd, env=None, cwd=None: 0, 32.0, lambda: nyt, joutilas=ej4)
        tarkista("ajosilmukka (kertaajo): joutilassivua ei avata lainkaan", ej4.tapahtumat == [], str(ej4.tapahtumat))

        # oikea sivu oikealla portilla: odotuksen aikana vastaa, ajon aikana portti vapaa (analyysi saa sen)
        portti = vapaa_portti()
        oikea = nw.JoutilasSivu(portti, t, loki=lambda *_: None)
        ajon_portti, odotuksen_tila = [], []

        def ajuri_portti(cmd, env=None, cwd=None):
            ajon_portti.append(_portti_kiinni(portti))
            return 7 if lahde_komennosta(cmd)[1].endswith("b.mp4") else 0

        class PortinPysaytys(EsiPysaytys):
            def nuku(self, s):
                odotuksen_tila.append(pyyda(portti, "/tila")[0] == 200 and json.loads(pyyda(portti, "/tila")[2]).get("joutilas") is True)
                return super().nuku(s)
        a = np_.Asetukset(dict(NETTI_DATA=os.path.join(t, "data2"), NETTI_LAHDE=v, NETTI_ODOTUS_S="5", NETTI_VAKAUS_S="30", NETTI_WEB_PORTTI=str(portti)))          # oma datakansio: videot eivat ole jo kasiteltyja
        with contextlib.redirect_stdout(io.StringIO()):
            np_.palvele(a, PortinPysaytys(pysayta_nukkumisella=2), ajuri_portti, 32.0, lambda: nyt, joutilas=oikea)
        tarkista("ajosilmukka + oikea sivu: portti on vapaa jokaisen ajon alkaessa (analyysi voi avata sen)", ajon_portti and all(ajon_portti), str(ajon_portti))
        tarkista("ajosilmukka + oikea sivu: odotuksen aikana portissa vastaa joutilassivu", odotuksen_tila and all(odotuksen_tila), str(odotuksen_tila))
        tarkista("ajosilmukka + oikea sivu: silmukan paattyessa portti on vapaa", _portti_kiinni(portti) and not oikea.avoinna)

        # main(): joutilassivu luodaan ja annetaan silmukalle, kun selainportti on kaytossa (ei, kun NETTI_WEB_PORTTI=0)
        kaapattu = []
        vanha_palvele, vanha_pysaytys = np_.palvele, np_.Pysaytys

        class EiSignaaleja:
            def asenna(self):
                pass
        np_.palvele = lambda a, pysaytys, **k: (kaapattu.append((a, k.get("joutilas"))), 0)[1]
        np_.Pysaytys = EiSignaaleja
        env_vanha = {k: os.environ.get(k) for k in ("NETTI_DATA", "NETTI_LAHDE", "NETTI_WEB_PORTTI")}
        try:
            for web in (str(portti), "0"):
                os.environ.update(NETTI_DATA=t, NETTI_LAHDE=v, NETTI_WEB_PORTTI=web)
                with contextlib.redirect_stdout(io.StringIO()):
                    np_.main([])
        finally:
            np_.palvele, np_.Pysaytys = vanha_palvele, vanha_pysaytys
            for k, x in env_vanha.items():
                if x is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = x
        tarkista("main: web-portti asetettu -> silmukka saa JoutilasSivun (portti ja tulosten kansio asetuksista, ei vielä avattu)", len(kaapattu) == 2 and isinstance(kaapattu[0][1], nw.JoutilasSivu) and kaapattu[0][1].portti == portti
                 and kaapattu[0][1].ulos_kansio == kaapattu[0][0].ulos_kansio and not kaapattu[0][1].avoinna, str(kaapattu[:1]))
        tarkista("main: NETTI_WEB_PORTTI=0 (selainnakyma pois) -> ei joutilassivua", len(kaapattu) == 2 and kaapattu[1][1] is None)


def main():
    for t in (testaa_ymparisto, testaa_asetukset, testaa_komento, testaa_tiedostot_ja_muisti, testaa_silmukka, testaa_linkkilista, testaa_odotuksen_ylivuoto, testaa_lapsiprosessi, testaa_web, testaa_netti_live_palvelinosat,
              testaa_videon_nimi_palvelin, testaa_web_resoluutio, testaa_web_videon_nimi, testaa_web_syote, testaa_web_lisaviive, testaa_web_tilasto, testaa_joutilas):
        t()
    print()
    if VIRHEITA:
        print(f"{len(VIRHEITA)} testia epaonnistui: {', '.join(VIRHEITA)}")
        return 1
    print("kaikki testit OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
