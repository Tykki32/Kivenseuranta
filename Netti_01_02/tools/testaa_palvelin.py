"""Palvelinversion (Netti_01_02) testit: kontin resurssirajat (netti_ymparisto), asetukset ja ajosilmukka (netti_palvelin), selainnakyma (netti_web), sulkusignaali ja tilatiedosto
(netti_live) sekä kayttajan kalibrointikansio (kalibrointi_valinta).

    python tools/testaa_palvelin.py        (kontissa: docker run --rm IMAGE python /app/tools/testaa_palvelin.py)

Ei tarvitse videota, YouTubea eika verkkoa. Poistumiskoodi 1, jos jokin testi epaonnistuu."""
import base64
import collections
import contextlib
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
import types
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
                                  "NETTI_LISAVALITSIMET", "NETTI_AANI", "NETTI_KOODEKKI", "NETTI_EI_KIERRETTA", "NETTI_HILJAA", "NETTI_ODOTUS_S")}
        a = np_.Asetukset(dict(env, **tyhjat))
        tarkista("tyhjat muuttujat (Unraid valittaa ne) = ei asetettu: oletukset", a.laji == "tiedostot" and a.jatkuva and a.tallenna_video and a.csv and a.nopea and a.viive == 10.0 and a.saikeet is None
                 and a.web_portti == 63884 and a.lisa == [] and a.aani == "auto" and a.koodekki == "cpu" and not a.ei_kierretta and a.odotus_s == 60.0)
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
        tarkista("web-valitsimet ja tilatiedosto", "--web 63884" in j and "--web-leveys 1280" in j and "--tila-tiedosto /tmp/tila.json" in j)
        tarkista("tiedostolle ei --lataus eika --kiri", "--lataus" not in c and "--kiri" not in c)
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


K = "https://www.youtube.com/@Matarmaa/streams"
K2 = "https://www.youtube.com/channel/UCabc123/videos"


def vid(n, otsikko="Matarmaa peli", kesto_min=90, tila="was_live", saatavuus="public", osoite=None):
    """Kanavan luettelon video (kuten luettele_kanava palauttaa); tunnus vid0000000N on 11 merkkia."""
    tunnus = f"vid{n:08d}"
    return dict(id=tunnus, otsikko=otsikko, kesto_s=None if kesto_min is None else kesto_min * 60.0, tila=tila, saatavuus=saatavuus, osoite=osoite or f"https://www.youtube.com/watch?v={tunnus}")


def tunnukset(jono):
    """Kanavavahdin palauttamasta jonosta videoiden tunnukset."""
    return [ref.rsplit("=", 1)[1] for laji, ref in jono if laji == "youtube_lista"]


class EsiKanava:
    """Testi-luettele_kanava: palauttaa kanavan 'sivuston' tilan. aikataulu: [(sekuntia alusta, lista uusin ensin)] -- viimeinen kelvollinen lista on voimassa; virhe: heitettava poikkeus."""

    def __init__(self, listat, kello=None, alku=1_800_000_000.0):
        self.listat = listat if isinstance(listat, dict) else {K: listat}
        self.kello, self.alku = kello, alku
        self.haut = []
        self.virhe = None

    def aseta(self, lista, kanava=K):
        self.listat[kanava] = lista

    def __call__(self, osoite, n):
        self.haut.append((osoite, n))
        if self.virhe is not None:
            raise self.virhe
        lista = self.listat[osoite]
        if isinstance(lista, list) and lista and isinstance(lista[0], tuple):              # aikataulu
            kulunut = self.kello() - self.alku
            lista = [x for t0, x in lista if t0 <= kulunut][-1]
        return [dict(v) for v in lista]


@contextlib.contextmanager
def lokiin():
    puskuri = io.StringIO()
    with contextlib.redirect_stdout(puskuri):
        yield puskuri


class HiljainenVahti(np_.Kanavavahti):
    """Kanavavahti, jonka lokirivit eivat sotke testitulostetta (lokiin():n sisalla ne kerataan sinne)."""

    def uudet(self, *a, **kw):
        if isinstance(sys.stdout, io.StringIO):
            return super().uudet(*a, **kw)
        with lokiin():
            return super().uudet(*a, **kw)


def testaa_kanavat():
    print("kanavat (netti_palvelin.Kanavavahti): uusien lahetysten automaattinen ajo")
    # ---- tunnistus
    kanavat = [("https://www.youtube.com/@Matarmaa/streams", K), ("https://www.youtube.com/@Matarmaa", K), ("@Matarmaa", K), ("@Matarmaa/", K), ("  \"@Matarmaa\"  ", K),
               ("https://www.youtube.com/@Matarmaa/videos", "https://www.youtube.com/@Matarmaa/videos"), ("@Matarmaa/videos", "https://www.youtube.com/@Matarmaa/videos"),
               ("https://www.youtube.com/@Matarmaa/featured?view=2", K), ("https://m.youtube.com/@Matarmaa/streams?sort=dd#x", K), ("http://youtube.com/@Matarmaa/", K),
               ("https://www.youtube.com/channel/UCabc123/videos", K2), ("https://www.youtube.com/c/Nimi", "https://www.youtube.com/c/Nimi/streams"),
               ("https://www.youtube.com/user/Joku/videos", "https://www.youtube.com/user/Joku/videos"), ("https://www.youtube.com/@Mätärmaa", "https://www.youtube.com/@Mätärmaa/streams"),
               ("https://WWW.YouTube.com/@Matarmaa/STREAMS", K),
               ("https://www.youtube.com/@Matarmaa/live", None), ("https://www.youtube.com/channel/UCabc/live", None), ("https://www.youtube.com/watch?v=hnTTLqHr5x8", None),
               ("https://youtu.be/hnTTLqHr5x8", None), ("https://www.youtube.com/live/hnTTLqHr5x8", None), ("https://www.youtube.com/shorts/hnTTLqHr5x8", None),
               ("https://www.youtube.com/playlist?list=PL1", None), ("hnTTLqHr5x8", None), ("@", None), ("", None), ("https://www.youtube.com/@Matarmaa/streams/lisaa", None),
               ("https://example.com/@Matarmaa", None), ("rtsp://kamera.lan/x", None)]
    huonot = [(s, np_.kanava_osoite(s)) for s, odotettu in kanavat if np_.kanava_osoite(s) != odotettu]
    tarkista(f"kanava_osoite: {len(kanavat)} muotoa (@kahva, channel/, c/, user/, valilehdet, kyselyosa; video, /live, soittolista, tunnus eivat ole kanava)", not huonot, str(huonot))
    tarkista("kanavan /live-osoite on edelleen tavallinen YouTube-lahde (kanavan nykyinen suora lahetys)", np_.luokittele_rivi("https://www.youtube.com/@Matarmaa/live") == ("youtube", "https://www.youtube.com/@Matarmaa/live")
             and np_.luokittele_lahde("https://www.youtube.com/@Matarmaa/live", "/x") == ("youtube", "https://www.youtube.com/@Matarmaa/live"))
    tarkista("luokittele_rivi / luokittele_lahde: kanava", np_.luokittele_rivi("@Matarmaa") == ("kanava", K) and np_.luokittele_lahde("@Matarmaa", "/ei/ole") == ("kanava", K)
             and np_.luokittele_lahde("https://www.youtube.com/@Matarmaa", "/ei/ole") == ("kanava", K))
    tarkista("kanavan_nimi: @kahva ja channel/UC...", np_.kanavan_nimi(K) == "@Matarmaa" and np_.kanavan_nimi(K2) == "channel/UCabc123")

    # ---- asetukset
    with tempfile.TemporaryDirectory() as t:
        env = dict(NETTI_DATA=t)
        a = np_.Asetukset(env)
        tarkista("kanava-asetusten oletukset: ei kanavia, haku 5 min valein, ei vanhoja, alle 20 min pois, ei suodatinta, suorat vasta paattyneina",
                 a.kanavat == [] and a.kanava_vali_min == 5.0 and a.kanava_vanhat == 0 and a.kanava_min_kesto_min == 20.0 and a.kanava_suodatin is None and not a.kanava_live)
        a = np_.Asetukset(dict(env, NETTI_KANAVAT="@Matarmaa, https://www.youtube.com/channel/UCabc123/videos  @Matarmaa/streams\n@Toinen;", NETTI_KANAVA_VALI_MIN="2,5", NETTI_KANAVA_VANHAT="3",
                               NETTI_KANAVA_MIN_KESTO_MIN="0", NETTI_KANAVA_SUODATIN="peli|OTTELU", NETTI_KANAVA_LIVE="1"))
        tarkista("NETTI_KANAVAT: pilkku / valilyonti / rivinvaihto / puolipiste erottaa, sama kanava vain kerran", a.kanavat == [K, K2, "https://www.youtube.com/@Toinen/streams"], str(a.kanavat))
        tarkista("kanava-asetukset: valimin 2,5, vanhat 3, ei minimikestoa, suodatin (kirjainkoolla ei valia), suorat heti",
                 a.kanava_vali_min == 2.5 and a.kanava_vanhat == 3 and a.kanava_min_kesto_min == 0.0 and a.kanava_suodatin.search("Matarmaa PELI 3") and a.kanava_suodatin.search("ottelu") and not a.kanava_suodatin.search("harjoitus") and a.kanava_live)
        a = np_.Asetukset(dict(env, NETTI_LAHDE="https://www.youtube.com/@Matarmaa/streams"))
        tarkista("NETTI_LAHDE = kanava: tavallinen kansion valvonta, kanava seurattavana", a.laji == "tiedostot" and a.arvo["kokoelma"] and a.jatkuva and a.kanavat == [K], f"{a.laji} {a.kanavat}")
        a = np_.Asetukset(dict(env, NETTI_LAHDE="@Matarmaa", NETTI_KANAVAT="https://www.youtube.com/@Matarmaa"))
        tarkista("sama kanava NETTI_LAHDE:ssa ja NETTI_KANAVAT:ssa vain kerran", a.kanavat == [K])
        a = np_.Asetukset(dict(env, NETTI_LAHDE="https://www.youtube.com/@Matarmaa/live"))
        tarkista("NETTI_LAHDE = kanavan /live-osoite: yksittainen YouTube-lahde kuten ennen, ei kanavan seurantaa", (a.laji, a.arvo, a.kanavat) == ("youtube", "https://www.youtube.com/@Matarmaa/live", []))
        for nimi, arvo in (("NETTI_KANAVAT", "https://www.youtube.com/watch?v=hnTTLqHr5x8"), ("NETTI_KANAVAT", "@kanava hnTTLqHr5x8"), ("NETTI_KANAVAT", "https://www.youtube.com/@Matarmaa/live"),
                           ("NETTI_KANAVA_VALI_MIN", "0"), ("NETTI_KANAVA_VALI_MIN", "paljon"), ("NETTI_KANAVA_VANHAT", "21"), ("NETTI_KANAVA_VANHAT", "1.5"), ("NETTI_KANAVA_MIN_KESTO_MIN", "-1"),
                           ("NETTI_KANAVA_SUODATIN", "(auki"), ("NETTI_KANAVA_LIVE", "ehka")):
            try:
                np_.Asetukset(dict(env, **{nimi: arvo}))
                tarkista(f"{nimi}={arvo!r} -> AsetusVirhe", False)
            except np_.AsetusVirhe as e:
                tarkista(f"{nimi}={arvo!r} -> AsetusVirhe", nimi in str(e), str(e)[:80])
        a = np_.Asetukset(dict(env, **{n: "" for n in ("NETTI_KANAVAT", "NETTI_KANAVA_VALI_MIN", "NETTI_KANAVA_VANHAT", "NETTI_KANAVA_MIN_KESTO_MIN", "NETTI_KANAVA_SUODATIN", "NETTI_KANAVA_LIVE")}))
        tarkista("tyhjat kanavamuuttujat (Unraid valittaa ne) = oletukset", a.kanavat == [] and a.kanava_vali_min == 5.0 and a.kanava_vanhat == 0 and a.kanava_suodatin is None and not a.kanava_live)

        # linkkilista: kanavarivit
        p = os.path.join(t, "youtube.txt")
        kirjoita(p, "# kanavat\n@Matarmaa\nhttps://www.youtube.com/@Matarmaa/streams   # sama kanava\nhttps://www.youtube.com/channel/UCabc123/videos Matarmaan toinen kanava\nhnTTLqHr5x8\nhttps://www.youtube.com/@Matarmaa/live\n")
        tarkista("lue_linkit: kanavarivit (@kahva, osoite, perassa tekstia), sama kanava vain kerran, /live ja tunnus tavallisina lahteina",
                 np_.lue_linkit(p) == [("kanava", K), ("kanava", K2), ("youtube", "hnTTLqHr5x8"), ("youtube", "https://www.youtube.com/@Matarmaa/live")], str(np_.lue_linkit(p)))

    # ---- luettele_kanava: yt_dlp korvattu (YouTubeen ei ole yhteytta; haun asetukset ja tuloksen tulkinta)
    class EsiYoutubeDL:
        opts, kutsut, vastaus, viive = None, [], None, 0.0

        def __init__(self, opts):
            EsiYoutubeDL.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, url, download=True):
            EsiYoutubeDL.kutsut.append((url, download))
            if EsiYoutubeDL.viive:
                time.sleep(EsiYoutubeDL.viive)
            if isinstance(EsiYoutubeDL.vastaus, BaseException):
                raise EsiYoutubeDL.vastaus
            return EsiYoutubeDL.vastaus

    def yt_entry(n, **kw):
        return dict({"_type": "url", "ie_key": "Youtube", "id": f"vid{n:08d}", "url": f"https://www.youtube.com/watch?v=vid{n:08d}", "title": f"Peli {n}", "duration": 5400, "live_status": "was_live", "availability": "public"}, **kw)

    esi = types.ModuleType("yt_dlp")
    esi.YoutubeDL = EsiYoutubeDL
    vanha_ytdlp, vanha_keksit, vanha_raja = sys.modules.get("yt_dlp"), os.environ.get("YTDLP_KEKSIT"), np_.KANAVA_HAUN_AIKARAJA_S
    sys.modules["yt_dlp"] = esi
    try:
        os.environ.pop("YTDLP_KEKSIT", None)
        EsiYoutubeDL.vastaus = {"_type": "playlist", "entries": [
            yt_entry(3), yt_entry(2, title=None, duration=None, live_status="is_live", availability=None), yt_entry(1, url="https://www.youtube.com/shorts/vid00000001", duration=30, live_status=None),
            yt_entry(9, ie_key="Twitch"), yt_entry(8, id="liian_lyhyt_tunnus"), yt_entry(7, id=None), "ei-sanakirja", None,
            {"_type": "playlist", "entries": [yt_entry(4, ie_key=None, duration=12.5)]}]}
        lista = np_.luettele_kanava(K, 30)
        tarkista("luettele_kanava: video-tunnukset luettelon jarjestyksessa, sisakkainen luettelo litistetty, muut sivuutettu",
                 [v["id"] for v in lista] == ["vid00000003", "vid00000002", "vid00000001", "vid00000004"], str([v["id"] for v in lista]))
        tarkista("luettele_kanava: kentat (otsikko, kesto s, tila, saatavuus, osoite); puuttuva nimi tyhja, puuttuva kesto None",
                 lista[0] == dict(id="vid00000003", otsikko="Peli 3", kesto_s=5400.0, tila="was_live", saatavuus="public", osoite="https://www.youtube.com/watch?v=vid00000003")
                 and lista[1]["otsikko"] == "" and lista[1]["kesto_s"] is None and lista[1]["tila"] == "is_live" and "/shorts/" in lista[2]["osoite"] and lista[3]["kesto_s"] == 12.5)
        tarkista("luettele_kanava: yt_dlp:lle vain luettelo (extract_flat, ei latausta), enintaan n videota, aikakatkaisu, ei keksitiedostoa ilman YTDLP_KEKSIT",
                 EsiYoutubeDL.kutsut == [(K, False)] and EsiYoutubeDL.opts["extract_flat"] == "in_playlist" and EsiYoutubeDL.opts["playlistend"] == 30 and EsiYoutubeDL.opts["skip_download"]
                 and EsiYoutubeDL.opts["socket_timeout"] > 0 and "cookiefile" not in EsiYoutubeDL.opts, str(EsiYoutubeDL.opts))
        with tempfile.TemporaryDirectory() as t:
            keksi = kirjoita(os.path.join(t, "cookies.txt"), "# Netscape HTTP Cookie File\n")
            os.environ["YTDLP_KEKSIT"] = keksi
            np_.luettele_kanava(K, 5)
            kopio = EsiYoutubeDL.opts.get("cookiefile")
            tarkista("luettele_kanava: YTDLP_KEKSIT -> cookiefile (kopio, alkuperaista ei kosketa)", kopio and kopio != keksi and open(kopio).read() == open(keksi).read() and EsiYoutubeDL.opts["playlistend"] == 5, str(kopio))
            os.environ.pop("YTDLP_KEKSIT", None)
        EsiYoutubeDL.vastaus = {"_type": "playlist", "entries": []}
        tarkista("luettele_kanava: tyhja luettelo ja luettelo ilman entries -> tyhja lista", np_.luettele_kanava(K) == [] and (setattr(EsiYoutubeDL, "vastaus", None), np_.luettele_kanava(K))[1] == [])
        EsiYoutubeDL.vastaus = Exception("\x1b[0;31mERROR:\x1b[0m [youtube:tab] @Matarmaa: This channel\ndoes not have a streams tab")
        try:
            np_.luettele_kanava(K)
            tarkista("luettele_kanava: yt_dlp:n virhe -> KanavaVirhe", False)
        except np_.KanavaVirhe as e:
            tarkista("luettele_kanava: yt_dlp:n virhe -> KanavaVirhe, viesti yhdella rivilla ilman varikoodeja", "streams tab" in str(e) and "\x1b" not in str(e) and "\n" not in str(e), str(e))
        EsiYoutubeDL.vastaus = SystemExit(1)
        try:
            np_.luettele_kanava(K)
            tarkista("luettele_kanava: SystemExit -> KanavaVirhe (ei lopeta palvelinta)", False)
        except np_.KanavaVirhe:
            tarkista("luettele_kanava: SystemExit -> KanavaVirhe (ei lopeta palvelinta)", True)
        EsiYoutubeDL.vastaus, EsiYoutubeDL.viive, np_.KANAVA_HAUN_AIKARAJA_S = {"entries": []}, 3.0, 0.3
        t0 = time.monotonic()
        try:
            np_.luettele_kanava(K)
            tarkista("luettele_kanava: jumiin jaanyt haku keskeytetaan aikarajalla", False)
        except np_.KanavaVirhe as e:
            tarkista("luettele_kanava: jumiin jaanyt haku keskeytetaan aikarajalla (valvonta ei jumitu)", "ei vastannut" in str(e) and time.monotonic() - t0 < 2.5, f"{time.monotonic() - t0:.1f} s")
        sys.modules["yt_dlp"] = None                                                # import yt_dlp -> ImportError
        try:
            np_.luettele_kanava(K)
            tarkista("luettele_kanava: yt_dlp puuttuu -> KanavaVirhe", False)
        except np_.KanavaVirhe as e:
            tarkista("luettele_kanava: yt_dlp puuttuu -> KanavaVirhe", "yt_dlp puuttuu" in str(e))
    finally:
        np_.KANAVA_HAUN_AIKARAJA_S = vanha_raja
        if vanha_ytdlp is None:
            sys.modules.pop("yt_dlp", None)
        else:
            sys.modules["yt_dlp"] = vanha_ytdlp
        if vanha_keksit is None:
            os.environ.pop("YTDLP_KEKSIT", None)
        else:
            os.environ["YTDLP_KEKSIT"] = vanha_keksit

    # ---- Kanavavahti: alkutila, uudet lahetykset, haun valit
    def vahti_ja(t, lista, **env):
        a = np_.Asetukset(dict(NETTI_DATA=t, **env))
        k = np_.Kasitellyt(a.kasitellyt_polku)
        ek = EsiKello()
        fk = EsiKanava(lista, ek)
        return a, k, ek, fk, HiljainenVahti(k, a, fk, ek)

    vanhat = [vid(n) for n in (5, 4, 3, 2, 1)]                                          # uusin ensin
    with tempfile.TemporaryDirectory() as t:
        a, k, ek, fk, v = vahti_ja(t, vanhat)
        with lokiin() as loki_ulos:
            u = v.uudet([K])
        tarkista("ensimmainen haku: kanavalla jo olevia videoita ei ajeta", u == [], str(u))
        tarkista("ensimmainen haku: kanavan luettelon osoite ja ikkunan koko", fk.haut == [(K, np_.KANAVA_LISTAUS_N)], str(fk.haut))
        r = np_.Kasitellyt(a.kasitellyt_polku).kanava(K)
        tarkista("alkutila tallentuu kasitellyt.json:iin (kanava:<osoite>): kaikki olemassa olevat tunnukset", r is not None and r["ohitetut"] == sorted(x["id"] for x in vanhat) and r["alustettu"] and "kanava:" + K in json.load(open(a.kasitellyt_polku)))
        tarkista("loki kertoo ohitetuista ja siita, etta uudet analysoidaan", "5 olemassa olevaa videota ohitetaan; uudet lahetykset analysoidaan automaattisesti" in loki_ulos.getvalue() and "@Matarmaa" in loki_ulos.getvalue(), loki_ulos.getvalue()[-200:])
        ek.t += 299
        tarkista("haku vain NETTI_KANAVA_VALI_MIN (5 min) valein: 299 s kuluttua ei uutta hakua", v.uudet([K]) == [] and len(fk.haut) == 1)
        fk.aseta([vid(6, "Matarmaa peli 6")] + vanhat)
        ek.t += 1
        u = v.uudet([K])
        tarkista("5 min kuluttua haku, uusi lahetys havaitaan: youtube_lista + watch-osoite", len(fk.haut) == 2 and u == [("youtube_lista", "https://www.youtube.com/watch?v=vid00000006")], str(u))
        ek.t += 60
        tarkista("ennen seuraavaa hakua sama jono (valmiit suodattaa lahteet)", v.uudet([K]) == u and len(fk.haut) == 2)
        fk.aseta([vid(8), vid(7), vid(6)] + vanhat)
        ek.t += 240
        tarkista("kaksi uutta: vanhin ensin", tunnukset(v.uudet([K])) == ["vid00000006", "vid00000007", "vid00000008"], str(tunnukset(v.uudet([K]))))
        # uudelleenkaynnistys: sama tiedosto, uusi vahti -> ei uutta alkutilaa, uudet edelleen jonossa
        k2 = np_.Kasitellyt(a.kasitellyt_polku)
        v2 = HiljainenVahti(k2, a, fk, ek)
        with lokiin() as loki_ulos:
            u2 = tunnukset(v2.uudet([K]))
        tarkista("uudelleenkaynnistys: alkutilaa ei perusteta uudelleen, ajamattomat uudet ovat edelleen jonossa", u2 == ["vid00000006", "vid00000007", "vid00000008"] and k2.kanava(K)["ohitetut"] == r["ohitetut"]
                 and "ohitetaan" not in loki_ulos.getvalue(), str(u2))
        tarkista("loki kertoo uudesta lahetyksesta kerran (ei joka haulla)", loki_ulos.getvalue().count("uusi lahetys, analysoidaan") == 3, loki_ulos.getvalue())
        with lokiin() as loki_ulos:
            ek.t += 300
            v2.uudet([K])
        tarkista("seuraava haku ei toista ilmoituksia", "uusi lahetys" not in loki_ulos.getvalue(), loki_ulos.getvalue())
        # kanava poistettu asetuksista: sen videoita ei enaa palauteta
        tarkista("kanava, jota ei enaa seurata, ei tuota lahteita", v2.uudet([]) == [])

    # suorat lahetykset ja tulevat
    for live in (False, True):
        with tempfile.TemporaryDirectory() as t:
            perus = [vid(3), vid(2)]
            a, k, ek, fk, v = vahti_ja(t, [vid(9, "Tuleva", None, "is_upcoming"), vid(8, "Suora", None, "is_live"), vid(7, "Käsitellään", None, "post_live")] + perus, **(dict(NETTI_KANAVA_LIVE="1") if live else {}))
            u = v.uudet([K])
            tarkista(f"alkuhaku (suorat {'heti' if live else 'vasta paattyneena'}): suora, tuleva ja kasiteltava lahetys eivat tule alkutilaan (ne ovat uusia), vanhat tulevat",
                     set(k.kanava(K)["ohitetut"]) == {"vid00000003", "vid00000002"} and tunnukset(u) == (["vid00000008"] if live else []), str(tunnukset(u)))
            fk.aseta([vid(9, "Tuleva", None, "is_live"), vid(8, "Suora", 120, "was_live"), vid(7, "Käsitellään", 60, "was_live")] + perus)
            ek.t += 300
            tarkista("tuleva lahetys alkaa suorana, aiemmat paattyvat: " + ("tuleva heti, " if live else "") + "paattyneet ajetaan",
                     tunnukset(v.uudet([K])) == (["vid00000007", "vid00000008", "vid00000009"] if live else ["vid00000007", "vid00000008"]), str(tunnukset(v.uudet([K]))))
            fk.aseta([vid(9, "Tuleva", 100, "was_live"), vid(8, "Suora", 120, "was_live"), vid(7, "Käsitellään", 60, "was_live")] + perus)
            ek.t += 300
            tarkista("kaikki paattyneet: kaikki ajettavia", tunnukset(v.uudet([K])) == ["vid00000007", "vid00000008", "vid00000009"], str(tunnukset(v.uudet([K]))))

    # suodattimet
    uusia = [vid(17, "Matarmaa OTTELU 17"), vid(16, "Muu video"), vid(15, "Tuntematon kesto", None, "was_live"), vid(14, "Matarmaa peli 14", 95), vid(13, "Shorts", 1, osoite="https://www.youtube.com/shorts/vid00000013"),
             vid(12, "Jäsenille", 100, saatavuus="subscriber_only"), vid(11, "Maksullinen", 100, saatavuus="premium_only"), vid(10, "Matarmaa traileri", 3), vid(9, "Matarmaa unlisted", 60, saatavuus="unlisted"),
             vid(18, "Matarmaa ilman saatavuustietoa", 60, saatavuus=None)]
    for env, odotettu, selitys in (
            (dict(), [9, 14, 15, 16, 17, 18], "oletus: alle 20 min, Shorts sekä jäsenille / maksulliset pois; tuntematon kesto, ei-julkinen (unlisted) ja puuttuva saatavuustieto kelpaavat"),
            (dict(NETTI_KANAVA_MIN_KESTO_MIN="0"), [9, 10, 14, 15, 16, 17, 18], "minimikesto pois (0): traileri kelpaa, muut suodattimet ennallaan"),
            (dict(NETTI_KANAVA_MIN_KESTO_MIN="70"), [14, 15, 16, 17], "minimikesto 70 min: 60 min videot pois, tuntematon kesto kelpaa"),
            (dict(NETTI_KANAVA_SUODATIN="matarmaa (peli|ottelu)"), [14, 17], "nimisuodatin (saannollinen lauseke, kirjainkoolla ei valia)")):
        with tempfile.TemporaryDirectory() as t:
            a, k, ek, fk, v = vahti_ja(t, vanhat, **env)
            v.uudet([K])
            fk.aseta(uusia + vanhat)
            ek.t += 300
            u = sorted(tunnukset(v.uudet([K])))
            tarkista("suodatus, " + selitys, u == [f"vid{n:08d}" for n in odotettu], str(u))

    # NETTI_KANAVA_VANHAT: N uusinta olemassa olevaa ajetaan ensimmaisella kerralla, muut ohitetaan; asetuksen myohempi muutos ei herata vanhoja
    kuusi = [vid(6), vid(5, "Matarmaa traileri", 3), vid(4, "Shorts", 1, osoite="https://www.youtube.com/shorts/vid00000004"), vid(3), vid(2), vid(1)]
    with tempfile.TemporaryDirectory() as t:
        a, k, ek, fk, v = vahti_ja(t, kuusi, NETTI_KANAVA_VANHAT="2")
        with lokiin() as loki_ulos:
            u = tunnukset(v.uudet([K]))
        tarkista("NETTI_KANAVA_VANHAT=2: kaksi uusinta (suodattimille kelpaavaa) ajetaan, vanhin ensin", u == ["vid00000003", "vid00000006"], str(u))
        tarkista("VANHAT: loput ohitetaan (myos suodattimiin osumattomat), ne eivat nouse myohemmin uusiksi", set(k.kanava(K)["ohitetut"]) == {"vid00000001", "vid00000002", "vid00000004", "vid00000005"}, str(k.kanava(K)))
        tarkista("VANHAT: loki kertoo", "lisaksi 2 uusinta olemassa olevaa" in loki_ulos.getvalue() and "4 olemassa olevaa videota ohitetaan" in loki_ulos.getvalue(), loki_ulos.getvalue()[-250:])
        a2 = np_.Asetukset(dict(NETTI_DATA=t, NETTI_KANAVA_VANHAT="5", NETTI_KANAVA_MIN_KESTO_MIN="0"))
        v2 = HiljainenVahti(np_.Kasitellyt(a2.kasitellyt_polku), a2, fk, ek)
        ek.t += 300
        tarkista("asetusten myohempi muutos (VANHAT, minimikesto) ei nosta alkutilassa ohitettuja uusiksi", tunnukset(v2.uudet([K])) == ["vid00000003", "vid00000006"], str(tunnukset(v2.uudet([K]))))

    # tyhja ensimmainen luettelo ei perusta mitaan (muuten seuraavalla haulla koko vanha sisalto olisi uutta)
    with tempfile.TemporaryDirectory() as t:
        a, k, ek, fk, v = vahti_ja(t, [])
        with lokiin() as loki_ulos:
            u = v.uudet([K])
        tarkista("tyhja ensimmainen luettelo: ei alkutilaa eika ajettavaa", u == [] and k.kanava(K) is None and "luettelo on tyhja" in loki_ulos.getvalue())
        fk.aseta(vanhat)
        ek.t += 300
        u = v.uudet([K])
        tarkista("seuraava haku (luettelo ei enaa tyhja): alkutila perustetaan, vanhaa sisaltoa ei ajeta", u == [] and len(k.kanava(K)["ohitetut"]) == 5)
        fk.aseta([vid(6)] + vanhat)
        ek.t += 300
        tarkista("... ja uusi lahetys havaitaan normaalisti", tunnukset(v.uudet([K])) == ["vid00000006"])

    # haun virheet: pidentyva valinpito, lokiin kerran, toipuminen, mikaan poikkeus ei kaada
    with tempfile.TemporaryDirectory() as t:
        a, k, ek, fk, v = vahti_ja(t, vanhat)
        v.uudet([K])
        fk.aseta([vid(6)] + vanhat)
        fk.virhe = np_.KanavaVirhe("ERROR: Sign in to confirm you're not a bot")
        valit = []
        with lokiin() as loki_ulos:
            for _ in range(7):
                ek.t = v.seuraava[K]
                v.uudet([K])
                valit.append(v.seuraava[K] - ek.t)
        tarkista("haku epaonnistuu: uusintavali 5, 10, 20, 40, 60, 60, 60 min (enintaan tunti)", valit == [300.0, 600.0, 1200.0, 2400.0, 3600.0, 3600.0, 3600.0], str(valit))
        tarkista("sama virhe kirjoitetaan lokiin ensimmaisella kerralla ja sen jalkeen enintaan kerran tunnissa (7 yritysta -> 4 riviä), ohje evasteista", loki_ulos.getvalue().count("luettelon haku epaonnistui") == 4 and "YTDLP_KEKSIT" in loki_ulos.getvalue(), str(loki_ulos.getvalue().count("luettelon haku epaonnistui")))
        fk.virhe = None
        ek.t = v.seuraava[K]
        with lokiin() as loki_ulos:
            u = tunnukset(v.uudet([K]))
        tarkista("toipuminen: lokiin ilmoitus, jono tulee, virhelaskuri nollautuu, seuraava haku normaalin valin paassa", u == ["vid00000006"] and "haettu taas onnistuneesti" in loki_ulos.getvalue() and v.virheita[K] == 0 and v.seuraava[K] == ek.t + 300)
        fk.virhe = ValueError("odottamaton virhe hakijassa")
        ek.t += 300
        with lokiin() as loki_ulos:
            u = tunnukset(v.uudet([K]))
        tarkista("odottamaton poikkeus hakijassa ei kaada valvontaa; aiemmin havaittu uusi lahetys pysyy jonossa", u == ["vid00000006"] and "odottamaton virhe" in loki_ulos.getvalue())
        v.virheita[K] = 100000
        try:
            v._virhe("x", K, ValueError("e"), ek.t)
            tarkista("virhelaskurin kasvu ei ylivuoda (2 ** n)", v.seuraava[K] - ek.t == np_.KANAVA_VIRHE_MAX_S)
        except OverflowError as e:
            tarkista("virhelaskurin kasvu ei ylivuoda (2 ** n)", False, str(e))
        fk.virhe = np_.KanavaVirhe("ERROR: [youtube:tab] This channel does not have a streams tab")
        v.virheita[K] = 0
        ek.t = v.seuraava[K]
        with lokiin() as loki_ulos:
            v.uudet([K])
        tarkista("Streams-valilehden puuttuminen: ohje kayttaa .../videos-osoitetta", "/videos" in loki_ulos.getvalue(), loki_ulos.getvalue()[-200:])

    # lokin turvallisuus: videon nimi ei voi lisata lokiin omia rivejaan eika ohjausmerkkeja
    with tempfile.TemporaryDirectory() as t:
        a, k, ek, fk, v = vahti_ja(t, vanhat)
        v.uudet([K])
        fk.aseta([vid(6, "Peli\n[palvelin] huijaus \x1b[31m" + "x" * 200)] + vanhat)
        ek.t += 300
        with lokiin() as loki_ulos:
            v.uudet([K])
        rivit = loki_ulos.getvalue().splitlines()
        tarkista("loki: videon nimi yhdelle riville, ohjausmerkit pois, pituus rajattu", len(rivit) == 1 and rivit[0].startswith("[palvelin] kanava @Matarmaa:") and "\x1b" not in rivit[0] and len(rivit[0]) < 220, str(rivit)[:200])

    # ---- lahteet: jarjestys (tiedostot, linkit, kanavat), kaksoiskappaleet, milloin kanavia ei haeta
    with tempfile.TemporaryDirectory() as t:
        nyt = time.time()
        kirjoita(os.path.join(t, "video", "a.mp4"), "a", nyt - 100)
        kirjoita(os.path.join(t, "youtube.txt"), "hnTTLqHr5x8\n@Linkkikanava\nhttps://www.youtube.com/watch?v=vid00000006\n")
        k2url = "https://www.youtube.com/@Linkkikanava/streams"
        a = np_.Asetukset(dict(NETTI_DATA=t, NETTI_KANAVAT="@Matarmaa", NETTI_ODOTUS_S="60"))
        k = np_.Kasitellyt(a.kasitellyt_polku)
        fk = EsiKanava({K: vanhat, k2url: [vid(31)]})
        v = HiljainenVahti(k, a, fk, EsiKello())
        np_.lahteet(a, k, nyt, set(), v)                                            # alkutilat
        fk.aseta([vid(6), vid(7)] + vanhat, K)
        fk.aseta([vid(32), vid(31)], k2url)
        v.seuraava.clear()
        jono = np_.lahteet(a, k, nyt, set(), v)
        watch = lambda n: ("youtube_lista", f"https://www.youtube.com/watch?v=vid{n:08d}")        # noqa: E731
        tarkista("lahteet: videotiedostot, linkkilistan rivit, sitten kanavien uudet (asetuksen kanavat ensin, sitten linkkilistan), sama video vain kerran",
                 jono == [("tiedostot", os.path.join(t, "video", "a.mp4")), ("youtube_lista", "hnTTLqHr5x8"), watch(6), watch(7), watch(32)], str(jono))
        k.kirjaa("youtube_lista", "hnTTLqHr5x8", True)
        k.kirjaa("youtube_lista", "https://www.youtube.com/watch?v=vid00000007", True)
        tarkista("lahteet: valmiiksi kirjattu kanavan video ei palaa jonoon (vaikka vahdin vanha jono sisaltaa sen)", watch(7) not in np_.lahteet(a, k, nyt, set(), v) and ("youtube_lista", "hnTTLqHr5x8") not in np_.lahteet(a, k, nyt, set(), v))
        haut = len(fk.haut)
        jono = np_.lahteet(a, k, nyt)
        tarkista("lahteet: ilman vahtia kanavia ei haeta (linkkilistan omat rivit ajetaan)", len(fk.haut) == haut and jono == [("tiedostot", os.path.join(t, "video", "a.mp4")), watch(6)], str(jono))
        for env, selitys in ((dict(NETTI_JATKUVA="0"), "kertaajo (ei valvontaa)"), (dict(NETTI_LAHDE=os.path.join(t, "video", "a.mp4")), "yksittainen tiedosto"), (dict(NETTI_LAHDE="hnTTLqHr5x8"), "yksittainen YouTube-lahde")):
            haut = len(fk.haut)
            a2 = np_.Asetukset(dict(dict(NETTI_DATA=t, NETTI_KANAVAT="@Matarmaa"), **env))
            v2 = HiljainenVahti(np_.Kasitellyt(a2.kasitellyt_polku), a2, fk, EsiKello())
            np_.lahteet(a2, None, nyt, set(), v2)
            tarkista(f"lahteet: kanavia ei haeta, kun {selitys}", len(fk.haut) == haut)

    # ---- palvele: uusi lahetys kanavalle -> ajo kerran; epaonnistuminen -> uusinta pidentyvin valein; uudelleenkaynnistys ei toista
    with tempfile.TemporaryDirectory() as t:
        ek = EsiKello()
        a = np_.Asetukset(dict(NETTI_DATA=t, NETTI_KANAVAT="@Matarmaa", NETTI_ODOTUS_S="60"))
        fk = EsiKanava({K: [(0, vanhat), (900, [vid(6, "Matarmaa SM-finaali")] + vanhat)]}, ek)
        ajot = []

        def ajuri(cmd, env=None, cwd=None):
            ajot.append((ek.t - 1_800_000_000.0, lahde_komennosta(cmd), cmd))
            return 0

        with lokiin() as loki_ulos:
            rc = np_.palvele(a, EsiPysaytys(pysayta_nukkumisella=30, kello=ek), ajuri, 32.0, ek, fk)
        tarkista("silmukka: kanavalla jo olleita ei ajeta, uusi lahetys ajetaan kerran heti kun se havaitaan", rc == 0 and len(ajot) == 1 and ajot[0][1] == ("--youtube", "https://www.youtube.com/watch?v=vid00000006") and 900 <= ajot[0][0] < 1200,
                 str([(x[0], x[1]) for x in ajot]))
        tarkista("silmukka: tulostiedostojen nimet saavat videon nimen paikan ({otsikko}) ja tunnisteen yt_<tunnus>", any("kivenseuranta_{otsikko}_yt_vid00000006_" in x for x in ajot[0][2]) and any("heitot_{otsikko}_yt_vid00000006_" in x for x in ajot[0][2]))
        tarkista("silmukka: kanavaa haettiin säännöllisesti (5 min valein)", 4 <= len(fk.haut) <= 8, str(len(fk.haut)))
        tarkista("silmukka: valmis kirjattu muistiin (kasitellyt.json) ja kanavan alkutila tallessa", np_.Kasitellyt(a.kasitellyt_polku).valmis("youtube_lista", "https://www.youtube.com/watch?v=vid00000006") and np_.Kasitellyt(a.kasitellyt_polku).kanava(K) is not None)
        tarkista("silmukka: loki kertoo kanavan, haun valin ja uuden lahetyksen", "kanava @Matarmaa: uusi lahetys, analysoidaan: 'Matarmaa SM-finaali'" in loki_ulos.getvalue() and "olemassa olevaa videota ohitetaan" in loki_ulos.getvalue(), loki_ulos.getvalue()[:300])
        ajot.clear()
        ek2 = EsiKello(ek.t + 7200)
        fk2 = EsiKanava({K: [(0, [vid(6, "Matarmaa SM-finaali")] + vanhat)]}, ek2, alku=ek2.t)
        with lokiin():
            np_.palvele(a, EsiPysaytys(pysayta_nukkumisella=10, kello=ek2), ajuri, 32.0, ek2, fk2)
        tarkista("uudelleenkaynnistys: aiemmin ajettua kanavan videota ei ajeta uudelleen eika alkutilaa perusteta uudelleen", ajot == [], str(ajot))

    with tempfile.TemporaryDirectory() as t:
        ek = EsiKello()
        a = np_.Asetukset(dict(NETTI_DATA=t, NETTI_LAHDE="@Matarmaa", NETTI_ODOTUS_S="60"))
        fk = EsiKanava({K: [(0, vanhat), (300, [vid(6)] + vanhat)]}, ek)
        ajat = []

        def ajuri_vika(cmd, env=None, cwd=None):
            ajat.append(ek.t - 1_800_000_000.0)
            return 1

        with lokiin():
            np_.palvele(a, EsiPysaytys(pysayta_nukkumisella=130, kello=ek), ajuri_vika, 32.0, ek, fk)
        tarkista(f"epaonnistuva kanavan video: {np_.LINKKI_YRITYKSIA_MAX} yritysta pidentyvin valein (kuten linkkilistan rivi), sen jalkeen se jaa pois", [x - ajat[0] for x in ajat] == [0, 60, 180, 420, 900, 1800, 2700, 3600, 4500, 5400], str(ajat))

    # ---- vanha kayttaytyminen sailyy: kanavan /live-osoite on yksittainen lahde; odotuksen kasvu ei ylivuoda (ennen ~1000 perakkaisen nopean epaonnistumisen jalkeen OverflowError)
    with tempfile.TemporaryDirectory() as t:
        ek = EsiKello()
        a = np_.Asetukset(dict(NETTI_DATA=t, NETTI_LAHDE="https://www.youtube.com/@Matarmaa/live", NETTI_JATKUVA="1", NETTI_ODOTUS_S="60"))
        ajot = []

        def ajuri_ei_lahetysta(cmd, env=None, cwd=None):
            ajot.append(lahde_komennosta(cmd))
            return 1

        ps = EsiPysaytys(pysayta_nukkumisella=1300, kello=ek)
        try:
            with lokiin():
                rc = np_.palvele(a, ps, ajuri_ei_lahetysta, 32.0, ek, EsiKanava({}))
            tarkista("kanavan /live-osoite: ajetaan sellaisenaan (--youtube=...), kanavan haku ei kaynnisty", ajot[0] == ("--youtube", "https://www.youtube.com/@Matarmaa/live") and a.kanavat == [])
            tarkista("odotus pysyy rajoissa 1300 perakkaisen nopean epaonnistumisen (~15 vrk) jalkeen: ei ylivuotoa, odotus enintaan 15 min", rc == 0 and len(ps.nukkumiset) == 1300 and max(ps.nukkumiset) == np_.ODOTUS_MAX_S, f"{rc} {len(ps.nukkumiset)} {max(ps.nukkumiset)}")
        except OverflowError as e:
            tarkista("odotus pysyy rajoissa 1300 perakkaisen nopean epaonnistumisen jalkeen: ei ylivuotoa", False, str(e))

    # ---- kaynnistaja: kanavat nakyvat kaynnistyslokissa; vaarat asetukset -> selkea virhe; kanavia ei seurata, jos valvontaa ei ole
    with tempfile.TemporaryDirectory() as t:
        env = dict(os.environ, NETTI_DATA=t, NETTI_WEB_PORTTI="0", NETTI_KANAVAT="@Matarmaa, https://www.youtube.com/channel/UCabc123/videos", NETTI_KANAVA_VANHAT="1")
        for muuttuja in ("NETTI_LAHDE", "NETTI_JATKUVA", "NETTI_KANAVA_LIVE"):
            env.pop(muuttuja, None)
        kirjoita(os.path.join(t, "youtube.txt"), "https://www.youtube.com/@Linkkikanava\n")
        p = subprocess.run([sys.executable, os.path.join(KOTI, "netti_palvelin.py"), "--nayta"], env=env, capture_output=True, text=True, timeout=60)
        tarkista("--nayta: kaynnistysloki listaa kanavat (asetukset + linkkilista), haun valin ja vanhat; verkkoa ei kaytetty", p.returncode == 0 and "kanavat: @Matarmaa, channel/UCabc123, @Linkkikanava (luettelo 5 min valein" in p.stdout
                 and "lisaksi 1 uusinta" in p.stdout and "suorat vasta kun ne ovat paattyneet" in p.stdout and "komento:" in p.stdout, p.stdout[-300:] + p.stderr[-200:])
        p = subprocess.run([sys.executable, os.path.join(KOTI, "netti_palvelin.py"), "--nayta"], env=dict(env, NETTI_LAHDE="hnTTLqHr5x8"), capture_output=True, text=True, timeout=60)
        tarkista("--nayta: kanavat ohitetaan ja siita huomautetaan, kun lahde on yksittainen video", p.returncode == 0 and "kanavia (NETTI_KANAVAT / NETTI_LAHDE) seurataan vain kansion valvonnassa" in p.stdout, p.stdout[-300:])
        p = subprocess.run([sys.executable, os.path.join(KOTI, "netti_palvelin.py"), "--nayta"], env=dict(env, NETTI_KANAVAT="hnTTLqHr5x8"), capture_output=True, text=True, timeout=60)
        tarkista("vaara NETTI_KANAVAT: paluukoodi 2 ja selkea viesti", p.returncode == 2 and "NETTI_KANAVAT" in p.stderr and "hnTTLqHr5x8" in p.stderr, p.stderr.strip()[-200:])


def main():
    for t in (testaa_ymparisto, testaa_asetukset, testaa_komento, testaa_tiedostot_ja_muisti, testaa_silmukka, testaa_linkkilista, testaa_kanavat, testaa_lapsiprosessi, testaa_web, testaa_netti_live_palvelinosat,
              testaa_videon_nimi_palvelin, testaa_web_videon_nimi):
        t()
    print()
    if VIRHEITA:
        print(f"{len(VIRHEITA)} testia epaonnistui: {', '.join(VIRHEITA)}")
        return 1
    print("kaikki testit OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
