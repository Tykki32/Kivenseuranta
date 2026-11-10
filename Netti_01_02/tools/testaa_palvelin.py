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


def main():
    for t in (testaa_ymparisto, testaa_asetukset, testaa_komento, testaa_tiedostot_ja_muisti, testaa_silmukka, testaa_linkkilista, testaa_lapsiprosessi, testaa_web, testaa_netti_live_palvelinosat,
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
