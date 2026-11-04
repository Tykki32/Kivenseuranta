"""Testit tallenteen aanelle (netti_aani.py, netti_live.Kirjoittaja, tools/liita_aani.py): liitos ei saa koskaan lyhentaa kuvaa, huono aanilahde jattaa tallenteen ennalleen, YouTube-aani ladataan
taustalla (simuloitu hidas lataus) ja liitetaan lopuksi, ja verkkoaanen katkot eivat lyhenna kuvaa. Tarvitsee ffmpegin (PATH tai imageio-ffmpeg); ilman sita testit ohitetaan.

    python tools/testaa_aani.py

Poistumiskoodi 1, jos jokin testi epaonnistuu."""
import http.server
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import types

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import netti_aani as na  # noqa: E402

VIRHEITA = []
EXE = na.etsi_ffmpeg()


def tarkista(nimi, ehto, teksti=""):
    print(("  OK   " if ehto else "  VIRHE ") + nimi + (f": {teksti}" if teksti else ""))
    if not ehto:
        VIRHEITA.append(nimi)


def ffmpeg(*args):
    subprocess.run([EXE, "-hide_banner", "-loglevel", "error", "-y", *args], check=True, capture_output=True)


def tee_aineisto(kansio):
    """Synteettiset tiedostot: kuva 12 s (ei aanta), aani 30 s ja 5 s (sini), aanettoman videon 'aani'."""
    p = {k: os.path.join(kansio, k) for k in ("kuva.mp4", "aani30.m4a", "aani5.m4a", "roska.m4a", "mykka.m4a")}
    ffmpeg("-f", "lavfi", "-i", "testsrc=size=320x180:rate=30", "-t", "12", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", p["kuva.mp4"])
    ffmpeg("-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100", "-t", "30", "-c:a", "aac", "-b:a", "128k", p["aani30.m4a"])
    ffmpeg("-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100", "-t", "5", "-c:a", "aac", "-b:a", "128k", p["aani5.m4a"])
    ffmpeg("-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo", "-t", "12", "-c:a", "aac", p["mykka.m4a"])
    with open(p["roska.m4a"], "wb") as f:
        f.write(os.urandom(5000))
    return p


def kopio(src, kansio, nimi):
    dst = os.path.join(kansio, nimi)
    shutil.copyfile(src, dst)
    return dst


def keskiaanenvoimakkuus(polku, alku_s, kesto_s):
    """Aanen keskivoimakkuus (dB) valilla [alku, alku + kesto] (ffmpeg volumedetect); -91 = mykka."""
    r = subprocess.run([EXE, "-hide_banner", "-ss", str(alku_s), "-t", str(kesto_s), "-i", polku, "-map", "0:a:0", "-af", "volumedetect", "-f", "null", "-"], capture_output=True, text=True, errors="replace")
    m = re.search(r"mean_volume:\s*(-?[\d.]+|-inf) dB", r.stderr)
    return None if m is None else (-120.0 if m.group(1) == "-inf" else float(m.group(1)))


def testaa_liitos(kansio, p):
    print("liitos (netti_aani.liita)")
    kesto0, _, _ = na.tiedot(EXE, p["kuva.mp4"])
    tarkista("aineisto: kuva 12 s ilman aanta", abs(kesto0 - 12.0) < 0.1, f"{kesto0}")
    k = kopio(p["kuva.mp4"], kansio, "pitka.mp4")
    ok, huom = na.liita(EXE, k, p["aani30.m4a"])
    kesto, aanta, kuvaa = na.tiedot(EXE, k)
    tarkista("pitka aani: liitetty, kesto sailyy (12 s), kuva- ja aaniraita, ei varoituksia, aanen tiedot mukana", ok and abs(kesto - 12.0) < 0.15 and aanta and kuvaa and "huom:" not in huom and "keskivoimakkuus" in huom, f"{ok} {kesto} {huom!r}")
    tarkista("pitka aani: ei valitiedostoa", not os.path.exists(k + ".aani.mp4"))
    k = kopio(p["kuva.mp4"], kansio, "lyhyt.mp4")
    ok, huom = na.liita(EXE, k, p["aani5.m4a"])
    kesto, aanta, kuvaa = na.tiedot(EXE, k)
    tarkista("lyhyt aani (5 s < kuva 12 s): kuvaa EI lyhenneta (vanha -shortest teki 5 s:n tiedoston), huomautus", ok and abs(kesto - 12.0) < 0.15 and aanta and "mykka" in huom, f"{ok} {kesto} {huom!r}")
    v_alku, v_loppu = keskiaanenvoimakkuus(k, 1.0, 2.0), keskiaanenvoimakkuus(k, 8.0, 3.0)
    tarkista("lyhyt aani: alussa aania, lopussa mykka", v_alku is not None and v_alku > -40.0 and v_loppu is not None and v_loppu < -60.0, f"{v_alku} dB / {v_loppu} dB")
    k = kopio(p["kuva.mp4"], kansio, "alku.mp4")
    ok, huom = na.liita(EXE, k, p["aani30.m4a"], alku_s=25.0)
    kesto, _, _ = na.tiedot(EXE, k)
    tarkista("aanen alku 25 s (aanta jaljella 5 s): kesto sailyy, huomautus", ok and abs(kesto - 12.0) < 0.15 and "mykka" in huom, f"{kesto} {huom!r}")
    k = kopio(p["kuva.mp4"], kansio, "mykka.mp4")
    ok, huom = na.liita(EXE, k, p["mykka.m4a"])
    tarkista("mykka aani: liitetaan, mutta varoitus (ladattu aani on mykka)", ok and "mykka: tarkista aanilahde" in huom, huom[:110])
    tila = na.aanen_tila(EXE, p["aani30.m4a"], 0.0, 12.0)
    tarkista("aanen_tila: sini 440 Hz: kesto 12 s (dekoodattu valilta), keskivoimakkuus -35...-10 dB, mono AAC", tila is not None and abs(tila["kesto"] - 12.0) < 0.1 and -35.0 < tila["keski_db"] < -10.0 and tila["koodekki"] == "aac", str(tila))
    tarkista("aanen_tila: rikkinainen tiedosto -> None", na.aanen_tila(EXE, p["roska.m4a"]) is None)
    rivit = na.virrat(EXE, k)
    tarkista("virrat: kesto-, kuva- ja aanirivi", any("Duration" in x for x in rivit) and any("Video:" in x for x in rivit) and any("Audio:" in x for x in rivit), f"{len(rivit)} riviä")
    # huono lahde: tallenne jaa tavu tavulta ennalleen
    for nimi, lahde in (("aaniraidaton tiedosto", p["kuva.mp4"]), ("roskatiedosto", p["roska.m4a"]), ("puuttuva tiedosto", os.path.join(kansio, "ei_ole.m4a"))):
        k = kopio(p["kuva.mp4"], kansio, "ennalleen.mp4")
        ennen = open(k, "rb").read()
        ok, viesti = na.liita(EXE, k, lahde)
        tarkista(f"{nimi}: ei liitetty, tallenne ennallaan, ei valitiedostoa", (not ok) and open(k, "rb").read() == ennen and not os.path.exists(k + ".aani.mp4"), viesti[:70])
    k = kopio(p["kuva.mp4"], kansio, "ulos_lahde.mp4")
    ulos = os.path.join(kansio, "ulos_tulos.mp4")
    ennen = open(k, "rb").read()
    ok, _ = na.liita(EXE, k, p["aani30.m4a"], ulos=ulos)
    tarkista("--ulos: tulos omaan tiedostoon, alkuperainen koskematta", ok and open(k, "rb").read() == ennen and na.tiedot(EXE, ulos)[1])


class Palvelin(http.server.BaseHTTPRequestHandler):
    """Pieni Range-tukea tarjoava palvelin; katko = lahetetaan vain tama monta tavua ja suljetaan yhteys (aina = joka pyynnolla, muuten vain ensimmaisella)."""
    tiedosto = None
    katko = None
    aina = False
    pyynnot = 0

    def log_message(self, *a):
        pass

    def do_GET(self):
        d = open(self.tiedosto, "rb").read()
        n = len(d)
        a, b = 0, n - 1
        rng = self.headers.get("Range")
        if rng and rng.startswith("bytes="):
            x, y = rng[6:].split("-")
            a, b = int(x or 0), (int(y) if y else n - 1)
        Palvelin.pyynnot += 1
        self.send_response(206 if rng else 200)
        self.send_header("Content-Type", "audio/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(b - a + 1))
        if rng:
            self.send_header("Content-Range", f"bytes {a}-{b}/{n}")
        self.end_headers()
        data = d[a:b + 1]
        katkaise = self.katko is not None and (self.aina or Palvelin.pyynnot == 1 or (rng and a == 0))
        if katkaise:
            data = data[:self.katko]
        try:
            self.wfile.write(data)
            self.wfile.flush()
        except Exception:
            pass
        if katkaise:
            self.connection.close()


def palvele(tiedosto, katko=None, aina=False):
    Palvelin.tiedosto, Palvelin.katko, Palvelin.aina, Palvelin.pyynnot = tiedosto, katko, aina, 0
    s = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Palvelin)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    return s, f"http://127.0.0.1:{s.server_address[1]}/aani.m4a"


def testaa_verkko(kansio, p):
    print("verkkoaani (ffmpeg lukee suoraan; vanha tapa YouTubessa)")
    for nimi, katko, aina in (("normaali palvelin", None, False), ("yhteys katkeaa kerran, jatko Rangella", 60000, False), ("pysyva katkos (aani loppuu kesken)", 60000, True)):
        s, url = palvele(p["aani30.m4a"], katko, aina)
        try:
            k = kopio(p["kuva.mp4"], kansio, "verkko.mp4")
            ok, viesti = na.liita(EXE, k, url)
            kesto, aanta, kuvaa = na.tiedot(EXE, k)
        finally:
            s.shutdown()
        tarkista(f"{nimi}: kuva ei lyhene (12 s), aaniraita mukana", ok and abs(kesto - 12.0) < 0.15 and aanta and kuvaa, f"{ok} {kesto} {viesti[:60]}")


def fake_lataus(src, tahti_s=0.05, pala=40000, epaonnistu=None, jumi=False):
    """Simuloitu yt_dlp-lataus: kopioi src hitaasti kohde.m4a:ksi ja ilmoittaa etenemisen; epaonnistu = poikkeus; jumi = ei etene lainkaan."""
    def lataa(kohde, edistys):
        if jumi:
            time.sleep(30.0)
        if epaonnistu is not None:
            time.sleep(0.2)
            raise epaonnistu
        d = open(src, "rb").read()
        dst = kohde + ".m4a"
        with open(dst, "wb") as f:
            for i in range(0, len(d), pala):
                f.write(d[i:i + pala])
                edistys(min(i + pala, len(d)), len(d))
                time.sleep(tahti_s)
        return dst
    return lataa


def testaa_lataus(kansio, p):
    print("YouTube-aanen lataus taustalla (simuloitu) ja liitos lopuksi")
    viestit = []
    loki = viestit.append
    k = kopio(p["kuva.mp4"], kansio, "yt1.mp4")
    lahde = na.Aanilahde("youtube", "TUNNUS")
    lahde.kaynnista(os.path.join(kansio, "yt1.aani"), EXE, lataa=fake_lataus(p["aani30.m4a"], tahti_s=0.4))
    ok = na.liita_tallenteeseen(EXE, k, lahde, loki=loki)
    kesto, aanta, kuvaa = na.tiedot(EXE, k)
    tarkista("hidas lataus: odotetaan, liitetty, kesto 12 s", ok and abs(kesto - 12.0) < 0.15 and aanta and kuvaa, f"{ok} {kesto}")
    tarkista("hidas lataus: etenemisilmoitus ja onnistumisilmoitus", any("Odotetaan aanen latausta" in v for v in viestit) and any(v.startswith("Aani liitetty") for v in viestit), "; ".join(viestit)[:150])
    tarkista("onnistuneen liitoksen jalkeen ladattu aanitiedosto on siivottu", not os.path.exists(os.path.join(kansio, "yt1.aani.m4a")))
    # lataus ehtii valmiiksi ennen lopetusta: liitos heti
    k = kopio(p["kuva.mp4"], kansio, "yt2.mp4")
    lahde = na.Aanilahde("youtube", "TUNNUS")
    lahde.kaynnista(os.path.join(kansio, "yt2.aani"), EXE, lataa=fake_lataus(p["aani30.m4a"], tahti_s=0.0))
    lahde.lataus.valmis.wait(10.0)
    t = time.time()
    ok = na.liita_tallenteeseen(EXE, k, lahde, loki=lambda x: None)
    tarkista("valmiiksi ladattu aani: liitos sekunneissa", ok and time.time() - t < 10.0, f"{time.time() - t:.1f} s")
    # lataus epaonnistuu: tallenne ehjana, syy kerrotaan
    viestit.clear()
    k = kopio(p["kuva.mp4"], kansio, "yt3.mp4")
    ennen = open(k, "rb").read()
    lahde = na.Aanilahde("youtube", "TUNNUS")
    lahde.kaynnista(os.path.join(kansio, "yt3.aani"), EXE, lataa=fake_lataus(p["aani30.m4a"], epaonnistu=RuntimeError("HTTP Error 403: Forbidden")))
    ok = na.liita_tallenteeseen(EXE, k, lahde, loki=loki)
    tarkista("lataus epaonnistuu: ei liitetty, tallenne ennallaan, syy (403) ja ohje nakyvat", (not ok) and open(k, "rb").read() == ennen and any("403" in v for v in viestit) and any("ehja" in v for v in viestit), "; ".join(viestit)[:150])
    # lataus jumittaa: odotus lopetetaan, kun edistysta ei tule
    viestit.clear()
    k = kopio(p["kuva.mp4"], kansio, "yt4.mp4")
    lataus = na.AaniLataus("TUNNUS", os.path.join(kansio, "yt4.aani"), EXE, lataa=fake_lataus(p["aani30.m4a"], jumi=True))
    t = time.time()
    valmis = lataus.odota(ei_edistysta_s=1.0, raportti_s=0.5, loki=loki)
    tarkista("jumittunut lataus: odotus keskeytyy ~1 s:n jalkeen (False)", (not valmis) and time.time() - t < 6.0, f"{time.time() - t:.1f} s")
    # ladattu aani on liian lyhyt: huomautus, kuva ehja
    k = kopio(p["kuva.mp4"], kansio, "yt5.mp4")
    lahde = na.Aanilahde("youtube", "TUNNUS")
    lahde.kaynnista(os.path.join(kansio, "yt5.aani"), EXE, lataa=fake_lataus(p["aani5.m4a"], tahti_s=0.0))
    viestit.clear()
    ok = na.liita_tallenteeseen(EXE, k, lahde, loki=loki)
    tarkista("ladattu aani lyhyempi kuin kuva: liitetty, kesto 12 s, huomautus", ok and abs(na.tiedot(EXE, k)[0] - 12.0) < 0.15 and any("mykka" in v for v in viestit), "; ".join(viestit)[:150])


def testaa_yt_dlp_logiikka(kansio, p):
    print("YouTube-latauksen formaattien kokeilu ja tarkistus (simuloitu yt_dlp)")

    def sim_modulit(skripti):
        """skripti: {formaatti: (format_id, lahdetiedosto tai None = virhe, kesto)}; palauttaa korvaavan yt_dlp-moduulin."""
        class SimYDL:
            def __init__(self, opts):
                self.opts = opts
                self.skr = skripti[opts["format"]]

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def extract_info(self, url, download=False):
                fid, lahde, kesto = self.skr
                if lahde is None:
                    self.opts["logger"].error("HTTP Error 403: Forbidden")
                    raise RuntimeError("HTTP Error 403: Forbidden")
                if not download:
                    return {"format_id": fid}
                polku = self.opts["outtmpl"] % {"ext": "m4a"}
                shutil.copyfile(lahde, polku)
                self.opts["logger"].warning("simuloitu varoitus")
                return {"format_id": fid, "ext": "m4a", "protocol": "https", "acodec": "aac", "abr": 128, "duration": kesto, "requested_downloads": [{"filepath": polku, "format_id": fid}]}

            def prepare_filename(self, info):
                return self.opts["outtmpl"] % {"ext": "m4a"}
        return types.SimpleNamespace(YoutubeDL=SimYDL)

    F = na.YT_AANI_FORMAATIT
    vanha = sys.modules.get("yt_dlp")
    try:
        def kokeile(skripti, nimi):
            sys.modules["yt_dlp"] = sim_modulit(skripti)
            kohde = os.path.join(kansio, f"ytl_{nimi}")
            l = na.AaniLataus("TUNNUS", kohde, EXE)
            l.valmis.wait(60.0)
            return l, kohde
        l, kohde = kokeile({F[0]: ("140", p["aani30.m4a"], 30.0), F[1]: ("140", p["aani30.m4a"], 30.0), F[2]: ("140", p["aani30.m4a"], 30.0)}, "a")
        tarkista("1. formaatti kelpaa: ladattu, tila ja muoto tallessa, ei turhia latauksia", l.virhe is None and l.polku and l.muoto.startswith("140") and l.tila is not None and abs(l.tila["kesto"] - 30.0) < 0.2
                 and len([x for x in os.listdir(kansio) if x.startswith("ytl_a.")]) == 1, f"{l.virhe!r} {l.muoto} {l.tila}")
        l, kohde = kokeile({F[0]: ("233", p["mykka.m4a"], 30.0), F[1]: ("140", p["aani30.m4a"], 30.0), F[2]: ("140", p["aani30.m4a"], 30.0)}, "b")
        tarkista("1. formaatti mykka -> 2. formaatti kelpaa: hyvaksytaan 2., mykka tiedosto poistettu, syy talletettu", l.virhe is None and l.muoto.startswith("140") and any("ei kelvannut" in v for v in l.viestit)
                 and len([x for x in os.listdir(kansio) if x.startswith("ytl_b.")]) == 1, f"{l.virhe!r} {l.muoto} {l.viestit[:2]}")
        l, kohde = kokeile({F[0]: ("233", p["mykka.m4a"], 30.0), F[1]: ("233", p["mykka.m4a"], 30.0), F[2]: ("234", p["aani5.m4a"], 30.0)}, "c")
        tarkista("sama formaatti ei lataudu kahdesti; kun mikaan ei kelpaa, paras (pisin ei-mykka, 5 s) valitaan", l.virhe is None and l.muoto.startswith("234") and l.tila is not None and abs(l.tila["kesto"] - 5.0) < 0.2, f"{l.virhe!r} {l.muoto} {l.tila}")
        k = os.path.join(kansio, "ytl_c.mp4")
        shutil.copyfile(p["kuva.mp4"], k)
        lahde = na.Aanilahde("youtube", "TUNNUS")
        lahde.lataus = l
        viestit = []
        ok = na.liita_tallenteeseen(EXE, k, lahde, loki=viestit.append)
        tarkista("liitos kayttaa valittua (lyhytta) aanta: liitetty, kesto 12 s, varoitus 'aani loppuu', formaatti ja yt_dlp:n viestit nakyvat", ok and abs(na.tiedot(EXE, k)[0] - 12.0) < 0.15 and any("Ladattu aani: 234" in v for v in viestit)
                 and any("simuloitu varoitus" in v for v in viestit), "; ".join(viestit)[:200])
        l, kohde = kokeile({F[0]: ("1", None, 0), F[1]: ("2", None, 0), F[2]: ("3", None, 0)}, "d")
        tarkista("kaikki formaatit epaonnistuvat: virhe, jossa yt_dlp:n syy (403)", l.virhe is not None and "403" in str(l.virhe), repr(l.virhe))
    finally:
        if vanha is not None:
            sys.modules["yt_dlp"] = vanha
        else:
            sys.modules.pop("yt_dlp", None)


def testaa_kirjoittaja(kansio, p):
    print("Kirjoittaja (netti_live): tallenne + aani ajon lopussa")
    try:
        import numpy as np
        import netti_live as nl
    except Exception as e:
        print(f"  ohitetaan (netti_live ei lataudu: {e!r})")
        return
    ulos = os.path.join(kansio, "kirjoittaja.mp4")
    lahde = na.Aanilahde("youtube", "TUNNUS")
    lahde.kaynnista(os.path.join(kansio, "kirjoittaja.aani"), EXE, lataa=fake_lataus(p["aani30.m4a"], tahti_s=0.1))        # lataus kayntiin heti, kuten ajon alussa
    k = nl.Kirjoittaja(ulos, 30.0, "cpu", None, "ultrafast", lahde)
    fr = np.zeros((1080, 1920, 3), np.uint8)
    for i in range(60):
        fr[:] = (i * 4) % 255
        k.kirjoita(fr.copy())
    k.sulje()
    kesto, aanta, kuvaa = na.tiedot(EXE, ulos)
    tarkista("Kirjoittaja: 60 ruutua -> 2,0 s, kuva- ja aaniraita", abs(kesto - 2.0) < 0.1 and aanta and kuvaa and k.n == 60, f"{kesto} s, {k.n} ruutua")
    tarkista("Kirjoittaja: valitiedostoja ei jaa", not os.path.exists(ulos + ".aani.mp4") and not os.path.exists(os.path.join(kansio, "kirjoittaja.aani.m4a")))
    # valitse_aani: YouTube -> lataus ei kaynnisty ennen Kirjoittajaa
    a = types.SimpleNamespace(aani="auto", youtube="TUNNUS", ffmpeg=None, url=None)
    l = types.SimpleNamespace(live=False, tiedosto=None, alku_s=0.0)
    v = nl.valitse_aani(a, l)
    tarkista("valitse_aani: YouTube -> 'youtube', lataus ei ole viela kaynnissa", v is not None and v.laji == "youtube" and v.lataus is None)
    tarkista("valitse_aani: --aani ei -> None; live -> None", nl.valitse_aani(types.SimpleNamespace(aani="ei", youtube="X", ffmpeg=None, url=None), l) is None
             and nl.valitse_aani(a, types.SimpleNamespace(live=True, tiedosto=None, alku_s=0.0)) is None)
    a2 = types.SimpleNamespace(aani="auto", youtube=None, ffmpeg=None, url=None)
    v = nl.valitse_aani(a2, types.SimpleNamespace(live=False, tiedosto=p["aani30.m4a"], alku_s=3.0))
    tarkista("valitse_aani: tiedosto, jossa on aani -> 'tiedosto', alku mukana", v is not None and v.laji == "tiedosto" and v.alku_s == 3.0)
    v = nl.valitse_aani(a2, types.SimpleNamespace(live=False, tiedosto=p["kuva.mp4"], alku_s=0.0))
    tarkista("valitse_aani: tiedostolla ei aanta -> None", v is None)


def testaa_tyokalu(kansio, p):
    print("tools/liita_aani.py")
    k = kopio(p["kuva.mp4"], kansio, "tyokalu.mp4")
    r = subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "liita_aani.py"), k, "--aani", p["aani30.m4a"]], capture_output=True, text=True, errors="replace")
    tarkista("tyokalu: liittaa paikallisen aanen (poistumiskoodi 0, aaniraita)", r.returncode == 0 and na.tiedot(EXE, k)[1], r.stdout[-120:] + r.stderr[-120:])
    k = kopio(p["kuva.mp4"], kansio, "tyokalu2.mp4")
    ennen = open(k, "rb").read()
    r = subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "liita_aani.py"), k, "--aani", p["roska.m4a"]], capture_output=True, text=True, errors="replace")
    tarkista("tyokalu: huono aani -> poistumiskoodi 1, tallenne ennallaan", r.returncode == 1 and open(k, "rb").read() == ennen, r.stdout[-120:])


def main():
    if EXE is None:
        print("ffmpegia ei loydy: testit ohitetaan")
        return 0
    with tempfile.TemporaryDirectory() as kansio:
        p = tee_aineisto(kansio)
        for t in (testaa_liitos, testaa_verkko, testaa_lataus, testaa_yt_dlp_logiikka, testaa_kirjoittaja, testaa_tyokalu):
            t(kansio, p)
    print()
    if VIRHEITA:
        print(f"{len(VIRHEITA)} testia epaonnistui: {', '.join(VIRHEITA)}")
        return 1
    print("kaikki testit OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
