"""Reaaliaikainen kivenseuranta: kuva tulostuu VIIVE_S sekuntia analyysin perassa, ilman top-down-muunnosta.

    python netti_live.py --youtube hnTTLqHr5x8                     # YouTube (yt_dlp + OpenCV, kuten oma testikoodi), ikkuna taydella resoluutiolla, 'q' lopettaa
    python netti_live.py --youtube hnTTLqHr5x8 --ulos              # sama + tallennus tiedostoon kivenseuranta_<paivays>_<aika>.mp4 (tai --ulos oma.mp4)
    python netti_live.py --lahde netti.mp4                         # tiedosto reaaliajassa (nopeus kuten live-lahteessa)
    python netti_live.py --lahde netti.mp4 --nopea --ulos ulos.mp4 --ei-ikkunaa     # simulaatio: ajetaan niin nopeasti kuin ehtii, tulos kirjoitetaan tiedostoon

Toiminta (kaikissa tiloissa sama):
  * Jokainen saapuva ruutu analysoidaan heti (netti_seuranta.Analyysi): kopioruudut pois, kiven tunnistus hackista, kahvan vari + tarkka siluettisovitus
    kiven ymparilla, heiton tunnistus ja liun ja irroituksen viivaestimaatti, joka tarkentuu koko ajan; heiton jalkeen kiven kierre (netti_kierre) ja hog-hog-aika
    videon puoliskoista (kierrosmaara).
  * Tulostettava kuva on analyysia VIIVE_S (oletus 5 s) jäljessa (puskuri). Kun kivi alkaa liikkua tulosteessa, analyysi on jo nahnyt seuraavat
    VIIVE_S sekuntia: toisen paan (kaukopaa) kuvaan piirtyy heti liukulinja ja irroituslinja, ja heittopaassa kiven keskikohta piirtyy pisteina
    (live: piste tulee kun sen ruutu nayttyy), pistejoukko jaa nakyviin JALKI_S (10 s) heiton jalkeen. Tietopaneeli: kivi (vari + numero), liuku, heitto, irroitus,
    kierre, irroituksen etaisyys hog-viivaan, nopeudet ja hidastuvuus.
  * Ruudut sailyvat alkuperaisen nakoisina (1920 x 1080), piirrokset heijastetaan kamerageometrialla (netti_geom.py).
  * Saikeistys: lukija + analyysi (saie 1), piirto (saie 2: JPEG-purku rinnakkain, piirrokset), tallennus (saie 3: ffmpeg-putki tai OpenCV) ja ikkuna (paasaie) toimivat
    toisistaan riippumatta: hidas ikkuna tai levy ei hidasta analyysia eika piirtoa, eika tallenteesta puutu ruutuja; ikkunassa nakyy aina tuorein ruutu.
"""
import argparse
import collections
import concurrent.futures
import csv
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
import traceback

import cv2
import numpy as np

import netti_aani as na
import netti_geom as ng
import netti_piirto as npi
import netti_seuranta as ns

RUUTU_W, RUUTU_H = 1920, 1080
YOUTUBE_OLETUS = "hnTTLqHr5x8"


# ---------------------------------------------------------------------------------------------------------------------
# LAHTEET
# ---------------------------------------------------------------------------------------------------------------------
YT_FORMAATTI = "bestvideo[protocol=https][vcodec^=avc1]/bestvideo[protocol=https]/bestvideo"       # oma testikoodi


YT_VALINTA_RUUTUJA = 12       # kalibrointikansion automaattinen valinta virrasta: mediaanikuva tasmalleen taman monesta ruudusta (haku virrassa on hidas)


def youtube_virta(ref, formaatti=YT_FORMAATTI):
    """YouTube-tunnus tai -osoite -> (virran URL, onko live). Sama kuin oma testikoodi: yt_dlp hakee virran osoitteen, OpenCV (FFmpeg) lukee sen."""
    try:
        import yt_dlp
    except ImportError:
        raise SystemExit("yt_dlp puuttuu: pip install -U yt-dlp")
    url = ref if ("youtube.com" in ref or "youtu.be" in ref) else f"https://www.youtube.com/watch?v={ref}"
    ydl_opts = {
        "format": formaatti,
        "quiet": False,
        "noplaylist": True,
        "js_runtimes": {"deno": {}},
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if info.get("height") and info["height"] > RUUTU_H:
                print(f"Huom: virran korkeus {info['height']} px > {RUUTU_H}; ruudut pienennetaan (kalibrointi on tehty {RUUTU_W} x {RUUTU_H}). "
                      f"Kevyempi: --yt-formaatti 'bestvideo[height<={RUUTU_H}][protocol=https][vcodec^=avc1]/bestvideo[height<={RUUTU_H}]'", file=sys.stderr)
            return info["url"], bool(info.get("is_live"))
    except Exception as e:                                # yt_dlp.utils.DownloadError ym.
        raise SystemExit(f"YouTube-virran haku epaonnistui ({url}): {e}\n"
                         "Tarkista verkko, paivita yt-dlp (pip install -U yt-dlp) ja etta JavaScript-ajuri (deno) on asennettu.")


def valitse_aani(a, lahde):
    """Tallenteen aanilahde (netti_aani.Aanilahde) tai None. Aani liitetaan tallenteeseen lahdevideon aikajanalla (ruutu f <-> aani f / fps), joten se on aina samassa kohdassa kuin kuva
    (tulosteen viive vaikuttaa vain ikkunan nayttohetkeen, ei tiedoston aikajanaan). --aani ei = ei aanta; --aani POLKU / URL = oma aanilahde (sama aikajana kuin video). YouTube:
    aani ladataan taustasaikeessa heti tallenteen avautuessa (Kirjoittaja kaynnistaa latauksen), joten YouTuben hidas lataus ei viivasta lopetusta."""
    if a.aani == "ei":
        return None
    if lahde.live:
        print("Huom: live-virran aanta ei liiteta tallenteeseen (aani ja viivastetty kuva eivat pysyisi samassa kohdassa).", file=sys.stderr)
        return None
    alku = lahde.alku_s if lahde.tiedosto else 0.0
    if a.aani != "auto":
        if na.onko_youtube(a.aani):
            return na.Aanilahde("youtube", a.aani)
        return na.Aanilahde("verkko" if str(a.aani).startswith(("http://", "https://")) else "tiedosto", a.aani, alku)
    if a.youtube:
        return na.Aanilahde("youtube", a.youtube)
    if lahde.tiedosto:
        exe = _etsi_ffmpeg(a.ffmpeg)
        return na.Aanilahde("tiedosto", lahde.tiedosto, alku) if exe is not None and na.tiedot(exe, lahde.tiedosto)[1] else None
    return na.Aanilahde("verkko", a.url) if a.url else None


class Lahde:
    """Ruutujen lahde (tiedosto / YouTube) yhtena iteraattorina (ruutunumero, BGR-ruutu 1920 x 1080). Live-virrassa katkon jalkeen yritetaan avata uudelleen."""

    def __init__(self, tiedosto=None, youtube=None, alku_s=0.0, loppu_s=None, yt_formaatti=YT_FORMAATTI, url=None):
        self.tiedosto, self.youtube, self.yt_formaatti, self.url = tiedosto, youtube, yt_formaatti, url
        self.alku_s, self.loppu_s = alku_s, loppu_s
        self.live = False
        self.cap = None
        self.fps = 30.0
        self._avaa()

    def _avaa(self):
        if self.youtube:
            osoite, self.live = youtube_virta(self.youtube, self.yt_formaatti)
            self.cap = cv2.VideoCapture(osoite, cv2.CAP_FFMPEG)
        elif self.url:                                    # suora virran osoite (http / https / rtsp / m3u8): sama OpenCV-polku kuin YouTubessa
            self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        else:
            self.cap = cv2.VideoCapture(self.tiedosto)
        if not self.cap.isOpened():
            raise SystemExit("Videolahdetta ei voitu avata")
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = float(fps) if fps and 1.0 < fps < 120.0 else 30.0
        if self.tiedosto and self.alku_s:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(self.alku_s * self.fps))

    def ruudut(self):
        f = int(self.alku_s * self.fps) if self.tiedosto else 0           # virroissa ruutunumero alkaa nollasta
        f_loppu = int(self.loppu_s * self.fps) if self.loppu_s else None
        virheita = 0
        varoitettu = False
        while f_loppu is None or f < f_loppu:
            ok, fr = self.cap.read()
            if not ok:
                if not self.live:
                    return
                virheita += 1                             # live: katko -> uudelleenavaus, enintaan 5 yritysta
                if virheita > 5:
                    return
                time.sleep(min(2.0 * virheita, 10.0))
                try:
                    self.cap.release()
                    self._avaa()
                except Exception:
                    pass
                continue
            virheita = 0
            if fr.shape[1] != RUUTU_W or fr.shape[0] != RUUTU_H:
                if not varoitettu:
                    print(f"Huom: ruudun koko {fr.shape[1]} x {fr.shape[0]}, skaalataan {RUUTU_W} x {RUUTU_H} (kalibrointi on tehty talle koolle)", file=sys.stderr)
                    varoitettu = True
                fr = cv2.resize(fr, (RUUTU_W, RUUTU_H), interpolation=cv2.INTER_AREA if fr.shape[1] > RUUTU_W else cv2.INTER_LINEAR)
            yield f, fr
            f += 1


class Esilataus:
    """Ruutujen luku omassa saikeessa: dekoodaus ja verkko (cv2 vapauttaa GIL:n) kulkevat rinnakkain analyysin kanssa ja virran katkot tasoittuvat. Jarjestys ei muutu.

    n: ruutuja luetaan enintaan taman verran analyysin edella raakana (BGR, 6 Mt / ruutu; 0 = ei erillista saiettta).
    pitka_s > 0 (YouTube / virta): luku jatkuu jatkuvasti enintaan pitka_s sekuntia analyysin edelle; puskurin ruudut pakataan YUV 4:2:0 -muotoon (3,1 Mt / ruutu: 30 s = 2,8 Gt),
    joten raskaan hetken (heitto) aikana luettua videota on jo valmiina, ja kevyella hetkella analyysi kirii myohastymisen kiinni. Pakkauksen kautta kulkevat kaikki ruudut (analyysin
    syote ei riipu kuormasta). Purku takaisin BGR:ksi tehdaan omassa saikeessa PURKU_ENNAKKO ruutua etukateen, joten analyysisaie ei kuluta aikaansa siihen (~1,4 ms / ruutu).
    Kaytto: for f, ruutu in Esilataus(...); t_luettu = viimeksi annetun ruudun lukuhetki (time.time()), pituus_s = puskurissa oleva video (s)."""
    PURKU_ENNAKKO = 4

    def __init__(self, iteraattori, n=24, pitka_s=0.0, fps=30.0):
        self.it = iteraattori
        self.n = int(n)
        self.fps = float(fps)
        self.pakattu = pitka_s > 0 and n > 0
        self.t_luettu = time.time()
        self._kapasiteetti = max(self.n, int(round(pitka_s * self.fps))) if self.pakattu else self.n
        self._loppu, self._virhe, self._seis = object(), [], threading.Event()
        self._jono = None
        self._purettu = None                                  # purettujen (BGR) ruutujen lyhyt jono (vain pakatussa tilassa)
        if self.n > 0:
            if self.pakattu:
                k = max(1, min(self.PURKU_ENNAKKO, self._kapasiteetti - 2))
                self._purettu = queue.Queue(maxsize=k)
                self._jono = queue.Queue(maxsize=max(1, self._kapasiteetti - k - 1))          # kokonaispuskuri (kapasiteetti) sisaltaa myos purkusaikeen kasissa olevan ruudun
                threading.Thread(target=self._pura, name="esilataus-purku", daemon=True).start()
            else:
                self._jono = queue.Queue(maxsize=self._kapasiteetti)
            threading.Thread(target=self._lue, name="esilataus", daemon=True).start()

    @property
    def pituus_s(self):
        return 0.0 if self._jono is None else (self._jono.qsize() + (0 if self._purettu is None else self._purettu.qsize())) / self.fps

    def _laita(self, x, jono=None):
        jono = self._jono if jono is None else jono
        while not self._seis.is_set():
            try:
                jono.put(x, timeout=0.2)
                return True
            except queue.Full:
                pass
        return False

    def _lue(self):
        try:
            for f, fr in self.it:
                x = (f, cv2.cvtColor(fr, cv2.COLOR_BGR2YUV_I420), True, time.time()) if self.pakattu else (f, fr, False, time.time())
                if not self._laita(x):
                    return
        except Exception as e:                                # luku katkesi: annetaan poikkeus kuluttajalle
            self._virhe.append(e)
        finally:
            self._laita(self._loppu)

    def _pura(self):
        """Pakatut ruudut (YUV) -> BGR lyhyeen jonoon analyysisaikeen edelle."""
        try:
            while not self._seis.is_set():
                try:
                    x = self._jono.get(timeout=0.2)
                except queue.Empty:
                    continue
                if x is not self._loppu:
                    f, fr, _, t = x
                    x = (f, cv2.cvtColor(fr, cv2.COLOR_YUV2BGR_I420), t)
                if not self._laita(x, self._purettu) or x is self._loppu:
                    return
        except Exception as e:
            self._virhe.append(e)
            self._laita(self._loppu, self._purettu)

    def __iter__(self):
        return self

    def __next__(self):
        if self._jono is None:                                # ei erillista saiettta
            f, fr = next(self.it)
            self.t_luettu = time.time()
            return f, fr
        x = (self._purettu if self._purettu is not None else self._jono).get()
        if x is self._loppu:
            self._seis.set()
            if self._virhe:
                raise self._virhe[0]
            raise StopIteration
        if self._purettu is not None:
            f, fr, self.t_luettu = x
            return f, fr
        f, fr, pakattu, self.t_luettu = x
        return f, fr

    def sulje(self):
        self._seis.set()


def esilataus(iteraattori, n=24, pitka_s=0.0, fps=30.0):
    """Esilataus-iteraattori (ks. Esilataus)."""
    return Esilataus(iteraattori, n, pitka_s, fps)


# ---------------------------------------------------------------------------------------------------------------------
# ULOSTULO: ikkuna (paasaie) ja tallennus (oma saie)
# ---------------------------------------------------------------------------------------------------------------------
def versio_teksti():
    """Ohjelman versio lokiin: git-commit ja -aika (jos kansio on git-repo ja git loytyy), Python ja kayttojarjestelma. Auttaa selvittamaan, mita versiota ajettiin."""
    kansio = os.path.dirname(os.path.abspath(__file__))
    git = ""
    try:
        r = subprocess.run(["git", "-C", kansio, "log", "-1", "--format=%h %cd", "--date=format:%Y-%m-%d %H:%M"], capture_output=True, text=True, timeout=5,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if r.returncode == 0 and r.stdout.strip():
            git = f"git {r.stdout.strip()}"
    except Exception:
        pass
    return f"{git or 'ei git-tietoa'}; Python {sys.version.split()[0]} ({sys.platform})"


def _nayton_koko():
    """Paanayton koko pikseleina (leveys, korkeus) tai None. Windowsissa fyysiset pikselit (DPI-tietoisena), muualla tkinterilla jos saatavilla."""
    try:
        import ctypes
        u = ctypes.windll.user32
        return int(u.GetSystemMetrics(0)), int(u.GetSystemMetrics(1))
    except Exception:
        pass
    try:
        import tkinter
        r = tkinter.Tk()
        r.withdraw()
        koko = (int(r.winfo_screenwidth()), int(r.winfo_screenheight()))
        r.destroy()
        return koko
    except Exception:
        return None


def _dpi_tietoinen():
    """Windows: ilman tata skaalattu nayttö (esim. 125 %) pienentaisi ikkunan kuvaa; DPI-tietoisena ikkunan koko on fyysisina pikseleina."""
    try:
        import ctypes
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


class Naytto:
    """Ikkuna. OpenCV:n GUI-kutsut (imshow, waitKey) vain siita saikeesta, joka luo ikkunan (paasaie): paasilmukka() nayttaa tuoreimman annetun kuvan (anna() mista tahansa
    saikeesta; hitaalla koneella ikkuna ohittaa ruutuja, tallenne ei). Nappaimet: 'q' / ESC lopettaa, 'f' vaihtaa kokoruudun; ikkunan sulkeminen lopettaa.
    tila: 'auto' (kokoruutu jos naytto ei ole ruutua suurempi: kuva 1:1), 'ikkuna' (leveys px; 0 = ruudun koko 1920 x 1080) tai 'kokoruutu'."""
    NIMI = "Kivenseuranta (kuva viiveella)"

    def __init__(self, tila="auto", leveys=0):
        _dpi_tietoinen()
        koko = _nayton_koko()
        kokoruutu = tila == "kokoruutu" or (tila == "auto" and not leveys and koko is not None and koko[0] <= RUUTU_W and koko[1] <= RUUTU_H)
        self.kokoruutu = False
        self.leveys = int(leveys) if leveys else RUUTU_W
        self.koko = koko
        self._lukko = threading.Lock()
        self._kuva = None
        self._uusi = threading.Event()
        self._nahty = 0                                  # ikkunaan piirrettyjen kuvien maara (sulkemisen tunnistus vasta kun ikkuna on ehtinyt aueta)
        self.n_nayt = 0                                  # nayttetyt ruudut (ruutuja voi jaada nayttamatta, jos kone ei ehdi)
        try:
            cv2.namedWindow(self.NIMI, cv2.WINDOW_NORMAL)
        except cv2.error as e:
            raise SystemExit(f"Ikkunaa ei voitu avata ({e}). Onko OpenCV ilman ikkunatukea (opencv-python-headless)? Asenna 'pip install opencv-python' tai kayta --ei-ikkunaa.")
        self._ikkuna_koko()
        if kokoruutu:
            self._aseta_kokoruutu(True)

    def _ikkuna_koko(self):
        w = self.leveys
        if self.koko is not None:                        # ei yli naytön: ikkuna skaalautuu
            w = min(w, max(320, self.koko[0] - 16))
        cv2.resizeWindow(self.NIMI, int(w), int(round(w * RUUTU_H / RUUTU_W)))

    def _aseta_kokoruutu(self, paalle):
        try:
            cv2.setWindowProperty(self.NIMI, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN if paalle else cv2.WINDOW_NORMAL)
            self.kokoruutu = paalle
            if not paalle:
                self._ikkuna_koko()
        except cv2.error:
            pass

    def anna(self, kuva):
        """Seuraava naytettava kuva (mista tahansa saikeesta); edellinen nayttamaton kuva korvautuu."""
        with self._lukko:
            self._kuva = kuva
        self._uusi.set()

    def _ota(self):
        with self._lukko:
            k, self._kuva = self._kuva, None
        return k

    def _sulkeutunut(self):
        if self._nahty < 2:
            return False
        try:
            return cv2.getWindowProperty(self.NIMI, cv2.WND_PROP_VISIBLE) < 1       # suljettu ikkuna: 0 tai -1
        except cv2.error:
            return True                                                             # ikkunaa ei ole enaa

    def _nappain(self, odotus_ms=1):
        """Nappaimen kasittely; palauttaa True jos lopetus on pyydetty."""
        avain = cv2.waitKey(odotus_ms) & 0xFF
        if avain in (ord("q"), ord("Q"), 27):
            return True
        if avain in (ord("f"), ord("F")):
            self._aseta_kokoruutu(not self.kokoruutu)
        return self._sulkeutunut()

    def nayta_heti(self, kuva):
        """Synkroninen: kuva ikkunaan heti (simulaatio, yksi saie); palauttaa True jos lopetus on pyydetty."""
        cv2.imshow(self.NIMI, kuva)
        self._nahty += 1
        self.n_nayt += 1
        return self._nappain(1)

    def paasilmukka(self, lopeta, valmis, ilmoitus=None):
        """Paasaie: nayttaa annetut kuvat ja kuuntelee nappaimia, kunnes lopeta (Event) asetetaan, kayttaja lopettaa tai valmis() on tosi ja kuvat on nahty.
        ilmoitus(): kuva (tai None), jota nautetaan kunnes ensimmainen oikea kuva tulee (puskurointi-ilmoitus)."""
        viim_ilm = 0.0
        while not lopeta.is_set():
            self._uusi.wait(0.005)
            self._uusi.clear()
            kuva = self._ota()
            if kuva is not None:
                cv2.imshow(self.NIMI, kuva)
                self._nahty += 1
                self.n_nayt += 1
            elif self.n_nayt == 0 and ilmoitus is not None and time.time() - viim_ilm > 0.25:
                ilm = ilmoitus()
                if ilm is not None:
                    cv2.imshow(self.NIMI, ilm)
                    self._nahty += 1
                viim_ilm = time.time()
            if self._nappain(1):
                lopeta.set()
                break
            if valmis() and self._kuva is None:
                break

    def sulje(self):
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass


def _etsi_ffmpeg(polku=None):
    """ffmpeg-ohjelma: annettu polku, PATH tai imageio-ffmpeg-paketin mukana tuleva (pip install imageio-ffmpeg); None jos ei loydy."""
    return na.etsi_ffmpeg(polku)


def _ffmpeg_koodaimet(exe):
    try:
        r = subprocess.run([exe, "-hide_banner", "-encoders"], capture_output=True, timeout=20, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return r.stdout.decode("utf-8", "replace")
    except Exception:
        return ""


# laitteistokooderit (naytonohjain tekee pakkauksen: prosessori lahes vapaa); jokainen kokeillaan ennen kayttoa, koska laitteisto ja ffmpeg-versio vaihtelevat
LAITTEISTOKOODERIT = [
    ("h264_nvenc", ["-c:v", "h264_nvenc", "-preset", "p3", "-rc", "vbr", "-cq", "24", "-b:v", "0"]),
    ("h264_qsv", ["-c:v", "h264_qsv", "-global_quality", "24", "-preset", "veryfast"]),
    ("h264_amf", ["-c:v", "h264_amf", "-quality", "speed", "-rc", "cqp", "-qp_i", "22", "-qp_p", "24"]),
    ("h264_videotoolbox", ["-c:v", "h264_videotoolbox", "-q:v", "60"]),
]


def _kokeile_kooderi(exe, args):
    """Toimiiko kooderi taalla: lyhyt testikoodaus nollaan (ffmpeg paattyy virhekoodilla 0)."""
    cmd = [exe, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", f"color=c=black:s={RUUTU_W}x{RUUTU_H}:r=30", "-frames:v", "6", *args, "-pix_fmt", "yuv420p", "-f", "null", "-"]
    try:
        return subprocess.run(cmd, capture_output=True, timeout=30, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).returncode == 0
    except Exception:
        return False


def valitse_kooderi(exe, pyynto="auto", preset="superfast"):
    """(nimi, ffmpeg-argumentit): pyynto 'auto' = laitteistokooderi jos toimii, muuten libx264 (preset), muuten mpeg4; 'cpu' / 'h264' = libx264; 'gpu' = vain laitteisto."""
    koodaimet = _ffmpeg_koodaimet(exe)
    if pyynto in ("auto", "gpu"):
        for nimi, args in LAITTEISTOKOODERIT:
            if nimi in koodaimet and _kokeile_kooderi(exe, args):
                return nimi, args
        if pyynto == "gpu":
            print("Huom: laitteistokooderia ei loytynyt tai se ei toimi; kaytetaan prosessorikoodausta.", file=sys.stderr)
    if "libx264" in koodaimet or not koodaimet:
        return f"libx264, {preset}", ["-c:v", "libx264", "-preset", preset, "-crf", "20", "-threads", "2"]
    return "mpeg4", ["-c:v", "mpeg4", "-q:v", "3"]


class _FfmpegTiedosto:
    """Tallennus ffmpeg-putkeen (raaka BGR stdin -> H.264 mp4): koodaus on erillisessa prosessissa, ei kuormita Pythonin saikeita, ja tiedosto valmistuu vaikka Python kaatuisi."""

    def __init__(self, exe, polku, fps, kooderi):
        nimi, koodaus = kooderi
        self.loki = tempfile.TemporaryFile()
        cmd = [exe, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{RUUTU_W}x{RUUTU_H}", "-r", f"{fps:.4f}", "-i", "-",      # I420: puolet BGR:n datasta putkessa
               "-an", *koodaus, "-pix_fmt", "yuv420p", "-movflags", "+faststart", polku]
        lisa = {}
        if os.name == "nt":                               # kuva odottaa kooderia (tallennusjono): matalaprioriteettinen kooderi nalkiintyisi ruuhkassa muiden saikeiden taakse ja viivastyttaisi kuvaa
            laitteisto = nimi.startswith("h264_")         # (prioriteettikaannos). Laitteistokooderi vie vain vahan prosessoria mutta tarvitsee nopean vuoronsa: yli normaalin; prosessorikooderi: normaali
            lisa = dict(creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) | (0x00008000 if laitteisto else 0))      # ABOVE_NORMAL_PRIORITY_CLASS
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=self.loki, **lisa)
        self.nimi = f"ffmpeg ({nimi})"

    def kirjoita(self, kuva):
        self.p.stdin.write(cv2.cvtColor(kuva, cv2.COLOR_BGR2YUV_I420).data)       # muunnos OpenCV:ssa (SIMD, vapauttaa GIL:n): 3,1 Mt / ruutu putkeen 6,2 Mt:n sijaan, ffmpegin ei tarvitse muuntaa

    def sulje(self):
        try:
            self.p.stdin.close()
        except Exception:
            pass
        rc = self.p.wait()
        if rc != 0:
            self.loki.seek(0)
            raise RuntimeError(f"ffmpeg paattyi virhekoodiin {rc}: {self.loki.read().decode('utf-8', 'replace')[-500:]}")
        self.loki.close()


class _CvTiedosto:
    def __init__(self, polku, fps, koodekki):
        self.vw = cv2.VideoWriter(polku, cv2.VideoWriter_fourcc(*koodekki), fps, (RUUTU_W, RUUTU_H))
        if not self.vw.isOpened():
            raise SystemExit(f"Tiedostoa {polku} ei voitu avata kirjoitettavaksi (koodekki {koodekki}); kokeile --koodekki mp4v tai toista tiedostonimi / kansio")
        self.nimi = f"OpenCV ({koodekki})"

    def kirjoita(self, kuva):
        self.vw.write(kuva)

    def sulje(self):
        self.vw.release()


class Kirjoittaja:
    """Tallennus tiedostoon omassa saikeessa (jono: tallenteesta ei puutu ruutuja, vaikka ikkuna tai analyysi hidastelisi; taynna oleva jono hidastaa piirtoa). koodekki:
    'auto' = ffmpeg (PATH tai imageio-ffmpeg): laitteistokooderi (NVENC / QSV / AMF, kokeillaan ensin) tai H.264 prosessorilla, muuten OpenCV mp4v; 'cpu' / 'h264' = vain
    libx264; 'gpu' = laitteistokooderi (muuten libx264); muu = OpenCV:n fourcc (mp4v, avc1, MJPG ...)."""

    def __init__(self, polku, fps=30.0, koodekki="auto", ffmpeg=None, preset="superfast", aani=None, jono_koko=30):
        self.polku = polku
        self.n = 0
        self.kirjoitus_s = 0.0                            # tallennussaikeen ajankaytto (seinakello / CPU) kooderille syottoon: muunnos + putkeen kirjoitus
        self.kirjoitus_cpu_s = 0.0
        self.virhe = None
        kansio = os.path.dirname(os.path.abspath(polku))
        os.makedirs(kansio, exist_ok=True)
        k = (koodekki or "auto").lower()
        ffmpeg_tila = k in ("auto", "gpu", "cpu", "h264", "x264", "libx264")
        exe = _etsi_ffmpeg(ffmpeg) if ffmpeg_tila else None
        if exe is not None:
            self.taustaosa = _FfmpegTiedosto(exe, polku, fps, valitse_kooderi(exe, "auto" if k in ("auto",) else ("gpu" if k == "gpu" else "cpu"), preset))
        else:
            if ffmpeg_tila:
                print("Huom: ffmpegia ei loytynyt, tallennus OpenCV:n mp4v-koodekilla (iso tiedosto, hitaampi). Vinkki: pip install imageio-ffmpeg (tai asenna ffmpeg) -> H.264, "
                      "pieni tiedosto ja laitteistokoodaus.", file=sys.stderr)
            if aani is not None:
                print("Huom: aani vaatii ffmpegin; tallenne tehdaan ilman aanta.", file=sys.stderr)
            self.taustaosa = _CvTiedosto(polku, fps, "mp4v" if ffmpeg_tila else koodekki)
        self.nimi = self.taustaosa.nimi
        self.aani = aani if exe is not None else None
        self.exe = exe
        if self.aani is not None:
            self.aani.kaynnista(os.path.splitext(os.path.abspath(polku))[0] + ".aani", exe)       # YouTube: lataus taustalle heti (analyysin aikana)
        self.jono_koko = max(1, int(jono_koko))
        self.q = queue.Queue(maxsize=self.jono_koko)         # taynna oleva jono hidastaa piirtoa: iso jono tasoittaa kooderin hetkelliset hidastelut (ei kuvan viivetta)
        self.t = threading.Thread(target=self._silmukka, name="tallennus", daemon=True)
        self.t.start()

    def _silmukka(self):
        while True:
            kuva = self.q.get()
            if kuva is None:
                return
            if self.virhe is not None:
                continue                                  # virheen jalkeen jono vain tyhjennetaan (tuottaja ei jaa jumiin)
            try:
                t0, c0 = time.perf_counter(), time.thread_time()
                self.taustaosa.kirjoita(kuva)
                self.kirjoitus_s += time.perf_counter() - t0
                self.kirjoitus_cpu_s += time.thread_time() - c0
                self.n += 1
            except Exception as e:
                self.virhe = e

    def kirjoita(self, kuva):
        self.q.put(kuva)

    def jonossa(self):
        return self.q.qsize()

    def sulje(self):
        self.q.put(None)
        self.t.join()
        try:
            self.taustaosa.sulje()
        except Exception as e:
            if self.virhe is None:
                self.virhe = e
        if self.virhe is not None:
            print(f"Tallennusvirhe ({self.nimi}): {self.virhe}", file=sys.stderr)
        elif self.aani is not None:
            self._liita_aani()

    def _liita_aani(self):
        """Aani liitetaan valmiiseen kuvatiedostoon vasta lopuksi (netti_aani.liita_tallenteeseen): kuvatallenne pysyy ehjana, vaikka aanilahde ei toimisi, eika sita koskaan lyhenneta."""
        na.liita_tallenteeseen(self.exe, self.polku, self.aani, loki=lambda x: print(x, flush=True))


# ---------------------------------------------------------------------------------------------------------------------
# KIRJAUS
# ---------------------------------------------------------------------------------------------------------------------
def _pilkku(x, des=2, merkki=False):
    if x is None:
        return ""
    s = f"{x:+.{des}f}" if merkki else f"{x:.{des}f}"
    return s.replace(".", ",")


def _aika_teksti(s):
    """Aika muodossa m:ss,d."""
    s = max(float(s), 0.0)
    m = int(s // 60)
    return f"{m}:{s - 60 * m:04.1f}".replace(".", ",")


class Kirjaaja:
    """Tulosten kirjaus konsoliin ja CSV:hen: heiton vahvistus, viivaestimaatti (liuku, heitto, irroitus, nopeudet), kierre (kierrokset hog-hog-valilla)."""

    OTSIKKO = ["heitto", "vari", "kivi_nro", "puoli", "alku_s", "loppu_s", "irroitus_Y_cm", "irroituksen_suunta_deg", "irroituksen_suunta_virhe_deg", "liun_suunta_deg",
               "liuku_T_cm", "heitto_T_cm", "irroitus_delta_cm", "kierre", "kierre_mitattu", "omega_deg_s", "hog_hog_s", "irroitus_ennen_hogia_m",
               "nopeus_ennen_m_s", "nopeus_jalkeen_m_s", "hidastuvuus_m_s2", "liun_hidastuvuus_m_s2", "vakaa",
               # paa, hog-ylitykset ja kierre (heittopaan ja kaukopaan mittaus yhdistettyna)
               "paa", "heitto_paassa", "hog1_s", "hog2_s", "hog2_lahde", "kierre_sigma", "kierre_varma", "kierre_lahde", "kierre_vaimeneminen_1_s", "omega_kauko_deg_s", "kierre_koko_rata", "kierre_koko_sigma",
               # kaukopaan seuranta: pysahtyminen
               "kauko_tila", "pysahtyi_s", "ulos_pelista_s", "lepo_X_cm", "lepo_Y_cm", "lepo_napista_cm", "nopeus_hog2_cm_s", "tormays_s",
               # kiven vakiot (rata_malli.py): kitkataso A ja kaarto kappa kaukopaan radasta
               "vakio_A", "vakio_A_sd", "vakio_kappa", "vakio_kappa_sd", "vakio_kappa_ok", "vakio_rms_y_cm", "vakio_rms_x_cm", "vakio_omega_lahde",
               # joukkueen kello: kello pysahtyy, kun heitetyn kiven etureuna osuu heittopaan T-viivaan (tviiva_s); miettimisaika ennen tata heittoa ja joukkueen aika yhteensa taman heiton jalkeen
               "tviiva_s", "miettimisaika_s", "joukkue_yht_s",
               # irroituksen uskottavuus: nopeus ja suunta hog-viivalla suoraan radasta (vertailu irroituksen arvoihin), ja kaukopaan pysahtymisen perusteella (rata_malli, harjaus <= 2 m): havaittu - ennustettu
               # pysahtymismatka (irr_matka_ero_m), kitkataso jolla malli osuu havaittuun matkaan (irr_A_matka; lahes vakio, jos nopeus on oikea), sivuero omalla kappalla (irr_sivu_ero_cm) ja johtopaatos
               "nopeus_hog1_m_s", "suunta_hog1_deg", "irr_matka_ero_m", "irr_A_matka", "irr_sivu_ero_cm", "irr_tarkistus",
               # heittopaan kierron mittauksen epavarmuus (posteriori): omegan hajonta (deg/s) ja valitun kiertosuunnan todennakoisyys
               "omega_sd_deg_s", "suunta_p",
               # lepopaikan lahde: seuranta (kivi pysahtyi nakyviin) | kartta (kivi hukattiin, lepopaikka loydettiin pesan kartasta, lepo_haku.py: pysahtymishetki tuntematon) | epaselva | ei_loytynyt
               "lepo_lahde",
               # pesan kartasta (lepo_haku.py) loydetty lepopaikka kaikille heitoille, joiden kivi loytyi; ero seurannan pysahtymispaikkaan (cm), kuinka monta ehdokasta oli (1 = yksiselitteinen) ja matka seurannan viimeisesta paikasta (cm; iso matka = tunnistus epavarmempi)
               "lepo_kartta_X_cm", "lepo_kartta_Y_cm", "lepo_kartta_ero_cm", "lepo_kartta_ehdokkaita", "lepo_kartta_matka_cm",
               # maalimerkki (merkki.py): kaukopaan harjanpaan ALIN KOHTA (heittajaa lahin paa) heiton ympariston konsensuksena, heittajan koordinaatistossa (X + = heittajan oikealle, Y = T-viivasta heittajaa kohti, + = ennen T-viivaa), kilpailevien harjojen maara (1 = yksiselitteinen, 2 = kaksi yhta hyvaa harjaa; 0 = ei loytynyt), osuus tavoiteikkunan (-2...+5 s heiton lahdosta) ruuduista, joissa harja nakyy, maalimerkin paikka kaukopaan T-viivalla (cm keskiviivasta, + = heittajan oikealle) kun suora viiva piirretaan heiton lahtohackista maalimerkin kautta, ja 1, kun kuvan reuna katkaisee harjanpaan (paikka likimaarainen)
               "merkki_X_cm", "merkki_Y_cm", "merkki_ehdokkaita", "merkki_esiintyma", "merkki_T_cm", "merkki_reunalla",
               # paneelin liuku ja heitto merkista: (suoran osuma T-viivalla - merkki_T_cm) * kierron merkki, + = kierteesta ulospain (myotapaivaan pyorivalla kivella merkista vasemmalle); tyhja, ellei merkkia tai kierron suuntaa tunneta
               "liuku_merkista_cm", "heitto_merkista_cm"]
    OTSIKKO_KIVET = ["vari", "kivi_nro", "heittoja", "A_mediaani", "A_hajonta", "kappa_heittoja", "kappa_mediaani", "kappa_hajonta", "heitot"]

    def __init__(self, csv_polku=None, hiljaa=False, fps=30.0, tapahtumat=False, radat_polku=None):
        self.csv_polku, self.hiljaa, self.fps = csv_polku, hiljaa, fps
        self.radat_polku = radat_polku                  # heittojen radat ja mittaukset (pickle) jalkianalyysia varten
        self.nayta_tapahtumat = tapahtumat              # tulosta seurannan kaikki tapahtumat (myos hylatyt heittoyritykset)
        self.tapahtumia = 0
        self.ilmoitettu = set()
        self.valmis = set()
        self.kierre_ilm = set()
        self.muuttunut = False
        self.kauko_ilm = set()
        self.kello_ilm = set()
        self.lepo_ilm = set()
        self.csv_sig = None
        self.kivet_polku = (os.path.splitext(csv_polku)[0] + "_kivet.csv") if csv_polku else None
        if csv_polku:
            self._kirjoita_csv(None)

    @staticmethod
    def nimi(h):
        return f"{h.vari.capitalize()} {h.kivi_nro}" if h.kivi_nro is not None else h.vari

    def _tulosta(self, teksti):
        if not self.hiljaa:
            print(teksti, flush=True)

    def paivita(self, an, lopuksi=False):
        """Kutsutaan analyysisaikeesta: ilmoittaa uudet vahvistukset ja valmistuneet tulokset. lopuksi: ajon loppu, ilmoita myos keskenerat."""
        if self.nayta_tapahtumat:
            for e in list(an.tapahtumat)[self.tapahtumia:]:
                self._tulosta(f"  [{e[2] / self.fps:7.1f} s] tapahtuma {e[0]} (yritys {e[1]}) {' '.join(str(x) for x in e[3:])}".rstrip())
            self.tapahtumia = len(an.tapahtumat)
        for h in list(an.heitot):
            n = list(an.heitot).index(h) + 1
            if h.tunnus not in self.ilmoitettu:
                self.ilmoitettu.add(h.tunnus)
                self._tulosta(f"[{h.f_vahvistus / self.fps:7.1f} s] heitto {n}: {self.nimi(h)} ({h.puoli}) vahvistettu, lahto {h.f_alku / self.fps:.1f} s")
            if h.tila == "valmis" and h.tunnus not in self.valmis and (h.viivat_lopullinen or (lopuksi and h.lopullinen)):
                self.valmis.add(h.tunnus)
                self.muuttunut = True
                s = ns.heiton_suureet(h)
                if s is None:
                    self._tulosta(f"          heitto {n} valmis: ei viivaestimaattia (kelvollisia B-mittauksia {len(h.kelvolliset_b())}, rata ei yltanyt irroitukseen tai mittauksia liian vahan)")
                else:
                    v = h.viivat
                    self._tulosta(f"          heitto {n} valmis: irroitus Y={v['irr_Y']:.0f} cm, suunta {v['irr_suunta_deg']:+.2f} +-{v['irr_virhe_deg']:.2f} deg, liuku {v['liuku_suunta_deg']:+.2f} deg, "
                                  f"nopeus {s['v_ennen']:.2f} -> {s['v_jalkeen']:.2f} m/s, hidastuvuus {s['hidastuvuus']:+.3f} m/s2 (liuku {v['liuku_hidastuvuus']:+.3f}); "
                                  f"kaukopaan T-viiva: liuku {s['liuku_cm']:+.0f}, irroitus {s['heitto_cm']:+.0f} cm; irroitus ennen hogia {s['irr_hog_m']:.2f} m")
            if h.tunnus in self.valmis and h.tunnus not in self.kierre_ilm and self._kierre_valmis(an, h, n, lopuksi):
                self.kierre_ilm.add(h.tunnus)
                self.muuttunut = True
            if h.tunnus in self.kierre_ilm and h.tunnus not in self.kauko_ilm and self._kauko_valmis(an, h, n, lopuksi):
                self.kauko_ilm.add(h.tunnus)
                self.muuttunut = True
        for h in list(an.heitot):
            lk = None if h.kauko is None else h.kauko.get("lepo_kartta")
            if lk is not None and h.tunnus not in self.lepo_ilm:
                self.lepo_ilm.add(h.tunnus)
                self.muuttunut = True
                n = list(an.heitot).index(h) + 1
                if "X" in lk and h.kauko["tila"] == "pysahtyi":
                    ero = float(np.hypot(lk["X"] - h.kauko["X_pys"], lk["Y"] - h.kauko["Y_pys"]))
                    if ero > 10.0:
                        self._tulosta(f"          heitto {n} kaukopaa: seurannan lepopaikka (X' {h.kauko['X_pys']:.0f}, Y' {h.kauko['Y_pys']:.0f}) poikkeaa pesan kartasta loydetysta (X' {lk['X']:.0f}, Y' {lk['Y']:.0f}) {ero:.0f} cm")
                elif "X" in lk:
                    self._tulosta(f"          heitto {n} kaukopaa: kivi hukattiin, lepopaikka loytyi pesan kartasta: X' {lk['X']:.0f}, Y' {lk['Y']:.0f} cm ({lk['ero_cm']:.0f} cm viimeisesta tunnetusta paikasta; pysahtymishetkea ei tiedeta)")
                elif h.kauko["tila"] == "pysahtyi":
                    pass
                elif lk.get("epaselva"):
                    self._tulosta(f"          heitto {n} kaukopaa: kivi hukattiin, lepopaikka epaselva ({lk['ehdokkaita']} ehdokasta)")
                else:
                    self._tulosta(f"          heitto {n} kaukopaa: kivi hukattiin, lepopaikkaa ei loytynyt pesan kartasta")
        self._kello_ilmoita(an, lopuksi)
        if self.csv_polku:
            sig = self._allekirjoitus(an)
            if lopuksi or self.muuttunut or sig != self.csv_sig:
                self._kirjoita_csv(an)
                self.csv_sig, self.muuttunut = sig, False

    def _kierre_valmis(self, an, h, n, lopuksi):
        """Kierteen tulostus, kun se on lopullinen (hog-hog-aika mitattu tai kaukopaan haku paattynyt); True kun asia on hoidettu."""
        if not an.laske_kierre or h.viivat is None:
            return True
        if h.kierto_tila == "virhe":
            self._tulosta(f"          heitto {n} kierre: ei saatu ({(h.kierto or {}).get('virhe', '?')})")
            return True
        if h.kierto_tila != "valmis":
            if lopuksi:
                self._tulosta(f"          heitto {n} kierre: ei mitattu")
                return True
            return False
        paattynyt = h.hog2_f is not None or h.saapuminen in ("ei_loytynyt", None) or h.hog1_f is None
        if not paattynyt and not lopuksi:
            return False
        s = ns.heiton_suureet(h)
        k = h.kierto
        suunta = "vastapaivaan" if s["merkki"] > 0 else ("myotapaivaan" if s["merkki"] < 0 else "suunta ?")
        if s["kierre"] is None:
            self._tulosta(f"          heitto {n} kierre: omega {k['omega_deg_s']:+.0f} deg/s ({suunta}), hog-hog-aikaa ei saatu")
        else:
            lahde = "mitattu videosta" if s["kierre_mitattu"] else "arvioitu mallilla (kaukopaan hog-ylitysta ei loytynyt)"
            sd = f" +-{k['omega_sd_deg_s']:.0f}" if k.get("omega_sd_deg_s") is not None else ""
            pv = f", suunta {100.0 * k['p_suunta']:.0f} %" if k.get("p_suunta") is not None else ""
            self._tulosta(f"          heitto {n} kierre: omega {k['omega_deg_s']:+.0f}{sd} deg/s ({suunta}{pv}, ruutuja {k['ruutuja']}, {'varma' if k.get('varma') else 'epavarma'}), "
                          f"hog-hog {s['T_hh']:.1f} s ({lahde}) -> {_pilkku(s['kierre'], 1, True)} kierrosta (- = vastapaivaan); "
                          f"irroitus (heitto - liuku) {_pilkku(s['irroitus_cm'], 0, True)} cm")
        return True

    @staticmethod
    def _allekirjoitus(an):
        """CSV:n sisaltoon vaikuttava tila: kirjoitetaan uusiksi vasta kun jokin heiton tulos (kierre, kaukopaa, vakiot, kello) on muuttunut."""
        return tuple((h.tunnus, h.tila, h.viivat_lopullinen, h.kierto_tila, h.kauko_kierto_tila, h.vakiot_tila, h.kauko is not None, h.hog2_f is not None,
                      None if h.kierre is None else round(h.kierre["N"], 2), h.f_alku, None if h.tviiva_f() is None else round(h.tviiva_f(), 1),
                      None if h.kauko is None else "lepo_kartta" in h.kauko) for h in list(an.heitot))

    def _kauko_valmis(self, an, h, n, lopuksi):
        """Kaukopaan seurannan, kaukopaan kierteen ja kiven vakioiden tulostus, kun ne ovat valmiit; True kun asia on hoidettu."""
        if not an.laske_kierre or h.viivat is None or h.hog1_f is None:
            return True
        k = h.kauko
        kesken = (k is None and h.saapuminen not in ("ei_loytynyt",) and h.hog2_f is None) or h.kauko_kierto_tila == "kaynnissa" or h.vakiot_tila == "kaynnissa"
        if kesken and not lopuksi:
            return False
        if k is None or "virhe" in k:
            self._tulosta(f"          heitto {n} kaukopaa: " + (f"seuranta ei onnistunut ({k['virhe']})" if k is not None else "kivea ei seurattu"))
            return True
        fps = self.fps
        osat = [{"pysahtyi": "pysahtyi", "hukassa": "hukassa (piilossa / pois kuvasta)"}.get(k["tila"], k["tila"])]
        if k["tila"] == "pysahtyi":
            osat[0] += f" {k['f_pys'] / fps:.1f} s paikkaan X' {k['X_pys']:.0f}, Y' {k['Y_pys']:.0f} cm"
        if k["f_hog2"] is not None:
            osat.append(f"hog-viiva {k['f_hog2'] / fps:.2f} s ({k['hog2_lahde']}), nopeus {_pilkku(k['v_hog_cm_s'] / 100.0, 2) if k['v_hog_cm_s'] else '?'} m/s")
        if k.get("tormays"):
            osat.append(f"tormays {k['tormays']['f'] / fps:.1f} s")
        self._tulosta(f"          heitto {n} kaukopaa: " + ", ".join(osat))
        kk = h.kauko_kierto
        if kk is not None and "virhe" not in kk:
            self._tulosta(f"          heitto {n} kaukopaan kierto: omega {kk['omega_deg_s']:+.0f} deg/s ({'varma' if kk.get('varma') else 'epavarma'}, ruutuja {kk['ruutuja']})")
        r = h.kierre
        if r is not None and kk is not None and "virhe" not in kk and h.kierto is not None and "virhe" not in h.kierto:
            self._tulosta(f"          heitto {n} kierrokset hog-hog: {_pilkku(abs(r['N']), 2)} +- {_pilkku(r['N_sigma'], 2)} (omega vaimenee {_pilkku(r['k'] * 100.0, 1)} % / s; "
                          f"{r['omega_lahde']}{', ristiriita' if r['ristiriita'] else ''}{', varma' if r['varma'] else ''})")
        v = h.vakiot
        if v is not None and v.get("ok"):
            kap = f"kappa {_pilkku(v['kappa'], 2)}" if v.get("kappa_ok") else "kappa ? (kierretta ei mitattu)"
            self._tulosta(f"          heitto {n} kiven vakiot: A {_pilkku(v['A'], 4)}, {kap} (jaannos Y {v['rms_y_cm']:.1f} cm, X {v['rms_x_cm']:.1f} cm)")
        elif v is not None:
            self._tulosta(f"          heitto {n} kiven vakioita ei saatu: {v.get('syy', '?')}")
        tk = (v or {}).get("irr_tarkistus") if v is not None else None
        if tk is not None and "virhe" not in tk:
            import netti_vakiot as nvk
            sivu = f", sivusuunnassa {_pilkku(tk['sivu_ero_cm'], 0, True)} cm" if tk.get("sivu_ero_cm") is not None else ""
            self._tulosta(f"          heitto {n} irroituksen uskottavuus: pysahtymismatka irroituksesta {tk['S_havaittu_m']:.1f} m, mallin mukaan {tk['S_malli_m']:.1f} m (ero {_pilkku(tk['matka_ero_m'], 1, True)} m{sivu}; "
                          f"kitkataso {_pilkku(tk['A_matka'], 4)})" + (f" -> EPAILYTTAVA: ero yli {nvk.MATKA_RAJA_M:.0f} m, irroituksen nopeus todennakoisesti vaarin" if tk["epailyttava"] else ""))
        return True

    def _kello_ilmoita(self, an, lopuksi):
        """Heittokohtainen miettimisaika ja joukkueen aika yhteensa konsoliin, kun edellisen heiton pysahtymishetki on selvilla."""
        hs = sorted((h for h in list(an.heitot) if h.f_alku is not None), key=lambda x: x.f_alku)
        if not hs or all(h.tunnus in self.kello_ilm for h in hs):
            return
        kt = an.kello()
        for i, h in enumerate(hs):
            if h.tunnus in self.kello_ilm:
                continue
            k = kt[h.tunnus]
            edell = hs[i - 1] if i else None
            if not lopuksi and an.kello_ruutu(h) is None:
                break                                       # heitetty kivi ei ole viela osunut T-viivaan: kello kay viela
            if edell is not None and edell.puoli == h.puoli and not lopuksi and not an.kauko_selvilla(edell):
                break                                       # edellisen heiton pysahtymishetki ei ole viela lopullinen: odotetaan kaukopaan seurantaa
            self.kello_ilm.add(h.tunnus)
            self.muuttunut = True
            n = list(an.heitot).index(h) + 1
            if edell is None or edell.puoli != h.puoli:
                self._tulosta(f"          heitto {n}: {self.nimi(h)} - paan ensimmainen heitto (kello ei kay paiden valilla); {h.vari} yht {_aika_teksti(k['joukkue_yht_s'])}")
            elif k["aika_s"] is None:
                self._tulosta(f"          heitto {n}: {self.nimi(h)} - miettimisaikaa ei saatu (edellisen kiven pysahtymista ei havaittu)")
            else:
                self._tulosta(f"          heitto {n}: {self.nimi(h)} miettimisaika {_aika_teksti(k['aika_s'])} (kello pysahtyi, kun kivi osui T-viivaan {k['f_stop'] / self.fps:.1f} s; "
                              f"joukkueen {h.vari} aika yhteensa {_aika_teksti(k['joukkue_yht_s'])})")

    def tallenna_radat(self, an):
        """Heittojen radat ja mittaukset pickleen (--radat): heittopaan paikat, viivaestimaatti, kierteen mittaukset, kaukopaan rata ja kiven vakiot; kaukopaan radalle voi ajaa vakiosovituksen
        (netti_vakiot.sovita) uudelleen ilman koko videon analyysia."""
        import pickle
        out = []
        for h in an.heitot:
            ns_ = [n for n in h.naytteet if n.paikka() is not None and h.f_alku is not None and n.f >= h.f_alku]
            out.append(dict(tunnus=h.tunnus, vari=h.vari, puoli=h.puoli, kivi_nro=h.kivi_nro, f_alku=h.f_alku, f_loppu=h.f_loppu, hog1_f=h.hog1_f, hog2_f=h.hog2_f,
                            near=[(n.f, float(n.paikka()[0]), float(n.paikka()[1]), bool(n.kivi())) for n in ns_], viivat=h.viivat, kierto=h.kierto, kauko_kierto=h.kauko_kierto,
                            kierre=h.kierre, kauko=h.kauko, vakiot=h.vakiot, saapuminen=h.saapuminen, merkki=h.merkki,
                            viivat_syote=[(n.f, float(n.B[0]), float(n.B[1])) for n in h.kelvolliset_b()],            # laske_viivat:n tarkka syote: irroituksen tunnistuksen voi ajaa uudelleen (tools/sovita_irroitus.py)
                            kahva_ala=[(n.f, float(n.ala), float(n.A[0]), float(n.A[1])) for n in h.naytteet if n.A is not None and n.ala]))
        with open(self.radat_polku, "wb") as fh:
            pickle.dump(out, fh)
        self._tulosta(f"Radat ja mittaukset tallennettu: {self.radat_polku} ({len(out)} heittoa)")

    def loppuyhteenveto(self, an):
        """Ajon lopuksi: joukkueiden miettimisaika yhteensa (kello kay, kun edellisen heiton kivi on pysahtynyt, ja pysahtyy heiton lahtiessa; ei kay paiden valilla) ja kaukopaan seurannan tulos."""
        if self.radat_polku and an.heitot:
            self.tallenna_radat(an)
        if self.hiljaa or not an.heitot:
            return
        kt = an.kello()
        yht, n = {}, {}
        for h in an.heitot:
            k = kt.get(h.tunnus)
            if k is not None and k["aika_s"] is not None:
                yht[h.vari] = yht.get(h.vari, 0.0) + k["aika_s"]
                n[h.vari] = n.get(h.vari, 0) + 1
        if yht:
            print("Joukkueiden kello (miettimisaika yhteensa): " + "; ".join(f"{v.capitalize()} {_aika_teksti(a)} ({n[v]} heittoa, keskimaarin {_aika_teksti(a / n[v])})" for v, a in sorted(yht.items())), flush=True)
        kv = an.kivien_vakiot()
        if kv:
            print(f"Kivikohtaiset vakiot: {len(kv)} kivea (A = kitkataso, kappa = kaarto; tiedosto {self.kivet_polku or '-'})", flush=True)

    def historia(self, an):
        """Estimaattien kehitys heitoittain (analyysiaika lahdosta, irroitus, suunta, T-viivan osumat): nayttaa miten alustava arvio tarkentuu."""
        if self.hiljaa:
            return
        for i, h in enumerate(an.heitot):
            print(f"heiton {i + 1} estimaattien kehitys (aika lahdosta, irroitus Y, irroituksen suunta, T-viiva irroitus / liuku, vakaa):")
            for f, Y, th, XT_i, XT_l, vakaa in h.viivat_loki[:: max(1, len(h.viivat_loki) // 14)]:
                print(f"   +{(f - h.f_alku) / self.fps:5.1f} s: Y={Y:5.0f} cm  {th:+6.2f} deg  T-viiva {XT_i:+5.0f} / {XT_l:+5.0f} cm  {'vakaa' if vakaa else 'alustava'}")

    @staticmethod
    def _n(x, des=2):
        return "" if x is None else f"{x:.{des}f}"

    def _rivi(self, n, h, kt, paa):
        """CSV-rivi heitolle h (n = jarjestysnumero); viivaestimaatin sarakkeet tyhjia, ellei estimaattia ole."""
        N = self._n
        s = ns.heiton_suureet(h)
        v = h.viivat
        fps = self.fps
        rivi = [n, h.vari, h.kivi_nro, h.puoli, f"{h.f_alku / fps:.2f}", f"{h.f_loppu / fps:.2f}"]
        if s is not None and v is not None:
            rivi += [f"{v['irr_Y']:.1f}", f"{v['irr_suunta_deg']:.3f}", f"{v['irr_virhe_deg']:.3f}", f"{v['liuku_suunta_deg']:.3f}", f"{s['liuku_cm']:.1f}", f"{s['heitto_cm']:.1f}",
                     N(s["irroitus_cm"], 1), N(s["kierre"]), int(s["kierre_mitattu"]), N(s["omega_deg_s"], 1), N(s["T_hh"]), f"{s['irr_hog_m']:.2f}",
                     f"{s['v_ennen']:.3f}", f"{s['v_jalkeen']:.3f}", f"{s['hidastuvuus']:.3f}", f"{v['liuku_hidastuvuus']:.3f}", int(s["vakaa"])]
        else:
            rivi += [""] * 17
        r = h.kierre
        k = h.kauko if (h.kauko is not None and "virhe" not in h.kauko) else None
        rivi += [paa["paa"], paa["heitto_paassa"], N(None if h.hog1_f is None else h.hog1_f / fps), N(None if h.hog2_f is None else h.hog2_f / fps),
                 "" if k is None or k["hog2_lahde"] is None else k["hog2_lahde"],
                 N(None if r is None else r["N_sigma"]), "" if r is None else int(bool(r["varma"])), "" if r is None else r["omega_lahde"], N(None if r is None else r["k"], 4),
                 N(None if s is None else s["omega_kauko_deg_s"], 1), N(None if s is None else s["kierre_koko"]), N(None if s is None or s["kierre_koko"] is None else s["kierre_koko_sigma"])]
        if k is not None:
            lepo = k["tila"] == "pysahtyi"                  # lepopaikka heittajan koordinaatistossa: X + = heittajan oikealle (kaukopaan X' on peilikuva), Y = T-viivasta heittajaa kohti (+ = ennen T-viivaa)
            lk = k.get("lepo_kartta")
            kartasta = (not lepo) and lk is not None and "X" in lk                  # hukattu kivi, jonka lepopaikka loytyi pesan kartasta (aikaa ei tiedeta)
            X_l, Y_l = (k["X_pys"], k["Y_pys"]) if lepo else ((lk["X"], lk["Y"]) if kartasta else (None, None))
            rivi += [k["tila"], N(k["f_pys"] / fps) if lepo else "", N(None if k.get("f_ulos") is None else k["f_ulos"] / fps), N(-X_l, 1) if X_l is not None else "", N(Y_l, 1) if X_l is not None else "",
                     N(float(np.hypot(X_l, Y_l)), 1) if X_l is not None else "", N(k["v_hog_cm_s"], 1), N(k["tormays"]["f"] / fps) if k.get("tormays") else ""]
        else:
            rivi += [""] * 8
        vk = h.vakiot
        if vk is not None and vk.get("ok"):
            rivi += [N(vk["A"], 5), N(vk.get("A_sd"), 5), N(vk["kappa"]) if vk.get("kappa_ok") else "", N(vk.get("kappa_sd")) if vk.get("kappa_ok") else "", int(bool(vk.get("kappa_ok"))),
                     N(vk["rms_y_cm"], 1), N(vk["rms_x_cm"], 1), vk.get("omega_lahde") or ""]
        else:
            rivi += [""] * 8
        rivi += [N(None if kt["f_stop"] is None else kt["f_stop"] / fps), N(kt["aika_s"], 1), N(kt["joukkue_yht_s"], 1)]
        tk = (vk or {}).get("irr_tarkistus") if vk is not None else None
        if tk is not None and "virhe" not in tk:
            teksti = "epailyttava" if tk["epailyttava"] else "ok"
            rivi += [N(None if v is None else v.get("nopeus_hog1"), 3), N(None if v is None else v.get("suunta_hog1_deg"), 3), N(tk["matka_ero_m"], 2), N(tk["A_matka"], 5), N(tk.get("sivu_ero_cm"), 0), teksti]
        else:
            rivi += [N(None if v is None else v.get("nopeus_hog1"), 3), N(None if v is None else v.get("suunta_hog1_deg"), 3), "", "", "", ""]
        rivi += [N(None if s is None else s["omega_sd_deg_s"], 1), N(None if s is None else s["suunta_p"], 4)]
        if k is None:
            rivi += [""] * 6
        else:
            lk = k.get("lepo_kartta")
            if k["tila"] == "pysahtyi":
                rivi.append("seuranta")
            else:
                rivi.append("" if lk is None else ("kartta" if "X" in lk else ("epaselva" if lk.get("epaselva") else "ei_loytynyt")))
            if lk is not None and "X" in lk:
                ero = float(np.hypot(lk["X"] - k["X_pys"], lk["Y"] - k["Y_pys"])) if k["tila"] == "pysahtyi" else None
                rivi += [N(-lk["X"], 1), N(lk["Y"], 1), N(ero, 1), lk["ehdokkaita"], N(lk["ero_cm"], 0)]
            else:
                rivi += [""] * 5
        mk = h.merkki
        if mk is None:
            rivi += [""] * 6
        elif "X" in mk:
            rivi += [N(-mk["X"], 1), N(mk["Y"], 1), mk["ehdokkaita"], N(mk["esiintyma"], 2), N(mk.get("T_cm"), 1), int(bool(mk.get("reunalla")))]
        else:
            rivi += ["", "", 0, "", "", ""]
        rivi += [N(None if s is None else s["liuku_merkista_cm"], 1), N(None if s is None else s["heitto_merkista_cm"], 1)]
        return rivi

    def _kirjoita_csv(self, an):
        """Koko CSV uusiksi: kaikki valmistuneet heitot (viivaestimaatin sarakkeet tyhjia, jos estimaattia ei saatu); kaukopaan tulokset, kierre, vakiot ja kello taydentyvat heiton
        valmistumisen jalkeen. Lisaksi kivikohtaiset vakiot (<csv>_kivet.csv)."""
        heitot = [h for h in (list(an.heitot) if an is not None else []) if h.tunnus in self.valmis and h.f_alku is not None and h.f_loppu is not None]
        kt = an.kello() if an is not None and heitot else {}
        with open(self.csv_polku, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(self.OTSIKKO)
            for h in heitot:
                w.writerow(self._rivi(list(an.heitot).index(h) + 1, h, kt[h.tunnus], kt[h.tunnus]))
        if self.kivet_polku and an is not None:
            with open(self.kivet_polku, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(self.OTSIKKO_KIVET)
                for (vari, nro), d in an.kivien_vakiot().items():
                    w.writerow([vari, nro, d["n"], self._n(d["A"], 5), self._n(d["A_sd"], 5), d["n_kappa"], self._n(d["kappa"]), self._n(d["kappa_sd"]), ";".join(str(t) for t in d["heitot"])])


# ---------------------------------------------------------------------------------------------------------------------
# SIMULAATIO: ajetaan niin nopeasti kuin ehtii, tulos riippumaton ajoituksesta (tyot odotetaan)
# ---------------------------------------------------------------------------------------------------------------------
def aja_sim(a, lahde, an, piirtaja, naytto, kirjoittaja, kirjaaja):
    D = int(round(a.viive * lahde.fps))
    cap2 = cv2.VideoCapture(lahde.tiedosto)          # toinen kahva: tulostettava kuva luetaan alkuperaisena (ei JPEG-pakkausta), D ruutua analyysin jalkeen
    cap2.set(cv2.CAP_PROP_POS_FRAMES, int(lahde.alku_s * lahde.fps))
    t0 = time.time()
    n = 0
    kello_loki = open(a.kello_loki, "w", encoding="utf-8") if getattr(a, "kello_loki", None) else None       # (testaus) kuvassa nakyvan kellon tila kerran sekunnissa
    if kello_loki is not None:
        kello_loki.write("aika_s,kay,juokseva_s,viim_heitto\n")

    def tulosta(f_out):
        ok, kuva = cap2.read()
        if not ok:
            return False
        piirtaja.piirra(kuva, f_out, an)
        if kirjoittaja is not None:
            kirjoittaja.kirjoita(kuva)
        return not (naytto is not None and naytto.nayta_heti(kuva))

    for f, fr in lahde.ruudut():
        an.syota(f, fr)
        n += 1
        kirjaaja.paivita(an)
        f_out = f - D
        if f_out >= int(lahde.alku_s * lahde.fps):
            if kello_loki is not None and an.heitot and f_out % int(round(lahde.fps)) == 0:
                kn = an.kello_nyt(f_out)
                kello_loki.write(f"{f_out / lahde.fps:.1f},{kn['kay'] or ''},{'' if kn['juokseva_s'] is None else round(kn['juokseva_s'], 1)},{'' if kn['viim'] is None else kn['viim'].tunnus}\n")
            if not tulosta(f_out):
                break
        if n % 300 == 0 and not a.hiljaa:
            print(f"  ... {n / lahde.fps:6.1f} s videota, {n / (time.time() - t0):.0f} r/s", flush=True)
    an.viimeistele()
    kirjaaja.paivita(an, lopuksi=True)
    kirjaaja.historia(an)
    kirjaaja.loppuyhteenveto(an)
    f_alku_out = int(lahde.alku_s * lahde.fps)
    viim = f_alku_out + n - D
    for f_out in range(max(viim, f_alku_out), f_alku_out + n):         # viimeiset D ruutua
        if not tulosta(f_out):
            break


# ---------------------------------------------------------------------------------------------------------------------
# REAALIAIKA: lukija + analyysi (saie), piirto (saie), tallennus (saie), ikkuna (paasaie)
# ---------------------------------------------------------------------------------------------------------------------
def aja_reaaliaika(a, lahde, an, piirtaja, naytto, kirjoittaja, kirjaaja):
    fps = lahde.fps
    t_ajo = time.time()
    jono = collections.deque()                       # (f, saapumisaika, pakkauksen tulevaisuus (laji, data))
    lopetettu = threading.Event()                    # lahde loppui
    pysayta = threading.Event()                      # kayttaja lopettaa tai virhe
    piirto_valmis = threading.Event()                # piirtosaie on antanut kaikki kuvat
    tilasto = dict(analysoitu=0, ohitettu=0, virheita=0, piirretty=0, myohassa_max=0.0, luku_s=0.0, luku_max=0.0, pysahdyksia=0, syota_s=0.0, syota_ms=[], viive_max=0.0, puskuri_max=0.0, analyysia_odotettu_s=0.0,
                  piirto_s=0.0, purku_odotus_s=0.0, kirjoitus_odotus_s=0.0, tallennus=(kirjoittaja.nimi if kirjoittaja is not None else None),      # piirtosaikeen ajankaytto: piirto, pakkauksen odotus, tallennusjonon odotus
                  syota_cpu_s=0.0, piirto_cpu_s=0.0, kirj_s=0.0, kirj_cpu_s=0.0, kirj_n=0.0, kiri_ruutuja=0, kiri_odotus_s=0.0, video_s=0.0, ajo_s=0.0)       # saikeiden prosessoriaika (verrattuna seinakelloon: odottaako saie vai laskeeko se) ja tallennussaikeen syotto kooderille
    jaettu = dict(puskuri_s=0.0)                     # analyysisaie -> piirtosaie: esilatauspuskurin pituus (s)
    pitka_s = a.lataus if (a.youtube or a.url) and a.esilataus > 0 else 0.0              # virta: video ladataan jatkuvasti enintaan taman verran analyysin edelle
    max_myohassa = a.max_myohassa if a.max_myohassa is not None else (0.8 if pitka_s <= 0 else min(6.0, max(0.8, a.viive - 3.0)))     # ruutu ohitetaan (ei analysoida), jos analyysi on taman verran myohassa
    piirtoa_tarvitaan = naytto is not None or kirjoittaja is not None
    pakkaus = concurrent.futures.ThreadPoolExecutor(2, thread_name_prefix="puskuri-pakkaus")     # cv2-kutsut vapauttavat GIL:n: rinnakkain analyysin kanssa
    purku = concurrent.futures.ThreadPoolExecutor(2, thread_name_prefix="puskuri-purku")
    muisti_gt = fyysinen_muisti_gt()
    puskuri_muoto = valitse_puskuri(a.puskuri, muisti_gt)                          # auto -> bgr (paljon muistia) tai yuv
    if not a.hiljaa and piirtoa_tarvitaan:
        print(f"Viivepuskuri: {puskuri_muoto}" + (f" (--puskuri auto, muistia {muisti_gt:.0f} Gt)" if a.puskuri == "auto" and muisti_gt is not None else ""), flush=True)
    D_ruutua = int(round(a.viive * fps))             # analyysi on taman verran kuvan edella
    bgr_raja, yuv_raja = puskuri_rajat(a.viive, fps, muisti_gt)       # muistin suoja: jonon kasvaessa (kone ei jaksa) uudet ruudut pakataan kevyemmin (bgr -> yuv -> jpeg)
    kiri_raja_s = a.kiri if pitka_s > 0 else 0.0                       # kiri vain VOD-virrassa, jossa on pitka esilatauspuskuri (YouTube / --url)
    kiri_tila = kiri_raja_s > 0 and not lahde.live and not a.ei_tahdistusta
    kiri_jono_max = max(D_ruutua + int(KIRI_JONO_MARGINAALI_S * fps), min(D_ruutua + int(a.etumatka * fps), (bgr_raja if puskuri_muoto == "bgr" else yuv_raja) - int(fps)))     # analyysin etumatka kuvaan (ruutua)
    aikataulu = KuvanAikataulu(fps, kiri_raja_s) if kiri_tila else None
    if kiri_tila and not a.hiljaa:
        print(f"Kiri: analyysi etenee ilman reaaliaikaan sidontaa (etumatka kuvaan enintaan {(kiri_jono_max - D_ruutua) / fps:.0f} s); kuva etenee 1x, mutta saa kiria kun esilatauspuskurissa on vahintaan {kiri_raja_s:.0f} s "
              "videota tai lahde loppui (--kiri 0 = pois)", flush=True)

    def pakkaa(fr, muoto):
        return pakkaa_ruutu(fr, muoto, a.jpeg)

    def pura(tulevaisuus):
        return pura_ruutu(*tulevaisuus.result())

    def analyysi_saie():
        t_alku = time.time()
        f_ensin = None
        t_ensin_ajo, viim_f_luettu = time.time(), None
        t_edell = time.perf_counter()
        ohitus_kaynnissa = False
        it = None
        try:
            it = esilataus(lahde.ruudut(), a.esilataus, pitka_s, fps)
            while True:
                t_luku0 = time.perf_counter()
                try:
                    f, fr = next(it)
                except StopIteration:
                    break
                dt_luku = time.perf_counter() - t_luku0
                tilasto["luku_s"] += dt_luku
                tilasto["luku_max"] = max(tilasto["luku_max"], dt_luku)
                if dt_luku > 0.25:
                    tilasto["pysahdyksia"] += 1                      # lahde (verkko / dekoodaus) viipyi: kone ei ehka ole hidas vaan virta takkuaa
                if pysayta.is_set():
                    break
                if a.debug_viive and f_ensin is not None and time.perf_counter() - t_edell > 0.2:
                    print(f"  [debug] ruutujen valissa {1e3 * (time.perf_counter() - t_edell):.0f} ms ennen ruutua {f} ({f / fps:.1f} s)", flush=True)
                t_edell = time.perf_counter()
                if f_ensin is None:
                    f_ensin = f
                    t_alku = t_ensin_ajo = time.time()                # tahdistuksen alku: ensimmaisen ruudun saapuminen (kaynnistyksen viive ei ole myohastymista)
                viim_f_luettu = f
                jaettu["puskuri_s"] = it.pituus_s
                tilasto["puskuri_max"] = max(tilasto["puskuri_max"], it.pituus_s)
                if kiri_tila:                                             # VOD-virta (kiri): analyysi ei odota reaaliaikaa vaan etenee niin nopeasti kuin ehtii; kuvan aikataulu on piirtosaikeella.
                    while piirtoa_tarvitaan and len(jono) >= kiri_jono_max and not (pysayta.is_set() or piirto_valmis.is_set()):
                        time.sleep(0.005)                                 # vastapaine: etumatka kuvaan on rajattu (viivepuskurin muisti), joten odotetaan kunnes kuva on kuluttanut jonoa
                        jaettu["puskuri_s"] = it.pituus_s                 # (lukija taydentaa esilatauspuskuria odotuksen aikana: kuvan aikataulu tarvitsee tuoreen arvon)
                    erapaiva = time.time()
                    myohassa = 0.0
                elif not lahde.live and not a.ei_tahdistusta:         # tiedosto / VOD-virta ilman kiria: tahdistus kuin live-virrassa
                    erapaiva = t_alku + (f - f_ensin) / fps               # ruudun oma aika (hetki, jolloin se olisi reaaliajassa)
                    odotus = erapaiva - time.time()
                    if odotus > 0:
                        time.sleep(odotus)
                    myohassa = -odotus
                elif lahde.live:                                       # live: ruutu on myohassa sen verran kuin se on odottanut puskurissa
                    erapaiva = it.t_luettu
                    myohassa = max(0.0, time.time() - erapaiva)
                else:
                    erapaiva = time.time()
                    myohassa = 0.0
                if piirtoa_tarvitaan:
                    jono.append((f, time.time(), pakkaus.submit(pakkaa, fr, puskurin_muoto(puskuri_muoto, len(jono), bgr_raja, yuv_raja)), erapaiva))
                if myohassa > max_myohassa and not an.toistettava:           # analyysi myohassa: ruutu nakyy tulosteessa, mutta sita ei analysoida (toistettavassa ajossa analysoidaan kaikki ruudut ja kuva odottaa)
                    tilasto["ohitettu"] += 1
                    if a.debug_viive and not ohitus_kaynnissa:
                        print(f"  [debug] ohitus alkaa: ruutu {f} ({f / fps:.1f} s), myohassa {myohassa:.2f} s", flush=True)
                    ohitus_kaynnissa = True
                    an.ohita()
                    continue
                if ohitus_kaynnissa and a.debug_viive:
                    print(f"  [debug] ohitus loppuu: ruutu {f} ({f / fps:.1f} s), myohassa {myohassa:.2f} s", flush=True)
                ohitus_kaynnissa = False
                try:
                    t_syota, c_syota = time.perf_counter(), time.thread_time()
                    an.syota(f, fr)
                    dt_syota = time.perf_counter() - t_syota
                    tilasto["syota_cpu_s"] += time.thread_time() - c_syota
                    tilasto["syota_s"] += dt_syota
                    tilasto["syota_ms"].append(1e3 * dt_syota)
                    tilasto["analysoitu"] += 1
                    kirjaaja.paivita(an)
                    if a.debug_viive and time.perf_counter() - t_syota > 0.12:
                        print(f"  [debug] hidas analyysi: ruutu {f} ({f / fps:.1f} s) {1e3 * (time.perf_counter() - t_syota):.0f} ms, myohassa {myohassa:.2f} s", flush=True)
                except Exception:                                   # yksittainen virhe ei saa pysayttaa lahdetta ja tulostetta
                    tilasto["virheita"] += 1
                    if tilasto["virheita"] <= 3:
                        print("Analyysivirhe (ruutu %d), jatketaan:" % f, file=sys.stderr)
                        traceback.print_exc()
        except Exception:
            print("Lahteen luku keskeytyi:", file=sys.stderr)
            traceback.print_exc()
        finally:
            if it is not None:
                it.sulje()
            if f_ensin is not None and viim_f_luettu is not None:
                tilasto["video_s"] = (viim_f_luettu - f_ensin + 1) / fps          # analyysin eteneminen: videota / seinakelloa (kiri nostaa yli 1:n)
                tilasto["ajo_s"] = time.time() - t_ensin_ajo
            lopetettu.set()
        # lahde loppui: taustatyot (B, estimaatit, kierre, kaukopaa) etenevat kunnes tuloste on piirretty
        try:
            raja = time.time() + 120.0
            while not pysayta.is_set() and not piirto_valmis.is_set() and time.time() < raja:
                an.odota()
                kirjaaja.paivita(an)
                time.sleep(0.05)
        except Exception:
            traceback.print_exc()

    def piirto_saie():
        odottaa = collections.deque()                # (f, erapaiva, purkutehtava)
        piirtovirhe = []
        varoitettu = False
        t_silmukka = time.time()
        t_tila, edellinen = time.time(), tilasto_otos(tilasto)           # viive-tilarivi: viimeisin tulostus ja silloinen tilasto
        kuorma = KoneenKuorma()                                          # koko koneen ja tamän ohjelman prosessorinkaytto tilariveille
        t_viivat = None                                                  # kiri: hetki, jolloin jonon ensimmainen kuva alkoi odottaa lopullista viivaestimaattia
        try:
            while not pysayta.is_set():
                nyt = time.time()
                dt_silmukka, t_silmukka = nyt - t_silmukka, nyt
                while jono and len(odottaa) < 6 and (kiri_tila or jono[0][1] + a.viive - nyt < 0.4):       # purku etukateen (rinnakkain)
                    f, t_s, tulevaisuus, erapaiva_ruutu = jono.popleft()
                    odottaa.append((f, t_s + a.viive, purku.submit(pura, tulevaisuus), erapaiva_ruutu))
                nayta, sprintti, aikataulu_ok, analyysi_ok = False, False, False, True
                if odottaa:
                    f0 = odottaa[0][0]
                    viim_f = getattr(an, "viim_f", None) or 0
                    if kiri_tila:                                      # kiri: kuvalla oma aikataulu (1x tasaisesti; kiri vain kun esilatauspuskurissa >= raja s tai lahde loppui)
                        erapaiva_nyt, sprintti = aikataulu.aika(f0, nyt, jaettu["puskuri_s"], odottaa[0][1], lopetettu.is_set())
                    else:
                        erapaiva_nyt = odottaa[0][1]
                    aikataulu_ok = erapaiva_nyt <= nyt
                    analyysi_ok = not an.toistettava or lopetettu.is_set() or viim_f >= f0 + D_ruutua - 3          # toistettavassa ajossa kuva odottaa analyysia (piirrokset tulevat analyysista)
                    nayta = aikataulu_ok and analyysi_ok
                    if nayta and sprintti and an.viivat_kesken(f0):    # kirissa analyysi etenee kuvaa nopeammin: heiton lopullinen viivaestimaatti (tyoprosessi) voi olla vasta tulossa -- odotetaan se (rajatusti)
                        t_viivat = nyt if t_viivat is None else t_viivat
                        if nyt - t_viivat < KIRI_VIIVAT_ODOTUS_S:
                            nayta = False
                            tilasto["kiri_odotus_s"] += dt_silmukka
                    elif nayta:
                        t_viivat = None
                if nayta:
                    f, erapaiva, tehtava, erapaiva_ruutu = odottaa.popleft()
                    if sprintti:
                        tilasto["kiri_ruutuja"] += 1
                    t_a = time.perf_counter()
                    kuva = tehtava.result()
                    tilasto["purku_odotus_s"] += time.perf_counter() - t_a
                    if kiri_tila:                                  # kiri: kuvan oma aika tulee aikataulusta (kirissa nyt: ei myohassa)
                        myohassa = max(0.0, nyt - erapaiva_nyt)
                        piirtaja.viive_nyt = myohassa
                    else:
                        myohassa = nyt - erapaiva
                        piirtaja.viive_nyt = max(0.0, time.time() - (erapaiva_ruutu + a.viive))      # kuinka paljon myohemmin kuva nakyy kuin sen oma aika + viive (analyysin ja piirron myohastyminen)
                    piirtaja.puskuri_s = jaettu["puskuri_s"] if pitka_s > 0 else None
                    tilasto["viive_max"] = max(tilasto["viive_max"], piirtaja.viive_nyt)
                    tilasto["myohassa_max"] = max(tilasto["myohassa_max"], myohassa)
                    if myohassa > 3.0 and not varoitettu:
                        varoitettu = True
                        print(viive_varoitus(myohassa, tilasto, an, fps), file=sys.stderr)
                    t_a, c_a = time.perf_counter(), time.thread_time()
                    try:
                        npi_kuva = piirtaja.piirra(kuva, f, an)
                    except Exception:                               # piirtovirhe: nayta ruutu ilman (loppuun asti piirrettyja) piirroksia
                        npi_kuva = kuva
                        if not piirtovirhe:
                            print("Piirtovirhe, jatketaan ilman piirroksia:", file=sys.stderr)
                            traceback.print_exc()
                            piirtovirhe.append(1)
                    tilasto["piirto_s"] += time.perf_counter() - t_a
                    tilasto["piirto_cpu_s"] += time.thread_time() - c_a
                    if naytto is not None:
                        naytto.anna(npi_kuva)                          # ikkuna ensin: tallennusjonon odotus ei viivasta jo valmista kuvaa
                    if kirjoittaja is not None:
                        t_a = time.perf_counter()
                        kirjoittaja.kirjoita(npi_kuva)                 # (blokkaa, kun tallennusjono on taynna: kooderi ei ehdi)
                        tilasto["kirjoitus_odotus_s"] += time.perf_counter() - t_a
                        tilasto["kirj_s"], tilasto["kirj_cpu_s"], tilasto["kirj_n"] = kirjoittaja.kirjoitus_s, kirjoittaja.kirjoitus_cpu_s, float(kirjoittaja.n)
                    tilasto["piirretty"] += 1
                    if not a.hiljaa and piirtaja.viive_nyt > 1.0 and time.time() - t_tila >= TILARIVI_VALI_S:
                        print(viive_tilarivi(time.time() - t_ajo, piirtaja.viive_nyt, tilasto, edellinen, (kirjoittaja.jonossa(), kirjoittaja.jono_koko) if kirjoittaja is not None else None, jaettu["puskuri_s"] if pitka_s > 0 else None,
                                             kuorma.erittely()), file=sys.stderr, flush=True)
                        t_tila, edellinen = time.time(), tilasto_otos(tilasto)
                    continue
                if odottaa and aikataulu_ok and not analyysi_ok:
                    tilasto["analyysia_odotettu_s"] += dt_silmukka                # kuva olisi esitettavissa, mutta analyysi ei ole viela ehtinyt mukaan
                if lopetettu.is_set() and not jono and not odottaa:
                    break
                time.sleep(0.002)
        except Exception:
            print("Piirtosaie keskeytyi:", file=sys.stderr)
            traceback.print_exc()
        finally:
            piirto_valmis.set()

    def ilmoitus():
        """Puskurointi-ilmoitus ennen ensimmaista kuvaa."""
        try:
            jaljella = max(0.0, jono[0][1] + a.viive - time.time())
        except IndexError:
            return None
        kuva = np.zeros((RUUTU_H, RUUTU_W, 3), np.uint8)
        cv2.putText(kuva, f"Puskuroidaan {jaljella:.1f} s (kuva tulee {a.viive:.0f} s viiveella)", (120, RUUTU_H // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 2, cv2.LINE_AA)
        return kuva

    s1 = threading.Thread(target=analyysi_saie, name="analyysi", daemon=True)
    s2 = threading.Thread(target=piirto_saie, name="piirto", daemon=True)
    if not piirtoa_tarvitaan:
        piirto_valmis.set()
    try:
        s1.start()
        if piirtoa_tarvitaan:
            s2.start()
        if naytto is not None:
            naytto.paasilmukka(pysayta, lambda: piirto_valmis.is_set() and not s2.is_alive(), ilmoitus)
        else:
            while (s2.is_alive() if piirtoa_tarvitaan else s1.is_alive()):
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("Keskeytetty (Ctrl-C), viimeistellaan ...", file=sys.stderr)
    finally:
        pysayta.set()
        s1.join(timeout=10.0)
        if piirtoa_tarvitaan:
            s2.join(timeout=10.0)
        pakkaus.shutdown(wait=False, cancel_futures=True)
        purku.shutdown(wait=False, cancel_futures=True)
        an.viimeistele()
        kirjaaja.paivita(an, lopuksi=True)
        kirjaaja.loppuyhteenveto(an)
        if not a.hiljaa and tilasto["syota_ms"]:
            ms = np.array(tilasto["syota_ms"])
            kesto = max(len(ms) / max(fps, 1.0), 1e-9)
            print(f"ajankaytto: analyysi {tilasto['syota_s']:.0f} s (ruudulta keskim. {ms.mean():.1f} ms, p95 {np.percentile(ms, 95):.1f}, max {ms.max():.0f}; reaaliaikaan riittaa alle {1e3 / max(fps, 1.0):.0f} ms), "
                  f"lahteen luku {tilasto['luku_s']:.0f} s (pisin {tilasto['luku_max']:.2f} s, yli 0,25 s: {tilasto['pysahdyksia']} kertaa); analyysin osuus videon kestosta {100.0 * tilasto['syota_s'] / kesto:.0f} %", flush=True)
            osat = sorted(an.aika.items(), key=lambda kv_: -kv_[1])
            print("  analyysin vaiheet: " + ", ".join(f"{k} {1e3 * v / len(ms):.1f} ms/ruutu" for k, v in osat), flush=True)
        if not a.hiljaa and kiri_tila and tilasto["analysoitu"] > 0 and tilasto["ajo_s"] > 0:
            print(f"kiri (esilatauspuskuri >= {kiri_raja_s:.0f} s): kuva eteni etuajassa {tilasto['kiri_ruutuja']} / {tilasto['piirretty']} ruudulla ({100.0 * tilasto['kiri_ruutuja'] / max(tilasto['piirretty'], 1):.0f} %), muuten 1x; "
                  f"analyysi eteni {tilasto['video_s'] / tilasto['ajo_s']:.2f}x reaaliajasta (videota {tilasto['video_s']:.0f} s, ajoa {tilasto['ajo_s']:.0f} s)"
                  + (f"; kuva odotti lopullisia viivoja {tilasto['kiri_odotus_s']:.1f} s" if tilasto["kiri_odotus_s"] > 0.05 else ""), flush=True)
        if not a.hiljaa and (pitka_s > 0 or tilasto["viive_max"] > 0.5):
            print(f"viive: suurin myohastyminen {tilasto['viive_max']:.1f} s" + (f"; esilatauspuskuri enintaan {tilasto['puskuri_max']:.0f} s (tavoite {pitka_s:.0f} s)" if pitka_s > 0 else ""), flush=True)
            if tilasto["viive_max"] > 1.0:
                print("  koko ajalta: " + viive_erittely(tilasto, tilasto_otos(dict()), (kirjoittaja.jonossa(), kirjoittaja.jono_koko) if kirjoittaja is not None else None, None), flush=True)
        if not a.hiljaa:
            print(f"analysoitu {tilasto['analysoitu']} ruutua, ohitettu (myohassa) {tilasto['ohitettu']}, virheita {tilasto['virheita']}; piirretty {tilasto['piirretty']} ruutua"
                  + (f", ikkunassa {naytto.n_nayt}" if naytto is not None else "") + f", suurin viive piirrossa {tilasto['myohassa_max']:.1f} s", flush=True)


BGR_PUSKURI_MIN_GT = 14                              # --puskuri auto: raaka BGR-puskuri (ei muunnoksia) kun fyysista muistia on vahintaan tama (Gt); nimellisesti 16 Gt:n kone ilmoittaa 14,9-15,9 Gt (laitteisto varaa osan)


def fyysinen_muisti_gt():
    """Koneen fyysinen muisti (Gt) tai None, jos sita ei saada selville (Windows: GlobalMemoryStatusEx; Linux: sysconf)."""
    try:
        if os.name == "nt":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong), ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = MEMORYSTATUSEX()
            m.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
                return m.ullTotalPhys / 2 ** 30
            return None
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30
    except Exception:
        return None


def valitse_puskuri(pyydetty, muisti_gt):
    """Viivepuskurin muoto: pyydetty (yuv / jpeg / bgr) sellaisenaan; auto = bgr, kun muistia on vahintaan BGR_PUSKURI_MIN_GT Gt (muunnoksia ei tarvita: ~3,5 ms / ruutu vahemman
    prosessoria, puskuri ~1,9 GB 10 s:n viiveella), muuten yuv (myos kun muistia ei saada selville)."""
    if pyydetty != "auto":
        return pyydetty
    return "bgr" if muisti_gt is not None and muisti_gt >= BGR_PUSKURI_MIN_GT else "yuv"


def pakkaa_ruutu(fr, muoto, jpeg_laatu=92):
    """Viivepuskurin ruutu: (muoto, data). bgr = ruutu sellaisenaan (ei muunnoksia; kopio tehdaan vasta puretaessa), jpeg, yuv = YUV 4:2:0."""
    if muoto == "bgr":
        return "bgr", fr
    if muoto == "jpeg":
        return "jpeg", cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_laatu)])[1]
    return "yuv", cv2.cvtColor(fr, cv2.COLOR_BGR2YUV_I420)


def pura_ruutu(laji, data):
    """pakkaa_ruutu:n vastakohta: BGR-ruutu. bgr: kopio (~1 ms purkusaikeessa), koska piirto muokkaa ruutua paikallaan, eika se saa koskaan muuttaa ruutua, jonka analyysi voi viela lukea
    (hyvin pieni --viive tai myohassa oleva analyysi live-virrassa)."""
    if laji == "bgr":
        return data.copy()
    if laji == "jpeg":
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    return cv2.cvtColor(data, cv2.COLOR_YUV2BGR_I420)


KIRI_JONO_MARGINAALI_S = 2.0                          # kiri: analyysi saa olla kuvaa edella vahintaan viive + tama (s) ennen kuin se jaa odottamaan kuvaa
KIRI_VIIVAT_ODOTUS_S = 3.0                           # kiri: kuva odottaa lopullista viivaestimaattia enintaan taman ajan (s)


class KuvanAikataulu:
    """VOD-virran (YouTube-nauhoitus; kiri) kuvan aikataulu. Analyysi etenee ilman tahdistusta niin nopeasti kuin ehtii (ja dataa on), mutta kuva etenee reaaliajassa (1x, tasaisesti ilman hyppyja)
    paitsi kun esilatauspuskurissa on vahintaan raja_s sekuntia videota (olemme jaljessa tai kyse on nauhoituksesta, joka halutaan ajaa nopeasti lapi) tai lahde on loppunut: silloin kuva saa kiria
    (ei aikataulua; tahdin maaraa analyysi, piirto ja tallennus). Kirin paattyessa 1x-aika jatkuu siita (ankkuri siirtyy), ja myohassa oleva kuva naytetaan heti kuten ennenkin.
    Ankkuri: ensimmaisen kuvan oma aika (jonoon tulo + viive)."""

    def __init__(self, fps, raja_s, hystereesi_s=1.0):
        self.fps, self.raja_s, self.hystereesi_s = float(fps), float(raja_s), float(hystereesi_s)
        self.f_eka = self.ankkuri = None
        self.kiri = False

    def kiri_sallittu(self, puskuri_s, loppu=False):
        """Saako kuva kiria: puskuri >= raja (kirin ollessa jo kaynnissa hystereesin verran alle, ettei tila vaihtele rajalla) tai lahde on loppunut."""
        return self.raja_s > 0 and (bool(loppu) or puskuri_s >= self.raja_s - (self.hystereesi_s if self.kiri else 0.0))

    def aika(self, f, nyt, puskuri_s, ensimmainen_aika, loppu=False):
        """(erapaiva, kiri): ruudun f oma aika. Kirissa nyt (heti); muuten ankkurista (1x). ensimmainen_aika: ensimmaisen kuvan oma aika (asettaa ankkurin)."""
        if self.ankkuri is None:
            self.f_eka, self.ankkuri = f, ensimmainen_aika
        if self.kiri_sallittu(puskuri_s, loppu):
            self.kiri = True
            return nyt, True
        if self.kiri:                                      # kiri paattyi: 1x jatkuu tasta
            self.ankkuri, self.kiri = nyt - (f - self.f_eka) / self.fps, False
        return self.ankkuri + (f - self.f_eka) / self.fps, False


def kirjoitusjonon_koko(muisti_gt):
    """Tallennusjonon pituus (ruutua): 30 (1 s); kun muistia on vahintaan BGR_PUSKURI_MIN_GT Gt, 90 (3 s, ~0,56 Gt): kooderin hetkellinen hidastelu (ruuhka heiton aikana) ei silloin pysayta piirtoa
    eika viivasta kuvaa. Pysyva ylivoima ei tasoitu jonolla -- silloin viive_varoitus kertoo, ettei kooderi ehdi."""
    return 90 if muisti_gt is not None and muisti_gt >= BGR_PUSKURI_MIN_GT else 30


def puskuri_rajat(viive, fps, muisti_gt):
    """(bgr_raja, yuv_raja): viivepuskurin jonon pituudet (ruutua), joiden jalkeen uudet ruudut pakataan kevyemmin muistille (bgr -> yuv, yuv -> jpeg). Pakkaus maksaa prosessoria (yuv ~6 ms,
    jpeg ~21 ms / ruutu) juuri silloin, kun kone on jo ruuhkassa ja kuva myohassa, ja kuormittaisi sita lisaa (myohastyminen -> jpeg -> lisakuorma -> myohastyminen), joten muistia kaytetaan
    ensin: rajat ovat vahintaan viive + 2 s / viive + 4 s ja kasvavat koneen muistin mukaan (bgr-jonolle 20 %, yuv-jonolle lisaksi 10 % muistista; ruutu 6,2 / 3,1 Mt)."""
    bgr, yuv = int((viive + 2.0) * fps), int((viive + 4.0) * fps)
    if muisti_gt is not None and muisti_gt > 0:
        bgr = max(bgr, int(0.20 * muisti_gt * 1024 / 6.2))
        yuv = max(yuv, bgr + int(0.10 * muisti_gt * 1024 / 3.1))
    return bgr, yuv


def puskurin_muoto(valittu, jonon_pituus, bgr_raja, yuv_raja):
    """Seuraavan ruudun puskurimuoto: valittu (--puskuri), mutta jonon kasvaessa (kone ei jaksa) raskaampi muisti vaihtuu kevyempaan: bgr -> yuv (jono > bgr_raja), yuv -> jpeg (jono > yuv_raja)."""
    if valittu == "bgr" and jonon_pituus <= bgr_raja:
        return "bgr"
    if valittu == "jpeg" or jonon_pituus > yuv_raja:
        return "jpeg"
    return "yuv"


TILARIVI_VALI_S = 20.0                               # viive-tilarivi (stderr): enintaan taman valein, kun kuva on yli 1 s myohassa


class KoneenKuorma:
    """Koko koneen ja tamän ohjelman pääprosessin prosessorinkaytto tilariveille: koneen kuorma (Windows: GetSystemTimes, Linux: /proc/stat) ja paaprosessin ytimet (process_time; ei sisalla
    ffmpegia eika tyoprosesseja). Jos kone on lahes taynna (>= 90 %), kaikki saikeet ja prosessit hidastuvat (analyysi, piirto, tallennus); jos ei, odotus on muualla (esim. GIL, kooderi, verkko)."""

    def __init__(self):
        self._kone = self._lue_kone()
        self._t, self._p = time.time(), time.process_time()

    @staticmethod
    def _lue_kone():
        """(kuormitettu, yhteensa) kertyvat ajat tai None, jos niita ei saada."""
        try:
            if os.name == "nt":
                import ctypes
                from ctypes import wintypes
                tyhja, ydin, kayttaja = wintypes.FILETIME(), wintypes.FILETIME(), wintypes.FILETIME()
                if not ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(tyhja), ctypes.byref(ydin), ctypes.byref(kayttaja)):
                    return None
                aika = lambda x: (x.dwHighDateTime << 32) | x.dwLowDateTime          # noqa: E731
                yht = aika(ydin) + aika(kayttaja)                                     # (ydinaika sisaltaa tyhjakaynnin)
                return yht - aika(tyhja), yht
            with open("/proc/stat") as fh:
                v = [int(x) for x in fh.readline().split()[1:9]]
            tyhja = v[3] + v[4]                                                       # idle + iowait
            return sum(v) - tyhja, sum(v)
        except Exception:
            return None

    def erittely(self):
        """Teksti edellisen kutsun (tai luonnin) jalkeen: koneen kuorma ja pääprosessin ytimet; nollaa vertailukohdan."""
        kone, t, p = self._lue_kone(), time.time(), time.process_time()
        osat = []
        if kone is not None and self._kone is not None and kone[1] > self._kone[1]:
            osat.append(f"kone {100.0 * (kone[0] - self._kone[0]) / (kone[1] - self._kone[1]):.0f} % kaytossa ({os.cpu_count()} loogista ydinta)")
        if t > self._t:
            osat.append(f"paaprosessi {(p - self._p) / (t - self._t):.1f} ydinta")
        self._kone, self._t, self._p = kone, t, p
        return ", ".join(osat)


def tilasto_otos(tilasto):
    """Tilaston kertyvat luvut (s / kpl) erotuksen laskemista varten."""
    return {k: float(tilasto.get(k, 0.0)) for k in ("analyysia_odotettu_s", "kirjoitus_odotus_s", "purku_odotus_s", "piirto_s", "piirto_cpu_s", "luku_s", "syota_s", "syota_cpu_s", "analysoitu", "piirretty",
                                                    "kirj_s", "kirj_cpu_s", "kirj_n")}


def viive_erittely(tilasto, edellinen, kirjoitusjono=None, puskuri_s=None):
    """Mihin aika on mennyt edellisen otoksen jalkeen (tilasto_otos; kirjoitusjono = (pituus, enimmaiskoko) tai None): analyysin odotus, tallennusjonon odotus, pakkauksen purun odotus, lahteen luvun odotus ja piirron / analyysin ruutuaika
    (seinakello ja saikeen oma prosessoriaika: jos seinakello on paljon suurempi kuin CPU, saie on odottanut prosessoria tai lukkoa), tallennussaikeen syotto kooderille (muunnos + putki).
    Piirtosaie odottaa analyysia (toistettava ajo) tai tallennusjonoa (kooderi ei ehdi); analyysisaie odottaa lukua (verkko / dekoodaus)."""
    nyt = tilasto_otos(tilasto)
    d = {k: nyt[k] - edellinen.get(k, 0.0) for k in nyt}
    piirretty, analysoitu = max(d["piirretty"], 1.0), max(d["analysoitu"], 1.0)
    teksti = (f"analyysia odotettu {d['analyysia_odotettu_s']:.1f} s, tallennusjonoa odotettu {d['kirjoitus_odotus_s']:.1f} s" + (f" (jono {kirjoitusjono[0]}/{kirjoitusjono[1]})" if kirjoitusjono is not None else "")
              + f", pakkauksen purkua odotettu {d['purku_odotus_s']:.1f} s, lahteen lukua odotettu {d['luku_s']:.1f} s" + (f" (esilatauspuskuri {puskuri_s:.0f} s)" if puskuri_s is not None else "")
              + f"; ruudun aika: piirto {1e3 * d['piirto_s'] / piirretty:.1f} ms (CPU {1e3 * d['piirto_cpu_s'] / piirretty:.1f}), analyysi {1e3 * d['syota_s'] / analysoitu:.1f} ms (CPU {1e3 * d['syota_cpu_s'] / analysoitu:.1f})")
    if d["kirj_n"] > 0:
        teksti += f"; tallennussaie: syotto kooderille {1e3 * d['kirj_s'] / d['kirj_n']:.1f} ms/ruutu (CPU {1e3 * d['kirj_cpu_s'] / d['kirj_n']:.1f})"
    return teksti


def viive_tilarivi(t_s, viive, tilasto, edellinen, kirjoitusjono=None, puskuri_s=None, kuorma=""):
    """Yksi tilarivi (stderr), kun kuva on yli 1 s myohassa: ajon aika, myohastyminen ja viime tilarivin jalkeinen erittely (viive_erittely) sekä koneen kuorma (KoneenKuorma.erittely)."""
    return f"[{t_s:6.0f} s] kuva {viive:.1f} s myohassa; viime ajalta: " + viive_erittely(tilasto, edellinen, kirjoitusjono, puskuri_s) + (f"; {kuorma}" if kuorma else "")


def viive_varoitus(myohassa, tilasto, an, fps):
    """Varoitusteksti, kun kuva on yli 3 s myohassa: syy ja keinot. Toistettavassa ajossa kuva odottaa analyysia (ks. Analyysi.aseta_toistettava): jos suurin osa myohastymisesta on odotusta,
    analyysi ei ehdi mukaan -- ruudun analyysin aika (tilasto syota_ms) ja tarkan sovituksen toteutus (C++ vai Python) kertovat miksi. Muuten syy on piirtosaikeen odotuksissa: tallennusjono
    (ffmpeg ei ehdi), lahteen luku (verkko / dekoodaus) tai piirto itse."""
    ms = tilasto["syota_ms"][-300:]
    keski = sum(ms) / len(ms) if ms else 0.0
    if tilasto["analyysia_odotettu_s"] > 0.5 * myohassa:
        sovitin = next(iter(an.M.values())).sovitin
        vihje = ""
        if "C++" not in sovitin:
            vihje = f" Tarkka sovitus ajetaan Pythonilla, joka on noin 6 kertaa hitaampi kuin C++ (netti_fit): {ns.netti_fit_selitys() or 'C++-moduuli ei ole kaytossa'}."
        return (f"Huom: analyysi ei ehdi mukaan reaaliajassa: kuva on {myohassa:.1f} s myohassa, koska se odottaa analyysia (ruudulta keskim. {keski:.0f} ms, reaaliaikaan riittaa alle "
                f"{1e3 / max(fps, 1.0):.0f} ms; tarkka sovitus: {sovitin}, B-tyoprosesseja --saikeet {an.saikeet}).{vihje} Tulos ei muutu, mutta kuvan viive kasvaa. "
                "Kokeile suurempaa --saikeet-arvoa (enintaan ytimet - 1), sulje muita raskaita ohjelmia tai aja --ulos ilman ikkunaa (--ei-ikkunaa).")
    if tilasto.get("kirjoitus_odotus_s", 0.0) > 0.5 * myohassa:
        nimi = tilasto.get("tallennus") or "tallennus"
        n = tilasto.get("kirj_n", 0.0)
        syotto = f" Tallennussaie syottaa kooderille keskimaarin {1e3 * tilasto.get('kirj_s', 0.0) / n:.0f} ms/ruutu (prosessoria {1e3 * tilasto.get('kirj_cpu_s', 0.0) / n:.0f} ms)." if n > 0 else ""
        if "h264_" in nimi:                                  # laitteistokooderi (NVENC / QSV / AMF): koodaus on jo naytonohjaimella, jaljella on kuvien valmistelu ja siirto ffmpegille
            keinot = ("Laitteistokooderi on jo kaytossa, joten hidastaa kuvien syotto kooderille (varimuunnos ja siirto ffmpegille) tai koneen ruuhka: sulje muita raskaita ohjelmia, "
                      "kokeile pienempaa --saikeet-arvoa tai aja ilman ikkunaa (--ei-ikkunaa).")
        else:
            keinot = "Kokeile --x264-preset ultrafast, --koodekki gpu (laitteistokoodaus: NVENC / QSV / AMF), sulje muita raskaita ohjelmia tai aja ilman --ulos."
        return (f"Huom: tallennus ei ehdi mukaan reaaliajassa: kuva on {myohassa:.1f} s myohassa, koska piirtosaie on odottanut tallennusjonoa yhteensa {tilasto['kirjoitus_odotus_s']:.1f} s "
                f"({nimi} ei saa 30 ruutua sekunnissa; analyysi ehtii: ruudulta keskim. {keski:.0f} ms).{syotto} Tallenteesta ei puutu ruutuja, mutta kuvan viive kasvaa. {keinot}")
    if tilasto.get("pysahdyksia", 0) > 0 and tilasto.get("luku_s", 0.0) > 0.5 * myohassa:
        return (f"Huom: videon luku takkuaa: kuva on {myohassa:.1f} s myohassa, koska lahde viipyi yhteensa {tilasto['luku_s']:.1f} s (pisin {tilasto.get('luku_max', 0.0):.1f} s, "
                f"{tilasto['pysahdyksia']} kertaa yli 0,25 s). Verkko tai dekoodaus ei pysy mukana; analyysi ehtii (ruudulta keskim. {keski:.0f} ms). "
                "Kokeile matalampaa resoluutiota (--yt-formaatti 'bestvideo[height<=1080][vcodec^=avc1]'), tarkista verkko tai lataa video tiedostoksi (yt-dlp) ja aja --lahde.")
    return (f"Huom: piirto / ikkuna on {myohassa:.1f} s jaljessa reaaliajasta (piirto {1e3 * tilasto.get('piirto_s', 0.0) / max(tilasto.get('piirretty', 0), 1):.0f} ms/ruutu, pakkauksen purkua odotettu "
            f"{tilasto.get('purku_odotus_s', 0.0):.1f} s; analyysi ehtii: ruudulta keskim. {keski:.0f} ms); ikkuna ohittaa ruutuja, tallenteesta ei puutu ruutuja, mutta kuvan viive kasvaa. "
            "Kokeile --saikeet pienemmaksi tai --ulos ilman ikkunaa (--ei-ikkunaa).")


def kalibrointitarkistus(video, kansio):
    """Varoitus, jos kansion kalibrointi ja moodikuva eivat sovi videon kamera-asentoon (toinen peli / kentta / pystytys): seuranta toimisi huonosti ja paikat olisivat vinossa (tools/tarkista_kalibrointi.py)."""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools"))
        import tarkista_kalibrointi as tk
        tulos = tk.siirtymat(video, kansio)
        sopimattomat = [(p, tk.arvioi(tulos[p])[1]) for p in ("vasen", "oikea") if tk.arvioi(tulos[p])[0] is False]
    except Exception as e:                                     # tarkistus ei saa kaataa ajoa
        print(f"(kalibrointitarkistus ei onnistunut: {e!r})", flush=True)
        return
    if sopimattomat:
        print(f"VAROITUS: kalibrointi (kansio {kansio}) ei sovi tahan videoon, tulokset ovat epatarkkoja: " + "; ".join(f"{p}: {t}" for p, t in sopimattomat)
              + f"\n  Kalibroi video: python tools/uusi_kalibrointi.py {video} --ulos <uusi kansio>  ja aja --kansio <uusi kansio>  (--ei-kalibrointitarkistusta ohittaa tarkistuksen)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--youtube", nargs="?", const=YOUTUBE_OLETUS, help=f"YouTube-tunnus tai -osoite (oletus {YOUTUBE_OLETUS})")
    ap.add_argument("--yt-formaatti", default=YT_FORMAATTI, help="yt_dlp-formaattivalitsin (oletus: oma testikoodi)")
    ap.add_argument("--lahde", help="videotiedosto")
    ap.add_argument("--url", help="suora virran osoite (http / https / rtsp / m3u8), luetaan kuten YouTube-virta")
    ap.add_argument("--kansio", default=None, help="kalibroinnit ja taustakuvat (kalibrointi_*.json, moodikuva_*[_puhdas].png, kivimalli_*.json, kierto_muoto_*.json). Oletus: tiedostolahteelle (ja YouTube / --url -virralle, ei live) sopiva kansio valitaan automaattisesti "
                    "(tulokset, tulokset_nettivideo2, ...) tai video kalibroidaan kansioon tulokset_auto_<nimi> (kalibrointi_valinta.py); muulle lahteelle tulokset")
    ap.add_argument("--viive", type=float, default=10.0, help="tulosteen viive analyysiin nahden (s); oletus 10: 97 %%:lla heitoista viiva- ja irroitusestimaatti on jo lopullinen (muutos alle 10 cm) kun viiva ilmestyy (5 s: 12 %%)")
    ap.add_argument("--jalki", type=float, default=10.0, help="pistejoukko nakyvissa taman ajan heiton jalkeen (s)")
    ap.add_argument("--viivat", type=float, default=60.0, help="linjat nakyvissa taman ajan heiton alusta, tai seuraavaan heittoon (s)")
    ap.add_argument("--ulos", nargs="?", const="auto", help="tallenna tuloste tiedostoon (mp4); ilman nimea: kivenseuranta_<paivays>_<aika>.mp4 nykyiseen kansioon")
    ap.add_argument("--koodekki", default="auto", help="tallenteen koodekki: auto (ffmpeg: laitteistokooderi jos toimii, muuten H.264 prosessorilla; ilman ffmpegia OpenCV mp4v), cpu (libx264), gpu "
                                                       "(laitteistokooderi) tai OpenCV:n fourcc (mp4v, avc1, MJPG)")
    ap.add_argument("--ffmpeg", default=None, help="ffmpeg-ohjelman polku (oletus: PATH tai imageio-ffmpeg)")
    ap.add_argument("--x264-preset", default="superfast", help="H.264-koodauksen nopeus (ultrafast = kevein prosessorille, isoin tiedosto; superfast; veryfast)")
    ap.add_argument("--nayttotila", choices=("auto", "ikkuna", "kokoruutu"), default="auto",
                    help="ikkuna: kuva taydella 1920 x 1080 resoluutiolla; auto = kokoruutu jos naytto ei ole sita suurempi (1:1); 'f' vaihtaa kokoruudun")
    ap.add_argument("--ikkuna-leveys", type=int, default=0, help="ikkunan leveys pikseleina (0 = taysi resoluutio 1920; pienempi skaalaa kuvaa)")
    ap.add_argument("--csv", nargs="?", const="auto", default="auto",
                    help="heittojen tiedot CSV:hen (oletus: heitot_<paivays>_<aika>.csv nykyiseen kansioon; kivikohtaiset vakiot viereen <nimi>_kivet.csv): heiton viivaestimaatti, kierre, "
                         "kaukopaan pysahtyminen, kiven vakiot (A, kappa), miettimisaika ja joukkueen aika")
    ap.add_argument("--ei-csv", action="store_true", help="ei CSV-tallennusta")
    ap.add_argument("--radat", help="tallenna heittojen radat ja mittaukset (pickle) jalkianalyysia varten, esim. kiven vakioiden uudelleensovitus ilman koko videon analyysia (kehitys)")
    ap.add_argument("--ei-kelloa", action="store_true", help="ei joukkueiden kelloa (miettimisaika) kuvassa")
    ap.add_argument("--kello-paikka", help="kellon laatikon vasen ylakulma kuvassa X,Y (pikselia); oletus: kuvan keskella mainosgrafiikan (Hyvinkaan Liikenne) alla")
    ap.add_argument("--ei-ikkunaa", action="store_true", help="ei ikkunaa (palvelin / testaus)")
    ap.add_argument("--ei-kierretta", action="store_true", help="ei kierteen eika hog-hog-ajan mittausta")
    ap.add_argument("--nopea", action="store_true", help="simulaatio: tiedosto ajetaan niin nopeasti kuin ehtii (tulos ei riipu ajoituksesta)")
    ap.add_argument("--alku", type=float, default=0.0, help="aloitus tiedostossa (s)")
    ap.add_argument("--loppu", type=float, default=None, help="lopetus tiedostossa (s)")
    ap.add_argument("--saikeet", type=int, default=max(1, min(3, (os.cpu_count() or 2) - 1)), help="tarkan sovituksen tyoprosessit (0 = vain kahvan vari: epatarkempi)")
    ap.add_argument("--fps-lahde", type=float, default=None, help="erillisten ruutujen nimellistaajuus (25 tassa videossa; oletus: paatellaan kopioruutujen osuudesta)")
    ap.add_argument("--puskuri", choices=("auto", "yuv", "jpeg", "bgr"), default="auto",
                    help="viivepuskurin muoto: auto (oletus) = bgr, kun koneessa on muistia vahintaan %d Gt, muuten yuv; yuv = raaka YUV 4:2:0 (3,1 MB / ruutu, ~940 MB 10 s:n viiveella, kevyt prosessorille); "
                         "jpeg (0,3 MB / ruutu, ~21 ms / ruutu prosessoria); bgr = ruutu sellaisenaan (6,2 MB / ruutu, ~1,9 GB 10 s:n viiveella; ei muunnoksia: ~3,5 ms / ruutu vahemman prosessoria)" % BGR_PUSKURI_MIN_GT)
    ap.add_argument("--jpeg", type=int, default=92, help="JPEG-laatu (--puskuri jpeg ja yuv-puskurin ylivuoto)")
    ap.add_argument("--tapahtumat", action="store_true", help="tulosta seurannan kaikki tapahtumat (viritys, lahto, vahvistus, hylkays, ei kivea, valmis): nayttaa myos heittoyritykset, jotka jaavat tunnistamatta")
    ap.add_argument("--hiljaa", action="store_true")
    ap.add_argument("--ei-kalibrointitarkistusta", action="store_true", help="ei tarkisteta (tiedostolahde), sopiiko --kansio:n kalibrointi ja moodikuva videon kamera-asentoon (tools/tarkista_kalibrointi.py)")
    ap.add_argument("--debug-viive", action="store_true", help="tulosta hitaat analyysiruudut (selvittaa miksi ruutuja ohitetaan)")
    ap.add_argument("--ohita-myohassa", action="store_true", help="reaaliaika: ohita analyysista ruutuja, kun kone ei jaksa (vanha tapa; tulokset riippuvat koneen nopeudesta). Oletus tiedostolla ja YouTube-toistolla: "
                    "kaikki ruudut analysoidaan ja tulos on toistettava (sama kuin --nopea), kuva odottaa analyysia")
    ap.add_argument("--aani", default="auto", help="aani tallenteeseen (--ulos): auto = YouTube: oma aanivirta (yt_dlp bestaudio), ladataan taustalla ajon aikana ja liitetaan lopuksi, tiedosto: tiedoston oma aani jos on, --url: sama osoite; ei = ei aanta; POLKU / URL = oma aanilahde "
                    "(sama aikajana kuin video). Aani on lahdevideon aikajanalla, joten se pysyy samassa kohdassa kuin viivastetty kuva; ikkuna on aaneton; live-virralla ei aanta")
    ap.add_argument("--lataus", type=float, default=30.0, help="YouTube / --url: video ladataan jatkuvasti puskuriin enintaan taman verran (s) analyysin edelle, jolloin raskaiden hetkien (heitto) myohastyminen kirii kiinni kevyina hetkina; "
                    "puskurin ruudut ovat pakattuja (3,1 Mt / ruutu: 30 s = 2,8 Gt muistia), 0 = pois. Ei vaikuta tiedostolahteeseen")
    ap.add_argument("--kiri", type=float, default=10.0, help="YouTube / --url (VOD): analyysi etenee ilman reaaliaikaan sidontaa niin nopeasti kuin ehtii (etuajassa), mutta kuva etenee reaaliajassa (1x, tasaisesti) "
                    "paitsi kun esilatauspuskurissa on vahintaan taman verran (s) videota (olemme jaljessa tai kyse on nauhoituksesta): silloin kuva (ikkuna ja tallenne) saa myos kiria niin nopeasti kuin ehtii; "
                    "tallenteen aikajana ja aani pysyvat oikeina. Oletus 10 s, 0 = ei kiria (aina 1x). Live-virrassa ei vaikutusta")
    ap.add_argument("--etumatka", type=float, default=10.0, help="kiri: analyysi saa olla kuvaa edella viiveen lisaksi enintaan taman verran (s); sen jalkeen se odottaa kuvaa (viivepuskurin muisti). Oletus 10 s")
    ap.add_argument("--max-myohassa", type=float, default=None, help="reaaliaika: ruutu ohitetaan (nakyy tulosteessa, mutta sita ei analysoida), jos analyysi on taman verran (s) myohassa. Oletus 0,8 s; kun --lataus on paalla "
                    "min(6, viive - 3) s: myohastyminen saa kasvaa ja kirii kiinni puskurin avulla (kuvan 'Viive'-rivi nayttaa sen)")
    ap.add_argument("--esilataus", type=int, default=24, help="reaaliaika: ruutuja luetaan erillisessa saikeessa enintaan taman verran analyysin edella (0 = pois; yksi ruutu on 6 Mt)")
    ap.add_argument("--ei-tahdistusta", action="store_true", help="(testaus) reaaliaikarakenne mutta tiedostoa luetaan niin nopeasti kuin analyysi ehtii (ei reaaliaikatahdistusta)")
    ap.add_argument("--kello-loki", help="(testaus, vain --nopea) kirjoita kuvassa nakyvan kellon tila kerran sekunnissa CSV:hen: aika, kayva joukkue, kuluva aika, viimeisin heitto")
    a = ap.parse_args()
    if a.kansio is None:
        a.kansio = "tulokset"
        if a.lahde and not a.ei_kalibrointitarkistusta:
            import kalibrointi_valinta as kval
            a.kansio, selitys = kval.valitse(a.lahde)
            if not a.hiljaa:
                print(f"Kalibrointi: {selitys}", flush=True)
            a.ei_kalibrointitarkistusta = True         # valinta on jo tarkistus
        elif (a.youtube or a.url) and not a.ei_kalibrointitarkistusta:
            # virta (ei live): mediaanikuva haetaan virrasta hakemalla ruutuja videon yli (YT_VALINTA_RUUTUJA kpl), ja sopiva kansio valitaan kuten tiedostolle (ei kalibroida virtaa)
            import kalibrointi_valinta as kval
            try:
                osoite, live = youtube_virta(a.youtube, a.yt_formaatti) if a.youtube else (a.url, False)
                if live:
                    print("Huom: live-virralle kalibrointikansiota ei voi valita automaattisesti: kaytetaan kansiota tulokset (anna --kansio)", flush=True)
                else:
                    print(f"Valitaan kalibrointikansio virrasta ({YT_VALINTA_RUUTUJA} ruutua, voi kestaa minuutin) ...", flush=True)
                    a.kansio, selitys = kval.valitse(osoite, luo=False, n=YT_VALINTA_RUUTUJA)
                    print(f"Kalibrointi: {selitys}", flush=True)
                    if "ei sovi" in selitys:
                        print("  Anna oikea kansio: --kansio tulokset_nettivideo3 (jne.) tai lataa video ja aja --lahde (silloin video kalibroidaan tarvittaessa)", flush=True)
            except SystemExit:
                raise
            except Exception as e:                      # noqa: BLE001
                print(f"Huom: kalibrointikansion automaattinen valinta virrasta epaonnistui ({e!r}): kaytetaan kansiota tulokset (anna --kansio)", flush=True)
            a.ei_kalibrointitarkistusta = True
    if not os.path.isdir(a.kansio):                    # ajettu muualta kuin projektin kansiosta
        oma = os.path.join(os.path.dirname(os.path.abspath(__file__)), a.kansio)
        if os.path.isdir(oma):
            a.kansio = oma
    leima = time.strftime("%Y%m%d_%H%M%S")
    if a.ulos == "auto":
        a.ulos = f"kivenseuranta_{leima}.mp4"
    if a.ei_csv:
        a.csv = None
    elif a.csv == "auto":
        a.csv = f"heitot_{leima}.csv"
    if not (a.youtube or a.lahde or a.url):
        ap.error("anna --youtube [TUNNUS], --lahde TIEDOSTO tai --url OSOITE")
    sim = bool(a.nopea and a.lahde)
    if a.lahde and not a.ei_kalibrointitarkistusta and not a.hiljaa:
        kalibrointitarkistus(a.lahde, a.kansio)
    # laskentaprosessit kayntiin ENNEN lahteen avaamista: spawn-prosessien kaynnistys kestaa 5-15 s eika virta saa odottaa
    an = ns.Analyysi(a.kansio, fps=30.0, saikeet=a.saikeet, tarkka=a.saikeet > 0, synkroninen=sim, fps_lahde=a.fps_lahde, kierre=not a.ei_kierretta)
    if not a.hiljaa:
        print(f"netti_live: {versio_teksti()}", flush=True)
        print("Kaynnistetaan laskentaprosessit ...", flush=True)
    t_lampea = time.time()
    n_prosessia = an.lampea()
    if not a.hiljaa:
        sovitin = next(iter(an.M.values())).sovitin
        print(f"  {n_prosessia} tyoprosessia valmiina ({time.time() - t_lampea:.0f} s); tarkka sovitus: {sovitin}"
              + ("" if "C++" in sovitin else f" ({ns.netti_fit_selitys()})"), flush=True)
    lahde = Lahde(tiedosto=a.lahde, youtube=a.youtube, alku_s=a.alku, loppu_s=a.loppu, yt_formaatti=a.yt_formaatti, url=a.url)
    an.fps = lahde.fps
    if not sim and not lahde.live and not a.ohita_myohassa:
        an.aseta_toistettava(True)
    elif not sim:
        an.matalaksi_b_tyoprosessit()                    # tulos saa riippua ajoituksesta: B-tyo vaistaa analyysi-, piirto- ja ikkunasaikeita
    geom = ng.Geometria(a.kansio)
    kello_paikka = tuple(int(x) for x in a.kello_paikka.split(",")) if a.kello_paikka else None
    piirtaja = npi.Piirtaja(geom, fps=lahde.fps, jalki_s=a.jalki, viivat_s=a.viivat, viive_s=a.viive, kello=not a.ei_kelloa, kello_paikka=kello_paikka)
    kirjoittaja = Kirjoittaja(a.ulos, lahde.fps, a.koodekki, a.ffmpeg, a.x264_preset, valitse_aani(a, lahde) if a.ulos else None, jono_koko=kirjoitusjonon_koko(fyysinen_muisti_gt())) if a.ulos else None
    naytto = None if a.ei_ikkunaa else Naytto(a.nayttotila, a.ikkuna_leveys)
    if not a.hiljaa:
        print(f"Kuva tulee {a.viive:.0f} s viiveella. " + ("Ikkuna: 'q' tai ESC (tai ikkunan sulkeminen) lopettaa, 'f' vaihtaa kokoruudun. " if naytto else "")
              + (f"Tallennus: {a.ulos} ({kirjoittaja.nimi})" if kirjoittaja else "Ei tallennusta (lisaa --ulos)."), flush=True)
        print(f"Heittojen tiedot: {a.csv} (kivikohtaiset vakiot: {os.path.splitext(a.csv)[0]}_kivet.csv)" if a.csv else "Ei CSV-tallennusta (--ei-csv).", flush=True)
    kirjaaja = Kirjaaja(a.csv, a.hiljaa, lahde.fps, a.tapahtumat, a.radat)
    try:
        if sim:
            aja_sim(a, lahde, an, piirtaja, naytto, kirjoittaja, kirjaaja)
        else:
            aja_reaaliaika(a, lahde, an, piirtaja, naytto, kirjoittaja, kirjaaja)
    finally:
        an.sulje()
        if kirjoittaja is not None:
            kirjoittaja.sulje()
        if naytto is not None:
            naytto.sulje()
        if a.ulos and not a.hiljaa:
            print(f"kirjoitettu {a.ulos} ({kirjoittaja.n} ruutua)")


if __name__ == "__main__":
    main()
