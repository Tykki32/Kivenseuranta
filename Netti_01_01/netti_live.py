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
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback

import cv2
import numpy as np

import netti_geom as ng
import netti_piirto as npi
import netti_seuranta as ns

RUUTU_W, RUUTU_H = 1920, 1080
YOUTUBE_OLETUS = "hnTTLqHr5x8"


# ---------------------------------------------------------------------------------------------------------------------
# LAHTEET
# ---------------------------------------------------------------------------------------------------------------------
YT_FORMAATTI = "bestvideo[protocol=https][vcodec^=avc1]/bestvideo[protocol=https]/bestvideo"       # oma testikoodi


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
    if polku:
        return polku if shutil.which(polku) or os.path.exists(polku) else None
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


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
        cmd = [exe, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{RUUTU_W}x{RUUTU_H}", "-r", f"{fps:.4f}", "-i", "-",
               "-an", *koodaus, "-pix_fmt", "yuv420p", "-movflags", "+faststart", polku]
        lisa = {}
        if os.name == "nt":                               # koodaus ei saa viedda prosessoria analyysilta: alempi prioriteetti
            lisa = dict(creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) | 0x00004000)       # BELOW_NORMAL_PRIORITY_CLASS
        elif shutil.which("nice"):                        # (preexec_fn ei ole turvallinen monisaikeisessa prosessissa)
            cmd = ["nice", "-n", "5"] + cmd
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=self.loki, **lisa)
        self.nimi = f"ffmpeg ({nimi})"

    def kirjoita(self, kuva):
        self.p.stdin.write(kuva.data)

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

    def __init__(self, polku, fps=30.0, koodekki="auto", ffmpeg=None, preset="superfast"):
        self.polku = polku
        self.n = 0
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
            self.taustaosa = _CvTiedosto(polku, fps, "mp4v" if ffmpeg_tila else koodekki)
        self.nimi = self.taustaosa.nimi
        self.q = queue.Queue(maxsize=30)
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
                self.taustaosa.kirjoita(kuva)
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
               "nopeus_hog1_m_s", "suunta_hog1_deg", "irr_matka_ero_m", "irr_A_matka", "irr_sivu_ero_cm", "irr_tarkistus"]
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
            self._tulosta(f"          heitto {n} kierre: omega {k['omega_deg_s']:+.0f} deg/s ({suunta}, ruutuja {k['ruutuja']}, {'varma' if k.get('varma') else 'epavarma'}), "
                          f"hog-hog {s['T_hh']:.1f} s ({lahde}) -> {_pilkku(s['kierre'], 1, True)} kierrosta (- = vastapaivaan); "
                          f"irroitus (heitto - liuku) {_pilkku(s['irroitus_cm'], 0, True)} cm")
        return True

    @staticmethod
    def _allekirjoitus(an):
        """CSV:n sisaltoon vaikuttava tila: kirjoitetaan uusiksi vasta kun jokin heiton tulos (kierre, kaukopaa, vakiot, kello) on muuttunut."""
        return tuple((h.tunnus, h.tila, h.viivat_lopullinen, h.kierto_tila, h.kauko_kierto_tila, h.vakiot_tila, h.kauko is not None, h.hog2_f is not None,
                      None if h.kierre is None else round(h.kierre["N"], 2), h.f_alku, None if h.tviiva_f() is None else round(h.tviiva_f(), 1)) for h in list(an.heitot))

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
                            kierre=h.kierre, kauko=h.kauko, vakiot=h.vakiot, saapuminen=h.saapuminen,
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
            rivi += [k["tila"], N(k["f_pys"] / fps) if lepo else "", N(None if k.get("f_ulos") is None else k["f_ulos"] / fps), N(-k["X_pys"], 1) if lepo else "", N(k["Y_pys"], 1) if lepo else "",
                     N(float(np.hypot(k["X_pys"], k["Y_pys"])), 1) if lepo else "", N(k["v_hog_cm_s"], 1), N(k["tormays"]["f"] / fps) if k.get("tormays") else ""]
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
    jono = collections.deque()                       # (f, saapumisaika, pakkauksen tulevaisuus (laji, data))
    lopetettu = threading.Event()                    # lahde loppui
    pysayta = threading.Event()                      # kayttaja lopettaa tai virhe
    piirto_valmis = threading.Event()                # piirtosaie on antanut kaikki kuvat
    tilasto = dict(analysoitu=0, ohitettu=0, virheita=0, piirretty=0, myohassa_max=0.0)
    piirtoa_tarvitaan = naytto is not None or kirjoittaja is not None
    pakkaus = concurrent.futures.ThreadPoolExecutor(2, thread_name_prefix="puskuri-pakkaus")     # cv2-kutsut vapauttavat GIL:n: rinnakkain analyysin kanssa
    purku = concurrent.futures.ThreadPoolExecutor(2, thread_name_prefix="puskuri-purku")
    laatu = [cv2.IMWRITE_JPEG_QUALITY, a.jpeg]
    yuv_raja = int((a.viive + 4.0) * fps)            # yuv-puskurin muistin suoja: tamän jonopituuden jalkeen uudet ruudut pakataan JPEG:ksi (kone ei jaksa: jono kasvaa)

    def pakkaa(fr, jpeg):
        if jpeg:
            return "jpeg", cv2.imencode(".jpg", fr, laatu)[1]
        return "yuv", cv2.cvtColor(fr, cv2.COLOR_BGR2YUV_I420)

    def pura(tulevaisuus):
        laji, data = tulevaisuus.result()
        if laji == "jpeg":
            return cv2.imdecode(data, cv2.IMREAD_COLOR)
        return cv2.cvtColor(data, cv2.COLOR_YUV2BGR_I420)

    def analyysi_saie():
        t_alku = time.time()
        f_ensin = None
        t_edell = time.perf_counter()
        ohitus_kaynnissa = False
        try:
            for f, fr in lahde.ruudut():
                if pysayta.is_set():
                    break
                if a.debug_viive and f_ensin is not None and time.perf_counter() - t_edell > 0.2:
                    print(f"  [debug] ruutujen valissa {1e3 * (time.perf_counter() - t_edell):.0f} ms ennen ruutua {f} ({f / fps:.1f} s)", flush=True)
                t_edell = time.perf_counter()
                if f_ensin is None:
                    f_ensin = f
                if not lahde.live:                                   # tiedosto: tahdistus kuin live-virrassa
                    odotus = t_alku + (f - f_ensin) / fps - time.time()
                    if odotus > 0:
                        time.sleep(odotus)
                    myohassa = -odotus
                else:
                    myohassa = 0.0
                if piirtoa_tarvitaan:
                    jono.append((f, time.time(), pakkaus.submit(pakkaa, fr, a.puskuri == "jpeg" or len(jono) > yuv_raja)))
                if myohassa > 0.8:                                  # analyysi myohassa: ruutu nakyy tulosteessa, mutta sita ei analysoida
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
                    t_syota = time.perf_counter()
                    an.syota(f, fr)
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
        try:
            while not pysayta.is_set():
                nyt = time.time()
                while jono and len(odottaa) < 6 and jono[0][1] + a.viive - nyt < 0.4:       # purku etukateen (rinnakkain)
                    f, t_s, tulevaisuus = jono.popleft()
                    odottaa.append((f, t_s + a.viive, purku.submit(pura, tulevaisuus)))
                if odottaa and odottaa[0][1] <= nyt:
                    f, erapaiva, tehtava = odottaa.popleft()
                    kuva = tehtava.result()
                    myohassa = nyt - erapaiva
                    tilasto["myohassa_max"] = max(tilasto["myohassa_max"], myohassa)
                    if myohassa > 3.0 and not varoitettu:
                        varoitettu = True
                        print(f"Huom: piirto / tallennus on {myohassa:.1f} s jaljessa reaaliajasta (kone ei jaksa); ikkuna ohittaa ruutuja, tallenteesta ei puutu ruutuja, "
                              "mutta kuvan viive kasvaa. Kokeile --saikeet pienemmaksi tai --ulos ilman ikkunaa (--ei-ikkunaa).", file=sys.stderr)
                    try:
                        npi_kuva = piirtaja.piirra(kuva, f, an)
                    except Exception:                               # piirtovirhe: nayta ruutu ilman (loppuun asti piirrettyja) piirroksia
                        npi_kuva = kuva
                        if not piirtovirhe:
                            print("Piirtovirhe, jatketaan ilman piirroksia:", file=sys.stderr)
                            traceback.print_exc()
                            piirtovirhe.append(1)
                    if kirjoittaja is not None:
                        kirjoittaja.kirjoita(npi_kuva)
                    if naytto is not None:
                        naytto.anna(npi_kuva)
                    tilasto["piirretty"] += 1
                    continue
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
        if not a.hiljaa:
            print(f"analysoitu {tilasto['analysoitu']} ruutua, ohitettu (myohassa) {tilasto['ohitettu']}, virheita {tilasto['virheita']}; piirretty {tilasto['piirretty']} ruutua"
                  + (f", ikkunassa {naytto.n_nayt}" if naytto is not None else "") + f", suurin viive piirrossa {tilasto['myohassa_max']:.1f} s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--youtube", nargs="?", const=YOUTUBE_OLETUS, help=f"YouTube-tunnus tai -osoite (oletus {YOUTUBE_OLETUS})")
    ap.add_argument("--yt-formaatti", default=YT_FORMAATTI, help="yt_dlp-formaattivalitsin (oletus: oma testikoodi)")
    ap.add_argument("--lahde", help="videotiedosto")
    ap.add_argument("--url", help="suora virran osoite (http / https / rtsp / m3u8), luetaan kuten YouTube-virta")
    ap.add_argument("--kansio", default="tulokset", help="kalibroinnit ja taustakuvat (kalibrointi_*.json, moodikuva_*[_puhdas].png, kivimalli_*.json, kierto_muoto_*.json)")
    ap.add_argument("--viive", type=float, default=5.0, help="tulosteen viive analyysiin nahden (s)")
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
    ap.add_argument("--puskuri", choices=("yuv", "jpeg"), default="yuv",
                    help="viivepuskurin muoto: yuv = raaka YUV 4:2:0 (3,1 MB / ruutu, ~470 MB 5 s:n viiveella, kevyt prosessorille) tai jpeg (0,3 MB / ruutu, ~21 ms / ruutu prosessoria)")
    ap.add_argument("--jpeg", type=int, default=92, help="JPEG-laatu (--puskuri jpeg ja yuv-puskurin ylivuoto)")
    ap.add_argument("--tapahtumat", action="store_true", help="tulosta seurannan kaikki tapahtumat (viritys, lahto, vahvistus, hylkays, ei kivea, valmis): nayttaa myos heittoyritykset, jotka jaavat tunnistamatta")
    ap.add_argument("--hiljaa", action="store_true")
    ap.add_argument("--debug-viive", action="store_true", help="tulosta hitaat analyysiruudut (selvittaa miksi ruutuja ohitetaan)")
    ap.add_argument("--kello-loki", help="(testaus, vain --nopea) kirjoita kuvassa nakyvan kellon tila kerran sekunnissa CSV:hen: aika, kayva joukkue, kuluva aika, viimeisin heitto")
    a = ap.parse_args()
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
    geom = ng.Geometria(a.kansio)
    kello_paikka = tuple(int(x) for x in a.kello_paikka.split(",")) if a.kello_paikka else None
    piirtaja = npi.Piirtaja(geom, fps=lahde.fps, jalki_s=a.jalki, viivat_s=a.viivat, viive_s=a.viive, kello=not a.ei_kelloa, kello_paikka=kello_paikka)
    kirjoittaja = Kirjoittaja(a.ulos, lahde.fps, a.koodekki, a.ffmpeg, a.x264_preset) if a.ulos else None
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
