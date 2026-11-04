"""Palvelinajon kaynnistaja (kontin paaohjelma): lukee asetukset ymparistomuuttujista, rakentaa netti_live.py:n komentorivin ja ajaa sen lapsiprosessina.

    python netti_palvelin.py                  # asetukset ymparistomuuttujista (NETTI_*, ks. README_unraid.md)
    python netti_palvelin.py --kiri 0         # tuntemattomat valitsimet annetaan sellaisenaan netti_live.py:lle
    python netti_palvelin.py --nayta          # tulosta asetukset ja komento, ala aja

Lahde (NETTI_LAHDE):
  * YouTube-tunnus tai -osoite, tai suora virran osoite (http / https / rtsp / m3u8)  -> --youtube / --url
  * videotiedosto (/data/video/peli.mp4)                                             -> --lahde (oletuksena --nopea: ajetaan niin nopeasti kuin ehtii)
  * kansio tai kuvio (/data/video, /data/video/*.mp4)                                 -> kaikki sen videot yksi kerrallaan; NETTI_JATKUVA=1: kansiota valvotaan, uudet videot analysoidaan
                                                                                       (valmiit kirjataan tiedostoon <ulos>/kasitellyt.json eika niita ajeta uudelleen)
  Oletus (NETTI_LAHDE puuttuu): kansio /data/video valvontatilassa.

NETTI_JATKUVA=1 = valvonta: kun ajo paattyy (live-lahetys loppui, tai YouTube-lahetysta ei viela ole) odotetaan NETTI_ODOTUS_S sekuntia ja yritetaan uudelleen; toistuvissa
nopeissa epaonnistumisissa odotus kaksinkertaistuu (enintaan 15 min), jotta YouTubea ei kuormiteta.

SIGTERM / SIGINT (docker stop): lapsiprosessi saa saman signaalin ja viimeistelee tallenteen ja CSV:n; jos se ei ehdi NETTI_SULKU_S sekunnissa, se tapetaan.
Vain standardikirjasto (+ netti_ymparisto.py)."""
import argparse
import glob
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time

import netti_ymparisto as ny

HERE = os.path.dirname(os.path.abspath(__file__))
VIDEOPAATTEET = (".mp4", ".mkv", ".mov", ".avi", ".webm", ".ts", ".m4v", ".mpg", ".mpeg")
YOUTUBE_TUNNUS = re.compile(r"^[A-Za-z0-9_-]{11}$")
VERKKO = re.compile(r"^(https?|rtsp|rtmps?|udp|tcp|srt)://", re.IGNORECASE)
TOTUUS = {"1", "true", "yes", "on", "kylla", "k", "y", "t"}
EPATOTUUS = {"0", "false", "no", "off", "ei", "e", "n", "f"}
NOPEA_EPAONNISTUMINEN_S = 120.0          # ajo, joka paattyy virheeseen nopeammin: lahde ei ole saatavilla (esim. lahetys ei ole alkanut) -> odotus pitenee
ODOTUS_MAX_S = 900.0
YRITYKSIA_MAX = 3                        # tiedosto, jonka analyysi epaonnistuu nain monta kertaa, ohitetaan valvonnassa


class AsetusVirhe(Exception):
    pass


def loki(*a):
    print("[palvelin]", *a, flush=True)


# ---------------------------------------------------------------------------------------------------------------------
# ASETUKSET
# ---------------------------------------------------------------------------------------------------------------------
def _totuus(env, nimi, oletus):
    arvo = env.get(nimi)
    if arvo is None or arvo.strip() == "":                         # tyhja = ei asetettu (Unraidin mallipohja valittaa myos tyhjat muuttujat kontille)
        return oletus
    a = arvo.strip().lower()
    if a in TOTUUS:
        return True
    if a in EPATOTUUS:
        return False
    raise AsetusVirhe(f"{nimi}={arvo!r}: odotettiin 1 tai 0 (kylla / ei)")


def _luku(env, nimi, oletus, tyyppi=float, alaraja=None, ylaraja=None):
    arvo = env.get(nimi)
    if arvo is None or arvo.strip() == "":
        return oletus
    try:
        x = tyyppi(arvo.strip().replace(",", "."))             # desimaalipilkku kelpaa
    except ValueError:
        raise AsetusVirhe(f"{nimi}={arvo!r}: odotettiin lukua")
    if (alaraja is not None and x < alaraja) or (ylaraja is not None and x > ylaraja):
        raise AsetusVirhe(f"{nimi}={arvo!r}: sallittu alue {alaraja} ... {ylaraja}")
    return x


def _teksti(env, nimi, oletus=None):
    arvo = env.get(nimi)
    return oletus if arvo is None or arvo.strip() == "" else arvo.strip()


class Asetukset:
    """Palvelinajon asetukset (ymparistomuuttujat NETTI_*); virheellinen arvo -> AsetusVirhe selkealla viestilla."""

    def __init__(self, env):
        self.data = _teksti(env, "NETTI_DATA", "/data")
        self.lahde = _teksti(env, "NETTI_LAHDE")
        self.video_kansio = _teksti(env, "NETTI_VIDEO_KANSIO", os.path.join(self.data, "video"))
        self.ulos_kansio = _teksti(env, "NETTI_ULOS_KANSIO", os.path.join(self.data, "ulos"))
        self.tyo_kansio = _teksti(env, "NETTI_TYOKANSIO", os.path.join(os.environ.get("TMPDIR", "/tmp"), "netti_tyo"))
        self.kansio = _teksti(env, "NETTI_KANSIO")                 # kalibrointikansio (nimi tai polku); tyhja = automaattinen valinta
        self.tallenna_video = _totuus(env, "NETTI_VIDEO", True)
        self.csv = _totuus(env, "NETTI_CSV", True)
        self.aani = _teksti(env, "NETTI_AANI", "auto")
        self.koodekki = _teksti(env, "NETTI_KOODEKKI", "cpu")      # palvelimella oletus libx264: laitteistokooderit vaativat ajurit kontissa (NETTI_KOODEKKI=auto kokeilee niita)
        self.preset = _teksti(env, "NETTI_X264_PRESET", "superfast")
        self.viive = _luku(env, "NETTI_VIIVE", 10.0, float, 0.0, 120.0)
        self.saikeet = _luku(env, "NETTI_SAIKEET", None, int, 0, 64)
        self.nopea = _totuus(env, "NETTI_NOPEA", True)             # vain tiedostolahteelle
        self.alku = _luku(env, "NETTI_ALKU", None, float, 0.0)
        self.loppu = _luku(env, "NETTI_LOPPU", None, float, 0.0)
        self.lataus = _luku(env, "NETTI_LATAUS", None, float, 0.0, 600.0)
        self.kiri = _luku(env, "NETTI_KIRI", None, float, 0.0, 600.0)
        self.yt_formaatti = _teksti(env, "NETTI_YT_FORMAATTI")
        self.ei_kierretta = _totuus(env, "NETTI_EI_KIERRETTA", False)
        self.ei_kelloa = _totuus(env, "NETTI_EI_KELLOA", False)
        self.hiljaa = _totuus(env, "NETTI_HILJAA", False)
        self.web_portti = _luku(env, "NETTI_WEB_PORTTI", 8080, int, 0, 65535)
        self.web_leveys = _luku(env, "NETTI_WEB_LEVEYS", 1280, int, 160, 1920)
        self.web_fps = _luku(env, "NETTI_WEB_FPS", 8.0, float, 0.5, 30.0)
        self.web_laatu = _luku(env, "NETTI_WEB_LAATU", 75, int, 10, 100)
        self.odotus_s = _luku(env, "NETTI_ODOTUS_S", 60.0, float, 1.0, 86400.0)
        self.vakaus_s = _luku(env, "NETTI_VAKAUS_S", 30.0, float, 0.0, 3600.0)       # valvonta: tiedoston on oltava muuttumatta tama aika (kopiointi kesken ei kelpaa)
        self.sulku_s = _luku(env, "NETTI_SULKU_S", 60.0, float, 1.0, 3600.0)
        try:
            self.lisa = shlex.split(env.get("NETTI_LISAVALITSIMET", ""))
        except ValueError as e:
            raise AsetusVirhe(f"NETTI_LISAVALITSIMET: {e}")
        self.laji, self.arvo = luokittele_lahde(self.lahde, self.video_kansio)
        jatkuva_oletus = self.lahde is None or self.laji == "tiedostot" and self.arvo.get("kokoelma", False)
        self.jatkuva = _totuus(env, "NETTI_JATKUVA", jatkuva_oletus)
        self.kasitellyt_polku = os.path.join(self.ulos_kansio, "kasitellyt.json")


def luokittele_lahde(lahde, video_kansio):
    """(laji, arvo): ('youtube', ref) | ('url', osoite) | ('tiedostot', {'kuvio': ..., 'kokoelma': bool}). Puuttuva lahde = video-kansio (valvonta)."""
    if lahde is None:
        return "tiedostot", dict(kuvio=os.path.join(video_kansio, "*"), kokoelma=True, kansio=True)
    s = lahde.strip().strip('"').strip("'")
    if VERKKO.match(s):
        return ("youtube", s) if re.search(r"youtube\.com|youtu\.be|youtube-nocookie\.com", s, re.IGNORECASE) else ("url", s)
    polku = s if os.path.isabs(s) else os.path.join(video_kansio, s)
    if os.path.isdir(polku):
        return "tiedostot", dict(kuvio=os.path.join(glob.escape(polku), "*"), kokoelma=True, kansio=True)
    if os.path.isfile(polku):
        return "tiedostot", dict(kuvio=polku, kokoelma=False, kansio=False)
    if os.path.isfile(s):                                           # suhteellinen polku nykyisesta kansiosta
        return "tiedostot", dict(kuvio=os.path.abspath(s), kokoelma=False, kansio=False)
    if any(c in s for c in "*?["):
        return "tiedostot", dict(kuvio=polku, kokoelma=True, kansio=False)
    if YOUTUBE_TUNNUS.match(s):
        return "youtube", s
    raise AsetusVirhe(f"NETTI_LAHDE={lahde!r}: ei YouTube-tunnus / verkko-osoite eika olemassa oleva tiedosto, kansio tai kuvio (videot haetaan kansiosta {video_kansio})")


def tiedostot(arvo, nyt=None, vakaus_s=0.0):
    """Lahteen videotiedostot aakkosjarjestyksessa. kokoelma: vain videopaatteiset, ei piilotiedostoja, ei viela kopioitavana olevia (muuttumaton vakaus_s sekuntia)."""
    nyt = time.time() if nyt is None else nyt
    if not arvo["kokoelma"]:                                        # yksittainen tiedosto: sellaisenaan (nimessa voi olla kuvion merkkeja)
        return [arvo["kuvio"]] if os.path.isfile(arvo["kuvio"]) else []
    out = []
    for p in sorted(glob.glob(arvo["kuvio"])):
        if not os.path.isfile(p):
            continue
        if arvo["kokoelma"]:
            nimi = os.path.basename(p)
            if nimi.startswith(".") or not nimi.lower().endswith(VIDEOPAATTEET):
                continue
            try:
                st = os.stat(p)
            except OSError:
                continue
            if st.st_size <= 0 or nyt - st.st_mtime < vakaus_s:
                continue
        out.append(p)
    return out


def tunniste(laji, arvo):
    """Lyhyt tunniste tulostiedostojen nimiin."""
    if laji == "youtube":
        m = re.search(r"(?:v=|youtu\.be/|/live/|/shorts/|/embed/)([A-Za-z0-9_-]{11})", arvo) or re.fullmatch(r"([A-Za-z0-9_-]{11})", arvo)
        return "yt_" + m.group(1) if m else "yt"
    if laji == "url":
        return "virta"
    return re.sub(r"[^0-9A-Za-z_-]+", "_", os.path.splitext(os.path.basename(arvo))[0])[:60].strip("_") or "video"


def valitse_lataus(muisti_gt):
    """YouTube / virta: esilatauspuskurin pituus (s) pienen muistin koneelle (None = netti_live:n oletus 30 s). Puskurin ruudut ovat pakattuja, ~3,1 Mt / ruutu: 30 s = 2,8 Gt."""
    if muisti_gt is None or muisti_gt >= 10.0:
        return None
    return 15.0 if muisti_gt >= 6.0 else 8.0


def rakenna_komento(a, laji, arvo, leima, tila_polku=None, muisti_gt=None, lisa=()):
    """netti_live.py:n komentorivi yhdelle ajolle."""
    tag = tunniste(laji, arvo)
    cmd = [sys.executable, "-u", os.path.join(HERE, "netti_live.py"), {"youtube": "--youtube", "url": "--url", "tiedostot": "--lahde"}[laji], arvo, "--ei-ikkunaa"]
    if a.web_portti:
        cmd += ["--web", str(a.web_portti), "--web-leveys", str(a.web_leveys), "--web-fps", f"{a.web_fps:g}", "--web-laatu", str(a.web_laatu)]
    if a.tallenna_video:
        cmd += ["--ulos", os.path.join(a.ulos_kansio, f"kivenseuranta_{tag}_{leima}.mp4"), "--koodekki", a.koodekki, "--x264-preset", a.preset, "--aani", a.aani]
    if a.csv:
        cmd += ["--csv", os.path.join(a.ulos_kansio, f"heitot_{tag}_{leima}.csv")]
    else:
        cmd += ["--ei-csv"]
    if a.kansio:
        cmd += ["--kansio", a.kansio]
    cmd += ["--viive", f"{a.viive:g}"]
    if a.saikeet is not None:
        cmd += ["--saikeet", str(a.saikeet)]
    if laji == "tiedostot":
        if a.nopea:
            cmd += ["--nopea"]
        if a.alku is not None:
            cmd += ["--alku", f"{a.alku:g}"]
        if a.loppu is not None:
            cmd += ["--loppu", f"{a.loppu:g}"]
    else:
        lataus = a.lataus if a.lataus is not None else valitse_lataus(muisti_gt)
        if lataus is not None:
            cmd += ["--lataus", f"{lataus:g}"]
        if a.kiri is not None:
            cmd += ["--kiri", f"{a.kiri:g}"]
    if a.yt_formaatti:
        cmd += ["--yt-formaatti", a.yt_formaatti]
    for lippu, paalla in (("--ei-kierretta", a.ei_kierretta), ("--ei-kelloa", a.ei_kelloa), ("--hiljaa", a.hiljaa)):
        if paalla:
            cmd.append(lippu)
    if tila_polku:
        cmd += ["--tila-tiedosto", tila_polku]
    return cmd + list(a.lisa) + list(lisa)


# ---------------------------------------------------------------------------------------------------------------------
# KASITELLYT (valvontatila)
# ---------------------------------------------------------------------------------------------------------------------
class Kasitellyt:
    """Valvonnan muisti (<ulos>/kasitellyt.json): mitka lahteet on jo analysoitu (tiedosto: koko + muokkausaika, jotta korvattu tiedosto ajetaan uudelleen) tai epaonnistunut liian monta kertaa."""

    def __init__(self, polku):
        self.polku = polku
        try:
            with open(polku, "r", encoding="utf-8") as f:
                self.data = json.load(f)
            if not isinstance(self.data, dict):
                self.data = {}
        except (OSError, ValueError):
            self.data = {}

    @staticmethod
    def avain(laji, arvo):
        return ("yt:" if laji == "youtube" else "url:" if laji == "url" else "tiedosto:") + arvo

    @staticmethod
    def _sormenjalki(laji, arvo):
        if laji != "tiedostot":
            return {}
        try:
            st = os.stat(arvo)
            return dict(koko=st.st_size, muutettu=round(st.st_mtime, 3))
        except OSError:
            return {}

    def valmis(self, laji, arvo):
        r = self.data.get(self.avain(laji, arvo))
        if not r:
            return False
        if laji == "tiedostot" and r.get("sormenjalki") != self._sormenjalki(laji, arvo):
            return False                                       # tiedosto on vaihtunut
        return r.get("tila") == "ok" or (r.get("tila") == "virhe" and r.get("yrityksia", 0) >= YRITYKSIA_MAX)

    def kirjaa(self, laji, arvo, onnistui):
        a = self.avain(laji, arvo)
        r = self.data.get(a, {})
        if laji == "tiedostot" and r.get("sormenjalki") != self._sormenjalki(laji, arvo):
            r = {}                                             # uusi tiedosto samalla nimella: laskuri alusta
        r = dict(tila="ok" if onnistui else "virhe", yrityksia=0 if onnistui else r.get("yrityksia", 0) + 1, aika=time.strftime("%Y-%m-%d %H:%M:%S"), sormenjalki=self._sormenjalki(laji, arvo))
        self.data[a] = r
        self.tallenna()
        return r

    def tallenna(self):
        try:
            os.makedirs(os.path.dirname(self.polku), exist_ok=True)
            tmp = self.polku + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=1, sort_keys=True)
            os.replace(tmp, self.polku)
        except OSError as e:
            loki(f"huom: {self.polku} ei voitu kirjoittaa ({e}); valvonta voi ajaa videon uudelleen kaynnistyksen jalkeen")


# ---------------------------------------------------------------------------------------------------------------------
# LAPSIPROSESSI JA SIGNAALIT
# ---------------------------------------------------------------------------------------------------------------------
class Pysaytys:
    """SIGTERM / SIGINT pyytaa lopettamaan: asettaa tapahtuman, jota silmukat ja lapsiprosessin odotus seuraavat."""

    def __init__(self):
        self.tapahtuma = threading.Event()
        self.signaali = None

    def asenna(self):
        for s in (signal.SIGTERM, signal.SIGINT):
            signal.signal(s, self._kasittele)

    def _kasittele(self, signum, frame):
        self.signaali = signum
        self.tapahtuma.set()

    def pyydetty(self):
        return self.tapahtuma.is_set()

    def nuku(self, sekuntia):
        """Keskeytettava odotus; True, jos lopetusta pyydettiin."""
        return self.tapahtuma.wait(sekuntia)


def aja_lapsi(komento, pysaytys, sulku_s=60.0, env=None, cwd=None):
    """Ajaa komennon ja odottaa sen loppuun; palauttaa paluukoodin. Lopetuspyynto valitetaan lapselle samana signaalina (se viimeistelee ajon); jos lapsi ei lopu sulku_s sekunnissa, se tapetaan."""
    lapsi = subprocess.Popen(komento, env=env, cwd=cwd)
    lahetetty = None
    while True:
        try:
            return lapsi.wait(timeout=0.25)
        except subprocess.TimeoutExpired:
            pass
        if pysaytys.pyydetty():
            nyt = time.monotonic()
            if lahetetty is None:
                lahetetty = nyt
                nimi = signal.Signals(pysaytys.signaali).name if pysaytys.signaali else "SIGTERM"
                loki(f"{nimi}: pyydetaan ajoa viimeistelemaan (enintaan {sulku_s:.0f} s)")
                _laheta(lapsi, pysaytys.signaali or signal.SIGTERM)
            elif nyt - lahetetty > sulku_s:
                loki(f"ajo ei lopu {sulku_s:.0f} sekunnissa: tapetaan")
                _laheta(lapsi, signal.SIGKILL if hasattr(signal, "SIGKILL") else signal.SIGTERM)
                return lapsi.wait()


def _laheta(lapsi, signum):
    try:
        lapsi.send_signal(signum)
    except (ProcessLookupError, OSError):
        pass


def lue_tila(polku):
    try:
        with open(polku, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------------------------------------------------------------
# PAASILMUKKA
# ---------------------------------------------------------------------------------------------------------------------
def lahteet(a, kasitellyt=None, nyt=None):
    """Taman kierroksen ajettavat lahteet: [(laji, arvo)]. Valvonnassa jo kasitellyt jaavat pois."""
    if a.laji != "tiedostot":
        jono = [(a.laji, a.arvo)]
    else:
        jono = [("tiedostot", p) for p in tiedostot(a.arvo, nyt, a.vakaus_s if a.arvo["kokoelma"] else 0.0)]
    if kasitellyt is not None and a.jatkuva:
        jono = [(l, v) for l, v in jono if not kasitellyt.valmis(l, v)]
    return jono


def valmistele_kansiot(a):
    """Luo ulos- ja tyokansion ja varmistaa, etta ulos-kansioon voi kirjoittaa: (True, "") tai (False, syy). Selkea virhe heti eika vasta, kun tuntien analyysi ei pääse kirjoittamaan tulosta."""
    for k in (a.ulos_kansio, a.tyo_kansio):
        try:
            os.makedirs(k, exist_ok=True)
        except OSError as e:
            return False, f"kansiota {k} ei voi luoda ({e.strerror}); tarkista kansion oikeudet (kontti ajaa kayttajalla uid {os.getuid()}, PUID / PGID)"
    testi = os.path.join(a.ulos_kansio, f".kirjoitustesti_{os.getpid()}")
    try:
        open(testi, "w").close()
        os.remove(testi)
    except OSError as e:
        return False, f"kansioon {a.ulos_kansio} ei voi kirjoittaa ({e.strerror}); tarkista jakokansion oikeudet (kontti ajaa kayttajalla uid {os.getuid()}, PUID / PGID)"
    return True, ""


def palvele(a, pysaytys, ajuri=None, muisti_gt=None, kello=time.time):
    """Ajosilmukka. Palauttaa prosessin paluukoodin (0 = kaikki hyvin tai pysaytetty signaalilla; 2 = kansiot eivat kelpaa)."""
    ajuri = ajuri or (lambda cmd, env=None, cwd=None: aja_lapsi(cmd, pysaytys, a.sulku_s, env, cwd))
    ok, syy = valmistele_kansiot(a)
    if not ok:
        print(f"[palvelin] virhe: {syy}", file=sys.stderr, flush=True)
        return 2
    tila_polku = os.path.join(a.tyo_kansio, "ajon_tila.json")
    kasitellyt = Kasitellyt(a.kasitellyt_polku)
    vikoja = 0
    paluu = 0
    viim_tyhja_ilmoitus = None
    while True:
        jono = lahteet(a, kasitellyt, kello())
        if not jono:
            if not a.jatkuva:
                loki("ei analysoitavia videoita" + (f" (kansio {os.path.dirname(a.arvo['kuvio'])})" if a.laji == "tiedostot" else ""))
                return paluu or 3
            nyt = time.monotonic()
            if viim_tyhja_ilmoitus is None or nyt - viim_tyhja_ilmoitus > 1800:
                loki(f"valvonta: ei uusia videoita, tarkistetaan {a.odotus_s:.0f} s valein" + (f" ({os.path.dirname(a.arvo['kuvio'])})" if a.laji == "tiedostot" else ""))
                viim_tyhja_ilmoitus = nyt
            if pysaytys.nuku(a.odotus_s):
                return 0
            continue
        viim_tyhja_ilmoitus = None
        for laji, arvo in jono:
            if pysaytys.pyydetty():
                return 0
            leima = time.strftime("%Y%m%d_%H%M%S")
            try:
                os.remove(tila_polku)
            except OSError:
                pass
            cmd = rakenna_komento(a, laji, arvo, leima, tila_polku, muisti_gt)
            loki("kaynnistetaan: " + " ".join(shlex.quote(x) for x in cmd[1:]))
            t0 = time.monotonic()
            rc = ajuri(cmd, env=None, cwd=a.tyo_kansio)
            kesto = time.monotonic() - t0
            if pysaytys.pyydetty():
                loki(f"pysaytetty (ajo paattyi koodiin {rc}, kesto {kesto:.0f} s)")
                return 0
            tila = lue_tila(tila_polku) or {}
            loki(f"ajo paattyi: koodi {rc}, kesto {kesto:.0f} s")
            if rc != 0:
                paluu = rc
            if not a.jatkuva:
                continue
            if laji == "tiedostot":
                r = kasitellyt.kirjaa(laji, arvo, rc == 0)
                if rc != 0:
                    loki(f"analyysi epaonnistui ({os.path.basename(arvo)}, yritys {r['yrityksia']} / {YRITYKSIA_MAX})" + ("; ohitetaan jatkossa" if r["yrityksia"] >= YRITYKSIA_MAX else ""))
            elif rc == 0 and not tila.get("live", True):          # VOD valmis: ei toisteta (live-lahetys sen sijaan ajetaan uudelleen, kun seuraava alkaa)
                kasitellyt.kirjaa(laji, arvo, True)
            if rc != 0 and kesto < NOPEA_EPAONNISTUMINEN_S:
                vikoja += 1
            elif rc == 0:
                vikoja = 0
            if laji != "tiedostot":
                if rc == 0 and kasitellyt.valmis(laji, arvo):
                    loki("lahde analysoitu kokonaan (ei live-lahetys): sita ei ajeta uudelleen")
                    continue
                odotus = min(a.odotus_s * 2 ** max(vikoja - 1, 0), ODOTUS_MAX_S)
                loki(f"valvonta: uusi yritys {odotus:.0f} s kuluttua")
                if pysaytys.nuku(odotus):
                    return 0
        if not a.jatkuva:
            return paluu
        if a.laji == "tiedostot" and pysaytys.nuku(a.odotus_s):         # YouTube / virta: odotus (ja sen pidentyminen) on jo tehty ylempana
            return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nayta", action="store_true", help="tulosta asetukset ja netti_live.py-komento, ala aja")
    omat, lisa = ap.parse_known_args(argv)
    try:
        a = Asetukset(os.environ)
    except AsetusVirhe as e:
        print(f"[palvelin] asetusvirhe: {e}", file=sys.stderr, flush=True)
        return 2
    muisti = ny.muisti_gt()
    loki(f"{ny.kuvaus()}; lahde: {a.laji} {a.arvo if a.laji != 'tiedostot' else a.arvo['kuvio']}; valvonta {'paalla' if a.jatkuva else 'pois'}; tulokset: {a.ulos_kansio}"
         + (f"; selain: portti {a.web_portti}" if a.web_portti else "; selainnakyma pois"))
    if omat.nayta:
        jono = lahteet(a, None, None) or [(a.laji, a.arvo if a.laji != "tiedostot" else "<video>")]
        loki("komento: " + " ".join(shlex.quote(x) for x in rakenna_komento(a, jono[0][0], jono[0][1], time.strftime("%Y%m%d_%H%M%S"), None, muisti, lisa)))
        return 0
    a.lisa = list(a.lisa) + list(lisa)
    pysaytys = Pysaytys()
    pysaytys.asenna()
    try:
        return palvele(a, pysaytys, muisti_gt=muisti)
    except Exception as e:                                         # silmukan virhe ei saa jattaa konttia hiljaa hengittamaan
        loki(f"virhe: {e!r}")
        raise


if __name__ == "__main__":
    sys.exit(main())
