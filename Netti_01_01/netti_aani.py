"""Aani tallenteeseen (--ulos): YouTube-aanen lataus taustalla ajon aikana ja liitos valmiiseen kuvatiedostoon.

Kuvatallenne tehdaan ensin aanettomana (ffmpeg-putki: kuva ei riipu aanesta) ja aani liitetaan vasta lopuksi (kuva kopioidaan sellaisenaan, aani koodataan AAC:ksi). Aani on lahdevideon
aikajanalla (ruutu f <-> aani f / fps), joten se pysyy samassa kohdassa kuin kuva.

Vanha tapa haki aanen ffmpegilla suoraan YouTube-osoitteesta vasta ajon lopuksi: (1) YouTube rajoittaa suoratoiston nopeutta, joten koko ohjelman pituisen aanen haku kesti kauan, ja
(2) jos aanivirta loppui kesken (verkkokatko, YouTube katkaisi yhteyden), `-shortest` lyhensi valmiin kuvatallenteen aanen pituiseksi: ffmpeg palautti silti 0 ja ohjelma korvasi
tallenteen lyhyella. Nyt: yt_dlp lataa aanen taustasaikeessa heti ajon alussa (palasittain, uudelleenyrityksin) analyysin kanssa rinnakkain, lopuksi liitetaan paikallinen tiedosto
sekunneissa, aani tayteteen mykalla (apad) kuvan mittaiseksi eika kuvaa koskaan lyhenneta, ja tulos tarkistetaan (kuva- ja aaniraita, kesto) ennen kuin alkuperainen tallenne
korvataan. Jos aanta ei saada, kuvatallenne jaa aanettomana ja ehjana ja ladattu aani jaa tiedostoon (liitos jalkikateen: tools/liita_aani.py)."""
import os
import re
import shutil
import subprocess
import threading
import time

YT_AANI_FORMAATTI = "bestaudio[ext=m4a]/bestaudio/best"          # viimeinen vaihtoehto: kuva + aani -tiedosto (aani poimitaan liitoksessa)
_KESTO = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
_EI_IKKUNAA = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def etsi_ffmpeg(polku=None):
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


def onko_youtube(ref):
    s = str(ref)
    return "youtube.com" in s or "youtu.be" in s


def tiedot(exe, polku, aikaraja_s=120):
    """(kesto s tai None, on aaniraita, on kuvaraita) ffmpeg -i:n tulosteesta (paikallinen tiedosto tai osoite)."""
    try:
        r = subprocess.run([exe, "-hide_banner", "-i", str(polku)], capture_output=True, text=True, errors="replace", timeout=aikaraja_s, creationflags=_EI_IKKUNAA)
    except Exception:
        return None, False, False
    s = r.stderr
    m = _KESTO.search(s)
    kesto = None if m is None else 3600.0 * int(m.group(1)) + 60.0 * int(m.group(2)) + float(m.group(3))
    return kesto, "Audio:" in s, "Video:" in s


def _poista(polku):
    try:
        os.remove(polku)
    except OSError:
        pass


def liita(exe, kuva, osoite, alku_s=0.0, otsakkeet=None, aikaraja_s=3600, ulos=None):
    """Liittaa aanen (paikallinen tiedosto tai http-osoite; alku_s = aanen kohta, joka vastaa kuvan ensimmaista ruutua) valmiiseen kuvatiedostoon. Palauttaa (ok, viesti): onnistuessa
    viesti on mahdollinen huomautus (esim. aani loppuu ennen kuvaa), muuten epaonnistumisen syy.
    Kuva kopioidaan sellaisenaan ja aani tayteteen mykalla kuvan mittaiseksi (-af apad, -t kuvan kesto): lyhyt tai kesken katkennut aani ei lyhenna kuvaa. Alkuperainen tiedosto
    korvataan vasta, kun tulos on tarkistettu (kuva- ja aaniraita, kesto vahintaan kuvan mittainen); muuten se jaa ennalleen. ulos = oma tulostiedosto (kuvatiedosto jaa koskematta)."""
    kuva_kesto, _, kuvaa = tiedot(exe, kuva)
    if not kuvaa:
        return False, "kuvatiedostoa ei voitu lukea"
    verkko = str(osoite).startswith(("http://", "https://"))
    huom = ""
    if not verkko:
        if not os.path.exists(osoite):
            return False, f"aanitiedostoa ei loydy ({osoite})"
        a_kesto, aanta, _ = tiedot(exe, osoite)
        if not aanta:
            return False, f"aanilahteessa ei ole aaniraitaa ({osoite})"
        if kuva_kesto and a_kesto is not None and a_kesto - alku_s < kuva_kesto - 2.0:
            huom = f" (huom: aani loppuu {kuva_kesto - (a_kesto - alku_s):.0f} s ennen kuvaa, loppu on mykka)"
    kohde = str(ulos or kuva)
    tmp = kohde + ".aani.mp4"
    cmd = [exe, "-hide_banner", "-loglevel", "error", "-y", "-i", str(kuva)]
    if verkko:                                                                          # verkkosyote: otsakkeet ja uudelleenyhdistys katkoissa (vain http-syotteille)
        if otsakkeet:
            cmd += ["-headers", "".join(f"{k}: {v}\r\n" for k, v in otsakkeet.items())]
        cmd += ["-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "5"]
    if alku_s:
        cmd += ["-ss", f"{alku_s:.3f}"]
    cmd += ["-i", str(osoite), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-af", "apad"]
    cmd += ["-t", f"{kuva_kesto:.3f}"] if kuva_kesto else ["-shortest"]
    cmd += ["-movflags", "+faststart", tmp]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=aikaraja_s, creationflags=_EI_IKKUNAA)
    except Exception as e:
        _poista(tmp)
        return False, f"ffmpeg ei toiminut ({e!r})"
    if r.returncode != 0 or not os.path.exists(tmp):
        _poista(tmp)
        return False, (r.stderr.strip()[-300:] or f"ffmpeg paattyi virhekoodiin {r.returncode}")
    t_kesto, aanta, kuvaa2 = tiedot(exe, tmp)
    if not (aanta and kuvaa2):
        _poista(tmp)
        return False, "valmiissa tiedostossa ei ole seka kuva- etta aaniraitaa"
    if kuva_kesto and t_kesto is not None and t_kesto < kuva_kesto - max(1.0, 0.001 * kuva_kesto):
        _poista(tmp)
        return False, f"tulos olisi lyhyempi kuin kuva ({t_kesto:.1f} s < {kuva_kesto:.1f} s)"
    try:
        os.replace(tmp, kohde)
    except OSError as e:
        return False, f"tiedostoa ei voitu korvata ({e}); aanellinen versio on tiedostossa {tmp}"
    return True, huom


class AaniLataus:
    """YouTube-aanen lataus taustasaikeessa (yt_dlp: palasittain, uudelleenyrityksin): YouTube rajoittaa suoratoiston nopeutta, joten lataus tehdaan analyysin aikana eika vasta lopuksi.
    ref = YouTube-tunnus tai -osoite; kohde = tiedostopolku ilman paatetta (yt_dlp lisaa paatteen); lataa(kohde, edistys) = oma latausfunktio (testaus): palauttaa tiedostopolun
    ja ilmoittaa etenemisen kutsulla edistys(tavua, yhteensa tai None)."""

    def __init__(self, ref, kohde, ffmpeg=None, lataa=None):
        self.ref, self.kohde, self.ffmpeg = ref, kohde, ffmpeg
        self.polku = None
        self.virhe = None
        self.tavua = 0
        self.yht = None
        self.t_alku = time.time()
        self.t_edistys = self.t_alku
        self.valmis = threading.Event()
        self._lataa = lataa or self._yt_dlp
        threading.Thread(target=self._aja, name="aanen-lataus", daemon=True).start()

    def _edistys(self, tavua, yht=None):
        if tavua > self.tavua:
            self.t_edistys = time.time()
        self.tavua = tavua
        if yht:
            self.yht = yht

    def _aja(self):
        try:
            self.polku = self._lataa(self.kohde, self._edistys)
            if not self.polku or not os.path.exists(self.polku):
                raise RuntimeError("ladattua aanitiedostoa ei loydy")
        except Exception as e:
            self.virhe = e
        finally:
            self.valmis.set()

    def _yt_dlp(self, kohde, edistys):
        import yt_dlp
        url = self.ref if (onko_youtube(self.ref) or str(self.ref).startswith(("http://", "https://"))) else f"https://www.youtube.com/watch?v={self.ref}"

        def koukku(d):
            if d.get("status") == "downloading":
                edistys(int(d.get("downloaded_bytes") or 0), d.get("total_bytes") or d.get("total_bytes_estimate"))

        opts = {"format": YT_AANI_FORMAATTI, "outtmpl": kohde + ".%(ext)s", "quiet": True, "no_warnings": True, "noprogress": True, "noplaylist": True, "js_runtimes": {"deno": {}},
                "http_chunk_size": 10 << 20, "retries": 10, "fragment_retries": 10, "socket_timeout": 30, "progress_hooks": [koukku]}
        if self.ffmpeg:
            opts["ffmpeg_location"] = self.ffmpeg
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            ladatut = info.get("requested_downloads") or []
            return (ladatut[0].get("filepath") if ladatut else None) or ydl.prepare_filename(info)

    def edistys_teksti(self):
        if not self.tavua:
            return "Odotetaan aanen latausta (haetaan tietoja) ..."
        mt = self.tavua / 1e6
        if self.yht:
            return f"Odotetaan aanen latausta: {100.0 * self.tavua / self.yht:.0f} % ({mt:.0f} / {self.yht / 1e6:.0f} Mt) ..."
        return f"Odotetaan aanen latausta: {mt:.0f} Mt ..."

    def odota(self, ei_edistysta_s=180.0, raportti_s=5.0, loki=print):
        """Odottaa latauksen valmistumista (raportoi etenemisen). True = ladattu. Keskeytyy, kun lataus ei etene ei_edistysta_s sekuntiin tai kayttaja painaa Ctrl-C (False)."""
        t_rap = 0.0
        try:
            while not self.valmis.wait(0.5):
                nyt = time.time()
                if nyt - self.t_edistys > ei_edistysta_s:
                    loki(f"Aanen lataus ei ole edennyt {ei_edistysta_s:.0f} sekuntiin; lopetetaan odotus.")
                    return False
                if nyt - t_rap >= raportti_s:
                    t_rap = nyt
                    loki(self.edistys_teksti())
        except KeyboardInterrupt:
            loki("Aanen odotus keskeytetty (Ctrl-C).")
            return False
        return self.virhe is None and self.polku is not None


class Aanilahde:
    """Tallenteen aanilahde. laji: 'tiedosto' (paikallinen tiedosto: lahdevideon oma aani tai --aani POLKU), 'verkko' (http-osoite: ffmpeg lukee suoraan) tai 'youtube' (yt_dlp lataa
    aanen taustalla heti kaynnista():sta). alku_s = aanen kohta, joka vastaa kuvan ensimmaista ruutua (lahdevideon aikajanalla)."""

    def __init__(self, laji, osoite, alku_s=0.0, otsakkeet=None):
        self.laji, self.osoite, self.alku_s, self.otsakkeet = laji, osoite, alku_s, otsakkeet
        self.lataus = None
        self.kohde = None

    def kaynnista(self, kohde, exe=None, lataa=None):
        """YouTube-lahteella kaynnistaa latauksen taustasaikeessa (kohde = ladatun tiedoston polku ilman paatetta); muille ei tee mitaan."""
        if self.laji == "youtube" and self.lataus is None:
            self.kohde = kohde
            self.lataus = AaniLataus(self.osoite, kohde, exe, lataa)

    def valmista(self, loki=print):
        """(osoite, alku_s, otsakkeet) liitettavaksi tai None (syy lokiin): youtube-lahteella odottaa latauksen valmistumista."""
        if self.laji != "youtube":
            return self.osoite, self.alku_s, self.otsakkeet
        if self.lataus is None:
            return None
        if not self.lataus.valmis.is_set():
            loki("Aanta ladataan viela; odotetaan (Ctrl-C jattaa kuvan aanettomaksi ja ehjaksi) ...")
        if not self.lataus.odota(loki=loki):
            if self.lataus.virhe is not None:
                loki(f"Huom: aanen lataus epaonnistui ({str(self.lataus.virhe)[:300]}).")
            return None
        return self.lataus.polku, self.alku_s, None

    def siivoa(self):
        """Ladattu aanitiedosto pois, kun liitos onnistui (epaonnistuessa se jaa: liitos jalkikateen tools/liita_aani.py)."""
        if self.lataus is not None and self.lataus.polku:
            _poista(self.lataus.polku)


def liita_tallenteeseen(exe, kuva, lahde, loki=print, ulos=None):
    """Valmiin kuvatallenteen aanen liitos (Kirjoittaja.sulje): odottaa tarvittaessa latauksen, liittaa ja tarkistaa. Palauttaa True, jos aani liitettiin; muuten kuvatallenne jaa
    aanettomana ja ehjana. ulos = oma tulostiedosto (kuvatallenne jaa koskematta)."""
    loki(f"Liitetaan aani tallenteeseen ({os.path.basename(str(kuva))}) ...")
    a = lahde.valmista(loki)
    if a is None:
        jalki = f" Ladattu aani on tiedostossa {lahde.lataus.polku} (liitos: python tools/liita_aani.py {kuva} --aani {lahde.lataus.polku})." if lahde.lataus is not None and lahde.lataus.polku else ""
        loki(f"Huom: aanta ei saatu liitettya; kuvatallenne on aaneton ja ehja.{jalki}")
        return False
    ok, viesti = liita(exe, kuva, *a, ulos=ulos)
    if ok:
        lahde.siivoa()
        loki("Aani liitetty." + viesti)
        return True
    jalki = f" Ladattu aani on tiedostossa {a[0]}." if lahde.laji == "youtube" else ""
    loki(f"Huom: aanta ei saatu liitettya ({viesti}); kuvatallenne on aaneton ja ehja.{jalki}")
    return False
