"""Reaaliaikainen kivenseuranta: kuva tulostuu VIIVE_S sekuntia analyysin perassa, ilman top-down-muunnosta.

    python netti_live.py --youtube hnTTLqHr5x8                     # YouTube (yt_dlp + OpenCV, kuten oma testikoodi), ikkuna + 'q' lopettaa
    python netti_live.py --youtube hnTTLqHr5x8 --ulos              # sama + tallennus tiedostoon kivenseuranta_<paivays>_<aika>.mp4 (tai --ulos oma.mp4)
    python netti_live.py --lahde netti.mp4                         # tiedosto reaaliaikana (nopeus kuten live-lahteessa)
    python netti_live.py --lahde netti.mp4 --nopea --ulos ulos.mp4 --ei-ikkunaa     # simulaatio: ajetaan niin nopeasti kuin ehtii, tulos kirjoitetaan tiedostoon

Toiminta (kaikissa tiloissa sama):
  * Jokainen saapuva ruutu analysoidaan heti (netti_seuranta.Analyysi): kopioruudut pois, kiven tunnistus hackista, kahvan vari + tarkka siluettisovitus
    kiven ymparilla, heiton tunnistus ja liun ja irroituksen viivaestimaatti, joka tarkentuu koko ajan.
  * Tulostettava kuva on analyysia VIIVE_S (oletus 5 s) jäljessa (puskuri). Kun kivi alkaa liikkua tulosteessa, analyysi on jo nahnyt seuraavat
    VIIVE_S sekuntia: toisen paan (kaukopaa) kuvaan piirtyy heti liukulinja ja irroituslinja, ja heittopaassa kiven keskikohta piirtyy pisteina
    (live: piste tulee kun sen ruutu nayttyy), pistejoukko jaa nakyviin JALKI_S (10 s) heiton jalkeen.
  * Ruudut sailyvat alkuperaisen nakoisina (1920 x 1080), piirrokset heijastetaan kamerageometrialla (netti_geom.py).
"""
import argparse
import collections
import csv
import os
import sys
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
# ULOSTULO
# ---------------------------------------------------------------------------------------------------------------------
class Ulos:
    """Ikkuna ja / tai tiedosto. Tiedosto kirjoitetaan oletuksena mp4v-koodekilla (toimii kaikkialla; iso tiedosto); H.264:ksi: ffmpeg -i ulos.mp4 -c:v libx264 -crf 23 valmis.mp4.
    Ikkuna: 'q' tai ESC lopettaa, myos ikkunan sulkeminen lopettaa; tiedosto viimeistellaan lopetettaessa (kirjoitusta ei pidä katkaista pakolla)."""

    def __init__(self, polku=None, ikkuna=True, fps=30.0, leveys_ikkuna=1280, koodekki="mp4v"):
        self.ikkuna = ikkuna
        self.nimi = "Kivenseuranta (kuva viiveella)"
        self.polku = polku
        self.n = 0
        self._nahty = 0                                  # ikkunaan piirrettyjen kuvien maara (sulkemisen tunnistus vasta kun ikkuna on ehtinyt aueta)
        if ikkuna:
            cv2.namedWindow(self.nimi, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.nimi, leveys_ikkuna, int(leveys_ikkuna * 9 / 16))
        self.vw = None
        if polku:
            kansio = os.path.dirname(os.path.abspath(polku))
            os.makedirs(kansio, exist_ok=True)
            self.vw = cv2.VideoWriter(polku, cv2.VideoWriter_fourcc(*koodekki), fps, (RUUTU_W, RUUTU_H))
            if not self.vw.isOpened():
                raise SystemExit(f"Tiedostoa {polku} ei voitu avata kirjoitettavaksi (koodekki {koodekki}); kokeile --koodekki mp4v tai toista tiedostonimi / kansio")

    def nayta_ilmoitus(self, kuva):
        """Ikkunaan kuva ilman tiedostoon kirjoitusta (puskurointi-ilmoitus)."""
        if self.ikkuna:
            cv2.imshow(self.nimi, kuva)
            self._nahty += 1

    def lopetus_pyydetty(self, odotus_ms=1):
        """Nappainpaine ('q' / ESC) tai ikkunan sulkeminen."""
        if not self.ikkuna:
            return False
        avain = cv2.waitKey(odotus_ms) & 0xFF
        if avain in (ord("q"), 27):
            return True
        if self._nahty >= 2:
            try:
                return cv2.getWindowProperty(self.nimi, cv2.WND_PROP_VISIBLE) < 1       # suljettu ikkuna: 0 tai -1
            except cv2.error:
                return True                                                             # ikkunaa ei ole enaa
        return False

    def nayta(self, kuva):
        """Kuva ikkunaan ja tiedostoon; palauttaa True, jos lopetus on pyydetty."""
        if self.vw is not None:
            self.vw.write(kuva)
        self.n += 1
        if self.ikkuna:
            cv2.imshow(self.nimi, kuva)
            self._nahty += 1
            return self.lopetus_pyydetty(1)
        return False

    def sulje(self):
        if self.vw is not None:
            self.vw.release()
        if self.ikkuna:
            cv2.destroyAllWindows()


class Kirjaaja:
    """Tulosten kirjaus konsoliin ja CSV:hen valmistuneista heitoista."""

    OTSIKKO = ["heitto", "vari", "puoli", "alku_s", "loppu_s", "irroitus_Y_cm", "irroituksen_suunta_deg", "irroituksen_suunta_virhe_deg", "liun_suunta_deg",
               "nopeus_ennen_m_s", "nopeus_jalkeen_m_s", "liun_hidastuvuus_m_s2", "T_liuku_cm", "T_irroitus_cm", "vakaa"]

    def __init__(self, csv_polku=None, hiljaa=False, fps=30.0):
        self.csv_polku, self.hiljaa, self.fps = csv_polku, hiljaa, fps
        self.ilmoitettu = set()
        self.valmis = set()
        if csv_polku:
            with open(csv_polku, "w", newline="") as f:
                csv.writer(f).writerow(self.OTSIKKO)

    def paivita(self, an):
        for h in an.heitot:
            if h.tunnus not in self.ilmoitettu:
                self.ilmoitettu.add(h.tunnus)
                if not self.hiljaa:
                    print(f"[{h.f_vahvistus / self.fps:7.1f} s] heitto {len(self.ilmoitettu)} ({h.vari}, {h.puoli}) vahvistettu, lahto {h.f_alku / self.fps:.1f} s", flush=True)
            if h.tila == "valmis" and h.tunnus not in self.valmis and h.viivat is not None and getattr(h, "lopullinen", False) and not an.est_tyot:
                self.valmis.add(h.tunnus)
                v = h.viivat
                L = ng.ETAISYYS_CM
                XT_l, XT_i = v["liuku_a"] + v["liuku_b"] * L, v["irr_X"] + np.tan(np.radians(v["irr_suunta_deg"])) * (L - v["irr_Y"])
                n = list(an.heitot).index(h) + 1
                if not self.hiljaa:
                    print(f"          heitto {n} valmis: irroitus Y={v['irr_Y']:.0f} cm, suunta {v['irr_suunta_deg']:+.2f} +-{v['irr_virhe_deg']:.2f} deg, liuku {v['liuku_suunta_deg']:+.2f} deg, "
                          f"nopeus {v['nopeus_ennen']:.2f} -> {v['nopeus_jalkeen']:.2f} m/s, liun hidastuvuus {v['liuku_hidastuvuus']:+.3f} m/s2; kaukopaan T-viiva: liuku {XT_l:+.0f}, irroitus {XT_i:+.0f} cm", flush=True)
                if self.csv_polku:
                    with open(self.csv_polku, "a", newline="") as f:
                        csv.writer(f).writerow([n, h.vari, h.puoli, f"{h.f_alku / self.fps:.2f}", f"{h.f_loppu / self.fps:.2f}", f"{v['irr_Y']:.1f}", f"{v['irr_suunta_deg']:.3f}",
                                                f"{v['irr_virhe_deg']:.3f}", f"{v['liuku_suunta_deg']:.3f}", f"{v['nopeus_ennen']:.3f}", f"{v['nopeus_jalkeen']:.3f}",
                                                f"{v['liuku_hidastuvuus']:.3f}", f"{XT_l:.1f}", f"{XT_i:.1f}", int(bool(v.get('vakaa')))])


    def historia(self, an):
        """Estimaattien kehitys heitoittain (analyysiaika lahdosta, irroitus, suunta, T-viivan osumat): nayttaa miten alustava arvio tarkentuu."""
        if self.hiljaa:
            return
        for i, h in enumerate(an.heitot):
            print(f"heiton {i + 1} estimaattien kehitys (aika lahdosta, irroitus Y, irroituksen suunta, T-viiva irroitus / liuku, vakaa):")
            for f, Y, th, XT_i, XT_l, vakaa in h.viivat_loki[:: max(1, len(h.viivat_loki) // 14)]:
                print(f"   +{(f - h.f_alku) / self.fps:5.1f} s: Y={Y:5.0f} cm  {th:+6.2f} deg  T-viiva {XT_i:+5.0f} / {XT_l:+5.0f} cm  {'vakaa' if vakaa else 'alustava'}")


# ---------------------------------------------------------------------------------------------------------------------
# SIMULAATIO: ajetaan niin nopeasti kuin ehtii, tulos riippumaton ajoituksesta (tyot odotetaan)
# ---------------------------------------------------------------------------------------------------------------------
def aja_sim(a, lahde, an, piirtaja, ulos, kirjaaja):
    D = int(round(a.viive * lahde.fps))
    cap2 = cv2.VideoCapture(lahde.tiedosto)          # toinen kahva: tulostettava kuva luetaan alkuperaisena (ei JPEG-pakkausta), D ruutua analyysin jalkeen
    cap2.set(cv2.CAP_PROP_POS_FRAMES, int(lahde.alku_s * lahde.fps))
    t0 = time.time()
    n = 0

    def tulosta(f_out, kuva=None):
        if kuva is None:
            ok, kuva = cap2.read()
            if not ok:
                return False
        out = piirtaja.piirra(kuva, f_out, an.kuvat())
        return not ulos.nayta(out)

    for f, fr in lahde.ruudut():
        an.syota(f, fr)
        n += 1
        kirjaaja.paivita(an)
        f_out = f - D
        if f_out >= int(lahde.alku_s * lahde.fps):
            if not tulosta(f_out):
                break
        if n % 300 == 0 and not a.hiljaa:
            print(f"  ... {n / lahde.fps:6.1f} s videota, {n / (time.time() - t0):.0f} r/s", flush=True)
    an.viimeistele()
    kirjaaja.paivita(an)
    kirjaaja.historia(an)
    f_alku_out = int(lahde.alku_s * lahde.fps)
    viim = f_alku_out + n - D
    for f_out in range(max(viim, f_alku_out), f_alku_out + n):         # viimeiset D ruutua
        if not tulosta(f_out):
            break
    kirjaaja.paivita(an)


# ---------------------------------------------------------------------------------------------------------------------
# REAALIAIKA: lukija + analyysi taustasaikeessa, tuloste paasaikeessa (ikkuna)
# ---------------------------------------------------------------------------------------------------------------------
def aja_reaaliaika(a, lahde, an, piirtaja, ulos, kirjaaja):
    jono = collections.deque()                       # (f, saapumisaika, jpeg)
    lopetettu = threading.Event()
    pysayta = threading.Event()
    fps = lahde.fps
    tilasto = dict(analysoitu=0, ohitettu=0, virheita=0)

    def saie():
        t_alku = time.time()
        f_ensin = None
        try:
            for f, fr in lahde.ruudut():
                if pysayta.is_set():
                    break
                if f_ensin is None:
                    f_ensin = f
                if not lahde.live:                                   # tiedosto: tahdistus kuin live-virrassa
                    odotus = t_alku + (f - f_ensin) / fps - time.time()
                    if odotus > 0:
                        time.sleep(odotus)
                    myohassa = -odotus
                else:
                    myohassa = 0.0
                ok, buf = cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, a.jpeg])
                jono.append((f, time.time(), buf))
                if myohassa > 0.8:                                  # analyysi myohassa: ruutu nakyy tulosteessa, mutta sita ei analysoida
                    tilasto["ohitettu"] += 1
                    an.ohita()
                    continue
                try:
                    an.syota(f, fr)
                    tilasto["analysoitu"] += 1
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

    s = threading.Thread(target=saie, daemon=True)
    s.start()
    odotus_nayttoa = False                           # True kun ensimmainen ruutu on nayttetty
    piirtovirhe = []
    viim_ilmoitus = 0.0
    try:
        while True:
            nyt = time.time()
            if jono and jono[0][1] + a.viive <= nyt:
                odotus_nayttoa = True
                f, _, buf = jono.popleft()
                kuva = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                try:
                    out = piirtaja.piirra(kuva, f, an.kuvat())
                except Exception:                                   # piirtovirhe: nayta ruutu ilman piirroksia
                    out = kuva
                    if not piirtovirhe:
                        print("Piirtovirhe, jatketaan ilman piirroksia:", file=sys.stderr)
                        traceback.print_exc()
                        piirtovirhe.append(1)
                lopeta = ulos.nayta(out)
                kirjaaja.paivita(an)
                if lopeta:
                    break
                continue
            if lopetettu.is_set() and not jono:
                break
            if ulos.ikkuna:
                if not odotus_nayttoa and jono and nyt - viim_ilmoitus > 0.25:      # ensimmaista ruutua odotettaessa: puskurointi-ilmoitus
                    ilm = np.zeros((RUUTU_H, RUUTU_W, 3), np.uint8)
                    cv2.putText(ilm, f"Puskuroidaan {max(0.0, jono[0][1] + a.viive - nyt):.1f} s (kuva tulee {a.viive:.0f} s viiveella)", (120, RUUTU_H // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 2, cv2.LINE_AA)
                    ulos.nayta_ilmoitus(ilm)
                    viim_ilmoitus = nyt
                if ulos.lopetus_pyydetty(3):
                    break
            else:
                time.sleep(0.003)
    finally:
        pysayta.set()
        s.join(timeout=5.0)
        an.viimeistele()
        kirjaaja.paivita(an)
        if not a.hiljaa:
            print(f"analysoitu {tilasto['analysoitu']} ruutua, ohitettu (myohassa) {tilasto['ohitettu']}, virheita {tilasto['virheita']}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--youtube", nargs="?", const=YOUTUBE_OLETUS, help=f"YouTube-tunnus tai -osoite (oletus {YOUTUBE_OLETUS})")
    ap.add_argument("--yt-formaatti", default=YT_FORMAATTI, help="yt_dlp-formaattivalitsin (oletus: oma testikoodi)")
    ap.add_argument("--lahde", help="videotiedosto")
    ap.add_argument("--url", help="suora virran osoite (http / https / rtsp / m3u8), luetaan kuten YouTube-virta")
    ap.add_argument("--kansio", default="tulokset", help="kalibroinnit ja taustakuvat (kalibrointi_*.json, moodikuva_*[_puhdas].png, kivimalli_*.json)")
    ap.add_argument("--viive", type=float, default=5.0, help="tulosteen viive analyysiin nahden (s)")
    ap.add_argument("--jalki", type=float, default=10.0, help="pistejoukko nakyvissa taman ajan heiton jalkeen (s)")
    ap.add_argument("--viivat", type=float, default=60.0, help="linjat nakyvissa taman ajan heiton alusta, tai seuraavaan heittoon (s)")
    ap.add_argument("--ulos", nargs="?", const="auto", help="tallenna tuloste tiedostoon (mp4); ilman nimea: kivenseuranta_<paivays>_<aika>.mp4 nykyiseen kansioon")
    ap.add_argument("--koodekki", default="mp4v", help="tallenteen koodekki (OpenCV fourcc; oletus mp4v, kokeile avc1 = H.264 jos OpenCV:si tukee)")
    ap.add_argument("--ikkuna-leveys", type=int, default=1280, help="ikkunan leveys pikseleina (kuva skaalautuu)")
    ap.add_argument("--csv", help="kirjaa valmistuneet heitot CSV:hen")
    ap.add_argument("--ei-ikkunaa", action="store_true", help="ei ikkunaa (palvelin / testaus)")
    ap.add_argument("--nopea", action="store_true", help="simulaatio: tiedosto ajetaan niin nopeasti kuin ehtii (tulos ei riipu ajoituksesta)")
    ap.add_argument("--alku", type=float, default=0.0, help="aloitus tiedostossa (s)")
    ap.add_argument("--loppu", type=float, default=None, help="lopetus tiedostossa (s)")
    ap.add_argument("--saikeet", type=int, default=max(1, min(3, (os.cpu_count() or 2) - 1)), help="tarkan sovituksen tyoprosessit (0 = vain kahvan vari: epatarkempi)")
    ap.add_argument("--fps-lahde", type=float, default=None, help="erillisten ruutujen nimellistaajuus (25 tassa videossa; oletus: paatellaan kopioruutujen osuudesta)")
    ap.add_argument("--jpeg", type=int, default=92, help="viivepuskurin JPEG-laatu (reaaliaika)")
    ap.add_argument("--hiljaa", action="store_true")
    a = ap.parse_args()
    if not os.path.isdir(a.kansio):                    # ajettu muualta kuin projektin kansiosta
        oma = os.path.join(os.path.dirname(os.path.abspath(__file__)), a.kansio)
        if os.path.isdir(oma):
            a.kansio = oma
    if a.ulos == "auto":
        a.ulos = time.strftime("kivenseuranta_%Y%m%d_%H%M%S.mp4")
    if not (a.youtube or a.lahde or a.url):
        ap.error("anna --youtube [TUNNUS], --lahde TIEDOSTO tai --url OSOITE")
    lahde = Lahde(tiedosto=a.lahde, youtube=a.youtube, alku_s=a.alku, loppu_s=a.loppu, yt_formaatti=a.yt_formaatti, url=a.url)
    sim = bool(a.nopea and a.lahde)
    an = ns.Analyysi(a.kansio, fps=lahde.fps, saikeet=a.saikeet, tarkka=a.saikeet > 0, synkroninen=sim, fps_lahde=a.fps_lahde)
    geom = ng.Geometria(a.kansio)
    piirtaja = npi.Piirtaja(geom, fps=lahde.fps, jalki_s=a.jalki, viivat_s=a.viivat, viive_s=a.viive)
    ulos = Ulos(a.ulos, ikkuna=not a.ei_ikkunaa, fps=lahde.fps, leveys_ikkuna=a.ikkuna_leveys, koodekki=a.koodekki)
    if not a.hiljaa:
        print(f"Kuva tulee {a.viive:.0f} s viiveella. " + ("Ikkuna: 'q' tai ESC (tai ikkunan sulkeminen) lopettaa. " if not a.ei_ikkunaa else "")
              + (f"Tallennus: {a.ulos}" if a.ulos else "Ei tallennusta (lisaa --ulos)."), flush=True)
    kirjaaja = Kirjaaja(a.csv, a.hiljaa, lahde.fps)
    try:
        if sim:
            aja_sim(a, lahde, an, piirtaja, ulos, kirjaaja)
        else:
            aja_reaaliaika(a, lahde, an, piirtaja, ulos, kirjaaja)
    finally:
        an.sulje()
        ulos.sulje()
        if a.ulos and not a.hiljaa:
            print(f"kirjoitettu {a.ulos} ({ulos.n} ruutua)")
            if a.koodekki == "mp4v":
                print(f"  pienempi H.264-tiedosto: ffmpeg -i {a.ulos} -c:v libx264 -crf 23 -pix_fmt yuv420p pieni_{os.path.basename(a.ulos)}")


if __name__ == "__main__":
    main()
