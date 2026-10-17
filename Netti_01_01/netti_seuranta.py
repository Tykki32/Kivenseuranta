"""Reaaliaikainen kiven seuranta nettivideolta (alkuperaiset ruudut, ei top-down-muunnosta).

Ruutua EI muunneta kokonaan oikaistuksi kuvaksi: vain kiven ymparilla oleva leike (ROI) oikaistaan suoraan alkuperaisesta ruudusta yhdella
cv2.remap-kutsulla (kierto ja linssikorjaus valmiissa kartoissa, Kuvain).

Kaksi mittaustasoa:
  A) KAHVAN VARI (kevyt, joka ruudulle, paaprosessissa): kahvan vari (keltainen / punainen) -> sentroidi -> maa-asema (z = 15 cm). Siemen (kivi
     levossa hackissa), heiton alun ja lopun tunnistus, ennuste seuraavaan ruutuun, alkuarvo B:lle. Hajonta ~1-2 cm.
  B) TARKKA (kivi_tarkka.py, sama mittaus kuin offline-ajossa): graniitin peittavyys leikkeessa, kiven akseli (X, Y) sovitettuna alipikselina
     kiintean kokoiseen runkomalliin. ~50 ms / ruutu, ajetaan tyoprosesseissa (vain leike siirtyy prosessille). Hajonta ~0,1-0,3 cm.

Heiton tunnistus (Heittopaa): IDLE -> ARMED (kahva hack-alueella >= 0,5 s ja siina todettu kivi) -> LIIKKUU (kivi lahtee lepopaikastaan) -> VAHVISTETTU
(ylittanyt Y = -100 cm nopeudella >= 1,2 m/s mitatuista paikoista ja liikkeen aikana vahintaan 8 kelvollista kivisovitusta: kantaminen, kavelija ja ihmisten
varusteet eivat tayta) -> VALMIS (kivi poistuu kuva-alalta). Hylatty yritys (heittaja kasittelee kiveä) palaa ARMED-tilaan. Vahvistus tulee vasta noin 1-2 s
lahdon jalkeen, mutta lahtohetki (f_alku) ajoitetaan taaksepain: 5 s:n viiveella tulostettava kuva ei ole vielakaan ehtinyt lahtohetkeen.

Heiton jalkeen (Analyysi): lopullinen viivaestimaatti (laske_viivat), kierteen mittaus (netti_kierre: kuvaleikkeet kerataan seurannan aikana, sovitus taustaprosessissa), hog-viivan
ylitys heittopaassa (Heitto.ylitys_f) ja kaukopaassa (Saapuminen: kahvan variblobit toisen puoliskon kuvassa) -> hog-hog-aika ja kierrosmaara (Analyysi._paivita_kierre), kivien
numerointi vareittain per heittosuunta (Analyysi._numeroi) ja paneelin suureet (heiton_suureet). Tarkka sovitus (B) kayttaa C++-moduulia netti_fit, jos se on kaannetty.

Taustakuva: tulokset/moodikuva_<puoli>_puhdas.png jos olemassa (netti_tausta.py; hackin haamukivi pois), muuten moodikuva_<puoli>.png.
"""
import concurrent.futures
import multiprocessing
import os
import sys
import sysconfig
import types
import warnings
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

import aikaperusta as aikap
import kahva_malli as km
import kivi_netti as kv
import kivi_tarkka as kt
import kivimalli
import liuku_irroitus as li
import netti_kierre as nk
import netti_tausta
import rata

_netti_fit_virhe = None
try:                              # C++-toteutus B-sovitukselle (cpp/netti_fit.cpp, CMake-kohde netti_fit); ilman sita Python-toteutus
    if os.environ.get("NETTI_EI_CPP"):
        raise ImportError("NETTI_EI_CPP")
    import netti_fit as _netti_fit
except ImportError as _e:
    _netti_fit, _netti_fit_virhe = None, _e


def netti_fit_selitys():
    """Miksi C++-sovitin (netti_fit) ei ole kaytossa (tyhja, jos se on). Tavallisin syy: moduuli on kaannetty toiselle Pythonin versiolle kuin milla ohjelmaa ajetaan
    (netti_fit.cp312-win_amd64.pyd latautuu vain Python 3.12:lla): tiedosto on kansiossa, mutta Python ei tunnista sita (ModuleNotFoundError)."""
    if _netti_fit is not None:
        return ""
    if os.environ.get("NETTI_EI_CPP"):
        return "pois kaytosta (NETTI_EI_CPP)"
    try:
        tiedostot = sorted(f for f in os.listdir(os.path.dirname(os.path.abspath(__file__))) if f.startswith("netti_fit") and f.endswith((".pyd", ".so")))
    except OSError:
        tiedostot = []
    if not tiedostot:
        return "nopeampi C++-versio: kaanna netti_fit, ks. README_netti.md"
    if isinstance(_netti_fit_virhe, ModuleNotFoundError):
        return (f"{tiedostot[0]} on kansiossa mutta ei sovi tahan Pythoniin ({sys.version_info.major}.{sys.version_info.minor}, tiedoston paatteen pitaisi olla "
                f"{sysconfig.get_config_var('EXT_SUFFIX')}): kaanna talle Pythonille (cmake ... -DPython_EXECUTABLE=<python.exe>) tai aja ohjelma silla Pythonilla, "
                f"jolle moduuli on kaannetty; ks. README_netti.md")
    return f"{tiedostot[0]} ei lataudu ({_netti_fit_virhe!r}); puuttuvatko OpenCV:n DLL-tiedostot moduulin vieresta? ks. README_netti.md"

KAHVA_Z = kv.KAHVA_Z_CM          # kahvan levyn korkeus (cm), jonka tasolle sentroidi heijastetaan
ETAISYYS_CM = rata.HOUSE_DISTANCE_CM    # pesien keskipisteiden etaisyys (T-viivojen valinen matka, cm)

# ---------------------------------------------------------------------------------------------------------------------
# PARAMETRIT
# ---------------------------------------------------------------------------------------------------------------------
HACK_X = (-75.0, 75.0)           # siemenalue (cm): kivi levossa hackissa
HACK_Y = (-370.0, -225.0)
KAHVA_ALA_PX = (50, 700)         # hyvaksyttava kahvan pinta-ala hack-alueella (keltainen ~150-260, punainen ~110-160)
LEPO_S = 0.5                     # kivi hack-alueella vahintaan tama ennen kuin se "viritetaan" (heittaja kasittelee kiveä, joten ei tiukkaa lepoa)
LEPO_CM = 8.0                    # viritys: sentroidin vaihtelu alle tama
LEPO_IKKUNA_S = 0.65             # viritys: vakaus tarkistetaan vain viimeisesta taman pituisesta ikkunasta (lista voi alkaa liikkeesta: kivi lasketaan hackiin, kantaminen)
LEPO_SIIRTYMA_CM = 5.0           # ... ja kivi ei saa siirtya ikkunassa enempaa (lepaava kivi 1-2 cm, kannettava / laskettava kivi enemman)
LIIKE_CM = 6.0                   # lahto: kolmen viimeisen havainnon mediaani yli tama lepopaikasta (lepopaikka = vanhempien havaintojen mediaani)
LEPO_N = 15                      # lepopaikan arvioon kaytetaan enintaan naita vanhempia havaintoja
HUKKA_LEPO_S = 2.0               # kahva poissa levossa >= tama -> takaisin IDLE:en (heittajan kasi peittaa kahvan hetkeksi; liian lyhyt aika pudotti kiven kesken valmistelun)
B_LEPO_POIK_CM = 12.0            # levossa (viritetty kahva) B kelpaa kiveksi vain jos se on enintaan tama kaukana kahvan (A) paikasta: kahva on kiven keskella (tosikivella mediaani 1-5 cm, p90 <= 9 cm, max 10,4 cm), kiven viereinen kenka / kasi on 12-27 cm paassa
B_LEPO_PEITTO = 0.5              # "kaukainen kivi": B-sovituksessa graniittia vahintaan tama, mutta yli B_LEPO_POIK_CM kahvan paikasta -> kahva ei ole kiven kahva
B_ULOTTUMA_PEITTO = 0.5          # lahdon jalkeen (heittoa ei viela vahvistettu): B kelpaa myos > B_MAX_POIKKEAMA_CM alkuarvosta, jos se on edellisen hyvaksytyn B:n ulottuvilla ja graniittia vahintaan taman verran
ULOTTUMA_V_CM_RUUTU = 8.0        # ... ulottuma = B_MAX_POIKKEAMA_CM + taman verran per ruutu edellisesta hyvaksytysta B:sta (kivi kiihtyy lahdossa jopa 8 cm / ruutu: 2,4 m/s)
ULOTTUMA_MAX_CM = 60.0
VAIHTO_S = 1.0                   # viritetyn, viela todentamattoman kahvan voi korvata toisella vakaalla kandidaatilla vasta taman ajan kuluttua (aika kahdelle B-sovitukselle)
VAIHTO_ETAISYYS_CM = 12.0        # ... ja saman varin kandidaatti on vahintaan taman verran viritetysta kahvasta
EI_KIVEA_B = 6                   # viritetty kahva hylataan ei-kivena, kun viimeiset taman monta B-sovitusta ovat selvasti ei-kivea (graniittia < EI_KIVEA_PEITTO) eika kivea ole todettu
EI_KIVEA_PEITTO = 0.25
EI_KIVEA_TAUKO_S = 1.0           # hylkayksen jalkeen IDLE-haku tauolla taman ajan (aiemmin 4 s: heitto ehti alkaa tauon aikana ja jai huomaamatta)
VAHVISTUS_Y_CM = -100.0          # heitto vahvistetaan kun Y yli taman ...
VAHVISTUS_V_CM_S = 120.0         # ... ja nopeus vahintaan tama (mitatuista paikoista; kavelija ei yllä)
HYLKAYS_AIKA_S = 10.0            # lahto, jonka jalkeen kivi ei vahvistu tassa ajassa -> hylataan (ylaraja)
EI_ETENE_S = 1.5                 # hylkays: tama aika liikkeessa ja liikealue alle EI_ETENE_CM -> kivea vain kasitellaan
EI_ETENE_CM = 12.0
LOPPU_Y_CM = 735.0               # kivi poistuu kuvasta (puolisko loppuu)
HUKKA_LIIKE_RUUTUA = 14          # liikkeessa: ei yhtaan hyvaksyttya mittausta tama monta yksilollista ruutua -> loppu
B_VAIN_KELV = 0.2                # B hyvaksytaan: kelvollisia pikseleita vahintaan
B_MAX_RMS = 0.30
B_MAX_POIKKEAMA_CM = 15.0        # B enintaan tama kaukana ennusteesta / A:sta
B_LEPO_VALI = 8                  # levossa B joka n:s yksilollinen ruutu
SOFT_KELV = 0.08                 # pehmea B (tiukka B ei kelpaa): kelvollisia vahintaan tama -> kelpaa seurantaan ja pisteisiin, ei estimaattiin
PEITTO_MIN = 0.35                # B kelpaa vain jos graniittia (alfa) on rungon siluetin sisalla vahintaan tama (oikea kivi ~0,4-0,95, ei-kivi ~0-0,2)
KIVI_B_LEPO = 2                  # ennen lahtoa hackissa on todettava kivi: vahintaan tama monta kelvollista B-sovitusta viimeisen 90 ruudun aikana
KIVI_B_VAHVISTUS = 8             # heiton vahvistus: kelvollisia B-sovituksia vahintaan tama monta liikkeen aikana
PAIKKA_PUUTE_RUUTUA = 36         # heitto paattyy kun yhtaan paikkaa (A / B) ei ole ollut taman monta ruutua
# vahvistetun heiton seuranta: B-pohjainen ennuste, B:n hyvaksyminen ennusteen suhteen, A:n (kahvan vari) harhojen torjunta. Punaisen kahvan blobiin tarttuu heittajan
# kasi / hanska (ala jopa 3-kertainen), jolloin A viivastyy kymmenia senttimetreja; ilman B-ankkuria ennuste ja seuranta hukkaavat kiven.
ENNUSTE_B_MIN = 4                # B-pohjainen ennuste: vahintaan nain monta B / pehmeaa B -paikkaa ...
ENNUSTE_B_IKKUNA = 30            # ... viimeisen taman ruutumaaran ajalta ...
ENNUSTE_B_TUORE = 16             # ... ja uusin niista enintaan taman ruutumaaran vanha
ENNUSTE_B_ROBUSTI_CM = 4.0       # B-suoran poikkeama, jonka yli B-paikka jaa pois suoran sovituksesta
A_HARHA_CM = 7.0                 # A (+ B-A -ero) poikkeaa B-ennusteesta yli taman: B:n alkuarvo on ennuste
B_HEIKKO_KELV = 0.02             # heikko B (vahvistettu heitto, ennuste B:sta): kelvollisia pikseleita vahintaan tama ...
B_HEIKKO_PEITTO = 0.5            # ... graniittia siluetin sisalla vahintaan tama ...
B_HEIKKO_RMS = 0.25
B_HEIKKO_POIK_CM = 8.0           # ... ja enintaan taman kaukana ennusteesta: kivi pysyy seurannassa pesan varillisilla renkailla (kelvollisia pikseleita vain 2-10 %)
ALA_VAIHTELU = (0.6, 1.6)        # kahvan ala (normitettu) paivittaa odotusarvoa vain tassa suhteessa mediaaniin (kasi ei paisuta odotusarvoa) ...
ALA_ULKONA_MAX = 6               # ... paitsi kun hyvaksytyt blobit ovat taman monta ruutua perakkain sen ulkopuolella (odotusarvo oli vaara): silloin odotusarvo vaihdetaan
HUKASSA_PORTTI_CM = 1.0          # A:n hakuportti kasvaa taman verran per ruutu, jolla paikkaa ei ole (enintaan HUKASSA_PORTTI_RUUTUA ruudun verran)
HUKASSA_PORTTI_RUUTUA = 15
IDLE_VALI = 3                    # IDLE: kandidaattihaku joka n:s yksilollinen ruutu
ESTIMAATTI_VALI = 10             # viivaestimaatti paivitetaan kun vahintaan n uutta kelvollista B-naytetta
ESTIMAATTI_MIN_Y_CM = 330.0      # vasta kun rata yltaa tahan (muuten ei irroitusta)
ESTIMAATTI = dict(min_jalkeen=70.0, Y_r_min=200.0, kohinakatto=(0.8, 0.15))   # osittaisen aineiston asetukset (liuku_irroitus.analysoi)
KOPIO_KYNNYS = 15.0              # kopioruutu: harmaasavyero (suurin 0,1 % pikseleista) alle taman (aja_kivi.KOPIO_KYNNYS)

# kierre, hog-viivat, numerointi
KIERTO_Y_MIN_CM = 200.0          # kierteen mittauksen ruudut (leikkeet) kerataan kun kiven Y on taman ja KIERTO_Y_MAX_CM:n valilla (irroitus on tavallisesti Y = 250-400)
KIERTO_Y_MAX_CM = 720.0
KIERTO_LEIKE_R = 72              # talteen otettavan leikkeen puolikas (px): kierre tarvitsee 56 px + varaa (paikka tarkentuu B:lla jalkikateen)
KIERTO_MAX_RUUTUJA = 160
KIVI_R_CM = 14.55                # kiven sade (suurin halkaisija 29,1 cm): etureunan etaisyys akselista
HOG_SISA_CM = 640.0              # hog-viivan sisareuna T-viivasta (kalibrointi_netti.HOG_Y_CM); viiva on n. 12,7 cm leveä
HOG_KESKI_CM = HOG_SISA_CM + 6.35   # hog-viivan keskiviiva: tama ylitys ajoitetaan molemmissa paissa (hog-hog-matka = ETAISYYS_CM - 2 * HOG_KESKI_CM = 21,8 m)
KIVIA_PAASSA = 8                 # numerointi alkaa alusta taman jalkeen (yhden joukkueen kivet / paa)
KAARTO_VARMA_1_CM = 4.0e-5       # vapaan lennon kaarevuus (1/cm), jonka yli kaarto ratkaisee kiertosuunnan jos kuvamittaus on epavarma
KAARTO_VIHJE_1_CM = 2.5e-5       # ... ja jonka yli kaarto kertoo suunnan kun kuvamittausta ei ole
# kaukopaan hog-viivan ylityksen haku (toisen puoliskon kuvasta)
SAAP_ALKU_S = 3.0                # haku alkaa taman verran heittopaan poistumisen jalkeen
SAAP_LOPPU_S = 45.0              # ... ja paattyy taman jalkeen
SAAP_X_CM = 300.0                # hakuvyohyke kaukopaassa: |X'| <= taman, Y' valilla
SAAP_Y_CM = (450.0, 800.0)
SAAP_PISTEITA = 6                # rata vahvistetaan kun vahintaan taman verran pisteita
SAAP_PEITTO_VARMA = 0.40         # kaukopaan rata kelpaa kiveksi, kun B-sovituksen peitto (graniittia siluetin sisalla) on radan viimeisessa pisteessa vahintaan tama ...
SAAP_PEITTO_MIN = 0.30           # ... tai kahdessa tarkistuksessa vahintaan tama; hylataan, kun SAAP_TARKISTUKSIA tarkistusta ei riita (kivi 0,34-0,71, ihmisen varusteet 0,06-0,28)
SAAP_TARKISTUKSIA = 3
SAAP_TARKISTUS_VALI = 2          # radan tarkistusten vali (ruutua)


class Kuvain:
    """Yhden puoliskon leike alkuperaisesta ruudusta suoraan oikaistuun (kierretty + linssikorjattu) kuvaan: yhdistetyt kartat."""

    def __init__(self, cam):
        x0, y0, w, h = kv.kn.PUOLET[cam.puoli]
        self.mx = (x0 + (w - 1) - cam.map2).astype(np.float32)       # oikaistun pikselin (u, v) lahde alkuperaisessa ruudussa
        self.my = (y0 + cam.map1).astype(np.float32)
        self.w, self.h = cam.w, cam.h

    def leike(self, ruutu, x0, y0, x1, y1):
        """Oikaistun kuvan alue [x0, x1) x [y0, y1) (rajattu kuvaan); palauttaa (kuva | None, x0, y0)."""
        x0, y0 = max(int(x0), 0), max(int(y0), 0)
        x1, y1 = min(int(x1), self.w), min(int(y1), self.h)
        if x1 - x0 < 8 or y1 - y0 < 8:
            return None, x0, y0
        return cv2.remap(ruutu, self.mx[y0:y1, x0:x1], self.my[y0:y1, x0:x1], cv2.INTER_LINEAR), x0, y0


def _leike_kamera(cam, x0, y0, w, h):
    """cam-kopio, jonka taustataulukot on rajattu leikkeeseen (maskit / alfa_kartta / havainto kayttavat vain taulukoita)."""
    jaa, vari = kt.tausta_maskit(cam)
    sl = (slice(y0, y0 + h), slice(x0, x0 + w))
    c = types.SimpleNamespace(mode=cam.mode[sl], mode_hsv=cam.mode_hsv[sl], lane=cam.lane[sl])
    c._tausta = (jaa[sl], vari[sl])
    c.maskit = types.MethodType(kv.Kamera.maskit, c)
    return c


def _lataa_kahva_asetukset(kansio, puoli):
    """Kahvan muovilevyn malli (kahva_malli.lataa_asetukset): puolen oma tiedosto, muuten vasemman (sama kahva) tai oletukset."""
    for p in (puoli, "vasen"):
        if os.path.exists(os.path.join(kansio, f"kahvamalli_asetukset_{p}.json")):
            return km.lataa_asetukset(kansio, p)
    return km.lataa_asetukset(None, puoli)


def _lataa_runko(kansio, puoli):
    """Kiven runkomalli (kivi_tarkka.Runko): puolen oma tiedosto, muuten vasemman (sama kivi, sama optiikka) tai oletusmalli."""
    for p in (puoli, "vasen"):
        if os.path.exists(os.path.join(kansio, f"kivimalli_{p}.json")):
            return kt.Runko.lataa(kansio, p)
    return kt.Runko()


class Mittari:
    """Yhden puoliskon mittaukset: kuva-alue, kahvan sentroidi (A) ja tarkka sovitus (B)."""

    def __init__(self, kansio, puoli, runko=True):
        self.puoli = puoli
        self.cam = kv.Kamera(puoli, os.path.join(kansio, f"kalibrointi_{puoli}.json"), netti_tausta.kaytettava(kansio, puoli))
        self.kuvain = Kuvain(self.cam)
        self.runko = _lataa_runko(kansio, puoli) if runko else None
        self.ast = _lataa_kahva_asetukset(kansio, puoli)            # kahvan muovilevyn malli (kierteen mittaus)
        self.kierto_muoto = nk.lataa_muoto(kansio, puoli)           # kahvan palkin muoto varille (kierteen mittaus)
        self._cpp = None
        if runko:
            kt.tausta_maskit(self.cam)                 # esilaskenta
            if _netti_fit is not None:
                try:
                    self._cpp = self._luo_cpp()
                except Exception as e:                  # esim. vaarat taulukot: Python-toteutus kayttoon
                    print(f"netti_fit (C++) ei kaytossa: {e!r}; B-sovitus Pythonilla", flush=True)

    def _luo_cpp(self):
        """C++-sovitin (netti_fit.BFit): taustakuva, kaistamaski, jaa- ja varimaskit, kamera ja kiven profiili kerran muistiin."""
        jaa, vari = kt.tausta_maskit(self.cam)
        r_prof, z_prof = self.runko.profiili()
        vakiot = dict(fg_kynnys=kv.FG_KYNNYS, g_vmin=kv.G_VMIN, g_vmax=kv.G_VMAX, g_smax=kv.G_SMAX, varjo=kv.VARJO_ALARAJA, graniitti=[float(x) for x in kv.GRANIITTI_BGR],
                      chr_max=kt.CHR_MAX, l_tumma=kt.L_TUMMA, reika_max_ala=kt.REIKA_MAX_ALA, dev_max_vari=kt.DEV_MAX_VARI)
        return _netti_fit.BFit(np.ascontiguousarray(self.cam.mode), np.ascontiguousarray(self.cam.lane), np.ascontiguousarray(jaa.astype(np.uint8)),
                               np.ascontiguousarray(vari.astype(np.uint8)), self.cam.K, self.cam.R, self.cam.t, r_prof, z_prof, self.runko.sigma_opt, self.runko.b_cm,
                               self.runko.gamma, vakiot)

    @property
    def sovitin(self):
        return "C++ (netti_fit)" if self._cpp is not None else "Python (kivi_tarkka)"

    # ---- koordinaatit (oikaistu kuva) ----------------------------------------------------------------------------
    def maa_oikaistuun(self, X, Y, z=KAHVA_Z):
        X = np.atleast_1d(np.asarray(X, float))
        u, v = kivimalli.project_3d(self.cam.K, self.cam.R, self.cam.t, np.column_stack([X, np.atleast_1d(np.asarray(Y, float)), np.full(len(X), float(z))]))
        return u, v

    def oikaistu_maahan(self, u, v, z=KAHVA_Z):
        return kv.maan_paikka(self.cam, u, v, z=z)

    def skaala(self, X, Y, z=KAHVA_Z):
        """Kuvamittakaava (px / cm) pisteessa (X, Y, z): sivusuunnassa ja pituussuunnassa (geometrinen keskiarvo)."""
        u, v = self.maa_oikaistuun([X, X + 5.0, X], [Y, Y, Y + 5.0], z)
        return float(np.sqrt(np.hypot(u[1] - u[0], v[1] - v[0]) * np.hypot(u[2] - u[0], v[2] - v[0])) / 5.0)

    def alue_maasta(self, X0, Y0, marg_px, r_cm=15.0, korkeus=KAHVA_Z):
        """Leikkeen rajat (x0, y0, x1, y1) kiven ymparille: kiven ala- ja ylaosan ympyrat + marginaali."""
        th = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        P = []
        for z in (0.0, korkeus + 4.0):
            P.append(np.column_stack([X0 + r_cm * np.cos(th), Y0 + r_cm * np.sin(th), np.full(12, z)]))
        u, v = kivimalli.project_3d(self.cam.K, self.cam.R, self.cam.t, np.vstack(P))
        return (int(np.floor(u.min() - marg_px)), int(np.floor(v.min() - marg_px)), int(np.ceil(u.max() + marg_px)), int(np.ceil(v.max() + marg_px)))

    def alue_kuvasta(self, u, v, r_px):
        return (int(u - r_px), int(v - r_px), int(u + r_px) + 1, int(v + r_px) + 1)

    # ---- A: kahvan sentroidi ------------------------------------------------------------------------------------
    def kahva(self, ruutu, alue, vari, ennuste_uv=None, max_etaisyys_px=45.0, min_ala=12, taustaero=False, kaikki=False):
        """Kahvan vari leikkeessa -> kandidaatit [(u, v, ala, bbox)] oikaistussa kuvassa (kokonaiskoordinaatit). Jos taustaero: vain taustasta poikkeavat
        pikselit (rengas ei yhdisty punaiseen kahvaan). Jos ennuste_uv: lahin kandidaatti ennusteeseen (<= max_etaisyys_px); kaikki: kaikki sen portin sisalla
        olevat kandidaatit lahimmasta alkaen."""
        roi, x0, y0 = self.kuvain.leike(ruutu, *alue)
        if roi is None:
            return []
        m = kv.kahva_maski(roi, vari)
        if taustaero:
            sl = (slice(y0, y0 + roi.shape[0]), slice(x0, x0 + roi.shape[1]))
            erot = np.max(np.abs(roi.astype(np.int16) - self.cam.mode[sl].astype(np.int16)), axis=2)
            m &= (erot > kv.FG_KYNNYS).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        n, lab, st, cen = cv2.connectedComponentsWithStats(m)
        out = []
        for k in range(1, n):
            if st[k, cv2.CC_STAT_AREA] < min_ala:
                continue
            u, v = cen[k][0] + x0, cen[k][1] + y0
            if ennuste_uv is not None and np.hypot(u - ennuste_uv[0], v - ennuste_uv[1]) > max_etaisyys_px:
                continue
            out.append((float(u), float(v), int(st[k, cv2.CC_STAT_AREA]), tuple(int(a) for a in st[k, :4])))
        if ennuste_uv is not None and out:
            out.sort(key=lambda c: np.hypot(c[0] - ennuste_uv[0], c[1] - ennuste_uv[1]))
            return out if kaikki else out[:1]
        return out

    # ---- B: tarkka sovitus ----------------------------------------------------------------------------------------
    def valmistele(self, ruutu, X0, Y0, marg_px=36):
        """B:n leike alkuarvon ymparilta: (roi, x0, y0) tai None. Vain tama siirtyy tyoprosessille (ei koko ruutua)."""
        alue = self.alue_maasta(X0, Y0, marg_px, r_cm=self.runko.R_max + 12.0)
        roi, x0, y0 = self.kuvain.leike(ruutu, *alue)
        return None if roi is None else (roi, x0, y0)

    def sovita_leike(self, roi, x0, y0, X0, Y0):
        """Kiven akseli (X, Y) leikkeesta (alkuarvo X0, Y0): dict(X, Y, rms, kelvolliset, peitto). Kaksi kierrosta (kaista 8 -> 5 px). C++ (netti_fit), jos kaytossa."""
        if self._cpp is not None:
            return self._cpp.sovita(roi, int(x0), int(y0), float(X0), float(Y0))
        return self.sovita_leike_python(roi, x0, y0, X0, Y0)

    def sovita_leike_python(self, roi, x0, y0, X0, Y0):
        h, w = roi.shape[:2]
        c = _leike_kamera(self.cam, x0, y0, w, h)
        fg, gran, handle = c.maskit(roi)
        alfa, ok = kt.havainto(c, roi, kv.alfa_kartta(c, roi, gran))
        s = kt.sovita(self.cam, self.runko, alfa, ok, X0, Y0, kaarros=(8, 5), ox=x0, oy=y0)
        # peitto: graniitin (alfa) keskiarvo sovitetun rungon siluetin sisalla (ei-kivi: ~0; kivi ~0,6-0,95, kasi voi peittaa osan)
        mc, _ = kt.malli_kuva(self.cam, self.runko, (x0, y0, w, h), s["X"], s["Y"])
        sis = mc > 0.5
        s["peitto"] = float(alfa[sis].mean()) if sis.any() else 0.0
        return s

    def tarkka(self, ruutu, X0, Y0):
        v = self.valmistele(ruutu, X0, Y0)
        return None if v is None else self.sovita_leike(*v, X0, Y0)


# ---------------------------------------------------------------------------------------------------------------------
# TYOPROSESSIT: B-taso ja viivaestimaatti
# ---------------------------------------------------------------------------------------------------------------------
_M = {}


def matala_prioriteetti():
    """Taustalaskennan prosessit alemmalle prioriteetille: analyysi-, piirto- ja ikkunasaikeet saavat prosessorin ensin kun ytimia on vahan (taustatyo vain viivastyy,
    eika analyysi jaa jalkeen ja ohita ruutuja)."""
    try:
        if os.name == "nt":
            import ctypes
            ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)       # BELOW_NORMAL_PRIORITY_CLASS
        else:
            os.nice(5)
    except Exception:
        pass


def alusta_tyo(kansio, puolet):
    matala_prioriteetti()
    cv2.setNumThreads(1)                                # pieniä kuvia: OpenCV:n oma saiepooli vain hidastaisi, kun tyoprosesseja on useita
    for p in puolet:
        _M[p] = Mittari(kansio, p)


def tyo_b(args):
    """B-tehtava: (avain, puoli, roi, x0, y0, X0, Y0) -> (avain, tulos)."""
    avain, puoli, roi, x0, y0, X0, Y0 = args
    try:
        return avain, _M[puoli].sovita_leike(roi, x0, y0, X0, Y0)
    except Exception as e:                              # yksittainen ruutu ei saa kaataa seurantaa
        return avain, dict(virhe=repr(e))


def tyo_nop():
    """Tyoprosessin kayntiin saattaminen etukateen (lampea): ei tee mitaan."""
    return True


def tee_pool(n, **kw):
    """Tyoprosessipooli 'spawn'-tavalla kaikilla alustoilla: fork kopioi monisaikeisen prosessin lukot (OpenCV:n saiepooli, putkien paat) ja voi jumittaa tyoprosessin tai
    pitaa ffmpeg-putken auki; spawn kayttaytyy kuten Windowsissa (oletus siella)."""
    return ProcessPoolExecutor(n, mp_context=multiprocessing.get_context("spawn"), **kw)


_MK = {}


def alusta_kierto(kansio, puolet):
    """Kierteen mittauksen tyoprosessi: vain kamerat (ei kivimallia eika taustamaskeja)."""
    matala_prioriteetti()
    cv2.setNumThreads(1)
    for p in puolet:
        _MK[p] = Mittari(kansio, p, runko=False)


def kierto_laske(m, args):
    """Kierteen mittaus: args = (puoli, vari, havainnot, t_irr), m = puolen Mittari. Palauttaa netti_kierre.arvioi:n dictin (tai dict(virhe=...))."""
    puoli, vari, havainnot, t_irr = args
    try:
        return nk.arvioi(m.cam, m.ast, vari, havainnot, t_irr, m.kierto_muoto.get(vari))
    except Exception as e:
        return dict(virhe=repr(e))


def tyo_kierto(args):
    return kierto_laske(_MK[args[0]], args)


def laske_viivat(f, X, Y, fps_lahde=25.0, fps_ulos=30.0, lisa_f=None):
    """Viivaestimaatti kelvollisista (ruutu, X, Y) -naytteista: aikaperusta (aikaperusta.Aikajana) + liu'un ja irroituksen tunnistus (liuku_irroitus). Palauttaa
    dict tai None. Kaytetaan sellaisenaan tyoprosessissa. fps_lahde = erillisten ruutujen nimellistaajuus (25 tassa videossa: 25 r/s -> 30 r/s kopioruuduilla),
    fps_ulos = videon ruutunopeus. lisa_f: muut ruutunumerot (kierteen mittauksen ruudut), joille palautetaan sama aika (t_lisa)."""
    aikap.FPS_LAHDE, aikap.FPS_ULOS = float(fps_lahde), float(fps_ulos)
    f = np.asarray(f, float)
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    o = np.argsort(f)
    f, X, Y = f[o], X[o], Y[o]
    if len(f) < 30 or Y.max() < ESTIMAATTI_MIN_Y_CM:
        return None
    try:
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            aj = aikap.Aikajana(f, Y)
            mk = np.array([int(fi) not in aj.kopio for fi in f])
            f, X, Y = f[mk], X[mk], Y[mk]
            t = aj.aika(f)
            t_lisa = [float(x) for x in aj.aika(np.asarray(lisa_f, float))] if lisa_f is not None and len(lisa_f) else None
            r = li.analysoi(dict(t=t, X=X, Y=Y), **ESTIMAATTI)
    except Exception as e:                              # liian vahva / outo aineisto: ei estimaattia
        return dict(virhe=repr(e))
    if r.get("irroitus") is None or r.get("tila") is None:
        return None
    ti, ys, la, ir = r["tila"], r["yhteenveto"], r["liuku"], r["irroitus"]
    vapaa_Y = float(Y.max() - ti["Y_cm"])
    return dict(
        liuku_a=ys["liun_suora_X_Y0_cm"], liuku_b=ys["liun_suora_kulmakerroin"], liuku_suunta_deg=ys["liun_suunta_deg"],
        liuku_Y0=ys["liun_suunta_alku_cm"], liuku_Y1=ys["liun_suunta_loppu_cm"],
        irr_X=ti["X_cm"], irr_Y=ti["Y_cm"], irr_suunta_deg=ti["suunta_deg"], irr_virhe_deg=ti["suunta_virhe_deg"],
        irr_Y_vali=(ir["Y_min"], ir["Y_max"]), merkitseva=bool(ir["merkitseva"]), dchi2=float(ir["dchi2_med"]),
        vapaa_matka_cm=vapaa_Y, vapaa_ruutuja=int(ti["ruutuja"]), Y_max=float(Y.max()), t_lahto=float(la["t_lahto"]),
        nopeus_ennen=ys["nopeus_ennen_m_s"], nopeus_jalkeen=ys["nopeus_jalkeen_m_s"],
        liuku_hidastuvuus=ys["liun_hidastuvuus_m_s2"], vapaa_hidastuvuus=ys["vapaan_lennon_hidastuvuus_m_s2"],
        kaarto=ti["kaarto"], kaarevuus_1_cm=float(ti["kaarevuus_1_cm"]), t_irr=float(ti["t_r"]), t_lisa=t_lisa, n=int(len(f)))


# ---------------------------------------------------------------------------------------------------------------------
# HEITTO JA HEITTOPAA
# ---------------------------------------------------------------------------------------------------------------------
class Naytteet:
    """Yksi yksilollinen ruutu: A (kahvan sentroidi) ja B (tarkka sovitus), maakoordinaatit cm."""
    __slots__ = ("f", "A", "ala", "B", "Bs", "rms", "kelv", "peitto", "odottaa", "poik", "lepo")

    def __init__(self, f, A=None, ala=0):
        self.f, self.A, self.ala = f, A, ala
        self.B, self.Bs, self.rms, self.kelv, self.peitto, self.odottaa = None, None, None, None, None, False
        self.poik = None                       # B:n etaisyys alkuarvosta (A / ennuste), cm
        self.lepo = False                      # levossa viritetylle kahvalle pyydetty B (tiukempi kiven hyvaksynta: B_LEPO_POIK_CM)

    def paikka(self):
        """Paras tunnettu paikka: B, muuten pehmea B (kelvollisia >= SOFT_KELV), muuten A."""
        if self.B is not None:
            return self.B
        if self.Bs is not None:
            return self.Bs
        return self.A

    def kivi(self):
        """Todettu kivi: B tai pehmea B kelvollinen (graniittia siluetin sisalla)."""
        return self.B is not None or self.Bs is not None


class Heitto:
    def __init__(self, tunnus, puoli, vari, f_lepo, lepo_XY):
        self.tunnus, self.puoli, self.vari = tunnus, puoli, vari
        self.f_lepo, self.lepo = f_lepo, np.array(lepo_XY, float)
        self.tila = "liikkuu"                  # liikkuu | vahvistettu | valmis | hylatty
        self.naytteet = []
        self.f_alku = None                     # lahtoruutu (taaksepain ajoitettu)
        self.f_loppu = None                    # viimeinen seurattu ruutu
        self.validoitu = False                 # viritetty kahva todettu kiven kahvaksi (Heittopaa._validoitu): vasta sitten kahvan liike on heiton lahto
        self.loppu_Y = None                    # diagnostiikka: viimeinen mitattu Y (cm) ja heiton paattymisen syy ("matka" = kivi poistui puoliskolta, "kivi hukassa")
        self.loppu_syy = ""
        self.viivat = None                     # viimeisin viivaestimaatti (laske_viivat) tai None
        self.viivat_f = None                   # ruutu, jonka aineistolla estimaatti on laskettu
        self.viivat_n = 0                      # estimaatin kelvollisten B-naytteiden maara
        self.viivat_hist = []                  # viimeisimpien estimaattien irroituspaikat (vakauden arvio)
        self.viivat_loki = []                  # kaikki estimaatit: (analyysin ruutu, irroitus Y, suunta, T-viiva irroitus, T-viiva liuku, vakaa)
        self.f_vahvistus = None
        self.lopullinen = False                # lopullinen estimaatti pyydetty (heitto paattynyt, kaikki aineisto)
        self.viivat_lopullinen = False         # lopullinen estimaatti valmis (tulos tai "ei estimaattia")
        self.lopullinen_jonossa = False        # lopullinen estimaatti odottaa kesken olevan estimaatin valmistumista
        self.kivi_nro = None                   # kiven numero vareittain per heittosuunta (oletus: heitetaan numerojarjestyksessa)
        self.b_laskuri = 0                     # B-tulosten maara (piirtajan valimuistin avain)
        self.kierto_kehykset = []              # kierteen mittauksen ruudut: [(Naytteet, leike, x0, y0)]
        self.kierto_lisa = None                # estimaatille annetut ruudut (aikojen t_lisa jarjestys)
        self.kierto = None                     # kierteen mittaus (netti_kierre.arvioi) tai dict(virhe=...)
        self.kierto_tila = None                # None | "kaynnissa" | "valmis" | "virhe"
        self.hog1_f = None                     # ruutu (murtoluku), jolla kivi ylitti heittopaan hog-viivan (vasemman puoliskon rata)
        self.hog2_f = None                     # ruutu, jolla kivi ylitti kaukopaan hog-viivan (oikean puoliskon kuva)
        self.saapuminen = None                 # kaukopaan haku: None | "haku" | "loytyi" | "ei_loytynyt"
        self.kierre = None                     # kierrosmaara hog-hog-valilla (_paivita_kierre)
        self.ala_med = None                    # kahvan normitettu ala (px / skaala^2): kaukopaan blobien tunnistus

    def kelvolliset_b(self):
        return [n for n in self.naytteet if n.B is not None]

    def ylitys_f(self, Y_raja, ikkuna_cm=150.0):
        """Ruutu (murtoluku), jolla kiven akseli ylittaa Y = Y_raja: lineaarinen sovitus ruutunumero(Y) rajan ymparilta (B ja pehmea B; jos niita on liian vahan,
        myos A). None jos rata ei ylita rajaa tai pisteita on liian vahan."""
        ns = list(self.naytteet)
        for vain_b in (True, False):
            pts = [(n.f, n.paikka()[1]) for n in ns if n.paikka() is not None and (not vain_b or n.kivi())]
            pts = [(f, y) for f, y in pts if abs(y - Y_raja) <= ikkuna_cm]
            if len(pts) >= 5 and min(y for _, y in pts) < Y_raja < max(y for _, y in pts):
                F = np.array([p[0] for p in pts], float)
                Yv = np.array([p[1] for p in pts], float)
                return float(np.polyval(np.polyfit(Yv, F, 1), Y_raja))
        return None

    def kivi_n(self, viim_ruutua=90, f=None):
        """Kelvollisten (kivi todettu) B-sovitusten maara viimeisten viim_ruutua ruudun aikana (f: nykyinen ruutu; oletus viimeisin nayte)."""
        if not self.naytteet:
            return 0
        f = self.naytteet[-1].f if f is None else f
        return sum(1 for n in self.naytteet[-200:] if f - n.f <= viim_ruutua and n.kivi())

    def pisteet(self, B_VALI=4, EROTUS_IKKUNA=20):
        """[(f, X, Y)] piirrettavat pisteet lahtoruudusta alkaen: B / pehmea B. A (kahvan vari) vain kun B-paikkaa ei ole B_VALI ruudun sisalla: pesassa heittajan kasi
        vetaa punaisen kahvan blobia kymmenia senttimetreja, joten A-pisteet B-paikkojen vieressa jatetaan pois (aukko tayttyy piirtajalla interpoloimalla). Yksinaiset
        A-pisteet korjataan lahialueen B - A -erolla (EROTUS_IKKUNA ruutua, mediaani, vahintaan 3 paria)."""
        ns = [n for n in list(self.naytteet) if self.f_alku is not None and n.f >= self.f_alku and n.paikka() is not None]
        fb = np.array([n.f for n in ns if n.kivi()], float)
        pari_f = np.array([n.f for n in ns if n.B is not None and n.A is not None], float)
        pari_d = np.array([np.subtract(n.B, n.A) for n in ns if n.B is not None and n.A is not None], float).reshape(-1, 2)
        out = []
        for n in ns:
            if n.kivi():
                p = n.paikka()
            else:
                if len(fb) and float(np.min(np.abs(fb - n.f))) <= B_VALI:
                    continue
                p = np.asarray(n.A, float)
                if len(pari_f):
                    lahella = np.abs(pari_f - n.f) <= EROTUS_IKKUNA
                    if int(lahella.sum()) >= 3:
                        p = p + np.median(pari_d[lahella], axis=0)
            out.append((n.f, float(p[0]), float(p[1])))
        return out


class Heittopaa:
    """Yhden pelipaan (vasen / oikea) heittojen tunnistus ja seuranta."""

    def __init__(self, analyysi, puoli):
        self.an = analyysi
        self.puoli = puoli
        self.m = analyysi.M[puoli]
        self.tila = "IDLE"
        self.heitto = None
        self.kandidaatit = {}                   # IDLE: vari -> lista (f, X, Y, u, v, ala)
        self.lepo = []                          # ARMED: viimeisimmat A-havainnot (f, X, Y, u, v, ala)
        self.f_nahty = None                     # viimeisin ruutu, jossa kahva nakyi (ARMED)
        self.vari = None
        self.n_ruutu = 0
        self.hukka = 0
        self.ala_norm = []                      # hyvaksyttyjen kahvojen ala / skaala^2 (viimeiset 8): odotetun alan mediaani
        self.f_liike = None
        self.edellinen_loppu_f = -10 ** 9
        self.ei_ennen_f = -10 ** 9               # IDLE-haku tauolla tahan ruutuun asti (kivettomat kandidaatit)
        self._ennuste_luotettava = False         # viimeisin ennuste oli B-pohjainen (_ennuste_b)
        self._ala_ulkona = []                    # perakkaiset hyvaksytyt blobit, joiden ala oli ALA_VAIHTELU:n ulkopuolella (normitettu)
        P = np.array([(x, y, z) for x in HACK_X for y in HACK_Y for z in (0.0, 20.0)])
        u, v = kivimalli.project_3d(self.m.cam.K, self.m.cam.R, self.m.cam.t, P)
        self.hack_alue = (int(u.min()) - 6, int(v.min()) - 6, int(u.max()) + 6, int(v.max()) + 6)

    # ---- apurit --------------------------------------------------------------------------------------------------
    def _a_maa(self, c):
        return self.m.oikaistu_maahan(c[0], c[1])

    @staticmethod
    def _hack_sisalla(X, Y):
        return HACK_X[0] <= X <= HACK_X[1] and HACK_Y[0] <= Y <= HACK_Y[1]

    def _konsol(self, n, o):
        """Konsolidoitu paikka seurantaan: B / pehmea B sellaisenaan, A korjattuna B-kehykseen (A - o)."""
        if n.kivi():
            return np.array(n.paikka(), float)
        if n.A is not None:
            return np.array(n.A, float) + o
        return None

    def _ennuste_b(self, f):
        """B-pohjainen ennuste vahvistetulle heitolle: B / pehmean B -paikkojen robusti suora ruutuindeksin funktiona (vakionopeus). None, jos heitto ei ole vahvistettu tai
        B-paikkoja ei ole tarpeeksi tuoreesti. A:ta (kahvan vari) ei kayteta: heittajan kasi ja hanska vetavat punaisen kahvan blobia kymmenia senttimetreja."""
        h = self.heitto
        if h.tila != "vahvistettu":
            return None
        pts = []
        for n in reversed(h.naytteet):
            if f - n.f > ENNUSTE_B_IKKUNA:
                break
            if n.kivi():
                pts.append((n.f, n.paikka()))
            if len(pts) >= 8:
                break
        if len(pts) < ENNUSTE_B_MIN or f - pts[0][0] > ENNUSTE_B_TUORE:
            return None
        ff = np.array([p[0] for p in pts], float)
        Q = np.array([p[1] for p in pts], float)

        def sovita(ok):
            fm = ff[ok].mean()
            c = np.linalg.lstsq(np.column_stack([np.ones(int(ok.sum())), ff[ok] - fm]), Q[ok], rcond=None)[0]      # c[0]: paikka fm:ssa, c[1]: nopeus / ruutu
            return c, fm

        ok = np.ones(len(ff), bool)
        for _ in range(3):
            c, fm = sovita(ok)
            uusi = np.hypot(*(Q - (c[0] + np.outer(ff - fm, c[1]))).T) <= ENNUSTE_B_ROBUSTI_CM
            if uusi.sum() < ENNUSTE_B_MIN or bool((uusi == ok).all()):
                break
            ok = uusi
        else:
            c, fm = sovita(ok)
        p = c[0] + c[1] * (f - fm)
        return float(p[0]), float(p[1])

    def _ennuste(self, f):
        """Ennuste (X, Y) ruudulle f: vahvistetulla heitolla B-pohjainen (_ennuste_b; self._ennuste_luotettava), muuten viimeisten (<= 7) konsolidoidun paikan suora
        ruutuindeksin funktiona (vakionopeus)."""
        pb = self._ennuste_b(f)
        self._ennuste_luotettava = pb is not None
        if pb is not None:
            return pb
        o = self._offset_b()
        pts = []
        for n in reversed(self.heitto.naytteet):
            if f - n.f > 60:
                break
            q = self._konsol(n, o)
            if q is not None:
                pts.append((n.f, q))
            if len(pts) >= 7:
                break
        if not pts:
            return tuple(self.heitto.lepo)
        pts.reverse()
        if len(pts) < 3:
            fb, qb = pts[-1]
            if len(pts) == 2 and pts[-1][0] > pts[0][0]:
                v = (pts[-1][1] - pts[0][1]) / (pts[-1][0] - pts[0][0])
                return tuple(qb + v * (f - fb))
            return tuple(qb)
        ff = np.array([p[0] for p in pts], float)
        Q = np.array([p[1] for p in pts], float)
        fm = ff.mean()
        A = np.column_stack([np.ones(len(ff)), ff - fm])
        c = np.linalg.lstsq(A, Q, rcond=None)[0]
        p = c[0] + c[1] * (f - fm)
        return float(p[0]), float(p[1])

    def _offset_b(self):
        """B - A -erotus viimeisista naytteista, joissa molemmat tunnetaan (A:n systemaattinen poikkeama): mediaani."""
        d = [np.array(n.B) - np.array(n.A) for n in self.heitto.naytteet[-12:] if n.B is not None and n.A is not None]
        return np.median(np.array(d), axis=0) if len(d) >= 3 else np.zeros(2)

    def _nopeus_ruutu(self):
        """Nopeus [cm / ruutu] viimeisten konsolidoitujen paikkojen ensimmaisen ja viimeisen valilla."""
        o = self._offset_b()
        pts = []
        for n in reversed(self.heitto.naytteet):
            q = self._konsol(n, o)
            if q is not None:
                pts.append((n.f, q))
            if len(pts) >= 6:
                break
        if len(pts) < 3:
            return 0.0
        (fb, qb), (fa, qa) = pts[0], pts[-1]
        return float(np.hypot(qb[0] - qa[0], qb[1] - qa[1])) / max(fb - fa, 1)

    def _vari_a(self, ruutu, ennuste_XY, gate_cm, tarkka_ala=True, ala_raja=(0.4, 2.2)):
        """A-mittaus: kahva ennusteen ymparilta; hyvaksytaan ennusteeseen lahin alaltaan (ala_raja * odotettu ala) ja sijainniltaan sopiva blobi. Palauttaa (X, Y, ala,
        skaala) tai None."""
        Xp, Yp = ennuste_XY
        up, vp = self.m.maa_oikaistuun(Xp, Yp)
        s = self.m.skaala(Xp, Yp)
        r_px = float(np.clip(45.0 + 0.25 * gate_cm * s, 45.0, 140.0))
        blobit = self.m.kahva(ruutu, self.m.alue_kuvasta(up[0], vp[0], r_px), self.vari, (up[0], vp[0]), max_etaisyys_px=gate_cm * s,
                              taustaero=(self.vari == "punainen" and abs(Yp) < 110.0), kaikki=True)
        for u, v, ala, bb in blobit:
            if tarkka_ala and self.ala_norm:
                odotus = float(np.median(self.ala_norm)) * s * s
                if not (ala_raja[0] * odotus <= ala <= ala_raja[1] * odotus):
                    continue
            X, Y = self._a_maa((u, v))
            return X, Y, ala, s
        return None

    def _idle_alkuun(self):
        self.tila = "IDLE"
        self.heitto = None
        self.kandidaatit = {}
        self.lepo = []

    # ---- IDLE: kandidaattihaku ----------------------------------------------------------------------------------
    def _idle(self, f, ruutu):
        if self.n_ruutu % IDLE_VALI:
            return
        k = self._vakaa_kandidaatti(f, ruutu)
        if k is not None:
            self._virita_kandidaatista(f, ruutu, *k)

    def _vakaa_kandidaatti(self, f, ruutu, pois=None):
        """Kandidaattihaku hack-alueelta (kahvan vari, molemmat varit); palauttaa (vari, lista, (X, Y)) ensimmaiselle vakaalle kandidaatille (keltainen ensin) tai None.
        pois = (vari, X, Y): taman varin ehdokkaat enintaan VAIHTO_ETAISYYS_CM paassa (X, Y):sta ohitetaan (jo viritetty kahva)."""
        loytyi = {}
        for vari in ("keltainen", "punainen"):
            for c in self.m.kahva(ruutu, self.hack_alue, vari, None, min_ala=KAHVA_ALA_PX[0] // 2, taustaero=False):
                if not (KAHVA_ALA_PX[0] <= c[2] <= KAHVA_ALA_PX[1]):
                    continue
                bb = c[3]
                if not (0.35 <= bb[2] / max(bb[3], 1) <= 2.8):
                    continue
                X, Y = self._a_maa(c)
                if not self._hack_sisalla(X, Y):
                    continue
                if pois is not None and vari == pois[0] and np.hypot(X - pois[1], Y - pois[2]) <= VAIHTO_ETAISYYS_CM:
                    continue
                loytyi.setdefault(vari, []).append((f, X, Y, c[0], c[1], c[2]))
        for vari in ("keltainen", "punainen"):
            lista = self.kandidaatit.setdefault(vari, [])
            uusi = loytyi.get(vari, [])
            # jatka kandidaattia jos sama paikka (<= LEPO_CM) kuin viimeisin, muuten aloita alusta
            if uusi and lista and min(np.hypot(c[1] - lista[-1][1], c[2] - lista[-1][2]) for c in uusi) <= LEPO_CM:
                lista.append(min(uusi, key=lambda c: np.hypot(c[1] - lista[-1][1], c[2] - lista[-1][2])))
            elif uusi:
                self.kandidaatit[vari] = [max(uusi, key=lambda c: c[5])]
            else:
                self.kandidaatit[vari] = []
                continue
            lista = self.kandidaatit[vari]
            if len(lista) > 60:
                del lista[:-60]
            # vakaus vain viimeisesta ikkunasta: jokainen <= LEPO_CM:n siirtyma edelliseen pitaa listan elossa, joten lista voi alkaa liikkeesta (kivi lasketaan hackiin) ja koko
            # listan vakaus ei koskaan toteutuisi: paikallaan lepaava kivi jai virittamatta ja heitto havisi (video 25 min: 160,6 s)
            ikk = [c for c in lista if lista[-1][0] - c[0] <= LEPO_IKKUNA_S * self.an.fps]
            if (ikk[-1][0] - ikk[0][0]) / self.an.fps >= LEPO_S and len(ikk) >= 4:
                X = float(np.median([c[1] for c in ikk]))
                Y = float(np.median([c[2] for c in ikk]))
                if max(np.hypot(c[1] - X, c[2] - Y) for c in ikk) <= LEPO_CM and np.hypot(ikk[-1][1] - ikk[0][1], ikk[-1][2] - ikk[0][2]) <= LEPO_SIIRTYMA_CM:
                    return vari, ikk, (X, Y)
        return None

    def _virita_kandidaatista(self, f, ruutu, vari, ikk, XY):
        self._virita(f, vari, ikk, XY)
        n = Naytteet(f, (ikk[-1][1], ikk[-1][2]), ikk[-1][5])
        n.lepo = True
        self.heitto.naytteet.append(n)
        self.an.pyyda_b(self, self.heitto, n, ruutu, XY[0], XY[1])

    def _virita(self, f, vari, lista, XY):
        self.tila = "ARMED"
        self.vari = vari
        self.lepo = list(lista)[-40:]
        self.f_nahty = f
        sk = self.m.skaala(XY[0], XY[1])
        self.ala_norm = [c[5] / (sk * sk) for c in lista][-8:]
        self._ala_ulkona = []
        self.kandidaatit = {}
        self.hukka = 0
        self.heitto = Heitto(self.an.uusi_tunnus(), self.puoli, vari, lista[0][0], XY)
        self.an.tapahtumat.append(("viritetty", self.heitto.tunnus, f, f"{self.puoli} {vari} ({XY[0]:.0f},{XY[1]:.0f})"))

    # ---- ARMED: kivi hack-alueella, odotetaan lahtoa --------------------------------------------------------------
    def _lepopaikka(self):
        """Lepopaikka = vanhempien havaintojen mediaani (kolme viimeisinta pois, jotta lahto ei vedä sitä mukanaan)."""
        vanh = self.lepo[:-3][-LEPO_N:] if len(self.lepo) > 8 else self.lepo[:max(len(self.lepo) - 3, 1)]
        return float(np.median([c[1] for c in vanh])), float(np.median([c[2] for c in vanh]))

    @staticmethod
    def _lepo_todisteet(h, f):
        """(sop, kauk): levossa viritetyn kahvan B-sovitukset, joissa kivi on kahvan kohdalla (sop; <= B_LEPO_POIK_CM) tai graniittia on mutta kahvan paikasta kauempana (kauk)."""
        sop = h.kivi_n(10 ** 6, f)
        kauk = sum(1 for n in h.naytteet[-300:] if n.lepo and n.peitto is not None and n.peitto >= B_LEPO_PEITTO and not n.kivi() and n.poik is not None and n.poik > B_LEPO_POIK_CM)
        return sop, kauk

    def _validoitu(self, h, f):
        """Viritetty kahva on kiven kahva: levossa vahintaan KIVI_B_LEPO kelvollista B-sovitusta kahvan kohdalla (<= B_LEPO_POIK_CM), eika kahvan vierestä ole löytynyt kiveä
        useammin kuin kahvan kohdalta (kiven viereinen esine, esim. heittäjän kenkä: B-sovitus osuu viereiseen kiveen 12-27 cm päähän; sen takia punainen kenkä keltaisen kiven
        vieressä viritettiin ja esti keltaisen heiton)."""
        sop, kauk = self._lepo_todisteet(h, f)
        return sop >= KIVI_B_LEPO and kauk <= sop

    def _vaihda_kandidaatti(self, f, ruutu):
        """Viritetty kahva ei ole (viela) todettu kiveksi, mutta toinen kandidaatti (toinen vari tai toinen paikka hack-alueella) on ollut vakaana: virityksen kohde vaihtuu."""
        X0, Y0 = self.heitto.lepo
        sop, kauk = self._lepo_todisteet(self.heitto, f)
        if kauk < sop:                                                  # kahvan kohdalla on kivea useammin kuin vieressa: oikea kivi, jonka B-sovitukset vain viipyvat (kasi peittaa)
            return False
        k = self._vakaa_kandidaatti(f, ruutu, (self.vari, X0, Y0))
        if k is None:
            return False
        self.an.tapahtumat.append(("vaihto", self.heitto.tunnus, f, f"{self.vari} ({X0:.0f},{Y0:.0f}) -> {k[0]} ({k[2][0]:.0f},{k[2][1]:.0f})"))
        self._virita_kandidaatista(f, ruutu, *k)
        return True

    def _armed(self, f, ruutu):
        h = self.heitto
        if not h.validoitu and self._validoitu(h, f):
            h.validoitu = True
        if (not h.validoitu and self.n_ruutu % IDLE_VALI == 0 and (f - h.naytteet[0].f) / self.an.fps >= VAIHTO_S
                and self._vaihda_kandidaatti(f, ruutu)):
            return
        X0, Y0 = h.lepo
        r = self._vari_a(ruutu, (X0, Y0), 35.0, tarkka_ala=False)
        if r is None:
            if (f - self.f_nahty) / self.an.fps >= HUKKA_LEPO_S:
                self._idle_alkuun()
            return
        X, Y, ala, s = r
        self.f_nahty = f
        u, v = self.m.maa_oikaistuun(X, Y)
        self.lepo = (self.lepo + [(f, X, Y, float(u[0]), float(v[0]), ala)])[-40:]
        n = Naytteet(f, (X, Y), ala)
        n.lepo = True
        h.naytteet.append(n)
        if len(h.naytteet) > 400:
            del h.naytteet[:100]
        if self.n_ruutu % B_LEPO_VALI == 0:                     # levossa B harvaan (tarkka lepopaikka estimaatin alkuun)
            self.an.pyyda_b(self, h, n, ruutu, X, Y)
        if len(self.lepo) < 6:
            return
        h.lepo = self._lepopaikka()
        X0, Y0 = h.lepo
        self.ala_norm = [q[5] / (s * s) for q in self.lepo[-8:]]
        viim = self.lepo[-3:]
        if np.hypot(np.median([q[1] for q in viim]) - X0, np.median([q[2] for q in viim]) - Y0) >= LIIKE_CM:
            if h.validoitu:                                     # kivi todettu viritetyn jakson aikana (ei vain viimeisen 3 s: kahva voi olla kaden alla, jolloin B ei onnistu hetkeen)
                self._lahto(f)
        elif self.an.b_valmis and self.n_ruutu % B_LEPO_VALI == 0 and self._ei_kivea(h, f):
            # kahva ilman kivea (esim. harjan paa, vaatetus): ei viritetty; haku jatkuu vasta hetken kuluttua
            self.an.tapahtumat.append(("ei_kivea", h.tunnus, f, f"{self.puoli} {self.vari}"))
            self.ei_ennen_f = f + int(EI_KIVEA_TAUKO_S * self.an.fps)
            self._idle_alkuun()

    def _ei_kivea(self, h, f):
        """Viritetty kahva ei ole kivi (esim. harjan paa): viime aikoina (~7 s) ei KIVI_B_LEPO kelvollista kivisovitusta, ja joko viritetysta on yli 3 s tai viimeiset EI_KIVEA_B
        sovitusta ovat selvasti ei-kivea (harjan paa: graniittia 0,02-0,07). Kivi, joka on todettu kerran, ei pudoteta pelkan kaden peittaman kahvan vuoksi."""
        if h.validoitu:
            return False
        if (f - h.naytteet[0].f) / self.an.fps > 3.0:
            return True
        tulokset = [n.peitto for n in h.naytteet[-80:] if n.peitto is not None]
        return len(tulokset) >= EI_KIVEA_B and all(p < EI_KIVEA_PEITTO for p in tulokset[-EI_KIVEA_B:])

    def _lahto(self, f):
        """Kivi lahti: LIIKKUU. Lahtoruutu ajoitetaan taaksepain: lepopaikasta poikenneiden havaintojen jakson ensimmainen ruutu."""
        h = self.heitto
        X0, Y0 = h.lepo
        h.tila = "liikkuu"
        self.tila = "LIIKKUU"
        self.f_liike = f
        self.hukka = 0
        i = len(self.lepo)
        while i > 0 and np.hypot(self.lepo[i - 1][1] - X0, self.lepo[i - 1][2] - Y0) > 2.0:
            i -= 1
        h.f_alku = self.lepo[i][0] if i < len(self.lepo) else f
        self.an.tapahtumat.append(("lahto", h.tunnus, f, f"f_alku {h.f_alku}"))
        # lahtoa edeltavat A-havainnot saavat B:n myohemmin (uusi pyynto ei onnistu ilman ruutuja): ne jaavat A-paikoiksi

    def _takaisin_viritykseen(self, f):
        """Hylatty yritys: jos kahva on yha hack-alueella, jatketaan viritettyna (heittaja kasittelee kiveä), muuten IDLE."""
        h = self.heitto
        ns = [n for n in h.naytteet if n.A is not None][-8:]
        if len(ns) >= 4 and all(self._hack_sisalla(*n.A) for n in ns[-3:]):
            lista = []
            for n in ns:
                u, v = self.m.maa_oikaistuun(*n.A)
                lista.append((n.f, n.A[0], n.A[1], float(u[0]), float(v[0]), n.ala))
            XY = (float(np.median([c[1] for c in lista])), float(np.median([c[2] for c in lista])))
            edelliset = [n for n in h.naytteet if f - n.f <= 90]        # sama kivi: todisteet kivesta (B-sovitukset) siirtyvat mukana
            self._virita(f, self.vari, lista, XY)
            self.heitto.naytteet = edelliset
            self.heitto.validoitu = h.validoitu                             # sama kivi
        else:
            self._idle_alkuun()

    # ---- LIIKKUU / VAHVISTETTU ---------------------------------------------------------------------------------
    def _liikkuu(self, f, ruutu):
        h = self.heitto
        Xp, Yp = self._ennuste(f)
        luotettava = self._ennuste_luotettava
        f_paikka_ed = max([m.f for m in h.naytteet[-60:] if m.paikka() is not None] or [self.f_liike])
        portti = 14.0 + 0.5 * self._nopeus_ruutu() + HUKASSA_PORTTI_CM * min(max(f - f_paikka_ed - 2, 0), HUKASSA_PORTTI_RUUTUA)      # kivi hukassa: hakuportti kasvaa
        a = self._vari_a(ruutu, (Xp, Yp), portti, ala_raja=(0.4, 1.8) if h.tila == "vahvistettu" else (0.4, 2.2))
        n = Naytteet(f)
        if a is not None:
            n.A, n.ala = (a[0], a[1]), a[2]
            a_n = a[2] / (a[3] * a[3])
            if len(self.ala_norm) < 3 or ALA_VAIHTELU[0] <= a_n / float(np.median(self.ala_norm)) <= ALA_VAIHTELU[1]:
                self.ala_norm = (self.ala_norm + [a_n])[-8:]               # paisunut blobi (kasi / hanska kiinni) ei nosta odotusarvoa
                self._ala_ulkona = []
            else:
                self._ala_ulkona = (self._ala_ulkona + [a_n])[-ALA_ULKONA_MAX:]
                if len(self._ala_ulkona) >= ALA_ULKONA_MAX:                # pysyva poikkeama: odotusarvo oli vaara
                    self.ala_norm, self._ala_ulkona = list(self._ala_ulkona), []
            o = self._offset_b()
            X0, Y0 = a[0] + o[0], a[1] + o[1]
            if luotettava and np.hypot(X0 - Xp, Y0 - Yp) > A_HARHA_CM:
                X0, Y0 = Xp, Yp                                            # A harhainen: B:n alkuarvo on B-ennuste
        else:
            X0, Y0 = Xp, Yp
        h.naytteet.append(n)
        self.an.pyyda_b(self, h, n, ruutu, X0, Y0, (Xp, Yp) if luotettava else None)
        if KIERTO_Y_MIN_CM <= Y0 <= KIERTO_Y_MAX_CM and len(h.kierto_kehykset) < KIERTO_MAX_RUUTUJA:
            self._kierto_leike(h, n, ruutu, X0, Y0)
        # paikan puute: viimeisin ruutu, jolla on jokin paikka (A tai B; B voi saapua myohemmin)
        f_paikka = max([m.f for m in h.naytteet[-60:] if m.paikka() is not None] or [self.f_liike])
        self.hukka = f - f_paikka
        viim = [m for m in h.naytteet[-12:] if m.paikka() is not None]
        p = viim[-1].paikka() if viim else (Xp, Yp)               # vahvistus vain MITATUSTA paikasta (ei ennusteesta)
        if h.tila == "liikkuu":
            v = self._nopeus_ruutu() * self.an.fps
            if viim and p[1] >= VAHVISTUS_Y_CM and v >= VAHVISTUS_V_CM_S and h.kivi_n(10 ** 6, f) >= KIVI_B_VAHVISTUS:
                h.tila = "vahvistettu"
                h.f_vahvistus = f
                self.an.tapahtumat.append(("vahvistettu", h.tunnus, f, f"v {v:.0f} cm/s Y {p[1]:.0f} kivi_B {h.kivi_n(10 ** 6, f)}"))
            elif (f - self.f_liike) / self.an.fps > HYLKAYS_AIKA_S or self.hukka > HUKKA_LIIKE_RUUTUA or v > 600.0 or self._ei_etene(f):
                h.tila = "hylatty"
                self.an.hylatyt.append(h)
                self.an.tapahtumat.append(("hylatty", h.tunnus, f, f"v {v:.0f} cm/s, hukka {self.hukka}"))
                self._takaisin_viritykseen(f)
        elif h.tila == "vahvistettu":
            if Yp >= LOPPU_Y_CM or self.hukka > PAIKKA_PUUTE_RUUTUA:
                self._valmis(f, "matka" if Yp >= LOPPU_Y_CM else "kivi hukassa")

    def _kierto_leike(self, h, n, ruutu, X0, Y0):
        """Leike oikaistusta kuvasta kiven akselin ymparilta kierteen mittausta varten (muovikartat lasketaan heiton jalkeen taustaprosessissa)."""
        u, v = self.m.maa_oikaistuun(X0, Y0, self.m.ast["z0_cm"])
        uc, vc = int(round(float(u[0]))), int(round(float(v[0])))
        roi, cx0, cy0 = self.m.kuvain.leike(ruutu, uc - KIERTO_LEIKE_R, vc - KIERTO_LEIKE_R, uc + KIERTO_LEIKE_R, vc + KIERTO_LEIKE_R)
        if roi is not None:
            h.kierto_kehykset.append((n, roi, cx0, cy0))

    def _ei_etene(self, f):
        """Liikkeessa vahintaan EI_ETENE_S, mutta kivi on viimeisen EI_ETENE_S aikana liikkunut alle EI_ETENE_CM (X- tai Y-vaihteluvali): kivea kasitellaan, ei
        heiteta. (Vaihteluvali eika nettoetenema: heiton alkua edeltava vastaliike hackissa kumoaa nettoetenemaan.)"""
        h = self.heitto
        if (f - self.f_liike) / self.an.fps < EI_ETENE_S:
            return False
        o = self._offset_b()
        pts = []
        for n in reversed(h.naytteet):
            if f - n.f > EI_ETENE_S * self.an.fps:
                break
            q = self._konsol(n, o)
            if q is not None:
                pts.append(q)
        if len(pts) < 6:
            return False
        P = np.array(pts)
        return float(max(np.ptp(P[:, 0]), np.ptp(P[:, 1]))) < EI_ETENE_CM

    def _valmis(self, f, syy=""):
        h = self.heitto
        h.tila = "valmis"
        ns = [n for n in h.naytteet if n.paikka() is not None]
        h.f_loppu = ns[-1].f if ns else f
        h.loppu_Y = float(ns[-1].paikka()[1]) if ns else None          # diagnostiikka: viimeinen mitattu Y ja miksi heitto paattyi
        h.loppu_syy = syy
        h.ala_med = float(np.median(self.ala_norm)) if self.ala_norm else None
        self.an.valmiit.append(h)
        self.edellinen_loppu_f = f
        self._idle_alkuun()
        self.an.loppu(h)

    # ---- paasilmukka -----------------------------------------------------------------------------------------------
    def kasittele(self, f, ruutu):
        self.n_ruutu += 1
        if self.tila == "IDLE":
            if f - self.edellinen_loppu_f > 2 * self.an.fps and f >= self.ei_ennen_f:
                self._idle(f, ruutu)
        elif self.tila == "ARMED":
            self._armed(f, ruutu)
        else:
            self._liikkuu(f, ruutu)
        h = self.heitto
        if h is not None and h.tila in ("liikkuu", "vahvistettu"):
            ns = [n for n in h.naytteet if n.paikka() is not None]
            h.f_loppu = ns[-1].f if ns else f


# ---------------------------------------------------------------------------------------------------------------------
# KAUKOPAA: kiven saapuminen toiseen puoliskoon ja hog-viivan ylitys (hog-hog-aika kierrosmaaraa varten)
# ---------------------------------------------------------------------------------------------------------------------
class _Rata(list):
    """Kaukopaan blobirata [(f, X', Y', ala), ...] ja sen kivitarkistukset (Saapuminen._kivi_kelpaa)."""

    def __init__(self, *a):
        super().__init__(*a)
        self.peitot = []                                  # B-sovituksen peitot tarkistuspisteissa
        self.f_tark = -10 ** 9                            # viimeisimman tarkistuksen ruutu
        self.hylatty = False                              # ei kivi (ihmisen varuste): rata seurataan loppuun mutta ei enaa tarkisteta eika hyvaksyta


class Saapuminen:
    """Heiton kiven saapuminen kaukopaahan (toiseen puoliskoon): kahvan variblobit hog-viivan ymparilla -> kaukopaan hog-viivan ylitysruutu (h.hog2_f).
    Kivi ilmestyy kuvan reunasta (alareuna, Y' ~ 735 cm), joten ennustetta ei ole: hakuvyohykkeen (hog-viivan ymparilla) taustasta poikeavat kahvan varin blobit
    yhdistetaan ratoihin, ja rata, jonka pisteet etenevat tasaisesti kohti pesaa (40-450 cm/s) ja alan puolesta sopivat kiven kahvaan (saman kiven kahvan ala heittopaassa),
    hyvaksytaan. Ylitysruutu = suoran Y'(ruutu) leikkaus hog-viivan keskiviivalla. Vain A-taso (varin sentroidi). Hakuvyohyke on laaja (koko radan leveys, ~1 Mpx), joten se
    tutkitaan harvennettuna (joka DECI. pikseli, ~1,5 ms / ruutu; koko tarkkuudella ~20 ms): kahvan blobi on harvennettunakin 15-30 px. Varin blobi ei yksin erota kivea ihmisen
    varusteista (sweeperin punainen kasivarsinauha kulkee hog-viivan yli ~1,5 s ennen kivea): hyvaksyttava rata tarkistetaan B-sovituksella (graniittirunko, _kivi_kelpaa)."""
    DECI = 3

    def __init__(self, analyysi, heitto):
        self.an, self.h = analyysi, heitto
        self.puoli = "oikea" if heitto.puoli == "vasen" else "vasen"
        self.m = analyysi.M[self.puoli]
        fps = analyysi.fps
        self.f_alku = int(heitto.f_loppu + SAAP_ALKU_S * fps)
        self.f_loppu = int(heitto.f_loppu + SAAP_LOPPU_S * fps)
        self.radat = []                                   # [[(f, X', Y', ala), ...], ...]
        P = np.array([(x, y, z) for x in (-SAAP_X_CM, SAAP_X_CM) for y in SAAP_Y_CM for z in (0.0, 22.0)])
        u, v = kivimalli.project_3d(self.m.cam.K, self.m.cam.R, self.m.cam.t, P)
        kv_ = self.m.kuvain
        x0, y0 = max(int(u.min()) - 10, 0), max(int(v.min()) - 10, 0)
        x1, y1 = min(int(u.max()) + 10, kv_.w), min(int(v.max()) + 10, kv_.h)
        d = self.DECI
        self.x0, self.y0 = x0, y0
        self.mx = np.ascontiguousarray(kv_.mx[y0:y1:d, x0:x1:d])               # harvennetut lahdekartat: ROI suoraan harvennettuna alkuperaisesta ruudusta
        self.my = np.ascontiguousarray(kv_.my[y0:y1:d, x0:x1:d])
        self.tausta = np.ascontiguousarray(self.m.cam.mode[y0:y1:d, x0:x1:d])
        self.ok = self.mx.size > 100
        heitto.saapuminen = "haku"

    def _maski(self, roi):
        """Kahvan varin maski harvennetussa ROI:ssa (kivi_netti.kahva_maski:n kynnykset, nopeammin: yksi varimuunnos ja inRange) ja taustasta poikkeavat pikselit."""
        if self.h.vari == "keltainen":
            lab = cv2.cvtColor(roi, cv2.COLOR_BGR2Lab)                        # b* - 128 >= 16 ja a* - 128 <= 10
            m = cv2.inRange(lab, (0, 0, 128 + 16), (255, 128 + 10, 255))
        else:
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            m = cv2.inRange(hsv, (0, kv.PUNAINEN_SMIN, kv.PUNAINEN_VMIN), (kv.PUNAINEN_H[0], 255, kv.PUNAINEN_VMAX))
            m |= cv2.inRange(hsv, (kv.PUNAINEN_H[1], kv.PUNAINEN_SMIN, kv.PUNAINEN_VMIN), (179, 255, kv.PUNAINEN_VMAX))
        ero = cv2.absdiff(roi, self.tausta)
        b, g, r = cv2.split(ero)
        ero = cv2.max(cv2.max(b, g), r)
        return cv2.bitwise_and(m, cv2.compare(ero, kv.FG_KYNNYS, cv2.CMP_GT))

    def _blobit(self, ruutu):
        """Kahvan varin blobit hakuvyohykkeella: [(u, v, ala, bbox)] oikaistun kuvan koordinaateissa ja koko tarkkuuden pikseleina."""
        d = self.DECI
        roi = cv2.remap(ruutu, self.mx, self.my, cv2.INTER_LINEAR)
        m = cv2.morphologyEx(self._maski(roi), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        n, lab, st, cen = cv2.connectedComponentsWithStats(m)
        out = []
        for k in range(1, n):
            ala = int(st[k, cv2.CC_STAT_AREA]) * d * d
            if ala < 25:
                continue
            bb = tuple(int(a) * d for a in st[k, :4])
            out.append((self.x0 + d * float(cen[k][0]), self.y0 + d * float(cen[k][1]), ala, bb))
        return out

    def kasittele(self, f, ruutu):
        """Yksi (yksilollinen) ruutu; palauttaa False kun haku on paattynyt (loytyi tai aika loppui)."""
        h = self.h
        if f < self.f_alku:
            return True
        if f > self.f_loppu or not self.ok:
            h.saapuminen = "ei_loytynyt"
            return False
        uudet = []
        for u, v, ala, bb in self._blobit(ruutu):
            if not (0.35 <= bb[2] / max(bb[3], 1) <= 2.8):
                continue
            X, Y = self.m.oikaistu_maahan(u, v)
            if not (SAAP_Y_CM[0] <= Y <= SAAP_Y_CM[1] and abs(X) <= SAAP_X_CM):
                continue
            s = self.m.skaala(X, Y)
            if h.ala_med:
                odotus = h.ala_med * s * s
                if not (0.45 * odotus <= ala <= 2.2 * odotus):
                    continue
            elif not (50 <= ala <= 900):
                continue
            uudet.append((f, float(X), float(Y), int(ala)))
        for q in uudet:                                   # yhdista lahimpaan rataan (viimeisin piste lahella, eri ruutu)
            paras, d_paras = None, None
            for r in self.radat:
                fl, Xl, Yl, _ = r[-1]
                if f <= fl or f - fl > 10:
                    continue
                d = float(np.hypot(q[1] - Xl, q[2] - Yl))
                if d <= min(25.0 + 12.0 * (f - fl), 70.0) and (paras is None or d < d_paras):
                    paras, d_paras = r, d
            if paras is not None:
                paras.append(q)
            else:
                self.radat.append(_Rata([q]))
        self.radat = [r for r in self.radat if f - r[-1][0] <= 20]
        for r in self.radat:
            if len(r) >= SAAP_PISTEITA and not r.hylatty:
                f_yl = self._ylitys(r)
                if f_yl is not None and self._kivi_kelpaa(r, f, ruutu):
                    h.hog2_f, h.saapuminen = f_yl, "loytyi"
                    self.an._paivita_kierre(h)
                    return False
        return True

    def _kivi_kelpaa(self, r, f, ruutu):
        """Onko radan viimeinen piste kivi eika ihmisen varuste. Kivella on graniittirunko: B-sovituksen peitto (graniittia siluetin sisalla) on kivella 0,34-0,71 ja sweeperin
        punaisella kasivarsinauhalla 0,06-0,28. Tarkistus tehdaan vain radalle, joka muuten kelpaisi, ja harvoin (SAAP_TARKISTUS_VALI); kelpaa, jos yksi tarkistus >= SAAP_PEITTO_VARMA
        tai kaksi >= SAAP_PEITTO_MIN, ja rata hylataan, kun SAAP_TARKISTUKSIA tarkistusta ei riita."""
        if r[-1][0] != f or (r.peitot and f - r.f_tark < SAAP_TARKISTUS_VALI):
            return False
        try:
            s = self.m.tarkka(ruutu, r[-1][1], r[-1][2])
        except Exception:
            s = None
        p = float(s.get("peitto", 0.0)) if s is not None and "virhe" not in s else 0.0
        r.peitot.append(p)
        r.f_tark = f
        if p >= SAAP_PEITTO_VARMA or sum(1 for q in r.peitot if q >= SAAP_PEITTO_MIN) >= 2:
            return True
        if len(r.peitot) >= SAAP_TARKISTUKSIA:
            r.hylatty = True
        return False

    def _ylitys(self, r):
        """Radan sovitus Y'(ruutu) = c0 f + c1 hog-viivan ymparilta (+-1,3 m; kiihtyvyys ei vaikuta): ylitysruutu (hog-viivan keskiviiva) tai None, jos rata ei kelpaa tai kivi ei ole
        viela ylittanyt viivaa (lisaa pisteita)."""
        fps = self.an.fps
        F = np.array([p[0] for p in r], float)
        X = np.array([p[1] for p in r], float)
        Y = np.array([p[2] for p in r], float)
        if Y.min() > HOG_KESKI_CM - 30.0:                 # viiva ei ole viela ylittynyt luotettavasti
            return None
        lahella = np.abs(Y - HOG_KESKI_CM) <= 130.0
        if lahella.sum() >= SAAP_PISTEITA - 1:
            F, X, Y = F[lahella], X[lahella], Y[lahella]
        if len(F) < SAAP_PISTEITA - 1 or F[-1] - F[0] < 0.2 * fps:
            return None
        c = np.polyfit(F, Y, 1)
        v = -c[0] * fps                                   # cm / s kohti pesaa
        if not (40.0 <= v <= 450.0) or float(np.sqrt(np.mean((Y - np.polyval(c, F)) ** 2))) > 6.0:
            return None
        if abs(np.polyfit(F, X, 1)[0]) > 0.5 * abs(c[0]):  # sivusuunnan liike: ei etene pitkin rataa
            return None
        f_yl = (HOG_KESKI_CM - c[1]) / c[0]
        if f_yl < F[0] - 0.5 * fps:                       # rata alkaa liian kaukaa viivan jalkeen
            return None
        return float(f_yl)


# ---------------------------------------------------------------------------------------------------------------------
# KIERRE: hog-hog-aika, kierrokset, suunta
# ---------------------------------------------------------------------------------------------------------------------
HIDASTUVUUS_TYYPILLINEN = 0.068  # m/s2: puhtaiden heittojen vapaan lennon hidastuvuus (0,06-0,07)


def hog_hog_aika_malli(v):
    """Hog-hog-aika (s) vakiohidastuvuuden mallilla (kun kaukopaan ylitysta ei mitattu): nopeus hog-viivalla irroituksen nopeudesta ja hidastuvuudesta. Vapaan lennon
    hidastuvuus on kohinainen (mittaus kattaa vain ~2 s: arvot -0,2 ... +0,5), joten se sekoitetaan tyypilliseen arvoon sen mukaan miten pitka vapaa lento on (paino 0 ... 1
    vapaasta matkasta 150 ... 550 cm); jos kivi ei sillakaan yllaisi kaukopaan hog-viivalle, kokeillaan tyypillista hidastuvuutta. None, jos kivi ei sillakaan yllaa."""
    a0 = HIDASTUVUUS_TYYPILLINEN
    a_mit = float(np.clip(v["vapaa_hidastuvuus"], 0.04, 0.14))
    w = float(np.clip((float(v.get("vapaa_matka_cm", 0.0)) - 150.0) / 400.0, 0.0, 1.0))
    v_r = max(float(v["nopeus_jalkeen"]), 0.3)
    s1 = max(HOG_KESKI_CM - v["irr_Y"], 0.0) / 100.0
    D = (ETAISYYS_CM - 2.0 * HOG_KESKI_CM) / 100.0
    for a in (a0 + w * (a_mit - a0), a0):
        v1sq = v_r * v_r - 2.0 * a * s1
        if v1sq <= 0.0:
            continue
        d = v1sq - 2.0 * a * D
        if d <= 0.0:
            continue
        return float((np.sqrt(v1sq) - np.sqrt(d)) / a)
    return None


def kierron_merkki(h):
    """Kiertosuunta: +1 vastapaivaan, -1 myotapaivaan, 0 tuntematon. Kuvamittaus (omega), ellei se ole epavarma ja vapaan lennon kaarto (kaarevuus < 0: kaartaa
    vasemmalle = vastapaivaan) ole selva; ilman kuvamittausta selva kaarto riittaa vihjeeksi."""
    k, v = h.kierto, h.viivat
    c = v.get("kaarevuus_1_cm") if v else None
    if k is not None and "virhe" not in k:
        s = 1 if k["omega_deg_s"] > 0 else -1
        if not k.get("varma") and c is not None and abs(c) >= KAARTO_VARMA_1_CM:
            s = 1 if c < 0 else -1
        return s
    if c is not None and abs(c) >= KAARTO_VIHJE_1_CM:
        return 1 if c < 0 else -1
    return 0


def heiton_suureet(h):
    """Paneelin ja kirjauksen suureet heitolle (Heitto): None, jos viivaestimaattia ei ole. liuku / heitto = suorien osumat kaukopaan T-viivalla (cm keskiviivasta, + =
    heittajan oikealle), irroitus = (heitto - liuku) * kierron merkki (vastapaivaan +1: etumerkki +, kun irroitus on kaarron vastaiseen suuntaan), kierre = kierrokset
    hog-hog-valilla (- = vastapaivaan), irr_hog_m = etaisyys irroituksesta hog-viivaan kiven etureunasta mitattuna."""
    v = h.viivat
    if v is None:
        return None
    L = ETAISYYS_CM
    XT_l = v["liuku_a"] + v["liuku_b"] * L
    XT_i = v["irr_X"] + np.tan(np.radians(v["irr_suunta_deg"])) * (L - v["irr_Y"])
    s = kierron_merkki(h)
    kierre = None
    if h.kierre is not None and s != 0:
        kierre = -s * abs(h.kierre["N"])
    return dict(liuku_cm=float(XT_l), heitto_cm=float(XT_i), merkki=s, irroitus_cm=(float(XT_i - XT_l) * s if s else None), kierre=kierre,
                kierre_mitattu=bool(h.kierre is not None and h.kierre["lahde"] == "mitattu"), T_hh=(h.kierre["T"] if h.kierre else None),
                irr_hog_m=float((HOG_SISA_CM - (v["irr_Y"] + KIVI_R_CM)) / 100.0), v_ennen=float(v["nopeus_ennen"]), v_jalkeen=float(v["nopeus_jalkeen"]),
                hidastuvuus=float(v["vapaa_hidastuvuus"]), vapaa_matka_cm=float(v["vapaa_matka_cm"]), virhe_cm=float((L - v["irr_Y"]) * np.tan(np.radians(v["irr_virhe_deg"]))), vakaa=bool(v.get("vakaa")),
                omega_deg_s=(h.kierto["omega_deg_s"] if h.kierto is not None and "virhe" not in h.kierto else None),
                kierto_varma=(bool(h.kierto.get("varma")) if h.kierto is not None and "virhe" not in h.kierto else None))


# ---------------------------------------------------------------------------------------------------------------------
# ANALYYSI: kopioruudut, molemmat paat, B-tyoprosessit, viivaestimaatti, kierre
# ---------------------------------------------------------------------------------------------------------------------
class Analyysi:
    """Syota ruudut yksi kerrallaan (syota); vahvistetut heitot (pisteet, viivat) luetaan heitot-listasta (piirtaja). synkroninen = True: B-tyot ja estimaatit
    odotetaan (sim-ajo: tulos ei riipu ajoituksesta); False: ne kulkevat taustalla (oikea reaaliaika). kierre = False: kierteen ja hog-hog-ajan mittaus pois."""

    def __init__(self, kansio="tulokset", fps=30.0, saikeet=3, tarkka=True, synkroninen=True, paat=("vasen", "oikea"), viive_b_ruutuja=6, fps_lahde=None, kierre=True):
        self.fps = float(fps)
        self.fps_lahde = 25.0 if fps_lahde is None else float(fps_lahde)      # None: paatellaan kopioruutujen osuudesta (_paata_fps_lahde)
        self._fps_auto = fps_lahde is None
        self.synkroninen = synkroninen
        self.viive_b = viive_b_ruutuja
        self.kansio = kansio
        self.tarkka = bool(tarkka)
        self.laske_kierre = bool(kierre) and self.tarkka
        self.M = {p: Mittari(kansio, p) for p in paat}
        self.pool = self.est_pool = self.kierto_pool = None
        if self.tarkka and saikeet > 0:
            self.pool = tee_pool(saikeet, initializer=alusta_tyo, initargs=(kansio, tuple(paat)))
        if not synkroninen or saikeet > 0:
            self.est_pool = tee_pool(1, initializer=matala_prioriteetti) if not synkroninen else None
        if self.laske_kierre and not synkroninen:
            self.kierto_pool = tee_pool(1, initializer=alusta_kierto, initargs=(kansio, tuple(paat)))
        self.b_valmis = self.pool is None            # B-tyoprosessit kaynnissa (vasta sitten "kivetonta" kahvaa ei hylata B-todisteen puutteen vuoksi): lampea()
        self.tyot = []                               # [(future, f, heitto, naytteet, alkuarvo)]
        self.est_tyot = []                           # [(future, heitto, f, n)]
        self.kierto_tyot = []                        # [(future, heitto)]
        self.saapumiset = []                         # kaukopaan hakijat (Saapuminen)
        self.paat = {p: Heittopaa(self, p) for p in paat}
        self.heitot = []                             # vahvistetut heitot (piirtajalle)
        self.valmiit, self.hylatyt = [], []
        self._tunnus = 0
        self._numerot = {}                           # (puoli, vari) -> viimeisin kiven numero
        self._viim_puoli = None
        self.edellinen_harmaa = None
        self.viim_f = None
        self.kopioita = 0
        self.yksilollisia = 0
        self.tapahtumat = []

    def lampea(self, odota=True, aikaraja=120.0):
        """Kaynnistaa tyoprosessit (spawn: tuonnit ja kameramallit, 5-15 s) ja odottaa niita, jotta ensimmaisen heiton B-mittaukset eivat myohastyisi (kivea ei
        muuten todettaisi heiton alussa). Palauttaa kaynnistettyjen prosessien maaran."""
        tehtavat = []
        for pool, n in ((self.pool, getattr(self.pool, "_max_workers", 1)), (self.est_pool, 1), (self.kierto_pool, 1)):
            if pool is not None:
                tehtavat += [pool.submit(tyo_nop) for _ in range(n)]
        if odota and tehtavat:
            valmiit, kesken = concurrent.futures.wait(tehtavat, timeout=aikaraja)
            virheet = [t.exception() for t in valmiit if t.exception() is not None]
            if virheet or kesken:
                print(f"VAROITUS: tyoprosessien kaynnistys ei onnistunut kokonaan ({virheet[0]!r})" if virheet else "VAROITUS: tyoprosessit eivat ehtineet kaynnistya", flush=True)
        self.b_valmis = True
        return len(tehtavat)

    def _paata_fps_lahde(self):
        """Erillisten ruutujen nimellistaajuus: jos kopioruutuja on alle 3 %, virta on kopioton ja erilliset ruudut tulevat ruutunopeudella; muuten (kuten tassa
        videossa, 26 % kopioita) lahdevirta on 25 r/s."""
        self._fps_auto = False
        osuus = self.kopioita / max(self.yksilollisia + self.kopioita, 1)
        self.fps_lahde = self.fps if osuus < 0.03 else 25.0

    def uusi_tunnus(self):
        self._tunnus += 1
        return self._tunnus

    def _numeroi(self, h):
        """Kiven numero vareittain per heittosuunta (oletus: kivet heitetaan numerojarjestyksessa); numerointi alkaa alusta, kun heittosuunta vaihtuu (uusi paa)
        tai kun joukkueen kivet (KIVIA_PAASSA) on heitetty."""
        if h.puoli != self._viim_puoli:
            for k in [k for k in self._numerot if k[0] == h.puoli]:
                del self._numerot[k]
            self._viim_puoli = h.puoli
        n = self._numerot.get((h.puoli, h.vari), 0) + 1
        if n > KIVIA_PAASSA:
            n = 1
        self._numerot[(h.puoli, h.vari)] = n
        h.kivi_nro = n

    # ---- kopioruutu ------------------------------------------------------------------------------------------------
    def _kopio(self, ruutu):
        """Kopioruutu (edellisen tarkka kopio uudelleenkoodattuna): harmaasavyero alle KOPIO_KYNNYS lahes kaikissa pikseleissa (alle 0,1 %) MOLEMMISSA puoliskoissa.
        (Vain toinen puolisko ei kelpaa: jos kivi liikkuu toisessa paassa, toinen puolisko voi olla lahes staattinen.)"""
        y0, h = kv.kn.PUOLET["vasen"][1], kv.kn.PUOLET["vasen"][3]
        g = cv2.cvtColor(ruutu[y0:y0 + h], cv2.COLOR_BGR2GRAY)
        kopio = False
        if self.edellinen_harmaa is not None:
            d = cv2.absdiff(g, self.edellinen_harmaa)
            puoli = d.shape[1] // 2
            raja = 0.001 * d.shape[0] * puoli
            kopio = np.count_nonzero(d[:, :puoli] > KOPIO_KYNNYS) < raja and np.count_nonzero(d[:, puoli:] > KOPIO_KYNNYS) < raja
        self.edellinen_harmaa = g
        return kopio

    # ---- B-tyot ---------------------------------------------------------------------------------------------------
    def pyyda_b(self, hp, heitto, naytteet, ruutu, X0, Y0, ennuste=None):
        """B-mittaus naytteelle: tyoprosessille (leike) tai suoraan (ei poolia / sim ilman prosesseja). Reaaliajassa ruuhkautuessa B jatetaan pois. ennuste: luotettava
        (B-pohjainen) paikkaennuste (X, Y), jonka suhteen tulos hyvaksytaan myos silloin, kun alkuarvo (A) oli harhainen."""
        if not self.tarkka:
            return
        if self.pool is not None and not self.synkroninen and len(self.tyot) > 24:
            return
        v = hp.m.valmistele(ruutu, X0, Y0)
        if v is None:
            return
        naytteet.odottaa = True
        alku = (X0, Y0) if ennuste is None else (X0, Y0, float(ennuste[0]), float(ennuste[1]))
        if self.pool is not None:
            fut = self.pool.submit(tyo_b, (id(naytteet), hp.puoli, v[0], v[1], v[2], X0, Y0))
            self.tyot.append((fut, naytteet.f, heitto, naytteet, alku))
        else:
            self._liita_b(heitto, naytteet, hp.m.sovita_leike(*v, X0, Y0), alku)

    def _liita_b(self, heitto, naytteet, s, alku):
        """B-tulos naytteelle: kivi todetaan (graniittia siluetin sisalla, pieni jaannos, enintaan B_MAX_POIKKEAMA_CM alkuarvosta tai B-ennusteesta); B kun kelvollisia
        pikseleita vahintaan B_VAIN_KELV, pehmea B (Bs) kun vahintaan SOFT_KELV, ja ennusteen lahella olevalla vahvistetun heiton paikalla heikko B (pesan varilliset
        renkaat: kelvollisia pikseleita vain muutama prosentti, mutta paikka on silti paljon tarkempi kuin A)."""
        naytteet.odottaa = False
        if s is None or "virhe" in s:
            return
        poik = float(np.hypot(s["X"] - alku[0], s["Y"] - alku[1]))
        poik_e = float(np.hypot(s["X"] - alku[2], s["Y"] - alku[3])) if len(alku) > 2 else None
        peitto = s.get("peitto", 1.0)
        kelv = s["kelvolliset"]
        naytteet.rms, naytteet.kelv, naytteet.peitto, naytteet.poik = s["rms"], kelv, peitto, poik
        raja = B_LEPO_POIK_CM if naytteet.lepo else B_MAX_POIKKEAMA_CM
        poik_ok = min(poik, poik_e if poik_e is not None else poik) <= raja
        if not poik_ok and heitto.tila == "liikkuu" and not naytteet.lepo and peitto >= B_ULOTTUMA_PEITTO:
            poik_ok = self._ulottuvilla(heitto, naytteet, s)
        kivi = peitto >= PEITTO_MIN and s["rms"] <= B_MAX_RMS and poik_ok
        if kivi and kelv >= B_VAIN_KELV:
            naytteet.B = (s["X"], s["Y"])
            heitto.b_laskuri += 1
        elif kivi and kelv >= SOFT_KELV:
            naytteet.Bs = (s["X"], s["Y"])
            heitto.b_laskuri += 1
        elif (kivi and poik_e is not None and poik_e <= B_HEIKKO_POIK_CM and peitto >= B_HEIKKO_PEITTO and s["rms"] <= B_HEIKKO_RMS and kelv >= B_HEIKKO_KELV):
            naytteet.Bs = (s["X"], s["Y"])
            heitto.b_laskuri += 1

    @staticmethod
    def _ulottuvilla(heitto, naytteet, s):
        """Lahdon jalkeen kivi kiihtyy nopeasti (jopa 8 cm / ruutu) ja kasi peittaa kahvan, joten ennuste (viimeisten paikkojen suora; A puuttuu) jaa jalkeen kymmenia senttimetreja ja
        sovitus, joka loysi kiven (graniittia 0,8-0,9), hylattiin 15 cm:n rajan takia: kivi hukattiin heti lahdon jalkeen (video 6820 s, punainen). Hyvaksytaan B, jos se on edellisen
        hyvaksytyn B:n ulottuvilla (kivi ei liiku nopeammin kuin ULOTTUMA_V_CM_RUUTU)."""
        for m in reversed(heitto.naytteet[-60:]):
            if m.f < naytteet.f and m.kivi():
                p = m.paikka()
                return bool(np.hypot(s["X"] - p[0], s["Y"] - p[1]) <= min(ULOTTUMA_MAX_CM, B_MAX_POIKKEAMA_CM + ULOTTUMA_V_CM_RUUTU * (naytteet.f - m.f)))
        return False

    def odota(self, f_asti=None):
        """Kerää valmiit B-tyot; f_asti: odota (estaen) kaikki tyot, joiden ruutu <= f_asti (sim-ajo)."""
        jaa = []
        for t in self.tyot:
            fut, f, heitto, n, alku = t
            if (f_asti is None and fut.done()) or (f_asti is not None and f <= f_asti):
                try:
                    _, s = fut.result()
                except Exception as e:
                    s = dict(virhe=repr(e))
                self.b_valmis = True
                self._liita_b(heitto, n, s, alku)
            else:
                jaa.append(t)
        self.tyot = jaa
        self._keraa_estimaatit()
        self._loppujen_tarkistus()
        self._keraa_kierto()

    # ---- viivaestimaatti ------------------------------------------------------------------------------------------
    def paivita_viivat(self, heitto, f, pakota=False):
        """Laske viivaestimaatti, jos B-aineisto on kasvanut tarpeeksi (tai pakota = lopullinen estimaatti)."""
        ns = heitto.kelvolliset_b()
        if len(ns) < 30 or (not pakota and len(ns) - heitto.viivat_n < ESTIMAATTI_VALI):
            if pakota:
                heitto.viivat_lopullinen, heitto.kierto_kehykset = True, []
            return
        ff, XX, YY = [n.f for n in ns], [n.B[0] for n in ns], [n.B[1] for n in ns]
        if max(YY) < ESTIMAATTI_MIN_Y_CM:
            if pakota:
                heitto.viivat_lopullinen, heitto.kierto_kehykset = True, []
            return
        heitto.viivat_n = len(ns)
        lisa = None
        if pakota and self.laske_kierre and heitto.kierto_kehykset:       # lopullinen estimaatti antaa myos kierteen mittauksen ruutujen ajat
            heitto.kierto_lisa = list(heitto.kierto_kehykset)
            lisa = [k[0].f for k in heitto.kierto_lisa]
        if self.est_pool is not None:
            if any(e.heitto is heitto for e in self.est_tyot):
                if pakota:                           # analyysisaie ei jumitu odottamaan: lopullinen kaynnistyy kun kesken oleva on valmis (_keraa_estimaatit)
                    heitto.lopullinen_jonossa = True
                return                               # edellinen laskenta kesken
            self.est_tyot.append(types.SimpleNamespace(fut=self.est_pool.submit(laske_viivat, ff, XX, YY, self.fps_lahde, self.fps, lisa), heitto=heitto, f=f, n=len(ns),
                                                       lopullinen=bool(pakota)))
        else:
            self._liita_viivat(heitto, laske_viivat(ff, XX, YY, self.fps_lahde, self.fps, lisa), f, bool(pakota))

    @staticmethod
    def _aseta_viivat(heitto, r, f, lopullinen=False):
        """Uusi estimaatti heitolle. vakaa = merkitseva irroitus, vapaata lentoa >= 250 cm, irroituspaikka muuttunut < 30 cm edellisesta estimaatista."""
        if r is None or "virhe" in r:
            return False
        heitto.viivat_hist = (heitto.viivat_hist + [r["irr_Y"]])[-3:]
        edel = heitto.viivat_hist[-2] if len(heitto.viivat_hist) >= 2 else None
        r["vakaa"] = bool(r["merkitseva"] and r["vapaa_matka_cm"] >= 250.0 and edel is not None and abs(r["irr_Y"] - edel) < 30.0 and r["irr_virhe_deg"] < 0.4)
        if lopullinen and r["merkitseva"]:
            r["vakaa"] = True
        L = ETAISYYS_CM
        heitto.viivat_loki.append((f, r["irr_Y"], r["irr_suunta_deg"], r["irr_X"] + np.tan(np.radians(r["irr_suunta_deg"])) * (L - r["irr_Y"]), r["liuku_a"] + r["liuku_b"] * L, r["vakaa"]))
        heitto.viivat, heitto.viivat_f = r, f
        return True

    def _liita_viivat(self, heitto, r, f, lopullinen=False):
        """Estimaatti heitolle (_aseta_viivat) ja sen seuraukset: lopullisesta estimaatista kierteen mittaus kayntiin, kierrokset uusiksi."""
        ok = self._aseta_viivat(heitto, r, f, lopullinen)
        if lopullinen:
            heitto.viivat_lopullinen = True
            if ok:
                self._kierto_aloita(heitto)
            heitto.kierto_kehykset, heitto.kierto_lisa = [], None            # leikkeet (~10 MB / heitto) eivat enaa tarpeen
        if ok:
            self._paivita_kierre(heitto)

    def _keraa_estimaatit(self, odota=False):
        jaa = []
        for e in self.est_tyot:
            if e.fut.done() or odota:
                try:
                    self._liita_viivat(e.heitto, e.fut.result(), e.f, e.lopullinen)
                except Exception:
                    if e.lopullinen:
                        e.heitto.viivat_lopullinen = True
            else:
                jaa.append(e)
        self.est_tyot = jaa
        for h in list(self.valmiit):                 # lopulliset estimaatit, jotka odottivat kesken olevaa
            if h.lopullinen_jonossa and not any(e.heitto is h for e in self.est_tyot):
                h.lopullinen_jonossa = False
                self.paivita_viivat(h, h.f_loppu, pakota=True)

    # ---- kierre ---------------------------------------------------------------------------------------------------
    def _kierto_aloita(self, h):
        """Kierteen mittaus heiton ruuduista (lopullisen estimaatin jalkeen: irroitushetki ja ruutujen ajat tiedossa): taustaprosessiin tai heti (sim)."""
        v = h.viivat
        if not self.laske_kierre or h.kierto_tila is not None or v is None or not v.get("t_lisa") or not h.kierto_lisa:
            return
        havainnot = []
        for (n, roi, cx0, cy0), t in zip(h.kierto_lisa, v["t_lisa"]):
            p = n.B if n.B is not None else n.Bs
            if p is not None:
                havainnot.append((float(t), n.f, float(p[0]), float(p[1]), roi, cx0, cy0))
        h.kierto_kehykset, h.kierto_lisa = [], None
        h.kierto_tila = "kaynnissa"
        args = (h.puoli, h.vari, havainnot, v["t_irr"])
        if self.kierto_pool is not None:
            self.kierto_tyot.append((self.kierto_pool.submit(tyo_kierto, args), h))
        else:
            self._aseta_kierto(h, kierto_laske(self.M[h.puoli], args))

    def _aseta_kierto(self, h, r):
        h.kierto = r
        h.kierto_tila = "virhe" if (r is None or "virhe" in r) else "valmis"
        self.tapahtumat.append(("kierto", h.tunnus, h.f_loppu, h.kierto_tila))
        self._paivita_kierre(h)

    def _keraa_kierto(self, odota=False):
        jaa = []
        for fut, h in self.kierto_tyot:
            if fut.done() or odota:
                try:
                    r = fut.result()
                except Exception as e:
                    r = dict(virhe=repr(e))
                self._aseta_kierto(h, r)
            else:
                jaa.append((fut, h))
        self.kierto_tyot = jaa

    def _paivita_kierre(self, h):
        """Kierrosmaara hog-hog-valille (h.kierre): omega (kuvamittaus) vaimenee eksponentiaalisesti; hog-hog-aika luetaan videon puoliskoista (vasemman puoliskon radan
        hog-ylitys ja oikean puoliskon kuvan hog-ylitys), ellei kaukopaan ylitysta ole (viela) loydetty: vakiohidastuvuuden malli (lahde 'malli')."""
        k, v = h.kierto, h.viivat
        if k is None or "virhe" in k or h.hog1_f is None or v is None:
            return
        fps = self.fps
        t1 = h.hog1_f / fps
        T_malli = hog_hog_aika_malli(v)
        T, lahde = None, None
        if h.hog2_f is not None:
            T_mit = (h.hog2_f - h.hog1_f) / fps
            if T_mit > 3.0 and (T_malli is None or 0.5 * T_malli <= T_mit <= 2.0 * T_malli):
                T, lahde = T_mit, "mitattu"
        if T is None and T_malli is not None:
            T, lahde = T_malli, "malli"
        if T is None:
            h.kierre = None
            return
        N = nk.kierrokset(k["omega_deg_s"], k["f_keski"] / fps, t1, t1 + T)
        h.kierre = dict(N=N, T=float(T), lahde=lahde, omega=k["omega_deg_s"])

    def _loppujen_tarkistus(self):
        for h in list(self.valmiit):
            if getattr(h, "odottaa_loppua", False) and not any(t[2] is h for t in self.tyot):
                self._lopullinen(h)

    def loppu(self, heitto):
        """Heitto paattyi: hog-viivan ylitys (heittopaa), kaukopaan haku, lopullinen estimaatti kaikesta aineistosta. Synkronisessa tilassa heti; reaaliajassa kun heiton
        B-tyot ovat valmiit (odota())."""
        self.tapahtumat.append(("valmis", heitto.tunnus, heitto.f_loppu, f"viimeinen Y {heitto.loppu_Y or 0:.0f} cm, syy: {heitto.loppu_syy or 'ajo loppui'}"))
        heitto.hog1_f = heitto.ylitys_f(HOG_KESKI_CM)
        if self.laske_kierre and heitto.hog1_f is not None:
            self.saapumiset.append(Saapuminen(self, heitto))
        if self.synkroninen:
            self.odota(f_asti=10 ** 12)
            self._lopullinen(heitto)
        else:
            heitto.odottaa_loppua = True

    def _lopullinen(self, heitto):
        heitto.odottaa_loppua = False
        heitto.lopullinen = True
        heitto.hog1_f = heitto.ylitys_f(HOG_KESKI_CM)          # kaikki B-paikat kasissa: tarkempi ylitysaika
        self.paivita_viivat(heitto, heitto.f_loppu, pakota=True)

    def paata_kesken(self):
        """Videon loppu kesken heiton: vahvistetut keskenerat heitot paatetaan (lopullinen estimaatti naista ruuduista)."""
        for hp in self.paat.values():
            h = hp.heitto
            if h is not None and h.tila == "vahvistettu":
                hp._valmis(self.viim_f if self.viim_f is not None else (h.f_loppu or 0))

    def viimeistele(self):
        """Videon loputtua: paata keskenerat heitot, odota kesken olevat tyot (B, estimaatit, kierre)."""
        self.paata_kesken()
        self.odota(f_asti=10 ** 12)
        self._loppujen_tarkistus()
        for _ in range(4):                           # kesken olevat ja jonossa odottavat lopulliset estimaatit
            self._keraa_estimaatit(odota=True)
            if not self.est_tyot and not any(h.lopullinen_jonossa for h in self.valmiit):
                break
        self._keraa_kierto(odota=True)

    def valmista_kaikki(self):
        """True kun mikaan tausta- tai kaukopaan tyo ei ole kesken (reaaliajassa: lopetus voi odottaa tulokset)."""
        return not (self.tyot or self.est_tyot or self.kierto_tyot or self.saapumiset or any(getattr(h, "odottaa_loppua", False) for h in self.valmiit))

    # ---- paasilmukka -----------------------------------------------------------------------------------------------
    def syota(self, f, ruutu):
        """Yksi ruutu (BGR, 1920 x 1080). Kopioruudut ohitetaan."""
        if self._kopio(ruutu):
            self.kopioita += 1
            return
        self.yksilollisia += 1
        self.viim_f = f
        if self._fps_auto and self.yksilollisia + self.kopioita >= 150:
            self._paata_fps_lahde()
        for hp in self.paat.values():
            hp.kasittele(f, ruutu)
            h = hp.heitto
            if h is not None and h.tila == "vahvistettu" and h not in self.heitot:
                self._numeroi(h)
                self.heitot.append(h)
        for s in list(self.saapumiset):
            if not s.kasittele(f, ruutu):
                self.saapumiset.remove(s)
        self.odota(f - self.viive_b if self.synkroninen else None)
        for hp in self.paat.values():
            if hp.heitto is not None and hp.heitto.tila == "vahvistettu":
                self.paivita_viivat(hp.heitto, f)

    def ohita(self):
        """Ruutu nakyy tulosteessa mutta sita ei analysoida (analyysi myohassa): kopioruututesti ei saa verrata kahta eri ruutua."""
        self.edellinen_harmaa = None

    def sulje(self):
        for p in (self.pool, self.est_pool, self.kierto_pool):
            if p is not None:
                p.shutdown(wait=False, cancel_futures=True)
