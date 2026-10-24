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
import os as _os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):       # tyoprosessit perivat nama: pienet matriisit eivat hyoty BLAS-saikeista, ja niiden ylikuorma hidastaa konetta
    _os.environ.setdefault(_v, "1")                                                                       # (spawn-prosessit ovat kaynnistyneet vasta tama jalkeen); yksi saie myos toistettavampi
import collections
import concurrent.futures
import multiprocessing
import os
import sys
import sysconfig
import time
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
KAHVA_ALA_CM2 = (89.0, 1250.0)   # hyvaksyttava kahvan pinta-ala hack-alueella maatason cm2:na (= px / skaala^2; kiintea 50-700 px videon 1 mittakaavalla 0,75 px/cm). Videoilla 1-2 keltainen ~150-260 px,
                                 # punainen ~110-160 px; videon 3 kamerassa (0,60 px/cm) punainen vain ~46-57 px, jolloin kiintea 50 px:n raja virittaa kiven vain sattumalta (video 3: heitto jai huomaamatta)
LEPO_S = 0.5                     # kivi hack-alueella vahintaan tama ennen kuin se "viritetaan" (heittaja kasittelee kiveä, joten ei tiukkaa lepoa)
LEPO_CM = 8.0                    # viritys: sentroidin vaihtelu alle tama
LEPO_IKKUNA_S = 0.65             # viritys: vakaus tarkistetaan vain viimeisesta taman pituisesta ikkunasta (lista voi alkaa liikkeesta: kivi lasketaan hackiin, kantaminen)
LEPO_SIIRTYMA_CM = 5.0           # ... ja kivi ei saa siirtya ikkunassa enempaa (lepaava kivi 1-2 cm, kannettava / laskettava kivi enemman)
LIIKE_CM = 6.0                   # lahto: kolmen viimeisen havainnon mediaani yli tama lepopaikasta (lepopaikka = vanhempien havaintojen mediaani)
LEPO_N = 15                      # lepopaikan arvioon kaytetaan enintaan naita vanhempia havaintoja
HUKKA_LEPO_S = 2.0               # kahva ja (todetun kiven) runko poissa levossa >= tama -> takaisin IDLE:en (heittajan kasi peittaa kahvan hetkeksi; liian lyhyt aika pudotti kiven kesken valmistelun)
B_LEPO_POIK_CM = 12.0            # levossa (viritetty kahva) B kelpaa kiveksi vain jos se on enintaan tama kaukana kahvan (A) paikasta: kahva on kiven keskella (tosikivella mediaani 1-5 cm, p90 <= 9 cm, max 10,4 cm), kiven viereinen kenka / kasi on 12-27 cm paassa
B_LEPO_PEITTO = 0.5              # "kaukainen kivi": B-sovituksessa graniittia vahintaan tama, mutta yli B_LEPO_POIK_CM kahvan paikasta -> kahva ei ole kiven kahva
B_ULOTTUMA_PEITTO = 0.5          # lahdon jalkeen (heittoa ei viela vahvistettu): B kelpaa myos > B_MAX_POIKKEAMA_CM alkuarvosta, jos se on edellisen hyvaksytyn B:n ulottuvilla ja graniittia vahintaan taman verran
ULOTTUMA_V_CM_RUUTU = 8.0        # ... ulottuma = B_MAX_POIKKEAMA_CM + taman verran per ruutu edellisesta hyvaksytysta B:sta (kivi kiihtyy lahdossa jopa 8 cm / ruutu: 2,4 m/s)
ULOTTUMA_MAX_CM = 60.0
VAIHTO_S = 1.0                   # viritetyn, viela todentamattoman kahvan voi korvata toisella vakaalla kandidaatilla vasta taman ajan kuluttua (aika kahdelle B-sovitukselle)
VAIHTO_ETAISYYS_CM = 12.0        # ... ja saman varin kandidaatti on vahintaan taman verran viritetysta kahvasta
LEPO_TODISTE_RUUTUA = 90         # levossa kiven todisteet (B-sovitukset kahvan kohdalla / kaukana) lasketaan vain taman ajan ajalta: kantamisen ja laskemisen aikaiset sovitukset eivat saa todentaa kahvaa, kun kivi on jo asettunut muualle
LEPO_KESKI_N = 3                 # levossa olevan kiven keskipiste = viimeisten taman monen graniittisovituksen (sovitus loytaa kiven lahtopaikasta riippumatta) mediaani, ...
LEPO_KESKI_HAJONTA_CM = 5.0      # ... kun ne ovat enintaan taman verran mediaanista (kivi lepaa)
VAIHTO_PARANNUS_CM = 3.0         # viritys vaihtuu toiseen kandidaattiin, jos se on enintaan B_LEPO_POIK_CM kiven keskipisteesta ja vahintaan taman verran lahempana kuin viritetty kahva (video 1, 3105 s: punainen kenkapohja 17 cm, keltainen kahva 6 cm)
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
VAHVISTUS_V_VARMA_CM_S = 150.0   # hitaampi kuin tama (120-150 cm/s) vahvistetaan vain, kun B-sovituksia on vahintaan KIVI_B_VAHVISTUS_HIDAS: kavelevan tuomarin kantama kivi (Ahrenberg, pelin lopussa: 135 cm/s, 32 B-sovitusta)
KIVI_B_VAHVISTUS_HIDAS = 60      # ... ei ole heitto; oikeiden heittojen nopeus vahvistuksessa oli 159-348 cm/s (450 heittoa) ja hitaimmilla 74-189 B-sovitusta
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
LIIKE_ALA_MAX = 4.5              # liikkeessa (heitto viela vahvistamatta) kahvan blobi saa olla jopa taman monikertainen, jos normaalin ala-ikkunan (0,4-2,2) blobia ei loydy: kahva sulautuu heittajan hanskaan
                                 # ja hackin takana oleviin punaisiin varusteisiin (ala 2,4-3,8 x odotettu), ja ilman tata kivi hukkuu juuri vahvistusta edeltavaksi 15 ruuduksi. Tallainen blobi ei paivita odotettua alaa.
HACK_HUKKA_Y_CM = -200.0         # kivi hukassa hackin lahella (viimeisin paikka Y < taman): heittajan vauhdinotto vie kiven hackin taakse ja siella se pysahtyy tai kaantyy, joten
HACK_HUKKA_RUUTUA = 3            # ... kun A on puuttunut vahintaan taman monta ruutua, kahvaa haetaan myos VIIMEISEN PAIKAN ymparilta (ennuste jatkaisi vauhdinottoa kauas kiven ohi) ...
HACK_HUKKA_PORTTI_CM = 45.0      # ... taman portin sisalta
HUKASSA_PORTTI_CM = 1.0          # A:n hakuportti kasvaa taman verran per ruutu, jolla paikkaa ei ole (enintaan HUKASSA_PORTTI_RUUTUA ruudun verran)
HUKASSA_PORTTI_RUUTUA = 15
IDLE_VALI = 3                    # IDLE: kandidaattihaku joka n:s yksilollinen ruutu
ESTIMAATTI_VALI = 10             # viivaestimaatti paivitetaan kun vahintaan n uutta kelvollista B-naytetta
ESTIMAATTI_MIN_Y_CM = 330.0      # vasta kun rata yltaa tahan (muuten ei irroitusta)
ESTIMAATTI = dict(min_jalkeen=70.0, Y_r_min=200.0, kohinakatto=(0.8, 0.15), esitieto=True)   # osittaisen aineiston asetukset (liuku_irroitus.analysoi); esitieto: vapaan lennon fysiikka (hidastuvuus, kaarevuus) pehmeana esitietona
KOPIO_KYNNYS = 15.0              # kopioruutu: harmaasavyero (suurin 0,1 % pikseleista) alle taman (aja_kivi.KOPIO_KYNNYS)

# kierre, hog-viivat, numerointi
KIERTO_Y_MIN_CM = 200.0          # kierteen mittauksen ruudut (leikkeet) kerataan kun kiven Y on taman ja KIERTO_Y_MAX_CM:n valilla (irroitus on tavallisesti Y = 360-610)
KIERTO_Y_MAX_CM = 800.0          # kuvan reuna ~ 750 cm: kiven keskipiste on kuvassa (ja kierre mitataan) siihen asti (aiemmin 720: ikkuna katkesi ennen kuvan reunaa)
KIERTO_LEIKE_R = 72              # talteen otettavan leikkeen puolikas (px): kierre tarvitsee 56 px + varaa (paikka tarkentuu B:lla jalkikateen)
KIERTO_MAX_RUUTUJA = 240
KIVI_R_CM = 14.55                # kiven sade (suurin halkaisija 29,1 cm): etureunan etaisyys akselista
HOG_SISA_CM = 640.0              # hog-viivan sisareuna T-viivasta (kalibrointi_netti.HOG_Y_CM); viiva on n. 12,7 cm leveä
HOG_KESKI_CM = HOG_SISA_CM + 6.35   # hog-viivan keskiviiva: tama ylitys ajoitetaan molemmissa paissa (hog-hog-matka = ETAISYYS_CM - 2 * HOG_KESKI_CM = 21,8 m)
ESTIMAATTI["Y_r_max"] = HOG_KESKI_CM - KIVI_R_CM    # saanto: irroitus ennen hog-viivaa (kiven etureuna ei ylita viivaa): kiven keskipiste Y <= 631,8 cm; sen jalkeinen rata on varmasti vapaata lentoa
HOG_HARHA_CM = ((590, +0.00), (600, +0.08), (610, +0.32), (620, +0.71), (630, +0.91), (640, +0.96), (650, +0.80), (660, +0.70), (670, +0.53), (680, +0.33), (690, +0.16), (700, +0.00))     # B-sovituksen toistuva Y-harha (Y cm, harha cm): kiven etureunan ylittaessa hog-viivan Y nayttaa ~1 cm liian suurelta (viivan vari siluetin reunalla), kaikilla heitoilla ja molemmilla
                                 # vareilla samoin (vapaan lennon heittojen mediaani: irroitus ennen Y = 470 cm, 38 / 112 heittoa; hajonta heitoittain ~0,3 cm). Harha = paikallinen kohouma, jonka 2. derivaatta (~0,4 m/s2) nayttaa tyonnolta ja sen loppumiselta,
                                 # joten se korjataan ennen irroituksen tunnistusta (valmistele_rata). Taulukon laskee tools/kalibroi_paikkavirhe.py radat.pkl:sta (aja uudelleen, jos kalibrointi tai kivimalli muuttuu).
HOG_IKKUNA_CM = 120.0            # hog-viivan kohdan paikallinen nopeus / suunta (laske_viivat: nopeus_hog1, suunta_hog1_deg): paikat taman matkan sisalla viivasta
KIVIA_PAASSA = 8                 # numerointi alkaa alusta taman jalkeen (yhden joukkueen kivet / paa)
KAARTO_VARMA_1_CM = 4.0e-5       # vapaan lennon kaarevuus (1/cm), jonka yli kaarto ratkaisee kiertosuunnan jos kuvamittaus on epavarma
KAARTO_VIHJE_1_CM = 2.5e-5       # ... ja jonka yli kaarto kertoo suunnan kun kuvamittausta ei ole
KAUKO_PYS_HIDASTUVUUS = 0.10          # m/s2: pysahtymisajan arvio hog-ylityksesta, kun kaukopaan seuranta ei onnistunut (tyypillinen hidastuvuus hogilta pysahtymiseen)
KIERRE_SIGMA_VARMA = 0.25         # kierrosmaara on 'varma', kun sen epavarmuus (netti_kierre.yhdista) on enintaan taman verran kierrosta eika mittaukset ole ristiriidassa
# kaukopaan hog-viivan ylityksen haku (toisen puoliskon kuvasta)
SAAP_ALKU_S = 3.0                # haku alkaa taman verran heittopaan poistumisen jalkeen
SAAP_LOPPU_S = 45.0              # ... ja paattyy taman jalkeen
SAAP_X_CM = 300.0                # hakuvyohyke kaukopaassa: |X'| <= taman, Y' valilla
SAAP_Y_CM = (450.0, 800.0)
SAAP_ALA_RAJAT = (0.45, 2.2)     # hakuvyohykkeen blobin ala suhteessa odotettuun (heittopaan kahvan normitettu ala x skaala^2)
SAAP_PISTEITA = 6                # rata vahvistetaan kun vahintaan taman verran pisteita
SAAP_PISTEITA_B = 4              # kaukopaan rata hyvaksytaan kiveksi, kun siina on vahintaan taman verran pisteita ...
SAAP_B_HYVAKSY = 2               # ... joista vahintaan taman monelle B-sovitus toteaa kiven (graniittia siluetin sisalla: kivi 0,34-0,95, ihmisen varusteet, esim. sweeperin punainen kasivarsinauha, 0,06-0,28) ...
SAAP_B_VALI = 3                  # ... ja B pyydetaan radan uusimmalle pisteelle taman monen ruudun valein (vain kulkevalle radalle)
# kaukopaan seuranta: kivi tunnistettu -> seuranta (A + B) pysahtymiseen asti
KAUKO_HUKKA_RUUTUA = 180         # seuranta paattyy, kun B-paikkaa ei ole ollut taman monta ruutua (kivi piilossa harjaajien alla tai poistunut kuvasta); ennuste kantaa piilossaolon yli
KAUKO_HIDASTUVUUS_CM_S2 = 8.0    # ennusteen hidastuvuus (cm/s2) piilossaolon aikana (0,08 m/s2: tyypillinen vapaa lento; pysahtymisvaiheessa suurempi)
KAUKO_A_LUOTTO_RUUTUA = 20        # A kelpaa kiven paikaksi vain, jos viimeisesta B-paikasta on enintaan taman monta ruutua ...
KAUKO_A_POIKK_CM = 12.0          # ... ja A (+ B-A-ero) on enintaan taman verran ennusteesta (muuten A voi olla harjaajan kasivarsinauha)
KAUKO_PORTTI_KASVU_CM = 2.0      # hakuportti kasvaa taman verran per paikaton ruutu ...
KAUKO_PORTTI_MAX_CM = 110.0      # ... enintaan tahan
SAAP_KARTTA_CM = 3.5             # kaukopaan kivikartta (ennen heittoa paikallaan olleet saman varin kahvat): taman sisalla olevia blobeja ei seurata liikkuvana kivena
SAAP_KARTTA_VALI = 60            # kartan paivitys (ruutua) kunnes liikkuva rata loytyy
KAUKO_PYS_V_CM_S = 12.0          # hukattu kivi on pysahtynyt piiloon, jos sen nopeus ennen hukkaa oli alle taman
KAUKO_EKSTRAP_CM = 220.0         # hog-ylitys ekstrapoloidaan radan alusta enintaan taman matkan (kivi tunnistettiin vasta viivan jalkeen)
KAUKO_PYS_CM = 4.5               # pysahtynyt: kiven B-paikka pysyy taman sisalla (vaihteluvali X ja Y; B-sovituksen kohina levossa on +-3 cm) ...
KAUKO_PYS_S = 0.8                # ... taman ajan
KELLO_TORMAYS_LISA_S = 4.0       # kello: tormayksessa liikkeelle lahteneet kivet pysahtyvat arviolta taman ajan kuluessa tormayksesta (heitetyn kiven pysahtymishetki, jos myohemmin)
KAUKO_KIIHTYY_CM_S = 70.0        # kaukopaan rata katkaistaan, kun hidastunut kivi (nopeus alle KAUKO_KIIHTYY_HIDAS_CM_S) alkaa liikkua: nopeus kasvaa taman verran (ja > 2-kertaiseksi) viimeisen 1,5 s:n
KAUKO_KIIHTYY_SUHDE = 2.0        # pienimmasta: kivi ei voi kiihtya, joten seurattu kohde on vaihtunut (kivea kuljetetaan kasin / seurataan toista kohdetta)
KAUKO_KIIHTYY_HIDAS_CM_S = 40.0
KELLO_ENNEN_S = 3.0              # kello: edellisen kiven pysahtymisen arvio saa olla enintaan taman verran seuraavan heiton lahdon jalkeen (muuten aika = ei saatu)
KELLO_TVIIVA_Y_CM = -KIVI_R_CM   # kello pysahtyy, kun kiven etureuna osuu heittopaan T-viivaan (Y = 0): kiven keskipiste on silloin Y = -kiven sade
KELLO_TVIIVA_IKKUNA_CM = 110.0   # T-viivan ylitysruudun sovitus: paikat taman matkan sisalla viivasta
KELLO_TVIIVA_VIIVE_S = 3.1       # ellei ylitysta voi mitata (rata ei yllä T-viivalle): heiton lahdosta (h.f_alku) tama aika (mitattu mediaani 112 heitolla: 3,1 s, vaihtelu 1,4-8,8 s)
KELLO_ARVIO_MAX_S = 12.0         # kello: pysahtymisen arvio hog-ylityksesta (ilman kaukopaan seurantaa) enintaan taman verran hog-ylityksen jalkeen
KELLO_NOPEA_M_S = 1.3            # ... ja jos nopeus hogilla on yli taman (poisto), enintaan KELLO_NOPEA_MAX_S
KELLO_NOPEA_MAX_S = 6.0
KELLO_MALLI_ODOTUS_S = 5.0       # kello: kaukopaan seuranta ei ole kertonut pysahtymista -> mallin (nopeus hog-viivalla, tyypillinen hidastuvuus) mukainen pysahtymishetki oletetaan toteutuneeksi, kun siita on kulunut taman verran ...
KELLO_ALKU_PYS_S = 31.0          # kello: ei viivaestimaattia / hog-ylitysta -> pysahtymisen arvio heiton lahdosta (mediaani 31 s, 10-90 %: 22-35 s, 74 heittoa)
KELLO_NAHTY_S = 2.0              # ... eika kivea ole nahty liikkeessa: kaukopaan B-paikat eivat ole siirtyneet yli KELLO_NAHTY_CM viimeisen taman ajan sisalla
KELLO_NAHTY_CM = 10.0
PELI_TAKA_Y_CM = -(183.0 + KIVI_R_CM)   # kivi on poissa pelista, kun sen keskipiste on kokonaan taka-viivan (Y' = -183 cm) takana ...
PELI_SIVU_X_CM = 237.5 + KIVI_R_CM       # ... tai sivuviivan (|X'| = 237,5 cm) ulkopuolella: kello kay (kivi ei enaa vaikuta peliin), vaikka kivi liukuisi viela
KAUKO_LAUTA_Y_CM = 250.0         # hukattu nopea kivi, joka on jo pesan takana (Y' < 0), osuu taka-aitaan taman Y':n (-Y') kohdalla ja pysahtyy siihen ...
KAUKO_LAUTA_LISA_S = 1.0         # ... + taman verran
KELLO_PIILOSSA_MAX_S = 8.0       # kello: hukatun (piiloon / pois kuvasta) kiven pysahtymishetki = viimeinen paikka + nopeus / KAUKO_PYS_HIDASTUVUUS, enintaan taman verran
KAUKO_PYS_RAJA_CM = 3.0          # pysahtymishetki radasta: kohinataso levossa (raakojen etaisyyksien 95 %:n fraktiili), vahintaan tama (cm) ...
KAUKO_PYS_LISA_S = 1.0           # ... + taman verran (kivi hiipii viimeiset sentit hitaasti: 4-5 cm lepopaikasta ~1 s ennen pysahtymista)
KAUKO_LOPPU_S = 80.0             # seurannan enimmaiskesto
KAUKO_B_LIIKKUU_RUUTUA = 1       # B joka n:s ruutu kun kivi liikkuu (> 30 cm/s) ...
KAUKO_B_HIDAS_RUUTUA = 4         # ... hitaasti (3-30 cm/s) ...
KAUKO_B_LEPO_RUUTUA = 10         # ... ja levossa
KAUKO_KIERTO_Y_MIN_CM = 150.0    # kierteen mittauksen ruudut kaukopaassa (kuten KIERTO_Y_MIN_CM / KIERTO_Y_MAX_CM heittopaassa)
KAUKO_KIERTO_Y_MAX_CM = 720.0
KAUKO_KIERTO_MAX_RUUTUJA = 240
KAUKO_KIERTO_KEHYKSIA = 240        # kaukopaan kierteen sovitukseen otetaan enintaan taman verran (tasaisesti valittuja) ruutuja


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
    c.graniitti = kv.graniitti_vari(cam)
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
        vakiot = dict(fg_kynnys=kv.FG_KYNNYS, g_vmin=kv.G_VMIN, g_vmax=kv.G_VMAX, g_smax=kv.G_SMAX, varjo=kv.VARJO_ALARAJA, graniitti=[float(x) for x in kv.graniitti_vari(self.cam)],
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


def alusta_kierto(kansio, puolet, matala=True):
    """Kierteen mittauksen tyoprosessi: vain kamerat (ei kivimallia eika taustamaskeja). matala = alempi prioriteetti (kaukopaan kierre ja kiven vakiot tarkentavat jalkikateen); heittopaan
    kierre ajetaan normaalilla prioriteetilla, koska se nakyy livekuvassa heti heiton jalkeen eika saa jaada muiden taustatoiden jalkeen."""
    if matala:
        matala_prioriteetti()
    cv2.setNumThreads(1)
    for p in puolet:
        _MK[p] = Mittari(kansio, p, runko=False)


def kierto_laske(m, args):
    """Kierteen mittaus: args = (puoli, vari, havainnot, t_irr) heittopaassa (netti_kierre.arvioi_posteriori) tai (puoli, vari, havainnot, t_alku, omega_ennakko) kaukopaassa
    (netti_kierre.arvioi_posteriori_kauko), m = puolen Mittari. omega_ennakko = (omega_deg_s, t_s, sd_deg_s) heittopaan varma mittaus (heikko ennakko, ratkaisee kaukopaan alias-minimien valilla; voi olla
    None). Palauttaa mittauksen dictin (tai dict(virhe=...))."""
    puoli, vari, havainnot, t_irr = args[:4]
    try:
        if len(args) <= 4:                                     # heittopaa: robusti posteriori kaikista ruuduista irroituksesta kuvan reunaan (netti_kierre.arvioi_posteriori)
            return nk.arvioi_posteriori(m.cam, m.ast, vari, havainnot, t_irr, m.kierto_muoto.get(vari))
        return nk.arvioi_posteriori_kauko(m.cam, m.ast, vari, havainnot, t_irr, m.kierto_muoto.get(vari), omega_ennakko=args[4])        # kaukopaa: kuten heittopaa, ruudukko tihea, heikko ennakko heittopaasta
    except Exception as e:
        return dict(virhe=repr(e))


def tyo_kierto(args):
    return kierto_laske(_MK[args[0]], args)


def vakiot_laske(args):
    """Kiven vakiot kaukopaan radasta (netti_vakiot.sovita); args = sovita-funktion nimetyt argumentit + valinnainen irr = netti_vakiot.tarkista_irroitus:n argumentit (irroituksen uskottavuus,
    tulos avaimella irr_tarkistus; kayttaa sovitetun kiven kappaa, jos se on luotettava). Palauttaa sovita-dictin (tai dict(ok=False, syy=...))."""
    args = dict(args)
    irr = args.pop("irr", None)
    try:
        import netti_vakiot as nv
    except Exception as e:
        return dict(ok=False, syy=repr(e))
    try:
        r = nv.sovita(**args)
    except Exception as e:
        r = dict(ok=False, syy=repr(e))
    if irr is not None:
        try:
            r["irr_tarkistus"] = nv.tarkista_irroitus(**irr, kappa=(r.get("kappa") if r.get("kappa_ok") else None))
        except Exception as e:
            r["irr_tarkistus"] = dict(virhe=repr(e))
    return r


def nv_omega_oletus():
    """Kierre (rad/s) irroituksessa, kun mittausta ei ole: netti_vakiot.OMEGA_OLETUS."""
    import netti_vakiot as nv
    return nv.OMEGA_OLETUS[0]


def vakiot_argumentit(kauko, kauko_kierto, kierre, hog2_f, fps):
    """netti_vakiot.sovita:n argumentit (dict) ja kierteen pakotteen lahde ("kauko" | "kierre" | None) heiton kaukopaan seurannasta (h.kauko), kaukopaan kierron mittauksesta
    (h.kauko_kierto), yhdistetysta kierteesta (h.kierre) ja hog-ylityksesta (h.hog2_f). Rata: aikaperustan aika; katkaistaan pysahtymiseen (+ 0,5 s), tormayksen alkuun (- 0,2 s) tai
    hukatulla kivella viimeiseen B-paikkaan (+ 0,3 s). Pakote: varma kaukopaan mittaus, muuten varma yhdistetty kierre (omega hogilla), muuten ei pakotetta."""
    P = np.asarray(kauko["pisteet"], float)
    F, t = P[:, 0], np.asarray(kauko["t"], float)
    t_f = lambda f: float(np.interp(f, F, t))
    pys = kauko["tila"] == "pysahtyi"
    t_loppu = t_f(kauko["f_pys"]) + 0.5 if pys and kauko.get("f_pys") is not None else None
    if not pys and kauko.get("f_viim_b") is not None:                  # hukattu kivi: viimeisen B-paikan jalkeiset A-paikat voivat olla harjaajien nauhoja
        t_loppu = t_f(kauko["f_viim_b"]) + 0.3
    if kauko.get("tormays"):                                           # tormayksen jalkeen rata ei noudata mallia
        t_t = t_f(kauko["tormays"]["f"]) - 0.2
        t_loppu = t_t if t_loppu is None else min(t_loppu, t_t)
        pys = False
    omega, lahde = None, None
    mf = nk.omega_mitta(kauko_kierto, fps)
    if mf is not None and mf[3]:                                       # varma mittaus kaukopaasta
        omega, lahde = (t_f(mf[0] * fps), abs(mf[1]), mf[2]), "kauko"
    elif kierre is not None and kierre.get("varma") and hog2_f is not None:          # yhdistetty kierre (omega hogilla): mittaus + vaimeneminen
        om2 = abs(float(np.radians(kierre["omega2_deg"])))
        omega, lahde = (t_f(hog2_f), om2, max(0.15, 0.15 * om2)), "kierre"
    return dict(t=t, X=P[:, 1], Y=P[:, 2], vahva=np.asarray(kauko["vahva"], bool), omega=omega, kaarto_merkki=None, t_loppu=t_loppu, pysahtynyt=pys), lahde


def hog_ikkuna(t, X, Y, y0=None, ikkuna=None):
    """Kiven nopeus (m/s) ja suunta (astetta, kulma Y-akseliin) hog-viivan kohdalla suoraan radasta: Y(t) ja X(Y) suorat paikoista |Y - y0| <= ikkuna. Hog-viivan jalkeen kivi on
    varmasti vapaassa lennossa (irroitus tapahtuu ennen viivaa), joten tama on irroituksen tunnistuksesta riippumaton mittaus (vertailu: nopeus_jalkeen >= nopeus_hog1,
    suunta_jalkeen ~ suunta_hog1). Palauttaa dict(v_m_s, suunta_deg, n, Y_keski) tai None (alle 8 paikkaa tai alle 100 cm:n matka)."""
    y0 = HOG_KESKI_CM if y0 is None else y0
    ikkuna = HOG_IKKUNA_CM if ikkuna is None else ikkuna
    m = np.abs(Y - y0) <= ikkuna
    if m.sum() < 8 or float(np.ptp(Y[m])) < 100.0:
        return None
    cy = np.polyfit(t[m], Y[m], 1)
    cx = np.polyfit(Y[m], X[m], 1)
    return dict(v_m_s=float(cy[0] * np.sqrt(1.0 + cx[0] ** 2) / 100.0), suunta_deg=float(np.degrees(np.arctan(cx[0]))), n=int(m.sum()), Y_keski=float(np.mean(Y[m])))


HOG_ALUE_CM = (590.0, 690.0)     # hog-viivan hairioalue (kiven keskipisteen Y): B-sovituksen Y-harha, joka vaihtelee heitoittain (1. videolla ~1 cm ja taulukolla korjattu, 2. videolla viivan ollessa ohut ja sininen
                                 # 0-4 cm kapeana kohoumana kiven etureunan ylittaessa viivan: nopeus nousee ja laskee takaisin, mika nayttaa tyonnon loppuna ja siirsi irroituksen 40-60 %:lla heitoista hog-viivalle)
HOG_ALUE_ENNEN_CM = 100.0        # tasoitus: alueen ylapuolelta (ennen sita) tama matka ja ...
HOG_ALUE_JALKEEN_CM = 30.0       # ... alueen jalkeen tama matka ovat reunaikkunat, jotka sovitetaan yhteen vapaan lennon paraabeliin (hidastuvuus A_VAPAA_M_S2)
HOG_ALUE_MIN_ENNEN, HOG_ALUE_MIN_JALKEEN = 8, 3
HOG_ALUE_RMS_CM = 1.0            # ... ja jos reunaikkunat sopivat siihen (rms alle taman), alueen naytteet korvataan paraabelilla: kivi ei voi saada ja menettaa nopeutta hog-viivalla, joten alueella ei ole tietoa
                                 # tyonnosta. Jos reunat eivat sovi (tyonto jatkuu alueella: nopeus alueen jalkeen suurempi), data jaa ennalleen.

def tasoita_hog_alue(t, X, Y):
    """(X, Y, tasoitettu): hog-viivan hairioalueen (HOG_ALUE_CM) B-sovituksen harha pois irroituksen tunnistuksesta. Alueen ylapuolen ja jalkeisen reunaikkunan naytteet sovitetaan yhdella vapaan lennon
    paraabelilla (Y = c0 + c1 t - 0,5 a t^2, a = liuku_irroitus.A_VAPAA_M_S2; X suoralla); jos sovite sopii reunoihin (rms <= HOG_ALUE_RMS_CM), alueen naytteet korvataan sen arvoilla: alueella
    ei ole nopeuden muutosta, joten mikaan siella nakyva nopeuden nousu ja lasku on mittausharhaa. Jos reunat eivat sovi yhteen (tyonto jatkuu alueella) tai reunanaytteita on liian vahan (rata ei yllä
    alueen yli), rata palautetaan ennallaan (tasoitettu = False)."""
    t, X, Y = np.asarray(t, float), np.asarray(X, float), np.asarray(Y, float)
    z0, z1 = HOG_ALUE_CM
    alue = (Y >= z0) & (Y <= z1)
    ennen = (Y >= z0 - HOG_ALUE_ENNEN_CM) & (Y < z0)
    jalkeen = (Y > z1) & (Y <= z1 + HOG_ALUE_JALKEEN_CM)
    if not alue.any() or int(ennen.sum()) < HOG_ALUE_MIN_ENNEN or int(jalkeen.sum()) < HOG_ALUE_MIN_JALKEEN:
        return X, Y, False
    reuna = ennen | jalkeen
    a = 100.0 * li.A_VAPAA_M_S2                                                    # cm/s2
    tm = float(t[reuna].mean())
    d = t[reuna] - tm
    M = np.column_stack([np.ones(len(d)), d])
    y = Y[reuna] + 0.5 * a * d * d
    try:
        c = np.linalg.lstsq(M, y, rcond=None)[0]
        cx = np.polyfit(d, X[reuna], 1)
    except np.linalg.LinAlgError:
        return X, Y, False
    if float(np.sqrt(np.mean((y - M @ c) ** 2))) > HOG_ALUE_RMS_CM:
        return X, Y, False
    da = t[alue] - tm
    Yt, Xt = Y.copy(), X.copy()
    Yt[alue] = c[0] + c[1] * da - 0.5 * a * da * da
    Xt[alue] = np.polyval(cx, da)
    return Xt, Yt, True


def korjaa_hog_harha(Y):
    """Kiven Y (cm) ilman B-sovituksen toistuvaa, paikkaan sidottua harhaa (HOG_HARHA_CM): taulukon kohouma vahennetaan. Tulos ei muuta heiton fysiikkaa, vain poistaa mittausvirheen, joka
    muuten nayttaa 'tyonnon loppuna' (ks. HOG_HARHA_CM)."""
    Y = np.asarray(Y, float)
    if not HOG_HARHA_CM:
        return Y
    ys, e = np.array(HOG_HARHA_CM, float).T
    return Y - np.interp(Y, ys, e, left=0.0, right=0.0)


def valmistele_rata(f, X, Y, fps_lahde=25.0, fps_ulos=30.0, lisa_f=None, korjaa=True, tiedot=None):
    """Kelvollisten (ruutu, X, Y) -naytteiden esikasittely viivaestimaattia varten: jarjestys, aikaperusta (aikaperusta.Aikajana: lahdevirran kopioruudut pois, todellinen aika) ja (korjaa)
    B-sovituksen hog-viivan harhan korjaus (korjaa_hog_harha). Palauttaa (t, X, Y, t_lisa); t_lisa = lisa_f:n ruutujen ajat (kierteen mittauksen ruudut) tai None.
    tiedot (dict): saa radan lahderuutujen taajuuden R (aikaperusta.arvioi_lahdetaajuus: 25 tai 30 r/s radan omasta liikkeesta)."""
    aikap.FPS_LAHDE, aikap.FPS_ULOS = float(fps_lahde), float(fps_ulos)
    f = np.asarray(f, float)
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    o = np.argsort(f)
    f, X, Y = f[o], X[o], Y[o]
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        aj = aikap.Aikajana(f, Y)
        if tiedot is not None:
            tiedot["R"] = float(aj.R)
        mk = np.array([int(fi) not in aj.kopio for fi in f])
        f, X, Y = f[mk], X[mk], Y[mk]
        t = aj.aika(f)
        t_lisa = [float(x) for x in aj.aika(np.asarray(lisa_f, float))] if lisa_f is not None and len(lisa_f) else None
    return t, X, (korjaa_hog_harha(Y) if korjaa else Y), t_lisa


def laske_viivat(f, X, Y, fps_lahde=25.0, fps_ulos=30.0, lisa_f=None):
    """Viivaestimaatti kelvollisista (ruutu, X, Y) -naytteista: aikaperusta (aikaperusta.Aikajana) + liu'un ja irroituksen tunnistus (liuku_irroitus). Palauttaa
    dict tai None. Kaytetaan sellaisenaan tyoprosessissa. fps_lahde = erillisten ruutujen nimellistaajuus (25 tassa videossa: 25 r/s -> 30 r/s kopioruuduilla),
    fps_ulos = videon ruutunopeus. lisa_f: muut ruutunumerot (kierteen mittauksen ruudut), joille palautetaan sama aika (t_lisa)."""
    if len(f) < 30 or np.max(np.asarray(Y, float)) < ESTIMAATTI_MIN_Y_CM:
        return None
    try:
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            tiedot = {}
            t, X, Y, t_lisa = valmistele_rata(f, X, Y, fps_lahde, fps_ulos, lisa_f, tiedot=tiedot)
            Xa, Ya, tiedot["hog_tasoitettu"] = tasoita_hog_alue(t, X, Y)               # irroitus tunnistetaan radasta, jonka hog-viivan hairioalue on tasoitettu (tarvittaessa); hog-nopeus mitataan raa'asta radasta
            r = li.analysoi(dict(t=t, X=Xa, Y=Ya), **ESTIMAATTI)
    except Exception as e:                              # liian vahva / outo aineisto: ei estimaattia
        return dict(virhe=repr(e))
    if r.get("irroitus") is None or r.get("tila") is None:
        return None
    ti, ys, la, ir = r["tila"], r["yhteenveto"], r["liuku"], r["irroitus"]
    vapaa_Y = float(Y.max() - ti["Y_cm"])
    hg = hog_ikkuna(t, X, Y)
    return dict(
        liuku_a=ys["liun_suora_X_Y0_cm"], liuku_b=ys["liun_suora_kulmakerroin"], liuku_suunta_deg=ys["liun_suunta_deg"],
        liuku_Y0=ys["liun_suunta_alku_cm"], liuku_Y1=ys["liun_suunta_loppu_cm"],
        irr_X=ti["X_cm"], irr_Y=ti["Y_cm"], irr_suunta_deg=ti["suunta_deg"], irr_virhe_deg=ti["suunta_virhe_deg"],
        irr_Y_vali=(ir["Y_min"], ir["Y_max"]), merkitseva=bool(ir["merkitseva"]), dchi2=float(ir["dchi2_med"]),
        vapaa_matka_cm=vapaa_Y, vapaa_ruutuja=int(ti["ruutuja"]), Y_max=float(Y.max()), t_lahto=float(la["t_lahto"]),
        nopeus_ennen=ys["nopeus_ennen_m_s"], nopeus_jalkeen=ys["nopeus_jalkeen_m_s"],
        liuku_hidastuvuus=ys["liun_hidastuvuus_m_s2"], vapaa_hidastuvuus=ys["vapaan_lennon_hidastuvuus_m_s2"],
        kaarto=ti["kaarto"], kaarevuus_1_cm=float(ti["kaarevuus_1_cm"]), t_irr=float(ti["t_r"]), t_lisa=t_lisa, n=int(len(t)),
        nopeus_hog1=(hg["v_m_s"] if hg else None), suunta_hog1_deg=(hg["suunta_deg"] if hg else None), hog1_Y_keski=(hg["Y_keski"] if hg else None), lahdetaajuus=tiedot.get("R"), hog_tasoitettu=bool(tiedot.get("hog_tasoitettu")))


# ---------------------------------------------------------------------------------------------------------------------
# HEITTO JA HEITTOPAA
# ---------------------------------------------------------------------------------------------------------------------
def tviiva_ruutu(pts, Y_raja=KELLO_TVIIVA_Y_CM, ikkuna=KELLO_TVIIVA_IKKUNA_CM):
    """Ruutu (murtoluku), jolla rata ylittaa Y = Y_raja (heittopaan T-viiva: kiven etureuna osuu viivaan, kun keskipiste on Y = -KIVI_R_CM). pts = [(ruutu, Y, kivi todettu)]. Ruutunumero
    sovitetaan Y:n funktiona (toisen asteen polynomi, koska kivi kiihtyy tyontoliikkeessa; liian vahalla aineistolla suora) viivan ymparilta: ensin vain todetut kivipaikat (B), sitten kaikki
    paikat, sitten laajempi ikkuna; karkeat pisteet (A:n harhat) karsitaan. Vaatii pisteita viivan molemmin puolin (interpolointi); None, jos niita ei ole."""
    if len(pts) < 4:
        return None
    F = np.array([p[0] for p in pts], float)
    Y = np.array([p[1] for p in pts], float)
    K = np.array([bool(p[2]) for p in pts])
    for vain_b, raja in ((True, ikkuna), (False, ikkuna), (False, 1.4 * ikkuna)):
        m = (np.abs(Y - Y_raja) <= raja) & (K if vain_b else True)
        if m.sum() < (8 if raja <= ikkuna else 5) or int((Y[m] < Y_raja - 3.0).sum()) < 2 or int((Y[m] > Y_raja + 3.0).sum()) < 2:
            continue
        f, y = F[m], Y[m]
        deg = 2 if len(f) >= 8 else 1
        for _ in range(2):
            c = np.polyfit(y, f, deg)
            r = f - np.polyval(c, y)
            hyva = np.abs(r) <= 3.0 * max(1.4826 * float(np.median(np.abs(r - np.median(r)))), 0.5) + 1.0
            if hyva.all() or hyva.sum() < (8 if deg == 2 else 4):
                break
            f, y = f[hyva], y[hyva]
        f_yl = float(np.polyval(c, Y_raja))
        if f.min() - 2.0 <= f_yl <= f.max() + 2.0:
            return f_yl
    return None


class Naytteet:
    """Yksi yksilollinen ruutu: A (kahvan sentroidi) ja B (tarkka sovitus), maakoordinaatit cm."""
    __slots__ = ("f", "A", "ala", "B", "Bs", "Bk", "rms", "kelv", "peitto", "odottaa", "poik", "lepo")

    def __init__(self, f, A=None, ala=0):
        self.f, self.A, self.ala = f, A, ala
        self.B, self.Bs, self.rms, self.kelv, self.peitto, self.odottaa = None, None, None, None, None, False
        self.Bk = None                         # levossa: graniittisovitus (kivi todettu) kauempana kuin B_LEPO_POIK_CM viritetysta kahvasta: kivi on muualla kuin kahva
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
        self.kauko = None                      # kaukopaan seuranta (KaukoSeuranta._valmis): dict(tila, paikat, pysahtyminen, hog-ylitys ...) tai None
        self.vakiot = None                     # kiven vakiot (netti_vakiot.sovita): kitkataso A ja kaarto kappa kaukopaan radasta, tai dict(ok=False, syy=...)
        self.vakiot_tila = None                # None | "kaynnissa" | "valmis" | "virhe"
        self.vakiot_omega_lahde = None         # kierteen pakotteen lahde vakioiden sovituksessa: "kauko" | "kierre" | None
        self.kauko_kierto = None               # kierteen mittaus kaukopaan kuvasta (netti_kierre.arvioi) tai dict(virhe=...)
        self.kauko_kierto_tila = None          # None | "kaynnissa" | "valmis" | "virhe"
        self.kierre = None                     # kierrosmaara hog-hog-valilla (_paivita_kierre)
        self.ala_med = None                    # kahvan normitettu ala (px / skaala^2): kaukopaan blobien tunnistus
        self._tviiva = (None, None)            # T-viivan ylitysruudun valimuisti: (avain, ruutu)

    def kelvolliset_b(self):
        return [n for n in self.naytteet if n.B is not None]

    def tviiva_f(self):
        """Ruutu (murtoluku), jolla kiven etureuna osui heittopaan T-viivaan (tviiva_ruutu), tai None, jos rata ei (viela) ylita viivaa. Valimuisti: lasketaan uudelleen vain, kun naytteet
        tai B-tulokset muuttuvat."""
        avain = (len(self.naytteet), self.b_laskuri)
        if self._tviiva[0] == avain:
            return self._tviiva[1]
        pts = []
        for n in list(self.naytteet):
            p = n.paikka()
            if p is not None:
                pts.append((n.f, p[1], n.kivi()))
        f = tviiva_ruutu(pts)
        self._tviiva = (avain, f)
        return f

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


class _Seuranta:
    """Seurannan yhteiset apurit: heittopaa (Heittopaa) ja kaukopaan seuranta (KaukoSeuranta) kayttavat samaa ennustetta, A-valintaa ja konsolidointia. Edellyttaa kentat an, m, vari,
    heitto (naytteet, lepo, tila), ala_norm ja _ennuste_luotettava."""

    def _a_maa(self, c):
        return self.m.oikaistu_maahan(c[0], c[1])

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

    def _vari_a(self, ruutu, ennuste_XY, gate_cm, tarkka_ala=True, ala_raja=(0.4, 2.2), ohita=None):
        """A-mittaus: kahva ennusteen ymparilta; hyvaksytaan ennusteeseen lahin alaltaan (ala_raja * odotettu ala) ja sijainniltaan sopiva blobi. ohita: [(X, Y)] paikallaan olevat
        kivet / esineet (kartta ennen heittoa): taman SAAP_KARTTA_CM:n sisalla olevat blobit hylataan. Palauttaa (X, Y, ala, skaala) tai None."""
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
            if ohita and any(np.hypot(X - q[0], Y - q[1]) <= SAAP_KARTTA_CM for q in ohita):
                continue
            return X, Y, ala, s
        return None

    def _kierto_leike(self, h, n, ruutu, X0, Y0):
        """Leike oikaistusta kuvasta kiven akselin ymparilta kierteen mittausta varten (muovikartat lasketaan heiton jalkeen taustaprosessissa)."""
        u, v = self.m.maa_oikaistuun(X0, Y0, self.m.ast["z0_cm"])
        uc, vc = int(round(float(u[0]))), int(round(float(v[0])))
        roi, cx0, cy0 = self.m.kuvain.leike(ruutu, uc - KIERTO_LEIKE_R, vc - KIERTO_LEIKE_R, uc + KIERTO_LEIKE_R, vc + KIERTO_LEIKE_R)
        if roi is not None:
            h.kierto_kehykset.append((n, roi, cx0, cy0))


class Heittopaa(_Seuranta):
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
        self.hack_skaala2 = float(self.m.skaala(0.5 * (HACK_X[0] + HACK_X[1]), 0.5 * (HACK_Y[0] + HACK_Y[1]))) ** 2          # px^2 / cm^2 hackin kohdalla

    # ---- apurit --------------------------------------------------------------------------------------------------
    @staticmethod
    def _hack_sisalla(X, Y):
        return HACK_X[0] <= X <= HACK_X[1] and HACK_Y[0] <= Y <= HACK_Y[1]

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
            for c in self.m.kahva(ruutu, self.hack_alue, vari, None, min_ala=max(int(KAHVA_ALA_CM2[0] * self.hack_skaala2) // 2, 4), taustaero=False):
                if not (KAHVA_ALA_CM2[0] * self.hack_skaala2 <= c[2] <= KAHVA_ALA_CM2[1] * self.hack_skaala2):
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
        """(sop, kauk): levossa viritetyn kahvan B-sovitukset viimeisen LEPO_TODISTE_RUUTUA ruudun ajalta, joissa kivi on kahvan kohdalla (sop; <= B_LEPO_POIK_CM) tai graniittia on mutta kahvan paikasta kauempana (kauk)."""
        sop = h.kivi_n(LEPO_TODISTE_RUUTUA, f)
        kauk = sum(1 for n in h.naytteet[-300:] if f - n.f <= LEPO_TODISTE_RUUTUA and n.lepo and n.peitto is not None and n.peitto >= B_LEPO_PEITTO and not n.kivi() and n.poik is not None and n.poik > B_LEPO_POIK_CM)
        return sop, kauk

    def _validoitu(self, h, f):
        """Viritetty kahva on kiven kahva: levossa vahintaan KIVI_B_LEPO kelvollista B-sovitusta kahvan kohdalla (<= B_LEPO_POIK_CM), eika kahvan vierestä ole löytynyt kiveä
        useammin kahvan kohdalta (kiven viereinen esine, esim. heittäjän kenkä: kiven B-sovitus osuu kahvasta 12-27 cm päähän). Todiste on tuore (LEPO_TODISTE_RUUTUA): kantamisen ja
        laskemisen aikana kiven ohi kulkeva kahvaehdokas ei todennu ikuisesti; todettu kahva pysyy todettuna, kun kahva on hetken piilossa (kasi), mutta peruuntuu, jos tuoreet
        sovitukset osoittavat kiven olevan muualla (KIVI_B_LEPO kertaa ja useammin kuin kahvan kohdalla)."""
        sop, kauk = self._lepo_todisteet(h, f)
        if h.validoitu:
            return not (kauk >= KIVI_B_LEPO and kauk > sop)
        return sop >= KIVI_B_LEPO and kauk <= sop

    def _kiven_keskipiste(self, h, f):
        """Levossa olevan kiven keskipiste: viimeisten LEPO_KESKI_N graniittisovituksen (kahvan kohdalla tai kaukana; ei A) mediaani LEPO_TODISTE_RUUTUA ruudun ajalta, kun ne ovat
        enintaan LEPO_KESKI_HAJONTA_CM mediaanista; muuten None (kivea ei nay tai se liikkuu: kantaminen, kasi)."""
        P = [n.B or n.Bs or n.Bk for n in h.naytteet[-200:] if n.lepo and f - n.f <= LEPO_TODISTE_RUUTUA and (n.B or n.Bs or n.Bk)]
        if len(P) < LEPO_KESKI_N:
            return None
        P = np.array(P[-LEPO_KESKI_N:], float)
        c = np.median(P, axis=0)
        return c if float(np.max(np.hypot(P[:, 0] - c[0], P[:, 1] - c[1]))) <= LEPO_KESKI_HAJONTA_CM else None

    def _vaihda_kandidaatti(self, f, ruutu):
        """Toinen kandidaatti (toinen vari tai toinen paikka hack-alueella) on ollut vakaana: virityksen kohde voi vaihtua. Kahva on kiven keskella, joten levossa olevan kiven
        keskipisteen (_kiven_keskipiste) perusteella kandidaatti vaihtuu, jos se on enintaan B_LEPO_POIK_CM keskipisteesta ja vahintaan VAIHTO_PARANNUS_CM lahempana kuin viritetty kahva.
        Ilman keskipistetta (kiven todetaan vasta) vaihto vain, jos viritetyn kahvan luona ei ole kivea (ei todettu, eika vierella ole kivea useammin kuin kahvan kohdalla)."""
        h = self.heitto
        X0, Y0 = h.lepo
        c = self._kiven_keskipiste(h, f)
        if c is None:
            if h.validoitu:
                return False                                        # kivi todettu, mutta ei nay levossa (kasi peittaa): viritys pysyy
            sop, kauk = self._lepo_todisteet(h, f)
            if kauk < sop:                                          # kahvan kohdalla on kivea useammin kuin vieressa: oikea kivi, jonka B-sovitukset vain viipyvat (kasi peittaa)
                return False
        k = self._vakaa_kandidaatti(f, ruutu, (self.vari, X0, Y0))
        if k is None:
            return False
        if c is not None:
            kd, ad = float(np.hypot(k[2][0] - c[0], k[2][1] - c[1])), float(np.hypot(X0 - c[0], Y0 - c[1]))
            if not (kd <= B_LEPO_POIK_CM and kd + VAIHTO_PARANNUS_CM <= ad):
                return False
        self.an.tapahtumat.append(("vaihto", h.tunnus, f, f"{self.vari} ({X0:.0f},{Y0:.0f}) -> {k[0]} ({k[2][0]:.0f},{k[2][1]:.0f})"))
        self._virita_kandidaatista(f, ruutu, *k)
        return True

    def _armed(self, f, ruutu):
        h = self.heitto
        h.validoitu = self._validoitu(h, f)
        if self.n_ruutu % IDLE_VALI == 0 and (f - h.naytteet[0].f) / self.an.fps >= VAIHTO_S and self._vaihda_kandidaatti(f, ruutu):
            return
        X0, Y0 = h.lepo
        r = self._vari_a(ruutu, (X0, Y0), 35.0, tarkka_ala=False)
        if r is None:
            if h.validoitu:                                     # kivi todettu levossa, mutta kahva on piilossa (heittajan kasi): kivi pysyy viritettyna, kunnes sen runkokaan ei nay levossa
                if self.n_ruutu % B_LEPO_VALI == 0:
                    n = Naytteet(f)
                    n.lepo = True
                    h.naytteet.append(n)
                    self.an.pyyda_b(self, h, n, ruutu, X0, Y0)
                if any(m.kivi() and f - m.f <= HUKKA_LEPO_S * self.an.fps for m in h.naytteet[-60:]):
                    self.f_nahty = f
                    return
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
        lavea = False
        if a is None and h.tila == "liikkuu":
            a = self._vari_a(ruutu, (Xp, Yp), portti, ala_raja=(0.4, LIIKE_ALA_MAX))
            ed = next((m for m in reversed(h.naytteet[-60:]) if m.paikka() is not None), None)
            if a is None and ed is not None and f - ed.f >= HACK_HUKKA_RUUTUA and ed.paikka()[1] < HACK_HUKKA_Y_CM:
                a = self._vari_a(ruutu, ed.paikka(), HACK_HUKKA_PORTTI_CM, ala_raja=(0.4, LIIKE_ALA_MAX))
            lavea = a is not None
        n = Naytteet(f)
        if a is not None:
            n.A, n.ala = (a[0], a[1]), a[2]
            a_n = a[2] / (a[3] * a[3])
            if lavea:
                pass                                                       # sulautunut blobi: ei odotetun alan paivitykseen
            elif len(self.ala_norm) < 3 or ALA_VAIHTELU[0] <= a_n / float(np.median(self.ala_norm)) <= ALA_VAIHTELU[1]:
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
            if viim and p[1] >= VAHVISTUS_Y_CM and v >= VAHVISTUS_V_CM_S and h.kivi_n(10 ** 6, f) >= KIVI_B_VAHVISTUS and (v >= VAHVISTUS_V_VARMA_CM_S or h.kivi_n(10 ** 6, f) >= KIVI_B_VAHVISTUS_HIDAS):
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
class KaukoHeitto:
    """Kaukopaan kiven seuranta heittona: B-hyvaksynta (Analyysi._liita_b), ennuste (_Seuranta) ja kierron leikkeet kayttavat samaa rajapintaa kuin Heitto (naytteet, tila,
    b_laskuri, kierto_kehykset)."""

    def __init__(self, tunnus, puoli, vari):
        self.tunnus, self.puoli, self.vari = tunnus, puoli, vari
        self.tila = "vahvistettu"              # B hyvaksytaan ennusteen / A:n suhteen (ei lahdon jalkeista ulottumaa)
        self.naytteet = []
        self.b_laskuri = 0
        self.lepo = np.zeros(2)                # ennusteen varapaikka: viimeksi tunnettu paikka
        self.kierto_kehykset = []


class _Rata(list):
    """Kaukopaan blobirata [(f, X', Y', ala), ...] ja sen naytteet (A; B-sovitukset liitetaan myohemmin): rata, jonka kivi todetaan B:lla, muuttuu kaukopaan seurannaksi."""

    def __init__(self, tunnus, puoli, vari):
        super().__init__()
        self.heitto = KaukoHeitto(tunnus, puoli, vari)
        self.f_b = -10 ** 9                               # viimeisimman B-pyynnon ruutu

    def lisaa(self, q):
        self.append(q)
        self.heitto.naytteet.append(Naytteet(q[0], (q[1], q[2]), q[3]))


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
        P2 = np.array([(x, y, z) for x in (-250.0, 250.0) for y in (-260.0, SAAP_Y_CM[1]) for z in (0.0, 22.0)])
        u2, v2 = kivimalli.project_3d(self.m.cam.K, self.m.cam.R, self.m.cam.t, P2)
        self.alue_kartta = (int(u2.min()) - 10, int(v2.min()) - 10, int(u2.max()) + 10, int(v2.max()) + 10)       # pelialue (kivikartta ennen heittoa)
        self.karttat = []                                 # [(ruutu, [(X', Y')])]: paikallaan olevat saman varin kahvat pelialueella
        self._kartta_f = -10 ** 9
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

    def _kartoita(self, f, ruutu):
        """Pelialueen saman varin kahvat (A-taso, taustasta poikkeavat): kivet, jotka ovat paikallaan ennen kuin heitetty kivi saapuu (kaukopaan seuranta ei saa tarttua niihin)."""
        h = self.h
        out = []
        try:
            blobit = self.m.kahva(ruutu, self.alue_kartta, h.vari, None, min_ala=25, taustaero=True)
        except Exception:
            blobit = []
        for u, v, ala, bb in blobit:
            if not (0.35 <= bb[2] / max(bb[3], 1) <= 2.8):
                continue
            X, Y = self.m.oikaistu_maahan(u, v)
            if abs(X) > 260.0 or Y > SAAP_Y_CM[1] or Y < -270.0:
                continue
            if h.ala_med:
                s = self.m.skaala(X, Y)
                if not (0.3 * h.ala_med * s * s <= ala <= 3.0 * h.ala_med * s * s):
                    continue
            out.append((float(X), float(Y)))
        self.karttat.append((f, out))
        self.karttat = self.karttat[-4:]

    def kartta_ennen(self, f_ensimmainen):
        """Viimeisin kartta, joka on otettu ennen kuin liikkuva rata ilmestyi (>= 20 ruutua ennen radan ensimmaista pistetta); muuten vanhin."""
        for fs, pos in reversed(self.karttat):
            if fs <= f_ensimmainen - 20:
                return list(pos)
        return list(self.karttat[0][1]) if self.karttat else []

    def kasittele(self, f, ruutu):
        """Yksi (yksilollinen) ruutu; palauttaa False kun haku on paattynyt (kivi tunnistettu: kaukopaan seuranta alkaa, tai aika loppui)."""
        h = self.h
        if f < self.f_alku:
            return True
        if f > self.f_loppu or not self.ok:
            h.saapuminen = "ei_loytynyt"
            return False
        if f - self._kartta_f >= SAAP_KARTTA_VALI:
            self._kartta_f = f
            self._kartoita(f, ruutu)
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
                if not (SAAP_ALA_RAJAT[0] * odotus <= ala <= SAAP_ALA_RAJAT[1] * odotus):
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
            if paras is None:
                paras = _Rata(h.tunnus, self.puoli, h.vari)
                self.radat.append(paras)
            paras.lisaa(q)
        self.radat = [r for r in self.radat if f - r[-1][0] <= 20]
        # B (asynkroninen) vain kulkevan radan uusimmalle pisteelle: kivella on graniittirunko (peitto 0,34-0,95), sweeperin kasivarsinauhalla ei (0,06-0,28)
        for r in self.radat:
            if len(r) >= 3 and r[-1][0] == f and f - r.f_b >= SAAP_B_VALI and self._kulkee(r):
                r.f_b = f
                self.an.pyyda_b(self, r.heitto, r.heitto.naytteet[-1], ruutu, r[-1][1], r[-1][2])
        for r in self.radat:
            if len(r) >= SAAP_PISTEITA_B and sum(1 for q in r.heitto.naytteet if q.kivi()) >= SAAP_B_HYVAKSY and self._kulkee(r):
                self._hyvaksy(r)
                return False
        return True

    def _kulkee(self, r):
        """Radan viimeiset pisteet etenevat tasaisesti kohti pesaa (40-450 cm/s, ei sivuttaisliikettaa): kulkeva kivi (tai sen mukana kulkeva esine), ei paikallaan oleva kahva."""
        n = min(len(r), 25)
        if n < 3:
            return False
        F = np.array([p[0] for p in r[-n:]], float)
        X = np.array([p[1] for p in r[-n:]], float)
        Y = np.array([p[2] for p in r[-n:]], float)
        if F[-1] - F[0] < 3:
            return False
        c = np.polyfit(F, Y, 1)
        v = -c[0] * self.an.fps
        if not (40.0 <= v <= 450.0) or float(np.sqrt(np.mean((Y - np.polyval(c, F)) ** 2))) > 8.0:
            return False
        return abs(np.polyfit(F, X, 1)[0]) <= 0.5 * abs(c[0])

    def _hyvaksy(self, r):
        """Kivi tunnistettu: hog-ylitys A-radasta (jos kivi on jo ylittanyt viivan; tarkentuu seurannassa) ja kaukopaan seuranta kayntiin."""
        h = self.h
        h.saapuminen = "loytyi"
        f_yl = self._ylitys(r)
        if f_yl is not None:
            h.hog2_f = f_yl
            self.an._paivita_kierre(h)
        self.an.kaukoseurannat.append(KaukoSeuranta(self.an, h, r, self.kartta_ennen(r[0][0])))

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


def pysahtymishetki(F, P, fps):
    """Radasta (ruudut F, paikat P (n, 2)) pysahtymisruutu ja lepopaikka: lepopaikka on viimeisten paikkojen mediaani; pysahtymishetki on ensimmainen hetki, josta alkaen silotettu (7 paikan
    liukuva mediaani) etaisyys lepopaikasta pysyy levon kohinatason sisalla (raakojen etaisyyksien 95 %:n fraktiili viimeisen 1,5 s:n ajalta; vahintaan KAUKO_PYS_RAJA_CM)
    seuraavat 2 s (tai radan loppuun), + KAUKO_PYS_LISA_S. (Paikat ovat B- ja A-paikkoja; kohina levossa on useita senttimetreja, joten kiinteaa 1,5 cm:n rajaa ei voi kayttaa; pitkan levon
    kohinahuiput eivat siirra hetkea myohemmaksi.)"""
    n = len(F)
    k = min(n, max(8, int(round(1.5 * fps))))
    lepo = np.median(P[-k:], axis=0)
    d = np.hypot(*(P - lepo).T)
    ds = np.array([float(np.median(d[max(0, i - 3):i + 4])) for i in range(n)])      # silotus: liukuva mediaani (7)
    raja = max(KAUKO_PYS_RAJA_CM, float(np.percentile(d[-k:], 95)))             # levon kohinataso: raakojen etaisyyksien 95 %:n fraktiili viimeisen 1,5 s:n ajalta
    i0 = n - 1
    for i in range(n):
        ikk = ds[(F >= F[i]) & (F <= F[i] + 2.0 * fps)]
        if float(ikk.max()) <= raja:
            i0 = i
            break
    f_pys = min(float(F[i0]) + KAUKO_PYS_LISA_S * fps, float(F[-1]))
    return f_pys, float(lepo[0]), float(lepo[1])


def kiihtymiskohta(F, P, fps, K=10):
    """Radan (ruudut F, paikat P (n, 2)) ensimmainen kohta, jossa hidastunut kivi (nopeus < KAUKO_KIIHTYY_HIDAS_CM_S) alkaa selvasti liikkua (nopeus kasvaa KAUKO_KIIHTYY_CM_S ja > KAUKO_KIIHTYY_SUHDE x
    edellisen 1,5 s:n pienimmasta): kivi ei voi kiihtya (vapaan liu'un aikana), joten sen jalkeinen rata on toista kohdetta (esim. pesan taakse mennytta kivea kuljetetaan kasin pois). Nopeus
    10 ruudun (0,4 s) erotuksesta (kohina +-3 cm / paikka). Palauttaa naytteen indeksin tai None."""
    n = len(F)
    if n < 3 * K:
        return None
    v = np.array([float(np.hypot(*(P[i + K] - P[i]))) / max((F[i + K] - F[i]) / fps, 1e-3) for i in range(n - K)])
    fk = np.array([0.5 * (F[i] + F[i + K]) for i in range(n - K)])
    for i in range(1, len(v)):
        m = (fk[:i] >= fk[i] - 1.5 * fps)
        if not m.any():
            continue
        vmin = float(v[:i][m].min())
        if vmin < KAUKO_KIIHTYY_HIDAS_CM_S and v[i] > vmin + KAUKO_KIIHTYY_CM_S and v[i] > KAUKO_KIIHTYY_SUHDE * max(vmin, 5.0):
            return int(i)                                   # ikkunan i alku: viimeinen ehja paikka
    return None


def poistumisruutu(F, P, fps):
    """Ruutu (float), jolla kivi poistui pelista (keskipiste kokonaan taka- tai sivuviivan ulkopuolella, ja pysyy siella >= 3 seuraavaa paikkaa), radan kaantopisteesta interpoloituna;
    None, jos kivi pysyi pelissa."""
    n = len(F)
    ulos = (P[:, 1] < PELI_TAKA_Y_CM) | (np.abs(P[:, 0]) > PELI_SIVU_X_CM)
    for i in range(n - 3):
        if ulos[i] and ulos[i + 1:i + 4].all():
            if i == 0:
                return float(F[0])
            y0, y1 = P[i - 1, 1], P[i, 1]                  # lineaarinen interpolointi taka-viivan ylitykseen (sivuviivan ylitys: lahimpaan naytteeseen)
            if P[i, 1] < PELI_TAKA_Y_CM <= y0 and y0 != y1:
                return float(F[i - 1] + (F[i] - F[i - 1]) * (y0 - PELI_TAKA_Y_CM) / (y0 - y1))
            return float(F[i])
    return None


class KaukoSeuranta(_Seuranta):
    """Kaukopaan kiven seuranta hog-viivalta pysahtymiseen. Saapuminen tunnisti kiven (kahvan variblobi, B toteaa graniittirungon, rata etenee kohti pesaa), minka jalkeen kivea
    seurataan kuten heittopaassa: B-pohjainen ennuste, A (kahvan vari) ennusteen ymparilta, B joka ruudulle liikkeessa. Harjaajat ovat kiven vieressa (punainen kasivarsinauha on
    kahvan variaa; kivi voi olla hetken piilossa), joten A hylataan, jos se poikkeaa B-ennusteesta, ja ennuste kantaa piilossaolon yli (KAUKO_HUKKA_RUUTUA). Tulos h.kauko:
    paikat (B / A), pysahtymisruutu ja -paikka, hog-ylitys B-radasta (tarkentaa Saapuminen:n A-arvion), nopeus hogilla, tormays (nopeus putoaa jyrkasti: kivi osui toiseen kiveen)."""

    def __init__(self, analyysi, heitto, rata, kartta=()):
        self.an, self.h = analyysi, heitto
        self.puoli = rata.heitto.puoli
        self.m = analyysi.M[self.puoli]
        self.vari = heitto.vari
        self.heitto = rata.heitto
        self.heitto.lepo = np.array(rata[-1][1:3], float)
        self.kartta = list(kartta)                        # paikallaan olevat saman varin kahvat ennen heittoa: ei seurata (loppuu tormayksessa)
        self.ala_norm = [heitto.ala_med] if heitto.ala_med else []
        self._ennuste_luotettava = False
        self._ala_ulkona = []
        self.f_alku = rata[0][0]
        self.tila = "SEURAA"                              # SEURAA -> PAATTYI (odottaa B-tuloksia) -> valmis (h.kauko)
        self.syy = ""
        self.f_paattyi = None
        self.n_ruutu = 0
        self.hukka = 0
        analyysi.tapahtumat.append(("kauko_alku", heitto.tunnus, rata[-1][0], f"{self.puoli} {heitto.vari}"))

    # ---- ennuste ----------------------------------------------------------------------------------------------------------------
    def _ennuste(self, f):
        """Ennuste (X', Y') ruudulle f: viimeisten (<= 10) B-paikkojen suora (nopeus) ja hidastuva liike viimeisesta B-paikasta (piilossaolo: harjaajat peittavat kiven); luotettava
        (self._ennuste_luotettava), kun >= 4 B-paikkaa ja uusin <= ENNUSTE_B_TUORE ruutua vanha."""
        kh = self.heitto
        o = self._offset_b()
        pts, n_b = [], 0
        for n in reversed(kh.naytteet):
            if f - n.f > 240:
                break
            q = self._konsol(n, o)
            if q is not None:
                pts.append((n.f, q))
                n_b += bool(n.kivi())
            if len(pts) >= 10:
                break
        self._ennuste_luotettava = False
        if not pts:
            return tuple(kh.lepo)
        pts.reverse()
        fl, pl = pts[-1]
        v = np.zeros(2)
        if len(pts) >= 3:
            ff = np.array([p[0] for p in pts], float)
            Q = np.array([p[1] for p in pts])
            c = np.linalg.lstsq(np.column_stack([np.ones(len(ff)), ff - ff[-1]]), Q, rcond=None)[0]
            pl, v = c[0], c[1]                                                  # paikka viimeisessa paikkaruudussa (silotettu), nopeus cm / ruutu
            f_b = max([n.f for n in kh.naytteet[-60:] if n.kivi()] or [-10 ** 9])
            self._ennuste_luotettava = bool(n_b >= 4 and f - f_b <= ENNUSTE_B_TUORE)
        kh.lepo = np.array(pl, float)
        vs = float(np.hypot(*v)) * self.an.fps                                # cm / s
        dt = (f - fl) / self.an.fps
        if vs > 1.0 and dt > 0:
            td = min(dt, vs / KAUKO_HIDASTUVUUS_CM_S2)
            pred = pl + v / np.hypot(*v) * (vs * td - 0.5 * KAUKO_HIDASTUVUUS_CM_S2 * td * td)
        else:
            pred = pl
        return float(pred[0]), float(pred[1])

    # ---- paasilmukka ------------------------------------------------------------------------------------------------
    def kasittele(self, f, ruutu):
        """Yksi (yksilollinen) ruutu; palauttaa False kun seuranta on valmis (tulos h.kauko)."""
        kh = self.heitto
        self.n_ruutu += 1
        if self.tila == "PAATTYI":
            if f - self.f_paattyi >= self.an.viive_b + 6 and not any(t[2] is kh for t in self.an.tyot):
                self._valmis()
                return False
            return True
        Xp, Yp = self._ennuste(f)
        luotettava = self._ennuste_luotettava
        f_paikka_ed = max([m.f for m in kh.naytteet[-240:] if m.kivi()] or [self.f_alku])
        portti = min(14.0 + 0.5 * self._nopeus_ruutu() + KAUKO_PORTTI_KASVU_CM * max(f - f_paikka_ed - 2, 0), KAUKO_PORTTI_MAX_CM)      # piilossa oleva kivi: hakuportti kasvaa
        a = self._vari_a(ruutu, (Xp, Yp), portti, ohita=self.kartta)
        n = Naytteet(f)
        a_kelpaa = a is not None
        if a is not None and (f - f_paikka_ed) > KAUKO_A_LUOTTO_RUUTUA:
            a_kelpaa = False                                                  # pitka B-aukko: A ei kelpaa paikaksi ennen kuin B toteaa kiven (A voi olla harjaajan kasivarsinauha)
        elif a is not None and np.hypot(a[0] + self._offset_b()[0] - Xp, a[1] + self._offset_b()[1] - Yp) > KAUKO_A_POIKK_CM:
            a_kelpaa = False
        if a is not None:
            if a_kelpaa:
                n.A, n.ala = (a[0], a[1]), a[2]
                a_n = a[2] / (a[3] * a[3])
                if len(self.ala_norm) < 3 or ALA_VAIHTELU[0] <= a_n / float(np.median(self.ala_norm)) <= ALA_VAIHTELU[1]:
                    self.ala_norm = (self.ala_norm + [a_n])[-8:]
                    self._ala_ulkona = []
                else:
                    self._ala_ulkona = (self._ala_ulkona + [a_n])[-ALA_ULKONA_MAX:]
                    if len(self._ala_ulkona) >= ALA_ULKONA_MAX:
                        self.ala_norm, self._ala_ulkona = list(self._ala_ulkona), []
            o = self._offset_b()
            X0, Y0 = a[0] + o[0], a[1] + o[1]
            if luotettava and np.hypot(X0 - Xp, Y0 - Yp) > A_HARHA_CM:
                X0, Y0 = Xp, Yp                                # A harhainen (harjaajan kasivarsinauha): B:n alkuarvo on B-ennuste
        else:
            X0, Y0 = Xp, Yp
        kh.naytteet.append(n)
        v = self._nopeus_ruutu()
        vali = KAUKO_B_LIIKKUU_RUUTUA if v > 1.0 else (KAUKO_B_HIDAS_RUUTUA if v > 0.1 else KAUKO_B_LEPO_RUUTUA)
        if self.n_ruutu % vali == 0:
            self.an.pyyda_b(self, kh, n, ruutu, X0, Y0, (Xp, Yp) if luotettava else None)
        if KAUKO_KIERTO_Y_MIN_CM <= Y0 <= KAUKO_KIERTO_Y_MAX_CM and len(kh.kierto_kehykset) < KAUKO_KIERTO_MAX_RUUTUJA and (a is not None or luotettava):
            self._kierto_leike(kh, n, ruutu, X0, Y0)
        f_paikka = max([m.f for m in kh.naytteet[-400:] if m.kivi()] or [self.f_alku])
        self.hukka = f - f_paikka
        if self._pysahtynyt(f):
            self._paata(f, "pysahtyi")
        elif self.hukka > KAUKO_HUKKA_RUUTUA:
            self._paata(f, "hukassa")
        elif (f - self.f_alku) / self.an.fps > KAUKO_LOPPU_S:
            self._paata(f, "aikaraja")
        return True

    def _pysahtynyt(self, f):
        """Kivi on pysahtynyt: viimeiset 4 B-paikkaa (enintaan 3 s vanhoja) ovat KAUKO_PYS_CM:n sisalla ja kattavat vahintaan KAUKO_PYS_S, ja kivi on liikkunut seurannan alusta yli 50 cm.
        (Hitaalla kivella B tulee harvemmin, joten ikkuna maaraytyy paikkojen lukumaaran mukaan.)"""
        kh = self.heitto
        pts = []
        for m in reversed(kh.naytteet):
            if f - m.f > 3.0 * self.an.fps:
                break
            if m.kivi():
                pts.append((m.f, np.array(m.paikka(), float)))
                if len(pts) >= 4:
                    break
        if len(pts) < 4 or pts[0][0] - pts[-1][0] < KAUKO_PYS_S * self.an.fps:
            return False
        P = np.array([p[1] for p in pts])
        if float(max(np.ptp(P[:, 0]), np.ptp(P[:, 1]))) > KAUKO_PYS_CM:
            return False
        q0 = next((np.array(m.paikka(), float) for m in kh.naytteet if m.kivi()), None)
        return q0 is not None and float(np.hypot(*(P[0] - q0))) > 50.0

    def _paata(self, f, syy):
        self.tila, self.f_paattyi, self.syy = "PAATTYI", f, syy

    def paata_kesken(self, f):
        """Videon loppu kesken seurannan."""
        if self.tila == "SEURAA":
            self._paata(f if f is not None else self.f_alku, "ajo loppui")

    def viimeistele(self):
        if self.tila == "PAATTYI":
            self._valmis()

    # ---- tulos ---------------------------------------------------------------------------------------------------------
    def _valmis(self):
        """Seurannan yhteenveto h.kauko: paikat, pysahtyminen, hog-ylitys B-radasta, tormays."""
        h, kh = self.h, self.heitto
        fps = self.an.fps
        o = self._offset_b()
        rivit = [(n, self._konsol(n, o)) for n in kh.naytteet]
        rivit = [(n, np.asarray(q, float)) for n, q in rivit if q is not None]
        self.tila = "VALMIS"
        if len(rivit) < 5:
            h.kauko = dict(tila=self.syy, virhe="liian vahan paikkoja")
            self.an.tapahtumat.append(("kauko_valmis", h.tunnus, self.f_paattyi, f"{self.syy}, liian vahan paikkoja ({len(rivit)})"))
            return
        F = np.array([n.f for n, _ in rivit], float)
        P = np.array([q for _, q in rivit])
        vahva = np.array([n.kivi() for n, _ in rivit])
        peitossa = False
        poimittu = None
        i_k = kiihtymiskohta(F, P, fps)
        if i_k is not None and i_k >= 8:                   # kohde vaihtui kesken radan: rata loppuu viimeiseen ehjaan paikkaan (kivi hidastui, ei kiihtynyt)
            poimittu = float(F[i_k])
            F, P, vahva = F[:i_k], P[:i_k], vahva[:i_k]
            self.syy = "hukassa"                           # (loppunopeus < KAUKO_PYS_V_CM_S: seuraava tarkistus muuttaa sen "pysahtyi (peitossa)")
        if self.syy == "hukassa":                          # kivi hukattiin, mutta se oli jo hidas: pysahtyi piiloon (harjaajat / pelaajat sen paalla)
            n_v = min(len(F), 8)
            dt_v = (F[-1] - F[-n_v]) / fps
            if n_v >= 4 and dt_v > 0 and float(np.hypot(*(P[-1] - P[-n_v]))) / dt_v < KAUKO_PYS_V_CM_S:
                self.syy, peitossa = "pysahtyi", True
        f_pys = X_pys = Y_pys = None
        if self.syy == "pysahtyi":
            f_pys, X_pys, Y_pys = pysahtymishetki(F, P, fps)
        else:                                              # kivi hukattiin (piilossa / poistui kuvasta): pysahtymishetki arvioidaan viimeisesta B-paikasta (A-paikat voivat olla harjaajien nauhoja)
            f_pys, X_pys, Y_pys = float(F[-1]), float(P[-1][0]), float(P[-1][1])
            ib = np.nonzero(vahva)[0]
            if len(ib) >= 4:
                jb = ib[-min(len(ib), 6):]
                dt_b = (F[jb[-1]] - F[jb[0]]) / fps
                v_b = float(np.hypot(*(P[jb[-1]] - P[jb[0]])) / dt_b) if dt_b > 0 else 0.0
                if v_b > 30.0 and P[jb[-1]][1] < 0.0:      # nopea, pesan takana: taka-aita pysayttaa
                    lisa = min(max(P[jb[-1]][1] + KAUKO_LAUTA_Y_CM, 0.0) / v_b, 6.0) + KAUKO_LAUTA_LISA_S
                else:                                      # piilossa: hidastuu tyypillisella hidastuvuudella pysahtymiseen
                    lisa = min(v_b / (100.0 * KAUKO_PYS_HIDASTUVUUS), KELLO_PIILOSSA_MAX_S)
                f_pys, X_pys, Y_pys = float(F[jb[-1]]) + lisa * fps, float(P[jb[-1]][0]), float(P[jb[-1]][1])
        f_ulos = poistumisruutu(F, P, fps)
        if f_ulos is None and self.syy == "hukassa" and vahva.sum() >= 4:             # hukattu kivi, joka oli matkalla pelista pois: ylitys ekstrapoloidaan viimeisesta B-nopeudesta
            jb = np.nonzero(vahva)[0][-min(int(vahva.sum()), 6):]
            dt_b = (F[jb[-1]] - F[jb[0]]) / fps
            v_b = float(-(P[jb[-1]][1] - P[jb[0]][1]) / dt_b) if dt_b > 0 else 0.0              # cm / s kohti taka-aitaa (Y' pienenee)
            Yb = float(P[jb[-1]][1])
            if v_b > 30.0 and Yb < 0.0 and (Yb - PELI_TAKA_Y_CM) / v_b <= 6.0:
                f_ulos = float(F[jb[-1]]) + max(Yb - PELI_TAKA_Y_CM, 0.0) / v_b * fps
        # hog-ylitys B-radasta (tarkentaa A-arvion) ja nopeus hogilla
        f_hog = v_hog = None
        hog_lahde = None
        m = vahva & (np.abs(P[:, 1] - HOG_KESKI_CM) <= 130.0)
        if m.sum() >= 5 and P[m, 1].min() < HOG_KESKI_CM < P[m, 1].max():
            c = np.polyfit(P[m, 1], F[m], 1)
            f_hog, hog_lahde = float(np.polyval(c, HOG_KESKI_CM)), "B-rata"
            v_hog = float(-fps / c[0]) if c[0] != 0 else None
        else:                                              # kivi tunnistettiin vasta hog-viivan jalkeen: ylitys ekstrapoloidaan radan alusta (<= KAUKO_EKSTRAP_CM)
            ia = np.nonzero(vahva)[0][:30]
            if len(ia) >= 8 and 0.0 < HOG_KESKI_CM - float(P[ia, 1].max()) <= KAUKO_EKSTRAP_CM:
                c = np.polyfit(P[ia, 1], F[ia], 1)
                f_e = float(np.polyval(c, HOG_KESKI_CM))
                if c[0] < 0 and F[ia][0] - 1.5 * fps <= f_e <= F[ia][0]:
                    f_hog, hog_lahde = f_e, "ekstrapolointi"
                    v_hog = float(-fps / c[0])
        if f_hog is not None:
            h.hog2_f = f_hog
            self.an._paivita_kierre(h)
        # tormays: nopeus putoaa jyrkasti (luonnollinen hidastuminen on alle 0,2 m/s2)
        K = 6
        nop = []
        for i in range(len(F) - K):
            dt = (F[i + K] - F[i]) / fps
            nop.append((0.5 * (F[i] + F[i + K]), float(np.hypot(*(P[i + K] - P[i]))) / dt if dt > 0 else 0.0))
        torm = None
        for (f1, v1), (f2, v2) in zip(nop, nop[K:]):
            if v1 >= 80.0 and v2 <= 0.3 * v1 and f2 - f1 <= 3 * K:
                torm = dict(f=float(f1), v_ennen=float(v1), v_jalkeen=float(v2))
                break
        try:                                               # aikaperusta (lahdevirran ruutuindeksi): todellinen aika paikoille
            t_korj = [float(x) for x in aikap.Aikajana(F, P[:, 1], kopiot_ohitettu=True, R=(h.viivat or {}).get("lahdetaajuus")).aika(F)]
        except Exception:
            t_korj = [float(x) / fps for x in F]
        f_viim_b = float(F[vahva][-1]) if vahva.any() else None
        h.kauko = dict(tila=self.syy, peitossa=peitossa, f_alku=float(F[0]), f_loppu=float(F[-1]), f_pys=f_pys, X_pys=X_pys, Y_pys=Y_pys, f_hog2=f_hog, hog2_lahde=hog_lahde, v_hog_cm_s=v_hog, tormays=torm, f_viim_b=f_viim_b, kohde_vaihtui_f=poimittu, f_ulos=f_ulos,
                       pisteet=[(float(a), float(q[0]), float(q[1])) for a, q in zip(F, P)], t=t_korj, vahva=[bool(x) for x in vahva], n_b=int(vahva.sum()), n=len(F))
        self.an._paivita_kierre(h)                         # kierrokset irroituksesta pysahtymiseen (N_koko) tarvitsevat kaukopaan tuloksen: sama tulos riippumatta siita, kumpi (heittopaan kierto / kaukopaa) valmistui ensin
        self.an._kauko_kierto_aloita(h, kh, F, P[:, 1])
        self.an._vakiot_aloita(h)
        self.an.tapahtumat.append(("kauko_valmis", h.tunnus, int(round(F[-1])), f"{self.syy}, pysahtyi {f_pys / fps:.1f} s paikkaan ({X_pys:.0f}, {Y_pys:.0f}) cm"
                                   + (f", tormays {torm['f'] / fps:.1f} s" if torm else "")))


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
    """Kiertosuunta: +1 vastapaivaan, -1 myotapaivaan, 0 tuntematon. Kuvamittaus (omega; heittopaa ja kaukopaa), ellei se ole epavarma ja vapaan lennon kaarto (kaarevuus < 0: kaartaa
    vasemmalle = vastapaivaan) ole selva; ilman kuvamittausta selva kaarto riittaa vihjeeksi. Kaksi varmaa mittausta samaan suuntaan ratkaisevat suunnan."""
    v = h.viivat
    c = v.get("kaarevuus_1_cm") if v else None
    mitat = [(bool(k.get("varma")), 1 if k["omega_deg_s"] > 0 else -1) for k in (h.kierto, h.kauko_kierto) if k is not None and "virhe" not in k]
    if mitat:
        varmat = [s_ for vm, s_ in mitat if vm]
        if varmat and len(set(varmat)) == 1:
            return varmat[0]
        s_ = (varmat or [m[1] for m in mitat])[0]
        if c is not None and abs(c) >= KAARTO_VARMA_1_CM:
            s_ = 1 if c < 0 else -1
        return s_
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
    kierre = kierre_koko = None
    if h.kierre is not None and s != 0:
        kierre = -s * abs(h.kierre["N"])
        if h.kierre.get("N_koko") is not None:
            kierre_koko = -s * abs(h.kierre["N_koko"])
    return dict(liuku_cm=float(XT_l), heitto_cm=float(XT_i), merkki=s, irroitus_cm=(float(XT_i - XT_l) * s if s else None), kierre=kierre,
                kierre_mitattu=bool(h.kierre is not None and h.kierre["lahde"] == "mitattu"), T_hh=(h.kierre["T"] if h.kierre else None),
                irr_hog_m=float((HOG_SISA_CM - (v["irr_Y"] + KIVI_R_CM)) / 100.0), v_ennen=float(v["nopeus_ennen"]), v_jalkeen=float(v["nopeus_jalkeen"]),
                hidastuvuus=float(v["vapaa_hidastuvuus"]), vapaa_matka_cm=float(v["vapaa_matka_cm"]), virhe_cm=float((L - v["irr_Y"]) * np.tan(np.radians(v["irr_virhe_deg"]))), vakaa=bool(v.get("vakaa")),
                omega_deg_s=(h.kierto["omega_deg_s"] if h.kierto is not None and "virhe" not in h.kierto else None),
                kierto_varma=(bool(h.kierto.get("varma")) if h.kierto is not None and "virhe" not in h.kierto else None),
                omega_sd_deg_s=(h.kierto.get("omega_sd_deg_s") if h.kierto is not None and "virhe" not in h.kierto else None),
                suunta_p=(h.kierto.get("p_suunta") if h.kierto is not None and "virhe" not in h.kierto else None),
                kierre_varma=(bool(h.kierre.get("varma")) if h.kierre is not None else None), kierre_sigma=(h.kierre.get("N_sigma") if h.kierre is not None else None),
                kierre_koko=kierre_koko, kierre_koko_sigma=(h.kierre.get("N_koko_sigma") if h.kierre is not None else None),
                kierre_omega_lahde=(h.kierre.get("omega_lahde") if h.kierre is not None else None), kierre_k=(h.kierre.get("k") if h.kierre is not None else None),
                omega_kauko_deg_s=(h.kauko_kierto["omega_deg_s"] if h.kauko_kierto is not None and "virhe" not in h.kauko_kierto else None),
                kauko=(h.kauko if h.kauko is not None and "virhe" not in h.kauko else None), vakiot=(h.vakiot if h.vakiot is not None and h.vakiot.get("ok") else None))


# ---------------------------------------------------------------------------------------------------------------------
# ANALYYSI: kopioruudut, molemmat paat, B-tyoprosessit, viivaestimaatti, kierre
# ---------------------------------------------------------------------------------------------------------------------
def kierto_prosesseja():
    """(heittopaan, kaukopaan) kierron laskennan prosessien maara. Heittopaan kierto on oma prosessinsa (ei jonoudu kaukopaan kierron ja kiven vakioiden taakse: kierre nakyy livekuvassa heti
    heiton jalkeen); kaukopaan kierto ja kiven vakiot jakavat toisen (2, jos ytimia on vahintaan 6): ne tarkentavat tulosta jalkikateen."""
    return 1, (2 if (os.cpu_count() or 2) >= 6 else 1)


class Analyysi:
    """Syota ruudut yksi kerrallaan (syota); vahvistetut heitot (pisteet, viivat) luetaan heitot-listasta (piirtaja). synkroninen = True: B-tyot ja estimaatit
    odotetaan (sim-ajo: tulos ei riipu ajoituksesta); False: ne kulkevat taustalla (oikea reaaliaika). kierre = False: kierteen ja hog-hog-ajan mittaus pois."""

    def __init__(self, kansio="tulokset", fps=30.0, saikeet=3, tarkka=True, synkroninen=True, paat=("vasen", "oikea"), viive_b_ruutuja=6, fps_lahde=None, kierre=True):
        self.fps = float(fps)
        self.fps_lahde = 25.0 if fps_lahde is None else float(fps_lahde)      # None: paatellaan kopioruutujen osuudesta (_paata_fps_lahde)
        self._fps_auto = fps_lahde is None
        self.synkroninen = synkroninen
        self.aika = collections.defaultdict(float)  # analyysin ajankaytto (s) vaiheittain (syota): ks. netti_live.py loppuyhteenveto
        self._lykatyt = {}                           # toistettava ajo: kaukopaan kierron aloitus odottaa heittopaan kierron valmistumista
        self.toistettava = False                     # reaaliaikarakenne, mutta tulos ei riipu ajoituksesta: sama kuin sim-ajossa (ks. aseta_toistettava)
        self.viive_b = viive_b_ruutuja
        self.kansio = kansio
        self.tarkka = bool(tarkka)
        self.laske_kierre = bool(kierre) and self.tarkka
        self.M = {p: Mittari(kansio, p) for p in paat}
        self.pool = self.est_pool = self.kierto_pool = self.kauko_pool = None
        if self.tarkka and saikeet > 0:
            self.pool = tee_pool(saikeet, initializer=alusta_tyo, initargs=(kansio, tuple(paat)))
        if not synkroninen or saikeet > 0:
            self.est_pool = tee_pool(1, initializer=matala_prioriteetti) if not synkroninen else None
        if self.laske_kierre and not synkroninen:
            n_lahi, n_kauko = kierto_prosesseja()
            self.kierto_pool = tee_pool(n_lahi, initializer=alusta_kierto, initargs=(kansio, tuple(paat), False))       # heittopaan kierre (nakyy livekuvassa)
            self.kauko_pool = tee_pool(n_kauko, initializer=alusta_kierto, initargs=(kansio, tuple(paat), True))        # kaukopaan kierre + kiven vakiot (tarkentavat jalkikateen)
        self.b_valmis = self.pool is None            # B-tyoprosessit kaynnissa (vasta sitten "kivetonta" kahvaa ei hylata B-todisteen puutteen vuoksi): lampea()
        self.tyot = []                               # [(future, f, heitto, naytteet, alkuarvo)]
        self.est_tyot = []                           # [(future, heitto, f, n)]
        self.kierto_tyot = []                        # [(future, heitto)]
        self.vakiot_tyot = []                        # [(future, heitto)]: kiven vakioiden sovitus (kauko_pool)
        self.saapumiset = []                         # kaukopaan hakijat (Saapuminen)
        self.kaukoseurannat = []                     # tunnistetut kaukopaan kivet seurannassa (KaukoSeuranta)
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
        for pool, n in ((self.pool, getattr(self.pool, "_max_workers", 1)), (self.est_pool, 1), (self.kierto_pool, 1), (self.kauko_pool, getattr(self.kauko_pool, "_max_workers", 1))):
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
    def aseta_toistettava(self, paalla=True):
        """Toistettava reaaliaika: tulos ei riipu koneen nopeudesta eika tyoprosessien ajoituksesta, vaan on sama kuin sim-ajossa (--nopea). B-tulokset otetaan kayttoon kiintealla viiveella
        (ruutu f - viive_b odotetaan, B:ta ei pudoteta), estimaatit lasketaan kaikki jarjestyksessa ja heittopaan kierto odotetaan ennen kaukopaan kierron ennakkoa. Ruutuja ei ohiteta
        (aja_reaaliaika). Ei sovi live-virtaan, jossa kone ei voi odottaa."""
        self.toistettava = bool(paalla)

    def pyyda_b(self, hp, heitto, naytteet, ruutu, X0, Y0, ennuste=None):
        """B-mittaus naytteelle: tyoprosessille (leike) tai suoraan (ei poolia / sim ilman prosesseja). Reaaliajassa ruuhkautuessa B jatetaan pois. ennuste: luotettava
        (B-pohjainen) paikkaennuste (X, Y), jonka suhteen tulos hyvaksytaan myos silloin, kun alkuarvo (A) oli harhainen."""
        if not self.tarkka:
            return
        if self.pool is not None and not self.synkroninen and not self.toistettava and len(self.tyot) > 24:
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
        if naytteet.lepo and not poik_ok and peitto >= B_LEPO_PEITTO and s["rms"] <= B_MAX_RMS and kelv >= B_VAIN_KELV:
            naytteet.Bk = (s["X"], s["Y"])                          # sovitus loytaa kiven lahtopaikasta riippumatta: tama on levossa olevan kiven paikka
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
        self._keraa_vakiot()

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
            if not self.toistettava and any(e.heitto is heitto for e in self.est_tyot):      # toistettavassa ajossa kaikki estimaatit lasketaan (yksi prosessi, jarjestyksessa): historia (vakaa) sama kuin sim-ajossa
                if pakota:                           # analyysisaie ei jumitu odottamaan: lopullinen kaynnistyy kun kesken oleva on valmis (_keraa_estimaatit)
                    heitto.lopullinen_jonossa = True
                return                               # edellinen laskenta kesken
            self.est_tyot.append(types.SimpleNamespace(fut=self.est_pool.submit(laske_viivat, ff, XX, YY, self.fps_lahde, self.fps, lisa), heitto=heitto, f=f, n=len(ns),
                                                       lopullinen=bool(pakota)))
        else:
            self._liita_viivat(heitto, laske_viivat(ff, XX, YY, self.fps_lahde, self.fps, lisa), f, bool(pakota))

    @staticmethod
    def _aseta_viivat(heitto, r, f, lopullinen=False):
        """Uusi estimaatti heitolle. vakaa = merkitseva irroitus, rata kattaa koko irroituksen hakualueen (kiven Y >= Y_r_max + min_jalkeen: myohaisempaa muutosta ei voi olla piilossa; aiempi ehto
        'vapaata lentoa >= 250 cm' hyvaksyi aikaisen arvion, kun myohainen tyonto ei viela nakynyt: 25 / 68 vakaasta arviosta oli > 100 cm lopullisesta), irroituspaikka muuttunut < 30 cm edellisesta
        estimaatista ja suunnan virhe < 0,4 astetta."""
        if r is None or "virhe" in r:
            return False
        heitto.viivat_hist = (heitto.viivat_hist + [r["irr_Y"]])[-3:]
        edel = heitto.viivat_hist[-2] if len(heitto.viivat_hist) >= 2 else None
        kattava = r["Y_max"] - ESTIMAATTI["min_jalkeen"] >= ESTIMAATTI["Y_r_max"]
        r["vakaa"] = bool(r["merkitseva"] and kattava and edel is not None and abs(r["irr_Y"] - edel) < 30.0 and r["irr_virhe_deg"] < 0.4)
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
            if heitto.kierto_tila is None:                                   # heittopaan kierto ei kaynnistynyt (ei estimaattia / ruutuja): lykattyna ollut kaukopaan kierto ei odota sita
                self._vapauta_lykatty(heitto)
        if ok:
            self._paivita_kierre(heitto)
        if lopullinen and self.toistettava:
            self._vakiot_aloita(heitto)

    def _keraa_estimaatit(self, odota=False):
        jaa = []
        for e in self.est_tyot:
            if e.fut.done() or odota:
                try:
                    self._liita_viivat(e.heitto, e.fut.result(), e.f, e.lopullinen)
                except Exception:
                    if e.lopullinen:
                        e.heitto.viivat_lopullinen = True
                        if self.toistettava:
                            self._vakiot_aloita(e.heitto)
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
            self.kierto_tyot.append((self.kierto_pool.submit(tyo_kierto, args), h, False))
        else:
            self._aseta_kierto(h, kierto_laske(self.M[h.puoli], args))

    def _aseta_kierto(self, h, r):
        h.kierto = r
        h.kierto_tila = "virhe" if (r is None or "virhe" in r) else "valmis"
        self.tapahtumat.append(("kierto", h.tunnus, h.f_loppu, h.kierto_tila))
        self._paivita_kierre(h)
        self._vapauta_lykatty(h)
        if self.toistettava:
            self._vakiot_aloita(h)

    def _vapauta_lykatty(self, h):
        """Toistettava ajo: kaukopaan kierron aloitus odotti heittopaan kierron selviamista (tulos tai se, ettei sita tule): kaynnistetaan nyt."""
        lykatty = self._lykatyt.pop(h.tunnus, None)
        if lykatty is not None:
            self._kauko_kierto_aloita(*lykatty)

    def _kauko_kierto_aloita(self, h, kh, F, Y):
        """Kierteen mittaus kaukopaan kuvasta (kiven seurannan jalkeen): ruudut ja B-paikat kaukopaan seurannasta, aika aikaperustalla (kiven rata F, Y'). Taustaprosessiin tai
        heti (sim). Sama mittaus kuin heittopaassa (netti_kierre.arvioi): omega kaukopaan puolen kameralla."""
        if not self.laske_kierre or h.kauko_kierto_tila is not None or not kh.kierto_kehykset:
            return
        if self.toistettava and (h.kierto_tila == "kaynnissa" or not h.viivat_lopullinen):
            self._lykatyt[h.tunnus] = (h, kh, F, Y)                # kaukopaan alias-ratkaisu kayttaa heittopaan mittausta ennakkona: odotetaan sen valmistumista (ei jumiteta analyysia), kuten sim-ajossa; myos silloin, kun lopullinen estimaatti (josta heittopaan kierto kaynnistyy) on viela kesken (kuormitettu kone)
            return
        try:
            aj = aikap.Aikajana(F, Y, kopiot_ohitettu=True, R=(h.viivat or {}).get("lahdetaajuus"))
        except Exception:
            h.kauko_kierto_tila = "virhe"
            h.kauko_kierto = dict(virhe="aikaperusta epaonnistui")
            return
        kehykset = list(kh.kierto_kehykset)
        if len(kehykset) > KAUKO_KIERTO_KEHYKSIA:                 # tasainen otanta (sovituksen kesto kasvaa ruutumaaran mukana)
            idx = np.unique(np.round(np.linspace(0, len(kehykset) - 1, KAUKO_KIERTO_KEHYKSIA)).astype(int))
            kehykset = [kehykset[i] for i in idx]
        havainnot = []
        for n, roi, cx0, cy0 in kehykset:
            p = n.B if n.B is not None else n.Bs
            if p is not None:
                havainnot.append((float(aj.aika(n.f)[0]), n.f, float(p[0]), float(p[1]), roi, cx0, cy0))
        kh.kierto_kehykset = []
        if len(havainnot) < nk.MIN_RUUTUJA:
            h.kauko_kierto_tila = "virhe"
            h.kauko_kierto = dict(virhe=f"liian vahan ruutuja kaukopaassa ({len(havainnot)})", ruutuja=len(havainnot))
            return
        h.kauko_kierto_tila = "kaynnissa"
        mn = nk.omega_mitta(h.kierto, self.fps)
        ennakko = (float(np.degrees(mn[1])), mn[0], float(np.degrees(mn[2]))) if mn is not None and mn[3] else None          # varma heittopaan mittaus (omega, aika, sd): heikko ennakko, ratkaisee kaukopaan alias-minimien valilla
        args = (kh.puoli, h.vari, havainnot, havainnot[0][0] - 1.0, ennakko)
        if self.kauko_pool is not None:
            self.kierto_tyot.append((self.kauko_pool.submit(tyo_kierto, args), h, True))
        else:
            self._aseta_kauko_kierto(h, kierto_laske(self.M[kh.puoli], args))

    def _aseta_kauko_kierto(self, h, r):
        h.kauko_kierto = r
        h.kauko_kierto_tila = "virhe" if (r is None or "virhe" in r) else "valmis"
        self.tapahtumat.append(("kauko_kierto", h.tunnus, h.f_loppu, h.kauko_kierto_tila + ("" if h.kauko_kierto_tila == "valmis" else f" {(r or {}).get('virhe', '')}")))
        self._paivita_kierre(h)
        self._vakiot_aloita(h)

    # ---- kiven vakiot (A, kappa) kaukopaan radasta -------------------------------------------------------------------------
    def _vakiot_aloita(self, h):
        """Kiven vakiot (kitkataso A, kaarto kappa; netti_vakiot.sovita) kaukopaan radasta hog-viivalta pysahtymiseen (tai tormayksen alkuun / viimeiseen B-paikkaan, jos kivi
        hukattiin): kierteen mittaus kaukopaasta (tai yhdistetyn kierteen omega hogilla) pakotteena. Kaynnistetaan, kun kaukopaan rata ja sen kierteen mittaus ovat valmiit.
        Taustaprosessiin (kauko_pool) tai heti (sim)."""
        if not self.laske_kierre or h.vakiot_tila is not None:
            return
        if self.toistettava and (not h.viivat_lopullinen or h.kierto_tila == "kaynnissa" or h.tunnus in self._lykatyt):
            return                                                 # kaynnistyy, kun lopullinen estimaatti, heittopaan kierto ja kaukopaan kierto ovat valmiit (_liita_viivat, _aseta_kierto, _aseta_kauko_kierto): sim-ajossa ne ovat aina valmiina
        k = h.kauko
        if k is None or "virhe" in k or "pisteet" not in k or h.kauko_kierto_tila == "kaynnissa":
            return
        args, lahde = vakiot_argumentit(k, h.kauko_kierto, h.kierre, h.hog2_f, self.fps)
        args["irr"] = self._irr_tarkistus_argumentit(h)
        h.vakiot_tila = "kaynnissa"
        h.vakiot_omega_lahde = lahde
        if self.kauko_pool is not None:
            self.vakiot_tyot.append((self.kauko_pool.submit(vakiot_laske, args), h))
        else:
            self._aseta_vakiot(h, vakiot_laske(args))

    def _irr_tarkistus_argumentit(self, h):
        """netti_vakiot.tarkista_irroitus:n argumentit heitolle (None, jos tarkistus ei ole mahdollinen: ei viivaestimaattia, kivi ei pysahtynyt tai tormasi): irroituksen nopeus, suunta ja paikka,
        kierre irroitushetkella (mitattu omega hog-viivalla + vaimeneminen takaisin irroitukseen; muuten tyypillinen), kaarron suunta (kierron merkki) ja kaukopaan pysahtymispaikka."""
        v, k = h.viivat, h.kauko
        if v is None or k is None or "virhe" in k or k.get("tila") != "pysahtyi" or k.get("tormays") or k.get("Y_pys") is None:
            return None
        omega0 = nv_omega_oletus()
        kr = h.kierre
        if kr is not None and kr.get("omega1_deg") is not None:
            v_r = max(float(v["nopeus_jalkeen"]), 0.3)
            dt = max(HOG_KESKI_CM - float(v["irr_Y"]), 0.0) / (100.0 * v_r)                 # irroituksesta hog-viivalle
            omega0 = abs(float(np.radians(kr["omega1_deg"]))) * float(np.exp(float(kr.get("k") or 0.0) * dt))
        return dict(y0_cm=float(ETAISYYS_CM - v["irr_Y"]), v_r=float(v["nopeus_jalkeen"]), omega0=float(omega0), kaarto_merkki=int(kierron_merkki(h)), X_rel=float(-v["irr_X"]),
                    suunta_deg=float(v["irr_suunta_deg"]), Y_pys=float(k["Y_pys"]), X_pys=float(k["X_pys"]))

    def _aseta_vakiot(self, h, r):
        r = dict(r or dict(ok=False, syy="ei tulosta"))
        r["omega_lahde"] = getattr(h, "vakiot_omega_lahde", None)
        h.vakiot = r
        h.vakiot_tila = "valmis" if r.get("ok") else "virhe"
        self.tapahtumat.append(("vakiot", h.tunnus, h.f_loppu, (f"A {r['A']:.4f} kappa {r['kappa']:.2f} (Y rms {r['rms_y_cm']:.1f} cm, X rms {r['rms_x_cm']:.1f} cm)" if r.get("ok") else f"ei saatu: {r.get('syy', '?')}")))

    def _keraa_vakiot(self, odota=False):
        jaa = []
        for fut, h in self.vakiot_tyot:
            if fut.done() or odota:
                try:
                    r = fut.result()
                except Exception as e:
                    r = dict(ok=False, syy=repr(e))
                self._aseta_vakiot(h, r)
            else:
                jaa.append((fut, h))
        self.vakiot_tyot = jaa

    def kivien_vakiot(self):
        """Kivikohtaiset vakiot: {(vari, kivi_nro): dict(n, A, A_sd, n_kappa, kappa, kappa_sd, heitot)}; kiven yksilointi = (vari, numero heittopaassa), kunkin kiven onnistuneiden
        sovitusten mediaani ja hajonta (1,48 x MAD); kappa vain sovituksista, joissa kierre oli rajattu mittauksella. Huom: vakiot ovat tehollisia (harjaus pienentaa kitkaa ja kaarroa)."""
        ryhmat = {}
        for h in self.heitot:
            if h.vakiot is not None and h.vakiot.get("ok") and h.kivi_nro is not None:
                ryhmat.setdefault((h.vari, h.kivi_nro), []).append(h)
        mad = lambda x: float(1.4826 * np.median(np.abs(x - np.median(x)))) if len(x) > 1 else None
        out = {}
        for avain, hs in sorted(ryhmat.items()):
            A_ = np.array([h.vakiot["A"] for h in hs])
            K_ = np.array([h.vakiot["kappa"] for h in hs if h.vakiot.get("kappa_ok")])
            out[avain] = dict(n=len(hs), A=float(np.median(A_)), A_sd=mad(A_), n_kappa=len(K_), kappa=(float(np.median(K_)) if len(K_) else None), kappa_sd=mad(K_),
                              heitot=[h.tunnus for h in hs])
        return out

    def _keraa_kierto(self, odota=False):
        jaa = []
        for fut, h, kauko in self.kierto_tyot:
            if fut.done() or odota:
                try:
                    r = fut.result()
                except Exception as e:
                    r = dict(virhe=repr(e))
                (self._aseta_kauko_kierto if kauko else self._aseta_kierto)(h, r)
            else:
                jaa.append((fut, h, kauko))
        self.kierto_tyot = jaa

    def _paivita_kierre(self, h):
        """Kierrosmaara hog-hog-valille (h.kierre): heittopaan ja kaukopaan kierronmittaukset yhdistetaan (netti_kierre.yhdista: omega vaimenee eksponentiaalisesti, vaimeneminen
        mitataan kun molemmat mittaukset ovat). Hog-hog-aika luetaan videon puoliskoista (vasemman puoliskon radan hog-ylitys ja oikean puoliskon kuvan hog-ylitys), ellei
        kaukopaan ylitysta ole (viela) loydetty: vakiohidastuvuuden malli (lahde 'malli')."""
        v = h.viivat
        if v is None or h.hog1_f is None:
            return
        fps = self.fps
        mn, mf = nk.omega_mitta(h.kierto, fps), nk.omega_mitta(h.kauko_kierto, fps)
        if mn is None and mf is None:
            return
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
        r = nk.yhdista(mn, mf, t1, t1 + T)
        if r is None:
            h.kierre = None
            return
        om = h.kierto["omega_deg_s"] if mn is not None else h.kauko_kierto["omega_deg_s"]
        N_koko = N_koko_sigma = None                      # kierrokset koko radalla (irroitus -> pysahtyminen; viimeisten sekuntien spin-down jaa arvioon mukaan eksponentiaalisena)
        k_ = h.kauko
        if k_ is not None and "virhe" not in k_ and v.get("t_irr") is not None and k_["tila"] == "pysahtyi":
            rk = nk.yhdista(mn, mf, float(v["t_irr"]), k_["f_pys"] / fps)
            if rk is not None:
                N_koko, N_koko_sigma = rk["N"], rk["N_sigma"]
        h.kierre = dict(N=r["N"], T=float(T), lahde=lahde, omega=om, omega_lahde=r["lahde"], k=r["k"], N_sigma=r["N_sigma"], ristiriita=r["ristiriita"], N_koko=N_koko, N_koko_sigma=N_koko_sigma,
                        omega1_deg=float(np.degrees(r["omega1"])), omega2_deg=float(np.degrees(r["omega2"])),
                        varma=bool(not r["ristiriita"] and r["N_sigma"] <= KIERRE_SIGMA_VARMA and ((mn is not None and mn[3]) or (mf is not None and mf[3]))))

    # ---- joukkueiden kello ------------------------------------------------------------------------------------------
    def malli_pysahtyi_f(self, h):
        """Mallipohjainen arvio ruudulle, jolla heiton kivi pysahtyy (ei vaadi kaukopaan seurantaa eika kierteen mittausta): nopeus lahipaan hog-viivalla (laske_viivat: nopeus_hog1,
        muuten irroituksen nopeudesta tyypillisella hidastuvuudella) -> hog-viivalta hog-viivalle vakiohidastuvuus (mitattu hog-ylitys / h.kierre['T'] jos on, muuten malli), kaukopaan hog-viivalta
        pysahtymiseen nopeus / KAUKO_PYS_HIDASTUVUUS (enintaan KELLO_ARVIO_MAX_S, nopealla kivella KELLO_NOPEA_MAX_S). Kivi, joka ei yllä kaukopaan hog-viivalle, pysahtyy vakiohidastuvuudella.
        Virhe (112 heiton aineisto, kaukopaan seuranta pysahtyi): mediaani -0,3 s, hajonta 3,8 s. Ilman viivaestimaattia tai hog-ylitysta: heiton lahto + KELLO_ALKU_PYS_S."""
        v = h.viivat
        hog1_f = getattr(h, "hog1_f", None)
        fps = self.fps
        if v is None or hog1_f is None:                                    # rata katkesi ennen hog-viivaa tms.: viimeinen keino, tyypillinen aika lahdosta
            f_alku = getattr(h, "f_alku", None)
            return None if f_alku is None else float(f_alku + fps * KELLO_ALKU_PYS_S)
        a1 = HIDASTUVUUS_TYYPILLINEN
        v1 = v.get("nopeus_hog1")
        if v1 is None:
            v1 = max(float(v["nopeus_jalkeen"]) ** 2 - 2.0 * a1 * max(HOG_KESKI_CM - float(v["irr_Y"]), 0.0) / 100.0, 0.0) ** 0.5
        v1 = max(float(v1), 0.3)
        D = (ETAISYYS_CM - 2.0 * HOG_KESKI_CM) / 100.0
        v2sq = v1 * v1 - 2.0 * a1 * D
        if v2sq <= 0.0:                                                    # ei yllä kaukopaan hog-viivalle: pysahtyy matkalla
            return float(hog1_f + fps * v1 / a1)
        v2 = v2sq ** 0.5
        kierre = getattr(h, "kierre", None)
        T = (float(kierre["T"]) if kierre is not None else (v1 - v2) / a1)
        f_hog2 = h.hog2_f if getattr(h, "hog2_f", None) is not None else hog1_f + fps * T
        t_pys = min(v2 / KAUKO_PYS_HIDASTUVUUS, KELLO_ARVIO_MAX_S)
        if v2 > KELLO_NOPEA_M_S:                                           # nopea kivi (poisto): tormaa tai osuu taka-aitaan nopeasti
            t_pys = min(t_pys, KELLO_NOPEA_MAX_S)
        return float(f_hog2 + fps * t_pys)

    def kivi_liikkeessa(self, h, f_nyt):
        """True, kun kaukopaan seuranta on nahnyt heiton kiven liikkeessa viimeisen KELLO_NAHTY_S ajan sisalla (B-paikkojen vaihteluvali yli KELLO_NAHTY_CM)."""
        ks = next((k for k in getattr(self, "kaukoseurannat", []) if k.h is h), None)
        if ks is None:
            return False
        P = []
        for m in reversed(ks.heitto.naytteet):
            if f_nyt - m.f > KELLO_NAHTY_S * self.fps:
                break
            if m.kivi():
                P.append(m.paikka())
        if len(P) < 2:
            return False
        P = np.asarray(P, float)
        return bool(float(np.hypot(*(P.max(axis=0) - P.min(axis=0)))) > KELLO_NAHTY_CM)

    def pysahtyi_f(self, h):
        """Ruutu, jolla heiton kivi (ja sen tormayksessa liikkeelle laittamat kivet) pysahtyi tai poistui pelista (taka- / sivuviivan ulkopuolelle) kaukopaassa: kaukopaan seurannan
        tulos (hukatulla kivella arvio viimeisesta B-paikasta); tormaysheitolla viimeistaan KELLO_TORMAYS_LISA_S tormayksen jalkeen. Ellei seurannan tulosta ole, malli
        (malli_pysahtyi_f): mallin mukainen pysahtymishetki oletetaan toteutuneeksi, kun siita on kulunut KELLO_MALLI_ODOTUS_S analyysiajassa eika kivea ole nahty liikkeessa
        (seuranta ei ole paattynyt tai kivea ei loytynyt; muuten kello ei lahtisi kayntiin). None, kun pysahtymista ei voi arvioida (ei viivaestimaattia) tai mallin mukaan se ei ole viela tapahtunut."""
        k = h.kauko
        if k is not None and "virhe" not in k and k.get("f_pys") is not None:
            f = float(k["f_pys"])
            if k.get("f_ulos") is not None:                    # peliin enaa vaikuttamaton (taka- / sivuviivan ulkopuolella) kivi: kello kay jo silloin
                f = min(f, float(k["f_ulos"]))
            if k.get("tormays"):
                f = max(f, float(k["tormays"]["f"]) + KELLO_TORMAYS_LISA_S * self.fps)
            return f
        f_m = self.malli_pysahtyi_f(h)
        if f_m is None:
            return None
        f_nyt = getattr(self, "viim_f", None)
        if f_nyt is not None and (f_nyt < f_m + KELLO_MALLI_ODOTUS_S * self.fps or self.kivi_liikkeessa(h, f_nyt)):
            return None
        return f_m

    def kello_ruutu(self, h):
        """Ruutu, jolla heiton miettimisaika paattyy: kiven etureuna osuu heittopaan T-viivaan (Heitto.tviiva_f); ellei rata ylita viivaa, heiton lahto + KELLO_TVIIVA_VIIVE_S (vain valmistuneelle
        heitolle). None: ei viela selvilla (heitto kesken)."""
        f = h.tviiva_f()
        if f is not None:
            return float(f)
        if h.f_alku is not None and (h.tila == "valmis" or h.lopullinen):
            return float(h.f_alku) + KELLO_TVIIVA_VIIVE_S * self.fps
        return None

    def kello(self):
        """Miettimisaika heittokohtaisesti ja joukkueittain: {tunnus: dict(aika_s, joukkue_yht_s, paa, heitto_paassa, f_pys_edell, f_stop)}. Joukkueen kello kayntiin kun edellisen heiton kivi
        (ja muut kivet) on pysahtynyt tai poistunut pelista ja pysahtyy kun joukkueen heitetty kivi osuu heittopaan T-viivaan (f_stop = kello_ruutu: kiven etureuna, ei kiven lahto hackista);
        paiden valilla kello ei kay (paan ensimmaiselle heitolle ei aikaa: aika None). Heittoja jarjestetaan lahtoruudun mukaan; paa vaihtuu kun heittopuoli vaihtuu."""
        out, yht = {}, {}
        paa = n_paassa = 0
        edell = None
        for h in sorted((x for x in self.heitot if x.f_alku is not None), key=lambda x: x.f_alku):
            uusi = edell is None or edell.puoli != h.puoli
            if uusi:
                paa += 1
                n_paassa = 0
            n_paassa += 1
            aika, f_pys = None, None
            f_stop = self.kello_ruutu(h)
            if not uusi and f_stop is not None:
                f_pys = self.pysahtyi_f(edell)
                if f_pys is not None and f_pys - f_stop <= KELLO_ENNEN_S * self.fps:        # pysahtyminen selvasti kellon pysahtymisen jalkeen: arvio on virheellinen
                    aika = max(0.0, (f_stop - f_pys) / self.fps)
            if aika is not None:
                yht[h.vari] = yht.get(h.vari, 0.0) + aika
            out[h.tunnus] = dict(aika_s=aika, joukkue_yht_s=float(yht.get(h.vari, 0.0)), paa=paa, heitto_paassa=n_paassa, f_pys_edell=f_pys, f_stop=f_stop)
            edell = h
        return out

    def kauko_selvilla(self, h):
        """True, kun heiton kiven pysahtymishetki on lopullinen: kaukopaan seuranta on valmis (h.kauko) tai sita ei tule (kivea ei loydetty, ei kierretta / hog-ylitysta)."""
        if h.kauko is not None:
            return True
        return (not self.laske_kierre) or h.hog1_f is None or h.saapuminen == "ei_loytynyt" or (h.lopullinen and h.viivat is None)

    def kello_nyt(self, f_out):
        """Kellon tila videon hetkella f_out (piirtajalle): dict(yht={vari: aika s}, kay=joukkueen vari tai None, juokseva_s, viim=viimeisin heitto, jonka kello on pysahtynyt, viim_aika_s). Kayva
        kello = edellisen heiton kivi on pysahtynyt (pysahtyi_f <= f_out), seuraava heitetty kivi ei ole viela osunut T-viivaan (kello kay myos silloin, kun kivi on jo lahtenyt hackista) ja
        paa on kesken; kayvalle joukkueelle (edellisen heiton vastavari) lasketaan mukaan kulunut aika."""
        k = self.kello()
        pysaytetyt = sorted((h for h in self.heitot if h.f_alku is not None and k[h.tunnus]["f_stop"] is not None and k[h.tunnus]["f_stop"] <= f_out), key=lambda x: x.f_alku)
        yht = {}
        for h in pysaytetyt:
            a = k[h.tunnus]["aika_s"]
            if a is not None:
                yht[h.vari] = yht.get(h.vari, 0.0) + a
        viim = pysaytetyt[-1] if pysaytetyt else None
        kay = juokseva = None
        if viim is not None and k[viim.tunnus]["heitto_paassa"] < 2 * KIVIA_PAASSA:
            f_pys = self.pysahtyi_f(viim)
            if f_pys is not None and f_out >= f_pys:
                kay = "punainen" if viim.vari == "keltainen" else "keltainen"
                juokseva = float(f_out - f_pys) / self.fps
                yht[kay] = yht.get(kay, 0.0) + juokseva
        return dict(yht=yht, kay=kay, juokseva_s=juokseva, viim=viim, viim_aika_s=(k[viim.tunnus]["aika_s"] if viim is not None else None))

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
        if self.synkroninen or self.toistettava:
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
        for ks in self.kaukoseurannat:
            ks.paata_kesken(self.viim_f)
        self.odota(f_asti=10 ** 12)
        for ks in self.kaukoseurannat:
            ks.viimeistele()
        self.kaukoseurannat = []
        self._loppujen_tarkistus()
        for _ in range(4):                           # kesken olevat ja jonossa odottavat lopulliset estimaatit
            self._keraa_estimaatit(odota=True)
            if not self.est_tyot and not any(h.lopullinen_jonossa for h in self.valmiit):
                break
        for h in [x[0] for x in list(self._lykatyt.values())]:                  # lykatty kaukopaan kierto ei jaa odottamaan ikuisesti (lopullista estimaattia ei tullut)
            self._vapauta_lykatty(h)
        self._keraa_kierto(odota=True)
        for h in self.heitot:
            self._paivita_kierre(h)                          # lopullinen tila kaikesta aineistosta (ei riipu valmistumisjarjestyksesta)
        self._keraa_vakiot(odota=True)

    def valmista_kaikki(self):
        """True kun mikaan tausta- tai kaukopaan tyo ei ole kesken (reaaliajassa: lopetus voi odottaa tulokset)."""
        return not (self.tyot or self.est_tyot or self.kierto_tyot or self.vakiot_tyot or self.saapumiset or self.kaukoseurannat or any(getattr(h, "odottaa_loppua", False) for h in self.valmiit))

    # ---- paasilmukka -----------------------------------------------------------------------------------------------
    def syota(self, f, ruutu):
        """Yksi ruutu (BGR, 1920 x 1080). Kopioruudut ohitetaan."""
        pc = time.perf_counter
        t0 = pc()
        kopio = self._kopio(ruutu)
        t1 = pc()
        self.aika["kopio"] += t1 - t0
        if kopio:
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
        t2 = pc()
        self.aika["heittopaat"] += t2 - t1
        for s in list(self.saapumiset):
            if not s.kasittele(f, ruutu):
                self.saapumiset.remove(s)
        for ks in list(self.kaukoseurannat):
            if not ks.kasittele(f, ruutu):
                self.kaukoseurannat.remove(ks)
        t3 = pc()
        self.aika["kaukopaa"] += t3 - t2
        self.odota(f - self.viive_b if (self.synkroninen or self.toistettava) else None)
        t4 = pc()
        self.aika["odota (B, taustatyot)"] += t4 - t3
        for hp in self.paat.values():
            if hp.heitto is not None and hp.heitto.tila == "vahvistettu":
                self.paivita_viivat(hp.heitto, f)
        self.aika["viivaestimaatit"] += pc() - t4

    def ohita(self):
        """Ruutu nakyy tulosteessa mutta sita ei analysoida (analyysi myohassa): kopioruututesti ei saa verrata kahta eri ruutua."""
        self.edellinen_harmaa = None

    def sulje(self):
        for p in (self.pool, self.est_pool, self.kierto_pool, self.kauko_pool):
            if p is not None:
                p.shutdown(wait=False, cancel_futures=True)
