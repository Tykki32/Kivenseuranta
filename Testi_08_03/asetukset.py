"""KAIKKI SAADETTAVAT ASETUKSET YHDESSA PAIKASSA.

Muuta arvoja tassa tiedostossa (ymparistomuuttujia ei kayteta; poikkeus: GPU_LAITE=cpu OpenCL-testaukseen).
Arvot ovat Testi_08_02:n testatut oletukset. Etaisyydet cm, ajat s, pikselimaarat 1280x720-kuvassa ellei toisin
mainita. Rata: Y kasvaa kohti kaukaista paata (heittopaa), X = 0 keskiviivalla; lahempi pesa (kameran puoli) on
pienemmalla Y:lla.

Kalibroinnin ja kuvantunnistuksen sisaiset vakiot (renkaiden varikynnykset, hakuruudukot, LM-parametrit) ovat
omissa moduuleissaan (kalibrointi_perus.py, kalibrointi.py, paneelit.py, profiili.py, siluetti.py, cpp/*.hpp):
ne ovat algoritmin osia, eivat kayton aikana saadettavia.

Osiot:
  1. Kalibrointi (moodikuva, kamera, hoglinet)
  2. Kiviprofiilin opettelu (3D-malli)
  3. Seuranta: HAKU (uudet kivet) ja SEURANTA (liikkuvat kivet)
  4. Radan elinkaari: vahvistus, pysahtyminen, kadotus, duplikaatit
  5. Heittoportti (mitka radat ovat heittoja)
  6. Hog-hog -analyysi, liuku ja irroitus
  7. Kahva ja kierteet
  8. Paikallinen taysi resoluutio (live)
  9. Kuvan esikasittely (stabilointi, valotasapaino, taustanvaimennus)
 10. Suorituskyky (saikeet, liukuhihna)
 11. Live-syote
 12. Puhelinnakyma (--katselu) ja debug-video
"""

# =====================================================================================================================
# 1. KALIBROINTI (seuranta.py, kalibrointi.py)
# =====================================================================================================================
KALIB_MOODI_S = 120.0                # moodikuvan naytteet videon / live-syotteen ensimmaisen 120 s ajalta
KALIB_NAYTEVALI_S = 5.0              # naytevali moodikuvaan (s)
KALIB_PANEELI_MAX_HYPPY_PX = 20.0    # paneelin keskipiste saa siirtya ruudusta toiseen enintaan nain paljon
KALIB_PANEELI_HISTORIA = 10          # paneelin paikkahistorian pituus (ruutua)
KALIB_STAB_MEDIAANI_RUUTUA = 20      # paneelistabiloinnin mediaanisuodatus (ruutua)
KALIB_HOGLINJA_TOLERANSSI_CM = 20.0  # hoglinjojen todellinen paikka voi poiketa nimellisesta (640 cm T:sta) +- tama
KALIB_KAUKOHOG_PAINO = 0.0           # kaukaisen hoglinjan mittauksen paino yhteisessa siirtymassa (lahi = 1)
MOODI_TIILI = 128                    # moodikuvan laskennan tiilikoko (C++ ModeEngine)
PANEELI_SAIKEET = 8                  # paneelien seuranta rinnan

# =====================================================================================================================
# 2. KIVIPROFIILIN OPETTELU (profiili.py)
# =====================================================================================================================
# Pelialue, jolta kivia etsitaan (fyysiset cm); parallaksivara: kiven keskipiste on ~7 cm jaan ylapuolella
PROFIILI_ALUE_X_MIN_CM = -200.0
PROFIILI_ALUE_X_MAX_CM = 200.0
PROFIILI_ALUE_Y_MIN_CM = 400.0
PROFIILI_ALUE_Y_MAX_CM = 2000.0
PROFIILI_ALUE_PARALLAKSIVARA_CM = 25.0
PROFIILI_SKANNAUS_VALI_S = 0.2       # tihea skannaus nain usein
PROFIILI_SKANNAUS_SAIKEET = 3        # siemenen seurannan skannaus rinnan
PROFIILI_SIEMEN_MIN_HAVAINNOT = 4    # kandidaattiradalla vahintaan nain monta havaintoa ennen kuin siita tulee siemen
PROFIILI_LIIKE_CM = 15.0             # siemen: rata on liikkunut radan suuntaan vahintaan nain paljon 2 s:ssa
PROFIILI_ESITARKISTUS_S = 3.0        # esitarkistuksen ikkuna (+-s siemenesta)
PROFIILI_ESITARKISTUS_MIN_HAVAINNOT = 5
PROFIILI_ESITARKISTUS_MAX_RMS_PX = 8.0
PROFIILI_ESITARKISTUS_R_MIN_CM = 12.0
PROFIILI_ESITARKISTUS_R_MAX_CM = 16.0
PROFIILI_IKKUNA_S = 30.0             # siemenen seurannan ikkuna (+-s); maaraa myos live-puskurin vahimmaishistorian
PROFIILI_HAVAINTOJA_KIVESTA = 40     # havaintoja yhdesta kivesta profiilisovitukseen
PROFIILI_YKSI_MAX_RMS_PX = 3.0       # yksittainen kivi kelpaa kokoelmaan (ei pelaaja)
PROFIILI_MAX_RMS_PX = 1.5            # koko kokoelman profiili riittava: jaannosvirhe
PROFIILI_MIN_HAVAINNOT = 15
PROFIILI_R_MIN_CM = 12.5             # sallittu kiven sade (saannot: ymparysmitta <= 91,44 cm -> R <= 14,55 cm)
PROFIILI_R_MAX_CM = 15.0
PROFIILI_MIN_KIVIA = 2               # profiili valmis kun nain monta kivea on hyvaksytty
ETEENPAINLUKU_MAX_RUUTUA = 25        # havaintoruutujen luku: eteenpain (grab) jos lahella, muuten haku
# Alfa-aariviiva (kiven reuna peittavyydesta): yhteinen taso = graniitin V / jaan V (~0,43)
ALFA_TASO_OLETUS = 0.434
ALFA_SUHDE_MIN = 0.2                 # havainnon graniitti/jaa-suhteen sallittu vali
ALFA_SUHDE_MAX = 0.8
ALFA_YLINAYTTEISTYS = 4
ALFA_RAJAUS_PUOLIKAS_PX = 60
ALFA_LAAJENNUS_PX = 5                # graniittimaskin laajennus aariviivan rajaukseen
ALFA_TUMMA_V_MAX = 60                # hylkays: tumma alue (V < tama) kiven ymparilla...
ALFA_TUMMA_MAX_PX = 100              # ... yli nain monta pikselia (lakaisijan harja/kenka)
ALFA_ALA_TOLERANSSI = 0.30           # hylkays: pinta-ala poikkeaa lahiruutujen mediaanista yli 30 %
ALFA_ALA_IKKUNA = 4

# =====================================================================================================================
# 3. SEURANTA: HAKU (uudet kivet) ja SEURANTA (liikkuvat kivet) (seuranta.py)
# =====================================================================================================================
MAX_KIVIA = 8                        # samanaikaisesti seurattavia ratoja enintaan
HAKU_VALI_RUUTUA = 10                # HAKU joka n:s ruutu (omassa saikeessaan)
HAKU_ALUE_X_PUOLIKAS_CM = 70.0       # hakualue: X = +-70 cm ...
HAKU_ALUE_ENNEN_HOGIA_CM = 100.0     # ... Y = kaukohog - 100 cm ...
HAKU_ALUE_HOGIN_TAKANA_CM = 300.0    # ... kaukohog + 300 cm
HAKU_KARKEA_ASKEL_CM = 10.0
HAKU_HIENO_ASKEL_CM = 2.0
HAKU_PISTEKYNNYS = 0.15              # esikarsinta (ehdokkaan lopullinen hyvaksynta C++:n yhteissovituksesta)
HAKU_MAX_TULOKSET = 4                # HAKU palauttaa kaikki kelvolliset ehdokkaat (ei vain ensimmaista)
HAKU_MAX_YRITYKSET = 8
# uuden ehdokkaan suodatin (C++): sovitus, runkopisteet, etaisyys keskiviivasta, pistemaara
HAKU_UUSI_MAX_RMS = 3.5
HAKU_UUSI_MIN_RUNKO = 14
HAKU_UUSI_MAX_RUNKO = 45
HAKU_UUSI_MAX_ABS_X_CM = 65.0
HAKU_UUSI_MAX_PISTEET = 0.9
HAKU_JO_SEURATTU_CM = 20.0           # ehdokas vahvistetun radan lahella -> jo seurattu
HAKU_SAMA_HAKU_CM = 30.0             # saman haun ehdokkaat lahekkain -> yksi
# ehdokas liikkuvan heittokiven takana (heittaja liukuu perassa, harjaajat) ei saa uutta rataa
HAKU_TAKANA_MIN_CM = 50.0
HAKU_TAKANA_MAX_CM = 800.0
HAKU_TAKANA_SIVU_CM = 100.0
# uuden ehdokkaan edessa (kameran puolella) ei saa olla isoa hahmoa: pelaajan paa projisoituu jaan tasoon vartalonsa
# taakse, joten sen edessa on vartalo ja jalat; aidon kiven edessa on lahtohetkella tyhjaa jaata (heittaja on takana)
EDESSA_VALI_CM = 5.0                 # kaistan alku kiven reunasta (cm)
EDESSA_PITUUS_CM = 100.0             # kaistan loppu kiven reunasta (cm)
EDESSA_PUOLILEVEYS_CM = 10.0         # kaistan puolileveys sivusuunnassa (cm)
EDESSA_MAX_OSUUS = 0.5               # hahmo-osuus suurempi -> ehdokas hylataan (aidot heitot MAH: 0-0.09; > 1 = pois)
SEURANTA_KARKEA_ASKEL_CM = 7.0
SEURANTA_HIENO_ASKEL_CM = 1.5
SEURANTA_PISTEKYNNYS = 0.35
# yhdistelmahaku (C++ seuranta.hpp): ristikkohaku ennustetusta paikasta + mean-shift viimeisesta ja ennustetusta paikasta;
# valitaan suurin pistemaara - rangaistukset (taaksepain, poikkeama ennusteesta)
SEURANTA_MS_KERROIN = 0.8            # mean-shiftin askelkerroin (gain)
SEURANTA_MS_MAX_ITER = 6
SEURANTA_MS_TOL_PX = 0.30            # lopetus kun askel < tama
SEURANTA_MS_SISAPAINO = 2.0          # hullin sisapikselien paino (1.1x-kehan paino 1)
SEURANTA_MS_MARGINAALI = 1.10        # kehan koko hullin suhteen
SEURANTA_MS_TAU = 0.5                # pehmean etumerkin leveys (0 = kova)
SEURANTA_MS_VIIMEISTELY_CM = 4.0     # lopuksi maennousu peitto-osuudella, aloitusaskel (puolittuu kunnes < 0,75 cm)
SEURANTA_VALINTA_TAAKSE_TOL_CM = 2.0  # taaksepain-siirtyma ilman rangaistusta
SEURANTA_VALINTA_TAAKSE_SAKKO = 0.15  # rangaistus / 10 cm taaksepain
SEURANTA_VALINTA_ENNUSTE_SAKKO = 0.02  # rangaistus / 10 cm poikkeamasta ennusteesta
# hakualue joka ruudulla kiven omasta viimeisesta paikasta: fysikaalinen nopeusraja x kulunut aika
SEURANTA_MAX_NOPEUS_Y_CM_S = 400.0
SEURANTA_SIVUNOPEUS_OSUUS = 0.10     # sivunopeus enintaan 10 % pitkittaisesta
SEURANTA_MAX_HAKUALUE_CM = 100.0
SEURANTA_HAKUALUE_VARA_CM = 5.0      # tarkennettu paikka saa olla enintaan nain paljon hakualueen ulkopuolella
SEURANTA_MAX_TAAKSE_CM = 100.0       # kivi ei liiku taaksepain (kumulatiivisesti) yli taman
SEURANTA_MAX_ABS_X_CM = 220.0        # paikka enintaan nain kaukana keskiviivasta (koko radan leveys)
# siluettituki graniittimaskista (siluetti.py): sisalla vahintaan tama osuus maskia, muuten havainto hylataan
SEURANTA_MIN_SISALLA = 0.4
SEURANTA_MIN_SISALLA_LAHELLA = 0.0   # lahella (Y <= SEURANTA_SIL_KAIKKI_Y_CM) myos tarkoille ruuduille (0 = ei porttia)
SEURANTA_SIL_SIIRTO_PX = 3           # siluettitarkennuksen suurin siirto
SEURANTA_SIL_KAIKKI_Y_CM = 2000.0    # kaukana (Y > tama) siluetti tarkentaa kaikki ruudut (kivi 5-6 px leveä)

# =====================================================================================================================
# 4. RADAN ELINKAARI (seuranta.py)
# =====================================================================================================================
VAHVISTUS_SIIRTYMA_CM = 25.0         # ehdokas vahvistetaan kun se on liikkunut nain paljon (rivit puskuroidaan siihen asti)
VAHVISTUS_MIN_RUUTUJA = 8            # vahvistushetkella: hylataan jos tarkka-osuus < VAHVISTUS_MIN_TARKKA ...
VAHVISTUS_MIN_TARKKA = 0.4
VAHVISTUS_PELASTUS_MAX_RMS = 12.0    # ... ja rms-mediaani >= tama (kaukaiset aidot heitot: rms 8-10 px)
VAHVISTETTU_TARKKA_IKKUNA = 650      # vahvistettu rata lopetetaan jos tarkka-osuus liukuvassa ikkunassa ...
VAHVISTETTU_MIN_TARKKA = 0.5         # ... putoaa alle taman (SEURANTA ajautunut pelaajaan)
KADONNUT_S = 1.0                     # rata kadotettu kun ei havaintoa nain pitkaan
PYSAHTYNYT_S = 1.0                   # kivi pysahtynyt: liikkunut < PYSAHTYNYT_CM viimeisen PYSAHTYNYT_S aikana
PYSAHTYNYT_CM = 20.0
LOPETA_LAHIHOGIN_JALKEEN_CM = 30.0   # seuranta lopetetaan kun kivi on nain paljon lahihogin ohi (0 = pesaan asti)
SEURAA_PYSAHTYMISEEN = False         # --full: vahvistettua kivea seurataan lahihogin jalkeenkin pysahtymiseen asti
PYSAHTYNYT_TARKKA_CM = 3.0           # --full: kivi pysahtynyt kun 5 ensimmaisen ja 5 viimeisen paikan mediaanit
                                     #   PYSAHTYNYT_S-ikkunassa ovat nain lahella
PYSAHDYS_VAHVISTUS_S = 2.0           # --full: pysahtyneen kiven on pysyttava nakyvissa ja paikallaan nain kauan ...
PYSAHDYS_MAX_SIIRTO_CM = 10.0        # ... (alle tama pysahdyspaikasta); muuten (pelaaja pysaytti/vei) ei pysahtymispaikkaa
PYSAHDYS_MIN_PAIKAT = 10             # ... ja vahvistusjaksolla on havaittu vahintaan nain monta paikkaa (peitossa -> odotetaan)
# pysahtymispaikan nollakohta (debug-ikkunan rivi "pysahtyi: x; y", hog-CSV): X keskiviivasta, Y lahemman pesan
# T-viivasta; siirto cm (X: + = kohti +X, Y: + = kohti kaukaista paata). Debug-ikkunassa vasen ja alas negatiivisia.
PYSAHDYS_NOLLA_X_CM = 0.0
PYSAHDYS_NOLLA_Y_CM = 0.0
LOPETA_TAAKSEPAIN_CM = 30.0          # rata lopetetaan jos Y kasvaa yli taman sekunnissa (pelaaja, takaisin vietava kivi)
DUPLIKAATTI_CM = 30.0                # kaksi rataa lahempana kuin tama ...
DUPLIKAATTI_RUUDUT = 3               # ... nain monta ruutua perakkain -> yhdistetaan
SUOJAA_ETEENPAIN_CM = 30.0           # radan suuntaan (1 s aikana) liikkuva rata suojataan yhdistamisessa/paikanvarauksessa
POISTA_MIN_RMS = 8.0                 # paikanvaraus: vahvistettu rata voidaan poistaa vain jos rms-mediaani >= tama

# =====================================================================================================================
# 5. HEITTOPORTTI (heitot.py)
# =====================================================================================================================
PORTTI_MATKA_CM = 1500.0             # heittomainen liike: matka eteenpain vahintaan
PORTTI_MIN_RIVIT = 50                # ... vahintaan nain monta rivia (Y(t)-sovitusta varten)
PORTTI_HIDASTUVUUS_MIN_MS2 = 0.03    # ... hidastuu kuin kitka: Y(t)-sovituksen hidastuvuus talla valilla (m/s^2);
PORTTI_HIDASTUVUUS_MAX_MS2 = 0.20    #     aidot heitot 0,035-0,17 (live 2026-10-07, MAH00014), pelaaja ~0 tai < 0
PORTTI_MAX_RMS = 3.0                 # hyva sovitus: rms-mediaani ja tarkka-osuus
PORTTI_MIN_TARKKA = 0.3
PORTTI_PELASTUS_MAX_RMS = 12.0       # heikko sovitus (lakaisija peittaa) kelpaa viela
PORTTI_PELASTUS_MIN_TARKKA = 0.1
PORTTI_TUPLA_RUUDUT = 40.0           # kaukohogin ylitykset nain lahella = sama heitto
PORTTI_ALKU_AUKKO_RUUDUT = 30        # radan alku pois jos sen jalkeen aukko > tama ...
PORTTI_ALKU_MAX_RIVIT = 40           # ... ja alku on lyhyempi kuin tama
PORTTI_YLITYS_EKSTRAPOLOINTI_MAX = 60.0

# =====================================================================================================================
# 6. HOG-HOG -ANALYYSI, LIUKU JA IRROITUS (hog_analyysi.py)
# =====================================================================================================================
HOG_MIN_R = 0.99                     # Y(t)-sovituksen R oltava yli taman
HOG_X_MAX_RMS_CM = 2.0               # X-kurvimallin jaannosvirhe (rms, huonoimmat pois) oltava alle taman
HOG_LAHI_MARGINAALI_CM = 50.0        # sovitusalue: lahihog + 50 cm ...
HOG_KAUKO_MARGINAALI_CM = 100.0      # ... kaukohog - 100 cm
HOG_MIN_PISTEET = 40                 # vahintaan nain monta pistetta sovitusalueella
HOG_PUDOTA_HUONOIMMAT = 10           # ensimmaisen sovituksen jalkeen huonoimmin sopivat pisteet pois ja uusi sovitus
HOG_KATTAVUUS_CM = 150.0             # radan pitaa kattaa sovitusalueen paat (+-)
HOG_HIDASTUVUUS_NOPEUDELLA_MS = 1.5  # hidastuvuus ilmoitetaan kitkamallista talla nopeudella
# Kitkamallin mu(v) = A + B ln v kiintea B (heittokohtaisesti sovitetaan vain A). Mitattu yhteissovituksella: live
# 2026-10-04 (66 heittoa) paras B = -0,0011, MAH00014 (19 heittoa) -0,0027. B = -0,001 -> hidastuvuus @1,5 m/s
# muuttuu hitailla heitoilla <= 0,4 % (live) / <= 3,9 % (MAH). Paivita jos jaa/kalibrointi muuttaa B:ta selvasti.
# 2026-10-06: MAH00014 (20 heittoa, yhteinen B, hog-hog Y(t)) paras B = -0,0030 (+-0,0005) -> otettu kayttoon.
HOG_MU_B = -0.003
HOG_TALLENNA_KUVA = True             # still-kuva <csv>_hog_kivi<N>.png jokaisesta onnistuneesta analyysista

# Liuku: suora X(Y) heiton alusta kaukohog + LIUKU_LOPPU_MARGINAALI_CM asti; lahtopisteeksi lisataan hakki.
LIUKU_LOPPU_MARGINAALI_CM = 100.0
LIUKU_MIN_PISTEET = 8
HAKKI_TAKARAJASTA_CM = 183.0         # hakki 1,83 m takarajan takana
TAKARAJA_TEESTA_CM = 183.0           # takaraja 1,83 m T-viivan takana
HOG_TEESTA_CM = 640.0                # T-viiva 6,40 m hogin takana
HAKKI_SIVU_CM = 15.0                 # hakit 15 cm keskiviivasta (lasketaan erikseen X = +15 ja X = -15)

# =====================================================================================================================
# 7. KAHVA JA KIERTEET (kierre.py)
# =====================================================================================================================
KAHVA_MIN_S = 80                     # kahvan pikseli: kylläisyys >= tama ...
KAHVA_MIN_V = 60                     # ... ja kirkkaus >= tama (jaa ja graniitti ovat harmaita)
KAHVA_SAVY_TOL = 10                  # kahvan savy +- tama (OpenCV H 0-179)
KAHVA_SADE_KERROIN = 2.0             # ympyran sade = kerroin x kahvan sade (kahvan karki mukaan)
KAHVA_Z_LISA_CM = 0.0                # ympyran keskipisteen korkeus H_total:n ylapuolella
KIERRE_HIDASTUVUUS = 0.02            # pyorimisen hidastuvuus rad/s^2 (MAH00014:n 12 selvan kiven yhteinen arvo)
KIERRE_Y_MAX_CM = 2300.0             # kierrepiirre kerataan vasta kun kivi on lahempana (kahva erottuu ~23 m:sta)
KIERRE_TAYSI_Y_MAX_CM = 2300.0       # sama taysresoluutioiselle piirteelle
KIERRE_MIN_RUUDUT = 60               # vahintaan nain monta ruutua jatkuvuussuodatuksen jalkeen
KIERRE_MIN_R2 = 0.02                 # kierrearvion sovituksen vahimmaisselitysaste (sekoitetut ruudut: <= 0,002)
KIERRE_MIN_EROTTUVUUS = 2.5          # paras huippu / paras muu huippu (ei +-30 %, x2, /2)
KIERRE_P_MIN_S = 0.6                 # loppukierrosajan hakuvali (s): laaja, ei oletusta kierrosmaarasta
KIERRE_P_MAX_S = 40.0
KIERRE_P_N = 500
KIERRE_PEILI_MIN_R2 = 0.03           # peilisignaali riittava -> sen kierrosajan on osuttava samaan (muuten hylataan)
KIERRE_PEILI_TOL = 0.15              # sallittu ero kierrosajoissa (suhteellinen)

# =====================================================================================================================
# 8. PAIKALLINEN TAYSI RESOLUUTIO (live; paikallinen.py)
# =====================================================================================================================
PAIKALLINEN_TAYSI = True             # SEURANTA kivien hakualueilla kameran taydella resoluutiolla
PAIK_RENGAS_S = 10.0                 # taysresoluutioisten ruutujen rengas (s); kamerassa raaka YUY2 ~1 GB / 10 s
PAIK_MARGINAALI_CM = 8.0             # kiven alueen reunus hakualueen ympari

# =====================================================================================================================
# 9. KUVAN ESIKASITTELY (esikasittely.py)
# =====================================================================================================================
GRANIITTI_EROKYNNYS = 10.0           # taustanvaimennus: ero moodikuvaan alle taman = tausta
VARJO_V_PUDOTUS_MIN = -3.0           # varjo: V (HSV) saa pudota MIN..MAX ja silti tausta (negatiivinen = pieni nousu sallittu)
VARJO_V_PUDOTUS_MAX = 50.0
JAA_S_MAX = 22                       # jaa: S < tama ja V > JAA_V_MIN on aina tausta
JAA_V_MIN = 128
VARIPORTTI_S_MAX = 60                # varillinen tausta (mainos) ei vaimene, jos savy poikkeaa > VARIPORTTI_H_TOL
VARIPORTTI_H_TOL = 10
VALO_HARVENNUS = 4                   # valotasapainon estimoinnin pikseliharvennus
VALO_KAYTTOON_VIIVE_RUUTUA = 10      # uusi valotasapaino kayttoon tasan nain monen ruudun paasta (toistettava ajo)
STAB_MAX_SIIRTO_PX = 10.0            # stabiloinnin siirto enintaan (turvaraja)
STAB_LEVEYS = 1280                   # leveampi kuva stabiloidaan pienennettyna tahan leveyteen

# =====================================================================================================================
# 10. SUORITUSKYKY (saikeet, liukuhihna)
# =====================================================================================================================
STAB_TALLENNA = True                 # joka ruudun stabilointi (siirto, kierto, skaala, laskenta-aika) -> <pohja>_stabilointi.csv
STAB_HARVENNUS_LIVE = 10            # live: vaihekorrelaatio vain joka N. ruudusta, valit interpoloidaan (1 = joka ruutu)
STAB_HARVENNUS_KYNNYS_PX = 0.15      # ... jos avainruutujen siirrot eroavat yli taman (tai tarina jatkuu), lasketaan kaikki
STAB_SAIKEET = 2                     # vaihekorrelaatiot rinnan (liukuhihnan vaihe A)
JONON_SYVYYS = 50                    # liukuhihnan jonot (2 s puskuri vaiheiden valissa)
GPU_VAIHE_B = True                   # warp + varjosuodatus OpenCL:lla jos kaytettavissa (Intel UHD)
VIDEO_LAITTEISTODEKOODAUS = False    # videotiedoston dekoodaus laitteistolla (Windows MSMF/D3D11); mittaa read(video) ms/ruutu
RAPORTTI_VALI_RUUTUA = 1000          # edistymisraportti
MAX_RUUTU = 0                        # > 0: pysayta tahan ruutuun (testi; --max-frame)

# =====================================================================================================================
# 11. LIVE-SYOTE (live.py, main.py; muut asetukset komentorivilta --live-*)
# =====================================================================================================================
LIVE_TAAKSE_SEURANNASSA_S = 2.0      # puskurin historia seurantavaiheessa (kalibroinnissa --live-taakse-s)
LIVE_KAMERA_KATKO_S = 3.0            # kamera katsotaan kadonneeksi, kun onnistunutta lukua ei ole tullut nain kauan
                                     #   (MSMF: lukukutsu odottaa ~10 s ja palauttaa E_PENDING, kun Cam Link ei anna kuvaa)
LIVE_KAMERA_UUDELLEEN_VALI_S = 1.0   # kamera katosi (esim. USB-katko, MSMF 0xC00D3EA2): avausyritys nain usein ...
LIVE_KAMERA_UUDELLEEN_MAX_S = 600.0  # ... enintaan nain kauan, sitten syote paattyy (0 = ei uudelleenavausta)

# =====================================================================================================================
# 12. PUHELINNAKYMA (--katselu) JA DEBUG-VIDEO (katselu.py, nakyma.py)
# =====================================================================================================================
KATSELU_VALI_S = 1.0                 # puhelinnakyma piirretaan kerran nain monessa sekunnissa
KATSELU_KORKEUS = 810                # JPEG-kuvan korkeus (px); nakyma on 1080 px korkea
KATSELU_LAATU = 75                   # JPEG-laatu
KATSELU_TALLENNUS_FPS = 1.0          # tallennetun <nimi>_katselu.avi:n ruutunopeus (1 = reaaliaika)
KATSELU_LIIKE_KYNNYS = 40            # liikkuva kohde = ero taustakuvaan > tama (0-255, suurin kanava); viiva piirretaan alle
KATSELU_VIIVE_MAX_S = 180.0          # keskikuvan viivepuskuri (s): Alku voi olla enintaan tama (n. 15 Mt)
KATSELU_RISTI_SAKARA_PX = 24         # irroitusristin sakaran pituus (px)
DEBUG_KOODAUS = "auto"               # debug-videon koodaus: auto (QSV jos on, muuten mp4v), qsv, x264, opencv

# Taustasaikeet/-prosessit (live-tallennus, ffmpeg) alemmalle prioriteetille, jotta seuranta saa ytimet ensin.
TAUSTA_ALEMPI_PRIORITEETTI = True
