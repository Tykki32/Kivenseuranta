"""Kaikki saadettavat vakiot (kynnykset, parametrit, asetukset) yhdessa paikassa."""

import math


# ============================================================
# PESIEN MITAT
# ============================================================
BLUE_OUTER_RADIUS_M = 1.829


BLUE_INNER_RADIUS_M = 1.219


RED_OUTER_RADIUS_M = 0.610


RED_INNER_RADIUS_M = 0.152


BLUE_OUTER_RADIUS_CM = BLUE_OUTER_RADIUS_M * 100.0


BLUE_INNER_RADIUS_CM = BLUE_INNER_RADIUS_M * 100.0


RED_OUTER_RADIUS_CM = RED_OUTER_RADIUS_M * 100.0


RED_INNER_RADIUS_CM = RED_INNER_RADIUS_M * 100.0


HOUSE_RADIUS_CM = BLUE_OUTER_RADIUS_CM


NEAR_HOUSE_Y_CM = HOUSE_RADIUS_CM


HOUSE_DISTANCE_CM = 3474.7


FAR_HOUSE_Y_CM = NEAR_HOUSE_Y_CM + HOUSE_DISTANCE_CM


# ============================================================
# LOPULLISEN (TOP-DOWN) KUVAN ASETUKSET
#
# HUOM: Y-akseli on KAANNETTY: fyysisesti pieni Y (lahempi pesa)
# piirretaan kuvan ALAOSAAN ja suuri Y (kaukainen pesa) kuvan
# YLAOSAAN.
# ============================================================
OUTPUT_X_MIN_CM = -200.0


OUTPUT_X_MAX_CM = 200.0


OUTPUT_Y_MIN_CM = 0.0


OUTPUT_Y_MAX_CM = 4000.0


PIXELS_PER_CM = 2.0


HOUSE_CROP_HALF_HEIGHT_CM = 300.0


# Hogline (etulinja) sijaitsee curlingissa 6.4 m paassa T-linjalta
# (pesan keskipisteesta), molemmin puolin kenttaa.
HOG_LINE_DISTANCE_FROM_TEE_CM = 640.0


# ============================================================
# LAHEMMAN PESAN ELLIPSIEN HAKUASETUKSET
# ============================================================
NEAR_BLUE_MIN_AREA = 1000


NEAR_BLUE_MIN_RATIO = 0.20


NEAR_BLUE_MIN_SIZE_RATIO = 0.30


NEAR_RED_MIN_AREA = 50


NEAR_RED_MIN_RATIO = 0.15


NEAR_RED_MIN_SIZE_RATIO = 0.15


# ============================================================
# OBJEKTIIVIN VAARISTYMAN ITSEKALIBROINTI JA HOMOGRAFIAN ROBUSTIUS
#
# Yksittainen 3x3-homografia on LINEAARINEN (projektiivinen) kuvaus,
# eika se pysty korjaamaan objektiivin EPALINEAARISTA sateittaista
# vaaristymaa (barrel/pincushion). Tama nakyy jaljella olevana pesien
# soikeutena ja hogline-viivojen vinoutena, vaikka korrespondenssi-
# pisteet olisi mitattu tarkasti - homografialla ei yksinkertaisesti
# ole vapausasteita korjata epalineaarista vaaristymaa.
#
# Koska erillisia kalibrointikuvia (esim. shakkilautaa) ei ole
# saatavilla ja objektiivin zoomaus voi vaihdella kuvien valilla,
# vaaristymaa EI kalibroida etukateen kiintein kertoimin. Sen sijaan
# yhden parametrin sateittainen vaaristyma (k1) ESTIMOIDAAN JOKA
# KUVASTA ERIKSEEN suoraan jo tunnetuista pesa-/hogline-korrespons-
# sipisteista (katso estimate_radial_distortion_k1) - sama periaate
# kuin "plumb-line"-itsekalibroinnissa: tunnetun geometrian (pesat
# ovat ympyroita, hoglinet suoria ja kohtisuorassa keskilinjaa
# vastaan) pitaisi toteutua kuvassa, ja k1 valitaan niin etta se
# toteutuu mahdollisimman hyvin.
# ============================================================

# HUOM: Homografiat (talla sivulla ja compute_corrected_homography/
# main:ssa) sovitetaan method=0:lla (tavallinen pienimman nelion
# sovitus), EI RANSAC:lla. Testattu oikealla datalla: nailla ~22
# korrespondenssipisteella (17 tiiviisti ryhmittynytta lahempaa +
# 5 harvaa kaukaista) RANSAC:n satunnaisotanta osuu usein kokonaan
# lahemman pesan tiiviiseen ryhmaan, mika tuottaa lahes degeneroi-
# tuneen homografian joka ekstrapoloituu hallitsemattomasti (tuhansien
# pikselien virhe) kaukaiselle pesalle. Plain LSQ oli aina vakaa ja
# jopa tarkempi.
K1_SEARCH_RANGE = 0.6


K1_SEARCH_STEPS = 25


K1_SEARCH_REFINE_ROUNDS = 3


K1_SEARCH_REFINE_SHRINK = 6.0


# Vaaristymaa ei oteta kayttoon, jos se ei parantaisi korrespondenssi-
# pisteiden RMS-uudelleenprojisointivirhetta riittavasti - talla
# valtetaan sovittamasta "vaaristymaa" pelkkaan mittauskohinaan.
K1_MIN_RELATIVE_IMPROVEMENT = 0.08


K1_MIN_MAGNITUDE = 0.01


HOMOGRAPHY_REFINE_MAX_ITERATIONS = 15


HOMOGRAPHY_REFINE_MIN_RELATIVE_IMPROVEMENT = 0.03


# Montako PERAKKAISTA ei-parantunutta kierrosta refine_homography_
# corrections sietaa ennen pysahtymista (katso sen sisainen kommentti).
STALL_PATIENCE = 3


# ============================================================
# GEOMETRIA-APURIT
# ============================================================


# Hogline (etulinja) sijaitsee curlingissa 6.4 m paassa T-linjalta
# (pesan keskipisteesta), molemmin puolin kenttaa.
NEAR_HOGLINE_Y_CM = NEAR_HOUSE_Y_CM + HOG_LINE_DISTANCE_FROM_TEE_CM


FAR_HOGLINE_Y_CM = FAR_HOUSE_Y_CM - HOG_LINE_DISTANCE_FROM_TEE_CM


# ============================================================
# JAA-SUODATUS-POHJAINEN LISAREUNALAHDE (Testi_02_02, kayttajan
# pyynnosta - katso keskusteluhistoria): joissakin videoissa (esim.
# MAH00014, jossa lahempi pesa on aivan kuvan reunassa) hogline/
# keskiviiva on niin haalea (matala kontrasti jaata vasten) etta
# tavallinen harmaasavy-Canny (alla) ei loyda sita luotettavasti -
# se hukkuu kirkkaan jaan omaan kohinaan. Rata-alueen S/V-histogrammin
# (Testi_02_01:ssa kehitetty jaa-suodatus, main.py:n ICE_S_MAX/ICE_
# V_MIN) HAVAITTIIN paljastavan tallaisen haalean viivan selvasti
# PAREMMIN kuin Canny: jaa on tasaisen kirkasta JA matalasaturoitunutta
# (S<LINE_ICE_S_MAX, V>LINE_ICE_V_MIN), joten kaikki mika EI tayta tata
# (mukaan lukien haalea harmaa viiva, JOKA ON hieman jaata tummempi)
# erottuu binaarimaskina huomattavasti kontrastikkaammin kuin raa'an
# harmaasavykuvan gradientti. Tama LISATAAN (ei korvata) alkuperaisen
# Cannyn rinnalle - molempien lahteiden segmentit yhdistetaan, joten
# jo ennestaan toimiva videon (0001) tulos ei voi huonontua (samat
# vanhat segmentit ovat yha mukana, uudet vain lisaavat kandidaatteja).
# ============================================================
LINE_ICE_S_MAX = 22


LINE_ICE_V_MIN = 165


# ============================================================
# GEOMETRINEN LOPPUVAIHE (kamera8_01.py)
#
# KAYTTAJAN VAATIMUS: lopullinen kuvamuunnos saa koostua VAIN
# objektiivin sateittaisen vaaristyman korjauksesta (k1, jo tehty
# ennen tata - koko frame_undistorted on oikaistu KERRAN) ja YHDESTA
# homografiasta - ei rivikohtaista korjausta, ei erillista
# jalkikasittelyvaihetta (kamera7_06.py:n apply_hogline_warp_correction
# hylattiin talla perusteella).
#
# RATKAISU: pesien renkaat ja hoglinet KAYTETAAN SUORAAN GEOMETRISINA
# RAJOITTEINA sen homografian sovituksessa, sen sijaan etta niista
# rakennettaisiin keinotekoisia pistevastaavuuksia jotka vaativat
# T-linjan SUUNNAN arvaamista (kamera7_xx.py:n dir_lateral/angle-
# painotus, hauras - katso kamera7_06.py:n compute_corrected_homography).
#
#   YMPYRARAJOITE (vastaa duaalikonia Q' = H^-T Q H^-1):
#     kaukaisen pesan renkaan reunapisteet TIEDETAAN olevan tunnetun-
#     sateisella ympyralla fyysisessa tasossa, mutta EI TIEDETA mika
#     reunapiste vastaa mitakin ympyran kohtaa (rengas on rotaatio-
#     symmetrinen) - pistekohtaista vastaavuutta ei siis voi eika
#     tarvitse muodostaa. Toteutettu pistejoukon RAJA-ARVONA: jokaisen
#     havaitun reunapisteen jaannos on sen ETAISYYS tunnettuun
#     ympyraan (ei etaisyys mihinkaan yksittaiseen kohdepisteeseen).
#     Tama on rajalla (paljon pisteita) sama rajoite kuin Q':n
#     algebrallinen yhtaloryhma, mutta ei vaadi konimatriisien
#     kasittelya.
#
#   VIIVARAJOITE (vastaa l' = H^-T l): hoglinen fyysinen Y-koordinaatti
#     tunnetaan tasan, mutta X ei (poikittainen viiva koko radan
#     leveydelta). Jokaisen havaitun hogline-pisteen jaannos on sen
#     etaisyys (Y-suunnassa) tunnettuun fyysiseen viivaan - ei vaadita
#     yksittaisen pisteen X-kohdetta.
#
# Molemmat rajoitteet YHDISTETAAN lahemman pesan 17 tavallisen piste-
# korrespondenssin kanssa YHTEEN epalineaariseen pienimman nelion
# sovitukseen (koko H, 8 vapausastetta, kerralla) - kaikki jaannokset
# FYYSISESSA YKSIKOSSA (cm), joten piste-/ympyra-/viivarajoitteiden
# VALINEN painotus tulee luonnostaan oikeaksi ilman keinotekoisia
# painokertoimia (piste antaa 2 residuaalia koska koko 2D-sijainti
# tunnetaan, ympyra-/viivapiste antaa 1 koska vain etaisyys tunnetaan -
# vastaa kunkin havainnon todellista informaatiosisaltoa; vrt. aiempi
# HOUSE_POINT_WEIGHT=6-viritys kamera7_05/06.py:ssa, joka ei ollut
# taman kaltaiseen fyysiseen perusteluun sidottu).
#
# Koska kaikki pisteet pidetaan TASSA silmukassa yhden kiinteän k1:n
# jo oikaisemassa kuvassa (frame_undistorted, katso main()), tata
# vaihetta EI tarvitse ajaa objektiivin vaaristyman uudelleenarvioinnin
# vuoksi (kamera7_xx.py:n "yhteisoptimointi") - kayttajan oman
# arvion mukaan objektiivikorjaus ei ole enaa kriittinen kun H
# lasketaan oikein, joten k1 estimoidaan vain KERRAN, ennen tata
# silmukkaa.
# ============================================================

# Ylaraja luottamukselle jota kaukaisen pesan rengas-/hogline-pisteet
# voivat saada painotuksessa (katso _robust_scale_cm) - lahemman pesan
# 17 pistetta (T-linjan/keskilinjan suoralla leikkauksella saatuja)
# ovat kaytannossa kohinattomia, ja niiden havaittu jaljelle jaava
# sovitusvirhe (plain DLT: ~1-2 cm, katso testaus) antaa tasta
# luonnollisen mittakaavan: mikaan muu havaintoryhma ei voi olla
# NAITA luotettavampi, joten sen paino ei voi ylittaa 1/NEAR_TRUST_FLOOR_CM.
NEAR_TRUST_FLOOR_CM = 1.0


# ============================================================
# KIVEN NIMELLISMITAT (WCF) - kaytetaan VAIN fit_stone_profile:in
# alkuarvauksena (lahtokohta hienosaadolle) ja jarkevyystarkistukseen.
# Seka sade etta korkeus RATKAISTAAN oikeasti havainnoista - katso
# fit_stone_profile ja sen ylla oleva kommentti kovakoodatusta
# karkeasta 3D-mallista. STONE_HEIGHT_CM/STONE_HEIGHT_MAX_CM ovat
# korkeuden ALA-/YLARAJA (WCF-vahimmaiskorkeus ... kayttajan antama
# 15cm katto) - itse H_total SOVITETAAN naiden valiin, ei kiinniteta.
# ============================================================
STONE_NOMINAL_RADIUS_CM = 91.44 / math.pi / 2.0


STONE_HEIGHT_CM = 11.43


STONE_HEIGHT_MAX_CM = 15.0


# ============================================================
# KIVEN (GRANIITTIOSAN) TUNNISTUS RAAKAKUVASTA
#
# Graniitti on tumma ja VAHASATURAATIOINEN (harmaa/musta) - sama
# vahasaturaatio kuin jaalla, joten erottelu ei voi perustua pelkkaan
# saturaatioon (S). Sen sijaan graniitti on PAIKALLISESTI TUMMEMPI
# kuin ymparoiva jaa - sama "paikallinen tummuus taustaan nahden"
# -periaate kuin kamera8_01.py:n detect_hogline_points kayttaa
# (GaussianBlur-taustavahennys), tassa 2D-versiona koko kuvalle.
# Varikas kahva (korkea saturaatio) suljetaan pois eksplisiittisesti.
# ============================================================
STONE_MAX_SATURATION = 60


STONE_DARKNESS_SIGMA = 25


STONE_MIN_DARKNESS = 15.0


# Suodatinrajat find_stone_candidates:lle - katso sen docstring: nama
# eivat ole mielivaltaisia, ne on haettu Kivilla.png:n 3 tunnetun kiven
# perusteella (katso kehityshistoria: ilman jaatasoporttia+muotosuodatinta
# tunnistus loysi 101+ virhekandidaattia - tekstia, viivoja, lakimiehen/
# pyyhkijan vaatteita; jaataso+muoto -yhdistelmalla juuri oikeat 3 jaa).
STONE_MIN_FILL_RATIO = 0.65


STONE_MIN_ASPECT_RATIO = 0.35


STONE_SHEET_MARGIN_CM = 250.0


# ============================================================
# KIVEN SEURANTA VIDEOSTA - USEITA KYMMENIA/SATOJA HAVAINTOJA YHDESTA
# KIVESTA LAAJALTA KULMA-ALUEELTA
#
# Kayttajan havainto: yksittainen liikkuva kivi lipuu heittopaasta
# (lahes vaakatasosta, pieni korkeuskulma - kamera kaukana, matala
# projektio) pesaa kohti (jyrkempi korkeuskulma, n. 15-20 astetta),
# eli SAMA fyysinen kivi antaa VALTAVASTI enemman kulmavaihtelua kuin
# 3 paikallaan olevaa kivea samassa kuvassa (jotka olivat kaikki
# suunnilleen samalla etaisyydella, n. 10-25 astetta). Tama on
# TARKEA, koska fit_stone_profile:in H_total-skannaus (katso sen
# kommentti) osoitti RMS:n olevan LITTEA H:n suhteen juuri PUUTTUVAN
# kulmavaihtelun takia - videosta saatu laajempi kulma-alue voi siis
# aidosti parantaa korkeuden erottelukykya.
#
# TUNNISTUSSTRATEGIA: sama create_granite_mask/find_stone_candidates
# -pohjainen tunnistus kuin still-kuville, mutta VAPAAMMILLA muoto-
# rajoilla (kivi on video kompressoinnin+liike-epaterävyyden takia
# usein huonommin rajautunut kuin still-kuvassa, ja PIENI kaukainen
# kivi jaa muuten helposti tiukkojen rajojen alle) - erottelu vaarista
# kandidaateista (pyyhkija, kiinteat jo-paikallaan-olevat kivet,
# kiinteat virhelahteet kuten logotekstit) EI perustu muotoon vaan
# JATKUVUUTEEN: valitaan aina se kandidaatti joka on LAHIMPANA
# edellisen (kasitellyn) framen sijaintia, hylataan jos hyppy on liian
# suuri (max_jump_cm) - staattiset objektit (mukaan lukien kiinteat
# virhekandidaatit) EIVAT liiku framesta toiseen, joten ne eivat
# yleensa hairitse jatkuvuuspohjaista seurantaa kunhan max_jump_cm on
# jarkevasti mitoitettu kiven todelliseen nopeuteen nahden.
#
# Kaytannossa toimivaksi havaittu (00011 - Trim.mp4): seurataan
# MOLEMPIIN suuntiin (framet kasvaen JA vahentyen) yhdesta hyvasta
# "siemen"-havainnosta - katso track_stone_in_video:in kutsuesimerkki
# taman tiedoston main()in tai kommenttien yhteydessa.
# ============================================================
STONE_TRACK_MIN_AREA = 80


STONE_TRACK_MIN_FILL_RATIO = 0.4


STONE_TRACK_MIN_ASPECT_RATIO = 0.15


STONE_TRACK_MAX_JUMP_CM = 45.0


STONE_TRACK_MAX_MISSES = 10


# Kahvan (muovi+kiinnitys) aiheuttaman kolon sade suhteessa R_max:iin.
# Kolon SIJAINTI on kayttajan pyynnosta MAARITELTY (ei sovitettu):
# KESKITETTY kiven pystyakselille (X0,Y0), korkeudella z=H_total - ei
# siirretty sivuun, koska kahvan tarkka atsimuuttikulma vaihtelee
# heitosta toiseen (kivi pyorii liu'un aikana) eika ole taten
# luotettavasti paateltavissa. SADE SEN SIJAAN ON (kayttajan pyynnosta,
# katso keskusteluhistoria - yhden kiven koko liu'un kattava seuranta
# antaa riittavasti kulmavaihtelua/dataa taman luotettavaan
# ratkaisuun) fit_stone_profile:in vapaa, sovitettu parametri - tama
# HANDLE_NOTCH_R_FRAC-vakio toimii enaa VAIN (1) sovituksen
# alkuarvauksena ja (2) stone_tracker.cpp:n elavan seurannan KIINTEANA
# arvona (C++-puolen notch-vahennysta EI ole viela tehty ajonaikaisesti
# parametroitavaksi - katso stone_tracker.cpp:n oma HANDLE_NOTCH_R_FRAC-
# kommentti). Sovitettu arvo on kaytannossa havaittu hyvin lahella
# tata (~0.68-0.70), joten ero on pieni.
HANDLE_NOTCH_R_FRAC = 0.70


HANDLE_NOTCH_R_FRAC_MIN = 0.30


HANDLE_NOTCH_R_FRAC_MAX = 0.95


# Kuinka voimakkaasti muotoa (kontrollipisteiden sateet) rangaistaan
# poikkeamasta kovakoodattuun mallinnukseen nahden (yksikko: "pikselia
# per r_frac-yksikko" - katso fit_stone_profile). Pitaa hienosaadon
# LAHELLA fysikaalisesti jarkevaa lahtokohtaa, estaa ylisovituksen
# kohinaisen aariviivan yli.
#
# HUOM (paivitetty - katso git-historia): kun profiilin KAIKKI
# kontrollipisteet vapautettiin (ei enaa vain "paivan ylapuoliset"),
# vapaita muotoparametreja oli 5 - reilusti enemman kuin ennen (2-3).
# Alkuperainen 25 (viritetty vanhalle, suppeammalle mallille) antoi
# 3 kiven aineistolla epatasaisia, lievasti epafyysisia tuloksia
# (pieni "olkapaa"-kohouma profiilissa) - nostettu 60:een, joka pitaa
# muodon sileampana/uskottavampana samalla kun sallii aidon korjauksen.
# (Symmetriapakon jalkeen vapaita muotoparametreja on enaa PUOLET
# taman verran - vain alapuoliskon kontrollipisteet, paiva pois lukien
# - ylapuolisko peilataan, ei sovita erikseen. Painoa ei ole viritetty
# uudelleen taman muutoksen jalkeen.)
STONE_SHAPE_REG_WEIGHT = 60.0


# ============================================================
# HAKUALUE ("kauempi paa, keskiviivasta +-70cm, hogline-1m...hogline+3m")
#
# Kayttajan pyynnosta kavennettu/kohdennettu (katso keskusteluhistoria):
# AIEMMIN hogline...pesan takareuna (tangentti 12-jalkaisen renkaan
# taakse, HOUSE_RADIUS_CM paan pesan keskipisteesta poispain hoglinesta),
# SITTEN hogline...hogline+3m - koska kivi ylittaa hoglinen aina heiton
# alkupaassa, 3m riittaa kattamaan sen EIKA alue enaa ulotu asti pesan
# kohdalle (jossa pelaajat/pyyhkijat useimmin seisovat/kavelevat, katso
# ENABLE_SHADOW_TOLERANT_STABILIZATION-tyon empiirinen havainto). X-
# leveys kasvatettu 50->70cm HAKU:n omaksi turvamarginaaliksi. Kaukainen
# paa (FAR_*) koska talla videolla kivi heitetaan lahempaa hoglinea
# kohti sita - katso kamera9_01.py:n track_stone_in_video-kommentti
# samasta videosta.
#
# NYT (kayttajan pyynnosta) kasvatettu hieman TAAKSEPAIN: hogline-1m...
# hogline+3m - alaraja ulottuu 100cm hoglinen TAAKSE (kohti kaukaista
# pesaa) kattaakseen myos kiven joka on juuri ja juuri ylittanyt
# hoglinen jo ennen ensimmaista hakutarkistusframea.
# ============================================================
SEARCH_X_HALF_WIDTH_CM = 70.0


SEARCH_Y_MIN_CM = FAR_HOGLINE_Y_CM - 100.0


SEARCH_Y_MAX_CM = FAR_HOGLINE_Y_CM + 300.0


# Karkea->hieno ristikkohaku (sama periaate kuin kamera8_01.py:n
# search_far_house) - kaksi tasoa riittaa, koska kolmas (ultra-hieno)
# taso ei enaa muuta lopputulosta merkittavasti mutta maksaa yhta
# paljon kuin ensimmainen.
SEARCH_COARSE_STEP_CM = 10.0


SEARCH_FINE_STEP_CM = 2.0


# Karkean ristikkohaun ESIKARSINTA (kayttajan pyynnosta laskettu, katso
# keskusteluhistoria): tama on VAIN nopea karsinta ENNEN yhteissovitusta
# (refinePositionJoint) - lopullinen hyvaksynta tehdaan stone_tracker.
# cpp:n HAKU_ACCEPT_SCORE_THRESHOLD:lla VASTA sovituksen jalkeen (samalla
# peitto-/ulkopuoli-pisteytyksella, katso sen oma kommentti). Pidetty
# TAHALLAAN MATALAMPANA kuin lopullinen kynnys, jotta kaukana/pienena
# nakyvat AIDOT kivet (epatarkka karkea ristikko-osuma ennen hienosaatoa)
# eivat karsiudu pois jo tassa vaiheessa - havaittu oikealla MAH-videolla
# etta vanha 0.55 esti useiden aitojen, 20m+ liu'un tehneiden kivien
# rekisteroinnin taalla kaukovyohykkeella.
SEARCH_SCORE_THRESHOLD = 0.15  # vain esikarsinta, katso yllaoleva kommentti


TRACK_COARSE_STEP_CM = 7.0


TRACK_FINE_STEP_CM = 1.5


TRACK_SCORE_THRESHOLD = 0.35  # matalampi - jatkuvuus on jo vahva prior


TRACK_LOST_MAX_MISSES = 5     # montako peräkkäistä huonoa framea ennen kuin palataan hakuun


# Pienempi resoluutio hylylle HAUSSA/SEURANNASSA (nopeampi, riittava
# karkean sijainnin loytamiseen - ei tarvita samaa tarkkuutta kuin
# esim. profiilin sovituksessa).
SEARCH_HULL_N_THETA = 14


SEARCH_HULL_N_PER_SEGMENT = 2


# ============================================================
# KAUKAISEN PESAN LOYTAMINEN + KOKO HOMOGRAFIAN RATKAISU
# (Testi_01_02, kayttajan pyynnosta - korvaa Testi_01_01:n mallipohjaisen
# COARSE/FINE/ULTRA-haun, joka osoittautui liian herkaksi skaala-syvyys-
# degeneraatiolle ja mainospaneelien vaareille osumille tallä kamera-
# kulmalla)
#
# Uusi menetelma (kehitetty/validoitu askel askeleelta yhdessa kayttajan
# kanssa Testi_01_01:n pohjalta, katso sen keskusteluhistoria):
#
# 1) Rakennetaan homografia (H_near_only) PELKASTAAN lahemman pesan 17
#    pisteesta (near_house_correspondences), laajennettuna kattamaan
#    koko rata + 10m takarajan yli. Tama ekstrapolointi on tiedetysti
#    epatarkka kaukana (Y-suunnassa), mutta X-suunnassa (sivuttain)
#    riittavan luotettava karkeaan paikannukseen.
# 2) Skannataan tama laajennettu topdown rivi kerrallaan: etsitaan rivi
#    jolla on sinista maskia keskiviivan MOLEMMIN puolin ja punaista
#    niiden VALISSA - tama kuvio loytyy vain kaukaisen pesan (ja
#    lahemman pesan, joka suljetaan pois) kohdalta, ei muualta radalta.
#    Loydettyjen rivien suurin yhtenainen klusteri = karkea arvio
#    kaukaisen pesan Y-sijainnista (ja sen X-keskikohdasta).
# 3) "Skaalataan" (affiini pystysuora venytys/puristus, ankkuroituna
#    lahempaan pesaan Y=0:ssa) laajennettu topdown niin etta loydetty
#    rivi osuu TARKALLEEN sille riville jolla FAR_HOUSE_Y_CM pitaisi
#    olla - karkea arvio nayttaa taten OIKEASSA fyysisessä Y-kohdassa,
#    vaikka nain saatu asteikko onkin viela epatarkka.
# 4) Muodostetaan alustava homografia (H_v1) lahemman pesan 17 pisteesta
#    + tasta loydetyn kaukaisen pesan KESKIPISTEESTA (fyysinen oletus:
#    (0, FAR_HOUSE_Y_CM), koska curling-saannot pakottavat pesan
#    keskiviivalle tasan tunnetulle etaisyydelle).
# 5) H_v1:sta alkaen ajetaan sama geometrinen tarkennussilmukka jota
#    kamera8_01.py:n refine_geometric_homography jo kayttaa (ympyra-
#    rajoitteet kaukaiselle renkaalle + viivarajoitteet molemmille
#    hoglineille, ratkaistuna solve_homography_geometric:illa) - MUTTA
#    kaukaisen renkaan tunnistus KORVATTU varipohjaisella (HSV-pinta-
#    alaosuuteen saadetty kynnys + sateittainen reunanhaku, katso
#    far_ring_points_color_based) ja hoglinen tunnistus KORVATTU
#    harmaasavy-kynnys-menetelmalla (katso hogline_points_gray_
#    threshold) - molemmat kayttajan kanssa erikseen todettu
#    luotettavammiksi talla kamerakulmalla kuin kamera8_01.py:n omat
#    (radiaalihaku+pakotettu-yhteismuoto, segmenttipohjainen hogline-
#    haku).
#
# Validoitu (2026-09) molemmilla testikuvilla (leikattu_kalibrointi_
# moodikuva.png, 00011_kalibrointi_moodikuva.png): molempien hoglinien
# kulmaero < 0.02 astetta (aiemmin jopa 47 astetta vaarin), kaukaisen
# pesan pyoreys 0.94-1.00, koko-virhe alle 3%.
# ============================================================
FAR_HOUSE_ROW_SCAN_MAX_GAP_PX = 15


FAR_HOUSE_ROW_SCAN_CENTER_MARGIN_PX = 5


FAR_HOUSE_ROI_TARGET_BLUE_FRACTION = 0.31


FAR_HOUSE_ROI_TARGET_RED_FRACTION = 0.11


HOGLINE_GRAY_HALF_BAND_CM = 50.0


HOGLINE_GRAY_TARGET_COVERAGE = 0.98


# Kayttajan pyynnosta: debug-ominaisuus joka tallentaa elavan moni-
# kiven seurannan ajalta UUDEN videotiedoston (<video>_debug_seuranta.
# mp4), jossa jokaisen tunnistetun/seuratun kiven ENNUSTETTU ääriviiva
# (predicted_stone_hull_fast - sama profiilimalli jota itse
# yhteissovituskin kayttaa) on piirretty framen paalle, jotta nakee
# SUORAAN missa/miksi HAKU tai SEURANTA tunnistaa jotain vaarin
# (esim. pelaajan kivena). Vihrea=tarkka, oranssi=ei-tarkka, keltainen
# =juuri HAKU:n loytama uusi kivi, punainen=SEURANTA hukkasi taman
# framen (piirretaan viimeisimpaan tunnettuun sijaintiin). HIDASTAA
# ajoa (VideoWriter-enkoodaus joka framella) - pida False normaali-
# ajoissa, aseta True vain debugatessa. Voidaan myos kytkea paalle
# dynaamisesti komentorivilta TATA VAKIOTA muokkaamatta: aja
# "python main.py --debug" (tai "-d") - katso if __name__=="__main__".
#
# Kayttajan pyynnosta: debug-videoon piirretaan moodikuvalla suodatettu
# frame_u_for_tracking (se mita HAKU/SEURANTA oikeasti kayttaa
# tunnistukseen - katso ENABLE_SHADOW_TOLERANT_STABILIZATION), EI
# alkuperaista frame_u:ta - nain nakee suoraan mita tunnistus itse
# asiassa "nakee".
DEBUG_SAVE_TRACKING_VIDEO = False


WALL_OFFSET = 40


GRAY_RADIUS = 2


HISTORY_LENGTH = 10


MORPH_SIZE = 3


MIN_COMPONENT_AREA = 100


MAX_COMPONENT_AREA = 250000


SUBPIX_WINDOW = (5, 5)


MAX_POSITION_ERROR = 20.0


REPORT_EVERY = 1000


MAX_WORKERS = 8


ROI_HALF_WIDTH = 100


ROI_HALF_HEIGHT = 120


# ============================================================
# GRANIITTIMASKIN TAUSTANVAIMENNUKSEN KYNNYSARVO (Testi_02_01, havaittu
# oikean Testivideo-julkaisun analyysissa - katso keskusteluhistoria):
# stone_tracker.cpp:n search_new_stone/track_stones_batch ottavat
# diff_threshold-parametrin, mutta main.py EI KOSKAAN antanut sita
# eksplisiittisesti ennen tata - kaytossa oli siis hiljaa C++:n oma
# OLETUSARVO 30.0 (katso PYBIND11-RAJAPINTA). Tama havaittiin LIIAN
# LOYHAKSI juuri lahemman pesan rengaskuvion (vaalea/tumma sinisten
# kolmioiden mosaiikki) kohdalla: diagnoosissa (kaksi oikeaa, koko
# matkan onnistuneesti seurattua kivea, molemmat menettivat SEURANNAN
# tasan pesan reunalla) graniittimaski oli arvolla 30 JOKO pirstoutunut
# ohuiksi viivoiksi TAI TAYSIN TYHJA (0 pikselia!) juuri niissa
# kohdissa missa kivi lepasi tumman kolmion paalla - koska diff_
# threshold=30 valkaisi (tulkitsi taustaksi) MYOS osan itse kivesta,
# ei vain oikeaa taustaa. Tiukempi (PIENEMPI) kynnys saattaa vain
# LAHELLA moodikuva-arvoa olevat pikselit taustaksi - kayttajan
# ehdottama korjaussuunta, vahvistettu vertailulla (+-3/+-6/+-10/+-20/
# +-30): arvolla 10 maski pysyi yhtenaisena molemmissa aidon kiven
# pysahdyskohdissa (725->1852px ja 0->809px), ilman etta+-3/+-6:n
# ylimaarainen kohina (94 vs 22 yhtenaista aluetta koko framessa) tulisi
# mukaan. HUOM: tama vaikuttaa SEKA HAKUun etta SEURANTAAN (molemmat
# saavat saman arvon alla) - HAKU on tasta herkempi (skannaa koko
# vyohykkeen joka sekunti, ei vain pientä paikallishakua kuten SEURANTA),
# joten jos vaarat HAKU-loydot lisaantyvat jatkossa, tata voi eriyttaa
# HAKU:lle omaksi (loyhemmaksi) arvokseen - MIN_CONFIRMED_THROW_
# DISPLACEMENT_CM (myohemmin tassa tiedostossa) antaa jo suojan naita
# vastaan.
# ============================================================
GRANITE_DIFF_THRESHOLD = 10.0


# ============================================================
# VARJONSIETOINEN TAUSTANVAIMENNUS + JOKA-FRAME SUB-PIKSELI-
# KOHDISTUS (kayttajan pyynnosta, katso keskusteluhistoria):
# empiirisesti havaittu etta paneiliseurantaan perustuva stabilointi
# EI ole taysin sub-pikseli-tarkka, ja etta myos VARJOSTUNEET (ei
# vain suoraan moodikuvaa vastaavat) pikselit kannattaa tulkita
# taustaksi - yhdessa nama auttoivat loytamaan JA pitamaan kiinni
# testivideon (0001.mp4, "Testivideo"-julkaisu) ENSIMMAISESTA
# heitosta, joka aluksi nakyy vain muutamana irrallisena pikselina
# pelaajan vierella (suurin osa kivesta pelaajan peitossa). Ilman
# tata HAKU ei loytanyt kyseista heittoa LAINKAAN; taman kanssa
# SEURANTA piti kiinni siita KOKO matkan (n. 22.7s) levolle asti -
# lopuksi jopa kiven kahva erottui selvasti, vahvistaen etta koko
# ajan seurattiin oikeaa kivea.
#
# HELPPO POISTAA KAYTOSTA: aseta ENABLE_SHADOW_TOLERANT_STABILIZATION
# = False - talloin varjonsietoinen taustanvaimennus palaa TASMALLEEN
# aiempaan kayttaytymiseen (raaka diff_threshold, ei varjosaantoa) -
# katso kayttokohta suppress_static_background():ssa.
#
# REHELLINEN VARAUS: tama on toistaiseksi validoitu vain YHDEN erittain
# hankalan (voimakkaasti okkludoidun) heiton osalta tarkalla, kasin
# ohjatulla diagnoosilla - EI VIELA koko videon lapikaynnilla A/B-
# vertailuna (kuten esim. GRANITE_DIFF_THRESHOLD/BODY_CONTOUR-poisto
# tassa samassa tiedostossa). Suositus: aja koko video lapi seka
# paalla etta pois paalta (tools/review_tracks.py) ennen kuin luotat
# tahan oikeassa ottelussa.
# ============================================================
ENABLE_SHADOW_TOLERANT_STABILIZATION = True


# Kayttajan kanssa kasin lasin 4:15/4:57-frameilla (MAH00014) testatut
# raja-arvot - katso keskusteluhistoria: -3<v_drop<50 antoi hieman
# paremman (pienemman fg-vuodon) tuloksen kuin alkuperainen 0<v_drop<=30
# SAMALLA color_diff<10-arvolla molemmissa testatuissa frameissa.
SHADOW_V_DROP_MIN = -3.0  # kuinka paljon V saa NOUSTA (negatiivinen pudotus) ja silti tulkita taustaksi


SHADOW_V_DROP_MAX = 50.0  # kuinka paljon V (HSV) saa pudota ja silti tulkita taustaksi/varjoksi


# ============================================================
# ABSOLUUTTINEN JAA-SUODATUS (S/V) - kayttajan pyynnosta, katso
# keskusteluhistoria: moodikuvasta (VAIN rata-alueen pikselit, rajattu
# sadetasoleikkauksella fyysisiin rajoihin OUTPUT_X/Y_MIN/MAX_CM:aan,
# EI koko kameranakymaa - laidat/katto/pesamainokset olisivat vaaris-
# taneet jakaumaa) laskettu S/V-2D-histogrammi paljasti etta selva
# enemmisto (~68%) radan omista jaapikseleista tayttaa S<22 & V>128:n.
#
# TARKISTETTU (kayttajan pyynnosta) etta tama EI syo kivea merkittavasti:
# alkuperainen epailys (kiven kirkkaat pikselit osuisivat samaan
# alueeseen) osoittautui SUURELTA OSIN oman rajausvirheen (liian valjan
# kivimaskin, joka sisalsi kiven ymparilla olevaa sumeaa jaa/kiiltohaloa)
# aiheuttamaksi - Canny-reunantunnistus vahvisti ettei kiven ja jaan
# valilla ollut edes oikeaa gradienttia siina kohtaa, ja pikseliarvot
# (~V170-186) vastasivat suoraan jaata (~V163), eivat kiven omaa tummaa
# ydinta (~V50). Tiukalla, todelliseen reunaan sovitetulla kivimaskilla
# haviota oli VAIN n. 0.1% kiven pikseleista (molemmat testikivet,
# stone_check_frame.png) - tama on positio-RIIPPUMATON suodatus (ei
# vertaa samaan pikseliin moodikuvassa kuten ylla olevat kaksi), joten
# se auttaa NIMENOMAAN tapauksissa joissa valotasapaino/kirkkaus on jo
# korjattu mutta yksittainen pikseli silti eroaa referenssista (esim.
# heijastus/kiilto joka liikkuu framen mukana) - katso myos ylla oleva
# valotasapainokorjaus, joka jo hoitaa suurimman osan hitaasta ajautu-
# masta ETUKATEEN, joten tama on lisasuoja sen PAALLE, ei korvaa sita.
# ============================================================
ICE_S_MAX = 22


ICE_V_MIN = 128


# ============================================================
# JOKA-FRAME SUB-PIKSELI-KOHDISTUS - OMA, ERILLINEN lippunsa (irrotettu
# ENABLE_SHADOW_TOLERANT_STABILIZATIONista, kayttajan pyynnosta: katso
# keskusteluhistoria). ENSIMMAINEN versio mittasi korjauksen VAIN
# yhdesta kiintesta pienesta ankkuripisteesta (kaukaisen pesan takana)
# ja sovelsi sen koko kuvaan jaykkana muunnoksena - koko videon
# lapikaynti paljasti etta tama HUONONSI seurannan vakautta muualla
# kuvassa (erityisesti lahella pesaa, kaukana ankkurista): pieni
# paikallinen korjaus ei ollut edustava koko kuvalle, ja kiertovirhe
# (kuvan keskipisteen ympari) vahvistui etaisyyden mukana.
#
# KORJATTU (kayttajan ehdotuksesta): kohdistus haetaan nyt KOKO NAYTON
# pikseleita vasten moodikuvaa vasten, taysresoluutioisena (kayttajan
# huomio: kuvan pienentaminen tuhoaisi juuri sub-pikseli-tarkkuuden
# jota haetaan). cv2.phaseCorrelate (FFT-pohjainen vaihekorrelaatio)
# laskee koko framen sub-pikseli-tarkan TRANSLAATION yhdella kutsulla -
# ei enaa ristikkohakua eika kiertoa (katso _phase_correlate_full_frame).
#
# PAALLA (kayttajan pyynnosta): validoitu koko videon lapikaynnilla
# (SUBPIXEL_ALIGN_CROP_FRACTION=0.67:lla) - ei enaa karkaavia haamuja,
# kaikki viisi tunnistettavissa olevaa oikeaa heittoa loytyivat. HUOM:
# lisaa merkittavasti laskenta-aikaa - katso NOPEUSSEURANTA-raportin
# oma rivi. Kayttajan pyynnosta SUBPIXEL_ALIGN_CROP_FRACTION nostettu
# takaisin 1.0:aan (koko rata) taman validoinnin jalkeen - EI VIELA
# uudelleenvalidoitu koko videon lapikaynnilla taysikokoisena.
#
# ENABLE_SUBPIXEL_ALIGNMENT-LIPPU POISTETTU (kayttajan havainto + oma
# empiirinen vahvistus, katso keskusteluhistoria "loppuvideon" PIKSELI-
# TARKKA SUORA STABILOINTI -kommentin kohdalla): talla lipulla ohjattu
# jalkikorjaus (estimate_subpixel_alignment alla) mittasi ALKUPERAISESTI
# pienen residuaalin epatarkan paneilipohjaisen stabiloinnin PAALLE -
# nyt kun paastabilointi TEKEE JO TASMALLEEN saman suoran moodikuva-
# vertailun, jalkikorjaus vain toisti saman mittauksen ja kasvatti
# virhetta (havaittu: taustanvaimennuksen reunavuoto +54% jalkikorjauksen
# kanssa vs. ilman). estimate_subpixel_alignment/_phase_correlate_full_
# frame jataan silti talteen - JALKIMMAINEN on edelleen KAYTOSSA "loppu-
# videon" paastabiloinnin OMANA ytimena (katso sen kaytto alempana).
# ============================================================
SUBPIXEL_ALIGN_RANGE_PX = 10.0  # turvaraja - katso _phase_correlate_full_frame (kayttajan pyynnosta 10x, oli 1.0)


# Kuinka suuri, kuvan keskelle keskitetty osuus (leveys JA korkeus)
# kaytetaan vaihekorrelaatioon - EI pienennys, vain RAJAUS (resoluutio
# sailyy taysimittaisena) - katso _phase_correlate_full_frame. Kayttajan
# pyynnosta 1.0 (koko rata/koko kuva, EI vain keskustaa) - nopeuden
# hillitsemiseksi aiemmin kokeiltu 0.67 (~38ms/ruutu) hylattiin, koska
# stabiloinnin pitaa kattaa koko radan, ei vain kuvan keskiosaa. Koko
# kuvan kaytto maksaa enemman (mitattu taysikokoisena ~130-140ms/ruutu
# tassa Linux-diagnoosiymparistossa).
SUBPIXEL_ALIGN_CROP_FRACTION = 1.0


STABILIZATION_MEDIAN_FRAMES = 20


# Moodikuvan laskenta tehdään paloissa
TILE_SIZE = 128


# ============================================================
# PANEELIEN AUTOMAATTITUNNISTUKSEN TOLERANSSI
#
# Kayttajan pyynnosta: paneelit ovat "samantyylisesti samalla
# seudulla", mutta kamera sijoitetaan/zoomataan aina hieman eri
# tavalla kasin, joten niiden sijainti VAIHTELEE hieman referenssiin
# nahden. detect_panel (alla) hakee jo omasta ROI:staan (ROI_HALF_
# WIDTH/HEIGHT = 100x120px) lahimman tummasuorakulmion annetun
# pisteen ymparilta - PANEL_REFERENCE_SEARCH_SCALE suurentaa taman
# haun VAIN referenssipohjaiselle automaattitunnistukselle (ei
# vaikuta paneiliSEURANTAAN framejen valilla, joka kayttaa detect_
# panel:ia normaalilla ROI:lla edellisen sijainnin ymparilta - kuvan
# drifti framejen valilla on paljon pienempi kuin kameran koko
# uudelleensijoittelun tuoma ero referenssiin).
# ============================================================
PANEL_REFERENCE_SEARCH_SCALE = 10


# ============================================================
# AUTOMAATTINEN KALIBROINTI - MOODIKUVA-POHJAINEN
#
# Kayttajan pyynnosta: sen sijaan etta kalibroidaan yhdesta
# raakaframesta (jossa voi olla liikkuvia kivia/pelaajia hairitsemassa
# lahemman pesan renkaiden tunnistusta), kerataan STABILOITUJA (mutta
# EI perspektiivikorjattuja - katso bootstrap_calibration:in kommentti
# MIKSI) frameja CALIB_MODE_SAMPLE_INTERVAL_SECONDS valein videon
# ensimmaisen CALIB_MODE_DURATION_SECONDS ajalta, ja lasketaan niista
# TILE_SIZE-kokoisittain moodikuva (olemassa oleva C++-mekanismi,
# sama add_mode_frame/start_mode_background/wait_for_mode kuin
# ENABLE_MODE_FILTER-ominaisuudessa) - liikkuvat kohteet haviavat,
# jaljelle jaa puhdas jaakuva jota vasten calibrate_camera_from_image
# (kamera9_01.py) toimii huomattavasti luotettavammin.
# ============================================================
CALIB_MODE_DURATION_SECONDS = 60.0


CALIB_MODE_SAMPLE_INTERVAL_SECONDS = 5.0


# ============================================================
# KOKO RADAN SKANNAUS LIIKKUVAN KIVEN LOYTAMISEKSI (3D-PROFIILIA
# VARTEN)
#
# Kayttajan pyynnosta: videon alussa kivi voi liikkua kumpaankin
# suuntaan eika sen sijaintia tiedeta etukateen - skannataan siis
# KOKO fyysinen rata (ei kamera9_02.py:n kiinteaa paata-rajattua
# HAKU-vyohyketta) STONE_SCAN_INTERVAL_SECONDS valein kunnes jotain
# loytyy - "ripea silloin kun vain odottaa kiveä" (kayttajan sanoin):
# harva aikavali pitaa taman odotusvaiheen halpana.
# ============================================================
STONE_SCAN_INTERVAL_SECONDS = 2.0


# HUOM: find_stone_candidates suodattaa jo OLETUKSENA koko radan
# fyysisiin rajoihin (OUTPUT_X/Y_MIN/MAX_CM + sheet_margin_cm) -
# ei tarvita omaa erillista rajausta, "koko rata" tulee ilmaiseksi.

# Kahden peräkkäisen skannauksen (STONE_SCAN_INTERVAL_SECONDS valein)
# valinen siirtyma, jonka ylittava sama (lahin) kandidaatti tulkitaan
# AIDOSTI LIIKKUVAKSI kiveksi (ei paikallaan olevaksi) - hidaskin
# liu'un loppuvaihe siirtyy enemman kuin tama 2 sekunnissa.
STONE_MOTION_THRESHOLD_CM = 15.0


# ============================================================
# 3D-PROFIILIN RIITTAVYYDEN VALIDOINTI
#
# Kayttajan pyynnosta: yksittainen loydetty/seurattu kivi ei aina
# riita hyvaan profiiliin (esim. kivi nakyy vain lyhyesti alussa) -
# sovitusta (fit_stone_profile) YRITETAAN jokaisen uuden loydetyn
# kiven jalkeen KAIKKIEN TAHAN ASTI kerattyjen havaintojen paalla, ja
# hyvaksytaan vasta kun se on RIITTAVAN hyva nailla kynnysarvoilla -
# muuten jatketaan koko radan skannausta ja kerataan lisaa (mahdollisesti
# eri kivesta/hetkesta), YHDISTAEN havainnot samaan sovitukseen.
# ============================================================

# HUOM (kayttajan huomio: osa heitoista osittain harjan peittamia):
# aidon kiven yksinainen jaannosvirhe voi olla 6-11px kun osa reunasta
# on harjan takana - kun useampi taman tasoinen havainto YHDISTETAAN
# (kayttajan pyynto: vahintaan PROFILE_MIN_ACCEPTED_STONES kivea),
# pooled-sovituksen RMS luonnollisesti kasvaa (yhteinen malli selittaa
# useampaa, kohinaisempaa havaintoa yhtaikaa) - 5.0px oli viritetty
# ajalta jolloin PARI puhdasta havaintoa riitti. 10.0px sallii tuon
# realistisen kohinatason turvautumatta R_max:in fysikaaliseen
# jarkevyystarkistukseen (katso alla) ainoana suojana vaarilta
# kandidaateilta.
PROFILE_MAX_RMS_PX = 10.0


PROFILE_MIN_SAMPLES = 15


PROFILE_SAMPLES_PER_STONE = 25


# Turvaverkko RMS-kynnyksen LISAKSI (ei sen sijaan): kamera9_01.py:n
# fit_stone_profile:in oma dokumentaatio sanoo R_max:in olevan
# fyysisesti n. 14.0-14.6cm (oikean kiven halkaisija ~28cm). Aiemmin
# testatessa loydettiin tapaus jossa vaara kandidaatti (mainosteksti)
# lapaisi RMS-kynnyksen mutta antoi R_max~66cm - n. 4.5x liian ison.
# Reilut rajat (paljon RMS-kynnysta tiukemmat vaatimukset olisivat
# turhia, mutta karsivat selvasti fysiikan vastaiset tulokset).
PROFILE_R_MAX_MIN_CM = 10.0


PROFILE_R_MAX_MAX_CM = 20.0


# Kayttajan huomio (tarkea arkkitehtuurikorjaus): koko radan HIDASTA
# segmentointipohjaista skannausta EI kannata jatkaa keraamaan monta
# kivea - sen ainoa tehtava on saada AIKAISEKSI jokin RIITTAVA
# profiili mahdollisimman NOPEASTI. Sen jalkeen ELAVA SEURANTA (katso
# alempana "ELAVA MONI-KIVEN SEURANTA") jo hakee UUDET kivet paljon
# tehokkaammin JA luotettavammin: mallipohjaisella ristikkohaulla
# VAIN kaukaisen paan kiinnealta vyohykkeelta (SEARCH_Y_MIN/MAX_CM
# = kaukainen hogline...hogline+3m, SEARCH_X_HALF_WIDTH_CM=
# +-70cm keskiviivasta) - ei enaa tarvitse luottaa hitaaseen, harjasta
# helposti hairiintyvaan vapaamuotoiseen liikkeentunnistukseen koko
# radalla.
#
# PROFILE_MIN_ACCEPTED_STONES=2 (kayttajan pyynnosta, nostettu 1:sta):
# profiili sovitetaan YHTEISESTI kaikkien hyvaksyttyjen kivien havain-
# noista (katso fit_stone_profile) - kahdella kivella yhdella sijasta
# profiili ei ylisovitu yhden kiven omiin (esim. kulman/valaistuksen
# aiheuttamiin) satunnaisvirheisiin, joten se yleistyy paremmin MUIHIN
# kiviin joita ELAVA SEURANTA sen jalkeen kayttaa.
PROFILE_MIN_ACCEPTED_STONES = 2


# Testatessa oikealla videolla loytyi KAKSI ongelmaa jotka nama
# kynnysarvot/mekanismit korjaavat:
#
#   1) find_stone_candidates hyvaksyy MYOS ei-kivia (kayttajan
#      dokumentoima aiempi havainto: "pelaajan vaatteet" jne. - katso
#      find_stone_candidates:in kommentti) - LIIKKUVA ei-kivi
#      (esim. pelaaja/harja) voi lapaista liikkeentunnistuksen. Siksi
#      JOKAINEN seurattu kandidaatti tarkistetaan YKSINAAN (SOLO_TRACK_
#      MAX_RMS_PX) ENNEN kuin sen havainnot lisataan pysyvaan kokoelmaan
#      - jos yksinaankin jaannosvirhe on aivan liian suuri (ei nayta
#      kivelta), havainnot HYLATAAN eika saastuteta koko kokoelmaa
#      pysyvasti.
#   2) SAMA fyysinen kohde (oli se sitten kivi tai hylatty ei-kivi)
#      loydettiin uudelleen JOKA skannauksella niin kauan kuin se pysyi
#      liikkeessa (esim. kavelevaa/juokseva pelaaja useiden 2s-ikkunoiden
#      ajan) - HUOM: sijaintipohjainen jaahdytys EI RIITA, koska nopeasti
#      liikkuva kohde (kayttajan testivideolla havaittu, luultavasti
#      pelaaja, n. 200cm/2s) karkaa sijaintikynnyksen ulkopuolelle ennen
#      jaahdytysajan paattymista. Sen sijaan STONE_SCAN_COOLDOWN_FRAMES
#      estaa uuden kalliin taydellisen seurannan koko taman monta framea
#      kattavalla IKKUNALLA viimeisimman yrityksen (kivi tai hylatty)
#      SIEMENFRAMESTA riippumatta kohteen nopeudesta - karkea mutta
#      luotettava nopeusrajoitin kalliille yrityksille.
# HUOM (kayttajan huomio: osa heitoista vaikeampia tunnistaa koko
# matkaa harjan takia): testatessa nakyi selva kahtiajako yksinaisen
# kandidaatin RMS-jakaumassa - selvasti ei-kivet (esim. pelaajat)
# antoivat RMS n. 12-40px, kun taas todennakoisesti AIDOT mutta
# osittain harjan peittamat kivet jaivat n. 6-11px valille (vrt.
# background-suodatuksen jalkeen hyvaksytyt puhtaat havainnot, jotka
# olivat 2-6px). SOLO_TRACK_MAX_RMS_PX:n tehtava on vain karsia
# SELVASTI ei-kivet pois pysyvasta kokoelmasta - lopullinen laatu-
# portti on POOLED-sovituksen oma, tiukempi PROFILE_MAX_RMS_PX
# (katso try_fit_profile) - joten kynnysta voi nostaa tanne asti
# ilman etta koko sovituksen lopullinen tarkkuus karsii.
SOLO_TRACK_MAX_RMS_PX = 12.0


# HUOM (kayttajan huomio: n. 9 heittoa videolla, mutta vain 2
# hyvaksyttiin): 750 framea (30s) osoittautui liian pitkaksi -
# lokista nakyi etta UUSI yritys kaynnistyi lahes AINA TASAN
# jaahdytyksen paatyttya (siis jotain "liikkuvaa" loytyy joka
# ikinen skannaus), joten jaahdytys itse rajoitti karkeasti
# videon_pituus/30s attempts-maaran - jos kaksi oikeaa heittoa
# tapahtuu alle 30s valein, jalkimmainen jai kokonaan loytamatta.
# track_stone_in_video_windowed on jo AIKAIKKUNAAN rajattu (ei
# koko videota) joten yrityksen hinta on paljon pienempi kuin
# alkuperaisessa "kallis" -perustelussa - lyhyempi jaahdytys on
# nyt varaa. Sama oikea kivi loytyneena kahdesti (esim. jos
# jaahdytys paattyy kesken sen oman liu'un) ei ole haitallista
# (background-suodatuksen jalkeen hyvaksytyt havainnot ovat
# aidosti kivia, joten kaksinkertainen havainto vain vahvistaa
# sovitusta, ei saastuta sita - toisin kuin aiempi ongelma
# vaarilla ei-kivi-kandidaateilla).
STONE_SCAN_COOLDOWN_FRAMES = 150   # 6s 25fps:lla


# ============================================================
# ELAVA MONI-KIVEN SEURANTA + CSV
#
# Kayttajan pyynnosta (katso keskusteluhistoria - alunperin 4, nostettu
# 8:aan): riittavasti tilaa samanaikaisesti aktiivisille kiville, myos
# silloin kun jokin (esim. paikallaan pysyva/hoglinen tuntumassa oleva)
# kohde tukkii yhden paikan pitkaksi aikaa - ei enaa estä uusien aitojen
# heittojen rekisteroitymista yhtä helposti. Yksinkertainen lahin-
# ehdokas-per-kivi -logiikka riittaa (kivet lahekkain vasta pysahtymisen
# jalkeen, jolloin ID:lla ei ole enaa merkitysta). Uusien kivien HAKU
# kaytta kamera9_02.py:n kiinteaa paata-rajattua vyohyketta (SEARCH_X/
# Y_*, k92-moduulista) - TOISIN kuin 3D-profiilin koko-radan-skannaus
# (Task 3): kivet HEITETAAN aina samaan suuntaan/paahan, joten kiinteä
# HAKU-vyohyke on jarkeva/tehokas tassa.
# ============================================================
MAX_CONCURRENT_STONES = 8


# ============================================================
# SEURANNAN SIETOKYKY HETKELLISELLE TAYDELLE PEITOLLE (Testi_02_01,
# kayttajan pyynnosta): kamera9_02.py:n oma TRACK_LOST_MAX_MISSES=5
# (0.2s 25fps:lla) on LIIAN LYHYT lahemman pesan luona - taman
# tiedoston OMA, jo aiemmin dokumentoitu havainto (katso COLOR_STOP_
# MAX_AVG_DIFF:in ylapuolinen kommentti) mainitsee mittaavan pelaajan
# peittaneen kiven "30+ ruutua/1.2s+" - SELVASTI pidempaan kuin 5
# ruutua. Koska HAKU (kamera9_02.py:n SEARCH_Y_MIN/MAX_CM) toimii VAIN
# kaukaisen paan vyohykkeella (katso taman tiedoston oma ASETUKSET-
# kommentti ylla: "kivet HEITETAAN aina samaan suuntaan/paahan"), kivi
# joka menettaa SEURANTANSA lahella pesaa EI VOI koskaan loytya
# uudelleen - se katoaa CSV:sta pysyvasti lopun liu'un/pysahtymisen
# ajaksi, vaikka peittava pelaaja/harja siirtyisi pois sekunnin
# sisalla. TRACK_LOST_MAX_MISSES=5 ei siis erota "hetkellinen taysi
# peitto" (pitaisi selvita, kivi on siina missa ennenkin - haku-ankkuri
# s["last_xy"] EI paivity misseilla, katso alempana SEURANTA-silmukka)
# tapauksesta "kivi todella poissa/vaara kandidaatti".
#
# PAIKALLINEN ylikirjoitus (EI muuteta kamera9_02.py:n omaa arvoa -
# sama periaate kuin HAKU_SEARCH_INTERVAL_FRAMES alla): nostetaan
# sallittu peraikkainen peitto TRACK_LOST_GRACE_SECONDS:iin asti,
# reilusti yli tuon dokumentoidun 1.2s+ havainnon ylapuolelle, jotta
# tavanomainen "pelaaja/harja kokonaan kiven edessa hetken" -tilanne ei
# enaa katkaise seurantaa. Kayttajan oma rajaus (aidosti PITKAKESTOINEN
# peitto, esim. koko lopun liu'un ajan joku seisoo kiven paalla, jaa
# vaistamatta havaitsematta) sailyy silti - kivi hylataan (kuten
# ennenkin) jos peitto kestaa PIDEMPAAN kuin tama reilu ikkuna.
# ============================================================
TRACK_LOST_GRACE_SECONDS = 3.0


# ============================================================
# NOPEUSRAJOITETTU SEURANTA-HAKUALUE + "EI TAAKSEPAIN" (kayttajan
# pyynnosta, katso keskusteluhistoria): SEURANTA:n ristikkohaku etsi
# aiemmin AINA kiintealta, isotrooppiselta alueelta (kamera9_02.py:n
# TRACK_HALF_RANGE_CM=35cm joka suuntaan) riippumatta siita kuinka
# kauan kivesta on todellisuudessa aikaa edellisesta havainnosta.
# Koska oikea curling-kivi EI VOI liikkua nopeammin kuin fysikaalinen
# yla­raja (arvioitu turvamarginaalilla: Y-suunnassa, radan pituus-
# suunnassa, TRACK_MAX_SPEED_Y_CM_S - kayttajan helposti muutettavissa
# alla; X-suunnassa, sivuttaisliike/curl, paljon hitaampaa - JOHDETTU
# suoraan Y-nopeudesta, TRACK_MAX_SPEED_X_FRACTION_OF_Y=10%:na siita,
# EI oma erillinen vakionsa), hakualue voidaan laskea
# SUORAAN nopeusrajasta JOKA framella JOKAISELLE kivelle erikseen:
#   half_range = max_speed_cm_s * (misses+1) / fps
# ("misses+1" framea sitten oli viimeisin VAHVISTETTU sijainti, koska
# s["last_xy"] EI paivity misseilla - katso TRACK_LOST_GRACE_SECONDS:in
# kommentti ylla) - haku ei siis KOSKAAN tarvitse etsia kauempaa, ja
# pitkan peiton (misses>0) jalkeen hakualue kasvaa automaattisesti
# oikeassa suhteessa. Katso kaytto SEURANTA-kutsun valmistelussa
# (build_track_half_ranges alempana) ja stone_tracker.cpp:n
# trackStoneUpdateOne (track_half_range_x_cm/track_half_range_y_cm,
# EI enaa yhta isotrooppista kamera9_02.py:n TRACK_HALF_RANGE_CM:ia).
#
# MAX_BACKWARD_CM: kivi liikkuu tassa videossa AINA Y:n pienetessa
# (katso Y-suunnan kommentti main.py:n alkupaassa) - aito liikkuva
# kivi hidastuu mutta EI KOSKAAN peruuta. Jos yhden SEURANTA-paivityksen
# tulos siirtaisi kiven yli metrin "taaksepain" (Y kasvaisi), kyseessa
# on lahes aina ristikkohaun ajautuminen vaaraan kohteeseen (esim.
# viereinen kivi/pelaaja) - hylataan (kuten oversized_reject, katso
# stone_tracker.cpp) sen sijaan etta hyvaksytaan virheellinen hyppy.
# ============================================================
TRACK_MAX_SPEED_Y_CM_S = 300.0  # kayttajan helposti muutettava arvo (3 m/s)


# X-suunnan nopeusraja JOHDETAAN Y-nopeudesta (kayttajan pyynnosta) - EI
# oma erillinen vakionsa - aina TRACK_MAX_SPEED_X_FRACTION_OF_Y verran
# TRACK_MAX_SPEED_Y_CM_S:sta, joten X paivittyy automaattisesti kun
# Y-arvoa muutetaan yllä.
TRACK_MAX_SPEED_X_FRACTION_OF_Y = 0.10


TRACK_MAX_SPEED_X_CM_S = TRACK_MAX_SPEED_Y_CM_S * TRACK_MAX_SPEED_X_FRACTION_OF_Y


TRACK_MAX_BACKWARD_CM = 100.0


# ============================================================
# ABSOLUUTTINEN KESKIVIIVAETAISYYSRAJA (kayttajan pyynnosta, katso
# keskusteluhistoria): kayttajan omaan kokemukseen perustuva havainto
# tasta radasta/pelityylista - kivi joka on koskaan yli 1.2m
# keskiviivasta (X=0) on hyvin todennakoisesti EI kivi (SEURANTA on
# ajautunut pelaajaan/lakaisijaan tms.), koska aidot heitot pysyvat
# tallä radalla kaytannossa aina tata lahempana keskiviivaa - HAKU:n
# oma hakuvyohyke (kamera9_02.py:n SEARCH_X_HALF_WIDTH_CM=70cm) on jo
# tatakin tiukempi UUDEN kiven loytohetkella, mutta SEURANNAN paikal-
# linen haku (TRACK_HALF_RANGE_CM=35cm/frame) voi ajan mittaan ajautua
# hakuvyohykkeen ulkopuolelle jos se tarttuu jatkuvasti sivuttain
# liikkuvaan kohteeseen (esim. pyyhkija) - sama periaate kuin TRACK_
# MAX_BACKWARD_CM:ssa ylla, mutta X-suunnassa ja molempiin suuntiin.
# ============================================================
MAX_ABS_X_FROM_CENTERLINE_CM = 120.0


# YLARAJA nopeuspohjaiselle hakualueelle (havaittu VALTTAMATTOMAKSI
# koko videon lapikaynnilla, katso keskusteluhistoria): half_range =
# max_speed_cm_s * elapsed_s KASVAA RAJATTA pitkien miss-sarjojen
# aikana (TRACK_LOST_GRACE_SECONDS=3.0s asti) - esim. 3s peitolla
# half_range_y = 450*3 = 1350cm, mika teki ristikkohausta (ja sen ROI-
# leikkeesta) valtavan hitaan (mitattu: SEURANTA-kutsu keskimaarin yli
# 1s/kutsu koko videon lapikaynnissa, n. 15-20x hitaampi kuin ennen
# tata ominaisuutta - videon lapikaynti olisi kestanyt yli 2h 5min
# videolle). Katakkaa hakualue tahan, jotta pahin tapaus pysyy
# hallittavana - jos kivi ei loydy edes tallä alueella pitkan peiton
# jalkeen, se joka tapauksessa kadotetaan (TRACK_LOST_MAX_MISSES/
# TRACK_LOST_GRACE_SECONDS) ja loytyy tarvittaessa uudelleen HAKU:n
# kautta, joten rajaus ei heikenna luotettavuutta merkittavasti.
TRACK_HALF_RANGE_MAX_CM = 100.0


# ============================================================
# "UUSI KIVI" -REKISTEROINNIN VAHVISTUS LIIKKEELLA (Testi_02_01,
# havaittu oikean 5min Testivideo-julkaisun analyysissa): kivien
# Y-koordinaatti PIENENEE ajassa taman videon suunnistuksessa (heitto-
# paa/HAKU-vyohyke = suuri Y, lahempi pesa = pieni Y - VAHVISTETTU
# oikealla datalla: aidot, koko matkan onnistuneesti seuratut heitot
# nayttavat sileaa, HIDASTUVAA Y:n pienenemista n. 1.8-1.9 m/s:sta
# muutamaan kymmeneen cm/s:iin - fysikaalisesti oikea kitkahidastuvuus).
#
# HAVAITTU ONGELMA (samalla oikealla datalla, 13 rekisteroitya stone_
# id:ta ~5-6 aidon heiton sijaan): HAKU (kiintea kaukaisen paan
# vyohyke) loytaa saannollisesti UUDELLEEN saman jo LEVOSSA olevan
# (aiemmin jo raportoidun/pysahtyneen TAI peiton takia kadotetun)
# kiven tai muun paikallaan olevan kohteen, koska se poistuu active_
# stones:ista ("pysahtynyt" TAI "kadotettu" jalkeen) ja seuraava HAKU-
# kierros (1s valein) loytaa saman paikallaan olevan kohteen taas -
# rekisteroiden sen VIRHEELLISESTI UUTENA kivena. Esimerkkeja oikealta
# datalta: monta stone_id:ta joiden koko elinika oli alle 5s ja netto-
# siirtyma vain 1-250cm (verrattuna aitojen heittojen n. 2900-3100cm:iin
# koko HAKU-loytohetkesta lahemman pesan levahdyspaikkaan).
#
# Korjaus: koska KAIKKI aidot kivet ovat jo LIIKKEESSA kun HAKU loytaa
# ne ja jatkavat matkaansa satoja senttteja ENNEN pysahtymista, mutta
# paikallaan-jo-olevat kohteet eivat LIIKU merkittavasti: vaaditaan etta
# uusi HAKU-loytö siirtyy vahintaan taman verran (itseisarvo, ensim-
# maisesta havainnosta) ENNEN kuin sen havainnot kirjataan CSV:hen
# aitona heittona - katso pending_rows/confirmed-kasittely alempana.
# Kunnes vahvistus tapahtuu, havainnot puskuroidaan (ei kirjoiteta) -
# jos kivi kadotetaan/pysahtyy KOSKAAN vahvistumatta, koko puskuroitu
# havaintosarja hylataan hiljaisesti (ei CSV-rivia, ei lasketa mukaan
# "n_stones_seen":iin). Aito heitetty kivi ylittaa tamän kynnyksen jo
# ensimmaisen sekunnin sisalla (katso yllapuoliset esimerkit); pai-
# kallaan lepäävä kohde ei koskaan.
MIN_CONFIRMED_THROW_DISPLACEMENT_CM = 25.0


# HAVAITTU ONGELMA (kayttajan raportoima, katso keskusteluhistoria):
# pelaaja/lakaisija HAKU-alueella (katso kamera9_02.py:n SEARCH_Y_MIN/
# MAX_CM) voi silloin talloin TAYTTAA HAKU:n omankin tiukan hyvaksynta-
# kriteerin (search_new_stone vaatii refined.tarkka:n - OIKEAN pyorean
# reunan/rengasrakenteen LM-sovituksen, EI pelkkaa peitto-osuutta, katso
# stone_tracker.cpp:n search_new_stone-kommentti) YHDELLA framella
# esim. vaatteen/harjan reunan sattuessa hetkeksi ympyramaiseksi. Kun
# tama tapahtuu, SEURANTA jatkaa sen jalkeen kandidaattia LOYSEMMALLA
# per-frame-kynnyksella (TRACK_SCORE_THRESHOLD, katso kamera9_02.py -
# EI vaadi tarkka:a joka framella, koska "jatkuvuus on jo vahva prior"
# - jarkeva oletus AIDOLLE, jo vahvistetulle kivelle, joka voi hetkeksi
# olla osittain peitossa). Tama sallii kandidaatin, joka EI OIKEASTI ole
# kivi, "ajautua" pelaajan mukana useita sekunteja ja YLITTAA yllaolevan
# siirtymakynnyksen, tullen vahvistetuksi VIRHEELLISESTI.
#
# Todellisella datalla (katso keskusteluhistoria, fullrun_positions.csv,
# frame 4530 -> 4716, ihmiseksi silmamaarin vahvistettu): tama kandi-
# daatti saavutti tarkka=1:n vain 3/8 (38%) ensimmaisesta 8 havain-
# nostaan (ennen 25cm-vahvistuskynnysta), kun taas kaikki pitkaan
# seuratut, selvasti aidot kivet (>=500 rivia koko elinajalta) olivat
# 100% tarkka=1 samassa ikkunassa. Siksi: vaaditaan MYOS etta riittavan
# pitkan (>=MIN_PRECONFIRM_TARKKA_OBSERVATIONS havaintoa - lyhyemmalla
# otoksella tarkka-osuus on liian kohinainen yksittaisten aitojenkin
# heittojen alkumetreilla hylattavaksi, katso keskusteluhistorian
# tilastot) puskuroidun ikkunan tarkka-osuus ylittaa MIN_PRECONFIRM_
# TARKKA_FRACTION:in ENNEN vahvistusta - muuten koko puskuroitu
# havaintosarja hylataan (kuten "ei liikkunut riittavasti" -tapauksessa)
# eika kandidaattia enaa seurata.
MIN_PRECONFIRM_TARKKA_OBSERVATIONS = 8


MIN_PRECONFIRM_TARKKA_FRACTION = 0.4


# JATKUVA TARKKA-OSUUSTARKISTUS VAHVISTETUILLE KIVILLE (Testi_02_02,
# kayttajan pyynnosta) - katso kayttokohdan kommentti. Eri (loyhempi)
# kynnys kuin MIN_PRECONFIRM_*: vahvistettu kivi on jo läpaissyt
# tiukemman esitarkistuksen kerran, joten tama on lisasuoja PITKAAN
# kestavaa ajautumaa vastaan, ei alkuperainen suodatin.
#
# 50 -> 150 (kayttajan pyynnosta tehdyn SEURANNAN maskiyhdistelma-
# korjauksen - stone_tracker.cpp:n createForegroundFromWhitened -
# jalkeen havaittu tarve, katso myos confirmed_tarkka_window:in oma
# kommentti kayttokohdassa): korjaus tekee findContourNear:ista
# herkemman todelliselle, mutta LYHYTAIKAISELLE kosketukselle naapuri-
# kohteeseen (esim. heittaja/lakaisija VIELA kiven vieressa muutaman
# sekunnin ajan HETI heiton jalkeen, ennen kuin kivi jaa yksin) - tama
# nakyy matalana tarkka-osuutena vain ENSIMMAISTEN havaintojen ajan.
# Ikkuna (deque) sailyttaa vain VIIMEISIMMAT N havaintoa, mutta koska
# ensimmainen tarkistus tapahtuu heti kun ikkuna on TAYTTYNYT ENSI
# KERRAN (havainto N), lyhyt N (50 = 2s) tarkistaa VIELA TASMALLEEN
# saman (kontaminoituneen) jakson kuin kumulatiivinen laskuri olisi -
# ikkuna ei ehdi "unohtaa" mitaan ennen ensimmaista tarkistusta.
# Mitattu oikealla datalla: aito kivi hylattiin virheellisesti N=50:n
# kohdalla (19/50=38%), vaikka sen havainnot valilta t=622.68-623.12
# (juuri se jakso jonka kayttaja alunperin liputti X-hyppimisongelmana)
# olivat jo taydellisia (n_body 26-29, rms 0.5-0.9px) - lyhyt n. 1.5s
# kontaminaatio heti heiton jalkeen ei ollut viela ehtinyt "laimentua"
# lyhyessa ikkunassa, ja N=50 katkaisi seurannan JUURI TAMAN puhtaan
# jakson keskella. N=150 (6s) antaa tallaiselle lyhyelle alkukonta-
# minaatiolle tarpeeksi tilaa laimentua ENNEN ensimmaista tarkistusta -
# uudella koko radan ajolla sama kivi sailyi nyt seurattuna KOKO sen
# kayttajan alunperin liputtaman puhtaan jakson ajan (t=622.68-623.12,
# rms edelleen 0.5-0.9px) ja viela n. 6 sekuntia sen ohi, ennen kuin
# SEURANTA ajautui my­ohemmin (t~624.2+) toiseen, jo ERIKSEEN tunnet-
# tuun ja kayttajan tietoisesti rajaamaan ei-kriittiseen harjaan-
# ajautumis-ongelmaan (katso keskusteluhistoria - tata TOISTA ongelmaa
# EI korjata tassa, kayttajan nimenomaisesta pyynnosta). Aidot ongel-
# malliset kandidaatit (havaittu samasta ajosta: n_body 260-280, rms
# 25-33px koko elinkaarensa ajan yhden ainoan alkuhavainnon jalkeen,
# ei yhtaan pitkaa puhdasta jaksoa) jaavat silti selvasti alle 0.5:n
# kynnyksen N=150:n kohdallakin - vain n. 100 ylimaaraista (vaaraa)
# CSV-riviä ehtii kertya ennen hylkaysta lyhyemman N:n sijaan.
#
# 150 -> 650 (kayttajan huomio: "yli 20 heittoa pitaisi pystya
# seuraamaan", mutta HAKU-korjauksen jalkeenkin vain n. 16 heittoa
# sai "luotettavan" (disp>15m, tarkka>75%) leiman - juurisyy loytyi
# TASTA kynnyksesta): HAKU-korjauksen (createForegroundFromWhitened)
# jalkeen SEURANTA yhdistaa rakeisuus- ja etualamaskit - tama tekee
# findContourNear:ista herkemman havaitsemaan MYOS PITKAKESTOISEN,
# ihan tavallisen curling-tilanteen: lakaisija joka kavelee/lakaisee
# JATKUVASTI aidon, juuri heitetyn kiven VIERELLA/PAALLA KOKO heiton
# ajan (ei vain hetken alussa, kuten N=150:n oma tapaus ylla) - tama
# EI ole sama tapaus kuin "SEURANTA ajautui kokonaan pois oikealta
# kivelta johonkin toiseen kohteeseen" (jota tama tarkistus alunperin
# yritti torjua), koska X/Y-sijainti itsessaan pysyy koko ajan
# sileana ja fysikaalisesti jarkevana (Y vahenee tasaisesti/hidastuen
# kohti pesaa, X pysyy lahella keskiviivaa) - VAIN tarkka/n_body/rms
# nayttavat huonolta, koska loydetty yhtenainen kontuuri sisaltaa
# seka kiven etta lakaisijan. Mitattu oikealla datalla (00014_0202_
# fp5_positions.csv): AINAKIN 14 todennakoisesti aitoa heittoa (esim.
# id=134: X=0.1-0.7m koko ajan, Y vahenee sileasti 33.4m->19.1m,
# n_body vaihtelee 12:sta yli 500:aan mutta sijainti ei koskaan
# hyppaa) katkaistiin TASAN N=150:n (tai N:n lahella) kohdalla, vaikka
# muut, kokonaan puhtaat heitot samalla videolla kestivat luonnol-
# lisesti 472-613 havaintoa (19-24s) ennen omaa "pysahtynyt"-
# paattymistaan - N=150 (6s) ei siis riittanyt EDES YHDEN normaalin
# heiton luonnolliseen kestoon, joten se katkaisi tehokkaasti LAHES
# JOKAISEN lakaisijan saattaman heiton keskelta. N=650 (26s) on
# reilusti pidempi kuin pisin havaittu aito heitto (613 havaintoa) -
# antaa siis KAIKILLE aidoille heitoille (myos
# koko ajan lakaistuille) tilaa kestaa luonnolliseen loppuunsa
# ("pysahtynyt"-tarkistus tai videon loppu) asti, samalla kun aidosti
# ongelmalliset kandidaatit (nolla-lahella tarkka-osuus koko elin-
# kaarensa, ei koskaan puhdasta/liikkuvaa jaksoa) jaavat silti
# selvasti alle 0.5:n kynnyksen - vain hieman my­ohemmin (26s vs 6s)
# kuin ennen.
MIN_CONFIRMED_TARKKA_OBSERVATIONS = 650


MIN_CONFIRMED_TARKKA_FRACTION = 0.5


# HAKU-valin PAIKALLINEN ylikirjoitus (kayttajan pyynnosta) - EI
# muuteta kamera9_02.py:n omaa SEARCH_EVERY_N_FRAMES:ia (se tiedosto
# on koskematon referenssi, katso taman tiedoston alkupaan kommentti).
#
# HUOM (kayttajan uusin pyynto, katso keskusteluhistoria): palautettu
# takaisin kamera9_02.py:n alkuperaiseen, framepohjaiseen tahtiin (10
# framea=0.4s 25fps:lla) - aiempi, tata harvempi sekuntipohjainen vali
# (1.0s, CPU-saastosta) korvattu, koska kavennettu HAKU-vyohyke
# (kamera9_02.py:n SEARCH_Y_MIN/MAX_CM, nyt vain hogline...hogline+3m
# eika enaa asti pesan takareunalle) tekee jokaisesta HAKU-kutsusta jo
# halvemman, joten tihempi tarkistus ei enaa maksa yhta paljon - ja
# uuden kiven ensimmaiset havainnot loytyvat aiempaa nopeammin.
HAKU_SEARCH_INTERVAL_FRAMES = 10


# Jos uusi HAKU-loytö on tata lahempana jotain jo AKTIIVISTA kiveä,
# tulkitaan samaksi kiveksi (ei uutta ID:ta) - estaa saman kiven
# kaksoiskirjautumisen. HUOM: HAKU:n karkea koko-framen ristikkohaku
# (SEARCH_COARSE/FINE_STEP_CM) ja SEURANTA:n oma hienompi haku
# (TRACK_COARSE/FINE_STEP_CM) eivat aina osu tarkalleen samaan
# pisteeseen samalla framella varsinkin nopeasti liikkuvalla kivella -
# testivideolla havaittu poikkeama n. 66 cm yhden ja saman kiven
# kahden eri hakumekanismin valilla, joten kynnys pidetaan reilusti
# sen ylapuolella.
#
# HUOM (kayttajan loytama bugi MAH-videolla, katso keskusteluhistoria):
# tama vertailu koski AIEMMIN KAIKKIA active_stones-listan kiviä, myos
# viela VAHVISTAMATTOMIA (s["confirmed"]==False) ehdokkaita. Jos HAKU
# tunnisti pelaajan/lakaisijan jalan viereltä kiveksi (aito ongelma -
# pelkkaan muotoon/kokoon perustuva sovitus ei tunne varia), tama
# vahvistamaton "kivi" esti TAYSIN HILJAA (ei lokiviestia) sen VIERESSA
# olevan OIKEAN kiven rekisteroitymisen niin kauan kuin vaara ehdokas
# pysyi aktiivisena (jopa useita sekunteja) - koska etaisyys mitattiin
# myos siihen. Korjaus: dedup-vertailu tehdaan nyt vain jo VAHVISTETTUIHIN
# kiviin (s["confirmed"]) - kaksi AITOA, jo vahvistettua kiveä eivat
# edelleenkaan voi saada duplikaatti-ID:ta, mutta viela vahvistamaton
# (mahdollisesti vaara) ehdokas ei enaa voi tukkia vieressa olevan
# oikean kiven havaitsemista.
NEW_STONE_DEDUP_CM = 100.0


# Kayttajan pyynnosta: kun kivi on ollut lahes paikallaan (liikkunut
# alle STOP_TRACKING_DISPLACEMENT_CM) STOP_TRACKING_SECONDS ajan,
# lopetetaan sen aktiivinen SEURANTA - viimeinen sijainti jaa CSV:hen.
# Kaksi hyotya: (1) sailyttaa aktiivisen paikan MAX_CONCURRENT_STONES:sta
# uusille kiville nopeammin, (2) estaa jo pysahtyneen kiven haun
# ajautumisen taustan muuhun sisaltoon pitkalla aikavalilla (havaittu
# testatessa: yksi pitkaan paikallaan seurattu kivi ajautui lopulta
# fyysisesti mahdottomaan sijaintiin, Y=-102cm).
STOP_TRACKING_SECONDS = 1.0


STOP_TRACKING_DISPLACEMENT_CM = 20.0


# VARIREFERENSSI (kayttajan pyynnosta): SEURANTA voi harvoin "hypata"
# pitkaan seuratusta oikeasta kivesta lahella olevaan vieraaseen
# kohteeseen (esim. pelaaja pesan lahella) - naytonmuoto-tarkistus
# (tarkka) menee tassa tapauksessa LAPI, koska vieras kohde sattuu
# olemaan riittavan kiven-muotoinen. Rakennetaan siis KERRAN, heti kun
# 3D-profiili on valmis, kiven OMA pintavarireferenssi hyvaksytyn
# profiilin havainnoista (build_stone_color_reference) - PER PISTE
# local_pts_body:sta (EI keskiarvoistettuna yhdeksi lukemaksi koko
# pinnalta!), jotta graniitin n. keskikorkeudella nakyva vaaleampi
# nauha ("paiva"/leveimmillaan-kohta, _TEMPLATE_EQUATOR_IDX) sailyy
# omana piirteenaan eika sekoitu muun pinnan keskiarvoon. Kayttajan
# huomio: nauha EI nay kaukana olevissa kivissa (resoluutio) - siksi
# seka referenssin rakennus etta elavan osuman vertailu HYLKAAVAT
# (skip, ei estoa) havainnot/osumat joissa kivi on kuvassa liian pieni
# nauhan (tai minkaan hienon piirteen) luotettavaan erottamiseen -
# tarkistus vaikuttaa siis kaytannossa vain lahelta (esim. pesan
# lahella) otettuihin osumiin, mika sopii yhteen sen kanssa etta juuri
# se on havaittu ongelma-alue.
#
# SUUNNITTELUPAATOS (useiden testikierrosten jalkeen - kayttajan
# prioriteetti "heittaja/lakaisija ei saa aiheuttaa kiven katoamista"
# ohjaa tata): varitarkistus EI vaikuta MITENKAAN normaaliin SEURANTAan
# (haku-ankkuri, s["misses"], "kadotettu") - kokeiltiin ensin useita
# hystereesi-/streak-pohjaisia versioita jotka KASVATTIVAT/vahensivat
# "huonoa" laskuria ja kayttivat sita joko hyvaksymiseen TAI kiven
# katoamiseen, mutta havaittiin etta MIKA TAHANSA vuotava/palautuva
# laskuri lopulta "antaa anteeksi" pysyvasti vaaran kohteen (esim.
# pelaajan jalka), koska aidosti kohiseva data tuottaa ENNEMMIN TAI
# MYOHEMMIN riittavan pitkan "hyvan" jakson kumotakseen kertyneen
# laskurin - TAMA PATEE YHTA LAILLA todelliseen kiveen jota peittaa
# pitkaan (havaittu: 30+ ruutua/1.2s+) esim. mittaava pelaaja, jonka
# aikana muototunnistus (refined["found"]) pysyy silti onnistuneena.
# Koska aito peitto ja aito identiteettihyppy nayttavat siis VARILTAAN
# samankaltaisilta pitkalla aikavalilla, MIKAAN kynnys+streak-yhdistelma
# ei erottele niita luotettavasti - vain KESTO eroaa (peitto loppuu,
# hyppy ei koskaan "korjaannu").
#
# Varitarkistus vaikutti aiemmin (Testi_02_01, alkuperainen versio)
# "pysahtynyt"-ilmoitukseen: liukuvan ikkunan (color_diff_history)
# KESKIARVON piti olla aidosti graniittimainen (alle COLOR_STOP_MAX_
# AVG_DIFF) ennen kuin pysahtyminen hyvaksyttiin. TARKOITUS oli etta
# tama EI voisi koskaan aiheuttaa oikean kiven katoamista (ilmoitus
# vain "lykkaantyisi") - mutta koko videon lapikaynti (kayttajan
# pyynnosta, katso keskusteluhistoria) osoitti etta oletus oli VAARIN:
# aidosti paikallaan olevia kivia EI koskaan tulostettu pysahtyneeksi
# (avg_diff pysyi kynnyksen ylapuolella), jolloin ne jaivat pysyvasti
# aktiivisiksi ja tukkivat MAX_CONCURRENT_STONES-paikat - estaen uusien
# aitojen heittojen rekisteroinnin (havaittu konkreettisesti: viimeinen
# heitto jai kokonaan puuttumaan CSV:sta). Varitarkistus on siis
# POISTETTU "pysahtynyt"-paatoksesta - katso SEURANTA-silmukan oma
# kommentti (STOP_TRACKING_DISPLACEMENT_CM/SECONDS RIITTAA yksinaan).
# color_match_median_diff/color_diff_history sailyvat silti (COLOR_DEBUG-
# diagnostiikkaa varten), vain paatoksentekoon ei enaa vaikuta.
COLOR_REF_MIN_RING_SPACING_PX = 2.5


COLOR_REF_MIN_OBSERVATIONS = 3


COLOR_MATCH_MIN_VALID_POINTS = 20


CSV_HEADER = [
    "frame", "timestamp_s", "stone_id", "x_m", "y_m", "tarkka",
    "n_runkopistetta", "n_reunapistetta", "rms_px", "rengas_r_cm",
]


# ============================================================
# PANEELIEN KASIN-MERKINTA (VARAMENETTELY)
#
# Kayttajan pyynnosta: jos automaattitunnistus (referenssin lahelta)
# ei loyda vahintaan MIN_AUTO_PANELS_BEFORE_MANUAL (6) paneelia, se ei
# riita luotettavaan stabilointiin koko videon ajaksi - avataan siis
# skaalattavan kokoinen (cv2.WINDOW_NORMAL, kayttaja voi venyttaa
# ikkunan haluamaansa kokoon) ikkuna jossa kayttaja klikkaa PUUTTUVAT
# paneelit hiirella. Jo automaattisesti loydetyt paneelit nakyvat
# valmiiksi merkittyina (vihrea nelikulmio + jarjestysnumero), joten
# kayttajan tarvitsee klikata vain loput. Jokainen klikkaus ajaa
# saman detect_panel-tunnistuksen (tarkka kulma-/subpikselisovitus)
# klikkauskohdan ymparilta - sama menetelma kuin automaattitunnistus,
# vain hakupisteen ALKUPERA on kasin annettu. Enter lopettaa (vaatii
# vahintaan 2 paneelia yhteensa), Esc peruuttaa kokonaan.
# ============================================================
MIN_AUTO_PANELS_BEFORE_MANUAL = 6


# ============================================================
# AIKAIKKUNAAN RAJATTU KIVEN SEURANTA (EI koko videota)
#
# track_stone_in_video_fast (kamera9_04.py, EI kosketa) skannaa
# TARKOITUKSELLA koko videon jokaisen framen - jarkevaa SEN omassa
# kayttotarkoituksessaan (kertaluontoinen kalibrointi lyhyella
# videolla). Tassa TIEDOSSA jo ON karkea siemen (koko radan skannaus
# loysi sen), joten koko videon uudelleenskannaus JOKAISELLA
# yrityksella (havaittu testatessa: n. 700s per yritys 7500 framen
# videolla) olisi kohtuuttoman hidas, varsinkin jos useampi kandidaatti
# (esim. pelaaja) hylataan ennen aidon kiven loytymista. Tama funktio
# rajaa saman algoritmin (sama jatkuvuuslogiikka, samat kynnysarvot)
# STONE_TRACK_WINDOW_SECONDS-ikkunaan siemenen ymparilta - riittava
# kattamaan koko liu'un (kivi ei ole jaalla montaa kymmenta sekuntia
# kerrallaan) mutta paljon halvempi kuin koko video.
# ============================================================
STONE_TRACK_WINDOW_SECONDS = 30.0


# KAYTTAJAN MITTAAMA LOYDOS (oikea video, todellinen skannaus alusta
# asti): track_stone_in_video_windowed oli 73.7% koko kalibrointi+
# skannausvaiheen ajasta, JA 5/6 kutsusta (83%) kohdistui HYLATTYYN
# ehdokkaaseen (pelaaja/kohina) - jokainen hylkays maksoi silti taysin
# saman KOKO 60s ikkunan (1501 framea) hinnan kuin hyvaksytty kivikin,
# koska solo_ok-kelvollisuustarkistus tehtiin vasta KOKO ikkunan
# skannauksen JALKEEN. Kaksi toisiaan taydentavaa optimointia (EI
# Python-porttaus/kamera9_04.py-muutos - vain tama tiedosto):
#
# 1) HALPA ESITARKISTUS ENNEN taytta ikkunaa: skannataan ensin vain
#    +-STONE_TRACK_PRECHECK_WINDOW_SECONDS (paljon lyhyempi), ja jos
#    tama nayttaa jo selvasti EI-kivelta, hylataan HETI ilman koko
#    60s ikkunan skannausta. Kayttaa LOYHEMPAA RMS-kynnysta kuin
#    lopullinen solo_ok-tarkistus (STONE_TRACK_PRECHECK_MAX_RMS_PX),
#    koska lyhyt ikkuna antaa vahemman kulmavaihtelua (katso taman
#    tiedoston toisen kommentin selitys profiilisovituksen kulma-
#    vaihtelutarpeesta) - tama tekisi RMS:sta systemaattisesti
#    huonomman myos AIDOLLE kivelle, joten liian tiukka kynnys
#    hylkaisi vaarin. R_max-jarkevyystarkistus (try_fit_profile:in
#    SISALLA, PROFILE_R_MAX_MIN/MAX_CM) pysyy silti samana - se on
#    fyysinen koko-tarkistus joka ei riipu ikkunan pituudesta, ja
#    havaitussa datassa (R_max=21-247cm hylatyilla vs. 12.28cm
#    hyvaksytylla) juuri TAMA erotti selvimmin.
#    Jos esitarkistuksen data on liian niukka paatokseen (alle
#    STONE_TRACK_PRECHECK_MIN_SAMPLES havaintoa), EI hylata datan
#    puutteen takia - jatketaan varovaisuudesta taydelliseen ikkunaan.
#
# 2) NAYTTEISTYS: taydessa (ja esitarkistus-) ikkunassa kasitellaan
#    VAIN joka STONE_TRACK_SAMPLE_STRIDE:s frame taysin (vaanto+tausta
#    vaimennus+kandidaattitunnistus - kalliit vaiheet), ei jokaista -
#    kandidaattitunnistukseen kaytetyt havainnot (PROFILE_SAMPLES_
#    PER_STONE=25) ovat joka tapauksessa paljon harvempia kuin taysi
#    1501 framen tiheys tuottaisi, joten tiheampi kuin naytteistetty
#    seuranta ei lisaa profiilin laatua. max_jump_cm skaalataan
#    STRIDE:lla (perustason kynnys on mitoitettu PERAKKAISILLE
#    framille) - muuten nopeasti liikkuva kivi voisi karata seurannasta
#    naytteiden valilla.
STONE_TRACK_PRECHECK_WINDOW_SECONDS = 3.0


STONE_TRACK_PRECHECK_MIN_SAMPLES = 5


STONE_TRACK_PRECHECK_MAX_RMS_PX = 24.0


# HUOM (kayttajan mittaama loydos + paatos): stride=5 JA stride=3
# heikensivat molemmat havaittavasti profiilisovituksen laatua (RMS
# n. 5.9px -> 10.9px stride=5:lla samalle oikealle kivelle taydessa
# 30s ikkunassa - riittavan lahella PROFILE_MAX_RMS_PX=10.0px
# -kynnysta etta yksi muuten kelvollinen kivi ei enaa riittanyt
# sellaisenaan). Kayttaja paatti sen sijaan pitaa TAYDEN
# framekohtaisen tarkkuuden (stride=1, degeneroituu tayteen
# tiheyteen build_indices:issa) ja nojata nopeuteen sen sijaan
# halpaan esitarkistukseen + C++-porttaukseen (katso alempana
# scan_stone_candidates/stone_tracker.cpp).
STONE_TRACK_SAMPLE_STRIDE = 1
