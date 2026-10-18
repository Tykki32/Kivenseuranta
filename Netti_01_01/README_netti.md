# Netti_01_01 – netistä tallennettu video

Pohja: Testi_09_03 (koodi sellaisenaan).

Video: GitHub-release `Nettivideo`, tiedosto `Recording.2026-10-14.115218.mp4`
(1920x1080, 30 r/s, 131 s, 146 Mt; liian iso repoon, ladataan release-sivulta).

Havainnot videosta:
* Ruutu on jaettu kahtia: vasen ja oikea puolisko ovat omat kameransa, kummassakin koko talo (rinki) näkyy ylhäältä.
* Kamerat ovat katossa ja kuva on vakaa, joten alkustabilointi ei ole tarpeen.
* Päällä on TV-grafiikkaa (sponsoribanneri ylhäällä, tulostaulu oikealla, minikartat ylänurkissa), joka pitää maskata pois.
* Radan keskikohta puuttuu, joten kiveä ei seurata koko matkaa; kummankin suunnan heitot näkyvät talon lähellä.

## Muutokset Testi_09_03:een nähden (tähän mennessä)

* `main.py --ei-paneeleja`: ei paneelien valintaa eikä paneelistabilointia moodikuvan keruussa (kamera katossa, kuva
  oletetaan riittävän vakaaksi). Moodikuvan jälkeen stabilointi toimii kuten ennenkin (vaihekorrelaatio moodikuvaan).
* `kalibrointi.py`: punainen pesä haetaan vain sinisen ulkorenkaan sisältä (puulattia/punaiset kivet eivät enää voi
  voittaa "suurin punainen alue" -valinnassa).
* `kalibrointi_perus.py` / `kalibrointi.py`: himmeiden viivojen varahaku (`detect_line_segments(..., faint=True)`)
  jos T-/keskiviivaa ei löydy normaalilla Canny-haulla (nettivideon pakattu kuva: viiva vain ~20 harmaasävyä tummempi).

## Esikäsittely nettivideolle (ffmpeg)

Vasen puolisko, TV-grafiikka (minikartta ylhäällä vasemmalla, banneri ylhäällä oikealla) rajattu pois ja kuva käännetty
niin, että rata kulkee vaakasuunnassa kuten alkuperäisessä kuvassa:
`ffmpeg -i Recording.mp4 -vf "crop=960:800:0:280,transpose=2" -an -c:v libx264 -crf 14 netti_v.mp4`

Tulos: moodikuva (24 näytettä) on puhdas, ihmiset ja kivet häviävät; sininen/punainen pesä, T-viiva ja keskiviiva
löytyvät. Kalibrointi pysähtyy vaiheeseen "kaukaista pesaa ei löytynyt": nykyinen kalibrointi olettaa, että kuvassa
näkyy sekä lähi- että kaukopesä ja molemmat hoglinjat. Tässä kuvassa näkyy vain yksi pesä + sen hogline.

## Yhden pesän kalibrointi (`kalibrointi_netti.py`)

`python kalibrointi_netti.py Recording.2026-10-14.115218.mp4 --ulos tulokset` kalibroi kummankin puoliskon (vasen, oikea):
moodikuva -> renkaat + T-/keski-/hogline -> homografia + objektiivin k1 -> top-down (origo = pesän keskipiste, Y kohti
hoglinea, 2 px/cm). Tulokset kansiossa `tulokset/` (`topdown_*.png`, `topdown_*_tarkistus.jpg`, `kalibrointi_*.json`).

Kaksivaiheinen sovitus:
1. Ellipsisovitus maskeista + viivat -> hyvä alkuarvo (RMS ~2 cm).
2. Reunasovitus: renkaiden neljä reunaa (sin_ulko 182,9 / sin_sisa 121,9 / pun_ulko 61,0 / pun_sisa 15,2 cm) haetaan
   säteittäisistä profiileista alipikselitarkkuudella (puolitasokohta), hoglinen molemmat reunat Y-profiileista, keski- ja
   T-viiva harjahaulla. Parametrit: homografia, k1 (tulee ~0), tumman alueen kutistuma g (px; sumea/teroitettu kuva: puoli-
   tasokohta on ~0,4-0,6 px tumman alueen sisäpuolella, cm-virhe on suurempi radan suunnassa) ja erikseen hoglinen g (~0).
3. **Hoglinen virallinen paikka on sen sisäreuna** (pesan puoleinen reuna): Y = 640 cm, ulkoreuna 650 cm (leveys 10 cm).

Residuaalit (cm RMS): vasen: renkaat 0,2-0,4, hoglinen reunat 0,3, keskiviiva 0,4, T-viiva 1,0 (vain 10 pistettä);
oikea: renkaat 0,25-0,45, hogline 0,3, keskiviiva 1,0, T-viiva 1,4.

### Kameran asento (kalibrointi_netti.py)

Kummankin puoliskon kuva on kameran 16:9-kuvan keskileikkaus (960x1080), joten optinen keskipiste on puoliskon keskellä (480, 540);
rajauksen (y0 = 190) ja 90 asteen kierron jälkeen cx = 540 - 190 = 350 px ja cy ~ 480 px.

**Korjaus (aiempi asento oli virheellinen).** Aiempi laskenta ratkaisi polttovälin f vain yhdestä ehdosta (r1 ja r2 kohtisuorassa) ja
ortonormeerasi R:n SVD:llä; tulos (f = 2293 px, kamera (4, 1380, 572) cm) EI toistanut maatason homografiaa, joten kaikki maatason
ulkopuoliset suureet (kiven korkeus, sivusuuntainen mittakaava korkeudella, kahvan korkeus, Y-asemat z > 0 ja nopeudet) olivat vinossa:
kiven säde näytti 15,0-15,7 cm ja nopeudet olivat 11-20 % liian pieniä (keskinopeus Y = 300-500 cm: 1,70 / 1,95 / 2,33 -> 1,94 / 2,17 / 2,81 m/s). Nyt (f, cy) ratkaistaan kahdesta ehdosta yhtä aikaa
(r1 . r2 = 0 ja |r1| = |r2|, cx = 350 kiinteä), R = [r1 r2 r1 x r2] ilman SVD:tä:
* vasen f = 3129 px, cy = 471,5 (odotettu ~480: cy EI ole annettu vaan tulee ehdoista), kamera (X, Y, Z) = (4, 1718, 524) cm;
  oikea f = 3139 px, cy = 484,5, kamera (-2, 1724, 527) cm;
* kamerat ovat siis radan KESKELLÄ (talon keskipisteestä 17,4 m, Y = 1737) ~5,2 m korkeudessa ja katsovat radan päihin loivasti
  (17-26 astetta vaakatasosta kiven kohdalla) -> kiven sivu näkyy lähes sivusta, yläpinta vain ohuena ellipsinä;
* maatason toisto 0,000 px; sivusuuntainen mittakaava Y = 464 cm kohdalla 2,31 px/cm (vanha 2,15), radan suuntaan 0,895 px/cm
  (vanha 1,15), korkeussuunnassa -2,14 px/cm (vanha -1,85).
Vanha asento on tallessa kalibrointitiedostoissa avaimella `vanha_asento`. Seuraukset ovat alla: kiven koko, kahvan muovi (nyt pyöreä
levy, ei soikio) ja nopeudet. Puolisko rajataan y = 190...1080 (hack ja heittäjä mukaan); TV-grafiikka jää rajauksen ulkopuolelle
paitsi minikartan alareuna.

### Heiluuko kamera? Kuvan siirtymä koko videolla (`tools/kameran_heilunta.py`)

Epäilys: kamera heiluu välillä, jolloin kalibrointi (kiinteä homografia) antaisi sen hetken kuville vinon paikan. Mittaus koko videolla
(`netti_yt.mp4`, 7625 s, kumpikin puolisko erikseen, 10 näytettä / s = 76 251 ruutua): puoliskon mediaanikuva (96 ruutua tasaisesti videon yli; ihmiset ja kivet
poistuvat mediaanista) on referenssi, ja jokaisesta ruudusta mitataan 11-12 staattisen piirteen (talon renkaat ja tekstit, LeaseDeal-teksti, alareunan laita; 192 x 192 px)
siirtymä referenssiin vaihekorrelaatiolla (`cv2.phaseCorrelate`, Hann-ikkuna, alipikselitarkkuus). Puoliskon siirtymä (dx, dy) on ikkunoiden mediaani, joten ikkunaan osuva ihminen
tai kivi ei vaikuta. Työkalun tarkkuus tarkistettu tunnetuilla siirroilla (`--testi`: 0,3 ... 8 px, suurin virhe 0,12 px, etumerkki oikein).

**Tulos: kamera ei heilu.** Vaste (vaihekorrelaation huippu, 1 = täysin sama kuva) on mediaanilta 1,00 / 0,96 eli ruudut täsmäävät referenssiin koko videon ajan:

| | vasen | oikea |
|---|---|---|
| poikkeama referenssistä: tyypillinen (p50) / 99 % / 99,9 % / suurin | 0,11 / 0,36 / 0,48 / **0,61 px** | 0,10 / 0,35 / 0,45 / **0,65 px** |
| ruutuja, joissa poikkeama > 1 px | **0** | **0** |
| ruudun x- ja y-suunnan keskihajonta | 0,05 / 0,14 px | 0,05 / 0,12 px |
| hidas ajelehtiminen (minuuttimediaanit, 2 h) | alle 0,25 px (dy noin +0,12 -> -0,18 px) | alle 0,25 px |

Kuva pysyy siis koko kahden tunnin ajan 0,1 px:n tarkkuudella paikallaan, eikä siirtymiä (askelta, tärähdystä, hidasta kulkua) ole yli 1 px. 30 r/s:n tarkistus pahimmalla jaksolla (6930-6950 s)
antaa saman: suurin poikkeama 0,5-0,6 px; mukana on pieni y-suuntainen värinä (±0,3 px, 1-6 Hz), jonka syy on joko kameran hienoinen värähtely tai videon pakkaus. Kamera on 5,2 m korkeudessa radan keskellä, joten mittakaava on hackin
kohdalla 0,67 cm/px (poikittain) ja 2,7 cm/px (radan suuntaan), pesän kohdalla 0,57 / 1,96 cm/px ja hog-viivalla 0,38 / 0,85 cm/px: suurinkin mitattu siirtymä (0,65 px) on enintään noin 1,8 cm
(radan suunnassa hackin kohdalla), tyypillinen 0,1 px alle 0,3 cm, eli selvästi pienempi kuin kuvakohtainen mittakohina (0,1-0,3 cm) ja radan sovituksen virhe. Siksi **ruutuja ei vakauteta**
(ei korjausta tarvita). Mittauksen voi toistaa millä tahansa videolla: `python tools/kameran_heilunta.py video.mp4 --ulos heilunta.npz` (koko 2 h video noin 12 min kolmella prosessilla;
`--alku/--loppu` ja `--vali 1` tarkistavat yksittäisen kohdan joka ruudulta) ja päätelmä (`VAKAA` / `HEILUU`) tulostuu puolisko kerrallaan; jos jokin toinen video heiluisi, siirtymä (dx, dy) pitäisi
vähentää kuvapisteistä ennen `Puoli.kuva_maahan` ja lisätä `Puoli.maa_kuvaan`:n tulokseen.

Samalla menetelmällä (kertamittaus, ei työkalussa) mitattiin myös ylänurkkien minikarttojen eli talon päältä kuvaavien TV-kameroiden vakaus (270 x 270 px, 4 ikkunaa / kuva, 128 px): poikkeama 99,9 %:ssa ruuduista alle 0,19 / 0,25 px, suurin 0,34 / 0,42 px, yli 0,5 px ei yhtään; nekin ovat siis täysin vakaat (minikarttoja ei käytetä seurannassa, ne maskataan pois).

Huom. `cv2.phaseCorrelate` kertoo ikkunafunktion syöttökuviin PAIKALLAAN (kun ikkunan koko on DFT:lle optimaalinen, esim. 192): ikkunat on annettava kopioina, muuten referenssikuva
muuttuu joka kutsulla (vaste putoaa nopeasti ja mittaus näyttäisi ruudun "liikkuvan" jopa 4 px/40 s, vaikka kuva on paikallaan).

## Kiven tunnistus ja mittaus (`kivi_netti.py`, `aja_kivi.py`, `piirra_radat.py`)

`python aja_kivi.py Recording.mp4 --puoli vasen --ulos tulokset --ikkunat 8-22,56-66,104-114` ->
`tulokset/kivet_vasen.csv` (kaikki havainnot), `tulokset/radat_vasen.csv` (radat: raaka + tasoitettu paikka, nopeus);
`python piirra_radat.py tulokset vasen` -> `tulokset/radat_vasen.png`. Oikealla puoliskolla ei heitetä (tässä videossa).

* **Videon aikaperusta:** nauhoite on 25 r/s -virrasta 30 r/s:ksi, joten joka 6. ruutu on edellisen kopio (|ero|:n 99,9. persentiili < 15, oikeilla ruuduilla
  staattisessakin kuvassa >= 30). Kopiot ohitetaan; `t_s` = ikkunan alku + yksilöllisen ruudun järjestysnumero / 25 on kuitenkin vain RUUTUAVAIN (kaikki muut tiedostot
  hakevat ruudun sillä), EI todellinen aika: lähdevirrasta puuttuu ruutuja. Todellinen aika `t_kor_s` ja ruutunumero `ruutu` ovat `radat_*.csv`:ssä, ks. "Videon todellinen aikaperusta".
* **Esikäsittely:** puolisko rajataan ja käännetään kuten kalibroinnissa, k1 korjataan, tausta = moodikuva. Siemenet:
  graniittimaski (taustasta poikkeava, V 75-175, S < 70) ja kahvamaski.
* **Taustan vaikutus (korjattu):** graniitin peittävyys `alfa_kartta` lasketaan värien sekoitusmallista pikseli = a * graniitti
  + (1 - a) * tausta, a = (p - b).(g - b) / |g - b|^2 (g = graniitin BGR (130, 126, 136), b = moodikuvan pikseli). Aiempi
  V-suhde jakoi tumman taustan (punainen rengas) pienellä nimittäjällä, jolloin kiven varjo punaisella renkaalla näytti
  graniitilta ja kiven reunapikselit sinisellä/punaisella renkaalla katosivat; nyt reuna on sama jäällä ja renkailla. Jään
  varjo (a < 0,3) nollataan (0,5-taso säilyy).
* **Siemenet ja tarkka sovitus:** tunnistus (`kivi_netti.tunnista`) käyttää karkeaa siluettimallia (`kivi_netti.Kivi`, R 13,7 / H 16 cm,
  pohjapiirros kivimalli.py) vain siemeniin ja alkuarvauksiin. Lopullinen paikka (X, Y) sovitetaan KIINTEÄLLÄ tarkalla runkomallilla
  (`kivi_tarkka.py`, ks. "Kiven tarkka runkomalli ja koon pysyvyys" alla): koko ei vapaudu ruudulta toiselle. Yläpinta (kiiltävä, ei graniittia) ja
  kahva ovat "ei väliä" -aluetta, ja ruudun kelvolliset pikselit (ei muovia, ihoa, hihoja, kiiltoreikiä) merkitään `kelvolliset`-sarakkeeseen.
* **Ketjutus ja aukkojen täyttö:** vakionopeusennuste (portti 18 cm + 8 cm/puuttuva ruutu, aukko enintään 14 ruutua); jäljelle
  jääneet aukot (kasi/vartalo kiven päällä, tunnistus ei löydä) täytetään sovittamalla siluetti interpoloidusta paikasta.
* **Tasoitus:** paikallinen 2. asteen sovitus +-6 ruutua, painot 1/(err + 0,1)^2.
* **Tulos (vasen puolisko, korjattu kamera + tarkka runkomalli):** kolme heittoa seurataan hackista (Y ~ -300 cm) kuvan reunaan (Y ~ 750 cm, hogline
  Y = 640): t = 8,3-17,8 s (229 pistettä), 59,0-63,9 s (105), 106,0-110,9 s (111). Keskinopeus Y = 300...500 cm: 1,94 / 2,17 / 2,76 m/s k/25-aikaperustalla (vanhalla,
  virheellisellä asennolla 1,70 / 1,95 / 2,33 m/s) ja 1,86 / 2,09 / 2,49 m/s todellisella aikaperustalla ("Videon todellinen aikaperusta", nykyiset `radat_*.csv`-arvot). Heitto 1 kulkee 4,3 astetta oikealle (X: 17 -> 65 cm T-viivasta hoglinelle), heitot 2 ja 3 lähes suoraan
  (+0,6 / -0,6 astetta). Hogline-ylityksen jälkeen kivi on kuvan reunassa (Y > 640: kiven oikea osa leikkautuu pois, malli sovitetaan näkyvään osaan).
  Ketjutus ei enää yhdistä eri ikkunoiden (8-22, 56-66, 104-114 s) ruutuja toisiinsa, ja radan on edettävä vähintään 500 cm.
* **Aiemmin tunnettu puute (korjattu):** nopeuskäyrässä oli ~±0,2-0,3 m/s aaltoilua (jakso ~1 s, kaikissa heitoissa, myös vapaasti liukuvalla kivellä), jonka tulkittiin
  johtuvan videon ruutuajoituksesta eikä korjattavan. Syy oli kuitenkin ruutuaikaperustassa k/25: lähdevirrasta puuttuu ruutuja, ja ajan voi lukea kiven omasta siirtymästä
  (ks. seuraava luku). Korjauksen jälkeen vapaan lennon nopeuden hajonta on 0,03-0,06 m/s (ennen 0,12-0,21).
* **Ohjatun vaiheen häiriöt:** kun heittäjän vartalo on kiven vieressä (Y ~ -100...200 cm), siluettiseurannan Y-paikassa on yksittäisiä 3-10 cm:n poikkeamia ja radassa
  aukkoja (jopa ~10 ruutua); `err`-sarake ei paljasta niitä. Liu'un/irroituksen tunnistus käyttää siksi robustia (Huber) sovitusta.

### Videon todellinen aikaperusta (`aikaperusta.py`, `aja_kivi.py`)

Aiempi oletus "aika = yksilöllisen ruudun järjestysnumero / 25" oli väärä. Nauhoitteessa on säännöllisen 25 -> 30 r/s -muunnoksen kopioruutujen (joka 6.) lisäksi ylimääräisiä
kopioruutuja: lähdevirrasta puuttuu tai viivästyy ruutuja, jolloin 30 r/s -videossa näkyy ylimääräinen kopio ja seuraavassa ruudussa kivi on siirtynyt KAKSI ruutuväliä
(8 cm nopeudella 2 m/s). Yksilöllisiä ruutuja on kiven liikkuessa 22,8-23,8 / s eikä 25 / s, joten k/25 jää jälkeen ja nopeus ja kiihtyvyys heilahtelevat. Kumpikaan yksinkertainen aikaperusta ei sovi: ruutunumero / 30 antaa 25 -> 30 r/s -muunnoksen 6 ruudun sahalaidan (0...27 ms, vapaan lennon Y(t)-sovituksen jäännöksessä +-2-3 cm) ja k/25 jää jälkeen aina,
kun lähdevirrasta puuttuu ruutu (askel 40 ms).

**Menetelmä** (`aikaperusta.Aikajana`): jokaiselle radan näytteelle määritetään LÄHDEVIRRAN RUUTUINDEKSI j (kokonaisluku) kiven siirtymästä. Kahden peräkkäisen näytteen Y-ero jaetaan
paikallisella nopeudella (nopeus 30 r/s -kellosta, tarkkuus ~2 %, riittää kokonaisluvun arviointiin): nopeudella 2 m/s yksi ruutuväli on 8 cm ja Y-kohina 0,5-0,9 cm, joten ruutujen
määrä (1, 2, 3, ...) on yksiselitteinen. Hitaassa vaiheessa (alle 3 cm/ruutu eli alle 0,75 m/s: lepo ja työnnön alku) j luetaan 30 r/s -ruutunumeroiden hilasta ensimmäisestä nopeasta
näytteestä taaksepäin (+-1 ruutu). Jos kaksi peräkkäistä näytettä on samassa paikassa (|dY| < 1 cm) vaikka kivi liikkuu, myöhempi on tunnistamatta jäänyt kopioruutu ja se pudotetaan
(3 / 1 / 0 näytettä). Aika `t_kor_s` = j / 25 + vakio (vakio: sama kuin videon aika ensimmäisessä nopeassa näytteessä). Sama aikajana (ruutunumero -> t) kelpaa kahvaradalle.
Koska j on kokonaisluku, korjaus ei voi "sovittaa kohinaa": alle ruutuvälin aikavirhe jää näkyviin.

**Tulos** (vapaa lento Y > 480 cm, Y(t):n 2. asteen sovituksen jäännöksen RMS): 3,3 / 2,1 / 2,5 cm (k/25) -> 0,55 / 0,67 / 0,78 cm; vapaan lennon nopeuden hajonta (Y 550...730) 0,21 / 0,12 / 0,14 -> 0,03 / 0,06 / 0,06 m/s;
keskinopeus Y = 300...500: 1,94 / 2,17 / 2,76 -> 1,86 / 2,09 / 2,49 m/s; vapaan lennon hidastuvuus -0,06 ... -0,10 m/s2 (uskottava). Videon kello jää lähdevirran kellosta 0,10 / 0,15 / 0,10 s heiton loppuun mennessä
(1,7 / 2,9 / 2,2 % ensimmäisestä nopeasta näytteestä lukien). `radat_*.csv` ja `kahvaradat_*.csv` saivat sarakkeet `ruutu` (30 r/s -ruutunumero) ja `t_kor_s` (todellinen aika); tasoitus ja nopeudet lasketaan `t_kor_s`:stä
(`piirra_radat.py` myös). `t_s` säilyy ruutuavaimena. Päivitys: `python aja_kivi.py VIDEO --puoli vasen --ulos tulokset --ketjuta-vain`; tarkistus: `python aikaperusta.py tulokset vasen VIDEO`.

### Väriseuranta (kahva) ja vertailu siluettiseurantaan (`aja_kivi.py`, `vertaa_seurannat.py`)

Kivi on nyt niin isolla resoluutiolla, että kahvan väri riittää yksin seurantaan (ei siluettimallia, ei taustan vaikutusta):
kahvan maski (keltainen: H 19-31, S >= 45, V >= 165; punainen: H <= 8 tai >= 170, S >= 100, V 55-168, kahva V ~75-156 on tummempi
kuin punainen rengas V ~180) -> lähimmän komponentin keskipiste (alipikseli) -> maa-asema z = 15 cm korkeudella. Ennuste =
edellinen keskipiste + edellinen siirtymä; väri valitaan automaattisesti (20 ruudun koe). Tulos `tulokset/kahvaradat_vasen.csv`
(raaka ja tasoitettu paikka, nopeus), kuva `tulokset/seurantojen_vertailu_vasen.png`.

* Kaikki kolme heittoa (keltainen, punainen, keltainen kahva) seurataan pelkällä värillä hackista hoglinelle (Y ≈ -230 -> 640 cm), 137/118/101 pistettä.
* Vapaana liikkuvan kiven (Y > 250 cm) sivusuuntainen raaka hajonta on 0,2-0,3 cm RMS (keltainen), eikä renkaiden, T-viivan tai hoglinen
  kohdalla ole hyppyä: tausta ei vaikuta.
* Y > 250 cm: väri- ja siluettiseuranta eroavat vakiolla ~2 cm (riippuu oletetusta kahvan korkeudesta) ja vaihtelevat alle 1 cm.
* Y < 250 cm (heittäjän käsi kahvan päällä): värimaskin sivusuuntainen hajonta on selvästi isompi (sormet peittävät kahvaa) -> hyödyllinen
  vihje irroitukseen.
* **Nopeuden aaltoilu ±0,2 m/s (jakso 0,9-1,0 s) oli samanlainen molemmilla menetelmillä ja jatkui vapaana liikkuvalla kivellä, joten se ei johtunut seurannasta, taustaviivoista eikä kahvasta.**
  Syy löytyi videon aikaperustasta: lähdevirrasta puuttuvia ruutuja ei ollut huomioitu (k/25), ks. "Videon todellinen aikaperusta". Korjatulla ajalla `t_kor_s` aaltoilu poistuu (0,03-0,06 m/s).

### Kiven mitat, kahvan muovin ellipsi ja korkeus (`mittaa_kahva.py`, `kivi_netti.py`: `kahva_silhuetti`, `sovita_kuvaellipsi`, `kahva_keskipiste_3d`)

> **HUOM (korjaus):** tämän osan luvut (kahvan korkeus z = 16,9-17,1 cm, soikio 10,5 x 7,3 cm, kiven R 14,5 / z = 3 cm oletukset) mitattiin vanhalla, virheellisellä
> kameran asennolla. Korjatulla asennolla kahvan muovi on PYÖREÄ levy (A = 10,1, B = 9,9 cm) tasolla z0 ~ 15 cm; ks. "Kiinteän kokoinen kahvamalli" ja "Kiven tarkka runkomalli" alla.

**Kiven mitat.** WCF-säännöt (varmat): ympärysmitta enintään 914 mm (halkaisija enintään 291 mm, R <= 14,55 cm), korkeus vähintään
114 mm (graniittirunko ilman kahvaa), paino kahvoineen 17,24-19,96 kg. Muut mitat vaihtelevat valmistajan mukaan (tyypillisiä, ei
säädeltyjä): runko 28-29 cm x 11,4-12,5 cm, kokonaiskorkeus kahvan kanssa ~17-18 cm, juoksupinta Ø ~12,5 cm, leveimmän kohdan
(iskukaista) korkeus ~2-4 cm jään yläpuolella. Mallissa: iskukaista R = 14,55 cm, z = 3 cm (oletus).

**Idea (käyttäjän):** kiven akselin keskikohta jäällä ja kahvan muovin juuren keskikohta korkeudella h ovat samalla pystysuoralla.

**Menetelmä:**
1. *Muovin silhuetti:* Lab-väri (keltainen: b* >= puolet tasanteesta, a* <= 10; punainen: a*, tumma, ja poikkeaa moodikuvasta dE >= 12, jolloin jään
   vaaleanpunaiset logot jäävät pois), raja alipikselinä 8x ylinäytteistetystä kuvasta (puolitaso). Iho (a* > 10) ei ole muovia.
2. *Ellipsi kuvatasossa* (keskipiste, puoliakselit a (sivu) ja b (radan suunta), kierto): etureuna (leveimmästä kohdasta kohti kameraa)
   tarkasti; kärjet (silhuetin ylin ja alin kohta) enintään 2 mm (0,35 px) ellipsin sisäpuolella; takaosa muovin silhuetin SISÄLLÄ
   (yksipuolinen rajoite: etäisyyskenttä maskista, rangaistus vain jos ellipsin piste on ulkopuolella, jolloin jäätä ei näy välistä).
   Mallin periaate: projisoitu muoto sovitetaan suoraan etureunan pisteisiin (point-to-contour), ei vapaata ellipsisovitusta; silhuetin
   sisällä pysyminen = "malli projisoituu siluetin sisälle" -termi (signed distance transform), kuten siluettipohjaisessa asennonmäärityksessä.
3. *3D:* ellipsin keskipiste (u, v) on kiven akselilla; Y otetaan alareunan kaaresta (iskukaista), jolloin (X, z) ratkeaa kahdesta yhtälöstä
   (u, v) = projektio(X, Y, z) -> kahvan juuren korkeus z.

**Tulos (vasen, Y = 200-500 cm, 3 heittoa, kuva `tulokset/kahva_ellipsi.jpg`):** etureunan RMS 0,3-0,8 px, kärkirako <= 0,15 px, ulkopuolelle
jäävä osa enintään 0,1-0,8 px (mediaani 0,2-0,3 px). Kahvan korkeus z = 16,9 (keltainen), 17,1 (punainen), 17,1 (keltainen) cm: sama
keltaisella ja punaisella kahvalla. Muovin juuri on SOIKIO (ei ympyrä): puoliakselit 10,5 x 7,3 cm (~21 x 14,5 cm), sama kummallakin
värillä. Akselin sivusiirto alareunan kaaren akseliin nähden +1,5 / -0,1 / +0,7 cm.
Huom: Y > 500 cm alareunan kaari häiriintyy (hogline sulautuu kiven reunaan), siellä z on virheellinen (8-9 cm) -> käytä Y < 500.
z:n epävarmuus ~ ±3 cm, koska Y tulee alareunan kaaresta (R = 14,55 cm, z = 3 cm oletukset): Y:n 5 cm virhe -> ~3 cm virhe z:ssä.

**Ellipsin tarkistus (palaute: vasemman sarakkeen kuvissa seuranta edelleen huono).** `piirra_kahva.py` piirtää valitut ruudut isona ja
selitteellä, ellipsin osat kirjaimilla (A etureuna, B yläkärki, C takareuna, D alakärki):
`python piirra_kahva.py VIDEO tulokset vasen --ruudut 0:236,1:467 --kirjaimet` (kuva `tulokset/kahva_vasen_sarake.jpg`).
Kokeilut (kokeilukoodi, ei repossa; 193 ruutua, kaikki kolme heittoa, Y = 150-640 cm):
* *Muotokoe:* etureunaan sovitettiin super-ellipsi |x/b|^n + |y/a|^n = 1 (n vapaa). n:n mediaani 2,04 (IQR 1,9-2,1) ja etureunan RMS ei
  parane (0,82 vs 0,87 px) -> muovin juuren etureuna on kuvassa ellipsin kaari; "laatikkomainen" muoto ei selitä huonoja kuvia.
* *Etureunan RMS Y:n mukaan:* 0,6 (150-250), 0,8 (250-350), 0,5 (350-450), 0,8 (450-550), 1,6 px (550-650, hogline kiven vieressä).
  Heitto 1 mediaani 0,5 px, heitot 2 ja 3 ~1,1 px.
* *Riippumaton reunatesti* (varikanavan puolitasoristeys riveittäin silhuetin oikealla reunalla): ellipsi on reunassa +-0,5 px
  (heitto 1 Y=234 keskimäärin -0,2 px, heitto 2 Y=470 +-1 px).
Siis lukuina vasemman sarakkeen kuvat (heitto 1 Y=236: RMS 0,25 px; heitto 2 Y=467: RMS 0,8 px) eivät poikkea muista; mikä niissä näyttää
huonolta, tarkentuu käyttäjän palautteesta (kirjain A-D).

### Kiinteän kokoinen kahvamalli ja kuvan sumeus (`kahva_malli.py`, `mittaa_kahva_malli.py`, `kalibroi_kahvamalli.py`, `piirra_kahva.py`)

> **HUOM:** alla olevat mitat (A 11,4 / B 8,6 cm, z = 17 cm) ovat vanhalta kameran asennolta; korjatulla asennolla A = 10,1, B = 9,9 cm, z0 = 15 cm, ks. kappale "Päivitys korjatulle kameralle".

Palaute (kuvat heitto 1 Y=236 ja heitto 2 Y=467): etureuna (A) hyvä; keltaisessa B ja D liian lyhyet (muovi jatkuu), punaisessa B liian alhaalla
(heijastukset). Syyt: (1) sisälle-rajoite ja kärkiehto käyttivät värimaskia, josta käsi/hanska puuttuu -> käsi pakotti ellipsin takareunan (C)
eteenpäin ja katkaisi kärjet; (2) videon kromakanavat (a*, b*) ovat paljon sumeampia kuin kirkkaus: renkaiden reunoista mitattu pistelevitys
sigma = 2,1 px (kroma) vs 0,8 px (L; kroma 4:2:0), joten värikynnyksellä saatu silhuetti perääntyy kärjissä (1-2 px).
Korjaus (käyttäjän ehdotus: kaikilla kahvoilla sama koko oppimisvaiheessa):
* muovin juuri = vaakasuora soikio kiven akselilla (korkeus z = 17 cm), puoliakselit A (sivu) ja B (radan suunta) yhteiset kaikille ruuduille;
  per ruutu sovitetaan vain akselin (X, Y); kärjet tulevat koosta, eivät silhuetista
* malli = soikion peitto sumennettuna sigma = 2,1 px, verrataan väriin (keltainen b*, punainen a*) alipikselinä; vahvistus ja siirto ruuduittain
  lineaarisesti; käsi, hanska ja iho ovat "ei väliä" (paino 0): käsi ellipsin sisällä ei haittaa, vain jää ei saa näkyä
* heijastukset: malli > havainto -jäännös painaa 0,6 (muovin sisällä oleva valo ei vedä ellipsiä pois)
* alkuarvo ruudukkohaulla (vanha silhuettipohjainen alkuarvo valitsi osassa punaisia ruutuja hoglinen vaaleanpunaisen raidan)

**Kalibrointi käyttäjän käsin säätämistä ellipseistä** (`tulokset/kahva_ellipsit_kasin.json`, 8 ruutua, 3 heittoa, molemmat värit; raahattava sivu).
Ensimmäisen mallin (A 11,05, B 7,9, ilman siirtoja) pisteet poikkesivat merkinnöistä 0,92 px RMS: B (yläkärki) oli kaikissa 8 ruudussa 0,5-2,4 px liian
lyhyt (keskimäärin 1,3), C (takareuna) keltaisissa 1,6 px liian edessä, A (etureuna) oikein (keskimäärin +0,06 px); D keltaisissa +0,5, punaisissa -0,6 px.
`kalibroi_kahvamalli.py` ratkaisee merkinnöistä: koko A = 11,4 cm, B = 8,6 cm (kuvan mittakaavassa), kromahavainnon vakiosiirtymät (x: keltainen +0,34,
punainen -0,50 px; y: +0,58 px kummallakin) -> asetukset `tulokset/kahvamalli_asetukset_vasen.json`. Jäännös merkintöihin 0,41 px RMS ja
leave-one-out (vakiot 7 ruudusta, ennuste 8.) 0,47 px (4 pistettä, 8 koordinaattia; ruutukohtaisesti 0,17-0,79). Keltainen ja punainen sopivat samaan
kokoon (A 11,5 / 11,2, B 8,6 / 8,2 erikseen sovitettuna), joten sama koko -oletus pitää alle 3 % tarkkuudella. Mitat ovat kuvan mittakaavassa;
kiven halkaisija mittautuu samalla mittakaavalla ~7 % WCF-maksimia isommaksi, joten todelliset mitat voivat olla ~7 % pienempiä (noin 10,6 x 8,0 cm).
Kuvat: `tulokset/kahva_malli_vs_merkinnat.jpg` (magenta = malli, vihreä = käsin merkitty), `tulokset/kahva_malli_uudet_ruudut.jpg` (15 merkitsemätöntä ruutua),
`tulokset/kahva_vasen_sarake_malli.jpg`, `tulokset/kahva_malli_kuusi.jpg` (ensimmäinen versio). Kaikki 193 ruutua: `tulokset/kahvamalli_vasen.csv`
(sovituksen RMS 0,06-0,07 kaikilla heitoilla).

**Päivitys korjatulle kameralle (kalibroi_kahvamalli.py uudelleen samoista 8 merkinnästä, ruudukko A 9,4-11,8, B 9,4-12,6 cm).** A = 10,1 cm (sivu), B = 9,9 cm
(radan suunta), jäännös 0,41 px, leave-one-out 0,54 px (aiemmin 0,41 / 0,47 px vanhalla asennolla), kromasiirrot (x: keltainen +0,36, punainen -0,49 px; y: +0,68 px).
Muovilevy on siis PYÖREÄ (A ~ B, Ø ~20 cm): vanhan asennon epäfyysinen 11,4 x 8,6 soikio oli kameran asentovirheen jälki. Koe: A ja B vapaina ruutu kerrallaan
(65 ruutua, Y 150-750 cm): A = 9,85 / 9,85 / 9,6 / 9,85 / 9,7 cm ja B = 9,4 / 9,9 / 9,4 / 9,9 / 9,65 cm Y-lokeroissa 250-350 ... 650-750, eli koko pysyy vakiona +-1,5 mm
(kameran sivumittakaavassa ei ole Y-riippuvaa virhettä). Tason korkeus z0 = 15,0 cm (kahvan säde osuu rungon akseliin 14,4-15,5 cm korkeudella). Kuva `tulokset/kivet_ruudut.jpg`
(magenta = kahvan levy), tulokset `tulokset/kahvamalli_vasen.csv`.

**Käsin säädettävät esimerkkiellipsit (`luo_ellipsityokalu.py`, `ellipsityokalu_pohja.html`).** Käyttäjä voi raahata ellipsin neljää pistettä (A etureuna,
B yläkärki, C takareuna, D alakärki) kahdeksassa ruudussa niin, että ellipsi sopii muoviin; piste liikkuu oman akselinsa suuntaan ja vastakkainen
pää pysyy paikallaan (nuolinäppäimet 0,1 px). Alkuarvo on kiinteän kokoisen mallin ellipsi. Sivu on Artifact (tietokantakokoelma `ellipsit`, yksi
dokumentti per ruutu: ellipsi ja pisteet ruutukoordinaateissa, alkuarvo, huomio). Sivun luonti:
`python luo_ellipsityokalu.py VIDEO tulokset vasen --ulos sivu.html [--ruudut rata:Y,...]`. Käyttäjän tallentamat ellipsit ovat "oikea vastaus" mallin
koon (A, B), sumeuden ja kärkien säätöön sekä ilman kokoa olettavan tunnistuksen arviointiin.

### Kiven tarkka runkomalli ja koon pysyvyys (`kivi_tarkka.py`, `kalibroi_kivi.py`, `testaa_kivi_koko.py`, `piirra_kivet.py`)

Palaute: tunnista kivet tarkasti niin, että ne ovat samankokoisia ja samankorkuisia koko matkan (millien erot sallittu). Ratkaisu: kivellä on yksi
KIINTEÄ 3D-malli (koko ei vapaudu ruudulta toiselle eikä kivestä toiseen), ruutukohtaisesti sovitetaan vain akselin paikka (X, Y); koon pysyvyys
tarkistetaan erikseen VAPAALLA ruutukohtaisella sovituksella (X, Y, R, H). Edellytys oli kameran asennon korjaus (ks. "Kameran asento"): vanhalla asennolla
kiven säde näytti 15,0-15,7 cm ja korkeus riippui etäisyydestä.

**Malli.** Pyörähdyskappale: profiili r(z) 11 solmusta (Catmull-Rom, z = 0 ... H kalibrointitasosta) ja korkeus H. Kuvan sumeus sigma_eff = hypot(0,31 px, b * s_lat),
b = 0,56 cm: reunan fysikaalinen pehmeys (kiiltävän graniitin reunalla tumma ydin päättyy ennen geometrista siluettia, joten pehmeys kasvaa kuvamittakaavan s_lat
(1,5-2,8 px/cm) mukana; mitattu reunan 10-90 %:n leveys 2,8-4,7 px) + gamma 1,3 (videon gamma: tumma kohde näkyy hieman pienempänä). Havainto: graniitin peittävyys
`alfa_kartta` (0..1); malli = kiven siluetti (renkaiden kupera peite) ilman yläpinnan ellipsiä, sumennettuna ja gammalla; residuaali (malli - alfa) * paino reunakaistalla
(+-6 px alkuasennon reunasta, alipikseli, 4x ylinäytteistys). Kelvolliset pikselit (`kivi_tarkka.havainto`): ei kahvan muovia, ihoa eikä hihoja (jäällä: Lab a*b* poikkeaa taustasta > 9
tai L < 28; värillisellä taustalla (renkaat): etäisyys janasta tausta-graniitti > 45), ei kiiltoreikiä graniitin sisällä (alfa < 0,5 -alue < 80 px), ei taustan reunoja
(gradientti); yläpinnan ellipsi on pehmeä "ei väliä" -alue. Profiili on siis TUMMAN GRANIITTIYDIMEN profiili: sen reuna on alfa = 0,5 -taso, ei geometrinen ulkoreuna.

**Opetus** (`python kalibroi_kivi.py VIDEO tulokset vasen --H 12.3`): kimppusovitus (bundle adjustment, scipy `least_squares`, harva Jacobin rakenne): profiilin solmut, sigma_opt ja b
yhteisinä parametreina, jokaiselle 241 ruudulle oma (X, Y); 486 000 residuaalia; silotus- ja yksihuippurangaistus (yläpuolella r ei saa kasvaa).
Tulos `tulokset/kivimalli_vasen.json`: r = 8,0 / 9,9 / 11,7 / 12,0 / 12,8 / 13,6 / 13,7 / 13,7 / 13,8 / 13,9 / 14,1 cm solmuissa z = 0 / 0,5 / 1,2 / 2,2 / 3,4 / 4,9 / 6,4 / 7,9 / 9,3 / 10,8 / 12,3 cm,
residuaalin RMS 0,066. Kun alkuarvoja muutetaan (r -5 % ... +4 %, b 0,3 ... 0,9 cm), yläosan profiili (z > 4,9 cm) toistuu +-0,3 cm:iin ja RMS pysyy 0,066-0,068:ssa;
jalka (z < 2 cm: r(0) = 6,9 ... 8,3) on heikosti määrätty. Akselin paikka ei silti riipu mallin valinnasta: eri malleilla sivusuunnan ero on <= 0,01 cm ja radan suunnan vakiosiirtymä
-0,3 ... +0,5 cm (ei Y-riippuvuutta).

**Koe: koko ja korkeus pysyvät vakiona** (`python testaa_kivi_koko.py VIDEO tulokset vasen` -> `tulokset/kivi_koko_vasen.png/.csv`): jokaiselle hyvin näkyvälle ruudulle
(kelvollisia pikseleitä > 35 %, 195 ruutua) sovitetaan vapaasti (X, Y, sade R keskikorkeudella, korkeus H) kiinteällä profiilin muodolla kolmesta alkuarvosta.

| Y (cm) | n | R keskikorkeudella, mediaani (hajonta) cm | H, mediaani (hajonta) cm | heitto 1 R / H | heitto 2 R / H | heitto 3 R / H |
|---|---|---|---|---|---|---|
| 150...300 | 30 | 13,86 (0,28) | 12,19 (0,99) | 13,77 / 12,15 | 13,98 / 12,66 | 13,82 / 11,76 |
| 300...450 | 50 | 13,66 (0,13) | 11,84 (0,68) | 13,57 / 11,85 | 13,69 / 11,79 | 13,76 / 11,87 |
| 450...600 | 45 | 13,56 (0,41) | 11,78 (0,85) | 13,46 / 11,73 | 13,56 / 11,54 | 13,72 / 12,02 |
| 600...750 | 24 | 13,55 (0,54) | 11,86 (1,09) | 13,51 / 12,14 | 13,59 / 11,60 | - |
| Y 100...650 (vapaa liuku) | 137 | 13,68 (0,31) | 11,89 (0,83) | 13,55 / 11,89 | 13,65 / 11,80 | 13,76 / 11,94 |

Lukuina: kolmen kiven korkeus on 11,89 / 11,80 / 11,94 cm (ero 1,4 mm) ja sivusuuntainen puolileveys 13,55 / 13,65 / 13,76 cm (ero 2 mm); radan varrella R pienenee 1-3 mm (13,86 -> 13,55 cm
lokeroissa Y = 150-300 -> 600-750, malliton mittaus 13,75 -> 13,61 cm) ja H pysyy +-2 mm:ssä (lokeroiden mediaanit 11,8-11,9 cm, hajonta ruutua kohti 0,7-1,1 cm). Täysin malliton vertailu (`R_suora`: alfa-kartan 50 %:n reunat kiven leveimmässä kohdassa, ei mallia): puolileveys 13,75 / 13,67 / 13,60 / 13,61 cm
Y-lokeroissa 150-300 / 300-450 / 450-600 / 600-720 (ruutukohtainen hajonta 0,05-0,10 cm) ja heitoittain 13,63 / 13,59 / 13,80 cm. Kahvan levyn (väriympyrä, terävä reuna) koko on Y:n suhteen
vakio (A, B +-1,5 mm), joten pieni R:n lasku Y:n mukana kuuluu tumman ytimen reunaan (valaistus/heijastus), ei kameran sivumittakaavaan. Valmis luku: **kiven (näkyvän tumman graniitin)
sivusuuntainen puolileveys 13,7 cm (halkaisija 27,3 cm), korkeus 12,3 cm kalibrointitasosta; koko pysyy samana +-1,5 mm radan varrella ja +-1-2 mm kolmella kivellä.**
Kuvat: `tulokset/kivi_koko_vasen.png` (R(Y), H(Y) heitoittain), `tulokset/kivet_ruudut.jpg` (siluetti 18 ruudussa: 3 heittoa x Y = -200 ... 700 cm; syaani = runko, keltainen = ylärengas,
magenta = kahvan levy).

**Rajoitteet (rehellisesti).**
* *Korkeus H ei ole siluetista tarkasti mitattavissa:* kiven yläpinta jää kahvan muovin alle, joten H on korreloitunut profiilin yläosan kanssa. Kun H kiinnitetään ja profiili sovitetaan, kustannus
  kasvaa H:n mukana (H = 9,5 / 11,5 / 12,5 / 13,5 / 14,5 / 15,5 cm: 960 / 977 / 1004 / 1072 / 1155 / 1276), mutta alle ~11 cm:n arvoilla profiilin yläreuna leviää epäfyysisesti (r 14,4 ... 16 cm). Siluetti suosii siis matalaa runkoa,
  jos yläreunan leveneminen sallitaan. H on kiinnitetty arvoon 12,3 cm (vapaa ruutukohtainen H-mediaani 11,9 cm; ilman levenevää yläreunaa alaraja on 13,5 cm; WCF: runko >= 11,4 cm, tyypillisesti 11,4-12,7 cm)
  ja pysyvyys osoitetaan sen ympärillä. Kalibrointitaso on maalattujen viivojen taso; jään pinta on sen yläpuolella (arviolta 0-2 cm). Kahvan levyn taso z0 ~ 15 cm on ~2,7 cm rungon yläreunan
  yläpuolella (levy on pyöreä ja ohut: A ~ B, ei kupua).
* *Taloalue (Y ~ -190 ... 190 cm):* maalatut renkaat ovat graniitin kanssa samankirkkaita, joten kiven reuna näkyy vain kromalla (sumea, sigma 2,1 px) ja kiven varjo sinisellä
  renkaalla on tumma; kelvollisia pikseleitä on vähän (6-30 %) ja sivusuuntainen hajonta 0,2-0,6 cm (vapaalla jäällä 0,03-0,16 cm). `kelvolliset`-sarake (kivet_vasen.csv) kertoo luotettavuuden.
* *Kahvan levyn ja rungon akselin ero:* levyn mallin akseli (z0 = 15 cm) ja rungon siluetin akseli eroavat radan suunnassa -0,3 / +0,2 / +1,4 / +0,9 / +0,6 cm (Y = 250-350 ... 650-750) ja sivusuunnassa
  -0,07 ... -0,45 cm; heitoittain dY = +0,5 / -0,1 / +1,6 cm. Kahvan säde osuu rungon akseliin korkeudella 14,4-15,5 cm. Kahvan väriseurannan keskipiste (`kahvaradat_*.csv`, z = 15 cm) poikkeaa
  rungon akselista enemmän (1,2 / 3,3 / 2,5 cm heitoittain): irroituksen tunnistuksessa on käytettävä levyn mallia (`kahvamalli_*.csv`), ei värin keskipistettä.

### Kiven liuku ja irroitus (`liuku_irroitus.py`)

Pyyntö: tunnista kiven liuku ja irroitus (irroituksen suunta). Hypoteesi: irroituksen voi todennäköisesti tunnistaa pelkästään siitä, että kivi kulkee näkyvällä matkalla hyvin tasaisesti
2. asteen yhtälön mukaan ilman teräviä suunnanmuutoksia. Toteutus on tämän hypoteesin tilastollinen koe, ja se toimii vain todellisella aikaperustalla (ks. "Videon todellinen aikaperusta": k/25-ajalla
ruutuajoituksen virhe, 2-3 cm RMS, peittää kiihtyvyyden muutoksen 0,1-0,3 m/s2 kokonaan).

`python liuku_irroitus.py VIDEO tulokset vasen [--ulos tulokset/liuku_irroitus_vasen] [--ei-ruutuja]` -> `<ulos>.json` (kaikki luvut), `<ulos>.png` (rata, suunta, nopeus ja kiihtyvyys, vaihtokohdan profiili),
`<ulos>_ruudut_h1..3.jpg` (kaikki yksilölliset ruudut irroituksen ympäriltä, punainen kehys = kinemaattinen irroitus, vihreä teksti = silmin todettu irtoamisväli). Vain numpy (+ matplotlib, cv2 kuviin);
syötteenä pelkkä `radat_<puoli>.csv`, ei kahvaa eikä kättä.

* **Liuku:** lähtö = viimeinen hetki ennen työntöä, jolloin nopeus < 0,15 m/s (hack); työnnön loppu = nopeus >= 90 % tasanteesta (tasanne = nopeuden mediaani Y = -50...300 cm). Jos rata alkaa jo liikkeestä
  (heitto 2: kivi näkyy vasta kun nopeus on 0,8 m/s), lähtö ARVIOIDAAN vakiokiihtyvyydellä taaksepäin (+-0,2 s). Liuku = lähdöstä irroitukseen.
* **Irroitus = vapaan lennon alku:** irroituksen jälkeen kivi on vapaa kappale, jolloin X(Y) on paraabeli (kaarto) JA Y(t) on 2. asteen polynomi (vakiohidastuvuus). Ennen irroitusta heittäjä työntää ja ohjaa:
  kiihtyvyys ja kaarevuus ovat toiset. Malli: kaksi paraabelia, joiden arvo JA ensimmäinen derivaatta (nopeus, suunta) jatkuvat vaihtokohdassa ja joiden 2. derivaatta hyppää (C1-sovitus), molemmille
  kanaville (Y(t) ja X(Y)) yhteinen vaihtokohta, khi^2-summa, Huber-kustannus (yksittäiset 3-10 cm:n mittausvirheet ohjatussa vaiheessa eivät hallitse). Kohinan sigma otetaan radan viimeisen 3 m:n
  (vapaa lento) robustista hajonnasta: sigma_Y 0,5-0,9 cm, sigma_X 0,06-0,27 cm. Vaihtokohta haetaan ruudukolla (20 ms), 2-sigma-väli profiilista.
* **Vakaus (kokoelma + tarkennus):** ohjattu vaihe ei ole paraabeli koko matkaltaan, joten vaihtokohta haetaan useilla ikkunoilla. Kierros 0: ikkuna alkaa työnnön lopusta + 0,5 ... 3 m (karkea). Tarkennuskierrokset:
  ikkuna alkaa L = 1,5 / 2 / 2,5 / 3 m ENNEN edellistä vaihtokohtaa (paikallinen ikkuna on lähempänä mallia), kunnes vaihtokohta muuttuu < 10 cm (2-4 kierrosta). Tulos = jäsenten mediaani, vaihteluväli = jäsenten ääriarvot.
* **Merkitsevyys:** khi^2-parannus yhteen paraabeliin verrattuna, kynnys 40. Nollahypoteesi (yksi paraabeli + Gaussin kohina, 40 simulointia/heitto): 99. persentiili 34 / 29 / 9, suurin 35; todellisissa vapaan lennon
  ikkunoissa (taustaviivat, logot mukana) suurin 33. Havaitut jäsenten arvot 45-1309 (mediaanit 326 / 1247 / 135).
* **Irroitustila:** vapaan lennon sovitus (t >= t_r + 0,12 s), arvot hetkellä t_r: paikka, suunta (kulma Y-akseliin, + = oikealle heittäjästä katsoen), nopeus, hidastuvuus, kaarevuus. Suunnan virhe =
  sovituksen LS-virhe + vaihtokohdan epävarmuus (kaarevuus * puoli Y-väliä) + radan paikallisen suunnan (ikkuna 120 cm) hajonta paraabelin ympärillä (taustaviivojen/logojen sivuttaishäiriö), neliösummana.

**Tulos (vasen puolisko).** Silmin todettu irtoaminen (`tulokset/irroitus_visuaalinen_vasen.json`, kontaktiarkit ruutu kerrallaan): `ruutu_ote` = viimeinen ruutu, jossa sormet vielä pitävät kahvasta, `ruutu_irti` = ensimmäinen
ruutu, jossa sormien ja kahvan välillä on selvä rako; fysikaalinen irroitus on niiden välissä (+-1 ruutu lukuepävarmuus, korjaa tarvittaessa).

| | heitto 1 (keltainen) | heitto 2 (punainen) | heitto 3 (keltainen) |
|---|---|---|---|
| lähtö (t, Y) | 12,28 s, -296 cm | 59,08 s, -286 cm (arvio) | 107,11 s, -297 cm |
| työntö: kesto, matka, tasanne | 1,20 s, 86 cm, 1,79 m/s | 0,82 s, 73 cm, 2,08 m/s | 0,84 s, 87 cm, 2,37 m/s |
| **irroitus (kinemaattinen)** Y, vaihteluväli | **376 cm** (352...408) | **254 cm** (208...280) | **361 cm** (344...384) |
| irroitus t | 16,71 s | 62,24 s | 110,42 s |
| silmin todettu irtoaminen Y (ruudut) | 359...393 cm (496...502) | 377...411 cm (1880...1884) | 354...394 cm (3309...3314) |
| ero silmin todettuun | välissä | **123 cm (0,6 s) ennen** | välissä |
| khi^2-parannus (jäsenten mediaani) | 326 | 1247 | 135 |
| liuku: kesto, matka | 4,43 s, 673 cm | 3,15 s, 540 cm | 3,31 s, 658 cm |
| **suunta irroituksessa** | **+4,97 +- 0,23 astetta** (oikealle) | **+1,04 +- 0,07** | **-0,20 +- 0,28** |
| nopeus irroituksessa | 1,87 m/s | 2,13 m/s | 2,52 m/s |
| hidastuvuus vapaassa lennossa | -0,064 m/s2 | -0,064 m/s2 | -0,098 m/s2 |
| kaarevuus (vapaa lento) | -0,63e-4 /cm (vasemmalle) | -0,21e-4 /cm | ~0 |
| X irroituksessa | 44,7 cm | -15,0 cm | -20,7 cm |

Herkkyys: vaihtokohta pysyy +-15 cm:ssä, kun ikkunajoukko (L = 1-2 m / 2-3 m / 1,5 ja 2,5 m) tai käytetyn radan loppu (Y <= 640 / 700 / kaikki) vaihtuu: 367-389 / 254-280 / 349-369 cm. Kanavat erikseen:
Y(t) 369 / 238 / 366 cm, X(Y) 391 / 273 / 327 cm.

**Tulkinta (rehellisesti).**
* *Hypoteesi toimii sellaisenaan:* vapaan lennon alku erottuu radasta selvästi (khi^2-parannus 45-1309 vs nollahypoteesin <= 35) ja on vakaa. Heitoissa 1 ja 3 kiihtyvyys vaihtuu työntävästä (+0,06 ... +0,37 m/s2) vapaan lennon
  hidastuvuuteen (-0,06 ... -0,10 m/s2) ja kaarevuus muuttuu samassa kohdassa; vaihtokohta osuu silmin todetun irtoamisvälin sisään (+-0,15 s).
* *Mitä vaihtokohta tarkoittaa:* se on hetki, jonka jälkeen käden vaikutus rataan on nolla (kivi on vapaa kappale), ei sormien irtoamishetki. Heitossa 2 käsi seuraa kahvassa vielä ~1,2 m (0,6 s) ilman että rata tai nopeus muuttuu
  (jatkokoe: vaihtokohdan jälkeen ei ole toista selvää vaihtokohtaa: khi^2-parannus 8-30 ikkunoilla Y >= 314 cm ja 47 ikkunalla Y >= 284 cm, eli kynnyksen tuntumassa), joten sormien irtoaminen ei näy radassa lainkaan: pelkällä ratatasaisuudella tulos on tässä aikaisempi kuin kuvasta nähtävä irtoaminen.
  Jos tarvitaan sormien irtoamishetki myös tällaisille liu'ille, se vaatii kuvapohjaisen käden/kahvan erotuksen (aiemmassa kokeessa keltaisella kahvalla +-0,05-0,2 s, punaisella ei toiminut).
* *Mittausraja:* vapaassa lennossa paikan kohina on Y 0,5-0,9 cm ja X 0,06-0,27 cm (taustaviivat/logot Y ~ 520-660: sivusuunnassa +-0,5 cm häiriöt); ohjatussa vaiheessa heittäjän vartalo aiheuttaa 3-10 cm:n yksittäisiä virheitä ja aukkoja
  (kiihtyvyys ennen irroitusta ei ole luotettava: heitossa 2 hold-vaiheen a1 ei ole fysikaalinen), joten vaihtokohdan tarkkuus on realistisesti +-0,1-0,15 s kun signaali (työntö) on selvä.
* *Kierre:* kierron nopeus ja suunta mitataan kuvasta erikseen (kahvan palkki, `kierto_netti.py`, ks. seuraava luku); radan kaarevuuden merkki on vain tukitieto (negatiivinen = rata kaartuu vasemmalle, tavallisen curl-säännön mukaan kierto vastapäivään).
* *Validointi on pieni:* kolme heittoa, silmämääräinen viite. Koe kannattaa toistaa useammalla heitolla (ja oikean puolen kameralla).

**Vastaukset kysymyksiin (liuku, irroitus, kierre).** `python suunnat_pesakuvaan.py tulokset vasen` kirjoittaa taulukon `tulokset/irroitus_yhteenveto_vasen.csv`. Määritelmät: *irroituksen suunta* = vapaan lennon suunta
irroitushetkellä (kulma Y-akseliin, + = oikealle heittäjästä katsoen), eli mihin kivi lähti viimeisen käden vaikutuksen jälkeen; *liun suunta* = X(Y):n pienimmän neliösumman suora hackista irroituspisteeseen, eli mihin kivi oli keskimäärin menossa
ennen viimeistä käden vaikutusta; *nopeus irroituksen jälkeen* = vapaan lennon sovituksen nopeus irroitushetkellä; *nopeus juuri ennen irroitusta* = keskinopeus viimeiseltä 0,4 s:lta ennen irroitusta; *liun hidastuvuus* = -(kiihtyvyys) liu'un aikana
työnnön jälkeen (työnnön loppu + 0,6 s ... irroitus - 0,15 s), paikallisten nopeuksien Theil-Sen-suora, 90 %:n väli bootstrapista; negatiivinen hidastuvuus = kivi kiihtyy liu'un aikana. Kierre: ks. "Kiven kierre".

| | heitto 1 (keltainen) | heitto 2 (punainen) | heitto 3 (keltainen) |
|---|---|---|---|
| **kierre** (heittopään kamera, kierron suunta ylhäältä katsoen) | **vastapäivään 54 astetta/s** (47...69) = 1 kierros / 6,7 s | **vastapäivään 99 astetta/s** (86...107) = 1 kierros / 3,7 s | **vastapäivään 70 astetta/s** (54...89) = 1 kierros / 5,2 s |
| **irroituksen suunta** | **+4,97 +- 0,23 astetta** | **+1,04 +- 0,07** | **-0,20 +- 0,28** |
| **liun suunta** (hackista irroitukseen) | **+4,31 astetta** | **-0,20** | **-0,65** |
| **nopeus irroituksen jälkeen** | **1,87 +- 0,01 m/s** | **2,13 +- 0,01** | **2,52 +- 0,02** |
| **nopeus juuri ennen irroitusta** (viim. 0,4 s) | **1,87 +- 0,01 m/s** | **2,07 +- 0,01** | **2,44 +- 0,02** |
| **liun hidastuvuus** | **-0,04 m/s2** (-0,07 ... -0,01) = kiihtyy | **+0,06 m/s2** (+0,01 ... +0,20) | **-0,14 m/s2** (-0,20 ... -0,09) = kiihtyy |
| (kiven hidastuvuus irroituksen jälkeen) | 0,064 m/s2 | 0,064 m/s2 | 0,098 m/s2 |
| suunnat toisen pään T-viivalla: irroitus / liuku (+ = heittäjän oikealle) | +314 / +278 cm | +43 / -27 cm | -32 / -56 cm |

Nopeus on jatkuva irroituksessa (liikemäärä ei hyppää): heitoissa 2 ja 3 "ennen"-arvo on 0,06-0,08 m/s pienempi, koska se on 0,4 s:n keskiarvo (keskipiste 0,2 s ennen irroitusta) ja kivi kiihtyy työnnön aikana (+0,1 ... +0,4 m/s2 viimeisen 1-1,5 s aikana).
Liu'un hidastuvuus on käytännössä nolla (+-0,15 m/s2): nopeus pysyy tai kasvaa hieman liu'un aikana; heitossa 2 ohjatun vaiheen mittaus on heikko (heittäjän vartalo, aukot), joten sen väli on leveä. Irroitus on kinemaattinen (käden vaikutus rataan päättyy, ei sormien irtoamishetki).

### Kiven kierre (`kierto_netti.py`)

Pyyntö: millä nopeudella ja suunnalla kivi lähti pyörimään; kierre tunnistetaan HEITTOPÄÄSTÄ (vasen kamera, irroituksen jälkeinen vapaa lento, 1,0-1,3 s:n jakso), ei kaukopäästä.
`python kierto_netti.py VIDEO tulokset vasen [--valimuisti tiedosto.pkl]` -> `tulokset/kierto_vasen.json` (kaikki luvut), `kierto_vasen.png` (sauvan kulma ajan funktiona ja kustannus omegan funktiona), `kierto_vasen_sovitus.png` (malli vs havainto).

* **Havainto:** kahvan muovilevyn (`kahva_malli.py`, pyöreä levy, z = 15 cm) ulkopuolella näkyy kahvan palkki. Kamera katsoo kiveä loivasti sivusta (17-26 astetta), joten palkki näkyy levyn vasemmalla puolella (korkeus siirtää sen kuvassa vasemmalle,
  ~2,1 px/cm) ja sen pystysuuntainen laajuus kertoo sivusuuntaisen suuntakomponentin. Havainto on kahva_malli.havainto:n muovin todennäköisyyskartta p (0..1) levyn (laajennetun) ulkopuolella, alueella 2...36 px levyn keskipisteestä vasemmalle, ja iho/hanska/hiha (`ei valia`-pikselit) jää pois;
  ruutu hylätään, jos kasi peittää yli 60 pikseliä alueesta.
* **Malli:** vaakasuora, kiven akselin kautta kulkeva palkki (silta) korkeudella h muovilevyn yläpuolella, pituus ell, leveys w; kulma theta (X-akselista Y-akseliin, + = vastapäivään ylhäältä katsoen) kiertyy tasaisesti: theta(t) = theta0 + omega (t - t_keski).
  Malli piirretään kameralla kuvaan (kuvan sumeus sigma = 2,1 px) ja sovitetaan p-karttaan kaikissa vapaan lennon ruuduissa yhdessä (soft-L1); muoto (h, ell, w) on yhteinen saman väriselle kahvalle (keltaiset heitot 1 ja 3 yhdessä: h = 5,3 cm, ell = 12,5 cm, w = 4,9 cm;
  punainen heitto 2: h = 4,8, ell = 11,9, w = 6,6 cm), theta0, omega ja kirkkaus heittokohtaisia. Aika on todellinen aika t_kor_s (`aikaperusta.py`). Alkuarvot: ruudukkohaku (theta0 x omega), 3 parasta per heitto, kaikki yhdistelmät optimoidaan.
* **Kaksi estimaattoria:** (1) yhteissovituksen omega, (2) ruutukohtaiset kulmat (paras theta kullekin ruudulle, muoto kiinteä) ja niiden Theil-Sen-kulmakerroin (bootstrap 90 %-väli). Lopullinen omega = näiden keskiarvo, väli = Theil-Senin bootstrap-väli (laajennettuna kattamaan yhteissovitus).
* **Suunta:** palkki on symmetrinen, joten kulma on määrätty modulo 180 astetta; kiertosuunta näkyy palkin näkyvän pään kallistuksen muutoksesta (alas/ylös), ja suunta on se, jolla sovitus on selvästi parempi: kustannus vastapäivään / myötäpäivään / ei kiertoa =
  380 / 409 / 403 (heitto 1), 181 / 242 / 227 (heitto 2), 380 / 397 / 392 (heitto 3).

| | heitto 1 (keltainen) | heitto 2 (punainen) | heitto 3 (keltainen) |
|---|---|---|---|
| kierre | **vastapäivään 54 astetta/s** (47...69) | **vastapäivään 99 astetta/s** (86...107) | **vastapäivään 70 astetta/s** (54...89) |
| yhteissovitus / Theil-Sen | 51 / 57 | 99 / 98 | 73 / 67 |
| kierrosaika | 6,7 s | 3,7 s | 5,2 s |
| ruutuja (aika vapaassa lennossa) | 31 (17,2 ... 18,5 s) | 31 (63,1 ... 64,4 s) | 23 (110,8 ... 111,8 s) |
| puolikkaat (1. / 2. puoli) | 60 / 70 | 60 / 80 | 20 / 130 |

Rajoitukset (rehellisesti): omega on keskiarvo näkyvältä vapaan lennon jaksolta (1,0-1,3 s, keskipiste 0,6-1,5 s irroituksen jälkeen; kierron hidastumista ei mallinneta, joten irroitushetkellä kierre on korkeintaan hieman suurempi); puolikkaiden arvot vaihtelevat paljon (12 ruudun ikkunat), joten väli on
suuntaa antava (+-15-25 %); malli on yksinkertaistus (suora palkki, ei jalkoja/hakaa) mutta sen ääriviivat seuraavat havaittua kahvaa kaikissa kolmessa heitossa (`kierto_vasen_sovitus.png`); kierron suunta on vastapäivään kaikilla heitoilla, mikä sopii radan vasemmalle kaartuvaan
kaarevuuteen (heitto 1: -0,63e-4 /cm). Kierto mitataan palkin suunnasta, ei kiven pinnasta, joten tulos riippuu siitä, että palkki on kiinteä kiveen.

### Suunnat toisen päädyn pesäkuvaan (`suunnat_pesakuvaan.py`)

`python suunnat_pesakuvaan.py tulokset vasen [--alkuperainen]` -> `tulokset/suunnat_pesakuva.png` (heittäjän näkökulma: kaukopää ylhäällä, + = heittäjän oikealle) ja `tulokset/suunnat_pesakuva_alkuperainen.png` (toisen pään oma top-down, hog ylhäällä; kuvassa oikea = heittäjän vasen).
Heittopään radasta lasketut suunnat (liun suunta ja irroituksen suunta, ks. yllä) jatketaan suorina viivoina toiseen päähän (T-viivojen etäisyys 34,747 m, `rata.HOUSE_DISTANCE_CM`) ja piirretään toisen pään pesän top-down-kuvaan (`topdown_oikea.png`, kalibrointi_netti.py; 2 px/cm); viivat
päättyvät T-viivaan ja niiden osumakohta (cm keskiviivasta) on merkitty. Koordinaatit: toisen pään fyysinen koordinaatisto on heittopään 180 asteen kierto pystyakselin ympäri: X' = -X, Y' = 3474,7 - Y (molemmat koordinaatistot alkavat oman pesän keskipisteestä, Y kohti radan keskikohtaa).
Tarkistus: kameroiden asennot ovat samassa pisteessä radan keskellä (X = +4 cm vasen / -2 cm oikea omissa koordinaatistoissaan ja Y = 1718 + 1724 = 3442 vs 3474,7 cm), ja kuvissa näkyvä puulattia on molemmissa kameroissa radan samalla puolella (vasemmassa -X, oikeassa +X').
Viivat ovat suuntia, eivät ennuste: kierre (curl) kaartaa kiveä matkalla (heitto 1: kaarevuus -0,63e-4 /cm eli suunta muuttuu noin -11 astetta 31 metrin matkalla), joten viivat osoittavat tähtäyspisteen. Heiton 1 viivat osuvat T-viivalle radan ulkopuolella (+314 / +278 cm; radan reuna 237,5 cm).
Suunnan virhe 31 m:n päässä: +-12 cm (heitto 1), +-4 cm (2), +-15 cm (3) irroituksen suunnalle (tilan virhearvio, ks. yllä). Kuvan pesän edessä näkyvä kivi on taustakuvan (moodikuva) kivi.

### Reaaliaikainen versio: kuva viiveellä ilman top-down-muunnosta (`netti_live.py`, `netti_seuranta.py`, `netti_piirto.py`, `netti_geom.py`, `netti_tausta.py`)

Pyyntö: kuvia ei muunneta top-down-kuviksi vaan ne säilyvät normaalin näköisinä; tulosteen kuva tulee 5 s viiveellä, jolloin analyysi ehtii seurata kiven kulun ennen kuin kivi liikkuu tulosteessa;
kun kiven piirto alkaa, toisessa päässä näytetään heti liukulinja ja irroituslinja ja heittopäässä kiven keskikohta pisteinä livenä (pistejoukko jää näkyviin 10 s); lähteenä YouTube (oma `yt_dlp` + OpenCV -testikoodi).

```
python netti_live.py --youtube [hnTTLqHr5x8]      # YouTube (yt_dlp + OpenCV kuten oma testikoodi), live-ikkuna TÄYDELLÄ resoluutiolla 1920 x 1080 (kuva 5 s viiveellä); 'q' / ESC / ikkunan sulkeminen lopettaa, 'f' vaihtaa kokoruudun
python netti_live.py --youtube --ulos             # sama + tallennus tiedostoon kivenseuranta_<päiväys>_<aika>.mp4 (tai --ulos oma.mp4); tiedosto valmistuu lopetettaessa
python netti_live.py --lahde netti.mp4            # tiedosto reaaliajassa (nopeus kuin live-lähteessä)
python netti_live.py --url http://.../virta.m3u8  # suora virran osoite (sama OpenCV-polku kuin YouTubessa)
python netti_live.py --lahde netti.mp4 --nopea --ulos ulos.mp4 --ei-ikkunaa --csv heitot.csv    # simulaatio: niin nopeasti kuin ehtii, tulos tiedostoon
```
Valitsimet: `--viive 5` (tulosteen viive s), `--jalki 10` (pistejoukko näkyvissä näin kauan heiton jälkeen), `--viivat 60` (linjat ja paneeli näkyvissä heiton alusta seuraavaan heittoon tai tähän aikaan asti), `--saikeet 3` (tarkan sovituksen työprosessit; 0 = vain kahvan väri),
`--ulos [tiedosto]` (tallenna tuloste; ilman nimeä aikaleimattu nimi), `--koodekki auto|cpu|gpu|mp4v|<fourcc>` (ks. alla), `--x264-preset superfast` (H.264-koodauksen nopeus; `ultrafast` = kevein prosessorille), `--ffmpeg POLKU`, `--nayttotila auto|ikkuna|kokoruutu` ja `--ikkuna-leveys` (ks. alla),
`--puskuri yuv|jpeg` (viivepuskurin muoto), `--ei-kierretta` (ei kierteen eikä hog-hog-ajan mittausta), `--tapahtumat` (tulostaa seurannan kaikki tapahtumat: viritys, lähtö, vahvistus, hylkäys, "ei kiveä", valmis; näyttää myös heittoyritykset, jotka jäävät tunnistamatta; käynnistyksessä tulostuu aina `netti_live: git <commit> <aika>; Python <versio>`, josta näkee mitä versiota ajettiin), `--ei-ikkunaa`, `--alku/--loppu` (tiedoston aikaväli), `--yt-formaatti` (yt_dlp-formaattivalitsin; oletus oma testikoodi), `--fps-lahde` (erillisten ruutujen nimellistaajuus aikaperustaa varten; oletus: päätellään kopioruutujen osuudesta,
tässä videossa 26 % kopioita -> 25 r/s, kopiottomassa virrassa ruutunopeus), `--csv`. Tarvitaan: `opencv-python`, `numpy`, `scipy`, `yt-dlp` (+ `deno` JavaScript-ajuriksi); suositus: `ffmpeg` (tai `pip install imageio-ffmpeg`) nopeaan tallennukseen ja valinnaisesti käännetty `netti_fit` (C++, ks. alla).

**Toistettava reaaliaika (tiedosto ja YouTube-toisto).** Tulos ei riipu koneen nopeudesta eikä työprosessien ajoituksesta: `netti_live.py` (ilman `--nopea`) antaa tiedostolla ja YouTube-toistolla CSV-rivit, jotka ovat tismalleen samat kuin simulaatio (`--nopea`). Aiemmin reaaliaikatila ja simulaatio eroasivat, koska B-sovitukset (tarkka kiven paikka) laskettiin työprosesseissa eikä seuranta odottanut niitä: tulokset otettiin käyttöön sitä mukaa kuin ehtivät, B jätettiin pois jonon ruuhkautuessa ja analyysi ohitti ruutuja, kun kone oli myöhässä (heitto 72: sama video, A 0,0080 / 0,0083 / 0,0092 / 0,0122 eri ajoilla). Nyt (`Analyysi.aseta_toistettava`): ruudun f B-tulokset otetaan käyttöön kiinteällä viiveella (ruutu f - 6 odotetaan, B:tä ei pudoteta), viivaestimaatit lasketaan kaikki järjestyksessä, kaukopään kierron aloitus odottaa heittopään kierron valmistumista (ennakko alias-ratkaisuun) ja kiven vakiot lasketaan vasta kun lopullinen estimaatti on valmis; ruutuja ei ohiteta, vaan jos kone ei jaksa, kuva tulee myöhässä (piirrokset odottavat analyysia). Live-virrassa (ei toistettava) ja `--ohita-myohassa`-valitsimella vanha tapa: ruutuja ohitetaan, tulos riippuu koneesta. Työprosessit käyttävät yhtä BLAS-säiettä (ei ylikuormaa, toistettavampi), ja ruudut luetaan erillisessä säikeessä (`--esilataus 24`). Loppuyhteenveto kertoo ajankäytön (analyysi ms/ruutu vaiheittain, lähteen luku, pisin katko); `--ei-tahdistusta` lukee tiedostoa niin nopeasti kuin analyysi ehtii (testaus). Tarkistus omalla koneella: `python tools/testaa_toistettavuus.py video.mp4 --alku 4825 --loppu 4885` (simulaatio vs. toistettava ajo, kaikkien sarakkeiden pitää olla samat). Koko video (7625 s) toistettavana ajona on yhtä pitkä kuin simulaatio mutta noin 2 x nopeampi (4 ydintä).

**Ikkuna täydellä resoluutiolla.** Ikkuna näyttää ruudun sellaisenaan 1920 x 1080 -pikselin kokoisena (aiemmin ikkuna pienennettiin 1280 x 720:ksi). `--nayttotila auto` (oletus): jos näyttö ei ole ruutua suurempi (esim. 1920 x 1080 -näyttö), ikkuna avautuu kokoruutuna, jolloin kuva on 1:1; suuremmalla näytöllä ikkuna on 1920 x 1080. `f` vaihtaa
kokoruudun päälle / pois ajon aikana; `--nayttotila ikkuna --ikkuna-leveys 1280` palauttaa pienemmän ikkunan. Windowsissa ohjelma asettaa itsensä DPI-tietoiseksi, jotta 125-150 %:n näyttöskaalaus ei pienennä kuvaa.

**Tallennus.** `--koodekki auto` (oletus): jos `ffmpeg` löytyy (PATH tai `imageio-ffmpeg`), tallennus tehdään sen putkeen: laitteistokooderi (NVIDIA NVENC, Intel QSV, AMD AMF; kokeillaan ensin lyhyellä testikoodauksella, joten vain toimiva kooderi valitaan) tai H.264 prosessorilla (`libx264`, `--x264-preset`); muuten OpenCV:n `mp4v` (iso tiedosto).
Koodaus tapahtuu omassa prosessissa ja omassa säikeessä alemmalla prioriteetilla, eikä tallenteesta puutu yhtään ruutua (ikkuna voi ohittaa ruutuja, tallenne ei); `--koodekki cpu` pakottaa `libx264`:n, `gpu` laitteistokooderin, `mp4v` OpenCV:n.

**Aikajana (kaikissa tiloissa sama).** Jokainen saapuva ruutu analysoidaan heti; tulostettava kuva on analyysia 5 s jäljessä (JPEG-puskuri, ~40 MB). Kun kivi alkaa liikkua tulosteessa, analyysi on jo nähnyt seuraavat 5 s: heitto on tunnistettu, kiven rata on tiedossa (liu'un ja
irroituksen mitta-aineisto) ja viivaestimaatti valmis, joten toisen pään kuvaan piirtyy liukulinja (oranssi) ja irroituslinja (magenta) siinä samassa ruudussa, jossa heittopään pistejono alkaa. Viivat tarkentuvat koko ajan (estimaatti päivitetään kun aineistoa on kertynyt ≥ 10 kelvollista ruutua lisää),
alustava irroituslinja on katkoviiva ja muuttuu yhtenäiseksi kun irroitus on merkitsevä, vapaata lentoa on ≥ 2,5 m ja irroituspaikka on pysynyt (< 30 cm) edellisestä estimaatista. Heittopäässä jokaisen ruudun kiven keskikohta jään tasolla (kiven akselin kohta jäällä, z = 0; ei kahvan korkeudella) piirretään pisteeksi, kun ruutu tulee
tulosteeseen; nykyinen paikka on rengas; pistejoukko poistuu 10 s heiton (viimeisen seurattavan ruudun) jälkeen. Linjat ovat suoria ja ne jatketaan toisen pään T-viivalle (osumakohta merkitty ympyrällä, nimike `LIUKU/IRROITUS xxx cm oik/vas`, `>>` jos osumakohta on kuvan ulkopuolella;
oik = heittäjän oikealle = kaukopään kuvassa oikealle); kivi kaartuu (kierre), joten todellinen rata poikkeaa irroituslinjasta sitä enemmän mitä pidemmälle mennään (tähtäyspiste, ei ennuste). Ylhäällä (kuvan ylälaidan vapaassa osassa) on tietopaneeli; sen kahdeksan riviä ja luvut selitetään alla kohdassa "Tietopaneeli".

**Ruutuja ei muunneta.** `netti_geom.py`: maa (X, Y, z; cm) <-> alkuperäisen 1920 x 1080 -ruudun pikselit (kierros-virhe < 1e-4 cm), `netti_seuranta.Kuvain`: vain kiven ympärillä oleva leike (ROI) oikaistaan suoraan alkuperäisestä ruudusta yhdellä `cv2.remap`-kutsulla (kierto ja linssikorjaus valmiissa
kartoissa; tulos = `Kamera.muunna` samalle alueelle, ero <= 1 harmaasävy). Piirrokset heijastetaan kamerageometrialla takaisin alkuperäiseen ruutuun; tulosteen kuva on sellaisenaan lähteen kuva.

**Seuranta (`netti_seuranta.py`).** Kaksi tasoa: A) kahvan väri (keltainen / punainen, ROI, 1-2 ms; alkuarvo, liipaisu, ennuste) ja B) kiven tarkka kiinteän kokoinen runkomalli (`kivi_tarkka.py`, sama mittaus kuin offline-ajossa) vain kiven ROI:ssa työprosesseissa (~55 ms/ruutu/prosessi; vain leike
siirtyy prosessille); pelkkä kahvan väri ei riitä viivoihin (hajonta 5-10 x suurempi ja Y:ssä systemaattisia 2-6 cm:n poikkeamia, tuloksena T-viivalla jopa 30 cm:n virheitä). B-sovitus hyväksytään vain jos graniittia on rungon siluetin sisällä (`peitto` >= 0,35; kivi 0,4-0,95, ei-kivi 0-0,2), tiukka B lisäksi
kelvollisia pikseleitä >= 0,2 (estimaattiin vain tiukat). Aikaperusta: kopioruudut tunnistetaan kuvaerosta molemmissa puoliskoissa, ja `aikaperusta.Aikajana` ajetaan kertyneelle radalle (sama kuin offline). Heiton tunnistus (kullekin päädylle, kumpi tahansa voi heittää): IDLE (haku hack-alueelta, kahva)
-> ARMED (kahva vakaana hack-alueella >= 0,5 s; lähtö vaatii todetun kiven kahvan kohdalla, ks. "Hukatut heitot koko videolla") -> LIIKKUU (kolmen viimeisen havainnon mediaani > 6 cm vanhempien havaintojen mediaanista) -> VAHVISTETTU (Y > -100 cm nopeudella >= 1,2 m/s mitatuista paikoista, >= 8 kivisovitusta) -> VALMIS (kivi poistuu kuva-alalta, Y ~ 735 cm).
Hylätty yritys palaa ARMED-tilaan: heittäjä käsittelee kiveä hackissa (kuljettaa, siirtelee, vastaliike ennen työntöä); lähtöruutu ajoitetaan taaksepäin (esim. lähtö 12,6 s vahvistuu 14,0 s, tulosteen piirto alkaa 12,6 s kohdalla). Viivaestimaatti: `liuku_irroitus.analysoi` osittaiselle radalle (uudet valinnat
`min_jalkeen` = 70 cm vapaata lentoa, `Y_r_min` = 200 cm aikaisin irroituspaikka, `kohinakatto` = (0,8; 0,15) cm; oletukset ennallaan, offline-tulokset eivät muutu).

**Punaisen kiven hukkuminen pesässä (korjattu).** Punaisella kahvalla seuranta pystyi hukkaamaan kiven pesän (punainen, valkoinen, sininen rengas) kohdalla, jolloin heitto päättyi kesken ja jäi ilman viivaestimaattia ("rata ei yltänyt irroitukseen"). Syy: heittäjä kuljettaa kiveä kädellään irroitukseen asti (Y ~ 250-500 cm), ja punaisen kahvan värimaskiin tarttuu käsi / hanska / hiha, joka on kahvan väristä samaa sävyä (HSV: kahva H 170-175, S 105-200, V 78-154; kahvan vieressä olevat punaiset pikselit H 170-175, S 102-173, V 67-150), joten väri ei erota niitä. Blobin ala paisuu 2-4-kertaiseksi (maatasolle normitettuna kahva ~220 cm2, pesässä jopa 800 cm2) ja sen keskipiste viivästyy kymmeniä senttimetrejä (A ~15 cm kiven takana). Tästä seurasi ketju: B-sovitus löysi kiven oikein mutta hylättiin, koska se oli > 15 cm:n päässä A-pohjaisesta alkuarvosta; A-pohjainen ennuste vääristyi; seuraavassa ruudussa lähin blobi oli käsi; kivi hukattiin eikä sitä löydetty uudelleen; 24 ruudun kuluttua heitto päätettiin. Keltaisella kahvalla ongelmaa ei ole (käsi ei ole keltainen). Toistettavissa pakkaamalla videon uudelleen heikommalle bittinopeudelle (`ffmpeg -i netti.mp4 -an -c:v libx264 -b:v 4000k -maxrate 6000k -bufsize 8000k -pix_fmt yuv420p pieni.mp4`): alkuperäisellä videolla punainen heitto 2 seurattiin loppuun, 4 Mbps:n kopiolla heitto päättyi 4 s lähdön jälkeen (ruutu 345 / 434) ja 2,5 Mbps:n kopiolla syntyi 9 ruudun aukko. Korjaukset (`netti_seuranta.py`, vain vahvistetun heiton seurantaan; heiton tunnistus ja lähtö ennallaan): (1) ennuste on B-pohjainen (`_ennuste_b`: viimeisten 8 B / pehmeän B -paikan robusti suora, ulkolaiset > 4 cm pois, vaatii >= 4 paikkaa joista uusin <= 16 ruutua vanha; A ei vaikuta), A-pohjainen vain varalla; (2) jos A poikkeaa B-ennusteesta > 7 cm, B:n alkuarvo on ennuste, ja B hyväksytään jos se on <= 15 cm alkuarvosta TAI ennusteesta; (3) A-blobeista valitaan ennusteeseen lähin, jonka ala sopii (0,4-1,8 x odotettu; ennen paisunut blobi saattoi nostaa odotusarvoa ja sitten hyväksyä itsensä): odotusarvoa päivittävät vain blobit 0,6-1,6 x mediaani, ja pysyvä poikkeama (6 ruutua) vaihtaa odotusarvon; (4) hukassa olevan kiven A-hakuportti kasvaa 1 cm per hukattu ruutu (enintään 15 cm); (5) heikko B (kelvollisia pikseleitä vain 2-8 %, mutta graniittia siluetin sisällä >= 0,5 ja <= 8 cm B-ennusteesta) otetaan pehmeäksi B:ksi: pesän väriset renkaat jättävät kelvollisia pikseleitä vain 2-20 %, ja ilman tätä joka neljäs-kolmas pesän ruutu oli pelkkää A:ta; (6) piirrettävät pisteet: pelkkä A-piste jätetään pois, jos B / pehmeä B on <= 4 ruudun päässä (aukko interpoloidaan), yksinäiset A-pisteet korjataan lähialueen B - A -erolla; (7) heitto päättyy vasta 36 ruudun paikattomuuden jälkeen (aiemmin 24). Tulos 4 Mbps:n kopiolla: heitto seurataan loppuun (ruutu 434, kuten alkuperäisellä), pisin paikaton aukko 4 ruutua; B / pehmeän B -paikat poikkeavat alkuperäisen videon vastaavista mediaanilta 0,08 cm.

**Hukatut heitot koko videolla (korjattu).** Käyttäjän reaaliaika-ajossa (25 min) tunnistettiin 23 heittoa 25:stä: pelissä jokaisessa päässä on 16 heittoa (8 keltaista ja 8 punaista vuorotellen), ja heitot 577 s ja 646,5 s jäivät huomaamatta. Koko alkuperäinen YouTube-virta (7625 s, 1,47 Mbps, 7 päätä = 112 heittoa) ajettiin simulaationa (`--nopea --tapahtumat`, C++), ja ajon rakenne tarkistettiin työkalulla `python tools/tarkista_heitot.py ajo.log` (värit vuorottelevat, 8 / väri / pää, kivinumerot 1..8). Ennen korjauksia vain heiton vahvistus (VAHVISTETTU) toimi, mutta heiton *tunnistus* (IDLE -> ARMED -> LIIKKUU) hukkasi heittoja, ja syyt olivat viisi eri vikaa:
1. *IDLE-kandidaattilista myrkyttyi.* Lista jatkui jokaisesta <= 8 cm:n siirtymästä, joten kun heittäjä kuljettaa kiven hackiin (liike on listan alussa), koko listan vakaus ei koskaan toteutunut ja paikallaan lepäävä kivi jäi virittämättä. Nyt vakaus tarkistetaan vain viimeisestä 0,65 s:sta (siirtymä <= 5 cm).
2. *"Ei kiveä" -hylkäys oli liian herkkä ja sen jälkeen haku oli 4 s tauolla.* Kahva hylättiin, kun viimeisen 3 s:n aikana ei ollut kahta kivisovitusta (käsi peittää kahvan), ja 4 s tauon aikana alkanut heitto jäi huomaamatta. Nyt kerran todettua kiveä ei pudoteta, hylkäys vaatii joko > 3 s ilman kiveä viritetystä tai kuusi peräkkäistä selvää ei-kivi-sovitusta (graniittia < 0,25), tauko on 1 s ja kahvan saa peittää 2 s (`HUKKA_LEPO_S`, aiemmin 1 s). Lähtö (`_lahto`) vaatii kiven todistuksen koko viritetyn jakson ajalta (ei vain viimeisen 3 s).
3. *Haamuviritys: väärän värin kahva kiven vieressä.* Heittäjän oranssi / punainen kenkä (pieni, 50-100 px) viritettiin "punaiseksi kahvaksi" keltaisen kiven vierestä; B-sovitus osui viereiseen kiveen (12-27 cm kenkää kauemmas) ja hyväksyttiin (raja oli 15 cm), joten haamu oli "kivi" eikä keltaista kahvaa enää haettu: oikean pään heitot Y7 (~3954 s) ja Y8 (~4156 s) jäivät huomaamatta. Nyt (a) levossa B kelpaa kiveksi vain <= 12 cm kahvan paikasta (tosikivillä mediaani 1-5 cm, max 10,4 cm), (b) viritys on *validoitu* vasta kun kahvan kohdalla on vähintään 2 kivisovitusta eikä kahvan vierestä ole löytynyt graniittia (peitto >= 0,5, > 12 cm) useammin kuin kahvan kohdalta, ja heiton lähtö (LIIKKUU) vaatii validoinnin, (c) validoimaton viritys (jolla ei ole kivitodisteita, tai todisteet ovat vierestä) korvataan toisella vakaalla kandidaatilla (toinen väri tai paikka >= 12 cm) 1 s virityksen jälkeen.
4. *Kivi hukkui heti lähdön jälkeen.* Punaisen kiven heitto 6820 s: kivi kiihtyy työnnössä 0,7 -> 2,4 m/s alle 0,2 s:ssa, käsi peittää kahvan (A puuttuu) ja ennuste (viimeisten paikkojen suora, jossa on vastaliike) jäi jälkeen 16-25 cm, joten B-sovitukset (graniittia 0,8-0,9) hylättiin 15 cm:n rajan takia ja heitto hylättiin 15 paikattoman ruudun jälkeen. Nyt ennen vahvistusta B kelpaa myös, jos se on edellisen hyväksytyn B:n ulottuvilla (15 cm + 8 cm / ruutu, enintään 60 cm; graniittia >= 0,5).
5. *Diagnostiikka.* `--tapahtumat` kertoo nyt myös heiton päättymissyyn (`valmis ... viimeinen Y 738 cm, syy: matka` tai `syy: kivi hukassa`) ja virityksen vaihdon (`vaihto punainen (1,-343) -> keltainen (-11,-328)`).
Tulos koko videolla (7625 s, simulaatio `--nopea`, C++, 4 ydintä 42-46 r/s): **112 / 112 heittoa** (ennen korjauksia 109: heitot Y7 ja Y8 päässä 4 sekä R3 päässä 7 puuttuivat, ja kivinumerot siirtyivät sen jälkeen), jokaisessa 7 päässä 16 heittoa, värit vuorottelevat, kivinumerot 1..8 (`tools/tarkista_heitot.py`: 0 huomautusta), kaikille heitoille viivaestimaatti ja jokainen heitto seurattiin kuvan reunaan asti (`syy: matka` 112 kertaa, `kivi hukassa` 0 kertaa). Virityksen vaihto (`vaihto`) tapahtui 64 kertaa 125 min:ssä, lähinnä pelin kauempana päässä olevien punaisten esineiden (varusteet) välillä; validoitu kivi ei vaihdu. Muiden heittojen tulokset eivät muuttuneet: 109 vastaavan heiton mediaaniero irroituspaikassa, irroituksen suunnassa, liun suunnassa, nopeudessa, hog-hog-ajassa ja kierteessä on 0,000 ja irroituspaikkojen jakauma on sama (mediaani 429 -> 432 cm; irroituspaikka on estimaatin heikoin suure ja hajoaa vähän aineiston muuttuessa jopa kymmeniä senttejä: p90-ero 20 cm). Reaaliaika-ajo (tiedosto tahdistettuna, piirto ja tallennus päällä, 3 työprosessia) löysi kolme aiemmin hukattua heittoa samoista kohdista, 0 ohitettua ruutua ja 0 virhettä.

**Kaukopään hog-ylitys ja ihmisten varusteet (korjattu).** Kaukopään haku etsii kahvan värisiä blobeja ilman ennustetta, ja sweeperien (raidallinen paita) punainen käsivarsinauha on punaisen kahvan väristä samaa sävyä, samankokoinen ja kulkee samaa vauhtia pesää kohti: se ylittää hog-viivan noin 1,5 s ennen kiveä (sweeperit kulkevat kiven edellä). Tässä videossa heitolla 2 (punainen) ensimmäinen hyväksytty rata (22 pistettä, X' = 38-42 cm, ylitys 76,7 s) oli käsivarsinauha, vaikka kivi ylitti viivan vasta 78,5 s (tarkistettu kuvista: kivi tulee viivalle kuvan alareunasta, sweeperit sen ympärillä): hog-hog-aika tuli 12,7 s oikean 14,5 s sijaan, ja kierrosmäärä noin -2,9 oikean noin -3,2...-3,3 sijaan (noin 10 % liian pieni; käyttäjän reaaliaika-ajossa sama heitto antoi oikean 14,5 s, joten virhe riippui siitä, miten blobiradat sattuivat muodostumaan). Korjaus (`Saapuminen._kivi_kelpaa`): ennen hyväksyntää radan viimeinen piste tarkistetaan B-sovituksella (sama kivisovitus kuin heittopäässä, 12 ms C++ / 75 ms Python, vain radalle joka muuten kelpaisi): kivellä on graniittirunko, ja B-sovituksen peitto (graniittia siluetin sisällä) on kivellä 0,34-0,71 ja käsivarsinauhalla 0,06-0,28 (rata 0,09-0,12 joka pisteessä). Rata kelpaa, kun yksi tarkistus >= 0,40 tai kaksi >= 0,30; kolmen epäonnistuneen tarkistuksen jälkeen rata merkitään hylätyksi (seurataan loppuun, ei tarkisteta uudelleen) ja haku jatkuu. Tulos: heitto 1 (keltainen) peitto 0,62 -> 19,8 s ennallaan, heitto 2 (punainen) nauharata 0,11 / 0,10 / 0,11 -> hylätty, kivirata peitto 0,65 -> ylitys 78,46 s, hog-hog 14,5 s, heitto 3 (keltainen) 0,74 -> 10,6 s ennallaan. Jos tarkistus hylkää kaikki radat, hog-hog-aika arvioidaan mallilla ("alustava").

**Taustakuva (`netti_tausta.py`).** Vanha moodikuva (koko videon mediaani) sisältää hackissa haamukiven ja heittäjän jalat (kivi lepää hackissa 0-8, 23-47 ja 83-101 s), jolloin lepäävän kiven alfa-kartta ja tarkka sovitus epäonnistuvat juuri siinä, missä heitto alkaa (peitto 0,09 vs 0,86 puhtaalla taustalla).
`python netti_tausta.py VIDEO --tyhja-vasen 18.5-22.5,66.5-76` korvaa hackin alueen niiden hetkien mediaanilla, jolloin hack on tyhjä -> `tulokset/moodikuva_vasen_puhdas.png` (live-koodi käyttää sitä jos se on; offline-skriptit käyttävät edelleen `moodikuva_<puoli>.png`:ää). Toinen puolisko (oikea): tässä videossa ihmisiä koko ajan, joten
korjausta ei ole. Toisella videolla / valaistuksella tausta ja kalibrointi on tehtävä uudelleen.

**Tietopaneeli (`netti_piirto.py`, luvut `netti_seuranta.heiton_suureet`).** Kahdeksan riviä ylälaidassa; luvut päivittyvät sitä mukaa kuin mittaukset tarkentuvat (otsikon perässä "alustava", kunnes irroitus on vakaa):

| rivi | sisältö |
|---|---|
| `Kivi: Punainen 1` | kiven väri ja numero. Kivet numeroidaan väreittäin heittosuunnan sisällä siinä järjestyksessä kuin ne heitetään (oletus: kivet heitetään numerojärjestyksessä); numerointi alkaa alusta, kun heittosuunta vaihtuu (uusi pää) tai kun yksi väri on heitetty 8 kertaa |
| `Liuku: -7cm` | liukulinjan (hackista irroituspisteeseen) jatke osuu kaukopään T-viivalla näin kauas keskiviivasta; + = heittäjän oikealle, - = vasemmalle |
| `Heitto: +5cm` | irroituslinjan (vapaan lennon suunta irroitushetkellä) jatke osuu kaukopään T-viivalla näin kauas keskiviivasta (sama etumerkki) |
| `Irroitus: +12cm` | (Heitto - Liuku) kierteen suunnan mukaan: vastapäivään pyörivällä kivellä (kaartaa vasemmalle) etumerkki on sama kuin erotuksella, myötäpäivään pyörivällä (kaartaa oikealle) päinvastainen; + = irroitus on kaarron vastaiseen suuntaan. `?` kunnes kierteen suunta tiedetään |
| `Kierre: -3,2` | arvio kiven kierrosmäärästä hog-hog-välillä; - = vastapäivään, + = myötäpäivään (ks. alla); pieni harmaa teksti: "±0,2" = varma (heittopään ja kaukopään mittaus yhdistetty, epävarmuus 1 sigma kierroksina), "alustava" = kaukopään mittaus kesken tai hog-hog-aika arvioitu mallilla, "mitataan" = kierteen mittaus kesken, "±0,5 epävarma" = mittaukset ristiriidassa, vain yksi epävarma mittaus tai pyörimisnopeus epäuskottavan suuri (> 130 astetta/s eli yli noin 4,5 kierrosta, tavallisesti 40-110) |
| `Irroitus: 1,5m` | kuinka monta metriä ennen heittopään hog-viivaa (sisäreuna 640 cm T-viivasta) irroitus tapahtui kiven etureunasta mitattuna (irroituspaikka + kiven säde 14,55 cm); negatiivinen = irroitus hog-viivan jälkeen |
| `Nopeus: Ennen 1,92 m/s Jälkeen 2,01 m/s` | mitattu nopeus ennen irroitusta (liuku) ja heti sen jälkeen (vapaa lento) |
| `Hidastuvuus: 0,07 m/s^2` | vapaan lennon hidastuvuus irroituksen jälkeen (positiivinen = kivi hidastuu; mittausjakso on vain ~2 s, joten arvo on suuntaa antava, +-0,03 m/s2); `?`, kun vapaata lentoa on liian vähän tai mittaus on kohinaa (negatiivinen eli kiihtyvä tai yli 0,25 m/s2) |

Luvut ovat desimaalipilkullisia; tekstit piirretään OpenCV:n Hershey-fontilla, joka ei sisällä ä/ö-kirjaimia, joten ne piirretään perusmerkistä ja pisteistä (`teksti_fi`). CSV (`--csv`) ja konsoli sisältävät samat luvut (+ omega, hog-hog-aika, liun hidastuvuus).

**Kierre ja kierrosmäärä hog-hog-välillä (`netti_kierre.py`, `netti_seuranta.py`).** Kolme mittausta:
1. *Kierron nopeus ja suunta* (heittopää, irroituksen jälkeinen vapaa lento ~1,3 s): sama mittaus kuin offline-versiossa (`kierto_netti.py`, ks. yllä): kahvan palkin malli sovitetaan muovin todennäköisyyskarttaan, omega = (yhteissovituksen ja ruutukohtaisten kulmien Theil-Senin) keskiarvo. Erot offline-versioon: palkin muoto (h, ell, w) on kiinteä (`tulokset/kierto_muoto_<puoli>.json` = offline-yhteissovituksen tulos väreittäin;
   `kierto_netti.py` kirjoittaa tiedoston), jolloin heittokohtaisia parametreja on vain kolme (theta0, omega, kirkkaus) ja sovitus kestää noin 10 s taustaprosessissa heiton päätyttyä; ruudut kerätään seurannan aikana leikkeinä (160 kpl enintään) ja kiven akseli haetaan jokaiseen ruutuun samalla kahvamallin sovituksella kuin offline (`kahva_malli.sovita`, alkuarvo B-sovituksesta; B:n ja kahvamallin akselit eroavat
   systemaattisesti ~2 cm, ja palkin malli on kalibroitu kahvamallin akselin suhteen); suunta valitaan kahden suunnan (omega > 0 / < 0) parhaiden sovitusten kustannuksesta (`varma` = ero >= 3 % ja molemmat estimaattorit samaa mieltä, vähintään 12 ruutua ja |omega| <= 130 astetta/s); jos kuvamittaus on epävarma ja vapaan lennon kaarevuus on selvä (>= 4e-5 /cm; kaarevuus < 0 = kaartaa vasemmalle = vastapäivään), kaarto ratkaisee suunnan; ennen kuin kuvamittaus on valmis, selvä kaarto (>= 2,5e-5 /cm) antaa suunnan vihjeeksi.
2. *Hog-hog-aika videon puoliskoista:* heittopään hog-ylitys luetaan seurannasta (kiven akseli ylittää hog-viivan keskiviivan, Y = 646,35 cm, lineaarinen sovitus ruutunumero(Y) viivan ympäriltä) ja kaukopään hog-ylitys toisen puoliskon kuvasta: heiton päätyttyä kaukopään hog-viivan ympäriltä (|X'| <= 3 m, Y' = 5,6...8,0 m; kivi ilmestyy kuvan alareunasta ~0,9 m ennen viivaa) haetaan kiven kahvan värisiä blobeja (taustasta poikkeavia, ala heittopään kahvan alan mukainen),
   ne yhdistetään ratoihin ja rata, jonka pisteet etenevät tasaisesti kohti pesää (0,4-4,5 m/s, RMS < 6 cm, ei sivuttaisliikettä) hyväksytään; ylitysruutu on suoran Y'(ruutu) leikkaus hog-viivan keskiviivalla. Aika = ruutunumero / ruutunopeus: molemmat puoliskot ovat samassa videoruudussa, joten puoliskojen oma vaihe-ero jää tämän tarkkuuden (~0,1 s) alle; ylityskohta on molemmissa päissä viivan keskiviiva (hog-hog-matka 21,8 m).
   Tässä videossa kaukopään ylitys löytyi kaikille kolmelle heitolle (hog-hog 19,8 / 14,5 / 10,6 s; heitolla 2 ensimmäinen löydetty rata oli sweeperin käsivarsinauha ja se hylätään, ks. "Kaukopään hog-ylitys ja ihmisten varusteet").
3. *Kierrosmäärä:* kierre vaimenee tasaisesti (IMU-mittauksissa n. 2 % / s, ks. `Testi_09_03/README.md`: 2,1 %/s, vaihteluväli 2-3 %/s): omega(t) = omega_mittaus exp(-k (t - t_mittaus)), k = 0,021 /s. Kierrokset aikavälillä [t1, t2] = omega_mittaus / (2 pi k) (exp(-k (t1 - t_mittaus)) - exp(-k (t2 - t_mittaus))), jossa t_mittaus on kierron
   mittausikkunan keskihetki (~1 s irroituksen jälkeen), t1 = heittopään hog-ylitys, t2 = kaukopään hog-ylitys. Tulos paneeliin etumerkillä (- = vastapäivään). Jos kaukopään ylitystä ei löydy (tai kivi ei yllä sinne), hog-hog-aika arvioidaan vakiohidastuvuuden mallilla (irroituksen nopeus ja hidastuvuus; mitattu vapaan lennon hidastuvuus on kohinainen, joten se sekoitetaan tyypilliseen arvoon 0,068 m/s2 vapaan lennon pituuden mukaan, ja jos kivi ei sillä yllä kaukopäähän, kokeillaan tyypillistä arvoa sellaisenaan) ja luku on "alustava"; jos kivi ei mallin mukaan yllä kaukopään hog-viivalle, kierrettä ei näytetä.
   Tarkkuus: omega +-10 % (toistoajot 52-57 astetta/s heitolla 1, offline 54; 62, kun analyysi ohitti ruutuja ylikuormituksessa), vaimenemisvakion epävarmuus (k = 1,6-2,6 %/s) +-12 % 20 s:n välillä, hog-hog-aika +-0,2 s (1 %) -> kierrosmäärä noin +-15 % (+-0,3-0,4 kierrosta); kierrosmäärän systemaattinen virhe tulee lähinnä vaimenemisvakiosta, ei ajasta.

| | heitto 1 | heitto 2 | heitto 3 |
|---|---|---|---|
| kivi | Keltainen 1 | Punainen 1 | Keltainen 2 |
| omega (mittaus, vapaa lento) | +54 astetta/s (vastapäivään; offline 54) | +93 (offline 99) | +78 (offline 70) |
| hog-hog-aika (heittopään ja kaukopään hog-ylitys videosta) | 19,8 s (18,1 s -> 37,9 s) | 14,5 s (64,0 -> 78,5 s) | 10,6 s (111,5 -> 122,1 s) |
| kierrosmäärä hog-hog | **-2,4** | **-3,2** | **-2,0** |
| liuku / heitto / irroitus (kaukopään T-viiva) | +273 / +317 / +44 cm | -6 / +39 / +45 cm | -53 / -28 / +25 cm |
| irroitus ennen hog-viivaa (etureuna) | 2,5 m | 3,5 m | 2,7 m |
| nopeus ennen / jälkeen, hidastuvuus | 1,87 / 1,87 m/s, 0,07 m/s2 | 2,06 / 2,12 m/s, 0,07 | 2,44 / 2,51 m/s, 0,08 |

(Heiton 1 kivi on hidas: kaukopään hog-viivalla nopeus on vain ~0,45 m/s ja viivan ylitys 19,8 s irroituksen jälkeen; visuaalinen tarkistus kaukopään kuvasta: ylitys +-0,3 s.)

**Kaukopään seuranta, kierteen varmuus, kiven vakiot, joukkueiden kello ja CSV (uusi).** Pyyntö: tallenna heittojen tiedot CSV:hen; tutki, miten kierteen määrä saadaan varmemmin (kiville voi olettaa vakiohidastuvuuden); tunnista kivi toisessa päässä, kun se tulee takaisin kuvaan, ja seuraa sitä siellä; laske sen avulla kivelle ne vakiot, joita kulkuradan ennusteessa (`rata_malli.py`) käytetään; bonuksena kellota joukkueiden miettimisaika. Ketju (kaikki `netti_seuranta.py`, `netti_vakiot.py`, `netti_kierre.py`; kaikki tieto jo valmiina: kivi on heittopäässä tunnistettu väriltään ja numeroltaan, ja kaukopäässä lepäävien kivien paikat ennen heittoa tunnetaan):

1. *Kivi kaukopään kuvassa (`Saapuminen`, `KaukoSeuranta`).* Kun heitto on päättynyt heittopäässä, kaukopään kuvasta haetaan saman värinen kivi (kahvan väriblobi + graniittirungon B-sovitus todistaa, että kyseessä on kivi eikä sweeperin punainen käsivarsinauha) alueelta, jonne kivi hog-viivan ylitysaikana (heittopään hog-ylitys + arvioitu hog-hog-aika) saapuu. *Ennen heittoa lepäävien saman värisen kivien paikat on kartoitettu* (`Saapuminen._kartoita`), joten ne eivät sekoitu saapuvaan kiveen. Löydettyä kiveä seurataan kuten heittopäässä (B-pohjainen ennuste, A (kahvan väri) ennusteen ympäriltä, B kun kivi liikkuu; hidastuva ennuste kantaa piilossaolon yli, koska sweeperit ovat kiven vieressä ja peittävät sen) pysähtymiseen asti. Tulos `h.kauko`: kiven rata (aika, X', Y'), hog-viivan ylitys B-radasta (tarkentaa hog-hog-aikaa), nopeus hogilla, pysähtymishetki ja -paikka, törmäys (nopeus putoaa jyrkästi), tila (`pysahtyi` / `hukassa`). Kiven rata katkaistaan, jos pysähtymään hidastunut "kivi" alkaa liikkua (seurattu kohde vaihtui: esim. pesän taakse mennyttä kiveä kuljetetaan käsin pois). Hukatun kiven pysähtymishetki arvioidaan viimeisestä B-paikasta (hidastuvuus 0,1 m/s2; pesän takana taka-aita pysäyttää).
2. *Aikaperustan korjaus kaukopäähän.* `aikaperusta.Aikajana` päätteli hitaassa vaiheessa (alle 75 cm/s: siirtymä alle 3 cm / ruutu) lähderuutujen lukumäärän ruutunumeroiden hilasta, mikä laski ohitetun kopioruudun (ruutuero 2) kahdeksi lähderuuduksi, kun näytteissä ei enää ole kopioruutuja (kaukopään hidas rata): aika venyi 16 % ja kierron nopeus näytti 14 % hitaammalta. Uusi valinta `kopiot_ohitettu=True` (vain kaukopää; heittopään käyttö ennallaan): ruutuero 1 tai 2 = yksi lähderuutu. Korjaus näkyy suoraan mitatussa kierron vaimenemisessa: heitoilla, joilla molemmat päät mitattiin varmasti, vaimeneminen on 2,1 % / s (26 heittoa: mediaani 2,0 %, hajonta 1,1 %; aiemmin ~3 % / s) - sama kuin IMU-mittauksissa (2,1 % / s).
3. *Kierteen varmuus (`netti_kierre.py`).* (a) *Kaukopään kierron mittaus*: sama palkkimalli kuin heittopäässä (palkin muoto kiinteä, vain theta0, omega, kirkkaus), kaukopään kameralla kiven seurannan aikana kerätyistä leikkeistä (ruutuja 10-100, käsi / hanska hylätään). (b) *Kahden pään yhdistäminen* (`yhdista`): omega vaimenee eksponentiaalisesti, ln omega(t) = a - k t; molemmat mittaukset (omega +- epävarmuus) ja ennakkotieto k = 2,1 +- 0,8 % / s sovitetaan yhdessä (painotettu pienimmän neliösumman sovitus), ja kierrosmäärä hog-hog-välillä (sekä irroituksesta pysähtymiseen) saadaan integraalina; mitattu vaimeneminen korjaa arvion, ja epävarmuus (N_sigma) lasketaan kovarianssista. Kierre on "varma", kun epävarmuus <= 0,25 kierrosta, mittaukset eivät ole ristiriidassa (etumerkit samat, 0 <= k <= 7 % / s) ja ainakin yksi mittaus on varma. (c) *Alias-minimit korjattu*: kaukopään leikkeet ovat ryhmissä (käsi ja sweeperit peittävät kiven), jolloin omegan kustannusprofiilissa on lähes yhtä hyviä minimejä ~40 astetta/s:n välein, eikä ruudukon paras minimi riitä (heitto 75: -104 astetta/s, vaikka oikea minimi on -65 astetta/s; heittopää -91, vaimeneminen 1,7 % / s). Nyt jokaisen kiertosuunnan useat omega-profiilin minimit jalostetaan (`ALKU_MAX` = 3) ja paras valitaan; toisen pään varma mittaus vaimeneminen huomioiden (ennakko, +-30 %) ratkaisee vain, kun minimien kustannukset ovat lähes samat. (d) *Vakiohidastuvuus*: kun kaukopään hog-ylitystä ei löydy, hog-hog-aika arvioidaan vakiohidastuvuuden mallilla (`hog_hog_aika_malli`, tyypillinen 0,068 m/s2, mitatun vapaan lennon hidastuvuuden kanssa sekoitettuna); lähde näkyy (`mitattu` / `malli`).
4. *Kiven vakiot (`netti_vakiot.py`).* `rata_malli.py`:n kytketty naarmu- ja kitkamalli (kitkataso `A`: mu(U) = (A + B ln Ub) (U / Ub)^NL; kaarto `kappa`: naarmuvoiman voimakkuus) sovitetaan kaukopäässä seurattuun rataan hog-viivalta pysähtymiseen (Levenberg-Marquardt; parametrit y0, v0, ln omega0, A, ln kappa; X'(t) = c0 + c1 S(t) + s H(t) kuten `rata_malli.ennusta`): A määräytyy pysähtymisprofiilista Y'(t), kappa sivusiirtymästä X'(t), kun kierre on rajattu kaukopään kierron mittauksella (varma; muuten yhdistetyn kierteen omega hogilla). Rata katkaistaan pysähtymiseen, törmäyksen alkuun tai viimeiseen B-paikkaan; sovitus hylätään (`ok = False`), jos rata on alle 3 s tai 150 cm, jäännös on Y 6 cm / X 3 cm yli tai A on parametrirajalla; kappa raportoidaan vain, kun kierre on rajattu mittauksella (`kappa_ok`). Vakiot ovat tehollisia: harjaus pienentää kitkaa ja kaarroa, eikä harjaamattomia heittoja tunnisteta. Kivi yksilöidään parilla (väri, numero heittopäässä); `Analyysi.kivien_vakiot()` ja tiedosto `<csv>_kivet.csv` antavat kivikohtaiset mediaanit ja hajonnan.
5. *Joukkueiden kello (`Analyysi.kello`, `kello_nyt`, `netti_piirto.kello_rivit`).* Kunkin heiton miettimisaika = ruutu, jolla heitetyn kiven etureuna osuu heittopään T-viivaan (kello ei pysähdy vielä kiven lähtiessä hackista, vaan vasta T-viivalla; ks. alla) - edellisen heiton kiven pysähtymisruutu (kaukopään seuranta; pelistä poistunut kivi, eli keskipiste kokonaan taka- tai sivuviivan ulkopuolella, pysäyttää odotuksen jo poistuessaan, koska se ei enää vaikuta peliin (15 heittoa; 7 hukatulle kivelle ekstrapoloituna); törmäysheitolla viimeistään 4 s törmäyksen jälkeen, koska törmäyksessä liikkeelle lähteneet kivet pysähtyvät arviolta tässä ajassa; hukatulla kivellä arvio viimeisestä B-paikasta; ilman seurantaa tai kun seuranta ei kerro pysähtymistä: **mallin mukainen pysähtymishetki** (nopeus lähipään hog-viivalla suoraan radasta -> 22 m hog-viivalta hog-viivalle vakiohidastuvuudella 0,068 m/s2 -> kaukopään hog-viivalta pysähtymiseen nopeus / 0,10 m/s2, enintään 12 s) oletetaan toteutuneeksi, kun siitä on kulunut 5 s analyysiajassa eikä kiveä ole nähty liikkeessa; ks. "Kello ei lähde käyntiin" alla); kello ei käy päiden välillä (pään ensimmäisellä heitolla ei aikaa; heittopuolen vaihtuessa alkaa uusi pää) eikä kun pään kaikki 16 heittoa on heitetty. Joukkueen aika yhteensä = miettimisajat väreittäin. Kuvassa (`Kello`-laatikko kuvan keskellä mainosgrafiikan (Hyvinkään Liikenne) alla, y = 200 px, vaakasuunnassa keskitetty; oma paikka `--kello-paikka X,Y`, pois `--ei-kelloa`): molempien joukkueiden aika yhteensä (kävevän joukkueen kuluva aika mukana), kävevä joukkue ja heittokohtainen aika (kuluva aika, kun kello käy; muuten viimeisimmän heiton aika). Konsoliin tulee jokaisesta heitosta rivi `heitto N: Punainen 3 miettimisaika 0:17,8 (kello pysahtyi, kun kivi osui T-viivaan 289,6 s; joukkueen punainen aika yhteensa 0:32,5)` ja ajon lopuksi yhteenveto.
   *T-viivan ylitys (`tviiva_ruutu`, `Heitto.tviiva_f`).* Kello pysähtyy, kun heitetyn kiven etureuna osuu heittopään T-viivaan (Y = 0 heittopään maakoordinaatistossa; kiven keskipiste Y = -14,55 cm), kuten pelissä: hackista työntö kestää, joten kivi on T-viivalla vasta mediaanilta 3,1 s (1,4-8,8 s) lähdön (`alku_s`) jälkeen, nopeudella ~2,1 m/s. Ylitysruutu sovitetaan heittopään radasta (B-paikat, muuten A; toisen asteen sovitus +-110 cm viivan ympäriltä, karkeat pisteet karsitaan; ruutujen välinen tarkkuus alle 0,05 s; suoran ja toisen asteen sovituksen ero 0,003 s) ja se löytyi kaikille 112 heitolle; ellei rata ylitä viivaa, käytetään lähtöä + 3,1 s. Ylitysaika on CSV:ssä (`tviiva_s`), ja kuvassa kello käy edelleen, kun kivi on jo lähtenyt hackista, ja pysähtyy T-viivalla.
6. *CSV (`--csv`, oletuksena päällä).* Ilman valitsinta tulostiedosto on `heitot_<päiväys>_<aika>.csv` nykyisessä kansiossa (oma nimi `--csv polku`, pois `--ei-csv`) ja kivikohtaiset vakiot sen vieressä (`<nimi>_kivet.csv`); tiedosto kirjoitetaan uusiksi aina kun jonkin heiton tulos täydentyy. Yksi rivi per valmistunut heitto (myös ilman viivaestimaattia: viivasarakkeet tyhjiä). Sarakkeet: `heitto, vari, kivi_nro, puoli, alku_s, loppu_s` (heiton järjestys, väri, kiven numero väreittäin per pää, heittopuoli, lähtö- ja loppuhetki videossa); `irroitus_Y_cm ... vakaa` (liuku, irroitus, nopeudet kuten ennen); `kierre` (kierrokset hog-hog, - = vastapäivään), `kierre_mitattu`, `omega_deg_s` (heittopään mittaus), `hog_hog_s`; `paa, heitto_paassa`; `hog1_s, hog2_s, hog2_lahde`; `kierre_sigma, kierre_varma, kierre_lahde, kierre_vaimeneminen_1_s, omega_kauko_deg_s, kierre_koko_rata, kierre_koko_sigma` (kierteen varmuus ja kierrokset irroituksesta pysähtymiseen); `kauko_tila, pysahtyi_s, ulos_pelista_s, lepo_X_cm, lepo_Y_cm, lepo_napista_cm, nopeus_hog2_cm_s, tormays_s` (lepopaikka heittäjän koordinaatistossa: X + = heittäjän oikealle, Y = T-viivasta heittäjää kohti); `vakio_A, vakio_A_sd, vakio_kappa, vakio_kappa_sd, vakio_kappa_ok, vakio_rms_y_cm, vakio_rms_x_cm, vakio_omega_lahde`; `tviiva_s` (hetki, jolloin kiven etureuna osui heittopään T-viivaan: kello pysähtyy), `miettimisaika_s, joukkue_yht_s`; `nopeus_hog1_m_s, suunta_hog1_deg` (kiven nopeus ja suunta lähipään hog-viivalla suoraan radasta, irroituksen tunnistuksesta riippumaton) ja irroituksen uskottavuus (`irr_matka_ero_m`, `irr_A_matka`, `irr_sivu_ero_cm`, `irr_tarkistus` = ok / epailyttava; ks. "Irroituksen tunnistuksen fysiikka"). Yhteenveto: `python tools/yhteenveto.py heitot.csv [--kivet]` (kaukopään seurannan onnistuminen, kierteen varmuus, kiven vakiot kivikohtaisesti ja varianssianalyysi, kello).

**Tulokset koko videolla (7625 s, 112 heittoa, simulaatio `--nopea`, C++; `tools/yhteenveto.py`, `tools/sovita_vakiot.py`, `tools/testaa_vakiot.py`).**
* *Heittopään tulokset ennallaan:* liuku, irroitus, T-viivan osumat ja nopeudet ovat kaikille 112 heitolle täsmälleen samat kuin ennen (ero 0,000); muuttuneet ovat vain kierteen sarakkeet (kierteen mittaus ja alias-korjaus, ks. alla). Heittojen rakenne: 112 / 112, 7 päätä x 16 (`tools/tarkista_heitot.py`: 0 huomautusta).
* *Kivi toisessa päässä:* kivi löytyi kaukopään kuvasta 106 / 112 heitolle (95 %; ennen tätä työtä kaukopään haku onnistui 73 % ensimmäisistä 25 heitosta); seurattu pysähtymiseen asti 60 heitolla (54 %), 46 heitolla kivi hukkui kesken (nopea poisto, joka osuu muihin kiviin / poistuu pesän takaa, tai sweeperit peittävät hitaan kiven; pysähtymishetki arvioidaan silti viimeisestä B-paikasta). Hog-viivan ylitys kaukopäässä 100 / 112 heitolle (89 %), niistä 90 B-radasta mitattuna; törmäys havaittu 33 heitolla (29 %). Pysähtyneistä 38 / 60 lepää pesässä; lepopaikka heittäjän koordinaatistossa on CSV:ssä (`lepo_X_cm`, `lepo_Y_cm`, `lepo_napista_cm`).
* *Kierre:* kierrosmäärä saatiin 102 / 112 heitolle. Heittopään mittaus onnistui 93:lla, kaukopään 61:llä (kaukopään kuvassa sweeperien jalat, harjat ja kädet peittävät kahvan palkin sovitusalueen: 43 heitolla kaikki ruudut hylättiin "käsi"-ehdolla; kaikkien ruutujen käyttö ja löysempi raja eivät auta, koska peitto kestää koko radan, eivätkä löysällä rajalla saadut sovitukset ole itsenäisiä mittauksia: omega-profiilin minimit ovat lähes yhtä hyviä). Kun molemmat päät mitattiin (50 heittoa), kierrosmäärän epävarmuus (1 sigma) on mediaanilta **0,21** kierrosta (90 % alle 0,39; 29 / 50 "varma"); vain heittopään mittauksella 0,48 (90 % alle 1,73; 7 / 43 varma) ja vain kaukopään mittauksella 0,36. Heittopään mittaus ja vakiovaimeneminen (ekstrapolointi 15-20 s eteenpäin) poikkeaa kahden pään yhdistelmästä keskihajonnalla 0,63 kierrosta (robusti 0,33, suurin 1,70): näin paljon vanha menetelmä oli keskimäärin pielessä. Mitattu vaimeneminen on **2,1 % / s** (26 heittoa, joilla molemmat mittaukset olivat varmoja: keskiarvo 2,1 %, mediaani 2,0 %, hajonta 1,1 % / s, josta mittauskohinaa ~0,7; 10-90 % välillä 0,9-3,2 %), eli oletus 2,1 % / s (IMU) pitää; ennakkotiedon hajonta kavennettiin 0,8 % / s:iin. Hog-hog-aika: vakiohidastuvuuden malli (0,068 m/s2, mitatun vapaan lennon hidastuvuuden kanssa sekoitettuna) ennustaa mitatun hog-hog-ajan (73 heittoa, joilla molemmat hog-ylitykset mitattiin) mediaanivirheellä +0,28 s, keskihajonta 1,37 s (4 %), mikä on kierrosmäärässä ~0,2 kierrosta, kun kaukopään ylitystä ei löydy (12 heittoa). Kierrokset koko radalla irroituksesta pysähtymiseen (`kierre_koko_rata`; 55 heittoa): mediaanilta 1,5 x hog-hog-kierrokset.
* *Kiven vakiot:* A (kitkataso) saatiin 49 / 112 heitolle (44 %: kivi pysähtyi kuvassa tai rata riittävän pitkä, hog-viivan nopeus <= 1,6 m/s, jäännös Y <= 6 / X <= 3 cm, A:n keskivirhe <= 20 %) ja kappa (kaarto) 30 / 112 heitolle (27 %; kierre rajattu varmalla mittauksella, ln kappan virhe <= 0,4). **A = 0,0064 (hajonta 0,0011 eli 17 %, vaihteluväli 0,0041-0,0082)** vastaa hidastuvuutta ~0,06-0,07 m/s2 (sama kuin vapaan lennon 0,068); **kappa = 3,0 (hajonta 0,8 eli 27 %; 1,6-5,5)**. Kivikohtaiset mediaanit (16 kiveä: väri + numero) ovat tiedostossa `<csv>_kivet.csv`; varianssianalyysi (kivi = väri + numero) ei osoita kivien välistä eroa (A: F = 0,90, p = 0,56; kappa: F = 0,70, p = 0,69; kiven sisäinen hajonta A 0,0010, kappa 0,84), eli tässä pelissä heitto-kohtainen hajonta (sweeperit, kierteen mittausvirhe, jään vaihtelu) peittää mahdolliset kivien väliset erot, tai kivet ovat yhtä hyvät; mediaani kasvavalla heittomäärällä tarkentuu, ja `kivien_vakiot()` päivittyy ajon aikana. Sovituksen oikeellisuus: simuloidulla radalla (kohina 2 cm) `tools/testaa_vakiot.py` palauttaa A:n alle 1 %:n ja kappan 0-4 %:n virheellä ja pysähtymispaikan alle 10 cm:ssä. Kulkulinja kiven vakioilla: `netti_vakiot.ennusta(A, kappa, y0, v0, w0, kaarto_merkki)` (pysähtymispaikka ja polku, kuten `rata_malli.ennusta`).
* *Kello:* miettimisaika saatiin 105 heitolle (112 - 7 päätä ensimmäistä heittoa; kello pysähtyy T-viivalla): keltaisella keskimäärin 0:42 / heitto (mediaani 30 s, 90 % alle 1:23, ylin 2:04) ja punaisella 0:37 (mediaani 27 s, 90 % alle 1:08, ylin 1:54); aika kasvaa pään loppua kohti (pään 1 viimeiset heitot 0:55-1:08); joukkueet yhteensä **keltainen 36:14,7** (52 heittoa) ja **punainen 32:21,5** (53 heittoa). Kun kello pysähtyi vasta T-viivalla eikä kiven lähtiessä, kertyi aikaa 2:31 (keltainen) ja 3:03 (punainen) enemmän (~2,9 s / heitto, sama kuin T-viivan ylitys lähdön jälkeen). Pysähtymishetki on kaukopään seurannan tulos (yleensä +-1 s kuvasta tarkistettuna: heitto 12 (video 670 s): pysähtyi 681,1 s, silmämääräisesti 681,6 s) tai pelistä poistuminen; törmäysheitoilla ja hukatuilla kivillä arvio (ks. yllä), joka voi olla muutaman sekunnin pielessä.

**Rajoitukset (kaukopää, vakiot, kello).** Kaukopään kierron mittaus puuttuu ~40 %:lta heitoista (sweeperit peittävät kahvan; ei korjattavissa lisäruuduilla); näillä heitoilla kierrosmäärä on yhä vain heittopään mittauksen ekstrapolointi (epävarmuus ~0,5 kierrosta, merkitty paneelissa "±0,5 epävarma" ja CSV:ssä `kierre_sigma`). Kiven vakiot ovat tehollisia (harjaus pienentää kitkaa ja kaarroa; harjaamattomia heittoja ei tunnisteta), ja kivi yksilöidään parilla (väri, numero heittopäässä), mikä edellyttää, että joukkue heittää kivensä samassa järjestyksessä (tämä ei ole tarkistettavissa videosta). Kello: törmäyksessä liikkeelle lähteneiden kivien pysähtymishetki arvioidaan (törmäys + 4 s), koska vain heitetyn kiven rataa seurataan; ilman kaukopään seurantaa (6 heittoa) pysähtymishetki arvioidaan hog-ylityksestä, ja ensimmäiselle heitolle kustakin päästä ei lasketa aikaa. Ruutujen ohitus reaaliajassa: tuore kierteen / vakioiden laskenta (kaukopään kierto ~12 s, heittopään ~6 s, vakiot ~2 s heittoa kohti) ajetaan yhdessä kierto-prosessissa heiton jälkeen; reaaliaika-ajossa (tiedosto tahdistettuna, 345 s, 8 heittoa) 0 ohitettua ruutua ja 0 virhettä.

**Irroituksen tunnistuksen fysiikka (korjattu: nopeudet ja suunnat).** Käyttäjä: välillä irroituksen suunnat ja nopeudet eivät näytä oikeilta; harjaus muuttaa matkaa enintään ~2 m ja suuntaa ~40 cm, joten jos kivi kulkee mallin mukaan selvästi muualle (tai kiven vakiot poikkeavat muista), irroituksen tunnistuksessa on todennäköisesti virhe. Koko videon heittopään radoista (112 heittoa, tarkka syöte `--radat`-tiedostossa avaimella `viivat_syote`):
* *Vika.* Entinen tunnistus sovitti kaksi paraabelia (ohjattu vaihe, vapaa lento) ja valitsi vaihtokohdan, jossa sovitus paranee eniten. Vapaan lennon paraabeli sai kiihtyä ja kaartua mielivaltaisesti, ja radassa on usein useita vaihtokohtia (työnnön loppu, liukuvaihe, irroitus): **vapaan lennon "hidastuvuus" oli negatiivinen (kivi kiihtyisi irroituksen jälkeen, mahdotonta) 46 / 112 heitolla (41 %)**; 31 / 112 heitolla (28 %; etenkin keltaisen joukkueen kivet 4-8, pari muutakin heittäjää) heittäjä työntää kiveä vielä irroituksen alla (nopeus irroituksen jälkeen 0,12 m/s tai enemmän pienempi kuin hog-viivalla): nopeus kasvaa 1,7 -> 2,0 m/s, kun käsi on yhä kahvalla (kontaktiarkki, heitto 13: sormet kahvalla Y = 399-563 cm, irtoaminen ~Y = 590 cm, mutta tunnistettu irroitus oli Y = 440 cm). Seuraus: irroituksen jälkeinen nopeus aliarvioitu 0,1-0,3 m/s (jopa 12 %), suunta jopa ~1 aste pielessä (ohjattua rataa mukana), ja irroituskohta hyppäsi 100-290 cm, kun syötteestä jätti 3 % paikoista pois (kinkkihaku on epävakaa: sama rata, eri vaihtokohta). Hukkuneen vauhdin seurauksena myös kellon pysähtymisarvio (nopeus -> pysähtyminen) oli väärä.
* *Käyttäjän kriteeri mitattuna.* Mallilla (`rata_malli`, irroituksen nopeudesta pysähtymiseen, kiven kitkataso A_matka = se kitkataso, jolla malli osuu kaukopäässä havaittuun pysähtymismatkaan; 35 heittoa, kivi pysähtyi eikä törmännyt): vanhoilla irroitusarvoilla A_matka hajosi **14,8 %** (suurin poikkeama 39 %, **12 / 35** heiton matka-ero yli 2 m vertailukitkaan nähden, mediaani +0,8 m, hajonta 2,1 m), uusilla **5,4 %** (**2 / 35** yli 2 m, mediaani -0,02 m, hajonta **1,1 m**). Eli kun irroituksen nopeus on oikea, harjauksen ja jään vaihtelu pysyy käyttäjän arvioiman ~2 m:n sisällä; vanhojen arvojen "kivi kulkee huomattavasti kauemmas kuin malli" -poikkeamat olivat juuri myöhäisen työnnön heittoja.
* *Korjaus (`liuku_irroitus.irroitus_hanta`).* Irroitus = **varhaisin kohta, josta alkaen rata on vapaan lennon mukainen**: ikkuna alkaa hog-viivan jälkeisestä radasta (irroitus on säännön mukaan ennen viivaa: kiven keskipiste Y <= 631,8 cm, joten sen jälkeinen rata on varmasti vapaata lentoa) ja kasvaa takaperin; kutakin ikkunaa vasten sovitetaan vapaan lennon malli (Y(t): vakiohidastuvuus, X(Y): kaartuva suora; **fysiikka pehmeinä esitietoina**: hidastuvuus 0,065 +- 0,03 m/s2, kaarevuus 0 +- 5e-5 1/cm) ja ikkunaa edeltävien 6 paikan standardoitujen ennustejäännösten |z|-mediaani tarkistetaan (Y- ja X-kanavan suurempi). Irroitus on kohdassa, jossa raja 3 ylittyy kolmessa peräkkäisessä ikkunassa. Ohjatun vaiheen mallia ei tarvita, ja tulos ei riipu monesta paikallisesta vaihtokohdasta: myöhäinen työntö tai sivuttaisliike katkaisee ikkunan. Ellei rata yllä hog-viivan taakse (osittainen aineisto, esim. reaaliaikainen alustava estimaatti) tai ankkuri-ikkunassa on alle 12 paikkaa, käytetään entistä kinkkihakua (`menetelma="kinkki"`), jossa vapaan lennon esitieto on myös mukana ja irroitus ei voi olla hog-viivan jälkeen. Tulos (sama 112 heiton aineisto): vapaan lennon hidastuvuus < 0 vain 2 / 112 (ennen 46 / 112); nopeus irroituksen jälkeen miinus nopeus hog-viivalla (suora mittaus viivan ympäriltä): mediaani **+0,070 m/s** (odotus +0,05...+0,15: kivi vain hidastuu), hajonta **0,024** (ennen 0,106), alle -0,05 m/s vain **2 / 112** (ennen 38 / 109); irroitus Y mediaani 451 cm (ennen 436), hajonta 289-551 cm (p10-p90); myöhäisen työnnön 31 heittoa (esim. 13, 15, 44, 71, 28) saavat irroituksen Y = 491-561 cm (mediaani 542, ennen 440) ja nopeuden mediaani 2,04 m/s (ennen 1,80; esim. heitto 13: 2,02 m/s vs 1,70, hog-viivalla mitattu 1,94). Kuvalla tarkistettu: heitto 13 sormet pois kahvalta Y ~ 590 cm (tunnistus 547 cm), heitto 37 käsi kahvalla vielä Y = 450 cm (tunnistus 450 cm; vanha 206 cm). Syntetiikkatesti (`tools/testaa_vakiot.py`, myöhäinen työntö): tosi irroitus Y = 532 cm / 2,03 m/s -> uusi 507 cm / 2,04 m/s, vanha 381 cm / 1,52 m/s (hidastuvuus -0,34).
* *Rajoitus: sileä rata.* Jos heittäjä kantaa kiven tasaisesti ja irroitus ei näy radassa (heitto 37 aiemmin: käsi kahvalla Y = 450 cm asti, rata silti sileä), irroituksen paikka on kinemaattisesti määräämätön (ikkuna ulottuu ohjatun vaiheen sileään osaan, enintään työnnön loppuun, Y >= 200 cm); nopeus ja suunta ovat silti oikeita (vapaa kappale, jonka rata on sileä, jatkuu samana). Suunta irroitushetkellä riippuu paikasta vähän (kiven kaarto kääntää suuntaa matkalla ~0,2-1 astetta 4 m:n matkalla, kaarevuus ~2-4e-5 1/cm kohti keskiviivaa), ja sen virhe on ~0,3-0,5 astetta (20-25 cm 30 m:n päässä): kaukopään sivusuuntainen ennuste omalla kappalla hajoaa 33-65 cm (hog-viivan mitatusta tilasta lähtevä 31-34 cm), joten sivusuunnan tarkistus (`irr_sivu_ero_cm`) on vain tiedoksi eikä aiheuta epäilyä.
* *Uskottavuustarkistus ajossa (`netti_vakiot.tarkista_irroitus`).* Kun kaukopään seuranta on valmis ja kivi pysähtyi ilman törmäystä, kiven vakioiden sovituksen kanssa samassa taustaprosessissa lasketaan irroituksesta (paikka, nopeus, suunta, kierre) mallilla ennustettu pysähtymismatka vertailukitkalla `A_ref(v) = 0,00761 + 0,00474 (v - 2,0)` (kasvaa nopeuden mukana: raskaampi heitto pysähtyy lyhyemmälle; hajonta suoran ympärillä 3,4 %) ja kiven oikealla kappalla sivusiirtymä; **ero > 2 m (`MATKA_RAJA_M`) = `epailyttava`** ("irroituksen nopeus todennäköisesti väärin"). Konsolirivi `heitto N irroituksen uskottavuus: pysahtymismatka irroituksesta 28,4 m, mallin mukaan 33,7 m (ero -5,3 m, ...)` ja CSV-sarakkeet `irr_matka_ero_m, irr_A_matka, irr_sivu_ero_cm, irr_tarkistus` sekä `nopeus_hog1_m_s, suunta_hog1_deg` (suora mittaus hog-viivalta); `tools/yhteenveto.py` laskee epäilyttävät. Normaalioloissa ~2 / 35 (~6 %) heitoista ylittää rajan (suuri harjaus tai virhe), ja jokainen raja-ylitys on syytä katsoa (heitto 16: kulki 5,8 m ennustettua pidemmälle).
* *Uudelleenajo ilman koko videota (`tools/sovita_irroitus.py`).* `python netti_live.py --lahde netti.mp4 --nopea --ei-ikkunaa --ei-kierretta --radat radat.pkl` (heittopään radat; ~25 min koko videolla) ja `python tools/sovita_irroitus.py radat.pkl [--kauko] [--menetelma kinkki] [--csv irroitus.csv]` ajaa tunnistuksen uudelleen kaikille heitoille (8 s) ja tulostaa uskottavuusmittarit (nopeus vs hog-nopeus, suunta vs hog-suunta, hidastuvuus, muutos tallennettuun, `--kauko`: pysähtymismatka vs malli).

**Kello ei lähde käyntiin (korjattu: mallipohjainen pysähtymisarvio).** Käyttäjä: välillä kello ei lähde käyntiin; jos mallin mukaan kivi on pysähtynyt jo 5 s sitten eikä kiveä ole näkynyt, oletetaan, että kivi on pysähtynyt. Vika: ilman kaukopään seurannan tulosta pysähtymisen arvio (`pysahtyi_f`) vaati sekä viivaestimaatin että hog-ylityksen että kierteen mittauksen (`h.kierre`, jota ei ole, jos kierteen mittaus epäonnistuu: ~9 % heitoista, ja se valmistuu vasta 10-25 s heiton jälkeen) ja kaukopään seuranta valmistuu vasta, kun kivi on todettu pysähtyneeksi tai hukatuksi (hukkautunut kivi: 6 s hukan jälkeen, aikaraja 80 s): kello jäi pyörimättä. Nyt (`Analyysi.malli_pysahtyi_f`, `pysahtyi_f`): kaukopään seurannan tulos käytetään, jos on; muuten malli: nopeus lähipään hog-viivalla (suoraan radasta, `nopeus_hog1`; muuten irroituksen nopeudesta) -> hog-viivalta hog-viivalle vakiohidastuvuus 0,068 m/s2 (mitattu hog-ylitys tai `h.kierre["T"]` jos on) -> kaukopään hog-viivalta pysähtymiseen v / 0,10 m/s2 (enintään 12 s, nopealla kivellä 6 s); kivi, joka ei yllä kaukopään hog-viivalle, pysähtyy vakiohidastuvuudella. Mallin mukainen pysähtymishetki **oletetaan toteutuneeksi, kun siitä on kulunut 5 s analyysiajassa (`KELLO_MALLI_ODOTUS_S`) eikä kaukopään seuranta ole nähnyt kiveä liikkeessä viimeisen 2 s aikana** (B-paikkojen vaihteluväli yli 10 cm; levossa oleva kivi ei estä). Kello alkaa silloin mallin pysähtymishetkestä (ei +5 s), ja kaukopään seurannan tuloksen valmistuessa arvo tarkentuu todelliseen. Mallin virhe (35 pysähtynyttä heittoa, ei törmäystä): mediaani -0,3 s, hajonta 3,8 s (puolet tästä on harjausta ja jään vaihtelua; 5 s:n odotus on ~1,3 hajontaa). Yksikkötestit: `tools/testaa_vakiot.py` (malli, 5 s:n raja, kivi nähty liikkeessä / levossa, mitattu hog-ylitys, ei yllä hog-viivalle, ei viivaestimaattia).

**Nopeus: säikeistys, YUV-puskuri ja C++ (`netti_live.py`, `cpp/netti_fit.cpp`).** Käyttäjän ajossa (Windows, YouTube, 14 heittoa) analyysi ehti mukaan, mutta tulostus (JPEG-purku, piirto, mp4v-koodaus, `imshow`, `waitKey`) tapahtui yhtenä jonona pääsäikeessä (~60 ms/ruutu -> 13 r/s): tallenteeseen päätyi 12814 / 29480 ruutua (43 %) ja viive kasvoi hallitsemattomasti, vaikka prosessori ei ollut täysillä. Nyt:
* **Säikeet erikseen:** (1) lukija + analyysi, (2) piirto (viivepuskurin purku kahdella rinnakkaisella säikeellä etukäteen, piirrokset ~3 ms/ruutu; piirrosten tulokset välimuistissa: siloitetut pisteet, linjojen kuvapisteet, paneeli piirretään vain alueelleen, ei koko ruudun sekoitusta), (3) tallennus (jono + ffmpeg-putki / OpenCV; täysi jono hidastaa piirtoa, mutta ruutuja ei pudoteta), (4) ikkuna (pääsäie: vain `imshow` + `waitKey`, näyttää tuoreimman ruudun; hidas näyttö ohittaa ruutuja, tallenne ei).
* **Viivepuskuri (`--puskuri yuv`, oletus):** raaka YUV 4:2:0 (3,1 MB/ruutu, 5 s viiveellä ~470 MB) JPEG:n (6 ms pakkaus + 15 ms purku / ruutu ≈ 0,6 ydintä) sijaan; muunnos ~3 ms; jos tulostus jää yli 4 s jälkeen, uudet ruudut pakataan JPEG:ksi (muistin suoja). `--puskuri jpeg` palauttaa pienen muistinkulutuksen.
* **Prosessit:** tarkan sovituksen, viivaestimaatin ja kierteen työprosessit käynnistetään (`spawn`, kuten Windowsissa; fork jumitti OpenCV:n kanssa) ja lämmitetään ennen lähteen avaamista (4 s), alemmalla prioriteetilla (`BELOW_NORMAL` / `nice 5`) ja ilman OpenCV:n omaa säiepoolia, jotta analyysi- ja piirtosäikeet saavat prosessorin ensin kun ytimiä on vähän (taustatyö vain viivästyy, ruutuja ei ohiteta);
  lopullinen estimaatti ei enää odota kesken olevaa laskentaa analyysisäikeessä (aiemmin ~1 s pysähdys heiton päättyessä -> ohitettuja ruutuja).
* **C++ (`cpp/netti_fit.cpp`, pybind11 + OpenCV):** tarkka sovitus (maskit, graniitin alfa, kelvolliset pikselit ja (X, Y)-sovitus runkomalliin, ~75 % koko analyysin laskennasta) C++:lla: sama malli ja samat OpenCV-kutsut, mutta Levenberg-Marquardt (Broyden-Jacobi) scipyn TRF:n tilalle, hullin laskenta vain kulmat antavilta profiilin tasoilta ja peiton alasnäytteistys ilman erillistä `resize`-kutsua:
  **12 ms / ruutu vs Python 75 ms (6 x nopeampi; hiekkalaatikon 4 ytimellä, tyhjä kone: 47 -> 12 ms)**. Tarkkuus: hyväksyttyjen sovitusten (peitto >= 0,35, kelvollisia >= 0,2; 109 / 144 testiruudusta) ero Pythoniin X: 0,10 cm, Y: 0,12 cm (keskihajonta; kuvakohtainen mittakohina on 0,1-0,3 cm), kiven hyväksymispäätökset samat 143 / 144 ruudussa, ruutujen väliset hypyt samat tai pienemmät (X 0,24 vs 0,34 cm), koko 3 heiton tulokset samat (irroituksen paikka +-5 cm, suunta +-0,03 astetta, T-viivalla +-1 cm).
  Python-toteutus säilyy varana: jos moduulia `netti_fit` ei löydy (tai `NETTI_EI_CPP=1`), käytetään sitä; käynnistyksessä tulostuu `tarkka sovitus: C++ (netti_fit)` tai `Python (kivi_tarkka)`.

  Käännös (kuten muut C++-moduulit, `CMakeLists.txt`; vain tämä moduuli): Windows: `cmake -B build -G "Visual Studio 17 2022" -A x64`, `cmake --build build --config Release --target netti_fit` -> `netti_fit.cp312-win_amd64.pyd` tulee `netti_live.py`:n viereen. **Python-versio:** `.pyd` latautuu vain sillä Pythonin versiolla, jolle se on käännetty (`cp312` = Python 3.12; oletus `C:/Python312`). Jos ohjelmaa ajetaan toisella Pythonilla (`python --version`), `import netti_fit` antaa `ModuleNotFoundError` (tiedosto on kansiossa, mutta Python ei tunnista sen päätettä) ja käynnistysrivi kertoo syyn. Joko ajetaan ohjelma 3.12:lla (`C:\Python312\python.exe netti_live.py ...`) tai käännetään moduuli sille Pythonille, jota käytetään: `python -m pip install pybind11`, poista vanha `build`-kansio (CMake muistaa edellisen Pythonin) ja `cmake -B build -G "Visual Studio 17 2022" -A x64 -DPython_EXECUTABLE="<polku>\python.exe"`, sitten `cmake --build build --config Release --target netti_fit` (polun saa komennolla `python -c "import sys; print(sys.executable)"`). Käännön lopussa jälkitarkistus (`cpp/tarkista_moduuli.py`) tulostaa kääntämiseen käytetyn Pythonin ja sen, latautuuko moduuli (`netti_fit latautuu: ...`); jos ei, se kertoo syyn (esim. OpenCV:n DLL-tiedostot puuttuvat `.pyd`:n vierestä). Myös `-DOpenCV_DIR=...`, `-Dpybind11_DIR=...` ja `-DCMAKE_TOOLCHAIN_FILE=...` voi antaa komentorivillä. Linux: `g++ -O2 -std=c++17 -shared -fPIC $(python3 -m pybind11 --includes) $(pkg-config --cflags opencv4) cpp/netti_fit.cpp -o netti_fit$(python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))") $(pkg-config --libs opencv4)`.
* **Kaukopään haku on harvennettu:** hakuvyöhyke kattaa koko radan leveyden (~340 000 px oikaistussa kuvassa), joten sitä tutkitaan joka 3. pikselistä (ROI suoraan harvennettuna alkuperäisestä ruudusta; kahvan blobi on silti 15-30 px; tulos sama maski pikselilleen kuin koko tarkkuudella): ~1 ms / ruutu (alussa täysillä pikseleillä 21-25 ms, mikä sai analyysisäikeen jäämään jälkeen ja ohittamaan ruutuja heiton jälkeen).
* **Mitattu (hiekkalaatikko, 4 ydintä, `netti.mp4` reaaliajassa tiedostona + tallennus + C++, koko 131 s):** 3927 ruutua analysoitu, **0 ohitettu**, 0 virhettä, 3927 ruutua piirretty ja kirjoitettu tiedostoon (täysi tallenne), suurin piirtoviive 0,2 s (ennen: tallenne 43 % ruuduista, viive kasvoi). Prosessorin käyttö heiton aikana (Python-sovituksella): pääprosessi ~1,6 ydintä, ffmpeg (libx264 superfast) ~1 ydin, kierteen työprosessi ~0,8;
  `--x264-preset ultrafast` säästää ~0,3 ydintä, ja NVENC / QSV / AMF vie koodauksen lähes kokonaan pois prosessorilta. `--debug-viive` tulostaa hitaat analyysiruudut ja ohitusjaksot (jos kone ei jaksa, ohitetaan ruutuja analyysista; ne piirretään ja tallennetaan silti).

**Tulokset (netti.mp4, simulaatio `--nopea`, koko video 131 s).**

| | heitto 1 (keltainen) | heitto 2 (punainen) | heitto 3 (keltainen) |
|---|---|---|---|
| lähtö = kiven piirto alkaa tulosteessa / heitto vahvistettu (analyysiaika) | 12,6 s / 14,0 s | 57,5 s / 60,5 s (työnnön alku 58,5 s, ennen sitä vastaliike hackissa) | 107,3 s / 108,4 s |
| irroituksen paikka Y | **375 cm** (offline 377) | **282 cm** (254) | **359 cm** (361) |
| irroituksen suunta | **+5,04 +- 0,22 astetta** (+4,97 +- 0,23) | **+0,96 +- 0,04** (+1,04 +- 0,07) | **-0,15 +- 0,28** (-0,20 +- 0,28) |
| liun suunta | **+4,23 astetta** (+4,31) | **+0,15** (-0,20) | **-0,58** (-0,65) |
| nopeus ennen / jälkeen irroituksen | **1,87 / 1,87 m/s** (1,87 / 1,87) | **2,06 / 2,11** (2,07 / 2,13) | **2,44 / 2,51** (2,44 / 2,52) |
| liun hidastuvuus | +0,02 m/s2 (-0,04) | +0,04 (+0,06) | -0,09 (-0,14) |
| kaukopään T-viiva: irroitus / liuku (+ = heittäjän oikealle) | **+318 / +273 cm** (+314 / +278) | **+39 / -7 cm** (+43 / -27) | **-29 / -52 cm** (-32 / -56) |

Suluissa offline-tulos (`liuku_irroitus.py`, siluettiseuranta koko aineistolla): reaaliaikainen versio osuu samoihin lukuihin (irroituksen suunta <= 0,09 astetta, T-viivalla <= 5 cm; liun suunta heitossa 2 poikkeaa 0,35 astetta eli 20 cm T-viivalla: ohjatun vaiheen mittaus on heikko, offline-tuloksen vaihteluväli on tätä luokkaa; liun hidastuvuuden virhe on +-0,15 m/s2).
Koko videolla (3 heittoa vasemmassa päässä, 131 s) ei tullut yhtään väärää heiton vahvistusta: heittäjän kiven käsittely hackissa (kuljetus, siirtely, vastaliike; useita yrityksiä) hylättiin ja oikean pään kahvakandidaateista (kivi hackissa, ihmisten varusteet) ei syntynyt heittoa. Simulaatio kesti 145 s (28-30 r/s, 4 ydintä; B-työt odotetaan synkronisesti),
reaaliaika-ajossa (tiedosto tahdistettuna, 28 s) kaikki ruudut ehdittiin analysoida ja tulostua 5 s viiveellä; analyysin ruutukohtainen kustannus on mediaanilta 2,5 ms (95 %: 21 ms, sisältää B-tulosten odotuksen sim-tilassa). Valmis esimerkkituloste: `tulokset/kivenseuranta_demo.mp4` (1920 x 1080, H.264, 9 MB; koko video, 3 heittoa, paneeli, kierre ja pisteet jään tasolla; reaaliaika-ajo + C++, uudelleenkoodattu pienemmäksi) ja `tulokset/live_heitot_vasen.csv` (samat luvut: liuku, heitto, irroitus, kierre, hog-hog-aika, nopeudet ...).

Alustavan estimaatin tarkentuminen (T-viivan osumakohta cm, aika heiton lähdöstä analyysissa; tulosteessa estimaatti on sillä hetkellä paras tieto, eli kun kivi lähtee tulosteessa, analyysi on 5 s eteenpäin):

| | irroituslinja T-viivalla | liukulinja T-viivalla |
|---|---|---|
| heitto 1 | +237 (+4,0 s) -> +270 (+4,5) -> +273 (+4,9) -> +281 (+5,3) -> **+318** (+5,7 s, vakaa) | +275 ... **+273** |
| heitto 2 | +25 (+5,1 s) -> +49 (+5,6) -> +41 (+6,1) -> +41 (+6,5, vakaa) -> **+39** (+6,9) | -6 ... **-7** |
| heitto 3 | -41 (+4,1 s) -> -10 (+4,6, vakaa) -> **-29** (+4,6 s, lopullinen) | **-52** |

Pitkä liuku (heitto 1: 4,4 s) on vaikein: 5 s:n kohdalla vapaata lentoa on alle sekunti ja irroituslinja on T-viivalla 40-80 cm pielessä (katkoviiva, "alustava"), minkä jälkeen se tarkentuu 1 s:ssa; liukulinja on oikein heti (+-7 cm). Heiton 2 ensimmäinen estimaatti valmistuu vasta 5,1 s lähdöstä (tulosteen lähdön jälkeen 0,1 s), koska lähtö on laskettu vastaliikkeen alusta.


**Rajoitukset (rehellisesti).**
* **YouTubea ei voitu testata tässä ympäristössä** (YouTube-yhteys estetty, `Tunnel connection failed: 403`): `yt_dlp`-haku on kopio omasta testikoodista ja virheet raportoidaan selkeästi, mutta virran lukeminen on testattu vain muilla lähteillä: tiedosto reaaliajassa ja saman videon suora http-virta (`--url`, sama `cv2.VideoCapture(..., CAP_FFMPEG)` -polku kuin YouTubessa). Koko YouTube-videon sisältö (7625 s, 1,47 Mbps H.264, 1920 x 1080) on ajettu tiedostona (käyttäjän lataama `netti_yt.mp4`, ks. "Hukatut heitot koko videolla"); tämä on sama pakkaus kuin YouTube-virrassa.
  Jos YouTube-virta on eri resoluutiota kuin 1920 x 1080, ruudut skaalataan siihen (kalibrointi on tehty sille); `--yt-formaatti 'bestvideo[height<=1080]...'` rajaa korkeuden.
* Alustavuus: 5 s:n viiveellä irroituslinja on heti käytettävissä, mutta pitkällä liu'ulla (heitto 1: 4,4 s) vapaata lentoa on alussa alle sekunti ja irroituslinja on aluksi T-viivalla jopa ~40 cm pielessä; se on katkoviiva ja tarkentuu 1-2 s:ssa (suurempi `--viive` parantaa). Liukulinja on alusta lähtien oikein (±7 cm 31 m:n päässä).
* Tunnistus edellyttää keltaisen tai punaisen kahvan (muut värit eivät laukaise), kameraparia samassa paikassa kuin kalibroinnissa (`tulokset/kalibrointi_*.json`), ja että hack-alue on näkyvissä. Aikaperusta olettaa 25 r/s:n lähdevirran (30 r/s:ksi tallennettuna kopioruutuineen; sama kuin offline).
* Oikean pään heitto (toinen pää heittäjänä) on symmetrinen koodipolku; se on testattu vain kääntämällä puoliskot keskenään (sama video, väärä kalibrointi toiselle puoliskolle), ei oikealla oikean pään heitolla.
* Kierre (kierron nopeus ja suunta) mitataan nyt reaaliajassa (ks. yllä), mutta tulos valmistuu vasta ~10-15 s heiton päättymisen jälkeen (tulosteessa paneelin riville ilmestyy luku, kun se on valmis; sitä ennen `?` / `mitataan`), kierrosmäärä vasta kun kaukopään hog-ylitys on nähty (~20 s heiton jälkeen; sitä ennen `alustava` mallin hog-hog-ajalla). Kaukopään ylitys perustuu kahvan väriin: jos kahva on piilossa (esim. harjaajat peittävät) tai ratoja on useita, aika voi puuttua tai heittää ~0,3 s.
* Kivien numerointi olettaa, että kivet heitetään numerojärjestyksessä ja että kaikki heitot havaitaan; jos heitto jää havaitsematta (tai oikealta puolelta heitetään harjoitusheitto), numerot siirtyvät. Numerointi alkaa alusta kun heittosuunta vaihtuu tai väri on heitetty 8 kertaa. Ajon tarkistus: `python tools/tarkista_heitot.py ajo.log` (tai `heitot.csv`) kertoo päittäin heittojen määrän ja kohdat, joissa sama väri heitetään kahdesti peräkkäin (välistä puuttuu heitto) tai kivinumerot ovat epäjatkuvat; ajon lokiksi `python netti_live.py ... --tapahtumat > ajo.log`.
* Prosessorikuorma: analyysi + puskurointi ~15 ms/ruutu pääsäikeessä; tarkka sovitus 3 työprosessia heiton aikana (nelisydämisellä koneella 0 ohitettua ruutua 28 s:n reaaliaika-ajossa). Jos kone ei jaksa, kuva tulee myöhässä mutta tulos ei muutu (toistettava reaaliaika, ks. yllä; vanha ruutujen ohitus vain live-virrassa tai `--ohita-myohassa`); `--saikeet 0` jättää tarkan sovituksen pois (epätarkka).
