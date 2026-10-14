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

Kummankin puoliskon kuva on kameran 16:9-kuvan keskileikkaus (960x1080), joten optinen keskipiste on puoliskon keskellä
(480, 540). Tasohomografiasta (Zhang) saatu K, R, t: vasen f = 2293 px, kamera (X, Y, Z) = (4, 1380, 572) cm; oikea f = 2285 px,
(-2, 1377, 576) cm. Kamerat ovat siis samanlaiset ja ~5,7 m korkeudessa, 66 astetta pystysuunnasta (katsovat radan suuntaan).
Näin itsenäisesti sovitetut puoliskot antavat samat f, korkeuden ja etäisyyden, mikä vahvistaa kalibroinnin.
Puolisko rajataan nyt y = 190...1080 (hack ja heittäjä mukaan); TV-grafiikka jää rajauksen ulkopuolelle paitsi minikartan alareuna.

## Kiven tunnistus ja mittaus (`kivi_netti.py`, `aja_kivi.py`, `piirra_radat.py`)

`python aja_kivi.py Recording.mp4 --puoli vasen --ulos tulokset --ikkunat 8-22,56-66,104-114` ->
`tulokset/kivet_vasen.csv` (kaikki havainnot), `tulokset/radat_vasen.csv` (radat: raaka + tasoitettu paikka, nopeus);
`python piirra_radat.py tulokset vasen` -> `tulokset/radat_vasen.png`. Oikealla puoliskolla ei heitetä (tässä videossa).

* **Videon aikaperusta:** nauhoite on 25 r/s -virrasta 30 r/s:ksi, joten joka 6. ruutu on edellisen kopio (|ero|:n 99,9.
  persentiili < 15, oikeilla ruuduilla staattisessakin kuvassa >= 30). Kopiot ohitetaan ja aika = yksilöllisen ruudun
  järjestysnumero / 25 (muuten nopeus vaihtelee ±2 cm/ruutu).
* **Esikäsittely:** puolisko rajataan ja käännetään kuten kalibroinnissa, k1 korjataan, tausta = moodikuva. Siemenet:
  graniittimaski (taustasta poikkeava, V 75-175, S < 70) ja kahvamaski.
* **Taustan vaikutus (korjattu):** graniitin peittävyys `alfa_kartta` lasketaan värien sekoitusmallista pikseli = a * graniitti
  + (1 - a) * tausta, a = (p - b).(g - b) / |g - b|^2 (g = graniitin BGR (130, 126, 136), b = moodikuvan pikseli). Aiempi
  V-suhde jakoi tumman taustan (punainen rengas) pienellä nimittäjällä, jolloin kiven varjo punaisella renkaalla näytti
  graniitilta ja kiven reunapikselit sinisellä/punaisella renkaalla katosivat; nyt reuna on sama jäällä ja renkailla. Jään
  varjo (a < 0,3) nollataan (0,5-taso säilyy).
* **Kahva ja yläpinta (mallinnettu):** yläpinta on kiiltävä ja vaalea eikä näy graniittina, ja kahva (keltainen, sormet
  peittävät) ei erotu sormista värillä. Siksi kiven yläpinnan ympyrä (z = H, sade 0,78 R) ja kahvan yläympyrä (sade 0,5 R,
  8 cm yläpinnan yläpuolella) muodostavat "ei väliä" -alueen (kupera peite), jossa siluetti ei saa eikä menetä. Graniittina
  mitataan vain sivuseinä + reuna (puolikuu); sovitus: 1 - IoU alipikselitarkkuudella (siluetti 4x ylinäytteistetty).
* **Mitat:** R_max 15,59 / 15,68 / 15,85 cm ja H_total 15,59 / 15,51 / 16,12 cm kolmessa heitossa (keskiarvo R 15,7, H 15,7;
  nimellinen WCF max R 14,55). Oikean kameran yksi saapuva kivi: R 14,9, H 16,1 (10 havaintoa, huono sovitus). Mittojen
  hajonta heitosta toiseen on alle 0,3 cm, joten ne toistuvat, mutta ovat ~8 % nimellistä suuremmat (lateraalinen leveys
  72 px vs 67 px nimellisellä); syy auki: kiven profiili on yksinkertainen malli, mahdollinen pikselien anisotropia tai
  mittakaava. Oletuksena käytetään mitattuja (MITATTU_R_CM, MITATTU_H_CM), koska kiven jalan paikka Y riippuu H:sta (kamera
  katsoo radan suuntaan: 4 cm korkeusvirhe ~ 9 cm Y-virhe).
* **Ketjutus ja aukkojen täyttö:** vakionopeusennuste (portti 18 cm + 8 cm/puuttuva ruutu, aukko enintään 14 ruutua); jäljelle
  jääneet aukot (kasi/vartalo kiven päällä, tunnistus ei löydä) täytetään sovittamalla siluetti interpoloidusta paikasta.
* **Tasoitus:** paikallinen 2. asteen sovitus +-6 ruutua, painot 1/(err + 0,1)^2.
* **Tulos (vasen puolisko):** kolme heittoa seurataan hackista (Y ~ -255 cm) hoglineen (Y = 640): t = 8-18 s, 59-64 s,
  104-111 s. Nopeus T-viivalla ~1,9 / 2,0 / 2,3 m/s. Heitto 1 kulkee 5,5 astetta oikealle (X: -5 -> 84 cm), 2 ja 3 lähes suoraan.
* **Tunnetut puutteet:** nopeuskäyrässä on ~±0,3 m/s aaltoilua (jakso ~160-190 cm, kaikissa heitoissa); todennäköinen syy on
  heittäjän/lakaisijoiden jalkojen ja käsien ajoittainen peittäminen (askelrytmi ~1 Hz), ei vielä varmistettu. Y-hajonta
  ruudulta ~1,5-2,5 cm (1 px = 1,3-3 cm radan suunnassa). Varsinaiseen liu'un/irroituksen tunnistukseen tarvitaan lisäksi
  kiven ja käden etäisyys (kahvan ympärillä oleva sormiaukko), joka on seuraava vaihe.

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
* **Nopeuden aaltoilu ±0,2 m/s (jakso 0,9-1,0 s) on samanlainen molemmilla menetelmillä ja jatkuu vapaana liikkuvalla kivellä, joten se ei
  johdu seurannasta, taustaviivoista eikä kahvasta.** Jakso on sama kolmessa heitossa (nopeudet 1,7-2,3 m/s), eli se on aikaperusteinen
  (videon ruutuajoitus), ei paikkaperusteinen. Vaihe ei ole sama kolmessa heitossa (koherenssi 0,45), joten sitä ei voi korjata yhdellä
  yhteisellä aikamallilla. Yksilöllisten ruutujen ajaksi oletetaan k/25 s (jäännöshajonta 1,0-1,6 cm vs 1,7-2,7 cm jos f/30).

### Kiven mitat, akseli ja kahvan korkeus kaarista (`mittaa_kahva.py`, `kivi_netti.py`: `kaaripisteet`, `sovita_akseli_ja_korkeus`) -- KESKEN, ALLA OLEVA KAHVAN KORKEUS PERUTTU

**Kiven mitat.** WCF-säännöt (varmat): ympärysmitta enintään 914 mm (halkaisija enintään 291 mm, R <= 14,55 cm), korkeus vähintään
114 mm (graniittirunko ilman kahvaa), paino kahvoineen 17,24-19,96 kg. Muut mitat vaihtelevat valmistajan mukaan (tyypillisiä, ei
säädeltyjä): runko 28-29 cm x 11,4-12,5 cm, kokonaiskorkeus kahvan kanssa ~17-18 cm, juoksupinta (running band) rengas Ø ~12,5 cm,
leveimmän kohdan (iskukaista) korkeus alhaalla ~2-4 cm jään yläpuolella. Mallissa: iskukaista R = 14,55 cm, z = 3 cm (oletus).

**Idea:** kiven akselin keskikohta jäällä ja kahvan keskikohta korkeudella h ovat samalla pystysuoralla -> h kuvapisteiden erosta.

**Mitä on tehty ja mikä on vialla:**
* Alareunan kaari (graniitin alfa 0,5 -rajan kameran puoleinen reuna) sopii 3D-ympyrään (z = 3 cm, R = 14,55 cm) 0,3-0,6 px:llä ->
  akselin (X, Y). Ympyrän säde ja korkeus ovat kuitenkin tästä kaaresta degeneroituneet (vapaa sovitus ei erota niitä).
* **Kahvan reuna: ensimmäinen versio sovitti väärän reunan.** Keltaisen kahvalevyn kameran puoleinen (oikea) reuna on suora viiva, koska
  kiven graniittireuna peittää levyn lähemmän puoliskon; se ei ole muovin reuna. Muovin näkyvä reuna on VASEN kaari (kaukana kamerasta).
  Siksi aiemmin raportoitu kahvan taso 17,0-17,6 cm on PERUTTU.
* Vasemman kaaren sovitus 3D-ympyrään (vaakataso, keskipiste kiven akselilla): RMS 1,0-3,4 px, säde 10-14 cm (keltainen) tai 7-9 cm
  (punainen), korkeus 14-19 cm -> epäyhtenäinen, malli ei kuvaa kahvan muotoa/asentoa. Kuvasta mitattu kahvalevyn ääriviiva on ellipsi
  kuvasuhteella 0,48-0,54 (vaakaympyrän ennuste 0,47-0,55) ja pitkä akseli 37-46 px (~15-16 cm), mutta kaaren keskipiste ei ole samalla
  pystysuoralla alareunan kaaren akselin kanssa (3-5 px sivusuunnassa).
* `KAHVA_Z_CM` palautettu oletukseen 15 cm (ei mitattu).

**Päivitys (kahvan etureuna oikein):** edellinen versio käytti vain keskimmäisiä 60 % riveistä ja jätti kaaren päät pois, jolloin etureuna
näytti suoralta. Nyt etureuna = kahvamaskin komponentin ulkoreunan pisteet leveimmästä kohdasta (ylin ja alin kärki) kohti kameraa, koko
pystyalue mukana (`kaaripisteet`). Tulokset (`tulokset/kahva_etureuna.jpg`; magenta = sovitus, vihreä = etureunan pisteet):
* Kahvan ympyrä (vaakataso, vapaa z ja r) etureunaan, keskipiste alareunan kaaren akselilla: RMS 1,2-2,6 px, r osuu rajaan 14 cm -> ei sovi.
* Sama kun keskipisteen sivusuuntainen siirto vapaa: keltaiset kahvat RMS 0,4-0,7 px, z 19,3-19,4 cm, r 11-12 cm, dX +0,9...+3,6 cm.
* Yhteissovitus (sama akseli (X, Y) sekä alareunan kaarelle että kahvan etureunalle, 4 parametria X, Y, z, r): keltainen heitto 1: z 19,4 cm
  (IQR 19,1-20,6), r 12,5 cm; heitto 3: z 20,2 cm, r 13,2 cm; RMS alakaari 0,7-1,1 px, kahva 0,7-0,9 px; akseli siirtyy alakaaren sovituksesta
  X +1,4...+1,8 cm (Y alle 0,1 cm). Punainen (heitto 2): z 18,0 cm, r 9,2 cm, kahvan RMS 1,8 px (maski epäpuhdas).
* Kuvan kahvalevy on leveydeltaan ~37-46 px (~20-24 cm), eli levy on isompi kuin oletettu 9 cm sade. Ellipsin kuvasuhde täsmää
  vaakaympyrään, mutta etukaaren pullistuma on pieni (4-5 px) levyn leveyteen nähden, joten sovitus on herkkä r:n ja z:n yhdistelmälle.

**Päivitys 2 (muovin tunnistus):** kahvan muovi tunnistetaan nyt Lab-värin keltaisuudesta (b* >= 16, a* <= 10) eikä HSV-kynnyksellä
(S >= 45, V >= 165), joka leikkasi levyn vaaleat/varjoisat kärjet pois ja teki etukaaresta liian loivan. Iho (a* > 10) jää pois. Muovi on
nuolenkärjen muotoinen: oikea (kameran puoleinen) reuna on sileä kaari, vasemmalla muovi kapenee varreksi. Etureuna = kaari ylimmästä
kärjestä alimpaan kärkeen kameran puolella; sovitus: yhteissovitus (akseli X, Y sekä alareunan kaarelle että kahvan etureunalle; z, r vapaat).
Tulos (heitot 1 ja 3, keltainen): kahvan taso z = 18,8-20,4 cm (mediaani ~19,4), r = 11-14 cm, RMS 0,7-1,0 px; r ja z korreloivat
(suurempi r -> pienempi z), joten luku z = 19,4 cm edellyttää r = 11,5-12 cm. `tulokset/kahva_etureuna.jpg`: magenta = kahvan ympyrä (paksu osa =
etukaari), vihreä = etureunan pisteet, syaani = alareunan ympyrä, keltainen = alakaaren pisteet.
