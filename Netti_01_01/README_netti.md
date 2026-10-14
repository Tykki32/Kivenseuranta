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

### Kiven mitat, kahvan muovin ellipsi ja korkeus (`mittaa_kahva.py`, `kivi_netti.py`: `kahva_silhuetti`, `sovita_kuvaellipsi`, `kahva_keskipiste_3d`)

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
