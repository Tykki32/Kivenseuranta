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
