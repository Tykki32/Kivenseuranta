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
