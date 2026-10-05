# Testi_08_02: haku ja seuranta täydellä resoluutiolla (kokeilu)

Pohja: Testi_08_01 v8.8. Ainoa muutos: kun kuva on leveämpi kuin `STAB_LEVEYS` (oletus 1280), stabiloinnin siirto
(vaihekorrelaatio) lasketaan 1280-levyiseksi pienennetystä kuvasta ja skaalataan täydelle resoluutiolle. Siirron
**löytäminen** on stabiloinnin raskas osa; sen **käyttö** (warp vaiheessa B), HAKU ja SEURANTA tehdään täydellä
resoluutiolla. Tiedostoajo käyttää videon omaa resoluutiota (1920×1080-video → kaikki täydellä resoluutiolla).
`STAB_LEVEYS=0` = vaihekorrelaatio täydellä resoluutiolla.

## Vertailu: `0001.mp4` (Testivideo-release), 1920×1080 vs 1280×720
Video on lomitettu (1080i, kampa liikkuvissa kivissä) → lomitus poistettu (ffmpeg yadif, 25p). Sama video pienennettynä
1280×720:ksi (INTER_AREA, kuten live) vs täysi 1920×1080, molemmat Testi_08_02:lla. Paneelit: ikkunaruudut kuvan oikeassa
reunassa (kamera eri asennossa kuin MAH-videoissa). Sandbox, 4 ydintä, ajot peräkkäin.

**Seurannan kohina** (paikan hajonta 1 s:n toisen asteen sovituksesta, mediaani; samat 3 heittoa):

| kivi | Y-kohina > 20 m | X-kohina > 20 m | Y-kohina 10–20 m | X-kohina 10–20 m |
|---|---|---|---|---|
| 1 | 3,20 → **1,94** cm | 0,49 → **0,33** cm | 0,56 → **0,40** cm | 0,28 → **0,10** cm |
| 2 | 3,91 → **3,19** cm | 0,64 → **0,40** cm | 0,61 → **0,43** cm | 0,19 → **0,10** cm |
| 3 | 4,07 → **2,38** cm | 0,54 → **0,36** cm | 1,48 → **1,18** cm | 0,25 → **0,11** cm |

Hog-analyysin sovitusvirheet: Y (kitkamalli) 1,67/1,93/3,17 → 1,08/1,46/2,59 cm, X (kurvimalli) 0,42/0,55/0,41 →
0,31/0,39/0,25 cm. Seuranta alkaa samassa ruudussa (Y ≈ 33 m) molemmilla. rms_px ennallaan → pikselirajat toimivat sellaisinaan.

**Kalibrointi** täydellä resoluutiolla parempi: pyöreys 0,938 → 0,976, kokovirhe 2,4 → 1,2 %, kaukohoglinen kulma 720p:ssä
−87,8° (vino) vs 89,7°. Tästä johtuen tulokset siirtyvät systemaattisesti (kaukohogin ylitys +0,2 s, nopeus +0,03 m/s,
X kaukohogilla +6 cm). Livessä kalibrointi tehdään jo täydellä resoluutiolla (v8.3), joten livessä hyöty on seurannan kohina.

**Nopeus** (sandbox): kokonaisnopeus 46,0 → 27,3 r/s. Vaihe B (warp + varjosuodatus koko kuvalle) 17,1 → 31,3 ms on
pullonkaula; HAKU 27 → 57 ms/kutsu; SEURANTA kiveä kohden noin +34 %; stabilointi (vaihe A) 4,8 → 6,0 ms (pienennetty).
Livessä (kannettava: B 17,7 ms, C 32 ms 720p:ssä) koko kuvan täysi resoluutio ei pysyisi 25 fps:ssä, ja live-puskuri
(≥ 66 s) veisi 1920×1080:lla ~10 GB muistia. → Seuraava askel: täysi resoluutio vain kivien ympäriltä (warp ja seuranta
paikallisesti), muu putki 720p:nä.

# Testi_08_01 (pohja Testi_07_01 v7.9, testattu toimivaksi)

## v8.8: kamera 50p + harvennuksen ohitus grab():lla
Sony HDR-CX405:n HDMI-lähtö 50i → **50p** (kameran asetus): Cam Link antaa 1920×1080 lomittamatonta kuvaa 50 fps (kameratesti
2026-10-05: 50,0 fps, ei myöhästyneitä ruutuja). 50i:ssä liikkuvien kivien reunoissa oli kampa (puolikuvat 20 ms eri aikaan).
Live käyttää joka toisen ruudun (`--live-fps 25`, `harvennus=2`).

Ensimmäinen 50p-live-ajo (50i-tallenteen toisto kamerasta, 641 s): 16 heittoa, katselunäkymän suurin viive 0,6 s. Vaihe A
(luku + stabilointi) 37,0 ms → kapasiteetti 27,0 r/s (aiemmin hitain vaihe 29,8 r/s): kamerasäie luki kaikki 50 ruutua
(`read()`, 13,8 ms CPU per käsitelty ruutu).

v8.8: harvennuksessa pois jäävät ruudut luetaan `grab()`:lla (ruutu pois ajurin jonosta ilman `retrieve()`-kopiota).
Loppuraporttiin `harvennuksessa ohitetut N ruutua grab():lla x ms/ruutu`. `LIVE_GRAB_OHITUS=0` = vanha tapa.

## v8.7: kierteet hog–hog-välillä

Kiven pyöriminen arvioidaan kahvan muodon toistumisesta:

* Kahvan ympyrästä (v8.6) otetaan joka ruudussa 32×32-pala ja siitä harmaasävyn gradientti (Sobel), kun kivi on alle
  `KIERRE_Y_MAX_CM`=23 m päässä (kauempana kahvaa ei erota). MAH00014:n 20 hog-kivellä gradientti löysi pyörimisjakson
  selvimmin (R² mediaani 0,78); nykyinen HSV-värimaski 0,44 (haalea keltainen kahva erottuu jäästä huonosti).
* Sivulta katsottuna vastakkaisiin suuntiin osoittava kahva näyttää lähes samalta, joten vahvin toisto on puolikierros
  (varmistettu kuvista kivillä 55 ja 81).
* Pyöriminen hidastuu vakiohidastuvuudella `KIERRE_HIDASTUVUUS` = **0,02 rad/s²** (kovakoodattu; MAH00014:n 12 selvän
  kiven yhteinen arvo, 90 % luottamusväli 0,012–0,028). Kierrosaika sovitetaan ruutuparien samankaltaisuuteen, ja
  kierteet = ∫ω dt / 2π kaukohogista lähi-hogiin (alkuosa ekstrapoloidaan mallilla).
* Laskenta hog-analyysin yhteydessä (n. 0,2 s per heitto pääsäikeessä; putken puskuri 2 s riittää). Heikko signaali (R² < `KIERRE_MIN_R2`=0,03) → ei arvoa.
* Paneeliin rivi `kierteita: 2.2` (`-` jos ei arvoa); hog-CSV:hen `kierteet`, `kierrosaika_far_s`,
  `kierrosaika_near_s`, `kierre_r2`. Pois päältä: `KIERRE_SEURANTA=0`.

## v8.6: kahvan väri ja näkyvyys

Seurannassa jokaisesta löydetystä ruudusta lasketaan kahvan värin pikselit:

* Ympyrä piirretään suoraan kuvaan (ei projisoitu ellipsi). Keskipiste on kiven pyörimisakselin yläpää (X, Y, H_total)
  kuvaan projisoituna, säde 2 × kahvan säde sivusuunnassa pikseleinä (`KAHVA_SADE_KERROIN`=2, kahvan kärki mukaan). Mukaan tuleva jää/graniitti on harmaata ja
  karsiutuu saturaatiorajalla (`KAHVA_MIN_S=80`, `KAHVA_MIN_V=60`).
* Ympyrän sisältä tallennetaan sävyhistogrammi (OpenCV H 0–179). Radan lopussa valitaan hallitseva sävy
  (±`KAHVA_H_TOL`=10, punainen kiertää 179→0) ja jokaiselle ruudulle lasketaan sen sävyisten pikselien määrä.
* `levy_px` = kahvan alla olevan levyn (vaakasuora kahvan säteen ympyrä) projisoidun ellipsin pinta-ala; levy näkyy
  aina, joten `kahva_netto_px = kahva_px - levy_px` jättää jäljelle kahvan kahvaosan, jonka näkyvyys vaihtelee pyörimisen
  mukana (pyörimisnopeuden tunnistusta varten).
* Tulos: `<csv>_kahva.csv` (frame, stone_id, kahva_h, kahva_vari, kahva_px, ympyra_px, levy_px, kahva_netto_px) ja hog-CSV:hen sarakkeet
  `kahva_h`, `kahva_vari`.
* Korkeustesti MAH00014:llä: kahvan värin osuus alueesta H_total +0 cm 39 %, +4 cm 27 %, +8 cm 5 %, +12 cm 0,2 %,
  joten keskipiste on akselin yläpäässä (`KAHVA_Z_EXTRA_CM`=0, ympäristömuuttujalla säädettävä).
* Pois päältä: `KAHVA_SEURANTA=0`. Tarkistuskuvat: `KAHVA_DEBUG=<kansio>`.

## v8.5: puhelimen/tabletin näyttö pysyy päällä katselusivulla
Ensimmäinen napautus (sama joka vie koko näytölle) pyytää selaimelta Wake Lockin; koska se toimii vain https:llä ja kotiverkon
osoite on http, varakeinona sivulla toistuu silmukassa pieni mykistetty video (16×16, 2 s, WebM + MP4, ~2 kt, sivun mukana),
joka pitää näytön päällä (sama tekniikka kuin NoSleep.js). Tilarivillä "näyttö päällä (video)". Testattu headless-Chromiumilla
lähiverkko-osoitteella: Wake Lock ei käytettävissä → video toistuu. Toimivuus pitää varmistaa omalla laitteella.

## v8.4: X-suunta kurvimallista (vakio sivukiihtyvyys)
Hog-analyysin X-sovitus (x ja suunta kaukohogilla, merkki) tehdään fysikaalisesta mallista: kitka hidastaa radan suuntaisesti,
kurvi on vakiosuuruinen sivukiihtyvyys k kohtisuoraan kulkusuuntaa vastaan → kulkusuunnan kulma θ' = k/|v|, X'(t) = v(t)·θ(t),
v(t) Y-sovituksesta. Kolme parametria kuten ennen (x, alkukulma, k); sama sovitusalue ja 10 huonointa pistettä pois.
Vanha toisen asteen yhtälö vastasi vakiokaarevuutta (ympyränkaari) eikä ennustanut kaukohogia hyvin. `KURVIMALLI=0` = vanha.
Heiton hyväksyntä (R_x) lasketaan edelleen kuten ennen → samat heitot.

Havaittu X kaukohogilla (±60 cm, ei sovituksessa) − ennuste:

| | vanha 2. aste | kurvimalli |
|---|---|---|
| live 2026-10-04 (66 heittoa) | rms 2,39 cm | **1,72 cm** |
| MAH00014 (19 heittoa) | rms 3,59 cm | **0,77 cm** |

Suunta kaukohogilla muuttuu rms 0,5–0,7° (max 1,9°). Sivukiihtyvyys |k| mediaani 0,009 m/s². Hog-CSV:hen `kurvi_k_ms2`,
`kurvi_rms_cm` sekä vanhan mallin `x_far_hog_cm_2aste`, `dir_far_hog_deg_2aste`. Täysi MAH00014-ajo: 20/20 heittoa, hog-hog ja
nopeus ennallaan. Live-aineiston keskimääräinen −1,1 cm johtuu mittauksesta kaukohogilla (heittäjä kiven takana +
kalibrointi), ei mallista.

## v8.3: kalibrointi kameran täydellä resoluutiolla (live)
Moodikuvan näytekohdissa (5 s välein, ~25 kpl) kamerasäie muuntaa ruudun myös täydellä resoluutiolla (1920×1080); pääsilmukka
stabiloi sen samalla, skaalatulla siirrolla, ja niiden mediaanista tehdään `<nimi>_kalibrointi_moodikuva_taysi.png`. Kalibrointi
(homografia, k1, keskiviiva) ja kameran asento (K, R, t) ratkaistaan siitä; tulos skaalataan seurannan resoluutiolle
(K × 1280/1920, H_final × diag(1,5, 1,5, 1), R ja t ennallaan) → seurannan kuorma ei muutu. Epäonnistuessa käytetään
1280×720-moodikuvaa kuten ennen. `CALIB_TAYSI=0` = vanha. Toimii myös `--live-sim`:llä, jos video on 1280×720:aa isompi.

Testi: MAH00014:n 3 min ylöskaalattuna 1920×1080:ksi, live-sim: taysresoluutioinen kalibrointi onnistui (RMS 1,25 cm,
pyöreys 1,000, kokovirhe 1,1 %), seuranta ja hog-analyysi toimivat skaalatulla kalibroinnilla, ei pudotettuja ruutuja.
**Sama video 1280×720:ksi pienennettynä (= nykyinen live-tapa) EI kalibroitunut**: "Kaukaista pesää ei löytynyt" – sama virhe
kuin kamera-ajossa 2026-10-03. Ylöskaalaus ei lisää todellista tarkkuutta, joten tarkkuusvertailu vaatii aidon 1920×1080-tallenteen.

## v8.2: k1 keskiviivan suoruudesta (kokeiltu, oletuksena pois) + täyden resoluution vertailu
`CALIB_K1_JOINT=1`: linssin k1 sovitetaan yhdessä homografian kanssa (kaikki rajoitepisteet raakakuvan koordinaateissa).
Tulos: k1 jäi ~0:aan (MAH00014 −0,003) ja heilui kierrosten välillä; keskiviivan ~0,5–1 cm taipuma kaukopäässä ei poistunut →
ei säteittäistä vääristymää → oletus pois.

Täysi resoluutio, video `00011 - Trim.mp4` (1920×1080, 40 s; moodikuva 25 ruudun mediaanista, ilman stabilointia), sama kuva
kalibroituna 1280×720:ksi pienennettynä vs täytenä:

| | 1280×720 | 1920×1080 |
|---|---|---|
| lähipesän RMS | 5,63 cm | 5,20 cm |
| pyöreys / kokovirhe | 1,000 / 4,0 % | 0,994 / 3,5 % |
| keskiviivan rms (kalibroinnin oma mittaus, k1 vapaa) | 0,41 cm | 0,37 cm |

Taipuman muoto on sama molemmilla resoluutioilla. HUOM 00011:n moodikuva on heikko (vain 40 s, pesässä kiviä koko ajan →
lähipesän RMS 5 cm, MAH00014:ssä 1,5 cm).

## v8.1: keskiviiva kalibrointiin koko radan matkalta
Kalibroinnin sivusuunnan nolla määräytyi kahden pesän keskipisteestä; MAH00014:n moodikuvassa keskiviiva oli kentän koordinaateissa
−1,0…−2,8 cm (kaukohogilla taittuen). Nyt geometrisen tarkennuksen jokaisella kierroksella keskiviivan pisteet haetaan
top-down-kuvasta 20 cm välein koko radalta (tumma laakso ±15 cm X = 0:sta, leveys ≤ 16 cm, keskikohta painotettuna; poikkeavat
3 MAD) ja lisätään sovitukseen rajoitteina X = 0. `CALIB_CENTERLINE=0` = vanha tapa.

MAH00014 (moodikuva 120 s): keskiviivapisteitä 162; keskiviiva −1,4…−3,1 cm → −0,5…+0,8 cm (kaukopäässä vielä +1,3…+1,7 cm
taipuma: linssimalli k1 ei suorista sitä homografialla). Lähipesän RMS 1,49 → 1,62 cm, pyöreys 0,990 → 0,994, koko 0,6 → 0,5 %.
Täysi ajo: heittoja 19 → 20, X-arvot siirtyvät keskimäärin 1,8 cm (keskiviivan siirtymä), kaukohogin systemaattinen virhe
(kurvimalli m = 0) −0,42 → −0,32 cm, rms 0,74 → 0,66 cm.

---

# Testi_07_01 (pohja Testi_06_01 v6.20)

## v7.9: heittoportti hyväksyy aina radat, joille hog-analyysi onnistui
Live 2026-10-04: portti hylkäsi 41/66 hog-heittoa, vaikka radat olivat ehjiä (alkavat Y ≈ 33 m): painoheitot (24) ruutukohtaisen
sovitusvirheen takia (mediaani 17–33 px > 12 px, tarkkoja ruutuja 2–36 %), lyönnit (16) liian vähien rivien (< 300) ja
nopeussuhteen (> 0,75) takia. Nyt rata, jolle hog-analyysi onnistui (R_y, R_x ja tulo > 0,99), on aina heitto (luokka 3,
etusijalla saman hogline-ylityksen tuplien karsinnassa); muut radat arvioidaan kuten ennen. Portti ajetaan vain kerran ajon
lopussa (sijainti-CSV) → ei vaikutusta käsittelyaikaan. Live-aineisto: 34 → 75 heittoa (kaikki 66 hog-heittoa + 9 muuta,
vanhan portin heitoista ei pudonnut yhtään). MAH00014: 20 → 20 (ei muutosta).

## v7.8: kitkamallin B kiinteä vakio −0,001
B on oletettavasti lähes vakio → kovakoodattu `MU_B = −0,001` (hog_analyysi.py) kaikille heitoille; heittokohtaisesti
sovitetaan vain A. Mittaukset (yhteissovitus, B yhteinen): live 2026-10-04 (66 heittoa) paras B = −0,0011, MAH00014 (19)
−0,0027. Vaikutus hidastuvuuteen @1,5 m/s verrattuna aineiston parhaaseen B:hen: hitaat heitot ≤ 0,4 % (live) / ≤ 3,9 % (MAH),
lyönnit (live) ≤ 1,2 %. Lyönneillä 1,5 m/s on ekstrapolointia: jos todellinen B olisi −0,0027, 3,3 m/s lyönnin hidastuvuus
olisi n. 18 % liian pieni. Live-aineisto uudelleen: hitaat 0,061–0,084, lyönnit 0,063–0,083 m/s² (v7.6: lyönnit −0,41…+0,93),
~20 ms/heitto. Hog-CSV:n sarake `kitka_b_kiintea` poistettu (B aina kiinteä).

## v7.7: kitkamalli nopeille heitoille (B kiinnitetty)
Live-ajossa 2026-10-04 (66 hog-tulosta) nopeiden heittojen (2,5–3,6 m/s kaukohogilla, lyönnit) hidastuvuus kitkamallista oli
järjetön (−0,41…+0,93 m/s²): kivi ei hog-hog-välillä hidastu lähellekään 1,5 m/s:a, joten B ja arvo 1,5 m/s:ssa ovat
ekstrapolointia. Nyt jos hitainkin nopeus sovitusvälillä on > 1,65 m/s, B kiinnitetään arvoon `MU_B_KIINTEA` (−0,0025;
hitaiden heittojen mediaani: live −0,0022, MAH00014 −0,0030) ja sovitetaan vain A. Hog-CSV:n `kitka_b_kiintea` = 1 näille.
Synteettinen testi (tosi A=0,0085, B=−0,003): hidas heitto 0,0715 (tosi 0,0715), lyönti 3,5 m/s 0,0659 (tosi 0,0715; vanha
keskiarvo 0,0464). MAH00014: ei muutoksia (kaikki heitot hitaita).

## v7.6: puhelinnäkymä koko näytölle napautuksella
Napauta kuvaa → sivu menee koko näytön tilaan (selaimen osoitepalkki ja järjestelmäpalkit piiloon, myös Firefox Androidissa);
uusi napautus palauttaa. Alareunan tilarivissä vihje "napauta = koko näyttö", kun ei olla koko näytössä. Sivu ei vierity.

## v7.5: puhelinnäkymä kerran sekunnissa, ei debug-videota
`--katselu` ei enää tee videota eikä kytke debug-videota päälle. Sama näkymä kuin debug-videossa (kuva, kivien ääriviivat,
tulospaneelit sekuntilaskureineen) piirretään omassa taustasäikeessään **kerran sekunnissa** (seinäkello) ja pakataan JPEG:ksi
(1100 px leveä, ~50 kt); puhelimen sivu hakee uuden kuvan kerran sekunnissa. Jos piirto on vielä kesken, kyseinen sekunti
ohitetaan – pääsäie ei koskaan odota. HLS-video ja MJPEG-tila poistettu (v7.1). Mitattu: 11,6 ms/kuva = ~1 % yhdestä ytimestä
(ennen debug-video 25 kuvaa/s + ffmpeg ≈ 0,2 ydintä + pääsäikeen odotus).

Suositeltu ajo (ei debug-videota, ei raakatallennetta koneelle):
```
python main.py --live --katselu
```
Asetukset: `KATSELU_VALI_S` (1.0), `KATSELU_LEVEYS` (1100), `KATSELU_LAATU` (75). Debug-videon saa edelleen `--debug`-valitsimella.

## v7.4: paneelit klikataan kamerakuvasta ennen live-vaihetta (ei tarvita paneelitiedostoa)
Live-tilassa ilman `--paneelit`-valitsinta ohjelma ottaa kamerasta **yhden kuvan ennen live-vaiheen alkua** ja avaa sen
ikkunaan: klikkaa paneelien keskelle (jokainen klikkaus tarkennetaan samalla paneelintunnistuksella kuin automaattihaku),
suositus vähintään 6 paneelia, **Enter** = valmis, **Esc** = peruuta. Vasta tämän jälkeen käynnistyvät puskuri, tallennus ja
moodikuvan keruu – valinnan aikaa ei puskuroida eikä tallenneta.
```
python main.py --live --live-tallenna --katselu
```
Tallennetaan `<nimi>_panel_corners.txt` (ja kuva `<nimi>_paneelikuva.png`); samalla kamera-asettelulla tiedoston voi seuraavalla
kerralla antaa `--paneelit`-arvoksi, jolloin paneelit haetaan automaattisesti. Testattu live-simulaatiolla (klikkaukset
simuloitu): paneelit 7/7, kalibrointi ja seuranta normaalisti.

## v7.3: kadonnut kivi unohdetaan 1 s:ssa, "sekuntia sitten" -laskuri, hidastuvuus kitkamallista, putki 50
* `TRACK_LOST_GRACE_SECONDS` 3 → **1 s**: jos kivi katoaa (esim. ihminen jää seisomaan eteen), rata pudotetaan sekunnissa
  (HAKU löytää sen uudelleen). MAH00014: heitot ja hog-tulokset samat kuin 3 s:lla (x kaukohogilla ero ≤ 0,24 cm),
  kivipäivityksiä 20 199 → 16 855 (−17 %), SEURANTA 17,5 → 16,1 ms/ruutu.
* Debug-videon paneelissa kiven ID:n vieressä **sekunnit kaukohoglinen ylityksestä**: live-tilassa seinäkelloon verrattuna
  (ylityksen kameraruudun aika vs. nyt), tiedostoajossa käsiteltävään ruutuun. Paneeli piirretään uudelleen kerran sekunnissa.
* **Hidastuvuus kitkamallista**: kitkakerroin μ(v) = A + B·ln(v), liikeyhtälö dv/dt = −g·μ(v). A ja B sovitetaan jokaiselle heitolle
  samoihin Y(t)-pisteisiin kuin hog-hog-sovitus (Levenberg–Marquardt, RK4-integrointi, ~50 ms/heitto). Näytetty hidastuvuus =
  g·μ(1,5 m/s). Muut luvut (nopeus kaukohogilla, hog-hog) ennallaan. Hog-CSV:ssä lisäksi `decel_keskim_ms2` (vanha keskiarvo),
  `mu_a`, `mu_b`, `kitka_rms_cm`. MAH00014 (19 heittoa): B aina negatiivinen (kitka kasvaa nopeuden laskiessa, −0,0055…0),
  sovitusvirhe pienempi kuin toisen asteen sovituksella (esim. 2,74 → 2,01 cm), hidastuvuus 0,067–0,087 m/s²
  (ero vanhaan keskiarvoon −0,004…+0,002). Jos sovitus epäonnistuu, käytetään vanhaa keskiarvoa.
* `PIPE_DEPTH` oletus 6 → **50** (2 s puskuri vaiheiden välissä, n. 800 Mt muistia).

## v7.2: kalibroinnin moodikuva 2 min ajalta
Moodikuvan näytteet kerätään ensimmäisen **120 s** ajalta (ennen 60 s), 5 s välein → enintään 25 näytettä (ennen 13).
Liikkuvat kohteet (pelaajat, kivet) häviävät moodikuvasta varmemmin. `CALIB_MODE_DURATION_SECONDS=60` = vanha.
Kalibrointi valmistuu vastaavasti n. 60 s myöhemmin. MAH00014 live-sim: 24 näytettä, hoglinet nimellisessä paikassa
RMS 1,49 cm (ennen 1,93 cm nimellisenä, 1,48 cm ±20 cm -valinnalla).

## v7.1: debug-video puhelimen selaimeen (`--katselu`)
Debug-videota voi katsoa elävänä samassa wifissä olevalla puhelimella selaimessa, muutaman sekunnin viiveellä.

**Käyttö** (kuten ennen, lisää `--katselu`):
```
python main.py --live --paneelit <paneelit.txt> --live-tallenna --katselu
```
Ohjelma tulostaa osoitteen, esim. `Katselu: avaa puhelimen selaimella http://192.168.1.23:8080/`. Avaa se puhelimella.
Windows kysyy ensimmäisellä kerralla palomuurin luvan Pythonille: salli **Yksityiset verkot**. Eri portti: `--katselu 8081`.
`--katselu` kytkee debug-videon päälle (paitsi jos `--no-debug`).

**Miksi kevyt koneelle:**
* Debug-video koodataan jo grafiikkapiirillä (QSV). ffmpeg kirjoittaa **samasta koodatusta virrasta** sekä MP4-tiedoston että
  1 s HLS-palat (tee-muxer) → ei toista koodausta, vain palojen kirjoitus (6 viimeisintä palaa pidetään, vanhat poistetaan).
* Avainruutu 1 s välein (HLS-palat alkavat avainruudulla) – videon koko kasvaa hieman, laskenta ei.
* HTTP-palvelin on Pythonin oma (`http.server`, taustasäie): jakaa vain valmiita tiedostoja.
* Puhelin toistaa HLS:n natiivisti (iPhone Safari, Android Chrome); muuten sivu lataa hls.js:n netistä.
* Varakeino **kuvatila** (`/mjpeg`, linkki sivun alareunassa, tai automaattisesti jos HLS ei ole käytössä esim. QSV:n
  puuttuessa): pienennetty JPEG 5 kuvaa/s, koodataan VAIN kun joku katsoo kuvatilaa.

Sivun alareunassa näkyy tila (kalibroidaan / etsitään kiveä profiiliin / seuranta käynnissä + ruutu ja live-viive).
Video alkaa, kun elävä seuranta alkaa (debug-video kirjoitetaan siitä lähtien). Jos toisto jää jälkeen yli 6 s,
sivu hyppää takaisin lähelle reaaliaikaa.

Asetukset (ympäristömuuttujat): `KATSELU_HLS_PALA_S` (1), `KATSELU_HLS_PALOJA` (6), `KATSELU_MJPEG_FPS` (5),
`KATSELU_MJPEG_LEVEYS` (734), `KATSELU_MJPEG_LAATU` (70).

Testattu (sandbox, live-sim, x264): sivu, tila, HLS-lista ja -palat (25 ruutua/pala, 1468x1080, ~350 kt/s ≈ 2,8 Mbit/s)
sekä MJPEG-kuva haettu ajon aikana; MP4-tiedosto ehjä. Windows-polku (`C:` ja välilyönnit) testattu.

---

# Testi_06_01 – elävä kamerasyöte (Cam Link) + ruutupuskuri

Pohja: **Testi_05_03 v5.13** (kaikki nopeutukset, tulokset identtiset v5.6:n kanssa). Lisäys: videotiedoston sijaan kuva voidaan lukea
**suoraan kamerasta** (Sony HDR-CX405 -> HDMI -> Elgato Cam Link, UVC). Tiedostoajo toimii kuten ennenkin.

## Käyttö
```
python main.py --live --paneelit D:\Tikku\Suorita\MAH00014_leikattu_panel_corners.txt --debug --live-kansio D:\Tikku\Live
python main.py --live 1 ...                 # laitenumero (oletus: etsitään 1920x1080-laite)
python main.py --live ... --live-tallenna   # tallentaa käsiteltävän kuvan myös videoksi (<pohja>_live.mp4, QSV) myöhempää uudelleenajoa varten
python main.py --live ... --live-kesto 3600 # lopettaa tunnin jälkeen (muuten Ctrl+C: puskuri käsitellään loppuun ja tulokset kirjoitetaan)
python tools/kamera_testi.py                # kameran kartoitus (koot, pakkausmuodot, todellinen fps) -> kamera_testi_<aika>.txt
```
Tulokset: `<kansio>/live_<päivä>_<aika>_*` (sijainti-CSV, hog-CSV, debug-video, kalibrointikuvat) + `..._kivien_sijainnit_live_aikaleimat.csv`
(käsitelty ruutu -> puskurin/kameran ruutu -> seinäkelloaika). Paneelit tunnistetaan ensimmäisestä ruudusta referenssitiedoston avulla (`--paneelit`;
ilman sitä avautuu valitsin kuten ennenkin).

## Miten toimii (`live_source.py`)
* **Kamerasäie** lukee kameraa jatkuvasti (Windows: DirectShow; `--live-taustaj MSMF`, `--live-fourcc MJPG/YUY2/NV12`), pienentää kuvan
  käsittelykokoon (`--live-koko`, oletus **1280x720** = testattu putki, INTER_AREA) ja harventaa tarvittaessa (`--live-fps 25`: jos kamera antaa
  ~50 fps, käytetään joka toista ruutua) ja kirjoittaa ruudut **RAM-puskuriin** juoksevalla indeksillä.
* **Peräkkäislukija** (pääsilmukka): `mode_engine` saa ruudut puskurista (`ModeEngine(w, h, fps)` + `set_frame`, C++:aan lisätty).
* **Hyppivät lukijat** (kalibroinnin profiilin opettelu lukee ±30 s siemenruudun ympäriltä, alfa-havainnot, värireferenssi, paneelien ensimmäinen ruutu)
  saavat `cv2.VideoCapture`-yhteensopivan `BufferedCapture`-olion: taaksepäin puskurin sisällä, eteenpäin odottaa kunnes kamera on tuottanut ruudun.
* **Historia**: kalibroinnin aikana `--live-taakse-s` (40 s) peräkkäislukijan takana; puskurin enimmäiskoko `--live-puskuri-s` (90 s; 1280x720 ~ 69 MB/s
  -> enintään ~6 GB RAM, käytännössä historia + viive). Täydellä puskurilla vanhinta historiaa poistetaan ensin 31 sekuntiin
  (profiilin opettelun ikkuna) ja vasta sitten pudotetaan uusia ruutuja.
* **Elävän seurannan alkaessa** hypätään uusimpaan ruutuun (`LIVE_HYPPY=1`): kalibroinnin aikana kertynyt viive pois, historia pienenee
  `LIVE_TAAKSE_SEURANTA_S`:iin (2 s). Seuranta alkaa vasta tästä, joten hyppy ei katkaise yhtään seurattua kiveä.
* Jos käsittely on hetkellisesti hitaampi kuin kamera, viive kasvaa (puskuri joustaa); edistymisrivillä `live-viive X s`. Vasta jos puskuri täyttyy
  lukemattomista ruuduista, uusimpia pudotetaan (varoitus + laskuri).
* CSV:n `timestamp_s` = käsiteltyjen ruutujen määrä / fps (hypyn kohdalla ei aukkoa); oikea kellonaika aikaleimatiedostosta.

## Testaus ilman kameraa
* `--live-sim <video> --live-sim-tahti 0` + `LIVE_HYPPY=0`: video "kamerana" niin nopeasti kuin käsittely ehtii (ei pudotuksia, ei hyppyä)
  -> **sijainti- ja raaka-CSV tavu tavulta identtiset tiedostoajon kanssa** (testattu MAH00014, 6000 ruutua). Puskuri, takaisinhypyt ja ModeEnginen
  live-tila toimivat siis oikein.
* `--live-sim <video>` (reaaliaikatahti) + hyppy + `--live-kesto 300`: kalibroinnin aikana viive kasvoi 41,8 s:iin (profiilin opettelu lukee 30 s
  eteenpäin), hyppy ohitti sen, seurannan aikana viive 0 s, ei pudotuksia; 6 heittoa / 6 hog-analyysiä 300 s:n aikana.

## Kameratesti (Windows-kone, 2026-10-02)
Cam Link: **1920x1080 YUY2 (pakkaamaton), 25,0 fps**, read ~40 ms (= ruutuväli), ei myöhästyneitä ruutuja (DirectShow). Kamera ei skaalaa
(1280x720-pyyntö -> 1920x1080), pienennys 1280x720:een ~3-4 ms/ruutu kamerasäikeessä. Laitenumerot: DirectShow 0 = Cam Link, 1 = webkamera;
MSMF päinvastoin (1 = Cam Link; pitkässä mittauksessa 2 myöhästynyttä ruutua). -> `--live` (automaattinen haku, DirectShow) tai `--live 0`.

## Ensimmäinen live-ajo (Windows, 2026-10-02, ~11 min, debug + tallennus)
16 273 kameran ruutua, käsitelty 14 798, hypyssä ohitettu 59 s, pudotettu 371 (kalibroinnissa, viive max 83 s), 18 heittoa / 18 hog-analyysiä.
**Seurannan aikana käsittely ei pysynyt tahdissa:** pääsäie 43,7 ms/ruutu (22,9 r/s) vs tiedostoajo 34,0 ms; viive ruudussa 14000 = 31 s.
Kaikki vaiheet ~25-40 % hitaampia kuin tiedostoajossa (prosessi 5,2 ydintä vs 4,7): live-tilan lisätyö (DirectShow YUY2->BGR 1080p,
pienennys 1080p->720p ~3,5 ms, tallennuksen BGR->NV12 ffmpegissä) + eri sisältö (87 rataa / 18 heittoa, kiviä seurannassa 74 % ruuduista).
v6.2:
* `--live-tallenna` syöttää ffmpegille valmiin YUV 4:2:0 -kuvan (kuten DEBUG_YUV) ja tallentaa nyt myös ensimmäiset ruudut (lähde käynnistyy
  vasta kun tallentaja on kytketty) sekä odottaa jonon loppuun sulkiessa.
* Aikaleimatiedostossa myös `kasittely_unix`, `viive_s` (lukuhetki - saapumishetki; lukuhetki = liukuhihnan vaihe A, ~0,5 s ennen pääsäiettä)
  ja `odottavia_ruutuja` -> viivekäyrä suoraan tiedostosta.
Seuraavaksi: tallenteen (`*_live.mp4`) ajo tiedostona erottaa live-tilan lisätyön ja sisällön vaikutuksen; kameran HDMI-lähtö 720p:ksi (jos
asetus löytyy) poistaisi pienennyksen ja puolittaisi muunnoksen.

## v6.3: kameran kuvan muunnos (`--live-muunnos ajuri|raw|gpu`)
Sisältö oli live-ajossa sama kuin MAH00014:ssä (video toistettiin kameran muistista HDMI:n kautta) -> hidastuminen johtuu live-tilan
lisätyöstä: 1080p YUY2 -> BGR -muunnos + pienennys 720p:ksi vie sandboxissa ~13,6 ms/ruutu yhdellä ytimellä (~0,5 ydintä 25 fps:llä),
mikä vastaa mitattua lisäkuormaa (5,2 vs 4,7 ydintä). Pienennys YUV-muodossa ennen muunnosta ei auta (14,4 ms).
* `ajuri` (oletus): DirectShow muuntaa BGR:ksi, sitten pienennys. `raw`: raaka YUY2 (CAP_PROP_CONVERT_RGB=0) + cv2.cvtColor (SIMD, monisäikeinen).
  `gpu`: raaka YUY2 + muunnos ja pienennys OpenCL:lla (cv2.UMat, Intel UHD). Tulos sama (sandbox: PSNR ~361 dB). Jos raakakuva ei ole YUY2, palataan ajuriin.
* Loppuraportissa kamerasäikeen ja tallennussäikeen CPU-aika/ruutu.
* `tools/kamera_testi.py` mittaa jokaiselle 1920x1080-laitteelle/taustajärjestelmälle tilat ajuri/raw/gpu: koko prosessin CPU ms/ruutu.
Miksi 1280x720: MAH-videot (Sony HDR-CX405 MP4) ovat 1280x720; kalibrointi, paneelireferenssi ja nopeusmittaukset on tehty sillä koolla.
HDMI-kuva on pakkaamaton, joten pienennetty kuva on vähintään yhtä hyvä kuin MAH-tiedoston H.264 (~3 Mbit/s). `--live-koko 0` = täysi 1080p (~2x hitaampi).

## v6.4: oletuksena MSMF + GPU-muunnos (Windows), väritarkistus
Kameratesti (Windows): luku + muunnos 1280x720 BGR:ksi, koko prosessin CPU/ruutu: DSHOW + ajuri **30,3 ms** (0,76 ydintä), MSMF + ajuri 35,1 ms,
MSMF + raaka + OpenCV 36,3 ms, **MSMF + raaka YUY2 + GPU (OpenCL) 11,2 ms (0,28 ydintä)**; DirectShow ei anna raakakuvaa (palauttaa aina BGR:n).
* `--live-muunnos auto` (oletus): Windowsissa ensin MSMF + gpu (etsii 1920x1080-laitteen, MSMF:llä Cam Link = laite 1), jos ei onnistu
  -> DirectShow + ajuri. `--live-taustaj` / `--live-muunnos ajuri|raw|gpu` ohittavat automaattivalinnan.
* Väritarkistus käynnistyksessä: ajurin muuntama ruutu vs. raakaruudun oma muunnos; tavujärjestys (YUY2/YVYU/UYVY) valitaan pienimmän
  eron mukaan (oikea ~1, vaihtuneet U/V ~8, UYVY ~46 sandboxissa) ja jos ero >= 5, palataan ajurin muunnokseen. Tulos `Live-lahde`-rivillä
  (`muunnos`, `raakajarjestys`, `varitarkistus_keskiero`).

## v6.5: raakatila asetetaan ennen ensimmäistä lukua
Ensimmäinen v6.4-ajo: MSMF antoi raakatilassa `(1, 8294400)` = 1920x1080x4 (RGB32) -> paluu DirectShow + ajuri. Syy: CAP_PROP_CONVERT_RGB=0
asetettiin vasta kun virta oli jo käynnissä BGR-tilassa (MSMF alustaa virran uudelleen RGB32:ksi; kameratesti asetti sen ennen lukua ja sai YUY2:n).
Nyt raakatilassa kamera suljetaan ja avataan uudelleen raakatila valmiiksi asetettuna; väritarkistuksen vertailukuva on tunnistuksessa luettu
BGR-ruutu (~1-2 s aiemmin) ja järjestys hyväksytään jos keskiero < 5 tai selvästi (< 0,5x) pienempi kuin seuraavaksi paras.

## v6.6: väritarkistus mediaanilla, `--live-raakajarjestys`
v6.5-ajo: raakakuva saatiin nyt YUY2:na, mutta tarkistus hylkäsi sen liian tiukasti (keskiero YUY2 6,8 / YVYU 12,9 / UYVY 48,0; suhde 0,53 > 0,5).
Nyt vertailu on mediaani pikselikohtaisista eroista (liike ei vääristä; sandbox 1,6 s ruutuvälillä: YUY2 1,7 / YVYU 3,0 / UYVY 45,0),
hyväksyntä: paras < 20 ja (paras < 4 tai < 0,7 x seuraavaksi paras). Tuloste näyttää erot ja parhaan kanavittaisen keskieron (B/G/R).
`--live-raakajarjestys YUY2` pakottaa järjestyksen (hylätään vain selvästi väärä kuva).

## v6.7: kalibroinnin varmistukset + hoglinet ±20 cm
Live-tallenteen (`live_..._live.mp4`) tiedostoajo kaatui: "Kaukaista pesaa ei loytynyt rivi-skannauksella", vaikka moodikuva
oli lähes sama kuin onnistuneessa live-ajossa (keskiero 1 harmaasävytaso). Syitä löytyi kaksi (toistettu sandboxissa
`Testi_05_03/live_20261002_163845_*moodikuva.png`):
* **Peilikuva:** `k8.build_image_directions` ottaa sivusuunnan merkin T-viivan janaparin *järjestyksestä*, joka vaihtelee
  kuvasta toiseen. Joskus fysikaaliset x:t vaihtuivat (+182,9 ↔ −182,9) ja homografia peilautui. Nyt sivusuunta
  asetetaan aina ei-peilaavaksi (ylhäältä katsova kamera ei peilaa). Kun merkki oli jo oikein, tulos on ennallaan.
* **k1 rajatapauksessa:** linssivääristymä k1 hyväksytään, kun se parantaa lähipesän RMS:ää > 8 %. Tässä kuvassa
  parannus oli 8,2 % (k1 = −0,08), ja 28 m päähän ekstrapoloitu kaukopesä jäi hakualueen ulkopuolelle. Jos kaukopesää
  ei löydy k1:llä, yritetään nyt uudelleen ilman vääristymäkorjausta (k1 = 0).

**Hoglinet ±20 cm (`HOGLINE_POSITION_TOLERANCE_CM`, 0 = vanha tapa):** hoglinien paikkaa ei ole mitattu tarkasti.
Kalibrointi laskee kaksi versiota: hoglinet nimellisessä paikassa (640 cm teestä) ja hoglinet vapaana ±20 cm
sisällä (hogline rajoittaa vain suoruutta/kiertoa). Vapaa valitaan, jos pesien pyöreys/koko ei huonone
(`k8.candidate_is_better`). Valittu hoglinen paikka asetetaan kaikkialle (`set_hogline_positions`): kameran asento,
heittoportti, hog-ylitys, hog-hog-analyysi ja piirrot. Tuloste: `Valittu H: ... | hoglinet: lahi 828.8 cm (+5.9), kauko 3004.6 cm (-13.0)`.

Kalibrointikuvat (sandbox):

| kuva | nimellinen | ±20 cm | valinta |
|---|---|---|---|
| MAH00014 | RMS 1,93, pyöreys 0,988, koko 0,5 % | RMS 1,40, 1,000, 0,9 %, lähi +5,9 / kauko −13,0 | ±20 |
| Testi_01 leikattu | 1,67, 0,990, 0,8 % | 1,37, 1,000, 0,5 %, +4,9 / −6,2 | ±20 |
| Testi_01 00011 (eri kenttä) | 5,50, 0,976, 1,4 % | 5,51, 0,978, 1,7 %, +1,1 / −13,6 | nimellinen |
| live-ajon moodikuva | 2,07, 0,978, 1,7 % | 1,27, 0,962, 0,3 %, +5,4 / −18,6 | ±20 |
| live-tallenteen moodikuva | 2,30, 1,000, 0,4 % | 2,45, 1,000, 2,6 %, +5,1 / +20,0 | nimellinen |

Lähi-hogline mitataan tasaisesti (+5 cm); kaukohogline (30 m päässä) määräytyy kuvasta heikommin (−6…−19 cm), siksi
laatuvertailu. Pelkän lähi-hoglinen vapauttaminen oli huonompi (pesät huononivat 4/5 kuvassa).

MAH00014 koko ajo (sandbox, vrt. v5.12): samat 22 heittoa, hog-hog-aika −0,02…−0,04 s, nopeus kaukohoglinella
−1,5 %, x kaukohoglinella −0,3…−0,9 cm. Muut kalibroinnit (peili/k1) bit-identtiset kaikilla vanhoilla moodikuvilla.

## v6.8: hoglinet symmetrisesti (sama etäisyys omasta T-viivasta)
Molemmat hoglinet ovat 640 + d cm omasta T-viivastaan (yksi yhteinen siirtymä d, |d| ≤ 20 cm). `HOGLINE_SYMMETRIC=0` = v6.7
(riippumattomat). d mitataan oletuksena vain lähi-hoglinesta (`HOGLINE_SYMMETRIC_FAR_WEIGHT=0`; kaukainen hogline rajoittaa
vain suoruutta/kiertoa). Valinta nimellinen vs. vapaa pesien laadulla kuten v6.7:ssä.

Sandbox, d samasta radasta: v6.7 lähi +4,9…+5,9 / kauko −6…−19 (yksi +20); symmetrinen kaukopaino 1: −10…+10;
**kaukopaino 0: +4,9…+5,9 kaikissa** (valittu). Pesien luvut eivät selvästi parane (MAH00014: RMS 1,48 cm, pyöreys 1,000,
koko 0,9 %; v6.7 1,40 / 1,000 / 0,9 %), mutta paikka-arvio on vakaa ja hog-hog-väli = nimellinen − 2d.

## v6.9: profiilihavaintojen ruudut kiinnitetään puskuriin
Live-simulaatio kaatui: `ruutu 1489 on jo poistettu puskurista (vanhin 1707)` kiven pintavarireferenssiä rakennettaessa.
Puskuri säilyttää 40 s historiaa, mutta täyttyessään (kalibroinnin viive 52 s) karsii sen 31 s:iin; profiilihavainnot olivat
~40 s taaksepäin. Nyt havaintojen ruudut kiinnitetään (`FrameStore.pin`) heti kun ne kerätään, ja vapautetaan
pintavarireferenssin jälkeen (`unpin_all`). Lisäksi `BufferedCapture.grab()` ei enää hae ohitettavaa ruutua
(eteenpäinluvussa välissä olevat ruudut saavat olla jo poistettu). Sandbox: MAH00014 `--live-sim --live-taakse-s 31` läpi.

## v6.11: seurannan laskennan keventäminen + nopeustesti (`tools/nopeustesti.py`)
Kamera-ajossa (tallennus päällä) kone on täysin kuormitettu ja kaikki laskenta hidastui ~30 % → laskentaa on vähennettävä.
Kolme muutosta (sandbox, MAH00014 koko video, ilman debug-videota):

1. **Prioriteetti (`LIVE_BG_PRIORITY=1`, oletus pois):** tallennuksen ja debug-videon ffmpeg-prosessit (Windows
   BELOW_NORMAL_PRIORITY_CLASS) sekä tallennussäie alemmalle prioriteetille. Kamerasäiettä ei lasketa (ajuri pudottaisi ruutuja).
   Ei muuta tuloksia. Mitataan nopeustestin ajossa `prio`.
2. **Peittolaskenta (`OVERLAP_FAST=1`, oletus):** `hullOverlapScore` yhdellä uudelleenkäytettävällä pohjalla (ei kahta väliaikaista
   kuvaa + countNonZero). Tulos tavutasolla identtinen. Hyöty jäi pieneksi: 0,014 ms/arvio molemmilla tavoilla.
3. **Ristikkohaun ohitus (`GRID_SKIP`) kokeiltiin – EI käytössä:** varjotila (21 363 kiven päivitystä): mean-shiftien pistemäärä
   on harvoin korkea (mediaani 0,66) ja ristikkohaun tulos valitaan ~44 %:ssa päivityksistä. Kynnyksellä 0,90 / 1 cm ohitettaisiin
   vain 1 % päivityksistä ja niistä 46 % muuttuisi (jopa 32 cm). **Tilalle: hienon vaiheen mäennousu (`FINE_CLIMB=1`, oletus).**
   SEURANNAN ristikkohaun hieno vaihe kävi läpi kaikki ~100 ehdokasta (1,5 cm välein ±7 cm); nyt mäennousu karkean parhaan
   pisteen ympäriltä: 14,6 arviota/haku. LM tarkentaa paikan joka tapauksessa.
   * ristikkohaun CPU 12,7 → 4,3 ms/ruutu (säästö ~8 ms suoritinaikaa ruudulta; auttaa täysin kuormitetulla koneella).
     Kiven päivityksen seinäkello vain 25,0 → 22,6 ms (ristikko ajettiin jo mean-shiftien rinnalla); sandboxin tiedostoajossa
     (kone ei täysin kuormitettu) pääsäie 29,9 → 29,4 ms/ruutu. Vaikutus kamera-ajoon mitataan nopeustestin live-tilalla.
   * heitot: kaikki 23 perustason hog-heittoa löytyvät (+1 uusi); suurin ero hog-hog 0,007 s, nopeus 0,002 m/s,
     x kaukohoglinella 0,38 cm, suunta 0,04°. Varjotila (`FINE_CLIMB=2`): sama piste 48 %, ero ka 3 cm – LM tasoittaa.

**Nopeustesti omalla koneella** (sama testi kaikille, videotiedostolla):
```
python tools/nopeustesti.py --video D:\Tikku\Live\live_20261002_163845_live.mp4 --paneelit D:\Tikku\Suorita\MAH00014_leikattu_panel_corners.txt
```
Ajaa ajot `v610` (vanha laskenta), `kiipea` (v6.11) ja `prio` kahdessa tilassa: `tiedosto` (deterministinen → heitot verrataan
v610:een) ja `live` (`--live-sim` reaaliaikatahdissa + `--live-tallenna`, kuten kamera-ajo → viive ja pudotukset).
Yhteenveto `nopeustesti_yhteenveto.txt` videon kansioon (`nopeustesti_<aika>`). Nopea kokeilu: `--max-frame 6000`.

## v6.12: nopeutukset oletuksiksi + esitarkistus 12–16 cm
* **Oletukset yhdistävät nopeutukset:** `FINE_CLIMB=1` (v6.11) ja nyt myös `LIVE_BG_PRIORITY=1` (ffmpeg-prosessit ja tallennussäie
  alemmalle prioriteetille). Nopeustesti (käyttäjän kone, live-tallenne, live-simulaatio + tallennus):

  | ajo | pääsäie | ristikkohaun CPU | suurin / ka viive seurannassa |
  |---|---|---|---|
  | v610 | 38,1 ms | 14,5 ms | 8,0 / 2,9 s |
  | kiipea (FINE_CLIMB) | 37,5 ms | 5,9 ms | 7,7 / 2,1 s |
  | prio (= v6.12 oletus) | 35,3 ms | 5,6 ms | 8,1 / 1,3 s |
* **Esitarkistus 12–16 cm** (`PRECHECK_R_MAX_MIN_CM/MAX_CM`; profiilikoe ja yksittäisen kiven tarkistus pysyvät 12,5–15,0 cm):
  live-tallenteen tiedostoajossa 3 s esitarkistus (5 havaintoa) hylkäsi kiviä R = 15,1–15,4 cm, vaikka lopullinen profiili antoi
  14,6 cm → profiili valmistui vasta ruudulla 10 026 ja seuranta kattoi vain viimeisen kolmanneksen (5 heittoa vs. live 17–18).
  MAH00014: tulokset tavutasolla samat kuin v6.11.
* `tools/nopeustesti.py`: oletusajot nyt `v610,oletus` (oletus = main.py:n nykyiset oletukset).

## v6.13: tallenteeseen kaikki ruudut + säiemäärätestit
* **Korjaus:** kun puskuri oli täynnä (kalibroinnin aikana kertynyt viive), pudotettu ruutu jäi pois myös `--live-tallenna`-tallenteesta
  (kamera-ajo 2026-10-02 23:52: 1475 ruutua = 59 s puuttui). Nyt tallentaja saa jokaisen kameran ruudun riippumatta puskurista.
* Kamera-ajo v6.12: pääsäie 46,4 ms/ruutu (tavoite < 40), SEURANTA 46,5 ms/kutsu; sama sisältö tiedostoajona 37,9 ms.
  Kone on kamera-ajossa ylikuormitettu (jokainen kivi käyttää 3–4 säiettä) → `tools/nopeustesti.py` ajot `nointra`, `noprep`,
  `vahsaie` mittaavat, onko vähempi säiemäärä nopeampi (tulokset identtiset, vain ajoitus muuttuu).

## v6.14: live-tilassa etualamaski samassa säikeessä (`PREP_PARALLEL=0`)
Nopeustesti (käyttäjän kone, live-tallenne, `--tilat live`, tallennus päällä):

| ajo | pääsäie | SEURANTA ms/kutsu | ka viive seurannassa | heitot |
|---|---|---|---|---|
| oletus (v6.13) | 38,2 ms | 39,6 | 1,95 s | 18 |
| nointra (`INTRA_PARALLEL=0`) | 42,1 ms | 41,5 | 6,2 s | 16 |
| **noprep (`PREP_PARALLEL=0`)** | **34,7 ms** | **33,3** | **1,2 s** | 18 |
| vahsaie (molemmat + `GRID_THREADS=1`) | 37,5 ms | 37,8 | 1,8 s | 17 |

Ylikuormitetulla koneella etualamaskin oma säie (yksi lisäsäie per kivi) hidastaa → live-tilassa oletuksena pois
(tiedostoajo ennallaan; `PREP_PARALLEL=1` ohittaa). Tulos on sama, vain ajoitus muuttuu.

## v6.15: seuranta loppuu lähi-hoglinelle (`TRACK_END_PAST_NEAR_HOG_CM=30`)
Kiveä ei seurata enää, kun se on 30 cm lähi-hoglinen ohi (hog-hog-analyysi käyttää vain pisteitä lähi-hog + 50 cm … kauko-hog − 100 cm).
Katkaistulla radalla heittoportin hidastuvuussuhde mitataan lähi-hoglinella eikä pesässä: MAH00014:n 24 heitolla 0,36–0,61 (ennen 0,18–0,47,
pelaaja ~1) → katkaistuille radoille oma raja `GATE_MAX_SPEED_RATIO_CUT=0,75` (0,6:lla katosi 3 heittoa; 0,7–0,85: kaikki 24, ei ylimääräisiä).

MAH00014 (sandbox, kaksi ajoa rinnakkain):

| | ennen (`TRACK_END_PAST_NEAR_HOG_CM=0`) | v6.15 |
|---|---|---|
| kiven päivityksiä | 27 789 | 23 093 (−17 %) |
| SEURANTA ms/ruutu | 21,5 | 18,2 (−15 %) |
| pääsäie | 27,6 ms | 24,4 ms (−12 %) |
| heitot / hog-hog | 24 | 24, hog-tulokset täsmälleen samat |

Kokeiltu ja hylätty: **LM-tarkennuksen keventäminen 10–24 m:n alueella** (1 ulkokierros / ei 2. LM:ää / ≤ 10 iteraatiota):
LM alueella −78…−85 %, mutta seuranta kokonaisuudessaan vain −6…−7 %, ja 3–4 heittoa 24:stä katosi (paikkavirhe 1 %:ssa päivityksistä > 10 cm).

## v6.20: hakualueen rajat koskevat myös seurannan lopullista paikkaa
Ristikkohaku ja mean-shift pysyivät jo hakualueen sisällä (X 30 cm/s, Y 300 cm/s, eli ±1,2 / ±12 cm ruudussa + ennuste),
mutta niiden jälkeinen tarkennus (lähimmän kivirungon sovitus) saattoi siirtää paikan minne tahansa; tarkistettiin vain
100 cm taaksepäin. MAH00014 621 s: harja peitti kiven ja tarkennus hyppäsi harjaan 22,3 cm sivulle yhdessä ruudussa →
heitto katosi. Nyt tarkennettu paikka saa olla enintään `TRACK_WINDOW_MARGIN_CM` (5 cm) hakualueen ulkopuolella
(alue viimeisestä paikasta tai ennusteesta), muuten havainto hylätään (MISS). `TRACK_WINDOW_MARGIN_CM=-1` = pois,
`TRACK_WINDOW_LOG=1` tulostaa hylkäykset.

MAH00014: **24/24 molemmilla profiloinneilla** (tiheä: ennen 23/24). Hylkäyksiä 2748, joista 95 % roskaradoilla; heittokivillä
133 (~1 % niiden päivityksistä). x kaukohogilla ero vanhaan ≤ 0,34 cm (vanha profiili) / ≤ 0,67 cm (tiheä profiili).

## v6.19: radan suuntaan liikkuva rata suojattu
v6.18:n tiedostoajossa 3 heittoa hukkui kesken liu'un: kahdesti oikea liukuva kivi yhdistettiin duplikaattina vanhempaan,
jo pysähtyneeseen rataan (vanhin id säilyi), kerran se poistettiin paikanvarauksessa ("huonoin sovitus").
Nyt rata, jonka Y on pienentynyt ≥ `TRACK_PROTECT_FORWARD_CM` (30 cm) viimeisen 1 s aikana:
* säilyy duplikaattien yhdistämisessä, jos toinen ei liiku (muuten vanha sääntö)
* ei joudu paikanvarauksessa poistettavaksi, jos muita vaihtoehtoja on.

MAH00014: `PROFIILI_TIHEA=0` 24/24 (x kaukohogilla ero ≤ 0,04 cm), tiheä profiili 21/24 → **23/24**. Jäljelle jäänyt
(621 s) katoaa n. 1 m ennen lähi-hoglinea (rata hyppää toiseen kohteeseen), eli kyseessä on seurannan herkkyys, ei sääntö.

## v6.18: tiheä C++-profiiliskannaus pelialueelta
Profiilivaiheessa pelialue (X ±2 m, Y 4–20 m) skannataan nyt **C++:lla 0,2 s välein** (`stone_tracker.scan_stone_candidates`,
sama kuin kandidaatin seurannassa; 7,2 ms/skannaus) entisen 2 s Python-skannausparin sijaan. Kandidaatit ketjutetaan lyhyiksi
radoiksi; liikkuva rata (≥ 4 havaintoa, siirtymä ≥ 15 cm, pääosin Y-suuntaan) tarkistetaan heti. Yhteistä 6 s jäähdytystä ei
ole: hylätty rata merkitään kokeilluksi niin kauan kuin se pysyy ketjussa. Skannaustulokset käytetään uudelleen seurannassa.
`PROFIILI_TIHEA=0` = vanha tapa, `PROFIILI_TIHEA_VALI_S` (0,2), `PROFIILI_TIHEA_MIN_HAV` (4).

Kandidaatin seuranta (`track_stone_in_video_windowed`) skannaa ruudut vasta kun rata niitä tarvitsee (`PROFIILI_LAISKA=1`):
tulos täsmälleen sama (MAH00014: CSV identtinen), mutta kadonnut rata ei enää lue/odota koko ±30 s ikkunaa.

MAH00014 live-sim (9000 ruutua): elävä seuranta alkaa **121,4 s** (ennen 145 s), hyppy alussa 33,6 s (ennen 58 s);
pysähdysrajalla (`PROFIILI_PYSAHDYS_S=2`) 106,7 s / 18,9 s.
Tiedostoajossa tiheä skannaus valitsee eri toisen kiven profiiliin (ruutu 2196 vs 2176) → profiili hieman eri
(R 13,65 vs 13,53 cm) ja seuranta reagoi siihen herkästi: 21/24 heittoa (puuttuvat 130,5 s, 139,85 s, 621 s; muiden
x kaukohogilla ±0,6 cm). Vertailuun `PROFIILI_TIHEA=0`.

Kokeiltu ja jätetty pois: seurannan lopetus kun kivi pysähtynyt (`PROFIILI_PYSAHDYS_S=2`): samoilla siemenillä 4 heittoa
24:stä katosi (värireferenssiin 24 havaintoa 51:n sijaan) → oletus 0. `PROFIILI_KIVIA=3` ja `SOLO_TRACK_MAX_RMS_PX=0.8`
eivät auttaneet (21/24 ja 20/24).

## v6.17: kalibrointi 3,5x nopeammaksi (profiilivaihe ei jää niin paljon jälkeen)
Profiilivaiheen aika MAH00014:llä (vanha, yhteensä 101 s): kalibrointi 43,7 s, `track_stone_in_video_windowed` 14,6 s,
vaihekorrelaatio 12,7 s, alfa+sovitus+värireferenssi ~11,5 s, **skannaukset vain 0,6 s** (Python `_candidates_in_frame`
35 ms/kpl, vain pelialue X ±2 m, Y 4–20 m). Kalibroinnista 44 s oli `measure_house_quality` → `_grid_search_shared_ellipse_shape`
(162 000 Python-lstsq-kutsua).
* `measure_house_quality` muistiin (sama ruutu + sama H → sama tulos): 44,6 → 22,3 s, tulokset identtiset.
* Ellipsihaku vektoroituna (normaaliyhtälöt keskitetyistä momenteista, eräratkaisu `np.linalg.solve`, virheessä vanha koodi):
  yhteensä **44,6 → 12,8 s** (`CALIB_VEC=0` = vanha). 9 moodikuvasta 8: identtinen H ja hoglinet; 1 (live_20261002_163845)
  vaihtoi ±20 cm -valinnan nimelliseen, koska mittarit ovat täsmälleen samat (RMS 1,530, pyöreys 0,958) ja `candidate_is_better`
  on tasapelissä liukulukuerojen takia.
* MAH00014 live-sim: elävä seuranta alkaa 145 s kohdalla (ennen 170 s).

Kokeiltu ja jätetty pois oletuksista:
* `STAB_EVERY=2` (stabilointi joka toiselle ruudulle): 2 heittoa 24:stä katosi → oletus 1.
* `LIVE_PROFIILI_UUSIN=1` (profiilin skannauspari uusimmasta ruudusta −5 s): profiili valmis 155 s vs 145 s, koska kalibroinnin
  aikana ohi menneet kivet ohitetaan → oletus 0.

## v6.16: taaksepäin liikkuvat radat pois + SIMD-maskit
**Mihin seurannan työ kuluu** (MAH00014, v6.15, kiven päivitykset luokiteltuina radan lopputuloksen mukaan):

| | ratoja | päivityksiä | seurannan ajasta |
|---|---|---|---|
| heitot | 24 | 43 % | 39 % |
| vahvistettu, ei heitto (pelaajat, lakaisijat, takaisin vietävät kivet) | 140 | 57 % | 60 % |
| vahvistamaton ehdokas | 13 | 1 % | 1 % |

Ei-heittoradat ovat enimmäkseen 1–10 m heitetystä kivestä 26–32 m:n alueella. Kokeillut erottelut:
* **Taaksepäin liikkuminen (käytössä, `TRACK_BACKWARD_STOP_CM=30`):** kivi ei liiku taaksepäin. Kaikilla 24 heitolla Y:n kasvu minkä tahansa
  sekunnin aikana on 0 cm; ei-heitoista suurella osalla 40–100 cm. Sääntö: viiden viimeisen ja sekuntia aiemman viiden paikan mediaanien ero > 30 cm
  → seuranta lopetetaan.
* Sovituksen laatu (rms, tarkka-osuus) ei erota: heittojen alussa (heittäjä pitää kivestä) rms 7–12 px ja lakaisijoiden peittäessä myöhemmin > 10 px.
* Haun tauottaminen heiton lentäessä: ei käy – seuraavan heiton rata alkaa usein, kun edellisen heiton/lakaisijan rata on vielä 10–26 m:ssä.
* Kaukohoglinen takana olevat radat heiton lentäessä: vain 5,6 %, ja riski (lakaisija voi olla kiven edellä) → ei käytössä.

**SIMD-maskit (`FAST_MASKS=1`):** graniittimaskin kynnys (`cv::compare`), etualamaski (`cv::inRange`) ja saturaatio (`cvtColor` BGR→HSV, S-kanava)
OpenCV:n vektorifunktioilla. Tulos tavutasolla sama (saturaatio tarkistetaan käynnistyksessä vanhaa laskentaa vastaan, ja nopeampi tapa valitaan
mittaamalla: tuloste `SEURANTA: nopeat maskit; saturaatio ...`). Windowsin MSVC ei välttämättä vektoroi vanhoja silmukoita (kamera-ajossa saturaatio
1,8 ms/kivi vs. sandbox 0,75 ms).

MAH00014 (sandbox, rinnakkain): samat 24 heittoa ja hog-tulokset; kivien päivitysten aika 478 → 442 s (−7,5 %), SEURANTA 18,0 → 17,1 ms/ruutu.
(Simuloitu säästö 10,6 %; osa lopetetuista pelaajista löytyy haulla uudelleen.)

## Avoimet / testattavaa kameralla
* Kameran todellinen fps ja pakkausmuoto (aja `tools/kamera_testi.py`), 1080p -> 720p -pienennyksen kustannus, käsittelynopeus vs. kameran fps
  (putki ~26-27 r/s debug-videolla Windows-koneella -> 25 fps:n kamera pysyy juuri ja juuri tahdissa).
* Kalibrointi tarvitsee alussa liikkuvan kiven (profiilin opettelu) kuten tiedostoajossakin.
* C++-moduulit (`mode_engine`, `stone_tracker`) pitää kääntää tässä kansiossa.

---

# Testi_05_02 (pohja: Testi_05_01 v5.0)

Kopio versiosta **Testi_05_01** (joka on kopio **Testi_03_04 v4.5**:stä) (variportti taustanvaimennukseen, siluettitarkennus 3x3-maskilla + kaistarajoitus, SEURANTA y > 20 m kaikille ruuduille, alfa-aariviiva profiilin opetteluun,
C++-tarkennus). Tässä kansiossa tehdään jatkossa isot lisäykset; Testi_03_04 jää vakaaksi viitteeksi.

* Koodi: `main.py`, `stone_tracker.cpp`, `mode_engine.cpp`, `haku_silhouette.py`, `kamera8_01.py`/`kamera9_0*.py`, `CMakeLists.txt` (sama rakenne kuin Testi_03_04).
* Työkalut ja kokeilut: `tools/` (mm. `seuranta_kokeilu.py`, `haku_kokeilu.py`, `run_profile.py`; kirjaukset `tools/SEURANTA_KOKEILU.md`, `tools/HAKU_KOKEILU.md`).
* Edelliset README:t: `README_alfa_profiili.md`, `README_pelialue_ja_profiili.md`, `README_edellinen_03_02.md`.
* `tulokset/` on tyhjä (Testi_03_04:n tulokset ja kuvat jäivät sinne).

Kääntäminen (Windows): `Remove-Item -Recurse -Force build; cmake -B build -S . -DCMAKE_BUILD_TYPE=Release; cmake --build build --config Release`.
Linux-testi: `g++ -O2 -std=c++17 -shared -fPIC $(python3 -m pybind11 --includes) $(pkg-config --cflags opencv4) stone_tracker.cpp -o stone_tracker.cpython-311-x86_64-linux-gnu.so $(pkg-config --libs opencv4) -lpthread`.

## Tunnetut avoimet asiat (Testi_03_04 v4.5:stä)
* Heitto 181 katoaa y ≈ 10 m:ssä (seuranta hyppää lakaisijan käsivarteen); hyppytesti (sivuttaishyppy/ruutu) testaamatta.
* Kahvan suojaus taustanvaimennuksessa jäällä; viivojen aiheuttamat kolot kiven kohdalla (täyttö viivamaskilla/raakakuvasta testaamatta); kiven varjo sinisen mainoksen päällä.
* Maski tummien kohteiden vieressä (tarkka vs 4x pienennetty sumennus), LM-tarkennuksen epäonnistuminen radan alussa (tarkka = 0).

## Hog-hog -analyysi (uusi, `hog_analyysi.py`, `HOG_ANALYSIS=1` oletus)
Kun kivi on kulkenut riittävästi (Y ≤ lähihog + 50 cm), radan pisteet väliltä **[lähihog + 50 cm, kaukohog − 100 cm]** (873–2918 cm) sovitetaan toisen asteen yhtälöön Y(t) = a t² + b t + c.
Sovituksen jälkeen **10 huonoiten sopivaa pistettä pudotetaan ja sovitus tehdään uudelleen** (poistaa satunnaiset epäonnistumiset). Jos R = √R² > 0,99:
* nopeus kaukohoglinella (yhtälön mukaan, m/s),
* keskihidastuvuus sovitusalueella (m/s², toisen asteen yhtälöllä vakio 2a),
* hog-hog-aika (yhtälön mukaan, lähihog 50 cm sovitusalueen ulkopuolella → ekstrapolaatio).
Tulostuu terminaaliin (`[frame N] Hog-hog (kivi X) | …`), debug-videoon (`--debug`) lähemmän hoglinen luona radan sivuun `HOG_OVERLAY_SECONDS` (5 s) ajaksi, still-kuvana
`<csv>_hog_kivi<id>.png` ja taulukkona `<csv>_hog.csv`. Koko video (26 heittoa): v ≈ 2,05 m/s (1,5–2,2), hidastuvuus ≈ 0,076 m/s² (0,043–0,092), hog-hog ≈ 14,8 s (12,8–21,0; heitto 76 poikkeava, R = 0,994).

### X-suuntainen analyysi (osa hog-hog-analyysia)
Samoilla pisteillä (väli [lähihog + 50 cm, kaukohog − 100 cm]) lasketaan **Y_yht = Y:n yhtälö hetkellä t** ja X sovitetaan toisen asteen yhtälöön Y_yht:n suhteen:
X = p·u² + q·u + r, u = (Y_yht − kaukohog)/100 (10 huonointa pistettä pois + uudelleensovitus). Jos R_x > 0,99:
* **suunta kaukohoglinella:** dX/dY kaukohogilla → kulma kulkusuunnassa (astetta, `+X`/`−X`), sekä X kaukohogilla (yhtälön mukaan),
* **X jonka suora saisi lähemmällä T-viivalla:** kaukohogin suunta jatkettuna suorana (Y = lähipesän keskipiste, 182,9 cm).
**Mitään ei tulosteta** ellei R_y > 0,99, R_x > 0,99 ja R_y·R_x > 0,99 (suodattaa kaiken huonon). Koko video: 24/26 heittoa läpäisee (heitto 47: R_x = 0,9895, heitto 76: R_y·R_x = 0,9585 jäävät pois).

## Debug-video (uusi ulkoasu, `python main.py --debug`)
* Sisältö: **stabiloitu + valotasapainokorjattu** kuva, **ei maskeja / taustanvaimennusta** (aiemmin taustanvaimennettu `frame_u_for_tracking`); kivien ääriviivat ja tunnisteet piirretään edelleen päälle.
* **Käännetty 90° vastapäivään** (kivet kulkevat ylhäältä alas), skaalattu korkeuteen 1080 (leveys 608 px 1280×720-lähteellä).
* Vasemmalla paneeli (430 px) **vasemmalle lähteneistä** heitoista, oikealla **oikealle lähteneistä** (suunta kaukohoglinella; +X-puoli lasketaan kameran projektiosta). Ylimpänä viimeisin, vanhemmat rullaavat alas kunnes eivät mahdu (6 laatikkoa/puoli).
* Laatikko per heitto (fontti 0,6 / paksuus 2 kuten aiemmin): `kiven ID`, `nopeus` (kaukohogilla), `hidastuvuus` (keskimääräinen), `hog-hog` (aika), `hogilla` (X kaukohogilla), `merkki` (X jonka kiven suora kaukohogilla saisi lähemmällä T-viivalla).
* Laatikko ilmestyy kun analyysi on valmis (kivi Y ≤ lähihog + 50 cm ja R_y, R_x, R_y·R_x > 0,99). Aiempi 5 s tekstiylikate poistettiin videosta (still-kuva `<csv>_hog_kivi<id>.png` säilyy).
Esimerkkiruutu: `tulokset/debug_video_uusi_ulkoasu.png`.


## Testi_05_02: LIUKU (uusi)
Suora X(Y) sovitetaan **heiton alusta kohtaan kaukohog + 100 cm** (kivi ei ole vielä ylittänyt hogia; vähintään 8 pistettä) ja sen perusteella lasketaan, **mihin suora osuisi lähemmällä T-viivalla** (Y = lähipesän keskipiste).
Paneelin ja terminaalin kohta `hogilla` korvattu nimellä **`liuku`** (arvo = suoran X T-viivalla alusta hog + 1 m, cm); `merkki` on edelleen hogin jälkeisen (sovitusvälin) suoran X T-viivalla.
Terminaalissa lisäksi liu'un suunta (astetta) ja pistemäärä; `<csv>_hog.csv`: `liuku_x_tee_cm`, `liuku_dir_deg`, `liuku_n`, `liuku_rms_cm`.
Koko video (24 heittoa): liuku −156…+186 cm (hajonta 134), merkki −184…+218 cm; hogin jälkeinen suunta on keskimäärin 0,86° jyrkempi kuin alun (irrotus), X T-viivalla siirtyy keskimäärin 41 cm; etumerkki sama 24/24, korrelaatio 0,993.
Esimerkki: `tulokset/debug_video_liuku.png`.

## Debug-videon nopeutus (Testi_05_02)
Debug-videon kustannus **52,2 → 11,8 ms/ruutu** (kokonaisaika/ruutu 86,4 → 54,0 ms; ruudut 0–2830, sama kone, ajettu rinnakkain):
* `apply_photometric_correction` käyttää kanavakohtaista **LUT:ia** (`cv2.LUT`, välimuistissa) float32-koko-ruutu-numpyn sijaan: 9 → 0,8 ms, tulos täsmälleen sama (tarkistettu: max ero 0). Nopeuttaa myös muita kutsujia.
* `DebugComposer`: pysyvä canvas, **paneelit piirretään uudelleen vain kun sisältö muuttuu**, skaalaus ennen kääntöä (INTER_LINEAR; kääntö pienemmälle kuvalle).
* `AsyncVideoWriter`: mp4-enkoodaus omassa säikeessä (`cv2.VideoWriter.write` vapauttaa GIL:n).
Ulkoasu ja hog-csv ennallaan.

## Pullonkaula-analyysi (nopeusraportin alussa + tiivis rivi edistymistulosteessa; `PULLONKAULA_RAPORTTI=0` pois)
Liukuhihnan vaiheet A (luku+stabilointi), B (warp+varjosuodatus) ja C (pääsäie: HAKU/SEURANTA/CSV/debug) ajavat rinnan; nopeuden määrää **hitain vaihe** (suurin palveluaika ms/ruutu = pienin kapasiteetti r/s).
Raportti kertoo per vaihe palveluajan, kapasiteetin, odotuksen syötteeseen (tyhjä jono) ja tulosteeseen (jono täynnä), kuormituksen, jonojen keskitäytön (B→C täynnä = pääsäie ei ehdi ottaa; tyhjä = ylävirran vaihe hitain),
prosessin CPU-käytön (ytimiä keskimäärin) ja pullonkaulavaiheen suurimmat osat + vihjeen. Pääsäikeen oma CPU% on matala C++-kutsun aikana (odottaa C++:n työsäikeitä) - käytä palveluaikaa ja jonojen täyttöä.

### Nopea nopeustesti omalla koneella
* `python main.py --max-frame 3000 --no-debug` → ajo pysähtyy ruutuun 3000 ja tulostaa nopeus- ja **pullonkaula-analyysin** (ilman debug-videota).
* `python main.py --max-frame 3000 --debug` → sama debug-videon kanssa (vertaa pääsäikeen palveluaikaa ja kapasiteettia).
* Edistymisrivit sisältävät tiiviin rivin `pullonkaula: ...`; lopussa täysi raportti (vaiheet A/B/C: palveluaika, kapasiteetti, odotukset, jonojen täyttö, CPU, pullonkaulavaiheen suurimmat osat + vihje).
* `PULLONKAULA_RAPORTTI=0` kytkee raportin pois. Lisäksi `--start/--end` leikkaa videosta pätkän (kuten ennen).

## v5.2 – debug-videon piirto taustasäikeeseen

Debug-videon koko piirto (valokorjaus, ääriviivat, kokoonpano, kirjoitus) ajetaan `AsyncVideoWriter`-säikeessä (`submit`).
Pääsäie antaa vain tilannekuvan (frame_u, ääriviivalista, kopio hog_results). Sandbox-mittaus (2900 ruutua):
pääsäikeen debug-aika ~8 ms → 0,1 ms/ruutu; C-vaihe 27–30 ms. Seuraavat rajoittajat: SEURANTA (C++) ja stabilointi (A, ~29 ms).

## Testi_05_03 v5.3 – seuranta vain joka toisella ruudulla kun Y < 15 m

Kun kivi on lähempänä kuin `SEURANTA_HALF_RATE_Y_CM` (oletus 1500 cm; 0 = pois; ympäristömuuttuja), sitä seurataan vain parillisilla ruuduilla.
Kaukana (Y ≥ 15 m) mittaus on epätarkempaa, joten siellä käytetään kaikki ruudut; lähellä riittää harvempi näytteistys.
- Ohitettu kivi pysyy aktiivisena; ohitetut ruudut kirjataan `s["gap_extra"]`:iin (nollataan kun kivi löytyy), ja hakualue sekä liike-ennuste kasvavat vastaavasti (`misses+1+gap_extra`).
- Ohitetulla ruudulla debug-videossa piirretään kiven viimeisin paikka (vihreä).
- CSV:hen ja hog-hog-analyysiin tulee lähialueelta puolet vähemmän pisteitä (analyysin alue 8,7–29 m, josta 15–29 m täydellä tiheydellä).
- Muu koodi = Testi_05_02 v5.2.

### Testitulos (sandbox, koko video, ilman debug-videota; vertailu = sama koodi `SEURANTA_HALF_RATE_Y_CM=0`)
- Nopeus 34,6 → 35,4 r/s (sandboxissa A-vaihe rajoittaa 40 r/s; pääsäikeen kapasiteetti 42,7 r/s ennen: SEURANTA 27,6 ms/kutsu, 20,3 ms/ruutu).
- Heitot 26 / 26, hog-hog 23 → 26. Samat 23 heittoa: nopeus kaukohogilla ka ero −0,003 m/s (max 0,006), suunta −0,01°, suoran X T-viivalla ka −0,5 cm (max 5,2 cm). Liuku (alusta hog+1m) identtinen.
- Korjattu: pysähtymisehto (ikkuna = stop_tracking_frames) ei täyttynyt parillisilla ruuduilla → sallitaan 1 ruudun vajaus (ilman tätä pysähtyneet kivet täyttivät paikat ja heittoja katosi).

### Debug-videon kirjoitus kahteen säikeeseen (v5.3b)
Windows-ajossa pääsäikeen debug-aika oli 7,2 ms/ruutu ja C-vaihe pysyi ~42 ms vaikka SEURANTA nopeutui 26,9 → 22,7 ms – epäilys: yhden taustasäikeen
piirto+koodaus (sandbox 10,6+14,7 ms, kuormitettuna hitaampi) ei pysy ruudun tahdissa ja pääsäie odottaa täyttä jonoa. Nyt piirto ja `VideoWriter.write`
ovat omissa säikeissään (jonot 8). Sandbox (2900 ruutua, debug päällä): video ehjä (2900 ruutua, 1468x1080), pääsäikeen debug 0,14 ms, 31,3 r/s.

### Stabiloinnin (vaihe A) nopeutustutkimus – `tools/bench_stabilointi.py`
`python tools/bench_stabilointi.py <video> [ruutuja]` mittaa omalla koneella: CPU-polun vaiheet, `cv2.setNumThreads`-vaikutuksen, usean ruudun rinnakkaisajon läpimenon (1–4 säiettä)
ja OpenCL/UMat-polun (esim. Intel UHD Graphics) sekä vertaa GPU-tuloksen CPU:hun.
Sandbox (1280x720, 4 ydintä, ei OpenCL:ää): CPU-polku ~7 ms/ruutu; 1 → 4 säiettä: 147 → 420 r/s (ruudut ovat toisistaan riippumattomia, koska jokainen verrataan moodikuvaan).
Keskitetyn rajauksen testi (ei pienennystä): 1024x576…800x450 -rajaus nopeutti 12 → 8…3,5 ms, mutta siirtymävirhe oli rms 0,13–0,19 px (max 0,7–0,8 px) -> ei käyttöön.

### Benchmark-tulos Windows-koneella (Intel UHD Graphics, 1280x720, OpenCV 4.12, OpenCL käytössä)
- CPU-polku yksin: 11,5 ms/ruutu (ikkuna 1,4 + DFT 3,3 + vaihe 1,7 + käänteis-DFT 4,3 + huippu 0,4). `cv2.setNumThreads` ei vaikuta DFT:hen.
- Usea ruutu rinnan: 1 säie 86 r/s, 2 säiettä 130, 3–4 säiettä 140 r/s (kyllästyy ~2–3 säikeeseen).
- OpenCL/UHD: 15,5 ms/ruutu eli hitaampi kuin CPU -> ei käyttöön (tulos identtinen).
- Putkessa vaihe A kestää silti 29–41 ms vs. 11,5 ms yksin -> ero johtuu kilpailusta muiden säikeiden kanssa (SEURANTA:n C++-säikeet, B), ei laskennasta.
- Koeajoon `CV_THREADS=n` (cv2.setNumThreads) ja bench-työkalun kohta 5 (warpAffine+remap CPU vs OpenCL).

### Debug-videon laitteistokoodaus (Intel Quick Sync)
`hog_analyysi.open_debug_writer`: debug-video koodataan ffmpeg-putken kautta `h264_qsv`-enkooderilla (grafiikkapiirin media-moottori) CPU:n sijaan.
`DEBUG_ENCODER=auto|qsv|x264|opencv` (oletus auto: kokeilee QSV:tä 3 mustalla ruudulla, muuten cv2.VideoWriter mp4v). Käytetty enkooderi tulostuu ajon alussa.
Perustelu: GPU ei auta DFT:ssä (15,5 vs 11,5 ms) eikä SEURANNASSA; koodaus on sen sijaan laskentaa jonka media-moottori tekee CPU:ta kuormittamatta,
ja debug-video hidasti kaikkia vaiheita ~20 % (26,0 → 20,5 r/s). Sandbox (ei QSV:tä): putki toimii (x264) ja varapolku (mp4v) toimii.
Tarkista oma ffmpeg: `ffmpeg -hide_banner -encoders | findstr qsv`.

### GPU-ristikkohaku (OpenCL, esim. Intel UHD) – oletuksena pois
`GPU_GRID=1` ottaa käyttöön SEURANNAN ristikkohaun HIENON vaiheen (±coarse-askel, 1,5 cm askel -> tyypillisesti 10×10 = 100 ehdokasta, kaikki pisteytetään) GPU:lla.
- OpenCL-kirjasto (OpenCL.dll / libOpenCL.so) ladataan ajonaikaisesti: ei build-riippuvuutta; jos laitetta ei löydy, käytetään CPU-polkua ja tulostetaan syy.
- Isäntä (CPU) laskee ehdokkaiden kokonaislukumonikulmiot täsmälleen kuten CPU-polku (projektio, trunkointi, marginaalihull); GPU-ydin rasteroi ne **täsmälleen kuten `cv::fillPoly`** (Bresenham-reunat + täyttö 16.16-kiintopisteellä, dx katkaistuna kuten OpenCV:ssä) ja laskee pikselimäärät
  (rivikohtaiset kumulatiiviset maskisummat -> ei pikselisilmukkaa); pistemäärä muodostetaan isännässä samalla kaavalla kuin CPU:lla. Leikkautuvat/ei-kuperat ehdokkaat pisteytetään CPU:lla (`hullOverlapScore`).
- Verifiointi: `GPU_GRID_VERIFY=1` laskee myös CPU-pistemäärät kaikille ehdokkaille ja kerryttää tilastoa. Sandbox (OpenCL-laite = pocl/CPU, 6851 ristikkoa / 685 100 ehdokasta): 0 eri pistemäärää, 0 eri voittajaa; stones.csv identtinen CPU-polun kanssa.
- `GPU_GRID_DEVICE=gpu|cpu|any` (oletus gpu). 3 peräkkäistä OpenCL-virhettä -> GPU pois käytöstä (CPU-varapolku).
- Pullonkaularaportissa C++-puolen rivit "ristikko: GPU-polku" + alarivit (isäntälaskenta / OpenCL-kutsu / pisteytys). Vertaa ristikkohaun CPU-aikaan ("haku: ristikko (yhdistelma...)").
- Käännös (Windows): C++-moduuli pitää kääntää uudelleen (`stone_tracker.cpp`: ei CMake-muutoksia; `windows.h` mukaan LoadLibrary-kutsua varten).
- Kokeilu: `$env:GPU_GRID=1; python main.py --max-frame 3000 --no-debug` vs. ilman; lisää `$env:GPU_GRID_VERIFY=1` kerran oikeellisuuden tarkistamiseen.

### Vertailuajo (GPU vs CPU) – `tools/vertailuajo.py`
`python tools/vertailuajo.py --video <alkuperäinen video> [--start 00:10:00 --end 00:21:00] [--outdir ...] [--no-debug] [--runs verify,gpu,cpu] [--main-args "--max-frame 3000"]`
Ajaa main.py:n kolmesti (verify = GPU+CPU-vertailu, gpu, cpu; oletuksena debug-video päällä) ja tallentaa kansioon (oletus `<videon kansio>/vertailuajo_<aikaleima>`): jokaisen ajon koko lokin, sijainti-CSV:t, ja
`vertailu_yhteenveto.txt` (nopeudet, pullonkaularaportit, GPU-tilasto, CSV-vertailu). Ilman `--video`-argumenttia avataan tiedostovalitsin kerran.
Lisäksi `main.py --video <polku>` ohittaa tiedostovalitsimen.

### v5.4: GPU-vaihe B (`GPU_B=1`) + joka toisen ruudun seuranta pois oletuksena
- **Puolitus pois:** `SEURANTA_HALF_RATE_Y_CM` oletus on nyt 0 (ei nopeuttanut mitään, mutta heikensi laatua: 24 -> 22 heittoa). Voi kytkeä: `SEURANTA_HALF_RATE_Y_CM=1500`.
- **GPU-vaihe B** (OpenCL, esim. Intel UHD; oletuksena pois): `warpAffine` + `remap` + valotasapaino + varjotoleranssi-taustanvaimennus ajetaan GPU:lla. Pullonkaula oli vaihe B (38–40 ms, 92 % kuormitus), ja sen CPU-aika kasvoi putkessa 4× yksittäismittauksesta.
  - Tulos on **bitti-identtinen** OpenCV 4.x:n CPU-polun kanssa: ydin toistaa OpenCV:n kiintopisteisen bilineaarisen interpoloinnin (AB_BITS=10, INTER_BITS=5, 15-bittiset painot, painotaulukon korjauskuvio) ja `suppress_shadow_background`:n kokonaislukulaskennan
    (sävylaskenta float-taulukolla ja ilman FMA-yhdistelyä). Testattu OpenCV 4.12:ta vastaan (28 kuvaa × 4 siirtymää, IPP päällä/pois): 0 eri arvoa. (OpenCV 5.0:n warpAffine/remap eroaa 4.x:stä – ei tuettu.)
  - Rajapinta: `gpu_b_init(map1, map2, ref, ...)`, `gpu_b_warp(frame, M)`, `gpu_b_suppress(gains, biases, H, W)`; `main.py` käyttää niitä `_LivePrep.process`:ssa, virheessä palataan CPU-polkuun.
  - GPU-polku siirtää kuvat GPU:lle ja takaisin (frame_u ja frame_for_tracking ladataan takaisin, koska pääsäie, debug-video ja muu koodi käyttävät niitä).
- `tools/vertailuajo.py`: oletus nyt `cpu,gpub,gpuall` (muut: `gpu`, `verify`).

### v5.5: liukuhihnan rinnakkaisuus (`STAB_WORKERS`, `PIPE_DEPTH`)
Perustelu (Windows-vertailuajo, GPU-vaihe B): B 37,7 -> 18,7 ms, CPU-kuorma ja lämpö laskivat (4,64 -> 3,64 ydintä), mutta nopeus pysyi 22,1–22,9 r/s, koska stabilointi (A, 34–37 ms) ja pääsäie (C, 38–40 ms) ovat nyt yhtä hitaat ja aikavaihtelu hukkaa kapasiteettia pienillä jonoilla.
- `STAB_WORKERS=n`: stabilointi (vaihekorrelaatio) n ruudulle rinnan; ruudut ovat toisistaan riippumattomia (kukin vs. moodikuva), tulokset jonoon alkuperäisessä järjestyksessä -> tulos identtinen (sandbox 2900 ruutua: CSV identtinen, 31,1 -> 34,1 r/s kun STAB_WORKERS=3, PIPE_DEPTH=6). Oletus 1 (ennallaan). A:n palveluaika raportissa = A-säikeen oma kierto; työntekijöiden yhteenlaskettu aika "bg: stabilointi tyoaika"-rivillä.
- `PIPE_DEPTH=n`: vaiheiden välisten jonojen koko (oletus 3).
- `tools/vertailuajo.py`: uudet ajot `par` (GPU_B + 2 rinnakkaista + jonot 6), `par3`, `parcpu`; oletus `cpu,gpub,par,par3`.

### v5.6: kiven sisäinen rinnakkaisuus (`INTRA_PARALLEL=1`)
Windows-ajossa (par3) pääsäie C oli ainoa pullonkaula (42 ms, 100 % kuormitus; A 4 ms, B 19 ms): SEURANTA 25 ms/ruutu = 60 % siitä. Kiven kolme hakua – ristikkohaku (~4–5 ms) ja kaksi mean-shiftiä (~3 ms kumpikin) – ovat toisistaan riippumattomia,
mutta ajettiin peräkkäin. `INTRA_PARALLEL=1` ajaa mean-shiftit omissa säikeissään ristikkohaun rinnalla (`locate_mode 5`, `stone_tracker.set_intra_parallel`); syötteet, ehdokkaiden järjestys ja valintalogiikka ovat samat -> tulos identtinen.
Sandbox (4 ydintä, jo kyllästetty): CSV identtinen (2900 ruutua), SEURANTA-kutsu 26,6 -> 26,1 ms; hyöty odotetaan vasta koneella jossa on vapaita ytimiä (Windows-kone: 3,9/8 ydintä käytössä).
`tools/vertailuajo.py`: uudet ajot `intra`, `intragrid`; oletus `cpu,par,intra,intragrid`. C++-moduuli pitää kääntää uudelleen.
`tools/vertailuajo.py --outdir <aiempi kansio> --runs par,intra,intragrid`: jos kansiossa on aiemman ajon `ajo_cpu_loki.txt` + `ajo_cpu_sijainnit.csv`, cpu-perusajoa ei tarvitse ajaa uudelleen; se otetaan yhteenvetoon ja CSV-vertailuun.

### v5.7: nopeutus ilman tulosmuutosta (CSV:t bitti-identtiset v5.6:n kanssa)
Tavoite: nopeampi ilman laadun heikkenemistä. Jokainen muutos tarkistettiin ajamalla koko MAH00014-video (`--debug`) ja vertaamalla
`kivien_sijainnit.csv`, `_raaka.csv` ja `_hog.csv` tavu tavulta v5.6:n tulokseen: **identtiset** (24 heittoa, hog-analyysi 22).
Sandbox (4 ydintä, Linux), koko video debug-videolla: **476,5 s -> ~390-400 s** (VM:n nopeus vaihtelee ajojen välillä ~5-10 %).

1. **NumPyn BLAS yhdelle säikeelle** (`main.py`:n alku, ennen numpy/cv2-importia: `OPENBLAS_NUM_THREADS`/`OMP_NUM_THREADS`/`MKL_NUM_THREADS` = 1).
   perf-profiloinnissa ~20 % koko prosessin CPU-ajasta kului OpenBLASin työsäikeiden tyhjäkäyntiin (`blas_thread_server` + `sched_yield`;
   Windowsin OpenBLAS pyörii samoin) - säikeet heräävät jokaisesta pienestä matriisikutsusta (3x3, polyfit, lstsq) ja vievät ytimiä
   SEURANNAN C++-säikeiltä. Koodin matriisit ovat pieniä, joten yksi säie ei hidasta mitään. 476,5 -> 397,6 s. `BLAS_THREADS=n` palauttaa.
2. **Pääsäikeen keventäminen** (pääsäie C on pullonkaula; py-spy: ~87 % sen ajasta SEURANTA-kutsussa):
   * SEURANNAN siluettitarkennus kaikille löydetyille kiville kerralla, kivet rinnan C++:ssa (`silhouette_refine_batch_cpp`, GIL vapaana;
     sama laskentaydin `silhouetteRefineCore` kuin `silhouette_refine_cpp`:ssä) kivi kerrallaan -silmukan sijaan.
   * SEURANNAN värivertailu (`color_match_median_diff` -> `color_diff_history`) lasketaan vain `COLOR_DEBUG`-tilassa: se ei vaikuta
     mihinkään päätökseen eikä tulosteeseen (poistettu "pysähtynyt"-päätöksestä jo aiemmin), mutta maksoi ~1 ms/ruutu pääsäikeessä.
3. **Kalibrointivaihe** (kertakustannus jokaisessa ajossa, sandbox ~73 -> ~61-63 s):
   * `fit_stone_profile_cpp`: Jacobin sarakkeet ja kivien residuaalilohkot rinnan (kukin kirjoittaa vain omaan kohtaansa; summausjärjestys ennallaan) 10,3 -> 3,0 s.
   * Profiilin opettelun seuranta (`track_stone_in_video_windowed`): remap + kandidaattiskannaus `SCAN_WORKERS` (oletus 3) säikeessä rinnan, tulokset ruutuindeksillä.
   * Alfa-havaintojen ja värireferenssin ruudut luetaan eteenpäin (`grab`) jos seuraava ruutu on <= `SEEK_MAX_SKIP` (25) ruudun päässä:
     seek maksoi ~50 ms/havainto, eteenpäin luku ~1,7 ms/ruutu; ruudut tarkistettu identtisiksi.

Valinnaiset ajoitussäädöt (eivät muuta tuloksia; oletuksena pois, koska sandboxissa ei hyötyä - kokeile omalla koneella `tools/vertailuajo.py`:llä):
* `BG_PRIORITY=1`: liukuhihnan A/B-säikeet, videon luku, valotasapaino ja debug-videon piirto/kirjoitus alemmalle prioriteetille
  (Windows `THREAD_PRIORITY_BELOW_NORMAL`) -> SEURANNAN säikeet saavat ytimet ensin. Sandboxissa (kaikki 4 ydintä käytössä) C nopeutui mutta A hidastui.
* `PY_SWITCH_INTERVAL_MS=1`: Pythonin GIL-vaihtoväli 5 ms -> 1 ms (pääsäie saa GIL:n nopeammin takaisin C++-kutsun jälkeen). Sandboxissa ei vaikutusta.

Tutkittu, ei otettu käyttöön: `cv2.remap` kiintopistekartoilla (identtinen, ei nopeampi), `CV_THREADS=1/2` (ei hyötyä),
debug-videon koodaus ffmpeg-putkella (mpeg4/x264, BGR tai valmis YUV) - nykyinen `cv2.VideoWriter` mp4v oli CPU:lla halvin (7,8 ms/ruutu).
`tools/vertailuajo.py`: uudet ajot `blasmt`/`intrablasmt` (vanha BLAS-käytös vertailuun), `intraprio`, `intrasw`.
C++-moduuli pitää kääntää uudelleen (`stone_tracker.cpp`).

### v5.8: debug-video ffmpegille valmiina YUV:na (`DEBUG_YUV=1`, valinnainen)
QSV-/x264-putkessa ffmpeg muunsi BGR-ruudut YUV:ksi omalla (hitaalla) swscalellaan CPU:lla. `DEBUG_YUV=1`: muunnos tehdään kirjoitussäikeessä
`cv2.cvtColor(BGR2YUV_I420)`:lla (SIMD) ja ffmpegille syötetään yuv420p -> putkeen puolet vähemmän dataa. Sandbox-mittaus (mpeg4/x264): ~2,5-3 ms/ruutu
vähemmän CPU:ta. Värit voivat poiketa aavistuksen (PSNR BGR-putkeen 42-45 dB). Ei vaikuta seurantaan (CSV identtinen). Ei koske cv2.VideoWriter-varapolkua.
`tools/vertailuajo.py`: ajo `intrayuv` (= intra + DEBUG_YUV=1).

### v5.9: nopein asetus oletukseksi + `SEURANTA_CV_THREADS`
Windows-vertailuajo (MAH00014, debug päällä, kaikki CSV:t identtiset): intra 640,5 s, intrablasmt 669,9 s (vanha BLAS), intraprio 692,0 s (huonompi),
intrasw 648,4 s, intrayuv 631,7 s (nopein). Uudet oletukset = intrayuv: **`GPU_B=1`, `STAB_WORKERS=2`, `PIPE_DEPTH=6`, `INTRA_PARALLEL=1`, `DEBUG_YUV=1`**
(GPU_B palaa CPU:lle jos OpenCL-laitetta ei löydy; DEBUG_YUV koskee vain ffmpeg-putkea). Vanhat arvot saa ympäristömuuttujilla (esim. `GPU_B=0`).
`tools/vertailuajo.py`: uudet ajot `oletus`, `oletusgrid`. Vanhat ajot (cpu, par, intra, ...) ajetaan
vanhoilla oletuksilla, jotta niiden merkitys ei muutu. C++-moduuli pitää kääntää uudelleen.

### v5.10: kiven sisäistä rinnakkaisuutta lisää (`GRID_THREADS`, `PREP_PARALLEL`)
Windows-vertailuajo v5.9: oletus 657,0 s, oletusgrid (GPU_GRID=1) 707,7 s -> GPU-ristikko ei kannata.
v5.9:n kokeiluasetus `SEURANTA_CV_THREADS` (OpenCV:n säiemäärä > 1 SEURANNAN aikana) **poistettu**: Windowsissa ajo kaatui
(`mode_engine.read`: "Unknown exception") - säiemäärän vaihtaminen joka ruudulla samalla kun videon luku käyttää samaa OpenCV-kirjastoa
rikkoo säiepoolin. Ennen kaatumista kirjoitetut sijainnit olivat identtiset.
Windows-mittauksen kivikohtainen ketju (INTRA_PARALLEL päällä): valmistelu ~7 ms -> haku max(ristikko 6,2 ms, mean-shiftit ~2,9 ms) -> LM 13,8 ms.
* `GRID_THREADS=2` (oletus): ristikkohaun hieno vaihe (~100 toisistaan riippumatonta ehdokasta) kahdelle säikeelle; kumpikin käy oman X-lohkonsa
  alkuperäisessä järjestyksessä ja tulokset yhdistetään samalla "aidosti suurempi voittaa" -säännöllä -> sama tulos. `GRID_THREADS=1` = ennallaan.
* `PREP_PARALLEL=1` (oletus): etualamaski (riippuu vain taustavaimennetusta rajauksesta) omassa säikeessään saturaation ja graniittimaskin rinnalla.
`tools/vertailuajo.py`: ajot `v59` (= v5.9:n oletukset), `oletus`, `grid3` (GRID_THREADS=3); oletuksena ajetaan nämä kolme.
C++-moduuli pitää kääntää uudelleen.

### v5.11: pääsäikeen nopeutus (tavoite >= 10 %), tulokset identtiset
Windows-loki v5.9 (oletus): pääsäie C 36,1 ms/ruutu = SEURANTA 26,6 + HAKUn odotus 2,4 + siluettitarkennus 1,5 + muu; kiven ketju (INTRA_PARALLEL)
valmistelu ~7 ms -> haku (ristikko 6,2 ms) -> LM 13,8 ms (josta 200 px:n reunustuksen kopio 1,4 ms). Muutokset (kaikki oletuksena päällä, 0 = pois):
1. `HAKU_AHEAD=1`: HAKU käynnistetään jo liukuhihnan vaiheessa B heti kun ruutu on valmis (syöte riippuu vain ruudusta ja kalibroinnista;
   EVICT_AT_CAP-oletuksella HAKU-ehto riippuu vain ruudun indeksistä) -> tulos on valmis kun pääsäie ehtii ruutuun, odotus ~0.
2. v5.10 `GRID_THREADS=2`, `PREP_PARALLEL=1` (katso yllä).
3. `SIL_IN_BATCH=1`: SEURANNAN siluettitarkennus lasketaan `track_stones_batch`:in kivisäikeessä heti kiven päivityksen jälkeen
   (sama `silhouetteRefineCore`, sama kuva ja syöte; `set_seuranta_silhouette`) -> ei erillistä vaihetta pääsäikeessä.
4. LM:n reunustus ilman kopioita: saturaatio luetaan `PaddedView`-näkymän kautta (sama koordinaatisto ja nollat kuin 200 px:n
   `copyMakeBorder`-kopiossa), maskiin 1 px:n nollareunus ja ääriviivan pisteet siirretään takaisin ennen pinta-ala/momenttilaskentaa.
   Ei kytkettävissä (identtinen).
Diagnostiikka: käynnistyksessä tulostuu `C++-moduulin OpenCV: ...` (Baseline/Dispatched = SIMD-optimoinnit, Intel IPP, säiekehys).
Graniittimaski (cvtColor + GaussianBlur) oli Windowsissa 4,7x sandboxia hitaampi, kun muut vaiheet ~2x -> epäily: vcpkg:n OpenCV ilman AVX2/IPP:tä.
`tools/vertailuajo.py`: oletuksena ajot `v59` (vanha) ja `oletus` (uusi). C++-moduuli pitää kääntää uudelleen.

### v5.12: OpenCV:n säiepoolia ei luoda uudelleen joka kutsulla (`CV_SINGLE_PERSIST=1`) + saturaatio rinnan
Windows-vertailuajo v5.11: oletus 611,6 s vs v59 650,9 s (-6,0 %), CSV:t identtiset. Pääsäie 36,3 -> 34,0 ms/ruutu.
OpenCV-diagnostiikka (Windows, vcpkg 4.12): AVX2 dispatch mukana, **ei IPP:tä**, `Parallel framework: Concurrency` (ConcRT).
* IPP:tä ei suositella: sandbox-testissä IPP muutti `cv2.resize(INTER_LINEAR)`-suurennuksen float-tuloksia (~2 milj. eri arvoa) eikä ollut nopeampi.
* `CV_SINGLE_PERSIST=1` (oletus): `ScopedSingleThreadedOpenCV` vaihtoi C++-OpenCV:n säiemäärää 1 <-> 8 jokaisella SEURANTA-/HAKU-kutsulla;
  ConcRT-taustalla jokainen `cv::setNumThreads` luo uuden ajastimen/säiepoolin (pthreads-taustalla halpa, siksi ei näkynyt sandboxissa;
  sama mekanismi kaatoi v5.9:n `SEURANTA_CV_THREADS`-kokeilun). Nyt elävän seurannan alussa asetetaan 1 säie kerran eikä palauteta.
  Kutsut ajettiin jo ennestään yhdellä säikeellä -> sama tulos. (Pythonin cv2 on erillinen kirjasto, ei vaikutusta A/B-vaiheisiin.)
* `PREP_PARALLEL=1` laajennettu: saturaatio lasketaan omassa säikeessään graniittimaskin tummuusosan (harmaasävy + taustan sumennus) rinnalla
  (`graniteDarkness` + `createGraniteMaskFromParts`; `createGraniteMask` käyttää samoja osia).
Koko MAH-video: CSV:t tavu tavulta identtiset v5.6:n kanssa. `tools/vertailuajo.py`: oletuksena `v59`, `v511` (= CV_SINGLE_PERSIST=0), `oletus`.
C++-moduuli pitää kääntää uudelleen.

### v5.13: saturaation rinnakkaistus pois oletuksista (`SAT_PARALLEL=0`)
Windows-vertailuajo v5.12: v59 652,0 s, v511 (= CV_SINGLE_PERSIST=0, saturaatio rinnan) 654,7 s, oletus 622,4 s; CSV:t identtiset.
* ConcRT-korjaus toimii: v511 -> oletus -4,9 %.
* Saturaation rinnakkaistus hidasti: ainoa ero v5.11:n (611,6 s) ja v5.12:n v511-ajon (654,7 s) välillä (~+7 %; ristikkohaun CPU-aika/kutsu 6,6 -> 10,2 ms)
  -> lisäsäie kilpailee hypersäikeistä/muistikaistasta jo valmiiksi rinnakkaisessa kivilaskennassa. Nyt `SAT_PARALLEL=1` erikseen, oletus pois.
Odotus: v5.11 (611,6 s) + ConcRT-korjaus (-5 %) ~ 580 s = ~-11 % v59:ään nähden.
`tools/vertailuajo.py`: oletuksena `v59, oletus, satpar, noprep, grid1` (kukin = oletus yhdellä muutoksella -> rinnakkaistusten erillisvaikutus);
lisäksi `nointra` (INTRA_PARALLEL=0). C++-moduuli pitää kääntää uudelleen.
