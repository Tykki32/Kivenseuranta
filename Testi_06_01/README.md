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

## v6.9: profiilihavaintojen ruudut kiinnitetään puskuriin
Live-simulaatio kaatui: `ruutu 1489 on jo poistettu puskurista (vanhin 1707)` kiven pintavarireferenssiä rakennettaessa.
Puskuri säilyttää 40 s historiaa, mutta täyttyessään (kalibroinnin viive 52 s) karsii sen 31 s:iin; profiilihavainnot olivat
~40 s taaksepäin. Nyt havaintojen ruudut kiinnitetään (`FrameStore.pin`) heti kun ne kerätään, ja vapautetaan
pintavarireferenssin jälkeen (`unpin_all`). Lisäksi `BufferedCapture.grab()` ei enää hae ohitettavaa ruutua
(eteenpäinluvussa välissä olevat ruudut saavat olla jo poistettu). Sandbox: MAH00014 `--live-sim --live-taakse-s 31` läpi.

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
