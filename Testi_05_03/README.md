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
