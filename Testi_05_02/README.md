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
