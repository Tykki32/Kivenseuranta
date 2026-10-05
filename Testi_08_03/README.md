# Testi_08_03 – siivottu versio (pohja Testi_08_02 t16)

Sama toiminta kuin Testi_08_02:ssa (t16: irroitus, vasen-/oikeakätinen, viiveellä näytetty keskikuva, puhelinnäkymän
tallennus), mutta koodi on siivottu:

* Python jaettu aiheittain omiin moduuleihin (kalibrointi, profiili, seuranta, heitot, näkymä, …), kuollut koodi ja
  kokeilutilat poistettu.
* **Kaikki säädettävät arvot yhdessä tiedostossa: `asetukset.py`** (ei enää ympäristömuuttujia).
* C++ jaettu aiheittain `cpp/`-kansioon, kokeilutilat ja käyttämättömät funktiot poistettu.
* Rivimäärä: Python 16 405 → 9 662, C++ 8 677 → 5 003.

Tulokset ovat samat kuin Testi_08_02:lla (ks. *Varmistus* alla).

## Käyttö

Komentorivi on sama kuin ennen:

```
python main.py --video MAH00014.mp4 --debug
python main.py --live --paneelit D:\Tikku\Suorita\MAH00014_leikattu_panel_corners.txt --katselu --live-kansio D:\Tikku\Live
python main.py --live-sim video.mp4 --live-sim-tahti 0 --paneelit paneelit.txt     # testaus ilman kameraa
python tools/kamera_testi.py                                                       # kameran kartoitus
```

`python main.py --help` listaa kaikki valitsimet (`--start/--end`, `--max-frame`, `--live-*`, `--katselu [portti]`).

## Asetukset (`asetukset.py`)

Kaikki käytön aikana säädettävät arvot (aiemmin hajallaan moduulien vakioina ja ympäristömuuttujina) on koottu
yhteen tiedostoon, osioittain:

| Osio | Sisältö |
|---|---|
| 1. Kalibrointi | moodikuvan näytteet (120 s / 5 s), paneelien seuranta, hoglinjojen toleranssi ±20 cm |
| 2. Kiviprofiilin opettelu | pelialue, siemen, esitarkistus, profiilin hyväksyntä, alfa-ääriviiva |
| 3. HAKU ja SEURANTA | hakualue, ristikon askeleet, kynnykset, uuden ehdokkaan suodatin, nopeusrajat, mean-shift ja ehdokkaan valinta, siluettituki |
| 4. Radan elinkaari | vahvistus, kadotus, pysähtyminen, lopetus, duplikaatit |
| 5. Heittoportti | mitkä radat kirjoitetaan heitoiksi CSV:hen |
| 6. Hog-hog, liuku, irroitus | sovitusalue, R-rajat, liuku ja hakki |
| 7. Kahva ja kierteet | kahvan värikynnykset, kierrearvio |
| 8. Paikallinen täysi resoluutio | live: SEURANTA kivien alueilla kameran täydellä resoluutiolla |
| 9. Esikäsittely | taustanvaimennus, varjo, jää, väriportti, valotasapaino, stabilointi |
| 10. Suorituskyky | säikeet, jonot, GPU-vaihe B, laitteistodekoodaus, raportointi |
| 11. Live-syöte | seurantavaiheen puskurihistoria (muut `--live-*`-valitsimilla) |
| 12. Puhelinnäkymä ja debug-video | kuvaväli, JPEG-koko, viive, liikemaski, risti, enkooderi, taustaprioriteetti |

Ympäristömuuttujat on poistettu kokonaan (kokeilutilat kiinnitetty testattuihin oletuksiin). Poikkeus: `GPU_LAITE=cpu|any`
valitsee OpenCL-laitteen GPU-vaiheelle B testausta varten (oletus GPU).

Kalibroinnin ja kuvantunnistuksen sisäiset vakiot (renkaiden värikynnykset, hakuruudukot, LM-parametrit, mallin muoto)
ovat edelleen omissa moduuleissaan, koska ne ovat algoritmin osia eivätkä käytön aikana säädettäviä.

## Rakenne

### Python

| Tiedosto | Sisältö |
|---|---|
| `main.py` | komentorivi, videon leikkaus, paneelien valinta, C++-asetukset, live-tilan käynnistys/lopetus |
| `asetukset.py` | kaikki säädettävät arvot |
| `seuranta.py` | pääsilmukka (`Seuranta`): kalibrointi → profiili → HAKU + SEURANTA, radan elinkaari, raportit |
| `kalibrointi.py` | kalibrointi moodikuvasta: kaukainen pesä, hoglinet, keskiviiva, kameran asento (K, R, t), täyden resoluution skaalaus |
| `kalibrointi_perus.py` | kalibroinnin perusfunktiot: lähipesän renkaat ja viivat, k1, geometrinen homografia, pesien laatu |
| `rata.py` | radan mitat, hoglinjat (nimellinen + mitattu), top-down-koordinaatit |
| `kivimalli.py` | kiven 3D-malli (profiili, kahvan lovi), projektio, ennustettu ääriviiva, linssikorjauskartat |
| `paneelit.py` | stabilointipaneelien tunnistus ja seuranta |
| `esikasittely.py` | valotasapaino, varjosietoinen taustanvaimennus, vaihekorrelaatio, liukuhihna (vaiheet A/B), GPU-vaihe B |
| `profiili.py` | kiviprofiilin opettelu: tiheä skannaus, siemen, seuranta ajassa, alfa-ääriviivat, profiilin sovitus |
| `siluetti.py` | siluettitarkennus (C++) |
| `paikallinen.py` | live: SEURANTA kivien alueilla täydellä resoluutiolla |
| `kierre.py` | kahvan väri ja kierteet |
| `heitot.py` | CSV-rivit, hog-tarkistus, heittoportti, lopullinen heitto-CSV |
| `hog_analyysi.py` | hog-hog-analyysi (nopeus, hidastuvuus, suunta, kurvimalli), liuku, irroitus, hakki |
| `nakyma.py` | debug-video ja puhelinnäkymä: paneelit, laatikot, irroitusviivat, enkooderit |
| `katselu.py` | puhelinnäkymän HTTP-palvelin (`--katselu`), viive, tallennus `<nimi>_katselu.avi` |
| `live.py` | kamerasyöte, RAM-puskuri, `BufferedCapture`, live-ModeEngine, tallennus |
| `yleiset.py` | aikamittaus, pullonkaula-analyysi, taustaprosessien prioriteetti, versiotieto |

### C++ (`cpp/`)

Kaksi pybind11-moduulia: `mode_engine` (videon luku ja moodikuva) ja `stone_tracker`. `stone_tracker.cpp` sisältää
Python-rajapinnan ja ottaa osat mukaan yhdeksi käännösyksiköksi (järjestys = riippuvuudet):

| Tiedosto | Sisältö |
|---|---|
| `yhteiset.hpp` | includet, aikamittaus, OpenCV:n yksisäikeisyysvartija, vakiot, numpy-muunnokset |
| `geometria.hpp` | projektio, kiven hulli, peitto-osuus, ristikkohaut (HAKU linearisoitu, SEURANTA laajeneva + mäennousu) |
| `maskit.hpp` | saturaatio, graniittimaski, väriportti, taustanvaimennus, etualamaski, ROI, bilineaarinen näyte |
| `sovitus.hpp` | runkokontuuri, rengaspisteet, jäännösvirheet, Levenberg-Marquardt (linearisoitu + tarkka varmistus) |
| `meanshift.hpp` | SEURANNAN mean-shift-paikannus |
| `siluetti.hpp` | siluettitarkennus |
| `seuranta.hpp` | `track_stones_batch` (yhdistelmähaku: ristikko + 2 mean-shiftiä rinnan, LM, siluetti) |
| `haku.hpp` | `search_new_stones` (ristikko, tarkennus, spawn-suodatin, floodFill-poisto) |
| `esikasittely.hpp` | `suppress_shadow_background`, vaihekorrelaation vaiheet |
| `gpu.hpp` | GPU-vaihe B (OpenCL: warp + remap + varjosuodatus) |
| `profiili.hpp` | `scan_stone_candidates`, `fit_stone_profile_cpp`, alfa-ääriviiva |

### Kääntäminen

Windows (vcpkg + Python 3.12, kuten ennen; `CMakeLists.txt` viittaa nyt `cpp/`-kansioon):

```
cmake -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

Valmiit `.pyd`-tiedostot kopioituvat projektikansioon. **Testi_08_02:n `.pyd`-tiedostot eivät käy** (rajapinta muuttui).

Linux (testaus):

```
S=$(python3 -c "import sysconfig;print(sysconfig.get_config_var('EXT_SUFFIX'))")
g++ -O2 -std=c++17 -shared -fPIC $(python3 -m pybind11 --includes) $(pkg-config --cflags opencv4) cpp/stone_tracker.cpp -o stone_tracker$S $(pkg-config --libs opencv4) -lpthread -ldl
g++ -O2 -std=c++17 -shared -fPIC $(python3 -m pybind11 --includes) $(pkg-config --cflags opencv4) cpp/mode_engine.cpp -o mode_engine$S $(pkg-config --libs opencv4) -lpthread
```

## Mitä poistettiin

Kokeilutilat kiinnitettiin Testi_08_02:n oletuksiin ja vaihtoehtoiset haarat poistettiin:

* **C++:** SEURANNAN hakutilat 0–4 (vain yhdistelmähaku jäi), ristikkohaun ohitus ja sen varjotila, hienon ristikon
  tyhjentävä haku (mäennousu jäi) ja rinnakkainen ristikko (`set_grid_threads` ei vaikuttanut mäennousun kanssa),
  GPU-ristikkohaku, vanhat maski- ja peitto-osuusfunktiot (SIMD-versiot jäivät), saturaation rinnakkaissäie,
  "liian iso kontuuri" -hylkäys (raja oli aina pois), `track_stone_update`, `search_new_stone`, `ms_debug`,
  `silhouette_refine_batch_cpp`, tilastofunktiot, ympäristömuuttujat (`LM_*`, `GRANITE_*`, `BOUNDARY_*`, `VIDEO_HW` →
  `asetukset.VIDEO_LAITTEISTODEKOODAUS`), `mode_engine`: perspektiivikorjaus, moodisuodatus ja videon kirjoitus.
* **Python:** C++-funktioiden Python-varapolut (C++-moduulit ovat pakollisia), `kamera9_*`-kokeiluversiot (vain käytetyt
  funktiot siirrettiin), käyttämättömät kalibrointifunktiot ja -vakiot, ympäristömuuttujat (`KURVIMALLI`, `KATSELU_*`,
  `DEBUG_*`, `LIVE_*`, …), värireferenssi (pelkkä diagnostiikka), siluetin varapolku SEURANNASSA.
* Vanhat työkalut (`tools/`, kokeiluskriptit ja README-historiat) jäivät Testi_08_02:een; mukana vain `tools/kamera_testi.py`.

C++-poistot tarkistettiin gcov-kattavuudella (MAH00014 + debug, 1080p-live-simulaatio, live-simulaatio + katselu):
jäljelle jääneessä koodissa ajamatta ovat vain virhepolut ja GPU-koodi (ei GPU:ta testikoneella).

## Varmistus (Linux, OpenCV 4.6)

Samat ajot Testi_08_02:lla ja Testi_08_03:lla, CSV:t verrattu riveittäin:

| Ajo | Tulos |
|---|---|
| `MAH00014.mp4 --no-debug` (Python-jako, vanhat .so) | sijainti-, raaka-, hog- ja kahva-CSV identtiset |
| `MAH00014.mp4 --no-debug` (uudet C++-moduulit) | VERTAILU_MAH |
| live-simulaatio 1080p (`--live-sim`, täysi resoluutio) | VERTAILU_SIM1080 |
| live-simulaatio + `--katselu` | VERTAILU_SIM |

GPU-vaihe B testattiin OpenCL-CPU-toteutuksella (pocl, `GPU_LAITE=cpu`): uusi ydin antaa bitilleen saman tuloksen kuin
Testi_08_02:n ydin ja CPU-polku.
