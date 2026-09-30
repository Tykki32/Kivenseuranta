# Testi_03_04 v4.1 – alfa-ääriviiva profiilin opetteluun

Pohja: Testi_03_03 (pelialue X ±2 m / Y 4–20 m, tiukat rms/R-rajat, C++-profiilisovitus). Tässä versiossa muutetaan **vain profiilin opettelua**,
ei live-seurantaa.

## Miksi

Profiilin opettelun ääriviiva oli joko
* **seurannan oma** (R ≈ 14,0, mutta vuotaa varjoon ja viivoihin; kivi 244: R = 35, kivi 222: lakaisija sotki), tai
* **varjosuodatettu binäärimaski** (varjo pois, mutta reuna sisenee ~1–2 px => R ≈ 12; ei läpäise porttia).

Syy jälkimmäiseen: kiven reunapikselit ovat graniitin ja jään sekoituksia (alipikselireuna), ja `findContours` kulkee reunapikselien keskipisteitä pitkin.

## Mitä muutettiin

1. **Alfa-kartta** (`main.py`: `alpha_observation`): `k = V/V_ref`, `kg` = graniitin k, `ks` = paikallinen taustataso (jää tai varjo, kiven ulkopuolelta),
   `alfa = (ks − k)/(ks − kg)` = kuinka suuri osa pikselistä on graniittia.
2. **Alipikselireuna** (`alpha_contour`): alfa-kartan tasa-arvokäyrä tasolla `ALPHA_LEVEL`, 4× ylinäytteistys, kuminauha (kupera peite, ei koloja),
   rajattuna graniittimaskiin A (laajennettu 5×5) – yläreuna ja kahvan puoli (lovi) tulevat graniittimaskista.
3. **Yhteinen alfa-taso**: graniitin V / jään V -suhteen mediaani kaikista hyväksytyistä havainnoista (MAH00014: ≈ 0,434; kivien välinen hajonta vain ±0,015).
   Yksittäisen havainnon suhde hylätään, jos se on välillä 0,2–0,8 ulkopuolella.
4. **Hylkäyssäännöt**: (1) tumma alue kiven ympärillä (> 100 px, V < 60, rengas 9–31 px A:sta) – lakaisija/harja/kenkä;
   (2) ääriviivan pinta-ala poikkeaa > 30 % lähiruutujen (±4) liukuvasta mediaanista – esim. kiveen liittyvä maalattu keskiviiva.
5. **C++**: `fit_stone_profile_cpp` ottaa ääriviivat liukulukuina (`double`; kokonaisluvut kelpaavat edelleen); alfa-kartta ja -ääriviiva lasketaan C++:ssa (ks. Nopeus).
6. `PROFILE_SAMPLES_PER_STONE` 25 → 40 (hylkäykset vähentävät havaintoja; min 15).
7. Muistiinpano live-seurannan kommentteihin: alfa-kartta parantaisi todennäköisesti luottamusta myös seurannassa – tarkastellaan, kun seuranta käydään läpi.

Portti ennallaan (rms ≤ 1,5 px, R 12,5–15 cm), `PROFILE_MIN_ACCEPTED_STONES = 2`. `PROFILE_ALPHA=0` palauttaa vanhan ääriviivan.

## Nopeus

Alfa-laskenta on C++:ssa (`stone_tracker.alpha_observation_cpp` / `alpha_contour_cpp`): ~7 ms/havainto (numpy/cv2-versio ~190 ms), tulos **bitti bitiltä sama**
(alfa-kartan ero 0, ääriviivat identtiset – tarkistettu 42 havainnolla). Valotasapainon estimointi on vielä Pythonissa (~7 ms). Varapolku: `ALPHA_CPP=0`.
Tarkempi optimointi myöhemmin.

## Testi

`python tools/alfa_profiili_testi.py <video> <calib_profile.pkl> <stones.csv> <ulos>` – jokaiselle kivelle erikseen profiilin opettelu
julkaistulla koodilla (alfa-ääriviiva, yhteinen taso) + vertailu seurannan omaan ääriviivaan. Tulokset: `tulokset/`.
