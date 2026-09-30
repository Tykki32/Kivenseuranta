# SEURANTA-kokeiluympäristö ja maskituki (Testi_03_04)

`tools/seuranta_kokeilu.py` (`TrackLab`): toistaa yhden kiven seurannan live-logiikalla (nopeusrajattu hakualue, liike-ennuste, taaksepäin-raja,
missit, pysähtynyt-sääntö) tallennetuilla live-ruuduilla (ks. HAKU_KOKEILU.md) eri parametreilla; `replay(..., sil=SilhouetteRefiner, min_inside=0.4, gate_on=..., sil_refine=...)`.

## Löydökset (alkupää, kivet 3 ja 51)

* Radan alkupäässä LM-tarkennus epäonnistuu lähes aina (`tarkka = 0`, rms ~8–10 px) – paikka tulee ristikkohausta ja laahaa 25–45 cm jäljessä (kivi 3: 32 cm).
* Kivi 51: seuranta tarttuu **pelaajan päähän** (pyöreä, kiven kokoinen, ei graniittia) koko 80 ruudun ajaksi (virhe ~285 cm).

## Maskituki + siluettitarkennus (oletuksena päällä, `SEURANTA_SILHOUETTE=0` kytkee pois)

Kun LM-tarkennus ei ole tarkka (`tarkka = 0`):
1. kiven 3D-mallin siluetin **sisällä pitää olla vähintään 40 % maskipikseleitä** (`SEURANTA_MIN_INSIDE`), muuten ruutu on miss (hakualue laajenee -> seuranta voi löytää oikean kiven);
2. hyväksytty paikka tarkennetaan siluetilla (max ±2 px, `SEURANTA_SIL_SHIFT_PX`).
Tarkat (`tarkka = 1`) LM-paikat jäävät ennalleen. HAKU-ehdokas hylätään samalla säännöllä (vaikutus pieni: 0/119 oikeaa hylätään).

Offline-toisto (virhe HAKU-pisteisiin): kivi 3: 32 -> 4 cm; kivi 51: 285 cm -> 17 cm (seuranta irtoaa päästä ja löytää oikean kiven ~ruudulta 3456).

## Koko video (MAH00014, precomputed profiili)

| | Ilman | HAKU-tarkennus | + SEURANTA-maskituki |
|---|---|---|---|
| Heittoja | 27 | 27 | **26** (oikea määrä; pelaajan päätä seurannut ylimääräinen rata #8 poistuu) |
| Rivejä | 14 621 | 15 082 | 14 629 |
| Koko radan rms-mediaani (ka) | 1,20 | 0,94 | 0,92 |
| SEURANNAN maskituki | – | – | 4,7 ms/ruutu (Python; nopeutetaan C++:ssa myöhemmin) |

Ratojen loppupäät (heitot #24–#27) vastaavat aiemmin validoituja (Testi_03_03) pysähtymispaikkoja: 2,2 / 3,7 / 3,8 / 2,8 m (ilman: −1,3 / 5,8 / 4,6 / 8,6 m).
HUOM: CSV:n `rms_px`-sarake on LM-sovituksen rms eikä siluetin laatu, joten alkupään rms (~9 px) pysyy, vaikka paikka on nyt tarkempi.

## C++ (silhouette_refine_cpp)

Siluettitarkennus on C++:ssa (`stone_tracker.silhouette_refine_cpp`, Python-kääre `haku_silhouette.SilhouetteRefiner`, `k94` annettava konstruktorille): ~0,7–0,9 ms/kutsu
(numpy/cv2-versio ~4,6–5,2 ms); siluetin siirtokorrelaatiot lasketaan rivikohtaisilla kumulatiivisilla summilla (ei FFT:tä). Tulos on sama kuin numpy-versiolla
(paikkaero 0,000 cm, pisteet samat; tasapelissä valitaan pienin siirto molemmissa). Varapolku: ilman `k94`:ää numpy/cv2-versio (`refine_py`).
Koko video: SEURANNAN maskituki 4,66 -> 0,63 ms/ruutu (15,8 % -> 2,4 %), ajoaika 438,6 s / 16 501 ruutua (37,6 r/s), 26 heittoa.
(C++-versio käyttää HAKUn/SEURANNAN tavallista graniittimaskia (sumennus pienennetyllä kuvalla), joten radat eroavat marginaalisesti numpy-ajosta.)

## v4.4 (siluettitarkennus: 3x3-maski + kaista, SEURANTA kaikille ruuduille kun Y > 20 m)
* Siluettitarkennuksella on OMA maski (3x3-avaus; `create_granite_mask`:n 5x5 poisti 5-6 px leveän kaukaisen kiven) ja ylitysrangaistus lasketaan vain 3 px:n kaistalla siluetin
  ympärillä (kaukainen tumma kohde, esim. heittäjä, ei vedä siluettia). C++ = Python (290 ruutua, paikkaero 0.00 cm), C++ ~1.4 ms, numpy ~4 ms.
* SEURANTA: kun Y > `SEURANTA_SIL_ALL_Y_CM` (2000) siluetti ajetaan myos tarkoille ruuduille (siirto max ±3 px); tarkassa ruudussa inside0 < 0.4 ei hylkaa ruutua. Lähempänä vanha saanto.
* Koko video (26 heittoa): sivusuunnan sileys y > 26 m 1.83 -> 0.81 cm, syvyys 7.65 -> 5.97 cm; y > 20 m: ei yhtaan ulkolaisrivia (|rx| > 6 cm tai |ry| > 30 cm), suurin |rx| 4.9 cm.
* Uusi ongelma: heitto 181 (uusi id 183) katoaa y ~ 11 m:ssa (f12705-12709 seuranta hyppaa 26 cm sivuun, inside0 0.54-0.60) eika palaa; aiemmin rata jatkui y = 3.7 m:iin.
* Debug: SEURANTA_SIL_DEBUG=a:b tulostaa [SILDBG]-rivit (siluetin siirto, inside, min_y).
