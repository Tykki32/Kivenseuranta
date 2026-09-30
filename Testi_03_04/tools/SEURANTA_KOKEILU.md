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
