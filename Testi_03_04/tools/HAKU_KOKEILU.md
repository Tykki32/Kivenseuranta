# HAKU-kokeiluympäristö (Testi_03_04)

Tarkoitus: kokeilla HAKUn (`stone_tracker.search_new_stones`) parametreja kivien **saapumiskohdissa** ilman koko videon läpikäyntiä (yksi HAKU-kutsu ~35 ms).
`tools/haku_kokeilu.py` (luokka `HakuLab`).

## Aineisto (MAH00014, live alkaa ruudulla 2227)

Tallennetut ruudut = live-HAKUn saama `frame_u_for_tracking` (valotasapaino + varjosietoinen taustanvaimennus), 236 ruutua, ~20 MB (ei repossa, `/tmp/w/hakudump`):

| Kivi | Live-HAKUn 1. ehdokas | Ruutuväli | Huomio |
|---|---|---|---|
| 17 | 2760 (Y = 3312) | 2710–2770 | siisti saapuminen: HAKU löytää kiven heti kun se ylittää alueen (ruutu 2756, Y ≈ 3328) ja jokaisella ruudulla sen jälkeen |
| 3 | 2240 (ehdokas 69) | 2227–2290 | seuranta katkesi ("pysähtynyt" 2295); sama kivi rekisteröitiin uudelleen 2260/2270/2280 (ehdokkaat 70–72) |
| 51 | 3430 (ehdokas 121) | 3400–3510 | "duplikaatti" 3465, uusi ehdokas 3480, CSV:n 1. rivi 3500; vieressä lakaisija |

Tallennus (kerran; tarvitsee `calib_profile.pkl`:n = kalibrointi + profiili):

    TRACKER_FRAME_DUMP_DIR=/tmp/w/hakudump TRACKER_FRAME_DUMP_RANGES="2227:2229,2230:2290,2710:2770,3400:3510" \
      python tools/run_profile.py <video> <calib_profile.pkl> <ulos> 3520

**Validointi:** lab antaa live-ajon ehdokkaille täsmälleen samat sijainnit (esim. 2240: (−4,6; 3294,7), 2760: (30,9; 3311,7), 3430: (−20,4; 3250,9)).

## Käyttö

    import sys; sys.path.insert(0, "tools")
    from haku_kokeilu import HakuLab, ENTRIES
    lab = HakuLab("/tmp/w/hakudump", "/tmp/w/mah_base/calib_profile.pkl", "/tmp/w/full03/stones.csv")
    lab.table(range(2750, 2763))                       # tulokset + lähin oikea kivi
    lab.table(range(2750, 2763), accept=0.4)           # yksi parametri kerrallaan
    lab.sweep(ENTRIES, "accept", [0.0, 0.1, 0.2, 0.3, 0.4])   # 1. löytöruutu / osumat / väärät
    lab.overlay(2756, "/tmp/kuva.png")                 # kuva

Parametrit (oletus = live): `x_half=70`, `y_min/y_max` (hogline −100/+300 cm), `coarse=10`, `fine=2`, `score_thr=0.15` (esikarsinta),
`accept=0.20` (C++ `HAKU_ACCEPT_SCORE_THRESHOLD`, nyt ajonaikainen: `set_haku_accept_score`), `max_results=4`, `max_attempts=8`,
`spawn=(rms_max 3.5, nb_min 14, nb_max 45, |X|max 65, score_max 0.9)`, `diff_threshold=0`.

"Oikea" sijainti tulee `stones.csv`:stä (ensimmäisestä CSV-rivistä alkaen); sitä edeltävä sijainti on vakionopeus-ekstrapolaatio, joka on
karkea (virhe >100 cm, kun ollaan >40 ruutua ennen CSV:n ensimmäistä riviä) – "VÄÄRÄ"-merkintä ei silloin ole luotettava, tarkista kuvasta.

## Lähtötilanne (oletusparametrit, 3 saapumiskohtaa yhteensä)

| Parametri | Arvot → tulos |
|---|---|
| accept | 0,0–0,3: sama (1. löytö 2756 / 2231 / 3451); 0,4: osumat vähenevät (kivi 51: 37 → 25), vääriä 42 → 22 |
| score_thr | 0,05–0,3: sama; 0,5: osumat vähenevät (kivi 51: 37 → 20), vääriä 42 → 21 |
| spawn rms_max | 3,5 → 2,0: vääriä 42 → 31, osumia vähän vähemmän (59 → 56, 37 → 33); suodatin pois: vääriä 49 |
| y_max | 3317,6 → 3500: vääriä 42 → 62, ei aikaisempaa löytöä |

"Väärät" = ehdokkaat jotka eivät osu seurattuun heittoon (mukana myös muut alueella olevat kivet/pelaajat, ei pelkästään roskaa).
Live ajaa HAKUn vain joka 10. ruudulla → kivi 17 löytyy 2760, vaikka se on löydettävissä jo 2756.
