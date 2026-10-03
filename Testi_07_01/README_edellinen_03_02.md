# Testi_03_02 - HAKU:n roskasuodatin (spawn-suodatin)

Pohja: Testi_03_01 v1.1 (liukuhihna). Muutos: HAKU hylkaa roskaehdokkaat jo rekisteroinnin hetkella
(`stone_tracker.set_spawn_filter`, `main.py` SPAWN_*-vakiot), ennen kuin ehdokkaalle kaytetaan yhtaan SEURANTA-paivitysta.

Kriteerit (kaikkien tayttyva): sovituksen rms <= 3.5 px, runkopisteita 14..45, |X| <= 65 cm, ristikkopistemaara <= 0.9.
Ymparistomuuttujat: SPAWN_FILTER=0 (pois), SPAWN_RMS_MAX, SPAWN_NB_MIN, SPAWN_NB_MAX, SPAWN_ABS_X_MAX, SPAWN_SCORE_MAX.

Koko MAH-video (16 501 ruutua, 4 ytimen kone), Testi_03_01 v1.1 -> Testi_03_02 v2.0:
- ehdokkaita 673 -> 348, vahvistettuja ratoja 389 -> 228
- kivia/SEURANTA-kutsu 3.18 -> 2.48, SEURANTA 25.9 -> 21.8 ms/ruutu
- koko ajo 498 s -> 465 s; kaikki 26 tunnettua heittoa loytyvat

## v2.1 (samassa kansiossa)

- SEURANNAN kevennykset (tulokset identtiset, offline-tarkistettu): saturaatio suoraan BGR:sta (ei koko HSV-muunnosta), jaettu
  maskin ja reunahaun kesken; turha taustanvaimennus/kopiointi ohitetaan kun kuva on jo vaimennettu Pythonissa.
- `BOUNDARY_CONVERGENCE_CM` 0.05 -> 0.5 (ymparistomuuttuja): ~20 % vahemman LM-tyota, paikkaero <= 0.5 cm (p99 0.1 cm).
- PAIKANVARAUS (`EVICT_AT_CAP=1`, `EVICT_MIN_RMS=8`): kun 8 aktiivista rataa on taynna, HAKU ajetaan silti ja uusi ehdokas
  saa paikan poistamalla huonoimman radan (ensin vahvistamattomat, sitten vahvistetut joiden rms >= 8). Korjaa tapauksen jossa
  taysi roskaratojen maara esti oikean heiton (frame 3230) rekisteroinnin.
- Koko video: 26/26 heittoa, 440 s (37 r/s), SEURANTA 19 ms/ruutu, 2.49 kivea/kutsu.
