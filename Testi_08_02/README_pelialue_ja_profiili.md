# Testi_03_03 - profiilin opettelu: pelialue, tiukat rajat, C++-sovitus

Pohja: Testi_03_02 v2.1. Muutokset vain kalibroinnin jalkeiseen PROFIILIN OPETTELUVAIHEESEEN (live-seuranta ennallaan):

- **Tiukat rajat:** PROFILE_MAX_RMS_PX 10 -> 1.5, SOLO_TRACK_MAX_RMS_PX 12 -> 3, STONE_TRACK_PRECHECK_MAX_RMS_PX 24 -> 8,
  R_max 10..20 -> 12.5..15 cm (rms < 1 px = oikean kokoinen kivi, < 5 px = jo vaaran kokoinen).
- **Pelialue:** X = -2...+2 m, Y = 4...20 m (+ 25 cm parallaksivara ehdokkaan sijaintiin). Kuvasta kasitellaan vain pelialueen
  projektiota vastaava suorakulmio (ROI, ~29 % kuvasta): taustanvaimennus, granittimaski ja aariviivat (Python `find_stone_candidates`
  roi=/bounds=, C++ `scan_stone_candidates` roi_x0..roi_y1). Skannaus 11.8 -> 3.1 ms/kuva.
- **Profiilin sovitus C++:ssa** (`stone_tracker.fit_stone_profile_cpp`, 1:1-porttaus fit_stone_profile + LM; Jacobian harvasti):
  tulokset identtiset Pythonin kanssa (ero ~1e-15), 25 havaintoa 39 s -> 1.2 s, 50 havaintoa ~165 s -> 2.7 s. `PROFILE_FIT_CPP=0` = Python.
- `tools/run_calib.py <video> <ulos> [max_frame]`: ajaa kalibroinnin + profiilin opettelun videon alusta ja tulostaa vaiheajat.

Tulos (MAH00014): live-seuranta alkaa frame 2227 (ennen 2377), pre-live-vaihe 233 s -> 64 s;
profiili R_max 13.66 cm, H 13.38 cm, kahvan lovi 0.691, rms 1.02 px (50 havaintoa).
