# Tulokset – Testi_03_04 v4.0, video MAH00014

`alfa_profiili_kivi_kerrallaan.csv` / `alfa_profiili_loki.txt`: jokaiselle 26 kivelle (24 live-heittoa + 2 profiilinopettelun aikaista) profiili **yksin**
julkaistulla koodilla (`tools/alfa_profiili_testi.py`). UUSI = alfa-ääriviiva + yhteinen alfa-taso (0,439) + hylkäyssäännöt;
VANHA = seurannan oma ääriviiva samoilla otoksilla (Testi_03_03:n tapa, ilman sovituksen omaa poikkeamakarsintaa).

| | Portti OK (rms ≤ 1,5, R 12,5–15) | R ka ± hajonta | R min–max | H ka ± hajonta | rms ka (max) | R > 14,55 |
|---|---|---|---|---|---|---|
| **UUSI (alfa)** | **23/26** | **13,66 ± 0,26** | 13,17–14,33 | 13,48 ± 0,13 | 0,74 (0,89) | 0 |
| VANHA (seuranta) | 10/26 | (mediaani ~14; 14 kivellä R > 14,55, osa satoja) | 12,98–186,6 | 13,05 ± 0,53 | 2,32 (7,21) | 14 |

Kolme kiveä ei läpäise porttia, koska hylkäyssäännöt poistavat lähes kaikki havainnot:
* **51**: 36/40 ruutua hylätty säännöllä 1 (tumma alue kiven vieressä; maalattu viiva/kenkä ~8 m:n kohdalla) – jäljellä 2 havaintoa.
* **222**: 0/40 (lakaisija koko radan ajan, 29 × tumma alue, 11 × suhde).
* **263**: seuranta tuotti vain 8 havaintoa, kaikki hylätty.
Kivet 38, 107, 244 jäävät 18–23 hyväksyttyyn havaintoon (yhä yli PROFILE_MIN_SAMPLES 15) ja läpäisevät portin.
Sääntö 1 (tumma alue > 100 px) voi olla liian tiukka – tarkistettava.
