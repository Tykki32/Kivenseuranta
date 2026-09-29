# Testi_02_03 – mean-shift-pohjainen hakuvaihe kivenseurantaan

Testi_02_03 on Testi_02_02:n kopio, johon on lisätty **vaihtoehtoinen tapa löytää kivi seuraavasta framesta**:
ristikkohaun (`locateByGridSearchTrackingFast`) tilalle iteratiivinen painopistesiirto (mean-shift),
jonka jälkeen normaali LM-yhteissovitus (`refinePositionJoint`) tarkentaa sub-pikselitasolle.
Kaikki uusi koodi on `stone_tracker.cpp`:ssä (`meanShiftLocate`), valinta `main.py`:ssä (`TRACKER_MODE`).

## Idea (käyttäjän ehdotus) ja sen toteutus

1. Kiven 3D-mallin projektio (hulli) ja sen 1,1-kertainen laajennus jaetaan hullin painopisteen kautta kulkevalla
   vaaka- ja pystyviivalla neljään osaan (sama 1,1-marginaali kuin `hullOverlapScore`:ssa).
2. Maskin pikselit lasketaan painotettuina: sisähullin alue paino 2, vain 1,1-kehään osuva paino 1.
3. Alaosan ja yläosan (sekä oikean ja vasemman) painotettujen pikselien erotus normalisoituna kokonaispainolla
   kertoo siirtosuunnan: `s_y = Σ w·[maski]·g(y−cy)/Σw`, `s_x` vastaavasti.
4. Signaali muunnetaan siirtymäksi **optimoidulla painokertoimella** ja malli siirretään; toistetaan kunnes
   siirtymä < `MS_TOL_PX`. Sen jälkeen LM tarkentaa.

### Painokerroin (siirtymän optimaalinen suuruus)

Pelkkä "erotus / leveys" ei ole lineaarinen siirtymän suhteen (signaali kyllästyy kun malli ja maski eivät enää
peity), joten siirtymä lasketaan `d = gain · r · f⁻¹(s)`, missä

* `f(δ)` on yksikköympyrän vastefunktio, joka lasketaan numeerisesti samoilla painoilla (2/1) ja marginaalilla
  (1,10) kuin itse mittaus (`getResponseLUT`); ellipsille (kiven perspektiiviprojektio) tämä on affiini-invariantti,
  kun `r` on hullin puolileveys/-korkeus kyseisessä akselissa;
* `f` nousee monotonisesti arvoon δ ≈ 0,89·r (n. 12–13 cm) – yhden askeleen kaappausalue;
* `gain` on empiirinen hienosäätö (oletus 0,8, ks. `tulokset/painokerroin_gain.txt`; optimi on litteä välillä 0,5–0,8).

Pikselisiirtymä muunnetaan (X, Y)-senteiksi paikallisella Jakobiaanilla (projektiosta).

### Kaksi poikkeamaa alkuperäisestä ideasta (mittausten perusteella)

Offline-mittaus MAH00014-videolla paljasti, että puhdas idea ei riitä:

* **Pehmeä painotus (`MS_TAU`).** Kaukana (Y ≈ 29 m) kivi näkyy vain ~7×11 px kokoisena ja 10 cm siirtymä on ~1 px.
  Kova `sign()` hyppii silloin pikseliruudukon mukaan; `g(u) = clamp(u/τ, −1, 1)` (τ = 0,5) on jatkuva.
* **Paikallinen viimeistely (`MS_POLISH`).** Maskin epäsymmetria (kahvan kolo, yläpinnan valaistus) siirtää
  painopistetasapainon pois ristikkohaun IoU-optimista (mitattu: −0,2 signaalitaso ≈ 0,75 px ≈ 7 cm kaukana), jolloin
  pelkän mean-shiftin pistemäärä jäi usein alle kynnyksen 0,35 (läpäisy 0,67). 8-naapuruston kukkulankiipeily
  `hullOverlapScore`:lla (askel 4 cm → 0,75 cm) nostaa läpäisyn arvoon 0,95 ja on halpa (~0,7 ms).
* **Varahaku.** Jos pistemäärä jää silti alle `TRACK_SCORE_THRESHOLD`:in, ajetaan alkuperäinen ristikkohaku
  (hybridi, `TRACKER_MODE=meanshift`). Se pelastaa 7 % kaikista päivityksistä (1 590/23 216), jotka pelkkä
  mean-shift menettäisi.

## Käyttö

Ympäristömuuttujat (`main.py`, kommentti ennen `run_pipeline`:a):

| muuttuja | merkitys |
|---|---|
| `TRACKER_MODE=ensemble` (**oletus**) | yhdistelmähaku (ks. "Kaikkien kivien seuranta") |
| `TRACKER_MODE=meanshift` | mean-shift + ristikkohaku varana |
| `TRACKER_MODE=grid` | alkuperäinen ristikkohaku – **identtinen Testi_02_02:n kanssa** (regressiotesti: 262 vertailua, 0 eroa) |
| `TRACKER_MODE=meanshift_pure` | pelkkä mean-shift (ei varahakua) |
| `TRACKER_MODE=compare` | ajaa grid + mean-shift (+ `TRACKER_GOLD=1`: tyhjentävä haku, `MS_GAINS=0.5,1.2`: lisägainit) samoilla syötteillä joka framella ja kirjaa CSV:hen (`TRACKER_COMPARE_LOG`) |
| `MS_GAIN, MS_MAX_ITER, MS_TOL_PX, MS_TAU, MS_POLISH, MS_INNER_WEIGHT, MS_MARGIN` | virityskertoimet (oletus 0,8 / 6 / 0,3 / 0,5 / 4,0 / 2,0 / 1,10) |
| `ENS_BACK_PEN, ENS_PRED_PEN, ENS_BACK_TOL_CM, PRED_LOOKBACK, PRED_MIN_SPAN` | yhdistelmähaun rangaistukset (oletus 0,15 / 0,02 / 2 cm) ja liike-ennusteen ikkuna (12 / 4 framea) |
| `NEW_STONE_DEDUP_CM`, `DUP_MERGE_CM`, `DUP_MERGE_FRAMES` | uuden kiven eston säde (oletus **20**, alkuperäinen 100), duplikaattien yhdistäminen (oletus **30 cm**, 3 framea; 0 = pois) |
| `HAKU_LOG=polku.csv` | kirjaa jokaisen HAKU-kutsun tuloksen ja sen, esti/salli päällekkäisyyden esto |
| `TRACKER_FRAME_DUMP_DIR`, `TRACKER_FRAME_DUMP_RANGES=a:b,c:d` | tallentaa jokaisen framen simulaattoria varten (`tools/sim_track.py`) |

**Alkuperäinen Testi_02_02-käytös:** `TRACKER_MODE=grid NEW_STONE_DEDUP_CM=100 DUP_MERGE_CM=0`.

Windows-käännös: `CMakeLists.txt` ennallaan. `mode_engine.cpp` avaa videon `CAP_MSMF`:llä vain Windowsissa
(`#ifdef _WIN32`), muualla `CAP_ANY` (jotta Linux-testaus onnistuu).

## Työkalut (`tools/`)

* `analyze_compare.py` – analysoi `compare`-ajon lokin (aika, tarkkuus pikseleinä, kerrostettuna).
* `compare_tracks.py`, `compare_throws.py` – vertailevat kahden koko putken ajon `_kivien_sijainnit.csv`:tä
  (rivikohdistus framen mukaan; "todellisten heittojen" kattavuus).
* `ms_offline.py` – offline-virityskehys tallennetuilla framedumpeilla (`TRACKER_DUMP_DIR`, `TRACKER_DUMP_RANGE=alku:loppu`).
* `sim_track.py`, `sim_experiments.py` – yhden kiven **peräkkäinen seurantasimulaattori** (toistaa `main.py`:n SEURANTA-logiikan: hakualue nopeusrajasta, "ei taaksepäin", missit, "pysähtynyt") tallennetuilla frameilla; vertaa rataa oikeaan.
* `analyze_haku.py` – HAKU-lokin analyysi (mikä esti minkä heiton, mistä etäisyydeltä).
* `plot_coverage.py`, `make_diff_images.py` – kattavuuskuva ja heittokohtaiset kuvat (`tulokset/kuvat/`).
* `stone_tracker.ms_debug(...)`, `ms_response_table(...)` – diagnostiikka.

## Tulokset (MAH00014_leikattu.mp4, 1280×720, 16 501 ruutua, 23 216 kivipäivitystä)

Kaikki aikavertailut ovat **saman ajon sisällä vuorotellen mitattuja** (kone on kohinainen: sama koodi vaihteli
ajojen välillä ±50 %), tarkkuus pikseleinä (kaukana 10 cm ≈ 1 px) suhteessa tyhjentävään ristikkohakuun ("gold").
Yksityiskohdat: `tulokset/`.

### Aika (ms / kivipäivitys)

| | prep (maski) | haku | LM-tarkennus | yhteensä |
|---|---|---|---|---|
| ristikkohaku (alkuperäinen) | 7,36 | 3,74 | 29,39 | **40,48** |
| mean-shift-hybridi | 7,38 | 2,54 | 29,82 | **39,75 (−1,8 %)** |
| mean-shift ilman varahakua (rivit joilla varahakua ei tarvittu) | – | 1,08 vs 2,68 | – | – |

Hakuvaihe nopeutuu 32 % (hybridi) / 60 % (ilman varahakua), mutta se on vain 9 % päivityksen ajasta, joten
**kokonaisnopeutus on vain ~2 %**. Varahakuun päätyy 17 % päivityksistä (niissä hakuaika ~9–10 ms).

### Tarkkuus (px, mediaani / p90; pienempi parempi)

| | haun tulos vs gold | lopullinen (LM:n jälkeen) vs gold |
|---|---|---|
| ristikkohaku | 0,73 / 1,78 | 0,49 / 1,61 |
| mean-shift-hybridi | **0,40 / 1,05** | **0,26 / 0,93** |

Aidoilla kivillä (gold `tarkka`=1, n = 4 461) ero on pieni: lopullinen 0,03 / 0,57 (ristikko) vs 0,03 / 0,55 (mean-shift).
Lähellä (Y ≤ 15 m) menetelmät ovat tasoissa (0,06 px); kaukana (Y > 25 m) mean-shift on selvästi tarkempi
(lopullinen 0,34 vs 0,77 px), koska ristikkohaun 1,5 cm askel/tasanko on kaukana karkea.
Löytöaste: mean-shift löysi 136 kertaa kun ristikko ei, ristikko 4 kertaa kun mean-shift ei.

### Koko putki (ratataso)

Kontrolli: kaksi ristikkohakuajoa vastaavat toisiaan (90 % riveistä < 30 cm päässä, mediaani 0 cm), joten alla
oleva ero on aito. Mean-shift-ajo löysi **12/15 todellista heittoa** (ruudut ≥ 2990), ristikkohaku **9–10/15**
(`tulokset/vertailu_heitot.txt`); radan lukumäärä oli 173 vs 206 ja rms-mediaani 1,3 vs 9,3 px (vähemmän
pelaaja-/lakaisijaratoja). Yhteisten todellisten heittojen sijaintiero: mediaani 0,12 cm, p90 7,4 cm.

**Varaus:** yksi video, ~15 heittoa. Kivien löytyminen riippuu myös ehdokashallinnan ketjureaktioista
(aktiivisten kivien lista, dedup), joten yksittäisten heittojen häviämistä/löytymistä ei voi yksiselitteisesti
lukea trackerin ansioksi; mean-shift-ajo ei myöskään löytänyt kolmea heittoa, jotka ristikkohaku löysi.

## Missä aika oikeasti kuluu (jatkosuunta)

Profilointi (`tulokset/lm_profilointi.txt`): päivitys = prep 7,5 + haku 5,3 + **LM-tarkennus 25,5 ms**; LM #1 on 80 %
tarkennuksesta, ja ulompi silmukka ajetaan keskimäärin 3,6 kertaa, koska suppenemisraja
`BOUNDARY_CONVERGENCE_CM` on 0,05 cm (kaukana ~0,005 px). Kokeessa rajan nosto 0,5 cm:iin lyhensi tarkennuksen
25,5 → 19,7 ms (−23 %) ja tulos oli 99 %:ssa päivityksistä identtinen (p99-ero 0,64 px). Sitä **ei ole otettu
käyttöön** tässä kansiossa. Koko putken tasolla muut isot erät ovat varjonvaimennus (~22–29 ms/ruutu),
stabilointi (~15–24) ja valotasapaino (~10–13).


## Kaikkien kivien seuranta (jatkotyö: miksi osa heitoista hukkui ja mitä tehtiin)

Ensimmäinen mean-shift-versio löysi 12/16 heittoa (ristikkohaku 9/16), mutta kaksi ristikkohaun löytämää heittoa
(ruudut 3500–4002 ja 5940–6685) puuttui. Syyt selvitettiin HAKU-lokilla (`HAKU_LOG`) ja simulaattorilla:

1. **Päällekkäisyyden esto hylkäsi oikean kiven.** HAKU löysi kiven molemmissa ajoissa, mutta `NEW_STONE_DEDUP_CM`
   (100 cm) esti rekisteröinnin, koska lähellä oli *vahvistettu roskarata* (ruudussa 3500: rata 124, 38 cm päässä;
   ruudussa 5940: rata 164, 49 cm päässä). Ristikkohakuajossa esto meni ohi vain sattumalta (100,2 cm). Etäisyys, tarkka-osuus
   tai radan liike **eivät erottele** oikeaa kaksoisosumaa roskaradasta (kaukana aidon kiven tarkka-osuus 0,23 vs roskan 0,09).
   Korjaus: esto vain alle 20 cm, ja kaksi rataa jotka ovat < 30 cm päässä toisistaan 3 peräkkäistä framea yhdistetään
   (uudempi/vahvistamaton poistetaan).
2. **Seuranta menettää kiven häiriökohdassa.** Simulaattorissa (kiven seuranta varhaisimmasta havainnosta) sekä
   ristikkohaku että mean-shift pysähtyivät heiton alussa (T1, ruutu 3480: peitto 4–6 %), koska maski on hetken huono
   (rms 9–12 px) ja kivi liikkuu ~8 cm/ruutu. Lisäksi kummallakin on oma virhetapansa, kun kiven viereen/taakse osuu
   pelaaja: mean-shift valuu pelaajan maskin massaan (T2), ristikkohaku ajautuu taaksepäin (T4). Molemmissa virhe on liike
   *taaksepäin* (Y kasvaa), mikä on fysikaalisesti mahdotonta.
3. **Ratkaisu: yhdistelmähaku (`TRACKER_MODE=ensemble`, `locate_mode=5`).** Kivikohtainen **liike-ennuste** (vakionopeus
   viimeisten 12 framen havainnoista) → ehdokkaat: ristikkohaku ennustetusta keskipisteestä, mean-shift viimeisestä
   paikasta ja mean-shift ennustetusta paikasta. Valinta = pistemäärä − 0,15/10 cm taaksepäin (yli 2 cm) − 0,02/10 cm
   poikkeamasta ennusteesta. Simulaattorin neljä heittoa (T1–T4): ristikko 0,04/0,99/1,00/0,01, ensimmäinen mean-shift
   0,06/0,00/1,00/1,00, **yhdistelmä 0,98/0,99/1,00/1,00** (peitto < 30 cm). Rangaistusten arvot ovat vakaita
   (back 0,15–0,3, pred 0–0,02); pred 0,05 rikkoo T4:n. Kokeiltu ja hylätty: kehän painon nollaus/negatiivinen,
   epäsymmetrinen hakulaatikko, ennuste-prioriehdokas, reunavarahaku (eivät korjanneet T2:ta tai huononsivat muita).

### Tulos koko videolla (MAH00014_leikattu.mp4, unioni 16 todellista heittoa)

| | ristikkohaku | ensimmäinen mean-shift | yhdistelmä | yhdistelmä + esto/yhdistys (oletus) |
|---|---|---|---|---|
| heittoja katettu | 9/16 | 12/16 | 15/16 | **16/16** |
| aitoja ratoja (liike > 5 m) | 11 | 14 | 19 | 20 |
| CSV-rivejä aidoilla radoilla | 5 687 | 7 605 | 10 538 | 10 647 |
| roskarivien osuus | 67 % | 51 % | 57 % | 57 % |

Kuva: `tulokset/kuvat/kattavuus_heitot.png`; heittokohtaiset kuvat `tulokset/kuvat/yhdistelma_heitto_*.png`;
taulukko `tulokset/kattavuus_yhdistelma.txt`. Yhdistelmän kustannus tyhjällä koneella (ilman ennustetta): haku 6,4 ms vs
ristikko 5,3 ms (+1 ms / päivitys, +1,7 %); koko putki seuraa enemmän kiviä (2,5 vs 2,0 kiveä/kutsu), joten kokonaisaika
kasvaa. Rinnakkain ajettujen ajojen absoluuttiset ajat eivät ole vertailukelpoisia.

**Varaukset:** yksi video; "kaikki heitot" on määritelty *unionina* kaikista ajoista (sääntö: pitkä, laadukas rata), joten
heitto jonka mikään ajo ei löydä jää huomaamatta; heittojen määrä kasvoi 13 → 16 menetelmien parantuessa. Simulaattori käyttää
tol 0,1 px / 10 iteraatiota, koko putken ajot oletuksia 0,3 px / 6 iteraatiota. Roskaratoja (pelaajat) syntyy edelleen ja ne
ovat > 50 % riveistä; niiden karsinta ei kuulunut tähän työhön.

## Heittoportti ja monihaku (käyttäjän hogline-ylitysaikalista, 26 heittoa)

Käyttäjän antama lista (karkeat kaukaisen hoglinen ylitysajat ±50 ruutua, joka kerta vain yksi kivi) toimi tavoitteena:
*poimi kaikki nämä heitot ja mahdollisimman vähän muita.* Yhdistelmähakuajo (yllä) löysi niistä 20/26, ja portittamatta CSV:ssä
oli 252 rataa (20 aitoa).

**Mitä puuttui ja miksi:** viisi heittoa (2425, 7350, 9250, 11950, 12375) ei ollut seurannassa lainkaan, koska HAKU palautti
yhdellä kutsulla vain ensimmäisen kelvollisen ehdokkaan - kiven vieressä oleva pelaaja/lakaisija voitti kilpailun ja kivi jäi
rekisteröimättä. Yksi (15525) oli seurannassa, mutta sovitus oli huono (lakaisija peittää kiven, rms 9,7 px).

**Muutokset:**
1. **Monihaku** (`search_new_stones`, `HAKU_MULTI=1`): HAKU palauttaa kaikki kelvolliset ehdokkaat (enintään `HAKU_MAX_RESULTS`=4,
   `HAKU_MAX_ATTEMPTS`=8); jokainen rekisteröidään (ei kaksoisrekisteröintiä < 30 cm, enintään 8 samanaikaista kiveä).
   Tämä nosti kattavuuden 20/26 → 25/26.
2. **Heittoportti** (`THROW_GATE=1`): rivit kirjoitetaan varsinaiseen CSV:hen vasta videon lopussa, kun rata on koko elinkaarensa
   perusteella *heitto*: matka ≥ 1500 cm, loppu-Y ≤ 1100 cm (lähihogline 823 + marginaali), ≥ 300 riviä, **hidastuu**
   (loppunopeus / alkunopeus ≤ 0,6) sekä sovitus joko hyvä (rms-mediaani ≤ 3 px ja tarkka-osuus ≥ 0,3) tai heikko mutta
   kelvollinen (rms ≤ 12 ja tarkka-osuus ≥ 0,1, esim. lakaisija peittää kiven). Lisäksi **yksi rata / hogline-ylitys**: kaksi
   rataa joiden ylitysajat < 40 ruutua toisistaan ovat sama heitto, ja parempi säilyy. Raaka CSV kirjoitetaan viereen
   (`*_raaka.csv`) omaa jälkisuodatusta varten.
   - Hidastuvuusehto poistaa pelaajan päätä seuraavan radan (211, ylitys 3182: nopeussuhde 1,05, päättyy Y = 1372), jonka
     pyöreä pää läpäisi muuten sovituskriteerit; aidoilla heitoilla suhde on 0,19–0,41.
   - 100 ruudun yhdistysraja pudotti ruudun 2425 heiton (kaksi oikeaa heittoa vain 70 ruudun välein) → raja on 40.

**Tulos (koko putki, oletusasetukset):** 26 heittoa portitetussa CSV:ssä, **26/26 vastaa listaa**, ylitysajan ero listaan −65…+40
ruutua (`tulokset/heittoportti.txt`), listaan kuulumattomia ratoja 0; rivejä 14 116 (raaka 31 045). CSV:
`tulokset/heitot_MAH00014_leikattu.csv`. Kuvat jokaisen ylityksen kohdalta: `tulokset/kuvat/hogline_ylitykset/`
(yleiskuva + nauhat ylitys −15/0/+15 ruutua; keltainen viiva = hogline, vihreä rengas = rata).

**Silmämääräinen tarkistus kuvista ja heitto #4:** ensimmäisessä versiossa #4 (ruutu 2360, rata 158) oli väärin: rengas oli
heittäjän jalan päällä. Selvitys (`tulokset/kuvat/hogline_ylitykset/heitto4_*.jpg`): seuraavaa kiveä ei ollut vielä
heitetty ruudussa 2360 - HAKU rekisteröi ehdokkaan heittäjän jalalta, rata menetti kohteen (65 ruudun rivitön aukko) ja lukkiutui
oikeaan kiveen vasta ruudussa 2439 (kivi lähtee heittäjältä hoglinen yli). Vasemmalla näkyvä kivi oli edellinen heitto (#3).
**Korjaus:** radan alku pudotetaan, jos sen jälkeen on yli 30 ruudun rivitön aukko ja alku on lyhyt (< 40 riviä)
(`GATE_HEAD_GAP_FRAMES`, `GATE_HEAD_MAX_ROWS`), ja jos rata alkaa hoglinen alapuolelta, ylitysaika ekstrapoloidaan taaksepäin radan
alun nopeudesta (enintään 60 ruutua). Vain rata 158 kärsi tästä (muiden aukot < 20 ruutua). Uusi ylitys 2411 (lista 2425, ero
−14); kaikkien 26 ylitysaikojen ero listaan ≤ 28 ruutua. Ekstrapolointi on likimääräinen: kivi liikkuu vapautuksessa aluksi hitaammin
(Y 2856 → 2863 ruuduissa 2439–2447), joten todellinen ylitys on ~10–25 ruutua ekstrapoloitua myöhemmin. Portti voidaan ajaa
raaka-CSV:lle uudelleen ilman videota: `python tools/regate_csv.py raaka.csv ulos.csv` (parametrit `GATE_*`).
Muut ylityskuvat: rengas on kiven päällä; kolmessa (#17, #19, #23) rengas on ruudussa ylitys −15 vielä tyhjällä jäällä (kivi ei
näy heittäjän takaa). Portin arvot on säädetty **yhdellä videolla ja tällä listalla** (esim. heikon sovituksen väylä osuu vain
heittoon 15525), joten yleistettävyys toiselle videolle on todentamatta.
