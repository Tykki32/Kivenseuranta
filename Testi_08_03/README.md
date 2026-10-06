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
python main.py --video MAH00014.mp4 --debug --full                               # seuranta pysähtymiseen asti
python main.py --live --paneelit D:\Tikku\Suorita\MAH00014_leikattu_panel_corners.txt --katselu --live-kansio D:\Tikku\Live
python main.py --live-sim video.mp4 --live-sim-tahti 0 --paneelit paneelit.txt     # testaus ilman kameraa
python tools/kamera_testi.py                                                       # kameran kartoitus
```

`python main.py --help` listaa kaikki valitsimet (`--start/--end`, `--max-frame`, `--live-*`, `--katselu [portti]`).

**`--full`:** vahvistettua kiveä seurataan lähihogin jälkeenkin, kunnes se pysähtyy (toimii myös live-tilassa).
- Debug-ikkunan (debug-video ja puhelinnäkymä) tulospaneeliin tulee rivi `pysahtyi: x; y`: kiven keskikohdan
  etäisyys keskiviivasta (x) ja lähemmän pesän T-viivasta (y), senttimetreinä. Debug-ikkunassa vasemmalla ja
  alhaalla luvut ovat negatiivisia, oikealla ja ylhäällä (kohti kaukaista päätä) positiivisia. Ennen pysähtymistä
  rivillä on `pysahtyi: -`.
- Nollakohta on `asetukset.py`:ssä: `PYSAHDYS_NOLLA_X_CM` (siirto keskiviivasta, + = fyysinen +X) ja
  `PYSAHDYS_NOLLA_Y_CM` (siirto T-viivasta, + = kohti kaukaista päätä). Oletus 0 / 0 = keskiviiva ja T-viiva.
- Hog-CSV:hen tulee sarakkeet `pysahtyi_x_cm`, `pysahtyi_y_cm` (nollakohdasta; x fyysisessä +X-suunnassa, ei
  debug-ikkunan puolen mukaan) ja `pysahtyi_X_cm`, `pysahtyi_Y_cm` (fyysinen koordinaatisto).
- Hog-hog-analyysi ja kierrearvio tehdään kuten ennenkin lähihogilla; kierrepiirteitä ei kerätä lähihogin jälkeen.
- Pysähtyminen: kun kivi on liikkunut alle 20 cm sekunnissa, sitä seurataan vielä, kunnes viimeisen sekunnin 5
  ensimmäisen ja 5 viimeisen paikan mediaanit ovat alle `A.PYSAHTYNYT_TARKKA_CM` (3 cm) toisistaan (enintään 3 s).
- Vahvistus: pysähtymispaikka annetaan vain kiville, jotka oikeasti pysähtyvät. Kiven on pysyttävä näkyvissä ja
  paikallaan (5 viimeisen paikan mediaani alle `A.PYSAHDYS_MAX_SIIRTO_CM` = 10 cm pysähdyspaikasta; yksittäinen huono
  sovitus ei hylkää) vielä `A.PYSAHDYS_VAHVISTUS_S` (2 s). Jos kivi liikkuu tai katoaa
  sinä aikana (pelaaja pysäyttää kiven jalalla tai harjalla tai vie sen), pysähtymispaikkaa ei anneta
  (`pysahtyi: -`). Paikka on vahvistusjakson paikkojen mediaani.

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

Mitä kukin osio tekee ja miten asiat lasketaan, on kerrottu luvussa *Toiminta osioittain* alla.

## Toiminta osioittain

Tämä luku kertoo, mitä kukin `asetukset.py`:n osio tekee ja miten asiat lasketaan. Asetusten nimet viittaavat
`asetukset.py`:hyn (`A.NIMI`); suluissa olevat arvot ovat oletuksia.

### Koordinaatisto ja kokonaiskuva

**Rata (cm):** X = 0 keskiviivalla, Y kasvaa kohti kaukaista päätä eli heittopäätä. Lähemmän pesän (kameran puoli)
takareuna on Y = 0, sen T-viiva (keskipiste) Y = 182,9 ja kaukaisen pesän T-viiva Y = 3657,6 (T-viivojen väli 34,747 m).
Hoglinjat ovat nimellisesti 6,40 m omasta T-viivastaan: lähihog Y = 822,9 ja kaukohog Y = 3017,6. Heitetty kivi
liikkuu siis Y:n **pienenemisen** suuntaan. Top-down-tarkistuskuvassa on 2 px/cm, X −200…200 cm ja Y 0…4000 cm.

**Kamera:** kameramalli on `pose = {K, R, t}`, ja jääpisteen (X, Y, Z) projektio kuvaan on p = K (R·(X, Y, Z) + t).
Objektiivin säteittäinen vääristymä on yksiparametrinen (k1). Kuva oikaistaan `cv2.remap`-kartoilla ennen seurantaa.

**Ajo etenee kolmessa vaiheessa samalla videon peräkkäisellä luvulla (`seuranta.py`):**
1. **Kalibrointi**: ensimmäisen 120 s ajalta kerätään moodikuva (tyhjä rata) ja kamera kalibroidaan siitä.
2. **Kiviprofiili**: liikkuvista kivistä opetellaan kiven 3D-muoto.
3. **Seuranta**: joka ruudulla haetaan uusia kiviä (HAKU) ja päivitetään liikkuvien kivien paikat (SEURANTA).
   Radoista lasketaan hog-hog-analyysi, ja lopuksi heittoportti kirjoittaa heitot CSV:hen.

---

### 1. Kalibrointi (`seuranta.py`, `kalibrointi.py`, `kalibrointi_perus.py`, `paneelit.py`, `cpp/mode_engine.cpp`)

**Paneelistabilointi moodikuvan keruun ajan.** Radan reunan mainospaneelit (vaalea suorakaide tummalla pohjalla) ovat
stabiloinnin ankkureita. Paneelit tunnistetaan ensimmäisestä ruudusta referenssitiedoston (`*_panel_corners.txt`)
läheltä. Puuttuvat paneelit voi klikata käsin, jos automaattisesti löytyy alle 6 paneelia.

Jokaisessa ruudussa kunkin paneelin keskipiste haetaan edellisen paikan ympäriltä ±100 × ±120 px:n alueelta, rinnakkain
`A.PANEELI_SAIKEET` (8) säikeessä:
- Kynnys on keskikohdan harmaasävy (5×5-keskiarvo) + 40. Pikselit kynnyksen alla muodostavat binäärikuvan, jolle tehdään
  3×3-avaus ja -sulkeminen.
- Valitaan yhtenäinen alue, johon keskipiste osuu (tai lähin riittävän iso alue, 100–250 000 px). Sen kulmat tarkennetaan
  alipikselitarkkuudella.
- Jos paneeli hyppää yli `A.KALIB_PANEELI_MAX_HYPPY_PX` (20 px) edellisestä paikasta, se ohitetaan tässä ruudussa.

Paneelien referenssipaikoista ja nykyisistä paikoista sovitetaan similariteettimuunnos (siirto, kierto, skaala;
`cv2.estimateAffinePartial2D`, RANSAC 3 px). Stabilointimatriisi on sen käänteismuunnos. Matriisi suodatetaan
elementeittäin mediaanilla viimeisten `A.KALIB_STAB_MEDIAANI_RUUTUA` (20) ruudun yli: tärinä poistuu, mutta hidas ajautuminen
seurataan.

**Moodikuva (C++ `ModeEngine`).** `A.KALIB_MOODI_S` (120 s) ajalta otetaan näyte `A.KALIB_NAYTEVALI_S` (5 s) välein,
eli enintään 25 stabiloitua ruutua.
- Moodi lasketaan koko pikselille (B, G, R yhdessä), tiileissä (`A.MOODI_TIILI` = 128 px) rinnakkain enintään 8 säikeessä.
- Kunkin pikselin näytteet ryhmitellään: näyte kuuluu ryhmään, jos jokainen kanava on enintään 10 yksikön päässä ryhmän
  ensimmäisestä jäsenestä.
- Tulos on suurimman ryhmän keskiarvo; tasatilanteessa voittaa pienempi edustaja-arvo. Kanavakohtainen moodi voisi
  tuottaa värin, jota mikään näyte ei sisältänyt.
- Lopuksi 3×3-mediaanisuodin poistaa yksittäiset poikkeavat pikselit, joita syntyy ohuilla viivoilla.

Moodikuva on "tyhjä rata": pelaajat ja kivet ovat eri kohdissa eri näytteissä, joten suurin ryhmä on tavallisesti tyhjä jää.

**Live-tilassa** samat näytteet otetaan myös kameran täydellä resoluutiolla (1920×1080). Niiden mediaanikuva kalibroidaan,
ja tulos skaalataan seurannan resoluutiolle (`scale_calibration`). Jos täyden resoluution kalibrointi epäonnistuu,
kalibroidaan 1280×720-moodikuvasta.

**Lähemmän pesän tunnistus (`kalibrointi_perus.py`):**
- **Värimaskit HSV:stä:**
  - sininen: H 90–140, S ≥ 60, V ≥ 40
  - punainen: H 0–10 tai 170–180, S ≥ 60, V ≥ 40
  - molemmille 5×5-avaus ja -sulkeminen.
- **Renkaat:** maskien kontuureihin sovitetaan ellipsit (`cv2.fitEllipse`).
  - Hyväksyntä: pinta-ala ≥ 1000 px (punainen 50), akselisuhde ≥ 0,20 (punainen 0,15).
  - Suurin ellipsi on ulkorengas. Sen sisällä oleva, vähintään 30 % (punainen 15 %) kokoinen ellipsi on sisärengas.
  - Tuloksena neljä ellipsiä: sininen ulko (r 182,9 cm), sininen sisä (121,9), punainen ulko (61,0) ja punainen sisä (15,2).
- **Viivat:** viivasegmentit haetaan kahdesta lähteestä, joiden segmentit yhdistetään:
  - harmaasävy-Canny pesän ympäriltä;
  - jääsuodatus (jäätä on S < 22 ja V > 165; kaikki muu, myös haalea viiva, erottuu binäärikuvana).
- **T-viiva ja keskiviiva:** pystysuuntainen segmenttipari on T-viiva ja vaakasuuntainen pari keskiviiva. Pari on
  segmentit pesän keskipisteen molemmin puolin. Jos toista puolta ei näy (pesä kuvan reunassa), käytetään parasta
  yksittäistä segmenttiä peilattuna. T-viivan ja keskiviivan leikkaus on pesän keskipiste.
- **17 pistevastaavuutta:** pesän keskipiste sekä kummankin viivan leikkaukset kunkin neljän ellipsin kanssa (4 × 2 × 2).
  Jokaisen pisteen fyysinen paikka tunnetaan renkaan säteestä.

**Objektiivin vääristymä k1.** k1 haetaan karkeasta hienoon (±0,6, 25 askelta, 3 kierrosta, alue kapenee 6-kertaisesti).
Jokaisella k1:llä 17 pistettä oikaistaan ja niihin sovitetaan homografia; voittaa pienin jäännösvirhe (RMS).
- k1 hyväksytään vain, jos se pienentää RMS:ää yli 8 % ja |k1| > 0,01.
- Jos kaukaista pesää ei löydy k1:n kanssa, yritetään k1 = 0.

**Kaukaisen pesän etsintä:**
- **Laajennettu kuva:** pelkän lähipesän homografialla tehdään 10 m takarajan yli ulottuva top-down-kuva.
- **Riviskannaus:** kuva skannataan rivi kerrallaan. Kaukaisen pesän rivillä on sinistä keskiviivan molemmin puolin
  (yli 5 px keskeltä) ja punaista niiden välissä. Toisin kuin "suurin sininen alue", kuvio ei osu mainospaneeleihin.
- **Pystysuuntainen venytys:** koko kuvaa venytetään pystysuunnassa (ankkurina lähipesä) niin, että löydetty rivi osuu
  kaukaisen T-viivan Y:hyn.
- **Keskipiste:** kaukaisen pesän cropista sovitetaan ympyrät kolmelle renkaalle.
  - Sinisen ja punaisen S-alaraja säädetään niin, että ROI:sta (sinisen pienin ympäröivä ympyrä) 31 % on sinistä ja
    11 % punaista.
  - Rengaspisteet haetaan säteittäin: kunkin suunnan sisä- ja ulkoreuna; peitetyt suunnat jäävät pois.
- **Alustava homografia:** lähipesän 17 pistettä + kaukaisen pesän keskipiste.

**Geometrinen tarkennus (iteroiden, enintään 15 kierrosta, pysähtyy kun RMS ei parane 3 %:lla kahteen kertaan).**
Joka kierroksella top-down-kuvasta mitataan uudet havainnot, ja koko homografia (8 vapausastetta) sovitetaan yhdellä
painotetulla Levenberg–Marquardt-sovituksella. Havainnot:
- **Lähipesän 17 pistettä:** tavallisia pistevastaavuuksia (fyysinen sijainti tunnetaan).
- **Kaukaisen pesän rengaspisteet:** ympyrärajoitteita (etäisyys pesän keskipisteestä = renkaan säde). Värimaskit
  kuten edellä, enintään 30 pistettä/rengas.
- **Hoglinjat:** viivarajoitteita (Y = hoglinjan Y).
  - Nimellisen hoglinjan ympäriltä otetaan ±50 cm kaista. Harmaasävykynnystä nostetaan, kunnes 98 %:ssa sarakkeista on
    tummaa.
  - Kunkin sarakkeen piste on tummien rivien tummuudella painotettu keskiarvo.
  - Jos kaukohogin kulma poikkeaa lähihogista yli 5°, kaukohogin pisteet jätetään pois.
- **Keskiviiva koko radalta:** rajoitteita X = 0.
  - Joka 20 cm (Y 150–3900) otetaan ±8 cm rivikaistan keskiarvoprofiili X:n suhteen ja haetaan tummin kohta ±15 cm
    alueelta.
  - Vaatimukset: tummuus ≥ 4 harmaasävyä taustaa tummempi ja leveys (puolikkaan tummuuden kohdalta) < 16 cm.
    Vaakaviivat (hog-, T- ja takaviiva) tummentavat koko kaistan, joten niistä ei synny laaksoa.
  - Poikkeavat pisteet pois: |X − mediaani| > max(3 MAD, 3 cm).
- **Painot:** kunkin ryhmän painona on 1 / sen oma robusti kohinataso (MAD, vähintään 1 cm), joten mikään ryhmä ei
  hallitse sovitusta.

**Hoglinjojen todellinen paikka.** Sovitus tehdään kahdesti:
- hoglinjat nimellisessä paikassa;
- hoglinjat saavat siirtyä ±`A.KALIB_HOGLINJA_TOLERANSSI_CM` (20 cm).

Molemmat hoglinjat ovat samalla etäisyydellä omasta T-viivastaan, joten niillä on yksi yhteinen siirtymä d. Se on
mittausten (mediaani-Y − nimellinen) painotettu keskiarvo; kaukohogin paino on `A.KALIB_KAUKOHOG_PAINO` (0), koska 30 m
päässä paikka määräytyy heikosti.

Vapaampi versio valitaan vain, jos pesien laatu ei huonone. Laatu mitataan molemmista pesistä top-down-kuvassa:
- pyöreys min(w, h)/max(w, h) ei saa huonontua yli 0,02;
- koon virhe ei saa kasvaa yli 1,5 %.

Valitut hoglinjat asetetaan `rata.set_hogline_positions()`:lla, ja kaikki myöhempi laskenta käyttää niitä.

**Kameran 3D-asento (K, R, t):**
- **Alkuarvaus:** homografia muutetaan taso-homografiaksi (fyysinen cm → kuva) ja puretaan Zhangin menetelmällä
  (H = K [r1 r2 t]; lähin ortonormaali R SVD:llä). Kahdesta peilikuvaratkaisusta valitaan se, jossa kamera on jään
  yläpuolella.
- **Polttoväli f:** haetaan karkeasta hienoon lähipesän 17 pisteellä.
- **Hienosäätö:** asento sovitetaan samoilla painotetuilla rajoitteilla (piste, ympyrä, viiva) kuin homografia.

---

### 2. Kiviprofiilin opettelu (`profiili.py`, `kivimalli.py`, `cpp/profiili.hpp`)

**Kivimalli.** Graniittirunko on pyörähdyskappale. Normalisoitu profiili r(z) (alapuolisko mitattu oikeasta kivestä,
yläpuolisko peilikuva) skaalataan kiven säteellä R_max ja korkeudella H_total. Väliarvot saadaan Catmull–Rom-splinillä.
Muotoa voi muuttaa neljällä muotopoikkeamalla (shape_deltas). Kahvan kohdalla kiven yläpinnassa on lovi: ympyrä, jonka
säde on handle_r_frac × R_max.

Seurannassa käytetään mallin pisteitä kiven omassa koordinaatistossa (28 kulmaa × renkaat). Ne vain siirretään paikkaan
(X, Y) ja projisoidaan. Kiven ääriviiva kuvassa on projisoitujen pisteiden kupera peite.

**1) Tiheä skannaus (C++ `scan_stone_candidates`)** `A.PROFIILI_SKANNAUS_VALI_S` (0,2 s) välein pelialueelta
(X ±200 cm, Y 400…2000 cm):
- **Taustanvaimennus:** pikselit, joiden harmaasävyero moodikuvaan on < 30, valkaistaan.
- **Graniittimaski:**
  - Kivi on matalakylläinen ja paikallista taustaa tummempi: S < 60 ja tummuus > 15.
  - Tummuus = (harmaasävy sumennettuna, σ 25 px) − harmaasävy. Sumennus lasketaan 4× pienennetystä kuvasta.
  - Lopuksi 5×5-avaus ja 3×3-sulkeminen.
- **Kandidaatit:** maskin kontuureihin sovitetaan ellipsi.
  - Ehdot: pinta-ala 80–200 000 px, täyttöaste (pinta-ala / ellipsin ala) ≥ 0,4, akselisuhde ≥ 0,15.
  - Jääpaikka saadaan ellipsin keskipisteen säteen ja jäätason (Z = 0) leikkauksesta.
- **Ketjutus:** kandidaatit ketjutetaan lyhyiksi radoiksi (lähimmät parit ensin, raja 30 cm + 12 cm/ruutu).
- **Siemen:** rata, jolla on vähintään 4 havaintoa ja joka on liikkunut radan suuntaan yli `A.PROFIILI_LIIKE_CM` (15 cm)
  2 s:ssa (|dY| > |dX|).

**2) Siemenen seuranta ajassa** molempiin suuntiin. Lähin kandidaatti kelpaa, jos hyppy on ≤ 45 cm; 10 perättäistä
hutia lopettaa.
- **Esitarkistus:** ensin ±`A.PROFIILI_ESITARKISTUS_S` (3 s). Viidestä havainnosta sovitettu profiili ei saa olla
  huonompi kuin RMS 8 px, ja R_max on oltava 12–16 cm. Muuten kohde ei ole kivi.
- **Koko ikkuna:** sitten ±`A.PROFIILI_IKKUNA_S` (30 s). Radalta otetaan tasavälein `A.PROFIILI_HAVAINTOJA_KIVESTA` (40)
  havaintoa.

**3) Alfa-ääriviiva (C++ `alpha_observation_cpp`).** Kiven reuna lasketaan alipikselitarkasti peittävyydestä
binäärimaskin sijaan. Lasketaan kiven ympäriltä:
- **Kiven alue:** valotasapainokorjaus ja varjosietoinen taustanvaimennus (osio 9). Graniittimaski lasketaan tarkalla
  σ 25 -sumennuksella; valitaan kiveä lähin yhtenäinen alue (≥ 60 px).
- **Kirkkaussuhde:** k = V / V_moodikuva (V = max(B, G, R)). Graniitin taso k_g on kiven sisäosan (eroosio 3×3)
  mediaani-k.
- **Jään taso k_s:** paikallinen, kiven ympäriltä 15…45 px:n renkaasta niistä pikseleistä, joille k > k_g + 0,25.
  Painotettu σ 5 -sumennus, rajattu välille [k_g + 0,25, 1,05].
- **Peittävyys:** alfa = (k_s − k) / max(k_s − k_g, 0,15), rajattu välille 0…1.
- **Ääriviiva:** alfa-kartta ylinäytteistetään 4× (bikuubinen). Tasa-arvokäyrä yhteisellä alfa-tasolla on kiven
  reuna, rajattuna graniittimaskiin (laajennus 5 px).

Hylkäykset:
- tumma alue kiven ympärillä (V < 60 yli 100 px 9…31 px:n renkaassa: harja tai kenkä);
- graniitti/jää-kirkkaussuhde välin 0,2–0,8 ulkopuolella;
- ääriviivan pinta-ala poikkeaa yli 30 % ±4 lähiruudun mediaanista.

Yhteinen alfa-taso on havaintojen graniitti/jää-suhteen mediaani (oletus 0,434).

**4) Profiilin sovitus (C++ `fit_stone_profile_cpp`).** Levenberg–Marquardt sovittaa samanaikaisesti:
- profiilin: R_max, H_total 11,43–15 cm, 4 muotopoikkeamaa (regularisointi 60) ja lovi 0,30–0,95;
- jokaisen havainnon paikan (X0, Y0).

Jäännöksenä on havaitun ääriviivan pisteiden etumerkillinen etäisyys mallin (lovellisen) ääriviivan reunaan. Jakobiaani
lasketaan harvasti: havainnon paikka vaikuttaa vain sen omaan jäännöslohkoon.

Hyväksyntä:
- Yksittäinen kivi hyväksytään kokoelmaan, jos sen oma profiili on RMS ≤ `A.PROFIILI_YKSI_MAX_RMS_PX` (3 px). Pelaaja
  ei näin saastuta kokoelmaa.
- Koko kokoelman profiili on valmis, kun RMS ≤ 1,5 px, vähintään 15 havaintoa, R_max 12,5–15 cm ja kiviä vähintään
  `A.PROFIILI_MIN_KIVIA` (2).

---

### 3. HAKU ja SEURANTA (`seuranta.py`, `cpp/haku.hpp`, `cpp/seuranta.hpp`, `cpp/geometria.hpp`, `cpp/sovitus.hpp`, `cpp/meanshift.hpp`, `cpp/siluetti.hpp`)

**Syötekuva.** Seurannan kuva on oikaistu, stabiloitu ja valotasapainotettu ruutu, jonka tausta on valkaistu (osio 9).
Kivi on siinä "ei-valkoista".

**Maskit kiven paikantamiseen:**
- **Graniittimaski:** S < 60 ja tummuus > 15 (kuten edellä), 5×5-avaus ja 3×3-sulkeminen.
- **Etualamaski:** ei-valkaistut pikselit.
- SEURANTA käyttää näiden yhdistelmää (OR). Pelkkä graniittimaski on kaukana harva, jolloin runkopisteitä on vähän ja
  paikka hyppii.
- Saturaatio lasketaan alkuperäisestä kuvasta, koska valkaisu loisi keinotekoisen saturaatiorajan kiven reunalle.

**Peitto-osuus (pistemäärä).** Kiven ennustettu ääriviiva (hulli) piirretään maskin päälle; sen ympärille piirretään
1,1-kertainen turvavyöhyke.
- Pistemäärä = (maskipikselit hullin sisällä / hullin ala) − (maskipikselit vyöhykkeellä / vyöhykkeen ala).
- Pyöreä kivi saa lähes 1, koska tumma alue loppuu mallin reunaan. Pelaaja tai kaksi vierekkäistä kiveä saa
  rangaistuksen, koska maskia on myös vyöhykkeellä.

**HAKU (uudet kivet, C++ `search_new_stones`, oma säie joka `A.HAKU_VALI_RUUTUA`:s ruutu):**
- **Hakualue:** X ±70 cm, Y (nimellinen kaukohog − 100) … (kaukohog + 300) cm.
- **Ristikkohaku:**
  - Karkea ristikko 10 cm:n askelin, sitten hieno ristikko 2 cm:n askelin parhaan pisteen ympärillä.
  - Alue on kapea ja kaukana, joten hulli lasketaan kerran alueen keskellä ja siirretään muihin pisteisiin paikallisella
    Jakobiaanilla (px/cm).
  - Paras alle `A.HAKU_PISTEKYNNYS` (0,15) lopettaa haun.
- **Tarkennus (`refinePositionJoint`):**
  - **Runkokontuuri:** lähin maskikontuuri (painopiste ≤ 40 px, ala ≥ 15 px). Siitä otetaan pisteet, joiden ulkopuolella
    2,5 px päässä on matalakylläistä jäätä (kahvan kohta jää pois); yli 40 pisteestä joka toinen.
  - **Rengaspisteet:** 72 suunnassa ensimmäinen kohta, jossa saturaatio laskee alle 70. Vaaditaan ≥ 12 pistettä ja
    ≥ 180° kattavuus.
  - **Sovitus:** Levenberg–Marquardt sovittaa (X, Y, renkaan säde) niin, että runkopisteiden etäisyys mallin ääriviivaan
    ja rengaspisteiden etäisyys projisoituun renkaaseen minimoituvat.
  - **Poikkeamat:** pisteet, joiden jäännös poikkeaa mediaanista yli 3 × 1,4826 × MAD, pois ja uusi sovitus. Toistetaan
    uusilla rengaspisteillä, kunnes paikka muuttuu alle 0,5 cm (enintään 5 kertaa).
  - **Linearisointi:** LM on linearisoitu (projektio on lineaarinen X:n ja Y:n suhteen ennen jakoa). Tulos tarkistetaan
    tarkalla jäännösfunktiolla ja ajetaan tarvittaessa tarkasti.
  - **tarkka:** sovitus onnistui eikä siirtynyt yli 25 cm lähtöpisteestä.
- **Hyväksyntä:**
  - Tarkennetun paikan peitto-osuus ≥ 0,20.
  - Spawn-suodatin: rms ≤ `A.HAKU_UUSI_MAX_RMS` (3,5 px), runkopisteitä 14–45, |X| ≤ 65 cm, ristikkopistemäärä ≤ 0,9.
- **Uudelleenyritykset:** hyväksytyn tai hylätyn kohteen koko yhtenäinen maskialue poistetaan (floodFill) ja haetaan
  uudelleen. Näin lähempänä oleva isompi pelaaja ei peitä kauempana olevaa kiveä. Tuloksia enintään 4, yrityksiä 8.
- **Python-tarkistukset ehdokkaalle:**
  - **Siluettitarkennus:** jos alle 40 % maskista osuu siluetin sisään, ehdokas hylätään.
  - **Jo seurattu:** alle 20 cm vahvistetusta radasta.
  - **Sama haku:** alle 30 cm saman haun toisesta ehdokkaasta.
  - **Heittokiven takana:** 50–800 cm liikkuvan heittokiven takana ja alle 100 cm sivussa (heittäjä tai harjaaja).
- **Paikkojen täyttyminen:** jos `A.MAX_KIVIA` (8) on täynnä, huonoin rata poistetaan. Ensin vahvistamaton, jolla on
  huonoin rms-mediaani; sitten vahvistettu, jos rms-mediaani ≥ 8 px. Radan suuntaan liikkuvat radat ovat suojattuja.

**Siluettitarkennus (`cpp/siluetti.hpp`).**
- Kiven 3D-mallin siluetti (kupera peite miinus kahvan lovi) piirretään 2× ylinäytteistettynä. Sitä siirretään
  kuvatasossa ±6 px (SEURANTA ±3 px) 0,5 px:n askelin.
- Siirron pistemäärä:
  - \+ maskipikseleiden osuus siluetin sisällä;
  - − ylitulo siluetin ulkopuolelle siluetin ympärillä olevalla 3 px:n kaistalla (paitsi kiven yläpuolella);
  - \+ 0,1 · exp(−(d / σ)²), missä d on maskin ja siluetin massakeskipisteiden etäisyys ja σ = 0,3 √ala.
- Maski on oma graniittimaski 3×3-avauksella (5×5 poistaisi kaukaisen, 5–6 px leveän kiven).
- Paras siirto muutetaan senteiksi Jakobiaanilla.

**SEURANTA (liikkuvat kivet, C++ `track_stones_batch`, kivet rinnan omissa säikeissään):**
- **Hakualue** kasvaa fysikaalisen nopeusrajan mukaan, kun kivi on ollut kadoksissa:
  - Y: ±min(300 cm/s × aika, 100 cm), X: ±10 % siitä.
  - Aika on (hutien määrä + 1) / fps.
- **Liike-ennuste:** vakionopeus viimeisistä havainnoista (enintään 12 ruutua taaksepäin, vähintään 4 ruudun väli),
  rajattuna hakualueeseen.
- **Yhdistelmähaku (kolme hakua rinnan):**
  1. **Ristikkohaku ennustetusta paikasta:** karkea vaihe 7 cm:n askelin laajenee keskeltä rengas kerrallaan ja
     pysähtyy, kun pistemäärä ≥ 0,95; sitten mäennousu 8-naapurustossa. Hieno vaihe on mäennousu 1,5 cm:n ruudukossa.
  2. **Mean-shift viimeisestä paikasta.** Hulli ja 1,1-kertainen kehä jaetaan painopisteen kautta kulkevilla viivoilla.
     - Maskipikselit painotetaan (sisällä `A.SEURANTA_MS_SISAPAINO` 2, kehällä 1).
     - Puolikkaiden painotettu ero s = Σ w·g(x − cx) / Σ w kertoo siirtosuunnan (g = pehmeä etumerkki, leveys
       `A.SEURANTA_MS_TAU` 0,5).
     - s muunnetaan pikseleiksi: d = kerroin × r × f⁻¹(s), missä f on yksikköympyrän vastefunktio samoilla painoilla
       ja r hullin puolileveys tai -korkeus.
     - Toistetaan (enintään 6 kertaa), kunnes askel < 0,3 px.
     - Lopuksi mäennousu peitto-osuudella 4 cm:n askeleesta puolittaen alle 0,75 cm:iin.
  3. **Mean-shift ennustetusta paikasta.**
- **Valinta:**
  - Ehdokkaista, joiden pistemäärä ≥ `A.SEURANTA_PISTEKYNNYS` (0,35), valitaan suurin:
    pistemäärä − 0,15 × (taaksepäin yli 2 cm)/10 cm − 0,02 × (etäisyys ennusteesta)/10 cm.
  - Jos mikään ei ylitä kynnystä, valitaan suurin pistemäärä.
  - Valittu paikka tarkennetaan LM:llä (kuten HAKUssa). Siirtymä taaksepäin yli 100 cm hylätään.
- **Siluettitarkennus** tehdään samassa säikeessä.
- **Python-tarkistukset (`_tarkista_havainto`):**
  - **Siluettituki:** siluetin sisällä on oltava ≥ `A.SEURANTA_MIN_SISALLA` (0,4) maskista.
    - Kaukana (Y > 2000 cm), missä kivi on vain 5–6 px leveä, siluetti siirtää paikan aina kun tuki riittää. Ilman tukea
      vain tarkka havainto kelpaa (alkuperäisessä paikassaan).
    - Lähellä tarkan havainnon kynnys on `A.SEURANTA_MIN_SISALLA_LAHELLA` (0, eli siluettitarkennuksen pitää vain
      onnistua), ja ei-tarkka havainto saa siluetin paikan. Jos siluettitarkennus ei onnistu lähellä, havainto
      hylätään.
  - **Hakualue:** lopullinen paikka saa olla enintään 5 cm hakualueen (viimeisestä paikasta tai ennusteesta)
    ulkopuolella.
  - **Ei taaksepäin:** Y ei saa kasvaa yli 100 cm radan pienimmästä Y:stä.
  - **Keskiviiva:** |X| ≤ 120 cm.

---

### 4. Radan elinkaari (`seuranta.py`)

- **Ehdokas → vahvistettu:**
  - Uusi rata on ehdokas, ja sen rivit puskuroidaan. Kun se on liikkunut `A.VAHVISTUS_SIIRTYMA_CM` (25 cm) Y-suunnassa
    ensimmäisestä havainnosta, se vahvistetaan ja puskuroidut rivit kirjoitetaan raaka-CSV:hen.
  - Hylkäys: vähintään 8 riviä ja tarkka-osuus < 0,4 ja rms-mediaani ≥ 12 px (pelaaja). Kaukaiset aidot heitot saavat
    huonon sovituksen (rms 8–10 px), joten pelkkä matala tarkka-osuus ei riitä.
- **Liukuva tarkka-ikkuna:** vahvistettu rata lopetetaan, jos tarkka-osuus viimeisten 650 havainnon ikkunassa putoaa
  alle 0,5 (seuranta ajautunut pelaajaan).
- **Lopetussäännöt:**
  - lähihogin ohitus yli 30 cm (hog-analyysi ei tarvitse enempää), paitsi `--full`-tilassa;
  - taaksepäin liikkuminen: Y kasvaa yli 30 cm sekunnissa (5 ensimmäisen ja 5 viimeisen mediaanit);
  - pysähtyminen: liikkunut alle 20 cm viimeisen sekunnin aikana (`--full`: lisäksi tarkka pysähtyminen, ks. yllä);
  - kadotus: ei havaintoa 1 s:iin (vähintään 5 ruutua).
- **Duplikaatit:** kaksi rataa alle 30 cm toisistaan 3 ruutua peräkkäin yhdistetään. Huonompi poistetaan tässä
  järjestyksessä:
  - pysähtynyt ennen radan suuntaan liikkuvaa (liikkunut ≥ 30 cm 1 s:ssa);
  - sitten vahvistamaton;
  - muuten uudempi.

---

### 5. Heittoportti (`heitot.py`)

Ajon lopussa vahvistetuista radoista valitaan heitot lopulliseen CSV:hen:
- **Radan alun karsinta:** jos alun jälkeen on yli 30 ruudun aukko ja alku on alle 40 riviä, alku pudotetaan (rata
  lukittui ensin väärään kohteeseen).
- **Heitto** on rata, jolle hog-hog-analyysi onnistui (aina), tai jolla on:
  - vähintään 300 riviä;
  - matka eteenpäin ≥ 1500 cm ja loppu-Y ≤ 1100 cm;
  - hidastuva liike: viimeisen 20 %:n nopeus / ensimmäisen 20 %:n nopeus ≤ 0,75 (aito kivi 0,2–0,6, pelaajan pää ~1);
  - sovitus: rms-mediaani ≤ 3 px ja tarkka-osuus ≥ 0,3, tai heikompana luokkana ≤ 12 px ja ≥ 0,1 (lakaisija peittää).
- **Yksi rata per hog-ylitys:**
  - Kaukohogin ylitysruutu interpoloidaan. Jos rata alkaa vasta hogin jälkeen, ylitys ekstrapoloidaan radan alun
    nopeudesta (enintään 60 ruutua).
  - Ylitykset alle 40 ruudun päässä toisistaan ovat sama heitto.
  - Säilyy parempi: hog-ok, sitten hyvä sovitus, pienin rms ja pisin rata.

---

### 6. Hog-hog-analyysi, liuku ja irroitus (`hog_analyysi.py`, `heitot.py`)

Analyysi tehdään kerran, kun vahvistettu kivi on lähihog + 50 cm kohdalla.

**Y(t)-sovitus.** Pisteet väliltä [lähihog + 50 cm, kaukohog − 100 cm]:
- Vaatimukset: vähintään 40 pistettä, ja datan on katettava välin päät ±150 cm.
- Sovitus Y(t) = a t² + b t + c. Sen jälkeen 10 huonoiten sopivaa pistettä pois ja uusi sovitus.
- Vaaditaan R = √R² > 0,99.

Tulokset:
- **Nopeus kaukohogilla:** v = −dY/dt hetkellä, jolloin Y(t) = kaukohog (juuri, jossa Y pienenee).
- **Hog-hog-aika:** t(lähihog) − t(kaukohog).
- **Hidastuvuus kitkamallista:**
  - Malli: kitkakerroin μ(v) = A + B ln v, liikeyhtälö dv/dt = −g μ(v).
  - Sovitus: y0, v0 ja A sovitetaan LM:llä; Y(t) integroidaan numeerisesti (RK4, 0,04 s). B on kiinteä
    `A.HOG_MU_B` (−0,001).
  - Raportoitava hidastuvuus on g μ(1,5 m/s). Jos kitkasovitus on selvästi huonompi kuin toisen asteen sovitus,
    käytetään keskimääräistä hidastuvuutta.

**X-suunta, kurvimalli.**
- Sivukiihtyvyys on vakio k kohtisuoraan kulkusuuntaa vastaan, joten kulkusuunnan kulman muutos θ' = k / |v|.
- X(t) = x0 + θ0 S(t) + k H(t), missä S = kuljettu matka, G = ∫dt / v ja H = ∫v G dt. Nopeus v(t) saadaan Y-sovituksesta.
- Lineaarinen pienimmän neliösumman sovitus; 10 huonointa pistettä pois.
- Tulos: X ja kulkusuunta kaukohogilla.
- Hyväksyntä: X-sovituksen R_x (toisen asteen X(Y)-sovituksesta) > 0,99 ja R_y·R_x > 0,99. Muuten mitään ei raportoida.

**Irroitus:** X, jonka kaukohogin suunta jatkettuna suorana saisi lähemmällä T-viivalla.

**Liuku:** suora X(Y) heiton alusta kohtaan kaukohog + 100 cm (vähintään 8 pistettä), jatkettuna lähemmälle T-viivalle.
- Hakki lisätään lähtöpisteeksi erikseen kohtiin X = +15 cm ja X = −15 cm. Hakki on 6,40 + 1,83 + 1,83 m kaukohogin
  takana.
- Tuloksena kaksi liukusuoraa, vasenkätiselle ja oikeakätiselle.

---

### 7. Kahva ja kierteet (`kierre.py`)

**Kahvan väri.** Joka seurantaruudussa kiven yläpinnan keskipiste (X, Y, H_total) projisoidaan kuvaan.
- Sen ympärille piirretään ympyrä, jonka säde on 2 × kahvan säde kuvassa.
- Ympyrän värikkäistä pikseleistä (S ≥ 80, V ≥ 60; jää ja graniitti ovat harmaita) kerätään sävyhistogrammi.
- Radan lopussa kahvan sävy on se, jonka ±10 välillä on eniten pikseleitä. Nimet: punainen, oranssi, keltainen,
  vihreä, sininen, violetti, pinkki.
- Kirjoitetaan `<pohja>_kahva.csv`:hen ruuduittain.

**Kierteet.**
- **Pala:** kun kivi on alle 23 m päässä, joka ruudusta otetaan 48×48-harmaasävypala kiven akselin suuntaan. Palan
  "ylös" on kiven akseli kuvassa (maan keskipisteestä (X, Y, 0) yläpinnan keskipisteeseen (X, Y, H_total)). Näin kahvan
  profiili on palassa aina samassa asennossa kameran kallistuksesta ja kuvan kierrosta riippumatta. Pala kattaa ±3 kahvan
  sädettä yläpinnan keskipisteen ympäriltä.
- **Kohdistus kiven runkoon:** sijainnin pienet virheet (esim. lakaisijan peittäessä kiveä) siirtäisivät kahvaa palassa.
  Siksi kunkin ruudun pala kohdistetaan graniittirungon gradienttiin. Runko näyttää samalta pyörimisestä riippumatta ja
  on harvoin peitossa. Mallina on rungon mediaani, ja siirto (±4 px) haetaan normalisoidulla korrelaatiolla (kaksi
  kierrosta). Kohdistetusta palasta leikataan ±2 kahvan säteen pala.
- **Piirre:** harmaasävyn gradientin suuruus (Sobel) kahvan ikkunasta (rivit 4–18 ja sarakkeet 7–24 32×32-palassa:
  kahva graniitin yläpuolella, ei taustaa sivuilta), keskitettynä ja normalisoituna. Lisäksi sama piirre palan
  peilikuvasta (peilaus kiven akselin suhteen).
- **Peittyneet ruudut pois:** jokaiselle ruudulle lasketaan piirteen mediaanisamankaltaisuus naapuriruutuihin
  (t ± 1, t ± 2). Ruutu hylätään, jos se on selvästi muita huonompi (< mediaani − 4 × 1,4826 × MAD).
- **Ruutuparit:** kaikkien ruutuparien samankaltaisuus (pistetulo, kaikki viiveet ≥ 0,08 s; enintään 12 000 satunnaista
  paria) sekä peilisamankaltaisuus (ruudun i piirre · ruudun j peilikuvan piirre). Ruudun oma taso (peitto, etäisyys,
  valaistus) ja hidas viiveriippuvuus (3. asteen polynomi) poistetaan; peilisamankaltaisuudesta myös hidas ajallinen
  muutos (viiveen ja ajan polynomi).
- **Pyörimisnopeus:** ω(t) = ω_loppu + `A.KIERRE_HIDASTUVUUS` (0,02 rad/s²) × (t_loppu − t). ω_loppu haetaan
  500 ehdokkaasta, loppukierrosaika 0,6–40 s (`A.KIERRE_P_MIN_S`, `A.KIERRE_P_MAX_S`). Hakuväli on tarkoituksella laaja,
  koska kierteitä voi olla hog-hog-välillä alle 1 tai yli 10.
- **Malli 1 (toisto):** samankaltaisuus = c1 cos(2Δφ) + c2 cos(4Δφ), Δφ = φ_i − φ_j (kahvan muoto toistuu
  puolikierroksen välein). Ehdokas kelpaa vain, jos c1 > 0 ja c2 ≤ c1. Muuten kaksinkertainen kierrosaika sopisi
  yliaallon kautta (puolitetut kierteet).
- **Malli 2 (peili):** kahva kulmassa φ näyttää kulman −φ peilikuvalta. Peilisamankaltaisuus = a cos(Σφ) + b sin(Σφ) +
  a2 cos(2Σφ) + b2 sin(2Σφ), Σφ = φ_i + φ_j. Tämä on toinen, riippumaton mittaus samasta pyörimisestä. Lakaisun rytmi
  (harjat kahvan takana) voi tuottaa toistoa, mutta ei tällaista kahvan absoluuttiseen kulmaan sidottua rakennetta.
- **Hyväksyntä:**
  - toiston selitysaste R² ≥ `A.KIERRE_MIN_R2` (0,02; sekoitetuilla ruuduilla ≤ 0,002) ja suodatuksen jälkeen
    vähintään 60 ruutua;
  - jos peilin selitysaste ≥ `A.KIERRE_PEILI_MIN_R2` (0,03), peilin kierrosajan on oltava ±15 %
    (`A.KIERRE_PEILI_TOL`) toiston kierrosajasta. Muuten tulos hylätään ristiriitaisena. Lopullinen kierrosaika on
    yhteisen spektrin (toisto + peili) huippu tällä välillä;
  - jos peilisignaali on heikko, toiston erottuvuuden on oltava ≥ `A.KIERRE_MIN_EROTTUVUUS` (2,5): paras R² / paras muu
    R², kun ±30 % ja kerrannaiset (×2, ×½) on rajattu pois.
  - Sekoitetuilla ruuduilla (MAH00014 ja 0001, 3 sekoitusta / kivi, 69 ajoa) yhtään tulosta ei hyväksytty.
- **Kierteet:** ∫ω dt / 2π kaukohogista lähihogiin. Kahva erottuu vasta n. 23 m:stä, joten alkuosa ekstrapoloidaan.
- **Tulos (koko ohjelma):** MAH00014: 17/20 hog-kiveä, toiston R² 0,13–0,33 (paitsi kivi 6: 0,04); seuranta-CSV:t
  identtiset aiempaan. 0001: 3/3; kivi 12 silmämääräisesti 2,45 kierrosta, menetelmä 2,43. Toisto ja peili täsmäävät
  kaikissa kivissä, joissa peilisignaali on riittävä (ero ≤ 0,07 kierrosta).
- **Hylätyt MAH00014:ssa (kivet 3, 71, 85) ja rajatapaus 6:** lakaisijoiden mustat jalat ja harjat ovat kahvan takana
  tai päällä lähes koko seurannan ajan (peitto kahvan ikkunassa 30–40 %). Kivessä 3 toisto ja peili antavat eri
  kierrosajan (11,7 s vs. 5,9 s), kivissä 71 ja 85 kumpikaan signaali ei ole riittävä. Kivessä 85 näkyy lakaisun
  rytmi (≈ 0,7 s), joka olisi pelkällä toistolla voinut tulla tulkituksi pyörimiseksi (7,9 kierrosta). Kivessä 6 peili
  on heikko, joten tulos (5,03) perustuu pelkkään toistoon ja on epävarma. 720p-kuvassa kahva on 3–8 px, joten
  näissä kivissä luotettavaa arviota ei saada; ne jätetään tyhjiksi väärän luvun sijaan. Livenä täysresoluutioinen
  arvio (osio 8) voi onnistua näissäkin.
- **Miksi harmaasävy eikä väri:** tallenteiden väri-informaatio on puolitetulla resoluutiolla (4:2:0) ja pakattu.
  Kahvan väri erottuu jäästä selvästi (Lab-väriero moodikuvaan 10–20, jään kohina ≈ 1), mutta värialue on pehmeä
  läikkä, joka ei seuraa kahvan muotoa. Väripiirteet (väriero moodikuvaan, S-maski eri kynnyksillä, sävymaski) antoivat
  kaikki selvästi heikomman tuloksen kuin harmaasävyn gradientti.
- Live-tilassa sama lasketaan myös täyden resoluution kuvasta (osio 8).
- Live-tilassa kierrearvio lasketaan taustasäikeessä (noin 1 s / piirre), jotta seuranta ei pysähdy lähihogilla.
  Tulos ilmestyy paneeliin ja lokiin (`[kierre] kivi N: kierteita hog-hog …`), kun laskenta on valmis. Tulostiedostot
  kirjoitetaan vasta, kun kaikki arviot ovat valmiita. Tiedostoajossa arvio lasketaan heti, joten tulokset eivät
  muutu ajosta toiseen. (Live 06.10.2026: ennen tätä jokainen heitto pysäytti käsittelyn noin 2,2 s:ksi, ja viive
  kasvoi heittosarjan aikana 6–10 s:iin.)

---

### 8. Paikallinen täysi resoluutio (live, `paikallinen.py`)

Putki (stabilointi, taustanvaimennus, HAKU, näkymät) toimii 1280×720:ssa. SEURANTA saa kuitenkin kuvan, jossa jokaisen
seurattavan kiven hakualue (+ 8 cm reunus) on käsitelty kameran täydellä resoluutiolla:
- Alueen pikselit haetaan puskurin täyden resoluution raakaruudusta; YUY2-muunnos tehdään vain alueelta.
- Käsittely: sama stabilointi (siirto skaalattuna), täyden resoluution linssikorjaus ja taustanvaimennus täyden
  resoluution moodikuvaa vasten. Muu kuva on valkoista eli taustaa.
- Kameramatriisi, siluettiasetukset ja rms-arvot skaalataan vastaavasti. Tulokset ovat senttimetreinä kuten ennen.

Raakaruutuja pidetään `A.PAIK_RENGAS_S` (10 s) rengaspuskurissa (~1 Gt). Jos käsittely on tätä enemmän jäljessä, ruutu
seurataan 720p:na.

- Kahvan väri ja kierrepiirteet (720p ja täysi resoluutio) lasketaan taustasäikeessä (yksi säie, järjestys säilyy).
  Se, kerätäänkö piirteet, päätetään pääsäikeessä. Hog-analyysi odottaa kiven piirteet valmiiksi ennen kierrearviota,
  ja täyden resoluution ruutu pidetään renkaassa, kunnes taustasäie on käsitellyt sen. Tulokset ovat samat kuin
  ennen (MAH00014-tiedostoajo identtinen).

---

### 9. Kuvan esikäsittely (`esikasittely.py`, `cpp/esikasittely.hpp`, `cpp/gpu.hpp`)

**Stabilointi seurannassa.**
- Jokainen ruutu verrataan suoraan moodikuvaan vaihekorrelaatiolla. Ketjutusta ei ole, joten virhe ei kerry.
- Kuva pienennetään tarvittaessa `A.STAB_LEVEYS` (1280) leveyteen. Hanning-ikkuna, moodikuvan FFT välimuistissa,
  huippu 5×5-painopisteellä.
- Siirto rajataan ±`A.STAB_MAX_SIIRTO_PX` (10 px).
- Laskenta-aika on noin 20–27 ms CPU-aikaa ruutua kohden. Se lasketaan liukuhihnan vaiheessa A kahdessa
  työsäikeessä (`A.STAB_SAIKEET`), rinnan pääsäikeen kanssa.
- **Tallennus** (`A.STAB_TALLENNA`): joka ruudun stabilointi kirjoitetaan tiedostoon `<pohja>_stabilointi.csv`.
  Sarakkeet: `frame`, `lahde` (`paneelit` kalibroinnin aikana, `vaihekorrelaatio` tai `vaihekorrelaatio (hihna)`),
  `dx_px`, `dy_px` (ruudun siirto 720p-kuvassa), `kierto_deg`, `skaala` ja `laskenta_ms`.
- **Mukautuva harvennus (vain live):** vaihekorrelaatio lasketaan vain joka `A.STAB_HARVENNUS_LIVE`. (10.)
  ruudusta (avainruutu).
  - Rauhallisena aikana: jos kahden peräkkäisen avainruudun siirrot eroavat alle `A.STAB_HARVENNUS_KYNNYS_PX`
    (0,15 px), väliruutujen siirto interpoloidaan lineaarisesti.
  - Tärinän aikana: jos ero on suurempi, kaikki väliruudut lasketaan rinnakkain. Tämän jälkeen lasketaan joka ruutu,
    kunnes kokonainen 10 ruudun jakso on rauhallinen (ruudusta ruutuun alle kynnyksen). Hystereesi estää
    värähtelyä jäämästä huomaamatta, vaikka kaksi avainruutua sattuisi samaan vaiheeseen.
  - Viive vaiheessa A on enintään 10 ruutua (0,4 s).
  - `_stabilointi.csv`:ssä lähde on `interpoloitu (hihna)` tai `vaihekorrelaatio (hihna)`. Lopuksi tulostetaan
    laskettujen ruutujen osuus.
  - Tiedostoajossa lasketaan aina joka ruutu, joten tulokset ovat toistettavat.
  - Simuloitu, kun live 06.10.2026 14:27 -ajosta ja MAH00014:stä laskettiin 27–29 % ruuduista: virhe rms
    0,04–0,05 px, p99 0,15–0,16 px, suurin 0,7 px. 0001-live-simulaatiossa laskettiin 10 % ruuduista: ero joka
    ruudun laskentaan rms 0,014 px, suurin 0,14 px, ja stabiloinnin CPU-aika putosi 9,4 → 1,0 ms/ruutu.
    MAH00014 live-simulaationa: laskettiin 29 % ruuduista; interpoloitujen ero joka ruudun laskentaan rms 0,05 px,
    p99 0,20 px, suurin 0,84 px.
  - Tärinä on todellista kameran liikettä: MAH00014:ssä seinän ja katon staattiset laikut värisevät samalla tavalla
    kuin koko kuvan mittaus (dy ruudusta ruutuun rms 0,50 vs. 0,52 px).

**Linssikorjaus:** `warpAffine` (stabilointi) ja sen jälkeen `remap` (k1-oikaisu).

**Valotasapaino.** Kanavakohtainen gain ja bias moodikuvaan robustilla pienimmällä neliösummalla:
- 3 kierrosta; jokaisen jälkeen pikselit, joiden jäännös on yli 80. persentiilin (etuala, pelaajat), pudotetaan.
- Joka 4. pikseli otetaan mukaan.
- Estimoidaan taustasäikeessä kerran sekunnissa. Otetaan käyttöön tasan `A.VALO_KAYTTOON_VIIVE_RUUTUA` (10) ruudun
  päästä, joten ajo on toistettava.

**Varjosietoinen taustanvaimennus (C++, valotasapaino samalla läpikäynnillä).** Pikseli on taustaa (valkaistaan), jos
yksikin seuraavista pätee:
- harmaasävyero moodikuvaan < `A.GRANIITTI_EROKYNNYS` (10);
- kirkkaus V on pudonnut moodikuvasta välillä (−3, 50) eli varjo;
- pikseli on jäätä: S < 22 ja V > 128.

Lisäksi väriportin on hyväksyttävä: joko moodikuvan pikseli on matalakylläinen (S < 60), tai sävyero on ≤ 10 ja ruudun
pikseli on kylläinen. Näin värikäs mutta eri sävyinen kohde (keltainen kahva sinisellä mainoksella) ei katoa.

**Liukuhihna (seurannassa):**
- **Vaihe A** (`A.STAB_SAIKEET` = 2 säiettä): luku, harmaasävy ja vaihekorrelaatio.
- **Vaihe B:** warp, remap, valotasapaino ja taustanvaimennus. GPU-polku (OpenCL) on bitti-identtinen CPU:n kanssa.
  Vaihe B käynnistää myös HAKUn heti, kun ruutu on valmis.
- **Vaihe C** (pääsäie): SEURANTA, radat ja CSV.
- Jonojen syvyys on `A.JONON_SYVYYS` (50). Ruutujen järjestys ja sisältö ovat samat kuin ilman hihnaa.

---

### 10. Suorituskyky

- `A.STAB_SAIKEET`, `A.JONON_SYVYYS`: liukuhihnan rinnakkaisuus.
- `A.GPU_VAIHE_B`: vaihe B OpenCL:lla, jos laite löytyy (virheessä paluu CPU:lle).
- `A.VIDEO_LAITTEISTODEKOODAUS`: videotiedoston dekoodaus laitteistolla.
- C++ ajaa kivet omissa säikeissään, ja kiven sisällä ristikkohaku ja kaksi mean-shiftiä rinnan. OpenCV:n oma
  rinnakkaistus kytketään pois näiden ajaksi, koska se kilpailisi samoista ytimistä.
- Ajon lopussa raportoidaan pullonkaula-analyysi (vaiheiden A/B/C palveluaika ja jonojen täyttö) ja vaihekohtaiset ajat
  (Python ja C++). `A.RAPORTTI_VALI_RUUTUA` (1000) on edistymisrivin väli; `A.MAX_RUUTU` pysäyttää testiajon.

---

### 11. Live-syöte (`live.py`)

- **Kamerasäie:** lukee kameraa jatkuvasti (Windows: DirectShow tai MSMF), muuntaa kuvan BGR:ksi ja pienentää sen
  käsittelykokoon (oletus 1280×720). Ruudut kirjoitetaan RAM-puskuriin juoksevalla indeksillä. Raaka YUY2 voidaan
  muuntaa ajurilla, `cv2.cvtColor`:lla tai OpenCL:llä.
- **Lukijat:** pääsilmukka lukee puskuria järjestyksessä. Hyppivät lukijat (profiilin opettelu, alfa-havainnot) saavat
  `cv2.VideoCapture`-yhteensopivan olion, joka lukee puskurista taaksepäin tai odottaa uutta ruutua.
- **Puskurin karsinta:** vanhat ruudut poistetaan, kun ne ovat yli historian verran peräkkäislukijan takana. Profiilin
  havaintoruudut kiinnitetään, jotta niitä ei poisteta.
- **Seurannan alku:** seurannan alussa hypätään uusimpaan ruutuun (kalibroinnin aikana kertynyt viive pois). Historia
  pienenee arvoon `A.LIVE_TAAKSE_SEURANNASSA_S` (2 s).
- **Aikaleimat:** `<pohja>_live_aikaleimat.csv` kertoo käsitellyn ruudun, kameran ruudun ja seinäkelloajan.

---

### 12. Puhelinnäkymä ja debug-video (`nakyma.py`, `katselu.py`)

**Asettelu.** Kuva käännetään 90° vastapäivään (kivet kulkevat ylhäältä alas) ja skaalataan 1080 px korkeaksi.
- Vasemmalla paneeli vasemmalle lähteneistä heitoista ja oikealla oikealle lähteneistä (suunta kaukohogilla).
- Uusin heitto on ylimpänä, ja vanhemmat rullaavat alas.
- Laatikossa: kiven ID, ikä (s kaukohogin ylityksestä), nopeus, hidastuvuus, hog-hog, liuku (vasen hakki punaisella,
  oikea vihreällä), irroitus ja kierteet (kahvan värillä).

**Puhelinnäkymä (`--katselu`).**
- **Palvelin:** näkymä piirretään omassa säikeessään kerran `A.KATSELU_VALI_S`:ssa ja lähetetään JPEG:nä (korkeus 810,
  laatu 75). Pieni HTTP-palvelin jakaa sivun.
- **Korostus:** puhelin lähettää Alku- ja Loppu-arvot. Laatikot, joiden ikä on Alku…Loppu s, korostetaan, ja niiden
  liukusuorat (vasen- ja oikeakätinen valinnan mukaan) sekä irroitusristi piirretään kuvaan.
- **Liikemaski:** viivat ja risti piirretään liikkuvien kohteiden alle. Liikkuva kohde = ero tyhjän radan
  taustakuvaan > `A.KATSELU_LIIKE_KYNNYS` (40) suurimmassa kanavassa; 3×3-avaus ja 5×5-laajennus.
- **Viive:** keskikuva näytetään Alku s myöhässä (JPEG-rengaspuskuri kaappausajan mukaan, enintään 180 s). Kun heitto
  ylittää kaukohogin videolla, laatikko samalla korostuu. Jos käsittely on Alkua enemmän jäljessä, näytetään uusin kuva
  ja viive punaisella.
- **Tallennus:** jokainen lähetetty kuva tallennetaan `<nimi>_katselu.avi`:hin.

**Debug-video (`--debug`):** sama asettelu jokaisesta ruudusta. Kivien ääriviivat ovat vihreitä (tarkka) tai oransseja.
Koodaus on `A.DEBUG_KOODAUS`; Intel QSV, jos saatavilla.

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
| `MAH00014.mp4 --no-debug` (uudet C++-moduulit) | sijainti-, raaka-, hog- ja kahva-CSV identtiset (76 rataa, 20 hog-analyysiä) |
| live-simulaatio 1080p (`--live-sim`, täysi resoluutio) | samat heitot ja hog-analyysit; pienet erot kuten Testi_08_02:lla itsellään (täyden resoluution rengas riippuu ajoituksesta, kaksi 08_02-ajoa eroavat yhtä paljon) |
| live-simulaatio + `--katselu` | sijainti-, raaka-, hog- ja kahva-CSV identtiset |

GPU-vaihe B testattiin OpenCL-CPU-toteutuksella (pocl, `GPU_LAITE=cpu`): uusi ydin antaa bitilleen saman tuloksen kuin
Testi_08_02:n ydin ja CPU-polku.
