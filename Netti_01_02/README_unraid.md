# Kivenseuranta – palvelinversio (Netti_01_02): Unraid / Docker

Tämä kansio on `Netti_01_01`:n **palvelimella ajettava versio**: sama analyysi (kiven tunnistus, liuku, irroitus, nopeus, kierre, joukkueiden kello), mutta ilman ikkunaa. Ohjelma pakataan
Docker-kuvaksi, jota Unraid (tai mikä tahansa Linux-palvelin, jossa on Docker) ajaa kontissa. Tulokset (CSV ja kuvallinen mp4) kirjoitetaan jakokansioon, ja kuvaa, lokia ja tiedostoja
voi katsoa selaimella osoitteessa `http://PALVELIN:63884`.

Analyysin algoritmit ovat täsmälleen samat kuin `Netti_01_01`:ssä; heittojen CSV on koeajossa tavu tavulta sama kuin natiivissa C++-ajossa (ks. [Testaus](#testaus)). Algoritmien dokumentaatio on
tiedostossa [`README_netti.md`](README_netti.md), tämä tiedosto kertoo vain palvelinajosta.

**Mitä on testattu ja mitä ei.** Kuva rakennettiin ja ajettiin oikeassa Docker-ympäristössä (Ubuntu 24.04 -pohja, Python 3.12, Docker 29): rakennus, kaikki testit kontissa (vanha testisarja, 494 tarkistusta, ja
palvelinversion 151 tarkistusta), videoleikkeen analyysi (tiedosto, `--nopea` ja reaaliaika), selainnäkymä, kansion valvonta, PUID/PGID, `docker stop` kesken ajon (tallenne ja CSV valmistuvat)
sekä virhetilanteet. **Ei testattu:** ajoa itse Unraid-koneella, YouTube-yhteyttä (testiympäristön verkko esti youtube.com:n; yt-dlp, deno ja ffmpeg on varmistettu kuvassa) eikä
GPU-koodausta. Ne ovat ensimmäisen oikean käyttökerran riskikohdat; ohjeet niiden selvittämiseen ovat alla.

---

## 1. Pikaohje (Unraid)

Nimet ovat vapaita; tässä oppaassa kaikki on nimetty `kivenseuranta`: jakokansio (share) `kivenseuranta`, Docker-kuva `kivenseuranta:latest` ja kontti `kivenseuranta`. Erillistä rakennuskansiota ei tarvita.

1. **Luo jakokansio.** Unraidin web-käyttöliittymä → *Shares* → *Add Share* → nimeksi `kivenseuranta`. Levyksi käy array tai cache. Varmista, että SMB-jako (*Export*) on päällä ja että oma käyttäjäsi
   saa kirjoittaa siihen. Isot mp4-tiedostot kannattaa pitää levyryhmän jaossa.
2. **Kopioi ohjelma omalta koneeltasi.** Avaa Windowsin Resurssienhallinnassa `\\PALVELIN\kivenseuranta` (PALVELIN = Unraidin nimi, oletus `Tower`, tai IP-osoite) ja kopioi sinne koko kansio `Netti_01_02`
   sellaisenaan. Tarkista, että tiedosto `\\PALVELIN\kivenseuranta\Netti_01_02\Dockerfile` löytyy suoraan (ei sisäkkäistä `Netti_01_02`-kansiota). Rakennuksen jälkeen kansion voi poistaa; sitä tarvitaan
   vain päivityksissä.
3. **Rakenna kuva** Unraidin terminaalissa (oikean yläkulman `>_`); ensimmäisellä kerralla muutama minuutti, lataa noin 1 Gt:
   ```bash
   cd /mnt/user/kivenseuranta/Netti_01_02
   docker build -t kivenseuranta:latest .
   ```
   Lopuksi `docker images kivenseuranta` näyttää kuvan.
4. **Kopioi mallipohja Unraidiin** (samassa terminaalissa):
   ```bash
   mkdir -p /boot/config/plugins/dockerMan/templates-user
   cp /mnt/user/kivenseuranta/Netti_01_02/unraid/kivenseuranta.xml /boot/config/plugins/dockerMan/templates-user/my-kivenseuranta.xml
   ```
5. **Lisää kontti:** *Docker*-välilehti → *Add Container* → *Template*-valikosta `kivenseuranta` (*User templates*; päivitä sivu, jos sitä ei näy). Jätä **Lähde (NETTI_LAHDE)** tyhjäksi, niin kontti valvoo kansiota
   `video` ja linkkilistaa `youtube.txt`. Paina *Apply*. Jos Unraid yrittää ensin ladata kuvaa verkosta ja antaa virheen `pull access denied`, se on odotettua, koska kuva on paikallinen; kontin pitäisi silti
   syntyä. (Vaihtoehto: [`docker-compose.yml`](docker-compose.yml) Compose Managerilla tai `docker compose up -d`.)
6. **Kokeile.** Ensimmäisen käynnistyksen jälkeen share-kansioon ilmestyvät kansiot `video`, `ulos` ja `kalibroinnit`. Kopioi videotiedosto kansioon `video`; analyysi alkaa noin 1–2 minuutin kuluttua, kun
   tiedosto on pysynyt muuttumattomana 30 s. **YouTube-linkit** kirjoitetaan tekstitiedostoon `youtube.txt` (jakokansion juuressa), yksi per rivi: kontin ei tarvitse käynnistyä uudelleen
   ([Käyttö c](#3-käyttö-tavallisimmat-tilanteet)). Selainnäkymä (`http://PALVELIN:63884`; *Docker*-välilehden kontin kuvake → *WebUI*) on auki **vain ajon aikana**; tyhjän kansion odotuksen aikana portti ei vastaa.
   Portti on valittu harvinaiseksi, ettei se törmää tavallisiin palveluihin ([Asetukset](#2-asetukset-ympäristömuuttujat)). Lokin näet kontin kuvakkeesta (*Logs*); tulokset tulevat kansioon `ulos`.
7. **Käynnistys koneen mukana:** *Docker*-välilehdellä kontin kohdalta *Autostart* päälle.

**Jos mallipohja ei toimi**, samat asetukset yhdellä komennolla:

```bash
docker run -d --name kivenseuranta --restart unless-stopped --stop-timeout 90 \
  -p 63884:63884 -v /mnt/user/kivenseuranta:/data \
  -e PUID=99 -e PGID=100 -e TZ=Europe/Helsinki \
  kivenseuranta:latest
```

Lähteeksi voi antaa myös kiinteän YouTube-tunnuksen tai -osoitteen asetuksella `NETTI_LAHDE` ([YouTube](#5-youtube)), mutta helpompaa on linkkilista `youtube.txt`.

---

## 2. Asetukset (ympäristömuuttujat)

Kaikki ovat valinnaisia. Tyhjä arvo tarkoittaa samaa kuin asettamatta jättäminen (Unraidin mallipohja välittää myös tyhjät). Luvuissa desimaalipilkku kelpaa.

| Muuttuja | Oletus | Merkitys |
|---|---|---|
| `NETTI_LAHDE` | kansio `/data/video` ja linkkilista `/data/youtube.txt` (valvonta) | Mitä analysoidaan: **YouTube-tunnus tai -osoite**, suora virran osoite (`http`/`https`/`rtsp`/`m3u8`), **videotiedosto** (`/data/video/peli.mp4` tai pelkkä `peli.mp4`), **kansio** tai **kuvio** (`/data/video/*.mp4`). Kiinteä YouTube-lähde vaatii kontin asetusten muuttamisen aina, kun osoite vaihtuu; vaihtuville osoitteille käytä linkkilistaa (jätä tämä tyhjäksi) |
| `NETTI_LINKIT` | `/data/youtube.txt` | Linkkilista: tekstitiedosto, jossa on yksi YouTube-tunnus tai verkko-osoite per rivi (`#` aloittaa kommentin). Luetaan joka valvontakierroksella, joten uuden linkin lisäys ei vaadi uudelleenkäynnistystä. Toimii vain kansion valvonnassa. `ei` = pois käytöstä |
| `NETTI_JATKUVA` | kansiolle 1, muulle 0 | Valvonta: kun ajo päättyy (video valmis, live-lähetys loppui, YouTube-lähetystä ei vielä ole), odotetaan `NETTI_ODOTUS_S` s ja yritetään uudelleen. Kansiossa uudet videot ja linkkilistan uudet linkit analysoidaan, valmiit eivät toistu |
| `NETTI_VIDEO` | 1 | Tallenna kuvallinen mp4 (`ulos/kivenseuranta_*.mp4`). 0 = vain CSV |
| `NETTI_CSV` | 1 | Tallenna heittojen tiedot CSV:hen |
| `NETTI_VIIVE` | 10 | Kuva tulee tämän verran (s) analyysin perässä (vaikuttaa näkymään ja tallenteen piirtoon) |
| `NETTI_KANSIO` | – | Kalibrointikansio (nimi tai polku), jota käytetään sellaisenaan; ohittaa `NETTI_KALIBROINTI`:n ([Kalibrointi](#4-kalibrointi)) |
| `NETTI_KALIBROINTI` | uusi | Ilman `NETTI_KANSIO`:ta: `uusi` = jokainen video (tiedosto tai YouTube-video, ei live) kalibroidaan aina uudelleen omista ruuduistaan (tiedosto noin 1 min, YouTube noin 0,5–2 min ennen analyysia). `valitse` = käytetään ensimmäistä videoon sopivaa vanhaa kalibrointia ja uusi tehdään vain, jos mikään ei sovi |
| `NETTI_NOPEA` | 1 | Tiedostolle: ajetaan niin nopeasti kuin ehtii (tulos ei riipu ajoituksesta). 0 = reaaliaikaa (esim. kun halutaan katsoa selaimessa kuin lähetystä) |
| `NETTI_ALKU`, `NETTI_LOPPU` | – | Tiedostolle: aloitus ja lopetus videossa (s) |
| `NETTI_AANI` | auto | Ääni tallenteeseen: `auto` (YouTube: oma ääniraita, tiedosto: tiedoston ääni), `ei`, tai polku / osoite. Live-lähetyksen ääntä ei liitetä |
| `NETTI_KOODEKKI` | cpu | `cpu` = H.264 prosessorilla (libx264), `auto` / `gpu` = kokeilee laitteistokooderia ([GPU](#7-resurssit-ja-suorituskyky)) |
| `NETTI_X264_PRESET` | superfast | `ultrafast` = kevein prosessorille ja suurin tiedosto; `veryfast` = pienempi tiedosto |
| `NETTI_SAIKEET` | automaattinen | Tarkan sovituksen työprosessit (oletus: käytettävät ytimet − 1, enintään 3) |
| `NETTI_LATAUS` | automaattinen | YouTube / virta: esilatauspuskurin pituus (s). Oletus 30; alle 10 Gt:n muistilla 15, alle 6 Gt:lla 8 (puskuri vie ~3,1 Mt / ruutu) |
| `NETTI_KIRI` | 10 | YouTube-videolle: katso `python netti_live.py --help` (`--kiri`) |
| `NETTI_YT_FORMAATTI` | – | yt-dlp-formaattivalitsin, esim. `bestvideo[height<=1080][protocol=https][vcodec^=avc1]` |
| `NETTI_EI_KIERRETTA`, `NETTI_EI_KELLOA`, `NETTI_HILJAA` | 0 | Ei kierteen mittausta / ei joukkueiden kelloa kuvassa / vähemmän tulostetta |
| `NETTI_WEB_PORTTI` | 63884 | Selainnäkymän portti **kontin sisällä**; 0 = ei selainnäkymää. Jos portti on palvelimella varattu, vaihda portin yhdistyksessä vain palvelimen puoli (Unraid: *Host Port*, compose: `ports`-rivin vasen luku) ja jätä tämä ennalleen. Tämän muuttaminen on tarpeen vain, jos kontti ajetaan `host`-verkossa (silloin kontin portti on palvelimen portti) |
| `NETTI_WEB_LEVEYS`, `NETTI_WEB_FPS`, `NETTI_WEB_LAATU` | 1920, 8, 75 | Selaimeen menevän kuvan leveys (px), enimmäistaajuus katsojaa kohden ja JPEG-laatu. Oletus 1920 = analysoitu kuva sellaisenaan (sama resoluutio kuin lähdevideosta luettu, ei pienennystä); pienempi leveys (esim. 1280) keventää verkkoa ja prosessoria. Kuva pakataan vain, kun joku katsoo |
| `NETTI_WEB_KAYTTAJA`, `NETTI_WEB_SALASANA` | – | Salasana selainnäkymään (HTTP Basic; kumpikin pakollinen) |
| `NETTI_ODOTUS_S` | 60 | Valvonta: odotus ajojen välillä. Toistuvissa nopeissa epäonnistumisissa odotus kaksinkertaistuu (enintään 15 min) |
| `NETTI_VAKAUS_S` | 30 | Valvonta: tiedoston on oltava muuttumatta näin kauan ennen analyysia (kopiointi voi olla kesken) |
| `NETTI_SULKU_S` | 60 | `docker stop`: näin kauan ajo saa viimeistellä tallenteen, ennen kuin se tapetaan |
| `NETTI_LISAVALITSIMET` | – | Mikä tahansa `netti_live.py`:n valitsin, esim. `--kiri 0 --lataus 15` |
| `YTDLP_PAIVITA` | 0 | 1 = päivitä yt-dlp kontin käynnistyessä (YouTube muuttuu usein; tarvitsee verkon) |
| `YTDLP_KEKSIT` | – | Polku `cookies.txt`-tiedostoon, jos YouTube vaatii kirjautumisen ([YouTube](#5-youtube)) |
| `PUID`, `PGID`, `UMASK` | 99, 100, 022 | Kenen oikeuksilla ajetaan ja luotavien tiedostojen oikeudet (Unraid: `nobody:users`) |
| `TZ` | Europe/Helsinki | Aikavyöhyke (lokin ja tiedostonimien aikaleimat) |

Kansiot (kontissa): `/data/video` (syötteet), `/data/ulos` (tulokset), `/data/kalibroinnit` (omat kalibroinnit). Ne voi siirtää muuttujilla `NETTI_VIDEO_KANSIO`, `NETTI_ULOS_KANSIO` ja
`NETTI_KALIBROINNIT`.

Ylimääräiset valitsimet voi antaa myös kontin komentona: `docker run … kivenseuranta:latest --alku 600 --loppu 900`. Kaikki valitsimet: `docker run --rm kivenseuranta:latest python /app/netti_live.py --help`.
Asetukset ja lopullinen komento näkee ajamatta: `docker run --rm -e NETTI_LAHDE=… kivenseuranta:latest --nayta`.

---

## 3. Käyttö: tavallisimmat tilanteet

**a) Videotiedoston analyysi (kertaajo).** Laita tiedosto kansioon `video`.

```bash
docker run --rm -v /mnt/user/kivenseuranta:/data -p 63884:63884 \
  -e NETTI_LAHDE=peli.mp4 -e NETTI_JATKUVA=0 \
  kivenseuranta:latest
```

Tulokset (videotiedostolle; YouTube-linkille nimessä on videon nimi, ks. [Tulokset](#6-tulokset-ja-selainnäkymä)): `ulos/kivenseuranta_peli_<aika>.mp4`, `ulos/heitot_peli_<aika>.csv` ja `ulos/heitot_peli_<aika>_kivet.csv`. Compose / Unraid: aseta `NETTI_LAHDE`, `NETTI_JATKUVA=0` ja kontin
*Restart* = `no`, muuten Docker käynnistää kontin uudelleen ajon päätyttyä.

**b) Kansion valvonta (oletus).** Ei `NETTI_LAHDE`-asetusta, Restart = `unless-stopped`. Pudota videoita kansioon `video`; jokainen analysoidaan kerran. Valmiit kirjataan tiedostoon
`ulos/kasitellyt.json` (tiedosto, koko ja muokkausaika), joten uudelleenkäynnistys ei aja niitä uudestaan; jos haluat ajaa videon uudelleen, poista sen rivi (tai koko tiedosto). Analyysi, joka
epäonnistuu kolme kertaa, jää pois (rikkinäinen tiedosto ei silmukoi ikuisesti).

**c) YouTube-linkit ilman uudelleenkäynnistystä (linkkilista).** Kun kontti valvoo kansiota (kohta b, oletus), se lukee myös tekstitiedostoa `youtube.txt` jakokansion juuresta
(`\\PALVELIN\kivenseuranta\youtube.txt`; Windowsin Muistio kelpaa). Kirjoita tai liimaa tiedostoon YouTube-osoite tai -tunnus, yksi per rivi, ja tallenna: kontti huomaa uuden rivin noin minuutissa, eikä sitä
tarvitse käynnistää uudelleen eikä sen asetuksia muuttaa. Esimerkki:

```text
# Matarmaan peli
https://www.youtube.com/watch?v=hnTTLqHr5x8
https://youtu.be/abcdefghijk   # toinen peli (kommentti rivin lopussa)
```

* Videot ja linkit käsitellään **yksi kerrallaan**: ensin kansion `video` tiedostot, sitten linkit tiedoston järjestyksessä. Jos jokin ajo on käynnissä (pitkä live-lähetys voi kestää tunteja), uusi rivi odottaa
  vuoroaan ja huomataan ajon päätyttyä.
* Jokainen linkki ajetaan **kerran**; sama video eri osoitemuodoissa (tunnus, `watch?v=`, `youtu.be`) lasketaan samaksi. Valmiit kirjataan tiedostoon `ulos/kasitellyt.json` (rivi `lista:yt_<tunnus>`), joten
  uudelleenkäynnistys ei aja niitä uudestaan. Vanhat rivit voi jättää tiedostoon muistilistaksi. Ajaaksesi videon uudelleen poista sen rivi tiedostosta `kasitellyt.json` (tekstieditorilla; muut rivit jäävät).
* Linkki voi olla valmis video, käynnissä oleva live-lähetys (analysoidaan reaaliajassa ja ajo päättyy lähetyksen mukana) tai tuleva lähetys. Jos lähetys ei ole vielä alkanut tai haku epäonnistuu, kontti yrittää
  uudelleen 1, 2, 4 ja 8 minuutin ja sen jälkeen 15 minuutin välein; kymmenen epäonnistumisen (noin 1,5 tuntia) jälkeen rivi jää pois, ja lokiin tulee siitä merkintä.
* Rivi on joko verkko-osoite (muu teksti sen perässä ohitetaan; esim. `rtsp://…` ja `https://….m3u8` kelpaavat myös) tai YouTube-tunnus **yksin rivillä**. Muut rivit ohitetaan, ja lokiin tulee niistä yksi huomautus.
* Kanavan live-osoitteen (`…/@KANAVA/live`) voi lisätä, mutta sekin ajetaan vain kerran. Jos haluat, että kontti odottaa kanavan jokaista uutta lähetystä, käytä kiinteää lähdettä (kohta d).
* Polun voi vaihtaa muuttujalla `NETTI_LINKIT` (`ei` = linkkilista pois). Linkkilista toimii vain kansion valvonnassa, ei silloin kun `NETTI_LAHDE` on yksittäinen YouTube-osoite tai tiedosto tai ajo on kertaajo.

**d) Kiinteä YouTube-video tai live-lähetys (`NETTI_LAHDE`).** Osoite annetaan kontin asetuksella, joten sen vaihtaminen vaatii asetuksen muuttamisen (Unraid: Edit → Apply; kuvaa ei tarvitse rakentaa uudelleen).

```bash
docker run --rm -v /mnt/user/kivenseuranta:/data -p 63884:63884 \
  -e NETTI_LAHDE=hnTTLqHr5x8 -e NETTI_JATKUVA=0 \
  kivenseuranta:latest
```

Live-lähetyksen odotus: `NETTI_LAHDE=https://www.youtube.com/@KANAVA/live`, `NETTI_JATKUVA=1`, Restart = `unless-stopped`. Kontti yrittää uudelleen minuutin välein (toistuvasta epäonnistumisesta
alkaen harvemmin), analysoi lähetyksen ja jatkaa odottamaan seuraavaa. Tavallinen YouTube-**video** (ei live) analysoidaan valvonnassa vain kerran.

**e) Katso tallenne kuin lähetystä.** Tiedostolle `NETTI_NOPEA=0`: video etenee reaaliajassa ja selainnäkymä näyttää kuvan viiveellä (`NETTI_VIIVE`).

**f) Apuohjelmat kontissa.** Nopeusmittaus, tarkistukset ja testit:

```bash
docker run --rm -v /mnt/user/kivenseuranta:/data kivenseuranta:latest python /app/tools/mittaa_nopeus.py /data/video/peli.mp4 --alku 300 --loppu 360
docker run --rm kivenseuranta:latest testaa
docker run --rm -it -v /mnt/user/kivenseuranta:/data kivenseuranta:latest bash
```

---

## 4. Kalibrointi

Kalibrointi ja taustakuva kuuluvat yhteen kameran asentoon: väärä kalibrointi antaa vinot paikat ja huonon seurannan. Kuvassa on valmiina kolme kalibrointikansiota (`tulokset`,
`tulokset_nettivideo2`, `tulokset_nettivideo3`: videot 1–3), mutta **oletuksena jokainen video kalibroidaan uudelleen omista ruuduistaan** (`NETTI_KALIBROINTI=uusi`). Syy: sama rata eri viikkoina
kuvattuna ei aina sovi vanhaan kalibrointiin (kamera on voinut siirtyä), eikä sopivuustarkistus kerro, oliko vanha kalibrointi alun perinkään hyvä. Kun `NETTI_KANSIO` on tyhjä:

* **tiedosto (`/data/video`):** video kalibroidaan (noin minuutti) kansioon `/data/kalibroinnit/tulokset_auto_<videon nimi>` ennen analyysia (osa-numero `_osa_NN` ei lasketa nimeen). Lokissa lukee
  `Kalibroidaan video uudelleen … kansioon …`, kalibroinnin laatuluvut (`tarkennus … RMS … cm`, kokeessa noin 0,6 cm) ja lopuksi `tarkistus: … sopii`; menetelmä on kuvattu tiedostossa
  [`README_netti.md`](README_netti.md).
* **YouTube-video (ei live):** sama kalibrointi tehdään suoraan virrasta (ruudut haetaan hakemalla virrasta): noin 0,5–2 min ennen analyysin alkua (selainnäkymä avautuu vasta sen jälkeen), kansio
  `/data/kalibroinnit/tulokset_auto_yt_<tunnus>`. Jos virta jumittaa, kunkin vaiheen aikaraja on 900 s. Videota luetaan mahdollisimman vähän: molempien puoliskojen ruudut haetaan kerralla (ei erikseen vasenta ja oikeaa), haku tehdään neljällä lukijalla rinnakkain ja tarkistuksen vertailukuvat luetaan taustalla jo kalibroinnin aikana (ennen 72 hakua peräkkäin, nyt 48 rinnakkain; tulokset samat kuin ennen), ks. [`README_netti.md`](README_netti.md).
* **Jos uusi kalibrointi epäonnistuu** (renkaita tai viivoja ei löydy mediaanikuvasta, virta jumittaa): käytetään videoon sopivaa vanhaa kansiota, ellei sellaista ole, lähintä vanhaa kansiota (poikkeama
  lokissa: `ikkunasiirtyma p90 px vasen/oikea`, raja 1,5), ja lokiin tulee `VAROITUS: kalibrointi ei sovi tahan videoon, joten heittoja ei todennakoisesti tunnisteta`. Silloin lataa video ja pudota se kansioon
  `video` (tiedosto kalibroidaan itse, tarvittaessa koko videon mediaanikuvalla) tai kalibroi käsin (`python /app/tools/uusi_kalibrointi.py VIDEO --ulos /data/kalibroinnit/oma` ja `NETTI_KANSIO=oma`).
* **Live-lähetys:** kuvaa ei voi hakea taaksepäin, joten sitä ei voi kalibroida eikä siihen voi valita kansiota: anna `NETTI_KANSIO` (esim. `tulokset_nettivideo3` tai aiemmin luotu `tulokset_auto_yt_<tunnus>`).
* **`NETTI_KALIBROINTI=valitse`** palauttaa vanhan tavan: videon kuvaa verrataan kunkin kansion (`tulokset*`, `/data/kalibroinnit/*`) taustakuvaan, käytetään **ensimmäistä** sopivaa (ikkunasiirtymän p90 < 1,5 px) ja
  uusi kalibrointi tehdään vain, jos mikään ei sovi. Se on nopeampi, kun kamera on varmasti samassa asennossa, mutta sopivuustarkistus on herkkä yksittäiselle huonolle ikkunalle, se valitsee ensimmäisen eikä
  parhaan sopivan, ja se ei kerro, oliko valittu kalibrointi alun perinkään hyvä.
* `NETTI_KANSIO` (kansion nimi tai polku) ohittaa molemmat: sitä kansiota käytetään sellaisenaan.

Jokaisesta kalibroidusta videosta jää kansio `/data/kalibroinnit/tulokset_auto_…` (noin 4 Mt); saman videon uusinta-ajo kirjoittaa sen uudelleen. Vanhat kansiot voi poistaa, jos et tarvitse niitä (`NETTI_KANSIO`-viittaukset
tai `valitse`-tila käyttävät niitä).

Oman kalibrointikansion voi luoda myös käsin (`python /app/tools/uusi_kalibrointi.py VIDEO --ulos /data/kalibroinnit/oma`) ja ottaa käyttöön `NETTI_KANSIO=oma`.

---

## 5. YouTube

* Helpoin tapa antaa YouTube-osoite on linkkilista `youtube.txt` ([Käyttö c](#3-käyttö-tavallisimmat-tilanteet)): ei uudelleenkäynnistystä.
* Kuvassa ovat `yt-dlp`, sen JavaScript-ajuri (`deno`) ja `yt-dlp-ejs`; YouTuben muutokset rikkovat yt-dlp:n aika ajoin. **Ensimmäinen keino kaikkiin YouTube-ongelmiin: `YTDLP_PAIVITA=1`**
  (päivittää yt-dlp:n joka käynnistyksellä) tai kuvan uudelleenrakennus (`docker build --no-cache …`).
* Virheilmoitus *Sign in to confirm you're not a bot* tai ikärajaviesti: YouTube vaatii kirjautumisen. Vie selaimesta (kirjautuneena) evästeet Netscape-muodossa tiedostoon
  `/mnt/user/kivenseuranta/cookies.txt` (jakokansion juureen, **ei** kansioon `ulos`; esim. selainlaajennus "Get cookies.txt") ja aseta `YTDLP_KEKSIT=/data/cookies.txt`. Tiedostoa ei muuteta (ohjelma käyttää kopiota). Käytä mieluiten erillistä
  tiliä: evästeet antavat pääsyn tiliin.
* Live-virralle yt-dlp antaa HLS-osoitteen, jota OpenCV lukee; jos katko pitää lähetyksen, ohjelma yrittää avata virran uudelleen viisi kertaa.
* Reaaliaikainen live-analyysi vaatii, että kone jaksaa analysoida 30 ruutua sekunnissa ([Resurssit](#7-resurssit-ja-suorituskyky)); muuten kuvan viive kasvaa ja ruutuja ohitetaan.

---

## 6. Tulokset ja selainnäkymä

`ulos/`:

| Tiedosto | Sisältö |
|---|---|
| `kivenseuranta_<videon nimi>_<lähde>_<aika>.mp4` | Alkuperäinen kuva piirroksineen (liuku- ja irroituslinjat, tietopaneeli, joukkueiden kello) lähdevideon aikajanalla ja mittasuhteilla (1920 × 1080). Linjat näkyvät heti heiton alusta, koska analyysi on `NETTI_VIIVE` s kuvaa edellä. Noin 1,5 Gt tunnissa |
| `heitot_<videon nimi>_<lähde>_<aika>.csv` | Yksi rivi heittoa kohti: viivaestimaatti, nopeus, hidastuvuus, kierre, hog–hog-aika, kaukopään pysähtyminen, kiven vakiot, kello. Sarakkeiden selitykset: [`README_netti.md`](README_netti.md) |
| `heitot_<videon nimi>_<lähde>_<aika>_kivet.csv` | Kivikohtaiset vakiot (kitkataso A, kaarto kappa) |
| `kasitellyt.json` | Valvonnan muisti (vain valvontatilassa): valmiit videot ja linkit, epäonnistuneiden yritykset |

**Pelaajakohtainen tilasto.** `docker exec -it kivenseuranta python /app/tools/pelaajatilasto.py /data/ulos/heitot_<…>.csv --pelaajia 4` laskee heitto-CSV:stä pelaajakohtaisesti mediaanin ja STD:n (liun hidastuvuus, irroituksen delta, irroitusvauhti = nopeus irroituksen jälkeen miinus ennen) ja tallentaa taulukon samaan kansioon (`…_pelaajat.csv`; `--fi` = Excelille puolipiste ja desimaalipilkku). Pelaajamäärä joukkueittain: `--keltainen 4 --punainen 3` (ilman niitä ohjelma kysyy). Kuvaus: [`README_netti.md`](README_netti.md).

**Tiedostojen nimet.** YouTube-lähteellä (`youtube.txt`, `NETTI_LAHDE`) videon nimi lisätään tiedostojen nimiin, jotta oikean pelin löytää helposti: yt-dlp kertoo nimen samalla kutsulla kuin virran osoitteen (ei lisäkutsuja), ja nimi näkyy
myös selainnäkymässä (sivun tilarivi ja välilehden otsikko) sekä lokissa (`Video: …`, `ajo paattyi (…)`). Esimerkki: `kivenseuranta_Matarmaa SM-finaali 2026_yt_hnTTLqHr5x8_20261104_163130.mp4`. `<lähde>` on `yt_<tunnus>` (YouTube),
tiedoston nimi (videotiedosto; sen nimi on jo mukana) tai `virta` (`--url`). Nimi siistitään tiedostonimeksi: kielletyt merkit (`< > : " / \ | ? *`), `%`, `$` ja emojit jätetään pois, välilyönnit yhdistetään, ja nimi lyhennetään enintään 80 merkkiin (120 tavuun).
Ääkköset säilyvät (Linux / SMB-jako); jos YouTube ei kerro nimeä, tiedostonimet ovat kuten ennen (`kivenseuranta_yt_<tunnus>_<aika>.mp4`).

CSV päivittyy ajon aikana heiton valmistuttua. Ajon päätyttyä tai pysäytyksen (`docker stop`, Unraidin Stop) yhteydessä tallenne suljetaan ja CSV kirjoitetaan valmiiksi.

**Selainnäkymä** (`http://PALVELIN:63884`): kuva (MJPEG; toimii kaikissa selaimissa ja puhelimessa), viimeiset lokirivit, ajon tila ja lista tulostiedostoista. Mp4 toistuu selaimessa ja CSV latautuu. Näkymä
kuuluu ajoon: **kun yksikään ajo ei ole käynnissä** (valvonnassa odotusaikana tai ajon päätyttyä) portti ei vastaa. Kuva pakataan vain, kun joku katsoo, joten näkymä ei kuormita ajoa, kun kukaan ei seuraa.

**Turvallisuus:** selainnäkymässä ei ole oletuksena kirjautumista, ja se näyttää tulostiedostot. Käytä sitä kotiverkossa äläkä avaa porttia internetiin. `NETTI_WEB_KAYTTAJA` ja `NETTI_WEB_SALASANA`
lisäävät HTTP Basic -tunnistuksen (salasana kulkee salaamatta; internetiin vain TLS-päätteisen käänteisen välityspalvelimen takaa). Näkymä antaa vain
lukuoikeuden, ja vain tulostiedostoihin (`mp4`, `csv`, `json`, `txt`, `log` suoraan kansiossa `ulos`). Lisää: [Tietoturva ja verkkoyhteydet](#11-tietoturva-ja-verkkoyhteydet).

---

## 7. Resurssit ja suorituskyky

* **Prosessori.** Reaaliaikainen analyysi (live, tai tiedosto `NETTI_NOPEA=0`) vaatii, että analyysi ehtii 30 ruutua sekunnissa; heiton aikana kuorma on suurimmillaan. Suositus vähintään 4 ydintä.
  Ohjelma lukee kontin ydinrajan (`--cpus`, CPU pinning / `cpuset`) ja mitoittaa työprosessit sen mukaan (loki: `N ydinta kaytossa`). Mittausreferenssi: testikoneella (4 ydintä, kontti) tiedoston
  analyysi `NETTI_NOPEA=1`-tilassa eteni noin 65 ruutua sekunnissa (2,2 × reaaliaika), ja reaaliaikainen tiedostoajo selainnäkymän kanssa pysyi kuvassa mukana alle 0,1 s viiveellä. Oman palvelimen
  riittävyyden voi mitata: `python /app/tools/mittaa_nopeus.py /data/video/peli.mp4 --alku 300 --loppu 360` (keskimääräisen ruudun ajan pitää olla alle 33 ms reaaliaikaan).
* **Muisti.** Viivepuskuri (10 s): noin 1,9 Gt raakakuvina (kun muistia on vähintään 14 Gt) tai 0,9 Gt tiivistettynä; YouTube / virta: esilatauspuskuri 30 s ≈ 2,8 Gt. Ohjelma lukee kontin muistirajan
  (`--memory`) ja valitsee puskurit sen mukaan; tiedostolle 4 Gt riittää, YouTubelle suositus 8 Gt tai enemmän. Jos kontti kuolee äkillisesti ilman virheilmoitusta, syy on yleensä muisti (`NETTI_LATAUS=8`).
* **Levy.** Kuvallinen mp4 noin 1,5 Gt tunnissa (mitattu: 53 Mt / 130 s). Vanhoja tuloksia ei poisteta automaattisesti.
* **GPU-koodaus (ei testattu).** Oletuksena kuva koodataan prosessorilla (`libx264 superfast`, noin 1–1,5 ydintä). `NETTI_KOODEKKI=auto` kokeilee laitteistokooderia: ffmpeg (6.1) sisältää
  NVENC-, QSV- ja VAAPI-kooderit, mutta ne toimivat vain, jos ajurit ovat kontissa. **Nvidia:** Unraidin Nvidia-ajuriliitännäinen, kontin lisäparametri `--runtime=nvidia` ja muuttujat
  `NVIDIA_VISIBLE_DEVICES=all`, `NVIDIA_DRIVER_CAPABILITIES=all`. **Intel (QSV):** laite `/dev/dri` kontille ja Intelin media-ajurit kuvaan (eivät ole mukana). Ohjelma kokeilee kooderin ennen
  käyttöä ja putoaa prosessorikoodaukseen, jos se ei toimi; lokista näkee valinnan (`Tallennus: … (ffmpeg (...))`).
* **Rajaa kontin käyttö** Unraidissa lisäparametreilla (`--cpus=4 --memory=8g`) tai CPU pinningillä.

---

## 8. Päivitys

* **Ohjelman päivitys:** kopioi uusi `Netti_01_02` palvelimelle **kokonaan, myös alikansio `tools/`** (osittain päivitetty kansio sekoittaa versioita: esim. uusi `netti_live.py` ja vanha `tools/uusi_kalibrointi.py`), aja
  `docker build -t kivenseuranta:latest .` uudelleen ja käynnistä kontti uudelleen (Unraid: *Force update* / poista ja luo kontti). Data (`/data`) säilyy. Tarkistus, että kuvan `tools/` on ajan tasalla:
  `docker exec kivenseuranta grep -c aikaraja_s /app/tools/uusi_kalibrointi.py` (0 = vanha tiedosto).
* **yt-dlp:** `YTDLP_PAIVITA=1` tai `docker build --no-cache`.
* Rakennus tarvitsee verkon (Ubuntun paketit, PyPI). Kuva on noin 1,1 Gt.

---

## 9. Vianetsintä

| Oire | Syy / korjaus |
|---|---|
| Kontti sammuu heti, loki: `asetusvirhe: …` | Virheellinen ympäristömuuttuja; viesti kertoo, mikä. Paluukoodi 2 |
| Loki: `ei analysoitavia videoita` (kertaajo) | Kansiossa ei ole videota; valvonnassa (`NETTI_JATKUVA=1`) kontti odottaa |
| Video ei käynnisty valvonnassa | Tiedosto on vielä muuttunut viimeisen `NETTI_VAKAUS_S` sekunnin aikana, tai se on jo `kasitellyt.json`:ssa |
| Linkkilistan rivi ei käynnisty | (1) Tiedoston nimen on oltava täsmälleen `youtube.txt` jakokansion juuressa; Windows voi piilottaa päätteen ja tallentaa nimellä `youtube.txt.txt`. (2) Kontin lähde-asetus (`NETTI_LAHDE`) on oltava tyhjä (tai kansio). (3) Rivi on osoite tai yksin rivillä oleva 11 merkin tunnus; muut rivit mainitaan lokissa. (4) Edellinen ajo on kesken: uusi rivi huomataan sen jälkeen. (5) Rivi on jo ajettu (`kasitellyt.json`) tai odottaa uutta yritystä (loki kertoo ajan) |
| `Permission denied` / tiedostot root-omisteisia | `PUID`/`PGID` eivät vastaa jakokansion omistajaa; Unraid: 99/100. Kontti vaihtaa vain kansioiden `ulos`, `video`, `kalibroinnit` omistajan, ei tiedostoja |
| Kontti ei käynnisty, Docker kertoo `port is already allocated` | Palvelimen portti on jo toisella kontilla tai palvelulla: vaihda portin yhdistyksessä palvelimen puoli (Unraid: *Host Port*) |
| Selainnäkymä ei aukea | Ajoa ei ole käynnissä (valvonnan odotus; näkymä on vain ajon aikana), väärä portti osoitteessa, tai `host`-verkossa portti on varattu (loki: `Web-porttia … ei voitu avata`) |
| `YouTube-virran haku epäonnistui` | [YouTube](#5-youtube): päivitä yt-dlp, tarkista verkko; live-lähetys ei ehkä ole alkanut (valvonnassa kontti yrittää uudelleen) |
| Kuvan viive kasvaa / `Huom: piirto / ikkuna on N s jäljessä` | Kone ei jaksa reaaliaikaa: lisää ytimiä, `NETTI_X264_PRESET=ultrafast`, pienempi `NETTI_WEB_LEVEYS`, tiedostolle `NETTI_NOPEA=1` |
| Selainnäkymä sanoo `Ajo käynnissä`, mutta kuva ei etene (`viimeisin kuva N s sitten` kasvaa, sivulla `kuva ei etene`) | **Jumivahti** kirjoittaa lokiin 60 s kuluttua raportin `Jumivahti: ajo ei ole edennyt …` (ajon tila ja kaikkien säikeiden pinot): lähetä se, niin syy selviää. Raja: `NETTI_LISAVALITSIMET=--jumivahti 30` (`0` = pois). Aiemmin tämän aiheutti pitkä jono kopioruutuja (hiljainen tauko videolla: kuva ja analyysi odottivat toisiaan, korjattu), ja virran yhteyden katko näkyy raportissa kohdassa `lähteestä luettu viimeksi N s sitten` |
| Loki: `Tallennusvirhe (ffmpeg …)` | Tallenne keskeytyi (levy täynnä tai kooderi kaatui); analyysi, CSV ja selainnäkymä jatkavat. Tarkista levytila (`ulos`) ja aja uudelleen |
| Loki: `tarkka sovitus: Python` | C++-sovitin ei lataudu; kuvan rakennus katkeaa siihen, joten tätä ei pitäisi tulla. Python-varapolku on hitaampi eikä anna samoja tuloksia |
| Loki: `automaattinen kalibrointi epaonnistui (TypeError(… unexpected keyword argument 'aikaraja_s'))` tai `Huom: … uusi_kalibrointi.py on vanha versio` | Ohjelmakansio on päivitetty vain osittain (tyypillisesti alikansio `tools/` jäi kopioimatta): kopioi koko `Netti_01_02` (myös `tools/`) ja rakenna kuva uudelleen ([Päivitys](#8-päivitys)). Nykyversio kalibroi silti (ilman aikarajaa) ja kertoo, mikä tiedosto on vanha; aiemmat versiot kaatuivat tähän virheeseen ja käyttivät vanhaa kansiota |
| Kalibrointivaroitus `kalibrointi … ei sovi tähän videoon` | Kamera-asento on eri kuin kalibroinnissa: ks. [Kalibrointi](#4-kalibrointi) |
| Heittoja ei tunnisteta (tai vain osa) ja lokissa `VAROITUS: kalibrointi ei sovi tahan videoon` | Uusi kalibrointi ei onnistunut (renkaita tai viivoja ei löytynyt) ja käytetään vanhaa kansiota: katso lokista kalibroinnin viestit (`kalibrointi epaonnistui …`, `kaytetaan lahinta vanhaa kansiota … poikkeama …`) ja ks. [Kalibrointi](#4-kalibrointi) |
| Videon alussa pitkä viive ennen kuin kuva tulee (YouTube) | Video kalibroidaan ensin virrasta (noin 0,5–2 min, lokissa `Kalibroidaan video uudelleen …`); `NETTI_KANSIO` ohittaa kalibroinnin, `NETTI_KALIBROINTI=valitse` käyttää sopivaa vanhaa |
| Kontti kuolee ilman virhettä | Muistin loppuminen: pienennä `NETTI_LATAUS` (8) tai anna kontille lisää muistia |
| Lokin käynnistysrivi | `N ydinta kaytossa (koneessa M…)`, `muistia X Gt`, lähde, valvonta, kansiot ja portti: tarkista, että rajat ovat odotetut |

Lokit: Unraidin Docker-välilehti → kontin *Logs*, tai `docker logs -f kivenseuranta`; selainnäkymän Loki-osa näyttää saman.

---

## 10. Mitä eroaa `Netti_01_01`:stä

* **Uudet tiedostot:** `Dockerfile`, `entrypoint.sh`, `docker-compose.yml`, `unraid/kivenseuranta.xml`, `requirements.txt`, `.dockerignore`, `.gitattributes`, `netti_palvelin.py` (käynnistäjä,
  asetukset, valvonta), `netti_web.py` (selainnäkymä), `netti_ymparisto.py` (kontin ydin- ja muistirajat), `tools/testaa_palvelin.py`.
* **Muutetut tiedostot (vain palvelinominaisuudet):** `netti_live.py` (valitsimet `--web`, `--web-leveys`, `--web-fps`, `--web-laatu`, `--tila-tiedosto`; SIGTERM käsitellään kuten Ctrl-C; simulaation
  keskeytys viimeistelee kirjauksen; työprosessien oletusmäärä ja muistin tunnistus kontin rajoilla; oma kalibrointikansio `NETTI_KALIBROINNIT`; YouTube-evästeet), `kalibrointi_valinta.py` (oma
  kalibrointikansio), `netti_aani.py` (evästeet), `cpp/netti_fit.cpp` (vain `#include`-rivit: tarvitaan vain OpenCV:n `core` ja `imgproc`). Algoritmit, parametrit ja tulokset ovat ennallaan.
* **Mukana ei ole:** vanhat kameraohjelmat (`live.py`, `main.py` …), opetus- ja kehitystyökalut, kulmaverkon painot (niitä ei käytetä ajossa), ikkunanäkymä (`--nayttotila`) ei ole käytössä (OpenCV on ilman ikkunatukea); palvelimella
  käytetään `--web`-näkymää.
* Kuvan Python on 3.12 (Ubuntu 24.04) ja OpenCV `opencv-python-headless` (ei ikkunatukea).

---

## 11. Tietoturva ja verkkoyhteydet

**Lyhyesti.** Kontti on tavallinen rajattu sovelluskontti: se ei ole `privileged`, sillä ei ole `docker.sock`-liitosta, laitteita eikä `host`-verkkoa, ja se näkee vain yhden jakokansion (`/data`).
Analyysi ajetaan tavallisena käyttäjänä (`PUID`/`PGID`, Unraidissa `nobody:users`); root-oikeuksia käytetään vain käynnistyksen alussa kansioiden ja käyttäjän luontiin. Suurin käytännön riski on
**selainnäkymän portti**, jos se päätyy julkiseen verkkoon.

**Milloin kontti ottaa yhteyttä verkkoon**

| Tilanne | Minne | Mihin tarvitaan |
|---|---|---|
| `docker build` | Docker Hub (`ubuntu:24.04`), Ubuntun pakettipalvelimet, PyPI | kuvan rakennus |
| Videotiedoston analyysi | ei minnekään | ajo onnistuu täysin ilman verkkoa (testattu verkottomassa nimiavaruudessa) |
| YouTube-linkki tai live | YouTuben ja Googlen palvelimet (esim. `youtube.com`, `googlevideo.com`) | videon ja äänen haku (yt-dlp) |
| Muu verkko-osoite, `rtsp://` | itse antamasi osoite | virran luku |
| `YTDLP_PAIVITA=1` | PyPI | yt-dlp:n päivitys käynnistyksessä (oletus pois) |
| Selainnäkymä | vastaanottaa yhteyksiä portissa 63884 | katselu |

Ohjelmassa ei ole telemetriaa, eikä se lähetä tuloksia minnekään. Omassa koodissa ainoa verkkopalvelu on selainnäkymä; ulosmenevät yhteydet tekevät yt-dlp (YouTube), OpenCV / FFmpeg (annettu virta)
ja valinnaisesti pip. Kontin ulosmenevää liikennettä ei ole rajoitettu: se voi ottaa yhteyden mihin tahansa osoitteeseen, joka kirjoitetaan linkkilistaan tai asetukseen `NETTI_LAHDE`.

**Selainnäkymä.** Se kuuntelee kaikkia verkkorajapintoja (portti 63884), oletuksena ilman salasanaa ja ilman salausta (HTTP). Se näyttää kuvan, lokin (polut ja osoitteet), ajon tilan ja kansion `ulos`
tulostiedostot (`mp4`, `csv`, `json`, `txt`, `log`; vain luku). Se ei anna pääsyä muuhun jakokansioon, ei voi ajaa komentoja eikä ota vastaan tiedostoja; polkuhyökkäysten esto on testattu.
Älä silti avaa porttia internetiin.

**Suositukset (tärkeimmästä alkaen)**

1. **Älä avaa selainnäkymän porttia internetiin** (ei portinavausta reitittimessä, ei DMZ:tä, ei UPnP:tä). Etäkäyttöön VPN (esim. Unraidin WireGuard tai Tailscale).
2. **Aseta salasana** (`NETTI_WEB_KAYTTAJA` ja `NETTI_WEB_SALASANA`), jos kotiverkossa on laitteita, joihin et luota (vierasverkko, älylaitteet), tai poista selainnäkymä (`NETTI_WEB_PORTTI=0`).
   Salasana kulkee HTTP:ssä salaamatta ja näkyy kontin asetuksissa ja `docker inspect`-tulosteessa, joten se sopii vain kotiverkkoon.
3. **Rajaa jakokansion kirjoitusoikeus.** Kansioon `video` ja tiedostoon `youtube.txt` kirjoittava päättää, mitä kontti analysoi ja mihin osoitteisiin se ottaa yhteyttä.
4. **Analysoi vain luotettuja videoita ja osoitteita.** Videon purku (FFmpeg / OpenCV) on tyypillinen hyökkäyspinta, jossa on aika ajoin haavoittuvuuksia. YouTube ja omat tallenteet ovat tavallisesti
   turvallisia; tuntemattomalta saatu tiedosto tai `rtsp://`-osoite ei ole.
5. **Päivitä kuva silloin tällöin:** `docker build --no-cache -t kivenseuranta:latest .` tuo Ubuntun tietoturvapäivitykset (esim. ffmpeg). Python-kirjastot numpy, scipy ja opencv on kiinnitetty
   versioihin (`requirements.txt`), joten niiden päivitys vaatii tiedoston muuttamista; yt-dlp ja deno ovat rakennushetken uusimmat.
6. **YouTube-evästeet** (`YTDLP_KEKSIT`) antavat pääsyn Google-tiliin. Käytä erillistä tiliä, pidä `cookies.txt` jakokansion juuressa (ei kansiossa `ulos`, jonka tiedostoja selainnäkymä tarjoaa) ja rajaa
   sen oikeudet. Ohjelma käyttää tiedostosta kopiota (oikeudet 0600), joka poistetaan ajon päättyessä.
7. **`YTDLP_PAIVITA=1`** asentaa paketin PyPI:stä kontin root-oikeuksilla käynnistyksessä. Pidä se pois päältä (oletus), ellet tarvitse päivitystä.

**YouTuben JavaScript.** YouTube edellyttää nykyisin, että yt-dlp ratkaisee sivun JavaScript-haasteita. Se tehdään `deno`-ajurilla, joka ajaa koodin ilman tiedosto-, verkko- tai ympäristöoikeuksia:
yt-dlp antaa `deno`lle vain valitsimet `--no-prompt --no-remote ...` eikä yhtään `--allow-*`-lupaa. Kokeessa tällainen ajo ei päässyt lukemaan tiedostoa eikä avaamaan verkkoyhteyttä (`NotCapable`),
eikä `deno` ottanut itse yhteyttä verkkoon.

**Mitä tässä ei ole:** ulosmenevän liikenteen rajoitusta, selainnäkymän salausta (TLS), kirjautumisyritysten rajoitusta eikä riippuvuuksien hash-tarkistusta. Raskaamman eristyksen (esim. `--cap-drop`,
`--read-only`) vaikutusta ei ole testattu kokonaisella analyysiajolla, joten sitä ei suositella tässä.

---

## Testaus

```bash
docker run --rm kivenseuranta:latest testaa                      # koko testisarja (tools/testaa_vakiot.py), useita minuutteja
docker run --rm kivenseuranta:latest python /app/tools/testaa_palvelin.py   # palvelinversion testit (alle 10 s)
```

Palvelinversion testit kattavat: asetusten luku ja virheilmoitukset, lähteen tunnistus, komentorivin rakennus, kansion valvonta ja muisti, ajosilmukka (kertaajo, valvonta, uudelleenyritykset,
odotuksen pidentyminen, pysäytys), signaalit (lapsiprosessin viimeistely ja tappaminen), selainnäkymä (sivu, kuva, MJPEG, tila, loki, tiedostot, Range, polkuhyökkäysten esto, salasana, varattu portti),
kontin ydin- ja muistirajat sekä kalibrointikansioiden haku.

Rakennuksen ja ajon varmistus tässä kansiossa (Docker 29, Ubuntu 24.04): 130 s:n videoleike (kaksi kokonaista heittoa) analysoitiin kontissa; tuloksena syntynyt CSV on **tavu tavulta sama** kuin
saman leikkeen natiivi ajo C++-sovittimella (Python 3.13), mp4 on ehjä (3 901 ruutua, 130,03 s) ja `docker stop` kesken ajon sulki tallenteen ja CSV:n (paluukoodi 0, ~1 s).
