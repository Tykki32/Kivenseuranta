"""Maalimerkin (kaukopaan harjanpaa) haku: yksi ruutu (ehdokkaat, etsi) ja moniruutuhaku (konsensus, MerkkiSeuranta).

Heittajan maali on kaukopaassa: joukkueen kippari pitaa harjaa jaalla harjanpaa kohti heittajaa, ja heittaja tahtaa sita kohti. Harjanpaa on muovipala, jonka sävy vaihtelee videoittain ja harjoittain
(oranssi H ~ 13-16, ruskea H ~ 8-11, tummanruskea V ~ 95-130, punaruskea H ~ 4-7), joten maali etsitaan varista ja muodosta. Seuraavat sekoittuvat siihen ja on erotettava:

  * kiven keltainen kahva (osassa videoita koko kahva on samaa savya kuin harjanpaa): kahva on pieni, kaareva ja sen pitka akseli on poikittain kenttaan nahden, harjanpaa taas kentan suuntainen,
  * puulattia (sama sävy kuin ruskea harjanpaa): leikataan pois taustakuvasta (_lattia) ja |X'| <= X_MAX_CM -kaistalla,
  * harjan puuvarsi ja harjanpaan punaisen renkaan / kahvan halo (samaa tai viereista savya, liittyvat harjanpaahan): savytarkennus ja paksummat avaukset irrottavat ne,
  * jaalla lojuvat ja sivussa seisovien pelaajien harjat pesan takana (Y' < TAKA_Y_MIN_CM).

Yksi ruutu: kaukopaan kuva oikaistuna (Mittari.kuvain.leike) -> HSV-maski (H 4-17, S >= 90, V >= 70) rajattuna jaalle -> avaus 5 px (ohuet rakenteet pois) -> yhtenaiset alueet. Alue on kelvollinen,
kun sen keskipiste (maatasolle, z = 3 cm) on kaistalla, ala on harjanpaan kokoinen (MERKKI_ALA_CM2), sivusuhde 1,4-7 ja pitka akseli enintaan MERKKI_KULMA_MAX astetta kentan suunnasta (oikaistussa kuvassa
vaaka). Koon, muodon tai pituuden (> PITUUS_MAX_CM) puolesta kelpaamaton kaistalla oleva alue pilkotaan: ensin savytarkennus (alueen oman savymoodin +- SAVY_KAISTA: halo ja viereiset kahvat pois), sitten
paksummat avaukset (9-13 px: puuvarsi ~5-9 px irtoaa, harjanpaa >= 10 px jaa). Havaintoja: harjanpaa on kaikissa viidessa videossa kentan suuntainen (0-19 astetta), kivien kahvat 60-90 astetta.

Moniruutuhaku: yhden ruudun tulos on altis peittymiselle (pelaaja, jalat, toinen harja), savyn vaihtelulle ja siihen, etta harja tuodaan paikalleen vasta heiton alettua. Siksi kaukopaata skannataan
SKANNAUS_VALI_S valein (MerkkiSeuranta) ja heiton ymparilta (KONSENSUS_ALKU_S ... KONSENSUS_LOPPU_S lahdosta) kerattyja ehdokkaita yhdistetaan: saman paikan havainnot eri ruuduissa muodostavat klusterin
(klusteroi), jonka pisteet = esiintyma (osuus tavoiteikkunan TAVOITE_ALKU_S ... TAVOITE_LOPPU_S ruuduista, joissa harja nakyy) x kokopaino. Maali on parhaat pisteet saanut klusteri, joka ei ole pesan takana,
esiintyy vahintaan MIN_ESIINTYMA ja on vahintaan MIN_ALA_CM2; sen paikka on havaintojen mediaani. Samalla linjalla (LINJA_DX_CM) yhtaikaa nakyvat klusterit ovat saman harjan osia (harjanpaa + puuvarsi tai
kahvaosa, joka on joissakin harjoissa samaa savya ja yhta suuri kuin harjanpaa): niista harjanpaa on heittajan puoleisin (Y' suurin), koska harja osoittaa harjanpaan heittajaa kohti. `ehdokkaita` kertoo,
montako harjaa sai vahintaan puolet parhaista pisteista (1 = yksiselitteinen, 2 = esim. kaksi pelaajaa pitaa harjaa pesassa: kumpi on kipparin, ei nay kuvasta). Maalia ei loydy (None), kun mikaan klusteri ei kelpaa: harjanpaata ei ole nakyvissa (kukaan ei pida harjaa pesassa, tai
harjanpaa on valkomusta tms., jota vari ei erota).

Koordinaatit kaukopaan koordinaateissa (X', Y', cm; origo pesan keskipiste, kuten kaukopaan seurannassa). Moduuli ei tunne seurantaa muuten kuin Mittarin (kuvain, kamera, koordinaatit) kautta."""
import collections

import cv2
import numpy as np

import kivimalli

H_MIN, H_MAX = 4, 17                    # harjanpaan sävy (OpenCV 0-179): punaruskea ~4-7, ruskea ~8-11, oranssi ~13-16, kiven keltainen kahva ~13-22 (videosta riippuen)
S_MIN, V_MIN = 90, 70                   # matala V_MIN: tummanruskea harjanpaa (V ~ 95-130) ja varjossa oleva kuuluvat mukaan
AVAUS_PX = (5, 9, 11, 13)                # avauskoot: 5 poistaa ohuet rakenteet (kahvan reunus ~3 px), loput irrottavat harjanpaan varresta (~9 px)
X_MAX_CM = 255.0                         # kaista: jaa (puulattia alkaa ~250 cm:ssa; lattia rajataan lisaksi taustakuvasta, _lattia)
LATTIA_H = (5, 22)                       # puulattian savy taustakuvassa (sama kuin ruskean harjanpaan)
LATTIA_S_MIN, LATTIA_V_MIN = 50, 80
LATTIA_MIN_PX = 3000                     # yhtenainen lattia-alue vahintaan taman kokoinen
LATTIA_LAAJENNUS_PX = 7
Y_RAJAT_CM = (-330.0, 520.0)
Z_CM = 3.0                               # harjanpaa on jaalla
Z_ALIN_CM = 0.0                          # harjanpaan alin kohta (heittajan puoleisen paan alareuna) on jaassa kiinni
MERKKI_ALA_CM2 = (45.0, 900.0)           # mitattu: 70-880 cm2, mediaani 370 (isoimmissa harjanpaa + varsi tai punaisen renkaan halo; osittain peittyneena pienempi)
MERKKI_SIVUSUHDE = (1.4, 7.0)
SAVY_KAISTA = 3                          # savytarkennus: harjanpaan oman savyn (kyllaisyydella painotettu moodi) +-taman sisalla olevat pikselit; halo ja viereiset kahvat pois
PITUUS_MAX_CM = 42.0                     # pidempi alue sisaltaa todennakoisesti puuvarren (harjanpaan pituus 25-40 cm): varsi irrotetaan paksummalla avauksella
MERKKI_KULMA_MAX = 35.0                  # harjanpaan pitka akseli kentan suuntaan (mitattu 0-19 astetta, mediaani 0; kiven kahvat 60-90)
TAKA_Y_MIN_CM = -190.0                   # pesan takaraja (12 jalan rengas 183 cm): harjanpaa on pesassa tai sen edessa; taempana olevat ovat lojuvia tai sivussa odottavien harjoja
MIN_PX = 25
REUNA_PX = 4                             # kuvan reunan katkaisema harjanpaa: alue koskettaa (laajennettua) kuvan ulkopuolta


def _alue(mittari):
    cam = mittari.cam
    P = np.array([(x, y, z) for x in (-X_MAX_CM - 40.0, X_MAX_CM + 40.0) for y in Y_RAJAT_CM for z in (0.0, 30.0)])
    u, v = kivimalli.project_3d(cam.K, cam.R, cam.t, P)
    return (int(u.min()) - 6, int(v.min()) - 6, int(u.max()) + 6, int(v.max()) + 6)


def _lattia(mittari):
    """Puulattia oikaistussa kuvassa (uint8 0/1, laajennettu): taustakuvan (jaa ilman ihmisia) yhtenainen ruskea alue. Puulattia on samaa savya kuin ruskea harjanpaa, joten se rajataan pois
    kaistan reunalta tasan (|X'| -raja ei riita: lattian reuna ei ole kuvassa X' = vakio). Valimuistissa Mittarilla."""
    v = getattr(mittari, "_merkki_lattia", None)
    if v is not None:
        return v
    hsv = cv2.cvtColor(mittari.cam.mode, cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 0] >= LATTIA_H[0]) & (hsv[..., 0] <= LATTIA_H[1]) & (hsv[..., 1] >= LATTIA_S_MIN) & (hsv[..., 2] >= LATTIA_V_MIN)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    f = np.zeros_like(m)
    for k in range(1, n):
        if st[k, cv2.CC_STAT_AREA] >= LATTIA_MIN_PX:
            f[lab == k] = 1
    f = cv2.dilate(f, np.ones((2 * LATTIA_LAAJENNUS_PX + 1, 2 * LATTIA_LAAJENNUS_PX + 1), np.uint8))
    mittari._merkki_lattia = f
    return f


def _reuna(mittari):
    """Oikaistun kuvan reunavyohyke (uint8 0/1): pikselit, joiden lahde on kameran puoliskon ulkopuolella (tai kuvan reunalla), laajennettuna REUNA_PX:lla. Harjanpaa, joka koskettaa vyohykettaa, on
    kuvan reunan katkaisema (nakyy vain osittain). Valimuistissa Mittarilla."""
    v = getattr(mittari, "_merkki_reuna", None)
    if v is not None:
        return v
    k = mittari.kuvain
    x0, y0, w, h = k.puolisko
    sisalla = (k.mx >= x0) & (k.mx <= x0 + w - 1) & (k.my >= y0) & (k.my <= y0 + h - 1)
    sisalla[[0, -1], :] = False
    sisalla[:, [0, -1]] = False
    r = cv2.dilate((~sisalla).astype(np.uint8), np.ones((2 * REUNA_PX + 1, 2 * REUNA_PX + 1), np.uint8))
    mittari._merkki_reuna = r
    return r


def _kaista(mittari, x0, y0, w, h, askel=8):
    """Leikkeen (x0, y0, w x h) pikselit, jotka ovat jaalla (|X'| <= X_MAX_CM, ei taustakuvan puulattiaa): puulattia on samaa savya kuin harjanpaa ja liittaisi sen mukaansa. Valimuistissa Mittarilla."""
    avain = (x0, y0, w, h)
    v = getattr(mittari, "_merkki_kaista", None)
    if v is not None and v[0] == avain:
        return v[1]
    yy, xx = np.mgrid[0:h + askel:askel, 0:w + askel:askel]
    X, _ = kivimalli.ray_plane_intersection(mittari.cam.K, mittari.cam.R, mittari.cam.t, (xx + x0).ravel(), (yy + y0).ravel(), Z_CM)
    X = cv2.resize(X.reshape(xx.shape).astype(np.float32), None, fx=askel, fy=askel, interpolation=cv2.INTER_LINEAR)[:h, :w]
    maski = (np.abs(X) <= X_MAX_CM).astype(np.uint8)
    maski &= 1 - _lattia(mittari)[y0:y0 + h, x0:x0 + w]
    mittari._merkki_kaista = (avain, maski)
    return maski


def _avaa(m, k):
    """Morfologinen avaus, joka poistaa alle k px paksut rakenteet mutta sailyttaa jaljelle jaavien osien koon."""
    e = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(cv2.morphologyEx(m, cv2.MORPH_OPEN, e), e) & m


def _savytarkenna(kmask, H, S):
    """Alueen (0/1) pikselit, joiden sävy on alueen oman savymoodin (kyllaisyydella painotettu histogrammin huippu) +- SAVY_KAISTA sisalla. Harjanpaan reunoilla on usein toisen savyn halo (punainen rengas
    / kahva) ja viereiset kiven kahvat liittyvat siihen; harjanpaan oma savy on tasainen, joten tarkennus erottaa sen naista."""
    m = kmask.astype(bool)
    hist = np.bincount(H[m].astype(int), weights=S[m].astype(float), minlength=180)
    moodi = int(np.argmax(np.convolve(hist, np.ones(3) / 3.0, mode="same")))
    return (m & (np.abs(H.astype(int) - moodi) <= SAVY_KAISTA)).astype(np.uint8)


def _alin_kohta(mittari, u, v, X, Y):
    """Alueen (oikaistun kuvan pikselit u, v; keskipiste maassa X, Y) alin kohta: kentan suunnassa heittajaa lahin paa (Y' suurin; kuvassa alin). Paa = 98. persentiilin ja 1,5 px:n sisalla olevat
    pikselit (kestaa yksittaiset reunapikselit), sivusuunnassa niiden keskikohta. Alareuna on jaassa (z = Z_ALIN_CM): kamera katsoo kaukopaata matalassa kulmassa (~16 astetta), joten
    harjanpaan paan pysty reuna nakyy ja sen alin pikseli on jaakosketuspiste. Palauttaa (X', Y') maassa."""
    pu, pv = mittari.maa_oikaistuun([X, X], [Y - 10.0, Y + 10.0], Z_CM)
    d = np.array([pu[1] - pu[0], pv[1] - pv[0]], float)
    d /= max(float(np.hypot(d[0], d[1])), 1e-9)                            # Y':n kasvusuunta oikaistussa kuvassa
    n = np.array([-d[1], d[0]])
    pit = u * d[0] + v * d[1]
    ala = pit >= np.percentile(pit, 98) - 1.5
    ext = float(pit[ala].max()) + 0.5                                      # paan reuna (pikselin ulkoreuna)
    lat = float(np.mean(u[ala] * n[0] + v[ala] * n[1]))
    Xa, Ya = mittari.oikaistu_maahan(ext * d[0] + lat * n[0], ext * d[1] + lat * n[1], z=Z_ALIN_CM)
    return float(Xa), float(Ya)


def _arvioi(mittari, m, x0, y0, H, S):
    """Maskin (0/1) kukin yhtenainen alue ehdokkaaksi: [(dict, alueen maski)]."""
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    reuna = _reuna(mittari)[y0:y0 + m.shape[0], x0:x0 + m.shape[1]]
    out = []
    for k in range(1, n):
        a = int(st[k, cv2.CC_STAT_AREA])
        if a < MIN_PX:
            continue
        kmask = lab == k
        ys, xs = np.nonzero(kmask)
        (_, _), (w, h), ang = cv2.minAreaRect(np.column_stack([xs, ys]).astype(np.float32))
        L, W = max(w, h), max(min(w, h), 1.0)
        kulma = (ang if w >= h else ang + 90.0) % 180.0          # pitkan akselin kulma kuvan vaakasuuntaan (= kentan suunta), 0-90
        kulma = min(kulma, 180.0 - kulma)
        u, v = float(cen[k][0] + x0), float(cen[k][1] + y0)
        X, Y = mittari.oikaistu_maahan(u, v, z=Z_CM)
        s = mittari.skaala(X, Y, Z_CM)
        ala = a / (s * s)
        syy = ""
        if abs(X) > X_MAX_CM or not (Y_RAJAT_CM[0] <= Y <= Y_RAJAT_CM[1]):
            syy = "kaistan ulkopuolella"
        elif not (MERKKI_ALA_CM2[0] <= ala <= MERKKI_ALA_CM2[1]):
            syy = "koko"
        elif not (MERKKI_SIVUSUHDE[0] <= L / W <= MERKKI_SIVUSUHDE[1]):
            syy = "muoto"
        elif kulma > MERKKI_KULMA_MAX:
            syy = "suunta"
        Xa, Ya = _alin_kohta(mittari, xs + x0, ys + y0, X, Y)
        out.append((dict(X=float(X), Y=float(Y), alin_X=Xa, alin_Y=Ya, u=u, v=v, ala_cm2=float(ala), sivusuhde=float(L / W), pituus_cm=float(L / s), leveys_cm=float(W / s), kulma=float(kulma),
                         sävy=float(np.median(H[kmask])), kyllaisyys=float(np.median(S[kmask])), reunalla=bool(reuna[kmask].any()), kelvollinen=(syy == ""), syy=syy), kmask.astype(np.uint8)))
    return out


def ehdokkaat(mittari, ruutu):
    """Kaikki harjanpaan nakoiset alueet kaukopaan kuvasta: [dict(X, Y, u, v, ala_cm2, sivusuhde, pituus_cm, leveys_cm, kulma, sävy, kyllaisyys, kelvollinen, syy)]. Ruutu: koko ruutu (BGR).
    Kaistalla oleva alue, jonka koko tai muoto ei kelpaa (harjanpaa + sen varsi samaa puunvarista), pilkotaan paksummilla avauksilla: varsi (~9 px) irtoaa, harjanpaa (>= 13 px) jaa."""
    roi, x0, y0 = mittari.kuvain.leike(ruutu, *_alue(mittari))
    if roi is None:
        return []
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    m = ((H >= H_MIN) & (H <= H_MAX) & (S >= S_MIN) & (V >= V_MIN)).astype(np.uint8)
    m = _avaa(cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) & _kaista(mittari, x0, y0, roi.shape[1], roi.shape[0]), AVAUS_PX[0])
    out = []
    for e, kmask in _arvioi(mittari, m, x0, y0, H, S):
        if e["syy"] in ("koko", "muoto") or e["pituus_cm"] > PITUUS_MAX_CM:
            # ensin savytarkennus (viereiset kahvat, halo), sitten yha paksummat avaukset (puuvarsi)
            yritykset = [_savytarkenna(kmask, H, S)] + [_avaa(kmask, k) for k in AVAUS_PX[1:]]
            yritykset[0] = _avaa(yritykset[0], AVAUS_PX[0])
            for ym in yritykset:
                osat = [f for f, _ in _arvioi(mittari, ym, x0, y0, H, S) if f["syy"] != "kaistan ulkopuolella"]
                if any(f["kelvollinen"] and f["pituus_cm"] <= PITUUS_MAX_CM for f in osat) or (e["syy"] and any(f["kelvollinen"] for f in osat)):
                    out += osat
                    break
            else:
                out.append(e)
        else:
            out.append(e)
    return out


def valitse(ehd):
    """Yhden ruudun maali: kelvollinen ehdokas, joka ei ole pesan takana (Y' >= TAKA_Y_MIN_CM: sielta loytyvat lojuvat tai sivussa seisovien pelaajien harjat), usean joukosta suurin;
    None, ellei sellaista ole. (Moniruutuhaku, konsensus, on luotettavampi.)"""
    kelp = [e for e in ehd if e["kelvollinen"] and e["Y"] >= TAKA_Y_MIN_CM]
    return max(kelp, key=lambda e: e["ala_cm2"]) if kelp else None


def etsi(mittari, ruutu):
    """(maali | None, kaikki ehdokkaat): maali = dict(X, Y, u, v, ...) kaukopaan koordinaateissa."""
    ehd = ehdokkaat(mittari, ruutu)
    return valitse(ehd), ehd


# ---------------------------------------------------------------------------------------------------------------------
# MONIRUUTUHAKU (konsensus): yhden ruudun tulos on altis peittymiselle, varivaihtelulle ja siihen, etta harja tuodaan paikalleen vasta heiton alettua
# ---------------------------------------------------------------------------------------------------------------------
SKANNAUS_VALI_S = 0.5                    # kuinka usein kaukopaan kuva tutkitaan
KONSENSUS_ALKU_S, KONSENSUS_LOPPU_S = -10.0, 12.0                # konsensusikkuna heiton lahdosta (s): ehdokkaat naiden aikojen ruuduista
TAVOITE_ALKU_S, TAVOITE_LOPPU_S = -2.0, 5.0                      # maalin pitaa nakya talla valilla (liuku ja irroitus); pituus ja sijainti mitataan tasta
KLUSTERI_DX_CM, KLUSTERI_DY_CM = 15.0, 30.0                      # saman harjan havainnot eri ruuduissa (harjaa pidetaan hieman liikkuen; Y' on kuvassa lyhentynyt, siksi laajempi)
MIN_ESIINTYMA = 0.3                      # osuus tavoiteikkunan ruuduista, joissa harja nakyy (esim. heiton alkua myohemmin paikalle tuotu harja)
MIN_ALA_CM2 = 40.0                       # klusterin (80. persentiili) alan alaraja (harjanpaa voi olla osittain peitossa tai lattian rajaama)
KOKO_VERTAILU_CM2 = 300.0                # tata pienemmat alueet saavat pienemman painon (kiven kahvat ja osittain peittyneet harjanpaat ovat 60-150 cm2, harjanpaa mediaanissa 340)
LINJA_DX_CM, LINJA_DY_CM = 20.0, 200.0   # saman harjan osat (harjanpaa, puuvarsi, kahvaosa) ovat samalla linjalla; harja on enintaan ~130 cm
YHTAIKAA_OSUUS = 0.3                     # osat nakyvat yhtaikaa (ei saman harjanpaan siirtymista toiseen paikkaan): vahintaan taman osuuden pienemman klusterin ruuduista


def klusteroi(sarja):
    """Saman paikan havainnot eri ruuduista klustereiksi. sarja: [(t, [ehdokas])] (t = aika heiton lahdosta, s; vain kelvolliset ehdokkaat huomioidaan). Palauttaa
    [dict(X, Y, jasenet=[(t, ehdokas)])]; klusterin keskipiste on jasenten keskiarvo, ruudusta otetaan klusteriin enintaan yksi (suurin) havainto."""
    kl = []
    for t, ehd in sarja:
        kasitelty = set()
        for e in sorted((e for e in ehd if e["kelvollinen"]), key=lambda e: -e["ala_cm2"]):
            paras, pd = None, 1e9
            for i, c in enumerate(kl):
                if i in kasitelty:
                    continue
                dx, dy = (c["X"] - e["X"]) / KLUSTERI_DX_CM, (c["Y"] - e["Y"]) / KLUSTERI_DY_CM
                if abs(dx) <= 1.0 and abs(dy) <= 1.0 and dx * dx + dy * dy < pd:
                    paras, pd = i, dx * dx + dy * dy
            if paras is None:
                kl.append(dict(X=e["X"], Y=e["Y"], jasenet=[(t, e)]))
                kasitelty.add(len(kl) - 1)
            else:
                c = kl[paras]
                c["jasenet"].append((t, e))
                c["X"] = float(np.mean([m["X"] for _, m in c["jasenet"]]))
                c["Y"] = float(np.mean([m["Y"] for _, m in c["jasenet"]]))
                kasitelty.add(paras)
    return kl


def konsensus(sarja):
    """Maali usean ruudun perusteella. sarja: [(t, [ehdokas])], t = aika heiton lahdosta (s), tutkitut ruudut tasavalein. Klusterit (klusteroi) pisteytetaan: esiintyma = osuus
    tavoiteikkunan (TAVOITE_ALKU_S .. TAVOITE_LOPPU_S) ruuduista, joissa harja nakyy, kerrottuna kokopainolla min(ala, KOKO_VERTAILU_CM2) / KOKO_VERTAILU_CM2 (ala = 80. persentiili: osittain
    peittynyt harja on valilla pienempi). Kelpaa vain klusteri, joka ei ole pesan takana (TAKA_Y_MIN_CM), esiintyy vahintaan MIN_ESIINTYMA ja on vahintaan MIN_ALA_CM2. Ellei sellaista ole,
    kelpaa kuvan reunan katkaisema harja (reunalla): se nakyy vain hetkittain, joten esiintyma lasketaan koko konsensusikkunan ruuduista.
    Palauttaa dict(X, Y, keski_X, keski_Y, ala_cm2, esiintyma, ehdokkaita, pisteet, reunalla, ...) tai None. Maali (X, Y) on harjanpaan alin kohta (heittajaa lahin paa, jaassa; mediaani
    tavoiteikkunan havainnoista), keski_X, keski_Y harjanpaan keskipiste. Kilpailevat klusterit (pisteet >= puolet parhaasta) ryhmitellaan harjoiksi: samalla linjalla (LINJA_DX_CM, LINJA_DY_CM)
    yhtaikaa nakyvat klusterit ovat saman harjan osia (harjanpaa + varsi tai kahvaosa), ja niista harjanpaa on heittajan puoleisin (Y' suurin). ehdokkaita = kilpailevien harjojen maara,
    1 = yksiselitteinen."""
    sarja = [(t, e) for t, e in sarja if KONSENSUS_ALKU_S <= t <= KONSENSUS_LOPPU_S]
    ikkuna = [t for t, _ in sarja if TAVOITE_ALKU_S <= t <= TAVOITE_LOPPU_S]
    if not ikkuna:
        return None
    kl = klusteroi(sarja)
    ehdolla = []
    for c in kl:
        jas = [(t, e) for t, e in c["jasenet"] if TAVOITE_ALKU_S <= t <= TAVOITE_LOPPU_S]
        esiintyma = len({t for t, _ in jas}) / len(ikkuna)
        ala = float(np.percentile([e["ala_cm2"] for _, e in (jas or c["jasenet"])], 80))
        Y = float(np.median([e["Y"] for _, e in (jas or c["jasenet"])]))
        if Y < TAKA_Y_MIN_CM or esiintyma < MIN_ESIINTYMA or ala < MIN_ALA_CM2:
            continue
        ehdolla.append(dict(c, jas=jas or c["jasenet"], ajat={t for t, _ in jas}, esiintyma=esiintyma, ala_cm2=ala, pisteet=esiintyma * min(ala, KOKO_VERTAILU_CM2) / KOKO_VERTAILU_CM2, Y=Y,
                            X=float(np.median([e["X"] for _, e in (jas or c["jasenet"])]))))
    if not ehdolla:
        ruudut = len({t for t, _ in sarja})
        for c in kl:
            jas = c["jasenet"]
            ala = float(np.percentile([e["ala_cm2"] for _, e in jas], 80))
            Y = float(np.median([e["Y"] for _, e in jas]))
            if not any(e.get("reunalla") for _, e in jas) or Y < TAKA_Y_MIN_CM or ala < MIN_ALA_CM2:
                continue
            esiintyma = len({t for t, _ in jas}) / ruudut
            ehdolla.append(dict(c, jas=jas, ajat={t for t, _ in jas if TAVOITE_ALKU_S <= t <= TAVOITE_LOPPU_S}, esiintyma=esiintyma, ala_cm2=ala,
                                pisteet=esiintyma * min(ala, KOKO_VERTAILU_CM2) / KOKO_VERTAILU_CM2, Y=Y, X=float(np.median([e["X"] for _, e in jas]))))
    if not ehdolla:
        return None
    paras = max(ehdolla, key=lambda c: (c["pisteet"], c["ala_cm2"]))
    kilpailijat = sorted((c for c in ehdolla if c["pisteet"] >= 0.5 * paras["pisteet"]), key=lambda c: (-c["pisteet"], -c["ala_cm2"]))
    harjat = []                                             # kilpailijat harjoittain: samalla linjalla olevat klusterit ovat saman harjan osia
    for c in kilpailijat:
        for ryhma in harjat:
            if abs(c["X"] - ryhma[0]["X"]) <= LINJA_DX_CM and abs(c["Y"] - ryhma[0]["Y"]) <= LINJA_DY_CM:
                ryhma.append(c)
                break
        else:
            harjat.append([c])
    # Harjanpaa on harjan heittajan puoleinen paa; takana samalla linjalla yhtaikaa nakyva on varsi tai kahvaosa (Video2:n harjoissa puunvarvinen ja yhta suuri kuin harjanpaa)
    ajat = paras["ajat"]
    paras = max([c for c in harjat[0] if c is paras or len(c["ajat"] & ajat) >= max(2, YHTAIKAA_OSUUS * min(len(c["ajat"]), len(ajat)))], key=lambda c: c["Y"])
    jas = paras["jas"]
    return dict(X=float(np.median([e.get("alin_X", e["X"]) for _, e in jas])), Y=float(np.median([e.get("alin_Y", e["Y"]) for _, e in jas])), keski_X=paras["X"], keski_Y=paras["Y"],
                ala_cm2=paras["ala_cm2"], esiintyma=paras["esiintyma"], pisteet=paras["pisteet"], ehdokkaita=len(harjat),
                pituus_cm=float(np.percentile([e["pituus_cm"] for _, e in jas], 80)), kulma=float(np.median([e["kulma"] for _, e in jas])), n=len(paras["jasenet"]),
                reunalla=bool(np.mean([bool(e.get("reunalla")) for _, e in jas]) >= 0.5))


def t_viivalle(X_alin, Y_alin, hack_X, hack_Y, etaisyys_cm):
    """Maalimerkin paikka kaukopaan T-viivalla (cm keskiviivasta, + = heittajan oikealle): suora viiva heiton lahtohackista (heittajan koordinaatisto: X + = oikealle, Y pitkin rataa lahipaan
    T-viivasta kaukopaahan, hack Y < 0) maalimerkin kautta kaukopaan T-viivalle (Y = etaisyys_cm). Maalimerkki (X_alin, Y_alin) on kaukopaan koordinaateissa (X' peilikuva, Y' = etaisyys
    kaukopaan T-viivaan heittajaa kohti), kuten konsensus palauttaa."""
    X_m, Y_m = -X_alin, etaisyys_cm - Y_alin
    return float(hack_X + (X_m - hack_X) * (etaisyys_cm - hack_Y) / (Y_m - hack_Y))


class MerkkiSeuranta:
    """Kaukopaan harjanpaiden jatkuva skannaus (SKANNAUS_VALI_S valein, molemmat puoliskot: heittosuunta vaihtuu) ja konsensus heiton ymparilta. Analyysi syottaa ruudut (syota);
    heiton lahdon jalkeen konsensus(puoli, f_alku) kertoo maalin kaukopaan koordinaateissa. Puskuri sailyttaa konsensusikkunan: ruudut heiton lahdosta -KONSENSUS_ALKU_S ... +KONSENSUS_LOPPU_S."""

    def __init__(self, mittarit, fps, vali_s=SKANNAUS_VALI_S):
        self.mittarit = dict(mittarit)                      # puoli -> Mittari (kuvan puolisko, jonka koordinaateissa ehdokkaat annetaan)
        self.fps = float(fps)
        self.vali = max(1, int(round(vali_s * self.fps)))
        self.puskuri = int(round((KONSENSUS_LOPPU_S - KONSENSUS_ALKU_S + 8.0) * self.fps))
        self.sarjat = {p: collections.deque() for p in self.mittarit}      # puoli -> [(ruutu, [kelvolliset ehdokkaat])]
        self._viim = None

    def syota(self, f, ruutu):
        """Skannaa ruudun, jos edellisesta skannauksesta on kulunut SKANNAUS_VALI_S (ruutunumerot videon ruutunopeudella). True, jos skannattiin."""
        if self._viim is not None and 0 <= f - self._viim < self.vali:
            return False
        self._viim = f
        for puoli, m in self.mittarit.items():
            s = self.sarjat[puoli]
            s.append((f, [e for e in ehdokkaat(m, ruutu) if e["kelvollinen"]]))
            while s and s[0][0] < f - self.puskuri:
                s.popleft()
        return True

    def konsensus(self, puoli, f_alku, f_asti=None):
        """Heiton (lahtoruutu f_alku) maali puoliskolla `puoli` (kaukopaa): merkki.konsensus puskurin ruuduista (f_asti: vain tahan asti), tai None."""
        sarja = [((f - f_alku) / self.fps, e) for f, e in self.sarjat[puoli] if f_asti is None or f <= f_asti]
        return konsensus(sarja)
