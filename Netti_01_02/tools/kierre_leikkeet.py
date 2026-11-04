"""Kierteen opetusdata: topdown- ja alkuperaisleikkeet kivesta heittopaassa irroituksen jalkeen.

Jokaisesta valitusta heitosta (radat.pkl: netti_live.py ... --radat radat.pkl) poimitaan NAYTTEITA_HEITOSSA ruutua tasavalein ajassa irroituksen jalkeiselta matkalta kuvan reunaan asti
(aikaperusta.Aikajana: lahdevirran kopioruudut pois, todellinen aika). Matka alkaa ensimmaisesta ruudusta, jossa kasi ei enaa ole kahvan ylapuolella (kasi_pikselit <= KASI_ALKU_PX); sen
jalkeen ruutuja ei suodateta, koska palkki nakyy usein kaden vierellakin, ja merkitsija voi merkita ruudun "ei nay" tai "kasi kahvassa". Kustakin ruudusta tehdaan kaksi leiketta:

  * topdown: ylhaalta katsottu nakyma kameramallilla (netti_geom.Puoli.maa_kuvaan, linssikorjaus mukana) kahvan palkin tasolle (z = BAR_Z_CM) kiven akselin ymparilta, ikkuna
    TD_IKKUNA_CM x TD_IKKUNA_CM, TD_PX_CM px / cm. Suunta on sama kuin alkuperaisessa videossa (kivi liikkuu alaspain), mutta ilman perspektiivia; kuva on ihan oikea ylhaalta
    katsottu kuva (ei peilattu), joten vastapaivaan kiertyva palkki kiertyy vastapaivaan myos leikkeessa.
  * alkuperainen: videoruudun leike (2 * RAW_PUOLI px) suurennettuna RAW_SKAALA-kertaiseksi (kuutiointerpolointi).

Palkki on vaakasuora ja korkeudella z = BAR_Z_CM jaan pinnasta (muovilevy z0 = 15 cm + palkki h ~ 5 cm: tulokset*/kahvamalli_asetukset_*.json, kierto_muoto_*.json), joten topdown-kuvassa
palkin suunta on suoraan kiven kierrekulma. Pystysuorat rakenteet (kahvan jalka, kiven kylki) ja liikesumeus venyvat topdown-kuvassa radan suuntaan (kamera katsoo rataa viistosti:
Y-suunnassa on noin 2,5 kertaa vahemman pikseleita kuin X-suunnassa), joten viereen annetaan myos alkuperainen leike; homografia H (topdown-pikseli -> alkuperaisen leikkeen pikseli
samalla z-tasolla) yhdistaa kuvat, ja merkintasivulla viiva nakyy molemmissa.

Tuloste: <ulos>/s/<nimi>_h<heitto>.jpg (yksi kuvakooste per heitto: ylarivilla topdown-leikkeet TD_N px, alarivilla alkuperaiset RAW_N px, sarake = ruutu), <ulos>/naytteet.json
(kaikki naytteet ja koneen arviot; ei mene merkintasivulle) ja valinnaisesti <ulos>/png/<nimi>/{td,raaka}/<id>.png (--png). Merkintasivu: tools/luo_kahvamerkitsin.py.

Kaytto (viisi videota, 3 heittoa kustakin; luo_kahvamerkitsin.py poimii niista satunnaiset heitot ja sekoittaa ruudut):
    python tools/kierre_leikkeet.py --ulos leikkeet --heittoja 3 \\
        --ajo Matarmaa Matarmaa.mp4 r20_Matarmaa/radat.pkl r20_Matarmaa/h.csv --ajo Ahrenberg Ahrenberg.mp4 r20_Ahrenberg/radat.pkl r20_Ahrenberg/h.csv ...
Kalibrointikansio valitaan videon nimesta (KANSIOT) tai --kansio-valitsimella: --kansio nimi=polku."""
import argparse
import csv
import json
import os
import pickle
import random
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import aikaperusta as aikap  # noqa: E402
import netti_geom as ng  # noqa: E402

BAR_Z_CM = 20.0                  # kahvan palkin taso (cm kalibrointitasosta): muovilevy z0 = 15 + palkin korkeus h ~ 5 (kierto_muoto: keltainen 5,3, punainen 4,8)
LEVY_Z_CM = 15.0                 # muovilevyn taso (kahvamalli_asetukset: z0_cm)
LEVY_R_CM = 10.0                 # muovilevyn sade (A_cm 10,1, B_cm 9,9)
TD_PX_CM = 3.5                   # topdown-leikkeen resoluutio (px / cm); alkuperaisen kuvan X-suunnan resoluutio on ~1,4 px / cm, Y-suunnan ~0,6 px / cm
TD_IKKUNA_CM = 32.0              # topdown-leikkeen sivu (cm): palkki (puolikas ~6 cm + sumeus) ja muovilevy mahtuvat, kasi nakyy reunalla
TD_N = int(round(TD_IKKUNA_CM * TD_PX_CM))     # 112 px (jaollinen 8:lla: JPEG-lohkot osuvat kuvakoosteen ruutujen rajoille)
RAW_PUOLI = 28                   # alkuperaisen leikkeen puolikas (px) -> 56 x 56 px
RAW_SKAALA = 4                   # suurennus (kuutiointerpolointi)
RAW_N = 2 * RAW_PUOLI * RAW_SKAALA             # 224 px
IRROITUS_MARGINAALI_CM = 60.0    # ruutuja vasta irroituksen jalkeen: kivi on liikkunut taman verran irroituspaikasta
Y_MAX_CM = 700.0                 # kuvan reuna / hog-alue: kivi ei saa leikkautua pois, kahva on viela riittavan iso
KASI_ROI_X, KASI_ROI_Y = 38, (36, 2)      # kasi-ROI muovilevyn keskipisteen ymparilla (px): kuten kierto_netti.py (DY_MAX, DX_MIN..DX_MAX), kuvassa palkin alue levyn ylapuolella
NAYTTEITA_HEITOSSA = 4           # ruutuja per heitto (tasavalein ajassa): neljasta ruudusta saa kulman muutosnopeuden, vaikka yksi ruutu jaisi ilman viivaa
KASI_ALKU_PX = 250               # ruudut otetaan ensimmaisesta ruudusta alkaen, jossa kasi-mitta (kasi_pikselit) on enintaan tama: silloin kasi on yleensa jo irti kahvasta
MIN_RUUTUJA = 6                  # heitto otetaan mukaan vain, jos tuollaisia ruutuja on vahintaan tama monta (6 ruutua = vahintaan 0,24 s)
JPEG_LAATU = 92                  # kuvakoosteen laatu (4:4:4, ei varien alinayttetysta: ohut palkki ja sen reunan vari sailyvat)
KANSIOT = {"Video1": "tulokset", "Video2": "tulokset_nettivideo2", "Saarelainen": "tulokset_nettivideo2", "Matarmaa": "tulokset_nettivideo3", "Ahrenberg": "tulokset_nettivideo3"}


# ---------------------------------------------------------------------------------------------------------------------
# GEOMETRIA: topdown- ja alkuperaisleike
# ---------------------------------------------------------------------------------------------------------------------
def suunta_merkit(P, X, Y, z):
    """Topdown-ruudukon akselien suunta: kuvan oikealle (j kasvaa) vastaa maakoordinaatin X:n suuntaa sx = +-1 ja alas (i kasvaa) Y:n suuntaa sy = +-1, jotta leike on samansuuntainen kuin
    alkuperainen kuva (kamerasta riippuva). Tarkistaa, etta kameran kuvaus on akselien suuntainen (muuten palauttaa None)."""
    d = 5.0
    p0 = P.maa_kuvaan([X], [Y], z)[0]
    px = P.maa_kuvaan([X + d], [Y], z)[0] - p0
    py = P.maa_kuvaan([X], [Y + d], z)[0] - p0
    if abs(px[0]) < 3 * abs(px[1]) or abs(py[1]) < 3 * abs(py[0]):
        return None
    return (1.0 if px[0] > 0 else -1.0), (1.0 if py[1] > 0 else -1.0)


def topdown_kartat(P, X, Y, z=BAR_Z_CM, px_cm=TD_PX_CM, ikkuna_cm=TD_IKKUNA_CM):
    """(mx, my, A, N): cv2.remap-kartat topdown-leikkeelle sekä A = (sx, sy) (j: X:n suunta, i: Y:n suunta) ja leikkeen koko N. Leikkeen keskipiste (N - 1) / 2 on maapiste (X, Y, z)."""
    N = int(round(ikkuna_cm * px_cm))
    sg = suunta_merkit(P, X, Y, z)
    if sg is None:
        raise ValueError("kamera ei ole akselien suuntainen kiven kohdalla")
    sx, sy = sg
    j, i = np.meshgrid(np.arange(N), np.arange(N))
    Xg = X + sx * (j - (N - 1) / 2.0) / px_cm
    Yg = Y + sy * (i - (N - 1) / 2.0) / px_cm
    uv = P.maa_kuvaan(Xg.ravel(), Yg.ravel(), z)
    return uv[:, 0].reshape(N, N).astype(np.float32), uv[:, 1].reshape(N, N).astype(np.float32), (sx, sy), N


def topdown_leike(frame, P, X, Y, z=BAR_Z_CM):
    mx, my, A, N = topdown_kartat(P, X, Y, z)
    return cv2.remap(frame, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE), A


def raaka_leike(frame, P, X, Y, z=BAR_Z_CM):
    """Alkuperaisen ruudun leike kiven akselin (X, Y, z) ymparilta suurennettuna. Palauttaa (kuva, x0, y0): x0, y0 = leikkeen vasen ylakulma videoruudussa (px)."""
    u, v = P.maa_kuvaan([X], [Y], z)[0]
    x0, y0 = int(round(u)) - RAW_PUOLI, int(round(v)) - RAW_PUOLI
    h, w = frame.shape[:2]
    xs = np.clip(np.arange(x0, x0 + 2 * RAW_PUOLI), 0, w - 1)
    ys = np.clip(np.arange(y0, y0 + 2 * RAW_PUOLI), 0, h - 1)
    pieni = frame[np.ix_(ys, xs)]
    return cv2.resize(pieni, None, fx=RAW_SKAALA, fy=RAW_SKAALA, interpolation=cv2.INTER_CUBIC), x0, y0


def homografia_td_raaka(P, X, Y, x0, y0, z=BAR_Z_CM):
    """3 x 3 -homografia, joka vie topdown-leikkeen pikselin (j, i; pikselin keskipiste kokonaisluvussa) alkuperaisen leikkeen pikseliksi (samalla tavalla) tasolla z. Sovitetaan 11 x 11
    -pisteen ruudukkoon tarkalla kameramallilla (linssikorjaus mukana; 56 px:n leikkeessa ero homografiaan on alle 0,05 px). Palauttaa (H, suurin jaannosvirhe leikkeen pikseleina)."""
    mx, my, A, N = topdown_kartat(P, X, Y, z)
    g = np.linspace(0, N - 1, 11).round().astype(int)
    jj, ii = np.meshgrid(g, g)
    src = np.column_stack([jj.ravel(), ii.ravel()]).astype(np.float64)
    u, v = mx[ii.ravel(), jj.ravel()].astype(np.float64), my[ii.ravel(), jj.ravel()].astype(np.float64)
    dst = np.column_stack([(u - x0 + 0.5) * RAW_SKAALA - 0.5, (v - y0 + 0.5) * RAW_SKAALA - 0.5])
    H, _ = cv2.findHomography(src, dst, 0)
    res = np.hypot(*(cv2.perspectiveTransform(src[None], H)[0] - dst).T)
    return H, float(res.max())


# ---------------------------------------------------------------------------------------------------------------------
# KASI-MITTA (tiedoksi): iho, hanska, hiha muovilevyn ulkopuolella kahvan palkin alueella, kuten kierto_netti.keraa_kehykset
# ---------------------------------------------------------------------------------------------------------------------
def ei_valia(lab, vari):
    """Pikselit, jotka eivat ole jaata, muovia eivatka kiveä: iho, hanska, hiha (kahva_malli.havainto: ei_valia)."""
    L = lab[..., 0] * (100.0 / 255.0)
    a, b = lab[..., 1] - 128.0, lab[..., 2] - 128.0
    if vari == "keltainen":
        return ((a > 10) & (b > 4)) | (L < 30)
    return (L < 25) | ((L > 52) & (b > 8) & (a > 8))


def kasi_pikselit(frame, P, X, Y, vari):
    """Kasi-ROI:n ei-valia-pikseleiden maara muovilevyn soikion ulkopuolella (kierto_netti.KASI_RAJA_PX = 60 on koneen hylkaysraja). Tallennetaan nayteluetteloon tiedoksi; ei valitse ruutuja:
    sama luku syntyy hihasta, joka ei peita palkkia, ja sormista, jotka peittavat."""
    th = np.linspace(0.0, 2.0 * np.pi, 90, endpoint=False)
    poly = P.maa_kuvaan(X + LEVY_R_CM * np.cos(th), Y + LEVY_R_CM * np.sin(th), LEVY_Z_CM)
    uc, vc = P.maa_kuvaan([X], [Y], LEVY_Z_CM)[0]
    x0, x1 = int(round(uc)) - KASI_ROI_X, int(round(uc)) + KASI_ROI_X
    y0, y1 = int(round(vc)) - KASI_ROI_Y[0], int(round(vc)) - KASI_ROI_Y[1]
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    if x1 <= x0 or y1 <= y0:
        return 10 ** 6
    lab = cv2.cvtColor(np.ascontiguousarray(frame[y0:y1, x0:x1]), cv2.COLOR_BGR2LAB).astype(np.float32)
    levy = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillConvexPoly(levy, np.round((poly - np.array([x0, y0]) + 0.5) * 16).astype(np.int32), 1, cv2.LINE_8, 4)
    levy = cv2.dilate(levy, np.ones((3, 3), np.uint8)) > 0
    return int((ei_valia(lab, vari) & ~levy).sum())


# ---------------------------------------------------------------------------------------------------------------------
# VALINTA
# ---------------------------------------------------------------------------------------------------------------------
def heiton_ehdokasruudut(h, fps_lahde=25.0, fps_ulos=30.0):
    """Heiton ruudut irroituksen jalkeen kuvan reunaan: [(f, X, Y, t_s)] yksilollisista ruuduista. Paikat ja aikaperusta ovat samat kuin viivaestimaatin syotteessa (viivat_syote =
    kelvolliset B-paikat; netti_seuranta.valmistele_rata: aikaperusta.Aikajana, lahdevirran kopioruudut pois, todellinen aika)."""
    v = h.get("viivat")
    syote = h.get("viivat_syote") or []
    if not v or v.get("irr_Y") is None or len(syote) < 30:
        return []
    syote = sorted((int(f), float(X), float(Y)) for f, X, Y in syote)
    f = np.array([p[0] for p in syote], float)
    X = np.array([p[1] for p in syote], float)
    Y = np.array([p[2] for p in syote], float)
    aikap.FPS_LAHDE, aikap.FPS_ULOS = float(fps_lahde), float(fps_ulos)
    aj = aikap.Aikajana(f, Y)
    uniikit = [i for i, fi in enumerate(f) if int(fi) not in aj.kopio]
    t = aj.aika(f[uniikit])
    irr = float(v["irr_Y"])
    return [(int(f[i]), float(X[i]), float(Y[i]), float(t[k])) for k, i in enumerate(uniikit) if irr + IRROITUS_MARGINAALI_CM <= Y[i] <= Y_MAX_CM]


def tasavalein(ruudut, n):
    """n tasavalista ruutua listasta (jarjestyksessa); ensimmainen ja viimeinen mukana."""
    if len(ruudut) <= n:
        return list(ruudut)
    idx = np.unique(np.round(np.linspace(0, len(ruudut) - 1, n)).astype(int))
    return [ruudut[i] for i in idx]


def kerros(h, r):
    """Heiton kerros valintaa varten: (kierron nopeus: |omega| < 70 / 70-120 / > 120 astetta/s, kierron suunta, koneen varmuus, vari); vari vaihtelee nopeimmin, joten vuorotellen otettaessa
    punaisia ja keltaisia tulee yhta monta. None, jos koneella ei ole arviota."""
    try:
        om = float(r["omega_deg_s"])
    except (KeyError, TypeError, ValueError):
        return None
    nop = 0 if abs(om) < 70 else (1 if abs(om) < 120 else 2)
    varma = str(r.get("kierre_varma", "")).strip().lower() in ("1", "true", "kylla", "k", "x")
    return (nop, om > 0, varma, h["vari"])


def valitse_heitot(rivit, heitot, n, siemen=1):
    """n heittoa kerrostetusti (kerros(): nopeus x kierron suunta x koneen varmuus x vari): kerrokset kaydaan satunnaisessa jarjestyksessa (kiinteä siemen) ja jokaisesta otetaan vuorotellen
    satunnainen heitto, joten pienikin n kattaa eri kerroksia (ei aina hitaita myotapaivaan kiertavia). Palauttaa heitto-indeksit (0-pohjainen)."""
    rnd = random.Random(siemen)
    kerrokset = {}
    for i, (h, r) in enumerate(zip(heitot, rivit)):
        k = kerros(h, r)
        if k is not None:
            kerrokset.setdefault(k, []).append(i)
    for k in kerrokset.values():
        rnd.shuffle(k)
    avaimet = sorted(kerrokset)
    rnd.shuffle(avaimet)
    valitut = []
    while len(valitut) < n and any(kerrokset.values()):
        for a in avaimet:
            if kerrokset[a] and len(valitut) < n:
                valitut.append(kerrokset[a].pop())
    return valitut


# ---------------------------------------------------------------------------------------------------------------------
# KUVAKOOSTE JA NAYTTEET
# ---------------------------------------------------------------------------------------------------------------------
def kuvakooste(td_lista, raaka_lista):
    """Heiton kuvakooste: ylarivilla topdown-leikkeet (TD_N px), alarivilla alkuperaiset (RAW_N px); ruudun k sarake on k * RAW_N (topdown vie sarakkeesta vain TD_N px)."""
    n = len(td_lista)
    arkki = np.zeros((TD_N + RAW_N, n * RAW_N, 3), np.uint8)
    for k, (td, raa) in enumerate(zip(td_lista, raaka_lista)):
        arkki[:TD_N, k * RAW_N:k * RAW_N + TD_N] = td
        arkki[TD_N:, k * RAW_N:(k + 1) * RAW_N] = raa
    return arkki


def kirjoita_jpeg(polku, kuva):
    ok, b = cv2.imencode(".jpg", kuva, [cv2.IMWRITE_JPEG_QUALITY, JPEG_LAATU, cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444])
    if not ok:
        raise IOError("JPEG-koodaus epaonnistui: " + polku)
    with open(polku, "wb") as fh:
        fh.write(b.tobytes())
    return len(b)


def kasi_vapaat_ruudut(cap, h, P):
    """Heiton ehdokasruudut ensimmaisesta ruudusta alkaen, jossa kasi-mitta on enintaan KASI_ALKU_PX: [(f, X, Y, t, kasi_px)]. Kasi seuraa kiveä irroituksen jalkeenkin (tracker-irroitus on
    kinemaattinen, kasi aukeaa vasta myohemmin), joten pelkka irroituspaikka ei riita; mitta ei kuitenkaan erota hihaa, joka ei peita palkkia, sormista, jotka peittavat, joten alkua ei
    tiukenneta: merkitsija voi merkita ruudun \"ei nay\" tai \"kasi kahvassa\"."""
    ehdokkaat = heiton_ehdokasruudut(h)
    mitat = []
    for f, X, Y, t in ehdokkaat:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, fr = cap.read()
        mitat.append(kasi_pikselit(fr, P, X, Y, h["vari"]) if ok else None)
    alku = next((j for j, m in enumerate(mitat) if m is not None and m <= KASI_ALKU_PX), None)
    if alku is None:
        return []
    return [(f, X, Y, t, m) for (f, X, Y, t), m in zip(ehdokkaat[alku:], mitat[alku:]) if m is not None]


def tee_heitto(cap, nimi, i, h, r, P, naytteita=NAYTTEITA_HEITOSSA):
    """Heiton (0-pohjainen indeksi i) naytteet: (luettelo, td-leikkeet, raaka-leikkeet). Luettelo = [nayte-sanakirja]; tyhja, jos kasi-vapaita ruutuja on liian vahan (MIN_RUUTUJA)."""
    ruudut = kasi_vapaat_ruudut(cap, h, P)
    if len(ruudut) < MIN_RUUTUJA:
        return [], [], []
    valitut = tasavalein(ruudut, naytteita)
    v = h.get("kierto") or {}
    luettelo, tds, raas = [], [], []
    for f, X, Y, t, kasi in valitut:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, fr = cap.read()
        if not ok:
            continue
        td, A = topdown_leike(fr, P, X, Y)
        raa, x0, y0 = raaka_leike(fr, P, X, Y)
        H, hvirhe = homografia_td_raaka(P, X, Y, x0, y0)
        tds.append(td)
        raas.append(raa)
        luettelo.append(dict(id=f"{nimi}_h{i + 1}_f{f}", nimi=nimi, heitto=i + 1, vari=h["vari"], puoli=h["puoli"], kivi_nro=int(h["kivi_nro"]), f=f, t_s=round(t, 4), X=round(X, 2), Y=round(Y, 2),
                             kasi_px=kasi, z_cm=BAR_Z_CM, px_cm=TD_PX_CM, A=[A[0], A[1]], raaka_x0=x0, raaka_y0=y0, H=[round(float(x), 7) for x in H.ravel()], H_virhe_px=round(hvirhe, 4),
                             kone=dict(omega_deg_s=r.get("omega_deg_s"), omega_sd=v.get("omega_sd_deg_s"), suunta_p=v.get("p_suunta"), varma=bool(v.get("varma")), kierre_lahde=r.get("kierre_lahde"))))
    for k, n_ in enumerate(luettelo):
        n_["k"], n_["n"] = k, len(luettelo)
        n_["sprite"] = f"s/{nimi}_h{i + 1}.jpg"
    return luettelo, tds, raas


def tee_naytteet(nimi, video, kansio, radat, csv_polku, ulos, heittoja=3, heitot=None, siemen=1, naytteita=NAYTTEITA_HEITOSSA, png=False, tulosta=print):
    """Poimii heitot ja ruudut videolta, kirjoittaa kuvakoosteet (<ulos>/s) ja palauttaa nayteluettelon. heitot = heittonumerot (1-pohjaisia): ohittaa kerrostetun valinnan."""
    d = pickle.load(open(radat, "rb"))
    rivit = list(csv.DictReader(open(csv_polku, encoding="utf-8")))
    G = ng.Geometria(kansio)
    os.makedirs(os.path.join(ulos, "s"), exist_ok=True)
    jarjestys = [int(x) - 1 for x in heitot] if heitot else valitse_heitot(rivit, d, heittoja + 12, siemen)          # varaa ehdokkaita puuttuvien tilalle
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise IOError("videota ei voi avata: " + video)
    kaikki, kasitelty = [], 0
    for i in jarjestys:
        if kasitelty >= heittoja and not heitot:
            break
        h, r = d[i], rivit[i]
        luettelo, tds, raas = tee_heitto(cap, nimi, i, h, r, G[h["puoli"]], naytteita)
        if not luettelo:
            tulosta(f"  {nimi} heitto {i + 1}: liian vahan kasi-vapaita ruutuja, ohitetaan")
            continue
        koko = kirjoita_jpeg(os.path.join(ulos, luettelo[0]["sprite"]), kuvakooste(tds, raas))
        if png:
            for n_, td, raa in zip(luettelo, tds, raas):
                for alikansio, kuva in (("td", td), ("raaka", raa)):
                    os.makedirs(os.path.join(ulos, "png", nimi, alikansio), exist_ok=True)
                    cv2.imwrite(os.path.join(ulos, "png", nimi, alikansio, n_["id"] + ".png"), kuva)
        kaikki.extend(luettelo)
        kasitelty += 1
        tulosta(f"  {nimi} heitto {i + 1} ({h['vari']} {h['kivi_nro']}, {h['puoli']}): {len(luettelo)} ruutua, kuvakooste {koko // 1024} kB, koneen omega {r.get('omega_deg_s')}")
    return kaikki


def yhdista_luettelo(polku, uudet):
    """Kirjoittaa nayteluettelon (<ulos>/naytteet.json): uudet naytteet korvaavat saman videon aiemmat, muiden videoiden naytteet sailyvat."""
    vanhat = []
    if os.path.exists(polku):
        vanhat = json.load(open(polku, encoding="utf-8"))
    nimet = {n["nimi"] for n in uudet}
    kaikki = [n for n in vanhat if n["nimi"] not in nimet] + list(uudet)
    with open(polku, "w", encoding="utf-8") as fh:
        json.dump(kaikki, fh, ensure_ascii=False, indent=0)
    return kaikki


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ajo", nargs=4, action="append", metavar=("NIMI", "VIDEO", "RADAT", "CSV"), required=True,
                    help="video (Video1 / Video2 / Saarelainen / Matarmaa / Ahrenberg), videotiedosto, netti_live.py --radat -pickle ja saman ajon heitto-CSV; voi toistaa")
    ap.add_argument("--ulos", required=True)
    ap.add_argument("--kansio", action="append", default=[], metavar="NIMI=POLKU", help="kalibrointikansio videolle (oletus KANSIOT)")
    ap.add_argument("--heittoja", type=int, default=3, help="heittoja per video (kerrostettu satunnainen valinta)")
    ap.add_argument("--naytteita", type=int, default=NAYTTEITA_HEITOSSA, help="ruutuja per heitto")
    ap.add_argument("--heitot", help="heittonumerot pilkuilla (yksi video; ohittaa kerrostetun valinnan)")
    ap.add_argument("--siemen", type=int, default=1)
    ap.add_argument("--png", action="store_true", help="kirjoita lisaksi yksittaiset PNG-leikkeet (<ulos>/png)")
    a = ap.parse_args()
    kans = dict(k.split("=", 1) for k in a.kansio)
    juuri = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    os.makedirs(a.ulos, exist_ok=True)
    for nimi, video, radat, csv_polku in a.ajo:
        kansio = kans.get(nimi) or os.path.join(juuri, KANSIOT[nimi])
        uudet = tee_naytteet(nimi, video, kansio, radat, csv_polku, a.ulos, a.heittoja, a.heitot.split(",") if a.heitot else None, a.siemen, a.naytteita, a.png)
        kaikki = yhdista_luettelo(os.path.join(a.ulos, "naytteet.json"), uudet)
        print(f"{nimi}: {len(uudet)} naytetta; luettelossa yhteensa {len(kaikki)}")


if __name__ == "__main__":
    main()
