"""Videon todellinen aikaperusta: python aikaperusta.py tulokset vasen   (tarkistus: aikaperusta-testi radoille)

Video on 25 r/s -virrasta tallennettu 30 r/s:ksi. Pelkka "yksilollisten ruutujen jarjestysnumero k / 25" EI riita aikaperustaksi: lahdevirrasta
puuttuu ruutuja (n. 5-10 %, ruudun pudotus nakyy 30 r/s -videossa ylimaaraisena kopioruutuna ja seuraavassa ruudussa kaksinkertaisena siirtymana),
jolloin k/25 jaa jalkeen ja nopeus sekä kiihtyvyys heilahtelevat (askel 40 ms = 8 cm 2 m/s:n nopeudella).

Korjaus: jokaiselle radan naytteelle maaritetaan LAHDEVIRRAN RUUTUINDEKSI j (kokonaisluku) siirtymasta: kahden perakkaisen naytteen valinen Y-ero jaettuna
paikallisella nopeudella (nopeus 30 r/s -kellosta f/30, jonka tarkkuus riittaa kokonaisluvun arviointiin) kertoo kuinka monen lahderuudun verran kivi on
liikkunut. Aika t = j / 25 (+ vakio, joka sovitetaan seinakelloon f/30). Hitaassa vaiheessa (siirtyma < 3 cm/ruutu) kokonaisluku johdetaan
30 r/s -ruutunumeroiden hilasta (5 yksilollista ruutua 6 paikkaa kohti). Jos kaksi perakkaista naytetta on samassa paikassa vaikka kivi liikkuu
(kopioruutu, jota ei tunnistettu), jalkimmainen hylataan (j = NaN)."""
import csv
import os
import sys

import numpy as np

FPS_LAHDE = 25.0
FPS_ULOS = 30.0
V_MIN_SIIRTYMA_CM = 3.0       # alle taman siirtyma/ruutu: kokonaisluku hilasta, ei siirtymasta
KOPIO_MAX_CM = 1.0            # liikkuvan kiven perakkaiset naytteet taman lahempana = kaksoiskappale
DRIFT_RAJA = 2.5              # lahdeindeksin ja seinakellon (R f / 30) eron suurin sallittu muutos viime naytteisiin (DRIFT_N) nahden (askelta)
DRIFT_N = 8
ASKEL_APU_N = 6               # askelvirheen korjaus: edeltavien naytteiden maara, joiden paraabelista seuraavan naytteen paikka ennustetaan
ASKEL_Y_ALKU_CM = 150.0       # ... ja korjaus vain taman jalkeen (tyonnon aikana kiihtyvyys muuttuu nopeasti eika paraabeli ennusta)
ASKEL_ENNUSTE_MAX = 3.0       # ... ennustettavan naytteen suurin etaisyys edeltavista (lahderuutua) ja
ASKEL_IKKUNA_MAX = 14.0       # ... edeltavien naytteiden ikkunan suurin pituus (lahderuutua): muuten ennuste ei ole luotettava
ASKEL_RAJA = 0.35             # ... ja ennustejaannoksen sallittu poikkeama kokonaisesta askelmaarasta (osuus askelesta), jotta se luetaan askelvirheeksi


def paikallinen_nopeus(f, Y, h=8):
    """Y:n paikallinen nopeus [cm/s] 30 r/s -kellossa (f / 30): lineaarinen regressio +-h naytetta."""
    ts = np.asarray(f, float) / FPS_ULOS
    v = np.zeros(len(f))
    for i in range(len(f)):
        a, b = max(0, i - h), min(len(f), i + h + 1)
        v[i] = np.polyfit(ts[a:b], Y[a:b], 1)[0] if b - a >= 3 else 0.0
    return v


def _hila_dj(df, kopiot_ohitettu, R=None):
    """Lahderuutujen lukumaara kahden perakkaisen naytteen valilla, kun ruutunumeroiden ero on df (hidas vaihe: ei siirtymaa, josta lukea). Kun kopioruudut on jo jatetty pois
    naytteista (kopiot_ohitettu), ero 1 tai 2 on yksi lahderuutu (ero 2 = ohitettu kopioruutu; vain 25 r/s -virrassa); muuten hila (5 yksilollista ruutua 6 paikkaa kohti)."""
    R = FPS_LAHDE if R is None else R
    if kopiot_ohitettu and df <= 2 and R < 0.5 * (25.0 + FPS_ULOS):
        return 1
    return max(1, int(round(df * R / FPS_ULOS)))


def arvioi_lahdetaajuus(f, Y):
    """Radan oma lahderuutujen taajuus R (r/s): paikallinen nopeus (cm / s, 30 r/s -kello) jaettuna perakkaisten ruutujen (ero 1) siirtymalla (yksi lahderuutu = yksi askel) ja niiden mediaani.
    Videon 'lahdevirta' voi olla 25 r/s (kopioruutu joka kuudes, yksi askel / lahderuutu: 1. nettivideo) tai kayttaytya kuin 30 r/s (kopioruutu joka kuudes, mutta kivi liikkuu 30 askelta / s ja
    yksi ruutu kuudesta pudonnut: 2. nettivideo, jossa 25 r/s:n oletus antoi 17 % liian pienet nopeudet). Palauttaa 25 tai 30, jos arvio on lahella (+-10 %), muuten arvion rajattuna
    20...36; FPS_LAHDE, jos nopeita pareja on alle 15."""
    f = np.asarray(f, float)
    Y = np.asarray(Y, float)
    if len(f) < 20:
        return FPS_LAHDE
    v = paikallinen_nopeus(f, Y)
    df = np.diff(f)
    dY = np.diff(Y)
    vm = 0.5 * (v[1:] + v[:-1])
    ok = (df == 1) & (np.abs(vm) >= 3.0 * FPS_LAHDE) & (np.abs(dY) > 1.0) & (np.sign(dY) == np.sign(vm))
    if int(ok.sum()) < 15:
        return FPS_LAHDE
    r = float(np.median(vm[ok] / dY[ok]))
    for nimellinen in (25.0, 30.0):
        if abs(r - nimellinen) <= 0.1 * nimellinen:
            return nimellinen
    return float(np.clip(r, 20.0, 36.0))


def korjaa_askelvirheet(j, Y):
    """Lahderuutuindeksin j (kokonaisluvut, NaN = hylatty) askelvirheiden korjaus radan sileyden perusteella. Siirtymasta luettu indeksin kasvu (lahdeindeksit) menee joskus yhdella pieleen
    (esim. ruutuero 3 ja siirtyma 2,5 askelta: pyoristys alas, tai lahdevirran ruudun pudotus ja nopeuden virhe), ja virhe siirtaa kaikkia myohempia aikoja yhden lahderuudun (40 ms = 8 cm
    2 m/s:n nopeudella) ja nayttaa radassa askeleena, joka sotkee kiihtyvyyden muutosten tunnistuksen. Kivi liikkuu sileasti (kiihtyvyys <= 0,5 m/s2 = 0,3 cm / ruutu2), joten kunkin
    naytteen paikka ennustetaan edeltavien ASKEL_APU_N naytteen paraabelista (korjatuilla indekseilla): jos jaannos on lahes kokonainen maara askelia (+-ASKEL_RAJA askelta) ja seuraavat
    kaksi naytetta vahvistavat saman siirtyman, indeksit siirretaan taman verran. Yksittainen mittausvirhe (ei vahvistusta) ohitetaan. Palauttaa (j korjattuna, korjausten maara)."""
    j = np.array(j, float)
    Y = np.asarray(Y, float)
    idx = np.where(np.isfinite(j))[0]
    siirto, n_korj = 0.0, 0
    jk = j.copy()
    jk[idx] = j[idx]
    pos = ASKEL_APU_N
    while pos < len(idx):
        i = idx[pos]
        pre = idx[pos - ASKEL_APU_N:pos]
        j0 = float(jk[pre[-1]])
        c = np.polyfit(jk[pre] - j0, Y[pre], 2)
        askel = float(c[1])                                  # cm / lahderuutu viimeisen edeltavan naytteen kohdalla
        if abs(askel) < V_MIN_SIIRTYMA_CM or Y[i] < ASKEL_Y_ALKU_CM or j[i] + siirto - j0 > ASKEL_ENNUSTE_MAX or j0 - float(jk[pre[0]]) > ASKEL_IKKUNA_MAX:     # aukko (B puuttuu) tai harva ikkuna: ekstrapolointi ei kerro mitaan
            jk[i] = j[i] + siirto
            pos += 1
            continue
        r = Y[i] - float(np.polyval(c, j[i] + siirto - j0))
        k = round(r / askel)
        if k != 0 and abs(r - k * askel) <= ASKEL_RAJA * abs(askel) and abs(k) <= 2:
            ok = True
            for q in range(pos + 1, min(pos + 3, len(idx))):                 # vahvistus: seuraavat kaksi naytetta samalla siirtymalla (vanhalla ennusteella)
                rq = Y[idx[q]] - float(np.polyval(c, j[idx[q]] + siirto - j0))
                if abs(rq - k * askel) > ASKEL_RAJA * abs(askel) + 1.0:
                    ok = False
            if ok:
                siirto += k
                n_korj += 1
        jk[i] = j[i] + siirto
        pos += 1
    return jk, n_korj


def lahdeindeksit(f, Y, kopiot_ohitettu=False, R=None):
    """Lahdevirran ruutuindeksi j kullekin naytteelle (kokonaisluku, ankkurinaytteelle 0), NaN = hylatty kaksoiskappale; palauttaa (j, i_ankkuri).
    f = 30 r/s -ruutunumerot (kasvavat), Y = rataa pitkin mitattu paikka [cm]. Nopeassa vaiheessa (>= V_MIN_SIIRTYMA_CM / ruutu) indeksin kasvu
    luetaan siirtymasta, hitaassa vaiheessa (lepo, tyonnon alku) 30 r/s -ruutunumeroiden hilasta ankkurista (ensimmainen nopea naytte) laskien.
    kopiot_ohitettu: naytteet ovat vain yksilollisia ruutuja (kopioruudut ohitettu jo ennen seurantaa, esim. kaukopaan hidas rata): ruutuero 2 on yksi lahderuutu (25 r/s).
    R: lahderuutujen taajuus (r/s); oletus arvioi_lahdetaajuus(f, Y)."""
    f = np.asarray(f, float)
    Y = np.asarray(Y, float)
    R = arvioi_lahdetaajuus(f, Y) if R is None else float(R)
    n = len(f)
    v = paikallinen_nopeus(f, Y)
    nopea = np.abs(v) / R >= V_MIN_SIIRTYMA_CM
    if not nopea.any():
        if kopiot_ohitettu:
            return np.concatenate([[0.0], np.cumsum([_hila_dj(d, True, R) for d in np.diff(f)])]), 0
        return np.round((f - f[0]) * R / FPS_ULOS), 0
    ia = int(np.argmax(nopea))
    j = np.full(n, np.nan)
    j[ia] = 0.0
    for i in range(ia - 1, -1, -1):               # ennen ankkuria: hila
        j[i] = j[i + 1] - _hila_dj(f[i + 1] - f[i], True, R) if kopiot_ohitettu else -np.round((f[ia] - f[i]) * R / FPS_ULOS)
    viim = ia
    ajautuma = [float(j[ia] - R * f[ia] / FPS_ULOS)]
    for i in range(ia + 1, n):
        vm = 0.5 * (v[i] + v[viim])
        dY = Y[i] - Y[viim]
        if abs(vm) / R >= V_MIN_SIIRTYMA_CM:
            if abs(dY) < KOPIO_MAX_CM:            # liikkuu mutta ei siirry: kopioruutu
                continue
            dj = int(round(dY / (vm / R)))
            if dj < 1:
                dj = _hila_dj(f[i] - f[viim], kopiot_ohitettu, R)
        else:                                      # hidas: hilasta
            dj = _hila_dj(f[i] - f[viim], kopiot_ohitettu, R)
        j[i] = j[viim] + dj
        # seinakellon sidos: lahdeindeksin ja seinakellon (R f / 30) ero (ajautuma) pysyy rajoissa, joten siirtymasta luettu indeksi, joka poikkeaa
        # viime naytteiden ajautumasta yli DRIFT_RAJA askelta (iso aukko radassa, jossa kivi kiihtyy, tai nopeuden virhe), korvataan seinakellon mukaisella
        d_ref = float(np.median(ajautuma[-DRIFT_N:])) if len(ajautuma) else float(j[ia] - R * f[ia] / FPS_ULOS)
        if abs(j[i] - R * f[i] / FPS_ULOS - d_ref) > DRIFT_RAJA:
            j[i] = np.round(R * f[i] / FPS_ULOS + d_ref)
        ajautuma.append(float(j[i] - R * f[i] / FPS_ULOS))
        viim = i
    j, _ = korjaa_askelvirheet(j, Y)
    return j, ia


def aika(f, j, ia, R=None):
    """Aika [s] = j / R + vakio (R = lahderuutujen taajuus, oletus FPS_LAHDE); vakio valitaan niin, etta aika = f / 30 (videon aika) ankkurinaytteessa (liu'un alku, nopea vaihe alkaa)."""
    R = FPS_LAHDE if R is None else R
    f = np.asarray(f, float)
    j = np.asarray(j, float)
    c = f[ia] / FPS_ULOS - j[ia] / R
    return j / R + c


def korjaa_rata(f, Y):
    """Palauttaa (j, t) kaikille naytteille (kaksoiskappaleille NaN)."""
    R = arvioi_lahdetaajuus(f, Y)
    j, ia = lahdeindeksit(f, Y, R=R)
    return j, aika(f, j, ia, R)


class Aikajana:
    """Ruutunumero -> lahdevirran ruutuindeksi ja aika, yhden ikkunan (heiton) rataa pitkin sovitettuna. Sama aikajana kelpaa kaikille
    kohteille (kivi, kahva): aika on ruudun ominaisuus, ei kohteen."""

    def __init__(self, f, Y, kopiot_ohitettu=False, R=None):
        f = np.asarray(f, float)
        self.R = arvioi_lahdetaajuus(f, Y) if R is None else float(R)         # lahderuutujen taajuus (r/s): 25 tai 30 radan omasta liikkeesta
        j, ia = lahdeindeksit(f, Y, kopiot_ohitettu, self.R)
        ok = np.isfinite(j)
        self.f, self.j = f[ok], j[ok]
        self.c = float(f[ia] / FPS_ULOS - j[ia] / self.R)
        self.kopio = {}                           # kaksoiskappale-ruutu -> edellisen hyvaksytyn ruudun indeksi (sama lahderuutu)
        viim = None
        for fi, ji in zip(f, j):
            if np.isfinite(ji):
                viim = ji
            elif viim is not None:
                self.kopio[int(fi)] = viim

    def indeksi(self, f):
        """Lahderuudun indeksi ruudulle f (taulukko): mitatuille tarkka, muille interpoloitu hilasta (+-1)."""
        f = np.atleast_1d(np.asarray(f, float))
        out = np.empty(len(f))
        for i, fi in enumerate(f):
            if int(fi) in self.kopio:
                out[i] = self.kopio[int(fi)]
            else:
                k = np.searchsorted(self.f, fi)
                if k < len(self.f) and self.f[k] == fi:
                    out[i] = self.j[k]
                elif 0 < k < len(self.f):
                    out[i] = np.round(np.interp(fi, self.f, self.j))
                elif k == 0:
                    out[i] = self.j[0] + np.round((fi - self.f[0]) * self.R / FPS_ULOS)
                else:
                    out[i] = self.j[-1] + np.round((fi - self.f[-1]) * self.R / FPS_ULOS)
        return out

    def aika(self, f):
        return self.indeksi(f) / self.R + self.c

    def hyvaksytty(self, f):
        """Totuusarvot: ruutu on mitattu, ei kaksoiskappale."""
        return np.array([int(fi) not in self.kopio for fi in np.atleast_1d(f)])


def lataa_radat_ruuduilla(kansio, puoli, video=None):
    """Rata-CSV:n naytteet ruutunumeroineen. Ruutunumero haetaan vanhasta aikaleimasta (t_s) T_KARTTA:n kautta (aja_kivi.yksilolliset_ruudut)."""
    import aja_kivi as ak
    if video is None:
        raise ValueError("video tarvitaan ruutunumeroiden hakuun")
    ak.yksilolliset_ruudut(video, [(8, 22), (56, 66), (104, 114)], puoli)
    t2f = {round(t, 3): fr for fr, t in ak.T_KARTTA.items()}
    out = {}
    for r in csv.DictReader(open(os.path.join(kansio, f"radat_{puoli}.csv"))):
        d = out.setdefault(int(r["rata"]), {"f": [], "t_avain": [], "X": [], "Y": []})
        d["t_avain"].append(float(r["t_s"]))
        d["f"].append(t2f[round(float(r["t_s"]), 3)])
        d["X"].append(float(r["X_cm"]))
        d["Y"].append(float(r["Y_cm"]))
    return {k: {n: np.array(a) for n, a in d.items()} for k, d in out.items()}


if __name__ == "__main__":
    kansio, puoli = sys.argv[1], sys.argv[2]
    video = sys.argv[3]
    radat = lataa_radat_ruuduilla(kansio, puoli, video)
    for k, d in sorted(radat.items()):
        j, t = korjaa_rata(d["f"], d["Y"])
        ok = np.isfinite(j)
        print(f"heitto {k + 1}: {len(j)} naytetta, hylatty {int((~ok).sum())}, puuttuvia lahderuutuja {int(np.nanmax(j) - ok.sum() + 1)}, "
              f"j-jänne {np.nanmax(j) / FPS_LAHDE:.2f} s, seinakello {(d['f'][-1] - d['f'][0]) / FPS_ULOS:.2f} s")
        m = ok & (d["Y"] > 480)
        if m.sum() > 10:
            tt = t[m] - t[m].mean()
            r = d["Y"][m] - np.polyval(np.polyfit(tt, d["Y"][m], 2), tt)
            print(f"   vapaa lento (Y > 480): 2. asteen sovituksen rms {r.std():.2f} cm (n = {m.sum()})")
