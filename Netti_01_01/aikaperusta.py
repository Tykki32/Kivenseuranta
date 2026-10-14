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


def paikallinen_nopeus(f, Y, h=8):
    """Y:n paikallinen nopeus [cm/s] 30 r/s -kellossa (f / 30): lineaarinen regressio +-h naytetta."""
    ts = np.asarray(f, float) / FPS_ULOS
    v = np.zeros(len(f))
    for i in range(len(f)):
        a, b = max(0, i - h), min(len(f), i + h + 1)
        v[i] = np.polyfit(ts[a:b], Y[a:b], 1)[0] if b - a >= 3 else 0.0
    return v


def lahdeindeksit(f, Y):
    """Lahdevirran ruutuindeksi j kullekin naytteelle (kokonaisluku, ankkurinaytteelle 0), NaN = hylatty kaksoiskappale; palauttaa (j, i_ankkuri).
    f = 30 r/s -ruutunumerot (kasvavat), Y = rataa pitkin mitattu paikka [cm]. Nopeassa vaiheessa (>= V_MIN_SIIRTYMA_CM / ruutu) indeksin kasvu
    luetaan siirtymasta, hitaassa vaiheessa (lepo, tyonnon alku) 30 r/s -ruutunumeroiden hilasta ankkurista (ensimmainen nopea naytte) laskien."""
    f = np.asarray(f, float)
    Y = np.asarray(Y, float)
    n = len(f)
    v = paikallinen_nopeus(f, Y)
    nopea = np.abs(v) / FPS_LAHDE >= V_MIN_SIIRTYMA_CM
    if not nopea.any():
        return np.round((f - f[0]) * FPS_LAHDE / FPS_ULOS), 0
    ia = int(np.argmax(nopea))
    j = np.full(n, np.nan)
    j[ia] = 0.0
    for i in range(ia):                           # ennen ankkuria: hila
        j[i] = -np.round((f[ia] - f[i]) * FPS_LAHDE / FPS_ULOS)
    viim = ia
    for i in range(ia + 1, n):
        vm = 0.5 * (v[i] + v[viim])
        dY = Y[i] - Y[viim]
        if abs(vm) / FPS_LAHDE >= V_MIN_SIIRTYMA_CM:
            if abs(dY) < KOPIO_MAX_CM:            # liikkuu mutta ei siirry: kopioruutu
                continue
            dj = int(round(dY / (vm / FPS_LAHDE)))
            if dj < 1:
                dj = max(1, int(round((f[i] - f[viim]) * FPS_LAHDE / FPS_ULOS)))
        else:                                      # hidas: hilasta
            dj = max(1, int(round((f[i] - f[viim]) * FPS_LAHDE / FPS_ULOS)))
        j[i] = j[viim] + dj
        viim = i
    return j, ia


def aika(f, j, ia):
    """Aika [s] = j / 25 + vakio; vakio valitaan niin, etta aika = f / 30 (videon aika) ankkurinaytteessa (liu'un alku, nopea vaihe alkaa)."""
    f = np.asarray(f, float)
    j = np.asarray(j, float)
    c = f[ia] / FPS_ULOS - j[ia] / FPS_LAHDE
    return j / FPS_LAHDE + c


def korjaa_rata(f, Y):
    """Palauttaa (j, t) kaikille naytteille (kaksoiskappaleille NaN)."""
    j, ia = lahdeindeksit(f, Y)
    return j, aika(f, j, ia)


class Aikajana:
    """Ruutunumero -> lahdevirran ruutuindeksi ja aika, yhden ikkunan (heiton) rataa pitkin sovitettuna. Sama aikajana kelpaa kaikille
    kohteille (kivi, kahva): aika on ruudun ominaisuus, ei kohteen."""

    def __init__(self, f, Y):
        f = np.asarray(f, float)
        j, ia = lahdeindeksit(f, Y)
        ok = np.isfinite(j)
        self.f, self.j = f[ok], j[ok]
        self.c = float(f[ia] / FPS_ULOS - j[ia] / FPS_LAHDE)
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
                    out[i] = self.j[0] + np.round((fi - self.f[0]) * FPS_LAHDE / FPS_ULOS)
                else:
                    out[i] = self.j[-1] + np.round((fi - self.f[-1]) * FPS_LAHDE / FPS_ULOS)
        return out

    def aika(self, f):
        return self.indeksi(f) / FPS_LAHDE + self.c

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
