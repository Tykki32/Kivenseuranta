"""Ruutujen luku videosta (tiedosto tai virta) hakemalla usealla lukijalla rinnakkain.

Haku virrasta (cv2.VideoCapture.set(CAP_PROP_POS_FRAMES)) on hidas: jokainen haku avaa yhteyden uudelleen ja dekoodaa palan alusta haluttuun ruutuun asti, ja yksi lukija ehtii vain
odottaa verkkoa tai dekooderia. Kun haettavat ruudut jaetaan usealle lukijalle (kullakin oma VideoCapture ja oma saie), viive ja dekoodaus menevat paallekkain: 24 hakua hitaalla HLS-virralla
(80 ms viive, 100 Mbit/s) 11,6 s -> 5,0 s. Tulokset ovat samat kuin yhdella lukijalla (sama ruutu samasta paikasta), vain luvun jarjestys ja kesto muuttuvat.

    from videon_luku import tiedot, lue_ruudut
    fps, ruutuja = tiedot(video)
    kuvat = lue_ruudut(video, [0, 150, 300], kasittele=lambda kuva: kuva[190:1080, 0:960])      # [kuva | None, ...] samassa jarjestyksessa kuin paikat
"""
from concurrent.futures import ThreadPoolExecutor

import cv2

SAIKEITA = 4                 # lukijoita rinnakkain: 3 saietta nopeutti hakuja 2,3-kertaisesti, 4 ei enaa merkittavasti (verkon viive ja dekoodaus menevat paallekkain, kaista on yhteinen)


def tiedot(video):
    """(fps, ruutuja): videon ruutunopeus (30, ellei tiedossa) ja ruutujen maara (virralla arvio kestosta)."""
    cap = cv2.VideoCapture(video)
    try:
        return cap.get(cv2.CAP_PROP_FPS) or 30.0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()


def lue_ruudut(video, paikat, kasittele=None, saikeita=None):
    """Ruudut paikoista (0-pohjaiset ruutunumerot) [kasittele(ruutu) | ruutu | None]: tulos on samassa jarjestyksessa kuin paikat, None = ruutua ei saatu luetuksi.
    kasittele: ruudun kasittely jo lukusaikeessa (rajaus, harmaasavy), ettei kokonaisia ruutuja kerry muistiin; sen palautusarvo on tulos. Kullakin lukijalla on oma VideoCapture ja oma
    perakkainen osa paikoista; saikeita = 1 lukee yhdella (kuten ennen), None = SAIKEITA."""
    saikeita = SAIKEITA if saikeita is None else saikeita
    paikat = [int(k) for k in paikat]
    tulos = [None] * len(paikat)
    if not paikat:
        return tulos
    m = max(1, min(int(saikeita), len(paikat)))
    osat = [range(i * len(paikat) // m, (i + 1) * len(paikat) // m) for i in range(m)]

    def tyo(osa):
        cap = cv2.VideoCapture(video)
        try:
            for i in osa:
                cap.set(cv2.CAP_PROP_POS_FRAMES, paikat[i])
                ok, kuva = cap.read()
                if ok:
                    tulos[i] = kasittele(kuva) if kasittele is not None else kuva
        finally:
            cap.release()

    if m == 1:
        tyo(osat[0])
        return tulos
    pooli = ThreadPoolExecutor(m)
    try:
        for f in [pooli.submit(tyo, osa) for osa in osat]:
            f.result()
    finally:
        pooli.shutdown(wait=True)
    return tulos
