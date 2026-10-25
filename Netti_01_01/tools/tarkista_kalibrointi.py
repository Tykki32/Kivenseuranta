"""Sopiiko kalibrointikansio (kalibrointi_*.json, moodikuva_*.png) videoon? Kuvan staattisten piirteiden siirtyma kansion moodikuvaan verrattuna.

    python tools/tarkista_kalibrointi.py video.mp4                      # kansio tulokset
    python tools/tarkista_kalibrointi.py video.mp4 --kansio tulokset_v2

Kalibrointi (homografia + kamera) ja taustakuva (moodikuva) kuuluvat yhteen kameran asentoon: jos kamera on toisessa asennossa tai zoomissa (toinen peli, toinen kentta, uusi pystytys), seuranta
kayttaa vaaraa taustaa (B-sovituksen haamureunat: rata harva ja kohiseva, hog-viivan kohdalla usean sentin askel) ja paikat ovat vinossa. Videon 2 kamera oli videon 1 kalibrointiin nahden
siirtynyt 3-13 px (ei pelkka siirto: zoomi/kierto, siirtyma vaihtelee kuvassa) -> paikkavirhe 6-17 cm ja nopeusvirhe ~1 %. Aja silloin `python tools/uusi_kalibrointi.py video.mp4 --ulos tulokset_x`.

Menetelma: videon mediaanikuva (n ruutua tasaisesti videon yli; ihmiset ja kivet poistuvat) ja kansion moodikuva (kaannetty takaisin puoliskon rajaukseksi) -> kuvan rikkaimmat staattiset ikkunat
(kameran_heilunta.valitse_ikkunat) -> ikkunakohtainen siirtyma vaihekorrelaatiolla. Tulostaa mediaanisiirtyman ja ikkunoiden suurimman siirtyman; paluuarvo 0 = sopii (kaikkien ikkunoiden siirtyma
alle --raja px), 1 = ei sovi."""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import kalibrointi_netti as kn  # noqa: E402
import kameran_heilunta as kh  # noqa: E402

VASTE_MIN = 0.15          # ikkunan vaihekorrelaatiovaste, jolla siirtyma luetaan luotettavaksi
RAJA_PX = 1.5             # alle taman: kalibrointi sopii (kameran oma heilunta on <= 0,65 px)


def moodikuva_puoliskoksi(kansio, puoli):
    """Kansion moodikuva (kaannetty 90 astetta vastapaivaan) takaisin puoliskon rajaukseksi, harmaana float32; None, jos tiedostoa ei ole. Puhdas tausta (hack) ei kelpaa vertailuun vain siksi, etta hackin alue eroaa."""
    polku = os.path.join(kansio, f"moodikuva_{puoli}.png")
    kuva = cv2.imread(polku)
    if kuva is None:
        return None
    return cv2.cvtColor(cv2.rotate(kuva, cv2.ROTATE_90_CLOCKWISE), cv2.COLOR_BGR2GRAY).astype(np.float32)


def siirtymat(video, kansio, n=24, alku_s=30.0, ref_video=None):
    """{puoli: dict(dx, dy, vaste, ikkunat=[(x, y, dx, dy, vaste)])} videon mediaanikuvan siirtymasta kansion moodikuvaan nahden (+dx: videon sisalto oikealle, +dy: alas).
    ref_video: valmiiksi lasketut videon mediaanikuvat (kameran_heilunta.referenssi), kun useaa kansiota verrataan samaan videoon."""
    if ref_video is None:
        ref_video = kh.referenssi(video, n=n, alku_s=alku_s)
    hann = cv2.createHanningWindow((kh.IKKUNA, kh.IKKUNA), cv2.CV_32F)
    out = {}
    for puoli in ("vasen", "oikea"):
        mood = moodikuva_puoliskoksi(kansio, puoli)
        if mood is None:
            out[puoli] = None
            continue
        v = ref_video[puoli]
        if v.shape != mood.shape:
            out[puoli] = dict(virhe=f"koko eri: video {v.shape[::-1]} px, moodikuva {mood.shape[::-1]} px")
            continue
        ikk = []
        for x, y in kh.valitse_ikkunat(mood, n=14):
            a = mood[y:y + kh.IKKUNA, x:x + kh.IKKUNA].copy()
            b = v[y:y + kh.IKKUNA, x:x + kh.IKKUNA].copy()
            (dx, dy), r = cv2.phaseCorrelate(a, b, hann)
            if r >= VASTE_MIN and abs(dx) < 40 and abs(dy) < 40:
                ikk.append((x, y, float(dx), float(dy), float(r)))
        out[puoli] = dict(ikkunat=ikk)
        if ikk:
            d = np.array([[i[2], i[3]] for i in ikk])
            out[puoli].update(dx=float(np.median(d[:, 0])), dy=float(np.median(d[:, 1])), suurin=float(np.max(np.hypot(d[:, 0], d[:, 1]))),
                              p90=float(np.percentile(np.hypot(d[:, 0], d[:, 1]), 90)), n=len(ikk))
    return out


def arvioi(tulos, raja=RAJA_PX):
    """(sopii: bool | None, teksti) yhden puoliskon tuloksesta (None = ei voitu arvioida)."""
    if tulos is None:
        return None, "moodikuvaa ei ole kansiossa"
    if "virhe" in tulos:
        return False, tulos["virhe"]
    if tulos.get("n", 0) < 4:
        return None, f"liian vahan vertailukelpoisia ikkunoita ({len(tulos['ikkunat'])}): kuva eroaa moodikuvasta muutenkin kuin siirtyman osalta"
    sopii = tulos["p90"] < raja
    return sopii, (f"mediaanisiirtyma ({tulos['dx']:+.2f}, {tulos['dy']:+.2f}) px, ikkunoiden suurin {tulos['suurin']:.2f} px (p90 {tulos['p90']:.2f}, {tulos['n']} ikkunaa) -> "
                   + ("sopii" if sopii else f"EI SOVI (yli {raja:g} px)"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--kansio", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tulokset"))
    ap.add_argument("--raja", type=float, default=RAJA_PX, help="suurin sallittu ikkunasiirtyma (p90, px)")
    ap.add_argument("-n", type=int, default=24, help="mediaanikuvaan otettavien ruutujen maara")
    a = ap.parse_args()
    tulos = siirtymat(a.video, a.kansio, a.n)
    kaikki = []
    for puoli in ("vasen", "oikea"):
        sopii, teksti = arvioi(tulos[puoli], a.raja)
        print(f"{puoli:6s}: {teksti}")
        kaikki.append(sopii)
    if any(s is False for s in kaikki):
        print(f"\nKalibrointi (kansio {a.kansio}) EI sovi tahan videoon: aja python tools/uusi_kalibrointi.py {a.video} --ulos <uusi kansio> ja kayta netti_live.py --kansio <uusi kansio>")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
