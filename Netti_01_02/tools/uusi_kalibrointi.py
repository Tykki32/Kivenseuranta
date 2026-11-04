"""Kalibrointikansio uudelle videolle (toinen kamera-asento, kentta tai peli): kameran kalibrointi ja taustakuva videosta, kivi- ja kahvamallit vanhasta kansiosta.

    python tools/uusi_kalibrointi.py uusi_video.mp4 --ulos tulokset_uusi              # kalibroi kummankin puoliskon ja tarkistaa tuloksen
    python netti_live.py --lahde uusi_video.mp4 --kansio tulokset_uusi ...            # seuranta uudella kalibroinnilla

Kalibrointi (kalibrointi_netti.py: homografia + kamera renkaista ja viivoista) ja moodikuva (taustakuva, jota B-sovitus vertaa ruutuun) kuuluvat yhteen kameran asentoon; toinen asento antaa
paikkavirheen (videolla 2: 6-17 cm), ~1 % nopeusvirheen ja B-sovitukseen haamureunoja (rata harva ja kohiseva, hog-viivan kohdalla 3-8 cm:n askel, irroitus lukittuu siihen). Tarkista, tarvitaanko
uutta kalibrointia: tools/tarkista_kalibrointi.py video.mp4 --kansio tulokset. Kivi-, kahva- ja kierremallit (kivimalli_*, kahvamalli_asetukset_*, kierto_muoto_*) kopioidaan --pohja-kansiosta
(oletus tulokset): ne ovat kiven ja kahvan muotoja senttimetreina, eivat kameran asentoa. Hackin haamukivi (netti_tausta.py) EI kopioidu: jos kivi lepaa hackissa yli puolet
moodikuvan ruuduista, aja `python netti_tausta.py VIDEO --tyhja-vasen ... --ulos <kansio>`."""
import argparse
import glob
import os
import shutil
import subprocess
import sys

import cv2

KOHDE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, KOHDE)

MALLIT = ("kivimalli_*.json", "kahvamalli_asetukset_*.json", "kierto_muoto_*.json")


def kopioi_mallit(pohja, ulos):
    n = 0
    os.makedirs(ulos, exist_ok=True)
    for kuvio in MALLIT:
        for p in glob.glob(os.path.join(pohja, kuvio)):
            shutil.copy2(p, os.path.join(ulos, os.path.basename(p)))
            n += 1
    return n


def kalibroi_video(video, ulos, pohja=None, koko_video=0, tulosta=print):
    """Kalibroi video kansioon ulos (mallit kopioidaan pohjasta, oletus tulokset); palauttaa 0, jos onnistui ja tulos sopii videoon (tools/tarkista_kalibrointi.py). Sama kuin komentorivi."""
    pohja = os.path.join(KOHDE, "tulokset") if pohja is None else pohja
    n = kopioi_mallit(pohja, ulos)
    tulosta(f"kopioitu {n} mallitiedostoa kansiosta {pohja}")
    komento = [sys.executable, os.path.join(KOHDE, "kalibrointi_netti.py"), video, "--ulos", ulos]
    r = subprocess.run(komento + (["--koko-video", str(koko_video)] if koko_video else []), cwd=KOHDE)
    if r.returncode and not koko_video:                        # videon alussa voi olla ihmisia viivoilla: moodikuva koko videon yli
        tulosta("kalibrointi epaonnistui ensimmaisen 2 minuutin moodikuvalla, yritetaan moodikuvalla koko videon yli")
        r = subprocess.run(komento + ["--koko-video", "40"], cwd=KOHDE)
    if r.returncode:
        tulosta("kalibrointi epaonnistui (renkaita tai viivoja ei loytynyt moodikuvasta): katso kalibrointi_netti.py:n tuloste")
        return r.returncode
    tulosta("\ntarkistus:")
    return subprocess.run([sys.executable, os.path.join(KOHDE, "tools", "tarkista_kalibrointi.py"), video, "--kansio", ulos], cwd=KOHDE).returncode


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--ulos", required=True, help="uusi kalibrointikansio")
    ap.add_argument("--pohja", default=os.path.join(KOHDE, "tulokset"), help="kansio, josta kivi- ja kahvamallit kopioidaan")
    ap.add_argument("--koko-video", type=int, default=0, metavar="N", help="moodikuva N ruudusta koko videon yli (oletus: ensin videon 2 ensimmaista minuuttia, sen epaonnistuessa 40 ruudusta koko videon yli)")
    a = ap.parse_args()
    if not cv2.VideoCapture(a.video).isOpened():
        print("videota ei voi avata:", a.video)
        return 1
    r = kalibroi_video(a.video, a.ulos, a.pohja, a.koko_video)
    if r == 0:
        print(f"\nvalmis: python netti_live.py --lahde {a.video} --kansio {a.ulos} ...")
    return r


if __name__ == "__main__":
    sys.exit(main())
