"""Puhdas taustakuva live-seurantaan: python netti_tausta.py VIDEO --tyhja-vasen 18.5-22.5,66.5-76 [--ulos tulokset]

Vanha moodikuva (kalibrointi_netti.moodikuva) on pikselikohtainen MEDIAANI koko videolta. Hackin kohdalla se sisaltaa "haamun": kivi, joka lepaa hackissa
(0-8, 23-47 ja 83-101 s) ja heittajan jalat ovat samassa paikassa yli puolet ajasta, joten niista tulee taustaa. Silloin lepaavan kiven alfa-kartta
(kivi_netti.alfa_kartta) ja sen myota tarkka sovitus (kivi_tarkka) epaonnistuvat juuri hackissa, jossa kivi on levossa ja heitto alkaa.
Tilastollinen korjaus ei auta (haamu on yli puolessa naytteista), joten hackin alue korvataan hetkella, jolloin hack on TYHJA (--tyhja-hack T0-T1, s):
mediaani naiden ruutujen hack-alueelta (--tyhja-vasen / --tyhja-oikea). Muu kuva on vanhaa moodikuvaa. Kirjoittaa tulokset/moodikuva_<puoli>_puhdas.png (kierretty 90 astetta vastapaivaan
kuten moodikuva_<puoli>.png); kamera lataa sen automaattisesti, jos se on olemassa (kaytettava)."""
import argparse
import json
import os

import cv2
import numpy as np

import kalibrointi_netti as kn
import kivimalli

HACK_ALUE_CM = ((-95.0, 95.0), (-440.0, -170.0))      # X, Y: hackin ymparisto (kivi levossa, heittaja jalkineineen)


def hack_laatikko(kansio, puoli, marg=14):
    d = json.load(open(os.path.join(kansio, f"kalibrointi_{puoli}.json")))
    K, R, t = np.array(d["K"], float), np.array(d["R"], float), np.array(d["t"], float)
    P = np.array([(x, y, z) for x in HACK_ALUE_CM[0] for y in HACK_ALUE_CM[1] for z in (0.0, 30.0)])
    u, v = kivimalli.project_3d(K, R, t, P)
    return int(u.min()) - marg, int(v.min()) - marg, int(u.max()) + marg, int(v.max()) + marg


def tyhja_mediaani(video, puoli, ikkunat, vali_s=0.5):
    """Mediaani ruuduista, joita otetaan ikkunoista [(t0, t1), ...] valein vali_s."""
    x, y, w, h = kn.PUOLET[puoli]
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    kuvat = []
    for t0, t1 in ikkunat:
        t = t0
        while t <= t1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
            ok, fr = cap.read()
            if not ok:
                break
            kuvat.append(fr[y:y + h, x:x + w])
            t += vali_s
    cap.release()
    med = np.median(np.stack(kuvat, axis=0), axis=0).astype(np.uint8)
    return cv2.rotate(med, cv2.ROTATE_90_COUNTERCLOCKWISE), len(kuvat)


def rakenna(video, kansio, puoli, ikkunat):
    pohja_p = os.path.join(kansio, f"moodikuva_{puoli}.png")
    if os.path.exists(pohja_p):
        pohja = cv2.imread(pohja_p)
    else:
        pohja, _ = kn.moodikuva(video, puoli)
    tyhja_kuva, n = tyhja_mediaani(video, puoli, ikkunat)
    x0, y0, x1, y1 = hack_laatikko(kansio, puoli)
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, pohja.shape[1]), min(y1, pohja.shape[0])
    # pehmea reuna: sekoitus 10 px:n yli, ettei reunaan jaa saumaa
    m = np.zeros(pohja.shape[:2], np.float32)
    m[y0:y1, x0:x1] = 1.0
    m = cv2.GaussianBlur(m, (0, 0), 4.0)
    m[y0 + 12:y1 - 12, x0 + 12:x1 - 12] = 1.0
    out = (pohja.astype(np.float32) * (1 - m[..., None]) + tyhja_kuva.astype(np.float32) * m[..., None]).round().astype(np.uint8)
    return out, n, (x0, y0, x1, y1)


def nimi(kansio, puoli):
    return os.path.join(kansio, f"moodikuva_{puoli}_puhdas.png")


def kaytettava(kansio, puoli):
    """Puhdas tausta jos olemassa, muuten vanha moodikuva."""
    p = nimi(kansio, puoli)
    return p if os.path.exists(p) else os.path.join(kansio, f"moodikuva_{puoli}.png")


def _ikkunat(txt):
    return [tuple(float(v) for v in w.split("-")) for w in txt.split(",")]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--ulos", default="tulokset")
    ap.add_argument("--tyhja-vasen", default="18.5-22.5,66.5-76", help="aikavalit (s), jolloin vasemman paan hack on tyhja, T0-T1,T0-T1 (tyhja = ei korjausta)")
    ap.add_argument("--tyhja-oikea", default="", help="sama oikealle (oletus: ei korjausta; oikeassa paassa on tassa videossa ihmisia koko ajan)")
    a = ap.parse_args()
    for puoli, txt in (("vasen", a.tyhja_vasen), ("oikea", a.tyhja_oikea)):
        if not txt:
            if os.path.exists(nimi(a.ulos, puoli)):
                os.remove(nimi(a.ulos, puoli))
            continue
        kuva, n, box = rakenna(a.video, a.ulos, puoli, _ikkunat(txt))
        cv2.imwrite(nimi(a.ulos, puoli), kuva)
        print(nimi(a.ulos, puoli), f"({n} ruutua, korvattu alue {box})")
