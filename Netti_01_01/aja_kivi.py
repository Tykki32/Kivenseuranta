"""Kivitunnistus videolta: python aja_kivi.py VIDEO --puoli vasen --ulos tulokset [--ikkunat 12-22,58-66] [--saikeet 4]

Video on 25 r/s -virrasta tallennettu 30 r/s:ksi, jolloin joka kuudes ruutu on edellisen kopio. Kopioruudut ohitetaan ja
aika lasketaan yksilollisista ruuduista (t = ikkunan alku + k / 25), jotta nopeus ja kiihtyvyys eivat vaihtele ruudun
mukana. Jokaisesta ruudusta tunnistetaan kivet (kivi_netti.tunnista, paikka pehmealla siluettisovituksella) ja
havainnot ketjutetaan radoiksi.

Kirjoittaa <ulos>/kivet_<puoli>.csv (kaikki havainnot) ja <ulos>/radat_<puoli>.csv (ketjutetut radat)."""
import argparse
import csv
import os
from multiprocessing import Pool

import cv2
import numpy as np

import kivi_netti as kv

FPS_LAHDE = 25.0          # alkuperaisen virran ruutunopeus
KOPIO_KYNNYS = 15.0       # |ero|:n 99,9. persentiili (harmaasavy, puolisko taysresoluutiolla) alle taman = kopioruutu
                          # (kopiot 2-12, oikeat ruudut staattisessakin kuvassa >= 30, liikkuvassa kivessa ~200)

_G = {}


def _alusta(video, puoli, ulos):
    _G["cam"] = kv.Kamera(puoli, os.path.join(ulos, f"kalibrointi_{puoli}.json"), os.path.join(ulos, f"moodikuva_{puoli}.png"))
    _G["kivi"] = kv.Kivi()
    _G["cap"] = cv2.VideoCapture(video)


def _tyo(i):
    cam, kivi, cap = _G["cam"], _G["kivi"], _G["cap"]
    fr = cam.ruutu(cap, i)
    if fr is None:
        return i, []
    dets, (fg, g, h) = kv.tunnista(cam, kivi, fr)
    al = kv.alfa_kartta(cam, fr, g, h)
    out = []
    for d in dets:
        X, Y, e = kv.sovita_pehmea(cam, kivi, al, d["X"], d["Y"])
        out.append((X, Y, e, d["peitto"]))
    return i, out


def yksilolliset_ruudut(video, ikkunat, puoli="vasen"):
    """[(ruutu, k)] ikkunoiden yksilollisille ruuduille; k = yksilollisten ruutujen jarjestysnumero ikkunassa."""
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = []
    for t0, t1 in ikkunat:
        f0, f1 = int(t0 * fps), min(n, int(t1 * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
        prev, k = None, 0
        for f in range(f0, f1):
            ok, fr = cap.read()
            if not ok:
                break
            x, y, w, h = kv.kn.PUOLET[puoli]
            g = cv2.cvtColor(fr[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY).astype(np.int16)
            kopio = prev is not None and float(np.percentile(np.abs(g - prev), 99.9)) < KOPIO_KYNNYS
            prev = g
            if not kopio:
                out.append((f, k))
                k += 1
    cap.release()
    return out


def ketjuta(rows, portti_cm=18.0, portti_per_ruutu_cm=8.0, max_aukko=14, min_pisteet=10, min_siirtyma_cm=150.0):
    """rows: [(frame, k, X, Y, err)] -> radat: listat (t, X, Y, err), t = radan ensimmaisen ruudun f/30 + (k - k0)/25
    (yksilollisten ruutujen tasavali, ei kopioruutuja). Vakionopeusennuste: havainto liitetaan rataan, jonka ennustettu
    paikka on lahinna (portti portti_cm + portti_per_ruutu_cm per puuttuva ruutu); aukko enintaan max_aukko ruutua."""
    rows = sorted(rows)
    tracks = []

    def ennuste(tr, k):
        if len(tr) >= 3:
            a, b = tr[max(0, len(tr) - 4)], tr[-1]
            dk = max(b[1] - a[1], 1)
            vx, vy = (b[2] - a[2]) / dk, (b[3] - a[3]) / dk
        else:
            vx = vy = 0.0
        return tr[-1][2] + vx * (k - tr[-1][1]), tr[-1][3] + vy * (k - tr[-1][1])

    for f in sorted(set(r[0] for r in rows)):
        dets = [r for r in rows if r[0] == f]
        k = dets[0][1]
        free = [tr for tr in tracks if 0 < k - tr[-1][1] <= max_aukko]
        pairs = []
        for di, d in enumerate(dets):
            for ti, tr in enumerate(free):
                px, py = ennuste(tr, k)
                dist = np.hypot(d[2] - px, d[3] - py)
                if dist <= portti_cm + portti_per_ruutu_cm * (k - tr[-1][1]):
                    pairs.append((dist, di, ti))
        pairs.sort()
        ud, ut = set(), set()
        for _, di, ti in pairs:
            if di in ud or ti in ut:
                continue
            ud.add(di)
            ut.add(ti)
            free[ti].append(dets[di])
        for di, d in enumerate(dets):
            if di not in ud:
                tracks.append([d])
    keep = []
    for tr in tracks:
        if len(tr) >= min_pisteet and np.hypot(tr[-1][2] - tr[0][2], tr[-1][3] - tr[0][3]) >= min_siirtyma_cm:
            f0, k0 = tr[0][0], tr[0][1]
            keep.append([(f0 / 30.0 + (d[1] - k0) / FPS_LAHDE, d[2], d[3], d[4]) for d in tr])
    return keep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--puoli", default="vasen")
    ap.add_argument("--ulos", default="tulokset")
    ap.add_argument("--ikkunat", default="12-22,58-66,106-114", help="sekuntivalit, esim. 8-22,28-35")
    ap.add_argument("--saikeet", type=int, default=4)
    a = ap.parse_args()
    ikk = [tuple(float(x) for x in w.split("-")) for w in a.ikkunat.split(",")]
    ruudut = yksilolliset_ruudut(a.video, ikk, a.puoli)
    print(len(ruudut), "yksilollista ruutua")
    kmap = dict(ruudut)
    fps = 30.0
    with Pool(a.saikeet, initializer=_alusta, initargs=(a.video, a.puoli, a.ulos)) as pool:
        res = pool.map(_tyo, [f for f, _ in ruudut], chunksize=4)
    rows = []
    path = os.path.join(a.ulos, f"kivet_{a.puoli}.csv")
    with open(path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["frame", "yksilo_k", "t_s", "X_cm", "Y_cm", "err", "peitto"])
        for i, dets in res:
            for X, Y, e, p in dets:
                wr.writerow([i, kmap[i], f"{i / fps:.3f}", f"{X:.2f}", f"{Y:.2f}", f"{e:.3f}", f"{p:.3f}"])
                rows.append((i, kmap[i], X, Y, e))
    radat = ketjuta(rows)
    path2 = os.path.join(a.ulos, f"radat_{a.puoli}.csv")
    with open(path2, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["rata", "t_s", "X_cm", "Y_cm", "err"])
        for k, tr in enumerate(radat):
            for t, X, Y, e in tr:
                wr.writerow([k, f"{t:.3f}", f"{X:.2f}", f"{Y:.2f}", f"{e:.3f}"])
    print(path, len(rows), "havaintoa;", path2, len(radat), "rataa:")
    for k, tr in enumerate(radat):
        print(f"  rata {k}: t {tr[0][0]:.2f}-{tr[-1][0]:.2f} s, {len(tr)} pistetta, Y {tr[0][2]:.0f} -> {tr[-1][2]:.0f} cm, X {tr[0][1]:.0f} -> {tr[-1][1]:.0f} cm")
