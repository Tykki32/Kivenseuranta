"""Kiven tarkka siluetti valituissa ruuduissa isona:
python piirra_kivet.py VIDEO tulokset vasen --ruudut 0:150,0:400,1:300,2:520 [--ulos tulokset/kivet_ruudut.jpg] [--kahva] [--alfa] [--sarakkeita 2] [--suurennus 8]

--ruudut     lista rata:Y_cm (rata = 0, 1, 2 = heitto 1, 2, 3; Y = lahin radan piste, radat_<puoli>.csv)
--kahva      piirtaa lisaksi kahvan muovin ympyran (kahva_malli.py; magenta)
--alfa       lisaa viereen graniitin havainnon (alfa) ja kelvolliset pikselit: punertava = ei huomioitu (kasi, muovi, jaan viivat, kiilto,
             kiven ylapinta)
Syaani = tumman graniittirungon siluetti (kiintea malli kivi_tarkka.Runko, kuvakohtainen akseli (X, Y)), keltainen = rungon ylarengas
(ylapinta ei ole graniittia), magenta = kahvan muovi. Koko on sama joka ruudussa: vain paikka sovitetaan."""
import argparse
import csv

import cv2
import numpy as np

import aja_kivi as ak
import kahva_malli as km
import kivi_netti as kv
import kivi_tarkka as kt

ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("kansio")
ap.add_argument("puoli")
ap.add_argument("--ruudut", required=True)
ap.add_argument("--ulos", default=None)
ap.add_argument("--suurennus", type=int, default=8)
ap.add_argument("--sarakkeita", type=int, default=1)
ap.add_argument("--kahva", action="store_true")
ap.add_argument("--alfa", action="store_true")
a = ap.parse_args()

cam = kv.Kamera(a.puoli, f"{a.kansio}/kalibrointi_{a.puoli}.json", f"{a.kansio}/moodikuva_{a.puoli}.png")
runko = kt.Runko.lataa(a.kansio, a.puoli)
cap = cv2.VideoCapture(a.video)
ak.yksilolliset_ruudut(a.video, [(8, 22), (56, 66), (104, 114)], a.puoli)
t2f = {round(t, 3): f for f, t in ak.T_KARTTA.items()}
rata_rows = list(csv.DictReader(open(f"{a.kansio}/radat_{a.puoli}.csv")))
kahva_rows = list(csv.DictReader(open(f"{a.kansio}/kahvaradat_{a.puoli}.csv"))) if a.kahva else []
ast = km.lataa_asetukset(a.kansio, a.puoli)
kev = km.Kevyt(cam)
S = a.suurennus
FONT = cv2.FONT_HERSHEY_SIMPLEX


def ruutu(rata, Y):
    r = min((r for r in rata_rows if int(r["rata"]) == int(rata)), key=lambda r: abs(float(r["Y_cm"]) - Y))
    f = t2f[round(float(r["t_s"]), 3)]
    fr = cam.ruutu(cap, f)
    alfa, ok = kt.havainto(cam, fr)
    s = kt.sovita(cam, runko, alfa, ok, float(r["X_cm"]), float(r["Y_cm"]))
    hull = runko.hull(cam, s["X"], s["Y"])
    top = runko.ylapinta(cam, s["X"], s["Y"])
    x0, y0, w, h = kt.ikkuna(cam, hull, marg=14)
    P = lambda pts: np.round((np.asarray(pts) - np.array([x0, y0]) + 0.5) * S * 16).astype(np.int32)
    big = cv2.resize(fr[y0:y0 + h, x0:x0 + w], None, fx=S, fy=S, interpolation=cv2.INTER_CUBIC)
    cv2.polylines(big, [P(hull)], True, (255, 255, 0), 1, cv2.LINE_AA, 4)
    cv2.polylines(big, [P(top)], True, (0, 200, 255), 1, cv2.LINE_AA, 4)
    if a.kahva:
        q = min((k for k in kahva_rows if int(k["rata"]) == int(rata)), key=lambda k: abs(float(k["t_s"]) - float(r["t_s"])))
        u, v = kv.kivimalli.project_3d(cam.K, cam.R, cam.t, np.array([[s["X"], s["Y"], ast["z0_cm"]]]))
        uv0 = (float(u[0]), float(v[0]))
        crop, mode, cx0, cy0 = km.leike(cam, fr, uv0)
        m = km.sovita_asetuksilla(kev, crop, mode, cx0, cy0, q["vari"], uv0, ast)
        if m is not None:
            cv2.polylines(big, [P(m["poly"])], True, (255, 0, 255), 1, cv2.LINE_AA, 4)
    teksti = f"heitto {int(rata) + 1} Y={s['Y']:.0f} cm X={s['X']:.1f}  rms {s['rms']:.3f}  kelv. {s['kelvolliset']:.2f}"
    cv2.putText(big, teksti, (6, 22), FONT, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
    out = big
    if a.alfa:
        c, wt = kt.malli_kuva(cam, runko, (x0, y0, w, h), s["X"], s["Y"])
        o = alfa[y0:y0 + h, x0:x0 + w]
        w_ok = ok[y0:y0 + h, x0:x0 + w] * wt
        al = cv2.cvtColor((cv2.resize(o, None, fx=S, fy=S, interpolation=cv2.INTER_NEAREST) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        wb = cv2.resize(w_ok, None, fx=S, fy=S, interpolation=cv2.INTER_NEAREST) > 0.5
        al[~wb] = (0.5 * al[~wb] + 0.5 * np.array([60, 60, 200])).astype(np.uint8)
        cv2.polylines(al, [P(hull)], True, (255, 255, 0), 1, cv2.LINE_AA, 4)
        out = np.hstack([big, al])
    return out


tiles = [ruutu(arg.split(":")[0], float(arg.split(":")[1])) for arg in a.ruudut.split(",")]
cols = max(1, min(a.sarakkeita, len(tiles)))
hh = max(t.shape[0] for t in tiles)
ww = max(t.shape[1] for t in tiles)
tiles = [np.pad(t, ((0, hh - t.shape[0]), (0, ww - t.shape[1]), (0, 0)), constant_values=255) for t in tiles]
rimg = [np.hstack(tiles[i:i + cols] + [np.full_like(tiles[0], 255)] * (cols - len(tiles[i:i + cols]))) for i in range(0, len(tiles), cols)]
img = np.vstack(rimg)
sel = np.full((50, img.shape[1], 3), 255, np.uint8)
cv2.putText(sel, f"syaani = tumman graniittirungon siluetti (R {runko.r[5]:.1f}-{runko.R_max:.1f} cm, H {runko.H:.1f} cm, kiintea malli), keltainen = rungon ylarengas"
            + ("), magenta = kahvan muovi" if a.kahva else ""), (10, 30), FONT, 0.6, (60, 60, 60), 2, cv2.LINE_AA)
img = np.vstack([img, sel])
ulos = a.ulos or f"{a.kansio}/kivet_ruudut.jpg"
cv2.imwrite(ulos, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
print("kirjoitettu", ulos, img.shape)
