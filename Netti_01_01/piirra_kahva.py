"""Kahvan muovin ellipsi valituissa ruuduissa isona, selitteen kanssa:
python piirra_kahva.py VIDEO tulokset vasen --ruudut 0:236,1:467 [--ulos tulokset/kahva_ruudut.jpg] [--kirjaimet] [--vanha] [--sarakkeita 2]

--ruudut    lista rata:Y_cm (rata = 0, 1, 2 = heitto 1, 2, 3; Y = lahin seurannan piste, kahvaradat_<puoli>.csv)
--kirjaimet merkitsee ellipsin osat kirjaimilla (A etureuna, B ylakarki, C takareuna, D alakarki)
--vanha     piirtaa lisaksi ruutukohtaisen vapaan ellipsin (silhuetin etureuna + karjet + sisalle-rajoite) ohuella valkoisella
Ellipsi on kiintean kokoinen malli (kahva_malli.py): koko tulokset/kahvamalli_koko_<puoli>.json (mittaa_kahva_malli.py --opi).
Vari: magenta = malli, syaani = muovin silhuetti (varikynnys, vertailuksi), valkoinen = vapaa ellipsi (--vanha)."""
import argparse
import csv
import json
import os

import cv2
import numpy as np

import aja_kivi as ak
import kahva_malli as km
import kivi_netti as kv

ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("kansio")
ap.add_argument("puoli")
ap.add_argument("--ruudut", required=True)
ap.add_argument("--ulos", default=None)
ap.add_argument("--suurennus", type=int, default=10)
ap.add_argument("--sarakkeita", type=int, default=2)
ap.add_argument("--kirjaimet", action="store_true")
ap.add_argument("--vanha", action="store_true")
a = ap.parse_args()

cam = kv.Kamera(a.puoli, f"{a.kansio}/kalibrointi_{a.puoli}.json", f"{a.kansio}/moodikuva_{a.puoli}.png")
kev = km.Kevyt(cam)
cap = cv2.VideoCapture(a.video)
ak.yksilolliset_ruudut(a.video, [(8, 22), (56, 66), (104, 114)], a.puoli)
t2f = {round(t, 3): f for f, t in ak.T_KARTTA.items()}
rows = list(csv.DictReader(open(f"{a.kansio}/kahvaradat_{a.puoli}.csv")))
A, B = km.A_CM, km.B_CM
json_polku = f"{a.kansio}/kahvamalli_koko_{a.puoli}.json"
if os.path.exists(json_polku):
    d = json.load(open(json_polku))
    A, B = d["A_cm"], d["B_cm"]
S = a.suurennus
FONT = cv2.FONT_HERSHEY_SIMPLEX


def ruutu(rata, Y):
    r = min((r for r in rows if r["rata"] == rata), key=lambda r: abs(float(r["Y_cm"]) - Y))
    f = t2f[round(float(r["t_s"]), 3)]
    fr = cam.ruutu(cap, f)
    u, v = kv.kivimalli.project_3d(cam.K, cam.R, cam.t, np.array([[float(r["X_cm"]), float(r["Y_cm"]), 15.0]]))
    uv0 = (float(u[0]), float(v[0]))
    crop, mode, x0, y0 = km.leike(cam, fr, uv0)
    m = km.sovita(kev, crop, mode, x0, y0, r["vari"], uv0, A, B)
    big = cv2.resize(crop, None, fx=S, fy=S, interpolation=cv2.INTER_CUBIC)
    P = lambda pt: (int(round((pt[0] - x0 + 0.5) * S)), int(round((pt[1] - y0 + 0.5) * S)))
    try:
        s = kv.kahva_silhuetti(fr, uv0, vari=r["vari"], tausta=cam.mode)
    except Exception:
        s = None
    if s is not None:
        C = s.contour
        for i in range(len(C) - 1):
            cv2.line(big, P(C[i]), P(C[i + 1]), (255, 255, 0), 1, cv2.LINE_AA)
        if a.vanha:
            E = kv.sovita_kuvaellipsi(s, painot=(1.0, 10.0, 10.0))["E"]
            for i in range(len(E) - 1):
                cv2.line(big, P(E[i]), P(E[i + 1]), (255, 255, 255), 1, cv2.LINE_AA)
    teksti = f"heitto {int(rata) + 1} ({r['vari']}) Y={float(r['Y_cm']):.0f} cm"
    if m is None:
        cv2.putText(big, teksti + "  (mallia ei sovitettu)", (8, 26), FONT, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        return big
    poly = np.vstack([m["poly"], m["poly"][:1]])
    for i in range(len(poly) - 1):
        cv2.line(big, P(poly[i]), P(poly[i + 1]), (255, 0, 255), 2, cv2.LINE_AA)
    if a.kirjaimet:
        def merkki(pt, kirjain, dx, dy):
            p = P(pt)
            q = (p[0] + dx, p[1] + dy)
            cv2.line(big, p, q, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.circle(big, q, 17, (255, 255, 255), -1)
            cv2.circle(big, q, 17, (0, 0, 0), 2)
            cv2.putText(big, kirjain, (q[0] - 9, q[1] + 8), FONT, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        E = m["poly"]
        merkki(E[np.argmax(E[:, 0])], "A", 45, 0)
        merkki(E[np.argmin(E[:, 1])], "B", 0, -40)
        merkki(E[np.argmin(E[:, 0])], "C", -45, 0)
        merkki(E[np.argmax(E[:, 1])], "D", 0, 40)
    cv2.putText(big, teksti, (8, 26), FONT, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(big, f"malli A = {A:.1f} cm, B = {B:.1f} cm, z = {km.Z0_CM:.0f} cm, sovituksen RMS {m['rms']:.3f}", (8, 2 * km.H * S - 12), FONT, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
    return big


tiles = [ruutu(*arg.split(":")[:1], float(arg.split(":")[1])) for arg in a.ruudut.split(",")]
cols = max(1, min(a.sarakkeita, len(tiles)))
rimg = [np.hstack(tiles[i:i + cols] + [np.zeros_like(tiles[0])] * (cols - len(tiles[i:i + cols]))) for i in range(0, len(tiles), cols)]
img = np.vstack(rimg)
sel = np.full((70, img.shape[1], 3), 255, np.uint8)
cv2.putText(sel, "magenta = kiintean kokoinen malli (soikio kiven akselilla)   syaani = muovin silhuetti (varikynnys)", (10, 26), FONT, 0.6, (60, 60, 60), 2, cv2.LINE_AA)
cv2.putText(sel, "valkoinen = vapaa ellipsi (vanha menetelma, --vanha)   kasi ja hanska eivat vaikuta sovitukseen", (10, 54), FONT, 0.6, (60, 60, 60), 2, cv2.LINE_AA)
img = np.vstack([img, sel])
ulos = a.ulos or f"{a.kansio}/kahva_ruudut.jpg"
cv2.imwrite(ulos, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
print("kirjoitettu", ulos, img.shape)
