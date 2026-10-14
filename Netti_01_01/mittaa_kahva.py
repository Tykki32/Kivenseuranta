"""Kiven akseli (alareunan kaari) ja kahvan korkeus (etureunan kaari): python mittaa_kahva.py VIDEO tulokset vasen
Lukee tulokset/kahvaradat_<puoli>.csv (variseuranta), kirjoittaa kaariradat_<puoli>.csv ja tulostaa korkeuden tilastot."""
import csv
import sys

import cv2
import numpy as np

import aja_kivi as ak
import kivi_netti as kv

video, kansio, puoli = sys.argv[1], sys.argv[2], sys.argv[3]
cam = kv.Kamera(puoli, f"{kansio}/kalibrointi_{puoli}.json", f"{kansio}/moodikuva_{puoli}.png")
cap = cv2.VideoCapture(video)
ikk = [(8, 22), (56, 66), (104, 114)]
ak.yksilolliset_ruudut(video, ikk, puoli)
t2f = {round(t, 3): f for f, t in ak.T_KARTTA.items()}
rows = list(csv.DictReader(open(f"{kansio}/kahvaradat_{puoli}.csv")))
out, zs = [], {}
for r in rows:
    f = t2f.get(round(float(r["t_s"]), 3))
    if f is None:
        continue
    u, v = kv.kivimalli.project_3d(cam.K, cam.R, cam.t, np.array([[float(r["X_cm"]), float(r["Y_cm"]), kv.KAHVA_Z_CM]]))
    fr = cam.ruutu(cap, f)
    try:
        H, G = kv.kaaripisteet(cam, fr, (u[0], v[0]), r["vari"])
        X, Y, z, e1, e2 = kv.sovita_akseli_ja_korkeus(cam, H, G, (u[0], v[0]))
    except Exception:
        continue
    out.append((r["rata"], r["t_s"], X, Y, z, e1, e2))
    if float(r["Y_cm"]) > 250:
        zs.setdefault(r["rata"], []).append(z)
with open(f"{kansio}/kaariradat_{puoli}.csv", "w", newline="") as fh:
    wr = csv.writer(fh)
    wr.writerow(["rata", "t_s", "X_akseli_cm", "Y_akseli_cm", "kahvan_taso_cm", "alakaaren_rms_px", "kahvan_rms_px"])
    for o in out:
        wr.writerow([o[0], o[1], f"{o[2]:.2f}", f"{o[3]:.2f}", f"{o[4]:.2f}", f"{o[5]:.2f}", f"{o[6]:.2f}"])
for k, z in zs.items():
    z = np.array(z)
    print(f"heitto {int(k) + 1}: kahvan taso (Y > 250 cm, r_kahva = {kv.R_KAHVA_CM} cm) mediaani {np.median(z):.1f} cm, IQR {np.percentile(z, 25):.1f}-{np.percentile(z, 75):.1f}, n = {len(z)}")
