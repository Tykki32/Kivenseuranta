"""Kahvan kupu (muovi) joka 3. ruudusta vapaasti liikkuvalle kivelle (Y > 300 cm): python mittaa_kupu.py VIDEO tulokset vasen
Lukee kaariradat_<puoli>.csv (akseli alareunan kaaresta) ja kahvaradat_<puoli>.csv (vari), kirjoittaa kupuradat_<puoli>.csv."""
import csv
import sys

import cv2
import numpy as np

import aja_kivi as ak
import kivi_netti as kv

video, kansio, puoli = sys.argv[1], sys.argv[2], sys.argv[3]
cam = kv.Kamera(puoli, f"{kansio}/kalibrointi_{puoli}.json", f"{kansio}/moodikuva_{puoli}.png")
cap = cv2.VideoCapture(video)
ak.yksilolliset_ruudut(video, [(8, 22), (56, 66), (104, 114)], puoli)
t2f = {round(t, 3): f for f, t in ak.T_KARTTA.items()}
kah = {(r["rata"], r["t_s"]): r for r in csv.DictReader(open(f"{kansio}/kahvaradat_{puoli}.csv"))}
rows = [r for r in csv.DictReader(open(f"{kansio}/kaariradat_{puoli}.csv")) if float(r["Y_akseli_cm"]) > 300 and kah[(r["rata"], r["t_s"])]["vari"] == "keltainen"]
out = []
prev = {}
for r in rows[::3]:
    f = t2f.get(round(float(r["t_s"]), 3))
    if f is None:
        continue
    X, Y = float(r["X_akseli_cm"]), float(r["Y_akseli_cm"])
    u, v = kv.kivimalli.project_3d(cam.K, cam.R, cam.t, np.array([[X, Y, 15.0]]))
    fr = cam.ruutu(cap, f)
    try:
        H, G = kv.kaaripisteet(cam, fr, (u[0], v[0]), "keltainen")
        s = kv.sovita_kupu(cam, fr, X, Y, (u[0], v[0]), H, alku=prev.get(r["rata"]))
    except Exception:
        continue
    prev[r["rata"]] = s["x"]
    out.append((r["rata"], r["t_s"], Y, s))
with open(f"{kansio}/kupuradat_{puoli}.csv", "w", newline="") as fh:
    wr = csv.writer(fh)
    wr.writerow(["rata", "t_s", "Y_akseli_cm", "dX_cm", "z_pohja_cm", "r_cm", "z_karki_cm", "etukaaren_rms_px", "iou"])
    for k, t, Y, s in out:
        wr.writerow([k, t, f"{Y:.1f}", f"{s['dX']:.2f}", f"{s['z_pohja']:.2f}", f"{s['r']:.2f}", f"{s['z_karki']:.2f}", f"{s['etukaari_rms_px']:.2f}", f"{s['iou']:.3f}"])
for k in sorted(set(o[0] for o in out)):
    a = np.array([[o[3]["z_pohja"], o[3]["r"], o[3]["z_karki"], o[3]["dX"], o[3]["etukaari_rms_px"], o[3]["iou"]] for o in out if o[0] == k])
    print(f"heitto {int(k) + 1}: n={len(a)}  z_pohja {np.median(a[:, 0]):.1f} cm (IQR {np.percentile(a[:, 0], 25):.1f}-{np.percentile(a[:, 0], 75):.1f}), "
          f"r {np.median(a[:, 1]):.1f} cm (IQR {np.percentile(a[:, 1], 25):.1f}-{np.percentile(a[:, 1], 75):.1f}), z_karki {np.median(a[:, 2]):.1f} cm, "
          f"dX {np.median(a[:, 3]):+.1f} cm, etukaaren RMS {np.median(a[:, 4]):.2f} px, IoU {np.median(a[:, 5]):.2f}")
