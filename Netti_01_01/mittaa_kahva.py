"""Kahvan muovin ellipsi ja korkeus: python mittaa_kahva.py VIDEO tulokset vasen [--askel 3] [--kuva kuva.jpg]

Joka seurannan ruudusta (tulokset/kahvaradat_<puoli>.csv):
 1) kiven alareunan kaari (iskukaista R = 14,55 cm, z = 3 cm) -> akseli jaalla (X, Y)
 2) kahvan muovin silhuetti (Lab-keltaisuus tai punaisuus, alipikseli) ja siihen kuvatason ellipsi: etureuna tarkasti, karjet
    lahes maksimikohdassa (~2 mm sisalla), takaosa muovin sisalla (yksipuolinen etaisyyskenttarajoite)
 3) ellipsin keskipiste on kiven akselilla: (X, z) ratkaistaan kun Y on alareunan kaaresta -> kahvan korkeus z
Kirjoittaa kahvaellipsit_<puoli>.csv ja tulostaa tilastot."""
import argparse
import csv

import cv2
import numpy as np

import aja_kivi as ak
import kivi_netti as kv

ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("kansio")
ap.add_argument("puoli")
ap.add_argument("--askel", type=int, default=3, help="joka n:s seurannan piste")
ap.add_argument("--ymin", type=float, default=200.0)
ap.add_argument("--ymax", type=float, default=600.0)
a = ap.parse_args()

cam = kv.Kamera(a.puoli, f"{a.kansio}/kalibrointi_{a.puoli}.json", f"{a.kansio}/moodikuva_{a.puoli}.png")
cap = cv2.VideoCapture(a.video)
ak.yksilolliset_ruudut(a.video, [(8, 22), (56, 66), (104, 114)], a.puoli)
t2f = {round(t, 3): f for f, t in ak.T_KARTTA.items()}
rows = [r for r in csv.DictReader(open(f"{a.kansio}/kahvaradat_{a.puoli}.csv"))]
out = []
for i, r in enumerate(rows):
    if i % a.askel or not (a.ymin <= float(r["Y_cm"]) <= a.ymax):
        continue
    f = t2f.get(round(float(r["t_s"]), 3))
    if f is None:
        continue
    fr = cam.ruutu(cap, f)
    u, v = kv.kivimalli.project_3d(cam.K, cam.R, cam.t, np.array([[float(r["X_cm"]), float(r["Y_cm"]), 15.0]]))
    try:
        G = kv.alareuna_pisteet(cam, fr, (u[0], v[0]))
        Xa, Ya, e_ala = kv.sovita_alakaari(cam, G, (u[0], v[0]))
        s = kv.kahva_silhuetti(fr, (u[0], v[0]), vari=r["vari"], tausta=cam.mode)
        o = kv.sovita_kuvaellipsi(s, painot=(1.0, 10.0, 10.0))
        uc, vc, ea, eb, phi = o["p"]
        Xh, z, res = kv.kahva_keskipiste_3d(cam, uc, vc, Ya, X0=Xa)
        pj = lambda x, y, zz: kv.kivimalli.project_3d(cam.K, cam.R, cam.t, np.array([[x, y, zz]]))
        u0, v0 = pj(Xh, Ya, z)
        sx = np.hypot(pj(Xh + 1, Ya, z)[0][0] - u0[0], pj(Xh + 1, Ya, z)[1][0] - v0[0])
        sy = np.hypot(pj(Xh, Ya + 1, z)[0][0] - u0[0], pj(Xh, Ya + 1, z)[1][0] - v0[0])
    except Exception as e:
        continue
    out.append(dict(rata=r["rata"], vari=r["vari"], t_s=r["t_s"], frame=f, X_ala=Xa, Y_ala=Ya, e_ala=e_ala, X_kahva=Xh, z_cm=z,
                    a_cm=ea / sx, b_cm=eb / sy, uc=uc, vc=vc, a_px=ea, b_px=eb, phi_deg=np.degrees(phi), rms_front=o["rms_front"],
                    cont_max=o["cont_max"], tip=max(o["tip_top"], o["tip_bot"])))
with open(f"{a.kansio}/kahvaellipsit_{a.puoli}.csv", "w", newline="") as fh:
    wr = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
    wr.writeheader()
    for o in out:
        wr.writerow({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in o.items()})
for k in sorted(set(o["rata"] for o in out)):
    q = [o for o in out if o["rata"] == k]
    ok = [o for o in q if o["rms_front"] < 0.8 and o["cont_max"] < 0.7]
    z = np.array([o["z_cm"] for o in ok])
    print(f"heitto {int(k) + 1} ({q[0]['vari']}): {len(q)} ruutua, hyvia {len(ok)}; etureunan RMS mediaani {np.median([o['rms_front'] for o in ok]):.2f} px, "
          f"ulkopuolella max mediaani {np.median([o['cont_max'] for o in ok]):.2f} px; kahvan korkeus z mediaani {np.median(z):.1f} cm "
          f"(IQR {np.percentile(z, 25):.1f}-{np.percentile(z, 75):.1f}); soikion puoliakselit {np.median([o['a_cm'] for o in ok]):.1f} x "
          f"{np.median([o['b_cm'] for o in ok]):.1f} cm; akselin sivusiirto {np.median([o['X_kahva'] - o['X_ala'] for o in ok]):+.1f} cm")
