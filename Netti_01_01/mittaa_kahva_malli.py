"""Kahvan muovin kiintean kokoinen malli ruuduittain: python mittaa_kahva_malli.py VIDEO tulokset vasen [--opi] [--askel 1] [--ymin 150] [--ymax 640]

--opi   oppii yhteisen koon (A, B) puhtaista keltaisista ruuduista (Y = 330-520 cm, ei kasia) ja tallentaa tulokset/kahvamalli_koko_<puoli>.json;
        ilman sita kaytetaan tallennettua kokoa (tai kahva_malli.py:n oletusta).
Lukee tulokset/kahvaradat_<puoli>.csv (kahvan vari-seuranta), sovittaa jokaiseen ruutuun akselin (X, Y) ja kirjoittaa
tulokset/kahvamalli_<puoli>.csv."""
import argparse
import csv
import json
import os
from multiprocessing import Pool

import cv2
import numpy as np

import aja_kivi as ak
import kahva_malli as km
import kivi_netti as kv


def _tyo(args):
    cam, crop, mode, x0, y0, vari, uv0, A, B = args
    r = km.sovita(cam, crop, mode, x0, y0, vari, uv0, A, B)
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("kansio")
    ap.add_argument("puoli")
    ap.add_argument("--opi", action="store_true")
    ap.add_argument("--askel", type=int, default=1)
    ap.add_argument("--ymin", type=float, default=150.0)
    ap.add_argument("--ymax", type=float, default=640.0)
    ap.add_argument("--opi-vari", default="keltainen")
    a = ap.parse_args()

    cam = kv.Kamera(a.puoli, f"{a.kansio}/kalibrointi_{a.puoli}.json", f"{a.kansio}/moodikuva_{a.puoli}.png")
    kev = km.Kevyt(cam)
    cap = cv2.VideoCapture(a.video)
    ak.yksilolliset_ruudut(a.video, [(8, 22), (56, 66), (104, 114)], a.puoli)
    t2f = {round(t, 3): f for f, t in ak.T_KARTTA.items()}
    rows = [r for i, r in enumerate(csv.DictReader(open(f"{a.kansio}/kahvaradat_{a.puoli}.csv")))
            if a.ymin <= float(r["Y_cm"]) <= a.ymax]
    rows = rows[::a.askel]
    leikkeet, meta = [], []
    for r in rows:
        f = t2f.get(round(float(r["t_s"]), 3))
        if f is None:
            continue
        fr = cam.ruutu(cap, f)
        u, v = kv.kivimalli.project_3d(cam.K, cam.R, cam.t, np.array([[float(r["X_cm"]), float(r["Y_cm"]), 15.0]]))
        crop, mode, x0, y0 = km.leike(cam, fr, (u[0], v[0]))
        leikkeet.append((crop, mode, x0, y0, r["vari"], (float(u[0]), float(v[0]))))
        meta.append(dict(rata=r["rata"], vari=r["vari"], t_s=r["t_s"], frame=f, Y_seuranta_cm=float(r["Y_cm"])))
    print(len(leikkeet), "ruutua")

    json_polku = f"{a.kansio}/kahvamalli_koko_{a.puoli}.json"
    A, B = km.A_CM, km.B_CM
    with Pool(4) as pool:
        if a.opi:
            puhtaat = [lk for lk, m in zip(leikkeet, meta) if m["vari"] == a.opi_vari and 330.0 <= m["Y_seuranta_cm"] <= 520.0]
            A, B, J = km.opi_koko(kev, puhtaat, pool=pool)
            json.dump(dict(A_cm=A, B_cm=B, J=J, z0_cm=km.Z0_CM, sigma_px=km.SIGMA_PX, ruutuja=len(puhtaat), vari=a.opi_vari), open(json_polku, "w"), indent=1)
            print(f"opittu koko: A = {A:.2f} cm, B = {B:.2f} cm (J = {J:.5f}, {len(puhtaat)} ruutua) -> {json_polku}")
        elif os.path.exists(json_polku):
            d = json.load(open(json_polku))
            A, B = d["A_cm"], d["B_cm"]
            print(f"tallennettu koko: A = {A:.2f} cm, B = {B:.2f} cm")
        tulokset = pool.map(_tyo, [(kev, c, m, x0, y0, v, uv, A, B) for (c, m, x0, y0, v, uv) in leikkeet])
    ulos = f"{a.kansio}/kahvamalli_{a.puoli}.csv"
    n_ok = 0
    with open(ulos, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["rata", "vari", "t_s", "frame", "Y_seuranta_cm", "X_akseli_cm", "Y_akseli_cm", "rms", "vahvistus", "siirto", "kontrasti",
                     "A_cm", "B_cm", "z0_cm", "v_ylin_px", "v_alin_px", "uc_px", "vc_px"])
        for m, r in zip(meta, tulokset):
            if r is None:
                continue
            n_ok += 1
            p = r["poly"]
            wr.writerow([m["rata"], m["vari"], m["t_s"], m["frame"], f"{m['Y_seuranta_cm']:.1f}", f"{r['X']:.2f}", f"{r['Y']:.2f}", f"{r['rms']:.4f}",
                         f"{r['vahvistus']:.3f}", f"{r['siirto']:.3f}", f"{r['kontrasti']:.1f}", f"{A:.2f}", f"{B:.2f}", f"{km.Z0_CM:.1f}",
                         f"{p[:, 1].min():.2f}", f"{p[:, 1].max():.2f}", f"{p[:, 0].mean():.2f}", f"{p[:, 1].mean():.2f}"])
    print(f"{n_ok}/{len(meta)} ruutua sovitettu -> {ulos}")
    for k in sorted(set(m["rata"] for m in meta)):
        q = [r["rms"] for m, r in zip(meta, tulokset) if m["rata"] == k and r is not None]
        print(f"heitto {int(k) + 1}: {len(q)} ruutua, RMS mediaani {np.median(q):.3f}, 90 % {np.percentile(q, 90):.3f}")
