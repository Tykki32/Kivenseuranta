"""Kahvamallin vakiot kasin merkityista ellipseista: python kalibroi_kahvamalli.py VIDEO tulokset vasen [--merkinnat tulokset/kahva_ellipsit_kasin.json]

Merkinnat tulevat raahattavasta sivusta (luo_ellipsityokalu.py): ellipsin neljä pistetta (A etureuna, B ylakarki, C takareuna, D alakarki)
ruutukoordinaateissa. Jokaiselle merkitylle ruudulle ajetaan koko soft-sovitus (kahva_malli.sovita) ilman siirtoa, ja malli - merkinta -jaannoksista
(8 lukua / ruutu) ratkaistaan:
  * koko (A, B): ruudukkohaku, valitaan pienin jaannos siirtojen korjauksen jalkeen
  * kromahavainnon vakiosiirtymat: vaakasuunta (du) varin mukaan, pystysuunta (dv) yhteinen
Lisaksi leave-one-out: vakiot sovitetaan 7 ruudusta ja ennustetaan 8. Kirjoittaa tulokset/kahvamalli_asetukset_<puoli>.json."""
import argparse
import csv
import json
from multiprocessing import Pool

import cv2
import numpy as np

import aja_kivi as ak
import kahva_malli as km
import kivi_netti as kv
from luo_ellipsityokalu import ellipsi_parametrit

ASETUS = {}


def _tyo(a):
    i, A, B = a
    l = ASETUS["leikkeet"][i]
    m = km.sovita(ASETUS["kev"], l["crop"], l["mode"], l["x0"], l["y0"], l["vari"], l["uv0"], A, B, siirto=(0.0, 0.0))
    if m is None:
        return None
    uc, vc, ea, eb, phi = ellipsi_parametrit(m["poly"])
    e1 = np.array([np.cos(phi), np.sin(phi)])
    e2 = np.array([-np.sin(phi), np.cos(phi)])
    c = np.array([uc, vc])
    return dict(A=c + e1 * eb, C=c - e1 * eb, B=c - e2 * ea, D=c + e2 * ea)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("kansio")
    ap.add_argument("puoli")
    ap.add_argument("--merkinnat", default=None)
    ap.add_argument("--A", default="11.0:11.8:0.2", help="alku:loppu:askel (cm), puoliakseli sivusuunnassa")
    ap.add_argument("--B", default="8.0:9.2:0.3", help="alku:loppu:askel (cm), puoliakseli radan suunnassa")
    a = ap.parse_args()
    merkinnat = json.load(open(a.merkinnat or f"{a.kansio}/kahva_ellipsit_kasin.json"))
    cam = kv.Kamera(a.puoli, f"{a.kansio}/kalibrointi_{a.puoli}.json", f"{a.kansio}/moodikuva_{a.puoli}.png")
    cap = cv2.VideoCapture(a.video)
    ak.yksilolliset_ruudut(a.video, [(8, 22), (56, 66), (104, 114)], a.puoli)
    rows = list(csv.DictReader(open(f"{a.kansio}/kahvaradat_{a.puoli}.csv")))
    leikkeet = []
    for d in merkinnat:
        r = min((r for r in rows if int(r["rata"]) == d["rata"]), key=lambda r: abs(float(r["t_s"]) - d["t_s"]))
        fr = cam.ruutu(cap, d["frame"])
        u, v = kv.kivimalli.project_3d(cam.K, cam.R, cam.t, np.array([[float(r["X_cm"]), float(r["Y_cm"]), 15.0]]))
        uv0 = (float(u[0]), float(v[0]))
        crop, mode, x0, y0 = km.leike(cam, fr, uv0)
        leikkeet.append(dict(crop=crop, mode=mode, x0=x0, y0=y0, vari=d["vari"], uv0=uv0))
    ASETUS["leikkeet"], ASETUS["kev"] = leikkeet, km.Kevyt(cam)
    n = len(merkinnat)
    vari = np.array([d["vari"] for d in merkinnat])
    kasin = [np.concatenate([np.array(d["pisteet_ruutu"][k], float) for k in "ABCD"]) for d in merkinnat]
    def ruudukko(spec):
        a0, a1, st = (float(x) for x in spec.split(":"))
        return [round(float(v), 3) for v in np.arange(a0, a1 + 1e-9, st)]

    As, Bs = ruudukko(a.A), ruudukko(a.B)
    R = {}
    with Pool(4) as pool:
        for A in As:
            for B in Bs:
                res = pool.map(_tyo, [(i, A, B) for i in range(n)])
                R[(A, B)] = np.array([np.concatenate([e[k] for k in "ABCD"]) - kasin[i] for i, e in enumerate(res)])

    def siirrot(r, idx):
        dv = r[idx][:, 1::2].mean()
        duy = r[[i for i in idx if vari[i] == "keltainen"]][:, 0::2].mean()
        dup = r[[i for i in idx if vari[i] == "punainen"]][:, 0::2].mean()
        return duy, dup, dv

    def jaannos(r, i, sh):
        x = r[i].copy()
        x[0::2] -= sh[0] if vari[i] == "keltainen" else sh[1]
        x[1::2] -= sh[2]
        return x

    def paras(idx):
        b = None
        for k, r in R.items():
            sh = siirrot(r, idx)
            e = np.sqrt(np.mean(np.concatenate([jaannos(r, i, sh) for i in idx]) ** 2))
            if b is None or e < b[0]:
                b = (e, k, sh)
        return b

    e, (A, B), sh = paras(list(range(n)))
    se = []
    for i in range(n):
        idx = [j for j in range(n) if j != i]
        _, k2, sh2 = paras(idx)
        se.append(jaannos(R[k2], i, sh2) ** 2)
    loo = float(np.sqrt(np.mean(np.concatenate(se))))
    lahto = float(np.sqrt(np.mean(R[(A, B)] ** 2)))
    print(f"{n} merkittya ruutua: A = {A:.1f} cm, B = {B:.1f} cm; siirrot (du keltainen, du punainen, dv) = ({sh[0]:+.2f}, {sh[1]:+.2f}, {sh[2]:+.2f}) px")
    print(f"jaannos merkintoihin: ilman siirtoja {lahto:.2f} px, siirtojen kanssa {e:.2f} px, leave-one-out {loo:.2f} px (4 pistetta, 8 koordinaattia)")
    asetukset = km.lataa_asetukset()
    asetukset.update(A_cm=A, B_cm=B, siirto_px={"keltainen": [round(float(sh[0]), 3), round(float(sh[2]), 3)], "punainen": [round(float(sh[1]), 3), round(float(sh[2]), 3)]},
                     jaannos_px=round(float(e), 3), loo_px=round(loo, 3), merkittuja=n, lahde="kasin merkityt ellipsit (kahva_ellipsit_kasin.json)")
    json.dump(asetukset, open(f"{a.kansio}/kahvamalli_asetukset_{a.puoli}.json", "w"), indent=1)
    print(f"-> {a.kansio}/kahvamalli_asetukset_{a.puoli}.json")
