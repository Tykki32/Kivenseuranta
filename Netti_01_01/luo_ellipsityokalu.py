"""Raahattavien ellipsien sivu (Artifact-HTML): python luo_ellipsityokalu.py VIDEO tulokset vasen --ulos sivu.html [--ruudut 0:234,1:470,...]

Jokaiselle valitulle ruudulle: kuva-ikkuna (80 x 80 px, 11-kertaisena), alkuarvoksi kiintean kokoisen mallin ellipsi (kahva_malli.py) ja
varikynnyksen silhuetti. Sivulla ellipsin neljaa pistetta (A etureuna, B ylakarki, C takareuna, D alakarki) voi raahata; tallennetut ellipsit
kirjoitetaan sivun tietokantaan (kokoelma ellipsit). Pohja: ellipsityokalu_pohja.html."""
import argparse
import base64
import csv
import json
import os

import cv2
import numpy as np

import aja_kivi as ak
import kahva_malli as km
import kivi_netti as kv

OLETUS = "0:234,1:470,0:549,2:538,1:258,2:293,1:406,0:370"
W, S = 80, 11


def ellipsi_parametrit(poly):
    """cv2.fitEllipse -> (uc, vc, a, b, phi): a = pitka puoliakseli (pysty), b = lyhyt (vaaka), phi = lyhyen akselin suunta (rad, kuvan x-akselista)."""
    (cx, cy), (d1, d2), ang = cv2.fitEllipse(poly.astype(np.float32))
    if d1 <= d2:
        b, a, phi = d1 / 2.0, d2 / 2.0, np.radians(ang)
    else:
        b, a, phi = d2 / 2.0, d1 / 2.0, np.radians(ang + 90.0)
    while phi > np.pi / 2:
        phi -= np.pi
    while phi <= -np.pi / 2:
        phi += np.pi
    return float(cx), float(cy), float(a), float(b), float(phi)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("kansio")
    ap.add_argument("puoli")
    ap.add_argument("--ulos", required=True)
    ap.add_argument("--ruudut", default=OLETUS)
    a = ap.parse_args()
    cam = kv.Kamera(a.puoli, f"{a.kansio}/kalibrointi_{a.puoli}.json", f"{a.kansio}/moodikuva_{a.puoli}.png")
    kev = km.Kevyt(cam)
    cap = cv2.VideoCapture(a.video)
    ak.yksilolliset_ruudut(a.video, [(8, 22), (56, 66), (104, 114)], a.puoli)
    t2f = {round(t, 3): f for f, t in ak.T_KARTTA.items()}
    rows = list(csv.DictReader(open(f"{a.kansio}/kahvaradat_{a.puoli}.csv")))
    ast = km.lataa_asetukset(a.kansio, a.puoli)
    frames = []
    for arg in a.ruudut.split(","):
        rata, Y = arg.split(":")
        r = min((r for r in rows if r["rata"] == rata), key=lambda r: abs(float(r["Y_cm"]) - float(Y)))
        f = t2f[round(float(r["t_s"]), 3)]
        fr = cam.ruutu(cap, f)
        u, v = kv.kivimalli.project_3d(cam.K, cam.R, cam.t, np.array([[float(r["X_cm"]), float(r["Y_cm"]), 15.0]]))
        uv0 = (float(u[0]), float(v[0]))
        crop, mode, cx0, cy0 = km.leike(cam, fr, uv0)
        m = km.sovita_asetuksilla(kev, crop, mode, cx0, cy0, r["vari"], uv0, ast)
        if m is None:
            print("ohitetaan (mallia ei sovitettu):", arg)
            continue
        uc, vc, ea, eb, phi = ellipsi_parametrit(m["poly"])
        x0, y0 = int(round(uc)) - W // 2, int(round(vc)) - W // 2
        h, w = fr.shape[:2]
        xs = np.clip(np.arange(x0, x0 + W), 0, w - 1)
        ys = np.clip(np.arange(y0, y0 + W), 0, h - 1)
        big = cv2.resize(fr[np.ix_(ys, xs)], None, fx=S, fy=S, interpolation=cv2.INTER_CUBIC)
        jpg = cv2.imencode(".jpg", big, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()
        sil = None
        try:
            s = kv.kahva_silhuetti(fr, uv0, vari=r["vari"], tausta=cam.mode)
            C = s.contour[::4] - np.array([x0, y0])
            sil = [[round(float(p[0]), 2), round(float(p[1]), 2)] for p in C]
        except Exception:
            pass
        Yk = int(round(float(r["Y_cm"])))
        frames.append(dict(id=f"h{int(rata) + 1}_y{Yk}", heitto=int(rata) + 1, rata=int(rata), vari=r["vari"], Y=Yk, frame=int(f), t_s=float(r["t_s"]),
                           x0=x0, y0=y0, img="data:image/jpeg;base64," + base64.b64encode(jpg).decode(),
                           model=dict(uc=round(uc - x0, 4), vc=round(vc - y0, 4), a=round(ea, 4), b=round(eb, 4), phi=round(phi, 6)), sil=sil))
        print(f"{frames[-1]['id']}: ellipsi a={ea:.2f} b={eb:.2f} phi={np.degrees(phi):.2f} deg, kuva {len(jpg) // 1024} kB")
    pohja = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ellipsityokalu_pohja.html"), encoding="utf-8").read()
    html = pohja.replace("/*__FRAMES__*/[]", json.dumps(frames, ensure_ascii=True)).replace("/*__S__*/11", str(S)).replace("/*__W__*/80", str(W))
    open(a.ulos, "w", encoding="utf-8").write(html)
    print("kirjoitettu", a.ulos, f"{len(html) / 1e6:.2f} MB")
