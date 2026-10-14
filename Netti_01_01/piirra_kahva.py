"""Kahvan muovin ellipsi valituissa ruuduissa isona, selitteen kanssa:
python piirra_kahva.py VIDEO tulokset vasen --ruudut 0:236,1:467 [--ulos tulokset/kahva_ruudut.jpg] [--kirjaimet] [--sarakkeita 2]

--ruudut    lista rata:Y_cm (rata = 0, 1, 2 = heitto 1, 2, 3; Y = lahin seurannan piste, kahvaradat_<puoli>.csv)
--kirjaimet merkitsee ellipsin osat kirjaimilla (A etureuna, B ylakarki, C takareuna, D alakarki)
Vari: magenta = ellipsi, vihrea = etureunan pisteet, syaani = muovin silhuetti (varimaski), oranssi = silhuetin ylin/alin kohta ja
karkien keski-u, punainen = ellipsin kohta silhuetin ulkopuolella yli 0,3 px."""
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
ap.add_argument("--ruudut", required=True)
ap.add_argument("--ulos", default=None)
ap.add_argument("--suurennus", type=int, default=10)
ap.add_argument("--puolikas", type=int, default=44)
ap.add_argument("--sarakkeita", type=int, default=2)
ap.add_argument("--kirjaimet", action="store_true")
a = ap.parse_args()

cam = kv.Kamera(a.puoli, f"{a.kansio}/kalibrointi_{a.puoli}.json", f"{a.kansio}/moodikuva_{a.puoli}.png")
cap = cv2.VideoCapture(a.video)
ak.yksilolliset_ruudut(a.video, [(8, 22), (56, 66), (104, 114)], a.puoli)
t2f = {round(t, 3): f for f, t in ak.T_KARTTA.items()}
rows = list(csv.DictReader(open(f"{a.kansio}/kahvaradat_{a.puoli}.csv")))
S, h = a.suurennus, a.puolikas
FONT = cv2.FONT_HERSHEY_SIMPLEX


def ruutu(rata, Y):
    r = min((r for r in rows if r["rata"] == rata), key=lambda r: abs(float(r["Y_cm"]) - Y))
    f = t2f[round(float(r["t_s"]), 3)]
    fr = cam.ruutu(cap, f)
    u, v = kv.kivimalli.project_3d(cam.K, cam.R, cam.t, np.array([[float(r["X_cm"]), float(r["Y_cm"]), 15.0]]))
    s = kv.kahva_silhuetti(fr, (u[0], v[0]), vari=r["vari"], tausta=cam.mode)
    o = kv.sovita_kuvaellipsi(s, painot=(1.0, 10.0, 10.0))
    cx, cy = int(round(u[0])), int(round(v[0]))
    x0, y0 = cx - h, cy - h
    big = cv2.resize(fr[y0:y0 + 2 * h, x0:x0 + 2 * h], None, fx=S, fy=S, interpolation=cv2.INTER_CUBIC)
    P = lambda pt: (int(round((pt[0] - x0 + 0.5) * S)), int(round((pt[1] - y0 + 0.5) * S)))
    C = s.contour
    for i in range(len(C) - 1):
        cv2.line(big, P(C[i]), P(C[i + 1]), (255, 255, 0), 1, cv2.LINE_AA)
    for pt in s.front:
        cv2.circle(big, P(pt), 3, (0, 255, 0), -1)
    for vv in (s.v_top, s.v_bot):
        cv2.line(big, P((x0, vv)), P((x0 + 2 * h, vv)), (0, 140, 255), 1)
    cv2.line(big, P((s.u_mid, y0)), P((s.u_mid, y0 + 2 * h)), (0, 140, 255), 1)
    E = o["E"]
    out = kv._sd_nayte(s, E) > 0.3
    for i in range(len(E) - 1):
        cv2.line(big, P(E[i]), P(E[i + 1]), (0, 0, 255) if out[i] and out[i + 1] else (255, 0, 255), 2, cv2.LINE_AA)
    if a.kirjaimet:
        def merkki(pt, teksti, dx, dy):
            p = P(pt)
            q = (p[0] + dx, p[1] + dy)
            cv2.line(big, p, q, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.circle(big, q, 17, (255, 255, 255), -1)
            cv2.circle(big, q, 17, (0, 0, 0), 2)
            cv2.putText(big, teksti, (q[0] - 9, q[1] + 8), FONT, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        merkki(E[np.argmax(E[:, 0])], "A", 45, 0)
        merkki(E[np.argmin(E[:, 1])], "B", 0, -40)
        merkki(E[np.argmin(E[:, 0])], "C", -45, 0)
        merkki(E[np.argmax(E[:, 1])], "D", 0, 40)
    uc, vc, ea, eb, phi = o["p"]
    cv2.putText(big, f"heitto {int(rata) + 1} ({r['vari']}) Y={float(r['Y_cm']):.0f} cm", (8, 26), FONT, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(big, f"etureunan RMS {o['rms_front']:.2f} px, ulkopuolella max {o['cont_max']:.2f} px, puoliakselit {ea:.1f} x {eb:.1f} px",
                (8, 2 * h * S - 12), FONT, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
    return big


tiles = []
for arg in a.ruudut.split(","):
    rata, Y = arg.split(":")
    tiles.append(ruutu(rata, float(Y)))
cols = max(1, min(a.sarakkeita, len(tiles)))
rimg = [np.hstack(tiles[i:i + cols] + [np.zeros_like(tiles[0])] * (cols - len(tiles[i:i + cols]))) for i in range(0, len(tiles), cols)]
img = np.vstack(rimg)
sel = np.full((70, img.shape[1], 3), 255, np.uint8)
cv2.putText(sel, "magenta = ellipsi   vihrea = etureunan pisteet   syaani = muovin silhuetti (varimaski)", (10, 26), FONT, 0.6, (60, 60, 60), 2, cv2.LINE_AA)
cv2.putText(sel, "oranssi = silhuetin ylin/alin kohta ja karkien keski-u   punainen = ellipsi silhuetin ulkopuolella yli 0,3 px", (10, 54), FONT, 0.6, (60, 60, 60), 2, cv2.LINE_AA)
img = np.vstack([img, sel])
ulos = a.ulos or f"{a.kansio}/kahva_ruudut.jpg"
cv2.imwrite(ulos, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
print("kirjoitettu", ulos, img.shape)
