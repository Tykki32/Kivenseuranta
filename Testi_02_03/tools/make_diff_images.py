"""Piirtaa kuvat heitoista joissa ristikkohaku ja mean-shift erosivat.

Kaytto: python make_diff_images.py video.mp4 calib_profile.pkl ristikko.csv meanshift.csv ulos_kansio
Jokaiselle heitolle: 3 ajanhetkea, rajaus (samat molemmille) kiven ymparilta, oranssi = ristikkohaun
rata/sijainti, vihrea = mean-shiftin. Jos ajolla ei ole rataa kyseisella framella, merkintaa ei piirreta.
"""
import sys, os, csv, pickle, types
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_throws import load_tracks, is_real, overlap

video, pkl, grid_csv, ms_csv, outdir = sys.argv[1:6]
os.makedirs(outdir, exist_ok=True)
pk = pickle.load(open(pkl, "rb"))
K, R, t = pk["pose"]["K"], pk["pose"]["R"], pk["pose"]["t"]
cam, k1 = pk["calib"]["camera_matrix"], pk["calib"]["best_k1"]
zc = float(pk["profile"]["H_total_cm"]) / 2.0
dist = np.array([k1, 0, 0, 0, 0], float)
G, M = load_tracks(grid_csv), load_tracks(ms_csv)


def proj(x, y):
    p = K @ (R @ np.array([x, y, zc]) + t)
    return p[:2] / p[2]


def pos_at(tracks, frame):
    """Kaikkien ratojen sijainnit annetulla framella -> lista (x,y)."""
    return [(x, y) for tr in tracks.values() for (f, x, y, *_r) in tr if f == frame]


def find_track(tracks, f0, f1, y0):
    best = None
    for k, v in tracks.items():
        fr = [r for r in v if f0 <= r[0] <= f1]
        if len(fr) > 20 and abs(fr[0][2] - y0) < 600:
            if best is None or len(fr) > best[1]:
                best = (k, len(fr))
    return best[0] if best else None


# (otsikko, ensisijainen rata (nimi, id), ruudut)
CASES = eval(sys.argv[6]) if len(sys.argv) > 6 else []
cap = cv2.VideoCapture(video)


def grab(frame):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, im = cap.read()
    return cv2.undistort(im, cam, dist)


W, H = 120, 80      # rajaus (px)
S = 4               # suurennus
for title, src, kid, (f0, f1) in CASES:
    src_tr = (G if src == "grid" else M)[kid]
    fr = [r[0] for r in src_tr if f0 <= r[0] <= f1]
    picks = [fr[int(len(fr) * q)] for q in (0.15, 0.5, 0.85)]
    panels = []
    for f in picks:
        im = grab(f)
        row = {r[0]: r for r in src_tr}.get(f)
        cx, cy = proj(row[1], row[2])
        x0 = int(np.clip(cx - W, 0, im.shape[1] - 2 * W)); y0 = int(np.clip(cy - H, 0, im.shape[0] - 2 * H))
        crop = cv2.resize(im[y0:y0 + 2 * H, x0:x0 + 2 * W], None, fx=S, fy=S, interpolation=cv2.INTER_CUBIC)
        for name, tr, col in (("ristikko", G, (0, 140, 255)), ("mean-shift", M, (0, 200, 0))):
            for (x, y) in pos_at(tr, f):
                px_, py_ = proj(x, y)
                if x0 <= px_ < x0 + 2 * W and y0 <= py_ < y0 + 2 * H:
                    cv2.circle(crop, (int((px_ - x0) * S), int((py_ - y0) * S)), 5 * S, col, 2, cv2.LINE_AA)
        cv2.rectangle(crop, (0, 0), (330, 34), (255, 255, 255), -1)
        cv2.putText(crop, f"ruutu {f}   Y = {row[2] / 100:.1f} m", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        panels.append(crop)
    strip = np.hstack(panels)
    head = np.full((44, strip.shape[1], 3), 255, np.uint8)
    cv2.putText(head, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(head, "oranssi = ristikkohaku   vihrea = mean-shift   (rengas vain jos ajo seuraa kohdetta)", (8, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 90), 1, cv2.LINE_AA)
    out = os.path.join(outdir, f"heitto_{f0}_{f1}_{src}.png")
    cv2.imwrite(out, np.vstack([head, strip]))
    print(out)
