"""Zoomatut kuvat jokaisesta heitosta hogline-ylityksen kohdalta.

Kaytto: python make_crossing_images.py video.mp4 calib_profile.pkl heitot.csv ulos_kansio [gt_frames.txt]
- yleiskuva: yksi laatta / heitto (ylitysframe), keltainen viiva = kaukainen hogline, vihrea rengas = rata
- nauhat: kolme framea (ylitys -15, 0, +15) / heitto -> nakee liikkuuko kohde kuin kivi
"""
import sys, os, csv, pickle, collections
import numpy as np, cv2

video, pkl, csv_path, outdir = sys.argv[1:5]
GT = [int(x) for x in open(sys.argv[5]).read().split()] if len(sys.argv) > 5 else []
os.makedirs(outdir, exist_ok=True)
pk = pickle.load(open(pkl, "rb"))
K, R, t = pk["pose"]["K"], pk["pose"]["R"], pk["pose"]["t"]
cam, k1 = pk["calib"]["camera_matrix"], pk["calib"]["best_k1"]
zc = float(pk["profile"]["H_total_cm"]) / 2.0
dist = np.array([k1, 0, 0, 0, 0], float)
HOG = 3017.6


def proj(x, y, z=None):
    p = K @ (R @ np.array([x, y, zc if z is None else z]) + t)
    return p[:2] / p[2]


tracks = collections.defaultdict(list)
for r in csv.DictReader(open(csv_path)):
    tracks[int(r["stone_id"])].append((int(r["frame"]), float(r["x_m"]) * 100, float(r["y_m"]) * 100))


def crossing(v):
    for a, b in zip(v, v[1:]):
        if a[2] > HOG >= b[2]:
            return a[0]
    if v[0][2] <= HOG and len(v) >= 10:
        m = min(len(v), 30)
        sp = (v[0][2] - v[m - 1][2]) / max(1, v[m - 1][0] - v[0][0])
        if sp > 1.0:
            return int(round(v[0][0] - min((HOG - v[0][2]) / sp, 60)))
    return v[0][0] if v[0][2] <= HOG else None


throws = sorted(((crossing(v), sid, v) for sid, v in tracks.items()), key=lambda x: x[0])
cap = cv2.VideoCapture(video)
W, H, S = 90, 60, 4          # rajaus puoli-leveys/korkeus (px), suurennus


def tile(frame, v, label, mark=True):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, im = cap.read()
    im = cv2.undistort(im, cam, dist)
    row = min(v, key=lambda r: abs(r[0] - frame))
    if frame < v[0][0] and len(v) >= 10:
        # radalla ei ole rivejä ennen alkuaan (esim. alkupää pudotettu): ekstrapoloi alun nopeudesta
        m = min(len(v), 30)
        vx = (v[m - 1][1] - v[0][1]) / max(1, v[m - 1][0] - v[0][0])
        vy = (v[m - 1][2] - v[0][2]) / max(1, v[m - 1][0] - v[0][0])
        d = frame - v[0][0]
        row = (frame, v[0][1] + vx * d, v[0][2] + vy * d)
    no_rows = frame < v[0][0] - 2 or frame > v[-1][0] + 2
    cx, cy = proj(row[1], row[2])
    x0 = int(np.clip(cx - W, 0, im.shape[1] - 2 * W)); y0 = int(np.clip(cy - H, 0, im.shape[0] - 2 * H))
    crop = cv2.resize(im[y0:y0 + 2 * H, x0:x0 + 2 * W], None, fx=S, fy=S, interpolation=cv2.INTER_CUBIC)
    # hogline-viiva
    a = proj(-150, HOG, 0.0); b = proj(150, HOG, 0.0)
    pa = ((a - [x0, y0]) * S).astype(int); pb = ((b - [x0, y0]) * S).astype(int)
    cv2.line(crop, tuple(pa), tuple(pb), (0, 220, 255), 1, cv2.LINE_AA)
    if mark and not no_rows:
        cv2.circle(crop, (int((cx - x0) * S), int((cy - y0) * S)), 5 * S, (0, 200, 0), 2, cv2.LINE_AA)
    cv2.rectangle(crop, (0, 0), (crop.shape[1], 26), (255, 255, 255), -1)
    cv2.putText(crop, label + (" (ei rivia)" if no_rows else ""), (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return crop


def gt_of(c):
    return min(GT, key=lambda g: abs(g - c)) if GT else None


# yleiskuva
tiles = []
for i, (c, sid, v) in enumerate(throws, 1):
    g = gt_of(c)
    lab = f"#{i} ruutu {c}" + (f" (lista {g})" if g else "")
    tiles.append(tile(c, v, lab))
cols = 4
rows = (len(tiles) + cols - 1) // cols
h, w = tiles[0].shape[:2]
sheet = np.full((rows * h, cols * w, 3), 255, np.uint8)
for i, t_ in enumerate(tiles):
    sheet[(i // cols) * h:(i // cols + 1) * h, (i % cols) * w:(i % cols + 1) * w] = t_
cv2.imwrite(os.path.join(outdir, "hogline_ylitykset_yleiskuva.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])

# nauhat: 3 framea / heitto, 7 heittoa / arkki
strips = []
for i, (c, sid, v) in enumerate(throws, 1):
    g = gt_of(c)
    row = [tile(c + d, v, f"#{i} ruutu {c + d} ({d:+d})") for d in (-15, 0, 15)]
    strips.append(np.hstack(row))
per = 7
for n in range(0, len(strips), per):
    cv2.imwrite(os.path.join(outdir, f"hogline_ylitykset_nauhat_{n // per + 1}.jpg"), np.vstack(strips[n:n + per]), [cv2.IMWRITE_JPEG_QUALITY, 88])
print(len(throws), "heittoa; kuvat:", outdir)
