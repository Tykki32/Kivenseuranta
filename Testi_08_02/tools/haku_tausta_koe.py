"""HAKU: taustanvaimennuksen vertailu (Testi_03_04). Vertaa live-vaimennusta (pikselikohtainen, C++ suppress_shadow_background) ja
TOLERANTTIA vaimennusta (pikseli on tausta, jos se on referenssin NAAPURUSTON min..max -vaihteluvalilla +- t) - jälkimmäinen poistaa ohuet
viivat/reunat jotka jaavat ruudun ja referenssin ali-/yhden pikselin kohdistusvirheesta (keskiviiva, tekstit, harjan varsi).

Kayttaa ruutuja joissa on myos 'raw' (TRACKER_FRAME_DUMP_RAW=1, katso tools/HAKU_KOKEILU.md). Kaytto:
  python tools/haku_tausta_koe.py <dump_dir> <calib_profile.pkl> <stones.csv>
"""
import sys, os, types
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2
from haku_kokeilu import HakuLab, ENTRIES


def suppress_tolerant(M, st, raw, ref, nb=3, t=10.0, use_shadow=True, use_ice=True, open_px=0):
    """Toleranttı taustanvaimennus. nb = naapuruston koko (3 = 3x3), t = marginaali. Palauttaa valkoiseksi vaimennetun BGR-ruudun."""
    g, b = M.estimate_photometric_correction(raw, ref)
    f = np.clip(raw.astype(np.float32) * np.asarray(g, np.float32)[None, None, :] + np.asarray(b, np.float32)[None, None, :], 0, 255).astype(np.uint8)
    k = np.ones((nb, nb), np.uint8)
    rmin = cv2.erode(ref, k); rmax = cv2.dilate(ref, k)
    inside = ((f.astype(np.int16) >= rmin.astype(np.int16) - t) & (f.astype(np.int16) <= rmax.astype(np.int16) + t)).all(axis=2)
    bg = inside
    v = f.max(axis=2).astype(np.int16); vr = ref.max(axis=2).astype(np.int16)
    if use_shadow:
        drop = vr - v
        bg |= (drop > M.SHADOW_V_DROP_MIN) & (drop < M.SHADOW_V_DROP_MAX)
    if use_ice:
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        bg |= (hsv[..., 1] < M.ICE_S_MAX) & (v > M.ICE_V_MIN)
    fg = (~bg).astype(np.uint8)
    if open_px > 1:
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((open_px, open_px), np.uint8))
    out = f.copy(); out[fg == 0] = 255
    return out


def zone_box(lab):
    """Hakualueen (X +-70, Y hogline-100..+300) rajaava laatikko kuvassa + reunus."""
    K, R, t = (np.asarray(lab.pose[k], float) for k in ("K", "R", "t")); R = R.reshape(3, 3); t = t.reshape(3)
    pts = []
    for x in (-70, 70):
        for y in (lab.k92.SEARCH_Y_MIN_CM, lab.k92.SEARCH_Y_MAX_CM):
            for z in (0, lab.H_total):
                p = K @ (R @ np.array([x, y, z]) + t); pts.append((p[0] / p[2], p[1] / p[2]))
    pts = np.array(pts)
    return int(pts[:, 0].min()) - 10, int(pts[:, 1].min()) - 10, int(pts[:, 0].max()) + 10, int(pts[:, 1].max()) + 10


def main(dump, pkl, csv_path):
    lab = HakuLab(dump, pkl, csv_path)
    M, st = lab.M, lab.st
    x0, y0, x1, y1 = zone_box(lab)
    print(f"hakualue kuvassa: x {x0}..{x1}, y {y0}..{y1}")
    variants = {"live (pikselikohtainen)": None, "tolerantti 3x3, t=10": dict(nb=3, t=10.0), "tolerantti 5x5, t=10": dict(nb=5, t=10.0),
                "tolerantti 3x3, t=10 + avaus 3x3": dict(nb=3, t=10.0, open_px=3)}
    frames = {f: np.load(os.path.join(dump, f"g{f:06d}.npz")) for f in lab.frames_available()}
    frames = {f: d for f, d in frames.items() if "raw" in d.files}
    print(f"{len(frames)} ruutua joissa raw")
    # tarkistus: C++ vaimennus raw:sta = tallennettu
    f0 = sorted(frames)[len(frames) // 2]
    g, b = M.estimate_photometric_correction(frames[f0]["raw"], lab.ref)
    chk = st.suppress_shadow_background(frames[f0]["raw"], lab.ref, np.asarray(g, float), np.asarray(b, float), float(M.GRANITE_DIFF_THRESHOLD),
                                        float(M.SHADOW_V_DROP_MIN), float(M.SHADOW_V_DROP_MAX), int(M.ICE_S_MAX), int(M.ICE_V_MIN))
    print(f"ruutu {f0}: raw -> C++ vaimennus vs tallennettu live-ruutu: eri pikseleita {(np.abs(chk.astype(int) - frames[f0]['frame'].astype(int)).max(axis=2) > 0).sum()}")
    stats = {k: dict(fg=[], hits=0, bad=0, first={}) for k in variants}
    for f in sorted(frames):
        d = frames[f]
        for name, kw in variants.items():
            img = d["frame"] if kw is None else suppress_tolerant(M, st, d["raw"], lab.ref, **kw)
            z = img[y0:y1, x0:x1]
            stats[name]["fg"].append(int((z.min(axis=2) < 250).sum()))
            res = lab.run(f, frame=np.ascontiguousarray(img))
            for r, dist, sid, ex in lab.classify(f, res):
                if lab.is_match(dist, ex):
                    stats[name]["hits"] += 1; stats[name]["first"].setdefault(sid, f)
                else:
                    stats[name]["bad"] += 1
    print(f"\n{'vaimennus':36s} {'fg-pikselit/ruutu (alue)':>26s} {'HAKU osumat':>12s} {'vaaria':>7s}  1. loyto kivi 17/3/51")
    for name, s in stats.items():
        fr = s["first"]
        print(f"{name:36s} {np.mean(s['fg']):26.0f} {s['hits']:12d} {s['bad']:7d}  {fr.get('17', '-')}/{fr.get('3', '-')}/{fr.get('51', '-')}")
    return lab, frames, variants


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
