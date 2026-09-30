"""HAKU-koe (Testi_03_04): hae 3D-mallin siluetille paras paikka graniittimaskista.

Pisteytys (maski = graniittimaski HAKU-ruudusta):
  + sisalla: maskipikselit siluetin (kupera peite MIINUS kahvan lovi) sisalla / siluetin ala        (palkinto)
  - ylitulo: maskipikselit siluetin (kupera peite) ULKOPUOLELLA / siluetin ala, PAITSI siluetin ylapuolella   (rangaistus, paino w)
  siluetin ylapuolinen alue = samojen sarakkeiden pikselit jotka ovat siluetin ylareunan ylapuolella -> ei rangaistusta.
Haku: karkea (3 cm) + hieno (1 cm) ristikko HAKU-loydon ymparilla (X +-40 cm, Y +-80 cm).
"""
import sys, os, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2
from haku_kokeilu import HakuLab

UP = 2


class SilhouetteSearch:
    def __init__(self, lab, w_leak=1.0):
        self.lab = lab; self.w = w_leak
        self.k9 = lab.M.k9; self.k94 = lab.M.k94
        self.pose = lab.pose
        self.K, self.R, self.t = (np.asarray(lab.pose[k], float) for k in ("K", "R", "t"))
        self.R = self.R.reshape(3, 3); self.t = self.t.reshape(3)
        pr = pickle.load(open(lab.pkl_path, "rb"))["profile"]
        self.sd = np.asarray(pr["shape_deltas"])
        self.body = lab.body

    def proj(self, x, y, z):
        p = self.K @ (self.R @ np.array([x, y, z]) + self.t)
        return p[0] / p[2], p[1] / p[2]

    def polys(self, X, Y):
        L = self.lab
        hull = self.k9._predicted_stone_hull(self.pose, X, Y, L.R_max, L.H_total, self.sd).reshape(-1, 2)
        notch = self.k9._handle_notch_hull(self.pose, X, Y, L.R_max, L.H_total, L.handle).reshape(-1, 2)
        return hull, notch

    def raster(self, poly, shape, ox, oy):
        m = np.zeros(shape, np.uint8)
        p = np.round((np.asarray(poly) - np.array([ox, oy]) + 0.5) * UP - 0.5).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(m, [p], 1)
        return m > 0

    def score(self, mask2, X, Y, ox, oy, detail=False):
        hull, notch = self.polys(X, Y)
        mh = self.raster(hull, mask2.shape, ox, oy); mn = self.raster(notch, mask2.shape, ox, oy)
        sil = mh & ~mn
        a = max(int(sil.sum()), 1)
        cols = sil.any(axis=0)
        above = (~(np.cumsum(mh, axis=0) > 0)) & cols[None, :]            # siluetin (hull) ylapuolella samoissa sarakkeissa
        inside = int((mask2 & sil).sum()); leak = int((mask2 & ~mh & ~above).sum())
        sc = inside / a - self.w * leak / a
        return (sc, inside / a, leak / a, sil, mh, above) if detail else sc

    def search(self, mask_crop, ox, oy, X0, Y0, dx=40.0, dy=80.0, coarse=3.0, fine=1.0):
        m2 = cv2.resize(mask_crop.astype(np.uint8), None, fx=UP, fy=UP, interpolation=cv2.INTER_NEAREST) > 0
        best = (-1e9, X0, Y0)
        for X in np.arange(X0 - dx, X0 + dx + 1e-6, coarse):
            for Y in np.arange(Y0 - dy, Y0 + dy + 1e-6, coarse):
                s = self.score(m2, X, Y, ox, oy)
                if s > best[0]: best = (s, X, Y)
        b2 = best
        for X in np.arange(best[1] - coarse, best[1] + coarse + 1e-6, fine):
            for Y in np.arange(best[2] - coarse, best[2] + coarse + 1e-6, fine):
                s = self.score(m2, X, Y, ox, oy)
                if s > b2[0]: b2 = (s, X, Y)
        return b2, m2
