"""HAKU:n siluettitarkennus (Testi_03_04, oletuksena PAALLA; HAKU_SILHOUETTE=0 kytkee pois).

HAKU loytaa ehdokkaan (X,Y). Tarkennus hakee kiven 3D-mallin siluetille paremman paikan GRANIITTIMASKISTA (HAKU-ruutu, tausta vaimennettu):
  + sisalla:  maskipikselit siluetin (kupera peite MINUS kahvan lovi) sisalla / siluetin ala            (palkinto)
  - ylitulo:  maskipikselit siluetin kuperan peitteen ULKOPUOLELLA / siluetin ala * W_LEAK, PAITSI kiven YLAPUOLELLA
              (kuva on kierretty: kiven ylapuoli = kuvassa OIKEALLA -> samoilla riveilla siluetin oikealla puolella ei rangaistusta)
  + keskitys: LAM * exp(-(d/sigma)^2), d = maskin (peitteen sisalla) ja siluetin massakeskipisteiden etaisyys (px), sigma = SIGMA_FRAC*sqrt(ala)
Ylitysrangaistus lasketaan vain siluetin ymparilla olevalla BAND_PX-kaistalla; maski on oma (3x3-avaus).
Siluetti siirretaan KUVATASOSSA (+-MAX_SHIFT_PX, 0.5 px askel, korrelaatiot cv2.matchTemplate:lla) ja paras siirto muutetaan
maatasoon (X,Y) Jacobin avulla. Yksi ehdokas ~ muutama ms. Kayttaa samaa profiilia/mallia kuin HAKU.
"""
import numpy as np
import cv2
try:
    import stone_tracker as _st
except Exception:          # pragma: no cover
    _st = None

UP = 2
W_LEAK = 1.0
LAM = 0.1
SIGMA_FRAC = 0.3
MAX_SHIFT_PX = 6
HALF = 26            # ikkunan puolikas (px)
MASK_MARGIN = 75     # graniittimaskin sumennuksen reunus (px)
OPEN_SIZE = 3        # siluettitarkennuksen oma maski: 3x3-avaus (5x5 poisti kaukaisen, 5-6 px leveän kiven)
BAND_PX = 3          # ylitysrangaistus vain siluetin ymparilla olevalla kaistalla (px): kaukainen tumma kohde (heittaja) ei vedä siluettia


class SilhouetteRefiner:
    def __init__(self, k9, pose, R_max, H_total, shape_deltas, handle_r_frac, w_leak=W_LEAK, lam=LAM, sigma_frac=SIGMA_FRAC,
                 max_shift_px=MAX_SHIFT_PX, k94=None, use_cpp=True, open_size=OPEN_SIZE, band_px=BAND_PX):
        self.open_size, self.band_px = int(open_size), int(band_px)
        self.k9 = k9; self.pose = pose
        self.max_shift_px = max_shift_px; self.w_leak, self.lam_c, self.sigfrac_c = w_leak, lam, sigma_frac
        # C++-versio (stone_tracker.silhouette_refine_cpp): tarvitsee mallin pisteet (k94.build_local_stone_rings, sama kuin HAKUn runko)
        self.body_pts = None
        if use_cpp and k94 is not None and _st is not None and hasattr(_st, "silhouette_refine_cpp"):
            self.body_pts = np.ascontiguousarray(k94.build_local_stone_rings(R_max, H_total, np.asarray(shape_deltas), n_theta=28, n_per_segment=5), dtype=np.float64)
        self.R_max, self.H_total, self.sd, self.lovi = R_max, H_total, np.asarray(shape_deltas), handle_r_frac
        self.w, self.lam, self.sigfrac, self.S = w_leak, lam, sigma_frac, int(max_shift_px * UP)
        self.K, self.R, self.t = (np.asarray(pose[k], float) for k in ("K", "R", "t"))
        self.R = self.R.reshape(3, 3); self.t = self.t.reshape(3)

    def _granite_mask(self, frame):
        """Kuten k9.create_granite_mask (sat<60 & tummuus>15, sigma=25), mutta avaus open_size x open_size (oletus 3; k9:ssa 5)."""
        k9 = self.k9
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        dark = cv2.GaussianBlur(gray, (0, 0), sigmaX=k9.STONE_DARKNESS_SIGMA) - gray
        m = ((hsv[:, :, 1] < k9.STONE_MAX_SATURATION) & (dark > k9.STONE_MIN_DARKNESS)).astype(np.uint8) * 255
        if self.open_size > 1:
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((self.open_size, self.open_size), np.uint8))
        return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    def proj(self, x, y, z):
        p = self.K @ (self.R @ np.array([x, y, z]) + self.t)
        return p[0] / p[2], p[1] / p[2]

    def _poly(self, X, Y):
        hull = self.k9._predicted_stone_hull(self.pose, X, Y, self.R_max, self.H_total, self.sd).reshape(-1, 2)
        notch = self.k9._handle_notch_hull(self.pose, X, Y, self.R_max, self.H_total, self.lovi).reshape(-1, 2)
        return hull, notch

    @staticmethod
    def _fill(poly, shape, ox, oy):
        m = np.zeros(shape, np.uint8)
        p = np.round((np.asarray(poly) - np.array([ox, oy]) + 0.5) * UP - 0.5).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(m, [p], 1)
        return m

    def refine(self, frame_bgr, X0, Y0):
        """Palauttaa (X, Y, info) - info: dict(score0, score1, inside0, inside1, shift_px, shift_cm, ok). Epaonnistuessa (X0, Y0, info ok=False).
        C++-polku jos k94 annettu ja stone_tracker.silhouette_refine_cpp olemassa (muuten numpy/cv2-versio refine_py)."""
        if self.body_pts is not None:
            return _st.silhouette_refine_cpp(np.ascontiguousarray(frame_bgr), self.body_pts, self.K, self.R, self.t,
                                             float(self.R_max), float(self.H_total), float(self.lovi), float(X0), float(Y0),
                                             float(self.w), float(self.lam), float(self.sigfrac), int(self.max_shift_px), int(HALF), int(MASK_MARGIN),
                                             int(self.open_size), int(self.band_px))
        return self.refine_py(frame_bgr, X0, Y0)

    def refine_py(self, frame_bgr, X0, Y0):
        H_img, W_img = frame_bgr.shape[:2]
        cx, cy = self.proj(X0, Y0, self.H_total / 2)
        cxi, cyi = int(round(cx)), int(round(cy))
        ox, oy = cxi - HALF, cyi - HALF
        x0c, y0c = max(0, ox - MASK_MARGIN), max(0, oy - MASK_MARGIN)
        x1c, y1c = min(W_img, ox + 2 * HALF + MASK_MARGIN), min(H_img, oy + 2 * HALF + MASK_MARGIN)
        if x1c - x0c < 40 or y1c - y0c < 40 or ox < 0 or oy < 0 or ox + 2 * HALF > W_img or oy + 2 * HALF > H_img:
            return X0, Y0, dict(ok=False)
        gm = self._granite_mask(np.ascontiguousarray(frame_bgr[y0c:y1c, x0c:x1c]))
        mask = gm[oy - y0c:oy - y0c + 2 * HALF, ox - x0c:ox - x0c + 2 * HALF] > 0
        if not mask.any():
            return X0, Y0, dict(ok=False)
        m2 = cv2.resize(mask.astype(np.uint8), None, fx=UP, fy=UP, interpolation=cv2.INTER_NEAREST).astype(np.float32)
        S = self.S
        hull, notch = self._poly(X0, Y0)
        shape = m2.shape
        mh = self._fill(hull, shape, ox, oy) > 0
        mn = self._fill(notch, shape, ox, oy) > 0
        sil = mh & ~mn
        a = float(sil.sum())
        if a < 10:
            return X0, Y0, dict(ok=False)
        rows = sil.any(axis=1)
        free = (~(np.cumsum(mh[:, ::-1], axis=1)[:, ::-1] > 0)) & rows[:, None]
        kb = 2 * self.band_px * UP + 1
        out = (~mh) & (~free) & (cv2.dilate(mh.astype(np.uint8), np.ones((kb, kb), np.uint8)) > 0)
        yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
        big = np.zeros((shape[0] + 2 * S, shape[1] + 2 * S), np.float32)
        big[S:S + shape[0], S:S + shape[1]] = m2

        def corr(tpl):
            return cv2.matchTemplate(big, tpl.astype(np.float32), cv2.TM_CCORR)        # (2S+1, 2S+1), [iy, ix]
        inside = corr(sil) / a
        leak = corr(out) / a
        cnt = corr(mh)
        sx, sy = corr(mh * xx), corr(mh * yy)
        ys_, xs_ = np.nonzero(sil)
        with np.errstate(divide="ignore", invalid="ignore"):
            mcx, mcy = sx / cnt, sy / cnt
            d = np.hypot(mcx - xs_.mean(), mcy - ys_.mean()) / UP
            sig = self.sigfrac * np.sqrt(a / (UP * UP))
            bonus = np.where(cnt > 0.5, self.lam * np.exp(-(d / sig) ** 2), 0.0)
        score = (inside - self.w * leak + bonus).astype(np.float64)
        # tasapeli (plateau): pienin siirto (vakaa tulos, ei riipu lukutarkkuudesta) - sama kuin C++-versiossa
        cand = np.argwhere(score >= score.max() - 1e-7)
        iy, ix = min(cand, key=lambda q: (q[1] - S) ** 2 + (q[0] - S) ** 2)
        du, dv = (ix - S) / UP, (iy - S) / UP                      # siluetin siirto kuvassa (px)
        s0, s1 = float(score[S, S]), float(score[iy, ix])
        ins0, ins1 = float(inside[S, S]), float(inside[iy, ix])
        if du == 0 and dv == 0:
            return X0, Y0, dict(ok=True, score0=s0, score1=s1, shift_px=0.0, shift_cm=0.0, inside0=ins0, inside1=ins1)
        # kuvasiirto -> maatason siirto (Jacobi numeerisesti)
        u0, v0 = self.proj(X0, Y0, self.H_total / 2)
        ux, vx = self.proj(X0 + 1.0, Y0, self.H_total / 2)
        uy, vy = self.proj(X0, Y0 + 1.0, self.H_total / 2)
        J = np.array([[ux - u0, uy - u0], [vx - v0, vy - v0]])
        try:
            dXY = np.linalg.solve(J, np.array([du, dv]))
        except np.linalg.LinAlgError:
            return X0, Y0, dict(ok=False)
        return X0 + float(dXY[0]), Y0 + float(dXY[1]), dict(ok=True, score0=s0, score1=s1, shift_px=float(np.hypot(du, dv)),
                                                          shift_cm=float(np.hypot(*dXY)), inside0=ins0, inside1=ins1)
