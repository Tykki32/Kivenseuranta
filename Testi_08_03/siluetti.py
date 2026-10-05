"""Kiven paikan siluettitarkennus graniittimaskista (C++: stone_tracker.silhouette_refine_cpp).

Ehdokkaan (X, Y) paikka tarkennetaan siirtamalla kiven 3D-mallin siluettia KUVATASOSSA (+-max_shift_px, 0,5 px askel)
ja valitsemalla siirto, jolla maskiin osuu parhaiten:
  + sisalla:  maskipikselit siluetin (kupera peite MINUS kahvan lovi) sisalla / siluetin ala            (palkinto)
  - ylitulo:  maskipikselit peitteen ULKOPUOLELLA / ala * W_LEAK, paitsi kiven ylapuolella             (rangaistus)
  + keskitys: LAM * exp(-(d/sigma)^2), d = maskin ja siluetin massakeskipisteiden etaisyys, sigma = SIGMA_FRAC*sqrt(ala)
Ylitysrangaistus lasketaan vain siluetin ymparilla olevalla BAND_PX-kaistalla; maski on oma (3x3-avaus: 5x5 poisti
kaukaisen, 5-6 px leveän kiven). Paras siirto muutetaan maatasoon (X, Y) Jacobin avulla.

HAKU: jokainen uusi ehdokas tarkennetaan (refine); SEURANTA: sama laskenta kivisaikeissa C++:ssa
(batch_config -> stone_tracker.set_seuranta_silhouette).
"""
import numpy as np

import stone_tracker

import kivimalli

W_LEAK = 1.0
LAM = 0.1
SIGMA_FRAC = 0.3
MAX_SHIFT_PX = 6
HALF = 26            # ikkunan puolikas (px)
MASK_MARGIN = 75     # graniittimaskin sumennuksen reunus (px)
OPEN_SIZE = 3        # maskin avaus
BAND_PX = 3          # ylitysrangaistuksen kaista siluetin ymparilla (px): kaukainen tumma kohde (heittaja) ei veda siluettia


class SilhouetteRefiner:
    def __init__(self, pose, R_max, H_total, shape_deltas, handle_r_frac, max_shift_px=MAX_SHIFT_PX,
                 open_size=OPEN_SIZE, band_px=BAND_PX):
        self.open_size, self.band_px = int(open_size), int(band_px)
        self.max_shift_px = max_shift_px
        self.body_pts = np.ascontiguousarray(
            kivimalli.build_local_stone_rings(R_max, H_total, np.asarray(shape_deltas), n_theta=28, n_per_segment=5),
            dtype=np.float64,
        )
        self.R_max, self.H_total, self.lovi = R_max, H_total, handle_r_frac
        K, R, t = (np.asarray(pose[k], float) for k in ("K", "R", "t"))
        self.K, self.R, self.t = K, R.reshape(3, 3), t.reshape(3)

    def refine(self, frame_bgr, X0, Y0):
        """Palauttaa (X, Y, info): info = dict(score0, score1, inside0, inside1, shift_px, shift_cm, ok)."""
        return stone_tracker.silhouette_refine_cpp(
            np.ascontiguousarray(frame_bgr), self.body_pts, self.K, self.R, self.t, float(self.R_max),
            float(self.H_total), float(self.lovi), float(X0), float(Y0), float(W_LEAK), float(LAM), float(SIGMA_FRAC),
            int(self.max_shift_px), int(HALF), int(MASK_MARGIN), int(self.open_size), int(self.band_px),
        )

    def refine_many(self, frame_bgr, xys):
        """refine() usealle (X0, Y0) -parille samasta ruudusta (C++, kivet rinnan). Lista (X, Y, info)."""
        if not xys:
            return []
        xs = np.ascontiguousarray([float(x) for x, _ in xys], dtype=np.float64)
        ys = np.ascontiguousarray([float(y) for _, y in xys], dtype=np.float64)
        return list(stone_tracker.silhouette_refine_batch_cpp(
            np.ascontiguousarray(frame_bgr), self.body_pts, self.K, self.R, self.t, float(self.R_max),
            float(self.H_total), float(self.lovi), xs, ys, float(W_LEAK), float(LAM), float(SIGMA_FRAC),
            int(self.max_shift_px), int(HALF), int(MASK_MARGIN), int(self.open_size), int(self.band_px),
        ))

    def batch_config(self, half=HALF, margin=MASK_MARGIN):
        """Argumentit stone_tracker.set_seuranta_silhouette:lle (SEURANTA laskee saman kivisaikeissa)."""
        return (self.body_pts, np.ascontiguousarray(self.K, dtype=np.float64), np.ascontiguousarray(self.R, dtype=np.float64),
                np.ascontiguousarray(self.t, dtype=np.float64), float(self.R_max), float(self.H_total), float(self.lovi),
                float(W_LEAK), float(LAM), float(SIGMA_FRAC), int(self.max_shift_px), int(half), int(margin),
                int(self.open_size), int(self.band_px))
