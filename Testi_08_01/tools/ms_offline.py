"""Offline-virityskehys mean-shift-seurannalle (Testi_02_03).

Lataa TRACKER_DUMP_DIR:iin tallennetut seurantaframet (fNNNNNN.npz) sekä
kalibroinnin/profiilin (calib_profile.pkl) ja ajaa stone_tracker.track_stones_
batch:in eri hakutiloilla samoilla syotteilla - ilman koko videon lapikaymista.

Kaytto (esimerkki):
    from ms_offline import Ctx
    ctx = Ctx("/tmp/w/dump", "/tmp/w/mah_base/calib_profile.pkl")
    res = ctx.run(frame_idx, mode=2, gain=1.0)
"""
import os, sys, glob, pickle, types, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
sys.path.insert(0, CODE)
for n in ("tkinter", "tkinter.filedialog"):
    if n not in sys.modules:
        sys.modules[n] = types.ModuleType(n)
sys.modules["tkinter"].filedialog = sys.modules["tkinter.filedialog"]


class Ctx:
    def __init__(self, dump_dir, pkl_path):
        import main as M
        import stone_tracker
        self.M, self.st = M, stone_tracker
        pk = pickle.load(open(pkl_path, "rb"))
        prof, self.pose, calib = pk["profile"], pk["pose"], pk["calib"]
        k94, k92 = M.k94, M.k92
        self.R_max, self.H_total = prof["R_max_cm"], prof["H_total_cm"]
        self.body = k94.build_local_stone_rings(self.R_max, self.H_total, prof["shape_deltas"], n_theta=28, n_per_segment=5)
        self.search = k94.build_local_stone_rings(self.R_max, self.H_total, prof["shape_deltas"],
                                                  n_theta=k92.SEARCH_HULL_N_THETA,
                                                  n_per_segment=k92.SEARCH_HULL_N_PER_SEGMENT)
        self.ring = prof["handle_r_frac"]
        self.ref = calib["frame_undistorted"]
        self.coarse, self.fine, self.thr = k92.TRACK_COARSE_STEP_CM, k92.TRACK_FINE_STEP_CM, k92.TRACK_SCORE_THRESHOLD
        self.back = M.TRACK_MAX_BACKWARD_CM
        self.files = sorted(glob.glob(os.path.join(dump_dir, "f*.npz")))
        self.dump_dir = dump_dir
        self._cache = {}

    def load(self, path):
        if path not in self._cache:
            if len(self._cache) > 40:
                self._cache.clear()
            self._cache[path] = dict(np.load(path))
        return self._cache[path]

    def run(self, path, mode=0, gain=1.0, max_iter=10, tol=0.10, w_in=2.0, margin=1.10,
            coarse=None, fine=None, hx=None, hy=None, tau=0.0, polish=0.0):
        d = self.load(path)
        r = self.st.track_stones_batch(
            d["frame"], self.ref, d["X0"], d["Y0"],
            d["hx"] if hx is None else hx, d["hy"] if hy is None else hy,
            self.body, self.search, self.pose["K"], self.pose["R"], self.pose["t"],
            self.coarse if coarse is None else coarse, self.fine if fine is None else fine,
            self.thr, self.R_max, self.H_total, self.ring, self.ring, self.back,
            float(d["diff_threshold"]),
            locate_mode=mode, ms_gain=gain, ms_max_iter=max_iter, ms_tol_px=tol,
            ms_inner_weight=w_in, ms_margin_scale=margin, ms_tau=tau, ms_polish_step_cm=polish)
        return d, r
