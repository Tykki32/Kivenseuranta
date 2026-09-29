"""Yhden kiven peräkkäinen seurantasimulaattori (Testi_02_03).

Toistaa main.py:n SEURANTA-logiikan (hakualue nopeusrajasta, "ei taaksepain"-tarkistus, missit,
"pysahtynyt"-saanto) yhdelle kivelle tallennetuilla frameilla (g<frame>.npz, katso main.py:n
TRACKER_FRAME_DUMP_*) ja vertaa rataa oikeaan (jalkikateen tunnettuun) rataan.

Kaytto (moduulina):
    from sim_track import Sim
    sim = Sim("/tmp/w/fdump", "/tmp/w/mah_base/calib_profile.pkl")
    res = sim.run(start_frame, start_xy, ref_track, variant_dict)
"""
import os, sys, math, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ms_offline import Ctx


class Sim(Ctx):
    def __init__(self, fdump_dir, pkl_path):
        super().__init__(fdump_dir, pkl_path)
        M = self.M
        self.fps = 25.02219425706555
        self.vy, self.vx = M.TRACK_MAX_SPEED_Y_CM_S, M.TRACK_MAX_SPEED_X_CM_S
        self.hmax = M.TRACK_HALF_RANGE_MAX_CM
        self.back = M.TRACK_MAX_BACKWARD_CM
        self.maxabsx = M.MAX_ABS_X_FROM_CENTERLINE_CM
        self.lost_max = max(M.k92.TRACK_LOST_MAX_MISSES, int(round(self.fps * M.TRACK_LOST_GRACE_SECONDS)))
        self.stop_frames = max(1, int(round(self.fps * M.STOP_TRACKING_SECONDS)))
        self.stop_disp = M.STOP_TRACKING_DISPLACEMENT_CM
        self._fr = {}

    def frame(self, f):
        if f not in self._fr:
            if len(self._fr) > 30:
                self._fr.clear()
            self._fr[f] = np.load(os.path.join(self.dump_dir, f"g{f:06d}.npz"))["frame"]
        return self._fr[f]

    def step(self, f, X0, Y0, hx, hy, kw, extra=None):
        fr = self.frame(f)
        args = [fr, self.ref, np.array([X0]), np.array([Y0]), np.array([hx]), np.array([hy]),
                self.body, self.search, self.pose["K"], self.pose["R"], self.pose["t"],
                self.coarse, self.fine, self.thr, self.R_max, self.H_total, self.ring, self.ring,
                self.back, 0.0]
        return self.st.track_stones_batch(*args, **kw)[0]

    def velocity(self, track, f, misses, lookback=12, min_gap=4):
        """Kiven nopeus (cm/frame) viimeisista havainnoista: kaukaisin havainto lookback-ikkunassa
        (vahintaan min_gap framea taaksepain) -> vakionopeus. None jos ei tarpeeksi dataa."""
        fr = [k for k in track if f - lookback <= k <= f - 1 - misses]
        if len(fr) < 2:
            return None
        k0, k1 = min(fr), max(fr)
        if k1 - k0 < min_gap:
            return None
        return ((track[k1][0] - track[k0][0]) / (k1 - k0), (track[k1][1] - track[k0][1]) / (k1 - k0))

    def run(self, start_frame, start_xy, ref, kw, last_frame=None, use_stop=True, pred=False,
            pred_gain=1.0, lookback=12):
        """ref: {frame: (x_cm, y_cm)}. Palauttaa dictin: rata {frame:(x,y)}, tapahtumat."""
        X, Y = start_xy
        last = (X, Y)
        min_y = Y
        misses = 0
        hist = [(start_frame, X, Y)]
        track = {start_frame: (X, Y)}
        ev = None
        f = start_frame + 1
        end = last_frame or max(int(os.path.basename(p)[1:7]) for p in glob.glob(os.path.join(self.dump_dir, "g*.npz")))
        while f <= end and os.path.exists(os.path.join(self.dump_dir, f"g{f:06d}.npz")):
            el = (misses + 1) / self.fps
            hx = min(self.vx * el, self.hmax)
            hy = min(self.vy * el, self.hmax)
            kw2 = dict(kw)
            if pred:
                v = self.velocity(track, f, misses, lookback)
                if v is not None:
                    n = misses + 1
                    kw2["pred_dx"] = np.array([np.clip(v[0] * n * pred_gain, -hx, hx)])
                    kw2["pred_dy"] = np.array([np.clip(v[1] * n * pred_gain, -hy, hy)])
            r = self.step(f, last[0], last[1], hx, hy, kw2)
            found = r["found"]
            if found and (r["Y_cm"] - min_y > self.back):
                found = False
            if found and abs(r["X_cm"]) > self.maxabsx:
                found = False
            if found:
                last = (r["X_cm"], r["Y_cm"])
                min_y = min(min_y, r["Y_cm"])
                misses = 0
                track[f] = last
                hist.append((f, last[0], last[1]))
                while hist[-1][0] - hist[0][0] > self.stop_frames:
                    hist.pop(0)
                if use_stop and hist[-1][0] - hist[0][0] >= self.stop_frames:
                    d = math.hypot(last[0] - hist[0][1], last[1] - hist[0][2])
                    if d < self.stop_disp:
                        ev = ("stop", f)
                        break
            else:
                misses += 1
                if misses >= self.lost_max:
                    ev = ("lost", f)
                    break
            f += 1
        return dict(track=track, event=ev, end_frame=f)


def evaluate(track, ref, tol=30.0):
    """Peittoprosentti: osuus ref-framista joissa rata on <tol cm paassa."""
    fr = [f for f in ref if f in track or True]
    hit = tot = 0
    err = []
    first_bad = None
    for f in sorted(ref):
        tot += 1
        if f in track:
            e = math.hypot(track[f][0] - ref[f][0], track[f][1] - ref[f][1])
            err.append(e)
            if e < tol:
                hit += 1
            elif first_bad is None:
                first_bad = f
        elif first_bad is None:
            first_bad = f
    return dict(coverage=hit / max(1, tot), n=tot, err_med=float(np.median(err)) if err else None, first_bad=first_bad)
