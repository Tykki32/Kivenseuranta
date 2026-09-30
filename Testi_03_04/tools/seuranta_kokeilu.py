"""SEURANTA-kokeiluymparisto (Testi_03_04): toistaa YHDEN kiven seurannan (live-logiikka: nopeusrajattu hakualue, liike-ennuste, 'ei taaksepain',
missit, pysahtynyt-saanto) tallennetuilla live-ruuduilla (g<frame>.npz = frame_u_for_tracking, katso HAKU_KOKEILU.md) eri parametreilla.

    import sys; sys.path.insert(0, "tools")
    from seuranta_kokeilu import TrackLab
    lab = TrackLab("/tmp/w/hakuraw", "/tmp/w/mah_base/calib_profile.pkl", "/tmp/w/full04_base/stones.csv")
    ref = lab.reference(start_frame=2280, frames=range(2240, 2300))           # vertailurata: myohemman siistin radan ekstrapolaatio taaksepain
    res = lab.replay(2240, (-4.6, 3294.7), 2300)                                # live-parametrit
    lab.report(res, ref)
Parametrit (kwargs replay:lle): mode (ensemble/grid/meanshift), coarse, fine, score_thr, max_speed_y, pred (True/False), ms_*/ens_* ym.
"""
import os, sys, glob, math, pickle, types, csv, collections
for n in ("tkinter", "tkinter.filedialog"):
    if n not in sys.modules:
        sys.modules[n] = types.ModuleType(n)
sys.modules["tkinter"].filedialog = sys.modules["tkinter.filedialog"]
import numpy as np

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE)
FPS = 25.02219425706555


class TrackLab:
    def __init__(self, dump_dir, pkl_path, stones_csv=None):
        import main as M
        import stone_tracker
        self.M, self.st = M, stone_tracker
        pk = pickle.load(open(pkl_path, "rb"))
        prof, self.pose, calib = pk["profile"], pk["pose"], pk["calib"]
        k94, self.k92 = M.k94, M.k92
        self.R_max, self.H_total = prof["R_max_cm"], prof["H_total_cm"]
        self.body = k94.build_local_stone_rings(self.R_max, self.H_total, prof["shape_deltas"], n_theta=28, n_per_segment=5)
        self.search = k94.build_local_stone_rings(self.R_max, self.H_total, prof["shape_deltas"],
                                                  n_theta=self.k92.SEARCH_HULL_N_THETA, n_per_segment=self.k92.SEARCH_HULL_N_PER_SEGMENT)
        self.ring = prof["handle_r_frac"]; self.ref_img = calib["frame_undistorted"]
        self.dump_dir = dump_dir; self._cache = {}
        self.rows = collections.defaultdict(list)
        if stones_csv:
            for r in csv.DictReader(open(stones_csv)):
                try: self.rows[r["stone_id"]].append((int(r["frame"]), float(r["x_m"]) * 100, float(r["y_m"]) * 100, float(r["rms_px"]) if r["rms_px"] else np.nan))
                except ValueError: pass

    def frame(self, f):
        if f not in self._cache:
            if len(self._cache) > 60: self._cache.clear()
            self._cache[f] = np.load(os.path.join(self.dump_dir, f"g{f:06d}.npz"))["frame"]
        return self._cache[f]

    def has(self, f): return os.path.exists(os.path.join(self.dump_dir, f"g{f:06d}.npz"))

    # ---- vertailurata: CSV:n siisti osa (rms < 2) + ekstrapolaatio taaksepain neliollisella sovituksella ----
    def reference(self, stone_key, frames, fit_rows=70):
        rs = [r for r in self.rows[stone_key] if r[3] < 2.0][:fit_rows]
        f = np.array([r[0] for r in rs], float); x = np.array([r[1] for r in rs]); y = np.array([r[2] for r in rs])
        px, py = np.polyfit(f - f[0], x, 2), np.polyfit(f - f[0], y, 2)
        ref = {int(ff): (float(np.polyval(px, ff - f[0])), float(np.polyval(py, ff - f[0]))) for ff in frames}
        res_fit = np.hypot(np.polyval(px, f - f[0]) - x, np.polyval(py, f - f[0]) - y)
        return ref, dict(fit_rms_cm=float(np.sqrt(np.mean(res_fit ** 2))), first_clean_frame=int(f[0]), n_fit=len(rs))

    # ---- seuranta ----
    def _step(self, f, X0, Y0, hx, hy, kw, pred):
        M = self.M
        kwargs = dict(locate_mode=M._TRACKER_MODE_ID[kw.get("mode", "ensemble")], ms_gain=kw.get("ms_gain", M.MS_GAIN), ms_max_iter=kw.get("ms_max_iter", M.MS_MAX_ITER),
                      ms_tol_px=kw.get("ms_tol_px", M.MS_TOL_PX), ms_inner_weight=kw.get("ms_inner_weight", M.MS_INNER_WEIGHT), ms_margin_scale=kw.get("ms_margin_scale", M.MS_MARGIN),
                      ms_tau=kw.get("ms_tau", M.MS_TAU), ms_polish_step_cm=kw.get("ms_polish", M.MS_POLISH_CM), ens_back_tol_cm=kw.get("ens_back_tol", M.ENS_BACK_TOL_CM),
                      ens_back_pen=kw.get("ens_back_pen", M.ENS_BACK_PEN), ens_pred_pen=kw.get("ens_pred_pen", M.ENS_PRED_PEN))
        if pred is not None and kw.get("mode", "ensemble") == "ensemble":
            kwargs["pred_dx"], kwargs["pred_dy"] = np.array([pred[0]]), np.array([pred[1]])
        args = [self.frame(f), self.ref_img, np.array([X0]), np.array([Y0]), np.array([hx]), np.array([hy]), self.body, self.search,
                self.pose["K"], self.pose["R"], self.pose["t"], kw.get("coarse", self.k92.TRACK_COARSE_STEP_CM), kw.get("fine", self.k92.TRACK_FINE_STEP_CM),
                kw.get("score_thr", self.k92.TRACK_SCORE_THRESHOLD), self.R_max, self.H_total, self.ring, self.ring, M.TRACK_MAX_BACKWARD_CM, 0.0]
        return self.st.track_stones_batch(*args, **kwargs)[0]

    def replay(self, start_frame, start_xy, last_frame, use_stop=True, pred=True, **kw):
        M = self.M
        vy = kw.get("max_speed_y", M.TRACK_MAX_SPEED_Y_CM_S); vx = vy * M.TRACK_MAX_SPEED_X_FRACTION_OF_Y; hmax = kw.get("half_max", M.TRACK_HALF_RANGE_MAX_CM)
        lost_max = max(self.k92.TRACK_LOST_MAX_MISSES, int(round(FPS * M.TRACK_LOST_GRACE_SECONDS)))
        stop_frames = max(1, int(round(FPS * M.STOP_TRACKING_SECONDS)))
        X, Y = start_xy; last = (X, Y); min_y = Y; misses = 0
        hist = [(start_frame, X, Y)]; pos_hist = [(start_frame, X, Y)]
        recs = {start_frame: dict(X=X, Y=Y, found=True, rms=np.nan, tarkka=np.nan, score=np.nan)}; ev = None; f = start_frame + 1
        while f <= last_frame and self.has(f):
            el = (misses + 1) / FPS
            hx, hy = min(vx * el, hmax), min(vy * el, hmax); pr = None
            if pred:
                h = [q for q in pos_hist if f - M.PRED_LOOKBACK <= q[0] <= f - 1 - misses]
                if len(h) >= 2 and h[-1][0] - h[0][0] >= M.PRED_MIN_SPAN:
                    n = misses + 1; dx = (h[-1][1] - h[0][1]) / (h[-1][0] - h[0][0]) * n; dy = (h[-1][2] - h[0][2]) / (h[-1][0] - h[0][0]) * n
                    pr = (max(-hx, min(hx, dx)), max(-hy, min(hy, dy)))
            r = self._step(f, last[0], last[1], hx, hy, kw, pr)
            found = bool(r["found"])
            if found and (r["Y_cm"] - min_y > M.TRACK_MAX_BACKWARD_CM): found = False
            if found and abs(r["X_cm"]) > M.MAX_ABS_X_FROM_CENTERLINE_CM: found = False
            nz = lambda v: np.nan if v is None else v
            recs[f] = dict(X=nz(r.get("X_cm")), Y=nz(r.get("Y_cm")), found=found, rms=nz(r.get("rms_px")), tarkka=int(bool(r.get("tarkka"))),
                           score=nz(r.get("score")), locX=nz(r.get("loc_X")), locY=nz(r.get("loc_Y")), n_body=r.get("n_body") or 0, hx=hx, hy=hy, misses=misses,
                           pred=pr)
            if found:
                last = (r["X_cm"], r["Y_cm"]); min_y = min(min_y, r["Y_cm"]); misses = 0
                pos_hist.append((f, last[0], last[1])); hist.append((f, last[0], last[1]))
                while hist[-1][0] - hist[0][0] > stop_frames: hist.pop(0)
                if use_stop and hist[-1][0] - hist[0][0] >= stop_frames and math.hypot(last[0] - hist[0][1], last[1] - hist[0][2]) < M.STOP_TRACKING_DISPLACEMENT_CM:
                    ev = ("pysahtynyt", f); break
            else:
                misses += 1
                if misses >= lost_max: ev = ("kadotettu", f); break
            f += 1
        return dict(recs=recs, event=ev, end=f)

    @staticmethod
    def errors(res, ref):
        out = {}
        for f, r in res["recs"].items():
            if f in ref and r.get("found"):
                out[f] = math.hypot(r["X"] - ref[f][0], r["Y"] - ref[f][1])
        return out

    def report(self, res, ref, step=2):
        er = self.errors(res, ref)
        print(f"{'ruutu':>6} {'X':>7} {'Y':>8} {'ref Y':>8} {'virhe cm':>8} {'rms':>6} {'tarkka':>6} {'score':>6} {'loc-X':>7} {'loc-Y':>8} {'hy':>5}")
        for f in sorted(res["recs"]):
            if (f - min(res["recs"])) % step: continue
            r = res["recs"][f]
            print(f"{f:6d} {r['X']:7.1f} {r['Y']:8.1f} {ref.get(f, (0, float('nan')))[1]:8.1f} {er.get(f, float('nan')):8.1f} {r.get('rms', float('nan')):6.2f} {r.get('tarkka', 0)!s:>6} {r.get('score', float('nan')):6.2f} {r.get('locX', float('nan')):7.1f} {r.get('locY', float('nan')):8.1f} {r.get('hy', 0):5.1f}")
        print("tapahtuma:", res["event"], " | virhe ka %.1f cm, mediaani %.1f (n=%d)" % (np.mean(list(er.values())) if er else float('nan'), np.median(list(er.values())) if er else float('nan'), len(er)))
