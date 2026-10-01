"""Kalibrointi + profiilin opettelu videon alusta (ei valmiiksi laskettuja): python3 tools/run_calib.py <video> <ulostulokansio> [max_frame]
Ajaa run_pipeline:n MAX_FRAME:een asti ja tulostaa lokista profiili-/opettelurivit."""
import sys, types, os, time
for n in ("tkinter", "tkinter.filedialog"):
    sys.modules[n] = types.ModuleType(n)
sys.modules["tkinter"].filedialog = sys.modules["tkinter.filedialog"]
video, outdir = sys.argv[1], sys.argv[2]
os.environ["MAX_FRAME"] = sys.argv[3] if len(sys.argv) > 3 else "3600"
code = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, code); os.chdir(code)
import main as M
import functools, collections
_T = collections.defaultdict(lambda: [0.0, 0])
def _wrap(name):
    fn = getattr(M, name)
    @functools.wraps(fn)
    def w(*a, **k):
        t0 = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            e = _T[name]; e[0] += time.perf_counter() - t0; e[1] += 1
    if name == "try_fit_profile":
        @functools.wraps(fn)
        def w(pose, stones, *a, **k):
            t0 = time.perf_counter()
            try:
                return fn(pose, stones, *a, **k)
            finally:
                print(f"KUTSU try_fit_profile: {len(stones)} havaintoa, {time.perf_counter() - t0:.1f} s", flush=True)
    setattr(M, name, w)
for _n in ("track_stone_in_video_windowed", "_scan_stone_candidates", "calibrate_camera_from_image_with_seed",
           "try_fit_profile", "build_stone_color_reference", "find_moving_candidate"):
    _wrap(_n)
panel_data = M.load_panel_data(video)
os.makedirs(outdir, exist_ok=True)
t = time.time()
res = M.run_pipeline(video, panel_data, outdir + "/calib.png", outdir + "/stones.csv")
p = res["profile"]
print("PROFIILI:", None if p is None else {k: p[k] for k in ("R_max_cm", "H_total_cm", "handle_r_frac", "residual_rms_px")}, "havaintoja", res["n_profile_observations"])
for k, (sec, n) in sorted(_T.items(), key=lambda kv: -kv[1][0]): print(f"AIKA {k:45s} {sec:8.1f} s  kutsuja {n}")
print("DONE", time.time() - t)
