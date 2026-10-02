"""Aikamittausajo: python3 tools/run_profile.py <video> <kalib_pkl> <ulostulokansio> [max_frame]
Ajaa run_pipeline:n valmiilla kalibroinnilla ja pysähtyy MAX_FRAME:n kohdalla (oletus 2900),
tulostaa vaihekohtaisen aikaraportin (Python + C++)."""
import sys, types, os, time, pickle
for n in ("tkinter", "tkinter.filedialog"):
    sys.modules[n] = types.ModuleType(n)
sys.modules["tkinter"].filedialog = sys.modules["tkinter.filedialog"]
video, pkl, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
os.environ["MAX_FRAME"] = sys.argv[4] if len(sys.argv) > 4 else "2900"
code = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, code); os.chdir(code)
import main as M
panel_data = M.load_panel_data(video)
os.makedirs(outdir, exist_ok=True)
pk = pickle.load(open(pkl, "rb"))
calib_result = {"calib": pk["calib"], "pose": pk["pose"]}
t = time.time()
M.run_pipeline(video, panel_data, outdir + "/calib.png", outdir + "/stones.csv",
               precomputed_calib_result=calib_result, precomputed_profile_result=pk["profile"])
print("DONE", time.time() - t)
