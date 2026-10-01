"""Kaikki C++-vaihemittaukset ms/kivipaivitys offline-dumpeilla: python3 tools/bench_all.py <dump> <pkl> [n] [step] [offset]"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from ms_offline import Ctx
dump, pkl = sys.argv[1:3]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 400; step = int(sys.argv[4]) if len(sys.argv) > 4 else 1; off = int(sys.argv[5]) if len(sys.argv) > 5 else 0
ctx = Ctx(dump, pkl); files = ctx.files[off::step][:n]
ctx.st.prof_reset(); tot = 0
for f in files:
    d = ctx.load(f)
    r = ctx.st.track_stones_batch(d["frame"], ctx.ref, d["X0"], d["Y0"], d["hx"], d["hy"], ctx.body, ctx.search,
        ctx.pose["K"], ctx.pose["R"], ctx.pose["t"], ctx.coarse, ctx.fine, ctx.thr, ctx.R_max, ctx.H_total, ctx.ring, ctx.ring, ctx.back,
        float(d["diff_threshold"]), locate_mode=5, ms_gain=0.8, ms_max_iter=6, ms_tol_px=0.3, ms_tau=0.5, ms_polish_step_cm=4.0,
        pred_dx=np.zeros(len(d["X0"])), pred_dy=np.full(len(d["X0"]), 8.0), ens_back_tol_cm=2.0, ens_back_pen=0.15, ens_pred_pen=0.02)
    tot += len(r)
print("kivipaivityksia", tot)
for name, ms, c in ctx.st.prof_snapshot():
    if c and not name.startswith("HAKU") and "LASKURI" not in name: print(f"{name:60s} {ms/tot:8.3f} ms/kivi  n={c}")
