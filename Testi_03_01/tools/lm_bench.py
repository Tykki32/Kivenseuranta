"""LM-nopeusvertailu offline-dumpeilla: python3 tools/lm_bench.py <dump_dir> <pkl> <out.pkl> [n_frames] [step]
Ajaa track_stones_batch (mean-shift) dumpatuille frameille, tallentaa tulokset ja tulostaa vaiheajat."""
import sys, os, pickle, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ms_offline import Ctx
dump, pkl, out = sys.argv[1:4]
n = int(sys.argv[4]) if len(sys.argv) > 4 else 300
step = int(sys.argv[5]) if len(sys.argv) > 5 else 3
ctx = Ctx(dump, pkl)
off = int(sys.argv[6]) if len(sys.argv) > 6 else 0
files = ctx.files[off::step][:n]
ctx.st.prof_reset()
res = []
t = time.time()
for f in files:
    if os.environ.get("BENCH_MODE") == "5":
        d = ctx.load(f)
        r = ctx.st.track_stones_batch(
            d["frame"], ctx.ref, d["X0"], d["Y0"], d["hx"], d["hy"], ctx.body, ctx.search,
            ctx.pose["K"], ctx.pose["R"], ctx.pose["t"], ctx.coarse, ctx.fine, ctx.thr, ctx.R_max, ctx.H_total,
            ctx.ring, ctx.ring, ctx.back, float(d["diff_threshold"]),
            locate_mode=5, ms_gain=0.8, ms_max_iter=6, ms_tol_px=0.3, ms_tau=0.5, ms_polish_step_cm=4.0,
            pred_dx=np.zeros(len(d["X0"])), pred_dy=np.full(len(d["X0"]), 8.0),
            ens_back_tol_cm=2.0, ens_back_pen=0.15, ens_pred_pen=0.02)
    else:
        d, r = ctx.run(f, mode=2, gain=0.8, max_iter=6, tol=0.3, tau=0.5, polish=4.0)
    res.append([(x["found"], x.get("X_cm"), x.get("Y_cm"), x.get("rms_px")) for x in r])
wall = time.time() - t
pickle.dump(res, open(out, "wb"))
snap = {k.strip(): (ms, c) for k, ms, c in ctx.st.prof_snapshot()}
tot = sum(len(r) for r in res)
print(f"frames {len(files)} stones {tot} wall {wall:.2f}s")
for k in ["LM-tarkennus yhteensa", "LM: sovitus #1", "LM: sovitus #2 (uusinta)", "LM: reunapisteet (detectBoundaryPoints)",
          "LM: runkokontuuri + jaasuodatus", "LM: MAD-poikkeamat", "LM: lopullinen residuaali", "granite: cvtColor x2 + convert", "granite: GaussianBlur", "granite: kynnys-silmukka", "granite: morfologia", "ristikko: predictedHull", "ristikko: hullOverlapScore", "LM: TARKKA VARAKEINO-LM (linearisointi hylatty)", "LM#1: lineaarinen LM", "LM#1: tarkka varmistus", "LM: linearisoinnin rakennus", "LM: residuaali+jacobi evaluaatiot"]:
    ms, c = snap[k]; print(f"{k:45s} {ms/max(tot,1):8.3f} ms/kivi  n={c}  {ms/max(c,1):.3f} ms/kutsu")
for k in snap:
    if k.startswith("LASKURI"): print(f"{k}: {snap[k][0] / max(snap[k][1], 1):.1f}")
