"""Testi_03_04: kivi kerrallaan profiilin opettelu JULKAISTULLA koodilla (main.py:n alfa-funktiot + try_fit_profile).

Kaytto: python3 tools/alfa_profiili_testi.py <video> <calib_profile.pkl> <stones.csv> <ulostulokansio> [ikkuna_s=15]

Jokaiselle kivelle (live-CSV:n heitot + 2 profiilinopettelun aikaista heittoa):
  1) seuranta (track_stone_in_video_windowed) -> PROFILE_SAMPLES_PER_STONE havaintoa (kuten run_pipeline)
  2) attach_alpha_observations (hylkayssaannot: suhde, tumma alue) + apply_area_rule
  3) yhteinen alfa-taso = KAIKKIEN kivien hyvaksyttyjen havaintojen suhteen mediaani
  4) kivi YKSIN: try_fit_profile alfa-aariviivalla (sama portti kuin putkessa: rms <= PROFILE_MAX_RMS_PX, R PROFILE_R_MAX_MIN..MAX)
     + vertailuna seurannan oma aariviiva samoilla havainnoilla (vanha tapa)
"""
import sys, os, csv, types, time, pickle, collections
for n in ("tkinter", "tkinter.filedialog"):
    sys.modules[n] = types.ModuleType(n)
sys.modules["tkinter"].filedialog = sys.modules["tkinter.filedialog"]
import numpy as np
video, pkl, csv_in, outdir = sys.argv[1:5]
window_s = float(sys.argv[5]) if len(sys.argv) > 5 else 15.0
code = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, code); os.chdir(code)
import main as M
os.makedirs(outdir, exist_ok=True)
pk = pickle.load(open(pkl, "rb")); calib, pose = pk["calib"], pk["pose"]

rows = collections.defaultdict(list)
for r in csv.DictReader(open(csv_in)):
    try:
        rows[int(r["stone_id"])].append((int(r["frame"]), float(r["x_m"]) * 100, float(r["y_m"]) * 100))
    except ValueError:
        pass
seeds = {}
for sid, rr in rows.items():
    rr.sort(); inside = [x for x in rr if 450 <= x[2] <= 1950]
    if inside:
        s = min(inside, key=lambda x: abs(x[2] - 1400)); seeds[f"{sid}"] = (rr[0][0], s[0], s[1], s[2])
seeds["E1_heitto1810"] = (1810, 2080, 54.0, 1304.0)      # profiilinopettelun aikaiset heitot (eivat live-CSV:ssa)
seeds["E2_heitto2025"] = (2025, 2220, -47.0, 1732.0)

cache_f = os.path.join(outdir, "alfa_havainnot.pkl")
cache = pickle.load(open(cache_f, "rb")) if os.path.exists(cache_f) else {}
for name, (f0, sf, sx, sy) in sorted(seeds.items(), key=lambda kv: kv[1][0]):
    if name in cache:
        continue
    t0 = time.time()
    track = M.track_stone_in_video_windowed(video, calib, pose, sf, (sx, sy), window_seconds=window_s,
                                            background_reference_undistorted=calib["frame_undistorted"], skip_precheck=True)
    if len(track) < 2:
        print(f"{name}: seuranta {len(track)} havaintoa - ohitetaan", flush=True); continue
    idxs = sorted(set(np.linspace(0, len(track) - 1, M.PROFILE_SAMPLES_PER_STONE).astype(int).tolist()))
    cand = [{"ellipse": track[i]["ellipse"], "contour": track[i]["contour"], "frame_idx": track[i]["frame_idx"], "pos_cm": track[i]["pos_cm"]} for i in idxs]
    obs, a_st = M.attach_alpha_observations(video, calib, cand)
    lvl = M.alpha_common_level(obs)
    obs, n_area = M.apply_area_rule(obs, lvl)
    cache[name] = dict(first_frame=f0, cand=cand, obs=obs, stats=dict(a_st, koko=n_area))
    pickle.dump(cache, open(cache_f, "wb"))
    print(f"{name}: seuranta {len(track)} hav -> {len(cand)} otosta -> {len(obs)} hyvaksytty  {cache[name]['stats']}  ({time.time() - t0:.0f} s)", flush=True)

allobs = [o for v in cache.values() for o in v["obs"]]
level = M.alpha_common_level(allobs)
print(f"\nYHTEINEN ALFA-TASO (kaikkien {len(allobs)} hyvaksytyn havainnon mediaani): {level:.3f}\n")
out_rows = []
for name in sorted(cache, key=lambda n: cache[n]["first_frame"]):
    v = cache[name]
    rec = dict(kivi=name, t_s=round(v["first_frame"] / 25.0, 1), otosta=len(v["cand"]), hyvaksytty=len(v["obs"]), **{f"hyl_{k}": x for k, x in v["stats"].items() if k != "in"})
    # UUSI: alfa-aariviiva, yhteinen taso
    fo = M.alpha_contour_observations(v["obs"], level)
    prof, ok_pool = M.try_fit_profile(pose, fo, label=f"{name} alfa (portti)")
    _, ok_solo = M.try_fit_profile(pose, fo, max_rms_px=M.SOLO_TRACK_MAX_RMS_PX, label=f"{name} alfa (yksin)")
    if prof is not None:
        rec.update(R=round(prof["R_max_cm"], 3), H=round(prof["H_total_cm"], 3), lovi=round(prof["handle_r_frac"], 3),
                   rms=round(prof["residual_rms_px"], 3), n_fit=len(fo), portti_ok=int(ok_pool), solo_ok=int(ok_solo))
    # VANHA: seurannan oma aariviiva samoilla otoksilla
    oldp, old_ok = M.try_fit_profile(pose, v["cand"], label=f"{name} seuranta (portti)")
    if oldp is not None:
        rec.update(R_vanha=round(oldp["R_max_cm"], 3), H_vanha=round(oldp["H_total_cm"], 3), rms_vanha=round(oldp["residual_rms_px"], 3), portti_vanha=int(old_ok))
    out_rows.append(rec)
keys = sorted({k for r in out_rows for k in r}, key=lambda k: (k != "kivi", k))
with open(os.path.join(outdir, "alfa_profiili_kivi_kerrallaan.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(out_rows)

def summ(rs, suf, tag):
    R = np.array([r["R" + suf] for r in rs if "R" + suf in r]); H = np.array([r["H" + suf] for r in rs if "H" + suf in r])
    rms = np.array([r["rms" + suf] for r in rs if "rms" + suf in r]); ok = sum(r.get("portti_ok" if suf == "" else "portti_vanha", 0) for r in rs)
    print(f"{tag}: {len(R)} kivea, portti OK {ok}/{len(rs)} | R ka {R.mean():.2f} hajonta {R.std():.2f} (min {R.min():.2f}, max {R.max():.2f}) | "
          f"H ka {H.mean():.2f} hajonta {H.std():.2f} | rms ka {rms.mean():.2f} (max {rms.max():.2f}) | R>14.55: {(R > 14.55).sum()}")
print("\nKIVI KERRALLAAN (portti: rms <= %.1f px, R %.1f..%.1f cm):" % (M.PROFILE_MAX_RMS_PX, M.PROFILE_R_MAX_MIN_CM, M.PROFILE_R_MAX_MAX_CM))
for r in out_rows:
    print(f"  {r['kivi']:14s} t={r['t_s']:6.1f}  hav {r['hyvaksytty']:2d}/{r['otosta']}  UUSI R={r.get('R', float('nan')):6.2f} H={r.get('H', float('nan')):5.2f} lovi={r.get('lovi', float('nan')):.2f} rms={r.get('rms', float('nan')):.2f} {'OK ' if r.get('portti_ok') else 'EI '}"
          f"| VANHA R={r.get('R_vanha', float('nan')):6.2f} H={r.get('H_vanha', float('nan')):5.2f} rms={r.get('rms_vanha', float('nan')):.2f} {'OK' if r.get('portti_vanha') else 'EI'}")
print()
summ(out_rows, "", "UUSI (alfa)")
summ(out_rows, "_vanha", "VANHA (seuranta)")
