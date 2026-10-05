"""Kivi kerrallaan oppiminen ja oppimisen luotettavuus.

Kaytto: python3 tools/oppimisen_luotettavuus.py <video> <calib_profile.pkl> <stones.csv> <ulostulokansio> [n_obs=40] [ikkuna_s=15]

Jokaisen kiven (live-CSV:n heitot + kaksi profiilinopettelun aikaista heittoa) havainnoista sovitetaan 3D-profiili ERIKSEEN. Sen jalkeen:
 1. yksittainen oppiminen: R, H, kahvan lovi, rms; hyvaksyisiko putken oma portti (rms <= 1.5, R 12.5-15) taman kiven yksin?
 2. bootstrap (havaintojen uudelleenotanta): kunkin parametrin epavarmuus yhdelle kivelle
 3. R-H-degeneraatio: R ja rms kun H kiinnitetaan arvoihin 11.5...15
 4. yhteinen koko: rms kun R ja H kiinnitetaan kaikkien kivien mediaaniin (sopiiko yksi yhteinen koko kaikille?)
 5. oppimiskayra: yhteissovitus k kivesta (k = 1,2,3,4,6,8,12,kaikki), satunnaisosajoukot -> kuinka paljon tulos vaihtelee k:n mukaan
"""
import sys, os, csv, types, time, pickle, collections, json
for n in ("tkinter", "tkinter.filedialog"):
    sys.modules[n] = types.ModuleType(n)
sys.modules["tkinter"].filedialog = sys.modules["tkinter.filedialog"]
import numpy as np
video, pkl, csv_in, outdir = sys.argv[1:5]
n_obs = int(sys.argv[5]) if len(sys.argv) > 5 else 40
window_s = float(sys.argv[6]) if len(sys.argv) > 6 else 15.0
code = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, code); os.chdir(code)
import main as M, stone_tracker as st, kamera9_01 as k9
os.makedirs(outdir, exist_ok=True)
pk = pickle.load(open(pkl, "rb")); calib, pose = pk["calib"], pk["pose"]
H_GRID = [11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0]

def fit(obs, r_fixed=-1.0, h_fixed=None):
    positions0 = []
    for o in obs:
        (cx, cy), _, _ = o["ellipse"]
        positions0.append(k9._stone_ground_position_z0(pose, cx, cy))
    hmin, hmax = (k9.STONE_HEIGHT_CM, k9.STONE_HEIGHT_MAX_CM) if h_fixed is None else (h_fixed - 0.005, h_fixed + 0.005)
    return st.fit_stone_profile_cpp(
        np.asarray(pose["K"], float), np.asarray(pose["R"], float), np.asarray(pose["t"], float),
        [np.ascontiguousarray(o["contour"], dtype=np.int32) for o in obs], np.asarray(positions0, float),
        40, float(k9.STONE_NOMINAL_RADIUS_CM), float(hmin), float(hmax),
        float(k9.HANDLE_NOTCH_R_FRAC_MIN), float(k9.HANDLE_NOTCH_R_FRAC_MAX), float(k9.HANDLE_NOTCH_R_FRAC),
        float(k9.STONE_SHAPE_REG_WEIGHT), 100, float(r_fixed))

def robust_fit(obs):
    f = fit(obs); keep = list(range(len(obs)))
    for _ in range(4):
        sizes = [min(40, len(np.asarray(obs[i]["contour"]).reshape(-1, 2))) for i in keep]
        res = np.asarray(f["residuals_px"]); pos = 0; per = []
        for n_i in sizes:
            per.append(float(np.sqrt(np.mean(res[pos:pos + n_i] ** 2)))); pos += n_i
        per = np.array(per); lim = max(2.0 * np.median(per), 2.5)
        good = [k for k, v in zip(keep, per) if v <= lim]
        if len(good) == len(keep) or len(good) < 8: break
        keep = good; f = fit([obs[i] for i in keep])
    return f, keep

# ---------- 0) havainnot ----------
rows = collections.defaultdict(list)
for r in csv.DictReader(open(csv_in)):
    try: rows[int(r["stone_id"])].append((int(r["frame"]), float(r["x_m"]) * 100, float(r["y_m"]) * 100))
    except ValueError: pass
seeds = {}
for sid, rr in rows.items():
    rr.sort(); inside = [x for x in rr if 450 <= x[2] <= 1950]
    if inside:
        s = min(inside, key=lambda x: abs(x[2] - 1400)); seeds[f"{sid}"] = (rr[0][0], s[0], s[1], s[2])
# profiilinopettelun aikaiset kaksi heittoa (eivat live-CSV:ssa): siemen skannauksesta (frame, x, y)
seeds["E1_heitto1810"] = (1810, 2080, 54.0, 1304.0)
seeds["E2_heitto2025"] = (2025, 2220, -47.0, 1732.0)
cache_f = os.path.join(outdir, "havainnot_cache.pkl")
cache = pickle.load(open(cache_f, "rb")) if os.path.exists(cache_f) else {}
for name, (f0, sf, sx, sy) in sorted(seeds.items(), key=lambda kv: kv[1][0]):
    if name in cache: continue
    t0 = time.time()
    track = M.track_stone_in_video_windowed(video, calib, pose, sf, (sx, sy), window_seconds=window_s,
                                            background_reference_undistorted=calib["frame_undistorted"], skip_precheck=True)
    if len(track) < 6:
        print(f"{name}: seuranta {len(track)} havaintoa - ohitetaan", flush=True); continue
    idx = np.unique(np.linspace(0, len(track) - 1, min(n_obs, len(track))).astype(int))
    cache[name] = dict(first_frame=f0, obs=[{"ellipse": track[i]["ellipse"], "contour": track[i]["contour"]} for i in idx])
    pickle.dump(cache, open(cache_f, "wb"))
    print(f"{name}: {len(track)} havaintoa seurannassa ({time.time() - t0:.0f} s)", flush=True)

# ---------- 1-4) kivi kerrallaan ----------
rng = np.random.default_rng(1)
res_rows = []; kept_obs = {}
names = sorted(cache, key=lambda k: cache[k]["first_frame"])
for name in names:
    obs = cache[name]["obs"]; t0 = time.time()
    f, keep = robust_fit(obs); kobs = [obs[i] for i in keep]; kept_obs[name] = kobs
    boots = []
    for _ in range(15):
        bi = rng.integers(0, len(kobs), len(kobs))
        try:
            b = fit([kobs[i] for i in bi]); boots.append((b["R_max_cm"], b["H_total_cm"], b["handle_r_frac"]))
        except Exception: pass
    boots = np.array(boots) if boots else np.full((1, 3), np.nan)
    hp = []
    for h0 in H_GRID:
        g = fit(kobs, h_fixed=h0); hp.append((h0, g["R_max_cm"], g["residual_rms_px"]))
    hp = np.array(hp)
    rec = dict(kivi=name, t_s=round(cache[name]["first_frame"] / 25.0, 1), n_hav=len(kobs), poistettu=len(obs) - len(kobs),
               R=round(f["R_max_cm"], 3), H=round(f["H_total_cm"], 3), lovi=round(f["handle_r_frac"], 3), rms=round(f["residual_rms_px"], 3),
               portti_ok=int(f["residual_rms_px"] <= M.PROFILE_MAX_RMS_PX and M.PROFILE_R_MAX_MIN_CM <= f["R_max_cm"] <= M.PROFILE_R_MAX_MAX_CM),
               R_boot_std=round(float(np.nanstd(boots[:, 0])), 3), H_boot_std=round(float(np.nanstd(boots[:, 1])), 3), lovi_boot_std=round(float(np.nanstd(boots[:, 2])), 3),
               R_kun_H12=round(float(hp[1, 1]), 2), R_kun_H13=round(float(hp[3, 1]), 2), R_kun_H14=round(float(hp[5, 1]), 2),
               rms_min_yli_H=round(float(hp[:, 2].min()), 3), rms_max_yli_H=round(float(hp[:, 2].max()), 3))
    res_rows.append(rec)
    print(f"{name:14s} t={rec['t_s']:6.1f}  R={rec['R']:.2f}±{rec['R_boot_std']:.2f} H={rec['H']:.2f}±{rec['H_boot_std']:.2f} lovi={rec['lovi']:.2f} rms={rec['rms']:.2f} "
          f"portti={'OK' if rec['portti_ok'] else 'EI'}  R(H=12/13/14)={rec['R_kun_H12']}/{rec['R_kun_H13']}/{rec['R_kun_H14']}  ({time.time() - t0:.0f} s)", flush=True)

good = [r for r in res_rows if r["portti_ok"]]
Rm = float(np.median([r["R"] for r in good])); Hm = float(np.median([r["H"] for r in good]))
print(f"\nYhteinen koko (portin lapaisseet {len(good)}/{len(res_rows)}): R={Rm:.2f}, H={Hm:.2f}")
for r in res_rows:
    g = fit(kept_obs[r["kivi"]], r_fixed=Rm, h_fixed=Hm)
    r["rms_yhteinen_koko"] = round(g["residual_rms_px"], 3); r["rms_lisays"] = round(g["residual_rms_px"] - r["rms"], 3)
with open(os.path.join(outdir, "oppiminen_kivi_kerrallaan.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(res_rows[0].keys())); w.writeheader(); w.writerows(res_rows)

# ---------- 5) oppimiskayra ----------
pool = {n: kept_obs[n][::max(1, len(kept_obs[n]) // 8)][:8] for n in kept_obs if any(r["kivi"] == n and r["portti_ok"] for r in res_rows)}
pn = sorted(pool, key=lambda n: cache[n]["first_frame"]); curve = []
print("\nOPPIMISKAYRA (yhteissovitus k kivesta, 8 hav./kivi):")
for k in [1, 2, 3, 4, 6, 8, 12, len(pn)]:
    if k > len(pn): continue
    subsets = [pn[:k]] if k == len(pn) else [pn[:k]] + [list(rng.choice(pn, k, replace=False)) for _ in range(5)]
    vals = []
    for sub in subsets:
        try:
            g = fit(sum([pool[n] for n in sub], []), h_fixed=None) if k >= 2 else fit(pool[sub[0]])
            vals.append((g["R_max_cm"], g["H_total_cm"], g["handle_r_frac"], g["residual_rms_px"]))
        except Exception as e:
            print("virhe", k, e)
    v = np.array(vals)
    row = dict(k=k, R_ka=float(v[:, 0].mean()), R_std=float(v[:, 0].std()), R_min=float(v[:, 0].min()), R_max=float(v[:, 0].max()),
               H_ka=float(v[:, 1].mean()), H_std=float(v[:, 1].std()), rms_ka=float(v[:, 3].mean()), n_osajoukkoa=len(vals))
    curve.append(row)
    print(f"  k={k:2d}: R {row['R_ka']:.2f} ± {row['R_std']:.2f} (min {row['R_min']:.2f}, max {row['R_max']:.2f})  H {row['H_ka']:.2f} ± {row['H_std']:.2f}  rms {row['rms_ka']:.2f}  ({row['n_osajoukkoa']} osajoukkoa)", flush=True)
json.dump(curve, open(os.path.join(outdir, "oppimiskayra.json"), "w"), indent=1)

R = np.array([r["R"] for r in good]); H = np.array([r["H"] for r in good])
print(f"\nYHTEENVETO: {len(good)}/{len(res_rows)} kivea läpäisee portin yksin. R: ka {R.mean():.2f}, hajonta kivien välillä {R.std():.2f}; "
      f"kivikohtainen bootstrap-epavarmuus keskimaarin {np.mean([r['R_boot_std'] for r in good]):.2f}. "
      f"H: ka {H.mean():.2f}, hajonta {H.std():.2f}, bootstrap {np.mean([r['H_boot_std'] for r in good]):.2f}. "
      f"rms-lisays yhteisella koolla keskimaarin {np.mean([r['rms_lisays'] for r in res_rows]):.3f} px (max {max(r['rms_lisays'] for r in res_rows):.3f}).")
