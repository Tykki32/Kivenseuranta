"""Jokaisen loydetyn kiven (heiton) OMA koko: sovittaa 3D-profiilin (R, H, kahvan lovi) yksittain kunkin kiven havaintoihin.

Kaytto: python3 tools/stone_sizes.py <video> <calib_profile.pkl> <stones.csv> <ulos.csv> [havaintoja=25] [ikkuna_s=15]
- calib_profile.pkl: {"calib":..., "pose":...} (esim. /tmp/w/mah_base/calib_profile.pkl)
- stones.csv: heittoportin lapaissyt CSV (sarakkeet frame, stone_id, x_m, y_m, rms_px, tarkka)
Jokaiselle kivelle: siemen = rivi jonka Y on lahinna 14 m (pelialueen keskella), aikaikkunaseuranta (esitarkistus OHITETAAN,
jotta erikokoiset kivet eivat karsiudu R-rajalla), tasavalisesti valitut havainnot -> C++-sovitus (vapaa R, H, lovi).
"""
import sys, os, csv, types, time, pickle, collections
for n in ("tkinter", "tkinter.filedialog"):
    sys.modules[n] = types.ModuleType(n)
sys.modules["tkinter"].filedialog = sys.modules["tkinter.filedialog"]
import numpy as np
video, pkl, csv_in, csv_out = sys.argv[1:5]
n_obs = int(sys.argv[5]) if len(sys.argv) > 5 else 40
window_s = float(sys.argv[6]) if len(sys.argv) > 6 else 15.0
code = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, code); os.chdir(code)
import main as M
pk = pickle.load(open(pkl, "rb"))
calib, pose = pk["calib"], pk["pose"]

rows = collections.defaultdict(list)
for r in csv.DictReader(open(csv_in)):
    try:
        rows[int(r["stone_id"])].append((int(r["frame"]), float(r["x_m"]) * 100, float(r["y_m"]) * 100,
                                         float(r["rms_px"]) if r["rms_px"] else float("nan"), int(r["tarkka"])))
    except ValueError:
        pass
out = []
for sid, rr in sorted(rows.items(), key=lambda kv: min(x[0] for x in kv[1])):
    rr.sort()
    inside = [x for x in rr if 450 <= x[2] <= 1950]
    if not inside:
        print(f"kivi {sid}: ei havaintoja pelialueella"); continue
    seed = min(inside, key=lambda x: abs(x[2] - 1400))
    t0 = time.time()
    track = M.track_stone_in_video_windowed(
        video, calib, pose, seed[0], (seed[1], seed[2]), window_seconds=window_s,
        background_reference_undistorted=calib["frame_undistorted"], skip_precheck=True)
    if len(track) < 6:
        print(f"kivi {sid}: seuranta antoi vain {len(track)} havaintoa"); continue
    idx = np.unique(np.linspace(0, len(track) - 1, min(n_obs, len(track))).astype(int))
    obs = [{"ellipse": track[i]["ellipse"], "contour": track[i]["contour"]} for i in idx]
    fit_raw = M._fit_stone_profile(pose, obs)
    # ROBUSTI: poistetaan havainnot joiden oma rms on selvasti muita suurempi (esim. pelaaja/harja/kohina liittyi kontuuriin
    # tai seuranta poimi vaaran kohteen) ja sovitetaan uudelleen - yksi huono havainto muuten venyttaa R:aa kohtuuttomasti
    fit = fit_raw
    keep = list(range(len(obs)))
    for _ in range(4):
        sizes = [min(40, len(np.asarray(obs[i]["contour"]).reshape(-1, 2))) for i in keep]
        res = np.asarray(fit["residuals_px"]); pos = 0; per_obs = []
        for n_i in sizes:
            per_obs.append(float(np.sqrt(np.mean(res[pos:pos + n_i] ** 2)))); pos += n_i
        per_obs = np.array(per_obs)
        limit = max(2.0 * np.median(per_obs), 2.5)
        good = [k for k, v in zip(keep, per_obs) if v <= limit]
        if len(good) == len(keep) or len(good) < 8:
            break
        keep = good
        fit = M._fit_stone_profile(pose, [obs[i] for i in keep])
    n_removed = len(obs) - len(keep)
    live_rms = np.nanmedian([x[3] for x in rr]); live_tk = np.mean([x[4] for x in rr])
    rec = dict(stone_id=sid, first_frame=rr[0][0], t_s=round(rr[0][0] / 25.0, 1), seed_frame=seed[0], n_track=len(track), n_obs=len(obs),
               R_raw_cm=round(fit_raw["R_max_cm"], 3), poistettu_hav=n_removed, R_cm=round(fit["R_max_cm"], 3), H_cm=round(fit["H_total_cm"], 3), handle_r_frac=round(fit["handle_r_frac"], 4),
               fit_rms_px=round(fit["residual_rms_px"], 3), live_rms_median=round(float(live_rms), 3), live_tarkka_osuus=round(float(live_tk), 3),
               diameter_cm=round(2 * fit["R_max_cm"], 2))
    out.append(rec)
    print(f"kivi {sid:4d} t={rec['t_s']:7.1f}s  R={rec['R_cm']:.2f} H={rec['H_cm']:.2f} lovi={rec['handle_r_frac']:.3f} sovitus-rms={rec['fit_rms_px']:.2f} "
          f"live-rms={rec['live_rms_median']:.2f}  (raaka R={rec['R_raw_cm']:.1f}, poistettu {n_removed}/{len(obs)} hav., {time.time() - t0:.0f} s)", flush=True)
if out:
    with open(csv_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
    R = np.array([o["R_cm"] for o in out]); H = np.array([o["H_cm"] for o in out]); rms = np.array([o["fit_rms_px"] for o in out])
    print(f"\nYHTEENVETO {len(out)} kivea: R keskiarvo {R.mean():.2f} cm (hajonta {R.std():.2f}, min {R.min():.2f}, max {R.max():.2f}), "
          f"H keskiarvo {H.mean():.2f} (min {H.min():.2f}, max {H.max():.2f}), sovitus-rms keskiarvo {rms.mean():.2f} (max {rms.max():.2f})")
    print("Tallennettu:", csv_out)
