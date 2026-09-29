"""Analysoi main.py:n TRACKER_MODE=compare -ajon lokin (TRACKER_COMPARE_LOG).

Kaytto:  python analyze_compare.py compare.csv calib_profile.pkl [--json out.json]

Vertailee grid-hakua (alkuperainen), mean-shift-hakua (ms, ms<gain>) ja
tyhjentavaa ristikkohakua (gold) samoilla syotteilla. Tarkkuus mitataan
PIKSELEINA (kaukana 10 cm ~ 1 px), koska cm-virhe riippuu etaisyydesta.
"""
import sys, os, csv, json, pickle, types
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))


def load_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def col(rows, k):
    return np.array([float(r[k]) if r[k] != "" else np.nan for r in rows])


def main():
    csv_path, pkl_path = sys.argv[1], sys.argv[2]
    out_json = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
    rows = load_rows(csv_path)
    pk = pickle.load(open(pkl_path, "rb"))
    K, R, t = pk["pose"]["K"], pk["pose"]["R"], pk["pose"]["t"]
    zc = float(pk["profile"]["H_total_cm"]) / 2.0

    def px(X, Y):
        pts = np.stack([X, Y, np.full_like(X, zc)], axis=1)
        cam = pts @ R.T + t
        img = cam @ K.T
        return img[:, :2] / img[:, 2:3]

    methods = [m for m in ["grid", "ms", "gold"] if f"{m}_found" in rows[0]]
    methods += sorted({k.split("_")[0] for k in rows[0] if k.startswith("ms") and k != "ms_wall_ms"
                       and k.split("_")[0] not in ("ms",) and k.endswith("_found")})
    n = len(rows)
    C = {m: {k: col(rows, f"{m}_{k}") for k in
             ["wall_ms", "found", "X", "Y", "rms_px", "tarkka", "score", "loc_X", "loc_Y",
              "prep_ms", "locate_ms", "refine_ms", "iters", "converged", "fallback"]} for m in methods}
    X0, Y0 = col(rows, "X0"), col(rows, "Y0")
    frames = col(rows, "frame")
    has_gold = "gold" in methods and np.any(~np.isnan(C["gold"]["loc_X"]))
    rep = {"rows": n, "methods": methods}

    print(f"rivit (kivi-frame-parit): {n}, framea: {len(set(frames))}")

    # ---- Aika ----
    print("\n== AIKA (ms / kivi-paivitys; keskiarvo) ==")
    print(f"{'menetelma':10s} {'prep':>6s} {'haku':>6s} {'LM':>6s} {'yht':>6s} | {'kutsu wall':>10s} med / p95")
    rep["time"] = {}
    for m in methods:
        d = C[m]
        ok = ~np.isnan(d["locate_ms"])
        tot = d["prep_ms"][ok] + d["locate_ms"][ok] + np.nan_to_num(d["refine_ms"][ok])
        print(f"{m:10s} {np.nanmean(d['prep_ms']):6.2f} {np.nanmean(d['locate_ms']):6.2f} "
              f"{np.nanmean(d['refine_ms']):6.2f} {np.mean(tot):6.2f} | {np.nanmean(d['wall_ms']):10.1f} "
              f"{np.nanmedian(d['wall_ms']):.1f} / {np.nanpercentile(d['wall_ms'], 95):.1f}")
        rep["time"][m] = dict(prep=float(np.nanmean(d["prep_ms"])), locate=float(np.nanmean(d["locate_ms"])),
                              refine=float(np.nanmean(d["refine_ms"])), total=float(np.mean(tot)),
                              wall_mean=float(np.nanmean(d["wall_ms"])))
    for m in methods:
        if m.startswith("ms"):
            print(f"  {m}: varahaku (fallback) osuus = {np.nanmean(C[m]['fallback']):.3f}, "
                  f"iteraatioita ka {np.nanmean(C[m]['iters'][C[m]['iters'] > 0]) if np.any(C[m]['iters'] > 0) else 0:.2f}, "
                  f"suppeni {np.nanmean(C[m]['converged'][C[m]['iters'] > 0]) if np.any(C[m]['iters'] > 0) else 0:.2f}")

    # ---- Tarkkuus ----
    if has_gold:
        g_ok = C["gold"]["found"] == 1
        gl = px(C["gold"]["loc_X"], C["gold"]["loc_Y"])
        gf = px(np.nan_to_num(C["gold"]["X"]), np.nan_to_num(C["gold"]["Y"]))
        disp_cm = np.abs(C["gold"]["loc_Y"] - Y0)
        far = Y0 > 2500
        mid = (Y0 > 1500) & (Y0 <= 2500)
        near = Y0 <= 1500
        strata = {"kaikki": np.ones(n, bool), "aito kivi (gold tarkka=1)": C["gold"]["tarkka"] == 1,
                  "liikkuva (>2cm)": disp_cm > 2.0, "paikallaan": disp_cm <= 2.0,
                  "lahella Y<=15m": near, "keski 15-25m": mid, "kaukana Y>25m": far}
        print("\n== TARKKUUS vs. tyhjentava haku (gold), PIKSELEINA ==")
        print("   loc = hakuvaiheen tulos, fin = LM:n jalkeinen lopullinen sijainti")
        rep["accuracy"] = {}
        for sname, sel in strata.items():
            s = sel & g_ok & ~np.isnan(gl[:, 0])
            if s.sum() < 5:
                continue
            print(f"\n[{sname}] n={int(s.sum())}")
            print(f"  {'menetelma':10s} {'loytyi':>6s} {'locScore':>8s} {'loc mediaani':>12s} {'loc p90':>8s} "
                  f"{'fin mediaani':>12s} {'fin p90':>8s} {'rms_px med':>10s} {'tarkka':>6s}")
            for m in methods:
                if m == "gold":
                    continue
                d = C[m]
                le = np.linalg.norm(px(np.nan_to_num(d["loc_X"]), np.nan_to_num(d["loc_Y"])) - gl, axis=1)[s]
                fnd = d["found"][s] == 1
                fe_all = np.linalg.norm(px(np.nan_to_num(d["X"]), np.nan_to_num(d["Y"])) - gf, axis=1)[s]
                fe = fe_all[fnd & (C["gold"]["found"][s] == 1)]
                rms = d["rms_px"][s][fnd]
                print(f"  {m:10s} {fnd.mean():6.2f} {np.nanmean(d['score'][s]):8.3f} {np.nanmedian(le):12.2f} "
                      f"{np.nanpercentile(le, 90):8.2f} {np.nanmedian(fe) if len(fe) else np.nan:12.2f} "
                      f"{np.nanpercentile(fe, 90) if len(fe) else np.nan:8.2f} {np.nanmedian(rms) if len(rms) else np.nan:10.2f} "
                      f"{np.nanmean(d['tarkka'][s]):6.2f}")
                rep["accuracy"].setdefault(sname, {})[m] = dict(
                    n=int(s.sum()), found=float(fnd.mean()), loc_med=float(np.nanmedian(le)), loc_p90=float(np.nanpercentile(le, 90)),
                    fin_med=float(np.nanmedian(fe)) if len(fe) else None, fin_p90=float(np.nanpercentile(fe, 90)) if len(fe) else None)
            gd = C["gold"]
            print(f"  {'gold':10s} {np.mean(gd['found'][s] == 1):6.2f} {np.nanmean(gd['score'][s]):8.3f}  (vertailukohta)")

        print("\n== ms vs grid: lopullinen sijainti toisiinsa nahden (px, LM jalkeen) ==")
        both = (C["ms"]["found"] == 1) & (C["grid"]["found"] == 1)
        dd = np.linalg.norm(px(np.nan_to_num(C["ms"]["X"]), np.nan_to_num(C["ms"]["Y"])) -
                            px(np.nan_to_num(C["grid"]["X"]), np.nan_to_num(C["grid"]["Y"])), axis=1)[both]
        print(f"  n={both.sum()} mediaani {np.median(dd):.2f} p90 {np.percentile(dd, 90):.2f} p99 {np.percentile(dd, 99):.2f} max {dd.max():.1f} px")
        print(f"  loytyi: grid {np.mean(C['grid']['found']):.3f}, ms {np.mean(C['ms']['found']):.3f}; "
              f"ms loysi kun grid ei: {int(((C['ms']['found']==1)&(C['grid']['found']!=1)).sum())}, "
              f"grid loysi kun ms ei: {int(((C['grid']['found']==1)&(C['ms']['found']!=1)).sum())}")
    if out_json:
        json.dump(rep, open(out_json, "w"), indent=1)


if __name__ == "__main__":
    main()
