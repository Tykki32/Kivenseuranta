"""Simulaattorikokeet: eri seurantavariantit samoilla heitoilla (Testi_02_03).

Kaytto: python sim_experiments.py fdump_dir calib_profile.pkl hk_grid_csv hk_ms_csv
Heitot maaritellaan alla (kehykset + oikean radan lahde).
"""
import sys, csv, collections, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim_track import Sim, evaluate

fdump, pkl, grid_csv, ms_csv = sys.argv[1:5]


def track_rows(path, sid):
    out = {}
    for r in csv.DictReader(open(path)):
        if int(r["stone_id"]) == sid:
            out[int(r["frame"])] = (float(r["x_m"]) * 100, float(r["y_m"]) * 100)
    return out


def merged(*parts):
    d = {}
    for p in parts:
        for k, v in p.items():
            d.setdefault(k, v)
    return d


def find_track(path, f0, f1, y0):
    """id jonka rivit osuvat ruutuvaliin ja alkupiste lahella y0."""
    best = None
    tr = collections.defaultdict(dict)
    for r in csv.DictReader(open(path)):
        tr[int(r["stone_id"])][int(r["frame"])] = (float(r["x_m"]) * 100, float(r["y_m"]) * 100)
    for sid, d in tr.items():
        fr = [f for f in d if f0 <= f <= f1]
        if len(fr) > 100 and abs(d[min(fr)][1] - y0) < 200:
            if best is None or len(fr) > best[1]:
                best = (sid, len(fr))
    return best[0] if best else None


THROWS = {
    # nimi: (alkuframe, alkupiste, oikea rata)
}
g179 = track_rows("/tmp/w/fix1/stones.csv", 179)
t119 = track_rows(grid_csv, 119)
THROWS["T1 f3480 (ms menetti)"] = (3480, (8.0, 3143.6), merged({f: v for f, v in g179.items() if f < 3500}, t119))
THROWS["T2 f5940 (ms menetti)"] = (5940, (-14.0, 3027.6), track_rows(grid_csv, 161))
sid = find_track(grid_csv, 6490, 6990, 2908)
tr = track_rows(grid_csv, sid); f0 = min(tr)
THROWS["T3 f%d (molemmat)" % f0] = (f0, tr[f0], tr)
sid = find_track(ms_csv, 8260, 8830, 3098)
tr = track_rows(ms_csv, sid); f0 = min(tr)
THROWS["T4 f%d (vain ms)" % f0] = (f0, tr[f0], tr)

MS = dict(locate_mode=1, ms_gain=0.8, ms_max_iter=6, ms_tol_px=0.3, ms_inner_weight=2.0, ms_margin_scale=1.10,
          ms_tau=0.5, ms_polish_step_cm=4.0)
VARIANTS = {
    "grid": (dict(locate_mode=0), False),
    "ms hybridi": (MS, False),
    "ms pure": (dict(MS, locate_mode=2), False),
    "ms+ennuste": (MS, True),
    "ms+reuna-varahaku": (dict(MS, ms_edge_fallback=True), False),
    "ms+ennuste+reuna": (dict(MS, ms_edge_fallback=True), True),
}
only = sys.argv[5].split(",") if len(sys.argv) > 5 and sys.argv[5] != "all" else list(VARIANTS)
throw_filter = sys.argv[6].split(",") if len(sys.argv) > 6 else None

sim = Sim(fdump, pkl)
print(f"{'heitto':26s} {'variantti':20s} {'peitto<30cm':>11s} {'virhe med':>9s} {'ensim. huono':>12s} {'paattyi':>16s}")
for tname, (f0, xy, ref) in THROWS.items():
    if throw_filter and not any(tname.startswith(t) for t in throw_filter):
        continue
    for vname in only:
        kw, pred = VARIANTS[vname]
        r = sim.run(f0, xy, ref, kw, pred=pred)
        ev = evaluate(r["track"], ref)
        endtxt = f"{r['event'][0]}@{r['event'][1]}" if r["event"] else f"ref-loppu@{r['end_frame']}"
        print(f"{tname:26s} {vname:20s} {ev['coverage']:11.2f} {ev['err_med'] if ev['err_med'] is not None else float('nan'):9.1f} {str(ev['first_bad']):>12s} {endtxt:>16s}")
