"""Ajaa heittoportin uudelleen RAAKALLE CSV:lle (ilman videon uudelleenajoa).

Kaytto: python regate_csv.py raaka.csv ulos.csv
Portin parametrit ymparistomuuttujilla (GATE_*, katso main.py). Kayttaa main.py:n _select_throws-funktiota.
"""
import sys, os, csv, types, collections
for n in ("tkinter", "tkinter.filedialog"):
    if n not in sys.modules:
        sys.modules[n] = types.ModuleType(n)
sys.modules["tkinter"].filedialog = sys.modules["tkinter.filedialog"]
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import main as M

src, dst = sys.argv[1:3]
tr = collections.defaultdict(list)
for r in csv.DictReader(open(src)):
    tr[int(r["stone_id"])].append((int(r["frame"]), float(r["timestamp_s"]), dict(
        X_cm=float(r["x_m"]) * 100, Y_cm=float(r["y_m"]) * 100, tarkka=int(r["tarkka"]),
        n_body=int(r["n_runkopistetta"]), n_ring=int(r["n_reunapistetta"]),
        rms_px=float(r["rms_px"]) if r["rms_px"] else None,
        ring_radius_cm=float(r["rengas_r_cm"]) if r["rengas_r_cm"] else None)))
sel = M._select_throws(list(tr.items()))
with open(dst, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(M.CSV_HEADER)
    for sid, rows in sel:
        for rf, rt, rr in rows:
            M._write_stone_csv_row(w, rf, rt, sid, rr)
print(f"{len(tr)} rataa -> {len(sel)} heittoa -> {dst}")
