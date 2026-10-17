"""Kiven vakioiden (A, kappa) uudelleensovitus tallennetuista radoista ilman koko videon analyysia.

    python netti_live.py --lahde netti.mp4 --nopea --ei-ikkunaa --csv heitot.csv --radat radat.pkl       # kerran: koko analyysi, radat ja mittaukset talteen
    python tools/sovita_vakiot.py radat.pkl                    # vakiot kullekin heitolle kaukopaan radasta (netti_vakiot.sovita) + kivikohtaiset mediaanit ja varianssianalyysi
    python tools/sovita_vakiot.py radat.pkl --csv vakiot.csv   # lisaksi heittokohtainen CSV

Sovituksen valinnat (rajat, kierteen pakote) ovat netti_vakiot.py:ssa ja netti_seuranta.vakiot_argumentit:ssa; kun muutat niita, tama ajaa ne uudelleen kaikille heitoille noin minuutissa."""
import argparse
import csv
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import netti_seuranta as ns  # noqa: E402
import netti_vakiot as nv  # noqa: E402
from yhteenveto import _anova  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("radat", help="netti_live.py --radat -tiedosto (pickle)")
    ap.add_argument("--csv", help="heittokohtaiset vakiot CSV:hen")
    ap.add_argument("--fps", type=float, default=30.0)
    a = ap.parse_args()
    heitot = pickle.load(open(a.radat, "rb"))
    rivit = []
    for i, h in enumerate(heitot, 1):
        k = h.get("kauko")
        if k is None or "virhe" in k or "pisteet" not in k:
            continue
        args, lahde = ns.vakiot_argumentit(k, h.get("kauko_kierto"), h.get("kierre"), h.get("hog2_f"), a.fps)
        r = nv.sovita(**args)
        rivit.append(dict(heitto=i, vari=h["vari"], kivi_nro=h["kivi_nro"], kauko_tila=k["tila"], omega_lahde=lahde, **{x: r.get(x) for x in ("ok", "syy", "A", "A_sd", "kappa", "kappa_sd", "kappa_ok", "v0", "w0", "rms_y_cm", "rms_x_cm", "n", "kesto_s")}))
    ok = [r for r in rivit if r["ok"]]
    kap = [r for r in ok if r["kappa_ok"]]
    print(f"{len(heitot)} heittoa, kaukopaan rata {len(rivit)}, A sovitettu {len(ok)}, kappa luotettava {len(kap)}")
    if ok:
        A = np.array([r["A"] for r in ok])
        print(f"A: mediaani {np.median(A):.4f}, hajonta (1,48 MAD) {1.4826 * np.median(np.abs(A - np.median(A))):.4f}, valille {A.min():.4f} ... {A.max():.4f}")
    if kap:
        K = np.array([r["kappa"] for r in kap])
        print(f"kappa: mediaani {np.median(K):.2f}, hajonta (1,48 MAD) {1.4826 * np.median(np.abs(K - np.median(K))):.2f}, valille {K.min():.2f} ... {K.max():.2f}")
    syyt = {}
    for r in rivit:
        if not r["ok"]:
            avain = (r["syy"] or "?").split("(")[0].strip()
            syyt[avain] = syyt.get(avain, 0) + 1
    if syyt:
        print("hylatyt:", ", ".join(f"{v} x {k}" for k, v in sorted(syyt.items(), key=lambda x: -x[1])))
    ryh, ryhk = {}, {}
    for r in ok:
        ryh.setdefault((r["vari"], r["kivi_nro"]), []).append(r["A"])
    for r in kap:
        ryhk.setdefault((r["vari"], r["kivi_nro"]), []).append(r["kappa"])
    print("kivikohtaisesti (vari numero: heittoja, A mediaani +- sd; kappa mediaani +- sd):")
    for avain in sorted(ryh, key=lambda x: (x[0], x[1] or 0)):
        A_ = np.array(ryh[avain])
        K_ = np.array(ryhk.get(avain, []))
        ka = f"{np.median(K_):.2f} +- {K_.std():.2f} (n = {len(K_)})" if len(K_) > 1 else (f"{K_[0]:.2f}" if len(K_) == 1 else "-")
        print(f"  {avain[0]:10s} {avain[1]}: {len(A_)} heittoa, A {np.median(A_):.4f} +- {A_.std():.4f}; kappa {ka}")
    for nimi, ryhmat in (("A", ryh), ("kappa", ryhk)):
        v = _anova(list(ryhmat.values()))
        if v is not None:
            F, p, sd_sis, sd_val = v
            print(f"{nimi}: varianssianalyysi (kivi = vari + numero): F = {F:.2f}, p = {p:.3f}; kiven sisainen hajonta {sd_sis:.4g}, kivien valinen {sd_val:.4g}")
    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rivit[0].keys()))
            w.writeheader()
            w.writerows(rivit)
        print("kirjoitettu", a.csv)


if __name__ == "__main__":
    main()
