"""Kivien radat (radat_<puoli>.csv) -> kuva: Y(t), X(Y) ja nopeus (todellinen aika t_kor_s, ks. aikaperusta.py). python piirra_radat.py tulokset vasen"""
import csv
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

kansio, puoli = sys.argv[1], sys.argv[2]
tr = {}
for r in csv.DictReader(open(f"{kansio}/radat_{puoli}.csv")):
    tr.setdefault(int(r["rata"]), []).append((float(r.get("t_kor_s") or r["t_s"]), float(r["X_tasoitettu_cm"]), float(r["Y_tasoitettu_cm"]), float(r["v_cm_s"]) / 100.0))
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
for k, v in tr.items():
    t, X, Y, sp = np.array(v).T
    t = t - t[np.argmax(Y > 0)] if np.any(Y > 0) else t       # t = 0: kivi ylittaa T-viivan
    ax[0].plot(t, Y, label=f"heitto {k + 1}")
    ax[1].plot(Y, X, label=f"heitto {k + 1}")
    ax[2].plot(Y, sp, label=f"heitto {k + 1}")
ax[0].set(xlabel="aika T-viivan ylityksesta (s)", ylabel="Y (cm), 0 = T-viiva", title="Etenema")
ax[1].set(xlabel="Y (cm)", ylabel="X (cm)", title="Reitti (X sivusuunta)")
ax[2].set(xlabel="Y (cm)", ylabel="nopeus (m/s)", title="Nopeus (tasoitettu)")
for a in ax:
    a.grid(alpha=0.3)
    a.axvline(640, color="gray", ls=":") if a is not ax[0] else None
ax[0].legend()
plt.tight_layout()
plt.savefig(f"{kansio}/radat_{puoli}.png", dpi=110)
