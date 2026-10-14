"""Kivien radat (radat_<puoli>.csv) -> kuva: Y(t), X(t) ja nopeus. python piirra_radat.py tulokset vasen"""
import csv
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

kansio, puoli = sys.argv[1], sys.argv[2]
tr = {}
for r in csv.DictReader(open(f"{kansio}/radat_{puoli}.csv")):
    tr.setdefault(int(r["rata"]), []).append((float(r["t_s"]), float(r["X_cm"]), float(r["Y_cm"]), float(r["err"])))
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
for k, v in tr.items():
    t, X, Y, e = np.array(v).T
    t = t - t[np.argmax(Y > 0)] if np.any(Y > 0) else t       # t = 0: kivi ylittaa T-viivan
    sp = []
    for i in range(len(t)):
        j0, j1 = max(0, i - 4), min(len(t), i + 5)
        sp.append(np.hypot(np.polyfit(t[j0:j1], X[j0:j1], 1)[0], np.polyfit(t[j0:j1], Y[j0:j1], 1)[0]) / 100.0)
    ax[0].plot(t, Y, label=f"heitto {k + 1}")
    ax[1].plot(Y, X, label=f"heitto {k + 1}")
    ax[2].plot(Y, sp, label=f"heitto {k + 1}")
ax[0].set(xlabel="aika T-viivan ylityksesta (s)", ylabel="Y (cm), 0 = T-viiva", title="Etenema")
ax[1].set(xlabel="Y (cm)", ylabel="X (cm)", title="Reitti (X sivusuunta)")
ax[2].set(xlabel="Y (cm)", ylabel="nopeus (m/s)", title="Nopeus (paikallinen sovitus +-4 ruutua)")
for a in ax:
    a.grid(alpha=0.3)
    a.axvline(640, color="gray", ls=":") if a is not ax[0] else None
ax[0].legend()
plt.tight_layout()
plt.savefig(f"{kansio}/radat_{puoli}.png", dpi=110)
