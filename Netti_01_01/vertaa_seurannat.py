"""Variseuranta (kahva) vs siluettiseuranta: nopeus ja sivusuuntainen jaannos viivojen yli. python vertaa_seurannat.py tulokset vasen"""
import csv
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

kansio, puoli = sys.argv[1], sys.argv[2]
sil, var = {}, {}
for r in csv.DictReader(open(f"{kansio}/radat_{puoli}.csv")):
    sil.setdefault(int(r["rata"]), []).append([float(r[c]) for c in ("t_s", "X_tasoitettu_cm", "Y_tasoitettu_cm", "v_cm_s")] + [float(r["X_cm"]), float(r["Y_cm"])])
for r in csv.DictReader(open(f"{kansio}/kahvaradat_{puoli}.csv")):
    var.setdefault(int(r["rata"]), []).append([float(r[c]) for c in ("t_s", "X_tasoitettu_cm", "Y_tasoitettu_cm", "v_cm_s", "X_cm", "Y_cm")] + [r["vari"]])
fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
col = ["tab:blue", "tab:orange", "tab:green"]
for k in sorted(var):
    a = np.array([v[:6] for v in var[k]], float)
    b = np.array(sil[k], float)
    ax[0].plot(a[:, 2], a[:, 3] / 100, color=col[k], label=f"heitto {k + 1} ({var[k][0][6]} kahva)")
    ax[0].plot(b[:, 2], b[:, 3] / 100, color=col[k], ls=":", alpha=0.8)
    # sivusuuntainen jaannos: raaka X - tasoitettu X
    ax[1].plot(a[:, 5], a[:, 4] - a[:, 1], color=col[k], lw=1)
    # Y-ero menetelmien valilla (kahva vs siluetti) samalla Y-akselilla
    yy = np.interp(a[:, 5], b[:, 4], b[:, 2])
    ax[2].plot(a[:, 5], a[:, 2] - np.interp(a[:, 0], b[:, 0], b[:, 2]), color=col[k], lw=1)
ax[0].set(title="Nopeus: kahvan väri (yhtenäinen) vs siluetti (pisteviiva)", xlabel="Y (cm)", ylabel="m/s")
ax[1].set(title="Sivusuunnan raaka jäännös, väriseuranta", xlabel="Y (cm)", ylabel="X - tasoitettu X (cm)", ylim=(-1.5, 1.5))
ax[2].set(title="Paikkaero kahva - siluetti (Y)", xlabel="Y (cm)", ylabel="cm")
for x in (-182.9, -121.9, -61, 0, 61, 121.9, 182.9, 640):
    ax[1].axvline(x, color="gray", ls=":", lw=0.8)
for a_ in ax:
    a_.grid(alpha=0.3)
ax[0].legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{kansio}/seurantojen_vertailu_{puoli}.png", dpi=110)
for k in sorted(var):
    a = np.array([v[:6] for v in var[k]], float)
    b = np.array(sil[k], float)
    t = a[:, 0]
    d = a[:, 1:3] - np.column_stack([np.interp(t, b[:, 0], b[:, 1]), np.interp(t, b[:, 0], b[:, 2])])
    sel = (a[:, 2] > -100) & (a[:, 2] < 560)
    print(f"heitto {k + 1}: kahva-siluetti ero X ka {d[sel, 0].mean():+.2f} cm (hajonta {d[sel, 0].std():.2f}), Y ka {d[sel, 1].mean():+.2f} cm (hajonta {d[sel, 1].std():.2f}); "
          f"X raaka jäännös RMS {np.sqrt(np.mean((a[sel, 4] - a[sel, 1]) ** 2)):.2f} cm")
