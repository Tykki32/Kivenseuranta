"""Piirtaa heittojen kattavuuskuvan (rivi = todellinen heitto, sarake = ajo).

Kaytto: python plot_coverage.py ulos.png nimi1=ajo1.csv nimi2=ajo2.csv ...
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_throws import load_tracks, is_real, overlap

out = sys.argv[1]
runs = dict(a.split("=", 1) for a in sys.argv[2:])
T = {n: load_tracks(p) for n, p in runs.items()}
throws = []
for n, tr in T.items():
    for k, v in tr.items():
        if is_real(v) and abs(v[-1][2] - v[0][2]) > 500 and v[0][0] >= 2900:
            if not any(overlap(v, w) >= 0.5 and overlap(w, v) >= 0.5 for _, _, w in throws):
                throws.append((n, k, v))
throws.sort(key=lambda x: x[2][0][0])
M = np.array([[max((overlap(v, w) for w in tr.values()), default=0) for tr in T.values()] for _, _, v in throws])
fig, ax = plt.subplots(figsize=(1.6 + 1.5 * len(T), 0.45 * len(throws) + 1.6))
ax.imshow(np.where(M >= 0.5, 1.0, 0.0), cmap=matplotlib.colors.ListedColormap(["#d9534f", "#5cb85c"]), aspect="auto", vmin=0, vmax=1)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", color="white", fontsize=9)
ax.set_xticks(range(len(T)))
ax.set_xticklabels(list(T), rotation=20, ha="right")
ax.set_yticks(range(len(throws)))
ax.set_yticklabels([f"ruudut {v[0][0]}–{v[-1][0]}" for _, _, v in throws], fontsize=9)
cnt = (M >= 0.5).sum(axis=0)
ax.set_title("Todellisten heittojen kattavuus (vihreä = seurattu ≥ 50 % ruuduista)\n" +
             "\n".join(f"{n}: {c}/{len(throws)}" for n, c in zip(T, cnt)), fontsize=9, loc="left")
plt.tight_layout()
plt.savefig(out, dpi=140)
print(out, "heittoja", len(throws), dict(zip(T, cnt.tolist())))
