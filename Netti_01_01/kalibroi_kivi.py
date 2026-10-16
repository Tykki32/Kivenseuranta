"""Kiven runkomallin opetus kaikista ruuduista: python kalibroi_kivi.py VIDEO tulokset vasen [--ikkunat 8-22,56-66,104-114] [--ymax 700] [--gamma 1.3]

Kimppusovitus (bundle adjustment): profiilin solmut r(z) (cm), korkeus H, optinen sumeus sigma_opt ja reunan pehmeys b yhteisina parametreina,
jokaiselle ruudulle oma akseli (X, Y). Havainnot: tulokset/radat_<puoli>.csv:n ruudut (kivi_tarkka.havainto: graniitin alfa + kelvolliset pikselit),
residuaalit (malli - alfa) * paino reunakaistalla. Profiilille silotus- ja yksihuippurangaistus (yla- ja alapuolella vain laskeva), jotta
piilossa oleva ylarengas ei ota mielivaltaista arvoa. Kirjoittaa tulokset/kivimalli_<puoli>.json (kivi_tarkka.Runko.lataa)."""
import argparse
import csv
import os
import time
from multiprocessing import Pool

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

import aja_kivi as ak
import kivi_netti as kv
import kivi_tarkka as kt

_G = {}


def _alusta(video, kansio, puoli):
    _G["cam"] = kv.Kamera(puoli, os.path.join(kansio, f"kalibrointi_{puoli}.json"), os.path.join(kansio, f"moodikuva_{puoli}.png"))
    _G["cap"] = cv2.VideoCapture(video)
    _G["runko"] = kt.Runko.lataa(kansio, puoli)


def _havainto_tyo(a):
    f, X0, Y0 = a
    cam, cap, runko = _G["cam"], _G["cap"], _G["runko"]
    fr = cam.ruutu(cap, f)
    alfa, ok = kt.havainto(cam, fr)
    s = kt.sovita(cam, runko, alfa, ok, X0, Y0)
    hull = runko.hull(cam, s["X"], s["Y"])
    x0, y0, w, h = kt.ikkuna(cam, hull, marg=40)
    return dict(frame=f, X0=s["X"], Y0=s["Y"], rms=s["rms"], kelvolliset=s["kelvolliset"], ox=x0, oy=y0,
                alfa=alfa[y0:y0 + h, x0:x0 + w].copy(), ok=ok[y0:y0 + h, x0:x0 + w].copy())


def kerää_havainnot(video, kansio, puoli, ikkunat, saikeet=4):
    """Kaikki ratojen ruudut: [dict(frame, rata, t, X0, Y0, rms, kelvolliset, ox, oy, alfa, ok)] (alfa ja ok leikkeina kiven ymparilta)."""
    ak.yksilolliset_ruudut(video, ikkunat, puoli)
    t2f = {round(t, 3): f for f, t in ak.T_KARTTA.items()}
    rows = []
    for r in csv.DictReader(open(os.path.join(kansio, f"radat_{puoli}.csv"))):
        f = t2f.get(round(float(r["t_s"]), 3))
        if f is not None:
            rows.append((int(r["rata"]), float(r["t_s"]), f, float(r["X_tasoitettu_cm"]), float(r["Y_tasoitettu_cm"])))
    with Pool(saikeet, initializer=_alusta, initargs=(video, kansio, puoli)) as pool:
        res = pool.map(_havainto_tyo, [(f, X, Y) for (_, _, f, X, Y) in rows], chunksize=4)
    for d, (rata, t, f, X, Y) in zip(res, rows):
        d.update(rata=rata, t=t)
    return res


def valitse(H, ymax=700.0, rms_max=0.12, kelv_min=0.25, per_lokero=10, lokero_cm=50, siemen=1):
    """Opetusruudut: hyvin sopivat, kelvollisia pikseleita riittavasti, enintaan per_lokero ruutua / (heitto, 50 cm Y-lokero)."""
    rng = np.random.default_rng(siemen)
    idx = []
    for rata in sorted(set(d["rata"] for d in H)):
        for lo in range(-400, int(ymax) + lokero_cm, lokero_cm):
            q = [i for i, d in enumerate(H) if d["rata"] == rata and lo <= d["Y0"] < lo + lokero_cm and d["rms"] < rms_max
                 and d["kelvolliset"] > kelv_min and d["Y0"] < ymax]
            if len(q) > per_lokero:
                q = list(rng.choice(q, per_lokero, replace=False))
            idx += q
    return sorted(idx)


class Kimppu:
    """Parametrit: [r_1..r_n (cm), H, sigma_opt, b_cm] + ruutukohtaiset (X, Y)."""

    def __init__(self, cam, H, idx, F, th0, gamma, lam_sm=2.0, lam_mono=5.0, kp=6, H_kiinteä=None):
        self.cam, self.F, self.gamma, self.lam_sm, self.lam_mono, self.kp = cam, np.asarray(F, float), gamma, lam_sm, lam_mono, kp
        self.H_kiinteä = H_kiinteä
        self.nk = len(F)
        runko0 = kt.Runko(F, th0[:self.nk], th0[self.nk], th0[self.nk + 1], th0[self.nk + 2], gamma)
        self.k = [kt.Kehys(cam, runko0, H[i]["alfa"], H[i]["ok"], H[i]["X0"], H[i]["Y0"], ox=H[i]["ox"], oy=H[i]["oy"], kaista=6, marg=12) for i in idx]
        self.idx = idx
        self.xy0 = np.array([[H[i]["X0"], H[i]["Y0"]] for i in idx]).ravel()
        self.n = len(idx)
        self.pituudet = [len(f.idx) for f in self.k]
        self.alut = np.r_[0, np.cumsum(self.pituudet)]
        self.nth = self.nk + 3

    def runko(self, th):
        return kt.Runko(self.F, th[:self.nk], th[self.nk], th[self.nk + 1], th[self.nk + 2], self.gamma)

    def fun(self, p):
        th, xy = p[:self.nth], p[self.nth:].reshape(-1, 2)
        runko = self.runko(th)
        res = [f.residuaalit(self.cam, runko, xy[j, 0], xy[j, 1]) for j, f in enumerate(self.k)]
        rk = th[:self.nk]
        d = np.diff(rk)
        mono = np.concatenate([np.maximum(0.0, -d[:self.kp]), np.maximum(0.0, d[self.kp:])]) * self.lam_mono
        return np.concatenate(res + [self.lam_sm * np.diff(rk, 2), mono])

    def harva(self):
        nres = self.alut[-1] + (self.nk - 2) + (self.nk - 1)
        S = lil_matrix((nres, self.nth + 2 * self.n), dtype=int)
        S[:, :self.nth] = 1
        for j in range(self.n):
            S[self.alut[j]:self.alut[j + 1], self.nth + 2 * j:self.nth + 2 * j + 2] = 1
        return S

    def ratkaise(self, th0, max_nfev=100, verbose=1):
        p0 = np.concatenate([th0, self.xy0])
        lb = np.concatenate([np.full(self.nk, 3.0), [8.0, 0.3, 0.0], np.full(2 * self.n, -1e3)])
        ub = np.concatenate([np.full(self.nk, 16.0), [24.0, 2.5, 1.5], np.full(2 * self.n, 1e3)])
        if self.H_kiinteä is not None:
            lb[self.nk], ub[self.nk] = self.H_kiinteä - 1e-6, self.H_kiinteä + 1e-6
            p0[self.nk] = self.H_kiinteä
        sc = np.concatenate([np.ones(self.nk), [1.0, 0.3, 0.3], np.full(2 * self.n, 1.0)])
        s = least_squares(self.fun, p0, bounds=(lb, ub), jac_sparsity=self.harva(), x_scale=sc, diff_step=2e-3, max_nfev=max_nfev,
                          method="trf", tr_solver="lsmr", verbose=verbose)
        return s


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("kansio")
    ap.add_argument("puoli")
    ap.add_argument("--ikkunat", default="8-22,56-66,104-114")
    ap.add_argument("--ymax", type=float, default=700.0)
    ap.add_argument("--gamma", type=float, default=kt.GAMMA)
    ap.add_argument("--saikeet", type=int, default=4)
    ap.add_argument("--max-nfev", type=int, default=100)
    ap.add_argument("--H", type=float, default=None, help="kiinnita korkeus H (cm), esim. kahvan korkeudesta")
    a = ap.parse_args()
    ikk = [tuple(float(x) for x in w.split("-")) for w in a.ikkunat.split(",")]
    cam = kv.Kamera(a.puoli, os.path.join(a.kansio, f"kalibrointi_{a.puoli}.json"), os.path.join(a.kansio, f"moodikuva_{a.puoli}.png"))
    t0 = time.time()
    H = kerää_havainnot(a.video, a.kansio, a.puoli, ikk, a.saikeet)
    idx = valitse(H, ymax=a.ymax)
    print(f"{len(H)} ruutua, opetukseen {len(idx)} ({time.time() - t0:.0f} s)", flush=True)
    F = kt.F_SOLMUT
    th0 = np.r_[kt.R_SOLMUT_CM, kt.H_CM, kt.SIGMA_OPT_PX, kt.REUNA_B_CM]
    kp = Kimppu(cam, H, idx, F, th0, a.gamma, H_kiinteä=a.H)
    print(kp.alut[-1], "residuaalia", flush=True)
    s = kp.ratkaise(th0, max_nfev=a.max_nfev, verbose=0)
    runko = kp.runko(s.x[:kp.nth])
    r = s.fun[:kp.alut[-1]]
    print(f"kimppusovitus {time.time() - t0:.0f} s: rms {np.sqrt(np.mean(r ** 2)):.4f}, H = {runko.H:.2f} cm, sigma_opt = {runko.sigma_opt:.2f} px, "
          f"reunan pehmeys b = {runko.b_cm:.3f} cm, gamma = {runko.gamma:.2f}")
    print("z (cm) :", np.round(runko.F * runko.H, 1))
    print("r (cm) :", np.round(runko.r, 2), f"  R_max = {runko.R_max:.2f}")
    polku = os.path.join(a.kansio, f"kivimalli_{a.puoli}.json")
    runko.tallenna(polku, ruutuja=len(idx), rms=round(float(np.sqrt(np.mean(r ** 2))), 4), lahde="kalibroi_kivi.py (kimppusovitus, radat_<puoli>.csv)")
    print("->", polku)
