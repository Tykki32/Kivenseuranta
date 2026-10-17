"""Kiven kierre (pyorimisnopeus ja -suunta) heittopaan kamerasta: python kierto_netti.py VIDEO tulokset vasen [--ulos tulokset/kierto_vasen] [--ei-kuvia]

Kahvan muovilevyn (kahva_malli.py) ulkopuolella nakyy kahvan palkki. Malli: vaakasuora sauva, joka kulkee kiven akselin kautta korkeudella h muovilevyn
yla-puolella (pituus ell, leveys w). Sauvan suunta theta (X-akselista Y-akseliin, + = vastapaivaan ylhaalta katsoen) kiertyy tasaisesti:
theta(t) = theta0 + omega (t - t_keski). Malli piirretaan kameralla kuvaan (kuvan sumeus sigma_px huomioiden) ja sovitetaan muovin
todennakoisyyskarttaan (kahva_malli.havainto) levyn ulkopuolella, kaikkiin vapaan lennon ruutuihin yhdessa; muoto (h, ell, w) on yhteinen saman
vareisille kahvoille, theta0, omega ja kirkkaus heittokohtaisia. Sauva on symmetrinen, joten theta on maaratty modulo 180 astetta; kiertosuunta
nakyy sauvan nakyvan paan kallistuksen (ylos/alas) muutoksesta, ja kierron suunta on se, jolla sovitus on selvasti parempi.
Aika on todellinen aika t_kor_s (aikaperusta.py). Vain heittopaan kamera (vasen); vapaan lennon ruudut irroituksen (liuku_irroitus.py) jalkeen,
ei kasi-ruutuja (kasi tunnistetaan muovimaskin vasemmasta reunasta).

Kirjoittaa <ulos>.json, <ulos>.png (kulma(t) ja kustannus(omega)) ja <ulos>_sovitus.png (malli vs havainto)."""
import argparse
import csv
import json
import os
import pickle
import time

import cv2
import numpy as np
from scipy.optimize import least_squares

import kahva_malli as km
import kivi_netti as kv

W = 40                       # leikkeen puolikas (px) sovituksessa
SS = 3                       # polygonin ylinaytteistus
DX_MIN, DX_MAX = -36, -2     # sovitusalue levyn keskipisteen vasemmalla puolella (px): kahva nakyy korkeuden vuoksi vasemmalla
DY_MAX = 38
KASI_RAJA_PX = 60            # sovitusalueella enintaan tama monta "ei valia" -pikselia (iho, hanska, hiha; kahva_malli.havainto): muuten kasi on kahvassa -> ruutu hylataan
VIIVE_S = 0.15               # vapaan lennon ruudut vasta irroituksen jalkeen
Y_MAX_CM = 700.0             # kuvan reuna: kivi leikkautuu pois
H_ALKU, ELL_ALKU, W_ALKU = 5.5, 13.0, 4.5
PROFIILI_DC = 5.0            # omegan vaihteluvali: profiilin kohdat, joissa kustannus <= minimi + tama (kustannus ~300-380)
ALARAJAT = [3.0, 7.0, 2.0]
YLARAJAT = [9.0, 20.0, 7.0]


# ---------------------------------------------------------------------------------------------------------------------
# AINEISTO
# ---------------------------------------------------------------------------------------------------------------------
def lue_aika(kansio, puoli):
    """{(rata, ruutu): t_kor_s} kahvaradat-tiedostosta."""
    return {(int(r["rata"]), int(r["ruutu"])): float(r["t_kor_s"]) for r in csv.DictReader(open(os.path.join(kansio, f"kahvaradat_{puoli}.csv")))}


def keraa_kehykset(video, kansio, puoli, t_irroitus):
    """Vapaan lennon ruutujen muovikartat. t_irroitus: {rata: t_r}. Palauttaa {rata: [dict(t, Y, p, roi, Xd, Yd, ox, oy, vari)]}."""
    cam = kv.Kamera(puoli, os.path.join(kansio, f"kalibrointi_{puoli}.json"), os.path.join(kansio, f"moodikuva_{puoli}.png"))
    cap = cv2.VideoCapture(video)
    ast = km.lataa_asetukset(kansio, puoli)
    aika = lue_aika(kansio, puoli)
    out = {k: [] for k in t_irroitus}
    hylatty = {k: 0 for k in t_irroitus}
    for r in csv.DictReader(open(os.path.join(kansio, f"kahvamalli_{puoli}.csv"))):
        k = int(r["rata"])
        f = int(r["frame"])
        Xd, Yd = float(r["X_akseli_cm"]), float(r["Y_akseli_cm"])
        t = aika.get((k, f))
        if t is None or k not in t_irroitus or t < t_irroitus[k] + VIIVE_S or Yd > Y_MAX_CM:
            continue
        fr = cam.ruutu(cap, f)
        u, v = kv.kivimalli.project_3d(cam.K, cam.R, cam.t, np.array([[Xd, Yd, ast["z0_cm"]]]))
        crop, mode, x0, y0 = km.leike(cam, fr, (float(u[0]), float(v[0])))
        h = km.havainto(crop, mode, r["vari"])
        if h is None:
            continue
        p, w0 = h[0], h[1] > 0.5
        N = p.shape[0]
        poly = km.soikio_kuvaan(cam, Xd, Yd, ast["z0_cm"], ast["A_cm"], ast["B_cm"])
        disc = np.zeros((N, N), np.uint8)
        cv2.fillConvexPoly(disc, np.round((poly - np.array([x0, y0]) + 0.5) * 16).astype(np.int32), 1, cv2.LINE_8, 4)
        disc = cv2.dilate(disc, np.ones((3, 3), np.uint8)) > 0
        iu, iv = int(round(u[0] - x0)), int(round(v[0] - y0))
        if iu - 56 < 0 or iv - W < 0 or iu + 4 > N or iv + W > N:
            continue
        iu0, iv0 = iu - W, iv - W
        roi = np.zeros((2 * W, 2 * W), bool)
        roi[W - DY_MAX:W + DY_MAX, W + DX_MIN:W + DX_MAX] = True
        roi &= ~disc[iv0:iv0 + 2 * W, iu0:iu0 + 2 * W]
        kelvoton = roi & ~w0[iv0:iv0 + 2 * W, iu0:iu0 + 2 * W]                 # iho, hanska, hiha sovitusalueella
        if int(kelvoton.sum()) > KASI_RAJA_PX:
            hylatty[k] += 1
            continue
        roi &= ~kelvoton                                                       # iho ei vaikuta sovitukseen (kuten kahva_malli.sovita)
        out[k].append(dict(t=t, Y=Yd, p=p[iv0:iv0 + 2 * W, iu0:iu0 + 2 * W].astype(np.float32), roi=roi, Xd=Xd, Yd=Yd, ox=x0 + iu0, oy=y0 + iv0, vari=r["vari"]))
    for k in out:
        out[k].sort(key=lambda d: d["t"])
    return out, hylatty, cam, ast


# ---------------------------------------------------------------------------------------------------------------------
# MALLI
# ---------------------------------------------------------------------------------------------------------------------
class Malli:
    """Palkkimallin piirto kuvaan (kameran K, R, t; levyn korkeus z0; kuvan sumeus sigma)."""

    def __init__(self, cam, z0, sigma):
        self.K, self.R, self.t, self.z0, self.sigma = cam.K, cam.R, cam.t, z0, sigma

    def piirra(self, d, theta, h, ell, w):
        e = np.array([np.cos(theta), np.sin(theta)])
        n = np.array([-np.sin(theta), np.cos(theta)])
        c0 = np.array([d["Xd"], d["Yd"]])
        a0, a1 = c0 - e * ell / 2, c0 + e * ell / 2
        z = self.z0 + h
        pts = np.array([[*(a0 + n * w / 2), z], [*(a1 + n * w / 2), z], [*(a1 - n * w / 2), z], [*(a0 - n * w / 2), z]])
        u, v = kv.kivimalli.project_3d(self.K, self.R, self.t, pts)
        q = np.column_stack([u - d["ox"], v - d["oy"]])
        m = np.zeros((2 * W * SS, 2 * W * SS), np.uint8)
        cv2.fillConvexPoly(m, np.round((q + 0.5) * SS * 16).astype(np.int32), 1, cv2.LINE_8, 4)
        cov = cv2.resize(m.astype(np.float32), (2 * W, 2 * W), interpolation=cv2.INTER_AREA)
        return cv2.GaussianBlur(cov, (0, 0), self.sigma)


def _g_paras(model, p, roi):
    """Kirkkauden g paras arvo (pienimman neliosumma): p ~ g * malli."""
    mm, pp = model[roi], p[roi]
    return float(np.clip(mm @ pp / max(mm @ mm, 1e-9), 0.2, 2.5))


def kustannus_ruudulle(mal, d, theta, h, ell, w, g=None):
    m = mal.piirra(d, theta, h, ell, w)
    if g is None:
        g = _g_paras(m, d["p"], d["roi"])
    return float(np.sum((d["p"] - g * m)[d["roi"]] ** 2)), g


def alkuarvaukset(mal, kehykset, n=3, shape=(H_ALKU, ELL_ALKU, W_ALKU)):
    """Karkea (theta0, omega) -ruudukkohaku kiintealla muodolla; palauttaa n parasta eri (theta0 mod 180, omega) -paria."""
    tm = np.mean([d["t"] for d in kehykset])
    kok = kehykset[::2]
    tulokset = []
    for th0 in range(0, 180, 15):
        for om in range(-150, 151, 15):
            c = 0.0
            for d in kok:
                c += kustannus_ruudulle(mal, d, np.radians(th0 + om * (d["t"] - tm)), *shape)[0]
            tulokset.append((c, th0, om))
    tulokset.sort()
    valitut = []
    for c, th0, om in tulokset:
        if all(abs(om - o) >= 25 for _, _, o in valitut):
            valitut.append((c, th0, om))
        if len(valitut) >= n:
            break
    return valitut


def sovita_ryhma(mal, ryhma, aloitukset, tm, kiinteat_omega=None, nfev=40, kappa=False):
    """Yhteissovitus: muoto (h, ell, w) jaettu, per heitto (theta0, omega, g[, kappa]). ryhma: [kehykset_heitto1, ...], aloitukset: [[(c, th0, om), ...] per heitto]
    (kaikki yhdistelmat kokeillaan). kiinteat_omega: [omega_k tai None] (profiili). Palauttaa (kustannus, theta)."""
    nper = 4 if kappa else 3
    yhdistelmat = [[]]
    for al in aloitukset:
        yhdistelmat = [y + [a] for y in yhdistelmat for a in al]
    lo = list(ALARAJAT) + sum([[-400, -300, 0.2] + ([-60] if kappa else []) for _ in ryhma], [])
    hi = list(YLARAJAT) + sum([[400, 300, 2.5] + ([60] if kappa else []) for _ in ryhma], [])

    def laske(theta):
        out = []
        h, ell, w = theta[:3]
        pos = 3
        for j, kehykset in enumerate(ryhma):
            th0, om, g = theta[pos:pos + 3]
            kap = theta[pos + 3] if kappa else 0.0
            pos += nper
            for d in kehykset:
                dt = d["t"] - tm[j]
                a = np.radians(th0 + om * dt + 0.5 * kap * dt * dt)
                m = mal.piirra(d, a, h, ell, w)
                out.append((d["p"] - g * m)[d["roi"]])
        return out

    def resid(theta):
        return np.concatenate(laske(theta))

    paras = None
    for yh in yhdistelmat:
        x0 = [H_ALKU, ELL_ALKU, W_ALKU]
        for (c, th0, om) in yh:
            x0 += [th0, om, 0.8] + ([0.0] if kappa else [])
        lo2, hi2 = list(lo), list(hi)
        if kiinteat_omega is not None:
            pos = 3
            for j in range(len(ryhma)):
                if kiinteat_omega[j] is not None:
                    x0[pos + 1] = kiinteat_omega[j]
                    lo2[pos + 1], hi2[pos + 1] = kiinteat_omega[j] - 1e-6, kiinteat_omega[j] + 1e-6
                pos += nper
        try:
            r = least_squares(resid, np.array(x0), bounds=(lo2, hi2), x_scale=[1, 2, 0.5] + sum([[20, 20, 0.2] + ([10] if kappa else []) for _ in ryhma], []),
                              max_nfev=nfev, diff_step=0.02, loss="soft_l1", f_scale=0.15)
        except Exception:
            continue
        if paras is None or r.cost < paras[0]:
            paras = (r.cost, r.x)
    return paras


def theil_sen(t, y, min_dt=0.15):
    """Robusti kulmakerroin (parittaisten kulmakertoimien mediaani)."""
    s = [(y[j] - y[i]) / (t[j] - t[i]) for i in range(len(t)) for j in range(i + 1, len(t)) if t[j] - t[i] > min_dt]
    return float(np.median(s))


def kulma_ruuduittain(mal, kehykset, shape, th0, om, tm, g):
    """Jokaiselle ruudulle paras theta (mod 180, 2 asteen ruudukko), kun muoto ja kirkkaus kiinnitetty; palautetaan sovituksen viivaa lahin vaihtoehto."""
    out = []
    for d in kehykset:
        ennuste = th0 + om * (d["t"] - tm)
        ehdokkaat = np.arange(-90, 90, 2.0)
        c = [kustannus_ruudulle(mal, d, np.radians(a), *shape, g)[0] for a in ehdokkaat]
        a = ehdokkaat[int(np.argmin(c))]
        # lahin ekvivalentti kulma (mod 180) ennusteeseen
        a = a + 180.0 * np.round((ennuste - a) / 180.0)
        out.append((d["t"], a, min(c)))
    return out


# ---------------------------------------------------------------------------------------------------------------------
# AJO
# ---------------------------------------------------------------------------------------------------------------------
def analysoi(video, kansio, puoli, ulos, kuvat=True, valimuisti=None):
    liuku = json.load(open(os.path.join(kansio, f"liuku_irroitus_{puoli}.json")))
    t_irr = {int(k): v["irroitus"]["t_r"] for k, v in liuku.items()}
    if valimuisti and os.path.exists(valimuisti):
        kehykset, hylatty, ast, vari = pickle.load(open(valimuisti, "rb"))
        cam = kv.Kamera(puoli, os.path.join(kansio, f"kalibrointi_{puoli}.json"), os.path.join(kansio, f"moodikuva_{puoli}.png"))
    else:
        kehykset, hylatty, cam, ast = keraa_kehykset(video, kansio, puoli, t_irr)
        vari = {k: (kehykset[k][0]["vari"] if kehykset[k] else None) for k in kehykset}
        if valimuisti:
            pickle.dump((kehykset, hylatty, ast, vari), open(valimuisti, "wb"))
    mal = Malli(cam, ast["z0_cm"], ast["sigma_px"])
    ryhmat = {}
    for k in sorted(kehykset):
        ryhmat.setdefault(vari[k], []).append(k)
    tulos = {}
    t0 = time.time()
    for v_, heitot in ryhmat.items():
        ryhma = [kehykset[k] for k in heitot]
        tm = [float(np.mean([d["t"] for d in kk])) for kk in ryhma]
        alo = [alkuarvaukset(mal, kk) for kk in ryhma]
        c, x = sovita_ryhma(mal, ryhma, alo, tm)
        h, ell, w = x[:3]
        ryhma_tulos = {}
        for j, k in enumerate(heitot):
            th0, om, g = x[3 + 3 * j:6 + 3 * j]
            # profiili omegan suhteen (muu uudelleen optimoitu)
            profiili = []
            for om_k in range(-120, 151, 15):
                om_fix = [None] * len(heitot)
                om_fix[j] = float(om_k)
                pr = sovita_ryhma(mal, ryhma, [[(0, th0, om_k)] if i == j else [(0, x[3 + 3 * i], x[4 + 3 * i])] for i in range(len(heitot))], tm, kiinteat_omega=om_fix, nfev=15)
                profiili.append((om_k, pr[0]))
            # puolikkaat (muoto kiinnitetty): ensimmainen ja toinen puoli
            puoli_om = []
            kk = ryhma[j]
            for osa in (kk[:len(kk) // 2], kk[len(kk) // 2:]):
                tmo = float(np.mean([d["t"] for d in osa]))
                best = None
                for th_ in range(0, 180, 10):
                    for om_ in range(-60, 181, 10):
                        c_ = sum(kustannus_ruudulle(mal, d, np.radians(th_ + om_ * (d["t"] - tmo)), h, ell, w, g)[0] for d in osa[::2])
                        if best is None or c_ < best[0]:
                            best = (c_, th_, om_)
                puoli_om.append(best[2])
            ruudut = kulma_ruuduittain(mal, kk, (h, ell, w), th0, om, tm[j], g)
            tt = np.array([x[0] for x in ruudut])
            aa = np.array([x[1] for x in ruudut])
            ts = theil_sen(tt, aa)
            rng = np.random.default_rng(0)
            bs = [theil_sen(tt[ii], aa[ii]) for ii in (rng.choice(len(tt), len(tt)) for _ in range(300))]
            ts_vali = [float(np.percentile(bs, 5)), float(np.percentile(bs, 95))]
            ryhma_tulos[k] = dict(omega_sovitus_deg_s=float(om), omega_theilsen_deg_s=ts, omega_theilsen_vali_deg_s=ts_vali, omega_deg_s=float(0.5 * (om + ts)), theta0_deg=float(th0), g=float(g), t_keski=tm[j], ruutuja=len(kk), t_alku=float(kk[0]["t"]), t_loppu=float(kk[-1]["t"]),
                                  profiili=profiili, puolikkaat_omega=puoli_om, kustannus=float(c), ruudut=ruudut,
                                  hylatty_kasi=hylatty[k], muoto=dict(h_cm=float(h), ell_cm=float(ell), w_cm=float(w)), vari=v_)
        tulos.update(ryhma_tulos)
        print(f"{v_}: heitot {[k + 1 for k in heitot]}  muoto h = {h:.2f} cm, pituus {ell:.1f} cm, leveys {w:.1f} cm ({time.time() - t0:.0f} s)")
    for k in sorted(tulos):
        r = tulos[k]
        # suunta ja epavarmuus: profiilin minimit vastapaivaan/myotapaivaan
        om = r["omega_deg_s"]
        prof = np.array(r["profiili"] + [(r["omega_sovitus_deg_s"], r["kustannus"])])
        prof = prof[np.argsort(prof[:, 0])]
        pos = prof[prof[:, 0] > 5][:, 1].min()
        neg = prof[prof[:, 0] < -5][:, 1].min()
        nol = prof[np.abs(prof[:, 0]) <= 5][:, 1].min()
        r["kustannus_vastapaivaan"], r["kustannus_myotapaivaan"], r["kustannus_ei_kiertoa"] = float(pos), float(neg), float(nol)
        # omegan vaihteluvali: profiilin kohdat, joissa kustannus <= minimi + DC (lineaarinen interpolointi reunoilla)
        o, c = prof[:, 0], prof[:, 1]
        cmin = c.min()
        ok = np.where(c <= cmin + PROFIILI_DC)[0]
        lo_i, hi_i = ok.min(), ok.max()
        lo = o[lo_i] if lo_i == 0 else o[lo_i - 1] + (o[lo_i] - o[lo_i - 1]) * (cmin + PROFIILI_DC - c[lo_i - 1]) / (c[lo_i] - c[lo_i - 1])
        hi = o[hi_i] if hi_i == len(o) - 1 else o[hi_i] + (o[hi_i + 1] - o[hi_i]) * (cmin + PROFIILI_DC - c[hi_i]) / (c[hi_i + 1] - c[hi_i])
        r["omega_vali_profiili_deg_s"] = [float(lo), float(hi)]
        # lopullinen: kahden estimaattorin keskiarvo (yhteissovitus, ruutukohtaisten kulmien Theil-Sen); vali = Theil-Senin bootstrap-90 %-vali, laajennettuna kattamaan yhteissovitus
        tv = r["omega_theilsen_vali_deg_s"]
        r["omega_vali_deg_s"] = [float(min(tv[0], r["omega_sovitus_deg_s"])), float(max(tv[1], r["omega_sovitus_deg_s"]))]
        r["suunta"] = "vastapaivaan" if om > 0 else "myotapaivaan"
        print(f"heitto {k + 1}: kierre {om:+.0f} deg/s ({r['suunta']}; 1 kierros / {360.0 / abs(om):.1f} s; vali {r['omega_vali_deg_s'][0]:.0f}...{r['omega_vali_deg_s'][1]:.0f}; "
              f"yhteissovitus {r['omega_sovitus_deg_s']:.0f}, Theil-Sen {r['omega_theilsen_deg_s']:.0f}), "
              f"ruutuja {r['ruutuja']} (t {r['t_alku']:.2f}...{r['t_loppu']:.2f}, kasi-ruutuja hylatty {r['hylatty_kasi']}); puolikkaat {r['puolikkaat_omega']}, "
              f"kustannus vastapaivaan {pos:.1f} / myotapaivaan {neg:.1f} / ei kiertoa {nol:.1f}")
    json.dump({str(k): v for k, v in tulos.items()}, open(ulos + ".json", "w"), indent=1)
    muodot = {}                                           # palkin muoto varille: netti_kierre.py (reaaliaikainen kierre) kayttaa tata kiinteana
    for k in sorted(tulos):
        muodot[tulos[k]["vari"]] = {n: round(float(v), 3) for n, v in tulos[k]["muoto"].items()}
    muodot["_lahde"] = "kierto_netti.py yhteissovitus (vareittain), " + os.path.basename(ulos) + ".json"
    json.dump(muodot, open(os.path.join(os.path.dirname(ulos) or ".", f"kierto_muoto_{puoli}.json"), "w"), indent=1)
    if kuvat:
        piirra(tulos, kehykset, mal, ulos)
    return tulos, kehykset, mal


def piirra(tulos, kehykset, mal, ulos):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    vari = {0: "#d4a017", 1: "#c0392b", 2: "#2c7fb8"}
    n = len(tulos)
    fig, ax = plt.subplots(2, n, figsize=(5.2 * n, 7.5))
    for j, k in enumerate(sorted(tulos)):
        r = tulos[k]
        a = ax[0, j]
        t = np.array([x[0] for x in r["ruudut"]])
        th = np.array([x[1] for x in r["ruudut"]])
        a.plot(t - r["t_keski"], th, ".", color=vari[k], ms=5, label="ruutukohtainen sauvan suunta")
        tt = np.linspace(t.min(), t.max(), 20) - r["t_keski"]
        a.plot(tt, r["theta0_deg"] + r["omega_sovitus_deg_s"] * tt, "k-", lw=1.2, label=f"yhteissovitus: {r['omega_sovitus_deg_s']:+.0f} astetta/s")
        c0 = float(np.median(th - r["omega_theilsen_deg_s"] * (t - r["t_keski"])))
        a.plot(tt, c0 + r["omega_theilsen_deg_s"] * tt, "--", color="0.35", lw=1.2, label=f"Theil-Sen: {r['omega_theilsen_deg_s']:+.0f} astetta/s")
        a.set(xlabel="aika (s, vapaan lennon keskipisteesta)", ylabel="sauvan suunta theta (aste, + = vastapaivaan)",
              title=f"heitto {k + 1}: {r['suunta']}, {abs(r['omega_deg_s']):.0f} astetta/s ({r['omega_vali_deg_s'][0]:.0f}...{r['omega_vali_deg_s'][1]:.0f})")
        a.legend(fontsize=7)
        a.grid(alpha=0.3)
        a = ax[1, j]
        prof = np.array(r["profiili"])
        a.plot(prof[:, 0], prof[:, 1], "-o", color=vari[k], ms=4)
        a.axvline(r["omega_sovitus_deg_s"], color="k", lw=1)
        a.axvline(0, color="0.6", lw=0.8)
        a.set(xlabel="kierto omega (aste/s, + = vastapaivaan)", ylabel="sovituksen kustannus", title="kustannus omegan funktiona (muu optimoitu)")
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ulos + ".png", dpi=100)
    # malli vs havainto
    fig, ax = plt.subplots(2 * n, 6, figsize=(12, 4.2 * n))
    for row, k in enumerate(sorted(tulos)):
        r = tulos[k]
        kk = kehykset[k]
        m = r["muoto"]
        idx = np.linspace(0, len(kk) - 1, 6).astype(int)
        for jj, i in enumerate(idx):
            d = kk[i]
            th = np.radians(r["theta0_deg"] + r["omega_sovitus_deg_s"] * (d["t"] - r["t_keski"]))
            mod = mal.piirra(d, th, m["h_cm"], m["ell_cm"], m["w_cm"])
            for q in (0, 1):
                a = ax[2 * row + q, jj]
                a.imshow(d["p"], vmin=0, vmax=1, cmap="gray")
                if q:
                    a.contour(r["g"] * mod, [0.25, 0.5], colors=["yellow", "red"], linewidths=0.9)
                a.contour(d["roi"].astype(float), [0.5], colors="c", linewidths=0.4)
                a.axis("off")
            ax[2 * row, jj].set_title(f"h{k + 1} Y={d['Y']:.0f} th={np.degrees(th) % 180:.0f}", fontsize=8)
    fig.tight_layout()
    fig.savefig(ulos + "_sovitus.png", dpi=70)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("kansio")
    ap.add_argument("puoli")
    ap.add_argument("--ulos", default=None)
    ap.add_argument("--ei-kuvia", action="store_true")
    ap.add_argument("--valimuisti", default=None, help="pickle-valimuisti ruutujen muovikartoille (nopeuttaa uusintaajoja)")
    a = ap.parse_args()
    analysoi(a.video, a.kansio, a.puoli, a.ulos or os.path.join(a.kansio, f"kierto_{a.puoli}"), not a.ei_kuvia, a.valimuisti)
