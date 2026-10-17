"""Kiven kierre reaaliajassa: pyorimisnopeus ja -suunta heittopaan kamerasta, ja kierrosmaara hog-hog-valille.

Mittaus on sama kuin kierto_netti.py:ssa (offline): kahvan muovilevyn ulkopuolella nakyy kahvan palkki. Malli: vaakasuora sauva kiven akselin kautta korkeudella h
levyn ylapuolella; suunta theta(t) = theta0 + omega (t - t_keski) (+ = vastapaivaan ylhaalta katsoen). Malli piirretaan kameralla kuvaan ja sovitetaan muovin
todennakoisyyskarttaan (kahva_malli.havainto) irroituksen jalkeisissa vapaan lennon ruuduissa. Erot offline-versioon:
  * sauvan muoto (h, ell, w) on kiinnitetty (tulokset/kierto_muoto_<puoli>.json, offline-yhteissovituksen tulos varille) -> vain (theta0, omega, g) per heitto, joten
    yhden heiton sovitus on nopea (~2 s) ja ajetaan taustaprosessissa heiton paatyttya;
  * kiertosuunta valitaan kahden suunnan (omega > 0 / omega < 0) parhaiden sovitusten kustannuksesta;
  * kiven akseli tulee reaaliaikaisen seurannan (B) paikoista; ruudut keratty seurannan aikana (leike oikaistusta kuvasta, kuten km.leike).

Kierrosmaara: kierre vaimenee tasaisesti (IMU-mittauksissa n. 2 % / s, ks. Testi_09_03/README.md): omega(t) = omega_keski exp(-k (t - t_keski)). Kierrokset
aikavalilla [t1, t2] = omega_keski / (2 pi k) (exp(-k (t1 - t_keski)) - exp(-k (t2 - t_keski))). Valin [t1, t2] = hog-hog-aika luetaan videon puoliskoista
(kivi ylittaa heittopaan hog-viivan vasemmassa puoliskossa ja kaukopaan hog-viivan oikeassa puoliskossa)."""
import json
import os

import cv2
import numpy as np
from scipy.optimize import least_squares

import kahva_malli as km
import kivi_netti as kv
import kierto_netti as ko

K_KIERRE = 0.021                # kierteen vaimenemisvakio (1/s): IMU 2,1 % / s (Testi_09_02), vaihteluvali 2-3 % / s
MIN_RUUTUJA = 8                 # vahintaan taman verran kelvollisia ruutuja sovitukseen
MARGINAALI_VARMA = 0.03         # suunta varma, jos vastakkaisen suunnan paras kustannus on yli taman suhteessa suurempi


def lataa_muoto(kansio, puoli):
    """Sauvan muoto varille {vari: (h, ell, w)}: tulokset/kierto_muoto_<puoli>.json (puolen oma, muuten vasen), muuten {} (muoto sovitetaan vapaasti)."""
    for p in (puoli, "vasen"):
        polku = os.path.join(kansio, f"kierto_muoto_{p}.json")
        if os.path.exists(polku):
            d = json.load(open(polku))
            return {v: (float(m["h_cm"]), float(m["ell_cm"]), float(m["w_cm"])) for v, m in d.items() if isinstance(m, dict)}
    return {}


def kierrokset(omega_deg_s, t_keski, t1, t2, k=K_KIERRE):
    """Kierrosten maara (etumerkillinen: + = vastapaivaan) aikavalilla [t1, t2], kun omega(t) = omega_keski exp(-k (t - t_keski)). Ajat sekunteina."""
    w = np.radians(omega_deg_s) / (2.0 * np.pi)
    return float(w / k * (np.exp(-k * (t1 - t_keski)) - np.exp(-k * (t2 - t_keski))))


# ---------------------------------------------------------------------------------------------------------------------
# AINEISTO: leikkeista muovikartat (kuten kierto_netti.keraa_kehykset)
# ---------------------------------------------------------------------------------------------------------------------
def _ikkuna(crop, cx0, cy0, uc, vc, mode):
    """N x N -ikkuna (leike, taustakuva) pisteen (uc, vc) ymparilta talletetun leikkeen sisalta (kuten km.leike, mutta ilman reunojen replikointia): (c, m, x0, y0) tai None."""
    N, H = km.N, km.H
    hm, wm = mode.shape[:2]
    x0, y0 = int(round(uc)) - H, int(round(vc)) - H
    sx, sy = x0 - cx0, y0 - cy0
    if sx < 0 or sy < 0 or sx + N > crop.shape[1] or sy + N > crop.shape[0] or x0 < 0 or y0 < 0 or x0 + N > wm or y0 + N > hm:
        return None                                                        # leike ei kata ikkunaa (kivi lahella kuvan reunaa)
    return crop[sy:sy + N, sx:sx + N], mode[y0:y0 + N, x0:x0 + N], x0, y0


def rakenna_kehykset(cam, ast, vari, havainnot, t_irr, viive_s=ko.VIIVE_S, y_max=ko.Y_MAX_CM):
    """havainnot: [(t, f, Xb, Yb, crop, cx0, cy0)]: t = aika (s, aikaperusta), f = ruutunumero, (Xb, Yb) = kiven akseli (B-sovitus), crop = BGR-leike oikaistusta kuvasta
    (vasen ylakulma (cx0, cy0)). Kuten offline-ketjussa: ensin kahvan muovilevyn malli sovitetaan ruutuun (km.sovita_asetuksilla; alkuarvo B:sta) -> kahvan akseli (Xd, Yd),
    jonka suhteen palkin malli on kalibroitu (B:n ja kahvamallin akselit eroavat systemaattisesti ~2 cm), sitten muovikartta ja sovitusalue.
    Palauttaa (kehykset [dict(t, f, Y, p, roi, Xd, Yd, ox, oy, vari)], hylattyja kasi-ruutuja)."""
    N, W = km.N, ko.W
    z0 = ast["z0_cm"]
    kev = km.Kevyt(cam)
    out, hylatty = [], 0
    for t, f, Xb, Yb, crop, cx0, cy0 in havainnot:
        if t < t_irr + viive_s or Yb > y_max + 12.0:
            continue
        ub, vb = kv.kivimalli.project_3d(cam.K, cam.R, cam.t, np.array([[Xb, Yb, z0]]))
        w = _ikkuna(crop, cx0, cy0, ub[0], vb[0], cam.mode)
        if w is None:
            continue
        s = km.sovita_asetuksilla(kev, w[0], w[1], w[2], w[3], vari, (float(ub[0]), float(vb[0])), ast)
        if s is None:
            continue
        Xd, Yd = float(s["X"]), float(s["Y"])
        if Yd > y_max:
            continue
        u, v = kv.kivimalli.project_3d(cam.K, cam.R, cam.t, np.array([[Xd, Yd, z0]]))
        w = _ikkuna(crop, cx0, cy0, u[0], v[0], cam.mode)                  # ikkuna kahvan akselin ymparille (kuten offline km.leike)
        if w is None:
            continue
        c, mode, x0, y0 = w
        h = km.havainto(c, mode, vari)
        if h is None:
            continue
        p, w0 = h[0], h[1] > 0.5
        poly = km.soikio_kuvaan(cam, Xd, Yd, z0, ast["A_cm"], ast["B_cm"])
        disc = np.zeros((N, N), np.uint8)
        cv2.fillConvexPoly(disc, np.round((poly - np.array([x0, y0]) + 0.5) * 16).astype(np.int32), 1, cv2.LINE_8, 4)
        disc = cv2.dilate(disc, np.ones((3, 3), np.uint8)) > 0
        iu, iv = int(round(float(u[0]) - x0)), int(round(float(v[0]) - y0))
        if iu - 56 < 0 or iv - W < 0 or iu + 4 > N or iv + W > N:
            continue
        iu0, iv0 = iu - W, iv - W
        roi = np.zeros((2 * W, 2 * W), bool)
        roi[W - ko.DY_MAX:W + ko.DY_MAX, W + ko.DX_MIN:W + ko.DX_MAX] = True
        roi &= ~disc[iv0:iv0 + 2 * W, iu0:iu0 + 2 * W]
        kelvoton = roi & ~w0[iv0:iv0 + 2 * W, iu0:iu0 + 2 * W]                 # iho, hanska, hiha sovitusalueella
        if int(kelvoton.sum()) > ko.KASI_RAJA_PX:
            hylatty += 1
            continue
        roi &= ~kelvoton
        out.append(dict(t=float(t), f=float(f), Y=Yd, p=p[iv0:iv0 + 2 * W, iu0:iu0 + 2 * W].astype(np.float32), roi=roi, Xd=Xd, Yd=Yd, ox=x0 + iu0, oy=y0 + iv0, vari=vari))
    out.sort(key=lambda d: d["t"])
    return out, hylatty


# ---------------------------------------------------------------------------------------------------------------------
# SOVITUS
# ---------------------------------------------------------------------------------------------------------------------
def _alkuarvaukset(mal, kehykset, muoto):
    """Karkea (theta0, omega) -ruudukko kiintealla muodolla: paras alkuarvo kummallekin kiertosuunnalle ja ruudukon paras kustannus: ((c, th0, om)+, (c, th0, om)-)."""
    tm = float(np.mean([d["t"] for d in kehykset]))
    kok = kehykset[::2]
    paras = {1: None, -1: None}
    for th0 in range(0, 180, 15):
        for om in range(-150, 151, 15):
            if om == 0:
                continue
            c = 0.0
            for d in kok:
                c += ko.kustannus_ruudulle(mal, d, np.radians(th0 + om * (d["t"] - tm)), *muoto)[0]
            s = 1 if om > 0 else -1
            if paras[s] is None or c < paras[s][0]:
                paras[s] = (c, th0, om)
    return paras[1], paras[-1], tm


def _sovita(mal, kehykset, muoto, aloitus, tm, merkki, nfev=40):
    """(theta0, omega, g) kiintealla muodolla; omegan etumerkki rajattu (merkki = +1 / -1). Palauttaa (kustannus, [theta0, omega, g])."""
    h, ell, w = muoto
    c0, th0, om0 = aloitus

    def resid(q):
        out = []
        for d in kehykset:
            m = mal.piirra(d, np.radians(q[0] + q[1] * (d["t"] - tm)), h, ell, w)
            out.append((d["p"] - q[2] * m)[d["roi"]])
        return np.concatenate(out)

    lo = [-400.0, 3.0 if merkki > 0 else -300.0, 0.2]
    hi = [400.0, 300.0 if merkki > 0 else -3.0, 2.5]
    r = least_squares(resid, np.array([th0, om0, 0.8]), bounds=(lo, hi), x_scale=[20, 20, 0.2], max_nfev=nfev, diff_step=0.02, loss="soft_l1", f_scale=0.15)
    return float(r.cost), r.x


def arvioi(cam, ast, vari, havainnot, t_irr, muoto=None):
    """Yhden heiton kierre. Palauttaa dict(omega_deg_s [+ = vastapaivaan], t_keski, ruutuja, kustannus_ccw, kustannus_cw, marginaali, varma, ...) tai dict(virhe=...)."""
    kehykset, hylatty = rakenna_kehykset(cam, ast, vari, havainnot, t_irr)
    if len(kehykset) < MIN_RUUTUJA:
        return dict(virhe=f"liian vahan kelvollisia ruutuja ({len(kehykset)}, hylatty kasi {hylatty})", ruutuja=len(kehykset))
    mal = ko.Malli(cam, ast["z0_cm"], ast["sigma_px"])
    if muoto is None:                                          # muoto vapaasti: offline-yhteissovituksen tapa yhdelle heitolle
        a1, a2, tm = _alkuarvaukset(mal, kehykset, (ko.H_ALKU, ko.ELL_ALKU, ko.W_ALKU))
        alo = [a1, a2] if a1[0] <= a2[0] else [a2, a1]
        c, x = ko.sovita_ryhma(mal, [kehykset], [[(a[0], a[1], a[2]) for a in alo]], [tm])
        h, ell, w = x[:3]
        muoto_k = (float(h), float(ell), float(w))
        om_sov, th0, g = float(x[4]), float(x[3]), float(x[5])
        kc = {1: float("nan"), -1: float("nan")}
        kc[1 if om_sov > 0 else -1] = float(c)
    else:
        muoto_k = tuple(muoto)
        a_ccw, a_cw, tm = _alkuarvaukset(mal, kehykset, muoto_k)
        r_ccw = _sovita(mal, kehykset, muoto_k, a_ccw, tm, +1)
        r_cw = _sovita(mal, kehykset, muoto_k, a_cw, tm, -1)
        kc = {1: r_ccw[0], -1: r_cw[0]}
        c, (th0, om_sov, g) = min((r_ccw, r_cw), key=lambda r: r[0])
    ruudut = ko.kulma_ruuduittain(mal, kehykset, muoto_k, th0, om_sov, tm, g)
    tt = np.array([x[0] for x in ruudut])
    aa = np.array([x[1] for x in ruudut])
    om_ts = ko.theil_sen(tt, aa) if len(tt) >= 4 else om_sov
    om = 0.5 * (om_sov + om_ts)
    voittaja, havio = (1, -1) if om_sov > 0 else (-1, 1)
    marg = float("nan")
    if np.isfinite(kc[havio]) and kc[voittaja] > 0:
        marg = float((kc[havio] - kc[voittaja]) / kc[voittaja])
    return dict(omega_deg_s=float(om), omega_sovitus_deg_s=float(om_sov), omega_theilsen_deg_s=float(om_ts), theta0_deg=float(th0), g=float(g),
                t_keski=float(tm), ruutuja=len(kehykset), hylatty_kasi=int(hylatty), kustannus_ccw=kc[1], kustannus_cw=kc[-1], marginaali=marg,
                varma=bool(np.isfinite(marg) and marg >= MARGINAALI_VARMA and np.sign(om_sov) == np.sign(om_ts)), muoto=dict(h_cm=muoto_k[0], ell_cm=muoto_k[1], w_cm=muoto_k[2]),
                f_keski=float(np.mean([d["f"] for d in kehykset])))
