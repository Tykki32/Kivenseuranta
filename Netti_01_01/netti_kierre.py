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
MIN_RUUTUJA_VARMA = 12          # tulos on "varma" vasta taman verran ruutuja kattavasta ikkunasta (8-10 ruudun mittaus antoi -27 / -105 / -159 deg/s, kun muut heitot 40-110)
MARGINAALI_VARMA = 0.03         # suunta varma, jos vastakkaisen suunnan paras kustannus on yli taman suhteessa suurempi
OMEGA_MAX_VARMA = 130.0         # pyorimisnopeus (deg/s), jonka yli tulos on epauskottava (tavallisesti 40-110; 130 deg/s = noin 4,5 kierrosta hog-hog-valilla; lyhyella ikkunalla 10 ruutua mitattu 159 / 213 deg/s ei ole luotettava): ei "varma"
ALKU_MAX = 3                    # kuinka monta paikallista minimia (omega-profiilissa) jalostetaan per kiertosuunta: kiertomittauksen ruudut voivat olla ryhmissa (kasi / harjaajat peittavat), jolloin
                                # omegan profiilissa on useita (~40 deg/s:n valein) lähes yhta hyvia minimeja (alias); vain ruudukon paras minimi ei riita
ALKU_SUHDE = 1.15               # jalostetaan vain minimit, joiden karkea kustannus on enintaan tama kerrannainen suunnan parhaasta
ENNAKKO_PAINO = 2.0             # omega-ennakkotiedon (toisen paan mittaus + vaimeneminen) sakko kustannusyksikkoina: 0,5 ((omega - ennakko) / sigma)^2 x taman; ratkaisee vain, kun alias-minimien kustannukset ovat lahes samat
ENNAKKO_SIGMA_OSUUS = 0.3       # ennakkotiedon epavarmuus osuutena omegasta (vaimenemisvakion hajonta ~ 0,013 / s x 20 s) ...
ENNAKKO_SIGMA_MIN = 10.0        # ... ja vahintaan (deg/s)


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


def rakenna_kehykset(cam, ast, vari, havainnot, t_irr, viive_s=ko.VIIVE_S, y_max=ko.Y_MAX_CM, kasi_raja=ko.KASI_RAJA_PX, roi_min=0):
    """havainnot: [(t, f, Xb, Yb, crop, cx0, cy0)]: t = aika (s, aikaperusta), f = ruutunumero, (Xb, Yb) = kiven akseli (B-sovitus), crop = BGR-leike oikaistusta kuvasta
    (vasen ylakulma (cx0, cy0)). kasi_raja = ruudun hylkaamisen raja (sovitusalueen "ei valia" -pikselit: iho, hanska, hiha, sweeperin housut); sallitut pikselit jatetaan sovituksen ulkopuolelle,
    ja roi_min = vahintaan tama monta sovitusalueen pikselia jaa jaljelle. Kuten offline-ketjussa: ensin kahvan muovilevyn malli sovitetaan ruutuun (km.sovita_asetuksilla; alkuarvo B:sta) -> kahvan akseli (Xd, Yd),
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
        if int(kelvoton.sum()) > kasi_raja:
            hylatty += 1
            continue
        roi &= ~kelvoton
        if int(roi.sum()) < roi_min:
            hylatty += 1
            continue
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


def _alkuarvaukset_monta(mal, kehykset, muoto, n=ALKU_MAX):
    """Kuten _alkuarvaukset, mutta jokaiselle kiertosuunnalle useita (enintaan n) omega-profiilin paikallista minimia (kustannus, theta0, omega) kustannusjarjestyksessa; mukana vain
    minimit, joiden kustannus on enintaan ALKU_SUHDE x suunnan paras. Ruudukko: omega 10 deg/s:n valein (alias-minimit ~40 deg/s:n valein), theta0 20 asteen valein."""
    tm = float(np.mean([d["t"] for d in kehykset]))
    kok = kehykset[::2]
    profiili = {1: {}, -1: {}}
    for om in range(-150, 151, 10):
        if om == 0:
            continue
        paras = None
        for th0 in range(0, 180, 20):
            c = 0.0
            for d in kok:
                c += ko.kustannus_ruudulle(mal, d, np.radians(th0 + om * (d["t"] - tm)), *muoto)[0]
            if paras is None or c < paras[0]:
                paras = (c, th0, om)
        profiili[1 if om > 0 else -1][om] = paras
    out = {}
    for s_, pr in profiili.items():
        oms = sorted(pr)
        minimit = [pr[om] for i, om in enumerate(oms)
                   if (i == 0 or pr[om][0] <= pr[oms[i - 1]][0]) and (i == len(oms) - 1 or pr[om][0] <= pr[oms[i + 1]][0])]
        minimit.sort(key=lambda x: x[0])
        out[s_] = [m for m in minimit if m[0] <= ALKU_SUHDE * minimit[0][0]][:n]
    return out, tm


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


def _ennakko(omega_ennakko, tm):
    """omega_ennakko = (omega_deg_s, t_s) toisen paan mittaus: odotettu omega hetkella tm vaimenemisvakiolla K_KIERRE -> (omega_deg_s, sigma_deg_s) tai None."""
    if omega_ennakko is None:
        return None
    om = float(omega_ennakko[0]) * float(np.exp(-K_KIERRE * (tm - float(omega_ennakko[1]))))
    return om, max(ENNAKKO_SIGMA_MIN, ENNAKKO_SIGMA_OSUUS * abs(om))


def arvioi(cam, ast, vari, havainnot, t_irr, muoto=None, omega_ennakko=None, kasi_raja=ko.KASI_RAJA_PX, roi_min=0):
    """Yhden heiton kierre. Palauttaa dict(omega_deg_s [+ = vastapaivaan], t_keski, ruutuja, kustannus_ccw, kustannus_cw, marginaali, varma, ...) tai dict(virhe=...).
    omega_ennakko = (omega_deg_s, t_s): toisen paan mittaus; kun ruutujen sovituksessa on useita lahes yhta hyvia omega-minimeja (alias), ennakko (vaimeneminen mittaushetkesta) ratkaisee
    niiden valilla (ENNAKKO_PAINO); selvasti parempi minimi voittaa ennakosta riippumatta."""
    kehykset, hylatty = rakenna_kehykset(cam, ast, vari, havainnot, t_irr, kasi_raja=kasi_raja, roi_min=roi_min)
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
        alut, tm = _alkuarvaukset_monta(mal, kehykset, muoto_k)
        enn = _ennakko(omega_ennakko, tm)
        voittaja_s = 1 if alut[1][0][0] <= alut[-1][0][0] else -1
        kc, ehdokkaat = {}, []
        for s_ in (1, -1):
            for st in (alut[s_] if s_ == voittaja_s else alut[s_][:1]):          # voittavalle suunnalle kaikki alias-minimit, toiselle vain paras (marginaalia varten)
                r_ = _sovita(mal, kehykset, muoto_k, st, tm, s_, nfev=30)
                kc[s_] = min(kc.get(s_, np.inf), r_[0])
                sakko = 0.0 if enn is None else ENNAKKO_PAINO * 0.5 * ((r_[1][1] - enn[0]) / enn[1]) ** 2
                ehdokkaat.append((r_[0] + sakko, r_[0], r_[1]))
        _, c, (th0, om_sov, g) = min(ehdokkaat, key=lambda e: e[0])
        vaihtoehdot = [(float(e[2][1]), float(e[1])) for e in sorted(ehdokkaat, key=lambda e: e[1])]
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
                varma=bool(np.isfinite(marg) and marg >= MARGINAALI_VARMA and np.sign(om_sov) == np.sign(om_ts) and abs(om) <= OMEGA_MAX_VARMA and len(kehykset) >= MIN_RUUTUJA_VARMA), muoto=dict(h_cm=muoto_k[0], ell_cm=muoto_k[1], w_cm=muoto_k[2]),
                f_keski=float(np.mean([d["f"] for d in kehykset])), vaihtoehdot=(vaihtoehdot if muoto is not None else None), ennakko=(None if muoto is None or enn is None else [float(enn[0]), float(enn[1])]))


# ---------------------------------------------------------------------------------------------------------------------
# YHDISTELMA: kierron mittaus heittopaasta (alku) ja kaukopaasta (loppu) -> kierrosmaara
# ---------------------------------------------------------------------------------------------------------------------
SIGMA_K = 0.008                 # vaimenemisvakion ennakkotieto (1/s): k = 0,021 +- 0,008 (IMU 2-3 % / s; koko videolla mitattu heittojen valinen hajonta 0,009 (n = 50, kun molemmat paat mitattu; mittauskohinan osuus 0,005 -> todellinen hajonta ~0,007))
OMEGA_SIGMA_MIN_DEG = 4.0       # kierron mittauksen pienin epavarmuus (deg/s)
K_SALLITTU = (0.0, 0.07)        # mitattu vaimeneminen uskottava, jos kahden mittauksen valinen k on tassa valissa


def omega_mitta(k, fps):
    """Kierron mittauksesta (arvioi) (t s, omega rad/s (+ = vastapaivaan), sigma rad/s, varma) tai None. Epavarmuus: sovituksen ja Theil-Senin ero, vahintaan OMEGA_SIGMA_MIN_DEG;
    ei-varma mittaus kaksinkertainen."""
    if k is None or "virhe" in k:
        return None
    om = float(k["omega_deg_s"])
    ero = abs(float(k.get("omega_sovitus_deg_s", om)) - float(k.get("omega_theilsen_deg_s", om)))
    sg = max(OMEGA_SIGMA_MIN_DEG, ero) * (1.0 if k.get("varma") else 2.0)
    return (float(k["f_keski"]) / fps, float(np.radians(om)), float(np.radians(sg)), bool(k.get("varma")))


def yhdista(mn, mf, t1, t2, k0=K_KIERRE):
    """Kierrosmaara [t1, t2] (hog-hog) kierron mittauksista: mn = heittopaa (t, omega, sigma, varma) ja mf = kaukopaa (sama muoto), kumpi tahansa voi puuttua. Malli:
    ln |omega(t)| = a - k (t - t_n) (eksponentiaalinen vaimeneminen); molemmat mittaukset ja ennakkotieto k = k0 +- SIGMA_K sovitetaan yhdessa (painotettu pienin neliösumma, lineaarinen
    a:ssa ja k:ssa), joten mitattu vaimeneminen korjaa kierrosmaaran (ei pelkkaa ekstrapolointia vakiolla). Palauttaa dict(N (+ = vastapaivaan), N_sigma, k, omega1, omega2, lahde,
    ristiriita) tai None. Etumerkit ristiriidassa -> varmemman mittauksen etumerkki, ristiriita = True."""
    mit = [m for m in (mn, mf) if m is not None]
    if not mit:
        return None
    ristiriita = False
    if len(mit) == 2 and np.sign(mn[1]) != np.sign(mf[1]):
        ristiriita = True
        mit = [min(mit, key=lambda m: (not m[3], m[2] / max(abs(m[1]), 1e-6)))]          # varma ensin, sitten pienin suhteellinen epavarmuus
    s = 1.0 if mit[0][1] >= 0 else -1.0
    t_ref = mit[0][0]
    # y = ln|omega|: yhtalot [1, -(t - t_ref)] [a, k]^T = y, paino 1 / (sigma / |omega|)
    rivit, y, w = [], [], []
    for t, om, sg, _ in mit:
        rivit.append([1.0, -(t - t_ref)])
        y.append(np.log(max(abs(om), 1e-3)))
        w.append(abs(om) / max(sg, 1e-6))
    rivit.append([0.0, 1.0]); y.append(k0); w.append(1.0 / SIGMA_K)           # k:n ennakkotieto
    Aa, yy, ww = np.array(rivit), np.array(y), np.array(w)
    W = np.diag(ww ** 2)
    cov = np.linalg.inv(Aa.T @ W @ Aa)
    a, k = cov @ (Aa.T @ W @ yy)
    lahde = "kaksi mittausta" if len(mit) == 2 else ("heittopaa" if mit[0] is mn else "kaukopaa")
    if len(mit) == 2 and not (K_SALLITTU[0] <= k <= K_SALLITTU[1]):
        ristiriita = True                                                          # mitattu vaimeneminen epauskottava: mittaukset eivat sovi yhteen

    def N_(a_, k_):
        k_ = max(k_, 1e-4)
        return s * float(np.exp(a_) / (2 * np.pi * k_) * (np.exp(-k_ * (t1 - t_ref)) - np.exp(-k_ * (t2 - t_ref))))

    N = N_(a, k)
    ja = (N_(a + 1e-3, k) - N_(a - 1e-3, k)) / 2e-3
    jk = (N_(a, k + 1e-4) - N_(a, k - 1e-4)) / 2e-4
    J = np.array([ja, jk])
    sg_N = float(np.sqrt(max(J @ cov @ J, 0.0)))
    return dict(N=float(N), N_sigma=sg_N, k=float(k), omega1=float(s * np.exp(a - k * (t1 - t_ref))), omega2=float(s * np.exp(a - k * (t2 - t_ref))), lahde=lahde,
                ristiriita=bool(ristiriita), t_ref=float(t_ref))
