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
# HEITTOPAAN KIERRE: robusti Bayes-estimaattori ruutukohtaisista kulmakustannuksista (arvioi_posteriori)
# ---------------------------------------------------------------------------------------------------------------------
# Miksi: palkin kulma on havaittavissa vain sitkeasti (kuvassa ~10 x 20 px:n sumea tahra): (1) kasi pitaa kahvasta kiinni vielä ~0,3-1,0 s irroituksen (ratadatasta paatellyn vapaan lennon alun)
# jalkeen, joten puhtaita ruutuja on vain ~0,5 s kuvan reunaan (Y ~ 750 cm) asti, kun irroitus on myohainen (keltaiset heittajat: Y ~ 540 cm) ja (2) kulma on havaittavissa vain
# modulo 180 astetta ja peilausepaselvyydella: palkki kulmassa theta nayttaa melkein samalta kuin kulmassa 2 a - theta (a ~ 90 astetta = palkki kameran katselusuunnassa; kallistusvihje on vain
# ~2-3 px), kun kasi peittaa. Ruutukohtainen paras kulma (argmin) loikkaa silloin peilikuvaan, ja suoran sovitus nayttaa "latauksena" (kulma kiertyy ensin vastakkaiseen suuntaan ja kaantyy
# V:n karjessa; todellisuudessa kierto on vakio koko ajan). Siksi ei kayteta ruutujen ennalta valittua parasta kulmaa eika kovaa kasirajausta, vaan jokaisen ruudun KOKO kustannuskayra
# (60 kulmaa), joka sisaltaa molemmat minimit; ruutujen yhteinen malli on vakiokierto theta(t) = theta0 + omega t, ja omegan posteriori lasketaan ruudukossa (theta0 x omega).
# Ruutujen painot kalibroitu videon omista pitkista heitoista (puhtaiden ruutujen suorat): ruudun kulmakohina sigma_i riippuu sovituksen jaannoksesta ja kasipikseleista palkin alueella.
# Ei omegan kovia rajoja (video: ~26 deg/s ... ~480 deg/s): ruudukko +-900 deg/s ja heikko lognormaali ennakko (mediaani 110, sigma_log 0,75: ~26 ja ~480 ovat +-2 sigmaa).
OM_RUUDUKKO = np.arange(-900.0, 900.1, 3.0)
TH_RUUDUKKO = np.arange(0.0, 180.0, 3.0)
KULMAT = np.arange(0.0, 180.0, 3.0)                     # ruutukohtaisen kustannuskayran kulmaruudukko (aste, jakso 180)
ZONE_KYNNYS = 0.03                                      # palkin jalanjalkialue: kaikkien kulmien mallien yhdiste > kynnys, laajennettu 2 px
SIG_HYVA = 4.5                                          # puhtaan ruudun kulmakohina (aste; MAD puhtaiden ruutujen suorista 4,3-5,1)
CZ_PISTEET = np.array([0.0, 6.0, 8.0, 12.0, 20.0, 40.0, 80.0])       # jalanjalkialueen paras kustannus Cz,min ...
CZ_SIGMA = np.array([4.4, 4.8, 6.5, 12.0, 16.0, 20.0, 60.0])         # ... -> kulmakohina (aste): puhtaat < 8: 4-5, 8-12: 10, > 12: 16+, > 40: umpimahkaan
KASI_PISTEET = np.array([0.0, 1.0, 6.0, 20.0, 60.0, 200.0])          # kasipikseleita jalanjalkialueella ...
KASI_SIGMA = np.array([4.4, 8.5, 12.6, 19.7, 25.6, 40.0])            # ... -> kulmakohina (aste)
HANTA_PAINO = 0.5                                       # ei-puhtaiden ruutujen (kasi alueella tai Cz,min >= 8) paino: sigma / sqrt(paino); ruudut ovat osin yhteisesti harhaisia (tasanne / peilaus)
PUHDAS_CZ = 8.0
S2_KAYRA = 0.3                                          # kustannuskayran lampotila: pen = min((C - C_min) / (2 s2 (sigma_i / SIG_HYVA)^2), KAYRA_KATKAISU); kalibroitu (z-hajonta ~ 1)
KAYRA_KATKAISU = 6.0
ENNAKKO_MEDIAANI = 110.0                                # |omega| ennakko (deg/s): lognormaali; video: 1 kierros / 14 s ... 20 kierrosta / 15 s (~26 ... ~480 deg/s) = +-2 sigmaa
ENNAKKO_SIGMA_LOG = 0.75
SUUNTA_VARMA = 0.99                                     # suunta "varma", kun valitun suunnan posteriori >= tama
MIN_RUUTUJA_UUSI = 6
Y_MAX_UUSI = 820.0                                      # kiven keskipiste: kuvan reuna ~ 750 cm; seuranta paattyy itse
MIN_JALANJALKI_PX = 100                                 # ruutuun vaaditaan vahintaan nain monta kelvollista pikselia palkin jalanjalkialueella


def _ikkuna_taytetty(crop, cx0, cy0, uc, vc, mode):
    """N x N -ikkuna pisteen (uc, vc) ymparilta talletetun leikkeen (vasen ylakulma (cx0, cy0)) sisalta; kuvan (leikkeen) ulkopuolinen osa taytetaan mustalla (kelvoton pikseli), taustakuva
    toistetaan reunoilta. Kuten _ikkuna, mutta ei hylkaa ruutua kiven ollessa kuvan reunalla (sovitusalue on kiven vasemmalla puolella)."""
    N, H = km.N, km.H
    hm, wm = mode.shape[:2]
    x0, y0 = int(round(uc)) - H, int(round(vc)) - H
    xs = np.clip(np.arange(x0, x0 + N), 0, wm - 1)
    ys = np.clip(np.arange(y0, y0 + N), 0, hm - 1)
    m = mode[np.ix_(ys, xs)]
    c = np.zeros((N, N, 3), np.uint8)
    ch, cw = crop.shape[:2]
    ix0, ix1 = max(x0, cx0), min(x0 + N, cx0 + cw)
    iy0, iy1 = max(y0, cy0), min(y0 + N, cy0 + ch)
    if ix1 - ix0 < 20 or iy1 - iy0 < 20:
        return None
    c[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0] = crop[iy0 - cy0:iy1 - cy0, ix0 - cx0:ix1 - cx0]
    return c, m, x0, y0


def kulmakayrat(cam, ast, vari, havainnot, t_irr, muoto, t0=0.0, y_max=Y_MAX_UUSI):
    """Ruutukohtaiset kulmakustannuskayrat. havainnot: [(t, f, Xb, Yb, crop, cx0, cy0)] (kuten rakenna_kehykset). Palauttaa dict(t, f, Y, Cz [n x 60], nkz, nvz) tai None.
    Jokaiselle ruudulle (t >= t_irr + t0): kahvan akseli (km.sovita_asetuksilla), muovikartta p ja kelvolliset pikselit (km.havainto), palkin jalanjalkialue (kaikkien kulmien mallien yhdiste),
    ja Cz[k] = sum (p - g m(theta_k))^2 jalanjalkialueen kelvollisissa pikseleissa (g = paras kirkkaus per kulma); nkz = kasipikselit, nvz = kelvolliset pikselit alueella."""
    N, W = km.N, ko.W
    z0 = ast["z0_cm"]
    kev = km.Kevyt(cam)
    mal = ko.Malli(cam, z0, ast["sigma_px"])
    h_, ell_, w_ = muoto
    t_l, f_l, Y_l, Cz_l, nkz_l, nvz_l = [], [], [], [], [], []
    for t, f, Xb, Yb, crop, cx0, cy0 in havainnot:
        if t < t_irr + t0 or Yb > y_max:
            continue
        ub, vb = kv.kivimalli.project_3d(cam.K, cam.R, cam.t, np.array([[Xb, Yb, z0]]))
        w = _ikkuna_taytetty(crop, cx0, cy0, ub[0], vb[0], cam.mode)
        if w is None:
            continue
        s = km.sovita_asetuksilla(kev, w[0], w[1], w[2], w[3], vari, (float(ub[0]), float(vb[0])), ast)
        if s is None:
            continue
        Xd, Yd = float(s["X"]), float(s["Y"])
        u, v = kv.kivimalli.project_3d(cam.K, cam.R, cam.t, np.array([[Xd, Yd, z0]]))
        w = _ikkuna_taytetty(crop, cx0, cy0, u[0], v[0], cam.mode)
        if w is None:
            continue
        c, mode, x0, y0 = w
        hh = km.havainto(c, mode, vari)
        if hh is None:
            continue
        p, w0 = hh[0], hh[1] > 0.5
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
        kelvoton = roi & ~w0[iv0:iv0 + 2 * W, iu0:iu0 + 2 * W]
        pp_kartta = p[iv0:iv0 + 2 * W, iu0:iu0 + 2 * W].astype(np.float32)
        d = dict(Xd=Xd, Yd=Yd, ox=x0 + iu0, oy=y0 + iv0)
        rend = [mal.piirra(d, np.radians(th), h_, ell_, w_) for th in KULMAT]
        zone = np.max(rend, axis=0) > ZONE_KYNNYS
        zone = (cv2.dilate(zone.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0) & roi
        zv = zone & ~kelvoton
        nvz = int(zv.sum())
        if nvz < 30:
            continue
        pz = pp_kartta[zv]
        cz = np.empty(len(KULMAT))
        for k, m in enumerate(rend):
            mz = m[zv]
            g = float(np.clip(mz @ pz / max(mz @ mz, 1e-9), 0.2, 2.5))
            cz[k] = float(np.sum((pz - g * mz) ** 2))
        t_l.append(float(t)); f_l.append(float(f)); Y_l.append(Yd); Cz_l.append(cz); nkz_l.append(int((zone & kelvoton).sum())); nvz_l.append(nvz)
    if not t_l:
        return None
    o = np.argsort(t_l)
    return dict(t=np.array(t_l)[o], f=np.array(f_l)[o], Y=np.array(Y_l)[o], Cz=np.array(Cz_l)[o], nkz=np.array(nkz_l)[o], nvz=np.array(nvz_l)[o])


def ruudun_sigma(cz_min, nkz):
    """Ruutukohtainen kulmakohinan arvio (aste): jalanjalkialueen paras kustannus ja kasipikselit alueella (empiirinen, videon pitkista heitoista); ei-puhtaat ruudut painotetaan HANTA_PAINO:lla."""
    sc = np.interp(cz_min, CZ_PISTEET, CZ_SIGMA)
    sn = np.interp(nkz, KASI_PISTEET, KASI_SIGMA)
    sg = np.sqrt(np.maximum(sc ** 2 + sn ** 2 - SIG_HYVA ** 2, 1.0))
    puhdas = (nkz == 0) & (cz_min < PUHDAS_CZ)
    return np.where(puhdas, sg, sg / np.sqrt(HANTA_PAINO)), puhdas


def _jaksollinen(C, ang):
    """C: (n, 60) kayrat 3 asteen ruudukossa (jakso 180); ang (..., n) kulmat (aste) -> lineaarisesti interpoloidut arvot."""
    x = (ang % 180.0) / 3.0
    i0 = np.floor(x).astype(int) % len(KULMAT)
    i1 = (i0 + 1) % len(KULMAT)
    w = x - np.floor(x)
    j = np.arange(C.shape[0])
    return C[j, i0] * (1 - w) + C[j, i1] * w


def ennakko_lognormaali(om):
    """Log-ennakko omegalle (asteikkoinvariantti lognormaali |omega|:lle, tasainen etumerkille); ei kovia rajoja."""
    a = np.maximum(np.abs(om), 1.0)
    return -0.5 * ((np.log(a) - np.log(ENNAKKO_MEDIAANI)) / ENNAKKO_SIGMA_LOG) ** 2 - np.log(a)


def posteriori_omega(k, s2=S2_KAYRA, katkaisu=KAYRA_KATKAISU, ennakko=ennakko_lognormaali, om_ruudukko=OM_RUUDUKKO, th_ruudukko=TH_RUUDUKKO):
    """Omegan posteriori vakiokiertomallille theta(t) = theta0 + omega (t - t_ka) ruutujen kulmakustannuskayrista k (kulmakayrat); theta0 marginalisoidaan (tasainen).
    Palauttaa dict(om, p, p_ccw, sig [ruutujen kohina], puhdas, t_ka, n_eff, J)."""
    cmin = np.min(k["Cz"], axis=1)
    sig, puhdas = ruudun_sigma(cmin, k["nkz"])
    dC = k["Cz"] - cmin[:, None]
    pen = np.minimum(dC / (2.0 * s2 * (sig / SIG_HYVA) ** 2)[:, None], katkaisu)
    w_t = 1.0 / sig ** 2
    t_ka = float(np.sum(w_t * k["t"]) / np.sum(w_t))                 # tietopaino-keskiaika (omegan vaimenemisen vertailuhetki)
    dt = k["t"] - t_ka
    J = np.empty((len(th_ruudukko), len(om_ruudukko)))
    lisa = om_ruudukko[:, None] * dt[None, :]                              # (omega, ruutu): kaukopaassa ruudukko voi olla tuhansia pisteita -> yksi theta0 kerrallaan
    for a in range(len(th_ruudukko)):
        J[a] = _jaksollinen(pen, th_ruudukko[a] + lisa).sum(axis=-1)
    lp = np.log(np.exp(-(J - J.min())).sum(axis=0) + 1e-300)
    if ennakko is not None:
        lp = lp + ennakko(om_ruudukko)
    lp -= lp.max()
    p = np.exp(lp)
    p /= p.sum()
    return dict(om=om_ruudukko, p=p, p_ccw=float(p[om_ruudukko > 0].sum()), sig=sig, puhdas=puhdas, t_ka=t_ka, n_eff=float(np.sum(1.0 / (sig / SIG_HYVA) ** 2)), J=J, th=th_ruudukko)


SARANA_T_ALKU = -0.6            # saranamalli: ruudut t >= t_irr + tama (s): plateau-ruudut ennen kierron alkua (kasi pitaa kahvaa, kulma vakio)
SARANA_TC_ALUE = (-0.6, 1.2)    # kierron alkuhetki t_c (s, suhteessa t_irr), tasainen ennakko; sormet irtoavat usein vasta 0,3-1,0 s irroituksen (reittidatan vapaan lennon alun) jalkeen
SARANA_TC_ASKEL = 0.05


def posteriori_omega_sarana(k, t_irr, s2=S2_KAYRA, katkaisu=KAYRA_KATKAISU, ennakko=ennakko_lognormaali, om_ruudukko=OM_RUUDUKKO, th_ruudukko=TH_RUUDUKKO):
    """Omegan posteriori saranamallille: kahvan kulma on vakio theta_p ennen kierron alkua t_c ja sen jalkeen theta0' + omega t (jatkuva: theta_p = theta0' + omega t_c); theta0' ja t_c marginalisoidaan
    (t_c tasainen valilla SARANA_TC_ALUE, myos 'ei saranaa' = t_c alussa). Syy: kierto alkaa, kun heittaja antaa kierteen kahvasta (sormet irtoavat), ei valttamatta reittidatan irroitushetkella; ennen sita
    kahvan kulma on vakio (tasanne), ja lineaarinen malli koko ruutujonon yli aliarvioisi omegan (punaiset heitot, joissa puhtaita ruutuja on vahan). Palauttaa kuten posteriori_omega."""
    cmin = np.min(k["Cz"], axis=1)
    sig, puhdas = ruudun_sigma(cmin, k["nkz"])
    dC = k["Cz"] - cmin[:, None]
    pen = np.minimum(dC / (2.0 * s2 * (sig / SIG_HYVA) ** 2)[:, None], katkaisu)
    t = k["t"] - t_irr
    n = len(t)
    jalkeen = t >= 0.0
    w_t = 1.0 / sig ** 2
    t_ka = float(np.sum((w_t * k["t"])[jalkeen]) / np.sum(w_t[jalkeen])) if jalkeen.any() else float(np.sum(w_t * k["t"]) / np.sum(w_t))
    tc = np.arange(SARANA_TC_ALUE[0], SARANA_TC_ALUE[1] + 1e-9, SARANA_TC_ASKEL)
    mc = np.searchsorted(t, tc)                                           # ruudut i >= mc[j] ovat saranan jalkeen
    pre = np.zeros((len(th_ruudukko), n + 1))                             # pre[a, m] = sum_{i < m} pen_i(theta_p = th_ruudukko[a]): plateau-kustannus
    for a, th in enumerate(th_ruudukko):
        pre[a, 1:] = np.cumsum(_jaksollinen(pen, np.full(n, th)))
    om = om_ruudukko
    acc = np.full(len(om), -np.inf)
    nth = len(th_ruudukko)
    for th0 in th_ruudukko:                                               # theta0' (aika-akselin nollassa t = t_irr)
        v = _jaksollinen(pen, th0 + om[:, None] * t[None, :])             # (omega, ruutu)
        suf = np.concatenate([np.cumsum(v[:, ::-1], axis=1)[:, ::-1], np.zeros((len(om), 1))], axis=1)
        for j in range(len(tc)):
            thp = (th0 + om * tc[j]) % 180.0
            x = thp / 3.0
            i0 = np.floor(x).astype(int) % nth
            i1 = (i0 + 1) % nth
            w = x - np.floor(x)
            Jv = suf[:, mc[j]] + pre[i0, mc[j]] * (1 - w) + pre[i1, mc[j]] * w
            acc = np.logaddexp(acc, -Jv)
    lp = acc
    if ennakko is not None:
        lp = lp + ennakko(om)
    lp -= lp.max()
    p = np.exp(lp)
    p /= p.sum()
    return dict(om=om, p=p, p_ccw=float(p[om > 0].sum()), sig=sig, puhdas=puhdas, t_ka=t_ka, n_eff=float(np.sum(1.0 / (sig / SIG_HYVA) ** 2)), J=None, th=th_ruudukko, jalkeen=jalkeen)


def tiivista_posteriori(P):
    """Posteriorin yhteenveto: (suunta +-1 [+ = vastapaivaan], omegan keskiarvo valitulla suunnalla, hajonta, moodi, suunnan todennakoisyys)."""
    om, p = P["om"], P["p"]
    s = 1 if P["p_ccw"] >= 0.5 else -1
    m = (om * s) > 0
    pp = p[m] / max(p[m].sum(), 1e-300)
    mu = float((pp * om[m]).sum())
    sd = float(np.sqrt((pp * (om[m] - mu) ** 2).sum()))
    return s, mu, sd, float(om[m][int(np.argmax(pp))]), float(max(P["p_ccw"], 1.0 - P["p_ccw"]))


def arvioi_posteriori(cam, ast, vari, havainnot, t_irr, muoto=None):
    """Heittopaan kierre: omega (+ = vastapaivaan) ja sen epavarmuus posteriorista; kaikki ruudut irroituksesta kuvan reunaan (ei kovaa kasirajausta, laatupainotus), vakiokiertomalli,
    peilausepaselvyys mukana kustannuskayrissa. Palauttaa dict(omega_deg_s [posteriorin keskiarvo valitulla suunnalla], omega_sd_deg_s, omega_ml_deg_s, p_suunta, suunta, p_ccw, t_keski,
    f_keski, ruutuja, ruutuja_puhdas, n_eff, varma [suunta >= SUUNTA_VARMA], ...) tai dict(virhe=...). Kierron suuruudelle ei aseteta rajoja."""
    muoto = tuple(muoto) if muoto is not None else (ko.H_ALKU, ko.ELL_ALKU, ko.W_ALKU)
    k = kulmakayrat(cam, ast, vari, havainnot, t_irr, muoto, t0=SARANA_T_ALKU)
    if k is None or int(np.sum(k["t"] >= t_irr)) < MIN_RUUTUJA_UUSI:
        n = 0 if k is None else int(np.sum(k["t"] >= t_irr))
        return dict(virhe=f"liian vahan kelvollisia ruutuja ({n})", ruutuja=n)
    P = posteriori_omega_sarana(k, t_irr)
    s, mu, sd, om_ml, p_suunta = tiivista_posteriori(P)
    th0 = float("nan")
    w_t = 1.0 / P["sig"] ** 2
    return dict(omega_deg_s=mu, omega_sd_deg_s=sd, omega_ml_deg_s=om_ml, omega_sovitus_deg_s=om_ml, omega_theilsen_deg_s=mu, theta0_deg=th0, p_suunta=p_suunta, suunta=int(s), p_ccw=P["p_ccw"],
                t_keski=P["t_ka"], f_keski=float(np.sum((w_t * k["f"])[P["jalkeen"]]) / np.sum(w_t[P["jalkeen"]])), ruutuja=int(P["jalkeen"].sum()), ruutuja_puhdas=int(P["puhdas"].sum()),
                hylatty_kasi=int((k["nkz"] > 60).sum()), ruutuja_ennen=int((~P["jalkeen"]).sum()), n_eff=P["n_eff"], varma=bool(p_suunta >= SUUNTA_VARMA), marginaali=float("nan"),
                muoto=dict(h_cm=muoto[0], ell_cm=muoto[1], w_cm=muoto[2]), menetelma="posteriori_sarana")


KAUKO_ENNAKKO_SLACK = 0.20      # kaukopaan omegan ennakon (heittopaan mittaus + vaimeneminen) lisaepavarmuus (ln-asteikolla), ettei ennakko mene todistusaineiston edelle
KAUKO_ENNAKKO_MIN_SIGMA = 0.30  # ... ja ennakon ln-hajonnan alaraja
KAUKO_STEP_KERROIN = 5.0        # omega-ruudukon askel (deg/s) = kerroin / ikkunan pituus (s): vaihe ei karkaa yli ~2,5 astetta ikkunan yli
KAUKO_MAX_PISTEITA = 6000       # omega-ruudukon enimmaiskoko


def _kauko_ruudukko(T, enn):
    """Kaukopaan omega-ruudukko (deg/s): askel kerroin / T (pitkalla ikkunalla tihea, koska vaiheen pitaa pysya oikeana ruutujen yli), alue: ennakon mukaan [exp(mu - 4 s), exp(mu + 4 s)] samalle
    etumerkille (spin ei kaanny eika voimistu; ennakko on heikko) tai ilman ennakkoa +-900 deg/s."""
    step = float(np.clip(KAUKO_STEP_KERROIN / max(T, 0.5), 0.15, 3.0))
    if enn is None:
        top = np.arange(-900.0, 900.0 + 1e-9, step)
        return top
    mu, sg, s = enn
    lo, hi = np.exp(mu - 4.0 * sg), np.exp(mu + 4.0 * sg)
    om = np.arange(max(lo, 1.0), hi + step, step)
    if len(om) > KAUKO_MAX_PISTEITA:
        om = np.linspace(max(lo, 1.0), hi, KAUKO_MAX_PISTEITA)
    return s * om


def arvioi_posteriori_kauko(cam, ast, vari, havainnot, t0, muoto=None, omega_ennakko=None):
    """Kaukopaan kierre kuten arvioi_posteriori (kaikki ruudut, laatupainotus, ei omegan rajoja), mutta ruudukko tihennetaan ikkunan pituuden mukaan ja heittopaan mittaus antaa heikon
    ennakon (alias-minimien valinta): omega_ennakko = (omega_deg_s, t_s[, sd_deg_s]) heittopaan varma mittaus; ennakko: ln |omega_kauko| ~ N(ln |omega_lahi| - K_KIERRE (t_kauko - t_lahi),
    sigma^2), sigma^2 = (sd / omega)^2 + (SIGMA_K dt)^2 + KAUKO_ENNAKKO_SLACK^2 (vahintaan KAUKO_ENNAKKO_MIN_SIGMA), etumerkki sama kuin heittopaassa. Palauttaa kuten arvioi_posteriori."""
    muoto = tuple(muoto) if muoto is not None else (ko.H_ALKU, ko.ELL_ALKU, ko.W_ALKU)
    k = kulmakayrat(cam, ast, vari, havainnot, t0, muoto)
    if k is None or len(k["t"]) < MIN_RUUTUJA_UUSI:
        n = 0 if k is None else len(k["t"])
        return dict(virhe=f"liian vahan kelvollisia ruutuja kaukopaassa ({n})", ruutuja=n)
    cmin = np.min(k["Cz"], axis=1)
    sig, puhdas = ruudun_sigma(cmin, k["nkz"])
    w_t = 1.0 / sig ** 2
    t_ka = float(np.sum(w_t * k["t"]) / np.sum(w_t))
    T = float(k["t"].max() - k["t"].min())
    enn = None
    if omega_ennakko is not None:
        om_n, t_n = float(omega_ennakko[0]), float(omega_ennakko[1])
        sd_n = float(omega_ennakko[2]) if len(omega_ennakko) > 2 else 0.15 * abs(om_n)
        dt = t_ka - t_n
        mu = np.log(max(abs(om_n), 1.0)) - K_KIERRE * dt
        sg = max(KAUKO_ENNAKKO_MIN_SIGMA, float(np.sqrt((sd_n / max(abs(om_n), 1.0)) ** 2 + (SIGMA_K * abs(dt)) ** 2 + KAUKO_ENNAKKO_SLACK ** 2)))
        enn = (mu, sg, 1.0 if om_n > 0 else -1.0)
    om_r = _kauko_ruudukko(T, enn)
    if enn is None:
        ennakko_f = ennakko_lognormaali
    else:
        mu, sg, _ = enn
        ennakko_f = lambda om: -0.5 * ((np.log(np.maximum(np.abs(om), 1.0)) - mu) / sg) ** 2 - np.log(np.maximum(np.abs(om), 1.0))
    P = posteriori_omega(k, ennakko=ennakko_f, om_ruudukko=om_r)
    s, mu_om, sd, om_ml, p_suunta = tiivista_posteriori(P)
    ia = int(np.argmin(np.abs(P["om"] - om_ml)))
    th0 = float(P["th"][int(np.argmin(P["J"][:, ia]))])
    # posteriorin keskittyneisyys: paamoodin ymparilla (+-3 sd tai +-3 deg/s) oleva osuus massasta (alias-lohkot hajottavat)
    m = (P["om"] * s) > 0
    pp = P["p"][m] / max(P["p"][m].sum(), 1e-300)
    oo = P["om"][m]
    kesk = float(pp[np.abs(oo - om_ml) <= max(3.0 * sd, 3.0)].sum()) if len(oo) else 0.0
    return dict(omega_deg_s=mu_om, omega_sd_deg_s=sd, omega_ml_deg_s=om_ml, omega_sovitus_deg_s=om_ml, omega_theilsen_deg_s=mu_om, theta0_deg=th0, p_suunta=p_suunta, suunta=int(s), p_ccw=P["p_ccw"],
                t_keski=P["t_ka"], f_keski=float(np.sum(w_t * k["f"]) / np.sum(w_t)), ruutuja=int(len(k["t"])), ruutuja_puhdas=int(P["puhdas"].sum()), hylatty_kasi=int((k["nkz"] > 60).sum()),
                n_eff=P["n_eff"], keskittyneisyys=kesk, ikkuna_s=T, varma=bool(p_suunta >= SUUNTA_VARMA and sd / max(abs(mu_om), 1.0) <= KAUKO_VARMA_SD and kesk >= 0.9), marginaali=float("nan"),
                muoto=dict(h_cm=muoto[0], ell_cm=muoto[1], w_cm=muoto[2]), menetelma="posteriori_kauko")


KAUKO_VARMA_SD = 0.15           # kaukopaan mittaus "varma", kun omegan suhteellinen hajonta <= tama (ja suunta varma, posteriori keskittynyt yhteen lohkoon)


# ---------------------------------------------------------------------------------------------------------------------
# YHDISTELMA: kierron mittaus heittopaasta (alku) ja kaukopaasta (loppu) -> kierrosmaara
# ---------------------------------------------------------------------------------------------------------------------
SIGMA_K = 0.008               # vaimenemisvakion ennakkotieto (1/s): k = 0,021 +- 0,008 (IMU 2-3 % / s; koko videolla mitattu heittojen valinen hajonta 0,009 (n = 50, kun molemmat paat mitattu; mittauskohinan osuus 0,005 -> todellinen hajonta ~0,007))
OMEGA_SIGMA_MIN_DEG = 4.0       # kierron mittauksen pienin epavarmuus (deg/s)
CHI2_RAJA = 9.0                 # kahden mittauksen yhteissovituksen khi^2 (1 vapausaste; 3 sigmaa): ylitys = mittaukset ja vaimenemisen ennakkotieto (k = 0,021 +- SIGMA_K) eivat sovi yhteen


def omega_mitta(k, fps):
    """Kierron mittauksesta (arvioi / arvioi_posteriori) (t s, omega rad/s (+ = vastapaivaan), sigma rad/s, varma) tai None. Epavarmuus: posteriorin hajonta (arvioi_posteriori) tai
    sovituksen ja Theil-Senin ero (arvioi; ei-varma mittaus kaksinkertainen), vahintaan OMEGA_SIGMA_MIN_DEG."""
    if k is None or "virhe" in k:
        return None
    om = float(k["omega_deg_s"])
    if "omega_sd_deg_s" in k:                      # posteriori (arvioi_posteriori): epavarmuus suoraan posteriorin hajonnasta (ei kaksinkertaistusta)
        sg = max(OMEGA_SIGMA_MIN_DEG, float(k["omega_sd_deg_s"]))
    else:
        ero = abs(float(k.get("omega_sovitus_deg_s", om)) - float(k.get("omega_theilsen_deg_s", om)))
        sg = max(OMEGA_SIGMA_MIN_DEG, ero) * (1.0 if k.get("varma") else 2.0)
    return (float(k["f_keski"]) / fps, float(np.radians(om)), float(np.radians(sg)), bool(k.get("varma")))


def yhdista(mn, mf, t1, t2, k0=K_KIERRE):
    """Kierrosmaara [t1, t2] (hog-hog) kierron mittauksista: mn = heittopaa (t, omega, sigma, varma) ja mf = kaukopaa (sama muoto), kumpi tahansa voi puuttua. Malli:
    ln |omega(t)| = a - k (t - t_n) (eksponentiaalinen vaimeneminen); molemmat mittaukset ja ennakkotieto k = k0 +- SIGMA_K sovitetaan yhdessa (painotettu pienin neliösumma, lineaarinen
    a:ssa ja k:ssa), joten mitattu vaimeneminen korjaa kierrosmaaran (ei pelkkaa ekstrapolointia vakiolla). Palauttaa dict(N (+ = vastapaivaan), N_sigma, k, omega1, omega2, lahde,
    ristiriita, chi2) tai None. Etumerkit ristiriidassa -> varmemman mittauksen etumerkki, ristiriita = True. Kun yhteissovituksen khi^2 > CHI2_RAJA (esim. omega kasvaa tai vaimenee
    moninkertaisella nopeudella), tulos on molempien yksittaisten arvioiden keskiarvo ja epavarmuus puolet niiden erosta (lahde 'ristiriita')."""
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
    chi2 = None
    if len(mit) == 2:
        res = (Aa @ np.array([a, k]) - yy) * ww
        chi2 = float(res @ res)
        if chi2 > CHI2_RAJA:                                                       # spin ei voimistu eika vaimene moninkertaisella nopeudella: toinen mittauksista on todennakoisesti pielessa
            ra, rb = yhdista(mn, None, t1, t2, k0), yhdista(None, mf, t1, t2, k0)
            N_ka = 0.5 * (ra["N"] + rb["N"])
            hajonta = 0.5 * abs(ra["N"] - rb["N"])                                 # ei tiedeta kumpi on oikein: keskiarvo ja puolet erosta epavarmuutena
            return dict(N=float(N_ka), N_sigma=float(np.sqrt(hajonta ** 2 + 0.25 * (ra["N_sigma"] ** 2 + rb["N_sigma"] ** 2))), k=float(k0), omega1=0.5 * (ra["omega1"] + rb["omega1"]),
                        omega2=0.5 * (ra["omega2"] + rb["omega2"]), lahde="ristiriita", ristiriita=True, t_ref=float(t_ref), chi2=chi2)

    def N_(a_, k_):
        k_ = max(k_, 1e-4)
        return s * float(np.exp(a_) / (2 * np.pi * k_) * (np.exp(-k_ * (t1 - t_ref)) - np.exp(-k_ * (t2 - t_ref))))

    N = N_(a, k)
    ja = (N_(a + 1e-3, k) - N_(a - 1e-3, k)) / 2e-3
    jk = (N_(a, k + 1e-4) - N_(a, k - 1e-4)) / 2e-4
    J = np.array([ja, jk])
    sg_N = float(np.sqrt(max(J @ cov @ J, 0.0)))
    return dict(N=float(N), N_sigma=sg_N, k=float(k), omega1=float(s * np.exp(a - k * (t1 - t_ref))), omega2=float(s * np.exp(a - k * (t2 - t_ref))), lahde=lahde,
                ristiriita=bool(ristiriita), t_ref=float(t_ref), chi2=chi2)
