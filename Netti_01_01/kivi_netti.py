"""Kiven tunnistus ja mittaus nettivideolla (yhden puoliskon nakyma, kalibrointi: kalibrointi_netti.py).

Ruutu: puolisko rajataan ja kaannetaan kuten kalibroinnissa (kalibrointi_netti.PUOLET, 90 astetta vastapaivaan), linssi-
vaaristyma (k1) korjataan ja kuvaa verrataan moodikuvaan (tausta). Kiven graniittiosa on tumma ja vahvasti epakylläinen
(V ~ 110-160, S < 30) ja kahva vaalea/värillinen; ihmiset ovat mustia (V < 70) tai valkoisia, joten graniittimaski
V-valilla [G_VMIN, G_VMAX] erottaa kiven. Paikka ja mitat sovitetaan kiven 3D-mallin siluettiin kameran asennolla (K, R, t).
"""
import json

import cv2
import numpy as np

import kalibrointi_netti as kn
import kivimalli

# graniittimaski (taustasta poikkeava, tumma, epakylläinen)
FG_KYNNYS = 28            # |ruutu - moodikuva| (suurin kanavaero) > tama
G_VMIN, G_VMAX, G_SMAX = 75, 175, 70
HANDLE_SMIN, HANDLE_VMIN = 45, 150
LANE_X_CM = (-130.0, 130.0)
LANE_Y_CM = (-430.0, 900.0)


class Kamera:
    """Puolison kalibrointi + kuvan esikasittely (rajaus, kierto, linssikorjaus) + taustakuva."""

    def __init__(self, puoli, kalib_json, moodikuva_png):
        self.puoli = puoli
        d = json.load(open(kalib_json))
        self.malli = kn.Malli(d["H_kuva_fyysinen"], d["k1"], d["kuvan_keskipiste"])
        self.K = np.array(d["K"], dtype=np.float64)
        self.R = np.array(d["R"], dtype=np.float64)
        self.t = np.array(d["t"], dtype=np.float64)
        self.pose = {"K": self.K, "R": self.R, "t": self.t}
        mode_raw = cv2.imread(moodikuva_png)
        self.h, self.w = mode_raw.shape[:2]
        # linssikorjauksen kartat: oikaistun kuvan pikseli -> alkuperaisen kuvan pikseli
        gu, gv = np.meshgrid(np.arange(self.w, dtype=np.float64), np.arange(self.h, dtype=np.float64))
        src = kn._distort_px(np.column_stack([gu.ravel(), gv.ravel()]), self.malli.k1, self.malli.centre)
        self.map1 = src[:, 0].reshape(self.h, self.w).astype(np.float32)
        self.map2 = src[:, 1].reshape(self.h, self.w).astype(np.float32)
        self.mode = cv2.remap(mode_raw, self.map1, self.map2, cv2.INTER_LINEAR)
        self.mode_hsv = cv2.cvtColor(self.mode, cv2.COLOR_BGR2HSV)
        self.lane = self._lane_mask()

    def _lane_mask(self):
        pts = np.array([[x, y, z] for x in LANE_X_CM for y in LANE_Y_CM for z in (0.0, 22.0)])
        u, v = kivimalli.project_3d(self.K, self.R, self.t, pts)
        hull = cv2.convexHull(np.column_stack([u, v]).astype(np.float32))
        m = np.zeros((self.h, self.w), np.uint8)
        cv2.fillConvexPoly(m, hull.astype(np.int32), 255)
        # minikartan alareuna (alkuperaisessa kuvassa x < 275, y < 272): rajauksessa y0 -> kierretyssa kuvassa
        # sarakkeet < 272 - y0 ja rivit >= (leveys - 1 - 275)
        x0, y0, cw, ch = kn.PUOLET[self.puoli]
        if y0 < 272:
            m[cw - 1 - 285:, 0:int(272 - y0) + 4] = 0
        return m

    def muunna(self, frame_full):
        x, y, w, h = kn.PUOLET[self.puoli]
        c = cv2.rotate(frame_full[y:y + h, x:x + w], cv2.ROTATE_90_COUNTERCLOCKWISE)
        return cv2.remap(c, self.map1, self.map2, cv2.INTER_LINEAR)

    def maskit(self, frame):
        """(fg, graniitti, kahva) binaarimaskit (uint8 0/255) oikaistusta kuvasta."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        diff = np.max(np.abs(cv2.GaussianBlur(frame, (3, 3), 0).astype(np.int16)
                             - cv2.GaussianBlur(self.mode, (3, 3), 0).astype(np.int16)), axis=2)
        fg = (diff > FG_KYNNYS) & (self.lane > 0)
        gran = fg & (hsv[..., 2] >= G_VMIN) & (hsv[..., 2] <= G_VMAX) & (hsv[..., 1] <= G_SMAX)
        handle = fg & (hsv[..., 1] >= HANDLE_SMIN) & (hsv[..., 2] >= HANDLE_VMIN) & (hsv[..., 1] > self.mode_hsv[..., 1] + 25)
        k3 = np.ones((3, 3), np.uint8)
        gran = cv2.morphologyEx(gran.astype(np.uint8) * 255, cv2.MORPH_OPEN, k3)
        gran = cv2.morphologyEx(gran, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        handle = cv2.morphologyEx(handle.astype(np.uint8) * 255, cv2.MORPH_OPEN, k3)
        return fg.astype(np.uint8) * 255, gran, handle

    def ruutu(self, video_cap, i):
        video_cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = video_cap.read()
        return self.muunna(fr) if ok else None


def komponentit(mask, min_area=150):
    n, lab, st, cen = cv2.connectedComponentsWithStats(mask)
    out = []
    for k in range(1, n):
        if st[k, cv2.CC_STAT_AREA] >= min_area:
            out.append(dict(area=int(st[k, cv2.CC_STAT_AREA]), bbox=tuple(int(v) for v in st[k, :4]), c=cen[k], label=k))
    return out, lab


# =====================================================================================================================
# KIVEN SILUETTIMALLI JA SOVITUS
# =====================================================================================================================
SHIFT = 4        # alipikselipiirto (cv2.fillConvexPoly, 1/16 px)


# KARKEA siluettimalli (vain tunnistuksen siemenet ja alkuarvaukset; tarkka kiintea runkomalli on kivi_tarkka.py ja sen opetus kalibroi_kivi.py).
# HUOM: aiemmat arvot R 15,7 / H 15,7 mitattiin virheellisella kameran asennolla (f = 2293 px, ei toistanut maatason homografiaa); korjatulla
# asennolla (f = 3129 px, kamera radan keskella) kiven sivusuuntainen puolileveys on 13,7-14,1 cm VAKIO koko radalla (Y = -270...750 cm).
MITATTU_R_CM = 13.7
MITATTU_H_CM = 16.0


class Kivi:
    """Kiven 3D-malli (graniittirunko, kivimalli.build_local_stone_rings) ja sen siluetti kuvassa."""

    def __init__(self, R_max=MITATTU_R_CM, H_total=MITATTU_H_CM):
        self.set_mitat(R_max, H_total)

    def set_mitat(self, R_max, H_total):
        self.R_max, self.H_total = float(R_max), float(H_total)
        self.pts = kivimalli.build_local_stone_rings(self.R_max, self.H_total, np.zeros(len(kivimalli.TEMPLATE_R_FRAC)),
                                                     n_theta=36, n_per_segment=5)

    def hull(self, cam, X, Y):
        u, v = kivimalli.project_3d(cam.K, cam.R, cam.t, self.pts + np.array([X, Y, 0.0]))
        return cv2.convexHull(np.column_stack([u, v]).astype(np.float32))[:, 0, :]

    def maski(self, cam, X, Y, ikkuna):
        """Siluetti ikkunassa ikkuna = (x0, y0, x1, y1): uint8-maski (0/255)."""
        x0, y0, x1, y1 = ikkuna
        h = self.hull(cam, X, Y) - np.array([x0, y0], dtype=np.float32)
        m = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillConvexPoly(m, np.round(h * (1 << SHIFT)).astype(np.int32), 255, cv2.LINE_8, SHIFT)
        return m


def maan_paikka(cam, u, v, z=6.0):
    X, Y = kivimalli.ray_plane_intersection(cam.K, cam.R, cam.t, u, v, z)
    return float(X[0]), float(Y[0])


def ikkuna_ymparille(cam, kivi, X, Y, marg=14):
    h = kivi.hull(cam, X, Y)
    x0, y0 = np.floor(h.min(0)).astype(int) - marg
    x1, y1 = np.ceil(h.max(0)).astype(int) + marg
    return max(0, x0), max(0, y0), min(cam.w, x1), min(cam.h, y1)


def iou_virhe(cam, kivi, havainto, X, Y, ikkuna):
    m = kivi.maski(cam, X, Y, ikkuna)
    o = havainto[ikkuna[1]:ikkuna[3], ikkuna[0]:ikkuna[2]]
    inter = np.count_nonzero((m > 0) & (o > 0))
    union = np.count_nonzero((m > 0) | (o > 0))
    return 1.0 - inter / max(union, 1)


def sovita_paikka(cam, kivi, havainto, X0, Y0, askel0=3.0, kierrokset=14):
    """(X, Y) kuviohaulla (koordinaattihaku, askel puolittuu): minimoi 1 - IoU siluetin ja havaintomaskin valilla."""
    ikk = ikkuna_ymparille(cam, kivi, X0, Y0, marg=40)
    X, Y, a = X0, Y0, askel0
    best = iou_virhe(cam, kivi, havainto, X, Y, ikk)
    for _ in range(kierrokset):
        improved = False
        for dx, dy in ((a, 0), (-a, 0), (0, a), (0, -a), (a, a), (-a, -a), (a, -a), (-a, a)):
            e = iou_virhe(cam, kivi, havainto, X + dx, Y + dy, ikk)
            if e < best - 1e-6:
                best, X, Y, improved = e, X + dx, Y + dy, True
        if not improved:
            a *= 0.5
            if a < 0.05:
                break
    return X, Y, best


def kandidaatit(cam, kivi, gran, handle, min_ratio=0.45, max_ratio=1.6, min_solid=0.75):
    """Kivikandidaatit graniittimaskin komponenteista: pinta-ala ja tayteisyys sopivat kiven malliin sijainnissa."""
    comps, lab = komponentit(gran, min_area=120)
    out = []
    for c in comps:
        u, v = c["c"]
        X, Y = maan_paikka(cam, u, v, z=0.5 * kivi.H_total)
        if not (LANE_X_CM[0] <= X <= LANE_X_CM[1] and LANE_Y_CM[0] <= Y <= LANE_Y_CM[1]):
            continue
        hull = kivi.hull(cam, X, Y)
        exp_area = cv2.contourArea(hull.astype(np.float32))
        ratio = c["area"] / max(exp_area, 1.0)
        mask = (lab == c["label"]).astype(np.uint8)
        cnt = max(cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea)
        solid = cv2.contourArea(cnt) / max(cv2.contourArea(cv2.convexHull(cnt)), 1.0)
        c.update(X=X, Y=Y, exp_area=exp_area, ratio=ratio, solid=solid)
        if min_ratio <= ratio <= max_ratio and solid >= min_solid:
            out.append(c)
    return out


def tunnista(cam, kivi, frame, max_err=0.5, min_peitto=0.55, yhdista_cm=15.0):
    """Kivet ruudusta: siemenet graniittikomponenteista ja kahvakomponenteista (kahva on kiven ylapinnalla -> maa-asema
    z = H + 3 cm), jokainen siemen sovitetaan siluettimallilla (1 - IoU) graniitti+kahva-maskiin. Palauttaa listan
    dict(X, Y, err, peitto, hull) parhaasta alkaen (paallekkaiset yhdistetty)."""
    fg, gran, handle = cam.maskit(frame)
    obs = cv2.bitwise_or(gran, handle)
    seeds = []
    for c in komponentit(gran, min_area=100)[0]:
        X, Y = maan_paikka(cam, c["c"][0], c["c"][1], z=0.5 * kivi.H_total)
        if LANE_Y_CM[0] <= Y <= LANE_Y_CM[1] and 0.3 <= c["area"] / max(cv2.contourArea(kivi.hull(cam, X, Y)), 1.0) <= 2.2:
            seeds.append((X, Y))
    for c in komponentit(handle, min_area=25)[0]:
        if c["area"] <= 900:
            seeds.append(maan_paikka(cam, c["c"][0], c["c"][1], z=kivi.H_total + 3.0))
    found = []
    for X0, Y0 in seeds:
        if not (LANE_X_CM[0] <= X0 <= LANE_X_CM[1] and LANE_Y_CM[0] <= Y0 <= LANE_Y_CM[1]):
            continue
        if any(np.hypot(X0 - f["X"], Y0 - f["Y"]) < 6.0 for f in found):
            continue
        X, Y, err = sovita_paikka(cam, kivi, obs, X0, Y0)
        ikk = ikkuna_ymparille(cam, kivi, X, Y, marg=2)
        m = kivi.maski(cam, X, Y, ikk) > 0
        peitto = float(np.count_nonzero(m & (obs[ikk[1]:ikk[3], ikk[0]:ikk[2]] > 0))) / max(np.count_nonzero(m), 1)
        found.append(dict(X=X, Y=Y, err=err, peitto=peitto))
    found = [f for f in found if f["err"] <= max_err and f["peitto"] >= min_peitto]
    found.sort(key=lambda f: f["err"])
    out = []
    for f in found:
        if all(np.hypot(f["X"] - g["X"], f["Y"] - g["Y"]) > yhdista_cm for g in out):
            f["hull"] = kivi.hull(cam, f["X"], f["Y"])
            out.append(f)
    return out, (fg, gran, handle)


def piirra(frame, dets, gran=None, handle=None):
    vis = frame.copy()
    if gran is not None:
        vis[gran > 0] = (0.5 * vis[gran > 0] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
        vis[handle > 0] = (0.5 * vis[handle > 0] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
    for d in dets:
        cv2.polylines(vis, [np.round(d["hull"]).astype(np.int32)], True, (255, 0, 0), 2, cv2.LINE_AA)
        u = int(d["hull"][:, 0].max()) + 6
        v = int(d["hull"][:, 1].min())
        cv2.putText(vis, f"{d['X']:.0f},{d['Y']:.0f} e{d['err']:.2f}", (min(u, vis.shape[1] - 150), v), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    return vis


# =====================================================================================================================
# PEHMEA (ALIPIKSELI) SOVITUS: alfa-havainto + ylinaytteistetty siluetti, kahva = "ei valia"-alue
# =====================================================================================================================
G_TASO = 135.0           # graniitin tyypillinen V (alfa = 1)
JAA_V = 235.0            # jaan V, jos moodikuvan pikseli ei ole jaata
SS = 4                   # siluetin ylinaytteistus
NEUTRAALI_S = 50         # graniitti on vareetonta: S alle taman (varjo punaisella/sinisella renkaalla ei ole)
VARJO_ALARAJA = 0.3      # alfa < tama = jaan varjo -> 0 (0,5-taso sailyy: alfa' = (a - 0,3) / 0,4)

# Ylapinta ja kahva: kiven ylapinta (ympyra z = H, sade TOP_R_FRAC * R) on valkoisen kiilteen takia vaalea eika nay graniittina,
# ja kahva (pystysylinteri akselilla, sade HANDLE_R_FRAC * R, korkeus HANDLE_H_CM ylapinnan ylapuolelle) peittaa ylapintaa.
# Niiden yhteinen siluetti (+ heittajan kasi) on "ei valia"-aluetta: siella siluetti ei saa eika menetta. Graniittina nakyy
# vain sivuseina + reuna (puolikuu).
TOP_R_FRAC = 0.78
HANDLE_R_FRAC = 0.5
HANDLE_H_CM = 8.0


GRANIITTI_BGR = np.array([130.0, 126.0, 136.0])      # graniitin vari videolla 1 (kuvan mediaani); vara, kun kameran jaan varia ei tunneta
GRANIITTI_SUHDE = np.array([0.543, 0.542, 0.563])    # graniitin pinnan vari / jaan vari (kanavittain), mitattu kelvollisista B-sovituksista: video 1 0,53-0,56, video 2 0,57-0,59, video 3 0,52-0,54


def jaan_vari(cam):
    """Kameran jaan vari (BGR): moodikuvan mediaani kaistan pikseleilla (jaa on enemmistona, renkaat ja viivat eivat vaikuta)."""
    v = getattr(cam, "_jaan_vari", None)
    if v is None:
        m = cam.mode.reshape(-1, 3)[cam.lane.reshape(-1) > 0]
        v = np.median(m.astype(np.float32), axis=0) if len(m) else None
        try:
            cam._jaan_vari = v
        except AttributeError:
            pass
    return v


def graniitti_vari(cam):
    """Graniitin vari (BGR) kameran kuvassa = jaan vari x GRANIITTI_SUHDE. Vakio GRANIITTI_BGR on videon 1 valotuksen mukainen: videon 3 kuva on ~10 % tummempi (jaa 213 vs 236)
    ja graniitti 111 vs 126, jolloin vakiovarilla alfa ja poikkeama (dev) menivat vaarin ja kivi todettiin kelvottomaksi varillisella jaalla."""
    g = getattr(cam, "graniitti", None)
    if g is None:
        j = jaan_vari(cam)
        g = GRANIITTI_BGR.copy() if j is None else (j * GRANIITTI_SUHDE).astype(np.float64)
        try:
            cam.graniitti = g
        except AttributeError:
            pass
    return g


def alfa_kartta(cam, frame, gran, handle=None):
    """Graniitin peittavyys 0..1 varien sekoitusmallista: pikseli = a * graniitti + (1 - a) * tausta (tausta = moodikuva
    pikselille), a = (p - b) . (g - b) / |g - b|^2. Toimii samoin jaalla ja sinisella/punaisella renkaalla: renkaan pikseli
    ilman kivea antaa a = 0, ja jaan/renkaan varjo (tumma tausta, sama savy) antaa vain pienen a:n, koska (p - b) on b:n
    suuntainen eika graniitin suuntainen. Jaan varjo (a < VARJO_ALARAJA) nollataan; 0,5-taso sailyy: a' = (a - 0,3) / 0,4.
    Vain graniittimaskin laajennetulla alueella ja kun |g - b| on riittava."""
    b = cam.mode.astype(np.float32)
    p = frame.astype(np.float32)
    d = graniitti_vari(cam).astype(np.float32)[None, None, :] - b
    dd = np.sum(d * d, axis=2)
    a = np.sum((p - b) * d, axis=2) / np.maximum(dd, 1.0)
    a = np.where(dd > 2500.0, a, 0.0)
    a = np.clip((np.clip(a, 0.0, 1.0) - VARJO_ALARAJA) / (1.0 - 2.0 * VARJO_ALARAJA), 0.0, 1.0)
    hsv_v = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[..., 2]
    near = cv2.dilate(gran, np.ones((5, 5), np.uint8)) > 0
    return np.where(near & (hsv_v > 60), a, 0.0).astype(np.float32)


def _ylinaytteista(h, ikkuna):
    x0, y0, x1, y1 = ikkuna
    hh = (h - np.array([x0, y0], dtype=np.float32)) * SS
    m = np.zeros(((y1 - y0) * SS, (x1 - x0) * SS), np.uint8)
    cv2.fillConvexPoly(m, np.round(hh * 16).astype(np.int32), 255, cv2.LINE_8, 4)
    return cv2.resize(m, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


def kahva_hull(cam, kivi, X, Y):
    """Ei valia -alueen siluetti: ylapinnan ympyra + kahvan ylaympyra (kupera peite)."""
    th = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
    pts = [[X + TOP_R_FRAC * kivi.R_max * np.cos(a), Y + TOP_R_FRAC * kivi.R_max * np.sin(a), kivi.H_total] for a in th]
    pts += [[X + HANDLE_R_FRAC * kivi.R_max * np.cos(a), Y + HANDLE_R_FRAC * kivi.R_max * np.sin(a), kivi.H_total + HANDLE_H_CM] for a in th]
    u, v = kivimalli.project_3d(cam.K, cam.R, cam.t, np.array(pts))
    return cv2.convexHull(np.column_stack([u, v]).astype(np.float32))[:, 0, :]


def pehmea_virhe(cam, kivi, alfa, X, Y, ikkuna, kahva_ei_valia=True):
    c = _ylinaytteista(kivi.hull(cam, X, Y), ikkuna)
    o = alfa[ikkuna[1]:ikkuna[3], ikkuna[0]:ikkuna[2]]
    if kahva_ei_valia:
        w = 1.0 - _ylinaytteista(kahva_hull(cam, kivi, X, Y), ikkuna)
    else:
        w = 1.0
    return 1.0 - float((np.minimum(c, o) * w).sum()) / max(float((np.maximum(c, o) * w).sum()), 1.0)


def sovita_pehmea(cam, kivi, alfa, X0, Y0, askel0=2.0):
    from scipy.optimize import minimize
    ikk = ikkuna_ymparille(cam, kivi, X0, Y0, marg=30)
    f = lambda p: pehmea_virhe(cam, kivi, alfa, p[0], p[1], ikk)
    r = minimize(f, [X0, Y0], method="Nelder-Mead", options=dict(initial_simplex=[[X0, Y0], [X0 + askel0, Y0], [X0, Y0 + askel0]],
                                                                  xatol=0.03, fatol=1e-5, maxiter=80))
    return float(r.x[0]), float(r.x[1]), float(r.fun)


# =====================================================================================================================
# VARISEURANTA: kahvan vari (keltainen) -> kuvan keskipiste alipikselitarkkuudella -> maa-asema
# =====================================================================================================================
KAHVA_H = (19, 31)        # kahvan sävy (OpenCV H 0..180); sormet ovat punertavampia
KAHVA_SMIN, KAHVA_VMIN = 45, 165
KAHVA_Z_CM = 15.0         # kahvan muovilevyn tason korkeus kalibrointitasosta (korjattu kamera: kahvan sade osuu rungon akseliin z = 14,4-15,5 cm, keskiarvo ~15)


PUNAINEN_H = (8, 170)     # punainen: H <= 8 tai H >= 170
PUNAINEN_SMIN, PUNAINEN_VMIN, PUNAINEN_VMAX = 100, 55, 168   # kahva V ~75-156 (varjossa matalampi), punainen rengas V ~180 -> erottuvat kirkkaudella


def kahva_maski(frame, vari="keltainen"):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    if vari == "keltainen":
        # Lab: muovi on keltainen myos vaalealla/varjoisella reunalla (b* >= 16) mutta ei punertava (iho a* > 10): tayttaa
        # kaaren paat, joita HSV-kynnys (S >= 45, V >= 165) leikkasi pois
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.int16)
        m = ((lab[..., 2] - 128) >= 16) & ((lab[..., 1] - 128) <= 10)
    else:
        m = ((hsv[..., 0] <= PUNAINEN_H[0]) | (hsv[..., 0] >= PUNAINEN_H[1])) & (hsv[..., 1] >= PUNAINEN_SMIN) & (hsv[..., 2] >= PUNAINEN_VMIN) & (hsv[..., 2] <= PUNAINEN_VMAX)
    return m.astype(np.uint8)


def kahva_sentroidi(cam, frame, ennuste_uv, vari="keltainen", max_etaisyys_px=30.0, min_ala=15):
    """Kahvakomponentti, jonka keskipiste on lahinna ennustetta (kuvapiste); palauttaa (u, v, ala) tai None."""
    m = kahva_maski(frame, vari)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    best = None
    for k in range(1, n):
        if st[k, cv2.CC_STAT_AREA] < min_ala:
            continue
        d = float(np.hypot(cen[k][0] - ennuste_uv[0], cen[k][1] - ennuste_uv[1]))
        if d <= max_etaisyys_px and (best is None or d < best[0]):
            best = (d, k)
    if best is None:
        return None
    k = best[1]
    ys, xs = np.where(lab == k)
    return float(xs.mean()), float(ys.mean()), int(st[k, cv2.CC_STAT_AREA])


def kuva_maahan(cam, u, v, z=KAHVA_Z_CM):
    return maan_paikka(cam, u, v, z=z)


# =====================================================================================================================
# KAARISOVITUS: alareunan kaari (jaalla) ja kahvan etureuna -> akseli ja korkeus
# =====================================================================================================================
# Kiven mitat (WCF: ymparysmitta <= 914 mm -> halkaisija <= 291 mm, korkeus >= 114 mm ilman kahvaa, paino 17,24-19,96 kg).
# Kuvasta nakyvan alareunan kaaren oletetaan olevan iskukaistan ympyra: sade R_ALA ja korkeus Z_ALA jaan ylapuolella (oletus).
R_ALA_CM = 14.55
Z_ALA_CM = 3.0
R_KAHVA_CM = 9.0          # kahvan peitelevyn sade (mitattu kuvasta: sivusuuntainen leveys ~17-18 cm), vain etureuna nakyy
TH_KAARI = np.linspace(0.0, 2.0 * np.pi, 181)


def ympyra_kuvaan(cam, X, Y, z, r):
    pts = np.column_stack([X + r * np.cos(TH_KAARI), Y + r * np.sin(TH_KAARI), np.full_like(TH_KAARI, z)])
    u, v = kivimalli.project_3d(cam.K, cam.R, cam.t, pts)
    return np.column_stack([u, v])


def etaisyys_monikulmioon(P, poly):
    A = poly
    B = np.roll(poly, -1, axis=0)
    AB = B - A
    L2 = np.maximum((AB ** 2).sum(1), 1e-9)
    t = np.clip(((P[:, None, :] - A[None]) * AB[None]).sum(2) / L2[None], 0, 1)
    Q = A[None] + t[..., None] * AB[None]
    return np.sqrt(((Q - P[:, None, :]) ** 2).sum(2)).min(1)


def kaaripisteet(cam, frame, uv0, vari="keltainen"):
    """(kahvan etureuna, kiven alareuna) kuvapisteina.
    Kahvan etureuna = muovilevyn ellipsin reuna LEVEIMMASTA KOHDASTA (ylin ja alin karki) KOHTI KAMERAA: kahvamaskin komponentin
    ulkoreunan pisteet, joiden u >= karkien keski-u (koko pystyalue mukana, myos kaaren paat; vasen puoli, varsi ja sormet jaavat pois).
    Alareuna: graniitin alfa 0,5 -rajan oikeanpuoleisin piste riveittain, keskimmaiset 60 % riveista."""
    m = cv2.morphologyEx(kahva_maski(frame, vari), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    k = min((i for i in range(1, n) if st[i, 4] >= 30), key=lambda i: np.hypot(cen[i][0] - uv0[0], cen[i][1] - uv0[1]))
    comp = (lab == k).astype(np.uint8)
    cs, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=len)[:, 0, :].astype(np.float64)
    top = c[c[:, 1] == c[:, 1].min()]
    bot = c[c[:, 1] == c[:, 1].max()]
    u_mid = 0.5 * (top[:, 0].mean() + bot[:, 0].mean())
    H = c[c[:, 0] >= u_mid] + 0.5
    fg, g, h = cam.maskit(frame)
    al = alfa_kartta(cam, frame, g)
    x0, y0 = int(uv0[0]) - 70, int(uv0[1]) - 70
    sub = (al[y0:y0 + 140, x0:x0 + 140] > 0.5).astype(np.uint8)
    n2, lab2, st2, cen2 = cv2.connectedComponentsWithStats(sub)
    k2 = max(range(1, n2), key=lambda i: st2[i, 4])
    ys2, xs2 = np.where(lab2 == k2)
    G = np.array([(xs2[ys2 == y].max() + 0.5 + x0, y + y0) for y in np.unique(ys2)], float)
    lo, hi = np.percentile(G[:, 1], [20, 80])
    G = G[(G[:, 1] >= lo) & (G[:, 1] <= hi)]
    return H, G


def sovita_akseli_ja_korkeus(cam, H, G, uv0, R_ala=R_ALA_CM, z_ala=Z_ALA_CM):
    """1) alareunan kaari -> akselin (X, Y) jaalla (ympyra z = z_ala, sade R_ala); 2) sama akseli, kahvan vasen (nakyva) kaari ->
    kahvalevyn ympyran korkeus z ja sade r (kaaren paat antavat sateen, sijainti korkeuden). Palauttaa
    (X, Y, z, r, rms_ala_px, rms_kahva_px)."""
    from scipy.optimize import least_squares
    X0, Y0 = maan_paikka(cam, uv0[0], uv0[1], z=15.0)
    s1 = least_squares(lambda p: etaisyys_monikulmioon(G, ympyra_kuvaan(cam, p[0], p[1], z_ala, R_ala)), [X0, Y0])
    X, Y = s1.x
    best = None
    for z0, r0 in ((10.0, 7.0), (15.0, 8.0), (20.0, 6.0)):
        s2 = least_squares(lambda p: etaisyys_monikulmioon(H, ympyra_kuvaan(cam, X, Y, p[0], p[1])), [z0, r0],
                           bounds=([2.0, 3.0], [40.0, 14.0]), loss="soft_l1", f_scale=1.0)
        if best is None or s2.cost < best.cost:
            best = s2
    d = etaisyys_monikulmioon(H, ympyra_kuvaan(cam, X, Y, best.x[0], best.x[1]))
    return X, Y, float(best.x[0]), float(best.x[1]), float(np.sqrt(s1.cost * 2 / len(G))), float(np.sqrt(np.mean(d ** 2)))


# =====================================================================================================================
# KAHVAN KUPU: kupera muovi = pohjaympyra (vaakataso, akselilla) + karki; siluetti = niiden kupera peite
# =====================================================================================================================
TH_KUPU = np.linspace(0.0, 2.0 * np.pi, 73)


def kupu_pisteet(cam, X, Y, z_pohja, r, z_karki):
    """Kuvun kuvapisteet: pohjaympyra (z_pohja, sade r) ja karki (z_karki) samalla pystyakselilla. Muovi on kupera ja peittaa itse
    pohjaympyran takapuoliskon, joten nakyva siluetti on pisteiden kupera peite: etureuna = pohjaympyran etupuolisko, takana
    tangentit karkeen (nuolenkarki)."""
    base = np.column_stack([X + r * np.cos(TH_KUPU), Y + r * np.sin(TH_KUPU), np.full_like(TH_KUPU, z_pohja)])
    pts = np.vstack([base, [[X, Y, z_karki]]])
    u, v = kivimalli.project_3d(cam.K, cam.R, cam.t, pts)
    return np.column_stack([u, v])


def kupu_peitto(P, ikkuna, S=4):
    x0, y0, x1, y1 = ikkuna
    h = cv2.convexHull(P.astype(np.float32))[:, 0, :] - np.array([x0, y0], np.float32)
    m = np.zeros(((y1 - y0) * S, (x1 - x0) * S), np.uint8)
    cv2.fillConvexPoly(m, np.round(h * S * 16).astype(np.int32), 255, cv2.LINE_8, 4)
    return cv2.resize(m, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


def muovi_maski(frame, uv0, ikkuna):
    """Muovin (keltainen, Lab b* >= 16, a* <= 10) komponentti ikkunassa: (koko, ilman vartta)."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.int16)
    m = (((lab[..., 2] - 128) >= 16) & ((lab[..., 1] - 128) <= 10)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, l, st, cen = cv2.connectedComponentsWithStats(m)
    k = min((i for i in range(1, n) if st[i, 4] >= 30), key=lambda i: np.hypot(cen[i][0] - uv0[0], cen[i][1] - uv0[1]))
    comp = (l == k).astype(np.uint8)
    opened = cv2.morphologyEx(comp, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    x0, y0, x1, y1 = ikkuna
    return comp[y0:y1, x0:x1].astype(np.float32), opened[y0:y1, x0:x1].astype(np.float32)


def sovita_kupu(cam, frame, X, Y, uv0, H, alku=None):
    """Kuvun sovitus: parametrit (dX, z_pohja, r, z_karki), (X, Y) alareunan kaaresta. Kustannus = etukaaren neliovirhe (px^2) +
    6 (1 - IoU(kupu, muovi ilman vartta)) + 8 (kuvun muovin ulkopuolinen osuus): kupu ei saa ylittaa muovia (jaata ei saa nakya
    valissa). alku = edellisen ruudun ratkaisu (lamminalku) tai None (monialku)."""
    from scipy.optimize import minimize
    ikk = (int(uv0[0]) - 45, int(uv0[1]) - 45, int(uv0[0]) + 45, int(uv0[1]) + 45)
    full, opened = muovi_maski(frame, uv0, ikk)

    def cost(p):
        dX, zb, r, za = p
        if not (3 < r < 16 and 2 < zb < 30 and zb < za < 45 and abs(dX) < 6):
            return 1e3
        E = ympyra_kuvaan(cam, X + dX, Y, zb, r)
        fa = np.mean(etaisyys_monikulmioon(H, E) ** 2)
        hm = kupu_peitto(kupu_pisteet(cam, X + dX, Y, zb, r, za), ikk)
        return (fa + 6.0 * (1 - np.minimum(hm, opened).sum() / max(np.maximum(hm, opened).sum(), 1.0))
                + 8.0 * np.maximum(hm - full, 0).sum() / max(hm.sum(), 1.0))

    starts = [np.array(alku, float)] if alku is not None else [np.array([1.0, zb, r, zb + dza]) for zb in (12, 17, 19) for r in (9, 11) for dza in (3, 7)]
    best = None
    for x0 in starts:
        sim = np.array([x0] + [x0 + np.eye(4)[i] * [2, 2, 1.5, 2][i] for i in range(4)])
        s = minimize(cost, x0, method="Nelder-Mead", options=dict(xatol=0.03, fatol=1e-4, maxiter=800, initial_simplex=sim))
        if best is None or s.fun < best.fun:
            best = s
    dX, zb, r, za = best.x
    E = ympyra_kuvaan(cam, X + dX, Y, zb, r)
    rms = float(np.sqrt(np.mean(etaisyys_monikulmioon(H, E) ** 2)))
    hm = kupu_peitto(kupu_pisteet(cam, X + dX, Y, zb, r, za), ikk)
    iou = float(np.minimum(hm, opened).sum() / max(np.maximum(hm, opened).sum(), 1.0))
    return dict(dX=float(dX), z_pohja=float(zb), r=float(r), z_karki=float(za), etukaari_rms_px=rms, iou=iou, x=best.x)


# =====================================================================================================================
# KAHVAN MUOVIN SILHUETTI JA RAJOITETTU ELLIPSISOVITUS
# =====================================================================================================================
# Kahvan muovi on kupera kappale, jonka juurella oleva ympyra (kiven ylapinnalla) nakyy ellipsina. Ellipsin tulee
#   (1) seurata silhuetin etureunaa (kameran puoli) tarkasti,
#   (2) kayda lahes silhuetin ylimmassa ja alimmassa kohdassa (leveimmat kohdat; enintaan ~2 mm pienempi),
#   (3) jaada takaa kokonaan muovin silhuetin SISALLE (jaata ei saa nakya ellipsin ja muovin valissa).
# Kohta (3) on yksipuolinen rajoite: etaisyyskentta (signed distance) muovin maskista, rangaistus vain jos ellipsin piste on
# maskin ulkopuolella (sama periaate kuin siluettipohjaisessa asennonmaarityksessa: "malli projisoituu siluetin sisalle").
# Muovin raja: Lab-keltaisuuden b* puolitaso (alipikseli, 8x ylinaytteistetty kuva), ei kovaa kynnysta.
KAHVA_YLINAYTE = 8
KAHVA_TIP_TOL_PX = 0.35        # ~2 mm (1,8 px/cm): ellipsi saa olla enintaan tama pienempi kuin silhuetti karjissa


class KahvaSilhuetti:
    pass


def kahva_silhuetti(frame, uv0, puolikas=40, Z=KAHVA_YLINAYTE, vari="keltainen", tausta=None):
    """Kahvan muovin silhuetti alipikselitarkkuudella: contour (N x 2, kuvapikselit), sd (etaisyyskentta ylinaytteistettuna,
    + ulkona, - sisalla, kuvapikseleina), front (etureunan pisteet), v_top, v_bot (silhuetin ylin ja alin kohta).
    vari: 'keltainen' (Lab b*) tai 'punainen' (Lab a*, tumma); tausta = moodikuva (Kamera.mode): staattiset samanvariset kohdat
    (vaaleanpunaiset logot jaalla, levossa olevat kivet) jaavat pois, koska muovi poikkeaa taustasta (dE)."""
    h, w = frame.shape[:2]
    cx, cy = int(round(uv0[0])), int(round(uv0[1]))
    x0, y0 = max(0, cx - puolikas), max(0, cy - puolikas)
    x1, y1 = min(w, cx + puolikas), min(h, cy + puolikas)
    crop = frame[y0:y1, x0:x1]
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[..., 0] * (100.0 / 255.0)
    a = cv2.GaussianBlur(lab[..., 1] - 128.0, (3, 3), 0)
    b = cv2.GaussianBlur(lab[..., 2] - 128.0, (3, 3), 0)
    if tausta is not None:
        lab_t = cv2.cvtColor(tausta[y0:y1, x0:x1], cv2.COLOR_BGR2LAB).astype(np.float32)
        dE = np.sqrt(((lab - lab_t) ** 2 * np.array([100.0 / 255.0, 1.0, 1.0], np.float32) ** 2).sum(2))
    else:
        dE = np.full(L.shape, 99.0, np.float32)
    ch, cw = b.shape
    if vari == "keltainen":
        feat = b
        coarse = (b >= 16) & (a <= 10) & (dE >= 8)
    else:
        feat = a
        coarse = (a >= 24) & (L <= 60) & (dE >= 15)
    coarse = cv2.morphologyEx(coarse.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, lab_c, st, cen = cv2.connectedComponentsWithStats(coarse)
    k = min((i for i in range(1, n) if st[i, 4] >= 30),
            key=lambda i: np.hypot(cen[i][0] + x0 - uv0[0], cen[i][1] + y0 - uv0[1]))
    plateau = float(np.percentile(feat[lab_c == k], 75))
    level = max(0.5 * plateau, 10.0)
    fu = cv2.resize(feat, (cw * Z, ch * Z), interpolation=cv2.INTER_CUBIC)
    au = cv2.resize(a, (cw * Z, ch * Z), interpolation=cv2.INTER_CUBIC)
    Lu = cv2.resize(L, (cw * Z, ch * Z), interpolation=cv2.INTER_CUBIC)
    du = cv2.resize(dE, (cw * Z, ch * Z), interpolation=cv2.INTER_CUBIC)
    if vari == "keltainen":
        mu = (fu >= level) & (au <= 10) & (du >= 8)
    else:
        mu = (fu >= level) & (Lu <= 70) & (du >= 12)
    mu = cv2.morphologyEx(mu.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab_u, st, cen = cv2.connectedComponentsWithStats(mu)
    ux, uy = (cen[k][0] + 0.5) * Z, (cen[k][1] + 0.5) * Z
    ku = min((i for i in range(1, n) if st[i, 4] >= 200), key=lambda i: np.hypot(cen[i][0] - ux, cen[i][1] - uy))
    comp = (lab_u == ku).astype(np.uint8)
    cs, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=len)[:, 0, :].astype(np.float64)
    filled = np.zeros_like(comp)
    cv2.fillPoly(filled, [c.astype(np.int32)], 1)
    d_out = cv2.distanceTransform(1 - filled, cv2.DIST_L2, 5)
    d_in = cv2.distanceTransform(filled, cv2.DIST_L2, 5)
    sd = ((d_out - d_in) / Z).astype(np.float32)
    contour = np.column_stack([(c[:, 0] + 0.5) / Z - 0.5 + x0, (c[:, 1] + 0.5) / Z - 0.5 + y0])
    top = contour[contour[:, 1] <= contour[:, 1].min() + 0.2]
    bot = contour[contour[:, 1] >= contour[:, 1].max() - 0.2]
    u_mid = 0.5 * (top[:, 0].mean() + bot[:, 0].mean())
    front = contour[contour[:, 0] >= u_mid][::Z // 2]
    s = KahvaSilhuetti()
    s.contour, s.sd, s.front, s.x0, s.y0, s.Z = contour, sd, front, x0, y0, Z
    s.v_top, s.v_bot, s.u_mid, s.level = float(contour[:, 1].min()), float(contour[:, 1].max()), float(u_mid), level
    return s


def sovita_alakaari(cam, G, uv0, R_ala=R_ALA_CM, z_ala=Z_ALA_CM):
    """Alareunan kaari -> akselin (X, Y) jaalla (ympyra z = z_ala, sade R_ala). Palauttaa (X, Y, rms_px)."""
    from scipy.optimize import least_squares
    X0, Y0 = maan_paikka(cam, uv0[0], uv0[1], z=15.0)
    s1 = least_squares(lambda p: etaisyys_monikulmioon(G, ympyra_kuvaan(cam, p[0], p[1], z_ala, R_ala)), [X0, Y0])
    return float(s1.x[0]), float(s1.x[1]), float(np.sqrt(s1.cost * 2 / len(G)))


def _sd_nayte(s, P):
    """Etaisyyskentan arvo kuvapisteissa P (N x 2) (bilineaarinen)."""
    mx = ((P[:, 0] - s.x0 + 0.5) * s.Z - 0.5).astype(np.float32)
    my = ((P[:, 1] - s.y0 + 0.5) * s.Z - 0.5).astype(np.float32)
    return cv2.remap(s.sd, mx.reshape(-1, 1), my.reshape(-1, 1), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).ravel()


def kahva_ellipsi_jaannokset(cam, s, p, painot=(1.0, 3.0, 6.0)):
    """Residuaalit (px) parametreille p = (X, Y, z, r): etureunan etaisyys ellipsiin, ellipsin ulkopuolelle jaavat pisteet,
    karkien rako. Palauttaa (residuaalit, E)."""
    w_front, w_cont, w_tip = painot
    E = ympyra_kuvaan(cam, p[0], p[1], p[2], p[3])
    front = etaisyys_monikulmioon(s.front, E)
    cont = np.maximum(_sd_nayte(s, E), 0.0)                       # ulkopuolella (jaata nakyisi valissa)
    tip_top = max(0.0, (E[:, 1].min() - s.v_top) - KAHVA_TIP_TOL_PX)
    tip_bot = max(0.0, (s.v_bot - E[:, 1].max()) - KAHVA_TIP_TOL_PX)
    return np.concatenate([w_front * front, w_cont * cont, [w_tip * tip_top, w_tip * tip_bot]]), E


def sovita_kahvaellipsi(cam, s, X0, Y0, prior_sx=3.0, prior_sy=6.0, painot=(1.0, 3.0, 6.0)):
    """Rajoitettu sovitus: vapaat X, Y (heikko priori alareunan kaaren akselille X0, Y0), z, r. Palauttaa dict."""
    from scipy.optimize import least_squares
    best = None
    for z0, r0 in ((12.0, 9.0), (16.0, 10.0), (19.0, 11.5)):
        def fn(p):
            r, _ = kahva_ellipsi_jaannokset(cam, s, p, painot)
            return np.concatenate([r, [(p[0] - X0) / prior_sx, (p[1] - Y0) / prior_sy]])
        sol = least_squares(fn, [X0, Y0, z0, r0], bounds=([X0 - 15, Y0 - 25, 2.0, 3.0], [X0 + 15, Y0 + 25, 40.0, 16.0]))
        if best is None or sol.cost < best.cost:
            best = sol
    p = best.x
    r, E = kahva_ellipsi_jaannokset(cam, s, p, (1.0, 1.0, 1.0))
    nf = len(s.front)
    front = r[:nf]
    cont = r[nf:nf + len(E)]
    return dict(X=p[0], Y=p[1], z=p[2], r=p[3], E=E, rms_front=float(np.sqrt(np.mean(front ** 2))),
                max_front=float(front.max()), cont_max=float(cont.max()), cont_n=int(np.sum(cont > 0.25)),
                tip_top=float(r[-2]), tip_bot=float(r[-1]),
                dX=float(p[0] - X0), dY=float(p[1] - Y0))


# ---------------------------------------------------------------------------------------------------------------------
# Kuvatason ellipsi (5 parametria) kahvan juurelle: etureuna tarkasti, karjet lahes maksimikohdassa, takaosa muovin alla
# ---------------------------------------------------------------------------------------------------------------------
TH_ELL = np.linspace(0.0, 2.0 * np.pi, 241)


def kuvaellipsi(uc, vc, a, b, phi):
    """Ellipsi kuvassa: keskipiste (uc, vc), puoliakselit a (pysty = sivusuunta) ja b (vaaka = radan suunta), kierto phi."""
    x, y = b * np.cos(TH_ELL), a * np.sin(TH_ELL)
    c, s = np.cos(phi), np.sin(phi)
    return np.column_stack([uc + c * x - s * y, vc + s * x + c * y])


def kuvaellipsi_jaannokset(s, p, painot=(1.0, 4.0, 8.0)):
    w_front, w_cont, w_tip = painot
    E = kuvaellipsi(*p)
    front = etaisyys_monikulmioon(s.front, E)
    cont = np.maximum(_sd_nayte(s, E), 0.0)                  # ulkopuolella = jaata nakyisi muovin ja ellipsin valissa
    tip_top = max(0.0, (E[:, 1].min() - s.v_top) - KAHVA_TIP_TOL_PX)
    tip_bot = max(0.0, (s.v_bot - E[:, 1].max()) - KAHVA_TIP_TOL_PX)
    return np.concatenate([w_front * front, w_cont * cont, [w_tip * tip_top, w_tip * tip_bot]]), E


def sovita_kuvaellipsi(s, painot=(1.0, 4.0, 8.0)):
    from scipy.optimize import least_squares
    vm = 0.5 * (s.v_top + s.v_bot)
    a0 = 0.5 * (s.v_bot - s.v_top)
    best = None
    for b0 in (6.0, 9.0, 13.0):
        lb = [s.u_mid - 12.0, vm - 8.0, 0.6 * a0, 2.0, -0.2]
        ub = [s.u_mid + 12.0, vm + 8.0, 1.15 * a0, 30.0, 0.2]
        sol = least_squares(lambda p: kuvaellipsi_jaannokset(s, p, painot)[0], [s.u_mid, vm, a0, b0, 0.0], bounds=(lb, ub))
        if best is None or sol.cost < best.cost:
            best = sol
    p = best.x
    r, E = kuvaellipsi_jaannokset(s, p, (1.0, 1.0, 1.0))
    nf = len(s.front)
    cont = r[nf:nf + len(E)]
    return dict(p=p, E=E, rms_front=float(np.sqrt(np.mean(r[:nf] ** 2))), max_front=float(r[:nf].max()),
                cont_max=float(cont.max()), cont_n=int(np.sum(cont > 0.25)), tip_top=float(r[-2]), tip_bot=float(r[-1]))


def kahva_keskipiste_3d(cam, uc, vc, Y_ala, X0=0.0):
    """Ellipsin keskipiste (uc, vc) = kahvan juuren ympyran keskipiste = kiven akseli: ratkaistaan (X, z) kun Y on alareunan kaaresta
    (akseli jaalla); 2 yhtalo, 2 tuntematonta. Palauttaa (X, z, jaannos px)."""
    from scipy.optimize import least_squares
    best = None
    for z0 in (8.0, 14.0, 20.0):
        def fn(q):
            u, v = kivimalli.project_3d(cam.K, cam.R, cam.t, np.array([[q[0], Y_ala, q[1]]]))
            return [u[0] - uc, v[0] - vc]
        sol = least_squares(fn, [X0, z0], bounds=([X0 - 40, 0.0], [X0 + 40, 45.0]))
        if best is None or sol.cost < best.cost:
            best = sol
    return float(best.x[0]), float(best.x[1]), float(np.sqrt(2 * best.cost))


def alareuna_pisteet(cam, frame, uv0):
    """Kiven alareunan (kameran puoleinen kaari) pisteet: graniitin alfa 0,5 -rajan oikeanpuoleisin piste riveittain,
    keskimmaiset 60 % riveista."""
    fg, g, h = cam.maskit(frame)
    al = alfa_kartta(cam, frame, g)
    x0, y0 = int(uv0[0]) - 70, int(uv0[1]) - 70
    sub = (al[y0:y0 + 140, x0:x0 + 140] > 0.5).astype(np.uint8)
    n2, lab2, st2, cen2 = cv2.connectedComponentsWithStats(sub)
    k2 = max(range(1, n2), key=lambda i: st2[i, 4])
    ys2, xs2 = np.where(lab2 == k2)
    G = np.array([(xs2[ys2 == y].max() + 0.5 + x0, y + y0) for y in np.unique(ys2)], float)
    lo, hi = np.percentile(G[:, 1], [20, 80])
    return G[(G[:, 1] >= lo) & (G[:, 1] <= hi)]
