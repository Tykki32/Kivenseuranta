"""Yhden pesan kalibrointi nettivideolle (kamera katossa, kuvassa yksi pesa + sen hogline).

Nettivideon ruutu on jaettu kahtia: vasen ja oikea puolisko ovat radan eri paiden kamerat, ja kummassakin nakyy yksi
pesa (heittopaan pesa: hack pesan takana, kivi liukuu pesan lapi ja irtoaa ennen hoglinea). Seurannan kalibrointi
(kalibrointi.py) vaatii lahi- ja kaukopesan seka molemmat hoglinjat, joten tama on oma, yksinkertaisempi kalibrointi:
kun pesa loytyy, kaikki muut viivat ovat tunnetussa paikassa pesaan nahden.

Fyysinen koordinaatisto (cm), per puolisko: origo = pesan keskipiste (T), Y kasvaa kohti hoglinea (hogline Y = 640),
X lateraalisesti. Kuva kaannetaan 90 astetta vastapaivaan, jotta pesa on vasemmalla ja hogline oikealla (kuten
alkuperaisessa kuvassa rata kulkee vaakasuunnassa): kuvassa +Y = oikealle, +X = alas (ei peilausta).

Menetelma:
 1) moodikuva (mediaani 5 s valein otetuista ruuduista) -> ihmiset ja kivet haviavat
 2) sinisen/punaisen renkaan ellipsit (kalibrointi_perus), T-viiva ja keskiviiva -> alkuhomografia 4 pisteesta
    (keskiviivan ja T-viivan leikkaukset ulkorenkaan kanssa)
 3) epalineaarinen sovitus (homografia + objektiivin k1): renkaiden reunat ovat ympyroita, T-viiva Y = 0, keskiviiva X = 0,
    hogline Y = 640. Viivapisteet haetaan joka kierroksella nykyisen mallin ennustamasta paikasta (tumma harju).
 4) top-down-kuva ja tarkistuskuva (ympyrat ja viivat paalle).
"""
import json
import math
import os

import cv2
import numpy as np
from scipy.optimize import least_squares

import kalibrointi_perus as kp
import rata

# puoliskon rajaus alkuperaisesta 1920x1080-ruudusta: (x, y, leveys, korkeus). Ylareunan TV-grafiikka (minikartta,
# sponsoribanneri) jaa rajauksen ylapuolelle.
PUOLET = {"vasen": (0, 190, 960, 890), "oikea": (960, 190, 960, 890)}

# Kummankin puoliskon kuva on kameran 16:9-kuvan keskileikkaus (960x1080), joten optinen keskipiste (ja vaaristyman keskipiste)
# on puoliskon keskella (480, 540). Kuvassa jonka moodikuva() antaa (rajaus + 90 asteen kierto vastapaivaan) se on:
# x = 540 - rajaus_y, y = (rajaus_leveys - 1) - 480.
PERIAPISTE_PUOLISKOSSA = (480.0, 540.0)


def periapiste(puoli):
    x0, y0, w, h = PUOLET[puoli]
    px, py = PERIAPISTE_PUOLISKOSSA
    return np.array([py - y0, (w - 1) - px])


HOG_Y_CM = 640.0                 # hoglinen etaisyys T-viivasta
RINGS_CM = {"sin_ulko": rata.BLUE_OUTER_RADIUS_CM, "sin_sisa": rata.BLUE_INNER_RADIUS_CM,
            "pun_ulko": rata.RED_OUTER_RADIUS_CM, "pun_sisa": rata.RED_INNER_RADIUS_CM}

TOPDOWN_X_CM = (-260.0, 260.0)
TOPDOWN_Y_CM = (-450.0, 900.0)   # alhaalla hack-puoli, ylhaalla hogline ja sen ylapuoli
TOPDOWN_PPC = 2.0                # px / cm
K1_SCALE_PX = 1000.0


# =====================================================================================================================
# MOODIKUVA
# =====================================================================================================================
def moodikuva(video, puoli, nayteväli_s=5.0, max_s=120.0):
    """Puolison moodikuva (mediaani), kaannettuna 90 astetta vastapaivaan."""
    x, y, w, h = PUOLET[puoli]
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    t = 0.0
    while t < max_s and int(t * fps) < n:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr[y:y + h, x:x + w])
        t += nayteväli_s
    cap.release()
    med = np.median(np.stack(frames, axis=0), axis=0).astype(np.uint8)
    return cv2.rotate(med, cv2.ROTATE_90_COUNTERCLOCKWISE), len(frames)


# =====================================================================================================================
# MALLI: kuvapiste -> (objektiivin k1) -> homografia -> fyysinen (X, Y) cm
# =====================================================================================================================
def _undistort_px(pts, k1, centre):
    d = pts - centre
    r2 = np.sum((d / K1_SCALE_PX) ** 2, axis=1, keepdims=True)
    return centre + d * (1.0 + k1 * r2)


def _distort_px(pts_u, k1, centre):
    """_undistort_px:n kaanteinen (kiintopisteiteraatio)."""
    p = pts_u.copy()
    for _ in range(8):
        d = p - centre
        r2 = np.sum((d / K1_SCALE_PX) ** 2, axis=1, keepdims=True)
        p = centre + (pts_u - centre) / (1.0 + k1 * r2)
    return p


def _apply_h(H, pts):
    hom = np.hstack([pts, np.ones((len(pts), 1))]) @ H.T
    return hom[:, :2] / hom[:, 2:3]


def _h_from_params(p):
    return np.append(p[:8], 1.0).reshape(3, 3)


class Malli:
    def __init__(self, H, k1, centre):
        self.H, self.k1, self.centre = np.asarray(H, dtype=np.float64), float(k1), np.asarray(centre, dtype=np.float64)

    def kuva_fyysiseksi(self, pts):
        return _apply_h(self.H, _undistort_px(np.asarray(pts, dtype=np.float64).reshape(-1, 2), self.k1, self.centre))

    def fyysinen_kuvaksi(self, xy):
        u = _apply_h(np.linalg.inv(self.H), np.asarray(xy, dtype=np.float64).reshape(-1, 2))
        return _distort_px(u, self.k1, self.centre)


# =====================================================================================================================
# TUNNISTUS
# =====================================================================================================================
def _ellipse_points(ellipse, n=90):
    (cx, cy), (w, h), ang = ellipse
    th = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    a = math.radians(ang)
    x, y = 0.5 * w * np.cos(th), 0.5 * h * np.sin(th)
    return np.column_stack([cx + x * math.cos(a) - y * math.sin(a), cy + x * math.sin(a) + y * math.cos(a)])


def etsi_renkaat(frame):
    """Sinisen ja punaisen renkaan ulko- ja sisaellipsit. Punainen haetaan vain sinisen ulkorenkaan sisalta."""
    blue_mask = kp.create_blue_mask(frame)
    b_out, b_in = kp.find_house_pair(blue_mask, kp.NEAR_BLUE_MIN_AREA, kp.NEAR_BLUE_MIN_RATIO, kp.NEAR_BLUE_MIN_SIZE_RATIO)
    if b_out is None or b_in is None:
        raise RuntimeError("Sinista rengasta ei loytynyt.")
    inside = np.zeros(blue_mask.shape[:2], np.uint8)
    cv2.ellipse(inside, b_out, 255, -1)
    red_mask = cv2.bitwise_and(kp.create_red_mask(frame), inside)
    r_out, r_in = kp.find_house_pair(red_mask, kp.NEAR_RED_MIN_AREA, kp.NEAR_RED_MIN_RATIO, kp.NEAR_RED_MIN_SIZE_RATIO)
    if r_out is None or r_in is None:
        raise RuntimeError("Punaista rengasta ei loytynyt.")
    return {"sin_ulko": b_out, "sin_sisa": b_in, "pun_ulko": r_out, "pun_sisa": r_in}


def _harju(gray, ksize=31):
    """Ymparistoaan tummemmat kapeat alueet (viivat): mediaaniulos - harmaasavy."""
    return (cv2.medianBlur(gray, ksize).astype(np.float32) - gray.astype(np.float32))


def alkuhomografia(frame, ellipses):
    """4 pisteen homografia: keskiviivan ja T-viivan leikkaukset ulkorenkaan kanssa. Kuva: +Y oikealle, +X alas."""
    segments = kp.detect_line_segments(frame, ellipses["sin_ulko"], faint=True)
    _t_pair, _cl_pair, t_line, centerline, centre = kp.select_t_and_centerline(segments, ellipses["sin_ulko"])
    r = rata.BLUE_OUTER_RADIUS_CM
    cl_pts = kp.line_ellipse_intersections(centerline, ellipses["sin_ulko"])     # keskiviiva: Y = -r (vasen), +r (oikea)
    t_pts = kp.line_ellipse_intersections(t_line, ellipses["sin_ulko"])          # T-viiva: X = -r (ylos), +r (alas)
    cl_pts = sorted(cl_pts, key=lambda p: p[0])
    t_pts = sorted(t_pts, key=lambda p: p[1])
    src = np.array([cl_pts[0], cl_pts[1], t_pts[0], t_pts[1]], dtype=np.float32)
    dst = np.array([[0, -r], [0, r], [-r, 0], [r, 0]], dtype=np.float32)
    return cv2.getPerspectiveTransform(src, dst).astype(np.float64)


def viivapisteet(harju, malli, viiva, rajat_cm, puoli_leveys_px, min_harju, ohita_pesa_cm=None, askel_cm=15.0):
    """Viivan pisteet kuvasta nykyisen mallin ennustamasta paikasta. viiva: ("X", x0) = viiva X = x0 (suunta Y) tai
    ("Y", y0) = viiva Y = y0 (suunta X). Palauttaa kuvapisteet (N x 2), joissa harju on selva."""
    akseli, arvo = viiva
    s = np.arange(rajat_cm[0], rajat_cm[1] + 1e-6, askel_cm)
    xy = np.column_stack([np.full_like(s, arvo), s]) if akseli == "X" else np.column_stack([s, np.full_like(s, arvo)])
    if ohita_pesa_cm is not None:
        keep = np.hypot(xy[:, 0], xy[:, 1]) > ohita_pesa_cm
        xy, s = xy[keep], s[keep]
    p = malli.fyysinen_kuvaksi(xy)
    # viivan suunta kuvassa -> normaali
    d = np.gradient(p, axis=0)
    n = np.column_stack([-d[:, 1], d[:, 0]])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    h, w = harju.shape
    offs = np.arange(-puoli_leveys_px, puoli_leveys_px + 1, 1.0)
    out = []
    for pi, ni in zip(p, n):
        q = pi[None, :] + offs[:, None] * ni[None, :]
        if (q[:, 0] < 2).any() or (q[:, 0] > w - 3).any() or (q[:, 1] < 2).any() or (q[:, 1] > h - 3).any():
            continue
        vals = cv2.remap(harju, q[:, 0:1].astype(np.float32), q[:, 1:2].astype(np.float32), cv2.INTER_LINEAR).ravel()
        vals = np.convolve(vals, np.ones(3) / 3.0, mode="same")
        i = int(np.argmax(vals[1:-1])) + 1
        if vals[i] < min_harju or i <= 2 or i >= len(vals) - 3:       # harju haun reunalla = ei oikea viiva
            continue
        # alipikselitarkennus paraabelilla
        a, b, c = vals[i - 1], vals[i], vals[i + 1]
        den = a - 2 * b + c
        di = 0.5 * (a - c) / den if abs(den) > 1e-9 else 0.0
        out.append(pi + (offs[i] + di) * ni)
    return np.array(out, dtype=np.float64).reshape(-1, 2)


# =====================================================================================================================
# SOVITUS
# =====================================================================================================================
def sovita(frame, ellipses, H0, centre, k1_vapaa=True, kierrokset=3, verbose=True):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    harju = _harju(gray)
    ring_pts = {k: _ellipse_points(e) for k, e in ellipses.items()}
    S = np.diag([1.0 / K1_SCALE_PX, 1.0 / K1_SCALE_PX, 1.0])      # parametrit: kuvakoordinaatit / 1000
    Hn = H0 @ np.linalg.inv(S)
    p0 = np.concatenate([(Hn / Hn[2, 2]).ravel()[:8], [0.0]])
    malli = Malli(H0, 0.0, centre)
    viivat = []

    def to_model(p):
        Hh = _h_from_params(p) @ S
        return Malli(Hh, p[8] if k1_vapaa else 0.0, centre)

    for kierros in range(kierrokset):
        # viivapisteet nykyisen mallin mukaan
        viivat = [
            ("keskiviiva", ("X", 0.0), (-450.0, 1000.0), 10, 5.0, 190.0),
            ("T-viiva", ("Y", 0.0), (-300.0, 300.0), 10, 5.0, 190.0),
            ("hogline", ("Y", HOG_Y_CM), (-260.0, 260.0), 22, 8.0, None),
        ]
        viivapts = {}
        for nimi, viiva, rajat, pw, thr, ohita in viivat:
            viivapts[nimi] = viivapisteet(harju, malli, viiva, rajat, pw, thr, ohita)

        def resid(p):
            m = to_model(p)
            parts = []
            for k, pts in ring_pts.items():
                xy = m.kuva_fyysiseksi(pts)
                parts.append(np.hypot(xy[:, 0], xy[:, 1]) - RINGS_CM[k])
            for nimi, pts in viivapts.items():
                if len(pts) < 3:
                    continue
                xy = m.kuva_fyysiseksi(pts)
                akseli, arvo = dict((n, v) for n, v, *_ in viivat)[nimi]
                parts.append((xy[:, 0] - arvo) if akseli == "X" else (xy[:, 1] - arvo))
            if k1_vapaa:
                parts.append(np.array([p[8] * 30.0]))      # kevyt regularisointi
            return np.concatenate(parts)

        sol = least_squares(resid, p0, loss="soft_l1", f_scale=2.0, max_nfev=400)
        p0 = sol.x
        malli = to_model(sol.x)
        if verbose:
            r = resid(sol.x)
            print(f"  kierros {kierros + 1}: RMS {math.sqrt(np.mean(r ** 2)):.2f} cm, k1 {malli.k1:+.4f}, viivapisteita "
                  + ", ".join(f"{n} {len(v)}" for n, v in viivapts.items()))
    return malli, viivapts, ring_pts


# ---------------------------------------------------------------------------------------------------------------------
# TARKENNUS: renkaiden ja hoglinen reunat alipikselitarkkuudella (puolitasokohta)
# ---------------------------------------------------------------------------------------------------------------------
HOG_LEVEYS_CM = 10.0                     # hoglinen leveys; virallinen mitta T-viivaan on SISAREUNALLE (pesan puoleinen reuna)
REUNA_IKKUNA_CM = 14.0                   # tasanteet r0 -+ (9..14) cm, reuna haetaan r0 +- 8 cm
REUNA_MIN_KONTRASTI = 50.0
RENGAS_REUNAT = {                        # nimi: (r0, kanava (BGR-indeksi), "sisa korkea" = sisapuolella kirkkaampi)
    "sin_ulko": (rata.BLUE_OUTER_RADIUS_CM, 2, False),
    "sin_sisa": (rata.BLUE_INNER_RADIUS_CM, 2, True),
    "pun_ulko": (rata.RED_OUTER_RADIUS_CM, 1, False),
    "pun_sisa": (rata.RED_INNER_RADIUS_CM, 1, True),
}


def _nayte(kanava_img, malli, xy):
    p = malli.fyysinen_kuvaksi(xy).astype(np.float32)
    return cv2.remap(kanava_img, p[:, 0:1], p[:, 1:2], cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE).ravel()


def rengasreunat(frame, malli, askel_aste=2.0):
    """Renkaiden reunapisteet kuvapisteina: sateittaiset profiilit nykyisen mallin top-down-avaruudessa, reuna = kohta
    jossa kanava ylittaa sisa- ja ulkotasanteen puolivalin (sumeassa kuvassa symmetrinen -> ei harhaa)."""
    kanavat = [frame[..., i].astype(np.float32) for i in range(3)]
    thetas = np.deg2rad(np.arange(0.0, 360.0, askel_aste))
    out = {}
    for nimi, (r0, ch, sisa_korkea) in RENGAS_REUNAT.items():
        radii = r0 + np.arange(-REUNA_IKKUNA_CM, REUNA_IKKUNA_CM + 1e-6, 0.25)
        pts = []
        for th in thetas:
            xy = np.column_stack([radii * math.cos(th), radii * math.sin(th)])
            prof = _nayte(kanavat[ch], malli, xy)
            rel = radii - r0
            a = float(np.mean(prof[(rel >= -REUNA_IKKUNA_CM) & (rel <= -9.0)]))     # sisatasanne
            b = float(np.mean(prof[(rel >= 9.0) & (rel <= REUNA_IKKUNA_CM)]))       # ulkotasanne
            if abs(a - b) < REUNA_MIN_KONTRASTI or (a > b) != sisa_korkea:
                continue
            half = 0.5 * (a + b)
            seg = np.where(np.abs(rel) <= 8.0)[0]
            f = prof[seg] - half
            cross = np.where(np.sign(f[:-1]) != np.sign(f[1:]))[0]
            if len(cross) == 0:
                continue
            k = cross[np.argmin(np.abs(rel[seg][cross]))]
            t = f[k] / (f[k] - f[k + 1])
            r_edge = rel[seg][k] + t * (rel[seg][k + 1] - rel[seg][k]) + r0
            pts.append([r_edge * math.cos(th), r_edge * math.sin(th)])
        if pts:
            P = np.array(pts)
            rr = np.hypot(P[:, 0], P[:, 1])[:, None]
            u = P / rr
            sc = 1.0 / np.maximum(np.linalg.norm(malli.fyysinen_kuvaksi(P + 0.5 * u) - malli.fyysinen_kuvaksi(P - 0.5 * u), axis=1), 1e-6)
            out[nimi] = (malli.fyysinen_kuvaksi(P), sc)         # kuvapisteet, cm / px reunan suunnassa
        else:
            out[nimi] = (np.zeros((0, 2)), np.zeros(0))
    return out


def hogreunat(frame, malli, x_vali=(-225.0, 225.0), askel_cm=6.0):
    """Hoglinen sisa- (pesan puoleinen, pienempi Y) ja ulkoreuna kuvapisteina. Profiili Y-suunnassa jokaisella X:lla:
    tummin kohta -> reunat puolitasokohdista valkoisen jaan ja viivan valilla. Logojen valkoiset kohdat ohitetaan."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    y_c = HOG_Y_CM + 0.5 * HOG_LEVEYS_CM
    ys = np.arange(y_c - 30.0, y_c + 30.0 + 1e-6, 0.25)
    inner, outer = [], []
    for x in np.arange(x_vali[0], x_vali[1] + 1e-6, askel_cm):
        prof = _nayte(gray, malli, np.column_stack([np.full_like(ys, x), ys]))
        prof = np.convolve(prof, np.ones(5) / 5.0, mode="same")
        i = int(np.argmin(prof[20:-20])) + 20
        lo = float(np.mean(prof[:8 * 4]))
        hi = float(np.mean(prof[-8 * 4:]))
        dark = float(np.mean(prof[max(0, i - 2):i + 3]))
        if min(lo, hi) - dark < 40.0 or abs(lo - hi) > 25.0:
            continue
        half = 0.5 * (0.5 * (lo + hi) + dark)

        def crossing(step):
            j = i
            while 0 < j < len(prof) - 1 and prof[j] < half:
                j += step
            if not 0 < j < len(prof) - 1:
                return None
            jj = j - step
            t = (half - prof[jj]) / (prof[j] - prof[jj])
            return ys[jj] + t * (ys[j] - ys[jj])

        y_in, y_out = crossing(-1), crossing(+1)
        if y_in is None or y_out is None or not 6.0 <= (y_out - y_in) <= 14.0:
            continue
        inner.append([x, y_in])
        outer.append([x, y_out])
    def f(a):
        if not a:
            return np.zeros((0, 2)), np.zeros(0)
        P = np.array(a)
        dy = np.column_stack([np.zeros(len(P)), np.full(len(P), 0.5)])
        sc = 1.0 / np.maximum(np.linalg.norm(malli.fyysinen_kuvaksi(P + dy) - malli.fyysinen_kuvaksi(P - dy), axis=1), 1e-6)
        return malli.fyysinen_kuvaksi(P), sc
    return f(inner), f(outer)


def sovita_reunoilla(frame, malli, k1_vapaa=True, kierrokset=4, verbose=True, hog_paino=1.0):
    """Parametrit: 8 homografian, k1 ja g = tumman alueen kutistuma kuvapikseleina (sumea ja teroitettu kuva: puolitasokohta
    on hieman tumman alueen sisapuolella; cm-virhe = g * cm/px reunan suunnassa -> suurempi radan suunnassa)."""
    centre = malli.centre
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    harju = _harju(gray)
    S = np.diag([1.0 / K1_SCALE_PX, 1.0 / K1_SCALE_PX, 1.0])
    Hn = malli.H @ np.linalg.inv(S)
    p0 = np.concatenate([(Hn / Hn[2, 2]).ravel()[:8], [malli.k1, 0.0, 0.0]])

    def to_model(p):
        return Malli(_h_from_params(p) @ S, p[8] if k1_vapaa else 0.0, centre)

    for kierros in range(kierrokset):
        rengas = rengasreunat(frame, malli)
        (hog_in, sc_in), (hog_out, sc_out) = hogreunat(frame, malli)
        lp = {
            "keskiviiva": viivapisteet(harju, malli, ("X", 0.0), (-450.0, 1000.0), 10, 5.0, 190.0),
            "T-viiva": viivapisteet(harju, malli, ("Y", 0.0), (-300.0, 300.0), 10, 5.0, 190.0),
        }

        def resid(p):
            m = to_model(p)
            g, gh = p[9], p[10]
            parts = []
            for nimi, (pts, sc) in rengas.items():
                if len(pts):
                    xy = m.kuva_fyysiseksi(pts)
                    sign = 1.0 if RENGAS_REUNAT[nimi][2] else -1.0
                    parts.append(np.hypot(xy[:, 0], xy[:, 1]) - RINGS_CM[nimi] - sign * g * sc)
            if len(hog_in):
                parts.append(hog_paino * (m.kuva_fyysiseksi(hog_in)[:, 1] - HOG_Y_CM - gh * sc_in))
                parts.append(hog_paino * (m.kuva_fyysiseksi(hog_out)[:, 1] - (HOG_Y_CM + HOG_LEVEYS_CM) + gh * sc_out))
            for nimi, col in (("keskiviiva", 0), ("T-viiva", 1)):
                if len(lp[nimi]) >= 3:
                    parts.append(m.kuva_fyysiseksi(lp[nimi])[:, col])
            if k1_vapaa:
                parts.append(np.array([p[8] * 30.0]))
            return np.concatenate(parts)

        sol = least_squares(resid, p0, loss="soft_l1", f_scale=1.0, max_nfev=400)
        p0 = sol.x
        malli = to_model(sol.x)
        malli.g_px, malli.g_hog_px = float(sol.x[9]), float(sol.x[10])
        if verbose:
            r = resid(sol.x)
            print(f"  tarkennus {kierros + 1}: RMS {math.sqrt(np.mean(r ** 2)):.2f} cm, MAD {1.4826 * np.median(np.abs(r)):.2f} cm, "
                  f"k1 {malli.k1:+.4f}, g {malli.g_px:+.3f} px, g_hog {malli.g_hog_px:+.3f} px; reunapisteita " + ", ".join(f"{n} {len(v[0])}" for n, v in rengas.items())
                  + f", hogline {len(hog_in)}, keski {len(lp['keskiviiva'])}, T {len(lp['T-viiva'])}")
    return malli, dict(rengas=rengas, hog_in=(hog_in, sc_in), hog_out=(hog_out, sc_out), viivat=lp, g=malli.g_px, g_hog=malli.g_hog_px)


def residuaalit_reunat(malli, d):
    """Residuaalit (cm) kutistuma g huomioiden: RMS, mediaani (etumerkillinen) ja n."""
    g, gh = d["g"], d["g_hog"]
    out = {}

    def stat(e):
        return (float(np.sqrt(np.mean(e ** 2))), float(np.median(e)), len(e))

    for nimi, (pts, sc) in d["rengas"].items():
        if len(pts):
            xy = malli.kuva_fyysiseksi(pts)
            sign = 1.0 if RENGAS_REUNAT[nimi][2] else -1.0
            out[nimi] = stat(np.hypot(xy[:, 0], xy[:, 1]) - RINGS_CM[nimi] - sign * g * sc)
    for nimi, (pts, sc), tgt, sg in (("hog_sisareuna", d["hog_in"], HOG_Y_CM, 1.0), ("hog_ulkoreuna", d["hog_out"], HOG_Y_CM + HOG_LEVEYS_CM, -1.0)):
        if len(pts):
            out[nimi] = stat(malli.kuva_fyysiseksi(pts)[:, 1] - tgt - sg * gh * sc)
    for nimi, col in (("keskiviiva", 0), ("T-viiva", 1)):
        pts = d["viivat"][nimi]
        if len(pts) >= 3:
            out[nimi] = stat(malli.kuva_fyysiseksi(pts)[:, col])
    return out


def kamera_asento(malli):
    """Kameran sisaiset ja ulkoiset parametrit tasohomografiasta (Zhang): K = [[f,0,cx],[0,f,cy],[0,0,1]] (optinen
    keskipiste tunnettu, f ortogonaalisuusehdosta r1.r2 = 0), p = K (R X + t), X fyysinen (X, Y, Z) cm, Z ylos jaasta.
    Kuvapisteet ovat vaaristymakorjattuja (k1 pois): kayta malli.kuva_oikaistuksi()."""
    Hp = np.linalg.inv(malli.H)          # fyysinen (X, Y, 1) -> oikaistu kuvapiste
    cx, cy = malli.centre
    col = lambda h: np.array([h[0] - cx * h[2], h[1] - cy * h[2], h[2]])
    a, b = col(Hp[:, 0]), col(Hp[:, 1])
    f = math.sqrt(abs(-(a[0] * b[0] + a[1] * b[1]) / (a[2] * b[2])))
    K = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])
    M = np.linalg.inv(K) @ Hp
    sc = 2.0 / (np.linalg.norm(M[:, 0]) + np.linalg.norm(M[:, 1]))
    r1, r2, t = M[:, 0] * sc, M[:, 1] * sc, M[:, 2] * sc
    U, _, Vt = np.linalg.svd(np.column_stack([r1, r2, np.cross(r1, r2)]))
    R = U @ Vt
    C = -R.T @ t
    if C[2] < 0:                          # kamera on jaan ylapuolella
        R = R.copy()
        R[:, :2] *= -1.0
        t = -t
        C = -R.T @ t
    return K, R, t, C


def residuaalit(malli, ring_pts, viivapts):
    """Selkokieliset residuaalit (cm)."""
    out = {}
    for k, pts in ring_pts.items():
        xy = malli.kuva_fyysiseksi(pts)
        out[k] = float(np.sqrt(np.mean((np.hypot(xy[:, 0], xy[:, 1]) - RINGS_CM[k]) ** 2)))
    for nimi, pts in viivapts.items():
        if len(pts) < 3:
            out[nimi] = None
            continue
        xy = malli.kuva_fyysiseksi(pts)
        col = 0 if nimi == "keskiviiva" else 1
        tgt = {"keskiviiva": 0.0, "T-viiva": 0.0, "hogline": HOG_Y_CM}[nimi]
        out[nimi] = float(np.sqrt(np.mean((xy[:, col] - tgt) ** 2)))
    return out


# =====================================================================================================================
# TOP-DOWN JA TARKISTUSKUVA
# =====================================================================================================================
def _topdown_ruudukko():
    xs = np.arange(TOPDOWN_X_CM[0], TOPDOWN_X_CM[1], 1.0 / TOPDOWN_PPC)
    ys = np.arange(TOPDOWN_Y_CM[1], TOPDOWN_Y_CM[0], -1.0 / TOPDOWN_PPC)       # ylhaalla suuri Y
    gx, gy = np.meshgrid(xs, ys)
    return gx, gy


def topdown(frame, malli):
    gx, gy = _topdown_ruudukko()
    xy = np.column_stack([gx.ravel(), gy.ravel()])
    p = malli.fyysinen_kuvaksi(xy).reshape(gx.shape + (2,)).astype(np.float32)
    return cv2.remap(frame, p[..., 0], p[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(40, 40, 40))


def _px(x_cm, y_cm):
    return int(round((x_cm - TOPDOWN_X_CM[0]) * TOPDOWN_PPC)), int(round((TOPDOWN_Y_CM[1] - y_cm) * TOPDOWN_PPC))


def piirra_overlay(td):
    out = td.copy()
    for r in RINGS_CM.values():
        cv2.circle(out, _px(0, 0), int(round(r * TOPDOWN_PPC)), (0, 220, 0), 1, cv2.LINE_AA)
    col = (0, 0, 255)
    cv2.line(out, _px(0, TOPDOWN_Y_CM[0]), _px(0, TOPDOWN_Y_CM[1]), col, 1, cv2.LINE_AA)                       # keskiviiva
    cv2.line(out, _px(TOPDOWN_X_CM[0], 0), _px(TOPDOWN_X_CM[1], 0), col, 1, cv2.LINE_AA)                       # T-viiva
    cv2.line(out, _px(TOPDOWN_X_CM[0], HOG_Y_CM), _px(TOPDOWN_X_CM[1], HOG_Y_CM), col, 1, cv2.LINE_AA)          # hogline (sisareuna)
    cv2.line(out, _px(TOPDOWN_X_CM[0], HOG_Y_CM + HOG_LEVEYS_CM), _px(TOPDOWN_X_CM[1], HOG_Y_CM + HOG_LEVEYS_CM), (0, 160, 255), 1, cv2.LINE_AA)  # ulkoreuna
    cv2.line(out, _px(TOPDOWN_X_CM[0], -rata.HOUSE_RADIUS_CM), _px(TOPDOWN_X_CM[1], -rata.HOUSE_RADIUS_CM), col, 1, cv2.LINE_AA)  # takaraja
    for x in (-237.5, 237.5):                                                                                   # radan reunat (4,75 m)
        cv2.line(out, _px(x, TOPDOWN_Y_CM[0]), _px(x, TOPDOWN_Y_CM[1]), (255, 160, 0), 1, cv2.LINE_AA)
    return out


# =====================================================================================================================
def kalibroi_puoli(video, puoli, ulos, k1_vapaa=True):
    print(f"=== {puoli} ===")
    frame, n = moodikuva(video, puoli)
    print(f"  moodikuva {frame.shape[1]}x{frame.shape[0]} ({n} naytetta)")
    os.makedirs(ulos, exist_ok=True)
    cv2.imwrite(os.path.join(ulos, f"moodikuva_{puoli}.png"), frame)
    ellipses = etsi_renkaat(frame)
    H0 = alkuhomografia(frame, ellipses)
    malli, viivapts, ring_pts = sovita(frame, ellipses, H0, periapiste(puoli), k1_vapaa=k1_vapaa)
    res_a = residuaalit(malli, ring_pts, viivapts)
    print("  ellipsisovitus, residuaalit (cm RMS): " + ", ".join(f"{k} {v:.2f}" if v is not None else f"{k} -" for k, v in res_a.items()))
    malli, d = sovita_reunoilla(frame, malli, k1_vapaa=k1_vapaa)
    resr = residuaalit_reunat(malli, d)
    print("  reunasovitus, residuaalit (cm RMS / mediaani / n): "
          + ", ".join(f"{k} {v[0]:.2f}/{v[1]:+.2f}/{v[2]}" for k, v in resr.items()))
    res = {k: {"rms_cm": v[0], "mediaani_cm": v[1], "n": v[2]} for k, v in resr.items()}
    td = topdown(frame, malli)
    cv2.imwrite(os.path.join(ulos, f"topdown_{puoli}.png"), td)
    cv2.imwrite(os.path.join(ulos, f"topdown_{puoli}_tarkistus.jpg"), piirra_overlay(td), [cv2.IMWRITE_JPEG_QUALITY, 90])
    K, R, t, C = kamera_asento(malli)
    print(f"  kamera: f {K[0, 0]:.0f} px, paikka (X, Y, Z) = ({C[0]:.0f}, {C[1]:.0f}, {C[2]:.0f}) cm, optisen akselin kulma pystysta "
          f"{math.degrees(math.acos(min(1.0, abs((R.T @ np.array([0, 0, 1.0]))[2])))):.1f} astetta")
    with open(os.path.join(ulos, f"kalibrointi_{puoli}.json"), "w") as f:
        json.dump({"puoli": puoli, "rajaus_xywh": PUOLET[puoli], "kierto": "90 astetta vastapaivaan",
                   "H_kuva_fyysinen": malli.H.tolist(), "k1": malli.k1, "kutistuma_g_px": getattr(malli, "g_px", 0.0), "kutistuma_hog_px": getattr(malli, "g_hog_px", 0.0), "k1_skaala_px": K1_SCALE_PX,
                   "kuvan_keskipiste": malli.centre.tolist(), "K": K.tolist(), "R": R.tolist(), "t": t.tolist(), "kameran_paikka_cm": C.tolist(), "hogline_y_cm": HOG_Y_CM, "residuaalit_cm_rms": res}, f, indent=1)
    return malli, frame, td


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Nettivideon yhden pesan kalibrointi ja top-down-kuva")
    ap.add_argument("video")
    ap.add_argument("--ulos", default="tulokset")
    ap.add_argument("--puolet", default="vasen,oikea")
    ap.add_argument("--ei-k1", action="store_true", help="ei objektiivin vaaristymaa")
    a = ap.parse_args()
    for pu in a.puolet.split(","):
        kalibroi_puoli(a.video, pu, a.ulos, k1_vapaa=not a.ei_k1)
