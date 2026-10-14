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
PUOLET = {"vasen": (0, 280, 960, 800), "oikea": (960, 280, 960, 800)}

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
        if vals[i] < min_harju:
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
def sovita(frame, ellipses, H0, k1_vapaa=True, kierrokset=3, verbose=True):
    h, w = frame.shape[:2]
    centre = np.array([w / 2.0, h / 2.0])
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
    cv2.line(out, _px(TOPDOWN_X_CM[0], HOG_Y_CM), _px(TOPDOWN_X_CM[1], HOG_Y_CM), col, 1, cv2.LINE_AA)          # hogline
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
    malli, viivapts, ring_pts = sovita(frame, ellipses, H0, k1_vapaa=k1_vapaa)
    res = residuaalit(malli, ring_pts, viivapts)
    print("  residuaalit (cm RMS): " + ", ".join(f"{k} {v:.2f}" if v is not None else f"{k} -" for k, v in res.items()))
    td = topdown(frame, malli)
    cv2.imwrite(os.path.join(ulos, f"topdown_{puoli}.png"), td)
    cv2.imwrite(os.path.join(ulos, f"topdown_{puoli}_tarkistus.jpg"), piirra_overlay(td), [cv2.IMWRITE_JPEG_QUALITY, 90])
    with open(os.path.join(ulos, f"kalibrointi_{puoli}.json"), "w") as f:
        json.dump({"puoli": puoli, "rajaus_xywh": PUOLET[puoli], "kierto": "90 astetta vastapaivaan",
                   "H_kuva_fyysinen": malli.H.tolist(), "k1": malli.k1, "k1_skaala_px": K1_SCALE_PX,
                   "kuvan_keskipiste": malli.centre.tolist(), "hogline_y_cm": HOG_Y_CM, "residuaalit_cm_rms": res}, f, indent=1)
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
