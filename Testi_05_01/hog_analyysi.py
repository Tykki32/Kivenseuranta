"""Hog-hog -analyysi (Testi_05_01): heiton nopeus kaukaisella hoglinella, keskihidastuvuus ja hog-hog-aika toisen asteen sovituksesta.

Datapisteet: radan rivit joiden Y on valilla [lahihog + 50 cm, kaukohog - 100 cm]. Sovitus Y(t) = a t^2 + b t + c (t sekunteina valin ensimmaisesta pisteesta).
Ensin sovitus kaikille pisteille; sitten DROP_WORST (10) huonoiten sopivaa pistetta (suurin |residuaali|) suodatetaan pois ja sovitus tehdaan uudelleen (poistaa satunnaiset
epaonnistumiset). R lasketaan uudelleensovituksesta. Jos R = sqrt(R^2) > MIN_R (0.99):
  * nopeus kaukaisella hoglinella v = -dY/dt hetkella jolloin Y(t) = kaukohog (yhtalon mukaan),
  * keskihidastuvuus alueella = (v(alku) - v(loppu)) / (t_loppu - t_alku) (toisen asteen yhtalolle vakio 2a),
  * hog-hog-aika = t(Y = lahihog) - t(Y = kaukohog) yhtalon mukaan (lahihog 50 cm sovitusalueen ulkopuolella -> ekstrapolaatio).
Y pienenee kun kivi etenee kohti lahempaa pesaa (lahihog = pienempi Y)."""
import numpy as np
import cv2

MIN_R = 0.99
NEAR_MARGIN_CM = 50.0     # lahihog + 50 cm
FAR_MARGIN_CM = 100.0     # kaukohog - 100 cm
MIN_POINTS = 40
DROP_WORST = 10           # ensimmaisen sovituksen jalkeen pudotetaan 10 huonoiten sopivaa pistetta ja sovitetaan uudelleen
COVER_TOL_CM = 150.0      # radan pitaa kattaa valin paat (+-)


def _roots(a, b, c_minus_y):
    if abs(a) < 1e-9:
        return [-c_minus_y / b] if abs(b) > 1e-12 else []
    d = b * b - 4 * a * c_minus_y
    if d < 0:
        return []
    s = np.sqrt(d)
    return [(-b - s) / (2 * a), (-b + s) / (2 * a)]


def _time_at(a, b, c, y, t_lo, t_hi):
    """Hetki jolloin Y(t) = y: juuri jossa Y pienenee (dY/dt < 0) ja joka on lahinna havaintovalia [t_lo, t_hi]."""
    cands = [t for t in _roots(a, b, c - y) if 2 * a * t + b < 0]
    if not cands:
        return None
    mid = 0.5 * (t_lo + t_hi)
    return float(min(cands, key=lambda t: abs(t - mid)))


def _r_of(tt, y, a, b, c):
    ss_res = float(np.sum((y - (a * tt * tt + b * tt + c)) ** 2)); ss_tot = float(np.sum((y - y.mean()) ** 2))
    return float(np.sqrt(max(1.0 - ss_res / ss_tot, 0.0))) if ss_tot > 0 else 0.0


def analyze_hog(rows, near_hog_cm, far_hog_cm, min_r=MIN_R):
    """rows: [(frame, timestamp_s, {"Y_cm": ...}), ...]. Palauttaa dict: ok (bool), reason, n, R, ..."""
    ylo, yhi = near_hog_cm + NEAR_MARGIN_CM, far_hog_cm - FAR_MARGIN_CM
    pts = [(float(t), float(r["Y_cm"])) for _, t, r in rows if r.get("Y_cm") is not None and ylo <= float(r["Y_cm"]) <= yhi]
    out = dict(ok=False, reason="", n=len(pts), y_lo_cm=ylo, y_hi_cm=yhi, near_hog_cm=near_hog_cm, far_hog_cm=far_hog_cm)
    if len(pts) < MIN_POINTS:
        out["reason"] = f"liian vahan pisteita ({len(pts)} < {MIN_POINTS})"
        return out
    t = np.array([p[0] for p in pts]); y = np.array([p[1] for p in pts])
    if y.max() < yhi - COVER_TOL_CM or y.min() > ylo + COVER_TOL_CM:
        out["reason"] = "kivi ei ole kulkenut riittavasti (data ei kata sovitusvalia)"
        return out
    t0 = float(t.min()); tt = t - t0
    a, b, c = np.polyfit(tt, y, 2)
    R1 = _r_of(tt, y, a, b, c)
    n_drop = min(DROP_WORST, max(0, len(y) - MIN_POINTS))
    if n_drop > 0:                                   # pudota n_drop huonoiten sopivaa pistetta ja sovita uudelleen
        res = np.abs(y - (a * tt * tt + b * tt + c))
        keep = np.argsort(res)[: len(y) - n_drop]
        keep.sort()
        tt, y = tt[keep], y[keep]
        a, b, c = np.polyfit(tt, y, 2)
    pred = a * tt * tt + b * tt + c
    ss_res = float(np.sum((y - pred) ** 2)); ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    R = float(np.sqrt(max(r2, 0.0)))
    out.update(R=R, R_ennen_suodatusta=R1, n_pudotettu=n_drop, n_kaytetty=int(len(y)), r2=r2, a=float(a), b=float(b), c=float(c), t0=t0,
               rms_cm=float(np.sqrt(ss_res / len(y))))
    if R <= min_r:
        out["reason"] = f"sovitus ei riittavan hyva (R = {R:.4f} <= {min_r})"
        return out
    t_hi_y = _time_at(a, b, c, yhi, tt.min(), tt.max())   # alueen alku (kaukopaa)
    t_lo_y = _time_at(a, b, c, ylo, tt.min(), tt.max())   # alueen loppu (lahipaa)
    t_far = _time_at(a, b, c, far_hog_cm, tt.min(), tt.max())
    t_near = _time_at(a, b, c, near_hog_cm, tt.min(), tt.max())
    if None in (t_hi_y, t_lo_y, t_far, t_near):
        out["reason"] = "yhtalolla ei ratkaisua hoglinelle"
        return out
    v = lambda tx: -(2 * a * tx + b)                       # cm/s (positiivinen = kohti lahempaa pesaa)
    v_far = v(t_far)
    decel = (v(t_hi_y) - v(t_lo_y)) / (t_lo_y - t_hi_y) if t_lo_y != t_hi_y else 0.0
    out.update(ok=True, v_far_hog_ms=v_far / 100.0, decel_ms2=decel / 100.0,
               hog_hog_s=float(t_near - t_far), t_far_hog_s=float(t_far + t0), t_near_hog_s=float(t_near + t0),
               v_near_hog_ms=v(t_near) / 100.0)
    return out


def format_lines(res, stone_id=None):
    """Tekstirivit (terminaali + kuva)."""
    hdr = "Hog-hog" + (f" (kivi {stone_id})" if stone_id is not None else "")
    if not res.get("ok"):
        return [hdr, "ei laskettu: " + res.get("reason", "?")]
    return [hdr,
            f"nopeus kaukohogilla: {res['v_far_hog_ms']:.2f} m/s",
            f"keskihidastuvuus: {res['decel_ms2']:.3f} m/s^2",
            f"hog-hog aika: {res['hog_hog_s']:.2f} s",
            f"R = {res['R']:.5f}  n = {res['n_kaytetty']} (pudotettu {res['n_pudotettu']})"]


def draw_overlay(img, lines, K, R, t, near_hog_cm, x_side_cm=300.0, lane_half_cm=237.0, color=(255, 255, 255)):
    """Piirtaa tekstirivit radan SIVUUN lahemman hoglinen kohdalle (ja hoglinen ohuena viivana). Palauttaa img."""
    K = np.asarray(K, float); R = np.asarray(R, float).reshape(3, 3); t = np.asarray(t, float).reshape(3)
    def proj(x, y):
        p = K @ (R @ np.array([x, y, 0.0]) + t); return float(p[0] / p[2]), float(p[1] / p[2])
    H, W = img.shape[:2]
    pa, pb = proj(-lane_half_cm, near_hog_cm), proj(lane_half_cm, near_hog_cm)
    cv2.line(img, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), (0, 200, 255), 2)
    sides = [proj(x_side_cm, near_hog_cm), proj(-x_side_cm, near_hog_cm)]
    lh = 26; box_h = lh * len(lines) + 10
    # valitse sivu jolla teksti mahtuu kuvaan (ylempi ensin; teksti piirretaan ankkurista ylos/alas poispain radasta)
    anchor = None
    for p, direction in ((min(sides, key=lambda q: q[1]), -1), (max(sides, key=lambda q: q[1]), 1)):
        y0 = p[1] - box_h if direction < 0 else p[1]
        if 0 <= y0 and y0 + box_h <= H:
            anchor = (p[0], y0); break
    if anchor is None:
        anchor = (sides[0][0], max(0, min(H - box_h, sides[0][1] - box_h)))
    x0 = int(max(5, min(W - 430, anchor[0] - 200))); y0 = int(anchor[1])
    cv2.rectangle(img, (x0 - 6, y0), (x0 + 424, y0 + box_h), (0, 0, 0), -1)
    for i, s in enumerate(lines):
        cv2.putText(img, s, (x0, y0 + 22 + i * lh), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color if i else (0, 255, 255), 2)
    return img
