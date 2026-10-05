"""Hog-hog -analyysi: heiton nopeus kaukaisella hoglinella, hidastuvuus, hog-hog-aika, suunta, liuku ja irroitus.

Datapisteet: radan rivit joiden Y on valilla [lahihog + marginaali, kaukohog - marginaali]. Sovitus Y(t) = a t^2 + b t + c;
huonoimmin sopivat pisteet pudotetaan ja sovitetaan uudelleen. Jos R = sqrt(R^2) > HOG_MIN_R:
  * nopeus kaukaisella hoglinella v = -dY/dt hetkella jolloin Y(t) = kaukohog,
  * hidastuvuus kitkamallista (mu = A + B ln v) nopeudella HOG_HIDASTUVUUS_NOPEUDELLA_MS,
  * hog-hog-aika = t(Y = lahihog) - t(Y = kaukohog).
X-SUUNTA: kurvimalli (vakio sivukiihtyvyys) samoille pisteille -> suunta kaukohogilla ja IRROITUS = X, jonka kaukohogin
suunta jatkettuna saisi lahemmalla T-viivalla. LIUKU: suora X(Y) heiton alusta kaukohog + 1 m asti (+ hakki lahtopisteena,
erikseen X = +-15 cm) -> X lahemmalla T-viivalla. Tulos hyvaksytaan vain jos R_y > HOG_MIN_R, R_x > HOG_MIN_R ja
R_y * R_x > HOG_MIN_TULO. Y pienenee kun kivi etenee kohti lahempaa pesaa.
"""
import numpy as np
import cv2

import asetukset as A


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


# ------------------------------------------------------------------
# HIDASTUVUUS KITKAMALLISTA: kitkakerroin mu(v) = A + B ln(v) (v m/s), liikeyhtalo dv/dt = -g mu(v). Heiton Y(t)-dataan
# sovitetaan y0, v0 ja A (B kiintea, A.HOG_MU_B; Levenberg-Marquardt, numeerinen integrointi RK4).
# Raportoitava hidastuvuus = g mu(A.HOG_HIDASTUVUUS_NOPEUDELLA_MS).
# ------------------------------------------------------------------
G_MS2 = 9.81
_FRIC_DT = 0.04               # integrointiaskel (s)


def _fric_y(params, tt):
    """Mallin Y(t) (cm) annetuilla ajoilla. params = (y0_cm, v0_ms, A, B). None jos nopeus putoaa nollaan."""
    y0, v0, mu_a, mu_b = params
    if v0 <= 0.05:
        return None
    n = int(np.ceil(tt.max() / _FRIC_DT)) + 1
    h = _FRIC_DT
    ts = np.arange(n + 1) * h
    vs = np.empty(n + 1); vs[0] = v0
    f = lambda v: -G_MS2 * (mu_a + mu_b * np.log(v))
    v = v0
    for i in range(n):
        k1 = f(v); v2 = v + 0.5 * h * k1
        if v2 <= 0.01: return None
        k2 = f(v2); v3 = v + 0.5 * h * k2
        if v3 <= 0.01: return None
        k3 = f(v3); v4 = v + h * k3
        if v4 <= 0.01: return None
        k4 = f(v4)
        v = v + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        if v <= 0.01:
            return None
        vs[i + 1] = v
    pos = np.concatenate([[0.0], np.cumsum(0.5 * (vs[1:] + vs[:-1]) * h)])      # kuljettu matka (m)
    return y0 - 100.0 * np.interp(tt, ts, pos)


def fit_log_friction(tt, y, v0_ms, decel0_ms2, iters=40, fixed_B=None):
    """Palauttaa dict(A, B, y0_cm, v0_ms, rms_cm) tai None. tt: s (alkaen 0), y: cm. fixed_B: B kiinnitetty (sovitetaan y0, v0, A)."""
    nb = 0.0 if fixed_B is None else float(fixed_B)
    p = np.array([float(y[0]), float(v0_ms), max(1e-4, float(decel0_ms2) / G_MS2 - nb * np.log(max(v0_ms, 0.2))), nb])
    steps = np.array([0.5, 0.002, 1e-5, 1e-5])
    npar = 4 if fixed_B is None else 3

    def resid(q):
        m = _fric_y(q, tt)
        return None if m is None else m - y

    r = resid(p)
    if r is None:
        return None
    cost = float(r @ r); lam = 1e-3
    for _ in range(iters):
        J = np.empty((len(y), npar))
        for j in range(npar):
            q = p.copy(); q[j] += steps[j]
            rj = resid(q)
            if rj is None:
                return None
            J[:, j] = (rj - r) / steps[j]
        JtJ = J.T @ J; g = J.T @ r
        improved = False
        for _try in range(8):
            try:
                d = -np.linalg.solve(JtJ + lam * np.diag(np.diag(JtJ) + 1e-12), g)
            except np.linalg.LinAlgError:
                lam *= 10; continue
            if npar < 4:
                d = np.concatenate([d, [0.0]])
            rn = resid(p + d)
            if rn is not None and float(rn @ rn) < cost:
                p = p + d; r = rn; cost_new = float(rn @ rn)
                lam = max(lam / 3, 1e-9); improved = True
                break
            lam *= 10
        if not improved:
            break
        if abs(cost - cost_new) < 1e-6 * max(cost, 1e-9):
            cost = cost_new
            break
        cost = cost_new
    return dict(y0_cm=float(p[0]), v0_ms=float(p[1]), A=float(p[2]), B=float(p[3]), rms_cm=float(np.sqrt(cost / len(y))))


# ------------------------------------------------------------------
# KURVIMALLI X-SUUNTAAN: kitka hidastaa radan suuntaisesti, kurvi on vakiosuuruinen sivukiihtyvyys a_n = k kohtisuoraan
# kulkusuuntaa vastaan -> kulkusuunnan kulma theta' = k / |v|, X'(t) = v(t) theta(t) (pieni kulma), v(t) Y-sovituksesta.
# Parametrit x, alkukulma ja k. Mitattu virhe kaukohogilla (ekstrapoloitu): rms 1,6 cm (live 2026-10-04) / 0,7 cm (MAH00014),
# toisen asteen yhtalolla 2,7 / 3,8 cm (2. asteen tulos tallennetaan vertailuun: *_2aste).
# ------------------------------------------------------------------


def _ctz(g, f):
    return np.concatenate([[0.0], np.cumsum((f[1:] + f[:-1]) / 2.0 * np.diff(g))])


def curl_basis(a, b, t_lo, t_hi, n=3000):
    """Palauttaa (g, S, G, H): S = kuljettu matka (cm), G = int dt / v (s / (m/s)), H = int v G dt; v = -(2 a t + b) cm/s."""
    g = np.linspace(t_lo, t_hi, n)
    v = np.maximum(-(2.0 * a * g + b), 2.0)            # cm/s
    S = _ctz(g, v)
    G = _ctz(g, 1.0 / (v / 100.0))
    H = _ctz(g, v * G)
    return g, S, G, H


def _r_of(tt, y, a, b, c):
    ss_res = float(np.sum((y - (a * tt * tt + b * tt + c)) ** 2)); ss_tot = float(np.sum((y - y.mean()) ** 2))
    return float(np.sqrt(max(1.0 - ss_res / ss_tot, 0.0))) if ss_tot > 0 else 0.0


def analyze_hog(rows, near_hog_cm, far_hog_cm, min_r=A.HOG_MIN_R, tee_cm=None, min_product=A.HOG_MIN_TULO):
    """rows: [(frame, timestamp_s, {"Y_cm": ...}), ...]. Palauttaa dict: ok (bool), reason, n, R, ..."""
    ylo, yhi = near_hog_cm + A.HOG_LAHI_MARGINAALI_CM, far_hog_cm - A.HOG_KAUKO_MARGINAALI_CM
    pts = [(float(t), float(r["Y_cm"])) for _, t, r in rows if r.get("Y_cm") is not None and ylo <= float(r["Y_cm"]) <= yhi]
    out = dict(ok=False, reason="", n=len(pts), y_lo_cm=ylo, y_hi_cm=yhi, near_hog_cm=near_hog_cm, far_hog_cm=far_hog_cm)
    if len(pts) < A.HOG_MIN_PISTEET:
        out["reason"] = f"liian vahan pisteita ({len(pts)} < {A.HOG_MIN_PISTEET})"
        return out
    t = np.array([p[0] for p in pts]); y = np.array([p[1] for p in pts])
    if y.max() < yhi - A.HOG_KATTAVUUS_CM or y.min() > ylo + A.HOG_KATTAVUUS_CM:
        out["reason"] = "kivi ei ole kulkenut riittavasti (data ei kata sovitusvalia)"
        return out
    t0 = float(t.min()); tt = t - t0
    a, b, c = np.polyfit(tt, y, 2)
    R1 = _r_of(tt, y, a, b, c)
    n_drop = min(A.HOG_PUDOTA_HUONOIMMAT, max(0, len(y) - A.HOG_MIN_PISTEET))
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
    # hidastuvuus kitkamallista (keskimaarainen 2a talteen vertailuun)
    decel_avg = decel / 100.0
    decel_out = decel_avg
    fr = fit_log_friction(tt, y, -b / 100.0, decel_avg, fixed_B=A.HOG_MU_B)
    if fr is not None and fr["rms_cm"] <= 1.5 * out["rms_cm"] + 0.5:
        decel_out = G_MS2 * (fr["A"] + fr["B"] * np.log(A.HOG_HIDASTUVUUS_NOPEUDELLA_MS))
        out.update(mu_a=fr["A"], mu_b=fr["B"], kitka_rms_cm=fr["rms_cm"])
    else:
        out.update(mu_a=float("nan"), mu_b=float("nan"), kitka_rms_cm=float("nan"))
    out.update(ok_y=True, v_far_hog_ms=v_far / 100.0, decel_ms2=float(decel_out), decel_keskim_ms2=float(decel_avg),
               hog_hog_s=float(t_near - t_far), t_far_hog_s=float(t_far + t0), t_near_hog_s=float(t_near + t0),
               v_near_hog_ms=v(t_near) / 100.0)

    # ---- X-suuntainen analyysi ----
    pts_x = [(float(t_), float(r["Y_cm"]), float(r["X_cm"])) for _, t_, r in rows
             if r.get("Y_cm") is not None and r.get("X_cm") is not None and ylo <= float(r["Y_cm"]) <= yhi]
    if len(pts_x) < A.HOG_MIN_PISTEET:
        out["ok"] = False; out["reason"] = "X-pisteita liian vahan"
        return out
    tx = np.array([p[0] for p in pts_x]) - t0
    xx = np.array([p[2] for p in pts_x])
    y_eq = a * tx * tx + b * tx + c                         # Y yhtalosta ajan funktiona
    u = (y_eq - far_hog_cm) / 100.0
    px, qx, rx = np.polyfit(u, xx, 2)
    Rx1 = _r_of(u, xx, px, qx, rx)
    nd = min(A.HOG_PUDOTA_HUONOIMMAT, max(0, len(xx) - A.HOG_MIN_PISTEET))
    if nd > 0:
        resx = np.abs(xx - (px * u * u + qx * u + rx))
        kx = np.sort(np.argsort(resx)[: len(xx) - nd])
        u, xx = u[kx], xx[kx]
        px, qx, rx = np.polyfit(u, xx, 2)
    Rx = _r_of(u, xx, px, qx, rx)
    out.update(R_x=Rx, R_x_ennen_suodatusta=Rx1, nx_kaytetty=int(len(xx)), nx_pudotettu=nd, px=float(px), qx=float(qx), rx=float(rx), R_tulo=R * Rx)
    if Rx <= min_r or R * Rx <= min_product:
        out["ok"] = False
        out["reason"] = f"X-sovitus tai R_y*R_x ei riittava (R_x = {Rx:.4f}, R_y*R_x = {R * Rx:.4f})"
        return out
    slope = qx / 100.0                                      # dX/dY kaukohoglinella (u = 0), cm / cm
    x_far = float(rx)
    dir_deg = float(np.degrees(np.arctan(-slope)))          # kulkusuunta (Y pienenee): + = kohti +X
    out.update(x_far_hog_cm_2aste=x_far, dir_far_hog_deg_2aste=dir_deg)
    # kurvimalli: X(t) = x0 + theta0 S(t) + k H(t); sama pistejoukko ja huonoimmat pois
    txa = np.array([p_[0] for p_ in pts_x]) - t0
    xxa = np.array([p_[2] for p_ in pts_x])
    g, S, G, Hh = curl_basis(a, b, min(txa.min(), t_far) - 0.05, txa.max() + 0.05)
    Mx = np.column_stack([np.ones_like(txa), np.interp(txa, g, S), np.interp(txa, g, Hh)])
    co, *_ = np.linalg.lstsq(Mx, xxa, rcond=None)
    if nd > 0:
        kk = np.sort(np.argsort(np.abs(xxa - Mx @ co))[: len(xxa) - nd])
        co, *_ = np.linalg.lstsq(Mx[kk], xxa[kk], rcond=None)
    x_far = float(co[0] + co[1] * np.interp(t_far, g, S) + co[2] * np.interp(t_far, g, Hh))
    theta_far = float(co[1] + co[2] * np.interp(t_far, g, G))
    slope = -theta_far                                  # dX/dY = X'/Y' = v theta / (-v)
    dir_deg = float(np.degrees(np.arctan(theta_far)))
    out.update(kurvi_k_ms2=float(co[2]), kurvi_rms_cm=float(np.sqrt(np.mean((xxa[kk] - Mx[kk] @ co) ** 2))) if nd > 0
               else float(np.sqrt(np.mean((xxa - Mx @ co) ** 2))))
    out.update(x_far_hog_cm=x_far, slope_dxdy=float(slope), dir_far_hog_deg=dir_deg)
    if tee_cm is not None:
        out["tee_y_cm"] = float(tee_cm)
        out["x_straight_at_tee_cm"] = float(x_far + slope * (tee_cm - far_hog_cm))
    # ---- LIUKU: suora X(Y) heiton alusta kaukohog + marginaaliin asti (+ hakki lahtopisteena) -> X lahemmalla T-viivalla ----
    if tee_cm is not None:
        pts_l = [(float(r["Y_cm"]), float(r["X_cm"])) for _, _, r in rows
                 if r.get("Y_cm") is not None and r.get("X_cm") is not None and float(r["Y_cm"]) >= far_hog_cm + A.LIUKU_LOPPU_MARGINAALI_CM]
        if len(pts_l) >= A.LIUKU_MIN_PISTEET:
            yl = np.array([p[0] for p in pts_l]); xl = np.array([p[1] for p in pts_l])
            sl, ic = np.polyfit(yl, xl, 1)
            out.update(liuku_x_tee_cm=float(sl * tee_cm + ic), liuku_dir_deg=float(np.degrees(np.arctan(-sl))), liuku_n=int(len(yl)),
                       liuku_rms_cm=float(np.std(xl - (sl * yl + ic))))
            hakki_y = far_hog_cm + A.HOG_TEESTA_CM + A.TAKARAJA_TEESTA_CM + A.HAKKI_TAKARAJASTA_CM
            out["hakki_y_cm"] = float(hakki_y)
            for tag, hx in (("p", A.HAKKI_SIVU_CM), ("m", -A.HAKKI_SIVU_CM)):
                sh, ih = np.polyfit(np.append(yl, hakki_y), np.append(xl, hx), 1)
                out[f"liuku_x_tee_cm_hakki_{tag}"] = float(sh * tee_cm + ih)
                out[f"liuku_dir_deg_hakki_{tag}"] = float(np.degrees(np.arctan(-sh)))
    out["ok"] = True
    return out


def format_lines(res, stone_id=None):
    """Tekstirivit (terminaali + kuva). Vain onnistuneelle (R_y, R_x, R_y*R_x) analyysille; muuten None."""
    if not res.get("ok"):
        return None
    hdr = "Hog-hog + X-suunta" + (f" (kivi {stone_id})" if stone_id is not None else "")
    d = res["dir_far_hog_deg"]
    side = "+X" if d > 0 else "-X"
    liuku = f"{res['liuku_x_tee_cm']:+.1f} cm (suunta {res['liuku_dir_deg']:+.2f} deg, n = {res['liuku_n']})" if "liuku_x_tee_cm" in res else "ei laskettu"
    if "liuku_x_tee_cm_hakki_p" in res:
        liuku += (f" | hakista X=+{A.HAKKI_SIVU_CM:.0f}: {res['liuku_x_tee_cm_hakki_p']:+.1f} cm, "
                  f"X=-{A.HAKKI_SIVU_CM:.0f}: {res['liuku_x_tee_cm_hakki_m']:+.1f} cm")
    lines = [hdr,
             f"nopeus kaukohogilla: {res['v_far_hog_ms']:.2f} m/s",
             f"hidastuvuus ({A.HOG_HIDASTUVUUS_NOPEUDELLA_MS:g} m/s): {res['decel_ms2']:.3f} m/s^2",
             f"hog-hog aika: {res['hog_hog_s']:.2f} s",
             f"suunta kaukohogilla: {abs(d):.2f} deg kohti {side}",
             f"liuku (suoran X T-viivalla, alusta hog+1m): {liuku}"]
    if "x_straight_at_tee_cm" in res:
        lines.append(f"merkki (suoran X T-viivalla, hogin jalkeen): {res['x_straight_at_tee_cm']:+.1f} cm")
    if res.get("kierteet") is not None:
        lines.append(f"kierteita hog-hog: {res['kierteet']:.1f} (kierrosaika {res['kierrosaika_far_s']:.1f} s -> {res['kierrosaika_near_s']:.1f} s)")
    if res.get("kierteet_taysi") is not None:
        lines.append(f"kierteita hog-hog (taysi resoluutio): {res['kierteet_taysi']:.1f} (kierrosaika {res['kierrosaika_far_s_taysi']:.1f} s -> "
                     f"{res['kierrosaika_near_s_taysi']:.1f} s)")
    lines.append(f"R_y = {res['R']:.5f}  R_x = {res['R_x']:.5f}  tulo = {res['R_tulo']:.5f}")
    return lines


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
    x0 = int(max(5, min(W - 570, anchor[0] - 200))); y0 = int(anchor[1])
    cv2.rectangle(img, (x0 - 6, y0), (x0 + 560, y0 + box_h), (0, 0, 0), -1)
    for i, s in enumerate(lines):
        cv2.putText(img, s, (x0, y0 + 22 + i * lh), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color if i else (0, 255, 255), 2)
    return img
