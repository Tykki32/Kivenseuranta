"""Hog-hog -analyysi (Testi_05_01/05_02) + X-suuntainen analyysi: heiton nopeus kaukaisella hoglinella, keskihidastuvuus ja hog-hog-aika toisen asteen sovituksesta.

Datapisteet: radan rivit joiden Y on valilla [lahihog + 50 cm, kaukohog - 100 cm]. Sovitus Y(t) = a t^2 + b t + c (t sekunteina valin ensimmaisesta pisteesta).
Ensin sovitus kaikille pisteille; sitten DROP_WORST (10) huonoiten sopivaa pistetta (suurin |residuaali|) suodatetaan pois ja sovitus tehdaan uudelleen (poistaa satunnaiset
epaonnistumiset). R lasketaan uudelleensovituksesta. Jos R = sqrt(R^2) > MIN_R (0.99):
  * nopeus kaukaisella hoglinella v = -dY/dt hetkella jolloin Y(t) = kaukohog (yhtalon mukaan),
  * keskihidastuvuus alueella = (v(alku) - v(loppu)) / (t_loppu - t_alku) (toisen asteen yhtalolle vakio 2a),
  * hog-hog-aika = t(Y = lahihog) - t(Y = kaukohog) yhtalon mukaan (lahihog 50 cm sovitusalueen ulkopuolella -> ekstrapolaatio).
Y pienenee kun kivi etenee kohti lahempaa pesaa (lahihog = pienempi Y).

X-SUUNTAINEN ANALYYSI: samoille pisteille lasketaan Y_yht = Y:n yhtalo hetkella t (sileä Y), ja X sovitetaan toisen asteen yhtaloon Y_yht:n suhteen: X = p u^2 + q u + r, u = (Y_yht - kaukohog)/100
(taas 10 huonoiten sopivaa pistetta pois + uudelleensovitus). Jos R_x > 0.99: suunta kaukohoglinella (dX/dY kaukohogilla -> kulma kulkusuunnassa, + = kohti +X) ja X-arvo jonka
SUORA (kaukohogin suunta jatkettuna) saisi lahemmalla T-viivalla (Y = lahipesan keskipiste). TULOSTETAAN VAIN jos R_y * R_x > MIN_PRODUCT (0.99) (ja kumpikin R > 0.99): suodattaa kaiken huonon."""
import numpy as np
import cv2

MIN_R = 0.99
MIN_PRODUCT = 0.99        # R_y * R_x -ehto tulostukselle
NEAR_MARGIN_CM = 50.0     # lahihog + 50 cm
FAR_MARGIN_CM = 100.0     # kaukohog - 100 cm
MIN_POINTS = 40
LIUKU_END_MARGIN_CM = 100.0  # liuku: suoran sovitus heiton alusta kaukohog + 100 cm asti
LIUKU_MIN_POINTS = 8
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


# ------------------------------------------------------------------
# Testi_07_01 v7.3: HIDASTUVUUS KITKAMALLISTA. Kitkakerroin mu(v) = A + B ln(v) (v = hetkellinen nopeus m/s, havaittu muoto),
# liikeyhtalo dv/dt = -g mu(v). Sovitetaan heiton Y(t)-dataan (samat pisteet kuin toisen asteen sovituksessa) parametrit
# y0, v0, A (v7.8: B kiintea MU_B) (Levenberg-Marquardt, numeerinen integrointi RK4). Raportoitava hidastuvuus = g mu(DECEL_REF_V_MS).
# ------------------------------------------------------------------
G_MS2 = 9.81
DECEL_REF_V_MS = 1.5          # hidastuvuus ilmoitetaan talla nopeudella
_FRIC_DT = 0.04               # integrointiaskel (s)
# v7.8: B KIINTEA, KOVAKOODATTU VAKIO (kayttajan paatos: B on oletettavasti lahes vakio). Jokaiselle heitolle sovitetaan vain A
# (+ alkupaikka ja -nopeus). HUOM mittaukset (yhteissovitus, B yhteinen, A heittokohtainen):
#   live 2026-10-04 (66 heittoa, myos lyonnit): paras B = -0,0011; MAH00014 (19 painoheittoa): paras B = -0,0027.
#   B = -0,001 -> hidastuvuus @1,5 m/s: hitaat heitot ero <= 0,4 % (live) / <= 3,9 % (MAH, sovitusvirhe 2,39 -> 2,51 cm);
#   lyonnit (live) <= 1,2 %. Lyonneilla 1,5 m/s on ekstrapolointia: jos todellinen B olisi MAH:n -0,0027, 3,3 m/s lyonnin
#   hidastuvuus olisi n. 0,013 m/s^2 (~18 %) liian pieni. Jos jaa/kalibrointi muuttaa B:ta selvasti, arvo paivitetaan tahan.
MU_B = -0.001


def _fric_y(params, tt):
    """Mallin Y(t) (cm) annetuilla ajoilla. params = (y0_cm, v0_ms, A, B). None jos nopeus putoaa nollaan."""
    y0, v0, A, B = params
    if v0 <= 0.05:
        return None
    n = int(np.ceil(tt.max() / _FRIC_DT)) + 1
    h = _FRIC_DT
    ts = np.arange(n + 1) * h
    vs = np.empty(n + 1); vs[0] = v0
    f = lambda v: -G_MS2 * (A + B * np.log(v))
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
# Testi_08_01 v8.4: KURVIMALLI X-SUUNTAAN. Aiempi X = p u^2 + q u + r (u = Y) vastaa vakiokaarevuutta (ympyranKaari) eika
# ennustanut kaukohogia hyvin (sovitusalue paattyy 1 m ennen sita). Uusi fysikaalinen malli radan omissa koordinaateissa:
# kitka hidastaa radan suuntaisesti, kurvi on vakiosuuruinen sivukiihtyvyys a_n = k kohtisuoraan kulkusuuntaa vastaan ->
# kulkusuunnan kulma theta' = k / |v|, X'(t) = v(t) theta(t) (pieni kulma), v(t) Y-sovituksesta. Kolme parametria kuten ennen
# (x, alkukulma, k). Mitattu (sama sovitusalue, kaukohog ekstrapoloitu): virhe kaukohogilla rms 2,7 -> 1,6 cm (live
# 2026-10-04, 64 heittoa) ja 3,8 -> 0,7 cm (MAH00014). k ~ 0,009 m/s^2, hajonta heittojen valilla ~20 %.
# KURVIMALLI=0 = vanha toisen asteen yhtalo.
# ------------------------------------------------------------------
import os as _os_km
KURVIMALLI = _os_km.environ.get("KURVIMALLI", "1") == "1"


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


def analyze_hog(rows, near_hog_cm, far_hog_cm, min_r=MIN_R, tee_cm=None, min_product=MIN_PRODUCT):
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
    # Testi_07_01 v7.3: hidastuvuus kitkamallista mu(v) = A + B ln v nopeudella DECEL_REF_V_MS (vanha keskiarvo talteen vertailuun)
    decel_avg = decel / 100.0
    decel_out = decel_avg
    fr = fit_log_friction(tt, y, -b / 100.0, decel_avg, fixed_B=MU_B)       # v7.8: B kiintea (MU_B), sovitetaan A
    if fr is not None and fr["rms_cm"] <= 1.5 * out["rms_cm"] + 0.5:
        decel_out = G_MS2 * (fr["A"] + fr["B"] * np.log(DECEL_REF_V_MS))
        out.update(mu_a=fr["A"], mu_b=fr["B"], kitka_rms_cm=fr["rms_cm"])
    else:
        out.update(mu_a=float("nan"), mu_b=float("nan"), kitka_rms_cm=float("nan"))
    out.update(ok_y=True, v_far_hog_ms=v_far / 100.0, decel_ms2=float(decel_out), decel_keskim_ms2=float(decel_avg),
               hog_hog_s=float(t_near - t_far), t_far_hog_s=float(t_far + t0), t_near_hog_s=float(t_near + t0),
               v_near_hog_ms=v(t_near) / 100.0)

    # ---- X-suuntainen analyysi ----
    pts_x = [(float(t_), float(r["Y_cm"]), float(r["X_cm"])) for _, t_, r in rows
             if r.get("Y_cm") is not None and r.get("X_cm") is not None and ylo <= float(r["Y_cm"]) <= yhi]
    if len(pts_x) < MIN_POINTS:
        out["ok"] = False; out["reason"] = "X-pisteita liian vahan"
        return out
    tx = np.array([p[0] for p in pts_x]) - t0
    xx = np.array([p[2] for p in pts_x])
    y_eq = a * tx * tx + b * tx + c                         # Y yhtalosta ajan funktiona
    u = (y_eq - far_hog_cm) / 100.0
    px, qx, rx = np.polyfit(u, xx, 2)
    Rx1 = _r_of(u, xx, px, qx, rx)
    nd = min(DROP_WORST, max(0, len(xx) - MIN_POINTS))
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
    if KURVIMALLI:
        # v8.4: kurvimalli (katso KURVIMALLI): X(t) = x0 + theta0 S(t) + k H(t); sama pistejoukko ja 10 huonointa pois
        txa = np.array([p_[0] for p_ in pts_x]) - t0
        xxa = np.array([p_[2] for p_ in pts_x])
        g, S, G, Hh = curl_basis(a, b, min(txa.min(), t_far) - 0.05, txa.max() + 0.05)
        A = np.column_stack([np.ones_like(txa), np.interp(txa, g, S), np.interp(txa, g, Hh)])
        co, *_ = np.linalg.lstsq(A, xxa, rcond=None)
        if nd > 0:
            kk = np.sort(np.argsort(np.abs(xxa - A @ co))[: len(xxa) - nd])
            co, *_ = np.linalg.lstsq(A[kk], xxa[kk], rcond=None)
        x_far = float(co[0] + co[1] * np.interp(t_far, g, S) + co[2] * np.interp(t_far, g, Hh))
        theta_far = float(co[1] + co[2] * np.interp(t_far, g, G))
        slope = -theta_far                                  # dX/dY = X'/Y' = v theta / (-v)
        dir_deg = float(np.degrees(np.arctan(theta_far)))
        out.update(kurvi_k_ms2=float(co[2]), kurvi_rms_cm=float(np.sqrt(np.mean((xxa[kk] - A[kk] @ co) ** 2))) if nd > 0
                   else float(np.sqrt(np.mean((xxa - A @ co) ** 2))))
    out.update(x_far_hog_cm=x_far, slope_dxdy=float(slope), dir_far_hog_deg=dir_deg)
    if tee_cm is not None:
        out["tee_y_cm"] = float(tee_cm)
        out["x_straight_at_tee_cm"] = float(x_far + slope * (tee_cm - far_hog_cm))
    # ---- LIUKU: suora X(Y) heiton alusta kohtaan kaukohog + LIUKU_END_MARGIN_CM (kivi ei ole viela ylittanyt hogia) -> mihin se osuisi lahemmalla T-viivalla ----
    if tee_cm is not None:
        pts_l = [(float(r["Y_cm"]), float(r["X_cm"])) for _, _, r in rows
                 if r.get("Y_cm") is not None and r.get("X_cm") is not None and float(r["Y_cm"]) >= far_hog_cm + LIUKU_END_MARGIN_CM]
        if len(pts_l) >= LIUKU_MIN_POINTS:
            yl = np.array([p[0] for p in pts_l]); xl = np.array([p[1] for p in pts_l])
            sl, ic = np.polyfit(yl, xl, 1)
            out.update(liuku_x_tee_cm=float(sl * tee_cm + ic), liuku_dir_deg=float(np.degrees(np.arctan(-sl))), liuku_n=int(len(yl)),
                       liuku_rms_cm=float(np.std(xl - (sl * yl + ic))))
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
    lines = [hdr,
             f"nopeus kaukohogilla: {res['v_far_hog_ms']:.2f} m/s",
             f"hidastuvuus ({DECEL_REF_V_MS:g} m/s): {res['decel_ms2']:.3f} m/s^2",
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


# ============================================================
# DEBUG-VIDEON UUSI ULKOASU: kaannetty 90 astetta vastapaivaan (kivet kulkevat ylhaalta alas), korkeus DEBUG_H (1080), vasemmalla ja oikealla
# tietopaneelit heitoista jotka lahtivat vasemmalle / oikealle (ylimpana viimeisin, vanhemmat rullaavat alas kunnes eivat mahdu).
# ============================================================
DEBUG_H = 1080
PANEL_W = 430
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6          # sama fontti kuin edellisessa (hog-hog -tekstissa)
FONT_THICK = 2
LINE_H = 26
BOX_PAD = 8
BOX_GAP = 8


def debug_layout(src_w, src_h):
    """Kaannetyn videon koko: (video_w, video_h, scale, total_w). Kaannos 90 astetta -> leveys = src_h, korkeus = src_w; skaalataan korkeus DEBUG_H:hon."""
    scale = DEBUG_H / float(src_w)
    video_w = int(round(src_h * scale))
    return video_w, DEBUG_H, scale, video_w + 2 * PANEL_W


def plus_x_is_right(K, R, t, y_cm=1500.0):
    """True jos +X on kaannetyssa (90 astetta vastapaivaan) videossa OIKEALLA. Kaannos: x' = y_alkuperainen."""
    K = np.asarray(K, float); R = np.asarray(R, float).reshape(3, 3); t = np.asarray(t, float).reshape(3)
    def py(x):
        p = K @ (R @ np.array([x, y_cm, 0.0]) + t); return p[1] / p[2]
    return py(100.0) > py(-100.0)


def throw_side(res, plus_right):
    """'L' tai 'R': mille puolelle (kaannetyssa videossa) heitto lahti kaukohoglinella (suunta dir_far_hog_deg: + = kohti +X)."""
    toward_plus = res["dir_far_hog_deg"] > 0
    return "R" if toward_plus == bool(plus_right) else "L"


def entry_lines(res):
    return [f"kiven ID: {res['stone_id']}",
            f"nopeus: {res['v_far_hog_ms']:.2f} m/s",
            f"hidastuvuus: {res['decel_ms2']:.3f} m/s^2",
            f"hog-hog: {res['hog_hog_s']:.2f} s",
            f"liuku: {res['liuku_x_tee_cm']:+.1f} cm" if "liuku_x_tee_cm" in res else "liuku: -",
            f"merkki: {res.get('x_straight_at_tee_cm', float('nan')):+.1f} cm",
            f"kierteita: {res['kierteet']:.1f}" if res.get("kierteet") is not None else "kierteita: -"]


def entry_age_s(res, now_video_s=None):
    """Testi_07_01 v7.3: montako sekuntia sitten kivi ylitti kaukohoglinen. Live: seinakello (t_far_wall) vs nyt;
    tiedosto: videoaika (t_far_hog_s) vs kasiteltava ruutu. None jos ei tiedossa."""
    import time as _t
    if res.get("t_far_wall") is not None:
        return max(0, int(_t.time() - res["t_far_wall"]))
    if now_video_s is not None and res.get("t_far_hog_s") is not None:
        return max(0, int(now_video_s - res["t_far_hog_s"]))
    return None


def render_panel(entries, ages=None):
    """entries: lista dict-tuloksia, UUSIN ENSIMMAISENA. Palauttaa (DEBUG_H x PANEL_W) kuvan; jokainen heitto omassa laatikossa.
    ages: sekunnit kaukohoglinen ylityksesta (sama jarjestys), naytetaan kiven ID:n vieressa."""
    img = np.zeros((DEBUG_H, PANEL_W, 3), np.uint8)
    box_h = LINE_H * 7 + 2 * BOX_PAD
    y = BOX_GAP
    for k, res in enumerate(entries):
        if y + box_h > DEBUG_H:
            break
        cv2.rectangle(img, (BOX_GAP, y), (PANEL_W - BOX_GAP, y + box_h), (45, 45, 45), -1)
        cv2.rectangle(img, (BOX_GAP, y), (PANEL_W - BOX_GAP, y + box_h), (0, 200, 255), 2)
        for i, s in enumerate(entry_lines(res)):
            cv2.putText(img, s, (BOX_GAP + BOX_PAD + 4, y + BOX_PAD + 20 + i * LINE_H), FONT, FONT_SCALE,
                        (0, 255, 255) if i == 0 else (255, 255, 255), FONT_THICK)
        if ages is not None and k < len(ages) and ages[k] is not None:
            txt = f"{ages[k]} s"
            (tw, _), _ = cv2.getTextSize(txt, FONT, FONT_SCALE, FONT_THICK)
            cv2.putText(img, txt, (PANEL_W - BOX_GAP - BOX_PAD - 4 - tw, y + BOX_PAD + 20), FONT, FONT_SCALE,
                        (0, 255, 255), FONT_THICK)
        y += box_h + BOX_GAP
    return img


def compose_debug_frame(base_bgr, labels, header, results, plus_right):
    """base_bgr: alkuperainen (stabiloitu+korjattu, piirretyt ääriviivat) kuva; labels: [(x, y, teksti, vari_bgr)] alkuperaisen kuvan pikselikoordinaateissa.
    Palauttaa kaannetyn+skaalatun kuvan paneeleineen: [vasen paneeli | video | oikea paneeli]."""
    H0, W0 = base_bgr.shape[:2]
    video_w, video_h, scale, _ = debug_layout(W0, H0)
    rot = cv2.rotate(base_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    vid = cv2.resize(rot, (video_w, video_h), interpolation=cv2.INTER_AREA)
    for (x, y, txt, col) in labels:                      # kaannos: (x, y) -> (y, W0 - 1 - x), sitten skaalaus
        xr, yr = int(round(y * scale)), int(round((W0 - 1 - x) * scale))
        cv2.putText(vid, txt, (max(2, min(video_w - 120, xr)), max(14, min(video_h - 4, yr))), FONT, FONT_SCALE, col, FONT_THICK)
    cv2.putText(vid, header, (10, 28), FONT, 0.6, (255, 255, 255), 2)
    left = [r for r in results if throw_side(r, plus_right) == "L"][::-1]
    right = [r for r in results if throw_side(r, plus_right) == "R"][::-1]
    return np.hstack([render_panel(left), vid, render_panel(right)])


# ============================================================
# NOPEA DEBUG-VIDEON KOOSTAJA (Testi_05_02): pysyva canvas, paneelit piirretaan uudelleen vain kun sisalto muuttuu, skaalaus ENNEN kaantoa (pienempi kuva),
# ja AsyncVideoWriter (mp4-enkoodaus omassa saikeessa; cv2.VideoWriter.write vapauttaa GIL:n).
# ============================================================
import threading
import queue as _queue


class DebugComposer:
    def __init__(self, src_w, src_h):
        self.src_w, self.src_h = int(src_w), int(src_h)
        self.video_w, self.video_h, self.scale, self.total_w = debug_layout(src_w, src_h)
        self.canvas = np.zeros((self.video_h, self.total_w, 3), np.uint8)
        self._key = {"L": None, "R": None}
        # skaalaus ennen kaantoa: (src_w x src_h) -> (video_h x video_w) = (DEBUG_H x video_w) kaantamattomana: leveys DEBUG_H, korkeus video_w
        self._pre_w, self._pre_h = DEBUG_H, self.video_w

    def compose(self, base_bgr, labels, header, results, plus_right, now_video_s=None):
        H0, W0 = base_bgr.shape[:2]
        small = cv2.resize(base_bgr, (self._pre_w, self._pre_h), interpolation=cv2.INTER_LINEAR)     # 1080 x 608
        vid = cv2.rotate(small, cv2.ROTATE_90_COUNTERCLOCKWISE)                                   # 608 x 1080 (leveys x korkeus)
        scale = self.scale
        for (x, y, txt, col) in labels:
            xr, yr = int(round(y * scale)), int(round((W0 - 1 - x) * scale))
            cv2.putText(vid, txt, (max(2, min(self.video_w - 120, xr)), max(14, min(self.video_h - 4, yr))), FONT, FONT_SCALE, col, FONT_THICK)
        cv2.putText(vid, header, (10, 28), FONT, 0.6, (255, 255, 255), 2)
        self.canvas[:, PANEL_W:PANEL_W + self.video_w] = vid
        for side, x0 in (("L", 0), ("R", PANEL_W + self.video_w)):
            ents = [r for r in results if throw_side(r, plus_right) == side][::-1]
            ages = [entry_age_s(r, now_video_s) for r in ents]
            key = tuple((r["stone_id"], r["frame"], a) for r, a in zip(ents, ages))
            if key != self._key[side]:                         # paneeli piirretaan uudelleen vain kun sisalto muuttui (sekuntilaskuri: kerran sekunnissa)
                self.canvas[:, x0:x0 + PANEL_W] = render_panel(ents, ages)
                self._key[side] = key
        return self.canvas


# ------------------------------------------------------------------
# Debug-videon koodaus laitteistolla (Intel Quick Sync, h264_qsv) ffmpeg-putken kautta: koodaus siirtyy CPU:lta grafiikkapiirin
# media-moottorille. DEBUG_ENCODER = auto (qsv -> opencv mp4v) | qsv | x264 | opencv. Pelkka debug-video, ei vaikuta seurantaan.
# ------------------------------------------------------------------
import os as _os
import subprocess as _sp

_ENCODER_ARGS = {
    "qsv": ["-vf", "format=nv12", "-c:v", "h264_qsv", "-global_quality", "26", "-look_ahead", "0", "-preset", "veryfast"],
    "x264": ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "23", "-pix_fmt", "yuv420p"],
}


def _yuv_input(w, h):
    """v5.8: DEBUG_YUV=1 -> ruudut muunnetaan BGR -> YUV 4:2:0 (I420) tassa prosessissa cv2.cvtColorilla (SIMD, nopea) ja ffmpegille
    syotetaan valmis yuv420p: ffmpegin hitaampi BGR-muunnos (swscale) jaa pois ja putkeen menee puolet vahemman tavuja.
    Varit voivat poiketa aavistuksen (eri muunnoskaava/krominanssin naytteistys). Vain debug-video, ei vaikuta seurantaan.
    Vaatii parilliset mitat."""
    return _os.environ.get("DEBUG_YUV", "1") == "1" and int(w) % 2 == 0 and int(h) % 2 == 0


def _ffmpeg_cmd(w, h, fps, encoder, out_path):
    pix = "yuv420p" if _yuv_input(w, h) else "bgr24"
    return (["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", pix,
             "-s", f"{int(w)}x{int(h)}", "-r", f"{float(fps):.4f}", "-i", "-", "-an"] + _ENCODER_ARGS[encoder] + [out_path])


def probe_encoder(encoder, w, h, fps):
    """Kokeilee koodausta 3 mustalla ruudulla (-f null). Palauttaa (ok, virheteksti)."""
    cmd = _ffmpeg_cmd(w, h, fps, encoder, "-")
    cmd = cmd[:-1] + ["-f", "null", "-"]
    try:
        frame_bytes = int(w) * int(h) * 3 // 2 if _yuv_input(w, h) else int(w) * int(h) * 3
        p = _sp.run(cmd, input=bytes(frame_bytes * 3), stdout=_sp.PIPE, stderr=_sp.PIPE, timeout=30)
        return p.returncode == 0, p.stderr.decode("utf-8", "replace").strip()[-300:]
    except Exception as e:                      # ffmpeg puuttuu / aikakatkaisu
        return False, repr(e)


class FfmpegPipeWriter:
    """cv2.VideoWriter-yhteensopiva (write/release): raakaruudut ffmpeg-prosessin stdiniin."""

    def __init__(self, path, fps, size, encoder):
        self._dead = False
        self._yuv = _yuv_input(size[0], size[1])
        self._p = _sp.Popen(_ffmpeg_cmd(size[0], size[1], fps, encoder, path), stdin=_sp.PIPE, stdout=_sp.DEVNULL, stderr=_sp.DEVNULL,
                            **bg_popen_kwargs())

    def write(self, img):
        if self._dead:
            return
        try:
            if self._yuv:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV_I420)
            self._p.stdin.write(np.ascontiguousarray(img).data)
        except Exception as e:                  # ffmpeg kuoli -> ei kaadeta seurantaa
            self._dead = True
            print(f"VAROITUS: debug-videon ffmpeg-koodaus keskeytyi ({e!r}); loput ruudut jatetaan kirjoittamatta.")

    def release(self):
        try:
            self._p.stdin.close()
        except Exception:
            pass
        self._p.wait()


def open_debug_writer(path, fps, size):
    """Valitsee debug-videon kirjoittajan: laitteisto-QSV jos kaytettavissa, muuten cv2.VideoWriter (mp4v)."""
    mode = _os.environ.get("DEBUG_ENCODER", "auto").lower()
    order = {"auto": ["qsv"], "qsv": ["qsv"], "x264": ["x264"], "opencv": []}.get(mode, ["qsv"])
    for enc in order:
        ok, err = probe_encoder(enc, size[0], size[1], fps)
        if ok:
            print(f"Debug-video: koodaus ffmpeg/{enc} ({'Intel Quick Sync, grafiikkapiiri' if enc == 'qsv' else 'CPU'})"
                  f"{', syote YUV 4:2:0 (DEBUG_YUV=1)' if _yuv_input(size[0], size[1]) else ''}")
            return FfmpegPipeWriter(path, fps, size, enc)
        print(f"Debug-video: ffmpeg/{enc} ei kaytettavissa ({err or 'tuntematon virhe'}) -> cv2.VideoWriter (mp4v, CPU)")
    return cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)


# v5.7: taustasaikeiden prioriteetti. Pullonkaula on paasaikeen SEURANTA (C++-tyosaikeet); liukuhihnan tuottajilla (A, B), debug-videon
# piirrolla/koodauksella ja valotasapainolla on kapasiteettivaraa (jonot puskuroivat). Kun naiden saikeiden prioriteetti on hieman alempi,
# kayttojarjestelma antaa ytimet ensin SEURANNALLE. Vaikuttaa vain ajoitukseen - ruutujen jarjestys ja tulokset ovat samat.
# OLETUKSENA POIS (BG_PRIORITY=1 kytkee paalle): sandboxissa (4 ydinta, kaikki kaytossa) C nopeutui mutta A hidastui yhta paljon ->
# ei kokonaishyotya; koneella jossa on vapaita loogisia ytimia vaikutus voi olla toinen (vertailuajo: ajo "prio").
# Windows: THREAD_PRIORITY_BELOW_NORMAL; Linux: saikeen nice +BG_NICE (oletus 5).

def lower_thread_priority(env_name="BG_PRIORITY"):
    """Laskee KUTSUVAN saikeen prioriteettia (vain kun env_name=1, oletus BG_PRIORITY). Virheet ohitetaan hiljaa."""
    import os as _o, sys as _s
    if _o.environ.get(env_name, "0") != "1":
        return
    try:
        if _s.platform == "win32":
            import ctypes
            k32 = ctypes.windll.kernel32
            k32.SetThreadPriority(k32.GetCurrentThread(), -1)      # THREAD_PRIORITY_BELOW_NORMAL
        elif hasattr(_o, "setpriority"):
            _o.setpriority(_o.PRIO_PROCESS, threading.get_native_id(), int(_o.environ.get("BG_NICE", "5")))
    except Exception:
        pass


def bg_popen_kwargs():
    """v6.11: LIVE_BG_PRIORITY=1 -> ffmpeg-aliprosessit (debug-video, live-tallennus) alemmalle prioriteetille
    (Windows BELOW_NORMAL_PRIORITY_CLASS, muuten nice +5). Vaikuttaa vain ajoitukseen; kuvat ja tulokset samat."""
    import os as _o, sys as _s
    if _o.environ.get("LIVE_BG_PRIORITY", "0") != "1":
        return {}
    if _s.platform == "win32":
        return {"creationflags": getattr(_sp, "BELOW_NORMAL_PRIORITY_CLASS", 0x00004000)}
    return {"preexec_fn": lambda: _o.nice(5)}


class AsyncVideoWriter:
    """cv2.VideoWriter kahdessa taustasaikeessa: (1) piirto/kokoonpano, (2) kirjoitus (koodaus). Pääsäie vain jonottaa tyon (submit)
    tai valmiin kuvan (write); jono taynna -> odottaa."""

    def __init__(self, writer, maxsize=8):
        self._w = writer
        self._qr = _queue.Queue(maxsize=maxsize)    # piirtotyot
        self._qw = _queue.Queue(maxsize=maxsize)    # valmiit kuvat kirjoitettavaksi
        self._tr = threading.Thread(target=self._run_render, daemon=True)
        self._tw = threading.Thread(target=self._run_write, daemon=True)
        self._tr.start()
        self._tw.start()

    def _run_render(self):
        lower_thread_priority()
        while True:
            item = self._qr.get()
            if item is None:
                self._qw.put(None)
                break
            fn, args = item
            # compose() palauttaa jaetun canvas-puskurin -> kopio ennen jonoon laittoa (kirjoitus on eri saikeessa)
            self._qw.put(fn(*args).copy())

    def _run_write(self):
        lower_thread_priority()
        while True:
            img = self._qw.get()
            if img is None:
                break
            self._w.write(img)

    def write(self, img):
        self._qw.put(img.copy())

    def submit(self, fn, *args):
        """Tyo (fn(*args) -> kuva) piirretaan taustasaikeessa ja kirjoitetaan toisessa; paasaie palaa heti."""
        self._qr.put((fn, args))

    def release(self):
        self._qr.put(None)
        self._tr.join()
        self._tw.join()
        self._w.release()
