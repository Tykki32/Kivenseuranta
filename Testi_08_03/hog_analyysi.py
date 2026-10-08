"""Hog-hog -analyysi: heiton nopeus kaukaisella hoglinella, hidastuvuus, hog-hog-aika, suunta, liuku ja irroitus.

Datapisteet: radan rivit joiden Y on valilla [lahihog + marginaali, kaukohog - marginaali]. Y(t) kitkamallista
dv/dt = -g (A + B ln v) (B = HOG_MU_B kiintea; sovitetaan y0, v0, A); huonoimmin sopivat pisteet pudotetaan ja sovitetaan
uudelleen. Malli integroidaan (KitkaRata) myos datan ulkopuolelle hoglineille. Jos R = sqrt(R^2) > HOG_MIN_R:
  * nopeus kaukaisella hoglinella v hetkella jolloin Y(t) = kaukohog,
  * hidastuvuus g (A + B ln v) nopeudella HOG_HIDASTUVUUS_NOPEUDELLA_MS,
  * hog-hog-aika = t(Y = lahihog) - t(Y = kaukohog).
X-SUUNTA: kurvimalli (vakio sivukiihtyvyys) samoille pisteille -> suunta kaukohogilla ja IRROITUS = X, jonka kaukohogin
suunta jatkettuna saisi lahemmalla T-viivalla. LIUKU: suora X(Y) heiton alusta kaukohog + 1 m asti (+ hakki lahtopisteena,
erikseen X = +-15 cm) -> X lahemmalla T-viivalla. Tulos hyvaksytaan vain jos R_y > HOG_MIN_R ja kurvimallin
jaannoksen rms < HOG_X_MAX_RMS_CM. Y pienenee kun kivi etenee kohti lahempaa pesaa.
"""
import numpy as np
import cv2

import asetukset as A


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
    for _ in range(12):                       # liian suuri kitka: mallin kivi pysahtyy ennen datan loppua -> pienempi A
        if r is not None:
            break
        p[2] *= 0.7
        r = resid(p)
    if r is None:
        return None
    cost = float(r @ r); lam = 1e-3
    for _ in range(iters):
        J = np.empty((len(y), npar))
        for j in range(npar):
            q = p.copy(); q[j] += steps[j]
            rj = resid(q)
            if rj is None:                    # eteenpain-askel pysayttaa kiven -> taaksepain-erotus
                q = p.copy(); q[j] -= steps[j]
                rj = resid(q)
                if rj is None:
                    return None
                J[:, j] = (r - rj) / steps[j]
            else:
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


def curl_basis(rata_, t_lo, t_hi, n=3000):
    """Palauttaa (g, S, G, H): S = kuljettu matka (cm), G = int dt / v (s / (m/s)), H = int v G dt; v kitkamallista (cm/s)."""
    g = np.linspace(t_lo, t_hi, n)
    v = np.maximum(100.0 * rata_.v(g), 2.0)            # cm/s
    S = _ctz(g, v)
    G = _ctz(g, 1.0 / (v / 100.0))
    H = _ctz(g, v * G)
    return g, S, G, H


def _fr_param(fr):
    return np.array([fr["y0_cm"], fr["v0_ms"], fr["A"], fr["B"]])


def _r_model(y, pred):
    if pred is None:
        return 0.0
    ss_res = float(np.sum((y - pred) ** 2)); ss_tot = float(np.sum((y - y.mean()) ** 2))
    return float(np.sqrt(max(1.0 - ss_res / ss_tot, 0.0))) if ss_tot > 0 else 0.0


class KitkaRata:
    """Kitkamallin rata tiheana taulukkona: t = 0 sovituksen alussa (Y = y0, v = v0), integroitu taaksepain t_taakse s
    ja eteenpain kunnes kivi pysahtyy (v < 0,02 m/s) tai t_max. dv/dt = -g (A + B ln v), Y pienenee nopeudella v."""

    def __init__(self, y0_cm, v0_ms, A_, B_, t_taakse=8.0, t_max=60.0, dt=0.01):
        f = lambda v: -G_MS2 * (A_ + B_ * np.log(max(v, 1e-3)))
        osat = []
        for h, n in ((-dt, int(t_taakse / dt)), (dt, int(t_max / dt))):
            ts, ys, vs = [0.0], [y0_cm], [v0_ms]
            tc, yc, vc = 0.0, float(y0_cm), float(v0_ms)
            for _ in range(n):
                k1 = f(vc); k2 = f(vc + h / 2 * k1); k3 = f(vc + h / 2 * k2); k4 = f(vc + h * k3)
                vn = vc + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                if h > 0 and vn <= 0.02:
                    break
                yc -= h * 100.0 * (vc + vn) / 2; vc = vn; tc += h
                ts.append(tc); ys.append(yc); vs.append(vc)
            osat.append((ts, ys, vs))
        (tb, yb, vb), (tf_, yf, vf) = osat
        self.t = np.array(tb[::-1] + tf_[1:]); self.y_cm = np.array(yb[::-1] + yf[1:]); self.v_ms = np.array(vb[::-1] + vf[1:])

    def y(self, t):
        return np.interp(t, self.t, self.y_cm)

    def v(self, t):
        return np.interp(t, self.t, self.v_ms)

    def t_at(self, Y):
        """Hetki jolloin Y(t) = Y (None jos mallin rata ei kata sita: kivi pysahtyy ennen tai Y on alun takana)."""
        if Y > self.y_cm[0] or Y < self.y_cm[-1]:
            return None
        return float(np.interp(Y, self.y_cm[::-1], self.t[::-1]))


def _r_of(tt, y, a, b, c):
    ss_res = float(np.sum((y - (a * tt * tt + b * tt + c)) ** 2)); ss_tot = float(np.sum((y - y.mean()) ** 2))
    return float(np.sqrt(max(1.0 - ss_res / ss_tot, 0.0))) if ss_tot > 0 else 0.0


def analyze_hog(rows, near_hog_cm, far_hog_cm, min_r=A.HOG_MIN_R, tee_cm=None, max_x_rms=A.HOG_X_MAX_RMS_CM, osittainen=False):
    """rows: [(frame, timestamp_s, {"Y_cm": ...}), ...]. Palauttaa dict: ok (bool), reason, n, R, ...
    osittainen=True: rata katkesi ennen lahihogia. Data riittaa, kun se kattaa kaukopaan ja vahintaan
    A.HOG_OSITTAIN_MIN_MATKA_CM; jos lahihogin paa puuttuu, hog-hog-aika, t_near ja v_near jaavat None:ksi
    (nopeus kaukohogilla, hidastuvuus, suunta, irroitus ja liuku lasketaan). out["osittainen"] = True."""
    ylo, yhi = near_hog_cm + A.HOG_LAHI_MARGINAALI_CM, far_hog_cm - A.HOG_KAUKO_MARGINAALI_CM
    pts = [(float(t), float(r["Y_cm"])) for _, t, r in rows if r.get("Y_cm") is not None and ylo <= float(r["Y_cm"]) <= yhi]
    out = dict(ok=False, reason="", n=len(pts), y_lo_cm=ylo, y_hi_cm=yhi, near_hog_cm=near_hog_cm, far_hog_cm=far_hog_cm)
    if len(pts) < A.HOG_MIN_PISTEET:
        out["reason"] = f"liian vahan pisteita ({len(pts)} < {A.HOG_MIN_PISTEET})"
        return out
    t = np.array([p[0] for p in pts]); y = np.array([p[1] for p in pts])
    lahi_ok = y.min() <= ylo + A.HOG_KATTAVUUS_CM
    if osittainen:
        out["osittainen"] = True
        if y.max() < yhi - A.HOG_KATTAVUUS_CM or y.max() - y.min() < A.HOG_OSITTAIN_MIN_MATKA_CM:
            out["reason"] = (f"rata katkesi liian aikaisin (data {y.min() / 100:.1f}-{y.max() / 100:.1f} m, "
                             f"vaaditaan {A.HOG_OSITTAIN_MIN_MATKA_CM / 100:.0f} m kaukopaasta)")
            return out
    elif y.max() < yhi - A.HOG_KATTAVUUS_CM or not lahi_ok:
        out["reason"] = "kivi ei ole kulkenut riittavasti (data ei kata sovitusvalia)"
        return out
    t0 = float(t.min()); tt = t - t0
    # KITKAMALLI: dv/dt = -g (A + B ln v), B = A.HOG_MU_B kiintea; sovitetaan y0, v0 ja A. Huonoimmin sopivat pisteet pois.
    m1 = tt <= tt.min() + 1.0
    v0_g = -np.polyfit(tt[m1], y[m1], 1)[0] / 100.0 if m1.sum() >= 5 else (y[0] - y[-1]) / max(tt[-1], 1e-3) / 100.0
    v0_g = max(v0_g, 0.3)
    T_ = max(float(tt[-1]), 1e-3); D_ = float(y[0] - y[-1]) / 100.0     # alkuarvaus: tasainen hidastuvuus koko datalle
    dec_g = float(np.clip(2.0 * (v0_g * T_ - D_) / T_ ** 2, 0.02, min(0.2, 0.9 * v0_g ** 2 / (2.0 * max(D_, 0.1)))))
    fr = fit_log_friction(tt, y, v0_g, dec_g, fixed_B=A.HOG_MU_B)
    if fr is None:
        out["reason"] = "kitkamallin sovitus ei onnistunut"
        return out
    R1 = _r_model(y, _fric_y(_fr_param(fr), tt))
    n_drop = min(A.HOG_PUDOTA_HUONOIMMAT, max(0, len(y) - A.HOG_MIN_PISTEET))
    if n_drop > 0:                                   # pudota n_drop huonoiten sopivaa pistetta ja sovita uudelleen
        keep = np.sort(np.argsort(np.abs(y - _fric_y(_fr_param(fr), tt)))[: len(y) - n_drop])
        tt, y = tt[keep], y[keep]
        fr2 = fit_log_friction(tt, y, fr["v0_ms"], G_MS2 * (fr["A"] + fr["B"] * np.log(max(fr["v0_ms"], 0.2))), fixed_B=A.HOG_MU_B)
        fr = fr2 if fr2 is not None else fr
    pred = _fric_y(_fr_param(fr), tt)
    ss_res = float(np.sum((y - pred) ** 2)); ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    R = float(np.sqrt(max(r2, 0.0)))
    out.update(R=R, R_ennen_suodatusta=R1, n_pudotettu=n_drop, n_kaytetty=int(len(y)), r2=r2, a=None, b=None, c=None, t0=t0,
               rms_cm=float(np.sqrt(ss_res / len(y))), mu_a=fr["A"], mu_b=fr["B"], kitka_rms_cm=float(np.sqrt(ss_res / len(y))))
    if R <= min_r:
        out["reason"] = f"sovitus ei riittavan hyva (R = {R:.4f} <= {min_r})"
        return out
    rata_ = KitkaRata(fr["y0_cm"], fr["v0_ms"], fr["A"], fr["B"])
    t_hi_y = rata_.t_at(yhi)                                              # alueen alku (kaukopaa)
    t_lo_y = rata_.t_at(max(ylo, float(y.min())) if osittainen else ylo)  # alueen loppu (lahipaa / datan loppu)
    t_far = rata_.t_at(far_hog_cm)
    t_near = rata_.t_at(near_hog_cm) if lahi_ok else None
    if None in (t_hi_y, t_lo_y, t_far) or (t_near is None and not osittainen):
        out["reason"] = "kitkamallin rata ei ylita hoglinea (kivi pysahtyy mallissa ennen sita)"
        return out
    v = lambda tx: 100.0 * rata_.v(tx)                     # cm/s (positiivinen = kohti lahempaa pesaa)
    decel_avg = (v(t_hi_y) - v(t_lo_y)) / (t_lo_y - t_hi_y) / 100.0 if t_lo_y != t_hi_y else 0.0
    decel_out = G_MS2 * (fr["A"] + fr["B"] * np.log(A.HOG_HIDASTUVUUS_NOPEUDELLA_MS))
    out.update(ok_y=True, v_far_hog_ms=v(t_far) / 100.0, decel_ms2=float(decel_out), decel_keskim_ms2=float(decel_avg),
               hog_hog_s=None if t_near is None else float(t_near - t_far), t_far_hog_s=float(t_far + t0),
               t_near_hog_s=None if t_near is None else float(t_near + t0),
               v_near_hog_ms=None if t_near is None else v(t_near) / 100.0)

    # ---- X-suuntainen analyysi ----
    pts_x = [(float(t_), float(r["Y_cm"]), float(r["X_cm"]), r.get("rms_px")) for _, t_, r in rows
             if r.get("Y_cm") is not None and r.get("X_cm") is not None and ylo <= float(r["Y_cm"]) <= yhi]
    if len(pts_x) < A.HOG_MIN_PISTEET:
        out["ok"] = False; out["reason"] = "X-pisteita liian vahan"
        return out
    tx = np.array([p[0] for p in pts_x]) - t0
    xx = np.array([p[2] for p in pts_x])
    y_eq = rata_.y(tx)                                     # Y kitkamallista ajan funktiona
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
    slope = qx / 100.0                                      # dX/dY kaukohoglinella (u = 0), cm / cm
    x_far = float(rx)
    dir_deg = float(np.degrees(np.arctan(-slope)))          # kulkusuunta (Y pienenee): + = kohti +X
    out.update(x_far_hog_cm_2aste=x_far, dir_far_hog_deg_2aste=dir_deg)
    # kurvimalli: X(t) = x0 + theta0 S(t) + k H(t); sama pistejoukko. Robusti painotettu sovitus: pisteen paino
    # siluettisovituksen laadusta w0 = 1 / (1 + (rms_px / HOG_X_PAINO_RMS_PX)^2) (lakaisijan/harjan kanssa sulautunut
    # havainto -> suuri rms_px) kertaa Huber-paino jaannoksesta (> HOG_X_HUBER_K robustia hajontaa -> pienempi paino).
    # Jaannoksen rms ja hyvaksynta vain pisteista, joiden paino > HOG_X_MUKANA_PAINO (kiintea "10 huonointa pois" ei
    # riittanyt, kun lahihogin puolella kymmenet havainnot ovat sulautuneet: live 2026-10-07, kivet 2315, 2347, 2348).
    txa = np.array([p_[0] for p_ in pts_x]) - t0
    xxa = np.array([p_[2] for p_ in pts_x])
    rpx = np.array([np.nan if p_[3] is None else float(p_[3]) for p_ in pts_x])
    g, S, G, Hh = curl_basis(rata_, min(txa.min(), t_far) - 0.05, txa.max() + 0.05)
    Mx = np.column_stack([np.ones_like(txa), np.interp(txa, g, S), np.interp(txa, g, Hh)])
    w0 = 1.0 / (1.0 + (np.nan_to_num(rpx, nan=A.HOG_X_PAINO_RMS_PX) / A.HOG_X_PAINO_RMS_PX) ** 2)
    w = w0.copy()
    for _ in range(10):
        sw = np.sqrt(w)
        co, *_ = np.linalg.lstsq(Mx * sw[:, None], xxa * sw, rcond=None)
        res_x = xxa - Mx @ co
        sig = max(1.4826 * float(np.median(np.abs(res_x))), 0.3)
        w = w0 * np.minimum(1.0, A.HOG_X_HUBER_K * sig / np.maximum(np.abs(res_x), 1e-9))
    kk = np.where(w > A.HOG_X_MUKANA_PAINO * w0.max())[0]
    if len(kk) < A.HOG_MIN_PISTEET:
        out["ok"] = False
        out["reason"] = f"X-pisteita liian vahan robustin painotuksen jalkeen ({len(kk)} < {A.HOG_MIN_PISTEET})"
        return out
    x_far = float(co[0] + co[1] * np.interp(t_far, g, S) + co[2] * np.interp(t_far, g, Hh))
    theta_far = float(co[1] + co[2] * np.interp(t_far, g, G))
    slope = -theta_far                                  # dX/dY = X'/Y' = v theta / (-v)
    dir_deg = float(np.degrees(np.arctan(theta_far)))
    kurvi_rms = float(np.sqrt(np.mean((xxa[kk] - Mx[kk] @ co) ** 2)))
    yk = np.array([pts_x[i][1] for i in kk])
    out.update(kurvi_k_ms2=float(co[2]), kurvi_rms_cm=kurvi_rms, kurvi_n_mukana=int(len(kk)),
               kurvi_y_mukana_cm=(float(yk.min()), float(yk.max())))
    # mukana olevien pisteiden on katettava riittavasti rataa ja kaukopaa (irroitus ja suunta lasketaan kaukohogilla)
    if yk.max() - yk.min() < A.HOG_X_MIN_KATTAVUUS_CM or yk.max() < yhi - A.HOG_X_KAUKOPAA_MAX_PUUTE_CM:
        out["ok"] = False
        out["reason"] = (f"X-pisteet eivat kata rataa robustin painotuksen jalkeen (Y {yk.min() / 100:.1f}-{yk.max() / 100:.1f} m)")
        return out
    # hyvaksynta kurvimallin jaannoksesta (cm): R_x (paraabeli) hylkasi vahan taipuvat ja paraabeliin sopimattomat heitot
    if not kurvi_rms < max_x_rms:
        out["ok"] = False
        out["reason"] = f"X-kurvimallin jaannos liian suuri (rms = {kurvi_rms:.2f} cm >= {max_x_rms:.1f} cm)"
        return out
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
    """Tekstirivit (terminaali + kuva). Vain onnistuneelle (R_y, X-kurvimallin rms) analyysille; muuten None."""
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
             f"hog-hog aika: {res['hog_hog_s']:.2f} s" if res.get("hog_hog_s") is not None else "hog-hog aika: - (rata katkesi ennen lahihogia)",
             f"suunta kaukohogilla: {abs(d):.2f} deg kohti {side}",
             f"liuku (suoran X T-viivalla, alusta hog+1m): {liuku}"]
    if "x_straight_at_tee_cm" in res:
        lines.append(f"merkki (suoran X T-viivalla, hogin jalkeen): {res['x_straight_at_tee_cm']:+.1f} cm")
    if res.get("kierteet") is not None:
        lines.append(f"kierteita hog-hog: {res['kierteet']:.1f} (kierrosaika {res['kierrosaika_far_s']:.1f} s -> {res['kierrosaika_near_s']:.1f} s)")
    if res.get("kierteet_taysi") is not None:
        lines.append(f"kierteita hog-hog (taysi resoluutio): {res['kierteet_taysi']:.1f} (kierrosaika {res['kierrosaika_far_s_taysi']:.1f} s -> "
                     f"{res['kierrosaika_near_s_taysi']:.1f} s)")
    lines.append(f"R_y = {res['R']:.5f}  X-kurvimallin rms = {res['kurvi_rms_cm']:.2f} cm  (R_x = {res['R_x']:.5f})")
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
