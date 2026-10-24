"""Kiven tarkka runkomalli ja kuvakohtainen sovitus (alipikseli).

Kivi on pyorahdyskappale, jonka profiili r(z) (solmut, cm) ja korkeus H ovat KIINTEAT koko videolla: kuvakohtaisesti sovitetaan vain
akselin paikka (X, Y). Koko pysyy siis samana koko matkan ja kaikilla kivilla; opetusvaiheessa (kalibroi_kivi.py) profiili ja H sovitetaan
yhdessa kaikkiin ruutuihin (kimppusovitus), ja testaa_kivi_koko.py tarkistaa vapaalla ruutukohtaisella sovituksella (X, Y, R, H), etta R ja H
pysyvat vakiona radan alusta loppuun ja kolmella kivella.

Havainto: graniitin peittavyys (kivi_netti.alfa_kartta, 0..1). Malli: kiven siluetti (renkaiden kupera peite) ilman ylapinnan ellipsia
(ylapinta + kahva ei ole graniittia), sumennettuna reunan pehmeydella ja gammalla:
  * sigma_eff = hypot(SIGMA_OPT_PX, REUNA_B_CM * s_lat): optinen sumeus (px) + reunan fysikaalinen pehmeys (cm; kiilto, jolloin tumma ydin
    paattyy ennen geometrista siluettia) kuvamittakaavassa s_lat (px/cm sivusuunnassa)
  * gamma: alfa_malli = peitto ** GAMMA (videon gamma: tumma kohde nakyy hieman pienempana)
Profiili on siksi "tumman graniittiytimen" profiili: sen reuna on alfa = 0,5 -taso, ei geometrinen ulkoreuna.
Kelvolliset pikselit: tausta jaata (ei maalattuja viivoja), ei varillista (kahvan muovi, iho) eika mustaa (hihat, hanska), ei graniitin sisaisia
kiiltoreikia; kiven ylapinnan ellipsi (ei graniittia) on pehmea "ei valia" -alue."""
import json
import os

import cv2
import numpy as np
from scipy.optimize import least_squares

import kivi_netti as kv
import kivimalli

SS = 4                               # peiton ylinaytteistus
NTH = 48                             # renkaan pisteita
ZD = np.linspace(0.0, 1.0, 31)       # tihea z-jako (suhteellinen)
TH = np.linspace(0.0, 2.0 * np.pi, NTH, endpoint=False)
CT, ST = np.cos(TH), np.sin(TH)

# Oletusmalli (kalibroi_kivi.py kirjoittaa tulokset/kivimalli_<puoli>.json, joka ohittaa nama)
F_SOLMUT = [0.0, 0.04, 0.10, 0.18, 0.28, 0.40, 0.52, 0.64, 0.76, 0.88, 1.0]
R_SOLMUT_CM = [8.0, 9.85, 11.74, 12.04, 12.84, 13.58, 13.69, 13.68, 13.79, 13.92, 14.14]
H_CM = 12.3
SIGMA_OPT_PX = 0.308
REUNA_B_CM = 0.557
GAMMA = 1.3

# kelvolliset pikselit
CHR_MAX = 9.0                        # Lab a*b* -etaisyys taustasta (kahvan muovi, iho; graniitti ja jaa ovat vareetonta)
L_TUMMA = 28.0                       # L < tama: musta hihaa / hanska
REIKA_MAX_ALA = 80                   # graniitin sisaiset alfa < 0,5 -alueet (kiilto) alle tama (px) -> ei valia
DEV_MAX_VARI = 45.0                  # varillisella taustalla: etaisyys janasta tausta-graniitti (BGR), jonka yli ei graniittia eika jaata
TAUSTA_DELTA_PX = 0.7                # taustakuvan rekisteroinnin alipikselivirhe (kameran heilunta <= 0,65 px, tools/kameran_heilunta.py)
TAUSTA_VIRHE_MAX = 0.25              # alfan odotettu virhe (delta x taustan gradientti / |graniitti - tausta|) enintaan tama
TAUSTA_DE_MIN = 22.0                 # graniitin ja taustan Lab-ero (dE) vahintaan tama: alfa on silloin erotettavissa taustasta

# MONIALKUHAKU (sovitus ei saa riippua siemenesta): paikallinen sovitus pysahtyy siemenen lahelle, kun kustannuspinta on tasainen (kiven kasi / hihat peittavat siluetin: kelvollisia
# pikseleita 15-40 %) ja siemen jaa nopeasti liikkuvan kiven jalkeen: tulos seurasi siementa +-25 cm. Silmamaarainen tarkistus 24 ruudusta, joissa vanha ja uusi tulos poikkesivat: uusi osui
# kiveen 20 ruudussa, vanha 2, 2 epaselvaa; kaikissa 21 selvassa ruudussa parempi siluetti oli se, jonka oma kaistakustannus (rms) oli pienempi, eli vanha sovitus jai paikalliseen minimiin.
MONI_DY = tuple(float(v) for v in np.arange(-48.0, 48.1, 6.0))   # skannattavat siirtymat siemenesta (cm, kulkusuunta)
MONI_DX = (-12.0, 0.0, 12.0)                                      # ... ja sivusuunta (cm)
MONI_KAISTA = 14                                                  # skannauksen kaista (px): leveampi kuin sovituksen (8 / 5), jotta etaiset reunat nakyvat
MONI_SKAN_KELV_MIN = 0.15                                         # skannauksen ehdokas: kaistan kelvollinen osuus vahintaan tama
MONI_EHDOKKAITA = 2                                               # tarkennettavien skannausehdokkaiden maara
MONI_ERO_CM = 14.0                                                # ehdokkaiden vahimmaisetaisyys toisistaan
MONI_S0 = 0.35                                                    # valintakustannus: kelvottoman kaistapikselin oletettu virhe (kelvottomaan paikkaan ei voi paeta)
MONI_EPAVARMA_KELV = 0.45                                         # monialkuhaku vain, kun ensimmaisen sovituksen kelvollinen osuus < tama ...
MONI_EPAVARMA_PEITTO = 0.75                                       # ... tai graniittipeitto < tama (valkoisella jaalla sovitus on varma eika siemenesta riipu)
MONI_PEITTO_MIN, MONI_RMS_MAX, MONI_KELV_MIN = 0.35, 0.30, 0.08   # ehdokas kelpaa vain, jos se tayttaa B:n hyvaksymisehdot (netti_seuranta.PEITTO_MIN, B_MAX_RMS, SOFT_KELV)


class Runko:
    """Kiven runkomalli (profiili solmuina) ja sen projektio."""

    def __init__(self, F=None, r_cm=None, H=None, sigma_opt=None, b_cm=None, gamma=None):
        self.F = np.asarray(F_SOLMUT if F is None else F, float)
        self.r = np.asarray(R_SOLMUT_CM if r_cm is None else r_cm, float)
        self.H = float(H_CM if H is None else H)
        self.sigma_opt = float(SIGMA_OPT_PX if sigma_opt is None else sigma_opt)
        self.b_cm = float(REUNA_B_CM if b_cm is None else b_cm)
        self.gamma = float(GAMMA if gamma is None else gamma)

    @staticmethod
    def lataa(kansio, puoli):
        p = os.path.join(kansio, f"kivimalli_{puoli}.json")
        if os.path.exists(p):
            d = json.load(open(p))
            return Runko(d["F"], d["r_cm"], d["H_cm"], d["sigma_opt_px"], d["reuna_b_cm"], d["gamma"])
        return Runko()

    def tallenna(self, polku, **lisa):
        d = dict(F=[float(x) for x in self.F], r_cm=[round(float(x), 3) for x in self.r], H_cm=round(self.H, 3),
                 sigma_opt_px=round(self.sigma_opt, 3), reuna_b_cm=round(self.b_cm, 3), gamma=round(self.gamma, 3))
        d.update(lisa)
        json.dump(d, open(polku, "w"), indent=1)

    @property
    def R_max(self):
        """Suurin profiilin sade (ylarengas voi olla vyotaroa leveampi)."""
        return float(self.r.max())

    @property
    def R_keski(self):
        """Sade keskikorkeudella z = H/2 (sivusuuntainen puolileveys, kuvasta suoraan mitattava)."""
        return float(kivimalli.catmull_rom_r_frac(np.array([0.5]), z_fracs=self.F, r_fracs=self.r)[0])

    def profiili(self, sR=1.0, H=None):
        if sR == 1.0 and H is None:                      # tavallinen sovitus: profiili on sama jokaisella kutsulla (puolet malli_kuva-ajasta) -> valimuisti
            avain = (self.H, self.F.tobytes(), self.r.tobytes())
            v = getattr(self, "_profiili", None)
            if v is None or v[0] != avain:
                v = (avain, kivimalli.catmull_rom_r_frac(ZD, z_fracs=self.F, r_fracs=self.r), ZD * self.H)
                self._profiili = v
            return v[1], v[2]
        r = kivimalli.catmull_rom_r_frac(ZD, z_fracs=self.F, r_fracs=self.r * sR)
        return r, ZD * (self.H if H is None else H)

    def pisteet(self, X, Y, sR=1.0, H=None):
        r, z = self.profiili(sR, H)
        return np.column_stack([(X + r[:, None] * CT[None, :]).ravel(), (Y + r[:, None] * ST[None, :]).ravel(), np.repeat(z, NTH)])

    def hull(self, cam, X, Y, sR=1.0, H=None):
        """Siluetin kupera peite kuvassa (N x 2)."""
        u, v = kivimalli.project_3d(cam.K, cam.R, cam.t, self.pisteet(X, Y, sR, H))
        return cv2.convexHull(np.column_stack([u, v]).astype(np.float32))[:, 0, :]

    def ylapinta(self, cam, X, Y, sR=1.0, H=None):
        r, z = self.profiili(sR, H)
        u, v = kivimalli.project_3d(cam.K, cam.R, cam.t, np.column_stack([X + r[-1] * CT, Y + r[-1] * ST, np.full(NTH, z[-1])]))
        return np.column_stack([u, v]).astype(np.float32)

    def sivumittakaava(self, cam, X, Y):
        """Sivusuuntainen kuvamittakaava (px/cm) kiven keskikorkeudella."""
        a = kivimalli.project_3d(cam.K, cam.R, cam.t, np.array([[X, Y, 7.0], [X + 1.0, Y, 7.0]]))
        return float(np.hypot(a[0][1] - a[0][0], a[1][1] - a[1][0]))


def peite(poly, x0, y0, w, h):
    q = (poly - np.array([x0, y0], np.float32) + 0.5) * SS - 0.5
    mk = np.zeros((h * SS, w * SS), np.uint8)
    cv2.fillConvexPoly(mk, np.round(q * 16).astype(np.int32), 255, cv2.LINE_8, 4)
    return cv2.resize(mk, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


def malli_kuva(cam, runko, win, X, Y, sR=1.0, H=None):
    """Mallin graniittipeitto (sumennettu, gamma) ja ylapinnan pehmea 'ei valia' -paino ikkunassa win = (x0, y0, w, h)."""
    x0, y0, w, h = win
    c = peite(runko.hull(cam, X, Y, sR, H), x0, y0, w, h)
    sig = float(np.hypot(runko.sigma_opt, runko.b_cm * runko.sivumittakaava(cam, X, Y)))
    if sig > 0.05:
        c = cv2.GaussianBlur(c, (0, 0), sigmaX=sig)
    c = np.power(np.clip(c, 0.0, 1.0), runko.gamma)
    t = cv2.GaussianBlur(peite(runko.ylapinta(cam, X, Y, sR, H), x0, y0, w, h), (0, 0), sigmaX=1.5)
    return c, 1.0 - np.clip(2.0 * t, 0.0, 1.0)


# ---------------------------------------------------------------------------------------------------------------------
# HAVAINTO: alfa + kelvolliset pikselit koko ruudusta
# ---------------------------------------------------------------------------------------------------------------------
def tausta_maskit(cam):
    """Taustakuvan (moodikuva) luokitus kerran per kamera: (jaa, vari) = pikselit, joilla kiven alfa (graniitin sekoitus taustaan) on luotettava; jaa = taustan vari on jaan vari,
    vari = varillinen tausta (renkaat, viivat, logot). Luotettavuus mitataan, ei kiintealla rajalla:
      * erotettavuus: Lab-ero graniitin ja taustan valilla (dE) >= TAUSTA_DE_MIN. Lab, koska graniitti on vareeton ja punainen / sininen tausta on varillinen: RGB-etaisyys
        (aiempi raja 60) hylkasi videon 1 punaisen ympyran (56) vaikka dE on ~45;
      * rekisterointivirhe: taustan gradientti x TAUSTA_DELTA_PX / |graniitti - tausta| (alfan virhe, kun moodikuva on delta px vinossa) < TAUSTA_VIRHE_MAX. Aiempi kiintea
        'gradientti > 60, laajennus 7 x 7' hylkasi 30-70 % kiven siluetin pikseleista talon renkaiden, viivojen ja tekstin kohdalla (kelvollisia B-sovituksia talossa vain 0-30 %),
        jolloin rata tuli pehmeasta B:sta ja kahvan blobista, joiden taso poikkeaa taustan mukaan +-3-5 cm: piirretty rata hyppi varirajoilla.
    Kiven alfa lasketaan pikselikohtaisella taustalla, joten rengas ei haittaa; ongelma on vain tumma varjo sinisella renkaalla (ks. havainto: L < L_TUMMA)."""
    if not hasattr(cam, "_tausta"):
        mode = cam.mode.astype(np.int16)
        jaa_vari = kv.jaan_vari(cam)
        ice = np.max(np.abs(mode - jaa_vari[None, None, :]), axis=2) < 24
        mf = cam.mode.astype(np.float32)
        gx = cv2.Sobel(mf, cv2.CV_32F, 1, 0, ksize=3) / 8.0
        gy = cv2.Sobel(mf, cv2.CV_32F, 0, 1, ksize=3) / 8.0
        grad = np.max(np.hypot(gx, gy), axis=2)                                    # taustan tason muutos / px (suurin kanava)
        g = kv.graniitti_vari(cam)
        dist = np.sqrt(np.sum((g.astype(np.float32)[None, None, :] - mf) ** 2, axis=2))
        virhe = TAUSTA_DELTA_PX * grad / np.maximum(dist, 1.0)
        lab_m = cv2.cvtColor(cam.mode, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab_g = cv2.cvtColor(np.clip(g, 0, 255).astype(np.uint8).reshape(1, 1, 3), cv2.COLOR_BGR2LAB).astype(np.float32).reshape(3)
        de = np.sqrt(((lab_m[..., 0] - lab_g[0]) * (100.0 / 255.0)) ** 2 + (lab_m[..., 1] - lab_g[1]) ** 2 + (lab_m[..., 2] - lab_g[2]) ** 2)
        luotettava = (de > TAUSTA_DE_MIN) & (virhe < TAUSTA_VIRHE_MAX) & (cam.lane > 0)
        cam._tausta = (ice & luotettava, luotettava & ~ice)
    return cam._tausta


def jaan_maski(cam):
    """Pikselit, joissa kiven alfa on kayttokelpoinen taustan puolesta (jaa tai varillinen tausta, jossa graniitti erottuu)."""
    jaa, vari = tausta_maskit(cam)
    return jaa | vari


def havainto(cam, frame, alfa=None, varillinen_tausta=True):
    """(alfa, ok): graniitin peittavyys 0..1 ja kelvollisten pikselien maski (bool) koko ruudulle.
    Kelvoton: kahvan muovi, iho, hihat (jaalla: kroma poikkeaa taustasta; varillisella taustalla: etaisyys janasta tausta-graniitti suuri),
    musta/tumma (L < L_TUMMA, myos tumma varjo renkaalla), graniitin sisaiset kiiltoreiat, taustan reunat."""
    if alfa is None:
        fg, g, h = cam.maskit(frame)
        alfa = kv.alfa_kartta(cam, frame, g)
    jaa, vari = tausta_maskit(cam)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    lm = cv2.cvtColor(cam.mode, cv2.COLOR_BGR2LAB).astype(np.float32)
    dch = cv2.GaussianBlur(np.hypot(lab[..., 1] - lm[..., 1], lab[..., 2] - lm[..., 2]), (0, 0), 1.0)
    tumma = lab[..., 0] * (100.0 / 255.0) < L_TUMMA
    # varillisella taustalla: poikkeama janasta [tausta, graniitti] (muovi, iho)
    b = cam.mode.astype(np.float32)
    p = frame.astype(np.float32)
    d = kv.graniitti_vari(cam).astype(np.float32)[None, None, :] - b
    a = np.clip(np.sum((p - b) * d, axis=2) / np.maximum(np.sum(d * d, axis=2), 1.0), 0.0, 1.0)
    rez = p - (b + a[..., None] * d)
    dev = cv2.GaussianBlur(np.sqrt(np.sum(rez * rez, axis=2)), (0, 0), 1.0)
    other_jaa = cv2.dilate(((dch > CHR_MAX) | tumma).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    other_vari = cv2.dilate(((dev > DEV_MAX_VARI) | tumma).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    ice = (alfa < 0.5).astype(np.uint8)
    n, lab_c, st, cen = cv2.connectedComponentsWithStats(ice, connectivity=4)
    h, w = ice.shape
    pieni = (st[:, 4] < REIKA_MAX_ALA) & (st[:, 0] > 0) & (st[:, 1] > 0) & (st[:, 0] + st[:, 2] < w) & (st[:, 1] + st[:, 3] < h)
    pieni[0] = False
    holes = cv2.dilate(pieni[lab_c].astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    ok = (jaa & ~other_jaa) & (~holes)
    if varillinen_tausta:
        ok = ok | (vari & ~other_vari & ~holes)
    return alfa.astype(np.float32), ok


# ---------------------------------------------------------------------------------------------------------------------
# KUVAKOHTAINEN SOVITUS: residuaalit (malli - alfa) * paino alkumallin reunakaistalla
# ---------------------------------------------------------------------------------------------------------------------
def ikkuna(cam, hull, marg=12, rajat=None):
    """Ikkuna (x0, y0, w, h) siluetin ymparilla; rajat = (x0, y0, x1, y1) rajaa (oletus koko kuva)."""
    rx0, ry0, rx1, ry1 = rajat if rajat is not None else (0, 0, cam.w, cam.h)
    x0, y0 = np.floor(hull.min(0)).astype(int) - marg
    x1, y1 = np.ceil(hull.max(0)).astype(int) + marg
    x0, y0, x1, y1 = max(rx0, x0), max(ry0, y0), min(rx1, x1), min(ry1, y1)
    return int(x0), int(y0), int(x1 - x0), int(y1 - y0)


class Kehys:
    """Yhden ruudun sovitusdata: ikkuna, alfa, kelvolliset pikselit ja reunakaistan indeksit alkuasennosta."""

    def __init__(self, cam, runko, alfa, ok, X0, Y0, sR=1.0, H=None, kaista=6, marg=14, ox=0, oy=0):
        """alfa, ok: koko ruutu tai leike, jonka vasen ylakulma on kuvassa (ox, oy)."""
        hull = runko.hull(cam, X0, Y0, sR, H)
        self.win = ikkuna(cam, hull, marg, (ox, oy, ox + alfa.shape[1], oy + alfa.shape[0]))
        x0, y0, w, h = self.win
        self.o = alfa[y0 - oy:y0 - oy + h, x0 - ox:x0 - ox + w].astype(np.float32)
        self.ok = ok[y0 - oy:y0 - oy + h, x0 - ox:x0 - ox + w].astype(np.float32)
        c0 = peite(hull, x0, y0, w, h) > 0.5
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * kaista + 1, 2 * kaista + 1))
        band = cv2.dilate(c0.astype(np.uint8), k) - cv2.erode(c0.astype(np.uint8), k)
        self.idx = np.flatnonzero(band.ravel() > 0)
        self.X0, self.Y0 = X0, Y0

    def residuaalit(self, cam, runko, X, Y, sR=1.0, H=None):
        c, wt = malli_kuva(cam, runko, self.win, X, Y, sR, H)
        return ((c - self.o) * self.ok * wt).ravel()[self.idx]

    def kelvolliset(self, cam, runko, X, Y, sR=1.0, H=None):
        """Kaistan pikselien kelvollinen osuus (0..1): ei-kelvolliset ja ylapinnan alue pois."""
        c, wt = malli_kuva(cam, runko, self.win, X, Y, sR, H)
        w = (self.ok * wt).ravel()[self.idx]
        return float(w.mean()) if len(w) else 0.0


def sovita(cam, runko, alfa, ok, X0, Y0, vapaa=False, kaarros=(10, 6), ox=0, oy=0):
    """Kiven akseli (X, Y) [cm, kalibrointitaso] kuvasta. vapaa = True: sovitetaan lisaksi sade (kerroin sR) ja korkeus H (testaus; kolme
    alkuarvoa, paras valitaan, jotta tulos ei riipu alkuarvosta). Kaksi vaihetta: leveampi kaista -> uusi kaista uudessa paikassa.
    Palauttaa dict(X, Y, rms, kelvolliset, sR, H)."""
    X, Y = float(X0), float(Y0)
    sR, H = 1.0, runko.H
    if not vapaa:
        for kaista in kaarros:
            f = Kehys(cam, runko, alfa, ok, X, Y, kaista=kaista, ox=ox, oy=oy)
            s = least_squares(lambda p: f.residuaalit(cam, runko, p[0], p[1]), [X, Y], bounds=([X - 8.0, Y - 12.0], [X + 8.0, Y + 12.0]),
                              diff_step=3e-3, max_nfev=40)
            X, Y = float(s.x[0]), float(s.x[1])
        r = f.residuaalit(cam, runko, X, Y)
        return dict(X=X, Y=Y, rms=float(np.sqrt(np.mean(r ** 2))), kelvolliset=f.kelvolliset(cam, runko, X, Y), sR=1.0, H=runko.H)
    # vapaa sovitus: ensin akseli kiinteilla mitoilla, sitten (X, Y, sR, H) kolmesta alkuarvosta
    for kaista in kaarros:
        f = Kehys(cam, runko, alfa, ok, X, Y, kaista=kaista, ox=ox, oy=oy)
        s = least_squares(lambda p: f.residuaalit(cam, runko, p[0], p[1]), [X, Y], bounds=([X - 8.0, Y - 12.0], [X + 8.0, Y + 12.0]),
                          diff_step=3e-3, max_nfev=40)
        X, Y = float(s.x[0]), float(s.x[1])
    f = Kehys(cam, runko, alfa, ok, X, Y, kaista=6, ox=ox, oy=oy)
    paras = None
    for sR0, dH in ((1.0, 0.0), (0.97, -1.5), (1.03, 1.5)):
        p0 = [X, Y, sR0, runko.H + dH]
        s = least_squares(lambda p: f.residuaalit(cam, runko, p[0], p[1], p[2], p[3]), p0,
                          bounds=([X - 6.0, Y - 8.0, 0.85, runko.H - 6.0], [X + 6.0, Y + 8.0, 1.15, runko.H + 6.0]),
                          diff_step=[3e-3, 3e-3, 2e-2, 2e-2], x_scale=[1.0, 1.0, 0.05, 1.0], max_nfev=60)
        if paras is None or s.cost < paras.cost:
            paras = s
    X, Y, sR, H = (float(v) for v in paras.x)
    r = f.residuaalit(cam, runko, X, Y, sR, H)
    return dict(X=X, Y=Y, rms=float(np.sqrt(np.mean(r ** 2))), kelvolliset=f.kelvolliset(cam, runko, X, Y, sR, H), sR=sR, H=H)


# ---------------------------------------------------------------------------------------------------------------------
# MONIALKUHAKU
# ---------------------------------------------------------------------------------------------------------------------
def kaistakustannus(cam, runko, alfa, ok, X, Y, kaista, ox=0, oy=0, s0=MONI_S0):
    """(rms, rangaistu, kelvollinen osuus) paikassa (X, Y): rms = reunakaistan painotettu residuaali (sama kuin sovituksen tavoite, kelvottomat pikselit pois); rangaistu = sama, mutta
    kelvoton kaistapikseli maksaa s0^2 (muuten kelvottomaan paikkaan, esim. hihaan, sijoitettu malli saisi pienen rms:n)."""
    k = Kehys(cam, runko, alfa, ok, X, Y, kaista=kaista, ox=ox, oy=oy)
    c, wt = malli_kuva(cam, runko, k.win, X, Y)
    w = (k.ok * wt).ravel()[k.idx]
    r = ((c - k.o) * k.ok * wt).ravel()[k.idx]
    rms = float(np.sqrt(np.sum(r * r) / max(float(np.sum(w * w)), 1e-6)))
    rang = float(np.sqrt(np.mean(r * r + (1.0 - np.clip(w, 0.0, 1.0)) * s0 * s0)))
    return rms, rang, float(w.mean()) if len(w) else 0.0


def moni_ehdokkaat(cam, runko, alfa, ok, X0, Y0, ox=0, oy=0):
    """Skannaus siemenen (X0, Y0) ymparilla (MONI_DY x MONI_DX): leveakaistaisen kustannuksen (rms, kelvollinen osuus >= MONI_SKAN_KELV_MIN) paikalliset minimit, parhaat
    MONI_EHDOKKAITA kpl vahintaan MONI_ERO_CM:n paassa toisistaan: [(kustannus, X, Y)]. Ehdokkaat, joiden ikkuna ei mahdu leikkeeseen, jatetaan pois."""
    h, w = alfa.shape[:2]
    out = []
    for dy in MONI_DY:
        for dx in MONI_DX:
            X, Y = X0 + dx, Y0 + dy
            try:
                hull = runko.hull(cam, X, Y)
                if hull[:, 0].min() < ox + 8 or hull[:, 1].min() < oy + 8 or hull[:, 0].max() > ox + w - 8 or hull[:, 1].max() > oy + h - 8:
                    continue
                rms, _, kelv = kaistakustannus(cam, runko, alfa, ok, X, Y, MONI_KAISTA, ox, oy)
            except Exception:
                continue
            if kelv >= MONI_SKAN_KELV_MIN:
                out.append((rms, X, Y))
    out.sort()
    valitut = []
    for c in out:
        if all(np.hypot(c[1] - v[1], c[2] - v[2]) >= MONI_ERO_CM for v in valitut):
            valitut.append(c)
            if len(valitut) >= MONI_EHDOKKAITA:
                break
    return valitut
