"""Piirrokset alkuperaiseen (1920 x 1080) videoruutuun, ei top-down-kuvaa.

Heittopaa (vasen / oikea puolisko): kiven keskikohta pisteina (kahvan levyn keskipiste z = PISTE_Z), live: piste ilmestyy kun sen ruutu nayttyy tulosteessa; koko
pistejoukko jaa nakyviin JALKI_S sekuntia heiton loputtua. Toinen paa: liukulinja ja irroituslinja heti heiton alusta (tulosteen kuva on analyysia VIIVE_S jäljessa,
joten estimaatti ehtii valmistua ennen kuin kivi liikahtaa tulosteessa); linjat jaavat nakyviin seuraavaan heittoon tai VIIVAT_S sekuntiin asti.

Linjat ovat suoria (liukulinja: heiton liu'un keskimaarainen suunta hackista irroituspisteeseen; irroituslinja: vapaan lennon suunta irroitushetkella) ja
ne jatketaan toisen paan T-viivalle; todellinen rata kaartaa (kierre), joten kiven rata poikkeaa irroituslinjasta sita enemman mita pidemmalle mennaan.
"""
import cv2
import numpy as np

import netti_geom as ng

PISTE_Z = 15.0
VARI_PISTE = (60, 230, 60)               # BGR: kirkas vihrea
VARI_PISTE_REUNA = (20, 20, 20)
VARI_NYKYINEN = (40, 220, 255)           # nykyinen paikka: keltainen rengas
VARI_LIUKU = (0, 140, 255)               # oranssi
VARI_IRROITUS = (255, 60, 220)           # magenta
VARI_PANEELI = (30, 30, 30)
FONTTI = cv2.FONT_HERSHEY_SIMPLEX
Y_ALKU_KAUKO = -170.0                    # linjan piirto kaukopaassa: Y' tasta ...
Y_LOPPU_KAUKO = 1100.0                   # ... tahan (cm T-viivasta)
PANEELI_ALUE = (290, 8, 690, 186)        # ylapalkin vapaa alue (x0, y0, x1, y1): ei logoa eika tulostaulua


def _viiva_kuvaan(img, pts, vari, paksuus, katkoviiva=False, taustavari=(255, 255, 255)):
    """Murtoviiva pikselikoordinaateissa; valkoinen tausta + vari -> nakyy jaalla ja renkaiden paalla. katkoviiva: lyhyet patkat."""
    pts = np.round(pts).astype(np.int32)
    h, w = img.shape[:2]
    segs = []
    for a, b in zip(pts[:-1], pts[1:]):
        ok, p, q = cv2.clipLine((0, 0, w, h), tuple(int(v) for v in a), tuple(int(v) for v in b))
        if ok:
            segs.append((p, q))
    if katkoviiva:
        segs = segs[::2]
    for p, q in segs:
        cv2.line(img, p, q, taustavari, paksuus + 2, cv2.LINE_AA)
    for p, q in segs:
        cv2.line(img, p, q, vari, paksuus, cv2.LINE_AA)


def _teksti(img, txt, org, koko=0.55, vari=(255, 255, 255), paksuus=1, tausta=True):
    (w, h), pohja = cv2.getTextSize(txt, FONTTI, koko, paksuus)
    x, y = int(org[0]), int(org[1])
    if tausta:
        cv2.rectangle(img, (x - 3, y - h - 3), (x + w + 3, y + pohja + 2), VARI_PANEELI, -1)
    cv2.putText(img, txt, (x, y), FONTTI, koko, vari, paksuus, cv2.LINE_AA)
    return w, h


def mediaani5(P):
    """Pisteiden (N x 2) liukuva mediaani (5), reunoilla katkaistu ikkuna: pehmeat sovitukset voivat sisaltaa yksittaisen ulkolaisen."""
    if len(P) < 3:
        return P
    out = P.copy()
    for i in range(len(P)):
        a, b = max(0, i - 2), min(len(P), i + 3)
        out[i] = np.median(P[a:b], axis=0)
    return out


class Piirtaja:
    def __init__(self, geom, fps=30.0, jalki_s=10.0, viivat_s=60.0, piste_z=PISTE_Z, paneeli=True, viive_s=5.0):
        self.g = geom
        self.fps = float(fps)
        self.jalki_s = jalki_s
        self.viivat_s = viivat_s
        self.z = piste_z
        self.paneeli = paneeli
        self.viive_s = viive_s

    # ---- pisteet heittopaassa ------------------------------------------------------------------------------------
    def _pisteet(self, img, h, f_out):
        pts = h["pisteet"]
        if not pts:
            return
        F = np.array([p[0] for p in pts])
        P = mediaani5(np.array([[p[1], p[2]] for p in pts]))
        # aukkojen (kasi peittaa, B ei kelpaa) lineaarinen taytto <= 12 ruutua: pisteet nakyvat yhtenaisena jonona
        Ff, Pf, interp = [F[0]], [P[0]], [False]
        for i in range(1, len(F)):
            gap = F[i] - F[i - 1]
            if 2 < gap <= 12:
                for k in range(2, int(gap), 2):
                    u = k / gap
                    Ff.append(F[i - 1] + k)
                    Pf.append(P[i - 1] + u * (P[i] - P[i - 1]))
                    interp.append(True)
            Ff.append(F[i]); Pf.append(P[i]); interp.append(False)
        Ff, Pf = np.array(Ff), np.array(Pf)
        n = int(np.searchsorted(Ff, f_out, side="right"))
        if n == 0:
            return
        uv = self.g[h["puoli"]].maa_kuvaan(Pf[:n, 0], Pf[:n, 1], self.z)
        for (u, v), ip in zip(uv[:-1], interp[:n - 1]):
            c = (int(round(u)), int(round(v)))
            cv2.circle(img, c, 3 if not ip else 2, VARI_PISTE_REUNA, -1, cv2.LINE_AA)
            cv2.circle(img, c, 2 if not ip else 1, VARI_PISTE, -1, cv2.LINE_AA)
        u, v = uv[-1]
        c = (int(round(u)), int(round(v)))
        if f_out <= h["f_loppu"] + 2:                     # kivi viela kuva-alalla: nykyinen paikka renkaana
            cv2.circle(img, c, 7, VARI_PISTE_REUNA, 3, cv2.LINE_AA)
            cv2.circle(img, c, 7, VARI_NYKYINEN, 1, cv2.LINE_AA)
        else:
            cv2.circle(img, c, 4, VARI_PISTE_REUNA, -1, cv2.LINE_AA)
            cv2.circle(img, c, 3, VARI_PISTE, -1, cv2.LINE_AA)

    # ---- viivat toisessa paassa ----------------------------------------------------------------------------------
    def _kaukopiste(self, kaukopuoli, X, Y):
        """Heittopaan maakoordinaatit -> kaukopaan kuvapiste."""
        Xk, Yk = ng.toiselle_paalle(X, Y)
        return self.g[kaukopuoli].maa_kuvaan(Xk, Yk, 0.0)

    def _suora_pikseleina(self, kaukopuoli, X_Y):
        """Suora X = X_Y(Y) (heittopaan koordinaatit; funktio) piirrettavaksi kaukopaan kuvassa: Y' = Y_ALKU_KAUKO ... Y_LOPPU_KAUKO."""
        Yk = np.linspace(Y_ALKU_KAUKO, Y_LOPPU_KAUKO, 120)
        Y = ng.ETAISYYS_CM - Yk
        X = X_Y(Y)
        return self.g[kaukopuoli].maa_kuvaan(-X, Yk, 0.0), -X

    def _viivat(self, img, h, f_out):
        v = h["viivat"]
        if v is None:
            return None
        kauko = ng.Geometria.toinen(h["puoli"])
        L = ng.ETAISYYS_CM
        # liukulinja
        a, b = v["liuku_a"], v["liuku_b"]
        pts, Xk = self._suora_pikseleina(kauko, lambda Y: a + b * Y)
        alustava = not v.get("vakaa", True)
        _viiva_kuvaan(img, pts, VARI_LIUKU, 3)
        XT_liuku = a + b * L
        # irroituslinja (vapaan lennon suunta irroituspisteesta); alustava = katkoviiva
        Xr, Yr, th = v["irr_X"], v["irr_Y"], np.tan(np.radians(v["irr_suunta_deg"]))
        pts2, _ = self._suora_pikseleina(kauko, lambda Y: Xr + th * (Y - Yr))
        _viiva_kuvaan(img, pts2, VARI_IRROITUS, 3, katkoviiva=alustava)
        XT_irr = Xr + th * (L - Yr)
        # T-viivan leikkauspisteet; nimike viivan alkuun (ensimmainen nakyva piste), kun leikkauspiste on kuva-alan ulkopuolella
        px0, py0, pw, ph = (ng.kn.PUOLET[kauko])
        for XT, vari, nimi, pts_ in ((XT_liuku, VARI_LIUKU, "LIUKU", pts), (XT_irr, VARI_IRROITUS, "IRROITUS", pts2)):
            uv = self._kaukopiste(kauko, XT, L)[0]
            c = (int(round(uv[0])), int(round(uv[1])))
            sisalla = px0 + 6 <= c[0] < px0 + pw - 6 and py0 + 6 <= c[1] < py0 + ph - 6
            if sisalla:
                cv2.circle(img, c, 9, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(img, c, 7, vari, -1, cv2.LINE_AA)
            sana = f"{nimi} {abs(XT):.0f} cm {'oik' if XT > 0 else 'vas'}"
            if c[0] >= px0 + pw - 6:
                sana += " >>"                             # leikkauspiste kuva-alan oikealla puolella
            elif c[0] < px0 + 6:
                sana = "<< " + sana
            # ensimmainen puoliskon sisalla oleva piste (pienin Y'): tasta nimike lahelle
            vis = [q for q in np.round(pts_).astype(int) if px0 + 4 <= q[0] < px0 + pw - 4 and py0 + 4 <= q[1] < py0 + ph - 4]
            if vis:
                q = vis[0]
                (tw, th), _ = cv2.getTextSize(sana, FONTTI, 0.5, 1)
                x = int(np.clip(q[0] - tw // 2, px0 + 6, px0 + pw - tw - 10))
                y = int(np.clip(q[1] + (30 if nimi == "IRROITUS" else 52), py0 + 24, py0 + ph - 8))
                _teksti(img, sana, (x, y), 0.5, vari, 1)
        virhe_cm = float((L - Yr) * np.tan(np.radians(v["irr_virhe_deg"])))        # irroituksen suunnan virhe T-viivalla
        return dict(XT_liuku=XT_liuku, XT_irr=XT_irr, alustava=alustava, virhe_cm=virhe_cm)

    # ---- tietopaneeli -----------------------------------------------------------------------------------------------
    def _paneeli(self, img, h, v, lisa, f_out):
        x0, y0, x1, y1 = PANEELI_ALUE
        ov = img.copy()
        cv2.rectangle(ov, (x0, y0), (x1, y1), VARI_PANEELI, -1)
        cv2.addWeighted(ov, 0.72, img, 0.28, 0, img)
        rivit = [(f"HEITTO {h.get('numero', h['tunnus'])}  {h['vari']}", (255, 255, 255), 0.62, 2)]
        if v is None:
            rivit.append(("analysoidaan ...", (200, 200, 200), 0.5, 1))
        else:
            tila = "alustava" if lisa["alustava"] else "valmis"
            rivit += [
                (f"liuku {v['liuku_suunta_deg']:+.1f} deg  hidast. {v['liuku_hidastuvuus']:+.2f} m/s2", VARI_LIUKU, 0.5, 1),
                (f"irroitus Y={v['irr_Y']:.0f} cm  suunta {v['irr_suunta_deg']:+.1f} +-{v['irr_virhe_deg']:.1f} deg", VARI_IRROITUS, 0.5, 1),
                (f"nopeus ennen {v['nopeus_ennen']:.2f}  jalkeen {v['nopeus_jalkeen']:.2f} m/s", (255, 255, 255), 0.5, 1),
                (f"T-viiva: liuku {lisa['XT_liuku']:+.0f}, irroitus {lisa['XT_irr']:+.0f} +-{lisa['virhe_cm']:.0f} cm ({tila})", (255, 255, 255), 0.45, 1),
            ]
        y = y0 + 24
        for txt, vari, koko, pks in rivit:
            cv2.putText(img, txt, (x0 + 8, y), FONTTI, koko, vari, pks, cv2.LINE_AA)
            y += int(14 + 20 * koko)
        cv2.putText(img, f"viive {self.viive_s:.0f} s", (x0 + 8, y1 - 8), FONTTI, 0.45, (170, 170, 170), 1, cv2.LINE_AA)

    # ---- paasisaankaynti ----------------------------------------------------------------------------------------------
    def piirra(self, ruutu, f_out, heitot):
        """Palauttaa merkityn kopion ruudusta. heitot: Analyysi.kuvat() (lista dict-tilannekuvia)."""
        img = ruutu.copy()
        jalki = int(self.jalki_s * self.fps)
        viivat_f = int(self.viivat_s * self.fps)
        voimassa = [h for h in heitot if h["f_alku"] is not None and f_out >= h["f_alku"]]
        if not voimassa:
            return img
        # viivat: viimeisin heitto, jonka alku on ehtinyt (seuraava heitto korvaa edellisen); vanhenevat viivat_s:n kuluttua
        viimeisin = max(voimassa, key=lambda h: h["f_alku"])
        for h in voimassa:
            if h["f_loppu"] is not None and f_out <= h["f_loppu"] + jalki:
                self._pisteet(img, h, f_out)
        if f_out <= viimeisin["f_alku"] + viivat_f:
            lisa = self._viivat(img, viimeisin, f_out)
            if self.paneeli:
                self._paneeli(img, viimeisin, viimeisin["viivat"], lisa, f_out)
        return img
