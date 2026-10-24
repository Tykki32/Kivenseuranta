"""Graniittihaku: kiven runko (graniitin alfa) hakualueella ilman kahvan varia.

Seuranta nojaa kahvan variblobiin (A) ja siita lahtevaan B-sovitukseen (kivi_tarkka). Kahva voi olla piilossa (heittajan kasi, harjaajien harja, punainen kasivarsinauha
sulautuu kahvaan), mutta kiven runko nakyy yleensa ylhaalta: pimea, vareeton graniittirengas jaata vasten. Graniittihaku etsii kivet suoraan rungosta, joten se ei riipu kahvan
varista, koosta eika piilossaolosta:

  1. hakualueen harvennettu ROI poimitaan suoraan alkuperaisesta ruudusta (kuten Saapuminen): remap + taustaero (FG_KYNNYS) halvalla;
  2. vain etualan pikseleille lasketaan graniitin alfa (a = (p - b) . (g - b) / |g - b|^2, kuten kivi_netti.alfa_kartta) ja poikkeama janasta tausta-graniitti (kahvan muovi, iho,
     vaatteet eivat ole graniittia); mustat pikselit (L < L_TUMMA) ja taustan epaluotettavat pikselit (kivi_tarkka.tausta_maskit) ei kelpaa;
  3. kelvolliset graniittipikselit -> yhtenaiset blobit -> blobin ala suhteessa mallin odotettuun nakyvaan graniittialaan sijainnissa (kivi: siluetti miinus ylapinta);
  4. ehdokas = blobin keskipiste maatasolle (z = Z_KESKI_CM) projisoituna.

Ehdokas EI ole viela kivi: harjaajien harmaat varusteet ja ihmisten osat voivat tayttaa pikselitestin. Kiven todentaa B-sovitus (graniittia siluetin sisalla, pieni jaannos) ja liike
(rata etenee tasaisesti), kuten kahvaehdokkailla. Moduuli ei tunne seurantaa."""
import cv2
import numpy as np

import kivi_netti as kv
import kivi_tarkka as kt
import kivimalli

ALFA_RAJA = 0.5                  # graniittipikseli: alfa vahintaan tama
DEV_MAX = kt.DEV_MAX_VARI        # ... ja poikkeama janasta tausta-graniitti (BGR) alle tama (muovi, iho, vaatteet pois)
L_MIN = kt.L_TUMMA               # ... ja ei musta (L* vahintaan tama)
MIN_ALA_PX = 40                  # blobin pienin ala (alkuperaisen kuvan px)
Z_KESKI_CM = 5.0                 # blobin keskipiste projisoidaan tahan korkeuteen (nakyvan graniittiosan painopiste; mitattu: poikkeama kiven akselista 0-4 cm)
ALA_KERROIN = 1.6                # mitattu blobin ala / mallin (siluetti miinus ylapinta) ala: mediaani 1,45-1,77 (kivet kaukopaassa, videot 2-3)
ALA_RAJAT = (0.35, 1.9)          # hyvaksyttava blobin ala suhteessa odotettuun (ALA_KERROIN x malli): osittain peitetty kivi ... kaksi yhdistynytta osaa
RUUTU_X_CM = 100.0               # odotetun alan ruudukko (cm)
RUUTU_Y_CM = 50.0


class GraniittiHaku:
    """Graniittiehdokkaat maatason suorakulmiosta X x Y (cm, kalibrointitaso): ROI haetaan harvennettuna (deci) suoraan alkuperaisesta ruudusta."""

    def __init__(self, mittari, X=(-300.0, 300.0), Y=(450.0, 800.0), deci=2, marg=10):
        self.m = mittari
        cam = mittari.cam
        self.X, self.Y, self.deci = X, Y, int(deci)
        P = np.array([(x, y, z) for x in X for y in Y for z in (0.0, 22.0)])
        u, v = kivimalli.project_3d(cam.K, cam.R, cam.t, P)
        kuvain = mittari.kuvain
        x0, y0 = max(int(u.min()) - marg, 0), max(int(v.min()) - marg, 0)
        x1, y1 = min(int(u.max()) + marg, kuvain.w), min(int(v.max()) + marg, kuvain.h)
        d = self.deci
        self.x0, self.y0 = x0, y0
        sl = (slice(y0, y1, d), slice(x0, x1, d))
        self.mx = np.ascontiguousarray(kuvain.mx[sl])
        self.my = np.ascontiguousarray(kuvain.my[sl])
        self.ok = self.mx.size > 100
        self.tausta = np.ascontiguousarray(cam.mode[sl])
        self._b = self.tausta.astype(np.float32)
        g = kv.graniitti_vari(cam).astype(np.float32)
        dvec = g[None, None, :] - self._b
        self._dvec = dvec
        self._dd = np.sum(dvec * dvec, axis=2)
        jaa, vari = kt.tausta_maskit(cam)
        # taustan puolesta luotettavat pikselit, joilla graniitti erottuu taustasta (|g - b|^2 > 2500, kuten alfa_kartta) ja jotka ovat kaistalla
        self._kelpaa = np.ascontiguousarray((jaa | vari)[sl]) & (self._dd > 2500.0) & (np.ascontiguousarray(cam.lane[sl]) > 0)
        self._odotus = {}

    # ---- odotettu ala ---------------------------------------------------------------------------------------------------------------
    def odotettu_ala(self, X, Y):
        """Nakyvan graniitin odotettu ala (alkuperaisen kuvan px) sijainnissa (X, Y): ALA_KERROIN x (mallin siluetti, jossa alfa > 0,5, miinus ylapinta). Ruudukossa (valimuisti)."""
        k = (int(round(X / RUUTU_X_CM)), int(round(Y / RUUTU_Y_CM)))
        v = self._odotus.get(k)
        if v is None:
            Xc, Yc = k[0] * RUUTU_X_CM, k[1] * RUUTU_Y_CM
            cam, runko = self.m.cam, self.m.runko
            win = kt.ikkuna(cam, runko.hull(cam, Xc, Yc), 8)
            c, wt = kt.malli_kuva(cam, runko, win, Xc, Yc)
            v = ALA_KERROIN * float(np.sum((c > 0.5) & (wt > 0.5)))
            self._odotus[k] = v
        return v

    # ---- graniittimaski ---------------------------------------------------------------------------------------------------------------
    def maski(self, roi):
        """Kelvolliset graniittipikselit (bool, ROI:n koko) harvennetusta ROI:sta. Raskaat laskut vain etualan pikseleille (taustaero > FG_KYNNYS)."""
        ero = cv2.absdiff(roi, self.tausta)
        ero = np.maximum(np.maximum(ero[..., 0], ero[..., 1]), ero[..., 2])
        fg = (ero > kv.FG_KYNNYS) & self._kelpaa
        G = np.zeros(fg.shape, bool)
        idx = np.nonzero(fg)
        if len(idx[0]) == 0:
            return G
        p = roi[idx].astype(np.float32)
        b = self._b[idx]
        dv = self._dvec[idx]
        e = p - b
        a = np.sum(e * dv, axis=1) / np.maximum(self._dd[idx], 1.0)
        res = e - np.clip(a, 0.0, 1.0)[:, None] * dv
        dev = np.sqrt(np.sum(res * res, axis=1))
        L = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)[..., 0][idx].astype(np.float32) * (100.0 / 255.0)
        G[idx] = (a > ALFA_RAJA) & (dev < DEV_MAX) & (L >= L_MIN)
        return G

    def blobit(self, roi):
        """Graniittiblobit ROI:sta: [dict(u, v, X, Y, ala, odotus, suhde, bbox)] (u, v oikaistun kuvan pikseleina); vain hakualueen sisalla, ala >= MIN_ALA_PX."""
        d = self.deci
        G = cv2.morphologyEx(self.maski(roi).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        n, lab, st, cen = cv2.connectedComponentsWithStats(G)
        out = []
        for k in range(1, n):
            ala = int(st[k, cv2.CC_STAT_AREA]) * d * d
            if ala < MIN_ALA_PX:
                continue
            u, v = self.x0 + d * float(cen[k][0]), self.y0 + d * float(cen[k][1])
            X, Y = kv.maan_paikka(self.m.cam, u, v, z=Z_KESKI_CM)
            if not (self.X[0] <= X <= self.X[1] and self.Y[0] <= Y <= self.Y[1]):
                continue
            odo = self.odotettu_ala(X, Y)
            out.append(dict(u=u, v=v, X=float(X), Y=float(Y), ala=ala, odotus=odo, suhde=ala / max(odo, 1.0), bbox=tuple(int(q) * d for q in st[k, :4])))
        return out

    def ehdokkaat(self, ruutu):
        """Kivikandidaatit alkuperaisesta ruudusta: blobit, joiden ala sopii kivelle (ALA_RAJAT), keskipiste (X, Y) maatasolla."""
        if not self.ok:
            return []
        roi = cv2.remap(ruutu, self.mx, self.my, cv2.INTER_LINEAR)
        return [c for c in self.blobit(roi) if ALA_RAJAT[0] <= c["suhde"] <= ALA_RAJAT[1]]
