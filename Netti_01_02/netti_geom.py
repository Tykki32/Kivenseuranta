"""Geometria ilman top-down-muunnosta: maakoordinaatit (cm) <-> alkuperaisen videoruudun (1920 x 1080) pikselit.

Kuvia EI muunneta: mittaukset tehdaan alkuperaisesta ruudusta ja piirrokset (viivat, pisteet) heijastetaan takaisin alkuperaiseen ruutuun.
Ketju (kalibrointi_netti / kivi_netti.Kamera.muunna kulkee samaa reittia toiseen suuntaan):

    ruutu (x, y) --(- puoliskon alkupiste)--> rajaus (xc, yc) --(90 astetta vastapaivaan)--> (x', y') = (yc, W - 1 - xc)
        --(linssikorjaus k1)--> oikaistu kuva --(kamera K, R, t, taso z)--> maa (X, Y)

Kunkin pelipaan oma koordinaatisto: origo paan pesan keskipiste (T-viiva), Y kohti radan keskikohtaa, X sivusuunta, Z ylos. Toisen paan
koordinaatit: X' = -X, Y' = ETAISYYS_CM - Y (molemmat paat katsovat samaa rataa vastakkaisista suunnista).
"""
import json
import os

import numpy as np

import kalibrointi_netti as kn
import kivimalli
import rata

ETAISYYS_CM = rata.HOUSE_DISTANCE_CM        # pesien keskipisteiden etaisyys (T-viivojen valinen matka)
RUUTU_W, RUUTU_H = 1920, 1080


def toiselle_paalle(X, Y):
    """Yhden paan koordinaatit -> toisen paan koordinaatit (X' = -X, Y' = ETAISYYS_CM - Y); toimii molempiin suuntiin."""
    return -np.asarray(X, float), ETAISYYS_CM - np.asarray(Y, float)


class Puoli:
    """Yhden puoliskon (vasen / oikea) kamera: pikselit <-> maa, suoraan alkuperaisen ruudun koordinaateissa."""

    def __init__(self, nimi, kalib_kansio):
        d = json.load(open(os.path.join(kalib_kansio, f"kalibrointi_{nimi}.json")))
        self.nimi = nimi
        self.K = np.array(d["K"], float)
        self.R = np.array(d["R"], float)
        self.t = np.array(d["t"], float)
        self.k1 = float(d["k1"])
        self.keskipiste = np.array(d["kuvan_keskipiste"], float)
        self.x0, self.y0, self.w, self.h = kn.PUOLET[nimi]
        self._Kinv = np.linalg.inv(self.K)
        self._C = -self.R.T @ self.t                      # kameran paikka
        self._Rt = self.R.T

    # ---- maa -> ruutu --------------------------------------------------------------------------------------------
    def maa_kuvaan(self, X, Y, z=0.0):
        """Maapisteet (X, Y, korkeus z) -> alkuperaisen ruudun pikselit (N x 2, float)."""
        X = np.atleast_1d(np.asarray(X, float))
        Y = np.atleast_1d(np.asarray(Y, float))
        Z = np.broadcast_to(np.asarray(z, float), X.shape)
        u, v = kivimalli.project_3d(self.K, self.R, self.t, np.column_stack([X, Y, Z]))
        d = kn._distort_px(np.column_stack([u, v]), self.k1, self.keskipiste)       # oikaistu -> vaaristynyt (rajaus, kierretty)
        xc = (self.w - 1) - d[:, 1]                                                  # kierron kaanto
        yc = d[:, 0]
        return np.column_stack([xc + self.x0, yc + self.y0])

    # ---- ruutu -> maa --------------------------------------------------------------------------------------------
    def kuva_maahan(self, x, y, z=0.0):
        """Alkuperaisen ruudun pikselit -> maa (X, Y) tasolla z; palauttaa (X, Y) taulukoina."""
        x = np.atleast_1d(np.asarray(x, float))
        y = np.atleast_1d(np.asarray(y, float))
        xc, yc = x - self.x0, y - self.y0
        p = np.column_stack([yc, (self.w - 1) - xc])                                 # kierto 90 astetta vastapaivaan
        u = kn._undistort_px(p, self.k1, self.keskipiste)
        ray = (self._Kinv @ np.column_stack([u, np.ones(len(u))]).T).T
        d = (self._Rt @ ray.T).T
        s = (z - self._C[2]) / d[:, 2]
        return self._C[0] + s * d[:, 0], self._C[1] + s * d[:, 1]

    def puolen_sisalla(self, xy, marg=0):
        xy = np.asarray(xy, float)
        return ((xy[:, 0] >= self.x0 + marg) & (xy[:, 0] < self.x0 + self.w - marg)
                & (xy[:, 1] >= self.y0 + marg) & (xy[:, 1] < self.y0 + self.h - marg))


class Geometria:
    def __init__(self, kalib_kansio="tulokset"):
        self.puolet = {n: Puoli(n, kalib_kansio) for n in ("vasen", "oikea")}

    def __getitem__(self, nimi):
        return self.puolet[nimi]

    @staticmethod
    def toinen(nimi):
        return "oikea" if nimi == "vasen" else "vasen"


def ympyra(r, n=120, X0=0.0, Y0=0.0):
    th = np.linspace(0.0, 2.0 * np.pi, n)
    return X0 + r * np.cos(th), Y0 + r * np.sin(th)
