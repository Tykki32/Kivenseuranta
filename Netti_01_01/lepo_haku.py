"""Levossa olevan kiven paikan palautus kaukopaassa, kun seuranta katkesi ("hukassa": sweeperit peittivat kiven, eika pysahtymista nahty).

Seuranta (KaukoSeuranta) menettaa kiven usein viimeistaan, kun harjaajat ovat sen paalla: radalta ei saada pysahtymisaikaa eika -paikkaa, vaikka kivi on pelin jalkeen levossa pesassa
nakyvissa. Palautus ei jatka seurantaa, vaan vertaa pesan kivia ennen heittoa ja heiton jalkeen:

  1. ennen heittoa (Saapuminen alussa) kirjataan pesan alueelta heiton kiven varisten kahvojen paikat (kartta_ennen: vanhat kivet);
  2. seurannan katkettua pesaa katsotaan LEPO_VALI ruudun valein; uusi ehdokas on kahvablobi, joka ei ole vanhan kiven paikalla (LEPO_KARTTA_CM) ja joka on kiven viimeisen tunnetun
     paikan edessa (kivi ei peruuta);
  3. ehdokas hyvaksytaan, kun se pysyy paikallaan (LEPO_N havaintoa LEPO_PIDETTY_S:n ajan, hajonta <= LEPO_HAJONTA_CM) ja B-sovitus toteaa sen paikalla graniittirungon (kahvan blobi
     ei yksin riita: harjan pää ja hihan nauha ovat kahvan variset). Lepopaikka on B:n paikka;
  4. jos kelvollisia ehdokkaita on useita, valitaan lahin kuvitellulle pysahtymispisteelle vain, jos se on selvasti lahempana kuin toinen; muuten tulos on epaselva eika paikkaa anneta.

Tulos (heiton h.kauko["lepo_kartta"]): dict(X, Y, f_vahv, ero_cm, ehdokkaita) kaukopaan koordinaateissa tai dict(epaselva=True / ei_loytynyt=True). Aika (pysahtymishetki) jaa tuntemattomaksi:
kartasta nakee, missa kivi on, ei milloin se pysahtyi. Moduuli ei tunne seurantaa muuten kuin Mittarin (kahva, B-sovitus, koordinaatit) kautta."""
import numpy as np

import kivimalli

LEPO_VALI = 15                    # ruutua: pesan katselun vali
LEPO_AIKARAJA_S = 45.0            # haku loppuu taman jalkeen seurannan katkeamisesta
LEPO_KARTTA_CM = 4.0              # ehdokas on vanha kivi, jos se on taman sisalla ennen heittoa olleesta samanvarisesta kahvasta
LEPO_N = 4                        # ehdokkaan perakkaisia havaintoja
LEPO_PIDETTY_S = 3.0              # ... jotka kattavat vahintaan taman ajan
LEPO_HAJONTA_CM = 2.5             # ... ja pysyvat taman sisalla mediaanipaikasta
LEPO_LIITOS_CM = 3.0              # havainto kuuluu ehdokkaalle, jos se on taman sisalla ehdokkaan viimeisesta paikasta
LEPO_NAKYMATON_N = 4              # ehdokas unohdetaan, kun sita ei ole nahty taman monella katselukerralla
LEPO_X_CM = 260.0                 # pesan alue kaukopaan koordinaateissa: |X'| <= taman, Y' valilla
LEPO_Y_CM = (-290.0, 470.0)
LEPO_EDESSA_CM = 30.0             # ehdokas enintaan taman verran viimeisen tunnetun paikan takana (Y' pienenee kiven edetessa)
LEPO_ULOTTUMA_CM = 1200.0         # ehdokas enintaan taman paassa viimeisesta tunnetusta paikasta
LEPO_HIDASTUVUUS_CM_S2 = 7.0      # kuvitellun pysahtymispisteen hidastuvuus (cm / s2)
LEPO_B_PEITTO = 0.5               # ehdokkaan B: graniittia siluetin sisalla vahintaan ...
LEPO_B_KELV = 0.2                 # ... kelvollisia pikseleita vahintaan ...
LEPO_B_RMS = 0.30                 # ... jaannos enintaan ...
LEPO_B_POIK_CM = 12.0             # ... ja enintaan taman verran kahvan paikasta
LEPO_B_YRITYKSIA = 3              # B-yrityksia / ehdokas
LEPO_SELVA_SUHDE = 2.0            # usean ehdokkaan kesken valinta vaatii: toinen on vahintaan taman kertaa kauempana kuvitellusta pysahtymispisteesta


class PesaKartta:
    """Pesan alueen kahvablobit (heiton kiven vari) kaukopaan kamerassa: paikat maatasolla (X', Y', cm) ja ala."""

    def __init__(self, mittari):
        self.m = mittari
        cam = mittari.cam
        P = np.array([(x, y, z) for x in (-LEPO_X_CM, LEPO_X_CM) for y in LEPO_Y_CM for z in (0.0, 22.0)])
        u, v = kivimalli.project_3d(cam.K, cam.R, cam.t, P)
        self.alue = (int(u.min()) - 8, int(v.min()) - 8, int(u.max()) + 8, int(v.max()) + 8)

    def kahvat(self, ruutu, vari, ala_med=None):
        """[(X', Y', ala_px)]: kiven varisten kahvablobien paikat pesan alueella. ala_med: odotettu ala (px / (px / cm)^2), rajaa pois liian pienet ja suuret blobit."""
        m = self.m
        out = []
        for u, v, ala, bb in m.kahva(ruutu, self.alue, vari, taustaero=(vari == "punainen")):
            X, Y = m.oikaistu_maahan(u, v)
            if not (abs(X) <= LEPO_X_CM and LEPO_Y_CM[0] <= Y <= LEPO_Y_CM[1]):
                continue
            if ala_med:
                s = m.skaala(X, Y)
                if not (0.3 * ala_med * s * s <= ala <= 3.0 * ala_med * s * s):
                    continue
            elif not (40 <= ala <= 1500):
                continue
            out.append((float(X), float(Y), int(ala)))
        return out


class LepoHaku:
    """Yhden heiton lepopaikan haku (ks. moduulin kuvaus). viim = (f, X', Y', vX', vY') kiven viimeinen tunnettu paikka ja nopeus (cm / ruutu) seurannasta."""

    def __init__(self, mittari, kartta, vari, ala_med, kartta_ennen, viim, f_alku, fps):
        self.m, self.kartta, self.vari, self.ala_med = mittari, kartta, vari, ala_med
        self.ennen = list(kartta_ennen)
        self.viim = viim
        self.f_alku = int(f_alku)
        self.fps = float(fps)
        self.ehdokkaat = []                        # dict(X, Y, hist=[(f, X, Y)], nahty (katselukerta), b=None | (X, Y), yritetty)
        self.n_kerta = 0
        self.f_edell = -10 ** 9
        self.tulos = None
        v = float(np.hypot(viim[3], viim[4])) * fps                    # cm / s
        if v > 1.0:
            s = v * v / (2.0 * LEPO_HIDASTUVUUS_CM_S2)
            self.ennuste = (viim[1] + viim[3] / np.hypot(viim[3], viim[4]) * s, viim[2] + viim[4] / np.hypot(viim[3], viim[4]) * s)
        else:
            self.ennuste = (viim[1], viim[2])
        self.ulottuma = min(LEPO_ULOTTUMA_CM, max(300.0, v * v / (2.0 * 3.0)))

    # ---- ehdokkaat ---------------------------------------------------------------------------------------------------------------
    def _kelpaa(self, X, Y):
        if any(np.hypot(X - q[0], Y - q[1]) <= LEPO_KARTTA_CM for q in self.ennen):
            return False                                                # vanha kivi (ei liikkunut)
        if Y > self.viim[2] + LEPO_EDESSA_CM:
            return False                                                # kivi ei peruuta
        return float(np.hypot(X - self.viim[1], Y - self.viim[2])) <= self.ulottuma

    def _paivita(self, f, blobit):
        self.n_kerta += 1
        kaytetty = set()
        for X, Y, ala in blobit:
            if not self._kelpaa(X, Y):
                continue
            paras, d_paras = None, None
            for i, c in enumerate(self.ehdokkaat):
                if i in kaytetty:
                    continue
                d = float(np.hypot(X - c["hist"][-1][1], Y - c["hist"][-1][2]))
                if d <= LEPO_LIITOS_CM and (paras is None or d < d_paras):
                    paras, d_paras = i, d
            if paras is None:
                self.ehdokkaat.append(dict(hist=[(f, X, Y)], nahty=self.n_kerta, b=None, yritetty=0))
                kaytetty.add(len(self.ehdokkaat) - 1)
            else:
                c = self.ehdokkaat[paras]
                c["hist"].append((f, X, Y))
                c["nahty"] = self.n_kerta
                kaytetty.add(paras)
        self.ehdokkaat = [c for c in self.ehdokkaat if self.n_kerta - c["nahty"] <= LEPO_NAKYMATON_N]

    def _paikallaan(self, c):
        h = c["hist"][-LEPO_N:]
        if len(h) < LEPO_N or (h[-1][0] - h[0][0]) < LEPO_PIDETTY_S * self.fps:
            return None
        P = np.array([(q[1], q[2]) for q in h])
        med = np.median(P, axis=0)
        if float(np.max(np.hypot(*(P - med).T))) > LEPO_HAJONTA_CM:
            return None
        return float(med[0]), float(med[1])

    def _b(self, ruutu, X, Y):
        """B-sovitus ehdokkaan paikalla: (X, Y), jos kivi (graniittirunko) todetaan, muuten None."""
        v = self.m.valmistele(ruutu, X, Y)
        if v is None:
            return None
        s = self.m.sovita_leike(*v, X, Y)
        if s is None or "virhe" in s:
            return None
        poik = float(np.hypot(s["X"] - X, s["Y"] - Y))
        if s.get("peitto", 1.0) >= LEPO_B_PEITTO and s["kelvolliset"] >= LEPO_B_KELV and s["rms"] <= LEPO_B_RMS and poik <= LEPO_B_POIK_CM:
            return float(s["X"]), float(s["Y"])
        return None

    # ---- paasilmukka -----------------------------------------------------------------------------------------------------------------
    def kasittele(self, f, ruutu):
        """Yksi (yksilollinen) ruutu; palauttaa False, kun haku on valmis (self.tulos)."""
        if self.tulos is not None:
            return False
        if f - self.f_alku > LEPO_AIKARAJA_S * self.fps:
            self._paata(f)
            return False
        if f - self.f_edell < LEPO_VALI:
            return True
        self.f_edell = f
        self._paivita(f, self.kartta.kahvat(ruutu, self.vari, self.ala_med))
        vahvistetut = []
        for c in self.ehdokkaat:
            p = self._paikallaan(c)
            if p is None:
                continue
            if c["b"] is None and c["yritetty"] < LEPO_B_YRITYKSIA:
                c["yritetty"] += 1
                c["b"] = self._b(ruutu, p[0], p[1])
            if c["b"] is not None:
                vahvistetut.append(c)
        if vahvistetut:
            self._ratkaise(f, vahvistetut)
            if self.tulos is not None:
                return False
        return True

    def _ratkaise(self, f, vahvistetut):
        pisteet = [c["b"] for c in vahvistetut]
        if len(vahvistetut) > 1:
            d = sorted(float(np.hypot(p[0] - self.ennuste[0], p[1] - self.ennuste[1])) for p in pisteet)
            erilaiset = max(np.hypot(a[0] - b[0], a[1] - b[1]) for a in pisteet for b in pisteet) > 6.0
            if erilaiset and not (d[1] >= LEPO_SELVA_SUHDE * max(d[0], 1.0)):
                self.tulos = dict(epaselva=True, ehdokkaita=len(vahvistetut), f_vahv=int(f))
                return
            i = int(np.argmin([np.hypot(p[0] - self.ennuste[0], p[1] - self.ennuste[1]) for p in pisteet]))
        else:
            i = 0
        X, Y = pisteet[i]
        self.tulos = dict(X=X, Y=Y, f_vahv=int(f), ero_cm=float(np.hypot(X - self.viim[1], Y - self.viim[2])), ehdokkaita=len(vahvistetut))

    def _paata(self, f):
        self.tulos = dict(ei_loytynyt=True, ehdokkaita=len(self.ehdokkaat), f_vahv=int(f))

    def paata_kesken(self, f):
        """Videon loppu kesken haun."""
        if self.tulos is None:
            self._paata(f)
