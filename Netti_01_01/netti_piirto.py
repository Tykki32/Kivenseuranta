"""Piirrokset alkuperaiseen (1920 x 1080) videoruutuun, ei top-down-kuvaa.

Heittopaa (vasen / oikea puolisko): kiven keskikohta pisteina jaan tasolla (kiven akselin kohta jaalla, z = PISTE_Z = 0). Koko tahan mennessa mitattu pistejono (liuku ja irroituksen
jalkeinen lento puoliskon loppuun) piirretaan kaikki kerralla heti kun heiton alku on tulosteessa: tulosteen kuva on analyysia VIIVE_S jäljessa, joten pistejono on yleensa valmis ennen
kuin kivi liikahtaa, ja kivi nakyy seuraavan valmiiksi piirrettya jonoa (nykyinen paikka on rengas). Jos analyysi on tulosteen perassa, pisteet ilmestyvat sita mukaa kuin ne valmistuvat.
Irroituksen kohta (viivaestimaatin irroituspiste) merkitaan mustalla ympyralla. Pistejoukko jaa nakyviin JALKI_S sekuntia heiton loputtua.
Toinen paa: liukulinja ja irroituslinja heti heiton alusta (tulosteen kuva on analyysia VIIVE_S jäljessa, joten estimaatti ehtii valmistua ennen kuin kivi liikahtaa tulosteessa);
linjat jaavat nakyviin seuraavaan heittoon tai VIIVAT_S sekuntiin asti.

Linjat ovat suoria (liukulinja: heiton liu'un keskimaarainen suunta hackista irroituspisteeseen; irroituslinja: vapaan lennon suunta irroitushetkella) ja
ne jatketaan toisen paan T-viivalle; todellinen rata kaartaa (kierre), joten kiven rata poikeaa irroituslinjasta sita enemman mita pidemmalle mennaan.

Tietopaneeli (ylapalkin vapaa alue): kivi (vari ja numero), merkki (T-viivalla keskiviivasta), liuku ja heitto (merkista, + = kierteesta ulospain; ilman merkkia keskiviivasta), irroitus (heitto - liuku kierteen suunnan mukaan), kierre (kierrokset hog-hog-valilla),
irroituksen etaisyys hog-viivaan, nopeudet ja hidastuvuus (netti_seuranta.heiton_suureet).

Piirto tapahtuu ruudun paalle (ruutu muuttuu); piirtaja lukee analyysin heittoja (netti_seuranta.Heitto) suoraan, joten sita kutsutaan yhdesta saikeesta.
"""
import cv2
import numpy as np

import netti_geom as ng
import netti_seuranta as ns

PISTE_Z = 0.0                            # pisteet kiven keskikohtaan jaan tasolle (ei kahvan korkeudelle)
VARI_PISTE = (60, 230, 60)               # BGR: kirkas vihrea
VARI_PISTE_REUNA = (20, 20, 20)
VARI_NYKYINEN = (40, 220, 255)           # nykyinen paikka: keltainen rengas
VARI_LIUKU = (0, 140, 255)               # oranssi
VARI_IRROITUS = (255, 60, 220)           # magenta
VARI_KIERRE = (255, 215, 110)            # vaalea sininen
VARI_IRROITUSPISTE = (0, 0, 0)           # irroituksen kohdan ympyra: musta (valkoinen reuna nakyy tummallakin jaalla)
IRROITUSPISTE_R = 11                     # ympyran sade (px) ja viivan paksuus
VARI_PANEELI = (30, 30, 30)
VARI_NIMI = {"keltainen": (60, 220, 255), "punainen": (120, 120, 255)}        # kiven nimen vari kuvassa (BGR)
VARI_HARMAA = (170, 170, 170)
VARI_VALKOINEN = (255, 255, 255)
FONTTI = cv2.FONT_HERSHEY_SIMPLEX
Y_ALKU_KAUKO = -170.0                    # linjan piirto kaukopaassa: Y' tasta ...
Y_LOPPU_KAUKO = 1100.0                   # ... tahan (cm T-viivasta)
IRROITUSPISTE_PAKSUUS = 3
PANEELI_PAIKKA = (284, 6)                # ylapalkin vapaa alue (x, y): ei logoa eika tulostaulua
PANEELI_KORKEUS_MAX = 204                # 9 riviä (Merkki mukana): 12 + 21 * 9 = 201
RIVI_PX = 21                             # paneelin riviväli
PANEELI_KOKO = 0.56
KELLO_Y = 200                            # joukkueiden kellon laatikko: kuvan keskella mainosgrafiikan (LeaseDeal / Hyvinkaan Liikenne, alareuna y ~ 193) alla (vaakasuunnassa keskitetty)
KELLO_LEVEYS = 270                       # laatikon vahimmaisleveys
PANEELI_OTSIKKO_KOKO = 0.62

_KIRJAIN_PERUS = {"ä": "a", "ö": "o", "å": "a", "Ä": "A", "Ö": "O", "Å": "A"}


# ---------------------------------------------------------------------------------------------------------------------
# PERUSPIIRROT
# ---------------------------------------------------------------------------------------------------------------------
def _leikatut_segmentit(pts, h, w, katkoviiva=False):
    """Murtoviivan segmentit rajattuna kuvaan (h x w): [2 x 2 int32 -taulukko]; katkoviiva: joka toinen segmentti. Riippuu vain pisteista ja kuvan koosta, joten Piirtaja valimuistittaa sen."""
    pts = np.round(pts).astype(np.int32)
    segs = []
    for a, b in zip(pts[:-1], pts[1:]):
        ok, p, q = cv2.clipLine((0, 0, w, h), (int(a[0]), int(a[1])), (int(b[0]), int(b[1])))
        if ok:
            segs.append(np.array([p, q], np.int32))
    return segs[::2] if katkoviiva else segs


def _piirra_segmentit(img, segs, vari, paksuus, taustavari=(255, 255, 255)):
    """Valmiiksi rajatut segmentit (_leikatut_segmentit): valkoinen tausta + vari, yksi polylines-kutsu per kerros."""
    if segs:
        cv2.polylines(img, segs, False, taustavari, paksuus + 2, cv2.LINE_AA)
        cv2.polylines(img, segs, False, vari, paksuus, cv2.LINE_AA)


def _viiva_kuvaan(img, pts, vari, paksuus, katkoviiva=False, taustavari=(255, 255, 255)):
    """Murtoviiva pikselikoordinaateissa; valkoinen tausta + vari -> nakyy jaalla ja renkaiden paalla. katkoviiva: lyhyet patkat. Segmentit rajataan kuvaan (clipLine) ja
    piirretaan yhdella polylines-kutsulla per kerros."""
    h, w = img.shape[:2]
    _piirra_segmentit(img, _leikatut_segmentit(pts, h, w, katkoviiva), vari, paksuus, taustavari)


def teksti_fi(img, txt, org, koko=0.55, vari=VARI_VALKOINEN, paksuus=1):
    """cv2.putText ei piirra ä / ö / å: piirretaan perusmerkki ja sen paalle pisteet (ä, ö) tai rengas (å). Palauttaa tekstin leveyden (px)."""
    if not any(c in _KIRJAIN_PERUS for c in txt):
        cv2.putText(img, txt, (int(org[0]), int(org[1])), FONTTI, koko, vari, paksuus, cv2.LINE_AA)
        return cv2.getTextSize(txt, FONTTI, koko, paksuus)[0][0]
    x, y = float(org[0]), float(org[1])
    h_iso = cv2.getTextSize("H", FONTTI, koko, paksuus)[0][1]
    jono = ""
    for c in txt + "\0":
        if c in _KIRJAIN_PERUS or c == "\0":
            if jono:
                cv2.putText(img, jono, (int(round(x)), int(y)), FONTTI, koko, vari, paksuus, cv2.LINE_AA)
                x += cv2.getTextSize(jono, FONTTI, koko, paksuus)[0][0]
                jono = ""
            if c == "\0":
                break
            perus = _KIRJAIN_PERUS[c]
            w = cv2.getTextSize(perus, FONTTI, koko, paksuus)[0][0]
            cv2.putText(img, perus, (int(round(x)), int(y)), FONTTI, koko, vari, paksuus, cv2.LINE_AA)
            ylos = h_iso * (1.22 if c.isupper() else 0.95)
            r = max(1, int(round(0.075 * h_iso * (1 + 0.3 * (paksuus - 1)))))
            if c in "äÄöÖ":
                for dx in (0.28, 0.72):
                    cv2.circle(img, (int(round(x + dx * w)), int(round(y - ylos))), r, vari, -1, cv2.LINE_AA)
            else:
                cv2.circle(img, (int(round(x + 0.5 * w)), int(round(y - ylos))), r + 1, vari, 1, cv2.LINE_AA)
            x += w
        else:
            jono += c
    return int(round(x - float(org[0])))


def _teksti(img, txt, org, koko=0.55, vari=VARI_VALKOINEN, paksuus=1, tausta=True):
    (w, h), pohja = cv2.getTextSize(txt, FONTTI, koko, paksuus)
    x, y = int(org[0]), int(org[1])
    if tausta:
        cv2.rectangle(img, (x - 3, y - h - 3), (x + w + 3, y + pohja + 2), VARI_PANEELI, -1)
    cv2.putText(img, txt, (x, y), FONTTI, koko, vari, paksuus, cv2.LINE_AA)
    return w, h


def mediaani5(P):
    """Pisteiden (N x 2) liukuva mediaani (5), reunoilla katkaistu ikkuna: pehmeat sovitukset voivat sisaltaa yksittaisen ulkolaisen."""
    n = len(P)
    if n < 3:
        return P
    out = P.copy()
    if n >= 5:
        out[2:n - 2] = np.median(np.lib.stride_tricks.sliding_window_view(P, 5, axis=0), axis=2)
        reunat = (0, 1, n - 2, n - 1)
    else:
        reunat = range(n)
    for i in reunat:
        a, b = max(0, i - 2), min(n, i + 3)
        out[i] = np.median(P[a:b], axis=0)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# NUMEROT PANEELIIN: suomalainen desimaalipilkku
# ---------------------------------------------------------------------------------------------------------------------
def _luku(x, des=1, merkki=False):
    s = f"{x:+.{des}f}" if merkki else f"{x:.{des}f}"
    if float(s) == 0.0:                                   # -0,0 -> 0,0
        s = s.lstrip("+-") if not merkki else "+" + s.lstrip("+-")
    return s.replace(".", ",")


def paneelin_rivit(h, s, kierre_paalla=True):
    """Paneelin rivit [(teksti, vari, lisateksti)] heitolle h (netti_seuranta.Heitto) ja sen suureille s (netti_seuranta.heiton_suureet tai None). Muoto kuten kayttajan
    maaritelma: Kivi, Merkki (maalimerkin paikka T-viivalla keskiviivasta, + = heittajan oikealle; "ei", ellei merkkia loytynyt; harmaa teksti: "reunalla" = kuvan reuna katkaisee
    harjanpaan, "2 harjaa" = kaksi kilpailevaa), Liuku, Heitto (suorien osumat kaukopaan T-viivalla MERKISTA, jos maalimerkki tunnetaan: + = kierteesta ulospain eli myotapaivaan pyorivalla kivella vasemmalle,
    ? kunnes kierron suunta tiedetaan; muuten keskiviivasta, + = heittajan oikealle; viite nakyy Liuku-rivin harmaassa tekstissa), Irroitus (heitto - liuku, + = kaarron vastaiseen
    suuntaan), Kierre (- = vastapaivaan), Irroitus (m ennen hog-viivaa, kiven etureunasta), Nopeus, Hidastuvuus. lisateksti: pieni harmaa huomautus (alustava)."""
    vari_nimi = h.vari.capitalize()
    nro = h.kivi_nro if h.kivi_nro is not None else h.tunnus
    rivit = [(f"Kivi: {vari_nimi} {nro}", VARI_VALKOINEN, "" if (s is not None and s["vakaa"]) else "alustava")]
    if s is None:
        return rivit + [("analysoidaan ...", VARI_HARMAA, "")]
    irr = f"{_luku(s['irroitus_cm'], 0, True)}cm" if s["irroitus_cm"] is not None else "?"
    kierre = _luku(s["kierre"], 1) if s["kierre"] is not None else "?"
    kierre_lisa = ""
    if s["kierre"] is None and kierre_paalla:
        kierre_lisa = "mitataan" if h.kierto_tila in (None, "kaynnissa") else "ei saatu"
    elif s["kierre"] is not None:
        kauko_kesken = h.kauko is None and h.saapuminen in (None, "haku", "loytyi") and h.kauko_kierto_tila != "valmis"            # kaukopaan mittaus tarkentaa viela
        sg = f"±{_luku(s['kierre_sigma'], 1)}" if s.get("kierre_sigma") is not None else ""                                     # epavarmuus kierroksina (1 sigma)
        if s.get("kierre_varma"):
            kierre_lisa = sg                                                                                                  # varma
        elif kauko_kesken or not s["kierre_mitattu"]:
            kierre_lisa = "alustava"
        else:
            kierre_lisa = (sg + " epävarma").strip()
    if s.get("merkki_T_cm") is not None:            # merkki tiedossa: liuku ja heitto merkista, + = kierteesta ulospain (myotapaivaan: vasemmalle); ? kunnes kierron suunta tiedetaan
        liuku = f"{_luku(s['liuku_merkista_cm'], 0, True)}cm" if s["liuku_merkista_cm"] is not None else "?"
        heitto = f"{_luku(s['heitto_merkista_cm'], 0, True)}cm" if s["heitto_merkista_cm"] is not None else "?"
        viite = "merkistä, + ulos"
    else:                                           # ei merkkia: keskiviivasta, + = heittajan oikealle
        liuku, heitto, viite = f"{_luku(s['liuku_cm'], 0, True)}cm", f"{_luku(s['heitto_cm'], 0, True)}cm", "keskiviivasta"
    if s.get("merkki_ehdokkaita") is not None:      # Merkki: maalimerkin paikka kaukopaan T-viivalla keskiviivasta (+ = heittajan oikealle); ei rivia, jos merkkia ei haeta
        if s["merkki_T_cm"] is not None:
            rivit.append((f"Merkki: {_luku(s['merkki_T_cm'], 0, True)}cm", VARI_VALKOINEN, "reunalla" if s["merkki_reunalla"] else ("2 harjaa" if s["merkki_ehdokkaita"] > 1 else "keskiviivasta")))
        else:
            rivit.append(("Merkki: ei", VARI_HARMAA, ""))
    rivit += [
        (f"Liuku: {liuku}", VARI_LIUKU, viite),
        (f"Heitto: {heitto}", VARI_IRROITUS, ""),
        (f"Irroitus: {irr}", VARI_VALKOINEN, ""),
        (f"Kierre: {kierre}", VARI_KIERRE, kierre_lisa),
        (f"Irroitus: {_luku(s['irr_hog_m'], 1)}m", VARI_VALKOINEN, ""),
        (f"Nopeus: Ennen {_luku(s['v_ennen'], 2)} m/s Jälkeen {_luku(s['v_jalkeen'], 2)} m/s", VARI_VALKOINEN, ""),
        (f"Hidastuvuus: {_luku(s['hidastuvuus'], 2) + ' m/s^2' if (s['vakaa'] or s['vapaa_matka_cm'] >= 250.0) and 0.0 <= s['hidastuvuus'] <= 0.25 else '?'}", VARI_VALKOINEN, ""),     # lyhyestä vapaasta lennosta tai kohinaisesta mittauksesta (negatiivinen tai > 0,25) ei luotettavaa arvoa
    ]
    return rivit


def _aika(s):
    """Aika muodossa m:ss,d (suomalainen desimaalipilkku)."""
    s = max(float(s), 0.0)
    m = int(s // 60)
    return f"{m}:{s - 60 * m:04.1f}".replace(".", ",")


def kello_rivit(kt):
    """Joukkueiden kellon rivit [(teksti, vari, lisateksti)] (netti_seuranta.Analyysi.kello_nyt): miettimisaika joukkueittain yhteensa (kayvalla joukkueella kulunut aika mukana) ja
    heittokohtainen aika: kayva kello -> kuluva aika, muuten viimeisimman heiton aika. Kello kay, kun edellisen heiton kivet ovat pysahtyneet, ja pysahtyy kun heitto lahtee liikkeelle;
    paan ensimmaiselle heitolle ei aikaa."""
    rivit = [("Kello", VARI_VALKOINEN, f"{kt['kay'].capitalize()} käy" if kt["kay"] else "seis")]
    for vari in ("keltainen", "punainen"):
        rivit.append((f"{vari.capitalize()}: {_aika(kt['yht'].get(vari, 0.0))}", VARI_NIMI[vari], "käy" if kt["kay"] == vari else ""))
    if kt["kay"] is not None:
        rivit.append((f"Heitto: {_aika(kt['juokseva_s'])}", VARI_VALKOINEN, "kuluva aika"))
    elif kt["viim"] is not None:
        h = kt["viim"]
        rivit.append((f"Heitto: {_aika(kt['viim_aika_s']) if kt['viim_aika_s'] is not None else '-'}", VARI_VALKOINEN, f"{h.vari.capitalize()} {h.kivi_nro if h.kivi_nro is not None else ''}".rstrip()))
    return rivit


class Piirtaja:
    def __init__(self, geom, fps=30.0, jalki_s=10.0, viivat_s=60.0, piste_z=PISTE_Z, paneeli=True, viive_s=10.0, kello=True, kello_paikka=None):
        self.g = geom
        self.kello = kello
        self.kello_paikka = kello_paikka            # (x, y) tai None: kuvan keskella mainosgrafiikan alla
        self.fps = float(fps)
        self.jalki_s = jalki_s
        self.viivat_s = viivat_s
        self.z = piste_z
        self.paneeli = paneeli
        self.viive_s = viive_s
        self._pisteet_vlk = {}                      # heiton tunnus -> (avain, F, uv, interp): siloteltu pistejono kuvapisteina (lasketaan kun aineisto muuttuu)
        self._paneeli_vlk = None                    # (avain, kuva, maski): valmiiksi piirretty paneeli
        self._kello_vlk = None                      # sama kellon laatikolle
        self.viive_nyt = None                       # reaaliaika (netti_live): kuinka monta sekuntia myohemmin kuva nakyy kuin sen oma aika + viive (None = ei tietoa: toistotilat); piirtosaie asettaa ennen piirra()
        self.puskuri_s = None                       # esilatauspuskurissa oleva video (s) tai None
        self._viive_vlk = None                      # Viive-rivin laatikko
        self._viivat_vlk = None                     # (avain, dict): linjojen kuvapisteet

    # ---- pisteet heittopaassa ------------------------------------------------------------------------------------
    def _pistejono(self, h):
        """Heiton pisteet kuvapisteina: mediaani-silotettu, aukot (kasi peittaa, B ei kelpaa) lineaarisesti taytettyina. Valimuisti: lasketaan uudelleen vain kun naytteet
        tai B-tulokset muuttuvat."""
        avain = (len(h.naytteet), h.b_laskuri, h.f_alku)
        v = self._pisteet_vlk.get(h.tunnus)
        if v is not None and v[0] == avain:
            return v[1:]
        pts = h.pisteet()
        if not pts:
            r = (np.zeros(0), np.zeros((0, 2)), [], np.zeros((0, 2)))
        else:
            F = np.array([p[0] for p in pts])
            P = mediaani5(np.array([[p[1], p[2]] for p in pts]))
            # aukkojen lineaarinen taytto <= 12 ruutua: pisteet nakyvat yhtenaisena jonona
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
            Pf = np.array(Pf)
            r = (np.array(Ff), self.g[h.puoli].maa_kuvaan(Pf[:, 0], Pf[:, 1], self.z), interp, Pf)
        self._pisteet_vlk[h.tunnus] = (avain,) + r
        return r

    @staticmethod
    def _liuku_n(h, P):
        """Liu'un pisteiden lukumaara pistejonossa: indeksit < n ovat liukua, n - 1 on viivaestimaatin irroituspistetta lahin piste. None, ellei estimaattia (viel) ole."""
        v = h.viivat
        if v is None or "irr_X" not in v or len(P) == 0:
            return None
        return int(np.argmin(np.hypot(P[:, 0] - v["irr_X"], P[:, 1] - v["irr_Y"]))) + 1

    def _irroituspiste(self, img, h):
        """Musta ympyra irroituksen kohtaan (viivaestimaatin irroituspiste jaan tasolla); valkoinen reuna, jotta se nakyy myos tummalla jaalla."""
        v = h.viivat
        u, w = (float(q[0]) for q in np.asarray(self.g[h.puoli].maa_kuvaan([v["irr_X"]], [v["irr_Y"]], self.z)).T)
        c = (int(round(u)), int(round(w)))
        cv2.circle(img, c, IRROITUSPISTE_R + IRROITUSPISTE_PAKSUUS // 2 + 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(img, c, IRROITUSPISTE_R, VARI_IRROITUSPISTE, IRROITUSPISTE_PAKSUUS, cv2.LINE_AA)

    def _pisteet(self, img, h, f_out):
        """Koko tahan mennessa mitattu pistejono (liuku + irroituksen jalkeinen lento) kaikki kerralla heti kun heiton alku on tulosteessa: kivi nakyy seuraavan valmiiksi piirrettya
        jonoa (analyysi on tulosteen edella, joten jono on yleensa valmis; jos ei, pisteet ilmestyvat sita mukaa kuin ne valmistuvat). Nykyinen paikka (tulosteen ruutu) renkaana ja
        irroituksen kohta mustalla ympyralla, kun viivaestimaatti on valmis."""
        Ff, uv, interp, P = self._pistejono(h)
        n_live = int(np.searchsorted(Ff, f_out, side="right"))
        if n_live == 0:
            return                                                          # heitto ei ole viela alkanut tulosteessa
        c = np.round(uv).astype(np.int32)
        for i in range(len(Ff)):
            if i == n_live - 1:
                continue                                                    # nykyinen paikka piirretaan viimeisena
            (u, v), ip = c[i], interp[i]
            cv2.circle(img, (int(u), int(v)), 3 if not ip else 2, VARI_PISTE_REUNA, -1, cv2.LINE_AA)
            cv2.circle(img, (int(u), int(v)), 2 if not ip else 1, VARI_PISTE, -1, cv2.LINE_AA)
        if self._liuku_n(h, P) is not None:
            self._irroituspiste(img, h)
        u, v = c[n_live - 1]
        if f_out <= (h.f_loppu if h.f_loppu is not None else 10 ** 9) + 2:       # kivi viela kuva-alalla: nykyinen paikka renkaana ja kiven nimi (vari + numero) vieressa
            cv2.circle(img, (int(u), int(v)), 7, VARI_PISTE_REUNA, 3, cv2.LINE_AA)
            cv2.circle(img, (int(u), int(v)), 7, VARI_NYKYINEN, 1, cv2.LINE_AA)
            if h.kivi_nro is not None:
                _teksti(img, f"{h.vari.capitalize()} {h.kivi_nro}", (int(u) + 14, int(v) - 12), 0.5, VARI_NIMI.get(h.vari, VARI_VALKOINEN), 1)
        else:
            cv2.circle(img, (int(u), int(v)), 4, VARI_PISTE_REUNA, -1, cv2.LINE_AA)
            cv2.circle(img, (int(u), int(v)), 3, VARI_PISTE, -1, cv2.LINE_AA)

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

    def _viivat_data(self, h, kauko):
        """Linjojen kuvapisteet (kaukopaan kuva) ja T-viivan osumat; valimuisti estimaatin arvoilla (estimaatti muuttuu harvoin, projektio on kallis)."""
        v = h.viivat
        avain = (kauko, v["liuku_a"], v["liuku_b"], v["irr_X"], v["irr_Y"], v["irr_suunta_deg"], bool(v.get("vakaa", True)))
        if self._viivat_vlk is not None and self._viivat_vlk[0] == avain:
            return self._viivat_vlk[1]
        L = ng.ETAISYYS_CM
        a, b = v["liuku_a"], v["liuku_b"]
        pts, _ = self._suora_pikseleina(kauko, lambda Y: a + b * Y)
        Xr, Yr, th = v["irr_X"], v["irr_Y"], np.tan(np.radians(v["irr_suunta_deg"]))
        pts2, _ = self._suora_pikseleina(kauko, lambda Y: Xr + th * (Y - Yr))
        d = dict(pts=pts, pts2=pts2, XT_liuku=a + b * L, XT_irr=Xr + th * (L - Yr), alustava=not v.get("vakaa", True),
                 uv_liuku=self._kaukopiste(kauko, a + b * L, L)[0], uv_irr=self._kaukopiste(kauko, Xr + th * (L - Yr), L)[0])
        self._viivat_vlk = (avain, d)
        return d

    def _viivat(self, img, h):
        if h.viivat is None:
            return
        kauko = ng.Geometria.toinen(h.puoli)
        d = self._viivat_data(h, kauko)
        koko = img.shape[:2]
        if d.get("segmentit_koko") != koko:                                                  # kuvaan rajatut segmentit lasketaan kerran per estimaatti (eivat muutu ruudusta toiseen)
            d["segmentit"] = (_leikatut_segmentit(d["pts"], *koko), _leikatut_segmentit(d["pts2"], *koko, katkoviiva=d["alustava"]))
            d["segmentit_koko"] = koko
        _piirra_segmentit(img, d["segmentit"][0], VARI_LIUKU, 3)
        _piirra_segmentit(img, d["segmentit"][1], VARI_IRROITUS, 3)                           # alustava = katkoviiva
        # T-viivan leikkauspisteet (piste, kun se on kuva-alan sisalla); viivoilla ei ole tekstia: luvut ovat tietopaneelissa
        px0, py0, pw, ph = (ng.kn.PUOLET[kauko])
        for uv, vari in ((d["uv_liuku"], VARI_LIUKU), (d["uv_irr"], VARI_IRROITUS)):
            c = (int(round(uv[0])), int(round(uv[1])))
            if px0 + 6 <= c[0] < px0 + pw - 6 and py0 + 6 <= c[1] < py0 + ph - 6:
                cv2.circle(img, c, 9, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(img, c, 7, vari, -1, cv2.LINE_AA)

    # ---- tietopaneeli -----------------------------------------------------------------------------------------------
    def _paneeli_kuva(self, rivit, vlk="_paneeli_vlk", leveys_min=330):
        """Paneeli valmiiksi piirrettyna (tausta + teksti) ja tekstin maski: (kuva, maski). Valimuisti (attribuutti vlk): piirretaan uudelleen vain kun rivit muuttuvat."""
        avain = tuple(rivit)
        v = getattr(self, vlk)
        if v is not None and v[0] == avain:
            return v[1], v[2]
        leveys = leveys_min
        for i, (txt, vari, lisa) in enumerate(rivit):
            koko = PANEELI_OTSIKKO_KOKO if i == 0 else PANEELI_KOKO
            leveys = max(leveys, 8 + cv2.getTextSize(txt, FONTTI, koko, 2 if i == 0 else 1)[0][0] + (90 if lisa else 0) + 12)
        korkeus = min(PANEELI_KORKEUS_MAX, 12 + RIVI_PX * len(rivit))
        kuva = np.full((korkeus, leveys, 3), VARI_PANEELI, np.uint8)
        maski = np.zeros((korkeus, leveys), np.uint8)
        y = 25
        for i, (txt, vari, lisa) in enumerate(rivit):
            koko, pks = (PANEELI_OTSIKKO_KOKO, 2) if i == 0 else (PANEELI_KOKO, 1)
            w = teksti_fi(kuva, txt, (8, y), koko, vari, pks)
            teksti_fi(maski, txt, (8, y), koko, 255, pks)
            if lisa:
                teksti_fi(kuva, lisa, (8 + w + 10, y), 0.42, VARI_HARMAA, 1)
                teksti_fi(maski, lisa, (8 + w + 10, y), 0.42, 255, 1)
            y += RIVI_PX if i else RIVI_PX + 3
        setattr(self, vlk, (avain, kuva, maski > 0))
        return kuva, maski > 0

    @staticmethod
    def _laatikko(img, kuva, maski, paikka):
        x0, y0 = paikka
        hh, ww = kuva.shape[:2]
        roi = img[y0:y0 + hh, x0:x0 + ww]
        if roi.shape[:2] != (hh, ww):
            return
        cv2.addWeighted(kuva, 0.72, roi, 0.28, 0, roi)         # tausta puoliksi lapinakyva
        cv2.copyTo(kuva, maski.view(np.uint8), roi)             # teksti taysin peittavana (sama kuin np.copyto(roi, kuva, where=maski), ~50 x nopeampi)

    def _paneeli(self, img, h, kierre_paalla=True):
        rivit = paneelin_rivit(h, ns.heiton_suureet(h), kierre_paalla)
        kuva, maski = self._paneeli_kuva([(t, v, l) for t, v, l in rivit])
        self._laatikko(img, kuva, maski, PANEELI_PAIKKA)

    def _viive(self, img, kello_leveys=None):
        """'Viive: x,x s' kellon laatikon ylapuolelle: kuinka paljon ollaan myohassa (analyysi / piirto ei pysy reaaliajassa; puskuri kirii kiinni). Valkoinen < 1 s, keltainen < 3 s, punainen yli."""
        v = float(self.viive_nyt)
        vari = VARI_VALKOINEN if v < 1.0 else ((0, 200, 255) if v < 3.0 else (60, 60, 255))
        lisa = f"puskuri {self.puskuri_s:.0f} s" if self.puskuri_s is not None else ""
        kuva, maski = self._paneeli_kuva([(f"Viive: {v:.1f} s".replace(".", ","), vari, lisa)], "_viive_vlk", KELLO_LEVEYS)
        if self.kello_paikka is not None:
            x0, y0 = self.kello_paikka
        else:
            x0, y0 = (img.shape[1] - kuva.shape[1]) // 2, KELLO_Y
        self._laatikko(img, kuva, maski, (x0, max(0, y0 - kuva.shape[0] - 4)))

    def _kello(self, img, kt):
        kuva, maski = self._paneeli_kuva(kello_rivit(kt), "_kello_vlk", KELLO_LEVEYS)
        paikka = self.kello_paikka if self.kello_paikka is not None else ((img.shape[1] - kuva.shape[1]) // 2, KELLO_Y)          # oletus: keskella mainosgrafiikan alla
        self._laatikko(img, kuva, maski, paikka)

    # ---- paasisaankaynti ----------------------------------------------------------------------------------------------
    def piirra(self, ruutu, f_out, an):
        """Piirtaa ruudun paalle (palauttaa saman ruudun). an: netti_seuranta.Analyysi (heitot-lista luetaan suoraan)."""
        jalki = int(self.jalki_s * self.fps)
        viivat_f = int(self.viivat_s * self.fps)
        voimassa = [h for h in list(an.heitot) if h.f_alku is not None and f_out >= h.f_alku]
        if self.paneeli and self.viive_nyt is not None:
            self._viive(ruutu)
        if self.paneeli and self.kello and hasattr(an, "kello_nyt") and any(h.f_alku is not None for h in list(an.heitot)):
            self._kello(ruutu, an.kello_nyt(f_out))
        if not voimassa:
            return ruutu
        # viivat: viimeisin heitto, jonka alku on ehtinyt (seuraava heitto korvaa edellisen); vanhenevat viivat_s:n kuluttua
        viimeisin = max(voimassa, key=lambda h: h.f_alku)
        for h in voimassa:
            if h.f_loppu is not None and f_out <= h.f_loppu + jalki:
                self._pisteet(ruutu, h, f_out)
        if f_out <= viimeisin.f_alku + viivat_f:
            self._viivat(ruutu, viimeisin)
            if self.paneeli:
                self._paneeli(ruutu, viimeisin, getattr(an, "laske_kierre", True))
        return ruutu
