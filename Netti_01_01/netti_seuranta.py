"""Reaaliaikainen kiven seuranta nettivideolta (alkuperaiset ruudut, ei top-down-muunnosta).

Ruutua EI muunneta kokonaan oikaistuksi kuvaksi: vain kiven ymparilla oleva leike (ROI) oikaistaan suoraan alkuperaisesta ruudusta yhdella
cv2.remap-kutsulla (kierto ja linssikorjaus valmiissa kartoissa, Kuvain).

Kaksi mittaustasoa:
  A) KAHVAN VARI (kevyt, joka ruudulle, paaprosessissa): kahvan vari (keltainen / punainen) -> sentroidi -> maa-asema (z = 15 cm). Siemen (kivi
     levossa hackissa), heiton alun ja lopun tunnistus, ennuste seuraavaan ruutuun, alkuarvo B:lle. Hajonta ~1-2 cm.
  B) TARKKA (kivi_tarkka.py, sama mittaus kuin offline-ajossa): graniitin peittavyys leikkeessa, kiven akseli (X, Y) sovitettuna alipikselina
     kiintean kokoiseen runkomalliin. ~50 ms / ruutu, ajetaan tyoprosesseissa (vain leike siirtyy prosessille). Hajonta ~0,1-0,3 cm.

Heiton tunnistus (Heittopaa): IDLE -> ARMED (kahva hack-alueella >= 0,5 s ja siina todettu kivi) -> LIIKKUU (kivi lahtee lepopaikastaan) -> VAHVISTETTU
(ylittanyt Y = -100 cm nopeudella >= 1,2 m/s mitatuista paikoista ja liikkeen aikana vahintaan 8 kelvollista kivisovitusta: kantaminen, kavelija ja ihmisten
varusteet eivat tayta) -> VALMIS (kivi poistuu kuva-alalta). Hylatty yritys (heittaja kasittelee kiveä) palaa ARMED-tilaan. Vahvistus tulee vasta noin 1-2 s
lahdon jalkeen, mutta lahtohetki (f_alku) ajoitetaan taaksepain: 5 s:n viiveella tulostettava kuva ei ole vielakaan ehtinyt lahtohetkeen.

Taustakuva: tulokset/moodikuva_<puoli>_puhdas.png jos olemassa (netti_tausta.py; hackin haamukivi pois), muuten moodikuva_<puoli>.png.
"""
import os
import types
import warnings
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

import aikaperusta as aikap
import kivi_netti as kv
import kivi_tarkka as kt
import kivimalli
import liuku_irroitus as li
import netti_tausta

KAHVA_Z = kv.KAHVA_Z_CM          # kahvan levyn korkeus (cm), jonka tasolle sentroidi heijastetaan

# ---------------------------------------------------------------------------------------------------------------------
# PARAMETRIT
# ---------------------------------------------------------------------------------------------------------------------
HACK_X = (-75.0, 75.0)           # siemenalue (cm): kivi levossa hackissa
HACK_Y = (-370.0, -225.0)
KAHVA_ALA_PX = (50, 700)         # hyvaksyttava kahvan pinta-ala hack-alueella (keltainen ~150-260, punainen ~110-160)
LEPO_S = 0.5                     # kivi hack-alueella vahintaan tama ennen kuin se "viritetaan" (heittaja kasittelee kiveä, joten ei tiukkaa lepoa)
LEPO_CM = 8.0                    # viritys: sentroidin vaihtelu alle tama
LIIKE_CM = 6.0                   # lahto: kolmen viimeisen havainnon mediaani yli tama lepopaikasta (lepopaikka = vanhempien havaintojen mediaani)
LEPO_N = 15                      # lepopaikan arvioon kaytetaan enintaan naita vanhempia havaintoja
HUKKA_LEPO_S = 1.0               # kahva poissa levossa >= tama -> takaisin IDLE:en
VAHVISTUS_Y_CM = -100.0          # heitto vahvistetaan kun Y yli taman ...
VAHVISTUS_V_CM_S = 120.0         # ... ja nopeus vahintaan tama (mitatuista paikoista; kavelija ei yllä)
HYLKAYS_AIKA_S = 10.0            # lahto, jonka jalkeen kivi ei vahvistu tassa ajassa -> hylataan (ylaraja)
EI_ETENE_S = 1.5                 # hylkays: tama aika liikkeessa ja liikealue alle EI_ETENE_CM -> kivea vain kasitellaan
EI_ETENE_CM = 12.0
LOPPU_Y_CM = 735.0               # kivi poistuu kuvasta (puolisko loppuu)
HUKKA_LIIKE_RUUTUA = 14          # liikkeessa: ei yhtaan hyvaksyttya mittausta tama monta yksilollista ruutua -> loppu
B_VAIN_KELV = 0.2                # B hyvaksytaan: kelvollisia pikseleita vahintaan
B_MAX_RMS = 0.30
B_MAX_POIKKEAMA_CM = 15.0        # B enintaan tama kaukana ennusteesta / A:sta
B_LEPO_VALI = 8                  # levossa B joka n:s yksilollinen ruutu
SOFT_KELV = 0.08                 # pehmea B (tiukka B ei kelpaa): kelvollisia vahintaan tama -> kelpaa seurantaan ja pisteisiin, ei estimaattiin
PEITTO_MIN = 0.35                # B kelpaa vain jos graniittia (alfa) on rungon siluetin sisalla vahintaan tama (oikea kivi ~0,4-0,95, ei-kivi ~0-0,2)
KIVI_B_LEPO = 2                  # ennen lahtoa hackissa on todettava kivi: vahintaan tama monta kelvollista B-sovitusta viimeisen 90 ruudun aikana
KIVI_B_VAHVISTUS = 8             # heiton vahvistus: kelvollisia B-sovituksia vahintaan tama monta liikkeen aikana
PAIKKA_PUUTE_RUUTUA = 24         # heitto paattyy kun yhtaan paikkaa (A / B) ei ole ollut taman monta ruutua
IDLE_VALI = 3                    # IDLE: kandidaattihaku joka n:s yksilollinen ruutu
ESTIMAATTI_VALI = 10             # viivaestimaatti paivitetaan kun vahintaan n uutta kelvollista B-naytetta
ESTIMAATTI_MIN_Y_CM = 330.0      # vasta kun rata yltaa tahan (muuten ei irroitusta)
ESTIMAATTI = dict(min_jalkeen=70.0, Y_r_min=200.0, kohinakatto=(0.8, 0.15))   # osittaisen aineiston asetukset (liuku_irroitus.analysoi)
KOPIO_KYNNYS = 15.0              # kopioruutu: harmaasavyero (suurin 0,1 % pikseleista) alle taman (aja_kivi.KOPIO_KYNNYS)


class Kuvain:
    """Yhden puoliskon leike alkuperaisesta ruudusta suoraan oikaistuun (kierretty + linssikorjattu) kuvaan: yhdistetyt kartat."""

    def __init__(self, cam):
        x0, y0, w, h = kv.kn.PUOLET[cam.puoli]
        self.mx = (x0 + (w - 1) - cam.map2).astype(np.float32)       # oikaistun pikselin (u, v) lahde alkuperaisessa ruudussa
        self.my = (y0 + cam.map1).astype(np.float32)
        self.w, self.h = cam.w, cam.h

    def leike(self, ruutu, x0, y0, x1, y1):
        """Oikaistun kuvan alue [x0, x1) x [y0, y1) (rajattu kuvaan); palauttaa (kuva | None, x0, y0)."""
        x0, y0 = max(int(x0), 0), max(int(y0), 0)
        x1, y1 = min(int(x1), self.w), min(int(y1), self.h)
        if x1 - x0 < 8 or y1 - y0 < 8:
            return None, x0, y0
        return cv2.remap(ruutu, self.mx[y0:y1, x0:x1], self.my[y0:y1, x0:x1], cv2.INTER_LINEAR), x0, y0


def _leike_kamera(cam, x0, y0, w, h):
    """cam-kopio, jonka taustataulukot on rajattu leikkeeseen (maskit / alfa_kartta / havainto kayttavat vain taulukoita)."""
    jaa, vari = kt.tausta_maskit(cam)
    sl = (slice(y0, y0 + h), slice(x0, x0 + w))
    c = types.SimpleNamespace(mode=cam.mode[sl], mode_hsv=cam.mode_hsv[sl], lane=cam.lane[sl])
    c._tausta = (jaa[sl], vari[sl])
    c.maskit = types.MethodType(kv.Kamera.maskit, c)
    return c


def _lataa_runko(kansio, puoli):
    """Kiven runkomalli (kivi_tarkka.Runko): puolen oma tiedosto, muuten vasemman (sama kivi, sama optiikka) tai oletusmalli."""
    for p in (puoli, "vasen"):
        if os.path.exists(os.path.join(kansio, f"kivimalli_{p}.json")):
            return kt.Runko.lataa(kansio, p)
    return kt.Runko()


class Mittari:
    """Yhden puoliskon mittaukset: kuva-alue, kahvan sentroidi (A) ja tarkka sovitus (B)."""

    def __init__(self, kansio, puoli, runko=True):
        self.puoli = puoli
        self.cam = kv.Kamera(puoli, os.path.join(kansio, f"kalibrointi_{puoli}.json"), netti_tausta.kaytettava(kansio, puoli))
        self.kuvain = Kuvain(self.cam)
        self.runko = _lataa_runko(kansio, puoli) if runko else None
        if runko:
            kt.tausta_maskit(self.cam)                 # esilaskenta

    # ---- koordinaatit (oikaistu kuva) ----------------------------------------------------------------------------
    def maa_oikaistuun(self, X, Y, z=KAHVA_Z):
        X = np.atleast_1d(np.asarray(X, float))
        u, v = kivimalli.project_3d(self.cam.K, self.cam.R, self.cam.t, np.column_stack([X, np.atleast_1d(np.asarray(Y, float)), np.full(len(X), float(z))]))
        return u, v

    def oikaistu_maahan(self, u, v, z=KAHVA_Z):
        return kv.maan_paikka(self.cam, u, v, z=z)

    def skaala(self, X, Y, z=KAHVA_Z):
        """Kuvamittakaava (px / cm) pisteessa (X, Y, z): sivusuunnassa ja pituussuunnassa (geometrinen keskiarvo)."""
        u, v = self.maa_oikaistuun([X, X + 5.0, X], [Y, Y, Y + 5.0], z)
        return float(np.sqrt(np.hypot(u[1] - u[0], v[1] - v[0]) * np.hypot(u[2] - u[0], v[2] - v[0])) / 5.0)

    def alue_maasta(self, X0, Y0, marg_px, r_cm=15.0, korkeus=KAHVA_Z):
        """Leikkeen rajat (x0, y0, x1, y1) kiven ymparille: kiven ala- ja ylaosan ympyrat + marginaali."""
        th = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        P = []
        for z in (0.0, korkeus + 4.0):
            P.append(np.column_stack([X0 + r_cm * np.cos(th), Y0 + r_cm * np.sin(th), np.full(12, z)]))
        u, v = kivimalli.project_3d(self.cam.K, self.cam.R, self.cam.t, np.vstack(P))
        return (int(np.floor(u.min() - marg_px)), int(np.floor(v.min() - marg_px)), int(np.ceil(u.max() + marg_px)), int(np.ceil(v.max() + marg_px)))

    def alue_kuvasta(self, u, v, r_px):
        return (int(u - r_px), int(v - r_px), int(u + r_px) + 1, int(v + r_px) + 1)

    # ---- A: kahvan sentroidi ------------------------------------------------------------------------------------
    def kahva(self, ruutu, alue, vari, ennuste_uv=None, max_etaisyys_px=45.0, min_ala=12, taustaero=False):
        """Kahvan vari leikkeessa -> kandidaatit [(u, v, ala, bbox)] oikaistussa kuvassa (kokonaiskoordinaatit). Jos taustaero: vain taustasta poikkeavat
        pikselit (rengas ei yhdisty punaiseen kahvaan). Jos ennuste_uv: lahin kandidaatti ennusteeseen (<= max_etaisyys_px)."""
        roi, x0, y0 = self.kuvain.leike(ruutu, *alue)
        if roi is None:
            return []
        m = kv.kahva_maski(roi, vari)
        if taustaero:
            sl = (slice(y0, y0 + roi.shape[0]), slice(x0, x0 + roi.shape[1]))
            erot = np.max(np.abs(roi.astype(np.int16) - self.cam.mode[sl].astype(np.int16)), axis=2)
            m &= (erot > kv.FG_KYNNYS).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        n, lab, st, cen = cv2.connectedComponentsWithStats(m)
        out = []
        for k in range(1, n):
            if st[k, cv2.CC_STAT_AREA] < min_ala:
                continue
            u, v = cen[k][0] + x0, cen[k][1] + y0
            if ennuste_uv is not None and np.hypot(u - ennuste_uv[0], v - ennuste_uv[1]) > max_etaisyys_px:
                continue
            out.append((float(u), float(v), int(st[k, cv2.CC_STAT_AREA]), tuple(int(a) for a in st[k, :4])))
        if ennuste_uv is not None and out:
            out.sort(key=lambda c: np.hypot(c[0] - ennuste_uv[0], c[1] - ennuste_uv[1]))
            return out[:1]
        return out

    # ---- B: tarkka sovitus ----------------------------------------------------------------------------------------
    def valmistele(self, ruutu, X0, Y0, marg_px=36):
        """B:n leike alkuarvon ymparilta: (roi, x0, y0) tai None. Vain tama siirtyy tyoprosessille (ei koko ruutua)."""
        alue = self.alue_maasta(X0, Y0, marg_px, r_cm=self.runko.R_max + 12.0)
        roi, x0, y0 = self.kuvain.leike(ruutu, *alue)
        return None if roi is None else (roi, x0, y0)

    def sovita_leike(self, roi, x0, y0, X0, Y0):
        """Kiven akseli (X, Y) leikkeesta (alkuarvo X0, Y0): dict(X, Y, rms, kelvolliset). Kaksi kierrosta (kaista 8 -> 5 px)."""
        h, w = roi.shape[:2]
        c = _leike_kamera(self.cam, x0, y0, w, h)
        fg, gran, handle = c.maskit(roi)
        alfa, ok = kt.havainto(c, roi, kv.alfa_kartta(c, roi, gran))
        s = kt.sovita(self.cam, self.runko, alfa, ok, X0, Y0, kaarros=(8, 5), ox=x0, oy=y0)
        # peitto: graniitin (alfa) keskiarvo sovitetun rungon siluetin sisalla (ei-kivi: ~0; kivi ~0,6-0,95, kasi voi peittaa osan)
        mc, _ = kt.malli_kuva(self.cam, self.runko, (x0, y0, w, h), s["X"], s["Y"])
        sis = mc > 0.5
        s["peitto"] = float(alfa[sis].mean()) if sis.any() else 0.0
        return s

    def tarkka(self, ruutu, X0, Y0):
        v = self.valmistele(ruutu, X0, Y0)
        return None if v is None else self.sovita_leike(*v, X0, Y0)


# ---------------------------------------------------------------------------------------------------------------------
# TYOPROSESSIT: B-taso ja viivaestimaatti
# ---------------------------------------------------------------------------------------------------------------------
_M = {}


def alusta_tyo(kansio, puolet):
    for p in puolet:
        _M[p] = Mittari(kansio, p)


def tyo_b(args):
    """B-tehtava: (avain, puoli, roi, x0, y0, X0, Y0) -> (avain, tulos)."""
    avain, puoli, roi, x0, y0, X0, Y0 = args
    try:
        return avain, _M[puoli].sovita_leike(roi, x0, y0, X0, Y0)
    except Exception as e:                              # yksittainen ruutu ei saa kaataa seurantaa
        return avain, dict(virhe=repr(e))


def laske_viivat(f, X, Y, fps_lahde=25.0, fps_ulos=30.0):
    """Viivaestimaatti kelvollisista (ruutu, X, Y) -naytteista: aikaperusta (aikaperusta.Aikajana) + liu'un ja irroituksen tunnistus (liuku_irroitus). Palauttaa
    dict tai None. Kaytetaan sellaisenaan tyoprosessissa. fps_lahde = erillisten ruutujen nimellistaajuus (25 tassa videossa: 25 r/s -> 30 r/s kopioruuduilla),
    fps_ulos = videon ruutunopeus."""
    aikap.FPS_LAHDE, aikap.FPS_ULOS = float(fps_lahde), float(fps_ulos)
    f = np.asarray(f, float)
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    o = np.argsort(f)
    f, X, Y = f[o], X[o], Y[o]
    if len(f) < 30 or Y.max() < ESTIMAATTI_MIN_Y_CM:
        return None
    try:
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            aj = aikap.Aikajana(f, Y)
            mk = np.array([int(fi) not in aj.kopio for fi in f])
            f, X, Y = f[mk], X[mk], Y[mk]
            t = aj.aika(f)
            r = li.analysoi(dict(t=t, X=X, Y=Y), **ESTIMAATTI)
    except Exception as e:                              # liian vahva / outo aineisto: ei estimaattia
        return dict(virhe=repr(e))
    if r.get("irroitus") is None or r.get("tila") is None:
        return None
    ti, ys, la, ir = r["tila"], r["yhteenveto"], r["liuku"], r["irroitus"]
    vapaa_Y = float(Y.max() - ti["Y_cm"])
    return dict(
        liuku_a=ys["liun_suora_X_Y0_cm"], liuku_b=ys["liun_suora_kulmakerroin"], liuku_suunta_deg=ys["liun_suunta_deg"],
        liuku_Y0=ys["liun_suunta_alku_cm"], liuku_Y1=ys["liun_suunta_loppu_cm"],
        irr_X=ti["X_cm"], irr_Y=ti["Y_cm"], irr_suunta_deg=ti["suunta_deg"], irr_virhe_deg=ti["suunta_virhe_deg"],
        irr_Y_vali=(ir["Y_min"], ir["Y_max"]), merkitseva=bool(ir["merkitseva"]), dchi2=float(ir["dchi2_med"]),
        vapaa_matka_cm=vapaa_Y, vapaa_ruutuja=int(ti["ruutuja"]), Y_max=float(Y.max()), t_lahto=float(la["t_lahto"]),
        nopeus_ennen=ys["nopeus_ennen_m_s"], nopeus_jalkeen=ys["nopeus_jalkeen_m_s"],
        liuku_hidastuvuus=ys["liun_hidastuvuus_m_s2"], vapaa_hidastuvuus=ys["vapaan_lennon_hidastuvuus_m_s2"],
        kaarto=ti["kaarto"], n=int(len(f)))


# ---------------------------------------------------------------------------------------------------------------------
# HEITTO JA HEITTOPAA
# ---------------------------------------------------------------------------------------------------------------------
class Naytteet:
    """Yksi yksilollinen ruutu: A (kahvan sentroidi) ja B (tarkka sovitus), maakoordinaatit cm."""
    __slots__ = ("f", "A", "ala", "B", "Bs", "rms", "kelv", "peitto", "odottaa")

    def __init__(self, f, A=None, ala=0):
        self.f, self.A, self.ala = f, A, ala
        self.B, self.Bs, self.rms, self.kelv, self.peitto, self.odottaa = None, None, None, None, None, False

    def paikka(self):
        """Paras tunnettu paikka: B, muuten pehmea B (kelvollisia >= SOFT_KELV), muuten A."""
        if self.B is not None:
            return self.B
        if self.Bs is not None:
            return self.Bs
        return self.A

    def kivi(self):
        """Todettu kivi: B tai pehmea B kelvollinen (graniittia siluetin sisalla)."""
        return self.B is not None or self.Bs is not None


class Heitto:
    def __init__(self, tunnus, puoli, vari, f_lepo, lepo_XY):
        self.tunnus, self.puoli, self.vari = tunnus, puoli, vari
        self.f_lepo, self.lepo = f_lepo, np.array(lepo_XY, float)
        self.tila = "liikkuu"                  # liikkuu | vahvistettu | valmis | hylatty
        self.naytteet = []
        self.f_alku = None                     # lahtoruutu (taaksepain ajoitettu)
        self.f_loppu = None                    # viimeinen seurattu ruutu
        self.viivat = None                     # viimeisin viivaestimaatti (laske_viivat) tai None
        self.viivat_f = None                   # ruutu, jonka aineistolla estimaatti on laskettu
        self.viivat_n = 0                      # estimaatin kelvollisten B-naytteiden maara
        self.viivat_hist = []                  # viimeisimpien estimaattien irroituspaikat (vakauden arvio)
        self.viivat_loki = []                  # kaikki estimaatit: (analyysin ruutu, irroitus Y, suunta, T-viiva irroitus, T-viiva liuku, vakaa)
        self.f_vahvistus = None
        self.lopullinen = False                # lopullinen estimaatti (heitto paattynyt, kaikki aineisto)

    def kelvolliset_b(self):
        return [n for n in self.naytteet if n.B is not None]

    def kivi_n(self, viim_ruutua=90, f=None):
        """Kelvollisten (kivi todettu) B-sovitusten maara viimeisten viim_ruutua ruudun aikana (f: nykyinen ruutu; oletus viimeisin nayte)."""
        if not self.naytteet:
            return 0
        f = self.naytteet[-1].f if f is None else f
        return sum(1 for n in self.naytteet[-200:] if f - n.f <= viim_ruutua and n.kivi())

    def pisteet(self):
        """[(f, X, Y)] piirrettavat pisteet lahtoruudusta alkaen (B, muuten A)."""
        out = []
        for n in list(self.naytteet):
            if self.f_alku is not None and n.f >= self.f_alku and n.paikka() is not None:
                p = n.paikka()
                out.append((n.f, p[0], p[1]))
        return out


class Heittopaa:
    """Yhden pelipaan (vasen / oikea) heittojen tunnistus ja seuranta."""

    def __init__(self, analyysi, puoli):
        self.an = analyysi
        self.puoli = puoli
        self.m = analyysi.M[puoli]
        self.tila = "IDLE"
        self.heitto = None
        self.kandidaatit = {}                   # IDLE: vari -> lista (f, X, Y, u, v, ala)
        self.lepo = []                          # ARMED: viimeisimmat A-havainnot (f, X, Y, u, v, ala)
        self.f_nahty = None                     # viimeisin ruutu, jossa kahva nakyi (ARMED)
        self.vari = None
        self.n_ruutu = 0
        self.hukka = 0
        self.ala_norm = []                      # hyvaksyttyjen kahvojen ala / skaala^2 (viimeiset 8): odotetun alan mediaani
        self.f_liike = None
        self.edellinen_loppu_f = -10 ** 9
        self.ei_ennen_f = -10 ** 9               # IDLE-haku tauolla tahan ruutuun asti (kivettomat kandidaatit)
        P = np.array([(x, y, z) for x in HACK_X for y in HACK_Y for z in (0.0, 20.0)])
        u, v = kivimalli.project_3d(self.m.cam.K, self.m.cam.R, self.m.cam.t, P)
        self.hack_alue = (int(u.min()) - 6, int(v.min()) - 6, int(u.max()) + 6, int(v.max()) + 6)

    # ---- apurit --------------------------------------------------------------------------------------------------
    def _a_maa(self, c):
        return self.m.oikaistu_maahan(c[0], c[1])

    @staticmethod
    def _hack_sisalla(X, Y):
        return HACK_X[0] <= X <= HACK_X[1] and HACK_Y[0] <= Y <= HACK_Y[1]

    def _konsol(self, n, o):
        """Konsolidoitu paikka seurantaan: B / pehmea B sellaisenaan, A korjattuna B-kehykseen (A - o)."""
        if n.kivi():
            return np.array(n.paikka(), float)
        if n.A is not None:
            return np.array(n.A, float) + o
        return None

    def _ennuste(self, f):
        """Ennuste (X, Y) ruudulle f: viimeisten (<= 7) konsolidoidun paikan suora ruutuindeksin funktiona (vakionopeus)."""
        o = self._offset_b()
        pts = []
        for n in reversed(self.heitto.naytteet):
            if f - n.f > 60:
                break
            q = self._konsol(n, o)
            if q is not None:
                pts.append((n.f, q))
            if len(pts) >= 7:
                break
        if not pts:
            return tuple(self.heitto.lepo)
        pts.reverse()
        if len(pts) < 3:
            fb, qb = pts[-1]
            if len(pts) == 2 and pts[-1][0] > pts[0][0]:
                v = (pts[-1][1] - pts[0][1]) / (pts[-1][0] - pts[0][0])
                return tuple(qb + v * (f - fb))
            return tuple(qb)
        ff = np.array([p[0] for p in pts], float)
        Q = np.array([p[1] for p in pts], float)
        fm = ff.mean()
        A = np.column_stack([np.ones(len(ff)), ff - fm])
        c = np.linalg.lstsq(A, Q, rcond=None)[0]
        p = c[0] + c[1] * (f - fm)
        return float(p[0]), float(p[1])

    def _offset_b(self):
        """B - A -erotus viimeisista naytteista, joissa molemmat tunnetaan (A:n systemaattinen poikkeama): mediaani."""
        d = [np.array(n.B) - np.array(n.A) for n in self.heitto.naytteet[-12:] if n.B is not None and n.A is not None]
        return np.median(np.array(d), axis=0) if len(d) >= 3 else np.zeros(2)

    def _nopeus_ruutu(self):
        """Nopeus [cm / ruutu] viimeisten konsolidoitujen paikkojen ensimmaisen ja viimeisen valilla."""
        o = self._offset_b()
        pts = []
        for n in reversed(self.heitto.naytteet):
            q = self._konsol(n, o)
            if q is not None:
                pts.append((n.f, q))
            if len(pts) >= 6:
                break
        if len(pts) < 3:
            return 0.0
        (fb, qb), (fa, qa) = pts[0], pts[-1]
        return float(np.hypot(qb[0] - qa[0], qb[1] - qa[1])) / max(fb - fa, 1)

    def _vari_a(self, ruutu, ennuste_XY, gate_cm, tarkka_ala=True):
        """A-mittaus: kahva ennusteen ymparilta; hyvaksytaan alaltaan ja sijainniltaan sopiva blobi. Palauttaa (X, Y, ala, skaala) tai None."""
        Xp, Yp = ennuste_XY
        up, vp = self.m.maa_oikaistuun(Xp, Yp)
        s = self.m.skaala(Xp, Yp)
        r_px = float(np.clip(45.0 + 0.25 * gate_cm * s, 45.0, 140.0))
        blobit = self.m.kahva(ruutu, self.m.alue_kuvasta(up[0], vp[0], r_px), self.vari, (up[0], vp[0]), max_etaisyys_px=gate_cm * s,
                              taustaero=(self.vari == "punainen" and abs(Yp) < 110.0))
        if not blobit:
            return None
        u, v, ala, bb = blobit[0]
        if tarkka_ala and self.ala_norm:
            odotus = float(np.median(self.ala_norm)) * s * s
            if not (0.4 * odotus <= ala <= 2.2 * odotus):
                return None
        X, Y = self._a_maa((u, v))
        return X, Y, ala, s

    def _idle_alkuun(self):
        self.tila = "IDLE"
        self.heitto = None
        self.kandidaatit = {}
        self.lepo = []

    # ---- IDLE: kandidaattihaku ----------------------------------------------------------------------------------
    def _idle(self, f, ruutu):
        if self.n_ruutu % IDLE_VALI:
            return
        loytyi = {}
        for vari in ("keltainen", "punainen"):
            for c in self.m.kahva(ruutu, self.hack_alue, vari, None, min_ala=KAHVA_ALA_PX[0] // 2, taustaero=False):
                if not (KAHVA_ALA_PX[0] <= c[2] <= KAHVA_ALA_PX[1]):
                    continue
                bb = c[3]
                if not (0.35 <= bb[2] / max(bb[3], 1) <= 2.8):
                    continue
                X, Y = self._a_maa(c)
                if self._hack_sisalla(X, Y):
                    loytyi.setdefault(vari, []).append((f, X, Y, c[0], c[1], c[2]))
        for vari in ("keltainen", "punainen"):
            lista = self.kandidaatit.setdefault(vari, [])
            uusi = loytyi.get(vari, [])
            # jatka kandidaattia jos sama paikka (<= LEPO_CM) kuin viimeisin, muuten aloita alusta
            if uusi and lista and min(np.hypot(c[1] - lista[-1][1], c[2] - lista[-1][2]) for c in uusi) <= LEPO_CM:
                lista.append(min(uusi, key=lambda c: np.hypot(c[1] - lista[-1][1], c[2] - lista[-1][2])))
            elif uusi:
                self.kandidaatit[vari] = [max(uusi, key=lambda c: c[5])]
            else:
                self.kandidaatit[vari] = []
                continue
            lista = self.kandidaatit[vari]
            if (lista[-1][0] - lista[0][0]) / self.an.fps >= LEPO_S and len(lista) >= 4:
                X = float(np.median([c[1] for c in lista]))
                Y = float(np.median([c[2] for c in lista]))
                if max(np.hypot(c[1] - X, c[2] - Y) for c in lista) <= LEPO_CM:
                    self._virita(f, vari, lista, (X, Y))
                    n = Naytteet(f, (lista[-1][1], lista[-1][2]), lista[-1][5])
                    self.heitto.naytteet.append(n)
                    self.an.pyyda_b(self, self.heitto, n, ruutu, X, Y)
                    return

    def _virita(self, f, vari, lista, XY):
        self.tila = "ARMED"
        self.vari = vari
        self.lepo = list(lista)[-40:]
        self.f_nahty = f
        sk = self.m.skaala(XY[0], XY[1])
        self.ala_norm = [c[5] / (sk * sk) for c in lista][-8:]
        self.kandidaatit = {}
        self.hukka = 0
        self.heitto = Heitto(self.an.uusi_tunnus(), self.puoli, vari, lista[0][0], XY)
        self.an.tapahtumat.append(("viritetty", self.heitto.tunnus, f, f"{self.puoli} {vari} ({XY[0]:.0f},{XY[1]:.0f})"))

    # ---- ARMED: kivi hack-alueella, odotetaan lahtoa --------------------------------------------------------------
    def _lepopaikka(self):
        """Lepopaikka = vanhempien havaintojen mediaani (kolme viimeisinta pois, jotta lahto ei vedä sitä mukanaan)."""
        vanh = self.lepo[:-3][-LEPO_N:] if len(self.lepo) > 8 else self.lepo[:max(len(self.lepo) - 3, 1)]
        return float(np.median([c[1] for c in vanh])), float(np.median([c[2] for c in vanh]))

    def _armed(self, f, ruutu):
        h = self.heitto
        X0, Y0 = h.lepo
        r = self._vari_a(ruutu, (X0, Y0), 35.0, tarkka_ala=False)
        if r is None:
            if (f - self.f_nahty) / self.an.fps >= HUKKA_LEPO_S:
                self._idle_alkuun()
            return
        X, Y, ala, s = r
        self.f_nahty = f
        u, v = self.m.maa_oikaistuun(X, Y)
        self.lepo = (self.lepo + [(f, X, Y, float(u[0]), float(v[0]), ala)])[-40:]
        n = Naytteet(f, (X, Y), ala)
        h.naytteet.append(n)
        if len(h.naytteet) > 400:
            del h.naytteet[:100]
        if self.n_ruutu % B_LEPO_VALI == 0:                     # levossa B harvaan (tarkka lepopaikka estimaatin alkuun)
            self.an.pyyda_b(self, h, n, ruutu, X, Y)
        if len(self.lepo) < 6:
            return
        h.lepo = self._lepopaikka()
        X0, Y0 = h.lepo
        self.ala_norm = [q[5] / (s * s) for q in self.lepo[-8:]]
        viim = self.lepo[-3:]
        if np.hypot(np.median([q[1] for q in viim]) - X0, np.median([q[2] for q in viim]) - Y0) >= LIIKE_CM:
            if h.kivi_n(90, f) >= KIVI_B_LEPO:
                self._lahto(f)
        elif h.kivi_n(90, f) < KIVI_B_LEPO and (f - h.naytteet[0].f) / self.an.fps > 3.0 and self.n_ruutu % B_LEPO_VALI == 0:
            # kahva ilman kivea (esim. harjan paa, vaatetus): ei viritetty; haku jatkuu vasta hetken kuluttua
            self.an.tapahtumat.append(("ei_kivea", h.tunnus, f, f"{self.puoli} {self.vari}"))
            self.ei_ennen_f = f + int(4 * self.an.fps)
            self._idle_alkuun()

    def _lahto(self, f):
        """Kivi lahti: LIIKKUU. Lahtoruutu ajoitetaan taaksepain: lepopaikasta poikenneiden havaintojen jakson ensimmainen ruutu."""
        h = self.heitto
        X0, Y0 = h.lepo
        h.tila = "liikkuu"
        self.tila = "LIIKKUU"
        self.f_liike = f
        self.hukka = 0
        i = len(self.lepo)
        while i > 0 and np.hypot(self.lepo[i - 1][1] - X0, self.lepo[i - 1][2] - Y0) > 2.0:
            i -= 1
        h.f_alku = self.lepo[i][0] if i < len(self.lepo) else f
        self.an.tapahtumat.append(("lahto", h.tunnus, f, f"f_alku {h.f_alku}"))
        # lahtoa edeltavat A-havainnot saavat B:n myohemmin (uusi pyynto ei onnistu ilman ruutuja): ne jaavat A-paikoiksi

    def _takaisin_viritykseen(self, f):
        """Hylatty yritys: jos kahva on yha hack-alueella, jatketaan viritettyna (heittaja kasittelee kiveä), muuten IDLE."""
        h = self.heitto
        ns = [n for n in h.naytteet if n.A is not None][-8:]
        if len(ns) >= 4 and all(self._hack_sisalla(*n.A) for n in ns[-3:]):
            lista = []
            for n in ns:
                u, v = self.m.maa_oikaistuun(*n.A)
                lista.append((n.f, n.A[0], n.A[1], float(u[0]), float(v[0]), n.ala))
            XY = (float(np.median([c[1] for c in lista])), float(np.median([c[2] for c in lista])))
            edelliset = [n for n in h.naytteet if f - n.f <= 90]        # sama kivi: todisteet kivesta (B-sovitukset) siirtyvat mukana
            self._virita(f, self.vari, lista, XY)
            self.heitto.naytteet = edelliset
        else:
            self._idle_alkuun()

    # ---- LIIKKUU / VAHVISTETTU ---------------------------------------------------------------------------------
    def _liikkuu(self, f, ruutu):
        h = self.heitto
        Xp, Yp = self._ennuste(f)
        a = self._vari_a(ruutu, (Xp, Yp), 14.0 + 0.5 * self._nopeus_ruutu())
        n = Naytteet(f)
        if a is not None:
            n.A, n.ala = (a[0], a[1]), a[2]
            self.ala_norm = (self.ala_norm + [a[2] / (a[3] * a[3])])[-8:]
            o = self._offset_b()
            X0, Y0 = a[0] + o[0], a[1] + o[1]
        else:
            X0, Y0 = Xp, Yp
        h.naytteet.append(n)
        self.an.pyyda_b(self, h, n, ruutu, X0, Y0)
        # paikan puute: viimeisin ruutu, jolla on jokin paikka (A tai B; B voi saapua myohemmin)
        f_paikka = max([m.f for m in h.naytteet[-60:] if m.paikka() is not None] or [self.f_liike])
        self.hukka = f - f_paikka
        viim = [m for m in h.naytteet[-12:] if m.paikka() is not None]
        p = viim[-1].paikka() if viim else (Xp, Yp)               # vahvistus vain MITATUSTA paikasta (ei ennusteesta)
        if h.tila == "liikkuu":
            v = self._nopeus_ruutu() * self.an.fps
            if viim and p[1] >= VAHVISTUS_Y_CM and v >= VAHVISTUS_V_CM_S and h.kivi_n(10 ** 6, f) >= KIVI_B_VAHVISTUS:
                h.tila = "vahvistettu"
                h.f_vahvistus = f
                self.an.tapahtumat.append(("vahvistettu", h.tunnus, f, f"v {v:.0f} cm/s Y {p[1]:.0f} kivi_B {h.kivi_n(10 ** 6, f)}"))
            elif (f - self.f_liike) / self.an.fps > HYLKAYS_AIKA_S or self.hukka > HUKKA_LIIKE_RUUTUA or v > 600.0 or self._ei_etene(f):
                h.tila = "hylatty"
                self.an.hylatyt.append(h)
                self.an.tapahtumat.append(("hylatty", h.tunnus, f, f"v {v:.0f} cm/s, hukka {self.hukka}"))
                self._takaisin_viritykseen(f)
        elif h.tila == "vahvistettu":
            if Yp >= LOPPU_Y_CM or self.hukka > PAIKKA_PUUTE_RUUTUA:
                self._valmis(f)

    def _ei_etene(self, f):
        """Liikkeessa vahintaan EI_ETENE_S, mutta kivi on viimeisen EI_ETENE_S aikana liikkunut alle EI_ETENE_CM (X- tai Y-vaihteluvali): kivea kasitellaan, ei
        heiteta. (Vaihteluvali eika nettoetenema: heiton alkua edeltava vastaliike hackissa kumoaa nettoetenemaan.)"""
        h = self.heitto
        if (f - self.f_liike) / self.an.fps < EI_ETENE_S:
            return False
        o = self._offset_b()
        pts = []
        for n in reversed(h.naytteet):
            if f - n.f > EI_ETENE_S * self.an.fps:
                break
            q = self._konsol(n, o)
            if q is not None:
                pts.append(q)
        if len(pts) < 6:
            return False
        P = np.array(pts)
        return float(max(np.ptp(P[:, 0]), np.ptp(P[:, 1]))) < EI_ETENE_CM

    def _valmis(self, f):
        h = self.heitto
        h.tila = "valmis"
        ns = [n for n in h.naytteet if n.paikka() is not None]
        h.f_loppu = ns[-1].f if ns else f
        self.an.valmiit.append(h)
        self.edellinen_loppu_f = f
        self._idle_alkuun()
        self.an.loppu(h)

    # ---- paasilmukka -----------------------------------------------------------------------------------------------
    def kasittele(self, f, ruutu):
        self.n_ruutu += 1
        if self.tila == "IDLE":
            if f - self.edellinen_loppu_f > 2 * self.an.fps and f >= self.ei_ennen_f:
                self._idle(f, ruutu)
        elif self.tila == "ARMED":
            self._armed(f, ruutu)
        else:
            self._liikkuu(f, ruutu)
        h = self.heitto
        if h is not None and h.tila in ("liikkuu", "vahvistettu"):
            ns = [n for n in h.naytteet if n.paikka() is not None]
            h.f_loppu = ns[-1].f if ns else f


# ---------------------------------------------------------------------------------------------------------------------
# ANALYYSI: kopioruudut, molemmat paat, B-tyoprosessit, viivaestimaatti
# ---------------------------------------------------------------------------------------------------------------------
class Analyysi:
    """Syota ruudut yksi kerrallaan (syota); vahvistetut heitot (pisteet, viivat) luetaan heitot-listasta (piirtaja). synkroninen = True: B-tyot ja estimaatit
    odotetaan (sim-ajo: tulos ei riipu ajoituksesta); False: ne kulkevat taustalla (oikea reaaliaika)."""

    def __init__(self, kansio="tulokset", fps=30.0, saikeet=3, tarkka=True, synkroninen=True, paat=("vasen", "oikea"), viive_b_ruutuja=6, fps_lahde=None):
        self.fps = float(fps)
        self.fps_lahde = 25.0 if fps_lahde is None else float(fps_lahde)      # None: paatellaan kopioruutujen osuudesta (_paata_fps_lahde)
        self._fps_auto = fps_lahde is None
        self.synkroninen = synkroninen
        self.viive_b = viive_b_ruutuja
        self.kansio = kansio
        self.tarkka = bool(tarkka)
        self.M = {p: Mittari(kansio, p) for p in paat}
        self.pool = self.est_pool = None
        if self.tarkka and saikeet > 0:
            self.pool = ProcessPoolExecutor(saikeet, initializer=alusta_tyo, initargs=(kansio, tuple(paat)))
        if not synkroninen or saikeet > 0:
            self.est_pool = ProcessPoolExecutor(1) if not synkroninen else None
        self.tyot = []                               # [(future, f, heitto, naytteet, alkuarvo)]
        self.est_tyot = []                           # [(future, heitto, f, n)]
        self.paat = {p: Heittopaa(self, p) for p in paat}
        self.heitot = []                             # vahvistetut heitot (piirtajalle)
        self.valmiit, self.hylatyt = [], []
        self._tunnus = 0
        self.edellinen_harmaa = None
        self.kopioita = 0
        self.yksilollisia = 0
        self.tapahtumat = []

    def _paata_fps_lahde(self):
        """Erillisten ruutujen nimellistaajuus: jos kopioruutuja on alle 3 %, virta on kopioton ja erilliset ruudut tulevat ruutunopeudella; muuten (kuten tassa
        videossa, 26 % kopioita) lahdevirta on 25 r/s."""
        self._fps_auto = False
        osuus = self.kopioita / max(self.yksilollisia + self.kopioita, 1)
        self.fps_lahde = self.fps if osuus < 0.03 else 25.0

    def uusi_tunnus(self):
        self._tunnus += 1
        return self._tunnus

    # ---- kopioruutu ------------------------------------------------------------------------------------------------
    def _kopio(self, ruutu):
        """Kopioruutu (edellisen tarkka kopio uudelleenkoodattuna): harmaasavyero alle KOPIO_KYNNYS lahes kaikissa pikseleissa (alle 0,1 %) MOLEMMISSA puoliskoissa.
        (Vain toinen puolisko ei kelpaa: jos kivi liikkuu toisessa paassa, toinen puolisko voi olla lahes staattinen.)"""
        y0, h = kv.kn.PUOLET["vasen"][1], kv.kn.PUOLET["vasen"][3]
        g = cv2.cvtColor(ruutu[y0:y0 + h], cv2.COLOR_BGR2GRAY)
        kopio = False
        if self.edellinen_harmaa is not None:
            d = cv2.absdiff(g, self.edellinen_harmaa)
            puoli = d.shape[1] // 2
            raja = 0.001 * d.shape[0] * puoli
            kopio = np.count_nonzero(d[:, :puoli] > KOPIO_KYNNYS) < raja and np.count_nonzero(d[:, puoli:] > KOPIO_KYNNYS) < raja
        self.edellinen_harmaa = g
        return kopio

    # ---- B-tyot ---------------------------------------------------------------------------------------------------
    def pyyda_b(self, hp, heitto, naytteet, ruutu, X0, Y0):
        """B-mittaus naytteelle: tyoprosessille (leike) tai suoraan (ei poolia / sim ilman prosesseja). Reaaliajassa ruuhkautuessa B jatetaan pois."""
        if not self.tarkka:
            return
        if self.pool is not None and not self.synkroninen and len(self.tyot) > 24:
            return
        v = hp.m.valmistele(ruutu, X0, Y0)
        if v is None:
            return
        naytteet.odottaa = True
        if self.pool is not None:
            fut = self.pool.submit(tyo_b, (id(naytteet), hp.puoli, v[0], v[1], v[2], X0, Y0))
            self.tyot.append((fut, naytteet.f, heitto, naytteet, (X0, Y0)))
        else:
            self._liita_b(heitto, naytteet, hp.m.sovita_leike(*v, X0, Y0), (X0, Y0))

    def _liita_b(self, heitto, naytteet, s, alku):
        naytteet.odottaa = False
        if s is None or "virhe" in s:
            return
        poik = float(np.hypot(s["X"] - alku[0], s["Y"] - alku[1]))
        peitto = s.get("peitto", 1.0)
        naytteet.rms, naytteet.kelv, naytteet.peitto = s["rms"], s["kelvolliset"], peitto
        kivi = peitto >= PEITTO_MIN and s["rms"] <= B_MAX_RMS and poik <= B_MAX_POIKKEAMA_CM
        if kivi and s["kelvolliset"] >= B_VAIN_KELV:
            naytteet.B = (s["X"], s["Y"])
        elif kivi and s["kelvolliset"] >= SOFT_KELV:
            naytteet.Bs = (s["X"], s["Y"])

    def odota(self, f_asti=None):
        """Kerää valmiit B-tyot; f_asti: odota (estaen) kaikki tyot, joiden ruutu <= f_asti (sim-ajo)."""
        jaa = []
        for t in self.tyot:
            fut, f, heitto, n, alku = t
            if (f_asti is None and fut.done()) or (f_asti is not None and f <= f_asti):
                try:
                    _, s = fut.result()
                except Exception as e:
                    s = dict(virhe=repr(e))
                self._liita_b(heitto, n, s, alku)
            else:
                jaa.append(t)
        self.tyot = jaa
        self._keraa_estimaatit()
        self._loppujen_tarkistus()

    # ---- viivaestimaatti ------------------------------------------------------------------------------------------
    def paivita_viivat(self, heitto, f, pakota=False):
        """Laske viivaestimaatti, jos B-aineisto on kasvanut tarpeeksi (tai pakota)."""
        ns = heitto.kelvolliset_b()
        if len(ns) < 30 or (not pakota and len(ns) - heitto.viivat_n < ESTIMAATTI_VALI):
            return
        ff, XX, YY = [n.f for n in ns], [n.B[0] for n in ns], [n.B[1] for n in ns]
        if max(YY) < ESTIMAATTI_MIN_Y_CM:
            return
        heitto.viivat_n = len(ns)
        if self.est_pool is not None:
            if any(e[1] is heitto for e in self.est_tyot):
                if not pakota:
                    return                           # edellinen laskenta kesken
                for e in [e for e in self.est_tyot if e[1] is heitto]:      # lopullinen estimaatti: odota kesken oleva (<= ~1 s), sitten uusi
                    try:
                        self._aseta_viivat(heitto, e[0].result(), e[2])
                    except Exception:
                        pass
                    self.est_tyot.remove(e)
            self.est_tyot.append((self.est_pool.submit(laske_viivat, ff, XX, YY, self.fps_lahde, self.fps), heitto, f, len(ns)))
        else:
            self._aseta_viivat(heitto, laske_viivat(ff, XX, YY, self.fps_lahde, self.fps), f, getattr(heitto, "lopullinen", False))

    @staticmethod
    def _aseta_viivat(heitto, r, f, lopullinen=False):
        """Uusi estimaatti heitolle. vakaa = merkitseva irroitus, vapaata lentoa >= 250 cm, irroituspaikka muuttunut < 30 cm edellisesta estimaatista."""
        if r is None or "virhe" in r:
            return
        heitto.viivat_hist = (heitto.viivat_hist + [r["irr_Y"]])[-3:]
        edel = heitto.viivat_hist[-2] if len(heitto.viivat_hist) >= 2 else None
        r["vakaa"] = bool(r["merkitseva"] and r["vapaa_matka_cm"] >= 250.0 and edel is not None and abs(r["irr_Y"] - edel) < 30.0 and r["irr_virhe_deg"] < 0.4)
        if lopullinen and r["merkitseva"]:
            r["vakaa"] = True
        L = 3474.7
        heitto.viivat_loki.append((f, r["irr_Y"], r["irr_suunta_deg"], r["irr_X"] + np.tan(np.radians(r["irr_suunta_deg"])) * (L - r["irr_Y"]), r["liuku_a"] + r["liuku_b"] * L, r["vakaa"]))
        heitto.viivat, heitto.viivat_f = r, f

    def kuvat(self):
        """Piirtajalle: vahvistettujen heitojen tilannekuvat (dict)."""
        return [dict(tunnus=h.tunnus, numero=i + 1, puoli=h.puoli, vari=h.vari, f_alku=h.f_alku, f_loppu=h.f_loppu, pisteet=h.pisteet(), viivat=h.viivat, tila=h.tila)
                for i, h in enumerate(list(self.heitot))]

    def _keraa_estimaatit(self, odota=False):
        jaa = []
        for e in self.est_tyot:
            fut, heitto, f, n = e
            if fut.done() or odota:
                try:
                    self._aseta_viivat(heitto, fut.result(), f, getattr(heitto, "lopullinen", False))
                except Exception:
                    pass
            else:
                jaa.append(e)
        self.est_tyot = jaa

    def loppu(self, heitto):
        """Heitto paattyi: lopullinen estimaatti kaikesta aineistosta. Synkronisessa tilassa heti; reaaliajassa kun heiton B-tyot ovat valmiit (odota())."""
        self.tapahtumat.append(("valmis", heitto.tunnus, heitto.f_loppu))
        if self.synkroninen:
            self.odota(f_asti=10 ** 12)
            self._lopullinen(heitto)
        else:
            heitto.odottaa_loppua = True

    def _lopullinen(self, heitto):
        heitto.odottaa_loppua = False
        heitto.lopullinen = True
        self.paivita_viivat(heitto, heitto.f_loppu, pakota=True)

    def _loppujen_tarkistus(self):
        for h in list(self.valmiit):
            if getattr(h, "odottaa_loppua", False) and not any(t[2] is h for t in self.tyot):
                self._lopullinen(h)

    def viimeistele(self):
        """Videon loputtua: odota kesken olevat tyot."""
        self.odota(f_asti=10 ** 12)
        self._loppujen_tarkistus()
        self._keraa_estimaatit(odota=True)

    # ---- paasilmukka -----------------------------------------------------------------------------------------------
    def syota(self, f, ruutu):
        """Yksi ruutu (BGR, 1920 x 1080). Kopioruudut ohitetaan."""
        if self._kopio(ruutu):
            self.kopioita += 1
            return
        self.yksilollisia += 1
        if self._fps_auto and self.yksilollisia + self.kopioita >= 150:
            self._paata_fps_lahde()
        for hp in self.paat.values():
            hp.kasittele(f, ruutu)
            h = hp.heitto
            if h is not None and h.tila == "vahvistettu" and h not in self.heitot:
                self.heitot.append(h)
        self.odota(f - self.viive_b if self.synkroninen else None)
        for hp in self.paat.values():
            if hp.heitto is not None and hp.heitto.tila == "vahvistettu":
                self.paivita_viivat(hp.heitto, f)

    def ohita(self):
        """Ruutu nakyy tulosteessa mutta sita ei analysoida (analyysi myohassa): kopioruututesti ei saa verrata kahta eri ruutua."""
        self.edellinen_harmaa = None

    def sulje(self):
        for p in (self.pool, self.est_pool):
            if p is not None:
                p.shutdown(wait=False, cancel_futures=True)
