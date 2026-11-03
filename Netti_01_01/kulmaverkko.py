"""Kahvan kulman neuroverkko (puhdas NumPy, ei JAX / torch -riippuvuutta): top-down-leike kivesta (56 x 56 x 3, uint8) -> kulman jakauma (60 lokeroa, 3 astetta, jakso 180), ja heiton vakiokiertosovitus.

Verkko on opetettu ihmisen merkitsemillä (epätäydellisillä) kulmilla (`kierre_validointi/`) ja heiton tarkalla kiertoradalla (kierto hidastuu lineaarisesti), joka jalostaa viivat ruutukohtaisiksi kulmiksi myös merkitsemättömille heitoille (tools/kulmaverkko_opeta.py); vanhan koodin kulmia ei käytetä.
Kulma: maan tason kulma X-akselista Y-akseliin, 0 = vaakasuora, + = vastapäivään ylhäältä katsoen, jakso 180 (sama kuin merkintäsivun `kulma_deg` top-down-leikkeessä, A = (-1, +1)).
Leike: `tools/kierre_leikkeet.topdown_leike` (z = 20 cm, 3,5 px / cm, 32 x 32 cm) pienennettynä 56 x 56 px:ksi (`cv2.INTER_AREA`); ks. `leike56`.

    mallit = [kv.lataa("kulmaverkko_paino_pieni.npz"), kv.lataa("kulmaverkko_paino_resf.npz")]
    p = kv.ennusta(mallit, leikkeet)                      # (n, 60): TTA (kierrot x peilaus) ja verkkojen geometrinen keskiarvo
    kulma, R = kv.kulma_jakaumasta(p)                     # ruutukohtainen kulma (jakauman ympyrakeskiarvo) ja keskittyneisyys
    r = kv.sovita_heitto(p, t)                            # vakiokierto: sovitettu kulma ruuduittain, omega, suunnan varmuus
"""
import cv2
import numpy as np

KUL = np.arange(0.0, 180.0, 3.0)        # kulmalokerot (aste)
NB = len(KUL)
KOKO = 56                               # leikkeen sivu (px)
K_KIERRE = 0.021                        # kierteen hidastumisvakio (1/s): omega(t) = omega_0 (1 - K (t - t_0)), IMU-mittaus 2,1 % / s (sama luku kuin netti_kierre.K_KIERRE)
OM_RUUDUKKO = np.arange(-600.0, 600.1, 6.0)     # omegan ruudukko (aste / s) vakiokiertosovituksessa
TTA_KIERROT = (-60, -30, 0, 30, 60)     # ennustuksen testiaikaiset kierrot (aste; 3:n monikertoja, jotta jakauma siirtyy kokonaisten lokeroiden verran)

VYOHYKE_Y_CM = (682.8, 702.6)    # kiven y-asema (cm), jossa sininen hog-viiva näkyy kuvassa kahvan takana (Video1 heitto 59: t = 0 ... 0,1 s); kamera kuvaa vinosta ja kahva on kiven yläreunassa, joten vyöhyke on eri kohdassa kuin itse hog-viiva
VYOHYKE2_Y_CM = (520.0, 540.0)   # toinen vyöhyke: kahvan takana on jäähän maalattu iso mainosteksti; sama kohta kaikissa videoissa, molemmilla puolilla ja kaikilla kiertosuunnilla (piikki y 522-533 cm)
VYOHYKKEET_Y_CM = (VYOHYKE2_Y_CM, VYOHYKE_Y_CM)    # kaikki tunnetut vyöhykkeet, joiden ruudut jätetään oletuksena pois radan sovituksesta
ENNAKKO_MEDIAANI, ENNAKKO_SIGMA_LOG = 100.0, 1.0   # |omega| lognormaali ennakko (aste / s): sulkee pois omegan ~ 0 ja ylisuuret nopeudet
MOODI_IKKUNA = 4                                    # kulman posteriorin huipun ympäristö: +- 4 lokeroa (+- 12 astetta)
SIGMA_VIIVA, EPS_VIIVA = 4.0, 0.08                 # ihmisen viivan epätäydellisyys: useimmat viivat 4 asteen hajonnalla, 8 % viivoista täysin pielessä (tasajakauma 180 asteen yli; 39 viivasta 2 poikkesi radastaan yli 15 astetta)


# ---------------------------------------------------------------------------------------------------------------------
# KUVAN MUUNNOKSET (opetuksen ja TTA:n yhteinen konventio; testattu tools/testaa_kulmaverkko.py)
# ---------------------------------------------------------------------------------------------------------------------
def kierra(kuva, a):
    """Kuva kulman a (aste) verran vastapäivään (näytön mukaan) leikkeen keskipisteen ympäri; palkin kulma kasvaa a:n verran."""
    c = (kuva.shape[0] - 1) / 2.0
    return cv2.warpAffine(kuva, cv2.getRotationMatrix2D((c, c), float(a), 1.0), (kuva.shape[1], kuva.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def peilaa(kuva):
    """Vaakapeilaus: palkin kulma theta -> -theta (jakso 180)."""
    return np.ascontiguousarray(kuva[:, ::-1])


def leike56(td):
    """Top-down-leike (112 x 112, `kierre_leikkeet.topdown_leike`) -> verkon syöte 56 x 56 x 3 (uint8)."""
    return cv2.resize(td, (KOKO, KOKO), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------------------------------------------------
# VERKKO (NumPy)
# ---------------------------------------------------------------------------------------------------------------------
def _konv(x, w, b, askel=1):
    """3 x 3 -konvoluutio, 'SAME' kuten JAX:n lax: askel 2 parillisella koolla täyttää vain oikean ja alareunan (pad_lo = 0, pad_hi = 1)."""
    B, H, W, C = x.shape
    if askel == 1:
        p = np.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)))
        Ho, Wo = H, W
        ikkunat = [p[:, i:i + H, j:j + W, :] for i in range(3) for j in range(3)]
    else:
        p = np.pad(x, ((0, 0), (0, 1), (0, 1), (0, 0)))
        Ho, Wo = H // 2, W // 2
        ikkunat = [p[:, i:i + 2 * Ho:2, j:j + 2 * Wo:2, :] for i in range(3) for j in range(3)]
    ulos = np.concatenate(ikkunat, axis=-1).reshape(B * Ho * Wo, -1) @ w.reshape(-1, w.shape[-1]) + b
    return ulos.reshape(B, Ho, Wo, -1)


def _relu(x):
    return np.maximum(x, 0.0)


def _pool(x):
    B, H, W, C = x.shape
    return x.reshape(B, H // 2, 2, W // 2, 2, C).max(axis=(2, 4))


def lataa(polku):
    """Verkko tiedostosta (npz: arch = 'pieni' | 'resf', blocks, parametrit 'kerros/w' ja 'kerros/b'; valinnainen 'kierrot' = verkon oletuksena käyttämät TTA-kierrot, jos ne poikkeavat TTA_KIERROT:sta)."""
    d = np.load(polku, allow_pickle=False)
    malli = dict(arch=str(d["arch"]), blocks=int(d["blocks"]), P={k: d[k] for k in d.files if k not in ("arch", "blocks", "kierrot")})
    if "kierrot" in d.files:
        malli["kierrot"] = tuple(float(a) for a in d["kierrot"])
    return malli


def tallenna(polku, arch, P, blocks=2, kierrot=None):
    """Verkko tiedostoon: P = {'kerros/w': taulukko, ...}; kierrot = verkon oletus-TTA-kierrot (aste), jos ne eivät ole TTA_KIERROT (esim. (0,) kun opetuksessa ei kierretty kuvaa)."""
    lisa = {} if kierrot is None else dict(kierrot=np.asarray(kierrot, np.float32))
    np.savez_compressed(polku, arch=arch, blocks=blocks, **lisa, **{k: np.asarray(v, np.float32) for k, v in P.items()})


def etenemista(malli, x):
    """Verkon läpivienti: x (B, 56, 56, 3) float32 [-0,5 ... 0,5] -> softmax (B, 60)."""
    P = malli["P"]
    if malli["arch"] == "pieni":
        for nimi in ("c1a", "c1b", "c2a", "c2b", "c3a", "c3b"):
            x = _relu(_konv(x, P[nimi + "/w"], P[nimi + "/b"]))
            if nimi.endswith("b"):
                x = _pool(x)
    else:
        x = _relu(_konv(x, P["stem/w"], P["stem/b"]))
        s = 0
        while f"s{s}_down/w" in P:
            x = _relu(_konv(x, P[f"s{s}_down/w"], P[f"s{s}_down/b"], askel=2))
            for b in range(malli["blocks"]):
                h = _relu(_konv(x, P[f"s{s}_b{b}_c1/w"], P[f"s{s}_b{b}_c1/b"]))
                x = _relu(x + _konv(h, P[f"s{s}_b{b}_c2/w"], P[f"s{s}_b{b}_c2/b"]))
            s += 1
    x = _relu(x.reshape(len(x), -1) @ P["d1/w"] + P["d1/b"])
    z = x @ P["d2/w"] + P["d2/b"]
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _ennusta_yksi(malli, leikkeet, tta, kierrot, peilaus, erakoko):
    """Yhden verkon jakauma (n, 60) alkuperäisille leikkeille; TTA: näkymät (kierto a, peilaus) -> jakauma siirretään takaisin (lokerot -a / 3; peilauksessa lokero k -> -k) ja log-keskiarvo.
    kierrot = None: verkon oma oletus (`malli["kierrot"]`, muuten TTA_KIERROT)."""
    if kierrot is None:
        kierrot = malli.get("kierrot", TTA_KIERROT)
    nakymat = []
    for peil in ((False, True) if (tta and peilaus) else (False,)):
        for a in (kierrot if tta else (0,)):
            xv = np.empty(leikkeet.shape, np.float32)
            for i in range(len(leikkeet)):
                x = leikkeet[i]
                if peil:
                    x = peilaa(x)
                if a:
                    x = kierra(x, a)
                xv[i] = x.astype(np.float32) / 255.0 - 0.5
            p = np.concatenate([etenemista(malli, xv[j:j + erakoko]) for j in range(0, len(xv), erakoko)])
            p = np.roll(p, -int(round(a / 3.0)), axis=1)             # poista kierto
            if peil:
                p = p[:, (-np.arange(NB)) % NB]                      # poista peilaus
            nakymat.append(p)
    return _log_keskiarvo(nakymat)


def _log_keskiarvo(jakaumat):
    if len(jakaumat) == 1:
        return jakaumat[0].astype(np.float32)
    lp = np.log(np.clip(np.stack(jakaumat), 1e-8, None)).mean(axis=0)
    lp -= lp.max(axis=1, keepdims=True)
    p = np.exp(lp)
    return (p / p.sum(axis=1, keepdims=True)).astype(np.float32)


def ennusta(mallit, leikkeet, tta=True, kierrot=None, peilaus=True, erakoko=64):
    """Kulman jakauma (n, 60) leikkeille (n, 56, 56, 3, uint8). mallit = yksi verkko tai lista (jakaumien geometrinen keskiarvo). kierrot = TTA-kierrot (aste); oletus (None): kunkin verkon oma oletus
    (opetuksessa kuvaa kiertänyt verkko: TTA_KIERROT; kierrotta opetettu verkko: vain peilaus, sillä kierretty näkymä on sille vieras, ks. README)."""
    if isinstance(mallit, dict):
        mallit = [mallit]
    return _log_keskiarvo([_ennusta_yksi(m, leikkeet, tta, kierrot, peilaus, erakoko) for m in mallit])


def vyohykkeessa(Y, vali=None):
    """Ruudut, joissa kivi on y-vyöhykkeessä, jossa kahvan takana on voimakas vaakasuora kuvio ja verkon ruutuarvio vetäytyy kohti vaakasuuntaa tai menee täysin pieleen (README, kolmas kierros): `VYOHYKE_Y_CM` (hog-viiva)
    ja `VYOHYKE2_Y_CM` (jäähän maalattu mainosteksti). vali = None: kaikki tunnetut vyöhykkeet (`VYOHYKKEET_Y_CM`); (alku, loppu) tai lista sellaisia: vain annetut välit (cm, päät mukana). Palauttaa (n,) bool;
    sovita_heitto(..., ohita=vyohykkeessa(Y)) jättää niiden jakaumat pois radan sovituksesta."""
    Y = np.asarray(Y, float)
    v = np.asarray(VYOHYKKEET_Y_CM if vali is None else vali, float)
    v = v[None, :] if v.ndim == 1 else v
    m = np.zeros(Y.shape, bool)
    for alku, loppu in v:
        m |= (Y >= alku) & (Y <= loppu)
    return m


def johdonmukaisuus(jakaumat, ajat, K=K_KIERRE, lam=0.5, ohita=None):
    """Verkon johdonmukaisuus ilman merkintöjä: kunkin heiton ruutujakaumiin (n_i, 60) ajoilla (n_i,) sovitetaan tarkka rata (`sovita_heitto`) ja mitataan ruutujen jakaumien huippujen ero sovitettuun kulmaan.
    Mittaa ruutukohtaista satunnaisvirhettä kaikilla kivillä, joilla ei ole merkintöjä (laki on tarkka, joten sovitus on vain 2 parametria ruutujen yli). Ei havaitse virhettä, joka on koko heiton yhteinen (esim. vakiosiirtymä).
    ohita = None / lista (n_i,) bool-taulukoita heittoa kohti (esim. `vyohykkeessa(Y)`): niiden ruutujen jakaumia ei käytetä radan sovituksessa, mutta ero niihin mitataan edelleen (`ohitettu` erottaa ne).
    Palauttaa dict: ero (kaikkien ruutujen |ero| asteina), kulma (sovitettu kulma mod 180, ruuduittain), heitto (heiton indeksi ruuduittain), ohitettu (ruuduittain, bool), suunta_p, omega (heittokohtaiset)."""
    ero, kul, hid, oh, sp, om = [], [], [], [], [], []
    for i, (p, t) in enumerate(zip(jakaumat, ajat)):
        o = None if ohita is None else np.asarray(ohita[i], bool)
        r = sovita_heitto(p, t, lam=lam, K=K, ohita=o)
        a1, _ = kulma_jakaumasta(p)
        ero.append(np.abs(ero180(a1, r["kulma"])))
        kul.append(r["kulma"] % 180.0)
        hid.append(np.full(len(t), i))
        oh.append(np.zeros(len(t), bool) if o is None else o)
        sp.append(r["suunta_p"])
        om.append(r["omega"])
    return dict(ero=np.concatenate(ero), kulma=np.concatenate(kul), heitto=np.concatenate(hid), ohitettu=np.concatenate(oh), suunta_p=np.array(sp), omega=np.array(om))


def kulma_jakaumasta(p):
    """Jakauma (n, 60) -> (kulma aste jakso 180 [ympyrakeskiarvo], R = keskittyneisyys 0 ... 1)."""
    z = (p * np.exp(2j * np.deg2rad(KUL))[None, :]).sum(axis=1)
    return (np.degrees(np.angle(z)) / 2.0) % 180.0, np.abs(z)


def ero180(a, b):
    """Kulmaero a - b välille -90 ... 90."""
    return (np.asarray(a, float) - np.asarray(b, float) + 90.0) % 180.0 - 90.0


# ---------------------------------------------------------------------------------------------------------------------
# HEITON KIERTORATA: kierto hidastuu lineaarisesti (tarkka laki): omega(s) = omega_0 (1 - K s), s = t - t_0, joten kulma(t) = th0 + omega_0 g(s), g(s) = s - K s^2 / 2
# ---------------------------------------------------------------------------------------------------------------------
def rata_aika(s, K=K_KIERRE):
    """Kiertoradan aikafunktio g(s) = s - K s^2 / 2 (lineaarisesti hidastuva kierto; s sekunteina alkuhetkestä, voi olla negatiivinen): kulma = th0 + omega_0 g(s). Kierto pysähtyy hetkellä s = 1 / K."""
    s = np.minimum(np.asarray(s, float), 1.0 / K)
    return s - 0.5 * K * s * s


def _jaksollinen(pen, th):
    """Rangaistus pen (n, 60) kulmassa th ((n,) tai (m, n)), lineaarinen interpolointi, jakso 180; rivi i käyttää pen[i]."""
    n, nb = pen.shape
    x = (np.asarray(th) % 180.0) / 3.0
    i0 = np.floor(x).astype(int) % nb
    i1 = (i0 + 1) % nb
    w = x - np.floor(x)
    rivi = np.arange(n) if x.ndim == 1 else np.arange(n)[None, :]
    return pen[rivi, i0] * (1 - w) + pen[rivi, i1] * w


def _ennakko(om, mediaani=ENNAKKO_MEDIAANI, sigma_log=ENNAKKO_SIGMA_LOG):
    a = np.abs(om)
    return np.where(a > 1.0, -0.5 * (np.log(np.maximum(a, 1e-9) / mediaani) / sigma_log) ** 2 - np.log(np.maximum(a, 1e-9)), -50.0)


def viiva_rangaistus(d, sigma=SIGMA_VIIVA, eps=EPS_VIIVA):
    """Ihmisen viivan epätäydellisyys: ero d (aste) viivaan on useimmiten pieni (normaali, sigma), mutta osuus eps viivoista on aivan pielessä (tasajakauma jaksolla 180). Rangaistus = -log (sekoitusjakauma), nolla kohdassa d = 0,
    ylhäältä rajoitettu: yksittäinen väärä viiva ei voi kumota muita viivoja eikä verkon todisteita."""
    nimitt = (1.0 - eps) / (sigma * np.sqrt(2.0 * np.pi))
    return -np.log((nimitt * np.exp(-0.5 * (d / sigma) ** 2) + eps / 180.0) / (nimitt + eps / 180.0))


def sovita_heitto(p, t, lam=0.5, katkaisu=6.0, K=K_KIERRE, viivat=None, ennakko=None, ohita=None):
    """Kiertoratasovitus ruutujen jakaumiin p (n, 60) ajoilla t (n, nouseva, s): posteriori (th0, omega) ruudukossa, rangaistus pen = min(lam (-log p - min), katkaisu); p = None: ei verkon todistetta (vain viivat).
    viivat = dict(idx, kulma, sigma, eps): ihmisen viivat ruuduissa idx (kulmat asteina) epätäydellisinä (`viiva_rangaistus`), rata on silti tarkka (lineaarinen hidastuminen, `rata_aika`).
    ennakko = (mediaani, sigma_log): |omega|:n lognormaali ennakko (oletus ENNAKKO_*).
    ohita = (n,) bool: ruudut, joiden jakaumaa ei käytetä sovituksessa (esim. `vyohykkeessa(Y)`); rata kulkee silti kaikkien ruutujen läpi ja `kulma` ym. lasketaan niille, `ohitettu` palautetaan sellaisenaan.
    Palauttaa dict: kulma (n,), R (n,) = sovitetun kulman posteriorin keskittyneisyys, sigma (n,) = sitä vastaava hajonta (aste), kulma_moodi / sigma_moodi / massa_moodi (n,) = posteriorin huippu, sen ympäristön (+- MOODI_IKKUNA
    lokeroa) hajonta (katkaistu ikkunaan) ja todennäköisyysmassa (viivojen mahdolliset virheet jäävät huipun ulkopuolelle), residuaali (n,) = |ruudun jakauman huippu - sovitettu kulma| (p annettu), omega ja omega_sd (aste / s,
    valitulla suunnalla), suunta_p (valitun kiertosuunnan todennakoisyys), p_ccw (vastapaivaan)."""
    t = np.asarray(t, float)
    n = len(t)
    if p is None:
        pen = np.zeros((n, NB))
    else:
        pen = np.minimum((-np.log(np.clip(p, 1e-6, None)) - (-np.log(np.clip(p, 1e-6, None))).min(axis=1, keepdims=True)) * lam, katkaisu)
        if ohita is not None:
            pen[np.asarray(ohita, bool)] = 0.0
    tau = rata_aika(t - t[0], K)
    om = OM_RUUDUKKO
    J = np.empty((NB, len(om)))
    for ia, th0 in enumerate(KUL):
        J[ia] = _jaksollinen(pen, th0 + om[:, None] * tau[None, :]).sum(axis=1)
    if viivat is not None and len(viivat["idx"]):
        idx = np.asarray(viivat["idx"], int)
        ero = ero180(KUL[:, None, None] + om[None, :, None] * tau[idx][None, None, :], np.asarray(viivat["kulma"], float)[None, None, :])
        J += viiva_rangaistus(ero, viivat.get("sigma", SIGMA_VIIVA), viivat.get("eps", EPS_VIIVA)).sum(axis=2)
    lw = -J + _ennakko(om, *(ennakko or (ENNAKKO_MEDIAANI, ENNAKKO_SIGMA_LOG)))[None, :]
    lw -= lw.max()
    w = np.exp(lw)
    w /= w.sum()
    kulma = np.empty(n)
    R = np.empty(n)
    k_moodi, s_moodi, massa = np.empty(n), np.empty(n), np.empty(n)
    ikkuna = np.arange(-MOODI_IKKUNA, MOODI_IKKUNA + 1)
    for i in range(n):
        ang = KUL[:, None] + om[None, :] * tau[i]
        z = (w * np.exp(2j * np.radians(ang))).sum()
        kulma[i] = (np.degrees(np.angle(z)) / 2.0) % 180.0
        R[i] = abs(z)
        hist = np.bincount((((ang % 180.0) / 3.0).astype(int) % NB).ravel(), weights=w.ravel(), minlength=NB)         # kulman posteriori lokeroina
        lok = (hist.argmax() + ikkuna) % NB                                                                              # huippulokero ja sen ympäristö
        massa[i] = hist[lok].sum()
        zm = (hist[lok] * np.exp(2j * np.radians(KUL[lok]))).sum() / max(massa[i], 1e-12)
        k_moodi[i] = (np.degrees(np.angle(zm)) / 2.0) % 180.0
        s_moodi[i] = max(np.degrees(0.5 * np.sqrt(-2.0 * np.log(np.clip(abs(zm), 1e-6, 1.0 - 1e-9)))), 1.0)
    p_om = w.sum(axis=0)
    p_ccw = float(p_om[om > 0].sum())
    s = 1 if p_ccw >= 0.5 else -1
    m = (om * s) > 0
    pp = p_om[m] / max(p_om[m].sum(), 1e-12)
    mu = float((om[m] * pp).sum())
    sigma = np.degrees(0.5 * np.sqrt(-2.0 * np.log(np.clip(R, 1e-6, 1.0 - 1e-9))))
    residuaali = np.abs(ero180(KUL[np.argmax(p, axis=1)], kulma)) if p is not None else np.zeros(n)
    return dict(kulma=kulma, R=R, sigma=sigma, kulma_moodi=k_moodi, sigma_moodi=s_moodi, massa_moodi=massa, residuaali=residuaali, ohitettu=np.zeros(n, bool) if ohita is None else np.asarray(ohita, bool), omega=mu, omega_sd=float(np.sqrt(((om[m] - mu) ** 2 * pp).sum())), suunta_p=max(p_ccw, 1.0 - p_ccw), p_ccw=p_ccw)
