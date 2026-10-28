"""Kahvan kulman neuroverkko (puhdas NumPy, ei JAX / torch -riippuvuutta): top-down-leike kivesta (56 x 56 x 3, uint8) -> kulman jakauma (60 lokeroa, 3 astetta, jakso 180), ja heiton vakiokiertosovitus.

Verkko on opetettu ihmisen merkitsemillä kulmilla (`kierre_validointi/`), niiden vakiokiertoradalla ja merkitsemättömien heittojen vakiokiertosovituksella (tools/kulmaverkko_opeta.py); vanhan koodin kulmia ei käytetä.
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
K_KIERRE = 0.021                        # kierteen vaimenemisvakio (1/s), sama kuin netti_kierre.K_KIERRE (IMU-mittaus)
OM_RUUDUKKO = np.arange(-600.0, 600.1, 6.0)     # omegan ruudukko (aste / s) vakiokiertosovituksessa
TTA_KIERROT = (-60, -30, 0, 30, 60)     # ennustuksen testiaikaiset kierrot (aste; 3:n monikertoja, jotta jakauma siirtyy kokonaisten lokeroiden verran)
ENNAKKO_MEDIAANI, ENNAKKO_SIGMA_LOG = 100.0, 1.0   # |omega| lognormaali ennakko (aste / s): sulkee pois omegan ~ 0 ja ylisuuret nopeudet


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
    """Verkko tiedostosta (npz: arch = 'pieni' | 'resf', blocks, parametrit 'kerros/w' ja 'kerros/b')."""
    d = np.load(polku, allow_pickle=False)
    return dict(arch=str(d["arch"]), blocks=int(d["blocks"]), P={k: d[k] for k in d.files if k not in ("arch", "blocks")})


def tallenna(polku, arch, P, blocks=2):
    """Verkko tiedostoon: P = {'kerros/w': taulukko, ...}."""
    np.savez_compressed(polku, arch=arch, blocks=blocks, **{k: np.asarray(v, np.float32) for k, v in P.items()})


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
    """Yhden verkon jakauma (n, 60) alkuperäisille leikkeille; TTA: näkymät (kierto a, peilaus) -> jakauma siirretään takaisin (lokerot -a / 3; peilauksessa lokero k -> -k) ja log-keskiarvo."""
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


def ennusta(mallit, leikkeet, tta=True, kierrot=TTA_KIERROT, peilaus=True, erakoko=64):
    """Kulman jakauma (n, 60) leikkeille (n, 56, 56, 3, uint8). mallit = yksi verkko tai lista (jakaumien geometrinen keskiarvo)."""
    if isinstance(mallit, dict):
        mallit = [mallit]
    return _log_keskiarvo([_ennusta_yksi(m, leikkeet, tta, kierrot, peilaus, erakoko) for m in mallit])


def kulma_jakaumasta(p):
    """Jakauma (n, 60) -> (kulma aste jakso 180 [ympyrakeskiarvo], R = keskittyneisyys 0 ... 1)."""
    z = (p * np.exp(2j * np.deg2rad(KUL))[None, :]).sum(axis=1)
    return (np.degrees(np.angle(z)) / 2.0) % 180.0, np.abs(z)


def ero180(a, b):
    """Kulmaero a - b välille -90 ... 90."""
    return (np.asarray(a, float) - np.asarray(b, float) + 90.0) % 180.0 - 90.0


# ---------------------------------------------------------------------------------------------------------------------
# HEITON VAKIOKIERTO: kulma(t) = th0 + omega G(t), G(t) = (1 - exp(-K (t - t_0))) / K  (kierteen vaimeneminen mukana; kierron alusta ei oleteta tasannetta kasi-vapailla ruuduilla)
# ---------------------------------------------------------------------------------------------------------------------
def _jaksollinen(pen, th):
    """Rangaistus pen (n, 60) kulmassa th ((n,) tai (m, n)), lineaarinen interpolointi, jakso 180; rivi i käyttää pen[i]."""
    n, nb = pen.shape
    x = (np.asarray(th) % 180.0) / 3.0
    i0 = np.floor(x).astype(int) % nb
    i1 = (i0 + 1) % nb
    w = x - np.floor(x)
    rivi = np.arange(n) if x.ndim == 1 else np.arange(n)[None, :]
    return pen[rivi, i0] * (1 - w) + pen[rivi, i1] * w


def _ennakko(om):
    a = np.abs(om)
    return np.where(a > 1.0, -0.5 * (np.log(np.maximum(a, 1e-9) / ENNAKKO_MEDIAANI) / ENNAKKO_SIGMA_LOG) ** 2 - np.log(np.maximum(a, 1e-9)), -50.0)


def sovita_heitto(p, t, lam=0.5, katkaisu=6.0, K=K_KIERRE):
    """Vakiokiertosovitus ruutujen jakaumiin p (n, 60) ajoilla t (n, nouseva, s): posteriori (th0, omega) ruudukossa, rangaistus pen = min(lam (-log p - min), katkaisu).
    Palauttaa dict: kulma (n,) ja R (n,) = sovitetun kulman posteriorin ympyrakeskiarvo / keskittyneisyys, residuaali (n,) = |ruudun jakauman huippu - sovitettu kulma|, omega ja omega_sd (aste / s,
    valitulla suunnalla), suunta_p (valitun kiertosuunnan todennakoisyys), p_ccw (vastapaivaan)."""
    t = np.asarray(t, float)
    n = len(t)
    pen = np.minimum((-np.log(np.clip(p, 1e-6, None)) - (-np.log(np.clip(p, 1e-6, None))).min(axis=1, keepdims=True)) * lam, katkaisu)
    tau = (1.0 - np.exp(-K * (t - t[0]))) / K
    om = OM_RUUDUKKO
    J = np.empty((NB, len(om)))
    for ia, th0 in enumerate(KUL):
        J[ia] = _jaksollinen(pen, th0 + om[:, None] * tau[None, :]).sum(axis=1)
    lw = -J + _ennakko(om)[None, :]
    lw -= lw.max()
    w = np.exp(lw)
    w /= w.sum()
    kulma = np.empty(n)
    R = np.empty(n)
    for i in range(n):
        z = (w * np.exp(2j * np.radians(KUL[:, None] + om[None, :] * tau[i]))).sum()
        kulma[i] = (np.degrees(np.angle(z)) / 2.0) % 180.0
        R[i] = abs(z)
    p_om = w.sum(axis=0)
    p_ccw = float(p_om[om > 0].sum())
    s = 1 if p_ccw >= 0.5 else -1
    m = (om * s) > 0
    pp = p_om[m] / max(p_om[m].sum(), 1e-12)
    mu = float((om[m] * pp).sum())
    return dict(kulma=kulma, R=R, residuaali=np.abs(ero180(KUL[np.argmax(p, axis=1)], kulma)), omega=mu, omega_sd=float(np.sqrt(((om[m] - mu) ** 2 * pp).sum())),
                suunta_p=max(p_ccw, 1.0 - p_ccw), p_ccw=p_ccw)
