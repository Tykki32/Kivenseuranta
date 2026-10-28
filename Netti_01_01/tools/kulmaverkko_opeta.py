"""Kahvan kulman neuroverkon opetus: ihmisen merkitsemät kulmat + kierteen vakionopeus (ei vanhan koodin tunnistamia kulmia).

Vaiheet (jokainen oma alikomentonsa; tulokset pickle-tiedostoina):

  leikkeet   Heittojen käsi-vapaat ruudut top-down-leikkeinä (56 x 56 x 3) videolta: irroituksen jälkeen kuvan reunaan, ensimmäisestä ruudusta alkaen, jossa käsi ei enää ole kahvalla (kuten merkintätyökalussa).
             python tools/kulmaverkko_opeta.py leikkeet --ajo Matarmaa Matarmaa.mp4 r_Matarmaa/radat.pkl --ajo ... --ulos leikkeet.pkl
  opettaja   Ensimmäinen verkko pelkillä ihmisen kulmilla: viivat sovitetaan heittokohtaiseen vakiokiertorataan theta(t) = th_ref + omega g(t), g = (1 - exp(-K s)) / K, ja rata antaa kulman koko merkityn aikaikkunan
             ruuduille (ankkuroidut ruudut); muiden heittojen yksittäiset viivat otetaan sellaisinaan. Opetuksessa kuvan kierto (kulma + kierto) ja peilaus (kulma -> -kulma) tuottavat kaikki kulmat.
             python tools/kulmaverkko_opeta.py opettaja --leikkeet leikkeet.pkl --ulos opettaja.pkl [--verkko pieni] [--askelta 4000]
  pseudo     Opettajan (tai verkkojen yhdistelmän) ennusteet merkitsemättömille heitoille (TTA) sovitetaan vakiokiertoon (kulmaverkko.sovita_heitto); sovitettu rata on heiton pseudomerkintä.
             python tools/kulmaverkko_opeta.py pseudo --leikkeet leikkeet.pkl --verkot opettaja.pkl --ulos pseudo.pkl
  oppilas    Uusi verkko ihmisen ruuduilla ja pseudomerkinnöillä (itseopetus; kierroksen voi toistaa: oppilas -> pseudo -> oppilas, README: toinen kierros paransi, kolmas ei enää).
             python tools/kulmaverkko_opeta.py oppilas --leikkeet leikkeet.pkl --pseudo pseudo.pkl --ulos oppilas.pkl [--verkko resf --kw '{"ch": [24, 48, 96], "blocks": 2}' --askelta 6000 --p-kierto 0.3 --kierto-max 30]
  arvioi     Verkkojen virhe ihmisen viivoihin (yksi ruutu + TTA; verkko + vakiokierto) pois jätetyillä heitoilla (--pois), jotka eivät ole missään opetuksessa.
             python tools/kulmaverkko_opeta.py arvioi --leikkeet leikkeet.pkl --verkot oppilas.pkl --pois Matarmaa:69,Saarelainen:68
  vie        Opetettu verkko NumPy-muotoon (kulmaverkko.py): python tools/kulmaverkko_opeta.py vie --verkko oppilas.pkl --ulos kulmaverkko_paino_resf.npz
  ihmisruudut  Ihmisen viivalla merkityt leikkeet ja kulmat pieneksi npz-tiedostoksi (reposta löytyvä testiaineisto toimitettujen painojen tarkistukseen, tools/testaa_kulmaverkko.py).

Opetus vaatii JAX:n ja optaxin (`pip install jax optax`; CPU riittää); ennustus (kulmaverkko.py) ei tarvitse niitä. Arkkitehtuurit: 'pieni' (6 konvoluutiota, 0,5 M parametria) ja 'resf' (jäännösverkko ilman normalisointia, 14 kerrosta,
1,7-2,5 M parametria; jäännöslohkon toinen konvoluutio alustetaan nollaksi). Pois jätettävät heitot (--pois) jäävät pois myös merkitsemättömien heittojen joukosta.
Kulmat: maan tason kulma, 0 = vaakasuora, + = vastapäivään, jakso 180 (merkintäsivun kulma_deg top-down-leikkeessä, A = (-1, +1))."""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np

KANSIO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, KANSIO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import kulmaverkko as kv  # noqa: E402

NB = kv.NB
KUL = kv.KUL
K_KIERRE = kv.K_KIERRE
SIGMA_VIIVA = 2.5                  # ihmisen viivan kohina (aste): suoran sovituksen jäännös 0,1-2,6 astetta
RAJA_RMS, RAJA_IKKUNA, MARGINAALI = 3.0, 0.39, 0.2     # ankkurointi: suoran jäännös (aste), merkittyjen ruutujen aikaikkuna (s), ankkurointi ikkunan ulkopuolelle (s)
LEIKE_KOKO = kv.KOKO


# ---------------------------------------------------------------------------------------------------------------------
# LEIKKEET VIDEOLTA
# ---------------------------------------------------------------------------------------------------------------------
def leikkeet_heitolle(cap, h, P, min_ruutuja=6, y_max_cm=800.0):
    """Heiton käsi-vapaat ruudut: dict(tiles, f, t, X, Y, kasi, t_irr, vari, puoli) tai None. Ruudut kuten tools/kierre_leikkeet.kasi_vapaat_ruudut, mutta kuvan reunaan (y_max_cm) asti; leikkeen pitää mahtua kuvaan."""
    import cv2
    import kierre_leikkeet as kl
    v = h.get("viivat")
    if not v or v.get("irr_Y") is None or not h.get("viivat_syote"):
        return None
    vanha = kl.Y_MAX_CM
    kl.Y_MAX_CM = y_max_cm
    try:
        ehdokkaat = kl.heiton_ehdokasruudut(h)
    finally:
        kl.Y_MAX_CM = vanha
    if len(ehdokkaat) < min_ruutuja:
        return None
    rivit, nyt = [], None
    for f, X, Y, t in ehdokkaat:
        if nyt is None or f < nyt or f - nyt > 40:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            nyt = f
        else:
            while nyt < f:
                cap.grab()
                nyt += 1
        ok, ruutu = cap.read()
        nyt += 1
        if not ok:
            continue
        try:
            mx, my, A, _ = kl.topdown_kartat(P, X, Y)
        except ValueError:
            continue
        if mx.min() < 0 or my.min() < 0 or mx.max() > ruutu.shape[1] - 1 or my.max() > ruutu.shape[0] - 1:
            continue
        td = cv2.remap(ruutu, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        rivit.append((f, X, Y, t, kl.kasi_pikselit(ruutu, P, X, Y, h["vari"]), kv.leike56(td)))
    alku = next((j for j, r in enumerate(rivit) if r[4] is not None and r[4] <= kl.KASI_ALKU_PX), None)
    if alku is None or len(rivit) - alku < min_ruutuja:
        return None
    rivit = rivit[alku:]
    vk, vkk = h.get("kierto") or {}, h.get("kauko_kierto") or {}
    return dict(tiles=np.array([r[5] for r in rivit], np.uint8), f=np.array([r[0] for r in rivit]), t=np.array([r[3] for r in rivit]), X=np.array([r[1] for r in rivit]), Y=np.array([r[2] for r in rivit]),
                kasi=np.array([r[4] for r in rivit], float), t_irr=float(v["t_irr"]), vari=h["vari"], puoli=h["puoli"],
                vertailu_vain=dict(vanha_omega=vk.get("omega_deg_s"), kauko_omega=vkk.get("omega_deg_s"), kauko_varma=vkk.get("varma")))     # vain raportointiin, ei opetukseen


def komento_leikkeet(a):
    import cv2
    import kierre_leikkeet as kl
    import netti_geom as ng
    kans = dict(k.split("=", 1) for k in a.kansio)
    ulos = {}
    for nimi, video, radat in a.ajo:
        kansio = kans.get(nimi) or os.path.join(KANSIO, kl.KANSIOT[nimi])
        G = ng.Geometria(kansio)
        d = pickle.load(open(radat, "rb"))
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            raise IOError("videota ei voi avata: " + video)
        t0, n = time.time(), 0
        for i, h in enumerate(d):
            try:
                e = leikkeet_heitolle(cap, h, G[h["puoli"]])
            except Exception as virhe:         # yksittäinen huono heitto ei kaada ajoa
                print(f"  {nimi} heitto {i + 1}: {virhe!r}")
                e = None
            if e is not None:
                ulos[(nimi, i + 1)] = e
                n += 1
        print(f"{nimi}: {n} / {len(d)} heittoa, {sum(len(e['f']) for k, e in ulos.items() if k[0] == nimi)} ruutua, {time.time() - t0:.0f} s")
    pickle.dump(ulos, open(a.ulos, "wb"), protocol=4)


# ---------------------------------------------------------------------------------------------------------------------
# IHMISEN MERKINNÄT -> OPETUSRUUDUT (viivat, vakiokiertorata)
# ---------------------------------------------------------------------------------------------------------------------
def g_aika(s, K=K_KIERRE):
    return (1.0 - np.exp(-K * np.asarray(s, float))) / K


def pura(a):
    """Kulmat (jakso 180) jatkuvaksi käyräksi: jokainen valitaan edellisen lähimmäksi."""
    u = [float(a[0])]
    for x in a[1:]:
        u.append(u[-1] + (float(x) - u[-1] + 90.0) % 180.0 - 90.0)
    return np.array(u)


def sovita_rata(t, a, K=K_KIERRE):
    """Vakiokiertorata viivoihin (aika t, kulma a): theta(t) = th_ref + omega g(t - t_ref); palauttaa dict(th_ref, om_ref, t_ref, rms, cov, t_lo, t_hi)."""
    t = np.asarray(t, float)
    u = pura(a)
    tr = t.mean()
    A = np.column_stack([np.ones_like(t), g_aika(t - tr, K)])
    x, *_ = np.linalg.lstsq(A, u, rcond=None)
    r = u - A @ x
    s2 = (r ** 2).sum() / max(len(t) - 2, 1)
    cov = s2 * np.linalg.inv(A.T @ A) if len(t) > 2 else np.full((2, 2), np.nan)
    return dict(th_ref=float(x[0]), om_ref=float(x[1]), t_ref=float(tr), K=K, rms=float(np.sqrt((r ** 2).mean())), cov=cov, t_lo=float(t.min()), t_hi=float(t.max()))


def rata_kulma(m, t):
    return (m["th_ref"] + m["om_ref"] * g_aika(np.asarray(t, float) - m["t_ref"], m["K"])) % 180.0


def rata_sigma(m, t):
    b = g_aika(np.asarray(t, float) - m["t_ref"], m["K"])
    c = m["cov"]
    return np.sqrt(np.maximum(c[0, 0] + 2 * b * c[0, 1] + b * b * c[1, 1], 0.0))


def lue_viivat(L, valinta_json, merkinnat_json, pois=()):
    """Ihmisen viivat leikkeiden heitoille: {(nimi, heitto): dict(viivat=[(ruutu, kulma, paino, id)], ankkuri=(ruudut, kulmat, sigmat) | None, rata=m | None)}. Heitot `pois` jätetään pois (vain testiin)."""
    val = {n["id"]: n for n in json.load(open(valinta_json, encoding="utf-8"))}
    mer = {d["id"]: d for d in json.load(open(merkinnat_json, encoding="utf-8")) if d.get("tila") in ("ok", "epavarma")}
    by = {}
    for id_, d in mer.items():
        n = val[id_]
        if (n["nimi"], n["heitto"]) in pois or (n["nimi"], n["heitto"]) not in L:
            continue
        by.setdefault((n["nimi"], n["heitto"]), []).append((n, d))
    out = {}
    for key, lst in by.items():
        e = L[key]
        viivat = []
        for n, d in sorted(lst, key=lambda x: x[0]["t_s"]):
            j = np.where(e["f"] == n["f"])[0]
            if len(j):
                viivat.append((int(j[0]), float(d["kulma_deg"]), 1.0 if d["tila"] == "ok" else 0.5, n["id"]))
        if not viivat:
            continue
        r = dict(viivat=viivat, ankkuri=None, rata=None)
        if len(viivat) >= 3:
            m = sovita_rata([e["t"][j] for j, *_ in viivat], [a for _, a, _, _ in viivat])
            if m["rms"] <= RAJA_RMS and (m["t_hi"] - m["t_lo"]) >= RAJA_IKKUNA:
                idx = np.where((e["t"] >= m["t_lo"] - MARGINAALI) & (e["t"] <= m["t_hi"] + MARGINAALI))[0]
                r["ankkuri"] = (idx, rata_kulma(m, e["t"][idx]), np.sqrt(SIGMA_VIIVA ** 2 + rata_sigma(m, e["t"][idx]) ** 2))
                r["rata"] = m
        out[key] = r
    return out


def ihmisen_aineisto(L, V, viiva_paino=4.0):
    """Opetusruudut ihmisen merkinnöistä: ankkuroidut ruudut (paino 1) ja muiden heittojen yksittäiset viivat (paino viiva_paino). Palauttaa (X, TH, SIG, W)."""
    X, TH, SIG, W = [], [], [], []
    for key, r in sorted(V.items()):
        if r["ankkuri"] is not None:
            idx, th, sg = r["ankkuri"]
            X.append(L[key]["tiles"][idx]); TH.append(th); SIG.append(sg); W.append(np.ones(len(idx)))
        else:
            for j, a, w, _ in r["viivat"]:
                X.append(L[key]["tiles"][j][None]); TH.append(np.array([a])); SIG.append(np.array([SIGMA_VIIVA])); W.append(np.array([viiva_paino * w]))
    return np.concatenate(X), np.concatenate(TH), np.concatenate(SIG), np.concatenate(W)


# ---------------------------------------------------------------------------------------------------------------------
# VERKOT JA OPETUS (JAX)
# ---------------------------------------------------------------------------------------------------------------------
def _jax():
    try:
        import jax
        import jax.numpy as jnp
        import optax
    except ImportError:
        sys.exit("Opetus vaatii JAX:n ja optaxin: pip install jax optax (CPU riittää)")
    return jax, jnp, optax


def pehmea(th, sigma):
    """Pehmeä tavoite: von Mises (jakso 180) kulmalle th ja hajonnalle sigma (aste), ruutu kerrallaan."""
    th = np.asarray(th, float)[:, None]
    s = np.radians(np.asarray(sigma, float))[:, None]
    kappa = 1.0 / (4.0 * s * s)
    q = np.exp(kappa * (np.cos(np.deg2rad(2.0 * (KUL[None, :] - th))) - 1.0))
    return (q / q.sum(1, keepdims=True)).astype(np.float32)


def _konv_alusta(jax, avain, cin, cout, nolla=False):
    w = jax.random.normal(avain, (3, 3, cin, cout)) * np.sqrt(2.0 / (9 * cin))
    return dict(w=w * 0.0 if nolla else w, b=np.zeros(cout, np.float32))


def _tiheaa_alusta(jax, avain, nin, nout):
    return dict(w=jax.random.normal(avain, (nin, nout)) * np.sqrt(2.0 / nin), b=np.zeros(nout, np.float32))


def alusta(verkko, siemen=0, ch=None, blocks=2, hid=None):
    """Verkon parametrit. 'pieni': ch (16, 32, 64), hid 128; 'resf': ch (24, 48, 96), hid 256, blocks lohkoa vaihetta kohti."""
    jax, jnp, optax = _jax()
    k = iter(jax.random.split(jax.random.PRNGKey(siemen), 200))
    if verkko == "pieni":
        c1, c2, c3 = ch or (16, 32, 64)
        hid = hid or 128
        return dict(c1a=_konv_alusta(jax, next(k), 3, c1), c1b=_konv_alusta(jax, next(k), c1, c1), c2a=_konv_alusta(jax, next(k), c1, c2), c2b=_konv_alusta(jax, next(k), c2, c2),
                    c3a=_konv_alusta(jax, next(k), c2, c3), c3b=_konv_alusta(jax, next(k), c3, c3), d1=_tiheaa_alusta(jax, next(k), 7 * 7 * c3, hid), d2=_tiheaa_alusta(jax, next(k), hid, NB))
    ch = ch or (24, 48, 96)
    hid = hid or 256
    P = dict(stem=_konv_alusta(jax, next(k), 3, ch[0]))
    cin = ch[0]
    for s, c in enumerate(ch):
        P[f"s{s}_down"] = _konv_alusta(jax, next(k), cin, c)
        for b in range(blocks):
            P[f"s{s}_b{b}_c1"] = _konv_alusta(jax, next(k), c, c)
            P[f"s{s}_b{b}_c2"] = _konv_alusta(jax, next(k), c, c, nolla=True)
        cin = c
    P["d1"] = _tiheaa_alusta(jax, next(k), 7 * 7 * ch[-1], hid)
    P["d2"] = _tiheaa_alusta(jax, next(k), hid, NB)
    return P


def eteenpain(verkko, P, x, avain=None, pudotus=0.0, blocks=2):
    """Logitit (B, 60); sama laskenta kuin kulmaverkko.etenemista (NumPy)."""
    jax, jnp, optax = _jax()

    def konv(x, p, askel=1):
        return jax.lax.conv_general_dilated(x, p["w"], (askel, askel), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC")) + p["b"]

    def pool(x):
        return jax.lax.reduce_window(x, -jnp.inf, jax.lax.max, (1, 2, 2, 1), (1, 2, 2, 1), "VALID")
    relu = jax.nn.relu
    if verkko == "pieni":
        for nimi in ("c1a", "c1b", "c2a", "c2b", "c3a", "c3b"):
            x = relu(konv(x, P[nimi]))
            if nimi.endswith("b"):
                x = pool(x)
    else:
        x = relu(konv(x, P["stem"]))
        s = 0
        while f"s{s}_down" in P:
            x = relu(konv(x, P[f"s{s}_down"], 2))
            for b in range(blocks):
                x = relu(x + konv(relu(konv(x, P[f"s{s}_b{b}_c1"])), P[f"s{s}_b{b}_c2"]))
            s += 1
    x = relu(x.reshape(x.shape[0], -1) @ P["d1"]["w"] + P["d1"]["b"])
    if avain is not None and pudotus > 0:
        x = x * jax.random.bernoulli(avain, 1.0 - pudotus, x.shape) / (1.0 - pudotus)
    return x @ P["d2"]["w"] + P["d2"]["b"]


def laajenna(rng, tiles, th, kierto_max=90.0, p_kierto=1.0, p_peilaus=0.5, siirto=3):
    """Opetusruutujen laajennus: peilaus (kulma -> -kulma), kierto a (kulma + a), siirto, kirkkaus / kontrasti / kohina, peitto. Palauttaa (float32 [-0,5 ... 0,5], uudet kulmat)."""
    B, N = len(tiles), tiles.shape[1]
    ulos = np.empty(tiles.shape, np.float32)
    th2 = np.asarray(th, float).copy()
    for i in range(B):
        x = tiles[i]
        if rng.random() < p_peilaus:
            x = kv.peilaa(x)
            th2[i] = (-th2[i]) % 180.0
        if rng.random() < p_kierto:
            a = rng.uniform(-kierto_max, kierto_max)
            x = kv.kierra(x, a)
            th2[i] = (th2[i] + a) % 180.0
        if siirto:
            dx, dy = rng.integers(-siirto, siirto + 1, 2)
            x = np.roll(x, (int(dy), int(dx)), axis=(0, 1))
        ulos[i] = x
    ulos /= 255.0
    ulos *= rng.uniform(0.8, 1.2, (B, 1, 1, 1)).astype(np.float32) * rng.uniform(0.93, 1.07, (B, 1, 1, 3)).astype(np.float32)
    ulos += rng.uniform(-0.1, 0.1, (B, 1, 1, 1)).astype(np.float32)
    ulos += rng.normal(0, 0.02, ulos.shape).astype(np.float32)
    for i in np.where(rng.random(B) < 0.2)[0]:
        h, w = rng.integers(6, 15, 2)
        y0, x0 = rng.integers(0, N - h), rng.integers(0, N - w)
        ulos[i, y0:y0 + h, x0:x0 + w] = rng.uniform(0.2, 0.9)
    return np.clip(ulos, 0, 1) - 0.5, th2 % 180.0


def opeta(X, TH, SIG, W, verkko="pieni", askelta=4000, bs=48, lr=1e-3, siemen=0, kw=None, pudotus=0.3, wd=1e-4, ema=0.995, kierto_max=90.0, p_kierto=1.0, p_peilaus=0.5, loki=print, loki_vali=500):
    """Opetus: X (N, 56, 56, 3) uint8, TH kulmat, SIG tavoitteen hajonta (aste), W otantapainot. Palauttaa EMA-parametrit (numpy)."""
    jax, jnp, optax = _jax()
    kw = dict(kw or {})
    blocks = kw.get("blocks", 2)
    P = alusta(verkko, siemen, **kw)
    rng = np.random.default_rng(siemen)
    p_otanta = np.asarray(W, float) / np.sum(W)
    opt = optax.adamw(optax.warmup_cosine_decay_schedule(0.0, lr, min(50, max(askelta // 10, 1)), askelta, lr * 0.02), weight_decay=wd)
    tila = opt.init(P)

    def tappio(P, x, q, avain):
        return -(q * jax.nn.log_softmax(eteenpain(verkko, P, x, avain, pudotus, blocks))).sum(-1).mean()

    @jax.jit
    def askel(P, tila, x, q, avain):
        l, g = jax.value_and_grad(tappio)(P, x, q, avain)
        up, tila = opt.update(g, tila, P)
        return optax.apply_updates(P, up), tila, l
    E = jax.tree_util.tree_map(lambda a: a, P)
    ema_paivita = jax.jit(lambda E, P, d: jax.tree_util.tree_map(lambda e, p: d * e + (1 - d) * p, E, P))
    avain = jax.random.PRNGKey(siemen + 1)
    t0, summa, n_summa = time.time(), 0.0, 0
    for s in range(askelta):
        idx = rng.choice(len(X), bs, p=p_otanta)
        xb, thb = laajenna(rng, X[idx], np.asarray(TH)[idx], kierto_max, p_kierto, p_peilaus)
        avain, a2 = jax.random.split(avain)
        P, tila, l = askel(P, tila, jnp.asarray(xb), jnp.asarray(pehmea(thb, np.asarray(SIG)[idx])), a2)
        E = ema_paivita(E, P, min(ema, (1 + s) / (10 + s)))
        summa += float(l)
        n_summa += 1
        if loki and ((s + 1) % loki_vali == 0 or s + 1 == askelta):
            loki(f"  askel {s + 1}/{askelta} tappio {summa / n_summa:.3f}  {time.time() - t0:.0f} s")
            summa, n_summa = 0.0, 0
    return jax.tree_util.tree_map(lambda a: np.asarray(a, np.float32), E)


def asetukset(a):
    """Komentorivin asetukset tallennettavaan muotoon (ilman argparsen alikomentofunktiota `f`, jota ei voi purkaa muualla kuin tästä tiedostosta)."""
    return {k: v for k, v in vars(a).items() if k != "f"}


def tallenna_verkko(polku, verkko, P, kw, cfg=None):
    pickle.dump(dict(E=P, verkko=verkko, kw=dict(kw or {}), cfg=cfg or {}), open(polku, "wb"))


def npz_muotoon(d):
    """Opetettu verkko (pickle-sanakirja) -> (arch, {'kerros/w': ..., 'kerros/b': ...}, blocks) kulmaverkko.tallenna-muotoon."""
    P = {}
    for k, v in d["E"].items():
        if "w" in v:
            P[f"{k}/w"] = np.asarray(v["w"], np.float32)
            P[f"{k}/b"] = np.asarray(v["b"], np.float32)
    return ("pieni" if d["verkko"] == "pieni" else "resf"), P, int(d["kw"].get("blocks", 2))


def jax_ennustaja(d):
    """Opetettu verkko (pickle-sanakirja) -> ennustaja f(leikkeet, tta=True) -> jakauma (n, 60); sama TTA kuin kulmaverkko.ennusta, mutta JAX:lla (nopea joukkoennustus opetuksen aikana)."""
    jax, jnp, optax = _jax()
    verkko, P, blocks = d["verkko"], d["E"], d["kw"].get("blocks", 2)
    f = jax.jit(lambda P, x: jax.nn.softmax(eteenpain(verkko, P, x, blocks=blocks)))

    def ennusta(leikkeet, tta=True, kierrot=kv.TTA_KIERROT, peilaus=True, erakoko=256):
        nakymat = []
        for peil in ((False, True) if (tta and peilaus) else (False,)):
            for a in (kierrot if tta else (0,)):
                xv = np.empty(leikkeet.shape, np.float32)
                for i in range(len(leikkeet)):
                    x = leikkeet[i]
                    if peil:
                        x = kv.peilaa(x)
                    if a:
                        x = kv.kierra(x, a)
                    xv[i] = x.astype(np.float32) / 255.0 - 0.5
                p = np.concatenate([np.asarray(f(P, jnp.asarray(xv[j:j + erakoko]))) for j in range(0, len(xv), erakoko)])
                p = np.roll(p, -int(round(a / 3.0)), axis=1)
                if peil:
                    p = p[:, (-np.arange(NB)) % NB]
                nakymat.append(p)
        return kv._log_keskiarvo(nakymat)
    return ennusta


def lataa_ennustajat(polut):
    return [jax_ennustaja(pickle.load(open(p, "rb"))) for p in polut]


def yhdistelma(ennustajat, leikkeet, tta=True):
    """Verkkojen jakaumien geometrinen keskiarvo."""
    return kv._log_keskiarvo([e(leikkeet, tta) for e in ennustajat])


# ---------------------------------------------------------------------------------------------------------------------
# PSEUDOMERKINNÄT: ennusteet -> vakiokiertosovitus
# ---------------------------------------------------------------------------------------------------------------------
def pseudomerkinnat(L, ennustajat, avaimet, lam=0.5, loki=print):
    """Heiton ruutujen ennusteet (TTA, verkkojen geometrinen keskiarvo) sovitetaan vakiokiertoon; palauttaa {avain: dict(kulma, R, residuaali, omega, omega_sd, suunta_p)}."""
    out = {}
    t0 = time.time()
    for i, k in enumerate(avaimet):
        p = yhdistelma(ennustajat, L[k]["tiles"])
        r = kv.sovita_heitto(p, L[k]["t"], lam)
        r["p"] = p
        out[k] = r
        if (i + 1) % 100 == 0:
            loki(f"  pseudomerkinnät {i + 1} / {len(avaimet)}  {time.time() - t0:.0f} s")
    return out


def pseudo_aineisto(L, PS, min_suunta_p=0.9, max_resid_heitto=15.0, max_resid_ruutu=25.0, sigma0=4.5, sigma1=15.0, min_ruutuja=8):
    """Pseudomerkityt ruudut opetukseen: heitto hyväksytään, kun kiertosuunta on varma (suunta_p), ruutujen jakaumat ovat johdonmukaisia sovituksen kanssa (keskiresiduaali) ja ruutu on sovituksen lähellä;
    tavoitteen hajonta sigma0 + sigma1 (1 - R), paino R^2. Palauttaa (X, TH, SIG, W, heittoja)."""
    X, TH, SIG, W, n = [], [], [], [], 0
    for k, r in PS.items():
        if len(L[k]["f"]) < min_ruutuja or r["suunta_p"] < min_suunta_p or r["residuaali"].mean() > max_resid_heitto:
            continue
        ok = r["residuaali"] <= max_resid_ruutu
        if ok.sum() < min_ruutuja:
            continue
        X.append(L[k]["tiles"][ok]); TH.append(r["kulma"][ok]); SIG.append(np.sqrt(sigma0 ** 2 + (sigma1 * (1 - r["R"][ok])) ** 2)); W.append(r["R"][ok] ** 2)
        n += 1
    return np.concatenate(X), np.concatenate(TH), np.concatenate(SIG), np.concatenate(W), n


# ---------------------------------------------------------------------------------------------------------------------
# ARVIOINTI
# ---------------------------------------------------------------------------------------------------------------------
def tilasto(e):
    e = np.abs(np.asarray(e, float))
    return f"n={len(e):3d} keski {e.mean():5.1f} med {np.median(e):5.1f} p90 {np.percentile(e, 90):5.1f} max {e.max():5.1f}  <=5: {np.mean(e <= 5) * 100:3.0f}%  <=10: {np.mean(e <= 10) * 100:3.0f}%  >30: {int((e > 30).sum())}"


def arvioi(L, V, ennustajat, heitot, lam=0.5, loki=print):
    """Ihmisen viivat heitoilla `heitot`: yksi ruutu + TTA ja verkko + vakiokierto (koko heiton käsi-vapaat ruudut)."""
    e1, e2 = [], []
    for k in heitot:
        if k not in V or k not in L:
            loki(f"  {k}: ei viivoja tai leikkeitä")
            continue
        p = yhdistelma(ennustajat, L[k]["tiles"])
        a1, _ = kv.kulma_jakaumasta(p)
        r = kv.sovita_heitto(p, L[k]["t"], lam)
        for j, a, _, _ in V[k]["viivat"]:
            e1.append(float(kv.ero180(a1[j], a)))
            e2.append(float(kv.ero180(r["kulma"][j], a)))
        loki(f"  {k[0][:6]} h{k[1]:<4d} ruutuja {len(L[k]['f']):3d}  omega {r['omega']:+7.1f} (+-{r['omega_sd']:.1f})  suunta-p {r['suunta_p']:.3f}")
    if e1:
        loki("yksi ruutu + TTA     : " + tilasto(e1))
        loki("verkko + vakiokierto : " + tilasto(e2))
    return e1, e2


def jaa_heitot(teksti):
    return {(x.split(":")[0], int(x.split(":")[1])) for x in teksti.split(",") if x} if teksti else set()


def komento_opettaja(a):
    L = pickle.load(open(a.leikkeet, "rb"))
    pois = jaa_heitot(a.pois)
    V = lue_viivat(L, a.valinta, a.merkinnat, pois)
    X, TH, SIG, W = ihmisen_aineisto(L, V, a.viivapaino)
    kw = json.loads(a.kw) if a.kw else {}
    print(f"opettaja: {a.verkko} {kw}, ihmisen ruutuja {len(X)} ({sum(1 for r in V.values() if r['ankkuri'] is not None)} ankkuroitua heittoa, {sum(1 for r in V.values() if r['ankkuri'] is None)} heittoa viivoina), {a.askelta} askelta")
    P = opeta(X, TH, SIG, W, a.verkko, a.askelta, a.bs, a.lr, a.siemen, kw, kierto_max=a.kierto_max)
    tallenna_verkko(a.ulos, a.verkko, P, kw, asetukset(a))


def komento_pseudo(a):
    L = pickle.load(open(a.leikkeet, "rb"))
    pois = jaa_heitot(a.pois)
    merkityt = set(lue_viivat(L, a.valinta, a.merkinnat).keys())
    avaimet = [k for k in sorted(L) if k not in merkityt and k not in pois]
    if a.enintaan:
        avaimet = avaimet[:a.enintaan]
    ennustajat = lataa_ennustajat(a.verkot)
    print(f"pseudo: {len(avaimet)} merkitsemätöntä heittoa, {len(ennustajat)} verkkoa")
    PS = pseudomerkinnat(L, ennustajat, avaimet, a.lam)
    sp = np.array([r["suunta_p"] for r in PS.values()])
    rm = np.array([r["residuaali"].mean() for r in PS.values()])
    print(f"suunta varma (p >= 0,9) {(sp >= 0.9).sum()} / {len(PS)}, p >= 0,99 {(sp >= 0.99).sum()}; heiton keskiresiduaali p50 / p90 {np.percentile(rm, 50):.1f} / {np.percentile(rm, 90):.1f} astetta")
    ag = tot = 0
    for k, r in PS.items():                      # vain raportointi: kaukopään (varma) kiertosuunta ei ole opetuksessa
        v = L[k].get("vertailu_vain", {})
        if v.get("kauko_omega") is not None and v.get("kauko_varma") and r["suunta_p"] >= 0.99:
            tot += 1
            ag += int(np.sign(v["kauko_omega"]) == np.sign(r["omega"]))
    print(f"kiertosuunta sama kuin kaukopään (varma): {ag} / {tot}  [vain raportointi]")
    pickle.dump(PS, open(a.ulos, "wb"), protocol=4)


def komento_oppilas(a):
    L = pickle.load(open(a.leikkeet, "rb"))
    pois = jaa_heitot(a.pois)
    V = lue_viivat(L, a.valinta, a.merkinnat, pois)
    Xh, THh, SIGh, Wh = ihmisen_aineisto(L, V, a.viivapaino)
    PS = pickle.load(open(a.pseudo, "rb"))
    Xp, THp, SIGp, Wp, n = pseudo_aineisto(L, PS, a.min_suunta_p, a.max_resid_heitto, a.max_resid_ruutu)
    X = np.concatenate([Xh, Xp])
    TH = np.concatenate([THh, THp])
    SIG = np.concatenate([SIGh, SIGp])
    W = np.concatenate([a.ihmis_osuus * Wh / Wh.sum(), (1 - a.ihmis_osuus) * Wp / Wp.sum()])
    kw = json.loads(a.kw) if a.kw else {}
    print(f"oppilas: {a.verkko} {kw}, ihmisen ruutuja {len(Xh)}, pseudoruutuja {len(Xp)} ({n} heittoa), {a.askelta} askelta")
    P = opeta(X, TH, SIG, W, a.verkko, a.askelta, a.bs, a.lr, a.siemen, kw, kierto_max=a.kierto_max, p_kierto=a.p_kierto)
    tallenna_verkko(a.ulos, a.verkko, P, kw, asetukset(a))


def komento_arvioi(a):
    L = pickle.load(open(a.leikkeet, "rb"))
    V = lue_viivat(L, a.valinta, a.merkinnat)
    heitot = sorted(jaa_heitot(a.pois)) if a.pois else sorted(V)
    arvioi(L, V, lataa_ennustajat(a.verkot), heitot, a.lam)


def komento_ihmisruudut(a):
    L = pickle.load(open(a.leikkeet, "rb"))
    V = lue_viivat(L, a.valinta, a.merkinnat, jaa_heitot(a.pois))
    rivit = [(L[k]["tiles"][j], ang, id_, f"{k[0]}:{k[1]}") for k, r in sorted(V.items()) for j, ang, _, id_ in r["viivat"]]
    np.savez_compressed(a.ulos, tiles=np.array([r[0] for r in rivit], np.uint8), kulma=np.array([r[1] for r in rivit], np.float32), id=np.array([r[2] for r in rivit]), heitto=np.array([r[3] for r in rivit]))
    print(f"{a.ulos}: {len(rivit)} ihmisen viivaa {len(V)} heitolta, {os.path.getsize(a.ulos) / 1e3:.0f} kt")


def komento_vie(a):
    arch, P, blocks = npz_muotoon(pickle.load(open(a.verkko, "rb")))
    kv.tallenna(a.ulos, arch, P, blocks)
    print(f"{a.ulos}: {arch}, {sum(v.size for v in P.values())} parametria, {os.path.getsize(a.ulos) / 1e6:.1f} MB")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="komento", required=True)
    mer = os.path.join(KANSIO, "kierre_validointi")

    def yhteiset(p, opetus=False, bs=48):
        p.add_argument("--leikkeet", required=True, help="leikkeet-alikomennon pickle")
        p.add_argument("--valinta", default=os.path.join(mer, "valinta.json"))
        p.add_argument("--merkinnat", default=os.path.join(mer, "merkinnat.json"))
        p.add_argument("--pois", default="", help="NIMI:HEITTO,... pois jätettävät heitot (vain testiin): ei opetuksessa eikä merkitsemättöminä")
        if opetus:
            p.add_argument("--verkko", default="pieni", choices=("pieni", "resf"))
            p.add_argument("--kw", default="", help='verkon koko json:na, esim. \'{"ch": [24, 48, 96], "blocks": 2}\'')
            p.add_argument("--askelta", type=int, default=4000)
            p.add_argument("--bs", type=int, default=bs, help="erän koko")
            p.add_argument("--lr", type=float, default=1e-3)
            p.add_argument("--siemen", type=int, default=0)
            p.add_argument("--kierto-max", dest="kierto_max", type=float, default=90.0, help="kuvan kierron enimmäiskulma opetuksessa (aste)")
            p.add_argument("--viivapaino", type=float, default=4.0, help="yksittäisten viivojen otantapaino")
            p.add_argument("--ulos", required=True)
    p = sp.add_parser("leikkeet")
    p.add_argument("--ajo", nargs=3, action="append", metavar=("NIMI", "VIDEO", "RADAT"), required=True, help="video (Video1 / Video2 / Saarelainen / Matarmaa / Ahrenberg), videotiedosto ja netti_live.py --radat -pickle")
    p.add_argument("--kansio", action="append", default=[], metavar="NIMI=POLKU", help="kalibrointikansio videolle (oletus kierre_leikkeet.KANSIOT)")
    p.add_argument("--ulos", required=True)
    p.set_defaults(f=komento_leikkeet)
    p = sp.add_parser("opettaja")
    yhteiset(p, True)
    p.set_defaults(f=komento_opettaja)
    p = sp.add_parser("pseudo")
    yhteiset(p)
    p.add_argument("--verkot", nargs="+", required=True)
    p.add_argument("--lam", type=float, default=0.5, help="ruutujen jakaumien lämpötila vakiokiertosovituksessa")
    p.add_argument("--enintaan", type=int, default=0, help="vain näin monta ensimmäistä heittoa (testaukseen)")
    p.add_argument("--ulos", required=True)
    p.set_defaults(f=komento_pseudo)
    p = sp.add_parser("oppilas")
    yhteiset(p, True, bs=64)
    p.add_argument("--pseudo", required=True)
    p.add_argument("--ihmis-osuus", dest="ihmis_osuus", type=float, default=0.3, help="ihmisen ruutujen osuus opetusotannasta")
    p.add_argument("--min-suunta-p", dest="min_suunta_p", type=float, default=0.9)
    p.add_argument("--max-resid-heitto", dest="max_resid_heitto", type=float, default=15.0)
    p.add_argument("--max-resid-ruutu", dest="max_resid_ruutu", type=float, default=25.0)
    p.add_argument("--p-kierto", dest="p_kierto", type=float, default=1.0, help="kuvan kierron todennäköisyys opetuksessa (1 = aina; 0,3 + --kierto-max 30 on lievä vaihtoehto)")
    p.set_defaults(f=komento_oppilas)
    p = sp.add_parser("arvioi")
    yhteiset(p)
    p.add_argument("--verkot", nargs="+", required=True)
    p.add_argument("--lam", type=float, default=0.5)
    p.set_defaults(f=komento_arvioi)
    p = sp.add_parser("ihmisruudut")
    yhteiset(p)
    p.add_argument("--ulos", required=True)
    p.set_defaults(f=komento_ihmisruudut)
    p = sp.add_parser("vie")
    p.add_argument("--verkko", required=True)
    p.add_argument("--ulos", required=True)
    p.set_defaults(f=komento_vie)
    a = ap.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
