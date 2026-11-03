"""Kahvan kulman neuroverkon opetus: ihmisen merkitsemät (epätäydelliset) kulmat + kierron tarkka lineaarinen hidastuminen (ei vanhan koodin tunnistamia kulmia).

Vaiheet (jokainen oma alikomentonsa; tulokset pickle-tiedostoina):

  leikkeet   Heittojen käsi-vapaat ruudut top-down-leikkeinä (56 x 56 x 3) videolta: irroituksen jälkeen kuvan reunaan, ensimmäisestä ruudusta alkaen, jossa käsi ei enää ole kahvalla (kuten merkintätyökalussa).
             python tools/kulmaverkko_opeta.py leikkeet --ajo Matarmaa Matarmaa.mp4 r_Matarmaa/radat.pkl --ajo ... --ulos leikkeet.pkl
  jalosta    Ihmisen viivat epätäydellisinä + tarkka kiertorata: kierto hidastuu lineaarisesti (omega(s) = omega_0 (1 - K s), kulma theta = th0 + omega_0 (s - K s^2 / 2)), joten heiton kaikki ruudut ovat kahden luvun (th0, omega_0)
             funktio. Radan sovitus viivoihin on robusti (yksittäinen väärä viiva ei kumoa muita) ja, jos --verkot annetaan, myös heitolle vieraiden verkkojen ruutujakaumiin; tulos on jalostettu kulma ruudulle (posteriorin keskiarvo ja hajonta)
             koko heitolle sekä lista viivoista, joita rata ei hyväksy (`--ulos` pickle: opettajan ja oppilaan `--jalostetut`). Rata on huonosti määrätty, kun heitolla on 1-2 viivaa tai viivat ovat keskenään ristiriitaisia (koe: väärä rata
             pilasi opetuksen), joten `--varovainen` jalostaa vain johdonmukaiset heitot (3+ viivaa, jäännös <= 3 astetta, aikaikkuna >= 0,39 s) ja antaa muiden viivat sellaisenaan (`--raaka`: kaikkien viivat sellaisenaan).
             python tools/kulmaverkko_opeta.py jalosta --leikkeet leikkeet.pkl --varovainen [--verkot vieras1.pkl vieras2.pkl] --ulos jalostetut.pkl
  opettaja   Ensimmäinen verkko pelkillä ihmisen kulmilla: oletuksena viivat sovitetaan heittokohtaiseen kiertorataan (ankkuroidut ruudut: 3+ viivaa, jäännös <= 3 astetta, rata antaa kulman viivojen aikaikkunan ± 0,2 s ruuduille;
             muiden heittojen yksittäiset viivat sellaisinaan) tai `--jalostetut` antaa jalostetut kulmat (suositus). Opetuksessa kuvan kierto (kulma + kierto) ja peilaus (kulma -> -kulma) tuottavat kaikki kulmat.
             python tools/kulmaverkko_opeta.py opettaja --leikkeet leikkeet.pkl --ulos opettaja.pkl [--verkko pieni] [--askelta 4000]
  pseudo     Opettajan (tai verkkojen yhdistelmän) ennusteet merkitsemättömille heitoille (TTA) sovitetaan vakiokiertoon (kulmaverkko.sovita_heitto); sovitettu rata on heiton pseudomerkintä.
             python tools/kulmaverkko_opeta.py pseudo --leikkeet leikkeet.pkl --verkot opettaja.pkl --ulos pseudo.pkl
  oppilas    Uusi verkko ihmisen ruuduilla ja pseudomerkinnöillä (itseopetus; kierroksen voi toistaa: oppilas -> pseudo -> oppilas; `--jatka` jatkaa opetusta aiemmista painoista).
             python tools/kulmaverkko_opeta.py oppilas --leikkeet leikkeet.pkl --pseudo pseudo.pkl --ulos oppilas.pkl [--verkko resf --kw '{"ch": [24, 48, 96], "blocks": 2}' --askelta 6000 --p-kierto 0.3 --kierto-max 30]
  arvioi     Verkkojen virhe ihmisen viivoihin (yksi ruutu + TTA; verkko + vakiokierto) pois jätetyillä heitoilla (--pois), jotka eivät ole missään opetuksessa.
             python tools/kulmaverkko_opeta.py arvioi --leikkeet leikkeet.pkl --verkot oppilas.pkl --pois Matarmaa:69,Saarelainen:68
  (y-vyöhykkeet) `jalosta`, `pseudo`, `arvioi` ja `johdonmukaisuus` jättävät oletuksena kiven y-välien 520-540 cm (jäähän maalattu mainosteksti kahvan takana) ja 682,8-702,6 cm (sininen hog-viiva kahvan takana) ruutujen jakaumat pois radan sovituksesta (`--vyohyke-mukana` ottaa ne mukaan): verkon arvio vetäytyy niissä kohti vaakasuuntaa tai menee pieleen; kamera kuvaa vinosta ja kahva on kiven yläreunassa, joten hog-viivan vyöhyke on eri kohdassa kuin itse hog-viiva.
  johdonmukaisuus  Verkkojen johdonmukaisuus ilman merkintöjä: ruutukohtainen ero tarkkaan rataan heitoilla, joilla ei ole viivoja (pidätetyt heitot: verkko ei saa olla opetettu niillä); kuvaa satunnaisvirhettä kaikilla kivillä, ei viivojen kohinan rajoittama.
             python tools/kulmaverkko_opeta.py johdonmukaisuus --leikkeet leikkeet.pkl --verkot oppilas.pkl [--heitot pidatetyt.json] [--pois ...]
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
SIGMA_VIIVA = 2.5                  # ankkuroinnin (ei-jalostetun) ihmisen viivan kohina (aste): suoran sovituksen jäännös 0,1-2,6 astetta
SIGMA_JAL0 = 3.0                   # jalostetun kulman perusepävarmuus (aste), yhdistetään radan posteriorin hajontaan
MAX_SIGMA_JAL = 7.0                # ruutu otetaan jalostettuun aineistoon, kun kulman posteriorin huipun hajonta (katkaistu +- 12 asteeseen) ruudulla on enintään tämä (aste)
MIN_MASSA_JAL = 0.8                # ... ja huippu (+- 12 astetta) sisältää vähintään tämän osan posteriorin massasta (muuten kulma on moniselitteinen tai ihmisen viiva on todennäköisesti pielessä)
RAAKA_PAINO = 3.0                  # raa'an viivan otantapaino (yksi viiva edustaa koko heittoa; jalostettu ruutu painaa 1)
LAM_JAL = 0.05                    # jalostuksessa vieraiden verkkojen ruutujakaumien paino (pseudomerkinnöissä 0,5): verkon jakaumat ovat ylivarmoja suhteessa virheeseensä (ks. README), viivat saavat enemmän painoa
ENNAKKO_JAL = (76.0, 0.6)          # jalostuksen |omega|-ennakko (mediaani aste / s, log-hajonta): 506 varman suunnan heiton omegat ovat mediaaniltaan 76 aste/s (log-hajonta 0,42); tiukempi ennakko poistaa viivojen aliasoitumisen
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
    return kv.rata_aika(s, K)


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
# JALOSTUS: epätäydelliset viivat + tarkka kiertorata (+ vieraiden verkkojen todiste) -> kulma koko heiton ruuduille
# ---------------------------------------------------------------------------------------------------------------------
def jalosta(L, V, ennustajat=None, lam=LAM_JAL, sigma_viiva=kv.SIGMA_VIIVA, eps=kv.EPS_VIIVA, sigma0=SIGMA_JAL0, max_sigma=MAX_SIGMA_JAL, vyohyke=True):
    """Jokaiselle merkitylle heitolle (V = lue_viivat) sovitetaan rata (th0, omega) ihmisen viivoihin epätäydellisinä (`kulmaverkko.viiva_rangaistus`) ja, jos `ennustajat` on annettu, myös niiden ruutujakaumiin
    (verkkojen pitää olla heitolle vieraita, muuten verkko vain toistaa viivat). Palauttaa {avain: dict(idx, kulma, sigma, viivat: [(id, ihmisen kulma, ero jalostettuun)], omega, omega_sd, suunta_p)}: idx = ruudut, joissa
    posteriorin huipun hajonta on enintään max_sigma ja huippu (+- 12 astetta) sisältää vähintään MIN_MASSA_JAL massasta, kulma = huipun keskiarvo ja sigma = sqrt(sigma0^2 + huipun hajonta^2) (viivojen mahdolliset virheet eivät laajenna tavoitetta,
    mutta ne voivat siirtää huippua tai tehdä ruudusta moniselitteisen)."""
    ulos = {}
    for k, r in sorted(V.items()):
        e = L[k]
        idx_v = np.array([j for j, *_ in r["viivat"]], int)
        a_v = np.array([a for _, a, _, _ in r["viivat"]], float)
        p = yhdistelma(ennustajat, e["tiles"]) if ennustajat else None
        ohita = kv.vyohykkeessa(e["Y"]) if (vyohyke and p is not None) else None          # hog-viivan vyöhykkeen ruutujen verkkotodistetta ei käytetä (viivat ovat ihmisen merkintöjä: ne pysyvät)
        f = kv.sovita_heitto(p, e["t"], lam, viivat=dict(idx=idx_v, kulma=a_v, sigma=sigma_viiva, eps=eps), ennakko=ENNAKKO_JAL, ohita=ohita)
        ok = np.where((f["sigma_moodi"] <= max_sigma) & (f["massa_moodi"] >= MIN_MASSA_JAL))[0]
        ulos[k] = dict(idx=ok, kulma=f["kulma_moodi"][ok], sigma=np.sqrt(sigma0 ** 2 + f["sigma_moodi"][ok] ** 2), omega=f["omega"], omega_sd=f["omega_sd"], suunta_p=f["suunta_p"],
                       viivat=[(id_, float(a), float(kv.ero180(f["kulma_moodi"][j], a))) for j, a, _, id_ in r["viivat"]])
    return ulos


def jalostettu_aineisto(L, JAL):
    """Jalostetut ruudut opetusaineistoksi: (X, TH, SIG, W); paino 1 / ruutu tai heiton oma `paino` (raa'at viivat: yksi viiva edustaa koko heittoa)."""
    X, TH, SIG, W = [], [], [], []
    for k, j in sorted(JAL.items()):
        if len(j["idx"]):
            X.append(L[k]["tiles"][j["idx"]]); TH.append(j["kulma"]); SIG.append(j["sigma"]); W.append(np.full(len(j["idx"]), float(j.get("paino", 1.0))))
    return np.concatenate(X), np.concatenate(TH), np.concatenate(SIG), np.concatenate(W)


def lataa_jalostetut(polut, pois=()):
    """Jalostetut kulmat pickle-tiedostoista (myöhemmät tiedostot ohittavat aiemmat samalle heitolle); pois jätettävät heitot ohitetaan."""
    JAL = {}
    for polku in polut:
        JAL.update(pickle.load(open(polku, "rb")))
    return {k: v for k, v in JAL.items() if k not in pois}


def ihmisen_ruudut(a, L, V):
    """Opetuksen ihmisaineisto komentorivin mukaan: jalostetut kulmat (--jalostetut) tai ankkuroidut ruudut ja yksittäiset viivat (`ihmisen_aineisto`)."""
    if a.jalostetut:
        JAL = lataa_jalostetut(a.jalostetut, jaa_heitot(a.pois))
        return jalostettu_aineisto(L, JAL), f"{len(JAL)} heittoa jalostettuina"
    return ihmisen_aineisto(L, V, a.viivapaino), f"{sum(1 for r in V.values() if r['ankkuri'] is not None)} ankkuroitua heittoa, {sum(1 for r in V.values() if r['ankkuri'] is None)} heittoa viivoina"


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


def opeta(X, TH, SIG, W, verkko="pieni", askelta=4000, bs=48, lr=1e-3, siemen=0, kw=None, pudotus=0.3, wd=1e-4, ema=0.995, kierto_max=90.0, p_kierto=1.0, p_peilaus=0.5, loki=print, loki_vali=500, P0=None):
    """Opetus: X (N, 56, 56, 3) uint8, TH kulmat, SIG tavoitteen hajonta (aste), W otantapainot. P0 = aiemmat painot (jatkoopetus), muuten satunnaisalustus. Palauttaa EMA-parametrit (numpy)."""
    jax, jnp, optax = _jax()
    kw = dict(kw or {})
    blocks = kw.get("blocks", 2)
    P = alusta(verkko, siemen, **kw) if P0 is None else jax.tree_util.tree_map(jnp.asarray, P0)
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


def tta_kierrot_asetuksista(a):
    """Verkon oletus-TTA-kierrot opetuksen kuvan kierrosta: `--tta-kierrot` (esim. "0" tai "-9,0,9") tai automaattisesti: ei kiertoa opetuksessa (`--p-kierto 0`) -> vain peilaus (kierretty näkymä on tällaiselle verkolle vieras:
    liikesumennus on todellisuudessa aina radan suuntainen); pieni kierto (`--kierto-max` < 20) -> (-a, 0, a) opetuksen kiertoalueella (a kolmen monikerta); muuten kv.TTA_KIERROT."""
    s = getattr(a, "tta_kierrot", "auto")
    if s and s != "auto":
        return tuple(float(x) for x in s.split(","))
    p_k, k_max = getattr(a, "p_kierto", 1.0), getattr(a, "kierto_max", 90.0)
    if p_k <= 0.0:
        return (0.0,)
    if k_max < 20.0:
        v = 3.0 * int(k_max // 3)
        return (-v, 0.0, v) if v > 0 else (0.0,)
    return tuple(float(x) for x in kv.TTA_KIERROT)


def asetukset(a):
    """Komentorivin asetukset tallennettavaan muotoon (ilman argparsen alikomentofunktiota `f`, jota ei voi purkaa muualla kuin tästä tiedostosta) sekä verkon oletus-TTA-kierrot (`tta_kierrot`)."""
    d = {k: v for k, v in vars(a).items() if k != "f"}
    d["tta_kierrot"] = tta_kierrot_asetuksista(a)
    return d


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
    oletus_kierrot = tuple(d.get("cfg", {}).get("tta_kierrot") or kv.TTA_KIERROT)          # opetuksessa valittu TTA-kierto (vanhoissa tiedostoissa ei ole: kv.TTA_KIERROT)
    f = jax.jit(lambda P, x: jax.nn.softmax(eteenpain(verkko, P, x, blocks=blocks)))

    def ennusta(leikkeet, tta=True, kierrot=None, peilaus=True, erakoko=256):
        if kierrot is None:
            kierrot = oletus_kierrot
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
def pseudomerkinnat(L, ennustajat, avaimet, lam=0.5, loki=print, vyohyke=True):
    """Heiton ruutujen ennusteet (TTA, verkkojen geometrinen keskiarvo) sovitetaan vakiokiertoon; palauttaa {avain: dict(kulma, R, residuaali, ohitettu, omega, omega_sd, suunta_p)}.
    vyohyke: y-vyöhykkeiden (`kulmaverkko.VYOHYKKEET_Y_CM`: mainosteksti ja hog-viiva kahvan takana) ruutujen jakaumia ei käytetä sovituksessa (verkon arvio vetäytyy niissä kohti vaakasuuntaa); niille tulee silti rata-kulma (`ohitettu` = True)."""
    out = {}
    t0 = time.time()
    for i, k in enumerate(avaimet):
        p = yhdistelma(ennustajat, L[k]["tiles"])
        r = kv.sovita_heitto(p, L[k]["t"], lam, ohita=kv.vyohykkeessa(L[k]["Y"]) if vyohyke else None)
        r["p"] = p
        out[k] = r
        if (i + 1) % 100 == 0:
            loki(f"  pseudomerkinnät {i + 1} / {len(avaimet)}  {time.time() - t0:.0f} s")
    return out


def pseudo_aineisto(L, PS, min_suunta_p=0.9, max_resid_heitto=15.0, max_resid_ruutu=25.0, sigma0=4.5, sigma1=15.0, min_ruutuja=8):
    """Pseudomerkityt ruudut opetukseen: heitto hyväksytään, kun kiertosuunta on varma (suunta_p), ruutujen jakaumat ovat johdonmukaisia sovituksen kanssa (keskiresiduaali) ja ruutu on sovituksen lähellä;
    tavoitteen hajonta sigma0 + sigma1 (1 - R), paino R^2. Sovituksessa pois jätetyt (`ohitettu`, hog-viivan vyöhyke) ruudut eivät vaikuta heiton hyväksyntään eivätkä pudota itseään: ne tulevat opetukseen radan
    antamalla kulmalla (rata on laskettu muista ruuduista, joten ruutu oppii "näkemään hog-viivan läpi"). Palauttaa (X, TH, SIG, W, heittoja)."""
    X, TH, SIG, W, n = [], [], [], [], 0
    for k, r in PS.items():
        oh = r.get("ohitettu", np.zeros(len(r["kulma"]), bool))
        res = r["residuaali"]
        keski = res[~oh].mean() if (~oh).any() else res.mean()
        if len(L[k]["f"]) < min_ruutuja or r["suunta_p"] < min_suunta_p or keski > max_resid_heitto:
            continue
        ok = (res <= max_resid_ruutu) | oh
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


def arvioi(L, V, ennustajat, heitot, lam=0.5, loki=print, vyohyke=True):
    """Ihmisen viivat heitoilla `heitot`: yksi ruutu + TTA ja verkko + vakiokierto (koko heiton käsi-vapaat ruudut)."""
    e1, e2 = [], []
    for k in heitot:
        if k not in V or k not in L:
            loki(f"  {k}: ei viivoja tai leikkeitä")
            continue
        p = yhdistelma(ennustajat, L[k]["tiles"])
        a1, _ = kv.kulma_jakaumasta(p)
        r = kv.sovita_heitto(p, L[k]["t"], lam, ohita=kv.vyohykkeessa(L[k]["Y"]) if vyohyke else None)
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
    (X, TH, SIG, W), kuvaus = ihmisen_ruudut(a, L, V)
    kw = json.loads(a.kw) if a.kw else {}
    print(f"opettaja: {a.verkko} {kw}, ihmisen ruutuja {len(X)} ({kuvaus}), {a.askelta} askelta" + (", jatkoa painoista " + a.jatka if a.jatka else ""))
    P = opeta(X, TH, SIG, W, a.verkko, a.askelta, a.bs, a.lr, a.siemen, kw, kierto_max=a.kierto_max, P0=jatko_painot(a, kw))
    tallenna_verkko(a.ulos, a.verkko, P, kw, asetukset(a))


def jatko_painot(a, kw):
    """--jatka: aiemman opetuksen painot (tämän työkalun pickle) jatkoopetuksen alkuarvoiksi; verkon rakenteen pitää täsmätä."""
    if not a.jatka:
        return None
    d = pickle.load(open(a.jatka, "rb"))
    if d["verkko"] != a.verkko or dict(d["kw"]) != dict(kw):
        sys.exit(f"--jatka: painojen rakenne ({d['verkko']} {d['kw']}) ei täsmää ({a.verkko} {kw})")
    return d["E"]


def raa_at_viivat(V, sigma=kv.SIGMA_VIIVA, paino=RAAKA_PAINO):
    """Ihmisen viivat sellaisenaan: vain viivojen ruudut, kulma = viiva, hajonta sigma, paino `paino` (ei rataa; suositus heitoille, joilla on 1-2 viivaa tai epäjohdonmukaiset viivat)."""
    return {k: dict(idx=np.array([j for j, *_ in r["viivat"]], int), kulma=np.array([a for _, a, _, _ in r["viivat"]], float), sigma=np.full(len(r["viivat"]), float(sigma)), paino=paino,
                    omega=float("nan"), omega_sd=float("nan"), suunta_p=float("nan"), viivat=[(id_, float(a), 0.0) for _, a, _, id_ in r["viivat"]]) for k, r in V.items()}


def komento_jalosta(a):
    L = pickle.load(open(a.leikkeet, "rb"))
    V = lue_viivat(L, a.valinta, a.merkinnat, jaa_heitot(a.pois))
    if a.raaka:
        JAL = raa_at_viivat(V)
        print(f"jalosta --raaka: {len(JAL)} heittoa, {sum(len(j['idx']) for j in JAL.values())} viivaa sellaisenaan (hajonta {kv.SIGMA_VIIVA} astetta, paino {RAAKA_PAINO})")
        pickle.dump(JAL, open(a.ulos, "wb"), protocol=4)
        return
    ennustajat = lataa_ennustajat(a.verkot) if a.verkot else None
    muut = {}
    if a.varovainen:                     # rata vain heitoille, joiden viivat ovat johdonmukaiset (ankkurointikriteeri: 3+ viivaa, suoran jäännös <= 3 astetta, aikaikkuna >= 0,39 s); muilla viivat sellaisenaan
        muut = raa_at_viivat({k: r for k, r in V.items() if r["ankkuri"] is None})
        V = {k: r for k, r in V.items() if r["ankkuri"] is not None}
    JAL = jalosta(L, V, ennustajat, a.lam, vyohyke=not a.vyohyke_mukana)
    print(f"jalosta: {len(JAL)} heittoa, " + (f"viivat + {len(a.verkot)} vierasta verkkoa" if a.verkot else "vain viivat") + (f"; {len(muut)} heitolle viivat sellaisenaan (ei johdonmukaisia)" if muut else ""))
    epailyttavia = 0
    for k, j in sorted(JAL.items()):
        epaily = [(id_, ihm, ero) for id_, ihm, ero in j["viivat"] if abs(ero) > 2.5 * kv.SIGMA_VIIVA]
        epailyttavia += len(epaily)
        print(f"  {k[0][:6]} h{k[1]:<4d} ruutuja {len(j['idx']):3d} / {len(L[k]['f']):3d}  omega {j['omega']:+7.1f} (+-{j['omega_sd']:.1f})  suunta-p {j['suunta_p']:.3f}  viivojen ero jalostettuun: "
              + " ".join(f"{ero:+.0f}" for _, _, ero in j["viivat"]) + ("   EPÄILYTTÄVÄ: " + ", ".join(f"{id_} (ihminen {ihm:.0f}, rata {ihm + ero:.0f})" for id_, ihm, ero in epaily) if epaily else ""))
    print(f"viivoja, jotka poikkeavat radasta yli {2.5 * kv.SIGMA_VIIVA:.0f} astetta: {epailyttavia}")
    JAL.update(muut)
    pickle.dump(JAL, open(a.ulos, "wb"), protocol=4)


def komento_pseudo(a):
    L = pickle.load(open(a.leikkeet, "rb"))
    pois = jaa_heitot(a.pois)
    merkityt = set(lue_viivat(L, a.valinta, a.merkinnat).keys())
    avaimet = [k for k in sorted(L) if k not in merkityt and k not in pois]
    if a.enintaan:
        avaimet = avaimet[:a.enintaan]
    ennustajat = lataa_ennustajat(a.verkot)
    print(f"pseudo: {len(avaimet)} merkitsemätöntä heittoa, {len(ennustajat)} verkkoa")
    PS = pseudomerkinnat(L, ennustajat, avaimet, a.lam, vyohyke=not a.vyohyke_mukana)
    sp = np.array([r["suunta_p"] for r in PS.values()])
    rm = np.array([r["residuaali"][~r["ohitettu"]].mean() if (~r["ohitettu"]).any() else r["residuaali"].mean() for r in PS.values()])
    if not a.vyohyke_mukana:
        print(f"y-vyöhykkeet {', '.join(f'{x[0]:g}-{x[1]:g}' for x in kv.VYOHYKKEET_Y_CM)} cm (mainosteksti, hog-viiva) jätetty pois sovituksesta: {sum(int(r['ohitettu'].sum()) for r in PS.values())} / {sum(len(r['kulma']) for r in PS.values())} ruutua")
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
    (Xh, THh, SIGh, Wh), kuvaus = ihmisen_ruudut(a, L, V)
    PS = pickle.load(open(a.pseudo, "rb"))
    Xp, THp, SIGp, Wp, n = pseudo_aineisto(L, PS, a.min_suunta_p, a.max_resid_heitto, a.max_resid_ruutu)
    X = np.concatenate([Xh, Xp])
    TH = np.concatenate([THh, THp])
    SIG = np.concatenate([SIGh, SIGp])
    W = np.concatenate([a.ihmis_osuus * Wh / Wh.sum(), (1 - a.ihmis_osuus) * Wp / Wp.sum()])
    kw = json.loads(a.kw) if a.kw else {}
    print(f"oppilas: {a.verkko} {kw}, ihmisen ruutuja {len(Xh)} ({kuvaus}), pseudoruutuja {len(Xp)} ({n} heittoa), {a.askelta} askelta" + (", jatkoa painoista " + a.jatka if a.jatka else ""))
    P = opeta(X, TH, SIG, W, a.verkko, a.askelta, a.bs, a.lr, a.siemen, kw, kierto_max=a.kierto_max, p_kierto=a.p_kierto, P0=jatko_painot(a, kw))
    tallenna_verkko(a.ulos, a.verkko, P, kw, asetukset(a))


def komento_arvioi(a):
    L = pickle.load(open(a.leikkeet, "rb"))
    V = lue_viivat(L, a.valinta, a.merkinnat)
    heitot = sorted(jaa_heitot(a.pois)) if a.pois else sorted(V)
    arvioi(L, V, lataa_ennustajat(a.verkot), heitot, a.lam, vyohyke=not a.vyohyke_mukana)


def komento_johdonmukaisuus(a):
    """Verkkojen johdonmukaisuus ilman merkintöjä (`kulmaverkko.johdonmukaisuus`): ruutukohtainen ero tarkkaan rataan, kulmaluokittain ja suunnan varmuus. Oletuksena kaikki heitot, joilla ei ole ihmisen viivoja;
    `--heitot` (JSON: [[nimi, nro], ...]) rajaa joukon, `--pois` jättää heittoja pois. Kuvaa satunnaisvirhettä, ei yhteistä poikkeamaa; verkon pitää olla opettamatta näillä heitoilla (pidätetyt heitot)."""
    L = pickle.load(open(a.leikkeet, "rb"))
    pois = jaa_heitot(a.pois)
    if a.heitot:
        avaimet = [tuple(x) for x in json.load(open(a.heitot))]
    else:
        merkityt = set(lue_viivat(L, a.valinta, a.merkinnat).keys())
        avaimet = [k for k in sorted(L) if k not in merkityt]
    avaimet = [k for k in avaimet if k in L and k not in pois]
    print(f"johdonmukaisuus: {len(avaimet)} heittoa, {sum(len(L[k]['f']) for k in avaimet)} ruutua")
    for polku in a.verkot:
        ennustajat = lataa_ennustajat(polku.split(","))
        ohita = None if a.vyohyke_mukana else [kv.vyohykkeessa(L[k]["Y"]) for k in avaimet]
        r = kv.johdonmukaisuus([yhdistelma(ennustajat, L[k]["tiles"]) for k in avaimet], [L[k]["t"] for k in avaimet], ohita=ohita)
        e, kul, oh = r["ero"], r["kulma"], r["ohitettu"]
        pysty = (kul >= 75) & (kul < 120)
        print(f"  {polku}: ero rataan keski {e.mean():.2f} mediaani {np.median(e):.2f} p90 {np.percentile(e, 90):.2f}, yli 10 astetta {np.mean(e > 10) * 100:.1f} %; lähes pysty (75-119) {e[pysty].mean():.2f}, muut {e[~pysty].mean():.2f}; "
              f"suunta varma (p >= 0,99) {np.mean(r['suunta_p'] >= 0.99) * 100:.0f} %")
        if oh.any():
            print(f"    y-vyöhykkeiden (mainosteksti, hog-viiva) ruudut ({oh.sum()}, jätetty pois sovituksesta): ero rataan keski {e[oh].mean():.2f}, yli 10 astetta {np.mean(e[oh] > 10) * 100:.1f} %; muut ruudut ({(~oh).sum()}): keski {e[~oh].mean():.2f}, yli 10 astetta {np.mean(e[~oh] > 10) * 100:.1f} %")


def komento_ihmisruudut(a):
    L = pickle.load(open(a.leikkeet, "rb"))
    V = lue_viivat(L, a.valinta, a.merkinnat, jaa_heitot(a.pois))
    rivit = [(L[k]["tiles"][j], ang, id_, f"{k[0]}:{k[1]}") for k, r in sorted(V.items()) for j, ang, _, id_ in r["viivat"]]
    np.savez_compressed(a.ulos, tiles=np.array([r[0] for r in rivit], np.uint8), kulma=np.array([r[1] for r in rivit], np.float32), id=np.array([r[2] for r in rivit]), heitto=np.array([r[3] for r in rivit]))
    print(f"{a.ulos}: {len(rivit)} ihmisen viivaa {len(V)} heitolta, {os.path.getsize(a.ulos) / 1e3:.0f} kt")


def komento_vie(a):
    d = pickle.load(open(a.verkko, "rb"))
    arch, P, blocks = npz_muotoon(d)
    kierrot = d.get("cfg", {}).get("tta_kierrot")
    kv.tallenna(a.ulos, arch, P, blocks, kierrot=kierrot if kierrot is not None and tuple(kierrot) != tuple(kv.TTA_KIERROT) else None)
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
            p.add_argument("--viivapaino", type=float, default=4.0, help="yksittäisten viivojen otantapaino (ilman --jalostetut)")
            p.add_argument("--jalostetut", nargs="*", default=[], metavar="PICKLE", help="jalosta-alikomennon tulokset ihmisen aineistoksi (ankkuroinnin ja yksittäisten viivojen sijaan); myöhempi tiedosto ohittaa aiemman samalle heitolle")
            p.add_argument("--jatka", default="", metavar="PICKLE", help="jatka opetusta aiemman opetuksen painoista (sama verkko; pienempi --lr, esim. 3e-4)")
            p.add_argument("--ulos", required=True)
    p = sp.add_parser("leikkeet")
    p.add_argument("--ajo", nargs=3, action="append", metavar=("NIMI", "VIDEO", "RADAT"), required=True, help="video (Video1 / Video2 / Saarelainen / Matarmaa / Ahrenberg), videotiedosto ja netti_live.py --radat -pickle")
    p.add_argument("--kansio", action="append", default=[], metavar="NIMI=POLKU", help="kalibrointikansio videolle (oletus kierre_leikkeet.KANSIOT)")
    p.add_argument("--ulos", required=True)
    p.set_defaults(f=komento_leikkeet)
    p = sp.add_parser("jalosta")
    yhteiset(p)
    p.add_argument("--verkot", nargs="*", default=[], help="heitolle vieraiden verkkojen pickle-tiedostot (ruutujakaumien todiste); ilman niitä jalostus perustuu vain viivoihin")
    p.add_argument("--lam", type=float, default=LAM_JAL, help="vieraiden verkkojen ruutujakaumien paino (pieni: viivat ovat verkkoa tarkempia)")
    p.add_argument("--raaka", action="store_true", help="ei rataa: vain viivojen ruudut sellaisenaan (heitoille, joilla on 1-2 viivaa tai epäjohdonmukaiset viivat)")
    p.add_argument("--varovainen", action="store_true", help="rata vain johdonmukaisille heitoille (3+ viivaa, jäännös <= 3 astetta; suositus), muille heitoille viivat sellaisenaan")
    p.add_argument("--vyohyke-mukana", dest="vyohyke_mukana", action="store_true", help="ota myös y-vyöhykkeiden (mainosteksti kiven y 520-540 cm ja hog-viiva 682,8-702,6 cm) ruutujen jakaumat mukaan radan sovitukseen (oletus: ne jätetään pois)")
    p.add_argument("--ulos", required=True)
    p.set_defaults(f=komento_jalosta)
    p = sp.add_parser("opettaja")
    yhteiset(p, True)
    p.set_defaults(f=komento_opettaja)
    p = sp.add_parser("pseudo")
    yhteiset(p)
    p.add_argument("--verkot", nargs="+", required=True)
    p.add_argument("--lam", type=float, default=0.5, help="ruutujen jakaumien lämpötila vakiokiertosovituksessa")
    p.add_argument("--enintaan", type=int, default=0, help="vain näin monta ensimmäistä heittoa (testaukseen)")
    p.add_argument("--vyohyke-mukana", dest="vyohyke_mukana", action="store_true", help="ota myös y-vyöhykkeiden (mainosteksti kiven y 520-540 cm ja hog-viiva 682,8-702,6 cm) ruutujen jakaumat mukaan radan sovitukseen (oletus: ne jätetään pois)")
    p.add_argument("--ulos", required=True)
    p.set_defaults(f=komento_pseudo)
    p = sp.add_parser("oppilas")
    yhteiset(p, True, bs=64)
    p.add_argument("--pseudo", required=True)
    p.add_argument("--ihmis-osuus", dest="ihmis_osuus", type=float, default=0.3, help="ihmisen ruutujen osuus opetusotannasta")
    p.add_argument("--min-suunta-p", dest="min_suunta_p", type=float, default=0.9)
    p.add_argument("--max-resid-heitto", dest="max_resid_heitto", type=float, default=15.0)
    p.add_argument("--max-resid-ruutu", dest="max_resid_ruutu", type=float, default=25.0)
    p.add_argument("--p-kierto", dest="p_kierto", type=float, default=1.0, help="kuvan kierron todennäköisyys opetuksessa (1 = aina; 0 = ei lainkaan, suositus oppilaalle: liikesumennus on aina radan suuntainen; 0,3 + --kierto-max 30 on lievä vaihtoehto)")
    p.add_argument("--tta-kierrot", dest="tta_kierrot", default="auto", help="verkon oletus-TTA-kierrot (aste, pilkuin erotettuina, esim. 0 tai -9,0,9) tai auto: opetuksen kuvan kierrosta (ks. tta_kierrot_asetuksista)")
    p.set_defaults(f=komento_oppilas)
    p = sp.add_parser("arvioi")
    yhteiset(p)
    p.add_argument("--verkot", nargs="+", required=True)
    p.add_argument("--lam", type=float, default=0.5)
    p.add_argument("--vyohyke-mukana", dest="vyohyke_mukana", action="store_true", help="ota myös y-vyöhykkeiden (mainosteksti kiven y 520-540 cm ja hog-viiva 682,8-702,6 cm) ruutujen jakaumat mukaan radan sovitukseen (oletus: ne jätetään pois)")
    p.set_defaults(f=komento_arvioi)
    p = sp.add_parser("johdonmukaisuus")
    yhteiset(p)
    p.add_argument("--verkot", nargs="+", required=True, help="verkkojen pickle-tiedostot (yhdistelmä: tiedostot pilkulla erotettuina yhtenä argumenttina)")
    p.add_argument("--heitot", help="JSON-tiedosto: [[nimi, nro], ...] (oletus: kaikki heitot, joilla ei ole ihmisen viivoja)")
    p.add_argument("--vyohyke-mukana", dest="vyohyke_mukana", action="store_true", help="ota myös y-vyöhykkeiden (mainosteksti kiven y 520-540 cm ja hog-viiva 682,8-702,6 cm) ruutujen jakaumat mukaan radan sovitukseen (oletus: ne jätetään pois)")
    p.set_defaults(f=komento_johdonmukaisuus)
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
