"""Kierteen mittauksen (netti_kierre) tarkistus ihmisen merkitsemia kahvan kulmia vastaan.

    python tools/kierre_vertailu.py --valinta kierre_validointi/valinta.json --merkinnat kierre_validointi/merkinnat.json \\
        --ajo Matarmaa Matarmaa.mp4 r20_Matarmaa/radat.pkl --ajo Video1 netti_yt.mp4 r20_Video1/radat.pkl ... [--tulos raportti.json]

Aineisto: tools/luo_kahvamerkitsin.py:n sivu ja sen valinta.json (ruudut, heitot, ajat) sekä merkinnat (tools/kierre_etiketit.py); kierre_validointi/ sisaltaa ensimmaisen aineiston (48 ruutua, 12 heittoa).
Videolta ajetaan heittopaan kierteen mittaus samoilla ruuduilla ja samalla koodilla kuin reaaliaikaisessa ohjelmassa (netti_seuranta._kierto_aloita: leikkeet oikaistusta kuvasta kiven ymparilta, B-paikat ja aika viivaestimaatin
syotteesta; netti_kierre.kulmakayrat ja arvioi_posteriori); tulos toistaa radat.pkl:n tallennetun kierron.

Vertailut:
  * RUUTUKOHTAINEN: koneen paras kulma (kulmakustannuskayran minimi) ihmisen viivaan (kulma maan tasossa, X-akselista Y-akseliin, vastapaivaan; jakso 180 astetta): ero, koneen kustannus ihmisen kulmalla (dC) ja
    kayran lahin paikallinen minimi; luokittain (vari, puhdas / ei-puhdas ruutu, ihmisen kulman alue). Koneen ruudun kohinatason oikeellisuus: pen = dC / (2 s2 (sigma / SIG_HYVA)^2) on kalibroidulla mallilla ~ khii^2 / 2
    (keski 0,5, mediaani 0,23); ihmisen oma kohina (~3-5 astetta) nostaa sita puhtailla ruuduilla.
  * HEITTOKOHTAINEN: omega ihmisen viivoista (kierre_etiketit: 180 asteen purku + suorasovitus) vs koneen omega (kaikki ruudut), koneen omega vain ihmisen ajanjaksolta (vakiokierto) ja kaukopaan mittaus
    vaimennettuna lahipaahan (omega_lahi = omega_kauko exp(K_KIERRE dt)): riippumaton viite, joka ei kayta lahipaan kuvia.
Rajoitus: ihmisen viiva on tulkinta sumeasta kuvasta (kasi, liikesumeus, kahvan jalat; keltaisella kahvalla palkki on lahes jaan varinen), joten kumpikaan ei ole varma totuus: ero ihmisen ja koneen valilla on arvio
mittausvirheesta vain niilta osin, joilla ihminen on oikeassa; kaukopaan vertailu auttaa ratkaisemaan, kumpi on lahempana."""
import argparse
import json
import os
import pickle
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import aikaperusta as aikap  # noqa: E402
import kierre_etiketit as ke  # noqa: E402
import kierre_leikkeet as kl  # noqa: E402
import netti_kierre as nk  # noqa: E402
import netti_seuranta as ns  # noqa: E402

PAIKALLINEN_MARGINAALI_S = 0.05      # koneen paikallinen omega: ihmisen ensimmaisen ja viimeisen ruudun aika +- tama


# ---------------------------------------------------------------------------------------------------------------------
# YHTEISET
# ---------------------------------------------------------------------------------------------------------------------
def ero180(a, b):
    """Kulmaero a - b (aste) valille [-90, 90) (jakso 180)."""
    return (np.asarray(a, float) - np.asarray(b, float) + 90.0) % 180.0 - 90.0


def maa_kulma(th_kuva, A):
    """Topdown-leikkeen kulma (kuvassa vastapaivaan, 0 = oikealle) -> maan tason kulma (X-akselista Y-akseliin; jakso 180). A = (sx, sy): leikkeen oikealle = X:n suunta sx, alas = Y:n suunta sy
    (kierre_leikkeet.suunta_merkit); leike ei ole peilattu, joten A = (-1, 1) antaa saman kulman."""
    sx, sy = A
    return float(np.degrees(np.arctan2(-sy * np.sin(np.radians(th_kuva)), sx * np.cos(np.radians(th_kuva))))) % 180.0


def kayran_minimit(C):
    """Kulmakayran (60 kulmaa, jakso) paikalliset minimit [(kulma, kustannus)] kustannuksen mukaan nousevasti."""
    n = len(C)
    return sorted(((float(nk.KULMAT[i]), float(C[i])) for i in range(n) if C[i] <= C[i - 1] and C[i] < C[(i + 1) % n]), key=lambda x: x[1])


def kustannus_kulmassa(C, kulma):
    """Kulmakayran arvo mielivaltaisessa kulmassa (lineaarinen interpolointi, jakso 180)."""
    x = (float(kulma) % 180.0) / 3.0
    i0 = int(np.floor(x)) % len(C)
    return float(C[i0] * (1.0 - (x - np.floor(x))) + C[(i0 + 1) % len(C)] * (x - np.floor(x)))


# ---------------------------------------------------------------------------------------------------------------------
# KONEEN MITTAUS VIDEOLTA
# ---------------------------------------------------------------------------------------------------------------------
def kierto_ruudut(h):
    """Heiton kierteen mittauksen ruudut: yksilolliset ruudut (aikaperusta, kuten netti_seuranta.valmistele_rata), joilla Y on valilla KIERTO_Y_MIN_CM...KIERTO_Y_MAX_CM. Palauttaa (f, X, Y, t, R)."""
    syote = sorted((int(f), float(X), float(Y)) for f, X, Y in h["viivat_syote"])
    f = np.array([p[0] for p in syote], float)
    X = np.array([p[1] for p in syote], float)
    Y = np.array([p[2] for p in syote], float)
    aikap.FPS_LAHDE, aikap.FPS_ULOS = 25.0, 30.0
    aj = aikap.Aikajana(f, Y)
    mk = np.array([int(fi) not in aj.kopio for fi in f])
    t = aj.aika(f[mk])
    f, X, Y = f[mk], X[mk], Y[mk]
    v = (Y >= ns.KIERTO_Y_MIN_CM) & (Y <= ns.KIERTO_Y_MAX_CM)
    return f[v].astype(int), X[v], Y[v], t[v], float(aj.R)


def kierto_havainnot(m, cap, h):
    """netti_kierre.kulmakayrat-funktion syote [(t, f, Xb, Yb, leike, cx0, cy0)] heiton ruuduille videolta (m = netti_seuranta.Mittari, cap = cv2.VideoCapture)."""
    f, X, Y, t, _ = kierto_ruudut(h)
    z0 = m.ast["z0_cm"]
    out = []
    for fi, Xb, Yb, ti in zip(f, X, Y, t):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, ruutu = cap.read()
        if not ok:
            continue
        u, v = m.maa_oikaistuun(Xb, Yb, z0)
        uc, vc = int(round(float(u[0]))), int(round(float(v[0])))
        roi, cx0, cy0 = m.kuvain.leike(ruutu, uc - ns.KIERTO_LEIKE_R, vc - ns.KIERTO_LEIKE_R, uc + ns.KIERTO_LEIKE_R, vc + ns.KIERTO_LEIKE_R)
        if roi is not None:
            out.append((float(ti), int(fi), float(Xb), float(Yb), roi, cx0, cy0))
    return out


def koneen_mittaus(m, cap, h):
    """Heiton kierteen mittaus (kulmakustannuskayrat k ja posteriorin tulos) videolta: dict(k, tulos)."""
    havainnot = kierto_havainnot(m, cap, h)
    muoto = m.kierto_muoto.get(h["vari"])
    muoto_ = tuple(muoto) if muoto is not None else (nk.ko.H_ALKU, nk.ko.ELL_ALKU, nk.ko.W_ALKU)
    k = nk.kulmakayrat(m.cam, m.ast, h["vari"], havainnot, h["viivat"]["t_irr"], muoto_, t0=nk.SARANA_T_ALKU)
    return dict(k=k, tulos=nk.arvioi_posteriori(m.cam, m.ast, h["vari"], havainnot, h["viivat"]["t_irr"], muoto), t_irr=h["viivat"]["t_irr"])


# ---------------------------------------------------------------------------------------------------------------------
# RUUTUKOHTAINEN VERTAILU
# ---------------------------------------------------------------------------------------------------------------------
def ruutuvertailu(valinta, merkinnat, mittaukset):
    """Rivi jokaista ihmisen viivalla (tila ok / epavarma) merkitsemaa ruutua kohti: ihmisen kulma (maan taso), koneen paras kulma, ero, dC, lahin paikallinen minimi, ruudun laatu.
    mittaukset: {(nimi, heitto): dict(k=kulmakayrat-tulos)}."""
    rivit = []
    for n in valinta:
        d = merkinnat.get(n["id"])
        mk = mittaukset.get((n["nimi"], n["heitto"]))
        if d is None or d["tila"] not in ("ok", "epavarma") or mk is None or mk["k"] is None:
            continue
        k = mk["k"]
        j = np.where(k["f"] == n["f"])[0]
        if len(j) == 0:
            continue
        j = int(j[0])
        C = k["Cz"][j]
        cmin = float(C.min())
        sig, puhdas = nk.ruudun_sigma(np.array([cmin]), np.array([k["nkz"][j]]), nk.HANTA_PAINO_LAHI)
        th = maa_kulma(d["kulma_deg"], n["A"])
        am = float(nk.KULMAT[int(np.argmin(C))])
        mi = kayran_minimit(C)
        dC = kustannus_kulmassa(C, th) - cmin
        rivit.append(dict(id=n["id"], nimi=n["nimi"], heitto=n["heitto"], vari=n["vari"], f=n["f"], t=n["t_s"], tila=d["tila"], th_ihminen=th, th_kone=am, ero=float(ero180(am, th)),
                          ero_lahin=float(min((ero180(m_, th) for m_, c in mi), key=abs)) if mi else float("nan"), dC=float(dC), cmin=cmin, nkz=int(k["nkz"][j]), nvz=int(k["nvz"][j]),
                          sigma=float(sig[0]), puhdas=bool(puhdas[0]), pen=float(dC / (2.0 * nk.S2_KAYRA * (sig[0] / nk.SIG_HYVA) ** 2))))
    return rivit


def ruutuyhteenveto(rivit):
    """Yhteenveto ruutukohtaisesta vertailusta: dict(kaikki / puhdas / ei_puhdas / punainen / keltainen / alueittain: n, |ero| mediaani ja keski, merkillinen keskiero, osuus <= 10 astetta, pen keski)."""
    def tilasto(r):
        if not r:
            return None
        e = np.array([x["ero"] for x in r])
        return dict(n=len(r), mediaani_abs=float(np.median(np.abs(e))), keski_abs=float(np.mean(np.abs(e))), keskiero=float(e.mean()), osuus_10=float(np.mean(np.abs(e) <= 10.0)),
                    osuus_20=float(np.mean(np.abs(e) <= 20.0)), lahin_10=float(np.mean([abs(x["ero_lahin"]) <= 10.0 for x in r])), pen_keski=float(np.mean([min(x["pen"], nk.KAYRA_KATKAISU) for x in r])))
    y = {"kaikki": tilasto(rivit), "puhdas": tilasto([x for x in rivit if x["puhdas"]]), "ei_puhdas": tilasto([x for x in rivit if not x["puhdas"]]),
         "punainen": tilasto([x for x in rivit if x["vari"] == "punainen"]), "keltainen": tilasto([x for x in rivit if x["vari"] == "keltainen"])}
    for lo, hi in ((0, 30), (30, 60), (60, 120), (120, 180)):
        y[f"ihmisen_kulma_{lo}_{hi}"] = tilasto([x for x in rivit if lo <= x["th_ihminen"] < hi])
    return y


# ---------------------------------------------------------------------------------------------------------------------
# HEITTOKOHTAINEN VERTAILU
# ---------------------------------------------------------------------------------------------------------------------
def paikallinen_omega(k, a, b):
    """Koneen omega (vakiokierto, kaikki kulmakayrat ajalla [a, b]) tai None, jos ruutuja on alle 4. Palauttaa (omega, sd, ruutuja)."""
    idx = np.where((k["t"] >= a) & (k["t"] <= b))[0]
    if len(idx) < 4:
        return None
    kk = dict(t=k["t"][idx], f=k["f"][idx], Y=k["Y"][idx], Cz=k["Cz"][idx], nkz=k["nkz"][idx], nvz=k["nvz"][idx])
    s, mu, sd, ml, ps = nk.tiivista_posteriori(nk.posteriori_omega(kk, hanta_paino=nk.HANTA_PAINO_LAHI))
    return float(mu), float(sd), int(len(idx))


def heittovertailu(valinta, merkinnat, mittaukset, radat):
    """Rivi jokaista heittoa kohti: ihmisen omega ja keskivirhe, koneen omega (kaikki ruudut), koneen omega ihmisen ajanjaksolta, kaukopaan mittaus ja siita laskettu lahipaan odotus. radat = {nimi: radat.pkl:n lista}."""
    rivit = []
    for key in sorted({(n["nimi"], n["heitto"]) for n in valinta}):
        rs = sorted([n for n in valinta if (n["nimi"], n["heitto"]) == key], key=lambda n: n["t_s"])
        v = [(n["t_s"], merkinnat[n["id"]]["kulma_deg"]) for n in rs if n["id"] in merkinnat and merkinnat[n["id"]]["tila"] in ("ok", "epavarma")]
        r = dict(nimi=key[0], heitto=key[1], vari=rs[0]["vari"], viivallisia=len(v))
        if len(v) >= ke.MIN_RUUTUJA and max(x[0] for x in v) - min(x[0] for x in v) >= ke.MIN_VALI_S:
            om, se, sd, hyppy = ke.sovita_omega([x[0] for x in v], [x[1] for x in v])
            r.update(omega_ihminen=om, se_ihminen=se, vali_s=max(x[0] for x in v) - min(x[0] for x in v), epaselva_purku=bool(hyppy > ke.HYPPY_RAJA))
        mk = mittaukset.get(key)
        if mk is not None and mk["tulos"] and "virhe" not in mk["tulos"]:
            t = mk["tulos"]
            r.update(omega_kone=t["omega_deg_s"], sd_kone=t["omega_sd_deg_s"], p_suunta=t["p_suunta"], varma=bool(t["varma"]), t_keski=t["t_keski"])
            if mk["k"] is not None and v:
                p = paikallinen_omega(mk["k"], min(x[0] for x in v) - PAIKALLINEN_MARGINAALI_S, max(x[0] for x in v) + PAIKALLINEN_MARGINAALI_S)
                if p is not None:
                    r.update(omega_paikallinen=p[0], sd_paikallinen=p[1], ruutuja_paikallinen=p[2])
        h = (radat.get(key[0]) or [None] * (key[1]))[key[1] - 1]
        kauko = (h or {}).get("kauko_kierto")
        if kauko and "virhe" not in kauko and "t_keski" in r:
            dt = kauko["t_keski"] - r["t_keski"]
            r.update(omega_kauko=kauko["omega_deg_s"], sd_kauko=kauko["omega_sd_deg_s"], kauko_varma=bool(kauko.get("varma")), dt_kauko=float(dt), omega_ennuste=float(kauko["omega_deg_s"] * np.exp(nk.K_KIERRE * dt)))
        if "omega_ihminen" in r and "omega_kone" in r:
            r["z_ihminen_kone"] = (r["omega_kone"] - r["omega_ihminen"]) / float(np.hypot(r["sd_kone"], r["se_ihminen"]))
            r["suunta_sama"] = bool(r["omega_kone"] * r["omega_ihminen"] > 0)
        rivit.append(r)
    return rivit


def heittoyhteenveto(rivit):
    """Heittovertailun yhteenveto: suunta (ihminen vs kone, kone vs kaukopaa), omegan ero (kaikki ruudut / ihmisen ajanjakso) ja z-arvojen rms."""
    v = [r for r in rivit if "omega_ihminen" in r and "omega_kone" in r]
    y = dict(heittoja=len(rivit), ihmisen_omegalla=len(v))
    if v:
        e = np.array([r["omega_kone"] - r["omega_ihminen"] for r in v])
        y.update(suunta_sama=int(sum(r["suunta_sama"] for r in v)), ero_abs_keski=float(np.mean(np.abs(e))), ero_abs_mediaani=float(np.median(np.abs(e))),
                 z_rms=float(np.sqrt(np.mean([r["z_ihminen_kone"] ** 2 for r in v]))))
    p = [r for r in v if "omega_paikallinen" in r]
    if p:
        e = np.array([r["omega_paikallinen"] - r["omega_ihminen"] for r in p])
        y.update(paikallinen_ero_abs_keski=float(np.mean(np.abs(e))), paikallinen_ero_abs_mediaani=float(np.median(np.abs(e))))
    k = [r for r in rivit if "omega_ennuste" in r and "omega_kone" in r and r.get("kauko_varma") and r.get("varma")]
    if k:
        y.update(kauko_vertailtu=len(k), kauko_suunta_sama=int(sum(r["omega_kone"] * r["omega_ennuste"] > 0 for r in k)))
    return y


# ---------------------------------------------------------------------------------------------------------------------
# RAPORTTI
# ---------------------------------------------------------------------------------------------------------------------
def tulosta_raportti(rr, hr, ry, hy, tulosta=print):
    tulosta(f"Ruutukohtainen vertailu ({ry['kaikki']['n'] if ry.get('kaikki') else 0} ruutua, joihin ihminen piirsi viivan): koneen paras kulma vs ihmisen viiva")
    for nimi in ("kaikki", "puhdas", "ei_puhdas", "punainen", "keltainen", "ihmisen_kulma_0_30", "ihmisen_kulma_30_60", "ihmisen_kulma_60_120", "ihmisen_kulma_120_180"):
        t = ry.get(nimi)
        if t:
            tulosta(f"  {nimi:<22} n={t['n']:>3}  |ero| mediaani {t['mediaani_abs']:5.1f}  keski {t['keski_abs']:5.1f}  keskiero {t['keskiero']:+5.1f}  <=10 astetta {t['osuus_10'] * 100:3.0f} %  <=20: {t['osuus_20'] * 100:3.0f} %  lahin minimi <=10: {t['lahin_10'] * 100:3.0f} %  pen {t['pen_keski']:.2f}")
    tulosta("\nHeittokohtainen omega (aste / s, + = vastapaivaan):")
    tulosta(f"{'heitto':<16}{'vari':<10}{'ihminen':>16}{'kone (kaikki)':>17}{'kone (ihm. aika)':>18}{'kauko':>8}{'kauko->lahi':>12}  suunta")
    for r in hr:
        ih = f"{r['omega_ihminen']:+7.1f}+-{r['se_ihminen']:4.1f}" if "omega_ihminen" in r else "-"
        ko = f"{r['omega_kone']:+7.1f}+-{r['sd_kone']:5.1f}" if "omega_kone" in r else "-"
        pa = f"{r['omega_paikallinen']:+7.1f}+-{r['sd_paikallinen']:5.1f}" if "omega_paikallinen" in r else "-"
        ka = f"{r['omega_kauko']:+7.1f}" if "omega_kauko" in r else "-"
        en = f"{r['omega_ennuste']:+7.1f}" if "omega_ennuste" in r else "-"
        su = {True: "sama", False: "ERI", None: ""}.get(r.get("suunta_sama"), "") + (" varma" if r.get("varma") else "")
        tulosta(f"{r['nimi'][:9] + ' h' + str(r['heitto']):<16}{r['vari']:<10}{ih:>16}{ko:>17}{pa:>18}{ka:>8}{en:>12}  {su}")
    if hy.get("ihmisen_omegalla"):
        tulosta(f"Suunta: kone ja ihminen samaa mielta {hy['suunta_sama']} / {hy['ihmisen_omegalla']} heitossa; omegan ero |kone - ihminen| keskimaarin {hy['ero_abs_keski']:.1f}, mediaani {hy['ero_abs_mediaani']:.1f} aste/s, z rms {hy['z_rms']:.2f}"
                + (f"; saman ajanjakson koneen omega: keski {hy['paikallinen_ero_abs_keski']:.1f}, mediaani {hy['paikallinen_ero_abs_mediaani']:.1f}" if "paikallinen_ero_abs_keski" in hy else ""))
    if hy.get("kauko_vertailtu"):
        tulosta(f"Kaukopaa (varma) ja lahipaa (varma): suunta sama {hy['kauko_suunta_sama']} / {hy['kauko_vertailtu']} heitossa.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--valinta", required=True)
    ap.add_argument("--merkinnat", required=True)
    ap.add_argument("--ajo", nargs=3, action="append", metavar=("NIMI", "VIDEO", "RADAT"), required=True, help="video, videotiedosto ja netti_live.py --radat -pickle; voi toistaa")
    ap.add_argument("--kansio", action="append", default=[], metavar="NIMI=POLKU", help="kalibrointikansio videolle (oletus kierre_leikkeet.KANSIOT)")
    ap.add_argument("--tulos", help="kirjoita rivit ja yhteenvedot JSON-tiedostoon")
    a = ap.parse_args()
    valinta = json.load(open(a.valinta, encoding="utf-8"))
    merkinnat = ke.lue_merkinnat(a.merkinnat)
    kans = dict(k.split("=", 1) for k in a.kansio)
    juuri = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    videot = {nimi: (video, radat) for nimi, video, radat in a.ajo}
    radat = {nimi: pickle.load(open(r, "rb")) for nimi, (v, r) in videot.items()}
    mitt, caps, mittaukset = {}, {}, {}
    for nimi, heitto in sorted({(n["nimi"], n["heitto"]) for n in valinta}):
        if nimi not in videot:
            continue
        h = radat[nimi][heitto - 1]
        if (nimi, h["puoli"]) not in mitt:
            mitt[(nimi, h["puoli"])] = ns.Mittari(kans.get(nimi) or os.path.join(juuri, kl.KANSIOT[nimi]), h["puoli"], runko=False)
        if nimi not in caps:
            caps[nimi] = cv2.VideoCapture(videot[nimi][0])
        mittaukset[(nimi, heitto)] = koneen_mittaus(mitt[(nimi, h["puoli"])], caps[nimi], h)
        print(f"  {nimi} heitto {heitto}: ruutuja kulmakayralla {0 if mittaukset[(nimi, heitto)]['k'] is None else len(mittaukset[(nimi, heitto)]['k']['t'])}", flush=True)
    rr = ruutuvertailu(valinta, merkinnat, mittaukset)
    ry = ruutuyhteenveto(rr)
    hr = heittovertailu(valinta, merkinnat, mittaukset, radat)
    hy = heittoyhteenveto(hr)
    tulosta_raportti(rr, hr, ry, hy)
    if a.tulos:
        json.dump(dict(ruudut=rr, ruutuyhteenveto=ry, heitot=hr, heittoyhteenveto=hy), open(a.tulos, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)


if __name__ == "__main__":
    main()
