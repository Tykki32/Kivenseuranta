"""Kontaktiarkki irroituksen ymparilta: silmin tarkistus, onko tunnistettu irroitus oikeassa kohdassa.

    python netti_live.py --lahde netti.mp4 --nopea --ei-ikkunaa --ei-kierretta --radat radat.pkl      # kerran (radat.pkl sisaltaa viivaestimaatin tarkan syotteen)
    python tools/irroitus_arkki.py radat.pkl netti.mp4 --heitot 13,37                                  # arkit/heitto_13.jpg, arkit/heitto_37.jpg
    python tools/irroitus_arkki.py radat.pkl netti.mp4 --heitot 13 --vertaa hanta --alue -250,300      # lisaksi vanhan menetelman irroitus sinisella, laajempi ruutuvali

Yksi kuva per heitto: kiven (kahvan levyn tason, z = 15 cm) ymparille rajatut, 2-kertaisesti suurennetut alkuperaisen videon ruudut radan Y-valilta [irroitus + alku, irroitus + loppu] (oletus -150 ... +250 cm),
vain yksilolliset ruudut (lahdevirran kopioruudut pois), aikajarjestyksessa vasemmalta oikealle ja ylhaalta alas. Teksti: ruutu, kiven Y ja aika irroituksesta. Punainen = irroituksen lahin ruutu
(`liuku_irroitus.MENETELMA`, tai --menetelma), sininen = --vertaa-menetelman irroitus. Ristikko = kiven keskipiste radalla.
Mita katsotaan: kinemaattinen irroitus on viimeinen hetki, jolloin kasi vaikuttaa rataan. Keltaisella kahvalla sormet irtoavat kahvalta 0-60 cm irroituksen jalkeen; punaisella kasi seuraa kahvaa usein
viela pitkaan ilman voimaa (irroitus on silloin aikaisemmin kuin sormien irtoaminen: kinemaattisesti ei tarkemmin maaritettavissa). Otsikossa nopeus ennen / jalkeen ja merkitsevyys."""
import argparse
import os
import pickle
import sys
import warnings

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import aikaperusta as aikap  # noqa: E402
import liuku_irroitus as li  # noqa: E402
import netti_geom as ng  # noqa: E402
import netti_seuranta as ns  # noqa: E402


def yksilolliset(f, Y, fps_lahde, fps_ulos):
    """(totuusarvotaulukko yksilollisille ruuduille [ei lahdevirran kopioruutuja; netti_seuranta.valmistele_rata:n tapa], kaikkien ruutujen aika todellisella aikaperustalla)."""
    aikap.FPS_LAHDE, aikap.FPS_ULOS = float(fps_lahde), float(fps_ulos)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        aj = aikap.Aikajana(f, Y)
        return np.array([int(fi) not in aj.kopio for fi in f]), np.asarray(aj.aika(f), float)


def irroitus(f, X, Y, menetelma, fps_lahde, fps_ulos):
    """netti_seuranta.laske_viivat valitulla menetelmalla (palauttaa dict tai None)."""
    vanha = li.MENETELMA
    li.MENETELMA = menetelma
    try:
        v = ns.laske_viivat(f, X, Y, fps_lahde, fps_ulos)
    finally:
        li.MENETELMA = vanha
    return None if v is None or "virhe" in v else v


def teksti(kuva, s, paikka, vari, koko=0.5):
    cv2.putText(kuva, s, paikka, cv2.FONT_HERSHEY_SIMPLEX, koko, (0, 0, 0), 3)
    cv2.putText(kuva, s, paikka, cv2.FONT_HERSHEY_SIMPLEX, koko, vari, 1)


def heiton_arkki(n, h, cap, geom, a):
    f, X, Y = (np.array(c, float) for c in zip(*h["viivat_syote"]))
    o = np.argsort(f)
    f, X, Y = f[o], X[o], Y[o]
    v = irroitus(f, X, Y, a.menetelma, a.fps_lahde, a.fps)
    if v is None:
        print(f"heitto {n}: ei viivaestimaattia")
        return False
    vv = irroitus(f, X, Y, a.vertaa, a.fps_lahde, a.fps) if a.vertaa else None
    mk, t = yksilolliset(f, Y, a.fps_lahde, a.fps)
    f, X, Y, t = f[mk], X[mk], Y[mk], t[mk]
    lo, hi = v["irr_Y"] + a.alue[0], v["irr_Y"] + a.alue[1]
    idx = [k for k in range(len(f)) if lo <= Y[k] <= hi][::a.askel]
    if not idx:
        print(f"heitto {n}: ei ruutuja valilla Y = {lo:.0f} ... {hi:.0f} cm")
        return False
    k_irr = min(idx, key=lambda k: abs(Y[k] - v["irr_Y"]))
    k_vrt = None if vv is None else min(idx, key=lambda k: abs(Y[k] - vv["irr_Y"]))
    G = geom[h["puoli"]]
    laatat = []
    for k in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f[k]))
        ok, fr = cap.read()
        if not ok:
            continue
        uv = G.maa_kuvaan([X[k]], [Y[k]], a.z)[0]
        cx, cy = int(round(uv[0])), int(round(uv[1]))
        x0 = max(0, min(cx - a.leveys // 2, fr.shape[1] - a.leveys))
        y0 = max(0, min(cy - a.korkeus // 2, fr.shape[0] - a.korkeus))
        laatta = cv2.resize(fr[y0:y0 + a.korkeus, x0:x0 + a.leveys].copy(), None, fx=a.zoom, fy=a.zoom, interpolation=cv2.INTER_CUBIC)
        cv2.drawMarker(laatta, (int((cx - x0) * a.zoom), int((cy - y0) * a.zoom)), (0, 255, 255), cv2.MARKER_CROSS, 12, 1)
        vari = (0, 0, 255) if k == k_irr else ((255, 128, 0) if k == k_vrt else (255, 255, 255))
        teksti(laatta, f"f={int(f[k])} Y={Y[k]:.0f} ({t[k] - v['t_irr']:+.2f} s)", (4, 16), vari)
        if k == k_irr:
            cv2.rectangle(laatta, (0, 0), (laatta.shape[1] - 1, laatta.shape[0] - 1), (0, 0, 255), 3)
        elif k == k_vrt:
            cv2.rectangle(laatta, (0, 0), (laatta.shape[1] - 1, laatta.shape[0] - 1), (255, 128, 0), 3)
        laatat.append(laatta)
    hh, ww = laatat[0].shape[:2]
    while len(laatat) % a.sarakkeita:
        laatat.append(np.zeros((hh, ww, 3), np.uint8))
    kuva = np.vstack([np.hstack(laatat[r:r + a.sarakkeita]) for r in range(0, len(laatat), a.sarakkeita)])
    ots = np.zeros((44, kuva.shape[1], 3), np.uint8)
    rivi = (f"heitto {n}: {h['vari']} {h.get('kivi_nro')} ({h['puoli']}), irroitus ({a.menetelma}) Y = {v['irr_Y']:.0f} cm [{v['irr_Y_vali'][0]:.0f} ... {v['irr_Y_vali'][1]:.0f}], "
            f"{'merkitseva' if v['merkitseva'] else 'EI merkitseva'}, nopeus {v['nopeus_ennen']:.2f} -> {v['nopeus_jalkeen']:.2f} m/s")
    teksti(ots, rivi, (6, 18), (255, 255, 255))
    if vv is not None:
        teksti(ots, f"{a.vertaa}: Y = {vv['irr_Y']:.0f} cm, nopeus {vv['nopeus_ennen']:.2f} -> {vv['nopeus_jalkeen']:.2f} m/s (sininen kehys)", (6, 38), (255, 128, 0))
    kuva = np.vstack([ots, kuva])
    os.makedirs(a.ulos, exist_ok=True)
    polku = os.path.join(a.ulos, f"heitto_{n}.jpg")
    cv2.imwrite(polku, kuva, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"heitto {n}: irroitus Y = {v['irr_Y']:.0f} cm{'' if vv is None else f' ({a.vertaa} {vv['irr_Y']:.0f} cm)'}, {len(idx)} ruutua -> {polku}")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("radat", help="netti_live.py --radat -tiedosto (pickle)")
    ap.add_argument("video", help="analysoitu video")
    ap.add_argument("--heitot", required=True, help="heittojen numerot pilkuilla eroteltuna (1-alkuinen, kuten CSV:ssa)")
    ap.add_argument("--menetelma", choices=("muutos", "hanta", "kinkki"), default=li.MENETELMA, help="irroituksen tunnistus (oletus: liuku_irroitus.MENETELMA)")
    ap.add_argument("--vertaa", choices=("muutos", "hanta", "kinkki"), default=None, help="merkitse lisaksi toisen menetelman irroitus (sininen)")
    ap.add_argument("--alue", default="-150,250", help="ruutuvali irroituksesta (cm): alku,loppu")
    ap.add_argument("--askel", type=int, default=2, help="joka n:s yksilollinen ruutu")
    ap.add_argument("--sarakkeita", type=int, default=4)
    ap.add_argument("--leveys", type=int, default=300, help="rajauksen leveys (px)")
    ap.add_argument("--korkeus", type=int, default=190, help="rajauksen korkeus (px)")
    ap.add_argument("--zoom", type=float, default=2.0)
    ap.add_argument("--z", type=float, default=15.0, help="kuvattava korkeus (cm): kahvan levyn taso")
    ap.add_argument("--kansio", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tulokset"), help="kalibrointikansio")
    ap.add_argument("--ulos", default="arkit", help="tulosten kansio")
    ap.add_argument("--fps-lahde", type=float, default=25.0)
    ap.add_argument("--fps", type=float, default=30.0)
    a = ap.parse_args()
    a.alue = tuple(float(x) for x in a.alue.split(","))
    heitot = pickle.load(open(a.radat, "rb"))
    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        print("videota ei voi avata:", a.video)
        return 1
    geom = ng.Geometria(a.kansio)
    ok = 0
    for n in (int(x) for x in a.heitot.split(",")):
        if not 1 <= n <= len(heitot) or not heitot[n - 1].get("viivat_syote"):
            print(f"heitto {n}: ei viivat_syote-tietoa radat-tiedostossa")
            continue
        ok += bool(heiton_arkki(n, heitot[n - 1], cap, geom, a))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
