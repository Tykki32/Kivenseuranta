"""Kameroiden heilumisen mittaus: kuvan siirtyma (dx, dy) ajan funktiona kummassakin puoliskossa (vasen ja oikea kamera ovat omia kameroitaan).

    python tools/kameran_heilunta.py video.mp4                        # koko video, joka 3. ruutu (10 r/s), 3 prosessia
    python tools/kameran_heilunta.py video.mp4 --alku 600 --loppu 700 --vali 1 --ulos heilunta.npz
    python tools/kameran_heilunta.py video.mp4 --testi                # itsetesti: tunnetut siirrot (0,3 ... 8 px) ruutuun -> mitataanko ne takaisin

Menetelma: referenssi = puoliskon mediaanikuva (n ruutua tasaisesti koko videolta: ihmiset ja kivet poistuvat mediaanista); referenssista valitaan kiintein piirtein rikkaimmat
ikkunat (viivat, talo, tekstit, laudat; ei ylakulmien minikarttoja), ja jokaiselle ruudulle lasketaan kunkin ikkunan siirtyma referenssiin vaihekorrelaatiolla (cv2.phaseCorrelate,
Hann-ikkuna, alipikselitarkkuus). Puoliskon siirtyma = ikkunoiden mediaani (liikkuva ihminen / kivi ikkunassa ei vaikuta); hajonta ikkunoiden valilla kertoo, onko liike muuta kuin
siirtymaa (kierto / zoomi). Tulostaa yhteenvedon (kohinataso, hajonta, ylitysten maara, pisimmat jaksot) ja tallentaa aikasarjan (--ulos .npz).

Etumerkki: (dx, dy) = kuinka monta pikselia ruudun sisalto on siirtynyt referenssiin nahden (+dx: oikealle, +dy: alas).
Huom: cv2.phaseCorrelate kertoo ikkunafunktion (Hann) syotekuviin PAIKALLAAN; siksi ikkunat annetaan kopioina (muuten referenssi muuttuisi joka kutsulla)."""
import argparse
import time
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

PUOLET = {"vasen": (0, 190, 960, 890), "oikea": (960, 190, 960, 890)}          # (x0, y0, leveys, korkeus), kuten kalibrointi_netti.PUOLET
IKKUNA = 192                       # vaihekorrelaation ikkuna (px)
IKKUNOITA = 14                     # kayttoon valittavien ikkunoiden maara puoliskoa kohti
YLAREUNA = 95                      # ylimmat rivit (puoliskon rajauksessa) jatetaan pois: minikartat ja sponsorigrafiikka
VASTE_MIN = 0.08                   # ikkunan vaihekorrelaatiovasteen vahimmaisarvo


def _harmaa(kuva):
    return cv2.cvtColor(kuva, cv2.COLOR_BGR2GRAY)


def _puolisko(harmaa, nimi):
    x, y, w, h = PUOLET[nimi]
    return harmaa[y:y + h, x:x + w]


def referenssi(video, n=96, alku_s=30.0):
    """Puoliskojen mediaanikuvat n ruudusta, tasaisesti videon yli: {nimi: float32}."""
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    kok = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ruudut = np.linspace(int(alku_s * fps), kok - int(alku_s * fps), n).astype(int)
    pino = {k: [] for k in PUOLET}
    for r in ruudut:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(r))
        ok, kuva = cap.read()
        if not ok:
            continue
        g = _harmaa(kuva)
        for k in PUOLET:
            pino[k].append(_puolisko(g, k))
    cap.release()
    return {k: np.median(np.stack(v), axis=0).astype(np.float32) for k, v in pino.items()}


def valitse_ikkunat(ref, n=IKKUNOITA, koko=IKKUNA):
    """Referenssikuvan rikkaimmat (gradienttienergia) ei-paallekkaiset ikkunat: [(x, y)] vasen ylakulma."""
    gx = cv2.Sobel(ref, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(ref, cv2.CV_32F, 0, 1, ksize=3)
    e = cv2.boxFilter(gx * gx + gy * gy, -1, (koko, koko), normalize=True)
    h, w = ref.shape
    ehdokkaat = []
    for y in range(YLAREUNA, h - koko + 1, 16):
        for x in range(0, w - koko + 1, 16):
            ehdokkaat.append((float(e[y + koko // 2, x + koko // 2]), x, y))
    ehdokkaat.sort(reverse=True)
    valitut = []
    for _, x, y in ehdokkaat:
        if all(abs(x - a) >= koko or abs(y - b) >= koko for a, b in valitut):
            valitut.append((x, y))
            if len(valitut) >= n:
                break
    return valitut


def mittaa_ruutu(g, ref, ikkunat, hann):
    """Puoliskon siirtyma referenssiin: (dx, dy, vaste, hajonta_ikkunoiden_valilla, kelvollisia ikkunoita)."""
    dxs, dys, vs = [], [], []
    koko = IKKUNA
    for x, y in ikkunat:
        a = ref[y:y + koko, x:x + koko].copy()          # phaseCorrelate kertoo ikkunafunktion syotteeseen paikallaan: kopiot, ettei referenssi muutu
        b = g[y:y + koko, x:x + koko].copy()
        (dx, dy), v = cv2.phaseCorrelate(a, b, hann)
        if v >= VASTE_MIN and abs(dx) < 20 and abs(dy) < 20:
            dxs.append(dx)
            dys.append(dy)
            vs.append(v)
    if len(dxs) < 3:
        return np.nan, np.nan, 0.0, np.nan, len(dxs)
    dx, dy = float(np.median(dxs)), float(np.median(dys))
    haj = float(np.median(np.hypot(np.array(dxs) - dx, np.array(dys) - dy)))
    return dx, dy, float(np.median(vs)), haj, len(dxs)


def _tyo(args):
    video, alku, loppu, vali, ref, ikk = args
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, alku)
    hann = cv2.createHanningWindow((IKKUNA, IKKUNA), cv2.CV_32F)
    ruudut, arvot = [], []
    i = alku
    while i < loppu:
        if (i - alku) % vali == 0:
            ok, kuva = cap.read()
            if not ok:
                break
            g = _harmaa(kuva)
            rivi = []
            for k in ("vasen", "oikea"):
                rivi += list(mittaa_ruutu(_puolisko(g, k).astype(np.float32), ref[k], ikk[k], hann))
            ruudut.append(i)
            arvot.append(rivi)
            i += 1
        else:
            if not cap.grab():
                break
            i += 1
    cap.release()
    return np.array(ruudut), np.array(arvot)


def jaksot(t, ylitys, min_vali_s=2.0):
    """Yhtenaiset jaksot, joilla ylitys on tosi (lyhyet aukot <= min_vali_s yhdistetaan): [[alku_s, loppu_s]]."""
    out = []
    for a in t[ylitys]:
        if out and a - out[-1][1] <= min_vali_s:
            out[-1][1] = a
        else:
            out.append([a, a])
    return out


def yhteenveto(t, arvot):
    for j, k in enumerate(("vasen", "oikea")):
        dx, dy, v, haj, n = (arvot[:, 5 * j + i] for i in range(5))
        ok = np.isfinite(dx)
        dx0, dy0 = np.nanmedian(dx), np.nanmedian(dy)
        r = np.hypot(dx - dx0, dy - dy0)
        print(f"\n{k.upper()} puolisko: {int(ok.sum())} / {len(t)} ruutua mitattu; vaste mediaani {np.nanmedian(v):.2f}, ikkunoita mediaani {int(np.nanmedian(n))}")
        print(f"  mediaanisiirtyma referenssiin ({dx0:+.2f}, {dy0:+.2f}) px; kohina (ikkunoiden valinen hajonta) mediaani {np.nanmedian(haj):.3f} px")
        print(f"  dx: sd {np.nanstd(dx):.3f} px, dy: sd {np.nanstd(dy):.3f} px")
        print(f"  poikkeama mediaanista: robusti sd {1.4826 * np.nanmedian(np.abs(r - np.nanmedian(r))):.3f} px, p50 {np.nanpercentile(r, 50):.3f}, p90 {np.nanpercentile(r, 90):.3f}, "
              f"p99 {np.nanpercentile(r, 99):.3f}, p99,9 {np.nanpercentile(r, 99.9):.3f}, max {np.nanmax(r):.2f} px")
        for raja in (0.3, 0.5, 1.0, 2.0, 4.0):
            m = r > raja
            print(f"  yli {raja:3.1f} px: {int(np.nansum(m))} ruutua ({100.0 * np.nansum(m) / max(ok.sum(), 1):.2f} %), jaksoja {len(jaksot(t, m))}")
        m = r > 1.0
        if m.any():
            js = jaksot(t, m)
            js.sort(key=lambda x: -(x[1] - x[0]))
            print("  pisimmat jaksot (> 1 px): " + ", ".join(f"{a:.0f}-{b:.0f} s" for a, b in js[:12]))
        # hidas ajelehtiminen: minuuttimediaanit
        mins = np.floor(t / 60.0).astype(int)
        med = np.array([np.nanmedian(np.hypot(dx[mins == u] - dx0, dy[mins == u] - dy0)) for u in np.unique(mins) if (mins == u).sum() > 20])
        if len(med):
            print(f"  minuuttimediaanien poikkeama: max {np.nanmax(med):.3f} px (hidas ajelehtiminen)")
        p999 = np.nanpercentile(r, 99.9)
        print("  -> " + (f"VAKAA: 99,9 % ruuduista alle {max(p999, 0.1):.1f} px:n poikkeama, ei korjattavaa" if p999 < 1.0 and np.nanmax(r) < 2.0
                         else f"HEILUU: {len(jaksot(t, r > 1.0))} jaksoa, joilla poikkeama > 1 px (suurin {np.nanmax(r):.1f} px); kuva kannattaa vakauttaa"))


def itsetesti(video, ref, ikk):
    """Tunnetut siirrot (warpAffine) yhteen ruutuun -> mitataanko ne takaisin; tulostaa virheen ja palauttaa suurimman virheen (px, ilman ruudun omaa siirtymaa)."""
    hann = cv2.createHanningWindow((IKKUNA, IKKUNA), cv2.CV_32F)
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(600 * (cap.get(cv2.CAP_PROP_FPS) or 30.0)))
    ok, kuva = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("ruutua ei voitu lukea")
    g = _harmaa(kuva)
    suurin = 0.0
    for k in ("vasen", "oikea"):
        puoli = _puolisko(g, k).astype(np.float32)
        perus = mittaa_ruutu(puoli, ref[k], ikk[k], hann)[:2]
        for sx, sy in ((0.3, 0.0), (0.0, -0.3), (0.5, 0.5), (1.0, -1.0), (2.5, 1.25), (-4.0, 3.0), (8.0, -6.0)):
            siirretty = cv2.warpAffine(puoli, np.float32([[1, 0, sx], [0, 1, sy]]), (puoli.shape[1], puoli.shape[0]), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            dx, dy, v, _, n = mittaa_ruutu(siirretty, ref[k], ikk[k], hann)
            ex, ey = dx - perus[0] - sx, dy - perus[1] - sy
            suurin = max(suurin, abs(ex), abs(ey))
            print(f"  {k:6s} asetettu ({sx:+5.2f}, {sy:+5.2f}) -> mitattu ({dx - perus[0]:+5.2f}, {dy - perus[1]:+5.2f}) virhe ({ex:+.2f}, {ey:+.2f}) px, vaste {v:.2f}, ikkunoita {n}")
    print(f"suurin virhe {suurin:.2f} px -> " + ("OK (siirrot mitataan takaisin oikealla etumerkilla)" if suurin < 0.3 else "EPAONNISTUI"))
    return suurin


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--alku", type=float, default=0.0, help="alku (s)")
    ap.add_argument("--loppu", type=float, default=None, help="loppu (s)")
    ap.add_argument("--vali", type=int, default=3, help="mitataan joka n. ruutu (3 = 10 r/s)")
    ap.add_argument("--prosesseja", type=int, default=3)
    ap.add_argument("--ulos", help="aikasarja (.npz): ruutunumerot ja arvot (dx, dy, vaste, hajonta, ikkunoita) vasen, oikea")
    ap.add_argument("--testi", action="store_true", help="itsetesti tunnetuilla siirroilla (ei mittaa koko videota)")
    a = ap.parse_args()
    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    kok = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    f0 = int(a.alku * fps)
    f1 = min(kok, int(a.loppu * fps)) if a.loppu else kok
    t0 = time.time()
    ref = referenssi(a.video)
    ikk = {k: valitse_ikkunat(ref[k]) for k in ref}
    print(f"referenssi {time.time() - t0:.0f} s; ikkunoita: " + ", ".join(f"{k} {len(v)}" for k, v in ikk.items()), flush=True)
    if a.testi:
        return 0 if itsetesti(a.video, ref, ikk) < 0.3 else 1
    n = max(1, a.prosesseja)
    rajat = np.linspace(f0, f1, n + 1).astype(int)
    rajat = [(int(b) // a.vali * a.vali if i else f0) for i, b in enumerate(rajat)]       # segmenttien alut vali-ruudukkoon
    tyot = [(a.video, rajat[i], rajat[i + 1] if i + 1 < n else f1, a.vali, ref, ikk) for i in range(n)]
    with ProcessPoolExecutor(n) as ex:
        tulokset = list(ex.map(_tyo, tyot))
    ruudut = np.concatenate([r for r, _ in tulokset])
    arvot = np.concatenate([v for _, v in tulokset])
    print(f"{len(ruudut)} ruutua mitattu {time.time() - t0:.0f} s")
    t = ruudut / fps
    yhteenveto(t, arvot)
    if a.ulos:
        np.savez(a.ulos, ruudut=ruudut, t=t, arvot=arvot, fps=fps)
        print("\ntallennettu", a.ulos)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
