"""Kivitunnistus videolta: python aja_kivi.py VIDEO --puoli vasen --ulos tulokset [--ikkunat 12-22,58-66] [--saikeet 4]

Video on 25 r/s -virrasta tallennettu 30 r/s:ksi, jolloin joka kuudes ruutu on edellisen kopio. Kopioruudut ohitetaan ja
aika lasketaan yksilollisista ruuduista (t = ikkunan alku + k / 25), jotta nopeus ja kiihtyvyys eivat vaihtele ruudun
mukana. Jokaisesta ruudusta tunnistetaan kivet (kivi_netti.tunnista, paikka pehmealla siluettisovituksella) ja
havainnot ketjutetaan radoiksi.

Kirjoittaa <ulos>/kivet_<puoli>.csv (kaikki havainnot) ja <ulos>/radat_<puoli>.csv (ketjutetut radat)."""
import argparse
import csv
import os
from multiprocessing import Pool

import cv2
import numpy as np

import aikaperusta as aikap
import kivi_netti as kv
import kivi_tarkka as kt

FPS_LAHDE = 25.0          # alkuperaisen virran ruutunopeus
KOPIO_KYNNYS = 15.0       # |ero|:n 99,9. persentiili (harmaasavy, puolisko taysresoluutiolla) alle taman = kopioruutu
                          # (kopiot 2-12, oikeat ruudut staattisessakin kuvassa >= 30, liikkuvassa kivessa ~200)

_G = {}
T_KARTTA = {}            # ruutu -> aika (s) = ikkunan alku + k / 25 (yksilollisten ruutujen tasavali)


def _alusta(video, puoli, ulos):
    _G["cam"] = kv.Kamera(puoli, os.path.join(ulos, f"kalibrointi_{puoli}.json"), os.path.join(ulos, f"moodikuva_{puoli}.png"))
    _G["kivi"] = kv.Kivi()                                    # karkea siluetti: vain siemenet ja alkuarvaus
    _G["runko"] = kt.Runko.lataa(ulos, puoli)                  # tarkka runkomalli (kiintea koko), kalibroi_kivi.py
    _G["cap"] = cv2.VideoCapture(video)


def _tyo(i):
    cam, kivi, cap, runko = _G["cam"], _G["kivi"], _G["cap"], _G["runko"]
    fr = cam.ruutu(cap, i)
    if fr is None:
        return i, []
    dets, (fg, g, h) = kv.tunnista(cam, kivi, fr)
    alfa, ok = kt.havainto(cam, fr, kv.alfa_kartta(cam, fr, g))
    out = []
    for d in dets:
        s = kt.sovita(cam, runko, alfa, ok, d["X"], d["Y"])
        out.append((s["X"], s["Y"], s["rms"], d["peitto"], s["kelvolliset"]))
    return i, out


def yksilolliset_ruudut(video, ikkunat, puoli="vasen"):
    """[(ruutu, k)] ikkunoiden yksilollisille ruuduille; k = yksilollisten ruutujen jarjestysnumero ikkunassa."""
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = []
    for t0, t1 in ikkunat:
        f0, f1 = int(t0 * fps), min(n, int(t1 * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
        prev, k = None, 0
        for f in range(f0, f1):
            ok, fr = cap.read()
            if not ok:
                break
            x, y, w, h = kv.kn.PUOLET[puoli]
            g = cv2.cvtColor(fr[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY).astype(np.int16)
            kopio = prev is not None and float(np.percentile(np.abs(g - prev), 99.9)) < KOPIO_KYNNYS
            prev = g
            if not kopio:
                out.append((f, k))
                T_KARTTA[f] = t0 + k / FPS_LAHDE
                k += 1
    cap.release()
    return out


def ketjuta(rows, portti_cm=18.0, portti_per_ruutu_cm=8.0, max_aukko=14, min_pisteet=10, min_siirtyma_cm=500.0):
    """rows: [(frame, k, X, Y, err)] -> radat: listat (frame, k, X, Y, err) (aikaleimaa() antaa ajan). Vakionopeusennuste: havainto liitetaan rataan, jonka ennustettu
    paikka on lahinna (portti portti_cm + portti_per_ruutu_cm per puuttuva ruutu); aukko enintaan max_aukko ruutua."""
    rows = sorted(rows)
    tracks = []

    def ennuste(tr, k):
        if len(tr) >= 3:
            a, b = tr[max(0, len(tr) - 4)], tr[-1]
            dk = max(b[1] - a[1], 1)
            vx, vy = (b[2] - a[2]) / dk, (b[3] - a[3]) / dk
        else:
            vx = vy = 0.0
        return tr[-1][2] + vx * (k - tr[-1][1]), tr[-1][3] + vy * (k - tr[-1][1])

    for f in sorted(set(r[0] for r in rows)):
        dets = [r for r in rows if r[0] == f]
        k = dets[0][1]
        # ikkunat (tallennuksen osat) ovat erillisia: ketjutetaan vain ruutuja, joiden numero on lahella (k alkaa ikkunassa nollasta)
        free = [tr for tr in tracks if 0 < k - tr[-1][1] <= max_aukko and 0 < f - tr[-1][0] <= 2 * max_aukko + 2]
        pairs = []
        for di, d in enumerate(dets):
            for ti, tr in enumerate(free):
                px, py = ennuste(tr, k)
                dist = np.hypot(d[2] - px, d[3] - py)
                if dist <= portti_cm + portti_per_ruutu_cm * (k - tr[-1][1]):
                    pairs.append((dist, di, ti))
        pairs.sort()
        ud, ut = set(), set()
        for _, di, ti in pairs:
            if di in ud or ti in ut:
                continue
            ud.add(di)
            ut.add(ti)
            free[ti].append(dets[di])
        for di, d in enumerate(dets):
            if di not in ud:
                tracks.append([d])
    keep = []
    for tr in tracks:
        if len(tr) >= min_pisteet and np.hypot(tr[-1][2] - tr[0][2], tr[-1][3] - tr[0][3]) >= min_siirtyma_cm:
            keep.append(tr)
    return keep


def taydenna(radat_raaka, ruudut, cam, runko, cap, max_err=0.15, min_kelv=0.2, max_poikkeama_cm=12.0):
    """Aukkojen taytto: radan puuttuvissa yksilollisissa ruuduissa sovitetaan siluetti lineaarisesti interpoloidusta paikasta
    (ilman erillista tunnistusta, joka epaonnistuu kun kasi/vartalo peittaa kiven). Hyvaksytaan jos rms <= max_err, kelvollisia pikseleita
    vahintaan min_kelv ja paikka on enintaan max_poikkeama_cm interpoloidusta. radat_raaka: ketjuta() ennen ajan muunnosta: listat (frame, k, X, Y, err)."""
    k2f = {}
    for f, k in ruudut:
        k2f.setdefault(k, f)
    out = []
    for tr in radat_raaka:
        tr = sorted(tr, key=lambda d: d[1])
        uusi = list(tr)
        for a, b in zip(tr[:-1], tr[1:]):
            for k in range(a[1] + 1, b[1]):
                f = None
                # ikkunoiden rajoilla k ei ole yksikasitteinen: etsi ruutu jonka numero on a:n ja b:n valissa
                for ff, kk in ruudut:
                    if kk == k and a[0] < ff < b[0]:
                        f = ff
                        break
                if f is None:
                    continue
                u = (k - a[1]) / (b[1] - a[1])
                X0, Y0 = a[2] + u * (b[2] - a[2]), a[3] + u * (b[3] - a[3])
                fr = cam.ruutu(cap, f)
                alfa, ok = kt.havainto(cam, fr)
                s = kt.sovita(cam, runko, alfa, ok, X0, Y0)
                if s["rms"] <= max_err and s["kelvolliset"] >= min_kelv and np.hypot(s["X"] - X0, s["Y"] - Y0) <= max_poikkeama_cm:
                    uusi.append((f, k, s["X"], s["Y"], s["rms"]))
        out.append(sorted(uusi, key=lambda d: d[1]))
    return out


def vari_seuranta(cam, cap, ruudut, f_alku, uv_alku, vari, max_hukka=12, max_etaisyys_px=35.0):
    """Puhdas varisuuntaus ilman siluettia: kahvan vari -> kuvan sentroidi -> maa-asema (z = KAHVA_Z_CM). Seuraa ruutu
    ruudulta (yksilolliset ruudut) alkupisteesta: ennuste = edellinen sentroidi + edellinen siirtyma. Palauttaa
    [(frame, k, X, Y, ala_px)]. Loppuu kun kahva puuttuu max_hukka ruutua perakkain."""
    kdict = dict(ruudut)
    seq = [(f, kdict[f]) for f, _ in ruudut if f >= f_alku]
    out = []
    prev, vel, hukka = np.array(uv_alku, float), np.zeros(2), 0
    for f, k in seq:
        fr = cam.ruutu(cap, f)
        r = kv.kahva_sentroidi(cam, fr, prev + vel, vari=vari, max_etaisyys_px=max_etaisyys_px)
        if r is None:
            hukka += 1
            if hukka > max_hukka:
                break
            continue
        hukka = 0
        uv = np.array(r[:2])
        vel = uv - prev if out else np.zeros(2)
        prev = uv
        X, Y = kv.kuva_maahan(cam, uv[0], uv[1])
        out.append((f, k, X, Y, r[2]))
    return out


def valitse_vari(cam, cap, ruudut, f_alku, uv_alku, n=20):
    """Kahvan vari: kokeillaan keltaista ja punaista n ruudun matkalta, valitaan se jolla loytyy enemman havaintoja."""
    best = None
    for vari in ("keltainen", "punainen"):
        tr = vari_seuranta(cam, cap, [r for r in ruudut if r[0] >= f_alku][:n], f_alku, uv_alku, vari, max_hukka=n)
        if best is None or len(tr) > best[0]:
            best = (len(tr), vari)
    return best[1]


def aikaleimaa(tr):
    """(frame, k, X, Y, err) -> (t, X, Y, err), t = T_KARTTA[frame]: kaikilla radoilla sama aikajana (ikkunan alku + k / 25)."""
    return [(T_KARTTA[d[0]], d[2], d[3], d[4]) for d in tr]


def silota(tr, puoli_ikkuna=6, min_paino_err=0.1):
    """Radan tasoitus: paikallinen toisen asteen sovitus +-puoli_ikkuna havaintoa, painot 1 / (err + 0,1)^2 (huonosti sopivat
    ruudut -- kasi tai vartalo kiven paalla -- vaikuttavat vahemman). Palauttaa taulukon (t, X, Y, vX, vY, v) [cm, cm/s]."""
    a = np.array(tr, dtype=np.float64)
    t, X, Y, e = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    out = np.zeros((len(t), 6))
    for i in range(len(t)):
        j0, j1 = max(0, i - puoli_ikkuna), min(len(t), i + puoli_ikkuna + 1)
        w = 1.0 / (e[j0:j1] + min_paino_err) ** 2
        tt = t[j0:j1] - t[i]
        deg = 2 if (j1 - j0) >= 5 else 1
        cx = np.polyfit(tt, X[j0:j1], deg, w=np.sqrt(w))
        cy = np.polyfit(tt, Y[j0:j1], deg, w=np.sqrt(w))
        vx, vy = cx[-2], cy[-2]
        out[i] = (t[i], cx[-1], cy[-1], vx, vy, np.hypot(vx, vy))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--puoli", default="vasen")
    ap.add_argument("--ulos", default="tulokset")
    ap.add_argument("--ikkunat", default="8-22,56-66,104-114", help="sekuntivalit, esim. 8-22,28-35")
    ap.add_argument("--saikeet", type=int, default=4)
    ap.add_argument("--ketjuta-vain", action="store_true", help="ei uutta tunnistusta: ketjutetaan olemassa olevan kivet_<puoli>.csv:n havainnot")
    a = ap.parse_args()
    ikk = [tuple(float(x) for x in w.split("-")) for w in a.ikkunat.split(",")]
    ruudut = yksilolliset_ruudut(a.video, ikk, a.puoli)
    print(len(ruudut), "yksilollista ruutua")
    kmap = dict(ruudut)
    fps = 30.0
    rows = []
    path = os.path.join(a.ulos, f"kivet_{a.puoli}.csv")
    if a.ketjuta_vain:
        rows = [(int(r["frame"]), int(r["yksilo_k"]), float(r["X_cm"]), float(r["Y_cm"]), float(r["err"])) for r in csv.DictReader(open(path))]
    else:
        with Pool(a.saikeet, initializer=_alusta, initargs=(a.video, a.puoli, a.ulos)) as pool:
            res = pool.map(_tyo, [f for f, _ in ruudut], chunksize=4)
        with open(path, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["frame", "yksilo_k", "t_s", "X_cm", "Y_cm", "err", "peitto", "kelvolliset"])
            for i, dets in res:
                for X, Y, e, p, kel in dets:
                    wr.writerow([i, kmap[i], f"{i / fps:.3f}", f"{X:.3f}", f"{Y:.3f}", f"{e:.4f}", f"{p:.3f}", f"{kel:.3f}"])
                    rows.append((i, kmap[i], X, Y, e))
    radat = ketjuta(rows)
    cam = kv.Kamera(a.puoli, os.path.join(a.ulos, f"kalibrointi_{a.puoli}.json"), os.path.join(a.ulos, f"moodikuva_{a.puoli}.png"))
    radat = taydenna(radat, ruudut, cam, kt.Runko.lataa(a.ulos, a.puoli), cv2.VideoCapture(a.video))
    # todellinen aikaperusta (aikaperusta.py): jokaiselle radalle oma aikajana; tunnistamatta jaaneet kopioruudut pois
    aikajanat = [aikap.Aikajana([d[0] for d in tr], [d[3] for d in tr]) for tr in radat]
    radat = [[d for d in tr if int(d[0]) not in aj.kopio] for tr, aj in zip(radat, aikajanat)]
    # kahvan variseuranta (ei siluettia): siemen = radan ensimmainen piste jossa kivi on liikkeella (Y > -230)
    vari_radat = []
    for tr in radat:
        q = next((d for d in tr if d[3] > -230), tr[0])
        u, v = kv.kivimalli.project_3d(cam.K, cam.R, cam.t, np.array([[q[2], q[3], kv.KAHVA_Z_CM]]))
        vari = valitse_vari(cam, cv2.VideoCapture(a.video), ruudut, q[0], (u[0], v[0]))
        vari_radat.append((vari, vari_seuranta(cam, cv2.VideoCapture(a.video), ruudut, q[0], (u[0], v[0]), vari)))
    radat_f = radat
    radat = [aikaleimaa(tr) for tr in radat_f]
    path2 = os.path.join(a.ulos, f"radat_{a.puoli}.csv")
    with open(path2, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["rata", "t_s", "X_cm", "Y_cm", "err", "X_tasoitettu_cm", "Y_tasoitettu_cm", "vX_cm_s", "vY_cm_s", "v_cm_s", "ruutu", "t_kor_s"])
        for k, (tf, tr, aj) in enumerate(zip(radat_f, radat, aikajanat)):
            tk = aj.aika([d[0] for d in tf])
            sm = silota([(tkk, X, Y, e) for tkk, (_, X, Y, e) in zip(tk, tr)])
            for d, (t, X, Y, e), q, tkk in zip(tf, tr, sm, tk):
                wr.writerow([k, f"{t:.3f}", f"{X:.2f}", f"{Y:.2f}", f"{e:.3f}", f"{q[1]:.2f}", f"{q[2]:.2f}", f"{q[3]:.1f}", f"{q[4]:.1f}", f"{q[5]:.1f}", d[0], f"{tkk:.3f}"])
    path3 = os.path.join(a.ulos, f"kahvaradat_{a.puoli}.csv")
    with open(path3, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["rata", "vari", "t_s", "X_cm", "Y_cm", "kahvan_ala_px", "X_tasoitettu_cm", "Y_tasoitettu_cm", "vX_cm_s", "vY_cm_s", "v_cm_s", "ruutu", "t_kor_s"])
        for k, ((vari, tr), aj) in enumerate(zip(vari_radat, aikajanat)):
            tt = aikaleimaa(tr)
            tk = aj.aika([d[0] for d in tr])
            sm = silota([(tkk, X, Y, 0.2) for tkk, (_, X, Y, _) in zip(tk, tt)])
            for d, (t, X, Y, _), q, tkk in zip(tr, tt, sm, tk):
                wr.writerow([k, vari, f"{t:.3f}", f"{d[2]:.2f}", f"{d[3]:.2f}", d[4], f"{q[1]:.2f}", f"{q[2]:.2f}", f"{q[3]:.1f}", f"{q[4]:.1f}", f"{q[5]:.1f}", d[0], f"{tkk:.3f}"])
    print(path, len(rows), "havaintoa;", path2, len(radat), "rataa;", path3)
    for k, tr in enumerate(radat):
        print(f"  rata {k}: t {tr[0][0]:.2f}-{tr[-1][0]:.2f} s, {len(tr)} pistetta, Y {tr[0][2]:.0f} -> {tr[-1][2]:.0f} cm, X {tr[0][1]:.0f} -> {tr[-1][1]:.0f} cm")
