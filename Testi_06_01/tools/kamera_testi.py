"""Testi_06_01: kameran (Cam Link / UVC) ominaisuuksien kartoitus live-tilaa varten. Tulokset terminaaliin JA tiedostoon.

    python tools/kamera_testi.py              # kaikki laitteet 0..7
    python tools/kamera_testi.py 1            # vain laite 1
    python tools/kamera_testi.py 1 --sekunnit 20

Mittaa jokaiselle toimivalle yhdistelmalle (taustajarjestelma DirectShow/MSMF x koko x pakkausmuoto):
  * saatu koko, ilmoitettu ja MITATTU fps (lukusilmukka ilman piirtoa), read()-ajan keskiarvo ja maksimi,
  * pakkausmuoto (FOURCC), puuttuvat/viivastuneet ruudut (ruutujen valinen aika > 1,5 x odotettu),
  * pienennys 1280x720:een (INTER_AREA) ms/ruutu,
  * tallentaa yhden ruudun kuvaksi (kamera_<laite>_<taustaj>_<koko>_<fourcc>.png).
Lopuksi pitka mittaus (--sekunnit) parhaalla yhdistelmalla: tasaisuus pitkalla aikavalilla.
Tulos: kamera_testi_<aika>.txt nykyiseen kansioon - lahetä se.
"""
import argparse
import sys
import time

import cv2
import numpy as np

try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except Exception:
    pass

LINES = []


def out(s=""):
    print(s, flush=True)
    LINES.append(s)


def fourcc_text(v):
    v = int(v)
    t = "".join(chr((v >> (8 * i)) & 0xFF) for i in range(4))
    return t if t.isprintable() and v else "?"


def backends():
    if sys.platform.startswith("win"):
        return [(cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_MSMF, "MSMF")]
    if sys.platform.startswith("linux"):
        return [(cv2.CAP_V4L2, "V4L2")]
    return [(cv2.CAP_ANY, "ANY")]


def measure(cap, seconds):
    times, gaps = [], []
    t_prev = None
    t_end = time.time() + seconds
    first = None
    n = 0
    while time.time() < t_end:
        t0 = time.perf_counter()
        ok, f = cap.read()
        t1 = time.perf_counter()
        if not ok or f is None:
            break
        if first is None:
            first = f
        times.append(t1 - t0)
        if t_prev is not None:
            gaps.append(t1 - t_prev)
        t_prev = t1
        n += 1
    return first, n, np.array(times), np.array(gaps)


def test_combo(dev, be, bname, w, h, fourcc, seconds, save):
    cap = cv2.VideoCapture(dev, be)
    if not cap.isOpened():
        return None
    if fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    ok, f = cap.read()
    if not ok or f is None:
        cap.release()
        return None
    for _ in range(5):          # lammitys
        cap.read()
    first, n, times, gaps = measure(cap, seconds)
    if n < 5:
        cap.release()
        return None
    fps_rep = cap.get(cv2.CAP_PROP_FPS)
    fcc = fourcc_text(cap.get(cv2.CAP_PROP_FOURCC))
    meas = n / max(gaps.sum() + times[0], 1e-6) if len(gaps) else 0.0
    exp_gap = 1.0 / meas if meas > 0 else 0.0
    late = int(np.sum(gaps > 1.5 * exp_gap)) if len(gaps) else 0
    hh, ww = first.shape[:2]
    t = time.perf_counter()
    for _ in range(20):
        cv2.resize(first, (1280, 720), interpolation=cv2.INTER_AREA)
    rs = (time.perf_counter() - t) / 20 * 1000
    res = dict(laite=dev, taustaj=bname, pyydetty=f"{w}x{h}/{fourcc or 'oletus'}", saatu=f"{ww}x{hh}", fourcc=fcc,
               fps_ilm=round(fps_rep, 2), fps_mit=round(meas, 2), read_ka_ms=round(times.mean() * 1000, 2),
               read_max_ms=round(times.max() * 1000, 1), myohassa=late, ruutuja=n, pienennys_ms=round(rs, 2),
               kanavat=first.shape[2] if first.ndim == 3 else 1)
    if save:
        name = f"kamera_{dev}_{bname}_{ww}x{hh}_{fcc}.png"
        cv2.imwrite(name, first)
        res["kuva"] = name
    cap.release()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("laite", nargs="?", type=int, default=None)
    ap.add_argument("--sekunnit", type=float, default=15.0, help="pitkan mittauksen kesto parhaalla yhdistelmalla")
    ap.add_argument("--lyhyt", type=float, default=3.0, help="kunkin yhdistelman mittauksen kesto")
    args = ap.parse_args()
    out(f"KAMERATESTI {time.strftime('%Y-%m-%d %H:%M:%S')} | OpenCV {cv2.__version__} | {sys.platform}")
    devs = [args.laite] if args.laite is not None else list(range(8))
    sizes = [(1920, 1080), (1280, 720)]
    fourccs = [None, "MJPG", "YUY2", "NV12"]
    results = []
    for dev in devs:
        for be, bname in backends():
            cap = cv2.VideoCapture(dev, be)
            if not cap.isOpened():
                continue
            cap.release()
            out(f"\nLaite {dev} ({bname}) avautuu - kokeillaan koot x pakkausmuodot ...")
            for (w, h) in sizes:
                for fcc in fourccs:
                    r = test_combo(dev, be, bname, w, h, fcc, args.lyhyt, save=(fcc is None))
                    if r is None:
                        out(f"  {w}x{h} {fcc or 'oletus'}: ei toimi")
                        continue
                    results.append(r)
                    out("  " + ", ".join(f"{k}={v}" for k, v in r.items()))
    if not results:
        out("\nYhtaan toimivaa kameraa ei loytynyt.")
    else:
        full = [r for r in results if r["saatu"] == "1920x1080"] or results
        # tasapelissa (fps pyoristettyna) DirectShow (live-tilan oletus Windowsissa), sitten vahiten myohastyneita ruutuja
        best = max(full, key=lambda r: (round(float(r["fps_mit"])), r["taustaj"] == "DSHOW", -r["myohassa"], -float(r["read_ka_ms"])))
        out("\nHUOM: laitenumero riippuu taustajarjestelmasta (DSHOW ja MSMF numeroivat laitteet eri jarjestyksessa).")
        out(f"Paras 1920x1080-yhdistelma: " + ", ".join(f"{k}={v}" for k, v in best.items()))
        out(f"Pitka mittaus {args.sekunnit:.0f} s ...")
        be = dict((n, b) for b, n in backends())[best["taustaj"]]
        fcc = best["pyydetty"].split("/")[1]
        r = test_combo(best["laite"], be, best["taustaj"], 1920, 1080, None if fcc == "oletus" else fcc, args.sekunnit, save=False)
        if r:
            out("  " + ", ".join(f"{k}={v}" for k, v in r.items()))
            tj = "" if best["taustaj"] in ("DSHOW", "V4L2") else f" --live-taustaj {best['taustaj']}"
            out(f"\nSuositus: python main.py --live {best['laite']}{tj} --paneelit <referenssi>_panel_corners.txt --debug"
                f"   (kamera ~{r['fps_mit']:.0f} fps; live-tila kayttaa {'joka toista ruutua' if r['fps_mit'] > 37 else 'kaikkia ruutuja'} kun --live-fps 25)")
    name = f"kamera_testi_{time.strftime('%Y%m%d_%H%M%S')}.txt"
    with open(name, "w", encoding="utf-8") as fh:
        fh.write("\n".join(LINES) + "\n")
    print(f"\nTulokset tallennettu: {name}")


if __name__ == "__main__":
    main()
