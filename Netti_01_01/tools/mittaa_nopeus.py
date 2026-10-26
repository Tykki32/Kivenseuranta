"""Analyysin nopeus tällä koneella: riittääkö se reaaliaikaan? Syöttää videon ruudut analyysille (netti_seuranta.Analyysi, samat työprosessit kuin netti_live.py) ilman piirtoa,
tallennusta ja ikkunaa niin nopeasti kuin ehtii, ja kertoo ruudun ajan (seinäkello ja analyysisäikeen CPU-aika) vaiheittain sekä sen, kuinka suuri osa reaaliajasta analyysi vie.

    python tools/mittaa_nopeus.py VIDEO [--kansio tulokset_nettivideo3] [--alku 300] [--loppu 360] [--saikeet 3]

Valitse jakso, jolla on heitto (esim. alkaa 25 s ennen heiton lähtöä ja kestää 60 s): raskas osa on heiton aikana. Tulos on yläraja analyysin osuudelle: varsinaisessa ajossa (netti_live.py)
samalla koneella ovat lisäksi videon dekoodaus, piirto, viivepuskurin pakkaus, tallennus (ffmpeg) ja ikkuna. Jos "ruudun aika keskimäärin" on yli 33 ms (30 r/s), analyysi ei ehdi reaaliaikaan ja
kuvan viive kasvaa; silloin kokeile suurempaa --saikeet-arvoa, varmista että tarkka sovitus on C++ (`tarkka sovitus: C++ (netti_fit)`), ja sulje muut raskaat ohjelmat."""
import argparse
import os
import platform
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import netti_seuranta as ns  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("video")
    ap.add_argument("--kansio", default="tulokset", help="kalibrointikansio (tulokset, tulokset_nettivideo2, tulokset_nettivideo3)")
    ap.add_argument("--alku", type=float, default=0.0, help="alku (s)")
    ap.add_argument("--loppu", type=float, default=None, help="loppu (s); oletus alku + 60")
    ap.add_argument("--saikeet", type=int, default=max(1, min(3, (os.cpu_count() or 2) - 1)), help="tarkan sovituksen työprosessit (kuten netti_live.py)")
    a = ap.parse_args()
    loppu = a.loppu if a.loppu is not None else a.alku + 60.0
    kansio = a.kansio if os.path.isdir(a.kansio) else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", a.kansio)
    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        raise SystemExit(f"Videota {a.video} ei voitu avata")
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = float(fps) if fps and 1.0 < fps < 120.0 else 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(a.alku * fps))
    print(f"Python {platform.python_version()} ({platform.system()}), OpenCV {cv2.__version__}, ytimiä {os.cpu_count()}, ruutunopeus {fps:.1f} r/s")
    an = ns.Analyysi(kansio, fps=fps, saikeet=a.saikeet, tarkka=a.saikeet > 0, synkroninen=False, fps_lahde=None, kierre=True)
    n_prosessia = an.lampea()
    sovitin = next(iter(an.M.values())).sovitin
    print(f"{n_prosessia} työprosessia (--saikeet {a.saikeet}); tarkka sovitus: {sovitin}" + ("" if "C++" in sovitin else f" ({ns.netti_fit_selitys()})"))
    an.aseta_toistettava(True)                                   # kuten tiedosto / VOD netti_live.py:ssä: ruutuja ei ohiteta, B-tulokset odotetaan
    an.fps = fps
    ajat, cpu = [], []
    n = int((loppu - a.alku) * fps)
    f0 = int(a.alku * fps)
    t0 = time.perf_counter()
    for i in range(n):
        ok, ruutu = cap.read()
        if not ok:
            break
        t, c = time.perf_counter(), time.thread_time()
        an.syota(f0 + i, ruutu)
        ajat.append(1e3 * (time.perf_counter() - t))
        cpu.append(1e3 * (time.thread_time() - c))
    an.viimeistele()
    kesto = time.perf_counter() - t0
    an.sulje()
    if not ajat:
        raise SystemExit("Ei ruutuja")
    ajat, cpu = np.array(ajat), np.array(cpu)
    tavoite = 1e3 / fps
    print(f"\n{len(ajat)} ruutua ({len(ajat) / fps:.0f} s videota), {an.kopioita} kopioruutua, heittoja {len(an.heitot)}")
    print(f"ruudun aika keskimäärin {ajat.mean():.1f} ms (analyysisäikeen CPU {cpu.mean():.1f} ms), p95 {np.percentile(ajat, 95):.1f} ms, suurin {ajat.max():.0f} ms; reaaliaikaan riittää alle {tavoite:.0f} ms")
    print("vaiheet (seinäkello, ms/ruutu): " + ", ".join(f"{k} {1e3 * v / len(ajat):.1f}" for k, v in sorted(an.aika.items(), key=lambda kv: -kv[1])))
    osuus = 100.0 * ajat.sum() / 1e3 / (len(ajat) / fps)
    print(f"analyysi vei {osuus:.0f} % videon kestosta (koko ajo {kesto:.0f} s, mukana videon luku)")
    if ajat.mean() > tavoite:
        print("-> analyysi EI ehdi reaaliaikaan tällä koneella: kuvan viive kasvaa. Kokeile --saikeet-arvoa, C++-sovitinta (netti_fit), --puskuri bgr tai sulje muita ohjelmia.")
    elif osuus > 60:
        print("-> analyysi ehtii niukasti: dekoodaus, piirto, tallennus ja ikkuna tarvitsevat lisäksi prosessoria, joten viive voi kasvaa raskailla jaksoilla.")
    else:
        print("-> analyysi ehtii reaaliaikaan selvästi.")


if __name__ == "__main__":
    main()
