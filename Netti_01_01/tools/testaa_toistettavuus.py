"""Toistettavuustesti: sama videopatka ajetaan simulaationa (--nopea) ja toistettavana reaaliaikarakenteena (--ei-tahdistusta, tyoprosessit, ei ruutujen ohitusta); CSV-rivien pitaa olla
tismalleen samat. Kayta lyhyelle patkalle (esim. yksi heitto + kaukopaan seuranta, 60 s) tiedostolla (ei YouTube-virta: virrassa ei ole kiintea ruutujono).

    python tools/testaa_toistettavuus.py video.mp4 --alku 4825 --loppu 4885 [--toistoja 2]
"""
import argparse
import csv
import os
import subprocess
import sys
import tempfile

KANSIO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def aja(video, alku, loppu, csv_polku, lisa):
    komento = [sys.executable, os.path.join(KANSIO, "netti_live.py"), "--lahde", video, "--alku", str(alku), "--loppu", str(loppu), "--ei-ikkunaa", "--csv", csv_polku, "--hiljaa"] + lisa
    r = subprocess.run(komento, capture_output=True, text=True, cwd=KANSIO)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:])
        raise SystemExit(f"ajo epaonnistui: {' '.join(lisa)}")
    with open(csv_polku, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--alku", type=float, default=0.0)
    ap.add_argument("--loppu", type=float, default=60.0)
    ap.add_argument("--toistoja", type=int, default=2, help="toistettavien ajojen maara (kukin verrataan simulaatioon)")
    a = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        ref = aja(a.video, a.alku, a.loppu, os.path.join(tmp, "sim.csv"), ["--nopea"])
        print(f"simulaatio: {len(ref)} heittoa")
        kaikki_ok = True
        for i in range(a.toistoja):
            rows = aja(a.video, a.alku, a.loppu, os.path.join(tmp, f"rt{i}.csv"), ["--ei-tahdistusta"])
            ero = []
            if len(rows) != len(ref):
                ero.append(f"heittojen maara {len(rows)} vs {len(ref)}")
            for n, (r1, r2) in enumerate(zip(ref, rows), 1):
                ero += [f"heitto {n}, {c}: {r1[c]} vs {r2.get(c)}" for c in r1 if r1[c] != r2.get(c)]
            print(f"toistettava ajo {i + 1}: " + ("kaikki sarakkeet samat" if not ero else f"{len(ero)} eroa"))
            for e in ero[:20]:
                print("   ", e)
            kaikki_ok &= not ero
        raise SystemExit(0 if kaikki_ok else 1)


if __name__ == "__main__":
    main()
