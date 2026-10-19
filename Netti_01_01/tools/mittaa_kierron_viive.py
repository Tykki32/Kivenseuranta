"""Mittaa, kuinka nopeasti kierteen tulokset tulevat kayttoon reaaliaikaisessa ajossa (diagnostiikka: kierre ei ehdi livenakymaan, jos taustaprosessit eivat jaksa):

    python tools/mittaa_kierron_viive.py --youtube TUNNUS --ulos testi.mp4 [muut netti_live.py:n valitsimet]
    python tools/mittaa_kierron_viive.py --lahde video.mp4 --alku 0 --loppu 500 --ei-ikkunaa

Ajaa netti_live.py:n (samat valitsimet) ja tulostaa lopuksi heitoittain viiveen (seinakellon sekunteja) siita, kun analyysi saavutti heiton viimeisen ruudun, siihen, kun heittopaan kierre, kaukopaan kierre
ja kiven vakiot tulivat kayttoon. Heittopaan kierteen pitaa tulla kayttoon muutamassa sekunnissa (paneelin "Kierre" ilmestyy silloin; kuva tulee viiveella `--viive`, oletus 10 s); jos viive kasvaa
heitosta toiseen, kierron laskenta ei jaksa (kone on liian hidas tai prosessoria vie jokin muu). Kaukopaan kierre ja vakiot tarkentavat tulosta jalkikateen, ja niille viive on pidempi."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    import numpy as np
    import netti_live
    import netti_seuranta as ns

    sys.argv = ["netti_live.py"] + sys.argv[1:]
    t0 = time.time()
    tulokset, naytteet = [], []
    alku_syota = ns.Analyysi.syota

    def syota(self, f, ruutu):
        r = alku_syota(self, f, ruutu)
        if f % 15 == 0:
            naytteet.append((time.time() - t0, f))
        return r

    def teko(laji, orig):
        def paikattu(self, h, r):
            tulokset.append((laji, h.tunnus, h.f_alku, h.f_loppu, time.time() - t0))
            return orig(self, h, r)
        return paikattu

    ns.Analyysi.syota = syota
    ns.Analyysi._aseta_kierto = teko("lahi", ns.Analyysi._aseta_kierto)
    ns.Analyysi._aseta_kauko_kierto = teko("kauko", ns.Analyysi._aseta_kauko_kierto)
    ns.Analyysi._aseta_vakiot = teko("vakiot", ns.Analyysi._aseta_vakiot)
    try:
        netti_live.main()
    finally:
        if naytteet and tulokset:
            w = np.array([x[0] for x in naytteet])
            fr = np.array([x[1] for x in naytteet])
            rivit = {}
            for laji, tunnus, f_alku, f_loppu, wall in tulokset:
                if f_loppu is None:
                    continue
                rivit.setdefault((f_alku, tunnus), dict(loppu=f_loppu))[laji] = wall - float(np.interp(f_loppu, fr, w))
            print("\nKierron tulosten viive (s seinakellolla heiton viimeisen ruudun analysoinnista):")
            print("  heitto  alku s  loppu s | heittopaa  kaukopaa  vakiot")
            ft = lambda x: "       -" if x is None else f"{x:8.1f}"
            for (f_alku, tunnus), r in sorted(rivit.items(), key=lambda kv: (kv[0][0] or 0)):
                print(f"  {tunnus:6d} {f_alku / 30.0:7.1f} {r['loppu'] / 30.0:8.1f} | {ft(r.get('lahi'))}  {ft(r.get('kauko'))}  {ft(r.get('vakiot'))}")
            lahi = [r["lahi"] for r in rivit.values() if "lahi" in r]
            if lahi:
                print(f"  heittopaan kierre: mediaani {np.median(lahi):.1f} s, suurin {max(lahi):.1f} s ({len(lahi)} heittoa)")


if __name__ == "__main__":
    main()
