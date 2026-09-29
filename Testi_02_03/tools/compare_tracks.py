"""Vertaa kahden koko putken ajon kivien sijainti-CSV:ta (_kivien_sijainnit.csv / stones.csv).

Kaytto: python compare_tracks.py A.csv B.csv [nimiA nimiB]

- yhteenveto: rivit, eri kivet (stone_id), ratojen pituus, tarkka-osuus, rms
- rivien kohdistus framen mukaan: jokaiselle A:n riville lahin B:n rivi samalta
  framelta (etaisyys cm) - kertoo ovatko radat samat vaikka stone_id:t eroavat
"""
import sys, csv, collections
import numpy as np


def load(path):
    rows = list(csv.DictReader(open(path)))
    out = collections.defaultdict(list)
    for r in rows:
        out[int(r["frame"])].append((int(r["stone_id"]), float(r["x_m"]) * 100, float(r["y_m"]) * 100,
                                     int(r["tarkka"]), float(r["rms_px"]) if r["rms_px"] else np.nan))
    return rows, out


def summary(name, rows):
    ids = collections.Counter(int(r["stone_id"]) for r in rows)
    tk = np.mean([int(r["tarkka"]) for r in rows])
    rms = np.array([float(r["rms_px"]) if r["rms_px"] else np.nan for r in rows])
    lens = np.array(sorted(ids.values()))
    print(f"{name}: rivit {len(rows)}, eri kivet {len(ids)}, rata-pituus (rivia) med {np.median(lens):.0f} "
          f"p90 {np.percentile(lens, 90):.0f} max {lens.max()}, tarkka-osuus {tk:.3f}, rms med {np.nanmedian(rms):.2f} px")


def main():
    a_path, b_path = sys.argv[1], sys.argv[2]
    na = sys.argv[3] if len(sys.argv) > 3 else "A"
    nb = sys.argv[4] if len(sys.argv) > 4 else "B"
    ra, A = load(a_path)
    rb, B = load(b_path)
    summary(na, ra)
    summary(nb, rb)

    def match(P, Q):
        d, unmatched = [], 0
        for f, prow in P.items():
            for (_, x, y, *_r) in prow:
                cands = Q.get(f, [])
                if not cands:
                    unmatched += 1
                    continue
                dd = min(np.hypot(x - cx, y - cy) for (_, cx, cy, *_q) in cands)
                d.append(dd)
        return np.array(d), unmatched

    dab, ua = match(A, B)
    dba, ub = match(B, A)
    print(f"\nkohdistus framen mukaan (lahin rivi samalla framella):")
    print(f"  {na}->{nb}: kohdistui {len(dab)} riviä, ilman paria {ua}; etaisyys cm: mediaani {np.median(dab):.2f} "
          f"p90 {np.percentile(dab, 90):.2f} p99 {np.percentile(dab, 99):.2f}")
    print(f"  {nb}->{na}: kohdistui {len(dba)} riviä, ilman paria {ub}")
    close = dab < 30.0
    print(f"  {na}-rivit joilla {nb}:ssa pari <30 cm paassa: {close.mean():.3f}")


if __name__ == "__main__":
    main()
