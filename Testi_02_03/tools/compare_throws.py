"""Luokittelee kivien radat (stone_id) 'todellisiksi heitoiksi' ja vertaa kahta ajoa.

Todellinen heitto: rata jossa >= MIN_ROWS riviä, Y-matka >= MIN_TRAVEL_CM,
tarkka-osuus >= MIN_TARKKA ja rms_px:n mediaani <= MAX_RMS.
Kaksi rataa vastaa toisiaan jos >= 50 % A:n rivien frameista on B:ssa <30 cm paassa.

Kaytto: python compare_throws.py A.csv B.csv [nimiA nimiB]
"""
import sys, csv, collections
import numpy as np

MIN_ROWS, MIN_TRAVEL_CM, MIN_TARKKA, MAX_RMS = 20, 100.0, 0.5, 3.0


def load_tracks(path):
    tr = collections.defaultdict(list)
    for r in csv.DictReader(open(path)):
        tr[int(r["stone_id"])].append((int(r["frame"]), float(r["x_m"]) * 100, float(r["y_m"]) * 100,
                                       int(r["tarkka"]), float(r["rms_px"]) if r["rms_px"] else np.nan))
    return tr


def is_real(rows):
    if len(rows) < MIN_ROWS:
        return False
    ys = np.array([r[2] for r in rows])
    return (ys.max() - ys.min() >= MIN_TRAVEL_CM and np.mean([r[3] for r in rows]) >= MIN_TARKKA
            and np.nanmedian([r[4] for r in rows]) <= MAX_RMS)


def overlap(ta, tb):
    fb = {f: (x, y) for f, x, y, *_ in tb}
    hit = tot = 0
    for f, x, y, *_ in ta:
        tot += 1
        if f in fb and np.hypot(x - fb[f][0], y - fb[f][1]) < 30.0:
            hit += 1
    return hit / max(1, tot)


def main():
    A, B = load_tracks(sys.argv[1]), load_tracks(sys.argv[2])
    na = sys.argv[3] if len(sys.argv) > 3 else "A"
    nb = sys.argv[4] if len(sys.argv) > 4 else "B"
    RA = {k: v for k, v in A.items() if is_real(v)}
    RB = {k: v for k, v in B.items() if is_real(v)}
    print(f"{na}: {len(A)} rataa, todellisia heittoja {len(RA)}   |   {nb}: {len(B)} rataa, todellisia heittoja {len(RB)}")

    def pairs(P, Q):
        res = {}
        for k, v in P.items():
            best = max(((overlap(v, w), j) for j, w in Q.items()), default=(0, None))
            res[k] = best
        return res

    pa = pairs(RA, B)   # A:n todellinen heitto -> paras vastine B:n KAIKISTA radoista
    pb = pairs(RB, A)
    miss_a = [k for k, (o, j) in pa.items() if o < 0.5]
    miss_b = [k for k, (o, j) in pb.items() if o < 0.5]
    print(f"\n{na}:n todellisista heitoista {len(RA) - len(miss_a)}/{len(RA)} loytyy {nb}:sta (>=50 % framesta <30 cm)")
    print(f"{nb}:n todellisista heitoista {len(RB) - len(miss_b)}/{len(RB)} loytyy {na}:sta")
    for name, miss, P, other in ((na, miss_a, RA, nb), (nb, miss_b, RB, na)):
        if miss:
            print(f"\n{name}:n todelliset heitot joita {other} EI loyda:")
            for k in miss:
                v = P[k]
                print(f"  id {k}: frame {v[0][0]}-{v[-1][0]}, {len(v)} riviä, Y {v[0][2]:.0f}->{v[-1][2]:.0f} cm, "
                      f"rms med {np.nanmedian([r[4] for r in v]):.2f}")
    # todellisten heittojen sijaintiero (yhteisilla frameilla)
    d = []
    for k, (o, j) in pa.items():
        if o >= 0.5:
            fb = {f: (x, y) for f, x, y, *_ in B[j]}
            d += [np.hypot(x - fb[f][0], y - fb[f][1]) for f, x, y, *_ in RA[k] if f in fb]
    if d:
        d = np.array(d)
        print(f"\nyhteisten todellisten heittojen sijaintiero: n={len(d)}, mediaani {np.median(d):.2f} cm, p90 {np.percentile(d, 90):.2f}, p99 {np.percentile(d, 99):.2f}")


if __name__ == "__main__":
    main()
