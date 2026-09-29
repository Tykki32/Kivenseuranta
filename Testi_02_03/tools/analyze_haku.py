import sys, csv, numpy as np, collections
sys.argv=['x']
sys.path.insert(0,'/home/user/Kivenseuranta/Testi_02_03/tools')
from compare_throws import load_tracks, is_real, overlap
runs = {'grid': '/tmp/w/hk_grid', 'ms': '/tmp/w/hk_ms'}
T = {n: load_tracks(p+'/stones.csv') for n, p in runs.items()}
H = {n: list(csv.DictReader(open(p+'/haku.csv'))) for n, p in runs.items()}
# toistettavuus edelliseen
for n, old in (('grid','/tmp/w/cmp2/stones.csv'),('ms','/tmp/w/e2e_ms/stones.csv')):
    a = sum(len(v) for v in T[n].values()); b = sum(len(v) for v in load_tracks(old).values())
    print(f'{n}: rivit {a} (edellinen ajo {b})')
# unioni todellisista heitoista (ruudut >= 2990 ja ei-degeneroitunut)
throws = []
for n, tr in T.items():
    for k, v in tr.items():
        if is_real(v) and (v[-1][2] < v[0][2] - 500 or v[0][2] < v[-1][2] - 500) and v[0][0] >= 2900:
            if not any(overlap(v, w) >= 0.5 and overlap(w, v) >= 0.5 for _, _, w in throws):
                throws.append((n, k, v))
throws.sort(key=lambda x: x[2][0][0])
print('todellisia heittoja (liike >5 m, ruudut>=2900):', len(throws))
def real_ids(n): return {k for k, v in T[n].items() if is_real(v)}
RI = {n: real_ids(n) for n in runs}
out = {}
for n in runs:
    print(f'\n=== {n} ===')
    covered = 0
    for src, k, v in throws:
        cov = max((overlap(v, w) for w in T[n].values()), default=0)
        ok = cov >= 0.5
        covered += ok
        if ok: continue
        # HAKU-tapahtumat heiton alkuvaiheessa: kandidaatti lahella heiton alkuasemaa
        f0 = v[0][0]
        ev = []
        for r in H[n]:
            f = int(r['frame'])
            if f0 - 40 <= f <= f0 + 60 and r['found'] == '1':
                x, y = float(r['X']), float(r['Y'])
                # heiton paikka framella f (interpoloi/ekstrapoloi alkupiste)
                fr = {q[0]: q for q in v}
                ref = fr.get(f) or (v[0] if f < v[0][0] else v[min(len(v)-1, f - v[0][0])])
                if np.hypot(x - ref[1], y - ref[2]) < 120:
                    ev.append((f, x, y, r['registered'], r['blocker_id'], r['blocker_dist_cm'], r['blocker_confirmed'], np.hypot(x - ref[1], y - ref[2])))
        reasons = collections.Counter('rekist.' if e[3]=='1' else 'estetty' for e in ev)
        print(f' MISS f{v[0][0]}-{v[-1][0]} ({src}:{k}), HAKU-osumia lähellä {len(ev)} {dict(reasons)}')
        for e in ev[:6]:
            print(f'     f{e[0]} pos=({e[1]:.0f},{e[2]:.0f}) reg={e[3]} blocker={e[4]} d={e[5]} conf={e[6]}  (etäisyys heiton polusta {e[7]:.0f})')
        if not ev: print('     (HAKU ei löytänyt heittoa lainkaan alkuvaiheessa)')
    print(f' katettu {covered}/{len(throws)}')

print('\n\n=== ESTOT: kuka esti ja kuinka kaukaa (vain reg=0, blocker vahvistettu) ===')
for n in runs:
    ids_real = RI[n]
    d_real, d_junk = [], []
    for r in H[n]:
        if r['found'] == '1' and r['registered'] == '0' and r['blocker_confirmed'] == '1':
            b = int(r['blocker_id']); d = float(r['blocker_dist_cm'])
            (d_real if b in ids_real else d_junk).append(d)
    d_real, d_junk = np.array(d_real), np.array(d_junk)
    print(f'{n}: estoja yhteensä {len(d_real)+len(d_junk)}; esto aidolla heittoradalla (oikea kaksoisosuma) n={len(d_real)}, '
          f'roskaradalla n={len(d_junk)}')
    if len(d_real):
        print('   aito rata: d cm  min %.1f med %.1f p90 %.1f p95 %.1f max %.1f' % (d_real.min(), np.median(d_real), np.percentile(d_real,90), np.percentile(d_real,95), d_real.max()))
        for lim in (15, 20, 25, 30, 40, 50, 66, 100):
            print(f'     aitoja kaksoisosumia alle {lim:3d} cm: {np.mean(d_real < lim):.3f}   roskaradan estoja alle {lim:3d} cm: {(d_junk < lim).sum():4d}/{len(d_junk)}')

print('\n\n=== ESTAVAN RADAN LIIKE (siirtyma viimeisen 8 framen aikana, cm) ===')
def disp(track, frame, span=8):
    rows = {q[0]: q for q in track}
    now = rows.get(frame) or next((q for q in reversed(track) if q[0] <= frame), None)
    past = next((q for q in reversed(track) if q[0] <= frame - span), None)
    if now is None or past is None: return None
    return float(np.hypot(now[1]-past[1], now[2]-past[2]))
for n in runs:
    real_d, junk_d = [], []
    for r in H[n]:
        if r['found'] == '1' and r['registered'] == '0' and r['blocker_confirmed'] == '1':
            b = int(r['blocker_id']); tr = T[n].get(b)
            if not tr: continue
            d = disp(tr, int(r['frame']))
            if d is None: continue
            (real_d if b in RI[n] else junk_d).append(d)
    real_d, junk_d = np.array(real_d), np.array(junk_d)
    print(f'{n}: aito rata (n={len(real_d)}): liike med {np.median(real_d):.0f} min {real_d.min():.0f} p10 {np.percentile(real_d,10):.0f}  | roskarata (n={len(junk_d)}): med {np.median(junk_d):.0f} p90 {np.percentile(junk_d,90):.0f} max {junk_d.max():.0f}')
    for lim in (20, 30, 40, 60, 80):
        print(f'    kynnys {lim:3d} cm/8fr: aidot ylittää {np.mean(real_d>=lim):.2f}, roskaradoista ylittää {np.mean(junk_d>=lim):.2f}  ({(junk_d>=lim).sum()}/{len(junk_d)})')
