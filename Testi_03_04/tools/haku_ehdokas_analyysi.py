"""HAKU-ehdokkaiden ominaisuudet ja spawn-suodattimien arviointi.
Kaytto: python3 tools/haku_ehdokas_analyysi.py <ajokansio> <ajologi>
Lukee lokista '[frame N] Uusi kivi-ehdokas ...[ehdokasominaisuudet score=.. rms=.. n_body=.. tarkka=..]' -rivit,
merkitsee 'oikeaksi' ehdokkaat joiden id on lopullisessa heittoportin lapaisseessa CSV:ssa, ja laskee
kuinka monta roskaehdokasta erilaiset spawn-hetken saannot hylkaisivat (ja kuinka monta oikeaa kaatuisi)."""
import re, sys, csv, numpy as np
d, logf = sys.argv[1], sys.argv[2]
fin = set(int(r['stone_id']) for r in csv.DictReader(open(f"{d}/stones.csv")))
g = lambda x: None if x == 'None' else float(x)
rows = []; fate = {}
for l in open(logf):
    m = re.match(r"\[frame (\d+)\] Uusi kivi-ehdokas (\d+): \((-?[\d.]+), (-?[\d.]+)\) cm .*?score=([\d.eE+-]+|None) rms=([\d.eE+-]+|None) n_body=(\d+|None) n_ring=(\d+|None) tarkka=(\w+)", l)
    if m:
        rows.append(dict(id=int(m.group(2)), frame=int(m.group(1)), x=float(m.group(3)), y=float(m.group(4)),
                         score=g(m.group(5)), rms=g(m.group(6)), nb=g(m.group(7)), tk=m.group(9) == 'True'))
        continue
    m = re.match(r"\[frame (\d+)\] (?:Ehdokas|Kivi|Rata) (\d+) (hylatty|kadotettu|pysahtynyt|yhdistetty)", l)
    if m: fate[int(m.group(2))] = int(m.group(1))
for r in rows: r['real'] = r['id'] in fin; r['dur'] = fate.get(r['id'], 10**9) - r['frame']
real = [r for r in rows if r['real']]; junk = [r for r in rows if not r['real']]
print(f"ehdokkaita {len(rows)} (oikeita {len(real)}, roskaa {len(junk)})")
ok = lambda r: r['rms'] is not None and r['rms'] <= 3.5 and 14 <= r['nb'] <= 45 and abs(r['x']) <= 65 and r['score'] <= 0.9
rules = {"rms<=3.5": lambda r: r['rms'] is not None and r['rms'] <= 3.5, "14<=n_body<=45": lambda r: 14 <= r['nb'] <= 45,
         "|x|<=65": lambda r: abs(r['x']) <= 65, "score<=0.9": lambda r: r['score'] <= 0.9, "kaikki yhdessa": ok}
for name, f in rules.items():
    kj = sum(not f(r) for r in junk); kr = sum(not f(r) for r in real)
    print(f"{name:18s} hylkaa roskaa {kj:4d}/{len(junk)} ({100*kj/len(junk):4.1f}%), oikeita hylatty {kr}/{len(real)}")
