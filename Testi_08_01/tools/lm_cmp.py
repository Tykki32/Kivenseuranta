import pickle, sys, numpy as np
a = pickle.load(open(sys.argv[1], "rb")); b = pickle.load(open(sys.argv[2], "rb"))
d = []; nf = 0; tot = 0; ra = []; rb = []; out = []
for fa, fb in zip(a, b):
    for x, y in zip(fa, fb):
        tot += 1
        if x[0] != y[0]: nf += 1; continue
        if x[0]:
            dd = np.hypot(x[1] - y[1], x[2] - y[2]); d.append(dd)
            if x[3] is not None and y[3] is not None:
                ra.append(x[3]); rb.append(y[3])
                if dd > 0.3: out.append((dd, x[3], y[3]))
d = np.array(d)
print(f"kivia {tot}, found-ero {nf}, pos-ero cm: p95 {np.percentile(d,95):.4f} p99 {np.percentile(d,99):.3f} max {d.max():.2f}, >0.3cm: {(d>0.3).sum()}; rms ka base {np.mean(ra):.3f} uusi {np.mean(rb):.3f}")
for o in out[:12]: print(f"   poikkeama {o[0]:.2f} cm rms base {o[1]:.2f} -> uusi {o[2]:.2f}")
