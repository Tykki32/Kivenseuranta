"""HAKU-kokeiluymparisto (Testi_03_04): ajaa stone_tracker.search_new_stones:in tallennetuilla live-ruuduilla ERI PARAMETREILLA
ilman koko videon lapikaymista (yksi HAKU-kutsu ~30-40 ms).

Ruudut: TRACKER_FRAME_DUMP_DIR-tallenteet (g<frame>.npz = frame_u_for_tracking, eli TASMALLEEN se mita live-HAKU nakee: valotasapaino +
varjosietoinen taustanvaimennus, tausta valkoinen). Tallennus (kerran):

    TRACKER_FRAME_DUMP_DIR=/tmp/w/hakudump TRACKER_FRAME_DUMP_RANGES="2230:2290,2710:2770,3450:3510" \\
      python tools/run_profile.py <video> <calib_profile.pkl> <ulos> 3520

Kayttö moduulina:
    import sys; sys.path.insert(0, "tools")
    from haku_kokeilu import HakuLab
    lab = HakuLab("/tmp/w/hakudump", "/tmp/w/mah_base/calib_profile.pkl", "/tmp/w/full03/stones.csv")
    lab.table(range(2230, 2291, 10))                       # oletusparametrit, tulokset + oikea sijainti
    lab.table(range(2230, 2291, 10), accept=0.05)           # yksi parametri kerrallaan
    lab.sweep(ENTRIES, "accept", [0.05, 0.1, 0.2, 0.3])     # parametrihaku: osumat / vaarat ehdokkaat / ensimmainen loytoruutu
    lab.overlay(2260, "/tmp/haku_2260.png")                 # kuva: ehdokkaat (vihrea = lahella oikeaa, punainen = vaara), oikea (keltainen)

Parametrit (oletukset = live): x_half=70, y_min/y_max = FAR_HOGLINE-100 / +300, coarse=10, fine=2, score_thr=0.15 (esikarsinta),
accept=0.20 (C++ HAKU_ACCEPT_SCORE_THRESHOLD, ajonaikainen), max_results=4, max_attempts=8,
spawn=(rms_max 3.5, nb_min 14, nb_max 45, abs_x_max 65, score_max 0.9), diff_threshold=0.0 (ruutu on jo vaimennettu).
HUOM: live ajaa HAKUn vain joka 10. ruudulla - silti kokeilussa voi ajaa jokaisen ruudun.
"""
import os, sys, glob, pickle, types, math, csv, collections
for n in ("tkinter", "tkinter.filedialog"):
    if n not in sys.modules:
        sys.modules[n] = types.ModuleType(n)
sys.modules["tkinter"].filedialog = sys.modules["tkinter.filedialog"]
import numpy as np, cv2

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE)

# kivien saapumiskohdat: (kivi: (live-HAKU:n ensimmainen ehdokas-ruutu, tallennettu ruutuvali)). Live alkoi ruudulla 2227 (profiilin opettelun jalkeen).
#  17: HAKU loytaa kiven kerralla heti kun se tulee alueelle (Y<=3317 cm) -> "siisti" saapuminen
#   3: live-HAKU loysi 2240 (ehdokas 69), seuranta katkesi ("pysahtynyt" 2295) ja sama kivi rekisteroitiin uudelleen 2260/2270/2280 (ehdokkaat 70-72, osa "duplikaatti")
#  51: ensimmainen ehdokas 3430 (121), "duplikaatti" 3465, uusi ehdokas 3480 (122) - CSV:n ensimmainen rivi 3500
ENTRIES = {"17": (2760, range(2710, 2771)), "3": (2240, range(2227, 2291)), "51": (3430, range(3400, 3511))}


class HakuLab:
    DEFAULTS = dict(x_half=70.0, y_min=None, y_max=None, coarse=10.0, fine=2.0, score_thr=0.15, accept=0.20,
                    max_results=4, max_attempts=8, spawn=(3.5, 14, 45, 65.0, 0.9), diff_threshold=0.0)

    def __init__(self, dump_dir, pkl_path, stones_csv=None, match_cm=60.0):
        import main as M
        import stone_tracker
        self.M, self.st = M, stone_tracker
        pk = pickle.load(open(pkl_path, "rb"))
        prof, self.pose, calib = pk["profile"], pk["pose"], pk["calib"]
        k94, self.k92 = M.k94, M.k92
        self.R_max, self.H_total = prof["R_max_cm"], prof["H_total_cm"]
        self.body = k94.build_local_stone_rings(self.R_max, self.H_total, prof["shape_deltas"], n_theta=28, n_per_segment=5)
        self.search = k94.build_local_stone_rings(self.R_max, self.H_total, prof["shape_deltas"],
                                                  n_theta=self.k92.SEARCH_HULL_N_THETA, n_per_segment=self.k92.SEARCH_HULL_N_PER_SEGMENT)
        self.ring = prof["handle_r_frac"]
        self.handle = prof["handle_r_frac"]
        self.ref = calib["frame_undistorted"]
        self.dump_dir = dump_dir
        self.match_cm = match_cm
        self._cache = {}
        self.rows = collections.defaultdict(dict)             # kivi -> {frame: (x_cm, y_cm)}
        if stones_csv:
            for r in csv.DictReader(open(stones_csv)):
                try:
                    self.rows[r["stone_id"]][int(r["frame"])] = (float(r["x_m"]) * 100, float(r["y_m"]) * 100)
                except ValueError:
                    pass

    # ---- ruudut ----
    def frame(self, f):
        if f not in self._cache:
            if len(self._cache) > 40:
                self._cache.clear()
            self._cache[f] = np.load(os.path.join(self.dump_dir, f"g{f:06d}.npz"))["frame"]
        return self._cache[f]

    def frames_available(self):
        return sorted(int(os.path.basename(p)[1:7]) for p in glob.glob(os.path.join(self.dump_dir, "g*.npz")))

    # ---- oikea sijainti (stones.csv: vasta ensimmaisesta loydosta; sita ennen vakionopeus-ekstrapolointi) ----
    def truth(self, f):
        out = []
        for sid, d in self.rows.items():
            fs = sorted(d)
            if not fs:
                continue
            if f in d:
                out.append((sid, d[f][0], d[f][1], False))
            elif f < fs[0] and fs[0] - f <= 80:
                n = [k for k in fs if k <= fs[0] + 15]
                if len(n) >= 4:
                    vx = (d[n[-1]][0] - d[n[0]][0]) / (n[-1] - n[0]); vy = (d[n[-1]][1] - d[n[0]][1]) / (n[-1] - n[0])
                    out.append((sid, d[fs[0]][0] + vx * (f - fs[0]), d[fs[0]][1] + vy * (f - fs[0]), True))
        return out

    # ---- HAKU ----
    def run(self, f, frame=None, **over):
        """frame: halutessa oma (esim. eri tavalla vaimennettu) ruutu tallennetun sijaan."""
        p = dict(self.DEFAULTS); p.update(over)
        k92 = self.k92
        y_min = p["y_min"] if p["y_min"] is not None else k92.SEARCH_Y_MIN_CM
        y_max = p["y_max"] if p["y_max"] is not None else k92.SEARCH_Y_MAX_CM
        self.st.set_spawn_filter(*p["spawn"])
        if hasattr(self.st, "set_haku_accept_score"):
            self.st.set_haku_accept_score(float(p["accept"]))
        res = self.st.search_new_stones(
            self.frame(f) if frame is None else frame, self.ref, self.body, self.search, self.pose["K"], self.pose["R"], self.pose["t"],
            0.0, float(p["x_half"]), (y_min + y_max) / 2.0, (y_max - y_min) / 2.0, float(p["coarse"]), float(p["fine"]),
            float(p["score_thr"]), self.R_max, self.H_total, self.ring, self.handle, float(p["diff_threshold"]),
            int(p["max_results"]), int(p["max_attempts"]))
        return [r for r in res if r.get("found")]

    def is_match(self, d, ex):
        """Osuma = lahella oikeaa kivea (ekstrapoloitu oikea sijainti on epatarkempi -> kaksinkertainen raja)."""
        return d is not None and d <= (2 * self.match_cm if ex else self.match_cm)

    def classify(self, f, res):
        """Jokaiselle ehdokkaalle lahin oikea kivi (tai None) ja etaisyys."""
        tr = self.truth(f)
        out = []
        for r in res:
            best = min(((math.hypot(r["X_cm"] - x, r["Y_cm"] - y), sid, ex) for sid, x, y, ex in tr), default=(None, None, None))
            out.append((r, best[0], best[1], best[2]))
        return out

    def table(self, frames, **over):
        print(f"{'ruutu':>6} {'X':>7} {'Y':>8} {'score':>6} {'rms':>6} {'n_body':>6}  lahin oikea (etaisyys cm)")
        for f in frames:
            res = self.run(f, **over)
            if not res:
                print(f"{f:6d}  -")
                continue
            for r, d, sid, ex in self.classify(f, res):
                lab = "ei tiedossa" if d is None else f"kivi {sid}{' (ekstrapol.)' if ex else ''}: {d:6.1f}" + ("  <- VAARA" if not self.is_match(d, ex) else "")
                print(f"{f:6d} {r['X_cm']:7.1f} {r['Y_cm']:8.1f} {r.get('score', float('nan')):6.3f} {r.get('rms_px', float('nan')):6.2f} {r.get('n_body', 0):6d}  {lab}")

    def sweep(self, entries, name, values, frames_step=1, **over):
        """Yhden parametrin pyyhkaisy kivien saapumiskohdissa. Tulostaa: ensimmainen loytoruutu kivelle, osumat, vaarat ehdokkaat."""
        print(f"{name:>10}  " + "  ".join(f"kivi {k}: 1.loyto/osumat" for k in entries) + "   vaaria")
        for v in values:
            cols = []; bad = 0
            for sid, (f0, frs) in entries.items():
                first = None; hits = 0
                for f in list(frs)[::frames_step]:
                    res = self.run(f, **{**over, name: v})
                    for r, d, s2, ex in self.classify(f, res):
                        if self.is_match(d, ex) and s2 == sid:
                            hits += 1
                            if first is None:
                                first = f
                        else:
                            bad += 1
                cols.append(f"{('%d' % first) if first else '-':>6}/{hits:<3d} (live {f0})")
            print(f"{str(v):>10}  " + "  ".join(cols) + f"   {bad}")

    def overlay(self, f, path, **over):
        res = self.run(f, **over)
        img = self.frame(f).copy()
        K, R, t = (np.asarray(self.pose[k], float) for k in ("K", "R", "t")); R = R.reshape(3, 3); t = t.reshape(3)
        def proj(x, y, z=0.0):
            p = K @ (R @ np.array([x, y, z]) + t); return int(round(p[0] / p[2])), int(round(p[1] / p[2]))
        for sid, x, y, ex in self.truth(f):
            cv2.circle(img, proj(x, y, self.H_total / 2), 14, (0, 215, 255), 1); cv2.putText(img, f"{sid}{'*' if ex else ''}", (proj(x, y, self.H_total / 2)[0] + 16, proj(x, y, self.H_total / 2)[1]), 0, 0.5, (0, 215, 255), 1)
        for r, d, sid, ex in self.classify(f, res):
            col = (0, 200, 0) if self.is_match(d, ex) else (0, 0, 255)
            u, v = proj(r["X_cm"], r["Y_cm"], self.H_total / 2); cv2.circle(img, (u, v), 10, col, 2)
            cv2.putText(img, f"{r.get('score', 0):.2f}", (u + 12, v + 14), 0, 0.45, col, 1)
        ys = [self.k92.SEARCH_Y_MIN_CM, self.k92.SEARCH_Y_MAX_CM]
        for y in ys:
            cv2.line(img, proj(-70, y), proj(70, y), (255, 128, 0), 1)
        x0, y0 = proj(0, (ys[0] + ys[1]) / 2)
        crop = img[max(0, y0 - 120):y0 + 120, max(0, x0 - 200):x0 + 200]
        cv2.imwrite(path, cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST))
        return res


if __name__ == "__main__":
    lab = HakuLab(sys.argv[1] if len(sys.argv) > 1 else "/tmp/w/hakudump", sys.argv[2] if len(sys.argv) > 2 else "/tmp/w/mah_base/calib_profile.pkl",
                  sys.argv[3] if len(sys.argv) > 3 else "/tmp/w/full03/stones.csv")
    for sid, (f0, frs) in ENTRIES.items():
        print(f"\n=== kivi {sid}: ensimmainen live-loyto ruudulla {f0} ===")
        lab.table([f for f in frs if f % 5 == 0 and lab.frames_available().count(f)])
