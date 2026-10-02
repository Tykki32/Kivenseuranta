"""Testi_06_01 v6.11: nopeustesti samalla videolla eri asetuksilla - tulokset TIEDOSTOON.

    python tools/nopeustesti.py --video D:\\Tikku\\Live\\live_20261002_163845_live.mp4 --paneelit D:\\Tikku\\Suorita\\MAH00014_leikattu_panel_corners.txt
    python tools/nopeustesti.py --video ... --paneelit ... --ajot v610,ohitus --tilat tiedosto
    python tools/nopeustesti.py --video ... --paneelit ... --max-frame 6000          # nopea kokeilu (ensimmaiset 6000 ruutua)

Tilat (--tilat, oletus molemmat):
  tiedosto : tavallinen tiedostoajo (debug-video paalla). Deterministinen -> heitot ja hog-hog-ajat verrataan v610-ajoon.
  live     : --live-sim reaaliaikatahdissa + --live-tallenna (kuten kamera-ajo: ruudut tulevat 25 r/s, tallennus paalla).
             Mittaa pysyyko kasittely tahdissa: paasaikeen palveluaika, live-viive ja pudotetut ruudut.
Ajot (--ajot, oletus v610,kiipea,prio):
  v610   : vanha laskenta (OVERLAP_FAST=0, FINE_CLIMB=0) = vertailukohta
  peitto : nopea peittolaskenta (OVERLAP_FAST=1, FINE_CLIMB=0) - tuloksen pitaa olla IDENTTINEN v610:n kanssa
  kiipea : peitto + ristikkohaun hieno vaihe maennousulla (FINE_CLIMB=1) = v6.11:n oletus
  prio   : kiipea + LIVE_BG_PRIORITY=1 (tallennuksen/debug-videon ffmpeg ja tallennussaie alemmalle prioriteetille)
  varjo  : peitto + FINE_CLIMB=2 (hieno vaihe ajetaan molemmilla tavoilla, tulos = v610; lokiin kuinka usein maennousu osuu samaan)
  ohitus : kiipea + GRID_SKIP=1 (mean-shiftien perusteella ohitettu ristikkohaku - EI kaytossa, ks. README)
Tulokset: --outdir (oletus: videon kansio/nopeustesti_<aika>): ajo_<nimi>_<tila>_loki.txt, CSV:t ja nopeustesti_yhteenveto.txt
(lahetä yhteenveto minulle).
"""
import argparse
import csv
import datetime
import os
import re
import shutil
import subprocess
import sys

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RUNS = [
    ("v610", {"OVERLAP_FAST": "0", "FINE_CLIMB": "0", "GRID_SKIP": "0"}),
    ("peitto", {"OVERLAP_FAST": "1", "FINE_CLIMB": "0", "GRID_SKIP": "0"}),
    ("kiipea", {"OVERLAP_FAST": "1", "FINE_CLIMB": "1", "GRID_SKIP": "0"}),
    ("prio", {"OVERLAP_FAST": "1", "FINE_CLIMB": "1", "GRID_SKIP": "0", "LIVE_BG_PRIORITY": "1"}),
    ("varjo", {"OVERLAP_FAST": "1", "FINE_CLIMB": "2", "GRID_SKIP": "0"}),
    ("ohitus", {"OVERLAP_FAST": "1", "FINE_CLIMB": "1", "GRID_SKIP": "1"}),
]
CLEAR_ENV = ("OVERLAP_FAST", "FINE_CLIMB", "GRID_SKIP", "GRID_SKIP_SCORE", "GRID_SKIP_AGREE_CM", "GRID_SKIP_DUMP", "LIVE_BG_PRIORITY", "BG_PRIORITY")


def run(cmd, env_extra, log_path):
    env = dict(os.environ)
    for k in CLEAR_ENV:
        env.pop(k, None)
    env.update(env_extra)
    env["PYTHONUNBUFFERED"] = "1"
    print(f"\n===== {os.path.basename(log_path)}: {' '.join(cmd)}  env={env_extra} =====", flush=True)
    t0 = datetime.datetime.now()
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        log.write(f"# {' '.join(cmd)}\n# env={env_extra}\n")
        p = subprocess.Popen(cmd, cwd=CODE_DIR, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, encoding="utf-8", errors="replace")
        for line in p.stdout:
            log.write(line)
            sys.stdout.write(line)
            sys.stdout.flush()
        rc = p.wait()
    return rc, (datetime.datetime.now() - t0).total_seconds()


def numbers(text):
    d = {}
    m = re.search(r"Koko ajo: ([\d.]+)s / (\d+) ruutua \(([\d.]+) r/s", text)
    if m:
        d["aika_s"], d["ruutuja"], d["r/s"] = float(m.group(1)), int(m.group(2)), float(m.group(3))
    for st in ("A", "B", "C"):
        m = re.search(r"^%s\s+.*?\s([\d.]+)\s+([\d.]+)\s+[\d.]+\s+[\d.]+\s+(\d+)\s*$" % st, text, re.M)
        if m:
            d[st + "_ms"] = float(m.group(1))
    m = re.search(r"SEURANTA: (\d+) kutsua, yhteensa ([\d.]+)s, ka ([\d.]+) ms/kutsu", text)
    if m:
        d["seur_ms/kutsu"] = float(m.group(3))
    m = re.search(r"Ristikkohaun ohitus: (\d+) / (\d+)", text)
    if m:
        d["ohitus_%"] = 100.0 * int(m.group(1)) / max(1, int(m.group(2)))
    m = re.search(r"haku: ristikko \(yhdistelma[^)]*\)\s+([\d.]+)", text)
    if m:
        d["ristikko_ms"] = float(m.group(1))
    m = re.search(r"seuranta/kivi-paivitys yhteensa \(CPU\)\s+([\d.]+)", text)
    if m:
        d["seur_cpu_ms"] = float(m.group(1))
    m = re.search(r"-> (\d+) heittoa \(yksi", text)
    if m:
        d["heitot"] = int(m.group(1))
    m = re.search(r"pudotettu \(puskuri taynna\) (\d+), suurin viive \d+ ruutua \(([\d.]+) s\)", text)
    if m:
        d["pudotettu"], d["max_viive_s"] = int(m.group(1)), float(m.group(2))
    viive = re.findall(r"live-viive ([\d.]+) s", text)
    if viive:
        d["viive_lopussa_s"] = float(viive[-1])
    return d


def tracking_lag(path):
    """Live-aikaleimoista: suurin ja keskimaarainen viive SEURANNAN aikana (hypyn jalkeen = kun viive on ensin pudonnut alle 5 s)."""
    if not path or not os.path.exists(path):
        return None
    vals = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                vals.append(float(r["viive_s"]))
            except (KeyError, ValueError):
                pass
    start = next((i for i, v in enumerate(vals) if v < 5.0 and i > len(vals) // 20), None)
    if start is None:
        return {"seuranta_max_viive_s": max(vals) if vals else None}
    tail = vals[start:]
    return {"seuranta_max_viive_s": max(tail), "seuranta_ka_viive_s": sum(tail) / len(tail)}


def load_hog(path):
    if not path or not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        return [r for r in csv.DictReader(f)]


def compare_hog(base, other):
    """Heitot paritetaan kaukohoglinen ylitysajan mukaan (kivien numerot voivat vaihtua)."""
    if not base or not other:
        return "ei vertailua (hog-CSV puuttuu)"
    lines = []
    used = set()
    worst = {"hog_hog_s": 0.0, "v_far_hog_ms": 0.0, "x_far_hog_cm": 0.0, "dir_far_hog_deg": 0.0}
    missing = 0
    for b in base:
        tb = float(b["t_far_hog_s"])
        cands = [(abs(float(o["t_far_hog_s"]) - tb), i) for i, o in enumerate(other) if i not in used]
        if not cands or min(cands)[0] > 1.0:
            missing += 1
            lines.append(f"    heitto t={tb:7.2f} s: PUUTTUU")
            continue
        _, i = min(cands)
        used.add(i)
        o = other[i]
        for k in worst:
            try:
                worst[k] = max(worst[k], abs(float(o[k]) - float(b[k])))
            except (KeyError, ValueError):
                pass
    extra = len(other) - len(used)
    head = (f"heittoja {len(base)} vs {len(other)} (puuttuu {missing}, ylimaaraisia {extra}); suurin ero: "
            f"hog-hog {worst['hog_hog_s']:.3f} s, nopeus {worst['v_far_hog_ms']:.3f} m/s, "
            f"x kaukohogilla {worst['x_far_hog_cm']:.2f} cm, suunta {worst['dir_far_hog_deg']:.2f} deg")
    return "\n".join([head] + lines)


def same_file(a, b):
    if not (a and b and os.path.exists(a) and os.path.exists(b)):
        return None
    with open(a, "rb") as fa, open(b, "rb") as fb:
        return fa.read() == fb.read()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--paneelit", required=True, help="referenssi-paneelitiedosto (*_panel_corners.txt)")
    ap.add_argument("--ajot", default="v610,kiipea,prio")
    ap.add_argument("--tilat", default="tiedosto,live")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--max-frame", type=int, default=0, help="lopeta tahan ruutuun (0 = koko video), molemmissa tiloissa")
    ap.add_argument("--no-debug", action="store_true", help="tiedostoajo ilman debug-videota")
    args = ap.parse_args()

    video = os.path.abspath(args.video)
    if not os.path.exists(video):
        sys.exit(f"Videota ei loydy: {video}")
    vdir = os.path.dirname(video)
    stem = os.path.splitext(os.path.basename(video))[0]
    outdir = args.outdir or os.path.join(vdir, "nopeustesti_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(outdir, exist_ok=True)
    wanted = [r.strip() for r in args.ajot.split(",") if r.strip()]
    modes = [m.strip() for m in args.tilat.split(",") if m.strip()]
    main_py = os.path.join(CODE_DIR, "main.py")

    res = {}
    try:
        for mode in modes:
            for name, env in RUNS:
                if name not in wanted:
                    continue
                tag = f"{name}_{mode}"
                log = os.path.join(outdir, f"ajo_{tag}_loki.txt")
                env = dict(env)
                if mode == "tiedosto":
                    cmd = [sys.executable, main_py, "--video", video, "--paneelit", args.paneelit,
                           "--no-debug" if args.no_debug else "--debug"]
                    if args.max_frame:
                        cmd += ["--max-frame", str(args.max_frame)]
                    rc, dt = run(cmd, env, log)
                    base = os.path.join(vdir, f"{stem}_leikattu_kivien_sijainnit")
                    for suf in (".csv", "_hog.csv", "_raaka.csv"):
                        if os.path.exists(base + suf):
                            shutil.copy2(base + suf, os.path.join(outdir, f"ajo_{tag}_sijainnit{suf}"))
                    lag_csv = None
                else:
                    nimi = f"ajo_{tag}"
                    cmd = [sys.executable, main_py, "--live-sim", video, "--paneelit", args.paneelit, "--debug",
                           "--live-tallenna", "--live-kansio", outdir, "--live-nimi", nimi]
                    if args.max_frame:
                        cmd += ["--max-frame", str(args.max_frame)]
                    rc, dt = run(cmd, env, log)
                    for suf in (".csv", "_hog.csv", "_raaka.csv"):
                        src = os.path.join(outdir, f"{nimi}_kivien_sijainnit{suf}")
                        if os.path.exists(src):
                            os.replace(src, os.path.join(outdir, f"ajo_{tag}_sijainnit{suf}"))
                    lag_csv = os.path.join(outdir, f"{nimi}_kivien_sijainnit_live_aikaleimat.csv")
                with open(log, encoding="utf-8", errors="replace") as f:
                    text = f.read()
                d = numbers(text)
                lag = tracking_lag(lag_csv)
                if lag:
                    d.update(lag)
                res[tag] = {"rc": rc, "wall": dt, "d": d, "text": text}
    except KeyboardInterrupt:
        print("\nKeskeytetty - kootaan yhteenveto valmiista ajoista.")

    out = [f"NOPEUSTESTI {datetime.datetime.now():%Y-%m-%d %H:%M}  video={video}  max-frame={args.max_frame or 'koko'}", ""]
    cols = ["r/s", "A_ms", "B_ms", "C_ms", "seur_ms/kutsu", "seur_cpu_ms", "ristikko_ms", "heitot", "pudotettu", "max_viive_s",
            "seuranta_max_viive_s", "seuranta_ka_viive_s", "viive_lopussa_s"]
    out.append("  ".join(f"{h:>14s}" for h in ["ajo", "paluu"] + cols))
    for tag, r in res.items():
        vals = [tag, str(r["rc"])] + [
            (f"{r['d'][c]:.2f}" if isinstance(r["d"].get(c), float) else str(r["d"].get(c, "-"))) for c in cols]
        out.append("  ".join(f"{v:>14s}" for v in vals))
    out.append("(C_ms = paasaikeen palveluaika / ruutu: 25 r/s vaatii < 40 ms. seur_cpu_ms / ristikko_ms = seurannan / ristikkohaun CPU-aika ms/ruutu.")
    out.append(" Live: seuranta_max_viive_s / seuranta_ka_viive_s = suurin / keskimaarainen viive seurannan aikana (hypyn jalkeen))")
    for tag, r in res.items():
        m = re.search(r"Hieno ristikko maennousulla: .*", r.get("text", ""))
        if m:
            out.append(f"{tag}: {m.group(0)}")
    out.append("")
    out.append("== TULOSTEN VERTAILU (vertailukohta v610 samassa tilassa) ==")
    for mode in modes:
        ref = f"v610_{mode}"
        for name, _ in RUNS:
            tag = f"{name}_{mode}"
            if tag == ref or tag not in res or ref not in res:
                continue
            ident = same_file(os.path.join(outdir, f"ajo_{ref}_sijainnit.csv"), os.path.join(outdir, f"ajo_{tag}_sijainnit.csv"))
            note = "" if mode == "tiedosto" else " (live: hyppy riippuu ajoituksesta -> pieniä eroja odotettavissa)"
            out.append(f"{tag} vs {ref}{note}: sijainti-CSV {'IDENTTINEN' if ident else 'eroaa' if ident is not None else 'puuttuu'}")
            out.append("  hog: " + compare_hog(load_hog(os.path.join(outdir, f"ajo_{ref}_sijainnit_hog.csv")),
                                               load_hog(os.path.join(outdir, f"ajo_{tag}_sijainnit_hog.csv"))))
    summary = "\n".join(out)
    path = os.path.join(outdir, "nopeustesti_yhteenveto.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(summary + "\n")
    print("\n" + summary)
    print(f"\nYhteenveto: {path} - lahetä se minulle.")


if __name__ == "__main__":
    main()
