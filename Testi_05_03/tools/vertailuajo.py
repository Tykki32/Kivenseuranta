"""GPU-ristikkohaun vertailuajo: ajaa main.py:n kolmesti peräkkäin ja kokoaa raportit TIEDOSTOON (terminaali ei riitä).

  python tools/vertailuajo.py                       # valitsee videon tiedostovalitsimella
  python tools/vertailuajo.py --video D:\\Tikku\\Suorita\\MAH00014.MP4
  python tools/vertailuajo.py --video ... --start 00:10:00 --end 00:21:00 --outdir D:\\Tikku\\Suorita\\vertailu

Ajot (kaikki samalla videolla/aikavälillä, oletuksena debug-video päällä), valitaan --runs:
  cpu    : kaikki CPU:lla (vertailu)            gpu : SEURANNAN ristikkohaku GPU:lla (GPU_GRID=1)
  gpub   : vaihe B GPU:lla (GPU_B=1)            gpuall : molemmat GPU:lla
  verify : GPU_GRID=1 + GPU_GRID_VERIFY=1 + GPU_B=1 (oikeellisuus, hidas)
  par    : GPU_B + STAB_WORKERS=2 + PIPE_DEPTH=6     par3 : kuten par, STAB_WORKERS=3     parcpu : STAB_WORKERS=2 + PIPE_DEPTH=6 (vaihe B CPU:lla)
  --outdir <aiempi kansio>: jos siina on ajo_cpu_loki.txt + ajo_cpu_sijainnit.csv, cpu-ajoa ei tarvitse ajaa uudelleen (--runs par,intra,intragrid) - aiempi mukaan yhteenvetoon ja CSV-vertailuun
  intra  : par + INTRA_PARALLEL=1 (kiven sisäinen rinnakkaisuus)     intragrid : intra + GPU_GRID=1
  v5.7:  blasmt / intrablasmt : cpu / intra vanhalla BLAS-asetuksella (OpenBLAS 8 säiettä, kuten <= v5.6) -> BLAS-korjauksen vaikutus
         intraprio : intra + BG_PRIORITY=1 (taustasäikeet alemmalle prioriteetille)    intrasw : intra + PY_SWITCH_INTERVAL_MS=1
  v5.8:  intrayuv : intra + DEBUG_YUV=1 (debug-video ffmpegille valmiina YUV:na; vain qsv/x264-putki)
Oletus: cpu,par,intra,intragrid.
Tulokset kansioon --outdir (oletus: videon kansio / vertailuajo_<aikaleima>):
  ajo_<nimi>_loki.txt          koko terminaalitulostus
  ajo_<nimi>_sijainnit.csv     kivien sijainnit (+ _hog.csv)
  vertailu_yhteenveto.txt      pullonkaularaportit, GPU-tilasto, nopeudet ja CSV-vertailu (tulostetaan myös lopuksi)
"""
import argparse
import csv
import datetime
import hashlib
import os
import re
import shutil
import subprocess
import sys

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RUNS = [
    ("cpu", {}),                                                          # vertailu: kaikki CPU:lla
    ("gpu", {"GPU_GRID": "1"}),                                           # SEURANNAN ristikkohaku GPU:lla
    ("gpub", {"GPU_B": "1"}),                                             # vaihe B (warp+remap+varjosuodatus) GPU:lla
    ("gpuall", {"GPU_GRID": "1", "GPU_B": "1"}),                          # molemmat GPU:lla
    ("verify", {"GPU_GRID": "1", "GPU_GRID_VERIFY": "1", "GPU_B": "1"}),  # oikeellisuus (hidas): ristikko verifioidaan CPU:hun
    ("par", {"GPU_B": "1", "STAB_WORKERS": "2", "PIPE_DEPTH": "6"}),      # vaihe B GPU:lla + stabilointi 2 ruudulle rinnan + jonot 6
    ("par3", {"GPU_B": "1", "STAB_WORKERS": "3", "PIPE_DEPTH": "6"}),     # kuten par, 3 rinnakkaista stabilointia
    ("parcpu", {"STAB_WORKERS": "2", "PIPE_DEPTH": "6"}),                 # vain liukuhihnan rinnakkaisuus (vaihe B CPU:lla)
    ("intra", {"GPU_B": "1", "STAB_WORKERS": "2", "PIPE_DEPTH": "6", "INTRA_PARALLEL": "1"}),                    # par + kiven sisainen rinnakkaisuus (ristikko + 2 mean-shiftia rinnan)
    ("intragrid", {"GPU_B": "1", "STAB_WORKERS": "2", "PIPE_DEPTH": "6", "INTRA_PARALLEL": "1", "GPU_GRID": "1"}),  # intra + ristikkohaun hieno vaihe GPU:lla
    # v5.7: vertailu vanhaan BLAS-kaytokseen (NumPyn OpenBLAS monisaikeisena, kuten <= v5.6) - nayttaa BLAS-korjauksen vaikutuksen tallä koneella
    ("blasmt", {"OPENBLAS_NUM_THREADS": "8", "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8"}),
    ("intrablasmt", {"GPU_B": "1", "STAB_WORKERS": "2", "PIPE_DEPTH": "6", "INTRA_PARALLEL": "1",
                     "OPENBLAS_NUM_THREADS": "8", "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8"}),
    # v5.7: valinnaiset ajoitussaadot (eivat muuta tuloksia): taustasaikeet alemmalle prioriteetille / lyhyempi GIL-vaihtovali
    ("intraprio", {"GPU_B": "1", "STAB_WORKERS": "2", "PIPE_DEPTH": "6", "INTRA_PARALLEL": "1", "BG_PRIORITY": "1"}),
    ("intrasw", {"GPU_B": "1", "STAB_WORKERS": "2", "PIPE_DEPTH": "6", "INTRA_PARALLEL": "1", "PY_SWITCH_INTERVAL_MS": "1"}),
    # v5.8: debug-video ffmpegille valmiina YUV 4:2:0:na (BGR-muunnos cv2:lla, ei ffmpegin swscalella) - koskee ffmpeg-putkea (qsv/x264)
    ("intrayuv", {"GPU_B": "1", "STAB_WORKERS": "2", "PIPE_DEPTH": "6", "INTRA_PARALLEL": "1", "DEBUG_YUV": "1"}),
]
DEFAULT_RUNS = "cpu,par,intra,intragrid"


def pick_video():
    try:
        import tkinter
        import tkinter.filedialog
        tkinter.Tk().withdraw()
        return tkinter.filedialog.askopenfilename(
            title="Valitse (alkuperäinen) video",
            filetypes=[("Videot", "*.mts *.MTS *.mp4 *.MP4 *.mov *.MOV *.avi *.AVI"), ("Kaikki", "*.*")])
    except Exception:
        return ""


def run_main(name, env_extra, video, start, end, debug, extra, log_path):
    env = dict(os.environ)
    for k in ("GPU_GRID", "GPU_GRID_VERIFY", "GPU_B", "STAB_WORKERS", "PIPE_DEPTH", "INTRA_PARALLEL",
              "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "BLAS_THREADS", "BG_PRIORITY", "PY_SWITCH_INTERVAL_MS", "DEBUG_YUV"):
        if k not in env_extra:
            env.pop(k, None)
    env.update(env_extra)
    env["PYTHONUNBUFFERED"] = "1"
    cmd = [sys.executable, os.path.join(CODE_DIR, "main.py"), "--video", video]
    if debug:
        cmd.append("-d")
    if start:
        cmd += ["--start", start]
    if end:
        cmd += ["--end", end]
    cmd += extra
    print(f"\n===== AJO '{name}': {' '.join(cmd)}  env={env_extra} =====", flush=True)
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
    dt = (datetime.datetime.now() - t0).total_seconds()
    print(f"===== AJO '{name}' valmis (paluukoodi {rc}, {dt:.0f} s) =====", flush=True)
    return rc, dt


def read_text(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def extract_blocks(text):
    """Poimii lokista pullonkaula- ja nopeusraportit sekä muut avainrivit."""
    lines = text.replace("\r", "\n").split("\n")
    out = []
    keys = ("GPU-ristikkohaku", "GPU-vaihe B", "Debug-video:", "Heittoportti", "Hog-hog -analyysi", "Versio:")
    seen = set()
    for ln in lines:
        if ln.startswith(keys) and ln not in seen:
            seen.add(ln)
            out.append(ln)
    for header, end_marker in (("=== NOPEUSSEURANTA", "=========="), ("=== PULLONKAULA-ANALYYSI", "==========")):
        i = None
        for k, ln in enumerate(lines):
            if ln.startswith(header):
                i = k
        if i is None:
            continue
        blk = [lines[i]]
        for ln in lines[i + 1:]:
            blk.append(ln)
            if ln.startswith(end_marker):
                break
        out.append("")
        out += blk
    last_progress = [ln for ln in lines if ln.startswith("Ruutu ") and "nopeus:" in ln]
    if last_progress:
        out.append("")
        out.append("Viimeinen etenemisrivi: " + last_progress[-1])
    return "\n".join(out)


def summarize_numbers(text):
    d = {}
    m = re.search(r"Koko ajo: ([\d.]+)s / (\d+) ruutua \(([\d.]+) r/s", text)
    if m:
        d["aika_s"], d["ruutuja"], d["rs"] = float(m.group(1)), int(m.group(2)), float(m.group(3))
    for st in ("A", "B", "C"):
        m = re.search(r"^%s\s+.*?\s([\d.]+)\s+([\d.]+)\s+[\d.]+\s+[\d.]+\s+(\d+)\s*$" % st, text, re.M)
        if m:
            d["ms_" + st] = float(m.group(1))
    m = re.search(r"SEURANTA: (\d+) kutsua, yhteensa ([\d.]+)s, ka ([\d.]+) ms/kutsu .*?, ([\d.]+) ms/ruutu", text)
    if m:
        d["seuranta_ms_ruutu"] = float(m.group(4))
    m = re.search(r"ristikko: GPU-polku \(hieno ristikko, yhteensa\)\s+([\d.]+)\s+([\d.]+)", text)
    if m:
        d["gpu_ristikko_ms_ruutu"], d["gpu_ristikko_ms_kutsu"] = float(m.group(1)), float(m.group(2))
    m = re.search(r"haku: ristikko \(yhdistelma[^)]*\)\s+([\d.]+)\s+([\d.]+)", text)
    if m:
        d["cpu_ristikko_ms_ruutu"], d["cpu_ristikko_ms_kutsu"] = float(m.group(1)), float(m.group(2))
    m = re.search(r"Heittoportti: (\d+) vahvistettua rataa -> (\d+) heittomaista rataa -> (\d+) heittoa", text)
    if m:
        d["heitot"] = int(m.group(3))
    m = re.search(r"Hog-hog -analyysi: (\d+) heittoa", text)
    if m:
        d["hog"] = int(m.group(1))
    return d


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compare_csv(a, b):
    """Palauttaa tekstin: identtinen vai mitkä rivit eroavat."""
    if not (os.path.exists(a) and os.path.exists(b)):
        return "puuttuva tiedosto"
    if sha(a) == sha(b):
        return "IDENTTINEN (tavutasolla)"

    def load(p):
        rows = {}
        with open(p, newline="", encoding="utf-8", errors="replace") as f:
            for r in csv.DictReader(f):
                rows[(r.get("frame"), r.get("stone_id"))] = r
        return rows
    ra, rb = load(a), load(b)
    only_a = len(set(ra) - set(rb))
    only_b = len(set(rb) - set(ra))
    n_diff, mx_x, mx_y = 0, 0.0, 0.0
    for k in set(ra) & set(rb):
        try:
            dx = abs(float(ra[k]["x_m"]) - float(rb[k]["x_m"])) * 100
            dy = abs(float(ra[k]["y_m"]) - float(rb[k]["y_m"])) * 100
        except Exception:
            continue
        if dx > 1e-9 or dy > 1e-9:
            n_diff += 1
            mx_x, mx_y = max(mx_x, dx), max(mx_y, dy)
    return (f"EROAA: rivejä {len(ra)} vs {len(rb)}, vain ensimmäisessä {only_a}, vain toisessa {only_b}, "
            f"eri sijainti {n_diff} riviä (max ero X {mx_x:.3f} cm, Y {mx_y:.3f} cm)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", default=None, help="alkuperäinen videotiedosto (ei valitsinta)")
    ap.add_argument("--start", default="00:10:00")
    ap.add_argument("--end", default="00:21:00")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--no-debug", action="store_true", help="aja ilman debug-videota")
    ap.add_argument("--runs", default=DEFAULT_RUNS, help="ajettavat ajot pilkuilla (cpu,gpu,gpub,gpuall,verify,par,par3,parcpu,intra,intragrid,blasmt,intrablasmt,intraprio,intrasw,intrayuv); oletus " + DEFAULT_RUNS)
    ap.add_argument("--main-args", default="", help="lisäargumentit main.py:lle lainausmerkeissä, esim. \"--max-frame 3000\"")
    args = ap.parse_args()

    video = args.video or pick_video()
    if not video or not os.path.exists(video):
        sys.exit("Videota ei annettu / ei löydy.")
    stem = os.path.splitext(os.path.basename(video))[0]
    vdir = os.path.dirname(os.path.abspath(video))
    outdir = args.outdir or os.path.join(vdir, "vertailuajo_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(outdir, exist_ok=True)
    csv_src = os.path.join(vdir, f"{stem}_leikattu_kivien_sijainnit.csv")
    hog_src = os.path.join(vdir, f"{stem}_leikattu_kivien_sijainnit_hog.csv")
    wanted = [r.strip() for r in args.runs.split(",") if r.strip()]
    extra = args.main_args.split() if args.main_args else []

    results = {}
    try:
        for name, env_extra in RUNS:
            if name not in wanted:
                continue
            log_path = os.path.join(outdir, f"ajo_{name}_loki.txt")
            rc, dt = run_main(name, env_extra, video, args.start, args.end, not args.no_debug, extra, log_path)
            text = read_text(log_path)
            if os.path.exists(csv_src):
                shutil.copy2(csv_src, os.path.join(outdir, f"ajo_{name}_sijainnit.csv"))
            if os.path.exists(hog_src):
                shutil.copy2(hog_src, os.path.join(outdir, f"ajo_{name}_sijainnit_hog.csv"))
            results[name] = {"rc": rc, "wall_s": dt, "text": text, "nums": summarize_numbers(text)}
    except KeyboardInterrupt:
        print("\nKeskeytetty - kootaan yhteenveto valmistuneista ajoista.")

    # Jos cpu-ajoa ei ajettu tassa mutta --outdir:ssa on aiemman cpu-ajon loki + CSV, otetaan se mukaan vertailuun (sama kansio = sama video/aikavali)
    prev_log = os.path.join(outdir, "ajo_cpu_loki.txt")
    if "cpu" not in results and os.path.exists(prev_log) and os.path.exists(os.path.join(outdir, "ajo_cpu_sijainnit.csv")):
        text = read_text(prev_log)
        results = {"cpu": {"rc": 0, "wall_s": 0.0, "text": text, "nums": summarize_numbers(text), "earlier": True}, **results}
        print("Mukaan otettu aiempi cpu-ajo kansiosta", outdir)

    out = []
    out.append(f"VERTAILUAJO {datetime.datetime.now():%Y-%m-%d %H:%M}  video={video}  {args.start}..{args.end}  debug={'ei' if args.no_debug else 'kyllä'}")
    out.append("")
    out.append("== YHTEENVETO ==")
    cols = ["aika_s", "rs", "heitot", "hog", "ms_A", "ms_B", "ms_C", "seuranta_ms_ruutu", "gpu_ristikko_ms_kutsu", "cpu_ristikko_ms_kutsu"]
    head = ["ajo", "paluukoodi"] + cols
    out.append("  ".join(f"{h:>14s}" for h in head))
    for name, r in results.items():
        row = [name, str(r["rc"])] + [(f"{r['nums'][c]:.2f}" if isinstance(r["nums"].get(c), float) else str(r["nums"].get(c, "-"))) for c in cols]
        out.append("  ".join(f"{v:>14s}" for v in row))
    out.append("(rs = ruutua/s; ms_A/B/C = pullonkaularaportin vaiheiden palveluaika; ristikko-ms/kutsu: GPU-polku vs. CPU-ristikkohaku)")
    out.append("")
    out.append("== CSV-VERTAILU ==")
    names = [n for n, _ in RUNS if n in results]
    pairs = [(a, "cpu") for a in names if a != "cpu"] if "cpu" in names else [(names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))]
    for a, b in pairs:
        pa = os.path.join(outdir, f"ajo_{a}_sijainnit.csv")
        pb = os.path.join(outdir, f"ajo_{b}_sijainnit.csv")
        if a in results and b in results:
            out.append(f"{a} vs {b}: {compare_csv(pa, pb)}")
        pa = os.path.join(outdir, f"ajo_{a}_sijainnit_hog.csv")
        pb = os.path.join(outdir, f"ajo_{b}_sijainnit_hog.csv")
        if a in results and b in results and os.path.exists(pa) and os.path.exists(pb):
            out.append(f"{a} vs {b} (hog): {compare_csv(pa, pb)}")
    for name, r in results.items():
        out.append("")
        out.append(f"=================== AJO {name}{' (AIEMPI AJO, ei ajettu nyt)' if r.get('earlier') else ''} (paluukoodi {r['rc']}, seinakello {r['wall_s']:.0f} s) ===================")
        out.append(extract_blocks(r["text"]))
    summary = "\n".join(out)
    path = os.path.join(outdir, "vertailu_yhteenveto.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(summary)
    print("\n" + summary)
    print(f"\nYhteenveto tallennettu: {path}\nLähetä tämä tiedosto (vertailu_yhteenveto.txt) minulle.")


if __name__ == "__main__":
    main()
