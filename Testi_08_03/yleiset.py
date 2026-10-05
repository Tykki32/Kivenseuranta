"""Yhteiset apuvalineet: vaihekohtainen aikamittaus, taustaprosessien prioriteetti ja versiotieto."""

import contextlib
import os
import subprocess
import sys
import threading
import time

import asetukset as A

# ------------------------------------------------------------------
# AIKAMITTAUS: PROF[nimi] = [sekunnit yhteensa, kutsuja]. Raportti tulostetaan ajon lopussa (seuranta.py).
# ------------------------------------------------------------------
PROF = {}


def prof_add(name, seconds):
    e = PROF.setdefault(name, [0.0, 0])
    e[0] += seconds
    e[1] += 1


@contextlib.contextmanager
def prof(name):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        prof_add(name, time.perf_counter() - t0)


# ------------------------------------------------------------------
# TAUSTAPROSESSIEN PRIORITEETTI: live-tallennuksen saie ja ffmpeg-aliprosessit (debug-video, live-tallennus)
# alemmalle prioriteetille, jotta seuranta saa ytimet ensin. Vaikuttaa vain ajoitukseen, ei tuloksiin.
# ------------------------------------------------------------------

def alempi_prioriteetti():
    """Laskee KUTSUVAN saikeen prioriteettia (Windows BELOW_NORMAL, muuten nice +5). Virheet ohitetaan."""
    if not A.TAUSTA_ALEMPI_PRIORITEETTI:
        return
    try:
        if sys.platform == "win32":
            import ctypes
            k32 = ctypes.windll.kernel32
            k32.SetThreadPriority(k32.GetCurrentThread(), -1)      # THREAD_PRIORITY_BELOW_NORMAL
        elif hasattr(os, "setpriority"):
            os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), 5)
    except Exception:
        pass


def bg_popen_kwargs():
    """subprocess.Popen-argumentit ffmpeg-aliprosessille alemmalla prioriteetilla."""
    if not A.TAUSTA_ALEMPI_PRIORITEETTI:
        return {}
    if sys.platform == "win32":
        return {"creationflags": getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0x00004000)}
    return {"preexec_fn": lambda: os.nice(5)}


# ------------------------------------------------------------------
# VERSIO: nakyy raporttien lopussa. Git-tiiviste ja C++-moduulien kaannosaika kertovat, onko .pyd/.so kaannetty
# uudelleen (vanha C++-moduuli + uusi Python-koodi on tyypillinen sekaannus).
# ------------------------------------------------------------------
SOFTWARE_VERSION = "Testi_08_03 v9.0 (siivottu versio Testi_08_02 v8.8-t16:sta: sama toiminta, koodi jaettu osiin, asetukset.py)"


def version_string():
    parts = [SOFTWARE_VERSION]
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        h = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=here, capture_output=True, text=True, timeout=3)
        if h.returncode == 0 and h.stdout.strip():
            d = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no", "."], cwd=here,
                               capture_output=True, text=True, timeout=3)
            parts.append("git " + h.stdout.strip() + ("+muokattu" if d.stdout.strip() else ""))
    except Exception:
        pass
    try:
        import stone_tracker
        import mode_engine
        for mod in (stone_tracker, mode_engine):
            try:
                parts.append(mod.build_info())
            except Exception:
                parts.append(getattr(mod, "__name__", "?") + " (ei build_info: vanha moduuli - kaanna uudelleen)")
    except ImportError:
        pass
    return " | ".join(parts)


# ------------------------------------------------------------------
# PULLONKAULA-ANALYYSI: liukuhihnan vaiheet A (luku + stabilointi), B (warp + varjosuodatus) ja C (paasaie: HAKU,
# SEURANTA, CSV, debug) ajavat rinnan; nopeuden maaraa HITAIN vaihe (suurin palveluaika ms/ruutu).
# ------------------------------------------------------------------

def _stage_numbers():
    P = lambda k: PROF.get(k, [0.0, 0])
    a_n = P("pipe A: stabilointi (vaihekorrelaatio)")[1]
    b_n = P("pipe B: odottaa vaihetta A")[1]
    c_n = P("pipe C: paasaie odottaa hihnaa (sisaltyy py: read(video)-riviin)")[1]
    if a_n < 20 or b_n < 20 or c_n < 20:
        return None
    a_busy = sum(v[0] for k, v in PROF.items() if k.startswith("pipe A:") and "odottaa" not in k) / a_n * 1000
    a_wout = P("pipe A: odottaa vaihetta B (jono taynna)")[0] / a_n * 1000
    b_busy = sum(v[0] for k, v in PROF.items() if k.startswith("pipe B:") and "odottaa" not in k) / b_n * 1000
    b_win = P("pipe B: odottaa vaihetta A")[0] / b_n * 1000
    b_wout = P("pipe B: odottaa paasaiketta (jono taynna)")[0] / b_n * 1000
    c_win = P("pipe C: paasaie odottaa hihnaa (sisaltyy py: read(video)-riviin)")[0] / c_n * 1000
    c_tot = P("FRAME_KOKO")[0] / max(P("FRAME_KOKO")[1], 1) * 1000
    c_busy = max(c_tot - c_win, 1e-6)
    return dict(a=(a_busy, 0.0, a_wout), b=(b_busy, b_win, b_wout), c=(c_busy, c_win, 0.0), c_tot=c_tot)


def _jonot(ps):
    if not ps["qC_n"]:
        return ""
    return (f"A->B {ps['qA_sum'] / max(ps['qA_n'], 1):.1f}/{ps['depth']} "
            f"B->C {ps['qC_sum'] / ps['qC_n']:.1f}/{ps['depth']}")


def pullonkaula_tiivis(pipe_stats):
    """Yhden rivin pullonkaulatieto edistymisraporttiin (tai None)."""
    s = _stage_numbers()
    if s is None:
        return None
    names = {"a": "A luku+stabilointi", "b": "B warp+varjosuodatus", "c": "C paasaie"}
    worst = max(("a", "b", "c"), key=lambda k: s[k][0])
    q = _jonot(pipe_stats)
    return (f"  pullonkaula: {names[worst]} ({s[worst][0]:.0f} ms/ruutu = max {1000 / s[worst][0]:.1f} r/s) | "
            f"A {s['a'][0]:.0f} ms, B {s['b'][0]:.0f} ms, C {s['c'][0]:.0f} ms" + (f" | jonot {q}" if q else ""))


def aikamittausraportti(n_frames, n_seuranta_updates, pipe_stats):
    """Ajon lopun raportti: pullonkaula + vaihekohtaiset ajat (Python ja C++)."""
    s = _stage_numbers()
    print()
    if s is not None:
        print("=== PULLONKAULA-ANALYYSI ===")
        names = {"a": "A  luku + gray + stabilointi (vaihekorrelaatio)", "b": "B  warpAffine+remap + valotasapaino + varjosuodatus",
                 "c": "C  paasaie: HAKU/SEURANTA/CSV/debug-video"}
        print(f"{'vaihe':52s} {'palvelu':>8s} {'kapas.':>7s} {'odottaa':>8s} {'odottaa':>8s} {'kuorm.':>7s}")
        print(f"{'':52s} {'ms/rt':>8s} {'r/s':>7s} {'syotetta':>8s} {'tulosta':>8s} {'%':>7s}")
        for k in ("a", "b", "c"):
            busy, win, wout = s[k]
            print(f"{names[k]:52s} {busy:8.1f} {1000 / busy:7.1f} {win:8.1f} {wout:8.1f} {100 * busy / (busy + win + wout):7.0f}")
        worst = max(("a", "b", "c"), key=lambda k: s[k][0])
        print(f"=> PULLONKAULA: {names[worst].split('  ', 1)[1].strip()} - kapasiteetti {1000 / s[worst][0]:.1f} r/s "
              f"(mitattu kokonaisnopeus {1000 / s['c_tot']:.1f} r/s)")
        q = _jonot(pipe_stats)
        if q:
            print(f"Jonojen keskitaytto: {q} (B->C taynna = paasaie ei ehdi ottaa; tyhja = ylavirran vaihe on hitain)")
        try:
            wall = time.perf_counter() - pipe_stats["wall0"]
            cpu = time.process_time() - pipe_stats["cpu0"]
            th = []
            if pipe_stats["cpu_a"] is not None:
                th.append(f"A {100 * pipe_stats['cpu_a'] / wall:.0f}%")
            if pipe_stats["cpu_b"] is not None:
                th.append(f"B {100 * pipe_stats['cpu_b'] / wall:.0f}%")
            th.append(f"paasaie {100 * time.thread_time() / max(wall, 1e-9):.0f}%")
            print(f"CPU: prosessi {cpu / wall:.2f} ydinta keskimaarin ({os.cpu_count()} loogista ydinta); saikeiden "
                  f"CPU/seinakello: " + ", ".join(th))
        except Exception:
            pass
        print()
    print("=== VAIHEKOHTAINEN AIKAMITTAUS ===")
    print(f"ruutuja: {n_frames}, kivipaivityksia: {n_seuranta_updates}")
    serial = sum(sec for k, (sec, n) in PROF.items() if k.startswith("py:") and "taustasaikeen oma kesto" not in k)
    tot = PROF.get("FRAME_KOKO", [0.0, 1])[0]
    PROF["py: MUU / JAANNOS (ei mitattu: FRAME_KOKO - mitatut sarjavaiheet)"] = [max(0.0, tot - serial), n_frames]
    print("--- Python-puoli (ms/ruutu, kutsuja) ---")
    for k, (sec, n) in sorted(PROF.items(), key=lambda kv: -kv[1][0]):
        print(f"{k:58s} {sec / n_frames * 1000:8.2f} ms/ruutu {100 * sec / max(tot, 1e-9):5.1f}%  n={n}")
    try:
        import stone_tracker
        snap = stone_tracker.prof_snapshot()
    except Exception:
        snap = []
    if snap:
        print("--- C++-puoli (summattu CPU-aika saikeiden yli) ---")
        print(f"{'vaihe':58s} {'ms/ruutu':>9s} {'ms/kutsu':>9s} {'kutsuja':>8s}")
        for name, ms, n in snap:
            if n == 0:
                continue
            if "ulkoiteraatioita" in name:
                print(f"{name:58s} {'':>9s} {'':>9s} {n:8d}   (ulkoiteraatioita / LM-tarkennus)")
                continue
            print(f"{name:58s} {ms / n_frames:9.2f} {ms / n:9.3f} {n:8d}")
    print("Versio: " + version_string())
    print("================================================")
