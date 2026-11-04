"""Ajoymparisto: kayttokelpoiset prosessoriytimet ja muisti, kontin (Docker / Unraid) rajat huomioiden.

Kontissa os.cpu_count() ja sysconf kertovat koko palvelimen resurssit, vaikka kontille olisi asetettu rajat (docker --cpus, --cpuset-cpus, --memory; Unraidissa "CPU pinning" ja
muistiraja). Oletusarvot (tyoprosessien maara, viivepuskurin muoto, esilatauksen pituus) valitaan taman moduulin luvuilla, jotta kontti ei yrita kayttaa enempaa kuin sille on sallittu:
liika muisti johtaisi siihen, etta ydin tappaa prosessin (OOM) ilman selkeaa virheilmoitusta.

Vain standardikirjasto; toimii myos Windowsissa ja ilman cgroupeja (silloin rajoja ei ole ja luvut ovat koneen omat)."""
import math
import os

CGROUP_V2_CPU = "/sys/fs/cgroup/cpu.max"
CGROUP_V1_KVOOTTI = "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"
CGROUP_V1_JAKSO = "/sys/fs/cgroup/cpu/cpu.cfs_period_us"
CGROUP_MUISTI = ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes")      # v2, v1
EI_RAJAA_TAVUA = 2 ** 60                                                                         # v1:n "ei rajaa" on 9223372036854771712


def _lue(polku):
    try:
        with open(polku, "r", encoding="ascii") as f:
            return f.read().strip()
    except (OSError, UnicodeDecodeError):
        return None


def cgroup_ytimet(v2=CGROUP_V2_CPU, v1_kvootti=CGROUP_V1_KVOOTTI, v1_jakso=CGROUP_V1_JAKSO):
    """Kontin prosessorikiintio ytimina (murtoluku mahdollinen, esim. 1.5) tai None, jos rajaa ei ole tai sita ei saada selville."""
    s = _lue(v2)
    if s is not None:                                             # cgroup v2: "max 100000" (ei rajaa) tai "<kvootti> <jakso>"
        osat = s.split()
        if len(osat) == 2 and osat[0] != "max":
            try:
                kiintio = float(osat[0]) / float(osat[1])
                return kiintio if kiintio > 0 else None
            except (ValueError, ZeroDivisionError):
                return None
        return None
    kvootti, jakso = _lue(v1_kvootti), _lue(v1_jakso)             # cgroup v1: kvootti -1 = ei rajaa
    try:
        if kvootti is not None and jakso is not None and int(kvootti) > 0 and int(jakso) > 0:
            return int(kvootti) / int(jakso)
    except ValueError:
        pass
    return None


def ytimet():
    """Kayttokelpoisten loogisten ytimien maara (vahintaan 1): prosessille sallitut ytimet (cpuset / taskset) ja cgroup-kiintio, pienempi naista (kiintio ylospain pyoristettyna)."""
    try:
        n = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):                             # Windows, macOS
        n = os.cpu_count() or 1
    kiintio = cgroup_ytimet()
    if kiintio is not None:
        n = min(n, max(1, int(math.ceil(kiintio - 1e-9))))
    return max(1, n)


def cgroup_muisti_gt(polut=CGROUP_MUISTI):
    """Kontin muistiraja (Gt) tai None, jos rajaa ei ole."""
    for p in polut:
        s = _lue(p)
        if s and s != "max":
            try:
                tavua = int(s)
            except ValueError:
                continue
            if 0 < tavua < EI_RAJAA_TAVUA:
                return tavua / 2 ** 30
    return None


def muisti_gt():
    """Kaytettavissa oleva fyysinen muisti (Gt): koneen muisti, korkeintaan kontin muistiraja; None, jos sita ei saada selville."""
    try:
        fyysinen = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30
    except (AttributeError, ValueError, OSError):                 # Windows
        fyysinen = None
    raja = cgroup_muisti_gt()
    if fyysinen is None:
        return raja
    return min(fyysinen, raja) if raja is not None else fyysinen


def kuvaus():
    """Yksi rivi lokiin: ytimet ja muisti sellaisina kuin ohjelma ne nakee."""
    m = muisti_gt()
    raja = cgroup_ytimet()
    return (f"{ytimet()} ydinta kaytossa (koneessa {os.cpu_count() or '?'}" + (f", kontin kiintio {raja:g}" if raja is not None else "") + ")"
            + (f", muistia {m:.1f} Gt" if m is not None else ", muistia ei tiedossa"))


if __name__ == "__main__":
    print(kuvaus())
