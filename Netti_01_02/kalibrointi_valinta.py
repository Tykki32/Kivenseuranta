"""Kalibrointikansion automaattinen valinta videolle (netti_live.py, tiedosto- ja YouTube-lahde ilman --kansio):

    1. kaydaan lapi projektikansion kalibrointikansiot (tulokset, tulokset_nettivideo2, tulokset_nettivideo3, tulokset_auto_*: kalibrointi_*.json + moodikuva_*.png kummallekin puoliskolle) ja
       valitaan ensimmainen, jonka moodikuva sopii videon kuvaan (kalibroinnin tarkistus: tools/tarkista_kalibrointi.py; kaikkien ikkunoiden siirtyma < 1,5 px)
    2. jos mikaan ei sovi (toinen kamera-asento, kentta tai mainokset), kalibroidaan video uuteen kansioon tulokset_auto_<videon nimi> (renkaat ja viivat moodikuvasta,
       tools/uusi_kalibrointi.py; kivi-, kahva- ja kierremallit kopioidaan kansiosta tulokset) ja kaytetaan sita; sama kansio sopii videon muihin osiin ja ajoihin (osa-numero poistetaan nimesta).
    YouTube-video (ei live): ruudut haetaan hakemalla virrasta (OpenCV / FFmpeg lukee HLS- tai https-virran), ja kansio nimetaan tunnuksella (tulokset_auto_yt_<tunnus>); live-lahetysta ei voi
    kalibroida eika siihen valita kansiota (netti_live.py kayttaa kansiota tulokset tai --kansio:ta).

Kalibrointi ja moodikuva kuuluvat yhteen kameran asentoon: vaara kansio antaa vinot paikat (6-17 cm), ~1 % nopeusvirheen ja B-sovitukseen haamureunoja, joiden vuoksi kivia hukkuu
(rata harva, B hylkaa kiven) ja irroitus lukittuu hog-viivalle."""
import hashlib
import os
import re
import sys

KOHDE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(KOHDE, "tools"))
LISAKANSIO_ENV = "NETTI_KALIBROINNIT"                      # palvelin / kontti: kayttajan oma kalibrointikansioiden juuri (esim. /data/kalibroinnit); sovelluksen oma kansio on kontissa vain luettavissa

TARVITTAVAT = ("kalibrointi_vasen.json", "kalibrointi_oikea.json", "moodikuva_vasen.png", "moodikuva_oikea.png")


def lisakansio():
    """Kayttajan kalibrointikansioiden juuri (NETTI_KALIBROINNIT) tai None."""
    k = os.environ.get(LISAKANSIO_ENV)
    return os.path.abspath(k) if k else None


def ehdokkaat(projekti=KOHDE):
    """Projektikansion kalibrointikansiot (tulokset ensin, sitten aakkosjarjestyksessa), ja perassa kayttajan omat (NETTI_KALIBROINNIT: mika tahansa alikansio, jossa on tarvittavat tiedostot)."""
    out = []
    for nimi in sorted(os.listdir(projekti), key=lambda n: (n != "tulokset", n)):
        k = os.path.join(projekti, nimi)
        if nimi.startswith("tulokset") and os.path.isdir(k) and all(os.path.exists(os.path.join(k, t)) for t in TARVITTAVAT):
            out.append(k)
    lisa = lisakansio()
    if lisa and os.path.isdir(lisa):
        for nimi in sorted(os.listdir(lisa)):
            k = os.path.join(lisa, nimi)
            if os.path.isdir(k) and all(os.path.exists(os.path.join(k, t)) for t in TARVITTAVAT):
                out.append(k)
    return out


def automaattinen_nimi(video):
    """tulokset_auto_<videon nimi ilman paatetta ja osa-numeroa>: videon osat (Matarmaa_osa_05.mp4) jakavat kansion."""
    nimi = re.sub(r"_osa_\d+$", "", os.path.splitext(os.path.basename(video))[0])
    return "tulokset_auto_" + re.sub(r"[^0-9A-Za-z_-]+", "_", nimi)


def virran_nimi(ref):
    """Virran lyhyt nimi kalibrointikansion nimea varten: YouTube-tunnus -> yt_<tunnus>, muu osoite -> virta_<tiiviste>. Osoitteen loppuosa ei kelpaa nimeksi (kaikilla HLS-virroilla se on index.m3u8)."""
    m = re.search(r"(?:v=|youtu\.be/|/live/|/shorts/|/embed/)([A-Za-z0-9_-]{11})", ref) or re.fullmatch(r"([A-Za-z0-9_-]{11})", ref)
    return "yt_" + m.group(1) if m else "virta_" + hashlib.sha1(ref.encode("utf-8")).hexdigest()[:8]


def on_virta(video):
    return isinstance(video, str) and re.match(r"(?i)^(https?|rtsp|rtmps?|udp|tcp|srt)://", video) is not None


def sopii(tulos, tk):
    return all(tk.arvioi(tulos[p])[0] is True for p in ("vasen", "oikea"))


def lahimmat(tulokset, n=3):
    """Lyhyt selitys 'ei sovi' -viestiin: n lahinta kansiota ja niiden ikkunasiirtyma (p90, px) vasen/oikea ('?' = liian vahan vertailukelpoisia ikkunoita). Erottaa melkein sopivan (esim. 0.2/1.9)
    kokonaan eri kamera-asennosta (esim. 11/9) ja kertoo, mista kansiosta oli kyse."""
    def p90(t, puoli):
        x = (t.get(puoli) or {}).get("p90")
        return None if x is None else float(x)
    rivit = []
    for k, t in tulokset.items():
        v, o = p90(t, "vasen"), p90(t, "oikea")
        rivit.append((max((x for x in (v, o) if x is not None), default=float("inf")), k, v, o))
    rivit.sort(key=lambda r: r[0])
    f = lambda x: "?" if x is None else f"{x:.1f}"                                                                      # noqa: E731
    return "; ".join(f"{os.path.basename(k)} {f(v)}/{f(o)}" for _, k, v, o in rivit[:n])


def valitse(video, projekti=KOHDE, luo=True, tulosta=print, n=24, nimi=None, aikaraja_s=None):
    """(kansio, selitys): ensimmainen videoon sopiva kalibrointikansio; ellei sellaista ole, uusi automaattinen kalibrointi (luo = True) tai tulokset ja varoitus.
    video: tiedosto tai virran osoite (YouTube-VOD: ruutuja haetaan hakemalla virrasta, myos uusi kalibrointi luetaan virrasta); n: mediaanikuvan ruutujen maara. nimi: uuden kansion nimi
    (tulokset_auto_<nimi>; oletus tiedoston nimi, virralle virran_nimi(osoite)); aikaraja_s: uuden kalibroinnin kunkin vaiheen enimmaisaika (virta: verkko voi jumittaa; tiedostolle None)."""
    import tarkista_kalibrointi as tk
    import kameran_heilunta as kh
    ref = kh.referenssi(video, n=n)
    tulokset = {}
    for k in ehdokkaat(projekti):
        t = tk.siirtymat(video, k, ref_video=ref)
        tulokset[k] = t
        if sopii(t, tk):
            kansion_nimi = os.path.relpath(k, projekti) if os.path.dirname(k) == os.path.abspath(projekti) else k
            return k, f"kalibrointikansio {kansion_nimi} sopii videoon ({', '.join(f'{p}: p90 {t[p]['p90']:.2f} px' for p in ('vasen', 'oikea'))})"
    oletus = os.path.join(projekti, "tulokset")
    virta = on_virta(video)
    ei_sovi = f"mikaan kalibrointikansio ei sovi videoon (ikkunasiirtyma p90 px vasen/oikea, raja {tk.RAJA_PX:g}; lahimmat: {lahimmat(tulokset)})"
    if not luo:
        return oletus, ei_sovi
    ulos = os.path.join(lisakansio() or projekti, automaattinen_nimi(nimi or (virran_nimi(video) if virta else video)))      # kontissa uudet kalibroinnit kayttajan volyymiin (sailyvat paivityksissa)
    tulosta(f"Mikaan kalibrointikansio ({', '.join(os.path.basename(k) for k in tulokset)}) ei sovi videoon ({lahimmat(tulokset)}): kalibroidaan video kansioon {os.path.basename(ulos)} "
            + ("(virrasta: ~2-5 min, haku virrasta on hidasta)" if virta else "(~1 min)"))
    try:
        import uusi_kalibrointi as uk
        r = uk.kalibroi_video(video, ulos, os.path.join(projekti, "tulokset"), tulosta=tulosta, aikaraja_s=aikaraja_s)
    except Exception as e:                                       # noqa: BLE001
        return oletus, f"automaattinen kalibrointi epaonnistui ({e!r}); kaytetaan kansiota tulokset (tulokset epatarkkoja)"
    if r != 0 and not all(os.path.exists(os.path.join(ulos, t)) for t in TARVITTAVAT):
        syy = "aikaraja ylittyi: virran luku on hidasta" if r == uk.AIKARAJA_KOODI else "renkaita tai viivoja ei loytynyt"
        return oletus, f"automaattinen kalibrointi epaonnistui ({syy}); kaytetaan kansiota tulokset (tulokset epatarkkoja)"
    return ulos, f"kalibroitu automaattisesti kansioon {os.path.basename(ulos)}"
