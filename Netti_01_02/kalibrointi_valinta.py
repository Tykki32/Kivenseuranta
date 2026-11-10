"""Kalibrointi videolle (netti_live.py, tiedosto- ja YouTube-lahde ilman --kansio). Kaksi tapaa (netti_live.py --kalibrointi):

    uusi (oletus): video kalibroidaan aina uudelleen omista ruuduistaan kansioon tulokset_auto_<videon nimi> (renkaat ja viivat moodikuvasta, tools/uusi_kalibrointi.py; kivi-, kahva- ja
       kierremallit kopioidaan kansiosta tulokset). Vanhoja kalibrointeja ei kayteta: sama rata eri viikkoina kuvattuna ei aina sovi vanhaan kalibrointiin (kamera on voinut siirtya), eika
       sopivuustarkistus kerro, oliko vanha kalibrointi alun perinkaan hyva. Ellei kalibrointi onnistu (renkaita tai viivoja ei loydy), kaytetaan videoon sopivaa tai lahinta vanhaa kansiota
       (poikkeama kerrotaan), ellei mitaan voi mitata, kansiota tulokset.
    valitse: kaydaan lapi projektikansion kalibrointikansiot (tulokset, tulokset_nettivideo2, tulokset_nettivideo3, tulokset_auto_*: kalibrointi_*.json + moodikuva_*.png kummallekin puoliskolle)
       ja valitaan ensimmainen, jonka moodikuva sopii videon kuvaan (tools/tarkista_kalibrointi.py; ikkunasiirtyman p90 < 1,5 px); vain jos mikaan ei sovi, kalibroidaan uusi (nopeampi,
       kun kamera pysyy samana).
    Sama kansio sopii videon osiin (osa-numero poistetaan nimesta). YouTube-video (ei live): ruudut haetaan hakemalla virrasta (OpenCV / FFmpeg lukee HLS- tai https-virran), ja kansio
    nimetaan tunnuksella (tulokset_auto_yt_<tunnus>); live-lahetysta ei voi kalibroida eika siihen valita kansiota (netti_live.py kayttaa kansiota tulokset tai --kansio:ta).

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


def _lajiteltu(tulokset):
    """[(huonoin p90, kansio, vasen p90, oikea p90)] lahimmasta alkaen: ensin kansiot, joiden molemmat puoliskot voitiin mitata (huonoimman puoliskon p90 pienimmasta), sitten ne, joilta toinen puolisko
    puuttuu (p90 = None: liian vahan vertailukelpoisia ikkunoita eli kuva eroaa moodikuvasta muutenkin kuin siirtyman osalta), ja viimeisena kansiot, joilta kumpikaan ei ole mitattavissa (huonoin = inf)."""
    def p90(t, puoli):
        x = (t.get(puoli) or {}).get("p90")
        return None if x is None else float(x)
    rivit = []
    for k, t in tulokset.items():
        v, o = p90(t, "vasen"), p90(t, "oikea")
        rivit.append((max((x for x in (v, o) if x is not None), default=float("inf")), k, v, o))
    rivit.sort(key=lambda r: ((r[2] is None) + (r[3] is None), r[0]))
    return rivit


def _px(x):
    return "?" if x is None else f"{x:.1f}"


def lahimmat(tulokset, n=3):
    """Lyhyt selitys 'ei sovi' -viestiin: n lahinta kansiota ja niiden ikkunasiirtyma (p90, px) vasen/oikea ('?' = liian vahan vertailukelpoisia ikkunoita). Erottaa melkein sopivan (esim. 0.2/1.9)
    kokonaan eri kamera-asennosta (esim. 11/9) ja kertoo, mista kansiosta oli kyse."""
    return "; ".join(f"{os.path.basename(k)} {_px(v)}/{_px(o)}" for _, k, v, o in _lajiteltu(tulokset)[:n])


def _mittaa(video, projekti, n, tk):
    """Kaikkien ehdokaskansioiden ikkunasiirtyma videoon nahden: {kansio: tk.siirtymat}."""
    import kameran_heilunta as kh
    ref = kh.referenssi(video, n=n)
    return {k: tk.siirtymat(video, k, ref_video=ref) for k in ehdokkaat(projekti)}


def _varakansio(video, projekti, n, tulokset, syy, tk):
    """Uusi kalibrointi epaonnistui (syy): videoon sopiva vanha kansio, ellei sellaista ole, lahin vanha kansio (poikkeama kerrotaan), ellei mitaan voi mitata, tulokset. (kansio, selitys).
    tulokset: jo mitatut {kansio: siirtymat} tai None (mitataan tassa)."""
    oletus = os.path.join(projekti, "tulokset")
    try:
        if tulokset is None:
            tulokset = _mittaa(video, projekti, n, tk)
    except Exception as e:                                       # noqa: BLE001
        return oletus, f"automaattinen kalibrointi epaonnistui ({syy}); kaytetaan kansiota tulokset (vanhojen kansioiden vertailu epaonnistui: {e!r}; tulokset epatarkkoja)"
    tulokset = {k: t for k, t in tulokset.items() if all(os.path.exists(os.path.join(k, f)) for f in TARVITTAVAT)}      # juuri tyhjennetty oma kansio ei ole ehdokas
    for k, t in tulokset.items():
        if sopii(t, tk):
            return k, f"automaattinen kalibrointi epaonnistui ({syy}); kaytetaan vanhaa kansiota {os.path.basename(k)}, joka sopii videoon ({', '.join(f'{p}: p90 {t[p]['p90']:.2f} px' for p in ('vasen', 'oikea'))})"
    rivit = _lajiteltu(tulokset)
    if rivit and rivit[0][0] != float("inf"):
        _, k, v, o = rivit[0]
        return k, (f"automaattinen kalibrointi epaonnistui ({syy}); kaytetaan lahinta vanhaa kansiota {os.path.basename(k)} (ikkunasiirtyma p90 px vasen/oikea {_px(v)}/{_px(o)}, raja {tk.RAJA_PX:g}): "
                   "tulokset epatarkkoja")
    return oletus, f"automaattinen kalibrointi epaonnistui ({syy}); kaytetaan kansiota tulokset (tulokset epatarkkoja)"


def valitse(video, projekti=KOHDE, luo=True, tulosta=print, n=24, nimi=None, aikaraja_s=None, uusi=False):
    """(kansio, selitys). uusi = False: ensimmainen videoon sopiva kalibrointikansio; ellei sellaista ole, uusi automaattinen kalibrointi (luo = True) tai tulokset ja varoitus.
    uusi = True: vanhoja kalibrointeja ei kayteta, vaan video kalibroidaan aina uudelleen (luo = True oletettu); ellei kalibrointi onnistu, kaytetaan videoon sopivaa tai lahinta vanhaa kansiota.
    video: tiedosto tai virran osoite (YouTube-VOD: ruutuja haetaan hakemalla virrasta, myos uusi kalibrointi luetaan virrasta); n: mediaanikuvan ruutujen maara. nimi: uuden kansion nimi
    (tulokset_auto_<nimi>; oletus tiedoston nimi, virralle virran_nimi(osoite)); aikaraja_s: uuden kalibroinnin kunkin vaiheen enimmaisaika (virta: verkko voi jumittaa; tiedostolle None)."""
    import tarkista_kalibrointi as tk
    virta = on_virta(video)
    tulokset = None
    if uusi:
        luo = True
    else:
        import kameran_heilunta as kh
        ref = kh.referenssi(video, n=n)
        tulokset = {}
        for k in ehdokkaat(projekti):
            t = tk.siirtymat(video, k, ref_video=ref)
            tulokset[k] = t
            if sopii(t, tk):
                kansion_nimi = os.path.relpath(k, projekti) if os.path.dirname(k) == os.path.abspath(projekti) else k
                return k, f"kalibrointikansio {kansion_nimi} sopii videoon ({', '.join(f'{p}: p90 {t[p]['p90']:.2f} px' for p in ('vasen', 'oikea'))})"
        if not luo:
            return os.path.join(projekti, "tulokset"), f"mikaan kalibrointikansio ei sovi videoon (ikkunasiirtyma p90 px vasen/oikea, raja {tk.RAJA_PX:g}; lahimmat: {lahimmat(tulokset)})"
    ulos = os.path.join(lisakansio() or projekti, automaattinen_nimi(nimi or (virran_nimi(video) if virta else video)))      # kontissa uudet kalibroinnit kayttajan volyymiin (sailyvat paivityksissa)
    arvio = "(virrasta: ~2-5 min, haku virrasta on hidasta)" if virta else "(~1 min)"
    if uusi:
        tulosta(f"Kalibroidaan video uudelleen (vanhoja kalibrointeja ei kayteta) kansioon {os.path.basename(ulos)} {arvio}")
    else:
        tulosta(f"Mikaan kalibrointikansio ({', '.join(os.path.basename(k) for k in tulokset)}) ei sovi videoon ({lahimmat(tulokset)}): kalibroidaan video kansioon {os.path.basename(ulos)} {arvio}")
    for f in TARVITTAVAT:                                        # edellisen kalibroinnin tiedostot pois: onnistuminen tarkoittaa silloin tuoreita tiedostoja (ei vanhan kalibroinnin jaannoksia)
        try:
            os.remove(os.path.join(ulos, f))
        except OSError:
            pass
    try:
        import uusi_kalibrointi as uk
        r = uk.kalibroi_video(video, ulos, os.path.join(projekti, "tulokset"), tulosta=tulosta, aikaraja_s=aikaraja_s)
    except Exception as e:                                       # noqa: BLE001
        return _varakansio(video, projekti, n, tulokset, repr(e), tk)
    if r != 0 and not all(os.path.exists(os.path.join(ulos, f)) for f in TARVITTAVAT):
        return _varakansio(video, projekti, n, tulokset, "aikaraja ylittyi: virran luku on hidasta" if r == uk.AIKARAJA_KOODI else "renkaita tai viivoja ei loytynyt", tk)
    return ulos, f"kalibroitu automaattisesti kansioon {os.path.basename(ulos)}"
