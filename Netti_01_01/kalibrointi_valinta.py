"""Kalibrointikansion automaattinen valinta videolle (netti_live.py, tiedostolahde ilman --kansio):

    1. kaydaan lapi projektikansion kalibrointikansiot (tulokset, tulokset_nettivideo2, tulokset_nettivideo3, tulokset_auto_*: kalibrointi_*.json + moodikuva_*.png kummallekin puoliskolle) ja
       valitaan ensimmainen, jonka moodikuva sopii videon kuvaan (kalibroinnin tarkistus: tools/tarkista_kalibrointi.py; kaikkien ikkunoiden siirtyma < 1,5 px)
    2. jos mikaan ei sovi (toinen kamera-asento, kentta tai mainokset), kalibroidaan video uuteen kansioon tulokset_auto_<videon nimi> (renkaat ja viivat moodikuvasta,
       tools/uusi_kalibrointi.py; kivi-, kahva- ja kierremallit kopioidaan kansiosta tulokset) ja kaytetaan sita; sama kansio sopii videon muihin osiin ja ajoihin (osa-numero poistetaan nimesta).

Kalibrointi ja moodikuva kuuluvat yhteen kameran asentoon: vaara kansio antaa vinot paikat (6-17 cm), ~1 % nopeusvirheen ja B-sovitukseen haamureunoja, joiden vuoksi kivia hukkuu
(rata harva, B hylkaa kiven) ja irroitus lukittuu hog-viivalle."""
import os
import re
import sys

KOHDE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(KOHDE, "tools"))

TARVITTAVAT = ("kalibrointi_vasen.json", "kalibrointi_oikea.json", "moodikuva_vasen.png", "moodikuva_oikea.png")


def ehdokkaat(projekti=KOHDE):
    """Projektikansion kalibrointikansiot (tulokset ensin, sitten aakkosjarjestyksessa)."""
    out = []
    for nimi in sorted(os.listdir(projekti), key=lambda n: (n != "tulokset", n)):
        k = os.path.join(projekti, nimi)
        if nimi.startswith("tulokset") and os.path.isdir(k) and all(os.path.exists(os.path.join(k, t)) for t in TARVITTAVAT):
            out.append(k)
    return out


def automaattinen_nimi(video):
    """tulokset_auto_<videon nimi ilman paatetta ja osa-numeroa>: videon osat (Matarmaa_osa_05.mp4) jakavat kansion."""
    nimi = re.sub(r"_osa_\d+$", "", os.path.splitext(os.path.basename(video))[0])
    return "tulokset_auto_" + re.sub(r"[^0-9A-Za-z_-]+", "_", nimi)


def sopii(tulos, tk):
    return all(tk.arvioi(tulos[p])[0] is True for p in ("vasen", "oikea"))


def valitse(video, projekti=KOHDE, luo=True, tulosta=print, n=24):
    """(kansio, selitys): ensimmainen videoon sopiva kalibrointikansio; ellei sellaista ole, uusi automaattinen kalibrointi (luo = True) tai tulokset ja varoitus.
    video: tiedosto tai virran osoite (YouTube, kun luo = False: ruutuja haetaan hakemalla n kohdasta, ei kalibroida virtaa); n: mediaanikuvan ruutujen maara."""
    import tarkista_kalibrointi as tk
    import kameran_heilunta as kh
    ref = kh.referenssi(video, n=n)
    tulokset = {}
    for k in ehdokkaat(projekti):
        t = tk.siirtymat(video, k, ref_video=ref)
        tulokset[k] = t
        if sopii(t, tk):
            return k, f"kalibrointikansio {os.path.relpath(k, projekti)} sopii videoon ({', '.join(f'{p}: p90 {t[p]['p90']:.2f} px' for p in ('vasen', 'oikea'))})"
    oletus = os.path.join(projekti, "tulokset")
    if not luo:
        return oletus, "mikaan kalibrointikansio ei sovi videoon"
    ulos = os.path.join(projekti, automaattinen_nimi(video))
    tulosta(f"Mikaan kalibrointikansio ({', '.join(os.path.basename(k) for k in tulokset)}) ei sovi videoon: kalibroidaan video kansioon {os.path.basename(ulos)} (~1 min)")
    try:
        import uusi_kalibrointi as uk
        r = uk.kalibroi_video(video, ulos, os.path.join(projekti, "tulokset"), tulosta=tulosta)
    except Exception as e:                                       # noqa: BLE001
        return oletus, f"automaattinen kalibrointi epaonnistui ({e!r}); kaytetaan kansiota tulokset (tulokset epatarkkoja)"
    if r != 0 and not all(os.path.exists(os.path.join(ulos, t)) for t in TARVITTAVAT):
        return oletus, "automaattinen kalibrointi epaonnistui (renkaita tai viivoja ei loytynyt); kaytetaan kansiota tulokset (tulokset epatarkkoja)"
    return ulos, f"kalibroitu automaattisesti kansioon {os.path.basename(ulos)}"
