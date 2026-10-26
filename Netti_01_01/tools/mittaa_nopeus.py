"""Analyysin nopeus tällä koneella: riittääkö se reaaliaikaan? Syöttää videon ruudut analyysille (netti_seuranta.Analyysi, samat työprosessit kuin netti_live.py) ilman piirtoa,
tallennusta ja ikkunaa niin nopeasti kuin ehtii, ja kertoo ruudun ajan (seinäkello ja analyysisäikeen CPU-aika) vaiheittain sekä sen, kuinka suuri osa reaaliajasta analyysi vie.

    python tools/mittaa_nopeus.py VIDEO.mp4 [--alku 300] [--loppu 360] [--saikeet 3] [--kansio tulokset_nettivideo3]
    python tools/mittaa_nopeus.py --youtube [TUNNUS tai osoite] [--alku 300] [--loppu 360] [--saikeet 3] [--kansio ...]
    python tools/mittaa_nopeus.py --url OSOITE [--alku 300] [--loppu 360]

Lähde avataan kuten netti_live.py:ssä (YouTube: yt_dlp hakee virran osoitteen ja OpenCV lukee sen; --yt-formaatti kuten siellä). Ilman --kansio kalibrointikansio valitaan videon mukaan kuten
netti_live.py:ssä (virralle ruutuja haetaan virrasta, voi kestää minuutin). Virrassa --alku siirtää lukukohtaa hakemalla ja, ellei virta salli hakua, lukemalla ruudut ohi, joten pitkä --alku
virrassa vie aikaa (videota ladataan alusta asti): valitse jakso, jolla on heitto, tai lataa video tiedostoksi (yt-dlp) ja anna tiedosto.

Valitse jakso, jolla on heitto (esim. alkaa 25 s ennen heiton lähtöä ja kestää 60 s): raskas osa on heiton aikana. Tulos on yläraja analyysin osuudelle: varsinaisessa ajossa (netti_live.py)
samalla koneella ovat lisäksi videon dekoodaus, piirto, viivepuskurin pakkaus, tallennus (ffmpeg) ja ikkuna. Jos "ruudun aika keskimäärin" on yli 33 ms (30 r/s), analyysi ei ehdi reaaliaikaan ja
kuvan viive kasvaa; silloin kokeile suurempaa --saikeet-arvoa, varmista että tarkka sovitus on C++ (`tarkka sovitus: C++ (netti_fit)`), ja sulje muut raskaat ohjelmat."""
import argparse
import os
import platform
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import netti_live as nl  # noqa: E402
import netti_seuranta as ns  # noqa: E402


def ohita_ruudut(cap, n):
    """Lukee ja hylkää n ruutua (grab: dekoodaus ilman värimuunnosta). Palauttaa ohitettujen määrän (pienempi, jos virta loppuu)."""
    for i in range(int(n)):
        if not cap.grab():
            return i
    return int(n)


def siirry(cap, alku_s, fps, tulosta=print):
    """Virta: lukukohta alku_s:ään. Ensin haku (toimii virroissa, joiden pituus tunnetaan; sama kuin kameran_heilunta.referenssi), ja jos lukukohta ei ole sen jälkeen alle sekunnin päässä
    tavoitteesta, loppumatka luetaan ruutu kerrallaan ohi. Palauttaa lukukohdan ruutuina."""
    tavoite = int(alku_s * fps)
    if tavoite <= 0:
        return 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, tavoite)
    kohta = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    if abs(kohta - tavoite) <= fps:
        return kohta
    if kohta > tavoite:
        tulosta(f"Huom: virran haku meni {(kohta - tavoite) / fps:.0f} s yli tavoitteen; mitataan siitä kohdasta.")
        return kohta
    jaljella = tavoite - kohta
    tulosta(f"Virta ei salli hakua: ohitetaan {jaljella / fps:.0f} s lukemalla ({jaljella} ruutua) ...")
    t = time.perf_counter()
    kohta += ohita_ruudut(cap, jaljella)
    tulosta(f"  ohitettu {time.perf_counter() - t:.0f} s")
    return kohta


def valitse_kansio(a, osoite, live):
    """Kalibrointikansio: --kansio sellaisenaan, muuten videon mukaan kuten netti_live.py:ssä (ei luo uutta kalibrointia: kansio tai varoitus)."""
    if a.kansio is not None:
        return a.kansio if os.path.isdir(a.kansio) else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", a.kansio)
    import kalibrointi_valinta as kval
    try:
        if osoite is None:
            kansio, selitys = kval.valitse(a.video, luo=False)
        elif live:
            print("Huom: live-virralle kalibrointikansiota ei voi valita automaattisesti: käytetään kansiota tulokset (anna --kansio)", flush=True)
            return os.path.join(kval.KOHDE, "tulokset")
        else:
            print(f"Valitaan kalibrointikansio virrasta ({nl.YT_VALINTA_RUUTUJA} ruutua, voi kestää minuutin) ...", flush=True)
            kansio, selitys = kval.valitse(osoite, luo=False, n=nl.YT_VALINTA_RUUTUJA)
    except Exception as e:                                       # noqa: BLE001
        print(f"Huom: kalibrointikansion automaattinen valinta epäonnistui ({e!r}): käytetään kansiota tulokset (anna --kansio)", flush=True)
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tulokset")
    print(f"Kalibrointi: {selitys}", flush=True)
    if "ei sovi" in selitys:
        print("  Anna oikea kansio: --kansio tulokset_nettivideo3 (jne.); väärällä kansiolla heittoja ei tunnisteta eikä tulos kuvaa raskasta osaa.", flush=True)
    return kansio


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("video", nargs="?", help="videotiedosto (tai --youtube / --url)")
    ap.add_argument("--youtube", nargs="?", const=nl.YOUTUBE_OLETUS, help=f"YouTube-tunnus tai -osoite (oletus {nl.YOUTUBE_OLETUS})")
    ap.add_argument("--yt-formaatti", default=nl.YT_FORMAATTI, help="yt_dlp-formaattivalitsin (kuten netti_live.py)")
    ap.add_argument("--url", help="suora virran osoite (http / https / rtsp / m3u8)")
    ap.add_argument("--kansio", default=None, help="kalibrointikansio (tulokset, tulokset_nettivideo2, tulokset_nettivideo3 ...); oletus: valitaan videon mukaan kuten netti_live.py:ssä")
    ap.add_argument("--alku", type=float, default=0.0, help="alku (s)")
    ap.add_argument("--loppu", type=float, default=None, help="loppu (s); oletus alku + 60")
    ap.add_argument("--saikeet", type=int, default=max(1, min(3, (os.cpu_count() or 2) - 1)), help="tarkan sovituksen työprosessit (kuten netti_live.py)")
    a = ap.parse_args()
    if sum(bool(x) for x in (a.video, a.youtube, a.url)) != 1:
        ap.error("anna täsmälleen yksi lähde: VIDEO, --youtube [TUNNUS] tai --url OSOITE")
    loppu = a.loppu if a.loppu is not None else a.alku + 60.0
    if loppu <= a.alku:
        ap.error("--loppu pitää olla suurempi kuin --alku")
    osoite, live = None, False                                   # virran osoite (YouTube: yt_dlp hakee, kerran) ja onko live
    if a.youtube:
        print(f"Haetaan YouTube-virran osoite ({a.youtube}) ...", flush=True)
        osoite, live = nl.youtube_virta(a.youtube, a.yt_formaatti)
    elif a.url:
        osoite = a.url
    kansio = valitse_kansio(a, osoite, live)
    print(f"Python {platform.python_version()} ({platform.system()}), OpenCV {cv2.__version__}, ytimiä {os.cpu_count()}", flush=True)
    # työprosessit käyntiin ennen lähteen avaamista (kuten netti_live.py: spawn-prosessien käynnistys kestää 5-15 s)
    an = ns.Analyysi(kansio, fps=30.0, saikeet=a.saikeet, tarkka=a.saikeet > 0, synkroninen=False, fps_lahde=None, kierre=True)
    n_prosessia = an.lampea()
    sovitin = next(iter(an.M.values())).sovitin
    print(f"{n_prosessia} työprosessia (--saikeet {a.saikeet}); tarkka sovitus: {sovitin}" + ("" if "C++" in sovitin else f" ({ns.netti_fit_selitys()})"), flush=True)
    virta = osoite is not None
    # virrassa ruutunumerot alkavat nollasta ja Lahde.loppu_s on lukujakson pituus; tiedostossa ne ovat videon aikaa (kuten netti_live.py)
    lahde = nl.Lahde(tiedosto=a.video, url=osoite, alku_s=a.alku, loppu_s=(loppu - a.alku) if virta else loppu)
    lahde.live = live
    fps = lahde.fps
    leveys, korkeus = int(lahde.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(lahde.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Lähde: {'YouTube ' + a.youtube if a.youtube else ('virta ' + a.url if a.url else a.video)}, {leveys} x {korkeus}, {fps:.1f} r/s" + (" (live)" if live else ""), flush=True)
    an.fps = fps
    an.aseta_toistettava(True)                                   # kuten tiedosto / VOD netti_live.py:ssä: ruutuja ei ohiteta, B-tulokset odotetaan
    if virta and a.alku > 0:
        siirry(lahde.cap, a.alku, fps)
    ajat, cpu = [], []
    t0 = time.perf_counter()
    for f, ruutu in lahde.ruudut():
        t, c = time.perf_counter(), time.thread_time()
        an.syota(f, ruutu)
        ajat.append(1e3 * (time.perf_counter() - t))
        cpu.append(1e3 * (time.thread_time() - c))
    an.viimeistele()
    kesto = time.perf_counter() - t0
    an.sulje()
    if not ajat:
        raise SystemExit("Ei ruutuja")
    ajat, cpu = np.array(ajat), np.array(cpu)
    tavoite = 1e3 / fps
    print(f"\n{len(ajat)} ruutua ({len(ajat) / fps:.0f} s videota), {an.kopioita} kopioruutua, heittoja {len(an.heitot)}")
    print(f"ruudun aika keskimäärin {ajat.mean():.1f} ms (analyysisäikeen CPU {cpu.mean():.1f} ms), p95 {np.percentile(ajat, 95):.1f} ms, suurin {ajat.max():.0f} ms; reaaliaikaan riittää alle {tavoite:.0f} ms")
    print("vaiheet (seinäkello, ms/ruutu): " + ", ".join(f"{k} {1e3 * v / len(ajat):.1f}" for k, v in sorted(an.aika.items(), key=lambda kv: -kv[1])))
    osuus = 100.0 * ajat.sum() / 1e3 / (len(ajat) / fps)
    print(f"analyysi vei {osuus:.0f} % videon kestosta (koko ajo {kesto:.0f} s, mukana videon luku" + (", virran lataus ja dekoodaus" if virta else "") + ")")
    if not an.heitot:
        print("-> Huom: jaksolla ei tunnistettu yhtään heittoa (ei heittoa jaksolla tai kalibrointikansio ei sovi videoon): raskas osa jäi mittaamatta, joten tulos on aliarvio. Valitse jakso, jolla on heitto.")
    if ajat.mean() > tavoite:
        print("-> analyysi EI ehdi reaaliaikaan tällä koneella: kuvan viive kasvaa. Kokeile --saikeet-arvoa, C++-sovitinta (netti_fit), --puskuri bgr tai sulje muita ohjelmia.")
    elif osuus > 60:
        print("-> analyysi ehtii niukasti: dekoodaus, piirto, tallennus ja ikkuna tarvitsevat lisäksi prosessoria, joten viive voi kasvaa raskailla jaksoilla.")
    else:
        print("-> analyysi ehtii reaaliaikaan selvästi.")


if __name__ == "__main__":
    main()
