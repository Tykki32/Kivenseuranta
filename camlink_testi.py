"""Lukee Sony HDR-CX405:n kuvaa Elgato Cam Linkin (UVC-laite) kautta.

Kayttaa vain 1920x1080-kuvaa antavaa laitetta (muut, esim. koneen oma
webkamera 640x480, ohitetaan) ja tulostaa terminaaliin kameran tiedot.

Kayttö:
    python camlink_testi.py                       # etsii 1920x1080-laitteen
    python camlink_testi.py 2                     # pakottaa laitenumeron 2
    python camlink_testi.py --tallenna ulos.mp4

Nappaimet: q / ESC = lopeta, s = tallenna kuva.
"""
import argparse
import sys
import time

import cv2

LEVEYS, KORKEUS, FPS = 1920, 1080, 30

try:  # hiljenna OpenCV:n varoitukset olemattomista laitteista
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except Exception:
    pass


def backend():
    # Windowsissa DirectShow, Linuxissa V4L2
    if sys.platform.startswith("win"):
        return cv2.CAP_DSHOW, "DirectShow"
    if sys.platform.startswith("linux"):
        return cv2.CAP_V4L2, "V4L2"
    return cv2.CAP_ANY, "ANY"


def avaa(indeksi):
    """Avaa laitteen 1920x1080-tilassa; palauttaa None jos ei toimi tai kuva ei ole 1920x1080."""
    cap = cv2.VideoCapture(indeksi, backend()[0])
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, LEVEYS)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, KORKEUS)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    ok, frame = cap.read()
    if not ok or frame is None or frame.shape[1] != LEVEYS or frame.shape[0] != KORKEUS:
        cap.release()
        return None
    return cap


def etsi_laite():
    for i in range(8):
        cap = avaa(i)
        if cap is not None:
            return i, cap
    return None, None


def fourcc_teksti(arvo):
    arvo = int(arvo)
    teksti = "".join(chr((arvo >> (8 * i)) & 0xFF) for i in range(4))
    return teksti if teksti.isprintable() and arvo else "?"


def tulosta_tiedot(cap, indeksi, ensimmainen):
    nimi = backend()[1]
    h, w = ensimmainen.shape[:2]
    print("=" * 44)
    print("KAMERAN TIEDOT")
    print("=" * 44)
    print(f"Laitenumero      : {indeksi}")
    print(f"Taustajarjestelma: {nimi}")
    print(f"Tarkkuus (luettu): {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print(f"Ruudun koko      : {w}x{h}, kanavia {ensimmainen.shape[2]}, {ensimmainen.dtype}")
    print(f"FPS (ilmoitettu) : {cap.get(cv2.CAP_PROP_FPS):.2f}")
    print(f"Pakkausmuoto     : {fourcc_teksti(cap.get(cv2.CAP_PROP_FOURCC))}")
    print(f"Taustakoodaus    : {cap.getBackendName()}")
    for nimi_, prop in [("Kirkkaus", cv2.CAP_PROP_BRIGHTNESS), ("Kontrasti", cv2.CAP_PROP_CONTRAST),
                        ("Saturaatio", cv2.CAP_PROP_SATURATION), ("Valotus", cv2.CAP_PROP_EXPOSURE),
                        ("Tarkennus", cv2.CAP_PROP_FOCUS)]:
        v = cap.get(prop)
        if v not in (0.0, -1.0) or nimi_ in ("Kirkkaus", "Kontrasti"):
            print(f"{nimi_:<17}: {v}")
    print("=" * 44)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("laite", nargs="?", type=int, default=None)
    ap.add_argument("--tallenna", default=None, help="tallenna video tiedostoon")
    args = ap.parse_args()

    if args.laite is None:
        indeksi, cap = etsi_laite()
    else:
        indeksi, cap = args.laite, avaa(args.laite)
    if cap is None:
        sys.exit(f"{LEVEYS}x{KORKEUS}-kuvaa antavaa laitetta ei loytynyt. Tarkista USB 3.0, "
                 "HDMI-kaapeli ja etta kamera on paalla kuvaustilassa (HDMI-tarkkuus 1080).")

    ok, ensimmainen = cap.read()
    if ok:
        tulosta_tiedot(cap, indeksi, ensimmainen)
    fps = cap.get(cv2.CAP_PROP_FPS) or FPS

    writer = None
    if args.tallenna:
        writer = cv2.VideoWriter(args.tallenna, cv2.VideoWriter_fourcc(*"mp4v"), fps, (LEVEYS, KORKEUS))

    n = 0
    t0 = time.time()
    viimeinen = t0
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Kuvan luku epaonnistui")
            break
        n += 1

        # >>> TAHAN OMA KASITTELY, esim. kivenseuranta(frame) <<<

        if writer is not None:
            writer.write(frame)
        cv2.imshow("Cam Link", frame)

        nyt = time.time()
        if nyt - viimeinen >= 1.0:  # tilanne terminaaliin kerran sekunnissa
            print(f"\rkuva {n}  |  {n / (nyt - t0):.1f} fps  |  {time.strftime('%H:%M:%S')}", end="", flush=True)
            viimeinen = nyt

        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        if k == ord("s"):
            nimi = f"kuva_{int(time.time())}.png"
            cv2.imwrite(nimi, frame)
            print(f"\nTallennettu {nimi}")

    print(f"\n{n} kuvaa, keskim. {n / max(time.time() - t0, 1e-6):.1f} fps")
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
