"""Lukee Sony HDR-CX405:n kuvaa Elgato Cam Linkin (UVC-laite) kautta.

Kayttö:
    python camlink_testi.py              # etsii Cam Linkin automaattisesti
    python camlink_testi.py 2            # kayttaa laitenumeroa 2
    python camlink_testi.py 0 --tallenna ulos.mp4

Nappaimet: q / ESC = lopeta, s = tallenna kuva.

Kamerassa: HDMI-ulostulo paalle ja kamera TOISTO-tilan sijaan kuvaustilaan
(Cam Link ottaa vastaan vain live-HDMI-signaalin). Aseta kamerasta
Asetukset > HDMI-tarkkuus = 1080i/1080p tai 720p ja "Naytteen tiedot" pois.
"""
import argparse
import sys
import time

import cv2


def avaa(indeksi):
    # Linuxissa V4L2, Windowsissa DirectShow (Cam Link toimii nailla parhaiten)
    if sys.platform.startswith("win"):
        backend = cv2.CAP_DSHOW
    elif sys.platform.startswith("linux"):
        backend = cv2.CAP_V4L2
    else:
        backend = cv2.CAP_ANY
    cap = cv2.VideoCapture(indeksi, backend)
    if not cap.isOpened():
        return None
    # Cam Link 4K antaa 1080p30 parhaiten MJPG/YUY2-muodossa
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)
    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        return None
    return cap


def etsi_laite():
    """Listaa toimivat laitteet; palauttaa ensimmaisen (yleensa Cam Link)."""
    loydetyt = []
    for i in range(8):
        cap = avaa(i)
        if cap is not None:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"  laite {i}: {w}x{h}")
            loydetyt.append(i)
            cap.release()
    return loydetyt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("laite", nargs="?", type=int, default=None)
    ap.add_argument("--tallenna", default=None, help="tallenna video tiedostoon")
    args = ap.parse_args()

    if args.laite is None:
        print("Etsitaan kameralaitteita...")
        laitteet = etsi_laite()
        if not laitteet:
            sys.exit("Yhtaan kuvaa antavaa laitetta ei loytynyt. Tarkista USB 3.0 -portti, "
                     "HDMI-kaapeli ja etta kamera on paalla kuvaustilassa.")
        print(f"Kaytetaan laitetta {laitteet[0]} (anna numero argumenttina jos se on vaara)")
        indeksi = laitteet[0]
    else:
        indeksi = args.laite

    cap = avaa(indeksi)
    if cap is None:
        sys.exit(f"Laitetta {indeksi} ei voitu avata.")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    print(f"Kuva: {w}x{h} @ {fps:.0f} fps")

    writer = None
    if args.tallenna:
        writer = cv2.VideoWriter(args.tallenna, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    n = 0
    t0 = time.time()
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
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        if k == ord("s"):
            nimi = f"kuva_{int(time.time())}.png"
            cv2.imwrite(nimi, frame)
            print("Tallennettu", nimi)

    print(f"{n} kuvaa, keskim. {n / max(time.time() - t0, 1e-6):.1f} fps")
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
