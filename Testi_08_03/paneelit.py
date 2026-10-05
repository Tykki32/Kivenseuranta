"""Mainospaneelit kalibroinnin stabiloinnin ankkureina.

Kalibroinnin moodikuvan keruun aikana kamera stabiloidaan seuraamalla radan reunan paneeleja (vaalea suorakaide
tummalla pohjalla): jokaisen paneelin keskipiste haetaan edellisen ymparilta (track_panel), ja paneelien siirtymista
lasketaan stabilointimatriisi (seuranta.py). Paneelit tunnistetaan referenssitiedoston (*_panel_corners.txt) lahelta
automaattisesti (detect_panels_from_reference); puuttuvat voi klikata kasin (select_panels_manually).
"""
import os

import cv2
import numpy as np

# paneelin tunnistus (detect_panel)
WALL_OFFSET = 40                 # paneelin reunan kynnys: keskikohdan harmaasavy + tama
GRAY_RADIUS = 2                  # keskikohdan harmaasavyn keskiarvon sade (px)
MORPH_SIZE = 3
MIN_COMPONENT_AREA = 100
MAX_COMPONENT_AREA = 250000
SUBPIX_WINDOW = (5, 5)           # kulmien alipikselitarkennus
ROI_HALF_WIDTH = 100             # hakualue klikkauksen/edellisen paikan ymparilla (px)
ROI_HALF_HEIGHT = 120
PANEL_REFERENCE_SEARCH_SCALE = 10   # referenssipaneelin hakualue = ROI x tama


# ============================================================
# APUFUNKTIOT
# ============================================================


def get_gray_value(gray, x, y, radius=GRAY_RADIUS):
    h, w = gray.shape

    x = int(round(x))
    y = int(round(y))

    x1 = max(0, x - radius)
    x2 = min(w, x + radius + 1)
    y1 = max(0, y - radius)
    y2 = min(h, y + radius + 1)

    roi = gray[y1:y2, x1:x2]

    if roi.size == 0:
        return 0

    return float(np.median(roi))


def order_points(pts):
    pts = np.asarray(pts, dtype=np.float32)

    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()

    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]

    return rect


# ============================================================
# PANEELIN TUNNISTUS
# ============================================================


def detect_panel(gray, x, y, roi_half_width=None, roi_half_height=None):

    if roi_half_width is None:
        roi_half_width = ROI_HALF_WIDTH

    if roi_half_height is None:
        roi_half_height = ROI_HALF_HEIGHT

    h, w = gray.shape

    x = int(round(x))
    y = int(round(y))

    if x < 0 or x >= w or y < 0 or y >= h:
        return None

    center_gray = get_gray_value(gray, x, y)

    threshold = center_gray + WALL_OFFSET
    threshold = min(255, threshold)

    # --------------------------------------------------------
    # ROI
    # --------------------------------------------------------

    x1 = max(0, x - roi_half_width)

    x2 = min(w, x + roi_half_width + 1)

    y1 = max(0, y - roi_half_height)

    y2 = min(h, y + roi_half_height + 1)

    roi_gray = gray[y1:y2, x1:x2]

    # --------------------------------------------------------
    # THRESHOLD
    # --------------------------------------------------------

    roi_binary = (roi_gray < threshold).astype(np.uint8) * 255

    # --------------------------------------------------------
    # MORFOLOGIA
    # --------------------------------------------------------

    kernel = np.ones((MORPH_SIZE, MORPH_SIZE), dtype=np.uint8)

    roi_binary = cv2.morphologyEx(roi_binary, cv2.MORPH_OPEN, kernel)

    roi_binary = cv2.morphologyEx(roi_binary, cv2.MORPH_CLOSE, kernel)

    # --------------------------------------------------------
    # CONNECTED COMPONENTS
    #
    # TÄRKEÄÄ:
    # tehdään vain ROI:lle, ei koko 1920x1080-kuvalle.
    # --------------------------------------------------------

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(roi_binary, connectivity=8)

    if num_labels <= 1:
        return None

    # Keskipiste ROI:n koordinaateissa
    local_x = x - x1
    local_y = y - y1

    # --------------------------------------------------------
    # VALITSE KOMPONENTTI
    # --------------------------------------------------------

    if 0 <= local_y < labels.shape[0] and 0 <= local_x < labels.shape[1]:
        center_label = labels[local_y, local_x]
    else:
        center_label = 0

    if center_label != 0:

        label = int(center_label)

    else:

        best_label = None
        best_distance = float("inf")

        for i in range(1, num_labels):

            area = stats[i, cv2.CC_STAT_AREA]

            if area < MIN_COMPONENT_AREA:
                continue

            if area > MAX_COMPONENT_AREA:
                continue

            cx, cy = centroids[i]

            # Muutetaan ROI-koordinaateiksi
            distance = (cx - local_x) ** 2 + (cy - local_y) ** 2

            if distance < best_distance:

                best_distance = distance
                best_label = i

        if best_label is None:
            return None

        label = best_label

    # --------------------------------------------------------
    # TARKISTA KOMPONENTIN KOKO
    # --------------------------------------------------------

    area = stats[label, cv2.CC_STAT_AREA]

    if area < MIN_COMPONENT_AREA:
        return None

    if area > MAX_COMPONENT_AREA:
        return None

    # --------------------------------------------------------
    # KOMPONENTIN MASKI
    # --------------------------------------------------------

    component = np.uint8(labels == label) * 255

    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)

    if cv2.contourArea(contour) < MIN_COMPONENT_AREA:
        return None

    # --------------------------------------------------------
    # NELIÖ / MIN AREA RECT
    # --------------------------------------------------------

    epsilon = 0.02 * cv2.arcLength(contour, True)

    approx = cv2.approxPolyDP(contour, epsilon, True)

    if len(approx) == 4:

        corners = approx.reshape(4, 2).astype(np.float32)

    else:

        rect = cv2.minAreaRect(contour)

        corners = cv2.boxPoints(rect).astype(np.float32)

    # --------------------------------------------------------
    # SIIRRETÄÄN ROI-KOORDINAATEISTA
    # KOKO KUVAN KOORDINAATEIKSI
    # --------------------------------------------------------

    corners[:, 0] += x1
    corners[:, 1] += y1

    corners = order_points(corners)

    # --------------------------------------------------------
    # SUBPIKSELI
    # --------------------------------------------------------

    gray_float = np.ascontiguousarray(gray)

    try:

        refined = cv2.cornerSubPix(
            gray_float,
            corners.reshape(-1, 1, 2),
            SUBPIX_WINDOW,
            (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01),
        )

        if refined is not None:

            corners = refined.reshape(4, 2)

    except cv2.error:
        pass

    # --------------------------------------------------------
    # KESKIPISTE
    # --------------------------------------------------------

    center = np.mean(corners, axis=0)

    return {"corners": corners, "center": center}


# ============================================================
# PANEELIN SEURANTA
# ============================================================


def track_panel(gray, previous_center):
    return detect_panel(gray, previous_center[0], previous_center[1])


# ============================================================
# PANEELITIEDOSTON TALLENNUS
# ============================================================


def save_panel_data(panel_data, video_file):

    base = os.path.splitext(video_file)[0]

    filename = base + "_panel_corners.txt"

    with open(filename, "w", encoding="utf-8") as f:

        for i, panel in enumerate(panel_data):

            f.write(f"PANEL {i + 1}\n")

            for p in panel["corners"]:

                f.write(f"{p[0]:.6f} " f"{p[1]:.6f}\n")

            f.write(f"CENTER " f"{panel['center'][0]:.6f} " f"{panel['center'][1]:.6f}\n\n")

    return filename


# ============================================================
# PANEELITIEDOSTON LATAUS
#
# _parse_panel_corners_file on jaettu jasennyslogiikka - load_panel_
# data lataa TAMAN videon OMAN (aiemmin taman ohjelman tallentaman)
# tiedoston, load_panel_reference lataa TOISEN (referenssi-) videon
# tiedoston jota kaytetaan automaattitunnistuksen pohjana.
# ============================================================


def _parse_panel_corners_file(filename):

    if not os.path.exists(filename):
        return None

    with open(filename, "r", encoding="utf-8") as f:

        lines = [line.strip() for line in f if line.strip()]

    panels = []
    i = 0

    while i < len(lines):

        if not lines[i].startswith("PANEL"):
            i += 1
            continue

        i += 1

        corners = []

        for _ in range(4):

            if i >= len(lines):
                raise ValueError("Kulmapisteitä puuttuu.")

            parts = lines[i].split()

            if len(parts) != 2:
                raise ValueError("Virheellinen kulmapiste.")

            corners.append([float(parts[0]), float(parts[1])])

            i += 1

        if i >= len(lines):
            raise ValueError("CENTER puuttuu.")

        parts = lines[i].split()

        if len(parts) != 3 or parts[0] != "CENTER":
            raise ValueError("Virheellinen CENTER-rivi.")

        center = [float(parts[1]), float(parts[2])]

        i += 1

        panels.append(
            {"corners": np.asarray(corners, dtype=np.float32), "center": np.asarray(center, dtype=np.float32)}
        )

    return panels


def load_panel_data(video_file):

    base = os.path.splitext(video_file)[0]

    filename = base + "_panel_corners.txt"

    try:

        panels = _parse_panel_corners_file(filename)

        if panels is None or len(panels) < 2:
            return None

        print(f"Ladattu paneelitiedosto: " f"{filename}")

        print(f"Paneeleita: {len(panels)}")

        return panels

    except Exception as e:

        print(f"Paneelitiedoston lataus epäonnistui: " f"{e}")

        return None


def load_panel_reference(filename):
    """Lataa REFERENSSITIEDOSTON (kayttajan aiemmasta, samantyyppisesta
    kamera-asettelusta antama *_panel_corners.txt) automaattitunnistuksen
    pohjaksi - sama tiedostomuoto kuin load_panel_data/save_panel_data,
    mutta EKSPLISIITTINEN polku (ei johdeta nykyisen videon nimesta)."""

    panels = _parse_panel_corners_file(filename)

    if panels is None:
        raise RuntimeError(f"Referenssitiedostoa ei loytynyt: {filename}")

    if len(panels) < 2:
        raise RuntimeError(f"Referenssitiedostossa liian vahan paneeleita: {filename}")

    print(f"Ladattu referenssipaneelitiedosto: {filename}")
    print(f"Referenssipaneeleita: {len(panels)}")

    return panels


# ============================================================
# PANEELIEN KASIN-MERKINTA (VARAMENETTELY)
#
# Kayttajan pyynnosta: jos automaattitunnistus (referenssin lahelta)
# ei loyda vahintaan MIN_AUTO_PANELS_BEFORE_MANUAL (6) paneelia, se ei
# riita luotettavaan stabilointiin koko videon ajaksi - avataan siis
# skaalattavan kokoinen (cv2.WINDOW_NORMAL, kayttaja voi venyttaa
# ikkunan haluamaansa kokoon) ikkuna jossa kayttaja klikkaa PUUTTUVAT
# paneelit hiirella. Jo automaattisesti loydetyt paneelit nakyvat
# valmiiksi merkittyina (vihrea nelikulmio + jarjestysnumero), joten
# kayttajan tarvitsee klikata vain loput. Jokainen klikkaus ajaa
# saman detect_panel-tunnistuksen (tarkka kulma-/subpikselisovitus)
# klikkauskohdan ymparilta - sama menetelma kuin automaattitunnistus,
# vain hakupisteen ALKUPERA on kasin annettu. Enter lopettaa (vaatii
# vahintaan 2 paneelia yhteensa), Esc peruuttaa kokonaan.
# ============================================================

MIN_AUTO_PANELS_BEFORE_MANUAL = 6


def select_panels_manually(
    gray, frame_bgr=None, existing_panel_data=None, window_name="Klikkaa puuttuvat paneelit (Enter=valmis, Esc=peruuta)"
):

    display_base = frame_bgr if frame_bgr is not None else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    display_base = display_base.copy()

    panel_data = list(existing_panel_data) if existing_panel_data else []
    n_preexisting = len(panel_data)

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        result = detect_panel(gray, float(x), float(y))
        if result is None:
            print(
                f"  Ei loytynyt paneelia klikkauskohdasta ({x},{y}) - kokeile uudelleen (klikkaa lahempana paneelin keskikohtaa)."
            )
            return
        panel_data.append(result)
        print(f"  Paneeli {len(panel_data)} merkitty kohtaan ({x},{y}).")

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)

    print(
        f"  Avataan kasin-merkinta-ikkuna: {n_preexisting} paneelia jo "
        f"automaattisesti loydetty, klikkaa loput hiirella. Enter lopettaa "
        f"(vahintaan 2 paneelia yhteensa), Esc peruuttaa."
    )

    try:
        while True:
            display = display_base.copy()
            for i, result in enumerate(panel_data):
                corners = result["corners"].astype(int)
                color = (0, 200, 255) if i < n_preexisting else (0, 255, 0)
                cv2.polylines(display, [corners], True, color, 2)
                center = tuple(result["center"].astype(int))
                cv2.putText(display, str(i + 1), center, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(
                display,
                f"Paneeleita merkitty: {len(panel_data)} - Enter=valmis, Esc=peruuta",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )
            cv2.imshow(window_name, display)
            key = cv2.waitKey(20) & 0xFF

            if key in (13, 10):
                if len(panel_data) >= 2:
                    break
                print("  Vahintaan 2 paneelia tarvitaan ennen kuin voi jatkaa.")
            elif key == 27:
                raise RuntimeError("Kayttaja peruutti paneelien kasin-merkinnan.")
    finally:
        cv2.destroyWindow(window_name)

    print(f"  Paneelien kasin-merkinta valmis: {len(panel_data)} paneelia yhteensa.")
    return panel_data


# ============================================================
# PANEELIEN AUTOMAATTITUNNISTUS REFERENSSIN LAHELTA
#
# Kayttajan pyynnosta (korvaa kasin-klikkauksen select_panels): jokaista
# referenssipaneelia kohti etsitaan TODELLINEN paneeli sen keskipisteen
# LAHELTA - detect_panel:in oma ROI (ROI_HALF_WIDTH/HEIGHT) tarjoaa jo
# perustoleranssin; suurennamme sita viela PANEL_REFERENCE_SEARCH_
# SCALE:lla, koska kamera sijoitetaan/zoomataan kasin joka kerta hieman
# eri tavalla (kayttajan sanoin: "samantyylisesti samalla seudulla...
# hieman hajontaa"). Referenssin OMAT kulmat/koko eivat ole muuten
# merkityksellisia - vain keskipiste kaytetaan hakupisteena, koska
# detect_panel loytaa TODELLISEN muodon/koon uudelleen jokaiselle
# videolle erikseen (kameran zoomi voi muuttaa paneelin PIKSELIKOKOA).
# ============================================================


def detect_panels_from_reference(gray, reference_panels, frame_bgr=None, search_scale=PANEL_REFERENCE_SEARCH_SCALE):

    search_w = int(round(ROI_HALF_WIDTH * search_scale))
    search_h = int(round(ROI_HALF_HEIGHT * search_scale))

    panel_data = []
    missing = 0

    for idx, ref in enumerate(reference_panels):

        ref_center = ref["center"]

        result = detect_panel(
            gray, float(ref_center[0]), float(ref_center[1]), roi_half_width=search_w, roi_half_height=search_h
        )

        if result is None:

            missing += 1

            print(
                f"VAROITUS: paneelia {idx + 1} ei loytynyt "
                f"referenssikohdasta ({ref_center[0]:.1f}, "
                f"{ref_center[1]:.1f}) - ohitetaan."
            )

            continue

        panel_data.append(result)

    print(f"Paneelien automaattitunnistus: {len(panel_data)}/" f"{len(reference_panels)} loytyi ({missing} puuttuu).")

    if len(panel_data) < MIN_AUTO_PANELS_BEFORE_MANUAL:
        print(
            f"Automaattitunnistus loysi vain {len(panel_data)}/"
            f"{MIN_AUTO_PANELS_BEFORE_MANUAL} vaadittua paneelia - "
            f"avataan kasin-merkinta puuttuvien loytamiseksi."
        )
        panel_data = select_panels_manually(gray, frame_bgr, existing_panel_data=panel_data)

    return panel_data
