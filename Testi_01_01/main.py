import os
import sys
import math
import csv
import time
import importlib.util
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from concurrent.futures import ThreadPoolExecutor
import mode_engine
from collections import deque


# ============================================================
# MUUT PROJEKTIN TIEDOSTOT (kamera8_01.py/kamera9_01.py/kamera9_04.py) -
# ladataan dynaamisesti Testi_01_01-kansion YLAPUOLELTA (Kivenseuranta-
# juurikansiosta), EI KOSKETA niita - sama periaate kuin kamera9_02.py
# -> kamera9_04.py -ketjussa tassa projektissa: uudelleenkaytetaan jo
# validoitua koodia (automaattinen kalibrointi, 3D-kiviprofiilin
# sovitus, nopeutettu yhteissovitus) sen sijaan etta kirjoitettaisiin
# se uudelleen.
# ============================================================

def _load_project_module(module_name, filename):
    here = os.path.dirname(os.path.abspath(__file__))
    parent = os.path.dirname(here)
    path = os.path.join(parent, filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


k94 = _load_project_module("k94_ref", "kamera9_04.py")
k93 = k94.k93
k92 = k94.k92
k9 = k94.k9
k8 = k94.k8


# ============================================================
# ASETUKSET
# ============================================================

ENABLE_MODE_FILTER = False

WALL_OFFSET = 40
GRAY_RADIUS = 2
HISTORY_LENGTH = 10
MORPH_SIZE = 3
MIN_COMPONENT_AREA = 100
MAX_COMPONENT_AREA = 250000
SUBPIX_WINDOW = (5, 5)

DISPLAY_WIDTH = 1400
DISPLAY_HEIGHT = 800

MAX_POSITION_ERROR = 20.0
REPORT_EVERY = 1000
MAX_WORKERS = 8

ROI_HALF_WIDTH = 100
ROI_HALF_HEIGHT = 120

MODE_DURATION_SECONDS = 120.0
MODE_FRAME_INTERVAL = 4

FILTER_DISTANCE_THRESHOLD = 20.0
STABILIZATION_MEDIAN_FRAMES = 20

# Moodikuvan laskenta tehdään paloissa
TILE_SIZE = 128

# ============================================================
# PANEELIEN AUTOMAATTITUNNISTUKSEN TOLERANSSI
#
# Kayttajan pyynnosta: paneelit ovat "samantyylisesti samalla
# seudulla", mutta kamera sijoitetaan/zoomataan aina hieman eri
# tavalla kasin, joten niiden sijainti VAIHTELEE hieman referenssiin
# nahden. detect_panel (alla) hakee jo omasta ROI:staan (ROI_HALF_
# WIDTH/HEIGHT = 100x120px) lahimman tummasuorakulmion annetun
# pisteen ymparilta - PANEL_REFERENCE_SEARCH_SCALE suurentaa taman
# haun VAIN referenssipohjaiselle automaattitunnistukselle (ei
# vaikuta paneiliSEURANTAAN framejen valilla, joka kayttaa detect_
# panel:ia normaalilla ROI:lla edellisen sijainnin ymparilta - kuvan
# drifti framejen valilla on paljon pienempi kuin kameran koko
# uudelleensijoittelun tuoma ero referenssiin).
# ============================================================

PANEL_REFERENCE_SEARCH_SCALE = 2.5

# ============================================================
# AUTOMAATTINEN KALIBROINTI - MOODIKUVA-POHJAINEN
#
# Kayttajan pyynnosta: sen sijaan etta kalibroidaan yhdesta
# raakaframesta (jossa voi olla liikkuvia kivia/pelaajia hairitsemassa
# lahemman pesan renkaiden tunnistusta), kerataan STABILOITUJA (mutta
# EI perspektiivikorjattuja - katso bootstrap_calibration:in kommentti
# MIKSI) frameja CALIB_MODE_SAMPLE_INTERVAL_SECONDS valein videon
# ensimmaisen CALIB_MODE_DURATION_SECONDS ajalta, ja lasketaan niista
# TILE_SIZE-kokoisittain moodikuva (olemassa oleva C++-mekanismi,
# sama add_mode_frame/start_mode_background/wait_for_mode kuin
# ENABLE_MODE_FILTER-ominaisuudessa) - liikkuvat kohteet haviavat,
# jaljelle jaa puhdas jaakuva jota vasten calibrate_camera_from_image
# (kamera9_01.py) toimii huomattavasti luotettavammin.
# ============================================================

CALIB_MODE_DURATION_SECONDS = 60.0
CALIB_MODE_SAMPLE_INTERVAL_SECONDS = 5.0

# ============================================================
# KOKO RADAN SKANNAUS LIIKKUVAN KIVEN LOYTAMISEKSI (3D-PROFIILIA
# VARTEN)
#
# Kayttajan pyynnosta: videon alussa kivi voi liikkua kumpaankin
# suuntaan eika sen sijaintia tiedeta etukateen - skannataan siis
# KOKO fyysinen rata (ei kamera9_02.py:n kiinteaa paata-rajattua
# HAKU-vyohyketta) STONE_SCAN_INTERVAL_SECONDS valein kunnes jotain
# loytyy - "ripea silloin kun vain odottaa kiveä" (kayttajan sanoin):
# harva aikavali pitaa taman odotusvaiheen halpana.
# ============================================================

STONE_SCAN_INTERVAL_SECONDS = 2.0

# HUOM: k9.find_stone_candidates suodattaa jo OLETUKSENA koko radan
# fyysisiin rajoihin (k8.OUTPUT_X/Y_MIN/MAX_CM + sheet_margin_cm) -
# ei tarvita omaa erillista rajausta, "koko rata" tulee ilmaiseksi.

# Kahden peräkkäisen skannauksen (STONE_SCAN_INTERVAL_SECONDS valein)
# valinen siirtyma, jonka ylittava sama (lahin) kandidaatti tulkitaan
# AIDOSTI LIIKKUVAKSI kiveksi (ei paikallaan olevaksi) - hidaskin
# liu'un loppuvaihe siirtyy enemman kuin tama 2 sekunnissa.
STONE_MOTION_THRESHOLD_CM = 15.0

# ============================================================
# 3D-PROFIILIN RIITTAVYYDEN VALIDOINTI
#
# Kayttajan pyynnosta: yksittainen loydetty/seurattu kivi ei aina
# riita hyvaan profiiliin (esim. kivi nakyy vain lyhyesti alussa) -
# sovitusta (k9.fit_stone_profile) YRITETAAN jokaisen uuden loydetyn
# kiven jalkeen KAIKKIEN TAHAN ASTI kerattyjen havaintojen paalla, ja
# hyvaksytaan vasta kun se on RIITTAVAN hyva nailla kynnysarvoilla -
# muuten jatketaan koko radan skannausta ja kerataan lisaa (mahdollisesti
# eri kivesta/hetkesta), YHDISTAEN havainnot samaan sovitukseen.
# ============================================================

PROFILE_MAX_RMS_PX = 5.0
PROFILE_MIN_SAMPLES = 15
PROFILE_SAMPLES_PER_STONE = 25

# Testatessa oikealla videolla loytyi KAKSI ongelmaa jotka nama
# kynnysarvot/mekanismit korjaavat:
#
#   1) find_stone_candidates hyvaksyy MYOS ei-kivia (kayttajan
#      dokumentoima aiempi havainto: "pelaajan vaatteet" jne. - katso
#      k9.find_stone_candidates:in kommentti) - LIIKKUVA ei-kivi
#      (esim. pelaaja/harja) voi lapaista liikkeentunnistuksen. Siksi
#      JOKAINEN seurattu kandidaatti tarkistetaan YKSINAAN (SOLO_TRACK_
#      MAX_RMS_PX) ENNEN kuin sen havainnot lisataan pysyvaan kokoelmaan
#      - jos yksinaankin jaannosvirhe on aivan liian suuri (ei nayta
#      kivelta), havainnot HYLATAAN eika saastuteta koko kokoelmaa
#      pysyvasti.
#   2) SAMA fyysinen kohde (oli se sitten kivi tai hylatty ei-kivi)
#      loydettiin uudelleen JOKA skannauksella niin kauan kuin se pysyi
#      liikkeessa (esim. kavelevaa/juokseva pelaaja useiden 2s-ikkunoiden
#      ajan) - HUOM: sijaintipohjainen jaahdytys EI RIITA, koska nopeasti
#      liikkuva kohde (kayttajan testivideolla havaittu, luultavasti
#      pelaaja, n. 200cm/2s) karkaa sijaintikynnyksen ulkopuolelle ennen
#      jaahdytysajan paattymista. Sen sijaan STONE_SCAN_COOLDOWN_FRAMES
#      estaa uuden kalliin taydellisen seurannan koko taman monta framea
#      kattavalla IKKUNALLA viimeisimman yrityksen (kivi tai hylatty)
#      SIEMENFRAMESTA riippumatta kohteen nopeudesta - karkea mutta
#      luotettava nopeusrajoitin kalliille yrityksille.
SOLO_TRACK_MAX_RMS_PX = 6.0
STONE_SCAN_COOLDOWN_FRAMES = 750   # 30s 25fps:lla

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

    center_gray = get_gray_value(
        gray,
        x,
        y
    )

    threshold = center_gray + WALL_OFFSET
    threshold = min(
        255,
        threshold
    )

    # --------------------------------------------------------
    # ROI
    # --------------------------------------------------------

    x1 = max(
        0,
        x - roi_half_width
    )

    x2 = min(
        w,
        x + roi_half_width + 1
    )

    y1 = max(
        0,
        y - roi_half_height
    )

    y2 = min(
        h,
        y + roi_half_height + 1
    )

    roi_gray = gray[
        y1:y2,
        x1:x2
    ]

    # --------------------------------------------------------
    # THRESHOLD
    # --------------------------------------------------------

    roi_binary = (
        roi_gray < threshold
    ).astype(
        np.uint8
    ) * 255

    # --------------------------------------------------------
    # MORFOLOGIA
    # --------------------------------------------------------

    kernel = np.ones(
        (
            MORPH_SIZE,
            MORPH_SIZE
        ),
        dtype=np.uint8
    )

    roi_binary = cv2.morphologyEx(
        roi_binary,
        cv2.MORPH_OPEN,
        kernel
    )

    roi_binary = cv2.morphologyEx(
        roi_binary,
        cv2.MORPH_CLOSE,
        kernel
    )

    # --------------------------------------------------------
    # CONNECTED COMPONENTS
    #
    # TÄRKEÄÄ:
    # tehdään vain ROI:lle, ei koko 1920x1080-kuvalle.
    # --------------------------------------------------------

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            roi_binary,
            connectivity=8
        )
    )

    if num_labels <= 1:
        return None

    # Keskipiste ROI:n koordinaateissa
    local_x = x - x1
    local_y = y - y1

    # --------------------------------------------------------
    # VALITSE KOMPONENTTI
    # --------------------------------------------------------

    if (
        0 <= local_y < labels.shape[0]
        and
        0 <= local_x < labels.shape[1]
    ):
        center_label = labels[
            local_y,
            local_x
        ]
    else:
        center_label = 0

    if center_label != 0:

        label = int(
            center_label
        )

    else:

        best_label = None
        best_distance = float("inf")

        for i in range(
            1,
            num_labels
        ):

            area = stats[
                i,
                cv2.CC_STAT_AREA
            ]

            if area < MIN_COMPONENT_AREA:
                continue

            if area > MAX_COMPONENT_AREA:
                continue

            cx, cy = centroids[i]

            # Muutetaan ROI-koordinaateiksi
            distance = (
                (cx - local_x) ** 2 +
                (cy - local_y) ** 2
            )

            if distance < best_distance:

                best_distance = distance
                best_label = i

        if best_label is None:
            return None

        label = best_label

    # --------------------------------------------------------
    # TARKISTA KOMPONENTIN KOKO
    # --------------------------------------------------------

    area = stats[
        label,
        cv2.CC_STAT_AREA
    ]

    if area < MIN_COMPONENT_AREA:
        return None

    if area > MAX_COMPONENT_AREA:
        return None

    # --------------------------------------------------------
    # KOMPONENTIN MASKI
    # --------------------------------------------------------

    component = np.uint8(
        labels == label
    ) * 255

    contours, _ = cv2.findContours(
        component,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    contour = max(
        contours,
        key=cv2.contourArea
    )

    if (
        cv2.contourArea(contour)
        < MIN_COMPONENT_AREA
    ):
        return None

    # --------------------------------------------------------
    # NELIÖ / MIN AREA RECT
    # --------------------------------------------------------

    epsilon = (
        0.02 *
        cv2.arcLength(
            contour,
            True
        )
    )

    approx = cv2.approxPolyDP(
        contour,
        epsilon,
        True
    )

    if len(approx) == 4:

        corners = (
            approx.reshape(
                4,
                2
            ).astype(
                np.float32
            )
        )

    else:

        rect = cv2.minAreaRect(
            contour
        )

        corners = (
            cv2.boxPoints(
                rect
            ).astype(
                np.float32
            )
        )

    # --------------------------------------------------------
    # SIIRRETÄÄN ROI-KOORDINAATEISTA
    # KOKO KUVAN KOORDINAATEIKSI
    # --------------------------------------------------------

    corners[:, 0] += x1
    corners[:, 1] += y1

    corners = order_points(
        corners
    )

    # --------------------------------------------------------
    # SUBPIKSELI
    # --------------------------------------------------------

    gray_float = np.ascontiguousarray(
        gray
    )

    try:

        refined = cv2.cornerSubPix(
            gray_float,
            corners.reshape(
                -1,
                1,
                2
            ),
            SUBPIX_WINDOW,
            (-1, -1),
            (
                cv2.TERM_CRITERIA_EPS +
                cv2.TERM_CRITERIA_MAX_ITER,
                30,
                0.01
            )
        )

        if refined is not None:

            corners = refined.reshape(
                4,
                2
            )

    except cv2.error:
        pass

    # --------------------------------------------------------
    # KESKIPISTE
    # --------------------------------------------------------

    center = np.mean(
        corners,
        axis=0
    )

    return {
        "corners": corners,
        "center": center
    }


# ============================================================
# PANEELIN SEURANTA
# ============================================================

def track_panel(gray, previous_center):
    return detect_panel(
        gray,
        previous_center[0],
        previous_center[1]
    )


# ============================================================
# PANEELITIEDOSTON TALLENNUS
# ============================================================

def save_panel_data(
    panel_data,
    video_file
):

    base = os.path.splitext(
        video_file
    )[0]

    filename = (
        base +
        "_panel_corners.txt"
    )

    with open(
        filename,
        "w",
        encoding="utf-8"
    ) as f:

        for i, panel in enumerate(
            panel_data
        ):

            f.write(
                f"PANEL {i + 1}\n"
            )

            for p in panel["corners"]:

                f.write(
                    f"{p[0]:.6f} "
                    f"{p[1]:.6f}\n"
                )

            f.write(
                f"CENTER "
                f"{panel['center'][0]:.6f} "
                f"{panel['center'][1]:.6f}\n\n"
            )

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

    with open(
        filename,
        "r",
        encoding="utf-8"
    ) as f:

        lines = [
            line.strip()
            for line in f
            if line.strip()
        ]

    panels = []
    i = 0

    while i < len(lines):

        if not lines[i].startswith(
            "PANEL"
        ):
            i += 1
            continue

        i += 1

        corners = []

        for _ in range(4):

            if i >= len(lines):
                raise ValueError(
                    "Kulmapisteitä puuttuu."
                )

            parts = lines[i].split()

            if len(parts) != 2:
                raise ValueError(
                    "Virheellinen kulmapiste."
                )

            corners.append(
                [
                    float(parts[0]),
                    float(parts[1])
                ]
            )

            i += 1

        if i >= len(lines):
            raise ValueError(
                "CENTER puuttuu."
            )

        parts = lines[i].split()

        if (
            len(parts) != 3
            or parts[0] != "CENTER"
        ):
            raise ValueError(
                "Virheellinen CENTER-rivi."
            )

        center = [
            float(parts[1]),
            float(parts[2])
        ]

        i += 1

        panels.append(
            {
                "corners": np.asarray(
                    corners,
                    dtype=np.float32
                ),
                "center": np.asarray(
                    center,
                    dtype=np.float32
                )
            }
        )

    return panels


def load_panel_data(video_file):

    base = os.path.splitext(
        video_file
    )[0]

    filename = (
        base +
        "_panel_corners.txt"
    )

    try:

        panels = _parse_panel_corners_file(filename)

        if panels is None or len(panels) < 2:
            return None

        print(
            f"Ladattu paneelitiedosto: "
            f"{filename}"
        )

        print(
            f"Paneeleita: {len(panels)}"
        )

        return panels

    except Exception as e:

        print(
            f"Paneelitiedoston lataus epäonnistui: "
            f"{e}"
        )

        return None


def load_panel_reference(filename):
    """Lataa REFERENSSITIEDOSTON (kayttajan aiemmasta, samantyyppisesta
    kamera-asettelusta antama *_panel_corners.txt) automaattitunnistuksen
    pohjaksi - sama tiedostomuoto kuin load_panel_data/save_panel_data,
    mutta EKSPLISIITTINEN polku (ei johdeta nykyisen videon nimesta)."""

    panels = _parse_panel_corners_file(filename)

    if panels is None:
        raise RuntimeError(
            f"Referenssitiedostoa ei loytynyt: {filename}"
        )

    if len(panels) < 2:
        raise RuntimeError(
            f"Referenssitiedostossa liian vahan paneeleita: {filename}"
        )

    print(f"Ladattu referenssipaneelitiedosto: {filename}")
    print(f"Referenssipaneeleita: {len(panels)}")

    return panels


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

def detect_panels_from_reference(gray, reference_panels,
                                  search_scale=PANEL_REFERENCE_SEARCH_SCALE):

    search_w = int(round(ROI_HALF_WIDTH * search_scale))
    search_h = int(round(ROI_HALF_HEIGHT * search_scale))

    panel_data = []
    missing = 0

    for idx, ref in enumerate(reference_panels):

        ref_center = ref["center"]

        result = detect_panel(
            gray,
            float(ref_center[0]),
            float(ref_center[1]),
            roi_half_width=search_w,
            roi_half_height=search_h
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

    print(
        f"Paneelien automaattitunnistus: {len(panel_data)}/"
        f"{len(reference_panels)} loytyi ({missing} puuttuu)."
    )

    if len(panel_data) < 2:
        raise RuntimeError(
            "Liian vahan paneeleita loytyi automaattisesti "
            "(vahintaan 2 tarvitaan stabilointiin)."
        )

    return panel_data


# ============================================================
# KOKO RADAN KIVIEHDOKKAIDEN HAKU + LIIKKEEN TUNNISTUS
#
# _scan_stone_candidates: sama segmentointipohjainen tunnistus kuin
# k9.track_stone_in_video/track_stone_in_video_fast kayttavat SISAISESTI
# (k9._candidates_in_frame) - EI mallipohjaista ristikkohakua, koska
# tassa vaiheessa 3D-profiilia (jota malli tarvitsisi) EI VIELA OLE -
# se on juuri se mita etsitaan.
#
# find_moving_candidate: vertaa KAHDEN PERAKKAISEN skannauksen (n.
# STONE_SCAN_INTERVAL_SECONDS valein) kandidaattilistoja - sama fyysinen
# kivi (lahin osuma) jonka sijainti on muuttunut enemman kuin STONE_
# MOTION_THRESHOLD_CM tulkitaan AIDOSTI LIIKKUVAKSI (ei jo-paikallaan-
# olevaksi) - tama siirtymä-havainto ANTAA SIEMENEN (frame_idx, X, Y)
# k94.track_stone_in_video_fast:lle, joka sitten seuraa koko liu'un.
# ============================================================

def _scan_stone_candidates(frame_bgr, calib, pose):
    return k9._candidates_in_frame(
        frame_bgr, calib, pose, calib["H_final"],
        k9.STONE_TRACK_MIN_AREA, k9.STONE_TRACK_MIN_FILL_RATIO,
        k9.STONE_TRACK_MIN_ASPECT_RATIO
    )


def find_moving_candidate(candidates_prev, candidates_curr,
                           motion_threshold_cm=STONE_MOTION_THRESHOLD_CM):

    if not candidates_prev or not candidates_curr:
        return None

    for curr in candidates_curr:

        nearest = min(
            candidates_prev,
            key=lambda p: math.hypot(
                p["pos_cm"][0] - curr["pos_cm"][0],
                p["pos_cm"][1] - curr["pos_cm"][1]
            )
        )

        d = math.hypot(
            nearest["pos_cm"][0] - curr["pos_cm"][0],
            nearest["pos_cm"][1] - curr["pos_cm"][1]
        )

        if d >= motion_threshold_cm:
            return curr["pos_cm"]

    return None


# ============================================================
# AIKAIKKUNAAN RAJATTU KIVEN SEURANTA (EI koko videota)
#
# k94.track_stone_in_video_fast (kamera9_04.py, EI kosketa) skannaa
# TARKOITUKSELLA koko videon jokaisen framen - jarkevaa SEN omassa
# kayttotarkoituksessaan (kertaluontoinen kalibrointi lyhyella
# videolla). Tassa TIEDOSSA jo ON karkea siemen (koko radan skannaus
# loysi sen), joten koko videon uudelleenskannaus JOKAISELLA
# yrityksella (havaittu testatessa: n. 700s per yritys 7500 framen
# videolla) olisi kohtuuttoman hidas, varsinkin jos useampi kandidaatti
# (esim. pelaaja) hylataan ennen aidon kiven loytymista. Tama funktio
# rajaa saman algoritmin (sama jatkuvuuslogiikka, samat kynnysarvot)
# STONE_TRACK_WINDOW_SECONDS-ikkunaan siemenen ymparilta - riittava
# kattamaan koko liu'un (kivi ei ole jaalla montaa kymmenta sekuntia
# kerrallaan) mutta paljon halvempi kuin koko video.
# ============================================================

STONE_TRACK_WINDOW_SECONDS = 30.0


def track_stone_in_video_windowed(video_path, calib, pose, seed_frame_idx,
                                   seed_pos_cm, window_seconds=STONE_TRACK_WINDOW_SECONDS):

    max_jump_cm = k9.STONE_TRACK_MAX_JUMP_CM
    max_misses = k9.STONE_TRACK_MAX_MISSES
    min_area = k9.STONE_TRACK_MIN_AREA
    min_fill_ratio = k9.STONE_TRACK_MIN_FILL_RATIO
    min_aspect_ratio = k9.STONE_TRACK_MIN_ASPECT_RATIO

    H_final = calib["H_final"]
    camera_matrix = calib["camera_matrix"]
    dist_coeffs = np.array(
        [calib["best_k1"], 0.0, 0.0, 0.0, 0.0], dtype=np.float64
    )

    cap = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    map1, map2 = k94._build_undistort_maps(
        camera_matrix, dist_coeffs, (frame_w, frame_h)
    )

    window_frames = int(round(window_seconds * fps))
    start_frame = max(0, seed_frame_idx - window_frames)
    end_frame = min(n_frames - 1, seed_frame_idx + window_frames)

    # YKSI haku ikkunan alkuun (ei per-frame haku - katso kamera9_04.py:n
    # kommentti seek:in hitaudesta/epatarkkuudesta), sitten sekvenssiluku
    # ikkunan yli - taman kayttotarkoituksen (karkean siemenen tarkka
    # seuranta) ei tarvitse olla framen tarkka, toisin kuin lopullinen
    # kalibrointi.
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    candidates_cache = {}
    idx = start_frame

    while idx <= end_frame:

        ok, frame = cap.read()

        if not ok:
            break

        frame_u = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)

        candidates_cache[idx] = k94._candidates_in_frame_fast(
            frame_u, pose, H_final, min_area, min_fill_ratio, min_aspect_ratio
        )

        idx += 1

    cap.release()

    def candidates_at(i):
        return candidates_cache.get(i, [])

    seed_cands = candidates_at(seed_frame_idx)

    if not seed_cands:
        return []

    seed = min(
        seed_cands,
        key=lambda c: math.hypot(
            c["pos_cm"][0] - seed_pos_cm[0], c["pos_cm"][1] - seed_pos_cm[1]
        )
    )
    seed["frame_idx"] = seed_frame_idx

    def track_direction(step):

        track = []
        last_pos = seed["pos_cm"]
        misses = 0
        idx = seed_frame_idx + step

        while start_frame <= idx <= end_frame and misses < max_misses:

            cands = candidates_at(idx)

            if cands:

                best = min(
                    cands,
                    key=lambda c: math.hypot(
                        c["pos_cm"][0] - last_pos[0], c["pos_cm"][1] - last_pos[1]
                    )
                )

                d = math.hypot(
                    best["pos_cm"][0] - last_pos[0], best["pos_cm"][1] - last_pos[1]
                )

                if d <= max_jump_cm:
                    best["frame_idx"] = idx
                    track.append(best)
                    last_pos = best["pos_cm"]
                    misses = 0
                else:
                    misses += 1
            else:
                misses += 1

            idx += step

        return track

    backward = track_direction(-1)
    forward = track_direction(+1)

    full_track = list(reversed(backward)) + [seed] + forward
    full_track.sort(key=lambda t: t["frame_idx"])

    return full_track


def try_fit_profile(pose, stones, max_rms_px=PROFILE_MAX_RMS_PX,
                     min_samples=PROFILE_MIN_SAMPLES, label="profiilikoe"):
    """Yrittaa sovittaa 3D-profiilin annettuihin havaintoihin - palauttaa
    (profile, riittava_bool). Riittavyys: min_samples verran havaintoja
    JA jaannosvirhe max_rms_px:n sisalla. Kaytetaan SEKA yksittaisen
    seuratun kiven OMAN kelvollisuuden tarkistukseen (katso taman
    tiedoston alkupaan kommentti SOLO_TRACK_MAX_RMS_PX:sta) etta koko
    kerätyn kokoelman lopulliseen riittavyystarkistukseen."""

    if len(stones) < min_samples:
        return None, False

    profile = k9.fit_stone_profile(pose, stones)

    riittava = profile["residual_rms_px"] <= max_rms_px

    print(
        f"  {label}: {len(stones)} havaintoa, "
        f"RMS={profile['residual_rms_px']:.2f}px "
        f"(kynnys {max_rms_px}px) -> "
        f"{'RIITTAVA' if riittava else 'ei riittava'}"
    )

    return profile, riittava


# ============================================================
# YKSI PAAPUTKI - kaikki vaiheet (kalibrointi -> kiviprofiilin haku ->
# elava moni-kiven seuranta) jakavat SAMAN videon peräkkäisen luvun ja
# SAMAN joka-frame paneiliseurannan/stabiloinnin (kayttajan pyynnosta
# tama on hidas, tasainen driftinseuranta - katso stabilointimatriisin
# mediaanisuodatuksen kommentti alla, EI kosketa sita). Vaiheet
# etenevat sisainen tila (calib_result, profile_result, jne.)
# perusteella - kukin vaihe kaynnistyy vasta edellisen valmistuttua.
# ============================================================

def run_pipeline(
    video_file,
    panel_data,
    calib_diag_output
):

    engine = mode_engine.ModeEngine(
        video_file,
        TILE_SIZE
    )

    width = engine.width()
    height = engine.height()
    fps = engine.fps()
    total_frames = engine.total_frames()

    print()
    print(
        f"Resoluutio: "
        f"{width} x {height}"
    )

    print(
        f"Ruutuja: {total_frames}"
    )

    print(
        f"FPS: {fps:.3f}"
    )

    # ========================================================
    # AUTOMAATTISEN KALIBROINNIN MOODINAYTTEENOTON AJASTUS
    #
    # HUOM: engine.set_perspective_matrix EI kutsuta - perspective_
    # enabled_ pysyy C++:ssa false:na, joten add_mode_frame tuottaa
    # VAIN stabiloidun (EI ylhaaltapain-warpatun) framen. calibrate_
    # camera_from_image TARVITSEE juuri taman - se ratkaisee itse
    # vaantokertoimen+homografian RAAasta/vaantyneesta kuvasta, joten
    # syotteen ei pida olla jo perspektiivikorjattu.
    # ========================================================

    calib_mode_max_frames = int(
        round(fps * CALIB_MODE_DURATION_SECONDS)
    )

    calib_sample_every = max(
        1, int(round(fps * CALIB_MODE_SAMPLE_INTERVAL_SECONDS))
    )

    print(
        f"Kalibroinnin moodikuvaan kerataan naytteet videon "
        f"ensimmaisen {CALIB_MODE_DURATION_SECONDS:.0f} sekunnin "
        f"ajalta, yksi naytepiste {CALIB_MODE_SAMPLE_INTERVAL_SECONDS:.0f} "
        f"sekunnin valein (enintaan "
        f"{calib_mode_max_frames // calib_sample_every + 1} naytetta)."
    )

    reference_centers = [
        np.asarray(
            panel["center"],
            dtype=np.float32
        ).copy()
        for panel in panel_data
    ]

    histories = [
        [center.copy()]
        for center in reference_centers
    ]

    frame_index = 0
    next_calib_sample_frame = 0

    calib_result = None

    stone_scan_interval_frames = max(
        1, int(round(fps * STONE_SCAN_INTERVAL_SECONDS))
    )
    next_stone_scan_frame = 0
    prev_scan_candidates = None
    accumulated_stones = []
    profile_result = None
    next_allowed_scan_track_frame = 0

    previous_stabilization_matrix = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ],
        dtype=np.float64
    )
    stabilization_history = deque(
        maxlen=STABILIZATION_MEDIAN_FRAMES
    )

    start_time = time.time()

    total_gray_time = 0.0
    total_tracking_time = 0.0
    total_transform_time = 0.0
    total_sample_time = 0.0

    executor = ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    )

    try:

        while True:

            frame = engine.read()

            if frame is None or frame.size == 0:
                break
            t0 = time.perf_counter()
            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )
            t1 = time.perf_counter()
            total_gray_time += t1 - t0

            # ------------------------------------------------
            # PANEELIEN SEURANTA
            # ------------------------------------------------

            t2 = time.perf_counter()
            futures = []

            for panel_index in range(
                len(panel_data)
            ):

                if histories[panel_index]:

                    previous_center = (
                        histories[
                            panel_index
                        ][-1]
                    )

                else:

                    previous_center = (
                        reference_centers[
                            panel_index
                        ]
                    )

                futures.append(
                    executor.submit(
                        track_panel,
                        gray,
                        previous_center
                    )
                )

            current_centers = []

            for panel_index, future in enumerate(
                futures
            ):

                result = future.result()

                if result is None:

                    current_centers.append(
                        None
                    )

                    continue

                center = np.asarray(
                    result["center"],
                    dtype=np.float32
                )

                previous_center = (
                    histories[
                        panel_index
                    ][-1]
                    if histories[
                        panel_index
                    ]
                    else reference_centers[
                        panel_index
                    ]
                )

                error = np.linalg.norm(
                    center -
                    previous_center
                )

                if error <= MAX_POSITION_ERROR:

                    histories[
                        panel_index
                    ].append(
                        center.copy()
                    )

                    if len(
                        histories[
                            panel_index
                        ]
                    ) > HISTORY_LENGTH:

                        histories[
                            panel_index
                        ].pop(0)

                    current_centers.append(
                        center
                    )

                else:

                    current_centers.append(
                        None
                    )

            t3 = time.perf_counter()
            total_tracking_time += t3 - t2
            # ------------------------------------------------
            # STABILOINTIMATRIISI
            # ------------------------------------------------

            valid_reference = []
            valid_current = []

            for i in range(
                len(reference_centers)
            ):

                if current_centers[i] is not None:

                    valid_reference.append(
                        reference_centers[i]
                    )

                    valid_current.append(
                        current_centers[i]
                    )

            stabilization_matrix = (
                previous_stabilization_matrix.copy()
            )

            if len(valid_reference) >= 2:

                reference = np.asarray(
                    valid_reference,
                    dtype=np.float32
                )

                current = np.asarray(
                    valid_current,
                    dtype=np.float32
                )

                M, inliers = (
                    cv2.estimateAffinePartial2D(
                        reference,
                        current,
                        method=cv2.RANSAC,
                        ransacReprojThreshold=3.0,
                        maxIters=2000,
                        confidence=0.99
                    )
                )

                if M is not None:

                    stabilization_matrix = (
                        cv2.invertAffineTransform(
                            M
                        )
                    )

            # ------------------------------------------------
            # STABILOINNIN MEDIAANISUODATUS
            #
            # Käytetään peräkkäisten ruutujen mediaania.
            # Tämä poistaa pientä värinää, mutta seuraa
            # edelleen hidasta driftia.
            # ------------------------------------------------

            stabilization_history.append(
                stabilization_matrix.copy()
            )

            stabilization_matrix = np.median(
                np.stack(
                    stabilization_history,
                    axis=0
                ),
                axis=0
            )

            stabilization_matrix = np.asarray(
                stabilization_matrix,
                dtype=np.float64
            )


            previous_stabilization_matrix = (
                stabilization_matrix.copy()
            )

            # ------------------------------------------------
            # LÄHETÄ STABILOINNIN MATRIX C++:LLE
            # ------------------------------------------------
            t4 = time.perf_counter()
            engine.set_transform(
                stabilization_matrix
            )
            t5 = time.perf_counter()
            total_transform_time += t5 - t4

            # ------------------------------------------------
            # AUTOMAATTISEN KALIBROINNIN MOODINAYTTEET
            #
            # (katso taman funktion alkupaan kommentti MIKSI
            # set_perspective_matrix:ia ei kutsuta - add_mode_frame
            # tuottaa siis stabiloidun mutta MUUTEN raa'an framen).
            # ------------------------------------------------

            if calib_result is None:

                if (
                    frame_index < calib_mode_max_frames
                    and frame_index >= next_calib_sample_frame
                ):

                    t6 = time.perf_counter()

                    engine.add_mode_frame()

                    t7 = time.perf_counter()

                    total_sample_time += (
                        t7 - t6
                    )

                    next_calib_sample_frame += calib_sample_every

                # --------------------------------------------
                # Kun kalibroinnin moodinaytteet on kerätty,
                # lasketaan moodikuva JA kalibroidaan siita -
                # tama on KERTALUONTEINEN tauko (moodikuvan
                # lasku + calibrate_camera_from_image/build_
                # pose_from_calibration kestavat yhteensa
                # kymmenia sekunteja) - videon lukua EI jatketa
                # taman valmistumisen aikana, mika on hyvaksyttavaa
                # koska taman vaiheen ei tarvitse olla reaaliaikainen
                # (vain elava kiven seuranta myohemmin vaatii sen).
                # --------------------------------------------

                if (
                    next_calib_sample_frame >= calib_mode_max_frames
                ):

                    print()
                    print(
                        "Kalibroinnin moodinaytteet ovat kasassa "
                        f"({engine.mode_frame_count()} kpl). "
                        "Lasketaan moodikuva..."
                    )

                    calib_mode_output = (
                        os.path.splitext(video_file)[0] +
                        "_kalibrointi_moodikuva.png"
                    )

                    engine.start_mode_background(
                        calib_mode_output
                    )

                    engine.wait_for_mode()

                    print(
                        f"Moodikuva valmis: {calib_mode_output}"
                    )

                    print(
                        "Kalibroidaan moodikuvasta "
                        "(calibrate_camera_from_image + "
                        "build_pose_from_calibration)..."
                    )

                    calib = k9.calibrate_camera_from_image(
                        calib_mode_output
                    )

                    pose = k9.build_pose_from_calibration(calib)

                    print(
                        f"  fokaalivali f = {pose['K'][0, 0]:.1f} px, "
                        f"kameran sijainti (cm): "
                        f"{np.round(pose['camera_position_cm'], 1)}"
                    )

                    print(
                        f"  lahemman pesan jaannosvirhe (px): "
                        f"keskiarvo="
                        f"{pose['near_reproj_err_px']['mean']:.2f}, "
                        f"max={pose['near_reproj_err_px']['max']:.2f}"
                    )

                    topdown = cv2.warpPerspective(
                        calib["frame_undistorted"],
                        calib["H_final"],
                        (calib["output_w"], calib["output_h"])
                    )

                    cv2.imwrite(calib_diag_output, topdown)

                    print(
                        f"Kalibroinnin topdown-tarkistuskuva: "
                        f"{calib_diag_output}"
                    )

                    calib_result = {"calib": calib, "pose": pose}

            # ------------------------------------------------
            # KOKO RADAN SKANNAUS LIIKKUVAN KIVEN LOYTAMISEKSI
            # (3D-PROFIILIA VARTEN) - vasta kun kalibrointi on
            # valmis (tarvitaan calib/pose fyysisten sijaintien
            # laskentaan) JA profiili ei viela ole riittava.
            # ------------------------------------------------

            elif profile_result is None:

                if frame_index >= next_stone_scan_frame:

                    curr_candidates = _scan_stone_candidates(
                        frame, calib_result["calib"], calib_result["pose"]
                    )

                    seed_pos = find_moving_candidate(
                        prev_scan_candidates, curr_candidates
                    )

                    # ------------------------------------------
                    # JAAHDYTYS: katso taman tiedoston alkupaan
                    # kommentti MIKSI taman on oltava frame-ikkuna-
                    # pohjainen (ei sijaintipohjainen) - nopeasti
                    # liikkuva ei-kivi karkaisi sijaintikynnyksesta.
                    # ------------------------------------------

                    if frame_index < next_allowed_scan_track_frame:
                        seed_pos = None

                    if seed_pos is not None:

                        print()
                        print(
                            f"[frame {frame_index}] Liikkuva kandidaatti "
                            f"loytyi kohdasta ({seed_pos[0]:.1f}, "
                            f"{seed_pos[1]:.1f}) cm - seurataan koko "
                            f"liu'un ajan..."
                        )

                        track = track_stone_in_video_windowed(
                            video_file, calib_result["calib"],
                            calib_result["pose"],
                            seed_frame_idx=frame_index,
                            seed_pos_cm=seed_pos
                        )

                        print(
                            f"  seuranta valmis: {len(track)} havaintoa."
                        )

                        next_allowed_scan_track_frame = (
                            frame_index + STONE_SCAN_COOLDOWN_FRAMES
                        )

                        if len(track) >= 2:

                            idxs = sorted(set(
                                np.linspace(
                                    0, len(track) - 1,
                                    PROFILE_SAMPLES_PER_STONE
                                ).astype(int).tolist()
                            ))

                            candidate_observations = [
                                {
                                    "ellipse": track[i]["ellipse"],
                                    "contour": track[i]["contour"],
                                }
                                for i in idxs
                            ]

                            # --------------------------------
                            # YKSINAINEN KELVOLLISUUSTARKISTUS
                            # ENNEN pysyvaan kokoelmaan lisaamista
                            # (katso taman tiedoston alkupaan
                            # kommentti - estaa ei-kivien, esim.
                            # pelaajien, saastuttamasta koko
                            # kokoelmaa pysyvasti).
                            # --------------------------------

                            _, solo_ok = try_fit_profile(
                                calib_result["pose"],
                                candidate_observations,
                                max_rms_px=SOLO_TRACK_MAX_RMS_PX,
                                label="  yksittaisen kandidaatin tarkistus"
                            )

                            if not solo_ok:

                                print(
                                    "  hylatty - ei nayta kivelta "
                                    "(esim. pelaaja/muu liikkuva "
                                    "kohde), ei lisata kokoelmaan."
                                )

                            else:

                                accumulated_stones.extend(
                                    candidate_observations
                                )

                                profile, riittava = try_fit_profile(
                                    calib_result["pose"],
                                    accumulated_stones
                                )

                                if riittava:

                                    print(
                                        "3D-kiviprofiili riittava - "
                                        "lopetetaan koko radan "
                                        "skannaus."
                                    )

                                    profile_result = profile

                    prev_scan_candidates = curr_candidates
                    next_stone_scan_frame = frame_index + stone_scan_interval_frames

            frame_index += 1

            # ------------------------------------------------
            # ETA
            # ------------------------------------------------

            if (
                frame_index % REPORT_EVERY == 0
                or frame_index == total_frames
            ):

                elapsed = (
                    time.time() -
                    start_time
                )

                if frame_index > 0:

                    frames_per_second = (
                        frame_index /
                        elapsed
                    )

                    remaining_frames = (
                        total_frames -
                        frame_index
                    )

                    eta_seconds = (
                        remaining_frames /
                        frames_per_second
                    )

                else:

                    frames_per_second = 0
                    eta_seconds = 0

                elapsed_minutes = int(
                    elapsed // 60
                )

                elapsed_secs = int(
                    elapsed % 60
                )

                eta_minutes = int(
                    eta_seconds // 60
                )

                eta_secs = int(
                    eta_seconds % 60
                )

                print(
                    f"Ruutu "
                    f"{frame_index} / "
                    f"{total_frames} | "
                    f"C++:lle tallennettuja kuvia: "
                    f"{engine.mode_frame_count()} | "
                    f"nopeus: "
                    f"{frames_per_second:.1f} r/s | "
                    f"kulunut: "
                    f"{elapsed_minutes:02d}:"
                    f"{elapsed_secs:02d} | "
                    f"ETA: "
                    f"{eta_minutes:02d}:"
                    f"{eta_secs:02d}"
                )
                processed = frame_index
                print(
                    f"  keskimäärin: "
                    f"gray {(total_gray_time / processed) * 1000:.1f} ms | "
                    f"tracking {(total_tracking_time / processed) * 1000:.1f} ms | "
                    f"transform {(total_transform_time / processed) * 1000:.3f} ms | "
                    f"sample {(total_sample_time / max(1, engine.mode_frame_count())) * 1000:.1f} ms"
                )

    finally:

        executor.shutdown(
            wait=True
        )

    print()

    print(
        f"Videon lapikaynti valmis "
        f"({frame_index}/{total_frames} ruutua)."
    )

    if calib_result is None:
        raise RuntimeError(
            "Kalibrointi ei onnistunut - video loppui kesken "
            "moodinaytteiden keruun (video liian lyhyt?)."
        )

    if profile_result is None:
        print(
            "VAROITUS: 3D-kiviprofiili ei tullut riittavaksi ennen "
            f"videon loppua ({len(accumulated_stones)} havaintoa "
            "kerattyna)."
        )

    return {
        "engine": engine,
        "calib": calib_result["calib"],
        "pose": calib_result["pose"],
        "profile": profile_result,
        "n_profile_observations": len(accumulated_stones),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    root = tk.Tk()
    root.withdraw()

    video_file = filedialog.askopenfilename(
        title="Valitse video",
        filetypes=[
            (
                "Videot",
                "*.mts *.MTS *.mp4 *.MP4 "
                "*.mov *.MOV *.avi *.AVI"
            ),
            (
                "Kaikki tiedostot",
                "*.*"
            )
        ]
    )

    root.destroy()

    if not video_file:

        print(
            "Videota ei valittu."
        )

        return

    print()
    print(
        f"Video: {video_file}"
    )

    # --------------------------------------------------------
    # PANEELIT - automaattitunnistus referenssitiedoston lahelta
    # (katso detect_panels_from_reference:in kommentti) - EI enaa
    # kasin klikkausta.
    # --------------------------------------------------------

    panel_data = load_panel_data(
        video_file
    )

    if panel_data is None:

        print()

        root = tk.Tk()
        root.withdraw()

        reference_file = filedialog.askopenfilename(
            title="Valitse referenssi-paneelitiedosto "
                  "(*_panel_corners.txt samantyyppisesta "
                  "kamera-asettelusta)",
            filetypes=[
                ("Paneelitiedostot", "*_panel_corners.txt"),
                ("Kaikki tiedostot", "*.*"),
            ]
        )

        root.destroy()

        if not reference_file:
            print("Referenssitiedostoa ei valittu.")
            return

        reference_panels = load_panel_reference(reference_file)

        cap = cv2.VideoCapture(video_file)

        if not cap.isOpened():
            raise RuntimeError("Videotiedostoa ei voitu avata.")

        ret, first_frame = cap.read()
        cap.release()

        if not ret:
            raise RuntimeError("Videon ensimmaista kuvaa ei voitu lukea.")

        first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

        panel_data = detect_panels_from_reference(
            first_gray, reference_panels
        )

        filename = save_panel_data(
            panel_data,
            video_file
        )

        print()

        print(
            f"Paneelien koordinaatit "
            f"tallennettu: {filename}"
        )

    # --------------------------------------------------------
    # PAAPUTKI: kalibrointi -> 3D-kiviprofiilin haku -> (Task 4
    # jatkaa tasta elavalla moni-kiven seurannalla myohemmin)
    # --------------------------------------------------------

    calib_diag_output = (
        os.path.splitext(video_file)[0] +
        "_kalibrointi_topdown.png"
    )

    result = run_pipeline(
        video_file,
        panel_data,
        calib_diag_output
    )

    print()
    print("=" * 60)
    print("KALIBROINTI + 3D-KIVIPROFIILI VALMIS")
    print("=" * 60)

    print(
        f"Resoluutio: "
        f"{result['engine'].width()} x "
        f"{result['engine'].height()}"
    )

    print(
        f"Topdown-tarkistuskuva: {calib_diag_output}"
    )

    if result["profile"] is not None:

        profile = result["profile"]

        print(
            f"3D-kiviprofiili: R_max={profile['R_max_cm']:.2f} cm, "
            f"H_total={profile['H_total_cm']:.2f} cm, "
            f"RMS={profile['residual_rms_px']:.2f} px "
            f"({result['n_profile_observations']} havaintoa)"
        )

    else:

        print(
            "3D-kiviprofiili EI valmistunut riittavaksi "
            f"({result['n_profile_observations']} havaintoa kerattyna)."
        )


if __name__ == "__main__":
    main()