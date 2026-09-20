import os
import time
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from concurrent.futures import ThreadPoolExecutor
import mode_engine
from collections import deque


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
# PERSPEKTIIVIKORJAUS
# ============================================================

HOUSE_X = 182.9

NEAR_HOUSE_Y = 182.9
NEAR_HOG_Y = 823.0
FAR_HOG_Y = 3017.5
FAR_HOUSE_Y = 3657.6

BACKLINE_Y = 3840.5
EXTRA_BEYOND_FAR_BACKLINE = 200.0

Y_MIN_CM = 0.0
Y_MAX_CM = BACKLINE_Y + EXTRA_BEYOND_FAR_BACKLINE

X_MIN_CM = -200.0
X_MAX_CM = 200.0

PERSPECTIVE_OUTPUT_WIDTH = int(
    round(X_MAX_CM - X_MIN_CM)
)

PERSPECTIVE_OUTPUT_HEIGHT = int(
    np.ceil(Y_MAX_CM - Y_MIN_CM)
) + 1

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
# PERSPEKTIIVIPISTEIDEN VALINTA
#
# 1 = lähempi pesä, vasen reuna
# 2 = lähempi pesä, oikea reuna
# 3 = lähempi hogline, keskikohta
# 4 = kauempi hogline, keskikohta
# 5 = kauempi pesä, vasen reuna
# 6 = kauempi pesä, oikea reuna
# ============================================================

def select_perspective_points(video_file):

    cap = cv2.VideoCapture(
        video_file
    )

    if not cap.isOpened():
        raise RuntimeError(
            "Videon avaaminen perspektiivipisteitä "
            "varten epäonnistui."
        )

    ok, frame = cap.read()

    cap.release()

    if not ok or frame is None:
        raise RuntimeError(
            "Videon ensimmäistä framea ei voitu lukea."
        )

    original = frame.copy()

    image_height, image_width = (
        original.shape[:2]
    )

    zoom = 1.0
    pan_x = 0.0
    pan_y = 0.0

    points = []

    dragging = False

    drag_start_x = 0
    drag_start_y = 0

    pan_start_x = 0.0
    pan_start_y = 0.0

    # --------------------------------------------------------
    # Koordinaattimuunnokset
    # --------------------------------------------------------

    def screen_to_image(sx, sy):

        image_center_x = (
            image_width / 2.0
        )

        image_center_y = (
            image_height / 2.0
        )

        ix = (
            (
                sx
                - DISPLAY_WIDTH / 2.0
                - pan_x
            )
            / zoom
            + image_center_x
        )

        iy = (
            (
                sy
                - DISPLAY_HEIGHT / 2.0
                - pan_y
            )
            / zoom
            + image_center_y
        )

        return ix, iy


    def image_to_screen(ix, iy):

        image_center_x = (
            image_width / 2.0
        )

        image_center_y = (
            image_height / 2.0
        )

        sx = (
            (ix - image_center_x)
            * zoom
            + DISPLAY_WIDTH / 2.0
            + pan_x
        )

        sy = (
            (iy - image_center_y)
            * zoom
            + DISPLAY_HEIGHT / 2.0
            + pan_y
        )

        return sx, sy


    # --------------------------------------------------------
    # Sovitetaan kuva ikkunaan
    # --------------------------------------------------------

    zoom = min(
        DISPLAY_WIDTH / image_width,
        DISPLAY_HEIGHT / image_height
    )

    pan_x = 0.0
    pan_y = 0.0


    # --------------------------------------------------------
    # Piirto
    # --------------------------------------------------------

    def redraw():

        canvas = np.zeros(
            (
                DISPLAY_HEIGHT,
                DISPLAY_WIDTH,
                3
            ),
            dtype=np.uint8
        )

        # Näkyvän alueen koko alkuperäisessä kuvassa

        src_w = (
            DISPLAY_WIDTH /
            zoom
        )

        src_h = (
            DISPLAY_HEIGHT /
            zoom
        )

        center_x = (
            image_width / 2.0
            - pan_x / zoom
        )

        center_y = (
            image_height / 2.0
            - pan_y / zoom
        )

        x1 = int(round(
            center_x - src_w / 2.0
        ))

        y1 = int(round(
            center_y - src_h / 2.0
        ))

        x2 = int(round(
            center_x + src_w / 2.0
        ))

        y2 = int(round(
            center_y + src_h / 2.0
        ))

        ix1 = max(
            0,
            x1
        )

        iy1 = max(
            0,
            y1
        )

        ix2 = min(
            image_width,
            x2
        )

        iy2 = min(
            image_height,
            y2
        )

        if (
            ix2 > ix1
            and
            iy2 > iy1
        ):

            crop = original[
                iy1:iy2,
                ix1:ix2
            ]

            target_w = max(
                1,
                int(round(
                    (ix2 - ix1) * zoom
                ))
            )

            target_h = max(
                1,
                int(round(
                    (iy2 - iy1) * zoom
                ))
            )

            crop_resized = cv2.resize(
                crop,
                (
                    target_w,
                    target_h
                ),
                interpolation=cv2.INTER_LINEAR
            )

            screen_x = int(round(
                DISPLAY_WIDTH / 2.0
                + (ix1 - center_x)
                * zoom
            ))

            screen_y = int(round(
                DISPLAY_HEIGHT / 2.0
                + (iy1 - center_y)
                * zoom
            ))

            dx1 = max(
                0,
                screen_x
            )

            dy1 = max(
                0,
                screen_y
            )

            dx2 = min(
                DISPLAY_WIDTH,
                screen_x + target_w
            )

            dy2 = min(
                DISPLAY_HEIGHT,
                screen_y + target_h
            )

            sx1 = (
                dx1 - screen_x
            )

            sy1 = (
                dy1 - screen_y
            )

            sx2 = (
                sx1
                + (dx2 - dx1)
            )

            sy2 = (
                sy1
                + (dy2 - dy1)
            )

            if (
                dx2 > dx1
                and
                dy2 > dy1
            ):

                canvas[
                    dy1:dy2,
                    dx1:dx2
                ] = crop_resized[
                    sy1:sy2,
                    sx1:sx2
                ]


        # ----------------------------------------------------
        # Pisteet
        # ----------------------------------------------------

        for i, (ix, iy) in enumerate(
            points
        ):

            sx, sy = image_to_screen(
                ix,
                iy
            )

            sx = int(round(sx))
            sy = int(round(sy))

            cv2.circle(
                canvas,
                (sx, sy),
                6,
                (0, 0, 255),
                -1
            )

            cv2.circle(
                canvas,
                (sx, sy),
                9,
                (255, 255, 255),
                2
            )

            cv2.putText(
                canvas,
                str(i + 1),
                (sx + 12, sy - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )


        # ----------------------------------------------------
        # Ohjeet
        # ----------------------------------------------------

        instructions = [
            "1 = lahempi pesa, vasen reuna",
            "2 = lahempi pesa, oikea reuna",
            "3 = lahempi hogline, keskikohta",
            "4 = kauempi hogline, keskikohta",
            "5 = kauempi pesa, vasen reuna",
            "6 = kauempi pesa, oikea reuna",
            "",
            "Vasen klikkaus = piste",
            "Oikea klikkaus = poista viimeinen",
            "Hiiren rulla = zoom",
            "Keskinappi + veto = panorointi",
            "Enter = hyvaksy",
            "Esc = peruuta",
            "",
            f"Zoom: {zoom:.2f}x",
            f"Pisteita: {len(points)}/6"
        ]

        y = 25

        for text in instructions:

            if text == "":
                y += 8
                continue

            cv2.putText(
                canvas,
                text,
                (15, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

            y += 22

        return canvas


    # --------------------------------------------------------
    # Hiiritapahtumat
    # --------------------------------------------------------

    def mouse_callback(
        event,
        x,
        y,
        flags,
        param
    ):

        nonlocal zoom
        nonlocal pan_x
        nonlocal pan_y
        nonlocal dragging
        nonlocal drag_start_x
        nonlocal drag_start_y
        nonlocal pan_start_x
        nonlocal pan_start_y

        # ----------------------------------------------------
        # Vasen = lisää piste
        # ----------------------------------------------------

        if event == cv2.EVENT_LBUTTONDOWN:

            if len(points) >= 6:
                return

            ix, iy = screen_to_image(
                x,
                y
            )

            if (
                ix < 0
                or iy < 0
                or ix >= image_width
                or iy >= image_height
            ):
                return

            points.append(
                (
                    float(ix),
                    float(iy)
                )
            )

            print(
                f"Perspektiivipiste "
                f"{len(points)}: "
                f"x={ix:.3f}, "
                f"y={iy:.3f}"
            )


        # ----------------------------------------------------
        # Oikea = poista viimeinen
        # ----------------------------------------------------

        elif event == cv2.EVENT_RBUTTONDOWN:

            if points:

                removed = points.pop()

                print(
                    f"Poistettu piste: "
                    f"{removed[0]:.3f}, "
                    f"{removed[1]:.3f}"
                )


        # ----------------------------------------------------
        # Keskinappi = panorointi
        # ----------------------------------------------------

        elif event == cv2.EVENT_MBUTTONDOWN:

            dragging = True

            drag_start_x = x
            drag_start_y = y

            pan_start_x = pan_x
            pan_start_y = pan_y


        elif event == cv2.EVENT_MBUTTONUP:

            dragging = False


        elif event == cv2.EVENT_MOUSEMOVE:

            if dragging:

                pan_x = (
                    pan_start_x
                    + (x - drag_start_x)
                )

                pan_y = (
                    pan_start_y
                    + (y - drag_start_y)
                )


        # ----------------------------------------------------
        # Rulla = zoom
        # ----------------------------------------------------

        elif event == cv2.EVENT_MOUSEWHEEL:

            if flags > 0:
                zoom *= 1.25
            else:
                zoom /= 1.25

            zoom = max(
                0.05,
                min(
                    zoom,
                    30.0
                )
            )

            ix, iy = screen_to_image(
                x,
                y
            )

            image_center_x = (
                image_width / 2.0
            )

            image_center_y = (
                image_height / 2.0
            )

            pan_x = (
                x
                - DISPLAY_WIDTH / 2.0
                - (
                    ix - image_center_x
                ) * zoom
            )

            pan_y = (
                y
                - DISPLAY_HEIGHT / 2.0
                - (
                    iy - image_center_y
                ) * zoom
            )


    # --------------------------------------------------------
    # Ikkuna
    # --------------------------------------------------------

    window_name = (
        "Perspektiivikorjaus - "
        "valitse 6 pistetta"
    )

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL
    )

    cv2.resizeWindow(
        window_name,
        DISPLAY_WIDTH,
        DISPLAY_HEIGHT
    )

    cv2.setMouseCallback(
        window_name,
        mouse_callback
    )


    accepted = False

    while True:

        display = redraw()

        cv2.imshow(
            window_name,
            display
        )

        key = cv2.waitKey(20) & 0xFF

        if key in (13, 10):

            if len(points) == 6:

                accepted = True
                break

            print(
                f"Tarvitaan 6 pistetta. "
                f"Nyt valittuna {len(points)}."
            )

        elif key in (8, 127):

            if points:

                removed = points.pop()

                print(
                    f"Poistettu piste: "
                    f"{removed[0]:.3f}, "
                    f"{removed[1]:.3f}"
                )

        elif key == 27:

            break


    cv2.destroyAllWindows()


    if not accepted:

        raise RuntimeError(
            "Perspektiivipisteiden valinta peruttiin."
        )


    return points


# ============================================================
# PERSPEKTIIVIHOMOGRAFIA
# ============================================================

def build_perspective_homography(points):

    p = np.asarray(
        points,
        dtype=np.float64
    )

    if p.shape != (6, 2):

        raise ValueError(
            "Perspektiivikorjaukseen tarvitaan "
            "täsmälleen 6 pistettä."
        )

    # --------------------------------------------------------
    # Lähdekoordinaatit
    # --------------------------------------------------------

    x1, y1 = p[0]
    x2, y2 = p[1]
    x3, y3 = p[2]
    x4, y4 = p[3]
    x5, y5 = p[4]
    x6, y6 = p[5]

    # --------------------------------------------------------
    # Kohdekoordinaatit senttimetreinä
    # --------------------------------------------------------

    target = np.array(
        [
            [-HOUSE_X, NEAR_HOUSE_Y],
            [ HOUSE_X, NEAR_HOUSE_Y],
            [ 0.0,      NEAR_HOG_Y],
            [ 0.0,      FAR_HOG_Y],
            [-HOUSE_X, FAR_HOUSE_Y],
            [ HOUSE_X, FAR_HOUSE_Y],
        ],
        dtype=np.float64
    )

    # --------------------------------------------------------
    # DLT
    # --------------------------------------------------------

    A = []

    for (x, y), (X, Y) in zip(
        p,
        target
    ):

        A.append(
            [
                x,
                y,
                1.0,
                0.0,
                0.0,
                0.0,
                -X * x,
                -X * y,
            ]
        )

        A.append(
            [
                0.0,
                0.0,
                0.0,
                x,
                y,
                1.0,
                -Y * x,
                -Y * y,
            ]
        )

    A = np.asarray(
        A,
        dtype=np.float64
    )

    b = target.reshape(-1)

    h, residuals, rank, singular_values = (
        np.linalg.lstsq(
            A,
            b,
            rcond=None
        )
    )

    H = np.array(
        [
            [h[0], h[1], h[2]],
            [h[3], h[4], h[5]],
            [h[6], h[7], 1.0],
        ],
        dtype=np.float64
    )

    # --------------------------------------------------------
    # Fyysiset cm -> output-pikselit
    #
    # X:
    #   -200 -> 0
    #   +200 -> 400
    #
    # Y:
    #   0 -> kuvan alaosa
    #   4040.5 -> kuvan yläosa
    # --------------------------------------------------------

    translation = np.array(
        [
            [1.0,  0.0, -X_MIN_CM],
            [0.0, -1.0,  Y_MAX_CM],
            [0.0,  0.0,  1.0],
        ],
        dtype=np.float64
    )

    H_output = (
        translation @ H
    )

    return H_output


# ============================================================
# TARKISTA PERSPEKTIIVIHOMOGRAFIA
# ============================================================

def print_perspective_check(
    points,
    H_output
):

    source_points = np.asarray(
        points,
        dtype=np.float32
    ).reshape(
        -1,
        1,
        2
    )

    mapped_points = cv2.perspectiveTransform(
        source_points,
        H_output.astype(np.float64)
    ).reshape(
        -1,
        2
    )

    expected_physical = np.array(
        [
            [-HOUSE_X, NEAR_HOUSE_Y],
            [ HOUSE_X, NEAR_HOUSE_Y],
            [ 0.0,      NEAR_HOG_Y],
            [ 0.0,      FAR_HOG_Y],
            [-HOUSE_X, FAR_HOUSE_Y],
            [ HOUSE_X, FAR_HOUSE_Y],
        ],
        dtype=np.float64
    )

    expected_pixels = np.empty_like(
        expected_physical
    )

    expected_pixels[:, 0] = (
        expected_physical[:, 0]
        - X_MIN_CM
    )

    expected_pixels[:, 1] = (
        Y_MAX_CM
        - expected_physical[:, 1]
    )

    print()
    print(
        "Perspektiivikorjauksen tarkistus:"
    )
    print()

    for i, (actual, expected) in enumerate(
        zip(
            mapped_points,
            expected_pixels
        ),
        start=1
    ):

        error = np.linalg.norm(
            actual - expected
        )

        print(
            f"Piste {i}: "
            f"laskettu=("
            f"{actual[0]:.2f}, "
            f"{actual[1]:.2f}) "
            f"odotettu=("
            f"{expected[0]:.2f}, "
            f"{expected[1]:.2f}) "
            f"virhe={error:.2f} px"
        )

    print()
    print("Homografia:")
    print(H_output)
    print()

# ============================================================
# PANEELIN TUNNISTUS
# ============================================================

def detect_panel(gray, x, y):
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
        x - ROI_HALF_WIDTH
    )

    x2 = min(
        w,
        x + ROI_HALF_WIDTH + 1
    )

    y1 = max(
        0,
        y - ROI_HALF_HEIGHT
    )

    y2 = min(
        h,
        y + ROI_HALF_HEIGHT + 1
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
# PANEELIEN VALINTA
# ============================================================

def select_panels(video_file):

    cap = cv2.VideoCapture(
        video_file
    )

    if not cap.isOpened():
        raise RuntimeError(
            "Videotiedostoa ei voitu avata."
        )

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(
            "Videon ensimmäistä kuvaa ei voitu lukea."
        )

    display_frame = frame.copy()

    scale = min(
        DISPLAY_WIDTH / frame.shape[1],
        DISPLAY_HEIGHT / frame.shape[0]
    )

    if scale < 1.0:
        display = cv2.resize(
            frame,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA
        )
    else:
        display = frame.copy()
        scale = 1.0

    selected_points = []

    window_name = (
        "Valitse paneelit - "
        "Enter valmis, oikea nappi poistaa"
    )

    def redraw():
        nonlocal display

        if scale < 1.0:
            display = cv2.resize(
                display_frame,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA
            )
        else:
            display = display_frame.copy()

        for i, (px, py) in enumerate(
            selected_points
        ):
            dx = int(round(px * scale))
            dy = int(round(py * scale))

            cv2.circle(
                display,
                (dx, dy),
                8,
                (0, 0, 255),
                -1
            )

            cv2.putText(
                display,
                str(i + 1),
                (dx + 10, dy - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        cv2.imshow(
            window_name,
            display
        )

    def mouse_callback(event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:

            original_x = x / scale
            original_y = y / scale

            selected_points.append(
                (original_x, original_y)
            )

            print(
                f"Paneeli {len(selected_points)}: "
                f"{original_x:.1f}, "
                f"{original_y:.1f}"
            )

            redraw()

        elif event == cv2.EVENT_RBUTTONDOWN:

            if selected_points:
                selected_points.pop()

                print(
                    "Viimeinen paneeli poistettu."
                )

                redraw()

    cv2.namedWindow(
        window_name
    )

    cv2.setMouseCallback(
        window_name,
        mouse_callback
    )

    redraw()

    while True:

        key = cv2.waitKey(50) & 0xFF

        if key == 13:

            if len(selected_points) >= 2:
                break

        elif key == 27:

            cv2.destroyAllWindows()

            raise RuntimeError(
                "Paneelien valinta peruutettiin."
            )

    cv2.destroyAllWindows()

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    panel_data = []

    for point in selected_points:

        result = detect_panel(
            gray,
            point[0],
            point[1]
        )

        if result is None:

            x, y = point

            corners = np.array(
                [
                    [x - 5, y - 5],
                    [x + 5, y - 5],
                    [x + 5, y + 5],
                    [x - 5, y + 5]
                ],
                dtype=np.float32
            )

            center = np.array(
                [x, y],
                dtype=np.float32
            )

            result = {
                "corners": corners,
                "center": center
            }

        panel_data.append(
            result
        )

    return panel_data


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
# ============================================================

def load_panel_data(video_file):

    base = os.path.splitext(
        video_file
    )[0]

    filename = (
        base +
        "_panel_corners.txt"
    )

    if not os.path.exists(filename):
        return None

    try:

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

        if len(panels) < 2:
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



# ============================================================
# STABILOINTI + KUVA SEKUNNISSA
# ============================================================

def stabilize_and_collect_median_frames(
    video_file,
    panel_data,
    output_file
):

    engine = mode_engine.ModeEngine(
        video_file,
        TILE_SIZE
    )

    width = engine.width()
    height = engine.height()
    fps = engine.fps()
    total_frames = engine.total_frames()

    # ========================================================
    # PERSPEKTIIVIKORJAUS
    # ========================================================

    perspective_points = select_perspective_points(
        video_file
    )

    print()
    print("Valitut perspektiivipisteet:")
    print()

    for i, (x, y) in enumerate(
        perspective_points,
        start=1
    ):

        print(
            f"Piste {i}: "
            f"x={x:.3f}, "
            f"y={y:.3f}"
        )

    H_output = build_perspective_homography(
        perspective_points
    )

    print_perspective_check(
        perspective_points,
        H_output
    )

    engine.set_perspective_output_size(
        PERSPECTIVE_OUTPUT_WIDTH,
        PERSPECTIVE_OUTPUT_HEIGHT
    )

    engine.set_perspective_matrix(
        H_output
    )

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

    mode_max_frames = int(
        round(
            fps *
            MODE_DURATION_SECONDS
        )
    )

    sample_every = MODE_FRAME_INTERVAL

    print(
        f"Moodikuvaan käytetään videon "
        f"ensimmäiset "
        f"{MODE_DURATION_SECONDS:.0f} sekuntia."
    )

    print(
        f"Moodikuvaan otetaan yksi kuva "
        f"joka {sample_every}. frame."
    )

    print(
        f"Moodikuvaan tulee enintään noin "
        f"{mode_max_frames // sample_every} "
        f"kuvaa."
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
    next_sample_frame = 0

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

    mode_started = False
    #mode_output_file = None

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
            # MOODIKUVA + VIDEO
            # ------------------------------------------------

            if ENABLE_MODE_FILTER:

                # --------------------------------------------
                # Kerätään moodikuvan näytteet ensimmäisestä
                # 120 sekunnista.
                # --------------------------------------------

                if (
                    frame_index < mode_max_frames
                    and frame_index >= next_sample_frame
                ):

                    t6 = time.perf_counter()

                    engine.add_mode_frame()

                    t7 = time.perf_counter()

                    total_sample_time += (
                        t7 - t6
                    )

                    next_sample_frame += sample_every


                # --------------------------------------------
                # Kun kaikki moodinäytteet on kerätty,
                # käynnistetään moodikuvan laskenta
                # taustalla.
                # --------------------------------------------

                if (
                    not mode_started
                    and next_sample_frame >= mode_max_frames
                ):

                    print()
                    print(
                        "Ensimmäisten 2 minuutin "
                        "moodinäytteet ovat nyt kasassa."
                    )

                    print(
                        "C++ aloittaa moodikuvan "
                        "laskennan taustalla."
                    )

                    engine.start_mode_background(
                        output_file
                    )

                    mode_started = True


                # --------------------------------------------
                # Moodisuodatus päällä.
                #
                # C++ ei kirjoita ruutua ennen kuin
                # moodikuva on valmis.
                # --------------------------------------------

                engine.process_frame(
                    FILTER_DISTANCE_THRESHOLD
                )


            else:

                # --------------------------------------------
                # Moodisuodatus pois.
                #
                # Negatiivinen arvo tarkoittaa C++:lle:
                # kirjoita stabiloitu/perspektiivikorjattu
                # ruutu sellaisenaan.
                #
                # Video alkaa siis heti.
                # --------------------------------------------

                engine.process_frame(
                    -1.0
                )

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
        f"Stabilointi valmis. "
        f"C++ keräsi "
        f"{engine.mode_frame_count()} "
        f"stabiloitua kuvaa."
    )

    return engine




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
    # PANEELIT
    # --------------------------------------------------------

    panel_data = load_panel_data(
        video_file
    )

    if panel_data is None:

        print()

        print(
            "Paneelien koordinaattitiedostoa "
            "ei löytynyt tai se oli virheellinen."
        )

        print(
            "Valitse paneelit ensimmäisestä ruudusta."
        )

        panel_data = select_panels(
            video_file
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
    # C++ LASKEE MOODIKUVAN
    # --------------------------------------------------------

    base = os.path.splitext(
        video_file
    )[0]

    output_file = (
        base + ".png"
    )

    engine = stabilize_and_collect_median_frames(
        video_file,
        panel_data,
        output_file
    )

    print()
    print(
        "Python-vaihe valmis."
    )

    if ENABLE_MODE_FILTER:

        print(
            "Odotetaan, että C++:n "
            "taustalla laskema moodikuva valmistuu."
        )

        engine.wait_for_mode()

        print()
        print("=" * 60)
        print("VALMIS")
        print("=" * 60)

        print(
            f"Moodikuva: "
            f"{output_file}"
        )

        print(
            f"Resoluutio: "
            f"{engine.width()} x {engine.height()}"
        )

        print(
            f"Moodiin käytettyjä "
            f"stabiloituja kuvia: "
            f"{engine.mode_frame_count()}"
        )

    else:

        print()
        print("=" * 60)
        print("VALMIS - ILMAN MOODISUODATUSTA")
        print("=" * 60)

        print(
            f"Resoluutio: "
            f"{engine.width()} x {engine.height()}"
        )


if __name__ == "__main__":
    main()