# ============================================================
# kamera9_02.py - KIVEN PAIKANNUS MALLIPOHJAISELLA HAULLA (EI VARIN/
# MUODON SEGMENTOINNILLA)
#
# kamera9_01.py:n find_stone_candidates tunnistaa kiven segmentoimalla
# graniitin (varisaturaatio+paikallinen tummuus) ja suodattamalla
# muodolla - toimii, mutta on herkka virhekandidaateille (pyyhkija,
# tekstit, valaistus) ja vaatii jatkuvuuspohjaisen seurannan niiden
# poissulkemiseksi (katso kamera9_01.py:n track_stone_in_video).
#
# Tama tiedosto (kayttajan pyynnosta, MERKITTAVA lisays - siksi oma
# tiedostonsa, kamera9_01.py jatetaan koskemattomaksi) kaantaa
# lahestymistavan PAINVASTAISEKSI: kamera9_01.py:n jo sovitetusta
# kovakoodattu+hienosaadetusta 3D-mallista (fit_stone_profile) TEHDAAN
# oikea 3D-malli, ja sita YRITETAAN SIJOITTAA jaalle eri (X,Y)-
# pisteisiin - jokaisessa pisteessa ennustetaan mika kiven siluetti
# NAYTTAISI silta kohdalta (kamera9_01.py:n _predicted_stone_hull,
# joka jo kayttaa TAYTTA 3D-kameramallia - K,R,t - eika arvaa mitaan
# etaisyydesta/kulmasta), ja verrataan sita HAVAITTUUN graniittimaskiin
# (kamera9_01.py:n create_granite_mask). Piste jossa ennuste osuu
# parhaiten havaittuun maskiin ON kivi.
#
# Tama on kayttajan pyytama HAKU: "yrittaa sijoittaa sita jaalla eri
# pisteisiin jotta loytaa kiven" - siis KARKEA->HIENO ristikkohaku
# (sama periaate kuin kamera8_01.py:n search_far_house, katso sen
# kommentti), ei minkaanlaista segmentointia. Etuna: koska malli
# sijoitetaan SUORAAN kiven OMAAN pystyakseliin (X0,Y0) - ei kiven
# NAKYVAN ellipsin keskipisteeseen - loydetty (X0,Y0) ON JO Z-
# KORJATTU (jaataso-keskikohta), ilman erillista puolikorkeus-
# approksimaatiota tai sovitusta (katso kamera9_01.py:n kommentit
# aiemmista, epatarkemmista yrityksista).
#
# KAKSI VAIHETTA:
#   1) HAKU: kunnes kivi loytyy, koeta paikantaa se JOKA 10. FRAMESSA
#      rajatulta alueelta (kauempi paa: keskiviivasta +-50cm,
#      kaukaisen hoglinen ja pesan takarajan valilla - tyypillinen
#      alue jonne heitetty kivi paatyy).
#   2) SEURANTA: kun kivi loytyy, sen sijaintia paivitetaan JOKA
#      FRAMESSA (myos haku-alueen ULKOPUOLELLA) pienella paikallisella
#      haulla edellisen sijainnin ymparilta - halvempi kuin koko
#      alueen haku, koska jatkuvuus antaa jo vahvan alkuarvauksen.
# ============================================================

import os
import csv
import math
import time
import importlib.util

import numpy as np
import cv2


def _load_kamera9_01():
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        "kamera9_01", os.path.join(here, "kamera9_01.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


k9 = _load_kamera9_01()
k8 = k9.k8


# ============================================================
# HAKUALUE ("kauempi paa, keskiviivasta +-50cm, hogline-takaraja")
#
# "Takaraja" = pesan takareuna (back line) - tangentti 12-jalkaisen
# renkaan taakse, HOUSE_RADIUS_CM paan pesan keskipisteesta poispain
# hoglinesta. Kaukainen paa (FAR_*) koska talla videolla kivi heitetaan
# lahempaa hoglinea kohti sita - katso kamera9_01.py:n
# track_stone_in_video-kommentti samasta videosta.
# ============================================================

SEARCH_X_HALF_WIDTH_CM = 50.0
SEARCH_Y_MIN_CM = k8.FAR_HOGLINE_Y_CM
SEARCH_Y_MAX_CM = k8.FAR_HOUSE_Y_CM + k8.HOUSE_RADIUS_CM
SEARCH_EVERY_N_FRAMES = 10

# Karkea->hieno ristikkohaku (sama periaate kuin kamera8_01.py:n
# search_far_house) - kaksi tasoa riittaa, koska kolmas (ultra-hieno)
# taso ei enaa muuta lopputulosta merkittavasti mutta maksaa yhta
# paljon kuin ensimmainen.
SEARCH_COARSE_STEP_CM = 10.0
SEARCH_FINE_STEP_CM = 2.0
SEARCH_SCORE_THRESHOLD = 0.55  # peitto-osuus jolla "loytyi" hyvaksytaan

TRACK_HALF_RANGE_CM = 35.0
TRACK_COARSE_STEP_CM = 7.0
TRACK_FINE_STEP_CM = 1.5
TRACK_SCORE_THRESHOLD = 0.35  # matalampi - jatkuvuus on jo vahva prior
TRACK_LOST_MAX_MISSES = 5     # montako peräkkäistä huonoa framea ennen kuin palataan hakuun

# Pienempi resoluutio hylylle HAUSSA/SEURANNASSA (nopeampi, riittava
# karkean sijainnin loytamiseen - ei tarvita samaa tarkkuutta kuin
# esim. profiilin sovituksessa).
SEARCH_HULL_N_THETA = 14
SEARCH_HULL_N_PER_SEGMENT = 2


# ============================================================
# PEITTO-OSUUS: ennustetun siluetin ja havaitun graniittimaskin
# vastaavuus annetussa (X,Y)-pisteessa.
# ============================================================

def _hull_overlap_score(mask, hull):
    """
    Palauttaa kuinka suuri osuus ennustetun hylyn (kupera peite)
    pinta-alasta on graniittimaskissa "paalla" - 1.0 = taydellinen
    osuma, 0.0 = ei mitaan osumaa (esim. tyhjaa jaata).
    """

    if hull is None or len(hull) < 3:
        return 0.0

    hull_int = hull.astype(np.int32)
    x, y, w, h = cv2.boundingRect(hull_int)

    x0, y0 = max(x, 0), max(y, 0)
    x1 = min(x + w, mask.shape[1])
    y1 = min(y + h, mask.shape[0])

    if x1 <= x0 or y1 <= y0:
        return 0.0

    hull_shifted = hull_int - np.array([x0, y0])
    canvas = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    cv2.fillPoly(canvas, [hull_shifted], 255)

    hull_area = int(np.count_nonzero(canvas))

    if hull_area == 0:
        return 0.0

    mask_crop = mask[y0:y1, x0:x1]
    overlap = int(np.count_nonzero((canvas > 0) & (mask_crop > 0)))

    return overlap / hull_area


def _score_candidate(mask, pose, profile, X, Y):

    hull = k9._predicted_stone_hull(
        pose, X, Y, profile["R_max_cm"], profile["H_total_cm"], profile["shape_deltas"],
        n_theta=SEARCH_HULL_N_THETA, n_per_segment=SEARCH_HULL_N_PER_SEGMENT
    )

    return _hull_overlap_score(mask, hull)


def _grid_search_best(mask, pose, profile, x_vals, y_vals):

    best_score = -1.0
    best_xy = (float(x_vals[0]), float(y_vals[0]))

    for X in x_vals:
        for Y in y_vals:
            score = _score_candidate(mask, pose, profile, float(X), float(Y))
            if score > best_score:
                best_score = score
                best_xy = (float(X), float(Y))

    return best_xy, best_score


def _locate_by_grid_search(mask, pose, profile, x_center, x_half_range, y_center, y_half_range,
                            coarse_step, fine_step):
    """
    Karkea->hieno ristikkohaku annetulla alueella - katso taman
    tiedoston alkupaan kommentti periaatteesta. Palauttaa parhaan
    (X,Y) (cm) ja sen peitto-osuuden.
    """

    x_vals = np.arange(x_center - x_half_range, x_center + x_half_range + 1e-6, coarse_step)
    y_vals = np.arange(y_center - y_half_range, y_center + y_half_range + 1e-6, coarse_step)
    (bx, by), score = _grid_search_best(mask, pose, profile, x_vals, y_vals)

    x_vals = np.arange(bx - coarse_step, bx + coarse_step + 1e-6, fine_step)
    y_vals = np.arange(by - coarse_step, by + coarse_step + 1e-6, fine_step)
    (bx, by), score = _grid_search_best(mask, pose, profile, x_vals, y_vals)

    return (bx, by), score


# ============================================================
# HAKU + SEURANTA KOKO VIDEOLLE, CSV-TULOSTUS
# ============================================================

def search_and_track_stones(video_path, calib, pose, profile, csv_path,
                             search_x_half_width=SEARCH_X_HALF_WIDTH_CM,
                             search_y_min=SEARCH_Y_MIN_CM, search_y_max=SEARCH_Y_MAX_CM,
                             search_every_n=SEARCH_EVERY_N_FRAMES,
                             progress=True):
    """
    Kayy videon LAPI JOKA FRAME (ei valiin jattamista seurannassa - vain
    haku tapahtuu harvemmin, katso taman tiedoston alkupaan kommentti).
    Tilakone: HAKU (kunnes kivi loytyy rajatulta alueelta) -> SEURANTA
    (paivitetaan sijaintia joka frame pienella paikallisella haulla,
    myos haku-alueen ulkopuolella) -> jos kivi "katoaa" (huono peitto
    monta framea peräkkain), palataan HAKUUN (seuraava kivi saa uuden
    ID:n - tama sallii useamman perakkaisen heiton kasittelyn samasta
    videosta).

    Kirjoittaa CSV-tiedoston (frame, timestamp_s, stone_id, x_m, y_m) -
    x_m/y_m ovat kiven PYSTYAKSELIN sijainti jaatasolla (Z=0) - katso
    alkupaan kommentti miksi tama on jo Z-korjattu ilman erillista
    laskentaa.
    """

    cap = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 25.0

    camera_matrix = calib["camera_matrix"]
    dist_coeffs = np.array([calib["best_k1"], 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    state = "HAKU"
    stone_id = 0
    last_xy = None
    misses = 0
    n_written = 0

    t_start = time.time()

    with open(csv_path, "w", newline="") as f:

        writer = csv.writer(f)
        writer.writerow(["frame", "timestamp_s", "stone_id", "x_m", "y_m"])

        frame_idx = 0

        while frame_idx < n_frames:

            ok, frame = cap.read()

            if not ok:
                break

            do_this_frame = (state == "SEURANTA") or (frame_idx % search_every_n == 0)

            if not do_this_frame:
                frame_idx += 1
                continue

            frame_u = cv2.undistort(frame, camera_matrix, dist_coeffs)
            mask = k9.create_granite_mask(frame_u)

            if state == "HAKU":

                x_center = 0.0
                y_center = (search_y_min + search_y_max) / 2.0
                y_half = (search_y_max - search_y_min) / 2.0

                (bx, by), score = _locate_by_grid_search(
                    mask, pose, profile, x_center, search_x_half_width, y_center, y_half,
                    SEARCH_COARSE_STEP_CM, SEARCH_FINE_STEP_CM
                )

                if score >= SEARCH_SCORE_THRESHOLD:
                    stone_id += 1
                    last_xy = (bx, by)
                    misses = 0
                    state = "SEURANTA"

                    if progress:
                        print(f"[frame {frame_idx}] LOYTYI kivi {stone_id}: "
                              f"({bx:.1f}, {by:.1f}) cm, peitto={score:.2f}")

                    timestamp = frame_idx / fps
                    writer.writerow([frame_idx, f"{timestamp:.3f}", stone_id,
                                      f"{bx / 100.0:.4f}", f"{by / 100.0:.4f}"])
                    n_written += 1

            else:  # SEURANTA

                (bx, by), score = _locate_by_grid_search(
                    mask, pose, profile, last_xy[0], TRACK_HALF_RANGE_CM,
                    last_xy[1], TRACK_HALF_RANGE_CM,
                    TRACK_COARSE_STEP_CM, TRACK_FINE_STEP_CM
                )

                if score >= TRACK_SCORE_THRESHOLD:
                    last_xy = (bx, by)
                    misses = 0

                    timestamp = frame_idx / fps
                    writer.writerow([frame_idx, f"{timestamp:.3f}", stone_id,
                                      f"{bx / 100.0:.4f}", f"{by / 100.0:.4f}"])
                    n_written += 1
                else:
                    misses += 1

                    if misses >= TRACK_LOST_MAX_MISSES:
                        if progress:
                            print(f"[frame {frame_idx}] Kivi {stone_id} kadotettu "
                                  f"({misses} huonoa framea), palataan hakuun.")
                        state = "HAKU"
                        last_xy = None

            if progress and frame_idx % 100 == 0:
                elapsed = time.time() - t_start
                print(f"  ... frame {frame_idx}/{n_frames} (tila={state}, "
                      f"{elapsed:.0f}s kulunut, {n_written} rivia kirjoitettu)")

            frame_idx += 1

    cap.release()

    if progress:
        print(f"Valmis: {n_written} rivia kirjoitettu tiedostoon {csv_path} "
              f"({time.time() - t_start:.0f}s)")

    return {"n_written": n_written, "n_stones": stone_id}


# ============================================================
# PAAOHJELMA
#
# HUOM: profiilin sovitus (kamera9_01.py:n build_stone_profile_from_video)
# tarvitsee VIDEOKOHTAISEN "siemen"-framen ja -sijainnin (katso sen
# kommentti - automaattista "mika on oikea kivi" -paattelya ei ole
# ilman tallaista kasin annettua alkupistetta). Alla olevat arvot on
# jo loydetty/validoitu taman projektin "00011 - Trim.mp4":lle
# (katso kamera9_01.py:n kehityshistoria) - toiselle videolle nama
# pitaisi maarittaa uudelleen.
# ============================================================

DEFAULT_SEED_FRAME_IDX = 500
DEFAULT_SEED_POS_CM = (72.0, 1163.0)


def main():

    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()

    calib_filename = filedialog.askopenfilename(
        title="Valitse kalibrointikuva (jolla H_final/poosi ratkaistaan)",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
            ("All files", "*.*"),
        ]
    )

    if not calib_filename:
        print("Kalibrointikuvaa ei valittu.")
        root.destroy()
        return

    video_filename = filedialog.askopenfilename(
        title="Valitse video (kiven paikannukseen/seurantaan)",
        filetypes=[
            ("Video files", "*.mp4 *.avi *.mov *.mkv"),
            ("All files", "*.*"),
        ]
    )

    root.destroy()

    if not video_filename:
        print("Videota ei valittu.")
        return

    print("Kalibroidaan kamera (kamera8_01.py:n putki)...")
    calib = k9.calibrate_camera_from_image(calib_filename)
    pose = k9.build_pose_from_calibration(calib)
    print(f"  fokaalivali f = {pose['K'][0, 0]:.1f} px, "
          f"kameran sijainti (cm): {np.round(pose['camera_position_cm'], 1)}")

    kivilla_stones = k9.find_stone_candidates(calib["frame_undistorted"], calib["H_final"])
    print(f"  loydettiin {len(kivilla_stones)} paikallaan olevaa kivea kalibrointikuvasta")

    print(f"Sovitetaan 3D-profiili (kalibrointikuvan kivet + video, "
          f"siemen frame={DEFAULT_SEED_FRAME_IDX})...")
    result = k9.build_stone_profile_from_video(
        calib, pose, video_filename,
        seed_frame_idx=DEFAULT_SEED_FRAME_IDX, seed_pos_cm=DEFAULT_SEED_POS_CM,
        extra_stones=kivilla_stones, n_video_samples=25,
    )
    profile = result["profile"]
    print(f"  R_max={profile['R_max_cm']:.2f} cm, H_total={profile['H_total_cm']:.2f} cm, "
          f"RMS={profile['residual_rms_px']:.2f} px")

    print(f"Hakualue: X=+-{SEARCH_X_HALF_WIDTH_CM:.0f}cm keskiviivasta, "
          f"Y={SEARCH_Y_MIN_CM:.0f}..{SEARCH_Y_MAX_CM:.0f}cm "
          f"(kaukainen hogline - pesan takaraja)")
    print("Haetaan ja seurataan kivea/kivia videolta...")

    csv_path = "kivien_sijainnit.csv"
    search_and_track_stones(video_filename, calib, pose, profile, csv_path)


if __name__ == "__main__":
    main()
