"""Komentorivi ja kaynnistys."""

import os
import argparse
import subprocess
import cv2
import tkinter as tk
from tkinter import filedialog
from config import DEBUG_SAVE_TRACKING_VIDEO
from panel import (
    detect_panels_from_reference,
    load_panel_data,
    load_panel_reference,
    save_panel_data,
)
from pipeline import run_pipeline


# ============================================================
# MAIN
# ============================================================
def _time_str_to_seconds(time_str):
    """Muuntaa "[HH:]MM:SS"-muotoisen ajan (tai pelkan sekuntiluvun)
    sekunneiksi."""

    seconds = 0.0

    for part in time_str.split(":"):
        seconds = seconds * 60 + float(part)

    return seconds


def main(debug=None, start_time=None, end_time=None):

    # debug=None (oletus): kayta DEBUG_SAVE_TRACKING_VIDEO-vakion
    # arvoa (katso sen kommentti). debug=True/False komentoriviltä
    # (-d/--debug, katso alempana if __name__=="__main__") ohittaa
    # vakion - kayttajan pyynnosta dynaaminen paalle/pois-kytkenta
    # ilman lahdekoodin muokkausta.
    effective_debug = (
        DEBUG_SAVE_TRACKING_VIDEO if debug is None else debug
    )

    root = tk.Tk()
    root.withdraw()
    
    input_file = filedialog.askopenfilename(
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

    if not input_file:

        print(
            "Videota ei valittu."
        )

        return

    input_stem, _input_ext = os.path.splitext(os.path.basename(input_file))

    video_file = os.path.join(
        os.path.dirname(input_file),
        f"{input_stem}_leikattu.mp4"
    )

    print()
    print(
        f"Video: {input_file}"
    )

    command = ["ffmpeg"]

    if start_time is not None:
        command += ["-ss", start_time]

    command += ["-i", input_file]

    if end_time is not None:
        # HUOM: "-to" input-option "-ss":n jalkeen EI viittaa alkuperaisen
        # videon aikajanaan vaan "-ss":n siirtamaan (nollasta alkavaan)
        # ulostulon aikajanaan - "--end 00:21:00" leikkaisi siis 21 min
        # PITUISEN palan alkaen "--start"-kohdasta, ei alkuperaisen videon
        # kohtaan 21 min asti. Lasketaan siksi KESTO ("-t") itse.
        start_seconds = (
            _time_str_to_seconds(start_time) if start_time is not None else 0.0
        )
        end_seconds = _time_str_to_seconds(end_time)
        duration_seconds = end_seconds - start_seconds

        if duration_seconds <= 0:
            raise ValueError(
                f"--end ({end_time}) on ennen tai samassa kohdassa kuin "
                f"--start ({start_time})."
            )

        command += ["-t", str(duration_seconds)]

    command += ["-c", "copy", video_file]
    
    subprocess.run(command, check=True)
    
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
            first_gray, reference_panels, frame_bgr=first_frame
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
    # PAAPUTKI: kalibrointi -> 3D-kiviprofiilin haku -> elava
    # moni-kiven seuranta + CSV
    # --------------------------------------------------------

    calib_diag_output = (
        os.path.splitext(video_file)[0] +
        "_kalibrointi_topdown.png"
    )

    csv_output = (
        os.path.splitext(video_file)[0] +
        "_kivien_sijainnit.csv"
    )

    debug_video_output = None

    if effective_debug:

        debug_video_output = (
            os.path.splitext(video_file)[0] +
            "_debug_seuranta.mp4"
        )

        print(
            f"Debug-seurantavideo: {debug_video_output}"
        )

    result = run_pipeline(
        video_file,
        panel_data,
        calib_diag_output,
        csv_output,
        debug_video_output=debug_video_output
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
            f"kahvan_r={profile['handle_r_frac']:.3f} "
            f"({profile['handle_r_frac']*profile['R_max_cm']:.2f} cm), "
            f"RMS={profile['residual_rms_px']:.2f} px "
            f"({result['n_profile_observations']} havaintoa)"
        )

    else:

        print(
            "3D-kiviprofiili EI valmistunut riittavaksi "
            f"({result['n_profile_observations']} havaintoa kerattyna)."
        )

    if result["csv_output"] is not None:

        print(
            f"Kivien sijainti-CSV: {result['csv_output']} "
            f"({result['n_stones_seen']} eri kivea havaittu)"
        )

    if result["debug_video_output"] is not None:

        print(
            f"Debug-seurantavideo: {result['debug_video_output']}"
        )


if __name__ == "__main__":

    # Kayttajan pyynnosta: debug-seurantavideon (DEBUG_SAVE_TRACKING_
    # VIDEO, katso sen kommentti) voi kytkea paalle komentorivilta
    # ilman lahdekoodin muokkausta, esim: python main.py --debug
    _arg_parser = argparse.ArgumentParser()
    _arg_parser.add_argument(
        "--debug", "-d", action="store_true", default=None,
        help=(
            "Tallenna debug-seurantavideo (<video>_debug_seuranta.mp4) "
            "jossa HAKU/SEURANTA-tunnistusten ennustetut ääriviivat on "
            "piirretty framejen paalle. Ohittaa DEBUG_SAVE_TRACKING_"
            "VIDEO-vakion. Ilman tata lippua kaytetaan vakion arvoa."
        )
    )

    _arg_parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Videon alkuaika, esim. 00:10:00"
    )

    _arg_parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="Videon loppuaika, esim. 00:21:00"
    )
    
    _args = _arg_parser.parse_args()

    main(debug=_args.debug,
        start_time=_args.start,
        end_time=_args.end)
