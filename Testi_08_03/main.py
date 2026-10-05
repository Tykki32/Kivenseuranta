"""Kivenseuranta: curling-kivien seuranta videosta tai elavasta kamerakuvasta.

Kaytto:
  python main.py --video VIDEO.mp4 [--start 00:10:00] [--end 00:21:00] [--paneelit REF_panel_corners.txt] [--debug]
  python main.py --live [--katselu] [--paneelit REF_panel_corners.txt] [--live-tallenna]
  python main.py --live-sim VIDEO.mp4 [--live-sim-tahti 0]       (testaus ilman kameraa)

Rakenne (katso README.md):
  main.py              komentorivi, videon leikkaus, paneelit, live-lahteen kaynnistys
  asetukset.py         KAIKKI saadettavat asetukset
  seuranta.py          paaputki: kalibrointi -> kiviprofiili -> seuranta + CSV
  kalibrointi.py       kameran kalibrointi moodikuvasta (+ kalibrointi_perus.py, rata.py)
  profiili.py          kiven 3D-profiilin opettelu (+ kivimalli.py)
  esikasittely.py      stabilointi, valotasapaino, taustanvaimennus, liukuhihna
  siluetti.py          siluettitarkennus; paikallinen.py: seuranta kameran taydella resoluutiolla
  heitot.py            hog-analyysin kaynnistys, heittoportti, CSV:t (+ hog_analyysi.py, kierre.py)
  nakyma.py            debug-video ja puhelinnakyma; katselu.py: puhelimen selainsivu; live.py: kamera + puskuri
  C++: stone_tracker (HAKU, SEURANTA, maskit, profiilisovitus) ja mode_engine (videon luku, moodikuva)
"""
import argparse
import os
import signal
import subprocess
import sys
import threading
import time

import cv2

import stone_tracker

import asetukset as A
import katselu
import live
import paneelit
import seuranta


def alusta_cpp():
    """C++-moduulin (stone_tracker) asetukset ja tietoja kaynnistyksessa."""
    try:
        keys = ("General configuration for OpenCV", "Baseline:", "Dispatched code generation:", "requested:", "Parallel framework:",
                "Intel IPP:", "at:", "C++ flags (Release):", "Configuration:", "Built as dynamic libs?:")
        lines = [l.strip() for l in stone_tracker.opencv_build_info().splitlines() if any(k in l for k in keys)]
        print("C++-moduulin OpenCV: " + " | ".join(lines[:14]))
    except Exception as e:
        print(f"C++-moduulin OpenCV: tietoja ei saatu ({e!r})")
    stone_tracker.set_color_gate(A.VARIPORTTI_S_MAX, A.VARIPORTTI_H_TOL)
    stone_tracker.set_prep_parallel(1)
    simd, us_scalar, us_simd = stone_tracker.sat_simd_info()
    print(f"SEURANTA: saturaatio {'cvtColor (SIMD)' if simd else 'oma silmukka'} "
          f"(300x300: oma {us_scalar:.0f} us, cvtColor {us_simd:.0f} us)")


def _time_str_to_seconds(time_str):
    """"[HH:]MM:SS" (tai pelkka sekuntiluku) -> sekunnit."""
    seconds = 0.0
    for part in time_str.split(":"):
        seconds = seconds * 60 + float(part)
    return seconds


def _valitse_tiedosto(title, filetypes):
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    root.destroy()
    return path


def _leikkaa_video(input_file, start_time, end_time):
    """Videosta leikataan <nimi>_leikattu.mp4 (ffmpeg -c copy). --end on alkuperaisen videon aikaa."""
    stem = os.path.splitext(os.path.basename(input_file))[0]
    video_file = os.path.join(os.path.dirname(input_file), f"{stem}_leikattu.mp4")
    command = ["ffmpeg", "-y"]
    if start_time is not None:
        command += ["-ss", start_time]
    command += ["-i", input_file]
    if end_time is not None:
        # "-to" -ss:n jalkeen viittaisi ulostulon aikajanaan -> lasketaan kesto (-t) itse
        start_s = _time_str_to_seconds(start_time) if start_time is not None else 0.0
        duration = _time_str_to_seconds(end_time) - start_s
        if duration <= 0:
            raise ValueError(f"--end ({end_time}) on ennen tai samassa kohdassa kuin --start ({start_time}).")
        command += ["-t", str(duration)]
    command += ["-c", "copy", video_file]
    subprocess.run(command, check=True)
    return video_file


def _paneelit(video_file, panel_ref):
    """Paneelit: tallennettu tiedosto videon vierella, tai automaattitunnistus referenssitiedoston lahelta."""
    panel_data = paneelit.load_panel_data(video_file)
    if panel_data is not None:
        return panel_data
    print()
    if panel_ref:
        reference_file = panel_ref
        print(f"Referenssi-paneelitiedosto: {reference_file}")
    else:
        reference_file = _valitse_tiedosto(
            "Valitse referenssi-paneelitiedosto (*_panel_corners.txt samantyyppisesta kamera-asettelusta)",
            [("Paneelitiedostot", "*_panel_corners.txt"), ("Kaikki tiedostot", "*.*")],
        )
    if not reference_file:
        print("Referenssitiedostoa ei valittu.")
        return None
    reference_panels = paneelit.load_panel_reference(reference_file)
    cap = live.open_capture(video_file)
    if not cap.isOpened():
        raise RuntimeError("Videotiedostoa ei voitu avata.")
    ret, first_frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Videon ensimmaista kuvaa ei voitu lukea.")
    first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    panel_data = paneelit.detect_panels_from_reference(first_gray, reference_panels, frame_bgr=first_frame)
    filename = paneelit.save_panel_data(panel_data, video_file)
    print()
    print(f"Paneelien koordinaatit tallennettu: {filename}")
    return panel_data


def main(debug=False, start_time=None, end_time=None, video=None, live_cfg=None, panel_ref=None):
    if live_cfg is not None:
        # elava syote: video_file = tulostiedostojen nimipohja (tiedostoa ei ole, ruudut tulevat puskurista)
        video_file = live_cfg["video_file"]
        print()
        print(f"Live-syote: {live_cfg['kuvaus']}")
        print(f"Tulokset: {os.path.splitext(video_file)[0]}_*")
    else:
        input_file = video or _valitse_tiedosto("Valitse video", [("Videot", "*.mts *.MTS *.mp4 *.MP4 *.mov *.MOV *.avi *.AVI"),
                                                                   ("Kaikki tiedostot", "*.*")])
        if not input_file:
            print("Videota ei valittu.")
            return
        print()
        print(f"Video: {input_file}")
        video_file = _leikkaa_video(input_file, start_time, end_time)

    panel_data = _paneelit(video_file, panel_ref)
    if panel_data is None:
        return

    base = os.path.splitext(video_file)[0]
    calib_diag_output = base + "_kalibrointi_topdown.png"
    csv_output = base + "_kivien_sijainnit.csv"
    if katselu.active() is not None:      # puhelimelle lahetetty nakyma tallennetaan (yksi kuva = yksi videoruutu)
        katselu.active().tallenna(base + "_katselu.avi")
    debug_video_output = None
    if debug:
        debug_video_output = base + "_debug_seuranta.mp4"
        print(f"Debug-seurantavideo: {debug_video_output}")

    result = seuranta.Seuranta(video_file, panel_data, calib_diag_output, csv_output, debug_video_output).aja()

    print()
    print("=" * 60)
    print("KALIBROINTI + 3D-KIVIPROFIILI VALMIS")
    print("=" * 60)
    print(f"Resoluutio: {result['width']} x {result['height']}")
    print(f"Topdown-tarkistuskuva: {calib_diag_output}")
    profile = result["profile"]
    if profile is not None:
        print(f"3D-kiviprofiili: R_max={profile['R_max_cm']:.2f} cm, H_total={profile['H_total_cm']:.2f} cm, "
              f"kahvan_r={profile['handle_r_frac']:.3f} ({profile['handle_r_frac'] * profile['R_max_cm']:.2f} cm), "
              f"RMS={profile['residual_rms_px']:.2f} px ({result['n_profile_observations']} havaintoa)")
    else:
        print(f"3D-kiviprofiili EI valmistunut riittavaksi ({result['n_profile_observations']} havaintoa kerattyna).")
    if result["csv_output"] is not None:
        print(f"Kivien sijainti-CSV: {result['csv_output']} ({result['n_stones_seen']} eri kivea havaittu)")
    if result["debug_video_output"] is not None:
        print(f"Debug-seurantavideo: {result['debug_video_output']}")


# =====================================================================================================================
# LIVE
# =====================================================================================================================

def _parse_size(txt):
    if not txt or str(txt).strip() in ("0", "kamera"):
        return 0, 0
    w, h = str(txt).lower().split("x")
    return int(w), int(h)


def _start_live(args):
    """Kaynnistaa kameran (tai simulaation) taustasaikeeseen ja aktivoi puskurin."""
    out_w, out_h = _parse_size(args.live_koko)
    cam_w, cam_h = _parse_size(args.live_kamerakoko)
    # puskurin pitaa mahtua: historia taaksepain + profiilin opettelun ikkuna eteenpain + varaa
    buffer_s = max(args.live_puskuri_s, args.live_taakse_s + A.PROFIILI_IKKUNA_S + 5.0)
    # pakollinen historia (profiilin opettelu lukee ikkunan verran taaksepain siemenruudusta)
    min_back_s = min(args.live_taakse_s, A.PROFIILI_IKKUNA_S + 1.0)
    outdir = args.live_kansio or os.getcwd()
    os.makedirs(outdir, exist_ok=True)
    base = os.path.join(outdir, args.live_nimi or f"live_{time.strftime('%Y%m%d_%H%M%S')}")
    recorder = None
    if args.live_sim:
        src = live.FileSimSource(args.live_sim, realtime=bool(args.live_sim_tahti), buffer_s=buffer_s,
                                 keep_back_s=args.live_taakse_s, out_w=out_w, out_h=out_h, min_keep_back_s=min_back_s)
        kuvaus = f"SIMULAATIO {args.live_sim} ({'reaaliaika' if args.live_sim_tahti else 'ei tahdistusta'})"
    else:
        dev = None if args.live in (None, "auto") else int(args.live)

        def _open(backend, conv, device=dev):
            return live.CameraSource(device=device, cap_w=cam_w or 1920, cap_h=cam_h or 1080, out_w=out_w, out_h=out_h,
                                     target_fps=args.live_fps, buffer_s=buffer_s, keep_back_s=args.live_taakse_s,
                                     fourcc=args.live_fourcc, backend_name=backend, min_keep_back_s=min_back_s,
                                     conversion=conv, raw_order=args.live_raakajarjestys)

        src = None
        if args.live_muunnos == "auto" and sys.platform.startswith("win") and not args.live_taustaj:
            # Windows: MSMF + raaka YUY2 + GPU-muunnos (~11 ms CPU/ruutu) ensin, varalla DirectShow + ajuri (~30 ms).
            # MSMF numeroi laitteet eri jarjestyksessa kuin DSHOW -> MSMF etsii 1920x1080-laitteen itse.
            try:
                cand = _open("MSMF", "gpu", device=None)
                if cand.conversion == "gpu":
                    src = cand
                else:
                    print(f"Live: MSMF + gpu ei kaytettavissa (muunnos {cand.conversion}) -> DirectShow + ajuri")
                    cand.close()
            except Exception as e:
                print(f"Live: MSMF-kameraa ei saatu auki ({e}) -> DirectShow + ajuri")
            if src is None:
                src = _open("DSHOW", "ajuri")
        else:
            src = _open(args.live_taustaj, "ajuri" if args.live_muunnos == "auto" else args.live_muunnos)
        kuvaus = f"KAMERA laite {src.device}"
    if not args.paneelit:
        # ei paneelitiedostoa -> paneelit klikataan yhdesta kamerakuvasta ennen live-vaiheen alkua; valinta tallennetaan
        # <nimi>_panel_corners.txt:ksi (kelpaa seuraavalla kerralla --paneelit-arvoksi)
        try:
            still = src.grab_still()
            cv2.imwrite(base + "_paneelikuva.png", still)
            print()
            print(f"Paneelien valinta: klikkaa paneelien keskelle kuvassa (suositus vahintaan "
                  f"{paneelit.MIN_AUTO_PANELS_BEFORE_MANUAL}), Enter = valmis, Esc = peruuta. Live-vaihe alkaa vasta valinnan jalkeen.")
            panels = paneelit.select_panels_manually(cv2.cvtColor(still, cv2.COLOR_BGR2GRAY), still,
                                                     window_name="Klikkaa paneelit (Enter=valmis, Esc=peruuta)")
        except BaseException:
            src.close()
            raise
        fn = paneelit.save_panel_data(panels, base + ".mp4")
        print(f"Paneelit tallennettu: {fn} ({len(panels)} kpl). Seuraavalla kerralla samalla kamera-asettelulla: --paneelit \"{fn}\"")
    if args.live_tallenna:
        try:
            recorder = live.FfmpegRecorder(base + "_live.mp4", src.store.width, src.store.height, src.store.fps)
            src.recorder = recorder
            print(f"Live-tallennus: {recorder.path} (ffmpeg/{recorder.encoder})")
        except Exception as e:
            print(f"Live-tallennus ei kaytettavissa: {e}")
    print("Live-lahde: " + ", ".join(f"{k}={v}" for k, v in src.info.items()))
    st = src.store
    print(f"Live-puskuri: enintaan {buffer_s:.0f} s (~{buffer_s * st.fps * st.width * st.height * 3 / 1e9:.1f} GB), historia "
          f"kalibroinnissa {args.live_taakse_s:.0f} s, seurannassa {A.LIVE_TAAKSE_SEURANNASSA_S:.0f} s. Lopetus: Ctrl+C"
          + (f" tai {args.live_kesto:.0f} s" if args.live_kesto > 0 else ""))
    # kalibroinnin naytekohdissa myos taysresoluutioinen ruutu (vain jos kamera/video on seurantaa isompi)
    st.hires_every = max(1, int(round(st.fps * A.KALIB_NAYTEVALI_S)))
    live.activate(src)
    src.start()          # vasta tallentajan kytkemisen jalkeen

    def _sigint(_sig, _frm):
        print("\nLive: lopetus pyydetty (Ctrl+C) - kasitellaan puskuri loppuun ja kirjoitetaan tulokset. "
              "Paina Ctrl+C uudelleen keskeyttaaksesi heti.")
        src.stop("Ctrl+C")
        signal.signal(signal.SIGINT, signal.default_int_handler)

    signal.signal(signal.SIGINT, _sigint)
    if args.live_kesto > 0:
        threading.Timer(args.live_kesto, lambda: src.stop(f"--live-kesto {args.live_kesto:.0f} s")).start()
    return src, dict(video_file=base + ".mp4", kuvaus=kuvaus, base=base, recorder=recorder)


def _finish_live(src, cfg):
    src.stop("ajo valmis")
    src.join(5.0)
    st = src.store
    fps = st.fps
    print()
    print("=== LIVE-SYOTE ===")
    print(f"Kasiteltyja ruutuja {len(st.log)}, puskurin ruutuja yhteensa {st._head}, ohitettu hypyissa {st.skipped} "
          f"({st.skipped / fps:.1f} s), pudotettu (puskuri taynna) {st.dropped}, suurin viive {st.max_lag_frames} ruutua "
          f"({st.max_lag_frames / fps:.1f} s). Lopetuksen syy: {st._stop_reason or '-'}")
    if getattr(src, "n_conv", 0):
        print(f"Kamerasaie (CPU/ruutu): luku {src.cpu_read / max(1, src.n_conv) * 1000:.1f} ms + muunnos/pienennys "
              f"{src.cpu_conv / src.n_conv * 1000:.1f} ms (muunnos: {src.conversion})"
              + (f"; harvennuksessa ohitetut {src.n_grab} ruutua grab():lla {src.cpu_grab / src.n_grab * 1000:.1f} ms/ruutu"
                 if getattr(src, "n_grab", 0) else ""))
    rec = cfg.get("recorder")
    if rec is not None and getattr(rec, "n", 0):
        print(f"Tallennussaie (CPU/ruutu): {rec.cpu / rec.n * 1000:.1f} ms")
    path = cfg["base"] + "_kivien_sijainnit_live_aikaleimat.csv"
    if st.write_log(path):
        print(f"Live-aikaleimat (kasitelty ruutu -> kameran ruutu -> seinakello): {path}")
    if rec is not None:
        rec.close()
        print(f"Live-tallennus: {rec.path} (pudotettu {rec.dropped})")


def _argumentit():
    p = argparse.ArgumentParser(description="Curling-kivien seuranta (katso README.md)")
    p.add_argument("--video", default=None, help="Videotiedosto (ilman tata avautuu tiedostovalitsin).")
    p.add_argument("--start", default=None, help="Videon alkuaika, esim. 00:10:00")
    p.add_argument("--end", default=None, help="Videon loppuaika (alkuperaisen videon aikaa), esim. 00:21:00")
    p.add_argument("--debug", "-d", action="store_true", help="Tallenna debug-seurantavideo (<video>_debug_seuranta.mp4).")
    p.add_argument("--no-debug", action="store_true", help="Ei debug-videota (ohittaa --debug).")
    p.add_argument("--max-frame", type=int, default=0, help="Pysayta ajo kun tama ruutu on kasitelty (nopea testi).")
    p.add_argument("--paneelit", default=None,
                   help="Referenssi-paneelitiedosto (*_panel_corners.txt). Live-tilassa ilman tata paneelit klikataan kamerakuvasta.")
    p.add_argument("--katselu", nargs="?", type=int, const=8080, default=None,
                   help="Seurannan nakyma puhelimen selaimeen samassa wifissa (portti, oletus 8080).")
    # elava kamerasyote
    p.add_argument("--live", nargs="?", const="auto", default=None,
                   help="Elava kuva kamerasta (Cam Link). Valinnainen laitenumero, esim. --live 1 (oletus: etsi 1920x1080-laite).")
    p.add_argument("--live-sim", default=None, help="Testaus ilman kameraa: videotiedosto 'kamerana'.")
    p.add_argument("--live-sim-tahti", type=int, default=1,
                   help="--live-sim: 1 = reaaliaikainen tahti (oletus), 0 = niin nopeasti kuin kasittely ehtii (ei pudotuksia).")
    p.add_argument("--live-koko", default="1280x720", help="Kasittelykoko (kameran kuva pienennetaan), oletus 1280x720. 0 = kameran oma.")
    p.add_argument("--live-kamerakoko", default="1920x1080", help="Kameralta pyydetty koko (oletus 1920x1080).")
    p.add_argument("--live-fourcc", default=None, help="Kameralta pyydetty pakkausmuoto, esim. MJPG, YUY2, NV12.")
    p.add_argument("--live-taustaj", default=None, help="Kameran taustajarjestelma: DSHOW tai MSMF (oletus: automaattinen).")
    p.add_argument("--live-muunnos", default="auto", choices=["auto", "ajuri", "raw", "gpu"],
                   help="Kameran kuvan muunnos BGR:ksi: auto (Windowsissa ensin MSMF + gpu), ajuri, raw (YUY2 + OpenCV), gpu (YUY2 + OpenCL).")
    p.add_argument("--live-raakajarjestys", default=None, choices=["YUY2", "YVYU", "UYVY"], help="Pakota raakakuvan tavujarjestys.")
    p.add_argument("--live-fps", type=float, default=25.0,
                   help="Kasittelyn tavoite-fps: jos kamera antaa selvasti enemman (esim. 50), kaytetaan joka n:s ruutu. 0 = kaikki.")
    p.add_argument("--live-puskuri-s", type=float, default=90.0, help="Puskurin enimmaiskoko sekunteina (1280x720 ~ 69 MB/s).")
    p.add_argument("--live-taakse-s", type=float, default=40.0, help="Kalibroinnin aikana sailytettava historia (s).")
    p.add_argument("--live-kesto", type=float, default=0.0, help="Lopeta automaattisesti N sekunnin jalkeen (0 = Ctrl+C).")
    p.add_argument("--live-kansio", default=None, help="Tulosten kansio (oletus: nykyinen kansio).")
    p.add_argument("--live-nimi", default=None, help="Tulostiedostojen nimipohja (oletus live_<paiva>_<aika>).")
    p.add_argument("--live-tallenna", action="store_true", help="Tallenna kasiteltava kuva myos videoksi (<pohja>_live.mp4).")
    return p.parse_args()


if __name__ == "__main__":
    args = _argumentit()
    if args.max_frame:
        A.MAX_RUUTU = args.max_frame
    alusta_cpp()
    if args.katselu is not None and katselu.start(args.katselu) is not None:
        katselu.set_state("Kalibroidaan")
    live_cfg = live_src = None
    if args.live is not None or args.live_sim:
        # live-tilassa etualamaski samassa saikeessa (kone kuormitettu: kamera + tallennus): seurannan viive pienempi
        stone_tracker.set_prep_parallel(0)
        print("Live: etualamaski samassa saikeessa")
        live_src, live_cfg = _start_live(args)
    try:
        main(debug=args.debug and not args.no_debug, start_time=args.start, end_time=args.end, video=args.video,
             live_cfg=live_cfg, panel_ref=args.paneelit)
    finally:
        if live_src is not None:
            _finish_live(live_src, live_cfg)
        katselu.stop()
