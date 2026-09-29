"""Paaputki: videon lapikaynti, kalibrointi, kiviprofiili ja elava moni-kiven seuranta + CSV."""

import os
import math
import csv
import time
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import mode_engine
import stone_tracker
from config import (
    CALIB_MODE_DURATION_SECONDS,
    CALIB_MODE_SAMPLE_INTERVAL_SECONDS,
    CSV_HEADER,
    ENABLE_SHADOW_TOLERANT_STABILIZATION,
    GRANITE_DIFF_THRESHOLD,
    HAKU_SEARCH_INTERVAL_FRAMES,
    HISTORY_LENGTH,
    MAX_ABS_X_FROM_CENTERLINE_CM,
    MAX_CONCURRENT_STONES,
    MAX_POSITION_ERROR,
    MAX_WORKERS,
    MIN_CONFIRMED_TARKKA_FRACTION,
    MIN_CONFIRMED_TARKKA_OBSERVATIONS,
    MIN_CONFIRMED_THROW_DISPLACEMENT_CM,
    MIN_PRECONFIRM_TARKKA_FRACTION,
    MIN_PRECONFIRM_TARKKA_OBSERVATIONS,
    NEW_STONE_DEDUP_CM,
    PROFILE_MIN_ACCEPTED_STONES,
    PROFILE_SAMPLES_PER_STONE,
    REPORT_EVERY,
    SEARCH_COARSE_STEP_CM,
    SEARCH_FINE_STEP_CM,
    SEARCH_HULL_N_PER_SEGMENT,
    SEARCH_HULL_N_THETA,
    SEARCH_SCORE_THRESHOLD,
    SEARCH_X_HALF_WIDTH_CM,
    SEARCH_Y_MAX_CM,
    SEARCH_Y_MIN_CM,
    SOLO_TRACK_MAX_RMS_PX,
    STABILIZATION_MEDIAN_FRAMES,
    STONE_SCAN_COOLDOWN_FRAMES,
    STONE_SCAN_INTERVAL_SECONDS,
    STOP_TRACKING_DISPLACEMENT_CM,
    STOP_TRACKING_SECONDS,
    TILE_SIZE,
    TRACK_COARSE_STEP_CM,
    TRACK_FINE_STEP_CM,
    TRACK_HALF_RANGE_MAX_CM,
    TRACK_LOST_GRACE_SECONDS,
    TRACK_LOST_MAX_MISSES,
    TRACK_MAX_BACKWARD_CM,
    TRACK_MAX_SPEED_X_CM_S,
    TRACK_MAX_SPEED_Y_CM_S,
    TRACK_SCORE_THRESHOLD,
)
from calibration import build_pose_from_calibration, calibrate_camera_from_image_with_seed
from stone_model import _build_undistort_maps, build_local_stone_rings, predicted_stone_hull_fast
from panel import track_panel
from stabilization import (
    _phase_correlate_full_frame,
    apply_photometric_correction,
    estimate_photometric_correction,
    suppress_static_background,
)
from stone_tracking import (
    _scan_stone_candidates,
    _write_stone_csv_row,
    build_stone_color_reference,
    color_match_median_diff,
    find_moving_candidate,
    track_stone_in_video_windowed,
    try_fit_profile,
)


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
    calib_diag_output,
    csv_output,
    precomputed_calib_result=None,
    precomputed_profile_result=None,
    debug_video_output=None
):

    engine = mode_engine.ModeEngine(
        video_file,
        TILE_SIZE
    )

    width = engine.width()
    height = engine.height()
    fps = engine.fps()
    total_frames = engine.total_frames()

    # katso HAKU_SEARCH_INTERVAL_FRAMES:in kommentti - korvaa
    # kamera9_02.py:n SEARCH_EVERY_N_FRAMES:in elavan seurannan
    # HAKU-ajastuksessa (koskematon kamera9_02.py itse ennallaan).
    haku_interval_frames = max(1, HAKU_SEARCH_INTERVAL_FRAMES)

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

    calib_result = precomputed_calib_result

    stone_scan_interval_frames = max(
        1, int(round(fps * STONE_SCAN_INTERVAL_SECONDS))
    )
    stop_tracking_frames = max(
        1, int(round(fps * STOP_TRACKING_SECONDS))
    )
    # katso ASETUKSET-kommentti TRACK_LOST_GRACE_SECONDS:in kohdalla -
    # PAIKALLINEN ylikirjoitus kamera9_02.py:n TRACK_LOST_MAX_MISSES:lle
    # (max(), koska ylikirjoitus EI saa koskaan olla tiukempi kuin
    # alkuperainen, vain sallivampi).
    track_lost_max_misses = max(
        TRACK_LOST_MAX_MISSES, int(round(fps * TRACK_LOST_GRACE_SECONDS))
    )
    next_stone_scan_frame = 0
    prev_scan_candidates = None
    accumulated_stones = []
    n_accepted_stones = 0
    best_sufficient_profile = None
    profile_result = precomputed_profile_result
    next_allowed_scan_track_frame = 0

    if precomputed_calib_result is not None:
        print(
            "Kalibrointi annettu valmiiksi laskettuna "
            "(ohitetaan moodikuva-bootstrap)."
        )
    if precomputed_profile_result is not None:
        print(
            "3D-kiviprofiili annettu valmiiksi laskettuna "
            "(ohitetaan koko radan skannaus)."
        )

    active_stones = []
    next_stone_id = 0
    # Lasketaan VAIN liikevahvistetut (katso MIN_CONFIRMED_THROW_
    # DISPLACEMENT_CM) kivet - EI raakoja HAKU-ehdokkaita, joista suurin
    # osa (katso ASETUKSET-kommentti) on paikallaan-jo-olevien kohteiden
    # virheellisia uudelleenlöytöjä, ei aitoja heittoja.
    n_confirmed_stones = 0
    live_state = None
    csv_writer = None
    csv_file = None
    debug_video_writer = None

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

    # PIKSELITARKKA SUORA STABILOINTI (kayttajan pyynnosta, katso
    # kommentti alempana kaytonkohdalla) - moodikuvan RAAKA (ei viela
    # undistorted) harmaasavyreferenssi, jota vasten "loppuvideon" joka
    # frame vaihekorrelaatiolla verrataan. Asetetaan heti kun calib_
    # result tulee valmiiksi (katso alempana) - TAI HETI TASSA jos
    # calib_result annettiin jo valmiiksi laskettuna (precomputed_calib_
    # result, esim. testeissa): korjattu bugi, jossa loppuvideo_ref_gray
    # jai koskaan asettamatta talla polulla ja _phase_correlate_full_
    # frame kaatui "NoneType has no attribute shape" heti loppuvideo-
    # vaiheen ensimmaisella framella.
    loppuvideo_ref_gray = None

    if calib_result is not None:
        loppuvideo_ref_gray = cv2.cvtColor(
            calib_result["calib"]["frame"], cv2.COLOR_BGR2GRAY
        )

    # VALOTASAPAINO/KIRKKAUS-KORJAUS (kayttajan pyynnosta, katso
    # estimate_photometric_correction:in kommentti): xy-siirtyma haetaan
    # JOKA framella, mutta valotasapaino/kirkkaus vain KERRAN SEKUNNISSA
    # (hidas ajautuminen, liian hidas laskea joka framelle). photo_gain/
    # photo_bias pysyvat viimeisimpana laskettuina arvoina niiden
    # valissa olevilla frameilla.
    photo_gain = None
    photo_bias = None
    next_photo_update_frame = 0
    total_photometric_time = 0.0

    start_time = time.time()

    total_gray_time = 0.0
    total_tracking_time = 0.0
    total_transform_time = 0.0
    total_sample_time = 0.0

    # --------------------------------------------------------
    # NOPEUSSEURANTA (kayttajan pyynnosta): HAKU (stone_tracker.
    # search_new_stone) ja SEURANTA (stone_tracker.track_stones_
    # batch) ovat nyt molemmat C++:aa - tallennetaan niiden
    # kutsumaarat+kokonaisajat tanne jotta REPORT_EVERY-valein
    # tulostettava yhteenveto (ja lopuksi koko ajon yhteenveto)
    # kertoo TARKALLEEN missa aika kuluu MYOS kayttajan omalla
    # koneella - katso raportin tulostus alempana.
    # --------------------------------------------------------

    total_haku_time = 0.0
    n_haku_calls = 0

    total_seuranta_time = 0.0
    n_seuranta_calls = 0
    n_seuranta_stone_updates = 0

    # Lisamittarit "mihin loput ajasta menee" -selvitykseen (kayttajan
    # pyynnosta): engine.read() (videon luku+dekoodaus), stabilointi-
    # matriisin RANSAC-laskenta, ja warpAffine+remap (KOKO framelle,
    # jokaisella elavan seurannan framella - nama eivat olleet aiemmin
    # ollenkaan ajastettuja).
    total_read_time = 0.0
    total_stabilize_compute_time = 0.0
    total_warp_remap_time = 0.0

    # ENABLE_SHADOW_TOLERANT_STABILIZATION (kayttajan huomio: tama EI
    # ollut mukana "Yhteensa mitattu" -summassa aiemmin, vaikka se
    # ajetaan joka elavan seurannan framella - raportti siis ALIARVIOI
    # kokonaisajan kun se on paalla).
    total_shadow_suppress_time = 0.0

    executor = ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    )

    # HAKU (stone_tracker.search_new_stone) ja SEURANTA (stone_tracker.
    # track_stones_batch) ovat riippumattomia (molemmat lukevat vain
    # jo valmiin frame_u:n, eivat toistensa tulosta) - kayttajan
    # pyynnosta HAKU kaynnistetaan omalle taustasaikeelle JA SEURANTA
    # ajetaan SAMAAN AIKAAN paasaikeessa, katso alempana "HAKU:
    # kaynnistetaan..." ja "HAKU:n tuloksen keraaminen...". Uuden
    # kiven rekisterointi pysyy SILTI samalla framella kuin ennen -
    # HAKU:n tulos noudetaan (.result()) SEURANTAN VALMISTUTTUA,
    # ennen seuraavaan frameen siirtymista, EI viivastu. Vain 1
    # HAKU-kutsu voi olla kerrallaan kesken (SEARCH_EVERY_N_FRAMES
    # varmistaa etta edellinen on aina jo koottu ennen seuraavaa).
    haku_executor = ThreadPoolExecutor(max_workers=1)

    def _run_haku_timed(*args):
        t0 = time.time()
        result = stone_tracker.search_new_stone(*args)
        return result, time.time() - t0

    try:

        while True:

            t_read0 = time.perf_counter()
            frame = engine.read()
            total_read_time += time.perf_counter() - t_read0

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
            # PANEELIEN SEURANTA + STABILOINTIMATRIISI - AJETAAN
            # ENAA VAIN MOODIKUVAN RAKENTAMISEN AIKANA (kayttajan
            # pyynnosta empiirisesti korjattu, katso PIKSELITARKKA
            # SUORA STABILOINTI -kommentti alempana): panel_
            # stabilization_matrix EI ENAA OLE KAYTOSSA "loppuvideon"
            # stabilointiin kalibroinnin jalkeen (suora moodikuva-
            # vertailu korvasi senkin), joten koko paneiliseuranta+
            # RANSAC-laskenta (mitattu 37 ms/ruutu - lahes koko 25fps-
            # reaaliaikabudjetti) ohitetaan silloin kokonaan sen sijaan
            # etta laskettaisiin turhaan joka framella.
            #
            # HUOM (korjattu bugi): t_stab0 on asetettava TASSA, ENNEN
            # if calib_result is None -haaraa, koska total_stabilize_
            # compute_time += time.perf_counter() - t_stab0 lasketaan
            # AINA (myos "loppuvideon" suoran stabiloinnin ajaksi) -
            # jos t_stab0 asetettaisiin vain haaran SISALLA, se jaisi
            # jaatyneeksi vanhaan arvoon heti kun calib_result valmistuu,
            # ja mittari kasvaisi rajattomasti (havaittu: 50000 ms/ruutu).
            # ------------------------------------------------

            t_stab0 = time.perf_counter()

            if calib_result is None:

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

                t_stab0 = time.perf_counter()

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

                panel_stabilization_matrix = (
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

                        panel_stabilization_matrix = (
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
                    panel_stabilization_matrix.copy()
                )

                panel_stabilization_matrix = np.median(
                    np.stack(
                        stabilization_history,
                        axis=0
                    ),
                    axis=0
                )

                panel_stabilization_matrix = np.asarray(
                    panel_stabilization_matrix,
                    dtype=np.float64
                )


                previous_stabilization_matrix = (
                    panel_stabilization_matrix.copy()
                )

            # ------------------------------------------------
            # PIKSELITARKKA SUORA STABILOINTI "LOPPUVIDEOLLE" MOODIKUVAAN
            # VERRATEN (kayttajan pyynnosta empiirisesti korjattu, katso
            # keskusteluhistoria): AIEMPI versio KETJUTTI joka framen
            # pienen deltan edellisen framen jo-vahvistettuun sijaintiin,
            # olettaen etta koko framen kaytto (ei vain muutama paneelin
            # kulmapiste) estaa karkaamisen. Kayttajan oma lyhyt testi
            # osoitti etta kivi/tracking EI silti alkanut samasta nakymasta
            # kuin moodikuva - oma diagnostiikka (Linux-ymparistossa, 65s
            # klippi) VAHVISTI tarkan syyn: KETJUTUS ON KUMULATIIVINEN
            # SUMMA jokaisen framen pienesta mittausvirheesta (klassinen
            # "dead reckoning" -ongelma), ja vaikka yksittainen delta on
            # pieni, se kertyy - samalla klipilla ajautui 60 sekunnissa yli
            # 70 pikselia sivuun (fg-pikselien maara kasvoi 13186:sta
            # 752325:een). SUORA vertailu joka framella KIINTEAAN
            # moodikuvareferenssiin sen sijaan EI kerry - jokainen mittaus
            # on itsenainen (ei riipu edellisesta framesta), joten
            # yksittaisen framen mittausvirhe (esim. liikkuva pelaaja
            # hairitsee vaihekorrelaatiota) EI jaa pysyvasti - seuraava
            # frame korjaa itsensa taas suoraan referenssiin, ei edellisen
            # (jo vaaran) framen pohjalta. SAMALLA klipilla mitattu siirtyma
            # pysyi koko 60s ajan alle 1 pikselin (kamera on kaytannossa
            # paikallaan) - katso diag_direct_ref_fix.py-tulokset.
            #
            # Vertailu tehdaan RAA'ASSA (ei viela vaannon/undistortin
            # jalkeisessa) pikseliavaruudessa suoraan calib["frame"]:a
            # (moodikuvan RAAKA, paneilistabiloitu mutta EI viela
            # undistorted versio) vasten - loppuvideo_ref_gray on
            # valimuistitettu heti kun calib_result tulee valmiiksi (katso
            # sen laskenta ylempana). Paneiliseurantaa/panel_stabilization_
            # matrix:ia kaytetaan siis enaa VAIN moodikuvan rakentamiseen
            # (calib_result is None) - sen jalkeen (profiiliskannaus +
            # elava seuranta) KOKO frame vaihekorrelaatiolla suoraan
            # referenssiin, EI ketjutettuna.
            # ------------------------------------------------

            if calib_result is None:

                stabilization_matrix = panel_stabilization_matrix

            else:

                # HUOM (korjattu merkkivirhe, kayttajan "miksi viivat eivat
                # haviaisi/miksi huononee" -kysymysten paljastama, katso
                # keskusteluhistoria): _phase_correlate_full_frame(A,B)
                # palauttaa siirtyman JOLLA A (referenssi) piti siirtaa
                # jotta saadaan B (nykyinen frame) - EI siirtyma jolla B
                # pitaa siirtaa jotta saadaan A. Nama ovat VASTAKKAISET
                # (kaanteiset toisiinsa nahden) puhtaalle translaatiolle.
                # Koska frame pitaa siirtaa PAIN referenssia (ei referenssi
                # pain framea), oikea korjaus on -dx,-dy, EI dx,dy suoraan.
                # Vahvistettu oikealla videodatalla (MAH00014, t=659s):
                # vanha (suora dx,dy) merkki antoi taustanvaimennuksessa
                # 266403 vuotanutta pikselia - HUONOMMIN kuin EI MITAAN
                # korjausta (234413) - kun taas -dx,-dy antoi 194114
                # (selvasti paras). Tama selittaa miksi reunaviivat eivat
                # havinneet kokonaan eivatka pysyneet tasaisina ajan
                # mukana: vaara suunta kasvatti virhetta sita enemman mita
                # suurempi todellinen siirtyma oli.
                dx, dy = _phase_correlate_full_frame(
                    loppuvideo_ref_gray, gray
                )

                stabilization_matrix = np.array(
                    [[1.0, 0.0, -dx], [0.0, 1.0, -dy]],
                    dtype=np.float64
                )

            total_stabilize_compute_time += time.perf_counter() - t_stab0

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
                        "(calibrate_camera_from_image_with_seed + "
                        "build_pose_from_calibration)..."
                    )

                    calib = calibrate_camera_from_image_with_seed(
                        calib_mode_output
                    )

                    pose = build_pose_from_calibration(calib)

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
                    # LOPPUVIDEON SUORAN STABILOINNIN REFERENSSI
                    # (kayttajan pyynnosta empiirisesti korjattu, katso
                    # PIKSELITARKKA SUORA STABILOINTI -kommentti alempana
                    # kaytonkohdalla): valimuistitetaan moodikuvan RAAKA
                    # (ei viela undistorted) harmaasavykuva heti kun
                    # calib_result on valmis - jokainen "loppuvideon"
                    # frame verrataan TASTA LAHTIEN suoraan tahan, EI
                    # ketjutettuna edellisen framen sijaintiin (katso
                    # miksi ketjutus hylattiin).
                    # ------------------------------------------------

                    loppuvideo_ref_gray = cv2.cvtColor(
                        calib["frame"], cv2.COLOR_BGR2GRAY
                    )

            # ------------------------------------------------
            # KOKO RADAN SKANNAUS LIIKKUVAN KIVEN LOYTAMISEKSI
            # (3D-PROFIILIA VARTEN) - vasta kun kalibrointi on
            # valmis (tarvitaan calib/pose fyysisten sijaintien
            # laskentaan) JA profiili ei viela ole riittava.
            # ------------------------------------------------

            elif profile_result is None:

                if frame_index >= next_stone_scan_frame:

                    # Kandidaattitunnistuksen syote stabiloidaan SAMALLA
                    # matriisilla kuin moodikuva-referenssi aikanaan
                    # laskettiin (engine.add_mode_frame) - muuten
                    # taustavertailu (suppress_static_background) ei
                    # osu kohdalleen, ja pikselikoordinaatit eivat
                    # vastaa poseen kalibrointireferenssia.
                    stabilized_scan_frame = cv2.warpAffine(
                        frame, stabilization_matrix, (width, height)
                    )

                    curr_candidates = _scan_stone_candidates(
                        stabilized_scan_frame, calib_result["calib"],
                        calib_result["pose"],
                        background_reference=calib_result["calib"]["frame"]
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
                            seed_pos_cm=seed_pos,
                            background_reference_undistorted=(
                                calib_result["calib"]["frame_undistorted"]
                            )
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
                                    # frame_idx/pos_cm: EI kayteta
                                    # profiilisovitukseen (try_fit_
                                    # profile), vain myohempaan
                                    # varireferenssin rakennukseen
                                    # (build_stone_color_reference) -
                                    # katso kommentti COLOR_REF_*-
                                    # vakioiden kohdalla.
                                    "frame_idx": track[i]["frame_idx"],
                                    "pos_cm": track[i]["pos_cm"],
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
                                n_accepted_stones += 1

                                profile, riittava = try_fit_profile(
                                    calib_result["pose"],
                                    accumulated_stones
                                )

                                if riittava:

                                    best_sufficient_profile = profile

                                    if (
                                        n_accepted_stones >=
                                        PROFILE_MIN_ACCEPTED_STONES
                                    ):

                                        print(
                                            "3D-kiviprofiili riittava "
                                            f"({n_accepted_stones} "
                                            "hyvaksyttya kiveä) - "
                                            "lopetetaan koko radan "
                                            "skannaus."
                                        )

                                        profile_result = profile

                                    else:

                                        print(
                                            f"  profiili jo riittava, "
                                            "mutta jatketaan viela "
                                            f"lisaa kivia varten "
                                            f"({n_accepted_stones}/"
                                            f"{PROFILE_MIN_ACCEPTED_STONES})."
                                        )

                    prev_scan_candidates = curr_candidates
                    next_stone_scan_frame = frame_index + stone_scan_interval_frames

            # ------------------------------------------------
            # ELAVA MONI-KIVEN SEURANTA + CSV - vasta kun SEKA
            # kalibrointi ETTA 3D-kiviprofiili ovat valmiit.
            # ------------------------------------------------

            else:

                if live_state is None:

                    profile = profile_result
                    R_max = profile["R_max_cm"]
                    H_total = profile["H_total_cm"]
                    shape_deltas = profile["shape_deltas"]

                    camera_matrix = calib_result["calib"]["camera_matrix"]
                    dist_coeffs = np.array(
                        [calib_result["calib"]["best_k1"], 0.0, 0.0, 0.0, 0.0],
                        dtype=np.float64
                    )

                    map1, map2 = _build_undistort_maps(
                        camera_matrix, dist_coeffs, (width, height)
                    )

                    local_pts_body = build_local_stone_rings(
                        R_max, H_total, shape_deltas, n_theta=28, n_per_segment=5
                    )
                    local_pts_search = build_local_stone_rings(
                        R_max, H_total, shape_deltas,
                        n_theta=SEARCH_HULL_N_THETA,
                        n_per_segment=SEARCH_HULL_N_PER_SEGMENT
                    )

                    # stone_tracker.cpp:n refine_position_joint-portin
                    # ring_r_frac_guess - alkuarvaus ERILLISELLE saturaatio-
                    # pohjaiselle "rengashaulle" (kahvan kiinnityslevyn reunan
                    # etsintaan, katso stone_tracker.cpp:n oma kommentti) -
                    # kaytetaan sovitettua handle_r_frac:ia myos taman haun
                    # alkuarvauksena, koska se on parempi (datasta sovitettu)
                    # lahtokohta kuin vanha kiintea HANDLE_NOTCH_R_FRAC-vakio.
                    #
                    # handle_r_frac - kahvan aiheuttaman kolon SADE, joka
                    # VAHENNETAAN graniittirungon konveksista peitteesta
                    # (predictedNotchHull/profileResiduals, katso kamera9_01.
                    # py:n HANDLE_NOTCH_R_FRAC-kommentti). Kayttajan pyynnosta
                    # fit_stone_profile SOVITTAA taman datasta, ja stone_
                    # tracker.cpp:n elava seuranta kayttaa NYT samaa sovitettua
                    # arvoa ajonaikaisena parametrina (EI enaa kiintea C++-vakio).
                    ring_r_frac_guess = profile["handle_r_frac"]
                    handle_r_frac = profile["handle_r_frac"]

                    print(
                        "Rakennetaan kiven pintavarireferenssia "
                        f"({len(accumulated_stones)} havainnosta)..."
                    )

                    stone_color_reference = build_stone_color_reference(
                        video_file, calib_result["calib"],
                        calib_result["pose"], local_pts_body, 28,
                        accumulated_stones
                    )

                    csv_file = open(csv_output, "w", newline="")
                    csv_writer = csv.writer(csv_file)
                    csv_writer.writerow(CSV_HEADER)

                    if debug_video_output is not None:

                        debug_video_writer = cv2.VideoWriter(
                            debug_video_output,
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            fps, (width, height)
                        )

                    print()
                    print(
                        f"Elava moni-kiven seuranta alkaa (frame "
                        f"{frame_index}) - CSV: {csv_output}"
                    )

                    live_state = {
                        "map1": map1, "map2": map2,
                        "local_pts_body": local_pts_body,
                        "local_pts_search": local_pts_search,
                        "R_max": R_max, "H_total": H_total,
                        "ring_r_frac_guess": ring_r_frac_guess,
                        "handle_r_frac": handle_r_frac,
                        "color_reference": stone_color_reference,
                    }

                # HUOM: kokeiltiin taalla yhdistaa warpAffine+remap
                # yhdeksi remap-kutsuksi (cv2.transform map1/map2:lle +
                # yksi remap) - eristetty mikrobenchmark nayttii ensin
                # parannusta, mutta TOISTETTUNA (ja koko putken sisalla)
                # tulos vaihteli suunnasta toiseen ajojen valilla - ero
                # oli taman hiekkalaatikon oman kohinan sisalla, EI
                # luotettavasti mitattavissa (map1/map2 ovat float32
                # 2-kanavaisia, 8 tavua/pikseli - enemman dataa per
                # pikseli kuin itse BGR-kuva, joten "halpa" koordinaatti-
                # muunnos ei ollutkaan niin halpa). Palautettu alku-
                # peraiseen kahteen erilliseen vaiheeseen - EI otettu
                # kayttoon todentamatonta optimointia.
                t_warp0 = time.perf_counter()
                stabilized = cv2.warpAffine(
                    frame, stabilization_matrix, (width, height)
                )
                frame_u = cv2.remap(
                    stabilized, live_state["map1"], live_state["map2"],
                    interpolation=cv2.INTER_LINEAR
                )
                total_warp_remap_time += time.perf_counter() - t_warp0

                timestamp = frame_index / fps
                pose = calib_result["pose"]
                local_pts_body = live_state["local_pts_body"]
                local_pts_search = live_state["local_pts_search"]

                # --------------------------------------------
                # ENABLE_SHADOW_TOLERANT_STABILIZATION (katso sen kommentti
                # taman tiedoston alkupaassa). HUOM: frame_u ITSE (vari-
                # tarkistus, debug-video) EI koskaan taustanvaimenneta tassa
                # - vain erillinen frame_u_for_tracking-kopio, jota kaytetaan
                # VAIN HAKU/SEURANTA-kutsuissa alla.
                #
                # ENABLE_SUBPIXEL_ALIGNMENT:in erillinen jalkikorjaus (tama
                # kommenttiblokki oli aiemmin tassa) POISTETTU (kayttajan
                # havainto + oma empiirinen vahvistus, katso keskustelu-
                # historia): kayttaja huomasi etta jaa/katto/seinat eivat
                # olleet kaytannossa taysin valkoisia debug-videossa - oma
                # testi paljasti etta tama jalkikorjaus (alunperin suunni-
                # teltu PIENEKSI hienosaadoksi epatarkan paneilipohjaisen
                # stabiloinnin paalle) on NYT TARPEETON JA HAITALLINEN, koska
                # PAASTABILOINTI (stabilization_matrix, katso PIKSELITARKKA
                # SUORA STABILOINTI ylempana) tekee jo TASMALLEEN saman
                # suoran moodikuvavertailun - jalkikorjaus vain mittasi
                # lahes saman asian UUDELLEEN ja lisasi sen PAALLE (havaittu:
                # jalkikorjaus -1.7px samaan suuntaan kuin paastabiloinnin jo
                # tekema -0.89px korjaus), mika KASVATTI reunapikselien
                # vuotoa taustanvaimennuksessa 115599:sta 178113:een (+54%)
                # samalla testiframella, ei pienentanyt sita.
                # --------------------------------------------
                ref_undist_live = calib_result["calib"]["frame_undistorted"]

                # VALOTASAPAINO/KIRKKAUS - vain kerran sekunnissa (katso
                # estimate_photometric_correction:in kommentti) - ei
                # kosketa frame_u:ta itsea (varitarkistus/debug-video),
                # vain erillinen frame_u_photo-kopio taustanvaimennukselle.
                if (
                    photo_gain is None
                    or frame_index >= next_photo_update_frame
                ):
                    t_photo0 = time.perf_counter()
                    photo_gain, photo_bias = estimate_photometric_correction(
                        frame_u, ref_undist_live
                    )
                    total_photometric_time += time.perf_counter() - t_photo0
                    next_photo_update_frame = frame_index + max(
                        1, int(round(fps))
                    )

                frame_u_photo = apply_photometric_correction(
                    frame_u, photo_gain, photo_bias
                )

                if ENABLE_SHADOW_TOLERANT_STABILIZATION:
                    t_shadow0 = time.perf_counter()
                    frame_u_for_tracking = suppress_static_background(
                        frame_u_photo, ref_undist_live, diff_threshold=GRANITE_DIFF_THRESHOLD
                    )
                    total_shadow_suppress_time += time.perf_counter() - t_shadow0
                    # frame_u_for_tracking on jo taustanvaimennettu (myos
                    # varjonsietoisesti) - C++:n OMA sisainen vaimennus
                    # HAKU/SEURANTA-kutsuissa ohitetaan antamalla sille 0.0,
                    # jotta frame_u_for_tracking:ia ei vaimenneta uudelleen.
                    haku_seuranta_diff_threshold = 0.0
                else:
                    frame_u_for_tracking = frame_u_photo
                    haku_seuranta_diff_threshold = GRANITE_DIFF_THRESHOLD

                # --------------------------------------------
                # HAKU: uusia kiviä kiinteältä paata-rajatulta
                # vyohykkeelta (kamera9_02.py:n SEARCH_*), vain
                # jos tilaa (< MAX_CONCURRENT_STONES) ja tama on
                # hakutarkistusframe. C++-porttaus (stone_tracker.
                # cpp:n search_new_stone, Task 6) - ristikkohaku
                # JA yhteissovitus yhdessa kutsussa, samaan tapaan
                # kuin SEURANTA (Task 5).
                #
                # KAYNNISTETAAN OMALLE TAUSTASAIKEELLE (kayttajan
                # pyynnosta): HAKU ja SEURANTA ovat riippumattomia
                # (molemmat lukevat vain jo valmiin frame_u:n,
                # eivat toistensa tulosta talta framelta), joten
                # ne ajetaan SAMANAIKAISESTI - HAKU taustasaikeessa,
                # SEURANTA paasaikeessa alempana. TULOS KERATAAN
                # VASTA SEURANTAN JALKEEN (katso "HAKU:n tuloksen
                # keraaminen" alempana) - uuden kiven rekisterointi
                # pysyy SILTI TASMALLEEN samalla framella kuin ennen,
                # EI viivastu, koska odotamme HAKU:n tuloksen ennen
                # seuraavaan frameen siirtymista. Katso stone_tracker.
                # cpp:n search_new_stone/track_stones_batch -kommentit
                # (ScopedSingleThreadedOpenCV) OpenCV:n oman sisaisen
                # rinnakkaistuksen turvallisesta poiskytkennasta taman
                # samanaikaisuuden ajaksi.
                # --------------------------------------------

                haku_future = None

                if (
                    len(active_stones) < MAX_CONCURRENT_STONES
                    and frame_index % haku_interval_frames == 0
                ):

                    x_center = 0.0
                    y_center = (
                        SEARCH_Y_MIN_CM + SEARCH_Y_MAX_CM
                    ) / 2.0
                    y_half = (
                        SEARCH_Y_MAX_CM - SEARCH_Y_MIN_CM
                    ) / 2.0

                    haku_future = haku_executor.submit(
                        _run_haku_timed,
                        frame_u_for_tracking, ref_undist_live,
                        local_pts_body, local_pts_search,
                        pose["K"], pose["R"], pose["t"],
                        x_center, SEARCH_X_HALF_WIDTH_CM, y_center, y_half,
                        SEARCH_COARSE_STEP_CM, SEARCH_FINE_STEP_CM,
                        SEARCH_SCORE_THRESHOLD,
                        live_state["R_max"], live_state["H_total"],
                        live_state["ring_r_frac_guess"], live_state["handle_r_frac"],
                        haku_seuranta_diff_threshold
                    )

                # --------------------------------------------
                # SEURANTA: paivitetaan JOKAINEN aktiivinen kivi
                # JOKA frame - C++-porttaus (stone_tracker.cpp,
                # Task 5) laskee KAIKKIEN taman framen SEURANTA-
                # kivien ROI-rajatun maskin/saturaation, ristikko-
                # haun ja LM-yhteissovituksen YHDESSA std::thread-
                # rinnakkaistetussa kutsussa (yksi worker-saie per
                # kivi, atominen tyonvarastus - katso mode_engine.
                # cpp:n save_mode-kommentti samasta periaatteesta).
                # TASMALLEEN sama matematiikka/algoritmi kuin
                # locate_by_grid_search_fast + refine_
                # position_joint_fast - katso stone_tracker.cpp:n
                # oma kommentti numeerisesta validoinnista (oikealla
                # videolla, Python-tulosta vasten) seka tiedoston
                # alun kommentti sen tunnetusta, algoritmin OMASTA
                # (ei porttausvirheen) numeerisesta herkkyydesta
                # refine_position_joint:in MAD-poikkeavien-hylkays-
                # kynnyksella. Kayttajan pyynnosta (katso git-historia):
                # SAA nyt taustanvaimennuksen (calib_result:in staattinen
                # referenssikuva) SAMOIN kuin HAKU jo aiemmin - ilman
                # tata kivi menetti tarkkuutensa (rms_px jopa 130-146px
                # normaalin ~2px sijaan) aina kun se ylitti staattisen
                # jaamerkinnan (pesan renkaat, hogline, mainokset).
                # --------------------------------------------

                still_active = []

                # DEBUG_SAVE_TRACKING_VIDEO:in kerays - katso taman
                # tiedoston alkupaan kommentti. Lista (X_cm, Y_cm,
                # stone_id, vari_bgr, teksti) - piirretaan framelle
                # HAKU:n tuloksen keraamisen jalkeen alempana.
                debug_draw_items = []

                # HAKU:n (jos kaynnissa) mahdollisesti loytama uusi
                # kivi EI ole viela active_stones:issa tassa vaiheessa
                # (sen tulos kerataan vasta alempana) - ei siis
                # tarvetta erikseen suodattaa sita pois taalta.
                seuranta_stones = active_stones

                if seuranta_stones:

                    X0_arr = np.array(
                        [s["last_xy"][0] for s in seuranta_stones],
                        dtype=np.float64
                    )
                    Y0_arr = np.array(
                        [s["last_xy"][1] for s in seuranta_stones],
                        dtype=np.float64
                    )

                    # TRACK_MAX_SPEED_Y/X_CM_S:in kommentti (taman
                    # tiedoston alkupaassa): hakualue lasketaan JOKA
                    # framella JOKAISELLE kivelle erikseen suoraan
                    # fysikaalisesta nopeusrajasta ja siita kuinka monta
                    # frameä on todellisuudessa kulunut viimeisimmasta
                    # VAHVISTETUSTA sijainnista (misses+1, koska last_xy
                    # ei paivity misseilla) - EI enaa kamera9_02.py:n
                    # kiinteaa, isotrooppista TRACK_HALF_RANGE_CM:ia.
                    elapsed_frames_arr = np.array(
                        [s["misses"] + 1 for s in seuranta_stones],
                        dtype=np.float64
                    )
                    elapsed_seconds_arr = elapsed_frames_arr / fps
                    half_range_y_arr = np.minimum(
                        TRACK_MAX_SPEED_Y_CM_S * elapsed_seconds_arr, TRACK_HALF_RANGE_MAX_CM
                    )
                    half_range_x_arr = np.minimum(
                        TRACK_MAX_SPEED_X_CM_S * elapsed_seconds_arr, TRACK_HALF_RANGE_MAX_CM
                    )

                    t_seuranta0 = time.time()
                    batch_results = stone_tracker.track_stones_batch(
                        frame_u_for_tracking, ref_undist_live,
                        X0_arr, Y0_arr,
                        half_range_x_arr, half_range_y_arr,
                        local_pts_body, local_pts_search,
                        pose["K"], pose["R"], pose["t"],
                        TRACK_COARSE_STEP_CM, TRACK_FINE_STEP_CM,
                        TRACK_SCORE_THRESHOLD,
                        live_state["R_max"], live_state["H_total"],
                        live_state["ring_r_frac_guess"], live_state["handle_r_frac"],
                        TRACK_MAX_BACKWARD_CM,
                        haku_seuranta_diff_threshold
                    )
                    total_seuranta_time += time.time() - t_seuranta0
                    n_seuranta_calls += 1
                    n_seuranta_stone_updates += len(seuranta_stones)

                    color_ref = live_state["color_reference"]
                    frame_u_f64 = None

                    for s, refined in zip(seuranta_stones, batch_results):

                        # --------------------------------
                        # VARIHISTORIAN KERAYS (COLOR_DEBUG-diagnostiikkaa
                        # varten - EI enaa vaikuta "pysahtynyt"-paatokseen,
                        # katso kommentti COLOR_REF_*-vakioiden kohdalla
                        # taman tiedoston alkupaassa). HUOM (kayttajan pyynnosta
                        # tehty uudelleensuunnittelu): tama EI vaikuta
                        # MITENKAAN normaaliin SEURANTAan (haku-ankkuri,
                        # s["misses"], "kadotettu") - position-/miss-
                        # logiikka alla on identtinen kuin ennen vari-
                        # tarkistuksen lisaamista, joten heittaja/
                        # lakaisija kiven paalla EI VOI aiheuttaa kiven
                        # katoamista sen enempaa kuin ennenkaan. Vari
                        # keraytyy vain LIUKUVAAN IKKUNAAN (color_diff_
                        # history, sama aikaikkuna kuin position_history)
                        # ja sen KESKIARVOA kaytetaan LISAEHTONA vasta
                        # "pysahtynyt"-ilmoituksen kohdalla (alempana) -
                        # jos vari ei vahvista, ilmoitus vain LYKKAANTYY
                        # (kivi pysyy normaalisti seurattuna), ei koskaan
                        # katoa/vaaristu taman takia.
                        # --------------------------------

                        if refined["found"] and color_ref is not None:

                            if frame_u_f64 is None:
                                frame_u_f64 = frame_u.astype(np.float64)

                            median_diff = color_match_median_diff(
                                frame_u_f64, local_pts_body, pose,
                                refined["X_cm"], refined["Y_cm"],
                                color_ref, width, height
                            )

                            if median_diff is not None:

                                diff_history = s["color_diff_history"]
                                diff_history.append((frame_index, median_diff))

                                while (
                                    diff_history
                                    and frame_index - diff_history[0][0]
                                    > stop_tracking_frames
                                ):
                                    diff_history.pop(0)

                                if os.environ.get("COLOR_DEBUG"):
                                    print(
                                        f"[COLOR_DEBUG] frame={frame_index} "
                                        f"stone={s['stone_id']} "
                                        f"diff={median_diff:.3f}"
                                    )

                        # --------------------------------
                        # KUMULATIIVINEN "EI TAAKSEPAIN" -TARKISTUS
                        # (kayttajan pyynnosta, katso keskusteluhistoria):
                        # stone_tracker.cpp:n trackStoneUpdateOne hylkaa jo
                        # YHDEN paivityksen joka siirtaisi kiven >100cm
                        # taaksepain EDELLISESTA framesta - mutta tama EI
                        # estä montaa PIENTA (<100cm) taaksepain-askelta
                        # kasautumasta suureksi ajautumaksi usean sekunnin/
                        # minuutin aikana (havaittu koko videon lapikaynnissa:
                        # levossa ollut kivi "liukui" pikkuhiljaa kohti
                        # heittopaata, todennakoisesti SEURANNAN tarttuessa
                        # kiven vierella kavelevaan pelaajaan/lakaisijaan -
                        # tama tayttaa lopulta KAIKKI MAX_CONCURRENT_STONES-
                        # paikat haamuilla, estaen uusien aitojen heittojen
                        # rekisteroinnin). Verrataan siis KOKO elinkaaren
                        # PIENIMPAAN havaittuun Y-arvoon (s["min_y_seen"],
                        # ei vain edelliseen frameen) - kumulatiivinen
                        # ajautuma yli TRACK_MAX_BACKWARD_CM:n hylataan
                        # samoin kuin yksittainen liian iso hyppy.
                        # --------------------------------

                        if refined["found"] and (
                            refined["Y_cm"] - s["min_y_seen"] > TRACK_MAX_BACKWARD_CM
                        ):
                            refined = dict(refined)
                            refined["found"] = False

                        # --------------------------------
                        # KESKIVIIVAETAISYYSTARKISTUS: katso ASETUKSET-
                        # kommentti MAX_ABS_X_FROM_CENTERLINE_CM:in
                        # kohdalla - hylataan (kuten TAAKSEPAIN-tarkistus
                        # ylla) havainto joka veisi kiven liian kauas
                        # keskiviivasta, todennakoisesti SEURANNAN
                        # ajauduttua sivuttain liikkuvaan kohteeseen.
                        # --------------------------------

                        if refined["found"] and (
                            abs(refined["X_cm"]) > MAX_ABS_X_FROM_CENTERLINE_CM
                        ):
                            refined = dict(refined)
                            refined["found"] = False

                        if refined["found"]:

                            s["last_xy"] = (
                                refined["X_cm"], refined["Y_cm"]
                            )
                            s["min_y_seen"] = min(
                                s["min_y_seen"], refined["Y_cm"]
                            )
                            s["misses"] = 0

                            debug_draw_items.append((
                                refined["X_cm"], refined["Y_cm"],
                                s["stone_id"],
                                (0, 255, 0) if refined["tarkka"] else (0, 165, 255),
                                str(s["stone_id"])
                            ))

                            # --------------------------------
                            # LIIKEVAHVISTUS: katso ASETUKSET-kommentti
                            # MIN_CONFIRMED_THROW_DISPLACEMENT_CM:in
                            # kohdalla - ei kirjoiteta CSV:hen ENNEN
                            # kuin kivi on siirtynyt riittavasti ensim-
                            # maisesta havainnostaan (torjuu paikallaan
                            # jo olevien kohteiden toistuvan virheelli-
                            # sen uudelleenrekisteroinnin). Puskuroidut
                            # rivit kirjoitetaan KAIKKI KERRALLA heti
                            # kun vahvistus tapahtuu - ei menetetä aikai-
                            # sia, aitoja havaintoja.
                            # --------------------------------

                            reject_low_tarkka = False

                            if not s["confirmed"]:

                                s["pending_rows"].append(
                                    (frame_index, timestamp, dict(refined))
                                )

                                if (
                                    abs(refined["Y_cm"] - s["y0_first_cm"])
                                    >= MIN_CONFIRMED_THROW_DISPLACEMENT_CM
                                ):

                                    # --------------------------------
                                    # TARKKA-OSUUSTARKISTUS: katso
                                    # ASETUKSET-kommentti MIN_PRECONFIRM_
                                    # TARKKA_OBSERVATIONS/FRACTION:in
                                    # kohdalla - torjuu pelaajan/
                                    # lakaisijan SEURANNAN loysemman
                                    # per-frame-kynnyksen kautta
                                    # "ajautumisen" virheellisesti
                                    # vahvistetuksi kiveksi.
                                    # --------------------------------

                                    n_pending = len(s["pending_rows"])
                                    n_tarkka = sum(
                                        1 for _, _, prow in s["pending_rows"]
                                        if prow.get("tarkka")
                                    )
                                    tarkka_frac = (
                                        n_tarkka / n_pending
                                        if n_pending > 0 else 1.0
                                    )

                                    if (
                                        n_pending >= MIN_PRECONFIRM_TARKKA_OBSERVATIONS
                                        and tarkka_frac < MIN_PRECONFIRM_TARKKA_FRACTION
                                    ):

                                        reject_low_tarkka = True
                                        s["pending_rows"] = []
                                        print(
                                            f"[frame {frame_index}] Ehdokas "
                                            f"{s['stone_id']} hylatty (tarkka-"
                                            f"osuus {n_tarkka}/{n_pending} "
                                            f"({tarkka_frac:.0%}) liian "
                                            "matala ennen liikevahvistusta "
                                            "- todennakoisesti pelaaja/"
                                            "lakaisija, ei aito kivi)."
                                        )

                                    else:

                                        s["confirmed"] = True
                                        n_confirmed_stones += 1

                                        for pf, pt, prow in s["pending_rows"]:
                                            _write_stone_csv_row(
                                                csv_writer, pf, pt,
                                                s["stone_id"], prow
                                            )
                                            s["confirmed_tarkka_window"].append(
                                                bool(prow.get("tarkka"))
                                            )

                                        s["pending_rows"] = []

                            else:

                                # --------------------------------
                                # JATKUVA TARKKA-OSUUSTARKISTUS (Testi_
                                # 02_02, kayttajan pyynnosta - katso
                                # keskusteluhistoria): MIN_PRECONFIRM_
                                # TARKKA_* tarkistaa tarkka-osuuden VAIN
                                # kerran, liikevahvistushetkella. Havait-
                                # tiin oikealla datalla (0001/MAH00014,
                                # jaasuodatuksen jalkeen) etta muutama
                                # kandidaatti (esim. pyyhkija joka kavelee
                                # pitkan matkan aidon kiven vieressa)
                                # lapaisee talla hetkella riittavan
                                # tarkka-osuuden mutta putoaa sen jalkeen
                                # pysyvasti matalaksi (esim. 9-28% n. 400
                                # havainnon ajan) - vahentaa 400+ vaarin
                                # CSV-riviä 50:aan. Kynnys (0.5, n>=50)
                                # validoitu koko videoiden oikealla
                                # datalla: kaikki selvasti aidot pitkaan
                                # seuratut kivet pysyivat aina >=51.5%:ssa
                                # (marginaali), kaikki selvasti ongelmal-
                                # liset kandidaatit alittivat 0.5:n heti
                                # ensimmaisen 50 havainnon jalkeen (13-48%).
                                #
                                # LIUKUVA IKKUNA (ei enaa kumulatiivinen
                                # koko elinkaarelta, kayttajan pyynnosta
                                # tehdyn maskiyhdistelma-korjauksen
                                # (stone_tracker.cpp, createForeground-
                                # FromWhitened) jalkeen havaittu tarve):
                                # aito kivi voi olla hetken (esim. juuri
                                # heiton jalkeen, heittaja/lakaisija viela
                                # vierella) kosketuksissa pelaajaan JA
                                # SEN JALKEEN seurata puhtaasti pitkan
                                # matkaa yksin - kumulatiivinen koko-
                                # elinkaaren osuus jaisi talloin PYSYVASTI
                                # alle kynnyksen tuon alkuhetken takia,
                                # vaikka loppuosa olisi taydellinen (havait-
                                # tiin: n_body 26-29/rms 0.5-0.9px koko
                                # lopun ajan, mutta silti hylattiin 49
                                # havainnon kohdalla). Ikkuna (deque,
                                # maxlen=MIN_CONFIRMED_TARKKA_OBSERVATIONS)
                                # "unohtaa" vanhan kontaminaation automaat-
                                # tisesti kun tarpeeksi uusia havaintoja on
                                # kertynyt, mutta havaitsee silti PYSYVAN
                                # ajautumisen (jolloin koko ikkuna on jat-
                                # kuvasti matala) - sama kynnys (0.5,
                                # n>=50) sailyy, koska se on jo validoitu.
                                # --------------------------------

                                s["confirmed_tarkka_window"].append(
                                    bool(refined.get("tarkka"))
                                )
                                window = s["confirmed_tarkka_window"]

                                if (
                                    len(window) >= MIN_CONFIRMED_TARKKA_OBSERVATIONS
                                    and (sum(window) / len(window))
                                    < MIN_CONFIRMED_TARKKA_FRACTION
                                ):

                                    reject_low_tarkka = True
                                    print(
                                        f"[frame {frame_index}] Kivi "
                                        f"{s['stone_id']} lopetetaan "
                                        f"(tarkka-osuus (liukuva ikkuna) "
                                        f"{sum(window)}/{len(window)} "
                                        f"({100*sum(window)/len(window):.0f}%) "
                                        "pudonnut pysyvasti liian matalaksi "
                                        "- todennakoisesti SEURANTA ajautunut "
                                        "pelaajaan/lakaisijaan aidon kiven "
                                        "vierella)."
                                    )

                                else:

                                    _write_stone_csv_row(
                                        csv_writer, frame_index, timestamp,
                                        s["stone_id"], refined
                                    )

                            # --------------------------------
                            # PYSAHTYMISTARKISTUS: katso taman
                            # tiedoston alkupaan kommentti STOP_
                            # TRACKING_SECONDS/DISPLACEMENT_CM:sta.
                            # Liukuva ikkuna framen INDEKSIN, ei
                            # listan pituuden, mukaan - kestaa
                            # satunnaiset valiin jaavat missit.
                            # --------------------------------

                            history = s["position_history"]
                            history.append((
                                frame_index,
                                refined["X_cm"], refined["Y_cm"]
                            ))

                            while (
                                history[-1][0] - history[0][0]
                                > stop_tracking_frames
                            ):
                                history.pop(0)

                            stopped = False

                            if (
                                history[-1][0] - history[0][0]
                                >= stop_tracking_frames
                            ):
                                _, old_x, old_y = history[0]
                                displacement = math.hypot(
                                    s["last_xy"][0] - old_x,
                                    s["last_xy"][1] - old_y
                                )

                                if displacement < STOP_TRACKING_DISPLACEMENT_CM:

                                    # --------------------------------
                                    # PYSAHTYMINEN pelkasta GEOMETRIASTA
                                    # (kayttajan pyynnosta, katso keskus-
                                    # teluhistoria): aiemmin vaadittiin
                                    # TAMAN LISAKSI varivahvistus (viimeisen
                                    # sekunnin ka. vari COLOR_STOP_MAX_AVG_
                                    # DIFF:in alle) - havaittiin koko videon
                                    # lapikaynnissa etta tama esti "pysah-
                                    # tynyt"-ilmoituksen aidoilla, selvasti
                                    # paikallaan olevilla kivilla (esim.
                                    # kivi joka ei liikkunut yli minuuttiin
                                    # ei koskaan tulostanut "pysahtynyt" -
                                    # jaljelle jaaneet MAX_CONCURRENT_STONES-
                                    # paikat tukkeutuivat, estaen uusien
                                    # aitojen heittojen rekisteroinnin).
                                    # Nyt: 3D-mallin/maskin sijainti (EI
                                    # liikkunut yli STOP_TRACKING_DISPLACE-
                                    # MENT_CM:aa STOP_TRACKING_SECONDS:in
                                    # aikana) RIITTAA yksinaan.
                                    # --------------------------------

                                    stopped = True
                                    if s["confirmed"]:
                                        print(
                                            f"[frame {frame_index}] Kivi "
                                            f"{s['stone_id']} pysahtynyt "
                                            f"(liikkunut {displacement:.1f}cm "
                                            f"viimeisen {STOP_TRACKING_SECONDS:.0f}s "
                                            "aikana) - lopetetaan seuranta."
                                        )
                                    else:
                                        # Ei koskaan liikkunut riittavasti
                                        # (katso MIN_CONFIRMED_THROW_
                                        # DISPLACEMENT_CM) - todennakoisesti
                                        # jo paikallaan ollut kohde, ei aito
                                        # heitto. Puskuroidut havainnot
                                        # hylataan hiljaisesti (EI CSV-riviä).
                                        s["pending_rows"] = []
                                        print(
                                            f"[frame {frame_index}] Ehdokas "
                                            f"{s['stone_id']} hylatty "
                                            "(ei liikkunut riittavasti - "
                                            "todennakoisesti jo paikallaan "
                                            "ollut kohde, ei aito heitto)."
                                        )

                            if not stopped and not reject_low_tarkka:
                                still_active.append(s)

                        else:

                            s["misses"] += 1

                            debug_draw_items.append((
                                s["last_xy"][0], s["last_xy"][1],
                                s["stone_id"], (0, 0, 255),
                                f"{s['stone_id']} MISS"
                            ))

                            if s["misses"] < track_lost_max_misses:
                                still_active.append(s)
                            elif s["confirmed"]:
                                print(
                                    f"[frame {frame_index}] Kivi "
                                    f"{s['stone_id']} kadotettu."
                                )
                            else:
                                s["pending_rows"] = []
                                print(
                                    f"[frame {frame_index}] Ehdokas "
                                    f"{s['stone_id']} hylatty (kadotettu "
                                    "ennen kuin liikkui riittavasti - "
                                    "todennakoisesti jo paikallaan ollut "
                                    "kohde, ei aito heitto)."
                                )

                active_stones = still_active

                # --------------------------------------------
                # HAKU:n tuloksen keraaminen - SEURANTA (ylla) ehti
                # jo laskea RINNAN HAKU:n kanssa, joten odotus tassa
                # (.result(), jos HAKU on viela kesken) on vain sen
                # verran kuin HAKU oli SEURANTAa hitaampi (yleensa
                # HAKU on selvasti hitaampi -> odotus n. HAKU_aika -
                # SEURANTA_aika, joskus jopa 0 jos SEURANTA oli
                # hitaampi). Uusi kivi lisataan TASSA active_stones:
                # iin - TASMALLEEN samalla framella/timestampilla
                # kuin ennen, EI viivastynyt kayttaytyminen.
                # --------------------------------------------

                if haku_future is not None:

                    haku_result, haku_dt = haku_future.result()
                    total_haku_time += haku_dt
                    n_haku_calls += 1

                    if haku_result["found"]:

                        refined = haku_result
                        bx, by = refined["X_cm"], refined["Y_cm"]

                        already_tracked = any(
                            math.hypot(
                                bx - s["last_xy"][0], by - s["last_xy"][1]
                            ) < NEW_STONE_DEDUP_CM
                            for s in active_stones
                            if s["confirmed"]
                        )

                        # --------------------------------------------
                        # DIAGNOSTIIKKA (kayttajan raportoima bugi, katso
                        # keskusteluhistoria): HAKU tunnisti PELAAJAN/
                        # LAKAISIJAN kiveksi (pelkkaan muotoon/kokoon
                        # perustuva C++-yhteissovitus ei tunne varia).
                        # Tulostetaan TASSA vain diagnostiikkana (ei viela
                        # hylkaa mitaan) uuden ehdokkaan varipoikkeama
                        # kivivarireferenssiin - kaytetaan naiden lukujen
                        # keraamiseen sopivan hylkayskynnyksen maarittamiseksi
                        # (katso HAKU_COLOR_MAX_DIFF alempana taman
                        # validoinnin jalkeen).
                        # --------------------------------------------

                        if os.environ.get("HAKU_COLOR_DEBUG") and live_state.get("color_reference") is not None:
                            haku_frame_u_f64 = frame_u.astype(np.float64)
                            haku_median_diff = color_match_median_diff(
                                haku_frame_u_f64, local_pts_body, pose,
                                bx, by, live_state["color_reference"],
                                width, height
                            )
                            print(
                                f"[HAKU_COLOR_DEBUG] frame={frame_index} "
                                f"ehdokas ({bx:.1f},{by:.1f}) "
                                f"varidiff={haku_median_diff}"
                            )

                        if not already_tracked:

                            stone_id = next_stone_id
                            next_stone_id += 1

                            active_stones.append({
                                "stone_id": stone_id,
                                "last_xy": (
                                    refined["X_cm"], refined["Y_cm"]
                                ),
                                "min_y_seen": refined["Y_cm"],
                                "misses": 0,
                                "color_diff_history": [],
                                "position_history": [(
                                    frame_index,
                                    refined["X_cm"], refined["Y_cm"]
                                )],
                                # katso ASETUKSET-kommentti MIN_CONFIRMED_
                                # THROW_DISPLACEMENT_CM:in kohdalla - ei
                                # kirjoiteta CSV:hen ennen kuin liike on
                                # vahvistettu (torjuu paikallaan-jo-olevien
                                # kohteiden toistuvan uudelleenrekisterointi-
                                # ongelman).
                                "y0_first_cm": refined["Y_cm"],
                                "confirmed": False,
                                "pending_rows": [
                                    (frame_index, timestamp, dict(refined))
                                ],
                                "confirmed_tarkka_window": deque(
                                    maxlen=MIN_CONFIRMED_TARKKA_OBSERVATIONS
                                ),
                            })

                            debug_draw_items.append((
                                refined["X_cm"], refined["Y_cm"],
                                stone_id, (0, 255, 255),
                                f"{stone_id} UUSI"
                            ))

                            print(
                                f"[frame {frame_index}] Uusi kivi-ehdokas "
                                f"{stone_id}: "
                                f"({refined['X_cm']:.1f}, "
                                f"{refined['Y_cm']:.1f}) cm "
                                "(odottaa liikevahvistusta ennen CSV-kirjausta)"
                            )

                # --------------------------------------------
                # DEBUG_SAVE_TRACKING_VIDEO: piirretaan taman framen
                # kaikkien HAKU/SEURANTA-havaintojen (debug_draw_items,
                # katso yllapuoliset kohdat) ENNUSTETUT ääriviivat
                # (predicted_stone_hull_fast - sama profiilimalli
                # jota itse yhteissovituskin kayttaa) frame_u:n paalle
                # ja kirjoitetaan debug-videoon. Katso taman tiedoston
                # alkupaan DEBUG_SAVE_TRACKING_VIDEO-kommentti varien
                # merkityksesta.
                # --------------------------------------------

                if debug_video_writer is not None:

                    # Kayttajan pyynnosta: debug-videoon tallennetaan
                    # moodikuvalla suodatettu frame_u_for_tracking (se
                    # mita HAKU/SEURANTA oikeasti NAKEE), EI alkuperaista
                    # frame_u:ta - nain debug-videosta voi suoraan
                    # tarkistaa mita tunnistus itse asiassa kaytti.
                    debug_frame = frame_u_for_tracking.copy()

                    for bx, by, s_id, color, label in debug_draw_items:

                        hull = predicted_stone_hull_fast(
                            local_pts_body, pose, bx, by
                        )

                        if hull is None:
                            continue

                        hull_i = hull.astype(np.int32)

                        cv2.polylines(
                            debug_frame, [hull_i], True, color, 2
                        )

                        cv2.putText(
                            debug_frame, label,
                            (
                                int(hull_i[:, 0, 0].min()),
                                int(hull_i[:, 0, 1].min()) - 8
                            ),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
                        )

                    cv2.putText(
                        debug_frame,
                        f"frame {frame_index}  t={timestamp:.2f}s  "
                        f"kivia={len(active_stones)}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 255, 255), 2
                    )

                    debug_video_writer.write(debug_frame)

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
                print(
                    f"  kivenseuranta (C++): "
                    f"HAKU {n_haku_calls} kutsua, "
                    f"ka {(total_haku_time / max(1, n_haku_calls)) * 1000:.1f} ms/kutsu, "
                    f"{(total_haku_time / processed) * 1000:.2f} ms/ruutu ka | "
                    f"SEURANTA {n_seuranta_calls} kutsua, "
                    f"ka {(total_seuranta_time / max(1, n_seuranta_calls)) * 1000:.1f} ms/kutsu "
                    f"({n_seuranta_stone_updates / max(1, n_seuranta_calls):.1f} kivea/kutsu ka), "
                    f"{(total_seuranta_time / processed) * 1000:.2f} ms/ruutu ka"
                )
                print(
                    f"  muu (ei viela optimoitu): "
                    f"read(video) {(total_read_time / processed) * 1000:.2f} ms/ruutu | "
                    f"stabilointimatriisi {(total_stabilize_compute_time / processed) * 1000:.2f} ms/ruutu | "
                    f"warpAffine+remap(koko frame) {(total_warp_remap_time / processed) * 1000:.2f} ms/ruutu | "
                    f"valotasapaino {(total_photometric_time / processed) * 1000:.2f} ms/ruutu | "
                    f"varjosuodatus {(total_shadow_suppress_time / processed) * 1000:.2f} ms/ruutu"
                )

    finally:

        executor.shutdown(
            wait=True
        )

        haku_executor.shutdown(
            wait=True
        )

        if csv_file is not None:
            csv_file.close()

        if debug_video_writer is not None:
            debug_video_writer.release()

    print()

    print(
        f"Videon lapikaynti valmis "
        f"({frame_index}/{total_frames} ruutua)."
    )

    # --------------------------------------------------------
    # NOPEUSSEURANNAN LOPPUYHTEENVETO - kopioi/liita tama takaisin
    # jos haluat kertoa miten kivenseuranta (HAKU+SEURANTA, molemmat
    # C++:aa) kayttaytyy omalla koneellasi oikealla datalla.
    # --------------------------------------------------------

    total_elapsed = time.time() - start_time
    processed_frames = max(1, frame_index)

    print()
    print("=== NOPEUSSEURANTA (kivenseuranta C++: HAKU+SEURANTA) ===")
    print(f"Koko ajo: {total_elapsed:.1f}s / {frame_index} ruutua "
          f"({processed_frames / total_elapsed:.2f} r/s keskimaarin)")
    print(f"HAKU: {n_haku_calls} kutsua, yhteensa {total_haku_time:.2f}s, "
          f"ka {(total_haku_time / max(1, n_haku_calls)) * 1000:.2f} ms/kutsu, "
          f"{(total_haku_time / processed_frames) * 1000:.2f} ms/ruutu "
          "(koko videon yli keskiarvoistettuna)")
    print(f"SEURANTA: {n_seuranta_calls} kutsua, yhteensa "
          f"{total_seuranta_time:.2f}s, "
          f"ka {(total_seuranta_time / max(1, n_seuranta_calls)) * 1000:.2f} ms/kutsu "
          f"({n_seuranta_stone_updates / max(1, n_seuranta_calls):.2f} kivea/kutsu "
          "keskimaarin), "
          f"{(total_seuranta_time / processed_frames) * 1000:.2f} ms/ruutu "
          "(koko videon yli keskiarvoistettuna)")
    print(f"read(video): {total_read_time:.2f}s yhteensa, "
          f"{(total_read_time / processed_frames) * 1000:.2f} ms/ruutu")
    print(f"stabilointimatriisin laskenta (paneili-RANSAC moodikuvavaiheessa, "
          f"suora vaihekorrelaatio moodikuvaan loppuvideolla): "
          f"{total_stabilize_compute_time:.2f}s yhteensa, "
          f"{(total_stabilize_compute_time / processed_frames) * 1000:.2f} ms/ruutu")
    print(f"warpAffine+remap (KOKO frame, joka elavan seurannan ruutu): "
          f"{total_warp_remap_time:.2f}s yhteensa, "
          f"{(total_warp_remap_time / processed_frames) * 1000:.2f} ms/ruutu")
    print(f"valotasapaino/kirkkaus-korjaus (kerran sekunnissa): "
          f"{total_photometric_time:.2f}s yhteensa, "
          f"{(total_photometric_time / processed_frames) * 1000:.2f} ms/ruutu")
    print(f"varjonsietoinen taustanvaimennus (ENABLE_SHADOW_TOLERANT_STABILIZATION): "
          f"{total_shadow_suppress_time:.2f}s yhteensa, "
          f"{(total_shadow_suppress_time / processed_frames) * 1000:.2f} ms/ruutu")
    muu_yhteensa = (
        total_read_time + total_stabilize_compute_time
        + total_warp_remap_time + total_gray_time
        + total_tracking_time + total_transform_time
        + total_photometric_time + total_shadow_suppress_time
    )
    print(f"Yhteensa HAKU+SEURANTA+muu mitattu: "
          f"{((total_haku_time + total_seuranta_time + muu_yhteensa) / processed_frames) * 1000:.2f} "
          "ms/ruutu keskimaarin (25fps-reaaliaikatavoite = 40.0 ms/ruutu, "
          "tiukempi tavoite hyvalla marginaalilla = 20.0 ms/ruutu)")
    print("===========================================================")

    if calib_result is None:
        raise RuntimeError(
            "Kalibrointi ei onnistunut - video loppui kesken "
            "moodinaytteiden keruun (video liian lyhyt?)."
        )

    if profile_result is None and best_sufficient_profile is not None:
        print(
            f"VAROITUS: alle {PROFILE_MIN_ACCEPTED_STONES} kivea "
            f"loytyi ({n_accepted_stones} kpl) ennen videon loppua - "
            "kaytetaan viimeisinta riittavaa profiilia silti."
        )
        profile_result = best_sufficient_profile

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
        "csv_output": csv_output if live_state is not None else None,
        "n_stones_seen": n_confirmed_stones,
        "debug_video_output": (
            debug_video_output if debug_video_writer is not None else None
        ),
    }
