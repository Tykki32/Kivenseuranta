"""Kuvan stabilointi ja valon/varjon tasaus: taustan vaimennus, fotometrinen korjaus, vaihekorrelaatio."""

import cv2
import numpy as np
from config import (
    ENABLE_SHADOW_TOLERANT_STABILIZATION,
    ICE_S_MAX,
    ICE_V_MIN,
    SHADOW_V_DROP_MAX,
    SHADOW_V_DROP_MIN,
    SUBPIXEL_ALIGN_CROP_FRACTION,
    SUBPIXEL_ALIGN_RANGE_PX,
)


# ============================================================
# KOKO RADAN KIVIEHDOKKAIDEN HAKU + LIIKKEEN TUNNISTUS
#
# _scan_stone_candidates: sama segmentointipohjainen tunnistus kuin
# track_stone_in_video/track_stone_in_video_fast kayttavat SISAISESTI
# (_candidates_in_frame) - EI mallipohjaista ristikkohakua, koska
# tassa vaiheessa 3D-profiilia (jota malli tarvitsisi) EI VIELA OLE -
# se on juuri se mita etsitaan.
#
# find_moving_candidate: vertaa KAHDEN PERAKKAISEN skannauksen (n.
# STONE_SCAN_INTERVAL_SECONDS valein) kandidaattilistoja - sama fyysinen
# kivi (lahin osuma) jonka sijainti on muuttunut enemman kuin STONE_
# MOTION_THRESHOLD_CM tulkitaan AIDOSTI LIIKKUVAKSI (ei jo-paikallaan-
# olevaksi) - tama siirtymä-havainto ANTAA SIEMENEN (frame_idx, X, Y)
# track_stone_in_video_fast:lle, joka sitten seuraa koko liu'un.
#
# suppress_static_background: create_granite_mask:in kommentti
# (kamera9_01.py) sanoo 5x5-avauksen poistavan OHUET staattiset
# rakenteet (sponsoritekstin kirjaimet, maalatut viivat) - mutta
# testissa havaittiin etta TAMA EI RIITTANYT: staattinen mainosteksti
# tuli silti virheellisesti tunnistetuksi "kiveksi" (n. 4.5x liian
# suuri R_max 3D-profiilin sovituksessa, katso git-historia/keskustelu).
# Koska meilla ON jo kalibroinnin moodikuva (puhdas, kivi-/pelaaja-
# vapaa staattinen tausta - juuri se mita moodisuodatus on suunniteltu
# tuottamaan), kayttajan ehdotuksesta: verrataan nykyista framea
# TAHAN referenssiin ja PEITETAAN (korvataan valkoisella) alueet jotka
# vastaavat sita - jaljelle jaa vain AIDOSTI poikkeava/vaihtuva sisalto
# (kivet, pelaajat), joten mikaan staattinen painettu sisalto ei voi
# enaa tulla virhetunnistetuksi kiveksi tassa vaiheessa.
# ============================================================
def _shadow_tolerant_background_mask(frame_bgr, reference_bgr, diff_threshold,
                                      shadow_v_drop_max, shadow_v_drop_min=0.0):
    """ENABLE_SHADOW_TOLERANT_STABILIZATION:in ydinsaanto (katso sen
    kommentti): tausta = (tavallinen pieni erotus) TAI (saturaatio
    lahes sama mutta V-pudotus valilla (shadow_v_drop_min,
    shadow_v_drop_max) - eli varjo, ei aito objekti) TAI (absoluuttinen
    jaa-suodatus, katso ICE_S_MAX/ICE_V_MIN:in kommentti - positio-
    riippumaton, toimii vaikka pikseli poikkeaisi referenssista).
    shadow_v_drop_min voi olla negatiivinen (sallii pienen V:n NOUSUN
    silti taustaksi - kayttajan kanssa kasin testattu, katso
    SHADOW_V_DROP_MIN:in kommentti)."""

    diff = cv2.absdiff(frame_bgr, reference_bgr)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    background_mask = diff_gray < diff_threshold

    frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV).astype(np.int16)
    ref_hsv = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2HSV).astype(np.int16)
    v_drop = ref_hsv[..., 2] - frame_hsv[..., 2]
    shadow_mask = (v_drop > shadow_v_drop_min) & (v_drop < shadow_v_drop_max)

    ice_mask = (
        (frame_hsv[..., 1] < ICE_S_MAX) & (frame_hsv[..., 2] > ICE_V_MIN)
    )

    return background_mask | shadow_mask | ice_mask


def suppress_static_background(frame_bgr, reference_bgr, diff_threshold=30):

    if reference_bgr is None or frame_bgr.shape != reference_bgr.shape:
        return frame_bgr

    if ENABLE_SHADOW_TOLERANT_STABILIZATION:
        background_mask = _shadow_tolerant_background_mask(
            frame_bgr, reference_bgr, diff_threshold, SHADOW_V_DROP_MAX,
            SHADOW_V_DROP_MIN
        )
    else:
        diff = cv2.absdiff(frame_bgr, reference_bgr)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        background_mask = diff_gray < diff_threshold

    out = frame_bgr.copy()
    out[background_mask] = (255, 255, 255)

    return out


# ============================================================
# VALOTASAPAINO/KIRKKAUS-KORJAUS "LOPPUVIDEOLLE" (kayttajan pyynnosta,
# katso keskusteluhistoria): pikselitarkka geometrinen stabilointi (katso
# PIKSELITARKKA SUORA STABILOINTI) EI korjaa valon/kameran auto-
# valotuksen/valkotasapainon HIDASTA ajautumista pitkan (>10 min) videon
# aikana - kayttaja huomasi etta debug-videossa jaa/katto/seinat eivat
# olleet TAYSIN valkoisia myohemmin videolla, vaikka geometrinen
# kohdistus oli kunnossa (vahvistettu diff-heatmapilla: HAJA, ei terava,
# ero - viittaa valotasapainoon/kirkkauteen, ei sijaintivirheeseen).
#
# ENSIMMAINEN yritys (histogrammin persentiilikohdistus koko framelle)
# HUONONSI tulosta kaikilla testatuilla frameilla (+7% - +107% vuotoa) -
# jaan lahes saturoitunut valkoisuus + etualan (pelaajien) poikkeavat
# varit vaaristivat globaalin persentiilipohjaisen arvion.
#
# TOIMIVA ratkaisu (kayttajan ehdotuksesta): kayttaa SAMAA periaatetta
# kuin stabiloinnin siirtymahaku - EI kiintea/analyyttinen kaava koko
# kuvalle, vaan ROBUSTI ITEROITU pienimman neliosumman sovitus (per
# BGR-kanava): sovitetaan lineaarinen gain+bias frame->referenssi,
# poistetaan sovituksen JALKEEN suurimman jaannoksen pikselit (etuala:
# pelaajat, kivet - niiden varit eivat mitenkaan liity valotasapainoon)
# ja toistetaan - lahentyy nopeasti (3 kierrosta riittaa) taustan
# TODELLISEEN foto­metriseen suhteeseen. Validoitu: n. sama parannus
# kuin raaka ristikkohaku fg-pikselien maaralla mitattuna (esim. 194114
# -> 92657 vs. haun 90738), mutta ~15x nopeampi (~250-750ms/kutsu vs.
# 5-7s/kutsu) - silti liian hidas JOKA framelle (25fps-budjetti 40ms),
# joten kaytetaan vain KERRAN SEKUNNISSA (katso kaytto run_pipeline:ssa)
# - valotasapainon ajautuminen on hidasta, ei tarvitse paivittaa joka
# framella kuten geometrinen siirtyma.
# ============================================================
def _robust_gain_bias_single_channel(frame_values, reference_values, n_iter=3):
    """Palauttaa (gain,bias) joka minimoi (gain*frame_values+bias -
    reference_values)**2:n, ROBUSTISTI - jokaisen kierroksen jalkeen
    suurimman jaannoksen (80. persentiili ylittavat, tyypillisesti
    etualan/pelaajien/kivien pikselit) pikselit poistetaan seuraavasta
    kierroksesta."""

    mask = np.ones(frame_values.shape, dtype=bool)
    gain, bias = 1.0, 0.0

    for _ in range(n_iter):

        x = frame_values[mask]
        y = reference_values[mask]

        mean_x = x.mean()
        mean_y = y.mean()
        centered_x = x - mean_x

        denom = np.dot(centered_x, centered_x)

        if denom > 1e-6:
            gain = float(np.dot(centered_x, y - mean_y) / denom)
        else:
            gain = 1.0

        bias = float(mean_y - gain * mean_x)

        residual = np.abs(
            reference_values - (gain * frame_values + bias)
        )
        threshold = np.percentile(residual, 80)
        mask = residual < threshold

    return gain, bias


def estimate_photometric_correction(frame_bgr, reference_bgr):
    """Ajaa _robust_gain_bias_single_channel:in erikseen jokaiselle BGR-
    kanavalle - palauttaa (gains,biases), molemmat 3-alkioisia listoja.
    Kanavakohtaisuus kattaa seka yleisen KIRKKAUDEN (kaikki kanavat
    samansuuntaisesti) etta VALKOTASAPAINON/varisavyn ajautumisen (kanavat
    eri suuntiin) yhdella samalla mekanismilla."""

    gains = []
    biases = []

    for channel in range(3):

        gain, bias = _robust_gain_bias_single_channel(
            frame_bgr[:, :, channel].astype(np.float64).ravel(),
            reference_bgr[:, :, channel].astype(np.float64).ravel()
        )

        gains.append(gain)
        biases.append(bias)

    return gains, biases


def apply_photometric_correction(frame_bgr, gains, biases):

    out = frame_bgr.astype(np.float32).copy()

    for channel in range(3):
        out[:, :, channel] = (
            out[:, :, channel] * gains[channel] + biases[channel]
        )

    return np.clip(out, 0, 255).astype(np.uint8)


_hann_window_cache = {}


def _phase_correlate_full_frame(gray_a_full, gray_b_full):
    """Yhteinen vaihekorrelaatio-ydin (kayttaa SUBPIXEL_ALIGN_CROP_
    FRACTION-rajausta + valimuistitettua Hanning-ikkunaa) - kaytetaan
    "loppuvideon" (profiiliskannaus + elava seuranta) PIKSELITARKASSA
    SUORASSA STABILOINNISSA (katso sen kommentti run_pipeline:ssa):
    jokainen frame verrataan suoraan moodikuvareferenssiin taman
    funktion kautta.
    gray_a_full/gray_b_full: TAYSRESOLUUTIOISET harmaasavykuvat (uint8
    tai float32). Palauttaa (dx,dy): siirto joka pitaa lisata jotta
    gray_b linjautuisi gray_a:n kanssa, SUBPIXEL_ALIGN_RANGE_PX:aan
    rajattuna."""

    h, w = gray_a_full.shape[:2]
    cw = int(round(w * SUBPIXEL_ALIGN_CROP_FRACTION))
    ch = int(round(h * SUBPIXEL_ALIGN_CROP_FRACTION))
    x0 = (w - cw) // 2
    y0 = (h - ch) // 2

    key = (x0, y0, cw, ch)
    hann = _hann_window_cache.get(key)
    if hann is None:
        hann = cv2.createHanningWindow((cw, ch), cv2.CV_32F)
        _hann_window_cache[key] = hann

    a_crop = gray_a_full[y0:y0 + ch, x0:x0 + cw].astype(np.float32)
    b_crop = gray_b_full[y0:y0 + ch, x0:x0 + cw].astype(np.float32)

    (dx, dy), _response = cv2.phaseCorrelate(a_crop, b_crop, hann)

    dx = float(np.clip(dx, -SUBPIXEL_ALIGN_RANGE_PX, SUBPIXEL_ALIGN_RANGE_PX))
    dy = float(np.clip(dy, -SUBPIXEL_ALIGN_RANGE_PX, SUBPIXEL_ALIGN_RANGE_PX))
    return dx, dy
