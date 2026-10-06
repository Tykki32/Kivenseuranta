"""Heitot: hog-hog-analyysin kaynnistys radalle, heittoportti (mitka radat ovat heittoja) ja tulos-CSV:t.

Tulostiedostot (<pohja> = videon nimi ilman paatetta, tai live-nimi):
  <pohja>_kivien_sijainnit_raaka.csv   kaikki vahvistetut radat ruutu ruudulta (kirjoitetaan seurannan aikana)
  <pohja>_kivien_sijainnit.csv         vain heittoportin lapaisseet radat (yksi rata / kaukohogin ylitys)
  <pohja>_kivien_sijainnit_hog.csv     hog-hog-analyysi jokaisesta onnistuneesta heitosta
  <pohja>_kivien_sijainnit_kahva.csv   kahvan vari ja nakyvyys ruuduittain (kierre.py)
  <pohja>_kivien_sijainnit_hog_kivi<N>.png  still-kuva hog-tekstilla

HEITTOPORTTI: rata on heitto jos hog-hog-analyysi onnistui sille (aina), tai jos sen liike on heittomainen (matka
eteenpain >= A.PORTTI_MATKA_CM, loppu-Y <= A.PORTTI_LOPPU_Y_MAX_CM, hidastuu, >= A.PORTTI_MIN_RIVIT rivia) ja
sovitus riittava (rms-mediaani, tarkka-osuus). Kahden radan kaukohogin ylitys < A.PORTTI_TUPLA_RUUDUT toisistaan =
sama heitto (vain yksi kivi ylittaa hoglinjan kerrallaan) -> parempi sailyy.
"""
import csv
import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

import asetukset as A
import hog_analyysi
import kierre
import live
import rata

CSV_HEADER = ["frame", "timestamp_s", "stone_id", "x_m", "y_m", "tarkka", "n_runkopistetta", "n_reunapistetta", "rms_px",
              "rengas_r_cm"]

HOG_CSV_SARAKKEET = [
    "stone_id", "frame", "R", "R_ennen_suodatusta", "n_kaytetty", "n_pudotettu", "v_far_hog_ms", "decel_ms2", "hog_hog_s",
    "t_far_hog_s", "t_near_hog_s", "v_near_hog_ms", "a", "b", "c", "R_x", "R_x_ennen_suodatusta", "R_tulo", "x_far_hog_cm",
    "dir_far_hog_deg", "slope_dxdy", "x_straight_at_tee_cm", "px", "qx", "rx", "liuku_x_tee_cm", "liuku_dir_deg", "liuku_n",
    "liuku_rms_cm", "decel_keskim_ms2", "mu_a", "mu_b", "kitka_rms_cm", "kurvi_k_ms2", "kurvi_rms_cm", "x_far_hog_cm_2aste",
    "dir_far_hog_deg_2aste", "kahva_h", "kahva_vari", "liuku_x_tee_cm_hakki_p", "liuku_x_tee_cm_hakki_m",
    "liuku_dir_deg_hakki_p", "liuku_dir_deg_hakki_m", "kierteet", "kierrosaika_far_s", "kierrosaika_near_s", "kierre_r2",
    "kierteet_taysi", "kierrosaika_far_s_taysi", "kierrosaika_near_s_taysi", "kierre_r2_taysi", "kierre_ruutuja_taysi",
]
# --full: pysahtymispaikka. x/y nollakohdasta (A.PYSAHDYS_NOLLA_*; oletus keskiviiva ja lahemman pesan T-viiva),
# X/Y fyysisessa koordinaatistossa (rata.py). x: + = kohti +X (fyysinen suunta, ei debug-ikkunan puoli).
HOG_CSV_PYSAHDYS = ["pysahtyi_x_cm", "pysahtyi_y_cm", "pysahtyi_X_cm", "pysahtyi_Y_cm"]


def write_stone_csv_row(writer, frame_index, timestamp, stone_id, refined):
    writer.writerow([
        frame_index, f"{timestamp:.3f}", stone_id, f"{refined['X_cm'] / 100.0:.5f}", f"{refined['Y_cm'] / 100.0:.5f}",
        int(refined["tarkka"]), refined["n_body"], refined["n_ring"],
        f"{refined['rms_px']:.3f}" if refined["rms_px"] is not None else "",
        f"{refined['ring_radius_cm']:.3f}" if refined["ring_radius_cm"] is not None else "",
    ])


# =====================================================================================================================
# HOG-HOG -ANALYYSI RADALLE
# =====================================================================================================================

def _kierre_tulos(res, obs, fps, suffix, label):
    try:
        kr = kierre.kierre_arvio(obs, fps, float(res["t_far_hog_s"]), float(res["t_near_hog_s"]))
    except Exception as e_:
        kr = None
        print(f"[kierre] {label}arvio epaonnistui: {e_!r}")
    if kr is not None:
        res["kierteet" + suffix] = kr["kierrokset"]
        res["kierrosaika_far_s" + suffix] = kr["kierrosaika_far_s"]
        res["kierrosaika_near_s" + suffix] = kr["kierrosaika_near_s"]
        res["kierre_r2" + suffix] = kr["r2"]


_KIERRE_POOL = None          # live: kierrearvio taustasaikeessa (ei pysayta seurantaa lahihogilla)
_KIERRE_KESKEN = []


def _kierteet(res, obs, obs_taysi, fps):
    """Kierrearvio 720p- ja taysresoluutioisesta piirteesta -> res."""
    _kierre_tulos(res, obs, fps, "", "")
    if obs_taysi:
        _kierre_tulos(res, obs_taysi, fps, "_taysi", "taysi resoluutio: ")
        res["kierre_ruutuja_taysi"] = len(obs_taysi)


def _kierteet_tausta(res, obs, obs_taysi, fps):
    _kierteet(res, obs, obs_taysi, fps)
    k = res.get("kierteet_taysi", res.get("kierteet"))
    print(f"[kierre] kivi {res['stone_id']}: " + (f"kierteita hog-hog {k:.1f}" if k is not None else "kierteita ei saatu"))


def odota_kierteet():
    """Odottaa taustasaikeen kierrearviot (ennen tulostiedostojen kirjoitusta)."""
    global _KIERRE_POOL
    for f in _KIERRE_KESKEN:
        try:
            f.result()
        except Exception as e_:
            print(f"[kierre] taustalaskenta epaonnistui: {e_!r}")
    _KIERRE_KESKEN.clear()
    if _KIERRE_POOL is not None:
        _KIERRE_POOL.shutdown(wait=True)
        _KIERRE_POOL = None


def hog_check(s, frame_index, fps, results, frame_img, csv_output, pose, odota=None):
    """Kun vahvistettu kivi on lahihogin + marginaalin kohdalla, radalle tehdaan hog-hog-analyysi (kerran) ja
    kierrearvio. Onnistunut tulos lisataan results-listaan ja tulostetaan; still-kuva tallennetaan lahihogilla."""
    rows = s.get("all_rows")
    if not rows:
        return
    y = rows[-1][2].get("Y_cm")
    if y is None:
        return
    near_hog = rata.NEAR_HOGLINE_Y_CM
    if "hog_result" not in s and y <= near_hog + A.HOG_LAHI_MARGINAALI_CM:
        res = hog_analyysi.analyze_hog(rows, near_hog, rata.FAR_HOGLINE_Y_CM, tee_cm=rata.NEAR_HOUSE_Y_CM)
        res["stone_id"] = s["stone_id"]
        res["frame"] = frame_index
        if res.get("ok") and live.active() is not None:
            # kaukohogin ylityksen seinakelloaika (paneelin "sekuntia sitten"): nykyisen ruudun kaappausaika -
            # (nykyisen ruudun aika - ylityksen aika) videoajassa
            cap = live.active().store.capture_time(rows[-1][0])
            if cap is not None:
                res["t_far_wall"] = cap - (float(rows[-1][1]) - float(res["t_far_hog_s"]))
        if res.get("ok"):
            if odota is not None:
                odota()            # taustalla lasketut kierrepiirteet valmiiksi
            obs, obs_taysi = s.pop("kierre", None) or [], s.pop("kierre_taysi", None) or []
            if live.active() is not None:
                # live: kierrearvio (~1 s / piirre) taustasaikeeseen; tulos paneeliin kun valmis
                global _KIERRE_POOL
                if _KIERRE_POOL is None:
                    _KIERRE_POOL = ThreadPoolExecutor(max_workers=1)
                _KIERRE_KESKEN.append(_KIERRE_POOL.submit(_kierteet_tausta, res, obs, obs_taysi, fps))
            else:
                _kierteet(res, obs, obs_taysi, fps)
        s["hog_result"] = res
        if res.get("ok"):          # vain onnistunut (R_y, R_x, R_y*R_x > 0.99) tulostetaan
            results.append(res)
            print(f"[frame {frame_index}] " + " | ".join(hog_analyysi.format_lines(res, s["stone_id"])))
    r = s.get("hog_result")
    if r and r.get("ok") and not s.get("hog_overlay_started") and y <= near_hog:
        s["hog_overlay_started"] = True
        if A.HOG_TALLENNA_KUVA and frame_img is not None and csv_output:
            try:
                lines = hog_analyysi.format_lines(r, s["stone_id"])
                snap = hog_analyysi.draw_overlay(frame_img.copy(), lines, pose["K"], pose["R"], pose["t"], near_hog)
                cv2.imwrite(os.path.splitext(csv_output)[0] + f"_hog_kivi{s['stone_id']}.png", snap)
            except Exception as e_:       # still-kuva ei saa kaataa seurantaa
                print(f"[hog] still-kuvan tallennus epaonnistui: {e_}")


def hog_kirjoita_csv(results, csv_output):
    if not results:
        return
    path = os.path.splitext(csv_output)[0] + "_hog.csv"
    sarakkeet = HOG_CSV_SARAKKEET + (HOG_CSV_PYSAHDYS if A.SEURAA_PYSAHTYMISEEN else [])
    with open(path, "w", newline="") as hf:
        w = csv.writer(hf)
        w.writerow(sarakkeet)
        for r in results:
            w.writerow([r.get(c_) for c_ in sarakkeet])
    print(f"Hog-hog -analyysi: {len(results)} heittoa -> {path}")


# =====================================================================================================================
# HEITTOPORTTI
# =====================================================================================================================

def _track_kinematics(rows):
    """(matka, loppu-Y, hidastuvuussuhde, kaukohogin ylitysruutu). Hidastuvuussuhde = loppuvaiheen (viimeiset 20 %)
    nopeus / alkuvaiheen nopeus: aito kivi hidastuu (0,2-0,6), pelaajan paa liikkuu tasaisesti (~1)."""
    ys = np.array([r["Y_cm"] for _, _, r in rows])
    fr = np.array([f for f, _, _ in rows])
    n = len(rows)
    a = max(2, n // 5)
    va = (ys[0] - ys[a]) / max(1, fr[a] - fr[0])
    vb = (ys[-a - 1] - ys[-1]) / max(1, fr[-1] - fr[-a - 1])
    ratio = vb / va if va > 0.5 else 9.9
    cross = None
    hog = rata.FAR_HOGLINE_Y_CM
    for i in range(n - 1):
        if ys[i] > hog >= ys[i + 1]:
            cross = float(fr[i])
            break
    if cross is None and ys[0] <= hog:
        cross = float(fr[0])
    return float(ys[0] - ys.min()), float(ys[-1]), float(ratio), cross


def _trim_track_head(rows):
    """Pudottaa radan alun, jos sen jalkeen on > A.PORTTI_ALKU_AUKKO_RUUDUT rivitön aukko ja alku on lyhyt
    (< A.PORTTI_ALKU_MAX_RIVIT): rata on alussa lukkiutunut vaaraan kohteeseen ja loytanyt kiven vasta aukon jalkeen."""
    for i in range(min(len(rows) - 1, A.PORTTI_ALKU_MAX_RIVIT)):
        if rows[i + 1][0] - rows[i][0] > A.PORTTI_ALKU_AUKKO_RUUDUT:
            return rows[i + 1:]
    return rows


def _estimate_crossing(rows):
    """Kaukohogin ylitysruutu. Jos rata alkaa jo hogin lahipuolelta (rekisteroity myohassa), ylitys ekstrapoloidaan
    radan alun nopeudesta (enintaan A.PORTTI_YLITYS_EKSTRAPOLOINTI_MAX ruutua)."""
    cross = _track_kinematics(rows)[3]
    ys = np.array([r["Y_cm"] for _, _, r in rows])
    fr = np.array([f for f, _, _ in rows], dtype=float)
    hog = rata.FAR_HOGLINE_Y_CM
    if ys[0] <= hog and len(rows) >= 10:
        m = min(len(rows), 30)
        v = (ys[0] - ys[m - 1]) / max(1.0, fr[m - 1] - fr[0])   # cm/ruutu, Y pienenee -> v > 0
        if v > 1.0:
            back = min((hog - ys[0]) / v, A.PORTTI_YLITYS_EKSTRAPOLOINTI_MAX)
            return float(fr[0] - back)
    return cross


def track_throw_class(rows):
    """0 = ei heitto, 2 = heitto hyvalla sovituksella, 1 = heitto heikolla sovituksella (esim. lakaisija peittaa)."""
    if len(rows) < A.PORTTI_MIN_RIVIT:
        return 0
    travel, yend, ratio, _ = _track_kinematics(rows)
    if travel < A.PORTTI_MATKA_CM or yend > A.PORTTI_LOPPU_Y_MAX_CM or ratio > A.PORTTI_MAX_NOPEUSSUHDE:
        return 0
    rms = [r["rms_px"] for _, _, r in rows if r.get("rms_px") is not None]
    if not rms:
        return 0
    med = float(np.median(rms))
    tk = sum(1 for _, _, r in rows if r.get("tarkka")) / len(rows)
    if med <= A.PORTTI_MAX_RMS and tk >= A.PORTTI_MIN_TARKKA:
        return 2
    if med <= A.PORTTI_PELASTUS_MAX_RMS and tk >= A.PORTTI_PELASTUS_MIN_TARKKA:
        return 1
    return 0


def select_throws(tracks, hog_ok_ids=()):
    """tracks: [(stone_id, rows)] -> heittoportin lapaisseet, yksi rata / kaukohogin ylitys (parempi sailyy: hog-ok,
    hyva sovitus, pienin rms, pisin). hog_ok_ids: radat joille hog-hog-analyysi onnistui -> aina heitto."""
    hog_ok_ids = set(hog_ok_ids)
    cands = []
    for sid, rows in tracks:
        rows = _trim_track_head(rows)
        cls = 3 if sid in hog_ok_ids else track_throw_class(rows)
        if cls == 0:
            continue
        rms = [r["rms_px"] for _, _, r in rows if r.get("rms_px") is not None]
        cross = _estimate_crossing(rows)
        cands.append((-cls, float(np.median(rms)) if rms else 1e9, -len(rows), sid, rows, cross))
    cands.sort(key=lambda c: c[:3])
    kept = []
    for c in cands:
        cross = c[5]
        if cross is not None and any(k[5] is not None and abs(k[5] - cross) < A.PORTTI_TUPLA_RUUDUT for k in kept):
            continue
        kept.append(c)
    return sorted(((k[3], k[4]) for k in kept), key=lambda t: t[1][0][0])


def kirjoita_heitot(stone_registry, hog_results, csv_output):
    """Heittoportin lapaisseet radat -> <pohja>_kivien_sijainnit.csv."""
    all_tr = [(sid, st_["all_rows"]) for sid, st_ in stone_registry.items() if st_.get("all_rows")]
    hog_ok = {r["stone_id"] for r in hog_results if r.get("ok")}
    n_before = sum(1 for sid_, rw in all_tr if sid_ in hog_ok or track_throw_class(rw) > 0)
    gated = select_throws(all_tr, hog_ok)
    with open(csv_output, "w", newline="") as gf:
        gw = csv.writer(gf)
        gw.writerow(CSV_HEADER)
        for sid, rows in gated:
            for rf, rt, rr in rows:
                write_stone_csv_row(gw, rf, rt, sid, rr)
    print(f"Heittoportti: {len(stone_registry)} vahvistettua rataa -> {n_before} heittomaista rataa -> {len(gated)} heittoa "
          f"(yksi / hogline-ylitys) ({csv_output})")
