# ============================================================
# review_tracks.py - Testi_01_01/tools
#
# TARKASTELUMALLI (kayttajan pyynnosta): piirtaa jokaisen taydellisen
# (hog-to-hog) kiviheiton reitin OMAAN paneeliinsa koko radan
# topdown-kuvan paalle, kaikki paneelit yhdessa kuvassa samassa
# koossa - sama formaatti kuin aiemmin hyvaksytty combined_grid_v3.
# Aja tama JOKAISEN run_pipeline-ajon jalkeen CSV:n paalle nahdaksesi
# nopeasti onnistuiko ajo (loytyivatko/seurattiinko heitot koko
# matkalta jarkevasti, vai onko trackereissa ajautumista/katkoksia).
#
# EI KOSKETA main.py:ta tai kamera8_01/9_0X.py:ta - tama lukee vain
# niiden jo tuottaman CSV:n ja kalibroinnin topdown-tarkistuskuvan.
#
# Kaytto:
#   python3 review_tracks.py <csv_path> <topdown_png_path> <output_png_path>
# ============================================================

import sys
import csv
import math
from collections import defaultdict

import numpy as np
import cv2

# Samat vakiot kuin kamera8_01.py:ssa (ei tuoda moduulina, jotta
# tama tyokalu ei riipu tkinter-stubista tms. - vain kopioitu tahan
# vertailua/piirtoa varten, EI muuta niiden omaa maaritelmaa).
OUTPUT_X_MIN_CM = -200.0
OUTPUT_X_MAX_CM = 200.0
OUTPUT_Y_MIN_CM = 0.0
OUTPUT_Y_MAX_CM = 4000.0
PIXELS_PER_CM = 2.0

NEAR_HOGLINE_Y_CM = 822.9
FAR_HOGLINE_Y_CM = 3017.6

# --------------------------------------------------------------
# "TAYDELLINEN HEITTO" -kynnysarvo: reitin on ylitettava molemmat
# hoglinet (pienella marginaalilla) jotta se lasketaan naytettavaksi
# - tama on ainoa jarkeva "onnistuiko ajo" -mittari, koska kiven
# lopullinen levahdyspaikka EI kerro mitaan seurannan onnistumisesta
# (kivet pysahtyvat usein samaan kohtaan riippumatta siita seurattiinko
# niita oikein koko matkalta - katso taman projektin oma, kayttajan
# korjaama virhe aiemmin: paikkaan EI SAA perustaa "sama kivi"-paatosta).
# --------------------------------------------------------------
HOGLINE_MARGIN_CM = 100.0

# --------------------------------------------------------------
# PATKIEN YHDISTAMINEN (sama stone_id-katkeaa-ja-loytyy-uudelleen
# -periaate kuin aiemmin hyvaksytyssa versiossa): kaksi PERAKKAISTA
# (ajassa) stone_id-patkaa yhdistetaan SAMAKSI fyysiseksi kiveksi
# VAIN JOS seka aikavali etta paikkahyppy ovat pienia - EI KOSKAAN
# pelkan lopullisen sijainnin perusteella (katso ylla).
# --------------------------------------------------------------
MERGE_MAX_GAP_FRAMES = 75      # 3s 25fps:lla
MERGE_MAX_JUMP_CM = 50.0


def load_segments(csv_path):
    rows_by_id = defaultdict(list)
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = int(row["stone_id"])
            rows_by_id[sid].append((
                int(row["frame"]), float(row["timestamp_s"]),
                float(row["x_m"]) * 100.0, float(row["y_m"]) * 100.0
            ))

    segments = []
    for sid, pts in rows_by_id.items():
        pts.sort(key=lambda p: p[0])
        segments.append({"ids": [sid], "pts": pts})

    segments.sort(key=lambda s: s["pts"][0][0])
    return segments


def merge_segments(segments):
    """Ahne, ajassa etenevä yhdistys: jokainen patka liitetaan
    LAHIMPAAN (pienin aikavali) avoimeen ketjuun joka tayttaa seka
    aikavali- etta paikkahyppy-ehdon, muuten se aloittaa uuden
    ketjun. Katso tiedoston alun kommentti perusteluista."""

    open_chains = []  # list of dicts: {"ids":[...], "pts":[...]}

    for seg in segments:
        f0, t0, x0, y0 = seg["pts"][0]

        best_chain = None
        best_gap = None

        for chain in open_chains:
            fl, tl, xl, yl = chain["pts"][-1]
            gap = f0 - fl
            if gap <= 0 or gap > MERGE_MAX_GAP_FRAMES:
                continue
            jump = math.hypot(x0 - xl, y0 - yl)
            if jump > MERGE_MAX_JUMP_CM:
                continue
            if best_gap is None or gap < best_gap:
                best_gap = gap
                best_chain = chain

        if best_chain is not None:
            best_chain["ids"].extend(seg["ids"])
            best_chain["pts"].extend(seg["pts"])
        else:
            open_chains.append({"ids": list(seg["ids"]), "pts": list(seg["pts"])})

    return open_chains


def to_output_px(x_cm, y_cm):
    px = (x_cm - OUTPUT_X_MIN_CM) * PIXELS_PER_CM
    py = (OUTPUT_Y_MAX_CM - y_cm) * PIXELS_PER_CM
    return px, py


def lerp_color_bgr(t):
    # t=0 -> sininen (alku), t=1 -> punainen (loppu), lineaarinen
    # RGB-interpolointi (kulkee violetin/magentan kautta, EI vihrean/
    # keltaisen - sama visuaalinen tyyli kuin aiemmin hyvaksytyssa
    # versiossa).
    t = max(0.0, min(1.0, t))
    b = int(round(255 * (1.0 - t)))
    r = int(round(255 * t))
    return (b, 0, r)


def render_panel(topdown_img, chain, panel_w, panel_h):
    canvas = topdown_img.copy()
    pts = chain["pts"]
    n = len(pts)

    for i, (frame, ts, x_cm, y_cm) in enumerate(pts):
        px, py = to_output_px(x_cm, y_cm)
        px, py = int(round(px)), int(round(py))
        if 0 <= px < canvas.shape[1] and 0 <= py < canvas.shape[0]:
            color = lerp_color_bgr(i / max(1, n - 1))
            cv2.circle(canvas, (px, py), 4, color, -1, lineType=cv2.LINE_AA)

    panel = cv2.resize(canvas, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
    return panel


def label_letter(i):
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    out = ""
    i += 1
    while i > 0:
        i -= 1
        out = letters[i % 26] + out
        i //= 26
    return out


def build_grid(topdown_path, chains, output_path, n_cols=4,
                panel_w=300, panel_h=1150, label_h=26):
    # panel_h/panel_w EI sailyta lahdekuvan (800x8000, suhde 1:10)
    # omaa kuvasuhdetta tahallaan - taman kuvan tarkoitus on NOPEA
    # silmamaarainen tarkastus (ajautuiko seuranta pois radalta, jai
    # kesken, jne.), ei senttimetrien tarkka lukeminen, joten hieman
    # "litistetty" (~3.8:1) paneeli pysyy luettavana yhdessa kuvassa
    # vaikka heittoja olisi kymmenia.

    topdown_img = cv2.imread(topdown_path)
    if topdown_img is None:
        raise RuntimeError(f"Ei voitu lukea topdown-kuvaa: {topdown_path}")

    n = len(chains)
    if n == 0:
        raise RuntimeError("Ei taysia (hog-to-hog) kivenreitteja naytettavaksi.")

    n_rows = (n + n_cols - 1) // n_cols

    cell_w = panel_w
    cell_h = panel_h + label_h

    grid = np.full((n_rows * cell_h, n_cols * cell_w, 3), 230, dtype=np.uint8)

    for idx, chain in enumerate(chains):
        row = idx // n_cols
        col = idx % n_cols

        panel = render_panel(topdown_img, chain, panel_w, panel_h)

        y0 = row * cell_h + label_h
        x0 = col * cell_w
        grid[y0:y0 + panel_h, x0:x0 + panel_w] = panel

        pts = chain["pts"]
        t0, t1 = pts[0][1], pts[-1][1]
        ids_str = "+".join(str(i) for i in chain["ids"])
        letter = label_letter(idx)
        label = f"{letter} (id{ids_str}) {t0:.0f}-{t1:.0f}s n={len(pts)}"

        cv2.putText(
            grid, label, (x0 + 4, row * cell_h + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA
        )

    cv2.imwrite(output_path, grid)
    return output_path


def main():
    if len(sys.argv) != 4:
        print("Kaytto: python3 review_tracks.py <csv_path> <topdown_png_path> <output_png_path>")
        sys.exit(1)

    csv_path, topdown_path, output_path = sys.argv[1:4]

    segments = load_segments(csv_path)
    chains = merge_segments(segments)

    full_chains = []
    for chain in chains:
        ys = [p[3] for p in chain["pts"]]
        y_min, y_max = min(ys), max(ys)
        if (y_min <= NEAR_HOGLINE_Y_CM + HOGLINE_MARGIN_CM and
                y_max >= FAR_HOGLINE_Y_CM - HOGLINE_MARGIN_CM):
            full_chains.append(chain)

    full_chains.sort(key=lambda c: c["pts"][0][0])

    print(f"Loytyi {len(chains)} liikeketjua (yhdistettyna), "
          f"joista {len(full_chains)} ylittaa molemmat hoglinet "
          f"(taysi heitto).")

    build_grid(topdown_path, full_chains, output_path)
    print(f"Tallennettu: {output_path}")


if __name__ == "__main__":
    main()
