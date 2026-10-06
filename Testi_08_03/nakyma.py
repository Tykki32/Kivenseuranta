"""Nakymat: debug-video ja puhelinnakyma (kaannetty kuva + tulospaneelit), tulospaneelien laatikot ja videon kirjoittajat.

Kuva kaannetaan 90 astetta vastapaivaan (kivet kulkevat ylhaalta alas) ja skaalataan korkeuteen DEBUG_H. Vasemmalla ja
oikealla paneelit heitoista, jotka lahtivat vasemmalle / oikealle (ylimpana uusin, vanhemmat rullaavat alas).
Puhelinnakymassa lisaksi: korostettujen laatikoiden liukusuorat ja irroitusristi liikkuvien kohteiden alla, keskikuva
Alku-sekuntia myohassa (rengaspuskuri) ja tuplafontti.
"""
import queue
import subprocess
import threading
import time

import cv2
import numpy as np

import asetukset as A
import yleiset

DEBUG_H = 1080
PANEL_W = 430
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6          # sama fontti kuin edellisessa (hog-hog -tekstissa)
FONT_THICK = 2
LINE_H = 26
BOX_PAD = 8
BOX_GAP = 8
VIIVA_BGR = {"v": (32, 32, 255), "o": (32, 224, 32)}    # irroitusviivat: vasenkatinen punainen, oikeakatinen vihrea


def debug_layout(src_w, src_h):
    """Kaannetyn videon koko: (video_w, video_h, scale, total_w). Kaannos 90 astetta -> leveys = src_h, korkeus = src_w; skaalataan korkeus DEBUG_H:hon."""
    scale = DEBUG_H / float(src_w)
    video_w = int(round(src_h * scale))
    return video_w, DEBUG_H, scale, video_w + 2 * PANEL_W


def plus_x_is_right(K, R, t, y_cm=1500.0):
    """True jos +X on kaannetyssa (90 astetta vastapaivaan) videossa OIKEALLA. Kaannos: x' = y_alkuperainen."""
    K = np.asarray(K, float); R = np.asarray(R, float).reshape(3, 3); t = np.asarray(t, float).reshape(3)
    def py(x):
        p = K @ (R @ np.array([x, y_cm, 0.0]) + t); return p[1] / p[2]
    return py(100.0) > py(-100.0)


def throw_side(res, plus_right):
    """'L' tai 'R': mille puolelle (kaannetyssa videossa) heitto lahti kaukohoglinella (suunta dir_far_hog_deg: + = kohti +X)."""
    toward_plus = res["dir_far_hog_deg"] > 0
    return "R" if toward_plus == bool(plus_right) else "L"


RIVI_KIERRE = 6           # entry_lines: kierrerivin indeksi (oma vari)


def entry_lines(res, plus_right=None):
    lines = [f"kiven ID: {res['stone_id']}",
             f"nopeus: {res['v_far_hog_ms']:.2f} m/s",
             f"hidastuvuus: {res['decel_ms2']:.3f} m/s^2",
             f"hog-hog: {res['hog_hog_s']:.2f} s",
             f"liuku: {res['liuku_x_tee_cm']:+.0f} cm" if "liuku_x_tee_cm" in res else "liuku: -",
             f"irroitus: {res.get('x_straight_at_tee_cm', float('nan')):+.0f} cm",
             kierre_text(res)[0]]
    if A.SEURAA_PYSAHTYMISEEN:
        lines.append(pysahdys_text(res, plus_right))
    return lines


def pysahdys_text(res, plus_right=None):
    """--full: "pysahtyi: x; y" (cm, nollakohdasta A.PYSAHDYS_NOLLA_*). Debug-ikkunan suunnat: oikea ja ylos (kohti
    kaukaista paata) positiivisia, vasen ja alas negatiivisia."""
    if res.get("pysahtyi_x_cm") is None:
        return "pysahtyi: -"
    x = res["pysahtyi_x_cm"] * (1.0 if plus_right is None or bool(plus_right) else -1.0)
    return f"pysahtyi: {x:.0f}; {res['pysahtyi_y_cm']:.0f}"


def kierre_text(res):
    """(teksti, vari BGR). Taysi resoluutio -> vihrea; muuten 720p-arvo punaisella; ei arvoa -> valkoinen."""
    if res.get("kierteet_taysi") is not None:
        return f"kierteita: {res['kierteet_taysi']:.1f}", (0, 220, 0)
    if res.get("kierteet") is not None:
        return f"kierteita: {res['kierteet']:.1f}", (0, 0, 255)
    return "kierteita: -", (255, 255, 255)


def entry_age_s(res, now_video_s=None):
    """Montako sekuntia sitten kivi ylitti kaukohoglinen. Live: seinakello (t_far_wall) vs nyt;
    tiedosto: videoaika (t_far_hog_s) vs kasiteltava ruutu. None jos ei tiedossa."""
    if res.get("t_far_wall") is not None:
        return max(0, int(time.time() - res["t_far_wall"]))
    if now_video_s is not None and res.get("t_far_hog_s") is not None:
        return max(0, int(now_video_s - res["t_far_hog_s"]))
    return None


def liuku_segments(res, plus_right, nayta=(True, True)):
    """[(teksti, vari)] liuku-riville: vasemman hakin (kaannetyssa videossa) arvo punaisella, oikean vihrealla.
    t12: nayta = (vasenkatinen, oikeakatinen) - kumpi hakin luku naytetaan (puhelimen valintaruudut)."""
    if "liuku_x_tee_cm_hakki_p" not in res:
        return [(entry_lines(res)[4], (255, 255, 255))]
    vas, oik = ("m", "p") if plus_right is None or bool(plus_right) else ("p", "m")
    out = [("liuku:", (255, 255, 255))]
    for tag, col, on in ((vas, (0, 0, 255), nayta[0]), (oik, (0, 220, 0), nayta[1])):
        if on:
            out += [(" ", (255, 255, 255)), (f"{res['liuku_x_tee_cm_hakki_' + tag]:+.0f}", col)]
    if len(out) > 1:
        out.append((" cm", (255, 255, 255)))
    return out


# Paneelin mitat: debug-video ja puhelinnakyma (tuplafontti; rivivali tiiviimpi, jotta 3 laatikkoa mahtuu kummallekin puolelle).
PANEELI_NORMAALI = dict(pw=PANEL_W, lh=LINE_H, bp=BOX_PAD, bg=BOX_GAP, fs=FONT_SCALE, ft=FONT_THICK, ty=20, rt=2)
PANEELI_PUHELIN = dict(pw=570, lh=44, bp=12, bg=10, fs=2 * FONT_SCALE, ft=3, ty=30, rt=3)


def laatikon_mitat(L):
    """(rivivali, laatikon korkeus). --full lisaa rivin; rivivalia tiivistetaan tarvittaessa, jotta 3 laatikkoa mahtuu."""
    rivit = 8 if A.SEURAA_PYSAHTYMISEEN else 7
    lh = min(L["lh"], int((DEBUG_H / 3 - L["bg"] - 2 * L["bp"]) // rivit))
    return lh, lh * rivit + 2 * L["bp"]


def render_panel(entries, ages=None, plus_right=None, nayta=(True, True), L=None):
    """entries: lista dict-tuloksia, UUSIN ENSIMMAISENA. Palauttaa (DEBUG_H x pw) kuvan; jokainen heitto omassa laatikossa.
    ages: sekunnit kaukohoglinen ylityksesta (sama jarjestys), naytetaan kiven ID:n vieressa. L: paneelin mitat (oletus normaali)."""
    L = L or PANEELI_NORMAALI
    pw, lh, bp, bg, fs, ft = L["pw"], L["lh"], L["bp"], L["bg"], L["fs"], L["ft"]
    img = np.zeros((DEBUG_H, pw, 3), np.uint8)
    lh, box_h = laatikon_mitat(L)
    y = bg
    for k, res in enumerate(entries):
        if y + box_h > DEBUG_H:
            break
        cv2.rectangle(img, (bg, y), (pw - bg, y + box_h), (45, 45, 45), -1)
        cv2.rectangle(img, (bg, y), (pw - bg, y + box_h), (0, 200, 255), L["rt"])
        _lines = entry_lines(res, plus_right)
        for i, s in enumerate(_lines):
            _x = bg + bp + 4
            _y = y + bp + L["ty"] + i * lh
            if i == 4:
                for _t, _c in liuku_segments(res, plus_right, nayta):
                    cv2.putText(img, _t, (_x, _y), FONT, fs, _c, ft)
                    _x += cv2.getTextSize(_t, FONT, fs, ft)[0][0]
                continue
            _col = (0, 255, 255) if i == 0 else (kierre_text(res)[1] if i == RIVI_KIERRE else (255, 255, 255))
            cv2.putText(img, s, (_x, _y), FONT, fs, _col, ft)
        if ages is not None and k < len(ages) and ages[k] is not None:
            txt = f"{ages[k]} s"
            (tw, _), _ = cv2.getTextSize(txt, FONT, fs, ft)
            cv2.putText(img, txt, (pw - bg - bp - 4 - tw, y + bp + L["ty"]), FONT, fs, (0, 255, 255), ft)
        y += box_h + bg
    return img


class DebugComposer:
    def __init__(self, src_w, src_h, puhelin=False):
        self.src_w, self.src_h = int(src_w), int(src_h)
        self.video_w, self.video_h, self.scale, self.total_w = debug_layout(src_w, src_h)
        # puhelinnakyma - tuplafontti paneeleissa (leveammat paneelit), kivilla ei aariviivoja, ID kiven oikealla puolella
        self.puhelin = bool(puhelin)
        self.L = PANEELI_PUHELIN if self.puhelin else PANEELI_NORMAALI
        self.pw = self.L["pw"]
        self.total_w = self.video_w + 2 * self.pw
        self.canvas = np.zeros((self.video_h, self.total_w, 3), np.uint8)
        self._key = {"L": None, "R": None}
        # skaalaus ennen kaantoa: (src_w x src_h) -> (video_h x video_w) = (DEBUG_H x video_w) kaantamattomana: leveys DEBUG_H, korkeus video_w
        self._pre_w, self._pre_h = DEBUG_H, self.video_w
        self.boxes = []           # laatikoiden paikat ja iat (puhelimen korostus)
        self.boxes_t = 0.0
        self.project = None       # (X, Y) cm -> (u, v) korjatussa kuvassa; asetetaan render_frame:ssa
        self.tausta = None        # tyhjan radan taustakuva (liikkuvien kohteiden tunnistus)
        self.nayta = (True, True) # puhelimen valinnat (katselu.py asettaa)
        self.korostus = None
        self.viive = None

    def _geom(self, res, plus_right, W0):
        """Liukusuorat (vasen hakki punainen, oikea vihrea) ja irroitus (oranssi) kuvapisteina kankaalla.
        self.project(X, Y) -> (u, v) lahdekuvassa (asetetaan render_frame:ssa); ilman sita tyhja."""
        pr = self.project
        if pr is None or "liuku_x_tee_cm_hakki_p" not in res or "tee_y_cm" not in res:
            return {}

        def cv_(X, Y):
            u, v = pr(X, Y)
            return [self.pw + v * self.scale, (W0 - 1 - u) * self.scale]
        tee, hy = res["tee_y_cm"], res.get("hakki_y_cm")
        if hy is None:
            return {}
        vas, oik = ("m", "p") if plus_right is None or bool(plus_right) else ("p", "m")
        viivat = []
        for tag, col, puoli in ((vas, "#ff2020", "v"), (oik, "#20e020", "o")):
            xt = res["liuku_x_tee_cm_hakki_" + tag]
            sl = -np.tan(np.radians(res["liuku_dir_deg_hakki_" + tag]))
            viivat.append(dict(p=[cv_(xt + sl * (hy - tee), hy), cv_(xt, tee)], c=col, s=puoli))
        g = dict(viivat=viivat)
        if res.get("x_straight_at_tee_cm") is not None:
            g["risti"] = cv_(res["x_straight_at_tee_cm"], tee)
        return g

    def _liike_maski(self, vid):
        """Liikkuvat kohteet (ihmiset, kivet, harjat) = ero tyhjan radan taustakuvaan (self.tausta, kalibroinnin
        moodikuva korjatussa kuvassa) samassa pienessa, kaannetyssa koossa. 255 = liikkuva kohde (viivaa ei piirreta sen paalle)."""
        t = self.tausta
        if t is None:
            return None
        if getattr(self, "_tausta_src", None) is not t:
            self._tausta_src = t
            self._tausta_small = cv2.rotate(cv2.resize(t, (self._pre_w, self._pre_h), interpolation=cv2.INTER_AREA),
                                            cv2.ROTATE_90_COUNTERCLOCKWISE)
        if self._tausta_small.shape != vid.shape:
            return None
        c0, c1, c2 = cv2.split(cv2.absdiff(vid, self._tausta_small))
        _, m = cv2.threshold(cv2.max(cv2.max(c0, c1), c2), A.KATSELU_LIIKE_KYNNYS, 255, cv2.THRESH_BINARY)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))       # yksittaiset kohinapikselit pois
        return cv2.dilate(m, np.ones((5, 5), np.uint8))                          # pieni reunavara kohteen ymparille

    def _viivastetty(self, vid, cap_wall, viive):
        """Puhelinnakyman keskikuva viive s myohassa (Alku). Videokuvat (kaannetty pieni kuva aariviivoineen)
        JPEG:na rengaspuskuriin kaappausajan (seinakello) mukaan; palautetaan uusin kuva jonka kaappausaika <= nyt - viive."""
        if not hasattr(self, "_rengas"):
            self._rengas = []
        nyt = time.time()
        ok, buf = cv2.imencode(".jpg", vid, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            self._rengas.append((nyt if cap_wall is None else float(cap_wall), buf))
        while self._rengas and self._rengas[0][0] < nyt - A.KATSELU_VIIVE_MAX_S:
            self._rengas.pop(0)
        if viive <= 0 or not self._rengas:
            return vid, 0.0
        tavoite = nyt - viive
        valinta = self._rengas[0]
        for e in self._rengas:
            if e[0] <= tavoite:
                valinta = e
            else:
                break
        if valinta[1] is buf:
            return vid, nyt - valinta[0]
        img = cv2.imdecode(valinta[1], cv2.IMREAD_COLOR)
        return (img if img is not None and img.shape == vid.shape else vid), nyt - valinta[0]

    def _piirra_viivat(self, vid, mask_r, gs):
        """Korostettujen laatikoiden liukusuorat ja irroitusristi videokuvaan liikkuvien kohteiden ALLE (vain mask_r == 0)."""
        nayta = tuple(self.nayta)
        ov = vid.copy()
        lm = np.zeros(vid.shape[:2], np.uint8)          # viivojen pikselit
        n = 0
        for g in gs:
            for v in g.get("viivat", []):
                if (v["s"] == "v" and not nayta[0]) or (v["s"] == "o" and not nayta[1]):
                    continue
                p0 = (int(round(v["p"][0][0] - self.pw)), int(round(v["p"][0][1])))
                p1 = (int(round(v["p"][1][0] - self.pw)), int(round(v["p"][1][1])))
                cv2.line(ov, p0, p1, VIIVA_BGR[v["s"]], 2, cv2.LINE_AA)
                cv2.line(lm, p0, p1, 255, 4)
                n += 1
            if g.get("risti") is not None:
                mx, my = g["risti"][0] - self.pw, g["risti"][1]
                for d in (-90, 30, 150):
                    t = np.radians(d)
                    q0, q1 = (int(round(mx)), int(round(my))), (int(round(mx + A.KATSELU_RISTI_SAKARA_PX * np.cos(t))), int(round(my + A.KATSELU_RISTI_SAKARA_PX * np.sin(t))))
                    cv2.line(ov, q0, q1, (0, 153, 255), 3, cv2.LINE_AA)
                    cv2.line(lm, q0, q1, 255, 5)
                n += 1
        if not n:
            return
        if mask_r is not None:
            lm = cv2.bitwise_and(lm, cv2.bitwise_not(mask_r))
        cv2.copyTo(ov, lm, vid)

    def compose(self, base_bgr, labels, header, results, plus_right, now_video_s=None, cap_wall=None):
        H0, W0 = base_bgr.shape[:2]
        small = cv2.resize(base_bgr, (self._pre_w, self._pre_h), interpolation=cv2.INTER_LINEAR)     # 1080 x 608
        vid = cv2.rotate(small, cv2.ROTATE_90_COUNTERCLOCKWISE)                                   # 608 x 1080 (leveys x korkeus)
        scale = self.scale
        self.boxes = []          # laatikoiden paikat ja iat puhelinnakyman korostusta varten
        self.boxes_t = time.time()
        L = self.L
        box_h = laatikon_mitat(L)[1]
        lfs, lft = (0.8, 2) if self.puhelin else (FONT_SCALE, FONT_THICK)
        for (x, y, txt, col) in labels:
            xr, yr = int(round(y * scale)), int(round((W0 - 1 - x) * scale))
            if self.puhelin:          # (x, y) = kiven oikea reuna kaannetyssa kuvassa, teksti pystysuunnassa keskelle
                (tw, th), _ = cv2.getTextSize(txt, FONT, lfs, lft)
                xr, yr = xr + 4, yr + th // 2
            cv2.putText(vid, txt, (max(2, min(self.video_w - 60, xr)), max(14, min(self.video_h - 4, yr))), FONT, lfs, col, lft)
        cv2.putText(vid, header, (10, 28), FONT, 0.6, (255, 255, 255), 2)
        viive = self.viive            # puhelin - keskikuva Alku s myohassa (None = ei viivetta, debug-video)
        try:
            viive = None if viive is None else max(0.0, float(viive))
        except (TypeError, ValueError):
            viive = 0.0
        if viive is not None:
            # uusin kuva jonka kaappausaika <= nyt - Alku. Jos kasittely on jaljessa enemman kuin Alku (tai mika tahansa
            # virhe), naytetaan uusin kuva; todellinen viive naytetaan aina, punaisena jos se on yli Alku-arvon.
            _cap = cap_wall
            try:
                vid, tod = self._viivastetty(vid, cap_wall, viive)
            except Exception:
                tod = None
            if tod is None or _cap is not None:
                tod = max(tod or 0.0, time.time() - _cap) if _cap is not None else (tod or 0.0)
            vid = vid.copy()
            _txt = f"viive: {tod:.0f} s"
            _col = (0, 0, 255) if tod > viive + 1.5 else (0, 255, 255)
            (tw, _), _ = cv2.getTextSize(_txt, FONT, 0.8, 2)
            cv2.putText(vid, _txt, (self.video_w - tw - 10, 30), FONT, 0.8, _col, 2)
        sides = []
        vali = self.korostus         # puhelimen Alku..Loppu (s); None -> viivoja ei piirreta
        piirra = []
        for side, x0 in (("L", 0), ("R", self.pw + self.video_w)):
            ents = [r for r in results if throw_side(r, plus_right) == side][::-1]
            ages = [entry_age_s(r, now_video_s) for r in ents]
            sides.append((side, x0, ents, ages))
            for k_, (r_, a_) in enumerate(zip(ents, ages)):
                yb = L["bg"] + k_ * (box_h + L["bg"])
                if yb + box_h > DEBUG_H:
                    break
                if a_ is not None:
                    self.boxes.append(dict(x=x0 + L["bg"], y=yb, w=self.pw - 2 * L["bg"], h=box_h, ika=float(a_),
                                           wall=r_.get("t_far_wall")))
                    if vali is not None and vali[0] <= a_ <= vali[1]:
                        piirra.append(self._geom(r_, plus_right, W0))
        if piirra:
            if not vid.flags.writeable or viive is not None:
                vid = vid.copy()
            self._piirra_viivat(vid, self._liike_maski(vid), piirra)
        self.canvas[:, self.pw:self.pw + self.video_w] = vid
        for side, x0, ents, ages in sides:
            nayta = tuple(self.nayta)     # puhelimen vasen-/oikeakatinen-valinta
            key = (nayta,) + tuple((r["stone_id"], r["frame"], a) for r, a in zip(ents, ages))
            if key != self._key[side]:                         # paneeli piirretaan uudelleen vain kun sisalto muuttui (sekuntilaskuri: kerran sekunnissa)
                self.canvas[:, x0:x0 + self.pw] = render_panel(ents, ages, plus_right=plus_right, nayta=nayta, L=L)
                self._key[side] = key
        return self.canvas




_ENCODER_ARGS = {
    "qsv": ["-vf", "format=nv12", "-c:v", "h264_qsv", "-global_quality", "26", "-look_ahead", "0", "-preset", "veryfast"],
    "x264": ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "23", "-pix_fmt", "yuv420p"],
}


def _yuv_input(w, h):
    """Ruudut muunnetaan BGR -> YUV 4:2:0 tassa prosessissa (cv2.cvtColor, SIMD) ja ffmpegille syotetaan valmis yuv420p:
    ffmpegin hitaampi BGR-muunnos jaa pois ja putkeen menee puolet vahemman tavuja. Vaatii parilliset mitat."""
    return int(w) % 2 == 0 and int(h) % 2 == 0


def _ffmpeg_cmd(w, h, fps, encoder, out_path):
    pix = "yuv420p" if _yuv_input(w, h) else "bgr24"
    return (["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", pix,
             "-s", f"{int(w)}x{int(h)}", "-r", f"{float(fps):.4f}", "-i", "-", "-an"] + _ENCODER_ARGS[encoder] + [out_path])


def probe_encoder(encoder, w, h, fps):
    """Kokeilee koodausta 3 mustalla ruudulla (-f null). Palauttaa (ok, virheteksti)."""
    cmd = _ffmpeg_cmd(w, h, fps, encoder, "-")
    cmd = cmd[:-1] + ["-f", "null", "-"]
    try:
        frame_bytes = int(w) * int(h) * 3 // 2 if _yuv_input(w, h) else int(w) * int(h) * 3
        p = subprocess.run(cmd, input=bytes(frame_bytes * 3), stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
        return p.returncode == 0, p.stderr.decode("utf-8", "replace").strip()[-300:]
    except Exception as e:                      # ffmpeg puuttuu / aikakatkaisu
        return False, repr(e)


class FfmpegPipeWriter:
    """cv2.VideoWriter-yhteensopiva (write/release): raakaruudut ffmpeg-prosessin stdiniin."""

    def __init__(self, path, fps, size, encoder):
        self._dead = False
        self._yuv = _yuv_input(size[0], size[1])
        self._p = subprocess.Popen(_ffmpeg_cmd(size[0], size[1], fps, encoder, path), stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, **yleiset.bg_popen_kwargs())

    def write(self, img):
        if self._dead:
            return
        try:
            if self._yuv:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV_I420)
            self._p.stdin.write(np.ascontiguousarray(img).data)
        except Exception as e:                  # ffmpeg kuoli -> ei kaadeta seurantaa
            self._dead = True
            print(f"VAROITUS: debug-videon ffmpeg-koodaus keskeytyi ({e!r}); loput ruudut jatetaan kirjoittamatta.")

    def release(self):
        try:
            self._p.stdin.close()
        except Exception:
            pass
        self._p.wait()


def open_debug_writer(path, fps, size):
    """Valitsee debug-videon kirjoittajan: laitteisto-QSV jos kaytettavissa, muuten cv2.VideoWriter (mp4v)."""
    order = {"auto": ["qsv"], "qsv": ["qsv"], "x264": ["x264"], "opencv": []}.get(A.DEBUG_KOODAUS, ["qsv"])
    for enc in order:
        ok, err = probe_encoder(enc, size[0], size[1], fps)
        if ok:
            print(f"Debug-video: koodaus ffmpeg/{enc} ({'Intel Quick Sync, grafiikkapiiri' if enc == 'qsv' else 'CPU'})"
                  f"{', syote YUV 4:2:0' if _yuv_input(size[0], size[1]) else ''}")
            return FfmpegPipeWriter(path, fps, size, enc)
        print(f"Debug-video: ffmpeg/{enc} ei kaytettavissa ({err or 'tuntematon virhe'}) -> cv2.VideoWriter (mp4v, CPU)")
    return cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)




class AsyncVideoWriter:
    """cv2.VideoWriter kahdessa taustasaikeessa: (1) piirto/kokoonpano, (2) kirjoitus (koodaus). Pääsäie vain jonottaa tyon (submit)
    tai valmiin kuvan (write); jono taynna -> odottaa."""

    def __init__(self, writer, maxsize=8):
        self._w = writer
        self._qr = queue.Queue(maxsize=maxsize)    # piirtotyot
        self._qw = queue.Queue(maxsize=maxsize)    # valmiit kuvat kirjoitettavaksi
        self._tr = threading.Thread(target=self._run_render, daemon=True)
        self._tw = threading.Thread(target=self._run_write, daemon=True)
        self._tr.start()
        self._tw.start()

    def _run_render(self):
        while True:
            item = self._qr.get()
            if item is None:
                self._qw.put(None)
                break
            fn, args = item
            # compose() palauttaa jaetun canvas-puskurin -> kopio ennen jonoon laittoa (kirjoitus on eri saikeessa)
            self._qw.put(fn(*args).copy())

    def _run_write(self):
        while True:
            img = self._qw.get()
            if img is None:
                break
            self._w.write(img)

    def write(self, img):
        self._qw.put(img.copy())

    def submit(self, fn, *args):
        """Tyo (fn(*args) -> kuva) piirretaan taustasaikeessa ja kirjoitetaan toisessa; paasaie palaa heti."""
        self._qr.put((fn, args))

    def release(self):
        self._qr.put(None)
        self._tr.join()
        self._tw.join()
        self._w.release()
