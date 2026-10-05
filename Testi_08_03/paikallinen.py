"""SEURANTA paikallisesti kameran taydella resoluutiolla (live, kun kamera/video on seurantaa suurempi).

Putki (stabilointi, warp, varjosuodatus, HAKU, nakymat) toimii 1280x720:ssa. SEURANTA saa kuvan, jossa jokaisen
seurattavan kiven hakualue on kasitelty kameran taydella resoluutiolla: alueen pikselit haetaan puskurin
taysresoluutioisesta raakaruudusta (YUY2-muunnos vain alueelta) samalla stabiloinnilla (siirto skaalattuna) ja
linssikorjauksella (taysresoluutioinen kalibrointi) ja taustasuodatetaan taysresoluutioista moodikuvaa vasten; muu kuva
on valkoista (= tausta). Kameramatriisi taydella resoluutiolla; tulokset senttimetreina kuten ennen. Jos ruudun
taysresoluutioista kuvaa ei enaa ole puskurissa (viive > A.PAIK_RENGAS_S), ruutu seurataan 720p:na.
Myos kierrepiirre lasketaan taysresoluutioisesta kuvasta (rinnakkain 720p:n kanssa).
"""
import cv2
import numpy as np

import stone_tracker

import asetukset as A
import esikasittely
import kierre
import kivimalli
import siluetti


class PaikallinenTaysi:
    def __init__(self, calib_result, store, fps, offset, alku, profile, sil_lo):
        """offset: paasilmukan ruutunumero -> puskurin indeksi (live-hypyn jalkeen); alku: viimeinen ennen hyppya luettu
        ruutu. sil_lo: SEURANNAN siluettiasetus 720p:lle (stone_tracker.set_seuranta_silhouette)."""
        c = calib_result["calib"]
        pose = calib_result["pose"]
        self.store = store
        self.ref = np.ascontiguousarray(c["taysi_frame_undistorted"])
        self.H, self.W = self.ref.shape[:2]
        dist = np.array([c["best_k1"], 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.map1, self.map2 = kivimalli.undistort_maps(c["taysi_camera_matrix"], dist, (self.W, self.H))
        self.K = pose["K_taysi"]
        self.s = self.W / float(c["image_width"])
        self.canvas = np.full((self.H, self.W, 3), 255, np.uint8)
        self.prev = []
        self.offset = offset
        self.alku = alku
        self.tarkistus = 0
        # rengas kattaa myos liukuhihnan jonot (vaihe A lukee jopa 2 x A.JONON_SYVYYS ruutua seurannan edella)
        store.hires_done = -1
        store.hires_seuranta = max(int(round(A.PAIK_RENGAS_S * fps)), 2 * A.JONON_SYVYYS + 25)
        # SEURANNAN siluettitarkistus (C++) projisoi kiven omalla kameramatriisillaan -> taydelle resoluutiolle oma
        # asetus (K taysi, pikselimitat x kerroin); vaihdetaan sen mukaan kumpi kuva seurannalle annetaan
        self.sil_lo = sil_lo
        self.sil_hi = None
        self.sil_tila = "lo"
        if sil_lo is not None:
            s_ = self.s
            ref_hi = siluetti.SilhouetteRefiner(
                dict(pose, K=self.K), profile["R_max_cm"], profile["H_total_cm"], profile["shape_deltas"],
                profile["handle_r_frac"], max_shift_px=int(round(A.SEURANTA_SIL_SIIRTO_PX * s_)),
                open_size=int(round(siluetti.OPEN_SIZE * s_)) | 1, band_px=int(round(siluetti.BAND_PX * s_)),
            )
            self.sil_hi = ref_hi.batch_config(half=int(round(siluetti.HALF * s_)), margin=int(round(siluetti.MASK_MARGIN * s_)))
        self.stat = {"taysi": 0, "720p": 0, "alueita": 0, "pikseleita": 0, "puute_ika": []}

    @staticmethod
    def alusta(calib_result, store, fps, offset, alku, profile, sil_lo):
        """PaikallinenTaysi tai None (ei taysresoluutioista kalibrointia tai virhe)."""
        if "taysi_frame_undistorted" not in calib_result["calib"] or "K_taysi" not in calib_result["pose"]:
            print("Paikallinen taysi resoluutio: ei taysresoluutioista kalibrointia -> seuranta 720p:na")
            return None
        try:
            p = PaikallinenTaysi(calib_result, store, fps, offset, alku, profile, sil_lo)
        except Exception as e:
            print(f"Paikallinen taysi resoluutio: alustus epaonnistui ({e!r}) -> seuranta 720p:na")
            return None
        print(f"SEURANTA paikallisesti taydella resoluutiolla ({p.W}x{p.H}, kerroin {p.s:.3f}; "
              f"rengas {store.hires_seuranta} ruutua)")
        return p

    def ruutu_alussa(self, frame_index, frame):
        """Ruudun alussa: vanhemmat taysresoluutioiset ruudut pois renkaasta; kolmella ensimmaisella ruudulla tarkistetaan
        etta puskurin ruutu samalla indeksilla on kasiteltava ruutu. Palauttaa False jos paikallinen tila pitaa sulkea."""
        self.store.mark_hires_done(frame_index + self.offset - 1)
        if self.tarkistus < 3 and frame is not None and frame_index > self.alku:
            self.tarkistus += 1
            fs = self.store.get(frame_index + self.offset, timeout=0)
            ok = fs is not None and fs.shape == frame.shape and np.array_equal(fs, frame)
            print(f"Paikallinen taysi resoluutio: ruutu {frame_index} -> puskurin indeksi {frame_index + self.offset}: "
                  f"{'OK' if ok else 'EI TASMAA -> pois kaytosta'}")
            if not ok:
                self.store.hires_seuranta = 0
                return False
        return True

    def _alue(self, frame_index, Mi, ax, ay, bx, by):
        """Stabiloitu + linssikorjattu taysresoluutioinen alue [ay:by, ax:bx] BGR:na tai None (ei puskurissa)."""
        W, H = self.W, self.H
        ax, ay, bx, by = max(0, ax), max(0, ay), min(W, bx), min(H, by)
        if bx - ax < 4 or by - ay < 4:
            return None
        mx = self.map1[ay:by, ax:bx]
        my = self.map2[ay:by, ax:bx]
        sx = Mi[0, 0] * mx + Mi[0, 1] * my + Mi[0, 2]
        sy = Mi[1, 0] * mx + Mi[1, 1] * my + Mi[1, 2]
        rx0 = int(max(0, np.floor(sx.min()) - 2))
        rx1 = int(min(W, np.ceil(sx.max()) + 3))
        ry0 = int(max(0, np.floor(sy.min()) - 2))
        ry1 = int(min(H, np.ceil(sy.max()) + 3))
        if rx1 - rx0 < 2 or ry1 - ry0 < 2:
            return None
        roi, ox = self.store.get_hires_roi(frame_index, rx0, ry0, rx1, ry1)
        if roi is None:
            return None
        return cv2.remap(roi, (sx - ox).astype(np.float32), (sy - ry0).astype(np.float32), cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)

    def _inv_stab(self, stab_M):
        M = np.asarray(stab_M, dtype=np.float64)[:2].copy()
        M[:, 2] *= self.s
        return cv2.invertAffineTransform(M)

    def _kehys(self, frame_index, stab_M, X0, Y0, hx, hy, R_max, H_total, pose, gain, bias):
        """Seurannan taysresoluutioinen kuva (valkoinen pohja + kivien hakualueet) tai None (ruutua ei puskurissa)."""
        cv_ = self.canvas
        for (a, b, c_, d) in self.prev:
            cv_[a:b, c_:d] = 255
        self.prev = []
        Mi = self._inv_stab(stab_M)
        W, H = self.W, self.H
        r = R_max + A.PAIK_MARGINAALI_CM
        for x0c, y0c, hxc, hyc in zip(X0, Y0, hx, hy):
            xs = [x0c - hxc - r, x0c + hxc + r]
            ys = [y0c - hyc - r, y0c + hyc + r]
            pts = np.array([[x, y, z] for x in xs for y in ys for z in (0.0, H_total + 3.0)], dtype=np.float64)
            u, v = kivimalli.project_3d(self.K, pose["R"], pose["t"], pts)
            if not (np.all(np.isfinite(u)) and np.all(np.isfinite(v))):
                continue
            ax, bx = int(max(0, np.floor(u.min()) - 4)), int(min(W, np.ceil(u.max()) + 5))
            ay, by = int(max(0, np.floor(v.min()) - 4)), int(min(H, np.ceil(v.max()) + 5))
            if bx - ax < 4 or by - ay < 4:
                continue
            mx = self.map1[ay:by, ax:bx]
            my = self.map2[ay:by, ax:bx]
            sx = Mi[0, 0] * mx + Mi[0, 1] * my + Mi[0, 2]
            sy = Mi[1, 0] * mx + Mi[1, 1] * my + Mi[1, 2]
            rx0 = int(max(0, np.floor(sx.min()) - 2))
            rx1 = int(min(W, np.ceil(sx.max()) + 3))
            ry0 = int(max(0, np.floor(sy.min()) - 2))
            ry1 = int(min(H, np.ceil(sy.max()) + 3))
            if rx1 - rx0 < 2 or ry1 - ry0 < 2:
                continue
            roi, ox = self.store.get_hires_roi(frame_index, rx0, ry0, rx1, ry1)
            if roi is None:
                return None
            patch = cv2.remap(roi, (sx - ox).astype(np.float32), (sy - ry0).astype(np.float32), cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
            ref = np.ascontiguousarray(self.ref[ay:by, ax:bx])
            cv_[ay:by, ax:bx] = esikasittely.suppress_shadow_background(np.ascontiguousarray(patch), ref, gain, bias)
            self.prev.append((ay, by, ax, bx))
            self.stat["alueita"] += 1
            self.stat["pikseleita"] += (by - ay) * (bx - ax)
        return cv_

    def seurantakuva(self, frame_index, stab_M, X0, Y0, hx, hy, R_max, H_total, pose, gain, bias):
        """(kuva, referenssi, K) taysresoluutioisena tai None (seurataan 720p:na). Vaihtaa SEURANNAN siluettiasetuksen
        vastaamaan annettavaa kuvaa."""
        pk = None if frame_index <= self.alku else self._kehys(frame_index + self.offset, stab_M, X0, Y0, hx, hy,
                                                                 R_max, H_total, pose, gain, bias)
        want = "hi" if pk is not None else "lo"
        if self.sil_hi is not None and self.sil_tila != want:
            stone_tracker.set_seuranta_silhouette(*(self.sil_hi if want == "hi" else self.sil_lo))
            self.sil_tila = want
        if pk is not None:
            self.stat["taysi"] += 1
            return pk, self.ref, self.K
        self.stat["720p"] += 1
        self.stat["puute_ika"].append(self.store._head - (frame_index + self.offset))   # kamera seurannan edella
        return None

    def skaalaa_rms(self, batch_results):
        """Pikseleina annetut rajat (portit, varakeinot) on viritetty 720p:lle -> rms_px 720p-mittakaavaan."""
        for r in batch_results:
            if isinstance(r, dict) and r.get("rms_px") is not None:
                r["rms_px"] = r["rms_px"] / self.s

    def kierre_piirre(self, frame_index, stab_M, cx, cy, rp, gx, gy):
        """Kierrepiirre taysresoluutioisesta kuvasta (cx, cy, rp, gx, gy 720p-kuvassa) tai None."""
        if frame_index <= self.alku:
            return None
        Mi = self._inv_stab(stab_M)
        s_ = self.s
        X, Y, R = cx * s_, cy * s_, rp * s_
        Rk = 1.42 * R                  # akselin suuntainen pala voi olla vino: puolilavistaja
        ax, ay = int(np.floor(X - Rk)) - 3, int(np.floor(Y - Rk)) - 3
        bx, by = int(np.ceil(X + Rk)) + 4, int(np.ceil(Y + Rk)) + 4
        a0, b0 = max(0, ax), max(0, ay)
        patch = self._alue(frame_index + self.offset, Mi, ax, ay, bx, by)
        if patch is None:
            return None
        return kierre.kierre_piirre(patch, X - a0, Y - b0, gx * s_ - a0, gy * s_ - b0, R)

    def raportti(self):
        st = self.stat
        n = max(1, st["alueita"])
        print(f"Paikallinen taysi resoluutio: SEURANTA-ruutuja taydella resoluutiolla {st['taysi']}, 720p:na {st['720p']} "
              f"(taysresoluutioinen ruutu ei enaa puskurissa); kiven alueita {st['alueita']}, keskimaarin "
              f"{st['pikseleita'] / n:.0f} px/alue")
        if st["puute_ika"]:
            pa = np.asarray(st["puute_ika"], dtype=np.float64)
            print(f"  720p-ruuduissa kamera oli seurannan edella: mediaani {np.median(pa):.0f}, max {pa.max():.0f} ruutua "
                  f"(rengas {self.store.hires_seuranta} ruutua)")
