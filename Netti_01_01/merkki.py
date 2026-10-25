"""Maalimerkin (kaukopaan harjanpaa) haku yksittaisesta ruudusta.

Heittajan maali on kaukopaassa: joukkueen kippari pitaa harjaa jaalla harjanpaa kohti heittajaa, ja heittaja tahtaa sita kohti. Harjanpaa on oranssinruskea muovipala (OpenCV-sävy H ~ 7-17, kylläisyys >= 95),
joten maalin voi etsia varista ja muodosta, kunhan erotetaan siita

  * kiven keltainen kahva (kahvan reunus ja osassa videoita koko kahva on samaa savya): kahva on pieni, kaareva ja sen pitkä akseli on poikittain kenttaan nahden, harjanpaa taas kentan suuntainen,
  * puulattia (sama sävy): |X'| <= X_MAX_CM -kaista leikataan jaan reunaan,
  * harjan varsi (puuvarsi on samaa savya ja liittyy harjanpaahan): paksumpi avaus irrottaa varren.

Menetelma (yksi ruutu; harjanpaan oletetaan pysyvan paikallaan heiton ajan): kaukopaan kuva oikaistuna (Mittari.kuvain.leike) -> HSV-maski (H 7-17, S >= 95, V >= 110) rajattuna jaalle -> avaus 5 px
(ohuet rakenteet pois) -> yhtenaiset alueet. Alue on ehdokas, kun sen keskipiste (maatasolle, z = 3 cm) on kaistalla, ala on harjanpaan kokoinen (MERKKI_ALA_CM2), sivusuhde 1,4-7 ja pitka akseli enintaan
MERKKI_KULMA_MAX astetta kentan suunnasta (kentan suunta on oikaistussa kuvassa vaaka). Kaistalla oleva mutta koon tai muodon puolesta kelpaamaton alue (harjanpaa + varsi) pilkotaan paksummilla avauksilla (9-13 px).
Useita ehdokkaita voi olla (muiden pelaajien harjat, kivien kahvat): maali on pesan alueella (|Y'| <= MERKKI_Y_PESA_CM) oleva, usean joukosta suurin; ellei pesan alueella ole yhtaan, kelpaa muukin
kelvollinen (vartijakivi-maali). Havaintoja: kipparin harjanpaa on kaikissa viidessa videossa kentan suuntainen (0-9 astetta), kivien kahvat 60-90 astetta.

Koordinaatit kaukopaan koordinaateissa (X', Y', cm; origo pesan keskipiste, kuten kaukopaan seurannassa). Moduuli ei tunne seurantaa muuten kuin Mittarin (kuvain, kamera, koordinaatit) kautta."""
import cv2
import numpy as np

import kivimalli

H_MIN, H_MAX = 7, 17                    # harjanpaan sävy (OpenCV 0-179): ruskeampi harjanpaa ~8-11, oranssi ~13-16, kiven keltainen kahva ~13-22 (videosta riippuen)
S_MIN, V_MIN = 95, 110                  # pienempi V_MIN: varjossa oleva harjanpaa (V ~ 130-140) kuuluu mukaan
AVAUS_PX = (5, 9, 11, 13)                # avauskoot: 5 poistaa ohuet rakenteet (kahvan reunus ~3 px), loput irrottavat harjanpaan varresta (~9 px)
X_MAX_CM = 245.0                         # kaista: jaa (puulattia alkaa ~250 cm:ssa)
Y_RAJAT_CM = (-330.0, 520.0)
Z_CM = 3.0                               # harjanpaa on jaalla
MERKKI_ALA_CM2 = (45.0, 900.0)           # mitattu: 70-520 cm2 (osittain peittyneena pienempi)
MERKKI_SIVUSUHDE = (1.4, 7.0)
MERKKI_KULMA_MAX = 35.0                  # harjanpaan pitka akseli kentan suuntaan (mitattu 0-9 astetta; kiven kahvat 60-90)
MERKKI_Y_PESA_CM = 235.0                 # pesan alue (12 jalan rengas 183 cm + harjan ulottuma)
MIN_PX = 25


def _alue(mittari):
    cam = mittari.cam
    P = np.array([(x, y, z) for x in (-X_MAX_CM - 40.0, X_MAX_CM + 40.0) for y in Y_RAJAT_CM for z in (0.0, 30.0)])
    u, v = kivimalli.project_3d(cam.K, cam.R, cam.t, P)
    return (int(u.min()) - 6, int(v.min()) - 6, int(u.max()) + 6, int(v.max()) + 6)


def _kaista(mittari, x0, y0, w, h, askel=8):
    """Leikkeen (x0, y0, w x h) pikselit, jotka ovat jaalla (|X'| <= X_MAX_CM): puulattia on samaa savya kuin harjanpaa ja liittaisi sen mukaansa. Valimuistissa Mittarilla."""
    avain = (x0, y0, w, h)
    v = getattr(mittari, "_merkki_kaista", None)
    if v is not None and v[0] == avain:
        return v[1]
    yy, xx = np.mgrid[0:h + askel:askel, 0:w + askel:askel]
    X, _ = kivimalli.ray_plane_intersection(mittari.cam.K, mittari.cam.R, mittari.cam.t, (xx + x0).ravel(), (yy + y0).ravel(), Z_CM)
    X = cv2.resize(X.reshape(xx.shape).astype(np.float32), None, fx=askel, fy=askel, interpolation=cv2.INTER_LINEAR)[:h, :w]
    maski = (np.abs(X) <= X_MAX_CM).astype(np.uint8)
    mittari._merkki_kaista = (avain, maski)
    return maski


def _avaa(m, k):
    """Morfologinen avaus, joka poistaa alle k px paksut rakenteet mutta sailyttaa jaljelle jaavien osien koon."""
    e = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(cv2.morphologyEx(m, cv2.MORPH_OPEN, e), e) & m


def _arvioi(mittari, m, x0, y0, H, S):
    """Maskin (0/1) kukin yhtenainen alue ehdokkaaksi: [(dict, alueen maski)]."""
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    out = []
    for k in range(1, n):
        a = int(st[k, cv2.CC_STAT_AREA])
        if a < MIN_PX:
            continue
        kmask = lab == k
        ys, xs = np.nonzero(kmask)
        (_, _), (w, h), ang = cv2.minAreaRect(np.column_stack([xs, ys]).astype(np.float32))
        L, W = max(w, h), max(min(w, h), 1.0)
        kulma = (ang if w >= h else ang + 90.0) % 180.0          # pitkan akselin kulma kuvan vaakasuuntaan (= kentan suunta), 0-90
        kulma = min(kulma, 180.0 - kulma)
        u, v = float(cen[k][0] + x0), float(cen[k][1] + y0)
        X, Y = mittari.oikaistu_maahan(u, v, z=Z_CM)
        s = mittari.skaala(X, Y, Z_CM)
        ala = a / (s * s)
        syy = ""
        if abs(X) > X_MAX_CM or not (Y_RAJAT_CM[0] <= Y <= Y_RAJAT_CM[1]):
            syy = "kaistan ulkopuolella"
        elif not (MERKKI_ALA_CM2[0] <= ala <= MERKKI_ALA_CM2[1]):
            syy = "koko"
        elif not (MERKKI_SIVUSUHDE[0] <= L / W <= MERKKI_SIVUSUHDE[1]):
            syy = "muoto"
        elif kulma > MERKKI_KULMA_MAX:
            syy = "suunta"
        out.append((dict(X=float(X), Y=float(Y), u=u, v=v, ala_cm2=float(ala), sivusuhde=float(L / W), pituus_cm=float(L / s), leveys_cm=float(W / s), kulma=float(kulma),
                         sävy=float(np.median(H[kmask])), kyllaisyys=float(np.median(S[kmask])), kelvollinen=(syy == ""), syy=syy), kmask.astype(np.uint8)))
    return out


def ehdokkaat(mittari, ruutu):
    """Kaikki harjanpaan nakoiset alueet kaukopaan kuvasta: [dict(X, Y, u, v, ala_cm2, sivusuhde, pituus_cm, leveys_cm, kulma, sävy, kyllaisyys, kelvollinen, syy)]. Ruutu: koko ruutu (BGR).
    Kaistalla oleva alue, jonka koko tai muoto ei kelpaa (harjanpaa + sen varsi samaa puunvarista), pilkotaan paksummilla avauksilla: varsi (~9 px) irtoaa, harjanpaa (>= 13 px) jaa."""
    roi, x0, y0 = mittari.kuvain.leike(ruutu, *_alue(mittari))
    if roi is None:
        return []
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    m = ((H >= H_MIN) & (H <= H_MAX) & (S >= S_MIN) & (V >= V_MIN)).astype(np.uint8)
    m = _avaa(cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) & _kaista(mittari, x0, y0, roi.shape[1], roi.shape[0]), AVAUS_PX[0])
    out = []
    for e, kmask in _arvioi(mittari, m, x0, y0, H, S):
        if e["syy"] in ("koko", "muoto"):
            for k in AVAUS_PX[1:]:
                osat = [f for f, _ in _arvioi(mittari, _avaa(kmask, k), x0, y0, H, S) if f["syy"] != "kaistan ulkopuolella"]
                if any(f["kelvollinen"] for f in osat):
                    out += osat
                    break
            else:
                out.append(e)
        else:
            out.append(e)
    return out


def valitse(ehd):
    """Maali: kelvollinen ehdokas, ensisijaisesti pesan alueella (|Y'| <= MERKKI_Y_PESA_CM; muualla ovat muiden pelaajien harjat, mutta myos vartijakivi-maali), usean joukosta suurin; None, ellei kelvollisia ole."""
    kelp = [e for e in ehd if e["kelvollinen"]]
    kelp = [e for e in kelp if abs(e["Y"]) <= MERKKI_Y_PESA_CM] or kelp
    return max(kelp, key=lambda e: e["ala_cm2"]) if kelp else None


def etsi(mittari, ruutu):
    """(maali | None, kaikki ehdokkaat): maali = dict(X, Y, u, v, ...) kaukopaan koordinaateissa."""
    ehd = ehdokkaat(mittari, ruutu)
    return valitse(ehd), ehd
