"""Löytyivätkö kaikki heitetyt kivet? Heittojen täydellisyys CSV:stä (netti_live.py --csv), ilman totuusaineistoa.

    python tools/tarkista_kivet.py heitot.csv [--loki ajo.log]

Curling-pää on 16 heittoa (8 keltaista + 8 punaista, kivi_nro 1-8 kummallekin) ja joukkueet heittävät vuorotellen, joten puuttuva heitto näkyy ilman mitään totuusaineistoa:
  * pään heittojen määrä on muu kuin 16 (videon ensimmäinen ja viimeinen pää voivat olla osittaisia: video voi alkaa kesken päätä ja loppua kesken päätä, joten niiden määrää ei tarkisteta, vain ettei ylity 16),
  * värit eivät vuorottele pään sisällä (kaksi samanvärisiä peräkkäin: väliin jäänyt heitto; tarkistetaan kaikissa päissä) tai värejä on epätasan (vain täysissä päissä),
  * (vain huomautus, ei virhe) aikaväli kahden heiton välillä on yli 2,2 x mediaani: pään lopussa joukkueet miettivät pitkään, joten pitkä väli yksin ei kerro hukatusta heitosta, kun määrä ja värien vuorottelu ovat kunnossa.
Lisäksi --loki (netti_live.py --tapahtumat -tuloste): hylätyt heittoyritykset, joissa kiven nopeus oli >= 100 cm/s, ovat mahdollisia hukattuja heittoja (kivi liikkui mutta yritys hylättiin).
Paluuarvo 0 = kaikki kunnossa, 1 = epäilyttäviä kohtia."""
import argparse
import csv
import re
import sys
from collections import defaultdict

import numpy as np

HEITTOJA_PAASSA = 16
AIKAVALI_KERROIN = 2.2
HYLATTY_V_MIN_CM_S = 100.0


def lue(polku):
    with open(polku, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def tarkista(rivit):
    """(ongelmat: [str], tiedot: dict) CSV-riveistä."""
    ongelmat, huomiot, paat = [], [], defaultdict(list)
    for r in rivit:
        paat[int(r["paa"])].append(r)
    if not rivit:
        return ["ei heittoja"], {}
    ajat = []
    for k in sorted(paat):
        ajat += [float(b["alku_s"]) - float(a["alku_s"]) for a, b in zip(paat[k], paat[k][1:])]
    mediaani = float(np.median(ajat)) if ajat else float("nan")
    tiedot = dict(heittoja=len(rivit), paita=len(paat), aikavali_mediaani_s=mediaani, paat={}, huomiot=huomiot)
    for k in sorted(paat):
        h = paat[k]
        v = "".join(r["vari"][:1] for r in h)
        kelt, pun = v.count("k"), v.count("p")
        rikot = [i + 1 for i in range(1, len(v)) if v[i] == v[i - 1]]
        aukot = [(i + 1, float(b["alku_s"]) - float(a["alku_s"])) for i, (a, b) in enumerate(zip(h, h[1:])) if float(b["alku_s"]) - float(a["alku_s"]) > AIKAVALI_KERROIN * mediaani]
        t = dict(heittoja=len(h), keltaisia=kelt, punaisia=pun, vari_jono=v, alkaa_s=float(h[0]["alku_s"]), vuorottelun_rikkoja=rikot, pitkia_valeja=aukot)
        tiedot["paat"][k] = t
        osittainen = k in (min(paat), max(paat)) and len(paat) > 1                     # videon ensimmainen ja viimeinen paa
        if len(h) > HEITTOJA_PAASSA or (len(h) < HEITTOJA_PAASSA and not osittainen):
            ongelmat.append(f"pää {k}: {len(h)} heittoa ({kelt} keltaista, {pun} punaista), pitäisi olla {HEITTOJA_PAASSA}")
        elif len(h) < HEITTOJA_PAASSA:
            huomiot.append(f"pää {k}: {len(h)} heittoa (video alkaa tai loppuu kesken päätä?)")
        elif kelt != pun:
            ongelmat.append(f"pää {k}: {kelt} keltaista ja {pun} punaista")
        if rikot:
            ongelmat.append(f"pää {k}: värit eivät vuorottele kohdissa {rikot} (väliin jäänyt heitto?): {v}")
        if aukot:
            huomiot.append(f"pää {k}: pitkä väli heitossa {', '.join(f'{i} -> {i + 1} ({s:.0f} s, mediaani {mediaani:.0f} s)' for i, s in aukot)}")
    return ongelmat, tiedot


def hylatyt_liikkuvat(loki):
    """Hylätyt heittoyritykset, joissa v >= HYLATTY_V_MIN_CM_S: [(aika_s, yritys, v)]."""
    out = []
    for rivi in open(loki, encoding="utf-8", errors="replace"):
        m = re.search(r"\[\s*([\d.]+) s\] tapahtuma hylatty \(yritys (\d+)\) v (-?\d+) cm/s", rivi)
        if m and abs(float(m.group(3))) >= HYLATTY_V_MIN_CM_S:
            out.append((float(m.group(1)), int(m.group(2)), float(m.group(3))))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--loki", help="netti_live.py --tapahtumat -tuloste")
    a = ap.parse_args()
    ongelmat, tiedot = tarkista(lue(a.csv))
    print(f"{tiedot.get('heittoja', 0)} heittoa, {tiedot.get('paita', 0)} päätä; heittoväli mediaani {tiedot.get('aikavali_mediaani_s', float('nan')):.0f} s")
    for k, t in tiedot.get("paat", {}).items():
        print(f"  pää {k:2d}: {t['heittoja']:2d} heittoa ({t['keltaisia']} k, {t['punaisia']} p) alkaa {t['alkaa_s']:7.0f} s  {t['vari_jono']}")
    if a.loki:
        h = hylatyt_liikkuvat(a.loki)
        print(f"hylättyjä yrityksiä, joissa v >= {HYLATTY_V_MIN_CM_S:.0f} cm/s: {len(h)}" + (": " + ", ".join(f"{t:.0f} s (v {v:.0f})" for t, _, v in h[:20]) if h else ""))
    for o in tiedot.get("huomiot", []):
        print("  huomio: " + o)
    if ongelmat:
        print("\nEPÄILYTTÄVÄÄ:")
        for o in ongelmat:
            print("  - " + o)
        return 1
    print("\nOK: kaikissa päissä 16 heittoa (8 + 8), värit vuorottelevat")
    return 0


if __name__ == "__main__":
    sys.exit(main())
