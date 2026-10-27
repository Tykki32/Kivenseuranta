"""Kahvan kulman merkintojen (tools/luo_kahvamerkitsin.py:n sivu) luku ja vertailu koneen arvioon.

    python tools/kierre_etiketit.py --valinta sivu/valinta.json --merkinnat merkinnat.json [--tulos raportti.json]

Merkinnat: sivun "Lataa tiedostona" -JSON (lista asiakirjoja viivat/<id>) tai ArtifactData-listaus (lista {id, data}). valinta.json on luo_kahvamerkitsin.py:n tuottama (ruutu, heitto, aika ja
koneen arvio). Merkinnan kulma on topdown-leikkeen kulma (0 = vaakasuora, kasvaa vastapaivaan; leike on oikea ylhaalta katsottu kuva, joten kierron suunta on sama kuin kivella ylhaalta katsoen).

Heittokohtainen kierto: heiton ruutujen kulmat (tila ok tai epavarma) aika-jarjestyksessa puretaan 180 asteen jaksosta (kahva on symmetrinen) ja niihin sovitetaan suora (pienimman neliosumman,
aikana t_s): kulmakerroin on omega (aste / s, + = vastapaivaan ylhaalta katsoen, kuten koneen omega_deg_s) ja sen keskivirhe. Koneen arvion kanssa verrataan suuntaa (merkki) ja suuruutta.
Lisaksi vertaillaan vanhan kasi-suodattimen (kasi_px > KASI_RAJA_PX) ja merkitsijan "ei nay" / "kasi kahvassa" -merkintojen yhteensopivuutta."""
import argparse
import json

import numpy as np

KASI_RAJA_PX = 60           # kierto_netti.KASI_RAJA_PX: koneen kasi-suodatin hylkaa ruudun, jos kasi_px on tata suurempi
MIN_RUUTUJA = 3             # heiton omega arvioidaan vasta, kun viivallisia ruutuja on vahintaan tama monta
MIN_VALI_S = 0.3            # ... ja ne ulottuvat vahintaan taman pitkalle ajassa
SUUNTA_RAJA = 20.0          # suunta verrataan vain, kun |omega| (merkinnoista) on tata suurempi (aste / s) ja yli 3 keskivirhetta
HYPPY_RAJA = 60.0           # perakkaisten ruutujen kulmaero (aste), jonka ylittyessa 180 asteen purku on epaselva


def lue_merkinnat(polku):
    """{id: asiakirja}; hyvaksyy listan asiakirjoja tai ArtifactData-listauksen ({id, data})."""
    raaka = json.load(open(polku, encoding="utf-8"))
    if isinstance(raaka, dict):
        raaka = raaka.get("merkinnat") or raaka.get("documents") or list(raaka.values())
    out = {}
    for d in raaka:
        d = d.get("data", d) if isinstance(d, dict) and isinstance(d.get("data"), dict) else d
        if isinstance(d, dict) and d.get("id") and d.get("tila") in ("ok", "epavarma", "ei_nay", "kasi"):
            out[d["id"]] = d
    return out


def pura_kulmat(a):
    """Kulmat (aste, jakso 180) peräkkäisyysjärjestyksessä jatkuvaksi käyräksi: jokainen kulma valitaan edellisen lähimmäksi (ero -90...90). Palauttaa (purettu, suurin hyppy)."""
    u = [float(a[0])]
    hyppy = 0.0
    for x in a[1:]:
        d = (float(x) - u[-1] + 90.0) % 180.0 - 90.0
        hyppy = max(hyppy, abs(d))
        u.append(u[-1] + d)
    return np.array(u), hyppy


def sovita_omega(t, a):
    """Suora kulma(t): (omega aste / s, keskivirhe, jaannosten keskihajonta aste, suurin hyppy). Keskivirhe: jaannos-sd / sqrt(sum (t - keskiarvo)^2)."""
    t = np.asarray(t, float)
    o = np.argsort(t)
    t, a = t[o], np.asarray(a, float)[o]
    u, hyppy = pura_kulmat(a)
    p = np.polyfit(t, u, 1)
    jaannos = u - np.polyval(p, t)
    n = len(t)
    sd = float(np.sqrt(np.sum(jaannos ** 2) / max(n - 2, 1)))
    se = sd / float(np.sqrt(np.sum((t - t.mean()) ** 2)))
    return float(p[0]), se, sd, hyppy


def arvioi(valinta, merkinnat):
    """Heittokohtaiset tulokset ja yhteenveto. valinta = valinta.json:n luettelo, merkinnat = {id: asiakirja}."""
    heitot = {}
    for n in valinta:
        heitot.setdefault((n["nimi"], n["heitto"]), []).append(n)
    rivit = []
    for avain in sorted(heitot):
        ruudut = sorted(heitot[avain], key=lambda n: n["t_s"])
        viiv = [(n["t_s"], merkinnat[n["id"]]["kulma_deg"]) for n in ruudut if n["id"] in merkinnat and merkinnat[n["id"]]["tila"] in ("ok", "epavarma")]
        tilat = [merkinnat[n["id"]]["tila"] if n["id"] in merkinnat else None for n in ruudut]
        r = dict(nimi=avain[0], heitto=avain[1], vari=ruudut[0]["vari"], puoli=ruudut[0]["puoli"], ruutuja=len(ruudut), merkitty=sum(1 for x in tilat if x), viivallisia=len(viiv),
                 ei_nay=tilat.count("ei_nay"), kasi=tilat.count("kasi"), epavarma=tilat.count("epavarma"))
        try:
            r["kone_omega"] = float(ruudut[0]["kone"]["omega_deg_s"])
        except (KeyError, TypeError, ValueError):
            r["kone_omega"] = None
        if len(viiv) >= MIN_RUUTUJA and max(v[0] for v in viiv) - min(v[0] for v in viiv) >= MIN_VALI_S:
            om, se, sd, hyppy = sovita_omega([v[0] for v in viiv], [v[1] for v in viiv])
            r.update(omega=om, omega_se=se, jaannos_sd=sd, vali_s=max(v[0] for v in viiv) - min(v[0] for v in viiv), epaselva_purku=hyppy > HYPPY_RAJA)
            if r["kone_omega"] is not None:
                r["ero"] = r["kone_omega"] - om
                r["suunta_vertailtu"] = abs(om) > max(SUUNTA_RAJA, 3 * se)
                r["suunta_sama"] = bool(r["kone_omega"] * om > 0) if r["suunta_vertailtu"] else None
        rivit.append(r)
    arvioidut = [r for r in rivit if "omega" in r]
    vert = [r for r in arvioidut if r.get("suunta_vertailtu")]
    yht = dict(ruutuja=len(valinta), merkitty=sum(1 for n in valinta if n["id"] in merkinnat), heittoja=len(rivit), heittoja_omegalla=len(arvioidut), suunta_vertailtu=len(vert),
               suunta_sama=sum(1 for r in vert if r["suunta_sama"]), tilat={t: sum(1 for m in merkinnat.values() if m["tila"] == t) for t in ("ok", "epavarma", "ei_nay", "kasi")})
    ero = [abs(r["ero"]) for r in arvioidut if "ero" in r and not r["epaselva_purku"]]
    if ero:
        yht["keski_abs_ero"], yht["mediaani_abs_ero"] = float(np.mean(ero)), float(np.median(ero))
    kp = [(n["kasi_px"], merkinnat[n["id"]]["tila"]) for n in valinta if n["id"] in merkinnat and n.get("kasi_px") is not None]
    nakyva = [k for k, t in kp if t in ("ok", "epavarma")]
    piilossa = [k for k, t in kp if t in ("ei_nay", "kasi")]
    yht["vanha_suodatin"] = dict(viiva_mutta_hylattaisiin=sum(1 for k in nakyva if k > KASI_RAJA_PX), viivallisia=len(nakyva),
                                 ei_nay_mutta_hyvaksyttaisiin=sum(1 for k in piilossa if k <= KASI_RAJA_PX), ei_nakyvia=len(piilossa))
    return rivit, yht


def tulosta_raportti(rivit, yht, tulosta=print):
    tulosta(f"Ruutuja {yht['ruutuja']}, merkitty {yht['merkitty']} ({', '.join(f'{k} {v}' for k, v in yht['tilat'].items())}); heittoja {yht['heittoja']}, joista omega arvioitu {yht['heittoja_omegalla']}.")
    tulosta(f"{'heitto':<18}{'vari':<10}{'ruutuja':>8}{'viiva':>6}{'ei nay':>7}{'kasi':>5}{'omega merk.':>13}{'+-':>6}{'kone':>9}{'ero':>8}  suunta")
    for r in rivit:
        om = f"{r['omega']:+.1f}" if "omega" in r else "-"
        se = f"{r['omega_se']:.1f}" if "omega" in r else ""
        kone = f"{r['kone_omega']:+.1f}" if r["kone_omega"] is not None else "-"
        ero = f"{r['ero']:+.1f}" if "ero" in r else ""
        sm = {True: "sama", False: "ERI", None: "(pieni)"}.get(r.get("suunta_sama"), "") if "omega" in r else ""
        tulosta(f"{r['nimi'][:9] + ' h' + str(r['heitto']):<18}{r['vari']:<10}{r['ruutuja']:>8}{r['viivallisia']:>6}{r['ei_nay']:>7}{r['kasi']:>5}{om:>13}{se:>6}{kone:>9}{ero:>8}  {sm}{' (purku epaselva)' if r.get('epaselva_purku') else ''}")
    if yht["suunta_vertailtu"]:
        tulosta(f"Suunta: kone ja merkinta samaa mielta {yht['suunta_sama']} / {yht['suunta_vertailtu']} heitossa (|omega| > {SUUNTA_RAJA:.0f} aste/s ja yli 3 keskivirhetta).")
    if "keski_abs_ero" in yht:
        tulosta(f"Omegan ero koneen ja merkintojen valilla: keskimaarin {yht['keski_abs_ero']:.1f}, mediaani {yht['mediaani_abs_ero']:.1f} aste/s.")
    vs = yht["vanha_suodatin"]
    tulosta(f"Vanha kasi-suodatin (kasi_px > {KASI_RAJA_PX}): hylkaisi {vs['viiva_mutta_hylattaisiin']} / {vs['viivallisia']} ruudusta, joihin merkitsija sai viivan; "
            f"hyvaksyisi {vs['ei_nay_mutta_hyvaksyttaisiin']} / {vs['ei_nakyvia']} ruutua, joissa palkki ei nakynyt tai kasi oli kahvassa.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--valinta", required=True, help="luo_kahvamerkitsin.py:n valinta.json")
    ap.add_argument("--merkinnat", required=True, help="sivun JSON-vienti tai ArtifactData-listaus")
    ap.add_argument("--tulos", help="kirjoita heittokohtaiset tulokset ja yhteenveto JSON-tiedostoon")
    a = ap.parse_args()
    valinta = json.load(open(a.valinta, encoding="utf-8"))
    merkinnat = lue_merkinnat(a.merkinnat)
    rivit, yht = arvioi(valinta, merkinnat)
    tulosta_raportti(rivit, yht)
    if a.tulos:
        json.dump(dict(heitot=rivit, yhteenveto=yht), open(a.tulos, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
