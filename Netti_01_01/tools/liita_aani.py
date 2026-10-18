"""Aanen liittaminen valmiiseen kuvatallenteeseen jalkikateen (esim. kun ajon lopun liitos ei onnistunut, ajo keskeytettiin ennen aanen latauksen valmistumista tai tallenne tehtiin --aani ei):

    python tools/liita_aani.py tallenne.mp4 --youtube TUNNUS             # YouTube: lataa aanen (yt_dlp) ja liittaa
    python tools/liita_aani.py tallenne.mp4 --aani tallenne.aani.m4a     # paikallinen aanitiedosto (ajon jattama ladattu aani) tai http-osoite
    python tools/liita_aani.py tallenne.mp4 --aani lahde.mp4 --alku 120  # lahdevideon oma aani; tallenteen ensimmainen ruutu = lahteen kohta 120 s

    python tools/liita_aani.py tallenne.mp4 --tarkista                    # ei muuta mitaan: tulostaa tiedoston kuva- ja aanirivit ja aanen todellisen keston ja voimakkuuden (dekoodaa aanen)

Aani tayteteen mykalla kuvan mittaiseksi eika kuvaa koskaan lyhenneta. Alkuperainen tallenne korvataan vasta, kun tulos on tarkistettu (kuva- ja aaniraita, kesto);
--ulos UUSI.mp4 jattaa alkuperaisen koskematta."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import netti_aani as na


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tallenne", help="valmis kuvatiedosto (mp4)")
    ap.add_argument("--youtube", help="YouTube-tunnus tai -osoite: aani ladataan yt_dlp:lla (aanitiedosto jaa tallenteen viereen <nimi>.aani.<paate>, kunnes liitos onnistuu)")
    ap.add_argument("--aani", help="aanilahde: paikallinen tiedosto (aani- tai videotiedosto) tai http(s)-osoite")
    ap.add_argument("--alku", type=float, default=0.0, help="aanilahteen kohta (s), joka vastaa tallenteen ensimmaista ruutua (oletus 0: sama aikajana kuin kuva)")
    ap.add_argument("--ulos", help="kirjoita tulos tahan tiedostoon (oletus: korvaa tallenne tarkistuksen jalkeen)")
    ap.add_argument("--ffmpeg", help="ffmpeg-ohjelman polku (oletus: PATH tai imageio-ffmpeg)")
    ap.add_argument("--tarkista", action="store_true", help="ei liitosta: tulosta tallenteen kuva- ja aanirivit sekä aanen todellinen kesto ja voimakkuus (poistumiskoodi 1, jos aanta ei ole / sita ei voi dekoodata)")
    a = ap.parse_args()
    if not os.path.exists(a.tallenne):
        raise SystemExit(f"tallennetta ei loydy: {a.tallenne}")
    exe = na.etsi_ffmpeg(a.ffmpeg)
    if exe is None:
        raise SystemExit("ffmpegia ei loydy (asenna ffmpeg tai pip install imageio-ffmpeg, tai anna --ffmpeg POLKU)")
    if a.tarkista:
        for rivi in na.virrat(exe, a.tallenne):
            print(rivi)
        t = na.aanen_tila(exe, a.tallenne)
        print("Aani (dekoodattu): " + (na.aani_teksti(t) + (" -> MYKKA" if t["keski_db"] <= -60.0 else "") if t else "tallenteessa ei ole aaniraitaa tai sita ei voi dekoodata"))
        raise SystemExit(0 if t else 1)
    if bool(a.youtube) == bool(a.aani):
        ap.error("anna tasan toinen: --youtube TUNNUS tai --aani POLKU (tai --tarkista)")
    if a.youtube or na.onko_youtube(a.aani or ""):
        lahde = na.Aanilahde("youtube", a.youtube or a.aani, a.alku)
        lahde.kaynnista(os.path.splitext(os.path.abspath(a.ulos or a.tallenne))[0] + ".aani", exe)
    else:
        lahde = na.Aanilahde("verkko" if a.aani.startswith(("http://", "https://")) else "tiedosto", a.aani, a.alku)
    ok = na.liita_tallenteeseen(exe, a.tallenne, lahde, ulos=a.ulos)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
