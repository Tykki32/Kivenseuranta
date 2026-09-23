# ============================================================
# kamera9_03.py - ALLE PIKSELIN TARKKUUS: KOKO 3D-KAPPALEEN JA
# GRANIITTI/MUOVI-RAJAN YHTEISSOVITUS
#
# kamera9_02.py paikantaa kiven vertaamalla ennustettua KOKO siluettia
# (kupera peite) HAVAITTUUN BINAARISEEN graniittimaskiin karkea->hieno
# ristikkohaulla - tarkkuus on rajattu ristikon askelvaliin (kamera9_02
# .py:ssa n. 1.5cm) JA itse binaarimaskin pikselitason kynnysarvoon
# (yksi pikseli on joko "graniittia" tai ei - ei valipikselitarkkuutta).
#
# Tama tiedosto (kayttajan pyynnosta, MERKITTAVA lisays - kamera9_02.py
# jatetaan koskemattomaksi) parantaa Y-tarkkuutta OLENNAISESTI YHDIS-
# TAMALLA KAKSI RIIPPUMATONTA HAVAINTOLAHDETTA SAMAAN SOVITUKSEEN (EI
# kahta erillista lukua - kayttajan pyynnosta VAIN yksi paras yhteinen
# sijainti per frame):
#
#   1) KOKO GRANIITTISILUETTI (kamera9_01.py:n taysi 3D-profiili)
#      sovitettuna kiven ULKOreunaan jaata vasten - sama periaate kuin
#      kamera9_01.py:n locate_stone_from_profile, mutta TARKEA LISAYS:
#      kirkas kahva PEITTAA osan graniitista talta (ylhaaltapain-)
#      kuvakulmalta, joten graniittimaskin kontuurin osa seuraa
#      TODELLISUUDESSA KAHVAN reunaa, ei kiven reunaa - nama pisteet
#      SUODATETAAN POIS (_filter_ice_boundary_points, katso sen
#      kommentti) ETTEIVAT ne vinouta sovitusta.
#
#   2) GRANIITTI/MUOVI-RAJAN alipikselihavainnot (katso alla) - TERAVA
#      reuna (varisaturaation hyppy), toisin kuin kiven ULKOreuna
#      jaata vasten joka on usein PEHMEA/epaselva (valaistus, varjot,
#      jaan oma tekstuuri) - antaa ALIPIKSELITARKAN lisatiedon.
#
# Rajan alipikseli-ilmaisu: jokaiselta kulmalta (n. 70 kpl) skannataan
# saturaatioarvo BILINEAARISESTI interpoloituna (ei vain lahin pikseli)
# ja kynnysarvon ylitys ratkaistaan LINEAARISELLA INTERPOLOINNILLA
# kahden naytteen valilla - standarditekniikka joka antaa tyypillisesti
# < 0.1 pikselin tarkkuuden reunan sijainnille.
#
# YHTEISSOVITUS: molempien lahteiden jaannosvirheet (etaisyys ennus-
# tettuun malliin - koko rungon KUPERA PEITE JA renkaan ympyra, samalla
# (X,Y)-akselilla) yhdistetaan SAMAAN LM-sovitukseen [X,Y,rengas_sade] -
# yksi paras yhteinen vastaus, ei kahta erillista. Molemmat lahteet
# ovat AINA mukana kun saatavilla - runko-osuus toimii jo pelkastaan
# (myos hyvin kaukana, katso kamera9_01.py:n "kauimpana havaittu kivi"),
# reunaosuus antaa lisatarkkuutta silloin kun rengas on riittavan
# suuri (lahella kameraa) resolvoitavaksi.
#
# HUOM (kayttajan pyynnosta): TATA TIEDOSTOA EI OLE tarkoitus suodattaa/
# tasoittaa frame-framelta - jokainen frame ratkaistaan itsenaisesti
# (ei liukuvaa keskiarvoa tms.), jotta jaljelle jaava frame-framelta-
# kohina kertoo kayttajalle rehellisesti kuinka tarkasti mittaus
# todellisuudessa onnistuu (kayttaja tekee oman suodatuksensa erikseen).
# ============================================================

import os
import csv
import math
import time
import importlib.util

import numpy as np
import cv2


def _load_kamera9_02():
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        "kamera9_02", os.path.join(here, "kamera9_02.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


k92 = _load_kamera9_02()
k9 = k92.k9
k8 = k92.k8
