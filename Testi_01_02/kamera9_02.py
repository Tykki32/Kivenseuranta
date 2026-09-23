# ============================================================
# kamera9_02.py - KIVEN PAIKANNUS MALLIPOHJAISELLA HAULLA (EI VARIN/
# MUODON SEGMENTOINNILLA)
#
# kamera9_01.py:n find_stone_candidates tunnistaa kiven segmentoimalla
# graniitin (varisaturaatio+paikallinen tummuus) ja suodattamalla
# muodolla - toimii, mutta on herkka virhekandidaateille (pyyhkija,
# tekstit, valaistus) ja vaatii jatkuvuuspohjaisen seurannan niiden
# poissulkemiseksi (katso kamera9_01.py:n track_stone_in_video).
#
# Tama tiedosto (kayttajan pyynnosta, MERKITTAVA lisays - siksi oma
# tiedostonsa, kamera9_01.py jatetaan koskemattomaksi) kaantaa
# lahestymistavan PAINVASTAISEKSI: kamera9_01.py:n jo sovitetusta
# kovakoodattu+hienosaadetusta 3D-mallista (fit_stone_profile) TEHDAAN
# oikea 3D-malli, ja sita YRITETAAN SIJOITTAA jaalle eri (X,Y)-
# pisteisiin - jokaisessa pisteessa ennustetaan mika kiven siluetti
# NAYTTAISI silta kohdalta (kamera9_01.py:n _predicted_stone_hull,
# joka jo kayttaa TAYTTA 3D-kameramallia - K,R,t - eika arvaa mitaan
# etaisyydesta/kulmasta), ja verrataan sita HAVAITTUUN graniittimaskiin
# (kamera9_01.py:n create_granite_mask). Piste jossa ennuste osuu
# parhaiten havaittuun maskiin ON kivi.
#
# Tama on kayttajan pyytama HAKU: "yrittaa sijoittaa sita jaalla eri
# pisteisiin jotta loytaa kiven" - siis KARKEA->HIENO ristikkohaku
# (sama periaate kuin kamera8_01.py:n search_far_house, katso sen
# kommentti), ei minkaanlaista segmentointia. Etuna: koska malli
# sijoitetaan SUORAAN kiven OMAAN pystyakseliin (X0,Y0) - ei kiven
# NAKYVAN ellipsin keskipisteeseen - loydetty (X0,Y0) ON JO Z-
# KORJATTU (jaataso-keskikohta), ilman erillista puolikorkeus-
# approksimaatiota tai sovitusta (katso kamera9_01.py:n kommentit
# aiemmista, epatarkemmista yrityksista).
#
# KAKSI VAIHETTA:
#   1) HAKU: kunnes kivi loytyy, koeta paikantaa se JOKA 10. FRAMESSA
#      rajatulta alueelta (kauempi paa: keskiviivasta +-50cm,
#      kaukaisen hoglinen ja pesan takarajan valilla - tyypillinen
#      alue jonne heitetty kivi paatyy).
#   2) SEURANTA: kun kivi loytyy, sen sijaintia paivitetaan JOKA
#      FRAMESSA (myos haku-alueen ULKOPUOLELLA) pienella paikallisella
#      haulla edellisen sijainnin ymparilta - halvempi kuin koko
#      alueen haku, koska jatkuvuus antaa jo vahvan alkuarvauksen.
# ============================================================

import os
import csv
import math
import time
import importlib.util

import numpy as np
import cv2


def _load_kamera9_01():
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        "kamera9_01", os.path.join(here, "kamera9_01.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


k9 = _load_kamera9_01()
k8 = k9.k8


# ============================================================
# HAKUALUE ("kauempi paa, keskiviivasta +-50cm, hogline-takaraja")
#
# "Takaraja" = pesan takareuna (back line) - tangentti 12-jalkaisen
# renkaan taakse, HOUSE_RADIUS_CM paan pesan keskipisteesta poispain
# hoglinesta. Kaukainen paa (FAR_*) koska talla videolla kivi heitetaan
# lahempaa hoglinea kohti sita - katso kamera9_01.py:n
# track_stone_in_video-kommentti samasta videosta.
# ============================================================

SEARCH_X_HALF_WIDTH_CM = 50.0
SEARCH_Y_MIN_CM = k8.FAR_HOGLINE_Y_CM
SEARCH_Y_MAX_CM = k8.FAR_HOUSE_Y_CM + k8.HOUSE_RADIUS_CM

# Karkea->hieno ristikkohaku (sama periaate kuin kamera8_01.py:n
# search_far_house) - kaksi tasoa riittaa, koska kolmas (ultra-hieno)
# taso ei enaa muuta lopputulosta merkittavasti mutta maksaa yhta
# paljon kuin ensimmainen.
SEARCH_COARSE_STEP_CM = 10.0
SEARCH_FINE_STEP_CM = 2.0
SEARCH_SCORE_THRESHOLD = 0.55  # peitto-osuus jolla "loytyi" hyvaksytaan

TRACK_HALF_RANGE_CM = 35.0
TRACK_COARSE_STEP_CM = 7.0
TRACK_FINE_STEP_CM = 1.5
TRACK_SCORE_THRESHOLD = 0.35  # matalampi - jatkuvuus on jo vahva prior
TRACK_LOST_MAX_MISSES = 5     # montako peräkkäistä huonoa framea ennen kuin palataan hakuun

# Pienempi resoluutio hylylle HAUSSA/SEURANNASSA (nopeampi, riittava
# karkean sijainnin loytamiseen - ei tarvita samaa tarkkuutta kuin
# esim. profiilin sovituksessa).
SEARCH_HULL_N_THETA = 14
SEARCH_HULL_N_PER_SEGMENT = 2
