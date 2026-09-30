# Kivien omat koot (MAH00014, 24 heittoa live-vaiheessa)

`kivien_koot_MAH00014.csv`: jokaiselle heittoportin lapaisseelle kivelle oma 3D-profiilin sovitus (`tools/stone_sizes.py`):
R_cm (raaka ja robusti), H_cm, kahvan lovi, sovitus-rms, poistettujen havaintojen maara, live-seurannan rms-mediaani.
Robusti = havainnot joiden oma rms > max(2 x mediaani, 2.5 px) poistetaan ja sovitetaan uudelleen (kontuuriin liittyva pelaaja/harja/kohina).
