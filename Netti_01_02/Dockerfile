# Kivenseuranta (Netti_01_02): palvelinversio Dockerille / Unraidille. Vain perinteisia Dockerfile-ominaisuuksia (toimii myos ilman BuildKitia).
#   docker build -t netti-kivenseuranta:latest .            (ks. README_unraid.md)
#
# Pohja: Ubuntu 24.04 (Python 3.12, sama versio kuin kehityskoneella). Kolme vaihetta:
#   perus      ajonaikaiset paketit (Python, ffmpeg, OpenCV:n C++-kirjastot C++-sovittimelle)
#   rakennus   Python-riippuvuudet venviin + netti_fit (C++, B-sovituksen nopea toteutus) kaannetaan talle Pythonille
#   lopullinen perus + valmis venv + ohjelma
ARG BASE_IMAGE=ubuntu:24.04

FROM ${BASE_IMAGE} AS perus
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        python3 python3-venv ca-certificates tzdata tini ffmpeg libgomp1 \
        libopencv-core406t64 libopencv-imgproc406t64 \
 && rm -rf /var/lib/apt/lists/*

FROM perus AS rakennus
RUN apt-get update \
 && apt-get install -y --no-install-recommends g++ python3-dev libopencv-core-dev libopencv-imgproc-dev \
 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m venv /opt/venv \
 && /opt/venv/bin/pip install --no-cache-dir -r /tmp/requirements.txt pybind11
COPY cpp/netti_fit.cpp /tmp/cpp/netti_fit.cpp
# -O3 kuten CMakeLists.txt (ei -march=native: tulokset eivat saa riippua palvelimen prosessorista)
RUN cd /tmp/cpp \
 && g++ -O3 -std=c++17 -shared -fPIC \
        $(/opt/venv/bin/python -m pybind11 --includes) -I/usr/include/opencv4 \
        netti_fit.cpp \
        -o "netti_fit$(/opt/venv/bin/python -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')" \
        -lopencv_core -lopencv_imgproc -lpthread -ldl

FROM perus AS lopullinen
COPY --from=rakennus /opt/venv /opt/venv
COPY --from=rakennus /tmp/cpp/netti_fit*.so /app/
COPY . /app/
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    LANG=C.UTF-8 \
    HOME=/tmp/home \
    XDG_CACHE_HOME=/tmp/cache \
    DENO_DIR=/tmp/deno \
    NETTI_DATA=/data \
    NETTI_KALIBROINNIT=/data/kalibroinnit \
    TZ=Europe/Helsinki
WORKDIR /app
# CRLF -> LF (Windowsissa kopioitu skripti), oikeudet kaikille luettaviksi (ajo tapahtuu PUID:lla, ei rootina), tavukoodi valmiiksi (/app on ajossa vain luettava)
RUN sed -i 's/\r$//' /app/entrypoint.sh \
 && chmod +x /app/entrypoint.sh \
 && chmod -R a+rX /app \
 && python -m compileall -q /app \
 && python cpp/tarkista_moduuli.py netti_fit \
 && python -c "import cv2, numpy, scipy, yt_dlp, netti_fit; print('Python-paketit ok:', 'OpenCV', cv2.__version__, 'numpy', numpy.__version__, 'scipy', scipy.__version__, 'yt-dlp', yt_dlp.version.__version__)"

VOLUME ["/data"]
EXPOSE 8080
ENTRYPOINT ["/usr/bin/tini", "--", "/app/entrypoint.sh"]
