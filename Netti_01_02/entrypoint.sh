#!/bin/sh
# Kontin kaynnistys: kayttaja PUID / PGID (Unraid: 99 / 100), tyokansiot, valinnainen yt-dlp-paivitys, sitten ajo ilman root-oikeuksia.
#   docker run IMAGE                  -> netti_palvelin.py (asetukset NETTI_*-ymparistomuuttujista)
#   docker run IMAGE --kiri 0         -> ylimaaraiset valitsimet netti_live.py:lle
#   docker run IMAGE testaa           -> ohjelman testit
#   docker run -it IMAGE bash         -> kuori (myos python / ffmpeg / ffprobe / yt-dlp / deno suoraan)
set -eu

PUID="${PUID:-99}"
PGID="${PGID:-100}"
UMASK="${UMASK:-022}"
DATA="${NETTI_DATA:-/data}"

if [ "$(id -u)" = "0" ]; then
    # kayttaja ja ryhma nimilla (esim. whoami, getpass); -o sallii PUID:n, joka on jo toisella kayttajalla (Unraid: 99 = nobody)
    getent group "$PGID" >/dev/null 2>&1 || groupadd -o -g "$PGID" netti >/dev/null 2>&1 || true
    getent passwd "$PUID" >/dev/null 2>&1 || useradd -o -u "$PUID" -g "$PGID" -d /tmp/home -s /usr/sbin/nologin -M netti >/dev/null 2>&1 || true
fi

mkdir -p "$DATA/ulos" "$DATA/video" "$DATA/kalibroinnit" /tmp/home /tmp/cache /tmp/deno /tmp/netti_tyo 2>/dev/null || true
if [ "$(id -u)" = "0" ]; then
    # vain kansiot itse (ei -R: iso jakokansio kayttaisi minuutteja); virhe ei estä kaynnistysta
    chown "$PUID:$PGID" "$DATA/ulos" "$DATA/video" "$DATA/kalibroinnit" /tmp/home /tmp/cache /tmp/deno /tmp/netti_tyo 2>/dev/null \
        || echo "[entrypoint] huom: kansioiden $DATA/{ulos,video,kalibroinnit} omistajaa ei voitu vaihtaa; tarkista jakokansion oikeudet (PUID=$PUID PGID=$PGID)" >&2
fi
umask "$UMASK"

if [ "${YTDLP_PAIVITA:-0}" = "1" ] && [ "$(id -u)" = "0" ]; then
    echo "[entrypoint] paivitetaan yt-dlp ..."
    pip install --no-cache-dir -q -U "yt-dlp[default]" 2>&1 | tail -n 3 || echo "[entrypoint] yt-dlp:n paivitys ei onnistunut, jatketaan kuvan versiolla" >&2
fi

kaynnista() {
    if [ "$(id -u)" = "0" ]; then
        exec setpriv --reuid="$PUID" --regid="$PGID" --clear-groups "$@"
    fi
    exec "$@"
}

cd /app
case "${1:-}" in
    testaa)
        shift
        cd /app/tools
        kaynnista python testaa_vakiot.py "$@"
        ;;
    bash|sh|python|python3|ffmpeg|ffprobe|yt-dlp|deno|pip)
        kaynnista "$@"
        ;;
    *)
        kaynnista python -u /app/netti_palvelin.py "$@"
        ;;
esac
