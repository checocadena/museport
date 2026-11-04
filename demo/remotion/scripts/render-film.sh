#!/bin/bash
# Render the film to an MP4 for submission: H.264 (CRF 17, preset slow), BT.709, AAC 256k,
# +faststart.
# usage (from demo/remotion): scripts/render-film.sh out/museport-film.mp4
# Set CHROME=/path/to/chrome to use a system browser instead of Remotion's own download.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:-out/museport-film.mp4}
mkdir -p "$(dirname "$OUT")"
RAW="${OUT%.mp4}-raw.mp4"
args=(render src/index.ts Film "$RAW" --codec=h264 --crf=17 --x264-preset=slow
      --color-space=bt709 --audio-codec=aac --audio-bitrate=256k --image-format=jpeg
      --jpeg-quality=95)
[ -n "${CHROME:-}" ] && args+=(--browser-executable="$CHROME")
npx remotion "${args[@]}"
# the whole film to -16 LUFS (web level; the silences between lines pull the average down),
# and the index to the front so the file starts playing before it has fully loaded
ffmpeg -y -loglevel error -i "$RAW" -c:v copy -af loudnorm=I=-16:TP=-1.5:LRA=11 -ar 48000 \
  -c:a aac -b:a 256k -movflags +faststart "$OUT"
rm -f "$RAW"
ffprobe -v error -show_entries format=duration:stream=codec_name,width,height \
  -of default=nw=1 "$OUT"
echo "film: $OUT"
# the YouTube thumbnail beside it
npx remotion still src/index.ts Thumbnail "${OUT%.mp4}-thumbnail.png" ${CHROME:+--browser-executable="$CHROME"} --log=error
echo "thumbnail: ${OUT%.mp4}-thumbnail.png"
