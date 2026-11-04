#!/bin/bash
# Render the film to an MP4 for submission: 4K H.264 (CRF 12, preset slow), BT.709, AAC,
# +faststart.
# usage (from demo/remotion): scripts/render-film.sh out/museport-film.mp4
# Set CHROME=/path/to/chrome to use a system browser instead of Remotion's own download.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:-out/museport-film.mp4}
mkdir -p "$(dirname "$OUT")"
RAW="${OUT%.mp4}-raw.mp4"
# 4K (SCALE=2) from JPEG frames at quality 100 (PNG at 4K needs ~15 GB of temp space): YouTube gives 4K uploads a far higher bitrate, so the
# film stays sharp even when watched at 1080p. SCALE=1 for a quick 1080p preview.
SCALE=${SCALE:-2}
args=(render src/index.ts Film "$RAW" --codec=h264 --crf=12 --x264-preset=slow
      --color-space=bt709 --audio-codec=aac --audio-bitrate=320k --image-format=jpeg --jpeg-quality=100
      --scale="$SCALE" --concurrency="${CONCURRENCY:-2}"
      --offthreadvideo-cache-size-in-bytes=1500000000)  # 4K frames: keep under ~8 GB of RAM
[ -n "${CHROME:-}" ] && args+=(--browser-executable="$CHROME")
npx remotion "${args[@]}"
# the whole film to -16 LUFS (web level; the silences between lines pull the average down),
# and the index to the front so the file starts playing before it has fully loaded
ffmpeg -y -loglevel error -i "$RAW" -c:v copy -af loudnorm=I=-16:TP=-1.5:LRA=11 -ar 48000 \
  -c:a aac -b:a 320k -movflags +faststart "$OUT"
rm -f "$RAW"
ffprobe -v error -show_entries format=duration:stream=codec_name,width,height \
  -of default=nw=1 "$OUT"
echo "film: $OUT"
# the YouTube thumbnail beside it
npx remotion still src/index.ts Thumbnail "${OUT%.mp4}-thumbnail.png" --scale=1.5 ${CHROME:+--browser-executable="$CHROME"} --log=error
echo "thumbnail: ${OUT%.mp4}-thumbnail.png"
