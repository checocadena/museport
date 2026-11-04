#!/bin/bash
# Remove room echo and background noise from Cuecard's takes with ElevenLabs' Voice Isolator,
# keeping the real voice (no cloning, no re-voicing). Run it where your ElevenLabs key is.
#
# usage: ELEVENLABS_API_KEY=... scripts/isolate-takes.sh takes.zip
#   or, with the key in a .env file:  set -a; source /path/to/.env.local; set +a
#                                      scripts/isolate-takes.sh takes.zip
# Writes takes-isolated.zip next to takes.zip (01.mp3 ... 15.mp3). The key is never printed.
# Cost: ElevenLabs credits for the takes' total length (about 2.5 minutes here).
set -euo pipefail
IN=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
: "${ELEVENLABS_API_KEY:?set ELEVENLABS_API_KEY (see usage at the top of this script)}"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
unzip -q "$IN" -d "$WORK/in"
mkdir "$WORK/out"
for f in "$WORK"/in/*; do
  n=$(basename "${f%.*}")
  curl -sS --fail-with-body -X POST https://api.elevenlabs.io/v1/audio-isolation \
    -H "xi-api-key: $ELEVENLABS_API_KEY" -F "audio=@$f" -o "$WORK/out/$n.mp3"
  echo "$n.mp3"
done
(cd "$WORK/out" && zip -q "$WORK/isolated.zip" ./*.mp3)
OUT="$(dirname "$IN")/takes-isolated.zip"
mv "$WORK/isolated.zip" "$OUT"
echo "wrote $OUT"
