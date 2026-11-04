#!/bin/bash
# Voiceover: clean each take, measure it, and (optionally) get Whisper's word timings.
# usage (from demo/remotion): scripts/prepare-voice.sh
#
# Put the best take of each line in public/vo-raw/ as NN.<ext> (01.m4a, 02.m4a, ...; NN is the
# line's "n" in public/lines.json). Phone recordings (.m4a) are fine.
#
# Writes public/vo/vo-NN.wav, public/vo-lengths.json and, with Whisper, public/vo-words.json.
# Whisper, either:
#   WHISPER_CLI=whisper-cli WHISPER_MODEL=/path/ggml-small-q5_1.bin   (local), or
#   WHISPER_SSH=user@host WHISPER_MODEL=/opt/whisper/ggml-small-q5_1.bin (a machine that has it)
# Without either, captions share each line's time by word count.
set -euo pipefail
cd "$(dirname "$0")/.."
PUB=$PWD/public
mkdir -p "$PUB/vo"
shopt -s nullglob

# The BALI chain: left channel to mono, high-pass, denoise, gentle compression, trim silence at
# both ends, loudness to -16 LUFS, 48 kHz mono.
# NO_DENOISE=1 skips the denoiser, for takes already cleaned (scripts/isolate-takes.sh).
DENOISE=",afftdn=nf=-60"; [ -n "${NO_DENOISE:-}" ] && DENOISE=""
CHAIN="pan=mono|c0=c0,highpass=f=80${DENOISE},acompressor=threshold=-20dB:ratio=2.5:attack=10:release=200,silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.05,areverse,silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.08,areverse,loudnorm=I=-16:TP=-1.5:LRA=9"

for raw in "$PUB"/vo-raw/*; do
  base=$(basename "$raw"); n=${base%%.*}; n=$(printf '%02d' "$((10#$n))")
  ffmpeg -y -loglevel error -i "$raw" -af "$CHAIN" -ar 48000 -ac 1 "$PUB/vo/vo-$n.wav"
  echo "vo-$n.wav"
done

python3 - "$PUB/vo" > "$PUB/vo-lengths.json" <<'EOF'
import json, subprocess, sys
from pathlib import Path
out = {}
for f in sorted(Path(sys.argv[1]).glob("vo-*.wav")):
    secs = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                           "-of", "csv=p=0", str(f)], capture_output=True, text=True).stdout
    out[str(int(f.stem.split("-")[1]))] = round(float(secs), 3)
print(json.dumps(out, indent=1))
EOF
echo "lengths: $(python3 -c "import json;d=json.load(open('$PUB/vo-lengths.json'));print(len(d),'takes,',round(sum(d.values()),1),'s spoken')")"

if [ -z "${WHISPER_CLI:-}" ] && [ -z "${WHISPER_SSH:-}" ]; then
  echo "no Whisper configured: captions will share each line's time by word count"
  exit 0
fi
TMP=$(mktemp -d)
for wav in "$PUB"/vo/vo-*.wav; do
  b=$(basename "$wav" .wav)
  # whisper.cpp wants 16 kHz mono; -ml 1 gives one word per segment, -sow splits on words
  ffmpeg -y -loglevel error -i "$wav" -ar 16000 -ac 1 "$TMP/$b.wav"
  if [ -n "${WHISPER_SSH:-}" ]; then
    scp -q "$TMP/$b.wav" "$WHISPER_SSH:/tmp/$b.wav"
    ssh "$WHISPER_SSH" "whisper-cli -m $WHISPER_MODEL -ml 1 -sow -oj -of /tmp/$b -f /tmp/$b.wav >/dev/null 2>&1"
    scp -q "$WHISPER_SSH:/tmp/$b.json" "$TMP/$b.json"
    ssh "$WHISPER_SSH" "rm -f /tmp/$b.wav /tmp/$b.json"
  else
    "$WHISPER_CLI" -m "$WHISPER_MODEL" -ml 1 -sow -oj -of "$TMP/$b" -f "$TMP/$b.wav" >/dev/null 2>&1
  fi
done
python3 - "$TMP" > "$PUB/vo-words.json" <<'EOF'
import json, sys
from pathlib import Path
out = {}
for f in sorted(Path(sys.argv[1]).glob("vo-*.json")):
    segs = json.loads(f.read_text()).get("transcription", [])
    words = [{"w": s["text"].strip(), "t0": s["offsets"]["from"] / 1000,
              "t1": s["offsets"]["to"] / 1000} for s in segs if s["text"].strip()]
    out[str(int(f.stem.split("-")[1]))] = words
print(json.dumps(out))
EOF
rm -rf "$TMP"
echo "word timings: public/vo-words.json"
