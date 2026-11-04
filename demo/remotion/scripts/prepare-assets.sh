#!/bin/bash
# Everything the film shows except the voiceover: numbers, rendered clips, the report capture.
# usage (from demo/remotion): scripts/prepare-assets.sh [--force]   (needs ffmpeg, Playwright)
# --force re-renders the clips even when public/clips/ already has them.
set -euo pipefail
cd "$(dirname "$0")/.."
FILM=$PWD
ROOT=$(cd ../.. && pwd)
PUB=$FILM/public
mkdir -p "$PUB/clips" "$PUB/report"

echo "== numbers (from evidence/)"
(cd "$ROOT" && PYTHONPATH=src python3 demo/make_data.py > demo/data.json)
(cd "$ROOT" && PYTHONPATH=src python3 demo/remotion/make_film_data.py > "$PUB/film-data.json")

echo "== clips (demo/render.mjs)"
CLIPS="disclosure term_scan pipeline term_bug replay l1"
todo=""
for c in $CLIPS; do
  if [ "${1:-}" = "--force" ] || [ ! -s "$PUB/clips/$c.mp4" ]; then todo="$todo $c"; fi
done
if [ -n "$todo" ]; then
  node "$ROOT/demo/render.mjs" "$PUB/clips" $todo
  rm -f "$PUB"/clips/*.png
fi
python3 - "$PUB/clips" > "$PUB/clips.json" <<'EOF'
import json, subprocess, sys
from pathlib import Path
out = {}
for f in sorted(Path(sys.argv[1]).glob("*.mp4")):
    secs = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                           "-of", "csv=p=0", str(f)], capture_output=True, text=True).stdout
    out[f.stem] = round(float(secs), 3)
print(json.dumps(out, indent=1))
EOF

echo "== report capture"
SITE=$(mktemp -d)
(cd "$ROOT" && museport report evidence --out "$SITE" >/dev/null)
node scripts/capture-report.mjs "$SITE" "$PUB/report/overview.png"
rm -rf "$SITE"

# until scripts/prepare-voice.sh runs, empty voiceover files (the film then estimates each line)
[ -f "$PUB/vo-lengths.json" ] || echo '{}' > "$PUB/vo-lengths.json"
[ -f "$PUB/vo-words.json" ] || echo '{}' > "$PUB/vo-words.json"

echo "== manifest"
music=$(cd "$PUB" && ls music.wav music.mp3 music.m4a 2>/dev/null | head -1 || true)
python3 - "$music" "$PUB/music.json" > "$PUB/assets.json" <<'EOF'
import json, sys
from pathlib import Path
m = {"music": sys.argv[1] or None, "report": True}
extra = Path(sys.argv[2])  # optional: {"musicGain": 0.18, "musicCredit": "..."}
if m["music"] and extra.exists():
    m.update(json.loads(extra.read_text()))
print(json.dumps(m))
EOF
cat "$PUB/assets.json"
