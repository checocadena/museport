#!/usr/bin/env python3
"""
Build the press kit: renders the still images in media/ to high-res JPEGs,
fills proposal.html with details.json, and prints the PDF with headless Chrome.

  python3 proposal/build.py

Same build approach as the Interminable Nothingness proposal kit; the look is Muse Port's own.
"""
import json
import os
import subprocess
import tempfile
from urllib.parse import quote

HERE = os.path.dirname(os.path.abspath(__file__))
CHROME = os.environ.get("CHROME", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")

with open(os.path.join(HERE, "details.json")) as f:
    D = json.load(f)
F = D["fields"]


def chrome(*args):
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--allow-file-access-from-files", *args],
                   check=True, capture_output=True)


# 1. Stills: each media/still-N.html is a 1920x1080 page, captured at 2x (3840x2160).
images = []
for n, src in enumerate(D["stills"], 1):
    name = f'{D["title"]} - {F["ARTIST"]} - {n}.jpg'
    with tempfile.TemporaryDirectory() as tmp:
        png = os.path.join(tmp, "still.png")
        chrome("--hide-scrollbars", "--window-size=1920,1080", "--force-device-scale-factor=2",
               f"--screenshot={png}", "file://" + quote(os.path.join(HERE, src)))
        subprocess.run(["sips", "-s", "format", "jpeg", "-s", "formatOptions", "92", png,
                        "--out", os.path.join(HERE, name)], check=True, capture_output=True)
    images.append(name)
    print("wrote", name)

# 2. PDF
page = open(os.path.join(HERE, "proposal.html")).read()
fills = dict(F, **{f"IMG{n}": quote(name) for n, name in enumerate(images, 1)})
for key, value in fills.items():
    page = page.replace("{{" + key + "}}", value)
assert "{{" not in page, "unfilled field in proposal.html: " + page[page.index("{{"):][:40]

with tempfile.NamedTemporaryFile("w", suffix=".html", dir=HERE, delete=False) as tmp:
    tmp.write(page)
out = os.path.join(HERE, D["pdf"])
try:
    chrome("--no-pdf-header-footer", f"--print-to-pdf={out}", "file://" + quote(tmp.name))
finally:
    os.unlink(tmp.name)
print("wrote", out)
