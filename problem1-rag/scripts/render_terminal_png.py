"""Render captured terminal output (a UTF-8 text file) as a terminal-style PNG.

Used to save the submission evidence (re-ingest and end-to-end query) as images. The text
is the real, unedited output of scripts/demo_reingest.ps1 and scripts/demo_query.ps1.

Usage:  python scripts/render_terminal_png.py <input.txt> <output.png> [title]
"""

import sys
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONT_CANDIDATES = ["C:/Windows/Fonts/consola.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"]
BG, FG, HEAD, BAR = (12, 12, 12), (204, 204, 204), (97, 214, 214), (40, 40, 40)


def main() -> None:
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    title = sys.argv[3] if len(sys.argv) > 3 else "Windows PowerShell"
    font_path = next(p for p in FONT_CANDIDATES if Path(p).exists())
    font = ImageFont.truetype(font_path, 15)
    width_chars = 130
    lines = []
    for raw in src.read_text(encoding="utf-8-sig").splitlines():
        lines += textwrap.wrap(raw, width_chars, replace_whitespace=False, drop_whitespace=False) or [""]
    cw, lh = font.getlength("M"), 20  # advance width, not glyph bbox, so long lines are not clipped
    width_px = int(cw * width_chars) + 40
    img = Image.new("RGB", (width_px, lh * len(lines) + 60), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, img.width, 28], fill=BAR)
    d.text((12, 6), title, font=font, fill=FG)
    for i, line in enumerate(lines):
        d.text((20, 40 + i * lh), line, font=font, fill=HEAD if line.startswith("===") else FG)
    dst.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst)
    print("wrote", dst)


if __name__ == "__main__":
    main()
