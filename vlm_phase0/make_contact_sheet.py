"""Make a labeled preview grid for candidate or final Phase 0 images."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


def make_sheet(folder: Path, output: Path) -> None:
    files = sorted(path for path in folder.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    if not files:
        raise ValueError(f"No images in {folder}")
    tile_w, tile_h, columns = 320, 270, 4
    rows = (len(files) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_w, rows * tile_h), "white")
    draw = ImageDraw.Draw(sheet)
    for index, path in enumerate(files):
        x, y = index % columns * tile_w, index // columns * tile_h
        with Image.open(path) as image:
            thumbnail = ImageOps.contain(image.convert("RGB"), (tile_w - 12, tile_h - 36))
            sheet.paste(thumbnail, (x + (tile_w - thumbnail.width) // 2, y + 4))
        draw.text((x + 6, y + tile_h - 27), path.name, fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    make_sheet(args.folder, args.output)
