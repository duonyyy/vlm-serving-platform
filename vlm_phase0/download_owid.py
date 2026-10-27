"""Download four fixed Our World in Data chart images for Phase 0 review."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

from PIL import Image

CHARTS = [
    ("life_expectancy", "life-expectancy", "country=~OWID_WRL&tab=line&time=1950..2023"),
    ("diabetes_bar", "diabetes-prevalence", "country=USA~GBR~JPN~VNM&tab=discrete-bar&time=2024"),
    ("population_growth", "annual-population-growth", "country=~OWID_WRL&tab=line&time=1950..2023"),
    ("vaccination_bar", "share-people-vaccinated-covid", "country=USA~GBR~JPN~VNM&tab=discrete-bar&time=2024-08-12"),
]
LICENSE_PAGE = "https://ourworldindata.org/faqs"


def download(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, slug, query in CHARTS:
        source_page = f"https://ourworldindata.org/grapher/{slug}?{query}"
        image_url = f"https://ourworldindata.org/grapher/{slug}.png?{query}&imWidth=1200&imDetails=1"
        target = out / f"{name}.png"
        if not target.exists():
            request = urllib.request.Request(image_url, headers={"User-Agent": "SE373-VLM-Phase0/1.0 educational image set"})
            with urllib.request.urlopen(request, timeout=45) as response:
                if response.headers.get_content_type() != "image/png":
                    raise ValueError(f"Unexpected content type for {slug}")
                target.write_bytes(response.read())
        with Image.open(target) as image:
            image.verify()
        rows.append({"name": name, "file": target.name, "source_page": source_page,
                     "image_url": image_url, "license": "CC BY 4.0 (OWID chart; verify third-party data terms)",
                     "license_page": LICENSE_PAGE, "artist": "Our World in Data"})
        print(name, target.stat().st_size)
    (out / "sources.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    download(args.out)
