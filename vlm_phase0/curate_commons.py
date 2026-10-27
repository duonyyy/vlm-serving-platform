"""Find and download candidate Phase 0 images from Wikimedia Commons.

Run `collect` first, visually review contact sheets, then freeze a selected
set. Category membership is discovery only; licenses are checked per file.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image

API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "SE373-VLM-Phase0/1.0 (educational evaluation; contact: Commons file pages)"
CATEGORIES = {
    "scene": ["Landscape photography"],
    "document": ["Scanned texts", "Documents", "Scanned documents"],
    "text": ["Road signs with text", "Valued images of signs", "Photographs of signage"],
    "chart": ["Line charts", "Bar charts", "Pie charts"],
}
ALLOWED_LICENSE_PREFIXES = ("CC BY", "CC0", "Public domain", "PD-")


def request_bytes(url: str, retries: int = 4) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt == retries - 1:
                raise
            time.sleep(10 * (attempt + 1))
    raise AssertionError("unreachable")


def plain(value: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", " ", value or "")).strip()


def category_pages(category: str, limit: int = 80) -> list[dict]:
    query = urllib.parse.urlencode({
        "action": "query", "format": "json", "generator": "categorymembers",
        "gcmtitle": f"Category:{category}", "gcmtype": "file", "gcmlimit": limit,
        "prop": "imageinfo", "iiprop": "url|size|mime|extmetadata", "iiurlwidth": 800,
    })
    payload = json.loads(request_bytes(f"{API}?{query}"))
    return list(payload.get("query", {}).get("pages", {}).values())


def candidates_for(category: str, group: str, limit: int = 80) -> list[dict]:
    results = []
    for page in category_pages(category, limit):
        info = (page.get("imageinfo") or [{}])[0]
        metadata = info.get("extmetadata") or {}
        def meta(name: str) -> str:
            return plain(metadata.get(name, {}).get("value", ""))
        license_name = meta("LicenseShortName")
        if not license_name.startswith(ALLOWED_LICENSE_PREFIXES):
            continue
        mime = info.get("mime", "")
        if mime not in {"image/jpeg", "image/png", "image/webp", "image/svg+xml"}:
            continue
        original_is_manageable = (mime in {"image/jpeg", "image/png", "image/webp"}
                                  and info.get("size", 10**9) <= 4_000_000
                                  and info.get("width", 10**9) * info.get("height", 10**9) <= 20_000_000)
        url = info.get("url") if original_is_manageable else (info.get("thumburl") or info.get("url"))
        if not url:
            continue
        results.append({
            "group": group, "category": category, "title": page["title"],
            "page_url": info.get("descriptionurl"), "image_url": url,
            "license": license_name, "license_url": meta("LicenseUrl"),
            "artist": meta("Artist"), "credit": meta("Credit"),
            "mime": mime, "original_width": info.get("width"),
            "original_height": info.get("height"),
        })
    return results


def collect(out: Path, per_group: int, groups: list[str], metadata_only: bool = False,
            indices: list[int] | None = None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for group, categories in CATEGORIES.items():
        if group not in groups:
            continue
        found = []
        titles = set()
        for category in categories:
            for candidate in candidates_for(category, group):
                if candidate["title"] not in titles:
                    titles.add(candidate["title"])
                    found.append(candidate)
        # Keep a bounded number of candidates, with representation from each category.
        picked = []
        for category in categories:
            for candidate in found:
                if candidate["category"] == category and candidate not in picked:
                    picked.append(candidate)
                    if len([x for x in picked if x["category"] == category]) >= max(2, per_group // len(categories)):
                        break
        for candidate in found:
            if len(picked) >= per_group:
                break
            if candidate not in picked:
                picked.append(candidate)
        for index, candidate in enumerate(picked[:per_group], 1):
            suffix = ".png" if candidate["image_url"].lower().split("?")[0].endswith(".png") or candidate["mime"] == "image/svg+xml" else ".jpg"
            filename = f"{group}_{index:02d}{suffix}"
            target = out / filename
            candidate["preview_file"] = filename
            all_rows.append(candidate)
    (out / "candidates.json").write_text(json.dumps(all_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if metadata_only:
        return
    for candidate in all_rows:
        index = int(candidate["preview_file"].split("_")[-1].split(".")[0])
        if indices and index not in indices:
            continue
        target = out / candidate["preview_file"]
        if not target.exists():
            try:
                target.write_bytes(request_bytes(candidate["image_url"], retries=1))
            except urllib.error.HTTPError as exc:
                if exc.code == 429:
                    print("RATE_LIMITED", target.name, candidate["title"])
                    time.sleep(10)
                    continue
                raise
            time.sleep(4)
        with Image.open(target) as image:
            image.verify()
        print(candidate["group"], target.name, candidate["title"], candidate["license"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--per-group", type=int, default=9)
    parser.add_argument("--groups", nargs="+", choices=sorted(CATEGORIES), default=list(CATEGORIES))
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--indices", nargs="+", type=int)
    args = parser.parse_args()
    collect(args.out, args.per_group, args.groups, args.metadata_only, args.indices)


if __name__ == "__main__":
    main()
