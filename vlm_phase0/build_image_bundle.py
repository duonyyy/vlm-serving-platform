"""Freeze 20 reviewed public images and provenance into a Kaggle-ready ZIP."""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

from PIL import Image

from .core import sha256, validate_manifest

ROOT = Path(__file__).parent
SELECTION = [
    ("scene", "commons_candidates", "scene_01.jpg", "Vật nổi bật trên sườn đồi là gì?", "Một cây đứng đơn độc trên sườn đồi cỏ."),
    ("scene", "commons_candidates", "scene_02.jpg", "Người trong ảnh đang đứng ở đâu?", "Một người đứng trên đỉnh đá cao, phía sau là núi tuyết."),
    ("scene", "commons_candidates", "scene_03.jpg", "Phương tiện màu đỏ trong ảnh là gì?", "Một cabin cáp treo màu đỏ đi trên dây cáp phía trên khu đồi có nhà."),
    ("scene", "commons_candidates", "scene_04.jpg", "Những con chim đang đứng ở môi trường nào?", "Một đàn chim chân dài đứng trong vùng nước nông."),
    ("scene", "commons_candidates", "scene_06.jpg", "Bầu trời và ánh sáng trong ảnh gợi thời điểm nào trong ngày?", "Mặt trời thấp gần đường chân trời, ánh sáng cam của hoàng hôn."),
    ("document", "commons_candidates", "document_01.jpg", "Trang này là chữ viết tay hay chữ in?", "Trang giấy cũ có nhiều dòng chữ viết tay mờ."),
    ("document", "commons_candidates", "document_02.jpg", "Bố cục phần nội dung chính của trang gồm mấy cột?", "Trang in cũ có hai cột chữ dày đặc."),
    ("document", "commons_candidates", "document_03.jpg", "Phía trên trang tài liệu có biểu tượng hay hình gì?", "Một quốc huy hoặc biểu trưng nhỏ đặt ở giữa phần đầu trang."),
    ("document", "commons_candidates", "document_04.jpg", "Đây là bản vẽ mặt bằng hay một trang văn xuôi?", "Bản vẽ kỹ thuật với các đường thẳng và ô/phòng, không phải văn xuôi."),
    ("document", "commons_candidates", "document_05.jpg", "Tài liệu có phần tiêu đề và chữ ký ở cuối trang không?", "Một công văn đánh máy với tiêu đề/đầu thư và chữ ký ở phần dưới."),
    ("text", "commons_text_candidates", "text_01.jpg", "Đọc hai từ lớn trên biển màu vàng.", "Biển ghi EAT LOCALS."),
    ("text", "commons_text_candidates", "text_02.jpg", "Biển xanh ghi tên thị trấn và độ cao bao nhiêu feet?", "FAIRPLAY; ELEV 9953 FT."),
    ("text", "commons_text_candidates", "text_03.jpg", "Đọc tên trên biển ngôi sao lớn.", "Biển lớn ghi ROY'S; phía dưới có MOTEL và CAFE."),
    ("text", "commons_text_candidates", "text_05.jpg", "Đọc hai từ đầu tiên trên tấm áp phích chữ Cyrillic.", "Hai từ đầu là БЕЛГОРОД và ТЮМЕНЬ."),
    ("text", "commons_candidates", "scene_05.jpg", "Đọc câu trích dẫn in dưới bức ảnh đen trắng.", "THE EAST WIND WAS WHISTLING OUT OF THE BAY."),
    ("chart", "commons_chart_candidates", "chart_01.png", "Đường màu xanh thay đổi thế nào ở đoạn cuối biểu đồ?", "Đường xanh giảm mạnh gần cuối, xuống khoảng 70%."),
    ("chart", "owid_candidates", "diabetes_bar.png", "Quốc gia nào có thanh dài nhất?", "United States có thanh dài nhất trong bốn quốc gia."),
    ("chart", "owid_candidates", "life_expectancy.png", "Xu hướng chung của đường biểu diễn từ trái sang phải là gì?", "Tuổi thọ trung bình toàn cầu tăng dài hạn, dù có vài đoạn giảm ngắn."),
    ("chart", "owid_candidates", "population_growth.png", "Giá trị ở cuối chuỗi cao hay thấp hơn mức đỉnh?", "Giá trị cuối chuỗi thấp hơn mức đỉnh trước đó."),
    ("chart", "owid_candidates", "vaccination_bar.png", "Quốc gia nào có tỷ lệ cao nhất trên biểu đồ?", "Vietnam là quốc gia có thanh dài nhất, khoảng 90.8%."),
]


def build() -> None:
    bundle = ROOT / "kaggle_bundle"
    images = bundle / "images"
    images.mkdir(parents=True, exist_ok=True)
    metadata = {}
    for folder in ("commons_candidates", "commons_text_candidates", "commons_chart_candidates"):
        path = ROOT / folder / "candidates.json"
        for row in json.loads(path.read_text(encoding="utf-8")):
            metadata[(folder, row["preview_file"])] = row
    for row in json.loads((ROOT / "owid_candidates" / "sources.json").read_text(encoding="utf-8")):
        metadata[("owid_candidates", row["file"])] = row
    manifest, provenance = [], []
    counts: dict[str, int] = {}
    for category, folder, name, prompt, expected in SELECTION:
        counts[category] = counts.get(category, 0) + 1
        sample_id = f"{category}_{counts[category]:03d}"
        source = ROOT / folder / name
        if not source.is_file():
            raise FileNotFoundError(source)
        target = images / f"{sample_id}{source.suffix.lower()}"
        shutil.copyfile(source, target)
        with Image.open(target) as image:
            width, height = image.size
            image.verify()
        details = metadata[(folder, name)]
        if folder == "owid_candidates":
            page = details["source_page"]
            image_url = details["image_url"]
            license_name = details["license"]
            license_url = details["license_page"]
            artist = details["artist"]
        else:
            page = details["page_url"]
            image_url = details["image_url"]
            license_name = details["license"]
            license_url = details["license_url"]
            artist = details["artist"] or details["credit"]
        if not page or not license_name:
            raise ValueError(f"Missing provenance for {sample_id}")
        manifest.append({"sample_id": sample_id, "image_path": f"images/{target.name}",
                         "category": category, "prompt": prompt,
                         "expected_observation": expected})
        provenance.append({"sample_id": sample_id, "source_page": page, "image_url": image_url,
                           "license": license_name, "license_url": license_url,
                           "artist": artist, "downloaded_from": name,
                           "sha256": sha256(target), "width": width, "height": height,
                           "bytes": target.stat().st_size})
    manifest_path = bundle / "manifest.jsonl"
    manifest_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in manifest) + "\n", encoding="utf-8")
    (bundle / "sources.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    credits = ["# Sources and licenses", "", "The source page controls the file's license. Keep these credits with any shared copy of the images.", "",
               "| Sample | File page | License | Author / credit |", "|---|---|---|---|"]
    for item in provenance:
        author = (item["artist"] or "See file page").replace("|", "\\|").replace("\n", " ")
        license_label = item["license"].replace("|", "\\|")
        license_cell = f"[{license_label}]({item['license_url']})" if item["license_url"] else license_label
        credits.append(f"| {item['sample_id']} | [Commons/OWID page]({item['source_page']}) | {license_cell} | {author} |")
    (bundle / "SOURCES.md").write_text("\n".join(credits) + "\n", encoding="utf-8")
    for name in ("__init__.py", "core.py", "qwen.py", "run_phase0.py"):
        target = bundle / "vlm_phase0" / name
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    validate_manifest(manifest_path, bundle)
    archive = ROOT / "phase0_bundle.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        for path in sorted(bundle.rglob("*")):
            if path.is_file():
                zip_file.write(path, path.relative_to(bundle))
    print(f"Built {len(manifest)} images: {counts}; ZIP: {archive} ({archive.stat().st_size} bytes)")


if __name__ == "__main__":
    build()
