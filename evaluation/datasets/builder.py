"""Builds the 50-sample curated Vietnamese Benchmark Dataset and images."""

from __future__ import annotations

import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def create_sample_image(
    path: Path,
    text_lines: list[str],
    bg_color: str = "#FFFFFF",
    text_color: str = "#000000",
    size: tuple[int, int] = (640, 480),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, color=bg_color)
    draw = ImageDraw.Draw(img)

    # Simple text rendering
    y = 30
    for line in text_lines:
        draw.text((30, y), line, fill=text_color)
        y += 28

    img.save(path, format="JPEG", quality=90)


def build_benchmark_dataset(dest_dir: Path) -> Path:
    """Generate 50 benchmark cases: 20 invoices, 15 OCR text, 15 VQA."""
    images_dir = dest_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    samples: list[dict] = []

    # 1. 20 Invoices
    sellers = [
        "Công ty Cổ phần Bách Hóa Xanh",
        "Tập đoàn Bán lẻ WinCommerce",
        "Nhà hàng Phở Thìn Lò Đúc",
        "Công ty TNHH Cà Phê Trung Nguyên",
        "Siêu thị Điện Máy Xanh",
        "Nhà thuốc Long Châu",
        "Cửa hàng Tiện lợi Circle K",
        "Khách sạn Mường Thanh Grand",
        "Công ty Xăng Dầu Petrolimex",
        "Công ty Cổ phần Fahasa",
        "Nhà hàng Pizza 4P's",
        "Trung tâm Anh ngữ VUS",
        "Công ty May Nhà Bè",
        "Công ty Cổ phần Viễn thông FPT",
        "Bệnh viện Đa khoa Tâm Anh",
        "Công ty Sữa Vinamilk",
        "Cửa hàng Highlands Coffee",
        "Công ty Cổ phần Vàng Bạc Đá Quý PNJ",
        "Công ty Giao Hàng Tiết Kiệm",
        "Chuỗi Siêu thị Co.opmart",
    ]

    for i, seller in enumerate(sellers, 1):
        sample_id = f"inv_{i:03d}"
        inv_no = f"HD-2026-{1000 + i * 37}"
        inv_date = f"2026-10-{10 + (i % 15):02d}"
        amount = 50000.0 * (i * 2 + 3)
        img_file = images_dir / f"{sample_id}.jpg"

        lines = [
            "HÓA ĐƠN BÁN HÀNG TÀI CHÍNH",
            f"Đơn vị bán: {seller}",
            f"Số hóa đơn: {inv_no}",
            f"Ngày xuất: {inv_date}",
            "---------------------------------",
            f"Tổng thanh toán: {amount:,.0f} VND".replace(",", "."),
            "Cảm ơn quý khách đã mua hàng!",
        ]
        # Simulate different visual conditions via background / font colors
        bg = "#F8FAFC" if i % 2 == 0 else "#FFFBEB"
        fg = "#0F172A" if i % 4 != 0 else "#334155"
        create_sample_image(img_file, lines, bg_color=bg, text_color=fg)

        samples.append({
            "sample_id": sample_id,
            "image_path": f"images/{sample_id}.jpg",
            "task_type": "invoice",
            "category": "receipt",
            "prompt": "Trích xuất thông tin hóa đơn gồm seller_name, invoice_number, invoice_date, total_amount, currency sang JSON.",
            "ground_truth": {
                "seller_name": seller,
                "invoice_number": inv_no,
                "invoice_date": inv_date,
                "total_amount": amount,
                "currency": "VND",
            },
            "difficulty": "easy" if i <= 10 else ("medium" if i <= 16 else "hard"),
            "quality_tags": ["clean"] if i <= 10 else ["photographed", "low-light"],
        })

    # 2. 15 Document OCR cases
    ocr_texts = [
        "Cộng hòa Xã hội Chủ nghĩa Việt Nam. Độc lập - Tự do - Hạnh phúc.",
        "Thông báo về việc áp dụng quy chế đào tạo tín chỉ và quản lý sinh viên năm học mới.",
        "Nghị quyết đại hội cổ đông thường niên thông qua kế hoạch kinh doanh và phân phối lợi nhuận.",
        "Hợp đồng cung cấp dịch vụ công nghệ thông tin và chuyển đổi số doanh nghiệp.",
        "Biên bản nghiệm thu hoàn thành các hạng mục công trình xây lắp dân dụng.",
        "Quyết định bổ nhiệm cán bộ quản lý và kiện toàn bộ máy tổ chức điều hành.",
        "Quy trình kiểm soát chất lượng sản phẩm xuất khẩu theo tiêu chuẩn ISO 9001.",
        "Báo cáo tài chính kiểm toán độc lập cho năm tài chính kết thúc ngày 31 tháng 12.",
        "Chính sách bảo mật thông tin khách hàng và bảo vệ dữ liệu cá nhân trên không gian mạng.",
        "Quy định về thời giờ làm việc, thời giờ nghỉ ngơi và chế độ bảo hộ lao động cho công nhân.",
        "Chỉ thị về tăng cường công tác phòng cháy chữa cháy và cứu nạn cứu hộ tại cơ sở.",
        "Kế hoạch triển khai chương trình kích cầu du lịch và xúc tiến thương mại quốc tế.",
        "Giấy phép đăng ký kinh doanh và mã số doanh nghiệp do Sở Kế hoạch Đầu tư cấp.",
        "Hướng dẫn thủ tục kê khai thuế giá trị gia tăng điện tử cho hộ kinh doanh cá thể.",
        "Cam kết bảo đảm an toàn vệ sinh thực phẩm trong chuỗi cung ứng nông sản sạch.",
    ]

    for j, text in enumerate(ocr_texts, 1):
        sample_id = f"ocr_{j:03d}"
        img_file = images_dir / f"{sample_id}.jpg"
        lines = [
            f"VĂN BẢN TRÍCH ĐOẠN SỐ {j:02d}",
            "--------------------------------------------------",
            text,
        ]
        create_sample_image(img_file, lines, bg_color="#FFFFFF", text_color="#1E293B", size=(720, 260))

        samples.append({
            "sample_id": sample_id,
            "image_path": f"images/{sample_id}.jpg",
            "task_type": "ocr",
            "category": "document",
            "prompt": "Hãy đọc và trích xuất nguyên văn toàn bộ nội dung văn bản trong ảnh.",
            "ground_truth": text,
            "difficulty": "easy" if j <= 7 else "medium",
            "quality_tags": ["clean"],
        })

    # 3. 15 Visual QA cases
    vqa_questions = [
        ("Mô tả màu sắc và hình dạng của vật thể trung tâm.", "Vật thể hình tròn màu xanh lam trên nền trắng.", "scene"),
        ("Biểu đồ thể hiện xu hướng tăng hay giảm qua các quý?", "Xu hướng tăng trưởng liên tục qua cả bốn quý.", "chart"),
        ("Có bao nhiêu cột được hiển thị trong hình vẽ biểu đồ?", "Có 4 cột tương ứng với các quý Q1, Q2, Q3, Q4.", "chart"),
        ("Biển báo giao thông trong ảnh có nội dung gì?", "Biển báo cấm đi ngược chiều hình tròn viền đỏ nền trắng.", "text"),
        ("Khu vực cảnh quan trong ảnh là ban ngày hay ban đêm?", "Ảnh chụp cảnh quan ban ngày dưới ánh sáng mặt trời.", "scene"),
        ("Logo thương hiệu hiển thị tên công ty nào?", "Logo hiển thị chữ Vinamilk màu xanh đặc trưng.", "text"),
        ("Trục tung của biểu đồ đo lường chỉ số gì?", "Trục tung đo lường doanh thu tính bằng tỷ đồng.", "chart"),
        ("Hai vật thể nổi bật nhất trong khung cảnh là gì?", "Tòa nhà cao tầng và hàng cây xanh bên vỉa hè.", "scene"),
        ("Cửa hàng trong ảnh đang mở cửa hay đóng cửa?", "Cửa hàng có biển hiệu Open và đang mở cửa đón khách.", "scene"),
        ("Tài liệu trong ảnh có chữ ký và con dấu đỏ không?", "Tài liệu có chữ ký mực xanh và con dấu tròn màu đỏ.", "document"),
        ("Màu sắc chủ đạo của bức tranh phong cảnh là màu gì?", "Màu xanh ngọc bích của mặt hồ và màu xanh lá cây của rừng núi.", "scene"),
        ("Dòng chữ in hoa lớn nhất trên áp phích ghi chữ gì?", "Dòng chữ HỘI NGHỊ KHOA HỌC QUỐC TẾ 2026.", "text"),
        ("Thanh nào trong biểu đồ có giá trị cao nhất?", "Thanh cột Quý 4 có giá trị cao nhất đạt 120 tỷ.", "chart"),
        ("Biển báo cho biết vận tốc tối đa cho phép là bao nhiêu?", "Vận tốc tối đa cho phép là 60 km/h.", "text"),
        ("Ảnh chụp tài liệu có bị mờ hoặc nhòe góc nào không?", "Ảnh rõ nét, đầy đủ ánh sáng và không bị nhòe góc.", "document"),
    ]

    for k, (question, answer, cat) in enumerate(vqa_questions, 1):
        sample_id = f"vqa_{k:03d}"
        img_file = images_dir / f"{sample_id}.jpg"
        lines = [
            f"MẪU THÍ NGHIỆM VQA #{k:02d} ({cat.upper()})",
            f"Nội dung minh họa: {answer}",
        ]
        create_sample_image(img_file, lines, bg_color="#F1F5F9", text_color="#0F172A", size=(640, 320))

        samples.append({
            "sample_id": sample_id,
            "image_path": f"images/{sample_id}.jpg",
            "task_type": "vqa",
            "category": cat,
            "prompt": question,
            "ground_truth": answer,
            "difficulty": "medium",
            "quality_tags": ["photographed"],
        })

    manifest_path = dest_dir / "vietnamese_benchmark_50.jsonl"
    with manifest_path.open("w", encoding="utf-8") as fh:
        for item in samples:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    return manifest_path
