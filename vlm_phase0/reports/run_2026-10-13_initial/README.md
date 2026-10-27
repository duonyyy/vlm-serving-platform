# Kaggle T4 run — 2026-10-13

Các file JSON/JSONL trong thư mục này do người dùng tải từ Kaggle và cung cấp sau lần chạy Phase 0 đầu tiên. `manifest_sha256` khớp `manifest.jsonl` trong bundle ban đầu. Đây là bằng chứng chạy GPU, chưa có `smoke_review.csv` đã chấm.

- Model: `Qwen/Qwen3-VL-2B-Instruct`, revision `89644892e4d85e24eaac8bacfd4f463576704203`.
- Hai Tesla T4 được cấp; baseline dùng riêng `cuda:0` với FP16, eager attention, batch size 1, tối đa 128 output tokens và ảnh gốc không giới hạn resize.
- 20 ảnh được thử: 17 `completed`, ba `oom` (`scene_003`, `document_003`, `document_005`). `runtime_exit_met=false`; `visual_correctness=UNREVIEWED`.
- Các ảnh lỗi có kích thước lần lượt 1920×1280, 1240×1755 và 1280×1849. Log cho thấy cấp phát CUDA cần thêm 2,75–5,13 GiB trên GPU 0 khi bộ nhớ trống không đủ.

Đây là lần chạy thất bại có giá trị chẩn đoán, không phải benchmark chất lượng hoặc độ trễ đầy đủ. Lần chạy kế tiếp dùng chính sách giới hạn cạnh dài 1280 pixel và phải lưu báo cáo riêng; không trộn số liệu hai cấu hình.
