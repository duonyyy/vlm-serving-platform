# Kaggle T4 rerun — 2026-10-13, max edge 1280

Các file JSON/JSONL trong thư mục này do người dùng tải từ Kaggle sau lần chạy lại Phase 0. Đã đối chiếu với [lần chạy đầu](../run_2026-10-13_initial/README.md): cùng 20 `sample_id`, SHA-256 của từng ảnh gốc, `manifest_sha256` và model revision `89644892e4d85e24eaac8bacfd4f463576704203`.

- Phần cứng: hai Tesla T4 được cấp; baseline dùng `cuda:0`. PyTorch `2.11.0+cu128`, Transformers `4.57.6`.
- Cấu hình: FP16, eager attention, batch size 1, tối đa 128 output tokens, `do_sample=false`, PIL LANCZOS với cạnh dài tối đa 1280 pixel. Kích thước ảnh đưa vào model có trong từng dòng `smoke_results.jsonl`.
- 20/20 dòng có trạng thái `completed` và `prediction_text` không rỗng; không có OOM. `runtime_exit_met=true`.
- Peak PyTorch allocated lớn nhất trong các dòng đã hoàn thành: khoảng 6,86 GiB; lần đầu khoảng 12,79 GiB. Đây là hai cấu hình ảnh khác nhau và hai phiên chạy, chưa đủ để kết luận hiệu năng hoặc chất lượng vượt trội.
- [smoke_review.csv](smoke_review.csv) đã được nhận và khớp chính xác 20 prediction của lần chạy này. File vẫn có 20 `verdict` là `UNREVIEWED` và không có `reviewer_note`; sau đó chủ dự án xác nhận đã xem từng ảnh và chấp nhận kết quả để qua Phase 0. Chưa có nhãn từng dòng để tính tỷ lệ đúng. Việc thu nhỏ có thể làm mất chữ nhỏ trong tài liệu.
- Kiểm tra nhanh ba ảnh từng OOM: `document_005` trả lời đúng ý chính về tiêu đề và chữ ký; `scene_003` có cụm “xe điện ngầm” dù ảnh là cabin cáp treo; `document_003` suy đoán “chiếc cốc” cho biểu trưng trên trang. Đây là ghi chú rà soát sơ bộ, chưa phải verdict cho toàn bộ tập.
- `text_004` hỏi hai từ đầu trên biển, nhưng prediction chỉ có một từ `БЕЛГОРОД`; cần đánh dấu thiếu thông tin được yêu cầu. `chart_005` nêu thêm ngày 30/6/2023 trong khi ảnh ghi 31/12/2023; kiểm tra phần thông tin không được hỏi trước khi chấm.

Để chấm: mở ảnh gốc tương ứng theo `sample_id` trong `../../kaggle_bundle/images/`, so sánh `prediction_text` với câu hỏi và nội dung nhìn thấy; điền `verdict` cho từng dòng (`PASS` = đúng và không có lỗi quan trọng, `PARTIAL` = đúng ý chính nhưng thiếu hoặc thêm thông tin sai, `FAIL` = sai ý chính), ghi lý do ngắn ở `reviewer_note`. `expected_observation` là gợi ý, không thay thế việc xem ảnh.

Trạng thái: **đạt cổng thực thi 20/20; chủ dự án xác nhận đã xem từng ảnh và thông qua Phase 0 ngày 13/10/2026**. Chưa có verdict từng ảnh trong `smoke_review.csv`, nên chưa công bố tỷ lệ đúng hoặc điểm chất lượng.
