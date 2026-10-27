# Phase 0 — VLM feasibility trên Kaggle T4

Trạng thái: **Phase 0 được chủ dự án thông qua để chuyển sang Phase 1** ngày 13/10/2026. [Lần chạy đầu](reports/run_2026-10-13_initial/README.md) đạt 17/20 output không rỗng và ba ảnh OOM; [lần chạy lại với resize](reports/run_2026-10-13_resize1280/README.md) đạt **20/20 output không rỗng, không còn OOM**, đáp ứng cổng smoke trong kế hoạch gốc. Chủ dự án xác nhận đã xem từng ảnh và chấp nhận kết quả cho Phase 0. File CSV vẫn chứa `UNREVIEWED` vì chưa ghi verdict từng dòng; điều này không có nghĩa người dùng chưa xem ảnh, nhưng cũng không tạo ra tỷ lệ đúng có thể kiểm chứng. [Bộ dữ liệu ZIP](phase0_bundle.zip) giữ nguyên 20 ảnh gốc và giới hạn cạnh dài ảnh đầu vào ở 1280 pixel trước suy luận. Hai notebook PaddleOCR-VL 1.6 và DeepSeek-OCR-2 hiện có ở [nhánh OCR](kaggle_vietnamese_ocr/README.md), vẫn `NOT_EVALUATED`.

## Chạy nhanh với 20 ảnh đã chọn

1. Trên Kaggle, tạo phiên bản mới cho **private Dataset** `vlm-phase0-bundle` bằng file `phase0_bundle.zip` mới; chỉ tải file ZIP này. Dataset gồm 5 cảnh vật, 5 tài liệu, 5 ảnh chữ và 5 biểu đồ. Xem `kaggle_bundle/sources.json` để ghi công và kiểm tra điều kiện của từng nguồn.
2. Tải [notebook đã cập nhật](notebooks/kaggle_worker.ipynb) lên Kaggle, Add Input là phiên bản Dataset mới, bật GPU T4 và Internet theo quota/điều khoản của Kaggle. Khởi động phiên mới để tránh module cũ còn trong Python cache. Ô đầu xác nhận `resize_longest_edge == 1280`.
3. Chạy tuần tự. Nếu slug Dataset khác, sửa `INPUT_ROOT` trong ô đầu tiên. ZIP được giải nén vào `/kaggle/working/vlm_phase0_bundle`.
4. Lưu Notebook Output và gửi lại `hardware_report.json`, `smoke_results.jsonl`, `smoke_report.json`, `smoke_review.csv` để kiểm tra. So sánh lần chạy mới với lần đầu theo cùng model revision, manifest SHA và chính sách resize đã ghi; kiểm tra lại cả 20 ảnh. Chữ nhỏ có thể khó đọc sau khi thu nhỏ. Những ghi chú đối chiếu được tạo trước khi chạy model, nhưng vẫn cần người xem lại ảnh và prediction.

## Thay bằng dữ liệu riêng (tùy chọn)

Tạo một Kaggle Dataset (ví dụ `vlm-phase0-bundle`) gồm:

```text
vlm_phase0/
  __init__.py
  core.py
  qwen.py
  run_phase0.py
manifest.jsonl
images/
  ... ít nhất 20 ảnh JPEG/PNG/WebP khác nhau ...
```

`manifest.jsonl` cần một JSON mỗi dòng:

```json
{"sample_id":"scene_001","image_path":"images/scene_001.jpg","category":"scene","prompt":"Mô tả hai vật nổi bật nhất trong ảnh.","expected_observation":"Người kiểm tra ghi mô tả có thể đối chiếu trực tiếp với ảnh."}
```

Gồm ít nhất năm ảnh ở mỗi nhóm `scene`, `document`, `text`, `chart`; tổng ít nhất 20 ảnh riêng biệt. Hãy dùng ảnh được phép sử dụng, không đưa hóa đơn có thông tin cá nhân lên dataset công khai. `expected_observation` là ghi chú cho người chấm, không được tạo từ output của model. Script kiểm tra đường dẫn, SHA-256 ảnh, định dạng, giới hạn 20 MiB/20 MP và trùng byte.

## Chạy

1. Mở [notebook Kaggle](notebooks/kaggle_worker.ipynb), bật GPU và Internet nếu được phép. Gắn Kaggle Dataset trên làm Input, sửa `INPUT_ROOT` đúng slug thực tế.
2. Chạy preflight; kiểm tra GPU, driver, CUDA, phiên bản package, VRAM và quota hiển thị trong Kaggle. Quota phải ghi thủ công vì API phần cứng không biết quota tài khoản.
3. Chạy 20 mẫu với `batch_size=1`, FP16, eager attention, greedy decoding, `max_new_tokens=128`. Model và processor được tải cùng một commit SHA. Nếu model load/OOM, lưu log và dừng, không thay checkpoint âm thầm.
4. Tải `reports/hardware_report.json`, `smoke_results.jsonl`, `smoke_report.json`, `smoke_review.csv`, `pip_freeze.txt` từ Notebook Output. Điền `verdict` cho từng dòng trong CSV sau khi xem ảnh gốc; `valid_nonempty_outputs` chỉ là kiểm tra thực thi, không phải điểm đúng.
5. Chạy riêng hai notebook OCR đã có nếu cần đánh giá khả thi Paddle/DeepSeek. Chúng có cổng xác nhận nhãn và không chứng minh tương thích T4 cho tới khi smoke thực tế thành công.

Chạy từ dòng lệnh trên Kaggle nếu không dùng notebook:

```bash
python -m vlm_phase0.run_phase0 --manifest /kaggle/input/vlm-phase0-bundle/manifest.jsonl --dataset-root /kaggle/input/vlm-phase0-bundle --output-dir /kaggle/working/vlm_phase0/reports
```

## Ranh giới bằng chứng

`reports/hardware_report.json` ở cấp này vẫn là placeholder `NOT_RUN`; dữ liệu GPU thật nằm trong hai thư mục lần chạy riêng. Bản ZIP hiện tại khác lần chạy đầu vì thêm chính sách resize. Phase 0 được khép theo cổng smoke của kế hoạch gốc: ít nhất 20 kết quả chạy thật, lỗi/OOM được ghi lại, và chủ dự án đã xem từng ảnh rồi chấp nhận cho phase này. Chưa có verdict lưu theo từng dòng để tính điểm chất lượng; không suy ra điểm đúng từ `runtime_exit_met=true`. T4 thứ hai không được dùng ngầm; baseline chạy trên `cuda:0`.

Bộ 20 ảnh này là **smoke test tính khả thi**, gồm nhiều ngôn ngữ và kiểu ảnh. Nó không đại diện cho hóa đơn tiếng Việt, không có nhãn chuẩn cho CER/WER và không đủ để kết luận chất lượng OCR hay chọn model production. Ảnh được giữ nguyên ở kích thước đã tải; `sources.json` ghi URL, tác giả/giấy phép, kích thước và SHA-256. Khi chia sẻ bộ ảnh, giữ kèm `sources.json` để ghi công theo giấy phép từng nguồn.

Nguồn công thức inference: [Qwen3-VL-2B model card](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct). Xem [bảng tương thích backend](docs/backend_compatibility.md).
