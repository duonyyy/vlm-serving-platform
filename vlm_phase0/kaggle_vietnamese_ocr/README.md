# Benchmark OCR tiếng Việt trên Kaggle

Ba notebook độc lập so sánh hệ OCR đầu cuối `PaddlePaddle/PaddleOCR-VL-1.6` và `deepseek-ai/DeepSeek-OCR-2` bằng CC-OCR Vietnamese. Dự án hiện ở trạng thái **NOT_EVALUATED**: mã được tạo và kiểm tra offline; chưa có Kaggle T4, chưa tải TSV hay checkpoint trong Kaggle, chưa xác nhận nhãn, chưa chạy smoke/full. Không có số CER, WER, latency hoặc VRAM thật trong repository.

## Bắt đầu trên Kaggle

1. Tạo notebook Kaggle mới, bật **Internet** và **GPU T4 x2**. Upload [Notebook 01](01_paddleocr_vl_1_6_kaggle.ipynb) hoặc [Notebook 02](02_deepseek_ocr_2_kaggle.ipynb) dưới dạng notebook. Mỗi notebook có mã nguồn nhúng, không cần upload các `.py`.
2. Chạy tới phần xem 5 ảnh/`question`/`answer`; xác nhận `answer` đúng là văn bản trong ảnh. Nếu đúng, đặt `CONFIRM_GROUND_TRUTH_SEMANTICS = True`. Nếu sai, dừng và ghi nhận. Schema thực tế được lưu trong `dataset_schema.json` trước khi parser chạy.
3. Giữ `MODE = 'SMOKE_TEST'`, chạy 5 mẫu và kiểm tra prediction thực tế, trạng thái và GPU. Sửa lỗi tương thích có ghi log trước khi chuyển sang full.
4. Ghi commit dataset từ output Notebook 01; đặt `DATASET_REVISION_OVERRIDE` ở Notebook 02 bằng commit đó. Có thể dùng `MODEL_REVISION_OVERRIDE` ở Notebook 02 để tái lập một commit DeepSeek cũ. Đặt `MODE = 'FULL_BENCHMARK'` rồi chạy lại từng model trong phiên Kaggle riêng. Không chạy hai model đồng thời. `predictions.jsonl` được ghi sau mỗi mẫu; chạy lại cùng notebook và cùng revision sẽ bỏ qua các mẫu đã có trạng thái cuối. Kaggle chỉ giữ file giữa các lần chạy nếu output được lưu thành version hoặc gắn lại thành Dataset Input; phục hồi qua phiên mới cần khôi phục nguyên output về cùng `/kaggle/working/vietnamese_ocr/...` trước khi chạy.
5. **Save Version** cho mỗi notebook. Thêm hai Notebook Outputs hoặc Kaggle Datasets vào [Notebook 03](03_compare_results.ipynb), chọn **CPU/None**, chạy. Nếu có nhiều output full, đặt `PADDLE_OUTPUT_DIR` và `DEEPSEEK_OUTPUT_DIR` rõ ràng. Notebook 03 chặn kết quả nếu ID/hash/nhãn/tập mẫu khác hoặc thiếu mẫu.

Mọi file xuất dưới `/kaggle/working/vietnamese_ocr` hoặc `/kaggle/working/vietnamese_ocr_compare`. `manifest.csv` và `images/` trong mỗi output cho phép Notebook 03 hiển thị ảnh gốc. `rejected_samples.csv` giữ lý do loại mẫu; ảnh trùng vẫn được giữ và đánh dấu `duplicate_of`. Tên `Vietnamese_150.tsv` không bảo đảm có đúng 150 ảnh hợp lệ.

## Phương pháp

- Cùng TSV theo commit SHA, cùng ảnh bytes/hash, cùng `answer` và sample ID. Thứ tự gốc TSV cố định; seed 42 được dành cho mọi thao tác có ngẫu nhiên (hiện pipeline không lấy mẫu ngẫu nhiên). Batch size 1, GPU 0.
- Strict: giữ nguyên Unicode, chữ hoa, dấu câu và xuống dòng. Normalized: chỉ Unicode NFC và khoảng trắng. Không xóa dấu. CER là tổng edit distance / tổng ký tự nhãn; WER là edit distance token theo khoảng trắng / tổng token nhãn. CER có thể vượt 1.
- Raw output và plain-text extraction đều được lưu. Loại bỏ cú pháp Markdown phổ biến để chấm, giữ nguyên nội dung chữ; đây là một nguồn sai lệch cần kiểm tra ở ảnh lỗi.
- `failed`, `oom`, `timeout` ở lại trong JSONL. Chất lượng đầu cuối tính chúng như prediction rỗng và báo runtime error rate riêng. `timeout` là ngưỡng hậu kiểm 300 giây; không có cơ chế ngắt một lệnh GPU đang treo.
- Cả hai giới hạn sinh ở 2048 token. DeepSeek suy đoán `truncated` từ độ dài token decode; Paddle không cung cấp cờ truncation tin cậy nên ghi `null` và cần kiểm tra output ở smoke.
- Đồng bộ CUDA trước/sau inference. Warm-up trên ảnh đầu tiên được gọi riêng và không tính vào latency chính. NVML lấy mẫu VRAM toàn GPU mỗi 50 ms; có thể bỏ lỡ spike và bị ảnh hưởng bởi tiến trình khác. DeepSeek còn lưu peak PyTorch allocated/reserved; Paddle không dùng allocator PyTorch.
- Tốc độ chỉ kết luận trên cùng cấu hình T4, driver, CUDA, mức tải và phiên chạy; không phải phép đo riêng vision encoder.

## Trạng thái và giới hạn quan trọng

| Hạng mục | Trạng thái |
|---|---|
| Nguồn chính thức/model ID | Đã đối chiếu tài liệu, xem `PHASE1_RESEARCH.md` |
| Schema và ngữ nghĩa nhãn thực tế | **NOT_EVALUATED**; notebook kiểm tra và có cổng xác nhận |
| Notebook mở và chạy trên Kaggle | **NOT_EVALUATED** |
| Smoke/full đúng checkpoint trên T4 | **NOT_EVALUATED** |
| CER/WER/VRAM/latency | **NOT_EVALUATED** |

Paddle pipeline v1.6 dùng layout model cộng VLM; mã ghi commit của HF model card để tham chiếu nhưng pipeline Paddle có thể tải artifact qua registry khác. Chưa thể chứng minh hash trọng số đang chạy trùng byte với repository HF `PaddlePaddle/PaddleOCR-VL-1.6` cho đến khi kiểm tra trên Kaggle. Đây là cổng xác nhận trước khi tuyên bố so sánh chính xác hai checkpoint theo nghĩa byte-level.

DeepSeek mã chính thức hardcode BF16 trong `infer()` và 8192 token. Notebook tải checkpoint theo commit SHA, thay năm vị trí BF16 thực thi (và một comment) bằng FP16 và hai lần giới hạn sinh bằng 2048; có hash trước/sau và dừng nếu source khác. Đây là bản port precision chưa được nhóm tác giả xác nhận; khác biệt độ chính xác cần được ghi trong báo cáo. Nó không thay model, weights, prompt hoặc preprocessing. `eager` attention có tên trong bảng dispatch của mã chính thức, nhưng cần xác nhận lúc load/smoke trên T4.

## Phát triển và kiểm tra offline

`benchmark_core.py`, `model_adapters.py`, `compare.py` là nguồn để review. `python build_notebooks.py` tái tạo ba notebook standalone. `python -m unittest discover -s tests -v` chạy kiểm tra parser, Unicode, CER/WER và cổng so sánh bằng dữ liệu tổng hợp; đây không phải bằng chứng về chất lượng model hay Kaggle runtime.

Nguồn: [PaddleOCR-VL usage](https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/PaddleOCR-VL.html), [PaddleOCR 3.7.0](https://pypi.org/project/paddleocr/3.7.0/), [DeepSeek-OCR-2 official code](https://github.com/deepseek-ai/DeepSeek-OCR-2), [DeepSeek model card](https://huggingface.co/deepseek-ai/DeepSeek-OCR-2), [CC-OCR Vietnamese TSV](https://huggingface.co/datasets/wulipc/CC-OCR/tree/main/multi_lan_ocr/Vietnamese).
