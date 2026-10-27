# VLM Phase 0: tương thích backend trên Kaggle T4

Trạng thái ngày 13/10/2026: **Qwen3-VL-2B đã chạy trên Kaggle T4 với 20/20 output không rỗng sau khi giới hạn cạnh dài ảnh ở 1280 pixel; chất lượng chưa được chấm**. PaddleOCR-VL và DeepSeek-OCR-2 vẫn chỉ được đánh giá từ tài liệu/source trong nhánh này. Kiểm tra runtime và versions của từng backend ngay trên Kaggle trước khi kết luận.

| Model / đường chạy | Bằng chứng hiện có | Quyết định Phase 0 | Cổng xác nhận còn thiếu |
|---|---|---|---|
| Qwen3-VL-2B-Instruct / Transformers | [Model card chính thức](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) có recipe `Qwen3VLForConditionalGeneration` và `AutoProcessor`; [báo cáo T4](../reports/run_2026-10-13_resize1280/README.md) ghi commit SHA, versions, 20/20 output và VRAM | FP16, eager attention, `cuda:0`, giới hạn cạnh dài ảnh 1280 pixel đã qua cổng thực thi 20 ảnh trên T4 | Review từng ảnh/prediction; kiểm tra chữ nhỏ sau resize |
| PaddleOCR-VL 1.6 / PaddlePaddle | [PaddleOCR-VL docs](https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/PaddleOCR-VL.html) mô tả đường chạy và yêu cầu GPU; mã thử có sẵn tại [notebook OCR](../kaggle_vietnamese_ocr/01_paddleocr_vl_1_6_kaggle.ipynb) | Chạy notebook OCR riêng, không trộn environment với Qwen | Driver/CUDA wheel, tải artifact, smoke và nguồn weights thực tế |
| DeepSeek-OCR-2 / Transformers tùy biến | [Model card](https://huggingface.co/deepseek-ai/DeepSeek-OCR-2) và source của tác giả; notebook hiện có sửa BF16 sang FP16 có guard/hash | Chỉ là thử nghiệm port precision, không xem là official T4 support | Load, output đúng kiểu, OOM/latency và sai lệch do FP16 |
| vLLM / SGLang cho PaddleOCR-VL | [PaddleOCR-VL docs](https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/PaddleOCR-VL.html) đặt điều kiện GPU cho các engine; T4 CC 7.5 | Không đưa vào baseline Paddle trên T4 | Chỉ xét lại khi có GPU/backend hỗ trợ theo phiên bản cài thực tế |

Không lấy benchmark từ các model, prompt, precision hay máy khác để so sánh. Qwen general VLM và hai pipeline document OCR cần tập dữ liệu, metric và báo cáo riêng. Bằng chứng runtime Qwen nằm trong hai thư mục báo cáo Kaggle; phần Paddle/DeepSeek vẫn là đánh giá tương thích trước thử nghiệm.
