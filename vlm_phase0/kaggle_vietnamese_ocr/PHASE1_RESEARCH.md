# Phase 1 — Nguồn và rủi ro trước thực nghiệm

Ngày kiểm tra: 12-10-2026. Chưa có Kaggle runtime trong workspace này.

| Vấn đề | Bằng chứng hiện có | Ảnh hưởng / cổng kiểm tra |
|---|---|---|
| PaddleOCR-VL 1.6 trên T4 | [Tài liệu chính thức](https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/PaddleOCR-VL.html) nêu PaddlePaddle/Transformers cần CC ≥7.0, CUDA ≥11.8; vLLM cần CC ≥8.0 và T4 dễ timeout/OOM. | Dùng `PaddleOCRVL(pipeline_version='v1.6')` với PaddlePaddle local, không vLLM. Kiểm tra GPU/driver/CUDA trước cài. |
| Bản Paddle đúng 1.6 | [PyPI PaddleOCR](https://pypi.org/project/paddleocr/) cho thấy 3.6.0 phát hành cùng VL-1.6, 3.7.0 tiếp tục hỗ trợ. | Pin `paddleocr[doc-parser]==3.7.0`; pipeline version v1.6. Cần xác nhận artifact được tải và hash thực tế trên Kaggle; commit HF chưa chứng minh bytes của backend Paddle. |
| DeepSeek trên T4 | [README chính thức](https://github.com/deepseek-ai/DeepSeek-OCR-2) và [model card](https://huggingface.co/deepseek-ai/DeepSeek-OCR-2) dùng BF16 + FlashAttention 2; T4 CC 7.5 không hỗ trợ BF16 tính toán chuẩn. | Bắt buộc FP16 + eager attention; patch cụ thể có hash và guard. Smoke test trước full. Nếu lỗi, dừng, không fallback model. |
| Mã `infer()` DeepSeek | [Source chính thức](https://huggingface.co/deepseek-ai/DeepSeek-OCR-2/raw/main/modeling_deepseekocr2.py) hardcode BF16/8192 token; `eval_mode=True` trả chuỗi decode, `save_results=True` ghi `result.mmd`. | Notebook dùng `eval_mode=True`, kiểm tra kiểu trả về; giới hạn 2048 token qua patch. FP16 có thể thay đổi output. |
| Attention backend | [Dispatch trong source chính thức](https://huggingface.co/deepseek-ai/DeepSeek-OCR-2/blob/main/modeling_deepseekv2.py) có `mla_eager` và `mha_eager`. | Chọn `_attn_implementation='eager'`, xác nhận config sau load; chưa chứng minh thực thi trên T4. |
| Dataset | [HF file listing](https://huggingface.co/datasets/wulipc/CC-OCR/tree/main/multi_lan_ocr/Vietnamese) xác nhận `Vietnamese_150.tsv`, ~117 MB. | Schema/`answer` **chưa xác thực** vì dữ liệu chỉ tải trên Kaggle theo yêu cầu. Notebook in schema và 5 ca trước cổng xác nhận nhãn. Không giả định 150 ảnh hợp lệ. |
| Cân bằng hệ OCR | [Paddle docs](https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/PaddleOCR-VL.html) nêu pipeline gồm layout và VLM. | Báo cáo là so sánh hệ OCR đầu cuối; preprocessing gốc mỗi hệ được ghi. Không diễn giải là so sánh vision encoder. |
| So sánh thời gian/VRAM | Kaggle T4 x2 là hai GPU riêng; tài nguyên giữa phiên có thể khác. | Đo tuần tự trên `cuda:0`, batch 1, log environment, model init, warm-up, per-image. NVML là VRAM toàn GPU và có giới hạn tần suất lấy mẫu. |

### Cổng sau Phase 1

1. Xác nhận cột `answer` từ ảnh thật trên Kaggle trước khi chấm.
2. Smoke 5 ảnh và kiểm tra chất lượng output, dtype, attention, VRAM, lỗi thực tế.
3. Chỉ chuyển full khi hai smoke đều qua; nếu DeepSeek FP16/eager không chạy thì ghi lỗi, dừng mô hình đó.
4. Chỉ so sánh full khi cả hai hoàn thành cùng manifest và có kết quả thật.
