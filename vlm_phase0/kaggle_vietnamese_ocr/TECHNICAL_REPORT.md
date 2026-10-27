# Báo cáo kỹ thuật — trạng thái trước khi chạy Kaggle

1. **Dataset và giới hạn:** CC-OCR `Vietnamese_150.tsv`; schema, số ảnh hợp lệ và ý nghĩa `answer`: **NOT_EVALUATED**.
2. **Mô hình:** `PaddlePaddle/PaddleOCR-VL-1.6` pipeline v1.6 và `deepseek-ai/DeepSeek-OCR-2` zero-shot. Commit thực tế: **NOT_EVALUATED**.
3. **Phần cứng/môi trường:** Kaggle T4 x2 là mục tiêu. Python, CUDA, driver, VRAM thực tế: **NOT_EVALUATED**.
4. **Phương pháp:** manifest theo ảnh/hash, batch 1, strict/normalized CER/WER, checkpoint per image. Chi tiết trong README.
5. **Bảng kết quả:** CER, WER, Exact Match, p50/p95, throughput, peak VRAM: **NOT_EVALUATED**.
6. **Lỗi tiếng Việt:** dấu, đ/d, chữ hoa, số, dấu câu, thiếu/thừa nội dung: **NOT_EVALUATED**.
7. **Độ chính xác và tài nguyên:** **NOT_EVALUATED**.
8. **Threats to validity:** nhãn chưa xác nhận; FP16 DeepSeek là bản port; pipeline Paddle gồm layout; sample nhỏ; thời gian/VRAM phụ thuộc phiên GPU; Markdown extraction có thể thay đổi điểm; NVML đo toàn GPU.
9. **Kết luận:** Chưa đủ bằng chứng để xếp hạng hai mô hình. Notebook 03 tạo báo cáo đo được sau khi cả hai full run hợp lệ.
