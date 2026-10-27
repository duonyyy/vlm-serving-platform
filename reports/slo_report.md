# Báo cáo Kiểm định SLO và Đánh giá Tải (Phase 5)

* **Ngày đánh giá:** 2026-10-26 19:12:45 UTC
* **Mục tiêu kiểm thử:** Gateway Production Control Plane (`http://127.0.0.1:64788`)
* **Trạng thái tuân thủ SLO:** **PASSED (Compliant)**

---

## 1. Tóm tắt Kiểm định Chỉ số Cấp độ Dịch vụ (SLO Compliance)

| Tiêu chí SLO | Ngưỡng cam kết (Target) | Kết quả đo được (Actual) | Trạng thái |
|---|:---:|:---:|:---:|
| **Availability (Tính sẵn sàng)** | `>= 99.0%` | `100.0%` | ✅ ĐẠT |
| **p95 E2E Latency** | `<= 1500.0 ms` | `247.5 ms` | ✅ ĐẠT |
| **Error Rate (Tỷ lệ lỗi)** | `<= 1.0%` | `0.0%` | ✅ ĐẠT |

---

## 2. Chi tiết Kết quả Thử nghiệm Tải (Load Test Metrics)

* **Tổng số request gửi:** `25`
* **Số request thành công:** `25`
* **Số request thất bại:** `0`
* **Mức độ đồng thời (Concurrency):** `5`
* **Tổng thời gian kiểm thử:** `0.55 s`
* **Throughput (Thông lượng tổng):** `45.06 req/s`
* **Goodput (Thông lượng đạt SLO):** `45.06 req/s`

### Phân bố Độ trễ (Turnaround Latency Percentiles):
* **p50 (Median):** `3.3 ms`
* **p90:** `182.8 ms`
* **p95:** `247.5 ms`
* **p99:** `313.6 ms`

---

## 3. Khuyến nghị Vận hành Production

1. **Auto-scaling Trigger:** Khi `vlm_pending_jobs > 20` hoặc `p95_latency_ms > 1200ms` kéo dài quá 2 phút, kích hoạt thêm 1 GPU Worker replica.
2. **Alerting Rules:** Cấu hình cảnh báo nếu `Error Rate > 1%` trong cửa sổ 5 phút qua Prometheus Alertmanager.
3. **Capacity Planning:** Cụm 1 GPU Tesla T4 đáp ứng tối đa 3–5 concurrent requests trước khi hàng đợi bắt đầu chịu độ trễ tích lũy.
