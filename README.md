# VLM Serving & Evaluation

Thư mục này chứa dự án VLM Serving Platform (chuyển dịch từ thực nghiệm Kaggle T4 sang Serving Production).

* **Kế hoạch tổng thể 6 Phase:** [KEHOACH.md](KEHOACH.md)
* **API Specification:** [docs/api_contract.md](docs/api_contract.md)

---

## Tiến độ các Phase

| Phase | Thành phần | Trạng thái |
|---|---|---|
| **0 — Feasibility** | `vlm_phase0/` (smoke test 20/20 không OOM trên Kaggle T4 với resize 1280px) | **Hoàn thành (Passed)** |
| **1 — Phục vụ 1 VLM** | `packages/contracts/`, `packages/image_processing/`, `packages/model_adapters/`, `packages/worker_runtime/`, `client/`, unit & integration tests | **Hoàn thành (Implemented & Verified)** |
| **2 — Gateway & Worker từ xa** | `apps/gateway/` (Control Plane, Worker Registry, Bounded Queue Dispatcher), `packages/worker_runtime/remote_client.py` (Outbound WebSocket Client, auto-reconnect), `notebooks/kaggle_worker_phase2.ipynb`, `infra/` (Docker & Cloudflare Tunnel), Fault-injection tests | **Hoàn thành (Implemented & Verified)** |
| **3 — Đánh giá & Benchmark** | `evaluation/` (chỉ số CER/WER, Field F1, ANLS, p50/p95 latency, bộ 50 mẫu test, error taxonomy, reports & plots) | **Hoàn thành (Implemented & Verified)** |
| **4 — Tối ưu & Ablation** | `packages/caching/` (SHA-256 LRU cache), `packages/image_processing/dynamic_resizer.py` (resolution scaling), `apps/gateway/scheduler.py` (LeastLoaded / RoundRobin), `evaluation/ablation_runner.py`, `evaluation/pareto.py`, `reports/ablation.md`, Pareto chart | **Hoàn thành (Implemented & Verified)** |
| **5 — GPU Server Production** | Docker GPU worker (`infra/docker/Dockerfile.worker_gpu`), Docker Compose (`docker-compose.prod.yml`), Prometheus monitoring, Metrics Exporter (`/metrics`), Load Test & SLO verification (`reports/slo_report.md`), Production Deployment Runbook (`docs/deployment_runbook.md`), CI/CD workflow (`.github/workflows/ci.yml`) | **Hoàn thành (Implemented & Verified)** |



---

## Cấu trúc mã nguồn

```text
vlm-serving-platform/
├── packages/
│   ├── contracts/             # Pydantic schemas (OpenAI multimodal chat, extraction, models, errors)
│   ├── image_processing/      # Decode, validate (MIME, max bytes/pixels), resize Lanczos 1280, SSRF safety
│   ├── model_adapters/        # BaseModelAdapter, MockAdapter (testing), QwenVLAdapter (Transformers T4)
│   ├── telemetry/             # Request ID context propagation, structured JSON logging, timer
│   └── worker_runtime/        # FastAPI worker app, concurrency lock, CLI runner
├── client/                    # Python Client SDK (VLMClient, chat_image, extract)
├── evaluation/                # Suite đánh giá & Benchmark (Phase 3)
│   ├── datasets/              # Bộ 50 mẫu vàng tiếng Việt (Invoice, OCR, VQA) + schema
│   ├── quality/               # CER, WER, Exact Match, Field F1, ANLS, chuẩn hóa tiền tệ/ngày
│   ├── performance/           # Đo p50/p90/p95/p99 latency, throughput (req/s & tok/s), peak VRAM
│   ├── qualitative/           # Phân loại lỗi theo taxonomy 8 nhóm & tính Cohen's Kappa
│   ├── results/               # Kết quả raw JSONL, summary JSON, export CSV
│   ├── reports/               # Báo cáo Markdown & biểu đồ trực quan hóa
│   └── runner.py              # Orchestrator chạy benchmark tự động
├── examples/
│   └── quickstart_phase1.py   # Demo hoàn chỉnh phục vụ & client
├── tests/
│   ├── unit/                  # Unit tests cho contracts, image_processing, adapters, telemetry, metrics
│   └── integration/           # Integration tests cho worker API, client E2E, benchmark runner
├── docs/
│   └── api_contract.md        # Đặc tả chi tiết chuẩn API
├── vlm_phase0/                # Kết quả và bundle của Phase 0
├── pyproject.toml             # Cấu hình dependency & package
└── README.md
```

---

## Hướng dẫn sử dụng nhanh Phase 1

### 1. Chạy Quickstart Demo
Chạy kịch bản hoàn chỉnh (khởi tạo worker, kiểm tra health, list models, multimodal chat với ảnh, trích xuất hóa đơn JSON):

```powershell
python examples/quickstart_phase1.py
```

### 2. Khởi chạy Worker API Server
Chạy worker với `MockAdapter` để kiểm thử cục bộ:
```powershell
python -m packages.worker_runtime.server --host 127.0.0.1 --port 8000 --adapter mock
```

Hoặc chạy với `QwenVLAdapter` trên GPU NVIDIA T4:
```powershell
python -m packages.worker_runtime.server --host 0.0.0.0 --port 8000 --adapter qwen --device cuda:0
```

### 3. Sử dụng Python Client SDK
```python
from client import VLMClient

client = VLMClient(base_url="http://127.0.0.1:8000")

# Kiểm tra trạng thái
health = client.get_health()
print("Worker Status:", health.status)

# Gửi ảnh và prompt
resp = client.chat_image(
    image="path/to/image.jpg",
    prompt="Mô tả bức ảnh này.",
    model="qwen3-vl-2b"
)
print("Assistant:", resp.choices[0].message.content)

# Trích xuất hóa đơn có cấu trúc
result = client.extract(
    image="path/to/invoice.jpg",
    schema_name="invoice"
)
print("Invoice Data:", result.data)
```

---

## Hướng dẫn sử dụng Phase 2 (Gateway & Remote Worker)

### 1. Khởi chạy Gateway Control Plane
```powershell
python -m apps.gateway.server --host 0.0.0.0 --port 8000 --token vlm_worker_secret_2026
```

### 2. Thiết lập Cloudflare Tunnel
Xem hướng dẫn chi tiết tại [infra/cloudflare/setup_tunnel.md](infra/cloudflare/setup_tunnel.md):
```powershell
cloudflared tunnel --url http://localhost:8000
```
Lấy URL Cloudflare Tunnel (ví dụ: `wss://<subdomain>.trycloudflare.com/workers/connect`).

### 3. Khởi chạy Worker trên Kaggle GPU (Tesla T4)
Mở notebook [notebooks/kaggle_worker_phase2.ipynb](notebooks/kaggle_worker_phase2.ipynb) trên Kaggle, điền URL WebSocket và chạy cell để kết nối outbound về Gateway.

---

## Hướng dẫn sử dụng Phase 4 (Tối ưu & Ablation Study)

Chạy phân tích đa mục tiêu (Ablation 4 trục + Pareto Frontier Analysis):
```powershell
python -m evaluation.ablation_runner
```
Kết quả sinh ra:
* Báo cáo Markdown chi tiết: [reports/ablation.md](reports/ablation.md)
* Đồ thị phân bố tối ưu Pareto: [reports/pareto_frontier.png](reports/pareto_frontier.png)

---

## Hướng dẫn sử dụng Phase 5 (Production GPU Server & Observability)

### 1. Khởi chạy toàn bộ cụm Production với Docker Compose
```bash
docker compose -f infra/docker/docker-compose.prod.yml up -d --build
```
Stack bao gồm:
* **Gateway Control Plane:** `http://localhost:8000` (OpenAI-compatible multimodal chat, `/v1/models`, `/health`)
* **Prometheus Metrics:** `http://localhost:8000/metrics`
* **Prometheus Dashboard:** `http://localhost:9090`
* **GPU Worker Container:** Tự động phát hiện NVIDIA GPU (`nvidia-smi`), nạp `QwenVLAdapter` và kết nối WebSocket outbound về Gateway.

Xem sổ tay vận hành hoàn chỉnh tại [docs/deployment_runbook.md](docs/deployment_runbook.md).

### 2. Chạy Kiểm định Tải và Xác minh SLO (Load Testing & SLO Verification)
```powershell
python -m evaluation.load_test_runner --url http://127.0.0.1:8000 --requests 50 --concurrency 5 --reports-dir reports
```
Kết quả sinh ra báo cáo Markdown: [reports/slo_report.md](reports/slo_report.md) kiểm định 3 tiêu chí cam kết SLO:
* **Availability:** >= 99%
* **p95 Latency:** <= 1500 ms
* **Error Rate:** <= 1%

---

## Chạy kiểm thử tự động (Unit, Integration & Fault Injection Tests)

Chạy toàn bộ 72 test case (Phase 1, Phase 2, Phase 3, Phase 4, Phase 5):
```powershell
python -m unittest discover -s tests -v
```

Bao gồm:
- **Phase 1 Contract, Processing, Adapter & Runtime Tests:** 32 tests
- **Phase 2 Registry, Dispatcher, E2E WebSocket & Resilience Tests:** 12 tests
- **Phase 3 Quality Metrics, Performance & Qualitative Tests:** 13 tests
- **Phase 4 Caching, Dynamic Resizer, Scheduler & Ablation Tests:** 13 tests
- **Phase 5 Metrics Exporter & Load Testing / SLO Integration Tests:** 2 tests

Chạy kiểm thử hồi quy Phase 0 (14 test cases):
```powershell
python -m unittest discover -s vlm_phase0/tests -v
python -m unittest discover -s vlm_phase0/kaggle_vietnamese_ocr/tests -v
```
*(Tổng cộng: **86/86 tests passed 100%**)*



