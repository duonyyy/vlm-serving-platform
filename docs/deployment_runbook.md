# Sổ Tay Vận Hành Triển Khai Production (Deployment Runbook)

Tài liệu này cung cấp quy trình chuẩn hóa để chuyển dịch hệ thống **VLM Serving Platform** từ môi trường thực nghiệm Kaggle sang GPU Server Production được phép vận hành lâu dài (AWS EC2, RunPod, Lambda Labs hoặc máy chủ On-Premise).

---

## 1. Yêu Cầu Hạ Tầng Phần Cứng (Hardware Requirements)

| Thành phần | Cấu hình tối thiểu | Khuyến nghị Production | Ghi chú |
|---|---|---|---|
| **GPU** | 1x NVIDIA Tesla T4 (16GB VRAM) | 1x NVIDIA A10G (24GB) hoặc A100 | Hỗ trợ CUDA 12.1+, CC >= 7.5 |
| **CPU** | 4 vCPU | 8 vCPU | Xử lý image decoding và WebSocket serialization |
| **RAM** | 16 GB | 32 GB | Tránh OOM khi nạp nhiều ảnh đồng thời |
| **Disk** | 50 GB SSD (NVMe) | 100 GB NVMe SSD | Cache model weights (Qwen-VL ~4.5GB - 9GB) |
| **Hệ điều hành** | Ubuntu 22.04 LTS | Ubuntu 22.04 LTS | Cài đặt NVIDIA Container Toolkit |

---

## 2. Quản Trị Bí Mật và Biến Môi Trường (Secrets Management)

> [!CAUTION]
> Tuyệt đối không commit token hay API key vào Git repository hoặc notebook.

Tạo file `.env.prod` trên server production với quyền truy cập nghiêm ngặt (`chmod 600 .env.prod`):

```bash
# Gateway Auth Secret
VLM_WORKER_AUTH_TOKEN="sec_prod_vlm_worker_xyz789_2026"
# Client API Key (nếu mở public ingress)
VLM_CLIENT_API_KEY="sec_prod_client_key_abcdef123"

# Model & Optimization Settings
VLM_MODEL_ID="Qwen/Qwen2-VL-2B-Instruct"
VLM_ENABLE_CACHE="true"
VLM_CACHE_MAX_ENTRIES="5000"
VLM_SCHEDULING_STRATEGY="least_loaded"
```

---

## 3. Quy Trình Cài Đặt GPU Server Lần Đầu

### Bước 1: Cài đặt NVIDIA Driver & NVIDIA Container Toolkit
```bash
# 1. Cập nhật hệ thống
sudo apt update && sudo apt install -y curl git docker.io docker-compose-plugin

# 2. Cài đặt NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 3. Kiểm tra GPU trong Docker
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

### Bước 2: Clone Repository và Chuẩn Bị Môi Trường
```bash
git clone https://github.com/your-org/vlm-serving-platform.git /opt/vlm-serving
cd /opt/vlm-serving
```

---

## 4. Khởi Chạy Production Stack Bằng Docker Compose

```bash
# Khởi chạy toàn bộ cụm: Gateway, GPU Worker và Prometheus
docker compose -f infra/docker/docker-compose.prod.yml --env-file .env.prod up -d --build
```

Kiểm tra trạng thái các container:
```bash
docker compose -f infra/docker/docker-compose.prod.yml ps
```

---

## 5. Giám Sát và Kiểm Tra Sức Khỏe (Healthcheck & Observability)

1. **Liveness & Readiness Check:**
   ```bash
   curl -s http://localhost:8000/health | jq .
   ```
   *Kỳ vọng:* `status: "healthy"`, `active_workers >= 1`.

2. **Kiểm tra Danh mục Model đang phục vụ:**
   ```bash
   curl -s http://localhost:8000/v1/models | jq .
   ```

3. **Thu thập Prometheus Metrics:**
   ```bash
   curl -s http://localhost:8000/metrics | head -n 30
   ```
   Truy cập Prometheus UI tại `http://<server-ip>:9090`.

4. **Kiểm tra Cache Hiệu Năng:**
   ```bash
   curl -s http://localhost:8000/v1/cache/stats | jq .
   ```

---

## 6. Quy Trình Khôi Phục Khi Có Sự Cố (Rollback & Incident Recovery)

### Kịch bản 1: GPU Worker bị lỗi OOM hoặc treo
1. Khóa hàng đợi Gateway và khởi động lại container worker:
   ```bash
   docker restart vlm-worker-gpu-01
   ```
2. Gateway sẽ tự động nhận diện worker ngắt kết nối, dọn dẹp các job dở dang và tái đăng ký phiên mới sau 1-2 giây mà không cần khởi động lại Gateway.

### Kịch bản 2: Model Version bị lỗi, cần Rollback phiên bản cũ
1. Sửa biến `VLM_MODEL_ID` hoặc git commit SHA trong `.env.prod`.
2. Khởi chạy lại container worker:
   ```bash
   docker compose -f infra/docker/docker-compose.prod.yml up -d --no-deps worker-gpu
   ```
3. Xác minh qua API `/v1/models` xem model cũ đã sẵn sàng phục vụ hay chưa.

### Kịch bản 3: Xả bộ nhớ đệm (Flush Cache)
Nếu phát hiện kết quả suy luận cũ bị lưu cache không đúng:
```bash
curl -X POST http://localhost:8000/v1/cache/clear
```
