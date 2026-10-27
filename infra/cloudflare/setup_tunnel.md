# Thiết Lập Cloudflare Tunnel Cho VLM Serving Platform (Phase 2)

Tài liệu này hướng dẫn cách kết nối Gateway chạy cục bộ (hoặc trên VPS) với Remote Worker chạy trên Kaggle/Colab GPU.

---

## 1. Tại Sao Cần Cloudflare Tunnel?

- **Kaggle GPU Không Mở Ingress Port**: Kaggle cấm tất cả các kết nối inbound (không có public IP, không mở port trực tiếp).
- **Outbound WebSocket Model**:
  1. Gateway mở cổng HTTP/WebSocket (mặc định port `8000`).
  2. Cloudflare Tunnel tạo một URL HTTPS/WSS công khai (ví dụ `https://xxx.trycloudflare.com`).
  3. Kaggle Worker kết nối **outbound** tới `wss://xxx.trycloudflare.com/workers/connect`.
  4. Người dùng gọi API Gateway bình thường tại `https://xxx.trycloudflare.com/v1/chat/completions`.

---

## 2. Cách 1: Sử Dụng Quick Tunnel (Nhanh nhất - Không cần domain riêng)

Nếu bạn chỉ cần test nhanh mà không có domain:

```bash
# Cài đặt cloudflared (hoặc tải executable từ github.com/cloudflare/cloudflared)
# Chạy tunnel tạm thời trỏ vào port 8000 của Gateway:
cloudflared tunnel --url http://localhost:8000
```

Kết quả in ra terminal sẽ có dạng:
```
+--------------------------------------------------------------------------------------------+
|  Your quick Tunnel has been created! Visit it at (it may take some time to be reachable):  |
|  https://random-words-1234.trycloudflare.com                                               |
+--------------------------------------------------------------------------------------------+
```

Sao chép URL này và dùng trong Kaggle:
- **WebSocket URL**: `wss://random-words-1234.trycloudflare.com/workers/connect`
- **Client API URL**: `https://random-words-1234.trycloudflare.com/v1/chat/completions`

---

## 3. Cách 2: Sử Dụng Named Tunnel (Production với Domain riêng)

### Bước 1: Đăng nhập Cloudflare
```bash
cloudflared tunnel login
```

### Bước 2: Tạo Named Tunnel
```bash
cloudflared tunnel create vlm-gateway
```
Lưu lại `Tunnel-UUID`.

### Bước 3: Cấu hình DNS Route
```bash
cloudflared tunnel route dns vlm-gateway vlm.yourdomain.com
```

### Bước 4: Tạo file cấu hình `config.yml`
Tham khảo [tunnel_config.example.yml](file:///c:/Users/Admin/Desktop/ExcelMCP/SE373%20-%20AGENTIC%20AI/vlm-serving-platform/infra/cloudflare/tunnel_config.example.yml):
```yaml
tunnel: YOUR_TUNNEL_UUID
credentials-file: /path/to/credentials.json

ingress:
  - hostname: vlm.yourdomain.com
    service: http://localhost:8000
  - service: http_status:404
```

### Bước 5: Chạy Tunnel
```bash
cloudflared tunnel run vlm-gateway
```

Hoặc qua Docker Compose:
```bash
CLOUDFLARE_TUNNEL_TOKEN="<token>" docker compose -f infra/docker/docker-compose.yml up -d
```
