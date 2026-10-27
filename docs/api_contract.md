# VLM Serving Platform — API Contract (Phase 1)

**Version:** 1.0  
**Status:** Implemented & Verified  
**Compatibility:** OpenAI Multimodal Chat Completions API (`/v1/chat/completions`)

---

## 1. Overview & Headers

The VLM Worker API provides a low-latency, strictly validated HTTP serving interface for Vision-Language Models (e.g. Qwen3-VL-2B).

### Standard Headers
* `Content-Type`: `application/json`
* `X-Request-ID` *(optional in request, guaranteed in response)*: Tracing ID (`req_<uuid>`).
* `X-Process-Time-Ms` *(response)*: Total wall-clock time spent processing request in milliseconds.
* `Authorization` *(optional)*: `Bearer <token>` (if API token authentication is configured).

---

## 2. Endpoints

### 2.1. Health Check
* **Method:** `GET`
* **Path:** `/health`
* **Description:** Returns worker status, active model state, uptime, and GPU allocation statistics.

#### Response `200 OK`
```json
{
  "status": "healthy",
  "worker_id": "worker_01a",
  "uptime_seconds": 128.45,
  "active_model": "Qwen/Qwen3-VL-2B-Instruct",
  "active_model_status": "READY",
  "cuda_available": true,
  "gpu": {
    "device_name": "Tesla T4",
    "vram_allocated_bytes": 7365998592,
    "vram_reserved_bytes": 8128561152
  },
  "timestamp": "2026-10-26T15:40:00Z"
}
```

---

### 2.2. List Models
* **Method:** `GET`
* **Path:** `/v1/models`
* **Description:** Lists all loaded and available models conforming to OpenAI `/v1/models`.

#### Response `200 OK`
```json
{
  "object": "list",
  "data": [
    {
      "id": "Qwen/Qwen3-VL-2B-Instruct",
      "object": "model",
      "created": 1728800000,
      "owned_by": "vlm-platform",
      "status": "READY",
      "capabilities": {
        "image_chat": true,
        "ocr_markdown": true,
        "structured_json": true,
        "max_image_edge": 1280,
        "supports_concurrency": false
      },
      "revision": "89644892e4d85e24eaac8bacfd4f463576704203",
      "device": "cuda:0"
    }
  ]
}
```

---

### 2.3. Multimodal Chat Completion
* **Method:** `POST`
* **Path:** `/v1/chat/completions`
* **Description:** Standard OpenAI-compatible multimodal endpoint accepting text and image inputs.

#### Request Body
```json
{
  "model": "qwen3-vl-2b",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Mô tả hai vật nổi bật nhất trong ảnh."},
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
            "detail": "auto"
          }
        }
      ]
    }
  ],
  "max_tokens": 128,
  "temperature": 0.0
}
```

#### Image Input Rules:
1. **Data URI:** `data:image/jpeg;base64,...`, `data:image/png;base64,...`, `data:image/webp;base64,...`
2. **Raw Base64:** Valid base64 encoded byte string.
3. **HTTP/HTTPS URL:** Validated against SSRF (private/loopback/metadata IPs blocked).
4. **Limits:** Maximum 20 MiB bytes, maximum 20,000,000 pixels.
5. **Preprocessing:** Longest edge resized to at most 1280 pixels using Lanczos resampling (Phase 0 policy).

#### Response `200 OK`
```json
{
  "id": "chatcmpl-7c19ba820f12",
  "object": "chat.completion",
  "created": 1728800000,
  "model": "Qwen/Qwen3-VL-2B-Instruct",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Ảnh hiển thị một toa cáp treo màu đỏ và dãy núi phía sau."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 312,
    "completion_tokens": 24,
    "total_tokens": 336
  },
  "timing": {
    "preprocessing_ms": 12.35,
    "inference_ms": 345.82,
    "total_ms": 361.20
  }
}
```

---

### 2.4. Structured Extraction
* **Method:** `POST`
* **Path:** `/v1/extract`
* **Description:** Specialized endpoint for extracting structured fields (e.g. Vietnamese invoices).

#### Request Body
```json
{
  "model": "qwen3-vl-2b",
  "image_url": "data:image/png;base64,...",
  "schema_name": "invoice",
  "fields": ["seller_name", "invoice_date", "total_amount", "currency"]
}
```

#### Response `200 OK`
```json
{
  "request_id": "req_f8120b12",
  "model": "qwen3-vl-2b",
  "status": "completed",
  "data": {
    "seller_name": "Công ty TNHH Cà Phê Việt",
    "invoice_number": "HD-2026-9921",
    "invoice_date": "2026-10-13",
    "total_amount": 150000.0,
    "currency": "VND"
  },
  "usage": {
    "prompt_tokens": 340,
    "completion_tokens": 45,
    "total_tokens": 385
  },
  "timing": {
    "preprocessing_ms": 10.2,
    "inference_ms": 412.0,
    "total_ms": 425.5
  }
}
```

---

## 3. Error Handling

All errors return JSON adhering to the `ErrorResponse` schema:

```json
{
  "error": {
    "message": "Image size 25165824 bytes exceeds limit of 20971520 bytes (20 MiB)",
    "type": "invalid_request_error",
    "param": "image_url",
    "code": "bad_input"
  },
  "request_id": "req_88a912bd"
}
```

### HTTP Status Codes
| Code | Error Type | Cause |
|---|---|---|
| `400 Bad Request` | `invalid_request_error` | Corrupted image, unsupported format, SSRF violation, oversize payload. |
| `404 Not Found` | `not_found_error` | Model ID not found. |
| `422 Unprocessable` | `validation_error` | Pydantic payload schema violation. |
| `503 Service Unavailable` | `runtime_error` | Model not loaded or worker initializing. |
| `500 Internal Error` | `internal_failure` | Unexpected model crash or GPU driver fault. |

---

## 4. Python Client Usage

```python
from client import VLMClient

client = VLMClient(base_url="http://127.0.0.1:8000")

# 1. Health
print(client.get_health())

# 2. Chat with Image
resp = client.chat_image(
    image="path/to/invoice.jpg",
    prompt="Đọc tổng tiền trên hóa đơn.",
    model="qwen3-vl-2b"
)
print(resp.choices[0].message.content)

# 3. Structured Extraction
data = client.extract(
    image="path/to/invoice.jpg",
    schema_name="invoice"
)
print(data.data)
```
