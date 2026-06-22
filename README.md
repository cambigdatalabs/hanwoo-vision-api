# Hanwoo Vision API

FastAPI service for Hanwoo image matching workflows. The current implemented
service manages a lot-scoped gallery, stores gallery embeddings in Qdrant, and
matches a query image against the requested lot.

Full endpoint reference with parameter tables and curl examples:
[API_ENDPOINTS.md](API_ENDPOINTS.md)

## Current Status

- Matching service is implemented.
- Qdrant is used as the embedding database.
- Gallery images are stored on disk under a lot/date folder structure.
- GPU runtime is supported through Docker Compose GPU override.
- `HANWOO_DEVICE=auto` uses CUDA when available and falls back to CPU when CUDA
  is not detected.
- Anomaly service folders exist as scaffold, but the active API surface is the
  matching service.

## Architecture

```text
Client
  |
  v
FastAPI matching service
  |
  +-- preprocessing: optional background removal, tilt correction, crop
  |
  +-- encoder: matching model -> 256-d embedding
  |
  +-- Qdrant: vectors and metadata
  |
  +-- disk storage: gallery images
```

Gallery separation is handled with Qdrant payload filters:

- `lot_id`: required for enroll, match, delete, and clear.
- `capture_date`: optional `YYYY-MM-DD`; defaults to server date on enroll.
- `name`: sanitized image stem.
- `variant`: `original` or rotated variant.

Images are stored at:

```text
storage/matching/gallery_images/{lot_id}/{capture_date}/{name}.png
```

Embeddings and metadata are stored in Qdrant collection:

```text
hanwoo_matching_gallery
```

## Repository Layout

```text
hanwoo-vision-api/
├── docker-compose.yml
├── docker-compose.gpu.yml
├── .env.example
├── pyproject.toml
├── Dockerfile
├── README.md
├── API_ENDPOINTS.md
├── src/hanwoo/
│   ├── core/
│   │   ├── config.py
│   │   ├── preprocessing.py
│   │   ├── encoders/
│   │   ├── vectorstore/
│   │   ├── models.py
│   │   ├── schemas.py
│   │   ├── telemetry.py
│   │   └── gpu.py
│   └── services/
│       ├── matching/
│       └── anomaly/
├── models/
├── gateway/
├── scripts/
└── tests/
```

## Requirements

Base runtime:

- Docker
- Docker Compose
- Matching checkpoint at `models/matching/encoder.pt`
- U2NET model files under `models/u2net`

GPU runtime:

- NVIDIA GPU
- NVIDIA driver
- NVIDIA Container Toolkit
- Docker Compose with GPU support

## Configuration

Copy the example file when running outside Docker Compose defaults:

```bash
cp .env.example .env
```

Environment variables:

| Name | Default | Description |
| --- | --- | --- |
| `HANWOO_DEVICE` | `auto` | `auto`, `cuda`, or `cpu`. `auto` prefers CUDA and falls back to CPU. |
| `MATCHING_MODEL_PATH` | `/app/models/matching/encoder.pt` | Matching model checkpoint path. |
| `HANWOO_STORAGE_DIR` | `/app/storage/matching` | Runtime storage directory. |
| `U2NET_HOME` | `/app/models/u2net` | Background removal model directory. |
| `DEFAULT_TOP_K` | `5` | Default match result count. |
| `QDRANT_URL` | `http://qdrant:6333` | Qdrant service URL. |
| `QDRANT_COLLECTION` | `hanwoo_matching_gallery` | Qdrant collection name. |

## Model Files

Model weights are not committed to Git. Put them here before running:

```text
models/
├── matching/
│   └── encoder.pt
├── u2net/
│   └── u2net.onnx
└── anomaly/
    ├── memory_bank.faiss
    └── threshold.json
```

The matching service needs `models/matching/encoder.pt`. Background removal
needs U2NET files under `models/u2net`.

## Run With Docker

CPU or automatic fallback mode:

```bash
docker compose up -d --build
```

GPU mode:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

Check containers:

```bash
docker compose ps
```

Check logs:

```bash
docker compose logs -f matching
```

Stop:

```bash
docker compose down
```

## Verify Runtime

Health check:

```bash
curl "http://localhost:8000/health"
```

Expected GPU response includes:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda"
}
```

If no GPU is available and `HANWOO_DEVICE=auto`, the service runs on CPU:

```json
{
  "device": "cpu"
}
```

If `HANWOO_DEVICE=cuda` is set and CUDA is unavailable, startup fails instead
of silently using CPU.

## API Summary

Base URL:

```text
http://localhost:8000
```

Implemented endpoints:

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Check service, model, and device status. |
| `GET` | `/metadata` | Return matching checkpoint/model metadata. |
| `GET` | `/gallery/images` | List gallery images, optionally by lot/date. |
| `POST` | `/gallery/images` | Upload images into a lot-scoped gallery. |
| `POST` | `/gallery/import-directory` | Import server-side directory into a lot-scoped gallery. |
| `DELETE` | `/gallery/images/{name}` | Delete one image by name inside a lot/date scope. |
| `DELETE` | `/gallery/images` | Clear a lot or lot/date gallery scope. |
| `POST` | `/match` | Match one query image against a lot/date gallery scope. |

Common parameters:

| Name | Required | Used By | Description |
| --- | --- | --- | --- |
| `lot_id` | Yes for upload, import, match, delete, clear | Gallery and matching endpoints | Separates gallery pools by production lot. |
| `capture_date` | No | Gallery and matching endpoints | Narrows a lot to one date. Format: `YYYY-MM-DD`. |
| `preprocess` | No | Upload, import, match | Enables background removal and normalization. |
| `top_k` | No | Match | Number of nearest matches to return. |

## API Examples

Upload one image:

```bash
curl -X POST "http://localhost:8000/gallery/images" \
  -F "lot_id=LOT-001" \
  -F "capture_date=2026-06-22" \
  -F "preprocess=true" \
  -F "files=@before_packaging.jpg"
```

Upload multiple images:

```bash
curl -X POST "http://localhost:8000/gallery/images" \
  -F "lot_id=LOT-001" \
  -F "capture_date=2026-06-22" \
  -F "files=@image_1.jpg" \
  -F "files=@image_2.jpg"
```

List one lot:

```bash
curl "http://localhost:8000/gallery/images?lot_id=LOT-001"
```

List one lot/date:

```bash
curl "http://localhost:8000/gallery/images?lot_id=LOT-001&capture_date=2026-06-22"
```

Match a query image against one lot:

```bash
curl -X POST "http://localhost:8000/match?lot_id=LOT-001&top_k=5" \
  -F "file=@after_packaging.jpg"
```

Match a query image against one lot/date:

```bash
curl -X POST "http://localhost:8000/match?lot_id=LOT-001&capture_date=2026-06-22&top_k=5" \
  -F "file=@after_packaging.jpg"
```

Delete one image from one lot:

```bash
curl -X DELETE "http://localhost:8000/gallery/images/before_packaging?lot_id=LOT-001"
```

Clear one lot/date:

```bash
curl -X DELETE "http://localhost:8000/gallery/images?lot_id=LOT-001&capture_date=2026-06-22"
```

More examples and response bodies are in [API_ENDPOINTS.md](API_ENDPOINTS.md).

## Data Organization Strategy

Use `lot_id` as the primary gallery partition. Use `capture_date` as a secondary
filter when a lot spans multiple production days.

Recommended naming:

```text
lot_id = LOT-20260622-A
capture_date = 2026-06-22
```

This lets the same image name exist in different lots without collision:

```text
LOT-001/2026-06-22/tray_001.png
LOT-002/2026-06-22/tray_001.png
```

Matching always searches only the requested `lot_id`. Add `capture_date` when
the query must be restricted to one production day.

## Development

Install package in editable mode:

```bash
pip install -e .
```

Run service locally:

```bash
uvicorn hanwoo.services.matching.main:app --host 0.0.0.0 --port 8000
```

Compile check:

```bash
python3 -m compileall src/hanwoo
```

## Troubleshooting

Qdrant not reachable:

```bash
docker compose ps qdrant
docker compose logs qdrant
```

Model missing:

```text
FileNotFoundError: /app/models/matching/encoder.pt
```

Fix by placing the checkpoint at `models/matching/encoder.pt`.

CUDA not used:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
curl "http://localhost:8000/health"
```

If health still reports `cpu`, verify NVIDIA runtime on the host:

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

## 한국어 안내

Hanwoo Vision API는 한우 이미지 매칭을 위한 FastAPI 서비스입니다. 현재 구현된
기능은 로트별 갤러리 등록, Qdrant 기반 임베딩 검색, 쿼리 이미지 매칭입니다.

전체 엔드포인트 파라미터와 curl 예시는 [API_ENDPOINTS.md](API_ENDPOINTS.md)에
정리되어 있습니다.

## 현재 구현 상태

- 매칭 서비스가 구현되어 있습니다.
- 임베딩 DB로 Qdrant를 사용합니다.
- 갤러리 원본 이미지는 로컬 디스크에 `lot_id/capture_date` 구조로 저장합니다.
- Docker Compose GPU override로 GPU 실행을 지원합니다.
- `HANWOO_DEVICE=auto`이면 CUDA 사용 가능 시 GPU를 사용하고, 없으면 CPU로
  fallback합니다.
- anomaly 서비스 폴더는 scaffold 상태이며, 현재 활성 API는 matching 서비스입니다.

## 데이터 저장 구조

이미지 파일:

```text
storage/matching/gallery_images/{lot_id}/{capture_date}/{name}.png
```

Qdrant 컬렉션:

```text
hanwoo_matching_gallery
```

Qdrant payload 주요 필드:

| 필드 | 설명 |
| --- | --- |
| `lot_id` | 생산 로트 ID. 갤러리를 분리하는 핵심 값입니다. |
| `capture_date` | 촬영일 또는 생산일. `YYYY-MM-DD` 형식입니다. |
| `name` | 이미지 파일명에서 확장자를 제거한 값입니다. |
| `variant` | 원본 방향 또는 회전 방향 임베딩 구분입니다. |

## 실행 방법

CPU 또는 자동 fallback 모드:

```bash
docker compose up -d --build
```

GPU 모드:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

상태 확인:

```bash
curl "http://localhost:8000/health"
```

GPU 사용 중이면 응답에 `"device": "cuda"`가 포함됩니다.

## 설정값

| 이름 | 기본값 | 설명 |
| --- | --- | --- |
| `HANWOO_DEVICE` | `auto` | `auto`, `cuda`, `cpu`. `auto`는 GPU 우선, 없으면 CPU입니다. |
| `MATCHING_MODEL_PATH` | `/app/models/matching/encoder.pt` | 매칭 모델 checkpoint 경로입니다. |
| `HANWOO_STORAGE_DIR` | `/app/storage/matching` | 런타임 저장소 경로입니다. |
| `U2NET_HOME` | `/app/models/u2net` | 배경 제거 모델 경로입니다. |
| `DEFAULT_TOP_K` | `5` | 기본 매칭 결과 개수입니다. |
| `QDRANT_URL` | `http://qdrant:6333` | Qdrant 접속 URL입니다. |
| `QDRANT_COLLECTION` | `hanwoo_matching_gallery` | Qdrant 컬렉션 이름입니다. |

## 모델 파일

모델 파일은 Git에 포함하지 않습니다. 실행 전 아래 위치에 배치해야 합니다.

```text
models/
├── matching/
│   └── encoder.pt
├── u2net/
│   └── u2net.onnx
└── anomaly/
    ├── memory_bank.faiss
    └── threshold.json
```

매칭 서비스는 `models/matching/encoder.pt`가 필요합니다. 전처리의 배경 제거를
사용하려면 `models/u2net` 아래에 U2NET 모델이 필요합니다.

## API 요약

기본 URL:

```text
http://localhost:8000
```

| Method | Path | 설명 |
| --- | --- | --- |
| `GET` | `/health` | 서비스, 모델, device 상태를 확인합니다. |
| `GET` | `/metadata` | 매칭 모델 checkpoint와 metadata를 반환합니다. |
| `GET` | `/gallery/images` | 갤러리 이미지를 조회합니다. lot/date 필터 가능. |
| `POST` | `/gallery/images` | 이미지를 특정 lot 갤러리에 업로드합니다. |
| `POST` | `/gallery/import-directory` | 서버 내부 디렉터리 이미지를 lot 갤러리에 등록합니다. |
| `DELETE` | `/gallery/images/{name}` | 특정 lot/date 안의 이미지 1개를 삭제합니다. |
| `DELETE` | `/gallery/images` | 특정 lot 또는 lot/date 갤러리를 비웁니다. |
| `POST` | `/match` | 쿼리 이미지를 특정 lot/date 갤러리와 매칭합니다. |

주요 파라미터:

| 이름 | 필수 여부 | 설명 |
| --- | --- | --- |
| `lot_id` | 등록, 매칭, 삭제에서 필수 | 갤러리를 로트별로 분리합니다. |
| `capture_date` | 선택 | 날짜 단위로 추가 필터링합니다. 형식은 `YYYY-MM-DD`. |
| `preprocess` | 선택 | 배경 제거, tilt 보정, crop 전처리를 수행합니다. |
| `top_k` | 선택 | 반환할 매칭 결과 개수입니다. |

## API 예시

이미지 1개 등록:

```bash
curl -X POST "http://localhost:8000/gallery/images" \
  -F "lot_id=LOT-001" \
  -F "capture_date=2026-06-22" \
  -F "preprocess=true" \
  -F "files=@before_packaging.jpg"
```

로트별 이미지 목록 조회:

```bash
curl "http://localhost:8000/gallery/images?lot_id=LOT-001"
```

로트와 날짜 기준 조회:

```bash
curl "http://localhost:8000/gallery/images?lot_id=LOT-001&capture_date=2026-06-22"
```

이미지 매칭:

```bash
curl -X POST "http://localhost:8000/match?lot_id=LOT-001&capture_date=2026-06-22&top_k=5" \
  -F "file=@after_packaging.jpg"
```

이미지 1개 삭제:

```bash
curl -X DELETE "http://localhost:8000/gallery/images/before_packaging?lot_id=LOT-001"
```

특정 로트/날짜 전체 삭제:

```bash
curl -X DELETE "http://localhost:8000/gallery/images?lot_id=LOT-001&capture_date=2026-06-22"
```

## 로트 관리 권장 방식

`lot_id`를 갤러리 분리의 기본 단위로 사용하세요. 같은 날짜에 여러 생산 로트가
있으면 로트별로 별도 `lot_id`를 부여합니다.

권장 예시:

```text
lot_id = LOT-20260622-A
capture_date = 2026-06-22
```

같은 파일명이라도 서로 다른 lot에 안전하게 저장됩니다.

```text
LOT-001/2026-06-22/tray_001.png
LOT-002/2026-06-22/tray_001.png
```

매칭은 항상 요청한 `lot_id` 안에서만 수행됩니다. 날짜까지 제한해야 하면
`capture_date`를 함께 전달하세요.

## 문제 해결

Qdrant 상태 확인:

```bash
docker compose ps qdrant
docker compose logs qdrant
```

매칭 모델 파일 누락 시:

```text
FileNotFoundError: /app/models/matching/encoder.pt
```

`models/matching/encoder.pt` 위치에 checkpoint를 넣으세요.

CUDA 확인:

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```
