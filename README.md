# Hanwoo Vision API

FastAPI services for Hanwoo vision workflows.

Current implemented service:

- Matching: gallery enrollment, Qdrant-backed vector search, top-k matching.

## Run

CPU/fallback mode:

```bash
docker compose up -d
```

GPU mode:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

`HANWOO_DEVICE=auto` uses CUDA when PyTorch detects a GPU and falls back to CPU
when no GPU is available.

## Runtime Files

Model weights are not committed. Mount them under:

```text
models/matching/encoder.pt
models/u2net/u2net.onnx
models/anomaly/memory_bank.faiss
models/anomaly/threshold.json
```

Gallery images are stored on disk in `storage/matching/gallery_images`.
Embeddings and metadata are stored in Qdrant collection
`hanwoo_matching_gallery`.

## Endpoints

- `GET /health`
- `GET /metadata`
- `GET /gallery/images`
- `POST /gallery/images`
- `POST /gallery/import-directory`
- `DELETE /gallery/images/{name}`
- `DELETE /gallery/images`
- `POST /match`

`preprocess=true` enables background removal, tilt correction, and crop before
gallery insert or matching.
