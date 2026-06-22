# Hanwoo Vision API Endpoints

Matching base URL:

```text
http://localhost:8888
```

Anomaly base URL:

```text
http://localhost:8889
```

Current implemented services:

- Matching gallery enrollment/query on host port `8888`.
- Anomaly detection inference on host port `8889`.

Gallery data is scoped by `lot_id`. Images are stored under
`storage/matching/gallery_images/{lot_id}/{capture_date}`. Embeddings and
metadata are stored in Qdrant collection `hanwoo_matching_gallery`.

## Common Params

| Name | Location | Required | Type | Description |
| --- | --- | --- | --- | --- |
| `lot_id` | form/query/body | Yes for enroll, match, delete, clear | string | Lot identifier used to separate gallery pools. Same filename can exist in different lots. |
| `capture_date` | form/query/body | No | string, `YYYY-MM-DD` | Narrows gallery scope by date. During upload, defaults to server date when omitted. |
| `preprocess` | form/query/body | No | boolean | Applies background removal, tilt correction, and crop. |
| `top_k` | query | No | integer, 1-50 | Number of nearest matches to return. Defaults to server config. |

## GET `/health`

Checks service status and model/device state.

### Params

None.

### Example

```bash
curl "http://localhost:8888/health"
```

### Response Example

```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda",
  "storage_dir": "/app/storage/matching"
}
```

## GET `/metadata`

Returns matching checkpoint and model metadata.

### Params

None.

### Example

```bash
curl "http://localhost:8888/metadata"
```

### Response Example

```json
{
  "checkpoint_path": "/app/models/matching/encoder.pt",
  "architecture": "SiameseViT",
  "backbone": "swin",
  "embedding_dim": 256,
  "image_size": 224,
  "epoch": 10,
  "metrics": {}
}
```

## GET `/gallery/images`

Lists enrolled gallery images. Can list all images or filter by lot/date.

### Params

| Name | Location | Required | Type | Description |
| --- | --- | --- | --- | --- |
| `lot_id` | query | No | string | Return images only from this lot. |
| `capture_date` | query | No | string, `YYYY-MM-DD` | Return images only from this date. |

### Examples

List all gallery images:

```bash
curl "http://localhost:8888/gallery/images"
```

List one lot:

```bash
curl "http://localhost:8888/gallery/images?lot_id=LOT-001"
```

List one lot and date:

```bash
curl "http://localhost:8888/gallery/images?lot_id=LOT-001&capture_date=2026-06-22"
```

### Response Example

```json
{
  "count": 1,
  "filenames": ["before_packaging"],
  "images": [
    {
      "name": "before_packaging",
      "lot_id": "LOT-001",
      "capture_date": "2026-06-22",
      "image_path": "/app/storage/matching/gallery_images/LOT-001/2026-06-22/before_packaging.png",
      "original_filename": "before_packaging.jpg",
      "preprocessed": false,
      "created_at": "2026-06-22T01:23:45+00:00"
    }
  ]
}
```

## POST `/gallery/images`

Uploads one or more images into a lot-scoped gallery. Each image creates two
Qdrant vectors: original orientation and rotated orientation.

### Params

Multipart form-data.

| Name | Location | Required | Type | Description |
| --- | --- | --- | --- | --- |
| `files` | form file | Yes | one or more image files | Images to enroll. |
| `lot_id` | form | Yes | string | Lot identifier used to scope matching. |
| `capture_date` | form | No | string, `YYYY-MM-DD` | Date folder and metadata. Defaults to server date. |
| `preprocess` | form | No | boolean | Defaults to `true`. |

### Example

```bash
curl -X POST "http://localhost:8888/gallery/images" \
  -F "lot_id=LOT-001" \
  -F "capture_date=2026-06-22" \
  -F "preprocess=false" \
  -F "files=@before_packaging.jpg"
```

Multiple files:

```bash
curl -X POST "http://localhost:8888/gallery/images" \
  -F "lot_id=LOT-001" \
  -F "capture_date=2026-06-22" \
  -F "preprocess=true" \
  -F "files=@image_1.jpg" \
  -F "files=@image_2.jpg"
```

### Response Example

```json
{
  "added": [
    {
      "name": "before_packaging",
      "lot_id": "LOT-001",
      "capture_date": "2026-06-22",
      "path": "/app/storage/matching/gallery_images/LOT-001/2026-06-22/before_packaging.png"
    }
  ],
  "count": 1
}
```

## POST `/gallery/import-directory`

Imports every file in a server-side directory into a lot-scoped gallery.

Important: `directory` must exist inside the API container or be mounted into it.

### Params

JSON body.

| Name | Location | Required | Type | Description |
| --- | --- | --- | --- | --- |
| `directory` | body | Yes | string | Directory path visible to API container. |
| `lot_id` | body | Yes | string | Lot identifier used to scope matching. |
| `capture_date` | body | No | string, `YYYY-MM-DD` | Date folder and metadata. Defaults to server date. |
| `preprocess` | body | No | boolean | Defaults to `true`. |

### Example

```bash
curl -X POST "http://localhost:8888/gallery/import-directory" \
  -H "Content-Type: application/json" \
  -d '{
    "directory": "/app/data/lot_001",
    "lot_id": "LOT-001",
    "capture_date": "2026-06-22",
    "preprocess": true
  }'
```

### Response Example

```json
{
  "added": [
    {
      "name": "image_1",
      "lot_id": "LOT-001",
      "capture_date": "2026-06-22",
      "path": "/app/storage/matching/gallery_images/LOT-001/2026-06-22/image_1.png"
    }
  ],
  "skipped": []
}
```

## DELETE `/gallery/images/{name}`

Deletes one enrolled image from one lot. If `capture_date` is omitted, deletes
matching `name` entries across all dates in that lot.

### Params

| Name | Location | Required | Type | Description |
| --- | --- | --- | --- | --- |
| `name` | path | Yes | string | Sanitized image name returned by enroll/list. No extension. |
| `lot_id` | query | Yes | string | Lot identifier used to scope deletion. |
| `capture_date` | query | No | string, `YYYY-MM-DD` | Delete only this date. |

### Examples

Delete from a lot:

```bash
curl -X DELETE "http://localhost:8888/gallery/images/before_packaging?lot_id=LOT-001"
```

Delete from a lot/date:

```bash
curl -X DELETE "http://localhost:8888/gallery/images/before_packaging?lot_id=LOT-001&capture_date=2026-06-22"
```

### Response Example

```json
{
  "removed": "before_packaging",
  "lot_id": "LOT-001",
  "capture_date": "2026-06-22"
}
```

## DELETE `/gallery/images`

Clears a scoped gallery. `lot_id` is required. If `capture_date` is supplied,
only that date is cleared; otherwise the whole lot is cleared.

### Params

| Name | Location | Required | Type | Description |
| --- | --- | --- | --- | --- |
| `lot_id` | query | Yes | string | Lot identifier used to scope deletion. |
| `capture_date` | query | No | string, `YYYY-MM-DD` | Clear only this date. |

### Examples

Clear one lot:

```bash
curl -X DELETE "http://localhost:8888/gallery/images?lot_id=LOT-001"
```

Clear one date in one lot:

```bash
curl -X DELETE "http://localhost:8888/gallery/images?lot_id=LOT-001&capture_date=2026-06-22"
```

### Response Example

```json
{
  "removed_count": 12,
  "lot_id": "LOT-001",
  "capture_date": null
}
```

## POST `/match`

Matches a query image against the gallery for the requested lot. If
`capture_date` is supplied, matching is restricted to that lot/date.

### Params

Multipart form-data plus query params.

| Name | Location | Required | Type | Description |
| --- | --- | --- | --- | --- |
| `file` | form file | Yes | image file | Query image. |
| `lot_id` | query | Yes | string | Lot identifier used to scope matching. |
| `capture_date` | query | No | string, `YYYY-MM-DD` | Search only this date. |
| `top_k` | query | No | integer, 1-50 | Number of matches to return. |
| `preprocess` | query | No | boolean | Defaults to `false`. |

### Examples

Match against one lot:

```bash
curl -X POST "http://localhost:8888/match?lot_id=LOT-001&top_k=5&preprocess=false" \
  -F "file=@after_packaging.jpg"
```

Match against one lot/date:

```bash
curl -X POST "http://localhost:8888/match?lot_id=LOT-001&capture_date=2026-06-22&top_k=5&preprocess=false" \
  -F "file=@after_packaging.jpg"
```

### Response Example

```json
{
  "query_file": "after_packaging.jpg",
  "lot_id": "LOT-001",
  "capture_date": "2026-06-22",
  "top_k": 1,
  "preprocess": false,
  "matches": [
    {
      "rank": 1,
      "name": "before_packaging",
      "lot_id": "LOT-001",
      "capture_date": "2026-06-22",
      "distance": 0.0,
      "similarity": 100.0,
      "image_path": "/app/storage/matching/gallery_images/LOT-001/2026-06-22/before_packaging.png",
      "matched_variant": "original"
    }
  ]
}
```

## Error Responses

Missing or invalid params usually return `422`.

```json
{
  "detail": "lot_id is required"
}
```

Invalid image uploads return `400`.

```json
{
  "detail": "Invalid image: cannot identify image file"
}
```

Empty matching scope returns `404`.

```json
{
  "detail": "Gallery scope is empty"
}
```

# Anomaly Endpoints

Anomaly service runs separately on host port `8889`.

```text
http://localhost:8889
```

The anomaly service requires `models/anomaly/memory_bank.pth`. If the file is
missing or invalid, the container stays up and `/health` returns `not_loaded`.

## GET `/health`

Checks anomaly memory bank, threshold, and runtime device.

### Example

```bash
curl "http://localhost:8889/health"
```

### Response Example

```json
{
  "status": "healthy",
  "bank_loaded": true,
  "bank_size": 92381,
  "threshold": 31.5798974609375,
  "device": "cuda"
}
```

Not loaded example:

```json
{
  "status": "not_loaded",
  "bank_loaded": false,
  "bank_size": 0,
  "threshold": null,
  "device": "cuda"
}
```

## POST `/infer`

Runs anomaly detection on one uploaded image.

### Params

Multipart form-data plus query params.

| Name | Location | Required | Type | Description |
| --- | --- | --- | --- | --- |
| `file` | form file | Yes | image file | Hanwoo image to inspect. |
| `preprocess` | query | No | boolean | Defaults to `true`. Applies background removal, tilt correction, and crop. |

### Example

```bash
curl -X POST "http://localhost:8889/infer?preprocess=true" \
  -F "file=@sample.jpg"
```

### Response Example

```json
{
  "filename": "sample.jpg",
  "preprocess": true,
  "preprocess_ms": 120.4,
  "infer_ms": 85.2,
  "total_ms": 205.6,
  "anomaly_score": 64.5881,
  "is_anomaly": true,
  "threshold": 31.5799,
  "regions": ["상단 중앙"],
  "heatmap_b64": "..."
}
```

## GET `/threshold`

Returns the active anomaly threshold.

### Example

```bash
curl "http://localhost:8889/threshold"
```

### Response Example

```json
{
  "threshold": 31.5798974609375
}
```

## PUT `/threshold`

Updates and persists the anomaly threshold.

### Params

JSON body.

| Name | Location | Required | Type | Description |
| --- | --- | --- | --- | --- |
| `threshold` | body | Yes | float | Must be greater than `0`. |

### Example

```bash
curl -X PUT "http://localhost:8889/threshold" \
  -H "Content-Type: application/json" \
  -d '{"threshold": 31.5798974609375}'
```

### Response Example

```json
{
  "threshold": 31.5798974609375,
  "updated": true
}
```

## POST `/evaluate`

Evaluates anomaly performance against server-side test folders. This endpoint
expects images and labels to exist inside the API container.

### Params

JSON body.

| Name | Location | Required | Type | Description |
| --- | --- | --- | --- | --- |
| `test_base_dir` | body | No | string | Base directory for anomaly category folders. Defaults to `/app/data/test`. |
| `category_dirs` | body | No | string array | Category folders under `test_base_dir`. |
| `images2_dir` | body | No | string | Normal images folder. Defaults to `/app/data/test/images2`. |
| `preprocess` | body | No | boolean | Defaults to `true`. |

### Example

```bash
curl -X POST "http://localhost:8889/evaluate" \
  -H "Content-Type: application/json" \
  -d '{
    "test_base_dir": "/app/data/test",
    "category_dirs": ["비닐", "뼈", "실", "정맥혈응고체", "천"],
    "images2_dir": "/app/data/test/images2",
    "preprocess": true
  }'
```
