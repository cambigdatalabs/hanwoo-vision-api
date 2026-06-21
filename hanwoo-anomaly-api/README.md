# Hanwoo Anomaly API

한우 이물질 이상탐지 API입니다. DINOv2 + PatchCore 기반으로 이물질(비닐, 뼈, 실, 정맥혈응고체, 천)을 탐지합니다.

## 구조

```
hanwoo-anomaly-api/
├── src/hanwoo/
│   ├── core/
│   │   ├── config.py            # 환경변수 설정
│   │   ├── gpu.py               # 디바이스 선택
│   │   ├── preprocessing.py     # 배경제거 + 기울기보정 + 크롭
│   │   ├── encoders/dinov2.py   # DINOv2 ViT-B/14 특징 추출
│   │   └── vectorstore/memory_bank.py  # PatchCore 메모리뱅크
│   └── services/anomaly/
│       ├── main.py              # FastAPI 앱
│       ├── pipeline.py          # 추론 로직
│       └── routes.py            # /infer, /threshold, /evaluate
├── models/
│   ├── anomaly/                 # memory_bank.pth 위치
│   └── u2net/                   # u2net 모델 자동 다운로드
├── scripts/
│   ├── build_memory_bank.py     # 메모리뱅크 구축
│   └── calibrate_threshold.py   # 임계값 보정
└── .env.example
```

## 설치

```bash
pip install -e .
```

## 모델 준비

`models/anomaly/memory_bank.pth`에 학습된 메모리뱅크를 복사합니다.


## 실행

```bash
uvicorn hanwoo.services.anomaly.main:app --host 0.0.0.0 --port 8001
```

## 환경변수

`.env.example`을 복사해 `.env`로 사용합니다.

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `HANWOO_DEVICE` | `auto` | `auto` / `cuda` / `cpu` |
| `ANOMALY_MODEL_PATH` | `models/anomaly/memory_bank.pth` | 메모리뱅크 경로 |
| `ANOMALY_THRESHOLD_PATH` | `models/anomaly/threshold.json` | 임계값 경로 |
| `U2NET_HOME` | `models/u2net` | u2net 모델 경로 |

## API 엔드포인트

### `GET /health`
서버 상태 확인

### `POST /infer`
이미지 1장 이상탐지

| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `file` | image | - | 검사할 이미지 |
| `preprocess` | bool | `true` | 배경제거 + 기울기보정 + 크롭 적용 여부 |

**응답 예시**
```json
{
  "filename": "sample.jpg",
  "preprocess": true,
  "anomaly_score": 30.1276,
  "is_anomaly": false,
  "threshold": 31.5799,
  "regions": ["이상 없음"],
  "heatmap_b64": "..."
}
```

### `GET /threshold`
현재 임계값 조회

### `PUT /threshold`
임계값 수동 변경

```json
{ "threshold": 31.5799 }
```

### `POST /evaluate`
테스트 데이터셋 전체 성능평가

```json
{
  "test_base_dir": "benchmark_v3/test",
  "category_dirs": ["비닐", "뼈", "실", "정맥혈응고체", "천"],
  "images2_dir": "benchmark_v3/test/images2",
  "preprocess": true
}
```

테스트 데이터 폴더 구조:
```
benchmark_v3/test/
├── 비닐/
│   ├── images/   ← 이상 이미지
│   └── labels/   ← 마스크 (이미지명_mask.png)
├── 뼈/
│   ├── images/
│   └── labels/
└── images2/      ← 정상 이미지 (라벨 없음)
```

## 전처리 파이프라인

`preprocess=true` 시 아래 순서로 전처리를 적용합니다.

1. **배경제거** — rembg u2net + 마스크 정제 + 어두운 트레이 복원
2. **기울기보정** — Hough Line 검출로 상단 수평선 각도 보정
3. **크롭** — 전경 영역 기준 자동 크롭

## Swagger UI

서버 실행 후 `http://localhost:8001/docs`에서 API를 테스트할 수 있습니다.
