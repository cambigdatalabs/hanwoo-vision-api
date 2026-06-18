from __future__ import annotations

import os
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parents[1]

MODELS_DIR = Path(os.getenv("HANWOO_MODELS_DIR", PROJECT_ROOT / "models"))
STORAGE_DIR = Path(
    os.getenv("HANWOO_STORAGE_DIR", PROJECT_ROOT / "storage" / "matching")
)

MATCHING_MODEL_PATH = Path(
    os.getenv("MATCHING_MODEL_PATH", MODELS_DIR / "matching" / "encoder.pt")
)
U2NET_HOME = Path(os.getenv("U2NET_HOME", MODELS_DIR / "u2net"))

DEVICE = os.getenv("HANWOO_DEVICE", "auto")
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))

GALLERY_DIR = Path(os.getenv("HANWOO_GALLERY_DIR", STORAGE_DIR / "gallery_images"))

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "hanwoo_matching_gallery")
