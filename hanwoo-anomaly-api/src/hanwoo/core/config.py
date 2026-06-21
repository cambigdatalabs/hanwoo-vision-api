from __future__ import annotations

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parents[1]

MODELS_DIR = Path(os.getenv("HANWOO_MODELS_DIR", PROJECT_ROOT / "models"))

U2NET_HOME = Path(os.getenv("U2NET_HOME", MODELS_DIR / "u2net"))

DEVICE = os.getenv("HANWOO_DEVICE", "auto")

# ── Anomaly ───────────────────────────────────────────────────────────────────
ANOMALY_MODEL_PATH = Path(
    os.getenv("ANOMALY_MODEL_PATH", MODELS_DIR / "anomaly" / "memory_bank.pth")
)
ANOMALY_THRESHOLD_PATH = Path(
    os.getenv("ANOMALY_THRESHOLD_PATH", MODELS_DIR / "anomaly" / "threshold.json")
)
ANOMALY_DINO_LAYERS: list[int] = [
    int(x) for x in os.getenv("ANOMALY_DINO_LAYERS", "10,11").split(",")
]
ANOMALY_CORESET_RATIO: float = float(os.getenv("ANOMALY_CORESET_RATIO", "0.08"))
ANOMALY_K_NEIGHBORS: int = int(os.getenv("ANOMALY_K_NEIGHBORS", "3"))
ANOMALY_TOP_K_RATIO: float = float(os.getenv("ANOMALY_TOP_K_RATIO", "0.4"))
ANOMALY_THRESH_PERCENTILE: int = int(os.getenv("ANOMALY_THRESH_PERCENTILE", "87"))
ANOMALY_IMAGE_SIZE: tuple[int, int] = (672, 672)
ANOMALY_HEATMAP_POW: float = float(os.getenv("ANOMALY_HEATMAP_POW", "2.5"))
