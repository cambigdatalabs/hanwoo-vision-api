"""Re-calibrate the anomaly threshold on a validation set without rebuilding the memory bank.

Usage
-----
    python scripts/calibrate_threshold.py \
        --bank       models/anomaly/memory_bank.pth \
        --val-dir    data/val \
        --output     models/anomaly/threshold.json \
        --percentile 87 \
        --device     auto
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hanwoo.core.config import ANOMALY_IMAGE_SIZE
from hanwoo.core.encoders.dinov2 import DINOv2Extractor
from hanwoo.core.vectorstore.memory_bank import MemoryBank
from hanwoo.services.matching.pipeline import choose_device


_TRANSFORM = transforms.Compose([
    transforms.Resize(ANOMALY_IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class NormalImageDataset(Dataset):
    def __init__(self, root: str, label: str = "") -> None:
        self.root = Path(root)
        self.files = sorted(
            f for f in os.listdir(root)
            if not f.startswith("._")
            and "_before" in f.lower()
            and Path(f).suffix.lower() in _IMAGE_EXTS
        )
        print(f"  [{label or root}] {len(self.files)} 장 로드 (_before만)")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.root / self.files[idx]).convert("RGB")
        return _TRANSFORM(img)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate anomaly threshold")
    parser.add_argument("--bank",        required=True)
    parser.add_argument("--val-dir",     required=True)
    parser.add_argument("--output",      default="models/anomaly/threshold.json")
    parser.add_argument("--percentile",  type=int, default=87)
    parser.add_argument("--device",      default="auto")
    parser.add_argument("--batch-size",  type=int, default=8)
    parser.add_argument("--workers",     type=int, default=4)
    args = parser.parse_args()

    device = choose_device(args.device)
    print(f"장치: {device}")

    print("\nDINOv2 로드 중...")
    extractor = DINOv2Extractor().to(device).half()
    extractor.eval()

    print(f"메모리뱅크 로드: {args.bank}")
    bank = MemoryBank(device=device)
    bank.load(Path(args.bank))
    print(f"  뱅크 크기: {bank.size:,} 패치  /  기존 임계값: {bank.threshold}")

    val_ds = NormalImageDataset(args.val_dir, label="val")
    if len(val_ds) == 0:
        print("❌ '_before' 이미지 없음. 종료.")
        return

    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers, pin_memory=True,
    )

    print(f"\n정상 점수 계산 중 (percentile={args.percentile})...")
    val_scores: list[float] = []
    with torch.no_grad():
        for imgs in tqdm(val_loader):
            pf = extractor(imgs.to(device))
            scores, _ = bank.predict(pf)
            val_scores.extend(scores.tolist())

    threshold = float(np.percentile(val_scores, args.percentile))
    print(
        f"  점수 — min: {min(val_scores):.4f} / "
        f"mean: {np.mean(val_scores):.4f} / "
        f"max: {max(val_scores):.4f}"
    )
    print(f"  새 임계값 (percentile {args.percentile}): {threshold:.4f}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"threshold": threshold, "percentile": args.percentile}, f, indent=2)
    print(f"\n임계값 저장: {out_path}")
    print("✅ 완료!")


if __name__ == "__main__":
    main()
