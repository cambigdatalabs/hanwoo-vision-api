"""Build the PatchCore memory bank from normal (before-packaging) training images.

Usage
-----
    python scripts/build_memory_bank.py \
        --train-dir  data/train \
        --val-dir    data/val \
        --output     models/anomaly/memory_bank.pth \
        --threshold  models/anomaly/threshold.json \
        --device     auto \
        --batch-size 8 \
        --workers    4

Only images whose filename contains '_before' (case-insensitive) are used,
mirroring the original training convention.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

# Allow running from repo root without installing the package
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hanwoo.core.config import (
    ANOMALY_CORESET_RATIO,
    ANOMALY_IMAGE_SIZE,
    ANOMALY_THRESH_PERCENTILE,
)
from hanwoo.core.encoders.dinov2 import DINOv2Extractor
from hanwoo.core.vectorstore.memory_bank import MemoryBank
from hanwoo.core.gpu import choose_device


_TRANSFORM = transforms.Compose([
    transforms.Resize(ANOMALY_IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class NormalImageDataset(Dataset):
    """Loads '_before' images from a flat directory."""

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
        path = self.root / self.files[idx]
        img = Image.open(path).convert("RGB")
        return _TRANSFORM(img)


def seed_all(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def extract_patch_features(
    loader: DataLoader,
    extractor: DINOv2Extractor,
    device: torch.device,
    desc: str,
) -> list[torch.Tensor]:
    """Return list of (N_patches, D) float32 tensors — one per image."""
    features = []
    for imgs in tqdm(loader, desc=desc):
        pf = extractor(imgs.to(device))          # (B, N, D) float16
        pf = pf.float().cpu()
        for b in range(pf.shape[0]):
            features.append(pf[b])               # (N, D)
    return features


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PatchCore memory bank")
    parser.add_argument("--train-dir",  required=True)
    parser.add_argument("--val-dir",    default=None)
    parser.add_argument("--output",     default="models/anomaly/memory_bank.pth")
    parser.add_argument("--threshold",  default="models/anomaly/threshold.json")
    parser.add_argument("--device",     default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers",    type=int, default=4)
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    seed_all(args.seed)
    device = choose_device(args.device)
    print(f"장치: {device}")

    # ── Extractor ──────────────────────────────────────────────────────
    print("\nDINOv2 로드 중...")
    extractor = DINOv2Extractor().to(device).half()
    extractor.eval()

    # ── Train dataset → memory bank ────────────────────────────────────
    train_ds = NormalImageDataset(args.train_dir, label="train")
    if len(train_ds) == 0:
        print("❌ '_before' 이미지 없음. 종료.")
        return

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers, pin_memory=True,
    )

    print("\n[PatchCore] 메모리뱅크 구축 중...")
    all_feats = extract_patch_features(train_loader, extractor, device, "정상 패치 추출 (train)")

    bank = MemoryBank(device=device)
    bank.build(all_feats)
    print(f"  메모리 뱅크 크기: {bank.size:,} 패치")

    # ── Threshold calibration ──────────────────────────────────────────
    val_dir = args.val_dir or args.train_dir
    val_label = "val" if args.val_dir else "train (fallback)"
    val_ds = NormalImageDataset(val_dir, label=val_label)

    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers, pin_memory=True,
    )

    print(f"\n[임계값] {val_label} 이미지로 정상 점수 계산 중...")
    val_scores: list[float] = []
    for imgs in tqdm(val_loader, desc="임계값 계산"):
        pf = extractor(imgs.to(device))
        scores, _ = bank.predict(pf)
        val_scores.extend(scores.tolist())

    threshold = float(np.percentile(val_scores, ANOMALY_THRESH_PERCENTILE))
    bank.threshold = threshold
    print(
        f"  점수 — min: {min(val_scores):.3f} / "
        f"mean: {np.mean(val_scores):.3f} / "
        f"max: {max(val_scores):.3f}"
    )
    print(f"  임계값 (percentile {ANOMALY_THRESH_PERCENTILE}): {threshold:.4f}")

    # ── Save ───────────────────────────────────────────────────────────
    out_path = Path(args.output)
    bank.save(out_path)
    print(f"\n메모리뱅크 저장: {out_path}")

    thr_path = Path(args.threshold)
    thr_path.parent.mkdir(parents=True, exist_ok=True)
    with open(thr_path, "w") as f:
        json.dump({"threshold": threshold, "percentile": ANOMALY_THRESH_PERCENTILE}, f, indent=2)
    print(f"임계값 저장: {thr_path}")
    print("\n✅ 완료!")


if __name__ == "__main__":
    main()
