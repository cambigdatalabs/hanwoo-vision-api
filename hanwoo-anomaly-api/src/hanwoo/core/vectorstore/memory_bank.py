from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from hanwoo.core.config import (
    ANOMALY_CORESET_RATIO,
    ANOMALY_IMAGE_SIZE,
    ANOMALY_K_NEIGHBORS,
    ANOMALY_TOP_K_RATIO,
)


class MemoryBank:
    """PatchCore coreset memory bank backed by pure PyTorch (no FAISS).

    Workflow
    --------
    1. Call :meth:`build` with a list of patch-feature tensors extracted from
       normal training images to construct the coreset.
    2. Call :meth:`save` to persist the bank and threshold to disk.
    3. On inference, call :meth:`load` then :meth:`predict`.

    Attributes
    ----------
    bank : torch.Tensor | None
        Coreset vectors on CPU, shape (M, D).
    bank_gpu : torch.Tensor | None
        Same vectors pinned on the inference device.
    bank_shape : tuple[int, int]
        Spatial grid of patches: (H // 14, W // 14).
    threshold : float | None
        Image-level anomaly score threshold set during calibration.
    """

    PATCH_SIZE = 14

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.bank: torch.Tensor | None = None
        self.bank_gpu: torch.Tensor | None = None
        self.bank_shape: tuple[int, int] = (
            ANOMALY_IMAGE_SIZE[0] // self.PATCH_SIZE,
            ANOMALY_IMAGE_SIZE[1] // self.PATCH_SIZE,
        )
        self.threshold: float | None = None

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    def build(self, patch_features: list[torch.Tensor]) -> None:
        """Construct coreset from a list of per-image patch feature tensors.

        Args:
            patch_features: Each tensor has shape (N_patches, D) on CPU,
                            dtype float32.
        """
        all_patches = torch.cat(patch_features, dim=0)          # (M_total, D)

        # Two-stage random coreset sampling (mirrors original training code)
        intermediate_k = max(1, int(len(all_patches) * ANOMALY_CORESET_RATIO * 5))
        idx = torch.randperm(len(all_patches))[:intermediate_k]
        intermediate = all_patches[idx]

        final_k = max(1, int(intermediate_k * ANOMALY_CORESET_RATIO))
        idx2 = torch.randperm(len(intermediate))[:final_k]
        self.bank = intermediate[idx2].float()

        self._pin_to_gpu()

    def _pin_to_gpu(self) -> None:
        if self.bank is not None:
            self.bank_gpu = self.bank.to(self.device)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "memory_bank": self.bank,
                "bank_shape": self.bank_shape,
                "threshold": self.threshold,
            },
            path,
        )

    def load(self, path: Path) -> None:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.bank = ck["memory_bank"]
        self.bank_shape = ck["bank_shape"]
        self.threshold = ck.get("threshold")
        self._pin_to_gpu()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        patch_features: torch.Tensor,
        bg_masks: torch.Tensor | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute per-image anomaly scores and raw patch-score heatmaps.

        Args:
            patch_features: (B, N, D) float16 on any device.
            bg_masks: Optional (B, H, W) bool tensor — True where background.
                      Background patches are zeroed out before scoring.

        Returns:
            scores : np.ndarray (B,) — image-level anomaly scores.
            hmaps  : np.ndarray (B, H_patch, W_patch) — raw patch scores.
        """
        if self.bank_gpu is None:
            raise RuntimeError("Memory bank not loaded. Call load() or build() first.")

        B, N, D = patch_features.shape
        pf_flat = patch_features.float().reshape(B * N, D)

        # KNN distance in feature space
        dist_flat = torch.cdist(pf_flat, self.bank_gpu.float())
        ps_all = (
            dist_flat.topk(ANOMALY_K_NEIGHBORS, dim=1, largest=False)
            .values.mean(dim=1)
            .reshape(B, N)
        )

        scores, hmaps = [], []
        for b in range(B):
            ps = ps_all[b].clone()

            if bg_masks is not None:
                bg_p = F.interpolate(
                    bg_masks[b : b + 1].unsqueeze(0).float(),
                    size=self.bank_shape,
                    mode="nearest",
                ).flatten().bool()
                ps[bg_p] = 0.0

            meat_ps = ps[ps > 0] if bg_masks is not None else ps
            if len(meat_ps) == 0:
                meat_ps = ps

            k = max(1, int(len(meat_ps) * ANOMALY_TOP_K_RATIO))
            scores.append(torch.topk(meat_ps, k).values.mean().item())
            hmaps.append(ps.reshape(*self.bank_shape).cpu().float().numpy())

        return np.array(scores, dtype=np.float32), np.array(hmaps, dtype=np.float32)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def is_loaded(self) -> bool:
        return self.bank is not None

    @property
    def size(self) -> int:
        return len(self.bank) if self.bank is not None else 0
