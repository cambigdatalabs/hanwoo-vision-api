from __future__ import annotations

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.nn as nn
from hanwoo.core.config import ANOMALY_DINO_LAYERS


class DINOv2Extractor(nn.Module):
    def __init__(self, layers: list[int] | None = None) -> None:
        super().__init__()
        self._layers = layers if layers is not None else ANOMALY_DINO_LAYERS
        self._outs: dict[int, torch.Tensor] = {}
        self.dino = torch.hub.load(
            "facebookresearch/dinov2",
            "dinov2_vitb14",
            pretrained=True,
            verbose=False,
            trust_repo=True,
            force_reload=False,
        )
        for li in self._layers:
            self.dino.blocks[li].register_forward_hook(
                lambda _m, _i, out, li=li: self._outs.update({li: out[:, 1:, :]})
            )

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._outs.clear()
        self.dino(x.half())
        return torch.cat([self._outs[li] for li in self._layers], dim=-1)
