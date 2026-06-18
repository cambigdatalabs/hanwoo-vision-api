from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SiameseViT(nn.Module):
    """Siamese image matcher architecture used by the Streamlit app."""

    def __init__(self, backbone: str = "swin", embedding_dim: int = 256, image_size: int = 224):
        super().__init__()
        self.backbone_name = backbone

        if backbone == "swin":
            from transformers import SwinConfig, SwinModel

            self.encoder = SwinModel(
                SwinConfig(
                    image_size=image_size,
                    embed_dim=128,
                    depths=[2, 2, 18, 2],
                    num_heads=[4, 8, 16, 32],
                    window_size=7,
                )
            )
            feature_dim = 1024
        elif backbone == "swin-tiny":
            from transformers import SwinConfig, SwinModel

            self.encoder = SwinModel(
                SwinConfig(
                    image_size=image_size,
                    embed_dim=96,
                    depths=[2, 2, 6, 2],
                    num_heads=[3, 6, 12, 24],
                    window_size=7,
                )
            )
            feature_dim = 768
        elif backbone == "deit":
            from transformers import DeiTConfig, DeiTModel

            self.encoder = DeiTModel(
                DeiTConfig(
                    image_size=image_size,
                    hidden_size=768,
                    intermediate_size=3072,
                    num_hidden_layers=12,
                    num_attention_heads=12,
                )
            )
            feature_dim = 768
        elif backbone == "deit-small":
            from transformers import DeiTConfig, DeiTModel

            self.encoder = DeiTModel(
                DeiTConfig(
                    image_size=image_size,
                    hidden_size=384,
                    intermediate_size=1536,
                    num_hidden_layers=12,
                    num_attention_heads=6,
                )
            )
            feature_dim = 384
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        self.projection = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x).pooler_output
        embedding = self.projection(features)
        return F.normalize(embedding, p=2, dim=1)
