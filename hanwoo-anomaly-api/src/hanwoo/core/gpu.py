from __future__ import annotations

import torch


def choose_device(device_name: str = "auto") -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "HANWOO_DEVICE=cuda but CUDA is not available."
        )
    return torch.device(device_name)
