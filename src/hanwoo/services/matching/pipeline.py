from __future__ import annotations

from pathlib import Path
from threading import RLock

import torch
from PIL import Image
from torchvision import transforms

from hanwoo.core.config import (
    GALLERY_DIR,
    MATCHING_MODEL_PATH,
    QDRANT_COLLECTION,
    QDRANT_URL,
)
from hanwoo.core.encoders.swin import SiameseViT
from hanwoo.core.vectorstore.qdrant_store import QdrantGalleryStore


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def choose_device(device_name: str = "auto") -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "HANWOO_DEVICE=cuda but CUDA is not available. Run on a GPU host with NVIDIA Container Toolkit."
        )
    return torch.device(device_name)


def torch_load(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


class MatchingService:
    def __init__(
        self,
        model_path: Path = MATCHING_MODEL_PATH,
        gallery_dir: Path = GALLERY_DIR,
        device_name: str = "auto",
        qdrant_url: str = QDRANT_URL,
        qdrant_collection: str = QDRANT_COLLECTION,
    ):
        self.model_path = model_path
        self.gallery_dir = gallery_dir
        self.qdrant_url = qdrant_url
        self.qdrant_collection = qdrant_collection
        self.device = choose_device(device_name)
        self.model: SiameseViT | None = None
        self.store: QdrantGalleryStore | None = None
        self.backbone = ""
        self.embedding_dim = 0
        self.image_size = 224
        self.checkpoint_metadata: dict = {}
        self.transform = self._build_transform(self.image_size)
        self.lock = RLock()

    @staticmethod
    def _build_transform(image_size: int):
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Matching checkpoint not found: {self.model_path}")

        checkpoint = torch_load(self.model_path, self.device)
        self.backbone = checkpoint.get("backbone", "swin")
        self.embedding_dim = int(checkpoint.get("embedding_dim", 256))
        self.image_size = int(checkpoint.get("image_size", 224))
        self.transform = self._build_transform(self.image_size)
        self.checkpoint_metadata = {
            "backbone": self.backbone,
            "embedding_dim": self.embedding_dim,
            "image_size": self.image_size,
            "epoch": checkpoint.get("epoch"),
            "metrics": checkpoint.get("metrics"),
        }

        model = SiameseViT(
            backbone=self.backbone,
            embedding_dim=self.embedding_dim,
            image_size=self.image_size,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(self.device)
        model.eval()
        self.model = model

        self.gallery_dir.mkdir(parents=True, exist_ok=True)
        self.store = QdrantGalleryStore(
            url=self.qdrant_url,
            collection=self.qdrant_collection,
            vector_size=self.embedding_dim,
        )
        self.store.ensure_collection()

    def _ensure_loaded(self) -> SiameseViT:
        if self.model is None:
            raise RuntimeError("Matching model is not loaded")
        return self.model

    def _ensure_store(self) -> QdrantGalleryStore:
        if self.store is None:
            raise RuntimeError("Qdrant gallery store is not initialized")
        return self.store

    @staticmethod
    def safe_stem(filename: str) -> str:
        stem = Path(filename).stem.strip().replace(" ", "_")
        allowed = []
        for ch in stem:
            if ch.isalnum() or ch in {"_", "-", "."}:
                allowed.append(ch)
        return "".join(allowed) or "image"

    def embed_image(self, img: Image.Image) -> torch.Tensor:
        model = self._ensure_loaded()
        img_tensor = self.transform(img.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = model(img_tensor).squeeze(0).cpu()
        return emb

    def embed_image_dual(self, img: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        return self.embed_image(img), self.embed_image(img.rotate(180))

    def add_gallery_image(
        self,
        name: str,
        image: Image.Image,
        preprocessed: bool | None = None,
    ) -> dict:
        with self.lock:
            store = self._ensure_store()
            existing = set(store.list_names())
            base_name = self.safe_stem(name)
            final_name = base_name
            counter = 2
            while final_name in existing:
                final_name = f"{base_name}_{counter}"
                counter += 1

            save_path = self.gallery_dir / f"{final_name}.png"
            image = image.convert("RGB")
            image.save(save_path)

            emb_orig, emb_rot = self.embed_image_dual(image)
            store.upsert_image(
                name=final_name,
                image_path=save_path,
                original_filename=name,
                original_vector=emb_orig.tolist(),
                rotated_vector=emb_rot.tolist(),
                preprocessed=preprocessed,
            )

        return {"name": final_name, "path": str(save_path)}

    def remove_gallery_image(self, name: str) -> bool:
        with self.lock:
            store = self._ensure_store()
            if name not in set(store.list_names()):
                return False

            store.remove_image(name)

            for ext in IMAGE_EXTENSIONS:
                path = self.gallery_dir / f"{name}{ext}"
                if path.exists():
                    path.unlink()
        return True

    def clear_gallery(self) -> int:
        with self.lock:
            store = self._ensure_store()
            count = len(store.list_names())
            store.clear()
            for path in self.gallery_dir.iterdir():
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    path.unlink()
        return count

    def list_gallery(self) -> dict:
        with self.lock:
            names = self._ensure_store().list_names()
            return {
                "count": len(names),
                "filenames": names,
            }

    def find_matches(self, query_img: Image.Image, top_k: int) -> list[dict]:
        query_emb = self.embed_image(query_img)
        with self.lock:
            store = self._ensure_store()
            names = store.list_names()
            if not names:
                return []
            candidates = store.search(query_emb.tolist(), limit=min(len(names) * 2, top_k * 8))

        best_by_name = {}
        for point in candidates:
            current = best_by_name.get(point.name)
            if current is None or point.distance < current.distance:
                best_by_name[point.name] = point

        results = []
        for rank, point in enumerate(
            sorted(best_by_name.values(), key=lambda item: item.distance)[:top_k],
            start=1,
        ):
            similarity = max(0.0, min(1.0, 1.0 - point.distance / 2.0)) * 100.0
            results.append(
                {
                    "rank": rank,
                    "name": point.name,
                    "distance": point.distance,
                    "similarity": float(similarity),
                    "image_path": point.image_path or str(self.gallery_dir / f"{point.name}.png"),
                    "matched_variant": point.variant,
                }
            )
        return results
