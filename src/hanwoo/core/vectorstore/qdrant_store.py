from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient
from qdrant_client.http import models


@dataclass(frozen=True)
class GalleryPoint:
    name: str
    variant: str
    distance: float
    image_path: str
    original_filename: str | None = None
    preprocessed: bool | None = None


class QdrantGalleryStore:
    def __init__(
        self,
        url: str,
        collection: str,
        vector_size: int,
        distance: models.Distance = models.Distance.EUCLID,
    ):
        self.client = QdrantClient(url=url)
        self.collection = collection
        self.vector_size = vector_size
        self.distance = distance

    def ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection):
            info = self.client.get_collection(self.collection)
            vectors = info.config.params.vectors
            if isinstance(vectors, models.VectorParams):
                existing_size = vectors.size
            else:
                existing_size = vectors[""].size
            if existing_size != self.vector_size:
                raise RuntimeError(
                    f"Qdrant collection {self.collection} vector size is {existing_size}, expected {self.vector_size}"
                )
            return

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=models.VectorParams(
                size=self.vector_size,
                distance=self.distance,
            ),
        )

    def list_names(self) -> list[str]:
        names: set[str] = set()
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                if point.payload and point.payload.get("variant") == "original":
                    names.add(str(point.payload["name"]))
            if offset is None:
                break
        return sorted(names)

    def upsert_image(
        self,
        *,
        name: str,
        image_path: Path,
        original_filename: str,
        original_vector: Iterable[float],
        rotated_vector: Iterable[float],
        preprocessed: bool | None = None,
    ) -> None:
        created_at = datetime.now(timezone.utc).isoformat()
        points = []
        for variant, vector in (
            ("original", original_vector),
            ("rotated", rotated_vector),
        ):
            point_id = str(uuid5(NAMESPACE_URL, f"{self.collection}:{name}:{variant}"))
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=list(vector),
                    payload={
                        "name": name,
                        "variant": variant,
                        "image_path": str(image_path),
                        "original_filename": original_filename,
                        "preprocessed": preprocessed,
                        "created_at": created_at,
                    },
                )
            )
        self.client.upsert(collection_name=self.collection, points=points)

    def search(self, vector: Iterable[float], limit: int) -> list[GalleryPoint]:
        response = self.client.query_points(
            collection_name=self.collection,
            query=list(vector),
            limit=limit,
            with_payload=True,
        )
        points = []
        for result in response.points:
            payload = result.payload or {}
            points.append(
                GalleryPoint(
                    name=str(payload["name"]),
                    variant=str(payload["variant"]),
                    distance=float(result.score),
                    image_path=str(payload.get("image_path", "")),
                    original_filename=payload.get("original_filename"),
                    preprocessed=payload.get("preprocessed"),
                )
            )
        return points

    def remove_image(self, name: str) -> None:
        self.client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="name",
                            match=models.MatchValue(value=name),
                        )
                    ]
                )
            ),
        )

    def clear(self) -> None:
        self.client.delete_collection(collection_name=self.collection)
        self.ensure_collection()
