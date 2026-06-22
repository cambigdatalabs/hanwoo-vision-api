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
    lot_id: str
    capture_date: str
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

    @staticmethod
    def _filter(
        *,
        lot_id: str | None = None,
        capture_date: str | None = None,
        name: str | None = None,
        variant: str | None = None,
    ) -> models.Filter | None:
        conditions = []
        for key, value in (
            ("lot_id", lot_id),
            ("capture_date", capture_date),
            ("name", name),
            ("variant", variant),
        ):
            if value is not None:
                conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value),
                    )
                )
        if not conditions:
            return None
        return models.Filter(must=conditions)

    def list_names(self) -> list[str]:
        return [image["name"] for image in self.list_images()]

    def list_images(
        self,
        *,
        lot_id: str | None = None,
        capture_date: str | None = None,
    ) -> list[dict]:
        seen: set[tuple[str | None, str | None, str]] = set()
        images = []
        offset = None
        scroll_filter = self._filter(
            lot_id=lot_id,
            capture_date=capture_date,
            variant="original",
        )
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=scroll_filter,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                if not point.payload:
                    continue
                name = str(point.payload["name"])
                key = (
                    point.payload.get("lot_id"),
                    point.payload.get("capture_date"),
                    name,
                )
                if key in seen:
                    continue
                seen.add(key)
                images.append(
                    {
                        "name": name,
                        "lot_id": point.payload.get("lot_id"),
                        "capture_date": point.payload.get("capture_date"),
                        "image_path": point.payload.get("image_path"),
                        "original_filename": point.payload.get("original_filename"),
                        "preprocessed": point.payload.get("preprocessed"),
                        "created_at": point.payload.get("created_at"),
                    }
                )
            if offset is None:
                break
        return sorted(
            images,
            key=lambda item: (
                str(item.get("lot_id") or ""),
                str(item.get("capture_date") or ""),
                str(item["name"]),
            ),
        )

    def upsert_image(
        self,
        *,
        name: str,
        lot_id: str,
        capture_date: str,
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
            point_id = str(
                uuid5(
                    NAMESPACE_URL,
                    f"{self.collection}:{lot_id}:{capture_date}:{name}:{variant}",
                )
            )
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=list(vector),
                    payload={
                        "name": name,
                        "lot_id": lot_id,
                        "capture_date": capture_date,
                        "variant": variant,
                        "image_path": str(image_path),
                        "original_filename": original_filename,
                        "preprocessed": preprocessed,
                        "created_at": created_at,
                    },
                )
            )
        self.client.upsert(collection_name=self.collection, points=points)

    def search(
        self,
        vector: Iterable[float],
        limit: int,
        *,
        lot_id: str,
        capture_date: str | None = None,
    ) -> list[GalleryPoint]:
        response = self.client.query_points(
            collection_name=self.collection,
            query=list(vector),
            query_filter=self._filter(lot_id=lot_id, capture_date=capture_date),
            limit=limit,
            with_payload=True,
        )
        points = []
        for result in response.points:
            payload = result.payload or {}
            points.append(
                GalleryPoint(
                    name=str(payload["name"]),
                    lot_id=str(payload["lot_id"]),
                    capture_date=str(payload["capture_date"]),
                    variant=str(payload["variant"]),
                    distance=float(result.score),
                    image_path=str(payload.get("image_path", "")),
                    original_filename=payload.get("original_filename"),
                    preprocessed=payload.get("preprocessed"),
                )
            )
        return points

    def remove_image(
        self,
        name: str,
        *,
        lot_id: str,
        capture_date: str | None = None,
    ) -> None:
        self.client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(
                filter=self._filter(
                    lot_id=lot_id,
                    capture_date=capture_date,
                    name=name,
                )
            ),
        )

    def clear(
        self,
        *,
        lot_id: str,
        capture_date: str | None = None,
    ) -> None:
        self.client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(
                filter=self._filter(lot_id=lot_id, capture_date=capture_date)
            ),
        )
