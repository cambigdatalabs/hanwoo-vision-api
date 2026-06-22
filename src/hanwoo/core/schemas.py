from __future__ import annotations

from pydantic import BaseModel


class DirectoryImportRequest(BaseModel):
    directory: str
    lot_id: str
    capture_date: str | None = None
    preprocess: bool = True
