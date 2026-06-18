from __future__ import annotations

from pydantic import BaseModel


class DirectoryImportRequest(BaseModel):
    directory: str
    preprocess: bool = True
