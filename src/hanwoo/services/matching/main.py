from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from hanwoo.core.auth import APIKeyAuthMiddleware, get_required_api_key
from hanwoo.core.config import DEVICE
from hanwoo.services.matching.pipeline import MatchingService
from hanwoo.services.matching.routes import router, set_matching_service


matching_service = MatchingService(device_name=DEVICE)
set_matching_service(matching_service)


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_required_api_key()
    matching_service.load()
    yield


app = FastAPI(
    title="Hanwoo Matching API",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(APIKeyAuthMiddleware)
app.include_router(router)
