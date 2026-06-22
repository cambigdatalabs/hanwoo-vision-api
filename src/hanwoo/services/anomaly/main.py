from __future__ import annotations

from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI

from hanwoo.core.auth import APIKeyAuthMiddleware, get_required_api_key
from hanwoo.core.config import DEVICE
from hanwoo.services.anomaly.pipeline import AnomalyService
from hanwoo.services.anomaly.routes import router, set_anomaly_service


anomaly_service = AnomalyService(device_name=DEVICE)
set_anomaly_service(anomaly_service)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_required_api_key()
    try:
        anomaly_service.load()
    except Exception as exc:
        logger.warning("Anomaly service not loaded: %s", exc)
    yield


app = FastAPI(
    title="Hanwoo Anomaly API",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(APIKeyAuthMiddleware)
app.include_router(router)
