from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from hanwoo.core.config import DEVICE
from hanwoo.services.anomaly.pipeline import AnomalyService
from hanwoo.services.anomaly.routes import router, set_anomaly_service


anomaly_service = AnomalyService(device_name=DEVICE)
set_anomaly_service(anomaly_service)


@asynccontextmanager
async def lifespan(app: FastAPI):
    anomaly_service.load()
    yield


app = FastAPI(
    title="Hanwoo Anomaly API",
    version="0.1.0",
    lifespan=lifespan,
)
app.include_router(router)
