"""FastAPI application entry point."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.api import detect, health, history, metrics, reports, videos
from backend.config import PROJECT_ROOT, get_settings
from backend.database import init_database
from backend.services.yolo_service import get_yolo_service


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_database()
    get_yolo_service().load()
    yield


settings = get_settings()
app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[f"http://127.0.0.1:{settings.frontend_port}", f"http://localhost:{settings.frontend_port}"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
for router in (health.router, detect.router, videos.router, metrics.router, history.router, reports.router):
    app.include_router(router, prefix="/api/v1")
app.mount("/storage", StaticFiles(directory=PROJECT_ROOT / "storage"), name="storage")
app.mount(
    "/metrics-assets",
    StaticFiles(directory=PROJECT_ROOT / "models/metrics/evaluation_runs", check_dir=False),
    name="metrics-assets",
)


@app.get("/")
def root() -> dict:
    return {"name": settings.app_name, "docs": "/docs", "health": "/api/v1/health"}
