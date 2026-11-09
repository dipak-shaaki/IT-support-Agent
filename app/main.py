import logging
import re
from contextlib import asynccontextmanager
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from app.api.routes import router
from app.core.config import get_settings, BASE_DIR
from app.core.logging import configure_logging
from app.core.metrics import HTTP_DURATION, HTTP_REQUESTS
from app.rag.vectorstore import get_embeddings
from app.services.audit import init_db


configure_logging()
logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    started_at = perf_counter()
    logger.info("Preloading embedding model name=%s", settings.embedding_model)
    get_embeddings()
    logger.info(
        "Embedding model ready name=%s duration_ms=%d",
        settings.embedding_model,
        int((perf_counter() - started_at) * 1000),
    )
    yield


app = FastAPI(title=settings.app_name, version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip().rstrip("/")
        for origin in settings.cors_origins.split(",")
        if origin.strip()
    ],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-Admin-Key"],
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", "")
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", request_id):
        request_id = uuid4().hex
    request.state.request_id = request_id
    started_at = perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        _record_request(request, 500, perf_counter() - started_at)
        raise
    duration = perf_counter() - started_at
    _record_request(request, response.status_code, duration)
    response.headers["X-Request-ID"] = request_id
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    logger.info(
        "request method=%s path=%s status=%d duration_ms=%d request_id=%s",
        request.method,
        request.url.path,
        response.status_code,
        int(duration * 1000),
        request_id,
    )
    return response


def _record_request(request: Request, status: int, duration: float) -> None:
    if request.url.path == "/metrics":
        return
    route = request.scope.get("route")
    # Use the route template so IDs do not create unbounded label values.
    path = getattr(route, "path", None) or "unmatched"
    HTTP_REQUESTS.labels(request.method, path, str(status)).inc()
    HTTP_DURATION.labels(request.method, path).observe(duration)


@app.get("/metrics")
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


app.include_router(router)
FRONTEND_DIR = BASE_DIR / "frontend" / "dist"
FRONTEND_INDEX = FRONTEND_DIR / "index.html"
FRONTEND_ASSETS = FRONTEND_DIR / "assets"

if FRONTEND_ASSETS.is_dir():
    app.mount(
        "/assets",
        StaticFiles(directory=str(FRONTEND_ASSETS)),
        name="frontend-assets",
    )


def frontend_page() -> FileResponse:
    if not FRONTEND_INDEX.is_file():
        raise HTTPException(
            status_code=503,
            detail=(
                "The frontend build is missing. Run `cd frontend && npm install && "
                "npm run build`, or use the Vite development server."
            ),
        )
    return FileResponse(FRONTEND_INDEX)


@app.get("/")
def home() -> FileResponse:
    return frontend_page()


@app.get("/dashboard")
def dashboard() -> FileResponse:
    return frontend_page()
