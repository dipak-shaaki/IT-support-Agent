import logging
from contextlib import asynccontextmanager
from time import perf_counter

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from app.api.routes import router
from app.core.config import get_settings, BASE_DIR
from app.core.logging import configure_logging
from app.rag.vectorstore import get_embeddings
from app.services.audit import init_db


configure_logging()
logger = logging.getLogger(__name__)
settings = get_settings()
init_db()


@asynccontextmanager
async def lifespan(_app: FastAPI):
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
