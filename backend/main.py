"""
NexusAir — FastAPI application entry point.

Key fixes vs original:
  - CORS restricted to localhost origins only (was allow_origins=["*"])
  - vector_service.initialize() and inference_service.load_model() run in
    executor so the event loop is never blocked during startup
  - Model path uses settings.default_model (was hardcoded filename)
  - Alembic called programmatically (not via subprocess)
  - Global exception handlers for custom exceptions and bare Exception
  - Request-ID middleware (X-Request-ID header on every response)
  - Timing middleware (logs method + path + status + duration)
  - All print() replaced with structured logging
  - Shutdown: inference stop signal + sleep before engine.dispose()
"""

import asyncio
import logging
import time
import uuid

from alembic import command as alembic_command
from alembic.config import Config as AlembicConfig
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import settings
from db.engine import AsyncSessionLocal, engine
from exceptions import DocumentNotFoundError, ModelNotLoadedError, NexusAirError
from routers import ai, documents, graph, health, ingest
from services.graph_service import graph_service
from services.inference_service import inference_service
from services.vector_service import vector_service

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG if settings.debug_mode else logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Alembic migrations ────────────────────────────────────────────────────────

def run_migrations() -> None:
    """Run Alembic upgrade head programmatically — no subprocess dependency."""
    logger.info("Running database migrations...")
    alembic_cfg = AlembicConfig(str(settings.base_dir / "alembic.ini"))
    try:
        alembic_command.upgrade(alembic_cfg, "head")
        logger.info("Database migrations complete.")
    except Exception as exc:
        logger.error("Migration failed: %s", exc, exc_info=True)
        raise RuntimeError("Database migrations failed. Cannot start.") from exc


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup order ─────────────────────────────────────────────────────────
    # 1. Alembic — must run before any service touches the DB
    run_migrations()

    # 2. GraphService — needs DB to be ready
    logger.info("Building knowledge graph from database...")
    async with AsyncSessionLocal() as session:
        await graph_service.build_from_db(session)
    logger.info(
        "Graph ready: %d nodes, %d edges",
        graph_service.graph.number_of_nodes(),
        graph_service.graph.number_of_edges(),
    )

    # 3. VectorService — loads SentenceTransformer (CPU-heavy) in executor
    loop = asyncio.get_event_loop()
    logger.info("Initialising VectorService (embedding model + LanceDB)...")
    await loop.run_in_executor(None, vector_service.initialize)

    # 4. InferenceService — loads GGUF model (very CPU-heavy) in executor
    model_path = str(settings.llm_models_path / settings.default_model)
    logger.info("Loading LLM model: %s", model_path)
    try:
        await loop.run_in_executor(
            None, lambda: inference_service.load_model(model_path)
        )
    except FileNotFoundError as exc:
        logger.warning("LLM model not found — starting in degraded mode. %s", exc)
    except Exception as exc:
        logger.error("Failed to load LLM model: %s", exc, exc_info=True)

    logger.info("NexusAir backend startup complete.")
    yield

    # ── Shutdown ──────────────────────────────────────────────────────────────
    logger.info("Shutting down NexusAir backend...")

    # Signal inference thread to stop; give it a moment to drain
    inference_service.stop()
    await asyncio.sleep(0.2)

    # Release database connections
    await engine.dispose()
    logger.info("Shutdown complete.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="NexusAir Backend",
    description="Knowledge Graph & Documents API — 100% local, zero external calls.",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS — localhost only ─────────────────────────────────────────────────────
# "app://" covers Electron's custom protocol on packaged builds.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://localhost:3000",
        "http://localhost:5173",    # Vite dev server
        "http://127.0.0.1",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "app://.",                  # Electron production
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Middleware ─────────────────────────────────────────────────────────────────

@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """Attach a short UUID to every request for log correlation."""
    rid = str(uuid.uuid4())[:8]
    request.state.request_id = rid
    response = await call_next(request)
    response.headers["X-Request-ID"] = rid
    return response


@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    """Log method, path, status code, and round-trip duration for every request."""
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "%s %s → %d  (%.1f ms)",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


# ── Exception handlers ────────────────────────────────────────────────────────

@app.exception_handler(DocumentNotFoundError)
async def document_not_found_handler(request: Request, exc: DocumentNotFoundError):
    return JSONResponse(status_code=404, content={"error": str(exc)})


@app.exception_handler(ModelNotLoadedError)
async def model_not_loaded_handler(request: Request, exc: ModelNotLoadedError):
    return JSONResponse(status_code=503, content={"error": str(exc)})


@app.exception_handler(NexusAirError)
async def nexus_domain_error_handler(request: Request, exc: NexusAirError):
    logger.warning("Domain error on %s: %s", request.url.path, exc)
    return JSONResponse(status_code=400, content={"error": str(exc)})


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(
        "Unhandled exception on %s %s: %s",
        request.method, request.url.path, exc,
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error. Check backend logs for details."},
    )


# ── Routers ────────────────────────────────────────────────────────────────────

app.include_router(health.router,    prefix="/api/health",    tags=["Health"])
app.include_router(documents.router, prefix="/api/documents", tags=["Documents"])
app.include_router(graph.router,     prefix="/api/graph",     tags=["Graph"])
app.include_router(ai.router,        prefix="/api/ai",        tags=["AI"])
app.include_router(ingest.router,    prefix="/api/ingest",    tags=["Ingest"])


@app.get("/", include_in_schema=False)
async def root():
    return {"message": "NexusAir API", "host": settings.host, "port": settings.port}


# ── Dev entry-point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug_mode,
        log_level="debug" if settings.debug_mode else "info",
    )
