"""
Health router — reports the liveness status of every backend subsystem.
"""

import logging

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from db.engine import get_db
from models.schemas import HealthResponse
from services.graph_service import graph_service
from services.inference_service import inference_service
from services.vector_service import vector_service

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/", response_model=HealthResponse)
async def health_check(session: AsyncSession = Depends(get_db)):
    """
    Returns structured health status for every backend subsystem.
    Used by the Electron shell to determine when the app is ready.
    """
    # Database
    db_status = "ok"
    try:
        await session.execute(text("SELECT 1"))
    except Exception as exc:
        db_status = f"error: {exc}"
        logger.warning("health: DB probe failed: %s", exc)

    # Graph (read stats without holding the lock long)
    stats = await graph_service.get_stats()
    graph_nodes = stats["node_count"]
    graph_edges = stats["edge_count"]

    # Vector index
    if not vector_service.available:
        vector_status = "uninitialized"
    elif vector_service.table is not None:
        vector_status = "ok"
    else:
        vector_status = "error"

    overall = "ok" if db_status == "ok" else "degraded"

    return HealthResponse(
        status=overall,
        database=db_status,
        graph_nodes=graph_nodes,
        graph_edges=graph_edges,
        vector_index=vector_status,
        model_loaded=inference_service.is_loaded,
        model_name=settings.default_model,
    )
