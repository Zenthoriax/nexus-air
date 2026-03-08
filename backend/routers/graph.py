"""
Graph router.

Key fixes vs original:
  - All handlers are async def (were sync def — unsafe with asyncio.Lock in GraphService)
  - response_model= on every route
  - Traversal returns nodes list + truncated flag
"""

import logging

from fastapi import APIRouter, HTTPException

from exceptions import DocumentNotFoundError
from models.schemas import (
    GraphResponse,
    GraphStatsResponse,
    TraversalResponse,
    TraverseRequest,
)
from services.graph_service import graph_service

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/", response_model=GraphResponse)
async def get_graph():
    """Returns the full graph structure (nodes and edges) for visualization."""
    data = await graph_service.get_all_nodes_edges()
    return GraphResponse(**data)


@router.post("/traverse", response_model=TraversalResponse)
async def traverse_graph(request: TraverseRequest):
    """BFS traversal from a starting node up to depth hops."""
    result = await graph_service.traverse(request.start_id, depth=request.depth)
    if not result["nodes"]:
        raise HTTPException(status_code=404, detail=f"Node '{request.start_id}' not found in graph.")
    return TraversalResponse(**result)


@router.get("/stats", response_model=GraphStatsResponse)
async def get_graph_stats():
    """Graph aggregate metrics: node count, edge count, orphans, top hubs."""
    stats = await graph_service.get_stats()
    return GraphStatsResponse(**stats)
