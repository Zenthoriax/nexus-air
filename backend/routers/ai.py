"""
AI router — SSE streaming inference endpoint.

Key fixes vs original:
  - Inference runs in a ThreadPoolExecutor via asyncio.Queue bridge —
    the event loop is NEVER blocked by the synchronous llama-cpp generator.
  - N+1 query replaced with one bulk IN query for graph-traversal nodes.
  - graph + vector search run in PARALLEL with asyncio.gather().
  - 5-second retrieval timeout via asyncio.wait_for().
  - 503 returned as JSON (not SSE) when model is not loaded.
  - Empty / whitespace query returns 422.
  - LIKE special chars escaped before title search.
  - Query parameter stripped of whitespace.
"""

import asyncio
import concurrent.futures
import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from config import settings
from db.engine import get_db
from exceptions import ModelNotLoadedError, RetrievalError
from models.orm import Document
from models.schemas import StopResponse
from services.context_builder import context_builder
from services.graph_service import graph_service
from services.inference_service import inference_service
from services.vector_service import vector_service

logger = logging.getLogger(__name__)
router = APIRouter()

# One shared executor for inference threads
_inference_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="llm")


# ── Stop endpoint ──────────────────────────────────────────────────────────────

@router.post("/stop", response_model=StopResponse)
async def stop_generation():
    """Signals the running inference thread to stop after the current token."""
    inference_service.stop()
    return StopResponse(message="Stop signal sent")


# ── Stream endpoint ────────────────────────────────────────────────────────────

@router.get("/stream")
async def stream_ai_response(
    query: str = Query(...),
    request: Request = None,
    session: AsyncSession = Depends(get_db),
):
    """
    Full RAG pipeline with SSE streaming.

    Pipeline order:
      1. Validate & sanitise query
      2. Parallel: graph traversal seed lookup + vector search
      3. Bulk-fetch graph-traversal documents (single IN query)
      4. RRF fusion → context assembly → prompt
      5. Stream tokens from InferenceService via threadpool → asyncio.Queue → SSE
    """
    # 1. Validate
    query = query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="Query must not be empty.")

    if not inference_service.is_loaded:
        return JSONResponse(
            status_code=503,
            content={"error": "LLM model is not loaded yet. Please wait for startup to complete."},
        )

    # 2. Parallel retrieval with timeout
    async def _retrieve():
        # Escape LIKE wildcards to prevent unintended pattern matches
        safe_q = query.replace("%", r"\%").replace("_", r"\_")
        search_term = f"%{safe_q}%"

        # Graph seed: titles containing the query keywords
        title_match_result = await session.execute(
            select(Document.id).where(Document.title.ilike(search_term))
        )
        seed_ids: List[str] = [row[0] for row in title_match_result.all()]

        # Run graph traversal and vector search in PARALLEL
        async def _graph_traverse():
            all_node_ids: List[str] = []
            for seed_id in seed_ids:
                result = await graph_service.traverse(seed_id, depth=settings.graph_hop_depth)
                for node in result.get("nodes", []):
                    all_node_ids.append(node["id"])
            return list(dict.fromkeys(all_node_ids))  # deduplicate, preserve order

        graph_task = asyncio.create_task(_graph_traverse())
        vector_task = asyncio.create_task(
            vector_service.search(query, top_k=settings.top_k)
        )

        traversed_ids, vector_results = await asyncio.gather(graph_task, vector_task)
        return traversed_ids, vector_results

    try:
        traversed_ids, vector_results = await asyncio.wait_for(_retrieve(), timeout=5.0)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Retrieval timed out. Try a shorter query.")

    # 3. Bulk-fetch graph documents — ONE query, not N queries in a loop
    graph_results = []
    if traversed_ids:
        bulk_result = await session.execute(
            select(Document).where(Document.id.in_(traversed_ids))
        )
        docs_by_id = {d.id: d for d in bulk_result.scalars().all()}
        # Preserve traversal order
        for nid in traversed_ids:
            if nid in docs_by_id:
                d = docs_by_id[nid]
                graph_results.append({
                    "id":      d.id,
                    "title":   d.title,
                    "content": d.content,
                })

    # 4. Fuse → context → prompt
    fused   = context_builder.fuse_results(graph_results, vector_results)
    context = context_builder.build_context(fused, token_budget=settings.max_context_tokens)
    prompt  = context_builder.build_prompt(context, query)

    logger.info(
        "AI stream: query=%r graph_docs=%d vector_blocks=%d fused=%d",
        query[:60], len(graph_results), len(vector_results), len(fused),
    )

    # 5. Token streaming via asyncio.Queue bridge
    # The llama-cpp generator is a blocking C iterator — it runs in a
    # ThreadPoolExecutor. Tokens are put onto a Queue that the async
    # generator awaits without blocking the event loop.
    token_queue: asyncio.Queue = asyncio.Queue(maxsize=256)
    loop = asyncio.get_event_loop()

    def _run_inference():
        try:
            for chunk in inference_service.stream_response(prompt):
                token = chunk["choices"][0].get("text", "")
                if token:
                    asyncio.run_coroutine_threadsafe(token_queue.put(token), loop).result()
        except Exception as exc:
            error_msg = f"[INFERENCE_ERROR: {exc}]"
            asyncio.run_coroutine_threadsafe(token_queue.put(error_msg), loop).result()
        finally:
            asyncio.run_coroutine_threadsafe(token_queue.put(None), loop).result()

    _inference_executor.submit(_run_inference)

    async def event_generator():
        while True:
            if request and await request.is_disconnected():
                inference_service.stop()
                logger.info("AI stream: client disconnected, stop sent")
                break

            try:
                token = await asyncio.wait_for(token_queue.get(), timeout=30.0)
            except asyncio.TimeoutError:
                yield "data: [TIMEOUT]\n\n"
                inference_service.stop()
                break

            if token is None:
                yield "data: [DONE]\n\n"
                break

            yield f"data: {token}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
