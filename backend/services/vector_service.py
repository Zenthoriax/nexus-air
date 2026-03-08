"""
VectorService — LanceDB + SentenceTransformers for semantic search.

Key fixes vs original:
  - .metric("cosine") added to every search (was L2/Euclidean — wrong for ST embeddings)
  - encode() always run in executor — never blocks the event loop
  - Bare except: pass replaced with logged warnings
  - self.available flag — if LanceDB init fails, search returns [] gracefully
  - IVF-PQ index creation once enough data accumulates (> 256 rows)
  - LanceDB connection opened once in initialize(), never re-opened in search()
  - delete_document() wrapped in executor (sync LanceDB call)
"""

import asyncio
import logging
import os
import uuid
from typing import List, Dict, Any

import lancedb
import pyarrow as pa
from sentence_transformers import SentenceTransformer

from config import settings

logger = logging.getLogger(__name__)


class VectorService:
    def __init__(self):
        self.model: SentenceTransformer | None = None
        self.db = None
        self.table = None
        self.table_name = "nexusair_blocks"
        self.available: bool = False      # set True only after successful initialize()

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def initialize(self) -> None:
        """
        Loads the SentenceTransformer model and opens/creates the LanceDB table.
        Called from lifespan via run_in_executor — must stay synchronous.
        Sets self.available=True only on full success.
        """
        try:
            logger.info("VectorService: loading embedding model '%s'", settings.embedding_model)
            self.model = SentenceTransformer(settings.embedding_model)
            dim = self.model.get_sentence_embedding_dimension()

            os.makedirs(settings.vector_db_path, exist_ok=True)
            self.db = lancedb.connect(str(settings.vector_db_path))

            if self.table_name not in self.db.table_names():
                logger.info("VectorService: creating LanceDB table '%s' (dim=%d)", self.table_name, dim)
                schema = pa.schema([
                    pa.field("vector",     pa.list_(pa.float32(), dim)),
                    pa.field("doc_id",     pa.string()),
                    pa.field("block_id",   pa.string()),
                    pa.field("content",    pa.string()),     # renamed from 'text' for consistency
                    pa.field("block_type", pa.string()),
                    pa.field("updated_at", pa.string()),     # ISO datetime string
                ])
                self.table = self.db.create_table(self.table_name, schema=schema)
            else:
                self.table = self.db.open_table(self.table_name)
                logger.info("VectorService: opened existing LanceDB table '%s'", self.table_name)

            self.available = True
            logger.info("VectorService: ready")

        except Exception as exc:
            logger.error("VectorService: initialization failed: %s", exc, exc_info=True)
            self.available = False

    # ── Index management ───────────────────────────────────────────────────────

    def _maybe_create_index(self) -> None:
        """
        Creates an IVF-PQ ANN index once the table has enough rows to benefit.
        IVF-PQ requires at least num_partitions (256) rows; call after ingest batches.
        This is a synchronous operation — call from executor when needed.
        """
        if self.table is None:
            return
        try:
            row_count = self.table.count_rows()
            if row_count > 256:
                logger.info("VectorService: creating IVF-PQ index (%d rows)", row_count)
                self.table.create_index(
                    metric="cosine",
                    num_partitions=256,
                    num_sub_vectors=96,
                )
        except Exception as exc:
            logger.warning("VectorService: index creation skipped: %s", exc)

    async def create_index_if_ready(self) -> None:
        """Async wrapper — schedules index creation in executor after ingest."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._maybe_create_index)

    # ── Embedding ──────────────────────────────────────────────────────────────

    async def embed_document(self, doc_id: str, blocks: list) -> None:
        """
        Embeds all blocks for a document and upserts them into LanceDB.
        Encoding is run in an executor — never on the event loop.
        """
        if not self.available or not blocks:
            return

        texts: List[str] = []
        block_ids: List[str] = []

        for b in blocks:
            if isinstance(b, dict):
                texts.append(b.get("content", b.get("text", "")))
                block_ids.append(b.get("id", str(uuid.uuid4())))
            elif hasattr(b, "content"):
                texts.append(b.content)
                block_ids.append(getattr(b, "id", str(uuid.uuid4())))
            else:
                texts.append(str(b))
                block_ids.append(str(uuid.uuid4()))

        # Filter blanks
        pairs = [(t, bid) for t, bid in zip(texts, block_ids) if t.strip()]
        if not pairs:
            return
        texts, block_ids = zip(*pairs)
        texts = list(texts)
        block_ids = list(block_ids)

        # Encode all blocks in one batched call — never one-by-one
        loop = asyncio.get_running_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: self.model.encode(texts, batch_size=32, show_progress_bar=False)
        )

        # Delete previous vectors for this doc (emulates upsert)
        try:
            await loop.run_in_executor(
                None, self.table.delete, f"doc_id = '{doc_id}'"
            )
        except Exception as exc:
            logger.warning("VectorService: pre-delete for doc_id=%s failed: %s", doc_id, exc)

        now_str = __import__("datetime").datetime.utcnow().isoformat()
        data = [
            {
                "vector":     emb.tolist(),
                "doc_id":     doc_id,
                "block_id":   bid,
                "content":    text,
                "block_type": "block",
                "updated_at": now_str,
            }
            for emb, bid, text in zip(embeddings, block_ids, texts)
        ]

        try:
            await loop.run_in_executor(None, self.table.add, data)
        except Exception as exc:
            logger.error("VectorService: failed to insert vectors for doc_id=%s: %s", doc_id, exc, exc_info=True)

    # ── Search ─────────────────────────────────────────────────────────────────

    async def search(self, query_text: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Encodes the query then runs cosine ANN search in LanceDB.
        Returns [] gracefully if the service is unavailable.
        """
        if not self.available or not self.model or not self.table:
            if not self.available:
                logger.warning("VectorService: search skipped — service unavailable")
            return []

        loop = asyncio.get_running_loop()
        query_vector = await loop.run_in_executor(
            None,
            lambda: self.model.encode(query_text, show_progress_bar=False)
        )

        try:
            results = await loop.run_in_executor(
                None,
                lambda: (
                    self.table
                    .search(query_vector.tolist())
                    .metric("cosine")        # CRITICAL: cosine, not L2
                    .limit(top_k)
                    .to_list()
                )
            )
        except Exception as exc:
            logger.error("VectorService: search failed: %s", exc, exc_info=True)
            return []

        return [
            {
                "doc_id":   r["doc_id"],
                "block_id": r.get("block_id", ""),
                "text":     r.get("content", r.get("text", "")),
                "score":    r.get("_distance", 0.0),
            }
            for r in results
        ]

    # ── Delete ─────────────────────────────────────────────────────────────────

    async def delete_document(self, doc_id: str) -> None:
        """Removes all vectors for a doc_id. Runs in executor — sync LanceDB call."""
        if not self.table:
            return
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(
                None, self.table.delete, f"doc_id = '{doc_id}'"
            )
        except Exception as exc:
            logger.warning(
                "VectorService: delete_document failed for doc_id=%s: %s", doc_id, exc
            )


# Singleton
vector_service = VectorService()
