"""
IngestService — Markdown and PDF document ingestion pipeline.

Key fixes vs original:
  - File size checked before processing (max_file_size_mb from settings)
  - session.commit() removed — transaction owned by get_db() dependency
  - CHUNK_SIZE from settings.pdf_chunk_size
  - fitz.open() run in executor (CPU-bound)
  - Link weight incremented on re-ingest (not reset to 1)
  - Wikilink parser now strips |Alias suffix
  - session.flush() used to materialise IDs mid-transaction safely
  - Structured logging replaces print()
"""

import asyncio
import hashlib
import logging
import uuid
from typing import Any, Dict

import frontmatter
import fitz  # PyMuPDF
from markdown_it import MarkdownIt
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from config import settings
from exceptions import DuplicateDocumentError, IngestError
from models.orm import Block, Document, Link
from services.graph_service import graph_service
from services.vector_service import vector_service
from utils import parse_wikilinks

logger = logging.getLogger(__name__)

_MAX_BYTES = settings.max_file_size_mb * 1024 * 1024


class IngestService:
    def __init__(self):
        self.md_parser = MarkdownIt("commonmark")

    # ── Shared helpers ─────────────────────────────────────────────────────────

    def _check_size(self, data: bytes, label: str) -> None:
        if len(data) > _MAX_BYTES:
            raise IngestError(
                f"{label} exceeds maximum upload size of {settings.max_file_size_mb} MB."
            )

    async def _find_duplicate(self, content_hash: str, session: AsyncSession) -> str | None:
        """Returns existing doc_id if content_hash already in DB, else None."""
        result = await session.execute(
            select(Document.id).where(Document.content_hash == content_hash)
        )
        row = result.scalars().first()
        return row

    async def _get_or_create_stub(self, title: str, session: AsyncSession) -> Document:
        """Looks up a document by title; creates an empty stub if missing."""
        result = await session.execute(
            select(Document).where(Document.title == title)
        )
        doc = result.scalars().first()
        if doc:
            return doc
        stub = Document(
            title=title,
            content="",
            content_hash=hashlib.sha256(b"").hexdigest(),
        )
        session.add(stub)
        await session.flush()
        return stub

    async def _process_wikilinks(
        self, doc: Document, session: AsyncSession
    ) -> None:
        """
        Extracts wikilinks from doc.content, creates stub documents for
        missing targets, and upserts links in SQLite + in-memory graph.
        On re-ingest, increments weight of existing links instead of resetting.
        """
        link_titles = parse_wikilinks(doc.content)
        new_targets_info: Dict[str, str] = {}

        for title in link_titles:
            target = await self._get_or_create_stub(title, session)
            new_targets_info[target.id] = target.title

            # Check if this link already exists — increment weight if so
            existing = await session.execute(
                select(Link).where(
                    Link.source_doc_id == doc.id,
                    Link.target_doc_id == target.id,
                )
            )
            link_row = existing.scalars().first()
            if link_row:
                link_row.weight += 1
            else:
                session.add(Link(
                    source_doc_id=doc.id,
                    target_doc_id=target.id,
                    weight=1,
                ))

        await session.flush()
        await graph_service.update_document_links(
            doc.id, doc.title, old_targets=set(), new_targets_info=new_targets_info
        )

    # ── Markdown ───────────────────────────────────────────────────────────────

    async def process_markdown(
        self, raw_content: str, filename: str, session: AsyncSession
    ) -> Dict[str, Any]:
        """
        Parses Markdown + YAML frontmatter, deduplicates, persists, and
        fires async embedding. Transaction is owned by the caller (get_db).
        """
        raw_bytes = raw_content.encode("utf-8")
        self._check_size(raw_bytes, filename)

        # 1. Dedup check BEFORE any insert
        content_hash = hashlib.sha256(raw_bytes).hexdigest()
        existing_id = await self._find_duplicate(content_hash, session)
        if existing_id:
            logger.info("IngestService: duplicate markdown '%s' → existing %s", filename, existing_id)
            return {"status": "duplicate", "existing_id": existing_id}

        # 2. Parse frontmatter
        parsed = frontmatter.loads(raw_content)
        title = parsed.metadata.get("title", filename.removesuffix(".md"))
        tags = parsed.metadata.get("tags", {})
        if isinstance(tags, list):
            tags = {"metadata": tags}
        content_body = parsed.content

        # 3. Insert document
        new_doc = Document(
            title=title,
            content=raw_content,
            tags=tags,
            content_hash=content_hash,
            word_count=len(raw_content.split()),
        )
        session.add(new_doc)
        await session.flush()      # materialise new_doc.id — no commit yet

        # 4. Parse blocks via markdown-it AST
        blocks_data = []
        for token in self.md_parser.parse(content_body):
            if token.type == "inline" and token.content.strip():
                blocks_data.append(token.content.strip())
            elif token.type in ("fence", "code_block") and token.content.strip():
                blocks_data.append(token.content.strip())

        if not blocks_data and content_body.strip():
            blocks_data = [content_body.strip()]

        created_blocks = []
        for pos, text in enumerate(blocks_data):
            block = Block(
                doc_id=new_doc.id,
                content=text,
                block_type="markdown",
                position=pos,
            )
            created_blocks.append(block)

        session.add_all(created_blocks)
        new_doc.block_count = len(created_blocks)

        # 5. Wikilinks & graph
        await self._process_wikilinks(new_doc, session)

        # NOTE: no session.commit() here — caller (get_db) owns the transaction

        # 6. Fire async embedding AFTER data is staged (will commit before embed runs)
        asyncio.create_task(vector_service.embed_document(new_doc.id, created_blocks))

        logger.info("IngestService: ingested markdown '%s' as doc_id=%s (%d blocks)",
                    filename, new_doc.id, len(created_blocks))
        return {"status": "imported", "doc_id": new_doc.id}

    # ── PDF ────────────────────────────────────────────────────────────────────

    async def process_pdf(
        self, file_bytes: bytes, filename: str, session: AsyncSession
    ) -> Dict[str, Any]:
        """
        PyMuPDF extraction, chunking, dedup, persist, async embed.
        PyMuPDF is CPU-bound — extraction runs in executor.
        """
        self._check_size(file_bytes, filename)

        # 1. Dedup
        content_hash = hashlib.sha256(file_bytes).hexdigest()
        existing_id = await self._find_duplicate(content_hash, session)
        if existing_id:
            logger.info("IngestService: duplicate PDF '%s' → existing %s", filename, existing_id)
            return {"status": "duplicate", "existing_id": existing_id}

        # 2. Extract text in executor (CPU-bound)
        loop = asyncio.get_running_loop()
        try:
            full_text = await loop.run_in_executor(None, _extract_pdf_text, file_bytes)
        except Exception as exc:
            logger.error("IngestService: PDF extraction failed for '%s': %s", filename, exc, exc_info=True)
            raise IngestError(f"Failed to process PDF '{filename}': {exc}") from exc

        title = filename.removesuffix(".pdf")

        # 3. Insert document
        new_doc = Document(
            title=title,
            content=full_text,
            tags={"format": "pdf"},
            content_hash=content_hash,
            word_count=len(full_text.split()),
        )
        session.add(new_doc)
        await session.flush()

        # 4. Chunk into ~512-token blocks
        chunk_size = settings.pdf_chunk_size
        created_blocks = []
        for pos, i in enumerate(range(0, len(full_text), chunk_size)):
            chunk = full_text[i : i + chunk_size].strip()
            if chunk:
                created_blocks.append(Block(
                    doc_id=new_doc.id,
                    content=chunk,
                    block_type="pdf_chunk",
                    position=pos,
                ))

        session.add_all(created_blocks)
        new_doc.block_count = len(created_blocks)

        # 5. Wikilinks & graph
        await self._process_wikilinks(new_doc, session)

        # 6. Fire async embedding
        asyncio.create_task(vector_service.embed_document(new_doc.id, created_blocks))

        logger.info("IngestService: ingested PDF '%s' as doc_id=%s (%d chunks)",
                    filename, new_doc.id, len(created_blocks))
        return {"status": "imported", "doc_id": new_doc.id}


# ── Pure function for executor ─────────────────────────────────────────────────

def _extract_pdf_text(file_bytes: bytes) -> str:
    """Runs in ThreadPoolExecutor — no async constructs allowed."""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages = []
    for page in doc:
        pages.append(page.get_text())
    return "\n".join(pages)


# Singleton
ingest_service = IngestService()
