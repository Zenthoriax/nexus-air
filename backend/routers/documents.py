"""
Documents router.

Key fixes vs original:
  - POST /: Dedup check before insert (was missing)
  - get_or_create_document_by_title uses flush() not commit() — no mid-request commits
  - PUT /{id}: embedding task uses real Block objects (not raw string)
  - DELETE /{id}: uses graph_service.delete_document() not direct .graph access
  - UUID path params validated via Annotated[UUID, ...]
  - response_model= on every route
  - Structured logging
"""

import asyncio
import hashlib
import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload

from db.engine import get_db
from exceptions import DocumentNotFoundError
from models.orm import Block, Document, Link
from models.schemas import (
    DocumentCreate,
    DocumentDetail,
    DocumentSummary,
    DocumentUpdate,
)
from services.graph_service import graph_service
from services.vector_service import vector_service
from utils import parse_wikilinks

router = APIRouter()
logger = logging.getLogger(__name__)


# ── Internal helper ────────────────────────────────────────────────────────────

async def _get_or_create_stub(title: str, session: AsyncSession) -> Document:
    """
    Looks up a document by title. Creates an empty stub with flush() if absent.
    Uses flush() (not commit()) — the outer transaction boundary owns the commit.
    """
    result = await session.execute(select(Document).where(Document.title == title))
    doc = result.scalars().first()
    if doc:
        return doc
    stub = Document(
        title=title,
        content="",
        content_hash=hashlib.sha256(b"").hexdigest(),
    )
    session.add(stub)
    await session.flush()    # materialise stub.id — no commit
    return stub


# ── Routes ─────────────────────────────────────────────────────────────────────

@router.post("/", response_model=DocumentDetail, status_code=201)
async def create_document(
    doc_in: DocumentCreate,
    session: AsyncSession = Depends(get_db),
):
    # 1. Compute hash and dedup
    content_hash = hashlib.sha256(doc_in.content.encode("utf-8")).hexdigest()
    existing = await session.execute(
        select(Document.id).where(Document.content_hash == content_hash)
    )
    if existing.scalars().first():
        raise HTTPException(status_code=409, detail="A document with identical content already exists.")

    # 2. Create document
    new_doc = Document(
        title=doc_in.title,
        content=doc_in.content,
        tags=doc_in.tags,
        content_hash=content_hash,
        word_count=len(doc_in.content.split()),
    )
    session.add(new_doc)
    await session.flush()

    # 3. Parse wikilinks
    new_targets_info = {}
    for link_title in parse_wikilinks(doc_in.content):
        target = await _get_or_create_stub(link_title, session)
        new_targets_info[target.id] = target.title
        session.add(Link(source_doc_id=new_doc.id, target_doc_id=target.id, weight=1))

    await session.flush()

    # 4. Update graph (in-memory, no await needed for the dict prep)
    await graph_service.update_document_links(
        new_doc.id, new_doc.title, old_targets=set(), new_targets_info=new_targets_info
    )

    # get_db will commit; re-fetch with blocks for response
    result = await session.execute(
        select(Document)
        .options(selectinload(Document.blocks))
        .where(Document.id == new_doc.id)
    )
    doc = result.scalars().first()
    logger.info("documents: created doc_id=%s title=%r", doc.id, doc.title)
    return doc


@router.get("/", response_model=list[DocumentSummary])
async def list_documents(session: AsyncSession = Depends(get_db)):
    result = await session.execute(
        select(Document).order_by(Document.updated_at.desc())
    )
    return result.scalars().all()


@router.get("/{doc_id}", response_model=DocumentDetail)
async def get_document(doc_id: UUID, session: AsyncSession = Depends(get_db)):
    result = await session.execute(
        select(Document)
        .options(selectinload(Document.blocks))
        .where(Document.id == str(doc_id))
    )
    doc = result.scalars().first()
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found.")
    return doc


@router.put("/{doc_id}", response_model=DocumentDetail)
async def update_document(
    doc_id: UUID,
    doc_in: DocumentUpdate,
    session: AsyncSession = Depends(get_db),
):
    result = await session.execute(
        select(Document)
        .options(selectinload(Document.blocks))
        .where(Document.id == str(doc_id))
    )
    doc = result.scalars().first()
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found.")

    content_changed = False

    if doc_in.title is not None:
        doc.title = doc_in.title
    if doc_in.content is not None and doc.content != doc_in.content:
        doc.content = doc_in.content
        doc.content_hash = hashlib.sha256(doc_in.content.encode("utf-8")).hexdigest()
        doc.word_count = len(doc_in.content.split())
        content_changed = True
    if doc_in.tags is not None:
        doc.tags = doc_in.tags

    if content_changed:
        # Get old link targets
        old_links_res = await session.execute(
            select(Link.target_doc_id).where(Link.source_doc_id == str(doc_id))
        )
        old_targets = {row[0] for row in old_links_res.all()}

        # Delete old links
        await session.execute(delete(Link).where(Link.source_doc_id == str(doc_id)))
        await session.flush()

        # Parse new wikilinks
        new_targets_info = {}
        for link_title in parse_wikilinks(doc.content):
            target = await _get_or_create_stub(link_title, session)
            new_targets_info[target.id] = target.title
            session.add(Link(source_doc_id=doc.id, target_doc_id=target.id, weight=1))

        await graph_service.update_document_links(
            doc.id, doc.title, old_targets=old_targets, new_targets_info=new_targets_info
        )

        # Re-embed using real blocks (not raw content string)
        blocks_res = await session.execute(
            select(Block).where(Block.doc_id == str(doc_id))
        )
        blocks = blocks_res.scalars().all()
        asyncio.create_task(vector_service.embed_document(doc.id, blocks))

    await session.flush()    # let get_db commit

    # Refresh with blocks
    result = await session.execute(
        select(Document)
        .options(selectinload(Document.blocks))
        .where(Document.id == str(doc_id))
    )
    doc = result.scalars().first()
    logger.info("documents: updated doc_id=%s content_changed=%s", doc_id, content_changed)
    return doc


@router.delete("/{doc_id}", status_code=200)
async def delete_document(doc_id: UUID, session: AsyncSession = Depends(get_db)):
    result = await session.execute(
        select(Document).where(Document.id == str(doc_id))
    )
    doc = result.scalars().first()
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found.")

    # SQLite CASCADE (enabled via PRAGMA foreign_keys=ON) will handle
    # blocks and links automatically. We explicitly delete links here
    # for the graph service update.
    await session.execute(
        delete(Link).where(
            (Link.source_doc_id == str(doc_id)) | (Link.target_doc_id == str(doc_id))
        )
    )
    await session.delete(doc)    # blocks cascade via ORM relationship

    # Update in-memory graph using the service method (respects asyncio.Lock)
    await graph_service.delete_document(str(doc_id))

    # Remove vectors asynchronously
    asyncio.create_task(vector_service.delete_document(str(doc_id)))

    logger.info("documents: deleted doc_id=%s", doc_id)
    return {"message": f"Document {doc_id} deleted successfully."}
