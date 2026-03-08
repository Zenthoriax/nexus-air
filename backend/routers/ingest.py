"""
Ingest router.

Key fixes vs original:
  - 415 (not 400) for wrong file types
  - File size validated before reading full content into memory
  - Filename sanitised against path traversal
  - session.commit() not called here — get_db() owns the transaction
"""

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from db.engine import get_db
from exceptions import IngestError
from models.schemas import IngestResponse
from services.ingest_service import ingest_service

router = APIRouter()
logger = logging.getLogger(__name__)

_MAX_BYTES = settings.max_file_size_mb * 1024 * 1024


def _safe_filename(upload: UploadFile) -> str:
    """Strip directory components to prevent path-traversal attacks."""
    name = Path(upload.filename).name    # keeps only the final component
    if not name or ".." in name or name.startswith("/"):
        raise HTTPException(status_code=400, detail="Invalid filename.")
    return name


@router.post("/markdown", response_model=IngestResponse)
async def ingest_markdown(
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_db),
):
    filename = _safe_filename(file)
    if not filename.lower().endswith(".md"):
        raise HTTPException(status_code=415, detail="Only Markdown (.md) files are accepted.")

    # Read with size cap — avoid reading unbounded content into memory
    raw = await file.read(_MAX_BYTES + 1)
    if len(raw) > _MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {settings.max_file_size_mb} MB.",
        )

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File could not be decoded as UTF-8.")

    try:
        result = await ingest_service.process_markdown(text, filename, session)
    except IngestError as exc:
        logger.error("ingest/markdown: %s", exc, exc_info=True)
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error("ingest/markdown: unexpected error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Ingest processing failed.")

    return IngestResponse(**result)


@router.post("/pdf", response_model=IngestResponse)
async def ingest_pdf(
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_db),
):
    filename = _safe_filename(file)
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="Only PDF (.pdf) files are accepted.")

    raw = await file.read(_MAX_BYTES + 1)
    if len(raw) > _MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {settings.max_file_size_mb} MB.",
        )

    try:
        result = await ingest_service.process_pdf(raw, filename, session)
    except IngestError as exc:
        logger.error("ingest/pdf: %s", exc, exc_info=True)
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error("ingest/pdf: unexpected error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="PDF processing failed.")

    return IngestResponse(**result)
