import pytest
import hashlib
from unittest.mock import AsyncMock, patch, MagicMock
from services.ingest_service import IngestService

@pytest.fixture
def ingest_svc():
    return IngestService()

@pytest.mark.asyncio
async def test_markdown_processing_duplicate_check(ingest_svc):
    mock_session = AsyncMock()
    
    raw_md = "---\ntitle: Test File\ntags: [a, b]\n---\n# Header\nThis is a test block."
    content_hash = hashlib.sha256(raw_md.encode('utf-8')).hexdigest()
    
    # Setup mock to return a duplicate document
    # session.execute() returns a Result object.
    # Result.scalars() returns a ScalarResult object.
    # ScalarResult.first() is synchronous in SQLAlchemy 2.0 but in async sessions it's often awaited or we mock the chain.
    mock_result = MagicMock()
    mock_scalars = MagicMock()
    mock_scalars.first.return_value = MagicMock(id="existing_duplicate_id")
    mock_result.scalars.return_value = mock_scalars
    mock_session.execute.return_value = mock_result
    
    result = await ingest_svc.process_markdown(raw_md, "test.md", mock_session)
    
    assert result["status"] == "duplicate"
    assert result["existing_id"] == "existing_duplicate_id"

@pytest.mark.asyncio
async def test_markdown_processing_success(ingest_svc):
    mock_session = AsyncMock()
    
    raw_md = "---\ntitle: Test File\ntags: [a, b]\n---\n# Header\nThis is a test block."
    
    # Setup mock to return NO duplicate document
    mock_result = MagicMock()
    mock_scalars = MagicMock()
    mock_scalars.first.return_value = None
    mock_result.scalars.return_value = mock_scalars
    mock_session.execute.return_value = mock_result
    
    # We must patch the vector service and graph service otherwise they will error out internally
    with patch("services.ingest_service.vector_service.embed_document", new_callable=AsyncMock) as mock_embed:
        with patch("services.ingest_service.graph_service.update_document_links") as mock_graph:
            
            # Sub-mock the internal get_or_create to prevent trying to hit DB for wikilinks
            with patch.object(ingest_svc, "_get_or_create_document_by_title", new_callable=AsyncMock) as mock_get_doc:
                result = await ingest_svc.process_markdown(raw_md, "test.md", mock_session)
                
                assert result["status"] == "imported"
                assert "doc_id" in result
                
                # Verify document added to DB mock
                assert mock_session.add.called
                assert mock_session.commit.called

@pytest.mark.asyncio
async def test_pdf_processing_duplicate(ingest_svc):
    mock_session = AsyncMock()
    pdf_bytes = b"fake_pdf_data"
    
    # Setup mock to return a duplicate document
    mock_result = MagicMock()
    mock_scalars = MagicMock()
    mock_scalars.first.return_value = MagicMock(id="duplicate_pdf_id")
    mock_result.scalars.return_value = mock_scalars
    mock_session.execute.return_value = mock_result
    
    result = await ingest_svc.process_pdf(pdf_bytes, "test.pdf", mock_session)
    
    assert result["status"] == "duplicate"
    assert result["existing_id"] == "duplicate_pdf_id"
