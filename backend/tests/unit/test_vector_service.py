import pytest
import os
import asyncio
from services.vector_service import VectorService
from config import settings

@pytest.fixture(scope="module")
def vector_svc():
    """
    Creates an isolated VectorService connected to a testing LanceDB directory.
    Normally in a full CI/CD pipeline this would be a mocked out instance, 
    but for these integration tests we will test actual LanceDB writes locally.
    """
    # Overwrite the DB path for testing to avoid polluting main state
    import config
    original_path = config.settings.vector_db_path
    
    test_db_path = config.settings.base_dir / "data" / "test_vectors"
    config.settings.vector_db_path = test_db_path

    svc = VectorService()
    svc.initialize() # loads the small SentenceTransformers model and creates the test_blocks table
    
    yield svc
    
    # Cleanup DB connection and restore path
    config.settings.vector_db_path = original_path

@pytest.mark.asyncio
async def test_embed_and_search(vector_svc):
    doc_id = "test_doc_1"
    
    # Fake Blocks
    blocks = [
        {"id": "b1", "content": "The quick brown fox jumps over the lazy dog."},
        {"id": "b2", "content": "Machine learning revolves around neural networks and gradients."}
    ]
    
    # Embed asynchronously
    await vector_svc.embed_document(doc_id, blocks)
    
    # Let LanceDB settle its index
    await asyncio.sleep(0.5)
    
    # Search for something related to block 2
    results = await vector_svc.search("neural nets", top_k=2)
    
    # Ensure it found results
    assert len(results) > 0
    
    # Ensure the top result is from doc_id 1
    assert any(res["doc_id"] == doc_id for res in results)
    
    # Ensure the text is reasonably matched
    matched_texts = [res["text"] for res in results]
    assert any("neural networks" in text for text in matched_texts)

@pytest.mark.asyncio
async def test_vector_deletion(vector_svc):
    doc_id = "test_doc_delete"
    blocks = [{"id": "del1", "content": "This block will be deleted."}]
    
    await vector_svc.embed_document(doc_id, blocks)
    await asyncio.sleep(0.5)
    
    # Verify it exists
    res1 = await vector_svc.search("deleted", top_k=5)
    assert any(r["doc_id"] == doc_id for r in res1)
    
    # Delete it
    await vector_svc.delete_document(doc_id)
    await asyncio.sleep(0.5)
    
    # Verify it is gone
    res2 = await vector_svc.search("deleted", top_k=5)
    assert not any(r["doc_id"] == doc_id for r in res2)
