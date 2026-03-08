import pytest
from services.context_builder import ContextBuilder

def test_rrf_fusion_sorting():
    cb = ContextBuilder()

    # Mock Graph results 
    # Let's say doc1 is rank 0, doc2 is rank 1
    graph_results = [
        {"id": "doc1", "title": "Quantum Nodes", "content": "Graph content 1"},
        {"id": "doc2", "title": "AI Theory", "content": "Graph content 2"}
    ]

    # Mock Vector results
    # Let's say doc2 is rank 0, doc3 is rank 1
    vector_results = [
        {"doc_id": "doc2", "title": "AI Theory", "text": "Vector content 2"},
        {"doc_id": "doc3", "title": "Deep Learning", "text": "Vector content 3"}
    ]

    fused = cb.fuse_results(graph_results, vector_results)

    # doc2 appeared in both lists: (1/61) + (1/60)
    # doc1 appeared only in graph: (1/60)
    # doc3 appeared only in vector: (1/61)
    
    # Expected order: doc2 (highest), doc1 (middle), doc3 (lowest)
    
    assert len(fused) == 3
    assert fused[0]["id"] == "doc2"
    assert fused[1]["id"] == "doc1"
    assert fused[2]["id"] == "doc3"

    # Verify score math
    expected_doc2_score = (1.0 / 61) + (1.0 / 60)
    assert abs(fused[0]["score"] - expected_doc2_score) < 1e-6

def test_context_building_token_budget():
    cb = ContextBuilder()
    
    mock_fused_docs = [
        {"title": "Doc A", "content": "A" * 40}, # 10 tokens 
        {"title": "Doc B", "content": "B" * 400}, # 100 tokens
        {"title": "Doc C", "content": "C" * 800}  # 200 tokens
    ]
    
    # Total chars without titles: 1240 (~310 tokens). 
    # With formatting [Document: '...'] it'll be slightly more.
    
    # Budget of 500 tokens should fit everything easily
    res1 = cb.build_context(mock_fused_docs, token_budget=500)
    assert "Doc A" in res1
    assert "Doc C" in res1
    
    # Budget of 50 tokens should immediately kick Doc C and Doc B out
    # Notice that `build_context` processes from top to bottom and kicks
    # out the LOWEST-scoring items (at the end of the array) when token limit exceeded
    res2 = cb.build_context(mock_fused_docs, token_budget=50)
    assert "Doc A" in res2
    assert "Doc C" not in res2
    assert "Doc B" not in res2
