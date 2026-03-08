import pytest
from services.graph_service import GraphService

@pytest.fixture
def graph_service():
    """Returns a fresh isolated GraphService instance."""
    return GraphService()

def test_graph_node_creation(graph_service):
    # Setup some fake targets
    targets = {
        "doc2": "Target Title",
        "doc3": "Another Title"
    }

    # Creating a doc should link it and create the stub nodes in the graph
    graph_service.update_document_links(
        doc_id="doc1",
        doc_title="Source Title",
        old_targets=set(),
        new_targets_info=targets
    )

    # Graph should have 3 nodes: doc1, doc2, doc3
    assert len(graph_service.graph.nodes) == 3
    
    # Validate node titles
    assert graph_service.graph.nodes["doc1"]["title"] == "Source Title"
    assert graph_service.graph.nodes["doc2"]["title"] == "Target Title"
    
    # Validate edges
    assert graph_service.graph.has_edge("doc1", "doc2")
    assert graph_service.graph.has_edge("doc1", "doc3")

def test_graph_traversal(graph_service):
    # Linear graph: docA -> docB -> docC
    graph_service.update_document_links("docA", "A", set(), {"docB": "B"})
    graph_service.update_document_links("docB", "B", set(), {"docC": "C"})

    # Traverse Depth 1
    d1 = graph_service.traverse("docA", depth=1)
    # Expected: docA (depth 0), docB (depth 1)
    assert len(d1) == 2
    ids = [n["id"] for n in d1]
    assert "docA" in ids
    assert "docB" in ids
    assert "docC" not in ids

    # Traverse Depth 2
    d2 = graph_service.traverse("docA", depth=2)
    # Expected: docA, docB, docC
    assert len(d2) == 3
    ids2 = [n["id"] for n in d2]
    assert "docC" in ids2

def test_graph_node_deletion(graph_service):
    graph_service.update_document_links("doc1", "T1", set(), {"doc2": "T2"})
    
    assert graph_service.graph.has_node("doc1")
    graph_service.delete_document("doc1")
    
    # GraphService should cleanly eject doc1 and its outgoing edges
    assert not graph_service.graph.has_node("doc1")
    # doc2 might remain depending on implementation, but doc1 is definitely gone.
