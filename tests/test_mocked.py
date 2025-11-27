# File: tests/test_mocked.py
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from app.models import DocumentInput, EdgeInput, SearchResult
import numpy as np

# Mock dependencies BEFORE importing app.main
with patch('app.database.neo4j_driver') as mock_neo4j, \
     patch('app.database.faiss_index') as mock_faiss, \
     patch('app.services.embedding.embedding_service') as mock_embedding:
    
    from app.main import app

    client = TestClient(app)

    def test_ingest_document():
        # Setup mocks
        mock_embedding.encode.return_value = np.array([0.1, 0.2, 0.3])
        mock_faiss.add.return_value = 123
        
        mock_session = MagicMock()
        mock_neo4j.get_session.return_value.__enter__.return_value = mock_session
        
        # Execute
        doc = {"text": "Test document", "title": "Test", "metadata": {"type": "test"}}
        response = client.post("/nodes", json=doc)
        
        # Verify
        assert response.status_code == 200
        data = response.json()
        assert data["vector_id"] == 123
        assert data["text"] == "Test document"
        
        # Verify calls
        mock_embedding.encode.assert_called_once()
        mock_faiss.add.assert_called_once()
        mock_session.run.assert_called_once()

    def test_create_edge():
        # Setup mocks
        mock_session = MagicMock()
        mock_neo4j.get_session.return_value.__enter__.return_value = mock_session
        mock_result = MagicMock()
        mock_record = {'r': {'weight': 0.8, 'type': 'RELATED_TO'}}
        mock_result.single.return_value = mock_record
        mock_session.run.return_value = mock_result
        
        # Execute
        edge = {"source": "id1", "target": "id2", "type": "RELATED_TO", "weight": 0.8}
        response = client.post("/edges", json=edge)
        
        # Verify
        assert response.status_code == 200
        assert response.json()["status"] == "created"
        
        mock_session.run.assert_called_once()

    def test_hybrid_search():
        # Setup mocks
        # 1. Vector Search Mocks
        mock_embedding.encode.return_value = np.array([0.1, 0.2])
        # Return distances, indices
        mock_faiss.search.return_value = (np.array([[0.9, 0.8]]), np.array([[0, 1]]))
        mock_faiss.id_map = {0: "doc1", 1: "doc2"}
        
        # 2. Neo4j Mocks (for fetching doc details AND graph scoring)
        mock_session = MagicMock()
        mock_neo4j.get_session.return_value.__enter__.return_value = mock_session
        
        # Mocking multiple calls to session.run
        # Call 1 & 2: Fetching doc details for vector search results
        # Call 3: Graph scoring
        
        def side_effect(query, **kwargs):
            if "MATCH (d:Document {id: $id})" in query:
                # Return doc details
                doc_id = kwargs['id']
                return MagicMock(single=lambda: {'d': {'text': f"Content of {doc_id}", 'title': f"Title {doc_id}"}})
            elif "UNWIND $candidate_ids" in query:
                # Return graph scores
                # doc1 has degree 5, doc2 has degree 2
                return [
                    {'cid': 'doc1', 'degree': 5},
                    {'cid': 'doc2', 'degree': 2}
                ]
            return MagicMock()
            
        mock_session.run.side_effect = side_effect
        
        # Execute
        req = {
            "query_text": "test query",
            "vector_weight": 0.5,
            "graph_weight": 0.5,
            "top_k": 2
        }
        response = client.post("/search/hybrid", json=req)
        
        # Verify
        assert response.status_code == 200
        results = response.json()
        assert len(results) == 2
        
        # Check if scores are calculated (mocked values)
        # doc1: v_score=0.9, g_score=5. Normalized: v=1.0, g=1.0 -> final = 0.5*1 + 0.5*1 = 1.0
        # doc2: v_score=0.8, g_score=2. Normalized: v=0.88, g=0.4 -> final = 0.5*0.88 + 0.5*0.4 = 0.64
        
        print("\nMocked Hybrid Results:", results)
        assert results[0]['id'] == 'doc1'
        assert results[0]['score'] > results[1]['score']
