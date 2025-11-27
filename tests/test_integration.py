import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.database import neo4j_driver
import time

client = TestClient(app)

def wait_for_neo4j():
    retries = 30
    for i in range(retries):
        try:
            with neo4j_driver.get_session() as session:
                session.run("RETURN 1")
            print("Neo4j is ready!")
            return
        except Exception as e:
            print(f"Waiting for Neo4j... ({i+1}/{retries})")
            time.sleep(2)
    raise Exception("Neo4j did not start in time")

def test_health():
    wait_for_neo4j()
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_ingest_and_search():
    # 1. Ingest Documents
    doc1 = {"text": "Neo4j is a graph database.", "title": "Neo4j Intro", "metadata": {"type": "db"}}
    doc2 = {"text": "FAISS is a library for efficient similarity search.", "title": "FAISS Intro", "metadata": {"type": "lib"}}
    doc3 = {"text": "Graph databases are great for connected data.", "title": "Graph DBs", "metadata": {"type": "concept"}}
    
    r1 = client.post("/nodes", json=doc1)
    r2 = client.post("/nodes", json=doc2)
    r3 = client.post("/nodes", json=doc3)
    
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r3.status_code == 200
    
    id1 = r1.json()["id"]
    id2 = r2.json()["id"]
    id3 = r3.json()["id"]
    
    # 2. Create Edges
    # Neo4j -> Graph DBs (RELATED_TO)
    edge1 = {"source": id1, "target": id3, "type": "RELATED_TO", "weight": 0.9}
    client.post("/edges", json=edge1)
    
    # 3. Vector Search
    # "similarity search" should match FAISS doc
    res_vec = client.post("/search/vector", json={"query_text": "similarity search", "top_k": 2})
    assert len(res_vec.json()) > 0
    assert res_vec.json()[0]["id"] == id2
    
    # 4. Hybrid Search
    # "graph database" should match Neo4j and Graph DBs.
    # Neo4j doc has a connection, so it might get boosted if we query for something related.
    res_hybrid = client.post("/search/hybrid", json={
        "query_text": "graph database", 
        "vector_weight": 0.5, 
        "graph_weight": 0.5
    })
    results = res_hybrid.json()
    assert len(results) >= 2
    
    # Check if graph info is present
    assert "graph_info" in results[0]
    print("\nHybrid Results:", results)

if __name__ == "__main__":
    test_health()
    test_ingest_and_search()
    print("All tests passed!")
