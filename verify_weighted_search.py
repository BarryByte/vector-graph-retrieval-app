import sys
import os
import requests
import time

# Add current directory to path to allow imports if needed, 
# but we will try to use API where possible.
sys.path.append(os.getcwd())

from app.database import neo4j_driver

API_URL = "http://localhost:8000"

def setup_data():
    print("Setting up data in Neo4j...")
    with neo4j_driver.get_session() as session:
        # 1. Create Entity 'TestEntity'
        session.run("MERGE (e:Entity {name: 'TestEntity', id: 'entity-x'})")
        
        # 2. Create Document A (High Weight Edge)
        session.run("""
            MERGE (d:Document {id: 'doc-a', title: 'Doc A', text: 'Document A content'})
            MERGE (e:Entity {id: 'entity-x'})
            MERGE (e)-[r:MENTIONS]->(d)
            SET r.weight = 5.0
        """)
        
        # 3. Create Document B (Low Weight Edge)
        session.run("""
            MERGE (d:Document {id: 'doc-b', title: 'Doc B', text: 'Document B content'})
            MERGE (e:Entity {id: 'entity-x'})
            MERGE (e)-[r:MENTIONS]->(d)
            SET r.weight = 1.0
        """)
        
        # 4. Create dummy vector entries in FAISS (mocking or ensuring they exist if hybrid search needs them)
        # Hybrid search gets candidates from Vector Search AND Graph Expansion.
        # If we query for "TestEntity", and it's not in vector index, vector search returns nothing.
        # But Graph Expansion should find Doc A and Doc B via 'TestEntity'.
        # So we don't strictly need them in FAISS for the graph part to work, 
        # BUT hybrid_search might expect them to be in 'candidates' map which is built from vector results?
        # Let's check logic:
        # candidates = {r.id: r for r in vector_results}
        # ...
        # if doc_id not in candidates: candidates[doc_id] = ...
        # So yes, graph expansion ADDS to candidates.
        pass

def test_weighted_search():
    print("\nTesting Weighted Search...")
    
    # Query for "TestEntity"
    # vector_weight=0, graph_weight=1 to isolate graph effect
    payload = {
        "query_text": "TestEntity",
        "vector_weight": 0.0,
        "graph_weight": 1.0,
        "top_k": 10,
        "graph_expand_depth": 1
    }
    
    res = requests.post(f"{API_URL}/search/hybrid", json=payload)
    if res.status_code != 200:
        print(f"Search failed: {res.text}")
        sys.exit(1)
        
    results = res.json()
    print(f"Received {len(results)} results")
    
    doc_a_score = -1
    doc_b_score = -1
    
    for r in results:
        print(f"Doc: {r['id']}, Score: {r['score']}, GraphInfo: {r.get('graph_info')}")
        if r['id'] == 'doc-a':
            doc_a_score = r['score']
        elif r['id'] == 'doc-b':
            doc_b_score = r['score']
            
    if doc_a_score > doc_b_score:
        print("\nSUCCESS: Doc A (Weight 5.0) ranked higher than Doc B (Weight 1.0)")
    else:
        print(f"\nFAILURE: Doc A ({doc_a_score}) did not rank higher than Doc B ({doc_b_score})")
        sys.exit(1)

def cleanup():
    print("\nCleaning up...")
    with neo4j_driver.get_session() as session:
        session.run("MATCH (e:Entity {id: 'entity-x'}) DETACH DELETE e")
        session.run("MATCH (d:Document {id: 'doc-a'}) DETACH DELETE d")
        session.run("MATCH (d:Document {id: 'doc-b'}) DETACH DELETE d")

if __name__ == "__main__":
    try:
        setup_data()
        # Give a moment for any async indexing (though direct DB write is sync)
        time.sleep(1)
        test_weighted_search()
    finally:
        cleanup()
