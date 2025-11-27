# File: app/main.py
from fastapi import FastAPI, HTTPException
from app.database import neo4j_driver
from app.models import DocumentInput, Document, EdgeInput, SearchRequest, HybridSearchRequest, SearchResult
from app.services.ingestion import ingest_document, create_edge, get_node, update_node, delete_node, get_edge
from app.services.search import vector_search, graph_search, hybrid_search
from typing import List

app = FastAPI(title="Hybrid Vector + Graph Retrieval")

@app.on_event("shutdown")
def shutdown_event():
    neo4j_driver.close()

@app.get("/")
def read_root():
    return {"message": "Hybrid Retrieval System Online"}

@app.get("/health")
def health_check():
    try:
        with neo4j_driver.get_session() as session:
            session.run("RETURN 1")
        return {"status": "ok", "neo4j": "connected"}
    except Exception as e:
        return {"status": "error", "neo4j": str(e)}

# --- Ingestion ---

@app.post("/nodes", response_model=Document)
def create_node(doc: DocumentInput):
    try:
        return ingest_document(doc)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/edges")
def create_relationship(edge: EdgeInput):
    result = create_edge(edge)
    if not result:
        raise HTTPException(status_code=400, detail="Could not create edge")
    return {
        "status": "created", 
        "type": edge.type,
        "id": result.element_id if hasattr(result, 'element_id') else result.id
    }

# --- Node CRUD ---

@app.get("/nodes/{node_id}")
def read_node(node_id: str):
    node = get_node(node_id)
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")
    return node

@app.put("/nodes/{node_id}")
def update_node_endpoint(node_id: str, doc: DocumentInput):
    node = update_node(node_id, doc)
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")
    return node

@app.delete("/nodes/{node_id}")
def delete_node_endpoint(node_id: str):
    delete_node(node_id)
    return {"status": "deleted", "id": node_id}

# --- Edge CRUD ---

@app.get("/edges/{edge_id}")
def read_edge(edge_id: str):
    edge = get_edge(edge_id)
    if not edge:
        raise HTTPException(status_code=404, detail="Edge not found")
    return edge

# --- Debug / Inspection ---

@app.get("/debug/documents")
def get_all_documents():
    query = "MATCH (d:Document) RETURN d"
    documents = []
    with neo4j_driver.get_session() as session:
        result = session.run(query)
        for record in result:
            documents.append(record['d'])
    return documents

@app.get("/debug/faiss/info")
def get_faiss_info():
    from app.database import faiss_index
    return {
        "total_vectors": faiss_index.count(),
        "dimension": faiss_index.dimension,
        "id_map": faiss_index.id_map
    }

@app.get("/debug/faiss/vector/{vector_id}")
def get_vector_by_id(vector_id: int):
    from app.database import faiss_index
    vector = faiss_index.get_vector(vector_id)
    if not vector:
        raise HTTPException(status_code=404, detail="Vector not found")
    return {"vector_id": vector_id, "embedding": vector}

# --- Search ---

@app.post("/search/vector", response_model=List[SearchResult])
def search_vector(req: SearchRequest):
    return vector_search(req.query_text, req.top_k)

@app.get("/search/graph")
def search_graph(start_id: str, depth: int = 3):
    return graph_search(start_id, depth)

@app.post("/search/hybrid", response_model=List[SearchResult])
def search_hybrid(req: HybridSearchRequest):
    return hybrid_search(
        req.query_text, 
        req.vector_weight, 
        req.graph_weight, 
        req.top_k, 
        req.graph_expand_depth
    )
