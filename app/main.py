from fastapi import FastAPI, HTTPException
from app.database import neo4j_driver
from app.models import DocumentInput, Document, EdgeInput, SearchRequest, HybridSearchRequest, SearchResult
from app.services.ingestion import ingest_document, create_edge
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
    return {"status": "created", "type": edge.type}

# --- Search ---

@app.post("/search/vector", response_model=List[SearchResult])
def search_vector(req: SearchRequest):
    return vector_search(req.query_text, req.top_k)

@app.get("/search/graph")
def search_graph(start_id: str, depth: int = 2):
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
