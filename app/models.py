from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class DocumentInput(BaseModel):
    text: str
    title: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = {}

class Document(DocumentInput):
    id: str
    vector_id: Optional[int] = None

class Entity(BaseModel):
    id: Optional[str] = None
    name: str
    type: str
    metadata: Optional[Dict[str, Any]] = {}

class EdgeInput(BaseModel):
    source: str
    target: str
    type: str
    weight: float = 1.0
    metadata: Optional[Dict[str, Any]] = {}

class SearchRequest(BaseModel):
    query_text: str
    top_k: int = 10

class HybridSearchRequest(SearchRequest):
    vector_weight: float = 0.7
    graph_weight: float = 0.3
    graph_expand_depth: int = 1

class SearchResult(BaseModel):
    id: str
    text: Optional[str] = None
    score: float
    metadata: Optional[Dict[str, Any]] = {}
    graph_info: Optional[Dict[str, Any]] = None
