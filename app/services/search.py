from app.database import neo4j_driver, faiss_index
from app.services.embedding import embedding_service
from app.models import SearchResult
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

def vector_search(query_text: str, top_k: int) -> List[SearchResult]:
    # 1. Encode query
    query_vector = embedding_service.encode(query_text)
    
    # 2. Search FAISS
    distances, indices = faiss_index.search(query_vector, top_k)
    
    results = []
    for i, idx in enumerate(indices):
        if idx == -1: continue
        doc_id = faiss_index.id_map.get(idx)
        if not doc_id: continue
        
        # Fetch details from Neo4j
        with neo4j_driver.get_session() as session:
            res = session.run("MATCH (d:Document {id: $id}) RETURN d", id=doc_id)
            record = res.single()
            if record:
                node = record['d']
                results.append(SearchResult(
                    id=doc_id,
                    text=node.get('text'),
                    score=float(distances[i]),
                    metadata=dict(node)
                ))
    return results

def graph_search(start_id: str, depth: int) -> List[Dict]:
    query = """
    MATCH (start {id:$start_id})-[*1..%d]-(n)
    RETURN n, min(length(shortestPath((start)-[*]-(n)))) AS hops
    ORDER BY hops ASC
    LIMIT 50
    """ % depth
    
    results = []
    with neo4j_driver.get_session() as session:
        res = session.run(query, start_id=start_id)
        for record in res:
            node = record['n']
            results.append({
                "node": dict(node),
                "hops": record['hops']
            })
    return results

def hybrid_search(query_text: str, vector_weight: float, graph_weight: float, top_k: int, graph_depth: int) -> List[SearchResult]:
    # 1. Vector Search (Get more candidates than top_k to re-rank)
    initial_k = top_k * 3
    vector_results = vector_search(query_text, initial_k)
    
    if not vector_results:
        return []

    candidate_ids = [r.id for r in vector_results]
    
    # 2. Graph Scoring
    # PRD Formula: graph_score = ( sum(edge_weights) ) / (1 + hop_distance)
    # For this implementation, we calculate the sum of adjacent edge weights (adj_weight)
    # If we had a start node from the query, we could use hop_distance. 
    # Here we assume hop_distance=0 (relevance of the node itself based on its connections).
    
    query_graph = """
    UNWIND $candidate_ids AS cid
    MATCH (c {id:cid})
    OPTIONAL MATCH (c)-[r]-(nbr)
    RETURN cid,
           sum(coalesce(r.weight, 1.0)) AS adj_weight,
           count(r) AS degree
    """
    
    graph_scores = {}
    with neo4j_driver.get_session() as session:
        res = session.run(query_graph, candidate_ids=candidate_ids)
        for record in res:
            # Using adj_weight as the raw graph score
            graph_scores[record['cid']] = record['adj_weight']
            
    # Normalize scores
    max_v_score = max([r.score for r in vector_results]) if vector_results else 1.0
    max_g_score = max(graph_scores.values()) if graph_scores else 1.0
    
    if max_v_score == 0: max_v_score = 1
    if max_g_score == 0: max_g_score = 1

    final_results = []
    for r in vector_results:
        v_score_norm = r.score / max_v_score
        g_score_raw = graph_scores.get(r.id, 0)
        g_score_norm = g_score_raw / max_g_score
        
        final_score = (vector_weight * v_score_norm) + (graph_weight * g_score_norm)
        
        r.score = final_score
        r.graph_info = {"graph_score_raw": g_score_raw, "original_vector_score": r.score}
        final_results.append(r)
        
    # Sort by final score
    final_results.sort(key=lambda x: x.score, reverse=True)
    return final_results[:top_k]
