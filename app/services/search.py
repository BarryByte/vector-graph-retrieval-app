# File: app/services/search.py
from app.database import neo4j_driver, faiss_index
from app.services.embedding import embedding_service
from app.models import SearchResult
from typing import List, Dict, Set
import logging
import spacy
import math
import numpy as np

logger = logging.getLogger(__name__)

# Load Spacy model for query parsing
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.warning("Spacy model not found. Query parsing will be limited.")
    nlp = None

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
                    metadata=dict(node),
                    graph_info={}
                ))
    return results

def graph_search(start_id: str, depth: int, relationship_types: List[str] = None) -> Dict:
    # Construct relationship pattern
    # If types provided: -[:TYPE1|TYPE2*1..depth]-
    # If not: -[*1..depth]-
    
    rel_pattern = ""
    if relationship_types:
        # Sanitize types to prevent injection (basic check)
        safe_types = [t for t in relationship_types if t.isalnum() or "_" in t]
        if safe_types:
            rel_pattern = ":" + "|".join(safe_types)
            
    # Fetch nodes and relationships within depth
    # We use the explicit ID query pattern we established earlier
    final_query = f"""
    MATCH (start {{id:$start_id}})-[{rel_pattern}*0..{depth}]-(n)
    WITH collect(distinct n) as nodes
    UNWIND nodes as source
    MATCH (source)-[r{rel_pattern}]->(target)
    WHERE target IN nodes
    RETURN source, r, target
    """
    
    data = {"nodes": [], "edges": []}
    
    with neo4j_driver.get_session() as session:
        res = session.run(final_query, start_id=start_id)
        seen_nodes = set()
        seen_edges = set()
        
        for record in res:
            source = record['source']
            target = record['target']
            rel = record['r']
            
            # Helper to safely get ID
            def get_node_id(node):
                return node.get('id') or node.element_id if hasattr(node, 'element_id') else str(node.id)

            source_id = get_node_id(source)
            target_id = get_node_id(target)
            
            if source_id not in seen_nodes:
                s_dict = dict(source)
                s_dict['id'] = source_id # Ensure ID is present for frontend
                data["nodes"].append(s_dict)
                seen_nodes.add(source_id)
            
            if target_id not in seen_nodes:
                t_dict = dict(target)
                t_dict['id'] = target_id # Ensure ID is present for frontend
                data["nodes"].append(t_dict)
                seen_nodes.add(target_id)
                
            edge_key = (source_id, target_id, rel.type)
            if edge_key not in seen_edges:
                data["edges"].append({
                    "source": source_id,
                    "target": target_id,
                    "type": rel.type,
                    "weight": rel.get('weight', 1.0)
                })
                seen_edges.add(edge_key)
                
    return data

def hybrid_search(query_text: str, vector_weight: float, graph_weight: float, top_k: int, graph_depth: int, query_embedding: List[float] = None) -> "HybridSearchResponse":
    # 0. Normalize alpha / beta so they sum to 1
    total = vector_weight + graph_weight
    if total <= 0:
        alpha, beta = 1.0, 0.0
    else:
        alpha = vector_weight / total
        beta = graph_weight / total

    # 1. NLP Query Parsing (Extract Entities)
    query_entities = []
    if nlp:
        doc = nlp(query_text)
        query_entities = [ent.text for ent in doc.ents]

    logger.info(f"Query Entities: {query_entities}")

    # 2. Vector Search (Candidates Set A)
    # If query_embedding is provided, use it directly (convert to numpy)
    # Otherwise, encode query_text
    if query_embedding:
        query_vector = np.array(query_embedding, dtype=np.float32)
        distances, indices = faiss_index.search(query_vector, top_k * 3)
        vector_results = []
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
                    vector_results.append(SearchResult(
                        id=doc_id,
                        text=node.get('text'),
                        score=float(distances[i]),
                        metadata=dict(node),
                        graph_info={}
                    ))
    else:
        initial_k = top_k * 3
        vector_results = vector_search(query_text, initial_k)

    candidates: Dict[str, SearchResult] = {r.id: r for r in vector_results}

    # 3. Graph Expansion from Query Entities (Candidates Set B)
    if query_entities:
        with neo4j_driver.get_session() as session:
            query_expansion = """
            UNWIND $names AS name
            MATCH (e:Entity) WHERE toLower(e.name) = toLower(name)
            MATCH (e)-[r]-(d:Document)
            RETURN d, r.weight AS edge_weight
            LIMIT 50
            """
            res = session.run(query_expansion, names=query_entities)
            for record in res:
                node = record["d"]
                doc_id = node.get("id")
                edge_weight = record.get("edge_weight", 1.0)

                if doc_id not in candidates:
                    candidates[doc_id] = SearchResult(
                        id=doc_id,
                        text=node.get("text"),
                        score=0.0,  # vector score placeholder
                        metadata=dict(node),
                        graph_info={"hops": 1, "expansion_weight": edge_weight},
                    )
                else:
                    gi = candidates[doc_id].graph_info
                    gi["hops"] = 1
                    gi["expansion_weight"] = edge_weight

    if not candidates:
        from app.models import HybridSearchResponse
        return HybridSearchResponse(
            query_text=query_text,
            vector_weight=vector_weight,
            graph_weight=graph_weight,
            results=[]
        )

    candidate_ids = list(candidates.keys())

    # 4. Graph Scoring (Connectivity)
    connectivity_scores: Dict[str, float] = {}
    with neo4j_driver.get_session() as session:
        query_graph = """
        UNWIND $candidate_ids AS cid
        MATCH (c {id: cid})
        OPTIONAL MATCH (c)-[r]-(nbr)
        RETURN cid, sum(coalesce(r.weight, 1.0)) AS adj_weight
        """
        res = session.run(query_graph, candidate_ids=candidate_ids)
        for record in res:
            connectivity_scores[record["cid"]] = record["adj_weight"] or 0.0

    # Choose a scale for saturating graph scores (typical connectivity)
    if connectivity_scores:
        avg_c = sum(connectivity_scores.values()) / len(connectivity_scores)
    else:
        avg_c = 1.0
    graph_scale = max(1.0, avg_c)

    final_results_items = []

    for doc_id, r in candidates.items():
        # --- Vector Component ---
        raw_v = r.score  # FAISS similarity
        # clamp vector similarity into [0,1]; adjust if your model behaves differently
        v_score_norm = max(0.0, min(1.0, raw_v))

        # --- Graph Connectivity Component ---
        c_raw = connectivity_scores.get(doc_id, 0.0)
        # saturating mapping: 0 -> 0, big -> ~1, no per-query max dependency
        c_score_norm = 1.0 - math.exp(-c_raw / graph_scale)

        hops = r.graph_info.get("hops", 0)

        if beta > 0:
            g_component = (c_score_norm) / (1 + hops)
        else:
            g_component = 0.0

        # --- Final Hybrid Score (bounded in [0,1]) ---
        final_score = (alpha * v_score_norm) + (beta * g_component)

        # Construct Info Dict
        info = {
            "hop": hops,
            "raw_vector_score": raw_v,
            "connectivity_score_raw": c_raw
        }
        if "expansion_weight" in r.graph_info:
            info["edge_weight"] = r.graph_info["expansion_weight"]
        
        from app.models import HybridSearchResultItem
        final_results_items.append(HybridSearchResultItem(
            id=doc_id,
            text=r.text, # Use text instead of title
            vector_score=raw_v,
            graph_score=g_component,
            final_score=final_score,
            info=info
        ))

    final_results_items.sort(key=lambda x: x.final_score, reverse=True)
    
    from app.models import HybridSearchResponse
    return HybridSearchResponse(
        query_text=query_text,
        vector_weight=vector_weight,
        graph_weight=graph_weight,
        results=final_results_items[:top_k]
    )