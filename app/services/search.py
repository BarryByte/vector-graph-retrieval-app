# File: app/services/search.py
from app.database import neo4j_driver, faiss_index
from app.services.embedding import embedding_service
from app.models import SearchResult
from typing import List, Dict, Set
import logging
import spacy

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

def graph_search(start_id: str, depth: int) -> Dict:
    # Fetch nodes and relationships within depth
    query = """
    MATCH p=(start {id:$start_id})-[*1..%d]-(n)
    UNWIND relationships(p) AS r
    RETURN start, n, collect(DISTINCT r) AS rels
    """ % depth
    
    nodes = {}
    edges = []
    
    with neo4j_driver.get_session() as session:
        res = session.run(query, start_id=start_id)
        for record in res:
            # Add start node
            start_node = record['start']
            nodes[start_node['id']] = dict(start_node)
            
            # Add other node
            node = record['n']
            nodes[node['id']] = dict(node)
            
            # Add edges
            for r in record['rels']:
                edges.append({
                    "source": r.start_node['id'], # Note: Neo4j driver returns start_node/end_node objects but they might not have 'id' property if not loaded. 
                    # Actually, r.start_node.id is the internal ID. We stored 'id' as a property.
                    # We need to be careful here. The driver returns Relationship objects.
                    # It's safer to return properties.
                    "id": r.element_id if hasattr(r, 'element_id') else r.id,
                    "type": r.type,
                    "properties": dict(r),
                    # We need the 'id' property of the nodes to link them in vis.js
                    # But r.start_node only gives us the node reference.
                    # Let's adjust the query to return source/target IDs explicitly.
                })

    # Better Query for explicit IDs
    query_explicit = """
    MATCH (start {id:$start_id})
    CALL apoc.path.subgraphAll(start, {maxLevel: %d})
    YIELD nodes, relationships
    RETURN nodes, relationships
    """ % depth
    
    # Fallback if APOC is not available (Community Edition often has it but maybe not enabled)
    # Let's use standard Cypher
    query_standard = """
    MATCH (start {id:$start_id})-[*0..%d]-(n)
    WITH collect(distinct n) as nodes
    UNWIND nodes as n
    OPTIONAL MATCH (n)-[r]-(m)
    WHERE m IN nodes
    RETURN n, collect(distinct r) as rels
    """ % depth
    
    data = {"nodes": [], "edges": []}
    
    # query_standard execution removed as it was redundant and incomplete
    
    # Let's try a query that returns exactly what we need
    final_query = """
    MATCH (start {id:$start_id})-[*0..%d]-(n)
    WITH collect(distinct n) as nodes
    UNWIND nodes as source
    MATCH (source)-[r]->(target)
    WHERE target IN nodes
    RETURN source, r, target
    """ % depth
    
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

def hybrid_search(query_text: str, vector_weight: float, graph_weight: float, top_k: int, graph_depth: int) -> List[SearchResult]:
    # 1. NLP Query Parsing (Extract Entities)
    query_entities = []
    if nlp:
        doc = nlp(query_text)
        query_entities = [ent.text for ent in doc.ents]
    
    logger.info(f"Query Entities: {query_entities}")

    # 2. Vector Search (Candidates Set A)
    # Get more candidates to allow re-ranking
    initial_k = top_k * 3
    vector_results = vector_search(query_text, initial_k)
    
    # Map doc_id -> SearchResult
    candidates: Dict[str, SearchResult] = {r.id: r for r in vector_results}
    
    # 3. Graph Expansion from Query Entities (Candidates Set B)
    # If we found entities in the query, find documents connected to them
    if query_entities:
        with neo4j_driver.get_session() as session:
            # Find Entity nodes matching query entities (case-insensitive)
            # Then find connected Documents
            query_expansion = """
            UNWIND $names AS name
            MATCH (e:Entity) WHERE toLower(e.name) = toLower(name)
            MATCH (e)-[r]-(d:Document)
            RETURN d, r.weight AS edge_weight
            LIMIT 20
            """
            res = session.run(query_expansion, names=query_entities)
            for record in res:
                node = record['d']
                doc_id = node.get('id')
                edge_weight = record['edge_weight'] or 1.0
                
                if doc_id not in candidates:
                    # New candidate from graph
                    candidates[doc_id] = SearchResult(
                        id=doc_id,
                        text=node.get('text'),
                        score=0.0, # Zero vector score for now
                        metadata=dict(node),
                        graph_info={"hops": 1, "expansion_weight": edge_weight}
                    )
                else:
                    # Existing candidate, mark as graph-relevant
                    candidates[doc_id].graph_info["hops"] = 1
                    candidates[doc_id].graph_info["expansion_weight"] = edge_weight

    if not candidates:
        return []

    candidate_ids = list(candidates.keys())
    
    # 4. Graph Scoring (Connectivity)
    # Calculate degree/connectivity for all candidates
    query_graph = """
    UNWIND $candidate_ids AS cid
    MATCH (c {id:cid})
    OPTIONAL MATCH (c)-[r]-(nbr)
    RETURN cid,
           sum(coalesce(r.weight, 1.0)) AS adj_weight
    """
    
    connectivity_scores = {}
    with neo4j_driver.get_session() as session:
        res = session.run(query_graph, candidate_ids=candidate_ids)
        for record in res:
            connectivity_scores[record['cid']] = record['adj_weight'] or 0.0
            
    # 5. Final Scoring
    # Formula: final_score = α * vector_score + β * graph_score / (1 + hop_count)
    
    # Normalize vector scores
    max_v_score = max([r.score for r in candidates.values()]) if candidates else 1.0
    if max_v_score == 0: max_v_score = 1.0
    
    # Normalize connectivity scores
    max_c_score = max(connectivity_scores.values()) if connectivity_scores else 1.0
    if max_c_score == 0: max_c_score = 1.0

    final_results = []
    for doc_id, r in candidates.items():
        # Vector Score Component
        v_score_norm = r.score / max_v_score
        
        # Graph Score Component
        # Base graph score is connectivity (how central/connected is this node?)
        c_score_raw = connectivity_scores.get(doc_id, 0)
        c_score_norm = c_score_raw / max_c_score
        
        # Hop Count Logic
        # If it was found via query entity expansion, hops=1 (close).
        # If it was ONLY vector match, we treat it as hops=0 (it IS the match).
        # Wait, usually hops=0 is better. 
        # Let's say:
        # - Direct Vector Match: hops=0
        # - Neighbor of Query Entity: hops=1
        # If both, we take min(hops) -> 0.
        # But we want to boost nodes that are BOTH vector relevant AND graph relevant.
        
        hops = r.graph_info.get("hops", 0) # Default to 0 if vector match
        
        # Refined Formula:
        # graph_score = (connectivity_norm + expansion_bonus) / (1 + hops)
        # expansion_bonus = 1.0 if found via query entity, else 0.0
        
        expansion_bonus = 1.0 if "expansion_weight" in r.graph_info else 0.0
        
        # Combine
        # We use the user's formula: graph_score / (1 + hop_count)
        # Here graph_score is the connectivity score.
        
        g_component = (c_score_norm + expansion_bonus) / (1 + hops)
        
        final_score = (vector_weight * v_score_norm) + (graph_weight * g_component)
        
        r.score = final_score
        r.graph_info.update({
            "vector_score_norm": v_score_norm,
            "connectivity_score_norm": c_score_norm,
            "hops": hops,
            "expansion_bonus": expansion_bonus
        })
        final_results.append(r)
        
    # Sort by final score
    final_results.sort(key=lambda x: x.score, reverse=True)
    return final_results[:top_k]
