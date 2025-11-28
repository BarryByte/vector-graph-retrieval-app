# File: app/services/ingestion.py
import uuid
from app.database import neo4j_driver, faiss_index
from app.services.embedding import embedding_service
from app.models import DocumentInput, Document, EdgeInput
import logging
import spacy
from bs4 import BeautifulSoup
from langdetect import detect
import numpy as np
import ftfy



logger = logging.getLogger(__name__)

# Load Spacy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.warning("Spacy model 'en_core_web_sm' not found. NER will be disabled.")
    nlp = None

def clean_text(text: str) -> str:
    """
    Advanced text cleaning:
    1. Remove HTML tags using BeautifulSoup.
    2. Remove extra whitespace.
    """
    # 1. HTML Cleaning
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text(separator=" ")
    
    # 2. Whitespace Cleaning
    cleaned = " ".join(text.split())

    #3. Fix Text
    cleaned = ftfy.fix_text(cleaned)

    return cleaned

def recursive_chunking(text: str, chunk_size: int = 256, overlap: int = 12) -> list[str]:
    """
    Recursive chunking strategy:
    1. Split by paragraphs (double newline).
    2. If chunk > chunk_size, split by sentences.
    3. If still > chunk_size, split by words.
    4. Apply overlap.
    """
    # Simple implementation for MVP: Split by words with overlap
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

def _create_semantic_edges(doc_id: str, embedding: np.ndarray, vector_id: int):
    """Creates RELATED_TO edges based on vector similarity."""
    distances, indices = faiss_index.search(embedding, top_k=5)
    
    with neo4j_driver.get_session() as session:
        for j, idx in enumerate(indices):
            # Exclude self: if vector_id is valid, use it. Otherwise, check if neighbor_id is the same as doc_id.
            neighbor_id = faiss_index.id_map.get(idx)
            if idx != -1 and (vector_id != -1 and idx != vector_id) or (vector_id == -1 and neighbor_id != doc_id):
                sim_score = float(distances[j])
                if sim_score > 0.85:
                    if neighbor_id:
                        rel_query = """
                        MATCH (a:Document {id: $id})
                        MATCH (b:Document {id: $neighbor_id})
                        MERGE (a)-[r:RELATED_TO]->(b)
                        SET r.weight = $weight, r.type = 'semantic'
                        """
                        session.run(rel_query, id=doc_id, neighbor_id=neighbor_id, weight=sim_score)
                        logger.info(f"Created Semantic Edge: {doc_id} <-> {neighbor_id} (Score: {sim_score:.4f})")

def _extract_and_link_entities(doc_id: str, text: str):
    """Extracts entities using NER and creates MENTIONS edges."""
    if not nlp: return

    doc = nlp(text)
    with neo4j_driver.get_session() as session:
        for ent in doc.ents:
            if ent.label_ in ["ORG", "PERSON", "GPE", "DATE"]:
                # Create Entity Node
                # We use ON CREATE SET to assign an ID only if it's a new node
                merge_entity_query = """
                MERGE (e:Entity {name: $name, type: $type})
                ON CREATE SET e.id = $id
                RETURN e
                """
                # Generate a UUID for the entity (will be used only if created)
                ent_id = str(uuid.uuid4())
                session.run(merge_entity_query, name=ent.text, type=ent.label_, id=ent_id)
                
                # Create MENTIONS relationship
                create_rel_query = """
                MATCH (d:Document {id: $doc_id})
                MATCH (e:Entity {name: $name, type: $type})
                MERGE (d)-[r:MENTIONS]->(e)
                SET r.weight = 1.0
                """
                session.run(create_rel_query, doc_id=doc_id, name=ent.text, type=ent.label_)

def ingest_document(doc_input: DocumentInput) -> Document:
    logger.info(f"--- Starting Ingestion for Document: {doc_input.title} ---")
    
    # 1. Clean Text
    cleaned_text = clean_text(doc_input.text)
    
    # Optional: Language Detection
    try:
        lang = detect(cleaned_text)
        logger.info(f"Detected Language: {lang}")
    except:
        lang = "unknown"

    # 2. Chunking
    # We treat each chunk as a separate "Document" node for granular retrieval
    chunks = recursive_chunking(cleaned_text)
    logger.info(f"Generated {len(chunks)} chunks.")

    first_doc_id = None

    for i, chunk_text in enumerate(chunks):
        doc_id = str(uuid.uuid4())
        if i == 0: first_doc_id = doc_id
        
        chunk_title = f"{doc_input.title} (Chunk {i+1})" if doc_input.title else f"Chunk {i+1}"
        
        # 3. Generate Embedding
        embedding = embedding_service.encode(chunk_text)
        
        # 4. Add to FAISS
        vector_id = faiss_index.add(embedding, doc_id)
        
        # 5. Create Node in Neo4j
        query = """
        CREATE (d:Document {
            id: $id,
            text: $text,
            title: $title,
            vector_id: $vector_id,
            lang: $lang,
            chunk_index: $chunk_index
        })
        SET d += $metadata
        RETURN d
        """
        
        with neo4j_driver.get_session() as session:
            session.run(query, 
                        id=doc_id, 
                        text=chunk_text, 
                        title=chunk_title, 
                        vector_id=vector_id,
                        lang=lang,
                        chunk_index=i,
                        metadata=doc_input.metadata)
            
        # 6. Semantic Edge Creation
        _create_semantic_edges(doc_id, embedding, vector_id)

        # 7. NER Extraction & Edge Creation
        _extract_and_link_entities(doc_id, chunk_text)

    return Document(
        id=first_doc_id if first_doc_id else "error",
        text=cleaned_text, # Return full ext
        metadata=doc_input.metadata,
        vector_id=vector_id
    )

def create_edge(edge_input: EdgeInput):
    query = f"""
    MATCH (source {{id: $source_id}})
    MATCH (target {{id: $target_id}})
    MERGE (source)-[r:{edge_input.type}]->(target)
    SET r.weight = $weight
    SET r += $metadata
    RETURN r
    """
    
    with neo4j_driver.get_session() as session:
        logger.info(f"Creating edge from {edge_input.source} to {edge_input.target}")
        result = session.run(query, 
                    source_id=edge_input.source, 
                    target_id=edge_input.target, 
                    weight=edge_input.weight,
                    metadata=edge_input.metadata)
        record = result.single()
        if not record:
            logger.error(f"Could not create edge between {edge_input.source} and {edge_input.target}. Nodes might not exist.")
            return None
        return record['r']

def get_node(node_id: str):
    query = """
    MATCH (n {id: $id})
    OPTIONAL MATCH (n)-[r]->(target)
    RETURN n, collect({
        target_id: coalesce(target.id, elementId(target)), 
        type: type(r), 
        weight: coalesce(r.weight, 1.0)
    }) as relationships
    """
    with neo4j_driver.get_session() as session:
        res = session.run(query, id=node_id)
        record = res.single()
        if record:
            node_data = dict(record['n'])
            # Filter out empty relationships (if OPTIONAL MATCH found nothing)
            rels = [r for r in record['relationships'] if r['target_id'] is not None]
            node_data['relationships'] = rels
            return node_data
    return None

def update_node(node_id: str, update_data: "NodeUpdate"):
    # 1. Update Neo4j
    # Build dynamic SET clause
    set_clauses = []
    params = {"id": node_id}
    
    if update_data.text is not None:
        set_clauses.append("n.text = $text")
        params["text"] = update_data.text
    if update_data.title is not None:
        set_clauses.append("n.title = $title")
        params["title"] = update_data.title
    if update_data.metadata:
        set_clauses.append("n += $metadata")
        params["metadata"] = update_data.metadata
        
    if not set_clauses:
        # Nothing to update in Neo4j, but maybe embedding regen requested?
        pass
    else:
        query = f"MATCH (n {{id: $id}}) SET {', '.join(set_clauses)} RETURN n"
        with neo4j_driver.get_session() as session:
            res = session.run(query, **params)
            record = res.single()
            if not record:
                return None
            # node = record['n'] # We'll fetch fresh below anyway
            
    # Fetch fresh node to check labels and current text
    node_data = get_node(node_id)
    if not node_data: return None

    # 2. Update FAISS & Relationships if requested
    if update_data.regen_embedding and "Document" in node_data.get('labels', ['Document']) and node_data.get('vector_id') is not None:
        text_to_embed = update_data.text if update_data.text is not None else node_data.get('text')
        if text_to_embed:
            # A. Update Vector
            embedding = embedding_service.encode(text_to_embed)
            faiss_index.update_document(node_id, embedding)
            logger.info(f"Regenerated embedding for node {node_id}")
            
            # B. Delete Old Relationships
            del_query = """
            MATCH (n {id: $id})-[r:RELATED_TO|MENTIONS]->()
            DELETE r
            """
            with neo4j_driver.get_session() as session:
                session.run(del_query, id=node_id)
                logger.info(f"Deleted old relationships for node {node_id}")
            
            # C. Re-create Relationships
            _create_semantic_edges(node_id, embedding, -1) 
            _extract_and_link_entities(node_id, text_to_embed)
            
            # Refresh node data to include new relationships
            node_data = get_node(node_id)
        
    return node_data

def delete_node(node_id: str):
    # 1. Delete from Neo4j
    query = "MATCH (n {id: $id}) DETACH DELETE n"
    with neo4j_driver.get_session() as session:
        session.run(query, id=node_id)
        
    # 2. Remove from FAISS
    faiss_index.remove_document(node_id)
    return True

def get_edge(edge_id: str):
    # Using elementId for edge lookup
    query = "MATCH ()-[r]-() WHERE elementId(r) = $id RETURN r"
    with neo4j_driver.get_session() as session:
        res = session.run(query, id=edge_id)
        record = res.single()
        if record:
            r = record['r']
            return {
                "id": r.element_id,
                "type": r.type,
                "properties": dict(r),
                # Try to get 'id' property of nodes, fallback to elementId
                "source": r.start_node.get('id') if 'id' in r.start_node else r.start_node.element_id,
                "target": r.end_node.get('id') if 'id' in r.end_node else r.end_node.element_id
            }
    return None
