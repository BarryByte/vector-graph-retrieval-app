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

def recursive_chunking(text: str, chunk_size: int = 400, overlap: int = 50) -> list[str]:
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
            
            # 6. Semantic Edge Creation (RELATED_TO)
            # Find similar existing chunks
            # We query FAISS for top 5 similar chunks
            # Note: This might return the current chunk itself if FAISS index isn't updated instantly or if we just added it.
            # faiss_index.add updates the index immediately.
            distances, indices = faiss_index.search(embedding, top_k=5)
            
            for j, idx in enumerate(indices):
                if idx != -1 and idx != vector_id: # Exclude self
                    sim_score = float(distances[j]) # Inner product is cosine similarity if normalized
                    if sim_score > 0.85:
                        neighbor_id = faiss_index.id_map.get(idx)
                        if neighbor_id:
                            # Create semantic edge
                            rel_query = """
                            MATCH (a:Document {id: $id})
                            MATCH (b:Document {id: $neighbor_id})
                            MERGE (a)-[r:RELATED_TO]->(b)
                            SET r.weight = $weight, r.type = 'semantic'
                            """
                            session.run(rel_query, id=doc_id, neighbor_id=neighbor_id, weight=sim_score)
                            logger.info(f"Created Semantic Edge: {doc_id} <-> {neighbor_id} (Score: {sim_score:.4f})")

            # 7. NER Extraction & Edge Creation
            if nlp:
                doc = nlp(chunk_text)
                for ent in doc.ents:
                    # Only specific types
                    if ent.label_ in ["ORG", "PERSON", "GPE", "DATE"]:
                        # Create Entity Node
                        merge_entity_query = """
                        MERGE (e:Entity {name: $name, type: $type})
                        RETURN e
                        """
                        session.run(merge_entity_query, name=ent.text, type=ent.label_)
                        
                        # Create MENTIONS relationship
                        create_rel_query = """
                        MATCH (d:Document {id: $doc_id})
                        MATCH (e:Entity {name: $name, type: $type})
                        MERGE (d)-[r:MENTIONS]->(e)
                        SET r.weight = 1.0
                        """
                        session.run(create_rel_query, doc_id=doc_id, name=ent.text, type=ent.label_)

    return Document(
        id=first_doc_id if first_doc_id else "error",
        text=cleaned_text, # Return full text
        title=doc_input.title,
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
    query = "MATCH (n {id: $id}) RETURN n"
    with neo4j_driver.get_session() as session:
        res = session.run(query, id=node_id)
        record = res.single()
        if record:
            return dict(record['n'])
    return None

def update_node(node_id: str, doc_input: DocumentInput):
    # 1. Update Neo4j
    query = """
    MATCH (n {id: $id})
    SET n.text = $text, n.title = $title
    SET n += $metadata
    RETURN n
    """
    with neo4j_driver.get_session() as session:
        res = session.run(query, id=node_id, text=doc_input.text, title=doc_input.title, metadata=doc_input.metadata)
        record = res.single()
        if not record:
            return None
        node = record['n']
    
    # 2. Update FAISS if text changed and it's a Document
    if "Document" in node.labels and node.get('vector_id') is not None:
        embedding = embedding_service.encode(doc_input.text)
        faiss_index.update_document(node_id, embedding)
        
    return dict(node)

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
