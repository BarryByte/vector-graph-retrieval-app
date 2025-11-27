import uuid
from app.database import neo4j_driver, faiss_index
from app.services.embedding import embedding_service
from app.models import DocumentInput, Document, EdgeInput
import logging
import spacy

logger = logging.getLogger(__name__)

# Load Spacy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.warning("Spacy model 'en_core_web_sm' not found. NER will be disabled.")
    nlp = None

def clean_text(text: str) -> str:
    """Basic text cleaning: remove extra whitespace, strip."""
    # We can add more advanced cleaning here (regex, removing special chars) if needed
    cleaned = " ".join(text.split())
    return cleaned

def chunk_text(text: str, chunk_size: int = 500) -> list[str]:
    """Simple chunking by character count (for now). 
    In a real app, we'd use sentence boundary detection or recursive character splitting.
    """
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def ingest_document(doc_input: DocumentInput) -> Document:
    logger.info(f"--- Starting Ingestion for Document: {doc_input.title} ---")
    
    # 1. Clean Text
    cleaned_text = clean_text(doc_input.text)
    logger.info(f"Original Text Length: {len(doc_input.text)} -> Cleaned Length: {len(cleaned_text)}")
    
    # 2. Chunking (Optional: For now we still ingest the whole doc as one node, 
    # but we could create multiple nodes per chunk. 
    # For this MVP, let's keep it simple: 1 Doc Node, but maybe store chunks in metadata or just log it)
    chunks = chunk_text(cleaned_text)
    logger.info(f"Generated {len(chunks)} chunks.")

    # 3. Generate ID
    doc_id = str(uuid.uuid4())
    
    # 4. Generate Embedding (on the full cleaned text for the main node)
    logger.info("Generating embedding...")
    embedding = embedding_service.encode(cleaned_text)
    
    # 5. Add to FAISS
    vector_id = faiss_index.add(embedding, doc_id)
    logger.info(f"Added to FAISS with Vector ID: {vector_id}")
    
    # 6. Create Node in Neo4j
    query = """
    CREATE (d:Document {
        id: $id,
        text: $text,
        title: $title,
        vector_id: $vector_id
    })
    SET d += $metadata
    RETURN d
    """
    
    with neo4j_driver.get_session() as session:
        session.run(query, 
                    id=doc_id, 
                    text=cleaned_text, 
                    title=doc_input.title, 
                    vector_id=vector_id,
                    metadata=doc_input.metadata)
        logger.info(f"Created Document Node in Neo4j: {doc_id}")
        
        # 7. NER Extraction & Edge Creation
        if nlp:
            logger.info("Running NER extraction...")
            doc = nlp(cleaned_text)
            entities_found = 0
            for ent in doc.ents:
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
                entities_found += 1
            logger.info(f"Extracted and linked {entities_found} entities.")
        
    return Document(
        id=doc_id,
        text=cleaned_text,
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
        result = session.run(query, 
                    source_id=edge_input.source, 
                    target_id=edge_input.target, 
                    weight=edge_input.weight,
                    metadata=edge_input.metadata)
        record = result.single()
        if not record:
            logger.warning(f"Could not create edge between {edge_input.source} and {edge_input.target}")
            return None
        return record['r']
