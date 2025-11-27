# File: app/database.py
from neo4j import GraphDatabase
import faiss
import numpy as np
from app.config import settings

class Neo4jDriver:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            settings.NEO4J_URI, 
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
        )

    def close(self):
        self.driver.close()

    def get_session(self):
        return self.driver.session()

import os
import pickle

class FaissIndex:
    def __init__(self):
        self.dimension = settings.VECTOR_DIM
        self.index = faiss.IndexFlatIP(self.dimension) # Inner Product (Cosine Similarity if normalized)
        self.id_map = {} # Maps FAISS ID to Document ID
        self.current_id = 0
        self.index_path = "data/faiss_index.bin"
        self.map_path = "data/faiss_map.pkl"
        self.load()

    def add(self, embedding: np.ndarray, doc_id: str):
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        faiss.normalize_L2(embedding)
        self.index.add(embedding)
        self.id_map[self.current_id] = doc_id
        vector_id = self.current_id
        self.current_id += 1
        self.save() # Auto-save on add (for simple persistence)
        return vector_id

    def search(self, query_vector: np.ndarray, top_k: int):
        if self.index.ntotal == 0:
            return [], []
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        faiss.normalize_L2(query_vector)
        distances, indices = self.index.search(query_vector, top_k)
        return distances[0], indices[0]

    def save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.map_path, "wb") as f:
            pickle.dump({"id_map": self.id_map, "current_id": self.current_id}, f)

    def load(self):
        if os.path.exists(self.index_path) and os.path.exists(self.map_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.map_path, "rb") as f:
                data = pickle.load(f)
                self.id_map = data["id_map"]
                self.current_id = data["current_id"]

    def get_vector(self, vector_id: int) -> list:
        try:
            return self.index.reconstruct(vector_id).tolist()
        except:
            return []

    def count(self):
        return self.index.ntotal

# Global instances
neo4j_driver = Neo4jDriver()
faiss_index = FaissIndex()
