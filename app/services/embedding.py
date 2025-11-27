from sentence_transformers import SentenceTransformer
from app.config import settings
import numpy as np

class EmbeddingService:
    def __init__(self):
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL)

    def encode(self, text: str) -> np.ndarray:
        return self.model.encode(text)

embedding_service = EmbeddingService()
