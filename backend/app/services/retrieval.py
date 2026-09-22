import os
from typing import List, Dict, Any, Tuple
import chromadb
from sentence_transformers import SentenceTransformer
from app.core.config import settings
from app.utils.logging_config import logger

class RetrievalService:
    def __init__(self):
        self.vector_store_path = settings.VECTOR_STORE_PATH
        self.collection_name = settings.COLLECTION_NAME
        self.model_name = settings.EMBEDDING_MODEL_NAME
        self.client = None
        self.collection = None
        self.embed_model = None

    def initialize(self):
        logger.info(f"Initializing embedding model: {self.model_name}")
        self.embed_model = SentenceTransformer(self.model_name)

        logger.info(f"Connecting to ChromaDB persistent vector store at: {self.vector_store_path}")
        os.makedirs(self.vector_store_path, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.vector_store_path)
        self.collection = self.client.get_or_create_collection(name=self.collection_name)
        doc_count = self.collection.count()
        logger.info(f"Connected to collection '{self.collection_name}'. Current document count: {doc_count}")

    def retrieve(self, query_text: str, top_k: int = None) -> Tuple[List[str], List[Dict[str, Any]]]:
        if top_k is None:
            top_k = settings.TOP_K

        if self.collection is None or self.embed_model is None:
            self.initialize()

        count = self.collection.count()
        if count == 0:
            logger.warning("Vector store is currently empty.")
            return [], []

        actual_k = min(top_k, count)
        query_embedding = self.embed_model.encode([query_text]).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=actual_k
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        # Attach retrieval distance to each metadata record for relevance scoring
        for meta, dist in zip(metadatas, distances):
            meta["distance"] = dist

        return documents, metadatas

retrieval_service = RetrievalService()
