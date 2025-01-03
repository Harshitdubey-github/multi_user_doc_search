import os
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

class EmbeddingIndex:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.doc_texts = []

    def build_index(self, texts):
        if not texts:
            print("No texts provided for building index.")
            return
        
        try:
            embeddings = self.model.encode(texts, show_progress_bar=True)
            embeddings = np.array(embeddings, dtype="float32")
        except Exception as e:
            print(f"Error during embeddings generation: {e}")
            return
        
        if embeddings.shape[0] == 0:
            print("Embedding generation failed: No embeddings created.")
            return

        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        self.doc_texts = texts


    def search(self, query, top_k=3):
        if self.index is None:
            return []
        query_emb = self.model.encode([query])
        query_emb = np.array(query_emb, dtype="float32")
        
        distances, indices = self.index.search(query_emb, top_k)
        print(f"Search distances: {distances}")
        print(f"Search indices: {indices}")

        results = []
        for idx in indices[0]:
            if idx < len(self.doc_texts):
                results.append(self.doc_texts[idx])
        return results


def chunk_text(text, chunk_size=300, overlap=50):
    """
    A simple chunking approach to split a large text into overlapping chunks
    for more fine-grained retrieval. 
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = words[start:end]
        chunks.append(" ".join(chunk))
        start += (chunk_size - overlap)
    return chunks
