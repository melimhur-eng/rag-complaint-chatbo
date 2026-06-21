# src/embedding.py

from sentence_transformers import SentenceTransformer


MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


def load_embedding_model():
    return SentenceTransformer(MODEL_NAME)