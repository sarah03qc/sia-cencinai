"""Carga BGE-M3 y genera embeddings normalizados para chunks y preguntas.

Se normalizan los vectores porque el índice FAISS usado en retrieval.py es
IndexFlatL2 (distancia euclidiana): con vectores normalizados, la distancia
L2 queda relacionada monótonamente con la similitud coseno.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-m3"


def load_embedding_model(device: str | None = None) -> SentenceTransformer:
    """Carga BGE-M3. `device` puede ser "cuda", "cpu", o None (autodetecta)."""
    return SentenceTransformer(MODEL_NAME, device=device)


def embed_chunks(chunks: list[dict], model: SentenceTransformer, batch_size: int = 32) -> np.ndarray:
    """Genera embeddings para una lista de chunks (dicts con clave "text").

    Returns:
        np.ndarray de forma (n_chunks, dimensión), dtype float32, normalizado.
    """
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    return embeddings.astype("float32")


def embed_query(question: str, model: SentenceTransformer) -> np.ndarray:
    """Genera el embedding de una sola pregunta, listo para buscar en FAISS."""
    embedding = model.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embedding.astype("float32")
