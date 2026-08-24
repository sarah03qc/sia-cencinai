"""Indexa los embeddings de los 7 documentos en un único FAISS index (IndexFlatL2)
y recupera el top-k de chunks más relevantes para una pregunta.

Se usa un solo índice combinado para todo el corpus: el retrieval no sabe de
antemano a qué documento pertenece la pregunta, busca en todos los chunks juntos.
"""

import faiss
import numpy as np


def build_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """Construye un IndexFlatL2 a partir de los embeddings de todos los chunks."""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index


def retrieve_top_k(query_embedding: np.ndarray, index: faiss.IndexFlatL2, chunks: list[dict], k: int = 5) -> list[dict]:
    """Busca los k chunks más cercanos a `query_embedding` en `index`.

    `chunks` debe estar en el mismo orden en que se pasaron a `build_index`,
    para que los índices que devuelve FAISS mapeen correctamente.

    Returns:
        Lista de hasta k dicts (copias de los chunks originales) con una
        clave adicional "distance" (float, distancia L2 al query).
    """
    distances, indices = index.search(query_embedding, k)

    results = []
    for rank, idx in enumerate(indices[0]):
        if idx == -1:
            continue
        chunk = dict(chunks[idx])
        chunk["distance"] = float(distances[0][rank])
        results.append(chunk)

    return results
