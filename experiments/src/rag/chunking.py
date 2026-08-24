"""Divide el texto extraído de un documento PDF en chunks de tamaño fijo con overlap.

Cada chunk conserva metadata (documento fuente + índice) para poder trazar
de qué documento salió cada fragmento recuperado en el retrieval.
"""


def chunk_text(text: str, source_document: str, chunk_size: int = 300, overlap: int = 50) -> list[dict]:
    """Parte `text` en chunks de ~`chunk_size` palabras con `overlap` palabras de solapamiento.

    Args:
        text: texto plano ya extraído del PDF (sin limpieza adicional).
        source_document: nombre del documento fuente (ej. "PRO-AIAIM-P-01.pdf"),
            se guarda en cada chunk para trazabilidad.
        chunk_size: cantidad de palabras por chunk.
        overlap: cantidad de palabras que se repiten entre un chunk y el siguiente.

    Returns:
        Lista de dicts: {"text": str, "source_document": str, "chunk_index": int}.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap debe ser menor que chunk_size")

    words = text.split()
    if not words:
        return []

    step = chunk_size - overlap
    chunks = []
    chunk_index = 0
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append({
            "text": " ".join(chunk_words),
            "source_document": source_document,
            "chunk_index": chunk_index,
        })
        chunk_index += 1
        if end >= len(words):
            break
        start += step

    return chunks
