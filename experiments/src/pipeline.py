"""Pipeline end-to-end del benchmark.

Extrae texto de los PDFs en data/source_documents/, arma (o reusa) el
índice RAG, corre las 300 preguntas de data/benchmark_300.json contra un
modelo y guarda los resultados crudos en results/raw/<model_key>.json.

Uso (desde experiments/):
    python src/pipeline.py qwen2.5-32b

Formato esperado de benchmark_300.json: una lista de objetos con al menos
las claves "id", "categoria", "pregunta", "ground_truth" (ver
scripts/excel_to_json.py, que genera este archivo). Si el archivo está
vacío o no existe, el pipeline falla temprano con un mensaje claro en vez
de arrancar a cargar el modelo en vano.
"""

import json
import sys
import time
import traceback
from pathlib import Path

import faiss
import pypdf

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SRC_DIR / "models"))  # run_model.py hace `from config import ...` (import plano)

from rag.chunking import chunk_text
from rag.embeddings import load_embedding_model, embed_chunks, embed_query
from rag.retrieval import build_index, retrieve_top_k
from config import MODEL_CONFIGS
from run_model import get_generate_fn

PROJECT_ROOT = SRC_DIR.parent  # experiments/
SOURCE_DOCS_DIR = PROJECT_ROOT / "data" / "source_documents"
QA_PATH = PROJECT_ROOT / "data" / "benchmark_300.json"
INDEX_DIR = PROJECT_ROOT / "data" / "rag_index"
RESULTS_DIR = PROJECT_ROOT / "results" / "raw"
TOP_K = 5


# ---------------------------------------------------------------------
# 1. Extracción de PDFs + índice RAG (con caché en disco)
# ---------------------------------------------------------------------

def extract_pdf_text(path: Path) -> str:
    reader = pypdf.PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _build_rag_index_from_scratch(source_dir: Path):
    pdf_paths = sorted(source_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No se encontraron PDFs en {source_dir}")

    print(f"Extrayendo texto de {len(pdf_paths)} documentos...")
    all_chunks = []
    for pdf_path in pdf_paths:
        text = extract_pdf_text(pdf_path)
        doc_chunks = chunk_text(text, source_document=pdf_path.name)
        all_chunks.extend(doc_chunks)
        print(f"  {pdf_path.name}: {len(doc_chunks)} chunks")

    print(f"Total: {len(all_chunks)} chunks. Generando embeddings (BGE-M3)...")
    embedding_model = load_embedding_model()
    embeddings = embed_chunks(all_chunks, embedding_model)

    print("Construyendo índice FAISS...")
    index = build_index(embeddings)

    return index, all_chunks, embedding_model


def build_or_load_rag_index(source_dir: Path = SOURCE_DOCS_DIR, index_dir: Path = INDEX_DIR):
    """Reusa el índice guardado en `index_dir` si existe y carga bien;
    si no, lo reconstruye desde cero a partir de los PDFs y lo guarda.

    Reconstruir el índice (extracción + embeddings de 7 PDFs, algunos de
    >40MB) es el paso más lento del pipeline. Cachearlo en disco importa
    porque hoy vamos a correr varios modelos seguidos contra el mismo
    corpus — no tiene sentido pagar ese costo por cada uno.
    """
    index_path = index_dir / "index.faiss"
    chunks_path = index_dir / "chunks.json"

    embedding_model = load_embedding_model()

    if index_path.exists() and chunks_path.exists():
        try:
            print(f"Cargando índice RAG existente de {index_dir}...")
            index = faiss.read_index(str(index_path))
            with open(chunks_path, encoding="utf-8") as f:
                chunks = json.load(f)
            print(f"  {len(chunks)} chunks cargados")
            return index, chunks, embedding_model
        except Exception:
            print("No se pudo cargar el índice existente, se reconstruye desde cero.")
            traceback.print_exc()

    index, chunks, embedding_model = _build_rag_index_from_scratch(source_dir)

    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)
    print(f"Índice guardado en {index_dir} (se reusa en la próxima corrida)")

    return index, chunks, embedding_model


# ---------------------------------------------------------------------
# 2. Dataset de preguntas
# ---------------------------------------------------------------------

def load_qa_dataset(path: Path = QA_PATH) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}")

    with open(path, encoding="utf-8") as f:
        raw = f.read()

    if not raw.strip():
        raise ValueError(
            f"{path} está vacío — todavía no se cargaron las 300 preguntas del "
            f"benchmark. Hace falta poblarlo antes de correr el pipeline."
        )

    data = json.loads(raw)
    if not data:
        raise ValueError(f"{path} no tiene preguntas (lista vacía).")

    required_keys = {"pregunta", "ground_truth"}
    missing = required_keys - set(data[0])
    if missing:
        raise ValueError(
            f"Cada pregunta debe tener al menos las claves {required_keys}; "
            f"faltan {missing} en el primer elemento de {path}."
        )

    return data


# ---------------------------------------------------------------------
# 3. Construcción del prompt
# ---------------------------------------------------------------------

def build_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    context = "\n\n".join(
        f"[Fuente: {c['source_document']}]\n{c['text']}" for c in retrieved_chunks
    )
    return (
        "Responde la siguiente pregunta usando únicamente la información "
        "del contexto proporcionado. Si la respuesta no está en el "
        "contexto, indica que no tienes suficiente información.\n\n"
        f"Contexto:\n{context}\n\n"
        f"Pregunta: {question}\n"
        "Respuesta:"
    )


# ---------------------------------------------------------------------
# 4. Loop del benchmark
# ---------------------------------------------------------------------

def run_pipeline(model_key: str):
    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"Modelo desconocido: {model_key}. Opciones: {list(MODEL_CONFIGS)}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f"{model_key}.json"

    # Validar el dataset ANTES de cargar el índice/modelo, para fallar rápido.
    qa_dataset = load_qa_dataset()

    index, chunks, embedding_model = build_or_load_rag_index()

    print(f"Cargando modelo {model_key}...")
    generate_fn = get_generate_fn(model_key)
    print("Modelo cargado. Empezando benchmark...")

    results = []
    total = len(qa_dataset)
    start = time.time()

    for i, item in enumerate(qa_dataset, start=1):
        pregunta_id = item.get("id", i)
        categoria = item.get("categoria")
        pregunta = item["pregunta"]
        ground_truth = item.get("ground_truth")

        result = {
            "pregunta_id": pregunta_id,
            "categoria": categoria,
            "pregunta": pregunta,
            "ground_truth": ground_truth,
            "respuesta_generada": None,
            "chunks_usados": None,
            "error": None,
        }

        try:
            query_embedding = embed_query(pregunta, embedding_model)
            retrieved = retrieve_top_k(query_embedding, index, chunks, k=TOP_K)
            prompt = build_prompt(pregunta, retrieved)
            answer = generate_fn(prompt)

            result["respuesta_generada"] = answer
            result["chunks_usados"] = [
                {"source_document": c["source_document"], "chunk_index": c["chunk_index"]}
                for c in retrieved
            ]
        except Exception as e:
            result["error"] = f"{type(e).__name__}: {e}"
            traceback.print_exc()

        results.append(result)

        # Guardar incrementalmente: no perder todo si se corta a mitad de camino.
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        if i % 10 == 0 or i == total:
            elapsed = time.time() - start
            print(f"{i}/{total} procesadas ({elapsed:.0f}s transcurridos)")

    n_errors = sum(1 for r in results if r["error"])
    total_elapsed = time.time() - start
    print(
        f"\nListo: {total} preguntas, {n_errors} errores, "
        f"{total_elapsed:.0f}s total. Resultados en {output_path}"
    )
    return output_path


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in MODEL_CONFIGS:
        print(f"Uso: python pipeline.py <model_key>\nOpciones: {list(MODEL_CONFIGS)}")
        sys.exit(1)

    run_pipeline(sys.argv[1])
