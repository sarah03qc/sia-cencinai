import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RAW_PATH = ROOT / "results" / "raw" / "qwen2.5-32b.json"
BENCHMARK_PATH = ROOT / "data" / "benchmark_final_299.json"
EVIDENCE_PATH = ROOT / "data" / "evidence_annotations_final_299.json"
CHUNKS_PATH = ROOT / "data" / "rag_index" / "chunks.json"

OUT_DIR = ROOT / "results" / "retrieval_audit"
OUT_JSON = OUT_DIR / "historical_top5_final_299.json"
OUT_CSV = OUT_DIR / "historical_top5_final_299.csv"


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def key(obj):
    return (
        obj["source_document"],
        int(obj["chunk_index"])
    )


benchmark = load_json(BENCHMARK_PATH)
evidence = load_json(EVIDENCE_PATH)
raw = load_json(RAW_PATH)
chunks = load_json(CHUNKS_PATH)

benchmark_by_id = {
    x["id"]: x
    for x in benchmark
}

evidence_by_id = {
    x["id"]: x
    for x in evidence
}

raw_by_id = {
    str(x["pregunta_id"]): x
    for x in raw
}


# ------------------------------------------------------------
# Mapeo histórico de chunks
# ------------------------------------------------------------

chunk_by_key = {}

for global_idx, chunk in enumerate(chunks):

    k = key(chunk)

    if k in chunk_by_key:
        raise ValueError(
            f"Chunk duplicado encontrado: {k}"
        )

    chunk_copy = dict(chunk)
    chunk_copy["global_chunk_id"] = global_idx

    chunk_by_key[k] = chunk_copy


# ------------------------------------------------------------
# Validaciones
# ------------------------------------------------------------

if len(benchmark) != 299:
    raise ValueError(
        f"Esperaba 299 preguntas, encontré {len(benchmark)}"
    )

if "sino_023" in benchmark_by_id:
    raise ValueError(
        "sino_023 no debería estar en benchmark_final_299"
    )


OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

json_output = []
csv_output = []
missing_chunks = []


# ------------------------------------------------------------
# Recuperar Top-5 HISTÓRICO
# ------------------------------------------------------------

for n, item in enumerate(benchmark, start=1):

    qid = item["id"]

    if qid not in raw_by_id:
        raise ValueError(
            f"{qid} no aparece en resultados raw"
        )

    if qid not in evidence_by_id:
        raise ValueError(
            f"{qid} no aparece en evidence annotations"
        )

    raw_item = raw_by_id[qid]

    historical_refs = (
        raw_item.get("chunks_usados") or []
    )

    if len(historical_refs) != 5:
        raise ValueError(
            f"{qid}: esperaba 5 chunks históricos, "
            f"encontré {len(historical_refs)}"
        )

    retrieved = []

    for rank, ref in enumerate(
        historical_refs,
        start=1
    ):

        k = key(ref)

        if k not in chunk_by_key:
            missing_chunks.append(
                (qid, rank, k)
            )
            continue

        chunk = chunk_by_key[k]

        row = {
            "rank": rank,
            "global_chunk_id":
                chunk["global_chunk_id"],
            "source_document":
                chunk["source_document"],
            "chunk_index":
                int(chunk["chunk_index"]),
            "text":
                chunk["text"],
        }

        retrieved.append(row)

        ev = evidence_by_id[qid]

        csv_output.append({
            "question_id":
                qid,

            "categoria":
                item["categoria"],

            "pregunta":
                item["pregunta"],

            "ground_truth":
                item["ground_truth"],

            "gold_source_document":
                ev["source_document"],

            "gold_page_or_section":
                ev["page_or_section"],

            "gold_supporting_excerpt":
                ev["supporting_excerpt"],

            "rank":
                rank,

            "retrieved_global_chunk_id":
                chunk["global_chunk_id"],

            "retrieved_source_document":
                chunk["source_document"],

            "retrieved_chunk_index":
                int(chunk["chunk_index"]),

            "retrieved_text":
                chunk["text"],
        })


    json_output.append({
        "id":
            qid,

        "categoria":
            item["categoria"],

        "pregunta":
            item["pregunta"],

        "ground_truth":
            item["ground_truth"],

        "authoritative_evidence":
            evidence_by_id[qid],

        "top5_historical":
            retrieved,
    })


    if n % 25 == 0 or n == len(benchmark):
        print(
            f"{n}/{len(benchmark)} preguntas procesadas"
        )


if missing_chunks:
    print("\nERROR: faltan chunks:")
    for x in missing_chunks[:20]:
        print(x)

    raise ValueError(
        f"No se encontraron {len(missing_chunks)} chunks históricos"
    )


if len(csv_output) != 299 * 5:
    raise ValueError(
        f"Esperaba 1495 filas de retrieval, "
        f"obtuve {len(csv_output)}"
    )


with open(
    OUT_JSON,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        json_output,
        f,
        ensure_ascii=False,
        indent=2
    )


fieldnames = list(csv_output[0].keys())

with open(
    OUT_CSV,
    "w",
    encoding="utf-8",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(csv_output)


print()
print("=" * 65)
print("TOP-5 HISTÓRICO EXPORTADO")
print("=" * 65)

print(f"Preguntas:        {len(json_output)}")
print(f"Chunks Top-5:     {len(csv_output)}")
print(f"Chunks faltantes: {len(missing_chunks)}")

print()
print("JSON:", OUT_JSON)
print("CSV: ", OUT_CSV)
