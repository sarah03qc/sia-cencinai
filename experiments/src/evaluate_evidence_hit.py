import csv
import json
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT
    / "results"
    / "retrieval_audit"
    / "historical_top5_final_299.json"
)

OUT_DIR = ROOT / "results" / "retrieval_audit"

OUT_QUESTIONS = OUT_DIR / "evidence_hit_summary_299.csv"
OUT_CHUNKS = OUT_DIR / "evidence_hit_chunk_details_1495.csv"
OUT_REVIEW = OUT_DIR / "evidence_hit_manual_review.csv"
OUT_JSON = OUT_DIR / "evidence_hit_results_299.json"


def normalize(text):
    if not text:
        return ""

    text = str(text).lower()

    text = "".join(
        c for c in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(c)
    )

    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def tokens(text):
    # eliminar tokens muy cortos para reducir ruido
    return [
        x
        for x in normalize(text).split()
        if len(x) >= 3
    ]


def source_documents(value):
    """
    La anotación puede contener uno o varios PDFs separados
    por ; u otras combinaciones.
    """
    if not value:
        return []

    return [
        x.strip()
        for x in re.split(r"[;\n]+", str(value))
        if x.strip()
    ]


def document_match(gold_docs, retrieved_doc):
    retrieved_norm = normalize(retrieved_doc)

    for doc in gold_docs:
        if normalize(doc) == retrieved_norm:
            return True

    return False


def token_recall(evidence, chunk):
    """
    Qué proporción de los tokens únicos de la evidencia gold
    están presentes en el chunk recuperado.
    """
    ev = set(tokens(evidence))
    ch = set(tokens(chunk))

    if not ev:
        return 0.0

    return len(ev & ch) / len(ev)


def token_precision(evidence, chunk):
    ev = set(tokens(evidence))
    ch = set(tokens(chunk))

    if not ch:
        return 0.0

    return len(ev & ch) / len(ch)


def exact_normalized_match(evidence, chunk):
    ev = normalize(evidence)
    ch = normalize(chunk)

    if not ev or not ch:
        return False

    return ev in ch


def classify_match(
    evidence,
    chunk,
    same_document
):
    """
    Clasificación conservadora.

    STRONG:
      evidencia prácticamente contenida en el chunk.

    REVIEW:
      mismo documento y solapamiento sustancial, pero no
      suficiente para afirmar automáticamente el hit.

    NO:
      evidencia no localizada.
    """

    exact = exact_normalized_match(
        evidence,
        chunk
    )

    recall = token_recall(
        evidence,
        chunk
    )

    precision = token_precision(
        evidence,
        chunk
    )

    if same_document and exact:
        return "STRONG", recall, precision, "exact_normalized"

    # Alta cobertura de la evidencia dentro del chunk.
    if same_document and recall >= 0.80:
        return "STRONG", recall, precision, "token_recall>=0.80"

    # Casos que necesitan inspección.
    if same_document and recall >= 0.35:
        return "REVIEW", recall, precision, "same_doc_partial_overlap"

    return "NO", recall, precision, "no_match"


with open(INPUT, encoding="utf-8") as f:
    data = json.load(f)


question_rows = []
chunk_rows = []
review_rows = []
json_results = []


for item in data:

    qid = item["id"]
    ev = item["authoritative_evidence"]

    evidence_text = ev.get(
        "supporting_excerpt",
        ""
    )

    gold_docs = source_documents(
        ev.get("source_document", "")
    )

    evaluated_chunks = []

    for chunk in item["top5_historical"]:

        same_doc = document_match(
            gold_docs,
            chunk["source_document"]
        )

        status, recall, precision, method = classify_match(
            evidence_text,
            chunk["text"],
            same_doc
        )

        result = {
            "rank": chunk["rank"],
            "global_chunk_id":
                chunk["global_chunk_id"],
            "source_document":
                chunk["source_document"],
            "chunk_index":
                chunk["chunk_index"],
            "same_gold_document":
                same_doc,
            "match_status":
                status,
            "match_method":
                method,
            "evidence_token_recall":
                recall,
            "evidence_token_precision":
                precision,
            "text":
                chunk["text"],
        }

        evaluated_chunks.append(result)

        chunk_rows.append({
            "question_id":
                qid,
            "categoria":
                item["categoria"],
            "pregunta":
                item["pregunta"],
            "ground_truth":
                item["ground_truth"],

            "gold_source_document":
                ev.get("source_document", ""),
            "gold_page_or_section":
                ev.get("page_or_section", ""),
            "gold_supporting_excerpt":
                evidence_text,

            "rank":
                chunk["rank"],
            "retrieved_source_document":
                chunk["source_document"],
            "retrieved_chunk_index":
                chunk["chunk_index"],

            "same_gold_document":
                same_doc,
            "match_status":
                status,
            "match_method":
                method,
            "evidence_token_recall":
                round(recall, 4),
            "retrieved_text":
                chunk["text"],
        })


    # -----------------------------------------------------
    # Hits automáticos SOLO con STRONG
    # REVIEW queda pendiente hasta inspección
    # -----------------------------------------------------

    strong_ranks = [
        x["rank"]
        for x in evaluated_chunks
        if x["match_status"] == "STRONG"
    ]

    review_ranks = [
        x["rank"]
        for x in evaluated_chunks
        if x["match_status"] == "REVIEW"
    ]

    best_rank = (
        min(strong_ranks)
        if strong_ranks
        else None
    )

    hit1 = int(
        any(r <= 1 for r in strong_ranks)
    )

    hit3 = int(
        any(r <= 3 for r in strong_ranks)
    )

    hit5 = int(
        any(r <= 5 for r in strong_ranks)
    )

    needs_review = (
        not strong_ranks
        and bool(review_ranks)
    )

    outcome = (
        "HIT"
        if strong_ranks
        else (
            "REVIEW"
            if review_ranks
            else "MISS"
        )
    )


    question_rows.append({
        "question_id":
            qid,
        "categoria":
            item["categoria"],
        "pregunta":
            item["pregunta"],
        "ground_truth":
            item["ground_truth"],

        "gold_source_document":
            ev.get("source_document", ""),
        "gold_page_or_section":
            ev.get("page_or_section", ""),

        "auto_hit_at_1":
            hit1,
        "auto_hit_at_3":
            hit3,
        "auto_hit_at_5":
            hit5,

        "auto_best_evidence_rank":
            best_rank if best_rank else "",

        "retrieval_outcome":
            outcome,

        "manual_review_needed":
            needs_review,

        "review_candidate_ranks":
            ",".join(map(str, review_ranks))
    })


    if outcome != "HIT":

        # guardar los 5 chunks para poder juzgar el caso
        review_rows.append({
            "question_id":
                qid,
            "categoria":
                item["categoria"],
            "pregunta":
                item["pregunta"],
            "ground_truth":
                item["ground_truth"],

            "gold_source_document":
                ev.get("source_document", ""),
            "gold_page_or_section":
                ev.get("page_or_section", ""),
            "gold_supporting_excerpt":
                evidence_text,

            "automatic_outcome":
                outcome,

            "rank1_status":
                evaluated_chunks[0]["match_status"],
            "rank1_recall":
                round(
                    evaluated_chunks[0][
                        "evidence_token_recall"
                    ],
                    4
                ),

            "rank2_status":
                evaluated_chunks[1]["match_status"],
            "rank2_recall":
                round(
                    evaluated_chunks[1][
                        "evidence_token_recall"
                    ],
                    4
                ),

            "rank3_status":
                evaluated_chunks[2]["match_status"],
            "rank3_recall":
                round(
                    evaluated_chunks[2][
                        "evidence_token_recall"
                    ],
                    4
                ),

            "rank4_status":
                evaluated_chunks[3]["match_status"],
            "rank4_recall":
                round(
                    evaluated_chunks[3][
                        "evidence_token_recall"
                    ],
                    4
                ),

            "rank5_status":
                evaluated_chunks[4]["match_status"],
            "rank5_recall":
                round(
                    evaluated_chunks[4][
                        "evidence_token_recall"
                    ],
                    4
                ),

            # columnas para decisión final manual
            "manual_hit":
                "",
            "manual_best_rank":
                "",
            "manual_notes":
                "",
        })


    json_results.append({
        "id":
            qid,
        "categoria":
            item["categoria"],
        "pregunta":
            item["pregunta"],
        "ground_truth":
            item["ground_truth"],
        "authoritative_evidence":
            ev,
        "retrieval_outcome":
            outcome,
        "auto_hit_at_1":
            hit1,
        "auto_hit_at_3":
            hit3,
        "auto_hit_at_5":
            hit5,
        "auto_best_evidence_rank":
            best_rank,
        "chunks":
            evaluated_chunks,
    })


# ---------------------------------------------------------
# GUARDAR
# ---------------------------------------------------------

with open(
    OUT_QUESTIONS,
    "w",
    encoding="utf-8",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            question_rows[0].keys()
        )
    )

    writer.writeheader()
    writer.writerows(question_rows)


with open(
    OUT_CHUNKS,
    "w",
    encoding="utf-8",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            chunk_rows[0].keys()
        )
    )

    writer.writeheader()
    writer.writerows(chunk_rows)


with open(
    OUT_REVIEW,
    "w",
    encoding="utf-8",
    newline=""
) as f:

    if review_rows:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                review_rows[0].keys()
            )
        )

        writer.writeheader()
        writer.writerows(review_rows)


with open(
    OUT_JSON,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        json_results,
        f,
        ensure_ascii=False,
        indent=2
    )


# ---------------------------------------------------------
# RESUMEN
# ---------------------------------------------------------

n = len(question_rows)

hit1 = sum(
    x["auto_hit_at_1"]
    for x in question_rows
)

hit3 = sum(
    x["auto_hit_at_3"]
    for x in question_rows
)

hit5 = sum(
    x["auto_hit_at_5"]
    for x in question_rows
)

hits = sum(
    x["retrieval_outcome"] == "HIT"
    for x in question_rows
)

reviews = sum(
    x["retrieval_outcome"] == "REVIEW"
    for x in question_rows
)

misses = sum(
    x["retrieval_outcome"] == "MISS"
    for x in question_rows
)


print()
print("=" * 65)
print("EVIDENCE HIT — AUTOMATIC SCREENING")
print("=" * 65)

print(f"Preguntas: {n}")

print()
print(
    f"Auto Hit@1: {hit1}/{n} "
    f"({hit1/n:.1%})"
)

print(
    f"Auto Hit@3: {hit3}/{n} "
    f"({hit3/n:.1%})"
)

print(
    f"Auto Hit@5: {hit5}/{n} "
    f"({hit5/n:.1%})"
)

print()
print(f"HIT fuerte:        {hits}")
print(f"Revisión manual:   {reviews}")
print(f"MISS automático:   {misses}")

print()
print("IMPORTANTE:")
print(
    "Estos son resultados preliminares automáticos. "
    "Los casos REVIEW/MISS deben verificarse antes "
    "de reportar Hit@k final."
)

print()
print("Summary:", OUT_QUESTIONS)
print("Chunks: ", OUT_CHUNKS)
print("Review: ", OUT_REVIEW)
print("JSON:   ", OUT_JSON)
