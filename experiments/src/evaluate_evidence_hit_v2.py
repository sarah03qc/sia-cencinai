import json
import csv
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT / "results" / "retrieval_audit"
    / "historical_top5_final_299.json"
)

OUT = (
    ROOT / "results" / "retrieval_audit"
    / "evidence_hit_v2_screening.csv"
)

REVIEW_OUT = (
    ROOT / "results" / "retrieval_audit"
    / "evidence_hit_v2_remaining_review.csv"
)


STOPWORDS = {
    "de","la","el","los","las","un","una","unos","unas",
    "y","o","en","del","al","por","para","con","sin",
    "que","se","es","son","como","su","sus","a"
}


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


def content_tokens(text):
    return {
        t for t in normalize(text).split()
        if len(t) >= 3 and t not in STOPWORDS
    }


def split_evidence(text):
    """
    Divide evidencias largas/multifuente en unidades más pequeñas.
    """
    if not text:
        return []

    # Separadores frecuentes en nuestras anotaciones
    parts = re.split(
        r'(?<=[.!?])\s+|(?:\s*/\s*)|(?:\s*;\s*)|\n+',
        str(text)
    )

    cleaned = []

    for p in parts:
        p = p.strip(' "\'')
        toks = content_tokens(p)

        # Evitar fragmentos demasiado pequeños/genéricos
        if len(toks) >= 4:
            cleaned.append(p)

    # fallback si no logró segmentar
    if not cleaned and len(content_tokens(text)) >= 4:
        cleaned = [text]

    return cleaned


def recall(reference, chunk):
    r = content_tokens(reference)
    c = content_tokens(chunk)

    if not r:
        return 0.0

    return len(r & c) / len(r)


def exact_contains(reference, chunk):
    r = normalize(reference)
    c = normalize(chunk)

    return bool(r) and r in c


with open(INPUT, encoding="utf-8") as f:
    data = json.load(f)


rows = []
review_rows = []

for item in data:

    evidence = item["authoritative_evidence"]

    gold_text = evidence.get(
        "supporting_excerpt", ""
    )

    segments = split_evidence(gold_text)

    chunk_results = []

    for chunk in item["top5_historical"]:

        best_recall = 0.0
        exact = False
        best_segment = ""

        for segment in segments:

            r = recall(
                segment,
                chunk["text"]
            )

            if r > best_recall:
                best_recall = r
                best_segment = segment

            if exact_contains(
                segment,
                chunk["text"]
            ):
                exact = True
                best_recall = 1.0
                best_segment = segment
                break

        # Criterio de alta confianza.
        # No depende del nombre del documento.
        if exact:
            status = "HIGH_CONFIDENCE_HIT"

        elif best_recall >= 0.90:
            status = "HIGH_CONFIDENCE_HIT"

        elif best_recall >= 0.65:
            status = "REVIEW"

        else:
            status = "NO_MATCH"

        chunk_results.append({
            "rank": chunk["rank"],
            "status": status,
            "best_segment_recall": best_recall,
            "best_segment": best_segment,
            "source_document": chunk["source_document"],
            "chunk_index": chunk["chunk_index"],
            "text": chunk["text"],
        })


    hit_ranks = [
        x["rank"]
        for x in chunk_results
        if x["status"] == "HIGH_CONFIDENCE_HIT"
    ]

    review_ranks = [
        x["rank"]
        for x in chunk_results
        if x["status"] == "REVIEW"
    ]

    best_hit_rank = (
        min(hit_ranks)
        if hit_ranks
        else None
    )

    hit1 = int(
        any(r <= 1 for r in hit_ranks)
    )

    hit3 = int(
        any(r <= 3 for r in hit_ranks)
    )

    hit5 = int(
        any(r <= 5 for r in hit_ranks)
    )

    if hit_ranks:
        outcome = "HIGH_CONFIDENCE_HIT"
    elif review_ranks:
        outcome = "REVIEW"
    else:
        outcome = "NO_MATCH"


    rows.append({
        "question_id":
            item["id"],

        "categoria":
            item["categoria"],

        "pregunta":
            item["pregunta"],

        "ground_truth":
            item["ground_truth"],

        "high_conf_hit_at_1":
            hit1,

        "high_conf_hit_at_3":
            hit3,

        "high_conf_hit_at_5":
            hit5,

        "best_hit_rank":
            best_hit_rank or "",

        "outcome":
            outcome,

        "review_candidate_ranks":
            ",".join(
                str(r)
                for r in review_ranks
            )
    })


    if outcome != "HIGH_CONFIDENCE_HIT":

        # Mejor candidato entre los 5
        best = max(
            chunk_results,
            key=lambda x:
                x["best_segment_recall"]
        )

        review_rows.append({
            "question_id":
                item["id"],

            "categoria":
                item["categoria"],

            "pregunta":
                item["pregunta"],

            "ground_truth":
                item["ground_truth"],

            "answer_from_corpus":
                evidence.get(
                    "answer_from_corpus",
                    ""
                ),

            "gold_evidence":
                gold_text,

            "automatic_outcome":
                outcome,

            "best_candidate_rank":
                best["rank"],

            "best_candidate_recall":
                round(
                    best["best_segment_recall"],
                    4
                ),

            "best_candidate_document":
                best["source_document"],

            "best_evidence_segment":
                best["best_segment"],

            "best_candidate_chunk":
                best["text"],

            # para adjudicación final
            "final_hit":
                "",

            "final_best_rank":
                "",

            "adjudication_notes":
                ""
        })


with open(
    OUT,
    "w",
    encoding="utf-8",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(rows[0].keys())
    )

    writer.writeheader()
    writer.writerows(rows)


if review_rows:

    with open(
        REVIEW_OUT,
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                review_rows[0].keys()
            )
        )

        writer.writeheader()
        writer.writerows(
            review_rows
        )


n = len(rows)

h1 = sum(
    x["high_conf_hit_at_1"]
    for x in rows
)

h3 = sum(
    x["high_conf_hit_at_3"]
    for x in rows
)

h5 = sum(
    x["high_conf_hit_at_5"]
    for x in rows
)

remaining = len(review_rows)


print()
print("=" * 65)
print("EVIDENCE HIT V2 — HIGH-CONFIDENCE SCREENING")
print("=" * 65)

print(f"Preguntas: {n}")

print()
print(
    f"High-confidence Hit@1: "
    f"{h1}/{n} ({h1/n:.1%})"
)

print(
    f"High-confidence Hit@3: "
    f"{h3}/{n} ({h3/n:.1%})"
)

print(
    f"High-confidence Hit@5: "
    f"{h5}/{n} ({h5/n:.1%})"
)

print()
print(
    f"Casos restantes para adjudicar: "
    f"{remaining}"
)

print()
print("Screening:", OUT)
print("Review:   ", REVIEW_OUT)
