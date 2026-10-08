import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SUMMARY = (
    ROOT / "results" / "retrieval_audit"
    / "evidence_hit_summary_299.csv"
)

CHUNKS = (
    ROOT / "results" / "retrieval_audit"
    / "evidence_hit_chunk_details_1495.csv"
)

with open(SUMMARY, encoding="utf-8") as f:
    summary = {
        r["question_id"]: r
        for r in csv.DictReader(f)
    }

by_question = defaultdict(list)

with open(CHUNKS, encoding="utf-8") as f:
    for r in csv.DictReader(f):
        by_question[r["question_id"]].append(r)


# Casos donde:
# 1) no fue HIT
# 2) pero existe overlap >= 0.70
suspicious = []

for qid, s in summary.items():

    if s["retrieval_outcome"] == "HIT":
        continue

    chunks = by_question[qid]

    best = max(
        chunks,
        key=lambda x: float(
            x["evidence_token_recall"]
        )
    )

    recall = float(
        best["evidence_token_recall"]
    )

    if recall >= 0.70:
        suspicious.append(
            (
                recall,
                qid,
                s,
                best,
                chunks
            )
        )


print()
print("=" * 80)
print("CASOS SOSPECHOSOS: overlap >= 0.70 pero NO clasificados HIT")
print("=" * 80)

print(f"\nTotal: {len(suspicious)}")


for recall, qid, s, best, chunks in sorted(
    suspicious,
    reverse=True
):

    print("\n" + "=" * 80)
    print(f"ID: {qid}")
    print(f"Categoría: {s['categoria']}")
    print(f"Outcome actual: {s['retrieval_outcome']}")
    print(f"Pregunta: {s['pregunta']}")

    print("\nGOLD DOCUMENT:")
    print(s["gold_source_document"])

    print("\nGOLD PAGE:")
    print(s["gold_page_or_section"])

    print(
        f"\nMEJOR CHUNK: "
        f"rank={best['rank']} "
        f"recall={recall:.3f}"
    )

    print("Retrieved document:")
    print(best["retrieved_source_document"])

    print(
        "same_gold_document:",
        best["same_gold_document"]
    )

    print(
        "match_status:",
        best["match_status"]
    )

    print("\nTodos los Top-5:")

    for c in chunks:
        print(
            f"  rank {c['rank']} | "
            f"recall={float(c['evidence_token_recall']):.3f} | "
            f"same_doc={c['same_gold_document']} | "
            f"{c['retrieved_source_document']}"
        )
