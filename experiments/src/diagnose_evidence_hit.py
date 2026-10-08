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


review_bins = {
    ">=0.70": 0,
    "0.60-0.69": 0,
    "0.50-0.59": 0,
    "0.35-0.49": 0,
    "<0.35": 0,
}

miss_same_doc = 0
miss_wrong_doc = 0

category_stats = defaultdict(
    lambda: {"HIT": 0, "REVIEW": 0, "MISS": 0}
)

examples_review = []
examples_miss_same_doc = []
examples_miss_wrong_doc = []


for qid, s in summary.items():

    outcome = s["retrieval_outcome"]
    categoria = s["categoria"]

    category_stats[categoria][outcome] += 1

    chunks = by_question[qid]

    recalls = [
        float(x["evidence_token_recall"])
        for x in chunks
    ]

    max_recall = max(recalls)

    any_same_doc = any(
        str(x["same_gold_document"]).lower()
        == "true"
        for x in chunks
    )

    if outcome == "REVIEW":

        if max_recall >= 0.70:
            review_bins[">=0.70"] += 1
        elif max_recall >= 0.60:
            review_bins["0.60-0.69"] += 1
        elif max_recall >= 0.50:
            review_bins["0.50-0.59"] += 1
        elif max_recall >= 0.35:
            review_bins["0.35-0.49"] += 1
        else:
            review_bins["<0.35"] += 1

        examples_review.append(
            (max_recall, qid, s["pregunta"])
        )

    elif outcome == "MISS":

        if any_same_doc:
            miss_same_doc += 1
            examples_miss_same_doc.append(
                (max_recall, qid, s["pregunta"])
            )
        else:
            miss_wrong_doc += 1
            examples_miss_wrong_doc.append(
                (max_recall, qid, s["pregunta"])
            )


print()
print("=" * 65)
print("DIAGNÓSTICO EVIDENCE HIT")
print("=" * 65)

print("\nREVIEW por máximo token recall:")
for k, v in review_bins.items():
    print(f"  {k}: {v}")

print()
print("MISS:")
print(
    f"  Documento gold aparece en Top-5: {miss_same_doc}"
)
print(
    f"  Documento gold NO aparece en Top-5: {miss_wrong_doc}"
)

print()
print("Por categoría:")

for cat in sorted(category_stats):
    x = category_stats[cat]
    print(
        f"  {cat}: "
        f"HIT={x['HIT']} "
        f"REVIEW={x['REVIEW']} "
        f"MISS={x['MISS']}"
    )

print()
print("Top 10 REVIEW con mayor overlap:")

for recall, qid, question in sorted(
    examples_review,
    reverse=True
)[:10]:
    print(
        f"  {qid}: {recall:.3f} | {question[:90]}"
    )

print()
print("MISS pero documento correcto sí fue recuperado:")

for recall, qid, question in sorted(
    examples_miss_same_doc,
    reverse=True
)[:10]:
    print(
        f"  {qid}: {recall:.3f} | {question[:90]}"
    )

print()
print("MISS con documento gold ausente del Top-5:")

for recall, qid, question in sorted(
    examples_miss_wrong_doc,
    reverse=True
)[:10]:
    print(
        f"  {qid}: {recall:.3f} | {question[:90]}"
    )
