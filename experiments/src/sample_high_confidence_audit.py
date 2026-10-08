import csv
import random
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT / "results" / "retrieval_audit"
    / "evidence_hit_v2_screening.csv"
)

OUTPUT = (
    ROOT / "results" / "retrieval_audit"
    / "high_confidence_random_sample_30.csv"
)

SEED = 20261006
N_PER_CATEGORY = 10

random.seed(SEED)

by_cat = defaultdict(list)

with open(INPUT, encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

for r in rows:
    if r["outcome"] == "HIGH_CONFIDENCE_HIT":
        by_cat[r["categoria"]].append(r)

sample = []

for cat in ["sino", "corta", "abierta"]:
    pool = by_cat[cat]

    if len(pool) < N_PER_CATEGORY:
        raise ValueError(
            f"No hay suficientes casos en {cat}: {len(pool)}"
        )

    chosen = random.sample(pool, N_PER_CATEGORY)
    sample.extend(chosen)

with open(
    OUTPUT,
    "w",
    encoding="utf-8",
    newline=""
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "question_id",
            "categoria",
            "pregunta",
            "ground_truth",
            "best_hit_rank"
        ]
    )

    writer.writeheader()

    for r in sample:
        writer.writerow({
            "question_id": r["question_id"],
            "categoria": r["categoria"],
            "pregunta": r["pregunta"],
            "ground_truth": r["ground_truth"],
            "best_hit_rank": r["best_hit_rank"]
        })

print("=" * 60)
print("MUESTRA ALEATORIA HIGH-CONFIDENCE")
print("=" * 60)
print(f"Seed: {SEED}")
print(f"Total: {len(sample)}")

for cat in ["sino", "corta", "abierta"]:
    ids = [
        r["question_id"]
        for r in sample
        if r["categoria"] == cat
    ]

    print(f"\n{cat.upper()} ({len(ids)}):")
    for qid in ids:
        print(" ", qid)

print(f"\nArchivo: {OUTPUT}")
