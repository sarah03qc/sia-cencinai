"""Build the fixed 30-question sample used for the efficiency benchmark

The draw is stratified by category with 10 questions from each group
The same sample is used by all three model configurations
"""

import csv
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = ROOT / "data" / "benchmark_final_299.json"
OUTPUT_JSON = ROOT / "data" / "efficiency_sample_30.json"
OUTPUT_CSV = ROOT / "data" / "efficiency_sample_30.csv"

SEED = 20261010
N_PER_CATEGORY = 10
CATEGORIES = ["sino", "corta", "abierta"]


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    benchmark = load_json(BENCHMARK_PATH)
    by_category = {category: [] for category in CATEGORIES}

    for item in benchmark:
        category = item["categoria"]
        if category in by_category:
            by_category[category].append(item)

    for category in CATEGORIES:
        if len(by_category[category]) < N_PER_CATEGORY:
            raise ValueError(
                f"No hay suficientes preguntas en {category}: "
                f"{len(by_category[category])} disponibles"
            )

    rng = random.Random(SEED)
    selected = []

    for category in CATEGORIES:
        pool = sorted(by_category[category], key=lambda x: x["id"])
        picks = rng.sample(pool, N_PER_CATEGORY)
        picks = sorted(picks, key=lambda x: x["id"])

        for item in picks:
            selected.append(
                {
                    "id": item["id"],
                    "categoria": item["categoria"],
                    "pregunta": item["pregunta"],
                }
            )

    metadata = {
        "seed": SEED,
        "sampling": "stratified_random",
        "n_per_category": N_PER_CATEGORY,
        "n_total": len(selected),
        "categories": CATEGORIES,
        "questions": selected,
    }

    OUTPUT_JSON.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with open(OUTPUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "categoria", "pregunta"])
        writer.writeheader()
        writer.writerows(selected)

    print(f"Muestra creada con seed {SEED}")
    print(f"Preguntas: {len(selected)}")
    for category in CATEGORIES:
        ids = [x["id"] for x in selected if x["categoria"] == category]
        print(f"{category}: {', '.join(ids)}")
    print(f"JSON: {OUTPUT_JSON}")
    print(f"CSV:  {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
