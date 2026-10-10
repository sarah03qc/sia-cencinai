#!/usr/bin/env python3

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "qualitative_analysis"
BENCHMARK = ROOT / "data" / "benchmark_final_299.json"
METADATA = OUT / "qualitative_sample_metadata.json"

SEED = 20261009
REPLACED = {
    "corta": "corta_040",
    "abierta": "abierta_073",
}


def main():
    benchmark = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))

    replacements = {}

    for category in ["corta", "abierta"]:
        selected = set(metadata["selected_ids"][category])
        candidates = sorted(
            x["id"]
            for x in benchmark
            if x["categoria"] == category and x["id"] not in selected
        )
        rng = random.Random(SEED)
        replacements[category] = rng.sample(candidates, 1)[0]

    expected = {"corta": "corta_096", "abierta": "abierta_095"}
    if replacements != expected:
        raise ValueError(f"Unexpected replacement selection {replacements}")

    output = {
        "seed": SEED,
        "selection_method": "independent random draw per category from questions outside the original sample",
        "replaced": REPLACED,
        "replacements": replacements,
        "reason": "original sampled item did not provide enough evidence for a fair manual adjudication",
    }

    path = OUT / "qualitative_replacements_metadata.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print(replacements)
    print(path)


if __name__ == "__main__":
    main()
