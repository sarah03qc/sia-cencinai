#!/usr/bin/env python3

import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def check(condition, message):
    if not condition:
        raise AssertionError(message)
    print(f"OK  {message}")


def main():
    benchmark = read_json(ROOT / "data" / "benchmark_final_299.json")
    evidence = read_json(ROOT / "data" / "evidence_annotations_final_299.json")
    counts = Counter(x["categoria"] for x in benchmark)

    check(len(benchmark) == 299, "benchmark has 299 questions")
    check(counts == {"sino": 99, "corta": 100, "abierta": 100}, "benchmark category counts are correct")
    check(len(evidence) == 299, "evidence file has 299 rows")
    check(all(x["supported_by_corpus"] == "YES" for x in evidence), "all frozen benchmark items are marked supported by corpus")

    raw_dir = ROOT / "results" / "raw"
    raw_files = [
        raw_dir / "qwen2.5-32b.json",
        raw_dir / "llama3.3-70b.json",
        raw_dir / "deepseek-r1-distill-qwen-32b.json",
    ]
    raw = [read_json(x) for x in raw_files]
    check(all(len(x) == 300 for x in raw), "each raw model file has 300 original generations")

    retrieval = read_csv(ROOT / "results" / "retrieval_audit" / "retrieval_final_299.csv")
    retrieval_counts = Counter(x["final_hit_at_5"] for x in retrieval)
    check(len(retrieval) == 299, "final retrieval file has 299 questions")
    check(retrieval_counts["YES"] == 282 and retrieval_counts["NO"] == 17, "final retrieval result is 282 Hit@5 and 17 Miss@5")

    metrics = read_csv(ROOT / "results" / "revision_metrics" / "automatic_metrics_per_question.csv")
    check(len(metrics) == 897, "automatic metric file has 897 model-question rows")

    bootstrap = read_csv(ROOT / "results" / "revision_metrics" / "bootstrap" / "bootstrap_pairwise_differences.csv")
    check(len(bootstrap) == 15, "bootstrap file has 15 pairwise metric comparisons")

    judge_input = read_json(ROOT / "results" / "llm_judge" / "llm_judge_200_fixed_ids.json")
    judge_eval = read_json(ROOT / "results" / "llm_judge" / "llm_judge_evaluaciones_600.json")
    check(len(judge_input) == 200, "LLM judge input has 200 free-form questions")
    check(len(judge_eval) == 600, "LLM judge output has 600 response evaluations")

    final_sample = read_csv(ROOT / "results" / "qualitative_analysis" / "qualitative_final_60_responses.csv")
    manual = read_csv(ROOT / "results" / "qualitative_analysis" / "manual_verdicts_60.csv")
    manual_counts = Counter(x["manual_grade"] for x in manual)
    check(len(final_sample) == 60, "final qualitative sample has 60 responses")
    check(len(manual) == 60, "manual verdict file has 60 ratings")
    check(manual_counts == {"B": 47, "P": 5, "M": 8}, "manual verdict counts are 47 B, 5 P, 8 M")

    crosswalk = read_csv(ROOT / "results" / "qualitative_analysis" / "analisis_cualitativo_crosswalk_60.csv")
    check(len(crosswalk) == 60, "qualitative crosswalk has 60 rows")

    print("\nRevision artifacts validated")


if __name__ == "__main__":
    main()
