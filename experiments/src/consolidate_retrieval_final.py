#!/usr/bin/env python3

import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = ROOT / "results" / "retrieval_audit"
BENCHMARK = ROOT / "data" / "benchmark_final_299.json"
SCREENING = AUDIT_DIR / "evidence_hit_v2_screening.csv"
SEMANTIC = AUDIT_DIR / "retrieval_semantic_review_95_final.csv"
OUT_CSV = AUDIT_DIR / "retrieval_final_299.csv"
OUT_JSON = AUDIT_DIR / "retrieval_final_summary.json"


def load_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    with open(BENCHMARK, encoding="utf-8") as f:
        benchmark = json.load(f)

    screening = {x["question_id"]: x for x in load_csv(SCREENING)}
    semantic = {x["question_id"]: x for x in load_csv(SEMANTIC)}

    if len(benchmark) != 299:
        raise ValueError(f"Expected 299 questions, found {len(benchmark)}")

    if len(semantic) != 95:
        raise ValueError(f"Expected 95 semantic review rows, found {len(semantic)}")

    rows = []

    for item in benchmark:
        qid = item["id"]
        auto = screening[qid]

        if auto["outcome"] == "HIGH_CONFIDENCE_HIT":
            final_hit = "YES"
            final_best_rank = auto.get("best_hit_rank", "")
            method = "automatic_high_confidence_screening"
        else:
            reviewed = semantic.get(qid)
            if reviewed is None:
                raise ValueError(f"Missing semantic review for {qid}")
            final_hit = reviewed["hit_at_5"]
            final_best_rank = reviewed.get("best_k", "")
            method = "strict_semantic_review"

        rows.append({
            "question_id": qid,
            "category": item["categoria"],
            "automatic_outcome": auto["outcome"],
            "automatic_best_hit_rank": auto.get("best_hit_rank", ""),
            "final_hit_at_5": final_hit,
            "final_best_rank_if_available": final_best_rank,
            "adjudication_method": method,
        })

    counts = Counter(x["final_hit_at_5"] for x in rows)

    if counts["YES"] != 282 or counts["NO"] != 17:
        raise ValueError(f"Unexpected final retrieval counts {dict(counts)}")

    with open(OUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    by_category = {}
    for category in ["sino", "corta", "abierta"]:
        subset = [x for x in rows if x["category"] == category]
        hits = sum(x["final_hit_at_5"] == "YES" for x in subset)
        by_category[category] = {
            "n": len(subset),
            "hits_at_5": hits,
            "misses_at_5": len(subset) - hits,
            "hit_rate_at_5": hits / len(subset),
        }

    summary = {
        "benchmark_size": len(rows),
        "final_hits_at_5": counts["YES"],
        "final_misses_at_5": counts["NO"],
        "final_hit_rate_at_5": counts["YES"] / len(rows),
        "by_category": by_category,
        "reported_metric": "Hit@5 only",
        "note": "Hit@1 and Hit@3 are not reported as final metrics because automatic and semantic rank adjudication used different methods",
    }

    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Final Hit@5 {counts['YES']}/{len(rows)} = {counts['YES'] / len(rows):.1%}")
    print(OUT_CSV)
    print(OUT_JSON)


if __name__ == "__main__":
    main()
