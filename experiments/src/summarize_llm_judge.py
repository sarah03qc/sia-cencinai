#!/usr/bin/env python3

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "results" / "llm_judge"
INPUT = DIR / "llm_judge_evaluaciones_600.json"
OUT_CSV = DIR / "llm_judge_summary.csv"
OUT_JSON = DIR / "llm_judge_summary.json"

MODEL_KEY = {
    "A": "Qwen2.5-32B",
    "B": "Llama3.3-70B",
    "C": "DeepSeek-R1-Distill-Qwen-32B",
}

SCORE_FIELDS = [
    "factual_correctness",
    "completeness",
    "faithfulness_to_evidence",
    "unsupported_information",
]


def main():
    data = json.loads(INPUT.read_text(encoding="utf-8"))

    if len(data) != 600:
        raise ValueError(f"Expected 600 evaluations, found {len(data)}")

    seen = set()
    grouped = defaultdict(list)

    for row in data:
        key = (row["question_id"], row["response_id"])
        if key in seen:
            raise ValueError(f"Duplicate evaluation {key}")
        seen.add(key)
        grouped[MODEL_KEY[row["response_id"]]].append(row)

    summary_rows = []
    summary_json = {}

    for model, rows in grouped.items():
        verdicts = Counter(x["overall_verdict"] for x in rows)
        averages = {
            field: sum(float(x[field]) for x in rows) / len(rows)
            for field in SCORE_FIELDS
        }

        category_pass = {}
        for category in ["corta", "abierta"]:
            subset = [x for x in rows if x["category"] == category]
            passed = sum(x["overall_verdict"] == "PASS" for x in subset)
            category_pass[category] = {
                "pass": passed,
                "n": len(subset),
                "pass_rate": passed / len(subset),
            }

        record = {
            "model": model,
            "n": len(rows),
            "pass": verdicts["PASS"],
            "partial": verdicts["PARTIAL"],
            "fail": verdicts["FAIL"],
            "pass_rate": verdicts["PASS"] / len(rows),
            **averages,
            "short_pass_rate": category_pass["corta"]["pass_rate"],
            "open_pass_rate": category_pass["abierta"]["pass_rate"],
        }
        summary_rows.append(record)
        summary_json[model] = {
            "verdict_counts": dict(verdicts),
            "averages": averages,
            "by_category": category_pass,
        }

    with open(OUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    OUT_JSON.write_text(json.dumps(summary_json, ensure_ascii=False, indent=2), encoding="utf-8")

    for row in summary_rows:
        print(row["model"], row["pass"], row["partial"], row["fail"])


if __name__ == "__main__":
    main()
