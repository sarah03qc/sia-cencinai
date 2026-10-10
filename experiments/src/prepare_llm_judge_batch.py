#!/usr/bin/env python3

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = ROOT / "results" / "raw"
OUT = ROOT / "results" / "llm_judge"

BENCHMARK = DATA / "benchmark_final_299.json"
EVIDENCE = DATA / "evidence_annotations_final_299.json"

MODEL_FILES = {
    "A": RAW / "qwen2.5-32b.json",
    "B": RAW / "llama3.3-70b.json",
    "C": RAW / "deepseek-r1-distill-qwen-32b.json",
}


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    benchmark = load_json(BENCHMARK)
    evidence = {x["id"]: x for x in load_json(EVIDENCE)}
    raw = {
        key: {x["pregunta_id"]: x for x in load_json(path)}
        for key, path in MODEL_FILES.items()
    }

    rows = []

    for item in benchmark:
        if item["categoria"] not in {"corta", "abierta"}:
            continue

        qid = item["id"]
        ev = evidence[qid]
        authoritative = (
            f"Source document: {ev['source_document']}\n"
            f"Page/section: {ev['page_or_section']}\n"
            f"Supporting excerpt: {ev['supporting_excerpt']}"
        )

        rows.append({
            "question_id": qid,
            "category": item["categoria"],
            "question": item["pregunta"],
            "ground_truth": item["ground_truth"],
            "authoritative_evidence": authoritative,
            "response_A": raw["A"][qid]["respuesta_generada"],
            "response_B": raw["B"][qid]["respuesta_generada"],
            "response_C": raw["C"][qid]["respuesta_generada"],
        })

    if len(rows) != 200:
        raise ValueError(f"Expected 200 free-form questions, found {len(rows)}")

    json_path = OUT / "llm_judge_200_fixed_ids.json"
    csv_path = OUT / "llm_judge_200_fixed_ids.csv"

    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Questions {len(rows)}")
    print(f"Responses prepared {len(rows) * 3}")
    print(json_path)
    print(csv_path)


if __name__ == "__main__":
    main()
