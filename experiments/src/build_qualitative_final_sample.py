#!/usr/bin/env python3

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "qualitative_analysis"
BENCHMARK = ROOT / "data" / "benchmark_final_299.json"
EVIDENCE = ROOT / "data" / "evidence_annotations_final_299.json"
METRICS = ROOT / "results" / "revision_metrics" / "automatic_metrics_per_question.csv"

MODEL_FILES = {
    "Qwen2.5-32B": ROOT / "results" / "raw" / "qwen2.5-32b.json",
    "Llama3.3-70B": ROOT / "results" / "raw" / "llama3.3-70b.json",
    "DeepSeek-R1-Distill-Qwen-32B": ROOT / "results" / "raw" / "deepseek-r1-distill-qwen-32b.json",
}


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_metrics():
    with open(METRICS, encoding="utf-8-sig", newline="") as f:
        return {(x["question_id"], x["model"]): x for x in csv.DictReader(f)}


def main():
    sample = load_json(OUT / "qualitative_sample_metadata.json")
    replacements = load_json(OUT / "qualitative_replacements_metadata.json")
    benchmark = {x["id"]: x for x in load_json(BENCHMARK)}
    evidence = {x["id"]: x for x in load_json(EVIDENCE)}
    metrics = load_metrics()
    answers = {
        model: {x["pregunta_id"]: x for x in load_json(path)}
        for model, path in MODEL_FILES.items()
    }

    final_ids = sample["selected_ids"]["corta"] + sample["selected_ids"]["abierta"]
    final_ids = [
        replacements["replacements"]["corta"] if qid == replacements["replaced"]["corta"] else
        replacements["replacements"]["abierta"] if qid == replacements["replaced"]["abierta"] else
        qid
        for qid in final_ids
    ]

    rows = []
    question_rows = []

    for qid in final_ids:
        item = benchmark[qid]
        ev = evidence[qid]
        base = {
            "question_id": qid,
            "category": item["categoria"],
            "question": item["pregunta"],
            "ground_truth": item["ground_truth"],
            "source_document": ev.get("source_document", ""),
            "page_or_section": ev.get("page_or_section", ""),
            "authoritative_evidence": ev.get("supporting_excerpt", ""),
        }
        question_rows.append(base)

        for model in MODEL_FILES:
            score = metrics[(qid, model)]
            rows.append({
                **base,
                "model": model,
                "generated_response": answers[model][qid].get("respuesta_generada", ""),
                "accuracy": score.get("accuracy", ""),
                "token_f1": score.get("token_f1", ""),
                "bertscore_f1": score.get("bertscore_f1", ""),
                "rouge_l": score.get("rouge_l", ""),
                "meteor": score.get("meteor", ""),
            })

    q_path = OUT / "qualitative_final_20_questions.csv"
    r_path = OUT / "qualitative_final_60_responses.csv"
    m_path = OUT / "qualitative_final_metadata.json"

    with open(q_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(question_rows[0].keys()))
        writer.writeheader()
        writer.writerows(question_rows)

    with open(r_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "initial_seed": sample["seed"],
        "replacement_seed": replacements["seed"],
        "final_ids": final_ids,
        "n_questions": 20,
        "n_model_responses": 60,
    }
    m_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(q_path)
    print(r_path)
    print(m_path)


if __name__ == "__main__":
    main()
