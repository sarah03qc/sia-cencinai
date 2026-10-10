#!/usr/bin/env python3

import csv
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261008
N_PER_CATEGORY = 10

BENCHMARK = ROOT / "data" / "benchmark_final_299.json"
EVIDENCE = ROOT / "data" / "evidence_annotations_final_299.json"
METRICS = ROOT / "results" / "revision_metrics" / "automatic_metrics_per_question.csv"
OUT = ROOT / "results" / "qualitative_analysis"

MODEL_FILES = {
    "Qwen2.5-32B": ROOT / "results" / "raw" / "qwen2.5-32b.json",
    "Llama3.3-70B": ROOT / "results" / "raw" / "llama3.3-70b.json",
    "DeepSeek-R1-Distill-Qwen-32B": ROOT / "results" / "raw" / "deepseek-r1-distill-qwen-32b.json",
}


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_metrics(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return {(x["question_id"], x["model"]): x for x in csv.DictReader(f)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    benchmark = load_json(BENCHMARK)
    evidence = {x["id"]: x for x in load_json(EVIDENCE)}
    metrics = load_metrics(METRICS)
    by_id = {x["id"]: x for x in benchmark}
    answers = {
        model: {x["pregunta_id"]: x for x in load_json(path)}
        for model, path in MODEL_FILES.items()
    }

    rng = random.Random(SEED)
    short_ids = sorted(x["id"] for x in benchmark if x["categoria"] == "corta")
    open_ids = sorted(x["id"] for x in benchmark if x["categoria"] == "abierta")

    selected_short = rng.sample(short_ids, N_PER_CATEGORY)
    selected_open = rng.sample(open_ids, N_PER_CATEGORY)
    selected = [("corta", x) for x in selected_short] + [("abierta", x) for x in selected_open]

    question_rows = []
    response_rows = []

    for category, qid in selected:
        item = by_id[qid]
        ev = evidence[qid]
        base = {
            "question_id": qid,
            "category": category,
            "question": item["pregunta"],
            "ground_truth": item["ground_truth"],
            "source_document": ev.get("source_document", ""),
            "page_or_section": ev.get("page_or_section", ""),
            "authoritative_evidence": ev.get("supporting_excerpt", ""),
        }
        question_rows.append(base)

        for model in MODEL_FILES:
            score = metrics.get((qid, model), {})
            response_rows.append({
                **base,
                "model": model,
                "generated_response": answers[model][qid].get("respuesta_generada", ""),
                "accuracy": score.get("accuracy", ""),
                "token_f1": score.get("token_f1", ""),
                "bertscore_f1": score.get("bertscore_f1", ""),
                "rouge_l": score.get("rouge_l", ""),
                "meteor": score.get("meteor", ""),
            })

    q_path = OUT / "qualitative_sample_20_questions.csv"
    r_path = OUT / "qualitative_sample_60_responses.csv"
    m_path = OUT / "qualitative_sample_metadata.json"

    with open(q_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(question_rows[0].keys()))
        writer.writeheader()
        writer.writerows(question_rows)

    with open(r_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(response_rows[0].keys()))
        writer.writeheader()
        writer.writerows(response_rows)

    metadata = {
        "selection_method": "stratified random sample",
        "seed": SEED,
        "categories": {"corta": 10, "abierta": 10},
        "selected_ids": {"corta": selected_short, "abierta": selected_open},
        "n_questions": 20,
        "n_model_responses": 60,
        "models": list(MODEL_FILES),
    }
    m_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print("CORTA")
    for qid in selected_short:
        print(qid)
    print("ABIERTA")
    for qid in selected_open:
        print(qid)


if __name__ == "__main__":
    main()
