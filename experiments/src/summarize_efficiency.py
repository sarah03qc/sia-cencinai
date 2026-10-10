"""Combine the three model efficiency runs into the final comparison table"""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "efficiency"

MODEL_KEYS = [
    "qwen2.5-32b",
    "llama3.3-70b",
    "deepseek-r1-distill-qwen-32b",
]

OUTPUT_CSV = RESULTS_DIR / "efficiency_summary.csv"
OUTPUT_JSON = RESULTS_DIR / "efficiency_summary.json"
OUTPUT_MD = RESULTS_DIR / "efficiency_summary.md"


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    summaries = []
    per_question_ids = {}

    for model_key in MODEL_KEYS:
        summary_path = RESULTS_DIR / f"{model_key}_efficiency_summary.json"
        rows_path = RESULTS_DIR / f"{model_key}_efficiency_30.json"

        if not summary_path.exists() or not rows_path.exists():
            raise FileNotFoundError(
                f"Faltan resultados de {model_key}. "
                f"Corra benchmark_efficiency.py para ese modelo"
            )

        summary = load_json(summary_path)
        rows = load_json(rows_path)

        if len(rows) != 30:
            raise ValueError(f"{model_key} tiene {len(rows)} filas y deberían ser 30")

        ids = [row["question_id"] for row in rows]
        per_question_ids[model_key] = ids
        summaries.append(summary)

    reference_ids = per_question_ids[MODEL_KEYS[0]]
    for model_key in MODEL_KEYS[1:]:
        if per_question_ids[model_key] != reference_ids:
            raise ValueError(f"La muestra de {model_key} no coincide con la referencia")

    columns = [
        "model_key",
        "model_name",
        "quantization",
        "gpu",
        "n_questions",
        "generation_seconds_median",
        "generation_seconds_mean",
        "tokens_per_second_median",
        "tokens_per_second_mean",
        "peak_vram_gb_max",
        "baseline_vram_gb_mean",
        "output_tokens_mean",
        "measured_generation_seconds_total",
        "wall_seconds_measured_loop",
        "model_load_seconds",
    ]

    with open(OUTPUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for item in summaries:
            writer.writerow({key: item.get(key) for key in columns})

    payload = {
        "n_models": 3,
        "n_questions_per_model": 30,
        "same_question_ids": True,
        "latency_scope": "fixed_historical_top5_context_generation",
        "retrieval_included": False,
        "models": summaries,
    }
    OUTPUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Efficiency benchmark summary",
        "",
        "The benchmark uses the same 30 questions and the archived historical Top-5 context for all models",
        "",
        "Retrieval latency is not included",
        "",
        "| Model | Median generation latency s | Median tokens/s | Peak VRAM GB | Avg output tokens | Model load s |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for item in summaries:
        lines.append(
            f"| {item['model_name']} | "
            f"{item['generation_seconds_median']:.3f} | "
            f"{item['tokens_per_second_median']:.2f} | "
            f"{item['peak_vram_gb_max']:.2f} | "
            f"{item['output_tokens_mean']:.1f} | "
            f"{item['model_load_seconds']:.1f} |"
        )

    lines.extend(
        [
            "",
            "VRAM is the maximum memory allocated by PyTorch during measured generation",
            "",
            "The comparison represents the deployed quantized configurations rather than intrinsic architecture efficiency",
        ]
    )

    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print(f"CSV: {OUTPUT_CSV}")
    print(f"JSON: {OUTPUT_JSON}")
    print(f"Markdown: {OUTPUT_MD}")


if __name__ == "__main__":
    main()
