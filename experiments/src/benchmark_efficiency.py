"""Measure generation efficiency on a fixed 30-question sample

The benchmark reuses the historical Top-5 chunks from the original RAG run
This keeps the model comparison focused on the deployed generation configs
Retrieval is not rerun and is therefore not included in the latency values

Usage from experiments

    python src/benchmark_efficiency.py qwen2.5-32b
    python src/benchmark_efficiency.py llama3.3-70b
    python src/benchmark_efficiency.py deepseek-r1-distill-qwen-32b

Use --dry-run to validate the sample and archived contexts without loading a model
"""

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR / "models"))

from config import MODEL_CONFIGS

SAMPLE_PATH = ROOT / "data" / "efficiency_sample_30.json"
TOP5_PATH = ROOT / "results" / "retrieval_audit" / "historical_top5_final_299.json"
OUTPUT_DIR = ROOT / "results" / "efficiency"

MODEL_KEYS = list(MODEL_CONFIGS)
GB = 1024 ** 3


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    context = "\n\n".join(
        f"[Fuente: {chunk['source_document']}]\n{chunk['text']}"
        for chunk in retrieved_chunks
    )
    return (
        "Responde la siguiente pregunta usando únicamente la información "
        "del contexto proporcionado. Si la respuesta no está en el "
        "contexto, indica que no tienes suficiente información.\n\n"
        f"Contexto:\n{context}\n\n"
        f"Pregunta: {question}\n"
        "Respuesta:"
    )


def prepare_inputs(torch, tokenizer, model, model_key: str, prompt: str, forced_think_close: str):
    config = MODEL_CONFIGS[model_key]
    messages = [{"role": "user", "content": prompt}]

    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    if config.skip_reasoning:
        close_ids = tokenizer(
            forced_think_close,
            add_special_tokens=False,
            return_tensors="pt",
        )
        encoded["input_ids"] = torch.cat(
            [encoded["input_ids"], close_ids["input_ids"]],
            dim=1,
        )
        encoded["attention_mask"] = torch.cat(
            [encoded["attention_mask"], torch.ones_like(close_ids["input_ids"])],
            dim=1,
        )

    encoded = {key: value.to(model.device) for key, value in encoded.items()}
    return encoded


def generate_measured(torch, model, tokenizer, model_key: str, prompt: str, forced_think_close: str):
    config = MODEL_CONFIGS[model_key]

    inference_start = time.perf_counter()
    encoded = prepare_inputs(torch, tokenizer, model, model_key, prompt, forced_think_close)
    input_tokens = int(encoded["input_ids"].shape[1])

    torch.cuda.synchronize()
    baseline_vram_gb = torch.cuda.memory_allocated() / GB
    torch.cuda.reset_peak_memory_stats()

    generation_start = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=config.max_new_tokens,
            do_sample=config.do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.synchronize()
    generation_seconds = time.perf_counter() - generation_start

    peak_vram_gb = torch.cuda.max_memory_allocated() / GB
    new_tokens = output_ids[0][input_tokens:]
    output_tokens = int(new_tokens.shape[0])
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    inference_seconds = time.perf_counter() - inference_start

    tokens_per_second = (
        output_tokens / generation_seconds if generation_seconds > 0 else None
    )

    return {
        "answer": answer,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "generation_seconds": generation_seconds,
        "inference_seconds": inference_seconds,
        "tokens_per_second": tokens_per_second,
        "baseline_vram_gb": baseline_vram_gb,
        "peak_vram_gb": peak_vram_gb,
    }


def detect_offload(model):
    device_map = getattr(model, "hf_device_map", None)
    if not device_map:
        return False, None

    values = {str(value) for value in device_map.values()}
    offload = any(value in {"cpu", "disk"} for value in values)
    return offload, device_map


def write_rows(csv_path: Path, json_path: Path, rows: list[dict]):
    json_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    fieldnames = list(rows[0].keys()) if rows else []
    if not fieldnames:
        return

    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_summary(model_key: str, rows: list[dict], gpu_name: str, torch_version: str, transformers_version: str):
    generation = [row["generation_seconds"] for row in rows]
    inference = [row["inference_seconds"] for row in rows]
    throughput = [row["tokens_per_second"] for row in rows]
    output_tokens = [row["output_tokens"] for row in rows]
    peak_vram = [row["peak_vram_gb"] for row in rows]
    baseline_vram = [row["baseline_vram_gb"] for row in rows]

    return {
        "model_key": model_key,
        "model_name": MODEL_CONFIGS[model_key].name,
        "quantization": MODEL_CONFIGS[model_key].quantization,
        "gpu": gpu_name,
        "n_questions": len(rows),
        "latency_scope": "fixed_historical_top5_context_generation",
        "retrieval_included": False,
        "generation_seconds_mean": statistics.mean(generation),
        "generation_seconds_median": statistics.median(generation),
        "inference_seconds_mean": statistics.mean(inference),
        "inference_seconds_median": statistics.median(inference),
        "tokens_per_second_mean": statistics.mean(throughput),
        "tokens_per_second_median": statistics.median(throughput),
        "output_tokens_mean": statistics.mean(output_tokens),
        "output_tokens_median": statistics.median(output_tokens),
        "baseline_vram_gb_mean": statistics.mean(baseline_vram),
        "peak_vram_gb_max": max(peak_vram),
        "measured_generation_seconds_total": sum(generation),
        "torch_version": torch_version,
        "transformers_version": transformers_version,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_key", choices=MODEL_KEYS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-offload", action="store_true")
    args = parser.parse_args()

    sample_data = load_json(SAMPLE_PATH)
    sample = sample_data["questions"]
    top5_data = load_json(TOP5_PATH)
    top5_by_id = {item["id"]: item for item in top5_data}

    if len(sample) != 30:
        raise ValueError(f"La muestra debe tener 30 preguntas y tiene {len(sample)}")

    missing = [item["id"] for item in sample if item["id"] not in top5_by_id]
    if missing:
        raise ValueError(f"Faltan contextos históricos para: {missing}")

    prompts = []
    for item in sample:
        historical = top5_by_id[item["id"]]
        chunks = historical["top5_historical"]
        if len(chunks) != 5:
            raise ValueError(f"{item['id']} no tiene exactamente 5 chunks históricos")
        prompts.append((item, chunks, build_prompt(item["pregunta"], chunks)))

    print(f"Modelo: {args.model_key}")
    print(f"Preguntas: {len(prompts)}")
    print(f"Seed de muestra: {sample_data['seed']}")
    print("Contexto: Top-5 histórico archivado")

    if args.dry_run:
        print("Dry run correcto")
        for category in ["sino", "corta", "abierta"]:
            ids = [item["id"] for item, _, _ in prompts if item["categoria"] == category]
            print(f"{category}: {', '.join(ids)}")
        return

    import torch
    import transformers
    from run_model import FORCED_THINK_CLOSE, load_model

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no está disponible")

    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"Se esperaba una sola GPU visible y hay {torch.cuda.device_count()}"
        )

    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print(f"Cargando {MODEL_CONFIGS[args.model_key].hf_repo}")

    load_start = time.perf_counter()
    model, tokenizer = load_model(args.model_key)
    model_load_seconds = time.perf_counter() - load_start
    offload_detected, device_map = detect_offload(model)

    if offload_detected and not args.allow_offload:
        raise RuntimeError(
            "Se detectó offload a CPU o disco. "
            "No se mide para evitar una comparación inconsistente. "
            "Use --allow-offload solo si quiere registrar esa configuración"
        )

    if device_map:
        print(f"Device map: {device_map}")

    warmup_item, _, warmup_prompt = prompts[0]
    print(f"Warm-up con {warmup_item['id']}")
    _ = generate_measured(torch, model, tokenizer, args.model_key, warmup_prompt, FORCED_THINK_CLOSE)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / f"{args.model_key}_efficiency_30.csv"
    json_path = OUTPUT_DIR / f"{args.model_key}_efficiency_30.json"
    summary_path = OUTPUT_DIR / f"{args.model_key}_efficiency_summary.json"

    rows = []
    run_start = time.perf_counter()

    for index, (item, chunks, prompt) in enumerate(prompts, start=1):
        measured = generate_measured(torch, model, tokenizer, args.model_key, prompt, FORCED_THINK_CLOSE)

        row = {
            "question_id": item["id"],
            "category": item["categoria"],
            "model_key": args.model_key,
            "model_name": MODEL_CONFIGS[args.model_key].name,
            "quantization": MODEL_CONFIGS[args.model_key].quantization,
            "gpu": gpu_name,
            "input_tokens": measured["input_tokens"],
            "output_tokens": measured["output_tokens"],
            "generation_seconds": round(measured["generation_seconds"], 6),
            "inference_seconds": round(measured["inference_seconds"], 6),
            "tokens_per_second": round(measured["tokens_per_second"], 6),
            "baseline_vram_gb": round(measured["baseline_vram_gb"], 6),
            "peak_vram_gb": round(measured["peak_vram_gb"], 6),
            "top_k": len(chunks),
            "answer": measured["answer"],
        }
        rows.append(row)
        write_rows(csv_path, json_path, rows)

        print(
            f"{index:02d}/30 {item['id']} "
            f"gen={row['generation_seconds']:.2f}s "
            f"tok/s={row['tokens_per_second']:.2f} "
            f"peak={row['peak_vram_gb']:.2f}GB"
        )

    wall_seconds = time.perf_counter() - run_start
    summary = make_summary(
        args.model_key,
        rows,
        gpu_name,
        torch.__version__,
        transformers.__version__,
    )
    summary["wall_seconds_measured_loop"] = wall_seconds
    summary["model_load_seconds"] = model_load_seconds
    summary["warmup_question_id"] = warmup_item["id"]
    summary["offload_detected"] = offload_detected
    summary["sample_seed"] = sample_data["seed"]

    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Listo")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    print(f"Resumen: {summary_path}")


if __name__ == "__main__":
    main()
