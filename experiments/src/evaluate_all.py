"""Evalúa los 3 modelos ya corridos, leyendo results/raw/*.json.

Corre evaluate_closed (accuracy de etiqueta Sí/No para "sino", F1 de token
para "corta") y evaluate_open (BERTScore + ROUGE-L + METEOR) sobre
"abierta", por cada modelo. Guarda el
detalle por pregunta en results/metrics/<model_key>_detalle.json y un
resumen comparativo de los 3 modelos en results/metrics/resumen_comparativo.csv,
además de imprimirlo en pantalla.

Ver el mensaje de ayuda al final de este archivo para las
instrucciones de instalación local.

Uso (desde experiments/):
    python src/evaluate_all.py
"""

import csv
import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from evaluation.metrics_closed import evaluate_closed
from evaluation.metrics_open import evaluate_open

PROJECT_ROOT = SRC_DIR.parent
RAW_DIR = PROJECT_ROOT / "results" / "raw"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"

SUMMARY_FIELDS = [
    "model_key", "sino_accuracy", "sino_sin_etiqueta_clara", "corta_f1",
    "abierta_bertscore", "abierta_rouge_l", "abierta_meteor",
]


def load_results(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_model(resultados: list[dict]) -> dict:
    closed = evaluate_closed(resultados)
    abierta = evaluate_open(resultados)

    detalle = closed["detalle"] + abierta["detalle"]
    detalle.sort(key=lambda d: str(d.get("pregunta_id")))

    promedios = {
        "sino_accuracy": closed["promedios"].get("sino_accuracy"),
        "sino_sin_etiqueta_clara": closed["promedios"].get("sino_sin_etiqueta_clara"),
        "corta_f1": closed["promedios"].get("corta"),
        "abierta_bertscore": abierta["promedios"].get("bertscore"),
        "abierta_rouge_l": abierta["promedios"].get("rouge_l"),
        "abierta_meteor": abierta["promedios"].get("meteor"),
    }

    return {"promedios": promedios, "detalle": detalle, "nota_meteor": abierta.get("nota_meteor")}


def _fmt(value) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def print_summary_table(rows: list[dict]):
    widths = {h: max(len(h), *(len(_fmt(r.get(h))) for r in rows)) for h in SUMMARY_FIELDS}

    header_line = "  ".join(h.ljust(widths[h]) for h in SUMMARY_FIELDS)
    print("\n" + header_line)
    print("-" * len(header_line))
    for r in rows:
        print("  ".join(_fmt(r.get(h)).ljust(widths[h]) for h in SUMMARY_FIELDS))


def main():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    raw_paths = sorted(RAW_DIR.glob("*.json"))
    if not raw_paths:
        raise FileNotFoundError(f"No hay resultados en {RAW_DIR} (corré primero pipeline.py)")

    resumen_rows = []
    nota_meteor = None

    for raw_path in raw_paths:
        model_key = raw_path.stem
        print(f"Evaluando {model_key}...")
        resultados = load_results(raw_path)

        evaluacion = evaluate_model(resultados)
        nota_meteor = evaluacion["nota_meteor"]

        detalle_path = METRICS_DIR / f"{model_key}_detalle.json"
        with open(detalle_path, "w", encoding="utf-8") as f:
            json.dump(evaluacion["detalle"], f, ensure_ascii=False, indent=2)
        print(f"  Detalle guardado en {detalle_path}")

        resumen_rows.append({"model_key": model_key, **evaluacion["promedios"]})

    resumen_path = METRICS_DIR / "resumen_comparativo.csv"
    with open(resumen_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(resumen_rows)
    print(f"\nResumen comparativo guardado en {resumen_path}")

    print_summary_table(resumen_rows)

    if nota_meteor:
        print(f"\nNota sobre METEOR: {nota_meteor}")


if __name__ == "__main__":
    main()
