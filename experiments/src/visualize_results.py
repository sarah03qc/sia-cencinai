"""Genera 2 visualizaciones comparativas a partir de
results/metrics/resumen_comparativo.csv: un gráfico de barras agrupadas y
un gráfico de radar, con el mismo esquema de color modelo->color en ambos
(para que se puedan mirar juntas en el paper sin confundir qué color es
qué modelo).

METEOR se excluye de las 2 visualizaciones a propósito: ya documentamos en
evaluation/metrics_open.py que la implementación de nltk usa WordNet +
PorterStemmer en inglés, sin stemming ni sinónimos reales para español —
su validez como métrica semántica acá es limitada, así que no tiene
sentido darle el mismo peso visual que a sino_accuracy/corta_f1/BERTScore/
ROUGE-L en una figura pensada para el paper. Sigue disponible en el CSV y
en el detalle por pregunta, solo no se grafica.

No hardcodea los números de ningún modelo: todo sale de leer el CSV, así
que corre igual si mañana hay un cuarto modelo o cambian los promedios.

Uso (desde experiments/):
    python src/visualize_results.py
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_ROOT / "results" / "metrics" / "resumen_comparativo.csv"
FIGURES_DIR = PROJECT_ROOT / "results" / "metrics" / "figures"

METRICS = ["sino_accuracy", "corta_f1", "abierta_bertscore", "abierta_rouge_l"]
METRIC_LABELS = {
    "sino_accuracy": "Sí/No\n(accuracy)",
    "corta_f1": "Respuesta corta\n(F1 token)",
    "abierta_bertscore": "Abiertas\n(BERTScore)",
    "abierta_rouge_l": "Abiertas\n(ROUGE-L)",
}

DPI = 200

# Paleta fija (no depende del orden de filas del CSV): así el color de
# cada modelo queda estable entre corridas y consistente entre las 2 figuras.
_PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]


def load_summary(csv_path: Path = CSV_PATH) -> list[dict]:
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"{csv_path} está vacío")

    missing = set(METRICS) - set(rows[0])
    if missing:
        raise ValueError(f"Faltan columnas {missing} en {csv_path}")

    for row in rows:
        for metric in METRICS:
            row[metric] = float(row[metric])

    return rows


def assign_colors(rows: list[dict]) -> dict[str, str]:
    model_keys = sorted(row["model_key"] for row in rows)
    if len(model_keys) > len(_PALETTE):
        raise ValueError(f"Paleta fija de {len(_PALETTE)} colores no alcanza para {len(model_keys)} modelos")
    return {key: _PALETTE[i] for i, key in enumerate(model_keys)}


def plot_bars(rows: list[dict], colors: dict[str, str], out_path: Path):
    n_metrics = len(METRICS)
    n_models = len(rows)
    x = np.arange(n_metrics)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(10, 6), dpi=DPI)

    for i, row in enumerate(rows):
        values = [row[m] for m in METRICS]
        offset = (i - (n_models - 1) / 2) * width
        bars = ax.bar(x + offset, values, width, label=row["model_key"], color=colors[row["model_key"]])
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS])
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.08)
    ax.set_title("Comparación de modelos por métrica")
    ax.legend(title="Modelo")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_radar(rows: list[dict], colors: dict[str, str], out_path: Path):
    n_metrics = len(METRICS)
    angles = [n / n_metrics * 2 * np.pi for n in range(n_metrics)]
    angles += angles[:1]  # cerrar el polígono

    fig, ax = plt.subplots(figsize=(7, 7), dpi=DPI, subplot_kw={"polar": True})

    for row in rows:
        values = [row[m] for m in METRICS]
        values += values[:1]
        color = colors[row["model_key"]]
        ax.plot(angles, values, label=row["model_key"], color=color, linewidth=2, marker="o")
        ax.fill(angles, values, color=color, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS])
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_title("Comparación de modelos por métrica (radar)", pad=30)
    ax.legend(title="Modelo", loc="upper right", bbox_to_anchor=(1.35, 1.1))

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_summary()
    colors = assign_colors(rows)

    bars_path = FIGURES_DIR / "comparativo_barras.png"
    plot_bars(rows, colors, bars_path)
    print(f"Guardado: {bars_path}")

    radar_path = FIGURES_DIR / "comparativo_radar.png"
    plot_radar(rows, colors, radar_path)
    print(f"Guardado: {radar_path}")


if __name__ == "__main__":
    main()
