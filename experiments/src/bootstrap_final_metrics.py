#!/usr/bin/env python3

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT
    / "results"
    / "revision_metrics"
    / "automatic_metrics_per_question.csv"
)

OUT_DIR = (
    ROOT
    / "results"
    / "revision_metrics"
    / "bootstrap"
)

N_BOOT = 10_000
SEED = 20261008

MODELS = [
    "Qwen2.5-32B",
    "Llama3.3-70B",
    "DeepSeek-R1-Distill-Qwen-32B",
]

METRICS = {
    "sino_accuracy": {
        "category": "sino",
        "column": "accuracy",
    },
    "corta_token_f1": {
        "category": "corta",
        "column": "token_f1",
    },
    "abierta_bertscore_f1": {
        "category": "abierta",
        "column": "bertscore_f1",
    },
    "abierta_rouge_l": {
        "category": "abierta",
        "column": "rouge_l",
    },
    "abierta_meteor": {
        "category": "abierta",
        "column": "meteor",
    },
}


def percentile_ci(values):
    return (
        float(np.percentile(values, 2.5)),
        float(np.percentile(values, 97.5)),
    )


def main():

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(INPUT)

    print("=" * 78)
    print("PAIRED BOOTSTRAP — FINAL AUTOMATIC METRICS")
    print("=" * 78)
    print(f"Bootstrap samples: {N_BOOT}")
    print(f"Seed: {SEED}")
    print()

    model_ci_rows = []
    pairwise_rows = []

    rng = np.random.default_rng(SEED)

    for metric_name, config in METRICS.items():

        category = config["category"]
        column = config["column"]

        print("-" * 78)
        print(metric_name)

        sub = df[
            df["category"] == category
        ].copy()

        # --------------------------------------------------------
        # Pivot: una fila por pregunta,
        # una columna por modelo
        # --------------------------------------------------------

        pivot = sub.pivot(
            index="question_id",
            columns="model",
            values=column,
        )

        # Garantizar mismo conjunto de preguntas
        missing_models = [
            m for m in MODELS
            if m not in pivot.columns
        ]

        if missing_models:
            raise ValueError(
                f"{metric_name}: faltan modelos "
                f"{missing_models}"
            )

        pivot = pivot[MODELS]

        if pivot.isna().any().any():
            raise ValueError(
                f"{metric_name}: hay scores faltantes."
            )

        question_ids = pivot.index.tolist()
        values = pivot.to_numpy(
            dtype=float
        )

        n_questions = len(question_ids)

        print(
            f"Preguntas pareadas: "
            f"{n_questions}"
        )

        # --------------------------------------------------------
        # Observed means
        # --------------------------------------------------------

        observed = values.mean(axis=0)

        # --------------------------------------------------------
        # Bootstrap PAREADO
        #
        # Se generan los mismos índices de pregunta
        # para los 3 modelos.
        # --------------------------------------------------------

        boot_means = np.empty(
            (
                N_BOOT,
                len(MODELS),
            ),
            dtype=float,
        )

        for b in range(N_BOOT):

            indices = rng.integers(
                0,
                n_questions,
                size=n_questions,
            )

            sample = values[
                indices,
                :
            ]

            boot_means[b, :] = (
                sample.mean(axis=0)
            )

        # --------------------------------------------------------
        # CI individual de cada modelo
        # --------------------------------------------------------

        for j, model in enumerate(MODELS):

            low, high = percentile_ci(
                boot_means[:, j]
            )

            model_ci_rows.append({
                "metric":
                    metric_name,

                "category":
                    category,

                "model":
                    model,

                "n_questions":
                    n_questions,

                "observed_mean":
                    float(observed[j]),

                "ci95_low":
                    low,

                "ci95_high":
                    high,

                "n_boot":
                    N_BOOT,

                "seed":
                    SEED,
            })

            print(
                f"{model:34s} "
                f"{observed[j]:.6f} "
                f"[{low:.6f}, "
                f"{high:.6f}]"
            )

        # --------------------------------------------------------
        # CI de diferencias PAREADAS
        # --------------------------------------------------------

        print()
        print("Diferencias pareadas:")

        for model_a, model_b in combinations(
            MODELS,
            2,
        ):

            a = MODELS.index(model_a)
            b = MODELS.index(model_b)

            observed_diff = (
                observed[a]
                - observed[b]
            )

            boot_diff = (
                boot_means[:, a]
                - boot_means[:, b]
            )

            low, high = percentile_ci(
                boot_diff
            )

            excludes_zero = (
                low > 0
                or high < 0
            )

            if low > 0:
                favored = model_a
            elif high < 0:
                favored = model_b
            else:
                favored = "none"

            pairwise_rows.append({
                "metric":
                    metric_name,

                "category":
                    category,

                "model_a":
                    model_a,

                "model_b":
                    model_b,

                "n_questions":
                    n_questions,

                "observed_difference_a_minus_b":
                    float(observed_diff),

                "ci95_low":
                    low,

                "ci95_high":
                    high,

                "ci_excludes_zero":
                    bool(excludes_zero),

                "favored_if_ci_excludes_zero":
                    favored,

                "n_boot":
                    N_BOOT,

                "seed":
                    SEED,
            })

            print(
                f"{model_a} - {model_b}: "
                f"{observed_diff:+.6f} "
                f"[{low:+.6f}, "
                f"{high:+.6f}] "
                f"{'*' if excludes_zero else ''}"
            )

        print()

    # ============================================================
    # EXPORT
    # ============================================================

    model_df = pd.DataFrame(
        model_ci_rows
    )

    pair_df = pd.DataFrame(
        pairwise_rows
    )

    model_path = (
        OUT_DIR
        / "bootstrap_model_cis.csv"
    )

    pair_path = (
        OUT_DIR
        / "bootstrap_pairwise_differences.csv"
    )

    json_path = (
        OUT_DIR
        / "bootstrap_summary.json"
    )

    model_df.to_csv(
        model_path,
        index=False,
    )

    pair_df.to_csv(
        pair_path,
        index=False,
    )

    payload = {
        "method": {
            "type":
                "paired nonparametric bootstrap",

            "n_boot":
                N_BOOT,

            "confidence_interval":
                "percentile 95%",

            "seed":
                SEED,

            "pairing":
                "same resampled question indices "
                "used for all models within each metric",
        },

        "model_confidence_intervals":
            model_ci_rows,

        "pairwise_differences":
            pairwise_rows,
    }

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            payload,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("=" * 78)
    print("BOOTSTRAP COMPLETADO")
    print("=" * 78)

    print(model_path)
    print(pair_path)
    print(json_path)

    print()
    print(
        "* = CI 95% de la diferencia "
        "no incluye 0."
    )


if __name__ == "__main__":
    main()
