#!/usr/bin/env python3

import csv
import json
import os
import re
import string
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BENCHMARK_PATH = ROOT / "data" / "benchmark_final_299.json"
RAW_DIR = ROOT / "results" / "raw"
OUT_DIR = ROOT / "results" / "revision_metrics"

MODEL_FILES = {
    "Qwen2.5-32B": RAW_DIR / "qwen2.5-32b.json",
    "Llama3.3-70B": RAW_DIR / "llama3.3-70b.json",
    "DeepSeek-R1-Distill-Qwen-32B":
        RAW_DIR / "deepseek-r1-distill-qwen-32b.json",
}

EXPECTED = {
    "sino": 99,
    "corta": 100,
    "abierta": 100,
}


# ============================================================
# MÉTRICAS CLOSED
# ============================================================

def strip_accents(text):
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(
        c for c in nfkd
        if not unicodedata.combining(c)
    )


def normalize_tokens(text):
    if not isinstance(text, str):
        return []

    text = text.lower()
    text = strip_accents(text)

    text = "".join(
        c for c in text
        if c not in string.punctuation
    )

    text = re.sub(r"\s+", " ", text).strip()

    return text.split()


def compute_f1(prediction, ground_truth):
    pred_tokens = normalize_tokens(prediction)
    gold_tokens = normalize_tokens(ground_truth)

    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    n_same = sum(common.values())

    if n_same == 0:
        return 0.0

    precision = n_same / len(pred_tokens)
    recall = n_same / len(gold_tokens)

    return (
        2 * precision * recall
        / (precision + recall)
    )


def extract_binary_label(text):
    if not isinstance(text, str) or not text.strip():
        return None

    normalized = strip_accents(
        text.strip().lower()
    )

    normalized = normalized.lstrip(
        string.punctuation + " "
    )

    first_word = re.split(
        r"[\s,.;:!?]",
        normalized,
        maxsplit=1,
    )[0]

    if first_word == "si":
        return "sí"

    if first_word == "no":
        return "no"

    return None


# ============================================================
# MÉTRICAS OPEN
# ============================================================

def compute_bertscore(predictions, references):

    from bert_score import score as bertscore_score

    print(
        "BERTScore: bert-base-multilingual-cased, "
        "CPU, batch_size=1"
    )

    _, _, f1 = bertscore_score(
        predictions,
        references,
        lang="es",
        rescale_with_baseline=True,
        batch_size=1,
        verbose=True,
    )

    return f1.tolist(), True
def compute_rouge_l(predictions, references):

    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(
        ["rougeL"],
        use_stemmer=False,
    )

    scores = []

    for pred, ref in zip(
        predictions,
        references,
    ):
        result = scorer.score(
            ref,
            pred,
        )

        scores.append(
            result["rougeL"].fmeasure
        )

    return scores


def ensure_nltk():

    import nltk

    nltk_dir = os.environ.get(
        "NLTK_DATA",
        str(
            ROOT
            / ".nltk_data"
        ),
    )

    os.environ["NLTK_DATA"] = nltk_dir

    Path(nltk_dir).mkdir(
        parents=True,
        exist_ok=True,
    )

    if nltk_dir not in nltk.data.path:
        nltk.data.path.insert(
            0,
            nltk_dir,
        )

    resources = {
        "wordnet": "corpora/wordnet",
        "omw-1.4": "corpora/omw-1.4",
        "punkt": "tokenizers/punkt",
        "punkt_tab": "tokenizers/punkt_tab",
    }

    for resource, path in resources.items():

        try:
            nltk.data.find(path)

        except LookupError:
            nltk.download(
                resource,
                download_dir=nltk_dir,
                quiet=True,
            )


def compute_meteor(
    predictions,
    references,
):

    ensure_nltk()

    from nltk.translate.meteor_score import (
        meteor_score,
    )

    from nltk.tokenize import word_tokenize

    scores = []

    for pred, ref in zip(
        predictions,
        references,
    ):

        pred_tokens = (
            word_tokenize(pred.lower())
            if isinstance(pred, str)
            and pred.strip()
            else []
        )

        ref_tokens = (
            word_tokenize(ref.lower())
            if isinstance(ref, str)
            and ref.strip()
            else []
        )

        if not pred_tokens or not ref_tokens:
            scores.append(0.0)
            continue

        scores.append(
            meteor_score(
                [ref_tokens],
                pred_tokens,
            )
        )

    return scores


# ============================================================
# HELPERS
# ============================================================

def load_json(path):

    if not path.exists():
        raise FileNotFoundError(
            f"No existe: {path}"
        )

    with open(
        path,
        encoding="utf-8",
    ) as f:
        return json.load(f)


def normalize_ws(text):

    return " ".join(
        str(text or "").split()
    )


def validate_benchmark(benchmark):

    if len(benchmark) != 299:
        raise ValueError(
            f"Esperaba 299 preguntas, "
            f"encontré {len(benchmark)}"
        )

    ids = [
        x["id"]
        for x in benchmark
    ]

    if len(ids) != len(set(ids)):
        raise ValueError(
            "Hay IDs duplicados."
        )

    if "sino_023" in ids:
        raise ValueError(
            "sino_023 sigue presente."
        )

    counts = {}

    for x in benchmark:
        cat = x["categoria"]
        counts[cat] = (
            counts.get(cat, 0) + 1
        )

    if counts != EXPECTED:
        raise ValueError(
            f"Conteos incorrectos: {counts}"
        )


def align_results(
    benchmark,
    raw,
    model_name,
):

    by_id = {
        x["pregunta_id"]: x
        for x in raw
    }

    if len(by_id) != len(raw):
        raise ValueError(
            f"{model_name}: IDs duplicados"
        )

    aligned = []

    for final in benchmark:

        qid = final["id"]

        if qid not in by_id:
            raise ValueError(
                f"{model_name}: falta {qid}"
            )

        old = by_id[qid]

        if normalize_ws(
            old.get("pregunta")
        ) != normalize_ws(
            final["pregunta"]
        ):
            raise ValueError(
                f"{model_name}: "
                f"cambió pregunta {qid}"
            )

        aligned.append({
            "pregunta_id":
                qid,

            "categoria":
                final["categoria"],

            "pregunta":
                final["pregunta"],

            # GT DEFINITIVO
            "ground_truth":
                final["ground_truth"],

            # RESPUESTA HISTÓRICA
            "respuesta_generada":
                old.get(
                    "respuesta_generada"
                ) or "",

            "error":
                old.get("error"),
        })

    return aligned


# ============================================================
# MAIN
# ============================================================

def main():

    benchmark = load_json(
        BENCHMARK_PATH
    )

    validate_benchmark(
        benchmark
    )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 78)
    print(
        "RECALCULO DE METRICAS "
        "— BENCHMARK FINAL 299"
    )
    print("=" * 78)

    print(
        "99 sino | 100 corta | "
        "100 abierta"
    )

    print(
        "sino_023 excluida"
    )

    print()

    all_rows = []
    summaries = []

    bert_baseline_used = None

    for model_name, raw_path in (
        MODEL_FILES.items()
    ):

        print("-" * 78)
        print(
            f"MODELO: {model_name}"
        )

        raw = load_json(
            raw_path
        )

        data = align_results(
            benchmark,
            raw,
            model_name,
        )

        # ====================================================
        # SINO
        # ====================================================

        sino_scores = []
        no_clear = 0

        for item in data:

            if item["categoria"] != "sino":
                continue

            pred_label = (
                extract_binary_label(
                    item[
                        "respuesta_generada"
                    ]
                )
            )

            gold_label = (
                extract_binary_label(
                    item["ground_truth"]
                )
            )

            if pred_label is None:
                no_clear += 1

            accuracy = float(
                pred_label is not None
                and pred_label == gold_label
            )

            sino_scores.append(
                accuracy
            )

            all_rows.append({
                "question_id":
                    item["pregunta_id"],

                "category":
                    "sino",

                "model":
                    model_name,

                "accuracy":
                    accuracy,

                "token_f1":
                    "",

                "bertscore_f1":
                    "",

                "rouge_l":
                    "",

                "meteor":
                    "",

                "predicted_label":
                    pred_label or "",

                "expected_label":
                    gold_label or "",

                "generation_error":
                    bool(
                        item.get("error")
                    )
                    or not bool(
                        item[
                            "respuesta_generada"
                        ].strip()
                    ),
            })

        sino_accuracy = (
            sum(sino_scores)
            / len(sino_scores)
        )

        # ====================================================
        # CORTA
        # ====================================================

        short_scores = []

        for item in data:

            if item["categoria"] != "corta":
                continue

            f1 = compute_f1(
                item[
                    "respuesta_generada"
                ],
                item[
                    "ground_truth"
                ],
            )

            short_scores.append(
                f1
            )

            all_rows.append({
                "question_id":
                    item["pregunta_id"],

                "category":
                    "corta",

                "model":
                    model_name,

                "accuracy":
                    "",

                "token_f1":
                    f1,

                "bertscore_f1":
                    "",

                "rouge_l":
                    "",

                "meteor":
                    "",

                "predicted_label":
                    "",

                "expected_label":
                    "",

                "generation_error":
                    bool(
                        item.get("error")
                    )
                    or not bool(
                        item[
                            "respuesta_generada"
                        ].strip()
                    ),
            })

        short_f1 = (
            sum(short_scores)
            / len(short_scores)
        )

        # ====================================================
        # ABIERTA
        # ====================================================

        abiertas = [
            x for x in data
            if x["categoria"]
            == "abierta"
        ]

        predictions = [
            x["respuesta_generada"]
            for x in abiertas
        ]

        references = [
            x["ground_truth"]
            for x in abiertas
        ]

        print(
            "Calculando BERTScore..."
        )

        bert, baseline_used = (
            compute_bertscore(
                predictions,
                references,
            )
        )

        if bert_baseline_used is None:
            bert_baseline_used = (
                baseline_used
            )

        elif (
            bert_baseline_used
            != baseline_used
        ):
            raise ValueError(
                "BERTScore usó baseline "
                "de forma inconsistente "
                "entre modelos."
            )

        print(
            "Calculando ROUGE-L..."
        )

        rouge = compute_rouge_l(
            predictions,
            references,
        )

        print(
            "Calculando METEOR..."
        )

        meteor = compute_meteor(
            predictions,
            references,
        )

        if not (
            len(bert)
            == len(rouge)
            == len(meteor)
            == 100
        ):
            raise ValueError(
                "Open metrics no produjo "
                "100 scores."
            )

        for (
            item,
            bs,
            rl,
            mt,
        ) in zip(
            abiertas,
            bert,
            rouge,
            meteor,
        ):

            all_rows.append({
                "question_id":
                    item["pregunta_id"],

                "category":
                    "abierta",

                "model":
                    model_name,

                "accuracy":
                    "",

                "token_f1":
                    "",

                "bertscore_f1":
                    float(bs),

                "rouge_l":
                    float(rl),

                "meteor":
                    float(mt),

                "predicted_label":
                    "",

                "expected_label":
                    "",

                "generation_error":
                    bool(
                        item.get("error")
                    )
                    or not bool(
                        item[
                            "respuesta_generada"
                        ].strip()
                    ),
            })

        bert_mean = (
            sum(bert)
            / len(bert)
        )

        rouge_mean = (
            sum(rouge)
            / len(rouge)
        )

        meteor_mean = (
            sum(meteor)
            / len(meteor)
        )

        summary = {
            "model":
                model_name,

            "n_total":
                299,

            "n_sino":
                99,

            "sino_accuracy":
                sino_accuracy,

            "sino_no_clear_label":
                no_clear,

            "n_corta":
                100,

            "corta_token_f1":
                short_f1,

            "n_abierta":
                100,

            "abierta_bertscore_f1":
                bert_mean,

            "abierta_rouge_l":
                rouge_mean,

            "abierta_meteor":
                meteor_mean,
        }

        summaries.append(
            summary
        )

        print()
        print(
            f"Accuracy : "
            f"{sino_accuracy:.6f}"
        )

        print(
            f"Token F1 : "
            f"{short_f1:.6f}"
        )

        print(
            f"BERTScore: "
            f"{bert_mean:.6f}"
        )

        print(
            f"ROUGE-L  : "
            f"{rouge_mean:.6f}"
        )

        print(
            f"METEOR   : "
            f"{meteor_mean:.6f}"
        )

        print(
            f"No clear binary label: "
            f"{no_clear}"
        )

        print()

    # ========================================================
    # VALIDACIONES
    # ========================================================

    if len(all_rows) != 897:
        raise ValueError(
            f"Esperaba 897 filas, "
            f"encontré {len(all_rows)}"
        )

    seen = set()

    for row in all_rows:

        key = (
            row["question_id"],
            row["model"],
        )

        if key in seen:
            raise ValueError(
                f"Duplicado: {key}"
            )

        seen.add(key)

    # ========================================================
    # EXPORTAR
    # ========================================================

    per_question_path = (
        OUT_DIR
        / "automatic_metrics_per_question.csv"
    )

    fields = [
        "question_id",
        "category",
        "model",
        "accuracy",
        "token_f1",
        "bertscore_f1",
        "rouge_l",
        "meteor",
        "predicted_label",
        "expected_label",
        "generation_error",
    ]

    with open(
        per_question_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(
            all_rows
        )

    summary_csv = (
        OUT_DIR
        / "automatic_metrics_summary.csv"
    )

    summary_fields = [
        "model",
        "n_total",
        "n_sino",
        "sino_accuracy",
        "sino_no_clear_label",
        "n_corta",
        "corta_token_f1",
        "n_abierta",
        "abierta_bertscore_f1",
        "abierta_rouge_l",
        "abierta_meteor",
    ]

    with open(
        summary_csv,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=summary_fields,
        )

        writer.writeheader()
        writer.writerows(
            summaries
        )

    summary_json = (
        OUT_DIR
        / "automatic_metrics_summary.json"
    )

    with open(
        summary_json,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            {
                "benchmark": {
                    "n_total": 299,
                    "sino": 99,
                    "corta": 100,
                    "abierta": 100,
                    "excluded": [
                        "sino_023"
                    ],
                    "ground_truth_source":
                        "benchmark_final_299.json",
                },

                "bertscore": {
                    "lang": "es",
                    "rescale_with_baseline":
                        bert_baseline_used,
                },

                "models":
                    summaries,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("=" * 78)
    print("RESULTADOS FINALES")
    print("=" * 78)

    print(
        f"{'MODEL':34s}"
        f"{'ACC':>10s}"
        f"{'F1':>10s}"
        f"{'BERT':>10s}"
        f"{'ROUGE':>10s}"
        f"{'METEOR':>10s}"
    )

    print("-" * 84)

    for s in summaries:

        print(
            f"{s['model']:34s}"
            f"{s['sino_accuracy']:10.6f}"
            f"{s['corta_token_f1']:10.6f}"
            f"{s['abierta_bertscore_f1']:10.6f}"
            f"{s['abierta_rouge_l']:10.6f}"
            f"{s['abierta_meteor']:10.6f}"
        )

    print()

    print(
        "BERTScore "
        "rescale_with_baseline = "
        f"{bert_baseline_used}"
    )

    print()

    print(
        "Archivos creados:"
    )

    print(
        per_question_path
    )

    print(
        summary_csv
    )

    print(
        summary_json
    )

    print()

    print(
        "OK — métricas recalculadas "
        "sobre benchmark final 299."
    )


if __name__ == "__main__":
    main()
