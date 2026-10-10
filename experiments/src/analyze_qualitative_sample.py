#!/usr/bin/env python3

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QDIR = ROOT / "results" / "qualitative_analysis"
JDIR = ROOT / "results" / "llm_judge"
RDIR = ROOT / "results" / "retrieval_audit"

SAMPLE = QDIR / "qualitative_final_60_responses.csv"
MANUAL = QDIR / "manual_verdicts_60.csv"
ERRORS = QDIR / "manual_error_taxonomy_13.csv"
JUDGE = JDIR / "llm_judge_evaluaciones_600.json"
RETRIEVAL = RDIR / "retrieval_final_299.csv"
OUT_CSV = QDIR / "analisis_cualitativo_crosswalk_60.csv"
OUT_JSON = QDIR / "qualitative_analysis_summary.json"
OUT_MD = QDIR / "analisis_cualitativo_taxonomia_60.md"

MODEL_KEY = {
    "A": "Qwen2.5-32B",
    "B": "Llama3.3-70B",
    "C": "DeepSeek-R1-Distill-Qwen-32B",
}

ORDER = ["FAIL", "PARTIAL", "PASS"]


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def kappa(a, b, weighted=False):
    n = len(a)
    index = {label: i for i, label in enumerate(ORDER)}
    counts_a = Counter(a)
    counts_b = Counter(b)

    def weight(x, y):
        if not weighted:
            return 1.0 if x == y else 0.0
        return 1.0 - abs(index[x] - index[y]) / (len(ORDER) - 1)

    observed = sum(weight(x, y) for x, y in zip(a, b)) / n
    expected = 0.0
    for x in ORDER:
        for y in ORDER:
            expected += (counts_a[x] / n) * (counts_b[y] / n) * weight(x, y)

    if expected == 1.0:
        return 1.0
    return (observed - expected) / (1.0 - expected)


def mean(values):
    values = [float(x) for x in values if str(x).strip()]
    return sum(values) / len(values) if values else None


def main():
    sample = read_csv(SAMPLE)
    manual = {(x["question_id"], x["model"]): x["manual_grade"] for x in read_csv(MANUAL)}
    errors = {(x["question_id"], x["model"]): x for x in read_csv(ERRORS)}
    retrieval = {x["question_id"]: x for x in read_csv(RETRIEVAL)}
    judge_raw = json.loads(JUDGE.read_text(encoding="utf-8"))
    judge = {(x["question_id"], MODEL_KEY[x["response_id"]]): x for x in judge_raw}

    rows = []
    manual_map = {"B": "PASS", "P": "PARTIAL", "M": "FAIL"}

    for item in sample:
        key = (item["question_id"], item["model"])
        grade = manual[key]
        j = judge[key]
        err = errors.get(key)
        rows.append({
            **item,
            "manual_grade": grade,
            "manual_equivalent": manual_map[grade],
            "primary_error_type": err["primary_error_type"] if err else "NONE_MANUAL",
            "error_rationale": err["rationale"] if err else "",
            "llm_judge_verdict": j["overall_verdict"],
            "judge_factual_correctness": j["factual_correctness"],
            "judge_completeness": j["completeness"],
            "judge_faithfulness": j["faithfulness_to_evidence"],
            "judge_unsupported_information": j["unsupported_information"],
            "judge_brief_justification": j["brief_justification"],
            "retrieval_hit_at_5": retrieval[item["question_id"]]["final_hit_at_5"],
        })

    if len(rows) != 60:
        raise ValueError(f"Expected 60 rows, found {len(rows)}")

    with open(OUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    manual_counts = defaultdict(Counter)
    judge_counts = defaultdict(Counter)
    for row in rows:
        manual_counts[row["model"]][row["manual_grade"]] += 1
        judge_counts[row["model"]][row["llm_judge_verdict"]] += 1

    manual_equiv = [x["manual_equivalent"] for x in rows]
    judge_verdict = [x["llm_judge_verdict"] for x in rows]
    agreement = sum(a == b for a, b in zip(manual_equiv, judge_verdict))

    open_rows = [x for x in rows if x["category"] == "abierta"]
    metric_means = {}
    for grade in ["B", "P", "M"]:
        subset = [x for x in open_rows if x["manual_grade"] == grade]
        metric_means[grade] = {
            "bertscore_f1": mean(x["bertscore_f1"] for x in subset),
            "rouge_l": mean(x["rouge_l"] for x in subset),
            "meteor": mean(x["meteor"] for x in subset),
        }

    summary = {
        "n_questions": 20,
        "n_responses": 60,
        "manual_counts": {model: dict(counts) for model, counts in manual_counts.items()},
        "judge_counts": {model: dict(counts) for model, counts in judge_counts.items()},
        "manual_judge_agreement": agreement / 60,
        "manual_judge_agreement_n": agreement,
        "cohen_kappa": kappa(manual_equiv, judge_verdict),
        "weighted_cohen_kappa": kappa(manual_equiv, judge_verdict, weighted=True),
        "taxonomy_counts": dict(Counter(x["primary_error_type"] for x in rows)),
        "all_sampled_questions_hit_at_5": all(x["retrieval_hit_at_5"] == "YES" for x in rows),
        "open_metric_means_by_manual_grade": metric_means,
    }
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Qualitative analysis and error taxonomy",
        "",
        "## Manual results",
        "",
        "| Model | B | P | M | B rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in ["Qwen2.5-32B", "Llama3.3-70B", "DeepSeek-R1-Distill-Qwen-32B"]:
        c = manual_counts[model]
        lines.append(f"| {model} | {c['B']} | {c['P']} | {c['M']} | {c['B'] / 20:.1%} |")

    lines += [
        "",
        "## Error taxonomy",
        "",
        f"- Generation failure or no-answer: {summary['taxonomy_counts'].get('GENERATION_FAILURE_NO_ANSWER', 0)}",
        f"- Incomplete or underdeveloped: {summary['taxonomy_counts'].get('INCOMPLETE_OR_UNDERDEVELOPED', 0)}",
        f"- Retrieval miss in the final sample: {0 if summary['all_sampled_questions_hit_at_5'] else 'present'}",
        "",
        "## Manual review vs LLM-as-a-Judge",
        "",
        f"Agreement: {agreement}/60 = {agreement / 60:.1%}",
        "",
        f"Cohen kappa: {summary['cohen_kappa']:.3f}",
        "",
        f"Linear weighted kappa: {summary['weighted_cohen_kappa']:.3f}",
        "",
        "## Automatic metrics on open questions",
        "",
        "| Manual | BERTScore | ROUGE-L | METEOR |",
        "|:---:|---:|---:|---:|",
    ]
    for grade in ["B", "P", "M"]:
        m = metric_means[grade]
        lines.append(f"| {grade} | {m['bertscore_f1']:.3f} | {m['rouge_l']:.3f} | {m['meteor']:.3f} |")

    lines += [
        "",
        "## Interpretation",
        "",
        "All 30 short-answer responses in the final qualitative sample were rated B",
        "",
        "Differences between models appeared in the open-ended questions",
        "",
        "All 20 sampled questions were final Hit@5 cases, so the manual errors in this sample are generation or context-use errors rather than pure retrieval misses",
        "",
        "The LLM-as-a-Judge and manual review agreed on the clear failures and disagreed mainly on borderline B or P cases and unsupported extra details",
        "",
        "Automatic metrics separate B, P, and M on average but still overlap at the individual response level",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print(OUT_CSV)
    print(OUT_JSON)
    print(OUT_MD)


if __name__ == "__main__":
    main()
