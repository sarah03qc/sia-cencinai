# Script map

## Canonical revision workflow

1. `prepare_final_jsons.py` builds the frozen 299-question benchmark and final evidence annotations from the two audit workbooks
2. `check_retrieval_consistency.py` confirms that the three historical model runs received the same Top-5 chunks
3. `export_historical_top5.py` rebuilds the historical Top-5 dataset from the raw model output and the archived chunk snapshot
4. `evaluate_evidence_hit_v2.py` performs the conservative automatic retrieval screening
5. `consolidate_retrieval_final.py` combines the automatic screening with the strict 95-case semantic review and produces final Hit@5
6. `recompute_final_metrics.py` recalculates the automatic metrics on the frozen 299-question benchmark
7. `bootstrap_final_metrics.py` runs the paired 10,000-sample bootstrap with seed 20261008
8. `prepare_llm_judge_batch.py` rebuilds the blinded 200-question judge input
9. `summarize_llm_judge.py` validates and summarizes the archived 600 judge evaluations
10. `prepare_qualitative_sample.py` recreates the original stratified random sample with seed 20261008
11. `prepare_qualitative_replacements.py` recreates the two replacement choices with seed 20261009
12. `build_qualitative_final_sample.py` builds the final 20-question and 60-response qualitative sample
13. `analyze_qualitative_sample.py` joins manual ratings, judge results, automatic metrics, retrieval, and the error taxonomy

## Original benchmark generation

`pipeline.py`, `models/`, and `rag/` contain the original RAG generation path

That path needs the seven source PDFs and a GPU environment

## Intermediate scripts

`evaluate_evidence_hit.py`, `diagnose_evidence_hit.py`, and `inspect_suspicious_matches.py` are retained because they document the earlier retrieval audit stages

The paper should use `retrieval_final_299.csv` rather than the intermediate screening counts
