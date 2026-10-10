# Validation status

The repository was checked after the reproducibility cleanup

## Rerun successfully from the packaged files

- `prepare_final_jsons.py`
- `check_retrieval_consistency.py`
- `export_historical_top5.py`
- `evaluate_evidence_hit_v2.py`
- `consolidate_retrieval_final.py`
- `bootstrap_final_metrics.py`
- `prepare_llm_judge_batch.py`
- `summarize_llm_judge.py`
- `prepare_qualitative_sample.py`
- `prepare_qualitative_replacements.py`
- `build_qualitative_final_sample.py`
- `analyze_qualitative_sample.py`
- `validate_revision_artifacts.py`

## Preserved from a previously successful run

`recompute_final_metrics.py` was not rerun during packaging because BERTScore requires the multilingual model cache or a download

Its archived per-question and summary outputs are present under `results/revision_metrics/`

The script was syntax checked and its output is the input used by the rerun bootstrap analysis

## Not rerun during packaging

The original three-model generation pipeline was not rerun because it requires the seven source PDFs and the Kabré GPU environment

The historical raw outputs from all three model runs are archived under `results/raw/`

## Final integrity check

`python scripts/validate_revision_artifacts.py` passed with these core counts

- 299 frozen benchmark questions
- 282 final retrieval Hit@5 cases and 17 Miss@5 cases
- 897 automatic metric rows
- 600 LLM-as-a-Judge evaluations
- 60 final qualitative responses
- 47 B, 5 P, 8 M manual verdicts

## Efficiency benchmark preparation

The efficiency scripts were added after the main revision workflow was frozen

Validated locally without GPU

- fixed 30-question sample generation
- 10 questions per category
- historical Top-5 availability for all 30 questions
- dry-run validation for all three model keys
- Python syntax for the three efficiency scripts

GPU execution remains pending on Kabré
