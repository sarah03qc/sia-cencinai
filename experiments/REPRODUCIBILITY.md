# Reproducibility guide

This repository now preserves the full revision workflow from the frozen benchmark through the final qualitative analysis

The post-processing and statistical results are reproducible from the archived raw model outputs

The original generation run is almost reproducible but still requires the seven institutional PDFs and the Kabré GPU environment

## What is archived

- Frozen benchmark with 299 questions
- Final authoritative evidence annotations
- Historical raw answers from Qwen, Llama, and DeepSeek
- Historical Top-5 chunk references used by all three model runs
- Snapshot of every unique chunk used in those Top-5 results
- Automatic metric scores by question
- Paired bootstrap results
- Final retrieval Hit@5 labels
- Blinded LLM-as-a-Judge input and 600 archived evaluations
- Final manual qualitative verdicts for 60 responses
- Error taxonomy and crosswalk

## Fast validation

From `experiments` run

```bash
python scripts/validate_revision_artifacts.py
```

This checks the main counts used in the revised paper

## Canonical post-processing order

Run these commands from `experiments`

```bash
python src/prepare_final_jsons.py
python src/check_retrieval_consistency.py
python src/export_historical_top5.py
python src/evaluate_evidence_hit_v2.py
python src/consolidate_retrieval_final.py
python src/bootstrap_final_metrics.py
python src/prepare_llm_judge_batch.py
python src/summarize_llm_judge.py
python src/prepare_qualitative_sample.py
python src/prepare_qualitative_replacements.py
python src/build_qualitative_final_sample.py
python src/analyze_qualitative_sample.py
python scripts/validate_revision_artifacts.py
```

`bootstrap_final_metrics.py` uses the archived per-question metric file

To recompute BERTScore, ROUGE-L, METEOR, Yes/No accuracy, and Short F1 from the raw generations first run

```bash
python src/recompute_final_metrics.py
python src/bootstrap_final_metrics.py
```

The BERTScore step is the slowest post-processing step and downloads `bert-base-multilingual-cased` if it is not already cached

## Revision metric environment

The metric recomputation that produced the archived results used

- Python 3.9
- torch 2.6.0 CPU build
- transformers 4.51.3
- bert-score 0.3.13
- rouge-score 0.1.2
- nltk 3.9.1

A lightweight dependency file is included as `requirements_revision.txt`

On Kabré the revision environment was activated after clearing a conflicting `PYTHONPATH`

```bash
unset PYTHONPATH
unset PYTHONHOME
source /work/squesada/venvs/cencinai-revision/bin/activate
export PYTHONNOUSERSITE=1
export HF_HOME=/work/squesada/.cache/huggingface
export NLTK_DATA=/work/squesada/.cache/nltk
export XDG_CACHE_HOME=/work/squesada/.cache
hash -r
```

The same setup is saved in `scripts/kabre_revision_setup.sh`

## Retrieval audit

The three original raw result files contain the historical `chunks_usados` for each question

`check_retrieval_consistency.py` verifies that Qwen, Llama, and DeepSeek received the same five chunks for all 300 original questions

The complete original FAISS index is not stored in this package

`data/rag_index/retrieved_chunks_snapshot.json` stores the 431 unique chunks that appeared in the historical Top-5 and is enough to rebuild the retrieval audit used in the revision

The retrieval audit has two stages

1. Conservative automatic evidence matching accepted 204 high-confidence Hit@5 cases
2. The remaining 95 cases were reviewed semantically against the authoritative evidence using a strict sufficiency criterion

The final result is 282/299 Hit@5 or 94.3 percent

Only final Hit@5 is used as the reported retrieval metric because rank-level decisions came from two different adjudication methods

## LLM-as-a-Judge

`prepare_llm_judge_batch.py` recreates the exact 200-question blinded input

The fixed mapping is stored separately in `results/llm_judge/llm_judge_model_key.txt`

The judge rubric is stored in `prompts/llm_judge_prompt_es.md`

The 600 returned evaluations are archived in `results/llm_judge/llm_judge_evaluaciones_600.json`

The external judge execution itself is not fully reproducible from this repository because the exact hosted model runtime is not pinned

The input construction, rubric, returned records, and all downstream summaries are preserved

## Qualitative sample

The original sample used a fixed stratified random draw with seed 20261008

It selected 10 short-answer and 10 open-ended questions and evaluated all three models on the same questions

Two sampled questions were replaced because the available evidence was not sufficient for a fair manual adjudication

The replacements were selected independently by category with seed 20261009

- `corta_040` was replaced by `corta_096`
- `abierta_073` was replaced by `abierta_095`

The final manual result is 47 B, 5 P, and 8 M across 60 responses

## Original model generation

The original RAG pipeline is in `src/pipeline.py`

It uses

- BGE-M3 embeddings
- FAISS exact L2 search over normalized embeddings
- Top-k 5
- 300-word chunks
- 50-word overlap
- deterministic generation with `do_sample=False`

The exact model settings are stored in `src/models/config.py` and `configs/`

The source PDFs are intentionally not included in this package

`data/source_documents/README.md` lists the expected filenames

## Known caveat

A later qualitative spot check identified `corta_012` as an item where the stored ground truth and the cited evidence are not perfectly aligned

The revision metrics were frozen because of the deadline and were not recomputed after that observation

Do not use that item as a qualitative example when discussing Token F1

## Efficiency benchmark added for the revision

A separate lightweight benchmark is included to report the basic computational efficiency requested by Reviewer 3

The sample is fixed in `data/efficiency_sample_30.json`

It contains 30 questions selected with seed 20261010

- 10 yes/no
- 10 short-answer
- 10 open-ended

All three models receive the same archived historical Top-5 chunks used in the original RAG experiment

This means the efficiency comparison measures the deployed generation configurations without introducing a new retrieval result

Run from `experiments`

```bash
python src/prepare_efficiency_sample.py
python src/benchmark_efficiency.py qwen2.5-32b
python src/benchmark_efficiency.py llama3.3-70b
python src/benchmark_efficiency.py deepseek-r1-distill-qwen-32b
python src/summarize_efficiency.py
```

A dry run can be used before requesting GPU time

```bash
python src/benchmark_efficiency.py qwen2.5-32b --dry-run
```

The three measured runs should use the same GPU type and expose only one GPU to the process

One warm-up generation is executed before each model run and excluded from the reported measurements

The primary reported values are

- median generation latency
- median generated tokens per second
- maximum PyTorch allocated VRAM
- mean output tokens

Model load time and total measured loop time are archived as supplementary values

Retrieval latency is not included because this revision benchmark reuses the exact historical Top-5 context

This should be described as an efficiency comparison of the evaluated deployable configurations rather than a comparison of intrinsic model architectures
