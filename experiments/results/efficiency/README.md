# Efficiency benchmark

This folder stores the basic computational efficiency profile requested during review

The benchmark uses a fixed stratified sample of 30 questions

- 10 yes/no
- 10 short-answer
- 10 open-ended
- seed 20261010

Every model receives the exact historical Top-5 chunks archived from the original RAG run

This design keeps the model comparison focused on the deployed generation configuration and avoids changing the retrieval context between runs

The measured values are

- generation latency
- inference latency including tokenization and decoding but excluding retrieval
- generated tokens
- tokens per second
- baseline PyTorch GPU memory
- peak PyTorch GPU memory

One warm-up generation is run before measurement and is not included in the results

The benchmark must be run on the same GPU type for the three models

The comparison represents the evaluated deployable configurations

It is not an intrinsic architecture comparison because the models use different 4-bit quantization methods

Expected files after the three runs

- `qwen2.5-32b_efficiency_30.csv`
- `llama3.3-70b_efficiency_30.csv`
- `deepseek-r1-distill-qwen-32b_efficiency_30.csv`
- one JSON copy and one summary JSON for each model
- `efficiency_summary.csv`
- `efficiency_summary.json`
- `efficiency_summary.md`
