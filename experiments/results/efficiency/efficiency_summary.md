# Efficiency benchmark summary

The benchmark uses the same 30 questions and the archived historical Top-5 context for all models

Retrieval latency is not included

| Model | Median generation latency s | Median tokens/s | Peak VRAM GB | Avg output tokens | Model load s |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-32B-Instruct | 5.661 | 16.36 | 19.63 | 141.1 | 130.1 |
| Llama 3.3 70B Instruct | 11.516 | 6.91 | 39.18 | 181.1 | 189.5 |
| DeepSeek-R1-Distill-Qwen-32B | 4.933 | 15.91 | 19.62 | 161.4 | 260.3 |

VRAM is the maximum memory allocated by PyTorch during measured generation

The comparison represents the deployed quantized configurations rather than intrinsic architecture efficiency