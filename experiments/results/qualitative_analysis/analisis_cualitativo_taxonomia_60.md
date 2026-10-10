# Qualitative analysis and error taxonomy

## Manual results

| Model | B | P | M | B rate |
|---|---:|---:|---:|---:|
| Qwen2.5-32B | 14 | 3 | 3 | 70.0% |
| Llama3.3-70B | 17 | 1 | 2 | 85.0% |
| DeepSeek-R1-Distill-Qwen-32B | 16 | 1 | 3 | 80.0% |

## Error taxonomy

- Generation failure or no-answer: 8
- Incomplete or underdeveloped: 5
- Retrieval miss in the final sample: 0

## Manual review vs LLM-as-a-Judge

Agreement: 51/60 = 85.0%

Cohen kappa: 0.639

Linear weighted kappa: 0.686

## Automatic metrics on open questions

| Manual | BERTScore | ROUGE-L | METEOR |
|:---:|---:|---:|---:|
| B | 0.232 | 0.225 | 0.297 |
| P | 0.209 | 0.196 | 0.261 |
| M | 0.152 | 0.158 | 0.139 |

## Interpretation

All 30 short-answer responses in the final qualitative sample were rated B

Differences between models appeared in the open-ended questions

All 20 sampled questions were final Hit@5 cases, so the manual errors in this sample are generation or context-use errors rather than pure retrieval misses

The LLM-as-a-Judge and manual review agreed on the clear failures and disagreed mainly on borderline B or P cases and unsupported extra details

Automatic metrics separate B, P, and M on average but still overlap at the individual response level