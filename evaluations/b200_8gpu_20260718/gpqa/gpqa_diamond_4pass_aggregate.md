# GPQA Diamond: four-pass B200 aggregate

| Pass | Correct | Wrong | Accuracy | Truncated/no answer | Hit 100K cap | Errors |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 181 | 17 | 91.41% | 1 | 9 | 0 |
| 2 | 178 | 20 | 89.90% | 3 | 9 | 0 |
| 3 | 179 | 19 | 90.40% | 3 | 8 | 0 |
| 4 | 179 | 19 | 90.40% | 0 | 5 | 0 |

**Four-pass mean / pooled repeated-generation accuracy:** 717/792 = **90.53%**.
Pass range: 89.90%-91.41%; sample SD across passes: 0.64 percentage points.
Truncated/no-answer generations: 7; hit 100K output cap: 31; API errors: 0.

## Token and throughput summary

Total completion tokens: 20043342; mean 25307.2; median 14558.0; p90 71196.8; p99 100000.0.
Mean per-pass aggregate generation throughput: 12.27 tok/s; end-to-end: 12.25 tok/s.
Finish reasons: `{"length": 31, "stop": 761}`.

## Methodology disclosure

Pass 1 is the canonical, validated merge of a deterministic 64-item shard and its exact 134-item complement. The other passes are ordinary full 198-item runs. Every pass used llm-decode-bench 0.4.29, fixed concurrency 64, max_tokens=100000, temperature 1.0, top_p 0.95, deterministic option shuffling, and exact option-letter scoring. Every pass has 198 unique official item IDs, zero errors, and interrupted=false.

The 792 observations are repeated stochastic generations over 198 questions, not 792 unique questions. Accordingly, the per-pass distribution is more informative than a naive pooled confidence interval.
