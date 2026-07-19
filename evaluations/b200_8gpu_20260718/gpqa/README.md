# GPQA Diamond interim artifacts

This directory publishes three validated passes of a planned four-pass
GPQA Diamond evaluation on 8x NVIDIA B200.

## Validated results

| Pass | Correct | Wrong | Accuracy | Scored | Errors | Interrupted |
|---|---:|---:|---:|---:|---:|---|
| 1 | 181 | 17 | 91.41% | 198 | 0 | false |
| 2 | 178 | 20 | 89.90% | 198 | 0 | false |
| 4 | 179 | 19 | 90.40% | 198 | 0 | false |

Pass 1's canonical file merges the deterministic 64-item probe shard and exact
134-item complement. `merge_gpqa_pass1.py` is the merge/validation program. The
source and canonical artifacts are retained to make that construction auditable.

Protocol: `llm-decode-bench` 0.4.29; official GPQA Diamond 198; concurrency 64;
`max_tokens=100000`; temperature 1.0; top-p 0.95; exact option-letter scoring.
Dataset SHA-256:
`a8472c5a82ea2df8f209c17713aba1a6d409120c609ec0582dae0cb940c7e28c`.

Pass 3 and the final four-pass aggregate will be added after validation.
