# Results

All performance figures below are measured results, not projections. Raw files
are retained under `evaluations/`; server logs and container manifests are under
`evidence/`.

## Decode and capacity

| Configuration | 0 context | 32K | 128K | KV tokens | Outcome |
|---|---:|---:|---:|---:|---|
| Original/A2A baseline, MTP3 | 62.3 | 54.6 | 59.3 | 908,032 | superseded |
| Full quality stack, DCP4/MTP3 | 61.2 | 59.2 | 53.8 | 740,608 | best KLD, lower capacity |
| Full quality stack, DCP2/MTP3 | 64.1 | 64.2 | 59.1 | 370,432 | rejected: half the DCP4 capacity |
| Slim deterministic, DCP4/MTP2 | 60.3 | 58.4 | 54.5 | 922,880 | slower than MTP3 |
| **Slim deterministic, DCP4/MTP3** | **62.5** | **63.2** | **62.6** | **923,136** | selected |
| Slim deterministic, DCP4/MTP5 | 50.0 | 43.3 | 42.2 | 923,136 | rejected |
| DeepSpark BF16 head, DCP4 | 48.1 | 30.9 | not run | 673,483 | rejected |
| MTP3, utilization 0.970, c4/128K | — | — | 0.0, four errors | 989,184 | rejected: CUDA OOM |

Decode runs used `llm-decode-bench` v0.4.28, concurrency 1 unless explicitly
marked, 30-second sustained cells, 4,096 maximum output tokens, and prefill
excluded from the throughput window.

## MTP and DeepSpark acceptance

- MTP2: 3,768 accepted of 5,468 drafted tokens, **68.9%**.
- MTP5: 3,503 accepted of 8,110 drafted tokens, **43.2%**.
- DeepSpark BF16: 1,679 accepted of 9,380 drafted tokens, **17.9%** overall;
  acceptance degraded sharply at 32K.
- The NVFP4 DeepSpark head did not initialize. Its packed LM head was width 3,072
  while the port instantiated width 6,144.

## Prefill

| Candidate | 8K | 16K | 64K |
|---|---:|---:|---:|
| Original Python chunk-128 path | about 800 | — | — |
| Fused route packing + tile balancing | 1,329 | 1,329 | 1,270 |
| Reconstruct/HGEMM threshold 128 | 1,387 | 1,407 | 1,366 |
| Selected slim deterministic serving stack | 1,059 | 1,280 | 1,265 |

The selected stack trades some peak prefill performance for a larger KV pool and
keeps the production NVFP4 DS-MLA KV format. The 8K selected-stack measurement
showed additional cold-start variance; raw samples are retained.

## KLD against reference logits

The repeated runs use a 2,048-token WikiText window (2,047 scored positions).
Lower is better.

| Variant | KV cache | Runs | Mean KLD | Sample SD |
|---|---|---:|---:|---:|
| Stock/baseline stack | NVFP4 DS-MLA | 5 | 0.206443 | 0.009406 |
| Full-clean exclusions | NVFP4 DS-MLA | 5 | 0.177819 | 0.003421 |
| Full quality stack, AG + deterministic | NVFP4 DS-MLA | 5 | **0.168030** | 0.005749 |
| Full quality stack | FP8 | 5 | 0.140098 | 0.003100 |
| Selected slim deterministic capacity stack | NVFP4 DS-MLA | 1 | 0.198088 | — |

The full-quality NVFP4 stack has the best measured NVFP4 KLD, but its exclusions
reduce the DCP4 KV pool to 740,608 tokens. The selected production image uses the
slimmer quality stack to retain 923,136 tokens. KLD is not affected by MTP depth;
MTP2/3/5 only alter speculative serving.

## Overnight accuracy suites

The selected MTP3/0.964 server is evaluated with four concurrent requests:

| Suite | Requested scope | Status/result |
|---|---:|---|
| LAVD | 10 runs | running |
| Estonia | 10 runs | queued |
| GPQA-Diamond | all 198 questions | queued |
| GSM8K | all 1,319 questions | queued |

This table is replaced with final results before publication.
