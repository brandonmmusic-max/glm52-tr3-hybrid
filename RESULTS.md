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

## GPQA Diamond on 8x NVIDIA B200 (interim)

Three of four planned independent passes are complete and validated. These runs
used two unchanged TP4/DCP4 endpoints, each backed by four B200 GPUs, while the
passes ran concurrently across the eight-GPU host.

| Pass | Correct | Wrong | Accuracy | Wilson 95% CI | Hit 100K cap | Truncated/no answer | Errors |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 181 | 17 | **91.41%** | 86.68%-94.57% | 9 | 1 | 0 |
| 2 | 178 | 20 | **89.90%** | 84.91%-93.37% | 9 | 3 | 0 |
| 4 | 179 | 19 | **90.40%** | 85.50%-93.77% | 5 | 0 | 0 |

Both artifacts report requested = attempted = scored = 198 and
`interrupted=false`. Truncated/no-answer items are included as wrong, not
dropped. The protocol was `llm-decode-bench` 0.4.29, official GPQA Diamond
198, fixed concurrency 64, `max_tokens=100000`, temperature 1.0, top-p 0.95,
and exact option-letter scoring after deterministic per-item option shuffling.
The dataset hash is
`a8472c5a82ea2df8f209c17713aba1a6d409120c609ec0582dae0cb940c7e28c`.

Pass 1 was executed as a deterministic 64-item shard followed by its exact
134-item complement. The two item-ID/index sets were validated disjoint, with
zero omissions or duplicates and an exact 198-item union, then merged in
canonical item-ID order. Both source shards, the merge script, and the canonical
result are retained. Pass 3 is still running; IFBench and Aider Polyglot
will follow, so no four-pass aggregate is claimed yet.

### External comparison context

Across the three completed passes, this model has 538 correct responses over
three repeats of the same 198-item set: a **90.57% interim mean**, with individual
passes at 91.41%, 89.90%, and 90.40%. This is 1.05 percentage points above the
89.52 FP8 baseline reported on NVIDIA's GLM-5.2-NVFP4 card, 1.18 points above
NVIDIA's 89.39 full-NVFP4 result, and 1.68 points above the 88.89 madeby561 hybrid v3.6
result. These differences are descriptive, not causal: the published cards
align on temperature 1.0 and top-p 0.95, and NVIDIA specifies a 100,000-token
GPQA output cap, but the serving stacks, prompt construction, and harnesses are
not proven byte-for-byte identical.

The official Z.ai GLM-5.2 card separately reports 91.2 on GPQA-Diamond. Because
that figure appears in a broader evaluation table with a different reporting
context, it is retained as general base-model context rather than a matched
baseline. The final four-pass aggregate will replace the interim mean after
passes 3-4 validate.

Sources: [NVIDIA GLM-5.2-NVFP4](https://huggingface.co/nvidia/GLM-5.2-NVFP4),
[madeby561 GLM-5.2-MXFP8-NVFP4-NF3-Hybrid](https://huggingface.co/madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid),
and [official Z.ai GLM-5.2](https://huggingface.co/zai-org/GLM-5.2).
