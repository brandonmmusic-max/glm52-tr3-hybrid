# GLM-5.2 NVFP4/TR3 Hybrid: reproducible serving and evaluation

This repository is the public engineering record for
[`brandonmusic/GLM-5.2-NVFP4-TR3-Hybrid`](https://huggingface.co/brandonmusic/GLM-5.2-NVFP4-TR3-Hybrid).
It contains the production Docker build, serving command, raw benchmark output,
KLD logs, failed experiments, container manifests, and hardware provenance.

The selected production configuration is **TP4 + DCP4/A2A + MTP3**, with
`nvfp4_ds_mla` KV cache, four maximum sequences, and GPU utilization `0.964` on
four RTX PRO 6000 Blackwell GPUs. The measured KV pool is **923,136 tokens**.

## Published image

```bash
docker pull verdictai/glm52-tr3-hybrid:mtp3-dcp4-nvfp4-20260713@sha256:863f01a3cbdef0d0d03c3c871e90131568d914b6c66bf9b250efc18e944fbf46
```

The image bakes in the loader, EXL3 extension, compatibility shim, quality flags,
MTP3 command, and A2A serving configuration that were previously host bind
mounts. Artifact hashes are recorded in
[`evidence/image/overlay-sha256.txt`](evidence/image/overlay-sha256.txt).

## Run

Download the Hugging Face model into
`/home/brandonmusic/models/GLM-5.2-NVFP4-TR3-Hybrid`, then run:

```bash
./scripts/run_server.sh
docker logs -f glm52-tr3-9300
```

The expanded `docker run` command is in
[`scripts/run_server.sh`](scripts/run_server.sh). The default endpoint is the
OpenAI-compatible server on port 9300.

## Selected decode result

Single-user sustained decode, 30 seconds per cell, 4,096-token cap:

| MTP depth | 0 context | 32K | 128K | KV pool |
|---|---:|---:|---:|---:|
| MTP2 | 60.3 tok/s | 58.4 tok/s | 54.5 tok/s | 922,880 |
| **MTP3** | **62.5 tok/s** | **63.2 tok/s** | **62.6 tok/s** | **923,136** |
| MTP5 | 50.0 tok/s | 43.3 tok/s | 42.2 tok/s | 923,136 |

MTP3 won at every measured context. MTP2 accepted a larger fraction of drafts
but its shorter verification window lost throughput; MTP5's extra verification
cost outweighed its accepted tokens.

## Memory-utilization boundary

Utilization `0.970` booted with a 989,184-token KV pool, but left only 107 MiB
free on GPU1. A four-concurrent 128K stress test triggered a real 64 MiB CUDA
allocation failure in the sparse-attention indexer, killed the engine, and
returned four request errors. The failed result and full server log are retained
under [`evaluations/decode`](evaluations/decode) and [`evidence/containers`](evidence/containers).

## Evaluation record

### Interim GPQA Diamond results (8x NVIDIA B200)

Two of four planned independent stochastic passes are fully validated and
published. Each pass scores the complete official 198-item GPQA Diamond split.

| Pass | Correct | Wrong | Accuracy | API errors | Status |
|---|---:|---:|---:|---:|---|
| 1 | 181 | 17 | **91.41%** | 0 | validated |
| 2 | 178 | 20 | **89.90%** | 0 | validated |

Protocol: `llm-decode-bench` 0.4.29, fixed concurrency 64,
`max_tokens=100000`, temperature 1.0, top-p 0.95, deterministic per-item option
shuffle, and exact option-letter scoring. Pass 1 is a canonical merge of a
deterministic 64-item shard and its exact 134-item complement; their item-ID
sets are disjoint and their union is exactly all 198 items. Passes 3-4,
IFBench, and Aider Polyglot remain in progress, so these are deliberately
labeled interim rather than a final aggregate.

Raw JSON, terminal captures, the pass-1 merge script, and checksums are under
[`evaluations/b200_8gpu_20260718/gpqa`](evaluations/b200_8gpu_20260718/gpqa).

- [`RESULTS.md`](RESULTS.md): summarized decode, prefill, KLD, DeepSpark, and
  capacity results.
- [`evaluations/`](evaluations): raw JSON, CSV, and KLD logs for all retained
  variants.
- [`evidence/`](evidence): container inspections, full logs, image manifest,
  hardware topology, and exact hashes.
- [`scripts/run_evaluations.sh`](scripts/run_evaluations.sh): LAVD, Estonia,
  GPQA-Diamond, and GSM8K runner with concurrency 4, Rich TUI capture, validation,
  and retries.
- [`METHODOLOGY.md`](METHODOLOGY.md): hardware, test controls, selection rules,
  and known limitations.

No failed result was removed. DeepSpark, MTP2, MTP5, and utilization 0.970 are
included alongside the winner so the selection can be independently audited.

## Upstream and license

The model and serving components retain their upstream licenses. GLM-5.2 and the
Hugging Face model repository are MIT-licensed; EXL3/exllamav3 is MIT-licensed.
GPQA data is CC BY 4.0 and is fetched by the benchmark rather than committed here;
GSM8K is sourced from the MIT-licensed grade-school-math repository.
