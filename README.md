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

### GPQA Diamond results (8x NVIDIA B200)

All four planned independent stochastic passes are fully validated and
published. Each pass scores the complete official 198-item GPQA Diamond split.

| Pass | Correct | Wrong | Accuracy | API errors | Status |
|---|---:|---:|---:|---:|---|
| 1 | 181 | 17 | **91.41%** | 0 | validated |
| 2 | 178 | 20 | **89.90%** | 0 | validated |
| 3 | 179 | 19 | **90.40%** | 0 | validated |
| 4 | 179 | 19 | **90.40%** | 0 | validated |

Protocol: `llm-decode-bench` 0.4.29, fixed concurrency 64,
`max_tokens=100000`, temperature 1.0, top-p 0.95, deterministic per-item option
shuffle, and exact option-letter scoring. Pass 1 is a canonical merge of a
deterministic 64-item shard and its exact 134-item complement; their item-ID
sets are disjoint and their union is exactly all 198 items.

The four-pass mean is **90.53%**: 717 correct across 792 repeated stochastic
generations of the same 198-item set, with a per-pass range of 89.90%-91.41%
and sample SD 0.64 percentage points. Across all passes there were 75 wrong
answers, seven truncated/no-answer generations, 31 generations reaching the
100K cap, zero API errors, and 20,043,342 completion tokens. The pooled 792
count is a generation-level summary, not 792 independent benchmark questions.

Raw JSON, terminal captures, the pass-1 merge script, and checksums are under
[`evaluations/b200_8gpu_20260718/gpqa`](evaluations/b200_8gpu_20260718/gpqa).

### IFBench results (8x NVIDIA B200)

The official 300-prompt IFBench evaluation scored **76.67% prompt-level loose**
(230/300), the primary metric reported by the IFBench paper, and **73.67%
prompt-level strict** (221/300). Instruction-level accuracy was 78.78% loose
and 75.87% strict.

Protocol: the official IFBench test set and scorer at commit
`1091c4c3de6c1f6ed12c012ed68f11ea450b0117`, temperature 0, seed 0,
`max_tokens=32768`, and deterministic even/odd sharding across the two B200
endpoints. All 300 unique prompts produced retained HTTP-success outcomes with
zero final API errors. Nine responses reached the token cap without final
answer content; they were retained as empty answers and scored as failures.

Raw responses, transformed scorer input, per-example strict/loose results,
logs, protocol metadata, and checksums are under
[`evaluations/b200_8gpu_20260718/ifbench/official_20260719`](evaluations/b200_8gpu_20260718/ifbench/official_20260719).

### Aider Polyglot results (8x NVIDIA B200)

The finalized portion of the official 225-task Aider Polyglot pass@2 run scored
**43.89% pass@1** (97/221) and **85.52% pass@2** (189/221). Coverage is
221/225 tasks (98.22%). Four tasks entered repeated client-timeout/backoff loops
and the run was called by user decision: `go/connect`, `go/robot-simulator`,
`java/rational-numbers`, and `java/zipper`. These four are
**infrastructure-incomplete, excluded from the denominator, and not counted as
incorrect solutions**.

| Language | Finalized | Pass@1 | Pass@2 |
|---|---:|---:|---:|
| C++ | 26 | 9/26 (34.62%) | 23/26 (88.46%) |
| Go | 37 | 21/37 (56.76%) | 31/37 (83.78%) |
| Java | 45 | 18/45 (40.00%) | 39/45 (86.67%) |
| JavaScript | 49 | 20/49 (40.82%) | 42/49 (85.71%) |
| Python | 34 | 15/34 (44.12%) | 31/34 (91.18%) |
| Rust | 30 | 14/30 (46.67%) | 23/30 (76.67%) |

Protocol: official Aider commit `5dc9490`, Polyglot benchmark commit `7e0611e`,
whole-file edit format, two attempts per task, eight threads per four-GPU
endpoint, `max_tokens=32768`, and temperature disabled. The complete 120 MB raw
archive is published on Hugging Face; the GitHub evaluation directory contains
the validated per-task JSONL/CSV, summary, exact configuration, logs, and a
checksum manifest for the full archive.

Artifacts are under
[`evaluations/b200_8gpu_20260718/aider`](evaluations/b200_8gpu_20260718/aider).

#### Published-model comparison

The four validated passes average **90.53%** (717 correct across four repeats of
the same 198-item set; pass range 89.90%-91.41%). The other rows below are
reported by their linked model cards. This is useful context, not a controlled
A/B: sampling parameters align for the NVIDIA/madeby561 rows, but checkpoint,
serving stack, prompt formatting, and harness details may differ.

| Model/build | GPQA Diamond | Aider Polyglot pass@2 | SciCode | IFBench | AA-LCR | τ²-Bench Telecom |
|---|---:|---:|---:|---:|---:|---:|
| NVIDIA GLM-5.2 FP8 baseline | 89.52 | — | 49.85 | 74.95 | 69.38 | 97.9 |
| NVIDIA full NVFP4 | 89.39 | — | 49.04 | 75.81 | 70.13 | 98.25 |
| **GLM-5.2-NVFP4-TR3-Hybrid (this model)** | **90.53** (four-pass mean) | **85.52†** | *pending* | **76.67** | *pending* | *pending* |
| madeby561 MXFP8/NVFP4/NF3 Hybrid v3.6 | 88.89 | — | *pending* | *pending* | *pending* | *pending* |
| madeby561 previous build `718f3f7472ec` | 88.38 | — | *pending* | *pending* | *pending* | *pending* |
| REAP-594B prune (contrast) | 86.87 | — | 47.77 | - | - | - |

† Aider score uses the 221 finalized-task denominator; four infrastructure-incomplete
tasks are excluded as described above.

Sources: [NVIDIA GLM-5.2-NVFP4](https://huggingface.co/nvidia/GLM-5.2-NVFP4),
[madeby561 hybrid v3.6](https://huggingface.co/madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid),
and [its previous revision](https://huggingface.co/madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid/tree/718f3f7472ec).
For additional, separately reported context, the
[official Z.ai GLM-5.2 card](https://huggingface.co/zai-org/GLM-5.2) lists
GPQA-Diamond at 91.2; its broader evaluation table is not treated here as a
protocol-matched baseline.

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
