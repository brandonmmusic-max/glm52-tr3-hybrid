---
language:
- en
license: mit
base_model: lukealonso/GLM-5.2-NVFP4
tags:
- glm
- moe
- quantized
- nvfp4
- exl3
- trellis
- vllm
- text-generation-inference
pipeline_tag: text-generation
library_name: vllm
---

# GLM-5.2-NVFP4-TR3-Hybrid

A two-tier expert-quantized build of **GLM-5.2** (753B-class MoE, 78 transformer
layers + MTP head, 256 routed experts/layer, top-8 routing, 1,048,576-token
native context):

| tier | what | format | source |
|---|---|---|---|
| hot experts | 64 experts/MoE layer (layers 3-77) | **NVFP4** (ModelOpt, group-16) | lifted **byte-exact** from [lukealonso/GLM-5.2-NVFP4](https://huggingface.co/lukealonso/GLM-5.2-NVFP4) |
| tail experts | the other 192 experts/layer | **EXL3-lineage trellis @ exactly 3.0 bpw** (mcg codebook `0xCBAC1FED`, dual-Hadamard) | re-encoded from the same NVFP4 source with calibrated LDLQ (below) |
| MTP layer (78) | all 256 experts | NVFP4 | byte-exact carry |
| non-experts (attention/MLA, dense FFN 0-2, shared experts, router, embeddings) | BF16 on disk | served as **MXFP8** (e8m0/32 online requant) by the reference stack | byte-exact carry |

Per-layer expert tiering lives in [`tier_bitmap.json`](./tier_bitmap.json)
(`keep_nvfp4` = the 64 NVFP4 expert ids per layer + the measured per-expert
relative trellis RT-MSE that drove the split). The machine-readable production
record is `config.json:hybrid_tr3_tail`.

## The TR3 quantization process

The tail encode is the **verbatim exllamav3 v0.0.43 calibrated math**
(MIT, (c) turboderp-org; `exl3_lib/quantize.py` vendored unmodified):
Hadamard-128 incoherence pre-conditioning (`suh`/`svh` sign vectors), a
per-expert Hessian, block LDL decomposition, and LDLQ quantization in 16-row
blocks with error feedback, plus the golden-section `g_scale` search per slice.

**Calibration is real activations, not identity-H:**

- A capture pass ran the source model (TP=4, hooks on every MoE layer) over the
  owner's 4-axis calibration corpus: **1,773 samples, 1,049,589 tokens per
  layer** (seed `20260711`, 4,096-token truncation, axis balance preserved);
  corpus sha256 `cf247acc7c5da9f0600c7d6ab3b7c2fcfc54ec30b794e3b6047559285fa44df4`.
  The full sample-level record ships in-repo:
  [`calibration_manifest.json`](./calibration_manifest.json).
- Routing was recomputed in-worker with the model's own gate (exact
  `top8(sigmoid(x W_g^T) + e_score_correction_bias)` semantics), so each expert's
  Hessian `H_e = X_e^T X_e` uses exactly the tokens routed to it.
- gate/up share one H per expert (same input); **down projections use the
  diagonal block of `I_e^T I_e`** where `I_e = silu(X_e W_g^T) * (X_e W_u^T)`
  from the dequantized gate/up - slice-local LDLQ, consistent with
  slice-before-encode.
- Cold-expert fallbacks available in the pipeline were **never triggered**:
  `layer_h_fallback_experts_total = 0`, `q_fallback_slices_total = 0`
  (see `config.json:hybrid_tr3_tail.calibration`).

**Hot-64 selection:** per layer, the 64 experts with the highest pooled
relative trellis RT-MSE *measured under this calibrated encode* stay NVFP4
(the experts that trellis-3.0 would hurt most keep the crisper format);
the distribution is in `tier_bitmap.json`.

**TP4 slice-before-encode (why tensors are `rank{r}`-suffixed):** Hadamard
blocks break naive K-sharding, so sharding was decided *before* encoding.
gate/up are N-sliced (rank r owns output rows `[512r, 512r+512)` of
`[2048, 6144]`, full-K encode); down is K-sliced (rank r owns input columns
`[512r, 512r+512)` of `[6144, 2048]`) and **each rank's slice got its own
Hadamard + trellis encode**. Tensor schema per (layer L in 3-77, tail expert E,
proj, rank r):

```
model.layers.{L}.mlp.experts.{E}.{proj}.rank{r}.trellis  int16 [K/16, N/16, 48]
model.layers.{L}.mlp.experts.{E}.{proj}.rank{r}.suh      float16 [K]
model.layers.{L}.mlp.experts.{E}.{proj}.rank{r}.svh      float16 [N]
model.layers.{L}.mlp.experts.{E}.{proj}.rank{r}.mcg      int32 = 0xCBAC1FED
```

Forward semantics: `y = had_r_128( had_r_128(x * suh) @ W_dec ) * svh`.
**TP = 4 is therefore required at serve time** (the tail is physically
pre-sliced four ways).

**Bit-exactness chain:** every trellis slice passed pack/unpack index equality
and kernel-`reconstruct` bit-equality at encode; every carried (non-tail)
tensor was verified byte-exact against the source during assembly; and
[`MANIFEST.sha256`](./MANIFEST.sha256) covers every file in this repo - verify
your download with `sha256sum -c MANIFEST.sha256` before first boot.

## Size anatomy

| item | value |
|---|---|
| tensor payload on disk | **327.2 GiB (351.3 GB)**, 81 shards, 796,289 tensors |
| tail encode rate | exactly 3.0 bpw trellis + f16 `suh`/`svh` Hadamard vectors |
| non-expert tier | BF16 on disk (largest single block of the remainder) |
| VRAM at TP4 (reference stack) | **~78 GiB/GPU** (non-experts served MXFP8; design target, first-boot verified against the load log) |
| KV budget target | **~1M-token pool** with nvfp4 KV at `gpu_memory_utilization 0.968`, `max_model_len 1,048,576` on 4x 96GB |

Target hardware: **4x 96GB SM120** (RTX PRO 6000 Blackwell class), PCIe Gen5,
no NVLink required.

## Serving

**Stock vLLM cannot run this checkpoint** - upstream has no trellis kernels and
no loader for the `rank{r}`-sliced exl3 tensors. Use the reference image, which
bakes: the two-tier hybrid loader (`exl3_tr3` tier tag), fused GPU route
packing, tile-balanced trellis compute, and a prefill-only hybrid crossover:
tail experts with at least 128 routed rows are reconstructed once to FP16 and
run through tensor-core HGEMM, while smaller experts use the fused trellis
kernel. Decode remains on the original FULL-CUDA-graph path. The image also
includes MTP-3 speculative decoding and nvfp4 KV cache.

```bash
docker pull verdictai/vllm-glm52-tr3-hybrid:v2@sha256:f1325d3558012577d58fe52f5fdf2a1c7d9fcbc17dc9725f896c1832a0fa5c1e
```

Exact single-node command for 4x RTX PRO 6000 Blackwell (model downloaded at
`/home/brandonmusic/models/GLM-5.2-NVFP4-TR3-Hybrid`):

```bash
docker run -d --name glm52-tr3-9300 \
  --gpus all --runtime nvidia --network host --ipc host --shm-size 64g --init \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
  -e TR3_MODEL=/models/GLM-5.2-NVFP4-TR3-Hybrid \
  -e TR3_PORT=9300 -e TR3_TP=4 -e TR3_DCP=4 -e TR3_DCP_ILV=1 \
  -e TR3_KV=nvfp4_ds_mla -e TR3_MTP_N=3 \
  -e TR3_UTIL=0.96 -e TR3_MAXLEN=262144 \
  -e TR3_MAX_SEQS=4 -e TR3_MAX_BATCHED=2048 \
  -e TR3_PREFIX_CACHE=0 -e HYBRID_TR3_CHUNK=128 \
  -e HYBRID_TR3_DQ_THRESHOLD=128 \
  -v /home/brandonmusic/models:/models:ro \
  -v /home/brandonmusic/.cache/glm52-tr3:/cache \
  verdictai/vllm-glm52-tr3-hybrid:v2

docker logs -f glm52-tr3-9300
```

Cold-prefill measurements on 4x RTX PRO 6000, TP4/DCP4, prefix caching off:

| Path | 8k | 16k | 64k |
|---|---:|---:|---:|
| original Python chunk-128 | ~800 tok/s | — | — |
| fused route packing + tile balancing | 1,329 tok/s | 1,329 tok/s | 1,270 tok/s |
| fused + reconstruct/HGEMM threshold 128 | **1,387 tok/s** | **1,407 tok/s** | **1,366 tok/s** |

Quickstart (compose): download this repo to `./models/GLM-5.2-NVFP4-TR3-Hybrid`,
save the compose file below, then `docker compose up`. First boot verifies
every file against `MANIFEST.sha256` (one-time, ~351 GB of hashing), loads
~78 GiB/GPU, and captures CUDA graphs - expect a long first start; the OpenAI
endpoint then appears on `:8000` (`curl localhost:8000/v1/models`). Defaults:
TP4, DCP4, `nvfp4_ds_mla` KV, MTP-3, util 0.96, 262k `max_model_len` - every
knob is a `TR3_*` env override documented inline.

```yaml
services:
  glm52-tr3:
    image: verdictai/vllm-glm52-tr3-hybrid:v2
    network_mode: host
    ipc: host
    shm_size: 64g
    init: true
    ulimits: { memlock: -1, stack: 67108864 }
    volumes:
      - ./models:/models
      - tr3-jit-cache:/cache
    environment:
      TR3_MODEL: /models/GLM-5.2-NVFP4-TR3-Hybrid
      TR3_PORT: "8000"
      TR3_TP: "4"            # REQUIRED: tail tensors are pre-sliced rank0..3
      TR3_DCP: "4"           # decode context parallel (1 to disable)
      TR3_KV: nvfp4_ds_mla   # or fp8_ds_mla
      TR3_MTP_N: "3"         # MTP-3 default lane
      TR3_UTIL: "0.96"
      TR3_MAXLEN: "262144"
      TR3_MAX_SEQS: "4"
      TR3_MAX_BATCHED: "2048"
      TR3_DCP_ILV: "1"
      TR3_PREFIX_CACHE: "0"
      HYBRID_TR3_CHUNK: "128"
      HYBRID_TR3_DQ_THRESHOLD: "128"
      TR3_VERIFY_MANIFEST: auto
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 4
              capabilities: [gpu]
    healthcheck:
      test: ["CMD-SHELL", "curl -sf http://127.0.0.1:8000/v1/models || exit 1"]
      interval: 30s
      timeout: 10s
      retries: 20
      start_period: 3600s
    restart: unless-stopped
volumes:
  tr3-jit-cache:
```

An alternate DSpark speculator lane exists behind `TR3_SPEC_JSON` (see the
image entrypoint); MTP-3 is the default and needs no extra files.

## Provenance and license

- Base model: GLM-5.2 by Z.AI / Zhipu AI - **MIT** (LICENSE carried in-repo).
- NVFP4 source: [lukealonso/GLM-5.2-NVFP4](https://huggingface.co/lukealonso/GLM-5.2-NVFP4)
  (ModelOpt NVFP4); hot tier and all carried tensors are byte-exact from it.
- Tail quantizer math: [exllamav3](https://github.com/turboderp-org/exllamav3)
  v0.0.43 (MIT, (c) turboderp-org); serving uses the same project's CUDA
  kernels (`exl3_moe`).
- Producer: `encode_tr3.py` (capture -> calibrated LDLQ encode -> assemble ->
  upload; resumable, hash-audited at every step).
