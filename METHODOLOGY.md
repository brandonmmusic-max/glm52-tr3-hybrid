# Methodology

## Hardware and topology

- Four NVIDIA RTX PRO 6000 Blackwell 96 GB GPUs, PCIe Gen5 x16.
- TP4 and DCP4 on one host; no NVLink.
- NVIDIA driver 595.58.03 and CUDA 13.2 image stack.
- Peer-to-peer overrides were active and verified before testing.
- Full topology and driver reports are retained under `evidence/system/`.

## Selected serving controls

- Model: `GLM-5.2-NVFP4-TR3-Hybrid`.
- 64 hot NVFP4 experts and 192 EXL3 TR3 tail experts per MoE layer.
- Main KV cache: `nvfp4_ds_mla`.
- Tensor parallel 4, decode context parallel 4, A2A communication.
- MTP3 probabilistic speculative decoding.
- Maximum sequences: 4.
- Maximum batched tokens: 2,048.
- Maximum CUDA graph capture size: 8; FULL_AND_PIECEWISE graphs.
- GPU memory utilization: 0.964.
- Prefix caching enabled for the production server.

## Selection rule

Candidate changes were isolated whenever possible. Decode depths 2, 3, and 5
used the same model, DCP topology, KV format, utilization, graph settings, and
benchmark cells. The selected version had to win or remain competitive at 0,
32K, and 128K while preserving the target KV pool and surviving sustained load.

Utilization 0.970 was not selected merely because it booted. It was subjected to
four concurrent 128K requests and failed with a documented CUDA OOM. The proven
0.964 configuration was restored before accuracy evaluation.

## Accuracy suites

- `llm-decode-bench` v0.4.28, SHA256 recorded in `evidence/system/`.
- Fixed concurrency 4.
- LAVD: 10 runs, built-in 167-row context consistency profile.
- Estonia: 10 runs, built-in long-context profile.
- GPQA-Diamond: all 198 questions, deterministic option shuffling; dataset hash
  `a8472c5a82ea2df8f209c17713aba1a6d409120c609ec0582dae0cb940c7e28c`.
- GSM8K: all 1,319 test questions; dataset hash
  `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`.
- GPQA and GSM8K temperature: 0, as defined by the built-in profiles.
- Every suite must finish the requested item count with zero request errors.
- Each run produces JSON, a raw Rich terminal typescript, and an ANSI-clean text
  rendering. The runner retries an incomplete suite up to three times.

## Limitations

- KLD uses a single 2,048-token window per run; repeated variants use five runs,
  while the final slim capacity stack currently has one exact run.
- The selected model is specialized for TP4 because tail tensors are rank-sliced.
- Results are specific to this four-GPU PCIe topology and software build.
- Rich TUI hardware PCIe readings are coarse `nvidia-smi` diagnostics, not an
  Nsight Systems communication profile.
