# Official IFBench 300

## Result

| Metric | Correct | Total | Score |
|---|---:|---:|---:|
| Prompt-level strict | 221 | 300 | 73.67% |
| Prompt-level loose | 230 | 300 | 76.67% |
| Instruction-level strict | — | — | 75.87% |
| Instruction-level loose | — | — | 78.78% |

## Protocol

- Official `allenai/IFBench` test set: 300 unique prompts.
- Official scorer repository commit: `1091c4c3de6c1f6ed12c012ed68f11ea450b0117`.
- Model: `brandonmusic/GLM-5.2-NVFP4-TR3-Hybrid`, served by two TP4/DCP4 vLLM endpoints on 8 NVIDIA B200 GPUs.
- Deterministic even/odd prompt sharding across endpoints.
- Temperature 0, seed 0, maximum 32,768 completion tokens, request timeout 7,200 seconds.
- 64 generation workers per endpoint; infrastructure retries only.

## Validation and disclosure

- Requested, retained, and scored: 300/300; unique keys: 300; final API errors: 0.
- Total completion tokens: 2,072,396; total HTTP attempts including infrastructure retries: 376.
- Finish reasons: 291 `stop`, 9 `length`.
- Nine HTTP-success outcomes exhausted the completion budget without final `message.content`. They are retained as empty answers and scored as failures, rather than being retried as infrastructure errors.
- The reported comparison metric used by the IFBench paper is prompt-level loose accuracy: **76.67%**.

Raw responses, transformed official-scorer input, per-example strict/loose outputs, generation logs, and metadata are included alongside this file.
