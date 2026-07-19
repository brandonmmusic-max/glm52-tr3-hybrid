# Aider Polyglot — GLM-5.2 B200 evaluation

This run was called after 221 of 225 tasks finalized. Four tasks entered repeated client-timeout/backoff loops and were stopped by user decision. They are infrastructure-incomplete and are excluded from the score denominator; they are not counted as incorrect solutions.

| Scope | Finalized | Pass 1 | Pass 2 |
|---|---:|---:|---:|
| Overall | 221 | 97/221 (43.8914%) | 189/221 (85.5204%) |
| Shard A | 108 | 48/108 (44.4444%) | 93/108 (86.1111%) |
| Shard B | 113 | 49/113 (43.3628%) | 96/113 (84.9558%) |
| cpp | 26 | 9/26 (34.6154%) | 23/26 (88.4615%) |
| go | 37 | 21/37 (56.7568%) | 31/37 (83.7838%) |
| java | 45 | 18/45 (40.0000%) | 39/45 (86.6667%) |
| javascript | 49 | 20/49 (40.8163%) | 42/49 (85.7143%) |
| python | 34 | 15/34 (44.1176%) | 31/34 (91.1765%) |
| rust | 30 | 14/30 (46.6667%) | 23/30 (76.6667%) |

## Infrastructure-incomplete (excluded)

- `go/connect` — infrastructure/client timeout loop
- `go/robot-simulator` — infrastructure/client timeout loop
- `java/rational-numbers` — infrastructure/client timeout loop
- `java/zipper` — infrastructure/client timeout loop

## Reproducibility

See `summary.json`, `results.jsonl`, `results.csv`, `config/`, `logs/`, `raw/`, and `SHA256SUMS`.
