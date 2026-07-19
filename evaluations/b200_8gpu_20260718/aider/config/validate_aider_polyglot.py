#!/usr/bin/env python3
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


INCOMPLETE = [
    {"language": "go", "task": "connect", "reason": "infrastructure/client timeout loop"},
    {"language": "go", "task": "robot-simulator", "reason": "infrastructure/client timeout loop"},
    {"language": "java", "task": "rational-numbers", "reason": "infrastructure/client timeout loop"},
    {"language": "java", "task": "zipper", "reason": "infrastructure/client timeout loop"},
]


def pct(n, d):
    return round(100.0 * n / d, 4) if d else 0.0


def collect(root, shard):
    rows = []
    for path in sorted(Path(root).glob("**/.aider.results.json")):
        data = json.loads(path.read_text())
        parts = Path(data["testdir"]).parts
        try:
            language = parts[parts.index("exercises") - 1]
        except (ValueError, IndexError):
            language = path.relative_to(root).parts[0]
        outcomes = list(data.get("tests_outcomes") or [])
        row = {
            "id": f"{language}/{data['testcase']}",
            "shard": shard,
            "language": language,
            "task": data["testcase"],
            "pass_1": bool(outcomes[0]) if outcomes else False,
            "pass_2": any(outcomes[:2]),
            "tests_outcomes": outcomes,
            "duration_seconds": data.get("duration"),
            "prompt_tokens": data.get("prompt_tokens"),
            "completion_tokens": data.get("completion_tokens"),
            "test_timeouts": data.get("test_timeouts", 0),
            "error_outputs": data.get("num_error_outputs", 0),
            "malformed_responses": data.get("num_malformed_responses", 0),
            "exhausted_context_windows": data.get("num_exhausted_context_windows", 0),
            "syntax_errors": data.get("syntax_errors", 0),
            "indentation_errors": data.get("indentation_errors", 0),
            "lazy_comments": data.get("lazy_comments", 0),
            "model": data.get("model"),
            "edit_format": data.get("edit_format"),
            "aider_commit": data.get("commit_hash"),
            "result_path": str(path.relative_to(root)),
        }
        rows.append(row)
    return rows


def aggregate(rows):
    n = len(rows)
    p1 = sum(r["pass_1"] for r in rows)
    p2 = sum(r["pass_2"] for r in rows)
    return {"finalized": n, "pass_1": p1, "pass_1_percent": pct(p1, n),
            "pass_2": p2, "pass_2_percent": pct(p2, n)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-a", required=True)
    ap.add_argument("--shard-b", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    rows = collect(args.shard_a, "A") + collect(args.shard_b, "B")
    ids = [r["id"] for r in rows]
    duplicates = sorted(k for k, v in Counter(ids).items() if v > 1)
    if len(rows) != 221 or duplicates:
        raise SystemExit(f"validation failed: finalized={len(rows)}, duplicates={duplicates}")

    by_language = defaultdict(list)
    by_shard = defaultdict(list)
    for row in rows:
        by_language[row["language"]].append(row)
        by_shard[row["shard"]].append(row)

    categories = {}
    for key in ["test_timeouts", "error_outputs", "malformed_responses",
                "exhausted_context_windows", "syntax_errors", "indentation_errors", "lazy_comments"]:
        categories[key] = {"affected_tasks": sum(bool(r[key]) for r in rows),
                           "total_events": sum((r[key] or 0) for r in rows)}
    categories["failed_both_attempts"] = sum(not r["pass_2"] for r in rows)
    categories["passed_on_second_attempt"] = sum((not r["pass_1"]) and r["pass_2"] for r in rows)

    summary = {
        "benchmark": "Aider Polyglot",
        "requested_tasks": 225,
        "finalized_tasks": len(rows),
        "coverage_percent": pct(len(rows), 225),
        "score_denominator": "221 finalized tasks; four infrastructure-incomplete tasks excluded, not counted incorrect",
        "overall": aggregate(rows),
        "by_shard": {k: aggregate(v) for k, v in sorted(by_shard.items())},
        "by_language": {k: aggregate(v) for k, v in sorted(by_language.items())},
        "failure_categories": categories,
        "infrastructure_incomplete": INCOMPLETE,
        "unique_finalized_ids": len(set(ids)),
        "duplicate_ids": duplicates,
        "protocol": {
            "model": sorted(set(r["model"] for r in rows)),
            "edit_format": sorted(set(r["edit_format"] for r in rows)),
            "aider_commit": sorted(set(r["aider_commit"] for r in rows)),
            "polyglot_commit": "7e0611e",
            "max_tokens": 32768,
            "temperature": "disabled",
            "threads_per_shard": 8,
            "endpoint_shard_a": "localhost:19300 -> Vast:9300 (GPUs 0-3)",
            "endpoint_shard_b": "localhost:19400 -> Vast:9400 (GPUs 4-7)",
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "results.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    with (out / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "shard", "language", "task", "pass_1", "pass_2",
            "duration_seconds", "prompt_tokens", "completion_tokens", "test_timeouts", "error_outputs",
            "malformed_responses", "exhausted_context_windows", "syntax_errors", "indentation_errors",
            "lazy_comments", "model", "edit_format", "aider_commit", "result_path"])
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in writer.fieldnames})

    lines = [
        "# Aider Polyglot — GLM-5.2 B200 evaluation",
        "",
        "This run was called after 221 of 225 tasks finalized. Four tasks entered repeated client-timeout/backoff loops and were stopped by user decision. They are infrastructure-incomplete and are excluded from the score denominator; they are not counted as incorrect solutions.",
        "",
        "| Scope | Finalized | Pass 1 | Pass 2 |",
        "|---|---:|---:|---:|",
    ]
    def add(label, a):
        lines.append(f"| {label} | {a['finalized']} | {a['pass_1']}/{a['finalized']} ({a['pass_1_percent']:.4f}%) | {a['pass_2']}/{a['finalized']} ({a['pass_2_percent']:.4f}%) |")
    add("Overall", summary["overall"])
    for k, v in summary["by_shard"].items(): add(f"Shard {k}", v)
    for k, v in summary["by_language"].items(): add(k, v)
    lines += ["", "## Infrastructure-incomplete (excluded)", ""]
    lines += [f"- `{x['language']}/{x['task']}` — {x['reason']}" for x in INCOMPLETE]
    lines += ["", "## Reproducibility", "", "See `summary.json`, `results.jsonl`, `results.csv`, `config/`, `logs/`, `raw/`, and `SHA256SUMS`.", ""]
    (out / "RESULTS.md").write_text("\n".join(lines))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
