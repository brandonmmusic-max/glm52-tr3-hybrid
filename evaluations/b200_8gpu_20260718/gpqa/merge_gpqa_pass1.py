#!/usr/bin/env python3
"""Merge the disclosed 64+134 GPQA pass-1 shards into one canonical report."""

from __future__ import annotations

import copy
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

BASE = Path("/workspace/glm52-eval")
RESULTS = BASE / "results"
BENCH = BASE / "llm-inference-bench"
SHARD = RESULTS / "gpqa_diamond_100k_t1_p095_c64_probe_a.json"
COMPLEMENT = RESULTS / "gpqa_diamond_100k_t1_p095_c64_pass_1_complement134.json"
DATASET = Path("/root/.cache/llm_decode_bench/datasets/gpqa_diamond.jsonl")
OUTPUT = RESULTS / "gpqa_diamond_100k_t1_p095_c64_pass_1.json"
TUI = RESULTS / "gpqa_diamond_100k_t1_p095_c64_pass_1.tui"


def load(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def require_protocol(report: dict, requested: int) -> None:
    meta = report["metadata"]
    acc = report["accuracy"]
    expected = {
        "version": "0.4.29",
        "max_tokens": 100000,
        "fixed_concurrency": 64,
        "temperature": 1.0,
        "top_p": 0.95,
        "requested_runs": requested,
        "dataset_sha256": "a8472c5a82ea2df8f209c17713aba1a6d409120c609ec0582dae0cb940c7e28c",
    }
    for key, value in expected.items():
        assert meta.get(key) == value, (key, meta.get(key), value)
    assert meta.get("interrupted") is False
    assert acc.get("attempted") == requested
    assert acc.get("scored") == requested
    assert acc.get("errors") == 0
    assert len(report["runs"]) == requested


def main() -> None:
    sys.path.insert(0, str(BENCH))
    import llm_decode_bench as bench  # type: ignore

    shard = load(SHARD)
    complement = load(COMPLEMENT)
    require_protocol(shard, 64)
    require_protocol(complement, 134)

    dataset_rows = [json.loads(line) for line in DATASET.read_text().splitlines() if line.strip()]
    canonical_ids = [f"gpqa-{row['record_id']}" for row in dataset_rows]
    assert len(canonical_ids) == len(set(canonical_ids)) == 198

    shard_ids = {r["item_id"] for r in shard["runs"]}
    complement_ids = {r["item_id"] for r in complement["runs"]}
    assert len(shard_ids) == 64
    assert len(complement_ids) == 134
    assert not (shard_ids & complement_ids), "pass-1 shards overlap"
    assert shard_ids | complement_ids == set(canonical_ids), "pass-1 shards do not cover GPQA Diamond"

    by_id: dict[str, dict] = {}
    for source_name, report in (("probe64", shard), ("complement134", complement)):
        for raw in report["runs"]:
            run = copy.deepcopy(raw)
            run["source_shard"] = source_name
            run["source_run_index"] = run["run_index"]
            by_id[run["item_id"]] = run

    runs: list[dict] = []
    for canonical_index, item_id in enumerate(canonical_ids, 1):
        run = by_id[item_id]
        run["run_index"] = canonical_index
        runs.append(run)

    dataclass_fields = set(bench.CompletionStatsRun.__dataclass_fields__)
    typed_runs = [bench.CompletionStatsRun(**{k: v for k, v in run.items() if k in dataclass_fields}) for run in runs]
    summary = bench.summarize_completion_stats_runs(typed_runs)
    scored = [run for run in runs if run.get("correct") is not None]
    correct = sum(run.get("correct") is True for run in scored)
    wilson_low, wilson_high = bench.wilson_interval(correct, len(scored))

    categories: dict[str, dict] = defaultdict(lambda: {"scored": 0, "correct": 0})
    for run in scored:
        category = run.get("category") or "Unknown"
        categories[category]["scored"] += 1
        categories[category]["correct"] += int(run.get("correct") is True)
    category_summaries = [
        {
            "category": category,
            "scored": values["scored"],
            "correct": values["correct"],
            "accuracy": values["correct"] / values["scored"],
        }
        for category, values in sorted(categories.items())
    ]

    report = copy.deepcopy(complement)
    report["metadata"].update(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "requested_runs": 198,
            "min_results": 198,
            "dataset_items_selected": 198,
            "interrupted": False,
            "merged_pass": True,
            "merge_strategy": "canonical item_id order; deterministic probe64 plus exact complement134",
            "source_artifacts": [SHARD.name, COMPLEMENT.name],
        }
    )
    report["selected_summary"] = summary
    report["all_summary"] = copy.deepcopy(summary)
    report["level_summaries"] = [
        {"concurrency": 64, "phase": "merged-pass-1", "summary": summary, "selected": True, "improved": True}
    ]
    report["accuracy"] = {
        "items_total": 198,
        "items_selected": 198,
        "attempted": 198,
        "scored": 198,
        "correct": correct,
        "accuracy": correct / 198,
        "wilson95_low": wilson_low,
        "wilson95_high": wilson_high,
        "unparseable": sum(
            run.get("correct") is False and run.get("score_detail") == "unparseable" for run in scored
        ),
        "truncated_no_answer": sum(run.get("score_label") == "truncated" for run in scored),
        "hit_max_tokens": sum(bool(run.get("hit_max_tokens")) for run in runs),
        "errors": sum(not run.get("ok") for run in runs),
        "dataset_exhausted": False,
    }
    report["category_summaries"] = category_summaries
    report["wrong_runs"] = [run for run in runs if run.get("correct") is False or not run.get("ok")]
    report["hardware_run_summary"] = {
        "merged_segments": [shard.get("hardware_run_summary", {}), complement.get("hardware_run_summary", {})],
        "note": "Two sequential c64 segments on the same unchanged TP4/DCP4 endpoint; source summaries preserved.",
    }
    report["runs"] = runs
    report.setdefault("methodology", {})["pass1_merge"] = {
        "disclosed": True,
        "probe_selection": "indices floor(i*198/64), i=0..63",
        "complement": "exact remaining 134 canonical indices",
        "overlap": 0,
        "union": 198,
        "canonical_order_restored": True,
    }

    assert report["accuracy"]["correct"] == 181
    assert report["accuracy"]["errors"] == 0
    assert len({run["item_id"] for run in report["runs"]}) == 198

    temp = OUTPUT.with_suffix(".json.tmp")
    temp.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    temp.replace(OUTPUT)

    from rich.console import Console
    from rich.table import Table

    with TUI.open("w") as handle:
        console = Console(file=handle, force_terminal=True, color_system="truecolor", width=120)
        console.rule("[bold cyan]GLM-5.2 GPQA Diamond — Canonical Pass 1[/bold cyan]")
        protocol = Table(title="Validated Protocol", show_header=False)
        for key, value in (
            ("Dataset", "Official GPQA Diamond — 198/198 unique questions"),
            ("Sampling", "temperature=1.0, top_p=0.95, max_tokens=100000"),
            ("Execution", "TP4/DCP4, FP8 KV, MTP3, fixed concurrency=64"),
            ("Merge", "64-item deterministic shard + exact 134-item complement"),
            ("Coverage", "overlap=0, missing=0, duplicates=0, canonical order restored"),
            ("Errors", "0; interrupted=false"),
        ):
            protocol.add_row(key, value)
        console.print(protocol)

        results = Table(title="Accuracy")
        results.add_column("Completed", justify="right")
        results.add_column("Correct", justify="right", style="green")
        results.add_column("Wrong", justify="right", style="red")
        results.add_column("Accuracy", justify="right", style="bold cyan")
        results.add_column("Wilson 95%", justify="right")
        results.add_column("Hit 100k", justify="right")
        results.add_column("Truncated/no answer", justify="right")
        results.add_row(
            "198/198",
            str(correct),
            str(198 - correct),
            f"{correct / 198:.2%}",
            f"{wilson_low:.2%}–{wilson_high:.2%}",
            str(report["accuracy"]["hit_max_tokens"]),
            str(report["accuracy"]["truncated_no_answer"]),
        )
        console.print(results)

        cats = Table(title="Category Results")
        cats.add_column("Category")
        cats.add_column("Correct", justify="right")
        cats.add_column("Scored", justify="right")
        cats.add_column("Accuracy", justify="right")
        for row in category_summaries:
            cats.add_row(row["category"], str(row["correct"]), str(row["scored"]), f"{row['accuracy']:.2%}")
        console.print(cats)
        console.print("[dim]Raw source JSON/TUI artifacts are preserved separately for full disclosure and auditability.[/dim]")

    print(json.dumps({
        "output": str(OUTPUT),
        "tui": str(TUI),
        "correct": correct,
        "wrong": 198 - correct,
        "accuracy": correct / 198,
        "overlap": 0,
        "missing": 0,
        "duplicates": 0,
        "errors": 0,
        "interrupted": False,
    }))


if __name__ == "__main__":
    main()
