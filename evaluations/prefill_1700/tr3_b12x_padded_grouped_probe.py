#!/usr/bin/env python3
"""Probe padded grouped MXFP8 versus per-expert HGEMM on real TR3 weights.

This deliberately times the full per-request path: Hadamard input transform,
trellis reconstruction, layout copy, dynamic MXFP8 quantization, and GEMM.
The installed b12x grouped dense kernel has a uniform M across groups, so cases
include both uniform and ragged token counts padded to a 128-row boundary.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from safetensors import safe_open

import exllamav3_ext as ext
from b12x.gemm.dense import dense_gemm
from b12x.gemm.mxfp8_quant_cute import quantize_mxfp8_rows_cute
from b12x.gemm.wo_projection import (
    MXFP8Rows,
    empty_dense_gemm_mnl_view,
    empty_mxfp8_rows_for_dense_gemm,
)


def load_gate_experts(
    model: Path, layer: int, experts: list[int], rank: int, device: str
) -> list[dict[str, torch.Tensor]]:
    path = model / f"model-layer-{layer:03d}.safetensors"
    loaded: list[dict[str, torch.Tensor]] = []
    with safe_open(str(path), framework="pt", device="cpu") as f:
        for expert in experts:
            prefix = f"model.layers.{layer}.mlp.experts.{expert}.gate_proj.rank{rank}"
            loaded.append(
                {
                    field: f.get_tensor(f"{prefix}.{field}").to(device).contiguous()
                    for field in ("trellis", "suh", "svh")
                }
            )
    return loaded


def available_gate_experts(model: Path, layer: int, rank: int) -> list[int]:
    path = model / f"model-layer-{layer:03d}.safetensors"
    suffix = f".gate_proj.rank{rank}.trellis"
    with safe_open(str(path), framework="pt", device="cpu") as f:
        return sorted(
            {
                int(key.split(".experts.", 1)[1].split(".", 1)[0])
                for key in f.keys()
                if key.endswith(suffix) and ".experts." in key
            }
        )


def time_ms(fn, *, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end) / repeats)


def grouped_view(q: MXFP8Rows, *, groups: int, rows: int, width: int) -> MXFP8Rows:
    if rows % 128:
        raise ValueError("group rows must be a multiple of 128")
    values = q.values.view(groups, rows, width).permute(1, 2, 0)
    scale_rows = q.scale_rows.view(groups, rows, width // 32)
    m_tiles = rows // 128
    k_tiles = math.ceil((width // 32) / 4)
    # Total-row group-1 and uniform-M grouped physical scale layouts are
    # byte-equivalent when each group owns a whole number of 128-row tiles.
    scale_mma = (
        q.scale_mma[..., 0]
        .view(32, 4, groups, m_tiles, 4, k_tiles)
        .permute(0, 1, 3, 4, 5, 2)
    )
    return MXFP8Rows(values=values, scale_rows=scale_rows, scale_mma=scale_mma)


def rel_l2(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    r = reference.float()
    d = candidate.float() - r
    return float(torch.linalg.vector_norm(d) / torch.linalg.vector_norm(r).clamp_min(1e-20))


def run_case(
    tensors: list[dict[str, torch.Tensor]],
    counts: list[int],
    *,
    hidden: int,
    intermediate: int,
    warmup: int,
    repeats: int,
) -> dict[str, object]:
    device = tensors[0]["trellis"].device
    groups = len(counts)
    padded_m = math.ceil(max(counts) / 128) * 128

    sources = [
        (torch.randn((count, hidden), device=device, dtype=torch.float16) * 0.05)
        for count in counts
    ]
    baseline_a = [torch.empty_like(source) for source in sources]
    baseline_w = [
        torch.empty((hidden, intermediate), device=device, dtype=torch.float16)
        for _ in counts
    ]
    baseline_out = [
        torch.empty((count, intermediate), device=device, dtype=torch.float16)
        for count in counts
    ]

    def baseline():
        for g, count in enumerate(counts):
            del count
            ext.had_r_128(sources[g], baseline_a[g], tensors[g]["suh"], None, 1.0)
            ext.reconstruct(baseline_w[g], tensors[g]["trellis"], 3, True, False)
            ext.hgemm(baseline_a[g], baseline_w[g], baseline_out[g])

    a_phys = torch.zeros(
        (groups, padded_m, hidden), device=device, dtype=torch.float16
    )
    w_phys = torch.empty(
        (groups, intermediate, hidden), device=device, dtype=torch.float16
    )
    recon_tmp = torch.empty(
        (hidden, intermediate), device=device, dtype=torch.float16
    )
    a_q_total = empty_mxfp8_rows_for_dense_gemm(
        groups * padded_m, hidden, device=device
    )
    w_q_total = empty_mxfp8_rows_for_dense_gemm(
        groups * intermediate, hidden, device=device
    )
    a_q = grouped_view(a_q_total, groups=groups, rows=padded_m, width=hidden)
    w_q = grouped_view(
        w_q_total, groups=groups, rows=intermediate, width=hidden
    )
    grouped_out = empty_dense_gemm_mnl_view(
        padded_m, intermediate, groups, device=device, dtype=torch.float16
    )

    def grouped():
        a_phys.zero_()
        for g, count in enumerate(counts):
            ext.had_r_128(
                sources[g], a_phys[g, :count], tensors[g]["suh"], None, 1.0
            )
            ext.reconstruct(recon_tmp, tensors[g]["trellis"], 3, True, False)
            w_phys[g].copy_(recon_tmp.T)
        quantize_mxfp8_rows_cute(
            a_phys.view(groups * padded_m, hidden),
            a_q_total.values,
            a_q_total.scale_rows,
            a_q_total.scale_mma,
        )
        quantize_mxfp8_rows_cute(
            w_phys.view(groups * intermediate, hidden),
            w_q_total.values,
            w_q_total.scale_rows,
            w_q_total.scale_mma,
        )
        dense_gemm(
            (a_q.values, a_q.scale_mma),
            (w_q.values, w_q.scale_mma),
            out=grouped_out,
            ab_dtype="float8_e4m3fn",
            sf_dtype="float8_e8m0fnu",
            c_dtype="float16",
            sf_vec_size=32,
            expected_m=padded_m,
        )

    baseline()
    grouped()
    torch.cuda.synchronize()
    errors = [
        rel_l2(baseline_out[g], grouped_out[:count, :, g])
        for g, count in enumerate(counts)
    ]
    baseline_ms = time_ms(baseline, warmup=warmup, repeats=repeats)
    grouped_ms = time_ms(grouped, warmup=warmup, repeats=repeats)
    result = {
        "groups": groups,
        "counts": counts,
        "padded_m": padded_m,
        "padding_ratio": (groups * padded_m) / sum(counts),
        "baseline_ms": baseline_ms,
        "grouped_mxfp8_ms": grouped_ms,
        "speedup": baseline_ms / grouped_ms,
        "relative_l2_each": errors,
        "max_relative_l2": max(errors),
    }

    if hasattr(ext, "hgemm_grouped_ptrs"):
        grouped_fp16_out = [
            torch.empty((count, intermediate), device=device, dtype=torch.float16)
            for count in counts
        ]
        a_ptrs = torch.tensor(
            [tensor.data_ptr() for tensor in baseline_a],
            device=device,
            dtype=torch.int64,
        )
        b_ptrs = torch.tensor(
            [tensor.data_ptr() for tensor in baseline_w],
            device=device,
            dtype=torch.int64,
        )
        c_ptrs = torch.tensor(
            [tensor.data_ptr() for tensor in grouped_fp16_out],
            device=device,
            dtype=torch.int64,
        )
        rows_cpu = torch.tensor(counts, device="cpu", dtype=torch.int32)

        def grouped_fp16():
            for g in range(groups):
                ext.had_r_128(
                    sources[g], baseline_a[g], tensors[g]["suh"], None, 1.0
                )
                ext.reconstruct(
                    baseline_w[g], tensors[g]["trellis"], 3, True, False
                )
            ext.hgemm_grouped_ptrs(
                a_ptrs, b_ptrs, c_ptrs, rows_cpu, hidden, intermediate
            )

        grouped_fp16()
        torch.cuda.synchronize()
        fp16_errors = [
            rel_l2(baseline_out[g], grouped_fp16_out[g]) for g in range(groups)
        ]
        grouped_fp16_ms = time_ms(
            grouped_fp16, warmup=warmup, repeats=repeats
        )
        result.update(
            {
                "grouped_fp16_ms": grouped_fp16_ms,
                "grouped_fp16_speedup": baseline_ms / grouped_fp16_ms,
                "grouped_fp16_relative_l2_each": fp16_errors,
                "grouped_fp16_max_relative_l2": max(fp16_errors),
            }
        )
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    torch.manual_seed(20260713)
    experts = available_gate_experts(args.model, args.layer, args.rank)[:8]
    if len(experts) < 8:
        raise RuntimeError(f"layer {args.layer} exposes only {len(experts)} TR3 experts")
    tensors = load_gate_experts(args.model, args.layer, experts, args.rank, args.device)
    hidden = int(tensors[0]["suh"].numel())
    intermediate = int(tensors[0]["svh"].numel())
    cases = [
        [128],
        [128, 128],
        [128, 128, 128, 128],
        [128] * 8,
        [256, 224, 192, 128],
        [256, 224, 192, 160, 128, 128, 128, 128],
    ]
    results = []
    for counts in cases:
        result = run_case(
            tensors[: len(counts)],
            counts,
            hidden=hidden,
            intermediate=intermediate,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        print(json.dumps(result), flush=True)
        results.append(result)
        torch.cuda.empty_cache()
    report = {
        "layer": args.layer,
        "rank": args.rank,
        "hidden": hidden,
        "intermediate": intermediate,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
