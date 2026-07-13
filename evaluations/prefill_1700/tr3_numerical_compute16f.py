#!/usr/bin/env python3
"""Compare EXL3 fused trellis MoE with reconstruct-once/HGEMM on real TR3 weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open

import exllamav3_ext as ext


PROJS = ("gate_proj", "up_proj", "down_proj")
FIELDS = ("trellis", "suh", "svh")


def load_expert(model: Path, layer: int, expert: int, rank: int, device: str):
    path = model / f"model-layer-{layer:03d}.safetensors"
    tensors = {}
    with safe_open(str(path), framework="pt", device="cpu") as f:
        for proj in PROJS:
            for field in FIELDS:
                key = (
                    f"model.layers.{layer}.mlp.experts.{expert}."
                    f"{proj}.rank{rank}.{field}"
                )
                tensors[(proj, field)] = f.get_tensor(key).to(device).contiguous()
    return tensors


def ptr(t: torch.Tensor):
    return torch.tensor([t.data_ptr()], dtype=torch.int64, device=t.device)


def run_fused(x: torch.Tensor, tensors, concurrency: int):
    m, hidden = x.shape
    intermediate = tensors[("gate_proj", "svh")].numel()
    # exl3_moe keeps one trailing sentinel count bucket for routes that map
    # outside the local expert pointer table.
    counts = torch.tensor([m, 0], dtype=torch.int64, device=x.device)
    tokens = torch.arange(m, dtype=torch.int64, device=x.device)
    weights = torch.ones(m, dtype=torch.float16, device=x.device)
    out = torch.zeros((m, hidden), dtype=torch.float32, device=x.device)
    tg = torch.empty((concurrency, m, hidden), dtype=torch.float16, device=x.device)
    tu = torch.empty_like(tg)
    ig = torch.empty(
        (concurrency, m, intermediate), dtype=torch.float16, device=x.device
    )
    iu = torch.empty_like(ig)
    args = []
    for proj in PROJS:
        for field in FIELDS:
            args.append(ptr(tensors[(proj, field)]))
    ext.exl3_moe(
        x,
        out,
        counts,
        tokens,
        weights,
        tg,
        tu,
        ig,
        iu,
        0,
        3,
        3,
        3,
        *args,
        True,
        False,
        True,
        False,
        True,
        False,
        0.0,
    )
    return out


def run_reconstruct(x: torch.Tensor, tensors):
    m, hidden = x.shape
    intermediate = tensors[("gate_proj", "svh")].numel()
    yh_g = torch.empty_like(x)
    yh_u = torch.empty_like(x)
    ig = torch.empty((m, intermediate), dtype=torch.float16, device=x.device)
    iu = torch.empty_like(ig)
    ia = torch.empty_like(ig)
    out = torch.empty((m, hidden), dtype=torch.float32, device=x.device)
    recon_up = torch.empty(
        (hidden, intermediate), dtype=torch.float16, device=x.device
    )
    recon_down = torch.empty(
        (intermediate, hidden), dtype=torch.float16, device=x.device
    )

    sg = {field: tensors[("gate_proj", field)] for field in FIELDS}
    su = {field: tensors[("up_proj", field)] for field in FIELDS}
    sd = {field: tensors[("down_proj", field)] for field in FIELDS}

    ext.had_r_128(x, yh_g, sg["suh"], None, 1.0)
    ext.reconstruct(recon_up, sg["trellis"], 3, True, False)
    ext.hgemm(yh_g, recon_up, ig)
    ext.had_r_128(x, yh_u, su["suh"], None, 1.0)
    ext.reconstruct(recon_up, su["trellis"], 3, True, False)
    ext.hgemm(yh_u, recon_up, iu)
    ext.had_r_128(ig, ig, None, sg["svh"], 1.0)
    ext.had_r_128(iu, iu, None, su["svh"], 1.0)
    ext.silu_mul(ig, iu, ia, 0.0)
    ext.had_r_128(ia, ia, sd["suh"], None, 1.0)
    ext.reconstruct(recon_down, sd["trellis"], 3, True, False)
    ext.hgemm(ia, recon_down, out)
    ext.had_r_128(out, out, None, sd["svh"], 1.0)
    return out


def metrics(reference: torch.Tensor, candidate: torch.Tensor):
    r = reference.float()
    c = candidate.float()
    d = c - r
    rnorm = torch.linalg.vector_norm(r)
    dnorm = torch.linalg.vector_norm(d)
    return {
        "relative_l2": float((dnorm / rnorm.clamp_min(1e-20)).item()),
        "relative_rms": float(
            (d.square().mean().sqrt() / r.square().mean().sqrt().clamp_min(1e-20)).item()
        ),
        "max_abs": float(d.abs().max().item()),
        "reference_max_abs": float(r.abs().max().item()),
        "cosine": float(
            torch.nn.functional.cosine_similarity(r.flatten(), c.flatten(), dim=0).item()
        ),
        "finite": bool(torch.isfinite(r).all() and torch.isfinite(c).all()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=0.01)
    args = ap.parse_args()

    torch.manual_seed(20260712)
    device_index = torch.device(args.device).index or 0
    max_concurrency = ext.exl3_moe_max_concurrency(device_index)
    # One expert is active in each unit case, so a single worker group is
    # sufficient and avoids allocating max-concurrency scratch at m=2048.
    concurrency = 1
    cases = [
        (3, 0, 1, 0.05),
        (3, 0, 8, 0.05),
        (3, 0, 33, 0.05),
        (3, 0, 128, 0.05),
        (3, 0, 2048, 0.05),
        (40, 0, 128, 0.05),
        (77, 0, 128, 0.05),
        (3, 0, 128, 0.50),
    ]
    results = []
    loaded = None
    loaded_key = None
    for layer, expert, rows, amplitude in cases:
        key = (layer, expert)
        if loaded_key != key:
            del loaded
            torch.cuda.empty_cache()
            loaded = load_expert(args.model, layer, expert, 0, args.device)
            loaded_key = key
        x = (torch.randn((rows, 6144), device=args.device) * amplitude).half()
        fused = run_fused(x, loaded, concurrency)
        reconstructed = run_reconstruct(x, loaded)
        torch.cuda.synchronize()
        result = {
            "layer": layer,
            "expert": expert,
            "rank": 0,
            "rows": rows,
            "input_amplitude": amplitude,
            **metrics(fused, reconstructed),
        }
        result["pass"] = result["finite"] and result["relative_l2"] < args.threshold
        print(json.dumps(result), flush=True)
        results.append(result)
        del x, fused, reconstructed

    report = {
        "threshold": args.threshold,
        "concurrency": concurrency,
        "max_concurrency": max_concurrency,
        "cases": results,
        "max_relative_l2": max(r["relative_l2"] for r in results),
        "passed": all(r["pass"] for r in results),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"summary": report}, indent=2), flush=True)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
