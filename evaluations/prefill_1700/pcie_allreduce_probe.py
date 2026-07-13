#!/usr/bin/env python3
"""Measure the deployed 4-GPU PCIe DMA allreduce against NCCL."""

from __future__ import annotations

import json
import os
import statistics

import torch
import torch.distributed as dist

from b12x.distributed import PCIeDmaAllReduce


def max_rank(value: float, device: torch.device) -> float:
    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def bench(fn, device: torch.device, *, warmup: int = 8, iters: int = 30,
          samples: int = 5) -> float:
    stream = torch.cuda.current_stream(device)
    for _ in range(warmup):
        fn()
    stream.synchronize()
    values = []
    for _ in range(samples):
        dist.barrier(device_ids=[device.index])
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        for _ in range(iters):
            fn()
        end.record(stream)
        end.synchronize()
        values.append(max_rank(start.elapsed_time(end) * 1000.0 / iters, device))
    return float(statistics.median(values))


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    hidden = 6144
    rows_list = [64, 128, 256, 512, 1024, 2048, 4096]
    max_bytes = max(rows_list) * hidden * 2
    results = []

    for mode in ("nccl", "bf16", "ag", "ring"):
        dma = None
        if mode != "nccl":
            dma = PCIeDmaAllReduce(
                exchange_group=dist.group.WORLD,
                device=device,
                max_bytes=max_bytes,
                fp8="" if mode == "bf16" else mode,
            )
            dma.min_bytes = 0
        for rows in rows_list:
            generator = torch.Generator(device=device)
            generator.manual_seed(20260713 + rows * 17 + rank)
            inp = torch.randn(
                (rows, hidden), generator=generator,
                dtype=torch.bfloat16, device=device,
            ) * 0.05
            reference = inp.clone()
            dist.all_reduce(reference)
            out = torch.empty_like(inp)
            if mode == "nccl":
                work = inp.clone()

                def fn() -> None:
                    work.copy_(inp)
                    dist.all_reduce(work)

                result_tensor = work
            else:
                assert dma is not None

                def fn() -> None:
                    dma.all_reduce(inp, out=out)

                result_tensor = out
            latency_us = bench(fn, device)
            fn()
            torch.cuda.synchronize(device)
            rel = float(
                (result_tensor.float() - reference.float()).norm()
                / reference.float().norm().clamp_min(1e-20)
            )
            logical_bytes = inp.numel() * inp.element_size()
            alg_gbps = logical_bytes / (latency_us * 1e-6) / 1e9
            bus_gbps = alg_gbps * 2.0 * (world - 1) / world
            item = {
                "mode": mode,
                "rows": rows,
                "bytes": logical_bytes,
                "latency_us": latency_us,
                "logical_alg_gbps": alg_gbps,
                "logical_bus_gbps": bus_gbps,
                "relative_l2": rel,
            }
            results.append(item)
            if rank == 0:
                print(json.dumps(item), flush=True)
            del inp, out, reference
        if dma is not None:
            dist.barrier(device_ids=[device.index])
            dma.close()
            dist.barrier(device_ids=[device.index])

    if rank == 0:
        path = os.environ.get("OUTPUT", "/results/pcie_allreduce_probe.json")
        with open(path, "w") as f:
            json.dump({"world_size": world, "hidden": hidden, "results": results}, f, indent=2)
            f.write("\n")
    dist.barrier(device_ids=[device.index])
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
