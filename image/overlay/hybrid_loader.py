"""
hybrid_loader.py — Path A: TP4-native NVFP4/NF3 hybrid loader for vLLM (modelopt path).
Deploy via .pth (`import hybrid_loader`). Fires in every worker.

GLM-5.2 hybrid checkpoint: routed experts are per-layer mixed — top-K (=64) NVFP4 (crisp) +
the rest NF3 (3-bit, group-32). Non-experts bf16 (excluded). Stock modelopt would allocate
uniform NVFP4 for all 256 experts (~420 GiB → OOM on 384). This loader allocates COMPACT
two-group storage (~326 GiB) and does a reference two-pass forward.

Interceptions (armed by an import hook on vllm...modelopt):
  1. ModelOptNvFp4Config.from_config -> stash `hybrid_bit_map` (stock strips unknown keys).
  2. FusedMoEMethodCls -> HybridNvFp4MoE.

NO mapping patch: stock RoutedExperts.make_expert_params_mapping is PREFIX-based, so
`...gate_proj.weight_packed` -> `...routed_experts.w13_weight_packed`, `...weight_scale` ->
`...w13_weight_scale`, etc. We register compact params under those exact names; the per-layer
weight_loader demuxes NVFP4 vs NF3 by the expert's group (this layer's remap) and TP-shards.

Facts baked in (verified against the eldritch image):
  * apply(layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input): routing is
    upstream (sigmoid/noaux_tc/norm/2.5x already applied); shared experts run by the runner
    separately -> apply returns ROUTED-ONLY output.
  * moe_kernel stays None for hybrid layers -> is_monolithic False (via experts_cls) -> apply
    dispatched; maybe_make_prepare_finalize overridden to None so MK-init doesn't raise.
  * on-disk NVFP4 block scales are LINEAR (2D) -> dequantize_to_dtype(..., swizzle=False).
  * TP=4, moe_intermediate=2048 -> clean chunk, no padding. gate/up shard dim0, down shard dim1.

TR3 EXTENSION (HYBRID_EXL3_TR3=1, default OFF -> byte-identical baseline behavior):
  Serves GLM-5.2-NVFP4-TR3-Hybrid: per MoE layer (3..77) 64 kept experts are NVFP4
  (identical schema to the NF3 hybrid's kept tier) and the 192-tail is EXL3
  trellis 3.0 bpw, PRE-SLICED per TP rank at encode time:
      model.layers.{L}.mlp.experts.{E}.{proj}.rank{r}.{trellis|suh|svh|mcg}
      gate/up: K=6144 N=512 (N-sliced), down: K=512 N=6144 (K-sliced)
  There is NO hybrid_bit_map in this checkpoint; tiering comes from
  config.json:hybrid_tr3_tail + tier_bitmap.json ("keep_nvfp4" per layer).
  Loading: a stream filter CONSUMES the exl3 tensors before vLLM's
  load_weights ever sees them (their mapped names would KeyError) and copies
  ONLY this rank's rank{tp_rank} slices into preallocated per-layer slabs.
  Forward: kept-64 -> the validated b12x W4A16 "packed" object A unchanged;
  tail-192 -> ONE ext.exl3_moe fused launch per <=HYBRID_TR3_CHUNK-token chunk
  (gate+up+silu+down+weighted accumulate), pointer-table driven, mcg codebook.
  Routing tensors are built sync-free (sort + scatter_add_; torch.bincount is
  FORBIDDEN here - it D2H-syncs and breaks FULL cudagraph capture).
  Requires exllamav3_ext (0.0.43 cu132) importable in-container; on the
  torch 2.12 image the getCurrentCUDABlasHandle ABI shim is ctypes-loaded
  RTLD_GLOBAL by _ensure_exl3_ext() BEFORE the ext import (LD_PRELOAD does
  NOT work: torch libs are RTLD_LOCAL). Tail build failure = HARD ABORT
  (there is no NF3/reference fallback for tr3 tensors; fail closed).
"""
import os, sys, importlib.abc, importlib.util

_NF3_VALS = [-1.0, -0.6047, -0.3563, -0.1275, 0.1275, 0.3563, 0.6047, 1.0]
_HYBRID_DEBUG = os.environ.get("HYBRID_DEBUG") not in (None, "", "0")  # eager-only: per-tier norms
# triton | b12x | b12x_nf3 | ref.  b12x_nf3 = BOTH tiers through the b12x W4A16
# CuteDSL kernel: kept-64 NVFP4 as weight_layout="packed", NF3-192 as the new
# weight_layout="nf3_2p1" (e4m3_k32 scales).  HYBRID_NF3 is ignored in that mode.
_HYBRID_KEPT = os.environ.get("HYBRID_KEPT", "triton")
_HYBRID_TIER = os.environ.get("HYBRID_TIER", "both")  # both|a|b (b12x_nf3 isolation)
_HYBRID_ACT_CAPTURE = os.environ.get("HYBRID_ACT_CAPTURE", "")  # dir: save MoE-layer inputs for GPTQ Hessians
_act_store = {"n": 0, "buf": {}, "flushed": {}}
_HYBRID_NF3 = os.environ.get("HYBRID_NF3", "fast")    # fast | ref  (NF3 3-bit tier)
_HYBRID_PROFILE = os.environ.get("HYBRID_PROFILE", "0") == "1"  # per-tier CUDA-event timing at prefill M>64
# TC-decode fast path for m<=8 (2026-07-05): compile a SECOND launch per tier at
# size_m=max_m (small-size_m compiles are the broken family) with block-8 direct
# top-k + fused sum. Measured standalone: 0.27ms vs prod 0.36-0.38ms/launch at
# decode m, numerics rel<0.012. Gated OFF by default; flag-off = today's bytes.
_HYBRID_TC_DECODE = os.environ.get("HYBRID_TC_DECODE", "0") == "1"
# ---- exl3_tr3 tier (default OFF; OFF = byte-identical baseline) ----
_HYBRID_EXL3_TR3 = os.environ.get("HYBRID_EXL3_TR3", "0") == "1"
_TR3_CHUNK = int(os.environ.get("HYBRID_TR3_CHUNK", "128"))  # rows/exl3_moe call AND temp rows/expert
_TR3_DQ_THRESHOLD = int(os.environ.get("HYBRID_TR3_DQ_THRESHOLD", "0"))
_HYBRID_QUALITY_STACK = os.environ.get("HYBRID_QUALITY_STACK", "0") == "1"

# Quality-stack ablation switches.  Defaults preserve the original all-on
# behavior when HYBRID_QUALITY_STACK=1, while allowing each hypothesis to be
# measured independently without maintaining multiple loader copies.
def _quality_flag(name):
    return os.environ.get(
        name, "1" if _HYBRID_QUALITY_STACK else "0") == "1"


_QUALITY_FP32_ROUTER = _quality_flag("HYBRID_QUALITY_FP32_ROUTER")
_QUALITY_DETERMINISTIC_ACCUM = _quality_flag(
    "HYBRID_QUALITY_DETERMINISTIC_ACCUM")
_QUALITY_EXCLUDE_KV_B = _quality_flag("HYBRID_QUALITY_EXCLUDE_KV_B")
_QUALITY_EXCLUDE_SHARED = _quality_flag("HYBRID_QUALITY_EXCLUDE_SHARED")
_QUALITY_EXCLUDE_EARLY_MLP = _quality_flag(
    "HYBRID_QUALITY_EXCLUDE_EARLY_MLP")
_QUALITY_EXCLUDE_FINAL_ATTN = _quality_flag(
    "HYBRID_QUALITY_EXCLUDE_FINAL_ATTN")
_TR3_MCG = 0xCBAC1FED
_TR3_PROJS = ("gate_proj", "up_proj", "down_proj")
_TR3 = {   # per-process registry (each TP worker owns one)
    "map": None,      # int L -> {"keep": set, "tail": [asc global ids], "pos": {gid: li}}
    "meta": None,     # config.json:hybrid_tr3_tail
    "slabs": {},      # (L, proj, field) -> preallocated CUDA slab
    "fill": {},       # L -> payload-copy counter (expect Km*3*3)
    "mcg_seen": {},   # L -> mcg-scalar verify counter (expect Km*3)
    "rt": None,       # shared runtime scratch (xh/out32/temps/flat_token)
}
_prof_store = {"n": 0, "a_ms": 0.0, "b_ms": 0.0, "rows": 0, "pend": []}
_dbg_first_build = [True]   # keep kept-tier originals for ONLY the first b12x layer (VRAM)
_HBM = None  # cached hybrid_bit_map (per-process); read once from the checkpoint config.json

# ---- b12x_nf3 backend constants / shared runtime ----
# Pinned CTA tiles (fc1_tile_k, fc1_tile_n, fc2_tile_k, fc2_tile_n): the NF3
# flat-span weight layout is packed for a SPECIFIC tile_n, but the kernel's
# auto tile selection is m-dependent (fc1_tile_n flips 128<->256 across m).
# (64,256,64,256) validated for our shapes (fc1 N=1024 K=6144 / fc2 N=6144
# K=512) at BOTH moe_block_size 8 (decode) and 64 (prefill), both scale
# formats: smem fits (45-76KB <= 100.9KB) and the SM121 reg-count table has
# entries for (256,{1,4},16,4). It is also exactly what auto-selection picks
# for the max-m prefill, so prefill throughput is the natural one.
_B12X_NF3_TILES = (64, 256, 64, 256)
_B12X_NF3_DECODE_M = 8          # <=8 -> preplanned TC-decode launch (fused topk sum)
_B12X_NF3_MAX_TOKENS = int(os.environ.get("HYBRID_B12X_MAX_TOKENS", "8192"))
_b12x_nf3_rt = {                # module-level, shared across ALL layers (one scratch set)
    "max_m": None,              # fixed at first apply: max(env, first-call m)
    "topk": None,
    "launches": {},             # (E, layout, scale_fmt, topk, max_m) -> (dec, pre)
    "buffers": None,            # W4A16PackedBuffers planned at max_m/route_E=256
    "out_a": None,              # [max_m, H] bf16 per-tier outputs (fully overwritten
    "out_b": None,              #  by every run_w4a16_moe call that uses them)
}


def _load_hbm(quant_config=None):
    """Reliable in-worker hybrid_bit_map: config-object attr, else read config.json off disk.
    (from_config stash on the config object does NOT survive pickling to TP workers.)"""
    global _HBM
    if _HBM is not None:
        return _HBM
    hbm = getattr(quant_config, "hybrid_bit_map", None) if quant_config is not None else None
    if hbm is None:
        try:
            import json, os
            from vllm.config import get_current_vllm_config
            mp = get_current_vllm_config().model_config.model
            cfgp = os.path.join(mp, "config.json")
            qc = json.load(open(cfgp)).get("quantization_config", {})
            hbm = qc.get("hybrid_bit_map")
            print(f"[hybrid_loader] hbm read from {cfgp}: {len(hbm) if hbm else 0} layers", flush=True)
        except Exception as e:
            print("[hybrid_loader] hbm config.json read failed:", e, flush=True)
    _HBM = hbm
    return hbm


def _tr3_model_dir():
    from vllm.config import get_current_vllm_config
    return get_current_vllm_config().model_config.model


_EXL3_EXT = [None]


def _ensure_exl3_ext():
    """Import exllamav3_ext, first publishing the torch-2.12 ABI shim.
    The 0.0.43 ext (built vs torch 2.11) needs at::cuda::getCurrentCUDABlasHandle();
    torch >= 2.12 only exports the (bool) overload. The shim (linked against
    libtorch_cuda.so) is ctypes-loaded RTLD_GLOBAL AFTER torch, so the ext's
    dlopen resolves the old symbol from the global scope and the shim resolves
    the new one from its own dependency scope. On torch 2.11 (exl3_venv) the
    shim is unnecessary and its absence/failure is non-fatal."""
    if _EXL3_EXT[0] is not None:
        return _EXL3_EXT[0]
    import ctypes
    import torch  # noqa: F401  (must be resident before shim + ext)
    for cand in (
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "libexl3_torch212_compat.so"),
        os.environ.get("HYBRID_TR3_SHIM", ""),
    ):
        if cand and os.path.exists(cand):
            try:
                ctypes.CDLL(cand, mode=ctypes.RTLD_GLOBAL)
                print(f"[hybrid_loader] tr3 ABI shim loaded: {cand}", flush=True)
                break
            except OSError as e:
                print(f"[hybrid_loader] tr3 ABI shim load failed ({cand}): {e} "
                      f"(fine on torch<=2.11)", flush=True)
    import exllamav3_ext
    _EXL3_EXT[0] = exllamav3_ext
    return exllamav3_ext


def _load_tr3():
    """tier map from the checkpoint dir: config.json:hybrid_tr3_tail + tier_bitmap.json.
    Fail-closed: HYBRID_EXL3_TR3=1 with a checkpoint that lacks either file is a boot abort."""
    if _TR3["map"] is not None:
        return _TR3["map"]
    import json
    mp = _tr3_model_dir()
    with open(os.path.join(mp, "config.json")) as f:
        cfg = json.load(f)
    meta = cfg.get("hybrid_tr3_tail")
    assert meta is not None, f"HYBRID_EXL3_TR3=1 but {mp}/config.json has no hybrid_tr3_tail block"
    assert meta.get("format") == "exl3-trellis" and int(meta.get("mcg_multiplier", 0)) == _TR3_MCG, \
        f"unexpected hybrid_tr3_tail: format={meta.get('format')} mcg={meta.get('mcg_multiplier')}"
    assert int(meta.get("tp", 0)) == 4, f"checkpoint encoded for tp={meta.get('tp')}, this loader requires TP4"
    with open(os.path.join(mp, "tier_bitmap.json")) as f:
        tb = json.load(f)
    E = int(meta["experts_per_layer"])
    m = {}
    for ls, ent in tb.items():
        L = int(ls)
        keep = set(int(e) for e in ent["keep_nvfp4"])
        assert len(keep) == int(meta["nvfp4_keep_per_layer"]), f"layer {L}: keep={len(keep)}"
        tail = [e for e in range(E) if e not in keep]     # ascending == encode/local order
        m[L] = {"keep": keep, "tail": tail, "pos": {g: i for i, g in enumerate(tail)}}
    _TR3["map"] = m
    _TR3["meta"] = meta
    print(f"[hybrid_loader] tr3 map loaded: {len(m)} MoE layers, "
          f"keep={meta['nvfp4_keep_per_layer']} tail={meta['tr3_tail_per_layer']} "
          f"bits={meta['bits']} hessian={meta.get('hessian')}", flush=True)
    return m


def _tr3_slab_shapes(Km):
    """(proj, field) -> slab shape. trellis [K/16, N/16, 48] int16, suh [K] f16, svh [N] f16.
    gate/up: K=6144, N=512 (per-rank N-slice); down: K=512, N=6144 (per-rank K-slice)."""
    sh = {}
    for proj, (K, N) in (("gate_proj", (6144, 512)), ("up_proj", (6144, 512)),
                         ("down_proj", (512, 6144))):
        sh[(proj, "trellis")] = (Km, K // 16, N // 16, 48)
        sh[(proj, "suh")] = (Km, K)
        sh[(proj, "svh")] = (Km, N)
    return sh


def _tr3_stream_filter(weights):
    """Consume model.layers.{L}.mlp.experts.{E}.{proj}.rank{r}.{field} tensors:
    copy this rank's slices into the preallocated slabs, drop other ranks',
    pass every other tensor through untouched. Runs in every TP worker."""
    import re, torch
    from vllm.distributed import get_tensor_model_parallel_rank
    tp_rank = get_tensor_model_parallel_rank()
    pat = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
                     r"(gate_proj|up_proj|down_proj)\.rank(\d)\.(trellis|suh|svh|mcg)$")
    tr3map = _load_tr3()
    n_copied = 0
    n_dropped = 0
    for name, t in weights:
        mm = pat.match(name)
        if mm is None:
            yield name, t
            continue
        L, E, proj, r, field = (int(mm.group(1)), int(mm.group(2)), mm.group(3),
                                int(mm.group(4)), mm.group(5))
        if r != tp_rank:
            n_dropped += 1
            del t
            continue
        ent = tr3map.get(L)
        assert ent is not None and E in ent["pos"], \
            f"exl3 tensor {name}: layer/expert not in tier_bitmap tail"
        li = ent["pos"][E]
        if field == "mcg":
            # stored as SIGNED int32 (0xCBAC1FED wraps negative) - mask to compare
            v = int(t.reshape(()).item()) & 0xFFFFFFFF
            assert v == _TR3_MCG, f"{name}: mcg={v:#x} != {_TR3_MCG:#x}"
            _TR3["mcg_seen"][L] = _TR3["mcg_seen"].get(L, 0) + 1
            continue
        slab = _TR3["slabs"].get((L, proj, field))
        assert slab is not None, f"{name}: slab missing (create_weights did not run for layer {L}?)"
        row = slab[li]
        assert t.numel() == row.numel(), f"{name}: numel {t.numel()} != slab row {row.numel()}"
        if field == "trellis":
            assert t.dtype == torch.int16, f"{name}: dtype {t.dtype}"
        else:
            assert t.dtype == torch.float16, f"{name}: dtype {t.dtype}"
        row.copy_(t.reshape(row.shape), non_blocking=True)
        _TR3["fill"][L] = _TR3["fill"].get(L, 0) + 1
        n_copied += 1
        del t
        if n_copied % 20000 == 0:
            print(f"[hybrid_loader] tr3 stream: {n_copied} slices copied "
                  f"(rank {tp_rank}), {n_dropped} foreign-rank dropped", flush=True)
    print(f"[hybrid_loader] tr3 stream DONE: {n_copied} slices copied (rank {tp_rank}), "
          f"{n_dropped} foreign-rank dropped, {sum(_TR3['mcg_seen'].values())} mcg verified", flush=True)


def _unpack_nf3(packed, scale, out_cols, blk=32):
    """packed:[R, out_cols//8*3] uint8, scale:[R, out_cols//blk] fp8 -> [R, out_cols] bf16."""
    import torch
    nf = torch.tensor(_NF3_VALS, device=packed.device, dtype=torch.float32)
    R = packed.shape[0]
    p = packed.reshape(R, out_cols // 8, 3).to(torch.int32)
    w24 = p[..., 0] | (p[..., 1] << 8) | (p[..., 2] << 16)
    codes = torch.stack([(w24 >> (3 * i)) & 0x7 for i in range(8)], -1).reshape(R, out_cols)
    return (nf[codes.long()] * scale.float().repeat_interleave(blk, 1)).to(torch.bfloat16)


def _fp8_ne_transform(weights):
    """Dequant fp8 non-expert weights -> bf16 on the fly. Speaks BOTH scale
    dialects: `.weight_scale_fp8` (rev-2, bf16 per-channel) and `.weight_scale`
    U8 (rev-3 MXFP8, e8m0 per-32 groups, [out, in//32]). Expert tensors never
    match (uint8-packed weights / F8 scales) and pass through untouched."""
    import torch
    scales, pend = {}, {}

    def _deq(w, s):
        kind, st = s
        if kind == "chan":
            return (w.to(torch.float32) * st.to(torch.float32).unsqueeze(1)).to(torch.bfloat16)
        sc = torch.pow(2.0, st.to(torch.float32) - 127.0)
        return (w.to(torch.float32) * sc.repeat_interleave(32, 1)).to(torch.bfloat16)

    def _stage(t):
        # fastsafetensors yields CUDA tensors; holding tier pairs in VRAM
        # mid-shard OOMs the load. CPU-stage ONLY the F8 tier (16GB of 327) —
        # experts keep streaming GPU-direct.
        return t.cpu() if getattr(t, "is_cuda", False) else t

    for name, t in weights:
        if name.endswith(".weight_scale_fp8"):
            wn = name[:-len(".weight_scale_fp8")] + ".weight"
            w = pend.pop(wn, None)
            if w is not None:
                yield wn, _deq(w, ("chan", _stage(t)))
            else:
                scales[wn] = ("chan", _stage(t))
        elif (name.endswith(".weight_scale")
              and getattr(t, "dtype", None) == torch.uint8 and t.dim() == 2):
            wn = name[:-len(".weight_scale")] + ".weight"
            w = pend.pop(wn, None)
            if w is not None:
                yield wn, _deq(w, ("mx", _stage(t)))
            else:
                scales[wn] = ("mx", _stage(t))
        elif name.endswith(".weight") and getattr(t, "dtype", None) == torch.float8_e4m3fn:
            s = scales.pop(name, None)
            if s is not None:
                yield name, _deq(_stage(t), s)
            else:
                pend[name] = _stage(t)
        else:
            yield name, t
    for wn, t in pend.items():   # unmatched fp8 weight (shouldn't happen) -> best-effort
        yield wn, t.to(torch.bfloat16)


def _deq_nv(w, s, g, gs, deq):
    """NVFP4 dequant (linear on-disk scales). g is [2] for fused w13 or scalar for w2."""
    import torch
    if g.numel() == 2:  # fused w13: gate rows use g[0], up rows use g[1]
        half = w.shape[0] // 2
        a = deq(w[:half], s[:half], g[0], torch.bfloat16, gs, swizzle=False)
        b = deq(w[half:], s[half:], g[1], torch.bfloat16, gs, swizzle=False)
        return torch.cat([a, b], 0)
    return deq(w, s, g.reshape(()), torch.bfloat16, gs, swizzle=False)


def _patch(mod):
    import re, torch
    import torch.nn.functional as F
    from vllm.model_executor.utils import set_weight_attrs
    from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import dequantize_to_dtype
    from vllm.distributed import (get_tensor_model_parallel_rank,
                                  get_tensor_model_parallel_world_size)
    Base = mod.ModelOptNvFp4FusedMoE
    Cfg = mod.ModelOptNvFp4Config

    # ---- 1. keep hybrid_bit_map alive through from_config ----
    _orig_fc = getattr(Cfg.from_config, "__func__", Cfg.from_config)
    def _from_config(cls, config):
        obj = _orig_fc(cls, config)
        hbm = None
        if isinstance(config, dict):
            hbm = config.get("hybrid_bit_map")
            if hbm is None and isinstance(config.get("quantization"), dict):
                hbm = config["quantization"].get("hybrid_bit_map")
        try:
            obj.hybrid_bit_map = hbm
            print(f"[hybrid_loader] hybrid_bit_map stashed: "
                  f"{len(hbm) if hbm else 0} layers", flush=True)
        except Exception as e:
            print("[hybrid_loader] stash failed:", e, flush=True)
        return obj
    Cfg.from_config = classmethod(_from_config)

    class HybridNvFp4MoE(Base):
        def maybe_make_prepare_finalize(self, *a, **k):
            return None  # we own the forward via apply(); no external prepare/finalize

        def _layer_idx(self, layer):
            pfx = getattr(layer, "prefix", None)
            if pfx is None:
                pfx = getattr(layer, "layer_name", "") or ""
            m = re.search(r"layers\.(\d+)\b", str(pfx))
            return int(m.group(1)) if m else None

        def _bits(self, layer):
            Lidx = self._layer_idx(layer)
            if _HYBRID_EXL3_TR3:
                # tr3 checkpoint: tier from tier_bitmap.json (no hybrid_bit_map on disk).
                # bits value 3 tags the exl3 tail (group 1); MTP/nextn layer absent
                # from the bitmap -> None -> uniform-NVFP4 all-kept path below.
                tr3map = _load_tr3()
                ent = tr3map.get(Lidx) if Lidx is not None else None
                if ent is None:
                    return None
                E = int(_TR3["meta"]["experts_per_layer"])
                return [4 if e in ent["keep"] else 3 for e in range(E)]
            hbm = _load_hbm(self.quant_config)
            b = hbm.get(str(Lidx)) if (hbm and Lidx is not None) else None
            if b is None:
                pfx = getattr(layer, "prefix", None) or getattr(layer, "layer_name", "") or ""
                print(f"[hybrid][MISS] pfx={pfx!r} hbm_len={len(hbm) if hbm else 0} "
                      f"idx={Lidx}", flush=True)
            return b

        def create_weights(self, layer, num_experts, hidden_size,
                           intermediate_size_per_partition, params_dtype, **extra):
            bits = self._bits(layer)
            if bits is None:
                # Non-hybrid MoE (the MTP/nextn layer) — its experts are uniform NVFP4.
                # Route it through OUR path as all-kept (bits=4) so it uses our Triton NVFP4
                # kernel + our weight_loader. (Falling to super() uses the stock MoE loader,
                # whose weight-scale quant_method check at routed_experts.py:906 rejects it
                # under our method -> the MTP boot crash.)
                bits = [4] * num_experts
            H, I = hidden_size, intermediate_size_per_partition
            gs = self.quant_config.group_size
            tp_rank = get_tensor_model_parallel_rank()
            tp_size = get_tensor_model_parallel_world_size()
            kept = [e for e, b in enumerate(bits) if b == 4]
            dem = [e for e, b in enumerate(bits) if b == 3]
            Kn, Km = len(kept), len(dem)
            layer.hyb = {"remap": {**{e: (0, i) for i, e in enumerate(kept)},
                                   **{e: (1, i) for i, e in enumerate(dem)}},
                         "H": H, "I": I, "gs": gs, "E": num_experts}
            _pfx = getattr(layer, "prefix", None) or getattr(layer, "layer_name", "?")
            layer.hyb["lname"] = str(_pfx)
            print(f"[hybrid] {_pfx}: {Kn}NVFP4 + {Km}NF3 (tp{tp_rank}/{tp_size} I={I})", flush=True)

            def wl(param, loaded, name_mapped=None, *, shard_id=None,
                   expert_id=None, return_success=False, **kw):
                nm = name_mapped or ""
                if "input_scale" in nm:            # W4A16 reference -> unused
                    return True
                grp, li = layer.hyb["remap"][int(expert_id)]
                fam = "w13" if "w13_" in nm else "w2"
                sh = shard_id
                if "weight_scale_2" in nm:         # NVFP4 per-tensor global (kept only)
                    tgt = getattr(layer, f"{fam}_weight_scale_2")
                    if fam == "w13":
                        tgt.data[li, 0 if sh == "w1" else 1] = loaded.reshape(()).to(tgt.dtype)
                    else:
                        tgt.data[li] = loaded.reshape(()).to(tgt.dtype)
                    return True
                # TP shard the block-quantized 2D tensor (gate/up -> dim0, down -> dim1)
                if tp_size > 1 and loaded.ndim >= 2:
                    if sh in ("w1", "w3"):
                        loaded = loaded.chunk(tp_size, 0)[tp_rank]
                    elif sh == "w2":
                        loaded = loaded.chunk(tp_size, 1)[tp_rank]
                if "weight_scale" in nm:           # block scale -> real storage, demux by group
                    tgt = getattr(layer, f"{fam}_nv_s" if grp == 0 else f"{fam}_nf_s")
                elif "weight_packed" in nm:        # NF3 packed weight
                    tgt = getattr(layer, f"{fam}_weight_packed")
                else:                              # plain NVFP4 weight
                    tgt = getattr(layer, f"{fam}_weight")
                d = tgt.data[li]
                if fam == "w13" and sh in ("w1", "w3"):   # gate->top half, up->bottom half
                    half = d.shape[0] // 2
                    d = d[:half] if sh == "w1" else d[half:]
                d.copy_(loaded.reshape(d.shape).to(d.dtype))
                return True

            def P(name, shape, dt=torch.uint8):
                p = torch.nn.Parameter(torch.zeros(shape, dtype=dt,
                                       device=torch.cuda.current_device()), requires_grad=False)
                set_weight_attrs(p, {"weight_loader": wl})
                layer.register_parameter(name, p)

            mk = lambda n: max(n, 1)
            tr3_tail = _HYBRID_EXL3_TR3 and Km > 0
            layer.hyb["_tr3"] = tr3_tail
            # --- names the stock (prefix-based) mapping produces (routing needs no patch) ---
            P("w13_weight",         (mk(Kn), 2 * I, H // 2))         # NVFP4 weight (kept)
            if not tr3_tail:
                P("w13_weight_packed", (mk(Km), 2 * I, H // 8 * 3))  # NF3 packed weight (demoted)
            P("w13_weight_scale",   (1,))                            # dispatcher (routes to *_s)
            P("w13_weight_scale_2", (mk(Kn), 2), torch.float32)      # NVFP4 global (kept)
            P("w13_input_scale",    (1,), torch.float32)             # dispatcher (ignored)
            P("w2_weight",          (mk(Kn), H, I // 2))
            if not tr3_tail:
                P("w2_weight_packed", (mk(Km), H, I // 8 * 3))
            P("w2_weight_scale",    (1,))
            P("w2_weight_scale_2",  (mk(Kn),), torch.float32)
            P("w2_input_scale",     (1,), torch.float32)
            # --- real block-scale storage (filled by dispatcher; not mapping-routed) ---
            scale_storage = [("w13_nv_s", (mk(Kn), 2 * I, H // gs)),
                             ("w2_nv_s",  (mk(Kn), H, I // gs))]
            if not tr3_tail:
                scale_storage += [("w13_nf_s", (mk(Km), 2 * I, H // 32)),
                                  ("w2_nf_s",  (mk(Km), H, I // 32))]
            for nm, sh in scale_storage:
                layer.register_parameter(nm, torch.nn.Parameter(
                    torch.zeros(sh, dtype=torch.float8_e4m3fn,
                                device=torch.cuda.current_device()), requires_grad=False))
            if tr3_tail:
                # exl3 tail slabs: plain tensors (NOT parameters; the stock loader must
                # never see them). The stream filter fills them; rows stay contiguous
                # so per-expert data_ptr tables can be built arithmetically.
                Lidx = self._layer_idx(layer)
                assert Lidx is not None, f"tr3 layer index unresolved: {layer.hyb['lname']}"
                assert Km == int(_TR3["meta"]["tr3_tail_per_layer"]), \
                    f"layer {Lidx}: Km={Km} != meta tail {_TR3['meta']['tr3_tail_per_layer']}"
                dev = torch.cuda.current_device()
                shapes = _tr3_slab_shapes(Km)
                for (proj, field), sh in shapes.items():
                    dt = torch.int16 if field == "trellis" else torch.float16
                    _TR3["slabs"][(Lidx, proj, field)] = torch.empty(
                        sh, dtype=dt, device=dev)
                layer.hyb["tr3_L"] = Lidx
                gib = sum(t.numel() * t.element_size()
                          for k, t in _TR3["slabs"].items() if k[0] == Lidx) / 2**30
                if Lidx in (3, 40, 77):
                    print(f"[hybrid] layer {Lidx}: tr3 slabs allocated "
                          f"({gib:.2f} GiB/rank, Km={Km})", flush=True)

        def _build_kept_b12x(self, layer):
            """Build a REAL b12x NVFP4 fused-MoE kernel over just the Kn kept experts.
            Reuses the production kernel (graph-safe, sm120) via a cloned num_experts=Kn
            FusedMoEConfig. Validated standalone: correct + skips out-of-range ids + cudagraph-safe.
            apply() remaps topk so kept->[0,Kn), non-kept->Kn (the kernel drops the sentinel)."""
            import dataclasses
            import torch.nn as _nn
            from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
                select_nvfp4_moe_backend, convert_to_nvfp4_moe_kernel_format,
                make_nvfp4_moe_quant_config, make_nvfp4_moe_kernel)
            from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig
            from vllm.model_executor.layers.quantization.utils.quant_utils import kNvfp4Static
            from vllm.model_executor.layers.fused_moe.activation import MoEActivation
            E = layer.hyb["E"]
            dev = layer.w13_weight.device
            remap = layer.hyb["remap"]
            Kn = sum(1 for (grp, li) in remap.values() if grp == 0)
            # Build the kept kernel as a NON-PARALLEL (tp=1) MoE over the PER-RANK intermediate:
            # the weights are already TP-sharded by the weight_loader (ispp per rank), so the
            # kernel must see intermediate_size == ispp and tp=1 (else it mis-indexes full vs
            # sharded -> per-element garbage). The layer's post-apply all-reduce handles TP.
            ispp = self.moe.intermediate_size_per_partition
            kept_moe = dataclasses.replace(
                self.moe, num_experts=Kn, num_local_experts=Kn, num_logical_experts=Kn,
                intermediate_size=ispp,
                moe_parallel_config=FusedMoEParallelConfig.make_no_parallel())
            backend, experts_cls = select_nvfp4_moe_backend(
                config=kept_moe, weight_key=kNvfp4Static, activation_key=None)
            KL = _nn.Module()
            KL.activation = getattr(layer, "activation", MoEActivation.SILU)
            KL.moe_config = kept_moe
            KL.local_num_experts = Kn
            a13 = torch.ones(Kn, 2, device=dev, dtype=torch.float32)
            a2 = torch.ones(Kn, device=dev, dtype=torch.float32)
            w13ks2 = layer.w13_weight_scale_2[:, 0].contiguous()
            conv = convert_to_nvfp4_moe_kernel_format(
                nvfp4_backend=backend, layer=KL,
                w13=layer.w13_weight, w13_scale=layer.w13_nv_s, w13_scale_2=w13ks2, a13_scale=a13,
                w2=layer.w2_weight, w2_scale=layer.w2_nv_s, w2_scale_2=layer.w2_weight_scale_2,
                a2_scale=a2, is_act_and_mul=True, use_a16=True)
            (cw13, cw13s, cw13s2, ca13, cw2, cw2s, cw2s2, ca2) = conv
            for k, v in [("w13_weight", cw13), ("w13_weight_scale", cw13s),
                         ("w13_weight_scale_2", cw13s2), ("w13_input_scale", ca13),
                         ("w2_weight", cw2), ("w2_weight_scale", cw2s),
                         ("w2_weight_scale_2", cw2s2), ("w2_input_scale", ca2)]:
                setattr(KL, k, v)
            qconf = make_nvfp4_moe_quant_config(backend, cw13s, cw2s, cw13s2, cw2s2,
                                                ca13, ca2, use_a16=True)
            kk = make_nvfp4_moe_kernel(qconf, kept_moe, experts_cls, routing_tables=None)
            kk.fused_experts.process_weights_after_loading(KL)
            # Mark this method as owning its modular kernel (supports_internal_mk -> True) so
            # vLLM's post-load maybe_init_modular_kernel() returns early instead of rebuilding
            # the parent kernel from the (freed) standard weight attrs.
            self.moe_kernel = kk
            # remap: global -> kept-local, else sentinel Kn (b12x skips ids >= num_experts)
            nvfp4_remap = torch.full((E,), Kn, dtype=torch.int32, device=dev)
            for g, (grp, li) in remap.items():
                if grp == 0:
                    nvfp4_remap[g] = li
            if _HYBRID_DEBUG and not _b12x_nf3_rt.get("_selftested"):
                _b12x_nf3_rt["_selftested"] = True
                try:
                    from vllm.distributed import get_tensor_model_parallel_rank as _gr
                    _rk = _gr()
                except Exception:
                    _rk = -1
                xt = torch.randn(4, layer.hyb["H"], device=dev, dtype=torch.bfloat16) * 0.03
                ids = (torch.arange(32, device=dev, dtype=torch.int32) % Kn).reshape(4, 8)
                tww = torch.full((4, 8), 0.125, device=dev, dtype=torch.float32)
                o = kk.apply(xt, KL.w13_weight, KL.w2_weight, tww, ids,
                             activation=KL.activation, global_num_experts=Kn,
                             expert_map=None, apply_router_weight_on_input=False,
                             shared_experts=None, shared_experts_input=None)
                torch.cuda.synchronize()
                print(f"[selftest] r{_rk} {getattr(layer, 'prefix', '?')}: "
                      f"out={o.float().norm():.4f} inf={bool(torch.isinf(o).any())} "
                      f"nan={bool(torch.isnan(o).any())} "
                      f"cw13={KL.w13_weight.view(torch.uint8).float().mean():.3f} "
                      f"cw13s2_max={KL.w13_weight_scale_2.float().abs().max():.3e} "
                      f"cw2s2_max={KL.w2_weight_scale_2.float().abs().max():.3e}", flush=True)
            layer.hyb["kept_kernel"] = kk
            layer.hyb["kept_KL"] = KL
            layer.hyb["Kn"] = Kn
            layer.hyb["nvfp4_remap"] = nvfp4_remap
            # CRITICAL: keep the converted tensors alive past this function. b12x's
            # process_weights compacts the source weight to (0,) but its prepared weights
            # VIEW the converted `cw13`/`cw2`; if those locals are freed on return the views
            # dangle -> per-layer garbage (works in a script where they stay in scope).
            layer.hyb["_keepalive"] = conv
            # free the BIG compact kept originals (KL holds the converted copies) -> flat VRAM.
            # keep the tiny *_weight_scale* params (get_fused_moe_quant_config may still read them).
            keep_dbg = _HYBRID_DEBUG and _dbg_first_build[0]   # keep originals for ONE layer only
            if keep_dbg:
                _dbg_first_build[0] = False
            for nm in ("w13_weight", "w2_weight", "w13_nv_s", "w2_nv_s"):
                if not keep_dbg and hasattr(layer, nm):
                    try:
                        delattr(layer, nm)
                    except Exception:
                        setattr(layer, nm, None)
            print(f"[hybrid] {getattr(layer, 'prefix', '?')}: kept b12x kernel built "
                  f"(Kn={Kn}, backend={backend}, kept_ispp={kept_moe.intermediate_size_per_partition}, "
                  f"kept_tp={kept_moe.moe_parallel_config.tp_size}, "
                  f"w13={tuple(KL.w13_weight.shape)})", flush=True)

        def _build_b12x_nf3(self, layer):
            """HYBRID_KEPT=b12x_nf3: drive BOTH tiers through the b12x W4A16
            CuteDSL MoE kernel.

            Object A (kept NVFP4)  -> the PRODUCTION vLLM chain via
              _build_kept_b12x (select_nvfp4_moe_backend -> convert_to_nvfp4_
              moe_kernel_format -> make_nvfp4_moe_kernel, no-parallel clone,
              keepalive). Manual composition through the prepare building
              blocks is numerically WRONG for varying real-range scales
              (harness-proven 2026-07-01: only the production convert chain
              passes vs the dequant reference; probe test_b12x.py TESTS A-D).
            Object B (NF3 192)     -> weight_layout="nf3_2p1"/e4m3_k32 packed
              from the 837 checkpoint planes (chunked over experts) through
              our NF3 kernel format (GPU unit test + in-model norms PASS).
            Launches/buffers are built lazily at first apply() (topk + real max
            m known there; the first forward is vLLM's EAGER profile run, so
            nothing compiles inside CUDA-graph capture)."""
            import torch
            from b12x.moe.fused.w4a16.prepare import (
                PreparedNF3MoeWeights,
                W4A16PackedWeights,
                _make_workspace,
                _nf3_pack_code_experts,
                _nf3_pack_scale_experts,
                _permute_nvfp4_scales,
                _repack_weight,
            )
            import nf3_replan
            hyb = layer.hyb
            E, H, I = hyb["E"], hyb["H"], hyb["I"]
            remap = hyb["remap"]
            dev = layer.w13_weight.device
            Kn = sum(1 for (grp, _li) in remap.values() if grp == 0)
            Km = sum(1 for (grp, _li) in remap.values() if grp == 1)
            emap_a = torch.full((E,), -1, dtype=torch.int32, device=dev)
            emap_b = torch.full((E,), -1, dtype=torch.int32, device=dev)
            for g, (grp, li) in remap.items():
                (emap_a if grp == 0 else emap_b)[g] = li
            fc1_tn, fc2_tn = _B12X_NF3_TILES[1], _B12X_NF3_TILES[3]

            # ---- object B: NF3 -> "nf3_2p1" (chunked over experts) ----
            # (built FIRST: object A's production build frees the kept originals)
            prep_b = None
            if hyb.get("_tr3"):
                # tr3 tail: exl3 slabs were filled by the stream filter; build the
                # per-expert pointer tables here instead of any NF3 packing.
                self._build_exl3_tr3_tables(layer)
            elif Km > 0:
                chunk = 16   # bound transient VRAM (codes int32 = ~400MB/16 w13 experts)
                w13_planes, w2_planes = [], []
                for e0 in range(0, Km, chunk):
                    codes = nf3_replan.unpack_837_codes(
                        layer.w13_weight_packed[e0:e0 + chunk], H)
                    w13_planes.append(_nf3_pack_code_experts(
                        codes, size_k=H, size_n=2 * I, tile_n=fc1_tn))
                    del codes
                for e0 in range(0, Km, chunk):
                    codes = nf3_replan.unpack_837_codes(
                        layer.w2_weight_packed[e0:e0 + chunk], I)
                    w2_planes.append(_nf3_pack_code_experts(
                        codes, size_k=I, size_n=H, tile_n=fc2_tn))
                    del codes
                w13_nf3 = torch.cat(w13_planes, 0).contiguous(); del w13_planes
                w2_nf3 = torch.cat(w2_planes, 0).contiguous(); del w2_planes
                w13_ns = _nf3_pack_scale_experts(
                    layer.w13_nf_s.float(), size_k=H, size_n=2 * I)
                w2_ns = _nf3_pack_scale_experts(
                    layer.w2_nf_s.float(), size_k=I, size_n=H)
                nf3_global = torch.full(
                    (Km,), 2.0 ** 116, dtype=torch.float32, device=dev)
                prep_b = PreparedNF3MoeWeights(
                    w13=w13_nf3, w13_scale=w13_ns, w13_global_scale=nf3_global,
                    w2=w2_nf3, w2_scale=w2_ns,
                    w2_global_scale=nf3_global.clone(),
                    workspace=_make_workspace(dev),
                    hidden_size=H, intermediate_size=I, num_experts=Km,
                    is_gated=True, params_dtype=torch.bfloat16,
                    fc1_tile_n=fc1_tn, fc2_tile_n=fc2_tn)

            hyb["prepB"] = prep_b
            hyb["prepA"] = None  # filled by the kept build below
            hyb["emap_a"], hyb["emap_b"] = emap_a, emap_b
            hyb["Kn"], hyb["Km"] = Kn, Km
            keep_dbg = _HYBRID_DEBUG and _dbg_first_build[0]
            if keep_dbg:
                _dbg_first_build[0] = False
            # ---- object A: kept NVFP4 -> manual "packed" composition.
            # REHABILITATED 2026-07-01: byte-identical to the stock prepare entry
            # (verified) and PASSES the FIXED (swizzle=False) dequant reference on
            # ranks 0/1/2 at rel 0.006. The earlier condemnation used a broken
            # reference. Produces weight_layout="packed" -> TC-decode launches
            # compile; NO modular kernel / workspace manager in the forward.
            if Kn > 0:
                g13 = layer.w13_weight_scale_2[:Kn, 0].contiguous()
                g2 = layer.w2_weight_scale_2[:Kn].contiguous()
                w13_packed = _repack_weight(
                    layer.w13_weight.contiguous(), size_k=H, size_n=2 * I)
                w2_packed = _repack_weight(
                    layer.w2_weight.contiguous(), size_k=I, size_n=H)
                w13_ps, w13_pg = _permute_nvfp4_scales(
                    layer.w13_nv_s, g13, size_k=H, size_n=2 * I,
                    a_dtype=torch.bfloat16)
                w2_ps, w2_pg = _permute_nvfp4_scales(
                    layer.w2_nv_s, g2, size_k=I, size_n=H,
                    a_dtype=torch.bfloat16)
                hyb["prepA"] = W4A16PackedWeights(
                    w13=w13_packed, w13_scale=w13_ps, w13_global_scale=w13_pg,
                    w2=w2_packed, w2_scale=w2_ps, w2_global_scale=w2_pg,
                    workspace=_make_workspace(dev),
                    hidden_size=H, intermediate_size=I, num_experts=Kn,
                    is_gated=True, params_dtype=torch.bfloat16,
                    source_format="modelopt_nvfp4", w13_layout="w13",
                    weight_layout="packed", scale_format="e4m3_k16")
                if not keep_dbg:
                    for nm in ("w13_weight", "w2_weight", "w13_nv_s", "w2_nv_s"):
                        p_ = getattr(layer, nm, None)
                        if p_ is not None and getattr(p_, "data", None) is not None:
                            p_.data = p_.data.new_empty((0,))
            if not keep_dbg:
                for nm in ("w13_weight_packed", "w2_weight_packed",
                           "w13_nf_s", "w2_nf_s"):
                    p = getattr(layer, nm, None)
                    if p is not None and getattr(p, "data", None) is not None:
                        p.data = p.data.new_empty((0,))
            print(f"[hybrid] {getattr(layer, 'prefix', '?')}: "
                  f"{'b12x+exl3_tr3' if hyb.get('_tr3') else 'b12x_nf3'} built "
                  f"(Kn={Kn} via production chain + Km={Km} "
                  f"{'exl3-trellis-3.0' if hyb.get('_tr3') else 'nf3_2p1'}, "
                  f"tiles={_B12X_NF3_TILES})", flush=True)

        def _build_exl3_tr3_tables(self, layer):
            """Pointer tables + tail emap for the exl3_moe fused tail. Fail-closed:
            an incomplete slab fill (missing checkpoint slices) aborts the boot."""
            import torch
            _ensure_exl3_ext()   # publishes the ABI shim, then imports the ext
            hyb = layer.hyb
            L = hyb["tr3_L"]
            ent = _load_tr3()[L]
            Km = len(ent["tail"])
            filled = _TR3["fill"].get(L, 0)
            expect = Km * 3 * 3     # 3 projs x (trellis, suh, svh)
            assert filled == expect, \
                f"layer {L}: tr3 slab fill {filled}/{expect} (incomplete checkpoint load)"
            mcg_n = _TR3["mcg_seen"].get(L, 0)
            assert mcg_n == Km * 3, f"layer {L}: mcg verified {mcg_n}/{Km * 3}"
            dev = _TR3["slabs"][(L, "gate_proj", "trellis")].device
            ptrs = {}
            for proj in _TR3_PROJS:
                for field in ("trellis", "suh", "svh"):
                    slab = _TR3["slabs"][(L, proj, field)]
                    assert slab.is_contiguous(), f"layer {L} {proj}.{field} slab not contiguous"
                    step = slab.stride(0) * slab.element_size()
                    base = slab.data_ptr()
                    ptrs[(proj, field)] = torch.tensor(
                        [base + i * step for i in range(Km)],
                        dtype=torch.int64, device=dev)
            hyb["tr3_ptrs"] = [
                ptrs[("gate_proj", "trellis")], ptrs[("gate_proj", "suh")], ptrs[("gate_proj", "svh")],
                ptrs[("up_proj", "trellis")], ptrs[("up_proj", "suh")], ptrs[("up_proj", "svh")],
                ptrs[("down_proj", "trellis")], ptrs[("down_proj", "suh")], ptrs[("down_proj", "svh")],
            ]
            hyb["tr3_slabs"] = {
                proj: {field: _TR3["slabs"][(L, proj, field)]
                       for field in ("trellis", "suh", "svh")}
                for proj in _TR3_PROJS
            }
            # global expert id -> tail-local id; hot/kept ids -> Km (the sentinel
            # bucket: sorts last, expert_count[Km] is ignored by the kernel).
            emap = torch.full((hyb["E"],), Km, dtype=torch.int64, device=dev)
            for g in ent["tail"]:
                emap[g] = ent["pos"][g]
            hyb["tr3_emap"] = emap
            hyb["tr3_rev"] = torch.tensor(
                ent["tail"], dtype=torch.int64, device=dev)
            hyb["tr3_Km"] = Km
            print(f"[hybrid] layer {L}: exl3_tr3 tables built (Km={Km}, 9 ptr tables)", flush=True)

        def _apply_exl3_tr3(self, layer, x, topk_weights, topk_ids):
            """Tail-192 through fused GPU route packing for eager prefill.

            Decode/small batches retain the proven chunked ABI for FULL graph
            capture. Prefill maps, histograms, stably packs and computes all rows
            inside one extension host call; the CUDA compute kernel internally
            tiles expert spans to the bounded temp-row capacity.

            Sync-free by construction (sort + scatter_add_, preallocated scratch,
            fixed python chunk count per batch size) -> FULL-cudagraph capturable.
            Chunking bounds every per-expert row count by cap == temp rows, so the
            kernel's count>max_tokens_per_expert skip can NEVER trigger (exactness)."""
            import torch
            _e3 = _ensure_exl3_ext()
            hyb = layer.hyb
            rt = _TR3["rt"]
            m = int(x.shape[0])
            topk = int(topk_ids.shape[1])
            Km = hyb["tr3_Km"]
            xh = rt["xh"][:m]
            xh.copy_(x)                      # bf16 -> f16 (exl3_moe wants kHalf)
            out32 = rt["out32"][:m]
            out32.zero_()                    # kernel accumulates weighted results
            cap = rt["cap"]
            if m > cap and hasattr(_e3, "exl3_moe_fused"):
                dq_threshold = _TR3_DQ_THRESHOLD
                _e3.exl3_moe_fused(
                    xh, out32, topk_ids, topk_weights, hyb["tr3_emap"],
                    rt["expert_count"], rt["expert_offsets"],
                    rt["token_sorted"], rt["weight_sorted"],
                    rt["tg"], rt["tu"], rt["ig"], rt["iu"],
                    0, 3, 3, 3,
                    *hyb["tr3_ptrs"],
                    True, False, True, False, True, False, 0.0,
                    dq_threshold)
                if dq_threshold > 0:
                    # Match exllamav3 BlockSparseMLP.run_single_expert_dq:
                    # reconstruct each high-row expert once, then use tensor-core
                    # HGEMM. Counts are read only on eager prefill; decode remains
                    # sync-free on the legacy/captured path.
                    counts = rt["expert_count"].tolist()
                    start = 0
                    slabs = hyb["tr3_slabs"]
                    for expert_idx, count in enumerate(counts[:Km]):
                        end = start + count
                        if count >= dq_threshold:
                            tok = rt["token_sorted"][start:end]
                            if _QUALITY_FP32_ROUTER:
                                # Preserve the original FP32 router weights for
                                # reconstructed experts. The fused route pack
                                # intentionally stores FP16 weights for the
                                # trellis kernel ABI, but HGEMM output is FP32
                                # and does not need that extra rounding.
                                gid = hyb["tr3_rev"][expert_idx]
                                ids_for_tok = topk_ids.index_select(0, tok)
                                route_col = (ids_for_tok == gid).to(
                                    torch.int64).argmax(dim=1, keepdim=True)
                                w = topk_weights.index_select(0, tok).gather(
                                    1, route_col).float()
                            else:
                                w = rt["weight_sorted"][start:end].unsqueeze(1)
                            current = xh.index_select(0, tok)
                            yh = rt["dq_yh"][:count * 2]
                            yh_g, yh_u = yh[:count], yh[count:count * 2]
                            interm = rt["dq_interm"][:count * 2]
                            ig, iu = interm[:count], interm[count:count * 2]
                            ia = rt["dq_act"][:count]
                            out = rt["dq_out"][:count]

                            sg = slabs["gate_proj"]
                            su = slabs["up_proj"]
                            sd = slabs["down_proj"]
                            _e3.had_r_128(current, yh_g, sg["suh"][expert_idx], None, 1.0)
                            _e3.reconstruct(rt["dq_recon_up"], sg["trellis"][expert_idx], 3, True, False)
                            _e3.hgemm(yh_g, rt["dq_recon_up"], ig)
                            _e3.had_r_128(current, yh_u, su["suh"][expert_idx], None, 1.0)
                            _e3.reconstruct(rt["dq_recon_up"], su["trellis"][expert_idx], 3, True, False)
                            _e3.hgemm(yh_u, rt["dq_recon_up"], iu)
                            _e3.had_r_128(ig, ig, None, sg["svh"][expert_idx], 1.0)
                            _e3.had_r_128(iu, iu, None, su["svh"][expert_idx], 1.0)
                            _e3.silu_mul(ig, iu, ia, 0.0)
                            _e3.had_r_128(ia, ia, sd["suh"][expert_idx], None, 1.0)
                            _e3.reconstruct(rt["dq_recon_down"], sd["trellis"][expert_idx], 3, True, False)
                            _e3.hgemm(ia, rt["dq_recon_down"], out)
                            _e3.had_r_128(out, out, None, sd["svh"][expert_idx], 1.0)
                            out.mul_(w)
                            if _QUALITY_DETERMINISTIC_ACCUM:
                                # token indices are unique inside one expert.
                                # Process experts in a fixed loop order and use
                                # an index copy instead of CUDA index_add_
                                # atomics, making the reconstructed portion's
                                # accumulation deterministic.
                                accum = out32.index_select(0, tok)
                                accum.add_(out)
                                out32.index_copy_(0, tok, accum)
                            else:
                                out32.index_add_(0, tok, out)
                        start = end
                return out32
            ids_l = hyb["tr3_emap"][topk_ids.long()]          # [m, topk] int64
            w_h = topk_weights.to(torch.float16)
            for c0 in range(0, m, cap):
                mc = min(cap, m - c0)
                flat = ids_l[c0:c0 + mc].reshape(-1)
                order = torch.argsort(flat)                   # group by tail-local expert
                tok = rt["flat_token"][:mc * topk].index_select(0, order).contiguous()
                ws = w_h[c0:c0 + mc].reshape(-1).index_select(0, order).contiguous()
                counts = torch.zeros(Km + 1, dtype=torch.int64, device=flat.device)
                counts.scatter_add_(0, flat, rt["ones"][:mc * topk])   # graph-safe bincount
                _e3.exl3_moe(
                    xh[c0:c0 + mc], out32[c0:c0 + mc], counts, tok, ws,
                    rt["tg"], rt["tu"], rt["ig"], rt["iu"],
                    0,                       # MOE_ACT_SILU
                    3, 3, 3,                 # K = bits (gate, up, down)
                    *hyb["tr3_ptrs"],
                    True, False, True, False, True, False,   # mcg codebook, all three
                    0.0)                     # act_limit: no clamp (GLM silu)
            return out32                     # fp32 [m, H]

        def _ensure_b12x_nf3_runtime(self, layer, m, topk):
            """First-apply init: pinned-tile preplanned launches (per object) +
            ONE module-level scratch/buffer set. First apply = vLLM's eager
            profile run at max_num_batched_tokens, so max_m sizes itself to the
            real serving ceiling and nothing compiles during graph capture."""
            import dataclasses
            import torch
            from b12x.moe.fused.w4a16.kernel import compile_w4a16_fused_moe
            from b12x.moe.fused.w4a16.host import (
                make_w4a16_packed_buffers, max_packed_route_slots)
            hyb = layer.hyb
            st = _b12x_nf3_rt
            E, H, I = hyb["E"], hyb["H"], hyb["I"]
            dev = hyb["emap_a"].device
            if st["max_m"] is None:
                st["max_m"] = max(_B12X_NF3_MAX_TOKENS, int(m))
                st["topk"] = int(topk)
            if int(topk) != st["topk"]:
                raise RuntimeError(
                    f"b12x_nf3: topk changed {st['topk']} -> {topk}")
            props = torch.cuda.get_device_properties(dev)
            sms = int(props.multi_processor_count)
            max_shared_mem = int(getattr(
                props, "shared_memory_per_block_optin", 101_376))

            def launches(prepared):
                key = (prepared.num_experts, prepared.weight_layout,
                       prepared.scale_format, st["topk"], st["max_m"], H, I)
                got = st["launches"].get(key)
                if got is not None:
                    return got
                common = dict(
                    hidden_size=H, intermediate_size=I,
                    num_experts=prepared.num_experts, top_k=st["topk"],
                    activation="silu", apply_router_weight_on_input=False,
                    element_dtype="bf16", fast_math=True, sms=sms,
                    max_shared_mem=max_shared_mem,
                    weight_layout=prepared.weight_layout,
                    scale_format=prepared.scale_format,
                    force_tile_config=_B12X_NF3_TILES)
                # ONE launch per object for ALL m (1..max_m): spec-0 (size_m>=2)
                # block-64 packed-route + expert_map + zero_fc2_output=True.
                # VALIDATED on real ckpt data at m=1/8/33 (rel 0.02-0.05).
                # TC-decode/block-8 and the size_m=1 compile are BROKEN for these
                # shapes+tiles (rel 1.4 / 197) — never use them. Decode padding
                # cost ~0: blocks exist only for ACTIVE experts; pad slots waste
                # ALU, not HBM bytes, and decode is bandwidth-bound.
                cap_slots = max_packed_route_slots(st["max_m"] * st["topk"], 64, E)
                pre = compile_w4a16_fused_moe(
                    size_m=st["max_m"], zero_fc2_output=True,
                    moe_block_size=64, max_m_blocks=(cap_slots + 63) // 64,
                    direct_topk_routes=False, tc_decode_fused_sum=False,
                    **common)
                assert (int(pre.fc1_tile_n), int(pre.fc2_tile_n)) == (
                    _B12X_NF3_TILES[1], _B12X_NF3_TILES[3]), "tile pin failed"
                dec = pre
                if _HYBRID_TC_DECODE:
                    # dec launch: SMALL size_m (8) direct-topk + fused-sum at FORCED
                    # PIN tiles. Needs the op-boundary tile-config passthrough fix
                    # (b12x_nf3_*_fix/kernel.py): the stock op re-resolves its kernel
                    # WITHOUT force_tile_config, silently swapping PIN->auto so the
                    # 256-pack was read with wrong geometry (garbage rel 1.3-1.5 at
                    # m=1,6,8). Fixed kernel honors the forced PIN -> reads the SAME
                    # production 256-pack (NO 2nd pack, NO pool cost). Opus-validated
                    # rel 0.006-0.011 @ m=1..8, single-compile reuse worst 0.0099.
                    # 0.277ms vs packed 0.365ms = ~14ms/step decode bounty.
                    # Run side: LUT-mapped local ids, NO expert_map (direct-topk).
                    try:
                        cand = compile_w4a16_fused_moe(
                            size_m=_B12X_NF3_DECODE_M, zero_fc2_output=False,
                            moe_block_size=8,
                            max_m_blocks=_B12X_NF3_DECODE_M * st["topk"],
                            direct_topk_routes=True, tc_decode_fused_sum=True,
                            **common)  # common carries force_tile_config=_B12X_NF3_TILES
                        assert (int(cand.fc1_tile_n), int(cand.fc2_tile_n)) == (
                            _B12X_NF3_TILES[1], _B12X_NF3_TILES[3]), "tc tile pin failed"
                        dec = cand
                        print(f"[hybrid] tc-decode launch armed (force PIN) "
                              f"E={prepared.num_experts} tiles="
                              f"{int(cand.fc1_tile_n)},{int(cand.fc2_tile_n)}",
                              flush=True)
                    except Exception as _te:
                        print(f"[hybrid] tc-decode compile failed, packed fallback: "
                              f"{_te}", flush=True)
                st["launches"][key] = (dec, pre)
                return st["launches"][key]

            if hyb.get("prepA") is not None:
                hyb["launchA"] = launches(hyb["prepA"])
            if hyb.get("prepB") is not None:
                hyb["launchB"] = launches(hyb["prepB"])
            if st["buffers"] is None:
                prep_any = hyb.get("prepA") or hyb.get("prepB")
                if prep_any is None:
                    # uniform-NVFP4 layer (MTP) first: no NF3 object anywhere yet;
                    # the kept kernel manages its own workspace -> no buffers needed.
                    hyb["_rt_ready"] = True
                    return
                buf = make_w4a16_packed_buffers(
                    prep_any, m=st["max_m"], topk=st["topk"],
                    dtype=torch.bfloat16, device=dev, route_num_experts=E)
                # the preplanned prefill launch validates route capacity at
                # moe_block_size=64; the plan's own block choice can be smaller
                # for small max_m -> upsize the route buffers if needed.
                need_slots = max_packed_route_slots(st["max_m"] * st["topk"], 64, E)
                need_blocks = (need_slots + 63) // 64
                if (buf.packed_route_indices.numel() < need_slots
                        or buf.block_expert_ids.numel() < need_blocks):
                    buf = dataclasses.replace(
                        buf,
                        packed_route_indices=torch.empty(
                            (need_slots,), dtype=torch.int32, device=dev),
                        block_expert_ids=torch.empty(
                            (need_blocks,), dtype=torch.int32, device=dev))
                st["buffers"] = buf
                st["out_a"] = buf.output          # [max_m, H], fully overwritten per call
                st["out_b"] = torch.empty_like(buf.output)
                print(f"[hybrid] b12x_nf3 runtime ready: max_m={st['max_m']} "
                      f"topk={st['topk']} sms={sms}", flush=True)
            if hyb.get("_tr3") and _TR3["rt"] is None:
                _e3 = _ensure_exl3_ext()
                C = _e3.exl3_moe_max_concurrency(torch.cuda.current_device())
                cap = _TR3_CHUNK
                Hh, Ipp = hyb["H"], hyb["I"]
                _TR3["rt"] = {
                    "cap": cap,
                    "xh": torch.empty((st["max_m"], Hh), dtype=torch.float16, device=dev),
                    "out32": torch.empty((st["max_m"], Hh), dtype=torch.float32, device=dev),
                    "tg": torch.empty((C, cap, Hh), dtype=torch.float16, device=dev),
                    "tu": torch.empty((C, cap, Hh), dtype=torch.float16, device=dev),
                    "ig": torch.empty((C, cap, Ipp), dtype=torch.float16, device=dev),
                    "iu": torch.empty((C, cap, Ipp), dtype=torch.float16, device=dev),
                    "flat_token": torch.arange(cap, dtype=torch.int64, device=dev
                                               ).repeat_interleave(st["topk"]),
                    "ones": torch.ones(cap * st["topk"], dtype=torch.int64, device=dev),
                    "expert_count": torch.empty(hyb["tr3_Km"] + 1,
                                                dtype=torch.int64, device=dev),
                    "expert_offsets": torch.empty(hyb["tr3_Km"] + 1,
                                                  dtype=torch.int64, device=dev),
                    "token_sorted": torch.empty(st["max_m"] * st["topk"],
                                                dtype=torch.int64, device=dev),
                    "weight_sorted": torch.empty(st["max_m"] * st["topk"],
                                                 dtype=torch.float16, device=dev),
                    "dq_yh": torch.empty((st["max_m"] * 2, Hh),
                                         dtype=torch.float16, device=dev),
                    "dq_interm": torch.empty((st["max_m"] * 2, Ipp),
                                             dtype=torch.float16, device=dev),
                    "dq_act": torch.empty((st["max_m"], Ipp),
                                          dtype=torch.float16, device=dev),
                    "dq_out": torch.empty((st["max_m"], Hh),
                                          dtype=torch.float32, device=dev),
                    "dq_recon_up": torch.empty((Hh, Ipp),
                                               dtype=torch.float16, device=dev),
                    "dq_recon_down": torch.empty((Ipp, Hh),
                                                 dtype=torch.float16, device=dev),
                }
                print(f"[hybrid] exl3_tr3 runtime ready: cap={cap} concurrency={C} "
                      f"max_m={st['max_m']} temps="
                      f"{(2 * C * cap * (Hh + Ipp) * 2) / 2**20:.0f} MiB", flush=True)
            hyb["_rt_ready"] = True

        def _apply_b12x_nf3(self, layer, x, topk_weights, topk_ids):
            import torch
            from b12x.moe.fused.w4a16.kernel import run_w4a16_moe
            hyb = layer.hyb
            st = _b12x_nf3_rt
            m = int(x.shape[0])
            topk = int(topk_ids.shape[1])
            if not hyb.get("_rt_ready"):
                self._ensure_b12x_nf3_runtime(layer, m, topk)
            if m > st["max_m"]:
                raise RuntimeError(
                    f"b12x_nf3: m={m} exceeds planned capacity {st['max_m']}; "
                    "set HYBRID_B12X_MAX_TOKENS >= max_num_batched_tokens")
            decode = m <= _B12X_NF3_DECODE_M
            if _HYBRID_ACT_CAPTURE and m > 8:
                try:
                    from vllm.distributed import get_tensor_model_parallel_rank
                    if get_tensor_model_parallel_rank() == 0:
                        pfx = str(hyb.get("lname") or getattr(layer, "prefix", None)
                                  or id(layer)).replace("/", "_").replace(".", "_")
                        b = _act_store["buf"].setdefault(pfx, {"x": [], "ids": []})
                        if _act_store["flushed"].get(pfx, 0) < 32768:  # cap tokens/layer
                            b["x"].append(x.detach().to(torch.float16).cpu())
                            b["ids"].append(topk_ids.detach().to(torch.int32).cpu())
                            if sum(t.shape[0] for t in b["x"]) >= 8192:
                                import os as _os
                                _os.makedirs(_HYBRID_ACT_CAPTURE, exist_ok=True)
                                fn = f"{_HYBRID_ACT_CAPTURE}/{pfx}.pt"
                                prev = torch.load(fn) if _os.path.exists(fn) else {"x": [], "ids": []}
                                prev["x"].append(torch.cat(b["x"]))
                                prev["ids"].append(torch.cat(b["ids"]))
                                torch.save(prev, fn)
                                _act_store["flushed"][pfx] = _act_store["flushed"].get(pfx, 0) + sum(t.shape[0] for t in b["x"])
                                b["x"], b["ids"] = [], []
                except Exception as _ce:
                    if not _act_store.get("_warned"):
                        _act_store["_warned"] = True
                        print("[hybrid][act-capture] failed:", repr(_ce), flush=True)
            tw = (topk_weights if topk_weights.dtype == torch.float32
                  else topk_weights.float())
            if not tw.is_contiguous():
                tw = tw.contiguous()
            buf = st["buffers"]

            def run(prepared, launch_pair, emap, out):
                use_dec = (_HYBRID_TC_DECODE and decode
                           and launch_pair[0] is not launch_pair[1])
                launch = launch_pair[0] if use_dec else launch_pair[1]
                ids = (topk_ids if topk_ids.dtype == torch.int32
                       else topk_ids.to(torch.int32))
                if not ids.is_contiguous():
                    ids = ids.contiguous()
                if use_dec:
                    # direct-topk path: kernel reads FLAT local ids and skips
                    # negatives itself; expert_map must NOT be used. emap doubles
                    # as the global->local LUT (graph-safe gather).
                    ids = emap[ids.long()].to(torch.int32).contiguous()
                    em = None
                else:
                    em = emap   # kernel translates global->local + drops -1 (zero=True pairs)
                return run_w4a16_moe(
                    x, prepared, tw, ids,
                    activation="silu",
                    intermediate_cache13=buf.intermediate_cache13,
                    intermediate_cache2=buf.intermediate_cache2,
                    output=out,
                    fc1_c_tmp=buf.fc1_c_tmp, fc2_c_tmp=buf.fc2_c_tmp,
                    packed_route_indices=buf.packed_route_indices,
                    block_expert_ids=buf.block_expert_ids,
                    packed_route_count=buf.packed_route_count,
                    expert_offsets=buf.expert_offsets,
                    expert_map=em,
                    fused_launch=launch)

            if hyb["Km"] == 0:
                # uniform-NVFP4 layer (MTP/nextn): single tier through OUR launcher
                out = torch.empty((m, hyb["H"]), dtype=x.dtype, device=x.device)
                out = run(hyb["prepA"], hyb["launchA"], hyb["emap_a"], out)
                if _HYBRID_DEBUG and not hyb.get("_dbg"):
                    hyb["_dbg"] = True
                    print(f"[dbg-nf3] {getattr(layer, 'prefix', '?')} m={m} "
                          f"A-only={out.float().norm():.1f}", flush=True)
                return out
            out_a = out_b = None
            if _HYBRID_TIER == "a":
                out = run(hyb["prepA"], hyb["launchA"], hyb["emap_a"],
                          st["out_a"][:m]).clone()
            elif _HYBRID_TIER == "b":
                if hyb.get("_tr3"):
                    out = self._apply_exl3_tr3(
                        layer, x, topk_weights, topk_ids).to(x.dtype).clone()
                else:
                    out = run(hyb["prepB"], hyb["launchB"], hyb["emap_b"],
                              st["out_b"][:m]).clone()
            else:
                prof = _HYBRID_PROFILE and m > 64
                if prof:
                    import time as _time
                    _w0 = _time.perf_counter()
                    ev = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
                    ev[0].record()
                out_a = run(hyb["prepA"], hyb["launchA"], hyb["emap_a"],
                            st["out_a"][:m])
                if prof:
                    ev[1].record()
                if hyb.get("_tr3"):
                    out_b = self._apply_exl3_tr3(layer, x, topk_weights, topk_ids)
                else:
                    out_b = run(hyb["prepB"], hyb["launchB"], hyb["emap_b"],
                                st["out_b"][:m])
                if prof:
                    ev[2].record()
                    _prof_store["wall_ms"] = _prof_store.get("wall_ms", 0.0) + \
                        (_time.perf_counter() - _w0) * 1000.0
                    _prof_store["pend"].append((m, ev))
                    _prof_store["n"] += 1
                    if _prof_store["n"] % 624 == 0:  # ~8 full 78-layer prefill passes
                        try:
                            torch.cuda.synchronize()
                            for pm, pe in _prof_store["pend"]:
                                _prof_store["a_ms"] += pe[0].elapsed_time(pe[1])
                                _prof_store["b_ms"] += pe[1].elapsed_time(pe[2])
                                _prof_store["rows"] += pm
                            _prof_store["pend"] = []
                            from vllm.distributed import get_tensor_model_parallel_rank
                            if get_tensor_model_parallel_rank() == 0:
                                n = _prof_store["n"]
                                a = _prof_store["a_ms"] / n
                                b = _prof_store["b_ms"] / n
                                mavg = _prof_store["rows"] / n
                                w = _prof_store.get("wall_ms", 0.0) / n
                                print(f"[hybrid-prof] n={n} avg_m={mavg:.0f} "
                                      f"A(nvfp4)={a:.3f}ms B(nf3)={b:.3f}ms "
                                      f"wall={w:.3f}ms /layer-chunk "
                                      f"-> MoE-gpu {78*(a+b):.0f}ms MoE-wall {78*w:.0f}ms "
                                      f"per {mavg:.0f}-tok chunk",
                                      flush=True)
                        except Exception as _pe:
                            print("[hybrid-prof] harvest failed:", repr(_pe), flush=True)
                if hyb.get("_tr3"):
                    out = torch.add(out_b, out_a).to(x.dtype)  # fp32 tail + bf16 hot
                else:
                    out = out_a + out_b
            if _HYBRID_DEBUG and not hyb.get("_dbg"):
                hyb["_dbg"] = True
                na = "-" if out_a is None else f"{out_a.float().norm():.1f}"
                nb = "-" if out_b is None else f"{out_b.float().norm():.1f}"
                print(f"[dbg-nf3] {getattr(layer, 'prefix', '?')} m={m} "
                      f"decode={decode} tier={_HYBRID_TIER} x={x.float().norm():.1f} "
                      f"A={na} B={nb} out={out.float().norm():.1f} "
                      f"nan={bool(torch.isnan(out).any())}", flush=True)
                if getattr(layer, "w13_weight", None) is not None and \
                        layer.w13_weight.numel() > 0:
                    try:
                        ra = self._apply_ref(layer, x, topk_weights, topk_ids, 0)
                        rb = self._apply_ref(layer, x, topk_weights, topk_ids, 1)
                        ea = float((out_a.float() - ra.float()).abs().sum()
                                   / (ra.float().abs().sum() + 1e-9))
                        eb = float((out_b.float() - rb.float()).abs().sum()
                                   / (rb.float().abs().sum() + 1e-9))
                        print(f"[dbgcmp-nf3] A_rel={ea:.4f} B_rel={eb:.4f}",
                              flush=True)
                    except Exception as _e:
                        print("[dbgcmp-nf3] ref failed:", repr(_e), flush=True)
            return out

        def process_weights_after_loading(self, layer):
            if not hasattr(layer, "hyb"):
                return super().process_weights_after_loading(layer)
            if _HYBRID_KEPT == "b12x_nf3":
                try:
                    self._build_b12x_nf3(layer)
                    layer.hyb["_b12x_nf3"] = True
                    layer.hyb["_b12x"] = False
                    layer.hyb["_kept_triton"] = False
                    layer.hyb["_fast"] = False
                    return
                except Exception as e:
                    import traceback
                    if layer.hyb.get("_tr3"):
                        # FAIL CLOSED: a tr3 layer has no NF3 tensors and no
                        # reference path for trellis weights -> abort the boot
                        # rather than serve a silently wrong tail.
                        print("[hybrid] exl3_tr3 build FAILED (fail-closed, "
                              "no fallback):", e, flush=True)
                        traceback.print_exc()
                        raise
                    print("[hybrid] b12x_nf3 build FAILED -> triton/ref "
                          "fallback:", e, flush=True)
                    traceback.print_exc()
                    layer.hyb["_b12x_nf3"] = False
                    # originals are freed only on success -> the stock flow
                    # below still has everything it needs.
            E = layer.hyb["E"]
            dev = layer.w13_weight.device
            # --- NF3 tier: expert map (demoted global id -> local NF3 index, else -1) ---
            emap = torch.full((E,), -1, dtype=torch.int32, device=dev)
            for g, (grp, li) in layer.hyb["remap"].items():
                if grp == 1:
                    emap[g] = li
            layer.hyb["emap_nf3"] = emap
            if _HYBRID_NF3 == "fast":
                try:
                    import nf3_kernel  # noqa: F401 (mounted alongside loader; warm import)
                    layer.hyb["_fast"] = True
                except Exception as e:
                    print("[hybrid] nf3_kernel import failed -> reference:", e, flush=True)
                    layer.hyb["_fast"] = False
            else:
                layer.hyb["_fast"] = False
            # --- NVFP4 kept tier ---
            layer.hyb["_b12x"] = False
            layer.hyb["_kept_triton"] = False
            if _HYBRID_KEPT in ("triton", "b12x_nf3"):
                # Custom Triton NVFP4 grouped-GEMM: reads the checkpoint weights DIRECTLY
                # (no convert/prepare/copy), graph-safe. emap: kept global -> local, else -1.
                # ("b12x_nf3" reaches here ONLY after a failed b12x_nf3 build ->
                #  fall back to the proven triton kept tier; originals intact.)
                kmap = torch.full((E,), -1, dtype=torch.int32, device=dev)
                for gid, (grp, li) in layer.hyb["remap"].items():
                    if grp == 0:
                        kmap[gid] = li
                layer.hyb["emap_nvfp4"] = kmap
                layer.hyb["g13"] = layer.w13_weight_scale_2[:, 0].contiguous()
                layer.hyb["g2"] = layer.w2_weight_scale_2.contiguous()
                try:
                    import nvfp4_kernel  # noqa: F401
                    layer.hyb["_kept_triton"] = True
                except Exception as e:
                    print("[hybrid] nvfp4_kernel import failed -> reference:", e, flush=True)
            elif _HYBRID_KEPT == "b12x":
                try:
                    self._build_kept_b12x(layer)
                    layer.hyb["_b12x"] = True
                except Exception as e:
                    import traceback
                    print("[hybrid] kept b12x build FAILED -> reference NVFP4:", e, flush=True)
                    traceback.print_exc()

        def _apply_ref(self, layer, x, topk_weights, topk_ids, only_grp):
            H, I, gs = layer.hyb["H"], layer.hyb["I"], layer.hyb["gs"]
            remap = layer.hyb["remap"]
            out = torch.zeros_like(x)
            for e in torch.unique(topk_ids).tolist():
                if e < 0 or e not in remap:
                    continue
                grp, li = remap[e]
                if grp != only_grp:
                    continue
                if grp == 0:
                    w13 = _deq_nv(layer.w13_weight[li], layer.w13_nv_s[li], layer.w13_weight_scale_2[li], gs, dequantize_to_dtype)
                    w2 = _deq_nv(layer.w2_weight[li], layer.w2_nv_s[li], layer.w2_weight_scale_2[li], gs, dequantize_to_dtype)
                else:
                    w13 = _unpack_nf3(layer.w13_weight_packed[li], layer.w13_nf_s[li], H)
                    w2 = _unpack_nf3(layer.w2_weight_packed[li], layer.w2_nf_s[li], I)
                sel = (topk_ids == e)
                tok = sel.any(-1)
                if not tok.any():
                    continue
                g, u = (x[tok] @ w13.t()).chunk(2, -1)
                y = (F.silu(g) * u) @ w2.t()
                wgt = (topk_weights * sel).sum(-1)[tok].unsqueeze(-1).to(y.dtype)
                out[tok] += wgt * y
            return out

        def apply(self, layer, x, topk_weights, topk_ids,
                  shared_experts=None, shared_experts_input=None):
            # routed experts only; shared experts + routing handled by the runner.
            if not hasattr(layer, "hyb"):
                return super().apply(layer, x, topk_weights, topk_ids,
                                     shared_experts, shared_experts_input)
            if layer.hyb.get("_b12x_nf3"):
                return self._apply_b12x_nf3(layer, x, topk_weights, topk_ids)
            H, I, E = layer.hyb["H"], layer.hyb["I"], layer.hyb["E"]
            # --- NVFP4 kept tier ---
            if layer.hyb.get("_kept_triton"):
                import nvfp4_kernel
                kept_out = nvfp4_kernel.nvfp4_moe_layer(
                    x, layer.w13_weight, layer.w13_nv_s, layer.hyb["g13"],
                    layer.w2_weight, layer.w2_nv_s, layer.hyb["g2"],
                    topk_ids, topk_weights, layer.hyb["emap_nvfp4"], E, H, I, 16)
            elif layer.hyb.get("_b12x"):
                kk = layer.hyb["kept_kernel"]
                KL = layer.hyb["kept_KL"]
                Kn = layer.hyb["Kn"]
                kept_ids = layer.hyb["nvfp4_remap"][topk_ids]   # kept->[0,Kn), else Kn (b12x skips)
                kept_out = kk.apply(
                    x, KL.w13_weight, KL.w2_weight, topk_weights, kept_ids,
                    activation=KL.activation, global_num_experts=Kn, expert_map=None,
                    apply_router_weight_on_input=False,
                    shared_experts=None, shared_experts_input=None)
            else:
                kept_out = self._apply_ref(layer, x, topk_weights, topk_ids, 0)
            # --- NF3 tier (192 experts) ---
            if layer.hyb.get("_fast"):
                import nf3_kernel
                nf3_out = nf3_kernel.nf3_moe_layer(
                    x, layer.w13_weight_packed, layer.w13_nf_s,
                    layer.w2_weight_packed, layer.w2_nf_s,
                    topk_ids, topk_weights, layer.hyb["emap_nf3"], E, 32, H, I)
            else:
                nf3_out = self._apply_ref(layer, x, topk_weights, topk_ids, 1)
            out = kept_out + nf3_out
            if _HYBRID_DEBUG and not layer.hyb.get("_dbg"):
                layer.hyb["_dbg"] = True
                print(f"[dbg] {getattr(layer, 'prefix', '?')} "
                      f"x={x.float().norm():.1f} kept={kept_out.float().norm():.1f} "
                      f"nf3={nf3_out.float().norm():.1f} out={out.float().norm():.1f} "
                      f"nan={bool(torch.isnan(out).any())}", flush=True)
                if layer.hyb.get("_b12x") and hasattr(layer, "w13_weight"):
                    try:
                        rk = self._apply_ref(layer, x, topk_weights, topk_ids, 0)
                        ratio = float(kept_out.float().norm() / (rk.float().norm() + 1e-9))
                        erel = float((kept_out.float() - rk.float()).abs().sum()
                                     / (rk.float().abs().sum() + 1e-9))
                        print(f"[dbgcmp] {getattr(layer, 'prefix', '?')} "
                              f"b12x={kept_out.float().norm():.3f} ref={rk.float().norm():.3f} "
                              f"ratio={ratio:.3f} elem_rel={erel:.3f}", flush=True)
                    except Exception as _e:
                        print("[dbgcmp] ref failed:", repr(_e), flush=True)
            return out

    mod.ModelOptNvFp4Config.FusedMoEMethodCls = HybridNvFp4MoE
    print("[hybrid_loader] HybridNvFp4MoE installed", flush=True)

    # ---- fp8 non-expert dequant-on-load (weight_scale_fp8 modules are in `ignore` -> bf16) ----
    try:
        from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
        if not getattr(DefaultModelLoader, "_hyb_wrapped", False):
            _oga = DefaultModelLoader.get_all_weights
            def get_all_weights(self, model_config, model):
                it = _oga(self, model_config, model)
                if _HYBRID_EXL3_TR3:
                    # consume exl3 tail tensors BEFORE load_weights (their mapped
                    # names have no registered params); rank-slice into tr3 slabs.
                    it = _tr3_stream_filter(it)
                return _fp8_ne_transform(it)
            DefaultModelLoader.get_all_weights = get_all_weights
            DefaultModelLoader._hyb_wrapped = True
            print("[hybrid_loader] fp8 non-expert dequant-on-load installed", flush=True)
    except Exception as e:
        print("[hybrid_loader] wrap get_all_weights failed:", e, flush=True)

    # ---- 3. HYBRID_MXFP8_NATIVE: serve the F8 ne-tier via online-mxfp8 ----
    # Instead of dequant-to-bf16 (2 B/param resident), excluded LinearBase
    # modules whose disk tensors are F8 get Mxfp8OnlineLinearMethod: loader
    # still feeds them our dequanted bf16, the method re-quantizes e8m0/32 at
    # load (bit-exact round trip vs rev-3 disk) and serves through the B12X
    # fp8 GEMM (VLLM_USE_B12X_FP8_GEMM=1 forces B12xMxfp8LinearKernel).
    # Net: ~-3.65 GiB/GPU weights. Activations on these linears become
    # dynamic-A8 (Festr serving convention) — KLD re-gate before publishing.
    if os.environ.get("HYBRID_MXFP8_NATIVE", "0") == "1":
        try:
            import json as _json
            from vllm.model_executor.layers.quantization.online.mxfp8 import (
                Mxfp8OnlineLinearMethod)
            from vllm.model_executor.layers.linear import (
                LinearBase, UnquantizedLinearMethod)
            from vllm.model_executor.layers.vocab_parallel_embedding import (
                ParallelLMHead)
            _tier_path = os.environ.get(
                "HYBRID_MXFP8_TIER_JSON",
                "/opt/venv/lib/python3.12/site-packages/mxfp8_tier.json")
            _allow = set(_json.load(open(_tier_path))["module_prefixes"])

            if any((
                    _QUALITY_EXCLUDE_KV_B,
                    _QUALITY_EXCLUDE_SHARED,
                    _QUALITY_EXCLUDE_EARLY_MLP,
                    _QUALITY_EXCLUDE_FINAL_ATTN,
            )):
                _allow_before = len(_allow)

                def _quality_stack_allows_mxfp8(prefix):
                    # kv_b_proj is absorbed into BF16 W_UK/W_UV, so native
                    # MXFP8 only adds activation rounding without a speed win.
                    if (_QUALITY_EXCLUDE_KV_B and
                            ".self_attn.kv_b_proj" in prefix):
                        return False
                    # Shared experts are both quality- and decode-sensitive.
                    if (_QUALITY_EXCLUDE_SHARED and
                            ".mlp.shared_experts." in prefix):
                        return False
                    # Keep the two dense MLP blocks immediately after layer 0
                    # on the dequantized BF16 execution path.
                    if (_QUALITY_EXCLUDE_EARLY_MLP and (
                            prefix.startswith("layers.1.mlp.") or
                            prefix.startswith("layers.2.mlp."))):
                        return False
                    # Protect q_a/q_b/o in the final 16 transformer layers.
                    head, _, suffix = prefix.partition(".self_attn.")
                    if (_QUALITY_EXCLUDE_FINAL_ATTN and
                            suffix in {"q_a_proj", "q_b_proj", "o_proj"}):
                        try:
                            layer_idx = int(head.split(".")[1])
                        except (IndexError, ValueError):
                            layer_idx = -1
                        if layer_idx >= 63:
                            return False
                    return True

                _allow = {p for p in _allow if _quality_stack_allows_mxfp8(p)}
                print(
                    "[hybrid_loader] quality stack MXFP8 filter: "
                    f"kept={len(_allow)} removed={_allow_before - len(_allow)} "
                    "flags="
                    f"kv_b={int(_QUALITY_EXCLUDE_KV_B)},"
                    f"shared={int(_QUALITY_EXCLUDE_SHARED)},"
                    f"early_mlp={int(_QUALITY_EXCLUDE_EARLY_MLP)},"
                    f"final_attn={int(_QUALITY_EXCLUDE_FINAL_ATTN)}",
                    flush=True,
                )

            def _mx_norm(p):
                i = p.find("layers.")
                return p[i:] if i >= 0 else p

            _base = mod.ModelOptQuantConfigBase
            if not getattr(_base, "_hyb_mxfp8_overlay", False):
                _orig_gqm = _base.get_quant_method
                _n_over = [0]

                def get_quant_method(self, layer, prefix):
                    m = _orig_gqm(self, layer, prefix)
                    if (type(m) is UnquantizedLinearMethod
                            and isinstance(layer, LinearBase)
                            and not isinstance(layer, ParallelLMHead)
                            and _mx_norm(prefix) in _allow):
                        _n_over[0] += 1
                        if _n_over[0] <= 4 or _n_over[0] % 128 == 0:
                            print(f"[hybrid_loader] mxfp8 overlay #{_n_over[0]}: "
                                  f"{prefix}", flush=True)
                        return Mxfp8OnlineLinearMethod()
                    return m

                _base.get_quant_method = get_quant_method
                _base._hyb_mxfp8_overlay = True
                print(f"[hybrid_loader] mxfp8 ne-tier overlay armed "
                      f"({len(_allow)} module prefixes)", flush=True)
        except Exception as e:
            import traceback
            print("[hybrid_loader] mxfp8 overlay FAILED:", e, flush=True)
            traceback.print_exc()


class _Hook(importlib.abc.MetaPathFinder):
    T = "vllm.model_executor.layers.quantization.modelopt"
    def find_spec(self, name, path, target=None):
        if name == self.T and not getattr(self, "_d", False):
            self._d = True
            spec = importlib.util.find_spec(name)
            if spec:
                real_exec = spec.loader.exec_module
                def ex(module):
                    real_exec(module)
                    try:
                        _patch(module)
                    except Exception as e:
                        import traceback
                        print("[hybrid_loader] patch FAILED:", e, flush=True)
                        traceback.print_exc()
                spec.loader.exec_module = ex
            return spec
        return None


if not any(isinstance(f, _Hook) for f in sys.meta_path):
    sys.meta_path.insert(0, _Hook())
    print("[hybrid_loader] import hook armed", flush=True)
