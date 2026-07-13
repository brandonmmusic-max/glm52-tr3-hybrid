FROM verdictai/glm52-nvfp4-kv-a2a:v1

LABEL org.opencontainers.image.title="GLM-5.2 NVFP4/TR3 Hybrid MTP3"
LABEL org.opencontainers.image.description="DCP4/A2A GLM-5.2 hybrid inference image with NVFP4 DS-MLA KV and EXL3 TR3 fast-prefill overlay"
LABEL org.opencontainers.image.source="https://github.com/brandonmmusic-max/glm52-tr3-hybrid"
LABEL org.opencontainers.image.documentation="https://huggingface.co/brandonmusic/GLM-5.2-NVFP4-TR3-Hybrid"

COPY overlay/hybrid_loader.py /opt/venv/lib/python3.12/site-packages/hybrid_loader.py
COPY overlay/exllamav3_ext.cpython-312-x86_64-linux-gnu.so /opt/venv/lib/python3.12/site-packages/exllamav3_ext.cpython-312-x86_64-linux-gnu.so
COPY overlay/libexl3_torch212_compat.so /opt/venv/lib/python3.12/site-packages/libexl3_torch212_compat.so

ENV B12X_MHC_MAX_TOKENS=16384 \
    B12X_MLA_SM120_UNIFIED=1 \
    B12X_PCIE_DMA_FP8=ag \
    HYBRID_B12X_MAX_TOKENS=2048 \
    HYBRID_EXL3_TR3=1 \
    HYBRID_PROFILE=0 \
    HYBRID_QUALITY_DETERMINISTIC_ACCUM=1 \
    HYBRID_QUALITY_EXCLUDE_EARLY_MLP=0 \
    HYBRID_QUALITY_EXCLUDE_FINAL_ATTN=0 \
    HYBRID_QUALITY_EXCLUDE_KV_B=1 \
    HYBRID_QUALITY_EXCLUDE_SHARED=0 \
    HYBRID_QUALITY_FP32_ROUTER=1 \
    HYBRID_QUALITY_STACK=1 \
    HYBRID_TR3_CHUNK=128 \
    HYBRID_TR3_DQ_THRESHOLD=128 \
    TORCHINDUCTOR_CACHE_DIR=/cache/jit/quality-stack/torchinductor \
    TRITON_CACHE_DIR=/cache/jit/quality-stack/triton \
    USES_B12X=True \
    VLLM_B12X_MLA_SPEC_DECODE_MAX_Q=8 \
    VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE=1 \
    VLLM_CACHE_DIR=/cache/jit/quality-stack/vllm \
    VLLM_DCP_A2A_LARGE_BACKEND=ag_rs \
    VLLM_DCP_A2A_MAX_TOKENS=64 \
    VLLM_ENABLE_PCIE_ALLREDUCE=1 \
    VLLM_PCIE_ALLREDUCE_BACKEND=b12x \
    VLLM_PCIE_DMA_FP8=ag \
    VLLM_PCIE_ONESHOT_FUSED_ADD_RMS_NORM_MAX_SIZE=84KB \
    VLLM_USE_AOT_COMPILE=1 \
    VLLM_USE_B12X_DCP_A2A=1 \
    VLLM_USE_B12X_MHC=1 \
    VLLM_USE_B12X_WO_PROJECTION=1 \
    VLLM_USE_BREAKABLE_CUDAGRAPH=0 \
    VLLM_USE_FLASHINFER_SAMPLER=1 \
    VLLM_USE_FUSED_MOE_GROUPED_TOPK=1 \
    XDG_CACHE_HOME=/cache/jit/quality-stack/xdg \
    MODEL_PATH=/models-archive/GLM-5.2-NVFP4-TR3-Hybrid \
    GPU_MEMORY_UTILIZATION=0.964

CMD ["/bin/bash", "-lc", "set -euo pipefail; unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE VLLM_B12X_MLA_EXTEND_MAX_CHUNKS; exec vllm serve \"${MODEL_PATH}\" --served-model-name GLM-5.2 qwen3.5-397b-nvfp4 --host 0.0.0.0 --port 9300 --trust-remote-code --tensor-parallel-size 4 --decode-context-parallel-size 4 --dcp-comm-backend a2a --dcp-kv-cache-interleave-size 64 --quantization modelopt_fp4 --kv-cache-dtype nvfp4_ds_mla --attention-backend B12X_MLA_SPARSE --moe-backend b12x --load-format safetensors --compilation-config '{\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"custom_ops\":[\"all\"],\"pass_config\":{\"fuse_allreduce_rms\":true}}' --gpu-memory-utilization \"${GPU_MEMORY_UTILIZATION}\" --max-model-len 262144 --max-num-seqs 4 --max-num-batched-tokens 2048 --max-cudagraph-capture-size 8 --async-scheduling --enable-chunked-prefill --enable-prefix-caching --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45 --hf-overrides '{\"use_index_cache\":true,\"index_topk_pattern\":\"FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS\"}' --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":3,\"draft_sample_method\":\"probabilistic\"}'"]
