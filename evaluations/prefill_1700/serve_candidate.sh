#!/usr/bin/env bash
# GLM-5.2-NVFP4-TR3-Hybrid serving: hot-64 NVFP4 (b12x) + tail-192 EXL3 trellis 3.0bpw
# (exl3_moe fused kernels) on verdictai/glm52-nvfp4-kv:v2. Modeled on
# zz_serve_glm52_mtp_PROD_9300.sh (the validated nvfp4-KV + DCP4 + MTP lane: that
# lane needs NO extra vllm mounts on v2) + the TR3 overlay mounts.
#
# Row-D arithmetic this boot must verify (log + BOOT_REPORT):
#   resident weights ~78 GiB/GPU, KV pool >= 1M tokens
#   at KV=nvfp4_ds_mla, util 0.968, maxlen 1,048,576, TP4/DCP4, MTP-3.
set -euo pipefail
IMAGE="${IMAGE:-verdictai/glm52-nvfp4-kv-a2a:v1}"
NAME="${NAME:-glm52-tr3-9300}"
MODELS_DIR="/home/brandonmusic/models"
MODEL_DIR_HOST="$MODELS_DIR/GLM-5.2-NVFP4-TR3-Hybrid"
MODEL="/models-archive/GLM-5.2-NVFP4-TR3-Hybrid"
SERVED="GLM-5.2"; PORT="9300"
SITE=/opt/venv/lib/python3.12/site-packages
OVL="${OVL:-/home/brandonmusic/klc-linux/glm52_hybrid_opt/local_integration}"
TP="${TP:-4}"; DCP="${DCP:-4}"; DCP_ILV="${DCP_ILV:-64}"; MTP_N="${MTP_N:-3}"   # MTP-3 = the 1M-pool lane default (roadmap: MTP-5 loses here)
UTIL="${UTIL:-0.968}"; MAXLEN="${MAXLEN:-1048576}"; MAX_SEQS="${MAX_SEQS:-8}"; MAX_BATCHED="${MAX_BATCHED:-2048}"; GRAPH_CAP="${GRAPH_CAP:-32}"
KV="${KV:-nvfp4_ds_mla}"; ATTN="${ATTN:-B12X_MLA_SPARSE}"; CUSTOM_OPS="${CUSTOM_OPS:-all}"; CG_MODE="${CG_MODE:-FULL_AND_PIECEWISE}"
DCP_BACKEND="${DCP_BACKEND:-ag_rs}"; A2A="${A2A:-1}"; A2A_MAX_TOKENS="${A2A_MAX_TOKENS:-64}"
TR3="${TR3:-1}"                      # HYBRID_EXL3_TR3; 0 = baseline loader behavior (will NOT load this ckpt)
TR3_CHUNK="${TR3_CHUNK:-128}"        # rows per exl3_moe launch == fused temp rows/expert
IDX="FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS"
CACHE="/home/brandonmusic/.cache/glm52-tr3"; mkdir -p "$CACHE/jit"
SPEC_JSON='{"method":"mtp","num_speculative_tokens":'"$MTP_N"',"draft_sample_method":"probabilistic"}'
echo "IMAGE=$IMAGE KV=$KV TP=$TP DCP=$DCP MTP=$MTP_N util=$UTIL maxlen=$MAXLEN tr3=$TR3 chunk=$TR3_CHUNK port=$PORT"

PC_FLAG="--enable-prefix-caching"; [ "${PREFIX_CACHE:-1}" = "0" ] && PC_FLAG="--no-enable-prefix-caching"
echo "== preflight =="
# --- GATE 0 (owner sequencing order 2026-07-12): Experiment E owns the GPUs and
# :9200 until contradiction_reports/EXPE_PROGRAM_REPORT.md exists AND the expe
# driver has exited. PID-file liveness only (kill -0) - NEVER pattern kills or
# pgrep -f (self-match hazard). REMOVE THIS BLOCK after the first successful
# TR3 verification boot.
EXPE_REPORT="/home/brandonmusic/klc-linux/contradiction_reports/EXPE_PROGRAM_REPORT.md"
[ -f "$EXPE_REPORT" ] || { echo "FATAL E-gate: $EXPE_REPORT absent - Experiment E not finalized; no TR3 GPU use permitted"; exit 1; }
for pf in /home/brandonmusic/klc-linux/*.pid; do
  [ -f "$pf" ] || continue
  pid=$(tr -cd '0-9' < "$pf")
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    echo "FATAL E-gate: live driver PID $pid ($pf) - Experiment E still running"; exit 1
  fi
done

# --- GATE 1: never touch prod; GPUs must be idle (prod dsv4-9200 owns them by default)
docker ps --format '{{.Names}}' | grep -qx dsv4-9200-prod && { echo "FATAL prod dsv4-9200-prod is up - GPUs are owned by prod, refusing"; exit 1; }
BUSY=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 2048 {n++} END {print n+0}')
[ "$BUSY" -eq 0 ] || { echo "FATAL GPU-idle gate: $BUSY GPU(s) have >2GiB resident - not idle"; exit 1; }
ss -ltn 2>/dev/null | grep -q ":$PORT " && { echo "FATAL port $PORT already has a listener"; exit 1; }

# --- GATE 2: image, model, overlay files
docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "FATAL image missing: $IMAGE"; exit 1; }
[ -f "$MODEL_DIR_HOST/config.json" ] && [ -f "$MODEL_DIR_HOST/tier_bitmap.json" ] && [ -f "$MODEL_DIR_HOST/MANIFEST.sha256" ] || { echo "FATAL model incomplete: $MODEL_DIR_HOST"; exit 1; }
for f in "$OVL/overlay/hybrid_loader.py" "$OVL/wheels_unpacked/exllamav3_ext.cpython-312-x86_64-linux-gnu.so" "$OVL/shim/libexl3_torch212_compat.so"; do
  [ -f "$f" ] || { echo "FATAL overlay file missing: $f"; exit 1; }
done

# --- GATE 3: MANIFEST.sha256 - EVERY file hash must match before first boot.
# Full pass ~351GB; a passed run drops a marker keyed to the manifest's own
# sha256 so later boots skip the re-hash. Delete the marker to force re-verify.
MARK="$MODEL_DIR_HOST/.manifest_verified"
MSUM=$(sha256sum "$MODEL_DIR_HOST/MANIFEST.sha256" | cut -d' ' -f1)
if [ -f "$MARK" ] && [ "$(cat "$MARK")" = "$MSUM" ]; then
  echo "manifest: previously verified (marker matches manifest sha $MSUM)"
else
  echo "manifest: verifying every file (sha256sum -c, ~351GB, be patient)"
  ( cd "$MODEL_DIR_HOST" && sha256sum -c MANIFEST.sha256 --quiet ) || { echo "FATAL MANIFEST.sha256 verification FAILED - do not boot this checkpoint"; exit 1; }
  N_MAN=$(wc -l < "$MODEL_DIR_HOST/MANIFEST.sha256")
  echo "manifest: all $N_MAN files verified OK"
  echo "$MSUM" > "$MARK"
fi

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --gpus all --runtime nvidia --ipc host --shm-size 32g --network host --init \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$MODELS_DIR":/models-archive:ro -v "$CACHE":/cache \
  -v "$OVL/overlay/hybrid_loader.py":$SITE/hybrid_loader.py:ro \
  -v "$OVL/wheels_unpacked/exllamav3_ext.cpython-312-x86_64-linux-gnu.so":$SITE/exllamav3_ext.cpython-312-x86_64-linux-gnu.so:ro \
  -v "$OVL/shim/libexl3_torch212_compat.so":$SITE/libexl3_torch212_compat.so:ro \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3 -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e CUDA_DEVICE_MAX_CONNECTIONS=32 -e CUTE_DSL_ARCH=sm_120a -e OMP_NUM_THREADS=16 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e SAFETENSORS_FAST_GPU=1 \
  -e NCCL_IB_DISABLE=1 -e NCCL_P2P_LEVEL=SYS -e NCCL_PROTO=LL,LL128,Simple \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 -e VLLM_USE_B12X_FP8_GEMM=1 -e VLLM_USE_B12X_MOE=1 -e VLLM_USE_B12X_SPARSE_INDEXER=1 -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e VLLM_ENABLE_PCIE_ALLREDUCE=1 -e VLLM_PCIE_ALLREDUCE_BACKEND=b12x -e VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE=64KB \
  -e VLLM_PCIE_ONESHOT_FUSED_ADD_RMS_NORM_MAX_SIZE=84KB -e VLLM_PCIE_DMA_FP8=ring -e B12X_PCIE_DMA_FP8=ring \
  -e B12X_DENSE_SPLITK_TURBO=1 -e B12X_W4A16_TC_DECODE=1 -e B12X_MOE_FORCE_A16=1 \
  -e VLLM_USE_AOT_COMPILE=1 -e VLLM_USE_BREAKABLE_CUDAGRAPH=0 -e VLLM_USE_FUSED_MOE_GROUPED_TOPK=1 \
  -e VLLM_USE_B12X_MHC=1 -e B12X_MHC_MAX_TOKENS=16384 -e VLLM_USE_B12X_WO_PROJECTION=1 -e B12X_MLA_SM120_UNIFIED=1 -e USES_B12X=True \
  -e HYBRID_TIER=both -e HYBRID_KEPT=b12x_nf3 -e HYBRID_NF3=b12x_nf3 -e HYBRID_B12X_MAX_TOKENS="$MAX_BATCHED" -e HYBRID_MXFP8_NATIVE=1 \
  -e HYBRID_EXL3_TR3="$TR3" -e HYBRID_TR3_CHUNK="$TR3_CHUNK" \
  -e HYBRID_TR3_DQ_THRESHOLD="${HYBRID_TR3_DQ_THRESHOLD:-0}" \
  -e HYBRID_PROFILE="${HYBRID_PROFILE:-0}" \
  -e VLLM_CACHE_DIR=/cache/jit/vllm -e TRITON_CACHE_DIR=/cache/jit/triton -e TORCH_EXTENSIONS_DIR=/cache/jit/torch_extensions \
  -e TORCHINDUCTOR_CACHE_DIR=/cache/jit/torchinductor -e FLASHINFER_WORKSPACE_BASE=/cache/jit/flashinfer -e XDG_CACHE_HOME=/cache/jit \
  -e GLM52_INDEX_TOPK_PATTERN="$IDX" \
  -e VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE=1 -e VLLM_B12X_MLA_SPEC_DECODE_MAX_Q=8 \
  -e VLLM_USE_B12X_DCP_A2A="$A2A" -e VLLM_DCP_A2A_MAX_TOKENS="$A2A_MAX_TOKENS" -e VLLM_DCP_A2A_LARGE_BACKEND=ag_rs \
  "$IMAGE" /bin/bash -lc "
    set -euo pipefail
    unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE VLLM_B12X_MLA_EXTEND_MAX_CHUNKS
    exec vllm serve '$MODEL' --served-model-name '$SERVED' 'qwen3.5-397b-nvfp4' --host 0.0.0.0 --port '$PORT' --trust-remote-code \
      --tensor-parallel-size '$TP' --decode-context-parallel-size '$DCP' --dcp-comm-backend '$DCP_BACKEND' --dcp-kv-cache-interleave-size '$DCP_ILV' \
      --quantization modelopt_fp4 --kv-cache-dtype '$KV' --attention-backend '$ATTN' --moe-backend b12x --load-format safetensors \
      --compilation-config '{\"cudagraph_mode\":\"$CG_MODE\",\"custom_ops\":[\"$CUSTOM_OPS\"],\"pass_config\":{\"fuse_allreduce_rms\":true}}' \
      --gpu-memory-utilization '$UTIL' --max-model-len '$MAXLEN' --max-num-seqs '$MAX_SEQS' --max-num-batched-tokens '$MAX_BATCHED' \
      --max-cudagraph-capture-size '$GRAPH_CAP' --async-scheduling --enable-chunked-prefill $PC_FLAG \
      --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45 \
      --hf-overrides '{\"use_index_cache\":true,\"index_topk_pattern\":\"$IDX\"}' \
      --speculative-config '$SPEC_JSON'
  "
echo "Launched $NAME (TR3 tail + MTP-$MTP_N, KV=$KV, DCP=$DCP) port=$PORT"
echo "First-boot verification: bash $OVL/tr3_boot_verify.sh   (logs weights-GiB/GPU + KV pool tokens)"
