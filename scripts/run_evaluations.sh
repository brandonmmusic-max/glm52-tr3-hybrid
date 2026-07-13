#!/usr/bin/env bash
set -euo pipefail

BENCH_DIR="${BENCH_DIR:-/home/brandonmusic/llm-inference-bench}"
RESULT_DIR="${RESULT_DIR:-/home/brandonmusic/klc-linux/glm52_hybrid_opt/public_evaluations/mtp3_u964_c4}"
MODEL="${MODEL:-GLM-5.2}"
HOST="${HOST:-localhost}"
PORT="${PORT:-9300}"
mkdir -p "$RESULT_DIR"

run_profile() {
  local profile="$1"
  local slug="$2"
  shift 2
  local output="$RESULT_DIR/${slug}.json"
  local terminal="$RESULT_DIR/${slug}.tui"
  local clean="$RESULT_DIR/${slug}.tui.txt"
  is_complete() {
    python3 - "$output" <<'PY'
import json
import sys
from pathlib import Path

try:
    report = json.loads(Path(sys.argv[1]).read_text())
    metadata = report["metadata"]
    summary = report["selected_summary"]
    requested = int(metadata["requested_runs"])
    completed = int(summary["completed"])
    errors = int(summary["errors"])
    interrupted = bool(metadata.get("interrupted", False))
    raise SystemExit(0 if requested > 0 and completed == requested and errors == 0 and not interrupted else 1)
except Exception:
    raise SystemExit(1)
PY
  }
  clean_terminal() {
    python3 - "$terminal" "$clean" <<'PY'
import re
import sys
from pathlib import Path

source = Path(sys.argv[1]).read_text(errors="replace")
source = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", source)
source = source.replace("\r", "")
Path(sys.argv[2]).write_text(source)
PY
  }

  if is_complete; then
    echo "SKIP complete profile=$profile output=$output"
    return 0
  fi

  for attempt in 1 2 3; do
    echo "START profile=$profile attempt=$attempt"
    rm -f "$output" "$terminal" "$clean"
    if printf 'n\n' | script -qefc \
      "cd '$BENCH_DIR' && python3 llm_decode_bench.py --host '$HOST' --port '$PORT' --model '$MODEL' --test-profile '$profile' --completion-stats-concurrency 4 --display-mode live --refresh-rate 0.2 --completion-stats-save-text --output '$output' $*" \
      "$terminal"; then
      clean_terminal
      if is_complete; then
        echo "DONE profile=$profile output=$output"
        return 0
      fi
    fi
    echo "RETRY profile=$profile attempt=$attempt" >&2
    sleep 30
  done
  echo "FAILED profile=$profile after=3" >&2
  return 1
}

run_profile lavd lavd_c4_r10 --completion-stats-runs 10
run_profile estonia estonia_c4_r10 --completion-stats-runs 10
run_profile gpqa-diamond gpqa_diamond_c4_full
run_profile gsm8k gsm8k_c4_full
