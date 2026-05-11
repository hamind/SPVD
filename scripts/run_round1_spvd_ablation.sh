#!/usr/bin/env bash
set -euo pipefail

# Round-1 SPVD ablations:
#   spvd_full
#   spvd_no_route
#   spvd_pos_only_route
#   spvd_neg_only_route
#   spvd_no_soft_cue

ROOT="/vepfs/code/SPVD"
LOG_DIR="$ROOT/outputs/logs/round1_spvd_ablation"
PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPATH

OVERWRITE=0
PARALLEL=0
for arg in "$@"; do
  case "$arg" in
    --overwrite) OVERWRITE=1 ;;
    --parallel) PARALLEL=1 ;;
    -h|--help)
      echo "Usage: bash scripts/run_round1_spvd_ablation.sh [--parallel] [--overwrite]"
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      echo "Usage: bash scripts/run_round1_spvd_ablation.sh [--parallel] [--overwrite]" >&2
      exit 2
      ;;
  esac
done

EXPERIMENTS=(
  spvd_full
  spvd_no_route
  spvd_pos_only_route
  spvd_neg_only_route
  spvd_no_soft_cue
)

CONFIGS=(
  configs/experiments/spvd_full.yaml
  configs/experiments/spvd_no_route.yaml
  configs/experiments/spvd_pos_only_route.yaml
  configs/experiments/spvd_neg_only_route.yaml
  configs/experiments/spvd_no_soft_cue.yaml
)

mkdir -p "$LOG_DIR"
cd "$ROOT"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate openclip
else
  echo "conda was not found on PATH; expecting the openclip environment to already be active." >&2
fi

run_one() {
  local name="$1"
  local config="$2"
  local out_dir="$ROOT/outputs/experiments/$name"
  local ckpt="$out_dir/$name/checkpoints/epoch_final.pt"
  local log_file="$LOG_DIR/$name.log"

  if [[ -f "$ckpt" && "$OVERWRITE" -ne 1 ]]; then
    echo "[$name] checkpoint exists: $ckpt"
    echo "[$name] skip to avoid overwriting. Pass --overwrite to run again."
    return 0
  fi

  mkdir -p "$out_dir"
  echo "[$name] config=$config"
  echo "[$name] output=$out_dir"
  python -m main --config "$config" --logs "$out_dir" --name "$name" 2>&1 | tee "$log_file"
}

pids=()
for idx in "${!EXPERIMENTS[@]}"; do
  if [[ "$PARALLEL" -eq 1 ]]; then
    run_one "${EXPERIMENTS[$idx]}" "${CONFIGS[$idx]}" &
    pids+=("$!")
  else
    run_one "${EXPERIMENTS[$idx]}" "${CONFIGS[$idx]}"
  fi
done

if [[ "$PARALLEL" -eq 1 ]]; then
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
fi
