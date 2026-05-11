#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
NAME="spvd_full"
CONFIG="$ROOT/configs/experiments/spvd_full.yaml"
OUT_DIR="$ROOT/outputs/experiments/$NAME"
LOG_DIR="$ROOT/outputs/logs/ddp_experiments"
LOG_FILE="$LOG_DIR/$NAME.log"
FINAL_CKPT="$OUT_DIR/$NAME/checkpoints/epoch_final.pt"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES NPROC_PER_NODE OMP_NUM_THREADS PYTHONPATH

TORCHRUN="${TORCHRUN:-/root/miniconda3/envs/openclip/bin/torchrun}"

OVERWRITE=0
EXTRA_ARGS=()
for arg in "$@"; do
  case "$arg" in
    --overwrite)
      OVERWRITE=1
      ;;
    -h|--help)
      echo "Usage: bash $(basename "$0") [--overwrite] [extra main.py args...]"
      echo "Example: bash $(basename "$0") --max-steps 100"
      exit 0
      ;;
    *)
      EXTRA_ARGS+=("$arg")
      ;;
  esac
done

if [[ -f "$FINAL_CKPT" && "$OVERWRITE" -ne 1 ]]; then
  echo "[$NAME] checkpoint exists: $FINAL_CKPT"
  echo "[$NAME] skip to avoid overwriting. Pass --overwrite to run again."
  exit 0
fi

mkdir -p "$OUT_DIR" "$LOG_DIR"
cd "$ROOT"

echo "[$NAME] root=$ROOT"
echo "[$NAME] config=$CONFIG"
echo "[$NAME] output=$OUT_DIR"
echo "[$NAME] log=$LOG_FILE"
echo "[$NAME] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[$NAME] NPROC_PER_NODE=$NPROC_PER_NODE"

"$TORCHRUN" --standalone --nproc_per_node="$NPROC_PER_NODE" -m main \
  --config "$CONFIG" \
  --logs "$OUT_DIR" \
  --name "$NAME" \
  "${EXTRA_ARGS[@]}" 2>&1 | tee "$LOG_FILE"
