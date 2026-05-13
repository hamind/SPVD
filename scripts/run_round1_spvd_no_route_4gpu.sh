#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/vepfs/code/SPVD}"
CONDA_ENV="${CONDA_ENV:-openclip}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
MASTER_PORT="${MASTER_PORT:-29604}"
EVAL_MASTER_PORT="${EVAL_MASTER_PORT:-29704}"
LOG_DIR="${LOG_DIR:-$ROOT/outputs/launch_logs/round1_spvd_no_route_4gpu}"

OVERWRITE=0
DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --overwrite) OVERWRITE=1 ;;
    --dry-run) DRY_RUN=1 ;;
    -h|--help)
      echo "Usage: bash scripts/run_round1_spvd_no_route_4gpu.sh [--dry-run] [--overwrite]"
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      echo "Usage: bash scripts/run_round1_spvd_no_route_4gpu.sh [--dry-run] [--overwrite]" >&2
      exit 2
      ;;
  esac
done

export CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS
export MKL_NUM_THREADS
export PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"

NAME="spvd_no_route_4gpu"
CONFIG="configs/experiments/spvd_no_route_4gpu.yaml"
BENCHMARK_CONFIG="configs/benchmark_eval_spvd_no_route_4gpu.yaml"
OUT_DIR="$ROOT/outputs/experiments/$NAME"
CKPT="$OUT_DIR/$NAME/checkpoints/epoch_final.pt"

mkdir -p "$LOG_DIR"
cd "$ROOT"

if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "$CONDA_ENV"
else
  echo "conda was not found on PATH; expecting $CONDA_ENV to already be active." >&2
fi

if [[ "$DRY_RUN" -eq 0 && "$OVERWRITE" -ne 1 && -f "$CKPT" ]]; then
  echo "[$NAME] checkpoint exists: $CKPT"
  echo "[$NAME] skip to avoid overwriting. Pass --overwrite to run again."
  exit 0
fi

mkdir -p "$OUT_DIR"
stamp="$(date +%Y%m%d_%H%M%S)"
log_file="$LOG_DIR/${NAME}_${stamp}.log"
ln -sfn "$(basename "$log_file")" "$LOG_DIR/${NAME}_latest.log"

{
  echo "===== SPVD round1 4GPU launch ====="
  echo "started_at=$(date -Is)"
  echo "host=$(hostname)"
  echo "name=$NAME"
  echo "config=$CONFIG"
  echo "output=$OUT_DIR"
  echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
  echo "nproc_per_node=$NPROC_PER_NODE"
  echo "master_port=$MASTER_PORT"
  echo "dry_run=$DRY_RUN"
  printf 'command='
  printf '%q ' torchrun --standalone "--nnodes=1" "--nproc_per_node=$NPROC_PER_NODE" "--master_port=$MASTER_PORT" -m main --config "$CONFIG" --logs "$OUT_DIR" --name "$NAME"
  [[ "$DRY_RUN" -eq 1 ]] && printf '%q ' --dry-run
  printf '\n'
  echo "==================================="
  set +e
  if [[ "$DRY_RUN" -eq 1 ]]; then
    torchrun --standalone --nnodes=1 "--nproc_per_node=$NPROC_PER_NODE" "--master_port=$MASTER_PORT" -m main --config "$CONFIG" --logs "$OUT_DIR" --name "$NAME" --dry-run
    status=$?
  else
    torchrun --standalone --nnodes=1 "--nproc_per_node=$NPROC_PER_NODE" "--master_port=$MASTER_PORT" -m main --config "$CONFIG" --logs "$OUT_DIR" --name "$NAME"
    status=$?
  fi
  set -e
  echo "===== SPVD round1 4GPU finished ====="
  echo "finished_at=$(date -Is)"
  echo "exit_code=$status"
  exit "$status"
} 2>&1 | tee "$log_file"

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "[$NAME] dry-run requested; skip full benchmark because no final checkpoint is produced."
  exit 0
fi
if [[ ! -f "$CKPT" ]]; then
  echo "[$NAME] missing final checkpoint for benchmark: $CKPT" >&2
  exit 3
fi

bench_stamp="$(date +%Y%m%d_%H%M%S)"
bench_log_file="$LOG_DIR/${NAME}_benchmark_${bench_stamp}.log"
ln -sfn "$(basename "$bench_log_file")" "$LOG_DIR/${NAME}_benchmark_latest.log"

{
  echo "===== SPVD round1 4GPU benchmark launch ====="
  echo "started_at=$(date -Is)"
  echo "host=$(hostname)"
  echo "name=$NAME"
  echo "benchmark_config=$BENCHMARK_CONFIG"
  echo "checkpoint=$CKPT"
  echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
  echo "nproc_per_node=$NPROC_PER_NODE"
  echo "eval_master_port=$EVAL_MASTER_PORT"
  printf 'command='
  printf '%q ' torchrun --standalone "--nnodes=1" "--nproc_per_node=$NPROC_PER_NODE" "--master_port=$EVAL_MASTER_PORT" -m eval.benchmarks --config "$BENCHMARK_CONFIG"
  printf '\n'
  echo "============================================="
  set +e
  torchrun --standalone --nnodes=1 "--nproc_per_node=$NPROC_PER_NODE" "--master_port=$EVAL_MASTER_PORT" -m eval.benchmarks --config "$BENCHMARK_CONFIG"
  status=$?
  set -e
  echo "===== SPVD round1 4GPU benchmark finished ====="
  echo "finished_at=$(date -Is)"
  echo "exit_code=$status"
  exit "$status"
} 2>&1 | tee "$bench_log_file"
