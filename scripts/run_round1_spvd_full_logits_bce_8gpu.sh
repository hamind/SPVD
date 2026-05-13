#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/vepfs/code/SPVD}"
CONDA_ENV="${CONDA_ENV:-openclip}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
MASTER_PORT="${MASTER_PORT:-29603}"
EVAL_MASTER_PORT="${EVAL_MASTER_PORT:-29703}"
LOG_DIR="${LOG_DIR:-$ROOT/outputs/launch_logs/round1_spvd_full_logits_bce_8gpu}"

OVERWRITE=0
DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --overwrite) OVERWRITE=1 ;;
    --dry-run) DRY_RUN=1 ;;
    -h|--help)
      echo "Usage: bash scripts/run_round1_spvd_full_logits_bce_8gpu.sh [--dry-run] [--overwrite]"
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      echo "Usage: bash scripts/run_round1_spvd_full_logits_bce_8gpu.sh [--dry-run] [--overwrite]" >&2
      exit 2
      ;;
  esac
done

export CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS
export MKL_NUM_THREADS
export PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"

EXPERIMENTS=(
  spvd_full_logits_bce_w003
  spvd_full_logits_bce_w001
)

CONFIGS=(
  configs/experiments/spvd_full_logits_bce_w003_8gpu.yaml
  configs/experiments/spvd_full_logits_bce_w001_8gpu.yaml
)

BENCHMARK_CONFIGS=(
  configs/benchmark_eval_spvd_full_logits_bce_w003.yaml
  configs/benchmark_eval_spvd_full_logits_bce_w001.yaml
)

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

run_benchmark() {
  local name="$1"
  local bench_config="$2"
  local ckpt="$3"
  local stamp
  stamp="$(date +%Y%m%d_%H%M%S)"
  local log_file="$LOG_DIR/${name}_benchmark_${stamp}.log"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[$name] dry-run requested; skip full benchmark because no final checkpoint is produced."
    return 0
  fi
  if [[ ! -f "$ckpt" ]]; then
    echo "[$name] missing final checkpoint for benchmark: $ckpt" >&2
    return 3
  fi

  ln -sfn "$(basename "$log_file")" "$LOG_DIR/${name}_benchmark_latest.log"
  {
    echo "===== SPVD round1 8GPU benchmark launch ====="
    echo "started_at=$(date -Is)"
    echo "host=$(hostname)"
    echo "name=$name"
    echo "benchmark_config=$bench_config"
    echo "checkpoint=$ckpt"
    echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
    echo "nproc_per_node=$NPROC_PER_NODE"
    echo "eval_master_port=$EVAL_MASTER_PORT"
    printf 'command='
    printf '%q ' torchrun --standalone "--nnodes=1" "--nproc_per_node=$NPROC_PER_NODE" "--master_port=$EVAL_MASTER_PORT" -m eval.benchmarks --config "$bench_config"
    printf '\n'
    echo "============================================="
    set +e
    torchrun --standalone --nnodes=1 "--nproc_per_node=$NPROC_PER_NODE" "--master_port=$EVAL_MASTER_PORT" -m eval.benchmarks --config "$bench_config"
    status=$?
    set -e
    echo "===== SPVD round1 8GPU benchmark finished ====="
    echo "finished_at=$(date -Is)"
    echo "exit_code=$status"
    exit "$status"
  } 2>&1 | tee "$log_file"
}

run_one() {
  local name="$1"
  local config="$2"
  local bench_config="$3"
  local out_dir="$ROOT/outputs/experiments/$name"
  local ckpt="$out_dir/$name/checkpoints/epoch_final.pt"
  local stamp
  stamp="$(date +%Y%m%d_%H%M%S)"
  local log_file="$LOG_DIR/${name}_${stamp}.log"

  if [[ "$DRY_RUN" -eq 0 && "$OVERWRITE" -ne 1 && -f "$ckpt" ]]; then
    echo "[$name] checkpoint exists: $ckpt"
    echo "[$name] skip to avoid overwriting. Pass --overwrite to run again."
    return 0
  fi

  mkdir -p "$out_dir"
  ln -sfn "$(basename "$log_file")" "$LOG_DIR/${name}_latest.log"
  {
    echo "===== SPVD round1 8GPU launch ====="
    echo "started_at=$(date -Is)"
    echo "host=$(hostname)"
    echo "name=$name"
    echo "config=$config"
    echo "output=$out_dir"
    echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
    echo "nproc_per_node=$NPROC_PER_NODE"
    echo "master_port=$MASTER_PORT"
    echo "dry_run=$DRY_RUN"
    printf 'command='
    printf '%q ' torchrun --standalone "--nnodes=1" "--nproc_per_node=$NPROC_PER_NODE" "--master_port=$MASTER_PORT" -m main --config "$config" --logs "$out_dir" --name "$name"
    [[ "$DRY_RUN" -eq 1 ]] && printf '%q ' --dry-run
    printf '\n'
    echo "==================================="
    set +e
    if [[ "$DRY_RUN" -eq 1 ]]; then
      torchrun --standalone --nnodes=1 "--nproc_per_node=$NPROC_PER_NODE" "--master_port=$MASTER_PORT" -m main --config "$config" --logs "$out_dir" --name "$name" --dry-run
      status=$?
    else
      torchrun --standalone --nnodes=1 "--nproc_per_node=$NPROC_PER_NODE" "--master_port=$MASTER_PORT" -m main --config "$config" --logs "$out_dir" --name "$name"
      status=$?
    fi
    set -e
    echo "===== SPVD round1 8GPU finished ====="
    echo "finished_at=$(date -Is)"
    echo "exit_code=$status"
    exit "$status"
  } 2>&1 | tee "$log_file"
  run_benchmark "$name" "$bench_config" "$ckpt"
}

for idx in "${!EXPERIMENTS[@]}"; do
  run_one "${EXPERIMENTS[$idx]}" "${CONFIGS[$idx]}" "${BENCHMARK_CONFIGS[$idx]}"
done
