#!/usr/bin/env bash
set -uo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_benchmark_eval.sh [eval.benchmarks args...]

Examples:
  bash scripts/run_benchmark_eval.sh --dry_run --limit 8 --models clip_vit_b32_openai,openclip_vit_b32_laion2b
  CONFIG=configs/benchmark_eval_pretrain_models.yaml bash scripts/run_benchmark_eval.sh --dry_run --limit 8

Environment:
  ROOT=/vepfs/code/SPVD
  CONFIG=configs/benchmark_eval.yaml
  CONDA_ENV=openclip
  CUDA_VISIBLE_DEVICES=0,1
  NPROC_PER_NODE=2
  OMP_NUM_THREADS=13
  MKL_NUM_THREADS=13
  LOG_DIR=$ROOT/outputs/launch_logs
  RUN_NAME=benchmark_eval
  INSTALL_BENCHMARK_DEPS=1
EOF
}

for arg in "$@"; do
  case "$arg" in
    -h|--help)
      usage
      exit 0
      ;;
  esac
done

ROOT="${ROOT:-/vepfs/code/SPVD}"
CONFIG="${CONFIG:-configs/benchmark_eval.yaml}"
CONDA_ENV="${CONDA_ENV:-openclip}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-13}"
MKL_NUM_THREADS="${MKL_NUM_THREADS:-13}"
LOG_DIR="${LOG_DIR:-$ROOT/outputs/launch_logs}"
RUN_NAME="${RUN_NAME:-benchmark_eval}"
INSTALL_BENCHMARK_DEPS="${INSTALL_BENCHMARK_DEPS:-1}"

export CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS
export MKL_NUM_THREADS
export PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$LOG_DIR"
cd "$ROOT" || exit 10

if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  # Non-interactive shells do not always have conda initialized.
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "$CONDA_ENV"
else
  echo "conda was not found; expecting $CONDA_ENV to already be active." >&2
fi

has_config=0
for arg in "$@"; do
  if [[ "$arg" == "--config" || "$arg" == --config=* ]]; then
    has_config=1
    break
  fi
done

if [[ "$#" -eq 0 ]]; then
  set -- --config "$CONFIG" --dry_run --limit 8 --models clip_vit_b32_openai,openclip_vit_b32_laion2b
elif [[ "$has_config" -eq 0 ]]; then
  set -- --config "$CONFIG" "$@"
fi

stamp="$(date +%Y%m%d_%H%M%S)"
log_file="$LOG_DIR/${RUN_NAME}_${stamp}.log"
latest_log="$LOG_DIR/${RUN_NAME}_latest.log"
pid_file="$LOG_DIR/${RUN_NAME}_${stamp}.pid"
latest_pid="$LOG_DIR/${RUN_NAME}_latest.pid"

echo "$$" > "$pid_file"
ln -sfn "$(basename "$pid_file")" "$latest_pid"

ensure_python_package() {
  local module="$1"
  local package="$2"

  if python -c "import ${module}" >/dev/null 2>&1; then
    echo "[deps] ${module} available"
    return 0
  fi
  if [[ "$INSTALL_BENCHMARK_DEPS" != "1" ]]; then
    echo "[deps] ${module} missing; set INSTALL_BENCHMARK_DEPS=1 to install ${package}" >&2
    return 1
  fi

  echo "[deps] installing ${package}"
  python -m pip install "$package"
  python -c "import ${module}"
}

set +e
(
  echo "===== SPVD benchmark launch ====="
  echo "started_at=$(date -Is)"
  echo "host=$(hostname)"
  echo "root=$ROOT"
  echo "conda_env=${CONDA_DEFAULT_ENV:-unknown}"
  echo "python=$(command -v python || true)"
  echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
  echo "nproc_per_node=$NPROC_PER_NODE"
  echo "omp_num_threads=$OMP_NUM_THREADS"
  echo "mkl_num_threads=$MKL_NUM_THREADS"
  echo "pythonpath=$PYTHONPATH"
  echo "log_file=$log_file"
  printf 'command='
  printf '%q ' torchrun "--nproc_per_node=$NPROC_PER_NODE" -m eval.benchmarks "$@"
  printf '\n'
  echo "install_benchmark_deps=$INSTALL_BENCHMARK_DEPS"
  echo "================================="
  ensure_python_package sentencepiece sentencepiece
  ensure_python_package google.protobuf protobuf
  torchrun "--nproc_per_node=$NPROC_PER_NODE" -m eval.benchmarks "$@"
  status=$?
  echo "===== SPVD benchmark finished ====="
  echo "finished_at=$(date -Is)"
  echo "exit_code=$status"
  exit "$status"
) 2>&1 | tee "$log_file"
status=${PIPESTATUS[0]}
set -e

ln -sfn "$(basename "$log_file")" "$latest_log"
rm -f "$pid_file"
rm -f "$latest_pid"

echo "LOG_FILE=$log_file"
echo "EXIT_CODE=$status"
exit "$status"
