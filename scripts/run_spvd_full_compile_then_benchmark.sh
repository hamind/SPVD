#!/usr/bin/env bash
set -euo pipefail

ROOT="/vepfs/code/SPVD"
RUN_NAME="${RUN_NAME:-spvd_full_config_8gpu_20260513_0930}"
TRAIN_CONFIG="${TRAIN_CONFIG:-$ROOT/configs/experiments/spvd_full.yaml}"
OUT_BASE="${OUT_BASE:-$ROOT/outputs/experiments/$RUN_NAME}"
LOG_DIR="${LOG_DIR:-$ROOT/outputs/logs/compile_then_benchmark}"
TRAIN_LOG="$LOG_DIR/${RUN_NAME}.train.log"
EVAL_LOG="$LOG_DIR/${RUN_NAME}.benchmark.log"
CHAIN_LOG="$LOG_DIR/${RUN_NAME}.chain.log"
FINAL_CKPT="$OUT_BASE/$RUN_NAME/checkpoints/epoch_final.pt"
BENCH_CONFIG="$ROOT/configs/benchmark_eval_${RUN_NAME}.yaml"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
EVAL_NPROC_PER_NODE="${EVAL_NPROC_PER_NODE:-8}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
TORCHRUN="${TORCHRUN:-/root/miniconda3/envs/openclip/bin/torchrun}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/openclip/bin/python}"
TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$ROOT/.cache/torchinductor/$RUN_NAME}"
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$ROOT/.cache/triton/$RUN_NAME}"

export CUDA_VISIBLE_DEVICES NPROC_PER_NODE OMP_NUM_THREADS PYTHONPATH
export TORCHINDUCTOR_CACHE_DIR TRITON_CACHE_DIR

mkdir -p "$LOG_DIR" "$OUT_BASE" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
cd "$ROOT"

exec >> "$CHAIN_LOG" 2>&1

echo "[chain] started_at=$(date -Is)"
echo "[chain] root=$ROOT"
echo "[chain] run_name=$RUN_NAME"
echo "[chain] train_config=$TRAIN_CONFIG"
echo "[chain] out_base=$OUT_BASE"
echo "[chain] final_ckpt=$FINAL_CKPT"
echo "[chain] cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
echo "[chain] nproc_per_node=$NPROC_PER_NODE"
echo "[chain] eval_nproc_per_node=$EVAL_NPROC_PER_NODE"
echo "[chain] torchinductor_cache_dir=$TORCHINDUCTOR_CACHE_DIR"
echo "[chain] triton_cache_dir=$TRITON_CACHE_DIR"

"$PYTHON_BIN" - <<'PY'
import os
from params import parse_args
args = parse_args(["--config", os.environ.get("TRAIN_CONFIG", "configs/experiments/spvd_full.yaml")])
print(
    "[chain] parsed_config "
    f"model={args.model} loss={args.loss_name} batch_size={args.batch_size} "
    f"workers={args.workers} torch_compile={args.torch_compile} "
    f"caption_same_image_mode={args.caption_same_image_mode} "
    f"branch_bce_weight={args.branch_bce_weight} "
    f"residual_variance_weight={args.residual_variance_weight}"
)
PY

echo "[train] launching compile training at $(date -Is)"
"$TORCHRUN" --standalone --nproc_per_node="$NPROC_PER_NODE" -m main \
  --config "$TRAIN_CONFIG" \
  --logs "$OUT_BASE" \
  --name "$RUN_NAME" \
  ${EXTRA_TRAIN_ARGS:-} > "$TRAIN_LOG" 2>&1
train_status=$?
echo "[train] finished_at=$(date -Is) status=$train_status"
if [[ "$train_status" -ne 0 ]]; then
  exit "$train_status"
fi

if [[ ! -f "$FINAL_CKPT" ]]; then
  echo "[train] missing final checkpoint: $FINAL_CKPT" >&2
  exit 10
fi
ls -lh "$FINAL_CKPT"

cat > "$BENCH_CONFIG" <<YAML
dataset_root: /vepfs/dataset/benchmark
model_root: /vepfs/model
output_dir: /vepfs/code/SPVD/outputs/benchmark_eval/$RUN_NAME
conda_env: openclip
device: cuda
dtype: bf16
batch_size: 64
num_workers: 8
distributed_eval: true
num_workers_per_gpu: 4
pin_memory: true
persistent_workers: true
prefetch_factor: 4
pair_chunk_size: 8192
perf_log_interval: 20
write_perf_jsonl: true
random_seed: 42
dry_run_limit: 8
compute_random_negative: true
save_figures: true

eval_mode:
  spvd_pairwise_mode: exact_cached
  retrieval_mode: global
  rerank_topk: 100
  score_chunk: 8192
  pair_chunk_size: 8192

benchmarks:
  enabled:
    - aro
    - sugarcrepe
    - sugarcrepe_pp
    - bivlc
    - winoground
  aro:
    root: /vepfs/dataset/benchmark/aro
    order_splits:
      - test
      - val
    order_max_words: 30
  sugarcrepe:
    root: /vepfs/dataset/benchmark/sugarcrepe
  sugarcrepe_pp:
    root: /vepfs/dataset/benchmark/sugarcrepe_pp
  bivlc:
    root: /vepfs/dataset/benchmark/bivlc
  winoground:
    root: /vepfs/dataset/benchmark/winoground

models:
  - name: $RUN_NAME
    model_type: spvd
    model_name: SPVD-ViT-B-16
    checkpoint_path: $FINAL_CKPT
    image_size: 224
YAML

echo "[benchmark] config=$BENCH_CONFIG"
echo "[benchmark] launching at $(date -Is)"
"$TORCHRUN" --standalone --nproc_per_node="$EVAL_NPROC_PER_NODE" -m eval.benchmarks \
  --config "$BENCH_CONFIG" \
  ${EXTRA_EVAL_ARGS:-} > "$EVAL_LOG" 2>&1
eval_status=$?
echo "[benchmark] finished_at=$(date -Is) status=$eval_status"
exit "$eval_status"
