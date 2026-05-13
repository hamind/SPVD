#!/usr/bin/env bash
set -euo pipefail
cd /vepfs/code/SPVD

source /root/miniconda3/etc/profile.d/conda.sh
conda activate openclip
python -m pip install -q sentencepiece

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/vepfs/code/SPVD/src:/vepfs/code/SPVD

NAME=spvd_full_logits_bce_w001
CONFIG=configs/benchmark_eval_spvd_full_logits_bce_w001.yaml
DATASET_ROOT=/vepfs/dataset/benchmark
MODEL_ROOT=/vepfs/model
BASE_OUT=/vepfs/code/SPVD/outputs/benchmark_retrieval/${NAME}
RUN_STAMP=$(date +%Y%m%d_%H%M%S)
COCO_OUT=${BASE_OUT}/spvd_full_val2017_run_${RUN_STAMP}
FLICKR_OUT=${BASE_OUT}/spvd_full_flickr30k_test_run_${RUN_STAMP}

mkdir -p "$BASE_OUT"

echo "===== w001 full-SPVD retrieval launch ====="
echo "started_at=$(date -Iseconds)"
echo "host=$(hostname)"
echo "name=$NAME"
echo "config=$CONFIG"
echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
echo "nproc_per_node=8"
echo "workers_per_gpu=12"
echo "coco_out=$COCO_OUT"
echo "flickr_out=$FLICKR_OUT"
echo "retrieval_mode=spvd_full"
echo "==========================================="

python - <<'PY'
from pathlib import Path
ckpt = Path('/vepfs/code/SPVD/outputs/experiments/spvd_full_logits_bce_w001/spvd_full_logits_bce_w001/checkpoints/epoch_final.pt')
if not ckpt.exists():
    raise SystemExit(f'missing checkpoint: {ckpt}')
print(f'checkpoint_ok={ckpt} size={ckpt.stat().st_size}')
PY

torchrun --standalone --nnodes=1 --nproc_per_node=8 --master_port=29731 \
  -m eval.retrieval_benchmarks \
  --config "$CONFIG" \
  --datasets coco \
  --split val2017 \
  --dataset-root "$DATASET_ROOT" \
  --model-root "$MODEL_ROOT" \
  --batch-size 128 \
  --device cuda \
  --dtype bf16 \
  --retrieval-mode spvd_full \
  --output-dir "$COCO_OUT"

echo "[${NAME}] COCO val2017 finished at $(date -Iseconds)"

torchrun --standalone --nnodes=1 --nproc_per_node=8 --master_port=29732 \
  -m eval.retrieval_benchmarks \
  --config "$CONFIG" \
  --datasets flickr30k \
  --split test \
  --dataset-root "$DATASET_ROOT" \
  --model-root "$MODEL_ROOT" \
  --batch-size 128 \
  --device cuda \
  --dtype bf16 \
  --retrieval-mode spvd_full \
  --output-dir "$FLICKR_OUT"

echo "[${NAME}] Flickr30k test finished at $(date -Iseconds)"
echo "===== w001 full-SPVD retrieval finished ====="
echo "finished_at=$(date -Iseconds)"
echo "coco_metrics=${COCO_OUT}/retrieval_metrics.csv"
echo "flickr_metrics=${FLICKR_OUT}/retrieval_metrics.csv"
