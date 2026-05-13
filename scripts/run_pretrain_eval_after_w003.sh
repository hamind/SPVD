#!/usr/bin/env bash
set -euo pipefail

cd /vepfs/code/SPVD

W003_COCO_CSV="/vepfs/code/SPVD/outputs/benchmark_retrieval/spvd_full_logits_bce_w003/spvd_full_val2017_run_20260512_032018/retrieval_metrics.csv"
W003_FLICKR_GLOB="/vepfs/code/SPVD/outputs/benchmark_retrieval/spvd_full_logits_bce_w003/spvd_full_flickr30k_test_run_*/retrieval_metrics.csv"
LOG_ROOT="/vepfs/code/SPVD/outputs/launch_logs/pretrain_eval_after_w003"
mkdir -p "${LOG_ROOT}"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

latest_w003_flickr_csv() {
  find /vepfs/code/SPVD/outputs/benchmark_retrieval/spvd_full_logits_bce_w003 \
    -maxdepth 2 -path '/vepfs/code/SPVD/outputs/benchmark_retrieval/spvd_full_logits_bce_w003/spvd_full_flickr30k_test_run_*/retrieval_metrics.csv' \
    -type f -print 2>/dev/null | sort | tail -n 1
}

wait_for_file() {
  local label="$1"
  local path="$2"
  log "waiting for ${label}: ${path}"
  while [ ! -s "${path}" ]; do
    sleep 60
  done
  log "${label} ready: ${path}"
}

wait_for_w003_flickr() {
  local csv_path=""
  log "waiting for w003 Flickr30k full-SPVD retrieval metrics"
  while true; do
    csv_path="$(latest_w003_flickr_csv || true)"
    if [ -n "${csv_path}" ] && [ -s "${csv_path}" ]; then
      log "w003 Flickr30k ready: ${csv_path}"
      return 0
    fi
    sleep 60
  done
}

wait_for_file "w003 COCO val2017 full-SPVD retrieval metrics" "${W003_COCO_CSV}"
wait_for_w003_flickr

source /root/miniconda3/etc/profile.d/conda.sh
conda activate openclip
pip install -q sentencepiece

export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
export PYTHONPATH=/vepfs/code/SPVD/src:/vepfs/code/SPVD

RUN_ID="$(date +%Y%m%d_%H%M%S)"
CFG="${LOG_ROOT}/benchmark_eval_pretrain_models_4gpu_${RUN_ID}.yaml"
BENCH_BASE="/vepfs/code/SPVD/outputs/benchmark_eval/pretrain_models_full_${RUN_ID}"
RET_BASE="/vepfs/code/SPVD/outputs/benchmark_retrieval/pretrain_models_full_${RUN_ID}"

cp configs/benchmark_eval_pretrain_models.yaml "${CFG}"
perl -0pi -e 's/^num_workers:\s*\d+/num_workers: 12/m; s/^num_workers_per_gpu:\s*\d+/num_workers_per_gpu: 12/m; s/retrieval_mode:\s*\S+/retrieval_mode: global/g' "${CFG}"
mkdir -p "${BENCH_BASE}" "${RET_BASE}"

log "starting pretrain benchmark eval"
log "benchmark output base: ${BENCH_BASE}"
torchrun --standalone --nnodes=1 --nproc_per_node=4 --master_port=29629 -m eval.benchmarks \
  --config "${CFG}" \
  --output_dir "${BENCH_BASE}" \
  --batch_size 64 \
  --num_workers 12
log "finished pretrain benchmark eval"

log "starting pretrain COCO val2017 retrieval"
torchrun --standalone --nnodes=1 --nproc_per_node=4 --master_port=29630 -m eval.retrieval_benchmarks \
  --config "${CFG}" \
  --datasets coco \
  --split val2017 \
  --dataset-root /vepfs/dataset/benchmark \
  --model-root /vepfs/model \
  --batch-size 128 \
  --device cuda \
  --dtype bf16 \
  --retrieval-mode global \
  --output-dir "${RET_BASE}/coco_val2017"
log "finished pretrain COCO val2017 retrieval"

log "starting pretrain Flickr30k test retrieval"
torchrun --standalone --nnodes=1 --nproc_per_node=4 --master_port=29631 -m eval.retrieval_benchmarks \
  --config "${CFG}" \
  --datasets flickr30k \
  --split test \
  --dataset-root /vepfs/code/SPVD/outputs/benchmark_retrieval/dataset_view \
  --model-root /vepfs/model \
  --batch-size 128 \
  --device cuda \
  --dtype bf16 \
  --retrieval-mode global \
  --output-dir "${RET_BASE}/flickr30k_test"
log "finished pretrain Flickr30k test retrieval"

log "all pretrain evaluations complete"
log "summary paths:"
find "${BENCH_BASE}" -maxdepth 3 -type f \( -name 'summary_all_models.csv' -o -name 'manifest.json' \) -print | sort
find "${RET_BASE}" -maxdepth 3 -type f \( -name 'retrieval_metrics.csv' -o -name 'retrieval_metrics.json' \) -print | sort
