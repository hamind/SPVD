#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/vepfs/code/SPVD}
LOG_DIR="${ROOT}/outputs/launch_logs/longcaption_repro_e20_benchmark"
mkdir -p "${LOG_DIR}"

stamp=$(date +%Y%m%d_%H%M%S)
outer_log="${LOG_DIR}/outer_${stamp}.log"
pid_file="${LOG_DIR}/latest.pid"

cd "${ROOT}"

nohup bash -lc "
  cd '${ROOT}'
  export CUDA_VISIBLE_DEVICES=0,1,2,3
  export NPROC_PER_NODE=4
  export OMP_NUM_THREADS=8
  export MKL_NUM_THREADS=8
  export INSTALL_BENCHMARK_DEPS=0
  export CONFIG=configs/benchmark_eval_longcaption_repro_e20.yaml
  export RUN_NAME=longcaption_repro_e20_full
  bash scripts/run_benchmark_eval.sh --models cc3m_recap_clip_e20_b896,cc3m_recap_siglip_e20_b896
" >"${outer_log}" 2>&1 &

pid=$!
echo "${pid}" > "${pid_file}"
echo "pid=${pid}"
echo "outer_log=${outer_log}"
echo "pid_file=${pid_file}"
